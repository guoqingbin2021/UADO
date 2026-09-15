from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence
from functools import wraps
from typing import Any


def gap_demand_points(
    env: object,
    ugv_positions: Mapping[str, tuple[float, float]],
    connectivity: Mapping[tuple[str, str], bool],
) -> list[dict[str, Any]]:
    """Return the model's per-workflow demand weight with current contact flags."""

    points: list[dict[str, Any]] = []
    workflow_states = getattr(env, "workflow_states", {})
    for workflow_id, state in workflow_states.items():
        if getattr(state, "status", None) != "active":
            continue
        owner = str(getattr(state, "owner_ugv"))
        if owner not in ugv_positions:
            continue
        instance = getattr(state, "instance")
        completed = set(getattr(state, "completed", ()))
        ledger = getattr(state, "file_ledger")
        missing_bytes = 0
        for parent_id, child_id in instance.edges:
            if child_id in completed:
                continue
            for file_id, size_bytes in instance.dependency_files(parent_id, child_id):
                if not ledger.has(file_id, owner):
                    missing_bytes += int(size_bytes)

        rsu_contact = any(
            bool(connected)
            and source == owner
            and str(target).startswith("rsu-")
            for (source, target), connected in connectivity.items()
        )
        uav_contact = any(
            bool(connected)
            and source == owner
            and str(target).startswith("uav-")
            for (source, target), connected in connectivity.items()
        )
        x_m, y_m = ugv_positions[owner]
        points.append(
            {
                "workflow_id": str(workflow_id),
                "owner_ugv": owner,
                "x_m": float(x_m),
                "y_m": float(y_m),
                "missing_bytes": int(missing_bytes),
                "weight": float(1.0 + math.log1p(missing_bytes)),
                "rsu_uncovered": not rsu_contact,
                "uav_contact": uav_contact,
            }
        )
    return points


def summarize_uav_events(
    events: Sequence[Mapping[str, Any]],
    routes: MutableMapping[str, dict[str, str]],
    rsu_uncovered_ugvs: set[str],
    workflow_owners: Mapping[str, str],
) -> dict[str, Any]:
    """Attribute transfer and compute events to UAV service in the current slot."""

    summary: dict[str, Any] = {
        "uav_transfer_bytes": 0.0,
        "uav_gap_transfer_bytes": 0.0,
        "gap_file_deliveries": 0,
        "uav_compute_completions": 0,
        "gap_uav_compute_completions": 0,
        "served_gap_endpoints": [],
        "gap_bytes_by_ugv": {},
        "gap_compute_by_ugv": {},
    }
    for event in events:
        transfer_id = str(event.get("transfer_id", ""))
        event_name = str(event.get("event", ""))
        if event_name == "transfer_start" and transfer_id:
            routes[transfer_id] = {
                "source": str(event.get("source", "")),
                "destination": str(event.get("destination", "")),
                "workflow_id": str(event.get("workflow_id", "")),
            }
        route = routes.get(transfer_id, {})
        source = str(event.get("source", route.get("source", "")))
        destination = str(
            event.get("destination", route.get("destination", ""))
        )
        uav_link = source.startswith("uav-") or destination.startswith("uav-")
        ugv_endpoint = ""
        if source.startswith("ugv-"):
            ugv_endpoint = source
        elif destination.startswith("ugv-"):
            ugv_endpoint = destination
        gap_endpoint = ugv_endpoint in rsu_uncovered_ugvs

        if event_name == "transfer_progress" and uav_link:
            delivered = float(event.get("transferred_bytes", 0.0))
            summary["uav_transfer_bytes"] += delivered
            if gap_endpoint:
                summary["uav_gap_transfer_bytes"] += delivered
                summary["gap_bytes_by_ugv"][ugv_endpoint] = (
                    float(summary["gap_bytes_by_ugv"].get(ugv_endpoint, 0.0))
                    + delivered
                )
        elif event_name == "file_delivery" and uav_link and gap_endpoint:
            summary["gap_file_deliveries"] += 1
            if ugv_endpoint not in summary["served_gap_endpoints"]:
                summary["served_gap_endpoints"].append(ugv_endpoint)
        elif event_name == "compute_complete" and str(
            event.get("executor", "")
        ).startswith("uav-"):
            summary["uav_compute_completions"] += 1
            workflow_id = str(event.get("workflow_id", ""))
            owner = workflow_owners.get(workflow_id)
            if owner in rsu_uncovered_ugvs:
                summary["gap_uav_compute_completions"] += 1
                summary["gap_compute_by_ugv"][owner] = (
                    int(summary["gap_compute_by_ugv"].get(owner, 0)) + 1
                )
    return summary


def transfer_route_records(env: object) -> dict[str, dict[str, str]]:
    """Return transfer endpoint metadata for every currently queued hop."""

    items: list[object] = []
    for queues_name in ("upload_queues", "return_queues"):
        for queue in getattr(env, queues_name, {}).values():
            items.extend(queue.items())
    items.extend(getattr(env, "waiting_forward_transfers", ()))
    records: dict[str, dict[str, str]] = {}
    for item in items:
        item_id = str(getattr(item, "item_id", ""))
        if not item_id or item_id in records:
            continue
        records[item_id] = {
            "source": str(getattr(item, "source", "")),
            "destination": str(getattr(item, "destination", "")),
            "workflow_id": str(getattr(item, "workflow_id", "")),
            "direction": str(getattr(item, "direction", "")),
            "file_id": str(getattr(item, "file_id", "")),
        }
    return records


class UAVCoverageTraceCollector:
    """Observe a formal episode without changing policy or simulator semantics."""

    def __init__(self, *, macro_interval_s: float) -> None:
        interval = float(macro_interval_s)
        if not math.isfinite(interval) or interval <= 0.0:
            raise ValueError("macro interval must be finite and positive")
        self.macro_interval_s = interval
        self.micro_slots: list[dict[str, Any]] = []
        self.routes: dict[str, dict[str, str]] = {}
        self._physical_state: dict[str, Any] | None = None
        self._attached_environment_ids: set[int] = set()

    @staticmethod
    def _positions(
        values: Mapping[str, tuple[float, float]],
    ) -> dict[str, list[float]]:
        return {
            str(node_id): [float(position[0]), float(position[1])]
            for node_id, position in values.items()
        }

    def set_physical_state(
        self,
        ugv_positions: Mapping[str, tuple[float, float]],
        rsu_positions: Mapping[str, tuple[float, float]],
        uav_positions: Mapping[str, tuple[float, float]],
        connectivity: Mapping[tuple[str, str], bool],
    ) -> None:
        self._physical_state = {
            "ugv_positions": self._positions(ugv_positions),
            "rsu_positions": self._positions(rsu_positions),
            "uav_positions": self._positions(uav_positions),
            "connectivity": {
                (str(source), str(destination)): bool(connected)
                for (source, destination), connected in connectivity.items()
            },
        }

    def attach(self, env: object) -> None:
        """Wrap one environment's micro-step with before/after observations."""

        env_id = id(env)
        if env_id in self._attached_environment_ids:
            raise ValueError("trace collector is already attached to this environment")
        original_step_micro = getattr(env, "step_micro")

        @wraps(original_step_micro)
        def traced_step_micro(
            assignments,
            *,
            current_connectivity,
            link_rates_bps=None,
        ):
            if self._physical_state is None:
                raise RuntimeError("physical state must be recorded before step_micro")
            physical = self._physical_state
            ugv_positions = {
                node_id: (position[0], position[1])
                for node_id, position in physical["ugv_positions"].items()
            }
            connectivity = physical["connectivity"]
            points = gap_demand_points(env, ugv_positions, connectivity)
            rsu_uncovered_ugvs = {
                str(point["owner_ugv"])
                for point in points
                if bool(point["rsu_uncovered"])
            }
            workflow_owners = {
                str(workflow_id): str(getattr(state, "owner_ugv"))
                for workflow_id, state in getattr(env, "workflow_states", {}).items()
            }
            self.routes.update(transfer_route_records(env))
            event_start = len(getattr(env, "event_log", ()))
            time_start_s = float(getattr(env, "current_time_s", 0.0))
            result = original_step_micro(
                assignments,
                current_connectivity=current_connectivity,
                link_rates_bps=link_rates_bps,
            )
            new_events = list(getattr(env, "event_log", ()))[event_start:]
            event_summary = summarize_uav_events(
                new_events,
                self.routes,
                rsu_uncovered_ugvs,
                workflow_owners,
            )
            gap_weight = sum(
                float(point["weight"])
                for point in points
                if bool(point["rsu_uncovered"])
            )
            contacted_gap_weight = sum(
                float(point["weight"])
                for point in points
                if bool(point["rsu_uncovered"]) and bool(point["uav_contact"])
            )
            service_points = [
                {
                    "ugv_id": ugv_id,
                    "x_m": float(ugv_positions[ugv_id][0]),
                    "y_m": float(ugv_positions[ugv_id][1]),
                    "bytes": float(delivered_bytes),
                }
                for ugv_id, delivered_bytes in sorted(
                    event_summary["gap_bytes_by_ugv"].items()
                )
                if ugv_id in ugv_positions
            ]
            compute_points = [
                {
                    "ugv_id": ugv_id,
                    "x_m": float(ugv_positions[ugv_id][0]),
                    "y_m": float(ugv_positions[ugv_id][1]),
                    "count": int(completion_count),
                }
                for ugv_id, completion_count in sorted(
                    event_summary["gap_compute_by_ugv"].items()
                )
                if ugv_id in ugv_positions
            ]
            self.micro_slots.append(
                {
                    "micro_slot": int(getattr(env, "micro_slot_index")),
                    "macro_slot": int(time_start_s // self.macro_interval_s) + 1,
                    "time_start_s": time_start_s,
                    "time_end_s": float(getattr(env, "current_time_s")),
                    "ugv_positions": physical["ugv_positions"],
                    "rsu_positions": physical["rsu_positions"],
                    "uav_positions": physical["uav_positions"],
                    "gap_demand_points": points,
                    "gap_demand_weight": float(gap_weight),
                    "contacted_gap_demand_weight": float(contacted_gap_weight),
                    "gap_contact_ratio": (
                        float(contacted_gap_weight / gap_weight)
                        if gap_weight > 0.0
                        else 0.0
                    ),
                    "uav_transfer_bytes": float(
                        event_summary["uav_transfer_bytes"]
                    ),
                    "uav_gap_transfer_bytes": float(
                        event_summary["uav_gap_transfer_bytes"]
                    ),
                    "gap_file_deliveries": int(
                        event_summary["gap_file_deliveries"]
                    ),
                    "uav_compute_completions": int(
                        event_summary["uav_compute_completions"]
                    ),
                    "gap_uav_compute_completions": int(
                        event_summary["gap_uav_compute_completions"]
                    ),
                    "served_gap_endpoints": list(
                        event_summary["served_gap_endpoints"]
                    ),
                    "gap_service_points": service_points,
                    "gap_compute_points": compute_points,
                }
            )
            return result

        setattr(env, "step_micro", traced_step_micro)
        self._attached_environment_ids.add(env_id)

    def payload(
        self,
        *,
        provenance: Mapping[str, Any],
        rollout_metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provenance": dict(provenance),
            "rollout_metrics": {
                str(key): value for key, value in rollout_metrics.items()
            },
            "micro_slots": list(self.micro_slots),
        }
