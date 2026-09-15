from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .allocator import DeterministicAllocator, ResourceRequest
from .causality import FileLedger
from .connectivity import advance_compute, advance_transfer
from .physics import communication_energy_j, compute_energy_j, rotary_wing_power_w
from .queues import ComputeItem, FiniteQueue, TransferItem
from .runtime_validation import RuntimeInvariantReport, RuntimeTraceValidator
from .types import WorkflowInstance


SLA_TIERS = ("gold", "silver", "bronze")


@dataclass(slots=True)
class WorkflowRuntimeState:
    instance: WorkflowInstance
    owner_ugv: str
    sla_tier: str
    arrival_time_s: float
    deadline_time_s: float
    ttl_time_s: float
    file_ledger: FileLedger
    topological_indices: Mapping[str, int]
    compute_completed: set[str] = field(default_factory=set)
    completed: set[str] = field(default_factory=set)
    active_tasks: set[str] = field(default_factory=set)
    scheduled: dict[str, str] = field(default_factory=dict)
    compute_enqueued: set[str] = field(default_factory=set)
    final_tasks_returned: set[str] = field(default_factory=set)
    status: str = "active"
    missed_deadline: bool = False
    remaining_at_episode_end: bool = False
    completion_time_s: float | None = None
    drop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class StepResult:
    reward: float
    costs: Mapping[str, float]
    terminated: bool
    info: Mapping[str, object]


class UAMCOEnv:
    """Causal event-driven CMDP environment with explicit byte/cycle queues."""

    def __init__(
        self,
        *,
        micro_slot_s: float,
        episode_s: float,
        executor_cpu_hz: Mapping[str, float],
        link_rate_bps: float,
        ttl_deadline_multiplier: float = 2.0,
        max_active_workflows_per_ugv: int = 8,
        default_queue_items: int = 128,
        default_buffer_bytes: int = 8 * 1024**3,
        macro_interval_s: float = 10.0,
        compute_capacitance: Mapping[str, float] | None = None,
        radio_power_w: Mapping[str, float] | None = None,
        mobile_energy_budget_j: Mapping[str, float] | None = None,
        allocator=None,
        queue_capacity_by_kind: Mapping[str, Mapping[str, int]] | None = None,
        proactive_high_fan_in_delivery: bool = False,
        unlock_on_compute_completion: bool = False,
        termination_mode: str = "fixed_horizon",
    ) -> None:
        if micro_slot_s <= 0 or episode_s <= 0 or link_rate_bps < 0:
            raise ValueError("environment timing and link capacity are invalid")
        if ttl_deadline_multiplier <= 1:
            raise ValueError("TTL multiplier must exceed one")
        if not executor_cpu_hz or any(value <= 0 for value in executor_cpu_hz.values()):
            raise ValueError("every executor must have positive compute capacity")
        self.micro_slot_s = float(micro_slot_s)
        self.macro_interval_s = float(macro_interval_s)
        self.episode_s = float(episode_s)
        self.executor_cpu_hz = {str(key): float(value) for key, value in executor_cpu_hz.items()}
        self.link_rate_bps = float(link_rate_bps)
        self.ttl_deadline_multiplier = float(ttl_deadline_multiplier)
        self.max_active_workflows_per_ugv = int(max_active_workflows_per_ugv)
        self.default_queue_items = int(default_queue_items)
        self.default_buffer_bytes = int(default_buffer_bytes)
        self.queue_capacity_by_kind = {
            kind: {
                "max_items": int((queue_capacity_by_kind or {}).get(kind, {}).get("max_items", self.default_queue_items)),
                "max_bytes": int((queue_capacity_by_kind or {}).get(kind, {}).get("max_bytes", self.default_buffer_bytes)),
            }
            for kind in ("ugv", "rsu", "uav")
        }
        if any(
            values["max_items"] <= 0 or values["max_bytes"] < 0
            for values in self.queue_capacity_by_kind.values()
        ):
            raise ValueError("node-type queue capacities are invalid")
        self.compute_capacitance = {
            "ugv": 1.0e-28,
            "rsu": 0.8e-28,
            "uav": 1.2e-28,
            **{str(key): float(value) for key, value in (compute_capacitance or {}).items()},
        }
        self.radio_power_w = {
            "ugv": 0.6,
            "rsu": 1.0,
            "uav": 0.8,
            **{str(key): float(value) for key, value in (radio_power_w or {}).items()},
        }
        if any(value < 0 for value in self.compute_capacitance.values()):
            raise ValueError("compute capacitance cannot be negative")
        if any(value < 0 for value in self.radio_power_w.values()):
            raise ValueError("radio power cannot be negative")
        self.mobile_energy_budget_j = {
            "ugv": math.inf,
            "uav": math.inf,
            **{
                str(key).lower(): float(value)
                for key, value in (mobile_energy_budget_j or {}).items()
            },
        }
        if set(self.mobile_energy_budget_j) != {"ugv", "uav"} or any(
            value <= 0 for value in self.mobile_energy_budget_j.values()
        ):
            raise ValueError("mobile energy budgets require positive UGV and UAV entries")
        self.allocator = allocator if allocator is not None else DeterministicAllocator()
        self.proactive_high_fan_in_delivery = bool(
            proactive_high_fan_in_delivery
        )
        self.unlock_on_compute_completion = bool(unlock_on_compute_completion)
        if termination_mode not in {"fixed_horizon", "drain_admitted_workflows"}:
            raise ValueError("unknown episode termination mode")
        self.termination_mode = termination_mode
        self.fixed_horizon_slot_budget: int | None = None
        if self.termination_mode == "fixed_horizon":
            slot_ratio = self.episode_s / self.micro_slot_s
            rounded_slots = round(slot_ratio)
            if not math.isclose(
                slot_ratio,
                rounded_slots,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError(
                    "fixed episode horizon must be an integer number of micro-slots"
                )
            self.fixed_horizon_slot_budget = int(rounded_slots)
        self.reset()

    def reset(self) -> dict[str, object]:
        self.micro_slot_index = 0
        self.current_time_s = 0.0
        self.admissions_closed = False
        self.workflow_states: dict[str, WorkflowRuntimeState] = {}
        self.upload_queues: dict[str, FiniteQueue] = {}
        self.compute_queues: dict[str, FiniteQueue] = {}
        self.running_compute: dict[str, ComputeItem] = {}
        self.return_queues: dict[str, FiniteQueue] = {}
        self.waiting_forward_transfers: list[TransferItem] = []
        self.uav_macro_targets: dict[str, tuple[float, float]] = {}
        self.event_log: list[dict[str, object]] = []
        self.mobile_compute_energy_j = 0.0
        self.rsu_compute_energy_j = 0.0
        self.communication_energy_j = 0.0
        self.local_reuse_bytes = 0
        self.premature_ready_count = 0
        self._premature_ready_keys: set[tuple[str, str, str]] = set()
        self.successor_start_before_input_count = 0
        self.byte_conservation_error_count = 0
        self.single_server_concurrency_error_count = 0
        self.runtime_invariant_report = RuntimeInvariantReport(0, 0, 0)
        self._runtime_validator = RuntimeTraceValidator()
        self._validated_event_count = 0
        self.transfer_pause_count = 0
        self.transfer_resume_count = 0
        self.transfer_restart_count = 0
        self.transfer_started_count = 0
        self.interrupted_transfer_count = 0
        self.wasted_retransmission_bytes = 0.0
        self.wasted_retransmission_energy_j = 0.0
        self.dependency_transmitted_bytes = 0.0
        self.final_return_transmitted_bytes = 0.0
        self.uav_flight_energy_j = 0.0
        self.mobile_energy_j = 0.0
        self.rsu_energy_j = 0.0
        self.energy_by_node_j = {
            node_id: 0.0 for node_id in self.executor_cpu_hz
        }
        self.energy_by_component_j = {
            f"{label} {component}": 0.0
            for label in ("UGV", "RSU", "UAV")
            for component in ("compute", "radio")
        }
        self.energy_by_component_j["UAV propulsion"] = 0.0
        return {"time_s": self.current_time_s, "active_workflows": 0}

    def close_admissions(self) -> None:
        self.admissions_closed = True

    def correctness_metrics(self) -> dict[str, float]:
        report = self.runtime_invariant_report
        return {
            "premature_ready_count": float(self.premature_ready_count),
            "successor_start_before_input_count": float(
                report.successor_start_before_input_count
            ),
            "byte_conservation_error_count": float(
                report.transfer_byte_error_count
            ),
            "single_server_concurrency_error_count": float(
                report.overlapping_compute_count
            ),
        }

    def mechanism_metrics(self) -> dict[str, float]:
        return {
            "local_reuse_bytes": float(self.local_reuse_bytes),
            "dependency_transmitted_bytes": float(
                self.dependency_transmitted_bytes
            ),
            "final_return_transmitted_bytes": float(
                self.final_return_transmitted_bytes
            ),
            "transfer_pause_count": float(self.transfer_pause_count),
            "transfer_resume_count": float(self.transfer_resume_count),
            "transfer_restart_count": float(self.transfer_restart_count),
            "transfer_started_count": float(self.transfer_started_count),
            "interrupted_transfer_count": float(self.interrupted_transfer_count),
            "interrupted_transfer_ratio": (
                self.interrupted_transfer_count
                / max(1, self.transfer_started_count)
            ),
            "wasted_retransmission_bytes": float(
                self.wasted_retransmission_bytes
            ),
            "wasted_retransmission_energy_j": float(
                self.wasted_retransmission_energy_j
            ),
            "communication_energy_j": float(self.communication_energy_j),
        }

    @staticmethod
    def _node_kind(node_id: str) -> str:
        kind = str(node_id).split("-", 1)[0].lower()
        if kind not in {"ugv", "rsu", "uav"}:
            raise ValueError(f"unknown executor kind: {node_id}")
        return kind

    def _record_node_energy(self, node_id: str, energy_j: float, *, compute: bool) -> None:
        if energy_j < 0 or not math.isfinite(energy_j):
            raise ValueError("node energy increment must be finite and nonnegative")
        kind = self._node_kind(node_id)
        self.energy_by_node_j[node_id] = self.energy_by_node_j.get(node_id, 0.0) + energy_j
        label = kind.upper()
        self.energy_by_component_j[f"{label} {'compute' if compute else 'radio'}"] += energy_j
        if kind == "rsu":
            self.rsu_energy_j += energy_j
            if compute:
                self.rsu_compute_energy_j += energy_j
        else:
            self.mobile_energy_j += energy_j
            if compute:
                self.mobile_compute_energy_j += energy_j

    def record_uav_flight_energy(self, speeds_mps: Mapping[str, float], *, dt_s: float) -> float:
        if dt_s < 0:
            raise ValueError("flight-energy interval cannot be negative")
        increment = 0.0
        for uav_id, speed in speeds_mps.items():
            if self._node_kind(uav_id) != "uav":
                raise ValueError("flight energy can only be assigned to UAVs")
            node_increment = rotary_wing_power_w(float(speed)) * float(dt_s)
            self.energy_by_node_j[uav_id] = self.energy_by_node_j.get(uav_id, 0.0) + node_increment
            increment += node_increment
        self.uav_flight_energy_j += increment
        self.mobile_energy_j += increment
        self.energy_by_component_j["UAV propulsion"] += increment
        return increment

    def remaining_energy_j(self, node_id: str) -> float:
        kind = self._node_kind(node_id)
        if kind == "rsu":
            return math.inf
        budget = self.mobile_energy_budget_j[kind]
        return max(0.0, budget - self.energy_by_node_j.get(node_id, 0.0))

    def _queue(self, collection: dict[str, FiniteQueue], node_id: str) -> FiniteQueue:
        if node_id not in collection:
            capacity = self.queue_capacity_by_kind[self._node_kind(node_id)]
            collection[node_id] = FiniteQueue(
                max_items=capacity["max_items"],
                max_bytes=capacity["max_bytes"],
            )
        return collection[node_id]

    def admit_workflow(
        self,
        instance: WorkflowInstance,
        *,
        owner_ugv: str,
        sla_tier: str,
        deadline_s: float,
    ) -> str:
        if self.admissions_closed:
            raise RuntimeError("workflow admissions are closed")
        tier = sla_tier.lower()
        if tier not in SLA_TIERS:
            raise ValueError(f"unknown SLA tier: {sla_tier}")
        if deadline_s <= 0:
            raise ValueError("workflow deadline must be positive")
        active_for_owner = sum(
            state.status == "active" and state.owner_ugv == owner_ugv
            for state in self.workflow_states.values()
        )
        if active_for_owner >= self.max_active_workflows_per_ugv:
            raise OverflowError(f"workflow admission queue is full for {owner_ugv}")
        workflow_id = instance.instance_id
        if workflow_id in self.workflow_states:
            suffix = 2
            while f"{workflow_id}#{suffix}" in self.workflow_states:
                suffix += 1
            workflow_id = f"{workflow_id}#{suffix}"
        deadline_time = self.current_time_s + float(deadline_s)
        file_ledger = FileLedger(instance.file_sizes)
        for task_id in instance.tasks:
            for file_id, _ in instance.external_input_files(task_id):
                file_ledger.place(file_id, owner_ugv)
        self.workflow_states[workflow_id] = WorkflowRuntimeState(
            instance=instance,
            owner_ugv=owner_ugv,
            sla_tier=tier,
            arrival_time_s=self.current_time_s,
            deadline_time_s=deadline_time,
            ttl_time_s=self.current_time_s + float(deadline_s) * self.ttl_deadline_multiplier,
            file_ledger=file_ledger,
            topological_indices=instance.topological_indices(),
        )
        for task_id in instance.tasks:
            for file_id, _ in instance.external_input_files(task_id):
                self.event_log.append(
                    {
                        "time_s": self.current_time_s,
                        "event": "file_placement",
                        "workflow_id": workflow_id,
                        "file_id": file_id,
                        "executor": owner_ugv,
                        "source": "external_input",
                    }
                )
        self.event_log.append(
            {"time_s": self.current_time_s, "event": "admit", "workflow_id": workflow_id}
        )
        return workflow_id

    def precedence_ready_task_ids(self, workflow_id: str) -> tuple[str, ...]:
        state = self.workflow_states[workflow_id]
        if state.status != "active":
            return ()
        return state.instance.ready_task_ids(state.compute_completed, state.active_tasks)

    def ready_task_ids(self, workflow_id: str) -> tuple[str, ...]:
        """Compatibility alias for precedence readiness, never data availability."""
        return self.precedence_ready_task_ids(workflow_id)

    def _is_high_fan_in_sink(
        self,
        state: WorkflowRuntimeState,
        task_id: str,
    ) -> bool:
        task = state.instance.tasks[task_id]
        return (
            not task.children
            and len(task.parents)
            > int(self.queue_capacity_by_kind["ugv"]["max_items"])
        )

    def causal_uav_delivery_guard_active(self) -> bool:
        """Keep macro motion from invalidating a promised UAV result return."""
        if not self.proactive_high_fan_in_delivery:
            return False

        def guarded_compute(item: ComputeItem) -> bool:
            state = self.workflow_states.get(item.workflow_id)
            return bool(
                state is not None
                and item.executor.startswith("uav-")
                and any(
                    self._is_high_fan_in_sink(state, child_id)
                    for child_id in state.instance.tasks[item.task_id].children
                )
            )

        if any(guarded_compute(item) for item in self.running_compute.values()):
            return True
        if any(
            guarded_compute(item)
            for queue in self.compute_queues.values()
            for item in queue.items()
            if isinstance(item, ComputeItem)
        ):
            return True
        return any(
            item.direction == "dependency_prefetch"
            and item.source.startswith("uav-")
            for item in (
                *self.waiting_forward_transfers,
                *(
                    transfer
                    for queue in self._all_transfer_queues()
                    for transfer in queue.items()
                    if isinstance(transfer, TransferItem)
                ),
            )
        )

    def is_data_ready(self, workflow_id: str, task_id: str, executor: str) -> bool:
        state = self.workflow_states[workflow_id]
        required = state.instance.required_input_files(task_id)
        missing = state.file_ledger.missing_at(required, executor)
        if not missing:
            return True
        if not self.unlock_on_compute_completion:
            return False

        task = state.instance.tasks[task_id]
        completed_producers: dict[str, str] = {}
        for file_id, _ in missing:
            producer = next(
                (
                    parent_id
                    for parent_id in task.parents
                    if file_id in state.instance.tasks[parent_id].output_files
                ),
                None,
            )
            if producer is None or producer not in state.compute_completed:
                return False
            completed_producers[file_id] = producer

        key = (workflow_id, task_id, executor)
        if key not in self._premature_ready_keys:
            self._premature_ready_keys.add(key)
            self.premature_ready_count += 1
            self.event_log.append(
                {
                    "time_s": self.current_time_s,
                    "event": "premature_ready",
                    "workflow_id": workflow_id,
                    "task_id": task_id,
                    "executor": executor,
                    "missing_predecessor_files": tuple(completed_producers),
                    "completed_producers": tuple(
                        (file_id, completed_producers[file_id])
                        for file_id in completed_producers
                    ),
                    "reason": "computation_completion_unlock",
                }
            )
        return True

    def actor_observation(
        self,
        workflow_id: str,
        *,
        position_history_xy: Sequence[tuple[float, float]],
        current_reachable_executors: Sequence[str],
        delay_weight: float,
    ) -> dict[str, object]:
        state = self.workflow_states[workflow_id]
        if not 0.0 <= delay_weight <= 1.0:
            raise ValueError("delay weight must lie in [0, 1]")
        task_features = tuple(
            {
                "task_id": task.task_id,
                "parents": task.parents,
                "children": task.children,
                "cycles": task.cycles,
                "input_bytes": task.input_bytes,
                "output_bytes": task.output_bytes,
                "ready": task.task_id in self.precedence_ready_task_ids(workflow_id),
                "completed": task.task_id in state.compute_completed,
            }
            for task in state.instance.tasks.values()
        )
        return {
            "time_s": self.current_time_s,
            "owner_ugv": state.owner_ugv,
            "sla_tier": state.sla_tier,
            "deadline_remaining_s": max(0.0, state.deadline_time_s - self.current_time_s),
            "delay_weight": float(delay_weight),
            "position_history_xy": tuple(position_history_xy),
            "current_reachable_executors": tuple(current_reachable_executors),
            "tasks": task_features,
        }

    def action_mask(
        self,
        workflow_id: str,
        task_id: str,
        *,
        current_reachable_executors: Sequence[str],
    ) -> dict[str, bool]:
        state = self.workflow_states[workflow_id]
        ready = task_id in self.precedence_ready_task_ids(workflow_id)
        reachable = set(current_reachable_executors)
        mask: dict[str, bool] = {}
        task = state.instance.tasks[task_id]
        proactive_sink = (
            self.proactive_high_fan_in_delivery
            and self._is_high_fan_in_sink(state, task_id)
        )
        for executor in self.executor_cpu_hz:
            target_queue = self._queue(self.compute_queues, executor)
            required_files = state.instance.required_input_files(task_id)
            missing_files = state.file_ledger.missing_at(
                required_files,
                executor,
            )
            maximum_item_bytes_by_source: dict[str, int] = {}
            sources_available = True
            for file_id, size_bytes in missing_files:
                locations = state.file_ledger.locations(file_id)
                if not locations:
                    sources_available = False
                    break
                source = (
                    state.owner_ugv
                    if state.owner_ugv in locations
                    else locations[0]
                )
                maximum_item_bytes_by_source[source] = max(
                    maximum_item_bytes_by_source.get(source, 0),
                    int(size_bytes),
                )
            transfer_buffers_available = sources_available
            for source, maximum_item_bytes in maximum_item_bytes_by_source.items():
                transfer_queue = self._queue(
                    self.upload_queues
                    if source == state.owner_ugv
                    else self.return_queues,
                    source,
                )
                # A high-fan-in DAG need not buffer every predecessor file at
                # once.  Initial hops use the same bounded waiting mechanism
                # as forwarded hops, so legality depends on whether each file
                # can ever fit, not whether the whole fan-in fits right now.
                if transfer_queue.max_bytes < maximum_item_bytes:
                    transfer_buffers_available = False
                    break
            allowed = (
                ready
                and (executor == state.owner_ugv or executor in reachable)
                and transfer_buffers_available
                and target_queue.remaining_items > 0
                and target_queue.remaining_bytes >= task.output_bytes
            )
            if proactive_sink:
                # The causal path streams every direct predecessor result back
                # while its remote computation finishes.  The aggregation
                # sink waits until that real delivery has completed, then runs
                # at the owner without duplicating in-flight transfers.
                allowed = bool(
                    allowed
                    and executor == state.owner_ugv
                    and not missing_files
                )
            mask[executor] = allowed
        mask["defer"] = ready
        return mask

    @staticmethod
    def _link_key(source: str, destination: str) -> tuple[str, str]:
        return tuple(sorted((source, destination)))

    def _enqueue_compute(
        self,
        workflow_id: str,
        task_id: str,
        executor: str,
        *,
        eligible_time_s: float | None = None,
    ) -> None:
        state = self.workflow_states[workflow_id]
        if task_id in state.compute_enqueued or task_id in state.compute_completed:
            return
        task = state.instance.tasks[task_id]
        item = ComputeItem(
            item_id=f"{workflow_id}/{task_id}/compute",
            workflow_id=workflow_id,
            task_id=task_id,
            executor=executor,
            total_cycles=task.cycles,
            remaining_cycles=task.cycles,
            sla_tier=state.sla_tier,
            deadline_s=state.deadline_time_s,
            result_bytes=task.output_bytes,
            remaining_critical_path_s=state.instance.remaining_critical_path_s(
                task_id
            ),
            topological_index=state.topological_indices[task_id],
            enqueue_time_s=(
                self.current_time_s
                if eligible_time_s is None
                else float(eligible_time_s)
            ),
            data_ready_time_s=(
                self.current_time_s
                if eligible_time_s is None
                else float(eligible_time_s)
            ),
        )
        if not self._queue(self.compute_queues, executor).enqueue(item):
            raise OverflowError(f"compute queue is full at {executor}")
        state.compute_enqueued.add(task_id)

    @staticmethod
    def _transfer_route(source: str, owner_ugv: str, destination: str) -> tuple[str, ...]:
        if source == destination:
            return (source,)
        if source == owner_ugv or destination == owner_ugv:
            return (source, destination)
        return (source, owner_ugv, destination)

    def _enqueue_file_transfer(
        self,
        *,
        workflow_id: str,
        task_id: str,
        file_id: str,
        size_bytes: int,
        route: tuple[str, ...],
        hop_index: int = 0,
        direction: str = "input",
        producer_task_id: str | None = None,
        attempt: int = 1,
        wait_if_full: bool = False,
    ) -> bool:
        if len(route) < 2:
            raise ValueError("a file transfer route must contain at least two nodes")
        state = self.workflow_states[workflow_id]
        source = route[hop_index]
        destination = route[hop_index + 1]
        item = TransferItem(
            item_id=(
                f"{workflow_id}/{task_id}/{file_id}/{direction}/"
                f"hop-{hop_index}/attempt-{attempt}"
            ),
            workflow_id=workflow_id,
            task_id=task_id,
            source=source,
            destination=destination,
            total_bytes=size_bytes,
            remaining_bytes=size_bytes,
            sla_tier=state.sla_tier,
            deadline_s=state.deadline_time_s,
            direction=direction,
            file_id=file_id,
            producer_task_id=producer_task_id,
            consumer_task_id=task_id if direction == "input" else None,
            final_destination=route[-1],
            route=route,
            hop_index=hop_index,
            attempt=attempt,
        )
        queues = self.upload_queues if source == state.owner_ugv else self.return_queues
        if not self._queue(queues, source).enqueue(item):
            if wait_if_full:
                self.waiting_forward_transfers.append(item)
                self.event_log.append(
                    {
                        "time_s": self.current_time_s + self.micro_slot_s,
                        "event": "transfer_wait_capacity",
                        "workflow_id": workflow_id,
                        "task_id": task_id,
                        "file_id": file_id,
                        "transfer_id": item.item_id,
                        "attempt": attempt,
                        "hop_index": hop_index,
                        "direction": direction,
                        "source": source,
                        "destination": destination,
                    }
                )
                return False
            raise OverflowError(f"transfer buffer is full at {source}")
        return True

    def reassign_task(
        self,
        workflow_id: str,
        task_id: str,
        new_executor: str,
        *,
        reason: str = "target_changed",
    ) -> None:
        if reason not in {"target_changed", "cache_invalidated"}:
            raise ValueError("unknown transfer restart reason")
        if new_executor not in self.executor_cpu_hz:
            raise ValueError(f"unknown executor: {new_executor}")
        state = self.workflow_states[workflow_id]
        if task_id not in state.scheduled or task_id in state.compute_completed:
            raise ValueError("only a scheduled unfinished task can be reassigned")
        if any(
            item.workflow_id == workflow_id and item.task_id == task_id
            for item in self.running_compute.values()
        ):
            raise RuntimeError("a running non-preemptive task cannot be reassigned")

        for queue in self.compute_queues.values():
            for item in tuple(queue.items()):
                if (
                    isinstance(item, ComputeItem)
                    and item.workflow_id == workflow_id
                    and item.task_id == task_id
                ):
                    queue.remove(item.item_id)
                    state.compute_enqueued.discard(task_id)

        attempts_by_file: dict[str, int] = {}
        for queue in self._all_transfer_queues():
            for item in tuple(queue.items()):
                if (
                    not isinstance(item, TransferItem)
                    or item.workflow_id != workflow_id
                    or item.task_id != task_id
                    or item.direction != "input"
                ):
                    continue
                if item.file_id is not None:
                    attempts_by_file[item.file_id] = max(
                        attempts_by_file.get(item.file_id, 0),
                        item.attempt,
                    )
                wasted_bytes = item.total_bytes - item.remaining_bytes
                wasted_energy_j = item.source_energy_j + item.destination_energy_j
                queue.remove(item.item_id)
                self.transfer_restart_count += 1
                self.wasted_retransmission_bytes += wasted_bytes
                self.wasted_retransmission_energy_j += wasted_energy_j
                self.event_log.append(
                    {
                        "time_s": self.current_time_s,
                        "event": "transfer_restart",
                        "workflow_id": workflow_id,
                        "task_id": task_id,
                        "file_id": item.file_id,
                        "transfer_id": item.item_id,
                        "attempt": item.attempt,
                        "hop_index": item.hop_index,
                        "source": item.source,
                        "destination": item.destination,
                        "reason": reason,
                        "new_executor": new_executor,
                        "wasted_bytes": wasted_bytes,
                        "wasted_retransmission_bytes": wasted_bytes,
                        "wasted_energy_j": wasted_energy_j,
                    }
                )

        state.scheduled[task_id] = new_executor
        required_files = state.instance.required_input_files(task_id)
        self.local_reuse_bytes += state.file_ledger.local_reuse_bytes(
            required_files,
            new_executor,
        )
        missing_files = state.file_ledger.missing_at(required_files, new_executor)
        if not missing_files:
            self._enqueue_compute(workflow_id, task_id, new_executor)
        else:
            task = state.instance.tasks[task_id]
            self._enqueue_compute(
                workflow_id,
                task_id,
                new_executor,
                eligible_time_s=math.inf,
            )
            for file_id, size_bytes in missing_files:
                locations = state.file_ledger.locations(file_id)
                if not locations:
                    raise RuntimeError(
                        f"required file has no causal source: {workflow_id}/{task_id}/{file_id}"
                    )
                source = (
                    state.owner_ugv
                    if state.owner_ugv in locations
                    else locations[0]
                )
                producer_task_id = next(
                    (
                        parent_id
                        for parent_id in task.parents
                        if file_id in state.instance.tasks[parent_id].output_files
                    ),
                    None,
                )
                self._enqueue_file_transfer(
                    workflow_id=workflow_id,
                    task_id=task_id,
                    file_id=file_id,
                    size_bytes=size_bytes,
                    route=self._transfer_route(
                        source,
                        state.owner_ugv,
                        new_executor,
                    ),
                    direction="input",
                    producer_task_id=producer_task_id,
                    attempt=attempts_by_file.get(file_id, 1) + 1,
                )
            self._try_enqueue_compute(workflow_id, task_id)

    def _try_enqueue_compute(
        self,
        workflow_id: str,
        task_id: str,
        *,
        eligible_time_s: float | None = None,
    ) -> None:
        state = self.workflow_states[workflow_id]
        executor = state.scheduled[task_id]
        if not self.is_data_ready(workflow_id, task_id, executor):
            return
        if task_id in state.compute_enqueued:
            running = self.running_compute.get(executor)
            if (
                isinstance(running, ComputeItem)
                and running.workflow_id == workflow_id
                and running.task_id == task_id
            ):
                return
            item_id = f"{workflow_id}/{task_id}/compute"
            queue = self._queue(self.compute_queues, executor)
            if item_id not in queue.item_ids:
                raise RuntimeError(
                    f"compute reservation is neither queued nor running: {item_id}"
                )
            item = queue.get(item_id)
            if not isinstance(item, ComputeItem):
                raise RuntimeError(f"invalid compute reservation: {item_id}")
            item.enqueue_time_s = (
                self.current_time_s
                if eligible_time_s is None
                else float(eligible_time_s)
            )
            item.data_ready_time_s = item.enqueue_time_s
            return
        self._enqueue_compute(
            workflow_id,
            task_id,
            executor,
            eligible_time_s=eligible_time_s,
        )

    def _schedule(
        self,
        assignments: Mapping[tuple[str, str], str],
        current_connectivity: Mapping[tuple[str, str], bool],
    ) -> dict[str, str]:
        scheduled: dict[str, str] = {}
        validated: list[tuple[str, str, str, tuple[str, ...]]] = []
        for (workflow_id, task_id), executor in assignments.items():
            state = self.workflow_states[workflow_id]
            if executor == "defer":
                continue
            reachable = tuple(
                node
                for node in self.executor_cpu_hz
                if node != state.owner_ugv
                and (
                    current_connectivity.get((state.owner_ugv, node), False)
                    or current_connectivity.get((node, state.owner_ugv), False)
                )
            )
            mask = self.action_mask(
                workflow_id,
                task_id,
                current_reachable_executors=reachable,
            )
            if not mask.get(executor, False):
                raise ValueError(f"illegal executor assignment: {workflow_id}/{task_id} -> {executor}")
            validated.append((workflow_id, task_id, executor, reachable))

        for workflow_id, task_id, executor, reachable in validated:
            state = self.workflow_states[workflow_id]
            mask = self.action_mask(
                workflow_id,
                task_id,
                current_reachable_executors=reachable,
            )
            if not mask.get(executor, False):
                self.event_log.append(
                    {
                        "time_s": self.current_time_s,
                        "event": "schedule_deferred_capacity",
                        "workflow_id": workflow_id,
                        "task_id": task_id,
                        "executor": executor,
                    }
                )
                continue
            task = state.instance.tasks[task_id]
            state.active_tasks.add(task_id)
            state.scheduled[task_id] = executor
            required_files = state.instance.required_input_files(task_id)
            self.local_reuse_bytes += state.file_ledger.local_reuse_bytes(
                required_files,
                executor,
            )
            missing_files = state.file_ledger.missing_at(required_files, executor)
            if not missing_files:
                self._enqueue_compute(workflow_id, task_id, executor)
            else:
                try:
                    # Reserve finite compute-queue capacity before input transfer.
                    # Otherwise several in-flight transfers can all observe one
                    # free slot and overflow the queue when they finish together.
                    self._enqueue_compute(
                        workflow_id,
                        task_id,
                        executor,
                        eligible_time_s=math.inf,
                    )
                    for file_id, size_bytes in missing_files:
                        locations = state.file_ledger.locations(file_id)
                        if not locations:
                            raise RuntimeError(
                                f"required file has no causal source: {workflow_id}/{task_id}/{file_id}"
                            )
                        source = (
                            state.owner_ugv
                            if state.owner_ugv in locations
                            else locations[0]
                        )
                        producer_task_id = next(
                            (
                                parent_id
                                for parent_id in task.parents
                                if file_id
                                in state.instance.tasks[parent_id].output_files
                            ),
                            None,
                        )
                        self._enqueue_file_transfer(
                            workflow_id=workflow_id,
                            task_id=task_id,
                            file_id=file_id,
                            size_bytes=size_bytes,
                            route=self._transfer_route(
                                source,
                                state.owner_ugv,
                                executor,
                            ),
                            direction="input",
                            producer_task_id=producer_task_id,
                            wait_if_full=True,
                        )
                    self._try_enqueue_compute(workflow_id, task_id)
                except (OverflowError, RuntimeError):
                    item_id = f"{workflow_id}/{task_id}/compute"
                    queue = self._queue(self.compute_queues, executor)
                    if item_id in queue.item_ids:
                        queue.remove(item_id)
                    state.compute_enqueued.discard(task_id)
                    state.active_tasks.remove(task_id)
                    state.scheduled.pop(task_id, None)
                    raise
            key = f"{workflow_id}/{task_id}"
            scheduled[key] = executor
            self.event_log.append(
                {
                    "time_s": self.current_time_s,
                    "event": "schedule",
                    "workflow_id": workflow_id,
                    "task_id": task_id,
                    "executor": executor,
                }
            )
        return scheduled

    def _all_transfer_queues(self) -> tuple[FiniteQueue, ...]:
        return tuple(self.upload_queues.values()) + tuple(self.return_queues.values())

    def _drain_waiting_forward_transfers(self) -> None:
        still_waiting: list[TransferItem] = []
        for item in self.waiting_forward_transfers:
            state = self.workflow_states[item.workflow_id]
            if state.status != "active":
                continue
            queues = (
                self.upload_queues
                if item.source == state.owner_ugv
                else self.return_queues
            )
            if self._queue(queues, item.source).enqueue(item):
                self.event_log.append(
                    {
                        "time_s": self.current_time_s,
                        "event": "transfer_capacity_resume",
                        "workflow_id": item.workflow_id,
                        "task_id": item.task_id,
                        "file_id": item.file_id,
                        "transfer_id": item.item_id,
                        "attempt": item.attempt,
                        "hop_index": item.hop_index,
                        "direction": item.direction,
                        "source": item.source,
                        "destination": item.destination,
                    }
                )
            else:
                still_waiting.append(item)
        self.waiting_forward_transfers = still_waiting

    def _advance_links(
        self,
        connectivity: Mapping[tuple[str, str], bool],
        link_rates_bps: Mapping[tuple[str, str], float] | None = None,
    ) -> None:
        self._drain_waiting_forward_transfers()
        groups: dict[tuple[str, str], list[tuple[FiniteQueue, TransferItem]]] = {}
        for queue in self._all_transfer_queues():
            for raw_item in queue.items():
                item = raw_item
                if not isinstance(item, TransferItem):
                    continue
                groups.setdefault(self._link_key(item.source, item.destination), []).append((queue, item))
        for link_key, entries in groups.items():
            connected = bool(
                connectivity.get(link_key, False)
                or connectivity.get((link_key[1], link_key[0]), False)
            )
            configured_rate = self.link_rate_bps
            if link_rates_bps is not None:
                configured_rate = float(
                    link_rates_bps.get(
                        link_key,
                        link_rates_bps.get((link_key[1], link_key[0]), self.link_rate_bps),
                    )
                )
                if configured_rate < 0 or not math.isfinite(configured_rate):
                    raise ValueError("per-link capacity must be finite and non-negative")
            link_capacity = min(self.link_rate_bps, configured_rate)
            requests = [
                ResourceRequest(
                    item.item_id,
                    item.sla_tier,
                    item.deadline_s,
                    item.remaining_bytes * 8.0 / self.micro_slot_s,
                )
                for _, item in entries
                if connected
            ]
            allocations = self.allocator.allocate(requests, capacity=link_capacity) if requests else {}
            completed: list[tuple[FiniteQueue, TransferItem]] = []
            for queue, item in entries:
                before_bytes = item.remaining_bytes
                was_paused = item.is_paused
                allocated_rate = allocations.get(item.item_id, 0.0)
                advance_transfer(
                    item,
                    allocated_rate_bps=allocated_rate,
                    dt_s=self.micro_slot_s,
                    connected=connected,
                )
                if not was_paused and item.is_paused:
                    self.transfer_pause_count += 1
                    counted_as_interruption = (
                        item.has_started and not item.has_been_interrupted
                    )
                    if counted_as_interruption:
                        item.has_been_interrupted = True
                        self.interrupted_transfer_count += 1
                    self.event_log.append(
                        {
                            "time_s": self.current_time_s,
                            "event": "transfer_pause",
                            "workflow_id": item.workflow_id,
                            "task_id": item.task_id,
                            "file_id": item.file_id,
                            "source": item.source,
                            "destination": item.destination,
                            "counted_as_interruption": counted_as_interruption,
                            "interrupted_transfer_count": self.interrupted_transfer_count,
                        }
                    )
                elif was_paused and not item.is_paused:
                    self.transfer_resume_count += 1
                    self.event_log.append(
                        {
                            "time_s": self.current_time_s,
                            "event": "transfer_resume",
                            "workflow_id": item.workflow_id,
                            "task_id": item.task_id,
                            "file_id": item.file_id,
                            "source": item.source,
                            "destination": item.destination,
                            "remaining_bytes": item.remaining_bytes,
                        }
                    )
                transferred_bytes = before_bytes - item.remaining_bytes
                if transferred_bytes > 0 and not item.has_started:
                    item.has_started = True
                    self.transfer_started_count += 1
                    self.event_log.append(
                        {
                            "time_s": self.current_time_s + self.micro_slot_s,
                            "event": "transfer_start",
                            "workflow_id": item.workflow_id,
                            "task_id": item.task_id,
                            "file_id": item.file_id,
                            "transfer_id": item.item_id,
                            "attempt": item.attempt,
                            "hop_index": item.hop_index,
                            "source": item.source,
                            "destination": item.destination,
                            "started_bytes": transferred_bytes,
                            "transfer_started_count": self.transfer_started_count,
                        }
                    )
                if transferred_bytes > 0 and item.file_id is not None:
                    if (
                        item.direction in {"input", "dependency_prefetch"}
                        and item.producer_task_id is not None
                    ):
                        self.dependency_transmitted_bytes += transferred_bytes
                    elif item.direction == "final_return":
                        self.final_return_transmitted_bytes += transferred_bytes
                transfer_energy = 0.0
                source_energy = 0.0
                destination_energy = 0.0
                if transferred_bytes > 0 and allocated_rate > 0:
                    duration_s = min(
                        self.micro_slot_s,
                        transferred_bytes * 8.0 / allocated_rate,
                    )
                    source_energy = communication_energy_j(
                        power_w=self.radio_power_w[self._node_kind(item.source)],
                        duration_s=duration_s,
                    )
                    destination_energy = communication_energy_j(
                        power_w=self.radio_power_w[self._node_kind(item.destination)],
                        duration_s=duration_s,
                    )
                    transfer_energy = source_energy + destination_energy
                    item.source_energy_j += source_energy
                    item.destination_energy_j += destination_energy
                    self.communication_energy_j += transfer_energy
                    self._record_node_energy(item.source, source_energy, compute=False)
                    self._record_node_energy(item.destination, destination_energy, compute=False)
                if transferred_bytes > 0 and item.file_id is not None:
                    state = self.workflow_states[item.workflow_id]
                    self.event_log.append(
                        {
                            "time_s": self.current_time_s + self.micro_slot_s,
                            "event": "transfer_progress",
                            "workflow_id": item.workflow_id,
                            "task_id": item.task_id,
                            "file_id": item.file_id,
                            "transfer_id": item.item_id,
                            "attempt": item.attempt,
                            "hop_index": item.hop_index,
                            "transferred_bytes": transferred_bytes,
                            "source_energy_j": source_energy,
                            "destination_energy_j": destination_energy,
                            "communication_energy_j": transfer_energy,
                            "expected_bytes": int(state.instance.file_sizes[item.file_id]),
                        }
                    )
                if item.complete:
                    completed.append((queue, item))
            for queue, item in completed:
                queue.remove(item.item_id)
                if item.file_id is not None:
                    self._handle_file_delivery(item)
                elif item.direction == "upload":
                    self._enqueue_compute(item.workflow_id, item.task_id, item.destination)
                else:
                    self._complete_task(item.workflow_id, item.task_id)

    def _handle_file_delivery(self, item: TransferItem) -> None:
        state = self.workflow_states[item.workflow_id]
        state.file_ledger.place(item.file_id, item.destination)
        expected_bytes = int(state.instance.file_sizes[item.file_id])
        self.event_log.append(
            {
                "time_s": self.current_time_s + self.micro_slot_s,
                "event": "file_delivery",
                "workflow_id": item.workflow_id,
                "task_id": item.task_id,
                "file_id": item.file_id,
                "source": item.source,
                "destination": item.destination,
                "transfer_id": item.item_id,
                "attempt": item.attempt,
                "hop_index": item.hop_index,
                "is_final_hop": item.hop_index == len(item.route) - 2,
                "delivered_bytes": item.total_bytes - item.remaining_bytes,
                "expected_bytes": expected_bytes,
            }
        )
        self.event_log.append(
            {
                "time_s": self.current_time_s + self.micro_slot_s,
                "event": "file_placement",
                "workflow_id": item.workflow_id,
                "file_id": item.file_id,
                "executor": item.destination,
                "source": "transfer",
                "transfer_id": item.item_id,
                "attempt": item.attempt,
                "hop_index": item.hop_index,
            }
        )
        if item.route and item.hop_index < len(item.route) - 2:
            self._enqueue_file_transfer(
                workflow_id=item.workflow_id,
                task_id=item.task_id,
                file_id=item.file_id,
                size_bytes=item.total_bytes,
                route=item.route,
                hop_index=item.hop_index + 1,
                direction=item.direction,
                producer_task_id=item.producer_task_id,
                attempt=item.attempt,
                wait_if_full=True,
            )
            return
        if item.direction == "input":
            self._try_enqueue_compute(
                item.workflow_id,
                item.task_id,
                eligible_time_s=self.current_time_s + self.micro_slot_s,
            )
            return
        if item.direction == "dependency_prefetch":
            return
        self._mark_final_file_returned(item.workflow_id, item.task_id)

    def _advance_compute_queues(self) -> None:
        for executor, queue in tuple(self.compute_queues.items()):
            if executor not in self.running_compute:
                waiting = tuple(
                    item
                    for item in queue.items()
                    if isinstance(item, ComputeItem)
                    and item.enqueue_time_s <= self.current_time_s + 1.0e-12
                    and self.is_data_ready(item.workflow_id, item.task_id, executor)
                )
                if waiting:
                    item = min(
                        waiting,
                        key=lambda candidate: (
                            SLA_TIERS.index(candidate.sla_tier),
                            candidate.deadline_s,
                            -candidate.remaining_critical_path_s,
                            candidate.topological_index,
                            candidate.enqueue_time_s,
                        ),
                    )
                    queue.remove(item.item_id)
                    self.running_compute[executor] = item
                    state = self.workflow_states[item.workflow_id]
                    item.compute_start_time_s = self.current_time_s
                    missing_files = state.file_ledger.missing_at(
                        state.instance.required_input_files(item.task_id),
                        executor,
                    )
                    if missing_files:
                        self.successor_start_before_input_count += 1
                        self.event_log.append(
                            {
                                "time_s": self.current_time_s,
                                "event": "successor_start_before_input",
                                "workflow_id": item.workflow_id,
                                "task_id": item.task_id,
                                "executor": executor,
                                "missing_files": tuple(
                                    file_id for file_id, _ in missing_files
                                ),
                            }
                        )
                    self.event_log.append(
                        {
                            "time_s": self.current_time_s,
                            "event": "compute_start",
                            "workflow_id": item.workflow_id,
                            "task_id": item.task_id,
                            "executor": executor,
                            "required_inputs": tuple(
                                {
                                    "file_id": file_id,
                                    "expected_bytes": size_bytes,
                                    "present_at_executor_before_start": state.file_ledger.has(
                                        file_id, executor
                                    ),
                                }
                                for file_id, size_bytes in state.instance.required_input_files(
                                    item.task_id
                                )
                            ),
                            "data_ready_time_s": item.data_ready_time_s,
                            "compute_start_time_s": item.compute_start_time_s,
                            "compute_complete_time_s": item.compute_complete_time_s,
                        }
                    )
            item = self.running_compute.get(executor)
            if item is None:
                continue
            before_cycles = item.remaining_cycles
            advance_compute(
                item,
                allocated_cycles_per_s=self.executor_cpu_hz[executor],
                dt_s=self.micro_slot_s,
            )
            executed_cycles = before_cycles - item.remaining_cycles
            if executed_cycles > 0:
                kind = self._node_kind(executor)
                energy = compute_energy_j(
                    effective_capacitance=self.compute_capacitance[kind],
                    cycles=executed_cycles,
                    frequency_hz=self.executor_cpu_hz[executor],
                )
                self._record_node_energy(executor, energy, compute=True)
            if item.complete:
                item.compute_complete_time_s = self.current_time_s + self.micro_slot_s
                self.running_compute.pop(executor)
                self.workflow_states[item.workflow_id].compute_enqueued.discard(
                    item.task_id
                )
                self._record_compute_completion(
                    item.workflow_id, item.task_id, item=item
                )

    def _record_compute_completion(
        self,
        workflow_id: str,
        task_id: str,
        *,
        item: ComputeItem | None = None,
    ) -> None:
        state = self.workflow_states[workflow_id]
        if task_id in state.compute_completed:
            return
        state.compute_completed.add(task_id)
        state.completed.add(task_id)
        state.active_tasks.discard(task_id)
        executor = state.scheduled[task_id]
        task = state.instance.tasks[task_id]
        for file_id in task.output_files:
            state.file_ledger.place(file_id, executor)
            self.event_log.append(
                {
                    "time_s": self.current_time_s + self.micro_slot_s,
                    "event": "file_placement",
                    "workflow_id": workflow_id,
                    "file_id": file_id,
                    "executor": executor,
                    "source": "compute_output",
                    "task_id": task_id,
                }
            )
        self.event_log.append(
            {
                "time_s": self.current_time_s + self.micro_slot_s,
                "event": "compute_complete",
                "workflow_id": workflow_id,
                "task_id": task_id,
                "executor": executor,
                "data_ready_time_s": (
                    item.data_ready_time_s if item is not None else None
                ),
                "compute_start_time_s": (
                    item.compute_start_time_s if item is not None else None
                ),
                "compute_complete_time_s": (
                    item.compute_complete_time_s if item is not None else None
                ),
            }
        )
        if (
            self.proactive_high_fan_in_delivery
            and executor != state.owner_ugv
        ):
            queued_files: set[str] = set()
            for child_id in task.children:
                if not self._is_high_fan_in_sink(state, child_id):
                    continue
                for file_id, size_bytes in state.instance.dependency_files(
                    task_id,
                    child_id,
                ):
                    if (
                        file_id in queued_files
                        or state.file_ledger.has(file_id, state.owner_ugv)
                    ):
                        continue
                    queued_files.add(file_id)
                    self._enqueue_file_transfer(
                        workflow_id=workflow_id,
                        task_id=child_id,
                        file_id=file_id,
                        size_bytes=size_bytes,
                        route=self._transfer_route(
                            executor,
                            state.owner_ugv,
                            state.owner_ugv,
                        ),
                        direction="dependency_prefetch",
                        producer_task_id=task_id,
                        wait_if_full=True,
                    )
        if task.children:
            return
        missing_final_files = tuple(
            (file_id, int(state.instance.file_sizes[file_id]))
            for file_id in task.output_files
            if not state.file_ledger.has(file_id, state.owner_ugv)
        )
        if not missing_final_files:
            state.final_tasks_returned.add(task_id)
            self._maybe_complete_workflow(workflow_id)
            return
        for file_id, size_bytes in missing_final_files:
            self._enqueue_file_transfer(
                workflow_id=workflow_id,
                task_id=task_id,
                file_id=file_id,
                size_bytes=size_bytes,
                route=self._transfer_route(
                    executor,
                    state.owner_ugv,
                    state.owner_ugv,
                ),
                direction="final_return",
                producer_task_id=task_id,
                wait_if_full=True,
            )

    def _mark_final_file_returned(self, workflow_id: str, task_id: str) -> None:
        state = self.workflow_states[workflow_id]
        task = state.instance.tasks[task_id]
        if all(
            state.file_ledger.has(file_id, state.owner_ugv)
            for file_id in task.output_files
        ):
            state.final_tasks_returned.add(task_id)
        self._maybe_complete_workflow(workflow_id)

    def _maybe_complete_workflow(self, workflow_id: str) -> None:
        state = self.workflow_states[workflow_id]
        if state.status != "active":
            return
        if len(state.compute_completed) != len(state.instance.tasks):
            return
        if not set(state.instance.sink_task_ids()).issubset(
            state.final_tasks_returned
        ):
            return
        state.status = "completed"
        state.completion_time_s = self.current_time_s + self.micro_slot_s
        self.event_log.append(
            {
                "time_s": state.completion_time_s,
                "event": "workflow_complete",
                "workflow_id": workflow_id,
            }
        )

    def _complete_task(self, workflow_id: str, task_id: str) -> None:
        state = self.workflow_states[workflow_id]
        if state.status != "active":
            return
        self._record_compute_completion(workflow_id, task_id)
        self.event_log.append(
            {
                "time_s": self.current_time_s + self.micro_slot_s,
                "event": "task_complete",
                "workflow_id": workflow_id,
                "task_id": task_id,
            }
        )
        self._maybe_complete_workflow(workflow_id)

    def _purge_workflow(self, workflow_id: str) -> None:
        self.waiting_forward_transfers = [
            item
            for item in self.waiting_forward_transfers
            if item.workflow_id != workflow_id
        ]
        for queue in self._all_transfer_queues() + tuple(self.compute_queues.values()):
            for item in tuple(queue.items()):
                if getattr(item, "workflow_id", None) == workflow_id:
                    queue.remove(item.item_id)
        for executor, item in tuple(self.running_compute.items()):
            if item.workflow_id == workflow_id:
                self.running_compute.pop(executor)
                self.event_log.append(
                    {
                        "time_s": self.current_time_s,
                        "event": "compute_cancel",
                        "workflow_id": item.workflow_id,
                        "task_id": item.task_id,
                        "executor": executor,
                        "reason": "workflow_purged",
                    }
                )
        self.workflow_states[workflow_id].compute_enqueued.clear()

    @staticmethod
    def _zero_costs() -> dict[str, float]:
        return {
            f"{tier}_{kind}": 0.0
            for tier in SLA_TIERS
            for kind in ("miss", "drop")
        }

    def _apply_sla_transitions(self) -> dict[str, float]:
        costs = self._zero_costs()
        for workflow_id, state in self.workflow_states.items():
            deadline_crossed = (
                state.status == "completed"
                and state.completion_time_s is not None
                and state.completion_time_s > state.deadline_time_s
            ) or (
                state.status == "active" and self.current_time_s > state.deadline_time_s
            )
            if deadline_crossed and not state.missed_deadline:
                state.missed_deadline = True
                costs[f"{state.sla_tier}_miss"] += 1.0
                self.event_log.append(
                    {"time_s": self.current_time_s, "event": "deadline_miss", "workflow_id": workflow_id}
                )
            if state.status != "active":
                continue
            if self.current_time_s > state.ttl_time_s:
                state.status = "dropped"
                state.drop_reason = "ttl"
                costs[f"{state.sla_tier}_drop"] += 1.0
                self._purge_workflow(workflow_id)
                self.event_log.append(
                    {"time_s": self.current_time_s, "event": "drop", "workflow_id": workflow_id, "reason": "ttl"}
                )
        return costs

    def step_micro(
        self,
        assignments: Mapping[tuple[str, str], str],
        *,
        current_connectivity: Mapping[tuple[str, str], bool],
        link_rates_bps: Mapping[tuple[str, str], float] | None = None,
    ) -> StepResult:
        already_drained = (
            self.termination_mode == "drain_admitted_workflows"
            and self.admissions_closed
            and not any(
                state.status == "active" for state in self.workflow_states.values()
            )
        )
        if (
            self.termination_mode == "fixed_horizon"
            and self.micro_slot_index >= int(self.fixed_horizon_slot_budget)
        ) or already_drained:
            raise RuntimeError("episode has already terminated")
        energy_before = {
            "mobile_compute_energy_j": self.mobile_compute_energy_j,
            "rsu_compute_energy_j": self.rsu_compute_energy_j,
            "communication_energy_j": self.communication_energy_j,
        }
        completed_before = sum(state.status == "completed" for state in self.workflow_states.values())
        scheduled = self._schedule(assignments, current_connectivity)
        self._advance_links(current_connectivity, link_rates_bps)
        self._advance_compute_queues()
        self.micro_slot_index += 1
        self.current_time_s = self.micro_slot_index * self.micro_slot_s
        new_events = self.event_log[self._validated_event_count :]
        self.runtime_invariant_report = self._runtime_validator.consume(new_events)
        self._validated_event_count = len(self.event_log)
        self.successor_start_before_input_count = (
            self.runtime_invariant_report.successor_start_before_input_count
        )
        self.byte_conservation_error_count = (
            self.runtime_invariant_report.transfer_byte_error_count
        )
        self.single_server_concurrency_error_count = (
            self.runtime_invariant_report.overlapping_compute_count
        )
        costs = self._apply_sla_transitions()
        if self.termination_mode == "fixed_horizon":
            terminated = self.micro_slot_index >= int(self.fixed_horizon_slot_budget)
        else:
            terminated = self.admissions_closed and not any(
                state.status == "active" for state in self.workflow_states.values()
            )
        if terminated and self.termination_mode == "fixed_horizon":
            for state in self.workflow_states.values():
                if state.status == "active":
                    state.remaining_at_episode_end = True
        completed_after = sum(state.status == "completed" for state in self.workflow_states.values())
        info = {
            "time_s": self.current_time_s,
            "scheduled": scheduled,
            "completed_workflows": completed_after,
            "dropped_workflows": sum(state.status == "dropped" for state in self.workflow_states.values()),
            "remaining_workflows": sum(state.status == "active" for state in self.workflow_states.values()) if terminated else 0,
            "mobile_compute_energy_j": self.mobile_compute_energy_j - energy_before["mobile_compute_energy_j"],
            "rsu_compute_energy_j": self.rsu_compute_energy_j - energy_before["rsu_compute_energy_j"],
            "communication_energy_j": self.communication_energy_j - energy_before["communication_energy_j"],
        }
        return StepResult(
            reward=float(completed_after - completed_before),
            costs=costs,
            terminated=terminated,
            info=info,
        )

    def step_macro(self, uav_targets_xy_m: Mapping[str, tuple[float, float]]) -> None:
        self.uav_macro_targets = {
            str(uav_id): (float(target[0]), float(target[1]))
            for uav_id, target in uav_targets_xy_m.items()
        }
