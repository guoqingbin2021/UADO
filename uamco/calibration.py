from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .data_integrity import verify_file_record


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def calibration_energy_per_admitted_dag(
    *,
    total_energy_j: float,
    shared_uav_flight_energy_j: float,
    active_workflows: int,
) -> float:
    """Match one-DAG calibration to the formal shared-flight energy boundary."""
    total = float(total_energy_j)
    shared = float(shared_uav_flight_energy_j)
    cohort = int(active_workflows)
    if (
        not math.isfinite(total)
        or not math.isfinite(shared)
        or total < 0.0
        or shared < 0.0
        or shared > total + 1.0e-9
        or cohort <= 0
    ):
        raise ValueError("calibration energy components or cohort size are invalid")
    workload_specific = max(0.0, total - shared)
    return workload_specific + shared / cohort


def calibration_config_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return exactly the merged parameters consumed by calibration simulation."""

    data = config["data"]
    scenario = config["scenario"]
    timing = config["timing"]
    resources = config["resources"]
    sla = config["sla"]
    snapshot = {
        "data": {"reference_frequency_hz": data["reference_frequency_hz"]},
        "scenario": {
            key: scenario[key]
            for key in (
                "width_m",
                "height_m",
                "ugv_count",
                "rsu_count",
                "uav_count",
                "rsu_radius_m",
                "uav_altitude_m",
                "uav_radius_m",
                "uav_max_speed_mps",
                "queue_capacity",
            )
        },
        "timing": {
            key: timing[key]
            for key in ("micro_slot_s", "macro_interval_s", "episode_s")
        },
        "resources": {
            key: resources[key]
            for key in (
                "executor_cpu_hz",
                "link_rate_bps",
                "channel",
                "radio_power_w",
                "compute_capacitance",
            )
        },
        "sla": {"ttl_deadline_multiplier": sla["ttl_deadline_multiplier"]},
        "experiments": {
            "main_completion": {
                "active_workflows_per_episode": config["experiments"][
                    "main_completion"
                ]["active_workflows_per_episode"]
            }
        },
    }
    return json.loads(json.dumps(snapshot, sort_keys=True))


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _validate_selection_payload(
    selection: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[tuple[str, ...], str, tuple[int, ...], tuple[str, ...]]:
    workflows = selection.get("workflows")
    if not isinstance(workflows, list) or len(workflows) != 3:
        raise ValueError("calibration requires exactly three workflow records")
    families: set[str] = set()
    instance_ids: list[str] = []
    for record_value in workflows:
        record = _require_mapping(record_value, "calibration workflow record")
        family = str(record.get("family", ""))
        instance_id = str(record.get("instance_id", ""))
        if family not in {"Montage", "Seismology", "Cycles"}:
            raise ValueError(f"invalid calibration workflow family: {family}")
        if not instance_id.startswith(f"{family}:"):
            raise ValueError(f"calibration workflow ID/family mismatch: {instance_id}")
        verify_file_record(project_root, record)
        families.add(family)
        instance_ids.append(instance_id)
    if families != {"Montage", "Seismology", "Cycles"} or len(set(instance_ids)) != 3:
        raise ValueError("calibration must reserve one unique workflow from each family")

    mobility = _require_mapping(selection.get("mobility_trace"), "calibration mobility record")
    trace_id = str(mobility.get("trace_id", ""))
    if trace_id != "00000":
        raise ValueError("calibration must reserve RELLIS-3D trace 00000")
    verify_file_record(project_root, mobility)
    seeds_raw = selection.get("seeds")
    if (
        not isinstance(seeds_raw, list)
        or not seeds_raw
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds_raw)
        or len(set(seeds_raw)) != len(seeds_raw)
    ):
        raise ValueError("calibration seeds must be a non-empty unique integer list")
    methods_raw = selection.get("method_ids")
    expected_methods = ("fixed_local", "fixed_reachable_edge")
    if not isinstance(methods_raw, list) or tuple(methods_raw) != expected_methods:
        raise ValueError(f"calibration method_ids must equal {expected_methods}")
    return (
        tuple(instance_ids),
        trace_id,
        tuple(int(seed) for seed in seeds_raw),
        expected_methods,
    )


def _finite_float(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "non-negative finite" if nonnegative else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _same_float(left: object, right: float) -> bool:
    try:
        value = float(left)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(value, right, rel_tol=1.0e-12, abs_tol=1.0e-12)


@dataclass(frozen=True, slots=True)
class CalibrationContract:
    workflow_instance_ids: tuple[str, ...]
    mobility_trace_id: str
    seeds: tuple[int, ...]
    method_ids: tuple[str, ...]
    selection_payload: Mapping[str, Any]
    provenance_sha256: str


def _resolved_project_file(path: str | Path, project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    candidate = Path(path)
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"calibration artifact must reside within project_root: {resolved}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _validate_objective_bounds_payload(
    payload: Mapping[str, Any],
    *,
    project_root: Path,
    active_config: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], tuple[str, ...], str, tuple[int, ...], tuple[str, ...]]:
    if payload.get("schema_version") != 1:
        raise ValueError("objective bounds artifact must use schema version 1")
    artifact_provenance = str(payload.get("provenance_sha256", ""))
    unsigned = {key: value for key, value in payload.items() if key != "provenance_sha256"}
    if not _SHA256.fullmatch(artifact_provenance) or canonical_sha256(unsigned) != artifact_provenance:
        raise ValueError("objective bounds artifact provenance mismatch")

    selection = _require_mapping(payload.get("selection_payload"), "objective selection_payload")
    selection_provenance = str(payload.get("selection_provenance_sha256", ""))
    if not _SHA256.fullmatch(selection_provenance) or canonical_sha256(selection) != selection_provenance:
        raise ValueError("objective selection provenance mismatch")
    instance_ids, trace_id, seeds, methods = _validate_selection_payload(
        selection,
        project_root=project_root,
    )
    if tuple(payload.get("method_ids", ())) != methods:
        raise ValueError("objective bounds calibration methods mismatch")
    if tuple(payload.get("seeds", ())) != seeds:
        raise ValueError("objective bounds calibration seeds mismatch")

    config_record = _require_mapping(
        payload.get("calibration_config"),
        "objective calibration_config",
    )
    config_path = verify_file_record(project_root, config_record)
    raw_config_path = str(config_record.get("path", ""))
    if "\\" in raw_config_path or Path(raw_config_path).is_absolute():
        raise ValueError("calibration config path must be project-relative POSIX")
    from .config import load_config

    recorded_config = load_config(config_path)
    recorded_snapshot = calibration_config_snapshot(recorded_config)
    stored_snapshot = config_record.get("invariant_parameters")
    if stored_snapshot != recorded_snapshot:
        raise ValueError("calibration config parameter snapshot is stale")
    stored_snapshot_hash = str(config_record.get("invariant_sha256", ""))
    if not _SHA256.fullmatch(stored_snapshot_hash) or canonical_sha256(recorded_snapshot) != stored_snapshot_hash:
        raise ValueError("calibration config invariant hash mismatch")
    if active_config is not None and calibration_config_snapshot(active_config) != recorded_snapshot:
        raise ValueError("active configuration changes calibration invariants")

    selected_records = list(selection["workflows"])
    selected_records.append(selection["mobility_trace"])
    expected_input_hashes = {
        str(record["path"]): str(record["sha256"])
        for record in selected_records
    }
    expected_input_hashes[raw_config_path] = str(config_record["sha256"])
    input_hashes = payload.get("input_hashes")
    if not isinstance(input_hashes, Mapping) or dict(input_hashes) != dict(
        sorted(expected_input_hashes.items())
    ):
        raise ValueError("objective input_hashes do not match the selected current inputs")

    observations = payload.get("observations")
    expected_count = len(instance_ids) * len(methods) * len(seeds)
    if not isinstance(observations, list) or len(observations) != expected_count:
        raise ValueError(f"objective bounds require exactly {expected_count} observations")
    family_by_instance = {
        str(record["instance_id"]): str(record["family"])
        for record in selection["workflows"]
    }
    expected_keys = {
        (instance_id, method_id, seed)
        for instance_id in instance_ids
        for method_id in methods
        for seed in seeds
    }
    observed_keys: set[tuple[str, str, int]] = set()
    delays: list[float] = []
    energies: list[float] = []
    statuses: list[str] = []
    observation_fields = {
        "method_id",
        "seed",
        "workflow_instance_id",
        "family",
        "mobility_trace_id",
        "mobility_original_duration_s",
        "mobility_extension",
        "effective_delay_s",
        "system_energy_per_admitted_dag_j",
        "status",
    }
    horizon_s = _finite_float(recorded_snapshot["timing"]["episode_s"], "calibration horizon")
    for index, observation_value in enumerate(observations):
        observation = _require_mapping(observation_value, f"calibration observation {index}")
        if set(observation) != observation_fields:
            raise ValueError(f"calibration observation {index} has invalid fields")
        instance_id = str(observation["workflow_instance_id"])
        method_id = str(observation["method_id"])
        seed_value = observation["seed"]
        if isinstance(seed_value, bool) or not isinstance(seed_value, int):
            raise ValueError(f"calibration observation {index} seed must be an integer")
        key = (instance_id, method_id, int(seed_value))
        if key not in expected_keys or key in observed_keys:
            raise ValueError(f"foreign or duplicate calibration observation: {key}")
        observed_keys.add(key)
        if str(observation["family"]) != family_by_instance[instance_id]:
            raise ValueError(f"calibration observation family mismatch: {instance_id}")
        if str(observation["mobility_trace_id"]) != trace_id:
            raise ValueError("calibration observation mobility trace mismatch")
        original_duration_s = _finite_float(
            observation["mobility_original_duration_s"],
            "calibration mobility duration",
        )
        if original_duration_s <= 0.0:
            raise ValueError("calibration mobility duration must be positive")
        extension = str(observation["mobility_extension"])
        expected_extension = (
            "measured_pose_mirror"
            if original_duration_s < horizon_s
            else "none"
        )
        if extension != expected_extension:
            raise ValueError(
                "calibration mobility extension does not match the fixed horizon"
            )
        status = str(observation["status"])
        if status not in {"completed", "remaining", "dropped"}:
            raise ValueError(f"invalid calibration observation status: {status}")
        delay = _finite_float(
            observation["effective_delay_s"],
            "calibration effective delay",
            nonnegative=True,
        )
        if delay > horizon_s + 1.0e-9:
            raise ValueError("calibration effective delay exceeds the finite horizon")
        if status in {"remaining", "dropped"} and not math.isclose(
            delay, horizon_s, rel_tol=1.0e-12, abs_tol=1.0e-9
        ):
            raise ValueError("unfinished calibration outcome must use the horizon effective delay")
        energy = _finite_float(
            observation["system_energy_per_admitted_dag_j"],
            "calibration system energy",
            nonnegative=True,
        )
        delays.append(delay)
        energies.append(energy)
        statuses.append(status)
    if observed_keys != expected_keys:
        raise ValueError("calibration observation Cartesian product is incomplete")
    if "completed" not in statuses:
        raise ValueError("calibration requires at least one completed workflow observation")

    recomputed_extrema = {
        "time_min_s": min(delays),
        "time_max_s": max(delays),
        "energy_min_j": min(energies),
        "energy_max_j": max(energies),
    }
    if recomputed_extrema["time_max_s"] <= recomputed_extrema["time_min_s"]:
        raise ValueError("calibration delay observations are degenerate")
    if recomputed_extrema["energy_max_j"] <= recomputed_extrema["energy_min_j"]:
        raise ValueError("calibration energy observations are degenerate")
    raw_extrema = _require_mapping(payload.get("raw_extrema"), "objective raw_extrema")
    if set(raw_extrema) != set(recomputed_extrema) or any(
        not _same_float(raw_extrema[key], value)
        for key, value in recomputed_extrema.items()
    ):
        raise ValueError("objective raw extrema do not match observations")
    margin_fraction = _finite_float(payload.get("margin_fraction"), "objective margin")
    if not math.isclose(margin_fraction, 0.10, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("objective bounds require the fixed 10% safety margin")
    time_span = recomputed_extrema["time_max_s"] - recomputed_extrema["time_min_s"]
    energy_span = recomputed_extrema["energy_max_j"] - recomputed_extrema["energy_min_j"]
    expected_bounds = {
        "time_min_s": recomputed_extrema["time_min_s"] - 0.10 * time_span,
        "time_max_s": recomputed_extrema["time_max_s"] + 0.10 * time_span,
        "energy_min_j": max(0.0, recomputed_extrema["energy_min_j"] - 0.10 * energy_span),
        "energy_max_j": recomputed_extrema["energy_max_j"] + 0.10 * energy_span,
    }
    if any(not _same_float(payload.get(key), value) for key, value in expected_bounds.items()):
        raise ValueError("objective bounds do not match the unique 10% margin formula")
    return selection, instance_ids, trace_id, seeds, methods


def load_calibration_contract(
    path: str | Path,
    *,
    project_root: str | Path,
    objective_bounds_path: str | Path | None = None,
    active_config: Mapping[str, Any] | None = None,
) -> CalibrationContract:
    """Load and verify the immutable calibration selection and optional bounds."""

    root = Path(project_root).resolve()
    resolved = _resolved_project_file(path, root)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("calibration manifest must use schema version 1")
    selection = payload.get("selection_payload")
    if not isinstance(selection, Mapping):
        raise ValueError("calibration selection_payload must be a mapping")
    provenance = str(payload.get("provenance_sha256", ""))
    if not _SHA256.fullmatch(provenance) or canonical_sha256(selection) != provenance:
        raise ValueError("calibration selection provenance mismatch")

    instance_ids, trace_id, seeds, methods = _validate_selection_payload(
        selection,
        project_root=root,
    )

    contract = CalibrationContract(
        workflow_instance_ids=instance_ids,
        mobility_trace_id=trace_id,
        seeds=seeds,
        method_ids=methods,
        selection_payload=selection,
        provenance_sha256=provenance,
    )
    if objective_bounds_path is not None:
        bounds_path = _resolved_project_file(objective_bounds_path, root)
        bounds_payload = json.loads(bounds_path.read_text(encoding="utf-8"))
        if not isinstance(bounds_payload, Mapping):
            raise ValueError("objective bounds artifact must be a mapping")
        embedded_selection, embedded_ids, embedded_trace, embedded_seeds, embedded_methods = (
            _validate_objective_bounds_payload(
                bounds_payload,
                project_root=root,
                active_config=active_config,
            )
        )
        if bounds_payload.get("selection_provenance_sha256") != contract.provenance_sha256:
            raise ValueError("objective bounds use a different calibration selection")
        if embedded_selection != contract.selection_payload:
            raise ValueError("objective bounds selection payload mismatch")
        if (
            embedded_ids != contract.workflow_instance_ids
            or embedded_trace != contract.mobility_trace_id
            or embedded_seeds != contract.seeds
            or embedded_methods != contract.method_ids
        ):
            raise ValueError("objective bounds embedded selection contract mismatch")
    return contract


@dataclass(frozen=True, slots=True)
class FrozenObjectiveBounds:
    """Calibration bounds fixed before training and shared by all objectives."""

    time_min_s: float
    time_max_s: float
    energy_min_j: float
    energy_max_j: float
    provenance_sha256: str

    def __post_init__(self) -> None:
        values = (
            self.time_min_s,
            self.time_max_s,
            self.energy_min_j,
            self.energy_max_j,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("frozen objective bounds must be finite")
        if self.time_max_s <= self.time_min_s:
            raise ValueError("frozen time bounds must be strictly increasing")
        if self.energy_max_j <= self.energy_min_j:
            raise ValueError("frozen energy bounds must be strictly increasing")
        if not _SHA256.fullmatch(str(self.provenance_sha256)):
            raise ValueError("frozen objective provenance must be a lowercase SHA-256")

    def normalize_time(self, value_s: float) -> float:
        # Zero is the physical origin.  The frozen calibration maximum is a
        # scale/cap, not a target: values below the observed calibration
        # minimum must retain a non-zero slope so that further improvements
        # are still rewarded.
        normalized = float(value_s) / self.time_max_s
        return min(1.0, max(0.0, normalized))

    def normalize_energy(self, value_j: float) -> float:
        normalized = float(value_j) / self.energy_max_j
        return min(1.0, max(0.0, normalized))

    @property
    def identity_sha256(self) -> str:
        payload = {
            "time_min_s": self.time_min_s,
            "time_max_s": self.time_max_s,
            "energy_min_j": self.energy_min_j,
            "energy_max_j": self.energy_max_j,
            "provenance_sha256": self.provenance_sha256,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def load_frozen_objective_bounds(
    path: str | Path,
    *,
    project_root: str | Path,
    active_config: Mapping[str, Any] | None = None,
) -> FrozenObjectiveBounds:
    """Load a calibration artifact without allowing paths outside the project."""

    root = Path(project_root).resolve()
    candidate = Path(path)
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("frozen objective bounds must reside within project_root") from exc
    if not resolved.is_file():
        raise FileNotFoundError(
            f"frozen objective bounds are required for formal runtime: {resolved}"
        )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid frozen objective bounds JSON: {resolved}") from exc
    if not isinstance(payload, dict):
        raise ValueError("frozen objective bounds must be a JSON object")
    _validate_objective_bounds_payload(
        payload,
        project_root=root,
        active_config=active_config,
    )
    required = {
        "time_min_s",
        "time_max_s",
        "energy_min_j",
        "energy_max_j",
        "provenance_sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"frozen objective bounds missing keys: {missing}")
    return FrozenObjectiveBounds(**{key: payload[key] for key in required})
