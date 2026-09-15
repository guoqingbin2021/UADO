from __future__ import annotations

import json
import math
import os
import re
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .calibration import canonical_sha256
from .data_integrity import sha256_file


FORMAL_METHODS = (
    "UAMCO-DAG",
    "MAPPO",
    "HAPPO",
    "AMCoEdge",
    "FDEdge",
    "MEC-UARA",
)
ABLATION_VARIANTS = (
    "computation_completion_unlock",
    "non_hierarchical",
    "without_gnn",
    "without_causal_contact_history",
    "without_cost_critics",
    "without_sla_allocator",
    "same_state_ppo",
    "macro_only",
    "file_endpoint_only",
    "without_finish_time_guard",
)
CHECKPOINT_SCHEMA_VERSION = 14
SEMANTIC_CONTRACT_FILES = (
    "uamco/connectivity.py",
    "uamco/env.py",
    "uamco/formal_engine.py",
    "uamco/model.py",
    "uamco/mobility_data.py",
    "uamco/objectives.py",
    "uamco/observations.py",
    "uamco/runtime_validation.py",
    "uamco/sota/api.py",
    "uamco/sota/amcoedge.py",
    "uamco/sota/fdedge.py",
    "uamco/sota/mec_uara.py",
    "uamco/training.py",
    "uamco/workflow_data.py",
)


def build_semantic_contract(
    config: Mapping,
    *,
    project_root: str | Path,
) -> dict[str, str]:
    root = Path(project_root).resolve()
    return {
        "config_sha256": canonical_sha256(config),
        "workflow_manifest_sha256": sha256_file(
            root / str(config["data"]["workflow_manifest"])
        ),
        "mobility_manifest_sha256": sha256_file(
            root / str(config["data"]["mobility_manifest"])
        ),
        "environment_semantics_sha256": canonical_sha256(
            {
                relative: sha256_file(root / relative)
                for relative in SEMANTIC_CONTRACT_FILES
            }
        ),
    }


def one_click_gate_commands(
    python_executable: str,
) -> tuple[tuple[str, ...], ...]:
    return (
        (
            str(python_executable),
            "-m",
            "uamco.cli",
            "validate-config",
            "--config",
            "configs/formal_a40x4.yaml",
        ),
        (
            str(python_executable),
            "tools/validate_completion_feasibility.py",
            "--config",
            "configs/formal_a40x4.yaml",
        ),
        (
            str(python_executable),
            "tools/validate_objective_semantics.py",
        ),
        (
            str(python_executable),
            "tools/validate_training_semantics.py",
            "--config",
            "configs/smoke_actual_delivery.yaml",
        ),
        (
            str(python_executable),
            "tools/validate_baseline_adapters.py",
            "--config",
            "configs/smoke_training.yaml",
        ),
        (
            str(python_executable),
            "tools/run_causal_advantage_smoke.py",
            "--config",
            "configs/smoke_causal_advantage.yaml",
            "--episodes",
            "3",
            "--fresh",
            "--require-advantage",
        ),
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")


@dataclass(frozen=True, slots=True)
class ExperimentJob:
    job_id: str
    stage: str
    method: str
    fold: str
    seed: int
    episodes: int
    delay_weight: float | None = None
    variant: str | None = None

    def command(self, *, config_path: str, python_executable: str) -> tuple[str, ...]:
        command = [
            python_executable,
            "-m",
            "uamco.cli",
            "train",
            "--config",
            config_path,
            "--job-id",
            self.job_id,
            "--stage",
            self.stage,
            "--method",
            self.method,
            "--fold",
            self.fold,
            "--seed",
            str(self.seed),
            "--episodes",
            str(self.episodes),
        ]
        if self.delay_weight is not None:
            command.extend(("--delay-weight", f"{self.delay_weight:.2f}"))
        if self.variant is not None:
            command.extend(("--variant", self.variant))
        return tuple(command)


def _job(
    *,
    stage: str,
    method: str,
    fold: str,
    seed: int,
    episodes: int,
    delay_weight: float | None = None,
    variant: str | None = None,
) -> ExperimentJob:
    components = [stage, method, fold, f"s{seed}"]
    if delay_weight is not None:
        components.append(f"b{int(round(delay_weight * 100)):02d}")
    if variant is not None:
        components.append(variant)
    return ExperimentJob(
        job_id="__".join(_slug(component) for component in components),
        stage=stage,
        method=method,
        fold=fold,
        seed=int(seed),
        episodes=int(episodes),
        delay_weight=delay_weight,
        variant=variant,
    )


def build_training_jobs(config: Mapping, *, suite: str = "all") -> tuple[ExperimentJob, ...]:
    if suite not in {"main", "all"}:
        raise ValueError("training suite must be 'main' or 'all'")
    folds = tuple(str(value) for value in config["training"]["workflow_folds"])
    seeds = tuple(int(value) for value in config["training"]["seeds"])
    method_episodes = {
        str(method): int(value)
        for method, value in config["training"]["main_method_episodes"].items()
    }
    if set(method_episodes) != set(FORMAL_METHODS):
        raise ValueError("main method episode budgets must cover all formal methods")
    if any(value <= 0 for value in method_episodes.values()):
        raise ValueError("main method episode budgets must be positive")
    ablation_episodes = int(config["training"]["ablation_episodes"])
    if ablation_episodes <= 0:
        raise ValueError("ablation episode budget must be positive")
    jobs: list[ExperimentJob] = []
    for fold in folds:
        for seed in seeds:
            jobs.append(
                _job(
                    stage="main",
                    method="UAMCO-DAG",
                    fold=fold,
                    seed=seed,
                    episodes=method_episodes["UAMCO-DAG"],
                )
            )
            for method in FORMAL_METHODS[1:]:
                jobs.append(
                    _job(
                        stage="main",
                        method=method,
                        fold=fold,
                        seed=seed,
                        episodes=method_episodes[method],
                    )
                )
    if suite == "all":
        for variant in ABLATION_VARIANTS:
            for fold in folds:
                for seed in seeds:
                    jobs.append(
                        _job(
                            stage="ablation",
                            method="UAMCO-DAG",
                            fold=fold,
                            seed=seed,
                            episodes=ablation_episodes,
                            variant=variant,
                        )
                    )
    if len({job.job_id for job in jobs}) != len(jobs):
        raise RuntimeError("experiment matrix generated duplicate job identifiers")
    return tuple(jobs)


def estimate_formal_rollout_count(
    config: Mapping,
    jobs: Sequence[ExperimentJob],
) -> int:
    validation_interval = int(config["training"]["validation_interval"])
    validation_episodes = int(config["training"]["validation_episodes"])
    evaluation_episodes = int(config["evaluation"]["episodes_per_condition"])
    sensitivity_episodes = int(config["evaluation"]["sensitivity_episodes"])
    studies = config["evaluation"]["studies"]
    sensitivity_condition_count = (
        len(studies["scalability_layouts"])
        + len(studies["workflow_trigger_distance_m"])
        + len(studies["contact_forecast_error"])
        + len(studies["unavailable_uavs"])
        + len(studies["deadline_multiplier_scale"])
    )
    total = 0
    for job in jobs:
        validation_checks = job.episodes // validation_interval
        validation_weights = 1
        if job.stage == "ablation":
            evaluation_conditions = 1
            evaluation_weights = 1
        else:
            evaluation_conditions = 3
            evaluation_weights = 1
        sensitivity_enabled = (
            job.stage == "main"
            and (
                job.method == "UAMCO-DAG"
                or job.method in FORMAL_METHODS[1:]
            )
        )
        total += int(job.episodes)
        total += (
            validation_checks * validation_weights * validation_episodes
        )
        total += (
            evaluation_conditions * evaluation_weights * evaluation_episodes
        )
        if sensitivity_enabled:
            total += sensitivity_condition_count * sensitivity_episodes
        elif job.stage == "ablation" and job.variant == "without_cost_critics":
            total += (
                len(studies["deadline_multiplier_scale"])
                * sensitivity_episodes
            )
    return int(total)


def assign_gpu_round_robin(
    jobs: Sequence[ExperimentJob],
    *,
    gpu_ids: Sequence[int],
) -> tuple[tuple[ExperimentJob, int], ...]:
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU identifiers must be unique and non-empty")
    return tuple((job, int(gpu_ids[index % len(gpu_ids)])) for index, job in enumerate(jobs))


def build_worker_slots(
    gpu_ids: Sequence[int],
    workers_per_gpu: int,
    workers_per_gpu_by_id: Mapping[int | str, int] | None = None,
) -> tuple[int, ...]:
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU identifiers must be unique and non-empty")
    if workers_per_gpu <= 0:
        raise ValueError("workers per GPU must be positive")
    overrides = workers_per_gpu_by_id or {}
    counts = {
        int(gpu_id): int(
            overrides.get(
                str(gpu_id),
                overrides.get(int(gpu_id), workers_per_gpu),
            )
        )
        for gpu_id in gpu_ids
    }
    if any(count <= 0 for count in counts.values()):
        raise ValueError("every configured GPU must have at least one worker")
    return tuple(
        int(gpu_id)
        for worker_index in range(max(counts.values()))
        for gpu_id in gpu_ids
        if worker_index < counts[int(gpu_id)]
    )


def dry_run_preview(
    jobs: Sequence[ExperimentJob],
    *,
    gpu_ids: Sequence[int],
    workers_per_gpu: int = 1,
    workers_per_gpu_by_id: Mapping[int | str, int] | None = None,
) -> dict:
    assignments = assign_gpu_round_robin(jobs, gpu_ids=gpu_ids)
    counts = {str(gpu_id): 0 for gpu_id in gpu_ids}
    for _, gpu_id in assignments:
        counts[str(gpu_id)] += 1
    return {
        "dry_run": True,
        "worker_count": len(
            build_worker_slots(
                gpu_ids,
                workers_per_gpu,
                workers_per_gpu_by_id,
            )
        ),
        "workers_per_gpu": int(workers_per_gpu),
        "workers_per_gpu_by_id": {
            str(gpu_id): build_worker_slots(
                gpu_ids,
                workers_per_gpu,
                workers_per_gpu_by_id,
            ).count(int(gpu_id))
            for gpu_id in gpu_ids
        },
        "job_count": len(jobs),
        "launch_count": 0,
        "jobs_per_gpu": counts,
        "first_jobs": [
            {**asdict(job), "gpu_id": gpu_id} for job, gpu_id in assignments[: min(8, len(assignments))]
        ],
    }


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information,
            False,
            int(pid),
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(
                handle,
                ctypes.byref(exit_code),
            ):
                return False
            return int(exit_code.value) == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), signal.SIG_DFL)
    except (OSError, ProcessLookupError):
        return False
    return True


def filter_pending_jobs(
    jobs: Sequence[ExperimentJob],
    *,
    status_dir: str | Path,
    resume: bool,
    alive: Callable[[int], bool] = process_is_alive,
) -> tuple[ExperimentJob, ...]:
    if not resume:
        return tuple(jobs)
    root = Path(status_dir)
    pending = []
    for job in jobs:
        path = root / f"{job.job_id}.json"
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
            state = status.get("state")
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            status = {}
            state = None
        if state != "succeeded":
            if state == "running":
                try:
                    pid = int(status.get("pid", 0))
                except (TypeError, ValueError):
                    pid = 0
                if pid > 0 and alive(pid):
                    continue
            pending.append(job)
    return tuple(pending)


def _formal_job_contract(job: ExperimentJob) -> dict[str, object]:
    return {**asdict(job), "protocol": "formal"}


def clean_incompatible_resume_artifacts(
    jobs: Sequence[ExperimentJob],
    *,
    output_root: str | Path,
    objective_bounds_identity_sha256: str,
    checkpoint_loader: Callable[[Path], Mapping],
    semantic_contract: Mapping | None = None,
) -> tuple[dict[str, object], ...]:
    """Delete only per-job generated artifacts that cannot be resumed safely."""
    root = Path(output_root).resolve()
    records: list[dict[str, object]] = []

    def artifact_paths(job_id: str) -> tuple[Path, ...]:
        if Path(job_id).name != job_id:
            raise ValueError("experiment job identifier cannot contain a path")
        paths = (
            root / "checkpoints" / f"{job_id}.pt",
            root / "checkpoints" / f"{job_id}.best.pt",
            root / "status" / f"{job_id}.json",
            root / "progress" / "episodes" / f"{job_id}.json",
            root / "progress" / "micro_slots" / f"{job_id}.json",
            root / "progress" / f"{job_id}.json",
            root / "logs" / f"{job_id}.log",
            root / "job_results" / f"{job_id}.json",
        )
        for path in paths:
            resolved = path.resolve()
            if root not in resolved.parents:
                raise ValueError("resume artifact path escapes the output root")
        return paths

    def checkpoint_reason(
        path: Path,
        expected_job: Mapping[str, object],
    ) -> str | None:
        try:
            payload = checkpoint_loader(path)
        except Exception:
            return f"{path.name} is unreadable"
        if not isinstance(payload, Mapping):
            return f"{path.name} payload is not a mapping"
        if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            return f"{path.name} uses an incompatible checkpoint schema"
        if payload.get("job") != expected_job:
            return f"{path.name} uses a different job contract"
        if (
            payload.get("objective_bounds_identity_sha256")
            != objective_bounds_identity_sha256
        ):
            return f"{path.name} uses different objective bounds"
        if (
            semantic_contract is not None
            and payload.get("semantic_contract") != semantic_contract
        ):
            return f"{path.name} uses incompatible configuration, data, or environment semantics"
        return None

    def result_reason(
        path: Path,
        expected_job: Mapping[str, object],
    ) -> str | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return f"{path.name} is unreadable"
        if not isinstance(payload, Mapping):
            return f"{path.name} payload is not a mapping"
        if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            return f"{path.name} uses an incompatible result schema"
        if payload.get("job") != expected_job:
            return f"{path.name} uses a different job contract"
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping) or (
            provenance.get("objective_bounds_identity_sha256")
            != objective_bounds_identity_sha256
        ):
            return f"{path.name} uses different objective bounds"
        if (
            semantic_contract is not None
            and provenance.get("semantic_contract") != semantic_contract
        ):
            return f"{path.name} uses incompatible configuration, data, or environment semantics"
        return None

    for job in jobs:
        paths = artifact_paths(job.job_id)
        resume_path, best_path, status_path, _, _, _, _, result_path = paths
        expected_job = _formal_job_contract(job)
        reason = None
        for checkpoint_path in (resume_path, best_path):
            if checkpoint_path.is_file():
                reason = checkpoint_reason(checkpoint_path, expected_job)
                if reason is not None:
                    break
        if reason is None and result_path.is_file():
            reason = result_reason(result_path, expected_job)
        status_state = None
        if reason is None and status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                status_state = status.get("state") if isinstance(status, Mapping) else None
            except (OSError, UnicodeError, json.JSONDecodeError):
                reason = f"{status_path.name} is unreadable"
        if reason is None and status_state == "succeeded" and not result_path.is_file():
            reason = "succeeded status has no compatible job result"
        if reason is None:
            continue
        removed: list[str] = []
        for path in paths:
            if path.is_file():
                path.unlink()
                removed.append(path.relative_to(root).as_posix())
        records.append(
            {
                "job_id": job.job_id,
                "reason": reason,
                "removed_paths": removed,
            }
        )
    return tuple(records)


def _command_line_owns_job(
    command_line: str,
    *,
    project_root: str | Path,
    job_id: str,
) -> bool:
    normalized = " ".join(str(command_line).split()).lower()
    project = str(Path(project_root).resolve()).lower()
    return (
        bool(normalized)
        and project in normalized
        and "-m uamco.cli train" in normalized
        and f"--job-id {job_id.lower()}" in normalized
    )


def recover_stale_running_jobs(
    jobs: Sequence[ExperimentJob],
    *,
    status_dir: str | Path,
    project_root: str | Path,
    alive: Callable[[int], bool],
    command_line_for_pid: Callable[[int], str],
    terminate: Callable[[int], object],
) -> tuple[str, ...]:
    """Recover statuses left behind after the exclusive queue owner has exited."""
    root = Path(status_dir)
    recovered: list[str] = []
    for job in jobs:
        status_path = root / f"{job.job_id}.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        if status.get("state") != "running":
            continue
        try:
            pid = int(status.get("pid", 0))
        except (TypeError, ValueError):
            pid = 0
        process_alive = pid > 0 and alive(pid)
        owns_job = False
        if process_alive:
            try:
                command_line = command_line_for_pid(pid)
            except OSError:
                command_line = ""
            owns_job = _command_line_owns_job(
                command_line,
                project_root=project_root,
                job_id=job.job_id,
            )
        if process_alive and owns_job:
            terminate(pid)
            if alive(pid):
                raise RuntimeError(
                    f"cannot terminate orphan training process {pid} for {job.job_id}"
                )
            reason = "verified orphan training process terminated before resume"
        elif process_alive:
            reason = "running status PID belongs to another process and was not terminated"
        else:
            reason = "running status has no live matching process"
        atomic_write_status(
            status_path,
            {
                **status,
                "state": "interrupted",
                "previous_state": "running",
                "recovery_reason": reason,
                "recovered_at_unix_s": time.time(),
            },
        )
        recovered.append(job.job_id)
    return tuple(recovered)


def _strict_json_value(value):
    if isinstance(value, Mapping):
        return {key: _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write_status(path: str | Path, payload: Mapping) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            _strict_json_value(dict(payload)),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, destination)
