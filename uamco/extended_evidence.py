from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .formal_engine import evaluation_episode_seed


SCHEMA_VERSION = 1
INTEGRITY_METRICS = (
    "byte_conservation_error_count",
    "premature_ready_count",
    "single_server_concurrency_error_count",
    "successor_start_before_input_count",
)
SERVICE_METRICS = (
    "dag_completion_ratio",
    "pnct_mean",
    "actual_delivery_progress_potential",
)


@dataclass(frozen=True)
class JointStressCondition:
    trigger_distance_m: float
    outage_probability: float


@dataclass(frozen=True)
class WorkflowComplexityStratum:
    label: str
    workflow: object
    task_count: int
    edge_count: int
    total_cycles: float
    file_footprint_bytes: int


def joint_stress_grid(
    trigger_distances: Sequence[float] = (5.0, 20.0, 40.0),
    outage_probabilities: Sequence[float] = (0.0, 0.15, 0.30),
) -> tuple[JointStressCondition, ...]:
    distances = tuple(float(value) for value in trigger_distances)
    outages = tuple(float(value) for value in outage_probabilities)
    if not distances or any(not math.isfinite(value) or value <= 0 for value in distances):
        raise ValueError("trigger distances must be finite and positive")
    if not outages or any(
        not math.isfinite(value) or not 0.0 <= value < 1.0 for value in outages
    ):
        raise ValueError("outage probabilities must lie in [0, 1)")
    if tuple(sorted(set(distances))) != distances:
        raise ValueError("trigger distances must be unique and increasing")
    if tuple(sorted(set(outages))) != outages:
        raise ValueError("outage probabilities must be unique and increasing")
    return tuple(
        JointStressCondition(distance, outage)
        for distance in distances
        for outage in outages
    )


def paired_evaluation_seed(base_seed: int, *, study: str) -> int:
    condition_index = {
        "joint_stress": 31,
        "workflow_complexity": 32,
    }.get(str(study))
    if condition_index is None:
        raise ValueError(f"unknown extended-evidence study: {study}")
    return evaluation_episode_seed(
        int(base_seed),
        condition_index=condition_index,
        weight_index=0,
        replicate_index=0,
    )


def workflow_complexity_strata(
    workflows: Sequence[object],
) -> tuple[WorkflowComplexityStratum, ...]:
    if len(workflows) < 3:
        raise ValueError("workflow-complexity evaluation requires at least three workflows")
    ordered = sorted(
        workflows,
        key=lambda workflow: (
            sum(float(task.cycles) for task in workflow.tasks.values()),
            str(workflow.instance_id),
        ),
    )
    indices = (0, len(ordered) // 2, len(ordered) - 1)
    labels = ("low", "medium", "high")
    rows = []
    for label, index in zip(labels, indices):
        workflow = ordered[index]
        rows.append(
            WorkflowComplexityStratum(
                label=label,
                workflow=workflow,
                task_count=len(workflow.tasks),
                edge_count=len(workflow.edges),
                total_cycles=sum(
                    float(task.cycles) for task in workflow.tasks.values()
                ),
                file_footprint_bytes=sum(
                    int(value) for value in workflow.file_sizes.values()
                ),
            )
        )
    return tuple(rows)


def _valid_cell(cell: Mapping) -> bool:
    try:
        metrics = cell["metrics"]
        integrity = cell["semantic_integrity"]
        return all(
            math.isfinite(float(metrics[key])) for key in SERVICE_METRICS
        ) and all(float(integrity[key]) == 0.0 for key in INTEGRITY_METRICS)
    except (KeyError, TypeError, ValueError):
        return False


def validate_extended_payload(payload: Mapping, *, expected_job_id: str) -> bool:
    try:
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            return False
        if str(payload["job"]["job_id"]) != str(expected_job_id):
            return False
        checkpoint_hash = str(payload["checkpoint_sha256"])
        if len(checkpoint_hash) != 64:
            return False
        joint = list(payload["joint_stress"])
        expected_grid = joint_stress_grid()
        if len(joint) != len(expected_grid):
            return False
        observed_grid = [
            (float(row["trigger_distance_m"]), float(row["outage_probability"]))
            for row in joint
        ]
        expected_pairs = [
            (row.trigger_distance_m, row.outage_probability) for row in expected_grid
        ]
        if observed_grid != expected_pairs or not all(_valid_cell(row) for row in joint):
            return False
        complexity = list(payload["workflow_complexity"])
        if [str(row["stratum"]) for row in complexity] != [
            "low",
            "medium",
            "high",
        ]:
            return False
        if any(not str(row["workflow_instance_id"]) for row in complexity):
            return False
        return all(_valid_cell(row) for row in complexity)
    except (KeyError, TypeError, ValueError):
        return False
