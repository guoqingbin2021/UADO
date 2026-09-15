from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, median
from typing import Iterable, Mapping, Sequence

import numpy as np

from .calibration import FrozenObjectiveBounds


@dataclass(frozen=True, slots=True)
class WorkflowOutcome:
    workflow_id: str
    family: str
    sla_tier: str
    arrival_time_s: float
    deadline_time_s: float
    censor_time_s: float
    status: str
    completion_time_s: float | None
    missed_deadline: bool
    mobile_energy_j: float
    rsu_energy_j: float
    drop_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"completed", "dropped", "remaining"}:
            raise ValueError(f"unknown workflow outcome: {self.status}")
        if not all(
            math.isfinite(float(value))
            for value in (
                self.arrival_time_s,
                self.deadline_time_s,
                self.censor_time_s,
                self.mobile_energy_j,
                self.rsu_energy_j,
            )
        ):
            raise ValueError("workflow times and energies must be finite")
        if self.deadline_time_s <= self.arrival_time_s:
            raise ValueError("workflow deadline must follow its arrival")
        if self.censor_time_s < self.arrival_time_s:
            raise ValueError("workflow censor time cannot precede its arrival")
        if self.status == "completed" and self.completion_time_s is None:
            raise ValueError("completed workflow requires a completion time")
        if self.completion_time_s is not None and not math.isfinite(
            float(self.completion_time_s)
        ):
            raise ValueError("workflow completion time must be finite")
        if self.status == "completed" and not (
            self.arrival_time_s <= float(self.completion_time_s) <= self.censor_time_s
        ):
            raise ValueError("completed workflow time must fall within its observation interval")
        if self.mobile_energy_j < 0 or self.rsu_energy_j < 0:
            raise ValueError("energy cannot be negative")


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    delay_norm: float
    energy_norm: float
    label: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.delay_norm) or not math.isfinite(self.energy_norm):
            raise ValueError("Pareto objectives must be finite")


def penalized_normalized_completion_time(outcome: WorkflowOutcome) -> float:
    if outcome.status != "completed":
        return 2.0
    deadline_duration = outcome.deadline_time_s - outcome.arrival_time_s
    completion_duration = float(outcome.completion_time_s) - outcome.arrival_time_s
    return min(max(0.0, completion_duration / deadline_duration), 2.0)


def observed_delay_s(outcome: WorkflowOutcome) -> float:
    endpoint = (
        float(outcome.completion_time_s)
        if outcome.status == "completed"
        else float(outcome.censor_time_s)
    )
    return max(0.0, endpoint - outcome.arrival_time_s)


def end_penalty_s(outcome: WorkflowOutcome, *, upper_bound_s: float) -> float:
    if not math.isfinite(float(upper_bound_s)) or upper_bound_s < 0:
        raise ValueError("effective-delay upper bound must be finite and non-negative")
    if outcome.status == "completed":
        return 0.0
    return max(0.0, float(upper_bound_s) - observed_delay_s(outcome))


def effective_delay_s(outcome: WorkflowOutcome, *, upper_bound_s: float) -> float:
    return min(
        float(upper_bound_s),
        observed_delay_s(outcome) + end_penalty_s(outcome, upper_bound_s=upper_bound_s),
    )


def pareto_point_from_episode_metrics(
    metrics: Mapping[str, float],
    *,
    frozen_bounds: FrozenObjectiveBounds,
    label: str,
) -> ParetoPoint:
    return ParetoPoint(
        delay_norm=frozen_bounds.normalize_time(
            float(metrics["effective_delay_mean_s"])
        ),
        energy_norm=frozen_bounds.normalize_energy(
            float(metrics["system_energy_per_admitted_dag_j"])
        ),
        label=str(label),
    )


def compute_episode_metrics(
    outcomes: Sequence[WorkflowOutcome],
    *,
    episode_duration_s: float,
    objective_bounds: FrozenObjectiveBounds | None = None,
) -> dict[str, float]:
    if (
        not outcomes
        or not math.isfinite(float(episode_duration_s))
        or episode_duration_s <= 0
    ):
        raise ValueError("episode metrics require outcomes and positive finite duration")
    pnct = [penalized_normalized_completion_time(outcome) for outcome in outcomes]
    effective_delays = [
        effective_delay_s(outcome, upper_bound_s=episode_duration_s) for outcome in outcomes
    ]
    completed_durations = [
        float(outcome.completion_time_s) - outcome.arrival_time_s
        for outcome in outcomes
        if outcome.status == "completed"
    ]
    count = len(outcomes)
    completed_count = sum(outcome.status == "completed" for outcome in outcomes)
    mobile_energy = sum(outcome.mobile_energy_j for outcome in outcomes)
    rsu_energy = sum(outcome.rsu_energy_j for outcome in outcomes)
    effective_delay_mean_s = float(mean(effective_delays))
    system_energy_per_admitted_j = (mobile_energy + rsu_energy) / count
    metrics = {
        "pnct_mean": float(mean(pnct)),
        "pnct_median": float(median(pnct)),
        "pnct_p95": float(np.percentile(pnct, 95)),
        "effective_delay_mean_s": effective_delay_mean_s,
        "normalized_effective_delay": (
            objective_bounds.normalize_time(effective_delay_mean_s)
            if objective_bounds is not None
            else effective_delay_mean_s / episode_duration_s
        ),
        "completion_time_mean_s": float(mean(completed_durations)) if completed_durations else math.nan,
        "completion_time_p95_s": float(np.percentile(completed_durations, 95)) if completed_durations else math.nan,
        "deadline_miss_ratio": sum(outcome.missed_deadline for outcome in outcomes) / count,
        "dag_drop_ratio": sum(outcome.status == "dropped" for outcome in outcomes) / count,
        "remaining_ratio": sum(outcome.status == "remaining" for outcome in outcomes) / count,
        "throughput_dag_per_s": completed_count / float(episode_duration_s),
        "mobile_energy_per_admitted_dag_j": mobile_energy / count,
        "rsu_energy_per_admitted_dag_j": rsu_energy / count,
        "system_energy_per_admitted_dag_j": system_energy_per_admitted_j,
        "system_energy_per_completed_dag_j": (
            (mobile_energy + rsu_energy) / completed_count
            if completed_count > 0
            else math.nan
        ),
        "dag_completion_ratio": completed_count / count,
    }
    if objective_bounds is not None:
        metrics["normalized_system_energy"] = objective_bounds.normalize_energy(
            system_energy_per_admitted_j
        )
    return metrics


def _same_objectives(
    left: ParetoPoint,
    right: ParetoPoint,
    *,
    atol: float,
    rtol: float,
) -> bool:
    return (
        math.isclose(left.delay_norm, right.delay_norm, abs_tol=atol, rel_tol=rtol)
        and math.isclose(left.energy_norm, right.energy_norm, abs_tol=atol, rel_tol=rtol)
    )


def nondominated_points(
    points: Iterable[ParetoPoint],
    *,
    atol: float = 1.0e-8,
    rtol: float = 1.0e-7,
) -> tuple[ParetoPoint, ...]:
    if atol < 0 or rtol < 0:
        raise ValueError("Pareto tolerances cannot be negative")
    unique: list[ParetoPoint] = []
    for point in points:
        if not any(_same_objectives(point, existing, atol=atol, rtol=rtol) for existing in unique):
            unique.append(point)
    values = tuple(unique)
    nondominated = []
    for point in values:
        dominated = any(
            other is not point
            and other.delay_norm <= point.delay_norm
            and other.energy_norm <= point.energy_norm
            and (other.delay_norm < point.delay_norm or other.energy_norm < point.energy_norm)
            for other in values
        )
        if not dominated:
            nondominated.append(point)
    return tuple(sorted(nondominated, key=lambda point: (point.delay_norm, point.energy_norm)))


def hypervolume_2d(
    points: Iterable[ParetoPoint],
    *,
    reference: tuple[float, float],
) -> float:
    reference_delay, reference_energy = map(float, reference)
    front = [
        point
        for point in nondominated_points(points)
        if point.delay_norm <= reference_delay and point.energy_norm <= reference_energy
    ]
    volume = 0.0
    previous_energy = reference_energy
    for point in front:
        if point.energy_norm < previous_energy:
            volume += (reference_delay - point.delay_norm) * (previous_energy - point.energy_norm)
            previous_energy = point.energy_norm
    return float(volume)


def pareto_summary(
    points: Sequence[ParetoPoint],
    *,
    reference: tuple[float, float] = (1.05, 1.05),
) -> dict[str, float | int | tuple[ParetoPoint, ...]]:
    front = nondominated_points(points)
    if len(front) <= 1:
        spread = 0.0
    else:
        distances = [
            math.dist(
                (left.delay_norm, left.energy_norm),
                (right.delay_norm, right.energy_norm),
            )
            for left, right in zip(front, front[1:])
        ]
        spread = float(np.std(distances))
    return {
        "front": front,
        "nondominated_count": len(front),
        "hypervolume": hypervolume_2d(front, reference=reference),
        "spread": spread,
        "reference": tuple(map(float, reference)),
    }
