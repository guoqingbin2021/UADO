from __future__ import annotations

import math
from typing import Sequence

from .calibration import FrozenObjectiveBounds
from .metrics import WorkflowOutcome, effective_delay_s


def bounded_unit(value: float) -> float:
    """Clip one frozen-normalized metric to the interval used by the proof."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("normalized objective value must be finite")
    return min(1.0, max(0.0, numeric))


def completion_dominance_coefficient(admitted_count: int) -> float:
    """Return the smallest simple integer coefficient used by the dominance proof."""
    count = int(admitted_count)
    if count <= 0:
        raise ValueError("completion dominance requires at least one admitted workflow")
    return float(2 * count + 1)


def delivery_first_episode_cost(
    *,
    normalized_delay: float,
    normalized_energy: float,
    actual_delivery_progress: float,
    failed_count: int,
    admitted_count: int,
) -> float:
    """Evaluate the fixed delivery-first finite-horizon cost.

    Completion dominates because the residual and bounded EDP terms together
    lie in [0, 2], while one additional failed DAG changes the first term by
    (2*N+1)/N > 2.
    """
    count = int(admitted_count)
    failures = int(failed_count)
    progress = float(actual_delivery_progress)
    if count <= 0 or failures < 0 or failures > count:
        raise ValueError("failed/admitted workflow counts are invalid")
    if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError("actual-delivery progress must lie in [0, 1]")
    delay = bounded_unit(normalized_delay)
    energy = bounded_unit(normalized_energy)
    failed_fraction = failures / count
    return float(
        completion_dominance_coefficient(count) * failed_fraction
        + (1.0 - progress)
        + delay * energy
    )


class FiniteHorizonObjectiveAccumulator:
    """Accumulates potential increments that telescope to frozen objectives."""

    def __init__(self, bounds: FrozenObjectiveBounds, *, admitted_count: int) -> None:
        if int(admitted_count) <= 0:
            raise ValueError("objective accounting requires at least one admitted workflow")
        self.bounds = bounds
        self.admitted_count = int(admitted_count)
        self.delay_cost_total = 0.0
        self.energy_cost_total = 0.0
        self.edp_cost_total = 0.0
        self._finalized = False

    def advance(
        self,
        *,
        active_workflow_ids: Sequence[str],
        dt_s: float,
        system_energy_increment_j: float,
    ) -> dict[str, float]:
        if self._finalized:
            raise RuntimeError("objective accumulator has already been finalized")
        if not math.isfinite(float(dt_s)) or float(dt_s) < 0:
            raise ValueError("micro-slot duration must be finite and non-negative")
        if not math.isfinite(float(system_energy_increment_j)):
            raise ValueError("system-energy increment must be finite")
        delay = (
            len(tuple(active_workflow_ids)) * float(dt_s) / self.admitted_count
            / self.bounds.time_max_s
        )
        energy = (
            float(system_energy_increment_j)
            / self.admitted_count
            / self.bounds.energy_max_j
        )
        self.delay_cost_total += delay
        self.energy_cost_total += energy
        current_edp = bounded_unit(self.delay_cost_total) * bounded_unit(
            self.energy_cost_total
        )
        edp_increment = current_edp - self.edp_cost_total
        self.edp_cost_total = current_edp
        return {"delay": delay, "energy": energy, "edp": edp_increment}

    def set_admitted_count(self, admitted_count: int) -> None:
        """Use the final known cohort size for terminal identity validation."""
        if self._finalized or int(admitted_count) < self.admitted_count:
            raise ValueError("admitted workflow count cannot decrease after accounting")
        self.admitted_count = int(admitted_count)

    def finalize(
        self,
        outcomes: Sequence[WorkflowOutcome],
        *,
        episode_duration_s: float,
    ) -> dict[str, float]:
        if self._finalized:
            raise ValueError("objective accumulator cannot be finalized twice")
        if len(outcomes) != self.admitted_count:
            raise ValueError("outcomes must contain every admitted workflow exactly once")
        if not math.isfinite(float(episode_duration_s)) or episode_duration_s <= 0:
            raise ValueError("episode duration must be finite and positive")
        effective_delay_mean_s = sum(
            effective_delay_s(outcome, upper_bound_s=float(episode_duration_s))
            for outcome in outcomes
        ) / self.admitted_count
        system_energy_per_admitted_j = sum(
            outcome.mobile_energy_j + outcome.rsu_energy_j for outcome in outcomes
        ) / self.admitted_count
        final_delay = self.bounds.normalize_time(effective_delay_mean_s)
        final_energy = self.bounds.normalize_energy(system_energy_per_admitted_j)
        final_edp = bounded_unit(final_delay) * bounded_unit(final_energy)
        correction = {
            "delay": final_delay - self.delay_cost_total,
            "energy": final_energy - self.energy_cost_total,
            "edp": final_edp - self.edp_cost_total,
        }
        self.delay_cost_total = final_delay
        self.energy_cost_total = final_energy
        self.edp_cost_total = final_edp
        self._finalized = True
        return correction
