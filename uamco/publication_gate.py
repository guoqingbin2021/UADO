from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


REQUIRED_FAMILIES = ("Montage", "Seismology", "Cycles")


@dataclass(frozen=True, slots=True)
class GateReport:
    passed: bool
    checks: Mapping[str, bool]
    failures: tuple[str, ...]
    allow_advantage_claims: bool


def _relative_reduction(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if candidate == 0 else float("-inf")
    return (baseline - candidate) / baseline


def validate_publication_results(payload: Mapping) -> GateReport:
    families = payload.get("families", {})
    if set(families) != set(REQUIRED_FAMILIES):
        raise ValueError(f"publication gate requires exactly {REQUIRED_FAMILIES}")
    checks: dict[str, bool] = {}
    failures: list[str] = []
    epsilon = 1e-12

    for family in REQUIRED_FAMILIES:
        values = families[family]
        proposed = float(values["proposed_pnct"])
        baseline = float(values["strongest_sota_pnct"])
        improvement = _relative_reduction(proposed, baseline)
        advantage_ok = improvement + epsilon >= 0.10
        significance_ok = float(values["adjusted_p"]) < 0.05
        checks[f"{family}_pnct_advantage"] = advantage_ok
        checks[f"{family}_pnct_significance"] = significance_ok
        if not advantage_ok:
            failures.append(
                f"{family}: PNCT improvement {100 * improvement:.2f}% is below the required 10%."
            )
        if not significance_ok:
            failures.append(f"{family}: Holm-adjusted PNCT comparison is not significant.")

    proposed_miss = float(payload["proposed_miss_ratio"])
    baseline_miss = float(payload["sota_miss_ratio"])
    proposed_remaining = float(payload["proposed_remaining_ratio"])
    baseline_remaining = float(payload["sota_remaining_ratio"])
    proposed_drop = float(payload["proposed_drop_ratio"])
    baseline_drop = float(payload["sota_drop_ratio"])
    miss_reduction = _relative_reduction(proposed_miss, baseline_miss)
    remaining_reduction = _relative_reduction(
        proposed_remaining, baseline_remaining
    )
    no_executability_regression = (
        proposed_miss <= baseline_miss + epsilon
        and proposed_remaining <= baseline_remaining + epsilon
        and proposed_drop <= baseline_drop + epsilon
    )
    executability_gain = (
        max(miss_reduction, remaining_reduction) + epsilon >= 0.20
    )
    checks["executability_no_regression"] = no_executability_regression
    checks["executability_twenty_percent_gain"] = executability_gain
    if not no_executability_regression:
        failures.append(
            "Deadline-miss, remaining-DAG, or TTL-drop ratio regresses against "
            "the strongest SOTA."
        )
    if not executability_gain:
        failures.append(
            "Neither deadline-miss nor remaining-DAG ratio improves by the "
            "required 20%."
        )

    proposed_edp = float(payload["proposed_edp"])
    baseline_edp = float(payload["sota_edp"])
    edp_reduction = _relative_reduction(proposed_edp, baseline_edp)
    edp_ok = edp_reduction + epsilon >= 0.10
    checks["edp_advantage"] = edp_ok
    if not edp_ok:
        failures.append(
            f"Normalized EDP reduction {100 * edp_reduction:.2f}% is below 10%."
        )

    proposed_energy_per_completion = float(
        payload["proposed_energy_per_completed_dag_j"]
    )
    baseline_energy_per_completion = float(
        payload["sota_energy_per_completed_dag_j"]
    )
    energy_efficiency_reduction = _relative_reduction(
        proposed_energy_per_completion,
        baseline_energy_per_completion,
    )
    energy_efficiency_ok = energy_efficiency_reduction + epsilon >= 0.05
    checks["energy_per_completed_dag_advantage"] = energy_efficiency_ok
    if not energy_efficiency_ok:
        failures.append(
            "Energy per successfully delivered DAG does not improve by the required 5%."
        )

    zero_shot = payload["m2dgr"]
    significant_degradation = (
        float(zero_shot["proposed_pnct"]) > float(zero_shot["sota_pnct"]) + epsilon
        and float(zero_shot["p_value"]) < 0.05
    )
    checks["m2dgr_no_significant_degradation"] = not significant_degradation
    if significant_degradation:
        failures.append("M2DGR zero-shot PNCT is significantly worse than the strongest SOTA.")

    passed = all(checks.values())
    return GateReport(
        passed=passed,
        checks=checks,
        failures=tuple(failures),
        allow_advantage_claims=passed,
    )
