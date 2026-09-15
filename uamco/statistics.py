from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class PairedComparison:
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    relative_improvement: float
    n: int
    shapiro_p: float


def confidence_interval_95(values: Sequence[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.size < 2 or not np.isfinite(data).all():
        raise ValueError("confidence interval requires at least two finite values")
    center = float(data.mean())
    margin = float(stats.t.ppf(0.975, data.size - 1) * stats.sem(data))
    return center - margin, center + margin


def paired_comparison(
    strongest_sota: Sequence[float],
    proposed: Sequence[float],
) -> PairedComparison:
    baseline = np.asarray(strongest_sota, dtype=float)
    candidate = np.asarray(proposed, dtype=float)
    if (
        baseline.ndim != 1
        or candidate.ndim != 1
        or baseline.shape != candidate.shape
        or baseline.size < 7
    ):
        raise ValueError(
            "paired formal comparison requires matching vectors with at least seven seeds"
        )
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("paired values must be finite")
    differences = baseline - candidate
    if np.allclose(differences, differences[0]):
        shapiro_p = 1.0
        test_name = "constant_paired_difference"
        statistic = math.copysign(math.inf, float(differences[0])) if differences[0] != 0 else 0.0
        p_value = 0.0 if differences[0] > 0 else 1.0
    else:
        shapiro_p = float(stats.shapiro(differences).pvalue)
    if not np.allclose(differences, differences[0]) and shapiro_p >= 0.05:
        result = stats.ttest_rel(baseline, candidate, alternative="greater")
        test_name = "paired_t"
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    elif not np.allclose(differences, differences[0]):
        result = stats.wilcoxon(differences, alternative="greater", zero_method="wilcox")
        test_name = "wilcoxon_signed_rank"
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    standard_deviation = float(differences.std(ddof=1))
    if standard_deviation > 0:
        effect = float(differences.mean() / standard_deviation)
    elif differences.mean() == 0:
        effect = 0.0
    else:
        effect = math.copysign(math.inf, float(differences.mean()))
    baseline_mean = float(baseline.mean())
    improvement = float(differences.mean() / baseline_mean) if baseline_mean != 0 else math.nan
    return PairedComparison(
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
        effect_size=effect,
        relative_improvement=improvement,
        n=int(baseline.size),
        shapiro_p=shapiro_p,
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values or any(not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("Holm adjustment requires p-values in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    previous = 0.0
    for rank, (key, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * float(value))
        previous = max(previous, candidate)
        adjusted[key] = previous
    return adjusted
