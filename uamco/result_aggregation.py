from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

import numpy as np

from .experiment_matrix import FORMAL_METHODS
from .calibration import FrozenObjectiveBounds, load_frozen_objective_bounds
from .metrics import ParetoPoint, pareto_point_from_episode_metrics, pareto_summary
from .statistics import confidence_interval_95, holm_adjust, paired_comparison


@dataclass(frozen=True, slots=True)
class AggregatedFormalResults:
    gate_input: dict
    plot_payload: dict
    summary: dict[str, float]
    statistical_report: dict


def _weight_key(value: float) -> float:
    return round(float(value), 6)


def _relative_reduction(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if candidate == 0 else -math.inf
    return (baseline - candidate) / baseline


def _finite_metric(value) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _pooled_energy_per_completed_dag(
    energy_per_admitted_dag: Sequence[float],
    completion_ratios: Sequence[float],
) -> float:
    """Estimate energy per success without conditioning away failed cells.

    Formal evaluation admits the same number of DAGs in every fold/seed cell.
    Consequently, total energy divided by total completed DAGs reduces to the
    ratio below.  A cell with no completion remains in the energy numerator and
    contributes zero to the denominator.  This avoids both division by zero at
    the cell level and the optimistic bias caused by dropping failed cells.
    """

    if len(energy_per_admitted_dag) != len(completion_ratios):
        raise ValueError("pooled energy and completion vectors must have equal length")
    denominator = float(sum(completion_ratios))
    if denominator <= 0.0:
        return math.inf
    return float(sum(energy_per_admitted_dag) / denominator)


def _pooled_energy_ci95(
    energy_per_admitted_dag: Sequence[float],
    completion_ratios: Sequence[float],
    *,
    bootstrap_samples: int = 10_000,
) -> tuple[float, float]:
    """Cell-paired bootstrap interval for the pooled energy estimator."""

    if len(energy_per_admitted_dag) != len(completion_ratios):
        raise ValueError("pooled energy and completion vectors must have equal length")
    if not energy_per_admitted_dag:
        raise ValueError("pooled energy estimator requires at least one cell")
    energy = np.asarray(energy_per_admitted_dag, dtype=float)
    completion = np.asarray(completion_ratios, dtype=float)
    if float(completion.sum()) <= 0.0:
        return math.inf, math.inf
    rng = np.random.default_rng(20260804)
    indices = rng.integers(0, len(energy), size=(bootstrap_samples, len(energy)))
    energy_sums = energy[indices].sum(axis=1)
    completion_sums = completion[indices].sum(axis=1)
    estimates = energy_sums[completion_sums > 0.0] / completion_sums[
        completion_sums > 0.0
    ]
    if estimates.size == 0:
        return math.inf, math.inf
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def common_pareto_delay_weights(config: Mapping) -> tuple[float, ...]:
    """Legacy reader for old artifacts; the active protocol has one fixed point."""
    del config
    return (0.5,)


def aggregate_formal_records(
    records: Sequence[Mapping],
    config: Mapping,
    *,
    objective_bounds: FrozenObjectiveBounds | None = None,
) -> AggregatedFormalResults:
    if objective_bounds is None:
        calibration = config.get("calibration", {})
        path = calibration.get("frozen_bounds")
        if not path:
            raise ValueError("formal aggregation requires calibration.frozen_bounds")
        objective_bounds = load_frozen_objective_bounds(path, project_root=".")
    seeds = tuple(int(value) for value in config["training"]["seeds"])
    folds = tuple(str(value) for value in config["training"]["workflow_folds"])
    primary_weight = 0.5
    grouped: dict[tuple[str, str, int, str, float, str], list[float]] = {}
    sensitivity_grouped: dict[tuple[str, str, float, str], list[float]] = {}
    reference_sensitivity_grouped: dict[
        tuple[str, float, float], list[float]
    ] = {}
    ablation_grouped: dict[tuple[str, str], list[float]] = {}
    for record in records:
        job = record.get("job", {})
        if job.get("protocol", "formal") != "formal":
            raise ValueError("smoke results cannot enter formal aggregation")
        if record.get("state") != "succeeded":
            continue
        method = str(job.get("method"))
        fold = str(job.get("fold"))
        seed = int(job.get("seed"))
        if job.get("stage") == "ablation":
            variant = str(job.get("variant"))
            for evaluation in record.get("evaluation", ()):
                pnct = _finite_metric(evaluation.get("metrics", {}).get("pnct_mean"))
                if (
                    evaluation.get("mobility_condition") == "RELLIS-3D-test"
                    and math.isclose(float(evaluation.get("delay_weight", -1)), primary_weight)
                    and pnct is not None
                ):
                    ablation_grouped.setdefault((variant, f"{fold}:{seed}"), []).append(
                        pnct
                    )
            continue
        if job.get("stage") != "main":
            continue
        for evaluation in record.get("evaluation", ()): 
            mobility = str(evaluation["mobility_condition"])
            weight = _weight_key(evaluation["delay_weight"])
            for metric, value in evaluation.get("metrics", {}).items():
                numeric = _finite_metric(value)
                if numeric is not None:
                    grouped.setdefault(
                        (method, fold, seed, mobility, weight, str(metric)), []
                    ).append(numeric)
        for evaluation in record.get("sensitivity", ()):
            study = str(evaluation["study"])
            if study == "hypervolume_reference":
                reference = evaluation.get("reference")
                hypervolume = _finite_metric(evaluation.get("hypervolume"))
                if (
                    not isinstance(reference, (list, tuple))
                    or len(reference) != 2
                    or hypervolume is None
                ):
                    raise ValueError(
                        "hypervolume-reference sensitivity requires a finite "
                        "two-dimensional reference and hypervolume"
                    )
                delay_reference = float(reference[0])
                energy_reference = float(reference[1])
                if not math.isfinite(delay_reference) or not math.isfinite(
                    energy_reference
                ):
                    raise ValueError(
                        "hypervolume-reference coordinates must be finite"
                    )
                reference_sensitivity_grouped.setdefault(
                    (method, delay_reference, energy_reference), []
                ).append(hypervolume)
                continue
            if "x" not in evaluation:
                raise ValueError(
                    f"sensitivity study {study!r} requires an x coordinate"
                )
            x_value = float(evaluation["x"])
            for metric, raw_value in evaluation.get("metrics", {}).items():
                numeric = _finite_metric(raw_value)
                if numeric is not None:
                    sensitivity_grouped.setdefault(
                        (study, method, x_value, str(metric)), []
                    ).append(numeric)

    def value(method: str, fold: str, seed: int, mobility: str, weight: float, metric: str) -> float:
        values = grouped.get((method, fold, seed, mobility, _weight_key(weight), metric), ())
        if not values:
            raise ValueError(
                f"incomplete formal results for {method}/{fold}/seed={seed}/{mobility}/"
                f"weight={weight}/{metric}"
            )
        return float(mean(values))

    def vector(method: str, fold: str, mobility: str, metric: str) -> list[float]:
        return [value(method, fold, seed, mobility, primary_weight, metric) for seed in seeds]

    family_payload: dict[str, dict] = {}
    raw_p_values: dict[str, float] = {}
    family_statistics: dict[str, dict] = {}
    for fold in folds:
        proposed = vector("UAMCO-DAG", fold, "RELLIS-3D-test", "pnct_mean")
        baseline_vectors = {
            method: vector(method, fold, "RELLIS-3D-test", "pnct_mean")
            for method in FORMAL_METHODS[1:]
        }
        strongest_method = min(
            baseline_vectors,
            key=lambda method: (mean(baseline_vectors[method]), method),
        )
        comparison = paired_comparison(baseline_vectors[strongest_method], proposed)
        family = fold.title()
        raw_p_values[family] = comparison.p_value
        family_statistics[family] = {
            "strongest_sota": strongest_method,
            "test": comparison.test_name,
            "raw_p": comparison.p_value,
            "effect_size": comparison.effect_size,
            "relative_improvement": comparison.relative_improvement,
            "shapiro_p": comparison.shapiro_p,
        }
        family_payload[family] = {
            "proposed_pnct": float(mean(proposed)),
            "strongest_sota_pnct": float(mean(baseline_vectors[strongest_method])),
            "adjusted_p": 1.0,
            "strongest_sota": strongest_method,
        }
    adjusted = holm_adjust(raw_p_values)
    for family, adjusted_p in adjusted.items():
        family_payload[family]["adjusted_p"] = adjusted_p
        family_statistics[family]["adjusted_p"] = adjusted_p

    def all_primary(method: str, mobility: str, metric: str) -> list[float]:
        return [
            value(method, fold, seed, mobility, primary_weight, metric)
            for fold in folds
            for seed in seeds
        ]

    def pooled_energy_per_completed(method: str, mobility: str) -> float:
        return _pooled_energy_per_completed_dag(
            all_primary(method, mobility, "system_energy_per_admitted_dag_j"),
            all_primary(method, mobility, "dag_completion_ratio"),
        )

    def pooled_energy_interval(method: str, mobility: str) -> tuple[float, float]:
        return _pooled_energy_ci95(
            all_primary(method, mobility, "system_energy_per_admitted_dag_j"),
            all_primary(method, mobility, "dag_completion_ratio"),
        )

    proposed_miss = float(mean(all_primary("UAMCO-DAG", "RELLIS-3D-test", "deadline_miss_ratio")))
    proposed_remaining = float(
        mean(all_primary("UAMCO-DAG", "RELLIS-3D-test", "remaining_ratio"))
    )
    proposed_drop = float(mean(all_primary("UAMCO-DAG", "RELLIS-3D-test", "dag_drop_ratio")))
    sota_miss = min(
        float(mean(all_primary(method, "RELLIS-3D-test", "deadline_miss_ratio")))
        for method in FORMAL_METHODS[1:]
    )
    sota_drop = min(
        float(mean(all_primary(method, "RELLIS-3D-test", "dag_drop_ratio")))
        for method in FORMAL_METHODS[1:]
    )
    sota_remaining = min(
        float(mean(all_primary(method, "RELLIS-3D-test", "remaining_ratio")))
        for method in FORMAL_METHODS[1:]
    )

    edp_by_method = {
        method: all_primary(method, "RELLIS-3D-test", "normalized_edp")
        for method in FORMAL_METHODS
    }
    strongest_edp_method = min(
        FORMAL_METHODS[1:], key=lambda method: (mean(edp_by_method[method]), method)
    )
    proposed_edp = float(mean(edp_by_method["UAMCO-DAG"]))
    sota_edp = float(mean(edp_by_method[strongest_edp_method]))
    proposed_energy_per_completed = pooled_energy_per_completed(
        "UAMCO-DAG", "RELLIS-3D-test"
    )
    sota_energy_per_completed = min(
        pooled_energy_per_completed(method, "RELLIS-3D-test")
        for method in FORMAL_METHODS[1:]
    )

    proposed_zero_shot = [
        float(
            mean(
                value(
                    "UAMCO-DAG", fold, seed, "M2DGR-Outdoor-zero-shot", primary_weight, "pnct_mean"
                )
                for fold in folds
            )
        )
        for seed in seeds
    ]
    zero_shot_baselines = {
        method: [
            float(
                mean(
                    value(
                        method, fold, seed, "M2DGR-Outdoor-zero-shot", primary_weight, "pnct_mean"
                    )
                    for fold in folds
                )
            )
            for seed in seeds
        ]
        for method in FORMAL_METHODS[1:]
    }
    strongest_zero_shot = min(
        zero_shot_baselines,
        key=lambda method: (mean(zero_shot_baselines[method]), method),
    )
    if np.allclose(proposed_zero_shot, zero_shot_baselines[strongest_zero_shot]):
        degradation_p = 1.0
    else:
        degradation_p = paired_comparison(
            proposed_zero_shot, zero_shot_baselines[strongest_zero_shot]
        ).p_value

    gate_input = {
        "families": family_payload,
        "proposed_miss_ratio": proposed_miss,
        "sota_miss_ratio": sota_miss,
        "proposed_remaining_ratio": proposed_remaining,
        "sota_remaining_ratio": sota_remaining,
        "proposed_drop_ratio": proposed_drop,
        "sota_drop_ratio": sota_drop,
        "proposed_edp": proposed_edp,
        "sota_edp": sota_edp,
        "proposed_energy_per_completed_dag_j": proposed_energy_per_completed,
        "sota_energy_per_completed_dag_j": sota_energy_per_completed,
        "m2dgr": {
            "proposed_pnct": float(mean(proposed_zero_shot)),
            "sota_pnct": float(mean(zero_shot_baselines[strongest_zero_shot])),
            "p_value": float(degradation_p),
            "strongest_sota": strongest_zero_shot,
        },
    }

    comparison_metrics = {
        "pnct": "pnct_mean",
        "effective_delay": "effective_delay_mean_s",
        "system_energy": "system_energy_per_admitted_dag_j",
        "completion": "dag_completion_ratio",
        "actual_delivery_progress": "actual_delivery_progress_potential",
        "normalized_edp": "normalized_edp",
        "deadline_miss": "deadline_miss_ratio",
        "dag_drop": "dag_drop_ratio",
        "throughput": "throughput_dag_per_s",
    }
    comparisons: dict[str, dict] = {}
    for plot_name, metric in comparison_metrics.items():
        comparisons[plot_name] = {}
        for method in FORMAL_METHODS:
            values = all_primary(method, "RELLIS-3D-test", metric)
            lower, upper = confidence_interval_95(values)
            comparisons[plot_name][method] = {
                "mean": float(mean(values)),
                "ci95": float((upper - lower) / 2.0),
            }
    comparisons["energy_per_completed_dag"] = {}
    for method in FORMAL_METHODS:
        estimate = pooled_energy_per_completed(method, "RELLIS-3D-test")
        lower, upper = pooled_energy_interval(method, "RELLIS-3D-test")
        comparisons["energy_per_completed_dag"][method] = {
            "mean": estimate,
            "ci95": float((upper - lower) / 2.0),
            "ci95_lower": lower,
            "ci95_upper": upper,
            "estimator": "pooled total energy / pooled completed DAGs",
        }
    efficiency_scatter: dict[str, list[tuple[float, float]]] = {}

    def mean_efficiency_point(method: str) -> ParetoPoint:
        return pareto_point_from_episode_metrics(
            {
                "effective_delay_mean_s": float(
                    mean(
                        value(
                            method,
                            fold,
                            seed,
                            "RELLIS-3D-test",
                            primary_weight,
                            "effective_delay_mean_s",
                        )
                        for fold in folds
                        for seed in seeds
                    )
                ),
                "system_energy_per_admitted_dag_j": float(
                    mean(
                        value(
                            method,
                            fold,
                            seed,
                            "RELLIS-3D-test",
                            primary_weight,
                            "system_energy_per_admitted_dag_j",
                        )
                        for fold in folds
                        for seed in seeds
                    )
                ),
            },
            frozen_bounds=objective_bounds,
            label="fixed",
        )

    for method in FORMAL_METHODS:
        point = mean_efficiency_point(method)
        efficiency_scatter[method] = [(point.delay_norm, point.energy_norm)]

    load_statistics: dict[str, dict[str, dict[str, float]]] = {}
    for metric in ("pnct_mean", "remaining_ratio", "throughput_dag_per_s"):
        load_statistics[metric] = {}
        for method in FORMAL_METHODS:
            values = all_primary(method, "RELLIS-3D-load", metric)
            lower, upper = confidence_interval_95(values)
            load_statistics[metric][method] = {
                "mean": float(mean(values)),
                "ci95": float((upper - lower) / 2.0),
            }

    family_improvements = [
        100.0
        * _relative_reduction(values["proposed_pnct"], values["strongest_sota_pnct"])
        for values in family_payload.values()
    ]
    summary = {
        "pnct_improvement_percent": float(min(family_improvements)),
        "edp_reduction_percent": float(100.0 * _relative_reduction(proposed_edp, sota_edp))
        if sota_edp > 0
        else math.inf,
        "energy_per_completed_dag_reduction_percent": float(
            100.0
            * _relative_reduction(
                proposed_energy_per_completed,
                sota_energy_per_completed,
            )
        ),
        "miss_reduction_percent": float(100.0 * _relative_reduction(proposed_miss, sota_miss)),
        "drop_reduction_percent": float(100.0 * _relative_reduction(proposed_drop, sota_drop)),
        "remaining_reduction_percent": float(
            100.0 * _relative_reduction(proposed_remaining, sota_remaining)
        ),
    }
    plot_payload: dict = {
        "comparisons": comparisons,
        "efficiency_scatter": efficiency_scatter,
    }
    energy_components = {
        "UGV compute": "energy_ugv_compute_per_admitted_dag_j",
        "UGV radio": "energy_ugv_radio_per_admitted_dag_j",
        "UAV compute": "energy_uav_compute_per_admitted_dag_j",
        "UAV radio": "energy_uav_radio_per_admitted_dag_j",
        "UAV propulsion": "energy_uav_propulsion_per_admitted_dag_j",
    }
    if all(
        (method, folds[0], seeds[0], "RELLIS-3D-test", primary_weight, metric) in grouped
        for method in FORMAL_METHODS
        for metric in energy_components.values()
    ):
        plot_payload["energy_breakdown"] = {
            method: {
                label: float(mean(all_primary(method, "RELLIS-3D-test", metric)))
                for label, metric in energy_components.items()
            }
            for method in FORMAL_METHODS
        }
    for study in ("scalability", "congestion", "contact_error", "uav_failure"):
        series: dict[str, dict[str, list[float]]] = {}
        for method in FORMAL_METHODS:
            x_values = sorted(
                {
                    key[2]
                    for key in sensitivity_grouped
                    if key[0] == study and key[1] == method and key[3] == "pnct_mean"
                }
            )
            if not x_values:
                continue
            means: list[float] = []
            intervals: list[float] = []
            for x_value in x_values:
                values = sensitivity_grouped[(study, method, x_value, "pnct_mean")]
                lower, upper = confidence_interval_95(values)
                means.append(float(mean(values)))
                intervals.append(float((upper - lower) / 2.0))
            series[method] = {"x": x_values, "mean": means, "ci95": intervals}
        if series:
            plot_payload[study] = series
    if ablation_grouped:
        variants = sorted({variant for variant, _ in ablation_grouped})
        means = []
        intervals = []
        for variant in variants:
            per_seed = [
                float(mean(values))
                for (name, _), values in ablation_grouped.items()
                if name == variant
            ]
            lower, upper = confidence_interval_95(per_seed)
            means.append(float(mean(per_seed)))
            intervals.append(float((upper - lower) / 2.0))
        plot_payload["ablation"] = {
            "UAMCO-DAG": {
                "x": list(range(len(variants))),
                "mean": means,
                "ci95": intervals,
            }
        }
        plot_payload["ablation_labels"] = variants

    reference_sensitivity_report: dict[str, list[dict[str, object]]] = {}
    for method in FORMAL_METHODS:
        rows = []
        for (
            grouped_method,
            delay_reference,
            energy_reference,
        ), values in sorted(reference_sensitivity_grouped.items()):
            if grouped_method != method:
                continue
            lower, upper = confidence_interval_95(values)
            rows.append(
                {
                    "reference": [delay_reference, energy_reference],
                    "mean": float(mean(values)),
                    "ci95": float((upper - lower) / 2.0),
                    "sample_count": len(values),
                }
            )
        if rows:
            reference_sensitivity_report[method] = rows

    return AggregatedFormalResults(
        gate_input=gate_input,
        plot_payload=plot_payload,
        summary=summary,
        statistical_report={
            "families": family_statistics,
            "strongest_edp_sota": strongest_edp_method,
            "strongest_m2dgr_sota": strongest_zero_shot,
            "fixed_horizon_load": load_statistics,
            "hypervolume_reference_sensitivity": reference_sensitivity_report,
            "energy_per_completed_dag_estimator": {
                "name": "pooled ratio estimator",
                "definition": (
                    "sum(system_energy_per_admitted_dag_j) / "
                    "sum(dag_completion_ratio)"
                ),
                "equal_admissions_per_cell": True,
                "zero_completion_cells_retained": True,
                "uncertainty": "paired cell bootstrap, 10000 resamples",
            },
        },
    )
