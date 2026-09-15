from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import stats


def validate_trace(
    payload: Mapping[str, Any],
    *,
    expected_micro_slots: int | None = None,
) -> None:
    """Validate trace structure and arithmetic domains before plotting."""

    if not isinstance(payload.get("provenance"), Mapping):
        raise ValueError("trace provenance is missing")
    if not isinstance(payload.get("rollout_metrics"), Mapping):
        raise ValueError("trace rollout metrics are missing")
    records = payload.get("micro_slots")
    if not isinstance(records, list) or not records:
        raise ValueError("trace micro-slot records are missing")
    if expected_micro_slots is not None and len(records) != int(expected_micro_slots):
        raise ValueError(
            f"trace has {len(records)} micro-slots, expected {expected_micro_slots}"
        )
    indices = [int(record.get("micro_slot", -1)) for record in records]
    if indices != list(range(1, len(records) + 1)):
        raise ValueError("trace micro-slot indices must be contiguous from one")
    for record in records:
        ratio = float(record.get("gap_contact_ratio", 0.0))
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("gap-contact ratios must lie in [0, 1]")
        for key in (
            "gap_demand_weight",
            "contacted_gap_demand_weight",
            "uav_transfer_bytes",
            "uav_gap_transfer_bytes",
        ):
            value = float(record.get(key, 0.0))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{key} must be finite and nonnegative")


def _aggregate_records(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    gap_weight = sum(float(record.get("gap_demand_weight", 0.0)) for record in records)
    contacted_weight = sum(
        float(record.get("contacted_gap_demand_weight", 0.0))
        for record in records
    )
    uav_bytes = sum(float(record.get("uav_transfer_bytes", 0.0)) for record in records)
    uav_gap_bytes = sum(
        float(record.get("uav_gap_transfer_bytes", 0.0)) for record in records
    )
    raw_gap_missing_byte_exposure = sum(
        float(point.get("missing_bytes", 0.0))
        for record in records
        for point in record.get("gap_demand_points", ())
        if bool(point.get("rsu_uncovered"))
    )
    return {
        "micro_slots": len(records),
        "raw_gap_missing_byte_exposure": float(raw_gap_missing_byte_exposure),
        "gap_demand_weight_exposure": float(gap_weight),
        "contacted_gap_demand_weight_exposure": float(contacted_weight),
        "gap_contact_ratio": (
            float(contacted_weight / gap_weight) if gap_weight > 0.0 else 0.0
        ),
        "uav_transfer_bytes": float(uav_bytes),
        "uav_gap_transfer_bytes": float(uav_gap_bytes),
        "uav_gap_byte_fraction": (
            float(uav_gap_bytes / uav_bytes) if uav_bytes > 0.0 else 0.0
        ),
        "gap_file_deliveries": sum(
            int(record.get("gap_file_deliveries", 0)) for record in records
        ),
        "uav_compute_completions": sum(
            int(record.get("uav_compute_completions", 0)) for record in records
        ),
        "gap_uav_compute_completions": sum(
            int(record.get("gap_uav_compute_completions", 0))
            for record in records
        ),
        "active_gap_micro_slots": sum(
            float(record.get("gap_demand_weight", 0.0)) > 0.0 for record in records
        ),
        "uav_gap_service_micro_slots": sum(
            float(record.get("uav_gap_transfer_bytes", 0.0)) > 0.0
            or int(record.get("gap_uav_compute_completions", 0)) > 0
            for record in records
        ),
    }


def select_macro_slots(
    macro_slots: Mapping[str, Mapping[str, Any]],
    *,
    count: int = 4,
) -> list[int]:
    """Select evenly spaced snapshots of the nonzero RSU-gap-demand period."""

    requested = int(count)
    if requested <= 0:
        raise ValueError("selected macro-slot count must be positive")
    active = sorted(
        int(slot)
        for slot, values in macro_slots.items()
        if float(values.get("gap_demand_weight_exposure", 0.0)) > 0.0
    )
    if not active:
        return []
    if len(active) <= requested:
        return active
    indices = np.rint(np.linspace(0, len(active) - 1, requested)).astype(int)
    return [active[int(index)] for index in indices]


def aggregate_trace(
    payload: Mapping[str, Any],
    *,
    macro_interval_s: float,
) -> dict[str, Any]:
    """Aggregate exposure-weighted coverage and realized UAV service totals."""

    interval = float(macro_interval_s)
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError("macro interval must be finite and positive")
    validate_trace(payload)
    records = list(payload["micro_slots"])
    overall = _aggregate_records(records)
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["macro_slot"])].append(record)
    overall["macro_interval_s"] = interval
    macro_summaries = {
        str(slot): _aggregate_records(grouped[slot]) for slot in sorted(grouped)
    }
    overall["macro_slots"] = macro_summaries
    active_gap_slots = [
        (int(slot), float(values["raw_gap_missing_byte_exposure"]))
        for slot, values in macro_summaries.items()
        if float(values.get("gap_demand_weight_exposure", 0.0)) > 0.0
    ]
    maximum_raw_exposure = max(
        (exposure for _, exposure in active_gap_slots),
        default=0.0,
    )
    overall["gap_demand_series"] = [
        {
            "macro_slot": slot,
            "raw_missing_byte_exposure": exposure,
            "normalized_residual": (
                exposure / maximum_raw_exposure if maximum_raw_exposure > 0.0 else 0.0
            ),
        }
        for slot, exposure in active_gap_slots
    ]
    overall["selected_macro_slots"] = select_macro_slots(macro_summaries, count=6)
    overall["rollout_metrics"] = dict(payload.get("rollout_metrics", {}))
    overall["provenance"] = dict(payload.get("provenance", {}))
    return overall


def cohort_gap_demand_statistics(
    summaries: list[Mapping[str, Any]],
    *,
    macro_slot_count: int,
) -> dict[str, Any]:
    """Return seed-level normalized mean curves and two-sided 95% t intervals."""

    count = int(macro_slot_count)
    if count <= 0:
        raise ValueError("macro-slot count must be positive")
    if not summaries:
        raise ValueError("at least one trace summary is required")
    curves: list[np.ndarray] = []
    for summary in summaries:
        macro_slots = summary.get("macro_slots", {})
        raw = np.asarray(
            [
                float(
                    macro_slots.get(str(slot), {}).get(
                        "raw_gap_missing_byte_exposure",
                        0.0,
                    )
                )
                for slot in range(1, count + 1)
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(raw)) or np.any(raw < 0.0):
            raise ValueError("raw gap-demand exposure must be finite and nonnegative")
        maximum = float(raw.max())
        curves.append(raw / maximum if maximum > 0.0 else np.zeros_like(raw))
    matrix = np.vstack(curves)
    mean = matrix.mean(axis=0)
    if len(curves) > 1:
        standard_error = matrix.std(axis=0, ddof=1) / math.sqrt(len(curves))
        critical = float(stats.t.ppf(0.975, df=len(curves) - 1))
        half_width = critical * standard_error
    else:
        half_width = np.zeros_like(mean)
    lower = np.clip(mean - half_width, 0.0, 1.0)
    upper = np.clip(mean + half_width, 0.0, 1.0)
    exposure_auc = matrix.sum(axis=1)
    exposure_auc_mean = float(exposure_auc.mean())
    if len(curves) > 1:
        exposure_auc_half_width = float(
            critical * exposure_auc.std(ddof=1) / math.sqrt(len(curves))
        )
    else:
        exposure_auc_half_width = 0.0
    return {
        "n": len(curves),
        "confidence_level": 0.95,
        "normalization": "within-seed active-period maximum",
        "normalized_exposure_auc_mean": exposure_auc_mean,
        "normalized_exposure_auc_ci": [
            max(0.0, exposure_auc_mean - exposure_auc_half_width),
            min(float(count), exposure_auc_mean + exposure_auc_half_width),
        ],
        "points": [
            {
                "macro_slot": slot,
                "mean": float(mean[slot - 1]),
                "ci_lower": float(lower[slot - 1]),
                "ci_upper": float(upper[slot - 1]),
            }
            for slot in range(1, count + 1)
        ],
    }


def weighted_kde_grid(
    points: list[Mapping[str, Any]],
    *,
    weight_key: str,
    width_m: float,
    height_m: float,
    bandwidth_m: float,
    grid_size: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a fixed-bandwidth weighted Gaussian KDE on the scenario grid."""

    width = float(width_m)
    height = float(height_m)
    bandwidth = float(bandwidth_m)
    size = int(grid_size)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("KDE spatial bounds must be positive")
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("KDE bandwidth must be finite and positive")
    if size < 3:
        raise ValueError("KDE grid size must be at least three")
    coordinates_x = np.linspace(0.0, width, size)
    coordinates_y = np.linspace(0.0, height, size)
    x_grid, y_grid = np.meshgrid(coordinates_x, coordinates_y)
    density = np.zeros_like(x_grid, dtype=float)
    inverse_two_variance = 1.0 / (2.0 * bandwidth * bandwidth)
    for point in points:
        weight = float(point.get(weight_key, 0.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("KDE weights must be finite and nonnegative")
        if weight == 0.0:
            continue
        x_m = float(point["x_m"])
        y_m = float(point["y_m"])
        density += weight * np.exp(
            -((x_grid - x_m) ** 2 + (y_grid - y_m) ** 2)
            * inverse_two_variance
        )
    return x_grid, y_grid, density
