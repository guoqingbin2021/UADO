from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .queues import ComputeItem, TransferItem


def restart_transfer(
    item: TransferItem,
    *,
    new_route: tuple[str, ...],
    reason: str,
) -> TransferItem:
    if reason not in {"target_changed", "cache_invalidated"}:
        raise ValueError("restart requires a target change or cache invalidation")
    if len(new_route) < 2:
        raise ValueError("restart route must contain at least two nodes")
    item.wasted_bytes += item.total_bytes - item.remaining_bytes
    item.remaining_bytes = float(item.total_bytes)
    item.route = tuple(new_route)
    item.hop_index = 0
    item.source = item.route[0]
    item.destination = item.route[1]
    item.final_destination = item.route[-1]
    item.attempt += 1
    item.is_paused = False
    item.item_id = (
        f"{item.workflow_id}/{item.task_id}/{item.file_id or 'aggregate'}/"
        f"{item.direction}/hop-0/attempt-{item.attempt}"
    )
    return item


def advance_transfer(
    item: TransferItem,
    *,
    allocated_rate_bps: float,
    dt_s: float,
    connected: bool,
) -> float:
    dt = float(dt_s)
    rate = float(allocated_rate_bps)
    if dt < 0 or rate < 0:
        raise ValueError("time and allocated rate must be non-negative")
    if item.complete or dt == 0:
        return 0.0
    if not connected:
        item.paused_s += dt
        item.is_paused = True
        return 0.0
    if item.is_paused:
        item.resume_count += 1
        item.is_paused = False
    transferable_bytes = rate * dt / 8.0
    transferred = min(item.remaining_bytes, transferable_bytes)
    item.remaining_bytes = max(0.0, item.remaining_bytes - transferred)
    return transferred


def advance_compute(
    item: ComputeItem,
    *,
    allocated_cycles_per_s: float,
    dt_s: float,
) -> float:
    dt = float(dt_s)
    rate = float(allocated_cycles_per_s)
    if dt < 0 or rate < 0:
        raise ValueError("time and allocated compute rate must be non-negative")
    processed = min(item.remaining_cycles, rate * dt)
    item.remaining_cycles = max(0.0, item.remaining_cycles - processed)
    return processed


@dataclass(frozen=True, slots=True)
class ContactForecast:
    predicted_distance_m: float
    horizon_s: float
    history_points: int


class CausalContactPredictor:
    def __init__(self, *, history_length: int = 5) -> None:
        if history_length < 2:
            raise ValueError("contact predictor requires at least two history samples")
        self.history_length = int(history_length)

    def forecast(
        self,
        position_history_xy: Sequence[tuple[float, float]],
        *,
        infrastructure_xy: tuple[float, float],
        horizon_s: float,
        sample_interval_s: float,
        infrastructure_history_xy: Sequence[tuple[float, float]] | None = None,
    ) -> ContactForecast:
        history = tuple(position_history_xy[-self.history_length :])
        if len(history) < 2:
            raise ValueError("insufficient causal history")
        horizon = float(horizon_s)
        if horizon < 0:
            raise ValueError("forecast horizon cannot be negative")
        sample_interval = float(sample_interval_s)
        if not math.isfinite(sample_interval) or sample_interval <= 0:
            raise ValueError(
                "contact history sample interval must be finite and positive"
            )
        velocity_x = (history[-1][0] - history[-2][0]) / sample_interval
        velocity_y = (history[-1][1] - history[-2][1]) / sample_interval
        infrastructure_history = tuple(
            (infrastructure_history_xy or ())[-self.history_length :]
        )
        infrastructure_velocity_x = 0.0
        infrastructure_velocity_y = 0.0
        if len(infrastructure_history) >= 2:
            infrastructure_velocity_x = (
                infrastructure_history[-1][0] - infrastructure_history[-2][0]
            ) / sample_interval
            infrastructure_velocity_y = (
                infrastructure_history[-1][1] - infrastructure_history[-2][1]
            ) / sample_interval
        predicted_position = (
            history[-1][0] + velocity_x * horizon,
            history[-1][1] + velocity_y * horizon,
        )
        predicted_infrastructure = (
            infrastructure_xy[0] + infrastructure_velocity_x * horizon,
            infrastructure_xy[1] + infrastructure_velocity_y * horizon,
        )
        return ContactForecast(
            predicted_distance_m=math.dist(
                predicted_position,
                predicted_infrastructure,
            ),
            horizon_s=horizon,
            history_points=len(history),
        )

    def forecast_distance(
        self,
        position_history_xy: Sequence[tuple[float, float]],
        *,
        infrastructure_xy: tuple[float, float],
        horizon_s: float,
        sample_interval_s: float,
        infrastructure_history_xy: Sequence[tuple[float, float]] | None = None,
    ) -> float:
        return self.forecast(
            position_history_xy,
            infrastructure_xy=infrastructure_xy,
            horizon_s=horizon_s,
            sample_interval_s=sample_interval_s,
            infrastructure_history_xy=infrastructure_history_xy,
        ).predicted_distance_m

    def forecast_contact_window_s(
        self,
        position_history_xy: Sequence[tuple[float, float]],
        *,
        infrastructure_xy: tuple[float, float],
        radius_m: float,
        horizon_s: float,
        sample_interval_s: float,
        infrastructure_history_xy: Sequence[tuple[float, float]] | None = None,
    ) -> float:
        """Return the first predicted link-exit time under causal constant velocity."""
        history = tuple(position_history_xy[-self.history_length :])
        if len(history) < 2:
            raise ValueError("insufficient causal history")
        radius = float(radius_m)
        horizon = float(horizon_s)
        interval = float(sample_interval_s)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("contact radius must be finite and positive")
        if not math.isfinite(horizon) or horizon < 0:
            raise ValueError("forecast horizon must be finite and non-negative")
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError(
                "contact history sample interval must be finite and positive"
            )
        relative_x = history[-1][0] - infrastructure_xy[0]
        relative_y = history[-1][1] - infrastructure_xy[1]
        if math.hypot(relative_x, relative_y) > radius:
            return 0.0
        infrastructure_history = tuple(
            (infrastructure_history_xy or ())[-self.history_length :]
        )
        infrastructure_velocity_x = 0.0
        infrastructure_velocity_y = 0.0
        if len(infrastructure_history) >= 2:
            infrastructure_velocity_x = (
                infrastructure_history[-1][0] - infrastructure_history[-2][0]
            ) / interval
            infrastructure_velocity_y = (
                infrastructure_history[-1][1] - infrastructure_history[-2][1]
            ) / interval
        velocity_x = (
            (history[-1][0] - history[-2][0]) / interval
            - infrastructure_velocity_x
        )
        velocity_y = (
            (history[-1][1] - history[-2][1]) / interval
            - infrastructure_velocity_y
        )
        speed_squared = velocity_x * velocity_x + velocity_y * velocity_y
        if speed_squared <= 1.0e-12:
            return horizon
        linear = 2.0 * (
            relative_x * velocity_x + relative_y * velocity_y
        )
        constant = relative_x * relative_x + relative_y * relative_y - radius * radius
        discriminant = linear * linear - 4.0 * speed_squared * constant
        if discriminant < 0.0:
            return horizon
        exit_time = (
            -linear + math.sqrt(max(0.0, discriminant))
        ) / (2.0 * speed_squared)
        if exit_time < 0.0:
            return 0.0
        return min(horizon, exit_time)
