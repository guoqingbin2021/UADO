from __future__ import annotations

import math
from typing import Sequence

from .types import MobilityTrace, ScenarioLayout


def rigid_transform_trace(
    trace: MobilityTrace,
    *,
    rotation_rad: float,
    translation_xy_m: tuple[float, float],
    trace_id: str | None = None,
) -> MobilityTrace:
    cosine = math.cos(float(rotation_rad))
    sine = math.sin(float(rotation_rad))
    translate_x, translate_y = map(float, translation_xy_m)
    positions = tuple(
        (
            cosine * x - sine * y + translate_x,
            sine * x + cosine * y + translate_y,
        )
        for x, y in trace.positions_xy_m
    )
    return MobilityTrace(
        trace_id=trace_id or trace.trace_id,
        source=trace.source,
        split=trace.split,
        timestamps_s=trace.timestamps_s,
        positions_xy_m=positions,
        provenance={
            **dict(trace.provenance),
            "rigid_rotation_rad": f"{float(rotation_rad):.17g}",
            "rigid_translation_xy_m": f"{translate_x:.17g},{translate_y:.17g}",
        },
    )


def distance_trigger_indices(
    trace: MobilityTrace,
    *,
    trigger_distance_m: float,
) -> tuple[int, ...]:
    threshold = float(trigger_distance_m)
    if threshold <= 0:
        raise ValueError("trigger distance must be positive")
    accumulated = 0.0
    indices: list[int] = []
    previous = trace.positions_xy_m[0]
    for index, current in enumerate(trace.positions_xy_m[1:], 1):
        accumulated += math.dist(previous, current)
        if accumulated + 1e-12 >= threshold:
            indices.append(index)
            accumulated = 0.0
        previous = current
    return tuple(indices)


def build_multizone_layout(
    traces: Sequence[MobilityTrace],
    *,
    rsu_positions_xy_m: Sequence[tuple[float, float]],
    uav_initial_positions_xyz_m: Sequence[tuple[float, float, float]],
    width_m: float,
    height_m: float,
) -> ScenarioLayout:
    if not traces:
        raise ValueError("layout requires at least one real trace")
    if not rsu_positions_xy_m or not uav_initial_positions_xyz_m:
        raise ValueError("layout requires explicit RSU and UAV positions")
    width = float(width_m)
    height = float(height_m)
    if width <= 0 or height <= 0:
        raise ValueError("layout dimensions must be positive")

    def inside_xy(position: tuple[float, float]) -> bool:
        return 0 <= position[0] <= width and 0 <= position[1] <= height

    if any(not inside_xy(tuple(map(float, position))) for position in rsu_positions_xy_m):
        raise ValueError("RSU position lies outside the configured area")
    if any(
        not inside_xy((float(position[0]), float(position[1]))) or float(position[2]) <= 0
        for position in uav_initial_positions_xyz_m
    ):
        raise ValueError("UAV position or altitude is invalid")
    return ScenarioLayout(
        traces=tuple(traces),
        rsu_positions_xy_m=tuple(tuple(map(float, position)) for position in rsu_positions_xy_m),
        uav_initial_positions_xyz_m=tuple(
            tuple(map(float, position)) for position in uav_initial_positions_xyz_m
        ),
        width_m=width,
        height_m=height,
    )

