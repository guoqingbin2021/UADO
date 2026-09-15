from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .types import MobilityTrace


RELLIS_SOURCE = "RELLIS-3D"
M2DGR_SOURCE = "M2DGR-Outdoor"
MAX_TRAJECTORY_LINE_CHARACTERS = 1_000_000


def _numeric_rows(path: str | Path) -> list[list[float]]:
    rows: list[list[float]] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if (
                "\x00" in raw_line
                or len(raw_line) > MAX_TRAJECTORY_LINE_CHARACTERS
            ):
                raise ValueError(
                    f"non-numeric trajectory value at {path}:{line_number}"
                ) from None
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append([float(value) for value in line.replace(",", " ").split()])
            except ValueError:
                raise ValueError(
                    f"non-numeric trajectory value at {path}:{line_number}"
                ) from None
    return rows


def _validate_trace(trace: MobilityTrace) -> MobilityTrace:
    if len(trace.timestamps_s) != len(trace.positions_xy_m) or len(trace.timestamps_s) < 2:
        raise ValueError("trajectory must contain at least two timestamp-aligned poses")
    if any(not math.isfinite(value) for value in trace.timestamps_s):
        raise ValueError("trajectory timestamps must be finite")
    if any(later <= earlier for earlier, later in zip(trace.timestamps_s, trace.timestamps_s[1:])):
        raise ValueError("trajectory timestamps must be strictly increasing")
    if any(not math.isfinite(value) for position in trace.positions_xy_m for value in position):
        raise ValueError("trajectory positions must be finite")
    return trace


def mirror_extend_measured_trace(
    trace: MobilityTrace,
    *,
    minimum_duration_s: float,
) -> MobilityTrace:
    """Continuously extend a short RELLIS trace using measured poses only.

    The pose order is reflected at each endpoint (forward, backward, forward)
    so the extension has no spatial jump and introduces no interpolated pose.
    It is used only to support a fixed observation horizon longer than one
    reserved RELLIS sequence; M2DGR zero-shot traces are never extended.
    """
    horizon = float(minimum_duration_s)
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("minimum trajectory duration must be finite and positive")
    _validate_trace(trace)
    if trace.source != RELLIS_SOURCE:
        raise ValueError("only RELLIS measured traces may use mirrored extension")
    original_duration = trace.timestamps_s[-1] - trace.timestamps_s[0]
    if original_duration >= horizon:
        return trace

    timestamps = list(trace.timestamps_s)
    positions = list(trace.positions_xy_m)
    current_index = len(trace.timestamps_s) - 1
    direction = -1
    while timestamps[-1] - timestamps[0] < horizon:
        next_index = current_index + direction
        if next_index < 0 or next_index >= len(trace.timestamps_s):
            direction *= -1
            next_index = current_index + direction
        delta_s = abs(
            trace.timestamps_s[next_index]
            - trace.timestamps_s[current_index]
        )
        if not math.isfinite(delta_s) or delta_s <= 0.0:
            raise ValueError("measured trajectory contains an invalid sample interval")
        timestamps.append(timestamps[-1] + delta_s)
        positions.append(trace.positions_xy_m[next_index])
        current_index = next_index

    return _validate_trace(
        MobilityTrace(
            trace_id=trace.trace_id,
            source=trace.source,
            split=trace.split,
            timestamps_s=tuple(timestamps),
            positions_xy_m=tuple(positions),
            provenance={
                **dict(trace.provenance),
                "horizon_extension": "measured_pose_mirror",
                "original_duration_s": f"{original_duration:.9g}",
                "extended_duration_s": f"{timestamps[-1] - timestamps[0]:.9g}",
            },
        )
    )


def load_rellis_pose_trace(
    poses_path: str | Path,
    *,
    trace_id: str,
    split: str,
    sampling_hz: float,
) -> MobilityTrace:
    if not math.isfinite(sampling_hz) or sampling_hz <= 0.0:
        raise ValueError("RELLIS sampling_hz must be finite and positive")
    pose_rows = _numeric_rows(poses_path)
    if any(len(row) not in (12, 16) for row in pose_rows):
        raise ValueError("RELLIS poses must be flattened 3x4 or 4x4 matrices")
    positions = tuple((row[3], row[7]) for row in pose_rows)
    timestamps = tuple(index / sampling_hz for index in range(len(pose_rows)))
    return _validate_trace(
        MobilityTrace(
            trace_id=str(trace_id),
            source=RELLIS_SOURCE,
            split=split,
            timestamps_s=timestamps,
            positions_xy_m=positions,
            provenance={
                "poses": str(Path(poses_path).resolve()),
                "fields": "pose",
                "time_axis": "sensor_rate",
                "sampling_hz": float(sampling_hz),
            },
        )
    )


def load_m2dgr_ground_truth(
    ground_truth_path: str | Path,
    *,
    trace_id: str,
) -> MobilityTrace:
    rows = _numeric_rows(ground_truth_path)
    if any(len(row) != 8 for row in rows):
        raise ValueError(
            "M2DGR ground truth must contain exactly 8 TUM fields: "
            "timestamp tx ty tz qx qy qz qw"
        )
    return _validate_trace(
        MobilityTrace(
            trace_id=str(trace_id),
            source=M2DGR_SOURCE,
            split="zero_shot",
            timestamps_s=tuple(row[0] for row in rows),
            positions_xy_m=tuple((row[1], row[2]) for row in rows),
            provenance={
                "ground_truth": str(Path(ground_truth_path).resolve()),
                "fields": "timestamp,RTK/INS pose",
            },
        )
    )


def assert_training_trace(trace: MobilityTrace) -> None:
    if trace.source == M2DGR_SOURCE or trace.split == "zero_shot":
        raise ValueError("M2DGR Outdoor is reserved for zero-shot evaluation")
    if trace.source != RELLIS_SOURCE or trace.split not in {"train", "validation"}:
        raise ValueError("training accepts only RELLIS-3D train/validation traces")


def validate_rellis_splits(splits: Mapping[str, Sequence[str]]) -> None:
    expected = {"calibration", "train", "validation", "test"}
    if set(splits) != expected:
        raise ValueError(f"RELLIS split manifest must contain exactly {sorted(expected)}")
    owners: dict[str, str] = {}
    for split, sequence_ids in splits.items():
        for sequence_id in sequence_ids:
            if sequence_id in owners:
                raise ValueError(
                    f"RELLIS split leakage: sequence {sequence_id} appears in {owners[sequence_id]} and {split}"
                )
            owners[sequence_id] = split
    if not all(splits[split] for split in expected):
        raise ValueError("every RELLIS split must contain at least one sequence")


def load_rellis_traces(
    dataset_root: str | Path,
    split_manifest: Mapping[str, Sequence[str]],
    *,
    sampling_hz: float,
) -> dict[str, tuple[MobilityTrace, ...]]:
    validate_rellis_splits(split_manifest)
    root = Path(dataset_root)
    loaded: dict[str, tuple[MobilityTrace, ...]] = {}
    for split, sequence_ids in split_manifest.items():
        loaded[split] = tuple(
            load_rellis_pose_trace(
                root / sequence_id / "poses.txt",
                trace_id=sequence_id,
                split=split,
                sampling_hz=sampling_hz,
            )
            for sequence_id in sequence_ids
        )
    return loaded


def load_m2dgr_outdoor_traces(
    dataset_root: str | Path,
    sequence_ids: Iterable[str],
) -> tuple[MobilityTrace, ...]:
    root = Path(dataset_root)
    return tuple(
        load_m2dgr_ground_truth(root / sequence_id / "ground_truth.txt", trace_id=sequence_id)
        for sequence_id in sequence_ids
    )


def load_mobility_manifest(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_rellis_splits(payload["rellis3d"]["splits"])
    if payload["m2dgr_outdoor"].get("training_allowed", True):
        raise ValueError("mobility manifest must reserve M2DGR Outdoor for zero-shot evaluation")
    return payload
