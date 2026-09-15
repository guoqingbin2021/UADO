from __future__ import annotations

from .api import SOTAPhysicalContext


class FixedCentroidMacroPolicy:
    """Non-learning support rule shared only by the three domain SOTA."""

    def targets(self, physical_context: SOTAPhysicalContext) -> tuple[tuple[float, float], ...]:
        active = [
            position
            for position, unfinished in zip(
                physical_context.owner_positions_xy_m,
                physical_context.unfinished_owner_mask,
            )
            if unfinished
        ]
        if not active or not physical_context.uav_ids:
            return ()
        centroid = (
            sum(point[0] for point in active) / len(active),
            sum(point[1] for point in active) / len(active),
        )
        return tuple(centroid for _ in physical_context.uav_ids)
