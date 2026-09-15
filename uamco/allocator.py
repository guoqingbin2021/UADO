from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import groupby


SLA_PRIORITY = {"gold": 0, "silver": 1, "bronze": 2}


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    request_id: str
    sla_tier: str
    deadline_s: float
    demand: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.sla_tier.lower() not in SLA_PRIORITY:
            raise ValueError(f"unknown SLA tier: {self.sla_tier}")
        if not math.isfinite(self.deadline_s):
            raise ValueError("deadline must be finite")
        if self.demand < 0 or self.weight <= 0:
            raise ValueError("resource demand and weight are invalid")


def _weighted_fair_share(
    requests: tuple[ResourceRequest, ...],
    capacity: float,
) -> dict[str, float]:
    allocation = {request.request_id: 0.0 for request in requests}
    active = list(requests)
    remaining = float(capacity)
    while active and remaining > 1e-12:
        total_weight = sum(request.weight for request in active)
        saturated: list[ResourceRequest] = []
        provisional = {
            request.request_id: remaining * request.weight / total_weight for request in active
        }
        for request in active:
            unmet = request.demand - allocation[request.request_id]
            if provisional[request.request_id] >= unmet - 1e-12:
                allocation[request.request_id] += max(0.0, unmet)
                remaining -= max(0.0, unmet)
                saturated.append(request)
        if not saturated:
            for request in active:
                share = provisional[request.request_id]
                allocation[request.request_id] += share
            remaining = 0.0
        else:
            active = [request for request in active if request not in saturated]
    return allocation


class DeterministicAllocator:
    def allocate(
        self,
        requests: list[ResourceRequest] | tuple[ResourceRequest, ...],
        *,
        capacity: float,
    ) -> dict[str, float]:
        available = float(capacity)
        if available < 0 or not math.isfinite(available):
            raise ValueError("resource capacity must be finite and non-negative")
        if len({request.request_id for request in requests}) != len(requests):
            raise ValueError("resource request identifiers must be unique")
        allocation = {request.request_id: 0.0 for request in requests}
        ordered = sorted(
            requests,
            key=lambda request: (
                SLA_PRIORITY[request.sla_tier.lower()],
                request.deadline_s,
                request.request_id,
            ),
        )
        for _, group_iterator in groupby(
            ordered,
            key=lambda request: (
                SLA_PRIORITY[request.sla_tier.lower()],
                request.deadline_s,
            ),
        ):
            group = tuple(group_iterator)
            if available <= 1e-12:
                break
            group_allocation = _weighted_fair_share(group, available)
            consumed = sum(group_allocation.values())
            allocation.update(group_allocation)
            available = max(0.0, available - consumed)
        return allocation


class TierAgnosticAllocator:
    """Ablation allocator: weighted proportional fairness without SLA or EDF ordering."""

    def allocate(
        self,
        requests: list[ResourceRequest] | tuple[ResourceRequest, ...],
        *,
        capacity: float,
    ) -> dict[str, float]:
        available = float(capacity)
        if available < 0 or not math.isfinite(available):
            raise ValueError("resource capacity must be finite and non-negative")
        if len({request.request_id for request in requests}) != len(requests):
            raise ValueError("resource request identifiers must be unique")
        return _weighted_fair_share(tuple(requests), available)
