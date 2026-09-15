"""Deterministic selection of one task candidate for each UGV."""

from __future__ import annotations

from dataclasses import dataclass

from .env import SLA_TIERS, UAMCOEnv


@dataclass(frozen=True, slots=True)
class CandidateTask:
    owner_ugv: str
    workflow_id: str
    task_id: str


def select_fixed_candidate(env: UAMCOEnv, owner_ugv: str) -> CandidateTask | None:
    """Return the highest-priority precedence-ready task owned by ``owner_ugv``."""
    candidates = []
    for workflow_id, state in env.workflow_states.items():
        if state.status != "active" or state.owner_ugv != owner_ugv:
            continue
        for task_id in env.precedence_ready_task_ids(workflow_id):
            candidates.append((workflow_id, task_id, state))
    if not candidates:
        return None
    workflow_id, task_id, _ = min(
        candidates,
        key=lambda item: (
            SLA_TIERS.index(item[2].sla_tier),
            item[2].deadline_time_s,
            -item[2].instance.remaining_critical_path_s(item[1]),
            item[0],
            item[1],
        ),
    )
    return CandidateTask(owner_ugv, workflow_id, task_id)
