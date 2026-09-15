from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence


BASELINE_NAMES = ("MAPPO", "HAPPO", "AMCoEdge", "FDEdge", "MEC-UARA")


@dataclass(frozen=True, slots=True)
class SharedBenchmarkContext:
    seed: int
    training_steps: int
    allocator_id: str
    executors: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.training_steps <= 0:
            raise ValueError("training budget must be positive")
        if self.allocator_id != "sla_edf_pf_v1":
            raise ValueError("all formal baselines must use the shared SLA-aware EDF+PF allocator")
        if len(self.executors) < 2 or len(set(self.executors)) != len(self.executors):
            raise ValueError("executor vocabulary must contain unique actions")


@dataclass(frozen=True, slots=True)
class ReadyNodeInput:
    node_key: str
    features: tuple[float, ...]
    legal_actions: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.features or not all(math.isfinite(value) for value in self.features):
            raise ValueError("ready-node features must be finite and non-empty")
        if not any(self.legal_actions):
            raise ValueError("a ready node must have at least one legal action")


@dataclass(frozen=True, slots=True)
class BaselineDecision:
    method: str
    executor_by_node: Mapping[str, str]
    allocator_id: str
    seed: int


class UpstreamScoringPolicy(Protocol):
    def score(self, method: str, nodes: Sequence[ReadyNodeInput]) -> Mapping[str, Sequence[float]]: ...


class FixedScorePolicy:
    """Deterministic test/dry-run policy; never used as a formal baseline fallback."""

    def __init__(self, scores: Mapping[str, Sequence[float]]) -> None:
        self._scores = {key: tuple(map(float, values)) for key, values in scores.items()}

    def score(self, method: str, nodes: Sequence[ReadyNodeInput]) -> Mapping[str, Sequence[float]]:
        del method
        return {node.node_key: self._scores[node.node_key] for node in nodes}


class BaselineAdapter:
    def __init__(
        self,
        name: str,
        context: SharedBenchmarkContext,
        policy: UpstreamScoringPolicy | None = None,
    ) -> None:
        if name not in BASELINE_NAMES:
            raise ValueError(f"unsupported formal baseline: {name}")
        self.name = name
        self.context = context
        self.policy = policy

    def select_ready_nodes(self, nodes: Sequence[ReadyNodeInput]) -> BaselineDecision:
        if self.policy is None:
            raise RuntimeError(
                f"{self.name} upstream policy is not loaded; refusing to substitute a heuristic"
            )
        raw_scores = self.policy.score(self.name, nodes)
        decisions: dict[str, str] = {}
        for node in nodes:
            if len(node.legal_actions) != len(self.context.executors):
                raise ValueError("ready-node mask does not match the shared executor vocabulary")
            scores = tuple(float(value) for value in raw_scores[node.node_key])
            if len(scores) != len(self.context.executors) or not all(math.isfinite(value) for value in scores):
                raise ValueError("upstream policy returned invalid action scores")
            masked = [score if allowed else -math.inf for score, allowed in zip(scores, node.legal_actions)]
            choice = max(range(len(masked)), key=lambda index: (masked[index], -index))
            decisions[node.node_key] = self.context.executors[choice]
        return BaselineDecision(
            method=self.name,
            executor_by_node=decisions,
            allocator_id=self.context.allocator_id,
            seed=self.context.seed,
        )


def make_baseline(
    name: str,
    context: SharedBenchmarkContext,
    *,
    policy: UpstreamScoringPolicy | None = None,
) -> BaselineAdapter:
    return BaselineAdapter(name, context, policy)


def make_all_baselines(
    context: SharedBenchmarkContext,
    *,
    policy_factory: Callable[[str], UpstreamScoringPolicy],
) -> tuple[BaselineAdapter, ...]:
    return tuple(
        make_baseline(name, context, policy=policy_factory(name)) for name in BASELINE_NAMES
    )
