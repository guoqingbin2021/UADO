from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class SOTAObservation:
    candidate_task_id: str
    owner_index: int
    input_bytes: float
    cycles: float
    executor_ids: tuple[str, ...]
    executor_queue_work: tuple[float, ...]
    executor_cpu_hz: tuple[float, ...]
    link_rate_bps: tuple[float, ...]
    action_mask: tuple[bool, ...]
    remaining_energy_j: tuple[float, ...]
    micro_slot: int
    absorbing: bool = False

    def __post_init__(self) -> None:
        count = len(self.executor_ids)
        fields = (
            self.executor_queue_work,
            self.executor_cpu_hz,
            self.link_rate_bps,
            self.action_mask,
            self.remaining_energy_j,
        )
        if count < 2 or any(len(values) != count for values in fields):
            raise ValueError("SOTA executor fields must have one common nontrivial length")
        if not any(self.action_mask):
            raise ValueError("SOTA observation has no legal action")
        if self.owner_index < 0 or self.micro_slot < 0:
            raise ValueError("SOTA owner and micro-slot indices must be nonnegative")
        if self.input_bytes < 0 or self.cycles < 0:
            raise ValueError("SOTA task workload cannot be negative")
        if any(value < 0 for value in self.executor_queue_work):
            raise ValueError("SOTA queue work cannot be negative")
        if any(value < 0 for value in self.executor_cpu_hz):
            raise ValueError("SOTA executor capacity cannot be negative")
        if any(value < 0 for value in self.link_rate_bps):
            raise ValueError("SOTA link rate cannot be negative")
        if any(value < 0 for value in self.remaining_energy_j):
            raise ValueError("SOTA remaining energy cannot be negative")

    @property
    def action_count(self) -> int:
        return len(self.executor_ids)


@dataclass(frozen=True, slots=True)
class SOTAPhysicalContext:
    bandwidth_hz: float
    owner_positions_xy_m: tuple[tuple[float, float], ...] = ()
    unfinished_owner_mask: tuple[bool, ...] = ()
    uav_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.bandwidth_hz <= 0 or not math.isfinite(self.bandwidth_hz):
            raise ValueError("SOTA physical bandwidth must be finite and positive")
        if len(self.owner_positions_xy_m) != len(self.unfinished_owner_mask):
            raise ValueError("SOTA owner positions and activity mask differ in length")


@dataclass(frozen=True, slots=True)
class SOTAAction:
    executor_index: int
    scores: tuple[float, ...]
    macro_targets_xy_m: tuple[tuple[float, float], ...] = ()
    selected_subset: tuple[int, ...] = ()
    allocation_fractions: tuple[float, ...] = ()
    latent_action_probabilities: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.executor_index < len(self.scores):
            raise ValueError("SOTA executor action is outside its score vocabulary")
        if not math.isfinite(self.scores[self.executor_index]):
            raise ValueError("SOTA selected an invalid masked score")
        if self.selected_subset and any(
            not 0 <= index < len(self.scores) for index in self.selected_subset
        ):
            raise ValueError("SOTA subset contains an executor outside the vocabulary")
        if self.allocation_fractions:
            if len(self.allocation_fractions) != len(self.scores):
                raise ValueError("SOTA allocation fractions differ from executor vocabulary")
            if any(value < 0 or not math.isfinite(value) for value in self.allocation_fractions):
                raise ValueError("SOTA allocation fractions must be finite and nonnegative")
            if not math.isclose(sum(self.allocation_fractions), 1.0, abs_tol=1.0e-9):
                raise ValueError("SOTA allocation fractions must sum to one")
        if self.latent_action_probabilities and len(self.latent_action_probabilities) != len(
            self.scores
        ):
            raise ValueError("SOTA latent feedback differs from executor vocabulary")


@dataclass(frozen=True, slots=True)
class SOTATransition:
    observation: SOTAObservation
    action: SOTAAction
    reward: float
    next_observation: SOTAObservation
    done: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.reward):
            raise ValueError("SOTA transition reward must be finite")
        if self.observation.action_count != self.next_observation.action_count:
            raise ValueError("SOTA transition changed the executor vocabulary")
        if not self.observation.action_mask[self.action.executor_index]:
            raise ValueError("SOTA transition contains an illegal action")


@runtime_checkable
class SOTARuntime(Protocol):
    method_name: str

    def reset_episode(self, seed: int) -> None: ...

    def act(
        self,
        observation: SOTAObservation,
        physical_context: SOTAPhysicalContext,
        *,
        deterministic: bool,
    ) -> SOTAAction: ...

    def observe(self, transition: SOTATransition) -> None: ...

    def update_if_ready(self) -> Mapping[str, float]: ...

    def tick_micro_slot(self, micro_slot: int) -> Mapping[str, float]: ...

    def state_dict(self) -> Mapping[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


def encode_observation(observation: SOTAObservation) -> np.ndarray:
    """Stable common state vector; each method retains its own learning core."""
    workload = [
        math.log1p(observation.input_bytes) / 30.0,
        math.log1p(observation.cycles) / 30.0,
        float(observation.owner_index),
        float(observation.micro_slot) / 10_000.0,
    ]
    executor_features: list[float] = []
    for queue, cpu, rate, energy, legal in zip(
        observation.executor_queue_work,
        observation.executor_cpu_hz,
        observation.link_rate_bps,
        observation.remaining_energy_j,
        observation.action_mask,
    ):
        executor_features.extend(
            (
                math.log1p(queue) / 30.0,
                math.log1p(cpu) / 30.0,
                2.0 if math.isinf(rate) else math.log1p(rate) / 30.0,
                2.0 if math.isinf(energy) else math.log1p(energy) / 20.0,
                float(legal),
            )
        )
    return np.asarray((*workload, *executor_features), dtype=np.float32)


def _contained_tensors(value):
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, nn.Module):
        yield from value.parameters()
        yield from value.buffers()
    elif isinstance(value, torch.optim.Optimizer):
        for state in value.state.values():
            yield from _contained_tensors(state)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _contained_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _contained_tensors(item)


def all_finite(*values) -> bool:
    return all(
        bool(torch.isfinite(tensor).all())
        for value in values
        for tensor in _contained_tensors(value)
    )


def clip_grad_norm_if_finite(
    parameters: Iterable[Tensor], max_grad_norm: float
) -> float | None:
    active = tuple(parameter for parameter in parameters if parameter.grad is not None)
    if not active:
        return 0.0
    total = nn.utils.clip_grad_norm_(
        active,
        float(max_grad_norm),
        error_if_nonfinite=False,
    )
    value = float(total.detach().cpu())
    if not math.isfinite(value):
        for parameter in active:
            parameter.grad = None
        return None
    return value


def hard_mask_scores(scores: np.ndarray, mask: tuple[bool, ...]) -> np.ndarray:
    result = np.asarray(scores, dtype=np.float64).copy()
    legal = np.asarray(mask, dtype=bool)
    if result.shape != (len(mask),):
        raise ValueError("SOTA score vector does not match executor vocabulary")
    if not bool(legal.any()):
        raise ValueError("SOTA action mask contains no legal action")
    result[~legal] = -math.inf
    if not bool(np.isfinite(result[legal]).any()):
        raise FloatingPointError("SOTA legal action scores are all non-finite")
    return result
