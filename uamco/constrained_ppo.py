from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


def compute_gae(
    signals: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    if signals.ndim != 1 or dones.shape != signals.shape or values.shape != (signals.shape[0] + 1,):
        raise ValueError("GAE expects signals/dones [T] and values [T+1]")
    advantages = torch.zeros_like(signals)
    running = torch.zeros((), dtype=signals.dtype, device=signals.device)
    for index in range(signals.shape[0] - 1, -1, -1):
        not_done = 1.0 - dones[index]
        delta = signals[index] + gamma * values[index + 1] * not_done - values[index]
        running = delta + gamma * gae_lambda * not_done * running
        advantages[index] = running
    return advantages, advantages + values[:-1]


class LagrangeController:
    def __init__(
        self,
        *,
        limits: Mapping[str, float],
        learning_rate: float,
        initial_value: float = 0.0,
    ) -> None:
        if learning_rate <= 0 or initial_value < 0:
            raise ValueError("invalid Lagrange controller parameters")
        if not limits or any(limit < 0 for limit in limits.values()):
            raise ValueError("cost limits must be non-negative and non-empty")
        self.limits = {key: float(limit) for key, limit in limits.items()}
        self.learning_rate = float(learning_rate)
        self._multipliers = {key: float(initial_value) for key in self.limits}

    def update(self, observed_costs: Mapping[str, float]) -> None:
        if set(observed_costs) != set(self.limits):
            raise ValueError("observed cost keys must match configured limits")
        for key, limit in self.limits.items():
            candidate = self._multipliers[key] + self.learning_rate * (
                float(observed_costs[key]) - limit
            )
            self._multipliers[key] = max(0.0, candidate)

    def values(self) -> dict[str, float]:
        return dict(self._multipliers)

    def state_dict(self) -> dict[str, dict[str, float] | float]:
        return {
            "limits": dict(self.limits),
            "learning_rate": self.learning_rate,
            "multipliers": dict(self._multipliers),
        }

    def load_state_dict(self, state: Mapping) -> None:
        if dict(state.get("limits", {})) != self.limits:
            raise ValueError("checkpoint Lagrange limits do not match the experiment")
        multipliers = state.get("multipliers", {})
        if set(multipliers) != set(self.limits):
            raise ValueError("checkpoint Lagrange multiplier keys do not match")
        normalized = {key: float(value) for key, value in multipliers.items()}
        if any(value < 0 for value in normalized.values()):
            raise ValueError("checkpoint Lagrange multipliers cannot be negative")
        self._multipliers = normalized


@dataclass(frozen=True, slots=True)
class PPOLosses:
    total: Tensor
    policy: Tensor
    reward_value: Tensor
    cost_value: Tensor
    entropy: Tensor


class ConstrainedPPOTrainer:
    def __init__(
        self,
        lagrange_controller: LagrangeController,
        *,
        clip_ratio: float,
        value_coefficient: float,
        cost_value_coefficient: float,
        entropy_coefficient: float,
    ) -> None:
        if not 0 < clip_ratio < 1:
            raise ValueError("PPO clip ratio must lie in (0, 1)")
        self.lagrange_controller = lagrange_controller
        self.clip_ratio = float(clip_ratio)
        self.value_coefficient = float(value_coefficient)
        self.cost_value_coefficient = float(cost_value_coefficient)
        self.entropy_coefficient = float(entropy_coefficient)

    def compute_loss(
        self,
        *,
        new_log_prob: Tensor,
        old_log_prob: Tensor,
        reward_advantage: Tensor,
        reward_value: Tensor,
        reward_return: Tensor,
        cost_advantages: Mapping[str, Tensor],
        cost_values: Mapping[str, Tensor],
        cost_returns: Mapping[str, Tensor],
        entropy: Tensor,
    ) -> PPOLosses:
        expected_keys = set(self.lagrange_controller.limits)
        if set(cost_advantages) != expected_keys or set(cost_values) != expected_keys or set(cost_returns) != expected_keys:
            raise ValueError("cost advantage/value/return keys must match Lagrange constraints")
        combined_advantage = reward_advantage
        multipliers = self.lagrange_controller.values()
        for key in sorted(expected_keys):
            combined_advantage = combined_advantage - multipliers[key] * cost_advantages[key]
        ratio = torch.exp(new_log_prob - old_log_prob)
        unclipped = ratio * combined_advantage
        clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * combined_advantage
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        reward_value_loss = torch.nn.functional.mse_loss(reward_value, reward_return)
        cost_value_loss = torch.zeros((), dtype=reward_value.dtype, device=reward_value.device)
        for key in sorted(expected_keys):
            cost_value_loss = cost_value_loss + torch.nn.functional.mse_loss(
                cost_values[key], cost_returns[key]
            )
        entropy_mean = entropy.mean()
        total = (
            policy_loss
            + self.value_coefficient * reward_value_loss
            + self.cost_value_coefficient * cost_value_loss
            - self.entropy_coefficient * entropy_mean
        )
        return PPOLosses(total, policy_loss, reward_value_loss, cost_value_loss, entropy_mean)

    def optimization_step(
        self,
        optimizer: torch.optim.Optimizer,
        loss: Tensor,
        parameters,
        *,
        max_grad_norm: float = 0.5,
    ) -> float:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        optimizer.step()
        return float(gradient_norm)
