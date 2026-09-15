from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .api import (
    SOTAAction,
    SOTAObservation,
    SOTAPhysicalContext,
    SOTATransition,
    all_finite,
    clip_grad_norm_if_finite,
    encode_observation,
    hard_mask_scores,
)
from .fixed_macro_policy import FixedCentroidMacroPolicy


def _extract(values: torch.Tensor, time: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, time)
    return out.reshape(time.shape[0], *((1,) * (len(shape) - 1)))


class _SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__(); self.dim = int(dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10_000.0) / max(half - 1, 1)
        frequencies = torch.exp(torch.arange(half, device=time.device) * -scale)
        angles = time[:, None].float() * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class _PolicyDenoiser(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_count: int, time_dim: int = 16) -> None:
        super().__init__()
        self.time_embedding = _SinusoidalPosEmb(time_dim)
        self.fc1 = nn.Linear(state_dim + action_count + time_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_count)

    def forward(self, noisy_action: torch.Tensor, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x = torch.cat((noisy_action, self.time_embedding(time), state.reshape(state.shape[0], -1)), dim=1)
        return F.softmax(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))), dim=1)


class _FeedbackDiffusionActor(nn.Module):
    """Equation-level port of the pinned FDEdge VP diffusion actor."""

    def __init__(self, state_dim: int, action_count: int, hidden_dim: int, steps: int) -> None:
        super().__init__()
        self.action_count = int(action_count); self.steps = int(steps); self.n_timesteps = int(steps)
        self.model = _PolicyDenoiser(state_dim, hidden_dim, action_count)
        time = np.arange(1, steps + 1)
        alpha = np.exp(-0.1 / steps - 0.5 * (10.0 - 0.1) * (2 * time - 1) / steps**2)
        betas = torch.tensor(1.0 - alpha, dtype=torch.float32)
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        previous = torch.cat((torch.ones(1), cumulative[:-1]))
        posterior_variance = betas * (1.0 - previous) / (1.0 - cumulative)
        buffers = {
            "betas": betas, "alphas_cumprod": cumulative, "alphas_cumprod_prev": previous,
            "sqrt_alphas_cumprod": torch.sqrt(cumulative),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - cumulative),
            "sqrt_recip_alphas_cumprod": torch.sqrt(1.0 / cumulative),
            "sqrt_recipm1_alphas_cumprod": torch.sqrt(1.0 / cumulative - 1.0),
            "posterior_variance": posterior_variance,
            "posterior_log_variance_clipped": torch.log(torch.clamp(posterior_variance, min=1e-20)),
            "posterior_mean_coef1": betas * torch.sqrt(previous) / (1.0 - cumulative),
            "posterior_mean_coef2": (1.0 - previous) * torch.sqrt(alphas) / (1.0 - cumulative),
        }
        for name, value in buffers.items(): self.register_buffer(name, value)

    def predict_start_from_noise(self, current: torch.Tensor, time: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return _extract(self.sqrt_recip_alphas_cumprod, time, current.shape) * current - _extract(
            self.sqrt_recipm1_alphas_cumprod, time, current.shape
        ) * noise

    def q_posterior(self, x_start: torch.Tensor, current: torch.Tensor, time: torch.Tensor):
        mean = _extract(self.posterior_mean_coef1, time, current.shape) * x_start + _extract(
            self.posterior_mean_coef2, time, current.shape
        ) * current
        return (mean, _extract(self.posterior_variance, time, current.shape),
                _extract(self.posterior_log_variance_clipped, time, current.shape))

    def p_sample(self, current: torch.Tensor, time: torch.Tensor, state: torch.Tensor, *, deterministic: bool) -> torch.Tensor:
        predicted_noise = self.model(current, time, state)
        reconstructed = self.predict_start_from_noise(current, time, predicted_noise)
        mean, _, log_variance = self.q_posterior(reconstructed, current, time)
        noise = torch.zeros_like(current) if deterministic else torch.randn_like(current)
        nonzero = (time != 0).float().reshape(current.shape[0], 1)
        return mean + nonzero * torch.exp(0.5 * log_variance) * noise

    def forward(self, state: torch.Tensor, latent: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        current = torch.zeros_like(latent) if deterministic else torch.randn_like(latent)
        for index in reversed(range(self.n_timesteps)):
            time = torch.full((state.shape[0],), index, device=state.device, dtype=torch.long)
            current = self.p_sample(current, time, state, deterministic=deterministic)
        return F.softmax(current, dim=-1)


class _TwinQ(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_count: int) -> None:
        super().__init__(); self.layers = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_count))
    def forward(self, state: torch.Tensor) -> torch.Tensor: return self.layers(state)


@dataclass(slots=True)
class _ReplayEntry:
    state: np.ndarray; mask: np.ndarray; action: int; latent_action_probabilities: np.ndarray
    reward: float; next_state: np.ndarray; next_mask: np.ndarray
    next_latent_action_probabilities: np.ndarray; done: bool


class FDEdgeRuntime:
    method_name = "FDEdge"; adapter_mode = "package_safe_core_port"

    def __init__(self, *, action_count: int, agent_count: int, device: str | torch.device = "cpu",
                 seed: int = 0, batch_size: int = 64, replay_capacity: int = 10_000,
                 hidden_dim: int = 128, actor_learning_rate: float = 1.0e-4,
                 critic_learning_rate: float = 1.0e-3, alpha_learning_rate: float = 3.0e-4,
                 alpha: float = 0.05, gamma: float = 0.95, tau: float = 0.005,
                 target_entropy: float = -1.0, denoising_steps: int = 5,
                 max_grad_norm: float = 0.5, **_: object) -> None:
        del agent_count
        if action_count < 2 or denoising_steps != 5: raise ValueError("FDEdge requires five denoising steps")
        if not math.isfinite(float(max_grad_norm)) or float(max_grad_norm) <= 0:
            raise ValueError("FDEdge max gradient norm must be finite and positive")
        self.action_count = int(action_count); self.state_dim = 4 + 5 * self.action_count
        self.device = torch.device(device); self.batch_size = int(batch_size); self.hidden_dim = int(hidden_dim)
        self.replay_capacity = int(replay_capacity); self.gamma = float(gamma); self.tau = float(tau)
        self.target_entropy = float(target_entropy); self.denoising_steps = int(denoising_steps)
        self.max_grad_norm = float(max_grad_norm)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed)); self.actor = _FeedbackDiffusionActor(self.state_dim, self.action_count, self.hidden_dim, self.denoising_steps).to(self.device)
            self.critic_1 = _TwinQ(self.state_dim, self.hidden_dim, self.action_count).to(self.device); self.critic_2 = _TwinQ(self.state_dim, self.hidden_dim, self.action_count).to(self.device)
        self.target_1 = copy.deepcopy(self.critic_1).requires_grad_(False); self.target_2 = copy.deepcopy(self.critic_2).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(actor_learning_rate)); self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=float(critic_learning_rate)); self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=float(critic_learning_rate))
        self.log_alpha = torch.tensor(np.log(float(alpha)), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=float(alpha_learning_rate))
        self.replay: deque[_ReplayEntry] = deque(maxlen=self.replay_capacity); self._rng = np.random.default_rng(int(seed)); self._macro = FixedCentroidMacroPolicy(); self._latent_by_task: dict[str, np.ndarray] = {}

    def reset_episode(self, seed: int) -> None: self._rng = np.random.default_rng(int(seed)); self._latent_by_task.clear()

    def _policy_probabilities(self, state: torch.Tensor, mask: torch.Tensor, latent: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        raw = self.actor(state, latent, deterministic=deterministic)
        if not all_finite(raw):
            raise FloatingPointError("FDEdge actor produced non-finite probabilities")
        probabilities = raw * mask.to(torch.float32)
        denominator = probabilities.sum(dim=1, keepdim=True)
        if not all_finite(denominator):
            raise FloatingPointError("FDEdge masked probability mass is non-finite")
        legal_count = mask.sum(dim=1, keepdim=True)
        if bool((legal_count <= 0).any()):
            raise ValueError("FDEdge action mask contains no positive probability mass")
        empty_mass = denominator <= 0
        safe_denominator = torch.where(
            empty_mass,
            torch.ones_like(denominator),
            denominator,
        )
        normalized = probabilities / safe_denominator
        legal_fallback = mask.to(torch.float32) / legal_count.to(torch.float32)
        normalized = torch.where(empty_mass, legal_fallback, normalized)
        if not all_finite(normalized):
            raise FloatingPointError("FDEdge normalized probabilities are non-finite")
        return normalized

    def act(self, observation: SOTAObservation, physical_context: SOTAPhysicalContext, *, deterministic: bool) -> SOTAAction:
        state = torch.from_numpy(encode_observation(observation)).to(self.device).unsqueeze(0)
        latent_array = self._latent_by_task.get(observation.candidate_task_id, np.zeros(self.action_count, dtype=np.float32))
        latent = torch.from_numpy(latent_array).to(self.device).unsqueeze(0); mask = torch.as_tensor(observation.action_mask, dtype=torch.bool, device=self.device)[None, :]
        with torch.no_grad(): probabilities = self._policy_probabilities(state, mask, latent, deterministic=deterministic).squeeze(0)
        feedback = probabilities.cpu().numpy(); self._latent_by_task[observation.candidate_task_id] = feedback
        scores = hard_mask_scores(torch.log(probabilities.clamp_min(1e-12)).cpu().numpy(), observation.action_mask)
        index = int(np.argmax(scores)) if deterministic else int(self._rng.choice(self.action_count, p=feedback))
        return SOTAAction(executor_index=index, scores=tuple(float(v) for v in scores), macro_targets_xy_m=self._macro.targets(physical_context), latent_action_probabilities=tuple(float(v) for v in feedback))

    def observe(self, transition: SOTATransition) -> None:
        state = encode_observation(transition.observation); next_state = encode_observation(transition.next_observation)
        current_latent = np.asarray(transition.action.latent_action_probabilities, dtype=np.float32)
        if current_latent.shape != (self.action_count,): current_latent = np.zeros(self.action_count, dtype=np.float32)
        with torch.no_grad():
            next_prob = self._policy_probabilities(torch.as_tensor(next_state, device=self.device)[None, :], torch.as_tensor(transition.next_observation.action_mask, dtype=torch.bool, device=self.device)[None, :], torch.as_tensor(current_latent, device=self.device)[None, :], deterministic=True).squeeze(0).cpu().numpy()
        self.replay.append(_ReplayEntry(state, np.asarray(transition.observation.action_mask, dtype=bool), int(transition.action.executor_index), current_latent, float(transition.reward), next_state, np.asarray(transition.next_observation.action_mask, dtype=bool), next_prob, bool(transition.done)))

    def _skipped_update(self) -> Mapping[str, float]:
        for optimizer in (
            self.actor_optimizer,
            self.critic_1_optimizer,
            self.critic_2_optimizer,
            self.alpha_optimizer,
        ):
            optimizer.zero_grad(set_to_none=True)
        return {
            "updated": 0.0,
            "nonfinite_skipped": 1.0,
            "actor_loss": 0.0,
            "critic_1_loss": 0.0,
            "critic_2_loss": 0.0,
            "alpha": float(self.log_alpha.exp().detach()),
            "terminal_bootstrap": 0.0,
            "actor_gradient_norm": 0.0,
            "critic_1_gradient_norm": 0.0,
            "critic_2_gradient_norm": 0.0,
            "alpha_gradient_norm": 0.0,
            "denoising_steps": 5.0,
        }

    def update_if_ready(self) -> Mapping[str, float]:
        if len(self.replay) < self.batch_size:
            return {"updated": 0.0, "denoising_steps": 5.0}
        entries = [self.replay[int(i)] for i in self._rng.choice(len(self.replay), size=self.batch_size, replace=False)]
        stack = lambda name: torch.as_tensor(np.stack([getattr(e, name) for e in entries]), device=self.device)
        states, masks, latent = stack("state"), stack("mask").bool(), stack("latent_action_probabilities")
        actions = torch.as_tensor([e.action for e in entries], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([e.reward for e in entries], dtype=torch.float32, device=self.device)
        next_states, next_masks, next_latent = stack("next_state"), stack("next_mask").bool(), stack("next_latent_action_probabilities")
        dones = torch.as_tensor([e.done for e in entries], dtype=torch.float32, device=self.device)
        if not all_finite(states, rewards, next_states, latent, next_latent):
            return self._skipped_update()
        with torch.no_grad():
            next_p = self._policy_probabilities(next_states, next_masks, next_latent)
            next_log = torch.log(next_p.clamp_min(1e-8))
            next_q = torch.minimum(self.target_1(next_states), self.target_2(next_states))
            next_value = (next_p * next_q).sum(1) + self.log_alpha.exp() * (-(next_p * next_log).sum(1))
            bootstrap = torch.where(dones.bool(), torch.zeros_like(next_value), self.gamma * next_value)
            target = rewards + bootstrap
        q1 = self.critic_1(states).gather(1, actions[:, None]).squeeze(1)
        q2 = self.critic_2(states).gather(1, actions[:, None]).squeeze(1)
        loss1 = F.mse_loss(q1, target)
        loss2 = F.mse_loss(q2, target)
        if not all_finite(next_p, next_q, target, q1, q2, loss1, loss2):
            return self._skipped_update()

        self.critic_1_optimizer.zero_grad(set_to_none=True)
        self.critic_2_optimizer.zero_grad(set_to_none=True)
        loss1.backward()
        loss2.backward()
        critic_1_norm = clip_grad_norm_if_finite(self.critic_1.parameters(), self.max_grad_norm)
        critic_2_norm = clip_grad_norm_if_finite(self.critic_2.parameters(), self.max_grad_norm)
        if critic_1_norm is None or critic_2_norm is None:
            return self._skipped_update()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        if not all_finite(self.critic_1, self.critic_2, self.critic_1_optimizer, self.critic_2_optimizer):
            raise FloatingPointError("FDEdge critic step produced non-finite state")

        p = self._policy_probabilities(states, masks, latent)
        logp = torch.log(p.clamp_min(1e-8))
        entropy = -(p * logp).sum(1)
        min_q = (p * torch.minimum(self.critic_1(states), self.critic_2(states)).detach()).sum(1)
        actor_loss = (-self.log_alpha.exp().detach() * entropy - min_q).mean()
        alpha_loss = ((entropy.detach() - self.target_entropy) * self.log_alpha.exp()).mean()
        if not all_finite(p, entropy, min_q, actor_loss, alpha_loss):
            return self._skipped_update()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_norm = clip_grad_norm_if_finite(self.actor.parameters(), self.max_grad_norm)
        if actor_norm is None:
            return self._skipped_update()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_norm = clip_grad_norm_if_finite((self.log_alpha,), self.max_grad_norm)
        if alpha_norm is None:
            return self._skipped_update()
        self.alpha_optimizer.step()
        if not all_finite(self.actor, self.log_alpha, self.actor_optimizer, self.alpha_optimizer):
            raise FloatingPointError("FDEdge actor step produced non-finite state")

        with torch.no_grad():
            for target_net, net in ((self.target_1, self.critic_1), (self.target_2, self.critic_2)):
                for target_parameter, parameter in zip(target_net.parameters(), net.parameters()):
                    target_parameter.mul_(1.0 - self.tau).add_(parameter, alpha=self.tau)
        if not all_finite(self.target_1, self.target_2):
            raise FloatingPointError("FDEdge target update produced non-finite state")
        return {
            "updated": 1.0,
            "nonfinite_skipped": 0.0,
            "actor_loss": float(actor_loss.detach()),
            "critic_1_loss": float(loss1.detach()),
            "critic_2_loss": float(loss2.detach()),
            "alpha": float(self.log_alpha.exp().detach()),
            "terminal_bootstrap": float(bootstrap.abs().mean().detach()),
            "actor_gradient_norm": float(actor_norm),
            "critic_1_gradient_norm": float(critic_1_norm),
            "critic_2_gradient_norm": float(critic_2_norm),
            "alpha_gradient_norm": float(alpha_norm),
            "denoising_steps": 5.0,
        }

    def tick_micro_slot(self, micro_slot: int) -> Mapping[str, float]: del micro_slot; return {"updated": 0.0}

    @staticmethod
    def _serialize_entry(e: _ReplayEntry) -> dict[str, object]: return {name: copy.deepcopy(getattr(e, name)) for name in e.__dataclass_fields__}
    def state_dict(self) -> Mapping[str, object]: return {"method": self.method_name, "actor": copy.deepcopy(self.actor.state_dict()), "critic_1": copy.deepcopy(self.critic_1.state_dict()), "critic_2": copy.deepcopy(self.critic_2.state_dict()), "target_1": copy.deepcopy(self.target_1.state_dict()), "target_2": copy.deepcopy(self.target_2.state_dict()), "log_alpha": self.log_alpha.detach().cpu().clone(), "optimizers": {"actor": copy.deepcopy(self.actor_optimizer.state_dict()), "critic_1": copy.deepcopy(self.critic_1_optimizer.state_dict()), "critic_2": copy.deepcopy(self.critic_2_optimizer.state_dict()), "alpha": copy.deepcopy(self.alpha_optimizer.state_dict())}, "replay": [self._serialize_entry(e) for e in self.replay], "max_grad_norm": self.max_grad_norm, "rng_state": copy.deepcopy(self._rng.bit_generator.state)}
    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("method") != self.method_name: raise ValueError("FDEdge checkpoint method mismatch")
        for name in ("actor", "critic_1", "critic_2", "target_1", "target_2"): getattr(self, name).load_state_dict(state[name])
        with torch.no_grad(): self.log_alpha.copy_(torch.as_tensor(state["log_alpha"], device=self.device))
        for name, optimizer in (("actor", self.actor_optimizer), ("critic_1", self.critic_1_optimizer), ("critic_2", self.critic_2_optimizer), ("alpha", self.alpha_optimizer)): optimizer.load_state_dict(state["optimizers"][name])
        self.replay.clear()
        for raw in state.get("replay", ()): self.replay.append(_ReplayEntry(state=np.asarray(raw["state"], dtype=np.float32), mask=np.asarray(raw["mask"], dtype=bool), action=int(raw["action"]), latent_action_probabilities=np.asarray(raw["latent_action_probabilities"], dtype=np.float32), reward=float(raw["reward"]), next_state=np.asarray(raw["next_state"], dtype=np.float32), next_mask=np.asarray(raw["next_mask"], dtype=bool), next_latent_action_probabilities=np.asarray(raw["next_latent_action_probabilities"], dtype=np.float32), done=bool(raw["done"])))
        self._rng.bit_generator.state = copy.deepcopy(state["rng_state"]); self._latent_by_task.clear()
        saved_max_grad_norm = float(state.get("max_grad_norm", self.max_grad_norm))
        if not math.isclose(saved_max_grad_norm, self.max_grad_norm):
            raise ValueError("FDEdge checkpoint max gradient norm mismatch")
        if not all_finite(
            self.actor, self.critic_1, self.critic_2,
            self.target_1, self.target_2, self.log_alpha,
            self.actor_optimizer, self.critic_1_optimizer,
            self.critic_2_optimizer, self.alpha_optimizer,
        ):
            raise FloatingPointError("FDEdge checkpoint contains non-finite state")
