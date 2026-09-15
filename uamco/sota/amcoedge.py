from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np
import torch
from torch import nn

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


@dataclass(slots=True)
class _ReplayEntry:
    state: np.ndarray
    subset_action: int
    reward: float
    next_state: np.ndarray
    next_subset_mask: np.ndarray
    done: bool


class AMCoEdgeRuntime:
    """AMCoEdge target-DQN plus its subset selection and CWA/HECWA stage.

    The source algorithm has five collaborating edge-server slots and 31
    non-empty subsets.  The common simulator can expose more executors, so each
    observation maps the five best legal compute executors (estimated finish
    time) into those slots.  CWA/HECWA produces fractions on the common action
    vocabulary; because a DAG node is indivisible, the executor with the
    largest fraction is the environment action and the complete subset and
    fractions remain in the transition/checkpoint for audit.
    """

    method_name = "AMCoEdge"
    adapter_mode = "faithful_pytorch_port"
    collaborator_slots = 5

    def __init__(
        self,
        *,
        action_count: int,
        agent_count: int,
        device: str | torch.device = "cpu",
        seed: int = 0,
        batch_size: int = 32,
        replay_capacity: int = 500,
        target_replace_interval: int = 200,
        learning_rate: float = 1.0e-3,
        gamma: float = 0.9,
        epsilon_max: float = 0.99,
        epsilon_increment: float = 0.001,
        learning_starts: int = 200,
        update_interval: int = 10,
        allocation_mode: str = "CWA",
        max_grad_norm: float = 0.5,
        **_: object,
    ) -> None:
        del agent_count
        if action_count < 2 or batch_size <= 0 or replay_capacity <= 0:
            raise ValueError("AMCoEdge runtime dimensions must be positive")
        if allocation_mode not in {"CWA", "HECWA"}:
            raise ValueError("AMCoEdge allocation mode must be CWA or HECWA")
        if not math.isfinite(float(max_grad_norm)) or float(max_grad_norm) <= 0:
            raise ValueError("AMCoEdge max gradient norm must be finite and positive")
        self.action_count = int(action_count)
        self.state_dim = 4 + 5 * self.action_count
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.replay_capacity = int(replay_capacity)
        self.target_replace_interval = int(target_replace_interval)
        self.gamma = float(gamma)
        self.epsilon_max = float(epsilon_max)
        self.epsilon_increment = float(epsilon_increment)
        self.learning_starts = int(learning_starts)
        self.update_interval = int(update_interval)
        self.allocation_mode = allocation_mode
        self.max_grad_norm = float(max_grad_norm)
        self.subsets = tuple(
            subset
            for size in range(1, self.collaborator_slots + 1)
            for subset in combinations(range(self.collaborator_slots), size)
        )
        self.q_action_count = len(self.subsets)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.eval_net = nn.Sequential(
                nn.Linear(self.state_dim, 20), nn.ReLU(),
                nn.Linear(20, 20), nn.ReLU(),
                nn.Linear(20, 20), nn.ReLU(),
                nn.Linear(20, self.q_action_count),
            ).to(self.device)
        self.target_net = copy.deepcopy(self.eval_net).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=float(learning_rate))
        self.replay: deque[_ReplayEntry] = deque(maxlen=self.replay_capacity)
        self.epsilon = 0.0
        self.learn_step = 0
        self.action_step = 0
        self._rng = np.random.default_rng(int(seed))
        self._macro = FixedCentroidMacroPolicy()

    def reset_episode(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

    @staticmethod
    def _finish_time(observation: SOTAObservation, index: int) -> float:
        cpu = max(float(observation.executor_cpu_hz[index]), 1.0e-12)
        wait = float(observation.executor_queue_work[index]) / cpu
        rate = float(observation.link_rate_bps[index])
        transfer = 0.0 if math.isinf(rate) else 8.0 * float(observation.input_bytes) / max(rate, 1.0e-12)
        return wait + transfer + float(observation.cycles) / cpu

    def _slot_executors(self, observation: SOTAObservation) -> tuple[int, ...]:
        defer = observation.executor_ids.index("defer")
        legal = [
            index for index, allowed in enumerate(observation.action_mask)
            if allowed and index != defer
        ]
        legal.sort(key=lambda index: (self._finish_time(observation, index), index))
        return tuple(legal[: self.collaborator_slots])

    def _subset_mask(self, observation: SOTAObservation) -> np.ndarray:
        count = len(self._slot_executors(observation))
        return np.asarray([all(slot < count for slot in subset) for subset in self.subsets], dtype=bool)

    def workload_allocation(
        self, observation: SOTAObservation, selected_subset: tuple[int, ...], *, mode: str
    ) -> np.ndarray:
        if mode not in {"CWA", "HECWA"} or not selected_subset:
            raise ValueError("AMCoEdge allocation needs a non-empty subset and CWA/HECWA mode")
        selected = np.asarray(selected_subset, dtype=np.int64)
        service = np.asarray([self._finish_time(observation, int(i)) for i in selected])
        wait = np.asarray([
            float(observation.executor_queue_work[int(i)])
            / max(float(observation.executor_cpu_hz[int(i)]), 1.0e-12)
            for i in selected
        ])
        variable = np.maximum(service - wait, 1.0e-12)
        if len(selected) == 1:
            local_fraction = np.ones(1, dtype=np.float64)
        elif mode == "CWA":
            matrix = np.zeros((len(selected), len(selected)), dtype=np.float64)
            rhs = np.ones(len(selected), dtype=np.float64)
            for row in range(len(selected) - 1):
                matrix[row, row] = variable[row]
                matrix[row, row + 1] = -variable[row + 1]
                rhs[row] = wait[row + 1] - wait[row]
            matrix[-1, :] = 1.0
            try:
                local_fraction = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                local_fraction = 1.0 / variable
            local_fraction = np.maximum(local_fraction, 0.0)
            if float(local_fraction.sum()) <= 0.0:
                local_fraction = 1.0 / variable
            local_fraction /= local_fraction.sum()
        else:
            # Official HECWA product-of-other-service-times, written in the
            # equivalent and numerically safer reciprocal form.
            local_fraction = 1.0 / variable
            local_fraction /= local_fraction.sum()
        result = np.zeros(self.action_count, dtype=np.float64)
        result[selected] = local_fraction
        return result

    def act(
        self,
        observation: SOTAObservation,
        physical_context: SOTAPhysicalContext,
        *,
        deterministic: bool,
    ) -> SOTAAction:
        if observation.action_count != self.action_count:
            raise ValueError("AMCoEdge executor vocabulary changed")
        if not all_finite(self.eval_net):
            raise FloatingPointError("AMCoEdge action network contains non-finite state")
        state = torch.from_numpy(encode_observation(observation)).to(self.device).unsqueeze(0)
        with torch.no_grad():
            raw_q_tensor = self.eval_net(state).squeeze(0)
        if not all_finite(raw_q_tensor):
            raise FloatingPointError("AMCoEdge action scores contain non-finite values")
        raw_q = raw_q_tensor.cpu().numpy()
        subset_mask = self._subset_mask(observation)
        slot_executors = self._slot_executors(observation)
        if not slot_executors:
            defer_index = observation.executor_ids.index("defer")
            forced_scores = np.full(self.action_count, -math.inf, dtype=np.float64)
            forced_scores[defer_index] = 0.0
            return SOTAAction(
                executor_index=defer_index,
                scores=tuple(float(value) for value in forced_scores),
                macro_targets_xy_m=self._macro.targets(physical_context),
            )
        masked_q = np.where(subset_mask, raw_q, -math.inf)
        legal_subset_actions = np.flatnonzero(subset_mask)
        exploit = deterministic or self._rng.random() < self.epsilon
        subset_action = int(np.argmax(masked_q)) if exploit else int(self._rng.choice(legal_subset_actions))
        selected = tuple(slot_executors[slot] for slot in self.subsets[subset_action])
        fractions = self.workload_allocation(observation, selected, mode=self.allocation_mode)
        lead = int(max(selected, key=lambda index: (fractions[index], -index)))
        executor_scores = np.full(self.action_count, -math.inf, dtype=np.float64)
        for index in slot_executors:
            candidates = [
                masked_q[position]
                for position, subset in enumerate(self.subsets)
                if subset_mask[position]
                and index in tuple(slot_executors[slot] for slot in subset)
            ]
            executor_scores[index] = max(candidates) if candidates else -math.inf
        executor_scores = hard_mask_scores(executor_scores, observation.action_mask)
        return SOTAAction(
            executor_index=lead,
            scores=tuple(float(value) for value in executor_scores),
            macro_targets_xy_m=self._macro.targets(physical_context),
            selected_subset=selected,
            allocation_fractions=tuple(float(value) for value in fractions),
        )

    def _subset_action_from_transition(self, transition: SOTATransition) -> int:
        slots = self._slot_executors(transition.observation)
        selected_slots = tuple(sorted(slots.index(index) for index in transition.action.selected_subset))
        return self.subsets.index(selected_slots)

    def observe(self, transition: SOTATransition) -> None:
        if not transition.action.selected_subset:
            return
        self.replay.append(_ReplayEntry(
            state=encode_observation(transition.observation),
            subset_action=self._subset_action_from_transition(transition),
            reward=float(transition.reward),
            next_state=encode_observation(transition.next_observation),
            next_subset_mask=self._subset_mask(transition.next_observation),
            done=bool(transition.done),
        ))
        self.action_step += 1

    def _skipped_update(self, *, target_replaced: float) -> Mapping[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        return {
            "updated": 0.0,
            "loss": 0.0,
            "target_replaced": float(target_replaced),
            "epsilon": float(self.epsilon),
            "gradient_norm": 0.0,
            "nonfinite_skipped": 1.0,
        }

    def update_if_ready(self) -> Mapping[str, float]:
        ready = (
            len(self.replay) >= self.batch_size
            and self.action_step > self.learning_starts
            and self.action_step % self.update_interval == 0
        )
        if not ready:
            return {"updated": 0.0, "loss": 0.0, "target_replaced": 0.0}
        replacement_due = self.learn_step % self.target_replace_interval == 0
        indices = self._rng.choice(len(self.replay), size=self.batch_size, replace=False)
        entries = [self.replay[int(index)] for index in indices]
        states = torch.as_tensor(np.stack([e.state for e in entries]), device=self.device)
        actions = torch.as_tensor([e.subset_action for e in entries], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([e.reward for e in entries], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([e.next_state for e in entries]), device=self.device)
        dones = torch.as_tensor([e.done for e in entries], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(np.stack([e.next_subset_mask for e in entries]), dtype=torch.bool, device=self.device)
        predicted = self.eval_net(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            bootstrap_net = self.eval_net if replacement_due else self.target_net
            raw_next_q = bootstrap_net(next_states)
            bootstrap_rows = (~dones.bool()) & next_masks.any(dim=1)
            next_max = torch.zeros_like(rewards)
            if bool(bootstrap_rows.any()):
                legal_next_q = raw_next_q[bootstrap_rows].masked_fill(
                    ~next_masks[bootstrap_rows], -torch.inf
                )
                next_max[bootstrap_rows] = legal_next_q.max(dim=1).values
            target = rewards + self.gamma * next_max
        loss = nn.functional.mse_loss(predicted, target)
        if not all_finite(
            self.eval_net, self.target_net, self.optimizer,
            states, rewards, next_states, predicted, raw_next_q, target, loss,
        ):
            return self._skipped_update(target_replaced=0.0)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = clip_grad_norm_if_finite(
            self.eval_net.parameters(), self.max_grad_norm
        )
        if gradient_norm is None:
            return self._skipped_update(target_replaced=0.0)
        if replacement_due:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer.step()
        if not all_finite(self.eval_net, self.optimizer):
            raise FloatingPointError(
                "AMCoEdge optimizer produced non-finite model state"
            )
        self.epsilon = min(self.epsilon_max, self.epsilon + self.epsilon_increment)
        self.learn_step += 1
        return {
            "updated": 1.0,
            "loss": float(loss.detach().cpu()),
            "target_replaced": float(replacement_due),
            "epsilon": self.epsilon,
            "gradient_norm": float(gradient_norm),
            "nonfinite_skipped": 0.0,
        }

    def tick_micro_slot(self, micro_slot: int) -> Mapping[str, float]:
        del micro_slot
        return {"updated": 0.0}

    @staticmethod
    def _serialize_entry(entry: _ReplayEntry) -> dict[str, object]:
        return {"state": entry.state.copy(), "subset_action": entry.subset_action,
                "reward": entry.reward, "next_state": entry.next_state.copy(),
                "next_subset_mask": entry.next_subset_mask.copy(), "done": entry.done}

    def state_dict(self) -> Mapping[str, object]:
        return {"method": self.method_name, "eval_net": copy.deepcopy(self.eval_net.state_dict()),
                "target_net": copy.deepcopy(self.target_net.state_dict()),
                "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                "replay": [self._serialize_entry(e) for e in self.replay],
                "epsilon": self.epsilon, "learn_step": self.learn_step,
                "action_step": self.action_step, "learning_starts": self.learning_starts,
                "update_interval": self.update_interval, "allocation_mode": self.allocation_mode,
                "max_grad_norm": self.max_grad_norm,
                "rng_state": copy.deepcopy(self._rng.bit_generator.state)}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("method") != self.method_name:
            raise ValueError("AMCoEdge checkpoint method mismatch")
        self.eval_net.load_state_dict(state["eval_net"]); self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"]); self.replay.clear()
        for raw in state.get("replay", ()):
            self.replay.append(_ReplayEntry(
                state=np.asarray(raw["state"], dtype=np.float32), subset_action=int(raw["subset_action"]),
                reward=float(raw["reward"]), next_state=np.asarray(raw["next_state"], dtype=np.float32),
                next_subset_mask=np.asarray(raw["next_subset_mask"], dtype=bool), done=bool(raw["done"])))
        self.epsilon = float(state["epsilon"]); self.learn_step = int(state["learn_step"])
        self.action_step = int(state["action_step"]); self.learning_starts = int(state["learning_starts"])
        self.update_interval = int(state["update_interval"]); self.allocation_mode = str(state["allocation_mode"])
        saved_max_grad_norm = float(state.get("max_grad_norm", self.max_grad_norm))
        if not math.isclose(saved_max_grad_norm, self.max_grad_norm):
            raise ValueError("AMCoEdge checkpoint max gradient norm mismatch")
        if not all_finite(self.eval_net, self.target_net, self.optimizer):
            raise FloatingPointError("AMCoEdge checkpoint contains non-finite state")
        self._rng.bit_generator.state = copy.deepcopy(state["rng_state"])
