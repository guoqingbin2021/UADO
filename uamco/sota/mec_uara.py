from __future__ import annotations

import copy
import math
from typing import Mapping

import numpy as np

from .api import (
    SOTAAction,
    SOTAObservation,
    SOTAPhysicalContext,
    SOTATransition,
    hard_mask_scores,
)
from .fixed_macro_policy import FixedCentroidMacroPolicy


class MECUARARuntime:
    """Direct online port of MEC-UARA's projected two-dual association rule."""

    method_name = "MEC-UARA"
    adapter_mode = "direct_primal_dual_port"

    def __init__(
        self,
        *,
        action_count: int,
        agent_count: int,
        device: str = "cpu",
        seed: int = 0,
        mu: float = 1.0,
        delta: float = 2.6,
        flops_per_watt: float = 10.0e9,
        dual_lr_communication: float = 0.01,
        dual_lr_compute: float = 2.0,
        dual_lr_decay: float = 0.0001,
        **_: object,
    ) -> None:
        del device
        if action_count < 3 or agent_count <= 0:
            raise ValueError("MEC-UARA requires local, remote, defer and at least one owner")
        self.action_count = int(action_count)
        self.agent_count = int(agent_count)
        self.mu = float(mu)
        self.delta = float(delta)
        self.flops_per_watt = float(flops_per_watt)
        self.default_lr1 = float(dual_lr_communication)
        self.default_lr2 = float(dual_lr_compute)
        self.default_gamma = float(dual_lr_decay)
        self._rng = np.random.default_rng(int(seed))
        self._macro = FixedCentroidMacroPolicy()
        self.remote_action_indices: tuple[int, ...] = ()
        self.nu = np.zeros((2, 0), dtype=np.float64)
        self.X = np.zeros((self.agent_count, 0), dtype=np.float64)
        self.d_i = np.zeros((self.agent_count, 1), dtype=np.float64)
        self.f_i = np.zeros((self.agent_count, 1), dtype=np.float64)
        self.P_i = np.zeros((self.agent_count, 1), dtype=np.float64)
        self.F_j = np.zeros((1, 0), dtype=np.float64)
        self.R = np.zeros((self.agent_count, 0), dtype=np.float64)
        self.micro_slot = -1
        self.last_updated_micro_slot = -1
        self.dual_tick_count = 0
        self.last_candidate_costs: tuple[float, ...] = ()

    def reset_episode(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))
        self.X.fill(0.0)
        self.d_i.fill(0.0)
        self.f_i.fill(0.0)
        self.P_i.fill(0.0)
        self.micro_slot = -1
        self.last_updated_micro_slot = -1
        self.dual_tick_count = 0

    def _ensure_remote_layout(self, observation: SOTAObservation) -> None:
        remote = tuple(
            index
            for index, executor_id in enumerate(observation.executor_ids)
            if index != 0 and executor_id != "defer"
        )
        if not remote:
            raise ValueError("MEC-UARA requires at least one infrastructure executor")
        if not self.remote_action_indices:
            self.remote_action_indices = remote
            count = len(remote)
            self.nu = np.zeros((2, count), dtype=np.float64)
            self.X = np.zeros((self.agent_count, count), dtype=np.float64)
            self.F_j = np.zeros((1, count), dtype=np.float64)
            self.R = np.zeros((self.agent_count, count), dtype=np.float64)
        elif remote != self.remote_action_indices:
            raise ValueError("MEC-UARA infrastructure vocabulary changed")

    def candidate_costs(
        self,
        observation: SOTAObservation,
        physical_context: SOTAPhysicalContext,
    ) -> np.ndarray:
        self._ensure_remote_layout(observation)
        owner = int(observation.owner_index)
        if owner >= self.agent_count:
            raise ValueError("MEC-UARA owner index exceeds configured agent count")
        remote = np.asarray(self.remote_action_indices, dtype=np.int64)
        capacity = np.maximum(
            np.asarray(observation.executor_cpu_hz, dtype=np.float64)[remote], 1.0e-12
        )
        rate = np.asarray(observation.link_rate_bps, dtype=np.float64)[remote]
        snr = np.asarray(
            [max(0.0, math.exp2(value / physical_context.bandwidth_hz) - 1.0) for value in rate]
        )
        recovered_rate = np.log2(1.0 + snr) * physical_context.bandwidth_hz
        recovered_rate = np.maximum(recovered_rate, 1.0e-12)
        communication_load = float(observation.input_bytes) * 8.0
        compute_load = float(observation.cycles)
        local_cpu = max(float(observation.executor_cpu_hz[0]), 1.0e-12)
        remaining_battery_wh = max(
            float(observation.remaining_energy_j[0]) / 3600.0, 1.0e-12
        )
        affected_fraction = 0.99
        energy_difference = (
            compute_load / self.flops_per_watt / 3600.0
            - self.delta * communication_load / recovered_rate
        )
        local_compute = compute_load * affected_fraction / local_cpu
        remote_compute = compute_load / capacity
        base = self.mu * energy_difference / remaining_battery_wh - local_compute - remote_compute
        communication_dual = self.nu[0] * np.sqrt(communication_load / recovered_rate)
        compute_dual = self.nu[1] * np.sqrt(
            compute_load * affected_fraction / capacity
        )
        self.d_i[owner, 0] = communication_load
        self.f_i[owner, 0] = compute_load
        self.P_i[owner, 0] = affected_fraction
        self.F_j[0, :] = capacity
        self.R[owner, :] = recovered_rate
        return base + communication_dual + compute_dual

    def act(
        self,
        observation: SOTAObservation,
        physical_context: SOTAPhysicalContext,
        *,
        deterministic: bool,
    ) -> SOTAAction:
        del deterministic
        if observation.action_count != self.action_count:
            raise ValueError("MEC-UARA executor vocabulary changed")
        costs = self.candidate_costs(observation, physical_context)
        self.last_candidate_costs = tuple(float(value) for value in costs)
        scores = np.full(self.action_count, -math.inf, dtype=np.float64)
        if observation.action_mask[0]:
            scores[0] = 0.0
        for position, action_index in enumerate(self.remote_action_indices):
            if observation.action_mask[action_index]:
                scores[action_index] = -costs[position]
        defer_index = observation.executor_ids.index("defer")
        if observation.action_mask[defer_index]:
            scores[defer_index] = -1.0e12
        scores = hard_mask_scores(scores, observation.action_mask)
        choice = int(np.argmax(scores))
        owner = int(observation.owner_index)
        self.X[owner, :] = 0.0
        if choice in self.remote_action_indices:
            self.X[owner, self.remote_action_indices.index(choice)] = 1.0
        self.micro_slot = max(self.micro_slot, int(observation.micro_slot))
        return SOTAAction(
            executor_index=choice,
            scores=tuple(float(value) for value in scores),
            macro_targets_xy_m=self._macro.targets(physical_context),
        )

    def observe(self, transition: SOTATransition) -> None:
        self.micro_slot = max(self.micro_slot, int(transition.observation.micro_slot))

    def update_if_ready(self) -> Mapping[str, float]:
        if self.micro_slot < 0 or self.micro_slot == self.last_updated_micro_slot:
            return {"updated": 0.0}
        learning_rate_1 = self.default_lr1 / (1.0 + self.default_gamma * self.micro_slot)
        learning_rate_2 = self.default_lr2 / (1.0 + self.default_gamma * self.micro_slot)
        if self.nu.shape[1]:
            self.nu[0] += learning_rate_1 * (
                -self.nu[0] / 2.0
                + np.sum(np.sqrt(self.d_i / np.maximum(self.R, 1.0e-12)) * self.X, axis=0)
            )
            self.nu[1] += learning_rate_2 * (
                -self.nu[1] / 2.0
                + np.sum(
                    np.sqrt(
                        self.f_i * self.P_i / np.maximum(self.F_j, 1.0e-12)
                    )
                    * self.X,
                    axis=0,
                )
            )
            np.maximum(self.nu, 0.0, out=self.nu)
        self.last_updated_micro_slot = self.micro_slot
        self.dual_tick_count += 1
        self.X.fill(0.0)
        self.d_i.fill(0.0)
        self.f_i.fill(0.0)
        self.P_i.fill(0.0)
        return {
            "updated": 1.0,
            "dual_lr_communication": learning_rate_1,
            "dual_lr_compute": learning_rate_2,
            "dual_norm": float(np.linalg.norm(self.nu)),
            "dual_tick_count": float(self.dual_tick_count),
        }

    def tick_micro_slot(self, micro_slot: int) -> Mapping[str, float]:
        """Advance both projected duals exactly once for every micro-slot.

        If no association was made in the slot, ``X`` is already zero and the
        official ``-nu/2`` term still decays both dual vectors.
        """
        self.micro_slot = max(self.micro_slot, int(micro_slot))
        return self.update_if_ready()

    def state_dict(self) -> Mapping[str, object]:
        return {
            "method": self.method_name,
            "remote_action_indices": self.remote_action_indices,
            "nu": self.nu.copy(),
            "X": self.X.copy(),
            "d_i": self.d_i.copy(),
            "f_i": self.f_i.copy(),
            "P_i": self.P_i.copy(),
            "F_j": self.F_j.copy(),
            "R": self.R.copy(),
            "micro_slot": self.micro_slot,
            "last_updated_micro_slot": self.last_updated_micro_slot,
            "dual_tick_count": self.dual_tick_count,
            "last_candidate_costs": self.last_candidate_costs,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("method") != self.method_name:
            raise ValueError("MEC-UARA checkpoint method mismatch")
        self.remote_action_indices = tuple(int(value) for value in state["remote_action_indices"])
        for name in ("nu", "X", "d_i", "f_i", "P_i", "F_j", "R"):
            setattr(self, name, np.asarray(state[name], dtype=np.float64).copy())
        self.micro_slot = int(state["micro_slot"])
        self.last_updated_micro_slot = int(state["last_updated_micro_slot"])
        self.dual_tick_count = int(state.get("dual_tick_count", 0))
        self.last_candidate_costs = tuple(float(value) for value in state["last_candidate_costs"])
        self._rng.bit_generator.state = copy.deepcopy(state["rng_state"])
