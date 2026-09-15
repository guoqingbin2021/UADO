# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

"""HATD3 algorithm."""
import torch
from harl.utils.envs_tools import check
from harl.algorithms.actors.haddpg import HADDPG


class HATD3(HADDPG):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super().__init__(args, obs_space, act_space, device)
        self.policy_noise = args["policy_noise"]
        self.noise_clip = args["noise_clip"]

    def get_target_actions(self, obs):
        """Get target actor actions for observations.
        Args:
            obs: (np.ndarray) observations of target actor, shape is (batch_size, dim)
        Returns:
            actions: (torch.Tensor) actions taken by target actor, shape is (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        actions = self.target_actor(obs)
        noise = torch.randn_like(actions) * self.policy_noise * self.scale
        noise = torch.clamp(
            noise, -self.noise_clip * self.scale, self.noise_clip * self.scale
        )
        actions += noise
        actions = torch.clamp(actions, self.low, self.high)
        return actions
