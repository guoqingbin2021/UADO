# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

import numpy as np
from .reward_function_base import BaseRewardFunction


class RelativeAltitudeReward(BaseRewardFunction):
    """
    RelativeAltitudeReward
    Punish if current fighter doesn't satisfy some constraints. Typically negative.
    - Punishment of relative altitude when larger than 1000  (range: [-1, 0])

    NOTE:
    - Only support one-to-one environments.
    """
    def __init__(self, config):
        super().__init__(config)
        self.KH = getattr(self.config, f'{self.__class__.__name__}_KH', 1.0)     # km

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the punishments.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        ego_z = env.agents[agent_id].get_position()[-1] / 1000    # unit: km
        enm_z = env.agents[agent_id].enemies[0].get_position()[-1] / 1000    # unit: km
        new_reward = min(self.KH - np.abs(ego_z - enm_z), 0)
        return self._process(new_reward, agent_id)
