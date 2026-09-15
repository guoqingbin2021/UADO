# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from .termination_condition_base import BaseTerminationCondition


class SafeReturn(BaseTerminationCondition):
    """
    SafeReturn.
    End up the simulation if:
        - the current aircraft has been shot down.
        - all the enemy-aircrafts has been destroyed while current aircraft is not under attack.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.

        End up the simulation if:
            - the current aircraft has been shot down.
            - all the enemy-aircrafts has been destroyed while current aircraft is not under attack.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        # the current aircraft has crashed
        if env.agents[agent_id].is_shotdown:
            self.log(f'{agent_id} has been shot down! Total Steps={env.current_step}')
            return True, False, info

        elif env.agents[agent_id].is_crash:
            self.log(f'{agent_id} has crashed! Total Steps={env.current_step}')
            return True, False, info

        # all the enemy-aircrafts has been destroyed while current aircraft is not under attack
        elif all([not enemy.is_alive for enemy in env.agents[agent_id].enemies]) \
                and all([not missile.is_alive for missile in env.agents[agent_id].under_missiles]):
            self.log(f'{agent_id} mission completed! Total Steps={env.current_step}')
            return True, True, info

        else:
            return False, False, info
