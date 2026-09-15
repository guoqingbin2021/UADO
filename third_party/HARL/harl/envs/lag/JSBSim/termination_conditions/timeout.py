# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from .termination_condition_base import BaseTerminationCondition


class Timeout(BaseTerminationCondition):
    """
    Timeout
    Episode terminates if max_step steps have passed.
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_steps = getattr(self.config, 'max_steps', 500)

    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.
        Terminate if max_step steps have passed

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        done = env.current_step >= self.max_steps
        if done:
            self.log(f"{agent_id} step limits! Total Steps={env.current_step}")
        success = False
        return done, success, info
