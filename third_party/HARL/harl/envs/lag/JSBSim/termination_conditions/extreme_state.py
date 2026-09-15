# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from .termination_condition_base import BaseTerminationCondition
from ..core.catalog import Catalog as c


class ExtremeState(BaseTerminationCondition):
    """
    ExtremeState
    End up the simulation if the aircraft is on an extreme state.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft is on an extreme state.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        done = bool(env.agents[agent_id].get_property_value(c.detect_extreme_state))
        if done:
            env.agents[agent_id].crash()
            self.log(f'{agent_id} is on an extreme state! Total Steps={env.current_step}')
        success = False
        return done, success, info
