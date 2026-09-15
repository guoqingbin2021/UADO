# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

import logging
from abc import ABC, abstractmethod


class BaseTerminationCondition(ABC):
    """
    Base TerminationCondition class
    Condition-specific get_termination method is implemented in subclasses
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.
        Overwritten by subclasses.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        raise NotImplementedError

    def log(self, msg):
        logging.debug(msg)
