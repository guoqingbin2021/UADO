# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

"""Critic registry."""
from harl.algorithms.critics.v_critic import VCritic
from harl.algorithms.critics.continuous_q_critic import ContinuousQCritic
from harl.algorithms.critics.twin_continuous_q_critic import TwinContinuousQCritic
from harl.algorithms.critics.soft_twin_continuous_q_critic import (
    SoftTwinContinuousQCritic,
)
from harl.algorithms.critics.discrete_q_critic import DiscreteQCritic

CRITIC_REGISTRY = {
    "happo": VCritic,
    "hatrpo": VCritic,
    "haa2c": VCritic,
    "mappo": VCritic,
    "haddpg": ContinuousQCritic,
    "hatd3": TwinContinuousQCritic,
    "hasac": SoftTwinContinuousQCritic,
    "had3qn": DiscreteQCritic,
    "maddpg": ContinuousQCritic,
    "matd3": TwinContinuousQCritic,
}
