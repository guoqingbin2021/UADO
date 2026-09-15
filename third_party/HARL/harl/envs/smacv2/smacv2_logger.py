# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from harl.envs.smac.smac_logger import SMACLogger


class SMACv2Logger(SMACLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(SMACv2Logger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
        self.win_key = "battle_won"
