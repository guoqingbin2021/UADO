# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from harl.common.base_logger import BaseLogger


class MAMuJoCoLogger(BaseLogger):
    def get_task_name(self):
        return f"{self.env_args['scenario']}-{self.env_args['agent_conf']}"
