# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

import torch.nn as nn
from harl.utils.models_tools import get_active_func


class PlainMLP(nn.Module):
    """Plain MLP"""

    def __init__(self, sizes, activation_func, final_activation_func="identity"):
        super().__init__()
        layers = []
        for j in range(len(sizes) - 1):
            act = activation_func if j < len(sizes) - 2 else final_activation_func
            layers += [nn.Linear(sizes[j], sizes[j + 1]), get_active_func(act)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
