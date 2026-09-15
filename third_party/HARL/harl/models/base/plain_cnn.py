# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

import torch.nn as nn
from harl.utils.models_tools import get_active_func
from harl.models.base.flatten import Flatten


class PlainCNN(nn.Module):
    """Plain CNN"""

    def __init__(
        self, obs_shape, hidden_size, activation_func, kernel_size=3, stride=1
    ):
        super().__init__()
        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]
        layers = [
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=hidden_size // 4,
                kernel_size=kernel_size,
                stride=stride,
            ),
            get_active_func(activation_func),
            Flatten(),
            nn.Linear(
                hidden_size
                // 4
                * (input_width - kernel_size + stride)
                * (input_height - kernel_size + stride),
                hidden_size,
            ),
            get_active_func(activation_func),
        ]
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)
        return x
