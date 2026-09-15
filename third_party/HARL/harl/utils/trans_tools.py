# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

"""Tools for HARL."""


def _t2n(value):
    """Convert torch.Tensor to numpy.ndarray."""
    return value.detach().cpu().numpy()


def _flatten(T, N, value):
    """Flatten the first two dimensions of a tensor."""
    return value.reshape(T * N, *value.shape[2:])


def _sa_cast(value):
    """This function is used for buffer data operation.
    Specifically, it transposes a tensor from (episode_length, n_rollout_threads, *dim) to (n_rollout_threads, episode_length, *dim).
    Then it combines the first two dimensions into one dimension.
    """
    return value.transpose(1, 0, 2).reshape(-1, *value.shape[2:])


def _ma_cast(value):
    """This function is used for buffer data operation.
    Specifically, it transposes a tensor from (episode_length, n_rollout_threads, num_agents, *dim) to (n_rollout_threads, num_agents, episode_length, *dim).
    Then it combines the first three dimensions into one dimension.
    """
    return value.transpose(1, 2, 0, 3).reshape(-1, *value.shape[3:])
