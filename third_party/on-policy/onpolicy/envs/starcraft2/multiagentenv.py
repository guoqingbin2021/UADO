# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class MultiAgentEnv(object):

    def step(self, actions):
        """Returns reward, terminated, info."""
        raise NotImplementedError

    def get_obs(self):
        """Returns all agent observations in a list."""
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        raise NotImplementedError

    def get_obs_size(self):
        """Returns the size of the observation."""
        raise NotImplementedError

    def get_state(self):
        """Returns the global state."""
        raise NotImplementedError

    def get_state_size(self):
        """Returns the size of the global state."""
        raise NotImplementedError

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        raise NotImplementedError

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        raise NotImplementedError

    def reset(self):
        """Returns initial observations and states."""
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        """Save a replay."""
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "obs_alone_shape": self.get_obs_alone_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
