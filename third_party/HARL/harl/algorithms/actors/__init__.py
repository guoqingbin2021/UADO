# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

"""Algorithm registry."""
from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.hatrpo import HATRPO
from harl.algorithms.actors.haa2c import HAA2C
from harl.algorithms.actors.haddpg import HADDPG
from harl.algorithms.actors.hatd3 import HATD3
from harl.algorithms.actors.hasac import HASAC
from harl.algorithms.actors.had3qn import HAD3QN
from harl.algorithms.actors.maddpg import MADDPG
from harl.algorithms.actors.matd3 import MATD3
from harl.algorithms.actors.mappo import MAPPO

ALGO_REGISTRY = {
    "happo": HAPPO,
    "hatrpo": HATRPO,
    "haa2c": HAA2C,
    "haddpg": HADDPG,
    "hatd3": HATD3,
    "hasac": HASAC,
    "had3qn": HAD3QN,
    "maddpg": MADDPG,
    "matd3": MATD3,
    "mappo": MAPPO,
}
