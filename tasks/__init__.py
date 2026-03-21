# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task registration for the IsaacLab project template.

TODO: Rename the task IDs and update entry_point paths when you rename or add
      environment classes (see template_flat.py / template_terrain.py).
"""

import gymnasium as gym
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="template_flat_v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.template_flat:MyFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo:RSLRLPPORunnerCfg",
    },
)

gym.register(
    id="template_terrain_v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.template_terrain:MyTerrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo:RSLRLPPORunnerCfg",
    },
)


