# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# BONUS TASK: Backflip with Recovery for Unitree Go2
# The robot learns to perform a backflip and land safely

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Rob6323-Go2-Backflip-Direct-v0",
    entry_point=f"{__name__}.rob6323_go2_backflip_env:Rob6323Go2BackflipEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rob6323_go2_backflip_env_cfg:Rob6323Go2BackflipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

