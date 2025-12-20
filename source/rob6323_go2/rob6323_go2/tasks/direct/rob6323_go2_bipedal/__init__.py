# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# BONUS TASK: Bipedal Walking for Unitree Go2
# Reference: https://arxiv.org/pdf/2509.00215v2 (DMO Paper)

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Rob6323-Go2-Bipedal-Direct-v0",
    entry_point=f"{__name__}.rob6323_go2_bipedal_env:Rob6323Go2BipedalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rob6323_go2_bipedal_env_cfg:Rob6323Go2BipedalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


