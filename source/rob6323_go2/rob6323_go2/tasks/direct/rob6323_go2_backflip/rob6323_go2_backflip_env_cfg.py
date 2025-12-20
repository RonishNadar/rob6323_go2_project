# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# BONUS TASK: Backflip with Recovery Configuration for Unitree Go2
#
# Backflip phases:
# 1. Preparation - crouch down
# 2. Launch - explosive jump upward
# 3. Rotation - pitch backward 360 degrees
# 4. Landing - extend legs, absorb impact
# 5. Recovery - return to standing

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.actuators import ImplicitActuatorCfg
import math


@configclass
class Rob6323Go2BackflipEnvCfg(DirectRLEnvCfg):
    """Configuration for backflip with recovery task."""
    
    # env
    decimation = 4
    episode_length_s = 5.0  # Short episodes - backflip should be quick
    
    # spaces definition
    action_scale = 0.5  # Larger actions for explosive movements
    action_space = 12
    # Observations: 3 (lin_vel) + 3 (ang_vel) + 3 (proj_grav) + 12 (joint_pos) + 12 (joint_vel) 
    #             + 12 (actions) + 1 (phase) + 1 (cumulative_pitch) = 47
    observation_space = 47
    state_space = 0
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    
    # robot
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # PD control gains - higher for explosive movements
    Kp = 40.0  # Higher stiffness for quick movements
    Kd = 1.0
    torque_limits = 100.0
    
    # Disable implicit PD
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,
        damping=0.0,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", 
        history_length=3, 
        update_period=0.005, 
        track_air_time=True
    )

    # Termination - more lenient for acrobatics
    base_height_min = 0.10  # Lower threshold during flip
    
    # ========================================
    # BACKFLIP REWARD SCALES
    # ========================================
    
    # Phase 1 & 2: Jump height reward
    jump_height_reward_scale = 10.0
    target_jump_height = 0.8  # Target peak height during flip
    
    # Phase 3: Rotation reward (pitch angular velocity)
    # Backward flip = negative pitch angular velocity (rotating head backward)
    pitch_velocity_reward_scale = 5.0
    target_pitch_velocity = -10.0  # rad/s (backward rotation)
    
    # Cumulative rotation tracking
    rotation_progress_reward_scale = 20.0  # Big reward for completing rotation
    target_rotation = -2 * math.pi  # Full 360 degree backflip
    
    # Phase 4 & 5: Landing and recovery
    landing_upright_reward_scale = 15.0
    landing_feet_contact_reward_scale = 5.0
    recovery_stable_reward_scale = 10.0
    
    # Penalties
    action_rate_reward_scale = -0.01
    torque_reward_scale = -0.0001
    collision_penalty_scale = -5.0  # Penalty for body hitting ground
    
    # Orientation penalties (gentle during flip)
    orient_reward_scale = -0.1
    ang_vel_xy_reward_scale = -0.01  # Don't penalize pitch rotation
    
    # No actuator friction for backflip - need max power
    enable_actuator_friction = False
    friction_viscous_range = (0.0, 0.0)
    friction_stiction_range = (0.0, 0.0)

