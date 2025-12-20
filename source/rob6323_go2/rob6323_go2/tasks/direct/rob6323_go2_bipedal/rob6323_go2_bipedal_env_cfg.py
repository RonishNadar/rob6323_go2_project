# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# BONUS TASK: Bipedal Walking Configuration for Unitree Go2
# Reference: https://arxiv.org/pdf/2509.00215v2 (DMO Paper - Table 3)

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class Rob6323Go2BipedalEnvCfg(DirectRLEnvCfg):
    """Configuration for bipedal walking on Unitree Go2.
    
    Based on DMO paper (arXiv:2509.00215v2) reward structure.
    The robot should stand on its rear legs with front legs lifted.
    """
    
    # env
    decimation = 4
    episode_length_s = 15.0  # Shorter episodes for harder task
    
    # spaces definition
    action_scale = 0.25
    action_space = 12
    # Observations: 3 (ang_vel) + 3 (proj_grav) + 12 (joint_pos) + 12 (joint_vel) + 12 (actions) + 4 (clock) = 46
    observation_space = 46
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
    
    # robot(s)
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # PD control gains
    Kp = 20.0
    Kd = 0.5
    torque_limits = 100.0
    
    # Disable implicit PD control
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
    
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # Termination
    base_height_min = 0.25  # Higher threshold for bipedal
    
    # Gait configuration for bipedal (only rear legs)
    gait_frequency = 2.0  # Slower gait for stability
    
    # ========================================
    # REWARD SCALES - Tuned for Bipedal Learning
    # Priority: 1) Stand up, 2) Lift front legs, 3) Walk
    # ========================================
    
    # Velocity tracking - REDUCED initially (only matters after standing)
    lin_vel_reward_scale = 0.5
    yaw_rate_reward_scale = 0.05
    
    # Penalize lateral velocity
    lin_vel_y_penalty_scale = -0.2
    
    # CRITICAL: Upright posture reward - VERY HIGH to prioritize standing
    upright_reward_scale = 5.0
    
    # CRITICAL: Base height reward - bipedal stance should be HIGH (~0.4-0.5m)
    base_height_target = 0.45  # Target height when standing on rear legs
    base_height_reward_scale = 3.0
    
    # CRITICAL: Front leg lift reward - STRONG to lift front legs
    front_leg_lift_reward_scale = 2.0
    front_leg_target_height = 0.25  # Front feet should be 25cm off ground
    
    # Action rate penalty - reduced for exploration
    action_rate_reward_scale = -0.01
    
    # Joint acceleration penalty
    joint_accel_reward_scale = -0.00005
    
    # Torque penalty - reduced for exploration
    torque_reward_scale = -0.001
    
    # Foot clearance penalty (rear feet only during swing)
    feet_clearance_reward_scale = -5.0
    
    # Contact forces shaped
    tracking_contacts_shaped_force_reward_scale = 0.5
    
    # Air time rewards/penalties - reduced
    air_time_penalty_scale = -10.0
    air_time_reward_scale = 2.0
    
    # Penalize front leg ground contact STRONGLY
    front_leg_contact_penalty_scale = -5.0
    
    # Stability - gentle penalties
    orient_reward_scale = -0.5
    lin_vel_z_reward_scale = -0.1
    ang_vel_xy_reward_scale = -0.02
    
    # Actuator friction model (BONUS) - DISABLED for easier learning
    enable_actuator_friction = False  # Enable after robot learns basic bipedal
    friction_viscous_range = (0.0, 0.1)   # Reduced
    friction_stiction_range = (0.0, 0.5)  # Reduced for bipedal

