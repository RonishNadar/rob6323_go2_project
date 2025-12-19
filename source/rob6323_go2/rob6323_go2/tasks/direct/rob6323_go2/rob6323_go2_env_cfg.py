# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.actuators import ImplicitActuatorCfg

# CONTROL MACROS
class ControlFlags:
    # Tutorial Part 1: History-based rewards (Action Rate)
    ENABLE_ACTION_HISTORY = True
    
    # Tutorial Part 2: Manual PD Controller (replaces implicit physics PD)
    # Note: If True, actuator stiffness/damping is set to 0.
    ENABLE_MANUAL_PD = True
    
    # Tutorial Part 3: Early termination if base falls too low
    ENABLE_HEIGHT_TERMINATION = True
    
    # Tutorial Part 4: Raibert Heuristic for Gait Shaping
    ENABLE_RAIBERT_HEURISTIC = True
    
    # Rubric: Base Stability Rewards (Orientation, Ang Vel, etc.)
    ENABLE_STABILITY_REWARDS = True
    
    # Rubric: Action Regularization (Torque Penalty)
    ENABLE_TORQUE_PENALTY = True
    
    # Bonus Task 1: Actuator Friction Model (Sim-to-Real)
    ENABLE_FRICTION_MODEL = True
    
    # Bonus Task 2: Bipedal Walking (FRONT Legs)
    ENABLE_BIPEDAL = True

# =============================================================================
# CONFIGURATION CLASS
# =============================================================================

@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition
    action_scale = 0.25
    action_space = 12
    
    # Observation Space Adjustment
    if ControlFlags.ENABLE_RAIBERT_HEURISTIC:
        # Original (48) + Clock Inputs (4)
        observation_space = 52 
    else:
        observation_space = 48
        
    state_space = 0
    debug_vis = True

    # PD Control & Physics Parameters
    if ControlFlags.ENABLE_MANUAL_PD:
        Kp = 20.0  # Proportional gain
        Kd = 0.5   # Derivative gain
        torque_limits = 100.0  # Max torque
        
    # Termination Thresholds
    if ControlFlags.ENABLE_HEIGHT_TERMINATION:
        base_height_min = 0.20

    # Friction Model Ranges (Bonus 1)
    if ControlFlags.ENABLE_FRICTION_MODEL:
        friction_range_viscous = (0.0, 0.3) # mu_v
        friction_range_static = (0.0, 2.5)  # F_s

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
    
    # Disable implicit PD if using Manual PD (Tutorial Part 2)
    if ControlFlags.ENABLE_MANUAL_PD:
        robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=0.0,  # Zero out implicit P
            damping=0.0,    # Zero out implicit D
        )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # =========================================================================
    # REWARD SCALES
    # =========================================================================
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5

    # Tutorial Part 1
    action_rate_reward_scale = -0.1 if ControlFlags.ENABLE_ACTION_HISTORY else 0.0
    
    # Tutorial Part 4
    raibert_heuristic_reward_scale = -10.0 if ControlFlags.ENABLE_RAIBERT_HEURISTIC else 0.0
    
    # Rubric: Base Stability
    orient_reward_scale = -5.0 if ControlFlags.ENABLE_STABILITY_REWARDS else 0.0
    lin_vel_z_reward_scale = -0.02 if ControlFlags.ENABLE_STABILITY_REWARDS else 0.0
    ang_vel_xy_reward_scale = -0.001 if ControlFlags.ENABLE_STABILITY_REWARDS else 0.0
    dof_vel_reward_scale = -0.0001 if ControlFlags.ENABLE_STABILITY_REWARDS else 0.0
    
    # Rubric: Action Regularization
    torque_reward_scale = -0.0001 if ControlFlags.ENABLE_TORQUE_PENALTY else 0.0

    # Bonus Task 2: Bipedal Rewards
    # Penalize touching ground with REAR feet (to encourage front-leg walking)
    bipedal_contact_reward_scale = -5.0 if ControlFlags.ENABLE_BIPEDAL else 0.0