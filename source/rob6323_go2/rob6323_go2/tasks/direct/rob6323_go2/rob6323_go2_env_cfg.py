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

@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition
    action_scale = 0.25
    action_space = 12
    # Updated observation space: 48 (base) + 4 (clock inputs) = 52
    observation_space = 52
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
    
    # Part 2: PD control gains - disable built-in PD controller
    Kp = 20.0  # Proportional gain
    Kd = 0.5   # Derivative gain
    torque_limits = 100.0  # Max torque
    
    # Update robot actuators to disable implicit PD control
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,  # CRITICAL: Set to 0 to disable implicit P-gain
        damping=0.0,    # CRITICAL: Set to 0 to disable implicit D-gain
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # Part 3: Early stopping - minimum base height
    base_height_min = 0.20  # Terminate if base is lower than 20cm

    # Part 4: Raibert Heuristic configuration
    gait_frequency = 3.0  # Gait frequency in Hz
    raibert_heuristic_reward_scale = -0.5  # REDUCED - was too harsh

    # Part 1: Action rate reward scale
    action_rate_reward_scale = -0.05  # REDUCED for smoother learning

    # Basic reward scales - INCREASED to dominate
    lin_vel_reward_scale = 2.0   # INCREASED - main objective
    yaw_rate_reward_scale = 1.0  # INCREASED

    # Part 5: Additional reward scales - REDUCED penalties
    orient_reward_scale = -1.0         # REDUCED from -5.0
    lin_vel_z_reward_scale = -0.01     # REDUCED
    dof_vel_reward_scale = -0.00005    # REDUCED
    ang_vel_xy_reward_scale = -0.0005  # REDUCED

    # Part 6: Advanced foot interaction reward scales - REDUCED
    feet_clearance_reward_scale = -5.0  # REDUCED from -30.0
    tracking_contacts_shaped_force_reward_scale = 2.0  # REDUCED

    # Action regularization: Torque penalty
    torque_reward_scale = -0.00005  # REDUCED

    # BONUS: Actuator friction model parameters
    # Set to False for easier learning, True for sim-to-real bonus
    enable_actuator_friction = False
    # Randomization ranges (sampled per environment at reset)
    friction_viscous_range = (0.0, 0.1)
    friction_stiction_range = (0.0, 0.5)
