# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# BONUS TASK: Bipedal Walking Environment for Unitree Go2
# Reference: https://arxiv.org/pdf/2509.00215v2 (DMO Paper)
#
# Key concepts from DMO paper:
# - Robot should stand on rear legs with front legs lifted
# - v_f: robot's forward axis in world frame
# - v_u: target upright direction (yaw-rotated world vector)
# - 1_stand: indicator when robot is sufficiently upright (v_f · v_u / ||v_u|| > 0.9)

from __future__ import annotations

import gymnasium as gym
import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from .rob6323_go2_bipedal_env_cfg import Rob6323Go2BipedalEnvCfg


class Rob6323Go2BipedalEnv(DirectRLEnv):
    """Bipedal walking environment for Unitree Go2.
    
    The robot learns to walk on its rear legs with front legs lifted,
    similar to a standing/walking pose.
    """
    
    cfg: Rob6323Go2BipedalEnvCfg

    def __init__(self, cfg: Rob6323Go2BipedalEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Actions
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        
        # Action history for action rate penalty
        self.last_actions = torch.zeros(self.num_envs, 12, 3, dtype=torch.float, device=self.device)
        
        # Joint velocity history for acceleration penalty
        self.last_joint_vel = torch.zeros(self.num_envs, 12, device=self.device)

        # Commands: [vx, vy, yaw_rate] - but for bipedal, we mainly care about forward motion
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # PD control parameters
        self.Kp = torch.tensor([cfg.Kp] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.Kd = torch.tensor([cfg.Kd] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.torque_limits = cfg.torque_limits
        self.desired_joint_pos = torch.zeros(self.num_envs, 12, device=self.device)
        self.applied_torques = torch.zeros(self.num_envs, 12, device=self.device)
        
        # Actuator friction parameters
        self.friction_viscous = torch.zeros(self.num_envs, 12, device=self.device)
        self.friction_stiction = torch.zeros(self.num_envs, 12, device=self.device)

        # Gait phase tracking (only for rear legs in bipedal)
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        
        # Contact states - for bipedal, only rear legs should contact
        # [FL, FR, RL, RR] - FL and FR should always be 0 (swing/lifted)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        
        # Foot positions
        self.foot_positions_w = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device)

        # Upright direction vector (from DMO paper)
        # v_u = R_z(θ) * [0.2, 0, -1.0] - target direction robot should face
        # For standing upright, we want the robot's forward axis to point upward
        self.upright_threshold = 0.9  # Threshold for 1_stand indicator

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_y_penalty",
                "upright_reward",
                "base_height",
                "action_rate",
                "joint_accel",
                "torque",
                "feet_clearance",
                "tracking_contacts",
                "air_time",
                "front_leg_lift",
                "front_leg_contact_penalty",
                "orient",
                "lin_vel_z",
                "ang_vel_xy",
            ]
        }
        
        # Body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        
        # Foot indices
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self._feet_ids = []
        for foot_name in foot_names:
            feet_idx, _ = self.robot.find_bodies(foot_name)
            self._feet_ids.append(feet_idx[0])
        self._feet_ids = torch.tensor(self._feet_ids, device=self.device, dtype=torch.long)
        
        self._feet_ids_sensor = []
        for foot_name in foot_names:
            feet_idx, _ = self._contact_sensor.find_bodies(foot_name)
            self._feet_ids_sensor.append(feet_idx[0])
        self._feet_ids_sensor = torch.tensor(self._feet_ids_sensor, device=self.device, dtype=torch.long)
        
        # Front leg indices (FL=0, FR=1) and rear leg indices (RL=2, RR=3)
        self.front_leg_indices = [0, 1]
        self.rear_leg_indices = [2, 3]

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        self.desired_joint_pos = (
            self.cfg.action_scale * self._actions 
            + self.robot.data.default_joint_pos
        )

    def _apply_action(self) -> None:
        # PD control
        torques_pd = (
            self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos)
            - self.Kd * self.robot.data.joint_vel
        )
        
        # Apply actuator friction
        if self.cfg.enable_actuator_friction:
            joint_vel = self.robot.data.joint_vel
            tau_stiction = self.friction_stiction * torch.tanh(joint_vel / 0.1)
            tau_viscous = self.friction_viscous * joint_vel
            torques_pd = torques_pd - (tau_stiction + tau_viscous)
        
        torques = torch.clip(torques_pd, -self.torque_limits, self.torque_limits)
        self.applied_torques = torques.clone()
        self.robot.set_joint_effort_target(torques)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        self._update_foot_positions()
        
        # Observations based on DMO paper Table 4 (for actor)
        # Note: We don't include base linear velocity (only for dynamics model)
        obs = torch.cat([
            self.robot.data.root_ang_vel_b,      # 3 - angular velocity
            self.robot.data.projected_gravity_b, # 3 - projected gravity
            self.robot.data.joint_pos - self.robot.data.default_joint_pos,  # 12
            self.robot.data.joint_vel,           # 12
            self._actions,                        # 12
            self.clock_inputs,                    # 4
        ], dim=-1)
        
        return {"policy": obs}

    def _update_foot_positions(self):
        for i, foot_id in enumerate(self._feet_ids):
            self.foot_positions_w[:, i, :] = self.robot.data.body_pos_w[:, foot_id, :]

    def _step_contact_targets_bipedal(self):
        """Update gait phase for bipedal - only rear legs alternate."""
        frequencies = torch.ones(self.num_envs, device=self.device) * self.cfg.gait_frequency
        
        self.gait_indices = torch.fmod(
            self.gait_indices + self.step_dt * frequencies, 1.0
        )
        
        # For bipedal: front legs always in swing (lifted), rear legs alternate
        # RL and RR are 180 degrees out of phase for bipedal walking
        foot_indices = torch.zeros(self.num_envs, 4, device=self.device)
        foot_indices[:, 0] = 0.0  # FL - always swing
        foot_indices[:, 1] = 0.0  # FR - always swing
        foot_indices[:, 2] = self.gait_indices  # RL
        foot_indices[:, 3] = torch.fmod(self.gait_indices + 0.5, 1.0)  # RR (180 deg offset)
        
        # Clock inputs for rear legs
        self.clock_inputs[:, 0] = torch.sin(2 * math.pi * foot_indices[:, 2])
        self.clock_inputs[:, 1] = torch.cos(2 * math.pi * foot_indices[:, 2])
        self.clock_inputs[:, 2] = torch.sin(2 * math.pi * foot_indices[:, 3])
        self.clock_inputs[:, 3] = torch.cos(2 * math.pi * foot_indices[:, 3])
        
        # Desired contact states
        # Front legs: always 0 (swing/lifted)
        # Rear legs: stance during first half of gait cycle
        self.desired_contact_states[:, 0] = 0.0  # FL - always swing
        self.desired_contact_states[:, 1] = 0.0  # FR - always swing
        self.desired_contact_states[:, 2] = (foot_indices[:, 2] < 0.5).float()  # RL
        self.desired_contact_states[:, 3] = (foot_indices[:, 3] < 0.5).float()  # RR

    def _compute_upright_reward(self) -> torch.Tensor:
        """Compute upright posture reward from DMO paper.
        
        v_f: robot's forward axis in world frame (x-axis of body frame)
        v_u: target upright direction = R_z(θ) * [0.2, 0, -1.0]
        
        Reward = k_up * (0.5 * v_f · v_n / ||v_n|| + 0.5)^2
        where v_n is normalized v_u
        """
        # Get robot's forward axis in world frame
        # Body frame x-axis rotated to world frame
        body_x_axis = torch.zeros(self.num_envs, 3, device=self.device)
        body_x_axis[:, 0] = 1.0  # x-axis in body frame
        
        # Rotate to world frame
        v_f = math_utils.quat_rotate(self.robot.data.root_quat_w, body_x_axis)
        
        # Target upright direction: for bipedal, we want the robot's x-axis to point upward
        # v_u = [0.2, 0, -1.0] rotated by yaw - but simplified, we just want x-axis to be vertical
        # Actually, for bipedal standing, the robot's x-axis (forward) should point UP
        v_u = torch.zeros(self.num_envs, 3, device=self.device)
        v_u[:, 0] = 0.2   # slight forward lean
        v_u[:, 2] = 1.0   # mostly pointing up (changed from -1 to 1 for upright)
        
        # Normalize v_u
        v_n = v_u / torch.norm(v_u, dim=1, keepdim=True)
        
        # Dot product
        dot_product = torch.sum(v_f * v_n, dim=1)
        
        # Reward: (0.5 * dot_product + 0.5)^2
        upright_reward = torch.square(0.5 * dot_product + 0.5)
        
        return upright_reward

    def _compute_stand_indicator(self) -> torch.Tensor:
        """Compute 1_stand indicator from DMO paper.
        
        1_stand = 1 if v_f · v_u / ||v_u|| > 0.9, else 0
        """
        body_x_axis = torch.zeros(self.num_envs, 3, device=self.device)
        body_x_axis[:, 0] = 1.0
        v_f = math_utils.quat_rotate(self.robot.data.root_quat_w, body_x_axis)
        
        v_u = torch.zeros(self.num_envs, 3, device=self.device)
        v_u[:, 0] = 0.2
        v_u[:, 2] = 1.0
        v_n = v_u / torch.norm(v_u, dim=1, keepdim=True)
        
        dot_product = torch.sum(v_f * v_n, dim=1)
        
        return (dot_product > self.upright_threshold).float()

    def _reward_front_leg_lift(self) -> torch.Tensor:
        """Reward for lifting front legs high (for bipedal pose)."""
        # Get front foot heights
        fl_height = self.foot_positions_w[:, 0, 2]  # FL
        fr_height = self.foot_positions_w[:, 1, 2]  # FR
        
        target_height = self.cfg.front_leg_target_height
        
        # Reward for getting close to target height
        # Using exponential reward for smooth gradient
        fl_reward = torch.exp(-torch.square(fl_height - target_height) / 0.05)
        fr_reward = torch.exp(-torch.square(fr_height - target_height) / 0.05)
        
        # Also reward any height (encourage lifting)
        height_bonus = torch.clamp(fl_height + fr_height, 0, 0.6) / 0.6
        
        return (fl_reward + fr_reward) / 2 + height_bonus

    def _reward_base_height(self) -> torch.Tensor:
        """Reward for maintaining target base height (higher for bipedal)."""
        base_height = self.robot.data.root_pos_w[:, 2]
        target_height = self.cfg.base_height_target
        
        # Exponential reward centered on target
        height_reward = torch.exp(-torch.square(base_height - target_height) / 0.02)
        
        # Also give bonus for any height above minimum
        height_bonus = torch.clamp((base_height - 0.25) / 0.25, 0, 1)
        
        return height_reward + height_bonus

    def _reward_front_leg_contact_penalty(self) -> torch.Tensor:
        """Penalize front legs touching the ground."""
        contact_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor[:2], :]
        contact_force_magnitudes = torch.norm(contact_forces, dim=-1)
        
        # Strong penalty if front legs are in contact
        front_contact = torch.sum((contact_force_magnitudes > 1.0).float(), dim=1)
        
        return front_contact

    def _reward_rear_feet_clearance(self) -> torch.Tensor:
        """Foot clearance reward only for rear legs during swing."""
        foot_heights = self.foot_positions_w[:, :, 2]
        desired_clearance = 0.03  # 3cm clearance for rear feet
        
        # Only penalize rear feet (indices 2, 3) during swing
        swing_mask = (self.desired_contact_states < 0.5).float()
        
        clearance_error = torch.zeros_like(foot_heights)
        # Only rear feet (indices 2, 3)
        for i in self.rear_leg_indices:
            clearance_error[:, i] = swing_mask[:, i] * torch.square(
                foot_heights[:, i] - desired_clearance
            )
        
        return torch.sum(clearance_error, dim=1)

    def _reward_tracking_contacts_bipedal(self) -> torch.Tensor:
        """Contact tracking for bipedal - front legs should never contact."""
        contact_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :]
        contact_force_magnitudes = torch.norm(contact_forces, dim=-1)
        
        force_threshold = 1.0
        in_contact = (contact_force_magnitudes > force_threshold).float()
        
        # Reward for matching desired contact:
        # - Front legs (0, 1): should NOT be in contact (desired = 0)
        # - Rear legs (2, 3): should match gait phase
        stance_mask = (self.desired_contact_states > 0.5).float()
        
        # Correct contact matching
        correct_contact = stance_mask * in_contact + (1.0 - stance_mask) * (1.0 - in_contact)
        
        return torch.sum(correct_contact, dim=1)

    def _reward_air_time_bipedal(self) -> torch.Tensor:
        """Air time reward/penalty for rear legs."""
        # Penalize rear feet being in air too long during stance, reward during swing
        contact_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :]
        contact_force_magnitudes = torch.norm(contact_forces, dim=-1)
        
        in_contact = (contact_force_magnitudes > 1.0).float()
        stance_mask = (self.desired_contact_states > 0.5).float()
        
        # For rear legs only
        air_penalty = 0.0
        air_reward = 0.0
        
        for i in self.rear_leg_indices:
            # Penalty: in stance phase but not in contact
            air_penalty += stance_mask[:, i] * (1.0 - in_contact[:, i])
            # Reward: in swing phase and not in contact
            air_reward += (1.0 - stance_mask[:, i]) * (1.0 - in_contact[:, i])
        
        # Combined: negative for penalty, positive for reward
        return air_reward * self.cfg.air_time_reward_scale + air_penalty * self.cfg.air_time_penalty_scale

    def _get_rewards(self) -> torch.Tensor:
        # Update gait for bipedal
        self._step_contact_targets_bipedal()
        
        # Get stand indicator for conditional rewards
        stand_indicator = self._compute_stand_indicator()
        
        # Velocity tracking (only reward when standing)
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25) * stand_indicator
        
        yaw_rate_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25) * stand_indicator
        
        # Penalize lateral velocity
        lin_vel_y_penalty = torch.square(self.robot.data.root_lin_vel_b[:, 1])
        
        # Upright posture reward (CRITICAL for bipedal)
        upright_reward = self._compute_upright_reward()
        
        # Action rate penalty
        rew_action_rate = torch.sum(
            torch.square(self._actions - self.last_actions[:, :, 0]), dim=1
        ) * (self.cfg.action_scale ** 2)
        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions
        
        # Joint acceleration penalty
        joint_accel = (self.robot.data.joint_vel - self.last_joint_vel) / self.step_dt
        rew_joint_accel = torch.sum(torch.square(joint_accel), dim=1)
        self.last_joint_vel = self.robot.data.joint_vel.clone()
        
        # Torque penalty
        rew_torque = torch.sum(torch.square(self.applied_torques), dim=1)
        
        # Foot clearance (rear feet only)
        rew_feet_clearance = self._reward_rear_feet_clearance()
        
        # Contact tracking
        rew_tracking_contacts = self._reward_tracking_contacts_bipedal()
        
        # Air time reward
        rew_air_time = self._reward_air_time_bipedal()
        
        # Front leg lift reward
        rew_front_leg_lift = self._reward_front_leg_lift()
        
        # Base height reward (critical for standing up)
        rew_base_height = self._reward_base_height()
        
        # Front leg contact penalty
        rew_front_leg_contact = self._reward_front_leg_contact_penalty()
        
        # Additional stability penalties
        rew_orient = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        rew_lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        rew_ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)
        
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale,
            "lin_vel_y_penalty": lin_vel_y_penalty * self.cfg.lin_vel_y_penalty_scale,
            "upright_reward": upright_reward * self.cfg.upright_reward_scale,
            "base_height": rew_base_height * self.cfg.base_height_reward_scale,
            "action_rate": rew_action_rate * self.cfg.action_rate_reward_scale,
            "joint_accel": rew_joint_accel * self.cfg.joint_accel_reward_scale,
            "torque": rew_torque * self.cfg.torque_reward_scale,
            "feet_clearance": rew_feet_clearance * self.cfg.feet_clearance_reward_scale,
            "tracking_contacts": rew_tracking_contacts * self.cfg.tracking_contacts_shaped_force_reward_scale,
            "air_time": rew_air_time,  # Already scaled
            "front_leg_lift": rew_front_leg_lift * self.cfg.front_leg_lift_reward_scale,
            "front_leg_contact_penalty": rew_front_leg_contact * self.cfg.front_leg_contact_penalty_scale,
            "orient": rew_orient * self.cfg.orient_reward_scale,
            "lin_vel_z": rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale,
            "ang_vel_xy": rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        for key, value in rewards.items():
            self._episode_sums[key] += value
            
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        cstr_termination_contacts = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
        )
        
        # For bipedal, also terminate if front legs hit ground hard
        front_feet_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor[:2], :]
        front_feet_contact = torch.any(torch.norm(front_feet_forces, dim=-1) > 50.0, dim=1)
        
        cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0.5  # More lenient for bipedal
        
        base_height = self.robot.data.root_pos_w[:, 2]
        cstr_base_height_min = base_height < self.cfg.base_height_min
        
        died = cstr_termination_contacts | cstr_upsidedown | cstr_base_height_min | front_feet_contact
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
            
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )
            
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_joint_vel[env_ids] = 0.0
        self.gait_indices[env_ids] = 0.0
        self.clock_inputs[env_ids] = 0.0
        self.desired_contact_states[env_ids, :2] = 0.0  # Front legs always swing
        self.desired_contact_states[env_ids, 2:] = 1.0  # Rear legs start in stance
        
        # Randomize friction parameters
        if self.cfg.enable_actuator_friction:
            num_reset_envs = len(env_ids) if not isinstance(env_ids, slice) else self.num_envs
            viscous_min, viscous_max = self.cfg.friction_viscous_range
            self.friction_viscous[env_ids] = torch.zeros(
                num_reset_envs, 12, device=self.device
            ).uniform_(viscous_min, viscous_max)
            stiction_min, stiction_max = self.cfg.friction_stiction_range
            self.friction_stiction[env_ids] = torch.zeros(
                num_reset_envs, 12, device=self.device
            ).uniform_(stiction_min, stiction_max)
        
        # Sample commands - for bipedal, start with small/zero velocity commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids])
        # Small forward velocity after standing is learned
        self._commands[env_ids, 0] = torch.zeros(len(env_ids) if not isinstance(env_ids, slice) else self.num_envs, device=self.device).uniform_(0.0, 0.3)
        
        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat

