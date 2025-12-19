# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import math
import torch
import numpy as np
import numpy as np
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

# Import ControlFlags to toggle logic
from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg, ControlFlags


class Rob6323Go2Env(DirectRLEnv):
    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Actions & Commands
        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # variables needed for action rate penalization
        # Shape: (num_envs, action_dim, history_length)
        self.last_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), 3, dtype=torch.float, device=self.device, requires_grad=False)

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # BUFFER INITIALIZATION
        # Tutorial Part 1: History Buffer
        if ControlFlags.ENABLE_ACTION_HISTORY:
            self.last_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), 3, dtype=torch.float, device=self.device, requires_grad=False)

        # Tutorial Part 2: PD Parameters
        if ControlFlags.ENABLE_MANUAL_PD:
            self.Kp = torch.tensor([cfg.Kp] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            self.Kd = torch.tensor([cfg.Kd] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            self.torque_limits = cfg.torque_limits

        # Bonus Task 1: Friction Parameters
        if ControlFlags.ENABLE_FRICTION_MODEL:
            self.friction_coeffs_viscous = torch.zeros(self.num_envs, 12, device=self.device)
            self.friction_coeffs_static = torch.zeros(self.num_envs, 12, device=self.device)

        # Tutorial Part 4: Raibert Heuristic Indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        if ControlFlags.ENABLE_RAIBERT_HEURISTIC:
            self._feet_ids = []
            foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            for name in foot_names:
                id_list, _ = self.robot.find_bodies(name)
                self._feet_ids.append(id_list[0])
            
            self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
            self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        # Bonus Task 2: Bipedal Indices
        # We need to find the REAR feet in the SENSOR to penalize contact (for Front-Leg Walking)
        if ControlFlags.ENABLE_BIPEDAL:
            self._rear_feet_ids_sensor = []
            for name in ["RL_foot", "RR_foot"]:
                id_list, _ = self._contact_sensor.find_bodies(name)
                self._rear_feet_ids_sensor.append(id_list[0])

        # LOGGING SETUP
        log_keys = ["track_lin_vel_xy_exp", "track_ang_vel_z_exp"]
        if ControlFlags.ENABLE_ACTION_HISTORY: log_keys.append("rew_action_rate")
        if ControlFlags.ENABLE_RAIBERT_HEURISTIC: log_keys.append("raibert_heuristic")
        if ControlFlags.ENABLE_STABILITY_REWARDS: log_keys.extend(["orient", "lin_vel_z", "dof_vel", "ang_vel_xy"])
        if ControlFlags.ENABLE_TORQUE_PENALTY: log_keys.append("torque")
        if ControlFlags.ENABLE_BIPEDAL: log_keys.append("bipedal_contact")

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in log_keys
        }

        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    # Helper property for Raibert Heuristic
    @property
    def foot_positions_w(self) -> torch.Tensor:
        if hasattr(self, "_feet_ids"):
            return self.robot.data.body_pos_w[:, self._feet_ids]
        return None

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        
        # Calculate desired positions (used for Manual PD)
        self.desired_joint_pos = (self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos)

        # fallback check
        if not ControlFlags.ENABLE_MANUAL_PD:
            # If manual PD is off, we must set _processed_actions for the original logic
            self._processed_actions = self.desired_joint_pos

    def _apply_action(self) -> None:
        if ControlFlags.ENABLE_MANUAL_PD:
            # Tutorial Part 2 & Bonus 1
            tau_pd = (
                self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos)
                - self.Kd * self.robot.data.joint_vel
            )
            
            if ControlFlags.ENABLE_FRICTION_MODEL:
                tau_friction = (
                    self.friction_coeffs_static * torch.tanh(self.robot.data.joint_vel / 0.1)
                    + self.friction_coeffs_viscous * self.robot.data.joint_vel
                )
                torques = tau_pd - tau_friction
            else:
                torques = tau_pd

            torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
            self.robot.set_joint_effort_target(torques)
        else:
            self.robot.set_joint_position_target(self._processed_actions)

    # Helper for Tutorial Part 4 (Raibert)
    def _step_contact_targets(self):
        frequencies = 3.
        phases = 0.5
        offsets = 0.
        bounds = 0.
        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)
        foot_indices = [self.gait_indices + phases + offsets + bounds,
                        self.gait_indices + offsets,
                        self.gait_indices + bounds,
                        self.gait_indices + phases]
        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)
        
        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

    # Helper for Tutorial Part 4 (Raibert)
    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(math_utils.quat_conjugate(self.robot.data.root_quat_w),
                                                            cur_footsteps_translated[:, i, :])
        desired_stance_width = 0.25
        desired_stance_length = 0.45
        desired_ys_nom = torch.tensor([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.device).unsqueeze(0)
        desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.device).unsqueeze(0)

        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = torch.tensor([3.0], device=self.device)
        x_vel_des = self._commands[:, 0:1]
        yaw_vel_des = self._commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))
        
        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset
        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)
        
        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])
        return torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        
        obs_list = [
            self.robot.data.root_lin_vel_b,
            self.robot.data.root_ang_vel_b,
            self.robot.data.projected_gravity_b,
            self._commands,
            self.robot.data.joint_pos - self.robot.data.default_joint_pos,
            self.robot.data.joint_vel,
            self._actions,
        ]
        
        # Tutorial Part 4: Add Clock Inputs
        if ControlFlags.ENABLE_RAIBERT_HEURISTIC:
            obs_list.append(self.clock_inputs)

        obs = torch.cat([tensor for tensor in obs_list if tensor is not None], dim=-1)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # 1. Linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # 2. Yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        
        # 3. Action rate penalization
        rew_action_rate = torch.sum(torch.square(self._actions - self.last_actions[:, :, 0]), dim=1) * (self.cfg.action_scale ** 2)
        rew_action_rate += torch.sum(torch.square(self._actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1) * (self.cfg.action_scale ** 2)

        # Update the prev action hist
        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions[:]

        # 4. Raibert Heuristic Reward
        self._step_contact_targets() 
        rew_raibert_heuristic = self._reward_raibert_heuristic()
        
        # 5. Refining Rewards
        rew_orient = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        rew_lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        rew_dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)
        rew_ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)
        
        # 6. Torque penalty (including friction effects)
        tau_pd = (self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos) - self.Kd * self.robot.data.joint_vel)
        tau_friction = (self.friction_coeffs_static * torch.tanh(self.robot.data.joint_vel / 0.1) + self.friction_coeffs_viscous * self.robot.data.joint_vel)
        torques = torch.clip(tau_pd - tau_friction, -self.torque_limits, self.torque_limits)
        rew_torque = torch.sum(torch.square(torques), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale,
        }

        # Tutorial Part 1: Action Rate Penalty
        if ControlFlags.ENABLE_ACTION_HISTORY:
            rew_action_rate = torch.sum(torch.square(self._actions - self.last_actions[:, :, 0]), dim=1) * (self.cfg.action_scale ** 2)
            rew_action_rate += torch.sum(torch.square(self._actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1) * (self.cfg.action_scale ** 2)
            rewards["rew_action_rate"] = rew_action_rate * self.cfg.action_rate_reward_scale
            # Update history
            self.last_actions = torch.roll(self.last_actions, 1, 2)
            self.last_actions[:, :, 0] = self._actions[:]

        # Tutorial Part 4: Raibert Heuristic
        if ControlFlags.ENABLE_RAIBERT_HEURISTIC:
            self._step_contact_targets() 
            rewards["raibert_heuristic"] = self._reward_raibert_heuristic() * self.cfg.raibert_heuristic_reward_scale

        # Rubric: Base Stability
        if ControlFlags.ENABLE_STABILITY_REWARDS:
            # Special Handling for Bipedal Mode:
            if ControlFlags.ENABLE_BIPEDAL:
                # Target: Pitch DOWN ~45 deg (Standing on FRONT legs)
                # Gravity vector in body frame (Positive X = Forward/Down)
                # Target Vector: [0.707, 0.0, -0.707] (Gravity pulls forward)
                target_g = torch.tensor([0.707, 0.0, -0.707], device=self.device)
                rewards["orient"] = torch.sum(torch.square(self.robot.data.projected_gravity_b - target_g), dim=1) * self.cfg.orient_reward_scale
            else:
                # Standard: Penalize non-flat orientation
                rewards["orient"] = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1) * self.cfg.orient_reward_scale
            
            rewards["lin_vel_z"] = torch.square(self.robot.data.root_lin_vel_b[:, 2]) * self.cfg.lin_vel_z_reward_scale
            rewards["dof_vel"] = torch.sum(torch.square(self.robot.data.joint_vel), dim=1) * self.cfg.dof_vel_reward_scale
            rewards["ang_vel_xy"] = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1) * self.cfg.ang_vel_xy_reward_scale

        # Rubric: Action Regularization (Torque Penalty)
        if ControlFlags.ENABLE_TORQUE_PENALTY and ControlFlags.ENABLE_MANUAL_PD:
            # We re-calculate the actually applied torque (PD - Friction) for the penalty
            tau_pd = self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos) - self.Kd * self.robot.data.joint_vel
            if ControlFlags.ENABLE_FRICTION_MODEL:
                tau_friction = self.friction_coeffs_static * torch.tanh(self.robot.data.joint_vel / 0.1) + self.friction_coeffs_viscous * self.robot.data.joint_vel
                applied_torques = torch.clip(tau_pd - tau_friction, -self.torque_limits, self.torque_limits)
            else:
                applied_torques = torch.clip(tau_pd, -self.torque_limits, self.torque_limits)
            rewards["torque"] = torch.sum(torch.square(applied_torques), dim=1) * self.cfg.torque_reward_scale
        
        # Bonus Task 2: Bipedal Contact Penalty
        if ControlFlags.ENABLE_BIPEDAL:
            # Penalize any contact force on the REAR feet
            # This encourages the policy to lift the REAR legs and walk on the FRONT.
            rear_forces = torch.norm(self._contact_sensor.data.net_forces_w[:, self._rear_feet_ids_sensor], dim=-1)
            # Sum forces on RL/RR and apply penalty
            rewards["bipedal_contact"] = torch.sum(rear_forces, dim=1) * self.cfg.bipedal_contact_reward_scale

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            if key in self._episode_sums:
                self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        cstr_termination_contacts = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0
        
        # Handle Bipedal Exception
        if ControlFlags.ENABLE_BIPEDAL:
            # Upside down check might trigger if we pitch down too much (>90 deg). 
            died = cstr_termination_contacts | cstr_upsidedown
        else:
            died = cstr_termination_contacts | cstr_upsidedown
        
        # Tutorial Part 3: Early Termination (Height)
        if ControlFlags.ENABLE_HEIGHT_TERMINATION:
            base_height = self.robot.data.root_pos_w[:, 2]
            died = died | (base_height < self.cfg.base_height_min)
            
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        
        # Reset Buffers
        if ControlFlags.ENABLE_ACTION_HISTORY:
            self.last_actions[env_ids] = 0.
        if ControlFlags.ENABLE_RAIBERT_HEURISTIC:
            self.gait_indices[env_ids] = 0

        # Bonus 1: Randomize Friction
        if ControlFlags.ENABLE_FRICTION_MODEL:
            mu_v_low, mu_v_high = self.cfg.friction_range_viscous
            Fs_low, Fs_high = self.cfg.friction_range_static
            mu_v_sample = torch.rand(len(env_ids), 1, device=self.device) * (mu_v_high - mu_v_low) + mu_v_low
            Fs_sample = torch.rand(len(env_ids), 1, device=self.device) * (Fs_high - Fs_low) + Fs_low
            self.friction_coeffs_viscous[env_ids] = mu_v_sample.repeat(1, 12)
            self.friction_coeffs_static[env_ids] = Fs_sample.repeat(1, 12)
        
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
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        
        return arrow_scale, arrow_quat