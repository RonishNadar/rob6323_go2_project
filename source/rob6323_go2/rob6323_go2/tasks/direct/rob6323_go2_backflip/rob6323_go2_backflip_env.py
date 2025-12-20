# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# BONUS TASK: Backflip with Recovery Environment for Unitree Go2
#
# The robot learns to:
# 1. Jump high into the air
# 2. Rotate backward (pitch) 360 degrees
# 3. Land on all four feet
# 4. Recover to stable standing

from __future__ import annotations

import gymnasium as gym
import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils

from .rob6323_go2_backflip_env_cfg import Rob6323Go2BackflipEnvCfg


class Rob6323Go2BackflipEnv(DirectRLEnv):
    """Backflip with recovery environment for Unitree Go2."""
    
    cfg: Rob6323Go2BackflipEnvCfg

    def __init__(self, cfg: Rob6323Go2BackflipEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Actions
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        
        # Action history
        self.last_actions = torch.zeros(self.num_envs, 12, 2, dtype=torch.float, device=self.device)

        # PD control
        self.Kp = torch.tensor([cfg.Kp] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.Kd = torch.tensor([cfg.Kd] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.torque_limits = cfg.torque_limits
        self.desired_joint_pos = torch.zeros(self.num_envs, 12, device=self.device)
        self.applied_torques = torch.zeros(self.num_envs, 12, device=self.device)

        # Backflip tracking
        # Cumulative pitch rotation (radians) - tracks total rotation
        self.cumulative_pitch = torch.zeros(self.num_envs, device=self.device)
        self.previous_pitch = torch.zeros(self.num_envs, device=self.device)
        
        # Maximum height reached during episode
        self.max_height_reached = torch.zeros(self.num_envs, device=self.device)
        
        # Phase tracking (0=ground, 1=airborne, 2=landed)
        self.flip_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Track if flip was successful
        self.flip_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Initial base height for reference
        self.initial_base_height = torch.zeros(self.num_envs, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "jump_height",
                "pitch_velocity",
                "rotation_progress",
                "landing_upright",
                "landing_feet_contact",
                "recovery_stable",
                "action_rate",
                "torque",
                "collision_penalty",
            ]
        }
        
        # Body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        
        # Foot indices
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self._feet_ids_sensor = []
        for foot_name in foot_names:
            feet_idx, _ = self._contact_sensor.find_bodies(foot_name)
            self._feet_ids_sensor.append(feet_idx[0])
        self._feet_ids_sensor = torch.tensor(self._feet_ids_sensor, device=self.device, dtype=torch.long)

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
        
        torques = torch.clip(torques_pd, -self.torque_limits, self.torque_limits)
        self.applied_torques = torques.clone()
        self.robot.set_joint_effort_target(torques)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        
        # Update backflip tracking
        self._update_flip_tracking()
        
        # Normalize phase to [0, 1] and cumulative pitch to [-1, 1] range
        phase_normalized = self.flip_phase.float() / 2.0
        pitch_normalized = self.cumulative_pitch / (2 * math.pi)  # Normalize by full rotation
        
        obs = torch.cat([
            self.robot.data.root_lin_vel_b,      # 3
            self.robot.data.root_ang_vel_b,      # 3
            self.robot.data.projected_gravity_b, # 3
            self.robot.data.joint_pos - self.robot.data.default_joint_pos,  # 12
            self.robot.data.joint_vel,           # 12
            self._actions,                        # 12
            phase_normalized.unsqueeze(1),        # 1
            pitch_normalized.unsqueeze(1),        # 1
        ], dim=-1)
        
        return {"policy": obs}

    def _update_flip_tracking(self):
        """Track the backflip progress."""
        # Get current base height
        base_height = self.robot.data.root_pos_w[:, 2]
        
        # Update max height
        self.max_height_reached = torch.max(self.max_height_reached, base_height)
        
        # Get current pitch angle from quaternion
        # Extract pitch (rotation around y-axis) from quaternion
        quat = self.robot.data.root_quat_w
        # Pitch = atan2(2(qw*qy + qx*qz), 1 - 2(qy^2 + qz^2)) -- but we use angular velocity
        
        # Use angular velocity for smoother tracking
        pitch_vel = self.robot.data.root_ang_vel_b[:, 1]  # Pitch angular velocity (y-axis in body)
        
        # Accumulate pitch rotation
        self.cumulative_pitch += pitch_vel * self.step_dt
        
        # Check if feet are in contact
        contact_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :]
        feet_in_contact = torch.sum((torch.norm(contact_forces, dim=-1) > 10.0).float(), dim=1)
        
        # Update phase
        # Phase 0: On ground (preparing or recovering)
        # Phase 1: Airborne (jumping/flipping)
        # Phase 2: Landed after flip
        
        # Transition to airborne when all feet leave ground and height increases
        just_launched = (feet_in_contact < 2) & (base_height > self.initial_base_height + 0.1) & (self.flip_phase == 0)
        self.flip_phase = torch.where(just_launched, torch.ones_like(self.flip_phase), self.flip_phase)
        
        # Transition to landed when feet touch ground after being airborne
        just_landed = (feet_in_contact >= 2) & (self.flip_phase == 1)
        self.flip_phase = torch.where(just_landed, 2 * torch.ones_like(self.flip_phase), self.flip_phase)
        
        # Check if flip was completed (rotated at least 270 degrees = 3/4 of full flip)
        self.flip_completed = self.cumulative_pitch < -1.5 * math.pi  # Backward = negative

    def _reward_jump_height(self) -> torch.Tensor:
        """Reward for jumping high."""
        base_height = self.robot.data.root_pos_w[:, 2]
        target = self.cfg.target_jump_height
        
        # Exponential reward centered on target
        height_reward = torch.exp(-torch.square(base_height - target) / 0.1)
        
        # Also reward any height above initial
        height_bonus = torch.clamp(base_height - self.initial_base_height, 0, 1.0)
        
        return height_reward + height_bonus

    def _reward_pitch_velocity(self) -> torch.Tensor:
        """Reward for backward pitch rotation (negative pitch velocity)."""
        pitch_vel = self.robot.data.root_ang_vel_b[:, 1]
        target = self.cfg.target_pitch_velocity
        
        # Only reward during airborne phase
        airborne = (self.flip_phase == 1).float()
        
        # Reward for matching target pitch velocity
        vel_reward = torch.exp(-torch.square(pitch_vel - target) / 10.0)
        
        # Also reward any backward rotation
        backward_bonus = torch.clamp(-pitch_vel / 10.0, 0, 1.0)
        
        return airborne * (vel_reward + backward_bonus)

    def _reward_rotation_progress(self) -> torch.Tensor:
        """Reward for cumulative rotation progress."""
        # Normalize by target rotation
        progress = torch.clamp(-self.cumulative_pitch / (-self.cfg.target_rotation), 0, 1.5)
        
        # Big bonus for completing the flip
        completed_bonus = self.flip_completed.float() * 2.0
        
        return progress + completed_bonus

    def _reward_landing_upright(self) -> torch.Tensor:
        """Reward for landing upright after flip."""
        # Only reward in landed phase
        landed = (self.flip_phase == 2).float()
        
        # Check if upright (gravity should point down in body frame)
        gravity_z = self.robot.data.projected_gravity_b[:, 2]
        upright = (gravity_z < -0.8).float()  # Upright when gravity points down
        
        return landed * upright

    def _reward_landing_feet_contact(self) -> torch.Tensor:
        """Reward for landing on feet."""
        landed = (self.flip_phase == 2).float()
        
        contact_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :]
        feet_in_contact = torch.sum((torch.norm(contact_forces, dim=-1) > 5.0).float(), dim=1)
        
        # Reward for all 4 feet in contact
        return landed * (feet_in_contact / 4.0)

    def _reward_recovery_stable(self) -> torch.Tensor:
        """Reward for stable recovery after landing."""
        landed = (self.flip_phase == 2).float()
        
        # Low velocity = stable
        lin_vel_magnitude = torch.norm(self.robot.data.root_lin_vel_b, dim=1)
        ang_vel_magnitude = torch.norm(self.robot.data.root_ang_vel_b, dim=1)
        
        stable = torch.exp(-(lin_vel_magnitude + ang_vel_magnitude) / 2.0)
        
        return landed * stable

    def _reward_collision_penalty(self) -> torch.Tensor:
        """Penalty for body (not feet) hitting ground."""
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        base_contact = torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0]
        
        # Squeeze any extra dimensions
        if base_contact.dim() > 1:
            base_contact = base_contact.squeeze(-1)
        
        # Penalty if base contacts ground hard
        collision = (base_contact > 50.0).float()
        
        return collision

    def _get_rewards(self) -> torch.Tensor:
        # Jump height reward
        rew_jump_height = self._reward_jump_height()
        
        # Pitch velocity reward
        rew_pitch_velocity = self._reward_pitch_velocity()
        
        # Rotation progress reward
        rew_rotation_progress = self._reward_rotation_progress()
        
        # Landing rewards
        rew_landing_upright = self._reward_landing_upright()
        rew_landing_feet = self._reward_landing_feet_contact()
        rew_recovery = self._reward_recovery_stable()
        
        # Penalties
        rew_action_rate = torch.sum(
            torch.square(self._actions - self.last_actions[:, :, 0]), dim=1
        ) * (self.cfg.action_scale ** 2)
        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions
        
        rew_torque = torch.sum(torch.square(self.applied_torques), dim=1)
        rew_collision = self._reward_collision_penalty()
        
        rewards = {
            "jump_height": rew_jump_height * self.cfg.jump_height_reward_scale,
            "pitch_velocity": rew_pitch_velocity * self.cfg.pitch_velocity_reward_scale,
            "rotation_progress": rew_rotation_progress * self.cfg.rotation_progress_reward_scale,
            "landing_upright": rew_landing_upright * self.cfg.landing_upright_reward_scale,
            "landing_feet_contact": rew_landing_feet * self.cfg.landing_feet_contact_reward_scale,
            "recovery_stable": rew_recovery * self.cfg.recovery_stable_reward_scale,
            "action_rate": rew_action_rate * self.cfg.action_rate_reward_scale,
            "torque": rew_torque * self.cfg.torque_reward_scale,
            "collision_penalty": rew_collision * self.cfg.collision_penalty_scale,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        for key, value in rewards.items():
            self._episode_sums[key] += value
            
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Terminate if base height too low (crashed)
        base_height = self.robot.data.root_pos_w[:, 2]
        crashed = base_height < self.cfg.base_height_min
        
        # Success termination: completed flip and stable
        success = self.flip_completed & (self.flip_phase == 2)
        
        # Check for upside down too long
        gravity_z = self.robot.data.projected_gravity_b[:, 2]
        upside_down = gravity_z > 0.5
        
        died = crashed  # Don't terminate for upside down during flip
        
        return died, time_out | success

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
            
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )
            
        # Reset actions
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        
        # Reset flip tracking
        self.cumulative_pitch[env_ids] = 0.0
        self.previous_pitch[env_ids] = 0.0
        self.max_height_reached[env_ids] = 0.0
        self.flip_phase[env_ids] = 0
        self.flip_completed[env_ids] = False
        
        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Store initial height
        self.initial_base_height[env_ids] = default_root_state[:, 2]
        
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        # Track success rate
        extras = dict()
        extras["Episode_Termination/crashed"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/success_or_timeout"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

