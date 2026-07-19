"""GPU-parallel 1/2/3-link inverted-pendulum environment in Isaac Sim."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from tasks.inverted_pendulum.contract import (
    CART_POSITION_OBS_SCALE,
    CART_VELOCITY_OBS_SCALE,
    OBSERVATION_DIM,
    POLE_VELOCITY_OBS_SCALE,
    TERMINATION_POLE_ANGLE_RAD,
)

if TYPE_CHECKING:
    from .inverted_pendulum_env_cfg import InvertedPendulum1EnvCfg


class SerialInvertedPendulumEnv(DirectRLEnv):
    """Only the cart is actuated; all serial pendulum joints are passive."""

    cfg: InvertedPendulum1EnvCfg

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._cart_dof_idx, _ = self.robot.find_joints("cart_slide")
        self._pole_dof_idx = []
        for index in range(1, self.cfg.order + 1):
            indices, _ = self.robot.find_joints(f"pole_{index}_hinge")
            self._pole_dof_idx.extend(indices)
        self._pole_dof_idx = torch.tensor(
            self._pole_dof_idx, dtype=torch.long, device=self.device
        )
        self.actions = torch.zeros(
            (self.num_envs, 1), dtype=torch.float32, device=self.device
        )
        self.previous_actions = self.actions.clone()
        self.joint_pos = self.robot.data.joint_pos.torch
        self.joint_vel = self.robot.data.joint_vel.torch

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # The benchmark has no free-falling bodies and therefore needs no
        # ground contact.  A plane touching the fixed rail injects a PhysX
        # constraint impulse that is absent from the matched MuJoCo model and
        # is especially destructive for the light serial triple pendulum.
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["cartpole"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.78, 0.82))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(torch.clamp(actions, -1.0, 1.0))

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target_index(
            target=self.cfg.action_force_scale * self.actions,
            joint_ids=self._cart_dof_idx,
        )

    def _relative_angles(self) -> torch.Tensor:
        return self.joint_pos[:, self._pole_dof_idx]

    def _absolute_angles(self) -> torch.Tensor:
        cumulative = torch.cumsum(self._relative_angles(), dim=-1)
        return torch.atan2(torch.sin(cumulative), torch.cos(cumulative))

    def _get_observations(self) -> dict:
        observation = torch.zeros(
            (self.num_envs, OBSERVATION_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        observation[:, 0] = (
            self.joint_pos[:, self._cart_dof_idx[0]] * CART_POSITION_OBS_SCALE
        )
        observation[:, 1] = (
            self.joint_vel[:, self._cart_dof_idx[0]] * CART_VELOCITY_OBS_SCALE
        )
        relative_angles = self._relative_angles()
        pole_velocity = self.joint_vel[:, self._pole_dof_idx]
        for pole_index in range(self.cfg.order):
            offset = 2 + 3 * pole_index
            observation[:, offset] = torch.sin(relative_angles[:, pole_index])
            observation[:, offset + 1] = torch.cos(relative_angles[:, pole_index])
            observation[:, offset + 2] = (
                pole_velocity[:, pole_index] * POLE_VELOCITY_OBS_SCALE
            )
        observation[:, 11 : 11 + self.cfg.order] = 1.0
        return {"policy": observation}

    def _get_rewards(self) -> torch.Tensor:
        absolute_angles = self._absolute_angles()
        mean_angle_squared = torch.mean(torch.square(absolute_angles), dim=-1)
        mean_pole_velocity_squared = torch.mean(
            torch.square(self.joint_vel[:, self._pole_dof_idx]), dim=-1
        )
        cart_position = self.joint_pos[:, self._cart_dof_idx[0]]
        cart_velocity = self.joint_vel[:, self._cart_dof_idx[0]]
        action = self.actions[:, 0]
        previous_action = self.previous_actions[:, 0]
        reward = 1.0 - 2.0 * mean_angle_squared
        reward -= 0.10 * torch.square(cart_position)
        reward -= 0.01 * torch.square(cart_velocity)
        reward -= 0.005 * mean_pole_velocity_squared
        reward -= 0.001 * torch.square(action)
        reward -= 0.01 * torch.square(action - previous_action)
        reward -= 5.0 * self.reset_terminated.float()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos.torch
        self.joint_vel = self.robot.data.joint_vel.torch
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        cart_out = (
            torch.abs(self.joint_pos[:, self._cart_dof_idx[0]])
            > self.cfg.max_cart_position
        )
        pole_out = torch.any(
            torch.abs(self._absolute_angles()) > TERMINATION_POLE_ANGLE_RAD,
            dim=-1,
        )
        return cart_out | pole_out, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        survived = self.reset_time_outs[env_ids] & ~self.reset_terminated[env_ids]
        self.extras.setdefault("log", {})["Metrics/success_rate"] = (
            survived.float().mean().item()
        )
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()
        joint_pos[:, self._cart_dof_idx] = torch.empty(
            (len(env_ids), 1), device=self.device
        ).uniform_(-0.25, 0.25)
        joint_vel[:, self._cart_dof_idx] = torch.empty(
            (len(env_ids), 1), device=self.device
        ).uniform_(-0.10, 0.10)
        joint_pos[:, self._pole_dof_idx] = torch.empty(
            (len(env_ids), self.cfg.order), device=self.device
        ).uniform_(-self.cfg.initial_angle_range, self.cfg.initial_angle_range)
        joint_vel[:, self._pole_dof_idx] = torch.empty(
            (len(env_ids), self.cfg.order), device=self.device
        ).uniform_(-0.10, 0.10)

        default_root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]
        default_root_vel = self.robot.data.default_root_vel.torch[env_ids].clone()
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0

        self.robot.write_root_pose_to_sim_index(
            root_pose=default_root_pose, env_ids=env_ids
        )
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=default_root_vel, env_ids=env_ids
        )
        self.robot.write_joint_position_to_sim_index(
            position=joint_pos, env_ids=env_ids
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=joint_vel, env_ids=env_ids
        )
