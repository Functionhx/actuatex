"""Vectorized MuJoCo environment for the full-dynamics ActuateX Sentinel."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
import torch

from actuatex_paths import REPO_ROOT, RSL_RL_ROOT

if RSL_RL_ROOT.is_dir():
    sys.path.insert(0, str(RSL_RL_ROOT))
sys.path.insert(0, str(REPO_ROOT))

try:
    from rsl_rl.env import VecEnv as VecEnvBase  # noqa: E402
except ModuleNotFoundError:
    class VecEnvBase:  # type: ignore[no-redef]
        pass


from tasks.robomaster.contract import (  # noqa: E402
    ACTION_DIM,
    ALL_JOINT_NAMES,
    FULL_ACTUATOR_DIM,
    POLICY_DECIMATION,
    POLICY_DT,
    RMUC_2026,
    SENTINEL_DEFAULT_JOINT_POSITION,
    SIM_DT,
)
from tasks.robomaster.locomotion import (  # noqa: E402
    OBSERVATION_DIM,
    OBSERVATION_NOISE_AMPLITUDE,
    TRACK_WIDTH_M,
    WHEEL_RADIUS_M,
    action_to_joint_targets,
    build_observation,
    compute_requested_joint_torque,
    projected_gravity,
)
from tasks.robomaster.powertrain import (  # noqa: E402
    RefereePowerMonitor,
    allocate_motor_power,
    integrate_motor_temperature,
)


CHASSIS_POWER_MASK = np.asarray(
    [name in ALL_JOINT_NAMES[:ACTION_DIM] for name in ALL_JOINT_NAMES],
    dtype=bool,
)
MANAGED_POWER_CEILING_W = 180.0


def _object_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Sentinel model is missing {name!r}")
    return object_id


def _euler_xyz_to_quaternion_wxyz(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


class MjSentinelEnv(VecEnvBase):
    """RSL-RL/SAC/TD3-compatible independent Sentinel batch."""

    is_recurrent = False

    def __init__(
        self,
        *,
        num_envs: int = 64,
        device: str = "cpu",
        seed: int = 1,
        num_threads: int = 8,
        episode_length_s: float = 20.0,
        add_noise: bool = True,
        randomize_reset: bool = True,
        maximum_command_delay_steps: int = 0,
        model_path: Path | None = None,
    ) -> None:
        if num_envs <= 0 or num_threads <= 0:
            raise ValueError("num_envs and num_threads must be positive")
        if episode_length_s <= 0.0:
            raise ValueError("episode_length_s must be positive")
        if maximum_command_delay_steps < 0:
            raise ValueError("maximum_command_delay_steps cannot be negative")
        self.num_envs = int(num_envs)
        self.num_obs = OBSERVATION_DIM
        self.num_privileged_obs = None
        self.num_actions = ACTION_DIM
        self.device = torch.device(device)
        self.dt = POLICY_DT
        self.max_episode_length = round(episode_length_s / POLICY_DT)
        self.add_noise = bool(add_noise)
        self.randomize_reset = bool(randomize_reset)
        self.maximum_command_delay_steps = int(maximum_command_delay_steps)
        self.rng = np.random.default_rng(seed)

        if model_path is None:
            model_path = (
                REPO_ROOT
                / "robots"
                / "robomaster"
                / "mjcf"
                / "actuatex_sentinel.xml"
            )
        self.model_path = model_path.resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.datas = [mujoco.MjData(self.model) for _ in range(self.num_envs)]
        self._pool = ThreadPoolExecutor(max_workers=max(1, num_threads))
        self._chunks = np.array_split(
            np.arange(self.num_envs), self._pool._max_workers
        )

        self.joint_ids = np.asarray(
            [
                _object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ALL_JOINT_NAMES
            ]
        )
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids]
        self.dof_addresses = self.model.jnt_dofadr[self.joint_ids]
        actuator_joint_names = tuple(
            mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(self.model.actuator_trnid[index, 0]),
            )
            for index in range(self.model.nu)
        )
        if actuator_joint_names != ALL_JOINT_NAMES:
            raise ValueError(
                "Sentinel actuator order drifted: "
                f"{actuator_joint_names} != {ALL_JOINT_NAMES}"
            )
        self.base_body_id = _object_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        self.base_geom_id = _object_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "base_collision"
        )
        self.floor_geom_id = _object_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.leg_geom_ids = {
            _object_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in (
                "left_upper_collision",
                "left_lower_collision",
                "right_upper_collision",
                "right_lower_collision",
            )
        }
        self.leg_joint_ranges = self.model.jnt_range[self.joint_ids[:4]].copy()

        shape = (self.num_envs,)
        self.obs_buf = torch.zeros(
            self.num_envs,
            OBSERVATION_DIM,
            dtype=torch.float32,
            device=self.device,
        )
        self.privileged_obs_buf = None
        self.rew_buf = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.reset_buf = torch.ones(shape, dtype=torch.long, device=self.device)
        self.timeout_buf = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(
            shape, dtype=torch.long, device=self.device
        )
        self.commands = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.fixed_command: np.ndarray | None = None
        self.command_steps_remaining = np.zeros(self.num_envs, dtype=np.int64)
        self.previous_action = np.zeros(
            (self.num_envs, ACTION_DIM), dtype=np.float64
        )
        self.previous_joint_velocity = np.zeros(
            (self.num_envs, FULL_ACTUATOR_DIM), dtype=np.float64
        )
        self.motor_temperature_c = np.full(
            (self.num_envs, FULL_ACTUATOR_DIM), 25.0, dtype=np.float64
        )
        self.referee_monitors = [
            RefereePowerMonitor() for _ in range(self.num_envs)
        ]
        self.last_electrical_power_w = np.zeros(
            (self.num_envs, FULL_ACTUATOR_DIM), dtype=np.float64
        )
        self.last_chassis_power_w = np.zeros(self.num_envs, dtype=np.float64)
        self.last_torque_scale = np.ones(self.num_envs, dtype=np.float64)
        self.last_applied_torque = np.zeros(
            (self.num_envs, FULL_ACTUATOR_DIM), dtype=np.float64
        )
        self.episode_reward = np.zeros(self.num_envs, dtype=np.float64)
        self.extras: dict[str, Any] = {}
        self.total_terminations = 0
        self.total_timeouts = 0
        self.last_terminal = np.zeros(self.num_envs, dtype=bool)
        self.last_timeout = np.zeros(self.num_envs, dtype=bool)

        history_length = self.maximum_command_delay_steps + 1
        self.position_target_history = np.broadcast_to(
            SENTINEL_DEFAULT_JOINT_POSITION,
            (self.num_envs, history_length, FULL_ACTUATOR_DIM),
        ).copy()
        self.velocity_target_history = np.zeros_like(self.position_target_history)
        self.command_delay_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._target_history_cursor = 0

    def _step_chunk(self, env_ids: np.ndarray) -> None:
        for env_id in env_ids:
            mujoco.mj_step(self.model, self.datas[int(env_id)])

    def _joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        position = np.stack(
            [data.qpos[self.qpos_addresses] for data in self.datas]
        )
        velocity = np.stack(
            [data.qvel[self.dof_addresses] for data in self.datas]
        )
        return position, velocity

    def _physics_rollout(self, action: np.ndarray) -> None:
        position_target, velocity_target = action_to_joint_targets(action)
        env_indices = np.arange(self.num_envs)
        history_length = self.maximum_command_delay_steps + 1
        for _ in range(POLICY_DECIMATION):
            cursor = self._target_history_cursor
            self.position_target_history[:, cursor] = position_target
            self.velocity_target_history[:, cursor] = velocity_target
            delayed_cursor = (
                cursor - self.command_delay_steps
            ) % history_length
            delayed_position_target = self.position_target_history[
                env_indices, delayed_cursor
            ]
            delayed_velocity_target = self.velocity_target_history[
                env_indices, delayed_cursor
            ]
            self._target_history_cursor = (cursor + 1) % history_length

            joint_position, joint_velocity = self._joint_state()
            requested = compute_requested_joint_torque(
                joint_position,
                joint_velocity,
                action,
                position_target_override=delayed_position_target,
                velocity_target_override=delayed_velocity_target,
            )
            enabled = np.asarray(
                [monitor.chassis_enabled for monitor in self.referee_monitors]
            )
            requested[~enabled, :ACTION_DIM] = 0.0
            buffer_energy = np.asarray(
                [
                    monitor.state.buffer_energy_j
                    for monitor in self.referee_monitors
                ]
            )
            managed_limit = RMUC_2026.full_auto_chassis_power_limit_w + (
                MANAGED_POWER_CEILING_W
                - RMUC_2026.full_auto_chassis_power_limit_w
            ) * (
                buffer_energy / RMUC_2026.chassis_buffer_energy_limit_j
            )
            managed_limit[~enabled] = 0.0
            allocation = allocate_motor_power(
                requested,
                joint_velocity,
                power_limit_w=managed_limit,
                power_mask=CHASSIS_POWER_MASK,
                temperature_c=self.motor_temperature_c,
                iterations=18,
            )
            self.last_applied_torque[:] = allocation.motor.joint_torque
            self.last_electrical_power_w[:] = allocation.motor.electrical_power_w
            self.last_chassis_power_w[:] = allocation.accounted_bus_power_w
            self.last_torque_scale[:] = allocation.torque_scale
            for env_id, data in enumerate(self.datas):
                data.ctrl[:] = allocation.motor.joint_torque[env_id]
            list(self._pool.map(self._step_chunk, self._chunks))
            self.motor_temperature_c[:] = integrate_motor_temperature(
                self.motor_temperature_c,
                allocation.motor.copper_loss_w,
                SIM_DT,
            )
            for env_id, monitor in enumerate(self.referee_monitors):
                monitor.step(self.last_chassis_power_w[env_id], SIM_DT)

    def _body_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        position = np.empty((self.num_envs, 3), dtype=np.float64)
        quaternion = np.empty((self.num_envs, 4), dtype=np.float64)
        angular_velocity = np.empty((self.num_envs, 3), dtype=np.float64)
        linear_velocity = np.empty((self.num_envs, 3), dtype=np.float64)
        for env_id, data in enumerate(self.datas):
            spatial_velocity = np.empty(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                self.model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                self.base_body_id,
                spatial_velocity,
                1,
            )
            position[env_id] = data.xpos[self.base_body_id]
            quaternion[env_id] = data.xquat[self.base_body_id]
            angular_velocity[env_id] = spatial_velocity[:3]
            linear_velocity[env_id] = spatial_velocity[3:]
        gravity = projected_gravity(quaternion)
        return position, quaternion, linear_velocity, angular_velocity, gravity

    def _contact_metrics(self) -> tuple[np.ndarray, np.ndarray]:
        base_contact = np.zeros(self.num_envs, dtype=bool)
        leg_contact_count = np.zeros(self.num_envs, dtype=np.float64)
        contact_force = np.empty(6, dtype=np.float64)
        for env_id, data in enumerate(self.datas):
            contacted_leg_geoms: set[int] = set()
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
                if {geom1, geom2} == {self.base_geom_id, self.floor_geom_id}:
                    base_contact[env_id] = True
                leg_geoms = {geom1, geom2} & self.leg_geom_ids
                if not leg_geoms:
                    continue
                mujoco.mj_contactForce(
                    self.model, data, contact_index, contact_force
                )
                if abs(contact_force[0]) > 1.0:
                    contacted_leg_geoms.update(leg_geoms)
            leg_contact_count[env_id] = len(contacted_leg_geoms)
        return base_contact, leg_contact_count

    def _reward(
        self,
        action: np.ndarray,
        base_position: np.ndarray,
        base_linear_velocity: np.ndarray,
        base_angular_velocity: np.ndarray,
        gravity: np.ndarray,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        leg_contact_count: np.ndarray,
        terminal: np.ndarray,
    ) -> np.ndarray:
        linear_error = np.sum(
            (self.commands[:, :2] - base_linear_velocity[:, :2]) ** 2,
            axis=1,
        )
        yaw_error = (
            self.commands[:, 2] - base_angular_velocity[:, 2]
        ) ** 2
        reward = 2.0 * np.exp(-linear_error / 0.45**2)
        reward += np.exp(-yaw_error / 0.45**2)
        active = np.abs(self.commands[:, 0]) > 0.10
        reward += (
            0.75
            * np.clip(
                base_linear_velocity[:, 0] * np.sign(self.commands[:, 0]),
                -1.5,
                1.5,
            )
            * active
        )
        reward += 0.25
        reward -= 200.0 * terminal
        reward -= 8.0 * np.sum(gravity[:, :2] ** 2, axis=1)
        reward -= 20.0 * (base_position[:, 2] - 0.515) ** 2
        reward -= 2.0 * base_linear_velocity[:, 2] ** 2
        reward -= 0.10 * np.sum(base_angular_velocity[:, :2] ** 2, axis=1)
        reward -= base_linear_velocity[:, 1] ** 2
        reward -= 0.50 * (
            (joint_position[:, 0] - joint_position[:, 2]) ** 2
            + (joint_position[:, 1] - joint_position[:, 3]) ** 2
        )
        reward -= 0.10 * np.sum(
            np.abs(
                joint_position[:, :4]
                - SENTINEL_DEFAULT_JOINT_POSITION[:4]
            ),
            axis=1,
        )
        desired_left = (
            self.commands[:, 0]
            - 0.5 * TRACK_WIDTH_M * self.commands[:, 2]
        ) / WHEEL_RADIUS_M
        desired_right = (
            self.commands[:, 0]
            + 0.5 * TRACK_WIDTH_M * self.commands[:, 2]
        ) / WHEEL_RADIUS_M
        desired_wheel_velocity = np.stack(
            (desired_left, desired_right), axis=1
        )
        reward -= 0.05 * np.sum(
            ((joint_velocity[:, 4:6] - desired_wheel_velocity) * 0.05) ** 2,
            axis=1,
        )
        reward -= 1.0e-4 * np.sum(self.last_applied_torque**2, axis=1)
        joint_acceleration = (
            joint_velocity - self.previous_joint_velocity
        ) / POLICY_DT
        reward -= 2.5e-7 * np.sum(joint_acceleration**2, axis=1)
        reward -= 0.01 * np.sum((action - self.previous_action) ** 2, axis=1)
        reward -= leg_contact_count
        below = np.clip(
            self.leg_joint_ranges[:, 0] - joint_position[:, :4],
            0.0,
            None,
        )
        above = np.clip(
            joint_position[:, :4] - self.leg_joint_ranges[:, 1],
            0.0,
            None,
        )
        reward -= 5.0 * np.sum(below + above, axis=1)
        return reward * POLICY_DT

    def _resample_commands(self, env_ids: np.ndarray) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64)
        if self.fixed_command is not None:
            self.commands[env_ids] = self.fixed_command[env_ids]
            self.command_steps_remaining[env_ids] = self.max_episode_length + 1
            return
        count = len(env_ids)
        self.commands[env_ids, 0] = self.rng.uniform(-1.0, 1.0, count)
        self.commands[env_ids, 1] = 0.0
        self.commands[env_ids, 2] = self.rng.uniform(-1.5, 1.5, count)
        standing = self.rng.random(count) < 0.05
        self.commands[env_ids[standing]] = 0.0
        self.command_steps_remaining[env_ids] = self.rng.integers(
            round(5.0 / POLICY_DT),
            round(8.0 / POLICY_DT) + 1,
            count,
        )

    def _reset_envs(self, env_ids: np.ndarray) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64)
        default_position_target, default_velocity_target = action_to_joint_targets(
            np.zeros((len(env_ids), ACTION_DIM), dtype=np.float64)
        )
        for local_index, env_id in enumerate(env_ids):
            data = self.datas[int(env_id)]
            mujoco.mj_resetDataKeyframe(self.model, data, 0)
            joint_position = SENTINEL_DEFAULT_JOINT_POSITION.copy()
            if self.randomize_reset:
                data.qpos[:3] = (
                    self.rng.uniform(-0.10, 0.10),
                    self.rng.uniform(-0.10, 0.10),
                    0.515,
                )
                data.qpos[3:7] = _euler_xyz_to_quaternion_wxyz(
                    self.rng.uniform(-0.03, 0.03),
                    self.rng.uniform(-0.05, 0.05),
                    self.rng.uniform(-math.pi, math.pi),
                )
                joint_position[:4] += self.rng.uniform(-0.05, 0.05, 4)
                joint_position[6:8] += self.rng.uniform(-0.03, 0.03, 2)
            else:
                data.qpos[:3] = (0.0, 0.0, 0.515)
                data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
            data.qpos[self.qpos_addresses] = joint_position
            data.qvel[:] = 0.0
            if self.randomize_reset:
                data.qvel[:3] = self.rng.uniform(-0.10, 0.10, 3)
                data.qvel[3:6] = self.rng.uniform(-0.10, 0.10, 3)
            data.ctrl[:] = 0.0
            data.qfrc_applied[:] = 0.0
            mujoco.mj_forward(self.model, data)
            self.referee_monitors[int(env_id)].reset()
            self.motor_temperature_c[env_id] = 25.0
            self.last_electrical_power_w[env_id] = 0.0
            self.last_chassis_power_w[env_id] = 0.0
            self.last_torque_scale[env_id] = 1.0
            self.last_applied_torque[env_id] = 0.0
            self.command_delay_steps[env_id] = self.rng.integers(
                0, self.maximum_command_delay_steps + 1
            )
            self.position_target_history[env_id] = default_position_target[
                local_index
            ]
            self.velocity_target_history[env_id] = default_velocity_target[
                local_index
            ]
        self._resample_commands(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.previous_action[env_ids] = 0.0
        self.previous_joint_velocity[env_ids] = 0.0
        self.episode_reward[env_ids] = 0.0

    def _compute_observation(self) -> None:
        _, _, linear_velocity, angular_velocity, gravity = self._body_state()
        joint_position, joint_velocity = self._joint_state()
        observation = build_observation(
            linear_velocity,
            angular_velocity,
            gravity,
            self.commands,
            joint_position,
            joint_velocity,
            self.previous_action,
        )
        if self.add_noise:
            observation += (
                self.rng.uniform(-1.0, 1.0, observation.shape)
                * OBSERVATION_NOISE_AMPLITUDE
            ).astype(np.float32)
        self.obs_buf.copy_(torch.from_numpy(observation).to(self.device))

    def step(self, actions: torch.Tensor):
        action = actions.detach().cpu().numpy().astype(np.float64)
        if action.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(
                f"actions must have shape {(self.num_envs, ACTION_DIM)}, "
                f"got {action.shape}"
            )
        np.clip(action, -1.0, 1.0, out=action)
        self._physics_rollout(action)
        self.episode_length_buf += 1
        self.command_steps_remaining -= 1
        command_reset_ids = np.flatnonzero(self.command_steps_remaining <= 0)
        if command_reset_ids.size:
            self._resample_commands(command_reset_ids)

        base_position, _, linear_velocity, angular_velocity, gravity = (
            self._body_state()
        )
        joint_position, joint_velocity = self._joint_state()
        base_contact, leg_contact_count = self._contact_metrics()
        tilt = np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0))
        terminal = base_contact | (tilt > 0.90) | (base_position[:, 2] < 0.26)
        timeout = (
            self.episode_length_buf.cpu().numpy() >= self.max_episode_length
        )
        resets = terminal | timeout
        reward = self._reward(
            action,
            base_position,
            linear_velocity,
            angular_velocity,
            gravity,
            joint_position,
            joint_velocity,
            leg_contact_count,
            terminal,
        )
        self.episode_reward += reward
        self.last_terminal[:] = terminal
        self.last_timeout[:] = timeout
        self.rew_buf.copy_(torch.from_numpy(reward).to(self.device))
        self.reset_buf.copy_(
            torch.from_numpy(resets.astype(np.int64)).to(self.device)
        )
        self.timeout_buf.copy_(torch.from_numpy(timeout).to(self.device))
        self.extras["time_outs"] = self.timeout_buf.clone()

        reset_ids = np.flatnonzero(resets)
        if reset_ids.size:
            self.total_terminations += int(np.count_nonzero(terminal[reset_ids]))
            self.total_timeouts += int(np.count_nonzero(timeout[reset_ids]))
            self.extras["episode"] = {
                "reward": float(np.mean(self.episode_reward[reset_ids])),
                "length": float(
                    self.episode_length_buf[reset_ids].float().mean().item()
                ),
                "success_rate": float(
                    np.mean(timeout[reset_ids] & ~terminal[reset_ids])
                ),
                "mean_motor_temperature_c": float(
                    np.mean(self.motor_temperature_c[reset_ids])
                ),
            }
        else:
            self.extras.pop("episode", None)

        self.previous_action[:] = action
        self.previous_joint_velocity[:] = joint_velocity
        if reset_ids.size:
            self._reset_envs(reset_ids)
        self._compute_observation()
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        elif hasattr(env_ids, "cpu"):
            env_ids = env_ids.cpu().numpy()
        self._reset_envs(np.asarray(env_ids, dtype=np.int64))
        self._compute_observation()
        return self.obs_buf, self.privileged_obs_buf

    def get_observations(self) -> torch.Tensor:
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def set_command(self, command: np.ndarray) -> None:
        command = np.asarray(command, dtype=np.float64)
        if command.shape == (3,):
            command = np.broadcast_to(command, self.commands.shape)
        if command.shape != self.commands.shape or not np.isfinite(command).all():
            raise ValueError(
                f"command must have shape (3,) or {self.commands.shape}"
            )
        self.fixed_command = np.asarray(command, dtype=np.float64).copy()
        self.commands[:] = self.fixed_command
        self.command_steps_remaining[:] = self.max_episode_length + 1

    def clear_command_override(self) -> None:
        self.fixed_command = None
        self._resample_commands(np.arange(self.num_envs))

    def diagnostics(self) -> dict[str, np.ndarray]:
        """Return copied state used by backend-neutral evaluation tools."""

        base_position, _, linear_velocity, angular_velocity, gravity = (
            self._body_state()
        )
        joint_position, joint_velocity = self._joint_state()
        return {
            "base_position": base_position,
            "base_linear_velocity_body": linear_velocity,
            "base_angular_velocity_body": angular_velocity,
            "projected_gravity": gravity,
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "motor_temperature_c": self.motor_temperature_c.copy(),
            "chassis_power_w": self.last_chassis_power_w.copy(),
            "buffer_energy_j": np.asarray(
                [
                    monitor.state.buffer_energy_j
                    for monitor in self.referee_monitors
                ]
            ),
            "chassis_enabled": np.asarray(
                [monitor.chassis_enabled for monitor in self.referee_monitors]
            ),
        }

    def close(self) -> None:
        self._pool.shutdown(wait=True)
