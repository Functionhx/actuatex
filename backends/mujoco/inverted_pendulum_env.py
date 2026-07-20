"""Vectorized MuJoCo environment for the 1/2/3-link cart-pole curriculum."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
    # SAC/TD3 use the same environment contract without importing RSL-RL.
    # PPO still receives the real VecEnv base whenever that optional package
    # is installed or RSL_RL_ROOT is configured.
    class VecEnvBase:  # type: ignore[no-redef]
        pass


from tasks.inverted_pendulum.contract import (  # noqa: E402
    ACTION_DIM,
    ACTION_FORCE_SCALE_N,
    DECIMATION,
    EPISODE_LENGTH_S,
    INITIAL_ANGLE_RANGE_RAD,
    OBSERVATION_DIM,
    POLICY_DT,
    absolute_pole_angles,
    build_observation,
    compute_reward,
    terminated,
    validate_order,
)


class MjSerialInvertedPendulumEnv(VecEnvBase):
    """RSL-RL-compatible batch of independent MuJoCo cart-poles."""

    is_recurrent = False

    def __init__(
        self,
        order: int,
        *,
        num_envs: int = 128,
        device: str = "cpu",
        seed: int = 1,
        num_threads: int = 8,
        episode_length_s: float = EPISODE_LENGTH_S,
        initial_angle_scale: float = 1.0,
        model_path: Path | None = None,
    ) -> None:
        self.order = validate_order(order)
        self.num_envs = num_envs
        self.num_obs = OBSERVATION_DIM
        self.num_privileged_obs = None
        self.num_actions = ACTION_DIM
        self.device = torch.device(device)
        self.dt = POLICY_DT
        self.max_episode_length = round(episode_length_s / POLICY_DT)
        self.initial_angle_range = (
            INITIAL_ANGLE_RANGE_RAD[self.order] * initial_angle_scale
        )
        self.rng = np.random.default_rng(seed)

        if model_path is None:
            model_path = (
                REPO_ROOT
                / "robots"
                / "inverted_pendulum"
                / "mjcf"
                / f"actuatex_cartpole_{self.order}.xml"
            )
        self.model_path = model_path.resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self._pool = ThreadPoolExecutor(max_workers=max(1, num_threads))

        self.cart_joint_id = self._joint_id("cart_slide")
        self.cart_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cart"
        )
        if self.cart_body_id < 0:
            raise ValueError("model is missing body 'cart'")
        self.pole_joint_ids = np.array(
            [self._joint_id(f"pole_{index}_hinge") for index in range(1, order + 1)]
        )
        self.cart_qpos_address = int(self.model.jnt_qposadr[self.cart_joint_id])
        self.cart_dof_address = int(self.model.jnt_dofadr[self.cart_joint_id])
        self.pole_qpos_addresses = self.model.jnt_qposadr[self.pole_joint_ids]
        self.pole_dof_addresses = self.model.jnt_dofadr[self.pole_joint_ids]

        shape = (num_envs,)
        self.obs_buf = torch.zeros(
            num_envs, OBSERVATION_DIM, dtype=torch.float32, device=self.device
        )
        self.privileged_obs_buf = None
        self.rew_buf = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.reset_buf = torch.ones(shape, dtype=torch.long, device=self.device)
        self.timeout_buf = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(
            shape, dtype=torch.long, device=self.device
        )
        self.previous_action = np.zeros(shape, dtype=np.float64)
        self.external_cart_force_n = np.zeros(shape, dtype=np.float64)
        self.episode_reward = np.zeros(shape, dtype=np.float64)
        self.extras: dict[str, Any] = {}
        self.total_terminations = 0
        self.total_timeouts = 0
        self.last_terminal = np.zeros(shape, dtype=bool)
        self.last_timeout = np.zeros(shape, dtype=bool)
        self.last_cart_position = np.zeros(shape, dtype=np.float64)
        self.last_absolute_angles = np.zeros((num_envs, self.order), dtype=np.float64)

    def _joint_id(self, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"model is missing joint {joint_name!r}")
        return joint_id

    def _state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cart_position = np.fromiter(
            (data.qpos[self.cart_qpos_address] for data in self.datas),
            dtype=np.float64,
            count=self.num_envs,
        )
        cart_velocity = np.fromiter(
            (data.qvel[self.cart_dof_address] for data in self.datas),
            dtype=np.float64,
            count=self.num_envs,
        )
        pole_angles = np.stack(
            [data.qpos[self.pole_qpos_addresses] for data in self.datas]
        )
        pole_velocities = np.stack(
            [data.qvel[self.pole_dof_addresses] for data in self.datas]
        )
        return cart_position, cart_velocity, pole_angles, pole_velocities

    def _step_chunk(self, env_ids: np.ndarray, action: np.ndarray) -> None:
        for env_id in env_ids:
            data = self.datas[int(env_id)]
            data.ctrl[0] = ACTION_FORCE_SCALE_N * action[env_id]
            data.xfrc_applied[self.cart_body_id, 0] = self.external_cart_force_n[env_id]
            for _ in range(DECIMATION):
                mujoco.mj_step(self.model, data)

    def step(self, actions: torch.Tensor):
        action = (
            actions.detach().cpu().numpy().reshape(self.num_envs).astype(np.float64)
        )
        np.clip(action, -1.0, 1.0, out=action)
        chunks = np.array_split(np.arange(self.num_envs), self._pool._max_workers)
        list(self._pool.map(lambda env_ids: self._step_chunk(env_ids, action), chunks))

        self.episode_length_buf += 1
        cart_position, cart_velocity, pole_angles, pole_velocities = self._state()
        terminal = terminated(cart_position, pole_angles)
        timeouts = self.episode_length_buf.cpu().numpy() >= self.max_episode_length
        resets = terminal | timeouts
        self.last_terminal[:] = terminal
        self.last_timeout[:] = timeouts
        self.last_cart_position[:] = cart_position
        self.last_absolute_angles[:] = absolute_pole_angles(pole_angles)
        reward = compute_reward(
            cart_position,
            cart_velocity,
            pole_angles,
            pole_velocities,
            action,
            self.previous_action,
            terminal,
        )
        self.episode_reward += reward

        self.rew_buf.copy_(torch.from_numpy(reward).to(self.device))
        self.reset_buf.copy_(torch.from_numpy(resets.astype(np.int64)).to(self.device))
        self.timeout_buf.copy_(torch.from_numpy(timeouts).to(self.device))
        self.extras["time_outs"] = self.timeout_buf.clone()

        reset_ids = np.flatnonzero(resets)
        if reset_ids.size:
            self.total_terminations += int(np.count_nonzero(terminal[reset_ids]))
            self.total_timeouts += int(np.count_nonzero(timeouts[reset_ids]))
            self.extras["episode"] = {
                "reward": float(np.mean(self.episode_reward[reset_ids])),
                "length": float(
                    self.episode_length_buf[reset_ids].float().mean().item()
                ),
                "success_rate": float(
                    np.mean(timeouts[reset_ids] & ~terminal[reset_ids])
                ),
            }
        else:
            self.extras.pop("episode", None)

        self.previous_action[:] = action
        if reset_ids.size:
            self._reset_envs(reset_ids)
        self._compute_observations()
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def _reset_envs(self, env_ids: np.ndarray) -> None:
        for env_id in np.asarray(env_ids, dtype=np.int64):
            data = self.datas[int(env_id)]
            mujoco.mj_resetData(self.model, data)
            data.qpos[self.cart_qpos_address] = self.rng.uniform(-0.25, 0.25)
            data.qvel[self.cart_dof_address] = self.rng.uniform(-0.10, 0.10)
            data.qpos[self.pole_qpos_addresses] = self.rng.uniform(
                -self.initial_angle_range,
                self.initial_angle_range,
                self.order,
            )
            data.qvel[self.pole_dof_addresses] = self.rng.uniform(
                -0.10, 0.10, self.order
            )
            data.ctrl[0] = 0.0
            mujoco.mj_forward(self.model, data)
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.previous_action[env_ids] = 0.0
        self.external_cart_force_n[env_ids] = 0.0
        self.episode_reward[env_ids] = 0.0

    def _compute_observations(self) -> None:
        state = self._state()
        observation = build_observation(*state, self.order)
        self.obs_buf.copy_(torch.from_numpy(observation).to(self.device))

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        elif hasattr(env_ids, "cpu"):
            env_ids = env_ids.cpu().numpy()
        self._reset_envs(np.asarray(env_ids, dtype=np.int64))
        self._compute_observations()
        return self.obs_buf, self.privileged_obs_buf

    def get_observations(self) -> torch.Tensor:
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def close(self) -> None:
        self._pool.shutdown(wait=True)
