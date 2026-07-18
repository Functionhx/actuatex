"""Vectorized MuJoCo VecEnv for TinyMal locomotion (matching Isaac Gym legged_robot).

Optimizations vs naive per-env loop:
- Physics rollout parallelized via ThreadPoolExecutor (mj_step releases GIL).
- All state extraction, reward, and obs computation vectorized across N envs
  using batched numpy (no per-env Python loops in the hot path).

Run:  conda run -n unitree-rl python train_mujoco.py
"""

import os
import numpy as np
import torch
import mujoco
from concurrent.futures import ThreadPoolExecutor

from rsl_rl.env import VecEnv as VecEnvBase
from observation_builder import (
    DEFAULT_DOF_POS, DEFAULT_ANGLES, COMMANDS_SCALE,
    OBS_SCALE_LIN_VEL, OBS_SCALE_ANG_VEL, OBS_SCALE_DOF_POS, OBS_SCALE_DOF_VEL,
    CLIP_OBS, ACTION_SCALE, DECIMATION, SIM_DT, INIT_POS,
)
from model_builder import build_floating_model, JOINT_NAMES


def quat_rotate_inverse_batch(q_xyzw, v):
    """Vectorized world->body rotation. q_xyzw: (N,4), v: (N,3) -> (N,3)."""
    n = q_xyzw[:, :3]
    w = q_xyzw[:, 3:4]  # (N,1)
    t = 2.0 * np.cross(n, v)
    return v - w * t + np.cross(n, t)


def project_gravity_batch(q_xyzw):
    g = np.zeros((len(q_xyzw), 3))
    g[:, 2] = -1.0
    return quat_rotate_inverse_batch(q_xyzw, g)


def euler_roll_pitch_batch(q_xyzw):
    x, y, z, w = q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2], q_xyzw[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


class MjTinyMalEnv(VecEnvBase):
    is_recurrent = False

    def __init__(self, urdf_path, num_envs=512, device="cpu",
                 episode_length_s=20, cmd_resample_s=10.0,
                 add_noise=True, seed=1, armature=0.01, kv=0.5,
                 num_threads=8, init_pose_noise=0.5,
                 command_mode="omni", dense_tracking=False,
                 only_positive_rewards=True):
        self.num_envs = num_envs
        self.num_obs = 48
        self.num_privileged_obs = None
        self.num_actions = 12
        self.device = torch.device(device)
        self.dt = SIM_DT * DECIMATION

        self.model = build_floating_model(urdf_path, armature=armature, kv=kv)
        self.model.opt.timestep = SIM_DT
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self._pool = ThreadPoolExecutor(max_workers=num_threads)

        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in JOINT_NAMES]
        self.qposadr = np.array([self.model.jnt_qposadr[i] for i in self.joint_ids])
        self.dofadr = np.array([self.model.jnt_dofadr[i] for i in self.joint_ids])
        self.default_q = DEFAULT_DOF_POS.copy()

        # Soft DOF limits (0.9 of range).
        soft = 0.9
        limits = np.array([self.model.jnt_range[i] for i in self.joint_ids])
        mid = (limits[:, 0] + limits[:, 1]) / 2.0
        half = (limits[:, 1] - limits[:, 0]) / 2.0 * soft
        self.dof_pos_limits = np.stack([mid - half, mid + half], axis=1)  # (12, 2)

        self.foot_geom_ids = np.array([9, 16, 23, 30])
        self.foot_radius = 0.015
        self.contact_thresh = self.foot_radius + 0.01
        # Thigh collision geom IDs (for collision penalty, matching Isaac Gym's
        # penalize_contacts_on=["thigh","calf"]; foot spheres excluded).
        self.thigh_geom_ids = np.array([6, 13, 20, 27])
        self.thigh_geom_set = set(self.thigh_geom_ids.tolist())

        self.max_episode_length = int(np.ceil(episode_length_s / self.dt))
        self.cmd_resample_steps = int(np.round(cmd_resample_s / self.dt))
        if command_mode not in ("forward", "omni"):
            raise ValueError(f"unsupported command_mode={command_mode!r}")
        self.command_mode = command_mode
        self.cmd_ranges = dict(lin_vel_x=[-0.6, 0.6], lin_vel_y=[-0.3, 0.3],
                               ang_vel_yaw=[-0.8, 0.8])
        if command_mode == "forward":
            self.cmd_ranges = dict(
                lin_vel_x=[0.25, 0.60], lin_vel_y=[0.0, 0.0],
                ang_vel_yaw=[0.0, 0.0],
            )

        self.rew_scales = dict(
            tracking_lin_vel=1.0, tracking_ang_vel=0.5,
            lin_vel_z=-2.0, ang_vel_xy=-0.05, orientation=-1.0,
            base_height=-2.0, torques=-0.0002, dof_pos_limits=-10.0,
            stand_still=-0.1, feet_air_time=1.0,
            # Inherited from legged_robot_config (active in Isaac Gym baseline):
            collision=-1.0, action_rate=-0.01, dof_acc=-2.5e-7,
            # Dense terms used by the native MuJoCo gait curriculum.  They
            # provide a non-saturating gradient away from the standing local
            # optimum while remaining disabled for the historical baseline.
            velocity_tracking_l2=-1.5 if dense_tracking else 0.0,
            commanded_planar_progress=1.0 if dense_tracking else 0.0,
            yaw_velocity_tracking_l2=-0.5 if dense_tracking else 0.0,
        )
        if dense_tracking:
            self.rew_scales["tracking_lin_vel"] = 3.0
            self.rew_scales["tracking_ang_vel"] = 1.0
        self.tracking_sigma = 0.25
        self.base_height_target = 0.24
        self.only_positive_rewards = only_positive_rewards

        self.add_noise = add_noise
        self.noise_vec = np.array(
            [0.1]*3 + [0.2]*3 + [0.05]*3 + [0]*3 +
            [0.01]*12 + [1.5]*12 + [0]*12)  # lin_vel, ang_vel, grav, cmd, dof_pos, dof_vel, act

        self.rng = np.random.RandomState(seed)
        self.init_pose_noise = init_pose_noise

        # Torch buffers.
        N = num_envs
        self.obs_buf = torch.zeros(N, self.num_obs, dtype=torch.float32, device=self.device)
        self.privileged_obs_buf = None
        self.rew_buf = torch.zeros(N, dtype=torch.float32, device=self.device)
        self.reset_buf = torch.ones(N, dtype=torch.long, device=self.device)
        self.timeout_buf = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=self.device)
        self.extras = {}

        # Per-env numpy state.
        self.commands = np.zeros((N, 3))
        self.last_actions = np.zeros((N, 12))
        self.last_dof_vel = np.zeros((N, 12))
        self.feet_air_time = np.zeros((N, 4))
        self.last_contacts = np.zeros((N, 4), dtype=bool)
        self._ep_len_log = np.zeros(N)
        self._ep_rew_log = np.zeros(N)

    def _step_env_chunk(self, env_indices, actions):
        """Thread worker: step physics for a subset of envs."""
        for e in env_indices:
            d = self.datas[e]
            d.ctrl[:] = self.default_q + ACTION_SCALE * actions[e]
            for _ in range(DECIMATION):
                mujoco.mj_step(self.model, d)

    def step(self, actions):
        actions_np = actions.detach().cpu().numpy().astype(np.float64)
        actions_np = np.clip(actions_np, -100.0, 100.0)
        N = self.num_envs

        # 1. Threaded physics rollout.
        chunks = np.array_split(np.arange(N), self._pool._max_workers)
        list(self._pool.map(lambda c: self._step_env_chunk(c, actions_np), chunks))

        # 2. Episode length + command resampling.
        self.episode_length_buf += 1
        ep_len_np = self.episode_length_buf.cpu().numpy()
        resample_mask = (ep_len_np % self.cmd_resample_steps == 0) & (ep_len_np > 0)
        if resample_mask.any():
            self._resample_commands(np.where(resample_mask)[0])

        # 3. Batched state extraction.
        base_pos = np.array([self.datas[e].qpos[:3] for e in range(N)])
        base_quat_wxyz = np.array([self.datas[e].qpos[3:7] for e in range(N)])
        base_quat_xyzw = base_quat_wxyz[:, [1, 2, 3, 0]]
        world_lin_vel = np.array([self.datas[e].qvel[:3] for e in range(N)])
        world_ang_vel = np.array([self.datas[e].qvel[3:6] for e in range(N)])
        dof_pos = np.array([self.datas[e].qpos[self.qposadr] for e in range(N)])
        dof_vel = np.array([self.datas[e].qvel[self.dofadr] for e in range(N)])
        torques = np.array([self.datas[e].actuator_force.copy() for e in range(N)])

        base_lin_vel = quat_rotate_inverse_batch(base_quat_xyzw, world_lin_vel)
        base_ang_vel = quat_rotate_inverse_batch(base_quat_xyzw, world_ang_vel)
        pg = project_gravity_batch(base_quat_xyzw)
        foot_z = np.array([[self.datas[e].geom_xpos[gid][2]
                            for gid in self.foot_geom_ids] for e in range(N)])
        contacts = foot_z < self.contact_thresh

        # Collision penalty: count contacts involving thigh geoms per env.
        collision_counts = self._count_thigh_collisions()

        # 4. Vectorized reward.
        rewards = self._compute_rewards_vec(
            base_pos, base_lin_vel, base_ang_vel, pg,
            dof_pos, dof_vel, torques, contacts, collision_counts, actions_np)

        # 5. Vectorized termination.
        roll, pitch = euler_roll_pitch_batch(base_quat_xyzw)
        terminated = (base_pos[:, 2] < 0.10) | (np.abs(roll) > 0.8) | (np.abs(pitch) > 1.0)
        timeouts = ep_len_np >= self.max_episode_length
        resets = terminated | timeouts

        # 6. Only-positive clipping.
        if self.only_positive_rewards:
            rewards = np.maximum(rewards, 0.0)

        # 7. Vectorized obs.
        obs_np = np.zeros((N, self.num_obs))
        obs_np[:, 0:3] = base_lin_vel * OBS_SCALE_LIN_VEL
        obs_np[:, 3:6] = base_ang_vel * OBS_SCALE_ANG_VEL
        obs_np[:, 6:9] = pg
        obs_np[:, 9:12] = self.commands[:, :3] * COMMANDS_SCALE
        obs_np[:, 12:24] = (dof_pos - self.default_q) * OBS_SCALE_DOF_POS
        obs_np[:, 24:36] = dof_vel * OBS_SCALE_DOF_VEL
        obs_np[:, 36:48] = self.last_actions
        if self.add_noise:
            obs_np += self.rng.uniform(-1, 1, (N, self.num_obs)) * self.noise_vec
        np.clip(obs_np, -CLIP_OBS, CLIP_OBS, out=obs_np)
        self.obs_buf[:] = torch.from_numpy(obs_np).float()

        # 8. Update buffers.
        self.rew_buf[:] = torch.from_numpy(rewards).float()
        self.timeout_buf[:] = torch.from_numpy(timeouts)
        self.reset_buf[:] = torch.from_numpy(resets.astype(np.int64))

        # 9. Logging + reset. Always set time_outs for proper value bootstrapping.
        self._ep_len_log += 1
        self._ep_rew_log += rewards
        self.extras["time_outs"] = self.timeout_buf.clone()
        reset_ids = np.where(resets)[0]
        if len(reset_ids) > 0:
            self.extras["episode"] = {
                "length": float(np.mean(self._ep_len_log[reset_ids])),
                "reward": float(np.mean(self._ep_rew_log[reset_ids])),
            }
            self._reset_envs(reset_ids)
            self._ep_len_log[reset_ids] = 0
            self._ep_rew_log[reset_ids] = 0
        else:
            self.extras.pop("episode", None)

        self.last_actions[:] = actions_np
        return (self.obs_buf, self.privileged_obs_buf, self.rew_buf,
                self.reset_buf, self.extras)

    def _count_thigh_collisions(self):
        """Count contacts involving thigh geoms per env (collision penalty).

        Matches Isaac Gym's _reward_collision: count penalised bodies
        (thigh) with contact force > 0.1 N. In MuJoCo we check contact
        geoms directly. Vectorized across envs.
        """
        counts = np.zeros(self.num_envs, dtype=np.float64)
        thigh_arr = self.thigh_geom_ids
        for e in range(self.num_envs):
            d = self.datas[e]
            nc = d.ncon
            if nc == 0:
                continue
            geoms = np.concatenate([
                d.contact[:nc].geom1, d.contact[:nc].geom2])
            counts[e] = np.count_nonzero(np.isin(geoms, thigh_arr))
        return counts

    def _compute_rewards_vec(self, base_pos, base_lin_vel, base_ang_vel, pg,
                             dof_pos, dof_vel, torques, contacts,
                             collision_counts, actions):
        N = self.num_envs
        cmd = self.commands  # (N, 3)
        cmd_xy_norm = np.linalg.norm(cmd[:, :2], axis=1)

        # tracking_lin_vel: exp(-||cmd[:2] - vel[:2]||^2 / sigma) * 1.0
        lin_err = np.sum((cmd[:, :2] - base_lin_vel[:, :2]) ** 2, axis=1)
        rew = np.exp(-lin_err / self.tracking_sigma) * self.rew_scales["tracking_lin_vel"]
        # tracking_ang_vel.
        ang_err = (cmd[:, 2] - base_ang_vel[:, 2]) ** 2
        rew += np.exp(-ang_err / self.tracking_sigma) * self.rew_scales["tracking_ang_vel"]
        # Dense tracking/progress terms prevent zero velocity from becoming a
        # broad local optimum under the exponential rewards alone.
        rew += lin_err * self.rew_scales["velocity_tracking_l2"]
        planar_progress = np.sum(cmd[:, :2] * base_lin_vel[:, :2], axis=1)
        rew += planar_progress * self.rew_scales["commanded_planar_progress"]
        rew += ang_err * self.rew_scales["yaw_velocity_tracking_l2"]
        # lin_vel_z.
        rew += base_lin_vel[:, 2] ** 2 * self.rew_scales["lin_vel_z"]
        # ang_vel_xy.
        rew += np.sum(base_ang_vel[:, :2] ** 2, axis=1) * self.rew_scales["ang_vel_xy"]
        # orientation.
        rew += np.sum(pg[:, :2] ** 2, axis=1) * self.rew_scales["orientation"]
        # base_height.
        rew += (base_pos[:, 2] - self.base_height_target) ** 2 * self.rew_scales["base_height"]
        # torques.
        rew += np.sum(torques ** 2, axis=1) * self.rew_scales["torques"]
        # dof_pos_limits.
        out_low = np.clip(-(dof_pos - self.dof_pos_limits[:, 0]), 0, None)
        out_high = np.clip(dof_pos - self.dof_pos_limits[:, 1], 0, None)
        rew += np.sum(out_low + out_high, axis=1) * self.rew_scales["dof_pos_limits"]
        # feet_air_time.
        contact_filt = contacts | self.last_contacts
        first_contact = (self.feet_air_time > 0.0) & contact_filt
        rew_air = np.sum((self.feet_air_time - 0.5) * first_contact, axis=1)
        rew_air *= (cmd_xy_norm > 0.1).astype(np.float64)
        rew += rew_air * self.rew_scales["feet_air_time"]
        self.feet_air_time += self.dt
        self.feet_air_time *= ~contact_filt
        self.last_contacts[:] = contacts
        # stand_still.
        rew += np.sum(np.abs(dof_pos - self.default_q), axis=1) * \
               (cmd_xy_norm < 0.1).astype(np.float64) * self.rew_scales["stand_still"]
        # collision (inherited from legged_robot_config: penalize thigh contacts).
        rew += collision_counts * self.rew_scales["collision"]
        # action_rate (inherited: penalize changes in actions).
        rew += np.sum((self.last_actions - actions) ** 2, axis=1) * \
               self.rew_scales["action_rate"]
        # dof_acc (inherited: penalize joint accelerations).
        dof_acc = np.sum(((self.last_dof_vel - dof_vel) / self.dt) ** 2, axis=1)
        rew += dof_acc * self.rew_scales["dof_acc"]
        # Update last_dof_vel for next step.
        self.last_dof_vel[:] = dof_vel
        return rew

    def _resample_commands(self, env_ids):
        n = len(env_ids)
        self.commands[env_ids, 0] = self.rng.uniform(*self.cmd_ranges["lin_vel_x"], n)
        self.commands[env_ids, 1] = self.rng.uniform(*self.cmd_ranges["lin_vel_y"], n)
        self.commands[env_ids, 2] = self.rng.uniform(*self.cmd_ranges["ang_vel_yaw"], n)
        if self.command_mode == "omni":
            norms = np.linalg.norm(self.commands[env_ids, :2], axis=1)
            self.commands[env_ids[norms < 0.2], :2] = 0.0

    def _reset_envs(self, env_ids):
        for e in env_ids:
            d = self.datas[e]
            d.qpos[:3] = INIT_POS + np.array([self.rng.uniform(-1, 1),
                                              self.rng.uniform(-1, 1), 0.0])
            d.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
            d.qpos[self.qposadr] = self.default_q * self.rng.uniform(0.5, 1.5, 12)
            d.qvel[:] = 0.0
            d.qvel[:6] = self.rng.uniform(-0.5, 0.5, 6)
            d.qfrc_applied[:] = 0.0
            d.ctrl[:] = self.default_q
            mujoco.mj_forward(self.model, d)
        self._resample_commands(env_ids)
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        elif hasattr(env_ids, "cpu"):
            env_ids = env_ids.cpu().numpy()
        self._reset_envs(np.asarray(env_ids))
        self._compute_obs_batch()
        return self.obs_buf, self.privileged_obs_buf

    def _compute_obs_batch(self):
        N = self.num_envs
        base_pos = np.array([self.datas[e].qpos[:3] for e in range(N)])
        base_quat_wxyz = np.array([self.datas[e].qpos[3:7] for e in range(N)])
        base_quat_xyzw = base_quat_wxyz[:, [1, 2, 3, 0]]
        world_lin_vel = np.array([self.datas[e].qvel[:3] for e in range(N)])
        world_ang_vel = np.array([self.datas[e].qvel[3:6] for e in range(N)])
        dof_pos = np.array([self.datas[e].qpos[self.qposadr] for e in range(N)])
        dof_vel = np.array([self.datas[e].qvel[self.dofadr] for e in range(N)])
        base_lin_vel = quat_rotate_inverse_batch(base_quat_xyzw, world_lin_vel)
        base_ang_vel = quat_rotate_inverse_batch(base_quat_xyzw, world_ang_vel)
        pg = project_gravity_batch(base_quat_xyzw)
        obs_np = np.zeros((N, self.num_obs))
        obs_np[:, 0:3] = base_lin_vel * OBS_SCALE_LIN_VEL
        obs_np[:, 3:6] = base_ang_vel * OBS_SCALE_ANG_VEL
        obs_np[:, 6:9] = pg
        obs_np[:, 9:12] = self.commands[:, :3] * COMMANDS_SCALE
        obs_np[:, 12:24] = (dof_pos - self.default_q) * OBS_SCALE_DOF_POS
        obs_np[:, 24:36] = dof_vel * OBS_SCALE_DOF_VEL
        obs_np[:, 36:48] = self.last_actions
        np.clip(obs_np, -CLIP_OBS, CLIP_OBS, out=obs_np)
        self.obs_buf[:] = torch.from_numpy(obs_np).float()

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None
