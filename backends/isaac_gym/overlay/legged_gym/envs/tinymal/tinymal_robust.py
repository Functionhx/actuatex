"""TinyMal environment used for the final sim-to-real robustness training.

This subclass leaves the baseline :class:`LeggedRobot` untouched and adds the
domain gaps identified by the three-engine experiment: rough contact geometry,
mass and actuator uncertainty, policy-rate control latency, joint friction and
armature, and sustained external pushes.
"""

import time

import numpy as np
import torch
from isaacgym import gymapi, gymtorch, terrain_utils
from isaacgym.torch_utils import torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot


class TinyMalRoughTerrain:
    """Global tiled heightfield with a randomized amplitude per patch."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.width_per_env_pixels = int(cfg.terrain_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(cfg.terrain_length / cfg.horizontal_scale)
        self.border = int(cfg.border_size / cfg.horizontal_scale)
        self.tot_cols = cfg.num_cols * self.width_per_env_pixels + 2 * self.border
        self.tot_rows = cfg.num_rows * self.length_per_env_pixels + 2 * self.border
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3), dtype=np.float32)
        self.patch_roughness = np.zeros((cfg.num_rows, cfg.num_cols), dtype=np.float32)
        self._build()

    def _build(self):
        cfg = self.cfg
        patch_count = cfg.num_rows * cfg.num_cols
        flat_count = int(round(patch_count * cfg.flat_patch_fraction))
        amplitudes = np.random.uniform(
            cfg.roughness_range[0], cfg.roughness_range[1], size=patch_count
        )
        amplitudes[:flat_count] = 0.0
        np.random.shuffle(amplitudes)

        for patch_index, amplitude in enumerate(amplitudes):
            row, col = np.unravel_index(patch_index, (cfg.num_rows, cfg.num_cols))
            terrain = terrain_utils.SubTerrain(
                "tinymal_rough_patch",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=cfg.vertical_scale,
                horizontal_scale=cfg.horizontal_scale,
            )
            if amplitude >= cfg.vertical_scale:
                terrain_utils.random_uniform_terrain(
                    terrain,
                    min_height=-float(amplitude),
                    max_height=float(amplitude),
                    step=cfg.roughness_step,
                    downsampled_scale=cfg.roughness_downsampled_scale,
                )
            self._add_patch(terrain, row, col)
            self.patch_roughness[row, col] = amplitude

    def _add_patch(self, terrain, row, col):
        cfg = self.cfg
        start_x = self.border + row * self.length_per_env_pixels
        end_x = start_x + self.length_per_env_pixels
        start_y = self.border + col * self.width_per_env_pixels
        end_y = start_y + self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        center_x = (row + 0.5) * cfg.terrain_length
        center_y = (col + 0.5) * cfg.terrain_width
        x1 = int((cfg.terrain_length / 2.0 - 1.0) / cfg.horizontal_scale)
        x2 = int((cfg.terrain_length / 2.0 + 1.0) / cfg.horizontal_scale)
        y1 = int((cfg.terrain_width / 2.0 - 1.0) / cfg.horizontal_scale)
        y2 = int((cfg.terrain_width / 2.0 + 1.0) / cfg.horizontal_scale)
        origin_z = float(np.max(terrain.height_field_raw[x1:x2, y1:y2]))
        origin_z *= cfg.vertical_scale
        self.env_origins[row, col] = [center_x, center_y, origin_z]


class TinyMalRobust(LeggedRobot):
    """LeggedRobot with sim-to-real domain randomization."""

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        if self.cfg.terrain.mesh_type == "plane":
            self._create_ground_plane()
        elif self.cfg.terrain.mesh_type == "heightfield":
            self.terrain = TinyMalRoughTerrain(self.cfg.terrain)
            self._create_heightfield()
        elif self.cfg.terrain.mesh_type == "trimesh":
            self.terrain = TinyMalRoughTerrain(self.cfg.terrain)
            self._create_trimesh()
        else:
            raise ValueError(
                "TinyMalRobust supports terrain.mesh_type 'plane', 'heightfield', "
                "or 'trimesh', "
                f"got {self.cfg.terrain.mesh_type!r}"
            )
        self._create_envs()

    def _create_heightfield(self):
        params = gymapi.HeightFieldParams()
        params.column_scale = self.cfg.terrain.horizontal_scale
        params.row_scale = self.cfg.terrain.horizontal_scale
        params.vertical_scale = self.cfg.terrain.vertical_scale
        params.nbRows = self.terrain.tot_cols
        params.nbColumns = self.terrain.tot_rows
        params.transform.p.x = -self.cfg.terrain.border_size
        params.transform.p.y = -self.cfg.terrain.border_size
        params.static_friction = self.cfg.terrain.static_friction
        params.dynamic_friction = self.cfg.terrain.dynamic_friction
        params.restitution = self.cfg.terrain.restitution
        # Preview 4's Python binding requires a contiguous one-dimensional
        # buffer even though the logical terrain is a rows x columns grid.
        height_samples = np.ascontiguousarray(
            self.terrain.height_field_raw.reshape(-1)
        )
        self.gym.add_heightfield(self.sim, height_samples, params)
        self.height_samples = torch.as_tensor(
            self.terrain.height_field_raw, device=self.device, dtype=torch.float
        ).view(self.terrain.tot_rows, self.terrain.tot_cols)

    def _create_trimesh(self):
        vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
            self.terrain.height_field_raw,
            self.cfg.terrain.horizontal_scale,
            self.cfg.terrain.vertical_scale,
            self.cfg.terrain.slope_treshold,
        )
        params = gymapi.TriangleMeshParams()
        params.nb_vertices = vertices.shape[0]
        params.nb_triangles = triangles.shape[0]
        params.transform.p.x = -self.cfg.terrain.border_size
        params.transform.p.y = -self.cfg.terrain.border_size
        params.static_friction = self.cfg.terrain.static_friction
        params.dynamic_friction = self.cfg.terrain.dynamic_friction
        params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            np.ascontiguousarray(vertices.reshape(-1)),
            np.ascontiguousarray(triangles.reshape(-1)),
            params,
        )
        self.terrain.vertices = vertices
        self.terrain.triangles = triangles
        self.height_samples = torch.as_tensor(
            self.terrain.height_field_raw, device=self.device, dtype=torch.float
        ).view(self.terrain.tot_rows, self.terrain.tot_cols)

    def _get_env_origins(self):
        if self.cfg.terrain.mesh_type == "plane":
            super()._get_env_origins()
            return

        self.custom_origins = True
        terrain_levels = torch.randint(
            0, self.cfg.terrain.num_rows, (self.num_envs,), device=self.device
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        terrain_types = torch.div(
            env_ids * self.cfg.terrain.num_cols,
            self.num_envs,
            rounding_mode="floor",
        ).long()
        terrain_types = torch.clamp(terrain_types, max=self.cfg.terrain.num_cols - 1)
        terrain_origins = torch.as_tensor(
            self.terrain.env_origins, device=self.device, dtype=torch.float
        )
        self.terrain_levels = terrain_levels
        self.terrain_types = terrain_types
        self.terrain_origins = terrain_origins
        self.env_origins = terrain_origins[terrain_levels, terrain_types]

    def _process_dof_props(self, props, env_id):
        # The asset-level numpy record is reused by LeggedRobot.  Copy it so one
        # environment's randomized values cannot leak into the next.
        props = props.copy()
        props = super()._process_dof_props(props, env_id)
        cfg = self.cfg.domain_rand

        if cfg.randomize_joint_friction:
            values = np.random.uniform(
                cfg.joint_friction_range[0],
                cfg.joint_friction_range[1],
                size=self.num_dof,
            )
            props["friction"][:] = values
            if env_id == 0:
                self.joint_friction_coeffs = np.empty(
                    (self.num_envs, self.num_dof), dtype=np.float32
                )
            self.joint_friction_coeffs[env_id] = values

        if cfg.randomize_joint_armature:
            values = np.random.uniform(
                cfg.joint_armature_range[0],
                cfg.joint_armature_range[1],
                size=self.num_dof,
            )
            props["armature"][:] = values
            if env_id == 0:
                self.joint_armature_coeffs = np.empty(
                    (self.num_envs, self.num_dof), dtype=np.float32
                )
            self.joint_armature_coeffs[env_id] = values
        return props

    def _process_rigid_body_props(self, props, env_id):
        cfg = self.cfg.domain_rand
        if cfg.randomize_base_mass:
            mass_scale = np.random.uniform(
                cfg.base_mass_scale_range[0], cfg.base_mass_scale_range[1]
            )
            props[0].mass *= mass_scale
            if env_id == 0:
                self.base_mass_scales = np.empty(self.num_envs, dtype=np.float32)
            self.base_mass_scales[env_id] = mass_scale
        return props

    def _init_buffers(self):
        super()._init_buffers()
        cfg = self.cfg.domain_rand

        self._nominal_p_gains = self.p_gains.clone().unsqueeze(0).repeat(
            self.num_envs, 1
        )
        self._nominal_d_gains = self.d_gains.clone().unsqueeze(0).repeat(
            self.num_envs, 1
        )
        self.p_gains = self._nominal_p_gains.clone()
        self.d_gains = self._nominal_d_gains.clone()
        self.motor_kp_scales = torch.ones_like(self.p_gains)
        self.motor_kd_scales = torch.ones_like(self.d_gains)

        max_delay = int(cfg.control_delay_range[1])
        self.control_delays = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.action_history = torch.zeros(
            self.num_envs,
            max_delay + 1,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
        )
        self.applied_actions = torch.zeros_like(self.actions)

        self._push_force = torch.zeros(self.num_envs, 3, device=self.device)
        self._push_substeps_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._external_force_tensor = torch.zeros(
            self.num_envs, self.num_bodies, 3, device=self.device
        )
        self._external_torque_tensor = torch.zeros_like(self._external_force_tensor)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        cfg = self.cfg.domain_rand

        self.action_history[env_ids] = 0.0
        self.applied_actions[env_ids] = 0.0
        if cfg.randomize_control_delay:
            low, high = cfg.control_delay_range
            self.control_delays[env_ids] = torch.randint(
                int(low), int(high) + 1, (len(env_ids),), device=self.device
            )
        else:
            self.control_delays[env_ids] = 0

        if cfg.randomize_motor_gains:
            kp_low, kp_high = cfg.motor_kp_scale_range
            kd_low, kd_high = cfg.motor_kd_scale_range
            self.motor_kp_scales[env_ids] = torch_rand_float(
                kp_low,
                kp_high,
                (len(env_ids), self.num_actions),
                device=self.device,
            )
            self.motor_kd_scales[env_ids] = torch_rand_float(
                kd_low,
                kd_high,
                (len(env_ids), self.num_actions),
                device=self.device,
            )
        else:
            self.motor_kp_scales[env_ids] = 1.0
            self.motor_kd_scales[env_ids] = 1.0
        self.p_gains[env_ids] = (
            self._nominal_p_gains[env_ids] * self.motor_kp_scales[env_ids]
        )
        self.d_gains[env_ids] = (
            self._nominal_d_gains[env_ids] * self.motor_kd_scales[env_ids]
        )

        self._push_force[env_ids] = 0.0
        self._push_substeps_remaining[env_ids] = 0

    def _delayed_actions(self):
        self.action_history[:, 1:] = self.action_history[:, :-1].clone()
        self.action_history[:, 0] = self.actions
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.applied_actions[:] = self.action_history[env_ids, self.control_delays]
        return self.applied_actions

    def _schedule_random_pushes(self):
        cfg = self.cfg.domain_rand
        if not cfg.randomize_push_force:
            return
        interval = max(1, int(cfg.push_interval))
        push_ids = (
            (self.episode_length_buf > 0)
            & (self.episode_length_buf % interval == 0)
        ).nonzero(as_tuple=False).flatten()
        if len(push_ids) == 0:
            return

        angles = torch_rand_float(
            -np.pi, np.pi, (len(push_ids), 1), device=self.device
        ).squeeze(1)
        magnitudes = torch_rand_float(
            cfg.push_force_range[0],
            cfg.push_force_range[1],
            (len(push_ids), 1),
            device=self.device,
        ).squeeze(1)
        self._push_force[push_ids, 0] = magnitudes * torch.cos(angles)
        self._push_force[push_ids, 1] = magnitudes * torch.sin(angles)
        self._push_force[push_ids, 2] = 0.0

        min_substeps = int(
            round(cfg.push_duration_range_s[0] / self.sim_params.dt)
        )
        max_substeps = int(
            round(cfg.push_duration_range_s[1] / self.sim_params.dt)
        )
        self._push_substeps_remaining[push_ids] = torch.randint(
            max(1, min_substeps),
            max(1, max_substeps) + 1,
            (len(push_ids),),
            device=self.device,
        )

    def _apply_external_pushes(self):
        active = self._push_substeps_remaining > 0
        if not bool(active.any()):
            return
        self._external_force_tensor.zero_()
        self._external_force_tensor[active, 0] = self._push_force[active]
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self._external_force_tensor),
            gymtorch.unwrap_tensor(self._external_torque_tensor),
            gymapi.ENV_SPACE,
        )
        self._push_substeps_remaining[active] -= 1

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._schedule_random_pushes()

    def step(self, actions):
        """Advance physics while applying each environment's delayed action."""
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        delayed_actions = self._delayed_actions()

        self.render()
        for _ in range(self.cfg.control.decimation):
            self._apply_external_pushes()
            self.torques = self._compute_torques(delayed_actions).view(
                self.torques.shape
            )
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time - elapsed_time > 0:
                    time.sleep(sim_time - elapsed_time)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )
