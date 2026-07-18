"""TinyMal environment with collision-accurate stair meshes."""

import numpy as np
import torch
from isaacgym import gymapi

from legged_gym.envs.base.legged_robot import LeggedRobot


def _append_box(vertices, triangles, x0, x1, y0, y1, z1):
    """Append a closed axis-aligned box to triangle-mesh buffers."""
    offset = len(vertices)
    vertices.extend(
        [
            (x0, y0, 0.0),
            (x1, y0, 0.0),
            (x1, y1, 0.0),
            (x0, y1, 0.0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ]
    )
    faces = (
        (0, 2, 1), (0, 3, 2),  # bottom
        (4, 5, 6), (4, 6, 7),  # top
        (0, 1, 5), (0, 5, 4),
        (3, 7, 6), (3, 6, 2),
        (0, 4, 7), (0, 7, 3),
        (1, 2, 6), (1, 6, 5),
    )
    triangles.extend(tuple(offset + index for index in face) for face in faces)


class TinyMalStairs(LeggedRobot):
    """LeggedRobot variant that adds one staircase in front of every robot."""

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs()
        self._create_stair_meshes()
        self._create_recording_camera()

    def _step_height_for_env(self, env_index):
        cfg = self.cfg.stairs
        if self.num_envs == 1 or not cfg.curriculum:
            return float(cfg.step_height)
        level = env_index % max(1, cfg.curriculum_levels)
        denominator = max(1, cfg.curriculum_levels - 1)
        ratio = level / denominator
        return float(
            cfg.min_step_height
            + ratio * (cfg.max_step_height - cfg.min_step_height)
        )

    def _create_stair_meshes(self):
        cfg = self.cfg.stairs
        vertices = []
        triangles = []
        heights = []
        origins = self.env_origins.detach().cpu().numpy()

        for env_index, origin in enumerate(origins):
            rise = self._step_height_for_env(env_index)
            heights.append(rise)
            # Mixed-task environments use a zero rise to represent a true
            # flat rehearsal cell.  Do not create coplanar, zero-volume boxes
            # on top of the ground plane for those cells.
            if rise <= 0.0:
                continue
            y0 = float(origin[1] - cfg.total_width / 2.0)
            y1 = float(origin[1] + cfg.total_width / 2.0)
            start = float(origin[0] + cfg.start_x)
            for step in range(cfg.num_steps):
                x0 = start + step * cfg.step_width
                x1 = start + (step + 1) * cfg.step_width
                _append_box(vertices, triangles, x0, x1, y0, y1, (step + 1) * rise)
            platform_x0 = start + cfg.num_steps * cfg.step_width
            platform_x1 = platform_x0 + cfg.top_length
            _append_box(
                vertices,
                triangles,
                platform_x0,
                platform_x1,
                y0,
                y1,
                cfg.num_steps * rise,
            )

        vertex_array = np.asarray(vertices, dtype=np.float32)
        triangle_array = np.asarray(triangles, dtype=np.uint32)
        params = gymapi.TriangleMeshParams()
        params.nb_vertices = vertex_array.shape[0]
        params.nb_triangles = triangle_array.shape[0]
        params.static_friction = self.cfg.terrain.static_friction
        params.dynamic_friction = self.cfg.terrain.dynamic_friction
        params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            vertex_array.flatten(),
            triangle_array.flatten(),
            params,
        )
        self._stair_step_heights_np = np.asarray(heights, dtype=np.float32)

    def _create_recording_camera(self):
        self.recording_camera = None
        if not self.cfg.stairs.record_video:
            return
        if self.graphics_device_id < 0:
            raise RuntimeError(
                "Video recording requires an off-screen graphics device; set "
                "cfg.env.enable_offscreen_rendering=True."
            )
        props = gymapi.CameraProperties()
        props.width = self.cfg.stairs.camera_width
        props.height = self.cfg.stairs.camera_height
        props.horizontal_fov = 68.0
        self.recording_camera = self.gym.create_camera_sensor(self.envs[0], props)
        origin = self.env_origins[0].detach().cpu().numpy()
        camera_position = gymapi.Vec3(
            float(origin[0] + 1.35),
            float(origin[1] - 2.15),
            0.82,
        )
        camera_target = gymapi.Vec3(
            float(origin[0] + 1.25),
            float(origin[1]),
            0.16,
        )
        self.gym.set_camera_location(
            self.recording_camera,
            self.envs[0],
            camera_position,
            camera_target,
        )

    def _init_buffers(self):
        super()._init_buffers()
        self.stair_step_heights = torch.as_tensor(
            self._stair_step_heights_np, device=self.device
        )

    def terrain_height_below_base(self):
        cfg = self.cfg.stairs
        local_x = self.root_states[: self.num_envs, 0] - self.env_origins[:, 0]
        local_y = self.root_states[: self.num_envs, 1] - self.env_origins[:, 1]
        relative_step = torch.ceil((local_x - cfg.start_x) / cfg.step_width)
        level = torch.clamp(relative_step, min=0, max=cfg.num_steps)
        inside_stair = torch.abs(local_y) <= cfg.total_width / 2.0
        return torch.where(
            inside_stair,
            level * self.stair_step_heights,
            torch.zeros_like(level),
        )

    def _reward_base_height(self):
        # World-frame height would penalize every successful upward step.
        relative_height = (
            self.root_states[: self.num_envs, 2] - self.terrain_height_below_base()
        )
        return torch.square(relative_height - self.cfg.rewards.base_height_target)

    def _reward_lateral_position(self):
        local_y = self.root_states[: self.num_envs, 1] - self.env_origins[:, 1]
        return torch.square(local_y)

    def _reward_world_forward_progress(self):
        """Reward task-frame progress even when the body yaws at a riser.

        The inherited velocity reward is expressed in the body frame and can
        therefore reward walking sideways after the robot turns.  World-frame
        x velocity is aligned with the staircase and supplies a dense gradient
        while contact exploration is still below the strict pass threshold.
        """
        return torch.clamp(
            self.root_states[: self.num_envs, 7], min=0.0, max=0.6
        )
