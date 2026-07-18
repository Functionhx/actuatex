"""Mixed flat-command and stair-climbing rehearsal for one TinyMal actor."""

import torch
from isaacgym.torch_utils import quat_apply

from legged_gym.envs.tinymal.tinymal_robust_stairs import TinyMalRobustStairs
from legged_gym.utils.math import wrap_to_pi


class TinyMalRobustMixed(TinyMalRobustStairs):
    """Split vectorized environments between flat commands and fixed stairs.

    Flat cells retain the complete vx/vy/yaw command distribution and receive
    sustained force pushes.  Stair cells receive a 0.3 m/s world-x command and
    heading correction.  Both subsets share one actor and the complete physics,
    actuator, latency, and observation randomization.
    """

    def _step_height_for_env(self, env_index):
        flat_count = int(round(self.num_envs * self.cfg.stairs.flat_env_fraction))
        if env_index < flat_count:
            return 0.0
        return float(self.cfg.stairs.step_height)

    def _init_buffers(self):
        super()._init_buffers()
        self.stair_env_mask = self.stair_step_heights > 0.0
        self.flat_env_mask = ~self.stair_env_mask

    def _resample_commands(self, env_ids):
        # heading_command is disabled in the mixed config, so the inherited
        # implementation samples the full direct vx/vy/yaw distribution.
        super()._resample_commands(env_ids)
        stair_ids = env_ids[self.stair_env_mask[env_ids]]
        if len(stair_ids) > 0:
            self.commands[stair_ids, 0] = self.cfg.stairs.command_speed
            self.commands[stair_ids, 1] = 0.0
            self.commands[stair_ids, 2] = 0.0
            self.commands[stair_ids, 3] = 0.0

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        stair_ids = self.stair_env_mask.nonzero(as_tuple=False).flatten()
        if len(stair_ids) == 0:
            return
        forward = quat_apply(self.base_quat[stair_ids], self.forward_vec[stair_ids])
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[stair_ids, 0] = self.cfg.stairs.command_speed
        self.commands[stair_ids, 1] = 0.0
        self.commands[stair_ids, 2] = torch.clamp(
            -self.cfg.stairs.heading_gain * wrap_to_pi(heading), -1.0, 1.0
        )

    def compute_observations(self):
        """Expose an optional out-of-range stair-mode marker to the actor."""
        super().compute_observations()
        if hasattr(self, "stair_env_mask"):
            self.obs_buf[self.stair_env_mask, 9] = (
                self.cfg.stairs.policy_command_speed * self.obs_scales.lin_vel
            )

    def _schedule_random_pushes(self):
        super()._schedule_random_pushes()
        # Preserve clean contact exploration on stairs; flat rehearsal cells
        # carry the sustained-force objective for the shared policy.
        self._push_substeps_remaining[self.stair_env_mask] = 0
        self._push_force[self.stair_env_mask] = 0.0

    def _reward_lateral_position(self):
        return super()._reward_lateral_position() * self.stair_env_mask

    def _reward_world_forward_progress(self):
        return super()._reward_world_forward_progress() * self.stair_env_mask
