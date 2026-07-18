"""TinyMal flat-ground env with external base-force injection for push-recovery.

Physics, reward, observation and PD control are identical to the baseline
`tinymal` task (class LeggedRobot). The only addition is a force window that
applies a world-frame xy force to the base rigid body on every physics substep
for a scheduled number of substeps, faithfully emulating a sustained push.
The force API mirrors NVIDIA's apply_forces.py example: a (num_envs, num_bodies,
3) tensor unwrapped to the flat rigid-body tensor, body index 0 = base.
"""

import torch
from isaacgym import gymapi, gymtorch  # noqa: F401  (unwrap_tensor / ENV_SPACE)

from legged_gym.envs.base.legged_robot import LeggedRobot


class TinyMalPush(LeggedRobot):
    """LeggedRobot + a per-env scheduled base-force window."""

    def _init_buffers(self):
        super()._init_buffers()
        # External push state (per env).
        self._push_force = torch.zeros(self.num_envs, 3, device=self.device)
        self._push_substeps_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Reusable per-body force/torque tensors. Layout (num_envs, num_bodies, 3)
        # maps, via gymtorch.unwrap_tensor, to Isaac Gym's flat
        # (num_envs*num_bodies, 3) rigid-body tensor.
        self._ext_force_tensor = torch.zeros(
            self.num_envs, self.num_bodies, 3, device=self.device
        )
        self._ext_torque_tensor = torch.zeros(
            self.num_envs, self.num_bodies, 3, device=self.device
        )

    def schedule_base_force(self, force_xy, substeps):
        """Schedule a world-frame base push.

        Args:
            force_xy: (num_envs, 2) world x/y force in Newtons.
            substeps: (num_envs,) long tensor; number of physics substeps
                (sim_params.dt each) to keep the force active. 0 => no push.
        """
        self._push_force[:, 0] = force_xy[:, 0]
        self._push_force[:, 1] = force_xy[:, 1]
        self._push_force[:, 2] = 0.0
        self._push_substeps_remaining[:] = substeps.to(self.device)

    def _apply_external_force(self):
        active = self._push_substeps_remaining > 0
        if not bool(active.any()):
            return
        self._ext_force_tensor.zero_()
        self._ext_force_tensor[active, 0, :2] = self._push_force[active, :2]
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self._ext_force_tensor),
            gymtorch.unwrap_tensor(self._ext_torque_tensor),
            gymapi.ENV_SPACE,
        )
        self._push_substeps_remaining[active] -= 1

    def step(self, actions):
        """LeggedRobot.step with an external base force injected before each
        physics substep while the scheduled window is active."""
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()
        for _ in range(self.cfg.control.decimation):
            self._apply_external_force()
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self.gym.simulate(self.sim)
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
