"""Native PhysX-5 TinyMal environment behavior."""

import torch

from isaaclab.envs import ManagerBasedRLEnv


class TinymalNativeRLEnv(ManagerBasedRLEnv):
    """Match legged_gym's ``only_positive_rewards`` behavior."""

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, extras = super().step(action)
        reward = torch.clamp(reward, min=0.0)
        self.reward_buf = reward
        return obs, reward, terminated, truncated, extras
