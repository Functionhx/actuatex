"""Left-right symmetry augmentation for the 48-D TinyMal deployment policy."""

from __future__ import annotations

import torch
from tensordict import TensorDict


def _switch_legs_left_right(values: torch.Tensor) -> torch.Tensor:
    switched = torch.empty_like(values)
    switched[..., 0:3] = values[..., 3:6]
    switched[..., 3:6] = values[..., 0:3]
    switched[..., 6:9] = values[..., 9:12]
    switched[..., 9:12] = values[..., 6:9]
    switched[..., [0, 3, 6, 9]] *= -1.0
    return switched


def _policy_left_right(obs: torch.Tensor) -> torch.Tensor:
    mirrored = obs.clone()
    device = obs.device
    mirrored[:, 0:3] *= torch.tensor([1.0, -1.0, 1.0], device=device)
    mirrored[:, 3:6] *= torch.tensor([-1.0, 1.0, -1.0], device=device)
    mirrored[:, 6:9] *= torch.tensor([1.0, -1.0, 1.0], device=device)
    mirrored[:, 9:12] *= torch.tensor([1.0, -1.0, -1.0], device=device)
    mirrored[:, 12:24] = _switch_legs_left_right(obs[:, 12:24])
    mirrored[:, 24:36] = _switch_legs_left_right(obs[:, 24:36])
    mirrored[:, 36:48] = _switch_legs_left_right(obs[:, 36:48])
    return mirrored


def _privileged_left_right(obs: torch.Tensor) -> torch.Tensor:
    mirrored = obs.clone()
    # [joint torques 12, foot contacts 4, height 1, stair level 1, actuator latents 3]
    mirrored[:, 0:12] = _switch_legs_left_right(obs[:, 0:12])
    mirrored[:, 12:16] = obs[:, [13, 12, 15, 14]]
    return mirrored


@torch.no_grad()
def compute_symmetric_states(env, obs: TensorDict | None = None,
                             actions: torch.Tensor | None = None):
    """Return original and left-right-mirrored batches for PPO augmentation."""
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"]
        obs_aug["policy"][batch_size:] = _policy_left_right(obs["policy"])
        if "privileged" in obs.keys():
            obs_aug["privileged"][:batch_size] = obs["privileged"]
            obs_aug["privileged"][batch_size:] = _privileged_left_right(obs["privileged"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.empty(
            batch_size * 2, actions.shape[1], device=actions.device, dtype=actions.dtype
        )
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = _switch_legs_left_right(actions)
    else:
        actions_aug = None
    return obs_aug, actions_aug
