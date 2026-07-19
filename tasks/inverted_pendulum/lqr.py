"""Torch utilities shared by exact-LQR evaluation and PPO warm starts."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .contract import (
    ACTION_FORCE_SCALE_N,
    CART_POSITION_OBS_SCALE,
    CART_VELOCITY_OBS_SCALE,
    INITIAL_ANGLE_RANGE_RAD,
    OBSERVATION_DIM,
    POLE_VELOCITY_OBS_SCALE,
    build_observation,
    validate_order,
)


class LQRActor(nn.Module):
    """Exact saturated state-feedback controller with the public 14-D input."""

    def __init__(self, order: int, gain: np.ndarray) -> None:
        super().__init__()
        self.order = validate_order(order)
        gain = np.asarray(gain, dtype=np.float32)
        expected_shape = (1, 2 * (order + 1))
        if gain.shape != expected_shape:
            raise ValueError(f"gain must have shape {expected_shape}, got {gain.shape}")
        self.register_buffer("gain", torch.from_numpy(gain))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        qpos = [observation[..., 0] / CART_POSITION_OBS_SCALE]
        qvel = [observation[..., 1] / CART_VELOCITY_OBS_SCALE]
        for pole_index in range(self.order):
            offset = 2 + 3 * pole_index
            qpos.append(
                torch.atan2(observation[..., offset], observation[..., offset + 1])
            )
            qvel.append(observation[..., offset + 2] / POLE_VELOCITY_OBS_SCALE)
        state = torch.stack(qpos + qvel, dim=-1)
        force = -(state @ self.gain.T)
        return torch.clamp(force / ACTION_FORCE_SCALE_N, -1.0, 1.0)


def behavior_clone_lqr(
    actor: nn.Module,
    *,
    order: int,
    gain: np.ndarray,
    steps: int = 800,
    batch_size: int = 4096,
    seed: int = 1,
    learning_rate: float = 1.0e-3,
) -> dict[str, float]:
    """Fit a nonlinear actor to LQR on a local randomized state cloud."""

    order = validate_order(order)
    device = next(actor.parameters()).device
    teacher = LQRActor(order, gain).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    final_loss = float("nan")
    final_max_error = float("nan")

    actor.train()
    for _ in range(steps):
        cart_position = rng.uniform(-1.2, 1.2, batch_size)
        cart_velocity = rng.uniform(-1.5, 1.5, batch_size)
        angle_range = max(INITIAL_ANGLE_RANGE_RAD[order] * 2.0, 0.12)
        relative_angles = rng.uniform(-angle_range, angle_range, (batch_size, order))
        pole_velocities = rng.uniform(-2.0, 2.0, (batch_size, order))
        observation = torch.from_numpy(
            build_observation(
                cart_position,
                cart_velocity,
                relative_angles,
                pole_velocities,
                order,
            )
        ).to(device)
        with torch.no_grad():
            target = teacher(observation)
        prediction = actor(observation)
        loss = torch.mean(torch.square(prediction - target))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        optimizer.step()
        final_loss = float(loss.item())
        final_max_error = float(torch.max(torch.abs(prediction - target)).item())

    actor.eval()
    return {
        "steps": float(steps),
        "batch_size": float(batch_size),
        "final_mse": final_loss,
        "final_batch_max_abs_error": final_max_error,
    }


def make_actor_mlp() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(OBSERVATION_DIM, 128),
        nn.ELU(),
        nn.Linear(128, 128),
        nn.ELU(),
        nn.Linear(128, 64),
        nn.ELU(),
        nn.Linear(64, 1),
    )
