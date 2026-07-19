"""Load PPO, SAC and TD3 Sentinel actors behind one inference contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tasks.inverted_pendulum.off_policy_rl import (
    CHECKPOINT_FORMAT as OFF_POLICY_CHECKPOINT_FORMAT,
    load_off_policy_actor,
)

from .contract import ACTION_DIM
from .locomotion import OBSERVATION_DIM


@dataclass(frozen=True)
class LoadedPolicy:
    actor: nn.Module
    algorithm: str
    checkpoint_format: str
    observation_dim: int
    action_dim: int
    metadata: dict[str, Any]


def _ppo_mlp(state: dict[str, torch.Tensor], prefix: str) -> nn.Sequential:
    weight_keys = sorted(
        (
            key
            for key in state
            if key.startswith(prefix) and key.endswith(".weight")
        ),
        key=lambda key: int(key[len(prefix) :].split(".", maxsplit=1)[0]),
    )
    if not weight_keys:
        raise ValueError("PPO checkpoint contains no actor linear layers")
    layers: list[nn.Module] = []
    for index, weight_key in enumerate(weight_keys):
        weight = state[weight_key]
        if weight.ndim != 2:
            raise ValueError(f"actor weight {weight_key} is not a matrix")
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        layers.append(linear)
        if index + 1 < len(weight_keys):
            layers.append(nn.ELU())
    actor = nn.Sequential(*layers)
    actor_state = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
        and (key.endswith(".weight") or key.endswith(".bias"))
    }
    actor.load_state_dict(actor_state, strict=True)
    return actor


def load_policy(
    checkpoint: Path | str | dict[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> LoadedPolicy:
    """Load deterministic inference from all ActuateX training backends."""

    if isinstance(checkpoint, dict):
        payload = checkpoint
    else:
        payload = torch.load(
            checkpoint,
            map_location=device,
            weights_only=False,
        )
    if payload.get("checkpoint_format") == OFF_POLICY_CHECKPOINT_FORMAT:
        observation_dim = int(payload["observation_dim"])
        action_dim = int(payload["action_dim"])
        if (observation_dim, action_dim) != (OBSERVATION_DIM, ACTION_DIM):
            raise ValueError(
                "off-policy checkpoint dimensions do not match Sentinel: "
                f"{observation_dim}x{action_dim}"
            )
        actor = load_off_policy_actor(payload, device=device)
        return LoadedPolicy(
            actor=actor,
            algorithm=str(payload["off_policy_algorithm"]),
            checkpoint_format="actuatex_off_policy_v1",
            observation_dim=observation_dim,
            action_dim=action_dim,
            metadata=dict(payload.get("metadata", {})),
        )

    if "model_state_dict" in payload:
        state = payload["model_state_dict"]
        actor = _ppo_mlp(state, "actor.").to(device).eval()
        checkpoint_format = "rsl_rl_1_actor_critic"
    elif "actor_state_dict" in payload and any(
        key.startswith("mlp.") for key in payload["actor_state_dict"]
    ):
        state = payload["actor_state_dict"]
        actor = _ppo_mlp(state, "mlp.").to(device).eval()
        checkpoint_format = "rsl_rl_5_split_actor"
    else:
        raise ValueError("unrecognized Sentinel policy checkpoint format")

    first_linear = next(module for module in actor if isinstance(module, nn.Linear))
    last_linear = next(
        module for module in reversed(actor) if isinstance(module, nn.Linear)
    )
    if (first_linear.in_features, last_linear.out_features) != (
        OBSERVATION_DIM,
        ACTION_DIM,
    ):
        raise ValueError(
            "PPO checkpoint dimensions do not match Sentinel: "
            f"{first_linear.in_features}x{last_linear.out_features}"
        )
    return LoadedPolicy(
        actor=actor,
        algorithm="ppo",
        checkpoint_format=checkpoint_format,
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        metadata=dict(payload.get("infos") or {}),
    )
