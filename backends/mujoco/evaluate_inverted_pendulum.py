#!/usr/bin/env python3
"""Evaluate either backend's 14-D cart-pole actor in MuJoCo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import mujoco
import torch
from torch import nn

from actuatex_paths import ARTIFACTS_ROOT, REPO_ROOT, RSL_RL_ROOT

if RSL_RL_ROOT.is_dir():
    sys.path.insert(0, str(RSL_RL_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from inverted_pendulum_env import MjSerialInvertedPendulumEnv  # noqa: E402
from tasks.inverted_pendulum.off_policy_rl import (  # noqa: E402
    CHECKPOINT_FORMAT,
    load_off_policy_actor,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _linear_stack(state: dict[str, torch.Tensor]) -> nn.Sequential:
    layers = []
    linear_indices = sorted(
        {
            int(key.split(".")[-2])
            for key in state
            if key.endswith(".weight") and key.split(".")[-2].isdigit()
        }
    )
    if not linear_indices:
        raise ValueError("checkpoint has no actor linear layers")
    for layer_number, index in enumerate(linear_indices):
        weight_key = next(
            key
            for key in state
            if key.endswith(f".{index}.weight") or key == f"{index}.weight"
        )
        prefix = weight_key[: -len("weight")]
        weight = state[weight_key]
        bias = state[prefix + "bias"]
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        layers.append(linear)
        if layer_number < len(linear_indices) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers).eval()


def load_actor(path: Path) -> nn.Module:
    try:
        module = torch.jit.load(str(path), map_location="cpu")
        return module.eval()
    except (RuntimeError, ValueError):
        pass

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_format") == CHECKPOINT_FORMAT:
        return load_off_policy_actor(payload)
    if "actor_state_dict" in payload:
        raw_state = payload["actor_state_dict"]
        state = {
            key[len("mlp.") :]: value
            for key, value in raw_state.items()
            if key.startswith("mlp.")
        }
    elif "model_state_dict" in payload:
        raw_state = payload["model_state_dict"]
        state = {
            key[len("actor.") :]: value
            for key, value in raw_state.items()
            if key.startswith("actor.")
        }
    else:
        raise KeyError(f"unsupported checkpoint keys: {sorted(payload)}")
    actor = _linear_stack(state)
    probe = actor(torch.zeros(1, 14))
    if probe.shape != (1, 1):
        raise ValueError(f"actor shape is {tuple(probe.shape)}, expected (1, 1)")
    return actor


def evaluate(args: argparse.Namespace) -> dict:
    actor = load_actor(args.checkpoint)
    env = MjSerialInvertedPendulumEnv(
        args.order,
        num_envs=args.episodes,
        num_threads=args.num_threads,
        seed=args.seed,
        initial_angle_scale=args.initial_angle_scale,
    )
    observation, _ = env.reset()
    active = np.ones(args.episodes, dtype=bool)
    success = np.zeros(args.episodes, dtype=bool)
    duration_steps = np.zeros(args.episodes, dtype=np.int64)
    cart_squared_sum = np.zeros(args.episodes, dtype=np.float64)
    angle_squared_sum = np.zeros(args.episodes, dtype=np.float64)
    sample_count = np.zeros(args.episodes, dtype=np.int64)
    action_abs_sum = np.zeros(args.episodes, dtype=np.float64)

    with torch.inference_mode():
        for step in range(env.max_episode_length):
            action = actor(observation).reshape(args.episodes, 1)
            observation, _, _, _, _ = env.step(action)
            active_ids = np.flatnonzero(active)
            if active_ids.size:
                cart_squared_sum[active_ids] += np.square(
                    env.last_cart_position[active_ids]
                )
                angle_squared_sum[active_ids] += np.mean(
                    np.square(env.last_absolute_angles[active_ids]), axis=1
                )
                action_abs_sum[active_ids] += np.abs(
                    action.cpu().numpy().reshape(-1)[active_ids]
                )
                sample_count[active_ids] += 1
            finished = active & (env.last_terminal | env.last_timeout)
            duration_steps[finished] = step + 1
            success[finished] = (
                env.last_timeout[finished] & ~env.last_terminal[finished]
            )
            active[finished] = False
            if not active.any():
                break
    duration_steps[active] = env.max_episode_length
    env.close()

    counts = np.maximum(sample_count, 1)
    result = {
        "schema_version": 1,
        "backend": f"MuJoCo {mujoco.__version__}",
        "policy_source": args.policy_source,
        "order": args.order,
        "episodes": args.episodes,
        "seed": args.seed,
        "initial_angle_scale": args.initial_angle_scale,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "successes": int(success.sum()),
        "success_rate": float(success.mean()),
        "mean_balance_duration_s": float(duration_steps.mean() * env.dt),
        "median_balance_duration_s": float(np.median(duration_steps) * env.dt),
        "cart_position_rmse_m": float(np.sqrt(np.mean(cart_squared_sum / counts))),
        "absolute_pole_angle_rmse_rad": float(
            np.sqrt(np.mean(angle_squared_sum / counts))
        ),
        "mean_abs_normalized_action": float(np.mean(action_abs_sum / counts)),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-source", default="mujoco")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--initial-angle-scale", type=float, default=1.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = (
            ARTIFACTS_ROOT
            / "checkpoints"
            / "inverted_pendulum"
            / f"mujoco_order_{args.order}.pt"
        )
    if args.out is None:
        args.out = (
            ARTIFACTS_ROOT
            / "inverted_pendulum"
            / "evaluation"
            / f"{args.policy_source}_to_mujoco_order_{args.order}.json"
        )
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.initial_angle_scale <= 0.0:
        parser.error("--initial-angle-scale must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    result = evaluate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
