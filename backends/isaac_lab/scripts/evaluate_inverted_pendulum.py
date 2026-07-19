#!/usr/bin/env python
"""Deterministically evaluate 1/2/3-link pendulum policies in Isaac Sim 6."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-InvertedPendulum-1-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--episodes", type=int, default=1024)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--initial_angle_scale", type=float, default=1.0)
parser.add_argument("--output", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
if args_cli.episodes <= 0 or args_cli.num_envs <= 0:
    parser.error("--episodes and --num_envs must be positive")
if args_cli.initial_angle_scale <= 0.0:
    parser.error("--initial_angle_scale must be positive")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import tinymal_lab  # noqa: E402,F401
from isaaclab.envs import DirectRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import importlib.metadata as metadata  # noqa: E402
from tasks.inverted_pendulum.off_policy_rl import (  # noqa: E402
    CHECKPOINT_FORMAT,
    load_off_policy_actor,
)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_actor(runner: OnPolicyRunner, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
    if "actor_state_dict" in payload:
        actor_state = payload["actor_state_dict"]
    elif "model_state_dict" in payload:
        actor_state = {
            "mlp." + key[len("actor.") :]: value
            for key, value in payload["model_state_dict"].items()
            if key.startswith("actor.")
        }
    else:
        raise KeyError(f"{checkpoint} has no supported actor state")
    incompatible = runner.alg.actor.load_state_dict(actor_state, strict=False)
    missing = [
        key for key in incompatible.missing_keys if not key.startswith("distribution.")
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "actor checkpoint mismatch: "
            f"missing={missing}, unexpected={list(incompatible.unexpected_keys)}"
        )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = min(args_cli.num_envs, args_cli.episodes)
    env_cfg.seed = args_cli.seed
    env_cfg.initial_angle_range *= args_cli.initial_angle_scale
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    agent_cfg.seed = args_cli.seed
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    checkpoint = Path(args_cli.checkpoint).resolve()
    payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
    off_policy_checkpoint = payload.get("checkpoint_format") == CHECKPOINT_FORMAT
    if off_policy_checkpoint:
        policy = load_off_policy_actor(
            payload,
            device=agent_cfg.device,
        )
    else:
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        _load_actor(runner, checkpoint)
        policy = runner.get_inference_policy(device=agent_cfg.device)

    observations = env.get_observations().to(agent_cfg.device)
    episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    durations: list[float] = []
    successes = 0
    angle_squared_sum = 0.0
    cart_squared_sum = 0.0
    abs_action_sum = 0.0
    metric_samples = 0

    while len(durations) < args_cli.episodes:
        with torch.inference_mode():
            policy_input = (
                observations["policy"] if off_policy_checkpoint else observations
            )
            actions = policy(policy_input)
            absolute_angles = raw_env.unwrapped._absolute_angles()
            cart_position = raw_env.unwrapped.joint_pos[
                :, raw_env.unwrapped._cart_dof_idx[0]
            ]
            angle_squared_sum += float(torch.square(absolute_angles).sum().item())
            cart_squared_sum += float(torch.square(cart_position).sum().item())
            abs_action_sum += float(torch.abs(actions).sum().item())
            metric_samples += env.num_envs
            observations, _, dones, extras = env.step(actions.to(env.device))
            observations = observations.to(agent_cfg.device)
        episode_steps += 1
        done_ids = torch.nonzero(dones, as_tuple=False).flatten()
        if done_ids.numel():
            timeouts = extras.get("time_outs")
            for env_id in done_ids.tolist():
                if len(durations) >= args_cli.episodes:
                    break
                durations.append(
                    float(episode_steps[env_id].item() * env.unwrapped.step_dt)
                )
                if timeouts is not None and bool(timeouts[env_id].item()):
                    successes += 1
            episode_steps[done_ids] = 0

    order = int(env_cfg.order)
    result = {
        "schema_version": 1,
        "backend": "Isaac Sim 6.0.1 GA / Isaac Lab 3.0.0-beta2.patch1",
        "task": args_cli.task,
        "order": order,
        "episodes": args_cli.episodes,
        "successes": successes,
        "success_rate": successes / args_cli.episodes,
        "mean_balance_duration_s": float(np.mean(durations)),
        "median_balance_duration_s": float(np.median(durations)),
        "absolute_pole_angle_rmse_rad": float(
            np.sqrt(angle_squared_sum / (metric_samples * order))
        ),
        "cart_position_rmse_m": float(np.sqrt(cart_squared_sum / metric_samples)),
        "mean_abs_normalized_action": abs_action_sum / metric_samples,
        "initial_angle_scale": args_cli.initial_angle_scale,
        "num_envs": env.num_envs,
        "seed": args_cli.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
    }
    if args_cli.output is None:
        output = (
            REPO_ROOT
            / "artifacts"
            / "inverted_pendulum"
            / "evaluation"
            / f"isaac_lab_to_isaac_order_{order}.json"
        )
    else:
        output = Path(args_cli.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[INFO] wrote {output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
