#!/usr/bin/env python
"""Train the shared SAC or TD3 actor on the Isaac Lab Sentinel task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--algorithm", choices=("sac", "td3"), required=True)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Sentinel-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--total_transitions", type=int, default=5_000_000)
parser.add_argument("--learning_starts", type=int, default=100_000)
parser.add_argument("--replay_capacity", type=int, default=2_000_000)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--updates_per_step", type=int, default=4)
parser.add_argument("--exploration_noise", type=float, default=0.10)
parser.add_argument("--seed", type=int, default=1101)
parser.add_argument("--log_interval_steps", type=int, default=50)
parser.add_argument("--output_checkpoint", type=Path)
parser.add_argument("--output_metrics", type=Path)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
if remaining_args:
    parser.error(f"unrecognized arguments: {' '.join(remaining_args)}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import tinymal_lab  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from tasks.inverted_pendulum.off_policy_rl import (  # noqa: E402
    OffPolicyConfig,
    ReplayBuffer,
    create_agent,
)
from tasks.robomaster.contract import ACTION_DIM, contract_sha256  # noqa: E402
from tasks.robomaster.locomotion import OBSERVATION_DIM  # noqa: E402


def main() -> None:
    positive = (
        args_cli.num_envs,
        args_cli.total_transitions,
        args_cli.replay_capacity,
        args_cli.batch_size,
        args_cli.updates_per_step,
        args_cli.log_interval_steps,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("environment, replay, update and logging sizes must be positive")
    if args_cli.learning_starts < 0:
        raise ValueError("learning_starts cannot be negative")
    if args_cli.exploration_noise < 0.0:
        raise ValueError("exploration_noise cannot be negative")
    device = torch.device(args_cli.device or "cuda:0")
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=str(device),
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    observation_dict, _ = env.reset()
    observation = observation_dict["policy"]
    if observation.shape != (args_cli.num_envs, OBSERVATION_DIM):
        raise RuntimeError(
            f"Sentinel observation contract drifted to {observation.shape}"
        )

    config = OffPolicyConfig(
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        hidden_dims=(512, 256, 128),
    )
    agent = create_agent(args_cli.algorithm, config, device)
    replay = ReplayBuffer(
        OBSERVATION_DIM,
        ACTION_DIM,
        args_cli.replay_capacity,
        device,
    )
    collected_transitions = 0
    environment_steps = 0
    gradient_updates = 0
    reward_accumulator = 0.0
    last_metrics: dict[str, float] = {}
    start = time.perf_counter()
    try:
        while collected_transitions < args_cli.total_transitions:
            transition_observation = observation.clone()
            if collected_transitions < args_cli.learning_starts:
                action = torch.empty(
                    (args_cli.num_envs, ACTION_DIM),
                    dtype=torch.float32,
                    device=device,
                ).uniform_(-1.0, 1.0)
            else:
                action = agent.act(
                    observation,
                    explore=args_cli.algorithm == "sac",
                )
                if args_cli.algorithm == "td3" and args_cli.exploration_noise:
                    action = (
                        action
                        + torch.randn_like(action) * args_cli.exploration_noise
                    ).clamp(-1.0, 1.0)
            next_observation_dict, reward, terminated, truncated, _ = env.step(
                action
            )
            next_observation = next_observation_dict["policy"]
            done = terminated | truncated
            replay.add(
                transition_observation,
                action,
                reward,
                next_observation,
                done,
            )
            observation = next_observation
            collected_transitions += args_cli.num_envs
            environment_steps += 1
            reward_accumulator += float(reward.mean())
            if (
                collected_transitions >= args_cli.learning_starts
                and replay.size >= args_cli.batch_size
            ):
                for _ in range(args_cli.updates_per_step):
                    last_metrics = agent.update(replay, args_cli.batch_size)
                    gradient_updates += 1
            if environment_steps % args_cli.log_interval_steps == 0:
                elapsed = time.perf_counter() - start
                throughput = collected_transitions / max(elapsed, 1.0e-9)
                mean_reward = reward_accumulator / args_cli.log_interval_steps
                reward_accumulator = 0.0
                metric_text = " ".join(
                    f"{key}={value:.4g}" for key, value in last_metrics.items()
                )
                print(
                    f"step={environment_steps} transitions={collected_transitions:,} "
                    f"replay={replay.size:,} updates={gradient_updates:,} "
                    f"reward={mean_reward:.4f} throughput={throughput:,.0f}/s "
                    f"{metric_text}"
                )
    finally:
        env.close()

    elapsed = time.perf_counter() - start
    checkpoint = args_cli.output_checkpoint or (
        REPO_ROOT
        / "artifacts"
        / "checkpoints"
        / "sentinel"
        / f"isaac_lab_{args_cli.algorithm}.pt"
    )
    metrics_path = args_cli.output_metrics or (
        REPO_ROOT
        / "artifacts"
        / "sentinel"
        / "training"
        / f"isaac_lab_{args_cli.algorithm}.json"
    )
    metadata = {
        "backend": "Isaac Sim 6.0.1 GA / Isaac Lab 3.0.0-beta2.patch1 / PhysX 5",
        "task": args_cli.task,
        "algorithm": args_cli.algorithm,
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "collected_transitions": collected_transitions,
        "environment_steps": environment_steps,
        "gradient_updates": gradient_updates,
        "wall_time_s": elapsed,
        "transitions_per_second": collected_transitions / max(elapsed, 1.0e-9),
        "contract_sha256": contract_sha256(),
        "last_update_metrics": last_metrics,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.checkpoint(metadata), checkpoint)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {checkpoint}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
