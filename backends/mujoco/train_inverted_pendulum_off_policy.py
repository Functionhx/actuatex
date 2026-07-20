#!/usr/bin/env python3
"""Train the shared SAC or TD3 actor in the MuJoCo cart-pole backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from actuatex_paths import ARTIFACTS_ROOT, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))

from inverted_pendulum_env import MjSerialInvertedPendulumEnv  # noqa: E402
from tasks.inverted_pendulum.contract import (  # noqa: E402
    ACTION_DIM,
    OBSERVATION_DIM,
)
from tasks.inverted_pendulum.off_policy_rl import (  # noqa: E402
    OffPolicyConfig,
    ReplayBuffer,
    create_agent,
)


DEFAULT_TRANSITIONS = {1: 2_000_000, 2: 5_000_000, 3: 10_000_000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("sac", "td3"), required=True)
    parser.add_argument("--order", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--total-transitions", type=int)
    parser.add_argument("--learning-starts", type=int, default=20_000)
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--updates-per-step", type=int, default=4)
    parser.add_argument("--exploration-noise", type=float, default=0.10)
    parser.add_argument("--initial-angle-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--log-interval-steps", type=int, default=100)
    parser.add_argument("--output-checkpoint", type=Path)
    parser.add_argument("--output-metrics", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_transitions = args.total_transitions or DEFAULT_TRANSITIONS[args.order]
    positive_values = (
        args.num_envs,
        args.num_threads,
        total_transitions,
        args.replay_capacity,
        args.batch_size,
        args.updates_per_step,
        args.log_interval_steps,
    )
    if any(value <= 0 for value in positive_values):
        raise ValueError(
            "environment, replay, update and logging sizes must be positive"
        )
    if args.learning_starts < 0:
        raise ValueError("learning-starts cannot be negative")
    if args.exploration_noise < 0.0:
        raise ValueError("exploration-noise cannot be negative")
    if args.initial_angle_scale <= 0.0:
        raise ValueError("initial-angle-scale must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    config = OffPolicyConfig(
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
    )
    agent = create_agent(args.algorithm, config, device)
    replay = ReplayBuffer(
        OBSERVATION_DIM,
        ACTION_DIM,
        args.replay_capacity,
        device,
    )
    env = MjSerialInvertedPendulumEnv(
        args.order,
        num_envs=args.num_envs,
        num_threads=args.num_threads,
        seed=args.seed,
        initial_angle_scale=args.initial_angle_scale,
    )
    observation, _ = env.reset()
    collected_transitions = 0
    environment_steps = 0
    gradient_updates = 0
    last_metrics: dict[str, float] = {}
    reward_accumulator = 0.0
    start = time.perf_counter()
    try:
        while collected_transitions < total_transitions:
            # MjSerialInvertedPendulumEnv reuses ``obs_buf`` and overwrites it
            # in-place inside step().  Preserve s_t before stepping; otherwise
            # replay would silently store s_{t+1} in both state fields.
            transition_observation = observation.clone()
            if collected_transitions < args.learning_starts:
                action_device = torch.empty(
                    (args.num_envs, ACTION_DIM),
                    device=device,
                ).uniform_(-1.0, 1.0)
            else:
                action_device = agent.act(
                    observation.to(device),
                    explore=args.algorithm == "sac",
                )
                if args.algorithm == "td3" and args.exploration_noise:
                    action_device = (
                        action_device
                        + torch.randn_like(action_device) * args.exploration_noise
                    ).clamp(-1.0, 1.0)
            next_observation, _, reward, done, _ = env.step(action_device.cpu())
            replay.add(
                transition_observation,
                action_device,
                reward,
                next_observation,
                done,
            )
            observation = next_observation
            collected_transitions += args.num_envs
            environment_steps += 1
            reward_accumulator += float(reward.mean().item())
            if (
                collected_transitions >= args.learning_starts
                and replay.size >= args.batch_size
            ):
                for _ in range(args.updates_per_step):
                    last_metrics = agent.update(replay, args.batch_size)
                    gradient_updates += 1
            if environment_steps % args.log_interval_steps == 0:
                elapsed = time.perf_counter() - start
                throughput = collected_transitions / max(elapsed, 1.0e-9)
                mean_reward = reward_accumulator / args.log_interval_steps
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
    checkpoint = args.output_checkpoint or (
        ARTIFACTS_ROOT
        / "checkpoints"
        / "inverted_pendulum"
        / f"mujoco_{args.algorithm}_order_{args.order}.pt"
    )
    metrics_path = args.output_metrics or (
        ARTIFACTS_ROOT
        / "inverted_pendulum"
        / "training"
        / f"mujoco_{args.algorithm}_order_{args.order}.json"
    )
    metadata = {
        "backend": "MuJoCo",
        "algorithm": args.algorithm,
        "order": args.order,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "num_threads": args.num_threads,
        "collected_transitions": collected_transitions,
        "environment_steps": environment_steps,
        "gradient_updates": gradient_updates,
        "wall_time_s": elapsed,
        "transitions_per_second": collected_transitions / max(elapsed, 1.0e-9),
        "initial_angle_scale": args.initial_angle_scale,
        "last_update_metrics": last_metrics,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.checkpoint(metadata), checkpoint)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {checkpoint}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
