#!/usr/bin/env python3
"""Train SAC or TD3 on the full-dynamics MuJoCo Sentinel."""

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

from sentinel_env import MjSentinelEnv  # noqa: E402
from tasks.inverted_pendulum.off_policy_rl import (  # noqa: E402
    OffPolicyConfig,
    ReplayBuffer,
    ReplayRatioScheduler,
    create_agent,
)
from tasks.robomaster.contract import ACTION_DIM, contract_sha256  # noqa: E402
from tasks.robomaster.locomotion import OBSERVATION_DIM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("sac", "td3"), required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--total-transitions", type=int, default=3_000_000)
    parser.add_argument("--learning-starts", type=int, default=50_000)
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--replay-sample-ratio",
        type=float,
        default=4.0,
        help=(
            "sampled replay rows per newly collected transition; converted to a "
            "fractional-safe gradient-update schedule"
        ),
    )
    parser.add_argument(
        "--updates-per-step",
        type=int,
        default=None,
        help=(
            "legacy fixed updates per vector-environment step; overrides the "
            "backend-independent replay sample ratio"
        ),
    )
    parser.add_argument("--exploration-noise", type=float, default=0.10)
    parser.add_argument("--maximum-command-delay-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--log-interval-steps", type=int, default=50)
    parser.add_argument("--output-checkpoint", type=Path)
    parser.add_argument("--output-metrics", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive = (
        args.num_envs,
        args.num_threads,
        args.total_transitions,
        args.replay_capacity,
        args.batch_size,
        args.log_interval_steps,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("environment, replay, update and logging sizes must be positive")
    if args.replay_sample_ratio <= 0.0:
        raise ValueError("replay sample ratio must be positive")
    if args.updates_per_step is not None and args.updates_per_step <= 0:
        raise ValueError("updates per step must be positive when set")
    if args.learning_starts < 0 or args.maximum_command_delay_steps < 0:
        raise ValueError("learning starts and command delay must be non-negative")
    if args.exploration_noise < 0.0:
        raise ValueError("exploration noise must be non-negative")
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
        hidden_dims=(512, 256, 128),
    )
    agent = create_agent(args.algorithm, config, device)
    replay = ReplayBuffer(
        OBSERVATION_DIM,
        ACTION_DIM,
        args.replay_capacity,
        device,
    )
    update_scheduler = ReplayRatioScheduler(
        batch_size=args.batch_size,
        replay_sample_ratio=args.replay_sample_ratio,
        updates_per_vector_step=args.updates_per_step,
    )
    if args.updates_per_step is not None:
        print(
            "[WARN] --updates-per-step uses backend-dependent legacy scheduling; "
            "prefer --replay-sample-ratio for comparisons"
        )
    env = MjSentinelEnv(
        num_envs=args.num_envs,
        num_threads=args.num_threads,
        seed=args.seed,
        maximum_command_delay_steps=args.maximum_command_delay_steps,
    )
    observation, _ = env.reset()
    collected_transitions = 0
    environment_steps = 0
    gradient_updates = 0
    reward_accumulator = 0.0
    last_metrics: dict[str, float] = {}
    start = time.perf_counter()
    try:
        while collected_transitions < args.total_transitions:
            transitions_before_step = collected_transitions
            transition_observation = observation.clone()
            if collected_transitions < args.learning_starts:
                action = torch.empty(
                    (args.num_envs, ACTION_DIM),
                    device=device,
                ).uniform_(-1.0, 1.0)
            else:
                action = agent.act(
                    observation.to(device),
                    explore=args.algorithm == "sac",
                )
                if args.algorithm == "td3" and args.exploration_noise:
                    action = (
                        action
                        + torch.randn_like(action) * args.exploration_noise
                    ).clamp(-1.0, 1.0)
            next_observation, _, reward, done, _ = env.step(action.cpu())
            replay.add(
                transition_observation,
                action,
                reward,
                next_observation,
                done,
            )
            observation = next_observation
            collected_transitions += args.num_envs
            environment_steps += 1
            reward_accumulator += float(reward.mean())
            if (
                collected_transitions >= args.learning_starts
                and replay.size >= args.batch_size
            ):
                eligible_new_transitions = collected_transitions - max(
                    transitions_before_step,
                    args.learning_starts,
                )
                updates_this_step = update_scheduler.updates_for(
                    eligible_new_transitions
                )
                for _ in range(updates_this_step):
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
                    f"sample_ratio={update_scheduler.achieved_ratio:.3f} "
                    f"reward={mean_reward:.4f} throughput={throughput:,.0f}/s "
                    f"{metric_text}"
                )
    finally:
        env.close()

    elapsed = time.perf_counter() - start
    checkpoint = args.output_checkpoint or (
        ARTIFACTS_ROOT
        / "checkpoints"
        / "sentinel"
        / f"mujoco_{args.algorithm}.pt"
    )
    metrics_path = args.output_metrics or (
        ARTIFACTS_ROOT
        / "sentinel"
        / "training"
        / f"mujoco_{args.algorithm}.json"
    )
    metadata = {
        "backend": "MuJoCo 3.10",
        "algorithm": args.algorithm,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "num_threads": args.num_threads,
        "collected_transitions": collected_transitions,
        "environment_steps": environment_steps,
        "gradient_updates": gradient_updates,
        "batch_size": args.batch_size,
        "replay_sample_ratio_target": (
            None if args.updates_per_step is not None else args.replay_sample_ratio
        ),
        "updates_per_vector_step_override": args.updates_per_step,
        "eligible_training_transitions": update_scheduler.eligible_transitions,
        "replay_sample_ratio_achieved": update_scheduler.achieved_ratio,
        "gradient_updates_per_transition": gradient_updates
        / max(1, update_scheduler.eligible_transitions),
        "wall_time_s": elapsed,
        "transitions_per_second": collected_transitions / max(elapsed, 1.0e-9),
        "maximum_command_delay_steps": args.maximum_command_delay_steps,
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
    main()
