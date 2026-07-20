#!/usr/bin/env python3
"""Train a PPO locomotion policy on the full-dynamics MuJoCo Sentinel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from actuatex_paths import ARTIFACTS_ROOT, REPO_ROOT, RSL_RL_ROOT

sys.path.insert(0, str(REPO_ROOT))
if RSL_RL_ROOT.is_dir():
    sys.path.insert(0, str(RSL_RL_ROOT))

from sentinel_env import MjSentinelEnv  # noqa: E402
from tasks.robomaster.contract import contract_sha256  # noqa: E402

try:
    from rsl_rl.runners import OnPolicyRunner  # noqa: E402
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "rsl_rl is required; install it or set RSL_RL_ROOT to its source tree"
    ) from error


def training_config(args: argparse.Namespace) -> dict:
    return {
        "seed": args.seed,
        "runner_class_name": "OnPolicyRunner",
        "policy": {
            "init_noise_std": args.initial_noise_std,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": args.entropy_coefficient,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": args.learning_rate,
            "schedule": args.schedule,
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "runner": {
            "policy_class_name": "ActorCritic",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": 24,
            "max_iterations": args.max_iterations,
            "save_interval": args.save_interval,
            "experiment_name": "sentinel_full_dynamics_mujoco",
            "run_name": f"ppo_seed{args.seed}",
            "resume": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--max-iterations", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=0.005)
    parser.add_argument("--initial-noise-std", type=float, default=0.30)
    parser.add_argument("--schedule", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--maximum-command-delay-steps", type=int, default=0)
    parser.add_argument("--no-observation-noise", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--output-checkpoint", type=Path)
    parser.add_argument("--output-metrics", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive = (
        args.num_envs,
        args.num_threads,
        args.max_iterations,
        args.save_interval,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("environment, iteration and save sizes must be positive")
    if args.maximum_command_delay_steps < 0:
        raise ValueError("maximum command delay cannot be negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    run_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = args.log_dir or (
        ARTIFACTS_ROOT
        / "mujoco"
        / "logs"
        / "sentinel_full_dynamics"
        / f"{run_stamp}_ppo_seed{args.seed}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    env = MjSentinelEnv(
        num_envs=args.num_envs,
        num_threads=args.num_threads,
        seed=args.seed,
        add_noise=not args.no_observation_noise,
        maximum_command_delay_steps=args.maximum_command_delay_steps,
    )
    start = time.perf_counter()
    try:
        runner = OnPolicyRunner(
            env=env,
            train_cfg=training_config(args),
            log_dir=str(log_dir),
            device=str(device),
        )
        runner.learn(args.max_iterations)
    finally:
        env.close()
    elapsed = time.perf_counter() - start

    final_checkpoint = log_dir / f"model_{args.max_iterations}.pt"
    checkpoint = args.output_checkpoint or (
        ARTIFACTS_ROOT / "checkpoints/sentinel/mujoco_ppo.pt"
    )
    metrics_path = args.output_metrics or (
        ARTIFACTS_ROOT / "sentinel/training/mujoco_ppo.json"
    )
    metadata = {
        "backend": "MuJoCo 3.10",
        "algorithm": "ppo",
        "seed": args.seed,
        "num_envs": args.num_envs,
        "num_threads": args.num_threads,
        "iterations": args.max_iterations,
        "transitions": args.num_envs * 24 * args.max_iterations,
        "wall_time_s": elapsed,
        "maximum_command_delay_steps": args.maximum_command_delay_steps,
        "contract_sha256": contract_sha256(),
        "checkpoint": str(checkpoint.resolve()),
    }
    payload = torch.load(
        final_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    payload["infos"] = metadata
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {checkpoint}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
