#!/usr/bin/env python3
"""Train 1/2/3-link cart-poles natively in MuJoCo with PPO."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from actuatex_paths import ARTIFACTS_ROOT, RSL_RL_ROOT

if RSL_RL_ROOT.is_dir():
    sys.path.insert(0, str(RSL_RL_ROOT))

from inverted_pendulum_env import MjSerialInvertedPendulumEnv  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


DEFAULT_ITERATIONS = {1: 120, 2: 240, 3: 480}


def training_config(
    order: int,
    iterations: int,
    seed: int,
    *,
    learning_rate: float,
    entropy_coef: float,
    init_noise_std: float,
) -> dict:
    return {
        "seed": seed,
        "runner_class_name": "OnPolicyRunner",
        "policy": {
            "init_noise_std": init_noise_std,
            "actor_hidden_dims": [128, 128, 64],
            "critic_hidden_dims": [128, 128, 64],
            "activation": "elu",
        },
        "algorithm": {
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": entropy_coef,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": learning_rate,
            "schedule": "fixed",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "runner": {
            "policy_class_name": "ActorCritic",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": 32,
            "max_iterations": iterations,
            "save_interval": max(20, iterations // 4),
            "experiment_name": f"inverted_pendulum_order_{order}",
            "run_name": f"mujoco_seed_{seed}",
            "resume": False,
        },
    }


def warm_start(
    runner: OnPolicyRunner,
    checkpoint: Path,
    *,
    init_noise_std: float,
    reference_coef: float,
) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor_state = {
        key[len("actor.") :]: value
        for key, value in payload["model_state_dict"].items()
        if key.startswith("actor.")
    }
    runner.alg.actor_critic.actor.load_state_dict(actor_state, strict=True)
    with torch.no_grad():
        runner.alg.actor_critic.std.fill_(init_noise_std)
    if reference_coef > 0.0:
        teacher = copy.deepcopy(runner.alg.actor_critic.actor).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        runner.alg.reference_actor = teacher
        runner.alg.reference_loss_coef = reference_coef
        runner.alg.reference_mask_fn = lambda observations: torch.ones(
            observations.shape[0], dtype=torch.bool, device=observations.device
        )
    print(f"[curriculum] loaded weights from {checkpoint}")


def train_order(
    order: int,
    *,
    iterations: int,
    num_envs: int,
    num_threads: int,
    seed: int,
    init_checkpoint: Path | None,
    output_root: Path,
    learning_rate: float,
    entropy_coef: float,
    init_noise_std: float,
    reference_coef: float,
    critic_warmup_iterations: int,
    initial_angle_scale: float,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = output_root / f"order_{order}" / f"{timestamp}_seed{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = MjSerialInvertedPendulumEnv(
        order,
        num_envs=num_envs,
        num_threads=num_threads,
        seed=seed,
        initial_angle_scale=initial_angle_scale,
    )
    runner = OnPolicyRunner(
        env=env,
        train_cfg=training_config(
            order,
            iterations,
            seed,
            learning_rate=learning_rate,
            entropy_coef=entropy_coef,
            init_noise_std=init_noise_std,
        ),
        log_dir=str(log_dir),
        device="cpu",
    )
    if init_checkpoint is not None:
        warm_start(
            runner,
            init_checkpoint,
            init_noise_std=init_noise_std,
            reference_coef=reference_coef,
        )

    transitions = num_envs * 32 * iterations
    print(
        f"[train] order={order} envs={num_envs} iterations={iterations} "
        f"transitions={transitions:,}"
    )
    start = time.perf_counter()
    warmup = min(critic_warmup_iterations, iterations)
    if warmup:
        for parameter in runner.alg.actor_critic.actor.parameters():
            parameter.requires_grad_(False)
        runner.alg.actor_critic.std.requires_grad_(False)
        print(f"[train] critic-only warmup for {warmup} iterations")
        runner.learn(warmup)
        for parameter in runner.alg.actor_critic.actor.parameters():
            parameter.requires_grad_(True)
        runner.alg.actor_critic.std.requires_grad_(True)
    if iterations > warmup:
        runner.learn(iterations - warmup)
    elapsed = time.perf_counter() - start
    source = log_dir / f"model_{iterations}.pt"
    destination = (
        ARTIFACTS_ROOT
        / "checkpoints"
        / "inverted_pendulum"
        / f"mujoco_order_{order}.pt"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(
        f"[done] order={order} elapsed={elapsed:.2f}s "
        f"terminations={env.total_terminations} successes={env.total_timeouts} "
        f"checkpoint={destination}"
    )
    env.close()
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Train order 1, then warm-start 2 and 3 through the shared 14-D interface.",
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--lqr-warm-start",
        action="store_true",
        help="Initialize each selected order from its behavior-cloned LQR actor.",
    )
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--init-noise-std", type=float, default=0.20)
    parser.add_argument("--initial-angle-scale", type=float, default=1.0)
    parser.add_argument(
        "--reference-coef",
        type=float,
        default=0.0,
        help="Keep the warm-start actor as a frozen action-space teacher.",
    )
    parser.add_argument(
        "--critic-warmup-iterations",
        type=int,
        default=0,
        help="Freeze the warm-start actor while fitting the value function.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ARTIFACTS_ROOT / "inverted_pendulum" / "mujoco" / "logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.entropy_coef < 0.0 or args.reference_coef < 0.0:
        raise ValueError("entropy and reference coefficients must be non-negative")
    if args.init_noise_std <= 0.0 or args.initial_angle_scale <= 0.0:
        raise ValueError("noise and initial-angle scales must be positive")
    if args.critic_warmup_iterations < 0:
        raise ValueError("--critic-warmup-iterations cannot be negative")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    orders = (1, 2, 3) if args.curriculum else (args.order,)
    checkpoint = args.init_checkpoint
    for order in orders:
        if args.lqr_warm_start:
            checkpoint = (
                ARTIFACTS_ROOT
                / "checkpoints"
                / "inverted_pendulum"
                / f"lqr_seed_order_{order}.pt"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"missing {checkpoint}; run inverted_pendulum_lqr.py --export-seeds"
                )
        iterations = args.iterations or DEFAULT_ITERATIONS[order]
        checkpoint = train_order(
            order,
            iterations=iterations,
            num_envs=args.num_envs,
            num_threads=args.num_threads,
            seed=args.seed + order - 1,
            init_checkpoint=checkpoint,
            output_root=args.output_root,
            learning_rate=args.learning_rate,
            entropy_coef=args.entropy_coef,
            init_noise_std=args.init_noise_std,
            reference_coef=args.reference_coef,
            critic_warmup_iterations=args.critic_warmup_iterations,
            initial_angle_scale=args.initial_angle_scale,
        )


if __name__ == "__main__":
    main()
