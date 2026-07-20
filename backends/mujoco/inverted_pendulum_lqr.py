#!/usr/bin/env python3
"""Design and evaluate traditional discrete-LQR cart-pole controllers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are
import torch

from actuatex_paths import ARTIFACTS_ROOT, REPO_ROOT, RSL_RL_ROOT

if RSL_RL_ROOT.is_dir():
    sys.path.insert(0, str(RSL_RL_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from inverted_pendulum_env import MjSerialInvertedPendulumEnv  # noqa: E402
from tasks.inverted_pendulum.contract import DECIMATION  # noqa: E402
from tasks.inverted_pendulum.lqr import (  # noqa: E402
    LQRActor,
    behavior_clone_lqr,
    make_actor_mlp,
)


def design_lqr(order: int, control_penalty: float = 1.0) -> tuple[np.ndarray, dict]:
    model_path = (
        REPO_ROOT
        / "robots"
        / "inverted_pendulum"
        / "mjcf"
        / f"actuatex_cartpole_{order}.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state_dim = 2 * model.nv + model.na
    one_step_a = np.zeros((state_dim, state_dim))
    one_step_b = np.zeros((state_dim, model.nu))
    mujoco.mjd_transitionFD(
        model, data, 1.0e-6, True, one_step_a, one_step_b, None, None
    )

    # Hold the same force for DECIMATION physics steps, matching both RL tasks.
    transition_a = np.eye(state_dim)
    transition_b = np.zeros_like(one_step_b)
    for _ in range(DECIMATION):
        transition_b = one_step_a @ transition_b + one_step_b
        transition_a = one_step_a @ transition_a

    cumulative = np.tril(np.ones((order, order)))
    cost_q = np.zeros((state_dim, state_dim))
    cost_q[0, 0] = 1.0
    cost_q[1 : order + 1, 1 : order + 1] = cumulative.T @ cumulative * (30.0 / order)
    velocity_start = order + 1
    cost_q[velocity_start, velocity_start] = 0.2
    cost_q[velocity_start + 1 :, velocity_start + 1 :] = (
        cumulative.T @ cumulative * (3.0 / order)
    )
    cost_r = np.eye(model.nu) * control_penalty
    riccati = solve_discrete_are(transition_a, transition_b, cost_q, cost_r)
    gain = np.linalg.solve(
        cost_r + transition_b.T @ riccati @ transition_b,
        transition_b.T @ riccati @ transition_a,
    )
    spectral_radius = float(
        np.max(np.abs(np.linalg.eigvals(transition_a - transition_b @ gain)))
    )
    audit = {
        "linearization": "MuJoCo central finite difference at upright",
        "physics_steps_per_control": DECIMATION,
        "control_penalty": control_penalty,
        "closed_loop_spectral_radius": spectral_radius,
        "gain": gain.tolist(),
    }
    return gain, audit


def evaluate_lqr(
    order: int,
    gain: np.ndarray,
    *,
    episodes: int,
    seed: int,
    num_threads: int,
    initial_angle_scale: float,
) -> dict:
    actor = LQRActor(order, gain)
    env = MjSerialInvertedPendulumEnv(
        order,
        num_envs=episodes,
        num_threads=num_threads,
        seed=seed,
        initial_angle_scale=initial_angle_scale,
    )
    observation, _ = env.reset()
    active = np.ones(episodes, dtype=bool)
    success = np.zeros(episodes, dtype=bool)
    duration_steps = np.zeros(episodes, dtype=np.int64)
    angle_squared_sum = np.zeros(episodes)
    cart_squared_sum = np.zeros(episodes)
    sample_count = np.zeros(episodes, dtype=np.int64)
    with torch.inference_mode():
        for step in range(env.max_episode_length):
            action = actor(observation)
            observation, _, _, _, _ = env.step(action)
            ids = np.flatnonzero(active)
            angle_squared_sum[ids] += np.mean(
                np.square(env.last_absolute_angles[ids]), axis=1
            )
            cart_squared_sum[ids] += np.square(env.last_cart_position[ids])
            sample_count[ids] += 1
            finished = active & (env.last_terminal | env.last_timeout)
            success[finished] = (
                env.last_timeout[finished] & ~env.last_terminal[finished]
            )
            duration_steps[finished] = step + 1
            active[finished] = False
            if not active.any():
                break
    duration_steps[active] = env.max_episode_length
    env.close()
    counts = np.maximum(sample_count, 1)
    return {
        "order": order,
        "episodes": episodes,
        "successes": int(success.sum()),
        "success_rate": float(success.mean()),
        "mean_balance_duration_s": float(duration_steps.mean() * env.dt),
        "absolute_pole_angle_rmse_rad": float(
            np.sqrt(np.mean(angle_squared_sum / counts))
        ),
        "cart_position_rmse_m": float(np.sqrt(np.mean(cart_squared_sum / counts))),
        "initial_angle_scale": initial_angle_scale,
        "seed": seed,
    }


def export_lqr_seed(order: int, gain: np.ndarray, args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed + order)
    actor = make_actor_mlp()
    clone_metrics = behavior_clone_lqr(
        actor,
        order=order,
        gain=gain,
        steps=args.bc_steps,
        batch_size=args.bc_batch_size,
        seed=args.seed + 100 + order,
    )
    checkpoint_path = (
        ARTIFACTS_ROOT
        / "checkpoints"
        / "inverted_pendulum"
        / f"lqr_seed_order_{order}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                f"actor.{key}": value.cpu() for key, value in actor.state_dict().items()
            },
            "lqr_gain": torch.from_numpy(gain),
            "order": order,
            "behavior_cloning": clone_metrics,
        },
        checkpoint_path,
    )
    return {"path": str(checkpoint_path.resolve()), **clone_metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, choices=(1, 2, 3))
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--initial-angle-scale", type=float, default=1.0)
    parser.add_argument("--control-penalty", type=float, default=1.0)
    parser.add_argument("--export-seeds", action="store_true")
    parser.add_argument("--bc-steps", type=int, default=800)
    parser.add_argument("--bc-batch-size", type=int, default=4096)
    parser.add_argument(
        "--out",
        type=Path,
        default=ARTIFACTS_ROOT
        / "inverted_pendulum"
        / "evaluation"
        / "lqr_baseline.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    orders = (args.order,) if args.order else (1, 2, 3)
    results = []
    gains = {}
    seeds = {}
    for order in orders:
        gain, audit = design_lqr(order, args.control_penalty)
        gains[str(order)] = audit
        result = evaluate_lqr(
            order,
            gain,
            episodes=args.episodes,
            seed=args.seed + order - 1,
            num_threads=args.num_threads,
            initial_angle_scale=args.initial_angle_scale,
        )
        results.append(result)
        if args.export_seeds:
            seeds[str(order)] = export_lqr_seed(order, gain, args)
        print(
            f"order={order} success={result['successes']}/{args.episodes} "
            f"duration={result['mean_balance_duration_s']:.3f}s"
        )

    output = {
        "schema_version": 1,
        "controller": "discrete LQR with saturated cart force",
        "backend": f"MuJoCo {mujoco.__version__}",
        "gains": gains,
        "results": results,
        "behavior_cloned_seeds": seeds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
