#!/usr/bin/env python
"""Identify a PhysX cart-pole model and export an Isaac-native LQR actor seed."""

from __future__ import annotations

import argparse
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
parser.add_argument("--seed", type=int, default=121)
parser.add_argument("--state_epsilon", type=float, default=1.0e-3)
parser.add_argument("--force_epsilon", type=float, default=0.1)
parser.add_argument("--control_penalty", type=float, default=1.0)
parser.add_argument("--bc_steps", type=int, default=2400)
parser.add_argument("--bc_batch_size", type=int, default=16384)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--output", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if args_cli.state_epsilon <= 0.0 or args_cli.force_epsilon <= 0.0:
    parser.error("finite-difference epsilons must be positive")
if args_cli.control_penalty <= 0.0:
    parser.error("--control_penalty must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import solve_discrete_are  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import tinymal_lab  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from tasks.inverted_pendulum.contract import ACTION_FORCE_SCALE_N  # noqa: E402
from tasks.inverted_pendulum.lqr import (  # noqa: E402
    behavior_clone_lqr,
    make_actor_mlp,
)


def _read_state(env) -> torch.Tensor:
    env.joint_pos = env.robot.data.joint_pos.torch
    env.joint_vel = env.robot.data.joint_vel.torch
    qpos = torch.cat(
        (
            env.joint_pos[:, env._cart_dof_idx],
            env.joint_pos[:, env._pole_dof_idx],
        ),
        dim=-1,
    )
    qvel = torch.cat(
        (
            env.joint_vel[:, env._cart_dof_idx],
            env.joint_vel[:, env._pole_dof_idx],
        ),
        dim=-1,
    )
    return torch.cat((qpos, qvel), dim=-1)


def _write_state(env, state: torch.Tensor) -> None:
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    dof_count = env.cfg.order + 1
    joint_pos = state[:, :dof_count].clone()
    joint_vel = state[:, dof_count:].clone()
    root_pose = env.robot.data.default_root_pose.torch[env_ids].clone()
    root_pose[:, :3] += env.scene.env_origins[env_ids]
    root_velocity = env.robot.data.default_root_vel.torch[env_ids].clone()
    env.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    env.robot.write_root_velocity_to_sim_index(
        root_velocity=root_velocity, env_ids=env_ids
    )
    env.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    env.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
    env.episode_length_buf.zero_()
    env.actions.zero_()
    env.previous_actions.zero_()


def identify(env, state_epsilon: float, force_epsilon: float):
    state_dim = 2 * (env.cfg.order + 1)
    expected_envs = 2 * (state_dim + 1) + 1
    if env.num_envs != expected_envs:
        raise ValueError(f"identification requires exactly {expected_envs} envs")
    initial_state = torch.zeros(
        (env.num_envs, state_dim), dtype=torch.float32, device=env.device
    )
    actions = torch.zeros((env.num_envs, 1), device=env.device)
    for column in range(state_dim):
        initial_state[2 * column, column] = state_epsilon
        initial_state[2 * column + 1, column] = -state_epsilon
    input_pair = 2 * state_dim
    actions[input_pair, 0] = force_epsilon / ACTION_FORCE_SCALE_N
    actions[input_pair + 1, 0] = -force_epsilon / ACTION_FORCE_SCALE_N

    _write_state(env, initial_state)
    env.step(actions)
    next_state = _read_state(env).double().cpu().numpy()
    baseline_id = expected_envs - 1
    print("[IDENTIFY] body names =", env.robot.body_names)
    print(
        "[IDENTIFY] zero-state link positions =",
        env.robot.data.body_link_pos_w.torch[baseline_id].detach().cpu().tolist(),
    )
    print(
        "[IDENTIFY] zero-state COM positions =",
        env.robot.data.body_com_pos_w.torch[baseline_id].detach().cpu().tolist(),
    )
    matrix_a = np.empty((state_dim, state_dim), dtype=np.float64)
    for column in range(state_dim):
        matrix_a[:, column] = (next_state[2 * column] - next_state[2 * column + 1]) / (
            2.0 * state_epsilon
        )
    matrix_b = (
        (next_state[input_pair] - next_state[input_pair + 1]) / (2.0 * force_epsilon)
    )[:, None]
    return matrix_a, matrix_b, next_state


def design_lqr(order: int, matrix_a, matrix_b, control_penalty: float):
    state_dim = 2 * (order + 1)
    cumulative = np.tril(np.ones((order, order)))
    cost_q = np.zeros((state_dim, state_dim))
    cost_q[0, 0] = 1.0
    cost_q[1 : order + 1, 1 : order + 1] = cumulative.T @ cumulative * (30.0 / order)
    velocity_start = order + 1
    cost_q[velocity_start, velocity_start] = 0.2
    cost_q[velocity_start + 1 :, velocity_start + 1 :] = (
        cumulative.T @ cumulative * (3.0 / order)
    )
    cost_r = np.eye(1) * control_penalty
    riccati = solve_discrete_are(matrix_a, matrix_b, cost_q, cost_r)
    gain = np.linalg.solve(
        cost_r + matrix_b.T @ riccati @ matrix_b,
        matrix_b.T @ riccati @ matrix_a,
    )
    spectral_radius = float(
        np.max(np.abs(np.linalg.eigvals(matrix_a - matrix_b @ gain)))
    )
    return gain, spectral_radius


def main() -> None:
    # The finite-difference batch contains +/- pairs for every state and input.
    task_order = int(args_cli.task.split("-")[-3])
    state_dim = 2 * (task_order + 1)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device or "cuda:0",
        num_envs=2 * (state_dim + 1) + 1,
        use_fabric=True,
    )
    env_cfg.seed = args_cli.seed
    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env = gym_env.unwrapped
    order = int(env.cfg.order)
    print(f"[IDENTIFY] configured gravity = {env.cfg.sim.gravity}")
    print(
        "[IDENTIFY] PhysX gravity direction =",
        env.robot.data.GRAVITY_VEC_W.torch[0].detach().cpu().tolist(),
    )
    matrix_a, matrix_b, next_state = identify(
        env, args_cli.state_epsilon, args_cli.force_epsilon
    )
    print("[IDENTIFY] A =", matrix_a.tolist())
    print("[IDENTIFY] B =", matrix_b.tolist())
    print("[IDENTIFY] next states =", next_state.tolist())
    gain, spectral_radius = design_lqr(
        order, matrix_a, matrix_b, args_cli.control_penalty
    )

    torch.manual_seed(args_cli.seed)
    actor = make_actor_mlp().to(env.device)
    clone_metrics = behavior_clone_lqr(
        actor,
        order=order,
        gain=gain,
        steps=args_cli.bc_steps,
        batch_size=args_cli.bc_batch_size,
        seed=args_cli.seed + 100,
        learning_rate=1.0e-3,
    )
    if args_cli.checkpoint is None:
        checkpoint = (
            REPO_ROOT
            / "artifacts"
            / "checkpoints"
            / "inverted_pendulum"
            / f"isaac_lqr_seed_order_{order}.pt"
        )
    else:
        checkpoint = Path(args_cli.checkpoint).resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                f"actor.{key}": value.detach().cpu()
                for key, value in actor.state_dict().items()
            },
            "lqr_gain": torch.from_numpy(gain),
            "identified_a": torch.from_numpy(matrix_a),
            "identified_b": torch.from_numpy(matrix_b),
            "order": order,
            "behavior_cloning": clone_metrics,
            "backend": "Isaac Sim 6.0.1 GA / PhysX",
        },
        checkpoint,
    )
    result = {
        "schema_version": 1,
        "backend": "Isaac Sim 6.0.1 GA / PhysX",
        "task": args_cli.task,
        "order": order,
        "state_epsilon": args_cli.state_epsilon,
        "force_epsilon_n": args_cli.force_epsilon,
        "control_penalty": args_cli.control_penalty,
        "closed_loop_spectral_radius": spectral_radius,
        "gain": gain.tolist(),
        "matrix_a": matrix_a.tolist(),
        "matrix_b": matrix_b.tolist(),
        "finite_difference_next_states": next_state.tolist(),
        "behavior_cloning": clone_metrics,
        "checkpoint": str(checkpoint),
    }
    if args_cli.output is None:
        output = (
            REPO_ROOT
            / "artifacts"
            / "inverted_pendulum"
            / "evaluation"
            / f"isaac_lqr_identification_order_{order}.json"
        )
    else:
        output = Path(args_cli.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[INFO] wrote {output}")
    gym_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
