#!/usr/bin/env python
"""Tune a saturated PhysX LQR controller with GPU-parallel CEM.

The local discrete LQR solution is an excellent stabilizer near upright, but
force saturation makes its region of attraction sensitive to the individual
gain coefficients.  This script keeps the feedback structure interpretable
and uses the Cross-Entropy Method (CEM) to maximize full-episode survival over
the actual nonlinear Isaac Sim dynamics.  The winning controller is then
distilled into the same MLP used by PPO and sim2sim evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-InvertedPendulum-3-Direct-v0")
parser.add_argument("--base_checkpoint", type=str, required=True)
parser.add_argument("--population", type=int, default=128)
parser.add_argument("--episodes_per_candidate", type=int, default=256)
parser.add_argument("--generations", type=int, default=24)
parser.add_argument("--elite_fraction", type=float, default=0.125)
parser.add_argument("--initial_log_std", type=float, default=0.30)
parser.add_argument("--minimum_log_std", type=float, default=0.025)
parser.add_argument("--validation_episodes_per_candidate", type=int, default=1024)
parser.add_argument("--seed", type=int, default=501)
parser.add_argument("--bc_steps", type=int, default=3000)
parser.add_argument("--bc_batch_size", type=int, default=16384)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--output", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

if args_cli.population < 4 or args_cli.episodes_per_candidate <= 0:
    parser.error("--population must be at least 4 and episode counts must be positive")
if args_cli.generations <= 0:
    parser.error("--generations must be positive")
if not 0.0 < args_cli.elite_fraction < 1.0:
    parser.error("--elite_fraction must lie in (0, 1)")
if args_cli.initial_log_std <= 0.0 or args_cli.minimum_log_std <= 0.0:
    parser.error("CEM standard deviations must be positive")
if args_cli.validation_episodes_per_candidate <= 0:
    parser.error("--validation_episodes_per_candidate must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import tinymal_lab  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from tasks.inverted_pendulum.contract import (  # noqa: E402
    ACTION_FORCE_SCALE_N,
    INITIAL_ANGLE_RANGE_RAD,
)
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


def _sample_initial_states(
    *, order: int, episodes: int, rng: np.random.Generator
) -> np.ndarray:
    dof_count = order + 1
    state = np.zeros((episodes, 2 * dof_count), dtype=np.float32)
    state[:, 0] = rng.uniform(-0.25, 0.25, episodes)
    state[:, 1:dof_count] = rng.uniform(
        -INITIAL_ANGLE_RANGE_RAD[order],
        INITIAL_ANGLE_RANGE_RAD[order],
        (episodes, order),
    )
    state[:, dof_count] = rng.uniform(-0.10, 0.10, episodes)
    state[:, dof_count + 1 :] = rng.uniform(-0.10, 0.10, (episodes, order))
    return state


def evaluate_population(
    gym_env,
    gains: torch.Tensor,
    *,
    episodes_per_candidate: int,
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    """Evaluate all gains on identical initial states in one PhysX batch."""

    env = gym_env.unwrapped
    candidate_count, state_dim = gains.shape
    expected_envs = candidate_count * episodes_per_candidate
    if expected_envs != env.num_envs:
        raise ValueError(
            f"population batch requires {expected_envs} envs, got {env.num_envs}"
        )
    initial_states = _sample_initial_states(
        order=env.cfg.order,
        episodes=episodes_per_candidate,
        rng=rng,
    )
    initial_states = torch.from_numpy(initial_states).to(env.device)
    state = (
        initial_states.unsqueeze(0)
        .expand(candidate_count, -1, -1)
        .reshape(env.num_envs, state_dim)
        .contiguous()
    )
    expanded_gains = (
        gains[:, None, :]
        .expand(-1, episodes_per_candidate, -1)
        .reshape(env.num_envs, state_dim)
    )
    _write_state(env, state)

    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(alive)
    duration_steps = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    max_steps = int(env.max_episode_length)
    for _ in range(max_steps):
        force = -torch.sum(state * expanded_gains, dim=-1, keepdim=True)
        actions = torch.clamp(force / ACTION_FORCE_SCALE_N, -1.0, 1.0)
        _, _, terminated, truncated, _ = gym_env.step(actions)
        duration_steps += alive
        first_done = alive & (terminated | truncated)
        success |= first_done & truncated & ~terminated
        alive &= ~(terminated | truncated)
        if not bool(torch.any(alive)):
            break
        state = _read_state(env)

    duration = duration_steps.view(candidate_count, episodes_per_candidate).float()
    success = success.view(candidate_count, episodes_per_candidate)
    success_rate = success.float().mean(dim=1)
    mean_duration_fraction = duration.mean(dim=1) / max_steps
    # A completed ten-second episode is the primary objective.  The duration
    # term supplies a smooth ranking before any candidate completes an episode.
    score = success_rate + 0.10 * mean_duration_fraction
    return {
        "score": score,
        "success_rate": success_rate,
        "mean_duration_steps": duration.mean(dim=1),
    }


def _load_gain(path: Path) -> np.ndarray:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    if "lqr_gain" not in payload:
        raise KeyError(f"{path} has no lqr_gain")
    gain = payload["lqr_gain"].detach().cpu().numpy().astype(np.float64)
    if gain.ndim != 2 or gain.shape[0] != 1:
        raise ValueError(f"invalid LQR gain shape {gain.shape}")
    return gain


def main() -> None:
    base_checkpoint = Path(args_cli.base_checkpoint).resolve()
    base_gain = _load_gain(base_checkpoint)
    state_dim = base_gain.shape[1]
    order = state_dim // 2 - 1
    expected_task_order = int(args_cli.task.split("-")[-3])
    if order != expected_task_order:
        raise ValueError(
            f"checkpoint order {order} does not match task order {expected_task_order}"
        )

    archive_slots = args_cli.generations + 1
    validation_candidate_count = math.ceil(archive_slots / 8.0) * 8
    search_envs = args_cli.population * args_cli.episodes_per_candidate
    validation_envs = (
        validation_candidate_count * args_cli.validation_episodes_per_candidate
    )
    if search_envs != validation_envs:
        raise ValueError(
            "search and validation batches must use the same environment count; "
            f"got {search_envs} and {validation_envs}. Adjust population or "
            "validation episodes."
        )

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device or "cuda:0",
        num_envs=search_envs,
        use_fabric=True,
    )
    env_cfg.seed = args_cli.seed
    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    gym_env.reset(seed=args_cli.seed)
    device = gym_env.unwrapped.device
    rng = np.random.default_rng(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    base_gain_tensor = torch.from_numpy(base_gain[0].astype(np.float32)).to(device)
    mean = torch.zeros(state_dim, dtype=torch.float32, device=device)
    std = torch.full_like(mean, args_cli.initial_log_std)
    elite_count = max(2, round(args_cli.population * args_cli.elite_fraction))
    history: list[dict] = []
    archived_gains = [base_gain_tensor.clone()]

    for generation in range(args_cli.generations):
        log_scales = mean + std * torch.randn(
            (args_cli.population, state_dim), device=device
        )
        log_scales.clamp_(-1.5, 1.5)
        # Always retain both the analytic LQR and the current distribution mean.
        log_scales[0].zero_()
        log_scales[1].copy_(mean)
        gains = base_gain_tensor * torch.exp(log_scales)
        metrics = evaluate_population(
            gym_env,
            gains,
            episodes_per_candidate=args_cli.episodes_per_candidate,
            rng=rng,
        )
        elite_ids = torch.topk(metrics["score"], elite_count).indices
        elite_log_scales = log_scales[elite_ids]
        new_mean = elite_log_scales.mean(dim=0)
        new_std = elite_log_scales.std(dim=0, unbiased=False)
        mean.mul_(0.20).add_(new_mean, alpha=0.80)
        std.mul_(0.20).add_(new_std, alpha=0.80)
        std.clamp_(min=args_cli.minimum_log_std, max=0.80)
        best_id = int(torch.argmax(metrics["score"]).item())
        archived_gains.append(gains[best_id].detach().clone())
        record = {
            "generation": generation,
            "best_score": float(metrics["score"][best_id].item()),
            "best_success_rate": float(metrics["success_rate"][best_id].item()),
            "best_mean_duration_s": float(
                metrics["mean_duration_steps"][best_id].item()
                * gym_env.unwrapped.step_dt
            ),
            "population_mean_success_rate": float(
                metrics["success_rate"].mean().item()
            ),
            "mean_log_scale": mean.detach().cpu().tolist(),
            "std_log_scale": std.detach().cpu().tolist(),
        }
        history.append(record)
        print(
            f"[CEM {generation + 1:02d}/{args_cli.generations}] "
            f"success={record['best_success_rate']:.3f} "
            f"duration={record['best_mean_duration_s']:.3f}s "
            f"population_mean={record['population_mean_success_rate']:.3f}"
        )

    archive = torch.stack(archived_gains)
    if archive.shape[0] < validation_candidate_count:
        padding = archive[0:1].expand(validation_candidate_count - archive.shape[0], -1)
        archive = torch.cat((archive, padding), dim=0)
    validation = evaluate_population(
        gym_env,
        archive,
        episodes_per_candidate=args_cli.validation_episodes_per_candidate,
        rng=np.random.default_rng(args_cli.seed + 10_000),
    )
    valid_count = len(archived_gains)
    best_archive_id = int(torch.argmax(validation["score"][:valid_count]).item())
    best_gain = archive[best_archive_id].detach().cpu().numpy()[None, :]
    direct_validation = {
        "candidate_archive_index": best_archive_id,
        "episodes": args_cli.validation_episodes_per_candidate,
        "success_rate": float(validation["success_rate"][best_archive_id].item()),
        "mean_balance_duration_s": float(
            validation["mean_duration_steps"][best_archive_id].item()
            * gym_env.unwrapped.step_dt
        ),
    }
    print(
        "[VALIDATION] "
        f"success={direct_validation['success_rate']:.4f} "
        f"duration={direct_validation['mean_balance_duration_s']:.3f}s"
    )

    torch.manual_seed(args_cli.seed + 20_000)
    actor = make_actor_mlp().to(device)
    clone_metrics = behavior_clone_lqr(
        actor,
        order=order,
        gain=best_gain,
        steps=args_cli.bc_steps,
        batch_size=args_cli.bc_batch_size,
        seed=args_cli.seed + 30_000,
        learning_rate=1.0e-3,
    )
    if args_cli.checkpoint is None:
        checkpoint = (
            REPO_ROOT
            / "artifacts"
            / "checkpoints"
            / "inverted_pendulum"
            / f"isaac_cem_lqr_seed_order_{order}.pt"
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
            "lqr_gain": torch.from_numpy(best_gain),
            "base_lqr_gain": torch.from_numpy(base_gain),
            "order": order,
            "behavior_cloning": clone_metrics,
            "direct_validation": direct_validation,
            "backend": "Isaac Sim 6.0.1 GA / PhysX / CEM",
        },
        checkpoint,
    )

    result = {
        "schema_version": 1,
        "backend": "Isaac Sim 6.0.1 GA / PhysX",
        "optimizer": "GPU-parallel Cross-Entropy Method over saturated LQR gains",
        "task": args_cli.task,
        "order": order,
        "population": args_cli.population,
        "episodes_per_candidate": args_cli.episodes_per_candidate,
        "generations": args_cli.generations,
        "elite_count": elite_count,
        "base_checkpoint": str(base_checkpoint),
        "base_gain": base_gain.tolist(),
        "best_gain": best_gain.tolist(),
        "history": history,
        "direct_validation": direct_validation,
        "behavior_cloning": clone_metrics,
        "checkpoint": str(checkpoint),
    }
    if args_cli.output is None:
        output = (
            REPO_ROOT
            / "artifacts"
            / "inverted_pendulum"
            / "evaluation"
            / f"isaac_cem_lqr_order_{order}.json"
        )
    else:
        output = Path(args_cli.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] checkpoint = {checkpoint}")
    print(f"[INFO] wrote {output}")
    gym_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
