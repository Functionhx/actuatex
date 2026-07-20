#!/usr/bin/env python
"""Train the TinyMal Isaac Lab task with RSL-RL.

Run via Isaac Sim's Python launcher:
    "$ISAAC_SIM_PYTHON" train_tinymal.py \
        --task Isaac-Velocity-Flat-TinyMal-v0 --num_envs 4096 \
        --max_iterations 1500 --seed 1
"""

import argparse
import os
import sys
from pathlib import Path

# Make the sibling tinymal_lab package importable.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-TinyMal-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--save_interval", type=int, default=None)
parser.add_argument("--learning_rate", type=float, default=None)
parser.add_argument("--entropy_coef", type=float, default=None)
parser.add_argument("--schedule", choices=("fixed", "adaptive"), default=None)
parser.add_argument("--init_noise_std", type=float, default=None)
parser.add_argument(
    "--initial_angle_scale",
    type=float,
    default=None,
    help="scale the reset-angle range for inverted-pendulum curriculum stages",
)
parser.add_argument(
    "--angle_curriculum",
    type=str,
    default=None,
    help=(
        "single-launch inverted-pendulum curriculum as "
        "SCALE:ITERATIONS pairs, for example 0.35:40,0.65:60,1.0:120"
    ),
)
parser.add_argument(
    "--critic_warmup_iterations",
    type=int,
    default=0,
    help="freeze a warm-started actor while fitting the critic",
)
parser.add_argument(
    "--output_checkpoint",
    type=str,
    default=None,
    help="also save the final RSL-RL checkpoint at this canonical path",
)
parser.add_argument(
    "--gpu_found_lost_pairs_capacity",
    type=int,
    default=None,
    help=(
        "override the PhysX GPU broad-phase found/lost pair buffer; robust/stair "
        "tasks otherwise size it automatically from --num_envs"
    ),
)
parser.add_argument(
    "--init_checkpoint",
    type=str,
    default=None,
    help="warm-start an actor from an old Gym or new Lab checkpoint",
)
parser.add_argument(
    "--init_critic",
    action="store_true",
    help="also warm-start a shape-compatible privileged critic from --init_checkpoint",
)
parser.add_argument("--resume", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
if (
    args_cli.gpu_found_lost_pairs_capacity is not None
    and args_cli.gpu_found_lost_pairs_capacity <= 0
):
    parser.error("--gpu_found_lost_pairs_capacity must be positive")
if args_cli.init_critic and args_cli.init_checkpoint is None:
    parser.error("--init_critic requires --init_checkpoint")
if args_cli.critic_warmup_iterations < 0:
    parser.error("--critic_warmup_iterations cannot be negative")
if args_cli.critic_warmup_iterations and args_cli.init_checkpoint is None:
    parser.error("--critic_warmup_iterations requires --init_checkpoint")
if args_cli.initial_angle_scale is not None and args_cli.initial_angle_scale <= 0.0:
    parser.error("--initial_angle_scale must be positive")
if args_cli.initial_angle_scale is not None and args_cli.angle_curriculum is not None:
    parser.error("use either --initial_angle_scale or --angle_curriculum, not both")
if args_cli.entropy_coef is not None and args_cli.entropy_coef < 0.0:
    parser.error("--entropy_coef must be non-negative")


def _parse_angle_curriculum(spec: str | None) -> list[tuple[float, int]]:
    if spec is None:
        return []
    stages: list[tuple[float, int]] = []
    try:
        for item in spec.split(","):
            scale_text, iterations_text = item.split(":", maxsplit=1)
            scale = float(scale_text)
            iterations = int(iterations_text)
            if scale <= 0.0 or iterations <= 0:
                raise ValueError
            stages.append((scale, iterations))
    except ValueError:
        parser.error("--angle_curriculum must contain positive SCALE:ITERATIONS pairs")
    return stages


angle_curriculum = _parse_angle_curriculum(args_cli.angle_curriculum)
if angle_curriculum and args_cli.max_iterations is not None:
    curriculum_iterations = sum(iterations for _, iterations in angle_curriculum)
    if args_cli.max_iterations != curriculum_iterations:
        parser.error(
            "--max_iterations must equal the sum of --angle_curriculum stages "
            f"({curriculum_iterations})"
        )
if angle_curriculum and args_cli.critic_warmup_iterations > angle_curriculum[0][1]:
    parser.error("--critic_warmup_iterations cannot exceed the first curriculum stage")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import logging  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401  (registers built-in tasks)
import tinymal_lab  # noqa: E402,F401     (registers Isaac-Velocity-Flat-TinyMal-v0)
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import importlib.metadata as _metadata  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tinymal_train")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

LOG_ROOT = os.path.join(
    os.environ.get("ACTUATEX_ARTIFACTS", str(REPO_ROOT / "artifacts")),
    "isaac_lab",
    "logs",
    "rsl_rl",
)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    base_initial_angle_range = getattr(env_cfg, "initial_angle_range", None)
    if args_cli.initial_angle_scale is not None:
        if not hasattr(env_cfg, "initial_angle_range"):
            raise ValueError(
                "--initial_angle_scale is only valid for an environment with "
                "initial_angle_range"
            )
        env_cfg.initial_angle_range *= args_cli.initial_angle_scale
        print(
            "[INFO] inverted-pendulum reset angle range = "
            f"{env_cfg.initial_angle_range:.6f} rad "
            f"(scale={args_cli.initial_angle_scale:g})"
        )
    elif angle_curriculum:
        if base_initial_angle_range is None:
            raise ValueError(
                "--angle_curriculum is only valid for an environment with "
                "initial_angle_range"
            )
        first_scale = angle_curriculum[0][0]
        env_cfg.initial_angle_range = base_initial_angle_range * first_scale
        print(
            "[INFO] inverted-pendulum curriculum = "
            + ", ".join(
                f"{scale:g}x/{iterations} iters"
                for scale, iterations in angle_curriculum
            )
        )
    pair_capacity = args_cli.gpu_found_lost_pairs_capacity
    if pair_capacity is None and (
        "Robust" in args_cli.task or "Stairs" in args_cli.task
    ):
        # Sim 6.0.1 reported 37,923,400 pairs for 4096 robust flat/stair
        # environments and 151,431,753 for 8192: this SAP broad-phase load
        # scales quadratically.  Three pairs per env squared leaves ~32.7%
        # headroom over both observations; round to a power of two for PhysX.
        required_pairs = max(2**21, 3 * env_cfg.scene.num_envs**2)
        pair_capacity = 1 << (required_pairs - 1).bit_length()
    if pair_capacity is not None:
        env_cfg.sim.physics.gpu_found_lost_pairs_capacity = pair_capacity
        print(
            "[INFO] PhysX gpu_found_lost_pairs_capacity = "
            f"{pair_capacity:,} for {env_cfg.scene.num_envs} environments"
        )
    if angle_curriculum:
        agent_cfg.max_iterations = sum(iterations for _, iterations in angle_curriculum)
    elif args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval
    if args_cli.learning_rate is not None:
        agent_cfg.algorithm.learning_rate = args_cli.learning_rate
    if args_cli.entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = args_cli.entropy_coef
    if args_cli.schedule is not None:
        agent_cfg.algorithm.schedule = args_cli.schedule
    if args_cli.init_noise_std is not None:
        agent_cfg.actor.distribution_cfg.init_std = args_cli.init_noise_std
    agent_cfg.seed = args_cli.seed
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Migrate the Isaac Lab actor/critic cfg into the rsl-rl-lib 5.x model schema
    # (adds actor/critic class_name + RslRlMLPModelCfg that OnPolicyRunner expects).
    _rsl_ver = _metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _rsl_ver)

    log_root_path = os.path.abspath(os.path.join(LOG_ROOT, agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir
    print(f"[INFO] log_dir = {log_dir}")

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )

    if args_cli.init_checkpoint is not None:
        init_path = os.path.abspath(args_cli.init_checkpoint)
        payload = torch.load(init_path, weights_only=False, map_location="cpu")
        if "model_state_dict" in payload:
            actor_state = {
                "mlp." + key[len("actor.") :]: value
                for key, value in payload["model_state_dict"].items()
                if key.startswith("actor.")
            }
        elif "actor_state_dict" in payload:
            actor_state = {
                key: value
                for key, value in payload["actor_state_dict"].items()
                if key.startswith("mlp.")
            }
        else:
            raise KeyError(
                f"{init_path} has neither model_state_dict nor actor_state_dict"
            )
        incompatible = runner.alg.actor.load_state_dict(actor_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("distribution.")
        ]
        if unexpected or missing:
            raise RuntimeError(
                f"actor warm-start mismatch: missing={missing}, unexpected={unexpected}"
            )
        if args_cli.init_critic:
            critic_state = payload.get("critic_state_dict")
            if critic_state is None:
                raise KeyError(f"{init_path} has no critic_state_dict")
            runner.alg.critic.load_state_dict(critic_state, strict=True)
        target_std = (
            args_cli.init_noise_std
            if args_cli.init_noise_std is not None
            else float(agent_cfg.actor.distribution_cfg.init_std)
        )
        distribution = runner.alg.actor.distribution
        if hasattr(distribution, "std_param"):
            distribution.std_param.data.fill_(target_std)
        elif hasattr(distribution, "log_std_param"):
            distribution.log_std_param.data.fill_(math.log(target_std))
        print(
            f"[INFO] actor warm-started from {init_path}; exploration std={target_std}"
        )
        if args_cli.init_critic:
            print(
                "[INFO] shape-compatible critic warm-started from the same checkpoint"
            )

    if args_cli.resume:
        from isaaclab_tasks.utils import get_checkpoint_path

        resume_path = get_checkpoint_path(log_root_path)
        print(f"[INFO] resume from {resume_path}")
        runner.load(resume_path)

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    t0 = time.time()
    warmup = min(args_cli.critic_warmup_iterations, agent_cfg.max_iterations)
    next_iteration = 0

    def learn_block(iterations: int, *, randomize_episode_length: bool) -> None:
        nonlocal next_iteration
        if iterations <= 0:
            return
        runner.current_learning_iteration = next_iteration
        runner.learn(
            num_learning_iterations=iterations,
            init_at_random_ep_len=randomize_episode_length,
        )
        next_iteration += iterations

    if warmup:
        for parameter in runner.alg.actor.parameters():
            parameter.requires_grad_(False)
        print(f"[INFO] critic-only warmup for {warmup} iterations")
        learn_block(warmup, randomize_episode_length=True)
        for parameter in runner.alg.actor.parameters():
            parameter.requires_grad_(True)

    if angle_curriculum:
        raw_env = gym_env.unwrapped
        for stage_index, (scale, stage_iterations) in enumerate(angle_curriculum):
            stage_remaining = stage_iterations
            if stage_index == 0:
                stage_remaining -= warmup
            else:
                raw_env.cfg.initial_angle_range = base_initial_angle_range * scale
                all_env_ids = torch.arange(
                    raw_env.num_envs,
                    dtype=torch.long,
                    device=raw_env.device,
                )
                raw_env._reset_idx(all_env_ids)
            if stage_remaining:
                print(
                    f"[INFO] curriculum stage {stage_index + 1}/"
                    f"{len(angle_curriculum)}: scale={scale:g}, "
                    f"iterations={stage_remaining}"
                )
                learn_block(
                    stage_remaining,
                    randomize_episode_length=next_iteration == 0,
                )
    else:
        learn_block(
            agent_cfg.max_iterations - warmup,
            randomize_episode_length=next_iteration == 0,
        )
    elapsed = time.time() - t0
    if args_cli.output_checkpoint is not None:
        output_path = os.path.abspath(args_cli.output_checkpoint)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        checkpoint_infos = {
            "backend": (
                "Isaac Sim 6.0.1 GA / "
                "Isaac Lab 3.0.0-beta2.patch1 / PhysX 5"
            ),
            "task": args_cli.task,
            "seed": args_cli.seed,
            "num_envs": env_cfg.scene.num_envs,
            "iterations": agent_cfg.max_iterations,
            "wall_time_s": elapsed,
        }
        if "Sentinel" in args_cli.task:
            from tasks.robomaster.contract import contract_sha256

            checkpoint_infos["contract_sha256"] = contract_sha256()
        runner.save(output_path, infos=checkpoint_infos)
        print(f"[INFO] canonical checkpoint = {output_path}")
    print(f"Training time: {round(elapsed, 2)} s")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
