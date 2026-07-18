"""Transfer a flat-ground TinyMal policy to a parallel stair curriculum."""

import os

import isaacgym  # noqa: F401  # Must precede torch.
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403  # Registers tasks.
from legged_gym.utils import get_args, task_registry


def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name="tinymal_stairs")
    # Upstream's argument helper applies --seed only to the PPO config, while
    # make_env() seeds from env_cfg.  Mirror it here so reported run seeds also
    # control physics randomization and initial states.
    if args.seed is not None:
        env_cfg.seed = args.seed
    fixed_height = os.environ.get("TINYMAL_TRAIN_STEP_HEIGHT")
    if fixed_height is not None:
        env_cfg.stairs.curriculum = False
        env_cfg.stairs.step_height = float(fixed_height)
        print(f"Training fixed stair height: {float(fixed_height) * 1000:.1f} mm")
    fixed_speed = os.environ.get("TINYMAL_TRAIN_SPEED")
    if fixed_speed is not None:
        speed = float(fixed_speed)
        env_cfg.commands.ranges.lin_vel_x = [speed, speed]
        print(f"Training fixed forward command: {speed:.3f} m/s")
    reward_profile = os.environ.get("TINYMAL_REWARD_PROFILE", "dense")
    if reward_profile == "legacy":
        # The dense profile is useful as a diagnostic, but can let stochastic
        # collection noise earn progress while the deployable actor mean stops.
        # This profile retains the longer horizon while restoring the reward
        # that produced the accepted 15 mm deterministic policy.
        env_cfg.rewards.scales.world_forward_progress = 0.0
        env_cfg.rewards.scales.tracking_lin_vel = 1.0
        env_cfg.rewards.scales.ang_vel_xy = -0.05
    elif reward_profile != "dense":
        raise ValueError(f"Unknown TINYMAL_REWARD_PROFILE={reward_profile!r}")
    print(f"Reward profile: {reward_profile}")

    learning_rate = os.environ.get("TINYMAL_LEARNING_RATE")
    if learning_rate is not None:
        train_cfg.algorithm.learning_rate = float(learning_rate)
        print(f"Initial PPO learning rate: {float(learning_rate):.2e}")
    schedule = os.environ.get("TINYMAL_LEARNING_SCHEDULE")
    if schedule is not None:
        if schedule not in {"fixed", "adaptive"}:
            raise ValueError(f"Unknown TINYMAL_LEARNING_SCHEDULE={schedule!r}")
        train_cfg.algorithm.schedule = schedule
        print(f"PPO learning schedule: {schedule}")
    env, _ = task_registry.make_env(
        name="tinymal_stairs", args=args, env_cfg=env_cfg
    )
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name="tinymal_stairs", args=args, train_cfg=train_cfg
    )

    default_checkpoint = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        "tinymal_baseline",
        "Jul17_23-52-15_std0p3_seed1",
        "model_1500.pt",
    )
    checkpoint_path = os.environ.get("TINYMAL_PRETRAINED", default_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=args.rl_device)
    runner.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])

    # The flat policy ended with std≈0.07, too small to discover higher swing
    # trajectories. Keep its mean action but reopen bounded exploration.
    transfer_std = float(os.environ.get("TINYMAL_TRANSFER_STD", "0.20"))
    with torch.no_grad():
        runner.alg.actor_critic.std.fill_(transfer_std)
    freeze_std = os.environ.get("TINYMAL_FREEZE_STD", "0") == "1"
    if freeze_std:
        # For final-stage distillation, keep collection noise close to the
        # deterministic deployment policy.  Otherwise PPO can assign the gait
        # to random actions while its actor mean collapses to standing still.
        runner.alg.actor_critic.std.requires_grad_(False)

    print(f"Transferred policy mean from: {checkpoint_path}")
    print(f"Reset action exploration std to: {transfer_std}")
    print(f"Freeze action exploration std: {freeze_std}")
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    train(get_args())
