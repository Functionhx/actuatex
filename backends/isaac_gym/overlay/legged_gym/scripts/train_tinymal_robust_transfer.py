"""Short robust-flat transfer stages for a stair-specialized TinyMal policy.

This launcher intentionally loads only the policy/value parameters from an
explicit checkpoint and starts a fresh PPO optimizer.  It is used between
stair curriculum stages to recover multi-direction velocity tracking and push
recovery without confusing RSL-RL's iteration counters.
"""

import os

import isaacgym  # noqa: F401  # Must precede torch.
import torch

from legged_gym.envs import *  # noqa: F401,F403  # Registers tasks.
from legged_gym.utils import get_args, task_registry


def train(args):
    if args.task != "tinymal_robust":
        raise ValueError(
            "train_tinymal_robust_transfer.py requires --task=tinymal_robust"
        )

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.seed is not None:
        env_cfg.seed = args.seed

    learning_rate = float(os.environ.get("TINYMAL_LEARNING_RATE", "5e-5"))
    train_cfg.algorithm.learning_rate = learning_rate
    train_cfg.algorithm.schedule = os.environ.get(
        "TINYMAL_LEARNING_SCHEDULE", "fixed"
    )
    train_cfg.runner.experiment_name = os.environ.get(
        "TINYMAL_TRANSFER_EXPERIMENT", "tinymal_sim2real_recovery"
    )

    pretrained = os.environ.get("TINYMAL_PRETRAINED")
    if not pretrained:
        raise ValueError("TINYMAL_PRETRAINED must point to the preceding policy")
    pretrained = os.path.abspath(pretrained)
    if not os.path.isfile(pretrained):
        raise FileNotFoundError(pretrained)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    checkpoint = torch.load(pretrained, map_location=args.rl_device)
    runner.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])

    transfer_std = float(os.environ.get("TINYMAL_TRANSFER_STD", "0.08"))
    with torch.no_grad():
        runner.alg.actor_critic.std.fill_(transfer_std)
    if os.environ.get("TINYMAL_FREEZE_STD", "0") == "1":
        runner.alg.actor_critic.std.requires_grad_(False)

    print(
        f"Robust-flat recovery from {pretrained}; lr={learning_rate:.2e} "
        f"exploration_std={transfer_std:.3f}"
    )
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    train(get_args())
