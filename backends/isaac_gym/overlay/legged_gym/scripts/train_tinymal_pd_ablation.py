"""PD-gain / action-scale ablation training for the flat TinyMal task.

Identical to the baseline (URDF, seed, reward, PPO hyperparameters) except for
control.stiffness / damping / action_scale, overridden via env vars before
make_env. These fields are not exposed on the CLI by update_cfg_from_args, so
they must be written on the cfg object first -- same pattern as
train_tinymal_stairs.py.
"""

import os

import isaacgym  # noqa: F401  # Must precede torch.
import torch

from legged_gym.envs import *  # noqa: F401,F403  # Registers tasks.
from legged_gym.utils import get_args, task_registry


def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name="tinymal")
    if args.seed is not None:
        env_cfg.seed = args.seed

    label = os.environ.get("TINYMAL_VARIANT", "baseline")
    kp = os.environ.get("TINYMAL_KP")
    kd = os.environ.get("TINYMAL_KD")
    ascale = os.environ.get("TINYMAL_ACTION_SCALE")
    if kp is not None:
        env_cfg.control.stiffness = {"joint": float(kp)}
    if kd is not None:
        env_cfg.control.damping = {"joint": float(kd)}
    if ascale is not None:
        env_cfg.control.action_scale = float(ascale)

    train_cfg.runner.experiment_name = "tinymal_pd_ablation"
    train_cfg.runner.run_name = label

    print(
        f"PD ablation variant={label}: "
        f"Kp={env_cfg.control.stiffness} Kd={env_cfg.control.damping} "
        f"action_scale={env_cfg.control.action_scale}"
    )

    env, _ = task_registry.make_env(name="tinymal", args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name="tinymal", args=args, train_cfg=train_cfg
    )
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    train(get_args())
