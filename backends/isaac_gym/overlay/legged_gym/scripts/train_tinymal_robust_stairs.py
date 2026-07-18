"""Stage-wise stair transfer while retaining the sim-to-real distribution."""

import os

import isaacgym  # noqa: F401  # Must precede torch.
import torch

from legged_gym.envs import *  # noqa: F401,F403  # Registers tasks.
from legged_gym.utils import get_args, task_registry


def train(args):
    if args.task != "tinymal_robust_stairs":
        raise ValueError(
            "train_tinymal_robust_stairs.py requires --task=tinymal_robust_stairs"
        )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.seed is not None:
        env_cfg.seed = args.seed

    step_height = float(os.environ.get("TINYMAL_TRAIN_STEP_HEIGHT", "0.010"))
    speed = float(os.environ.get("TINYMAL_TRAIN_SPEED", "0.30"))
    env_cfg.stairs.curriculum = False
    env_cfg.stairs.step_height = step_height
    env_cfg.commands.ranges.lin_vel_x = [speed, speed]
    env_cfg.domain_rand.randomize_push_force = (
        os.environ.get("TINYMAL_STAIR_PUSHES", "0") == "1"
    )
    learning_rate = float(os.environ.get("TINYMAL_LEARNING_RATE", "1e-4"))
    train_cfg.algorithm.learning_rate = learning_rate
    train_cfg.algorithm.schedule = os.environ.get(
        "TINYMAL_LEARNING_SCHEDULE", "fixed"
    )
    train_cfg.runner.save_interval = int(
        os.environ.get("TINYMAL_SAVE_INTERVAL", str(train_cfg.runner.save_interval))
    )

    pretrained = os.environ.get("TINYMAL_PRETRAINED")
    if not pretrained:
        raise ValueError("TINYMAL_PRETRAINED must point to the preceding policy")
    pretrained = os.path.abspath(pretrained)
    if not os.path.isfile(pretrained):
        raise FileNotFoundError(pretrained)

    print(
        f"Robust stair stage: height={step_height * 1000:.1f} mm "
        f"speed={speed:.2f} m/s pushes={env_cfg.domain_rand.randomize_push_force} "
        f"lr={learning_rate:.2e} save_interval={train_cfg.runner.save_interval}"
    )
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    checkpoint = torch.load(pretrained, map_location=args.rl_device)
    runner.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])

    # Optional quadratic trust region around a known general-purpose actor.
    # Adding the derivative through parameter hooks keeps the bundled PPO
    # implementation untouched while discouraging stair-only catastrophic
    # forgetting.  The coefficient is the gradient multiplier for
    # 0.5 * ||actor - anchor||^2.
    anchor_coefficient = float(os.environ.get("TINYMAL_ANCHOR_COEF", "0"))
    anchor_path = os.environ.get("TINYMAL_ANCHOR_CHECKPOINT", pretrained)
    if anchor_coefficient > 0.0:
        anchor_path = os.path.abspath(anchor_path)
        if not os.path.isfile(anchor_path):
            raise FileNotFoundError(anchor_path)
        anchor_state = torch.load(anchor_path, map_location=args.rl_device)[
            "model_state_dict"
        ]
        anchored_parameters = 0
        for name, parameter in runner.alg.actor_critic.named_parameters():
            if not name.startswith("actor."):
                continue
            anchor = anchor_state[name].detach().clone().to(parameter.device)
            if anchor.shape != parameter.shape:
                raise ValueError(f"anchor shape mismatch for {name}")

            def add_anchor_gradient(gradient, current=parameter, target=anchor):
                return gradient + anchor_coefficient * (current.detach() - target)

            parameter.register_hook(add_anchor_gradient)
            anchored_parameters += parameter.numel()
        print(
            f"Actor trust region: coefficient={anchor_coefficient:.3g} "
            f"anchor={anchor_path} parameters={anchored_parameters}"
        )

    transfer_std = float(os.environ.get("TINYMAL_TRANSFER_STD", "0.12"))
    with torch.no_grad():
        runner.alg.actor_critic.std.fill_(transfer_std)
    if os.environ.get("TINYMAL_FREEZE_STD", "0") == "1":
        runner.alg.actor_critic.std.requires_grad_(False)
    print(f"Transferred actor from {pretrained}; exploration std={transfer_std}")

    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    train(get_args())
