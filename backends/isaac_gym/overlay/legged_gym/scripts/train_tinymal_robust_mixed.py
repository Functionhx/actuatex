"""Train one actor on robust flat commands and explicit stairs together."""

import copy
import os

import isaacgym  # noqa: F401  # Must precede torch.
import torch

from legged_gym.envs import *  # noqa: F401,F403  # Registers tasks.
from legged_gym.utils import get_args, task_registry


def train(args):
    if args.task != "tinymal_robust_mixed":
        raise ValueError(
            "train_tinymal_robust_mixed.py requires --task=tinymal_robust_mixed"
        )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.seed is not None:
        env_cfg.seed = args.seed

    env_cfg.stairs.flat_env_fraction = float(
        os.environ.get("TINYMAL_FLAT_ENV_FRACTION", "0.5")
    )
    env_cfg.stairs.step_height = float(
        os.environ.get("TINYMAL_TRAIN_STEP_HEIGHT", "0.020")
    )
    env_cfg.stairs.command_speed = float(
        os.environ.get("TINYMAL_TRAIN_SPEED", "0.30")
    )
    env_cfg.stairs.policy_command_speed = float(
        os.environ.get(
            "TINYMAL_STAIR_POLICY_SPEED", str(env_cfg.stairs.command_speed)
        )
    )
    env_cfg.domain_rand.randomize_push_force = (
        os.environ.get("TINYMAL_MIXED_PUSHES", "1") == "1"
    )
    learning_rate = float(os.environ.get("TINYMAL_LEARNING_RATE", "5e-5"))
    train_cfg.algorithm.learning_rate = learning_rate
    train_cfg.algorithm.schedule = "fixed"
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
        f"Mixed rehearsal: flat={env_cfg.stairs.flat_env_fraction:.2f} "
        f"stairs={env_cfg.stairs.step_height * 1000:.1f} mm "
        f"task_speed={env_cfg.stairs.command_speed:.2f} m/s "
        f"policy_marker={env_cfg.stairs.policy_command_speed:.2f} "
        f"flat_pushes={env_cfg.domain_rand.randomize_push_force} "
        f"lr={learning_rate:.2e} save_interval={train_cfg.runner.save_interval}"
    )
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    checkpoint = torch.load(pretrained, map_location=args.rl_device)
    runner.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
    transfer_std = float(os.environ.get("TINYMAL_TRANSFER_STD", "0.08"))
    with torch.no_grad():
        runner.alg.actor_critic.std.fill_(transfer_std)
    print(f"Transferred actor from {pretrained}; exploration std={transfer_std}")

    # A stair mode outside the normal flat-command range should not require
    # relearning the shared proprioceptive feature extractor.  Optionally
    # freeze the first three actor layers and optimize only the final linear
    # readout.  This keeps the established flat gait much closer to the source
    # policy while still allowing command-conditioned action remapping.
    if os.environ.get("TINYMAL_FREEZE_ACTOR_FEATURES", "0") == "1":
        for name, parameter in runner.alg.actor_critic.actor.named_parameters():
            parameter.requires_grad_(name.startswith("6."))
        trainable_actor_parameters = sum(
            parameter.numel()
            for parameter in runner.alg.actor_critic.actor.parameters()
            if parameter.requires_grad
        )
        print(
            "Frozen actor feature extractor; trainable final-layer parameters="
            f"{trainable_actor_parameters}"
        )

    teacher_coefficient = float(os.environ.get("TINYMAL_TEACHER_COEF", "0"))
    flat_observation_mask = None
    if teacher_coefficient > 0.0:
        teacher_path = os.environ.get("TINYMAL_TEACHER_CHECKPOINT")
        if not teacher_path:
            raise ValueError(
                "TINYMAL_TEACHER_CHECKPOINT is required when "
                "TINYMAL_TEACHER_COEF > 0"
            )
        teacher_path = os.path.abspath(teacher_path)
        teacher_state = torch.load(teacher_path, map_location=args.rl_device)[
            "model_state_dict"
        ]
        teacher_actor = copy.deepcopy(runner.alg.actor_critic.actor)
        teacher_actor.load_state_dict(
            {
                name[len("actor.") :]: value
                for name, value in teacher_state.items()
                if name.startswith("actor.")
            },
            strict=True,
        )
        teacher_actor.eval()
        for parameter in teacher_actor.parameters():
            parameter.requires_grad_(False)

        stair_command_vx = (
            env_cfg.stairs.policy_command_speed
            * env_cfg.normalization.obs_scales.lin_vel
        )

        def flat_observation_mask(observations):
            stair_commands = (
                torch.isclose(
                    observations[:, 9],
                    observations.new_tensor(stair_command_vx),
                    rtol=0.0,
                    atol=1.0e-6,
                )
                & (observations[:, 10].abs() <= 1.0e-6)
            )
            return ~stair_commands

        runner.alg.reference_actor = teacher_actor
        runner.alg.reference_loss_coef = teacher_coefficient
        runner.alg.reference_mask_fn = flat_observation_mask
        print(
            f"Flat-policy teacher: coefficient={teacher_coefficient:.3g} "
            f"checkpoint={teacher_path}"
        )

    stair_teacher_coefficient = float(
        os.environ.get("TINYMAL_STAIR_TEACHER_COEF", "0")
    )
    if stair_teacher_coefficient > 0.0:
        stair_teacher_path = os.environ.get("TINYMAL_STAIR_TEACHER_CHECKPOINT")
        if not stair_teacher_path:
            raise ValueError(
                "TINYMAL_STAIR_TEACHER_CHECKPOINT is required when "
                "TINYMAL_STAIR_TEACHER_COEF > 0"
            )
        if flat_observation_mask is None:
            raise ValueError(
                "the stair teacher requires the flat teacher so their masks "
                "form an explicit partition"
            )
        stair_teacher_path = os.path.abspath(stair_teacher_path)
        stair_teacher_state = torch.load(
            stair_teacher_path, map_location=args.rl_device
        )["model_state_dict"]
        stair_teacher_actor = copy.deepcopy(runner.alg.actor_critic.actor)
        stair_teacher_actor.load_state_dict(
            {
                name[len("actor.") :]: value
                for name, value in stair_teacher_state.items()
                if name.startswith("actor.")
            },
            strict=True,
        )
        stair_teacher_actor.eval()
        for parameter in stair_teacher_actor.parameters():
            parameter.requires_grad_(False)

        stair_teacher_command_vx = (
            env_cfg.stairs.command_speed
            * env_cfg.normalization.obs_scales.lin_vel
        )

        def stair_reference_actor(observations):
            teacher_observations = observations.clone()
            teacher_observations[:, 9] = stair_teacher_command_vx
            return stair_teacher_actor(teacher_observations)

        runner.alg.secondary_reference_actor = stair_reference_actor
        runner.alg.secondary_reference_loss_coef = stair_teacher_coefficient
        runner.alg.secondary_reference_mask_fn = (
            lambda observations: ~flat_observation_mask(observations)
        )
        print(
            f"Stair-policy teacher: coefficient={stair_teacher_coefficient:.3g} "
            f"checkpoint={stair_teacher_path}"
        )

    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )
    if teacher_coefficient > 0.0:
        print(f"Final flat teacher loss={runner.alg.last_reference_loss:.6g}")
    if stair_teacher_coefficient > 0.0:
        print(
            "Final stair teacher loss="
            f"{runner.alg.last_secondary_reference_loss:.6g}"
        )
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    train(get_args())
