"""Deterministic TinyMal standing diagnostic used before PPO training."""

import isaacgym  # noqa: F401  # Isaac Gym must be imported before torch.
import torch

from legged_gym.envs import *  # noqa: F401,F403  # Registers tasks.
from legged_gym.utils import get_args, task_registry


def diagnose(args):
    env_cfg, _ = task_registry.get_cfgs("tinymal")
    env_cfg.env.num_envs = args.num_envs or 64
    env_cfg.env.episode_length_s = 10
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    env, _ = task_registry.make_env(name="tinymal", args=args, env_cfg=env_cfg)
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    reset_count = 0

    for step in range(500):
        _, _, _, dones, _ = env.step(actions)
        # reset_buf starts high by design, so the first transition is warm-up
        # rather than a physical failure.
        if step > 0:
            reset_count += int(dones.sum().item())
        if step in (0, 49, 99, 249, 499):
            print(
                f"step={step + 1:3d} "
                f"base_z={env.base_pos[:, 2].mean().item():.4f} "
                f"abs_roll={env.rpy[:, 0].abs().mean().item():.4f} "
                f"abs_pitch={env.rpy[:, 1].abs().mean().item():.4f} "
                f"cumulative_resets={reset_count}"
            )

    print(
        f"reset_rate_per_robot_second="
        f"{reset_count / (env.num_envs * 500 * env.dt):.6f}"
    )
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    diagnose(get_args())
