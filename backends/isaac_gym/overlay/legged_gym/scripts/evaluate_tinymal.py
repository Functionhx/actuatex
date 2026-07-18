"""Headless deterministic tracking evaluation for a trained TinyMal policy."""

import csv
import json
import os
from collections import defaultdict

import isaacgym  # noqa: F401  # Must precede torch.
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403  # Registers tasks.
from legged_gym.utils import export_policy_as_jit, get_args, task_registry


SEGMENTS = (
    ("stand", 2.0, (0.0, 0.0, 0.0)),
    ("forward_0p3", 3.0, (0.3, 0.0, 0.0)),
    ("forward_0p6", 3.0, (0.6, 0.0, 0.0)),
    ("backward_0p3", 3.0, (-0.3, 0.0, 0.0)),
    ("lateral_0p2", 3.0, (0.0, 0.2, 0.0)),
    ("yaw_0p5", 3.0, (0.0, 0.0, 0.5)),
)


def evaluate(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name="tinymal")
    env_cfg.env.num_envs = args.num_envs or 64
    env_cfg.env.episode_length_s = 30
    env_cfg.commands.heading_command = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False

    env, _ = task_registry.make_env(name="tinymal", args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env, name="tinymal", args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)

    # Keep the historical baseline artifacts intact when evaluating a new
    # checkpoint.  Callers can route each policy to its own evidence directory.
    output_dir = os.environ.get(
        "TINYMAL_EVAL_OUT",
        os.path.join(LEGGED_GYM_ROOT_DIR, "evaluation", "tinymal"),
    )
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "tracking.csv")
    summary_path = os.path.join(output_dir, "summary.json")
    export_dir = os.path.join(output_dir, "exported")
    export_policy_as_jit(runner.alg.actor_critic, export_dir)

    rows = []
    segment_samples = defaultdict(list)
    global_step = 0
    dt = env.dt

    with torch.inference_mode():
        for segment, duration, command in SEGMENTS:
            steps = int(round(duration / dt))
            settle_steps = int(round(min(1.0, duration / 3.0) / dt))
            for local_step in range(steps):
                env.commands[:, 0] = command[0]
                env.commands[:, 1] = command[1]
                env.commands[:, 2] = command[2]
                env.compute_observations()
                actions = policy(env.get_observations())
                _, _, _, dones, _ = env.step(actions)

                sample = {
                    "time_s": global_step * dt,
                    "segment": segment,
                    "cmd_vx": command[0],
                    "cmd_vy": command[1],
                    "cmd_yaw": command[2],
                    "vx_mean": env.base_lin_vel[:, 0].mean().item(),
                    "vx_std": env.base_lin_vel[:, 0].std().item(),
                    "vy_mean": env.base_lin_vel[:, 1].mean().item(),
                    "vy_std": env.base_lin_vel[:, 1].std().item(),
                    "yaw_mean": env.base_ang_vel[:, 2].mean().item(),
                    "yaw_std": env.base_ang_vel[:, 2].std().item(),
                    "base_z_mean": env.base_pos[:, 2].mean().item(),
                    "base_z_std": env.base_pos[:, 2].std().item(),
                    "abs_roll_mean": env.rpy[:, 0].abs().mean().item(),
                    "abs_pitch_mean": env.rpy[:, 1].abs().mean().item(),
                    "action_rms": actions.square().mean().sqrt().item(),
                    "torque_rms": env.torques.square().mean().sqrt().item(),
                    "reset_fraction": dones.float().mean().item(),
                }
                rows.append(sample)
                if local_step >= settle_steps:
                    segment_samples[segment].append(sample)
                global_step += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for segment, samples in segment_samples.items():
        cmd_vx = samples[0]["cmd_vx"]
        cmd_vy = samples[0]["cmd_vy"]
        cmd_yaw = samples[0]["cmd_yaw"]
        summary[segment] = {
            "vx_rmse": float(
                np.sqrt(np.mean([(s["vx_mean"] - cmd_vx) ** 2 for s in samples]))
            ),
            "vy_rmse": float(
                np.sqrt(np.mean([(s["vy_mean"] - cmd_vy) ** 2 for s in samples]))
            ),
            "yaw_rmse": float(
                np.sqrt(np.mean([(s["yaw_mean"] - cmd_yaw) ** 2 for s in samples]))
            ),
            "base_height_mean": float(np.mean([s["base_z_mean"] for s in samples])),
            "base_height_std": float(np.std([s["base_z_mean"] for s in samples])),
            "abs_roll_mean": float(np.mean([s["abs_roll_mean"] for s in samples])),
            "abs_pitch_mean": float(np.mean([s["abs_pitch_mean"] for s in samples])),
            "action_rms": float(np.mean([s["action_rms"] for s in samples])),
            "reset_fraction_mean": float(
                np.mean([s["reset_fraction"] for s in samples])
            ),
            "resets_total": int(
                round(sum(s["reset_fraction"] * env.num_envs for s in samples))
            ),
            "torque_rms": float(np.mean([s["torque_rms"] for s in samples])),
        }

    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"tracking_csv={csv_path}")
    print(f"summary_json={summary_path}")
    print(f"jit_policy={os.path.join(export_dir, 'policy_1.pt')}")
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    evaluate(get_args())
