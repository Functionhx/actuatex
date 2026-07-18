"""Evaluate one PD / action-scale ablation variant with its matching PD gains.

Same fixed-command segment protocol as evaluate_tinymal.py, but sets
control.stiffness / damping / action_scale on the cfg BEFORE make_env (these are
baked into p_gains/d_gains at _init_buffers, legged_robot.py:466-483) and loads
the variant checkpoint from logs/tinymal_pd_ablation/<run>/model_<ckpt>.pt.

Env vars:
  TINYMAL_VARIANT       label (e.g. kp40_kd1_as0p25)
  TINYMAL_KP / TINYMAL_KD / TINYMAL_ACTION_SCALE   the variant's control gains
  TINYMAL_PD_LOAD_RUN   run dir name under logs/tinymal_pd_ablation/
  TINYMAL_PD_CHECKPOINT checkpoint number (default 1500)
"""

import csv
import json
import os
from collections import defaultdict

import isaacgym  # noqa: F401  # Must precede torch.
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403  # Registers tasks.
from legged_gym.utils import get_args, task_registry


SEGMENTS = (
    ("stand", 2.0, (0.0, 0.0, 0.0)),
    ("forward_0p3", 3.0, (0.3, 0.0, 0.0)),
    ("forward_0p6", 3.0, (0.6, 0.0, 0.0)),
    ("backward_0p3", 3.0, (-0.3, 0.0, 0.0)),
    ("lateral_0p2", 3.0, (0.0, 0.2, 0.0)),
    ("yaw_0p5", 3.0, (0.0, 0.0, 0.5)),
)


def evaluate(args):
    label = os.environ.get("TINYMAL_VARIANT", "baseline")
    env_cfg, train_cfg = task_registry.get_cfgs(name="tinymal")
    env_cfg.env.num_envs = args.num_envs or 64
    env_cfg.env.episode_length_s = 30
    env_cfg.commands.heading_command = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False

    # PD / action_scale override BEFORE make_env (baked in at _init_buffers).
    if os.environ.get("TINYMAL_KP"):
        env_cfg.control.stiffness = {"joint": float(os.environ["TINYMAL_KP"])}
    if os.environ.get("TINYMAL_KD"):
        env_cfg.control.damping = {"joint": float(os.environ["TINYMAL_KD"])}
    if os.environ.get("TINYMAL_ACTION_SCALE"):
        env_cfg.control.action_scale = float(os.environ["TINYMAL_ACTION_SCALE"])
    print(f"PD variant={label}: Kp={env_cfg.control.stiffness} "
          f"Kd={env_cfg.control.damping} action_scale={env_cfg.control.action_scale}")

    env, _ = task_registry.make_env(name="tinymal", args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    train_cfg.runner.experiment_name = "tinymal_pd_ablation"
    train_cfg.runner.load_run = os.environ.get("TINYMAL_PD_LOAD_RUN")
    train_cfg.runner.checkpoint = int(os.environ.get("TINYMAL_PD_CHECKPOINT", "1500"))
    runner, _ = task_registry.make_alg_runner(
        env=env, name="tinymal", args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)

    output_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "evaluation", "pd_ablation", label)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "tracking.csv")
    summary_path = os.path.join(output_dir, "summary.json")

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
                    "time_s": global_step * dt, "segment": segment,
                    "cmd_vx": command[0], "cmd_vy": command[1], "cmd_yaw": command[2],
                    "vx_mean": env.base_lin_vel[:, 0].mean().item(),
                    "vy_mean": env.base_lin_vel[:, 1].mean().item(),
                    "yaw_mean": env.base_ang_vel[:, 2].mean().item(),
                    "base_z_mean": env.base_pos[:, 2].mean().item(),
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
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"variant": label,
               "Kp": env_cfg.control.stiffness, "Kd": env_cfg.control.damping,
               "action_scale": env_cfg.control.action_scale}
    for segment, samples in segment_samples.items():
        cmd_vx = samples[0]["cmd_vx"]; cmd_vy = samples[0]["cmd_vy"]; cmd_yaw = samples[0]["cmd_yaw"]
        summary[segment] = {
            "vx_rmse": float(np.sqrt(np.mean([(s["vx_mean"] - cmd_vx) ** 2 for s in samples]))),
            "vy_rmse": float(np.sqrt(np.mean([(s["vy_mean"] - cmd_vy) ** 2 for s in samples]))),
            "yaw_rmse": float(np.sqrt(np.mean([(s["yaw_mean"] - cmd_yaw) ** 2 for s in samples]))),
            "base_height_mean": float(np.mean([s["base_z_mean"] for s in samples])),
            "abs_roll_mean": float(np.mean([s["abs_roll_mean"] for s in samples])),
            "abs_pitch_mean": float(np.mean([s["abs_pitch_mean"] for s in samples])),
            "action_rms": float(np.mean([s["action_rms"] for s in samples])),
            "torque_rms": float(np.mean([s["torque_rms"] for s in samples])),
            "reset_fraction_mean": float(np.mean([s["reset_fraction"] for s in samples])),
        }
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"tracking_csv={csv_path}")
    print(f"summary_json={summary_path}")
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    evaluate(get_args())
