"""Isaac Gym evaluation of a MuJoCo-trained TinyMal policy (reverse sim2sim).

Loads the MuJoCo-trained actor weights into a fresh ActorCritic (same
architecture: 48->512->256->128->12 ELU) and runs the 6-segment tracking eval
in the Isaac Gym tinymal env. Writes tracking.csv + summary.json.

This is the reverse direction: MuJoCo policy deployed in Isaac Gym (PhysX).
Compare against the Isaac-Gym-trained policy's in-Isaac-Gym numbers and against
the MuJoCo policy's in-MuJoCo numbers.

Run: python evaluate_mujoco_policy_in_isaac.py --checkpoint MODEL.pt --headless
"""

import csv
import json
import os
import sys
from collections import defaultdict

import isaacgym  # noqa: F401  # Must precede torch.
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


SEGMENTS = (
    ("stand", 2.0, (0.0, 0.0, 0.0)),
    ("forward_0p3", 3.0, (0.3, 0.0, 0.0)),
    ("forward_0p6", 3.0, (0.6, 0.0, 0.0)),
    ("backward_0p3", 3.0, (-0.3, 0.0, 0.0)),
    ("lateral_0p2", 3.0, (0.0, 0.2, 0.0)),
    ("yaw_0p5", 3.0, (0.0, 0.0, 0.5)),
)

MUJOCO_CHECKPOINT = os.environ.get(
    "MUJOCO_CHECKPOINT",
    os.path.join(LEGGED_GYM_ROOT_DIR, "artifacts", "mujoco", "model.pt"),
)
OUTPUT_DIR = os.environ.get(
    "ISAAC_GYM_TRANSFER_OUT",
    os.path.join(LEGGED_GYM_ROOT_DIR, "artifacts", "reverse_sim2sim"),
)
POS_VX_INPUT_GAIN = float(os.environ.get("ISAAC_GYM_POS_VX_INPUT_GAIN", "1.0"))
NEG_VX_INPUT_GAIN = float(os.environ.get("ISAAC_GYM_NEG_VX_INPUT_GAIN", "1.0"))
VY_INPUT_GAIN = float(os.environ.get("ISAAC_GYM_VY_INPUT_GAIN", "1.0"))
YAW_INPUT_GAIN = float(os.environ.get("ISAAC_GYM_YAW_INPUT_GAIN", "1.0"))


def policy_command(command):
    vx_gain = POS_VX_INPUT_GAIN if command[0] >= 0.0 else NEG_VX_INPUT_GAIN
    return (
        command[0] * vx_gain,
        command[1] * VY_INPUT_GAIN,
        command[2] * YAW_INPUT_GAIN,
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

    # Build the runner to create an ActorCritic, then load MuJoCo weights.
    runner, _ = task_registry.make_alg_runner(
        env=env, name="tinymal", args=args, train_cfg=train_cfg
    )
    # Load MuJoCo-trained checkpoint into the actor_critic.
    print(f"Loading MuJoCo-trained weights from: {MUJOCO_CHECKPOINT}")
    ck = torch.load(MUJOCO_CHECKPOINT, map_location=env.device)
    runner.alg.actor_critic.load_state_dict(ck["model_state_dict"])
    print(f"  iter in checkpoint: {ck.get('iter', '?')}")
    # Verify key shapes match.
    for k in ["actor.0.weight", "actor.6.weight"]:
        print(f"  {k}: {ck['model_state_dict'][k].shape}")

    policy = runner.alg.actor_critic.act_inference
    runner.alg.actor_critic.eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "tracking.csv")
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")

    rows = []
    segment_samples = defaultdict(list)
    global_step = 0
    dt = env.dt

    with torch.inference_mode():
        for segment, duration, command in SEGMENTS:
            actor_command = policy_command(command)
            steps = int(round(duration / dt))
            settle_steps = int(round(min(1.0, duration / 3.0) / dt))
            for local_step in range(steps):
                env.commands[:, 0] = actor_command[0]
                env.commands[:, 1] = actor_command[1]
                env.commands[:, 2] = actor_command[2]
                env.compute_observations()
                actions = policy(env.get_observations())
                _, _, _, dones, _ = env.step(actions)

                # Fall detection per env (roll/pitch thresholds from legged_robot).
                fallen_mask = ((env.rpy[:, 0].abs() > 0.8) |
                               (env.rpy[:, 1].abs() > 1.0) |
                               (env.base_pos[:, 2] < 0.10))
                sample = {
                    "time_s": global_step * dt,
                    "segment": segment,
                    "cmd_vx": command[0],
                    "cmd_vy": command[1],
                    "cmd_yaw": command[2],
                    "policy_cmd_vx": actor_command[0],
                    "policy_cmd_vy": actor_command[1],
                    "policy_cmd_yaw": actor_command[2],
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
                    "fallen_fraction": fallen_mask.float().mean().item(),
                }
                rows.append(sample)
                if local_step >= settle_steps:
                    segment_samples[segment].append(sample)
                global_step += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for segment, samples in segment_samples.items():
        cmd_vx = samples[0]["cmd_vx"]
        cmd_vy = samples[0]["cmd_vy"]
        cmd_yaw = samples[0]["cmd_yaw"]
        fallen_fracs = [s["fallen_fraction"] for s in samples]
        reset_fracs = [s["reset_fraction"] for s in samples]
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
            "torque_rms": float(np.mean([s["torque_rms"] for s in samples])),
            "reset_fraction_mean": float(np.mean(reset_fracs)),
            "fallen_fraction_mean": float(np.mean(fallen_fracs)),
            "survived": bool(np.mean(fallen_fracs) < 0.5
                             and np.mean(reset_fracs) < 0.5),
        }

    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"tracking_csv={csv_path}")
    print(f"summary_json={summary_path}")
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    evaluate(get_args())
