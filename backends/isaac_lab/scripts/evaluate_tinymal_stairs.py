#!/usr/bin/env python
"""Evaluate and optionally record a native Isaac Lab TinyMal stair ascent."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt",
    required=True,
    nargs="+",
    help="one or more checkpoints; multiple paths are evaluated in one simulator launch",
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--vx", type=float, default=0.45)
parser.add_argument("--seed", type=int, default=5)
parser.add_argument("--video", action="store_true")
parser.add_argument(
    "--robust",
    action="store_true",
    help="evaluate with randomized friction/mass/PD, 20--60 ms delay, noise, and pushes",
)
parser.add_argument("--video_dir", default="evaluation/ppt_videos/isaac_lab_raw")
parser.add_argument("--video_prefix", default="tinymal_isaac_lab_stairs")
parser.add_argument("--out", default="evaluation/isaac_lab_native/stairs_result.json")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import tinymal_lab  # noqa: E402,F401
from tinymal_lab.tinymal_stair_env_cfg import TinymalStairEnvCfg  # noqa: E402
from tinymal_lab.tinymal_robust_env_cfg import TinymalRobustEnvCfg  # noqa: E402


def build_actor():
    layers = []
    previous = 48
    for width in (512, 256, 128):
        layers.extend((nn.Linear(previous, width), nn.ELU()))
        previous = width
    layers.append(nn.Linear(previous, 12))
    return nn.Sequential(*layers)


def load_actor(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in payload:
        state = {
            key[len("actor.") :]: value
            for key, value in payload["model_state_dict"].items()
            if key.startswith("actor.")
        }
    elif "actor_state_dict" in payload:
        state = {
            key[len("mlp.") :]: value
            for key, value in payload["actor_state_dict"].items()
            if key.startswith("mlp.")
        }
    else:
        raise KeyError("checkpoint has neither model_state_dict nor actor_state_dict")
    actor = build_actor().to(device)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor


def make_cfg():
    cfg = TinymalRobustEnvCfg() if args_cli.robust else TinymalStairEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = 1 if args_cli.video else args_cli.num_envs
    cfg.observations.policy.enable_corruption = args_cli.robust
    cfg.events.configure_stairs.params["flat_fraction"] = 0.0
    if not args_cli.robust:
        cfg.events.physics_material = None
        cfg.events.push_robot = None
        cfg.events.add_base_mass = None
        cfg.events.base_com = None
    cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)
    }
    cfg.events.reset_base.params["velocity_range"] = {
        axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    # Evaluate through the complete rollout instead of auto-resetting on top.
    cfg.terminations.stair_course_complete = None
    cfg.episode_length_s = max(20.0, args_cli.steps * 0.02 + 1.0)
    if args_cli.video:
        # A long top deck keeps the final part of the PPT shot clean.
        cfg.scene.stair_top.spawn.size = (2.2, 1.20, 0.10)
        cfg.scene.stair_top.init_state.pos = (2.45, 0.0, 0.05)
        cfg.viewer.eye = (2.7, -2.5, 1.25)
        cfg.viewer.lookat = (1.05, 0.0, 0.22)
        cfg.viewer.origin_type = "world"
    return cfg


def main():
    if args_cli.video and len(args_cli.ckpt) != 1:
        raise ValueError("--video accepts exactly one --ckpt path")
    cfg = make_cfg()
    render_mode = "rgb_array" if args_cli.video else None
    task = (
        "Isaac-Velocity-Native-Robust-TinyMal-v0"
        if args_cli.robust
        else "Isaac-Velocity-Native-Stairs-TinyMal-v0"
    )
    env = gym.make(task, cfg=cfg, render_mode=render_mode)
    if args_cli.video:
        os.makedirs(args_cli.video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.abspath(args_cli.video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.steps,
            name_prefix=args_cli.video_prefix,
            disable_logger=True,
        )

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    results = []
    for checkpoint in args_cli.ckpt:
        checkpoint = os.path.abspath(checkpoint)
        actor = load_actor(checkpoint, device)
        obs, _ = env.reset()
        command = env.unwrapped.command_manager.get_command("base_velocity")
        command[:, 0] = args_cli.vx
        command[:, 1:] = 0.0

        reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
        first_attempt_reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
        reset_seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
        reach_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        reset_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        max_local_x = torch.full((num_envs,), -1.0e9, device=device)
        max_local_height = torch.full((num_envs,), -1.0e9, device=device)

        # Isaac Lab performs in-place state writes during automatic resets.
        # ``inference_mode`` would permanently mark those buffers as inference
        # tensors and make the next checkpoint's explicit reset illegal.
        with torch.no_grad():
            for step in range(args_cli.steps):
                command[:] = torch.tensor((args_cli.vx, 0.0, 0.0), device=device)
                actions = actor(obs["policy"])
                obs, _, terminated, _, _ = env.step(actions)
                robot = env.unwrapped.scene["robot"].data
                local_pos = robot.root_pos_w - env.unwrapped.scene.env_origins
                max_local_x = torch.maximum(max_local_x, local_pos[:, 0])
                max_local_height = torch.maximum(max_local_height, local_pos[:, 2])
                just_reached = (~reached) & (local_pos[:, 0] >= 1.55)
                reach_step[just_reached] = step
                reached |= just_reached
                first_attempt_reached |= just_reached & (~reset_seen)
                reset_seen |= terminated
                reset_counts += terminated.to(dtype=torch.long)

        results.append(
            {
                "backend": "Isaac Sim / Isaac Lab / PhysX 5",
                "checkpoint": checkpoint,
                "seed": args_cli.seed,
                "num_envs": num_envs,
                "command_vx_mps": args_cli.vx,
                "step_height_m": 0.02,
                "num_steps": 5,
                "robust_randomization": args_cli.robust,
                "successes": int(reached.sum().item()),
                "success_rate": float(reached.float().mean().item()),
                "first_attempt_successes": int(first_attempt_reached.sum().item()),
                "first_attempt_success_rate": float(
                    first_attempt_reached.float().mean().item()
                ),
                "clean_rollout_successes": int(
                    (reached & (~reset_seen)).sum().item()
                ),
                "clean_rollout_success_rate": float(
                    (reached & (~reset_seen)).float().mean().item()
                ),
                "mean_time_to_top_s": (
                    float(reach_step[reached].float().mean().item() * 0.02)
                    if reached.any()
                    else None
                ),
                "resets_total": int(reset_counts.sum().item()),
                "envs_with_resets": int(reset_seen.sum().item()),
                "max_local_x_mean_m": float(max_local_x.mean().item()),
                "max_base_height_mean_m": float(max_local_height.mean().item()),
                "video_dir": os.path.abspath(args_cli.video_dir) if args_cli.video else None,
            }
        )
        del actor

    if len(results) == 1:
        result = results[0]
    else:
        ranked = sorted(
            results,
            key=lambda item: (
                -item["clean_rollout_success_rate"],
                -item["first_attempt_success_rate"],
                -item["success_rate"],
                item["resets_total"],
                item["mean_time_to_top_s"] if item["mean_time_to_top_s"] is not None else 1.0e9,
            ),
        )
        result = {
            "backend": "Isaac Sim / Isaac Lab / PhysX 5",
            "best_checkpoint": ranked[0]["checkpoint"],
            "ranking": [item["checkpoint"] for item in ranked],
            "results": results,
        }
    out_path = os.path.abspath(args_cli.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
