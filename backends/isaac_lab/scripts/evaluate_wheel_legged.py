#!/usr/bin/env python
"""Deterministically evaluate serial wheel-legged checkpoints in Isaac Sim 6."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, nargs="+")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=71)
parser.add_argument("--video", action="store_true")
parser.add_argument(
    "--video_dir",
    default="artifacts/isaac_sim_6/videos/wheel_legged",
)
parser.add_argument(
    "--video_prefix",
    default="serial_wheel_legged_robust_sim6",
)
parser.add_argument(
    "--delayed",
    action="store_true",
    help="evaluate with the robust asset's randomized 0--20 ms actuator delay",
)
parser.add_argument(
    "--mode",
    choices=("clean", "train_randomization", "holdout"),
    default="clean",
)
parser.add_argument(
    "--out",
    default="artifacts/isaac_sim_6/evaluation/wheel_legged_result.json",
)
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
from tinymal_lab.wheel_legged_env_cfg import (  # noqa: E402
    WheelLeggedFlatEnvCfg,
    WheelLeggedRobustEnvCfg,
)

BACKEND = "Isaac Sim 6.0.1 GA / Isaac Lab 3.0.0-beta2.patch1 / PhysX 5"
STEP_DT = 0.02
COMMAND_OBS_SLICE = slice(9, 12)
SEGMENTS = (
    ("stand", 2.0, (0.0, 0.0, 0.0)),
    ("forward_0p5", 4.0, (0.5, 0.0, 0.0)),
    ("forward_1p0", 4.0, (1.0, 0.0, 0.0)),
    ("backward_0p5", 4.0, (-0.5, 0.0, 0.0)),
    ("yaw_0p8", 4.0, (0.0, 0.0, 0.8)),
    ("arc_0p7_0p6", 4.0, (0.7, 0.0, 0.6)),
)
VIDEO_SEGMENTS = (
    ("stand", 1.5, (0.0, 0.0, 0.0)),
    ("forward_0p5", 2.0, (0.5, 0.0, 0.0)),
    ("backward_0p5", 2.0, (-0.5, 0.0, 0.0)),
    ("forward_1p0", 1.5, (1.0, 0.0, 0.0)),
    ("backward_1p0", 1.5, (-1.0, 0.0, 0.0)),
    ("yaw_left_0p8", 2.0, (0.0, 0.0, 0.8)),
    ("yaw_right_0p8", 2.0, (0.0, 0.0, -0.8)),
    ("arc_left", 2.5, (0.7, 0.0, 0.6)),
    ("arc_reverse", 2.5, (-0.7, 0.0, -0.6)),
)


def build_actor() -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = 28
    for width in (512, 256, 128):
        layers.extend((nn.Linear(previous, width), nn.ELU()))
        previous = width
    layers.append(nn.Linear(previous, 6))
    return nn.Sequential(*layers)


def load_actor(path: str, device: str) -> nn.Sequential:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in payload:
        raise KeyError(f"{path} has no actor_state_dict")
    state = {
        key[len("mlp.") :]: value
        for key, value in payload["actor_state_dict"].items()
        if key.startswith("mlp.")
    }
    actor = build_actor().to(device)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor


def set_command(env, observation, values) -> None:
    target = torch.tensor(values, dtype=torch.float32, device=env.unwrapped.device)
    env.unwrapped.command_manager.get_command("base_velocity")[:] = target
    observation["policy"][:, COMMAND_OBS_SLICE] = target


def make_cfg() -> WheelLeggedFlatEnvCfg:
    cfg = WheelLeggedRobustEnvCfg() if args_cli.delayed else WheelLeggedFlatEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = 1 if args_cli.video else args_cli.num_envs
    cfg.episode_length_s = 40.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.base_velocity.rel_standing_envs = 0.0

    if args_cli.mode == "clean":
        cfg.observations.policy.enable_corruption = False
        cfg.events.physics_material = None
        cfg.events.base_mass = None
        cfg.events.base_com = None
        cfg.events.actuator_gains = None
        cfg.events.push_robot = None
    elif args_cli.mode == "holdout":
        cfg.events.physics_material.params.update(
            {
                "static_friction_range": (0.40, 1.45),
                "dynamic_friction_range": (0.35, 1.35),
                "restitution_range": (0.0, 0.10),
            }
        )
        cfg.events.base_mass.params["mass_distribution_params"] = (0.82, 1.18)
        cfg.events.base_com.params["com_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (-0.015, 0.015),
        }
        cfg.events.actuator_gains.params["stiffness_distribution_params"] = (
            0.80,
            1.20,
        )
        cfg.events.actuator_gains.params["damping_distribution_params"] = (
            0.80,
            1.20,
        )
        cfg.events.push_robot.interval_range_s = (4.0, 6.0)
        cfg.events.push_robot.params["velocity_range"] = {
            "x": (-0.70, 0.70),
            "y": (-0.50, 0.50),
            "yaw": (-0.50, 0.50),
        }

    # Remove reset-state randomness in clean mode; randomized physics remains
    # independently sampled in both robustness modes.
    if args_cli.mode == "clean":
        cfg.events.reset_base.params["pose_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "roll", "pitch", "yaw")
        }
        cfg.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        cfg.events.reset_joints.params["position_range"] = (0.0, 0.0)
        cfg.events.reset_joints.params["velocity_range"] = (0.0, 0.0)
    if args_cli.video:
        cfg.viewer.eye = (1.3, -3.2, 1.35)
        cfg.viewer.lookat = (0.35, 0.0, 0.38)
        cfg.viewer.origin_type = "world"
        cfg.viewer.resolution = (1280, 720)
    return cfg


def evaluate_checkpoint(checkpoint: str) -> dict:
    task = (
        "Isaac-Velocity-Robust-SerialWheelLegged-v0"
        if args_cli.delayed
        else "Isaac-Velocity-Flat-SerialWheelLegged-v0"
    )
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(task, cfg=make_cfg(), render_mode=render_mode)
    segments = VIDEO_SEGMENTS if args_cli.video else SEGMENTS
    if args_cli.video:
        os.makedirs(args_cli.video_dir, exist_ok=True)
        video_steps = sum(round(duration / STEP_DT) for _, duration, _ in segments)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.abspath(args_cli.video_dir),
            step_trigger=lambda step: step == 0,
            video_length=video_steps,
            name_prefix=args_cli.video_prefix,
            disable_logger=True,
        )
    try:
        device = env.unwrapped.device
        num_envs = env.unwrapped.num_envs
        actor = load_actor(checkpoint, device)
        observation, _ = env.reset()
        reset_seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
        result: dict[str, dict] = {}

        with torch.no_grad():
            for name, duration, command in segments:
                set_command(env, observation, command)
                num_steps = round(duration / STEP_DT)
                settle_steps = round(min(1.0, duration / 3.0) / STEP_DT)
                squared_error = torch.zeros(3, device=device)
                upright_angle_sum = torch.zeros((), device=device)
                sample_count = 0
                reset_count = 0
                min_height = math.inf
                max_abs_pitch_rate = 0.0

                for step in range(num_steps):
                    action = actor(observation["policy"])
                    observation, _, terminated, truncated, _ = env.step(action)
                    set_command(env, observation, command)
                    reset_count += int(terminated.sum().item())
                    reset_seen |= terminated

                    data = env.unwrapped.scene["robot"].data
                    min_height = min(
                        min_height, float(data.root_pos_w.torch[:, 2].min().item())
                    )
                    max_abs_pitch_rate = max(
                        max_abs_pitch_rate,
                        float(data.root_ang_vel_b.torch[:, 1].abs().max().item()),
                    )
                    if step >= settle_steps:
                        actual = torch.stack(
                            (
                                data.root_lin_vel_b.torch[:, 0],
                                data.root_lin_vel_b.torch[:, 1],
                                data.root_ang_vel_b.torch[:, 2],
                            ),
                            dim=1,
                        )
                        target = torch.tensor(command, device=device).unsqueeze(0)
                        squared_error += torch.square(actual - target).sum(dim=0)
                        upright_angle_sum += torch.acos(
                            torch.clamp(
                                -data.projected_gravity_b.torch[:, 2], -1.0, 1.0
                            )
                        ).sum()
                        sample_count += actual.shape[0]
                    # A 40 s episode is longer than the 22 s suite, so any
                    # truncation here indicates an unexpected configuration bug.
                    if truncated.any():
                        raise RuntimeError("unexpected timeout during evaluation suite")

                rmse = torch.sqrt(squared_error / max(1, sample_count))
                result[name] = {
                    "command": list(command),
                    "vx_rmse": float(rmse[0].item()),
                    "vy_rmse": float(rmse[1].item()),
                    "yaw_rmse": float(rmse[2].item()),
                    "mean_upright_error_rad": float(
                        upright_angle_sum.item() / max(1, sample_count)
                    ),
                    "max_abs_pitch_rate_radps": max_abs_pitch_rate,
                    "minimum_base_height_m": min_height,
                    "falls": reset_count,
                }

        primary_errors = []
        for segment in result.values():
            command = segment["command"]
            if command[0] != 0.0:
                primary_errors.append(segment["vx_rmse"])
            if command[2] != 0.0:
                primary_errors.append(segment["yaw_rmse"])
            if command[0] == 0.0 and command[2] == 0.0:
                primary_errors.append(segment["vx_rmse"])
        return {
            "backend": BACKEND,
            "checkpoint": checkpoint,
            "mode": args_cli.mode,
            "randomized_actuator_delay_ms": [0, 20] if args_cli.delayed else None,
            "seed": args_cli.seed,
            "num_envs": num_envs,
            "video_dir": (
                os.path.abspath(args_cli.video_dir) if args_cli.video else None
            ),
            "segments": result,
            "falls_total": sum(item["falls"] for item in result.values()),
            "envs_with_falls": int(reset_seen.sum().item()),
            "clean_env_fraction": float((~reset_seen).float().mean().item()),
            "mean_primary_axis_rmse": sum(primary_errors) / len(primary_errors),
        }
    finally:
        env.close()


def main() -> None:
    if args_cli.video and len(args_cli.ckpt) != 1:
        raise ValueError("--video accepts exactly one checkpoint")
    evaluations = [
        evaluate_checkpoint(os.path.abspath(checkpoint)) for checkpoint in args_cli.ckpt
    ]
    ranking = sorted(
        evaluations,
        key=lambda item: (
            item["falls_total"],
            item["mean_primary_axis_rmse"],
        ),
    )
    payload = (
        evaluations[0]
        if len(evaluations) == 1
        else {
            "backend": BACKEND,
            "mode": args_cli.mode,
            "best_checkpoint": ranking[0]["checkpoint"],
            "ranking": [item["checkpoint"] for item in ranking],
            "results": evaluations,
        }
    )
    out_path = os.path.abspath(args_cli.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
    simulation_app.close()
