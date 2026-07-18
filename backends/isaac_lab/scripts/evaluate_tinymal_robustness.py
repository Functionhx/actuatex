#!/usr/bin/env python
"""Stress-test a TinyMal actor under the final Isaac Lab randomization stack."""

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
    help="one or more checkpoints; multiple paths share one simulator launch",
)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--out", default="evaluation/isaac_lab_native/robustness_result.json")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import tinymal_lab  # noqa: E402,F401
from tinymal_lab.tinymal_robust_env_cfg import TinymalRobustEnvCfg  # noqa: E402


SEGMENTS = (
    ("stand", 2.0, (0.0, 0.0, 0.0)),
    ("forward_0p3", 3.0, (0.3, 0.0, 0.0)),
    ("forward_0p6", 3.0, (0.6, 0.0, 0.0)),
    ("backward_0p3", 3.0, (-0.3, 0.0, 0.0)),
    ("lateral_0p2", 3.0, (0.0, 0.2, 0.0)),
    ("yaw_0p5", 3.0, (0.0, 0.0, 0.5)),
)


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


def set_command(env, obs, command):
    target = torch.tensor(command, device=env.unwrapped.device)
    env.unwrapped.command_manager.get_command("base_velocity")[:] = target
    obs["policy"][:, 9:12] = target * torch.tensor(
        (2.0, 2.0, 0.25), device=env.unwrapped.device
    )


def main():
    cfg = TinymalRobustEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.events.configure_stairs.params["flat_fraction"] = 1.0
    cfg.terminations.stair_course_complete = None
    cfg.episode_length_s = 30.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.base_velocity.rel_standing_envs = 0.0

    env = gym.make("Isaac-Velocity-Native-Robust-TinyMal-v0", cfg=cfg)
    device = env.unwrapped.device
    stressors = {
            "friction": [0.45, 1.35],
            "base_mass_scale": [0.85, 1.15],
            "base_com_m": {"xy": [-0.012, 0.012], "z": [-0.008, 0.008]},
            "pd_stiffness_scale": [0.85, 1.15],
            "pd_damping_scale": [0.70, 1.30],
            "actuator_delay_ms": [20, 60],
            "push_velocity_xy_mps": [-0.60, 0.60],
            "push_yaw_rate_radps": [-0.50, 0.50],
            "push_interval_s": [4.0, 7.0],
            "observation_noise": True,
    }
    evaluations = []
    for checkpoint in args_cli.ckpt:
        checkpoint = os.path.abspath(checkpoint)
        actor = load_actor(checkpoint, device)
        obs, _ = env.reset()
        result = {}

        # Keep gradients disabled while allowing Isaac Lab's automatic resets
        # to update state buffers in place between checkpoint rollouts.
        with torch.no_grad():
            for name, duration, command in SEGMENTS:
                set_command(env, obs, command)
                num_steps = int(round(duration / 0.02))
                settle_steps = int(round(min(1.0, duration / 3.0) / 0.02))
                square_error = torch.zeros(3, device=device)
                sample_count = 0
                reset_count = 0
                min_height = float("inf")
                for step in range(num_steps):
                    actions = actor(obs["policy"])
                    obs, _, terminated, _, _ = env.step(actions)
                    set_command(env, obs, command)
                    reset_count += int(terminated.sum().item())
                    robot = env.unwrapped.scene["robot"].data
                    min_height = min(
                        min_height, float(robot.root_pos_w[:, 2].min().item())
                    )
                    if step >= settle_steps:
                        actual = torch.stack(
                            (
                                robot.root_lin_vel_b[:, 0],
                                robot.root_lin_vel_b[:, 1],
                                robot.root_ang_vel_b[:, 2],
                            ),
                            dim=1,
                        )
                        target = torch.tensor(command, device=device).unsqueeze(0)
                        square_error += torch.square(actual - target).sum(dim=0)
                        sample_count += actual.shape[0]
                rmse = torch.sqrt(square_error / max(1, sample_count))
                result[name] = {
                    "command": list(command),
                    "vx_rmse": float(rmse[0].item()),
                    "vy_rmse": float(rmse[1].item()),
                    "yaw_rmse": float(rmse[2].item()),
                    "resets_total": reset_count,
                    "step_survival_fraction": 1.0
                    - reset_count / float(max(1, num_steps * args_cli.num_envs)),
                    "minimum_base_height_m": min_height,
                }

        main_errors = []
        for segment in result.values():
            command = segment["command"]
            if command[0] != 0.0 or (command[1] == 0.0 and command[2] == 0.0):
                main_errors.append(segment["vx_rmse"])
            elif command[1] != 0.0:
                main_errors.append(segment["vy_rmse"])
            else:
                main_errors.append(segment["yaw_rmse"])
        evaluations.append(
            {
                "backend": "Isaac Sim / Isaac Lab / PhysX 5",
                "checkpoint": checkpoint,
                "seed": args_cli.seed,
                "num_envs": args_cli.num_envs,
                "stressors": stressors,
                "segments": result,
                "resets_total": sum(
                    segment["resets_total"] for segment in result.values()
                ),
                "mean_main_axis_rmse": sum(main_errors) / len(main_errors),
            }
        )
        del actor

    if len(evaluations) == 1:
        payload = evaluations[0]
    else:
        ranking = sorted(
            evaluations,
            key=lambda item: (item["resets_total"], item["mean_main_axis_rmse"]),
        )
        payload = {
            "backend": "Isaac Sim / Isaac Lab / PhysX 5",
            "best_checkpoint": ranking[0]["checkpoint"],
            "ranking": [item["checkpoint"] for item in ranking],
            "results": evaluations,
        }
    out_path = os.path.abspath(args_cli.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
