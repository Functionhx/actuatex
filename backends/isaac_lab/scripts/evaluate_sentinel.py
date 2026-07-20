#!/usr/bin/env python
"""Evaluate RL, LQR and H-infinity Sentinel controllers in Isaac Sim 6."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", nargs="+", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=71)
parser.add_argument(
    "--mode",
    choices=("clean", "train_randomization", "holdout"),
    default="clean",
)
parser.add_argument("--duration_scale", type=float, default=1.0)
parser.add_argument("--command_ramp_s", type=float, default=0.0)
parser.add_argument("--trace_dir", type=Path)
parser.add_argument(
    "--output",
    type=Path,
    default=REPO_ROOT / "artifacts/sentinel/evaluation/isaac_sim_6.json",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if (
    args_cli.num_envs <= 0
    or args_cli.duration_scale <= 0.0
    or args_cli.command_ramp_s < 0.0
):
    parser.error("--num_envs and --duration_scale must be positive")
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import tinymal_lab  # noqa: E402,F401
from tasks.robomaster.contract import POLICY_DT, contract_sha256  # noqa: E402
from tasks.robomaster.evaluation import (  # noqa: E402
    SENTINEL_COMMAND_SEGMENTS,
    evaluation_settle_steps,
    ramped_command,
)
from tasks.robomaster.evaluation_trace import SentinelTraceRecorder  # noqa: E402
from tasks.robomaster.policy import load_policy  # noqa: E402
from tinymal_lab.sentinel_env_cfg import (  # noqa: E402
    SentinelFlatEnvCfg,
    SentinelRobustEnvCfg,
)


BACKEND = "Isaac Sim 6.0.1 GA / Isaac Lab 3.0.0-beta2.patch1 / PhysX 5"
COMMAND_OBSERVATION_SLICE = slice(9, 12)
def make_config() -> SentinelFlatEnvCfg:
    if args_cli.mode == "clean":
        config = SentinelFlatEnvCfg()
    else:
        config = SentinelRobustEnvCfg()
    config.seed = args_cli.seed
    config.scene.num_envs = args_cli.num_envs
    config.episode_length_s = 60.0
    config.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    config.commands.base_velocity.rel_standing_envs = 0.0

    if args_cli.mode == "clean":
        config.observations.policy.enable_corruption = False
        config.events.physics_material = None
        config.events.base_mass = None
        config.events.base_com = None
        config.events.actuator_gains = None
        config.events.push_robot = None
        config.events.reset_base.params["pose_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "roll", "pitch", "yaw")
        }
        config.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        config.events.reset_joints.params["position_range"] = (0.0, 0.0)
        config.events.reset_joints.params["velocity_range"] = (0.0, 0.0)
    elif args_cli.mode == "holdout":
        config.events.physics_material.params.update(
            {
                "static_friction_range": (0.35, 1.55),
                "dynamic_friction_range": (0.30, 1.45),
                "restitution_range": (0.0, 0.15),
            }
        )
        config.events.base_mass.params["mass_distribution_params"] = (
            0.80,
            1.20,
        )
        config.events.base_com.params["com_range"] = {
            "x": (-0.04, 0.04),
            "y": (-0.04, 0.04),
            "z": (-0.02, 0.02),
        }
        config.events.push_robot.interval_range_s = (4.0, 6.0)
        config.events.push_robot.params["velocity_range"] = {
            "x": (-0.85, 0.85),
            "y": (-0.60, 0.60),
            "yaw": (-0.65, 0.65),
        }
    return config


def set_command(env, observation: dict, values: tuple[float, ...]) -> None:
    target = torch.tensor(
        values,
        dtype=torch.float32,
        device=env.unwrapped.device,
    )
    env.unwrapped.command_manager.get_command("base_velocity")[:] = target
    observation["policy"][:, COMMAND_OBSERVATION_SLICE] = target


def evaluate_checkpoint(checkpoint: Path) -> dict:
    task = (
        "Isaac-Velocity-Flat-Sentinel-v0"
        if args_cli.mode == "clean"
        else "Isaac-Velocity-Robust-Sentinel-v0"
    )
    env = gym.make(task, cfg=make_config())
    try:
        device = env.unwrapped.device
        num_envs = env.unwrapped.num_envs
        loaded = load_policy(checkpoint, device=device)
        observation, _ = env.reset()
        reset_seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
        result: dict[str, dict] = {}
        previous_command = (0.0, 0.0, 0.0)
        trace = SentinelTraceRecorder(num_envs)
        elapsed_steps = 0
        robot = env.unwrapped.scene["robot"]
        actuator = robot.actuators["shared_dc_bank"]

        with torch.inference_mode():
            for segment in SENTINEL_COMMAND_SEGMENTS:
                duration = segment.duration_s * args_cli.duration_scale
                num_steps = max(1, round(duration / POLICY_DT))
                ramp_duration = min(args_cli.command_ramp_s, duration)
                settle_steps = evaluation_settle_steps(
                    num_steps=num_steps,
                    duration_s=duration,
                    dt=POLICY_DT,
                    ramp_duration_s=ramp_duration,
                )
                squared_error = torch.zeros(3, device=device)
                upright_error_sum = torch.zeros((), device=device)
                sample_count = 0
                falls = 0
                minimum_height = math.inf
                maximum_pitch_rate = 0.0
                maximum_temperature = 25.0
                minimum_buffer = 60.0
                maximum_chassis_power = 0.0
                disabled_samples = 0

                for step in range(num_steps):
                    current_command = ramped_command(
                        previous_command,
                        segment.command,
                        step=step,
                        dt=POLICY_DT,
                        ramp_duration_s=ramp_duration,
                    )
                    set_command(env, observation, current_command)
                    action = loaded.actor(observation["policy"])
                    if not bool(torch.isfinite(action).all()):
                        raise RuntimeError("policy produced a non-finite action")
                    observation, _, terminated, truncated, _ = env.step(action)
                    if bool(truncated.any()):
                        raise RuntimeError(
                            "unexpected timeout during Sentinel evaluation"
                        )
                    falls += int(terminated.sum().item())
                    reset_seen |= terminated
                    data = robot.data
                    minimum_height = min(
                        minimum_height,
                        float(data.root_pos_w.torch[:, 2].min().item()),
                    )
                    maximum_pitch_rate = max(
                        maximum_pitch_rate,
                        float(data.root_ang_vel_b.torch[:, 1].abs().max().item()),
                    )
                    maximum_temperature = max(
                        maximum_temperature,
                        float(actuator.motor_temperature_c.max().item()),
                    )
                    minimum_buffer = min(
                        minimum_buffer,
                        float(actuator.referee.buffer_energy_j.min().item()),
                    )
                    maximum_chassis_power = max(
                        maximum_chassis_power,
                        float(actuator.accounted_chassis_power_w.max().item()),
                    )
                    disabled_samples += int(
                        (~actuator.referee.chassis_enabled).sum().item()
                    )
                    trace.record(
                        time_s=elapsed_steps * POLICY_DT,
                        segment=segment.name,
                        command=current_command,
                        base_linear_velocity_body=(
                            data.root_lin_vel_b.torch.detach().cpu().numpy()
                        ),
                        base_angular_velocity_body=(
                            data.root_ang_vel_b.torch.detach().cpu().numpy()
                        ),
                        projected_gravity=(
                            data.projected_gravity_b.torch.detach().cpu().numpy()
                        ),
                        base_height_m=(
                            data.root_pos_w.torch[:, 2].detach().cpu().numpy()
                        ),
                        action=action.detach().cpu().numpy(),
                        chassis_power_w=(
                            actuator.accounted_chassis_power_w.detach()
                            .cpu()
                            .numpy()
                        ),
                        motor_temperature_c=(
                            actuator.motor_temperature_c.detach().cpu().numpy()
                        ),
                        buffer_energy_j=(
                            actuator.referee.buffer_energy_j.detach().cpu().numpy()
                        ),
                        terminal=terminated.detach().cpu().numpy(),
                    )
                    elapsed_steps += 1
                    if step >= settle_steps:
                        actual = torch.stack(
                            (
                                data.root_lin_vel_b.torch[:, 0],
                                data.root_lin_vel_b.torch[:, 1],
                                data.root_ang_vel_b.torch[:, 2],
                            ),
                            dim=1,
                        )
                        target = torch.tensor(
                            segment.command, device=device
                        ).unsqueeze(0)
                        squared_error += torch.square(actual - target).sum(dim=0)
                        upright_error_sum += torch.acos(
                            torch.clamp(
                                -data.projected_gravity_b.torch[:, 2],
                                -1.0,
                                1.0,
                            )
                        ).sum()
                        sample_count += num_envs

                rmse = torch.sqrt(squared_error / max(1, sample_count))
                result[segment.name] = {
                    "command": list(segment.command),
                    "duration_s": duration,
                    "command_ramp_s": ramp_duration,
                    "vx_rmse": float(rmse[0].item()),
                    "vy_rmse": float(rmse[1].item()),
                    "yaw_rmse": float(rmse[2].item()),
                    "mean_upright_error_rad": float(
                        upright_error_sum.item() / max(1, sample_count)
                    ),
                    "maximum_abs_pitch_rate_radps": maximum_pitch_rate,
                    "minimum_base_height_m": minimum_height,
                    "maximum_motor_temperature_c": maximum_temperature,
                    "minimum_buffer_energy_j": minimum_buffer,
                    "maximum_chassis_power_w": maximum_chassis_power,
                    "disabled_chassis_samples": disabled_samples,
                    "falls": falls,
                }
                previous_command = segment.command

        trace_csv = None
        trace_plot = None
        if args_cli.trace_dir is not None:
            trace_stem = (
                f"{checkpoint.stem}_seed{args_cli.seed}_"
                f"{args_cli.mode}_ramp{args_cli.command_ramp_s:g}"
            )
            trace_csv = args_cli.trace_dir / f"{trace_stem}.csv"
            trace_plot = args_cli.trace_dir / f"{trace_stem}.svg"
            trace.write_csv(trace_csv)
            trace.write_svg(
                trace_plot,
                title=(
                    f"Isaac Sim 6 · {loaded.algorithm.upper()} · "
                    f"{args_cli.mode} · seed {args_cli.seed}"
                ),
            )

        primary_errors = []
        for segment in result.values():
            command = segment["command"]
            if command[0] != 0.0:
                primary_errors.append(segment["vx_rmse"])
            if command[2] != 0.0:
                primary_errors.append(segment["yaw_rmse"])
            if command[0] == 0.0 and command[2] == 0.0:
                primary_errors.append(segment["vx_rmse"])
        source_backend = str(loaded.metadata.get("backend", "unknown"))
        if source_backend.startswith("Isaac"):
            transfer_kind = "native"
        elif source_backend.startswith("MuJoCo"):
            transfer_kind = "sim2sim_to_isaac"
        else:
            transfer_kind = "unknown_source"
        return {
            "backend": BACKEND,
            "checkpoint": str(checkpoint.resolve()),
            "algorithm": loaded.algorithm,
            "checkpoint_format": loaded.checkpoint_format,
            "source_backend": source_backend,
            "native_or_sim2sim": transfer_kind,
            "mode": args_cli.mode,
            "contract_sha256": contract_sha256(),
            "num_envs": num_envs,
            "seed": args_cli.seed,
            "randomized_command_delay_ms": (
                [0, 0] if args_cli.mode == "clean" else [0, 20]
            ),
            "command_ramp_s": args_cli.command_ramp_s,
            "trace_csv": None if trace_csv is None else str(trace_csv.resolve()),
            "trace_plot": (
                None if trace_plot is None else str(trace_plot.resolve())
            ),
            "segments": result,
            "falls_total": sum(item["falls"] for item in result.values()),
            "envs_with_falls": int(reset_seen.sum().item()),
            "clean_env_fraction": float((~reset_seen).float().mean().item()),
            "mean_primary_axis_rmse": float(sum(primary_errors) / len(primary_errors)),
        }
    finally:
        env.close()


def main() -> None:
    evaluations = [
        evaluate_checkpoint(checkpoint) for checkpoint in args_cli.checkpoint
    ]
    ranking = sorted(
        evaluations,
        key=lambda item: (item["falls_total"], item["mean_primary_axis_rmse"]),
    )
    payload: dict | list = (
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
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
