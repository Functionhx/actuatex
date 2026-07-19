#!/usr/bin/env python3
"""Evaluate PPO/SAC/TD3 Sentinel checkpoints natively or via sim2sim."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from actuatex_paths import ARTIFACTS_ROOT, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))

from sentinel_env import MjSentinelEnv  # noqa: E402
from tasks.robomaster.contract import (  # noqa: E402
    POLICY_DT,
    SIM_DT,
    contract_sha256,
)
from tasks.robomaster.policy import load_policy  # noqa: E402


SEGMENTS = (
    ("stand", 2.0, (0.0, 0.0, 0.0)),
    ("forward_0p5", 4.0, (0.5, 0.0, 0.0)),
    ("forward_1p0", 4.0, (1.0, 0.0, 0.0)),
    ("backward_0p5", 4.0, (-0.5, 0.0, 0.0)),
    ("yaw_0p8", 4.0, (0.0, 0.0, 0.8)),
    ("arc_0p7_0p6", 4.0, (0.7, 0.0, 0.6)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--maximum-command-delay-steps", type=int, default=0)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_ROOT / "sentinel/evaluation/mujoco.json",
    )
    return parser.parse_args()


def evaluate_checkpoint(args: argparse.Namespace, checkpoint: Path) -> dict:
    env = MjSentinelEnv(
        num_envs=args.num_envs,
        num_threads=args.num_threads,
        seed=args.seed,
        add_noise=False,
        randomize_reset=False,
        maximum_command_delay_steps=args.maximum_command_delay_steps,
        episode_length_s=60.0,
    )
    loaded = load_policy(checkpoint, device=args.device)
    actor = loaded.actor
    observation, _ = env.reset()
    reset_seen = np.zeros(args.num_envs, dtype=bool)
    result: dict[str, dict] = {}
    try:
        with torch.inference_mode():
            for segment_name, nominal_duration, command in SEGMENTS:
                duration = nominal_duration * args.duration_scale
                num_steps = max(1, round(duration / POLICY_DT))
                settle_steps = min(
                    num_steps - 1,
                    round(min(1.0, duration / 3.0) / POLICY_DT),
                )
                env.set_command(np.asarray(command))
                squared_error = np.zeros(3, dtype=np.float64)
                upright_error_sum = 0.0
                sample_count = 0
                falls = 0
                minimum_height = math.inf
                maximum_pitch_rate = 0.0
                maximum_temperature = 25.0
                minimum_buffer = 60.0
                maximum_chassis_power = 0.0
                disabled_samples = 0
                for step in range(num_steps):
                    action = actor(observation.to(args.device))
                    if not bool(torch.isfinite(action).all()):
                        raise RuntimeError("policy produced a non-finite action")
                    observation, _, _, _, _ = env.step(action.cpu())
                    terminal = env.last_terminal.copy()
                    falls += int(np.count_nonzero(terminal))
                    reset_seen |= terminal
                    diagnostics = env.diagnostics()
                    minimum_height = min(
                        minimum_height,
                        float(diagnostics["base_position"][:, 2].min()),
                    )
                    maximum_pitch_rate = max(
                        maximum_pitch_rate,
                        float(
                            np.abs(
                                diagnostics["base_angular_velocity_body"][:, 1]
                            ).max()
                        ),
                    )
                    maximum_temperature = max(
                        maximum_temperature,
                        float(diagnostics["motor_temperature_c"].max()),
                    )
                    minimum_buffer = min(
                        minimum_buffer,
                        float(diagnostics["buffer_energy_j"].min()),
                    )
                    maximum_chassis_power = max(
                        maximum_chassis_power,
                        float(diagnostics["chassis_power_w"].max()),
                    )
                    disabled_samples += int(
                        np.count_nonzero(~diagnostics["chassis_enabled"])
                    )
                    if step >= settle_steps:
                        actual = np.stack(
                            (
                                diagnostics["base_linear_velocity_body"][:, 0],
                                diagnostics["base_linear_velocity_body"][:, 1],
                                diagnostics["base_angular_velocity_body"][:, 2],
                            ),
                            axis=1,
                        )
                        squared_error += np.square(
                            actual - np.asarray(command)
                        ).sum(axis=0)
                        upright_error_sum += float(
                            np.arccos(
                                np.clip(
                                    -diagnostics["projected_gravity"][:, 2],
                                    -1.0,
                                    1.0,
                                )
                            ).sum()
                        )
                        sample_count += args.num_envs
                rmse = np.sqrt(squared_error / max(1, sample_count))
                result[segment_name] = {
                    "command": list(command),
                    "duration_s": duration,
                    "vx_rmse": float(rmse[0]),
                    "vy_rmse": float(rmse[1]),
                    "yaw_rmse": float(rmse[2]),
                    "mean_upright_error_rad": upright_error_sum
                    / max(1, sample_count),
                    "maximum_abs_pitch_rate_radps": maximum_pitch_rate,
                    "minimum_base_height_m": minimum_height,
                    "maximum_motor_temperature_c": maximum_temperature,
                    "minimum_buffer_energy_j": minimum_buffer,
                    "maximum_chassis_power_w": maximum_chassis_power,
                    "disabled_chassis_samples": disabled_samples,
                    "falls": falls,
                }
    finally:
        env.close()

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
    if source_backend.startswith("MuJoCo"):
        transfer_kind = "native"
    elif source_backend.startswith("Isaac"):
        transfer_kind = "sim2sim_to_mujoco"
    else:
        transfer_kind = "unknown_source"
    return {
        "backend": "MuJoCo 3.10",
        "checkpoint": str(checkpoint.resolve()),
        "algorithm": loaded.algorithm,
        "checkpoint_format": loaded.checkpoint_format,
        "source_backend": source_backend,
        "native_or_sim2sim": transfer_kind,
        "contract_sha256": contract_sha256(),
        "num_envs": args.num_envs,
        "seed": args.seed,
        "randomized_command_delay_ms": [
            0,
            args.maximum_command_delay_steps * 1000 * SIM_DT,
        ],
        "segments": result,
        "falls_total": sum(segment["falls"] for segment in result.values()),
        "envs_with_falls": int(np.count_nonzero(reset_seen)),
        "clean_env_fraction": float(np.mean(~reset_seen)),
        "mean_primary_axis_rmse": float(np.mean(primary_errors)),
    }


def main() -> None:
    args = parse_args()
    if min(args.num_envs, args.num_threads) <= 0:
        raise ValueError("num-envs and num-threads must be positive")
    if args.maximum_command_delay_steps < 0 or args.duration_scale <= 0.0:
        raise ValueError("delay must be non-negative and duration scale positive")
    evaluations = [
        evaluate_checkpoint(args, checkpoint) for checkpoint in args.checkpoint
    ]
    ranking = sorted(
        evaluations,
        key=lambda item: (item["falls_total"], item["mean_primary_axis_rmse"]),
    )
    payload: dict | list = (
        evaluations[0]
        if len(evaluations) == 1
        else {
            "backend": "MuJoCo 3.10",
            "best_checkpoint": ranking[0]["checkpoint"],
            "ranking": [item["checkpoint"] for item in ranking],
            "results": evaluations,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
