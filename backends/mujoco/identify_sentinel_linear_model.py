#!/usr/bin/env python3
"""Identify Sentinel's local action-level dynamics in MuJoCo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from actuatex_paths import ARTIFACTS_ROOT, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))

from sentinel_env import MjSentinelEnv  # noqa: E402
from tasks.robomaster.contract import (  # noqa: E402
    ACTION_DIM,
    POLICY_DT,
    contract_sha256,
)
from tasks.robomaster.linear_control import (  # noqa: E402
    CONTROL_STATE_DIM,
    CONTROL_STATE_NAMES,
    control_state_from_observation,
    controllability_rank,
    evaluate_affine_dynamics,
    fit_affine_dynamics,
)
from tasks.robomaster.policy import load_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=75)
    parser.add_argument("--collection-steps", type=int, default=400)
    parser.add_argument("--action-hold-steps", type=int, default=2)
    parser.add_argument("--leg-action-std", type=float, default=0.025)
    parser.add_argument("--wheel-action-std", type=float, default=0.050)
    parser.add_argument("--ridge", type=float, default=1.0e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--stabilizer-checkpoint", type=Path)
    parser.add_argument("--command-vx", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--output-model", type=Path)
    parser.add_argument("--output-report", type=Path)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.num_envs,
        args.num_threads,
        args.warmup_steps,
        args.collection_steps,
        args.action_hold_steps,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("environment and rollout sizes must be positive")
    if args.num_envs < 4:
        raise ValueError("at least four environments are required for split validation")
    if min(args.leg_action_std, args.wheel_action_std) <= 0.0:
        raise ValueError("exploration standard deviations must be positive")
    if args.ridge < 0.0 or not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("ridge and validation fraction are invalid")
    if not np.isfinite([args.command_vx, args.command_yaw]).all():
        raise ValueError("operating command must be finite")
    if abs(args.command_vx) > 1.5 or abs(args.command_yaw) > 2.0:
        raise ValueError("operating command is outside the identification envelope")


def _number_tag(value: float) -> str:
    return f"{value:+.3f}".rstrip("0").rstrip(".").replace("+", "p").replace(
        "-", "m"
    ).replace(".", "p")


def _exploration_action(
    rng: np.random.Generator,
    num_envs: int,
    leg_std: float,
    wheel_std: float,
) -> np.ndarray:
    scale = np.asarray([leg_std] * 4 + [wheel_std] * 2)
    action = rng.normal(0.0, scale, (num_envs, ACTION_DIM))
    limit = np.asarray([4.0 * leg_std] * 4 + [4.0 * wheel_std] * 2)
    return np.clip(action, -limit, limit)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    rng = np.random.default_rng(args.seed)
    zero_operating_point = args.command_vx == 0.0 and args.command_yaw == 0.0
    operating_suffix = (
        ""
        if zero_operating_point
        else (
            f"_vx{_number_tag(args.command_vx)}"
            f"_yaw{_number_tag(args.command_yaw)}"
        )
    )
    model_path = args.output_model or (
        ARTIFACTS_ROOT
        / "sentinel/control"
        / f"mujoco_local_model{operating_suffix}_seed{args.seed}.npz"
    )
    report_path = args.output_report or model_path.with_suffix(".json")

    env = MjSentinelEnv(
        num_envs=args.num_envs,
        num_threads=args.num_threads,
        seed=args.seed,
        episode_length_s=(
            args.warmup_steps + args.collection_steps + 100
        )
        * POLICY_DT,
        add_noise=False,
        randomize_reset=False,
        maximum_command_delay_steps=0,
    )
    zero_action = torch.zeros((args.num_envs, ACTION_DIM))
    operating_command = np.asarray(
        [args.command_vx, 0.0, args.command_yaw], dtype=np.float64
    )
    env.set_command(operating_command)
    observation, _ = env.reset()
    stabilizer = None
    stabilizer_sha256 = None
    if args.stabilizer_checkpoint is not None:
        stabilizer_path = args.stabilizer_checkpoint.resolve()
        if not stabilizer_path.is_file():
            raise FileNotFoundError(stabilizer_path)
        stabilizer = load_policy(stabilizer_path, device="cpu").actor
        stabilizer.eval()
        stabilizer_sha256 = hashlib.sha256(
            stabilizer_path.read_bytes()
        ).hexdigest()

    def baseline_action(current_observation: torch.Tensor) -> torch.Tensor:
        if stabilizer is None:
            return zero_action
        with torch.inference_mode():
            return stabilizer(current_observation).clamp(-1.0, 1.0)

    warmup_falls = 0
    try:
        center_samples = []
        center_action_samples = []
        for step in range(args.warmup_steps):
            action = baseline_action(observation)
            observation, _, _, _, _ = env.step(action)
            warmup_falls += int(np.count_nonzero(env.last_terminal))
            if step >= args.warmup_steps - min(20, args.warmup_steps):
                center_samples.append(
                    control_state_from_observation(observation.numpy())
                )
                center_action_samples.append(action.numpy())
        state_center = np.mean(np.stack(center_samples), axis=(0, 1))
        action_center = np.mean(
            np.stack(center_action_samples),
            axis=(0, 1),
        )

        state_steps = []
        action_steps = []
        next_state_steps = []
        valid_steps = []
        exploration = np.zeros((args.num_envs, ACTION_DIM))
        collection_falls = 0
        for step in range(args.collection_steps):
            if step % args.action_hold_steps == 0:
                exploration = _exploration_action(
                    rng,
                    args.num_envs,
                    args.leg_action_std,
                    args.wheel_action_std,
                )
            state = (
                control_state_from_observation(observation.numpy())
                - state_center
            )
            current_action = np.clip(
                baseline_action(observation).numpy() + exploration,
                -1.0,
                1.0,
            )
            next_observation, _, _, _, _ = env.step(
                torch.from_numpy(current_action).float()
            )
            next_state = (
                control_state_from_observation(next_observation.numpy())
                - state_center
            )
            terminal = env.last_terminal.copy()
            timeout = env.last_timeout.copy()
            collection_falls += int(np.count_nonzero(terminal))
            state_steps.append(state)
            action_steps.append(current_action - action_center)
            next_state_steps.append(next_state)
            valid_steps.append(~(terminal | timeout))
            observation = next_observation
    finally:
        env.close()

    state_tensor = np.stack(state_steps)
    action_tensor = np.stack(action_steps)
    next_state_tensor = np.stack(next_state_steps)
    valid_tensor = np.stack(valid_steps)
    validation_envs = max(1, round(args.num_envs * args.validation_fraction))
    train_envs = args.num_envs - validation_envs

    def select(environment_slice: slice) -> tuple[np.ndarray, ...]:
        valid = valid_tensor[:, environment_slice].reshape(-1)
        return tuple(
            values[:, environment_slice].reshape(-1, values.shape[-1])[valid]
            for values in (state_tensor, action_tensor, next_state_tensor)
        )

    train_state, train_action, train_next_state = select(slice(0, train_envs))
    validation_state, validation_action, validation_next_state = select(
        slice(train_envs, args.num_envs)
    )
    fit = fit_affine_dynamics(
        train_state,
        train_action,
        train_next_state,
        ridge=args.ridge,
    )
    validation_metrics = evaluate_affine_dynamics(
        fit,
        validation_state,
        validation_action,
        validation_next_state,
    )
    control_rank = controllability_rank(fit.matrix_a, fit.matrix_b)
    open_loop_spectral_radius = float(max(abs(np.linalg.eigvals(fit.matrix_a))))
    required_feature_rank = CONTROL_STATE_DIM + ACTION_DIM + 1
    equilibrium_gate = bool(
        warmup_falls == 0
        and abs(state_center[0]) <= 0.15
        and np.linalg.norm(state_center[1:3]) <= 0.15
        and np.linalg.norm(state_center[3:6]) <= 0.40
        and np.linalg.norm(state_center[6:8]) <= 0.15
    )
    collection_fall_fraction = collection_falls / (
        args.collection_steps * args.num_envs
    )
    quality_gate = bool(
        equilibrium_gate
        and collection_fall_fraction <= 0.01
        and fit.metrics.feature_rank == required_feature_rank
        and validation_metrics.coefficient_of_determination >= 0.90
        and validation_metrics.normalized_rmse <= 0.35
        and np.isfinite(open_loop_spectral_radius)
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        matrix_a=fit.matrix_a,
        matrix_b=fit.matrix_b,
        bias=fit.bias,
        state_center=state_center,
        action_center=action_center,
        operating_command=operating_command,
        control_state_names=np.asarray(CONTROL_STATE_NAMES),
    )
    report = {
        "backend": "MuJoCo 3.10",
        "contract_sha256": contract_sha256(),
        "seed": args.seed,
        "policy_dt_s": POLICY_DT,
        "num_envs": args.num_envs,
        "train_envs": train_envs,
        "validation_envs": validation_envs,
        "warmup_steps": args.warmup_steps,
        "collection_steps": args.collection_steps,
        "action_hold_steps": args.action_hold_steps,
        "leg_action_std": args.leg_action_std,
        "wheel_action_std": args.wheel_action_std,
        "ridge": args.ridge,
        "operating_command": operating_command.tolist(),
        "stabilizer_checkpoint": (
            None
            if args.stabilizer_checkpoint is None
            else str(args.stabilizer_checkpoint.resolve())
        ),
        "stabilizer_sha256": stabilizer_sha256,
        "warmup_falls": warmup_falls,
        "collection_falls": collection_falls,
        "collection_fall_fraction": collection_fall_fraction,
        "equilibrium_gate_passed": equilibrium_gate,
        "train_metrics": fit.metrics.as_dict(),
        "validation_metrics": validation_metrics.as_dict(),
        "controllability_rank": control_rank,
        "open_loop_spectral_radius": open_loop_spectral_radius,
        "required_feature_rank": required_feature_rank,
        "quality_gate_passed": quality_gate,
        "state_center": state_center.tolist(),
        "action_center": action_center.tolist(),
        "control_state_names": list(CONTROL_STATE_NAMES),
        "model_path": str(model_path.resolve()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not quality_gate:
        raise RuntimeError(
            "local dynamics quality gate failed; inspect the saved report before LQR"
        )


if __name__ == "__main__":
    main()
