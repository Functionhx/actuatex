#!/usr/bin/env python
"""Identify Sentinel's local action-level dynamics in Isaac Sim 6 / PhysX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--collection_steps", type=int, default=250)
parser.add_argument("--action_hold_steps", type=int, default=2)
parser.add_argument("--leg_action_std", type=float, default=0.025)
parser.add_argument("--wheel_action_std", type=float, default=0.050)
parser.add_argument("--ridge", type=float, default=1.0e-5)
parser.add_argument("--validation_fraction", type=float, default=0.25)
parser.add_argument("--stabilizer_checkpoint", type=Path)
parser.add_argument("--command_vx", type=float, default=0.0)
parser.add_argument("--command_yaw", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=1301)
parser.add_argument("--output_model", type=Path)
parser.add_argument("--output_report", type=Path)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import tinymal_lab  # noqa: E402,F401
from tasks.robomaster.contract import (  # noqa: E402
    ACTION_DIM,
    POLICY_DT,
    contract_sha256,
)
from tasks.robomaster.linear_control import (  # noqa: E402
    CONTROL_STATE_DIM,
    CONTROL_STATE_NAMES,
    controllability_rank,
    evaluate_affine_dynamics,
    fit_affine_dynamics,
    torch_control_state_from_observation,
)
from tasks.robomaster.locomotion import COMMAND_OBSERVATION_SLICE  # noqa: E402
from tasks.robomaster.policy import load_policy  # noqa: E402
from tinymal_lab.sentinel_env_cfg import SentinelFlatEnvCfg  # noqa: E402


BACKEND = "Isaac Sim 6.0.1 GA / Isaac Lab 3.0.0-beta2.patch1 / PhysX 5"


def validate_args() -> None:
    positive = (
        args_cli.num_envs,
        args_cli.warmup_steps,
        args_cli.collection_steps,
        args_cli.action_hold_steps,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("environment and rollout sizes must be positive")
    if args_cli.num_envs < 4:
        raise ValueError("at least four environments are required for validation")
    if min(args_cli.leg_action_std, args_cli.wheel_action_std) <= 0.0:
        raise ValueError("exploration standard deviations must be positive")
    if args_cli.ridge < 0.0 or not 0.0 < args_cli.validation_fraction < 0.5:
        raise ValueError("ridge and validation fraction are invalid")
    if not np.isfinite([args_cli.command_vx, args_cli.command_yaw]).all():
        raise ValueError("operating command must be finite")
    if abs(args_cli.command_vx) > 1.5 or abs(args_cli.command_yaw) > 2.0:
        raise ValueError("operating command is outside the identification envelope")


def make_config() -> SentinelFlatEnvCfg:
    config = SentinelFlatEnvCfg()
    config.seed = args_cli.seed
    config.scene.num_envs = args_cli.num_envs
    config.episode_length_s = (
        args_cli.warmup_steps + args_cli.collection_steps + 100
    ) * POLICY_DT
    config.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    config.commands.base_velocity.rel_standing_envs = 0.0
    config.observations.policy.enable_corruption = False
    config.events.physics_material = None
    config.events.base_mass = None
    config.events.base_com = None
    config.events.actuator_gains = None
    config.events.push_robot = None
    config.events.reset_base.params["pose_range"] = {
        axis: (0.0, 0.0) for axis in ("x", "y", "roll", "pitch", "yaw")
    }
    config.events.reset_base.params["velocity_range"] = {
        axis: (0.0, 0.0)
        for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    config.events.reset_joints.params["position_range"] = (0.0, 0.0)
    config.events.reset_joints.params["velocity_range"] = (0.0, 0.0)
    return config


def set_operating_command(
    env,
    observation: dict[str, torch.Tensor],
) -> None:
    command = env.unwrapped.command_manager.get_command("base_velocity")
    command[:, 0] = args_cli.command_vx
    command[:, 1] = 0.0
    command[:, 2] = args_cli.command_yaw
    observation["policy"][:, COMMAND_OBSERVATION_SLICE] = command


def _number_tag(value: float) -> str:
    return f"{value:+.3f}".rstrip("0").rstrip(".").replace("+", "p").replace(
        "-", "m"
    ).replace(".", "p")


def main() -> None:
    validate_args()
    zero_operating_point = (
        args_cli.command_vx == 0.0 and args_cli.command_yaw == 0.0
    )
    operating_suffix = (
        ""
        if zero_operating_point
        else (
            f"_vx{_number_tag(args_cli.command_vx)}"
            f"_yaw{_number_tag(args_cli.command_yaw)}"
        )
    )
    model_path = args_cli.output_model or (
        REPO_ROOT
        / "artifacts/sentinel/control"
        / f"isaac_local_model{operating_suffix}_seed{args_cli.seed}.npz"
    )
    report_path = args_cli.output_report or model_path.with_suffix(".json")
    env = gym.make("Isaac-Velocity-Flat-Sentinel-v0", cfg=make_config())
    try:
        device = env.unwrapped.device
        num_envs = env.unwrapped.num_envs
        zero_action = torch.zeros((num_envs, ACTION_DIM), device=device)
        stabilizer = None
        stabilizer_sha256 = None
        if args_cli.stabilizer_checkpoint is not None:
            stabilizer_path = args_cli.stabilizer_checkpoint.resolve()
            if not stabilizer_path.is_file():
                raise FileNotFoundError(stabilizer_path)
            stabilizer = load_policy(stabilizer_path, device=device).actor
            stabilizer.eval()
            stabilizer_sha256 = hashlib.sha256(
                stabilizer_path.read_bytes()
            ).hexdigest()

        def baseline_action(observation: torch.Tensor) -> torch.Tensor:
            if stabilizer is None:
                return zero_action
            with torch.inference_mode():
                return stabilizer(observation).clamp(-1.0, 1.0)

        observation, _ = env.reset()
        set_operating_command(env, observation)
        warmup_falls = 0
        center_samples = []
        center_action_samples = []
        with torch.inference_mode():
            for step in range(args_cli.warmup_steps):
                set_operating_command(env, observation)
                action = baseline_action(observation["policy"])
                observation, _, terminated, truncated, _ = env.step(action)
                if bool(truncated.any()):
                    raise RuntimeError("unexpected timeout during identification warmup")
                warmup_falls += int(terminated.sum().item())
                set_operating_command(env, observation)
                if step >= args_cli.warmup_steps - min(
                    20, args_cli.warmup_steps
                ):
                    center_samples.append(
                        torch_control_state_from_observation(
                            observation["policy"]
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    center_action_samples.append(
                        action.detach().cpu().numpy()
                    )
        state_center = np.mean(np.stack(center_samples), axis=(0, 1))
        action_center = np.mean(
            np.stack(center_action_samples),
            axis=(0, 1),
        )

        generator = torch.Generator(device=device)
        generator.manual_seed(args_cli.seed + 1)
        action_scale = torch.tensor(
            [args_cli.leg_action_std] * 4
            + [args_cli.wheel_action_std] * 2,
            dtype=torch.float32,
            device=device,
        )
        action_limit = 4.0 * action_scale
        exploration = torch.zeros((num_envs, ACTION_DIM), device=device)
        state_steps = []
        action_steps = []
        next_state_steps = []
        valid_steps = []
        collection_falls = 0
        with torch.inference_mode():
            for step in range(args_cli.collection_steps):
                set_operating_command(env, observation)
                if step % args_cli.action_hold_steps == 0:
                    exploration = torch.randn(
                        (num_envs, ACTION_DIM),
                        generator=generator,
                        device=device,
                    ) * action_scale
                    exploration = torch.clamp(
                        exploration,
                        min=-action_limit,
                        max=action_limit,
                    )
                state = (
                    torch_control_state_from_observation(observation["policy"])
                    .detach()
                    .cpu()
                    .numpy()
                    - state_center
                )
                action = torch.clamp(
                    baseline_action(observation["policy"]) + exploration,
                    -1.0,
                    1.0,
                )
                next_observation, _, terminated, truncated, _ = env.step(action)
                set_operating_command(env, next_observation)
                next_state = (
                    torch_control_state_from_observation(
                        next_observation["policy"]
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    - state_center
                )
                collection_falls += int(terminated.sum().item())
                state_steps.append(state)
                action_steps.append(
                    action.detach().cpu().numpy() - action_center
                )
                next_state_steps.append(next_state)
                valid_steps.append(
                    (~(terminated | truncated)).detach().cpu().numpy()
                )
                observation = next_observation
    finally:
        env.close()

    state_tensor = np.stack(state_steps)
    action_tensor = np.stack(action_steps)
    next_state_tensor = np.stack(next_state_steps)
    valid_tensor = np.stack(valid_steps)
    validation_envs = max(1, round(num_envs * args_cli.validation_fraction))
    train_envs = num_envs - validation_envs

    def select(environment_slice: slice) -> tuple[np.ndarray, ...]:
        valid = valid_tensor[:, environment_slice].reshape(-1)
        return tuple(
            values[:, environment_slice].reshape(-1, values.shape[-1])[valid]
            for values in (state_tensor, action_tensor, next_state_tensor)
        )

    train_state, train_action, train_next_state = select(slice(0, train_envs))
    validation_state, validation_action, validation_next_state = select(
        slice(train_envs, num_envs)
    )
    fit = fit_affine_dynamics(
        train_state,
        train_action,
        train_next_state,
        ridge=args_cli.ridge,
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
        args_cli.collection_steps * num_envs
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
        operating_command=np.asarray(
            [args_cli.command_vx, 0.0, args_cli.command_yaw]
        ),
        control_state_names=np.asarray(CONTROL_STATE_NAMES),
    )
    report = {
        "backend": BACKEND,
        "contract_sha256": contract_sha256(),
        "seed": args_cli.seed,
        "policy_dt_s": POLICY_DT,
        "num_envs": num_envs,
        "train_envs": train_envs,
        "validation_envs": validation_envs,
        "warmup_steps": args_cli.warmup_steps,
        "collection_steps": args_cli.collection_steps,
        "action_hold_steps": args_cli.action_hold_steps,
        "leg_action_std": args_cli.leg_action_std,
        "wheel_action_std": args_cli.wheel_action_std,
        "ridge": args_cli.ridge,
        "operating_command": [
            args_cli.command_vx,
            0.0,
            args_cli.command_yaw,
        ],
        "stabilizer_checkpoint": (
            None
            if args_cli.stabilizer_checkpoint is None
            else str(args_cli.stabilizer_checkpoint.resolve())
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
    try:
        main()
    finally:
        simulation_app.close()
