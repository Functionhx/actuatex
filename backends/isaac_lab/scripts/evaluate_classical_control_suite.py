#!/usr/bin/env python
"""Benchmark the classical controller suite directly in Isaac Lab / PhysX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--orders", type=int, nargs="+", choices=(1, 2, 3), default=(1, 2, 3)
)
parser.add_argument("--episodes", type=int, default=1024)
parser.add_argument("--seed", type=int, default=701)
parser.add_argument("--measurement_noise_std", type=float, default=0.002)
parser.add_argument(
    "--methods",
    nargs="+",
    default=None,
    help="optional controller names; order-specific unavailable names are skipped",
)
parser.add_argument(
    "--identification_dir",
    type=Path,
    default=REPO_ROOT / "artifacts" / "inverted_pendulum" / "evaluation",
)
parser.add_argument(
    "--cem_checkpoint",
    type=Path,
    default=REPO_ROOT
    / "artifacts"
    / "checkpoints"
    / "inverted_pendulum"
    / "isaac_cem_lqr_seed_order_3.pt",
)
parser.add_argument(
    "--output",
    type=Path,
    default=REPO_ROOT
    / "artifacts"
    / "inverted_pendulum"
    / "evaluation"
    / "classical_control_suite_isaac.json",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
if remaining_args:
    parser.error(f"unrecognized arguments: {' '.join(remaining_args)}")
args_cli.headless = True
if args_cli.episodes <= 0:
    parser.error("--episodes must be positive")
if args_cli.measurement_noise_std < 0.0:
    parser.error("--measurement_noise_std must be non-negative")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import tinymal_lab  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from tasks.inverted_pendulum.classical_control import (  # noqa: E402
    default_quadratic_cost,
    discrete_lqr_gain,
    finite_horizon_lqr_gain,
    pole_placement_gain,
    steady_state_kalman_gain,
)
from tasks.inverted_pendulum.contract import (  # noqa: E402
    INITIAL_ANGLE_RANGE_RAD,
    POLICY_DT,
)
from tasks.inverted_pendulum.robust_control import (  # noqa: E402
    h_infinity_state_feedback_gain,
)
from tasks.inverted_pendulum.state_estimators import (  # noqa: E402
    design_luenberger_gain,
)
from tasks.inverted_pendulum.torch_control import (  # noqa: E402
    TorchCascadedPIDController,
    TorchCollocatedFeedbackLinearizationController,
    TorchComplementaryFilterLQRController,
    TorchDiscreteSlidingModeController,
    TorchExtendedKalmanLQRController,
    TorchLinearOutputFeedbackController,
    TorchPIDController,
    TorchPartialFeedbackLinearizationController,
    TorchStateFeedbackController,
)


def _load_identification(order: int) -> tuple[np.ndarray, np.ndarray, Path]:
    path = (
        args_cli.identification_dir
        / f"isaac_lqr_identification_order_{order}.json"
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"missing PhysX identification {path}; run "
            "backends/isaac_lab/scripts/design_inverted_pendulum_lqr.py first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["order"]) != order:
        raise ValueError(f"identification order mismatch in {path}")
    return (
        np.asarray(payload["matrix_a"], dtype=np.float64),
        np.asarray(payload["matrix_b"], dtype=np.float64),
        path,
    )


def _paired_initial_state(order: int, episodes: int, seed: int) -> np.ndarray:
    """Match the per-environment MuJoCo reset draw order exactly."""

    rng = np.random.default_rng(seed)
    dof_count = order + 1
    state = np.zeros((episodes, 2 * dof_count), dtype=np.float32)
    angle_range = INITIAL_ANGLE_RANGE_RAD[order]
    for env_id in range(episodes):
        state[env_id, 0] = rng.uniform(-0.25, 0.25)
        state[env_id, dof_count] = rng.uniform(-0.10, 0.10)
        state[env_id, 1:dof_count] = rng.uniform(
            -angle_range, angle_range, order
        )
        state[env_id, dof_count + 1 :] = rng.uniform(-0.10, 0.10, order)
    return state


def _read_state(env) -> torch.Tensor:
    env.joint_pos = env.robot.data.joint_pos.torch
    env.joint_vel = env.robot.data.joint_vel.torch
    qpos = torch.cat(
        (
            env.joint_pos[:, env._cart_dof_idx],
            env.joint_pos[:, env._pole_dof_idx],
        ),
        dim=-1,
    )
    qvel = torch.cat(
        (
            env.joint_vel[:, env._cart_dof_idx],
            env.joint_vel[:, env._pole_dof_idx],
        ),
        dim=-1,
    )
    return torch.cat((qpos, qvel), dim=-1)


def _write_state(env, state: torch.Tensor) -> None:
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    dof_count = env.cfg.order + 1
    root_pose = env.robot.data.default_root_pose.torch[env_ids].clone()
    root_pose[:, :3] += env.scene.env_origins[env_ids]
    root_velocity = env.robot.data.default_root_vel.torch[env_ids].clone()
    env.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    env.robot.write_root_velocity_to_sim_index(
        root_velocity=root_velocity, env_ids=env_ids
    )
    env.robot.write_joint_position_to_sim_index(
        position=state[:, :dof_count], env_ids=env_ids
    )
    env.robot.write_joint_velocity_to_sim_index(
        velocity=state[:, dof_count:], env_ids=env_ids
    )
    env.episode_length_buf.zero_()
    env.actions.zero_()
    env.previous_actions.zero_()


def _build_controllers(
    order: int,
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, object], dict]:
    cost_q, cost_r = default_quadratic_cost(order)
    lqr_gain = discrete_lqr_gain(matrix_a, matrix_b, cost_q, cost_r)
    placed_gain, desired_poles = pole_placement_gain(matrix_a, matrix_b)
    h_infinity_gain, attenuation_gamma, h_infinity_poles = (
        h_infinity_state_feedback_gain(
            matrix_a,
            matrix_b,
            cost_q,
            cost_r,
            sample_time=POLICY_DT,
        )
    )
    controllers: dict[str, object] = {
        "pole_placement": TorchStateFeedbackController(
            placed_gain, device=device
        ),
        "lqr": TorchStateFeedbackController(lqr_gain, device=device),
        "h_infinity": TorchStateFeedbackController(
            h_infinity_gain, device=device
        ),
        "sliding_mode": TorchDiscreteSlidingModeController(
            matrix_a, matrix_b, lqr_gain, device=device
        ),
    }
    mpc_gains = {}
    for horizon in (40, 120, 240):
        gain = finite_horizon_lqr_gain(
            matrix_a,
            matrix_b,
            cost_q,
            cost_r,
            horizon=horizon,
            terminal_cost=cost_q,
        )
        controllers[f"linear_mpc_h{horizon}"] = TorchStateFeedbackController(
            gain, device=device
        )
        mpc_gains[str(horizon)] = gain.tolist()

    dof_count = order + 1
    measurement_c = np.zeros((dof_count, 2 * dof_count), dtype=np.float64)
    measurement_c[:, :dof_count] = np.eye(dof_count)
    process_covariance = np.eye(2 * dof_count) * 1.0e-5
    noise_scale = max(args_cli.measurement_noise_std, 1.0e-6)
    measurement_covariance = np.eye(dof_count) * noise_scale**2
    kalman_gain = steady_state_kalman_gain(
        matrix_a,
        measurement_c,
        process_covariance,
        measurement_covariance,
    )
    controllers["lqg"] = TorchLinearOutputFeedbackController(
        matrix_a,
        matrix_b,
        measurement_c,
        lqr_gain,
        kalman_gain,
        device=device,
    )
    observer_gain, observer_poles = design_luenberger_gain(
        matrix_a,
        measurement_c,
        slowest_rate=3.0,
        fastest_rate=18.0,
    )
    controllers["state_observer_lqr"] = TorchLinearOutputFeedbackController(
        matrix_a,
        matrix_b,
        measurement_c,
        lqr_gain,
        observer_gain,
        device=device,
    )
    high_gain, high_gain_poles = design_luenberger_gain(
        matrix_a,
        measurement_c,
        slowest_rate=12.0,
        fastest_rate=70.0,
    )
    controllers["high_gain_observer_lqr"] = (
        TorchLinearOutputFeedbackController(
            matrix_a,
            matrix_b,
            measurement_c,
            lqr_gain,
            high_gain,
            device=device,
        )
    )
    controllers["complementary_filter_lqr"] = (
        TorchComplementaryFilterLQRController(
            matrix_a,
            matrix_b,
            measurement_c,
            lqr_gain,
            device=device,
        )
    )

    metadata = {
        "lqr_gain": lqr_gain.tolist(),
        "pole_placement_gain": placed_gain.tolist(),
        "desired_discrete_poles": desired_poles.tolist(),
        "linear_mpc_first_gains": mpc_gains,
        "h_infinity_gain": h_infinity_gain.tolist(),
        "h_infinity_gamma": attenuation_gamma,
        "h_infinity_closed_loop_poles": [
            [float(value.real), float(value.imag)]
            for value in h_infinity_poles
        ],
        "sliding_surface": lqr_gain.tolist(),
        "kalman_gain": kalman_gain.tolist(),
        "state_observer_gain": observer_gain.tolist(),
        "state_observer_poles": observer_poles.tolist(),
        "high_gain_observer_gain": high_gain.tolist(),
        "high_gain_observer_poles": high_gain_poles.tolist(),
        "measurement": "cart and relative joint positions",
        "measurement_noise_std": args_cli.measurement_noise_std,
    }

    if order == 1:
        controllers["ekf_lqr"] = TorchExtendedKalmanLQRController(
            lqr_gain,
            measurement_noise_std=noise_scale,
            device=device,
        )
        upright_input_to_acceleration = float(matrix_b[2, 0] / POLICY_DT)
        controllers["feedback_linearization"] = (
            TorchCollocatedFeedbackLinearizationController(
                lqr_gain[0] * upright_input_to_acceleration,
                device=device,
            )
        )
        controllers["partial_feedback_linearization"] = (
            TorchPartialFeedbackLinearizationController(device=device)
        )
        force_coefficients = -lqr_gain[0]
        controllers["pid"] = TorchPIDController(
            cart_kp=float(force_coefficients[0]),
            cart_kd=float(force_coefficients[2]),
            angle_kp=float(force_coefficients[1]),
            angle_ki=0.10,
            angle_kd=float(force_coefficients[3]),
            device=device,
        )
        inner_kp = float(force_coefficients[1])
        controllers["cascaded_pid"] = TorchCascadedPIDController(
            outer_kp=float(force_coefficients[0] / inner_kp),
            outer_kd=float(force_coefficients[2] / inner_kp),
            inner_kp=inner_kp,
            inner_ki=0.05,
            inner_kd=float(force_coefficients[3]),
            device=device,
        )
        metadata["pid_tuning"] = "PhysX LQR-informed with bounded integral"

    if order == 3 and args_cli.cem_checkpoint.is_file():
        payload = torch.load(
            args_cli.cem_checkpoint, map_location="cpu", weights_only=False
        )
        cem_gain = payload["lqr_gain"].detach().cpu().numpy()
        controllers["physx_cem_lqr"] = TorchStateFeedbackController(
            cem_gain, device=device
        )
        metadata["physx_cem_lqr_gain"] = cem_gain.tolist()
        metadata["physx_cem_checkpoint"] = str(
            args_cli.cem_checkpoint.resolve()
        )
    return controllers, metadata


def _evaluate_controller(
    env,
    controller,
    initial_state: torch.Tensor,
    *,
    noise_seed: int,
) -> dict:
    env.reset()
    _write_state(env, initial_state)
    controller.reset(env.num_envs)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(active)
    duration_steps = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    angle_squared_sum = torch.zeros(
        env.num_envs, dtype=torch.float64, device=env.device
    )
    cart_squared_sum = torch.zeros_like(angle_squared_sum)
    action_sum = torch.zeros_like(angle_squared_sum)
    sample_count = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    noise_generator = torch.Generator(device=env.device)
    noise_generator.manual_seed(noise_seed)
    uses_measurement = hasattr(controller, "act_from_measurement")

    with torch.inference_mode():
        for step in range(env.max_episode_length):
            state = _read_state(env)
            if uses_measurement:
                measurement = state @ controller.measurement_c.T
                if args_cli.measurement_noise_std:
                    measurement = measurement + torch.randn(
                        measurement.shape,
                        generator=noise_generator,
                        device=env.device,
                        dtype=measurement.dtype,
                    ) * args_cli.measurement_noise_std
                action = controller.act_from_measurement(measurement)
            else:
                action = controller.act(state)
            action = torch.where(active, action, torch.zeros_like(action))
            active_ids = torch.nonzero(active, as_tuple=False).flatten()
            relative_angles = state[:, 1 : env.cfg.order + 1]
            absolute_angles = torch.cumsum(relative_angles, dim=-1)
            angle_squared_sum[active_ids] += torch.mean(
                torch.square(absolute_angles[active_ids]).double(), dim=1
            )
            cart_squared_sum[active_ids] += torch.square(
                state[active_ids, 0]
            ).double()
            action_sum[active_ids] += torch.abs(action[active_ids]).double()
            sample_count[active_ids] += 1
            _, _, terminated, truncated, _ = env.step(action[:, None])
            done = terminated | truncated
            newly_finished = active & done
            success[newly_finished] = truncated[newly_finished] & ~terminated[
                newly_finished
            ]
            duration_steps[newly_finished] = step + 1
            active &= ~done

    if bool(active.any().item()):
        raise RuntimeError("some PhysX evaluation episodes did not terminate")
    counts = torch.clamp(sample_count, min=1).double()
    durations = duration_steps.double() * env.step_dt
    result = {
        "episodes": env.num_envs,
        "successes": int(success.sum().item()),
        "success_rate": float(success.double().mean().item()),
        "mean_balance_duration_s": float(durations.mean().item()),
        "median_balance_duration_s": float(
            np.median(durations.detach().cpu().numpy())
        ),
        "absolute_pole_angle_rmse_rad": float(
            torch.sqrt(torch.mean(angle_squared_sum / counts)).item()
        ),
        "cart_position_rmse_m": float(
            torch.sqrt(torch.mean(cart_squared_sum / counts)).item()
        ),
        "mean_abs_normalized_action": float(
            torch.mean(action_sum / counts).item()
        ),
        "measurement_noise_std": (
            args_cli.measurement_noise_std if uses_measurement else 0.0
        ),
        "noise_seed": noise_seed,
    }
    return result


def main() -> None:
    device = torch.device(args_cli.device or "cuda:0")
    order_results = []
    for order in args_cli.orders:
        matrix_a, matrix_b, identification_path = _load_identification(order)
        controllers, metadata = _build_controllers(
            order, matrix_a, matrix_b, device
        )
        if args_cli.methods is not None:
            controllers = {
                name: controller
                for name, controller in controllers.items()
                if name in args_cli.methods
            }
            if not controllers:
                raise ValueError(
                    f"none of --methods {args_cli.methods} apply to order {order}"
                )

        task = f"Isaac-InvertedPendulum-{order}-Direct-v0"
        env_cfg = parse_env_cfg(
            task,
            device=str(device),
            num_envs=args_cli.episodes,
            use_fabric=True,
        )
        env_cfg.seed = args_cli.seed + order
        gym_env = gym.make(task, cfg=env_cfg)
        env = gym_env.unwrapped
        initial_seed = args_cli.seed + 100 * order
        initial_state = torch.from_numpy(
            _paired_initial_state(order, args_cli.episodes, initial_seed)
        ).to(device)
        results = {}
        for name, controller in controllers.items():
            result = _evaluate_controller(
                env,
                controller,
                initial_state,
                noise_seed=args_cli.seed + 100_000 + 100 * order,
            )
            results[name] = result
            print(
                f"order={order} controller={name} "
                f"success={result['success_rate']:.3f} "
                f"duration={result['mean_balance_duration_s']:.3f}s"
            )
        gym_env.close()
        order_results.append(
            {
                "order": order,
                "task": task,
                "initial_state_seed": initial_seed,
                "identification": str(identification_path),
                "linear_model_a": matrix_a.tolist(),
                "linear_model_b": matrix_b.tolist(),
                "controller_metadata": metadata,
                "results": results,
            }
        )

    output = {
        "schema_version": 1,
        "backend": "Isaac Sim 6.0.1 GA / Isaac Lab 3.0.0-beta2.patch1",
        "benchmark": "upright balance, shared ActuateX 10-second contract",
        "episodes_per_controller": args_cli.episodes,
        "paired_with_mujoco_reset_draw_order": True,
        "orders": order_results,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args_cli.output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
