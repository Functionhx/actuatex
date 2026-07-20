#!/usr/bin/env python3
"""Benchmark classical controllers on the shared MuJoCo cart-pole tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

from actuatex_paths import ARTIFACTS_ROOT, REPO_ROOT, RSL_RL_ROOT

if RSL_RL_ROOT.is_dir():
    sys.path.insert(0, str(RSL_RL_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from inverted_pendulum_env import MjSerialInvertedPendulumEnv  # noqa: E402
from tasks.inverted_pendulum.classical_control import (  # noqa: E402
    CascadedPIDController,
    LQGController,
    PIDController,
    StateFeedbackController,
    default_quadratic_cost,
    discrete_lqr_gain,
    finite_horizon_lqr_gain,
    pole_placement_gain,
    steady_state_kalman_gain,
)
from tasks.inverted_pendulum.contract import DECIMATION  # noqa: E402
from tasks.inverted_pendulum.state_estimators import (  # noqa: E402
    ComplementaryFilterLQRController,
    ExtendedKalmanLQRController,
    LinearObserverLQRController,
    design_luenberger_gain,
)
from tasks.inverted_pendulum.robust_control import (  # noqa: E402
    CollocatedFeedbackLinearizationController,
    DiscreteSlidingModeController,
    PartialFeedbackLinearizationController,
    h_infinity_state_feedback_gain,
)
from tasks.inverted_pendulum.contract import POLICY_DT  # noqa: E402


def linearize_policy_dynamics(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearize MuJoCo at upright and lift it to the 60 Hz policy step."""

    model_path = (
        REPO_ROOT
        / "robots"
        / "inverted_pendulum"
        / "mjcf"
        / f"actuatex_cartpole_{order}.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state_dim = 2 * model.nv + model.na
    one_step_a = np.zeros((state_dim, state_dim), dtype=np.float64)
    one_step_b = np.zeros((state_dim, model.nu), dtype=np.float64)
    mujoco.mjd_transitionFD(
        model,
        data,
        1.0e-6,
        True,
        one_step_a,
        one_step_b,
        None,
        None,
    )
    matrix_a = np.eye(state_dim)
    matrix_b = np.zeros_like(one_step_b)
    for _ in range(DECIMATION):
        matrix_b = one_step_a @ matrix_b + one_step_b
        matrix_a = one_step_a @ matrix_a
    return matrix_a, matrix_b


def read_state(env: MjSerialInvertedPendulumEnv) -> np.ndarray:
    cart_position, cart_velocity, pole_angles, pole_velocities = env._state()
    return np.concatenate(
        (
            cart_position[:, None],
            pole_angles,
            cart_velocity[:, None],
            pole_velocities,
        ),
        axis=1,
    )


def evaluate_controller(
    env: MjSerialInvertedPendulumEnv,
    controller,
    *,
    seed: int,
    measurement_noise_std: float = 0.0,
) -> dict:
    env.rng = np.random.default_rng(seed)
    env.reset()
    controller.reset(env.num_envs)
    measurement_rng = np.random.default_rng(seed + 100_000)
    active = np.ones(env.num_envs, dtype=bool)
    success = np.zeros(env.num_envs, dtype=bool)
    duration_steps = np.zeros(env.num_envs, dtype=np.int64)
    angle_squared_sum = np.zeros(env.num_envs, dtype=np.float64)
    cart_squared_sum = np.zeros(env.num_envs, dtype=np.float64)
    action_sum = np.zeros(env.num_envs, dtype=np.float64)
    sample_count = np.zeros(env.num_envs, dtype=np.int64)

    for step in range(env.max_episode_length):
        state = read_state(env)
        if hasattr(controller, "act_from_measurement"):
            measurement = state @ controller.measurement_c.T
            if measurement_noise_std:
                measurement += measurement_rng.normal(
                    0.0, measurement_noise_std, measurement.shape
                )
            action = controller.act_from_measurement(measurement)
        else:
            action = controller.act(state)
        env.step(torch.from_numpy(action.astype(np.float32))[:, None])

        ids = np.flatnonzero(active)
        angle_squared_sum[ids] += np.mean(
            np.square(env.last_absolute_angles[ids]), axis=1
        )
        cart_squared_sum[ids] += np.square(env.last_cart_position[ids])
        action_sum[ids] += np.abs(action[ids])
        sample_count[ids] += 1
        finished = active & (env.last_terminal | env.last_timeout)
        success[finished] = env.last_timeout[finished] & ~env.last_terminal[finished]
        duration_steps[finished] = step + 1
        active[finished] = False
        if not active.any():
            break

    duration_steps[active] = env.max_episode_length
    counts = np.maximum(sample_count, 1)
    return {
        "episodes": env.num_envs,
        "successes": int(success.sum()),
        "success_rate": float(success.mean()),
        "mean_balance_duration_s": float(duration_steps.mean() * env.dt),
        "median_balance_duration_s": float(np.median(duration_steps) * env.dt),
        "absolute_pole_angle_rmse_rad": float(
            np.sqrt(np.mean(angle_squared_sum / counts))
        ),
        "cart_position_rmse_m": float(np.sqrt(np.mean(cart_squared_sum / counts))),
        "mean_abs_normalized_action": float(np.mean(action_sum / counts)),
        "measurement_noise_std": measurement_noise_std,
        "seed": seed,
    }


def load_gain(checkpoint: Path) -> np.ndarray:
    payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
    if "lqr_gain" not in payload:
        raise KeyError(f"{checkpoint} has no lqr_gain")
    return payload["lqr_gain"].detach().cpu().numpy().astype(np.float64)


def build_controllers(
    order: int,
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    *,
    cem_checkpoint: Path | None,
    measurement_noise_std: float,
) -> tuple[dict[str, object], dict]:
    cost_q, cost_r = default_quadratic_cost(order)
    lqr_gain = discrete_lqr_gain(matrix_a, matrix_b, cost_q, cost_r)
    placed_gain, desired_poles = pole_placement_gain(matrix_a, matrix_b)
    controllers: dict[str, object] = {
        "pole_placement": StateFeedbackController(placed_gain),
        "lqr": StateFeedbackController(lqr_gain),
    }
    h_infinity_gain, attenuation_gamma, h_infinity_poles = (
        h_infinity_state_feedback_gain(
            matrix_a,
            matrix_b,
            cost_q,
            cost_r,
            sample_time=POLICY_DT,
        )
    )
    controllers["h_infinity"] = StateFeedbackController(h_infinity_gain)
    controllers["sliding_mode"] = DiscreteSlidingModeController(
        matrix_a,
        matrix_b,
        lqr_gain,
    )
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
        controllers[f"linear_mpc_h{horizon}"] = StateFeedbackController(gain)
        mpc_gains[str(horizon)] = gain.tolist()
    metadata = {
        "lqr_gain": lqr_gain.tolist(),
        "linear_mpc_first_gains": mpc_gains,
        "pole_placement_gain": placed_gain.tolist(),
        "desired_discrete_poles": desired_poles.tolist(),
        "h_infinity_gain": h_infinity_gain.tolist(),
        "h_infinity_gamma": attenuation_gamma,
        "h_infinity_closed_loop_poles": [
            [float(value.real), float(value.imag)] for value in h_infinity_poles
        ],
        "sliding_surface": lqr_gain.tolist(),
    }

    dof_count = order + 1
    measurement_c = np.zeros((dof_count, 2 * dof_count), dtype=np.float64)
    measurement_c[:, :dof_count] = np.eye(dof_count)
    process_covariance = np.eye(2 * dof_count) * 1.0e-5
    measurement_covariance = np.eye(dof_count) * max(measurement_noise_std, 1.0e-6) ** 2
    kalman_gain = steady_state_kalman_gain(
        matrix_a,
        measurement_c,
        process_covariance,
        measurement_covariance,
    )
    controllers["lqg"] = LQGController(
        matrix_a,
        matrix_b,
        measurement_c,
        lqr_gain,
        kalman_gain,
    )
    metadata["lqg_measurement"] = "cart and relative joint positions"
    metadata["kalman_gain"] = kalman_gain.tolist()

    observer_gain, observer_poles = design_luenberger_gain(
        matrix_a,
        measurement_c,
        slowest_rate=3.0,
        fastest_rate=18.0,
    )
    controllers["state_observer_lqr"] = LinearObserverLQRController(
        matrix_a,
        matrix_b,
        measurement_c,
        lqr_gain,
        observer_gain,
    )
    high_gain, high_gain_poles = design_luenberger_gain(
        matrix_a,
        measurement_c,
        slowest_rate=12.0,
        fastest_rate=70.0,
    )
    controllers["high_gain_observer_lqr"] = LinearObserverLQRController(
        matrix_a,
        matrix_b,
        measurement_c,
        lqr_gain,
        high_gain,
    )
    controllers["complementary_filter_lqr"] = ComplementaryFilterLQRController(
        matrix_a,
        matrix_b,
        measurement_c,
        lqr_gain,
    )
    metadata["state_observer_gain"] = observer_gain.tolist()
    metadata["state_observer_poles"] = observer_poles.tolist()
    metadata["high_gain_observer_gain"] = high_gain.tolist()
    metadata["high_gain_observer_poles"] = high_gain_poles.tolist()
    metadata["complementary_filter_model_velocity_weight"] = 0.94

    if order == 1:
        controllers["ekf_lqr"] = ExtendedKalmanLQRController(
            lqr_gain,
            measurement_noise_std=max(measurement_noise_std, 1.0e-6),
        )
        metadata["ekf_model"] = "nonlinear physical cart-pole with RK4 prediction"

    if order == 1:
        upright_input_to_acceleration = float(matrix_b[2, 0] / POLICY_DT)
        controllers["feedback_linearization"] = (
            CollocatedFeedbackLinearizationController(
                lqr_gain[0] * upright_input_to_acceleration
            )
        )
        controllers["partial_feedback_linearization"] = (
            PartialFeedbackLinearizationController()
        )
        metadata["feedback_linearization_output"] = "cart acceleration"
        metadata["partial_feedback_linearization_output"] = (
            "pole angular acceleration with cart-to-angle outer loop"
        )
        force_coefficients = -lqr_gain[0]
        controllers["pid"] = PIDController(
            cart_kp=float(force_coefficients[0]),
            cart_kd=float(force_coefficients[2]),
            angle_kp=float(force_coefficients[1]),
            angle_ki=0.10,
            angle_kd=float(force_coefficients[3]),
        )
        inner_kp = float(force_coefficients[1])
        controllers["cascaded_pid"] = CascadedPIDController(
            outer_kp=float(force_coefficients[0] / inner_kp),
            outer_kd=float(force_coefficients[2] / inner_kp),
            inner_kp=inner_kp,
            inner_ki=0.05,
            inner_kd=float(force_coefficients[3]),
        )
        metadata["pid_tuning"] = "LQR-informed gains with a bounded integral term"

    if order == 3 and cem_checkpoint is not None:
        controllers["physx_cem_lqr_sim2sim"] = StateFeedbackController(
            load_gain(cem_checkpoint)
        )
        metadata["physx_cem_checkpoint"] = str(cem_checkpoint.resolve())
    return controllers, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orders", type=int, nargs="+", choices=(1, 2, 3), default=(1, 2, 3)
    )
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--measurement_noise_std", type=float, default=0.002)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="optional controller names to evaluate; unavailable order-specific names are skipped",
    )
    parser.add_argument(
        "--cem_checkpoint",
        type=Path,
        default=ARTIFACTS_ROOT
        / "checkpoints"
        / "inverted_pendulum"
        / "isaac_cem_lqr_seed_order_3.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ARTIFACTS_ROOT
        / "inverted_pendulum"
        / "evaluation"
        / "classical_control_suite_mujoco.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.num_threads <= 0:
        raise ValueError("episode and thread counts must be positive")
    if args.measurement_noise_std < 0.0:
        raise ValueError("measurement noise must be non-negative")
    cem_checkpoint = args.cem_checkpoint if args.cem_checkpoint.is_file() else None
    order_results = []

    for order in args.orders:
        matrix_a, matrix_b = linearize_policy_dynamics(order)
        controllers, metadata = build_controllers(
            order,
            matrix_a,
            matrix_b,
            cem_checkpoint=cem_checkpoint,
            measurement_noise_std=args.measurement_noise_std,
        )
        if args.methods is not None:
            controllers = {
                name: controller
                for name, controller in controllers.items()
                if name in args.methods
            }
            if not controllers:
                raise ValueError(
                    f"none of --methods {args.methods} apply to order {order}"
                )
        env = MjSerialInvertedPendulumEnv(
            order,
            num_envs=args.episodes,
            num_threads=args.num_threads,
            seed=args.seed + order,
        )
        results = {}
        for name, controller in controllers.items():
            result = evaluate_controller(
                env,
                controller,
                seed=args.seed + 100 * order,
                measurement_noise_std=(
                    args.measurement_noise_std
                    if hasattr(controller, "act_from_measurement")
                    else 0.0
                ),
            )
            results[name] = result
            print(
                f"order={order} controller={name} "
                f"success={result['success_rate']:.3f} "
                f"duration={result['mean_balance_duration_s']:.3f}s"
            )
        env.close()
        order_results.append(
            {
                "order": order,
                "linear_model_a": matrix_a.tolist(),
                "linear_model_b": matrix_b.tolist(),
                "controller_metadata": metadata,
                "results": results,
            }
        )

    output = {
        "schema_version": 1,
        "backend": f"MuJoCo {mujoco.__version__}",
        "benchmark": "upright balance, shared ActuateX 10-second contract",
        "orders": order_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
