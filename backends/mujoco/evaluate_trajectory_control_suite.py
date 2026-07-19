#!/usr/bin/env python3
"""Optimize a nonlinear swing-up trajectory and compare replay with TVLQR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

from actuatex_paths import ARTIFACTS_ROOT, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))

from evaluate_swingup_suite import (  # noqa: E402
    SwingupBatch,
    evaluate_controller,
    linearize_policy_dynamics,
)
from tasks.inverted_pendulum.classical_control import (  # noqa: E402
    default_quadratic_cost,
    discrete_lqr_gain,
)
from tasks.inverted_pendulum.swingup_control import (  # noqa: E402
    EnergySwingupController,
    HybridEnergyLQRController,
)
from tasks.inverted_pendulum.trajectory_optimization import (  # noqa: E402
    TVLQRTrackingController,
    TrajectoryReplayController,
    iterative_lqr_swingup,
    solve_swingup_multiple_shooting,
    tvlqr_gains,
)
from tasks.inverted_pendulum.state_estimators import (  # noqa: E402
    nonlinear_single_cartpole_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--duration_s", type=float, default=10.0)
    parser.add_argument("--stable_duration_s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=931)
    parser.add_argument(
        "--out",
        type=Path,
        default=ARTIFACTS_ROOT
        / "inverted_pendulum"
        / "evaluation"
        / "trajectory_control_suite_mujoco.json",
    )
    parser.add_argument(
        "--trajectory_out",
        type=Path,
        default=ARTIFACTS_ROOT
        / "inverted_pendulum"
        / "trajectories"
        / "single_pole_swingup_tvlqr.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon <= 0 or args.episodes <= 0 or args.num_threads <= 0:
        raise ValueError("horizon, episodes and num_threads must be positive")
    solve_start = time.perf_counter()
    trajectory = solve_swingup_multiple_shooting(
        np.array([0.0, np.pi, 0.0, 0.0]),
        horizon=args.horizon,
    )
    solve_time_s = time.perf_counter() - solve_start
    gains = tvlqr_gains(trajectory.states, trajectory.forces)

    matrix_a, matrix_b = linearize_policy_dynamics()
    cost_q, cost_r = default_quadratic_cost(1)
    terminal_gain = discrete_lqr_gain(matrix_a, matrix_b, cost_q, cost_r)
    initial_state = np.array([0.0, np.pi, 0.0, 0.0])
    warm_controller = HybridEnergyLQRController(
        terminal_gain,
        swingup=EnergySwingupController(cart_kp=1.0, cart_kd=1.5),
    )
    warm_controller.reset(1)
    warm_state = initial_state[None, :].copy()
    warm_forces = np.empty(args.horizon, dtype=np.float64)
    for step in range(args.horizon):
        warm_forces[step] = 20.0 * warm_controller.act(warm_state)[0]
        warm_state = nonlinear_single_cartpole_step(
            warm_state, warm_forces[step : step + 1]
        )
    ilqr_start = time.perf_counter()
    ilqr_trajectory = iterative_lqr_swingup(
        initial_state,
        warm_forces,
        max_iterations=100,
    )
    ilqr_solve_time_s = time.perf_counter() - ilqr_start
    ilqr_gains = tvlqr_gains(ilqr_trajectory.states, ilqr_trajectory.forces)
    controllers = {
        "open_loop_trajectory": TrajectoryReplayController(trajectory.forces),
        "trajectory_tvlqr": TVLQRTrackingController(
            trajectory.states,
            trajectory.forces,
            gains,
            terminal_gain,
        ),
        "ilqr_open_loop": TrajectoryReplayController(ilqr_trajectory.forces),
        "ilqr_tvlqr": TVLQRTrackingController(
            ilqr_trajectory.states,
            ilqr_trajectory.forces,
            ilqr_gains,
            terminal_gain,
        ),
        "hybrid_energy_lqr": HybridEnergyLQRController(
            terminal_gain,
            swingup=EnergySwingupController(cart_kp=1.0, cart_kd=1.5),
        ),
    }
    batch = SwingupBatch(args.episodes, args.num_threads)
    results = {}
    try:
        for name, controller in controllers.items():
            result = evaluate_controller(
                batch,
                controller,
                seed=args.seed,
                duration_s=args.duration_s,
                stable_duration_s=args.stable_duration_s,
            )
            results[name] = result
            print(
                f"controller={name} entry={result['upright_entry_rate']:.3f} "
                f"success={result['success_rate']:.3f} "
                f"rail={result['rail_violation_rate']:.3f}"
            )
    finally:
        batch.close()

    args.trajectory_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.trajectory_out,
        states=trajectory.states,
        forces=trajectory.forces,
        tvlqr_gains=gains,
        ilqr_states=ilqr_trajectory.states,
        ilqr_forces=ilqr_trajectory.forces,
        ilqr_tvlqr_gains=ilqr_gains,
        terminal_gain=terminal_gain,
    )
    output = {
        "schema_version": 1,
        "backend": f"MuJoCo {mujoco.__version__}",
        "benchmark": "single-pole nonlinear trajectory optimization and TVLQR",
        "optimizer": "CasADi multiple shooting with RK4 dynamics and IPOPT",
        "horizon_steps": args.horizon,
        "horizon_s": args.horizon / 60.0,
        "solver_status": trajectory.return_status,
        "solver_iterations": trajectory.solver_iterations,
        "solve_wall_time_s": solve_time_s,
        "objective": trajectory.objective,
        "terminal_state": trajectory.states[-1].tolist(),
        "maximum_abs_cart_position_m": float(np.max(np.abs(trajectory.states[:, 0]))),
        "maximum_abs_force_n": float(np.max(np.abs(trajectory.forces))),
        "ilqr": {
            "status": ilqr_trajectory.return_status,
            "iterations": ilqr_trajectory.solver_iterations,
            "solve_wall_time_s": ilqr_solve_time_s,
            "objective": ilqr_trajectory.objective,
            "terminal_state": ilqr_trajectory.states[-1].tolist(),
            "maximum_abs_cart_position_m": float(
                np.max(np.abs(ilqr_trajectory.states[:, 0]))
            ),
            "maximum_abs_force_n": float(np.max(np.abs(ilqr_trajectory.forces))),
            "initialization": "energy-shaping plus hysteretic LQR rollout",
        },
        "trajectory_artifact": str(args.trajectory_out.resolve()),
        "evaluation": {
            "episodes": args.episodes,
            "duration_s": args.duration_s,
            "stable_duration_s": args.stable_duration_s,
            "seed": args.seed,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.trajectory_out}")


if __name__ == "__main__":
    main()
