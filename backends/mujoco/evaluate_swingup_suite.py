#!/usr/bin/env python3
"""Evaluate single-pole energy swing-up and hysteretic capture in MuJoCo."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from actuatex_paths import ARTIFACTS_ROOT, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))

from tasks.inverted_pendulum.classical_control import (  # noqa: E402
    default_quadratic_cost,
    discrete_lqr_gain,
)
from tasks.inverted_pendulum.contract import (  # noqa: E402
    ACTION_FORCE_SCALE_N,
    DECIMATION,
    POLICY_DT,
)
from tasks.inverted_pendulum.swingup_control import (  # noqa: E402
    EnergySwingupController,
    HybridEnergyLQRController,
    single_pole_energy,
    upright_target_energy,
    wrap_to_pi,
)


def linearize_policy_dynamics() -> tuple[np.ndarray, np.ndarray]:
    """Linearize the single cart-pole and lift dynamics to the policy step."""

    model_path = (
        REPO_ROOT / "robots" / "inverted_pendulum" / "mjcf" / "actuatex_cartpole_1.xml"
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


class SwingupBatch:
    """Independent MuJoCo instances without the upright-only termination."""

    def __init__(self, num_envs: int, num_threads: int) -> None:
        model_path = (
            REPO_ROOT
            / "robots"
            / "inverted_pendulum"
            / "mjcf"
            / "actuatex_cartpole_1.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.pool = ThreadPoolExecutor(max_workers=max(1, num_threads))
        self.cart_qpos = int(self.model.jnt_qposadr[0])
        self.pole_qpos = int(self.model.jnt_qposadr[1])
        self.cart_dof = int(self.model.jnt_dofadr[0])
        self.pole_dof = int(self.model.jnt_dofadr[1])

    def reset(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        for data in self.datas:
            mujoco.mj_resetData(self.model, data)
            data.qpos[self.cart_qpos] = rng.uniform(-0.05, 0.05)
            data.qvel[self.cart_dof] = rng.uniform(-0.03, 0.03)
            data.qpos[self.pole_qpos] = np.pi + rng.uniform(-0.04, 0.04)
            data.qvel[self.pole_dof] = rng.uniform(-0.06, 0.06)
            mujoco.mj_forward(self.model, data)

    def state(self) -> np.ndarray:
        return np.asarray(
            [
                (
                    data.qpos[self.cart_qpos],
                    data.qpos[self.pole_qpos],
                    data.qvel[self.cart_dof],
                    data.qvel[self.pole_dof],
                )
                for data in self.datas
            ],
            dtype=np.float64,
        )

    def step(self, action: np.ndarray, active: np.ndarray) -> None:
        chunks = np.array_split(np.flatnonzero(active), self.pool._max_workers)

        def step_chunk(env_ids: np.ndarray) -> None:
            for env_id in env_ids:
                data = self.datas[int(env_id)]
                data.ctrl[0] = ACTION_FORCE_SCALE_N * float(action[env_id])
                for _ in range(DECIMATION):
                    mujoco.mj_step(self.model, data)

        list(self.pool.map(step_chunk, chunks))

    def close(self) -> None:
        self.pool.shutdown(wait=True)


def evaluate_controller(
    batch: SwingupBatch,
    controller,
    *,
    seed: int,
    duration_s: float,
    stable_duration_s: float,
) -> dict:
    batch.reset(seed)
    controller.reset(batch.num_envs)
    max_steps = round(duration_s / POLICY_DT)
    required_stable_steps = round(stable_duration_s / POLICY_DT)
    active = np.ones(batch.num_envs, dtype=bool)
    entered_upright = np.zeros(batch.num_envs, dtype=bool)
    first_upright_step = np.full(batch.num_envs, -1, dtype=np.int64)
    stable_steps = np.zeros(batch.num_envs, dtype=np.int64)
    successes = np.zeros(batch.num_envs, dtype=bool)
    rail_violations = np.zeros(batch.num_envs, dtype=bool)
    action_sum = np.zeros(batch.num_envs, dtype=np.float64)
    energy_error_sum = np.zeros(batch.num_envs, dtype=np.float64)
    samples = np.zeros(batch.num_envs, dtype=np.int64)

    for step in range(max_steps):
        state = batch.state()
        state[:, 1] = wrap_to_pi(state[:, 1])
        action = controller.act(state)
        batch.step(action, active)
        next_state = batch.state()
        theta = wrap_to_pi(next_state[:, 1])
        upright = (
            (np.abs(theta) < 0.30)
            & (np.abs(next_state[:, 3]) < 2.5)
            & (np.abs(next_state[:, 0]) < 1.8)
        )
        new_entries = active & upright & ~entered_upright
        entered_upright[new_entries] = True
        first_upright_step[new_entries] = step + 1

        stable = (
            (np.abs(theta) < 0.30)
            & (np.abs(next_state[:, 3]) < 2.5)
            & (np.abs(next_state[:, 0]) < 2.2)
        )
        stable_steps[active & stable] += 1
        stable_steps[active & ~stable] = 0
        new_successes = active & (stable_steps >= required_stable_steps)
        successes[new_successes] = True

        rail = active & (np.abs(next_state[:, 0]) >= 2.2)
        nonfinite = active & ~np.all(np.isfinite(next_state), axis=1)
        rail_violations[rail] = True
        ids = np.flatnonzero(active)
        action_sum[ids] += np.abs(action[ids])
        energy_error_sum[ids] += np.abs(
            single_pole_energy(theta[ids], next_state[ids, 3]) - upright_target_energy()
        )
        samples[ids] += 1
        active[new_successes | rail | nonfinite] = False
        if not active.any():
            break

    first_times = first_upright_step[first_upright_step >= 0] * POLICY_DT
    counts = np.maximum(samples, 1)
    return {
        "episodes": batch.num_envs,
        "upright_entries": int(entered_upright.sum()),
        "upright_entry_rate": float(entered_upright.mean()),
        "successes": int(successes.sum()),
        "success_rate": float(successes.mean()),
        "mean_first_upright_s": (
            float(np.mean(first_times)) if first_times.size else None
        ),
        "median_first_upright_s": (
            float(np.median(first_times)) if first_times.size else None
        ),
        "rail_violations": int(rail_violations.sum()),
        "rail_violation_rate": float(rail_violations.mean()),
        "mean_abs_normalized_action": float(np.mean(action_sum / counts)),
        "mean_abs_energy_error_j": float(np.mean(energy_error_sum / counts)),
        "seed": seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--duration_s", type=float, default=15.0)
    parser.add_argument("--stable_duration_s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument(
        "--out",
        type=Path,
        default=ARTIFACTS_ROOT
        / "inverted_pendulum"
        / "evaluation"
        / "swingup_suite_mujoco.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.num_threads <= 0:
        raise ValueError("episodes and num_threads must be positive")
    if args.duration_s <= 0.0 or args.stable_duration_s <= 0.0:
        raise ValueError("durations must be positive")
    if args.stable_duration_s >= args.duration_s:
        raise ValueError("stable_duration_s must be shorter than duration_s")

    matrix_a, matrix_b = linearize_policy_dynamics()
    cost_q, cost_r = default_quadratic_cost(1)
    lqr_gain = discrete_lqr_gain(matrix_a, matrix_b, cost_q, cost_r)
    controllers = {
        "energy_pump": EnergySwingupController(),
        "energy_shaping": EnergySwingupController(cart_kp=1.0, cart_kd=1.5),
        "hybrid_energy_lqr": HybridEnergyLQRController(
            lqr_gain,
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

    output = {
        "schema_version": 1,
        "backend": f"MuJoCo {mujoco.__version__}",
        "benchmark": "single-pole downward swing-up and upright capture",
        "initial_state": {
            "theta_rad": "pi + Uniform(-0.04, 0.04)",
            "angular_velocity_rad_s": "Uniform(-0.06, 0.06)",
            "cart_position_m": "Uniform(-0.05, 0.05)",
            "cart_velocity_m_s": "Uniform(-0.03, 0.03)",
        },
        "duration_s": args.duration_s,
        "stable_success_duration_s": args.stable_duration_s,
        "upright_entry_condition": "|theta|<0.30 rad, |omega|<2.5 rad/s, |x|<1.8 m",
        "success_condition": (
            "upright condition held continuously for "
            f"{args.stable_duration_s:g} seconds"
        ),
        "rail_failure_condition": "|x|>=2.2 m",
        "lqr_gain": lqr_gain.tolist(),
        "controller_parameters": {
            "energy_gain": 10.0,
            "target_energy_offset_j": 0.06,
            "energy_shaping_cart_kp": 1.0,
            "energy_shaping_cart_kd": 1.5,
            "capture_angle_rad": 0.30,
            "release_angle_rad": 0.55,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
