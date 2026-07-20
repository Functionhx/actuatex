#!/usr/bin/env python3
"""Stress-test model-based controllers under dynamics and I/O shifts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
from torch import nn

from actuatex_paths import ARTIFACTS_ROOT, REPO_ROOT, RSL_RL_ROOT

if RSL_RL_ROOT.is_dir():
    sys.path.insert(0, str(RSL_RL_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from evaluate_classical_control_suite import (  # noqa: E402
    build_controllers,
    linearize_policy_dynamics,
    read_state,
)
from inverted_pendulum_env import MjSerialInvertedPendulumEnv  # noqa: E402
from tasks.inverted_pendulum.contract import build_observation  # noqa: E402


@dataclass(frozen=True)
class RobustnessScenario:
    name: str
    pole_mass_scale: float = 1.0
    pole_damping_scale: float = 1.0
    actuator_scale: float = 1.0
    action_delay_steps: int = 0
    measurement_noise_std: float = 0.0
    push_force_n: float = 0.0


SCENARIOS = (
    RobustnessScenario("nominal"),
    RobustnessScenario("light_poles", pole_mass_scale=0.75),
    RobustnessScenario("heavy_poles", pole_mass_scale=1.25),
    RobustnessScenario("high_joint_damping", pole_damping_scale=3.0),
    RobustnessScenario("weak_actuator", actuator_scale=0.80),
    RobustnessScenario("two_step_delay", action_delay_steps=2),
    RobustnessScenario("encoder_noise", measurement_noise_std=0.01),
    RobustnessScenario("opposed_pushes", push_force_n=5.0),
    RobustnessScenario(
        "combined_shift",
        pole_mass_scale=1.20,
        pole_damping_scale=2.0,
        actuator_scale=0.85,
        action_delay_steps=1,
        measurement_noise_std=0.005,
        push_force_n=4.0,
    ),
)


class PPOPolicyController:
    """Deterministic RSL-RL actor exposed through the controller interface."""

    def __init__(self, order: int, checkpoint: Path) -> None:
        self.order = order
        self.checkpoint = checkpoint.resolve()
        self.actor = nn.Sequential(
            nn.Linear(14, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )
        payload = torch.load(
            self.checkpoint,
            weights_only=False,
            map_location="cpu",
        )
        model_state = payload["model_state_dict"]
        actor_state = {
            key.removeprefix("actor."): value
            for key, value in model_state.items()
            if key.startswith("actor.")
        }
        self.actor.load_state_dict(actor_state, strict=True)
        self.actor.eval()

    def reset(self, num_envs: int) -> None:
        del num_envs

    def act(self, state: np.ndarray) -> np.ndarray:
        dof_count = self.order + 1
        observation = build_observation(
            state[:, 0],
            state[:, dof_count],
            state[:, 1:dof_count],
            state[:, dof_count + 1 :],
            self.order,
        )
        with torch.inference_mode():
            action = self.actor(torch.from_numpy(observation)).numpy()[:, 0]
        return np.clip(action, -1.0, 1.0)


def apply_dynamics_shift(
    env: MjSerialInvertedPendulumEnv, scenario: RobustnessScenario
) -> None:
    for pole_index in range(1, env.order + 1):
        body_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, f"pole_{pole_index}"
        )
        env.model.body_mass[body_id] *= scenario.pole_mass_scale
        env.model.body_inertia[body_id] *= scenario.pole_mass_scale
    env.model.dof_damping[env.pole_dof_addresses] *= scenario.pole_damping_scale
    env.model.actuator_gear[0, 0] *= scenario.actuator_scale
    mujoco.mj_setConst(env.model, env.datas[0])


def push_profile(step: int, force_n: float, num_envs: int) -> np.ndarray:
    """Apply paired half-second pushes in opposite directions."""

    force = np.zeros(num_envs, dtype=np.float64)
    if 120 <= step < 150:
        force.fill(force_n)
    elif 360 <= step < 390:
        force.fill(-force_n)
    return force


def evaluate_controller(
    env: MjSerialInvertedPendulumEnv,
    controller,
    scenario: RobustnessScenario,
    *,
    seed: int,
) -> dict:
    env.rng = np.random.default_rng(seed)
    env.reset()
    controller.reset(env.num_envs)
    noise_rng = np.random.default_rng(seed + 100_000)
    delay_queue = [
        np.zeros(env.num_envs, dtype=np.float64)
        for _ in range(scenario.action_delay_steps)
    ]
    active = np.ones(env.num_envs, dtype=bool)
    success = np.zeros(env.num_envs, dtype=bool)
    duration_steps = np.zeros(env.num_envs, dtype=np.int64)
    angle_squared_sum = np.zeros(env.num_envs, dtype=np.float64)
    cart_squared_sum = np.zeros(env.num_envs, dtype=np.float64)
    command_sum = np.zeros(env.num_envs, dtype=np.float64)
    samples = np.zeros(env.num_envs, dtype=np.int64)

    for step in range(env.max_episode_length):
        state = read_state(env)
        if hasattr(controller, "act_from_measurement"):
            measurement = state @ controller.measurement_c.T
            if scenario.measurement_noise_std:
                measurement += noise_rng.normal(
                    0.0,
                    scenario.measurement_noise_std,
                    measurement.shape,
                )
            command = controller.act_from_measurement(measurement)
        else:
            observed_state = state.copy()
            if scenario.measurement_noise_std:
                dof_count = env.order + 1
                observed_state[:, :dof_count] += noise_rng.normal(
                    0.0,
                    scenario.measurement_noise_std,
                    observed_state[:, :dof_count].shape,
                )
                observed_state[:, dof_count:] += noise_rng.normal(
                    0.0,
                    2.0 * scenario.measurement_noise_std,
                    observed_state[:, dof_count:].shape,
                )
            command = controller.act(observed_state)
        if delay_queue:
            applied_action = delay_queue.pop(0)
            delay_queue.append(command.copy())
        else:
            applied_action = command
        env.external_cart_force_n[:] = push_profile(
            step, scenario.push_force_n, env.num_envs
        )
        env.step(torch.from_numpy(applied_action.astype(np.float32))[:, None])

        ids = np.flatnonzero(active)
        angle_squared_sum[ids] += np.mean(
            np.square(env.last_absolute_angles[ids]), axis=1
        )
        cart_squared_sum[ids] += np.square(env.last_cart_position[ids])
        command_sum[ids] += np.abs(command[ids])
        samples[ids] += 1
        finished = active & (env.last_terminal | env.last_timeout)
        success[finished] = env.last_timeout[finished] & ~env.last_terminal[finished]
        duration_steps[finished] = step + 1
        active[finished] = False
        if not active.any():
            break

    duration_steps[active] = env.max_episode_length
    counts = np.maximum(samples, 1)
    return {
        "episodes": env.num_envs,
        "successes": int(success.sum()),
        "success_rate": float(success.mean()),
        "mean_balance_duration_s": float(duration_steps.mean() * env.dt),
        "absolute_pole_angle_rmse_rad": float(
            np.sqrt(np.mean(angle_squared_sum / counts))
        ),
        "cart_position_rmse_m": float(np.sqrt(np.mean(cart_squared_sum / counts))),
        "mean_abs_normalized_command": float(np.mean(command_sum / counts)),
        "seed": seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orders", type=int, nargs="+", choices=(1, 2, 3), default=(1, 2, 3)
    )
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=851)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=("lqr", "h_infinity", "sliding_mode"),
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=[scenario.name for scenario in SCENARIOS],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ARTIFACTS_ROOT
        / "inverted_pendulum"
        / "evaluation"
        / "robust_control_suite_mujoco.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.num_threads <= 0:
        raise ValueError("episodes and num_threads must be positive")
    scenario_lookup = {scenario.name: scenario for scenario in SCENARIOS}
    unknown_scenarios = sorted(set(args.scenarios) - scenario_lookup.keys())
    if unknown_scenarios:
        raise ValueError(f"unknown scenarios: {unknown_scenarios}")

    order_results = []
    for order in args.orders:
        matrix_a, matrix_b = linearize_policy_dynamics(order)
        all_controllers, metadata = build_controllers(
            order,
            matrix_a,
            matrix_b,
            cem_checkpoint=None,
            measurement_noise_std=0.01,
        )
        ppo_checkpoint = (
            ARTIFACTS_ROOT
            / "checkpoints"
            / "inverted_pendulum"
            / f"mujoco_order_{order}.pt"
        )
        if ppo_checkpoint.is_file():
            all_controllers["mujoco_ppo"] = PPOPolicyController(order, ppo_checkpoint)
            metadata["mujoco_ppo_checkpoint"] = str(ppo_checkpoint.resolve())
        controllers = {
            name: all_controllers[name]
            for name in args.methods
            if name in all_controllers
        }
        if not controllers:
            raise ValueError(f"none of {args.methods} apply to order {order}")
        scenario_results = []
        for scenario_name in args.scenarios:
            scenario = scenario_lookup[scenario_name]
            env = MjSerialInvertedPendulumEnv(
                order,
                num_envs=args.episodes,
                num_threads=args.num_threads,
                seed=args.seed + order,
            )
            apply_dynamics_shift(env, scenario)
            results = {}
            try:
                for name, controller in controllers.items():
                    result = evaluate_controller(
                        env,
                        controller,
                        scenario,
                        seed=args.seed + 100 * order,
                    )
                    results[name] = result
                    print(
                        f"order={order} scenario={scenario.name} method={name} "
                        f"success={result['success_rate']:.3f}"
                    )
            finally:
                env.close()
            scenario_results.append(
                {
                    "scenario": asdict(scenario),
                    "results": results,
                }
            )
        order_results.append(
            {
                "order": order,
                "controller_metadata": metadata,
                "scenarios": scenario_results,
            }
        )

    output = {
        "schema_version": 1,
        "backend": f"MuJoCo {mujoco.__version__}",
        "benchmark": "upright balance under paired dynamics, delay, noise and push shifts",
        "orders": order_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
