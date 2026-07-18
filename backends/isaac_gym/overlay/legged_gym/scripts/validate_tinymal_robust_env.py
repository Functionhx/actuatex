"""Fast assertions for the TinyMal sim-to-real environment.

This is intentionally an environment validation rather than a learning test.
It checks that every configured randomization reaches Isaac Gym, verifies the
per-environment action-delay queue, and runs a short finite-physics rollout.
"""

import json

import isaacgym  # noqa: F401  # Must precede torch.
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403  # Registers tasks.
from legged_gym.utils import get_args, task_registry


def _range(values):
    values = np.asarray(values, dtype=np.float64)
    return [float(values.min()), float(values.max())]


def _assert_range(name, values, expected, minimum_span=0.0):
    low, high = _range(values)
    tolerance = 1.0e-5
    if low < expected[0] - tolerance or high > expected[1] + tolerance:
        raise AssertionError(
            f"{name} outside configured range {expected}: observed [{low}, {high}]"
        )
    if high - low < minimum_span:
        raise AssertionError(
            f"{name} did not vary enough: observed span {high - low:.6f}"
        )
    return [low, high]


def validate(args):
    env_cfg, _ = task_registry.get_cfgs(name="tinymal_robust")
    env_cfg.env.num_envs = args.num_envs or 64
    env, _ = task_registry.make_env(
        name="tinymal_robust", args=args, env_cfg=env_cfg
    )
    all_env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset()

    cfg = env.cfg.domain_rand
    summary = {
        "num_envs": env.num_envs,
        "observation_shape": list(env.obs_buf.shape),
        "terrain_height_raw_range": _range(env.terrain.height_field_raw),
        "terrain_roughness_m_range": _range(env.terrain.patch_roughness),
        "flat_terrain_patches": int(np.count_nonzero(env.terrain.patch_roughness == 0.0)),
        "friction_range": _assert_range(
            "friction",
            env.friction_coeffs.cpu().numpy(),
            cfg.friction_range,
            minimum_span=0.25,
        ),
        "base_mass_scale_range": _assert_range(
            "base mass scale",
            env.base_mass_scales,
            cfg.base_mass_scale_range,
            minimum_span=0.10,
        ),
        "joint_friction_range": _assert_range(
            "joint friction",
            env.joint_friction_coeffs,
            cfg.joint_friction_range,
            minimum_span=0.04,
        ),
        "joint_armature_range": _assert_range(
            "joint armature",
            env.joint_armature_coeffs,
            cfg.joint_armature_range,
            minimum_span=0.008,
        ),
        "motor_kp_scale_range": _assert_range(
            "motor Kp scale",
            env.motor_kp_scales.cpu().numpy(),
            cfg.motor_kp_scale_range,
            minimum_span=0.15,
        ),
        "motor_kd_scale_range": _assert_range(
            "motor Kd scale",
            env.motor_kd_scales.cpu().numpy(),
            cfg.motor_kd_scale_range,
            minimum_span=0.15,
        ),
        "control_delay_values": sorted(
            set(int(value) for value in env.control_delays.cpu().tolist())
        ),
    }

    expected_delays = list(
        range(int(cfg.control_delay_range[0]), int(cfg.control_delay_range[1]) + 1)
    )
    if env.num_envs >= 16 and summary["control_delay_values"] != expected_delays:
        raise AssertionError(
            "control delay samples do not cover configured values: "
            f"{summary['control_delay_values']} vs {expected_delays}"
        )
    if summary["terrain_height_raw_range"] == [0.0, 0.0]:
        raise AssertionError("rough heightfield contains only zeros")

    # Query the properties back from Isaac Gym, rather than trusting only the
    # values cached while actors were being created.
    actor_body_props = env.gym.get_actor_rigid_body_properties(
        env.envs[0], env.actor_handles[0]
    )
    actor_dof_props = env.gym.get_actor_dof_properties(
        env.envs[0], env.actor_handles[0]
    )
    summary["queried_base_mass_kg"] = float(actor_body_props[0].mass)
    summary["queried_joint_friction_range"] = _range(actor_dof_props["friction"])
    summary["queried_joint_armature_range"] = _range(actor_dof_props["armature"])

    # Deterministic queue check: an environment with delay d must receive the
    # action issued exactly d policy steps earlier.
    tested_envs = min(env.num_envs, len(expected_delays))
    env.action_history.zero_()
    env.control_delays[:tested_envs] = torch.as_tensor(
        expected_delays[:tested_envs], device=env.device
    )
    observed = []
    for action_value in range(1, max(expected_delays) + 2):
        env.actions.fill_(float(action_value))
        delayed = env._delayed_actions().clone()
        row = []
        for env_index in range(tested_envs):
            delay = expected_delays[env_index]
            expected = max(0, action_value - delay)
            actual = int(round(float(delayed[env_index, 0].item())))
            if actual != expected:
                raise AssertionError(
                    f"delay queue mismatch env={env_index}, delay={delay}, "
                    f"step={action_value}: {actual} != {expected}"
                )
            row.append(actual)
        observed.append(row)
    summary["delay_queue_trace"] = observed

    # Schedule a batch directly to verify force magnitude and duration bounds.
    env._push_substeps_remaining.zero_()
    env.episode_length_buf[:] = int(cfg.push_interval)
    env._schedule_random_pushes()
    force_magnitudes = torch.linalg.vector_norm(env._push_force[:, :2], dim=1)
    summary["scheduled_push_force_n_range"] = _assert_range(
        "push force",
        force_magnitudes.cpu().numpy(),
        cfg.push_force_range,
        minimum_span=5.0 if env.num_envs >= 16 else 0.0,
    )
    summary["scheduled_push_substeps_range"] = _range(
        env._push_substeps_remaining.cpu().numpy()
    )

    # Restore a normal episode and ensure the actual heightfield supports the
    # robot for one simulated second with finite tensors.
    env.reset_idx(all_env_ids)
    zero_actions = torch.zeros(
        env.num_envs, env.num_actions, device=env.device, dtype=torch.float
    )
    with torch.inference_mode():
        for _ in range(int(round(1.0 / env.dt))):
            obs, _, rewards, _, _ = env.step(zero_actions)
            if not bool(torch.isfinite(obs).all() and torch.isfinite(rewards).all()):
                raise AssertionError("non-finite observation or reward in smoke rollout")
    summary["post_rollout_base_height_mean_m"] = float(env.base_pos[:, 2].mean().item())
    summary["post_rollout_base_height_min_m"] = float(env.base_pos[:, 2].min().item())
    if summary["post_rollout_base_height_mean_m"] < 0.12:
        raise AssertionError(
            "robots are not supported by the heightfield after the smoke rollout"
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    validate(get_args())
