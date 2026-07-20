"""Shared policy/control contract for the serial wheel-legged MuJoCo twin.

This module is intentionally NumPy-only.  It mirrors the Isaac Lab task's
28-dimensional observation and mixed position/velocity action contract without
importing either simulator, so unit tests can catch ordering or scaling drift.
"""

from __future__ import annotations

import numpy as np


POLICY_JOINT_NAMES = (
    "left_hip_joint",
    "left_knee_joint",
    "right_hip_joint",
    "right_knee_joint",
    "left_wheel_joint",
    "right_wheel_joint",
)
LEG_JOINT_NAMES = POLICY_JOINT_NAMES[:4]
WHEEL_JOINT_NAMES = POLICY_JOINT_NAMES[4:]

DEFAULT_JOINT_POSITION = np.array(
    [0.35, -0.70, 0.35, -0.70, 0.0, 0.0], dtype=np.float64
)
DEFAULT_LEG_POSITION = DEFAULT_JOINT_POSITION[:4].copy()

OBSERVATION_DIM = 28
ACTION_DIM = 6
LEG_ACTION_SCALE_RAD = 0.45
WHEEL_ACTION_SCALE_RADPS = 20.0
JOINT_VELOCITY_OBS_SCALE = 0.05

LEG_STIFFNESS = 40.0
LEG_DAMPING = 1.0
LEG_EFFORT_LIMIT = 30.0
WHEEL_STIFFNESS = 0.0
WHEEL_DAMPING = 0.5
WHEEL_EFFORT_LIMIT = 12.0

SIM_DT = 0.005
DECIMATION = 4
POLICY_DT = SIM_DT * DECIMATION
INITIAL_BASE_POSITION = np.array([0.0, 0.0, 0.50], dtype=np.float64)
GRAVITY_DIRECTION_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def _vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value")
    return result


def quat_wxyz_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return the body-to-world rotation matrix for a normalized quaternion."""

    w, x, y, z = _vector(quaternion, 4, "quaternion")
    norm = np.linalg.norm((w, x, y, z))
    if norm < 1.0e-12:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = np.array((w, x, y, z), dtype=np.float64) / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Express the unit gravity direction in the base frame."""

    rotation_body_to_world = quat_wxyz_to_rotation_matrix(quaternion_wxyz)
    return rotation_body_to_world.T @ GRAVITY_DIRECTION_WORLD


def build_observation(
    base_linear_velocity_body: np.ndarray,
    base_angular_velocity_body: np.ndarray,
    gravity_body: np.ndarray,
    command_vx_vy_yaw: np.ndarray,
    joint_position_policy_order: np.ndarray,
    joint_velocity_policy_order: np.ndarray,
    previous_action: np.ndarray,
) -> np.ndarray:
    """Assemble the exact 28-D Isaac Lab policy observation.

    Wheel angles are deliberately omitted because they are continuous and
    unbounded.  Leg angles are relative to the default standing pose.
    """

    base_linear_velocity_body = _vector(
        base_linear_velocity_body, 3, "base_linear_velocity_body"
    )
    base_angular_velocity_body = _vector(
        base_angular_velocity_body, 3, "base_angular_velocity_body"
    )
    gravity_body = _vector(gravity_body, 3, "gravity_body")
    command_vx_vy_yaw = _vector(command_vx_vy_yaw, 3, "command_vx_vy_yaw")
    joint_position_policy_order = _vector(
        joint_position_policy_order, ACTION_DIM, "joint_position_policy_order"
    )
    joint_velocity_policy_order = _vector(
        joint_velocity_policy_order, ACTION_DIM, "joint_velocity_policy_order"
    )
    previous_action = _vector(previous_action, ACTION_DIM, "previous_action")

    observation = np.concatenate(
        (
            base_linear_velocity_body,
            base_angular_velocity_body,
            gravity_body,
            command_vx_vy_yaw,
            joint_position_policy_order[:4] - DEFAULT_LEG_POSITION,
            joint_velocity_policy_order * JOINT_VELOCITY_OBS_SCALE,
            previous_action,
        )
    )
    if observation.shape != (OBSERVATION_DIM,):
        raise AssertionError(
            f"internal observation shape drifted to {observation.shape}"
        )
    return observation.astype(np.float32)


def action_to_targets(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map normalized actor output to four leg positions and two wheel speeds."""

    action = _vector(action, ACTION_DIM, "action")
    leg_position_target = DEFAULT_LEG_POSITION + LEG_ACTION_SCALE_RAD * action[:4]
    wheel_velocity_target = WHEEL_ACTION_SCALE_RADPS * action[4:]
    return leg_position_target, wheel_velocity_target


def compute_mixed_pd_torque(
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    leg_position_target: np.ndarray,
    wheel_velocity_target: np.ndarray,
    *,
    leg_stiffness: float = LEG_STIFFNESS,
    leg_damping: float = LEG_DAMPING,
    wheel_damping: float = WHEEL_DAMPING,
) -> np.ndarray:
    """Compute the clipped IdealPDActuator effort in policy joint order."""

    joint_position = _vector(joint_position, ACTION_DIM, "joint_position")
    joint_velocity = _vector(joint_velocity, ACTION_DIM, "joint_velocity")
    leg_position_target = _vector(leg_position_target, 4, "leg_position_target")
    wheel_velocity_target = _vector(wheel_velocity_target, 2, "wheel_velocity_target")

    torque = np.empty(ACTION_DIM, dtype=np.float64)
    torque[:4] = leg_stiffness * (leg_position_target - joint_position[:4])
    torque[:4] -= leg_damping * joint_velocity[:4]
    torque[4:] = wheel_damping * (wheel_velocity_target - joint_velocity[4:])
    torque[:4] = np.clip(torque[:4], -LEG_EFFORT_LIMIT, LEG_EFFORT_LIMIT)
    torque[4:] = np.clip(torque[4:], -WHEEL_EFFORT_LIMIT, WHEEL_EFFORT_LIMIT)
    return torque
