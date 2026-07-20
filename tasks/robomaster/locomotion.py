"""Backend-neutral locomotion policy and low-level control contract."""

from __future__ import annotations

import numpy as np

from .contract import (
    ACTION_DIM,
    ALL_JOINT_NAMES,
    FULL_ACTUATOR_DIM,
    SENTINEL_DEFAULT_JOINT_POSITION,
)

OBSERVATION_DIM = 28
LEG_ACTION_SCALE_RAD = 0.45
WHEEL_ACTION_SCALE_RADPS = 20.0
JOINT_VELOCITY_OBSERVATION_SCALE = 0.05
WHEEL_RADIUS_M = 0.075
TRACK_WIDTH_M = 0.44

BASE_LINEAR_VELOCITY_SLICE = slice(0, 3)
BASE_ANGULAR_VELOCITY_SLICE = slice(3, 6)
PROJECTED_GRAVITY_SLICE = slice(6, 9)
COMMAND_OBSERVATION_SLICE = slice(9, 12)
LEG_POSITION_OBSERVATION_SLICE = slice(12, 16)
JOINT_VELOCITY_OBSERVATION_SLICE = slice(16, 22)
PREVIOUS_ACTION_OBSERVATION_SLICE = slice(22, 28)

JOINT_STIFFNESS = np.asarray(
    [40.0, 40.0, 40.0, 40.0, 0.0, 0.0, 25.0, 30.0, 0.0, 0.0, 3.0],
    dtype=np.float64,
)
JOINT_DAMPING = np.asarray(
    [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.001, 0.001, 0.060],
    dtype=np.float64,
)
OBSERVATION_NOISE_AMPLITUDE = np.asarray(
    [0.10] * 3
    + [0.10] * 3
    + [0.05] * 3
    + [0.0] * 3
    + [0.01] * 4
    + [0.075] * 6
    + [0.0] * 6,
    dtype=np.float64,
)


def _last_axis(values: np.ndarray, size: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 0 or result.shape[-1] != size:
        raise ValueError(
            f"{label} must have last dimension {size}, got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains a non-finite value")
    return result


def action_to_joint_targets(
    action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map normalized locomotion actions to all 11 joint target arrays."""

    action = _last_axis(action, ACTION_DIM, "action")
    position = np.broadcast_to(
        SENTINEL_DEFAULT_JOINT_POSITION,
        action.shape[:-1] + (FULL_ACTUATOR_DIM,),
    ).copy()
    velocity = np.zeros_like(position)
    position[..., :4] += LEG_ACTION_SCALE_RAD * np.clip(
        action[..., :4], -1.0, 1.0
    )
    velocity[..., 4:6] = WHEEL_ACTION_SCALE_RADPS * np.clip(
        action[..., 4:6], -1.0, 1.0
    )
    return position, velocity


def compute_requested_joint_torque(
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    action: np.ndarray,
    *,
    position_target_override: np.ndarray | None = None,
    velocity_target_override: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the pre-electrical-limit explicit PD request in policy order."""

    position = _last_axis(
        joint_position, FULL_ACTUATOR_DIM, "joint_position"
    )
    velocity = _last_axis(
        joint_velocity, FULL_ACTUATOR_DIM, "joint_velocity"
    )
    if position.shape != velocity.shape:
        raise ValueError("joint position and velocity shapes must match")
    position_target, velocity_target = action_to_joint_targets(action)
    if position_target.shape != position.shape:
        position_target = np.broadcast_to(position_target, position.shape).copy()
        velocity_target = np.broadcast_to(velocity_target, velocity.shape).copy()
    if position_target_override is not None:
        override = _last_axis(
            position_target_override,
            FULL_ACTUATOR_DIM,
            "position_target_override",
        )
        position_target = np.broadcast_to(override, position.shape)
    if velocity_target_override is not None:
        override = _last_axis(
            velocity_target_override,
            FULL_ACTUATOR_DIM,
            "velocity_target_override",
        )
        velocity_target = np.broadcast_to(override, velocity.shape)
    return (
        JOINT_STIFFNESS * (position_target - position)
        + JOINT_DAMPING * (velocity_target - velocity)
    )


def projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Express the world gravity direction in each base frame."""

    quaternion = _last_axis(quaternion_wxyz, 4, "quaternion_wxyz")
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-12):
        raise ValueError("quaternion norm is zero")
    quaternion = quaternion / norm
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            2.0 * (z * z + w * w) - 1.0,
        ),
        axis=-1,
    ) * -1.0


def build_observation(
    base_linear_velocity_body: np.ndarray,
    base_angular_velocity_body: np.ndarray,
    gravity_body: np.ndarray,
    command_vx_vy_yaw: np.ndarray,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    previous_action: np.ndarray,
) -> np.ndarray:
    """Assemble the exact 28-D Isaac Lab policy observation."""

    base_linear_velocity = _last_axis(
        base_linear_velocity_body, 3, "base_linear_velocity_body"
    )
    base_angular_velocity = _last_axis(
        base_angular_velocity_body, 3, "base_angular_velocity_body"
    )
    gravity = _last_axis(gravity_body, 3, "gravity_body")
    command = _last_axis(command_vx_vy_yaw, 3, "command_vx_vy_yaw")
    position = _last_axis(joint_position, FULL_ACTUATOR_DIM, "joint_position")
    velocity = _last_axis(joint_velocity, FULL_ACTUATOR_DIM, "joint_velocity")
    action = _last_axis(previous_action, ACTION_DIM, "previous_action")
    leading_shapes = {
        value.shape[:-1]
        for value in (
            base_linear_velocity,
            base_angular_velocity,
            gravity,
            command,
            position,
            velocity,
            action,
        )
    }
    if len(leading_shapes) != 1:
        raise ValueError("all observation inputs must share leading dimensions")
    observation = np.concatenate(
        (
            base_linear_velocity,
            base_angular_velocity,
            gravity,
            command,
            position[..., :4] - SENTINEL_DEFAULT_JOINT_POSITION[:4],
            velocity[..., :ACTION_DIM] * JOINT_VELOCITY_OBSERVATION_SCALE,
            action,
        ),
        axis=-1,
    )
    if observation.shape[-1] != OBSERVATION_DIM:
        raise AssertionError(f"observation dimension drifted to {observation.shape}")
    return observation.astype(np.float32)


def actuator_property_by_name(values: np.ndarray) -> dict[str, float]:
    """Convert a policy-order vector to an Isaac Lab name-keyed mapping."""

    array = _last_axis(values, FULL_ACTUATOR_DIM, "values")
    if array.ndim != 1:
        raise ValueError("actuator properties must be one-dimensional")
    return {
        name: float(array[index]) for index, name in enumerate(ALL_JOINT_NAMES)
    }
