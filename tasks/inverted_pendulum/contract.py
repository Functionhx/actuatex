"""Backend-neutral contract for one-, two- and three-link cart-poles.

All task orders expose the same 14 observations so an actor can be warm-started
through the 1 -> 2 -> 3 curriculum and transferred between PhysX and MuJoCo.
Only the cart is actuated; every pendulum hinge remains passive.
"""

from __future__ import annotations

import math

import numpy as np


MAX_ORDER = 3
OBSERVATION_DIM = 14
ACTION_DIM = 1

SIM_DT = 1.0 / 120.0
DECIMATION = 2
POLICY_DT = SIM_DT * DECIMATION
EPISODE_LENGTH_S = 10.0

CART_TRAVEL_LIMIT_M = 2.4
TERMINATION_CART_POSITION_M = 2.2
TERMINATION_POLE_ANGLE_RAD = math.pi / 2.0
ACTION_FORCE_SCALE_N = 20.0

CART_POSITION_OBS_SCALE = 1.0 / CART_TRAVEL_LIMIT_M
CART_VELOCITY_OBS_SCALE = 0.2
POLE_VELOCITY_OBS_SCALE = 0.1

CART_MASS_KG = 1.0
POLE_MASS_KG = 0.20
POLE_LENGTH_M = 0.60
POLE_WIDTH_M = 0.04
CART_DAMPING = 0.05
POLE_DAMPING = 0.002

INITIAL_ANGLE_RANGE_RAD = {1: 0.25, 2: 0.16, 3: 0.10}


def validate_order(order: int) -> int:
    if order not in (1, 2, 3):
        raise ValueError(f"order must be 1, 2 or 3, got {order}")
    return order


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap radians to [-pi, pi] without a discontinuous observation."""

    angle = np.asarray(angle, dtype=np.float64)
    return np.arctan2(np.sin(angle), np.cos(angle))


def absolute_pole_angles(relative_joint_angles: np.ndarray) -> np.ndarray:
    """Convert serial hinge coordinates into each link's world-up angle."""

    relative_joint_angles = np.asarray(relative_joint_angles, dtype=np.float64)
    return wrap_angle(np.cumsum(relative_joint_angles, axis=-1))


def build_observation(
    cart_position: np.ndarray,
    cart_velocity: np.ndarray,
    pole_angles: np.ndarray,
    pole_velocities: np.ndarray,
    order: int,
) -> np.ndarray:
    """Build the shared padded observation for one or more environments.

    Layout: normalized cart x/v, then three ``sin, cos, omega`` slots, then
    three binary presence-mask entries.  Inactive pole slots are all zero.
    """

    order = validate_order(order)
    cart_position = np.asarray(cart_position, dtype=np.float64)
    cart_velocity = np.asarray(cart_velocity, dtype=np.float64)
    pole_angles = np.asarray(pole_angles, dtype=np.float64)
    pole_velocities = np.asarray(pole_velocities, dtype=np.float64)

    if cart_position.shape != cart_velocity.shape:
        raise ValueError("cart position and velocity shapes must match")
    expected_pole_shape = cart_position.shape + (order,)
    if pole_angles.shape != expected_pole_shape:
        raise ValueError(
            f"pole_angles must have shape {expected_pole_shape}, got {pole_angles.shape}"
        )
    if pole_velocities.shape != expected_pole_shape:
        raise ValueError(
            "pole_velocities must have shape "
            f"{expected_pole_shape}, got {pole_velocities.shape}"
        )

    observation = np.zeros(cart_position.shape + (OBSERVATION_DIM,), dtype=np.float64)
    observation[..., 0] = cart_position * CART_POSITION_OBS_SCALE
    observation[..., 1] = cart_velocity * CART_VELOCITY_OBS_SCALE
    for pole_index in range(order):
        offset = 2 + 3 * pole_index
        observation[..., offset] = np.sin(pole_angles[..., pole_index])
        observation[..., offset + 1] = np.cos(pole_angles[..., pole_index])
        observation[..., offset + 2] = (
            pole_velocities[..., pole_index] * POLE_VELOCITY_OBS_SCALE
        )
    observation[..., 11 : 11 + order] = 1.0
    return observation.astype(np.float32)


def compute_reward(
    cart_position: np.ndarray,
    cart_velocity: np.ndarray,
    pole_angles: np.ndarray,
    pole_velocities: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    terminated: np.ndarray,
) -> np.ndarray:
    """Dense balance reward shared by both physics backends."""

    cart_position = np.asarray(cart_position, dtype=np.float64)
    cart_velocity = np.asarray(cart_velocity, dtype=np.float64)
    pole_angles = absolute_pole_angles(pole_angles)
    pole_velocities = np.asarray(pole_velocities, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64).reshape(cart_position.shape)
    previous_action = np.asarray(previous_action, dtype=np.float64).reshape(
        cart_position.shape
    )
    terminated = np.asarray(terminated, dtype=bool)

    mean_angle_squared = np.mean(np.square(pole_angles), axis=-1)
    mean_pole_velocity_squared = np.mean(np.square(pole_velocities), axis=-1)
    # A quadratic angle term keeps a useful gradient all the way to the
    # termination boundary.  The first exponential formulation saturated for
    # failed rollouts and did not train even the single pole reliably.
    reward = 1.0 - 2.0 * mean_angle_squared
    reward -= 0.10 * np.square(cart_position)
    reward -= 0.01 * np.square(cart_velocity)
    reward -= 0.005 * mean_pole_velocity_squared
    reward -= 0.001 * np.square(action)
    reward -= 0.01 * np.square(action - previous_action)
    reward -= 5.0 * terminated.astype(np.float64)
    return reward.astype(np.float32)


def terminated(cart_position: np.ndarray, pole_angles: np.ndarray) -> np.ndarray:
    cart_position = np.asarray(cart_position, dtype=np.float64)
    pole_angles = absolute_pole_angles(pole_angles)
    return (np.abs(cart_position) > TERMINATION_CART_POSITION_M) | np.any(
        np.abs(pole_angles) > TERMINATION_POLE_ANGLE_RAD, axis=-1
    )
