"""Robust and nonlinear controllers for the inverted-pendulum arena."""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag, logm, solve_continuous_are

from .contract import (
    ACTION_FORCE_SCALE_N,
    CART_MASS_KG,
    POLE_LENGTH_M,
    POLE_MASS_KG,
    POLE_WIDTH_M,
)


def discrete_to_continuous(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    sample_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover a zero-order-hold continuous model with an augmented logarithm."""

    state_dim, input_dim = matrix_b.shape
    augmented = np.block(
        [
            [matrix_a, matrix_b],
            [np.zeros((input_dim, state_dim)), np.eye(input_dim)],
        ]
    )
    generator = np.real_if_close(logm(augmented) / sample_time).astype(np.float64)
    return generator[:state_dim, :state_dim], generator[:state_dim, state_dim:]


def h_infinity_state_feedback_gain(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    cost_q: np.ndarray,
    cost_r: np.ndarray,
    *,
    sample_time: float,
    gamma_candidates: tuple[float, ...] = (0.5, 0.75, 1.0, 2.0, 5.0, 10.0),
    disturbance_scale: float = 0.1,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Solve the continuous H-infinity game Riccati equation.

    State disturbances enter through ``disturbance_scale * I``.  The routine
    searches from the strongest requested attenuation toward easier games and
    accepts the first stabilizing real solution.
    """

    continuous_a, continuous_b = discrete_to_continuous(matrix_a, matrix_b, sample_time)
    disturbance_b = np.eye(matrix_a.shape[0]) * disturbance_scale
    for gamma in gamma_candidates:
        augmented_b = np.hstack((continuous_b, disturbance_b))
        game_r = block_diag(
            cost_r,
            -(gamma**2) * np.eye(disturbance_b.shape[1]),
        )
        try:
            riccati = solve_continuous_are(
                continuous_a,
                augmented_b,
                cost_q,
                game_r,
            )
        except np.linalg.LinAlgError:
            continue
        gain = np.linalg.solve(cost_r, continuous_b.T @ riccati)
        poles = np.linalg.eigvals(continuous_a - continuous_b @ gain)
        if np.all(np.isfinite(gain)) and np.max(np.real(poles)) < 0.0:
            return gain, gamma, poles
    raise RuntimeError("no stabilizing H-infinity Riccati solution was found")


class DiscreteSlidingModeController:
    """Discrete reaching-law sliding-mode controller with a boundary layer."""

    def __init__(
        self,
        matrix_a: np.ndarray,
        matrix_b: np.ndarray,
        sliding_surface: np.ndarray,
        *,
        contraction: float = 0.90,
        switching_strength: float = 0.08,
        boundary_layer: float = 0.25,
    ) -> None:
        if not 0.0 <= contraction < 1.0:
            raise ValueError("contraction must lie in [0, 1)")
        if switching_strength < 0.0 or boundary_layer <= 0.0:
            raise ValueError("sliding-mode gains must be positive")
        self.matrix_a = np.asarray(matrix_a, dtype=np.float64)
        self.matrix_b = np.asarray(matrix_b, dtype=np.float64)
        self.sliding_surface = np.asarray(sliding_surface, dtype=np.float64)
        denominator = float((self.sliding_surface @ self.matrix_b).item())
        if abs(denominator) < 1.0e-9:
            raise ValueError("sliding surface is orthogonal to the input channel")
        self.input_projection = denominator
        self.contraction = contraction
        self.switching_strength = switching_strength
        self.boundary_layer = boundary_layer

    def reset(self, num_envs: int) -> None:
        del num_envs

    def act(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        sliding_value = (state @ self.sliding_surface.T)[:, 0]
        desired_next = (
            self.contraction * sliding_value
            - self.switching_strength * np.tanh(sliding_value / self.boundary_layer)
        )
        open_loop_next = (state @ self.matrix_a.T @ self.sliding_surface.T)[:, 0]
        force = (desired_next - open_loop_next) / self.input_projection
        return np.clip(force / ACTION_FORCE_SCALE_N, -1.0, 1.0)


def _single_pole_mass_terms(theta: np.ndarray) -> tuple[np.ndarray, float, float]:
    center_length = POLE_LENGTH_M / 2.0
    center_inertia = POLE_MASS_KG * (POLE_WIDTH_M**2 + POLE_LENGTH_M**2) / 12.0
    hinge_inertia = center_inertia + POLE_MASS_KG * center_length**2
    coupling = POLE_MASS_KG * center_length * np.cos(theta)
    return coupling, hinge_inertia, center_length


class CollocatedFeedbackLinearizationController:
    """Exact input-output linearization of the cart acceleration."""

    def __init__(self, acceleration_gain: np.ndarray) -> None:
        gain = np.asarray(acceleration_gain, dtype=np.float64).reshape(-1)
        if gain.shape != (4,):
            raise ValueError("single-pole acceleration gain must contain four terms")
        self.acceleration_gain = gain

    def reset(self, num_envs: int) -> None:
        del num_envs

    def act(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        theta = state[:, 1]
        angular_velocity = state[:, 3]
        desired_cart_acceleration = -(state @ self.acceleration_gain)
        coupling, hinge_inertia, center_length = _single_pole_mass_terms(theta)
        pole_acceleration = (
            POLE_MASS_KG * 9.81 * center_length * np.sin(theta)
            - coupling * desired_cart_acceleration
        ) / hinge_inertia
        force = (
            (CART_MASS_KG + POLE_MASS_KG) * desired_cart_acceleration
            + coupling * pole_acceleration
            - POLE_MASS_KG * center_length * np.sin(theta) * np.square(angular_velocity)
        )
        return np.clip(force / ACTION_FORCE_SCALE_N, -1.0, 1.0)


class PartialFeedbackLinearizationController:
    """Linearize pole angular acceleration and regulate cart through its target."""

    def __init__(
        self,
        *,
        angle_kp: float = 45.0,
        angle_kd: float = 9.0,
        cart_to_angle_kp: float = 0.08,
        cart_to_angle_kd: float = 0.16,
    ) -> None:
        self.angle_kp = angle_kp
        self.angle_kd = angle_kd
        self.cart_to_angle_kp = cart_to_angle_kp
        self.cart_to_angle_kd = cart_to_angle_kd

    def reset(self, num_envs: int) -> None:
        del num_envs

    def act(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        cart_position, theta, cart_velocity, angular_velocity = state.T
        desired_angle = -(
            self.cart_to_angle_kp * cart_position
            + self.cart_to_angle_kd * cart_velocity
        )
        desired_pole_acceleration = -(
            self.angle_kp * (theta - desired_angle) + self.angle_kd * angular_velocity
        )
        coupling, hinge_inertia, center_length = _single_pole_mass_terms(theta)
        safe_coupling = np.where(
            np.abs(coupling) < 1.0e-4,
            np.copysign(1.0e-4, coupling + 1.0e-12),
            coupling,
        )
        desired_cart_acceleration = (
            POLE_MASS_KG * 9.81 * center_length * np.sin(theta)
            - hinge_inertia * desired_pole_acceleration
        ) / safe_coupling
        force = (
            (CART_MASS_KG + POLE_MASS_KG) * desired_cart_acceleration
            + coupling * desired_pole_acceleration
            - POLE_MASS_KG * center_length * np.sin(theta) * np.square(angular_velocity)
        )
        return np.clip(force / ACTION_FORCE_SCALE_N, -1.0, 1.0)
