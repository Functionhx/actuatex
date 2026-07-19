"""Backend-neutral building blocks for classical cart-pole controllers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import place_poles

from .contract import ACTION_FORCE_SCALE_N, POLICY_DT, validate_order


def default_quadratic_cost(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the shared state/action cost used by LQR and linear MPC."""

    order = validate_order(order)
    state_dim = 2 * (order + 1)
    cumulative = np.tril(np.ones((order, order)))
    cost_q = np.zeros((state_dim, state_dim), dtype=np.float64)
    cost_q[0, 0] = 1.0
    cost_q[1 : order + 1, 1 : order + 1] = cumulative.T @ cumulative * (30.0 / order)
    velocity_start = order + 1
    cost_q[velocity_start, velocity_start] = 0.2
    cost_q[velocity_start + 1 :, velocity_start + 1 :] = (
        cumulative.T @ cumulative * (3.0 / order)
    )
    return cost_q, np.eye(1, dtype=np.float64)


def discrete_lqr_gain(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    cost_q: np.ndarray,
    cost_r: np.ndarray,
) -> np.ndarray:
    """Solve the infinite-horizon discrete algebraic Riccati equation."""

    riccati = solve_discrete_are(matrix_a, matrix_b, cost_q, cost_r)
    return np.linalg.solve(
        cost_r + matrix_b.T @ riccati @ matrix_b,
        matrix_b.T @ riccati @ matrix_a,
    )


def finite_horizon_lqr_gain(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    cost_q: np.ndarray,
    cost_r: np.ndarray,
    *,
    horizon: int,
    terminal_cost: np.ndarray | None = None,
) -> np.ndarray:
    """Return the first receding-horizon gain of an unconstrained linear MPC."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    value = cost_q.copy() if terminal_cost is None else terminal_cost.copy()
    gain = np.zeros((matrix_b.shape[1], matrix_a.shape[0]), dtype=np.float64)
    for _ in range(horizon):
        gain = np.linalg.solve(
            cost_r + matrix_b.T @ value @ matrix_b,
            matrix_b.T @ value @ matrix_a,
        )
        value = (
            cost_q
            + matrix_a.T @ value @ matrix_a
            - matrix_a.T @ value @ matrix_b @ gain
        )
    return gain


def pole_placement_gain(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    *,
    policy_dt: float = POLICY_DT,
) -> tuple[np.ndarray, np.ndarray]:
    """Place distinct real closed-loop poles from slow cart to fast links."""

    state_dim = matrix_a.shape[0]
    continuous_rates = -np.geomspace(0.7, 12.0, state_dim)
    desired_poles = np.exp(continuous_rates * policy_dt)
    result = place_poles(matrix_a, matrix_b, desired_poles, method="YT")
    return np.asarray(result.gain_matrix), desired_poles


def steady_state_kalman_gain(
    matrix_a: np.ndarray,
    measurement_c: np.ndarray,
    process_covariance: np.ndarray,
    measurement_covariance: np.ndarray,
) -> np.ndarray:
    """Design a steady-state discrete Kalman correction gain."""

    covariance = solve_discrete_are(
        matrix_a.T,
        measurement_c.T,
        process_covariance,
        measurement_covariance,
    )
    innovation_covariance = (
        measurement_c @ covariance @ measurement_c.T + measurement_covariance
    )
    return np.linalg.solve(
        innovation_covariance,
        measurement_c @ covariance,
    ).T


@dataclass
class StateFeedbackController:
    """Saturated full-state feedback used by LQR, MPC and pole placement."""

    gain: np.ndarray

    def reset(self, num_envs: int) -> None:
        del num_envs

    def act(self, state: np.ndarray) -> np.ndarray:
        force = -(np.asarray(state) @ np.asarray(self.gain).T)
        return np.clip(force[:, 0] / ACTION_FORCE_SCALE_N, -1.0, 1.0)


@dataclass
class PIDController:
    """LQR-informed PID/PD balance controller for a single cart-pole."""

    cart_kp: float
    cart_kd: float
    angle_kp: float
    angle_ki: float
    angle_kd: float
    integral_limit: float = 0.25

    def reset(self, num_envs: int) -> None:
        self.integral = np.zeros(num_envs, dtype=np.float64)

    def act(self, state: np.ndarray) -> np.ndarray:
        if state.shape[1] != 4:
            raise ValueError("PIDController is defined for the single pole")
        cart_position, angle, cart_velocity, angular_velocity = state.T
        self.integral += angle * POLICY_DT
        np.clip(
            self.integral,
            -self.integral_limit,
            self.integral_limit,
            out=self.integral,
        )
        force = (
            self.cart_kp * cart_position
            + self.cart_kd * cart_velocity
            + self.angle_kp * angle
            + self.angle_ki * self.integral
            + self.angle_kd * angular_velocity
        )
        return np.clip(force / ACTION_FORCE_SCALE_N, -1.0, 1.0)


@dataclass
class CascadedPIDController:
    """Outer cart-position PD feeding an inner pole-angle PID."""

    outer_kp: float
    outer_kd: float
    inner_kp: float
    inner_ki: float
    inner_kd: float
    integral_limit: float = 0.25

    def reset(self, num_envs: int) -> None:
        self.integral = np.zeros(num_envs, dtype=np.float64)

    def act(self, state: np.ndarray) -> np.ndarray:
        if state.shape[1] != 4:
            raise ValueError("CascadedPIDController is defined for the single pole")
        cart_position, angle, cart_velocity, angular_velocity = state.T
        desired_angle = -(self.outer_kp * cart_position + self.outer_kd * cart_velocity)
        angle_error = angle - desired_angle
        self.integral += angle_error * POLICY_DT
        np.clip(
            self.integral,
            -self.integral_limit,
            self.integral_limit,
            out=self.integral,
        )
        force = (
            self.inner_kp * angle_error
            + self.inner_ki * self.integral
            + self.inner_kd * angular_velocity
        )
        return np.clip(force / ACTION_FORCE_SCALE_N, -1.0, 1.0)


class LQGController:
    """LQR feedback driven by a steady-state linear Kalman filter."""

    def __init__(
        self,
        matrix_a: np.ndarray,
        matrix_b: np.ndarray,
        measurement_c: np.ndarray,
        feedback_gain: np.ndarray,
        kalman_gain: np.ndarray,
    ) -> None:
        self.matrix_a = np.asarray(matrix_a)
        self.matrix_b = np.asarray(matrix_b)
        self.measurement_c = np.asarray(measurement_c)
        self.feedback_gain = np.asarray(feedback_gain)
        self.kalman_gain = np.asarray(kalman_gain)

    def reset(self, num_envs: int) -> None:
        self.estimate = np.zeros((num_envs, self.matrix_a.shape[0]), dtype=np.float64)
        self.previous_force = np.zeros((num_envs, 1), dtype=np.float64)

    def act_from_measurement(self, measurement: np.ndarray) -> np.ndarray:
        prediction = (
            self.estimate @ self.matrix_a.T + self.previous_force @ self.matrix_b.T
        )
        innovation = measurement - prediction @ self.measurement_c.T
        self.estimate = prediction + innovation @ self.kalman_gain.T
        force = -(self.estimate @ self.feedback_gain.T)
        np.clip(
            force,
            -ACTION_FORCE_SCALE_N,
            ACTION_FORCE_SCALE_N,
            out=force,
        )
        self.previous_force = force
        return force[:, 0] / ACTION_FORCE_SCALE_N
