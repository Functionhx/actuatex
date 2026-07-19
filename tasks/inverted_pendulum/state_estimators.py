"""State estimators paired with LQR for the cart-pole noise benchmark."""

from __future__ import annotations

import numpy as np
from scipy.signal import place_poles

from .contract import (
    ACTION_FORCE_SCALE_N,
    CART_MASS_KG,
    POLICY_DT,
    POLE_LENGTH_M,
    POLE_MASS_KG,
    POLE_WIDTH_M,
)


def design_luenberger_gain(
    matrix_a: np.ndarray,
    measurement_c: np.ndarray,
    *,
    fastest_rate: float,
    slowest_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Place distinct discrete observer poles through the dual system."""

    state_dim = matrix_a.shape[0]
    continuous_rates = -np.geomspace(slowest_rate, fastest_rate, state_dim)
    desired_poles = np.exp(continuous_rates * POLICY_DT)
    result = place_poles(
        matrix_a.T,
        measurement_c.T,
        desired_poles,
        method="YT",
    )
    return np.asarray(result.gain_matrix).T, desired_poles


class LinearObserverLQRController:
    """Luenberger state observer followed by saturated LQR feedback."""

    def __init__(
        self,
        matrix_a: np.ndarray,
        matrix_b: np.ndarray,
        measurement_c: np.ndarray,
        feedback_gain: np.ndarray,
        observer_gain: np.ndarray,
    ) -> None:
        self.matrix_a = np.asarray(matrix_a, dtype=np.float64)
        self.matrix_b = np.asarray(matrix_b, dtype=np.float64)
        self.measurement_c = np.asarray(measurement_c, dtype=np.float64)
        self.feedback_gain = np.asarray(feedback_gain, dtype=np.float64)
        self.observer_gain = np.asarray(observer_gain, dtype=np.float64)

    def reset(self, num_envs: int) -> None:
        self.estimate = np.zeros((num_envs, self.matrix_a.shape[0]), dtype=np.float64)
        self.previous_force = np.zeros((num_envs, 1), dtype=np.float64)

    def act_from_measurement(self, measurement: np.ndarray) -> np.ndarray:
        prediction = (
            self.estimate @ self.matrix_a.T + self.previous_force @ self.matrix_b.T
        )
        innovation = measurement - prediction @ self.measurement_c.T
        self.estimate = prediction + innovation @ self.observer_gain.T
        force = -(self.estimate @ self.feedback_gain.T)
        np.clip(force, -ACTION_FORCE_SCALE_N, ACTION_FORCE_SCALE_N, out=force)
        self.previous_force = force
        return force[:, 0] / ACTION_FORCE_SCALE_N


class ComplementaryFilterLQRController:
    """Fuse encoder finite differences with model-predicted velocity."""

    def __init__(
        self,
        matrix_a: np.ndarray,
        matrix_b: np.ndarray,
        measurement_c: np.ndarray,
        feedback_gain: np.ndarray,
        *,
        model_velocity_weight: float = 0.94,
    ) -> None:
        if not 0.0 <= model_velocity_weight <= 1.0:
            raise ValueError("model_velocity_weight must lie in [0, 1]")
        self.matrix_a = np.asarray(matrix_a, dtype=np.float64)
        self.matrix_b = np.asarray(matrix_b, dtype=np.float64)
        self.measurement_c = np.asarray(measurement_c, dtype=np.float64)
        self.feedback_gain = np.asarray(feedback_gain, dtype=np.float64)
        self.model_velocity_weight = model_velocity_weight
        self.dof_count = self.measurement_c.shape[0]

    def reset(self, num_envs: int) -> None:
        self.estimate = np.zeros((num_envs, self.matrix_a.shape[0]), dtype=np.float64)
        self.previous_measurement = np.zeros(
            (num_envs, self.dof_count), dtype=np.float64
        )
        self.previous_force = np.zeros((num_envs, 1), dtype=np.float64)
        self.initialized = False

    def act_from_measurement(self, measurement: np.ndarray) -> np.ndarray:
        measurement = np.asarray(measurement, dtype=np.float64)
        prediction = (
            self.estimate @ self.matrix_a.T + self.previous_force @ self.matrix_b.T
        )
        if self.initialized:
            measurement_delta = measurement - self.previous_measurement
            measurement_delta[:, 1:] = np.arctan2(
                np.sin(measurement_delta[:, 1:]),
                np.cos(measurement_delta[:, 1:]),
            )
            differentiated_velocity = measurement_delta / POLICY_DT
            velocity = (
                self.model_velocity_weight * prediction[:, self.dof_count :]
                + (1.0 - self.model_velocity_weight) * differentiated_velocity
            )
        else:
            velocity = prediction[:, self.dof_count :]
            self.initialized = True
        self.estimate[:, : self.dof_count] = measurement
        self.estimate[:, self.dof_count :] = velocity
        self.previous_measurement = measurement.copy()
        force = -(self.estimate @ self.feedback_gain.T)
        np.clip(force, -ACTION_FORCE_SCALE_N, ACTION_FORCE_SCALE_N, out=force)
        self.previous_force = force
        return force[:, 0] / ACTION_FORCE_SCALE_N


def _single_cartpole_derivative(state: np.ndarray, force: np.ndarray) -> np.ndarray:
    """Continuous nonlinear dynamics for the physical single rod."""

    state = np.asarray(state, dtype=np.float64)
    force = np.asarray(force, dtype=np.float64).reshape(-1)
    theta = state[:, 1]
    cart_velocity = state[:, 2]
    angular_velocity = state[:, 3]
    center_length = POLE_LENGTH_M / 2.0
    center_inertia = POLE_MASS_KG * (POLE_WIDTH_M**2 + POLE_LENGTH_M**2) / 12.0
    hinge_inertia = center_inertia + POLE_MASS_KG * center_length**2
    coupling = POLE_MASS_KG * center_length * np.cos(theta)
    mass_total = CART_MASS_KG + POLE_MASS_KG
    rhs_cart = force + POLE_MASS_KG * center_length * np.sin(theta) * np.square(
        angular_velocity
    )
    rhs_pole = POLE_MASS_KG * 9.81 * center_length * np.sin(theta)
    determinant = mass_total * hinge_inertia - np.square(coupling)
    cart_acceleration = (rhs_cart * hinge_inertia - coupling * rhs_pole) / determinant
    angular_acceleration = (mass_total * rhs_pole - coupling * rhs_cart) / determinant
    return np.column_stack(
        (cart_velocity, angular_velocity, cart_acceleration, angular_acceleration)
    )


def nonlinear_single_cartpole_step(state: np.ndarray, force: np.ndarray) -> np.ndarray:
    """One 60 Hz RK4 step used by the EKF prediction model."""

    state = np.asarray(state, dtype=np.float64)
    force = np.asarray(force, dtype=np.float64).reshape(-1)
    half_dt = 0.5 * POLICY_DT
    k1 = _single_cartpole_derivative(state, force)
    k2 = _single_cartpole_derivative(state + half_dt * k1, force)
    k3 = _single_cartpole_derivative(state + half_dt * k2, force)
    k4 = _single_cartpole_derivative(state + POLICY_DT * k3, force)
    result = state + (POLICY_DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    result[:, 1] = np.arctan2(np.sin(result[:, 1]), np.cos(result[:, 1]))
    return result


class ExtendedKalmanLQRController:
    """Nonlinear single-pole EKF using position encoders and LQR feedback."""

    def __init__(
        self,
        feedback_gain: np.ndarray,
        *,
        measurement_noise_std: float,
        process_noise_std: float = 0.002,
        jacobian_epsilon: float = 1.0e-5,
    ) -> None:
        gain = np.asarray(feedback_gain, dtype=np.float64)
        if gain.shape != (1, 4):
            raise ValueError("EKF controller is defined for the single cart-pole")
        if measurement_noise_std <= 0.0 or process_noise_std <= 0.0:
            raise ValueError("EKF covariance scales must be positive")
        self.feedback_gain = gain
        self.measurement_c = np.zeros((2, 4), dtype=np.float64)
        self.measurement_c[:, :2] = np.eye(2)
        self.measurement_covariance = (
            np.eye(2, dtype=np.float64) * measurement_noise_std**2
        )
        self.process_covariance = np.eye(4, dtype=np.float64) * process_noise_std**2
        self.jacobian_epsilon = jacobian_epsilon

    def reset(self, num_envs: int) -> None:
        self.estimate = np.zeros((num_envs, 4), dtype=np.float64)
        self.covariance = np.repeat(
            (np.eye(4, dtype=np.float64) * 0.05)[None, :, :], num_envs, axis=0
        )
        self.previous_force = np.zeros(num_envs, dtype=np.float64)
        self.initialized = False

    def _transition_jacobian(self, state: np.ndarray, force: np.ndarray) -> np.ndarray:
        batch_size = state.shape[0]
        jacobian = np.empty((batch_size, 4, 4), dtype=np.float64)
        epsilon = self.jacobian_epsilon
        for column in range(4):
            positive = state.copy()
            negative = state.copy()
            positive[:, column] += epsilon
            negative[:, column] -= epsilon
            jacobian[:, :, column] = (
                nonlinear_single_cartpole_step(positive, force)
                - nonlinear_single_cartpole_step(negative, force)
            ) / (2.0 * epsilon)
        return jacobian

    def act_from_measurement(self, measurement: np.ndarray) -> np.ndarray:
        measurement = np.asarray(measurement, dtype=np.float64)
        if not self.initialized:
            self.estimate[:, :2] = measurement
            self.initialized = True
        transition = self._transition_jacobian(self.estimate, self.previous_force)
        prediction = nonlinear_single_cartpole_step(self.estimate, self.previous_force)
        predicted_covariance = (
            transition @ self.covariance @ np.swapaxes(transition, 1, 2)
            + self.process_covariance
        )
        innovation = measurement - prediction[:, :2]
        innovation[:, 1] = np.arctan2(
            np.sin(innovation[:, 1]), np.cos(innovation[:, 1])
        )
        innovation_covariance = (
            self.measurement_c @ predicted_covariance @ self.measurement_c.T
            + self.measurement_covariance
        )
        kalman_gain = (
            predicted_covariance
            @ self.measurement_c.T
            @ np.linalg.inv(innovation_covariance)
        )
        self.estimate = prediction + np.einsum("nij,nj->ni", kalman_gain, innovation)
        identity = np.eye(4, dtype=np.float64)[None, :, :]
        self.covariance = (
            identity - kalman_gain @ self.measurement_c
        ) @ predicted_covariance
        force = -(self.estimate @ self.feedback_gain.T)[:, 0]
        np.clip(force, -ACTION_FORCE_SCALE_N, ACTION_FORCE_SCALE_N, out=force)
        self.previous_force = force
        return force / ACTION_FORCE_SCALE_N
