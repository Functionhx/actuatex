"""GPU-vectorized counterparts of the classical cart-pole controllers.

The NumPy implementations remain the reference used by the MuJoCo benchmark.
These controllers preserve the same equations while keeping Isaac Lab state,
observer covariance and actions on the simulation device.
"""

from __future__ import annotations

import torch

from .contract import (
    ACTION_FORCE_SCALE_N,
    CART_MASS_KG,
    POLICY_DT,
    POLE_LENGTH_M,
    POLE_MASS_KG,
    POLE_WIDTH_M,
)


def _tensor(
    value,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


class TorchStateFeedbackController:
    """Saturated state feedback for LQR, pole placement, MPC and H-infinity."""

    def __init__(
        self,
        gain,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.gain = _tensor(gain, device=self.device, dtype=dtype)

    def reset(self, num_envs: int) -> None:
        del num_envs

    def act(self, state: torch.Tensor) -> torch.Tensor:
        force = -(state.to(self.device, self.dtype) @ self.gain.T)
        return torch.clamp(force[:, 0] / ACTION_FORCE_SCALE_N, -1.0, 1.0)


class TorchPIDController:
    """Single-pole LQR-informed PID controller."""

    def __init__(
        self,
        *,
        cart_kp: float,
        cart_kd: float,
        angle_kp: float,
        angle_ki: float,
        angle_kd: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        integral_limit: float = 0.25,
    ) -> None:
        self.cart_kp = cart_kp
        self.cart_kd = cart_kd
        self.angle_kp = angle_kp
        self.angle_ki = angle_ki
        self.angle_kd = angle_kd
        self.integral_limit = integral_limit
        self.device = torch.device(device)
        self.dtype = dtype

    def reset(self, num_envs: int) -> None:
        self.integral = torch.zeros(
            num_envs, device=self.device, dtype=self.dtype
        )

    def act(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[1] != 4:
            raise ValueError("TorchPIDController is defined for the single pole")
        state = state.to(self.device, self.dtype)
        cart_position, angle, cart_velocity, angular_velocity = state.T
        self.integral.add_(angle * POLICY_DT).clamp_(
            -self.integral_limit, self.integral_limit
        )
        force = (
            self.cart_kp * cart_position
            + self.cart_kd * cart_velocity
            + self.angle_kp * angle
            + self.angle_ki * self.integral
            + self.angle_kd * angular_velocity
        )
        return torch.clamp(force / ACTION_FORCE_SCALE_N, -1.0, 1.0)


class TorchCascadedPIDController:
    """Outer cart-position PD feeding an inner pole-angle PID."""

    def __init__(
        self,
        *,
        outer_kp: float,
        outer_kd: float,
        inner_kp: float,
        inner_ki: float,
        inner_kd: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        integral_limit: float = 0.25,
    ) -> None:
        self.outer_kp = outer_kp
        self.outer_kd = outer_kd
        self.inner_kp = inner_kp
        self.inner_ki = inner_ki
        self.inner_kd = inner_kd
        self.integral_limit = integral_limit
        self.device = torch.device(device)
        self.dtype = dtype

    def reset(self, num_envs: int) -> None:
        self.integral = torch.zeros(
            num_envs, device=self.device, dtype=self.dtype
        )

    def act(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[1] != 4:
            raise ValueError(
                "TorchCascadedPIDController is defined for the single pole"
            )
        state = state.to(self.device, self.dtype)
        cart_position, angle, cart_velocity, angular_velocity = state.T
        desired_angle = -(
            self.outer_kp * cart_position + self.outer_kd * cart_velocity
        )
        angle_error = angle - desired_angle
        self.integral.add_(angle_error * POLICY_DT).clamp_(
            -self.integral_limit, self.integral_limit
        )
        force = (
            self.inner_kp * angle_error
            + self.inner_ki * self.integral
            + self.inner_kd * angular_velocity
        )
        return torch.clamp(force / ACTION_FORCE_SCALE_N, -1.0, 1.0)


class TorchLinearOutputFeedbackController:
    """Batched Kalman/Luenberger observer followed by saturated feedback."""

    def __init__(
        self,
        matrix_a,
        matrix_b,
        measurement_c,
        feedback_gain,
        correction_gain,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        values = (matrix_a, matrix_b, measurement_c, feedback_gain, correction_gain)
        (
            self.matrix_a,
            self.matrix_b,
            self.measurement_c,
            self.feedback_gain,
            self.correction_gain,
        ) = tuple(_tensor(value, device=self.device, dtype=dtype) for value in values)

    def reset(self, num_envs: int) -> None:
        state_dim = self.matrix_a.shape[0]
        self.estimate = torch.zeros(
            (num_envs, state_dim), device=self.device, dtype=self.dtype
        )
        self.previous_force = torch.zeros(
            (num_envs, 1), device=self.device, dtype=self.dtype
        )

    def act_from_measurement(self, measurement: torch.Tensor) -> torch.Tensor:
        measurement = measurement.to(self.device, self.dtype)
        prediction = (
            self.estimate @ self.matrix_a.T
            + self.previous_force @ self.matrix_b.T
        )
        innovation = measurement - prediction @ self.measurement_c.T
        self.estimate = prediction + innovation @ self.correction_gain.T
        force = -(self.estimate @ self.feedback_gain.T)
        force.clamp_(-ACTION_FORCE_SCALE_N, ACTION_FORCE_SCALE_N)
        self.previous_force = force
        return force[:, 0] / ACTION_FORCE_SCALE_N


class TorchComplementaryFilterLQRController:
    """GPU complementary position/velocity estimator followed by LQR."""

    def __init__(
        self,
        matrix_a,
        matrix_b,
        measurement_c,
        feedback_gain,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        model_velocity_weight: float = 0.94,
    ) -> None:
        if not 0.0 <= model_velocity_weight <= 1.0:
            raise ValueError("model_velocity_weight must lie in [0, 1]")
        self.device = torch.device(device)
        self.dtype = dtype
        self.matrix_a = _tensor(matrix_a, device=self.device, dtype=dtype)
        self.matrix_b = _tensor(matrix_b, device=self.device, dtype=dtype)
        self.measurement_c = _tensor(
            measurement_c, device=self.device, dtype=dtype
        )
        self.feedback_gain = _tensor(
            feedback_gain, device=self.device, dtype=dtype
        )
        self.model_velocity_weight = model_velocity_weight
        self.dof_count = self.measurement_c.shape[0]

    def reset(self, num_envs: int) -> None:
        state_dim = self.matrix_a.shape[0]
        self.estimate = torch.zeros(
            (num_envs, state_dim), device=self.device, dtype=self.dtype
        )
        self.previous_measurement = torch.zeros(
            (num_envs, self.dof_count), device=self.device, dtype=self.dtype
        )
        self.previous_force = torch.zeros(
            (num_envs, 1), device=self.device, dtype=self.dtype
        )
        self.initialized = False

    def act_from_measurement(self, measurement: torch.Tensor) -> torch.Tensor:
        measurement = measurement.to(self.device, self.dtype)
        prediction = (
            self.estimate @ self.matrix_a.T
            + self.previous_force @ self.matrix_b.T
        )
        if self.initialized:
            measurement_delta = measurement - self.previous_measurement
            angular_delta = measurement_delta[:, 1:]
            measurement_delta = torch.cat(
                (
                    measurement_delta[:, :1],
                    torch.atan2(torch.sin(angular_delta), torch.cos(angular_delta)),
                ),
                dim=1,
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
        self.previous_measurement = measurement.clone()
        force = -(self.estimate @ self.feedback_gain.T)
        force.clamp_(-ACTION_FORCE_SCALE_N, ACTION_FORCE_SCALE_N)
        self.previous_force = force
        return force[:, 0] / ACTION_FORCE_SCALE_N


class TorchDiscreteSlidingModeController:
    """GPU implementation of the discrete reaching-law sliding controller."""

    def __init__(
        self,
        matrix_a,
        matrix_b,
        sliding_surface,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        contraction: float = 0.90,
        switching_strength: float = 0.08,
        boundary_layer: float = 0.25,
    ) -> None:
        if not 0.0 <= contraction < 1.0:
            raise ValueError("contraction must lie in [0, 1)")
        if switching_strength < 0.0 or boundary_layer <= 0.0:
            raise ValueError("sliding-mode gains must be positive")
        self.device = torch.device(device)
        self.dtype = dtype
        self.matrix_a = _tensor(matrix_a, device=self.device, dtype=dtype)
        matrix_b = _tensor(matrix_b, device=self.device, dtype=dtype)
        self.sliding_surface = _tensor(
            sliding_surface, device=self.device, dtype=dtype
        )
        denominator = float((self.sliding_surface @ matrix_b).item())
        if abs(denominator) < 1.0e-9:
            raise ValueError("sliding surface is orthogonal to the input channel")
        self.input_projection = denominator
        self.contraction = contraction
        self.switching_strength = switching_strength
        self.boundary_layer = boundary_layer

    def reset(self, num_envs: int) -> None:
        del num_envs

    def act(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device, self.dtype)
        sliding_value = (state @ self.sliding_surface.T)[:, 0]
        desired_next = (
            self.contraction * sliding_value
            - self.switching_strength
            * torch.tanh(sliding_value / self.boundary_layer)
        )
        open_loop_next = (
            state @ self.matrix_a.T @ self.sliding_surface.T
        )[:, 0]
        force = (desired_next - open_loop_next) / self.input_projection
        return torch.clamp(force / ACTION_FORCE_SCALE_N, -1.0, 1.0)


def _single_pole_mass_terms(
    theta: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    center_length = POLE_LENGTH_M / 2.0
    center_inertia = (
        POLE_MASS_KG * (POLE_WIDTH_M**2 + POLE_LENGTH_M**2) / 12.0
    )
    hinge_inertia = center_inertia + POLE_MASS_KG * center_length**2
    coupling = POLE_MASS_KG * center_length * torch.cos(theta)
    return coupling, hinge_inertia, center_length


class TorchCollocatedFeedbackLinearizationController:
    """Exact cart-acceleration input-output linearization on the GPU."""

    def __init__(
        self,
        acceleration_gain,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.acceleration_gain = _tensor(
            acceleration_gain, device=self.device, dtype=dtype
        ).reshape(-1)
        if self.acceleration_gain.shape != (4,):
            raise ValueError("single-pole acceleration gain must contain four terms")

    def reset(self, num_envs: int) -> None:
        del num_envs

    def act(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device, self.dtype)
        theta = state[:, 1]
        angular_velocity = state[:, 3]
        desired_cart_acceleration = -(state @ self.acceleration_gain)
        coupling, hinge_inertia, center_length = _single_pole_mass_terms(theta)
        pole_acceleration = (
            POLE_MASS_KG * 9.81 * center_length * torch.sin(theta)
            - coupling * desired_cart_acceleration
        ) / hinge_inertia
        force = (
            (CART_MASS_KG + POLE_MASS_KG) * desired_cart_acceleration
            + coupling * pole_acceleration
            - POLE_MASS_KG
            * center_length
            * torch.sin(theta)
            * torch.square(angular_velocity)
        )
        return torch.clamp(force / ACTION_FORCE_SCALE_N, -1.0, 1.0)


class TorchPartialFeedbackLinearizationController:
    """GPU partial feedback linearization of pole angular acceleration."""

    def __init__(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        angle_kp: float = 45.0,
        angle_kd: float = 9.0,
        cart_to_angle_kp: float = 0.08,
        cart_to_angle_kd: float = 0.16,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.angle_kp = angle_kp
        self.angle_kd = angle_kd
        self.cart_to_angle_kp = cart_to_angle_kp
        self.cart_to_angle_kd = cart_to_angle_kd

    def reset(self, num_envs: int) -> None:
        del num_envs

    def act(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device, self.dtype)
        cart_position, theta, cart_velocity, angular_velocity = state.T
        desired_angle = -(
            self.cart_to_angle_kp * cart_position
            + self.cart_to_angle_kd * cart_velocity
        )
        desired_pole_acceleration = -(
            self.angle_kp * (theta - desired_angle)
            + self.angle_kd * angular_velocity
        )
        coupling, hinge_inertia, center_length = _single_pole_mass_terms(theta)
        safe_coupling = torch.where(
            torch.abs(coupling) < 1.0e-4,
            torch.copysign(
                torch.full_like(coupling, 1.0e-4), coupling + 1.0e-12
            ),
            coupling,
        )
        desired_cart_acceleration = (
            POLE_MASS_KG * 9.81 * center_length * torch.sin(theta)
            - hinge_inertia * desired_pole_acceleration
        ) / safe_coupling
        force = (
            (CART_MASS_KG + POLE_MASS_KG) * desired_cart_acceleration
            + coupling * desired_pole_acceleration
            - POLE_MASS_KG
            * center_length
            * torch.sin(theta)
            * torch.square(angular_velocity)
        )
        return torch.clamp(force / ACTION_FORCE_SCALE_N, -1.0, 1.0)


def _single_cartpole_derivative(
    state: torch.Tensor, force: torch.Tensor
) -> torch.Tensor:
    force = force.reshape(-1)
    theta = state[:, 1]
    cart_velocity = state[:, 2]
    angular_velocity = state[:, 3]
    center_length = POLE_LENGTH_M / 2.0
    center_inertia = (
        POLE_MASS_KG * (POLE_WIDTH_M**2 + POLE_LENGTH_M**2) / 12.0
    )
    hinge_inertia = center_inertia + POLE_MASS_KG * center_length**2
    coupling = POLE_MASS_KG * center_length * torch.cos(theta)
    mass_total = CART_MASS_KG + POLE_MASS_KG
    rhs_cart = (
        force
        + POLE_MASS_KG
        * center_length
        * torch.sin(theta)
        * torch.square(angular_velocity)
    )
    rhs_pole = POLE_MASS_KG * 9.81 * center_length * torch.sin(theta)
    determinant = mass_total * hinge_inertia - torch.square(coupling)
    cart_acceleration = (
        rhs_cart * hinge_inertia - coupling * rhs_pole
    ) / determinant
    angular_acceleration = (
        mass_total * rhs_pole - coupling * rhs_cart
    ) / determinant
    return torch.stack(
        (
            cart_velocity,
            angular_velocity,
            cart_acceleration,
            angular_acceleration,
        ),
        dim=1,
    )


def nonlinear_single_cartpole_step(
    state: torch.Tensor, force: torch.Tensor
) -> torch.Tensor:
    """Batched differentiable 60 Hz RK4 model used by the torch EKF."""

    half_dt = 0.5 * POLICY_DT
    k1 = _single_cartpole_derivative(state, force)
    k2 = _single_cartpole_derivative(state + half_dt * k1, force)
    k3 = _single_cartpole_derivative(state + half_dt * k2, force)
    k4 = _single_cartpole_derivative(state + POLICY_DT * k3, force)
    result = state + (POLICY_DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    wrapped_angle = torch.atan2(torch.sin(result[:, 1]), torch.cos(result[:, 1]))
    return torch.cat((result[:, :1], wrapped_angle[:, None], result[:, 2:]), dim=1)


class TorchExtendedKalmanLQRController:
    """Batched nonlinear single-pole EKF followed by LQR feedback."""

    def __init__(
        self,
        feedback_gain,
        *,
        measurement_noise_std: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        process_noise_std: float = 0.002,
        jacobian_epsilon: float = 1.0e-5,
    ) -> None:
        if measurement_noise_std <= 0.0 or process_noise_std <= 0.0:
            raise ValueError("EKF covariance scales must be positive")
        self.device = torch.device(device)
        self.dtype = dtype
        self.feedback_gain = _tensor(
            feedback_gain, device=self.device, dtype=dtype
        )
        if self.feedback_gain.shape != (1, 4):
            raise ValueError("EKF controller is defined for the single cart-pole")
        self.measurement_c = torch.zeros(
            (2, 4), device=self.device, dtype=dtype
        )
        self.measurement_c[:, :2] = torch.eye(
            2, device=self.device, dtype=dtype
        )
        self.measurement_covariance = (
            torch.eye(2, device=self.device, dtype=dtype)
            * measurement_noise_std**2
        )
        self.process_covariance = (
            torch.eye(4, device=self.device, dtype=dtype) * process_noise_std**2
        )
        self.jacobian_epsilon = jacobian_epsilon

    def reset(self, num_envs: int) -> None:
        self.estimate = torch.zeros(
            (num_envs, 4), device=self.device, dtype=self.dtype
        )
        self.covariance = (
            torch.eye(4, device=self.device, dtype=self.dtype)[None]
            .repeat(num_envs, 1, 1)
            .mul_(0.05)
        )
        self.previous_force = torch.zeros(
            num_envs, device=self.device, dtype=self.dtype
        )
        self.initialized = False

    def _transition_jacobian(
        self, state: torch.Tensor, force: torch.Tensor
    ) -> torch.Tensor:
        columns = []
        epsilon = self.jacobian_epsilon
        for column in range(4):
            offset = torch.zeros_like(state)
            offset[:, column] = epsilon
            columns.append(
                (
                    nonlinear_single_cartpole_step(state + offset, force)
                    - nonlinear_single_cartpole_step(state - offset, force)
                )
                / (2.0 * epsilon)
            )
        return torch.stack(columns, dim=-1)

    def act_from_measurement(self, measurement: torch.Tensor) -> torch.Tensor:
        measurement = measurement.to(self.device, self.dtype)
        if not self.initialized:
            self.estimate[:, :2] = measurement
            self.initialized = True
        transition = self._transition_jacobian(
            self.estimate, self.previous_force
        )
        prediction = nonlinear_single_cartpole_step(
            self.estimate, self.previous_force
        )
        predicted_covariance = (
            transition @ self.covariance @ transition.transpose(-1, -2)
            + self.process_covariance
        )
        innovation = measurement - prediction[:, :2]
        wrapped_angle = torch.atan2(
            torch.sin(innovation[:, 1]), torch.cos(innovation[:, 1])
        )
        innovation = torch.stack((innovation[:, 0], wrapped_angle), dim=1)
        measurement_c = self.measurement_c.expand(
            measurement.shape[0], -1, -1
        )
        innovation_covariance = (
            measurement_c
            @ predicted_covariance
            @ measurement_c.transpose(-1, -2)
            + self.measurement_covariance
        )
        kalman_gain = (
            predicted_covariance
            @ measurement_c.transpose(-1, -2)
            @ torch.linalg.inv(innovation_covariance)
        )
        self.estimate = prediction + (
            kalman_gain @ innovation.unsqueeze(-1)
        ).squeeze(-1)
        identity = torch.eye(4, device=self.device, dtype=self.dtype).expand(
            measurement.shape[0], -1, -1
        )
        self.covariance = (
            identity - kalman_gain @ measurement_c
        ) @ predicted_covariance
        force = -(self.estimate @ self.feedback_gain.T)[:, 0]
        force.clamp_(-ACTION_FORCE_SCALE_N, ACTION_FORCE_SCALE_N)
        self.previous_force = force
        return force / ACTION_FORCE_SCALE_N
