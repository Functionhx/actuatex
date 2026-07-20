"""Shared local system identification and linear control for Sentinel."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import block_diag, solve_discrete_are
import torch
from torch import nn

from .contract import ACTION_DIM
from .locomotion import (
    BASE_ANGULAR_VELOCITY_SLICE,
    BASE_LINEAR_VELOCITY_SLICE,
    COMMAND_OBSERVATION_SLICE,
    JOINT_VELOCITY_OBSERVATION_SLICE,
    LEG_POSITION_OBSERVATION_SLICE,
    OBSERVATION_DIM,
    PREVIOUS_ACTION_OBSERVATION_SLICE,
    PROJECTED_GRAVITY_SLICE,
    TRACK_WIDTH_M,
    WHEEL_ACTION_SCALE_RADPS,
    WHEEL_RADIUS_M,
)


LINEAR_CONTROLLER_CHECKPOINT_FORMAT = "actuatex_linear_feedback_v1"
SCHEDULED_LINEAR_CONTROLLER_CHECKPOINT_FORMAT = (
    "actuatex_scheduled_linear_feedback_v1"
)
CONTROL_STATE_NAMES = (
    "vx_error",
    "vy_error",
    "vertical_velocity",
    "roll_rate",
    "pitch_rate",
    "yaw_rate_error",
    "projected_gravity_x",
    "projected_gravity_y",
    "left_hip_position_error",
    "left_knee_position_error",
    "right_hip_position_error",
    "right_knee_position_error",
    "left_hip_velocity_scaled",
    "left_knee_velocity_scaled",
    "right_hip_velocity_scaled",
    "right_knee_velocity_scaled",
    "left_wheel_velocity_scaled",
    "right_wheel_velocity_scaled",
    "previous_left_hip_action_error",
    "previous_left_knee_action_error",
    "previous_right_hip_action_error",
    "previous_right_knee_action_error",
    "previous_left_wheel_action_error",
    "previous_right_wheel_action_error",
)
CONTROL_STATE_DIM = len(CONTROL_STATE_NAMES)
DEFAULT_STATE_COST_DIAGONAL = np.asarray(
    [
        10.0,
        4.0,
        3.0,
        2.0,
        8.0,
        5.0,
        60.0,
        60.0,
        8.0,
        8.0,
        8.0,
        8.0,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
    ],
    dtype=np.float64,
)
DEFAULT_ACTION_COST_DIAGONAL = np.asarray(
    [1.5, 1.5, 1.5, 1.5, 0.5, 0.5],
    dtype=np.float64,
)


def _sentinel_mirror_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Return signed permutations for reflection across the sagittal plane."""

    state = np.zeros((CONTROL_STATE_DIM, CONTROL_STATE_DIM), dtype=np.float64)
    for index, sign in enumerate((1, -1, 1, -1, 1, -1, 1, -1)):
        state[index, index] = sign
    for left, right in (
        (8, 10),
        (9, 11),
        (12, 14),
        (13, 15),
        (16, 17),
        (18, 20),
        (19, 21),
        (22, 23),
    ):
        state[left, right] = 1.0
        state[right, left] = 1.0

    action = np.zeros((ACTION_DIM, ACTION_DIM), dtype=np.float64)
    for left, right in ((0, 2), (1, 3), (4, 5)):
        action[left, right] = 1.0
        action[right, left] = 1.0
    state.setflags(write=False)
    action.setflags(write=False)
    return state, action


LEFT_RIGHT_STATE_MIRROR, LEFT_RIGHT_ACTION_MIRROR = (
    _sentinel_mirror_matrices()
)


def symmetrize_left_right_dynamics(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project an identified model onto Sentinel's exact mirror symmetry."""

    matrix_a = np.asarray(matrix_a, dtype=np.float64)
    matrix_b = np.asarray(matrix_b, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if matrix_a.shape != (CONTROL_STATE_DIM, CONTROL_STATE_DIM):
        raise ValueError("A has an invalid shape for Sentinel symmetry")
    if matrix_b.shape != (CONTROL_STATE_DIM, ACTION_DIM):
        raise ValueError("B has an invalid shape for Sentinel symmetry")
    if bias.shape != (CONTROL_STATE_DIM,):
        raise ValueError("bias has an invalid shape for Sentinel symmetry")
    if not all(
        np.isfinite(values).all() for values in (matrix_a, matrix_b, bias)
    ):
        raise ValueError("dynamics contain a non-finite value")
    state_mirror = LEFT_RIGHT_STATE_MIRROR
    action_mirror = LEFT_RIGHT_ACTION_MIRROR
    return (
        0.5 * (matrix_a + state_mirror @ matrix_a @ state_mirror),
        0.5 * (matrix_b + state_mirror @ matrix_b @ action_mirror),
        0.5 * (bias + state_mirror @ bias),
    )


def _numpy_last_axis(
    values: np.ndarray,
    size: int,
    label: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 0 or result.shape[-1] != size:
        raise ValueError(
            f"{label} must have last dimension {size}, got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains a non-finite value")
    return result


def control_state_from_observation(observation: np.ndarray) -> np.ndarray:
    """Extract the normalized, command-relative local control state."""

    obs = _numpy_last_axis(observation, OBSERVATION_DIM, "observation")
    command = obs[..., COMMAND_OBSERVATION_SLICE]
    linear_velocity_error = obs[..., BASE_LINEAR_VELOCITY_SLICE].copy()
    linear_velocity_error[..., :2] -= command[..., :2]
    angular_velocity_error = obs[..., BASE_ANGULAR_VELOCITY_SLICE].copy()
    angular_velocity_error[..., 2] -= command[..., 2]
    previous_action_error = obs[
        ..., PREVIOUS_ACTION_OBSERVATION_SLICE
    ].copy()
    previous_action_error -= feedforward_action_from_command(command)
    state = np.concatenate(
        (
            linear_velocity_error,
            angular_velocity_error,
            obs[..., PROJECTED_GRAVITY_SLICE.start : 8],
            obs[..., LEG_POSITION_OBSERVATION_SLICE],
            obs[..., JOINT_VELOCITY_OBSERVATION_SLICE],
            previous_action_error,
        ),
        axis=-1,
    )
    if state.shape[-1] != CONTROL_STATE_DIM:
        raise AssertionError(f"control state dimension drifted to {state.shape}")
    return state


def torch_control_state_from_observation(
    observation: torch.Tensor,
) -> torch.Tensor:
    """Torch equivalent of :func:`control_state_from_observation`."""

    if observation.ndim == 0 or observation.shape[-1] != OBSERVATION_DIM:
        raise ValueError(
            "observation must have last dimension "
            f"{OBSERVATION_DIM}, got {tuple(observation.shape)}"
        )
    command = observation[..., COMMAND_OBSERVATION_SLICE]
    linear_velocity_error = observation[..., BASE_LINEAR_VELOCITY_SLICE].clone()
    linear_velocity_error[..., :2] -= command[..., :2]
    angular_velocity_error = observation[
        ..., BASE_ANGULAR_VELOCITY_SLICE
    ].clone()
    angular_velocity_error[..., 2] -= command[..., 2]
    previous_action_error = observation[
        ..., PREVIOUS_ACTION_OBSERVATION_SLICE
    ].clone()
    previous_action_error -= torch_feedforward_action_from_command(command)
    return torch.cat(
        (
            linear_velocity_error,
            angular_velocity_error,
            observation[..., PROJECTED_GRAVITY_SLICE.start : 8],
            observation[..., LEG_POSITION_OBSERVATION_SLICE],
            observation[..., JOINT_VELOCITY_OBSERVATION_SLICE],
            previous_action_error,
        ),
        dim=-1,
    )


def feedforward_action_from_command(command: np.ndarray) -> np.ndarray:
    """Convert desired chassis twist to nominal differential-wheel actions."""

    command = _numpy_last_axis(command, 3, "command")
    action = np.zeros(command.shape[:-1] + (ACTION_DIM,), dtype=np.float64)
    left_velocity = (
        command[..., 0] - 0.5 * TRACK_WIDTH_M * command[..., 2]
    ) / WHEEL_RADIUS_M
    right_velocity = (
        command[..., 0] + 0.5 * TRACK_WIDTH_M * command[..., 2]
    ) / WHEEL_RADIUS_M
    action[..., 4] = left_velocity / WHEEL_ACTION_SCALE_RADPS
    action[..., 5] = right_velocity / WHEEL_ACTION_SCALE_RADPS
    return np.clip(action, -1.0, 1.0)


def torch_feedforward_action_from_command(
    command: torch.Tensor,
) -> torch.Tensor:
    if command.ndim == 0 or command.shape[-1] != 3:
        raise ValueError(
            f"command must have last dimension 3, got {tuple(command.shape)}"
        )
    action = torch.zeros(
        command.shape[:-1] + (ACTION_DIM,),
        dtype=command.dtype,
        device=command.device,
    )
    action[..., 4] = (
        command[..., 0] - 0.5 * TRACK_WIDTH_M * command[..., 2]
    ) / (WHEEL_RADIUS_M * WHEEL_ACTION_SCALE_RADPS)
    action[..., 5] = (
        command[..., 0] + 0.5 * TRACK_WIDTH_M * command[..., 2]
    ) / (WHEEL_RADIUS_M * WHEEL_ACTION_SCALE_RADPS)
    return torch.clamp(action, -1.0, 1.0)


@dataclass(frozen=True)
class DynamicsFitMetrics:
    sample_count: int
    state_dim: int
    action_dim: int
    feature_rank: int
    feature_condition_number: float
    aggregate_rmse: float
    normalized_rmse: float
    coefficient_of_determination: float
    maximum_absolute_error: float
    rmse_per_state: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "feature_rank": self.feature_rank,
            "feature_condition_number": self.feature_condition_number,
            "aggregate_rmse": self.aggregate_rmse,
            "normalized_rmse": self.normalized_rmse,
            "coefficient_of_determination": self.coefficient_of_determination,
            "maximum_absolute_error": self.maximum_absolute_error,
            "rmse_per_state": self.rmse_per_state.tolist(),
        }


@dataclass(frozen=True)
class AffineDynamicsFit:
    """One-step model ``x_next = A x + B u + bias``."""

    matrix_a: np.ndarray
    matrix_b: np.ndarray
    bias: np.ndarray
    metrics: DynamicsFitMetrics

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        state = _numpy_last_axis(state, self.matrix_a.shape[1], "state")
        action = _numpy_last_axis(action, self.matrix_b.shape[1], "action")
        if state.shape[:-1] != action.shape[:-1]:
            raise ValueError("state and action leading shapes must match")
        return (
            state @ self.matrix_a.T
            + action @ self.matrix_b.T
            + self.bias
        )


def evaluate_affine_dynamics(
    fit: AffineDynamicsFit,
    state: np.ndarray,
    action: np.ndarray,
    next_state: np.ndarray,
) -> DynamicsFitMetrics:
    state = _numpy_last_axis(state, fit.matrix_a.shape[1], "state")
    action = _numpy_last_axis(action, fit.matrix_b.shape[1], "action")
    target = _numpy_last_axis(next_state, fit.matrix_a.shape[0], "next_state")
    prediction = fit.predict(state, action)
    if prediction.shape != target.shape:
        raise ValueError("predicted and target next-state shapes must match")
    error = prediction - target
    rmse_per_state = np.sqrt(np.mean(np.square(error), axis=0))
    aggregate_rmse = float(np.sqrt(np.mean(np.square(error))))
    target_std = np.std(target, axis=0)
    active = target_std > 1.0e-8
    normalized_rmse = float(
        np.mean(rmse_per_state[active] / target_std[active])
        if np.any(active)
        else 0.0
    )
    centered = target - np.mean(target, axis=0)
    total_square = float(np.sum(np.square(centered)))
    residual_square = float(np.sum(np.square(error)))
    r_squared = (
        1.0 - residual_square / total_square
        if total_square > 1.0e-12
        else float(residual_square <= 1.0e-12)
    )
    features = np.concatenate(
        (state, action, np.ones((state.shape[0], 1))),
        axis=1,
    )
    return DynamicsFitMetrics(
        sample_count=state.shape[0],
        state_dim=state.shape[1],
        action_dim=action.shape[1],
        feature_rank=int(np.linalg.matrix_rank(features)),
        feature_condition_number=float(np.linalg.cond(features)),
        aggregate_rmse=aggregate_rmse,
        normalized_rmse=normalized_rmse,
        coefficient_of_determination=r_squared,
        maximum_absolute_error=float(np.max(np.abs(error))),
        rmse_per_state=rmse_per_state,
    )


def fit_affine_dynamics(
    state: np.ndarray,
    action: np.ndarray,
    next_state: np.ndarray,
    *,
    ridge: float = 1.0e-6,
) -> AffineDynamicsFit:
    """Fit a local one-step model with an unregularized affine intercept."""

    state = _numpy_last_axis(state, CONTROL_STATE_DIM, "state")
    action = _numpy_last_axis(action, ACTION_DIM, "action")
    next_state = _numpy_last_axis(
        next_state, CONTROL_STATE_DIM, "next_state"
    )
    if state.ndim != 2 or action.ndim != 2 or next_state.ndim != 2:
        raise ValueError("dynamics samples must be two-dimensional matrices")
    if not state.shape[0] == action.shape[0] == next_state.shape[0]:
        raise ValueError("dynamics sample counts must match")
    feature_dim = CONTROL_STATE_DIM + ACTION_DIM + 1
    if state.shape[0] < feature_dim:
        raise ValueError(
            f"at least {feature_dim} samples are required for local dynamics"
        )
    if not math_is_non_negative_finite(ridge):
        raise ValueError("ridge must be finite and non-negative")

    features = np.concatenate(
        (state, action, np.ones((state.shape[0], 1))),
        axis=1,
    )
    gram = features.T @ features
    regularization = ridge * np.eye(feature_dim)
    regularization[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        gram + regularization,
        features.T @ next_state,
    )
    fit = AffineDynamicsFit(
        matrix_a=coefficients[:CONTROL_STATE_DIM].T,
        matrix_b=coefficients[
            CONTROL_STATE_DIM : CONTROL_STATE_DIM + ACTION_DIM
        ].T,
        bias=coefficients[-1],
        metrics=DynamicsFitMetrics(
            sample_count=0,
            state_dim=CONTROL_STATE_DIM,
            action_dim=ACTION_DIM,
            feature_rank=0,
            feature_condition_number=0.0,
            aggregate_rmse=0.0,
            normalized_rmse=0.0,
            coefficient_of_determination=0.0,
            maximum_absolute_error=0.0,
            rmse_per_state=np.zeros(CONTROL_STATE_DIM),
        ),
    )
    metrics = evaluate_affine_dynamics(fit, state, action, next_state)
    return AffineDynamicsFit(fit.matrix_a, fit.matrix_b, fit.bias, metrics)


def math_is_non_negative_finite(value: float) -> bool:
    return bool(np.isfinite(value) and value >= 0.0)


def controllability_rank(matrix_a: np.ndarray, matrix_b: np.ndarray) -> int:
    matrix_a = np.asarray(matrix_a, dtype=np.float64)
    matrix_b = np.asarray(matrix_b, dtype=np.float64)
    if (
        matrix_a.ndim != 2
        or matrix_a.shape[0] != matrix_a.shape[1]
        or matrix_b.ndim != 2
        or matrix_b.shape[0] != matrix_a.shape[0]
    ):
        raise ValueError("A must be square and B must share its row count")
    blocks = [matrix_b]
    propagated = matrix_b
    for _ in range(1, matrix_a.shape[0]):
        propagated = matrix_a @ propagated
        blocks.append(propagated)
    return int(np.linalg.matrix_rank(np.concatenate(blocks, axis=1)))


@dataclass(frozen=True)
class LqrDesign:
    gain: np.ndarray
    riccati: np.ndarray
    closed_loop_eigenvalues: np.ndarray
    controllability_rank: int


@dataclass(frozen=True)
class HInfinityDesign:
    """Certified discrete-time state-feedback game solution."""

    gain: np.ndarray
    riccati: np.ndarray
    closed_loop_eigenvalues: np.ndarray
    gamma: float
    minimum_feasible_gamma_upper_bound: float
    certified_hinf_norm: float
    peak_frequency_rad_per_sample: float
    disturbance_input: np.ndarray
    control_hessian_min_eigenvalue: float
    disturbance_hessian_max_eigenvalue: float
    controllability_rank: int


def design_discrete_lqr(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    state_cost: np.ndarray,
    action_cost: np.ndarray,
) -> LqrDesign:
    """Synthesize saturated state feedback; saturation is applied by the actor."""

    matrix_a = np.asarray(matrix_a, dtype=np.float64)
    matrix_b = np.asarray(matrix_b, dtype=np.float64)
    state_cost = np.asarray(state_cost, dtype=np.float64)
    action_cost = np.asarray(action_cost, dtype=np.float64)
    state_dim = matrix_a.shape[0]
    action_dim = matrix_b.shape[1]
    if matrix_a.shape != (state_dim, state_dim):
        raise ValueError("A must be square")
    if matrix_b.shape != (state_dim, action_dim):
        raise ValueError("B has an invalid shape")
    if state_cost.shape != (state_dim, state_dim):
        raise ValueError("Q has an invalid shape")
    if action_cost.shape != (action_dim, action_dim):
        raise ValueError("R has an invalid shape")
    if not all(
        np.isfinite(values).all()
        for values in (matrix_a, matrix_b, state_cost, action_cost)
    ):
        raise ValueError("LQR matrices must be finite")
    riccati = solve_discrete_are(
        matrix_a,
        matrix_b,
        state_cost,
        action_cost,
    )
    gain = np.linalg.solve(
        action_cost + matrix_b.T @ riccati @ matrix_b,
        matrix_b.T @ riccati @ matrix_a,
    )
    eigenvalues = np.linalg.eigvals(matrix_a - matrix_b @ gain)
    return LqrDesign(
        gain=gain,
        riccati=riccati,
        closed_loop_eigenvalues=eigenvalues,
        controllability_rank=controllability_rank(matrix_a, matrix_b),
    )


def _symmetric_square_root(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    tolerance = 1.0e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError("performance cost must be positive semidefinite")
    return (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ (
        eigenvectors.T
    )


def discrete_hinf_frequency_norm(
    matrix_a_closed_loop: np.ndarray,
    disturbance_input: np.ndarray,
    performance_output: np.ndarray,
    *,
    grid_size: int = 4097,
) -> tuple[float, float]:
    """Estimate the discrete H-infinity norm on a dense unit-circle grid."""

    matrix_a_closed_loop = np.asarray(
        matrix_a_closed_loop, dtype=np.float64
    )
    disturbance_input = np.asarray(disturbance_input, dtype=np.float64)
    performance_output = np.asarray(performance_output, dtype=np.float64)
    state_dim = matrix_a_closed_loop.shape[0]
    if matrix_a_closed_loop.shape != (state_dim, state_dim):
        raise ValueError("closed-loop A must be square")
    if disturbance_input.ndim != 2 or disturbance_input.shape[0] != state_dim:
        raise ValueError("disturbance input has an invalid shape")
    if performance_output.ndim != 2 or performance_output.shape[1] != state_dim:
        raise ValueError("performance output has an invalid shape")
    if grid_size < 33 or grid_size % 2 == 0:
        raise ValueError("frequency grid size must be an odd integer >= 33")
    if not all(
        np.isfinite(values).all()
        for values in (
            matrix_a_closed_loop,
            disturbance_input,
            performance_output,
        )
    ):
        raise ValueError("H-infinity frequency model must be finite")
    if float(max(abs(np.linalg.eigvals(matrix_a_closed_loop)))) >= 1.0:
        raise ValueError("H-infinity norm is undefined for an unstable system")

    identity = np.eye(state_dim)
    peak_gain = 0.0
    peak_frequency = 0.0
    for frequency in np.linspace(0.0, np.pi, grid_size):
        transfer = performance_output @ np.linalg.solve(
            np.exp(1j * frequency) * identity - matrix_a_closed_loop,
            disturbance_input,
        )
        maximum_singular_value = float(
            np.linalg.svd(transfer, compute_uv=False)[0]
        )
        if maximum_singular_value > peak_gain:
            peak_gain = maximum_singular_value
            peak_frequency = float(frequency)
    return peak_gain, peak_frequency


def design_discrete_hinf_state_feedback(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    disturbance_input: np.ndarray,
    state_cost: np.ndarray,
    action_cost: np.ndarray,
    *,
    gamma_lower_bound: float = 0.05,
    gamma_upper_bound: float = 100.0,
    gamma_safety_factor: float = 1.25,
    bisection_iterations: int = 28,
    frequency_grid_size: int = 4097,
) -> HInfinityDesign:
    """Solve a discrete zero-sum game Riccati equation and certify it.

    The plant is ``x+ = A x + B u + Bw w`` and the infinite-horizon game
    penalizes ``x'Qx + u'Ru - gamma^2 w'w``.  The returned controller uses
    ``u = -Kx``.  A valid solution must satisfy the control/disturbance saddle
    Hessian signs, Schur stability, and a dense frequency-domain bounded-real
    check for ``z = [sqrt(Q)x, sqrt(R)u]``.
    """

    matrix_a = np.asarray(matrix_a, dtype=np.float64)
    matrix_b = np.asarray(matrix_b, dtype=np.float64)
    disturbance_input = np.asarray(disturbance_input, dtype=np.float64)
    state_cost = np.asarray(state_cost, dtype=np.float64)
    action_cost = np.asarray(action_cost, dtype=np.float64)
    state_dim = matrix_a.shape[0]
    action_dim = matrix_b.shape[1] if matrix_b.ndim == 2 else 0
    disturbance_dim = (
        disturbance_input.shape[1] if disturbance_input.ndim == 2 else 0
    )
    expected_shapes = (
        matrix_a.shape == (state_dim, state_dim),
        matrix_b.shape == (state_dim, action_dim),
        disturbance_input.shape == (state_dim, disturbance_dim),
        state_cost.shape == (state_dim, state_dim),
        action_cost.shape == (action_dim, action_dim),
    )
    if state_dim == 0 or action_dim == 0 or disturbance_dim == 0:
        raise ValueError("H-infinity model dimensions must be nonzero")
    if not all(expected_shapes):
        raise ValueError("H-infinity matrices have incompatible shapes")
    if not all(
        np.isfinite(values).all()
        for values in (
            matrix_a,
            matrix_b,
            disturbance_input,
            state_cost,
            action_cost,
        )
    ):
        raise ValueError("H-infinity matrices must be finite")
    _symmetric_square_root(state_cost)
    action_square_root = _symmetric_square_root(action_cost)
    if float(np.min(np.linalg.eigvalsh(action_cost))) <= 0.0:
        raise ValueError("H-infinity action cost must be positive definite")
    bounds_are_valid = (
        np.isfinite(gamma_lower_bound)
        and np.isfinite(gamma_upper_bound)
        and 0.0 < gamma_lower_bound < gamma_upper_bound
    )
    if not bounds_are_valid:
        raise ValueError("H-infinity gamma bounds are invalid")
    if not np.isfinite(gamma_safety_factor) or gamma_safety_factor < 1.0:
        raise ValueError("H-infinity gamma safety factor must be at least one")
    if bisection_iterations < 1:
        raise ValueError("H-infinity bisection iterations must be positive")

    augmented_input = np.hstack((matrix_b, disturbance_input))
    identity_disturbance = np.eye(disturbance_dim)

    def solve_game(gamma: float) -> tuple | None:
        game_cost = block_diag(
            action_cost,
            -(gamma**2) * identity_disturbance,
        )
        try:
            riccati = solve_discrete_are(
                matrix_a,
                augmented_input,
                state_cost,
                game_cost,
            )
            riccati = 0.5 * (riccati + riccati.T)
            game_hessian = (
                game_cost + augmented_input.T @ riccati @ augmented_input
            )
            game_gain = np.linalg.solve(
                game_hessian,
                augmented_input.T @ riccati @ matrix_a,
            )
        except (np.linalg.LinAlgError, ValueError):
            return None
        gain = game_gain[:action_dim]
        control_hessian = game_hessian[:action_dim, :action_dim]
        cross_hessian = game_hessian[:action_dim, action_dim:]
        disturbance_hessian = game_hessian[action_dim:, action_dim:]
        try:
            reduced_control_hessian = control_hessian - (
                cross_hessian
                @ np.linalg.solve(disturbance_hessian, cross_hessian.T)
            )
        except np.linalg.LinAlgError:
            return None
        disturbance_max = float(
            np.max(np.linalg.eigvalsh(disturbance_hessian))
        )
        control_min = float(
            np.min(np.linalg.eigvalsh(reduced_control_hessian))
        )
        closed_loop_eigenvalues = np.linalg.eigvals(
            matrix_a - matrix_b @ gain
        )
        spectral_radius = float(max(abs(closed_loop_eigenvalues)))
        finite = all(
            np.isfinite(values).all()
            for values in (
                riccati,
                gain,
                closed_loop_eigenvalues,
            )
        )
        tolerance = 1.0e-9
        if not (
            finite
            and disturbance_max < -tolerance
            and control_min > tolerance
            and spectral_radius < 1.0
        ):
            return None
        return (
            gain,
            riccati,
            closed_loop_eigenvalues,
            control_min,
            disturbance_max,
        )

    lower = float(gamma_lower_bound)
    upper = float(gamma_upper_bound)
    upper_solution = solve_game(upper)
    if upper_solution is None:
        raise RuntimeError(
            "no stabilizing H-infinity game solution at the gamma upper bound"
        )
    if solve_game(lower) is not None:
        upper = lower
        upper_solution = solve_game(lower)
    else:
        for _ in range(bisection_iterations):
            candidate = 0.5 * (lower + upper)
            candidate_solution = solve_game(candidate)
            if candidate_solution is None:
                lower = candidate
            else:
                upper = candidate
                upper_solution = candidate_solution
    if upper_solution is None:
        raise RuntimeError("no stabilizing H-infinity game solution was found")

    minimum_feasible_gamma = upper
    design_gamma = min(
        float(gamma_upper_bound),
        minimum_feasible_gamma * gamma_safety_factor,
    )
    design_solution = solve_game(design_gamma)
    if design_solution is None:
        raise RuntimeError("H-infinity safety-margin solution was not stabilizing")
    gain, riccati, eigenvalues, control_min, disturbance_max = design_solution
    closed_loop = matrix_a - matrix_b @ gain
    state_square_root = _symmetric_square_root(state_cost)
    performance_output = np.vstack(
        (state_square_root, -action_square_root @ gain)
    )
    certified_norm, peak_frequency = discrete_hinf_frequency_norm(
        closed_loop,
        disturbance_input,
        performance_output,
        grid_size=frequency_grid_size,
    )
    numerical_tolerance = 2.0e-3
    if certified_norm > design_gamma * (1.0 + numerical_tolerance):
        raise RuntimeError(
            "H-infinity bounded-real frequency check failed: "
            f"norm={certified_norm}, gamma={design_gamma}"
        )
    return HInfinityDesign(
        gain=gain,
        riccati=riccati,
        closed_loop_eigenvalues=eigenvalues,
        gamma=design_gamma,
        minimum_feasible_gamma_upper_bound=minimum_feasible_gamma,
        certified_hinf_norm=certified_norm,
        peak_frequency_rad_per_sample=peak_frequency,
        disturbance_input=disturbance_input,
        control_hessian_min_eigenvalue=control_min,
        disturbance_hessian_max_eigenvalue=disturbance_max,
        controllability_rank=controllability_rank(matrix_a, matrix_b),
    )


class LinearFeedbackActor(nn.Module):
    """Torch actor shared by LQR and H-infinity controller artifacts."""

    def __init__(
        self,
        gain: np.ndarray | torch.Tensor,
        *,
        state_center: np.ndarray | torch.Tensor | None = None,
        action_offset: np.ndarray | torch.Tensor | None = None,
        forward_feedforward_scale: float = 1.0,
        yaw_feedforward_scale: float = 1.0,
    ) -> None:
        super().__init__()
        gain_tensor = torch.as_tensor(gain, dtype=torch.float32)
        if gain_tensor.shape != (ACTION_DIM, CONTROL_STATE_DIM):
            raise ValueError(
                "linear feedback gain must have shape "
                f"{(ACTION_DIM, CONTROL_STATE_DIM)}"
            )
        if not bool(torch.isfinite(gain_tensor).all()):
            raise ValueError("linear feedback gain contains non-finite values")
        state_center_tensor = torch.as_tensor(
            np.zeros(CONTROL_STATE_DIM) if state_center is None else state_center,
            dtype=torch.float32,
        )
        action_offset_tensor = torch.as_tensor(
            np.zeros(ACTION_DIM) if action_offset is None else action_offset,
            dtype=torch.float32,
        )
        if state_center_tensor.shape != (CONTROL_STATE_DIM,):
            raise ValueError(
                f"state center must have shape {(CONTROL_STATE_DIM,)}"
            )
        if action_offset_tensor.shape != (ACTION_DIM,):
            raise ValueError(f"action offset must have shape {(ACTION_DIM,)}")
        if not bool(
            torch.isfinite(state_center_tensor).all()
            and torch.isfinite(action_offset_tensor).all()
        ):
            raise ValueError("linear controller centers must be finite")
        feedforward_scales = torch.as_tensor(
            [forward_feedforward_scale, yaw_feedforward_scale],
            dtype=torch.float32,
        )
        if not bool(
            torch.isfinite(feedforward_scales).all()
            and (feedforward_scales > 0.0).all()
        ):
            raise ValueError("feedforward scales must be finite and positive")
        self.register_buffer("gain", gain_tensor)
        self.register_buffer("state_center", state_center_tensor)
        self.register_buffer("action_offset", action_offset_tensor)
        self.register_buffer("feedforward_scales", feedforward_scales)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        state = torch_control_state_from_observation(observation)
        state = state - self.state_center
        command = observation[..., COMMAND_OBSERVATION_SLICE]
        scaled_command = command.clone()
        scaled_command[..., 0] *= self.feedforward_scales[0]
        scaled_command[..., 2] *= self.feedforward_scales[1]
        feedforward = torch_feedforward_action_from_command(scaled_command)
        feedback = state @ self.gain.T
        return torch.clamp(
            feedforward + self.action_offset - feedback,
            -1.0,
            1.0,
        )


class ScheduledLinearFeedbackActor(nn.Module):
    """Smooth command-space interpolation across local linear controllers."""

    def __init__(
        self,
        gains: np.ndarray | torch.Tensor,
        *,
        state_centers: np.ndarray | torch.Tensor,
        action_offsets: np.ndarray | torch.Tensor,
        operating_commands: np.ndarray | torch.Tensor,
        feedforward_scales: np.ndarray | torch.Tensor,
        command_distance_scales: Sequence[float] = (1.0, 1.0, 0.4),
        schedule_sharpness: float = 12.0,
    ) -> None:
        super().__init__()
        gains = torch.as_tensor(gains, dtype=torch.float32)
        state_centers = torch.as_tensor(state_centers, dtype=torch.float32)
        action_offsets = torch.as_tensor(action_offsets, dtype=torch.float32)
        operating_commands = torch.as_tensor(
            operating_commands, dtype=torch.float32
        )
        feedforward_scales = torch.as_tensor(
            feedforward_scales, dtype=torch.float32
        )
        schedule_count = gains.shape[0] if gains.ndim == 3 else 0
        expected_shapes = (
            gains.shape == (schedule_count, ACTION_DIM, CONTROL_STATE_DIM),
            state_centers.shape == (schedule_count, CONTROL_STATE_DIM),
            action_offsets.shape == (schedule_count, ACTION_DIM),
            operating_commands.shape == (schedule_count, 3),
            feedforward_scales.shape == (schedule_count, 2),
        )
        if schedule_count < 2 or not all(expected_shapes):
            raise ValueError("scheduled linear controller arrays have invalid shapes")
        distance_scales = torch.as_tensor(
            command_distance_scales, dtype=torch.float32
        )
        if distance_scales.shape != (3,):
            raise ValueError("command distance scales must have shape (3,)")
        numeric = (
            gains,
            state_centers,
            action_offsets,
            operating_commands,
            feedforward_scales,
            distance_scales,
        )
        if not all(bool(torch.isfinite(values).all()) for values in numeric):
            raise ValueError("scheduled linear controller contains non-finite data")
        if not bool((feedforward_scales > 0.0).all()):
            raise ValueError("feedforward scales must be positive")
        if not bool((distance_scales > 0.0).all()):
            raise ValueError("command distance scales must be positive")
        if not np.isfinite(schedule_sharpness) or schedule_sharpness <= 0.0:
            raise ValueError("schedule sharpness must be finite and positive")
        self.register_buffer("gains", gains)
        self.register_buffer("state_centers", state_centers)
        self.register_buffer("action_offsets", action_offsets)
        self.register_buffer("operating_commands", operating_commands)
        self.register_buffer("feedforward_scales", feedforward_scales)
        self.register_buffer("command_distance_scales", distance_scales)
        self.schedule_sharpness = float(schedule_sharpness)

    def schedule_weights(self, observation: torch.Tensor) -> torch.Tensor:
        command = observation[..., COMMAND_OBSERVATION_SLICE]
        normalized_error = (
            command.unsqueeze(-2) - self.operating_commands
        ) / self.command_distance_scales
        squared_distance = torch.sum(torch.square(normalized_error), dim=-1)
        return torch.softmax(
            -self.schedule_sharpness * squared_distance,
            dim=-1,
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        state = torch_control_state_from_observation(observation)
        centered_state = state.unsqueeze(-2) - self.state_centers
        feedback = torch.einsum(
            "...ms,mus->...mu", centered_state, self.gains
        )
        command = observation[..., COMMAND_OBSERVATION_SLICE]
        scaled_command = command.unsqueeze(-2).expand(
            command.shape[:-1] + (self.gains.shape[0], 3)
        ).clone()
        scaled_command[..., 0] *= self.feedforward_scales[:, 0]
        scaled_command[..., 2] *= self.feedforward_scales[:, 1]
        feedforward = torch_feedforward_action_from_command(scaled_command)
        candidates = torch.clamp(
            feedforward + self.action_offsets - feedback,
            -1.0,
            1.0,
        )
        weights = self.schedule_weights(observation)
        return torch.clamp(
            torch.sum(weights.unsqueeze(-1) * candidates, dim=-2),
            -1.0,
            1.0,
        )


def make_linear_controller_checkpoint(
    *,
    controller: str,
    gain: np.ndarray | torch.Tensor,
    state_center: np.ndarray | torch.Tensor | None = None,
    action_offset: np.ndarray | torch.Tensor | None = None,
    forward_feedforward_scale: float = 1.0,
    yaw_feedforward_scale: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = controller.lower().replace("∞", "inf")
    if normalized not in {"lqr", "hinf", "h_infinity"}:
        raise ValueError("controller must be LQR or H-infinity")
    normalized = "hinf" if normalized != "lqr" else "lqr"
    actor = LinearFeedbackActor(
        gain,
        state_center=state_center,
        action_offset=action_offset,
        forward_feedforward_scale=forward_feedforward_scale,
        yaw_feedforward_scale=yaw_feedforward_scale,
    )
    return {
        "checkpoint_format": LINEAR_CONTROLLER_CHECKPOINT_FORMAT,
        "linear_controller": normalized,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "control_state_names": list(CONTROL_STATE_NAMES),
        "gain": actor.gain.detach().cpu(),
        "state_center": actor.state_center.detach().cpu(),
        "action_offset": actor.action_offset.detach().cpu(),
        "forward_feedforward_scale": float(actor.feedforward_scales[0]),
        "yaw_feedforward_scale": float(actor.feedforward_scales[1]),
        "metadata": dict(metadata or {}),
    }


def make_scheduled_linear_controller_checkpoint(
    checkpoints: Sequence[dict[str, Any]],
    *,
    command_distance_scales: Sequence[float] = (1.0, 1.0, 0.4),
    schedule_sharpness: float = 12.0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine compatible local LQR or H-infinity artifacts."""

    if len(checkpoints) < 2:
        raise ValueError("a gain schedule needs at least two checkpoints")
    controller = str(checkpoints[0].get("linear_controller", ""))
    if controller not in {"lqr", "hinf"}:
        raise ValueError("scheduled controller must be LQR or H-infinity")
    for checkpoint in checkpoints:
        if checkpoint.get("checkpoint_format") != LINEAR_CONTROLLER_CHECKPOINT_FORMAT:
            raise ValueError("gain schedule source is not a local linear controller")
        if checkpoint.get("linear_controller") != controller:
            raise ValueError("gain schedule cannot mix controller families")
        if (
            int(checkpoint.get("observation_dim", -1)) != OBSERVATION_DIM
            or int(checkpoint.get("action_dim", -1)) != ACTION_DIM
        ):
            raise ValueError("gain schedule source dimensions do not match Sentinel")
    operating_commands = []
    for checkpoint in checkpoints:
        command = checkpoint.get("metadata", {}).get("operating_command")
        if command is None:
            raise ValueError("gain schedule source lacks an operating command")
        operating_commands.append(command)
    feedforward_scales = [
        [
            checkpoint.get("forward_feedforward_scale", 1.0),
            checkpoint.get("yaw_feedforward_scale", 1.0),
        ]
        for checkpoint in checkpoints
    ]
    actor = ScheduledLinearFeedbackActor(
        torch.stack([checkpoint["gain"] for checkpoint in checkpoints]),
        state_centers=torch.stack(
            [checkpoint["state_center"] for checkpoint in checkpoints]
        ),
        action_offsets=torch.stack(
            [checkpoint["action_offset"] for checkpoint in checkpoints]
        ),
        operating_commands=operating_commands,
        feedforward_scales=feedforward_scales,
        command_distance_scales=command_distance_scales,
        schedule_sharpness=schedule_sharpness,
    )
    return {
        "checkpoint_format": SCHEDULED_LINEAR_CONTROLLER_CHECKPOINT_FORMAT,
        "linear_controller": controller,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "control_state_names": list(CONTROL_STATE_NAMES),
        "gains": actor.gains.detach().cpu(),
        "state_centers": actor.state_centers.detach().cpu(),
        "action_offsets": actor.action_offsets.detach().cpu(),
        "operating_commands": actor.operating_commands.detach().cpu(),
        "feedforward_scales": actor.feedforward_scales.detach().cpu(),
        "command_distance_scales": (
            actor.command_distance_scales.detach().cpu()
        ),
        "schedule_sharpness": actor.schedule_sharpness,
        "metadata": dict(metadata or {}),
    }
