from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tasks.inverted_pendulum.classical_control import (  # noqa: E402
    CascadedPIDController,
    LQGController,
    PIDController,
    default_quadratic_cost,
    discrete_lqr_gain,
    finite_horizon_lqr_gain,
    pole_placement_gain,
    steady_state_kalman_gain,
)


def _controllable_double_integrator() -> tuple[np.ndarray, np.ndarray]:
    matrix_a = np.array([[1.0, 0.05], [0.0, 1.0]])
    matrix_b = np.array([[0.00125], [0.05]])
    return matrix_a, matrix_b


def test_lqr_and_long_horizon_mpc_stabilize_linear_model() -> None:
    matrix_a, matrix_b = _controllable_double_integrator()
    cost_q = np.diag([2.0, 0.2])
    cost_r = np.eye(1) * 0.1
    lqr_gain = discrete_lqr_gain(matrix_a, matrix_b, cost_q, cost_r)
    mpc_gain = finite_horizon_lqr_gain(
        matrix_a,
        matrix_b,
        cost_q,
        cost_r,
        horizon=400,
        terminal_cost=cost_q,
    )

    assert np.max(np.abs(np.linalg.eigvals(matrix_a - matrix_b @ lqr_gain))) < 1.0
    assert np.max(np.abs(np.linalg.eigvals(matrix_a - matrix_b @ mpc_gain))) < 1.0
    np.testing.assert_allclose(mpc_gain, lqr_gain, rtol=1.0e-4, atol=1.0e-4)


def test_pole_placement_matches_requested_poles() -> None:
    matrix_a, matrix_b = _controllable_double_integrator()
    gain, desired = pole_placement_gain(matrix_a, matrix_b, policy_dt=0.05)
    actual = np.linalg.eigvals(matrix_a - matrix_b @ gain)
    np.testing.assert_allclose(np.sort(actual), np.sort(desired), atol=1.0e-8)


def test_kalman_and_lqg_shapes_are_finite() -> None:
    matrix_a, matrix_b = _controllable_double_integrator()
    measurement_c = np.array([[1.0, 0.0]])
    kalman_gain = steady_state_kalman_gain(
        matrix_a,
        measurement_c,
        np.eye(2) * 1.0e-4,
        np.eye(1) * 1.0e-3,
    )
    feedback_gain = np.array([[1.0, 0.5]])
    controller = LQGController(
        matrix_a,
        matrix_b,
        measurement_c,
        feedback_gain,
        kalman_gain,
    )
    controller.reset(4)
    action = controller.act_from_measurement(np.zeros((4, 1)))

    assert kalman_gain.shape == (2, 1)
    assert action.shape == (4,)
    assert np.all(np.isfinite(action))


def test_pid_controllers_clip_actions() -> None:
    state = np.full((3, 4), 100.0)
    pid = PIDController(1.0, 1.0, 10.0, 1.0, 2.0)
    cascade = CascadedPIDController(0.1, 0.1, 10.0, 1.0, 2.0)
    pid.reset(3)
    cascade.reset(3)

    assert np.all(np.abs(pid.act(state)) <= 1.0)
    assert np.all(np.abs(cascade.act(state)) <= 1.0)


def test_default_cost_shapes_for_all_orders() -> None:
    for order in (1, 2, 3):
        cost_q, cost_r = default_quadratic_cost(order)
        assert cost_q.shape == (2 * (order + 1), 2 * (order + 1))
        assert cost_r.shape == (1, 1)
        assert np.all(np.linalg.eigvalsh(cost_q) >= 0.0)
