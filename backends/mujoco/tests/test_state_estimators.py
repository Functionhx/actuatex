from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tasks.inverted_pendulum.state_estimators import (  # noqa: E402
    ComplementaryFilterLQRController,
    ExtendedKalmanLQRController,
    LinearObserverLQRController,
    design_luenberger_gain,
    nonlinear_single_cartpole_step,
)


def test_luenberger_gain_places_requested_poles() -> None:
    matrix_a = np.array(
        [
            [1.0, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.1, 0.0],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    measurement_c = np.array([[1.0, 0.0, 0.0, 0.0]])
    gain, desired = design_luenberger_gain(
        matrix_a,
        measurement_c,
        slowest_rate=2.0,
        fastest_rate=8.0,
    )
    observed = np.linalg.eigvals(matrix_a - gain @ measurement_c)
    np.testing.assert_allclose(np.sort(observed), np.sort(desired), atol=1.0e-6)


def test_linear_and_complementary_observers_return_bounded_actions() -> None:
    matrix_a = np.eye(4)
    matrix_b = np.ones((4, 1)) * 0.01
    measurement_c = np.zeros((2, 4))
    measurement_c[:, :2] = np.eye(2)
    feedback_gain = np.ones((1, 4))
    linear = LinearObserverLQRController(
        matrix_a, matrix_b, measurement_c, feedback_gain, np.ones((4, 2)) * 0.1
    )
    complementary = ComplementaryFilterLQRController(
        matrix_a, matrix_b, measurement_c, feedback_gain
    )
    for controller in (linear, complementary):
        controller.reset(3)
        action = controller.act_from_measurement(np.ones((3, 2)) * 100.0)
        assert np.all(np.abs(action) <= 1.0)


def test_nonlinear_prediction_and_ekf_are_finite() -> None:
    state = np.array([[0.0, 0.1, 0.0, 0.0], [0.2, -0.2, 0.1, -0.1]])
    predicted = nonlinear_single_cartpole_step(state, np.zeros(2))
    assert predicted.shape == state.shape
    assert np.all(np.isfinite(predicted))

    controller = ExtendedKalmanLQRController(
        np.array([[-1.0, -30.0, -2.0, -6.0]]),
        measurement_noise_std=0.01,
    )
    controller.reset(2)
    action = controller.act_from_measurement(state[:, :2])
    assert np.all(np.isfinite(action))
    assert np.all(np.abs(action) <= 1.0)
