from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tasks.inverted_pendulum.robust_control import (  # noqa: E402
    CollocatedFeedbackLinearizationController,
    DiscreteSlidingModeController,
    PartialFeedbackLinearizationController,
    discrete_to_continuous,
    h_infinity_state_feedback_gain,
)


def test_discrete_to_continuous_round_trip_for_integrator() -> None:
    sample_time = 0.05
    matrix_a = np.array([[1.0, sample_time], [0.0, 1.0]])
    matrix_b = np.array([[0.5 * sample_time**2], [sample_time]])
    continuous_a, continuous_b = discrete_to_continuous(matrix_a, matrix_b, sample_time)
    np.testing.assert_allclose(continuous_a, [[0.0, 1.0], [0.0, 0.0]], atol=1e-9)
    np.testing.assert_allclose(continuous_b, [[0.0], [1.0]], atol=1e-9)


def test_h_infinity_gain_stabilizes_double_integrator() -> None:
    sample_time = 0.05
    matrix_a = np.array([[1.0, sample_time], [0.0, 1.0]])
    matrix_b = np.array([[0.5 * sample_time**2], [sample_time]])
    gain, gamma, poles = h_infinity_state_feedback_gain(
        matrix_a,
        matrix_b,
        np.diag([2.0, 0.2]),
        np.eye(1),
        sample_time=sample_time,
    )
    assert gain.shape == (1, 2)
    assert gamma > 0.0
    assert np.max(np.real(poles)) < 0.0


def test_nonlinear_and_sliding_controllers_bound_actions() -> None:
    matrix_a = np.eye(4)
    matrix_b = np.array([[0.01], [0.01], [0.02], [0.04]])
    sliding = DiscreteSlidingModeController(
        matrix_a, matrix_b, np.array([[1.0, 2.0, 3.0, 4.0]])
    )
    collocated = CollocatedFeedbackLinearizationController(np.ones(4))
    partial = PartialFeedbackLinearizationController()
    state = np.array([[10.0, 0.4, -5.0, 3.0]])
    for controller in (sliding, collocated, partial):
        controller.reset(1)
        action = controller.act(state)
        assert action.shape == (1,)
        assert np.isfinite(action).all()
        assert np.abs(action).max() <= 1.0
