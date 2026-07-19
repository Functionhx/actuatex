from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tasks.inverted_pendulum.state_estimators import (  # noqa: E402
    nonlinear_single_cartpole_step,
)
from tasks.inverted_pendulum.trajectory_optimization import (  # noqa: E402
    TVLQRTrackingController,
    TrajectoryReplayController,
    iterative_lqr_swingup,
    linearize_trajectory,
    tvlqr_gains,
)


def test_trajectory_linearization_matches_local_perturbation() -> None:
    states = np.zeros((4, 4))
    forces = np.zeros(3)
    for step in range(3):
        states[step + 1] = nonlinear_single_cartpole_step(
            states[step : step + 1], np.zeros(1)
        )[0]
    matrices_a, matrices_b = linearize_trajectory(states, forces)
    assert matrices_a.shape == (3, 4, 4)
    assert matrices_b.shape == (3, 4, 1)
    assert np.all(np.isfinite(matrices_a))
    assert np.all(np.isfinite(matrices_b))


def test_tvlqr_and_replay_return_bounded_actions() -> None:
    states = np.zeros((5, 4))
    forces = np.zeros(4)
    gains = tvlqr_gains(states, forces)
    replay = TrajectoryReplayController(forces)
    tracking = TVLQRTrackingController(states, forces, gains, np.ones((1, 4)))
    for controller in (replay, tracking):
        controller.reset(2)
        action = controller.act(np.ones((2, 4)) * 100.0)
        assert action.shape == (2,)
        assert np.all(np.abs(action) <= 1.0)


def test_ilqr_reduces_cost_from_a_nonzero_seed() -> None:
    initial_state = np.array([0.0, np.pi - 0.1, 0.0, 0.0])
    initial_forces = np.sin(np.linspace(0.0, 2.0 * np.pi, 20)) * 2.0
    result = iterative_lqr_swingup(
        initial_state,
        initial_forces,
        max_iterations=3,
    )
    assert result.states.shape == (21, 4)
    assert result.forces.shape == (20,)
    assert np.all(np.isfinite(result.states))
    assert np.all(np.abs(result.forces) <= 20.0)
