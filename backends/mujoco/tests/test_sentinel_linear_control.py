from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tasks.robomaster.contract import ACTION_DIM  # noqa: E402
from tasks.robomaster.linear_control import (  # noqa: E402
    CONTROL_STATE_DIM,
    LEFT_RIGHT_ACTION_MIRROR,
    LEFT_RIGHT_STATE_MIRROR,
    LINEAR_CONTROLLER_CHECKPOINT_FORMAT,
    SCHEDULED_LINEAR_CONTROLLER_CHECKPOINT_FORMAT,
    LinearFeedbackActor,
    ScheduledLinearFeedbackActor,
    control_state_from_observation,
    design_discrete_hinf_state_feedback,
    design_discrete_lqr,
    feedforward_action_from_command,
    fit_affine_dynamics,
    make_linear_controller_checkpoint,
    make_scheduled_linear_controller_checkpoint,
    symmetrize_left_right_dynamics,
    torch_control_state_from_observation,
)
from tasks.robomaster.locomotion import OBSERVATION_DIM  # noqa: E402
from tasks.robomaster.policy import load_policy  # noqa: E402


def test_control_state_is_command_relative_and_torch_matches_numpy() -> None:
    observation = np.zeros((2, OBSERVATION_DIM), dtype=np.float32)
    observation[:, :3] = (0.7, -0.2, 0.1)
    observation[:, 3:6] = (0.3, -0.4, 0.8)
    observation[:, 6:9] = (0.05, -0.03, -0.998)
    observation[:, 9:12] = (0.5, 0.0, 0.6)
    observation[:, 12:16] = (0.1, 0.2, 0.3, 0.4)
    observation[:, 16:22] = np.arange(6) * 0.1
    observation[:, 22:28] = np.arange(6) * -0.1

    state = control_state_from_observation(observation)
    assert state.shape == (2, CONTROL_STATE_DIM)
    np.testing.assert_allclose(state[0, :8], [0.2, -0.2, 0.1, 0.3, -0.4, 0.2, 0.05, -0.03])
    torch_state = torch_control_state_from_observation(
        torch.from_numpy(observation)
    )
    np.testing.assert_allclose(torch_state.numpy(), state, atol=1.0e-7)
    feedforward = feedforward_action_from_command(observation[:, 9:12])
    np.testing.assert_allclose(
        state[:, -ACTION_DIM:],
        observation[:, 22:28] - feedforward,
        atol=1.0e-7,
    )


def test_differential_wheel_feedforward_tracks_speed_and_yaw() -> None:
    command = np.asarray([[0.6, 0.0, 0.8], [10.0, 0.0, 0.0]])
    action = feedforward_action_from_command(command)
    assert action.shape == (2, ACTION_DIM)
    assert action[0, 4] < action[0, 5]
    assert action[0, 4] == pytest.approx((0.6 - 0.22 * 0.8) / 1.5)
    assert action[0, 5] == pytest.approx((0.6 + 0.22 * 0.8) / 1.5)
    np.testing.assert_allclose(action[1, 4:], 1.0)


def test_affine_dynamics_fit_recovers_known_local_model() -> None:
    rng = np.random.default_rng(17)
    matrix_a = 0.7 * np.eye(CONTROL_STATE_DIM)
    matrix_a += rng.normal(0.0, 0.005, matrix_a.shape)
    matrix_b = rng.normal(0.0, 0.1, (CONTROL_STATE_DIM, ACTION_DIM))
    bias = rng.normal(0.0, 0.01, CONTROL_STATE_DIM)
    state = rng.normal(size=(1200, CONTROL_STATE_DIM))
    action = rng.normal(size=(1200, ACTION_DIM))
    next_state = state @ matrix_a.T + action @ matrix_b.T + bias

    fit = fit_affine_dynamics(state, action, next_state, ridge=1.0e-10)
    np.testing.assert_allclose(fit.matrix_a, matrix_a, atol=1.0e-9)
    np.testing.assert_allclose(fit.matrix_b, matrix_b, atol=1.0e-9)
    np.testing.assert_allclose(fit.bias, bias, atol=1.0e-9)
    assert fit.metrics.coefficient_of_determination > 0.999999


def test_discrete_lqr_stabilizes_controllable_system() -> None:
    matrix_a = np.asarray([[1.0, 0.02], [0.0, 1.01]])
    matrix_b = np.asarray([[0.0], [0.02]])
    design = design_discrete_lqr(
        matrix_a,
        matrix_b,
        np.diag([10.0, 2.0]),
        np.diag([0.5]),
    )
    assert design.gain.shape == (1, 2)
    assert max(abs(design.closed_loop_eigenvalues)) < 1.0
    assert design.controllability_rank == 2


def test_discrete_hinf_solves_game_and_certifies_frequency_gain() -> None:
    matrix_a = np.asarray([[1.0, 0.02], [0.0, 1.01]])
    matrix_b = np.asarray([[0.0], [0.02]])
    disturbance_input = 0.02 * np.eye(2)
    state_cost = np.diag([10.0, 2.0])
    action_cost = np.diag([0.5])

    design = design_discrete_hinf_state_feedback(
        matrix_a,
        matrix_b,
        disturbance_input,
        state_cost,
        action_cost,
        gamma_lower_bound=0.01,
        gamma_upper_bound=50.0,
        gamma_safety_factor=1.25,
        bisection_iterations=24,
        frequency_grid_size=257,
    )
    lqr = design_discrete_lqr(
        matrix_a,
        matrix_b,
        state_cost,
        action_cost,
    )

    assert design.gain.shape == (1, 2)
    assert max(abs(design.closed_loop_eigenvalues)) < 1.0
    assert design.certified_hinf_norm < design.gamma
    assert design.gamma > design.minimum_feasible_gamma_upper_bound
    assert design.control_hessian_min_eigenvalue > 0.0
    assert design.disturbance_hessian_max_eigenvalue < 0.0
    assert not np.allclose(design.gain, lqr.gain)
    assert design.controllability_rank == 2


def test_sentinel_dynamics_symmetrization_enforces_mirror_equivariance() -> None:
    rng = np.random.default_rng(29)
    matrix_a = rng.normal(size=(CONTROL_STATE_DIM, CONTROL_STATE_DIM))
    matrix_b = rng.normal(size=(CONTROL_STATE_DIM, ACTION_DIM))
    bias = rng.normal(size=CONTROL_STATE_DIM)

    symmetric_a, symmetric_b, symmetric_bias = (
        symmetrize_left_right_dynamics(matrix_a, matrix_b, bias)
    )
    np.testing.assert_allclose(
        LEFT_RIGHT_STATE_MIRROR @ symmetric_a,
        symmetric_a @ LEFT_RIGHT_STATE_MIRROR,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        LEFT_RIGHT_STATE_MIRROR @ symmetric_b,
        symmetric_b @ LEFT_RIGHT_ACTION_MIRROR,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        LEFT_RIGHT_STATE_MIRROR @ symmetric_bias,
        symmetric_bias,
        atol=1.0e-12,
    )


def test_linear_checkpoint_uses_same_policy_loader_and_action_contract() -> None:
    gain = np.zeros((ACTION_DIM, CONTROL_STATE_DIM))
    gain[4, 0] = 0.1
    payload = make_linear_controller_checkpoint(
        controller="H∞",
        gain=gain,
        state_center=np.zeros(CONTROL_STATE_DIM),
        action_offset=np.zeros(ACTION_DIM),
        yaw_feedforward_scale=2.0,
        metadata={"backend": "MuJoCo 3.10"},
    )
    assert payload["checkpoint_format"] == LINEAR_CONTROLLER_CHECKPOINT_FORMAT
    loaded = load_policy(payload)
    assert loaded.algorithm == "hinf"
    assert isinstance(loaded.actor, LinearFeedbackActor)

    observation = torch.zeros(3, OBSERVATION_DIM)
    observation[:, 0] = 0.5
    action = loaded.actor(observation)
    assert action.shape == (3, ACTION_DIM)
    assert torch.allclose(action[:, 4], torch.full((3,), -0.05))

    observation.zero_()
    observation[:, 11] = 0.8
    action = loaded.actor(observation)
    expected = 2.0 * 0.5 * 0.44 * 0.8 / (0.075 * 20.0)
    assert torch.allclose(action[:, 4], torch.full((3,), -expected))
    assert torch.allclose(action[:, 5], torch.full((3,), expected))


def test_scheduled_linear_checkpoint_blends_local_controllers() -> None:
    gain = np.zeros((ACTION_DIM, CONTROL_STATE_DIM))
    base = make_linear_controller_checkpoint(
        controller="lqr",
        gain=gain,
        action_offset=np.zeros(ACTION_DIM),
        metadata={"operating_command": [0.0, 0.0, 0.0]},
    )
    turning_offset = np.zeros(ACTION_DIM)
    turning_offset[0] = 0.4
    turn = make_linear_controller_checkpoint(
        controller="lqr",
        gain=gain,
        action_offset=turning_offset,
        metadata={"operating_command": [0.0, 0.0, 0.8]},
    )
    payload = make_scheduled_linear_controller_checkpoint(
        [base, turn],
        schedule_sharpness=20.0,
    )
    assert (
        payload["checkpoint_format"]
        == SCHEDULED_LINEAR_CONTROLLER_CHECKPOINT_FORMAT
    )
    loaded = load_policy(payload)
    assert isinstance(loaded.actor, ScheduledLinearFeedbackActor)

    observation = torch.zeros(2, OBSERVATION_DIM)
    observation[1, 11] = 0.8
    weights = loaded.actor.schedule_weights(observation)
    assert weights[0, 0] > 0.999
    assert weights[1, 1] > 0.999
    action = loaded.actor(observation)
    assert action[0, 0].abs() < 1.0e-5
    assert action[1, 0] == pytest.approx(0.4, abs=1.0e-5)
