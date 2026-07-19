from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tasks.locomotion.contract import (  # noqa: E402
    ActionDelayLine,
    SafetyEnvelope,
    TermMajorHistory,
    joint_permutation,
    reorder_joints,
)
from tasks.locomotion import g1_29dof as g1  # noqa: E402


def test_joint_permutation_and_g1_upstream_round_trip():
    source = ("knee", "hip", "ankle")
    target = ("hip", "ankle", "knee")
    assert joint_permutation(source, target).tolist() == [1, 2, 0]
    np.testing.assert_array_equal(
        reorder_joints(np.array([10.0, 20.0, 30.0]), source, target),
        [20.0, 30.0, 10.0],
    )

    sdk = np.arange(g1.ACTION_DIM, dtype=np.float64)
    policy = g1.sdk_to_official_policy(sdk)
    np.testing.assert_array_equal(policy, sdk[list(g1.OFFICIAL_POLICY_TO_SDK)])
    np.testing.assert_array_equal(g1.official_policy_to_sdk(policy), sdk)


def test_g1_contract_dimensions_and_upstream_default_pose():
    g1.validate_contract()
    assert g1.ACTION_DIM == 29
    assert g1.SINGLE_FRAME_OBSERVATION_DIM == 96
    assert g1.OBSERVATION_DIM == 480
    assert g1.POLICY_FREQUENCY_HZ == 50.0
    assert len(g1.contract_sha256()) == 64

    # Exact order exported by Unitree RL Lab at the pinned revision.
    expected_policy_default = np.asarray(
        [
            -0.1,
            -0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.3,
            0.3,
            0.3,
            0.3,
            -0.2,
            -0.2,
            0.25,
            -0.25,
            0.0,
            0.0,
            0.0,
            0.0,
            0.97,
            0.97,
            0.15,
            -0.15,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    np.testing.assert_allclose(
        g1.sdk_to_official_policy(g1.DEFAULT_JOINT_POSITION),
        expected_policy_default,
    )


def test_term_major_history_is_not_frame_major():
    history = TermMajorHistory({"a": 1, "b": 2}, history_length=3)
    history.reset({"a": np.array([0.0]), "b": np.array([0.0, 0.0])})
    history.append({"a": np.array([1.0]), "b": np.array([10.0, 11.0])})
    observation = history.append(
        {"a": np.array([2.0]), "b": np.array([20.0, 21.0])}
    )
    np.testing.assert_array_equal(
        observation,
        [0.0, 1.0, 2.0, 0.0, 0.0, 10.0, 11.0, 20.0, 21.0],
    )


def test_g1_observation_history_and_source_order_conversion():
    batch = 2
    sdk_position = np.broadcast_to(g1.DEFAULT_JOINT_POSITION, (batch, g1.ACTION_DIM))
    source_position = g1.sdk_to_official_policy(sdk_position)
    terms = g1.build_observation_terms(
        np.ones((batch, 3)),
        np.broadcast_to([0.0, 0.0, -1.0], (batch, 3)),
        np.zeros((batch, 3)),
        source_position,
        np.zeros((batch, g1.ACTION_DIM)),
        np.zeros((batch, g1.ACTION_DIM)),
        source_joint_names=g1.OFFICIAL_POLICY_JOINT_NAMES,
    )
    np.testing.assert_allclose(terms["base_ang_vel"], 0.2)
    np.testing.assert_allclose(terms["joint_pos_rel"], 0.0, atol=1.0e-7)
    history = g1.make_observation_history()
    assert history.reset(terms).shape == (batch, g1.OBSERVATION_DIM)


def test_action_delay_line_has_exact_integer_latency():
    delay = ActionDelayLine(action_dim=1, delay_steps=2)
    delay.reset(np.array([0.0]))
    np.testing.assert_array_equal(delay.push(np.array([1.0])), [0.0])
    np.testing.assert_array_equal(delay.push(np.array([2.0])), [0.0])
    np.testing.assert_array_equal(delay.push(np.array([3.0])), [1.0])

    no_delay = ActionDelayLine(action_dim=1, delay_steps=0)
    np.testing.assert_array_equal(no_delay.push(np.array([7.0])), [7.0])


def test_g1_pd_effort_obeys_command_and_torque_speed_envelopes():
    target = g1.action_to_position_target(np.ones(g1.ACTION_DIM))
    assert np.all(target <= g1.SOFT_JOINT_UPPER)
    assert np.all(target >= g1.SOFT_JOINT_LOWER)

    zero_velocity = np.zeros(g1.ACTION_DIM)
    effort = g1.compute_pd_effort(
        g1.DEFAULT_JOINT_POSITION,
        zero_velocity,
        target,
    )
    assert np.all(np.abs(effort) <= g1.COMMAND_EFFORT_LIMIT + 1.0e-12)

    requested = np.ones(g1.ACTION_DIM) * 1.0e6
    full_torque = g1.torque_speed_limit(zero_velocity, requested)
    no_load_speed = np.asarray(
        [g1.MOTOR_CURVES[motor].x2 for motor in g1.MOTOR_TYPE]
    )
    no_load_torque = g1.torque_speed_limit(no_load_speed, requested)
    assert np.all(full_torque > 0.0)
    np.testing.assert_allclose(no_load_torque, 0.0, atol=1.0e-12)


def test_safety_envelope_slews_and_fails_closed():
    envelope = SafetyEnvelope(
        command_lower=(-0.5, -0.3, -1.0),
        command_upper=(1.0, 0.3, 1.0),
        command_acceleration=(1.0, 2.0, 4.0),
    )
    command = envelope.slew_command(
        np.array([5.0, -5.0, 5.0]), np.zeros(3), dt=0.1
    )
    np.testing.assert_allclose(command, [0.1, -0.2, 0.4])

    good = envelope.evaluate(
        [0.0, 0.0, -1.0],
        state_age_s=0.01,
        joint_position=g1.DEFAULT_JOINT_POSITION,
        joint_lower=g1.JOINT_LOWER,
        joint_upper=g1.JOINT_UPPER,
    )
    assert good.enabled
    assert good.reasons == ()

    stale = envelope.evaluate(
        [0.0, 0.0, 1.0],
        state_age_s=0.2,
        joint_position=g1.DEFAULT_JOINT_POSITION,
        joint_lower=g1.JOINT_LOWER,
        joint_upper=g1.JOINT_UPPER,
    )
    assert not stale.enabled
    assert set(stale.reasons) == {"tilt", "watchdog"}
    np.testing.assert_array_equal(
        envelope.sanitize_action(np.ones(g1.ACTION_DIM), enabled=False),
        np.zeros(g1.ACTION_DIM),
    )
