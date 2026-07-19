"""Industrial policy/deployment contract for the Unitree G1 29-DOF robot.

The canonical order in this file is the Unitree ``LowCmd``/``LowState`` SDK
order.  It is also the joint and actuator order in the official
``unitree_mujoco`` G1 29-DOF model.  Unitree RL Lab's current USD traversal is
interleaved; explicit conversion helpers keep its exported policies usable
without leaking that simulator-specific order into the hardware interface.

Nominal joint, PD and observation settings follow Unitree RL Lab.  Hard joint
and command-effort limits follow the official Unitree MuJoCo model.  The
torque-speed/friction envelope follows Unitree RL Lab's explicit actuator
model.  See ``robots/g1/upstream.json`` for the audited source revisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np

from .contract import SafetyEnvelope, TermMajorHistory, reorder_joints


SDK_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Isaac/Unitree RL Lab policy index -> Unitree SDK index.  This mapping is
# exported as ``joint_ids_map`` by the upstream deployment tool.
OFFICIAL_POLICY_TO_SDK = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
)
OFFICIAL_POLICY_JOINT_NAMES = tuple(
    SDK_JOINT_NAMES[index] for index in OFFICIAL_POLICY_TO_SDK
)

ACTION_DIM = len(SDK_JOINT_NAMES)
SIM_DT = 0.005
DECIMATION = 4
POLICY_DT = SIM_DT * DECIMATION
POLICY_FREQUENCY_HZ = 1.0 / POLICY_DT
HISTORY_LENGTH = 5
ACTION_SCALE_RAD = 0.25
JOINT_VELOCITY_SCALE = 0.05
BASE_ANGULAR_VELOCITY_SCALE = 0.2

OBSERVATION_TERM_DIMS = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "velocity_commands": 3,
    "joint_pos_rel": ACTION_DIM,
    "joint_vel_rel": ACTION_DIM,
    "last_action": ACTION_DIM,
}
SINGLE_FRAME_OBSERVATION_DIM = sum(OBSERVATION_TERM_DIMS.values())
OBSERVATION_DIM = HISTORY_LENGTH * SINGLE_FRAME_OBSERVATION_DIM

DEFAULT_JOINT_POSITION = np.asarray(
    [
        -0.10,
        0.0,
        0.0,
        0.30,
        -0.20,
        0.0,
        -0.10,
        0.0,
        0.0,
        0.30,
        -0.20,
        0.0,
        0.0,
        0.0,
        0.0,
        0.30,
        0.25,
        0.0,
        0.97,
        0.15,
        0.0,
        0.0,
        0.30,
        -0.25,
        0.0,
        0.97,
        -0.15,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)

STIFFNESS = np.asarray(
    [
        100,
        100,
        100,
        150,
        40,
        40,
        100,
        100,
        100,
        150,
        40,
        40,
        200,
        200,
        200,
        *([40] * 14),
    ],
    dtype=np.float64,
)
DAMPING = np.asarray(
    [
        2,
        2,
        2,
        4,
        2,
        2,
        2,
        2,
        2,
        4,
        2,
        2,
        5,
        5,
        5,
        *([10] * 14),
    ],
    dtype=np.float64,
)

JOINT_LOWER = np.asarray(
    [
        -2.5307,
        -0.5236,
        -2.7576,
        -0.087267,
        -0.87267,
        -0.2618,
        -2.5307,
        -2.9671,
        -2.7576,
        -0.087267,
        -0.87267,
        -0.2618,
        -2.618,
        -0.52,
        -0.52,
        -3.0892,
        -1.5882,
        -2.618,
        -1.0472,
        -1.97222,
        -1.61443,
        -1.61443,
        -3.0892,
        -2.2515,
        -2.618,
        -1.0472,
        -1.97222,
        -1.61443,
        -1.61443,
    ],
    dtype=np.float64,
)
JOINT_UPPER = np.asarray(
    [
        2.8798,
        2.9671,
        2.7576,
        2.8798,
        0.5236,
        0.2618,
        2.8798,
        0.5236,
        2.7576,
        2.8798,
        0.5236,
        0.2618,
        2.618,
        0.52,
        0.52,
        2.6704,
        2.2515,
        2.618,
        2.0944,
        1.97222,
        1.61443,
        1.61443,
        2.6704,
        1.5882,
        2.618,
        2.0944,
        1.97222,
        1.61443,
        1.61443,
    ],
    dtype=np.float64,
)
SOFT_JOINT_LOWER = 0.5 * (JOINT_LOWER + JOINT_UPPER) + 0.45 * (
    JOINT_LOWER - JOINT_UPPER
)
SOFT_JOINT_UPPER = 0.5 * (JOINT_LOWER + JOINT_UPPER) + 0.45 * (
    JOINT_UPPER - JOINT_LOWER
)

# MuJoCo motor ctrlrange / SDK command envelope, in N m.
COMMAND_EFFORT_LIMIT = np.asarray(
    [
        88,
        88,
        88,
        139,
        50,
        50,
        88,
        88,
        88,
        139,
        50,
        50,
        88,
        50,
        50,
        25,
        25,
        25,
        25,
        25,
        5,
        5,
        25,
        25,
        25,
        25,
        25,
        5,
        5,
    ],
    dtype=np.float64,
)
VELOCITY_LIMIT = np.asarray(
    [
        32,
        20,
        32,
        20,
        37,
        37,
        32,
        20,
        32,
        20,
        37,
        37,
        32,
        37,
        37,
        37,
        37,
        37,
        37,
        37,
        22,
        22,
        37,
        37,
        37,
        37,
        37,
        22,
        22,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class MotorCurve:
    x1: float
    x2: float
    y1: float
    y2: float
    static_friction: float
    dynamic_friction: float
    activation_velocity: float = 0.01


MOTOR_CURVES = {
    "N7520-14.3": MotorCurve(22.63, 35.52, 71.0, 83.3, 1.6, 0.16),
    "N7520-22.5": MotorCurve(14.5, 22.7, 111.0, 131.0, 2.4, 0.24),
    "N5020-16": MotorCurve(30.86, 40.13, 24.8, 31.9, 0.6, 0.06),
    "W4010-25": MotorCurve(15.3, 24.76, 4.8, 8.6, 0.6, 0.06),
}
MOTOR_TYPE = (
    "N7520-14.3",
    "N7520-22.5",
    "N7520-14.3",
    "N7520-22.5",
    "N5020-16",
    "N5020-16",
    "N7520-14.3",
    "N7520-22.5",
    "N7520-14.3",
    "N7520-22.5",
    "N5020-16",
    "N5020-16",
    "N7520-14.3",
    "N5020-16",
    "N5020-16",
    "N5020-16",
    "N5020-16",
    "N5020-16",
    "N5020-16",
    "N5020-16",
    "W4010-25",
    "W4010-25",
    "N5020-16",
    "N5020-16",
    "N5020-16",
    "N5020-16",
    "N5020-16",
    "W4010-25",
    "W4010-25",
)

DEPLOYMENT_SAFETY = SafetyEnvelope(
    command_lower=(-0.5, -0.3, -0.2),
    command_upper=(1.0, 0.3, 0.2),
    command_acceleration=(1.0, 1.0, 1.5),
    action_limit=1.0,
    maximum_tilt_rad=0.8,
    watchdog_timeout_s=0.10,
)


def _vector(values: np.ndarray, size: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 0 or result.shape[-1] != size:
        raise ValueError(f"{label} must have last dimension {size}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains a non-finite value")
    return result


def sdk_to_official_policy(values: np.ndarray) -> np.ndarray:
    """Convert an SDK/MuJoCo-ordered vector to upstream RL Lab policy order."""

    return reorder_joints(values, SDK_JOINT_NAMES, OFFICIAL_POLICY_JOINT_NAMES)


def official_policy_to_sdk(values: np.ndarray) -> np.ndarray:
    """Convert an upstream RL Lab policy vector to SDK/MuJoCo order."""

    return reorder_joints(values, OFFICIAL_POLICY_JOINT_NAMES, SDK_JOINT_NAMES)


def build_observation_terms(
    base_angular_velocity_body: np.ndarray,
    projected_gravity_body: np.ndarray,
    velocity_command: np.ndarray,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    previous_action: np.ndarray,
    *,
    source_joint_names: Sequence[str] = SDK_JOINT_NAMES,
) -> dict[str, np.ndarray]:
    """Build one G1 policy frame in canonical SDK joint order."""

    base_angular_velocity_body = _vector(
        base_angular_velocity_body, 3, "base_angular_velocity_body"
    )
    projected_gravity_body = _vector(
        projected_gravity_body, 3, "projected_gravity_body"
    )
    velocity_command = _vector(velocity_command, 3, "velocity_command")
    joint_position = reorder_joints(
        joint_position, source_joint_names, SDK_JOINT_NAMES
    )
    joint_velocity = reorder_joints(
        joint_velocity, source_joint_names, SDK_JOINT_NAMES
    )
    previous_action = reorder_joints(
        previous_action, source_joint_names, SDK_JOINT_NAMES
    )
    leading_shape = base_angular_velocity_body.shape[:-1]
    values = (
        projected_gravity_body,
        velocity_command,
        joint_position,
        joint_velocity,
        previous_action,
    )
    if any(value.shape[:-1] != leading_shape for value in values):
        raise ValueError("all observation inputs must share leading dimensions")
    return {
        "base_ang_vel": (
            base_angular_velocity_body * BASE_ANGULAR_VELOCITY_SCALE
        ).astype(np.float32),
        "projected_gravity": projected_gravity_body.astype(np.float32),
        "velocity_commands": velocity_command.astype(np.float32),
        "joint_pos_rel": (joint_position - DEFAULT_JOINT_POSITION).astype(
            np.float32
        ),
        "joint_vel_rel": (joint_velocity * JOINT_VELOCITY_SCALE).astype(
            np.float32
        ),
        "last_action": previous_action.astype(np.float32),
    }


def make_observation_history() -> TermMajorHistory:
    return TermMajorHistory(OBSERVATION_TERM_DIMS, HISTORY_LENGTH)


def action_to_position_target(action: np.ndarray) -> np.ndarray:
    """Convert normalized policy output into a soft-limit-safe PD target."""

    action = _vector(action, ACTION_DIM, "action")
    target = DEFAULT_JOINT_POSITION + ACTION_SCALE_RAD * np.clip(action, -1.0, 1.0)
    return np.clip(target, SOFT_JOINT_LOWER, SOFT_JOINT_UPPER)


def torque_speed_limit(
    joint_velocity: np.ndarray,
    requested_effort: np.ndarray,
) -> np.ndarray:
    """Return the direction-aware motor effort envelope at current speed."""

    joint_velocity = _vector(joint_velocity, ACTION_DIM, "joint_velocity")
    requested_effort = _vector(requested_effort, ACTION_DIM, "requested_effort")
    if joint_velocity.shape != requested_effort.shape:
        raise ValueError("joint_velocity and requested_effort shapes must match")
    result = np.empty_like(joint_velocity)
    for index, motor_type in enumerate(MOTOR_TYPE):
        curve = MOTOR_CURVES[motor_type]
        velocity = np.abs(joint_velocity[..., index])
        same_direction = (
            joint_velocity[..., index] * requested_effort[..., index]
        ) > 0.0
        plateau = np.where(same_direction, curve.y1, curve.y2)
        ramp = plateau * (curve.x2 - velocity) / (curve.x2 - curve.x1)
        result[..., index] = np.where(
            velocity <= curve.x1,
            plateau,
            np.clip(ramp, 0.0, None),
        )
    return np.minimum(result, COMMAND_EFFORT_LIMIT)


def compute_pd_effort(
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    position_target: np.ndarray,
) -> np.ndarray:
    """Compute PD torque with the Unitree torque-speed and friction envelope."""

    joint_position = _vector(joint_position, ACTION_DIM, "joint_position")
    joint_velocity = _vector(joint_velocity, ACTION_DIM, "joint_velocity")
    position_target = _vector(position_target, ACTION_DIM, "position_target")
    if not (
        joint_position.shape == joint_velocity.shape == position_target.shape
    ):
        raise ValueError("joint state and target shapes must match")
    requested = STIFFNESS * (position_target - joint_position)
    requested -= DAMPING * joint_velocity
    limit = torque_speed_limit(joint_velocity, requested)
    effort = np.clip(requested, -limit, limit)
    for index, motor_type in enumerate(MOTOR_TYPE):
        curve = MOTOR_CURVES[motor_type]
        velocity = joint_velocity[..., index]
        effort[..., index] -= curve.static_friction * np.tanh(
            velocity / curve.activation_velocity
        )
        effort[..., index] -= curve.dynamic_friction * velocity
    return np.clip(effort, -COMMAND_EFFORT_LIMIT, COMMAND_EFFORT_LIMIT)


def contract_dict() -> dict[str, object]:
    """Return a stable, JSON-serializable backend contract manifest."""

    return {
        "robot": "unitree_g1_29dof",
        "joint_order": list(SDK_JOINT_NAMES),
        "official_policy_to_sdk": list(OFFICIAL_POLICY_TO_SDK),
        "sim_dt": SIM_DT,
        "decimation": DECIMATION,
        "policy_dt": POLICY_DT,
        "history_length": HISTORY_LENGTH,
        "observation_terms": OBSERVATION_TERM_DIMS,
        "action_scale_rad": ACTION_SCALE_RAD,
        "default_joint_position": DEFAULT_JOINT_POSITION.tolist(),
        "stiffness": STIFFNESS.tolist(),
        "damping": DAMPING.tolist(),
        "joint_lower": JOINT_LOWER.tolist(),
        "joint_upper": JOINT_UPPER.tolist(),
        "command_effort_limit": COMMAND_EFFORT_LIMIT.tolist(),
        "velocity_limit": VELOCITY_LIMIT.tolist(),
        "motor_type": list(MOTOR_TYPE),
    }


def contract_sha256() -> str:
    payload = json.dumps(
        contract_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_contract() -> None:
    """Raise if a constant or derived dimension has drifted."""

    arrays = (
        DEFAULT_JOINT_POSITION,
        STIFFNESS,
        DAMPING,
        JOINT_LOWER,
        JOINT_UPPER,
        COMMAND_EFFORT_LIMIT,
        VELOCITY_LIMIT,
    )
    if any(array.shape != (ACTION_DIM,) for array in arrays):
        raise AssertionError("G1 contract array length drifted")
    if len(MOTOR_TYPE) != ACTION_DIM:
        raise AssertionError("G1 motor-type length drifted")
    if set(OFFICIAL_POLICY_TO_SDK) != set(range(ACTION_DIM)):
        raise AssertionError("upstream policy-to-SDK map is not a permutation")
    if np.any(JOINT_LOWER >= JOINT_UPPER):
        raise AssertionError("invalid G1 joint range")
    if np.any(DEFAULT_JOINT_POSITION < JOINT_LOWER) or np.any(
        DEFAULT_JOINT_POSITION > JOINT_UPPER
    ):
        raise AssertionError("default G1 pose violates a hard joint limit")
    if SINGLE_FRAME_OBSERVATION_DIM != 96 or OBSERVATION_DIM != 480:
        raise AssertionError("G1 observation dimension drifted")


validate_contract()
