"""Wheel-legged policy observations and dense balance rewards."""

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .wheel_legged_cfg import LEG_JOINT_NAMES, POLICY_JOINT_NAMES

LEG_JOINT_CFG = SceneEntityCfg(
    "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
)
POLICY_JOINT_CFG = SceneEntityCfg(
    "robot", joint_names=POLICY_JOINT_NAMES, preserve_order=True
)


def leg_joint_pos_rel(env, asset_cfg: SceneEntityCfg = LEG_JOINT_CFG) -> torch.Tensor:
    """Return bounded leg angles only; continuous wheel angles must not leak in."""
    robot: Articulation = env.scene[asset_cfg.name]
    return (
        robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
        - robot.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )


def joint_vel_scaled(
    env,
    scale: float = 0.05,
    asset_cfg: SceneEntityCfg = POLICY_JOINT_CFG,
) -> torch.Tensor:
    """Return all six joint velocities in a stable policy-defined order."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_vel.torch[:, asset_cfg.joint_ids] * scale


def commanded_forward_progress(
    env,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Signed progress along the commanded x direction.

    The standard exponential tracker gives a stationary policy a sizeable
    reward at small commands.  This dense term breaks that early local optimum
    without rewarding overspeed indefinitely.
    """
    robot: RigidObject = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    velocity_x = robot.data.root_lin_vel_b.torch[:, 0]
    direction = torch.sign(command_x)
    active = torch.abs(command_x) > 0.10
    return torch.clamp(velocity_x * direction, min=-1.5, max=1.5) * active


def lateral_velocity_l2(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize non-holonomic lateral slip in the body frame."""
    robot: RigidObject = env.scene[asset_cfg.name]
    return torch.square(robot.data.root_lin_vel_b.torch[:, 1])


def left_right_leg_symmetry_l2(
    env, asset_cfg: SceneEntityCfg = LEG_JOINT_CFG
) -> torch.Tensor:
    """Penalize left/right mismatch for the symmetric two-link mechanism."""
    robot: Articulation = env.scene[asset_cfg.name]
    q = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
    return torch.square(q[:, 0] - q[:, 2]) + torch.square(q[:, 1] - q[:, 3])


def wheel_velocity_mismatch_l2(
    env,
    command_name: str = "base_velocity",
    wheel_radius: float = 0.0675,
    track_width: float = 0.34,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=["left_wheel_joint", "right_wheel_joint"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    """Weak model-based prior for differential-drive wheel speeds.

    PPO remains free to deviate for balancing; this only supplies a useful
    gradient before the policy discovers rolling contact.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    desired_left = (command[:, 0] - 0.5 * track_width * command[:, 2]) / wheel_radius
    desired_right = (command[:, 0] + 0.5 * track_width * command[:, 2]) / wheel_radius
    desired = torch.stack((desired_left, desired_right), dim=1)
    actual = robot.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.square((actual - desired) * 0.05), dim=1)
