"""Observation helpers for the TinyMal Isaac Lab port.

The legged_gym observation applies per-term scales that Isaac Lab's default
``mdp.*`` observation functions do NOT apply:
    base_lin_vel * 2.0, base_ang_vel * 0.25, commands * [2, 2, 0.25],
    (dof_pos - default) * 1.0, dof_vel * 0.05, last_action (no scale)

To make the Isaac Lab observation bit-for-bit compatible with the Isaac Gym
baseline checkpoint (and with tools/sim2sim/observation_builder.py),
we wrap the default functions and multiply by the exact scale.

DOF ORDER
    The URDF declares joints in policy order FL,FR,RL,RR x hip,thigh,calf. The
    Isaac Lab articulation's internal joint order is whatever USD traversal
    produces, so every joint obs/action term uses ``joint_names=POLICY_JOINT_NAMES``
    to FORCE the policy order regardless of the articulation's internal ordering.
"""

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp import last_action as _last_action
from isaaclab.sensors import ContactSensor

# Exact URDF/policy DOF order (verified against sim2sim/observation_builder.py).
POLICY_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# Reusable entity cfg that resolves to policy-ordered joint indices.
POLICY_JOINT_CFG = SceneEntityCfg(
    "robot", joint_names=POLICY_JOINT_NAMES, preserve_order=True
)

# legged_gym normalization.obs_scales (see observation_builder.py).
SCALE_LIN_VEL = 2.0
SCALE_ANG_VEL = 0.25
SCALE_DOF_VEL = 0.05
COMMANDS_SCALE = (2.0, 2.0, 0.25)


def base_lin_vel_scaled(env, scale: float = SCALE_LIN_VEL,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, :3] * scale


def base_ang_vel_scaled(env, scale: float = SCALE_ANG_VEL,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, :3] * scale


def joint_vel_scaled(env, scale: float = SCALE_DOF_VEL,
                     asset_cfg: SceneEntityCfg = POLICY_JOINT_CFG) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    # default joint vel is zero -> relative == absolute, but keep the subtraction for safety.
    return (asset.data.joint_vel[:, joint_ids] - asset.data.default_joint_vel[:, joint_ids]) * scale


def joint_pos_rel_policy(env, asset_cfg: SceneEntityCfg = POLICY_JOINT_CFG) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    return asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]


def last_action_policy(env, action_name: str = "joint_pos") -> torch.Tensor:
    # JointPositionAction is configured with joint_names=POLICY_JOINT_NAMES, so the
    # raw action is already in policy order.
    return _last_action(env, action_name=action_name)


def scaled_commands(env, command_name: str = "base_velocity",
                    scale=COMMANDS_SCALE) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)[:, :3]
    scale_t = torch.tensor(scale, dtype=cmd.dtype, device=cmd.device)
    return cmd * scale_t


def velocity_tracking_l2(env, command_name: str = "base_velocity",
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Squared planar tracking error with a non-saturating gradient at zero velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :2]
    return torch.sum(torch.square(command - asset.data.root_lin_vel_b[:, :2]), dim=1)


def yaw_velocity_tracking_l2(env, command_name: str = "base_velocity",
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Dense yaw-rate tracking error; avoids saturation of the exponential objective."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, 2]
    return torch.square(command - asset.data.root_ang_vel_b[:, 2])


def commanded_planar_progress(env, command_name: str = "base_velocity",
                              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Velocity projected onto the commanded direction to break the standing attractor."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :2]
    command_norm = torch.linalg.vector_norm(command, dim=1, keepdim=True)
    direction = command / torch.clamp(command_norm, min=1.0e-6)
    progress = torch.sum(asset.data.root_lin_vel_b[:, :2] * direction, dim=1)
    return torch.clamp(progress, min=-1.0, max=1.0) * (command_norm[:, 0] > 0.1)


def configure_stair_cells(env, env_ids, flat_fraction: float = 0.5,
                          stair_asset_names=()) -> None:
    """Move the staircase underground in a deterministic fraction of environments."""
    flat_count = int(round(env.num_envs * flat_fraction))
    flat_env_ids = torch.arange(flat_count, device=env.device, dtype=torch.long)
    if flat_env_ids.numel() == 0:
        return
    for asset_name in stair_asset_names:
        asset: RigidObject = env.scene[asset_name]
        pose = asset.data.default_root_state[flat_env_ids, :7].clone()
        pose[:, 2] = -2.0
        asset.write_root_pose_to_sim(pose, env_ids=flat_env_ids)
        asset.write_root_velocity_to_sim(
            torch.zeros(flat_env_ids.numel(), 6, device=env.device),
            env_ids=flat_env_ids,
        )


def stair_height_below_base(env, start_x: float = 0.65, step_width: float = 0.14,
                            step_height: float = 0.02, num_steps: int = 5,
                            corridor_half_width: float = 0.6,
                            first_step_asset: str = "stair_step_1") -> torch.Tensor:
    asset: RigidObject = env.scene[first_step_asset]
    robot: Articulation = env.scene["robot"]
    local_xy = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    level = torch.floor((local_xy[:, 0] - start_x) / step_width) + 1.0
    level = torch.clamp(level, min=0.0, max=float(num_steps))
    inside = torch.abs(local_xy[:, 1]) <= corridor_half_width
    staircase_enabled = asset.data.root_pos_w[:, 2] > -1.0
    return level * step_height * inside * staircase_enabled


def stair_relative_base_height_l2(
    env,
    target_height: float = 0.24,
    start_x: float = 0.65,
    step_width: float = 0.14,
    step_height: float = 0.02,
    num_steps: int = 5,
    corridor_half_width: float = 0.6,
    first_step_asset: str = "stair_step_1",
) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    terrain_height = stair_height_below_base(
        env, start_x, step_width, step_height, num_steps,
        corridor_half_width, first_step_asset
    )
    relative_height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - terrain_height
    return torch.square(relative_height - target_height)


def stair_height_progress(
    env,
    start_x: float = 0.65,
    step_width: float = 0.14,
    step_height: float = 0.02,
    num_steps: int = 5,
    corridor_half_width: float = 0.6,
    first_step_asset: str = "stair_step_1",
) -> torch.Tensor:
    height = stair_height_below_base(
        env, start_x, step_width, step_height, num_steps,
        corridor_half_width, first_step_asset
    )
    return height / max(step_height * num_steps, 1.0e-6)


def stair_swing_foot_clearance_l2(
    env,
    target_clearance: float = 0.055,
    contact_threshold: float = 1.0,
    start_x: float = 0.65,
    step_width: float = 0.14,
    step_height: float = 0.02,
    num_steps: int = 5,
    corridor_half_width: float = 0.6,
    first_step_asset: str = "stair_step_1",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces",
        body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    """Penalize insufficient swing-foot clearance over an enabled staircase.

    The target is relative to the surface below each foot, rather than world
    height.  This produces a useful gradient before a foot hits the next riser
    while leaving stance feet and flat-only environments untouched.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    local_x = foot_pos_w[:, :, 0] - env.scene.env_origins[:, 0].unsqueeze(1)
    local_y = foot_pos_w[:, :, 1] - env.scene.env_origins[:, 1].unsqueeze(1)
    level = torch.floor((local_x - start_x) / step_width) + 1.0
    level = torch.clamp(level, min=0.0, max=float(num_steps))
    inside = torch.abs(local_y) <= corridor_half_width
    staircase_enabled = (
        env.scene[first_step_asset].data.root_pos_w[:, 2] > -1.0
    ).unsqueeze(1)
    terrain_height = level * step_height * inside * staircase_enabled
    foot_height = (
        foot_pos_w[:, :, 2]
        - env.scene.env_origins[:, 2].unsqueeze(1)
        - terrain_height
    )

    force_history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    peak_force = torch.linalg.vector_norm(force_history, dim=-1).amax(dim=1)
    swing = peak_force < contact_threshold
    clearance_deficit = torch.clamp(target_clearance - foot_height, min=0.0)

    command = env.command_manager.get_command(command_name)
    moving_forward = command[:, 0] > 0.1
    base_local_x = robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    near_course = (base_local_x >= start_x - 0.30) & (
        base_local_x <= start_x + num_steps * step_width + 0.30
    )
    active = moving_forward & near_course & staircase_enabled[:, 0]
    return torch.sum(torch.square(clearance_deficit) * swing, dim=1) * active


def stair_course_complete(env, threshold_x: float = 1.55,
                          first_step_asset: str = "stair_step_1") -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    local_x = robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    staircase_enabled = env.scene[first_step_asset].data.root_pos_w[:, 2] > -1.0
    return (local_x >= threshold_x) & staircase_enabled


def joint_torques_scaled(env, scale: float = 1.0 / 12.0,
                         asset_cfg: SceneEntityCfg = POLICY_JOINT_CFG) -> torch.Tensor:
    """Applied joint torques for the asymmetric critic, in policy order."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids] * scale


def foot_contact_state(env, threshold: float = 1.0,
                       sensor_cfg: SceneEntityCfg = SceneEntityCfg(
                           "contact_forces",
                           body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                           preserve_order=True,
                       )) -> torch.Tensor:
    """Binary foot contacts in FL, FR, RL, RR order for the privileged critic."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_history = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    peak_force = torch.linalg.vector_norm(force_history, dim=-1).amax(dim=1)
    return (peak_force > threshold).to(dtype=peak_force.dtype)


def stair_relative_base_height(
    env,
    target_height: float = 0.24,
    start_x: float = 0.65,
    step_width: float = 0.14,
    step_height: float = 0.02,
    num_steps: int = 5,
    corridor_half_width: float = 0.6,
    first_step_asset: str = "stair_step_1",
) -> torch.Tensor:
    """Signed base-height error relative to the local step surface."""
    robot: Articulation = env.scene["robot"]
    terrain_height = stair_height_below_base(
        env, start_x, step_width, step_height, num_steps,
        corridor_half_width, first_step_asset
    )
    relative_height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - terrain_height
    return (relative_height - target_height).unsqueeze(-1)


def stair_level_observation(
    env,
    start_x: float = 0.65,
    step_width: float = 0.14,
    step_height: float = 0.02,
    num_steps: int = 5,
    corridor_half_width: float = 0.6,
    first_step_asset: str = "stair_step_1",
) -> torch.Tensor:
    """Normalized local stair height, available only to the training critic."""
    height = stair_height_below_base(
        env, start_x, step_width, step_height, num_steps,
        corridor_half_width, first_step_asset
    )
    return (height / max(step_height * num_steps, 1.0e-6)).unsqueeze(-1)


def actuator_latents(env, actuator_name: str = "all_dofs") -> torch.Tensor:
    """Randomized PD gains and delay, normalized for the asymmetric critic."""
    actuator = env.scene["robot"].actuators[actuator_name]
    stiffness = actuator.stiffness.mean(dim=1, keepdim=True) / 20.0
    damping = actuator.damping.mean(dim=1, keepdim=True) / 0.5
    if hasattr(actuator, "positions_delay_buffer"):
        delay = actuator.positions_delay_buffer.time_lags.to(dtype=stiffness.dtype).unsqueeze(-1) / 12.0
    else:
        delay = torch.zeros_like(stiffness)
    return torch.cat((stiffness, damping, delay), dim=1)


# Re-export projected_gravity so all obs terms can come from one import namespace.
from isaaclab.envs.mdp import projected_gravity  # noqa: E402,F401
