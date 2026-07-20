"""Flat-ground RL task for the serial wheel-legged teaching robot."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials import RigidBodyMaterialCfg

from isaaclab_tasks.manager_based.locomotion.velocity import mdp as base_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    CommandsCfg as BaseCommandsCfg,
    LocomotionVelocityRoughEnvCfg,
)

from . import wheel_legged_mdp
from .wheel_legged_cfg import (
    LEG_JOINT_NAMES,
    POLICY_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    WHEEL_LEGGED_CFG,
    WHEEL_LEGGED_DELAYED_CFG,
)


@configclass
class WheelLeggedCommandsCfg(BaseCommandsCfg):
    """Non-holonomic forward/yaw commands; lateral command is always zero."""

    base_velocity = base_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=base_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.5, 1.5),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class WheelLeggedActionsCfg:
    """Four leg position targets followed by two wheel velocity targets."""

    leg_position = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        preserve_order=True,
        scale=0.45,
        use_default_offset=True,
    )
    wheel_velocity = base_mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        preserve_order=True,
        scale=20.0,
        use_default_offset=False,
    )


@configclass
class WheelLeggedPolicyObsCfg(ObsGroup):
    """28-D observation with no unbounded wheel-angle state."""

    base_lin_vel = ObsTerm(
        func=base_mdp.base_lin_vel,
        noise=Unoise(n_min=-0.10, n_max=0.10),
    )
    base_ang_vel = ObsTerm(
        func=base_mdp.base_ang_vel,
        noise=Unoise(n_min=-0.10, n_max=0.10),
    )
    projected_gravity = ObsTerm(
        func=base_mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(
        func=base_mdp.generated_commands,
        params={"command_name": "base_velocity"},
    )
    leg_joint_pos = ObsTerm(
        func=wheel_legged_mdp.leg_joint_pos_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
            )
        },
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel = ObsTerm(
        func=wheel_legged_mdp.joint_vel_scaled,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=POLICY_JOINT_NAMES, preserve_order=True
            )
        },
        noise=Unoise(n_min=-0.075, n_max=0.075),
    )
    actions = ObsTerm(func=base_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class WheelLeggedObservationsCfg:
    policy: WheelLeggedPolicyObsCfg = WheelLeggedPolicyObsCfg()


@configclass
class WheelLeggedRewardsCfg:
    """Dense velocity tracking plus balance and hardware-friendly penalties."""

    track_lin_vel_xy_exp = RewTerm(
        func=base_mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.45},
    )
    track_ang_vel_z_exp = RewTerm(
        func=base_mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.45},
    )
    commanded_forward_progress = RewTerm(
        func=wheel_legged_mdp.commanded_forward_progress,
        weight=0.75,
        params={"command_name": "base_velocity"},
    )
    alive = RewTerm(func=base_mdp.is_alive, weight=0.25)
    termination = RewTerm(func=base_mdp.is_terminated, weight=-200.0)

    flat_orientation_l2 = RewTerm(func=base_mdp.flat_orientation_l2, weight=-8.0)
    base_height_l2 = RewTerm(
        func=base_mdp.base_height_l2,
        weight=-20.0,
        params={"target_height": 0.50},
    )
    lin_vel_z_l2 = RewTerm(func=base_mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=base_mdp.ang_vel_xy_l2, weight=-0.10)
    lateral_velocity_l2 = RewTerm(
        func=wheel_legged_mdp.lateral_velocity_l2,
        weight=-1.0,
    )
    leg_symmetry_l2 = RewTerm(
        func=wheel_legged_mdp.left_right_leg_symmetry_l2,
        weight=-0.50,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
            )
        },
    )
    leg_deviation_l1 = RewTerm(
        func=base_mdp.joint_deviation_l1,
        weight=-0.10,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
            )
        },
    )
    wheel_velocity_prior_l2 = RewTerm(
        func=wheel_legged_mdp.wheel_velocity_mismatch_l2,
        weight=-0.05,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True
            ),
        },
    )
    joint_torques_l2 = RewTerm(func=base_mdp.joint_torques_l2, weight=-1.0e-4)
    joint_acc_l2 = RewTerm(func=base_mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    leg_contacts = RewTerm(
        func=base_mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*_upper_link", ".*_lower_link"]
            ),
            "threshold": 1.0,
        },
    )
    leg_joint_limits = RewTerm(
        func=base_mdp.joint_pos_limits,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
            )
        },
    )


@configclass
class WheelLeggedTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=base_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": 1.0,
        },
    )
    tipped = DoneTerm(
        func=base_mdp.bad_orientation,
        params={"limit_angle": 0.90},
    )
    too_low = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.25},
    )


@configclass
class WheelLeggedEventsCfg:
    """Moderate domain randomization enabled from the first baseline."""

    physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.3),
            "dynamic_friction_range": (0.5, 1.2),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )
    base_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.90, 1.10),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    base_com = EventTerm(
        func=base_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (-0.01, 0.01),
            },
        },
    )
    actuator_gains = EventTerm(
        func=base_mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINT_NAMES),
            "stiffness_distribution_params": (0.90, 1.10),
            "damping_distribution_params": (0.90, 1.10),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.10, 0.10),
                "y": (-0.10, 0.10),
                "roll": (-0.03, 0.03),
                "pitch": (-0.05, 0.05),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.10, 0.10),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-0.10, 0.10),
                "pitch": (-0.10, 0.10),
                "yaw": (-0.10, 0.10),
            },
        },
    )
    reset_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.10, 0.10),
        },
    )
    push_robot = EventTerm(
        func=base_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={
            "velocity_range": {
                "x": (-0.40, 0.40),
                "y": (-0.25, 0.25),
                "yaw": (-0.30, 0.30),
            }
        },
    )


@configclass
class WheelLeggedFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Isaac Sim 6.0.1 PhysX task, designed as the sim2sim source model."""

    sim: SimulationCfg = SimulationCfg(
        physics=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    )
    observations: WheelLeggedObservationsCfg = WheelLeggedObservationsCfg()
    actions: WheelLeggedActionsCfg = WheelLeggedActionsCfg()
    commands: WheelLeggedCommandsCfg = WheelLeggedCommandsCfg()
    rewards: WheelLeggedRewardsCfg = WheelLeggedRewardsCfg()
    terminations: WheelLeggedTerminationsCfg = WheelLeggedTerminationsCfg()
    events: WheelLeggedEventsCfg = WheelLeggedEventsCfg()
    curriculum = None

    def __post_init__(self):
        self.scene.robot = WHEEL_LEGGED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.scene.height_scanner = None
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=False,
        )

        super().__post_init__()

        self.scene.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(intensity=750.0),
        )
        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.0
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class WheelLeggedFlatEnvCfg_PLAY(WheelLeggedFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class WheelLeggedRobustEnvCfg(WheelLeggedFlatEnvCfg):
    """Stage-two task with 0--20 ms delay and wider dynamics randomization."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = WHEEL_LEGGED_DELAYED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.events.physics_material.params.update(
            {
                "static_friction_range": (0.45, 1.40),
                "dynamic_friction_range": (0.40, 1.30),
                "restitution_range": (0.0, 0.08),
            }
        )
        self.events.base_mass.params["mass_distribution_params"] = (0.85, 1.15)
        self.events.base_com.params["com_range"] = {
            "x": (-0.025, 0.025),
            "y": (-0.025, 0.025),
            "z": (-0.012, 0.012),
        }
        self.events.actuator_gains.params["stiffness_distribution_params"] = (
            0.85,
            1.15,
        )
        self.events.actuator_gains.params["damping_distribution_params"] = (
            0.85,
            1.15,
        )
        self.events.push_robot.interval_range_s = (5.0, 8.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.60, 0.60),
            "y": (-0.40, 0.40),
            "yaw": (-0.40, 0.40),
        }
