"""PhysX-5-native TinyMal curriculum stages.

The first stage samples only meaningful forward commands and adds a dense
velocity-gradient term.  This deliberately avoids the standing local optimum
seen in the original short port.  The second stage restores omnidirectional
commands after a forward gait exists.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as base_mdp

from . import mdp as tinymal_mdp
from .tinymal_flat_env_cfg import (
    TinymalCommandsCfg,
    TinymalEventCfg,
    TinymalFlatEnvCfg,
    TinymalPolicyObsCfg,
    TinymalRewardsCfg,
)


@configclass
class TinymalNativeForwardCommandsCfg(TinymalCommandsCfg):
    base_velocity = base_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=base_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.25, 0.60),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class TinymalNativeOmniCommandsCfg(TinymalCommandsCfg):
    base_velocity = base_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=base_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 0.6),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.8, 0.8),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class TinymalNativeRewardsCfg(TinymalRewardsCfg):
    # Keep the original exponential objective, but strengthen it and add a
    # non-saturating error/progress pair so standing is not a broad optimum.
    track_lin_vel_xy_exp = RewTerm(
        func=base_mdp.track_lin_vel_xy_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=base_mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    velocity_tracking_l2 = RewTerm(
        func=tinymal_mdp.velocity_tracking_l2,
        weight=-1.5,
        params={"command_name": "base_velocity"},
    )
    commanded_planar_progress = RewTerm(
        func=tinymal_mdp.commanded_planar_progress,
        weight=1.0,
        params={"command_name": "base_velocity"},
    )
    yaw_velocity_tracking_l2 = RewTerm(
        func=tinymal_mdp.yaw_velocity_tracking_l2,
        weight=-0.5,
        params={"command_name": "base_velocity"},
    )

    # Terms inherited by the Isaac-Gym task but omitted in the first Lab port.
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    joint_acc_l2 = RewTerm(
        func=base_mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=tinymal_mdp.POLICY_JOINT_NAMES)},
    )
    undesired_contacts = RewTerm(
        func=base_mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*_thigh", ".*_calf"]
            ),
            "threshold": 0.1,
        },
    )


@configclass
class TinymalNativeEventsCfg(TinymalEventCfg):
    # Learn a gait under nominal dynamics first.  Robustness randomization is
    # enabled in the later terrain stage after locomotion has emerged.
    physics_material = None
    push_robot = None
    add_base_mass = None
    base_com = None


@configclass
class TinymalNativePolicyObsCfg(TinymalPolicyObsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.enable_corruption = False


@configclass
class TinymalNativeForwardEnvCfg(TinymalFlatEnvCfg):
    commands: TinymalNativeForwardCommandsCfg = TinymalNativeForwardCommandsCfg()
    rewards: TinymalNativeRewardsCfg = TinymalNativeRewardsCfg()
    events: TinymalNativeEventsCfg = TinymalNativeEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.replace_cylinders_with_capsules = True
        actuator = self.scene.robot.actuators["all_dofs"]
        actuator.armature = 0.01
        actuator.friction = 0.05
        self.sim.physx.solver_type = 1
        self.sim.physx.enable_external_forces_every_iteration = True
        self.scene.robot.spawn.rigid_props.solver_position_iteration_count = 4
        self.scene.robot.spawn.rigid_props.solver_velocity_iteration_count = 1
        self.observations.policy.enable_corruption = False


@configclass
class TinymalNativeOmniEnvCfg(TinymalNativeForwardEnvCfg):
    commands: TinymalNativeOmniCommandsCfg = TinymalNativeOmniCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True
