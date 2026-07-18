"""Robust mixed-terrain TinyMal stage with asymmetric critic and actuator delay."""

from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as base_mdp

from . import mdp as tinymal_mdp
from .tinymal_native_env_cfg import (
    TinymalNativeOmniCommandsCfg,
    TinymalNativePolicyObsCfg,
)
from .tinymal_stair_env_cfg import TinymalStairEnvCfg, TinymalStairEventsCfg


@configclass
class TinymalPrivilegedObsCfg(ObsGroup):
    joint_torques = ObsTerm(
        func=tinymal_mdp.joint_torques_scaled,
        params={"asset_cfg": tinymal_mdp.POLICY_JOINT_CFG},
    )
    foot_contacts = ObsTerm(
        func=tinymal_mdp.foot_contact_state,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            )
        },
    )
    relative_base_height = ObsTerm(func=tinymal_mdp.stair_relative_base_height)
    stair_level = ObsTerm(func=tinymal_mdp.stair_level_observation)
    actuator_latents = ObsTerm(func=tinymal_mdp.actuator_latents)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class TinymalRobustObservationsCfg:
    policy: TinymalNativePolicyObsCfg = TinymalNativePolicyObsCfg()
    privileged: TinymalPrivilegedObsCfg = TinymalPrivilegedObsCfg()


@configclass
class TinymalRobustEventsCfg(TinymalStairEventsCfg):
    physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.45, 1.35),
            "dynamic_friction_range": (0.45, 1.35),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 96,
        },
    )
    add_base_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    base_com = EventTerm(
        func=base_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {
                "x": (-0.012, 0.012),
                "y": (-0.012, 0.012),
                "z": (-0.008, 0.008),
            },
        },
    )
    actuator_gains = EventTerm(
        func=base_mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=tinymal_mdp.POLICY_JOINT_NAMES),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.70, 1.30),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    push_robot = EventTerm(
        func=base_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 7.0),
        params={
            "velocity_range": {
                "x": (-0.60, 0.60),
                "y": (-0.60, 0.60),
                "yaw": (-0.50, 0.50),
            }
        },
    )


@configclass
class TinymalRobustEnvCfg(TinymalStairEnvCfg):
    commands: TinymalNativeOmniCommandsCfg = TinymalNativeOmniCommandsCfg()
    observations: TinymalRobustObservationsCfg = TinymalRobustObservationsCfg()
    events: TinymalRobustEventsCfg = TinymalRobustEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Preserve a balanced flat/stair mixture during the final robustness
        # stage; the stair-specialization stage intentionally uses 75% stairs.
        self.events.configure_stairs.params["flat_fraction"] = 0.5
        self.scene.robot.actuators["all_dofs"] = DelayedPDActuatorCfg(
            joint_names_expr=tinymal_mdp.POLICY_JOINT_NAMES,
            effort_limit=12.0,
            effort_limit_sim=12.0,
            velocity_limit=20.0,
            velocity_limit_sim=20.0,
            stiffness=20.0,
            damping=0.5,
            armature=0.01,
            friction=0.05,
            min_delay=4,
            max_delay=12,
        )
        self.observations.policy.enable_corruption = True
        # The stair parent narrows x to forward-only; restore the full robust command set.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.resampling_time_range = (5.0, 8.0)
        self.episode_length_s = 12.0


@configclass
class TinymalRobustStairEnvCfg(TinymalRobustEnvCfg):
    """All-stair robust fine-tuning stage for the presentation policy.

    The general robust task deliberately mixes flat terrain and full
    omnidirectional commands.  Keeping this separate prevents the exact stair
    gait from being diluted while retaining the identical actuator delay,
    observation noise, dynamics randomization, and push disturbances used by
    the general policy.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.configure_stairs.params["flat_fraction"] = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.55)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (12.0, 12.0)
