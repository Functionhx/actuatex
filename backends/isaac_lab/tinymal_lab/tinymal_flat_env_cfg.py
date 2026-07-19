# TinyMal FLAT velocity-tracking task for Isaac Lab.
# Mirrors legged_gym tinymal_config.py + LeggedRobot base defaults EXACTLY:
#   - 48-dim obs (scaled), 12-dim action (offset from default, scale 0.25)
#   - explicit clipped PD (Kp=20,Kd=0.5), effort 12 Nm, decimation 4, dt 0.005 -> 50 Hz control
#   - flat plane, command ranges vx[-.6,.6] vy[-.3,.3] wz[-.8,.8]
#   - reward weights match the migration spec exactly (10 active terms)
import math

import isaaclab.sim as sim_utils
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials import RigidBodyMaterialCfg

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    CommandsCfg as BaseCommandsCfg,
    EventsCfg as BaseEventCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as base_mdp

from .tinymal_cfg import TINYMAL_CFG
from . import mdp as tinymal_mdp


@configclass
class TinymalCommandsCfg(BaseCommandsCfg):
    """Velocity commands matching tinymal_config.commands.ranges (no heading ctrl)."""

    base_velocity = base_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
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
class TinymalActionsCfg:
    joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=tinymal_mdp.POLICY_JOINT_NAMES,   # force policy DOF order
        preserve_order=True,
        scale=0.25,                                    # legged_gym control.action_scale
        use_default_offset=True,                       # targets = default + scale*action
    )


@configclass
class TinymalPolicyObsCfg(ObsGroup):
    # Order MUST match legged_robot.compute_observations (and observation_builder.py).
    base_lin_vel = ObsTerm(func=tinymal_mdp.base_lin_vel_scaled,
                           noise=Unoise(n_min=-0.2, n_max=0.2))
    base_ang_vel = ObsTerm(func=tinymal_mdp.base_ang_vel_scaled,
                           noise=Unoise(n_min=-0.05, n_max=0.05))
    projected_gravity = ObsTerm(func=tinymal_mdp.projected_gravity,
                                noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_commands = ObsTerm(func=tinymal_mdp.scaled_commands,
                                params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=tinymal_mdp.joint_pos_rel_policy,
                        noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=tinymal_mdp.joint_vel_scaled,
                        noise=Unoise(n_min=-0.075, n_max=0.075))
    actions = ObsTerm(func=tinymal_mdp.last_action_policy, params={"action_name": "joint_pos"})

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class TinymalObservationsCfg:
    policy: TinymalPolicyObsCfg = TinymalPolicyObsCfg()


@configclass
class TinymalRewardsCfg:
    """10 active reward terms matching the migration spec (tracking_sigma=0.25 -> std=0.5)."""

    # task
    track_lin_vel_xy_exp = RewTerm(
        func=base_mdp.track_lin_vel_xy_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=base_mdp.track_ang_vel_z_exp, weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    feet_air_time = RewTerm(
        func=base_mdp.feet_air_time, weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "command_name": "base_velocity", "threshold": 0.5},
    )
    # stability penalties
    lin_vel_z_l2 = RewTerm(func=base_mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=base_mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=base_mdp.flat_orientation_l2, weight=-1.0)
    base_height_l2 = RewTerm(
        func=base_mdp.base_height_l2, weight=-2.0,
        params={"target_height": 0.24},
    )
    stand_still = RewTerm(
        # legged_gym _reward_stand_still = sum|dof_pos-default| when |cmd_xy| < 0.1
        func=base_mdp.stand_still_joint_deviation_l1, weight=-0.1,
        params={"command_name": "base_velocity", "command_threshold": 0.1},
    )
    # limits / actuation penalties
    dof_torques_l2 = RewTerm(func=base_mdp.joint_torques_l2, weight=-0.0002)
    dof_pos_limits = RewTerm(
        func=base_mdp.joint_pos_limits, weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=tinymal_mdp.POLICY_JOINT_NAMES)},
    )


@configclass
class TinymalTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=base_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class TinymalEventCfg(BaseEventCfg):
    """legged_gym domain_rand: push_robots + friction rand ON, base-mass/COM rand OFF."""

    # disable mass / COM randomization (randomize_base_mass=False in old task)
    add_base_mass = None
    base_com = None
    # friction randomization matching legged_gym friction_range [0.5, 1.25]
    physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.5, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    # random push every ~15 s, up to 1 m/s (legged_gym push_interval_s=15, max_push_vel_xy=1.0)
    push_robot = EventTerm(
        func=base_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(15.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class TinymalFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """TinyMal flat-terrain velocity tracking (1:1 port of the Isaac Gym baseline task)."""

    # Pin PhysX explicitly for the 6.0.1 migration. Isaac Lab 3 also exposes
    # Newton presets, but changing both simulator version and physics backend in
    # one experiment would make regression attribution impossible.
    sim: SimulationCfg = SimulationCfg(
        physics=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    )

    # Replace the MDP manager configs wholesale so nothing inherited leaks through.
    observations: TinymalObservationsCfg = TinymalObservationsCfg()
    actions: TinymalActionsCfg = TinymalActionsCfg()
    commands: TinymalCommandsCfg = TinymalCommandsCfg()
    rewards: TinymalRewardsCfg = TinymalRewardsCfg()
    terminations: TinymalTerminationsCfg = TinymalTerminationsCfg()
    events: TinymalEventCfg = TinymalEventCfg()
    curriculum = None  # flat ground -> no terrain curriculum; cmd curriculum not ported (see report)

    def __post_init__(self):
        # Configure the scene BEFORE the parent __post_init__ so the parent picks up
        # the flat-plane terrain (instead of loading ROUGH_TERRAINS_CFG / nucleus assets)
        # and skips the height-scanner update-period branch.
        self.scene.robot = TINYMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
        )

        super().__post_init__()

        # Nucleus-free dome light (the inherited sky_light references a Nucleus HDR that
        # is unavailable offline). Plain dome light is enough for headless training.
        self.scene.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(intensity=750.0),
        )

        # --- sim / control timing (exact): dt=0.005, decimation=4 -> 50 Hz ---
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class TinymalFlatEnvCfg_PLAY(TinymalFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
