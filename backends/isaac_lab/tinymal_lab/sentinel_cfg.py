"""ActuateX Sentinel asset for Isaac Sim 6.0.1 / Isaac Lab 3."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

from tasks.robomaster.contract import (
    ALL_JOINT_NAMES,
    MOTOR_SPECS,
    SENTINEL_DEFAULT_JOINT_POSITION,
)
from tasks.robomaster.locomotion import (
    JOINT_DAMPING,
    JOINT_STIFFNESS,
    actuator_property_by_name,
)

from .sentinel_actuator import SentinelDCMotorCfg
from .sim6_compat import enable_nested_rigid_body_contact_sensors

enable_nested_rigid_body_contact_sensors()

_REPO_ROOT = Path(__file__).resolve().parents[3]
SENTINEL_URDF_PATH = str(
    (
        _REPO_ROOT
        / "robots"
        / "robomaster"
        / "urdf"
        / "actuatex_sentinel.urdf"
    ).resolve()
)


def _motor_property(attribute: str) -> dict[str, float]:
    return {
        joint_name: float(getattr(spec, attribute))
        for joint_name, spec in zip(ALL_JOINT_NAMES, MOTOR_SPECS)
    }


_STIFFNESS = actuator_property_by_name(JOINT_STIFFNESS)
_DAMPING = actuator_property_by_name(JOINT_DAMPING)
_DEFAULT_JOINT_POS = dict(
    zip(ALL_JOINT_NAMES, SENTINEL_DEFAULT_JOINT_POSITION.tolist())
)

_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=100.0,
    max_angular_velocity=1500.0,
    max_depenetration_velocity=1.0,
)
_ARTICULATION_PROPS = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=2,
)


SENTINEL_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=SENTINEL_URDF_PATH,
        merge_fixed_joints=False,
        fix_base=False,
        collision_from_visuals=False,
        collision_type="Convex Hull",
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",
        ),
        activate_contact_sensors=True,
        rigid_props=_RIGID_PROPS,
        articulation_props=_ARTICULATION_PROPS,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.515),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos=_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "shared_dc_bank": SentinelDCMotorCfg(
            joint_names_expr=list(ALL_JOINT_NAMES),
            expected_joint_names=ALL_JOINT_NAMES,
            effort_limit=_motor_property("joint_torque_limit_nm"),
            effort_limit_sim=_motor_property("joint_torque_limit_nm"),
            velocity_limit=_motor_property("joint_speed_limit_rad_s"),
            velocity_limit_sim=_motor_property("joint_speed_limit_rad_s"),
            stiffness=_STIFFNESS,
            damping=_DAMPING,
            armature=_motor_property("reflected_armature"),
            friction=0.0,
            dynamic_friction=0.0,
            viscous_friction=0.0,
        )
    },
)


# Robust-training twin: a per-environment 0--20 ms command delay at the shared
# 2 ms physics tick, while electrical/thermal state remains identical.
SENTINEL_DELAYED_CFG = SENTINEL_CFG.replace(
    actuators={
        "shared_dc_bank": SENTINEL_CFG.actuators["shared_dc_bank"].replace(
            maximum_command_delay_steps=10
        )
    }
)
