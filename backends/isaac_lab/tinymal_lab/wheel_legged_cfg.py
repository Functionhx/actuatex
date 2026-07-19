"""Serial wheel-legged robot asset for Isaac Sim 6.0.1 / Isaac Lab 3.0.

The robot is intentionally built from URDF primitives so the teaching example
is auditable and does not depend on an opaque third-party CAD package.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

from .sim6_compat import enable_nested_rigid_body_contact_sensors

enable_nested_rigid_body_contact_sensors()

_REPO_ROOT = Path(__file__).resolve().parents[3]
WHEEL_LEGGED_URDF_PATH = str(
    (
        _REPO_ROOT
        / "robots"
        / "wheel_legged"
        / "urdf"
        / "actuatex_serial_wheel_legged.urdf"
    ).resolve()
)

LEG_JOINT_NAMES = [
    "left_hip_joint",
    "left_knee_joint",
    "right_hip_joint",
    "right_knee_joint",
]
WHEEL_JOINT_NAMES = ["left_wheel_joint", "right_wheel_joint"]
POLICY_JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES

_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=100.0,
    max_angular_velocity=100.0,
    max_depenetration_velocity=1.0,
)

_ARTICULATION_PROPS = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=2,
)


WHEEL_LEGGED_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=WHEEL_LEGGED_URDF_PATH,
        merge_fixed_joints=True,
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
        pos=(0.0, 0.0, 0.50),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "left_hip_joint": 0.35,
            "left_knee_joint": -0.70,
            "right_hip_joint": 0.35,
            "right_knee_joint": -0.70,
            ".*_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_knee_joint"],
            effort_limit=30.0,
            effort_limit_sim=30.0,
            velocity_limit=20.0,
            velocity_limit_sim=20.0,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
            friction=0.02,
        ),
        "wheels": IdealPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=12.0,
            effort_limit_sim=12.0,
            velocity_limit=50.0,
            velocity_limit_sim=50.0,
            stiffness=0.0,
            damping=0.5,
            armature=0.02,
            friction=0.0,
        ),
    },
)


# Stage-two sim-to-real source asset.  Delay is sampled independently per
# environment in physics ticks: 0--4 ticks at 5 ms = 0--20 ms.
WHEEL_LEGGED_DELAYED_CFG = WHEEL_LEGGED_CFG.replace(
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_knee_joint"],
            effort_limit=30.0,
            effort_limit_sim=30.0,
            velocity_limit=20.0,
            velocity_limit_sim=20.0,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
            friction=0.02,
            min_delay=0,
            max_delay=4,
        ),
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=12.0,
            effort_limit_sim=12.0,
            velocity_limit=50.0,
            velocity_limit_sim=50.0,
            stiffness=0.0,
            damping=0.5,
            armature=0.02,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),
    }
)
