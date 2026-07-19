# TinyMal articulation for Isaac Lab.
# Mirrors legged_gym tinymal_config.asset + control exactly:
#   - 12 revolute DOF (FL,FR,RL,RR x hip,thigh,calf), 4 fixed foot links kept (for contacts)
#   - implicit PD position controller: Kp=20, Kd=0.5, effort_limit=12, velocity_limit=20
#   - default joint angles: hip +-0.16, thigh 0.68, calf 1.3
#   - soft_dof_pos_limit = 0.9  (legged_gym soft_dof_pos_limit)

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

# Keep robot assets canonical at the repository root.  Isaac Sim 6 removed the
# importer option that converted URDF cylinders to capsules.  The old Gym/Lab
# experiments used that option, so the checked-in USD (generated and audited
# with the 5.1 importer) is the fidelity-preserving default.  Setting
# ACTUATEX_TINYMAL_URDF explicitly selects the 6.0 native-cylinder import path
# for controlled A/B experiments; ACTUATEX_TINYMAL_USD overrides the USD path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
TINYMAL_CAPSULE_USD_PATH = str(
    Path(
        os.environ.get(
            "ACTUATEX_TINYMAL_USD",
            _REPO_ROOT
            / "robots"
            / "tinymal"
            / "usd"
            / "capsule_compat"
            / "tinymal.usd",
        )
    ).expanduser().resolve()
)
TINYMAL_URDF_PATH = str(
    Path(
        os.environ.get(
            "ACTUATEX_TINYMAL_URDF",
            _REPO_ROOT / "robots" / "tinymal" / "urdf" / "tinymal.urdf",
        )
    ).expanduser().resolve()
)


_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
)
_ARTICULATION_PROPS = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=4,
    solver_velocity_iteration_count=0,
)


if "ACTUATEX_TINYMAL_URDF" in os.environ:
    _TINYMAL_SPAWN_CFG = sim_utils.UrdfFileCfg(
        asset_path=TINYMAL_URDF_PATH,
        # Keep fixed foot links as real bodies for contact sensing.
        merge_fixed_joints=False,
        fix_base=False,
        collision_from_visuals=False,
        collision_type="Convex Hull",
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=20.0, damping=0.5
            ),
        ),
        activate_contact_sensors=True,
        rigid_props=_RIGID_PROPS,
        articulation_props=_ARTICULATION_PROPS,
    )
else:
    _TINYMAL_SPAWN_CFG = sim_utils.UsdFileCfg(
        usd_path=TINYMAL_CAPSULE_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=_RIGID_PROPS,
        articulation_props=_ARTICULATION_PROPS,
    )

# Default joint angles (exact copy of tinymal_config.init_state.default_joint_angles).
# Hip signs: FL/RL = -0.16, FR/RR = +0.16 (matches URDF axis convention used in training).
_DEFAULT_JOINT_POS = {
    "FL_hip_joint": -0.16,
    "FR_hip_joint": 0.16,
    "RL_hip_joint": -0.16,
    "RR_hip_joint": 0.16,
    "FL_thigh_joint": 0.68,
    "FR_thigh_joint": 0.68,
    "RL_thigh_joint": 0.68,
    "RR_thigh_joint": 0.68,
    "FL_calf_joint": 1.3,
    "FR_calf_joint": 1.3,
    "RL_calf_joint": 1.3,
    "RR_calf_joint": 1.3,
}


TINYMAL_CFG = ArticulationCfg(
    spawn=_TINYMAL_SPAWN_CFG,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.28),  # legged_gym init_state.pos
        # Isaac Lab 3.0 standardized every quaternion-facing API on XYZW.
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos=_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,  # legged_gym rewards.soft_dof_pos_limit = 0.9
    actuators={
        # legged_gym control_type="P" computes and clips PD torques explicitly on
        # every physics step.  IdealPDActuator reproduces that law; an implicit
        # PhysX drive is not dynamically equivalent and made the legacy actor
        # settle without producing locomotion in this port.
        "all_dofs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=12.0,       # explicit torque clip, as in legged_gym
            effort_limit_sim=12.0,   # matching solver safety limit
            velocity_limit=20.0,
            velocity_limit_sim=20.0,
            stiffness=20.0,      # legged_gym control.stiffness {"joint": 20.0}
            damping=0.5,         # legged_gym control.damping   {"joint": 0.5}
        ),
    },
)
