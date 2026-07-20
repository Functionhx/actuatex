"""Matched 1/2/3-link cart-pole assets for Isaac Sim 6 / Isaac Lab."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg


REPO_ROOT = Path(__file__).resolve().parents[3]


def make_inverted_pendulum_cfg(order: int) -> ArticulationCfg:
    if order not in (1, 2, 3):
        raise ValueError(f"order must be 1, 2 or 3, got {order}")
    urdf_path = (
        REPO_ROOT
        / "robots"
        / "inverted_pendulum"
        / "urdf"
        / f"actuatex_cartpole_{order}.urdf"
    ).resolve()
    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf_path),
            fix_base=True,
            merge_fixed_joints=False,
            collision_from_visuals=False,
            collision_type="Convex Hull",
            self_collision=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="none",
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.0,
                stabilization_threshold=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "cart": ImplicitActuatorCfg(
                joint_names_expr=["cart_slide"],
                effort_limit=20.0,
                effort_limit_sim=20.0,
                velocity_limit=8.0,
                velocity_limit_sim=8.0,
                stiffness=0.0,
                damping=0.05,
                armature=0.001,
                friction=0.0,
            ),
            "passive_poles": ImplicitActuatorCfg(
                joint_names_expr=["pole_.*_hinge"],
                # The pole group is present only so Isaac Lab can preserve
                # the URDF's small viscous hinge damping.  The environment
                # never writes a target or effort to these joint ids.
                effort_limit=1.0,
                effort_limit_sim=1.0,
                velocity_limit=30.0,
                velocity_limit_sim=30.0,
                stiffness=0.0,
                damping=0.002,
                armature=0.001,
                friction=0.0,
            ),
        },
    )
