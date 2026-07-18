"""Mixed flat / exact 20-mm staircase curriculum for the native Lab policy."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp as tinymal_mdp
from .tinymal_native_env_cfg import (
    TinymalNativeEventsCfg,
    TinymalNativeForwardEnvCfg,
    TinymalNativeRewardsCfg,
)


STAIR_ASSET_NAMES = tuple([f"stair_step_{index}" for index in range(1, 6)] + ["stair_top"])


def _stair_object(name, center_x, size_x, height, color):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(size_x, 1.20, height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.01, rest_offset=0.0
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, metallic=0.0, roughness=0.7
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(center_x, 0.0, height / 2.0)
        ),
    )


@configclass
class TinymalStairEventsCfg(TinymalNativeEventsCfg):
    configure_stairs = EventTerm(
        func=tinymal_mdp.configure_stair_cells,
        mode="startup",
        params={"flat_fraction": 0.25, "stair_asset_names": STAIR_ASSET_NAMES},
    )


@configclass
class TinymalStairRewardsCfg(TinymalNativeRewardsCfg):
    base_height_l2 = RewTerm(
        func=tinymal_mdp.stair_relative_base_height_l2,
        weight=-2.0,
        params={"target_height": 0.24},
    )
    stair_height_progress = RewTerm(
        func=tinymal_mdp.stair_height_progress,
        weight=1.0,
        params={},
    )
    stair_swing_foot_clearance_l2 = RewTerm(
        func=tinymal_mdp.stair_swing_foot_clearance_l2,
        weight=-10.0,
        params={
            "target_clearance": 0.055,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
        },
    )
    stair_course_complete = RewTerm(
        func=tinymal_mdp.stair_course_complete,
        weight=50.0,
        params={"threshold_x": 1.55},
    )


@configclass
class TinymalStairEnvCfg(TinymalNativeForwardEnvCfg):
    rewards: TinymalStairRewardsCfg = TinymalStairRewardsCfg()
    events: TinymalStairEventsCfg = TinymalStairEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        start_x = 0.65
        step_width = 0.14
        step_height = 0.02
        colors = (
            (0.18, 0.38, 0.72),
            (0.20, 0.43, 0.78),
            (0.22, 0.48, 0.84),
            (0.25, 0.53, 0.88),
            (0.29, 0.58, 0.92),
        )
        for index in range(1, 6):
            height = index * step_height
            center_x = start_x + (index - 0.5) * step_width
            setattr(
                self.scene,
                f"stair_step_{index}",
                _stair_object(
                    f"stair_step_{index}", center_x, step_width, height, colors[index - 1]
                ),
            )
        setattr(
            self.scene,
            "stair_top",
            _stair_object("stair_top", 1.625, 0.55, 0.10, (0.32, 0.62, 0.95)),
        )
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (-0.05, 0.05),
            "yaw": (-0.05, 0.05),
        }
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.55)
        self.episode_length_s = 12.0
        self.terminations.stair_course_complete = DoneTerm(
            func=tinymal_mdp.stair_course_complete,
            time_out=True,
            params={"threshold_x": 1.55},
        )
