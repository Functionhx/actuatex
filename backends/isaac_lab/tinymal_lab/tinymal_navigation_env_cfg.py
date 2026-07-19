"""Single-environment TinyMal arena used by the ROS 2 Nav2 demo."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from .tinymal_native_env_cfg import TinymalNativeOmniEnvCfg


def _static_box(
    name: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
) -> AssetBaseCfg:
    """Create a globally colliding, static navigation obstacle."""

    return AssetBaseCfg(
        prim_path=f"/World/NavArena/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                metallic=0.0,
                roughness=0.72,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        collision_group=-1,
    )


@configclass
class TinymalNavigationEnvCfg(TinymalNativeOmniEnvCfg):
    """Nominal locomotion task inside the arena represented by the Nav2 map."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 0.0
        self.scene.robot.init_state.pos = (-3.5, -3.5, 0.28)
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 3600.0

        # Navigation commands are owned by Nav2 and overwritten every policy step.
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        wall_color = (0.12, 0.18, 0.27)
        obstacle_color = (0.08, 0.55, 0.72)
        arena_objects = {
            "wall_west": _static_box(
                "wall_west", (-4.9, 0.0, 0.25), (0.2, 10.0, 0.5), wall_color
            ),
            "wall_east": _static_box(
                "wall_east", (4.9, 0.0, 0.25), (0.2, 10.0, 0.5), wall_color
            ),
            "wall_south": _static_box(
                "wall_south", (0.0, -4.9, 0.25), (10.0, 0.2, 0.5), wall_color
            ),
            "wall_north": _static_box(
                "wall_north", (0.0, 4.9, 0.25), (10.0, 0.2, 0.5), wall_color
            ),
            "obstacle_vertical": _static_box(
                "obstacle_vertical",
                (-0.8, -1.0, 0.35),
                (0.4, 4.0, 0.7),
                obstacle_color,
            ),
            "obstacle_horizontal": _static_box(
                "obstacle_horizontal",
                (2.25, 0.75, 0.35),
                (2.5, 0.5, 0.7),
                obstacle_color,
            ),
            "obstacle_square": _static_box(
                "obstacle_square",
                (1.85, -1.65, 0.35),
                (0.7, 0.7, 0.7),
                obstacle_color,
            ),
        }
        for name, asset in arena_objects.items():
            setattr(self.scene, name, asset)

        self.viewer.eye = (-1.0, -8.5, 7.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.origin_type = "world"
