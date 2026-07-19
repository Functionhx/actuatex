"""Locomotion tasks for the power/thermal-aware ActuateX Sentinel."""

from isaaclab.utils.configclass import configclass

from tasks.robomaster.contract import SIM_DT
from tasks.robomaster.locomotion import TRACK_WIDTH_M, WHEEL_RADIUS_M

from .sentinel_cfg import SENTINEL_CFG, SENTINEL_DELAYED_CFG
from .wheel_legged_env_cfg import WheelLeggedFlatEnvCfg


@configclass
class SentinelFlatEnvCfg(WheelLeggedFlatEnvCfg):
    """50 Hz policy over the 500 Hz shared Sentinel rigid-body model."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SENTINEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.5
        self.decimation = 10
        self.sim.dt = SIM_DT
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.rewards.base_height_l2.params["target_height"] = 0.515
        self.rewards.wheel_velocity_prior_l2.params.update(
            {"wheel_radius": WHEEL_RADIUS_M, "track_width": TRACK_WIDTH_M}
        )
        self.terminations.too_low.params["minimum_height"] = 0.26
        self.sim.physics.gpu_max_rigid_patch_count = 20 * 2**15


@configclass
class SentinelFlatEnvCfg_PLAY(SentinelFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class SentinelRobustEnvCfg(SentinelFlatEnvCfg):
    """Wider dynamics, impacts and independent 0--20 ms actuator delay."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SENTINEL_DELAYED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.events.physics_material.params.update(
            {
                "static_friction_range": (0.45, 1.45),
                "dynamic_friction_range": (0.40, 1.35),
                "restitution_range": (0.0, 0.10),
            }
        )
        self.events.base_mass.params["mass_distribution_params"] = (0.85, 1.15)
        self.events.base_com.params["com_range"] = {
            "x": (-0.030, 0.030),
            "y": (-0.030, 0.030),
            "z": (-0.015, 0.015),
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
            "x": (-0.70, 0.70),
            "y": (-0.45, 0.45),
            "yaw": (-0.50, 0.50),
        }
