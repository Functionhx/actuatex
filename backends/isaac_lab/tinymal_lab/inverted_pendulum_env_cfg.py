"""Isaac Lab DirectRLEnv configurations for serial inverted pendulums."""

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg

from tasks.inverted_pendulum.contract import (
    ACTION_FORCE_SCALE_N,
    DECIMATION,
    EPISODE_LENGTH_S,
    INITIAL_ANGLE_RANGE_RAD,
    OBSERVATION_DIM,
    SIM_DT,
    TERMINATION_CART_POSITION_M,
)

from .inverted_pendulum_cfg import make_inverted_pendulum_cfg


@configclass
class InvertedPendulum1EnvCfg(DirectRLEnvCfg):
    order = 1
    decimation = DECIMATION
    episode_length_s = EPISODE_LENGTH_S
    action_space = 1
    observation_space = OBSERVATION_DIM
    state_space = 0
    action_force_scale = ACTION_FORCE_SCALE_N
    max_cart_position = TERMINATION_CART_POSITION_M
    initial_angle_range = INITIAL_ANGLE_RANGE_RAD[1]

    sim: SimulationCfg = SimulationCfg(
        dt=SIM_DT,
        render_interval=DECIMATION,
        physics=PhysxCfg(),
    )
    robot_cfg = make_inverted_pendulum_cfg(1).replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=6.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )


@configclass
class InvertedPendulum2EnvCfg(InvertedPendulum1EnvCfg):
    order = 2
    initial_angle_range = INITIAL_ANGLE_RANGE_RAD[2]
    robot_cfg = make_inverted_pendulum_cfg(2).replace(
        prim_path="/World/envs/env_.*/Robot"
    )


@configclass
class InvertedPendulum3EnvCfg(InvertedPendulum1EnvCfg):
    order = 3
    initial_angle_range = INITIAL_ANGLE_RANGE_RAD[3]
    robot_cfg = make_inverted_pendulum_cfg(3).replace(
        prim_path="/World/envs/env_.*/Robot"
    )
