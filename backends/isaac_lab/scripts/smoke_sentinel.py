#!/usr/bin/env python
"""Spawn and numerically validate the ActuateX Sentinel in Isaac Sim 6."""

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=250)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs <= 0 or args_cli.steps <= 0:
    parser.error("--num_envs and --steps must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402

from tasks.robomaster.contract import ALL_JOINT_NAMES, SIM_DT  # noqa: E402
from tinymal_lab.sentinel_cfg import SENTINEL_CFG  # noqa: E402


@configclass
class SentinelSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2500.0,
            color=(0.75, 0.75, 0.75),
        ),
    )
    robot: ArticulationCfg = SENTINEL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


def main() -> None:
    sim = SimulationContext(
        sim_utils.SimulationCfg(
            dt=SIM_DT,
            render_interval=10,
            device=args_cli.device,
        )
    )
    scene = InteractiveScene(
        SentinelSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    )
    sim.reset()

    robot = scene["robot"]
    root_pose = robot.data.default_root_pose.torch.clone()
    root_pose[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim_index(root_pose=root_pose)
    robot.write_root_velocity_to_sim_index(
        root_velocity=robot.data.default_root_vel.torch
    )
    robot.write_joint_position_to_sim_index(
        position=robot.data.default_joint_pos.torch
    )
    robot.write_joint_velocity_to_sim_index(
        velocity=robot.data.default_joint_vel.torch
    )
    scene.reset()

    position_target = robot.data.default_joint_pos.torch.clone()
    velocity_target = torch.zeros_like(position_target)
    effort_target = torch.zeros_like(position_target)
    for _ in range(args_cli.steps):
        robot.set_joint_position_target_index(target=position_target)
        robot.set_joint_velocity_target_index(target=velocity_target)
        robot.set_joint_effort_target_index(target=effort_target)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(SIM_DT)

    actuator = robot.actuators["shared_dc_bank"]
    tensors = (
        robot.data.root_pose_w.torch,
        robot.data.joint_pos.torch,
        robot.data.joint_vel.torch,
        actuator.applied_effort,
        actuator.motor_temperature_c,
    )
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise RuntimeError("Sentinel Isaac smoke produced a non-finite state")
    if set(robot.joint_names) != set(ALL_JOINT_NAMES):
        raise RuntimeError(
            "Sentinel joint-name set drifted: "
            f"expected={ALL_JOINT_NAMES}, actual={robot.joint_names}"
        )

    result = {
        "backend": "Isaac Sim 6.0.1 GA / Isaac Lab 3.0.0-beta2.patch1 / PhysX 5",
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "dt_s": SIM_DT,
        "joint_names": robot.joint_names,
        "maximum_abs_joint_velocity_rad_s": float(
            robot.data.joint_vel.torch.abs().max()
        ),
        "maximum_abs_applied_torque_nm": float(
            actuator.applied_effort.abs().max()
        ),
        "maximum_motor_temperature_c": float(
            actuator.motor_temperature_c.max()
        ),
        "minimum_buffer_energy_j": float(
            actuator.referee.buffer_energy_j.min()
        ),
        "all_chassis_enabled": bool(actuator.referee.chassis_enabled.all()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
