#!/usr/bin/env python
"""Sim2Sim cross-check: run an Isaac-Gym-trained TinyMal actor in Isaac Lab.

Loads .../tinymal_baseline/Jul17_23-52-15_std0p3_seed1/model_1500.pt
(actor.{0,2,4,6} -> MLP [48,512,256,128,12]) into a fresh torch MLP and rolls it out in the
Isaac Lab TinyMal flat env (PhysX 5). The Isaac Lab obs is built to be bit-compatible with
legged_gym (see tinymal_lab/mdp.py), so the old policy can be evaluated without retraining.

Run:
    "$ISAAC_SIM_PYTHON" play_old_policy.py \
        --vx 0.3 --num_envs 64 --steps 500

Metrics: forward-velocity tracking RMSE (commanded vx vs body-frame root_lin_vel_b[0]) and
survival (fraction of env-steps not terminated by base contact).
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt",
    type=str,
    default=str(
        Path(os.environ.get("ACTUATEX_ARTIFACTS", REPO_ROOT / "artifacts"))
        / "checkpoints"
        / "isaac_gym"
        / "model.pt"
    ),
)
parser.add_argument(
    "--ckpt_glob",
    type=str,
    default=None,
    help="optional checkpoint glob; evaluates every match in one simulator process",
)
parser.add_argument(
    "--ckpt_iterations",
    type=str,
    default=None,
    help="comma-separated model iterations to retain from --ckpt_glob",
)
parser.add_argument("--vx", type=float, default=0.5, help="commanded forward velocity (m/s)")
parser.add_argument("--vy", type=float, default=0.0, help="commanded lateral velocity (m/s)")
parser.add_argument("--yaw", type=float, default=0.0, help="commanded yaw velocity (rad/s)")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=500, help="policy steps per rollout (50 Hz)")
parser.add_argument(
    "--action_delay",
    type=int,
    default=0,
    help="fixed policy-step delay; robust source training sampled 1--3",
)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument(
    "--suite",
    action="store_true",
    help="run the same six command segments as the Isaac Gym/MuJoCo evaluators",
)
parser.add_argument(
    "--out",
    type=str,
    default=None,
    help="optional JSON result path (recommended for policy-specific evidence)",
)
parser.add_argument("--kp", type=float, default=20.0)
parser.add_argument("--kd", type=float, default=0.5)
parser.add_argument("--action_scale", type=float, default=0.25)
parser.add_argument("--effort_limit", type=float, default=12.0)
parser.add_argument("--armature", type=float, default=0.0)
parser.add_argument("--joint_friction", type=float, default=0.0)
parser.add_argument("--actuator", choices=("explicit", "implicit"), default="explicit")
parser.add_argument("--solver_position_iterations", type=int, default=4)
parser.add_argument("--solver_velocity_iterations", type=int, default=0)
parser.add_argument("--solver_type", type=int, choices=(0, 1), default=1)
parser.add_argument("--contact_offset", type=float, default=0.01)
parser.add_argument("--rest_offset", type=float, default=0.0)
parser.add_argument("--replace_cylinders_with_capsules", action="store_true")
parser.add_argument(
    "--non_instanceable",
    action="store_true",
    help="disable imported mesh instancing so collider contact offsets can be authored",
)
parser.add_argument("--external_forces_every_iteration", action="store_true")
parser.add_argument("--positive_vx_gain", type=float, default=1.0)
parser.add_argument("--negative_vx_gain", type=float, default=1.0)
parser.add_argument("--vy_gain", type=float, default=1.0)
parser.add_argument("--yaw_gain", type=float, default=1.0)
args_cli, _ = parser.parse_known_args()

# SimulationApp must be created before importing isaaclab. We keep it at module scope but the
# script is meant to be run directly (not imported while another SimulationApp is alive).
from isaacsim import SimulationApp  # noqa: E402
simulation_app = SimulationApp({"headless": True})

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks, tinymal_lab  # noqa: E402,F401
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg  # noqa: E402
from tinymal_lab.tinymal_flat_env_cfg import TinymalFlatEnvCfg  # noqa: E402
from tinymal_lab.mdp import POLICY_JOINT_NAMES  # noqa: E402


def build_actor(in_dim=48, hidden=(512, 256, 128), out_dim=12):
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ELU()]
        prev = h
    layers += [nn.Linear(prev, out_dim)]
    return nn.Sequential(*layers)


def load_old_actor(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = build_actor().to(device)
    # The checkpoint belongs to ActorCritic and therefore uses keys such as
    # ``actor.0.weight``.  ``net`` is the actor Sequential itself and expects
    # ``0.weight``.  Keeping the prefix with strict=False silently loads zero
    # parameters, which invalidates the sim2sim result.
    if "model_state_dict" in ck:
        actor_state = {
            key[len("actor.") :]: value
            for key, value in ck["model_state_dict"].items()
            if key.startswith("actor.")
        }
    elif "actor_state_dict" in ck:
        actor_state = {
            key[len("mlp.") :]: value
            for key, value in ck["actor_state_dict"].items()
            if key.startswith("mlp.")
        }
    else:
        raise KeyError("checkpoint has neither model_state_dict nor actor_state_dict")
    net.load_state_dict(actor_state, strict=True)
    net.eval()
    return net


def make_env(command, num_envs, device):
    vx, vy, yaw = command
    cfg = TinymalFlatEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.actions.joint_pos.scale = args_cli.action_scale
    actuator_cfg_type = (
        IdealPDActuatorCfg if args_cli.actuator == "explicit" else ImplicitActuatorCfg
    )
    cfg.scene.robot.actuators["all_dofs"] = actuator_cfg_type(
        joint_names_expr=POLICY_JOINT_NAMES,
        effort_limit=args_cli.effort_limit,
        effort_limit_sim=args_cli.effort_limit,
        velocity_limit=20.0,
        velocity_limit_sim=20.0,
        stiffness=args_cli.kp,
        damping=args_cli.kd,
        armature=args_cli.armature,
        friction=args_cli.joint_friction,
    )
    cfg.scene.robot.spawn.rigid_props.solver_position_iteration_count = (
        args_cli.solver_position_iterations
    )
    cfg.scene.robot.spawn.rigid_props.solver_velocity_iteration_count = (
        args_cli.solver_velocity_iterations
    )
    cfg.scene.robot.spawn.replace_cylinders_with_capsules = (
        args_cli.replace_cylinders_with_capsules
    )
    cfg.scene.robot.spawn.make_instanceable = not args_cli.non_instanceable
    if args_cli.non_instanceable:
        cfg.scene.robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=args_cli.contact_offset,
            rest_offset=args_cli.rest_offset,
        )
    cfg.sim.physx.solver_type = args_cli.solver_type
    cfg.sim.physx.enable_external_forces_every_iteration = (
        args_cli.external_forces_every_iteration
    )
    # Deterministic eval: kill noise / pushes / friction jitter; pin a constant forward cmd.
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    cfg.events.physics_material = None
    cfg.events.add_base_mass = None
    cfg.events.base_com = None
    cfg.events.reset_base.params["pose_range"] = {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)}
    cfg.events.reset_base.params["velocity_range"] = {
        k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.ranges.lin_vel_x = (vx, vx)
    cfg.commands.base_velocity.ranges.lin_vel_y = (vy, vy)
    cfg.commands.base_velocity.ranges.ang_vel_z = (yaw, yaw)
    return gym.make("Isaac-Velocity-Flat-TinyMal-v0", cfg=cfg)


SEGMENTS = (
    ("stand", 2.0, (0.0, 0.0, 0.0)),
    ("forward_0p3", 3.0, (0.3, 0.0, 0.0)),
    ("forward_0p6", 3.0, (0.6, 0.0, 0.0)),
    ("backward_0p3", 3.0, (-0.3, 0.0, 0.0)),
    ("lateral_0p2", 3.0, (0.0, 0.2, 0.0)),
    ("yaw_0p5", 3.0, (0.0, 0.0, 0.5)),
)


def _set_command(env, obs, command):
    command_tensor = env.unwrapped.command_manager.get_command("base_velocity")
    command_tensor[:, 0] = command[0]
    command_tensor[:, 1] = command[1]
    command_tensor[:, 2] = command[2]
    # Refresh the policy observation immediately at a segment boundary so the
    # first action does not see the preceding segment's command.
    return _adapt_policy_observation(
        env.unwrapped.observation_manager.compute(), command
    )


def _adapt_policy_observation(obs, command):
    """Keep the physical target unchanged while adapting the policy command input."""
    vx_gain = (
        args_cli.positive_vx_gain if command[0] >= 0.0 else args_cli.negative_vx_gain
    )
    obs["policy"][:, 9] = command[0] * 2.0 * vx_gain
    obs["policy"][:, 10] = command[1] * 2.0 * args_cli.vy_gain
    obs["policy"][:, 11] = command[2] * 0.25 * args_cli.yaw_gain
    return obs


def run_suite(env, actor, obs):
    summary = {}
    action_history = torch.zeros(
        args_cli.action_delay + 1,
        args_cli.num_envs,
        12,
        device=args_cli.device,
    )
    last_policy_action = torch.zeros(args_cli.num_envs, 12, device=args_cli.device)
    with torch.no_grad():
        for name, duration, command in SEGMENTS:
            obs = _set_command(env, obs, command)
            # legged_gym observes the latest policy output even when an older
            # action is selected from its actuator-delay FIFO.
            obs["policy"][:, 36:48] = last_policy_action
            initial_policy_obs = obs["policy"][0].detach().clone()
            initial_action = actor(obs["policy"])[0].detach().clone()
            steps = int(round(duration / 0.02))
            settle_steps = int(round(min(1.0, duration / 3.0) / 0.02))
            sq_error_sum = torch.zeros(3, device=args_cli.device)
            sample_count = 0
            resets = 0
            base_height_sum = 0.0
            action_square_sum = 0.0
            action_sample_count = 0
            for step in range(steps):
                action = torch.clamp(actor(obs["policy"]), -100.0, 100.0)
                action_square_sum += float(torch.square(action).sum().item())
                action_sample_count += action.numel()
                action_history[1:] = action_history[:-1].clone()
                action_history[0] = action
                applied_action = action_history[args_cli.action_delay]
                obs, _, terminated, _, _ = env.step(applied_action)
                obs = _adapt_policy_observation(obs, command)
                last_policy_action = action
                obs["policy"][:, 36:48] = last_policy_action
                resets += int(terminated.sum().item())
                if step >= settle_steps:
                    robot_data = env.unwrapped.scene["robot"].data
                    actual = torch.stack(
                        (
                            robot_data.root_lin_vel_b[:, 0],
                            robot_data.root_lin_vel_b[:, 1],
                            robot_data.root_ang_vel_b[:, 2],
                        ),
                        dim=1,
                    )
                    target = torch.tensor(command, device=actual.device).unsqueeze(0)
                    sq_error_sum += torch.square(actual - target).sum(dim=0)
                    sample_count += actual.shape[0]
                    base_height_sum += float(robot_data.root_pos_w[:, 2].sum().item())
            rmse = torch.sqrt(sq_error_sum / max(1, sample_count))
            summary[name] = {
                "command": list(command),
                "vx_rmse": float(rmse[0].item()),
                "vy_rmse": float(rmse[1].item()),
                "yaw_rmse": float(rmse[2].item()),
                "base_height_mean": base_height_sum / max(1, sample_count),
                "resets_total": resets,
                "step_survival_fraction": 1.0
                - resets / float(max(1, steps * args_cli.num_envs)),
                "action_rms": (action_square_sum / max(1, action_sample_count)) ** 0.5,
                "initial_obs_command": initial_policy_obs[9:12].cpu().tolist(),
                "initial_obs_joint_pos_max_abs": float(
                    initial_policy_obs[12:24].abs().max().item()
                ),
                "initial_obs_joint_vel_max_abs": float(
                    initial_policy_obs[24:36].abs().max().item()
                ),
                "initial_action": initial_action.cpu().tolist(),
            }
    return summary


def run_single(env, actor, obs, command):
    obs = _set_command(env, obs, command)
    sq_error_sum = torch.zeros(3, device=args_cli.device)
    resets = 0
    action_history = torch.zeros(
        args_cli.action_delay + 1,
        args_cli.num_envs,
        12,
        device=args_cli.device,
    )
    with torch.no_grad():
        for _ in range(args_cli.steps):
            action = torch.clamp(actor(obs["policy"]), -100.0, 100.0)
            action_history[1:] = action_history[:-1].clone()
            action_history[0] = action
            applied_action = action_history[args_cli.action_delay]
            obs, _, terminated, _, _ = env.step(applied_action)
            obs = _adapt_policy_observation(obs, command)
            obs["policy"][:, 36:48] = action
            robot_data = env.unwrapped.scene["robot"].data
            actual = torch.stack(
                (
                    robot_data.root_lin_vel_b[:, 0],
                    robot_data.root_lin_vel_b[:, 1],
                    robot_data.root_ang_vel_b[:, 2],
                ),
                dim=1,
            )
            target = torch.tensor(command, device=actual.device).unsqueeze(0)
            sq_error_sum += torch.square(actual - target).sum(dim=0)
            resets += int(terminated.sum().item())
    sample_count = args_cli.steps * args_cli.num_envs
    rmse = torch.sqrt(sq_error_sum / max(1, sample_count))
    return {
        "command": list(command),
        "vx_rmse": float(rmse[0].item()),
        "vy_rmse": float(rmse[1].item()),
        "yaw_rmse": float(rmse[2].item()),
        "resets_total": resets,
        "step_survival_fraction": 1.0 - resets / float(max(1, sample_count)),
    }


def _checkpoint_sort_key(path):
    match = re.search(r"model_(\d+)\.pt$", path)
    return (int(match.group(1)) if match else -1, path)


def _suite_score(segments):
    """Lower is better; emphasize commanded axes and strongly penalize falls."""
    commanded_errors = (
        segments["stand"]["vx_rmse"]
        + segments["stand"]["vy_rmse"]
        + 0.5 * segments["stand"]["yaw_rmse"]
        + segments["forward_0p3"]["vx_rmse"]
        + 0.25 * segments["forward_0p3"]["vy_rmse"]
        + 0.10 * segments["forward_0p3"]["yaw_rmse"]
        + segments["forward_0p6"]["vx_rmse"]
        + 0.25 * segments["forward_0p6"]["vy_rmse"]
        + 0.10 * segments["forward_0p6"]["yaw_rmse"]
        + segments["backward_0p3"]["vx_rmse"]
        + 0.25 * segments["backward_0p3"]["vy_rmse"]
        + 0.10 * segments["backward_0p3"]["yaw_rmse"]
        + segments["lateral_0p2"]["vy_rmse"]
        + 0.25 * segments["lateral_0p2"]["vx_rmse"]
        + 0.10 * segments["lateral_0p2"]["yaw_rmse"]
        + segments["yaw_0p5"]["yaw_rmse"]
        + 0.25 * segments["yaw_0p5"]["vx_rmse"]
        + 0.25 * segments["yaw_0p5"]["vy_rmse"]
    )
    resets = sum(segment["resets_total"] for segment in segments.values())
    return commanded_errors + 0.25 * resets


def main():
    device = args_cli.device
    initial_command = SEGMENTS[0][2] if args_cli.suite else (
        args_cli.vx,
        args_cli.vy,
        args_cli.yaw,
    )
    env = make_env(initial_command, args_cli.num_envs, device)
    checkpoint_paths = (
        sorted(glob.glob(args_cli.ckpt_glob), key=_checkpoint_sort_key)
        if args_cli.ckpt_glob
        else [args_cli.ckpt]
    )
    if args_cli.ckpt_iterations:
        requested_iterations = {
            int(value.strip()) for value in args_cli.ckpt_iterations.split(",") if value.strip()
        }
        checkpoint_paths = [
            path
            for path in checkpoint_paths
            if _checkpoint_sort_key(path)[0] in requested_iterations
        ]
    if not checkpoint_paths:
        raise FileNotFoundError(f"no checkpoints matched: {args_cli.ckpt_glob}")

    checkpoint_results = []
    for checkpoint_path in checkpoint_paths:
        actor = load_old_actor(checkpoint_path, device)
        obs, _ = env.reset()
        result = (
            run_suite(env, actor, obs)
            if args_cli.suite
            else run_single(env, actor, obs, initial_command)
        )
        checkpoint_results.append(
            {
                "checkpoint": os.path.abspath(checkpoint_path),
                "score": _suite_score(result) if args_cli.suite else None,
                "segments": result if args_cli.suite else {"single": result},
            }
        )
        print(
            "evaluated",
            os.path.basename(checkpoint_path),
            "score",
            checkpoint_results[-1]["score"],
        )

    ranking = sorted(
        checkpoint_results,
        key=lambda item: float("inf") if item["score"] is None else item["score"],
    )
    result_with_meta = {
        "backend": "Isaac Lab / PhysX 5",
        "checkpoint": checkpoint_results[0]["checkpoint"] if len(checkpoint_results) == 1 else None,
        "num_envs": args_cli.num_envs,
        "action_delay_policy_steps": args_cli.action_delay,
        "dynamics": {
            "actuator": args_cli.actuator,
            "kp": args_cli.kp,
            "kd": args_cli.kd,
            "action_scale": args_cli.action_scale,
            "effort_limit": args_cli.effort_limit,
            "armature": args_cli.armature,
            "joint_friction": args_cli.joint_friction,
            "solver_position_iterations": args_cli.solver_position_iterations,
            "solver_velocity_iterations": args_cli.solver_velocity_iterations,
            "solver_type": args_cli.solver_type,
            "contact_offset": args_cli.contact_offset,
            "rest_offset": args_cli.rest_offset,
            "replace_cylinders_with_capsules": args_cli.replace_cylinders_with_capsules,
            "make_instanceable": not args_cli.non_instanceable,
            "external_forces_every_iteration": args_cli.external_forces_every_iteration,
        },
        "policy_command_gains": {
            "positive_vx": args_cli.positive_vx_gain,
            "negative_vx": args_cli.negative_vx_gain,
            "vy": args_cli.vy_gain,
            "yaw": args_cli.yaw_gain,
        },
        "segments": checkpoint_results[0]["segments"] if len(checkpoint_results) == 1 else None,
        "checkpoint_results": checkpoint_results if len(checkpoint_results) > 1 else None,
        "ranking": [
            {"checkpoint": item["checkpoint"], "score": item["score"]}
            for item in ranking
        ] if len(checkpoint_results) > 1 else None,
    }
    print(json.dumps(result_with_meta, indent=2, sort_keys=True))

    if args_cli.out is not None:
        result_path = os.path.abspath(args_cli.out)
    else:
        result_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"sim2sim_result_vx{args_cli.vx:.2f}.json",
        )
    result_dir = os.path.dirname(result_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as stream:
        json.dump(result_with_meta, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("wrote", result_path)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
