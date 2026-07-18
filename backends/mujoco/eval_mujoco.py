"""Evaluate a trained TinyMal policy in MuJoCo (same 6 segments as Isaac Gym).

Loads the actor state_dict (keys actor.{0,2,4,6}.*), runs the 6-segment eval,
writes tracking.csv + summary.json. Uses MuJoCo position actuators (same model
as training).

Run:  conda run -n unitree-rl python eval_mujoco.py [--checkpoint PATH] [--out_dir DIR]
"""

import os
import sys
import csv
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_builder import build_floating_model, JOINT_NAMES
from observation_builder import (
    DEFAULT_DOF_POS, DEFAULT_ANGLES, COMMANDS_SCALE, OBS_SCALE_LIN_VEL,
    OBS_SCALE_ANG_VEL, OBS_SCALE_DOF_POS, OBS_SCALE_DOF_VEL, CLIP_OBS,
    ACTION_SCALE, DECIMATION, SIM_DT, INIT_POS,
    quat_rotate_inverse, project_gravity,
)
from actuatex_paths import ARTIFACTS_ROOT, ROBOT_URDF

URDF = str(ROBOT_URDF)

SEGMENTS = (
    ("stand", 2.0, (0.0, 0.0, 0.0)),
    ("forward_0p3", 3.0, (0.3, 0.0, 0.0)),
    ("forward_0p6", 3.0, (0.6, 0.0, 0.0)),
    ("backward_0p3", 3.0, (-0.3, 0.0, 0.0)),
    ("lateral_0p2", 3.0, (0.0, 0.2, 0.0)),
    ("yaw_0p5", 3.0, (0.0, 0.0, 0.5)),
)


def _euler_roll_pitch(q_xyzw):
    x, y, z, w = q_xyzw
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def _actor_state_dict(payload):
    """Normalize legacy RSL-RL, RSL-RL 2.x/Lab, and exported actor files."""
    if isinstance(payload, dict) and "actor_state_dict" in payload:
        return {
            key[len("mlp.") :]: value
            for key, value in payload["actor_state_dict"].items()
            if key.startswith("mlp.")
        }
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return {
            key[len("actor.") :]: value
            for key, value in payload["model_state_dict"].items()
            if key.startswith("actor.")
        }
    if isinstance(payload, dict) and "0.weight" in payload:
        return payload
    raise KeyError(
        "unsupported checkpoint: expected actor_state_dict, model_state_dict, "
        "or an exported Sequential actor state_dict"
    )


def load_actor(checkpoint_path):
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = _actor_state_dict(ck)
    layers = [
        (sd["0.weight"].numpy(), sd["0.bias"].numpy()),
        (sd["2.weight"].numpy(), sd["2.bias"].numpy()),
        (sd["4.weight"].numpy(), sd["4.bias"].numpy()),
        (sd["6.weight"].numpy(), sd["6.bias"].numpy()),
    ]
    return layers


def policy_forward(layers, obs48):
    x = np.asarray(obs48, dtype=np.float64)
    for i, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if i != len(layers) - 1:
            x = np.where(x > 0.0, x, np.expm1(x))  # ELU
    return x


def evaluate(
    checkpoint_path,
    out_dir,
    armature=0.01,
    kv=0.5,
    duration_override=None,
    yaw_input_gain=1.0,
):
    model = build_floating_model(URDF, armature=armature, kv=kv)
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)

    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    qposadr = np.array([model.jnt_qposadr[i] for i in joint_ids])
    dofadr = np.array([model.jnt_dofadr[i] for i in joint_ids])
    default_q = DEFAULT_DOF_POS.copy()

    layers = load_actor(checkpoint_path)

    rows = []
    segment_samples = defaultdict(list)
    summary = {}
    global_step = 0
    POLICY_DT = SIM_DT * DECIMATION

    for segment, duration, command in SEGMENTS:
        policy_command = command
        if segment == "yaw_0p5":
            policy_command = (command[0], command[1], command[2] * yaw_input_gain)
        if duration_override is not None:
            duration = duration_override
        data.qpos[:3] = INIT_POS
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        data.qpos[qposadr] = default_q
        data.qvel[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = default_q
        mujoco.mj_forward(model, data)

        last_action = np.zeros(12)
        steps = int(round(duration / POLICY_DT))
        settle = int(round(min(1.0, duration / 3.0) / POLICY_DT))
        fallen = False
        fall_step = None

        for local in range(steps):
            q_wxyz = data.qpos[3:7].copy()
            q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
            base_pos = data.qpos[:3].copy()
            base_lin_vel = quat_rotate_inverse(q_xyzw, data.qvel[0:3])
            base_ang_vel = quat_rotate_inverse(q_xyzw, data.qvel[3:6])
            pg = project_gravity(q_xyzw)
            roll, pitch = _euler_roll_pitch(q_xyzw)
            qj = data.qpos[qposadr].copy()
            dqj = data.qvel[dofadr].copy()

            obs = np.concatenate([
                base_lin_vel * OBS_SCALE_LIN_VEL,
                base_ang_vel * OBS_SCALE_ANG_VEL,
                pg,
                np.array(policy_command) * COMMANDS_SCALE,
                (qj - default_q) * OBS_SCALE_DOF_POS,
                dqj * OBS_SCALE_DOF_VEL,
                last_action,
            ])
            obs = np.clip(obs, -CLIP_OBS, CLIP_OBS)
            action = policy_forward(layers, obs)
            data.ctrl[:] = default_q + ACTION_SCALE * action

            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)

            last_action = action

            if not fallen and (abs(roll) > 0.8 or abs(pitch) > 1.0 or base_pos[2] < 0.10):
                fallen = True
                fall_step = local

            sample = {
                "time_s": global_step * POLICY_DT,
                "segment": segment,
                "cmd_vx": command[0], "cmd_vy": command[1], "cmd_yaw": command[2],
                "policy_cmd_yaw": policy_command[2],
                "vx_mean": float(base_lin_vel[0]),
                "vy_mean": float(base_lin_vel[1]),
                "yaw_mean": float(base_ang_vel[2]),
                "base_z_mean": float(base_pos[2]),
                "abs_roll_mean": abs(float(roll)),
                "abs_pitch_mean": abs(float(pitch)),
                "action_rms": float(np.sqrt(np.mean(action ** 2))),
                "torque_rms": float(np.sqrt(np.mean(data.actuator_force ** 2))),
                "fallen": int(fallen),
            }
            rows.append(sample)
            if local >= settle and not fallen:
                segment_samples[segment].append(sample)
            global_step += 1

        survival = duration if fall_step is None else fall_step * POLICY_DT
        summary[segment] = _summarize(segment_samples.get(segment, []),
                                      command, survival, duration)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "tracking.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"tracking_csv={csv_path}")
    print(f"summary_json={summary_path}")
    return summary


def _summarize(samples, command, survival, duration):
    if not samples:
        return {"survival_time_s": survival, "fallen": survival < duration}
    vx = np.array([s["vx_mean"] for s in samples])
    vy = np.array([s["vy_mean"] for s in samples])
    yaw = np.array([s["yaw_mean"] for s in samples])
    return {
        "vx_rmse": float(np.sqrt(np.mean((vx - command[0]) ** 2))),
        "vy_rmse": float(np.sqrt(np.mean((vy - command[1]) ** 2))),
        "yaw_rmse": float(np.sqrt(np.mean((yaw - command[2]) ** 2))),
        "base_height_mean": float(np.mean([s["base_z_mean"] for s in samples])),
        "abs_roll_mean": float(np.mean([s["abs_roll_mean"] for s in samples])),
        "abs_pitch_mean": float(np.mean([s["abs_pitch_mean"] for s in samples])),
        "action_rms": float(np.mean([s["action_rms"] for s in samples])),
        "torque_rms": float(np.mean([s["torque_rms"] for s in samples])),
        "survival_time_s": float(survival),
        "fallen": bool(survival < duration),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(ARTIFACTS_ROOT / "checkpoints" / "mujoco" / "model.pt"),
    )
    parser.add_argument(
        "--out_dir", default=str(ARTIFACTS_ROOT / "mujoco" / "evaluation")
    )
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--armature", type=float, default=0.01)
    parser.add_argument("--joint_damping", type=float, default=0.5)
    parser.add_argument("--yaw_input_gain", type=float, default=1.0)
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        args.out_dir,
        armature=args.armature,
        kv=args.joint_damping,
        duration_override=args.duration,
        yaw_input_gain=args.yaw_input_gain,
    )
