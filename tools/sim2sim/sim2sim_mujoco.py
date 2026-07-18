"""Sim2Sim backend A: replay the trained TinyMal policy in MuJoCo.

Cross-simulator check of the Isaac-Gym-trained flat policy. MuJoCo is a
completely different physics engine (not PhysX), so this is the strongest
sim2sim test. Loads the TinyMal URDF directly (it embeds a <mujoco><compiler
meshdir="./meshes"/></mujoco> block), applies the SAME hand-computed PD torque
as Isaac Gym (tau = Kp*(q0+0.25*a - q) - Kd*dq, clipped to +-12 N*m) via
qfrc_applied, at 200 Hz physics / 50 Hz control (decimation=4).

Run:  conda run -n unitree-rl python sim2sim_mujoco.py
"""

import csv
import json
import os
from collections import defaultdict

import numpy as np
import mujoco

from observation_builder import (
    build_obs, project_gravity, quat_rotate_inverse, dof_permutation,
    DEFAULT_DOF_POS, DEFAULT_ANGLES, COMMANDS_SCALE,
    KP, KD, ACTION_SCALE, TORQUE_LIMIT, DECIMATION, SIM_DT, POLICY_DT, INIT_POS,
)
from policy_numpy import NumpyPolicy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_ROOT = os.environ.get(
    "ACTUATEX_ARTIFACTS", os.path.join(REPO_ROOT, "artifacts")
)
DEFAULT_URDF = os.environ.get(
    "TINYMAL_SIM2SIM_URDF",
    os.path.join(REPO_ROOT, "robots", "tinymal", "urdf", "tinymal.urdf"),
)
DEFAULT_CHECKPOINT = os.environ.get(
    "TINYMAL_SIM2SIM_CHECKPOINT",
    os.path.join(ARTIFACTS_ROOT, "checkpoints", "isaac_gym", "model.pt"),
)
OUTPUT_DIR = os.environ.get(
    "TINYMAL_SIM2SIM_OUT", os.path.join(ROOT, "evaluation/sim2sim_mujoco"))

# Same command segments as evaluate_tinymal.py (apples-to-apples comparison).
SEGMENTS = (
    ("stand", 2.0, (0.0, 0.0, 0.0)),
    ("forward_0p3", 3.0, (0.3, 0.0, 0.0)),
    ("forward_0p6", 3.0, (0.6, 0.0, 0.0)),
    ("backward_0p3", 3.0, (-0.3, 0.0, 0.0)),
    ("lateral_0p2", 3.0, (0.0, 0.2, 0.0)),
    ("yaw_0p5", 3.0, (0.0, 0.0, 0.5)),
)


def _euler_roll_pitch(q_xyzw):
    """Roll/pitch (rad) from a quaternion for fall-thresholding only."""
    x, y, z, w = q_xyzw
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def build_floating_model(urdf_path):
    """Load the TinyMal URDF in MuJoCo with a proper floating base.

    MuJoCo's URDF importer welds the massful base link into the world for this
    SolidWorks-exported URDF (the base body disappears, its geometry becomes
    world geoms, and there is no 6-DOF free joint -> nv=12). We compile once to
    get correct meshes/inertias/joints, save the MJCF, then restructure it:
    wrap the base geometry + the four leg chains in a <body name='base'> with a
    <freejoint/> and the base inertial (mass/inertia read from the URDF), and
    add a floor plane.
    """
    import tempfile

    txt = open(urdf_path, "r", encoding="utf-8").read()
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    meshes_dir = os.path.join(os.path.dirname(urdf_dir), "meshes")  # sibling of urdf/
    txt = txt.replace('meshdir="./meshes"', f'meshdir="{meshes_dir}"')
    compiled = mujoco.MjModel.from_xml_string(txt)
    tmp = tempfile.mktemp(suffix=".mjcf")
    mujoco.mj_saveLastXML(tmp, compiled)
    s = open(tmp, "r", encoding="utf-8").read()

    a = s.find("<worldbody>") + len("<worldbody>")
    b = s.rfind("</worldbody>")
    inner = s[a:b].strip()  # base geoms + 4 leg chains, currently under world

    base_inertial = (
        '<inertial pos="0.0034198 6.4226e-06 0.0033633" mass="2.2657" '
        # fullinertia order is ixx iyy izz ixy ixz iyz (MuJoCo), not URDF order
        'fullinertia="0.0011588 0.0028416 0.0032559 4.4374e-07 -6.9655e-07 -9.2423e-07"/>'
    )
    floor = ('<geom name="floor" type="plane" size="0 0 0.1" '
             'friction="1 0.005 0.0001" rgba="0.4 0.4 0.4 1"/>')
    body = (f'<body name="base" pos="0 0 0"><freejoint name="root"/>{base_inertial}\n'
            f'      {inner}\n    </body>')
    new_s = s[:a] + "\n      " + floor + "\n      " + body + "\n    " + s[b:]
    return mujoco.MjModel.from_xml_string(new_s)


def run():
    model = build_floating_model(DEFAULT_URDF)
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)

    # Collect the 12 hinge joints and their qpos/qvel addresses (robust to order).
    hinge_ids = [i for i in range(model.njnt)
                 if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                   for i in hinge_ids]
    qposadr = np.array([model.jnt_qposadr[i] for i in hinge_ids])
    dofadr = np.array([model.jnt_dofadr[i] for i in hinge_ids])
    perm = dof_permutation(joint_names)        # mujoco -> policy
    inv_perm = np.argsort(perm)                # policy -> mujoco
    default_mujoco = np.array([DEFAULT_ANGLES[n] for n in joint_names])

    # Base body (free joint, root link "base").
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_id < 0:
        base_id = 0  # root body fallback

    policy = NumpyPolicy(DEFAULT_CHECKPOINT)

    def reset_state():
        # Spawn at standing height with default joint angles; zero velocities.
        data.qpos[:3] = INIT_POS
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz identity
        data.qpos[qposadr] = default_mujoco
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(model, data)

    rows = []
    segment_samples = defaultdict(list)
    summary = {}
    global_step = 0

    for segment, duration, command in SEGMENTS:
        reset_state()  # fresh spawn each segment (Isaac-Gym env auto-resets)
        last_action = np.zeros(12)
        steps = int(round(duration / POLICY_DT))
        settle = int(round(min(1.0, duration / 3.0) / POLICY_DT))
        fallen = False
        fall_step = None
        for local in range(steps):
            cmd = np.array(command, dtype=np.float64)

            base_pos = data.qpos[:3].copy()
            q_wxyz = data.qpos[3:7].copy()
            q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
            # Body-frame velocities, computed exactly as Isaac Gym does:
            # quat_rotate_inverse(base_quat, world_velocity). MuJoCo free-joint
            # qvel[0:3] is world linear, qvel[3:6] world angular.
            base_lin_vel = quat_rotate_inverse(q_xyzw, data.qvel[0:3])
            base_ang_vel = quat_rotate_inverse(q_xyzw, data.qvel[3:6])
            pg = project_gravity(q_xyzw)
            roll, pitch = _euler_roll_pitch(q_xyzw)

            qj = data.qpos[qposadr].copy()
            dqj = data.qvel[dofadr].copy()
            obs = build_obs(base_lin_vel, base_ang_vel, pg, cmd,
                            qj[perm], dqj[perm], last_action)
            action = policy(obs)  # policy order
            q_target_policy = DEFAULT_DOF_POS + ACTION_SCALE * action
            q_target_mujoco = q_target_policy[inv_perm]

            # Decimation substeps: recompute PD torque from current q,dq each step
            # (matches legged_robot.py step loop), action target held.
            tau_log = 0.0
            for _ in range(DECIMATION):
                q = data.qpos[qposadr]
                dq = data.qvel[dofadr]
                tau = KP * (q_target_mujoco - q) - KD * dq
                tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
                data.qfrc_applied[dofadr] = tau
                tau_log = tau
                mujoco.mj_step(model, data)
            last_action = action

            # Fall detection (Isaac-Gym-aligned: |roll|>0.8 or |pitch|>1.0).
            if not fallen and (abs(roll) > 0.8 or abs(pitch) > 1.0
                               or base_pos[2] < 0.10):
                fallen = True
                fall_step = local

            sample = {
                "time_s": global_step * POLICY_DT,
                "segment": segment,
                "cmd_vx": command[0], "cmd_vy": command[1], "cmd_yaw": command[2],
                "vx_mean": float(base_lin_vel[0]), "vx_std": 0.0,
                "vy_mean": float(base_lin_vel[1]), "vy_std": 0.0,
                "yaw_mean": float(base_ang_vel[2]), "yaw_std": 0.0,
                "base_z_mean": float(base_pos[2]), "base_z_std": 0.0,
                "abs_roll_mean": abs(float(roll)),
                "abs_pitch_mean": abs(float(pitch)),
                "action_rms": float(np.sqrt(np.mean(action ** 2))),
                "torque_rms": float(np.sqrt(np.mean(tau_log ** 2))),
                "fallen": int(fallen),
            }
            rows.append(sample)
            if local >= settle and not fallen:
                segment_samples[segment].append(sample)
            global_step += 1

        survival = 0.0 if fall_step is None else fall_step * POLICY_DT
        summary[segment] = _summarize(segment_samples.get(segment, []),
                                      command, survival, duration)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "tracking.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"tracking_csv={csv_path}")
    print(f"summary_json={summary_path}")


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
    run()
