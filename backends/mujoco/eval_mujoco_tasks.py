"""MuJoCo stair and sustained-force acceptance tests for one TinyMal policy."""

import argparse
import json
import os

import mujoco
import numpy as np

from eval_mujoco import _euler_roll_pitch, load_actor, policy_forward
from model_builder import JOINT_NAMES, build_floating_model
from observation_builder import (
    ACTION_SCALE,
    CLIP_OBS,
    COMMANDS_SCALE,
    DECIMATION,
    DEFAULT_DOF_POS,
    INIT_POS,
    OBS_SCALE_ANG_VEL,
    OBS_SCALE_DOF_POS,
    OBS_SCALE_DOF_VEL,
    OBS_SCALE_LIN_VEL,
    SIM_DT,
    project_gravity,
    quat_rotate_inverse,
)
from actuatex_paths import ROBOT_URDF

URDF = str(ROBOT_URDF)
POLICY_DT = SIM_DT * DECIMATION


def staircase_xml(step_height, step_width=0.14, num_steps=5,
                  start_x=0.55, total_width=3.10, top_length=1.40):
    geoms = []
    for step in range(num_steps):
        x0 = start_x + step * step_width
        x1 = x0 + step_width
        height = (step + 1) * step_height
        geoms.append(
            f'<geom name="stair_{step}" type="box" '
            f'pos="{(x0 + x1) / 2.0} 0 {height / 2.0}" '
            f'size="{step_width / 2.0} {total_width / 2.0} {height / 2.0}" '
            'friction="1 0.005 0.0001" rgba="0.35 0.35 0.35 1"/>'
        )
    platform_x0 = start_x + num_steps * step_width
    platform_x1 = platform_x0 + top_length
    top_height = num_steps * step_height
    geoms.append(
        '<geom name="stair_top" type="box" '
        f'pos="{(platform_x0 + platform_x1) / 2.0} 0 {top_height / 2.0}" '
        f'size="{top_length / 2.0} {total_width / 2.0} {top_height / 2.0}" '
        'friction="1 0.005 0.0001" rgba="0.35 0.35 0.35 1"/>'
    )
    return "\n    " + "\n    ".join(geoms)


class PolicyRollout:
    def __init__(self, checkpoint, worldbody_extras=None):
        self.model = build_floating_model(
            URDF, armature=0.01, kv=0.5, worldbody_extras=worldbody_extras
        )
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)
        self.layers = load_actor(checkpoint)
        joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]
        self.qposadr = np.asarray([self.model.jnt_qposadr[i] for i in joint_ids])
        self.dofadr = np.asarray([self.model.jnt_dofadr[i] for i in joint_ids])
        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        self.last_action = np.zeros(12)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = INIT_POS
        self.data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.qposadr] = DEFAULT_DOF_POS
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = DEFAULT_DOF_POS
        self.data.xfrc_applied[:] = 0.0
        self.last_action[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def policy_step(self, command, base_force=None):
        q_wxyz = self.data.qpos[3:7].copy()
        q_xyzw = q_wxyz[[1, 2, 3, 0]]
        base_lin_vel = quat_rotate_inverse(q_xyzw, self.data.qvel[:3])
        base_ang_vel = quat_rotate_inverse(q_xyzw, self.data.qvel[3:6])
        obs = np.concatenate(
            (
                base_lin_vel * OBS_SCALE_LIN_VEL,
                base_ang_vel * OBS_SCALE_ANG_VEL,
                project_gravity(q_xyzw),
                np.asarray(command) * COMMANDS_SCALE,
                (self.data.qpos[self.qposadr] - DEFAULT_DOF_POS)
                * OBS_SCALE_DOF_POS,
                self.data.qvel[self.dofadr] * OBS_SCALE_DOF_VEL,
                self.last_action,
            )
        )
        action = policy_forward(self.layers, np.clip(obs, -CLIP_OBS, CLIP_OBS))
        self.data.ctrl[:] = DEFAULT_DOF_POS + ACTION_SCALE * action
        for _ in range(DECIMATION):
            self.data.xfrc_applied[:] = 0.0
            if base_force is not None:
                self.data.xfrc_applied[self.base_body_id, :3] = base_force
            mujoco.mj_step(self.model, self.data)
        self.last_action[:] = action
        return self.state(base_lin_vel, base_ang_vel)

    def heading_yaw(self):
        """Return the base's world-frame yaw in radians."""
        w, x, y, z = self.data.qpos[3:7]
        return float(np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        ))

    def state(self, base_lin_vel=None, base_ang_vel=None):
        q_wxyz = self.data.qpos[3:7].copy()
        q_xyzw = q_wxyz[[1, 2, 3, 0]]
        if base_lin_vel is None:
            base_lin_vel = quat_rotate_inverse(q_xyzw, self.data.qvel[:3])
        if base_ang_vel is None:
            base_ang_vel = quat_rotate_inverse(q_xyzw, self.data.qvel[3:6])
        roll, pitch = _euler_roll_pitch(q_xyzw)
        return {
            "position": self.data.qpos[:3].copy(),
            "base_lin_vel": base_lin_vel.copy(),
            "base_ang_vel": base_ang_vel.copy(),
            "roll": float(roll),
            "pitch": float(pitch),
            "fallen": bool(
                abs(roll) > 0.8
                or abs(pitch) > 1.0
                or self.data.qpos[2] < 0.10
            ),
        }


def evaluate_stairs(
    checkpoint,
    step_height,
    command_speed=0.3,
    duration=12.0,
    heading_gain=0.5,
    centerline_tolerance=0.5,
):
    num_steps = 5
    step_width = 0.14
    start_x = 0.55
    rollout = PolicyRollout(
        checkpoint,
        worldbody_extras=staircase_xml(
            step_height, step_width=step_width, num_steps=num_steps, start_x=start_x
        ),
    )
    rollout.reset()
    success_x = start_x + num_steps * step_width + 0.20
    top_height = num_steps * step_height
    warmup_steps = int(round(1.0 / POLICY_DT))
    total_steps = warmup_steps + int(round(duration / POLICY_DT))
    max_x = -np.inf
    max_z = -np.inf
    max_abs_y = 0.0
    passed_at = None
    fallen = False
    for step in range(total_steps):
        if step < warmup_steps:
            command = (0.0, 0.0, 0.0)
        else:
            # Match TinyMalRobustMixed's world-heading controller.  A fixed
            # zero yaw-rate command only stops collision-induced rotation; it
            # cannot steer the robot back toward the staircase centerline.
            heading_error = np.arctan2(
                np.sin(-rollout.heading_yaw()),
                np.cos(-rollout.heading_yaw()),
            )
            command = (
                command_speed,
                0.0,
                float(np.clip(heading_gain * heading_error, -1.0, 1.0)),
            )
        state = rollout.policy_step(command)
        x, y, z = state["position"]
        max_x = max(max_x, float(x))
        max_z = max(max_z, float(z))
        max_abs_y = max(max_abs_y, abs(float(y)))
        fallen = fallen or state["fallen"]
        if (
            passed_at is None
            and x >= success_x
            and abs(y) <= 3.10 / 2.0
            and z >= top_height + 0.10
        ):
            passed_at = step * POLICY_DT
        if state["fallen"]:
            break
    collision_valid_pass = passed_at is not None and not fallen
    centered = max_abs_y <= centerline_tolerance
    return {
        "step_height_m": step_height,
        "num_steps": num_steps,
        "command_speed_mps": command_speed,
        "heading_gain": heading_gain,
        "centerline_tolerance_m": centerline_tolerance,
        "passed": collision_valid_pass,
        "centered": centered,
        "strict_passed": collision_valid_pass and centered,
        "time_to_pass_s": passed_at,
        "max_progress_x_m": max_x,
        "max_base_height_m": max_z,
        "max_abs_lateral_offset_m": max_abs_y,
        "fallen": fallen,
    }


def evaluate_pushes(checkpoint):
    rollout = PolicyRollout(checkpoint)
    cases = []
    for direction, vector in (
        ("+x", (1.0, 0.0, 0.0)),
        ("-x", (-1.0, 0.0, 0.0)),
        ("+y", (0.0, 1.0, 0.0)),
        ("-y", (0.0, -1.0, 0.0)),
    ):
        for magnitude in (20.0, 30.0, 40.0, 50.0):
            rollout.reset()
            push_start = int(round(2.0 / POLICY_DT))
            push_steps = int(round(0.2 / POLICY_DT))
            total_steps = int(round(5.0 / POLICY_DT))
            force = np.asarray(vector) * magnitude
            fallen = False
            recovered_at = None
            recovery_band_steps = 0
            recovery_hold_steps = int(round(0.2 / POLICY_DT))
            peak_roll = 0.0
            peak_pitch = 0.0
            for step in range(total_steps):
                active = push_start <= step < push_start + push_steps
                state = rollout.policy_step(
                    (0.3, 0.0, 0.0), base_force=force if active else None
                )
                peak_roll = max(peak_roll, abs(state["roll"]))
                peak_pitch = max(peak_pitch, abs(state["pitch"]))
                fallen = fallen or state["fallen"]
                if step >= push_start + push_steps and not fallen:
                    if abs(state["base_lin_vel"][0] - 0.3) < 0.15:
                        recovery_band_steps += 1
                    else:
                        recovery_band_steps = 0
                    if (
                        recovered_at is None
                        and recovery_band_steps >= recovery_hold_steps
                    ):
                        recovered_at = (
                            step
                            - push_start
                            - push_steps
                            - recovery_hold_steps
                            + 1
                        ) * POLICY_DT
                if state["fallen"]:
                    break
            cases.append(
                {
                    "direction": direction,
                    "magnitude_n": magnitude,
                    "duration_s": 0.2,
                    "fallen": fallen,
                    "recovered": recovered_at is not None and not fallen,
                    "recovery_time_s": recovered_at if not fallen else None,
                    "peak_abs_roll_rad": peak_roll,
                    "peak_abs_pitch_rad": peak_pitch,
                }
            )
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--step_height", type=float, default=0.02)
    parser.add_argument("--stair_speed", type=float, default=0.3)
    parser.add_argument("--stair_heading_gain", type=float, default=0.5)
    parser.add_argument("--centerline_tolerance", type=float, default=0.5)
    parser.add_argument("--skip_stairs", action="store_true")
    parser.add_argument("--skip_pushes", action="store_true")
    args = parser.parse_args()
    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
    }
    if not args.skip_stairs:
        result["stairs"] = evaluate_stairs(
            args.checkpoint,
            args.step_height,
            command_speed=args.stair_speed,
            heading_gain=args.stair_heading_gain,
            centerline_tolerance=args.centerline_tolerance,
        )
    if not args.skip_pushes:
        result["pushes"] = evaluate_pushes(args.checkpoint)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "tasks_summary.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("summary_json=" + path)


if __name__ == "__main__":
    main()
