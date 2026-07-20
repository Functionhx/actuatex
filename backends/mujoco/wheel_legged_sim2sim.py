#!/usr/bin/env python3
"""Evaluate the Isaac Sim 6 serial wheel-legged policy in MuJoCo.

The benchmark preserves the source policy's 28-D observation, six-action
ordering, 50 Hz action hold and mixed leg-position/wheel-velocity PD law.  It
can apply the same 0--20 ms actuator delay used during robust source training,
records every control step, and optionally renders an H.264 video.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from wheel_legged_contract import (
    ACTION_DIM,
    DECIMATION,
    DEFAULT_JOINT_POSITION,
    INITIAL_BASE_POSITION,
    LEG_DAMPING,
    LEG_STIFFNESS,
    POLICY_DT,
    POLICY_JOINT_NAMES,
    SIM_DT,
    WHEEL_DAMPING,
    action_to_targets,
    build_observation,
    compute_mixed_pd_torque,
    projected_gravity,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    REPO_ROOT / "robots" / "wheel_legged" / "mjcf" / "actuatex_serial_wheel_legged.xml"
)
DEFAULT_ACTOR = (
    REPO_ROOT
    / "artifacts"
    / "isaac_sim_6"
    / "checkpoints"
    / "serial_wheel_legged_robust_sim6.jit.pt"
)
DEFAULT_REFERENCE = (
    REPO_ROOT
    / "artifacts"
    / "isaac_sim_6"
    / "evaluation"
    / "wheel_legged_robust199_clean_nodelay_seed131.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "mujoco"
    / "sim2sim"
    / "wheel_legged_isaacsim6_to_mujoco"
    / "summary.json"
)

BENCHMARK_SEGMENTS = (
    ("stand", 2.0, (0.0, 0.0, 0.0)),
    ("forward_0p5", 4.0, (0.5, 0.0, 0.0)),
    ("forward_1p0", 4.0, (1.0, 0.0, 0.0)),
    ("backward_0p5", 4.0, (-0.5, 0.0, 0.0)),
    ("yaw_0p8", 4.0, (0.0, 0.0, 0.8)),
    ("arc_0p7_0p6", 4.0, (0.7, 0.0, 0.6)),
)

SHOWCASE_SEGMENTS = (
    ("stand", 1.5, (0.0, 0.0, 0.0)),
    ("forward_0p5", 2.0, (0.5, 0.0, 0.0)),
    ("backward_0p5", 2.0, (-0.5, 0.0, 0.0)),
    ("forward_1p0", 1.5, (1.0, 0.0, 0.0)),
    ("backward_1p0", 1.5, (-1.0, 0.0, 0.0)),
    ("yaw_left_0p8", 2.0, (0.0, 0.0, 0.8)),
    ("yaw_right_0p8", 2.0, (0.0, 0.0, -0.8)),
    ("arc_left", 2.5, (0.7, 0.0, 0.6)),
    ("arc_reverse", 2.5, (-0.7, 0.0, -0.6)),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    value = mujoco.mj_id2name(model, object_type, object_id)
    if value is None:
        raise ValueError(f"unnamed {object_type} id {object_id}")
    return value


def require_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, value: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, value)
    if object_id < 0:
        raise ValueError(f"MuJoCo model is missing {object_type}: {value}")
    return object_id


class TorchScriptActor:
    """Deterministic mean-action inference for the exported 28-to-6 actor."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.module = torch.jit.load(str(path), map_location="cpu").eval()
        probe = self(np.zeros(28, dtype=np.float32))
        if probe.shape != (ACTION_DIM,):
            raise ValueError(
                f"actor output shape is {probe.shape}, expected ({ACTION_DIM},)"
            )

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(observation, dtype=np.float32)).unsqueeze(
            0
        )
        with torch.inference_mode():
            action = self.module(tensor).squeeze(0).cpu().numpy()
        if not np.isfinite(action).all():
            raise RuntimeError("actor produced a non-finite action")
        return action.astype(np.float64, copy=False)


class TargetDelay:
    """Physics-tick FIFO matching DelayedPDActuatorCfg delay semantics."""

    def __init__(self, delay_ticks: int) -> None:
        if delay_ticks < 0:
            raise ValueError("delay_ticks cannot be negative")
        self.delay_ticks = delay_ticks
        self._queue: deque[np.ndarray] = deque(maxlen=delay_ticks + 1)
        self.reset()

    @staticmethod
    def _default_target() -> np.ndarray:
        return np.concatenate((DEFAULT_JOINT_POSITION[:4], np.zeros(2)))

    def reset(self) -> None:
        initial = self._default_target()
        self._queue.clear()
        for _ in range(self.delay_ticks + 1):
            self._queue.append(initial.copy())

    def __call__(self, current_target: np.ndarray) -> np.ndarray:
        self._queue.append(np.asarray(current_target, dtype=np.float64).copy())
        return self._queue[0].copy()


class H264Writer:
    """Stream RGB frames to system ffmpeg without extra Python codecs."""

    def __init__(self, path: Path, *, width: int, height: int, fps: int) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("--video requires ffmpeg on PATH")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.process = subprocess.Popen(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def append(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self.process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")


def build_indices(model: mujoco.MjModel) -> dict[str, Any]:
    joint_ids = np.array(
        [
            require_id(model, mujoco.mjtObj.mjOBJ_JOINT, value)
            for value in POLICY_JOINT_NAMES
        ],
        dtype=np.int32,
    )
    qpos_addresses = model.jnt_qposadr[joint_ids].copy()
    dof_addresses = model.jnt_dofadr[joint_ids].copy()
    actuator_joint_names = tuple(
        name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            int(model.actuator_trnid[actuator_id, 0]),
        )
        for actuator_id in range(model.nu)
    )
    if actuator_joint_names != POLICY_JOINT_NAMES:
        raise ValueError(
            "actuator order does not match the policy contract: "
            f"{actuator_joint_names} != {POLICY_JOINT_NAMES}"
        )
    return {
        "joint_ids": joint_ids,
        "qpos_addresses": qpos_addresses,
        "dof_addresses": dof_addresses,
        "base_body_id": require_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link"),
        "base_geom_id": require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "base_collision"),
        "floor_geom_id": require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"),
    }


def audit_model(model: mujoco.MjModel, indices: dict[str, Any]) -> dict[str, Any]:
    joint_ids = indices["joint_ids"]
    dof_addresses = indices["dof_addresses"]
    body_names = tuple(
        name(model, mujoco.mjtObj.mjOBJ_BODY, index) for index in range(1, model.nbody)
    )
    body_masses = {
        body_name: float(model.body_mass[index + 1])
        for index, body_name in enumerate(body_names)
    }
    return {
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "body_count_excluding_world": model.nbody - 1,
        "body_mass_kg": body_masses,
        "total_mass_kg": float(sum(body_masses.values())),
        "policy_joint_order": list(POLICY_JOINT_NAMES),
        "joint_ranges_rad": {
            POLICY_JOINT_NAMES[index]: model.jnt_range[joint_id].tolist()
            for index, joint_id in enumerate(joint_ids[:4])
        },
        "joint_armature": model.dof_armature[dof_addresses].tolist(),
        "joint_frictionloss": model.dof_frictionloss[dof_addresses].tolist(),
        "actuator_ctrlrange": model.actuator_ctrlrange.tolist(),
        "timestep_s": float(model.opt.timestep),
        "integrator": int(model.opt.integrator),
    }


def apply_parameter_overrides(
    model: mujoco.MjModel, indices: dict[str, Any], args: argparse.Namespace
) -> None:
    collision_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]
    ]
    model.geom_friction[collision_geoms, 0] = args.friction
    model.geom_condim[collision_geoms] = args.contact_dim
    model.geom_solref[collision_geoms, 0] = args.contact_time_constant

    dof_addresses = indices["dof_addresses"]
    model.dof_armature[dof_addresses[:4]] = args.leg_armature
    model.dof_armature[dof_addresses[4:]] = args.wheel_armature
    model.dof_frictionloss[dof_addresses[:4]] = args.leg_frictionloss

    base_body_id = indices["base_body_id"]
    model.body_mass[base_body_id] *= args.base_mass_scale
    model.body_inertia[base_body_id] *= args.base_mass_scale


def reset_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    indices: dict[str, Any],
    target_delay: TargetDelay,
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = INITIAL_BASE_POSITION
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[indices["qpos_addresses"]] = DEFAULT_JOINT_POSITION
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    target_delay.reset()
    mujoco.mj_forward(model, data)


def body_kinematics(
    model: mujoco.MjModel, data: mujoco.MjData, indices: dict[str, Any]
) -> dict[str, np.ndarray | float]:
    base_body_id = indices["base_body_id"]
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        base_body_id,
        velocity,
        1,
    )
    quaternion_wxyz = data.xquat[base_body_id].copy()
    gravity_body = projected_gravity(quaternion_wxyz)
    upright_error = math.acos(float(np.clip(-gravity_body[2], -1.0, 1.0)))
    return {
        "base_position": data.xpos[base_body_id].copy(),
        "quaternion_wxyz": quaternion_wxyz,
        "base_angular_velocity_body": velocity[:3].copy(),
        "base_linear_velocity_body": velocity[3:].copy(),
        "gravity_body": gravity_body,
        "upright_error_rad": upright_error,
        "joint_position": data.qpos[indices["qpos_addresses"]].copy(),
        "joint_velocity": data.qvel[indices["dof_addresses"]].copy(),
    }


def base_contact_reason(data: mujoco.MjData, indices: dict[str, Any]) -> bool:
    base_geom_id = indices["base_geom_id"]
    floor_geom_id = indices["floor_geom_id"]
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        if {int(contact.geom1), int(contact.geom2)} == {base_geom_id, floor_geom_id}:
            return True
    return False


def fall_reasons(
    data: mujoco.MjData,
    indices: dict[str, Any],
    state: dict[str, np.ndarray | float],
) -> list[str]:
    reasons: list[str] = []
    position = np.asarray(state["base_position"])
    if float(position[2]) < 0.25:
        reasons.append("base_height_below_0p25")
    if float(state["upright_error_rad"]) > 0.90:
        reasons.append("upright_error_above_0p90")
    if base_contact_reason(data, indices):
        reasons.append("base_contact")
    return reasons


def slew_command(
    current: np.ndarray,
    target: np.ndarray,
    *,
    linear_rate: float,
    yaw_rate: float,
) -> np.ndarray:
    """Apply a component-wise deployment command acceleration limit."""

    rate = np.array((linear_rate, linear_rate, yaw_rate), dtype=np.float64)
    maximum_delta = rate * POLICY_DT
    return current + np.clip(target - current, -maximum_delta, maximum_delta)


def select_reference(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "results" not in payload:
        return payload
    candidates = payload["results"]
    for candidate in candidates:
        if "model_199.pt" in candidate.get("checkpoint", ""):
            return candidate
    return candidates[0] if candidates else None


def compare_to_reference(
    result: dict[str, Any],
    reference: dict[str, Any] | None,
    reference_path: Path | None,
) -> dict[str, Any] | None:
    if reference is None:
        return None
    segment_delta: dict[str, dict[str, float]] = {}
    for segment_name, segment in result["segments"].items():
        source = reference.get("segments", {}).get(segment_name)
        if source is None:
            continue
        segment_delta[segment_name] = {
            metric: float(segment[metric] - source[metric])
            for metric in ("vx_rmse", "vy_rmse", "yaw_rmse")
        }
    return {
        "reference_path": str(reference_path) if reference_path else None,
        "reference_backend": reference.get("backend"),
        "reference_falls_total": reference.get("falls_total"),
        "reference_mean_primary_axis_rmse": reference.get("mean_primary_axis_rmse"),
        "target_minus_reference_primary_axis_rmse": (
            result["mean_primary_axis_rmse"] - reference["mean_primary_axis_rmse"]
        ),
        "segment_rmse_delta": segment_delta,
        "note": "The reference uses 256 randomized-delay PhysX environments; MuJoCo is one deterministic rollout.",
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    model_path = args.model.expanduser().resolve()
    actor_path = args.actor.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not actor_path.is_file():
        raise FileNotFoundError(actor_path)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    indices = build_indices(model)
    apply_parameter_overrides(model, indices, args)
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)

    delay_ticks_float = args.delay_ms / (SIM_DT * 1000.0)
    delay_ticks = round(delay_ticks_float)
    if not math.isclose(delay_ticks_float, delay_ticks, abs_tol=1.0e-9):
        raise ValueError(f"--delay-ms must be a multiple of {SIM_DT * 1000:.1f} ms")

    actor = TorchScriptActor(actor_path)
    target_delay = TargetDelay(delay_ticks)
    reset_state(model, data, indices, target_delay)
    previous_action = np.zeros(ACTION_DIM, dtype=np.float64)
    effective_command = np.zeros(3, dtype=np.float64)

    renderer = None
    camera = None
    video_writer = None
    if args.video is not None:
        renderer = mujoco.Renderer(
            model, height=args.video_height, width=args.video_width
        )
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = args.camera_distance
        camera.azimuth = args.camera_azimuth
        camera.elevation = args.camera_elevation
        video_writer = H264Writer(
            args.video.expanduser().resolve(),
            width=args.video_width,
            height=args.video_height,
            fps=round(1.0 / POLICY_DT),
        )

    segments = BENCHMARK_SEGMENTS if args.sequence == "benchmark" else SHOWCASE_SEGMENTS
    rows: list[dict[str, Any]] = []
    segment_results: dict[str, dict[str, Any]] = {}
    all_fall_reasons: Counter[str] = Counter()
    any_fall = False
    global_step = 0

    try:
        for segment_name, duration, command_values in segments:
            requested_command = np.asarray(command_values, dtype=np.float64)
            if args.reset_each_segment:
                reset_state(model, data, indices, target_delay)
                previous_action[:] = 0.0
                effective_command[:] = 0.0
            step_count = round(duration / POLICY_DT)
            settle_steps = round(min(1.0, duration / 3.0) / POLICY_DT)
            squared_error = np.zeros(3, dtype=np.float64)
            upright_error_sum = 0.0
            action_rms_sum = 0.0
            torque_rms_sum = 0.0
            sample_count = 0
            falls = 0
            first_fall_time_s: float | None = None
            minimum_height = math.inf
            max_abs_pitch_rate = 0.0
            segment_reason_counts: Counter[str] = Counter()

            for segment_step in range(step_count):
                effective_command = slew_command(
                    effective_command,
                    requested_command,
                    linear_rate=args.linear_command_slew,
                    yaw_rate=args.yaw_command_slew,
                )
                state = body_kinematics(model, data, indices)
                observation = build_observation(
                    np.asarray(state["base_linear_velocity_body"]),
                    np.asarray(state["base_angular_velocity_body"]),
                    np.asarray(state["gravity_body"]),
                    effective_command,
                    np.asarray(state["joint_position"]),
                    np.asarray(state["joint_velocity"]),
                    previous_action,
                )
                action = actor(observation)
                leg_target, wheel_target = action_to_targets(action)
                current_target = np.concatenate((leg_target, wheel_target))
                torque = np.zeros(ACTION_DIM, dtype=np.float64)

                for _ in range(DECIMATION):
                    delayed_target = target_delay(current_target)
                    joint_position = data.qpos[indices["qpos_addresses"]]
                    joint_velocity = data.qvel[indices["dof_addresses"]]
                    torque = compute_mixed_pd_torque(
                        joint_position,
                        joint_velocity,
                        delayed_target[:4],
                        delayed_target[4:],
                        leg_stiffness=args.leg_kp,
                        leg_damping=args.leg_kd,
                        wheel_damping=args.wheel_kd,
                    )
                    data.ctrl[:] = torque
                    mujoco.mj_step(model, data)
                previous_action = action.copy()

                state = body_kinematics(model, data, indices)
                position = np.asarray(state["base_position"])
                minimum_height = min(minimum_height, float(position[2]))
                angular_velocity = np.asarray(state["base_angular_velocity_body"])
                max_abs_pitch_rate = max(
                    max_abs_pitch_rate, abs(float(angular_velocity[1]))
                )
                reasons = fall_reasons(data, indices, state)
                if reasons:
                    falls += 1
                    any_fall = True
                    segment_reason_counts.update(reasons)
                    all_fall_reasons.update(reasons)
                    if first_fall_time_s is None:
                        first_fall_time_s = segment_step * POLICY_DT
                    reset_state(model, data, indices, target_delay)
                    previous_action[:] = 0.0
                    state = body_kinematics(model, data, indices)

                velocity = np.array(
                    [
                        np.asarray(state["base_linear_velocity_body"])[0],
                        np.asarray(state["base_linear_velocity_body"])[1],
                        np.asarray(state["base_angular_velocity_body"])[2],
                    ]
                )
                if segment_step >= settle_steps:
                    squared_error += np.square(velocity - requested_command)
                    upright_error_sum += float(state["upright_error_rad"])
                    action_rms_sum += float(np.sqrt(np.mean(np.square(action))))
                    torque_rms_sum += float(np.sqrt(np.mean(np.square(torque))))
                    sample_count += 1

                rows.append(
                    {
                        "time_s": global_step * POLICY_DT,
                        "segment": segment_name,
                        "command_vx": requested_command[0],
                        "command_vy": requested_command[1],
                        "command_yaw": requested_command[2],
                        "effective_command_vx": effective_command[0],
                        "effective_command_vy": effective_command[1],
                        "effective_command_yaw": effective_command[2],
                        "actual_vx": velocity[0],
                        "actual_vy": velocity[1],
                        "actual_yaw": velocity[2],
                        "base_height_m": np.asarray(state["base_position"])[2],
                        "upright_error_rad": state["upright_error_rad"],
                        "action_rms": float(np.sqrt(np.mean(np.square(action)))),
                        "torque_rms": float(np.sqrt(np.mean(np.square(torque)))),
                        "fall": bool(reasons),
                        "fall_reasons": "|".join(reasons),
                    }
                )
                global_step += 1

                if (
                    renderer is not None
                    and camera is not None
                    and video_writer is not None
                ):
                    camera.lookat[:] = np.asarray(state["base_position"]) + (
                        0.15,
                        0.0,
                        0.0,
                    )
                    renderer.update_scene(data, camera=camera)
                    video_writer.append(renderer.render())

            rmse = np.sqrt(squared_error / max(1, sample_count))
            segment_results[segment_name] = {
                "command": list(command_values),
                "vx_rmse": float(rmse[0]),
                "vy_rmse": float(rmse[1]),
                "yaw_rmse": float(rmse[2]),
                "mean_upright_error_rad": upright_error_sum / max(1, sample_count),
                "mean_action_rms": action_rms_sum / max(1, sample_count),
                "mean_torque_rms_nm": torque_rms_sum / max(1, sample_count),
                "minimum_base_height_m": minimum_height,
                "max_abs_pitch_rate_radps": max_abs_pitch_rate,
                "falls": falls,
                "first_fall_time_s": first_fall_time_s,
                "fall_reasons": dict(sorted(segment_reason_counts.items())),
            }
    finally:
        if video_writer is not None:
            video_writer.close()
        if renderer is not None:
            renderer.close()

    primary_errors: list[float] = []
    for segment in segment_results.values():
        command = segment["command"]
        if command[0] != 0.0:
            primary_errors.append(segment["vx_rmse"])
        if command[2] != 0.0:
            primary_errors.append(segment["yaw_rmse"])
        if command[0] == 0.0 and command[2] == 0.0:
            primary_errors.append(segment["vx_rmse"])

    output = {
        "schema_version": 1,
        "backend": f"MuJoCo {mujoco.__version__}",
        "source_backend": "Isaac Sim 6.0.1 GA / Isaac Lab 3.0.0-beta2.patch1 / PhysX 5",
        "model": {
            "path": str(model_path),
            "sha256": sha256(model_path),
            "audit": audit_model(model, indices),
        },
        "policy": {
            "path": str(actor_path),
            "sha256": sha256(actor_path),
            "observation_dim": 28,
            "action_dim": ACTION_DIM,
        },
        "control": {
            "physics_dt_s": SIM_DT,
            "policy_dt_s": POLICY_DT,
            "decimation": DECIMATION,
            "delay_ms": args.delay_ms,
            "delay_ticks": delay_ticks,
            "linear_command_slew_per_s2": args.linear_command_slew,
            "yaw_command_slew_per_s2": args.yaw_command_slew,
            "leg_kp": args.leg_kp,
            "leg_kd": args.leg_kd,
            "wheel_kd": args.wheel_kd,
        },
        "parameters": {
            "friction": args.friction,
            "contact_dim": args.contact_dim,
            "contact_time_constant_s": args.contact_time_constant,
            "leg_armature": args.leg_armature,
            "wheel_armature": args.wheel_armature,
            "leg_frictionloss": args.leg_frictionloss,
            "base_mass_scale": args.base_mass_scale,
        },
        "sequence": args.sequence,
        "reset_each_segment": args.reset_each_segment,
        "segments": segment_results,
        "falls_total": sum(segment["falls"] for segment in segment_results.values()),
        "clean_rollout": not any_fall,
        "fall_reasons": dict(sorted(all_fall_reasons.items())),
        "mean_primary_axis_rmse": float(np.mean(primary_errors)),
        "tracking_csv": str(args.tracking.expanduser().resolve()),
        "video": str(args.video.expanduser().resolve()) if args.video else None,
    }
    reference_path = args.reference.expanduser().resolve() if args.reference else None
    output["comparison_to_isaac_sim"] = compare_to_reference(
        output, select_reference(reference_path), reference_path
    )
    return output, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--actor", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tracking", type=Path)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--sequence", choices=("benchmark", "showcase"), default="benchmark"
    )
    parser.add_argument("--delay-ms", type=float, default=0.0)
    parser.add_argument(
        "--linear-command-slew",
        type=float,
        default=math.inf,
        help="Optional vx/vy acceleration limit; default preserves raw command steps.",
    )
    parser.add_argument(
        "--yaw-command-slew",
        type=float,
        default=math.inf,
        help="Optional yaw acceleration limit; default preserves raw command steps.",
    )
    parser.add_argument(
        "--reset-each-segment",
        action="store_true",
        help="Diagnostic protocol: start every command from the nominal standing state.",
    )
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument(
        "--contact-dim",
        type=int,
        choices=(1, 3, 4, 6),
        default=6,
        help="MuJoCo contact dimensions; 3 is the closest Coulomb-only PhysX analogue.",
    )
    parser.add_argument("--contact-time-constant", type=float, default=0.02)
    parser.add_argument("--leg-armature", type=float, default=0.01)
    parser.add_argument("--wheel-armature", type=float, default=0.02)
    parser.add_argument("--leg-frictionloss", type=float, default=0.02)
    parser.add_argument("--base-mass-scale", type=float, default=1.0)
    parser.add_argument("--leg-kp", type=float, default=LEG_STIFFNESS)
    parser.add_argument("--leg-kd", type=float, default=LEG_DAMPING)
    parser.add_argument("--wheel-kd", type=float, default=WHEEL_DAMPING)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=2.6)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-16.0)
    args = parser.parse_args()
    args.out = args.out.expanduser().resolve()
    if args.tracking is None:
        args.tracking = args.out.with_name("tracking.csv")
    if args.delay_ms < 0.0:
        parser.error("--delay-ms cannot be negative")
    for argument in (
        "friction",
        "contact_time_constant",
        "leg_armature",
        "wheel_armature",
        "base_mass_scale",
        "leg_kp",
        "leg_kd",
        "wheel_kd",
        "linear_command_slew",
        "yaw_command_slew",
    ):
        if getattr(args, argument) <= 0.0:
            parser.error(f"--{argument.replace('_', '-')} must be positive")
    if args.leg_frictionloss < 0.0:
        parser.error("--leg-frictionloss cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    result, rows = evaluate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.tracking.parent.mkdir(parents=True, exist_ok=True)
    with args.tracking.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "tracking": str(args.tracking),
                "video": result["video"],
                "falls_total": result["falls_total"],
                "mean_primary_axis_rmse": result["mean_primary_axis_rmse"],
                "clean_rollout": result["clean_rollout"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
