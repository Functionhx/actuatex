#!/usr/bin/env python3
"""Record a deterministic TinyMal MuJoCo staircase rollout as an H.264 MP4."""

import argparse
import os
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from eval_mujoco_tasks import POLICY_DT, PolicyRollout, staircase_xml


def _font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = os.path.join("/usr/share/fonts/truetype/dejavu", name)
    return ImageFont.truetype(path, size=size)


def _overlay(frame, time_s, command_speed, heading_gain, state, strict_passed):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    draw.rounded_rectangle((24, 22, width - 24, 128), radius=14,
                           fill=(8, 15, 28, 210), outline=(80, 180, 255, 230), width=2)
    draw.text((48, 36), "TinyMal  |  MuJoCo  |  5 x 20 mm stairs",
              font=_font(30, bold=True), fill=(245, 249, 255, 255))
    x, y, z = state["position"]
    telemetry = (
        f"t={time_s:4.1f} s    command={command_speed:.2f} m/s    "
        f"heading gain={heading_gain:.1f}    x={x:.2f} m    z={z:.2f} m"
    )
    draw.text((48, 82), telemetry, font=_font(21), fill=(180, 220, 255, 255))
    if strict_passed:
        label = "STRICT PASS: top reached, centered, no fall"
        box = draw.textbbox((0, 0), label, font=_font(27, bold=True))
        box_width = box[2] - box[0]
        draw.rounded_rectangle(
            (width - box_width - 72, height - 76, width - 28, height - 24),
            radius=12, fill=(20, 145, 78, 225),
        )
        draw.text((width - box_width - 50, height - 68), label,
                  font=_font(27, bold=True), fill=(255, 255, 255, 255))
    return np.asarray(image)


def record(args):
    os.environ.setdefault("MUJOCO_GL", "egl")
    step_height = 0.02
    step_width = 0.14
    num_steps = 5
    start_x = 0.55
    rollout = PolicyRollout(
        args.checkpoint,
        worldbody_extras=staircase_xml(
            step_height,
            step_width=step_width,
            num_steps=num_steps,
            start_x=start_x,
        ),
    )
    rollout.reset()

    # Presentation-only colors and lighting; physics parameters stay untouched.
    rollout.model.vis.headlight.ambient[:] = (0.28, 0.28, 0.28)
    rollout.model.vis.headlight.diffuse[:] = (0.68, 0.68, 0.68)
    rollout.model.vis.headlight.specular[:] = (0.18, 0.18, 0.18)
    for geom_id in range(rollout.model.ngeom):
        name = mujoco.mj_id2name(
            rollout.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        ) or ""
        if name == "floor":
            rollout.model.geom_rgba[geom_id] = (0.16, 0.19, 0.23, 1.0)
        elif name == "stair_top":
            rollout.model.geom_rgba[geom_id] = (0.10, 0.36, 0.62, 1.0)
        elif name.startswith("stair_"):
            rollout.model.geom_rgba[geom_id] = (0.93, 0.48, 0.12, 1.0)

    rollout.model.vis.global_.offwidth = args.width
    rollout.model.vis.global_.offheight = args.height
    renderer = mujoco.Renderer(rollout.model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = rollout.base_body_id
    camera.distance = 1.45
    camera.azimuth = 90.0
    camera.elevation = -17.0

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s:v", f"{args.width}x{args.height}",
            "-r", str(args.fps), "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", output,
        ],
        stdin=subprocess.PIPE,
    )

    warmup_steps = int(round(1.0 / POLICY_DT))
    total_steps = warmup_steps + int(round(args.duration / POLICY_DT))
    success_x = start_x + num_steps * step_width + 0.20
    success_z = num_steps * step_height + 0.10
    max_abs_y = 0.0
    strict_passed = False
    passed_at_step = None
    pass_frame = None
    final_state = None

    try:
        for step in range(total_steps):
            if step < warmup_steps:
                command = (0.0, 0.0, 0.0)
            else:
                heading_error = np.arctan2(
                    np.sin(-rollout.heading_yaw()),
                    np.cos(-rollout.heading_yaw()),
                )
                command = (
                    args.command_speed,
                    0.0,
                    float(np.clip(args.heading_gain * heading_error, -1.0, 1.0)),
                )
            state = rollout.policy_step(command)
            final_state = state
            x, y, z = state["position"]
            max_abs_y = max(max_abs_y, abs(float(y)))
            strict_passed = strict_passed or bool(
                x >= success_x
                and z >= success_z
                and max_abs_y <= args.centerline_tolerance
                and not state["fallen"]
            )
            if strict_passed and passed_at_step is None:
                passed_at_step = step
            renderer.update_scene(rollout.data, camera=camera)
            frame = renderer.render()
            frame = _overlay(
                frame,
                step * POLICY_DT,
                args.command_speed,
                args.heading_gain,
                state,
                strict_passed,
            )
            if strict_passed and pass_frame is None:
                pass_frame = frame.copy()
            ffmpeg.stdin.write(frame.tobytes())
            if state["fallen"]:
                break
            if (
                passed_at_step is not None
                and (step - passed_at_step) * POLICY_DT >= args.hold_after_pass
            ):
                break
    finally:
        renderer.close()
        if ffmpeg.stdin is not None:
            ffmpeg.stdin.close()
        return_code = ffmpeg.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")

    if pass_frame is not None and args.poster:
        poster = os.path.abspath(args.poster)
        os.makedirs(os.path.dirname(poster), exist_ok=True)
        Image.fromarray(pass_frame).save(poster)

    result = {
        "output": output,
        "strict_passed": strict_passed,
        "fallen": bool(final_state["fallen"]),
        "max_abs_lateral_offset_m": max_abs_y,
        "final_position_m": final_state["position"].tolist(),
    }
    print(result)
    if not strict_passed:
        raise SystemExit("rollout did not satisfy the strict staircase criterion")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--poster")
    parser.add_argument("--command_speed", type=float, default=0.5)
    parser.add_argument("--heading_gain", type=float, default=1.6)
    parser.add_argument("--centerline_tolerance", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--hold_after_pass", type=float, default=1.5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=50)
    record(parser.parse_args())


if __name__ == "__main__":
    main()
