#!/usr/bin/env python3
"""Record the optimized single-pole swing-up with TVLQR feedback."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from actuatex_paths import ARTIFACTS_ROOT, REPO_ROOT  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

from tasks.inverted_pendulum.contract import DECIMATION  # noqa: E402
from tasks.inverted_pendulum.trajectory_optimization import (  # noqa: E402
    TVLQRTrackingController,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=ARTIFACTS_ROOT
        / "inverted_pendulum"
        / "trajectories"
        / "single_pole_swingup_tvlqr.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_ROOT
        / "mujoco"
        / "videos"
        / "ActuateX_InvertedPendulum_Swingup_TVLQR_PPT.mp4",
    )
    parser.add_argument("--duration_s", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    return parser.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_overlay(
    frame: np.ndarray,
    *,
    elapsed_s: float,
    reference_duration_s: float,
    force_n: float,
    cart_position_m: float,
    pole_angle_rad: float,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(
        (28, 24, 692, 142),
        radius=18,
        fill=(10, 16, 26, 220),
        outline=(255, 111, 24, 230),
        width=2,
    )
    mode = (
        "OPTIMIZED TRAJECTORY + TVLQR"
        if elapsed_s < reference_duration_s
        else "TERMINAL LQR HOLD"
    )
    draw.text(
        (52, 42),
        "ActuateX  |  Nonlinear Cart-Pole Swing-up",
        font=font(26),
        fill=(245, 248, 252, 255),
    )
    draw.text(
        (52, 82),
        f"{mode}   t={elapsed_s:4.1f}s   u={force_n:+5.1f}N",
        font=font(20),
        fill=(255, 157, 78, 255),
    )
    draw.text(
        (52, 112),
        f"cart={cart_position_m:+.2f}m   pole={np.degrees(pole_angle_rad):+.1f}deg",
        font=font(18),
        fill=(190, 205, 222, 255),
    )
    return np.asarray(image)


def main() -> None:
    args = parse_args()
    if args.duration_s <= 0.0 or min(args.width, args.height, args.fps) <= 0:
        raise ValueError("duration, image size and fps must be positive")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to record the PPT video")
    if not args.trajectory.is_file():
        raise FileNotFoundError(
            f"missing {args.trajectory}; run evaluate_trajectory_control_suite.py first"
        )
    payload = np.load(args.trajectory)
    states = payload["states"]
    forces = payload["forces"]
    gains = payload["tvlqr_gains"]
    terminal_gain = payload["terminal_gain"]
    controller = TVLQRTrackingController(
        states,
        forces,
        gains,
        terminal_gain,
    )
    controller.reset(1)

    model_path = (
        REPO_ROOT / "robots" / "inverted_pendulum" / "mjcf" / "actuatex_cartpole_1.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = [0.03, np.pi + 0.025]
    data.qvel[:] = [0.0, -0.02]
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = [0.0, 0.0, 0.55]
    camera.distance = 5.7
    camera.azimuth = 90.0
    camera.elevation = -7.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{args.width}x{args.height}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    frame_count = round(args.duration_s * args.fps)
    reference_duration_s = forces.size / 60.0
    try:
        for frame_index in range(frame_count):
            state = np.array(
                [[data.qpos[0], data.qpos[1], data.qvel[0], data.qvel[1]]],
                dtype=np.float64,
            )
            action = controller.act(state)[0]
            force_n = 20.0 * action
            data.ctrl[0] = force_n
            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            frame = draw_overlay(
                frame,
                elapsed_s=frame_index / args.fps,
                reference_duration_s=reference_duration_s,
                force_n=force_n,
                cart_position_m=float(data.qpos[0]),
                pole_angle_rad=float(
                    np.arctan2(np.sin(data.qpos[1]), np.cos(data.qpos[1]))
                ),
            )
            if process.stdin is None:
                raise RuntimeError("ffmpeg stdin closed unexpectedly")
            process.stdin.write(frame.tobytes())
    finally:
        renderer.close()
        if process.stdin is not None:
            process.stdin.close()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
