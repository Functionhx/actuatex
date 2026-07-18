"""Evaluate TinyMal on physical stair meshes and optionally record MP4 video."""

import csv
import json
import os
import subprocess
from pathlib import Path

import isaacgym  # noqa: F401  # Must precede torch.
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from PIL import Image, ImageDraw, ImageFont

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403  # Registers tasks.
from legged_gym.utils import get_args, task_registry


def deterministic_reset(env):
    env_ids = torch.zeros(1, dtype=torch.int32, device=env.device)
    env.dof_pos[:] = env.default_dof_pos
    env.dof_vel[:] = 0.0
    env.root_states[0] = env.base_init_state
    env.root_states[0, :3] += env.env_origins[0]
    env.root_states[0, 7:13] = 0.0
    env.actions[:] = 0.0
    env.last_actions[:] = 0.0
    env.last_dof_vel[:] = 0.0
    env.episode_length_buf[:] = 0
    env.reset_buf[:] = 0
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids),
        1,
    )
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids),
        1,
    )
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.compute_observations()


def open_video_writer(path, width, height, fps):
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
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
        "18",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def get_frame(env, overlay):
    env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    image = env.gym.get_camera_image(
        env.sim,
        env.envs[0],
        env.recording_camera,
        gymapi.IMAGE_COLOR,
    )
    height = env.cfg.stairs.camera_height
    width = env.cfg.stairs.camera_width
    rgba = np.asarray(image).reshape(height, width, 4).astype(np.uint8)
    frame = Image.fromarray(rgba[:, :, :3], mode="RGB")
    draw = ImageDraw.Draw(frame, mode="RGBA")
    draw.rounded_rectangle((18, 16, 555, 135), radius=12, fill=(0, 0, 0, 165))
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23
        )
    except OSError:
        font = ImageFont.load_default()
    draw.multiline_text((34, 28), overlay, font=font, fill=(255, 255, 255, 255), spacing=6)
    return np.asarray(frame)


def evaluate(args):
    step_height = float(os.environ.get("TINYMAL_STEP_HEIGHT", "0.03"))
    record_video = os.environ.get("TINYMAL_RECORD_VIDEO", "0") == "1"
    command_speed = float(os.environ.get("TINYMAL_STAIR_SPEED", "0.30"))
    policy_command_speed = float(
        os.environ.get("TINYMAL_STAIR_POLICY_SPEED", str(command_speed))
    )
    test_duration = float(os.environ.get("TINYMAL_TEST_DURATION", "7.0"))
    policy_label = os.environ.get("TINYMAL_POLICY_LABEL", "flat_policy")
    policy_description = os.environ.get(
        "TINYMAL_POLICY_DESCRIPTION", "flat model_1500.pt"
    )

    task_name = args.task
    if task_name not in ("tinymal_stairs", "tinymal_robust_stairs"):
        raise ValueError(
            "stair evaluation requires --task=tinymal_stairs or "
            "--task=tinymal_robust_stairs"
        )
    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    # task_registry.make_env only applies --seed to train_cfg, while the
    # environment RNG is initialized from env_cfg.seed.  Propagate it here so
    # multi-seed actuator/delay/friction evaluation is not silently repeated
    # with the task's default seed.
    if args.seed is not None:
        env_cfg.seed = args.seed
        train_cfg.seed = args.seed
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = test_duration + 2.0
    # Keep --headless usable on servers without DISPLAY while retaining the
    # graphics context required by the camera sensor.
    env_cfg.env.enable_offscreen_rendering = record_video
    env_cfg.stairs.curriculum = False
    env_cfg.stairs.step_height = step_height
    env_cfg.stairs.record_video = record_video
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    if hasattr(env_cfg.domain_rand, "randomize_push_force"):
        env_cfg.domain_rand.randomize_push_force = False
    env_cfg.commands.heading_command = True

    env, _ = task_registry.make_env(
        name=task_name, args=args, env_cfg=env_cfg
    )
    direct_checkpoint = os.environ.get("TINYMAL_CHECKPOINT_PATH")
    train_cfg.runner.resume = direct_checkpoint is None
    runner, _ = task_registry.make_alg_runner(
        env=env, name=task_name, args=args, train_cfg=train_cfg
    )
    if direct_checkpoint is not None:
        direct_checkpoint = os.path.abspath(direct_checkpoint)
        checkpoint = torch.load(direct_checkpoint, map_location=env.device)
        runner.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded direct checkpoint: {direct_checkpoint}")
    policy = runner.get_inference_policy(device=env.device)
    deterministic_reset(env)

    output_dir = Path(
        os.environ.get(
            "TINYMAL_STAIR_OUT",
            str(Path(LEGGED_GYM_ROOT_DIR) / "evaluation" / "tinymal_stairs"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    millimeters = int(round(step_height * 1000))
    stem = f"{policy_label}_{millimeters:02d}mm"
    csv_path = output_dir / f"{stem}.csv"
    summary_path = output_dir / f"{stem}_summary.json"
    video_path = output_dir / f"{stem}.mp4"

    writer = None
    if record_video:
        writer = open_video_writer(
            video_path,
            env_cfg.stairs.camera_width,
            env_cfg.stairs.camera_height,
            int(round(1.0 / env.dt)),
        )

    warmup_steps = int(round(1.0 / env.dt))
    test_steps = int(round(test_duration / env.dt))
    rows = []
    reset_count = 0
    success_step = None
    left_stair_corridor = False
    start_x = env_cfg.stairs.start_x
    staircase_end = start_x + env_cfg.stairs.num_steps * env_cfg.stairs.step_width
    success_x = staircase_end + 0.20
    top_height = env_cfg.stairs.num_steps * step_height

    with torch.inference_mode():
        for step in range(warmup_steps + test_steps):
            speed = 0.0 if step < warmup_steps else command_speed
            env.commands[:, 0] = speed
            env.commands[:, 1] = 0.0
            env.commands[:, 3] = 0.0
            env.compute_observations()
            policy_observations = env.get_observations().clone()
            policy_observations[:, 9] = (
                policy_command_speed * env.obs_scales.lin_vel
            )
            actions = policy(policy_observations)
            _, _, _, dones, _ = env.step(actions)

            local_x = float((env.base_pos[0, 0] - env.env_origins[0, 0]).item())
            local_y = float((env.base_pos[0, 1] - env.env_origins[0, 1]).item())
            base_z = float(env.base_pos[0, 2].item())
            elapsed = step * env.dt
            reset = bool(dones[0].item())
            if abs(local_y) > env_cfg.stairs.total_width / 2.0:
                left_stair_corridor = True
            if reset:
                reset_count += 1
            if (
                success_step is None
                and local_x >= success_x
                and not left_stair_corridor
                and base_z >= top_height + 0.10
            ):
                success_step = step

            rows.append(
                {
                    "time_s": elapsed,
                    "command_vx": speed,
                    "local_x": local_x,
                    "local_y": local_y,
                    "base_z": base_z,
                    "roll_rad": float(env.rpy[0, 0].item()),
                    "pitch_rad": float(env.rpy[0, 1].item()),
                    "torque_rms": float(env.torques.square().mean().sqrt().item()),
                    "reset": int(reset),
                }
            )

            if writer is not None:
                state = "PASSED" if success_step is not None else ("FALL" if reset else "RUNNING")
                overlay = (
                    f"TinyMal stair test | {millimeters} mm x {env_cfg.stairs.num_steps} steps\n"
                    f"command {speed:.2f} m/s | x {local_x:.2f} m | z {base_z:.2f} m\n"
                    f"lateral y {local_y:.2f} m | status: {state}"
                )
                frame = get_frame(env, overlay)
                writer.stdin.write(frame.tobytes())

            if reset or (success_step is not None and step - success_step >= 25):
                break

    if writer is not None:
        writer.stdin.close()
        return_code = writer.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    max_x = max(row["local_x"] for row in rows)
    max_z = max(row["base_z"] for row in rows)
    max_abs_y = max(abs(row["local_y"]) for row in rows)
    result = {
        "policy": policy_description,
        "checkpoint": direct_checkpoint,
        "task": task_name,
        "seed": env_cfg.seed,
        "step_height_m": step_height,
        "step_width_m": env_cfg.stairs.step_width,
        "num_steps": env_cfg.stairs.num_steps,
        "total_elevation_m": top_height,
        "command_speed_mps": command_speed,
        "policy_command_speed_mps": policy_command_speed,
        "success_threshold_x_m": success_x,
        "passed": success_step is not None and reset_count == 0 and not left_stair_corridor,
        "time_to_pass_s": None if success_step is None else success_step * env.dt,
        "max_progress_x_m": max_x,
        "max_base_height_m": max_z,
        "max_abs_lateral_offset_m": max_abs_y,
        "left_stair_corridor": left_stair_corridor,
        "reset_count": reset_count,
        "video": str(video_path) if record_video else None,
    }
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"trajectory_csv={csv_path}")
    print(f"summary_json={summary_path}")
    if record_video:
        print(f"video={video_path}")

    if env.viewer is not None:
        env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    evaluate(get_args())
