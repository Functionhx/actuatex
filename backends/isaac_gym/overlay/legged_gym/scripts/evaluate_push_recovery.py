"""Push-recovery evaluation for the trained flat-ground TinyMal policy.

Injects world-frame base-force impulses of varying direction / magnitude /
duration into many parallel environments, then measures fall rate, recovery
time and peak attitude deviation. Eval-only: loads the baseline checkpoint,
no further training. Outputs evaluation/tinymal_push/{trajectories.csv,
summary.json}; figures are produced later by the report's analyze script.
"""

import csv
import json
import os

import isaacgym  # noqa: F401  # Must precede torch.
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403  # Registers tasks.
from legged_gym.utils import get_args, task_registry


DIRECTIONS = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}

# Magnitudes span from gentle to violent so the fall-rate curve actually rises.
# 50 N for 0.2 s ~ 8.8 N·s ~ 1.5 m/s impulse on the 5.66 kg robot.
SWEEP_MAGS = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
MAIN_DUR = 0.2  # reference impulse duration for the magnitude sweep (s)
# Separate duration study on the lateral direction (topples most easily).
DUR_STUDY_DIR = "+y"
DUR_STUDY_MAGS = [20.0, 30.0, 40.0]
DUR_STUDY_DURS = [0.1, 0.2, 0.3, 0.5]

T_PUSH = 2.0  # time after start to fire the impulse (s)
T_TOTAL = 8.0  # episode length (s)
CMD_VX = 0.3  # commanded forward speed (m/s)
RECOVERY_BAND = 0.1  # |vx - cmd| below this => recovered


def build_cells():
    cells = []
    # Main sweep: all directions x magnitude at the reference duration.
    for d in DIRECTIONS:
        for m in SWEEP_MAGS:
            cells.append((d, m, MAIN_DUR))
    # Duration study: one direction x selected magnitudes x several durations.
    for m in DUR_STUDY_MAGS:
        for dur in DUR_STUDY_DURS:
            cells.append((DUR_STUDY_DIR, m, dur))
    cells.append(("none", 0.0, 0.0))  # zero-force control
    # Dedup (duration study overlaps the main sweep at dur=MAIN_DUR).
    seen, unique = set(), []
    for c in cells:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def cell_name(cell):
    d, m, dur = cell
    if d == "none":
        return "control_0N"
    return f"{d}_{int(m)}N_{int(round(dur * 1000))}ms"


def evaluate(args):
    cells = build_cells()
    n_cells = len(cells)
    envs_per_cell = int(os.environ.get("TINYMAL_PUSH_PER_CELL", "64"))
    num_envs = envs_per_cell * n_cells

    env_cfg, train_cfg = task_registry.get_cfgs(name="tinymal_push")
    env_cfg.env.num_envs = num_envs
    env_cfg.env.episode_length_s = T_TOTAL
    env_cfg.commands.heading_command = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False

    env, _ = task_registry.make_env(name="tinymal_push", args=args, env_cfg=env_cfg)
    decimation = env_cfg.control.decimation
    dev = env.device

    # Map each env to a cell in contiguous blocks.
    env_cell = torch.tensor(
        np.repeat(np.arange(n_cells), envs_per_cell), device=dev, dtype=torch.long
    )

    # Per-env impulse schedule.
    force_xy = torch.zeros(num_envs, 2, device=dev)
    substeps = torch.zeros(num_envs, dtype=torch.long, device=dev)
    for ci, (d, m, dur) in enumerate(cells):
        idx = (env_cell == ci).nonzero(as_tuple=True)[0]
        if d == "none":
            continue
        dx, dy = DIRECTIONS[d]
        force_xy[idx, 0] = m * dx
        force_xy[idx, 1] = m * dy
        substeps[idx] = int(round(dur / env.sim_params.dt))

    # Defaults reproduce the historical baseline; env vars select a new policy
    # without overwriting or moving the original checkpoint.
    direct_checkpoint = os.environ.get("TINYMAL_PUSH_CHECKPOINT_PATH")
    train_cfg.runner.resume = direct_checkpoint is None
    train_cfg.runner.experiment_name = os.environ.get(
        "TINYMAL_PUSH_EXPERIMENT", "tinymal_baseline"
    )
    train_cfg.runner.load_run = os.environ.get(
        "TINYMAL_PUSH_LOAD_RUN", "Jul17_23-52-15_std0p3_seed1"
    )
    train_cfg.runner.checkpoint = int(os.environ.get("TINYMAL_PUSH_CHECKPOINT", "1500"))
    runner, _ = task_registry.make_alg_runner(
        env=env, name="tinymal_push", args=args, train_cfg=train_cfg
    )
    if direct_checkpoint is not None:
        direct_checkpoint = os.path.abspath(direct_checkpoint)
        checkpoint = torch.load(direct_checkpoint, map_location=dev)
        runner.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded direct checkpoint: {direct_checkpoint}")
    policy = runner.get_inference_policy(device=dev)

    dt = env.dt
    push_step = int(round(T_PUSH / dt))
    total_steps = int(round(T_TOTAL / dt))
    impulse_end_step = push_step + torch.ceil(substeps.float() / decimation).long()

    fallen = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    recovered = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    recovery_time = torch.full((num_envs,), float("nan"), device=dev)
    peak_roll = torch.zeros(num_envs, device=dev)
    peak_pitch = torch.zeros(num_envs, device=dev)

    counts = torch.zeros(n_cells, device=dev)
    counts.index_add_(0, env_cell, torch.ones(num_envs, device=dev))
    counts_safe = counts.clamp(min=1)

    output_dir = os.environ.get(
        "TINYMAL_PUSH_OUT",
        os.path.join(LEGGED_GYM_ROOT_DIR, "evaluation", "tinymal_push"),
    )
    os.makedirs(output_dir, exist_ok=True)
    traj_path = os.path.join(output_dir, "trajectories.csv")

    def per_cell_sum(vals):
        s = torch.zeros(n_cells, device=dev)
        s.index_add_(0, env_cell, vals)
        return s / counts_safe

    traj_rows = []
    with torch.inference_mode():
        for step in range(total_steps):
            env.commands[:, 0] = CMD_VX
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            if step == push_step:
                env.schedule_base_force(force_xy, substeps)

            env.compute_observations()
            actions = policy(env.get_observations())
            _, _, _, _dones, _extras = env.step(actions)

            t = step * dt
            roll = env.rpy[:, 0].abs()
            pitch = env.rpy[:, 1].abs()
            vx = env.base_lin_vel[:, 0]
            is_timeout = env.time_out_buf.bool()
            is_reset = env.reset_buf.bool()

            # Peak attitude only pre-fall.
            not_yet_fallen = ~fallen
            peak_roll = torch.maximum(
                peak_roll, torch.where(not_yet_fallen, roll, peak_roll)
            )
            peak_pitch = torch.maximum(
                peak_pitch, torch.where(not_yet_fallen, pitch, peak_pitch)
            )

            # Fall = non-timeout reset at/after the push.
            just_fell = is_reset & ~is_timeout & ~fallen & (step >= push_step)
            fallen = fallen | just_fell

            # Recovery = |vx - cmd| within band after the impulse ends, not fallen.
            step_t = torch.tensor(step, device=dev)
            in_band = (vx - CMD_VX).abs() < RECOVERY_BAND
            past_impulse = step_t >= impulse_end_step
            just_recovered = in_band & past_impulse & ~fallen & ~recovered
            recovered = recovered | just_recovered
            recovery_time = torch.where(
                just_recovered,
                (step_t.float() - impulse_end_step.float()) * dt,
                recovery_time,
            )

            # Per-cell means (vectorized; single sync per step).
            vx_pc = per_cell_sum(vx)
            roll_pc = per_cell_sum(roll)
            pitch_pc = per_cell_sum(pitch)
            fallfrac_pc = per_cell_sum(fallen.float())
            vx_l, roll_l, pitch_l, fall_l = (
                vx_pc.tolist(),
                roll_pc.tolist(),
                pitch_pc.tolist(),
                fallfrac_pc.tolist(),
            )
            for ci, cell in enumerate(cells):
                traj_rows.append(
                    {
                        "time_s": round(t, 4),
                        "cell": cell_name(cell),
                        "direction": cell[0],
                        "magnitude_N": cell[1],
                        "duration_ms": int(round(cell[2] * 1000)),
                        "cmd_vx": CMD_VX,
                        "vx_mean": vx_l[ci],
                        "abs_roll_mean": roll_l[ci],
                        "abs_pitch_mean": pitch_l[ci],
                        "fall_frac": fall_l[ci],
                        "n": int(envs_per_cell),
                    }
                )

    with open(traj_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=list(traj_rows[0].keys()))
        writer.writeheader()
        writer.writerows(traj_rows)

    summary = {}
    for ci, cell in enumerate(cells):
        mask = env_cell == ci
        idx = mask.nonzero(as_tuple=True)[0]
        rec_times = recovery_time[idx][recovered[idx]]
        summary[cell_name(cell)] = {
            "direction": cell[0],
            "magnitude_N": cell[1],
            "duration_ms": int(round(cell[2] * 1000)),
            "n_envs": int(mask.sum().item()),
            "fall_rate": fallen[idx].float().mean().item(),
            "recovery_rate": recovered[idx].float().mean().item(),
            "recovery_time_mean_s": float(rec_times.mean().item()) if rec_times.numel() else None,
            "recovery_time_median_s": float(rec_times.median().item()) if rec_times.numel() else None,
            "peak_roll_mean_rad": peak_roll[idx].mean().item(),
            "peak_pitch_mean_rad": peak_pitch[idx].mean().item(),
        }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"trajectories_csv={traj_path}")
    print(f"summary_json={summary_path}")
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    evaluate(get_args())
