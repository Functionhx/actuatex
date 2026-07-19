#!/usr/bin/env python3
"""Run reproducible Isaac-Lab transfer sweeps for the published Gym actor."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
ARTIFACTS_ROOT = Path(
    os.environ.get("ACTUATEX_ARTIFACTS", REPO_ROOT / "artifacts")
)
DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_CHECKPOINT = ARTIFACTS_ROOT / "checkpoints" / "isaac_gym" / "model.pt"
DEFAULT_OUTPUT = ARTIFACTS_ROOT / "isaac_lab" / "transfer_grid"


# Isaac Sim 6 removed importer-time cylinder replacement.  Every candidate in
# this sweep uses the same audited 4-capsule USD selected by the task config.
# Geometry is therefore held constant while actuator and timing parameters vary.
MEAN_DYNAMICS = ["--armature", "0.01", "--joint_friction", "0.05"]
GYM_GAINS = [
    "--positive_vx_gain", "1.5",
    "--negative_vx_gain", "2.0",
    "--vy_gain", "1.5",
]

CANDIDATES = [
    ("capsule_d0", []),
    ("capsule_d1", ["--action_delay", "1"]),
    ("capsule_d2", ["--action_delay", "2"]),
    ("capsule_d3", ["--action_delay", "3"]),
    ("mean_dyn_d1", MEAN_DYNAMICS + ["--action_delay", "1"]),
    ("mean_dyn_d2", MEAN_DYNAMICS + ["--action_delay", "2"]),
    (
        "mean_dyn_solver_d1",
        MEAN_DYNAMICS
        + [
            "--action_delay", "1",
            "--solver_velocity_iterations", "1",
            "--external_forces_every_iteration",
        ],
    ),
    ("mean_dyn_gym_gains_d1", MEAN_DYNAMICS + GYM_GAINS + ["--action_delay", "1"]),
    ("mean_dyn_gym_gains_d2", MEAN_DYNAMICS + GYM_GAINS + ["--action_delay", "2"]),
    (
        "soft_pd_d1",
        MEAN_DYNAMICS
        + ["--kp", "15", "--kd", "0.35", "--action_delay", "1"],
    ),
    (
        "stiff_pd_d1",
        MEAN_DYNAMICS
        + ["--kp", "25", "--kd", "0.7", "--action_delay", "1"],
    ),
    (
        "scale_0p20_d1",
        MEAN_DYNAMICS
        + ["--action_scale", "0.20", "--action_delay", "1"],
    ),
    (
        "scale_0p30_d1",
        MEAN_DYNAMICS
        + ["--action_scale", "0.30", "--action_delay", "1"],
    ),
    (
        "implicit_mean_d1",
        MEAN_DYNAMICS
        + ["--actuator", "implicit", "--action_delay", "1"],
    ),
]


PRIMARY_METRICS = (
    ("forward_0p3", "vx_rmse", 0.15),
    ("forward_0p6", "vx_rmse", 0.20),
    ("backward_0p3", "vx_rmse", 0.15),
    ("lateral_0p2", "vy_rmse", 0.10),
    ("yaw_0p5", "yaw_rmse", 0.20),
)


def score_result(result):
    segments = result["segments"]
    normalized = [segments[name][metric] / threshold for name, metric, threshold in PRIMARY_METRICS]
    stand = segments["stand"]
    stability = stand["vx_rmse"] / 0.15 + stand["vy_rmse"] / 0.15 + stand["yaw_rmse"] / 0.20
    resets = sum(segment["resets_total"] for segment in segments.values())
    return sum(normalized) + 0.5 * stability + 0.05 * resets


def row_from_result(name, result):
    segments = result["segments"]
    row = {
        "candidate": name,
        "score": score_result(result),
        "resets_total": sum(segment["resets_total"] for segment in segments.values()),
        "stand_vx": segments["stand"]["vx_rmse"],
        "stand_vy": segments["stand"]["vy_rmse"],
        "stand_yaw": segments["stand"]["yaw_rmse"],
    }
    for segment_name, metric, _ in PRIMARY_METRICS:
        row[f"{segment_name}_{metric}"] = segments[segment_name][metric]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    python = Path(os.environ.get("ISAAC_SIM_PYTHON", str(DEFAULT_PYTHON)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    results = {}

    for name, extra_args in CANDIDATES:
        result_path = args.out_dir / f"{name}.json"
        log_path = args.out_dir / f"{name}.log"
        if args.force or not result_path.exists():
            command = [
                str(python),
                str(ROOT / "play_old_policy.py"),
                "--ckpt", str(args.checkpoint),
                "--suite",
                "--num_envs", str(args.num_envs),
                "--out", str(result_path),
                *extra_args,
            ]
            with log_path.open("w", encoding="utf-8") as log_stream:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"candidate {name} failed with exit code {completed.returncode}; "
                    f"see {log_path}"
                )
        with result_path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        results[name] = result
        rows.append(row_from_result(name, result))
        print(f"{name}: score={rows[-1]['score']:.3f}", flush=True)

    rows.sort(key=lambda row: row["score"])
    csv_path = args.out_dir / "ranking.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {
        "checkpoint": str(args.checkpoint.resolve()),
        "best_candidate": rows[0]["candidate"],
        "ranking": rows,
        "candidate_results": results,
    }
    aggregate_path = args.out_dir / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(f"best={rows[0]['candidate']} score={rows[0]['score']:.3f}")
    print(f"ranking={csv_path}")
    print(f"aggregate={aggregate_path}")


if __name__ == "__main__":
    main()
