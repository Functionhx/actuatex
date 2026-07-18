"""Rank robust checkpoints using reproducible MuJoCo acceptance evidence."""

import csv
import json
import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = Path(
    os.environ.get("ACTUATEX_ARTIFACTS", PROJECT_ROOT / "artifacts")
)
EVALUATION_ROOT = ARTIFACTS_ROOT / "isaac_gym"
OUTPUT_DIR = ARTIFACTS_ROOT / "checkpoint_comparison"

TRACKING_GATES = {
    "forward_0p3": ("vx_rmse", 0.15),
    "forward_0p6": ("vx_rmse", 0.20),
    "backward_0p3": ("vx_rmse", 0.15),
    "lateral_0p2": ("vy_rmse", 0.15),
    "yaw_0p5": ("yaw_rmse", 0.25),
}


def _iteration(path):
    match = re.search(r"iter(\d+)$", path.parent.name)
    if match is None:
        raise ValueError(f"cannot parse iteration from {path}")
    return int(match.group(1))


def collect():
    rows = []
    for summary_path in sorted(
        EVALUATION_ROOT.glob("sim2sim_mujoco_robust_iter*/summary.json")
    ):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        iteration = _iteration(summary_path)
        row = {"iteration": iteration, "summary": str(summary_path)}
        normalized_errors = []
        gates_passed = True
        all_survived = True
        for segment, (metric, threshold) in TRACKING_GATES.items():
            value = data.get(segment, {}).get(metric)
            fallen = data.get(segment, {}).get("fallen", True)
            row[f"{segment}_{metric}"] = value
            row[f"{segment}_fallen"] = fallen
            if value is None:
                gates_passed = False
                normalized_errors.append(10.0)
            else:
                gates_passed &= value <= threshold
                normalized_errors.append(value / threshold)
            all_survived &= not fallen
        row["tracking_gates_passed"] = bool(gates_passed and all_survived)
        row["all_segments_survived"] = bool(all_survived)
        row["normalized_tracking_score"] = sum(normalized_errors) / len(
            normalized_errors
        )

        task_path = summary_path.with_name("tasks_summary.json")
        if task_path.exists():
            tasks = json.loads(task_path.read_text(encoding="utf-8"))
            row["stairs_20mm_passed"] = bool(tasks["stairs"]["passed"])
            pushes_30n = [
                case for case in tasks["pushes"] if case["magnitude_n"] == 30.0
            ]
            row["push_30n_pass_rate"] = sum(
                not case["fallen"] and case["recovered"] for case in pushes_30n
            ) / max(1, len(pushes_30n))
        else:
            row["stairs_20mm_passed"] = None
            row["push_30n_pass_rate"] = None
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            not row["tracking_gates_passed"],
            row["normalized_tracking_score"],
        ),
    )


def main():
    rows = collect()
    if not rows:
        raise RuntimeError("no robust MuJoCo checkpoint evaluations found")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "mujoco_checkpoint_comparison.json"
    csv_path = OUTPUT_DIR / "mujoco_checkpoint_comparison.csv"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"json={json_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
