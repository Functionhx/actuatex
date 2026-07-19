#!/usr/bin/env python3
"""Run the canonical Isaac Sim 6 -> MuJoCo wheel-legged transfer sweep.

The sweep deliberately keeps the raw zero-tuning transfer separate from the
deployable command-slew wrapper.  This makes it impossible for a presentation
to hide the difficult command-reversal transient behind the final setting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = Path(__file__).with_name("wheel_legged_sim2sim.py")
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
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "mujoco"
    / "sim2sim"
    / "wheel_legged_isaacsim6_to_mujoco"
    / "canonical"
)


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    arguments: tuple[str, ...] = ()
    note: str = ""


DEPLOYMENT_WRAPPER = (
    "--linear-command-slew",
    "6",
    "--yaw-command-slew",
    "4",
)

CASES = (
    Case(
        "raw_nominal",
        "transfer",
        note="Exact policy/control contract with discontinuous benchmark commands.",
    ),
    Case(
        "deployed_slew6",
        "transfer",
        DEPLOYMENT_WRAPPER,
        "Selected acceleration-limited command wrapper; no policy retuning.",
    ),
    Case(
        "slew8",
        "command_boundary",
        ("--linear-command-slew", "8", "--yaw-command-slew", "4"),
    ),
    Case(
        "slew9",
        "command_boundary",
        ("--linear-command-slew", "9", "--yaw-command-slew", "4"),
    ),
    Case("delay5ms", "delay", DEPLOYMENT_WRAPPER + ("--delay-ms", "5")),
    Case("delay10ms", "delay", DEPLOYMENT_WRAPPER + ("--delay-ms", "10")),
    Case("delay15ms", "delay", DEPLOYMENT_WRAPPER + ("--delay-ms", "15")),
    Case("delay20ms", "delay", DEPLOYMENT_WRAPPER + ("--delay-ms", "20")),
    Case("mass085", "base_mass", DEPLOYMENT_WRAPPER + ("--base-mass-scale", "0.85")),
    Case("mass115", "base_mass", DEPLOYMENT_WRAPPER + ("--base-mass-scale", "1.15")),
    Case("friction045", "friction", DEPLOYMENT_WRAPPER + ("--friction", "0.45")),
    Case("friction060", "friction", DEPLOYMENT_WRAPPER + ("--friction", "0.60")),
    Case("friction080", "friction", DEPLOYMENT_WRAPPER + ("--friction", "0.80")),
    Case("friction120", "friction", DEPLOYMENT_WRAPPER + ("--friction", "1.20")),
    Case("friction140", "friction", DEPLOYMENT_WRAPPER + ("--friction", "1.40")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--actor", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse an existing case JSON instead of running that case again.",
    )
    return parser.parse_args()


def load_case_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    required = {"falls_total", "mean_primary_axis_rmse", "clean_rollout"}
    missing = required.difference(result)
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")
    return result


def main() -> None:
    args = parse_args()
    for path in (args.runner, args.actor, args.reference):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in CASES:
        result_path = args.output_dir / f"{case.name}.json"
        tracking_path = args.output_dir / f"{case.name}.csv"
        if not (args.reuse and result_path.is_file()):
            command = [
                sys.executable,
                str(args.runner),
                "--actor",
                str(args.actor),
                "--reference",
                str(args.reference),
                "--out",
                str(result_path),
                "--tracking",
                str(tracking_path),
                *case.arguments,
            ]
            print(f"[{case.group}] {case.name}", flush=True)
            subprocess.run(command, check=True)

        result = load_case_result(result_path)
        rows.append(
            {
                "name": case.name,
                "group": case.group,
                "arguments": list(case.arguments),
                "note": case.note,
                "clean_rollout": result["clean_rollout"],
                "falls_total": result["falls_total"],
                "mean_primary_axis_rmse": result["mean_primary_axis_rmse"],
                "result": str(result_path.resolve()),
                "tracking_csv": str(tracking_path.resolve()),
            }
        )

    clean_cases = [row["name"] for row in rows if row["clean_rollout"]]
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "22 s / stand, forward, reversal, yaw and arc / one deterministic rollout",
        "actor": str(args.actor.resolve()),
        "reference": str(args.reference.resolve()),
        "selected_deployment_case": "deployed_slew6",
        "important_interpretation": {
            "raw_transfer_is_reported": True,
            "command_slew_is_a_deployment_wrapper_not_policy_retuning": True,
            "friction_sweep_is_expected_to_expose_non_monotonic_contact_sensitivity": True,
        },
        "clean_cases": clean_cases,
        "cases": rows,
    }
    summary_path = args.output_dir / "sweep_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
