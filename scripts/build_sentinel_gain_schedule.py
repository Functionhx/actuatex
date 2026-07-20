#!/usr/bin/env python3
"""Combine local Sentinel LQR/H-infinity checkpoints into a gain schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.robomaster.contract import contract_sha256  # noqa: E402
from tasks.robomaster.linear_control import (  # noqa: E402
    make_scheduled_linear_controller_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--command-distance-scales",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 0.4),
        metavar=("VX", "VY", "YAW"),
    )
    parser.add_argument("--schedule-sharpness", type=float, default=12.0)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-report", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _common_source_value(
    sources: list[dict],
    *,
    key: str,
    location: str = "metadata",
) -> str:
    values = []
    for source in sources:
        container = source if location == "root" else source.get(location, {})
        value = str(container.get(key, "")).strip()
        if not value:
            raise ValueError(f"gain schedule source lacks {location}.{key}")
        values.append(value)
    if len(set(values)) != 1:
        raise ValueError(
            f"gain schedule sources disagree on {location}.{key}: {values}"
        )
    return values[0]


def main() -> None:
    args = parse_args()
    checkpoint_paths = [path.resolve() for path in args.checkpoint]
    if len(set(checkpoint_paths)) != len(checkpoint_paths):
        raise ValueError("gain schedule sources must be unique")
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    sources = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in checkpoint_paths
    ]
    source_records = [
        {"checkpoint": str(path), "sha256": _sha256(path)}
        for path in checkpoint_paths
    ]
    backend = _common_source_value(sources, key="backend")
    controller = _common_source_value(
        sources, key="linear_controller", location="root"
    )
    source_contract = _common_source_value(sources, key="contract_sha256")
    if source_contract != contract_sha256():
        raise RuntimeError(
            "gain schedule sources and current Sentinel contract differ"
        )
    metadata = {
        "backend": backend,
        "controller": controller,
        "controller_structure": "command_space_gain_schedule",
        "contract_sha256": source_contract,
        "sources": source_records,
    }
    payload = make_scheduled_linear_controller_checkpoint(
        sources,
        command_distance_scales=args.command_distance_scales,
        schedule_sharpness=args.schedule_sharpness,
        metadata=metadata,
    )
    output_path = args.output_checkpoint.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    report_path = (
        args.output_report.resolve()
        if args.output_report is not None
        else output_path.with_suffix(".json")
    )
    report = {
        **metadata,
        "checkpoint": str(output_path),
        "checkpoint_sha256": _sha256(output_path),
        "operating_commands": payload["operating_commands"].tolist(),
        "command_distance_scales": payload[
            "command_distance_scales"
        ].tolist(),
        "schedule_sharpness": payload["schedule_sharpness"],
        "quality_gate_passed": True,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
