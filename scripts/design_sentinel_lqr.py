#!/usr/bin/env python3
"""Synthesize a Sentinel LQR artifact from a validated local model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.robomaster.contract import ACTION_DIM, contract_sha256  # noqa: E402
from tasks.robomaster.linear_control import (  # noqa: E402
    CONTROL_STATE_DIM,
    CONTROL_STATE_NAMES,
    DEFAULT_ACTION_COST_DIAGONAL,
    DEFAULT_STATE_COST_DIAGONAL,
    LEFT_RIGHT_ACTION_MIRROR,
    LEFT_RIGHT_STATE_MIRROR,
    design_discrete_lqr,
    feedforward_action_from_command,
    make_linear_controller_checkpoint,
    symmetrize_left_right_dynamics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-report", type=Path)
    parser.add_argument("--state-cost-scale", type=float, default=1.0)
    parser.add_argument("--yaw-rate-cost-scale", type=float, default=1.0)
    parser.add_argument("--lateral-stability-cost-scale", type=float, default=1.0)
    parser.add_argument("--leg-action-cost-scale", type=float, default=1.0)
    parser.add_argument(
        "--leg-antisymmetric-action-cost-scale", type=float, default=1.0
    )
    parser.add_argument("--wheel-action-cost-scale", type=float, default=1.0)
    parser.add_argument("--forward-feedforward-scale", type=float, default=1.0)
    parser.add_argument("--yaw-feedforward-scale", type=float, default=1.0)
    parser.add_argument("--enforce-left-right-symmetry", action="store_true")
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-report", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    scales = (
        args.state_cost_scale,
        args.yaw_rate_cost_scale,
        args.lateral_stability_cost_scale,
        args.leg_action_cost_scale,
        args.leg_antisymmetric_action_cost_scale,
        args.wheel_action_cost_scale,
        args.forward_feedforward_scale,
        args.yaw_feedforward_scale,
    )
    if not all(np.isfinite(value) and value > 0.0 for value in scales):
        raise ValueError("all LQR cost scales must be finite and positive")
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model_report_path = (
        args.model_report.resolve()
        if args.model_report is not None
        else model_path.with_suffix(".json")
    )
    if not model_report_path.is_file():
        raise FileNotFoundError(model_report_path)
    model_report = json.loads(model_report_path.read_text(encoding="utf-8"))
    if not model_report.get("quality_gate_passed", False):
        raise RuntimeError("refusing to synthesize from a failed model report")
    if model_report.get("contract_sha256") != contract_sha256():
        raise RuntimeError("local model and current Sentinel contract differ")
    operating_command = np.asarray(
        model_report.get("operating_command", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if operating_command.shape != (3,) or not np.isfinite(
        operating_command
    ).all():
        raise ValueError("model report contains an invalid operating command")

    with np.load(model_path) as model:
        matrix_a = model["matrix_a"]
        matrix_b = model["matrix_b"]
        bias = model["bias"]
        state_center = model["state_center"]
        action_center = model["action_center"]
        state_names = tuple(str(name) for name in model["control_state_names"])
    if matrix_a.shape != (CONTROL_STATE_DIM, CONTROL_STATE_DIM):
        raise ValueError(f"invalid A shape {matrix_a.shape}")
    if matrix_b.shape != (CONTROL_STATE_DIM, ACTION_DIM):
        raise ValueError(f"invalid B shape {matrix_b.shape}")
    if state_names != CONTROL_STATE_NAMES:
        raise RuntimeError("local-model control state order drifted")
    enforce_symmetry = args.enforce_left_right_symmetry
    if enforce_symmetry:
        matrix_a, matrix_b, bias = symmetrize_left_right_dynamics(
            matrix_a, matrix_b, bias
        )
        state_center = 0.5 * (
            state_center + LEFT_RIGHT_STATE_MIRROR @ state_center
        )
        action_center = 0.5 * (
            action_center + LEFT_RIGHT_ACTION_MIRROR @ action_center
        )

    state_cost_diagonal = (
        DEFAULT_STATE_COST_DIAGONAL * args.state_cost_scale
    )
    state_cost_diagonal[CONTROL_STATE_NAMES.index("yaw_rate_error")] *= (
        args.yaw_rate_cost_scale
    )
    for state_name in (
        "vy_error",
        "roll_rate",
        "projected_gravity_y",
    ):
        state_cost_diagonal[CONTROL_STATE_NAMES.index(state_name)] *= (
            args.lateral_stability_cost_scale
        )
    action_cost_diagonal = DEFAULT_ACTION_COST_DIAGONAL.copy()
    action_cost_diagonal[:4] *= args.leg_action_cost_scale
    action_cost_diagonal[4:] *= args.wheel_action_cost_scale
    action_cost = np.diag(action_cost_diagonal)
    antisymmetric_leg_extra = (
        args.leg_antisymmetric_action_cost_scale - 1.0
    ) * action_cost_diagonal[0]
    for left, right in ((0, 2), (1, 3)):
        direction = np.zeros(ACTION_DIM, dtype=np.float64)
        direction[left] = 2.0**-0.5
        direction[right] = -(2.0**-0.5)
        action_cost += antisymmetric_leg_extra * np.outer(direction, direction)
    design = design_discrete_lqr(
        matrix_a,
        matrix_b,
        np.diag(state_cost_diagonal),
        action_cost,
    )
    spectral_radius = float(max(abs(design.closed_loop_eigenvalues)))
    if design.controllability_rank != CONTROL_STATE_DIM:
        raise RuntimeError("identified local model is not fully controllable")
    if not np.isfinite(spectral_radius) or spectral_radius >= 1.0:
        raise RuntimeError(
            f"LQR failed the discrete stability gate: rho={spectral_radius}"
        )

    equilibrium_feedforward = feedforward_action_from_command(
        operating_command
    )
    controller_action_offset = action_center - equilibrium_feedforward
    metadata = {
        "backend": str(model_report["backend"]),
        "controller": "lqr",
        "contract_sha256": contract_sha256(),
        "source_model": str(model_path),
        "source_model_sha256": _sha256(model_path),
        "source_model_report": str(model_report_path),
        "state_cost_diagonal": state_cost_diagonal.tolist(),
        "yaw_rate_cost_scale": args.yaw_rate_cost_scale,
        "lateral_stability_cost_scale": args.lateral_stability_cost_scale,
        "forward_feedforward_scale": args.forward_feedforward_scale,
        "yaw_feedforward_scale": args.yaw_feedforward_scale,
        "action_cost_diagonal": action_cost_diagonal.tolist(),
        "action_cost_matrix": action_cost.tolist(),
        "leg_antisymmetric_action_cost_scale": (
            args.leg_antisymmetric_action_cost_scale
        ),
        "left_right_symmetry_enforced": enforce_symmetry,
        "closed_loop_spectral_radius": spectral_radius,
        "closed_loop_eigenvalues": [
            {"real": float(value.real), "imag": float(value.imag)}
            for value in design.closed_loop_eigenvalues
        ],
        "controllability_rank": design.controllability_rank,
        "gain_frobenius_norm": float(np.linalg.norm(design.gain)),
        "maximum_absolute_gain": float(np.max(np.abs(design.gain))),
        "affine_bias_norm": float(np.linalg.norm(bias)),
        "operating_command": operating_command.tolist(),
        "identified_action_center": action_center.tolist(),
        "equilibrium_feedforward": equilibrium_feedforward.tolist(),
        "controller_action_offset": controller_action_offset.tolist(),
        "stabilizer_checkpoint_used_for_identification": model_report.get(
            "stabilizer_checkpoint"
        ),
        "stabilizer_is_required_at_inference": False,
    }
    payload = make_linear_controller_checkpoint(
        controller="lqr",
        gain=design.gain,
        state_center=state_center,
        action_offset=controller_action_offset,
        forward_feedforward_scale=args.forward_feedforward_scale,
        yaw_feedforward_scale=args.yaw_feedforward_scale,
        metadata=metadata,
    )
    checkpoint_path = args.output_checkpoint.resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    report_path = (
        args.output_report.resolve()
        if args.output_report is not None
        else checkpoint_path.with_suffix(".json")
    )
    report = {
        **metadata,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
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
