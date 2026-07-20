#!/usr/bin/env python
"""Export the deterministic serial wheel-legged actor from an RSL-RL checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch
import torch.nn as nn


OBSERVATION_DIM = 28
ACTION_DIM = 6
HIDDEN_DIMS = (512, 256, 128)


def build_actor() -> nn.Sequential:
    """Reconstruct the policy MLP without an RSL-RL runtime dependency."""

    layers: list[nn.Module] = []
    previous = OBSERVATION_DIM
    for width in HIDDEN_DIMS:
        layers.extend((nn.Linear(previous, width), nn.ELU()))
        previous = width
    layers.append(nn.Linear(previous, ACTION_DIM))
    return nn.Sequential(*layers)


def actor_state_from_checkpoint(path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in payload:
        raise KeyError(f"{path} has no actor_state_dict")
    state = {
        key[len("mlp.") :]: value
        for key, value in payload["actor_state_dict"].items()
        if key.startswith("mlp.")
    }
    if not state:
        raise KeyError(f"{path} contains no actor MLP weights")
    return state


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name", default="serial_wheel_legged_robust")
    args = parser.parse_args()

    checkpoint = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    actor = build_actor()
    actor.load_state_dict(actor_state_from_checkpoint(checkpoint), strict=True)
    actor.eval()

    # Use more than one row so tracing validation also covers batched inference.
    generator = torch.Generator().manual_seed(20260719)
    example = torch.randn(17, OBSERVATION_DIM, generator=generator)
    with torch.no_grad():
        traced = torch.jit.trace(actor, example, strict=True)
        eager_output = actor(example)
        traced_output = traced(example)
    max_abs_error = float((eager_output - traced_output).abs().max().item())
    if max_abs_error != 0.0:
        raise RuntimeError(
            f"TorchScript output differs from eager actor: max_abs_error={max_abs_error}"
        )

    jit_path = os.path.join(out_dir, f"{args.name}.jit.pt")
    state_path = os.path.join(out_dir, f"{args.name}.state_dict.pt")
    torch.jit.save(traced, jit_path)
    torch.save(actor.state_dict(), state_path)

    manifest = {
        "architecture": "28-512-256-128-6 ELU",
        "backend_trained": (
            "Isaac Sim 6.0.1 GA / Isaac Lab 3.0.0-beta2.patch1 / PhysX 5"
        ),
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(checkpoint),
        "input": {
            "dimension": OBSERVATION_DIM,
            "layout": [
                "base_lin_vel[3]",
                "base_ang_vel[3]",
                "projected_gravity[3]",
                "command_vx_vy_yaw[3]",
                "leg_joint_pos_relative[4]",
                "joint_vel_scaled_0p05[6]",
                "previous_action[6]",
            ],
        },
        "output": {
            "dimension": ACTION_DIM,
            "layout": [
                "left_hip_position",
                "left_knee_position",
                "right_hip_position",
                "right_knee_position",
                "left_wheel_velocity",
                "right_wheel_velocity",
            ],
            "note": "Actions are normalized and must pass through the task action scales.",
        },
        "validation": {
            "batch_size": example.shape[0],
            "eager_vs_torchscript_max_abs_error": max_abs_error,
        },
        "artifacts": {
            os.path.basename(jit_path): {
                "bytes": os.path.getsize(jit_path),
                "sha256": sha256(jit_path),
            },
            os.path.basename(state_path): {
                "bytes": os.path.getsize(state_path),
                "sha256": sha256(state_path),
            },
        },
    }
    manifest_path = os.path.join(out_dir, f"{args.name}.export.json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
