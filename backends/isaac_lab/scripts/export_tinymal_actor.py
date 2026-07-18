#!/usr/bin/env python
"""Export the deterministic 48-D TinyMal actor from an RSL-RL checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch
import torch.nn as nn


def build_actor() -> nn.Sequential:
    layers = []
    previous = 48
    for width in (512, 256, 128):
        layers.extend((nn.Linear(previous, width), nn.ELU()))
        previous = width
    layers.append(nn.Linear(previous, 12))
    return nn.Sequential(*layers)


def actor_state_from_checkpoint(path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "actor_state_dict" in payload:
        return {
            key[len("mlp.") :]: value
            for key, value in payload["actor_state_dict"].items()
            if key.startswith("mlp.")
        }
    if "model_state_dict" in payload:
        return {
            key[len("actor.") :]: value
            for key, value in payload["model_state_dict"].items()
            if key.startswith("actor.")
        }
    raise KeyError("checkpoint has neither actor_state_dict nor model_state_dict")


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
    parser.add_argument("--name", default="tinymal_actor")
    args = parser.parse_args()

    checkpoint = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    actor = build_actor()
    actor.load_state_dict(actor_state_from_checkpoint(checkpoint), strict=True)
    actor.eval()
    example = torch.zeros(1, 48)
    with torch.no_grad():
        traced = torch.jit.trace(actor, example, strict=True)
        eager_output = actor(example)
        traced_output = traced(example)
    if not torch.equal(eager_output, traced_output):
        raise RuntimeError("TorchScript output differs from eager actor output")

    jit_path = os.path.join(out_dir, f"{args.name}.jit.pt")
    state_path = os.path.join(out_dir, f"{args.name}.state_dict.pt")
    torch.jit.save(traced, jit_path)
    torch.save(actor.state_dict(), state_path)

    manifest = {
        "architecture": "48-512-256-128-12 ELU",
        "checkpoint": checkpoint,
        "input": "48-D policy observation",
        "output": "12-D deterministic joint-position action",
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
