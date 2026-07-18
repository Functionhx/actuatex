"""Create reproducible linear blends of two compatible RSL-RL checkpoints."""

import argparse
import json
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--specialist", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    return parser.parse_args()


def alpha_label(alpha):
    return f"{alpha:.3f}".replace(".", "p")


def main():
    args = parse_args()
    base_path = os.path.abspath(args.base)
    specialist_path = os.path.abspath(args.specialist)
    base = torch.load(base_path, map_location="cpu")
    specialist = torch.load(specialist_path, map_location="cpu")
    base_state = base["model_state_dict"]
    specialist_state = specialist["model_state_dict"]
    if base_state.keys() != specialist_state.keys():
        raise ValueError("checkpoint model_state_dict keys differ")

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    for alpha in args.alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        blended = {}
        for key, base_tensor in base_state.items():
            specialist_tensor = specialist_state[key]
            if base_tensor.shape != specialist_tensor.shape:
                raise ValueError(f"shape mismatch for {key}")
            if torch.is_floating_point(base_tensor):
                blended[key] = torch.lerp(base_tensor, specialist_tensor, alpha)
            else:
                blended[key] = specialist_tensor.clone()
        nonfinite = [
            key for key, value in blended.items()
            if torch.is_floating_point(value) and not torch.isfinite(value).all()
        ]
        if nonfinite:
            raise ValueError(f"non-finite blended tensors: {nonfinite}")

        output_path = os.path.abspath(
            os.path.join(args.out_dir, f"blend_alpha_{alpha_label(alpha)}.pt")
        )
        metadata = {
            "base": base_path,
            "specialist": specialist_path,
            "alpha": alpha,
        }
        torch.save(
            {
                "model_state_dict": blended,
                "iter": 0,
                "blend_metadata": metadata,
            },
            output_path,
        )
        manifest.append({**metadata, "checkpoint": output_path})

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps(manifest, indent=2))
    print("manifest=" + os.path.abspath(manifest_path))


if __name__ == "__main__":
    main()
