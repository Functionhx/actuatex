#!/usr/bin/env python3
"""Apply the ActuateX TinyMal overlay to pinned Isaac Gym dependencies."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backends" / "isaac_gym"


def _run_patch(target: Path, patch_path: Path) -> str:
    base = ["patch", "--batch", "-p1", "-i", str(patch_path)]
    forward = subprocess.run(
        [*base, "--forward", "--dry-run"],
        cwd=target,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if forward.returncode == 0:
        subprocess.run([*base, "--forward"], cwd=target, check=True)
        return "applied"

    reverse = subprocess.run(
        [*base, "--reverse", "--dry-run"],
        cwd=target,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if reverse.returncode == 0:
        return "already applied"
    raise RuntimeError(
        f"{patch_path.name} does not match {target}. Use the pinned upstream "
        f"revision or inspect the rejected dry-run:\n{forward.stdout}"
    )


def _require(path: Path, relative: str) -> None:
    expected = path / relative
    if not expected.is_file():
        raise FileNotFoundError(f"expected upstream file is missing: {expected}")


def _copy_robot(unitree_root: Path) -> None:
    source = REPO_ROOT / "robots" / "tinymal"
    destinations = (
        unitree_root / "legged_gym" / "resources" / "robots" / "tinymal",
        unitree_root / "resources" / "robots" / "tinymal",
    )
    for destination in destinations:
        shutil.copytree(source, destination, dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unitree-root", required=True, type=Path)
    parser.add_argument("--rsl-rl-root", type=Path)
    args = parser.parse_args()

    unitree_root = args.unitree_root.expanduser().resolve()
    _require(unitree_root, "legged_gym/envs/__init__.py")
    _require(unitree_root, "legged_gym/envs/base/base_task.py")

    patches = BACKEND_ROOT / "patches"
    for name in (
        "0001-register-tinymal-tasks.patch",
        "0002-offscreen-rendering.patch",
    ):
        print(f"{name}: {_run_patch(unitree_root, patches / name)}")

    overlay = BACKEND_ROOT / "overlay"
    shutil.copytree(overlay, unitree_root, dirs_exist_ok=True)
    _copy_robot(unitree_root)
    print(f"overlay: copied to {unitree_root}")

    if args.rsl_rl_root is not None:
        rsl_root = args.rsl_rl_root.expanduser().resolve()
        _require(rsl_root, "rsl_rl/algorithms/ppo.py")
        name = "0003-reference-policy-distillation.patch"
        print(f"{name}: {_run_patch(rsl_root, patches / name)}")
    else:
        print("RSL-RL distillation patch: skipped (no --rsl-rl-root)")

    print("ActuateX Isaac Gym integration is ready.")


if __name__ == "__main__":
    main()
