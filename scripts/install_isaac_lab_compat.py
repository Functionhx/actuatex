#!/usr/bin/env python3
"""Validate and patch a pinned Isaac Lab checkout for the tested Isaac Sim build."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    REPO_ROOT
    / "backends"
    / "isaac_lab"
    / "patches"
    / "0001-isaac-sim-5.1-offline-compat.patch"
)
PINNED_COMMIT = "b4c321024792976150ca55fddb26fa34480d974e"


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _apply(root: Path) -> str:
    command = ["git", "-C", str(root), "apply"]
    quiet = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}
    if subprocess.run(
        [*command, "--check", str(PATCH)], check=False, **quiet
    ).returncode == 0:
        subprocess.run([*command, str(PATCH)], check=True)
        return "applied"
    if subprocess.run(
        [*command, "--reverse", "--check", str(PATCH)], check=False, **quiet
    ).returncode == 0:
        return "already applied"
    raise RuntimeError(
        "compatibility patch does not match this checkout; use the pinned "
        "revision or port the patch manually"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-lab-root", required=True, type=Path)
    parser.add_argument("--allow-version-mismatch", action="store_true")
    args = parser.parse_args()

    root = args.isaac_lab_root.expanduser().resolve()
    if not (root / ".git").is_dir():
        raise FileNotFoundError(f"not an Isaac Lab Git checkout: {root}")
    head = _git(root, "rev-parse", "HEAD")
    if head != PINNED_COMMIT and not args.allow_version_mismatch:
        raise RuntimeError(
            f"Isaac Lab HEAD is {head}; expected {PINNED_COMMIT}. "
            "Checkout the pinned commit or pass --allow-version-mismatch."
        )
    print(f"Isaac Lab revision: {head}")
    print(f"compatibility patch: {_apply(root)}")
    print("Run ActuateX's scripts in backends/isaac_lab/scripts with Isaac Sim Python.")


if __name__ == "__main__":
    main()
