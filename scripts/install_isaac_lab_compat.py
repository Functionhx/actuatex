#!/usr/bin/env python3
"""Validate and link the pinned Isaac Sim 6 / Isaac Lab 3 checkout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "backends" / "isaac_lab" / "upstream.json"


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _load_lock() -> dict:
    with LOCK_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _validate_sim_root(root: Path, expected_release: str, allow_mismatch: bool) -> str:
    required = (root / "python.sh", root / "isaac-sim.sh")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"not an Isaac Sim binary root; missing: {', '.join(missing)}")
    version_path = root / "VERSION"
    version = version_path.read_text(encoding="utf-8").strip() if version_path.is_file() else "unknown"
    expected_version = expected_release.split()[0]
    if version != "unknown" and not version.startswith(expected_version) and not allow_mismatch:
        raise RuntimeError(
            f"Isaac Sim VERSION is {version}; expected {expected_version}. "
            "Pass --allow-version-mismatch only for deliberate experiments."
        )
    return version


def _link_runtime(lab_root: Path, sim_root: Path) -> str:
    link = lab_root / "_isaac_sim"
    if os.path.lexists(link):
        if link.is_symlink() and link.resolve() == sim_root:
            return "already linked"
        raise RuntimeError(
            f"{link} already exists and was not changed. Remove or rename it explicitly "
            "after verifying that the old runtime is no longer needed."
        )
    link.symlink_to(sim_root, target_is_directory=True)
    return "linked"


def _verify_python(sim_root: Path) -> str:
    command = [
        str(sim_root / "python.sh"),
        "-c",
        (
            "import sys; "
            "assert sys.version_info[:2] == (3, 12), sys.version; "
            "print(sys.version.split()[0])"
        ),
    ]
    return subprocess.check_output(command, text=True).strip().splitlines()[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-lab-root", required=True, type=Path)
    parser.add_argument("--isaac-sim-root", type=Path)
    parser.add_argument(
        "--link-runtime",
        action="store_true",
        help="create Isaac Lab's _isaac_sim symlink after validation",
    )
    parser.add_argument(
        "--verify-python",
        action="store_true",
        help="run the bundled interpreter and require Python 3.12",
    )
    parser.add_argument("--allow-version-mismatch", action="store_true")
    args = parser.parse_args()

    lock = _load_lock()
    pinned_commit = lock["isaac_lab"]["commit"]
    root = args.isaac_lab_root.expanduser().resolve()
    if not (root / ".git").exists():
        raise FileNotFoundError(f"not an Isaac Lab Git checkout: {root}")
    head = _git(root, "rev-parse", "HEAD")
    if head != pinned_commit and not args.allow_version_mismatch:
        raise RuntimeError(
            f"Isaac Lab HEAD is {head}; expected {pinned_commit}. "
            "Checkout the pinned commit or pass --allow-version-mismatch."
        )
    print(f"Isaac Lab revision: {head}")
    version_path = root / "VERSION"
    if version_path.is_file():
        print(f"Isaac Lab version: {version_path.read_text(encoding='utf-8').strip()}")

    if args.link_runtime and args.isaac_sim_root is None:
        parser.error("--link-runtime requires --isaac-sim-root")
    if args.verify_python and args.isaac_sim_root is None:
        parser.error("--verify-python requires --isaac-sim-root")
    if args.isaac_sim_root is not None:
        sim_root = args.isaac_sim_root.expanduser().resolve()
        sim_version = _validate_sim_root(
            sim_root, lock["isaac_sim"]["release"], args.allow_version_mismatch
        )
        print(f"Isaac Sim version metadata: {sim_version}")
        if args.link_runtime:
            print(f"Isaac Sim link: {_link_runtime(root, sim_root)}")
        if args.verify_python:
            print(f"Isaac Sim Python: {_verify_python(sim_root)}")

    print("Isaac Sim 5.1 compatibility patch: not applied (legacy only)")
    print("Next: run ./isaaclab.sh -i 'rl[rsl-rl]' from the pinned Isaac Lab checkout.")


if __name__ == "__main__":
    main()
