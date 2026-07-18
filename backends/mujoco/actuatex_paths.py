"""Portable paths shared by the standalone MuJoCo tools."""

from __future__ import annotations

import os
from pathlib import Path


_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(
    os.environ.get("ACTUATEX_ROOT", str(_DEFAULT_REPO_ROOT))
).expanduser().resolve()
ROBOT_URDF = Path(
    os.environ.get(
        "ACTUATEX_TINYMAL_URDF",
        str(REPO_ROOT / "robots" / "tinymal" / "urdf" / "tinymal.urdf"),
    )
).expanduser().resolve()
ARTIFACTS_ROOT = Path(
    os.environ.get("ACTUATEX_ARTIFACTS", str(REPO_ROOT / "artifacts"))
).expanduser().resolve()
RSL_RL_ROOT = Path(
    os.environ.get("RSL_RL_ROOT", str(REPO_ROOT / "_deps" / "rsl_rl"))
).expanduser().resolve()
