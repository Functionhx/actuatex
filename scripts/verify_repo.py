#!/usr/bin/env python3
"""Fast repository guard: reject local paths, secrets, and generated artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_SIZE = 10 * 1024 * 1024
TEXT_SUFFIXES = {
    ".cfg",
    ".json",
    ".md",
    ".patch",
    ".py",
    ".txt",
    ".urdf",
    ".yaml",
    ".yml",
}
FORBIDDEN_SUFFIXES = {
    ".avi",
    ".ckpt",
    ".engine",
    ".mp4",
    ".onnx",
    ".pth",
    ".pt",
    ".tar",
    ".tgz",
    ".zip",
}
FORBIDDEN_TEXT = {
    "/home/" + "as/": "developer-specific absolute path",
    "vllm/" + "rsl-exp": "workspace-specific path",
    "-----BEGIN " + "PRIVATE KEY-----": "private key",
    "gh" + "p_": "possible GitHub token",
}
REQUIRED = {
    "LICENSE",
    "README.md",
    "README_en.md",
    "THIRD_PARTY.md",
    "docs/CODE_CHANGES_REPORT.zh-CN.md",
    "robots/tinymal/urdf/tinymal.urdf",
    "backends/isaac_gym/upstream.json",
    "backends/isaac_lab/upstream.json",
}
IGNORED_PARTS = {".git", "_deps", "__pycache__", "artifacts"}


def main() -> int:
    errors: list[str] = []
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts)
    )

    for relative in sorted(REQUIRED):
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    for path in files:
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_FILE_SIZE:
            errors.append(f"file exceeds 10 MiB: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"generated/binary artifact is not allowed: {relative}")
        if path.is_symlink() and ROOT not in path.resolve().parents:
            errors.append(f"symlink escapes repository: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"README", "LICENSE"}:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                errors.append(f"expected UTF-8 text: {relative}")
                continue
            for token, description in FORBIDDEN_TEXT.items():
                if token in content:
                    errors.append(f"{relative}: contains {description}: {token!r}")

    for path in (
        ROOT / "backends" / "isaac_gym" / "upstream.json",
        ROOT / "backends" / "isaac_lab" / "upstream.json",
    ):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid upstream manifest {path.relative_to(ROOT)}: {error}")

    if errors:
        print("Repository verification failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    total_size = sum(path.stat().st_size for path in files)
    print(f"Repository verification passed: {len(files)} files, {total_size / 1024 / 1024:.2f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
