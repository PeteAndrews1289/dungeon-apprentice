"""Small, crash-resistant helpers for reproducible run artifacts."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def create_run_directory(parent: Path, requested_name: str | None = None) -> Path:
    if requested_name:
        name = requested_name
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"run-{stamp}"
    run_directory = parent.expanduser().resolve() / name
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def git_snapshot(repository: Path) -> dict[str, Any]:
    def capture(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    status = capture("status", "--porcelain")
    return {
        "commit": capture("rev-parse", "HEAD"),
        "branch": capture("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def runtime_snapshot() -> dict[str, Any]:
    packages = {}
    for package in ("gymnasium", "minigrid", "numpy", "sb3-contrib", "stable-baselines3", "torch"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def atomic_model_save(model: Any, destination: Path) -> Path:
    """Save an SB3 model to a temporary archive, then atomically publish it."""

    destination = destination.with_suffix(".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_base = destination.parent / f".{destination.stem}.tmp"
    temporary_archive = temporary_base.with_suffix(".zip")
    model.save(temporary_base)
    source = temporary_base if temporary_base.exists() else temporary_archive
    os.replace(source, destination)
    return destination
