"""Small, crash-resistant helpers for reproducible run artifacts."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sync_directory(path: Path) -> None:
    """Best-effort directory sync so an atomic rename survives a sudden restart."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some filesystems support atomic replacement but not directory fsync.
        pass
    finally:
        os.close(descriptor)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


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
    root = parent.expanduser().resolve()
    if requested_name:
        requested_path = Path(requested_name)
        if (
            requested_path.is_absolute()
            or requested_path.parent != Path(".")
            or requested_path.name in {"", ".", ".."}
        ):
            raise ValueError("run name must be a single directory name beneath run root")
        name = requested_name
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"run-{stamp}"
    run_directory = root / name
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


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a checkpoint into memory."""

    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy_file(source: Path, destination: Path) -> Path:
    """Atomically publish a byte-for-byte copy of an existing artifact."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def ensure_disk_space(path: Path, minimum_free_bytes: int) -> int:
    """Fail early when the filesystem containing ``path`` lacks the requested space."""

    if minimum_free_bytes < 0:
        raise ValueError("minimum free bytes cannot be negative")
    probe = path.expanduser().resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < minimum_free_bytes:
        requested_gib = minimum_free_bytes / (1024**3)
        available_gib = free / (1024**3)
        raise OSError(
            f"insufficient disk space on {probe}: {available_gib:.2f} GiB free, "
            f"{requested_gib:.2f} GiB required"
        )
    return free


def atomic_model_save(model: Any, destination: Path) -> Path:
    """Save an SB3 model to a temporary archive, then atomically publish it."""

    destination = destination.with_suffix(".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = destination.parent / (
        f".{destination.stem}.{uuid.uuid4().hex}.tmp.zip"
    )
    try:
        model.save(temporary_archive)
        if not temporary_archive.exists():
            raise FileNotFoundError(
                f"model.save did not create its requested archive: {temporary_archive}"
            )
        with temporary_archive.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_archive, destination)
        _sync_directory(destination.parent)
    finally:
        temporary_archive.unlink(missing_ok=True)
    return destination
