#!/usr/bin/env python3
"""Create and atomically advance the frozen U2 cohort dashboard manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROTOCOL = "dungeon-apprentice-v0.2-u2"
COHORT_ID = "v0.2-u2-separated-20260723"
ACTION_CAP = 1_048_576
COHORT_SCIENTIFIC_CAP_BYTES = 6 * 1024**3
MEDIA_CAP_BYTES = 10 * 1024**3
COMBINED_CAP_BYTES = 16 * 1024**3
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{7,64}$")

LINEAGES: tuple[dict[str, Any], ...] = (
    {
        "id": "lineage-1",
        "label": "Apprentice 1 · lead",
        "directory": "v02-u2-seed-20260737",
        "seed": 20260737,
        "worker_streams": [20260737, 20260738, 20260739, 20260740],
        "parent_seed": 20260725,
        "parent_sha256": ("bcce9b8251e97ed4fddda32871c891c3783c057bbb1f89deedb3a3d32058102a"),
    },
    {
        "id": "lineage-2",
        "label": "Apprentice 2 · replication A",
        "directory": "v02-u2-seed-20260741",
        "seed": 20260741,
        "worker_streams": [20260741, 20260742, 20260743, 20260744],
        "parent_seed": 20260729,
        "parent_sha256": ("2a300927b48f966d5f6ddfeefe13d2e444da1e5c70bcd54e86abd6a9b2d1830b"),
    },
    {
        "id": "lineage-3",
        "label": "Apprentice 3 · replication B",
        "directory": "v02-u2-seed-20260745",
        "seed": 20260745,
        "worker_streams": [20260745, 20260746, 20260747, 20260748],
        "parent_seed": 20260733,
        "parent_sha256": ("3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104"),
    },
)
LINEAGE_IDS = tuple(str(lineage["id"]) for lineage in LINEAGES)
UPDATE_PHASES = ("training", "completed", "operational_failure", "interrupted")
LINEAGE_STATES = ("pending", "training", "completed", "mastered", "crashed", "interrupted")
SEGMENT_STATES = ("training", "completed", "mastered", "crashed", "interrupted")
RESUMABLE_KINDS = frozenset({"initial", "resume", "latest"})
CHECKPOINT_SCHEMA_VERSION = 4
MAX_JSON_BYTES = 4 * 1024 * 1024


class ManifestError(RuntimeError):
    """Raised when a launcher would mutate an unexpected cohort manifest."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _validate_root(root: Path) -> Path:
    candidate = root.expanduser()
    if not candidate.is_absolute():
        raise ManifestError("cohort root must be absolute")
    try:
        mode = candidate.lstat().st_mode
    except FileNotFoundError as error:
        raise ManifestError(f"cohort root does not exist: {candidate}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ManifestError(f"cohort root must be a regular directory: {candidate}")
    return candidate.resolve(strict=True)


def _validate_commit(source_commit: str) -> str:
    normalized = source_commit.strip().lower()
    if COMMIT_PATTERN.fullmatch(normalized) is None:
        raise ManifestError("source commit must be a hexadecimal Git object ID")
    return normalized


def _encoded(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _exclusive_write(path: Path, payload: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(_encoded(payload))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(_encoded(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()


def _read_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ManifestError(f"cohort manifest cannot be a symlink: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read cohort manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestError("cohort manifest must be a JSON object")
    return payload


def _expected_lineages() -> list[dict[str, Any]]:
    return [
        dict(lineage)
        | {
            "base_directory": lineage["directory"],
            "state": "pending",
            "segments": [],
        }
        for lineage in LINEAGES
    ]


def _expected_segment_directory(lineage: dict[str, Any], index: int) -> str:
    base = str(lineage["directory"])
    return base if index == 0 else f"{base}-segment-{index:03d}"


def _lineage_definition(lineage_id: str) -> dict[str, Any]:
    return next(lineage for lineage in LINEAGES if lineage["id"] == lineage_id)


def _manifest_lineage(payload: dict[str, Any], lineage_id: str) -> dict[str, Any]:
    lineages = payload["lineages"]
    return next(lineage for lineage in lineages if lineage["id"] == lineage_id)


def _safe_relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{label} must be a nonempty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ManifestError(f"{label} escapes the cohort root")
    return relative


def _read_json_regular(path: Path, *, label: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ManifestError(f"{label} is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ManifestError(f"{label} must be a regular file: {path}")
    if metadata.st_size <= 0 or metadata.st_size > MAX_JSON_BYTES:
        raise ManifestError(f"{label} has an unsafe byte length: {path}")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestError(f"{label} must be a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ManifestError(f"cannot hash resumable artifact {path}: {error}") from error
    return digest.hexdigest()


def _validate_resume_candidate(
    cohort_root: Path,
    *,
    lineage: dict[str, Any],
    source_commit: str,
    segment_directory: str,
    checkpoint_name: str,
) -> dict[str, Any]:
    relative_directory = _safe_relative_path(
        segment_directory,
        label="segment directory",
    )
    segment_root = cohort_root / relative_directory
    checkpoint_root = segment_root / "checkpoints"
    for directory, label in (
        (segment_root, "segment directory"),
        (checkpoint_root, "checkpoint directory"),
    ):
        try:
            directory_metadata = directory.lstat()
        except FileNotFoundError as error:
            raise ManifestError(f"{label} is missing: {directory}") from error
        if stat.S_ISLNK(directory_metadata.st_mode) or not stat.S_ISDIR(directory_metadata.st_mode):
            raise ManifestError(f"{label} is unsafe: {directory}")
    checkpoint = checkpoint_root / checkpoint_name
    try:
        checkpoint.resolve(strict=True).relative_to(cohort_root)
    except (OSError, ValueError) as error:
        raise ManifestError(f"resumable archive escapes the cohort: {checkpoint}") from error
    try:
        checkpoint_metadata = checkpoint.lstat()
    except FileNotFoundError as error:
        raise ManifestError(f"resumable archive is missing: {checkpoint}") from error
    if (
        stat.S_ISLNK(checkpoint_metadata.st_mode)
        or not stat.S_ISREG(checkpoint_metadata.st_mode)
        or checkpoint_metadata.st_size <= 0
    ):
        raise ManifestError(f"resumable archive is unsafe: {checkpoint}")
    sidecar_path = checkpoint.with_suffix(".json")
    integrity_path = checkpoint.with_suffix(".integrity.json")
    sidecar = _read_json_regular(sidecar_path, label="resume sidecar")
    integrity = _read_json_regular(integrity_path, label="resume integrity record")
    checkpoint_digest = _sha256(checkpoint)
    sidecar_digest = _sha256(sidecar_path)
    if (
        integrity.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or integrity.get("protocol") != PROTOCOL
        or integrity.get("checkpoint") != checkpoint.name
        or integrity.get("sidecar") != sidecar_path.name
        or integrity.get("checkpoint_sha256") != checkpoint_digest
        or integrity.get("sidecar_sha256") != sidecar_digest
    ):
        raise ManifestError(f"resume integrity record does not match {checkpoint}")
    source = sidecar.get("source")
    parent = sidecar.get("parent")
    qualification = sidecar.get("qualification")
    segment = sidecar.get("segment")
    progress = sidecar.get("progress")
    if (
        sidecar.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or sidecar.get("protocol") != PROTOCOL
        or sidecar.get("resume_eligible") is not True
        or sidecar.get("kind") not in RESUMABLE_KINDS
        or sidecar.get("checkpoint_sha256") != checkpoint_digest
        or not isinstance(source, dict)
        or source.get("dirty") is not False
        or source.get("commit") != source_commit
        or not isinstance(parent, dict)
        or parent.get("u2_child_seed") != lineage["seed"]
        or parent.get("checkpoint_sha256") != lineage["parent_sha256"]
        or not isinstance(qualification, dict)
        or qualification.get("source_commit") != source_commit
        or not isinstance(segment, dict)
        or segment.get("lineage_algorithm_seed") != lineage["seed"]
        or not isinstance(progress, dict)
    ):
        raise ManifestError(f"resume sidecar provenance changed: {sidecar_path}")
    try:
        child_actions = int(progress["child_trained_actions"])
        segment_index = int(segment["index"])
        trained_actions = int(progress["trained_actions"])
        collected_actions = int(progress["collected_actions"])
        remaining_actions = int(progress["remaining_child_actions"])
    except (KeyError, TypeError, ValueError) as error:
        raise ManifestError(f"resume sidecar counters are invalid: {sidecar_path}") from error
    if (
        not 0 <= child_actions <= ACTION_CAP
        or remaining_actions != ACTION_CAP - child_actions
        or trained_actions != collected_actions
        or segment_index < 0
    ):
        raise ManifestError(f"resume sidecar is not a safe trained boundary: {sidecar_path}")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_relative": str(checkpoint.relative_to(cohort_root)),
        "checkpoint_sha256": checkpoint_digest,
        "child_trained_actions": child_actions,
        "segment_index": segment_index,
        "kind": sidecar["kind"],
    }


def _latest_resume_candidate(
    cohort_root: Path,
    *,
    lineage: dict[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    valid: list[dict[str, Any]] = []
    failures: list[str] = []
    for segment_record in lineage["segments"]:
        directory = str(segment_record["directory"])
        for checkpoint_name in ("initial.zip", "resume.zip", "latest.zip"):
            path = cohort_root / directory / "checkpoints" / checkpoint_name
            if not path.exists() and not path.is_symlink():
                continue
            try:
                candidate = _validate_resume_candidate(
                    cohort_root,
                    lineage=lineage,
                    source_commit=source_commit,
                    segment_directory=directory,
                    checkpoint_name=checkpoint_name,
                )
            except ManifestError as error:
                failures.append(str(error))
            else:
                valid.append(candidate)
    if not valid:
        detail = f"; rejected candidates: {' | '.join(failures)}" if failures else ""
        raise ManifestError(f"no schema-4 safe resume bundle exists for {lineage['id']}{detail}")
    kind_order = {"initial": 0, "latest": 1, "resume": 2}
    return max(
        valid,
        key=lambda item: (
            int(item["child_trained_actions"]),
            int(item["segment_index"]),
            kind_order[str(item["kind"])],
        ),
    )


def _validate_frozen_manifest(
    payload: dict[str, Any],
    *,
    source_commit: str,
) -> None:
    if (
        payload.get("schema_version") != 2
        or payload.get("protocol") != PROTOCOL
        or payload.get("cohort_id") != COHORT_ID
        or payload.get("source_commit") != source_commit
        or payload.get("action_cap_per_lineage") != ACTION_CAP
    ):
        raise ManifestError("cohort manifest does not match the frozen U2 identity")
    lineages = payload.get("lineages")
    if not isinstance(lineages, list) or len(lineages) != len(LINEAGES):
        raise ManifestError("cohort manifest must contain exactly three lineages")
    for actual, expected in zip(lineages, LINEAGES, strict=True):
        for key, value in expected.items():
            if key == "directory":
                continue
            if actual.get(key) != value:
                raise ManifestError(f"cohort lineage {expected['id']} changed frozen field {key}")
        if (
            actual.get("base_directory") != expected["directory"]
            or actual.get("state") not in LINEAGE_STATES
            or not isinstance(actual.get("segments"), list)
        ):
            raise ManifestError(f"cohort lineage {expected['id']} has no state")
        segments = actual["segments"]
        for index, segment in enumerate(segments):
            expected_directory = _expected_segment_directory(expected, index)
            if (
                not isinstance(segment, dict)
                or segment.get("index") != index
                or segment.get("directory") != expected_directory
                or segment.get("state") not in SEGMENT_STATES
                or not isinstance(segment.get("started_at"), str)
                or not isinstance(segment.get("updated_at"), str)
                or (index == 0 and segment.get("resume_checkpoint") is not None)
                or (index > 0 and not isinstance(segment.get("resume_checkpoint"), str))
            ):
                raise ManifestError(f"cohort lineage {expected['id']} has invalid segment history")
        expected_current = (
            str(segments[-1]["directory"]) if segments else str(expected["directory"])
        )
        if actual.get("directory") != expected_current:
            raise ManifestError(f"cohort lineage {expected['id']} current directory changed")
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise ManifestError("cohort manifest has no durable history")


def _directory_usage(root: Path) -> int:
    total = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in directory_names:
            if (base / name).is_symlink():
                raise ManifestError(f"cohort storage contains a symlink: {base / name}")
        for name in file_names:
            path = base / name
            metadata = path.lstat()
            mode = metadata.st_mode
            if stat.S_ISLNK(mode):
                raise ManifestError(f"cohort storage contains a symlink: {path}")
            if stat.S_ISREG(mode):
                total += metadata.st_size
    return total


def _validate_media_directory(
    cohort_root: Path,
    value: str | Path | None,
) -> Path | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() or candidate.parent != cohort_root.parent:
        raise ManifestError(
            "media directory must be an absolute sibling of the cohort root"
        )
    if candidate == cohort_root:
        raise ManifestError("media directory must be separate from the cohort")
    try:
        metadata = candidate.lstat()
    except FileNotFoundError as error:
        raise ManifestError(f"media directory does not exist: {candidate}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ManifestError(f"media directory must be a regular directory: {candidate}")
    resolved = candidate.resolve(strict=True)
    if resolved.parent != cohort_root.parent.resolve(strict=True):
        raise ManifestError("media directory resolves outside the cohort parent")
    return resolved


def _storage_snapshot(
    cohort_root: Path,
    media_directory: str | Path | None,
) -> dict[str, Any]:
    scientific_used = _directory_usage(cohort_root)
    media_path = _validate_media_directory(cohort_root, media_directory)
    media_used = _directory_usage(media_path) if media_path is not None else 0
    combined_used = scientific_used + media_used
    if scientific_used > COHORT_SCIENTIFIC_CAP_BYTES:
        raise ManifestError("U2 scientific cohort exceeds its 6 GiB cap")
    if media_used > MEDIA_CAP_BYTES:
        raise ManifestError("U2 narrative media exceeds its 10 GiB cap")
    if combined_used > COMBINED_CAP_BYTES:
        raise ManifestError("U2 combined footprint exceeds its 16 GiB cap")
    return {
        "cohort_scientific_used_bytes": scientific_used,
        "cohort_scientific_cap_bytes": COHORT_SCIENTIFIC_CAP_BYTES,
        "media_used_bytes": media_used,
        "media_cap_bytes": MEDIA_CAP_BYTES,
        "combined_used_bytes": combined_used,
        "combined_cap_bytes": COMBINED_CAP_BYTES,
    }


def create_manifest(
    root: Path,
    *,
    source_commit: str,
    media_directory: Path | None = None,
) -> Path:
    cohort_root = _validate_root(root)
    commit = _validate_commit(source_commit)
    media_path = _validate_media_directory(cohort_root, media_directory)
    path = cohort_root / "cohort.json"
    if path.exists() or path.is_symlink():
        raise ManifestError(f"refusing to replace an existing cohort manifest: {path}")
    now = _utc_now()
    payload = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "cohort_id": COHORT_ID,
        "phase": "ready",
        "active_lineage_id": None,
        "started_at": now,
        "updated_at": now,
        "elapsed_seconds": 0,
        "source_commit": commit,
        "media_directory": str(media_path) if media_path is not None else None,
        "action_cap_per_lineage": ACTION_CAP,
        "cohort_action_cap": ACTION_CAP * len(LINEAGES),
        "storage": _storage_snapshot(cohort_root, media_path),
        "lineages": _expected_lineages(),
        "history": [
            {
                "at": now,
                "event": "cohort_created",
                "phase": "ready",
            }
        ],
    }
    _exclusive_write(path, payload)
    payload["storage"] = _storage_snapshot(cohort_root, media_path)
    _atomic_write(path, payload)
    return path


def update_manifest(
    root: Path,
    *,
    source_commit: str,
    phase: str,
    active_lineage_id: str | None,
    states: tuple[str, str, str],
) -> Path:
    cohort_root = _validate_root(root)
    commit = _validate_commit(source_commit)
    if phase not in UPDATE_PHASES:
        raise ManifestError(f"unsupported cohort phase: {phase}")
    if active_lineage_id is not None and active_lineage_id not in LINEAGE_IDS:
        raise ManifestError(f"unknown active lineage: {active_lineage_id}")
    if any(state not in LINEAGE_STATES for state in states):
        raise ManifestError("unsupported lineage state")
    path = cohort_root / "cohort.json"
    payload = _read_manifest(path)
    _validate_frozen_manifest(payload, source_commit=commit)
    started_at = datetime.fromisoformat(str(payload["started_at"]))
    if started_at.tzinfo is None:
        raise ManifestError("cohort start timestamp must include a timezone")
    now = datetime.now(UTC)
    payload["phase"] = phase
    payload["active_lineage_id"] = active_lineage_id
    payload["updated_at"] = now.isoformat(timespec="seconds")
    payload["elapsed_seconds"] = max(0, int((now - started_at).total_seconds()))
    payload["storage"] = _storage_snapshot(
        cohort_root,
        payload.get("media_directory"),
    )
    for lineage, state in zip(payload["lineages"], states, strict=True):
        lineage["state"] = state
    payload["history"].append(
        {
            "at": payload["updated_at"],
            "event": "cohort_state",
            "phase": phase,
            "active_lineage_id": active_lineage_id,
            "lineage_states": list(states),
        }
    )
    _atomic_write(path, payload)
    return path


def start_segment(
    root: Path,
    *,
    source_commit: str,
    lineage_id: str,
    directory: str,
    resume_checkpoint: str | None,
) -> Path:
    """Append one immutable segment start and point the dashboard at it."""

    cohort_root = _validate_root(root)
    commit = _validate_commit(source_commit)
    path = cohort_root / "cohort.json"
    payload = _read_manifest(path)
    _validate_frozen_manifest(payload, source_commit=commit)
    lineage = _manifest_lineage(payload, lineage_id)
    expected = _lineage_definition(lineage_id)
    index = len(lineage["segments"])
    expected_directory = _expected_segment_directory(expected, index)
    if directory != expected_directory:
        raise ManifestError(f"segment directory must be {expected_directory}, got {directory}")
    target = cohort_root / _safe_relative_path(directory, label="segment directory")
    if target.exists() or target.is_symlink():
        raise ManifestError(f"refusing to reuse U2 segment target: {target}")
    if index == 0:
        if resume_checkpoint is not None:
            raise ManifestError("the first U2 segment cannot name a resume checkpoint")
    else:
        if resume_checkpoint is None:
            raise ManifestError("a successor U2 segment requires a resume checkpoint")
        candidate = Path(resume_checkpoint).expanduser()
        if not candidate.is_absolute():
            candidate = cohort_root / candidate
        candidate = candidate.resolve(strict=True)
        try:
            candidate_relative = candidate.relative_to(cohort_root)
        except ValueError as error:
            raise ManifestError("resume checkpoint escapes the cohort root") from error
        latest = _latest_resume_candidate(
            cohort_root,
            lineage=lineage,
            source_commit=commit,
        )
        if candidate != Path(str(latest["checkpoint"])):
            raise ManifestError("successor did not select the latest schema-4 safe bundle")
        resume_checkpoint = str(candidate_relative)
    order = list(payload["lineages"]).index(lineage)
    preceding = payload["lineages"][:order]
    following = payload["lineages"][order + 1 :]
    if any(item["state"] not in {"completed", "mastered"} for item in preceding):
        raise ManifestError("U2 lineages must start in frozen sequential order")
    if any(item["state"] != "pending" for item in following):
        raise ManifestError("later U2 lineages must remain pending")
    if index == 0 and lineage["state"] != "pending":
        raise ManifestError("fresh U2 segment requires a pending lineage")
    if index > 0 and lineage["state"] not in {"interrupted", "crashed"}:
        raise ManifestError("resume requires an interrupted or crashed lineage")
    now = _utc_now()
    lineage["segments"].append(
        {
            "index": index,
            "directory": directory,
            "state": "training",
            "started_at": now,
            "updated_at": now,
            "completed_at": None,
            "resume_checkpoint": resume_checkpoint,
        }
    )
    lineage["directory"] = directory
    lineage["state"] = "training"
    payload["phase"] = "training"
    payload["active_lineage_id"] = lineage_id
    payload["updated_at"] = now
    payload["history"].append(
        {
            "at": now,
            "event": "segment_started",
            "lineage_id": lineage_id,
            "segment_index": index,
            "directory": directory,
            "resume_checkpoint": resume_checkpoint,
        }
    )
    _atomic_write(path, payload)
    return path


def finish_segment(
    root: Path,
    *,
    source_commit: str,
    lineage_id: str,
    directory: str,
    state: str,
) -> Path:
    """Durably close the current history entry after trainer termination."""

    if state not in {"completed", "mastered", "crashed", "interrupted"}:
        raise ManifestError(f"unsupported terminal segment state: {state}")
    cohort_root = _validate_root(root)
    commit = _validate_commit(source_commit)
    path = cohort_root / "cohort.json"
    payload = _read_manifest(path)
    _validate_frozen_manifest(payload, source_commit=commit)
    lineage = _manifest_lineage(payload, lineage_id)
    if not lineage["segments"]:
        raise ManifestError("cannot finish a lineage with no segment history")
    segment = lineage["segments"][-1]
    if (
        payload.get("active_lineage_id") != lineage_id
        or lineage.get("state") != "training"
        or segment.get("state") != "training"
        or segment.get("directory") != directory
    ):
        raise ManifestError("terminal event does not match the active U2 segment")
    now = _utc_now()
    segment["state"] = state
    segment["updated_at"] = now
    segment["completed_at"] = now
    status_path = cohort_root / directory / "status.json"
    if status_path.exists() and not status_path.is_symlink():
        status = _read_json_regular(status_path, label="trainer terminal status")
        progress = status.get("child_trained_timesteps")
        if type(progress) is int and 0 <= progress <= ACTION_CAP:
            segment["child_trained_actions"] = progress
        phase = status.get("phase")
        if isinstance(phase, str):
            segment["trainer_phase"] = phase
    lineage["state"] = state
    payload["phase"] = (
        "interrupted"
        if state == "interrupted"
        else "operational_failure"
        if state == "crashed"
        else "training"
    )
    payload["active_lineage_id"] = lineage_id if state in {"interrupted", "crashed"} else None
    payload["updated_at"] = now
    payload["storage"] = _storage_snapshot(
        cohort_root,
        payload.get("media_directory"),
    )
    payload["history"].append(
        {
            "at": now,
            "event": "segment_finished",
            "lineage_id": lineage_id,
            "segment_index": segment["index"],
            "directory": directory,
            "state": state,
        }
    )
    _atomic_write(path, payload)
    return path


def resume_plan(root: Path, *, source_commit: str) -> dict[str, Any]:
    """Return the only permitted next segment for an interrupted cohort."""

    cohort_root = _validate_root(root)
    commit = _validate_commit(source_commit)
    payload = _read_manifest(cohort_root / "cohort.json")
    _validate_frozen_manifest(payload, source_commit=commit)
    if payload.get("phase") not in {"interrupted", "operational_failure"}:
        raise ManifestError("resume mode requires a durably interrupted or crashed cohort")
    candidates = [
        lineage for lineage in payload["lineages"] if lineage["state"] in {"interrupted", "crashed"}
    ]
    if len(candidates) != 1:
        raise ManifestError("resume mode requires exactly one interrupted lineage")
    lineage = candidates[0]
    if payload.get("active_lineage_id") != lineage["id"]:
        raise ManifestError("cohort active lineage does not match its terminal state")
    latest = _latest_resume_candidate(
        cohort_root,
        lineage=lineage,
        source_commit=commit,
    )
    if int(latest["child_trained_actions"]) >= ACTION_CAP:
        raise ManifestError("interrupted lineage has no remaining child action budget")
    index = len(lineage["segments"])
    directory = _expected_segment_directory(
        _lineage_definition(str(lineage["id"])),
        index,
    )
    target = cohort_root / directory
    if target.exists() or target.is_symlink():
        raise ManifestError(f"next U2 segment target already exists: {target}")
    return {
        "lineage_id": lineage["id"],
        "lineage_order": list(payload["lineages"]).index(lineage) + 1,
        "seed": lineage["seed"],
        "directory": directory,
        "resume_checkpoint": latest["checkpoint"],
        "resume_checkpoint_sha256": latest["checkpoint_sha256"],
        "child_trained_actions": latest["child_trained_actions"],
        "segment_index": index,
        "lineage_states": [item["state"] for item in payload["lineages"]],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=("create", "update", "segment-start", "segment-finish", "resume-plan"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--phase", choices=UPDATE_PHASES)
    parser.add_argument("--active-lineage-id", choices=LINEAGE_IDS)
    parser.add_argument("--state", action="append", choices=LINEAGE_STATES, default=[])
    parser.add_argument("--lineage-id", choices=LINEAGE_IDS)
    parser.add_argument("--directory")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--media-directory", type=Path)
    parser.add_argument(
        "--segment-state",
        choices=("completed", "mastered", "crashed", "interrupted"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.operation != "create" and args.media_directory is not None:
        raise SystemExit("--media-directory is frozen when the cohort is created")
    if args.operation == "create":
        if any(
            (
                args.phase is not None,
                args.active_lineage_id is not None,
                bool(args.state),
                args.lineage_id is not None,
                args.directory is not None,
                args.resume_checkpoint is not None,
                args.segment_state is not None,
            )
        ):
            raise SystemExit(
                "create accepts root, source commit, and optional media directory"
            )
        path = create_manifest(
            args.root,
            source_commit=args.source_commit,
            media_directory=args.media_directory,
        )
    elif args.operation == "update":
        if args.phase is None or len(args.state) != len(LINEAGES):
            raise SystemExit("update requires one phase and exactly three ordered states")
        path = update_manifest(
            args.root,
            source_commit=args.source_commit,
            phase=args.phase,
            active_lineage_id=args.active_lineage_id,
            states=tuple(args.state),
        )
    elif args.operation == "segment-start":
        if args.lineage_id is None or args.directory is None:
            raise SystemExit("segment-start requires lineage ID and directory")
        path = start_segment(
            args.root,
            source_commit=args.source_commit,
            lineage_id=args.lineage_id,
            directory=args.directory,
            resume_checkpoint=args.resume_checkpoint,
        )
    elif args.operation == "segment-finish":
        if args.lineage_id is None or args.directory is None or args.segment_state is None:
            raise SystemExit("segment-finish requires lineage ID, directory, and terminal state")
        path = finish_segment(
            args.root,
            source_commit=args.source_commit,
            lineage_id=args.lineage_id,
            directory=args.directory,
            state=args.segment_state,
        )
    else:
        if any(
            (
                args.phase is not None,
                args.active_lineage_id is not None,
                bool(args.state),
                args.lineage_id is not None,
                args.directory is not None,
                args.resume_checkpoint is not None,
                args.media_directory is not None,
                args.segment_state is not None,
            )
        ):
            raise SystemExit("resume-plan accepts only root and source commit")
        plan = resume_plan(args.root, source_commit=args.source_commit)
        print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
        return
    print(path)


if __name__ == "__main__":
    main()
