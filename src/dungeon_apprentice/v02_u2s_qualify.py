"""Fail-closed qualification for the prospective U2-S stability ablation.

The qualification is intentionally created *after* the implementation commit
and annotated tag are published, but *before* the cohort root exists.  It
authenticates the exact confirmed U1 parent, the sealed U2 generator access,
the completed U2/U2r evidence, the extended training-layout exclusions, and
the four-arm intervention contract.  The resulting report is immutable input
to the sequential launcher; it never authorizes a confirmation partition or
promotes an ablation checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from dungeon_apprentice import u2_qualification_anchor, u2r_anchor, v02_u2s
from dungeon_apprentice import v02_u1_confirm as state_digests
from dungeon_apprentice import v02_u2 as frozen_u2
from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v02_u2r as frozen_u2r
from dungeon_apprentice.artifacts import file_sha256, utc_now
from dungeon_apprentice.contracts import (
    CURIOSITY_BUDGET,
    DEFAULT_CURIOSITY_SCALE,
    STEP_REWARD,
    SUCCESS_REWARD,
)
from dungeon_apprentice.u2_seed_guard import (
    PARTITION_BY_ROLE,
    SEED_PARTITIONS,
    U2SeedAccessError,
    U2SeedRole,
    authorize_u2_seed,
)
from dungeon_apprentice.u2_storage import (
    COHORT_SCIENTIFIC_CAP_BYTES,
    COMBINED_PLANNED_CAP_BYTES,
    LINEAGE_CAP_BYTES,
    MEDIA_CAP_BYTES,
)

PROTOCOL = v02_u2s.PROTOCOL
SCHEMA_VERSION = 1
KIND = "sealed_preflight_qualification"
VERDICT = "qualified"

QUALIFIED_TAG = "u2s-stability-ablation-v0.2-u2s-20260723"
QUALIFIED_REMOTE = "origin"
EXPECTED_ORIGIN_URL = "https://github.com/PeteAndrews1289/dungeon-apprentice.git"
PROTOCOL_DOCUMENT = Path("docs/protocol-v0.2-u2s-stability-ablation.md")

CANONICAL_QUALIFICATION_DIRECTORY = Path(
    "/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.2-u2s-20260723"
)
CANONICAL_REPORT = CANONICAL_QUALIFICATION_DIRECTORY / "report.json"
CANONICAL_CHECKSUM = CANONICAL_QUALIFICATION_DIRECTORY / "report.json.sha256"
CANONICAL_COHORT_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/u2s-ablation-20260723"
)
CANONICAL_MEDIA_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/u2s-ablation-media-20260723"
)

BASE_QUALIFICATION_REPORT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/qualifications/"
    "v0.2-u2-20260723/report.json"
)
U2R_RUN_DIRECTORY = Path(
    "/Volumes/T7 Developer/DungeonApprentice/u2r-stability-r1-20260723/"
    "v02-u2r-r1-seed-20260745"
)
U2R_REPORT_SHA256 = (
    "dcfbcbc9fb3e042d44c1bb7762479f005a24a989a96611b85b102c34f955fcc2"
)
U2R_REPORT_INTEGRITY_SHA256 = (
    "8547da16c80c8afb0e594928a59f067df52b42af26b2417b28af930fc42ece29"
)
U2R_EPISODE_STARTS_SHA256 = (
    "b07de4d7ff56257495783b2413076d87bc6e50525ee06b92d0b1a81d6683aad0"
)
U2R_TERMINAL_ACTIVE_RECORDS_SHA256 = (
    "93663439a363a4c47152cc4e320654a1f2807a708637777b1fe71b358817e522"
)

PARENT_SIDECAR_SHA256 = (
    "268f89361521dd855ba637254b236592186992718ad37ac374e85e5dbf408a07"
)
PARENT_MANIFEST_SHA256 = (
    "cd808ba8fea6b79225895b455d975122f85d66f1451569bd04a69737f8a6dfba"
)
PARENT_POLICY_TENSOR_SHA256 = (
    "e555d3f7e2364f74e3b43371938c2f25e2ddbf62558a810038509a70868e6888"
)
PARENT_OPTIMIZER_STATE_SHA256 = (
    "cc07791b374620d680e5cbb3602d59ab83eb55f195d26808abc0adf5f0e3bc5f"
)

_SHA256 = frozenset("0123456789abcdef")
_GIT_OBJECT_LENGTHS = frozenset({40, 64})
MAX_REPORT_BYTES = 2 * 1024**2
MAX_RELEASE_OUTPUT_BYTES = 2 * 1024**2
TAG_KIND = "u2s_source_preregistration"
TAG_SCHEMA_VERSION = 1
TAG_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "protocol",
        "tag",
        "remote",
        "remote_url",
        "source_commit",
        "protocol_document_sha256",
        "parent",
        "base_qualification_sha256",
        "u2r_terminal_report_sha256",
        "failed_r0_launch_sha256",
        "guard_mapping_sha256",
        "arm_contract_sha256",
        "algorithm_seed",
        "worker_streams",
        "child_action_budget_per_arm",
        "evaluation_interval",
        "exam_count_per_arm",
        "protected_seed_partitions",
        "protected_seed_partitions_sha256",
        "roots",
        "dashboard_port",
        "storage_caps",
        "terminal_rule",
        "resume_rule",
    }
)
RELEASE_COMMANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ruff", (".venv/bin/ruff", "check", ".")),
    ("full_pytest", (".venv/bin/pytest", "-q")),
    (
        "u2s_focused_pytest",
        (
            ".venv/bin/pytest",
            "-q",
            "tests/test_v02_u2s.py",
            "tests/test_v02_u2s_operational.py",
            "tests/test_v02_u2s_dashboard.py",
            "tests/test_v02_u2s_qualify.py",
            "tests/test_v02_u2s_smoke.py",
        ),
    ),
    (
        "u2s_disposable_integration_smoke",
        (
            ".venv/bin/python",
            "-m",
            "dungeon_apprentice.v02_u2s_smoke",
        ),
    ),
    (
        "mechanical_oracle",
        (
            ".venv/bin/python",
            "-m",
            "dungeon_apprentice.qualify",
            "--seeds",
            "100",
        ),
    ),
)
_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "kind",
        "verdict",
        "created_at",
        "source",
        "protocol_document",
        "parent",
        "base_qualification",
        "u2r_terminal",
        "guards",
        "sampler_preflight",
        "arm_contract",
        "resume_contract",
        "protected_partitions",
        "release_checks",
        "storage_caps",
        "storage_preflight",
        "restrictions",
    }
)


class U2SQualificationError(RuntimeError):
    """Raised when U2-S cannot prove its prospective launch contract."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_set_sha256(values: Sequence[str] | set[str] | frozenset[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(values)):
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in _SHA256 for character in result):
        raise U2SQualificationError(f"{label} is not a lowercase SHA-256 digest")
    return result


def _require_git_object(value: Any, label: str) -> str:
    result = str(value)
    if (
        len(result) not in _GIT_OBJECT_LENGTHS
        or any(character not in _SHA256 for character in result)
    ):
        raise U2SQualificationError(f"{label} is not a Git object ID")
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise U2SQualificationError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise U2SQualificationError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise U2SQualificationError(f"{label} is not a JSON object")
    return value


def _regular_file_bytes_once(path: Path, label: str, *, maximum: int) -> bytes:
    if maximum < 1:
        raise ValueError("maximum byte count must be positive")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise U2SQualificationError(f"{label} is missing or unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= maximum:
            raise U2SQualificationError(f"{label} size or file type is invalid")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                raise U2SQualificationError(f"{label} ended before its bound size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise U2SQualificationError(f"{label} grew while it was authenticated")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _reject_symlink_chain(path: Path) -> None:
    current = path.expanduser().absolute()
    while True:
        if current.is_symlink():
            raise U2SQualificationError(f"qualification path contains a symlink: {path}")
        parent = current.parent
        if parent == current:
            break
        current = parent


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _git(
    repository: Path,
    arguments: Sequence[str],
    *,
    runner: Runner = subprocess.run,
) -> str:
    try:
        result = runner(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise U2SQualificationError(
            f"cannot verify U2-S source with git {' '.join(arguments)}"
        ) from error
    return result.stdout.rstrip("\n")


def _protected_seed_partitions() -> list[dict[str, Any]]:
    protected = [
        partition.public_dict()
        for partition in SEED_PARTITIONS
        if (
            "confirmation" in partition.role.value
            or partition.role is U2SeedRole.FINAL_TEST
        )
    ]
    if len(protected) != 9:
        raise U2SQualificationError("U2-S protected seed inventory changed")
    return protected


def _resume_rule() -> dict[str, Any]:
    return {
        "bit_exact_environment_resume": False,
        "resume_supported": False,
        "reason": (
            "continuing only one interrupted factorial arm would break the "
            "matched random-number comparison"
        ),
        "interruption_disposition": "whole_cohort_operationally_incomplete",
        "continuation_from_arm_tip": False,
        "replacement_requires_all_four_fresh_from_parent": True,
        "replacement_requires_new_source_tag_protocol_attempt_and_root": True,
    }


def _terminal_rule() -> dict[str, Any]:
    return {
        "decision_child_actions": [
            v02_u2s.CHILD_ACTION_BUDGET
            - (v02_u2s.FINAL_STABILITY_EXAMS - 1 - offset)
            * v02_u2s.EVALUATION_INTERVAL
            for offset in range(v02_u2s.FINAL_STABILITY_EXAMS)
        ],
        "all_existing_lesson_gates": True,
        "u2_successes_at_least": v02_u2s.U2_STABILITY_SUCCESSES,
        "u2_panel_successes_at_least": v02_u2s.U2_STABILITY_PANEL_SUCCESSES,
        "mean_ineffective_interactions_at_most": v02_u2s.INEFFECTIVE_MEAN_MAX,
        "cases_with_at_least_10_ineffective_interactions": 0,
        "max_repeated_identical_interaction_run_exclusive": (
            v02_u2s.MAX_IDENTICAL_INTERACTION_RUN_EXCLUSIVE
        ),
        "fixed_priority": [arm.value for arm in v02_u2s.ARM_PRIORITY],
        "selects_configuration_only": True,
        "ablation_checkpoint_may_be_successor": False,
        "none_eligible_stops": True,
    }


def build_u2s_tag_payload(
    repository: Path,
    *,
    source_commit: str,
) -> dict[str, Any]:
    """Recompute the exact one-line annotated-tag preregistration payload."""

    commit = _require_git_object(source_commit, "U2-S tag source commit")
    protocol_path = repository.expanduser().resolve() / PROTOCOL_DOCUMENT
    if protocol_path.is_symlink() or not protocol_path.is_file():
        raise U2SQualificationError("U2-S protocol document is missing or unsafe")
    (
        parent,
        base_qualification,
        failed_u2_parent,
        terminal,
        failed_r0,
    ) = _authenticate_predecessors(repository)
    parent_public = _verify_parent_model(parent)
    mapping, guards = build_u2s_forbidden_layout_hashes(
        base_qualification.verified_report(),
        failed_u2_parent.confirmation_snapshot(),
        access=base_qualification.seed_access(),
        terminal_report=terminal,
    )
    if set(mapping) != set(lessons.LessonId):
        raise U2SQualificationError("U2-S tag guard mapping is incomplete")
    protected = _protected_seed_partitions()
    payload = {
        "schema_version": TAG_SCHEMA_VERSION,
        "kind": TAG_KIND,
        "protocol": PROTOCOL,
        "tag": QUALIFIED_TAG,
        "remote": QUALIFIED_REMOTE,
        "remote_url": EXPECTED_ORIGIN_URL,
        "source_commit": commit,
        "protocol_document_sha256": file_sha256(protocol_path),
        "parent": {
            "checkpoint_sha256": parent_public["checkpoint_sha256"],
            "sidecar_sha256": parent_public["sidecar_sha256"],
            "manifest_sha256": parent_public["manifest_sha256"],
            "policy_tensor_sha256": parent_public["policy_tensor_sha256"],
            "optimizer_state_sha256": parent_public["optimizer_state_sha256"],
            "trained_timesteps": parent_public["trained_timesteps"],
            "n_updates": parent_public["n_updates"],
        },
        "base_qualification_sha256": base_qualification.report_sha256,
        "u2r_terminal_report_sha256": U2R_REPORT_SHA256,
        "failed_r0_launch_sha256": _canonical_sha256(failed_r0),
        "guard_mapping_sha256": guards["applied_mapping_sha256"],
        "arm_contract_sha256": _canonical_sha256(_arm_contract()),
        "algorithm_seed": v02_u2s.ALGORITHM_SEED,
        "worker_streams": list(v02_u2s.WORKER_STREAMS),
        "child_action_budget_per_arm": v02_u2s.CHILD_ACTION_BUDGET,
        "evaluation_interval": v02_u2s.EVALUATION_INTERVAL,
        "exam_count_per_arm": v02_u2s.EXAM_COUNT,
        "protected_seed_partitions": protected,
        "protected_seed_partitions_sha256": _canonical_sha256(protected),
        "roots": {
            "qualification": str(CANONICAL_QUALIFICATION_DIRECTORY),
            "cohort": str(CANONICAL_COHORT_ROOT),
            "media": str(CANONICAL_MEDIA_ROOT),
        },
        "dashboard_port": 8787,
        "storage_caps": _storage_caps(),
        "terminal_rule": _terminal_rule(),
        "resume_rule": _resume_rule(),
    }
    if set(payload) != TAG_FIELDS:
        raise U2SQualificationError("internal U2-S tag payload schema is incomplete")
    return payload


def verify_source_tag(
    repository: Path,
    *,
    expected_source_commit: str | None = None,
    expected_tag_object: str | None = None,
    expected_tag_payload: Mapping[str, Any] | None = None,
    require_clean: bool = True,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Authenticate one clean source commit and the published annotated tag."""

    root = repository.expanduser().resolve()
    if not root.is_dir():
        raise U2SQualificationError("U2-S repository is missing")
    head = _require_git_object(
        _git(root, ["rev-parse", "--verify", "HEAD^{commit}"], runner=runner),
        "U2-S source commit",
    )
    if expected_source_commit is not None and head != _require_git_object(
        expected_source_commit, "expected U2-S source commit"
    ):
        raise U2SQualificationError("U2-S source commit differs from qualification")
    dirty = bool(_git(root, ["status", "--porcelain=v1", "--untracked-files=all"], runner=runner))
    if require_clean and dirty:
        raise U2SQualificationError("U2-S qualification requires a clean repository")
    if _git(root, ["cat-file", "-t", f"refs/tags/{QUALIFIED_TAG}"], runner=runner) != "tag":
        raise U2SQualificationError("U2-S source tag must be annotated")
    tag_object = _require_git_object(
        _git(root, ["rev-parse", "--verify", f"refs/tags/{QUALIFIED_TAG}"], runner=runner),
        "U2-S tag object",
    )
    if expected_tag_object is not None and tag_object != _require_git_object(
        expected_tag_object, "expected U2-S tag object"
    ):
        raise U2SQualificationError("U2-S tag object differs from qualification")
    peeled = _git(
        root,
        ["rev-list", "-n", "1", f"refs/tags/{QUALIFIED_TAG}"],
        runner=runner,
    )
    if peeled != head:
        raise U2SQualificationError("U2-S annotated tag does not point to HEAD")
    remote_url = _git(
        root,
        ["remote", "get-url", QUALIFIED_REMOTE],
        runner=runner,
    )
    if remote_url != EXPECTED_ORIGIN_URL:
        raise U2SQualificationError("U2-S origin is not the frozen GitHub repository")
    remote = _git(
        root,
        [
            "ls-remote",
            "--tags",
            QUALIFIED_REMOTE,
            f"refs/tags/{QUALIFIED_TAG}",
        ],
        runner=runner,
    ).split()
    if remote != [tag_object, f"refs/tags/{QUALIFIED_TAG}"]:
        raise U2SQualificationError("U2-S tag object is not frozen on origin")
    raw_message = _git(
        root,
        [
            "for-each-ref",
            "--format=%(contents)",
            f"refs/tags/{QUALIFIED_TAG}",
        ],
        runner=runner,
    )
    if "\n" in raw_message:
        raise U2SQualificationError("U2-S annotated tag message must be one JSON line")
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError as error:
        raise U2SQualificationError("U2-S tag message is not canonical JSON") from error
    if not isinstance(payload, Mapping) or set(payload) != TAG_FIELDS:
        raise U2SQualificationError("U2-S tag payload fields changed")
    expected_payload = (
        build_u2s_tag_payload(root, source_commit=head)
        if expected_tag_payload is None
        else json.loads(json.dumps(expected_tag_payload))
    )
    if payload != expected_payload:
        raise U2SQualificationError("U2-S tag payload differs from live protocol inputs")
    canonical_message = json.dumps(
        expected_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    if raw_message != canonical_message:
        raise U2SQualificationError("U2-S tag payload is not canonically encoded")
    return {
        "commit": head,
        "dirty": dirty,
        "tag": QUALIFIED_TAG,
        "tag_object": tag_object,
        "remote": QUALIFIED_REMOTE,
        "remote_url": remote_url,
        "tag_payload": dict(payload),
        "tag_payload_sha256": _canonical_sha256(payload),
    }


def _safe_lineage_path(root: Path, relative: Any, label: str) -> Path:
    candidate = Path(str(relative))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise U2SQualificationError(f"{label} escapes the U2r cohort")
    unresolved = (root / candidate).absolute()
    _reject_symlink_chain(unresolved)
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as error:
        raise U2SQualificationError(f"{label} is missing or unsafe") from error
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise U2SQualificationError(f"{label} escapes the U2r cohort") from error
    if not resolved.is_file():
        raise U2SQualificationError(f"{label} is missing or unsafe")
    return resolved


def extend_u2s_forbidden_layout_hashes(
    base_mapping: Mapping[lessons.LessonId, frozenset[str]],
    terminal_report: Mapping[str, Any],
    *,
    run_directory: Path = U2R_RUN_DIRECTORY,
) -> tuple[Mapping[lessons.LessonId, frozenset[str]], dict[str, Any]]:
    """Extend the static same-lesson guards with authenticated U2r-r1 starts.

    U0 deliberately remains the original 79-layout development diagnostic.
    Navigate, U1, and U2 receive every r1 episode-start and terminal-active
    layout from their own lesson.
    """

    if set(base_mapping) != set(lessons.LessonId):
        raise U2SQualificationError("base exclusions do not cover every lesson")
    root = run_directory.expanduser().resolve()
    if terminal_report.get("verdict") != "failed":
        raise U2SQualificationError("U2-S requires the authenticated failed U2r report")
    lineage = terminal_report.get("lineage_history")
    if not isinstance(lineage, Mapping):
        raise U2SQualificationError("U2r report lacks lineage history")
    segments = lineage.get("segments")
    if not isinstance(segments, list) or len(segments) != 1:
        raise U2SQualificationError("U2r-r1 must contain exactly one lineage segment")
    ledger_binding = segments[0].get("episode_start_ledger")
    if not isinstance(ledger_binding, Mapping):
        raise U2SQualificationError("U2r report lacks its episode-start ledger")
    cohort_root = root.parent.resolve()
    if Path(str(lineage.get("cohort_root", ""))).resolve() != cohort_root:
        raise U2SQualificationError("U2r lineage root changed")
    ledger_path = _safe_lineage_path(
        cohort_root,
        ledger_binding.get("path"),
        "U2r episode-start ledger",
    )
    ledger_sha256 = file_sha256(ledger_path)
    if (
        ledger_sha256 != U2R_EPISODE_STARTS_SHA256
        or ledger_binding.get("sha256") != ledger_sha256
    ):
        raise U2SQualificationError("U2r episode-start ledger digest changed")

    identity_fields = (
        "active",
        "episode_ordinal",
        "episode_started_at",
        "generator_profile",
        "generator_profile_version",
        "geometry_sha256",
        "layout_sha256",
        "lesson_id",
        "lesson_label",
        "reset_provenance",
        "seed",
        "seed_role",
        "worker_index",
        "worker_stream",
        "worker_transition_at_start",
    )
    by_lesson: dict[lessons.LessonId, set[str]] = {
        lesson: set() for lesson in lessons.LessonId
    }
    started_identities: set[str] = set()
    record_count = 0
    try:
        with ledger_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise U2SQualificationError(
                        f"U2r episode-start ledger has a blank line at {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, Mapping) or value.get("type") != "episode_start":
                    raise U2SQualificationError("U2r episode-start record is invalid")
                try:
                    lesson = lessons.LessonId(str(value.get("lesson_id")))
                except ValueError as error:
                    raise U2SQualificationError(
                        "U2r episode-start record has an unknown lesson"
                    ) from error
                by_lesson[lesson].add(
                    _require_sha256(value.get("layout_sha256"), "U2r started layout")
                )
                started_identities.add(
                    _canonical_sha256(
                        {field: value.get(field) for field in identity_fields}
                    )
                )
                record_count += 1
    except (OSError, json.JSONDecodeError) as error:
        raise U2SQualificationError(f"cannot parse U2r episode starts: {error}") from error
    if (
        record_count != int(ledger_binding.get("record_count", -1))
        or record_count != int(ledger_binding.get("episode_start_count", -1))
    ):
        raise U2SQualificationError("U2r episode-start count changed")

    terminal_active = lineage.get("terminal_active_workers")
    if not isinstance(terminal_active, Mapping):
        raise U2SQualificationError("U2r report lacks terminal active-worker evidence")
    active_records = terminal_active.get("records")
    if (
        not isinstance(active_records, list)
        or len(active_records) != int(terminal_active.get("count", -1))
        or len(active_records) != frozen_u2r.WORKERS
        or terminal_active.get("records_sha256")
        != U2R_TERMINAL_ACTIVE_RECORDS_SHA256
        or _canonical_sha256(active_records)
        != U2R_TERMINAL_ACTIVE_RECORDS_SHA256
    ):
        raise U2SQualificationError("U2r terminal active-worker inventory changed")
    active_by_lesson: dict[lessons.LessonId, set[str]] = {
        lesson: set() for lesson in lessons.LessonId
    }
    for value in active_records:
        if not isinstance(value, Mapping) or value.get("active") is not True:
            raise U2SQualificationError("U2r terminal active-worker record is invalid")
        try:
            lesson = lessons.LessonId(str(value.get("lesson_id")))
        except ValueError as error:
            raise U2SQualificationError(
                "U2r terminal worker has an unknown lesson"
            ) from error
        digest = _require_sha256(
            value.get("layout_sha256"), "U2r terminal active layout"
        )
        if (
            _canonical_sha256(
                {field: value.get(field) for field in identity_fields}
            )
            not in started_identities
        ):
            raise U2SQualificationError(
                "U2r terminal active identity is absent from episode starts"
            )
        active_by_lesson[lesson].add(digest)
        by_lesson[lesson].add(digest)

    applied: dict[lessons.LessonId, frozenset[str]] = {}
    added_by_lesson: dict[str, dict[str, Any]] = {}
    for lesson in lessons.LessonId:
        values = set(base_mapping[lesson])
        if lesson is not lessons.LessonId.VISIBLE_UNLOCK:
            values.update(by_lesson[lesson])
        applied[lesson] = frozenset(values)
        added = values - set(base_mapping[lesson])
        added_by_lesson[lesson.value] = {
            "episode_start_unique_layouts": len(by_lesson[lesson]),
            "terminal_active_unique_layouts": len(active_by_lesson[lesson]),
            "new_unique_layouts": len(added),
            "new_layout_set_sha256": _hash_set_sha256(added),
            "u0_development_only_guard_unchanged": (
                lesson is lessons.LessonId.VISIBLE_UNLOCK
            ),
        }
    if applied[lessons.LessonId.VISIBLE_UNLOCK] != base_mapping[
        lessons.LessonId.VISIBLE_UNLOCK
    ]:
        raise U2SQualificationError("U2-S must not extend the U0 development guard")

    mapping = MappingProxyType(applied)
    applied_by_lesson = {
        lesson.value: {
            "exact_layouts": len(mapping[lesson]),
            "exact_layout_set_sha256": _hash_set_sha256(mapping[lesson]),
            "rule": (
                "development_only_history_overlap_diagnostic"
                if lesson is lessons.LessonId.VISIBLE_UNLOCK
                else "same_lesson_full_history_plus_u2r_r1"
            ),
        }
        for lesson in lessons.LessonId
    }
    mapping_sha256 = _canonical_sha256(
        {lesson.value: sorted(mapping[lesson]) for lesson in lessons.LessonId}
    )
    evidence = {
        "u2r_report": str(root / "report.json"),
        "u2r_report_sha256": U2R_REPORT_SHA256,
        "episode_start_ledger": str(ledger_path),
        "episode_start_ledger_sha256": ledger_sha256,
        "episode_start_records": record_count,
        "terminal_active_workers": len(active_records),
        "terminal_active_records_sha256": U2R_TERMINAL_ACTIVE_RECORDS_SHA256,
        "terminal_active_identities_in_episode_starts": True,
        "added_by_lesson": added_by_lesson,
        "applied_by_lesson": applied_by_lesson,
        "applied_mapping_sha256": mapping_sha256,
    }
    return mapping, evidence


def build_u2s_forbidden_layout_hashes(
    qualification_report: Mapping[str, Any],
    confirmation_report: Mapping[str, Any],
    *,
    access: Any,
    terminal_report: Mapping[str, Any],
    run_directory: Path = U2R_RUN_DIRECTORY,
) -> tuple[Mapping[lessons.LessonId, frozenset[str]], dict[str, Any]]:
    """Build the exact U2-S exclusions from authenticated predecessor evidence."""

    base_mapping, base_evidence = frozen_u2r.build_u2r_forbidden_layout_hashes(
        qualification_report,
        confirmation_report,
        access=access,
    )
    mapping, extension = extend_u2s_forbidden_layout_hashes(
        base_mapping,
        terminal_report,
        run_directory=run_directory,
    )
    return mapping, {
        "base": base_evidence.public_dict(),
        "u2r_r1_extension": extension,
        "applied_by_lesson": extension["applied_by_lesson"],
        "applied_mapping_sha256": extension["applied_mapping_sha256"],
    }


def _storage_caps() -> dict[str, int]:
    return {
        "per_arm_bytes": LINEAGE_CAP_BYTES,
        "scientific_cohort_bytes": COHORT_SCIENTIFIC_CAP_BYTES,
        "media_bytes": MEDIA_CAP_BYTES,
        "combined_bytes": COMBINED_PLANNED_CAP_BYTES,
    }


def _storage_preflight(*, require_fresh_roots: bool) -> dict[str, Any]:
    volume = Path("/Volumes/T7 Developer")
    dungeon_root = volume / "DungeonApprentice"
    _reject_symlink_chain(volume)
    _reject_symlink_chain(dungeon_root)
    try:
        volume_metadata = volume.stat()
        parent_metadata = volume.parent.stat()
        dungeon_metadata = dungeon_root.stat()
    except OSError as error:
        raise U2SQualificationError("cannot inspect the U2-S storage mount") from error
    if (
        not stat.S_ISDIR(volume_metadata.st_mode)
        or not stat.S_ISDIR(dungeon_metadata.st_mode)
        or not os.path.ismount(volume)
        or volume_metadata.st_dev == parent_metadata.st_dev
        or dungeon_metadata.st_dev != volume_metadata.st_dev
    ):
        raise U2SQualificationError(
            "T7 Developer is not a separate regular mounted volume"
        )
    managed_roots = (CANONICAL_QUALIFICATION_DIRECTORY, CANONICAL_COHORT_ROOT, CANONICAL_MEDIA_ROOT)
    if require_fresh_roots and any(path.exists() or path.is_symlink() for path in managed_roots):
        raise U2SQualificationError("U2-S managed roots are not fresh")
    protected = (
        v02_u2s.PARENT_CHECKPOINT,
        frozen_u2.CANONICAL_U1_CONFIRMATION,
        BASE_QUALIFICATION_REPORT,
        U2R_RUN_DIRECTORY,
    )
    for artifact in protected:
        resolved = artifact.expanduser().resolve()
        if any(resolved == root or root in resolved.parents for root in managed_roots):
            raise U2SQualificationError(
                "U2-S managed roots overlap protected predecessor evidence"
            )
    usage = shutil.disk_usage(volume)
    minimum_free_bytes = int(v02_u2s.MINIMUM_FREE_GIB * 1024**3)
    if usage.free < minimum_free_bytes:
        raise U2SQualificationError(
            "T7 Developer lacks the frozen U2-S free-space reserve"
        )
    return {
        "volume": str(volume),
        "dungeon_root": str(dungeon_root),
        "volume_device": int(volume_metadata.st_dev),
        "parent_device": int(parent_metadata.st_dev),
        "separate_mounted_device": True,
        "managed_roots": [str(path) for path in managed_roots],
        "managed_roots_absent": bool(require_fresh_roots),
        "protected_artifacts_outside_managed_roots": True,
        "minimum_free_bytes": minimum_free_bytes,
        "measured_free_bytes": int(usage.free),
        "measured_total_bytes": int(usage.total),
        "passed": True,
        "measured_at": utc_now(),
    }


def _verify_storage_preflight(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise U2SQualificationError("U2-S storage preflight is missing")
    live = _storage_preflight(require_fresh_roots=False)
    if (
        value.get("volume") != live["volume"]
        or value.get("dungeon_root") != live["dungeon_root"]
        or value.get("volume_device") != live["volume_device"]
        or value.get("parent_device") != live["parent_device"]
        or value.get("separate_mounted_device") is not True
        or value.get("managed_roots")
        != [
            str(CANONICAL_QUALIFICATION_DIRECTORY),
            str(CANONICAL_COHORT_ROOT),
            str(CANONICAL_MEDIA_ROOT),
        ]
        or value.get("managed_roots_absent") is not True
        or value.get("protected_artifacts_outside_managed_roots") is not True
        or value.get("minimum_free_bytes") != live["minimum_free_bytes"]
        or not isinstance(value.get("measured_free_bytes"), int)
        or int(value["measured_free_bytes"]) < int(value["minimum_free_bytes"])
        or not isinstance(value.get("measured_total_bytes"), int)
        or value.get("passed") is not True
    ):
        raise U2SQualificationError("U2-S storage preflight evidence changed")
    measured_at = value.get("measured_at")
    if not isinstance(measured_at, str):
        raise U2SQualificationError("U2-S storage timestamp is missing")
    try:
        timestamp = datetime.fromisoformat(measured_at)
    except ValueError as error:
        raise U2SQualificationError("U2-S storage timestamp is invalid") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(None):
        raise U2SQualificationError("U2-S storage timestamp must be UTC")


class _PenaltyProbeEnv(gym.Env[np.ndarray, int]):
    metadata: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        *,
        changed_steps: frozenset[int] = frozenset(),
        terminate_at: int | None = None,
    ) -> None:
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(2, 2, 3),
            dtype=np.uint8,
        )
        self.changed_steps = changed_steps
        self.terminate_at = terminate_at
        self.steps = 0
        self.observation = np.zeros(self.observation_space.shape, dtype=np.uint8)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.steps = 0
        self.observation.fill(0)
        return self.observation.copy(), {"ineffective_interactions": 999_999}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del action
        self.steps += 1
        if self.steps in self.changed_steps:
            self.observation = np.full_like(
                self.observation,
                self.steps % 255,
            )
        return (
            self.observation.copy(),
            0.0,
            self.terminate_at == self.steps,
            False,
            {
                "ineffective_interactions": -999_999,
                "extrinsic_reward": 999_999.0,
            },
        )


def _penalty_wrapper_smoke() -> dict[str, Any]:
    interactions = sorted(v02_u2s.INTERACTION_ACTIONS)
    selected = interactions[-1]
    different_interaction = interactions[0]
    noninteraction = next(
        action for action in range(7) if action not in v02_u2s.INTERACTION_ACTIONS
    )
    environment = v02_u2s.NoEffectInteractionPenalty(
        _PenaltyProbeEnv(
            changed_steps=frozenset({3}),
            terminate_at=9,
        )
    )
    actions = [
        selected,
        selected,
        selected,
        selected,
        selected,
        noninteraction,
        selected,
        different_interaction,
        different_interaction,
    ]
    expected_streaks = [1, 2, 0, 1, 2, 0, 1, 1, 2]
    expected_penalties = [0.0, -0.01, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0, -0.01]
    _, reset_info = environment.reset()
    observed_streaks: list[int] = []
    observed_penalties: list[float] = []
    terminated = False
    for action in actions:
        _, _, terminated, _, info = environment.step(action)
        observed_streaks.append(int(info["u2s_identical_interaction_run"]))
        observed_penalties.append(float(info["u2s_no_effect_penalty"]))
    if (
        reset_info.get("ineffective_interactions") != 999_999
        or observed_streaks != expected_streaks
        or not np.allclose(observed_penalties, expected_penalties)
        or terminated is not True
    ):
        raise U2SQualificationError("U2-S penalty truth-table smoke failed")
    environment.reset()
    _, _, _, _, reset_probe = environment.step(selected)
    if (
        reset_probe["u2s_identical_interaction_run"] != 1
        or reset_probe["u2s_no_effect_penalty"] != 0.0
    ):
        raise U2SQualificationError("U2-S penalty episode reset smoke failed")
    environment.close()

    cap_environment = v02_u2s.NoEffectInteractionPenalty(_PenaltyProbeEnv())
    cap_environment.reset()
    cap_penalties = [
        float(cap_environment.step(selected)[4]["u2s_no_effect_penalty"])
        for _ in range(20)
    ]
    cap_environment.close()
    if not math.isclose(
        sum(cap_penalties),
        -v02_u2s.NO_EFFECT_EPISODE_CAP,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or any(value != 0.0 for value in cap_penalties[11:]):
        raise U2SQualificationError("U2-S penalty cap smoke failed")
    return {
        "actions": actions,
        "expected_streaks": expected_streaks,
        "observed_streaks": observed_streaks,
        "expected_penalties": expected_penalties,
        "observed_penalties": observed_penalties,
        "episode_cap_penalties": cap_penalties,
        "episode_cap_total": sum(cap_penalties),
        "reset_passed": True,
        "different_action_reset_passed": True,
        "noninteraction_reset_passed": True,
        "pixel_change_reset_passed": True,
        "terminal_then_reset_passed": True,
        "info_independence_passed": True,
    }


def _seed_boundary_smoke(access: Any) -> dict[str, Any]:
    validation_roles = (
        U2SeedRole.NAVIGATE_VALIDATION,
        U2SeedRole.U0_VALIDATION,
        U2SeedRole.U1_VALIDATION,
        U2SeedRole.U2_VALIDATION,
    )
    opened_validation: list[str] = []
    for role in validation_roles:
        partition = PARTITION_BY_ROLE[role]
        authorize_u2_seed(
            partition.start,
            expected_role=role,
            access=access,
        )
        opened_validation.append(role.value)
    denied: list[str] = []
    for partition in SEED_PARTITIONS:
        if (
            "confirmation" not in partition.role.value
            and partition.role is not U2SeedRole.FINAL_TEST
        ):
            continue
        try:
            authorize_u2_seed(
                partition.start,
                expected_role=partition.role,
                access=access,
            )
        except U2SeedAccessError:
            denied.append(partition.role.value)
        else:
            raise U2SQualificationError(
                f"U2-S training access opened {partition.role.value}"
            )
    protected = _protected_seed_partitions()
    if len(denied) != len(protected):
        raise U2SQualificationError("U2-S protected seed denial is incomplete")
    return {
        "validation_roles_authorized": opened_validation,
        "protected_roles_denied": denied,
        "protected_partition_count": len(protected),
        "confirmation_or_final_generated": False,
        "passed": True,
    }


def _release_checks(
    base_qualification: Any,
    *,
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    conservative_terminal_bounds = v02_u2s.optimizer_update_bounds(
        v02_u2s.ArmName.CONSERVATIVE,
        v02_u2s.CHILD_ACTION_BUDGET,
        parent_updates=v02_u2s.PARENT_LIFETIME_ACTIONS
        // v02_u2s.ROLLOUT_TRANSITIONS
        * v02_u2s.ORIGINAL_EPOCHS,
    )
    control_terminal_bounds = v02_u2s.optimizer_update_bounds(
        v02_u2s.ArmName.CONTROL,
        v02_u2s.CHILD_ACTION_BUDGET,
        parent_updates=v02_u2s.PARENT_LIFETIME_ACTIONS
        // v02_u2s.ROLLOUT_TRANSITIONS
        * v02_u2s.ORIGINAL_EPOCHS,
    )
    return {
        "fresh_parent_fingerprints": {
            "arms": {
                arm.value: {
                    "checkpoint_sha256": parent["checkpoint_sha256"],
                    "policy_tensor_sha256": parent["policy_tensor_sha256"],
                    "optimizer_state_sha256": parent["optimizer_state_sha256"],
                    "trained_timesteps": parent["trained_timesteps"],
                    "n_updates": parent["n_updates"],
                }
                for arm in v02_u2s.ARM_PRIORITY
            },
            "all_equal": True,
        },
        "penalty_wrapper": _penalty_wrapper_smoke(),
        "seed_boundary": _seed_boundary_smoke(base_qualification.seed_access()),
        "optimizer_update_accounting": {
            "control_terminal_bounds": list(control_terminal_bounds),
            "conservative_terminal_bounds": list(conservative_terminal_bounds),
            "target_kl_may_stop_after_each_completed_epoch": True,
            "passed": True,
        },
        "resume_entry_point": {
            "supported": False,
            "whole_cohort_invalidation_required": True,
            "passed": True,
        },
    }


def _verify_disposable_smoke_summary(
    value: Any,
    *,
    expected_source_commit: str | None = None,
) -> None:
    """Validate the bounded policy-updating release smoke without calling it a claim."""

    from dungeon_apprentice import v02_u2s_smoke as smoke

    if not isinstance(value, Mapping):
        raise U2SQualificationError(
            "U2-S disposable integration smoke summary is missing"
        )
    report = value.get("report")
    if (
        not isinstance(report, Mapping)
        or value.get("stdout_sha256")
        != hashlib.sha256(_canonical_json_bytes(report)).hexdigest()
    ):
        raise U2SQualificationError(
            "U2-S disposable integration smoke digest changed"
        )
    expected_fields = {
        "schema_version",
        "protocol",
        "kind",
        "result",
        "source",
        "parent",
        "integration",
        "arms",
        "canonical_boundaries_before",
        "canonical_boundaries_after",
        "canonical_or_claim_policy_updates",
        "engineering_smoke_copy_updated",
        "canonical_roots_untouched",
        "disposable_artifacts_removed",
        "evaluation_performed",
        "promotion_decision_performed",
        "promotable",
        "capability_claim",
    }
    source = report.get("source")
    parent = report.get("parent")
    integration = report.get("integration")
    arms = report.get("arms")
    if (
        set(report) != expected_fields
        or report.get("schema_version") != smoke.SCHEMA_VERSION
        or report.get("protocol") != smoke.SMOKE_PROTOCOL
        or report.get("kind") != "disposable_engineering_integration_only"
        or report.get("result") != "passed"
        or not isinstance(source, Mapping)
        or source.get("dirty") is not False
        or (
            expected_source_commit is not None
            and source.get("commit") != expected_source_commit
        )
        or report.get("canonical_or_claim_policy_updates") is not False
        or report.get("engineering_smoke_copy_updated") is not True
        or report.get("canonical_roots_untouched") is not True
        or report.get("disposable_artifacts_removed") is not True
        or report.get("evaluation_performed") is not False
        or report.get("promotion_decision_performed") is not False
        or report.get("promotable") is not False
        or report.get("capability_claim") is not False
        or report.get("canonical_boundaries_before")
        != report.get("canonical_boundaries_after")
        or not isinstance(parent, Mapping)
        or parent
        != {
            "checkpoint_sha256": v02_u2s.PARENT_CHECKPOINT_SHA256,
            "policy_tensor_sha256": PARENT_POLICY_TENSOR_SHA256,
            "optimizer_state_sha256": PARENT_OPTIMIZER_STATE_SHA256,
            "trained_timesteps": v02_u2s.PARENT_LIFETIME_ACTIONS,
            "optimizer_updates": v02_u2s.PARENT_OPTIMIZER_UPDATES,
        }
        or not isinstance(integration, Mapping)
        or integration.get("workers") != v02_u2s.WORKERS
        or integration.get("rollout_steps_per_worker")
        != v02_u2s.ROLLOUT_STEPS
        or integration.get("transitions_per_arm")
        != v02_u2s.ROLLOUT_TRANSITIONS
        or integration.get("arms")
        != [arm.value for arm in v02_u2s.ARM_PRIORITY]
        or integration.get("algorithm_seed") != v02_u2s.ALGORITHM_SEED
        or integration.get("worker_streams")
        != list(v02_u2s.WORKER_STREAMS)
        or integration.get("guard_mapping_sha256")
        != v02_u2s.QUALIFIED_GUARD_MAPPING_SHA256
        or integration.get("matched_initial_model_state") is not True
        or integration.get("matched_initial_rng_identity") is not True
        or not isinstance(integration.get("initial_rng_identity"), Mapping)
        or integration.get("matched_rollout_trajectory") is not True
        or not isinstance(arms, list)
        or len(arms) != len(v02_u2s.ARM_PRIORITY)
    ):
        raise U2SQualificationError(
            "U2-S disposable integration smoke contract changed"
        )
    _require_sha256(
        integration.get("sampler_preflight_sha256"),
        "U2-S smoke sampler preflight",
    )
    common_trajectory = _require_sha256(
        integration.get("trajectory_sha256"),
        "U2-S smoke matched trajectory",
    )
    common_episode_ledger = _require_sha256(
        integration.get("episode_ledger_sha256"),
        "U2-S smoke matched episode ledger",
    )
    try:
        common_initial_rng_identity = v02_u2s.verify_initial_rng_identity(
            integration["initial_rng_identity"]
        )
    except v02_u2s.U2SProtocolError as error:
        raise U2SQualificationError(
            "U2-S smoke initial RNG identity changed"
        ) from error
    before_expected = {
        "trained_timesteps": v02_u2s.PARENT_LIFETIME_ACTIONS,
        "optimizer_updates": v02_u2s.PARENT_OPTIMIZER_UPDATES,
        "policy_tensor_sha256": PARENT_POLICY_TENSOR_SHA256,
        "optimizer_state_sha256": PARENT_OPTIMIZER_STATE_SHA256,
        "optimizer_has_state": True,
    }
    for arm_report, arm in zip(arms, v02_u2s.ARM_PRIORITY, strict=True):
        if not isinstance(arm_report, Mapping):
            raise U2SQualificationError("U2-S smoke arm evidence is invalid")
        after = arm_report.get("after")
        optimizer = arm_report.get("optimizer")
        workers = arm_report.get("workers")
        seed_evidence = arm_report.get("seed_evidence")
        reward_evidence = arm_report.get("reward_evidence")
        episode_ledger = arm_report.get("episode_ledger")
        spec = v02_u2s.ARM_SPECS[arm]
        if (
            arm_report.get("arm") != arm.value
            or arm_report.get("configuration") != spec.public_dict()
            or arm_report.get("initial_rng_identity")
            != common_initial_rng_identity
            or arm_report.get("parent_copy_sha256")
            != v02_u2s.PARENT_CHECKPOINT_SHA256
            or arm_report.get("before") != before_expected
            or not isinstance(after, Mapping)
            or after.get("trained_timesteps")
            != v02_u2s.PARENT_LIFETIME_ACTIONS
            + v02_u2s.ROLLOUT_TRANSITIONS
            or after.get("policy_tensor_sha256")
            == PARENT_POLICY_TENSOR_SHA256
            or after.get("optimizer_state_sha256")
            == PARENT_OPTIMIZER_STATE_SHA256
            or after.get("optimizer_has_state") is not True
            or arm_report.get("reloaded_matches_after") is not True
            or arm_report.get("child_actions")
            != v02_u2s.ROLLOUT_TRANSITIONS
            or arm_report.get("rollouts") != 1
            or arm_report.get("trajectory_sha256") != common_trajectory
            or arm_report.get("updated_only_in_disposable_copy") is not True
            or arm_report.get("promotable") is not False
            or arm_report.get("capability_claim") is not False
            or not isinstance(optimizer, Mapping)
            or not isinstance(workers, list)
            or len(workers) != v02_u2s.WORKERS
            or not isinstance(seed_evidence, Mapping)
            or not isinstance(reward_evidence, Mapping)
            or not isinstance(episode_ledger, Mapping)
        ):
            raise U2SQualificationError(
                f"U2-S disposable smoke arm changed: {arm.value}"
            )
        completed_epochs = int(after["optimizer_updates"]) - (
            v02_u2s.PARENT_OPTIMIZER_UPDATES
        )
        minimum, maximum = v02_u2s.optimizer_update_bounds(
            arm,
            v02_u2s.ROLLOUT_TRANSITIONS,
        )
        expected_learning_rate = (
            v02_u2s.ChildActionLinearSchedule().for_child_actions(
                v02_u2s.ROLLOUT_TRANSITIONS
            )
            if spec.conservative_ppo
            else v02_u2s.ORIGINAL_LEARNING_RATE
        )
        try:
            measured_learning_rate = float.fromhex(
                str(optimizer.get("learning_rate_hex"))
            )
            measured_approx_kl = float.fromhex(
                str(optimizer.get("approx_kl_hex"))
            )
        except ValueError as error:
            raise U2SQualificationError(
                f"U2-S smoke optimizer evidence is invalid: {arm.value}"
            ) from error
        if (
            not minimum <= completed_epochs <= maximum
            or optimizer.get("epochs_planned") != spec.n_epochs
            or optimizer.get("epochs_completed") != completed_epochs
            or optimizer.get("epochs_skipped")
            != spec.n_epochs - completed_epochs
            or optimizer.get("kl_stop_triggered")
            is not (
                spec.target_kl is not None
                and completed_epochs < spec.n_epochs
            )
            or optimizer.get("target_kl")
            != (
                float(spec.target_kl).hex()
                if spec.target_kl is not None
                else None
            )
            or optimizer.get("expected_learning_rate_hex")
            != expected_learning_rate.hex()
            or optimizer.get("learning_rate_hex")
            != expected_learning_rate.hex()
            or not math.isfinite(measured_learning_rate)
            or not math.isfinite(measured_approx_kl)
            or seed_evidence.get("training_range_only") is not True
            or seed_evidence.get("protected_seed_hits") != []
            or seed_evidence.get("confirmation_or_final_seed_generated")
            is not False
            or reward_evidence.get("penalty_enabled")
            is not spec.no_effect_penalty
            or reward_evidence.get("reward_components_reconciled") is not True
            or episode_ledger.get("normalized_sha256")
            != common_episode_ledger
            or episode_ledger.get("active_workers") != v02_u2s.WORKERS
        ):
            raise U2SQualificationError(
                f"U2-S smoke optimization/seed evidence changed: {arm.value}"
            )
        _require_sha256(
            arm_report.get("worker_evidence_sha256"),
            f"{arm.value} smoke worker evidence",
        )
        _require_sha256(
            episode_ledger.get("active_workers_sha256"),
            f"{arm.value} smoke active workers",
        )
        if int(episode_ledger.get("records", 0)) != int(
            seed_evidence.get("episode_starts", -1)
        ):
            raise U2SQualificationError(
                f"U2-S smoke episode counts changed: {arm.value}"
            )
        aggregate_rewards: dict[str, float] = {}
        for key in (
            "reward_total_hex",
            "extrinsic_total_hex",
            "curiosity_total_hex",
            "penalty_total_hex",
        ):
            try:
                aggregate_rewards[key] = float.fromhex(
                    str(reward_evidence.get(key))
                )
            except ValueError as error:
                raise U2SQualificationError(
                    f"U2-S smoke reward aggregate is invalid: {arm.value}"
                ) from error
            if not math.isfinite(aggregate_rewards[key]):
                raise U2SQualificationError(
                    f"U2-S smoke reward aggregate is non-finite: {arm.value}"
                )
        if (
            not spec.no_effect_penalty
            and (
                aggregate_rewards["penalty_total_hex"] != 0.0
                or reward_evidence.get("penalty_events") != 0
                or reward_evidence.get("maximum_episode_penalty_count")
                != 0
            )
        ):
            raise U2SQualificationError(
                f"U2-S non-penalty arm recorded shaped reward: {arm.value}"
            )
        for worker_index, worker in enumerate(workers):
            if (
                not isinstance(worker, Mapping)
                or worker.get("worker_index") != worker_index
                or worker.get("transitions") != v02_u2s.ROLLOUT_STEPS
                or sum(int(value) for value in worker.get("action_counts", {}).values())
                != v02_u2s.ROLLOUT_STEPS
                or sum(
                    int(value)
                    for value in worker.get(
                        "lesson_transition_counts", {}
                    ).values()
                )
                != v02_u2s.ROLLOUT_STEPS
                or not 0
                <= int(worker.get("maximum_episode_penalty_count", -1))
                <= v02_u2s.NO_EFFECT_MAX_APPLIED_PER_EPISODE
            ):
                raise U2SQualificationError(
                    f"U2-S smoke worker evidence changed: {arm.value}"
                )
            for key in (
                "episode_starts_sha256",
                "trajectory_sha256",
                "reward_evidence_sha256",
            ):
                _require_sha256(
                    worker.get(key),
                    f"{arm.value} worker {worker_index} {key}",
                )
            for key in (
                "reward_total_hex",
                "extrinsic_total_hex",
                "curiosity_total_hex",
                "penalty_total_hex",
            ):
                try:
                    measured = float.fromhex(str(worker.get(key)))
                except ValueError as error:
                    raise U2SQualificationError(
                        f"U2-S smoke reward evidence is invalid: {arm.value}"
                    ) from error
                if not math.isfinite(measured):
                    raise U2SQualificationError(
                        f"U2-S smoke reward evidence is non-finite: {arm.value}"
                    )
    _require_sha256(value.get("stdout_sha256"), "U2-S smoke stdout")


def _verify_disposable_smoke_output(stdout: bytes) -> dict[str, Any]:
    """Parse and summarize the exact canonical JSON emitted by the smoke."""

    try:
        value = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise U2SQualificationError(
            "U2-S disposable integration smoke emitted invalid JSON"
        ) from error
    if not isinstance(value, dict) or stdout != _canonical_json_bytes(value):
        raise U2SQualificationError(
            "U2-S disposable integration smoke output is not canonical JSON"
        )
    summary = {
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "report": value,
    }
    source = value.get("source")
    _verify_disposable_smoke_summary(
        summary,
        expected_source_commit=(
            str(source.get("commit"))
            if isinstance(source, Mapping)
            else None
        ),
    )
    return summary


def _run_external_release_checks(
    repository: Path,
    *,
    source_commit: str,
) -> dict[str, Any]:
    """Run release gates without updating a canonical or claim-bearing policy.

    The disposable integration smoke intentionally updates four temporary
    engineering copies and proves they are removed before it reports success.
    """

    commit = _require_git_object(source_commit, "U2-S release source commit")
    if (
        _git(repository, ["rev-parse", "--verify", "HEAD^{commit}"]) != commit
        or _git(
            repository,
            ["status", "--porcelain=v1", "--untracked-files=all"],
        )
    ):
        raise U2SQualificationError(
            "U2-S release checks require the clean tagged source"
        )
    records: list[dict[str, Any]] = []
    engineering_smoke: dict[str, Any] | None = None
    for name, relative_command in RELEASE_COMMANDS:
        command = [str(repository / relative_command[0]), *relative_command[1:]]
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=repository,
                check=False,
                capture_output=True,
                timeout=1800,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise U2SQualificationError(
                f"U2-S release check could not finish: {name}"
            ) from error
        duration = time.monotonic() - started
        stdout = completed.stdout
        stderr = completed.stderr
        if (
            len(stdout) > MAX_RELEASE_OUTPUT_BYTES
            or len(stderr) > MAX_RELEASE_OUTPUT_BYTES
        ):
            raise U2SQualificationError(f"U2-S release output is too large: {name}")
        record = {
            "name": name,
            "command": list(relative_command),
            "exit_status": int(completed.returncode),
            "duration_seconds": round(duration, 6),
            "stdout_bytes": len(stdout),
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stdout_tail": stdout.decode("utf-8", errors="replace")[-1000:],
            "stderr_bytes": len(stderr),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "stderr_tail": stderr.decode("utf-8", errors="replace")[-1000:],
        }
        records.append(record)
        if completed.returncode != 0:
            raise U2SQualificationError(
                f"U2-S release check failed: {name}: "
                f"{record['stdout_tail']}{record['stderr_tail']}"
            )
        if name == "u2s_disposable_integration_smoke":
            engineering_smoke = _verify_disposable_smoke_output(stdout)
    if (
        _git(repository, ["rev-parse", "--verify", "HEAD^{commit}"]) != commit
        or _git(
            repository,
            ["status", "--porcelain=v1", "--untracked-files=all"],
        )
    ):
        raise U2SQualificationError("U2-S release checks changed the tagged source")
    if engineering_smoke is None:
        raise U2SQualificationError(
            "U2-S disposable integration smoke did not run"
        )
    return {
        "source_commit": commit,
        "commands": records,
        "commands_sha256": _canonical_sha256(records),
        "canonical_or_claim_policy_updates": False,
        "engineering_smoke_copy_updated": True,
        "engineering_smoke": engineering_smoke,
        "confirmation_or_final_seed_issuer_opened": False,
        "passed": True,
        "completed_at": utc_now(),
    }


def _verify_external_release_checks(
    value: Any,
    *,
    source_commit: str,
) -> None:
    if not isinstance(value, Mapping):
        raise U2SQualificationError("U2-S external release evidence is missing")
    records = value.get("commands")
    engineering_smoke = value.get("engineering_smoke")
    if (
        value.get("source_commit") != source_commit
        or value.get("canonical_or_claim_policy_updates") is not False
        or value.get("engineering_smoke_copy_updated") is not True
        or value.get("confirmation_or_final_seed_issuer_opened") is not False
        or value.get("passed") is not True
        or not isinstance(records, list)
        or len(records) != len(RELEASE_COMMANDS)
        or value.get("commands_sha256") != _canonical_sha256(records)
    ):
        raise U2SQualificationError("U2-S external release evidence changed")
    _verify_disposable_smoke_summary(
        engineering_smoke,
        expected_source_commit=source_commit,
    )
    for record, (name, command) in zip(records, RELEASE_COMMANDS, strict=True):
        if (
            not isinstance(record, Mapping)
            or record.get("name") != name
            or record.get("command") != list(command)
            or record.get("exit_status") != 0
            or not isinstance(record.get("duration_seconds"), int | float)
            or float(record["duration_seconds"]) < 0
            or not isinstance(record.get("stdout_bytes"), int)
            or not 0 <= int(record["stdout_bytes"]) <= MAX_RELEASE_OUTPUT_BYTES
            or not isinstance(record.get("stderr_bytes"), int)
            or not 0 <= int(record["stderr_bytes"]) <= MAX_RELEASE_OUTPUT_BYTES
        ):
            raise U2SQualificationError(
                f"U2-S external release record changed: {name}"
            )
        _require_sha256(record.get("stdout_sha256"), f"{name} stdout")
        _require_sha256(record.get("stderr_sha256"), f"{name} stderr")
        if not isinstance(record.get("stdout_tail"), str) or not isinstance(
            record.get("stderr_tail"), str
        ):
            raise U2SQualificationError(f"U2-S release output changed: {name}")
        if (
            name == "u2s_disposable_integration_smoke"
            and isinstance(engineering_smoke, Mapping)
            and record.get("stdout_sha256")
            != engineering_smoke.get("stdout_sha256")
        ):
            raise U2SQualificationError(
                "U2-S smoke output and summary digests differ"
            )
    completed_at = value.get("completed_at")
    if not isinstance(completed_at, str):
        raise U2SQualificationError("U2-S release completion timestamp is missing")
    try:
        timestamp = datetime.fromisoformat(completed_at)
    except ValueError as error:
        raise U2SQualificationError(
            "U2-S release completion timestamp is invalid"
        ) from error
    if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(None):
        raise U2SQualificationError("U2-S release completion timestamp must be UTC")


def _arm_contract() -> dict[str, Any]:
    schedule = v02_u2s.ChildActionLinearSchedule()
    pickup = int(next(iter(sorted(v02_u2s.INTERACTION_ACTIONS))))
    blank = np.zeros((2, 2, 3), dtype=np.uint8)
    changed = np.ones((2, 2, 3), dtype=np.uint8)
    streak_probe = v02_u2s.visible_no_effect_streaks(
        [pickup] * 5,
        [blank, blank, blank, changed, changed],
        [blank, blank, changed, changed, changed],
    )
    expected_probe = (1, 2, 0, 1, 2)
    if streak_probe != expected_probe:
        raise U2SQualificationError("pixels-only no-effect reset probe failed")

    horizon = max(spec.max_steps for spec in lessons.LESSON_SPECS.values())
    raw_late_success = STEP_REWARD * (horizon - 1) + SUCCESS_REWARD
    raw_penalized_success = raw_late_success - v02_u2s.NO_EFFECT_EPISODE_CAP
    raw_best_failure = STEP_REWARD * horizon + CURIOSITY_BUDGET
    discounted_success = sum(
        STEP_REWARD * v02_u2s.GAMMA**step for step in range(horizon - 1)
    )
    discounted_success += SUCCESS_REWARD * v02_u2s.GAMMA ** (horizon - 1)
    discounted_penalized_success = (
        discounted_success - v02_u2s.NO_EFFECT_EPISODE_CAP
    )
    remaining = CURIOSITY_BUDGET
    discounted_best_failure = 0.0
    for step in range(horizon):
        curiosity = min(DEFAULT_CURIOSITY_SCALE, remaining)
        remaining -= curiosity
        discounted_best_failure += (
            STEP_REWARD + curiosity
        ) * v02_u2s.GAMMA**step
    if not (
        raw_penalized_success > raw_best_failure
        and discounted_penalized_success > discounted_best_failure
    ):
        raise U2SQualificationError("bounded penalty breaks reward dominance")

    public_arms = {
        arm.value: v02_u2s.ARM_SPECS[arm].public_dict()
        for arm in v02_u2s.ARM_PRIORITY
    }
    control = public_arms[v02_u2s.ArmName.CONTROL.value]
    allowed_differences = {
        v02_u2s.ArmName.CONTROL.value: set(),
        v02_u2s.ArmName.CONSERVATIVE.value: {
            "clip_range",
            "conservative_ppo",
            "learning_rate",
            "n_epochs",
            "target_kl",
        },
        v02_u2s.ArmName.NO_EFFECT.value: {
            "no_effect_penalty",
            "no_effect_reward",
        },
        v02_u2s.ArmName.COMBINED.value: {
            "clip_range",
            "conservative_ppo",
            "learning_rate",
            "n_epochs",
            "no_effect_penalty",
            "no_effect_reward",
            "target_kl",
        },
    }
    observed_differences = {
        name: sorted(
            key
            for key in set(control) | set(arm)
            if key != "name" and control.get(key) != arm.get(key)
        )
        for name, arm in public_arms.items()
    }
    if any(
        set(observed_differences[name]) != allowed
        for name, allowed in allowed_differences.items()
    ):
        raise U2SQualificationError("U2-S arm matrix changes undeclared fields")

    return {
        "arm_order": [arm.value for arm in v02_u2s.ARM_PRIORITY],
        "arms": public_arms,
        "factorial_allowlist": {
            "reference": v02_u2s.ArmName.CONTROL.value,
            "allowed_differences": {
                name: sorted(values)
                for name, values in allowed_differences.items()
            },
            "observed_differences": observed_differences,
            "passed": True,
        },
        "matched_identity": {
            "algorithm_seed": v02_u2s.ALGORITHM_SEED,
            "worker_streams": list(v02_u2s.WORKER_STREAMS),
            "child_action_budget": v02_u2s.CHILD_ACTION_BUDGET,
            "evaluation_interval": v02_u2s.EVALUATION_INTERVAL,
            "exam_count": v02_u2s.EXAM_COUNT,
            "evaluation_cases_per_lesson": v02_u2s.EVALUATION_SEED_COUNT,
            "parent_checkpoint_sha256": v02_u2s.PARENT_CHECKPOINT_SHA256,
        },
        "conservative_schedule": schedule.public_dict(),
        "pixels_only_probe": {
            "streaks": list(streak_probe),
            "expected": list(expected_probe),
            "passed": True,
        },
        "reward_dominance": {
            "horizon": horizon,
            "raw_penalized_late_success": raw_penalized_success,
            "raw_best_curiosity_failure": raw_best_failure,
            "discounted_penalized_late_success": discounted_penalized_success,
            "discounted_best_curiosity_failure": discounted_best_failure,
            "passed": True,
        },
        "selection": {
            "decision_exams": [
                v02_u2s.CHILD_ACTION_BUDGET
                - (v02_u2s.FINAL_STABILITY_EXAMS - 1 - offset)
                * v02_u2s.EVALUATION_INTERVAL
                for offset in range(v02_u2s.FINAL_STABILITY_EXAMS)
            ],
            "fixed_priority": [arm.value for arm in v02_u2s.ARM_PRIORITY],
            "selects_configuration_only": True,
            "ablation_checkpoint_may_be_successor": False,
            "none_eligible_stops": True,
        },
    }


def _verify_parent_model(parent: frozen_u2.ParentProvenance) -> dict[str, Any]:
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError as error:
        raise U2SQualificationError(
            "U2-S qualification requires the frozen training dependencies"
        ) from error
    try:
        model = RecurrentPPO.load(parent.checkpoint, device="cpu")
    except BaseException as error:
        raise U2SQualificationError("cannot deserialize the U2-S parent") from error
    policy_sha256 = state_digests.policy_tensor_sha256(model)
    optimizer_sha256 = state_digests.optimizer_state_sha256(model)
    if (
        int(model.num_timesteps) != v02_u2s.PARENT_LIFETIME_ACTIONS
        or int(model._n_updates) != parent.n_updates
        or policy_sha256 != PARENT_POLICY_TENSOR_SHA256
        or optimizer_sha256 != PARENT_OPTIMIZER_STATE_SHA256
    ):
        raise U2SQualificationError("U2-S parent model state changed")
    return {
        **parent.public_dict(),
        "sidecar_sha256": file_sha256(Path(parent.sidecar)),
        "manifest_sha256": file_sha256(Path(parent.manifest)),
        "policy_tensor_sha256": policy_sha256,
        "optimizer_state_sha256": optimizer_sha256,
    }


def _authenticate_predecessors(
    repository: Path,
) -> tuple[
    frozen_u2.ParentProvenance,
    Any,
    frozen_u2r.U2rParent,
    dict[str, Any],
    dict[str, Any],
]:
    parent = frozen_u2.verify_parent(
        v02_u2s.PARENT_CHECKPOINT,
        frozen_u2.CANONICAL_U1_CONFIRMATION,
        child_seed=v02_u2s.PARENT_CHILD_SEED,
    )
    from dungeon_apprentice.v02_u2_qualify import canonical_qualification_paths

    base_anchor = u2_qualification_anchor.verify_external_anchor(
        repository,
        expected_source_commit=frozen_u2r.SOURCE_TRAINING_COMMIT,
    )
    base_qualification = frozen_u2.verify_qualification(
        canonical_qualification_paths()["report"],
        expected_source_commit=frozen_u2r.SOURCE_TRAINING_COMMIT,
        anchor=base_anchor,
    )
    failed_u2_parent = frozen_u2r.verify_u2r_parent()
    failed_r0 = u2r_anchor.verify_failed_r0_launch()
    if (
        file_sha256(U2R_RUN_DIRECTORY / "report.json") != U2R_REPORT_SHA256
        or file_sha256(U2R_RUN_DIRECTORY / "report.integrity.json")
        != U2R_REPORT_INTEGRITY_SHA256
    ):
        raise U2SQualificationError("frozen U2r terminal report changed")
    terminal = frozen_u2r.verify_terminal_report(U2R_RUN_DIRECTORY)
    if (
        terminal.get("verdict") != "failed"
        or terminal.get("eligible_for_fresh_confirmation") is not False
        or terminal.get("terminal_pair", {}).get("reasons")
        != ["unlock/u2-separated:terminal_ineffective_at_most_3"]
    ):
        raise U2SQualificationError("U2r terminal diagnosis changed")
    return parent, base_qualification, failed_u2_parent, terminal, failed_r0


def _collect_expected(
    repository: Path,
    *,
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, Mapping[lessons.LessonId, frozenset[str]]]:
    protocol_path = repository / PROTOCOL_DOCUMENT
    if protocol_path.is_symlink() or not protocol_path.is_file():
        raise U2SQualificationError("U2-S protocol document is missing or unsafe")
    protocol_document_sha256 = file_sha256(protocol_path)
    parent, base_qualification, failed_u2_parent, terminal, failed_r0 = (
        _authenticate_predecessors(repository)
    )
    parent_public = _verify_parent_model(parent)
    if (
        parent_public["checkpoint_sha256"] != v02_u2s.PARENT_CHECKPOINT_SHA256
        or parent_public["sidecar_sha256"] != PARENT_SIDECAR_SHA256
        or parent_public["manifest_sha256"] != PARENT_MANIFEST_SHA256
    ):
        raise U2SQualificationError("U2-S parent artifact digests changed")
    seed_access = base_qualification.seed_access()
    mapping, guards = build_u2s_forbidden_layout_hashes(
        base_qualification.verified_report(),
        failed_u2_parent.confirmation_snapshot(),
        access=seed_access,
        terminal_report=terminal,
    )
    sampler = frozen_u2r.preflight_u2r_training_layout_sampler(
        mapping,
        seed_access=seed_access,
        worker_streams=v02_u2s.WORKER_STREAMS,
        max_attempts=v02_u2s.LAYOUT_RESAMPLE_ATTEMPTS,
    )
    if len(sampler) != len(lessons.LessonId) * v02_u2s.WORKERS:
        raise U2SQualificationError("U2-S sampler preflight is incomplete")
    base_public = base_qualification.public_dict()
    terminal_pair = terminal["terminal_pair"]
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "kind": KIND,
        "verdict": VERDICT,
        "source": dict(source),
        "protocol_document": {
            "path": str(PROTOCOL_DOCUMENT),
            "sha256": protocol_document_sha256,
        },
        "parent": parent_public,
        "base_qualification": base_public,
        "u2r_terminal": {
            "failed_r0_launch": failed_r0,
            "failed_r0_launch_sha256": _canonical_sha256(failed_r0),
            "run_directory": str(U2R_RUN_DIRECTORY),
            "report": str(U2R_RUN_DIRECTORY / "report.json"),
            "report_sha256": U2R_REPORT_SHA256,
            "report_integrity": str(U2R_RUN_DIRECTORY / "report.integrity.json"),
            "report_integrity_sha256": U2R_REPORT_INTEGRITY_SHA256,
            "verdict": terminal["verdict"],
            "eligible_for_fresh_confirmation": terminal[
                "eligible_for_fresh_confirmation"
            ],
            "progress": terminal["progress"],
            "terminal_pair": terminal_pair,
            "policy_updates_during_u2r_evidence_authentication": False,
        },
        "guards": guards,
        "sampler_preflight": list(sampler),
        "arm_contract": _arm_contract(),
        "resume_contract": _resume_rule(),
        "protected_partitions": _protected_seed_partitions(),
        "release_checks": _release_checks(
            base_qualification,
            parent=parent_public,
        ),
        "storage_caps": _storage_caps(),
        "restrictions": {
            "cohort_root_must_not_exist_during_qualification": str(
                CANONICAL_COHORT_ROOT
            ),
            "media_root_must_not_exist_during_qualification": str(
                CANONICAL_MEDIA_ROOT
            ),
            "arms_run_sequentially": True,
            "each_arm_starts_from_exact_parent": True,
            "resume_supported": False,
            "interruption_invalidates_whole_cohort": True,
            "confirmation_or_final_seed_roles_opened": False,
            "canonical_or_claim_policy_updates_during_qualification": False,
            "disposable_engineering_smoke_copy_updated": True,
            "ablation_checkpoint_may_be_successor": False,
            "u3_activation_authorized": False,
        },
    }
    return expected, base_qualification, mapping


@dataclass(frozen=True)
class U2SQualificationEvidence:
    report: str
    report_sha256: str
    checksum: str
    source_commit: str
    tag: str
    tag_object: str
    tag_payload_sha256: str
    protocol_document_sha256: str
    verdict: str
    guard_sets: Mapping[str, Mapping[str, Any]]
    guard_mapping_sha256: str
    u2r_terminal_active_records_sha256: str
    sampler_preflight_sha256: str
    arm_contract_sha256: str
    protected_partitions_sha256: str
    resume_contract_sha256: str
    storage_preflight_sha256: str
    initial_rng_identity: Mapping[str, Any]
    storage_caps: Mapping[str, int]
    _report_bytes: bytes = field(repr=False, compare=False)
    _base_qualification: Any = field(repr=False, compare=False)
    _forbidden_layouts: Mapping[lessons.LessonId, frozenset[str]] = field(
        repr=False,
        compare=False,
    )

    def public_dict(self) -> dict[str, Any]:
        return {
            "report": self.report,
            "report_sha256": self.report_sha256,
            "checksum": self.checksum,
            "source_commit": self.source_commit,
            "tag": self.tag,
            "tag_object": self.tag_object,
            "tag_payload_sha256": self.tag_payload_sha256,
            "protocol_document_sha256": self.protocol_document_sha256,
            "verdict": self.verdict,
            "guard_sets": {
                lesson: dict(value) for lesson, value in self.guard_sets.items()
            },
            "guard_mapping_sha256": self.guard_mapping_sha256,
            "u2r_terminal_active_records_sha256": (
                self.u2r_terminal_active_records_sha256
            ),
            "sampler_preflight_sha256": self.sampler_preflight_sha256,
            "arm_contract_sha256": self.arm_contract_sha256,
            "protected_partitions_sha256": self.protected_partitions_sha256,
            "resume_contract_sha256": self.resume_contract_sha256,
            "storage_preflight_sha256": self.storage_preflight_sha256,
            "initial_rng_identity": json.loads(
                json.dumps(dict(self.initial_rng_identity))
            ),
            "storage_caps": dict(self.storage_caps),
        }

    def verified_report(self) -> dict[str, Any]:
        value = json.loads(self._report_bytes)
        if not isinstance(value, dict):
            raise U2SQualificationError("verified U2-S report bytes changed")
        return value

    def seed_access(self) -> Any:
        return self._base_qualification.seed_access()

    def forbidden_layout_hashes(
        self,
    ) -> Mapping[lessons.LessonId, frozenset[str]]:
        return MappingProxyType(
            {
                lesson: frozenset(values)
                for lesson, values in self._forbidden_layouts.items()
            }
        )


def _assert_canonical_report(path: Path) -> Path:
    requested = path.expanduser().absolute()
    if requested != CANONICAL_REPORT:
        raise U2SQualificationError(
            f"U2-S qualification report must be {CANONICAL_REPORT}"
        )
    _reject_symlink_chain(requested)
    return requested


def _verify_checksum(checksum_path: Path, digest: str) -> None:
    try:
        checksum_bytes = _regular_file_bytes_once(
            checksum_path,
            "U2-S qualification checksum",
            maximum=256,
        )
        fields = checksum_bytes.decode("ascii").strip().split()
    except (OSError, UnicodeDecodeError) as error:
        raise U2SQualificationError("cannot read U2-S checksum") from error
    if fields != [digest, CANONICAL_REPORT.name]:
        raise U2SQualificationError("U2-S qualification checksum changed")


def verify_u2s_qualification(
    path: Path,
    *,
    expected_source_commit: str,
    expected_tag_object: str | None = None,
    repository: Path | None = None,
) -> U2SQualificationEvidence:
    """Reauthenticate the canonical report and all live predecessor evidence."""

    report_path = _assert_canonical_report(path)
    repository = (
        Path(__file__).resolve().parents[2]
        if repository is None
        else repository.expanduser().resolve()
    )
    source = verify_source_tag(
        repository,
        expected_source_commit=expected_source_commit,
        expected_tag_object=expected_tag_object,
    )
    try:
        report_bytes = _regular_file_bytes_once(
            report_path,
            "U2-S qualification report",
            maximum=MAX_REPORT_BYTES,
        )
        report = json.loads(report_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise U2SQualificationError(f"cannot read U2-S report: {error}") from error
    if not isinstance(report, dict) or set(report) != _REPORT_FIELDS:
        raise U2SQualificationError("U2-S qualification report fields changed")
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    _verify_checksum(CANONICAL_CHECKSUM, report_sha256)
    expected, base_qualification, mapping = _collect_expected(
        repository,
        source=source,
    )
    if not isinstance(report.get("created_at"), str) or not report["created_at"]:
        raise U2SQualificationError("U2-S qualification timestamp is invalid")
    try:
        created_at = datetime.fromisoformat(report["created_at"])
    except ValueError as error:
        raise U2SQualificationError("U2-S qualification timestamp is invalid") from error
    if created_at.tzinfo is None or created_at.utcoffset() != UTC.utcoffset(None):
        raise U2SQualificationError("U2-S qualification timestamp must be UTC")
    release_checks = report.get("release_checks")
    if not isinstance(release_checks, Mapping):
        raise U2SQualificationError("U2-S release checks are missing")
    static_release_checks = dict(release_checks)
    external_release_checks = static_release_checks.pop("external", None)
    if static_release_checks != expected["release_checks"]:
        raise U2SQualificationError(
            "U2-S qualification evidence changed: release_checks"
        )
    _verify_external_release_checks(
        external_release_checks,
        source_commit=str(source["commit"]),
    )
    try:
        initial_rng_identity = v02_u2s.verify_initial_rng_identity(
            external_release_checks["engineering_smoke"]["report"][
                "integration"
            ]["initial_rng_identity"]
        )
    except (KeyError, TypeError, v02_u2s.U2SProtocolError) as error:
        raise U2SQualificationError(
            "U2-S qualification has no authenticated initial RNG identity"
        ) from error
    _verify_storage_preflight(report.get("storage_preflight"))
    for key, value in expected.items():
        if key == "release_checks":
            continue
        if report.get(key) != value:
            raise U2SQualificationError(
                f"U2-S qualification evidence changed: {key}"
            )
    if expected_tag_object is not None and source["tag_object"] != expected_tag_object:
        raise U2SQualificationError("U2-S qualification tag binding changed")
    guards = report["guards"]["applied_by_lesson"]
    return U2SQualificationEvidence(
        report=str(report_path),
        report_sha256=report_sha256,
        checksum=str(CANONICAL_CHECKSUM),
        source_commit=str(source["commit"]),
        tag=QUALIFIED_TAG,
        tag_object=str(source["tag_object"]),
        tag_payload_sha256=str(source["tag_payload_sha256"]),
        protocol_document_sha256=str(
            report["protocol_document"]["sha256"]
        ),
        verdict=VERDICT,
        guard_sets=MappingProxyType(
            {str(lesson): MappingProxyType(dict(value)) for lesson, value in guards.items()}
        ),
        guard_mapping_sha256=str(report["guards"]["applied_mapping_sha256"]),
        u2r_terminal_active_records_sha256=str(
            report["guards"]["u2r_r1_extension"][
                "terminal_active_records_sha256"
            ]
        ),
        sampler_preflight_sha256=_canonical_sha256(
            report["sampler_preflight"]
        ),
        arm_contract_sha256=_canonical_sha256(report["arm_contract"]),
        protected_partitions_sha256=_canonical_sha256(
            report["protected_partitions"]
        ),
        resume_contract_sha256=_canonical_sha256(report["resume_contract"]),
        storage_preflight_sha256=_canonical_sha256(
            report["storage_preflight"]
        ),
        initial_rng_identity=MappingProxyType(initial_rng_identity),
        storage_caps=MappingProxyType(dict(report["storage_caps"])),
        _report_bytes=report_bytes,
        _base_qualification=base_qualification,
        _forbidden_layouts=mapping,
    )


def verify_u2s_qualification_report(
    path: Path,
    *,
    expected_source_commit: str,
    expected_tag_object: str | None = None,
    repository: Path | None = None,
) -> U2SQualificationEvidence:
    """Compatibility spelling for callers that name the report explicitly."""

    return verify_u2s_qualification(
        path,
        expected_source_commit=expected_source_commit,
        expected_tag_object=expected_tag_object,
        repository=repository,
    )


def collect_u2s_qualification(
    *,
    repository: Path,
    output: Path = CANONICAL_REPORT,
) -> U2SQualificationEvidence:
    """Create the one canonical qualification without touching a cohort root."""

    report_path = _assert_canonical_report(output)
    if (
        report_path.parent.exists()
        or report_path.parent.is_symlink()
        or CANONICAL_COHORT_ROOT.exists()
        or CANONICAL_COHORT_ROOT.is_symlink()
        or CANONICAL_MEDIA_ROOT.exists()
        or CANONICAL_MEDIA_ROOT.is_symlink()
    ):
        raise U2SQualificationError(
            "U2-S qualification, cohort, or media root already exists"
        )
    source = verify_source_tag(repository)
    storage_preflight = _storage_preflight(require_fresh_roots=True)
    expected, _, _ = _collect_expected(repository, source=source)
    report = dict(expected)
    report["release_checks"] = {
        **dict(expected["release_checks"]),
        "external": _run_external_release_checks(
            repository,
            source_commit=str(source["commit"]),
        ),
    }
    report["storage_preflight"] = storage_preflight
    report["created_at"] = utc_now()
    if set(report) != _REPORT_FIELDS:
        raise U2SQualificationError("internal U2-S report schema is incomplete")

    parent = report_path.parent.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = parent / (
        f".{report_path.parent.name}.qualification-{uuid.uuid4().hex}"
    )
    try:
        temporary_directory.mkdir(mode=0o700)
        report_bytes = _canonical_json_bytes(report)
        temporary_report = temporary_directory / report_path.name
        temporary_checksum = temporary_directory / CANONICAL_CHECKSUM.name
        with temporary_report.open("xb") as handle:
            handle.write(report_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        digest = hashlib.sha256(report_bytes).hexdigest()
        with temporary_checksum.open("x", encoding="ascii") as handle:
            handle.write(f"{digest}  {report_path.name}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_directory, report_path.parent)
    except BaseException:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)
        raise
    return verify_u2s_qualification(
        report_path,
        expected_source_commit=str(source["commit"]),
        expected_tag_object=str(source["tag_object"]),
        repository=repository,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", type=Path, default=CANONICAL_REPORT)
    parser.add_argument(
        "--tag-payload-only",
        action="store_true",
        help="print the canonical one-line annotated-tag message before tag creation",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = args.repository.expanduser().resolve()
    if args.tag_payload_only:
        head = _require_git_object(
            _git(repository, ["rev-parse", "--verify", "HEAD^{commit}"]),
            "U2-S source commit",
        )
        if _git(
            repository,
            ["status", "--porcelain=v1", "--untracked-files=all"],
        ):
            raise SystemExit(
                "U2-S tag payload requires a clean committed repository"
            )
        payload = build_u2s_tag_payload(repository, source_commit=head)
        print(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        )
        return
    try:
        evidence = collect_u2s_qualification(
            repository=repository,
            output=args.output,
        )
    except U2SQualificationError as error:
        raise SystemExit(f"U2-S qualification failed closed: {error}") from error
    print(json.dumps(evidence.public_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
