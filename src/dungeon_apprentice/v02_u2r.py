"""Prospective stability remediation for the one failed U2 lineage.

U2r is not a retry of the consumed U2 confirmation.  It continues the exact
policy *and optimizer* from child 20260745's immutable U2 mastery archive for
the unused 360,448 actions in that child's original ceiling.  Training,
observations, reward, lesson mix, recovery behavior, and deterministic practice
panels remain unchanged.

Selection is deliberately terminal-only.  Every 32,768-action practice exam is
archived, but only the policy at exactly 1,048,576 cumulative U2 child actions
can become the remediation candidate, and only under the frozen two-exam
stability/anti-regression rule implemented by :func:`grade_stability_pair`.
No U2 or U2r confirmation seed can be opened by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
import uuid
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any

import gymnasium as gym
import numpy as np

from dungeon_apprentice import v02_u1_confirm as state_digests
from dungeon_apprentice import v02_u2 as frozen_u2
from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice.artifacts import (
    append_jsonl,
    atomic_copy_file,
    atomic_write_json,
    create_run_directory,
    ensure_disk_space,
    file_sha256,
    git_snapshot,
    runtime_snapshot,
    utc_now,
)
from dungeon_apprentice.contracts import MINIMUM_GAMMA
from dungeon_apprentice.u2_seed_guard import U2SeedRole
from dungeon_apprentice.u2_storage import audit_u2_storage

PROTOCOL = "dungeon-apprentice-v0.2-u2r-stability-r1"
CHECKPOINT_SCHEMA_VERSION = 1
CASE_EVIDENCE_SCHEMA_VERSION = 1

SOURCE_CHILD_SEED = 20260745
SOURCE_ALGORITHM_SEED = SOURCE_CHILD_SEED
SOURCE_WORKER_STREAMS = (20260745, 20260746, 20260747, 20260748)
SOURCE_U1_INHERITED_ACTIONS = 786_432
SOURCE_CHILD_ACTIONS = 688_128
SOURCE_LIFETIME_ACTIONS = 1_474_560
SOURCE_OPTIMIZER_UPDATES = 2_880
SOURCE_ARCHIVE = Path(
    "/Volumes/T7 Developer/DungeonApprentice/u2-separated-20260723/"
    "v02-u2-seed-20260745/checkpoints/mastered-separated-unlock.zip"
)
SOURCE_ARCHIVE_SHA256 = "56dc459fb94110f41a14b2304425f572fc235c8cfc07e1152306cbf77ac33dee"
SOURCE_POLICY_TENSOR_SHA256 = "9ec1bea9c2cc8968183e2bcc1688f5f7f72a677ebefe94f0cc4dead5a02121a5"
SOURCE_POLICY_MEMBER_SHA256 = "0783c6955ff1d62b87980d7801c8fbaedcb010174d640c4ee0f07f340371b6db"
SOURCE_OPTIMIZER_STATE_SHA256 = "e821dec631c76e32ce3ac2a0fc3aac70ae55dbe4cd625ed29a11b5b35febaba2"
SOURCE_SIDECAR_SHA256 = "68cfb84ec3fbfd4204cfa9679d5b8c1fe75b17bdf26d828ed1f92510b439cc29"
SOURCE_INTEGRITY_SHA256 = "250ebf0a8fe98ed5370d20c9b55531ec5e1be33df6821fc73b670628ddfb6151"
SOURCE_MANIFEST_SHA256 = "4d37f72e2679441c1bd3eeff81c1c95a079b2f180d21ace136b5d80a1972e2ec"
SOURCE_EVALUATIONS = SOURCE_ARCHIVE.parent.parent / "evaluations.jsonl"
SOURCE_EVALUATIONS_SHA256 = "d3ef6c0a59ff9b198076571ce78dc1a5d17b667cfa8aac92948a6616de3714d8"
SOURCE_TRAINING_COMMIT = "b7b5d361b0aa2eeabedc435fa0d4b9e1ffdd09db"

SOURCE_BASELINE: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "navigate/full": MappingProxyType(
            {
                "successes": 79,
                "panel_successes": (39, 40),
                "mean_ineffective_interactions": 0.025,
            }
        ),
        "unlock/u0-visible": MappingProxyType(
            {
                "successes": 76,
                "panel_successes": (37, 39),
                "mean_ineffective_interactions": 6.55,
            }
        ),
        "unlock/u1-local": MappingProxyType(
            {
                "successes": 77,
                "panel_successes": (38, 39),
                "mean_ineffective_interactions": 4.7375,
            }
        ),
        "unlock/u2-separated": MappingProxyType(
            {
                "successes": 72,
                "panel_successes": (36, 36),
                "mean_ineffective_interactions": 6.5375,
            }
        ),
    }
)

U2_CONFIRMATION_REPORT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/confirmations/v0.2-u2-20260723/report.json"
)
U2_CONFIRMATION_REPORT_SHA256 = "7522eb8742ed567577d1f02a2a9960d698981044128e0866daa262d36aaf1c69"
U2_CONFIRMATION_ATTEMPT = U2_CONFIRMATION_REPORT.with_name("attempt.json")
U2_CONFIRMATION_ATTEMPT_SHA256 = "4487b9b45bb6f599d9212beded7ce39b0b2f655073f375a641137c1f4ea63099"
U2_CONFIRMATION_SOURCE_COMMIT = "6c266e0a51cc951a9a37d98e611049f08b1e143f"

ADDITIONAL_ACTION_BUDGET = 360_448
TERMINAL_CHILD_ACTIONS = 1_048_576
TERMINAL_LIFETIME_ACTIONS = 1_835_008
EVALUATION_INTERVAL = frozen_u2.EVALUATION_INTERVAL
ROLLOUT_STEPS = frozen_u2.ROLLOUT_STEPS
WORKERS = frozen_u2.WORKERS
ROLLOUT_TRANSITIONS = frozen_u2.ROLLOUT_TRANSITIONS
PPO_EPOCHS = frozen_u2.PPO_EPOCHS
EVALUATION_SEED_COUNT = frozen_u2.EVALUATION_SEED_COUNT
MINIMUM_FREE_GIB = frozen_u2.MINIMUM_FREE_GIB
KEEP_ALL_EXAMS = ADDITIONAL_ACTION_BUDGET // EVALUATION_INTERVAL
POLICY_KWARGS = dict(frozen_u2.POLICY_KWARGS)

REMEDIATION_ALGORITHM_SEED = 20260749
REMEDIATION_WORKER_STREAMS = (20260749, 20260750, 20260751, 20260752)
DEFAULT_RUN_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/u2r-stability-r1-20260723"
)
DEFAULT_STORAGE_ROOT = Path("/Volumes/T7 Developer")
DEFAULT_MEDIA_DIRECTORY = Path(
    "/Volumes/T7 Developer/DungeonApprentice/u2r-stability-r1-media-20260723"
)
DEFAULT_RUN_NAME = "v02-u2r-r1-seed-20260745"
SEGMENT_SEED_OFFSET = 100_000
RESUME_ABANDONMENT_REASON = (
    "the authenticated interruption-time Gym environment cannot "
    "cross a process boundary; its exact identity remains retained "
    "as provenance while the next segment starts a fresh episode"
)

INEFFECTIVE_INTERACTION_MAX = 3.0
U2_STABILITY_OVERALL_REQUIRED = 72
U2_STABILITY_PANEL_REQUIRED = 34
MAX_TERMINAL_SUCCESS_DECLINE = 2
U2R_LAYOUT_RESAMPLE_ATTEMPTS = 8_192

U2R_APPLIED_EXCLUSION_RULES: Mapping[lessons.LessonId, str] = MappingProxyType(
    {
        lessons.LessonId.NAVIGATE: "same_lesson_full_historical_inventory",
        lessons.LessonId.VISIBLE_UNLOCK: "development_only_history_overlap_diagnostic",
        lessons.LessonId.LOCAL_UNLOCK: "same_lesson_full_historical_inventory",
        lessons.LessonId.SEPARATED_UNLOCK: "same_lesson_full_historical_inventory",
    }
)


class U2rProtocolError(RuntimeError):
    """Raised when immutable U2r evidence or state fails closed."""


@dataclass(frozen=True)
class U2rParent:
    checkpoint: str
    checkpoint_sha256: str
    policy_member_sha256: str
    sidecar: str
    sidecar_sha256: str
    integrity: str
    integrity_sha256: str
    manifest: str
    manifest_sha256: str
    evaluations: str
    evaluations_sha256: str
    source_baseline: Mapping[str, Mapping[str, Any]]
    source_commit: str
    u2_child_seed: int
    inherited_trained_actions: int
    child_trained_actions: int
    trained_timesteps: int
    n_updates: int
    confirmation_report: str
    confirmation_report_sha256: str
    confirmation_attempt: str
    confirmation_attempt_sha256: str
    confirmation_verdict: str
    confirmation_source_commit: str
    confirmation_u2_score: int
    confirmation_u2_panels: tuple[int, int]
    worker_streams: tuple[int, int, int, int]
    scheduler_state: Mapping[str, Any] = field(repr=False, compare=False)
    _confirmation_snapshot: Mapping[str, Any] = field(repr=False, compare=False)

    def public_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "policy_member_sha256": self.policy_member_sha256,
            "sidecar": self.sidecar,
            "sidecar_sha256": self.sidecar_sha256,
            "integrity": self.integrity,
            "integrity_sha256": self.integrity_sha256,
            "manifest": self.manifest,
            "manifest_sha256": self.manifest_sha256,
            "evaluations": self.evaluations,
            "evaluations_sha256": self.evaluations_sha256,
            "source_baseline": {
                lesson: {
                    "successes": int(value["successes"]),
                    "panel_successes": list(value["panel_successes"]),
                    "mean_ineffective_interactions": float(value["mean_ineffective_interactions"]),
                }
                for lesson, value in self.source_baseline.items()
            },
            "source_commit": self.source_commit,
            "u2_child_seed": self.u2_child_seed,
            "inherited_trained_actions": self.inherited_trained_actions,
            "child_trained_actions": self.child_trained_actions,
            "trained_timesteps": self.trained_timesteps,
            "n_updates": self.n_updates,
            "confirmation_report": self.confirmation_report,
            "confirmation_report_sha256": self.confirmation_report_sha256,
            "confirmation_attempt": self.confirmation_attempt,
            "confirmation_attempt_sha256": self.confirmation_attempt_sha256,
            "confirmation_verdict": self.confirmation_verdict,
            "confirmation_source_commit": self.confirmation_source_commit,
            "confirmation_u2_score": self.confirmation_u2_score,
            "confirmation_u2_panels": list(self.confirmation_u2_panels),
            "worker_streams": list(self.worker_streams),
        }

    def confirmation_snapshot(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._confirmation_snapshot))


@dataclass(frozen=True)
class ForbiddenLayoutEvidence:
    exact_layouts: int
    exact_layout_set_sha256: str
    qualification_and_validation_layouts: int
    completed_history_layouts: int
    accepted_confirmation_layouts: int
    inspectable_confirmation_outcomes: int
    inspectable_confirmation_layouts: int
    prior_u1_confirmation_layouts: int
    unavailable_original_terminal_active_worker_layouts_upper_bound: int
    unavailable_original_terminal_active_worker_layouts_status: str
    prior_u1_confirmation_report: str
    prior_u1_confirmation_report_sha256: str
    prior_u1_confirmation_set_sha256: str
    history_files: tuple[dict[str, Any], ...]
    selection_journal_files: int
    selection_journal_sha256: str
    applied_by_lesson: Mapping[str, Mapping[str, Any]]
    applied_mapping_sha256: str

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["history_files"] = [dict(item) for item in self.history_files]
        value["applied_by_lesson"] = {
            str(lesson): dict(evidence)
            for lesson, evidence in self.applied_by_lesson.items()
        }
        return value


@dataclass(frozen=True)
class StabilityDecision:
    eligible: bool
    reasons: tuple[str, ...]
    checks: Mapping[str, bool]

    def public_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "checks": dict(self.checks),
        }


@dataclass
class ControllerState:
    """U2r-only decision state; legacy U2 mastery is never reused as a gate."""

    last_decision_child_actions: int | None = None
    exam_records: list[dict[str, Any]] = field(default_factory=list)
    terminal_eligible: bool | None = None
    terminal_reasons: list[str] = field(default_factory=list)
    terminal_pair: dict[str, Any] | None = None

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ControllerState:
        expected = {
            "last_decision_child_actions",
            "exam_records",
            "terminal_eligible",
            "terminal_reasons",
            "terminal_pair",
        }
        if set(value) != expected:
            raise U2rProtocolError("U2r controller fields are incomplete or unknown")
        records = value.get("exam_records")
        reasons = value.get("terminal_reasons")
        terminal_eligible = value.get("terminal_eligible")
        if (
            not isinstance(records, list)
            or not all(isinstance(item, Mapping) for item in records)
            or not isinstance(reasons, list)
            or not all(isinstance(item, str) for item in reasons)
            or terminal_eligible not in {None, True, False}
            or (
                value.get("terminal_pair") is not None
                and not isinstance(value.get("terminal_pair"), Mapping)
            )
        ):
            raise U2rProtocolError("U2r controller contains invalid decision evidence")
        last = value.get("last_decision_child_actions")
        last_int = None if last is None else int(last)
        if last_int is not None and (
            last_int <= SOURCE_CHILD_ACTIONS
            or last_int > TERMINAL_CHILD_ACTIONS
            or last_int % EVALUATION_INTERVAL
        ):
            raise U2rProtocolError("U2r last decision is not a remediation boundary")
        normalized = [dict(item) for item in records]
        boundaries = [int(item.get("child_trained_actions", -1)) for item in normalized]
        if boundaries != sorted(set(boundaries)):
            raise U2rProtocolError("U2r exam records are duplicated or unordered")
        if boundaries and (
            boundaries[0] != SOURCE_CHILD_ACTIONS + EVALUATION_INTERVAL
            or any(right - left != EVALUATION_INTERVAL for left, right in pairwise(boundaries))
            or boundaries[-1] != last_int
        ):
            raise U2rProtocolError("U2r exam history is not a contiguous prefix")
        if terminal_eligible is not None and last_int != TERMINAL_CHILD_ACTIONS:
            raise U2rProtocolError("U2r terminal verdict exists before the action ceiling")
        for record, boundary in zip(normalized, boundaries, strict=True):
            if (
                set(record)
                != {
                    "child_trained_actions",
                    "remediation_trained_actions",
                    "allocation_valid",
                    "recovery",
                    "evaluations",
                    "case_evidence",
                }
                or int(record.get("remediation_trained_actions", -1))
                != boundary - SOURCE_CHILD_ACTIONS
                or not isinstance(record.get("allocation_valid"), bool)
                or not isinstance(record.get("recovery"), bool)
                or not isinstance(record.get("evaluations"), Mapping)
                or not isinstance(record.get("case_evidence"), Mapping)
            ):
                raise U2rProtocolError("U2r exam record lacks immutable decision evidence")
        return cls(
            last_decision_child_actions=last_int,
            exam_records=normalized,
            terminal_eligible=terminal_eligible,
            terminal_reasons=list(reasons),
            terminal_pair=(
                dict(value["terminal_pair"])
                if isinstance(value.get("terminal_pair"), Mapping)
                else None
            ),
        )


@dataclass(frozen=True)
class ResumeBundle:
    checkpoint: Path
    sidecar: Mapping[str, Any]
    curriculum: lessons.CurriculumState
    controller: ControllerState
    scheduler_state: Mapping[str, Any]
    lifetime_trained: int
    child_trained: int
    remediation_trained: int
    n_updates: int
    segment_index: int
    last_completed_allocation: Mapping[str, Any] | None
    last_completed_allocation_valid: bool | None
    active_workers: tuple[Mapping[str, Any], ...]
    active_worker_evidence_source: str
    interruption_evidence: Mapping[str, Any]
    policy_tensor_sha256: str
    optimizer_state_sha256: str


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise U2rProtocolError(f"missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise U2rProtocolError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise U2rProtocolError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise U2rProtocolError(f"{label} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise U2rProtocolError(f"{label} is not a SHA-256 digest") from error
    return value


def _hash_set_sha256(values: Sequence[str] | set[str] | frozenset[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise U2rProtocolError(f"missing or unsafe {label}: {path}")
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise U2rProtocolError(f"{label} contains a blank record at line {line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise U2rProtocolError(f"{label} record {line_number} is not an object")
                records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise U2rProtocolError(f"cannot read {label}: {path}") from error
    return records


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _integrity_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".integrity.json")


def _write_integrity(checkpoint: Path, sidecar: Path) -> dict[str, Any]:
    value = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_sha256(checkpoint),
        "sidecar": sidecar.name,
        "sidecar_sha256": file_sha256(sidecar),
    }
    atomic_write_json(_integrity_path(checkpoint), value)
    return value


def _verify_integrity(checkpoint: Path, sidecar: Path) -> Mapping[str, Any]:
    value = _read_json(_integrity_path(checkpoint), "U2r integrity record")
    if (
        value.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or value.get("protocol") != PROTOCOL
        or value.get("checkpoint") != checkpoint.name
        or value.get("sidecar") != sidecar.name
        or value.get("checkpoint_sha256") != file_sha256(checkpoint)
        or value.get("sidecar_sha256") != file_sha256(sidecar)
    ):
        raise U2rProtocolError("U2r checkpoint integrity record does not match")
    return value


def _exact_path(actual: Path, expected: Path, label: str) -> Path:
    if actual.is_symlink():
        raise U2rProtocolError(f"{label} cannot be a symlink")
    resolved = actual.expanduser().resolve()
    if resolved != expected.expanduser().resolve():
        raise U2rProtocolError(f"{label} is not the frozen path: {resolved}")
    return resolved


def _matching_failed_confirmation_entry(
    report: Mapping[str, Any],
) -> Mapping[str, Any]:
    entries = [
        item
        for item in report.get("checkpoints", ())
        if isinstance(item, Mapping)
        and isinstance(item.get("checkpoint"), Mapping)
        and int(item["checkpoint"].get("child_seed", -1)) == SOURCE_CHILD_SEED
    ]
    if len(entries) != 1:
        raise U2rProtocolError("U2 confirmation lacks the exact failed child")
    entry = entries[0]
    checkpoint = entry["checkpoint"]
    evaluations = entry.get("evaluations")
    if not isinstance(evaluations, list):
        raise U2rProtocolError("failed U2 confirmation evaluations are missing")
    by_lesson = {item.get("lesson_id"): item for item in evaluations if isinstance(item, Mapping)}
    u2_result = by_lesson.get(lessons.LessonId.SEPARATED_UNLOCK.value)
    if not isinstance(u2_result, Mapping):
        raise U2rProtocolError("failed U2 confirmation has no U2 result")
    no_update = (
        entry.get("policy_updates") is False
        and entry.get("passed") is False
        and checkpoint.get("checkpoint") == str(SOURCE_ARCHIVE)
        and checkpoint.get("checkpoint_sha256") == SOURCE_ARCHIVE_SHA256
        and int(checkpoint.get("child_trained_actions", -1)) == SOURCE_CHILD_ACTIONS
        and int(checkpoint.get("lifetime_trained_actions", -1)) == SOURCE_LIFETIME_ACTIONS
        and int(checkpoint.get("optimizer_updates", -1)) == SOURCE_OPTIMIZER_UPDATES
        and entry.get("policy_tensor_sha256_before")
        == entry.get("policy_tensor_sha256_after")
        == SOURCE_POLICY_TENSOR_SHA256
        and entry.get("optimizer_state_sha256_before")
        == entry.get("optimizer_state_sha256_after")
        == SOURCE_OPTIMIZER_STATE_SHA256
        and entry.get("model_num_timesteps_before")
        == entry.get("model_num_timesteps_after")
        == SOURCE_LIFETIME_ACTIONS
        and entry.get("model_updates_before")
        == entry.get("model_updates_after")
        == SOURCE_OPTIMIZER_UPDATES
        and entry.get("artifact_sha256s_before") == entry.get("artifact_sha256s_after")
        and int(u2_result.get("successes", -1)) == 169
        and list(u2_result.get("panel_successes", ())) == [84, 85]
        and isinstance(u2_result.get("gate"), Mapping)
        and u2_result["gate"].get("passed") is False
    )
    if not no_update:
        raise U2rProtocolError(
            "failed U2 confirmation does not preserve the frozen no-update identity"
        )
    return entry


def verify_u2r_parent(
    checkpoint: Path = SOURCE_ARCHIVE,
    confirmation_report: Path = U2_CONFIRMATION_REPORT,
) -> U2rParent:
    """Authenticate the sole U2r policy/optimizer parent and terminal failure."""

    archive = _exact_path(checkpoint, SOURCE_ARCHIVE, "U2r parent archive")
    sidecar_path = archive.with_suffix(".json")
    integrity_path = archive.with_suffix(".integrity.json")
    manifest_path = archive.parent.parent / "manifest.json"
    expected_files = {
        archive: SOURCE_ARCHIVE_SHA256,
        sidecar_path: SOURCE_SIDECAR_SHA256,
        integrity_path: SOURCE_INTEGRITY_SHA256,
        manifest_path: SOURCE_MANIFEST_SHA256,
        SOURCE_EVALUATIONS: SOURCE_EVALUATIONS_SHA256,
    }
    for path, digest in expected_files.items():
        if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
            raise U2rProtocolError(f"frozen U2r parent evidence changed: {path}")
    sidecar = _read_json(sidecar_path, "U2 mastery sidecar")
    integrity = _read_json(integrity_path, "U2 mastery integrity")
    manifest = _read_json(manifest_path, "U2 run manifest")
    progress = sidecar.get("progress")
    source = sidecar.get("source")
    parent = sidecar.get("parent")
    if (
        sidecar.get("schema_version") != frozen_u2.CHECKPOINT_SCHEMA_VERSION
        or sidecar.get("protocol") != frozen_u2.PROTOCOL
        or sidecar.get("kind") != "mastery"
        or sidecar.get("resume_eligible") is not False
        or sidecar.get("checkpoint_sha256") != SOURCE_ARCHIVE_SHA256
        or not isinstance(progress, Mapping)
        or int(progress.get("child_trained_actions", -1)) != SOURCE_CHILD_ACTIONS
        or int(progress.get("lifetime_trained_actions", -1)) != SOURCE_LIFETIME_ACTIONS
        or int(progress.get("inherited_trained_actions", -1)) != SOURCE_U1_INHERITED_ACTIONS
        or int(progress.get("optimizer_updates", -1)) != SOURCE_OPTIMIZER_UPDATES
        or not isinstance(source, Mapping)
        or source.get("dirty") is not False
        or source.get("commit") != SOURCE_TRAINING_COMMIT
        or not isinstance(parent, Mapping)
        or int(parent.get("u2_child_seed", -1)) != SOURCE_CHILD_SEED
        or sidecar.get("curriculum", {}).get("mastered") is not True
        or sidecar.get("controller", {}).get("mastery_artifact", {}).get("checkpoint_sha256")
        != SOURCE_ARCHIVE_SHA256
    ):
        raise U2rProtocolError("U2r parent sidecar is not the frozen mastery artifact")
    if (
        integrity.get("schema_version") != frozen_u2.CHECKPOINT_SCHEMA_VERSION
        or integrity.get("protocol") != frozen_u2.PROTOCOL
        or integrity.get("checkpoint") != archive.name
        or integrity.get("checkpoint_sha256") != SOURCE_ARCHIVE_SHA256
        or integrity.get("sidecar") != sidecar_path.name
        or integrity.get("sidecar_sha256") != SOURCE_SIDECAR_SHA256
        or manifest.get("protocol") != frozen_u2.PROTOCOL
        or manifest.get("source") != source
        or manifest.get("effective_config") != sidecar.get("effective_config")
    ):
        raise U2rProtocolError("U2r parent integrity/manifest binding changed")
    try:
        with zipfile.ZipFile(archive) as zipped:
            names = zipped.namelist()
            if names.count("policy.optimizer.pth") != 1:
                raise U2rProtocolError("U2r parent archive has no optimizer state")
            if (
                names.count("policy.pth") != 1
                or hashlib.sha256(zipped.read("policy.pth")).hexdigest()
                != SOURCE_POLICY_MEMBER_SHA256
            ):
                raise U2rProtocolError("U2r parent serialized policy member changed")
    except zipfile.BadZipFile as error:
        raise U2rProtocolError("U2r parent is not a readable PPO archive") from error

    report_path = _exact_path(
        confirmation_report,
        U2_CONFIRMATION_REPORT,
        "U2 confirmation report",
    )
    if file_sha256(report_path) != U2_CONFIRMATION_REPORT_SHA256:
        raise U2rProtocolError("terminal U2 confirmation report changed")
    report = _read_json(report_path, "terminal U2 confirmation report")
    report_source = report.get("source")
    report_source_after = report.get("source_after")
    if (
        report.get("schema_version") != 1
        or report.get("protocol") != "dungeon-apprentice-v0.2-u2-confirmation"
        or report.get("verdict") != "capability_failed"
        or report.get("policy_updates") is not False
        or report.get("checkpoint_scoring_performed") is not True
        or not isinstance(report_source, Mapping)
        or report_source.get("dirty") is not False
        or report_source.get("commit") != U2_CONFIRMATION_SOURCE_COMMIT
        or report_source_after != report_source
    ):
        raise U2rProtocolError("terminal U2 confirmation is not the frozen failure")
    matching_entry = _matching_failed_confirmation_entry(report)
    checkpoint_artifacts = matching_entry["checkpoint"].get("artifact_sha256s")
    if (
        not isinstance(checkpoint_artifacts, Mapping)
        or checkpoint_artifacts.get("evaluations") != SOURCE_EVALUATIONS_SHA256
    ):
        raise U2rProtocolError("U2 confirmation does not bind source evaluations")
    baseline_records = [
        item
        for item in _safe_jsonl(SOURCE_EVALUATIONS, "source U2 evaluations")
        if int(item.get("child_trained_actions", -1)) == SOURCE_CHILD_ACTIONS
        and item.get("counts_toward_gate") is True
    ]
    if len(baseline_records) != len(lessons.LessonId) or {
        item.get("lesson_id") for item in baseline_records
    } != set(SOURCE_BASELINE):
        raise U2rProtocolError("source U2 boundary lacks a complete exact baseline")
    for item in baseline_records:
        expected = SOURCE_BASELINE[str(item["lesson_id"])]
        if (
            item.get("checkpoint_sha256") != SOURCE_ARCHIVE_SHA256
            or int(item.get("successes", -1)) != int(expected["successes"])
            or tuple(int(value) for value in item.get("panel_successes", ()))
            != tuple(expected["panel_successes"])
            or not math.isclose(
                float(item.get("mean_ineffective_interactions", math.nan)),
                float(expected["mean_ineffective_interactions"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise U2rProtocolError(f"source U2 baseline changed for {item.get('lesson_id')}")

    attempt_path = _exact_path(
        U2_CONFIRMATION_ATTEMPT,
        U2_CONFIRMATION_ATTEMPT,
        "U2 confirmation attempt",
    )
    if file_sha256(attempt_path) != U2_CONFIRMATION_ATTEMPT_SHA256:
        raise U2rProtocolError("terminal U2 confirmation attempt changed")
    attempt = _read_json(attempt_path, "terminal U2 confirmation attempt")
    if (
        attempt.get("status") != "completed_with_report"
        or attempt.get("report_present") is not True
        or attempt.get("report_verdict") != "capability_failed"
        or attempt.get("report_sha256") != U2_CONFIRMATION_REPORT_SHA256
        or attempt.get("source_commit") != U2_CONFIRMATION_SOURCE_COMMIT
        or attempt.get("source_dirty") is not False
        or attempt.get("evaluator_exit_status") != 1
    ):
        raise U2rProtocolError("terminal U2 confirmation attempt is incomplete")

    return U2rParent(
        checkpoint=str(archive),
        checkpoint_sha256=SOURCE_ARCHIVE_SHA256,
        policy_member_sha256=SOURCE_POLICY_MEMBER_SHA256,
        sidecar=str(sidecar_path),
        sidecar_sha256=SOURCE_SIDECAR_SHA256,
        integrity=str(integrity_path),
        integrity_sha256=SOURCE_INTEGRITY_SHA256,
        manifest=str(manifest_path),
        manifest_sha256=SOURCE_MANIFEST_SHA256,
        evaluations=str(SOURCE_EVALUATIONS),
        evaluations_sha256=SOURCE_EVALUATIONS_SHA256,
        source_baseline=SOURCE_BASELINE,
        source_commit=SOURCE_TRAINING_COMMIT,
        u2_child_seed=SOURCE_CHILD_SEED,
        inherited_trained_actions=SOURCE_U1_INHERITED_ACTIONS,
        child_trained_actions=SOURCE_CHILD_ACTIONS,
        trained_timesteps=SOURCE_LIFETIME_ACTIONS,
        n_updates=SOURCE_OPTIMIZER_UPDATES,
        confirmation_report=str(report_path),
        confirmation_report_sha256=U2_CONFIRMATION_REPORT_SHA256,
        confirmation_attempt=str(attempt_path),
        confirmation_attempt_sha256=U2_CONFIRMATION_ATTEMPT_SHA256,
        confirmation_verdict="capability_failed",
        confirmation_source_commit=U2_CONFIRMATION_SOURCE_COMMIT,
        confirmation_u2_score=169,
        confirmation_u2_panels=(84, 85),
        worker_streams=SOURCE_WORKER_STREAMS,
        scheduler_state=dict(sidecar["scheduler"]),
        _confirmation_snapshot=report,
    )


def _safe_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise U2rProtocolError(f"missing or unsafe {label}: {path}")
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise U2rProtocolError(f"{label} line {line_number} is not an object")
                records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise U2rProtocolError(f"cannot read {label}: {path}") from error
    return records


def _completed_history_hashes(
    confirmation_report: Mapping[str, Any],
) -> tuple[
    Mapping[lessons.LessonId, frozenset[str]],
    tuple[dict[str, Any], ...],
]:
    references = confirmation_report.get("reference_exclusions")
    histories = references.get("u2_child_histories") if isinstance(references, Mapping) else None
    if not isinstance(histories, Mapping) or set(histories) != {
        "20260737",
        "20260741",
        "20260745",
    }:
        raise U2rProtocolError("U2 confirmation history evidence is incomplete")
    combined: dict[lessons.LessonId, set[str]] = {
        lesson: set() for lesson in lessons.LessonId
    }
    evidence: list[dict[str, Any]] = []
    for child in ("20260737", "20260741", "20260745"):
        item = histories[child]
        if not isinstance(item, Mapping):
            raise U2rProtocolError(f"U2 history evidence {child} is invalid")
        path = Path(str(item.get("path", ""))).expanduser().resolve()
        expected_digest = _require_sha256(item.get("file_sha256"), f"U2 history {child}")
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected_digest:
            raise U2rProtocolError(f"U2 completed history changed: {path}")
        per_lesson: dict[lessons.LessonId, set[str]] = {
            lesson: set() for lesson in lessons.LessonId
        }
        for record in _safe_jsonl(path, f"U2 history {child}"):
            try:
                lesson = lessons.LessonId(str(record["lesson_id"]))
            except (KeyError, ValueError) as error:
                raise U2rProtocolError(f"U2 history {child} contains an invalid lesson") from error
            digest = _require_sha256(record.get("layout_sha256"), f"U2 history {child} layout")
            per_lesson[lesson].add(digest)
            combined[lesson].add(digest)
        recorded_lessons = item.get("lessons")
        if not isinstance(recorded_lessons, Mapping):
            raise U2rProtocolError(f"U2 history {child} lacks lesson evidence")
        for lesson, values in per_lesson.items():
            expected = recorded_lessons.get(lesson.value)
            if (
                not isinstance(expected, Mapping)
                or int(expected.get("unique_layouts", -1)) != len(values)
                or expected.get("set_sha256") != _hash_set_sha256(values)
            ):
                raise U2rProtocolError(f"U2 history {child}/{lesson.value} set identity changed")
        evidence.append(
            {
                "child_seed": int(child),
                "path": str(path),
                "sha256": expected_digest,
                "completed_exact_layouts": len(set().union(*per_lesson.values())),
            }
        )
    return (
        MappingProxyType(
            {
                lesson: frozenset(values)
                for lesson, values in combined.items()
            }
        ),
        tuple(evidence),
    )


def _accepted_confirmation_hashes(
    confirmation_report: Mapping[str, Any],
) -> tuple[
    Mapping[lessons.LessonId, frozenset[str]],
    Mapping[lessons.LessonId, frozenset[str]],
    int,
    int,
    str,
]:
    journal = confirmation_report.get("selection_journal")
    if not isinstance(journal, Mapping):
        raise U2rProtocolError("U2 confirmation has no selection journal identity")
    identities = journal.get("identities")
    expected_file_count = int(journal.get("files", -1))
    expected_set_digest = _require_sha256(journal.get("file_set_sha256"), "U2 selection journal")
    if (
        not isinstance(identities, list)
        or len(identities) != expected_file_count
        or expected_file_count != 1_643
    ):
        raise U2rProtocolError("U2 selection journal file inventory changed")
    root = U2_CONFIRMATION_REPORT.parent / "selection-journal"
    measured_identities: list[dict[str, Any]] = []
    accepted: dict[lessons.LessonId, set[str]] = {lesson: set() for lesson in lessons.LessonId}
    inspectable: dict[lessons.LessonId, set[str]] = {
        lesson: set() for lesson in lessons.LessonId
    }
    inspectable_outcomes = 0
    for raw in identities:
        if not isinstance(raw, Mapping):
            raise U2rProtocolError("U2 selection journal identity is invalid")
        relative = Path(str(raw.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise U2rProtocolError("U2 selection journal path escapes its root")
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise U2rProtocolError("U2 selection journal path escapes its root") from error
        expected_digest = _require_sha256(raw.get("sha256"), "U2 selection journal file")
        expected_bytes = int(raw.get("byte_length", -1))
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != expected_bytes
            or file_sha256(path) != expected_digest
        ):
            raise U2rProtocolError(f"U2 selection journal file changed: {path}")
        identity = {
            "path": str(relative),
            "sha256": expected_digest,
            "byte_length": expected_bytes,
        }
        measured_identities.append(identity)
        if relative.name.endswith("-outcome.json"):
            outcome = _read_json(path, "U2 selection outcome")
            layout_digest = outcome.get("layout_sha256")
            if layout_digest is not None:
                try:
                    outcome_lesson = lessons.LessonId(str(outcome["lesson_id"]))
                except (KeyError, ValueError) as error:
                    raise U2rProtocolError(
                        "inspectable U2 confirmation outcome has an invalid lesson"
                    ) from error
                inspectable_outcomes += 1
                inspectable[outcome_lesson].add(
                    _require_sha256(
                        layout_digest,
                        "inspectable U2 confirmation layout",
                    )
                )
            if outcome.get("status") == "accepted":
                try:
                    lesson = lessons.LessonId(str(outcome["lesson_id"]))
                except (KeyError, ValueError) as error:
                    raise U2rProtocolError(
                        "accepted U2 confirmation outcome has an invalid lesson"
                    ) from error
                accepted[lesson].add(
                    _require_sha256(layout_digest, "accepted U2 confirmation layout")
                )
    if _canonical_sha256(measured_identities) != expected_set_digest:
        raise U2rProtocolError("U2 selection journal file-set identity changed")
    identity_recheck = confirmation_report.get("selected_layout_identity_recheck")
    if not isinstance(identity_recheck, Mapping):
        raise U2rProtocolError("U2 confirmation accepted-set recheck is missing")
    for lesson, values in accepted.items():
        item = identity_recheck.get(lesson.value)
        if (
            len(values) != 200
            or not isinstance(item, Mapping)
            or item.get("passed") is not True
            or int(item.get("cases", -1)) != 200
        ):
            raise U2rProtocolError(f"accepted U2 confirmation set changed for {lesson.value}")
    accepted_union = set().union(*accepted.values())
    inspectable_union = set().union(*inspectable.values())
    if (
        inspectable_outcomes != 819
        or len(inspectable_union) < len(accepted_union)
        or not accepted_union <= inspectable_union
    ):
        raise U2rProtocolError("U2 confirmation inspectable set lost accepted layouts")
    return (
        MappingProxyType(
            {
                lesson: frozenset(values)
                for lesson, values in accepted.items()
            }
        ),
        MappingProxyType(
            {
                lesson: frozenset(values)
                for lesson, values in inspectable.items()
            }
        ),
        inspectable_outcomes,
        expected_file_count,
        expected_set_digest,
    )


def _prior_u1_confirmation_hashes(
    confirmation_report: Mapping[str, Any],
) -> tuple[Mapping[lessons.LessonId, frozenset[str]], dict[str, Any]]:
    references = confirmation_report.get("reference_exclusions")
    reference = references.get("prior_u1_confirmation") if isinstance(references, Mapping) else None
    if not isinstance(reference, Mapping):
        raise U2rProtocolError("U2 report lacks prior U1 confirmation evidence")
    path = Path(str(reference.get("path", ""))).expanduser().resolve()
    expected_digest = _require_sha256(reference.get("sha256"), "prior U1 confirmation report")
    if (
        expected_digest != frozen_u2.EXPECTED_U1_CONFIRMATION_SHA256
        or path != frozen_u2.CANONICAL_U1_CONFIRMATION.expanduser().resolve()
        or path.is_symlink()
        or not path.is_file()
        or file_sha256(path) != expected_digest
    ):
        raise U2rProtocolError("frozen prior U1 confirmation report changed")
    report = _read_json(path, "prior U1 confirmation report")
    if (
        report.get("protocol") != "dungeon-apprentice-v0.2-u1-confirmation-v2"
        or report.get("verdict") != "confirmed"
        or report.get("checkpoint_scoring_performed") is not True
        or report.get("policy_updates") is not False
    ):
        raise U2rProtocolError("prior U1 confirmation is not the frozen result")
    expected_lessons = {
        lessons.LessonId.NAVIGATE,
        lessons.LessonId.VISIBLE_UNLOCK,
        lessons.LessonId.LOCAL_UNLOCK,
    }
    per_lesson: dict[lessons.LessonId, set[str]] = {}
    for selection in report.get("layout_qualification", ()):
        if not isinstance(selection, Mapping):
            raise U2rProtocolError("prior U1 confirmation selection is invalid")
        try:
            lesson = lessons.LessonId(str(selection["lesson_id"]))
        except (KeyError, ValueError) as error:
            raise U2rProtocolError("prior U1 confirmation lesson is invalid") from error
        cases = selection.get("cases")
        if (
            lesson not in expected_lessons
            or selection.get("result") != "passed"
            or not isinstance(cases, list)
            or len(cases) != 200
            or not all(isinstance(case, Mapping) for case in cases)
        ):
            raise U2rProtocolError("prior U1 confirmation panel is incomplete")
        per_lesson[lesson] = {
            _require_sha256(case.get("layout_sha256"), "prior U1 confirmation layout")
            for case in cases
        }
    if set(per_lesson) != expected_lessons:
        raise U2rProtocolError("prior U1 confirmation lacks all three lessons")
    recorded_lessons = reference.get("lessons")
    if not isinstance(recorded_lessons, Mapping):
        raise U2rProtocolError("U2 report lacks prior U1 set identities")
    for lesson, values in per_lesson.items():
        recorded = recorded_lessons.get(lesson.value)
        if (
            len(values) != 200
            or not isinstance(recorded, Mapping)
            or int(recorded.get("unique_layouts", -1)) != len(values)
            or recorded.get("set_sha256") != _hash_set_sha256(values)
        ):
            raise U2rProtocolError(f"prior U1 confirmation set changed for {lesson.value}")
    combined = set().union(*per_lesson.values())
    if len(combined) != 600:
        raise U2rProtocolError("prior U1 confirmation exact layouts are not unique")
    return (
        MappingProxyType(
            {
                lesson: frozenset(values)
                for lesson, values in per_lesson.items()
            }
        ),
        {
            "path": str(path),
            "sha256": expected_digest,
            "exact_layouts": len(combined),
            "set_sha256": _hash_set_sha256(combined),
        },
    )


def _qualification_and_validation_hashes_by_lesson(
    qualification_report: Mapping[str, Any],
    *,
    access: Any,
) -> tuple[Mapping[lessons.LessonId, frozenset[str]], frozenset[str]]:
    """Recover the lesson ownership hidden by the frozen U2 global union.

    The original U2 helper deliberately returned the same global union for all
    four lessons.  U2r-r1 retains that union as immutable historical inventory,
    while reconstructing which lesson owns each development hash so the finite
    U0 generator is not conditioned on its nearly exhaustive training history.
    """

    global_mapping = lessons.reserved_training_layout_hashes(
        qualification_report,
        access=access,
    )
    global_union = frozenset().union(*global_mapping.values())
    per_lesson: dict[lessons.LessonId, set[str]] = {
        lesson: set() for lesson in lessons.LessonId
    }
    cases = qualification_report.get("cases")
    if not isinstance(cases, list) or len(cases) != lessons.SEALED_QUALIFICATION_SEED_COUNT:
        raise U2rProtocolError("sealed U2 qualification cases changed")
    for case in cases:
        if not isinstance(case, Mapping):
            raise U2rProtocolError("sealed U2 qualification contains an invalid case")
        per_lesson[lessons.LessonId.SEPARATED_UNLOCK].add(
            _require_sha256(
                case.get("layout_sha256"),
                "sealed U2 qualification layout",
            )
        )

    validation_roles = {
        lessons.LessonId.NAVIGATE: U2SeedRole.NAVIGATE_VALIDATION,
        lessons.LessonId.VISIBLE_UNLOCK: U2SeedRole.U0_VALIDATION,
        lessons.LessonId.LOCAL_UNLOCK: U2SeedRole.U1_VALIDATION,
        lessons.LessonId.SEPARATED_UNLOCK: U2SeedRole.U2_VALIDATION,
    }
    for lesson in lessons.LessonId:
        role = validation_roles[lesson]
        for seed in lessons.validation_seeds(lesson, access=access):
            environment = lessons.U2LessonEnv(lesson=lesson)
            try:
                environment.reset(
                    seed=seed,
                    options={
                        "u2_seed_role": role.value,
                        "u2_seed_access": access,
                    },
                )
                if environment.layout is None:
                    raise U2rProtocolError(
                        f"{lesson.value} validation generated no exact layout"
                    )
                per_lesson[lesson].add(environment.layout.layout_sha256)
            finally:
                environment.close()

    measured_union = frozenset().union(*per_lesson.values())
    if measured_union != global_union:
        raise U2rProtocolError(
            "lesson-owned qualification/validation hashes differ from the frozen U2 union"
        )
    return (
        MappingProxyType(
            {
                lesson: frozenset(values)
                for lesson, values in per_lesson.items()
            }
        ),
        global_union,
    )


def build_u2r_forbidden_layout_hashes(
    qualification_report: Mapping[str, Any],
    confirmation_report: Mapping[str, Any],
    *,
    access: Any,
) -> tuple[Mapping[lessons.LessonId, frozenset[str]], ForbiddenLayoutEvidence]:
    """Build the exact training exclusions before any U2r environment exists."""

    if (
        confirmation_report.get("verdict") != "capability_failed"
        or confirmation_report.get("policy_updates") is not False
    ):
        raise U2rProtocolError("U2r exclusions require the terminal U2 report")
    base_by_lesson, base_union = _qualification_and_validation_hashes_by_lesson(
        qualification_report,
        access=access,
    )
    history_by_lesson, history_files = _completed_history_hashes(confirmation_report)
    (
        confirmed_by_lesson,
        inspectable_by_lesson,
        inspectable_confirmation_outcomes,
        journal_files,
        journal_digest,
    ) = _accepted_confirmation_hashes(confirmation_report)
    prior_u1_by_lesson, prior_u1_evidence = _prior_u1_confirmation_hashes(
        confirmation_report
    )
    histories = frozenset().union(*history_by_lesson.values())
    confirmed = frozenset().union(*confirmed_by_lesson.values())
    inspectable_confirmation = frozenset().union(*inspectable_by_lesson.values())
    prior_u1 = frozenset().union(*prior_u1_by_lesson.values())
    inventory = frozenset(
        set(base_union) | set(histories) | set(inspectable_confirmation) | set(prior_u1)
    )

    applied: dict[lessons.LessonId, frozenset[str]] = {}
    for lesson in lessons.LessonId:
        values = set(base_by_lesson[lesson])
        if lesson is not lessons.LessonId.VISIBLE_UNLOCK:
            values.update(history_by_lesson[lesson])
            values.update(inspectable_by_lesson[lesson])
            values.update(prior_u1_by_lesson.get(lesson, frozenset()))
        applied[lesson] = frozenset(values)
    mapping = MappingProxyType(applied)
    applied_public = {
        lesson.value: {
            "exact_layouts": len(mapping[lesson]),
            "exact_layout_set_sha256": _hash_set_sha256(mapping[lesson]),
            "rule": U2R_APPLIED_EXCLUSION_RULES[lesson],
        }
        for lesson in lessons.LessonId
    }
    applied_mapping_sha256 = _canonical_sha256(
        {
            lesson.value: sorted(mapping[lesson])
            for lesson in lessons.LessonId
        }
    )
    evidence = ForbiddenLayoutEvidence(
        exact_layouts=len(inventory),
        exact_layout_set_sha256=_hash_set_sha256(inventory),
        qualification_and_validation_layouts=len(base_union),
        completed_history_layouts=len(histories),
        accepted_confirmation_layouts=len(confirmed),
        inspectable_confirmation_outcomes=inspectable_confirmation_outcomes,
        inspectable_confirmation_layouts=len(inspectable_confirmation),
        prior_u1_confirmation_layouts=len(prior_u1),
        unavailable_original_terminal_active_worker_layouts_upper_bound=12,
        unavailable_original_terminal_active_worker_layouts_status=(
            "unauthenticated_unavailable_not_in_static_exclusion_set"
        ),
        prior_u1_confirmation_report=str(prior_u1_evidence["path"]),
        prior_u1_confirmation_report_sha256=str(prior_u1_evidence["sha256"]),
        prior_u1_confirmation_set_sha256=str(prior_u1_evidence["set_sha256"]),
        history_files=history_files,
        selection_journal_files=journal_files,
        selection_journal_sha256=journal_digest,
        applied_by_lesson=applied_public,
        applied_mapping_sha256=applied_mapping_sha256,
    )
    return mapping, evidence


def preflight_u2r_training_layout_sampler(
    forbidden_layout_hashes: Mapping[lessons.LessonId, frozenset[str]],
    *,
    seed_access: Any,
    worker_streams: Sequence[int] = REMEDIATION_WORKER_STREAMS,
    max_attempts: int = U2R_LAYOUT_RESAMPLE_ATTEMPTS,
) -> tuple[dict[str, Any], ...]:
    """Prove each fixed worker can sample every lesson before run creation.

    Each lesson/worker pair starts from that worker's declared RNG stream.  The
    accepted seed and one-based attempt are returned as deterministic preflight
    evidence; no policy is loaded and no protected confirmation role is opened.
    """

    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts < 1
    ):
        raise U2rProtocolError("U2r layout sampler attempt cap must be a positive integer")
    if set(forbidden_layout_hashes) != set(lessons.LessonId):
        raise U2rProtocolError("U2r sampler requires one applied guard per lesson")
    normalized_streams = tuple(int(stream) for stream in worker_streams)
    if not normalized_streams or len(set(normalized_streams)) != len(normalized_streams):
        raise U2rProtocolError("U2r sampler worker streams must be nonempty and unique")

    records: list[dict[str, Any]] = []
    for lesson in lessons.LessonId:
        forbidden = forbidden_layout_hashes[lesson]
        for worker_stream in normalized_streams:
            generator = np.random.default_rng(worker_stream)
            environment = lessons.U2LessonEnv(lesson=lesson)
            try:
                for attempt in range(1, max_attempts + 1):
                    candidate_seed = int(
                        generator.integers(0, lessons.TRAINING_SEED_LIMIT)
                    )
                    options: dict[str, Any] = {}
                    if lesson is lessons.LessonId.SEPARATED_UNLOCK:
                        options = {
                            "u2_seed_role": U2SeedRole.TRAINING.value,
                            "u2_seed_access": seed_access,
                        }
                    _, info = environment.reset(
                        seed=candidate_seed,
                        options=options or None,
                    )
                    layout_digest = _require_sha256(
                        info.get("layout_sha256"),
                        f"U2r {lesson.value} sampler layout",
                    )
                    if layout_digest in forbidden:
                        continue
                    records.append(
                        {
                            "lesson_id": lesson.value,
                            "worker_stream": worker_stream,
                            "accepted_attempt": attempt,
                            "accepted_seed": candidate_seed,
                            "layout_sha256": layout_digest,
                            "forbidden_layouts": len(forbidden),
                            "max_attempts": max_attempts,
                        }
                    )
                    break
                else:
                    raise U2rProtocolError(
                        "U2r sampler found no admissible "
                        f"{lesson.value} layout for worker {worker_stream} "
                        f"within {max_attempts} attempts"
                    )
            finally:
                environment.close()
    return tuple(records)


def _evaluation_value(result: Any, name: str) -> Any:
    if isinstance(result, Mapping):
        return result.get(name)
    return getattr(result, name)


def _normalized_evaluations(
    evaluations: Mapping[Any, Any],
) -> dict[lessons.LessonId, Any]:
    normalized: dict[lessons.LessonId, Any] = {}
    for key, result in evaluations.items():
        try:
            lesson = key if isinstance(key, lessons.LessonId) else lessons.LessonId(str(key))
        except ValueError:
            lesson = lessons.LessonId(str(_evaluation_value(result, "lesson_id")))
        normalized[lesson] = result
    if set(normalized) != set(lessons.LessonId):
        raise U2rProtocolError("stability exam does not contain all four lessons")
    return normalized


def _evaluation_public(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    public = getattr(result, "public_dict", None)
    if not callable(public):
        raise U2rProtocolError("lesson evaluation has no public evidence")
    value = public()
    if not isinstance(value, Mapping):
        raise U2rProtocolError("lesson evaluation public evidence is invalid")
    return dict(value)


def verify_exam_case_evidence(
    evaluations: Mapping[Any, Any],
    case_records: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Mapping[Any, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Recompute and bind all U2r exam aggregates from 320 immutable cases."""

    normalized = _normalized_evaluations(evaluations)
    expected_by_lesson: dict[lessons.LessonId, tuple[int, ...]] = {}
    if expected_seeds is not None:
        for raw_lesson, raw_seeds in expected_seeds.items():
            lesson = (
                raw_lesson
                if isinstance(raw_lesson, lessons.LessonId)
                else lessons.LessonId(str(raw_lesson))
            )
            expected_by_lesson[lesson] = tuple(int(seed) for seed in raw_seeds)
        if set(expected_by_lesson) != set(lessons.LessonId):
            raise U2rProtocolError("case evidence seed panels are incomplete")

    records = [dict(record) for record in case_records]
    if len(records) != len(lessons.LessonId) * EVALUATION_SEED_COUNT:
        raise U2rProtocolError("U2r exam does not contain exactly 320 cases")
    per_lesson: dict[str, dict[str, Any]] = {}
    all_case_digests: list[str] = []
    ordered_records: list[dict[str, Any]] = []
    for lesson in lessons.LessonId:
        selected = [record for record in records if record.get("lesson_id") == lesson.value]
        selected.sort(key=lambda record: int(record.get("case_index", -1)))
        if len(selected) != EVALUATION_SEED_COUNT:
            raise U2rProtocolError(f"U2r {lesson.value} case evidence is not exactly 80 cases")
        expected = expected_by_lesson.get(lesson)
        measured_seeds = tuple(int(record.get("seed", -1)) for record in selected)
        if expected is not None and measured_seeds != expected:
            raise U2rProtocolError(f"U2r {lesson.value} case evidence seed order changed")
        panel_counts = tuple(
            sum(int(record.get("panel", -1)) == panel for record in selected) for panel in (0, 1)
        )
        if panel_counts != (40, 40):
            raise U2rProtocolError(f"U2r {lesson.value} case evidence panels changed")
        for index, record in enumerate(selected):
            if (
                int(record.get("panel_case_index", -1)) != index % 40
                or _require_sha256(
                    record.get("layout_sha256"),
                    f"{lesson.value} case layout",
                )
                != record.get("layout_sha256")
                or _require_sha256(
                    record.get("geometry_sha256"),
                    f"{lesson.value} case geometry",
                )
                != record.get("geometry_sha256")
            ):
                raise U2rProtocolError(f"U2r {lesson.value} case identity changed")
        result = normalized[lesson]
        public = _evaluation_public(result)
        try:
            recomputed = lessons.aggregate_u2_case_evidence(
                lesson,
                selected,
                timestamp=str(public["timestamp"]),
            ).public_dict()
        except (KeyError, TypeError, ValueError) as error:
            raise U2rProtocolError(
                f"U2r {lesson.value} case evidence is invalid: {error}"
            ) from error
        if _canonical_sha256(recomputed) != _canonical_sha256(public):
            raise U2rProtocolError(f"U2r {lesson.value} aggregate disagrees with immutable cases")
        case_digests = [_canonical_sha256(record) for record in selected]
        if len(set(case_digests)) != EVALUATION_SEED_COUNT:
            raise U2rProtocolError(f"U2r {lesson.value} case evidence is duplicated")
        all_case_digests.extend(case_digests)
        ordered_records.extend(selected)
        per_lesson[lesson.value] = {
            "cases": EVALUATION_SEED_COUNT,
            "panel_cases": [40, 40],
            "seed_sequence_sha256": _canonical_sha256(list(measured_seeds)),
            "case_records_sha256": _canonical_sha256(selected),
            "case_set_sha256": _hash_set_sha256(case_digests),
            "aggregate_sha256": _canonical_sha256(public),
        }
    if len(set(all_case_digests)) != len(all_case_digests):
        raise U2rProtocolError("U2r exam case records are duplicated across lessons")
    return {
        "schema_version": CASE_EVIDENCE_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "case_records": len(ordered_records),
        "case_records_sha256": _canonical_sha256(ordered_records),
        "case_set_sha256": _hash_set_sha256(all_case_digests),
        "per_lesson": per_lesson,
    }


def _existing_gate_passed(lesson: lessons.LessonId, result: Any) -> bool:
    try:
        spec = lessons.LESSON_SPECS[lesson]
        successes = int(_evaluation_value(result, "successes"))
        panels = tuple(int(value) for value in _evaluation_value(result, "panel_successes"))
        return (
            int(_evaluation_value(result, "episodes")) == EVALUATION_SEED_COUNT
            and len(panels) == 2
            and sum(panels) == successes
            and successes >= spec.overall_required
            and all(value >= spec.panel_required for value in panels)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _exam_evidence_valid(evaluations: Mapping[lessons.LessonId, Any]) -> bool:
    try:
        for lesson in lessons.LessonId:
            result = evaluations[lesson]
            raw_successes = _evaluation_value(result, "successes")
            raw_episodes = _evaluation_value(result, "episodes")
            raw_panels = _evaluation_value(result, "panel_successes")
            if (
                isinstance(raw_successes, bool)
                or isinstance(raw_episodes, bool)
                or int(raw_successes) != raw_successes
                or int(raw_episodes) != raw_episodes
                or int(raw_episodes) != EVALUATION_SEED_COUNT
                or not isinstance(raw_panels, Sequence)
                or isinstance(raw_panels, (str, bytes))
                or len(raw_panels) != 2
                or any(isinstance(value, bool) or int(value) != value for value in raw_panels)
                or sum(int(value) for value in raw_panels) != int(raw_successes)
                or not 0 <= int(raw_successes) <= EVALUATION_SEED_COUNT
            ):
                return False
            if lesson is not lessons.LessonId.NAVIGATE:
                ineffective = float(_evaluation_value(result, "mean_ineffective_interactions"))
                if not math.isfinite(ineffective) or ineffective < 0:
                    return False
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return True


def grade_stability_pair(
    penultimate: Mapping[Any, Any],
    terminal: Mapping[Any, Any],
    *,
    penultimate_allocation_valid: bool,
    terminal_allocation_valid: bool,
    penultimate_recovery: bool,
    terminal_recovery: bool,
    penultimate_child_actions: int | None = None,
    terminal_child_actions: int | None = None,
) -> StabilityDecision:
    """Apply the frozen terminal-only U2r stability and anti-regression rule."""

    prior_actions = (
        TERMINAL_CHILD_ACTIONS - EVALUATION_INTERVAL
        if penultimate_child_actions is None
        else int(penultimate_child_actions)
    )
    final_actions = (
        TERMINAL_CHILD_ACTIONS if terminal_child_actions is None else int(terminal_child_actions)
    )
    checks: dict[str, bool] = {
        "terminal_boundary": final_actions == TERMINAL_CHILD_ACTIONS,
        "adjacent_boundary": final_actions - prior_actions == EVALUATION_INTERVAL,
        "penultimate_normal": not bool(penultimate_recovery),
        "terminal_normal": not bool(terminal_recovery),
        "penultimate_allocation": penultimate_allocation_valid is True,
        "terminal_allocation": terminal_allocation_valid is True,
    }
    try:
        prior = _normalized_evaluations(penultimate)
    except (AttributeError, KeyError, TypeError, ValueError, U2rProtocolError):
        prior = {}
    try:
        final = _normalized_evaluations(terminal)
    except (AttributeError, KeyError, TypeError, ValueError, U2rProtocolError):
        final = {}
    checks["penultimate_evidence_complete"] = set(prior) == set(
        lessons.LessonId
    ) and _exam_evidence_valid(prior)
    checks["terminal_evidence_complete"] = set(final) == set(
        lessons.LessonId
    ) and _exam_evidence_valid(final)
    if not (checks["penultimate_evidence_complete"] and checks["terminal_evidence_complete"]):
        reasons = tuple(name for name, passed in checks.items() if not passed)
        return StabilityDecision(
            eligible=False,
            reasons=reasons,
            checks=MappingProxyType(checks),
        )
    for lesson in lessons.LessonId:
        label = lesson.value
        checks[f"{label}:penultimate_existing_gate"] = _existing_gate_passed(lesson, prior[lesson])
        checks[f"{label}:terminal_existing_gate"] = _existing_gate_passed(lesson, final[lesson])
        prior_successes = int(_evaluation_value(prior[lesson], "successes"))
        final_successes = int(_evaluation_value(final[lesson], "successes"))
        checks[f"{label}:terminal_decline_at_most_2"] = (
            final_successes >= prior_successes - MAX_TERMINAL_SUCCESS_DECLINE
        )
    for lesson in (
        lessons.LessonId.VISIBLE_UNLOCK,
        lessons.LessonId.LOCAL_UNLOCK,
        lessons.LessonId.SEPARATED_UNLOCK,
    ):
        label = lesson.value
        checks[f"{label}:penultimate_ineffective_at_most_3"] = (
            float(_evaluation_value(prior[lesson], "mean_ineffective_interactions"))
            <= INEFFECTIVE_INTERACTION_MAX + 1e-12
        )
        checks[f"{label}:terminal_ineffective_at_most_3"] = (
            float(_evaluation_value(final[lesson], "mean_ineffective_interactions"))
            <= INEFFECTIVE_INTERACTION_MAX + 1e-12
        )
    for name, result in (("penultimate", prior), ("terminal", final)):
        u2_result = result[lessons.LessonId.SEPARATED_UNLOCK]
        panels = tuple(int(value) for value in _evaluation_value(u2_result, "panel_successes"))
        checks[f"u2:{name}_overall_at_least_72"] = (
            int(_evaluation_value(u2_result, "successes")) >= U2_STABILITY_OVERALL_REQUIRED
        )
        checks[f"u2:{name}_panels_at_least_34"] = len(panels) == 2 and all(
            value >= U2_STABILITY_PANEL_REQUIRED for value in panels
        )
    reasons = tuple(name for name, passed in checks.items() if not passed)
    return StabilityDecision(
        eligible=not reasons,
        reasons=reasons,
        checks=MappingProxyType(checks),
    )


class EpisodeEvidenceWrapper(gym.Wrapper):
    """Read-only instrumentation for every active vector-worker episode."""

    def __init__(
        self,
        env: gym.Env,
        *,
        worker_index: int,
        worker_stream: int,
        ledger: Path,
    ) -> None:
        super().__init__(env)
        self.worker_index = int(worker_index)
        self.worker_stream = int(worker_stream)
        self.ledger = ledger
        self.episode_ordinal = 0
        self.local_transition_count = 0
        self._active: dict[str, Any] | None = None

    @staticmethod
    def _diagnostics(info: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "milestones": dict(info.get("milestones", {})),
            "success": bool(info.get("success", False)),
            "terminal_reason": info.get("terminal_reason"),
            "collisions": int(info.get("collisions", 0) or 0),
            "ineffective_interactions": int(info.get("ineffective_interactions", 0) or 0),
            "unique_cells": int(info.get("unique_cells", 0) or 0),
            "action_histogram": dict(info.get("action_histogram", {}) or {}),
            "extrinsic_return": float(info.get("extrinsic_return", 0.0) or 0.0),
            "curiosity_return": float(info.get("curiosity_return", 0.0) or 0.0),
        }

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.episode_ordinal += 1
        self._active = {
            "worker_index": self.worker_index,
            "worker_stream": self.worker_stream,
            "episode_ordinal": self.episode_ordinal,
            "episode_started_at": utc_now(),
            "worker_transition_at_start": self.local_transition_count,
            "seed": int(info["seed"]),
            "lesson_id": str(info["lesson_id"]),
            "lesson_label": info.get("lesson_label"),
            "layout_sha256": _require_sha256(info.get("layout_sha256"), "active worker layout"),
            "geometry_sha256": _require_sha256(
                info.get("geometry_sha256"), "active worker geometry"
            ),
            "generator_profile": info.get("generator_profile"),
            "generator_profile_version": (
                info.get("generator_profile_version") or lessons.GENERATOR_PROFILE_VERSION
            ),
            "seed_role": "training",
            "reset_provenance": "curriculum_training_rng",
            "elapsed_steps": 0,
            "active": True,
            "diagnostics": self._diagnostics(info),
        }
        append_jsonl(
            self.ledger,
            {"timestamp": utc_now(), "type": "episode_start", **self._active},
        )
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.local_transition_count += 1
        if self._active is None:
            raise U2rProtocolError("worker stepped before episode-start evidence")
        self._active["elapsed_steps"] = int(
            info.get("elapsed_steps", self._active["elapsed_steps"] + 1)
            or self._active["elapsed_steps"] + 1
        )
        self._active["active"] = not bool(terminated or truncated)
        self._active["diagnostics"] = self._diagnostics(info)
        return observation, reward, terminated, truncated, info

    def evidence_state(self) -> dict[str, Any]:
        if self._active is None:
            state = {
                "worker_index": self.worker_index,
                "worker_stream": self.worker_stream,
                "episode_ordinal": self.episode_ordinal,
                "active": False,
                "not_started": True,
            }
        else:
            state = json.loads(json.dumps(self._active))
        return state | {
            "untrained_transitions": 0,
            "exact_environment_resume_supported": False,
            "environment_state_disposition": "discarded_on_process_resume",
            "environment_rng_state_disposition": (
                "discarded_on_process_resume; next segment uses a frozen fresh stream"
            ),
            "policy_recurrent_state_disposition": "discarded_on_process_resume",
        }


def effective_config(
    args: argparse.Namespace,
    *,
    parent: U2rParent,
    qualification: frozen_u2.QualificationProvenance,
    exclusions: ForbiddenLayoutEvidence,
    anchor: Any | None = None,
) -> dict[str, Any]:
    """Return the complete prospective U2r contract stored in every bundle."""

    return {
        "protocol": PROTOCOL,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "continuation": {
            "source_protocol": frozen_u2.PROTOCOL,
            "source_child_seed": SOURCE_CHILD_SEED,
            "source_checkpoint_sha256": parent.checkpoint_sha256,
            "source_child_actions": SOURCE_CHILD_ACTIONS,
            "source_lifetime_actions": SOURCE_LIFETIME_ACTIONS,
            "source_optimizer_updates": SOURCE_OPTIMIZER_UPDATES,
            "additional_action_budget": int(args.additional_budget),
            "terminal_child_actions": TERMINAL_CHILD_ACTIONS,
            "terminal_lifetime_actions": TERMINAL_LIFETIME_ACTIONS,
            "exact_optimizer_continuation": True,
        },
        "algorithm_seed": int(args.seed),
        "initial_worker_streams": list(REMEDIATION_WORKER_STREAMS),
        "resume_segments": {
            "segment_seed_offset": SEGMENT_SEED_OFFSET,
            "algorithm_seed_formula": ("20260749 + segment_index * 100000"),
            "worker_stream_formula": (
                "[20260749,20260750,20260751,20260752] + segment_index * 100000"
            ),
            "active_environment_restored": False,
            "active_episode_disposition": "discarded_on_process_resume",
            "policy_recurrent_state_restored": False,
            "prior_episode_identity_retained": True,
        },
        "qualification_report_sha256": qualification.report_sha256,
        "external_preregistration": (
            anchor.public_dict() if anchor is not None and hasattr(anchor, "public_dict") else None
        ),
        "environment": {
            "size": int(args.size),
            "workers": int(args.workers),
            "generator_profile_version": lessons.GENERATOR_PROFILE_VERSION,
            "layout_resample_attempts": U2R_LAYOUT_RESAMPLE_ATTEMPTS,
            "observation_shape": [56, 56, 3],
            "action_count": 7,
            "mechanics_changed_from_u2": False,
        },
        "observation": {
            "kind": "egocentric_partial_rgb",
            "shape": [56, 56, 3],
            "recurrent_state_units": 256,
            "lesson_metadata": False,
            "privileged_state": False,
        },
        "reward": {
            "success": 1.0,
            "ordinary_step": -0.001,
            "pixel_novelty": 0.002,
            "episodic_curiosity_cap": 0.1,
            "milestone_shaping": False,
            "timeout_is_terminal_failure": True,
            "curiosity_during_exams": False,
            "changed_from_u2": False,
        },
        "optimization": {
            "rollout_steps": int(args.rollout_steps),
            "rollout_transitions": int(args.workers * args.rollout_steps),
            "batch_size": int(args.batch_size),
            "n_epochs": int(args.n_epochs),
            "learning_rate": float(args.learning_rate),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "ent_coef": 0.01,
            "policy_kwargs": dict(POLICY_KWARGS),
            "changed_from_u2": False,
        },
        "lessons": [lesson.value for lesson in lessons.LessonId],
        "lesson_specs": {
            lesson.value: {
                "generator_profile": lessons.LESSON_SPECS[lesson].generator_profile,
                "horizon": lessons.LESSON_SPECS[lesson].max_steps,
                "validation_seed_base": lessons.LESSON_SPECS[lesson].validation_seed_base,
            }
            for lesson in lessons.LessonId
        },
        "transition_target_profiles": lessons.target_profiles_public(),
        "exclusions": exclusions.public_dict(),
        "artifacts": {
            "archive_every_exam": True,
            "expected_exam_archives": KEEP_ALL_EXAMS,
            "rolling_exam_pruning": False,
            "minimum_free_gib": float(args.minimum_free_gib),
            "lineage_cap_gib": 2,
            "cohort_cap_gib": 6,
            "optional_media_cap_gib": 10,
        },
        "evaluation": {
            "interval": int(args.evaluation_every),
            "seeds_per_lesson": int(args.evaluation_seeds),
            "panels": [40, 40],
            "existing_thresholds": {
                lesson.value: {
                    "overall": int(lessons.LESSON_SPECS[lesson].overall_required),
                    "panel": int(lessons.LESSON_SPECS[lesson].panel_required),
                }
                for lesson in lessons.LessonId
            },
            "terminal_only": True,
            "terminal_pair": [
                TERMINAL_CHILD_ACTIONS - EVALUATION_INTERVAL,
                TERMINAL_CHILD_ACTIONS,
            ],
            "stability": {
                "normal_practice_required_on_both": True,
                "allocation_valid_required_on_both": True,
                "existing_gates_required_on_both": True,
                "ineffective_interaction_lessons": [
                    lessons.LessonId.VISIBLE_UNLOCK.value,
                    lessons.LessonId.LOCAL_UNLOCK.value,
                    lessons.LessonId.SEPARATED_UNLOCK.value,
                ],
                "mean_ineffective_interactions_max": (INEFFECTIVE_INTERACTION_MAX),
                "u2_overall_required": U2_STABILITY_OVERALL_REQUIRED,
                "u2_panel_required": U2_STABILITY_PANEL_REQUIRED,
                "terminal_success_decline_max": MAX_TERMINAL_SUCCESS_DECLINE,
            },
        },
        "confirmation_access": {
            "consumed_u2_streams_available": False,
            "fresh_u2r_streams_available_during_training": False,
            "fresh_reserved_range": [15_240_000, 15_279_999],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=SOURCE_ARCHIVE)
    parser.add_argument(
        "--confirmation-report",
        type=Path,
        default=U2_CONFIRMATION_REPORT,
    )
    parser.add_argument(
        "--qualification-report",
        type=Path,
        default=Path(
            "/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.2-u2-20260723/report.json"
        ),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=REMEDIATION_ALGORITHM_SEED)
    parser.add_argument("--additional-budget", type=int, default=ADDITIONAL_ACTION_BUDGET)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=PPO_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--evaluation-every", type=int, default=EVALUATION_INTERVAL)
    parser.add_argument("--evaluation-seeds", type=int, default=EVALUATION_SEED_COUNT)
    parser.add_argument("--frame-every", type=int, default=2_048)
    parser.add_argument("--keep-rolling-exams", type=int, default=KEEP_ALL_EXAMS)
    parser.add_argument("--minimum-free-gib", type=float, default=MINIMUM_FREE_GIB)
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--media-directory", type=Path, default=DEFAULT_MEDIA_DIRECTORY)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "additional action budget": args.additional_budget,
        "workers": args.workers,
        "rollout steps": args.rollout_steps,
        "batch size": args.batch_size,
        "epochs": args.n_epochs,
        "evaluation interval": args.evaluation_every,
        "evaluation cases": args.evaluation_seeds,
        "frame interval": args.frame_every,
        "exam retention": args.keep_rolling_exams,
    }
    for label, value in positive.items():
        if int(value) <= 0:
            raise SystemExit(f"{label} must be positive")
    if (
        not math.isfinite(float(args.minimum_free_gib))
        or float(args.minimum_free_gib) < MINIMUM_FREE_GIB
    ):
        raise SystemExit("U2r requires a free-space reserve of at least 25 GiB")
    if not math.isfinite(float(args.gamma)) or not MINIMUM_GAMMA <= float(args.gamma) <= 1.0:
        raise SystemExit("gamma violates reward dominance")
    if not math.isfinite(float(args.learning_rate)) or float(args.learning_rate) <= 0:
        raise SystemExit("learning rate must be finite and positive")
    frozen = {
        "parent": (
            args.parent.expanduser().resolve(),
            SOURCE_ARCHIVE.expanduser().resolve(),
        ),
        "confirmation report": (
            args.confirmation_report.expanduser().resolve(),
            U2_CONFIRMATION_REPORT.expanduser().resolve(),
        ),
        "algorithm seed": (args.seed, REMEDIATION_ALGORITHM_SEED),
        "additional action budget": (
            args.additional_budget,
            ADDITIONAL_ACTION_BUDGET,
        ),
        "workers": (args.workers, WORKERS),
        "map size": (args.size, 9),
        "rollout steps": (args.rollout_steps, ROLLOUT_STEPS),
        "batch size": (args.batch_size, 256),
        "epochs": (args.n_epochs, PPO_EPOCHS),
        "learning rate": (args.learning_rate, 2.5e-4),
        "gamma": (args.gamma, 0.995),
        "GAE lambda": (args.gae_lambda, 0.98),
        "evaluation interval": (args.evaluation_every, EVALUATION_INTERVAL),
        "evaluation cases": (args.evaluation_seeds, EVALUATION_SEED_COUNT),
        "exam retention": (args.keep_rolling_exams, KEEP_ALL_EXAMS),
        "device": (args.device, "cpu"),
        "storage root": (
            args.storage_root.expanduser().resolve(),
            DEFAULT_STORAGE_ROOT.expanduser().resolve(),
        ),
        "run root": (
            args.run_root.expanduser().resolve(),
            DEFAULT_RUN_ROOT.expanduser().resolve(),
        ),
        "media directory": (
            args.media_directory.expanduser().resolve(),
            DEFAULT_MEDIA_DIRECTORY.expanduser().resolve(),
        ),
    }
    if args.resume is None:
        frozen["fresh run name"] = (args.run_name, DEFAULT_RUN_NAME)
    deviations = [
        f"{label}={actual!r} (requires {expected!r})"
        for label, (actual, expected) in frozen.items()
        if actual != expected
    ]
    if deviations:
        raise SystemExit("frozen U2r configuration changed:\n  - " + "\n  - ".join(deviations))
    rollout = int(args.workers) * int(args.rollout_steps)
    if rollout % int(args.batch_size):
        raise SystemExit("workers x rollout steps must be divisible by batch size")
    if int(args.additional_budget) % rollout:
        raise SystemExit("additional budget must align to a complete vector rollout")
    if int(args.additional_budget) % int(args.evaluation_every):
        raise SystemExit("additional budget must end on an exam boundary")


def verify_storage_mount(storage_root: Path = DEFAULT_STORAGE_ROOT) -> dict[str, Any]:
    """Refuse to create canonical evidence paths unless the external T7 is mounted."""

    target = storage_root.expanduser()
    if target.is_symlink() or not target.is_dir():
        raise U2rProtocolError(f"U2r storage root is missing or unsafe: {target}")
    resolved = target.resolve()
    if resolved != DEFAULT_STORAGE_ROOT:
        raise U2rProtocolError("U2r storage root is not the canonical T7 path")
    if not os.path.ismount(resolved):
        raise U2rProtocolError("U2r storage root is not an independently mounted volume")
    root_device = Path("/").stat().st_dev
    target_device = resolved.stat().st_dev
    if target_device == root_device:
        raise U2rProtocolError("U2r storage root resolves to the internal system device")
    return {
        "path": str(resolved),
        "is_mount": True,
        "device": int(target_device),
        "system_device": int(root_device),
        "distinct_from_system": True,
    }


def _verify_loaded_model(
    model: Any,
    args: argparse.Namespace,
    *,
    expected_actions: int,
    expected_updates: int,
    expected_policy_sha256: str | None = None,
    expected_optimizer_sha256: str | None = None,
) -> None:
    if int(model.num_timesteps) != int(expected_actions):
        raise U2rProtocolError(
            f"model/sidecar action mismatch: {model.num_timesteps} != {expected_actions}"
        )
    if int(model._n_updates) != int(expected_updates):
        raise U2rProtocolError(
            f"model/sidecar optimizer mismatch: {model._n_updates} != {expected_updates}"
        )
    optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
    if optimizer is None or not optimizer.state:
        raise U2rProtocolError("loaded U2/U2r archive has no optimizer state")
    learning_rate = model.learning_rate
    if callable(learning_rate):
        learning_rate = learning_rate(1.0)
    if (
        int(model.n_steps) != int(args.rollout_steps)
        or int(model.batch_size) != int(args.batch_size)
        or int(model.n_epochs) != int(args.n_epochs)
        or float(model.gamma) != float(args.gamma)
        or float(model.gae_lambda) != float(args.gae_lambda)
        or float(model.ent_coef) != 0.01
        or float(learning_rate) != float(args.learning_rate)
        or int(model.action_space.n) != 7
        or tuple(model.observation_space.shape) != (3, 56, 56)
        or int(model.policy.lstm_actor.hidden_size) != 256
        or int(model.policy.lstm_actor.num_layers) != 1
    ):
        raise U2rProtocolError("loaded archive does not match the unchanged U2 learner contract")
    measured_policy = state_digests.policy_tensor_sha256(model)
    measured_optimizer = state_digests.optimizer_state_sha256(model)
    if expected_policy_sha256 is not None and measured_policy != _require_sha256(
        expected_policy_sha256, "expected loaded policy"
    ):
        raise U2rProtocolError("loaded policy tensors differ from frozen evidence")
    if expected_optimizer_sha256 is not None and measured_optimizer != _require_sha256(
        expected_optimizer_sha256, "expected loaded optimizer"
    ):
        raise U2rProtocolError("loaded optimizer state differs from frozen evidence")


def _segment_run_name(segment_index: int) -> str:
    if segment_index < 0:
        raise U2rProtocolError("U2r segment index cannot be negative")
    return DEFAULT_RUN_NAME if segment_index == 0 else f"{DEFAULT_RUN_NAME}-resume-{segment_index}"


def _validated_resume_scheduler(
    curriculum: lessons.CurriculumState,
    scheduler_state: Mapping[str, Any],
    *,
    child_actions: int,
    last_decision_child_actions: int | None,
) -> dict[str, Any]:
    try:
        scheduler = lessons.TransitionDeficitScheduler(curriculum, seed=0)
        scheduler.load_state_dict(scheduler_state)
        snapshot = scheduler.snapshot()
    except (KeyError, TypeError, ValueError) as error:
        raise U2rProtocolError(f"U2r resume scheduler is invalid: {error}") from error
    lifetime = snapshot.get("lifetime_transitions")
    window = snapshot.get("window_transitions")
    if not isinstance(lifetime, Mapping) or not isinstance(window, Mapping):
        raise U2rProtocolError("U2r resume scheduler counts are incomplete")
    lifetime_total = sum(int(value) for value in lifetime.values())
    window_total = sum(int(value) for value in window.values())
    window_start = (
        last_decision_child_actions
        if last_decision_child_actions is not None
        else SOURCE_CHILD_ACTIONS
    )
    expected_window = child_actions - window_start
    if (
        lifetime_total != child_actions
        or expected_window < 0
        or expected_window >= EVALUATION_INTERVAL
        or window_total != expected_window
    ):
        raise U2rProtocolError(
            "U2r resume scheduler does not preserve the exact current exam window"
        )
    return snapshot


def _validate_resume_worker(
    raw: Mapping[str, Any],
    *,
    worker_index: int,
    worker_stream: int,
    source_commit: str,
    lifetime_actions: int,
    child_actions: int,
    untrained_transitions: int = 0,
) -> dict[str, Any]:
    record = dict(raw)
    try:
        seed = int(record["seed"])
        lesson = lessons.LessonId(str(record["lesson_id"]))
        elapsed = int(record["elapsed_steps"])
        ordinal = int(record["episode_ordinal"])
        transition_start = int(record["worker_transition_at_start"])
    except (KeyError, TypeError, ValueError) as error:
        raise U2rProtocolError(
            f"U2r active worker {worker_index} identity is incomplete"
        ) from error
    if (
        int(record.get("worker_index", -1)) != worker_index
        or int(record.get("worker_stream", -1)) != worker_stream
        or not 0 <= seed <= 999_999
        or elapsed < 0
        or ordinal <= 0
        or transition_start < 0
        or not isinstance(record.get("active"), bool)
        or not isinstance(record.get("diagnostics"), Mapping)
        or record.get("seed_role") != "training"
        or record.get("reset_provenance") != "curriculum_training_rng"
        or record.get("generator_profile_version") != lessons.GENERATOR_PROFILE_VERSION
        or record.get("untrained_transitions") != untrained_transitions
        or record.get("exact_environment_resume_supported") is not False
        or record.get("environment_state_disposition") != "discarded_on_process_resume"
        or record.get("environment_rng_state_disposition")
        != "discarded_on_process_resume; next segment uses a frozen fresh stream"
        or record.get("policy_recurrent_state_disposition") != "discarded_on_process_resume"
        or record.get("checkpoint_source_commit") != source_commit
        or int(record.get("checkpoint_lifetime_trained_actions", -1)) != lifetime_actions
        or int(record.get("checkpoint_child_trained_actions", -1)) != child_actions
        or not isinstance(record.get("episode_started_at"), str)
        or not record["episode_started_at"]
        or not isinstance(record.get("generator_profile"), str)
        or not record["generator_profile"]
    ):
        raise U2rProtocolError(f"U2r active worker {worker_index} resume disposition changed")
    _require_sha256(record.get("layout_sha256"), "active worker layout")
    _require_sha256(record.get("geometry_sha256"), "active worker geometry")
    record["lesson_id"] = lesson.value
    return record


def _case_evidence_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".cases.json")


def _verify_exam_case_document(
    path: Path,
    *,
    run_directory: Path,
    binding: Mapping[str, Any],
    evaluations: Mapping[Any, Any],
    checkpoint: Path,
    checkpoint_sha256: str,
    sidecar_sha256: str,
    child_actions: int,
    expected_seeds: Mapping[Any, Sequence[int]] | None,
) -> dict[str, Any]:
    expected_path = _case_evidence_path(checkpoint)
    if path.resolve() != expected_path.resolve():
        raise U2rProtocolError("U2r exam case-evidence path changed")
    expected_file_sha256 = _require_sha256(binding.get("file_sha256"), "U2r case-evidence file")
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected_file_sha256:
        raise U2rProtocolError("U2r exam case-evidence file changed")
    document = _read_json(path, "U2r exam case evidence")
    records = document.get("records")
    summary = document.get("summary")
    if (
        document.get("schema_version") != CASE_EVIDENCE_SCHEMA_VERSION
        or document.get("protocol") != PROTOCOL
        or document.get("kind") != "immutable_exam_cases"
        or document.get("checkpoint") != checkpoint.name
        or document.get("checkpoint_sha256") != checkpoint_sha256
        or document.get("checkpoint_sidecar_sha256") != sidecar_sha256
        or int(document.get("child_trained_actions", -1)) != child_actions
        or not isinstance(records, list)
        or not all(isinstance(record, Mapping) for record in records)
        or not isinstance(summary, Mapping)
    ):
        raise U2rProtocolError("U2r exam case-evidence document binding changed")
    measured = verify_exam_case_evidence(
        evaluations,
        records,
        expected_seeds=expected_seeds,
    )
    if dict(summary) != measured:
        raise U2rProtocolError("U2r exam case-evidence summary changed")
    try:
        relative_path = path.resolve().relative_to(run_directory.resolve())
    except ValueError as error:
        raise U2rProtocolError("U2r case evidence escapes its source segment") from error
    expected_binding = measured | {
        "path": str(relative_path),
        "file_sha256": expected_file_sha256,
        "checkpoint": str(checkpoint.resolve().relative_to(run_directory.resolve())),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_sidecar_sha256": sidecar_sha256,
        "child_trained_actions": child_actions,
    }
    if dict(binding) != expected_binding:
        raise U2rProtocolError("U2r exam case-evidence sidecar binding changed")
    return expected_binding


def _safe_report_artifact(
    run_directory: Path,
    raw_path: Any,
    *,
    label: str,
) -> Path:
    relative = Path(str(raw_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise U2rProtocolError(f"{label} path escapes the U2r run")
    candidate = run_directory / relative
    if candidate.is_symlink():
        raise U2rProtocolError(f"{label} is missing or unsafe")
    path = candidate.resolve()
    try:
        path.relative_to(run_directory.resolve())
    except ValueError as error:
        raise U2rProtocolError(f"{label} path escapes the U2r run") from error
    if not path.is_file():
        raise U2rProtocolError(f"{label} is missing or unsafe")
    return path


def _safe_lineage_artifact(
    cohort_root: Path,
    raw_path: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise U2rProtocolError(f"{label} path is missing")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise U2rProtocolError(f"{label} path escapes the U2r cohort")
    candidate = cohort_root / relative
    if candidate.is_symlink():
        raise U2rProtocolError(f"{label} is missing or unsafe")
    path = candidate.resolve()
    try:
        path.relative_to(cohort_root.resolve())
    except ValueError as error:
        raise U2rProtocolError(f"{label} path escapes the U2r cohort") from error
    if not path.is_file():
        raise U2rProtocolError(f"{label} is missing or unsafe")
    return path


def _jsonl_inventory(path: Path) -> dict[str, Any]:
    records = _read_jsonl(path, "U2r episode-start ledger")
    return {
        "record_count": len(records),
        "episode_start_count": sum(record.get("type") == "episode_start" for record in records),
        "interruption_abandonment_count": sum(
            record.get("type") == "episode_abandoned_at_interruption" for record in records
        ),
        "resume_abandonment_count": sum(
            record.get("type") == "episode_abandoned_on_resume" for record in records
        ),
    }


def _jsonl_artifact_binding(
    path: Path,
    *,
    cohort_root: Path,
) -> dict[str, Any]:
    if path.is_symlink():
        raise U2rProtocolError(f"unsafe U2r JSONL artifact: {path}")
    if not path.exists():
        return {
            "path": str(path.relative_to(cohort_root)),
            "exists": False,
            "sha256": None,
            "record_count": 0,
        }
    records = _read_jsonl(path, f"U2r {path.name} history")
    return {
        "path": str(path.relative_to(cohort_root)),
        "exists": True,
        "sha256": file_sha256(path),
        "record_count": len(records),
    }


def _lesson_evaluation_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = lessons.LessonEvaluation.__dataclass_fields__
    if any(name not in record for name in fields):
        raise U2rProtocolError("U2r baseline evaluation record is incomplete")
    return {name: record[name] for name in fields}


def _build_parent_baseline_inventory(
    segment_directory: Path,
    *,
    cohort_root: Path,
) -> dict[str, Any]:
    checkpoint = segment_directory / "checkpoints" / "initial.zip"
    sidecar = checkpoint.with_suffix(".json")
    integrity_path = _integrity_path(checkpoint)
    cases = _case_evidence_path(checkpoint)
    integrity = _verify_integrity(checkpoint, sidecar)
    sidecar_value = _read_json(sidecar, "U2r parent-baseline sidecar")
    evaluations_path = segment_directory / "evaluations.jsonl"
    evaluations_records = _read_jsonl(
        evaluations_path,
        "U2r segment-zero evaluations",
    )
    baseline_records = [
        record
        for record in evaluations_records
        if record.get("trigger") == "diagnostic_parent_baseline"
    ]
    if (
        len(baseline_records) != len(lessons.LessonId)
        or any(record.get("counts_toward_gate") is not False for record in baseline_records)
        or {record.get("lesson_id") for record in baseline_records}
        != {lesson.value for lesson in lessons.LessonId}
    ):
        raise U2rProtocolError("U2r segment zero lacks one frozen parent-baseline evaluation")
    evaluations = {
        lessons.LessonId(str(record["lesson_id"])): (_lesson_evaluation_from_record(record))
        for record in baseline_records
    }
    bindings = [record.get("case_evidence") for record in baseline_records]
    if not isinstance(bindings[0], Mapping) or any(
        binding != bindings[0] for binding in bindings[1:]
    ):
        raise U2rProtocolError("U2r parent-baseline cases are not one immutable panel")
    binding = dict(bindings[0])
    _verify_exam_case_document(
        cases,
        run_directory=segment_directory,
        binding=binding,
        evaluations=evaluations,
        checkpoint=checkpoint,
        checkpoint_sha256=file_sha256(checkpoint),
        sidecar_sha256=file_sha256(sidecar),
        child_actions=SOURCE_CHILD_ACTIONS,
        expected_seeds={
            lesson: tuple(
                lessons.LESSON_SPECS[lesson].validation_seed_base + index
                for index in range(EVALUATION_SEED_COUNT)
            )
            for lesson in lessons.LessonId
        },
    )
    for lesson, result in evaluations.items():
        expected = SOURCE_BASELINE[lesson.value]
        if (
            int(result["successes"]) != int(expected["successes"])
            or tuple(int(value) for value in result["panel_successes"])
            != tuple(expected["panel_successes"])
            or abs(
                float(result["mean_ineffective_interactions"])
                - float(expected["mean_ineffective_interactions"])
            )
            > 1e-12
        ):
            raise U2rProtocolError(f"U2r parent baseline changed for {lesson.value}")
    if (
        sidecar_value.get("kind") != "initial"
        or sidecar_value.get("resume_eligible") is not True
        or sidecar_value.get("model_state")
        != {
            "policy_tensor_sha256": SOURCE_POLICY_TENSOR_SHA256,
            "optimizer_state_sha256": SOURCE_OPTIMIZER_STATE_SHA256,
        }
    ):
        raise U2rProtocolError("U2r parent-baseline model state changed")
    return {
        "checkpoint": str(checkpoint.relative_to(cohort_root)),
        "checkpoint_sha256": str(integrity["checkpoint_sha256"]),
        "sidecar": str(sidecar.relative_to(cohort_root)),
        "sidecar_sha256": str(integrity["sidecar_sha256"]),
        "integrity": str(integrity_path.relative_to(cohort_root)),
        "integrity_sha256": file_sha256(integrity_path),
        "case_evidence": str(cases.relative_to(cohort_root)),
        "case_evidence_sha256": file_sha256(cases),
        "case_evidence_binding": binding,
        "evaluation_records_sha256": _canonical_json_sha256(baseline_records),
        "source_baseline": {
            lesson: {
                "successes": int(value["successes"]),
                "panel_successes": list(value["panel_successes"]),
                "mean_ineffective_interactions": float(value["mean_ineffective_interactions"]),
            }
            for lesson, value in SOURCE_BASELINE.items()
        },
    }


def _verify_interruption_resolution(
    document: Mapping[str, Any],
) -> None:
    at_signal = document.get("at_signal")
    resolution = document.get("resolution")
    resume_boundary = document.get("resume_boundary")
    workers = document.get("active_workers")
    if (
        not isinstance(at_signal, Mapping)
        or not isinstance(resolution, Mapping)
        or not isinstance(resume_boundary, Mapping)
        or not isinstance(resume_boundary.get("progress"), Mapping)
        or not isinstance(workers, list)
        or len(workers) != WORKERS
        or document.get("active_workers_sha256") != _canonical_json_sha256(workers)
    ):
        raise U2rProtocolError("U2r interruption resolution evidence is incomplete")
    try:
        signal_collected = int(at_signal["collected_actions"])
        signal_trained = int(at_signal["trained_actions"])
        signal_child = int(at_signal["child_trained_actions"])
        signal_updates = int(at_signal["optimizer_updates"])
        untrained_actions = int(at_signal["untrained_collected_actions"])
        untrained_per_worker = int(at_signal["untrained_transitions_per_worker"])
        boundary_progress = resume_boundary["progress"]
        boundary_trained = int(boundary_progress["trained_actions"])
        boundary_child = int(boundary_progress["child_trained_actions"])
        boundary_updates = int(boundary_progress["optimizer_updates"])
        resolved_trained = int(resolution["resolved_trained_actions"])
        resolved_child = int(resolution["resolved_child_trained_actions"])
        resolved_updates = int(resolution["resolved_optimizer_updates"])
        safe_trained = int(resolution["safe_boundary_trained_actions"])
        recovered_actions = int(resolution["recovered_complete_actions"])
        recovered_updates = int(resolution["recovered_complete_optimizer_updates"])
        discarded_post_safe = int(resolution["discarded_post_safe_rollout_actions"])
        unpublished_trained = int(resolution["unpublished_trained_actions"])
        resolution_untrained = int(resolution["untrained_collected_actions"])
        unpublished_complete = int(resolution["unpublished_complete_actions"])
    except (KeyError, TypeError, ValueError) as error:
        raise U2rProtocolError("U2r interruption resolution counters are incomplete") from error
    initial_classification = at_signal.get("pending_optimizer_boundary")
    final_classification = resolution.get("pending_optimizer_boundary")
    pending_actions = signal_collected - signal_trained
    trained_remediation = signal_trained - SOURCE_LIFETIME_ACTIONS
    expected_trained_updates = (
        SOURCE_OPTIMIZER_UPDATES + trained_remediation // ROLLOUT_TRANSITIONS * PPO_EPOCHS
    )
    pending_updates = signal_updates - expected_trained_updates
    common = (
        signal_collected >= signal_trained >= SOURCE_LIFETIME_ACTIONS
        and signal_child == signal_trained - SOURCE_U1_INHERITED_ACTIONS
        and untrained_actions == untrained_per_worker * WORKERS
        and untrained_actions >= 0
        and initial_classification in {"none", "partial", "complete"}
        and final_classification
        in {
            "none",
            "partial",
            "complete_recovered",
            "complete_recovery_failed",
        }
        and boundary_trained == safe_trained
        and boundary_child == boundary_trained - SOURCE_U1_INHERITED_ACTIONS
        and resolved_child == resolved_trained - SOURCE_U1_INHERITED_ACTIONS
        and discarded_post_safe == signal_collected - safe_trained
        and unpublished_trained == resolved_trained - safe_trained
        and resolution_untrained == untrained_actions
        and resolution.get("active_environment_state_disposition")
        == "discarded_at_process_boundary"
        and resolution.get("policy_recurrent_state_disposition") == "discarded_at_process_boundary"
    )
    if initial_classification == "none":
        coherent = (
            final_classification == "none"
            and pending_actions == 0
            and pending_updates == 0
            and untrained_actions == 0
            and resolved_trained == signal_trained
            and safe_trained <= signal_trained
            and resolved_updates == signal_updates
            and boundary_updates <= signal_updates
            and recovered_actions == recovered_updates == 0
            and unpublished_complete == 0
            and resolution.get("recovery_error") is None
        )
    elif initial_classification == "partial":
        coherent = (
            final_classification == "partial"
            and pending_actions >= 0
            and untrained_actions == pending_actions
            and resolved_trained == signal_trained
            and resolved_updates == signal_updates
            and safe_trained <= resolved_trained
            and boundary_updates <= resolved_updates
            and recovered_actions == recovered_updates == 0
            and unpublished_complete == 0
            and resolution.get("recovery_error") is None
        )
    elif final_classification == "complete_recovered":
        coherent = (
            pending_actions > 0
            and pending_actions % ROLLOUT_TRANSITIONS == 0
            and pending_updates == pending_actions // ROLLOUT_TRANSITIONS * PPO_EPOCHS
            and untrained_actions == 0
            and recovered_actions == pending_actions
            and recovered_updates == pending_updates
            and resolved_trained == signal_collected == safe_trained
            and resolved_updates == signal_updates == boundary_updates
            and unpublished_complete == 0
            and resolution.get("recovery_error") is None
        )
    else:
        coherent = (
            final_classification == "complete_recovery_failed"
            and pending_actions > 0
            and pending_actions % ROLLOUT_TRANSITIONS == 0
            and pending_updates == pending_actions // ROLLOUT_TRANSITIONS * PPO_EPOCHS
            and untrained_actions == 0
            and recovered_actions == recovered_updates == 0
            and safe_trained <= resolved_trained <= signal_collected
            and boundary_updates <= resolved_updates <= signal_updates
            and unpublished_complete == unpublished_trained
            and isinstance(resolution.get("recovery_error"), str)
            and bool(resolution.get("recovery_error"))
        )
    if not common or not coherent:
        raise U2rProtocolError("U2r interruption resolution is incoherent")


def _verify_resumed_worker_abandonment(
    *,
    segment: Mapping[str, Any],
    ledger_path: Path,
    source_workers: Sequence[Mapping[str, Any]],
    source_interruption: Path,
    source_interruption_sha256: str,
    source_active_workers_sha256: str,
) -> None:
    """Verify the process-boundary disposition of every inherited episode."""

    inherited = segment.get("inherited_active_workers")
    if (
        segment.get("inherited_active_workers_abandoned") is not True
        or not isinstance(inherited, list)
        or len(inherited) != WORKERS
        or len(source_workers) != WORKERS
    ):
        raise U2rProtocolError("resumed U2r segment lacks four inherited-worker abandonments")
    ledger_records = _read_jsonl(
        ledger_path,
        "U2r resumed-segment episode ledger",
    )
    resume_abandonments = [
        record for record in ledger_records if record.get("type") == "episode_abandoned_on_resume"
    ]
    if len(resume_abandonments) != WORKERS:
        raise U2rProtocolError("resumed U2r segment has an invalid abandonment-ledger count")
    expected_source = str(source_interruption.resolve())
    extra_fields = {
        "abandoned_on_resume",
        "abandoned_at",
        "abandonment_reason",
        "active_worker_evidence_source",
        "source_interruption",
        "source_interruption_sha256",
        "source_interruption_active_workers_sha256",
    }
    for index, source_worker in enumerate(source_workers):
        record = inherited[index]
        if (
            not isinstance(record, Mapping)
            or set(record) != set(source_worker) | extra_fields
            or any(record.get(field) != value for field, value in source_worker.items())
            or record.get("abandoned_on_resume") is not True
            or not isinstance(record.get("abandoned_at"), str)
            or not record.get("abandoned_at")
            or record.get("abandonment_reason") != RESUME_ABANDONMENT_REASON
            or record.get("active_worker_evidence_source") != "interruption.json"
            or record.get("source_interruption") != expected_source
            or record.get("source_interruption_sha256") != source_interruption_sha256
            or record.get("source_interruption_active_workers_sha256")
            != source_active_workers_sha256
        ):
            raise U2rProtocolError(f"resumed U2r worker {index} abandonment changed")
        expected_ledger_record = {
            "timestamp": record["abandoned_at"],
            "type": "episode_abandoned_on_resume",
            **dict(record),
        }
        if (
            sum(ledger_record == expected_ledger_record for ledger_record in resume_abandonments)
            != 1
        ):
            raise U2rProtocolError(f"resumed U2r worker {index} has no exact abandonment record")


def _build_lineage_history(
    run_directory: Path,
    *,
    terminal_sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    cohort_root = run_directory.parent.resolve()
    segment = terminal_sidecar.get("segment")
    if not isinstance(segment, Mapping):
        raise U2rProtocolError("terminal U2r sidecar lacks its segment identity")
    terminal_index = int(segment.get("index", -1))
    if (
        terminal_index < 0
        or run_directory.resolve() != (cohort_root / _segment_run_name(terminal_index)).resolve()
    ):
        raise U2rProtocolError("terminal U2r segment is not canonical")
    inventory: list[dict[str, Any]] = []
    for index in range(terminal_index + 1):
        segment_directory = cohort_root / _segment_run_name(index)
        if segment_directory.is_symlink() or not segment_directory.is_dir():
            raise U2rProtocolError(f"U2r lineage segment {index} is missing or unsafe")
        manifest_path = segment_directory / "manifest.json"
        ledger_path = segment_directory / "episode-starts.jsonl"
        manifest = _read_json(manifest_path, f"U2r segment {index} manifest")
        if (
            manifest.get("protocol") != PROTOCOL
            or not isinstance(manifest.get("segment"), Mapping)
            or int(manifest["segment"].get("index", -1)) != index
        ):
            raise U2rProtocolError(f"U2r lineage segment {index} manifest changed")
        ledger_inventory = _jsonl_inventory(ledger_path)
        history_ledgers = {
            name: _jsonl_artifact_binding(
                segment_directory / filename,
                cohort_root=cohort_root,
            )
            for name, filename in (
                ("episodes", "episodes.jsonl"),
                ("events", "events.jsonl"),
                ("optimizer", "optimizer.jsonl"),
                ("evaluations", "evaluations.jsonl"),
            )
        }
        interruption_path = segment_directory / "interruption.json"
        interruption_integrity_path = segment_directory / "interruption.integrity.json"
        interruption: dict[str, Any] | None = None
        if index < terminal_index:
            interruption_document = _read_json(
                interruption_path,
                f"U2r segment {index} interruption",
            )
            interruption_integrity = _read_json(
                interruption_integrity_path,
                f"U2r segment {index} interruption integrity",
            )
            if (
                interruption_integrity.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
                or interruption_integrity.get("protocol") != PROTOCOL
                or interruption_integrity.get("interruption") != interruption_path.name
                or interruption_integrity.get("interruption_sha256")
                != file_sha256(interruption_path)
            ):
                raise U2rProtocolError(f"U2r segment {index} interruption binding changed")
            interruption = {
                "path": str(interruption_path.relative_to(cohort_root)),
                "sha256": file_sha256(interruption_path),
                "integrity": str(interruption_integrity_path.relative_to(cohort_root)),
                "integrity_sha256": file_sha256(interruption_integrity_path),
                "active_workers_sha256": _require_sha256(
                    interruption_document.get("active_workers_sha256"),
                    f"U2r segment {index} interruption workers",
                ),
            }
        elif interruption_path.exists() or interruption_integrity_path.exists():
            raise U2rProtocolError("terminal U2r segment unexpectedly has interruption evidence")
        inventory.append(
            {
                "index": index,
                "run_name": segment_directory.name,
                "manifest": {
                    "path": str(manifest_path.relative_to(cohort_root)),
                    "sha256": file_sha256(manifest_path),
                },
                "episode_start_ledger": {
                    "path": str(ledger_path.relative_to(cohort_root)),
                    "sha256": file_sha256(ledger_path),
                    **ledger_inventory,
                },
                "history_ledgers": history_ledgers,
                "parent_baseline": (
                    _build_parent_baseline_inventory(
                        segment_directory,
                        cohort_root=cohort_root,
                    )
                    if index == 0
                    else None
                ),
                "interruption": interruption,
            }
        )
    active_workers = terminal_sidecar.get("active_workers")
    if not isinstance(active_workers, list) or len(active_workers) != WORKERS:
        raise U2rProtocolError("terminal U2r sidecar lacks four active-worker identities")
    return {
        "cohort_root": str(cohort_root),
        "terminal_segment_index": terminal_index,
        "segments": inventory,
        "terminal_active_workers": {
            "count": WORKERS,
            "records": [dict(record) for record in active_workers],
            "records_sha256": _canonical_json_sha256(active_workers),
        },
    }


def _verify_lineage_history(
    report: Mapping[str, Any],
    *,
    root: Path,
    terminal_sidecar: Mapping[str, Any],
) -> None:
    history = report.get("lineage_history")
    report_segment = report.get("segment")
    if not isinstance(history, Mapping) or not isinstance(report_segment, Mapping):
        raise U2rProtocolError("U2r terminal report lacks lineage history")
    terminal_index = int(report_segment.get("index", -1))
    cohort_root = root.parent
    if (
        cohort_root.is_symlink()
        or history.get("cohort_root") != str(cohort_root.resolve())
        or int(history.get("terminal_segment_index", -1)) != terminal_index
        or root.resolve() != (cohort_root / _segment_run_name(terminal_index)).resolve()
    ):
        raise U2rProtocolError("U2r terminal lineage root changed")
    entries = history.get("segments")
    if (
        not isinstance(entries, list)
        or len(entries) != terminal_index + 1
        or [int(entry.get("index", -1)) for entry in entries] != list(range(terminal_index + 1))
    ):
        raise U2rProtocolError("U2r terminal lineage segment order changed")
    previous_interruption: Mapping[str, Any] | None = None
    previous_interruption_workers: tuple[Mapping[str, Any], ...] | None = None
    previous_interruption_path: Path | None = None
    previous_interruption_sha256: str | None = None
    previous_active_workers_sha256: str | None = None
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise U2rProtocolError("U2r terminal lineage entry is invalid")
        expected_run_name = _segment_run_name(index)
        if entry.get("run_name") != expected_run_name:
            raise U2rProtocolError("U2r terminal lineage run name changed")
        manifest_binding = entry.get("manifest")
        ledger_binding = entry.get("episode_start_ledger")
        if not isinstance(manifest_binding, Mapping) or not isinstance(ledger_binding, Mapping):
            raise U2rProtocolError("U2r terminal lineage artifacts are incomplete")
        manifest_path = _safe_lineage_artifact(
            cohort_root,
            manifest_binding.get("path"),
            label=f"U2r segment {index} manifest",
        )
        ledger_path = _safe_lineage_artifact(
            cohort_root,
            ledger_binding.get("path"),
            label=f"U2r segment {index} episode ledger",
        )
        expected_directory = (cohort_root / expected_run_name).resolve()
        if (
            manifest_path.parent != expected_directory
            or ledger_path.parent != expected_directory
            or set(manifest_binding) != {"path", "sha256"}
            or manifest_binding.get("sha256") != file_sha256(manifest_path)
            or ledger_binding.get("sha256") != file_sha256(ledger_path)
            or {
                key: ledger_binding.get(key)
                for key in (
                    "record_count",
                    "episode_start_count",
                    "interruption_abandonment_count",
                    "resume_abandonment_count",
                )
            }
            != _jsonl_inventory(ledger_path)
        ):
            raise U2rProtocolError(f"U2r segment {index} lineage artifact changed")
        history_ledgers = entry.get("history_ledgers")
        if not isinstance(history_ledgers, Mapping) or set(history_ledgers) != {
            "episodes",
            "events",
            "optimizer",
            "evaluations",
        }:
            raise U2rProtocolError(f"U2r segment {index} history ledgers are incomplete")
        for name, filename in (
            ("episodes", "episodes.jsonl"),
            ("events", "events.jsonl"),
            ("optimizer", "optimizer.jsonl"),
            ("evaluations", "evaluations.jsonl"),
        ):
            binding = history_ledgers[name]
            if not isinstance(binding, Mapping):
                raise U2rProtocolError(f"U2r segment {index} {name} ledger binding is invalid")
            expected_path = expected_directory / filename
            if binding.get("path") != str(expected_path.relative_to(cohort_root)):
                raise U2rProtocolError(f"U2r segment {index} {name} ledger path changed")
            exists = binding.get("exists")
            if not isinstance(exists, bool):
                raise U2rProtocolError(f"U2r segment {index} {name} ledger state changed")
            if exists:
                path = _safe_lineage_artifact(
                    cohort_root,
                    binding["path"],
                    label=f"U2r segment {index} {name} ledger",
                )
                measured_records = _read_jsonl(
                    path,
                    f"U2r segment {index} {name} ledger",
                )
                if binding.get("sha256") != file_sha256(path) or int(
                    binding.get("record_count", -1)
                ) != len(measured_records):
                    raise U2rProtocolError(f"U2r segment {index} {name} ledger changed")
            elif (
                expected_path.exists()
                or expected_path.is_symlink()
                or binding.get("sha256") is not None
                or binding.get("record_count") != 0
            ):
                raise U2rProtocolError(f"U2r segment {index} absent {name} ledger changed")
        baseline = entry.get("parent_baseline")
        if index == 0:
            measured_baseline = _build_parent_baseline_inventory(
                expected_directory,
                cohort_root=cohort_root,
            )
            if baseline != measured_baseline:
                raise U2rProtocolError("U2r parent-baseline lineage binding changed")
        elif baseline is not None:
            raise U2rProtocolError("resumed U2r segment unexpectedly repeats parent baseline")
        manifest = _read_json(
            manifest_path,
            f"U2r segment {index} manifest",
        )
        segment = manifest.get("segment")
        if (
            manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or manifest.get("protocol") != PROTOCOL
            or manifest.get("source") != report.get("source")
            or manifest.get("effective_config") != report.get("effective_config")
            or manifest.get("parent") != report.get("parent")
            or manifest.get("qualification") != report.get("qualification")
            or manifest.get("exclusions") != report.get("exclusions")
            or manifest.get("external_preregistration") != report.get("external_preregistration")
            or manifest.get("storage_mount") != report.get("storage_mount")
            or not isinstance(segment, Mapping)
            or int(segment.get("index", -1)) != index
        ):
            raise U2rProtocolError(f"U2r segment {index} manifest provenance changed")
        if index == 0:
            if any(
                segment.get(field) is not None
                for field in (
                    "resume_checkpoint",
                    "resume_checkpoint_sha256",
                    "resume_model_state",
                    "resume_source_segment_index",
                )
            ):
                raise U2rProtocolError("initial U2r segment unexpectedly has resume provenance")
            if (
                segment.get("inherited_active_workers") != []
                or segment.get("inherited_active_workers_abandoned") is not False
                or _jsonl_inventory(ledger_path)["resume_abandonment_count"] != 0
            ):
                raise U2rProtocolError(
                    "initial U2r segment unexpectedly abandons inherited workers"
                )
        else:
            if (
                int(segment.get("resume_source_segment_index", -1)) != index - 1
                or not isinstance(segment.get("resume_checkpoint"), str)
                or _require_sha256(
                    segment.get("resume_checkpoint_sha256"),
                    f"U2r segment {index} resume checkpoint",
                )
                != segment.get("resume_checkpoint_sha256")
                or not isinstance(segment.get("resume_model_state"), Mapping)
                or previous_interruption is None
                or segment.get("resume_checkpoint") != previous_interruption.get("checkpoint")
                or segment.get("resume_checkpoint_sha256")
                != previous_interruption.get("checkpoint_sha256")
                or segment.get("resume_model_state") != previous_interruption.get("model_state")
            ):
                raise U2rProtocolError(f"U2r segment {index} resume link changed")
            if (
                previous_interruption_workers is None
                or previous_interruption_path is None
                or previous_interruption_sha256 is None
                or previous_active_workers_sha256 is None
            ):
                raise U2rProtocolError(f"U2r segment {index} lacks its predecessor interruption")
            _verify_resumed_worker_abandonment(
                segment=segment,
                ledger_path=ledger_path,
                source_workers=previous_interruption_workers,
                source_interruption=previous_interruption_path,
                source_interruption_sha256=previous_interruption_sha256,
                source_active_workers_sha256=(previous_active_workers_sha256),
            )
        interruption_binding = entry.get("interruption")
        if index < terminal_index:
            if not isinstance(interruption_binding, Mapping):
                raise U2rProtocolError(f"U2r segment {index} lacks interruption evidence")
            interruption_path = _safe_lineage_artifact(
                cohort_root,
                interruption_binding.get("path"),
                label=f"U2r segment {index} interruption",
            )
            interruption_integrity_path = _safe_lineage_artifact(
                cohort_root,
                interruption_binding.get("integrity"),
                label=f"U2r segment {index} interruption integrity",
            )
            if (
                interruption_path.parent != expected_directory
                or interruption_integrity_path.parent != expected_directory
                or interruption_binding.get("sha256") != file_sha256(interruption_path)
                or interruption_binding.get("integrity_sha256")
                != file_sha256(interruption_integrity_path)
            ):
                raise U2rProtocolError(f"U2r segment {index} interruption artifact changed")
            interruption_document = _read_json(
                interruption_path,
                f"U2r segment {index} interruption",
            )
            interruption_integrity = _read_json(
                interruption_integrity_path,
                f"U2r segment {index} interruption integrity",
            )
            if (
                interruption_integrity.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
                or interruption_integrity.get("protocol") != PROTOCOL
                or interruption_integrity.get("interruption") != interruption_path.name
                or interruption_integrity.get("interruption_sha256")
                != file_sha256(interruption_path)
                or interruption_document.get("segment") != segment
                or interruption_document.get("source") != report.get("source")
                or interruption_document.get("effective_config") != report.get("effective_config")
                or interruption_document.get("active_workers_sha256")
                != interruption_binding.get("active_workers_sha256")
            ):
                raise U2rProtocolError(f"U2r segment {index} interruption provenance changed")
            _verify_interruption_resolution(interruption_document)
            interruption_workers = interruption_document.get("active_workers")
            segment_ledger_records = _read_jsonl(
                ledger_path,
                f"U2r segment {index} interruption episode ledger",
            )
            if not isinstance(interruption_workers, list):
                raise U2rProtocolError(f"U2r segment {index} interruption workers changed")
            for worker in interruption_workers:
                matching_abandonments = [
                    record
                    for record in segment_ledger_records
                    if record.get("type") == "episode_abandoned_at_interruption"
                    and record.get("interruption_phase") == interruption_document.get("phase")
                    and all(
                        record.get(field) == worker.get(field)
                        for field in (
                            "worker_index",
                            "worker_stream",
                            "episode_ordinal",
                            "seed",
                            "lesson_id",
                            "layout_sha256",
                            "geometry_sha256",
                            "elapsed_steps",
                        )
                    )
                ]
                if len(matching_abandonments) != 1:
                    raise U2rProtocolError(f"U2r segment {index} worker abandonment changed")
            resume_boundary = interruption_document.get("resume_boundary")
            if not isinstance(resume_boundary, Mapping):
                raise U2rProtocolError(f"U2r segment {index} interruption has no resume boundary")
            safe_checkpoint = Path(str(resume_boundary.get("checkpoint", "")))
            safe_sidecar = safe_checkpoint.with_suffix(".json")
            safe_integrity_path = _integrity_path(safe_checkpoint)
            if (
                not safe_checkpoint.is_absolute()
                or safe_checkpoint.is_symlink()
                or not safe_checkpoint.is_file()
                or safe_checkpoint.parent.parent != expected_directory
                or safe_checkpoint.name not in {"initial.zip", "resume.zip", "latest-safe.zip"}
                or resume_boundary.get("sidecar") != str(safe_sidecar.resolve())
                or resume_boundary.get("integrity") != str(safe_integrity_path.resolve())
            ):
                raise U2rProtocolError(f"U2r segment {index} safe resume path changed")
            safe_integrity = _verify_integrity(
                safe_checkpoint,
                safe_sidecar,
            )
            safe_sidecar_value = _read_json(
                safe_sidecar,
                f"U2r segment {index} safe sidecar",
            )
            if (
                resume_boundary.get("checkpoint_sha256") != safe_integrity["checkpoint_sha256"]
                or resume_boundary.get("sidecar_sha256") != safe_integrity["sidecar_sha256"]
                or resume_boundary.get("integrity_sha256") != file_sha256(safe_integrity_path)
                or resume_boundary.get("model_state") != safe_sidecar_value.get("model_state")
                or resume_boundary.get("progress") != safe_sidecar_value.get("progress")
                or safe_sidecar_value.get("resume_eligible") is not True
                or safe_sidecar_value.get("segment") != segment
                or safe_sidecar_value.get("source") != report.get("source")
                or safe_sidecar_value.get("effective_config") != report.get("effective_config")
                or safe_sidecar_value.get("parent") != report.get("parent")
                or safe_sidecar_value.get("qualification") != report.get("qualification")
            ):
                raise U2rProtocolError(f"U2r segment {index} safe resume bundle changed")
            status = _read_json(
                expected_directory / "status.json",
                f"U2r segment {index} terminal status",
            )
            status_interruption = status.get("interruption")
            if (
                status.get("phase") not in {"interrupted", "crashed"}
                or not isinstance(status_interruption, Mapping)
                or status_interruption.get("sha256") != interruption_binding.get("sha256")
                or status_interruption.get("integrity_sha256")
                != interruption_binding.get("integrity_sha256")
                or status_interruption.get("active_workers_sha256")
                != interruption_binding.get("active_workers_sha256")
            ):
                raise U2rProtocolError(f"U2r segment {index} interruption status changed")
            verified_interruption, verified_workers = _verify_interruption_for_resume(
                expected_directory,
                checkpoint=safe_checkpoint,
                checkpoint_sidecar=safe_sidecar_value,
                expected_config=report["effective_config"],
                parent=report["parent"],
                qualification=report["qualification"],
                expected_source_commit=str(report["source"]["commit"]),
                segment_index=index,
            )
            if verified_interruption != interruption_document:
                raise U2rProtocolError(f"U2r segment {index} interruption semantics changed")
            previous_interruption = resume_boundary
            previous_interruption_workers = verified_workers
            previous_interruption_path = interruption_path
            previous_interruption_sha256 = file_sha256(interruption_path)
            previous_active_workers_sha256 = _require_sha256(
                interruption_document.get("active_workers_sha256"),
                f"U2r segment {index} active-worker evidence",
            )
        elif interruption_binding is not None:
            raise U2rProtocolError("terminal U2r lineage unexpectedly ends in interruption")
    terminal_workers = history.get("terminal_active_workers")
    sidecar_workers = terminal_sidecar.get("active_workers")
    if (
        not isinstance(terminal_workers, Mapping)
        or terminal_workers.get("count") != WORKERS
        or terminal_workers.get("records") != sidecar_workers
        or terminal_workers.get("records_sha256") != _canonical_json_sha256(sidecar_workers)
        or not isinstance(sidecar_workers, list)
        or len(sidecar_workers) != WORKERS
        or {int(worker.get("worker_index", -1)) for worker in sidecar_workers}
        != set(range(WORKERS))
    ):
        raise U2rProtocolError("U2r terminal active-worker lineage changed")
    terminal_streams = report_segment.get("segment_worker_streams")
    report_source = report.get("source")
    terminal_progress = terminal_sidecar.get("progress")
    if (
        not isinstance(terminal_streams, list)
        or len(terminal_streams) != WORKERS
        or not isinstance(report_source, Mapping)
        or not isinstance(terminal_progress, Mapping)
    ):
        raise U2rProtocolError("U2r terminal active-worker context is incomplete")
    validated_workers = [
        _validate_resume_worker(
            sidecar_workers[index],
            worker_index=index,
            worker_stream=int(terminal_streams[index]),
            source_commit=str(report_source.get("commit")),
            lifetime_actions=int(terminal_progress["lifetime_trained_actions"]),
            child_actions=int(terminal_progress["child_trained_actions"]),
            untrained_transitions=0,
        )
        for index in range(WORKERS)
    ]
    terminal_ledger = _read_jsonl(
        root / "episode-starts.jsonl",
        "U2r terminal episode-start ledger",
    )
    identity_fields = (
        "worker_index",
        "worker_stream",
        "episode_ordinal",
        "seed",
        "lesson_id",
        "layout_sha256",
        "geometry_sha256",
        "worker_transition_at_start",
    )
    for worker in validated_workers:
        matching_starts = [
            record
            for record in terminal_ledger
            if record.get("type") == "episode_start"
            and all(record.get(field) == worker.get(field) for field in identity_fields)
        ]
        if len(matching_starts) != 1:
            raise U2rProtocolError("U2r terminal worker has no unique authenticated episode start")


def verify_terminal_report(
    run_directory: Path,
    *,
    model_loader: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Verify an immutable U2r eligibility report and every selected exam case."""

    requested_root = run_directory.expanduser()
    if requested_root.is_symlink():
        raise U2rProtocolError("U2r terminal-report directory is missing or unsafe")
    root = requested_root.resolve()
    if not root.is_dir():
        raise U2rProtocolError("U2r terminal-report directory is missing or unsafe")
    report_path = root / "report.json"
    integrity_path = root / "report.integrity.json"
    report = _read_json(report_path, "U2r terminal report")
    integrity = _read_json(integrity_path, "U2r terminal report integrity")
    eligible = report.get("eligible_for_fresh_confirmation")
    if (
        integrity.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or integrity.get("protocol") != PROTOCOL
        or integrity.get("report") != report_path.name
        or integrity.get("report_sha256") != file_sha256(report_path)
        or report.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or report.get("protocol") != PROTOCOL
        or report.get("kind") != "terminal_stability_eligibility_report"
        or report.get("verdict") not in {"eligible", "failed"}
        or not isinstance(eligible, bool)
        or report.get("fresh_confirmation_opened") is not False
        or report.get("policy_updates_during_reporting") is not False
        or not isinstance(report.get("source"), Mapping)
        or report["source"].get("dirty") is not False
        or not isinstance(report.get("external_preregistration"), Mapping)
        or not isinstance(report.get("exclusions"), Mapping)
        or not isinstance(report.get("controller"), Mapping)
        or not isinstance(report.get("terminal_pair"), Mapping)
    ):
        raise U2rProtocolError("U2r terminal report identity changed")
    progress = report.get("progress")
    if (
        not isinstance(progress, Mapping)
        or int(progress.get("source_child_actions", -1)) != SOURCE_CHILD_ACTIONS
        or int(progress.get("remediation_trained_actions", -1)) != ADDITIONAL_ACTION_BUDGET
        or int(progress.get("child_trained_actions", -1)) != TERMINAL_CHILD_ACTIONS
        or int(progress.get("lifetime_trained_actions", -1)) != TERMINAL_LIFETIME_ACTIONS
        or int(progress.get("optimizer_updates", -1))
        != SOURCE_OPTIMIZER_UPDATES + ADDITIONAL_ACTION_BUDGET // ROLLOUT_TRANSITIONS * PPO_EPOCHS
    ):
        raise U2rProtocolError("U2r terminal report counters changed")
    controller = ControllerState.from_dict(report["controller"])
    if (
        controller.terminal_eligible is not eligible
        or controller.terminal_pair != report.get("terminal_pair")
        or (report["verdict"] == "eligible") is not controller.terminal_eligible
    ):
        raise U2rProtocolError("U2r terminal report verdict changed")
    if len(controller.exam_records) != KEEP_ALL_EXAMS:
        raise U2rProtocolError("U2r terminal report lacks all decision exams")
    prior_record, final_record = controller.exam_records[-2:]
    recomputed = grade_stability_pair(
        prior_record["evaluations"],
        final_record["evaluations"],
        penultimate_allocation_valid=bool(prior_record["allocation_valid"]),
        terminal_allocation_valid=bool(final_record["allocation_valid"]),
        penultimate_recovery=bool(prior_record["recovery"]),
        terminal_recovery=bool(final_record["recovery"]),
        penultimate_child_actions=int(prior_record["child_trained_actions"]),
        terminal_child_actions=int(final_record["child_trained_actions"]),
    )
    expected_terminal_pair = {
        "penultimate_child_actions": int(prior_record["child_trained_actions"]),
        "terminal_child_actions": int(final_record["child_trained_actions"]),
        **recomputed.public_dict(),
    }
    if (
        controller.terminal_eligible is not recomputed.eligible
        or controller.terminal_reasons != list(recomputed.reasons)
        or controller.terminal_pair != expected_terminal_pair
    ):
        raise U2rProtocolError("U2r terminal report stability verdict does not recompute")
    exams = report.get("exams")
    expected_boundaries = [
        SOURCE_CHILD_ACTIONS + index * EVALUATION_INTERVAL for index in range(1, KEEP_ALL_EXAMS + 1)
    ]
    if (
        not isinstance(exams, list)
        or len(exams) != KEEP_ALL_EXAMS
        or [int(item.get("child_trained_actions", -1)) for item in exams] != expected_boundaries
    ):
        raise U2rProtocolError("U2r terminal report exam inventory changed")
    for item, record in zip(exams, controller.exam_records, strict=True):
        if not isinstance(item, Mapping):
            raise U2rProtocolError("U2r terminal report exam entry is invalid")
        checkpoint = _safe_report_artifact(
            root, item.get("checkpoint"), label="U2r report exam checkpoint"
        )
        sidecar = _safe_report_artifact(root, item.get("sidecar"), label="U2r report exam sidecar")
        integrity_file = _safe_report_artifact(
            root, item.get("integrity"), label="U2r report exam integrity"
        )
        cases = _safe_report_artifact(
            root, item.get("case_evidence"), label="U2r report exam cases"
        )
        boundary = int(record["child_trained_actions"])
        expected_checkpoint = root / "checkpoints" / "rolling" / f"exam-{boundary:07d}.zip"
        if (
            checkpoint != expected_checkpoint
            or sidecar != expected_checkpoint.with_suffix(".json")
            or integrity_file != _integrity_path(expected_checkpoint)
            or cases != _case_evidence_path(expected_checkpoint)
            or item.get("checkpoint_sha256") != file_sha256(checkpoint)
            or item.get("sidecar_sha256") != file_sha256(sidecar)
            or item.get("integrity_sha256") != file_sha256(integrity_file)
            or item.get("case_evidence_sha256") != file_sha256(cases)
            or item.get("case_set_sha256") != record["case_evidence"].get("case_set_sha256")
            or item.get("aggregate_sha256s")
            != {
                lesson: evidence.get("aggregate_sha256")
                for lesson, evidence in record["case_evidence"]["per_lesson"].items()
            }
        ):
            raise U2rProtocolError("U2r terminal report exam digest changed")
        _verify_integrity(checkpoint, sidecar)
        _verify_exam_case_document(
            cases,
            run_directory=root,
            binding=record["case_evidence"],
            evaluations=record["evaluations"],
            checkpoint=checkpoint,
            checkpoint_sha256=file_sha256(checkpoint),
            sidecar_sha256=file_sha256(sidecar),
            child_actions=int(record["child_trained_actions"]),
            expected_seeds={
                lesson: tuple(
                    lessons.LESSON_SPECS[lesson].validation_seed_base + index
                    for index in range(EVALUATION_SEED_COUNT)
                )
                for lesson in lessons.LessonId
            },
        )
    evaluations = report.get("evaluations")
    if not isinstance(evaluations, Mapping):
        raise U2rProtocolError("U2r terminal report lacks evaluations")
    evaluations_path = _safe_report_artifact(
        root,
        evaluations.get("path"),
        label="U2r terminal evaluations",
    )
    if (
        set(evaluations) != {"path", "sha256"}
        or evaluations.get("path") != "evaluations.jsonl"
        or evaluations.get("sha256") != file_sha256(evaluations_path)
    ):
        raise U2rProtocolError("U2r terminal evaluations binding changed")
    terminal = report.get("terminal_artifact")
    if not isinstance(terminal, Mapping):
        raise U2rProtocolError("U2r terminal report has no terminal artifact")
    terminal_checkpoint = _safe_report_artifact(
        root, terminal.get("checkpoint"), label="U2r terminal checkpoint"
    )
    terminal_sidecar_path = _safe_report_artifact(
        root, terminal.get("sidecar"), label="U2r terminal sidecar"
    )
    terminal_integrity_path = _safe_report_artifact(
        root, terminal.get("integrity"), label="U2r terminal integrity"
    )
    terminal_integrity = _verify_integrity(terminal_checkpoint, terminal_sidecar_path)
    terminal_sidecar = _read_json(terminal_sidecar_path, "U2r terminal sidecar")
    final_exam_item = exams[-1]
    final_exam_checkpoint = _safe_report_artifact(
        root,
        final_exam_item.get("checkpoint"),
        label="U2r final rolling exam",
    )
    expected_terminal_kind = "terminal_eligible" if eligible else "terminal_failed"
    terminal_progress = terminal_sidecar.get("progress")
    terminal_exam_source = terminal_sidecar.get("exam_source")
    if (
        terminal_checkpoint != root / "checkpoints" / "terminal.zip"
        or terminal_sidecar_path != (root / "checkpoints" / "terminal.json")
        or terminal_integrity_path != (root / "checkpoints" / "terminal.integrity.json")
        or terminal.get("checkpoint_sha256") != terminal_integrity["checkpoint_sha256"]
        or terminal.get("sidecar_sha256") != terminal_integrity["sidecar_sha256"]
        or terminal.get("integrity_sha256") != file_sha256(terminal_integrity_path)
        or terminal.get("model_state") != terminal_sidecar.get("model_state")
        or terminal_integrity["checkpoint_sha256"] != file_sha256(final_exam_checkpoint)
        or terminal_sidecar.get("kind") != expected_terminal_kind
        or terminal_sidecar.get("resume_eligible") is not False
        or not isinstance(terminal_progress, Mapping)
        or int(terminal_progress.get("collected_actions", -1)) != TERMINAL_LIFETIME_ACTIONS
        or int(terminal_progress.get("trained_actions", -1)) != TERMINAL_LIFETIME_ACTIONS
        or int(terminal_progress.get("lifetime_trained_actions", -1)) != TERMINAL_LIFETIME_ACTIONS
        or int(terminal_progress.get("child_trained_actions", -1)) != TERMINAL_CHILD_ACTIONS
        or int(terminal_progress.get("remediation_trained_actions", -1)) != ADDITIONAL_ACTION_BUDGET
        or int(terminal_progress.get("remaining_remediation_actions", -1)) != 0
        or int(terminal_progress.get("optimizer_updates", -1))
        != SOURCE_OPTIMIZER_UPDATES + ADDITIONAL_ACTION_BUDGET // ROLLOUT_TRANSITIONS * PPO_EPOCHS
        or terminal_exam_source
        != {
            "checkpoint": str(final_exam_item["checkpoint"]),
            "checkpoint_sha256": str(final_exam_item["checkpoint_sha256"]),
            "sidecar": str(final_exam_item["sidecar"]),
            "sidecar_sha256": str(final_exam_item["sidecar_sha256"]),
        }
        or terminal_sidecar.get("controller") != report.get("controller")
        or terminal_sidecar.get("source") != report.get("source")
        or terminal_sidecar.get("effective_config") != report.get("effective_config")
        or terminal_sidecar.get("parent") != report.get("parent")
        or terminal_sidecar.get("qualification") != report.get("qualification")
        or terminal_sidecar.get("external_preregistration")
        != report.get("external_preregistration")
        or terminal_sidecar.get("exclusions") != report.get("exclusions")
        or terminal_sidecar.get("storage_mount") != report.get("storage_mount")
        or terminal_sidecar.get("segment") != report.get("segment")
    ):
        raise U2rProtocolError("U2r terminal report artifact binding changed")
    if model_loader is None:
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError as error:
            raise U2rProtocolError(
                "cannot authenticate the U2r terminal model without training dependencies"
            ) from error

        def load_terminal_model(path: Path) -> Any:
            return RecurrentPPO.load(path, device="cpu")

        model_loader = load_terminal_model
    try:
        terminal_model = model_loader(terminal_checkpoint)
    except BaseException as error:
        raise U2rProtocolError("cannot deserialize the U2r terminal model") from error
    model_state = terminal_sidecar.get("model_state")
    if not isinstance(model_state, Mapping):
        raise U2rProtocolError("U2r terminal model-state binding is missing")
    _verify_loaded_model(
        terminal_model,
        build_parser().parse_args([]),
        expected_actions=TERMINAL_LIFETIME_ACTIONS,
        expected_updates=(
            SOURCE_OPTIMIZER_UPDATES + ADDITIONAL_ACTION_BUDGET // ROLLOUT_TRANSITIONS * PPO_EPOCHS
        ),
        expected_policy_sha256=_require_sha256(
            model_state.get("policy_tensor_sha256"),
            "U2r terminal policy",
        ),
        expected_optimizer_sha256=_require_sha256(
            model_state.get("optimizer_state_sha256"),
            "U2r terminal optimizer",
        ),
    )
    _verify_lineage_history(
        report,
        root=root,
        terminal_sidecar=terminal_sidecar,
    )
    return json.loads(json.dumps(report))


def _verify_interruption_for_resume(
    source_run: Path,
    *,
    checkpoint: Path,
    checkpoint_sidecar: Mapping[str, Any],
    expected_config: Mapping[str, Any],
    parent: U2rParent | Mapping[str, Any],
    qualification: frozen_u2.QualificationProvenance | Mapping[str, Any],
    expected_source_commit: str,
    segment_index: int,
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    parent_public = parent.public_dict() if isinstance(parent, U2rParent) else dict(parent)
    qualification_public = (
        qualification.public_dict()
        if isinstance(qualification, frozen_u2.QualificationProvenance)
        else dict(qualification)
    )
    status = _read_json(source_run / "status.json", "U2r resume status")
    status_binding = status.get("interruption")
    if (
        status.get("protocol") != PROTOCOL
        or status.get("phase") not in {"interrupted", "crashed"}
        or not isinstance(status_binding, Mapping)
    ):
        raise U2rProtocolError("U2r resume requires bound interrupted/crashed status")
    interruption_path = source_run / "interruption.json"
    integrity_path = source_run / "interruption.integrity.json"
    document = _read_json(interruption_path, "U2r interruption evidence")
    integrity = _read_json(
        integrity_path,
        "U2r interruption evidence integrity",
    )
    if (
        set(status_binding)
        != {
            "path",
            "sha256",
            "integrity",
            "integrity_sha256",
            "phase",
            "active_workers_sha256",
        }
        or status_binding.get("path") != interruption_path.name
        or status_binding.get("sha256") != file_sha256(interruption_path)
        or status_binding.get("integrity") != integrity_path.name
        or status_binding.get("integrity_sha256") != file_sha256(integrity_path)
        or status_binding.get("phase") != status.get("phase")
        or status_binding.get("active_workers_sha256") != document.get("active_workers_sha256")
        or integrity.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or integrity.get("protocol") != PROTOCOL
        or integrity.get("interruption") != interruption_path.name
        or integrity.get("interruption_sha256") != file_sha256(interruption_path)
        or document.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or document.get("protocol") != PROTOCOL
        or document.get("kind") != "process_interruption_evidence"
        or document.get("phase") != status.get("phase")
        or document.get("resume_permitted") is not True
        or document.get("resume_blocker") is not None
        or document.get("source") != checkpoint_sidecar.get("source")
        or document.get("effective_config") != dict(expected_config)
        or document.get("parent") != parent_public
        or document.get("qualification") != qualification_public
        or document.get("exclusions") != expected_config.get("exclusions")
        or document.get("external_preregistration")
        != expected_config.get("external_preregistration")
        or document.get("storage_mount") != checkpoint_sidecar.get("storage_mount")
        or document.get("segment") != checkpoint_sidecar.get("segment")
    ):
        raise U2rProtocolError("U2r interruption evidence binding changed")
    at_signal = document.get("at_signal")
    resume_boundary = document.get("resume_boundary")
    workers_raw = document.get("active_workers")
    if (
        not isinstance(at_signal, Mapping)
        or not isinstance(resume_boundary, Mapping)
        or not isinstance(workers_raw, list)
        or len(workers_raw) != WORKERS
        or document.get("active_workers_sha256") != _canonical_json_sha256(workers_raw)
    ):
        raise U2rProtocolError("U2r interruption episode evidence changed")
    checkpoint_sidecar_path = checkpoint.with_suffix(".json")
    checkpoint_integrity_path = _integrity_path(checkpoint)
    expected_resume_boundary = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "sidecar": str(checkpoint_sidecar_path),
        "sidecar_sha256": file_sha256(checkpoint_sidecar_path),
        "integrity": str(checkpoint_integrity_path),
        "integrity_sha256": file_sha256(checkpoint_integrity_path),
        "model_state": checkpoint_sidecar.get("model_state"),
        "progress": checkpoint_sidecar.get("progress"),
    }
    if dict(resume_boundary) != expected_resume_boundary:
        raise U2rProtocolError("U2r interruption resume boundary changed")
    resolution = document.get("resolution")
    if not isinstance(resolution, Mapping):
        raise U2rProtocolError("U2r interruption resolution is missing")
    try:
        signal_trained = int(at_signal["trained_actions"])
        signal_child = int(at_signal["child_trained_actions"])
        signal_collected = int(at_signal["collected_actions"])
        signal_updates = int(at_signal["optimizer_updates"])
        untrained_actions = int(at_signal["untrained_collected_actions"])
        untrained_per_worker = int(at_signal["untrained_transitions_per_worker"])
        boundary_progress = resume_boundary["progress"]
        boundary_trained = int(boundary_progress["trained_actions"])
        boundary_child = int(boundary_progress["child_trained_actions"])
        boundary_updates = int(boundary_progress["optimizer_updates"])
        resolved_trained = int(resolution["resolved_trained_actions"])
        resolved_child = int(resolution["resolved_child_trained_actions"])
        resolved_updates = int(resolution["resolved_optimizer_updates"])
        safe_trained = int(resolution["safe_boundary_trained_actions"])
        recovered_actions = int(resolution["recovered_complete_actions"])
        recovered_updates = int(resolution["recovered_complete_optimizer_updates"])
        discarded_post_safe = int(resolution["discarded_post_safe_rollout_actions"])
        unpublished_trained = int(resolution["unpublished_trained_actions"])
        resolution_untrained = int(resolution["untrained_collected_actions"])
        unpublished_complete = int(resolution["unpublished_complete_actions"])
    except (KeyError, TypeError, ValueError) as error:
        raise U2rProtocolError("U2r interruption counters are incomplete") from error
    initial_classification = at_signal.get("pending_optimizer_boundary")
    final_classification = resolution.get("pending_optimizer_boundary")
    pending_actions = signal_collected - signal_trained
    trained_remediation = signal_trained - SOURCE_LIFETIME_ACTIONS
    expected_trained_updates = (
        SOURCE_OPTIMIZER_UPDATES + trained_remediation // ROLLOUT_TRANSITIONS * PPO_EPOCHS
    )
    pending_updates = signal_updates - expected_trained_updates
    if (
        signal_collected < signal_trained
        or signal_trained < SOURCE_LIFETIME_ACTIONS
        or signal_child != signal_trained - SOURCE_U1_INHERITED_ACTIONS
        or signal_updates < SOURCE_OPTIMIZER_UPDATES
        or untrained_actions != untrained_per_worker * WORKERS
        or untrained_actions < 0
        or (
            initial_classification == "partial"
            and untrained_actions != signal_collected - signal_trained
        )
        or (initial_classification in {"none", "complete"} and untrained_actions != 0)
        or initial_classification not in {"none", "partial", "complete"}
        or final_classification
        not in {
            "none",
            "partial",
            "complete_recovered",
            "complete_recovery_failed",
        }
        or boundary_trained != safe_trained
        or boundary_child != boundary_trained - SOURCE_U1_INHERITED_ACTIONS
        or resolved_child != resolved_trained - SOURCE_U1_INHERITED_ACTIONS
        or discarded_post_safe != signal_collected - safe_trained
        or unpublished_trained != resolved_trained - safe_trained
        or resolution_untrained != untrained_actions
        or resolution.get("active_environment_state_disposition") != "discarded_at_process_boundary"
        or resolution.get("policy_recurrent_state_disposition") != "discarded_at_process_boundary"
    ):
        raise U2rProtocolError("U2r interruption counters changed")
    if initial_classification == "none":
        coherent = (
            final_classification == "none"
            and pending_actions == 0
            and pending_updates == 0
            and resolved_trained == signal_trained
            and safe_trained <= signal_trained
            and resolved_updates == signal_updates
            and boundary_updates <= signal_updates
            and recovered_actions == recovered_updates == 0
            and unpublished_complete == 0
            and resolution.get("recovery_error") is None
        )
    elif initial_classification == "partial":
        coherent = (
            final_classification == "partial"
            and pending_actions >= 0
            and resolved_trained == signal_trained
            and resolved_updates == signal_updates
            and safe_trained <= resolved_trained
            and boundary_updates <= resolved_updates
            and recovered_actions == recovered_updates == 0
            and unpublished_complete == 0
            and resolution.get("recovery_error") is None
        )
    elif final_classification == "complete_recovered":
        coherent = (
            pending_actions > 0
            and pending_actions % ROLLOUT_TRANSITIONS == 0
            and pending_updates == pending_actions // ROLLOUT_TRANSITIONS * PPO_EPOCHS
            and recovered_actions == pending_actions
            and recovered_updates == pending_updates
            and resolved_trained == signal_collected == safe_trained
            and resolved_updates == signal_updates == boundary_updates
            and unpublished_complete == 0
            and resolution.get("recovery_error") is None
        )
    else:
        coherent = (
            final_classification == "complete_recovery_failed"
            and pending_actions > 0
            and pending_actions % ROLLOUT_TRANSITIONS == 0
            and pending_updates == pending_actions // ROLLOUT_TRANSITIONS * PPO_EPOCHS
            and recovered_actions == recovered_updates == 0
            and safe_trained <= resolved_trained <= signal_collected
            and boundary_updates <= resolved_updates <= signal_updates
            and unpublished_complete == unpublished_trained
            and isinstance(resolution.get("recovery_error"), str)
            and bool(resolution.get("recovery_error"))
        )
    if not coherent:
        raise U2rProtocolError("U2r interruption resolution is incoherent")
    expected_streams = [
        seed + segment_index * SEGMENT_SEED_OFFSET for seed in REMEDIATION_WORKER_STREAMS
    ]
    workers = tuple(
        _validate_resume_worker(
            workers_raw[index],
            worker_index=index,
            worker_stream=expected_streams[index],
            source_commit=expected_source_commit,
            lifetime_actions=signal_trained,
            child_actions=signal_child,
            untrained_transitions=int(at_signal.get("untrained_transitions_per_worker", -1)),
        )
        for index in range(WORKERS)
    )
    ledger = document.get("episode_start_ledger")
    ledger_path = source_run / "episode-starts.jsonl"
    if (
        not isinstance(ledger, Mapping)
        or ledger.get("path") != ledger_path.name
        or ledger.get("sha256") != file_sha256(ledger_path)
        or {
            key: ledger.get(key)
            for key in (
                "record_count",
                "episode_start_count",
                "interruption_abandonment_count",
                "resume_abandonment_count",
            )
        }
        != _jsonl_inventory(ledger_path)
        or int(ledger.get("interruption_abandonment_count", -1)) < WORKERS
    ):
        raise U2rProtocolError("U2r interruption episode ledger changed")
    ledger_records = _read_jsonl(
        ledger_path,
        "U2r interruption episode ledger",
    )
    start_identity_fields = (
        "worker_index",
        "worker_stream",
        "episode_ordinal",
        "seed",
        "lesson_id",
        "layout_sha256",
        "geometry_sha256",
        "worker_transition_at_start",
        "episode_started_at",
    )
    for worker in workers:
        matching_starts = [
            record
            for record in ledger_records
            if record.get("type") == "episode_start"
            and all(record.get(field) == worker.get(field) for field in start_identity_fields)
        ]
        if len(matching_starts) != 1:
            raise U2rProtocolError(
                "U2r interruption worker has no unique authenticated episode start"
            )
        matching_abandonments = [
            record
            for record in ledger_records
            if record.get("type") == "episode_abandoned_at_interruption"
            and record.get("interruption_phase") == document.get("phase")
            and all(
                record.get(field) == worker.get(field)
                for field in (
                    "worker_index",
                    "worker_stream",
                    "episode_ordinal",
                    "seed",
                    "lesson_id",
                    "layout_sha256",
                    "geometry_sha256",
                    "elapsed_steps",
                )
            )
        ]
        if len(matching_abandonments) != 1:
            raise U2rProtocolError("U2r interruption ledger does not identify every active worker")
    return json.loads(json.dumps(document)), workers


def load_resume(
    checkpoint: Path,
    *,
    expected_config: Mapping[str, Any],
    parent: U2rParent,
    qualification: frozen_u2.QualificationProvenance,
    expected_source_commit: str,
) -> ResumeBundle:
    """Load only a complete U2r post-decision policy/optimizer bundle."""

    requested_archive = checkpoint.expanduser()
    if requested_archive.suffix != ".zip":
        requested_archive = requested_archive.with_suffix(".zip")
    if requested_archive.is_symlink():
        raise U2rProtocolError("U2r resume checkpoint cannot be a symlink")
    archive = requested_archive.resolve()
    sidecar_path = archive.with_suffix(".json")
    sidecar = _read_json(sidecar_path, "U2r resume sidecar")
    _verify_integrity(archive, sidecar_path)
    if (
        sidecar.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or sidecar.get("protocol") != PROTOCOL
        or sidecar.get("resume_eligible") is not True
        or sidecar.get("kind") not in {"initial", "resume", "latest"}
        or sidecar.get("checkpoint_sha256") != file_sha256(archive)
        or sidecar.get("effective_config") != dict(expected_config)
        or sidecar.get("parent") != parent.public_dict()
        or sidecar.get("qualification") != qualification.public_dict()
        or sidecar.get("exclusions") != expected_config.get("exclusions")
        or sidecar.get("external_preregistration")
        != expected_config.get("external_preregistration")
        or not isinstance(sidecar.get("storage_mount"), Mapping)
    ):
        raise U2rProtocolError("checkpoint is not a matching resumable U2r artifact")
    source = sidecar.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("dirty") is not False
        or source.get("commit") != expected_source_commit
    ):
        raise U2rProtocolError("U2r resume source provenance changed")
    try:
        curriculum = lessons.CurriculumState.from_dict(sidecar["curriculum"])
        controller = ControllerState.from_dict(sidecar["controller"])
        scheduler_state = dict(sidecar["scheduler"])
        progress = sidecar["progress"]
        lifetime = int(progress["lifetime_trained_actions"])
        child = int(progress["child_trained_actions"])
        remediation = int(progress["remediation_trained_actions"])
        remaining = int(progress["remaining_remediation_actions"])
        collected = int(progress["collected_actions"])
        trained = int(progress["trained_actions"])
        n_updates = int(progress["optimizer_updates"])
        segment = sidecar["segment"]
        segment_index = int(segment["index"])
        last_allocation = sidecar.get("last_completed_allocation")
        last_allocation_valid = sidecar.get("last_completed_allocation_valid")
        active_workers_raw = sidecar["active_workers"]
        model_state = sidecar["model_state"]
        resume_semantics = sidecar["resume_semantics"]
        latest_case_evidence = sidecar.get("latest_exam_case_evidence")
    except (KeyError, TypeError, ValueError) as error:
        raise U2rProtocolError(f"invalid U2r resume sidecar: {error}") from error
    if (
        collected != trained
        or trained != lifetime
        or remediation != child - SOURCE_CHILD_ACTIONS
        or lifetime != SOURCE_LIFETIME_ACTIONS + remediation
        or remaining != ADDITIONAL_ACTION_BUDGET - remediation
        or not 0 <= remediation < ADDITIONAL_ACTION_BUDGET
        or remediation % ROLLOUT_TRANSITIONS
        or child != SOURCE_CHILD_ACTIONS + remediation
        or child > TERMINAL_CHILD_ACTIONS
    ):
        raise U2rProtocolError("U2r resume action counters are inconsistent")
    expected_updates = SOURCE_OPTIMIZER_UPDATES + remediation // ROLLOUT_TRANSITIONS * PPO_EPOCHS
    if n_updates != expected_updates:
        raise U2rProtocolError("U2r optimizer counter disagrees with trained actions")
    source_run = archive.parent.parent
    expected_source_run = DEFAULT_RUN_ROOT.resolve() / _segment_run_name(segment_index)
    if (
        source_run.is_symlink()
        or source_run != expected_source_run
        or archive.parent.name != "checkpoints"
        or archive.name not in {"initial.zip", "resume.zip", "latest-safe.zip"}
    ):
        raise U2rProtocolError("U2r resume checkpoint is outside its canonical segment")
    expected_algorithm_seed = REMEDIATION_ALGORITHM_SEED + segment_index * SEGMENT_SEED_OFFSET
    expected_workers = [
        seed + segment_index * SEGMENT_SEED_OFFSET for seed in REMEDIATION_WORKER_STREAMS
    ]
    if (
        segment_index < 0
        or int(segment.get("lineage_source_child_seed", -1)) != SOURCE_CHILD_SEED
        or int(segment.get("segment_algorithm_seed", -1)) != expected_algorithm_seed
        or segment.get("segment_worker_streams") != expected_workers
    ):
        raise U2rProtocolError("U2r resume segment identity changed")
    if segment_index == 0:
        if any(
            segment.get(field) is not None
            for field in (
                "resume_checkpoint",
                "resume_checkpoint_sha256",
                "resume_model_state",
                "resume_source_segment_index",
            )
        ):
            raise U2rProtocolError("initial U2r segment unexpectedly has resume provenance")
    else:
        linked_checkpoint = Path(str(segment.get("resume_checkpoint", "")))
        linked_state = segment.get("resume_model_state")
        if (
            type(segment.get("resume_source_segment_index")) is not int
            or segment.get("resume_source_segment_index") != segment_index - 1
            or not linked_checkpoint.is_absolute()
            or linked_checkpoint.is_symlink()
            or not linked_checkpoint.is_file()
            or linked_checkpoint.parent.parent
            != DEFAULT_RUN_ROOT.resolve() / _segment_run_name(segment_index - 1)
            or _require_sha256(
                segment.get("resume_checkpoint_sha256"),
                "U2r segment resume checkpoint",
            )
            != file_sha256(linked_checkpoint)
            or not isinstance(linked_state, Mapping)
            or set(linked_state) != {"policy_tensor_sha256", "optimizer_state_sha256"}
        ):
            raise U2rProtocolError("U2r resume segment link changed")
        _require_sha256(
            linked_state.get("policy_tensor_sha256"),
            "U2r segment inherited policy",
        )
        _require_sha256(
            linked_state.get("optimizer_state_sha256"),
            "U2r segment inherited optimizer",
        )
    if not isinstance(active_workers_raw, list) or len(active_workers_raw) != WORKERS:
        raise U2rProtocolError("U2r resume lacks four active-worker evidence records")
    try:
        _checkpoint_workers = tuple(
            _validate_resume_worker(
                active_workers_raw[index],
                worker_index=index,
                worker_stream=expected_workers[index],
                source_commit=expected_source_commit,
                lifetime_actions=lifetime,
                child_actions=child,
            )
            for index in range(WORKERS)
        )
    except (IndexError, TypeError) as error:
        raise U2rProtocolError("U2r resume lacks ordered active-worker evidence") from error
    if not isinstance(model_state, Mapping):
        raise U2rProtocolError("U2r resume has no explicit model-state digests")
    if (
        not isinstance(resume_semantics, Mapping)
        or resume_semantics.get("untrained_transitions") != 0
        or resume_semantics.get("exact_environment_resume_supported") is not False
        or resume_semantics.get("environment_state_disposition") != "discarded_on_process_resume"
        or resume_semantics.get("policy_recurrent_state_disposition")
        != "discarded_on_process_resume"
    ):
        raise U2rProtocolError("U2r resume semantics changed")
    policy_digest = _require_sha256(model_state.get("policy_tensor_sha256"), "U2r resume policy")
    optimizer_digest = _require_sha256(
        model_state.get("optimizer_state_sha256"), "U2r resume optimizer"
    )
    completed_exams = remediation // EVALUATION_INTERVAL
    expected_last_decision = (
        SOURCE_CHILD_ACTIONS + completed_exams * EVALUATION_INTERVAL if completed_exams else None
    )
    if controller.last_decision_child_actions != expected_last_decision:
        raise U2rProtocolError("U2r resume controller is not the exact completed-exam prefix")
    _validated_resume_scheduler(
        curriculum,
        scheduler_state,
        child_actions=child,
        last_decision_child_actions=controller.last_decision_child_actions,
    )
    if controller.last_decision_child_actions is not None:
        allocation_window = (
            last_allocation.get("window_transitions")
            if isinstance(last_allocation, Mapping)
            else None
        )
        if (
            not isinstance(last_allocation, Mapping)
            or not isinstance(last_allocation_valid, bool)
            or not isinstance(allocation_window, Mapping)
            or sum(int(value) for value in allocation_window.values()) != EVALUATION_INTERVAL
            or controller.exam_records[-1].get("allocation_valid") is not last_allocation_valid
            or latest_case_evidence != controller.exam_records[-1].get("case_evidence")
        ):
            raise U2rProtocolError("U2r resume lacks its completed allocation")
    elif last_allocation is not None or last_allocation_valid is not None:
        raise U2rProtocolError("U2r resume has completed-allocation evidence without an exam")
    seed_access = qualification.seed_access()
    expected_validation_seeds = {
        lesson: tuple(lessons.validation_seeds(lesson, access=seed_access))
        for lesson in lessons.LessonId
    }
    for record in controller.exam_records:
        boundary = int(record["child_trained_actions"])
        exam_checkpoint = source_run / "checkpoints" / "rolling" / f"exam-{boundary:07d}.zip"
        exam_sidecar = exam_checkpoint.with_suffix(".json")
        _verify_integrity(exam_checkpoint, exam_sidecar)
        binding = record["case_evidence"]
        relative = Path(str(binding.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise U2rProtocolError("U2r case-evidence path escapes its segment")
        case_path = (source_run / relative).resolve()
        _verify_exam_case_document(
            case_path,
            run_directory=source_run,
            binding=binding,
            evaluations=record["evaluations"],
            checkpoint=exam_checkpoint,
            checkpoint_sha256=file_sha256(exam_checkpoint),
            sidecar_sha256=file_sha256(exam_sidecar),
            child_actions=boundary,
            expected_seeds=expected_validation_seeds,
        )
    if curriculum.mastered or controller.terminal_eligible is not None:
        raise U2rProtocolError("terminal U2r evidence is not resumable")
    _build_parent_baseline_inventory(
        DEFAULT_RUN_ROOT.resolve() / _segment_run_name(0),
        cohort_root=DEFAULT_RUN_ROOT.resolve(),
    )
    interruption_evidence, interruption_workers = _verify_interruption_for_resume(
        source_run,
        checkpoint=archive,
        checkpoint_sidecar=sidecar,
        expected_config=expected_config,
        parent=parent,
        qualification=qualification,
        expected_source_commit=expected_source_commit,
        segment_index=segment_index,
    )
    return ResumeBundle(
        checkpoint=archive,
        sidecar=sidecar,
        curriculum=curriculum,
        controller=controller,
        scheduler_state=scheduler_state,
        lifetime_trained=lifetime,
        child_trained=child,
        remediation_trained=remediation,
        n_updates=n_updates,
        segment_index=segment_index + 1,
        last_completed_allocation=(
            dict(last_allocation) if isinstance(last_allocation, Mapping) else None
        ),
        last_completed_allocation_valid=last_allocation_valid,
        active_workers=interruption_workers,
        active_worker_evidence_source="interruption.json",
        interruption_evidence=interruption_evidence,
        policy_tensor_sha256=policy_digest,
        optimizer_state_sha256=optimizer_digest,
    )


def _carry_forward_exams(
    resume: ResumeBundle,
    *,
    run_directory: Path,
    storage_guard: Callable[[int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy every prior immutable exam bundle into a resumed lineage segment."""

    source_run = resume.checkpoint.parent.parent.resolve()
    source_rolling = source_run / "checkpoints" / "rolling"
    destination_rolling = run_directory / "checkpoints" / "rolling"
    carried: list[dict[str, Any]] = []
    records_by_boundary = {
        int(record["child_trained_actions"]): record for record in resume.controller.exam_records
    }
    sources = sorted(source_rolling.glob("exam-*.zip"))
    try:
        measured_boundaries = [int(source.stem.removeprefix("exam-")) for source in sources]
    except ValueError as error:
        raise U2rProtocolError("resume chain has an invalid U2r exam name") from error
    expected_count = resume.remediation_trained // EVALUATION_INTERVAL
    expected_boundaries = [
        SOURCE_CHILD_ACTIONS + index * EVALUATION_INTERVAL for index in range(1, expected_count + 1)
    ]
    if measured_boundaries != expected_boundaries:
        raise U2rProtocolError("resume chain exams are not the exact contiguous completed prefix")
    for source, boundary in zip(sources, measured_boundaries, strict=True):
        sidecar = source.with_suffix(".json")
        integrity = _integrity_path(source)
        source_cases = _case_evidence_path(source)
        _verify_integrity(source, sidecar)
        record = records_by_boundary.get(boundary)
        binding = record.get("case_evidence") if isinstance(record, Mapping) else None
        if not isinstance(binding, Mapping) or binding.get("file_sha256") != file_sha256(
            source_cases
        ):
            raise U2rProtocolError("carried U2r exam case evidence changed")
        destination = destination_rolling / source.name
        destination_sidecar = destination.with_suffix(".json")
        destination_integrity = _integrity_path(destination)
        destination_cases = _case_evidence_path(destination)
        if any(
            path.exists() or path.is_symlink()
            for path in (
                destination,
                destination_sidecar,
                destination_integrity,
                destination_cases,
            )
        ):
            raise U2rProtocolError(f"refusing to overwrite carried exam {destination}")
        bundle_bytes = sum(
            path.stat().st_size for path in (source, sidecar, integrity, source_cases)
        )
        storage_guard(bundle_bytes)
        destination_rolling.mkdir(parents=True, exist_ok=True)
        published: list[Path] = []
        try:
            for old, new in (
                (source, destination),
                (sidecar, destination_sidecar),
                (integrity, destination_integrity),
                (source_cases, destination_cases),
            ):
                atomic_copy_file(old, new)
                published.append(new)
            _verify_integrity(destination, destination_sidecar)
            _verify_exam_case_document(
                destination_cases,
                run_directory=run_directory,
                binding=binding,
                evaluations=record["evaluations"],
                checkpoint=destination,
                checkpoint_sha256=file_sha256(destination),
                sidecar_sha256=file_sha256(destination_sidecar),
                child_actions=boundary,
                expected_seeds=None,
            )
            storage_guard(0)
        except BaseException:
            for path in reversed(published):
                path.unlink(missing_ok=True)
            raise
        carried.append(
            {
                "path": str(destination.relative_to(run_directory)),
                "child_trained_actions": boundary,
                "checkpoint_sha256": file_sha256(destination),
                "sidecar_sha256": file_sha256(destination_sidecar),
                "integrity_sha256": file_sha256(destination_integrity),
                "case_evidence_sha256": file_sha256(destination_cases),
                "case_set_sha256": binding["case_set_sha256"],
            }
        )
    return carried


def _record_abandoned_resume_workers(
    run_directory: Path,
    resume: ResumeBundle,
) -> list[dict[str, Any]]:
    """Durably disclose the four partial episodes a process resume cannot restore."""

    abandoned: list[dict[str, Any]] = []
    timestamp = utc_now()
    source_interruption = resume.checkpoint.parent.parent / "interruption.json"
    for raw in sorted(
        resume.active_workers,
        key=lambda item: int(item.get("worker_index", -1)),
    ):
        record = dict(raw) | {
            "abandoned_on_resume": True,
            "abandoned_at": timestamp,
            "abandonment_reason": RESUME_ABANDONMENT_REASON,
            "active_worker_evidence_source": (resume.active_worker_evidence_source),
            "source_interruption": str(source_interruption.resolve()),
            "source_interruption_sha256": file_sha256(source_interruption),
            "source_interruption_active_workers_sha256": (
                resume.interruption_evidence["active_workers_sha256"]
            ),
        }
        append_jsonl(
            run_directory / "episode-starts.jsonl",
            {
                "timestamp": timestamp,
                "type": "episode_abandoned_on_resume",
                **record,
            },
        )
        abandoned.append(record)
    if len(abandoned) != WORKERS or {
        int(item.get("worker_index", -1)) for item in abandoned
    } != set(range(WORKERS)):
        raise U2rProtocolError("U2r resume abandonment evidence does not contain four workers")
    return abandoned


class _CallbackFactory:
    """Build the U2r callback while reusing only U2's artifact mechanics."""

    @staticmethod
    def create(base_callback: Any) -> type[Any]:
        frozen_base = frozen_u2._CallbackFactory.create(base_callback)

        class U2rCallback(frozen_base):
            def __init__(
                self,
                *,
                active_workers: Sequence[EpisodeEvidenceWrapper],
                exclusions: ForbiddenLayoutEvidence,
                storage_mount: Mapping[str, Any],
                **kwargs: Any,
            ) -> None:
                self.active_worker_recorders = tuple(active_workers)
                self.exclusions = exclusions
                self.storage_mount = dict(storage_mount)
                if len(self.active_worker_recorders) != WORKERS:
                    raise U2rProtocolError("U2r requires exactly four active-worker recorders")
                super().__init__(**kwargs)
                self.latest_exam_case_evidence = (
                    dict(self.controller.exam_records[-1]["case_evidence"])
                    if self.controller.exam_records
                    else None
                )
                self.terminal_report_evidence: dict[str, Any] | None = None
                self.interruption_evidence: dict[str, Any] | None = None

            @property
            def remediation_trained_actions(self) -> int:
                return self.child_trained_actions - SOURCE_CHILD_ACTIONS

            def _on_training_start(self) -> None:
                if int(self.model.num_timesteps) != self.trained_actions:
                    raise U2rProtocolError(
                        "model and U2r controller start-action boundaries differ"
                    )
                if int(self.model._n_updates) != self.last_updates:
                    raise U2rProtocolError("model and U2r controller optimizer boundaries differ")
                segment_index = int(self.segment.get("index", -1))
                inherited_model_state = self.segment.get("resume_model_state")
                current_model_state = {
                    "policy_tensor_sha256": (state_digests.policy_tensor_sha256(self.model)),
                    "optimizer_state_sha256": (state_digests.optimizer_state_sha256(self.model)),
                }
                if segment_index == 0:
                    if any(
                        self.segment.get(field) is not None
                        for field in (
                            "resume_checkpoint",
                            "resume_checkpoint_sha256",
                            "resume_model_state",
                            "resume_source_segment_index",
                        )
                    ):
                        raise U2rProtocolError("initial U2r segment has resume provenance")
                elif (
                    int(self.segment.get("resume_source_segment_index", -1)) != segment_index - 1
                    or not isinstance(inherited_model_state, Mapping)
                    or dict(inherited_model_state) != current_model_state
                    or _require_sha256(
                        self.segment.get("resume_checkpoint_sha256"),
                        "U2r inherited checkpoint",
                    )
                    != file_sha256(Path(str(self.segment["resume_checkpoint"])))
                ):
                    raise U2rProtocolError("U2r resumed model does not match its inherited state")
                initial = self._publish_live_archive(
                    "initial",
                    kind="initial",
                    resume_eligible=True,
                    replace=False,
                )
                self.latest_safe_checkpoint = initial
                self.latest_safe_trained_actions = self.trained_actions
                if int(self.segment.get("index", -1)) == 0:
                    self._write_status("qualifying_baseline")
                    self._evaluate_archive(
                        initial,
                        trigger="diagnostic_parent_baseline",
                        decide=False,
                    )
                else:
                    self._write_status("starting")
                    append_jsonl(
                        self.run_directory / "events.jsonl",
                        {
                            "timestamp": utc_now(),
                            "type": "resume_segment_started",
                            "segment_index": int(self.segment["index"]),
                            "diagnostic_evaluation_performed": False,
                            "reason": (
                                "the frozen parent baseline is measured once in "
                                "segment zero; resumes preserve its bytes without "
                                "adding development observations"
                            ),
                            "trained_actions": self.trained_actions,
                            "child_trained_actions": self.child_trained_actions,
                        },
                    )
                self._write_status("training")

            def _active_worker_evidence(self) -> list[dict[str, Any]]:
                records = [
                    recorder.evidence_state()
                    | {
                        "checkpoint_source_commit": self.source.get("commit"),
                        "checkpoint_lifetime_trained_actions": self.trained_actions,
                        "checkpoint_child_trained_actions": (self.child_trained_actions),
                    }
                    for recorder in self.active_worker_recorders
                ]
                if {int(item["worker_index"]) for item in records} != set(range(WORKERS)):
                    raise U2rProtocolError("U2r active-worker checkpoint evidence is incomplete")
                return sorted(records, key=lambda item: int(item["worker_index"]))

            def _sidecar(
                self,
                *,
                kind: str,
                checkpoint_sha256: str,
                resume_eligible: bool,
                exam_source: Mapping[str, Any] | None = None,
            ) -> dict[str, Any]:
                collected = int(self.model.num_timesteps)
                remediation = self.remediation_trained_actions
                return {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "created_at": utc_now(),
                    "kind": kind,
                    "resume_eligible": resume_eligible,
                    "checkpoint_sha256": checkpoint_sha256,
                    "source": dict(self.source),
                    "effective_config": dict(self.effective_config),
                    "parent": self.parent.public_dict(),
                    "qualification": self.qualification.public_dict(),
                    "exclusions": self.exclusions.public_dict(),
                    "external_preregistration": self.effective_config.get(
                        "external_preregistration"
                    ),
                    "storage_mount": dict(self.storage_mount),
                    "model_state": {
                        "policy_tensor_sha256": (state_digests.policy_tensor_sha256(self.model)),
                        "optimizer_state_sha256": (
                            state_digests.optimizer_state_sha256(self.model)
                        ),
                    },
                    "resume_semantics": {
                        "untrained_transitions": 0,
                        "exact_environment_resume_supported": False,
                        "environment_state_disposition": ("discarded_on_process_resume"),
                        "environment_rng_state_disposition": (
                            "discarded_on_process_resume; next segment uses a frozen fresh stream"
                        ),
                        "policy_recurrent_state_disposition": ("discarded_on_process_resume"),
                    },
                    "curriculum": self.state.public_dict(),
                    "controller": self.controller.public_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "last_completed_allocation": self.last_completed_allocation,
                    "last_completed_allocation_valid": (self.last_completed_allocation_valid),
                    "active_workers": self._active_worker_evidence(),
                    "latest_exam_case_evidence": (
                        dict(self.latest_exam_case_evidence)
                        if self.latest_exam_case_evidence is not None
                        else None
                    ),
                    "progress": {
                        "collected_actions": collected,
                        "trained_actions": self.trained_actions,
                        "lifetime_trained_actions": self.trained_actions,
                        "inherited_trained_actions": SOURCE_U1_INHERITED_ACTIONS,
                        "source_child_actions": SOURCE_CHILD_ACTIONS,
                        "child_trained_actions": self.child_trained_actions,
                        "remediation_trained_actions": remediation,
                        "remaining_remediation_actions": (ADDITIONAL_ACTION_BUDGET - remediation),
                        "segment_trained_actions": (
                            self.trained_actions - self.segment_start_actions
                        ),
                        "optimizer_updates": int(self.model._n_updates),
                    },
                    "segment": dict(self.segment),
                    "exam_source": (dict(exam_source) if exam_source is not None else None),
                }

            def _write_bundle_sidecar(
                self,
                checkpoint: Path,
                *,
                kind: str,
                resume_eligible: bool,
                exam_source: Mapping[str, Any] | None = None,
            ) -> Mapping[str, Any]:
                sidecar_path = checkpoint.with_suffix(".json")
                sidecar = self._sidecar(
                    kind=kind,
                    checkpoint_sha256=file_sha256(checkpoint),
                    resume_eligible=resume_eligible,
                    exam_source=exam_source,
                )
                atomic_write_json(sidecar_path, sidecar)
                return _write_integrity(checkpoint, sidecar_path)

            def _process_optimized_boundary(self) -> None:
                collected = int(self.model.num_timesteps)
                updates = int(self.model._n_updates)
                delta_actions = collected - self.trained_actions
                delta_updates = updates - self.last_updates
                if (
                    delta_actions <= 0
                    or delta_actions % ROLLOUT_TRANSITIONS
                    or delta_updates != delta_actions // ROLLOUT_TRANSITIONS * PPO_EPOCHS
                ):
                    raise U2rProtocolError("U2r observed an incomplete PPO optimizer boundary")
                self.trained_actions = collected
                self.last_updates = updates
                if not 0 <= self.remediation_trained_actions <= ADDITIONAL_ACTION_BUDGET:
                    raise U2rProtocolError("U2r crossed its frozen action ceiling")
                self._record_optimizer_metrics(delta_updates)
                if self.trained_actions >= self.next_evaluation:
                    if self.child_trained_actions % self.evaluation_interval:
                        raise U2rProtocolError("U2r exam was reached away from a frozen boundary")
                    exam = self._publish_live_archive(
                        f"rolling/exam-{self.child_trained_actions:07d}",
                        kind="exam",
                        resume_eligible=False,
                        replace=False,
                    )
                    self.latest_exam_checkpoint = exam
                    self._evaluate_archive(exam, trigger="scheduled", decide=True)
                    self.next_evaluation = self._next_child_boundary(
                        self.trained_actions, self.evaluation_interval
                    )
                if self.child_trained_actions < TERMINAL_CHILD_ACTIONS:
                    latest = self._publish_live_archive(
                        "latest-safe",
                        kind="latest",
                        resume_eligible=True,
                        replace=True,
                    )
                    self.latest_safe_checkpoint = latest
                    self.latest_safe_trained_actions = self.trained_actions
                else:
                    self.latest_safe_checkpoint = self.latest_exam_checkpoint
                    self.latest_safe_trained_actions = self.trained_actions
                self._write_status(
                    "eligible"
                    if self.controller.terminal_eligible is True
                    else ("failed" if self.controller.terminal_eligible is False else "training")
                )

            def _publish_exam_case_evidence(
                self,
                exam: Path,
                *,
                exam_sidecar_sha256: str,
                evaluations: Mapping[lessons.LessonId, Any],
                case_records: Sequence[Mapping[str, Any]],
                expected_seeds: Mapping[lessons.LessonId, Sequence[int]],
            ) -> dict[str, Any]:
                path = _case_evidence_path(exam)
                if path.exists() or path.is_symlink():
                    raise U2rProtocolError(f"refusing to overwrite immutable U2r cases: {path}")
                summary = verify_exam_case_evidence(
                    evaluations,
                    case_records,
                    expected_seeds=expected_seeds,
                )
                document = {
                    "schema_version": CASE_EVIDENCE_SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "kind": "immutable_exam_cases",
                    "checkpoint": exam.name,
                    "checkpoint_sha256": file_sha256(exam),
                    "checkpoint_sidecar_sha256": exam_sidecar_sha256,
                    "child_trained_actions": self.child_trained_actions,
                    "summary": summary,
                    "records": [dict(record) for record in case_records],
                }
                anticipated = len(
                    json.dumps(
                        document,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                )
                self._guard_storage(anticipated)
                atomic_write_json(path, document)
                binding = summary | {
                    "path": str(path.relative_to(self.run_directory)),
                    "file_sha256": file_sha256(path),
                    "checkpoint": str(exam.relative_to(self.run_directory)),
                    "checkpoint_sha256": file_sha256(exam),
                    "checkpoint_sidecar_sha256": exam_sidecar_sha256,
                    "child_trained_actions": self.child_trained_actions,
                }
                _verify_exam_case_document(
                    path,
                    run_directory=self.run_directory,
                    binding=binding,
                    evaluations=evaluations,
                    checkpoint=exam,
                    checkpoint_sha256=file_sha256(exam),
                    sidecar_sha256=exam_sidecar_sha256,
                    child_actions=self.child_trained_actions,
                    expected_seeds=expected_seeds,
                )
                self._guard_storage()
                return binding

            def _evaluate_archive(
                self,
                exam: Path,
                *,
                trigger: str,
                decide: bool,
            ) -> None:
                sidecar_path = exam.with_suffix(".json")
                exam_sidecar = _read_json(sidecar_path, "immutable U2r exam sidecar")
                integrity = _verify_integrity(exam, sidecar_path)
                archive_before = file_sha256(exam)
                sidecar_before = file_sha256(sidecar_path)
                graded_model = self._load_exam_model(exam, exam_sidecar)
                allocation_before = self.scheduler.snapshot()
                allocation_transitions = sum(
                    int(value) for value in allocation_before.get("window_transitions", {}).values()
                )
                allocation_complete = not decide or allocation_transitions == EVALUATION_INTERVAL
                allocation_ok = allocation_complete and self.scheduler.allocation_within()
                evaluations: dict[lessons.LessonId, Any] = {}
                case_records: list[dict[str, Any]] = []
                expected_seeds: dict[lessons.LessonId, tuple[int, ...]] = {}
                for lesson in lessons.LessonId:
                    frame_path = (
                        self.run_directory / "frames" / f"exam-{lesson.value.replace('/', '-')}.png"
                    )
                    lesson_seeds = tuple(self._validation_seeds(lesson))
                    expected_seeds[lesson] = lesson_seeds
                    result = lessons.evaluate_lesson(
                        graded_model,
                        lesson,
                        lesson_seeds,
                        frame_path=frame_path,
                        seed_access=self.validation_access,
                        case_evidence_sink=case_records.append,
                    )
                    evaluations[lesson] = result
                case_evidence = self._publish_exam_case_evidence(
                    exam,
                    exam_sidecar_sha256=sidecar_before,
                    evaluations=evaluations,
                    case_records=case_records,
                    expected_seeds=expected_seeds,
                )
                if decide:
                    self.latest_exam_case_evidence = dict(case_evidence)
                for lesson in lessons.LessonId:
                    result = evaluations[lesson]
                    append_jsonl(
                        self.run_directory / "evaluations.jsonl",
                        {
                            "timestamp": utc_now(),
                            "trigger": trigger,
                            "counts_toward_gate": decide,
                            "trained_actions": self.trained_actions,
                            "child_trained_actions": self.child_trained_actions,
                            "remediation_trained_actions": (self.remediation_trained_actions),
                            "optimizer_updates": int(self.model._n_updates),
                            "checkpoint": str(exam.relative_to(self.run_directory)),
                            "checkpoint_sha256": archive_before,
                            "sidecar": str(sidecar_path.relative_to(self.run_directory)),
                            "sidecar_sha256": sidecar_before,
                            "case_evidence": case_evidence,
                            "reloaded_for_grading": True,
                            "passed": lessons.lesson_passed(result),
                            **result.public_dict(),
                        },
                    )
                if (
                    file_sha256(exam) != archive_before
                    or file_sha256(sidecar_path) != sidecar_before
                ):
                    raise U2rProtocolError("immutable U2r exam bundle changed during grading")
                if (
                    int(self.model.num_timesteps) != self.trained_actions
                    or int(self.model._n_updates) != self.last_updates
                ):
                    raise U2rProtocolError("grading changed the live U2r model")
                if not decide and self.child_trained_actions == SOURCE_CHILD_ACTIONS:
                    for lesson, result in evaluations.items():
                        expected = self.parent.source_baseline[lesson.value]
                        if (
                            int(result.successes) != int(expected["successes"])
                            or tuple(int(value) for value in result.panel_successes)
                            != tuple(expected["panel_successes"])
                            or not math.isclose(
                                float(result.mean_ineffective_interactions),
                                float(expected["mean_ineffective_interactions"]),
                                rel_tol=0.0,
                                abs_tol=1e-12,
                            )
                        ):
                            raise U2rProtocolError(
                                f"reloaded U2r parent baseline does not reproduce at {lesson.value}"
                            )
                self.latest_evaluated_checkpoint = {
                    "path": str(exam.relative_to(self.run_directory)),
                    "checkpoint_sha256": archive_before,
                    "sidecar_sha256": sidecar_before,
                    "trained_actions": self.trained_actions,
                    "child_trained_actions": self.child_trained_actions,
                    "remediation_trained_actions": self.remediation_trained_actions,
                    "optimizer_updates": int(self.model._n_updates),
                    "trigger": trigger,
                    "counts_toward_gate": decide,
                    "case_evidence": case_evidence,
                }

                decision = frozen_u2.DecisionOutcome(
                    "Diagnostic U2r baseline; cannot count toward terminal selection"
                )
                if decide:
                    if self._last_evaluation_child_actions == self.child_trained_actions:
                        raise U2rProtocolError("duplicate U2r decision at one trained boundary")
                    self.last_completed_allocation = dict(allocation_before)
                    self.last_completed_allocation_valid = allocation_ok
                    decision = self._apply_decision(
                        evaluations,
                        allocation_ok=allocation_ok,
                        child_actions=self.child_trained_actions,
                        case_evidence=case_evidence,
                    )
                    self._last_evaluation_child_actions = self.child_trained_actions
                    if self.child_trained_actions < TERMINAL_CHILD_ACTIONS:
                        self.scheduler.start_window()
                        resume = self._copy_exam_artifact(
                            exam,
                            "resume",
                            kind="resume",
                            resume_eligible=True,
                            replace=True,
                            exam_integrity=integrity,
                        )
                        self.latest_safe_checkpoint = resume
                        self.latest_safe_trained_actions = self.trained_actions

                self.latest_evaluations = [
                    evaluations[lesson].public_dict()
                    | {"passed": lessons.lesson_passed(evaluations[lesson])}
                    for lesson in lessons.LessonId
                ]
                for lesson in lessons.LessonId:
                    result = evaluations[lesson]
                    self.evaluation_history.append(
                        {
                            "trained_actions": self.trained_actions,
                            "child_trained_actions": self.child_trained_actions,
                            "remediation_trained_actions": (self.remediation_trained_actions),
                            "lesson_id": result.lesson_id,
                            "lesson_label": result.lesson_label,
                            "successes": result.successes,
                            "episodes": result.episodes,
                            "success_rate": result.success_rate,
                            "panel_successes": result.panel_successes,
                            "mean_ineffective_interactions": (result.mean_ineffective_interactions),
                            "decision": decision.message,
                            "counts_toward_gate": decide,
                            "checkpoint_sha256": archive_before,
                            "case_evidence_sha256": case_evidence["case_set_sha256"],
                        }
                    )
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": ("stability_decision" if decide else "baseline_evaluation"),
                        "trained_actions": self.trained_actions,
                        "child_trained_actions": self.child_trained_actions,
                        "remediation_trained_actions": (self.remediation_trained_actions),
                        "decision": decision.message,
                        "checkpoint": str(exam.relative_to(self.run_directory)),
                        "checkpoint_sha256": archive_before,
                        "sidecar_sha256": sidecar_before,
                        "case_evidence": case_evidence,
                        "lesson_passes": {
                            lesson.value: lessons.lesson_passed(result)
                            for lesson, result in evaluations.items()
                        },
                        "curriculum": self.state.public_dict(),
                        "controller": self.controller.public_dict(),
                        "allocation": allocation_before,
                        "allocation_transitions": allocation_transitions,
                        "allocation_complete": allocation_complete,
                        "allocation_within_tolerance": allocation_ok,
                    },
                )
                self.frame_revision += 1

            def _advance_unchanged_curriculum(
                self,
                evaluations: Mapping[lessons.LessonId, Any],
                *,
                allocation_ok: bool,
            ) -> tuple[str, bool]:
                weak = tuple(lessons.weak_prerequisites(evaluations))
                in_recovery = bool(tuple(self.state.weak_prerequisites))
                exam_in_recovery = in_recovery or bool(weak)
                if in_recovery:
                    self.state.consecutive_passes = 0
                    if weak:
                        changed = tuple(self.state.weak_prerequisites) != weak
                        self.state.weak_prerequisites = weak
                        self.state.recovery_passes = 0
                        self.state.change()
                        prefix = "changed" if changed else "continues"
                        return (
                            "Recovery "
                            f"{prefix}: {','.join(item.value for item in weak)} "
                            "below gate",
                            exam_in_recovery,
                        )
                    if not allocation_ok:
                        self.state.recovery_passes = 0
                        self.state.change()
                        return (
                            "Recovery held: transition allocation outside tolerance",
                            exam_in_recovery,
                        )
                    self.state.recovery_passes += 1
                    self.state.change()
                    if self.state.recovery_passes < 2:
                        return (
                            "First clean recovery boundary; confirmation required",
                            exam_in_recovery,
                        )
                    self.state.weak_prerequisites = ()
                    self.state.recovery_passes = 0
                    self.state.consecutive_passes = 0
                    return (
                        "Prerequisites recovered; normal U2 practice resumes",
                        exam_in_recovery,
                    )
                if weak:
                    self.state.weak_prerequisites = weak
                    self.state.recovery_passes = 0
                    self.state.consecutive_passes = 0
                    self.state.change()
                    return (
                        f"Recovery started: {','.join(item.value for item in weak)} below gate",
                        exam_in_recovery,
                    )
                u2_passed = lessons.lesson_passed(evaluations[lessons.LessonId.SEPARATED_UNLOCK])
                if not allocation_ok or not u2_passed:
                    self.state.consecutive_passes = 0
                    self.state.change()
                    return (
                        (
                            "Held: transition allocation outside tolerance"
                            if not allocation_ok
                            else "Held: Separated Unlock below existing gate"
                        ),
                        exam_in_recovery,
                    )
                self.state.consecutive_passes = min(2, self.state.consecutive_passes + 1)
                self.state.change()
                return (
                    "Normal practice passed; terminal-only selection remains closed",
                    exam_in_recovery,
                )

            def _apply_decision(
                self,
                evaluations: Mapping[lessons.LessonId, Any],
                *,
                allocation_ok: bool,
                child_actions: int,
                case_evidence: Mapping[str, Any],
            ) -> frozen_u2.DecisionOutcome:
                if (
                    child_actions <= SOURCE_CHILD_ACTIONS
                    or child_actions > TERMINAL_CHILD_ACTIONS
                    or child_actions % EVALUATION_INTERVAL
                ):
                    raise U2rProtocolError("U2r decisions require a remediation exam boundary")
                previous = self.controller.last_decision_child_actions
                if previous is not None and child_actions - previous != EVALUATION_INTERVAL:
                    raise U2rProtocolError("U2r decisions are not one complete window apart")
                expected = (
                    SOURCE_CHILD_ACTIONS
                    + (len(self.controller.exam_records) + 1) * EVALUATION_INTERVAL
                )
                if child_actions != expected:
                    raise U2rProtocolError("U2r exam sequence has a missing boundary")
                curriculum_message, recovery = self._advance_unchanged_curriculum(
                    evaluations,
                    allocation_ok=allocation_ok,
                )
                public_evaluations = {
                    lesson.value: result.public_dict() for lesson, result in evaluations.items()
                }
                record = {
                    "child_trained_actions": child_actions,
                    "remediation_trained_actions": (child_actions - SOURCE_CHILD_ACTIONS),
                    "allocation_valid": bool(allocation_ok),
                    "recovery": bool(recovery),
                    "evaluations": public_evaluations,
                    "case_evidence": dict(case_evidence),
                }
                self.controller.last_decision_child_actions = child_actions
                self.controller.exam_records.append(record)

                if child_actions < TERMINAL_CHILD_ACTIONS:
                    return frozen_u2.DecisionOutcome(curriculum_message)
                if len(self.controller.exam_records) < 2:
                    raise U2rProtocolError("terminal U2r exam has no adjacent predecessor")
                prior = self.controller.exam_records[-2]
                final = self.controller.exam_records[-1]
                decision = grade_stability_pair(
                    prior["evaluations"],
                    final["evaluations"],
                    penultimate_allocation_valid=bool(prior["allocation_valid"]),
                    terminal_allocation_valid=bool(final["allocation_valid"]),
                    penultimate_recovery=bool(prior["recovery"]),
                    terminal_recovery=bool(final["recovery"]),
                    penultimate_child_actions=int(prior["child_trained_actions"]),
                    terminal_child_actions=int(final["child_trained_actions"]),
                )
                self.controller.terminal_eligible = decision.eligible
                self.controller.terminal_reasons = list(decision.reasons)
                self.controller.terminal_pair = {
                    "penultimate_child_actions": int(prior["child_trained_actions"]),
                    "terminal_child_actions": int(final["child_trained_actions"]),
                    **decision.public_dict(),
                }
                self.state.mastered = decision.eligible
                return frozen_u2.DecisionOutcome(
                    "U2r terminal policy eligible for fresh confirmation"
                    if decision.eligible
                    else "U2r terminal policy failed the frozen stability rule: "
                    + ", ".join(decision.reasons)
                )

            def _prune_rolling_exams(self) -> None:
                """U2r preserves every deterministic exam by protocol."""

            def _write_status(self, phase: str) -> None:
                elapsed = time.monotonic() - self.wall_start
                collected = int(self.model.num_timesteps)
                active = self._active_worker_evidence()
                if self.interruption_evidence is not None:
                    interruption_document = _read_json(
                        self.run_directory / str(self.interruption_evidence["path"]),
                        "U2r interruption status evidence",
                    )
                    active = [dict(worker) for worker in interruption_document["active_workers"]]
                atomic_write_json(
                    self.run_directory / "status.json",
                    {
                        "protocol": PROTOCOL,
                        "phase": phase,
                        "active_lineage": SOURCE_CHILD_SEED,
                        "parent": self.parent.public_dict(),
                        "qualification": self.qualification.public_dict(),
                        "exclusions": self.exclusions.public_dict(),
                        "external_preregistration": self.effective_config.get(
                            "external_preregistration"
                        ),
                        "storage_mount": dict(self.storage_mount),
                        "source": dict(self.source),
                        "started_at": self.started_at,
                        "updated_at": utc_now(),
                        "elapsed_seconds": elapsed,
                        "actions_per_second": (
                            (collected - self.segment_start_actions) / elapsed if elapsed else 0.0
                        ),
                        "fps": (
                            (collected - self.segment_start_actions) / elapsed if elapsed else 0.0
                        ),
                        "collected_actions": collected,
                        "collected_timesteps": collected,
                        "trained_actions": self.trained_actions,
                        "trained_timesteps": self.trained_actions,
                        "inherited_trained_actions": (SOURCE_U1_INHERITED_ACTIONS),
                        "source_child_actions": SOURCE_CHILD_ACTIONS,
                        "child_collected_actions": (collected - SOURCE_U1_INHERITED_ACTIONS),
                        "child_collected_timesteps": (collected - SOURCE_U1_INHERITED_ACTIONS),
                        "child_trained_actions": self.child_trained_actions,
                        "child_trained_timesteps": self.child_trained_actions,
                        "remediation_collected_actions": (collected - SOURCE_LIFETIME_ACTIONS),
                        "remediation_trained_actions": (self.remediation_trained_actions),
                        "remaining_remediation_actions": (
                            ADDITIONAL_ACTION_BUDGET - self.remediation_trained_actions
                        ),
                        "remaining_child_actions": (
                            TERMINAL_CHILD_ACTIONS - self.child_trained_actions
                        ),
                        "remaining_action_budget": (
                            TERMINAL_CHILD_ACTIONS - self.child_trained_actions
                        ),
                        "action_cap": TERMINAL_CHILD_ACTIONS,
                        "lifetime_trained_actions": self.trained_actions,
                        "lifetime_trained_timesteps": self.trained_actions,
                        "optimizer_updates": int(self.model._n_updates),
                        "episodes": self.episodes,
                        "current_lesson_id": (lessons.LessonId.SEPARATED_UNLOCK.value),
                        "current_lesson_label": (
                            lessons.LESSON_SPECS[lessons.LessonId.SEPARATED_UNLOCK].label
                        ),
                        "curriculum": self.state.public_dict(),
                        "controller": self.controller.public_dict(),
                        "practice_allocation": self.scheduler.snapshot(),
                        "last_completed_practice_allocation": (self.last_completed_allocation),
                        "last_completed_allocation_valid": (self.last_completed_allocation_valid),
                        "active_workers": active,
                        "latest_optimizer": self.latest_optimizer,
                        "next_evaluation": self.next_evaluation,
                        "latest_evaluations": self.latest_evaluations,
                        "evaluation_history": list(self.evaluation_history),
                        "lesson_frames": {
                            lesson.value: (f"frames/exam-{lesson.value.replace('/', '-')}.png")
                            for lesson in lessons.LessonId
                        },
                        "frame_revision": self.frame_revision,
                        "latest_exam_checkpoint": (
                            str(self.latest_exam_checkpoint.relative_to(self.run_directory))
                            if self.latest_exam_checkpoint
                            else None
                        ),
                        "latest_evaluated_checkpoint": (
                            dict(self.latest_evaluated_checkpoint)
                            if self.latest_evaluated_checkpoint is not None
                            else None
                        ),
                        "latest_exam_case_evidence": (
                            dict(self.latest_exam_case_evidence)
                            if self.latest_exam_case_evidence is not None
                            else None
                        ),
                        "terminal_report": (
                            dict(self.terminal_report_evidence)
                            if self.terminal_report_evidence is not None
                            else None
                        ),
                        "interruption": (
                            dict(self.interruption_evidence)
                            if self.interruption_evidence is not None
                            else None
                        ),
                        "latest_safe_checkpoint": (
                            str(self.latest_safe_checkpoint.relative_to(self.run_directory))
                            if self.latest_safe_checkpoint
                            else None
                        ),
                        "latest_safe_trained_actions": (self.latest_safe_trained_actions),
                        "latest_safe_checkpoint_sha256": (
                            file_sha256(self.latest_safe_checkpoint)
                            if self.latest_safe_checkpoint is not None
                            and self.latest_safe_checkpoint.is_file()
                            else None
                        ),
                        "storage": self._storage_status(),
                        "segment": dict(self.segment),
                    },
                )

            def _verify_complete_exam_chain(self) -> list[Path]:
                expected_boundaries = [
                    SOURCE_CHILD_ACTIONS + index * EVALUATION_INTERVAL
                    for index in range(1, KEEP_ALL_EXAMS + 1)
                ]
                records = self.controller.exam_records
                if [
                    int(record.get("child_trained_actions", -1)) for record in records
                ] != expected_boundaries:
                    raise U2rProtocolError(
                        "terminal U2r controller does not retain all eleven exams"
                    )
                archives = sorted(
                    (self.run_directory / "checkpoints" / "rolling").glob("exam-*.zip")
                )
                if [
                    int(archive.stem.removeprefix("exam-")) for archive in archives
                ] != expected_boundaries:
                    raise U2rProtocolError(
                        "terminal U2r artifact chain is not eleven contiguous exams"
                    )
                expected_seeds = {
                    lesson: tuple(self._validation_seeds(lesson)) for lesson in lessons.LessonId
                }
                for archive, record in zip(archives, records, strict=True):
                    sidecar = archive.with_suffix(".json")
                    integrity = _verify_integrity(archive, sidecar)
                    binding = record.get("case_evidence")
                    if not isinstance(binding, Mapping):
                        raise U2rProtocolError("terminal U2r exam lacks case-evidence binding")
                    _verify_exam_case_document(
                        _case_evidence_path(archive),
                        run_directory=self.run_directory,
                        binding=binding,
                        evaluations=record["evaluations"],
                        checkpoint=archive,
                        checkpoint_sha256=str(integrity["checkpoint_sha256"]),
                        sidecar_sha256=str(integrity["sidecar_sha256"]),
                        child_actions=int(record["child_trained_actions"]),
                        expected_seeds=expected_seeds,
                    )
                return archives

            def _publish_terminal_report(
                self,
                *,
                terminal: Path,
                phase: str,
                archives: Sequence[Path],
            ) -> dict[str, Any]:
                report_path = self.run_directory / "report.json"
                integrity_path = self.run_directory / "report.integrity.json"
                if any(
                    path.exists() or path.is_symlink() for path in (report_path, integrity_path)
                ):
                    raise U2rProtocolError("refusing to overwrite immutable U2r terminal report")
                terminal_sidecar_path = terminal.with_suffix(".json")
                terminal_integrity_path = _integrity_path(terminal)
                terminal_integrity = _verify_integrity(terminal, terminal_sidecar_path)
                terminal_sidecar = _read_json(
                    terminal_sidecar_path,
                    "terminal U2r sidecar",
                )
                exam_inventory: list[dict[str, Any]] = []
                for archive, record in zip(
                    archives,
                    self.controller.exam_records,
                    strict=True,
                ):
                    sidecar = archive.with_suffix(".json")
                    integrity = _integrity_path(archive)
                    cases = _case_evidence_path(archive)
                    binding = record["case_evidence"]
                    exam_inventory.append(
                        {
                            "child_trained_actions": int(record["child_trained_actions"]),
                            "checkpoint": str(archive.relative_to(self.run_directory)),
                            "checkpoint_sha256": file_sha256(archive),
                            "sidecar": str(sidecar.relative_to(self.run_directory)),
                            "sidecar_sha256": file_sha256(sidecar),
                            "integrity": str(integrity.relative_to(self.run_directory)),
                            "integrity_sha256": file_sha256(integrity),
                            "case_evidence": str(cases.relative_to(self.run_directory)),
                            "case_evidence_sha256": file_sha256(cases),
                            "case_set_sha256": binding["case_set_sha256"],
                            "aggregate_sha256s": {
                                lesson: evidence["aggregate_sha256"]
                                for lesson, evidence in binding["per_lesson"].items()
                            },
                        }
                    )
                storage = self._guard_storage()
                report = {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "kind": "terminal_stability_eligibility_report",
                    "created_at": utc_now(),
                    "verdict": phase,
                    "eligible_for_fresh_confirmation": (self.controller.terminal_eligible is True),
                    "fresh_confirmation_opened": False,
                    "policy_updates_during_reporting": False,
                    "source": dict(self.source),
                    "external_preregistration": self.effective_config.get(
                        "external_preregistration"
                    ),
                    "effective_config": dict(self.effective_config),
                    "parent": self.parent.public_dict(),
                    "qualification": self.qualification.public_dict(),
                    "exclusions": self.exclusions.public_dict(),
                    "storage_mount": dict(self.storage_mount),
                    "storage": dict(storage),
                    "segment": dict(self.segment),
                    "progress": {
                        "source_child_actions": SOURCE_CHILD_ACTIONS,
                        "remediation_trained_actions": (self.remediation_trained_actions),
                        "child_trained_actions": self.child_trained_actions,
                        "lifetime_trained_actions": self.trained_actions,
                        "optimizer_updates": int(self.model._n_updates),
                    },
                    "parent_baseline": self.parent.public_dict()["source_baseline"],
                    "controller": self.controller.public_dict(),
                    "terminal_pair": self.controller.terminal_pair,
                    "exams": exam_inventory,
                    "terminal_artifact": {
                        "checkpoint": str(terminal.relative_to(self.run_directory)),
                        "checkpoint_sha256": str(terminal_integrity["checkpoint_sha256"]),
                        "sidecar": str(terminal_sidecar_path.relative_to(self.run_directory)),
                        "sidecar_sha256": str(terminal_integrity["sidecar_sha256"]),
                        "integrity": str(terminal_integrity_path.relative_to(self.run_directory)),
                        "integrity_sha256": file_sha256(terminal_integrity_path),
                        "model_state": terminal_sidecar["model_state"],
                    },
                    "lineage_history": _build_lineage_history(
                        self.run_directory,
                        terminal_sidecar=terminal_sidecar,
                    ),
                    "evaluations": {
                        "path": "evaluations.jsonl",
                        "sha256": file_sha256(self.run_directory / "evaluations.jsonl"),
                    },
                }
                anticipated = len(
                    json.dumps(
                        report,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                )
                self._guard_storage(anticipated)
                atomic_write_json(report_path, report)
                integrity = {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "report": report_path.name,
                    "report_sha256": file_sha256(report_path),
                }
                atomic_write_json(integrity_path, integrity)
                verify_terminal_report(self.run_directory)
                self._guard_storage()
                return {
                    "path": report_path.name,
                    "sha256": file_sha256(report_path),
                    "integrity": integrity_path.name,
                    "integrity_sha256": file_sha256(integrity_path),
                    "verdict": phase,
                    "eligible_for_fresh_confirmation": (self.controller.terminal_eligible is True),
                }

            def _publish_interruption_evidence(
                self,
                *,
                phase: str,
                interrupted_at: str,
                active_workers: Sequence[Mapping[str, Any]],
                signal_collected_actions: int,
                signal_trained_actions: int,
                signal_child_actions: int,
                signal_optimizer_updates: int,
                untrained_transitions_per_worker: int,
                initial_boundary_classification: str,
                final_boundary_classification: str,
                recovered_complete_actions: int,
                recovered_complete_updates: int,
                recovery_error: str | None,
            ) -> dict[str, Any]:
                evidence_path = self.run_directory / "interruption.json"
                integrity_path = self.run_directory / "interruption.integrity.json"
                if any(
                    path.exists() or path.is_symlink() for path in (evidence_path, integrity_path)
                ):
                    raise U2rProtocolError(
                        "refusing to overwrite immutable U2r interruption evidence"
                    )
                workers = [dict(worker) for worker in active_workers]
                if len(workers) != WORKERS or {
                    int(worker.get("worker_index", -1)) for worker in workers
                } != set(range(WORKERS)):
                    raise U2rProtocolError("U2r interruption lacks four current worker identities")
                workers.sort(key=lambda worker: int(worker["worker_index"]))
                latest = self.latest_safe_checkpoint
                if latest is None or not latest.is_file():
                    raise U2rProtocolError("U2r interruption has no exact safe resume boundary")
                latest_sidecar = latest.with_suffix(".json")
                latest_integrity = _integrity_path(latest)
                _verify_integrity(latest, latest_sidecar)
                latest_sidecar_value = _read_json(
                    latest_sidecar,
                    "U2r interrupted latest-safe sidecar",
                )
                latest_progress = latest_sidecar_value.get("progress")
                if (
                    latest_sidecar_value.get("resume_eligible") is not True
                    or not isinstance(latest_progress, Mapping)
                    or int(latest_progress.get("trained_actions", -1))
                    != self.latest_safe_trained_actions
                    or int(latest_progress.get("collected_actions", -1))
                    != self.latest_safe_trained_actions
                    or self.latest_safe_trained_actions > signal_collected_actions
                ):
                    raise U2rProtocolError("U2r interruption safe-boundary counters changed")
                baseline_resume_permitted = True
                baseline_resume_blocker: str | None = None
                if int(self.segment.get("index", -1)) == 0:
                    try:
                        _build_parent_baseline_inventory(
                            self.run_directory,
                            cohort_root=self.run_directory.parent.resolve(),
                        )
                    except BaseException as error:
                        baseline_resume_permitted = False
                        baseline_resume_blocker = f"{type(error).__name__}: {error}"
                post_safe_actions = signal_collected_actions - self.latest_safe_trained_actions
                unpublished_trained_actions = max(
                    0,
                    self.trained_actions - self.latest_safe_trained_actions,
                )
                ledger = self.run_directory / "episode-starts.jsonl"
                for worker in workers:
                    append_jsonl(
                        ledger,
                        {
                            "timestamp": interrupted_at,
                            "type": "episode_abandoned_at_interruption",
                            "interruption_phase": phase,
                            "active_environment_state_disposition": (
                                "discarded_at_process_boundary"
                            ),
                            "partial_episode_identity_disposition": (
                                "retained_as_authenticated_provenance"
                            ),
                            "post_safe_rollout_experience_disposition": (
                                "none"
                                if post_safe_actions == 0
                                else "discarded_after_last_safe_checkpoint"
                            ),
                            "post_safe_rollout_actions": post_safe_actions,
                            "unpublished_trained_actions": (unpublished_trained_actions),
                            "untrained_collected_actions": (
                                untrained_transitions_per_worker * WORKERS
                            ),
                            **worker,
                        },
                    )
                document = {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "kind": "process_interruption_evidence",
                    "created_at": utc_now(),
                    "phase": phase,
                    "resume_permitted": baseline_resume_permitted,
                    "resume_blocker": baseline_resume_blocker,
                    "source": dict(self.source),
                    "effective_config": dict(self.effective_config),
                    "parent": self.parent.public_dict(),
                    "qualification": self.qualification.public_dict(),
                    "exclusions": self.exclusions.public_dict(),
                    "external_preregistration": self.effective_config.get(
                        "external_preregistration"
                    ),
                    "storage_mount": dict(self.storage_mount),
                    "segment": dict(self.segment),
                    "at_signal": {
                        "timestamp": interrupted_at,
                        "collected_actions": signal_collected_actions,
                        "trained_actions": signal_trained_actions,
                        "child_trained_actions": signal_child_actions,
                        "optimizer_updates": signal_optimizer_updates,
                        "pending_optimizer_boundary": (initial_boundary_classification),
                        "untrained_collected_actions": (untrained_transitions_per_worker * WORKERS),
                        "untrained_transitions_per_worker": (untrained_transitions_per_worker),
                    },
                    "resolution": {
                        "pending_optimizer_boundary": (final_boundary_classification),
                        "recovered_complete_actions": (recovered_complete_actions),
                        "recovered_complete_optimizer_updates": (recovered_complete_updates),
                        "discarded_post_safe_rollout_actions": (post_safe_actions),
                        "unpublished_trained_actions": (unpublished_trained_actions),
                        "untrained_collected_actions": (untrained_transitions_per_worker * WORKERS),
                        "unpublished_complete_actions": (
                            unpublished_trained_actions
                            if final_boundary_classification == "complete_recovery_failed"
                            else 0
                        ),
                        "recovery_error": recovery_error,
                        "resolved_trained_actions": self.trained_actions,
                        "resolved_child_trained_actions": (self.child_trained_actions),
                        "resolved_optimizer_updates": int(self.model._n_updates),
                        "safe_boundary_trained_actions": (self.latest_safe_trained_actions),
                        "active_environment_state_disposition": ("discarded_at_process_boundary"),
                        "policy_recurrent_state_disposition": ("discarded_at_process_boundary"),
                    },
                    "active_workers": workers,
                    "active_workers_sha256": _canonical_json_sha256(workers),
                    "episode_start_ledger": {
                        "path": ledger.name,
                        "sha256": file_sha256(ledger),
                        **_jsonl_inventory(ledger),
                    },
                    "resume_boundary": {
                        "checkpoint": str(latest.resolve()),
                        "checkpoint_sha256": file_sha256(latest),
                        "sidecar": str(latest_sidecar.resolve()),
                        "sidecar_sha256": file_sha256(latest_sidecar),
                        "integrity": str(latest_integrity.resolve()),
                        "integrity_sha256": file_sha256(latest_integrity),
                        "model_state": latest_sidecar_value["model_state"],
                        "progress": latest_sidecar_value["progress"],
                    },
                }
                anticipated = len(
                    json.dumps(
                        document,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                )
                self._guard_storage(anticipated)
                atomic_write_json(evidence_path, document)
                atomic_write_json(
                    integrity_path,
                    {
                        "schema_version": CHECKPOINT_SCHEMA_VERSION,
                        "protocol": PROTOCOL,
                        "interruption": evidence_path.name,
                        "interruption_sha256": file_sha256(evidence_path),
                    },
                )
                self._guard_storage()
                return {
                    "path": evidence_path.name,
                    "sha256": file_sha256(evidence_path),
                    "integrity": integrity_path.name,
                    "integrity_sha256": file_sha256(integrity_path),
                    "phase": phase,
                    "active_workers_sha256": document["active_workers_sha256"],
                }

            def finalize_failure(self, phase: str) -> None:
                if phase not in {"interrupted", "crashed"}:
                    raise U2rProtocolError("U2r failure phase must be interrupted or crashed")
                interrupted_at = utc_now()
                signal_collected = int(self.model.num_timesteps)
                signal_trained = self.trained_actions
                signal_child = self.child_trained_actions
                signal_updates = int(self.model._n_updates)
                (
                    pending_actions,
                    pending_updates,
                    initial_classification,
                ) = self._pending_optimizer_delta()
                if initial_classification == "partial":
                    if pending_actions < 0 or pending_actions % WORKERS:
                        raise U2rProtocolError("U2r partial rollout is not vector synchronized")
                    untrained_per_worker = pending_actions // WORKERS
                else:
                    untrained_per_worker = 0
                signal_workers = []
                for raw_worker in self._active_worker_evidence():
                    worker = dict(raw_worker)
                    worker["untrained_transitions"] = untrained_per_worker
                    signal_workers.append(worker)
                final_classification = initial_classification
                recovered_actions = 0
                recovered_updates = 0
                recovery_error: str | None = None
                if initial_classification == "complete":
                    try:
                        self._process_optimized_boundary()
                    except BaseException as error:
                        recovery_error = f"{type(error).__name__}: {error}"
                        final_classification = "complete_recovery_failed"
                    else:
                        recovered_actions = pending_actions
                        recovered_updates = pending_updates
                        final_classification = "complete_recovered"
                if (
                    int(self.model.num_timesteps) == TERMINAL_LIFETIME_ACTIONS
                    and self.trained_actions == TERMINAL_LIFETIME_ACTIONS
                    and self.child_trained_actions == TERMINAL_CHILD_ACTIONS
                    and int(self.model._n_updates)
                    == SOURCE_OPTIMIZER_UPDATES
                    + ADDITIONAL_ACTION_BUDGET // ROLLOUT_TRANSITIONS * PPO_EPOCHS
                    and self.controller.terminal_eligible in {True, False}
                ):
                    self.finalize("completed")
                    return
                self.interruption_evidence = self._publish_interruption_evidence(
                    phase=phase,
                    interrupted_at=interrupted_at,
                    active_workers=signal_workers,
                    signal_collected_actions=signal_collected,
                    signal_trained_actions=signal_trained,
                    signal_child_actions=signal_child,
                    signal_optimizer_updates=signal_updates,
                    untrained_transitions_per_worker=(untrained_per_worker),
                    initial_boundary_classification=initial_classification,
                    final_boundary_classification=final_classification,
                    recovered_complete_actions=recovered_actions,
                    recovered_complete_updates=recovered_updates,
                    recovery_error=recovery_error,
                )
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": f"run_{phase}",
                        "collected_actions": signal_collected,
                        "trained_actions": self.trained_actions,
                        "child_trained_actions": self.child_trained_actions,
                        "pending_optimizer_boundary": final_classification,
                        "recovered_complete_actions": recovered_actions,
                        "recovered_complete_optimizer_updates": (recovered_updates),
                        "discarded_partial_actions": (
                            max(0, pending_actions)
                            if final_classification in {"partial", "counter_regression"}
                            else 0
                        ),
                        "discarded_partial_optimizer_updates": (
                            max(0, pending_updates)
                            if final_classification in {"partial", "counter_regression"}
                            else 0
                        ),
                        "unpublished_complete_actions": (
                            max(0, pending_actions)
                            if final_classification == "complete_recovery_failed"
                            else 0
                        ),
                        "unpublished_complete_optimizer_updates": (
                            max(0, pending_updates)
                            if final_classification == "complete_recovery_failed"
                            else 0
                        ),
                        "recovery_error": recovery_error,
                        "interruption": dict(self.interruption_evidence),
                    },
                )
                self._write_status(phase)

            def finalize(self, _phase: str) -> None:
                if int(self.model._n_updates) > self.last_updates:
                    self._process_optimized_boundary()
                if (
                    int(self.model.num_timesteps) != self.trained_actions
                    or self.remediation_trained_actions != ADDITIONAL_ACTION_BUDGET
                    or self.child_trained_actions != TERMINAL_CHILD_ACTIONS
                    or self.trained_actions != TERMINAL_LIFETIME_ACTIONS
                    or int(self.model._n_updates)
                    != SOURCE_OPTIMIZER_UPDATES
                    + ADDITIONAL_ACTION_BUDGET // ROLLOUT_TRANSITIONS * PPO_EPOCHS
                    or self.controller.terminal_eligible not in {True, False}
                ):
                    raise U2rProtocolError(
                        "U2r did not terminate at its exact fully optimized ceiling"
                    )
                exam = self.latest_exam_checkpoint
                if exam is None or exam.name != f"exam-{TERMINAL_CHILD_ACTIONS:07d}.zip":
                    raise U2rProtocolError("U2r terminal selection is not the exact terminal exam")
                archives = self._verify_complete_exam_chain()
                terminal = self.run_directory / "checkpoints" / "terminal.zip"
                terminal_sidecar = terminal.with_suffix(".json")
                terminal_integrity_path = _integrity_path(terminal)
                terminal_exists = [
                    path.exists() or path.is_symlink()
                    for path in (
                        terminal,
                        terminal_sidecar,
                        terminal_integrity_path,
                    )
                ]
                if any(terminal_exists):
                    if not all(terminal_exists):
                        raise U2rProtocolError("partial U2r terminal bundle cannot be recovered")
                    terminal_integrity = _verify_integrity(
                        terminal,
                        terminal_sidecar,
                    )
                    terminal_value = _read_json(
                        terminal_sidecar,
                        "existing terminal U2r sidecar",
                    )
                    exam_integrity = _verify_integrity(
                        exam,
                        exam.with_suffix(".json"),
                    )
                    if (
                        terminal_integrity["checkpoint_sha256"]
                        != exam_integrity["checkpoint_sha256"]
                        or terminal_value.get("kind")
                        != (
                            "terminal_eligible"
                            if self.controller.terminal_eligible
                            else "terminal_failed"
                        )
                        or terminal_value.get("resume_eligible") is not False
                        or terminal_value.get("controller") != self.controller.public_dict()
                    ):
                        raise U2rProtocolError("existing U2r terminal bundle changed")
                else:
                    terminal = self._copy_exam_artifact(
                        exam,
                        "terminal",
                        kind=(
                            "terminal_eligible"
                            if self.controller.terminal_eligible
                            else "terminal_failed"
                        ),
                        resume_eligible=False,
                        replace=False,
                        exam_integrity=_verify_integrity(exam, exam.with_suffix(".json")),
                    )
                self.latest_safe_checkpoint = terminal
                self.latest_safe_trained_actions = self.trained_actions
                phase = "eligible" if self.controller.terminal_eligible else "failed"
                events_path = self.run_directory / "events.jsonl"
                terminal_events = [
                    event
                    for event in _read_jsonl(
                        events_path,
                        "U2r terminal events",
                    )
                    if event.get("type") == "run_terminal"
                ]
                if not terminal_events:
                    append_jsonl(
                        events_path,
                        {
                            "timestamp": utc_now(),
                            "type": "run_terminal",
                            "phase": phase,
                            "development_verdict": phase,
                            "terminal_checkpoint": str(terminal.relative_to(self.run_directory)),
                            "terminal_checkpoint_sha256": file_sha256(terminal),
                            "child_trained_actions": (self.child_trained_actions),
                            "remediation_trained_actions": (self.remediation_trained_actions),
                            "terminal_pair": self.controller.terminal_pair,
                            "terminal_report_published_after_event": True,
                        },
                    )
                elif (
                    len(terminal_events) != 1
                    or terminal_events[0].get("phase") != phase
                    or terminal_events[0].get("terminal_checkpoint_sha256") != file_sha256(terminal)
                    or terminal_events[0].get("terminal_pair") != self.controller.terminal_pair
                ):
                    raise U2rProtocolError("U2r terminal event history changed")
                report_path = self.run_directory / "report.json"
                report_integrity_path = self.run_directory / "report.integrity.json"
                report_exists = [
                    path.exists() or path.is_symlink()
                    for path in (report_path, report_integrity_path)
                ]
                if any(report_exists):
                    if not all(report_exists):
                        raise U2rProtocolError("partial U2r terminal report cannot be recovered")
                    report_value = verify_terminal_report(self.run_directory)
                    self.terminal_report_evidence = {
                        "path": report_path.name,
                        "sha256": file_sha256(report_path),
                        "integrity": report_integrity_path.name,
                        "integrity_sha256": file_sha256(report_integrity_path),
                        "verdict": report_value["verdict"],
                        "eligible_for_fresh_confirmation": report_value[
                            "eligible_for_fresh_confirmation"
                        ],
                    }
                else:
                    self.terminal_report_evidence = self._publish_terminal_report(
                        terminal=terminal,
                        phase=phase,
                        archives=archives,
                    )
                self._write_status(phase)

        return U2rCallback


def _best_effort(label: str, operation: Callable[[], Any]) -> None:
    try:
        operation()
    except BaseException as error:
        print(f"Warning: {label} failed during U2r finalization: {error}", flush=True)


def _write_setup_terminal_status(
    run_directory: Path,
    *,
    phase: str,
    started_at: str,
    wall_start: float,
    parent: U2rParent,
    qualification: frozen_u2.QualificationProvenance,
    source: Mapping[str, Any],
    exclusions: ForbiddenLayoutEvidence,
    external_preregistration: Mapping[str, Any],
    storage_mount: Mapping[str, Any],
    initial_trained_actions: int,
    child_trained_actions: int,
    remediation_trained_actions: int,
    expected_updates: int,
    segment: Mapping[str, Any] | None,
) -> None:
    now = utc_now()
    atomic_write_json(
        run_directory / "status.json",
        {
            "protocol": PROTOCOL,
            "phase": phase,
            "setup_complete": False,
            "active_lineage": SOURCE_CHILD_SEED,
            "parent": parent.public_dict(),
            "qualification": qualification.public_dict(),
            "source": dict(source),
            "exclusions": exclusions.public_dict(),
            "external_preregistration": dict(external_preregistration),
            "storage_mount": dict(storage_mount),
            "started_at": started_at,
            "updated_at": now,
            "elapsed_seconds": max(0.0, time.monotonic() - wall_start),
            "collected_actions": initial_trained_actions,
            "trained_actions": initial_trained_actions,
            "lifetime_trained_actions": initial_trained_actions,
            "child_trained_actions": child_trained_actions,
            "remediation_trained_actions": remediation_trained_actions,
            "remaining_remediation_actions": (
                ADDITIONAL_ACTION_BUDGET - remediation_trained_actions
            ),
            "remaining_action_budget": (TERMINAL_CHILD_ACTIONS - child_trained_actions),
            "action_cap": TERMINAL_CHILD_ACTIONS,
            "optimizer_updates": expected_updates,
            "segment": dict(segment) if segment is not None else None,
            "crash_report": "crash.json" if phase == "crashed" else None,
        },
    )
    append_jsonl(
        run_directory / "events.jsonl",
        {
            "timestamp": now,
            "type": f"run_{phase}",
            "classification": "pre_training_setup_failure",
            "trained_actions": initial_trained_actions,
            "child_trained_actions": child_trained_actions,
            "remediation_trained_actions": remediation_trained_actions,
            "optimizer_updates": expected_updates,
        },
    )


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    repository = Path(__file__).resolve().parents[2]
    source = git_snapshot(repository)
    if (
        source.get("dirty") is not False
        or not isinstance(source.get("commit"), str)
        or not source["commit"]
    ):
        raise SystemExit("U2r requires one clean identified source commit")

    # Authenticate all inherited evidence before creating a run directory,
    # deserializing the policy, or instantiating a training layout.
    parent = verify_u2r_parent(args.parent, args.confirmation_report)
    from dungeon_apprentice.u2_qualification_anchor import verify_external_anchor

    qualification_anchor = verify_external_anchor(
        repository,
        expected_source_commit=SOURCE_TRAINING_COMMIT,
    )
    qualification = frozen_u2.verify_qualification(
        args.qualification_report,
        expected_source_commit=SOURCE_TRAINING_COMMIT,
        anchor=qualification_anchor,
    )
    seed_access = qualification.seed_access()
    qualification_report = qualification.verified_report()
    forbidden_layout_hashes, exclusion_evidence = build_u2r_forbidden_layout_hashes(
        qualification_report,
        parent.confirmation_snapshot(),
        access=seed_access,
    )
    from dungeon_apprentice import u2r_anchor

    u2r_anchor.verify_failed_r0_launch()
    protocol_document_digest = u2r_anchor.protocol_document_sha256(repository)
    anchor = u2r_anchor.verify_external_anchor(
        repository,
        expected_source_commit=str(source["commit"]),
        expected_protocol_sha256=protocol_document_digest,
        expected_exclusions=exclusion_evidence,
    )
    config = effective_config(
        args,
        parent=parent,
        qualification=qualification,
        exclusions=exclusion_evidence,
        anchor=anchor,
    )
    resume_plan = None
    if args.resume is not None:
        requested_resume = args.resume.expanduser()
        if requested_resume.suffix != ".zip":
            requested_resume = requested_resume.with_suffix(".zip")
        if requested_resume.is_symlink():
            raise SystemExit("U2r resume checkpoint cannot be a symlink")
        resume_plan = u2r_anchor.select_resume_plan(
            args.run_root,
            expected_source_commit=str(source["commit"]),
        )
        if (
            requested_resume.resolve() != Path(resume_plan.checkpoint).resolve()
            or file_sha256(requested_resume.resolve()) != resume_plan.checkpoint_sha256
            or args.run_name != resume_plan.next_run_name
        ):
            raise SystemExit("U2r --resume is not the authenticated canonical lineage tip")
    resume = (
        load_resume(
            args.resume,
            expected_config=config,
            parent=parent,
            qualification=qualification,
            expected_source_commit=str(source["commit"]),
        )
        if args.resume is not None
        else None
    )
    if resume is not None and (
        resume_plan is None
        or resume.segment_index != resume_plan.next_segment_index
        or resume.child_trained != resume_plan.child_trained_actions
        or resume.remediation_trained != resume_plan.remediation_trained_actions
        or resume.lifetime_trained != resume_plan.lifetime_trained_actions
        or resume.n_updates != resume_plan.optimizer_updates
    ):
        raise SystemExit("U2r resume bundle differs from the authenticated plan")
    initial_trained = resume.lifetime_trained if resume is not None else SOURCE_LIFETIME_ACTIONS
    child_trained = resume.child_trained if resume is not None else SOURCE_CHILD_ACTIONS
    remediation_trained = resume.remediation_trained if resume is not None else 0
    remaining = ADDITIONAL_ACTION_BUDGET - remediation_trained
    if remaining <= 0:
        raise SystemExit("the U2r action budget is already exhausted")
    expected_run_name = (
        _segment_run_name(resume.segment_index) if resume is not None else DEFAULT_RUN_NAME
    )
    if args.run_name != expected_run_name:
        raise SystemExit(
            "U2r run name is not the deterministic segment identity: "
            f"{args.run_name!r} != {expected_run_name!r}"
        )
    sampler_segment_index = resume.segment_index if resume is not None else 0
    sampler_segment_offset = sampler_segment_index * SEGMENT_SEED_OFFSET
    sampler_worker_streams = tuple(
        seed + sampler_segment_offset for seed in REMEDIATION_WORKER_STREAMS
    )
    sampler_preflight = preflight_u2r_training_layout_sampler(
        forbidden_layout_hashes,
        seed_access=seed_access,
        worker_streams=sampler_worker_streams,
        max_attempts=U2R_LAYOUT_RESAMPLE_ATTEMPTS,
    )

    minimum_free_bytes = int(float(args.minimum_free_gib) * 1024**3)
    mount_evidence = verify_storage_mount(args.storage_root)
    if resume is not None and resume.sidecar.get("storage_mount") != mount_evidence:
        raise SystemExit("U2r resume storage mount identity changed")
    ensure_disk_space(args.run_root, minimum_free_bytes)
    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    except ImportError as error:
        raise SystemExit('Install training dependencies with: pip install -e ".[train]"') from error

    expected_updates = resume.n_updates if resume is not None else SOURCE_OPTIMIZER_UPDATES
    source_checkpoint = resume.checkpoint if resume is not None else Path(parent.checkpoint)
    try:
        preflight_model = RecurrentPPO.load(
            source_checkpoint,
            device=args.device,
        )
        _verify_loaded_model(
            preflight_model,
            args,
            expected_actions=initial_trained,
            expected_updates=expected_updates,
            expected_policy_sha256=(
                resume.policy_tensor_sha256 if resume is not None else SOURCE_POLICY_TENSOR_SHA256
            ),
            expected_optimizer_sha256=(
                resume.optimizer_state_sha256
                if resume is not None
                else SOURCE_OPTIMIZER_STATE_SHA256
            ),
        )
    except Exception as error:
        raise SystemExit(
            "U2r source model failed the pre-creation deserialization check"
        ) from error
    finally:
        if "preflight_model" in locals():
            del preflight_model

    run_directory = create_run_directory(args.run_root, args.run_name)
    started_at = utc_now()
    setup_wall_start = time.monotonic()
    segment: dict[str, Any] | None = None
    callback: Any | None = None
    environment_to_close: Any | None = None
    opened_environments: list[Any] = []

    def storage_guard(
        anticipated_lineage_bytes: int = 0,
    ) -> Mapping[str, Any]:
        return audit_u2_storage(
            storage_root=args.storage_root,
            cohort_directory=args.run_root,
            lineage_directory=run_directory,
            media_directory=args.media_directory,
            protected_paths=(
                Path(parent.checkpoint),
                Path(parent.sidecar),
                Path(parent.integrity),
                Path(parent.manifest),
                Path(parent.evaluations),
                Path(parent.confirmation_report),
                Path(parent.confirmation_attempt),
                Path(qualification.report),
            ),
            anticipated_lineage_bytes=anticipated_lineage_bytes,
        ).as_dict()

    def source_guard() -> None:
        current = git_snapshot(repository)
        if current.get("dirty") is not False or current.get("commit") != source["commit"]:
            raise U2rProtocolError("U2r source became dirty or changed during the active run")

    try:
        if resume is not None:
            state = resume.curriculum
            controller = resume.controller
            scheduler_state = resume.scheduler_state
            segment_index = resume.segment_index
            abandoned_workers = _record_abandoned_resume_workers(
                run_directory,
                resume,
            )
        else:
            source_sidecar = _read_json(Path(parent.sidecar), "U2r source mastery sidecar")
            state = lessons.CurriculumState.from_dict(source_sidecar["curriculum"])
            # Legacy U2 mastery does not satisfy the successor's stability gate.
            # Preserve the scheduler revision/RNG/lifetime counts while opening a
            # fresh normal-practice window.
            state.mastered = False
            state.consecutive_passes = 0
            state.weak_prerequisites = ()
            state.recovery_passes = 0
            controller = ControllerState()
            scheduler_state = parent.scheduler_state
            segment_index = 0
            abandoned_workers = []

        segment_offset = segment_index * SEGMENT_SEED_OFFSET
        segment_algorithm_seed = REMEDIATION_ALGORITHM_SEED + segment_offset
        segment_worker_streams = tuple(seed + segment_offset for seed in REMEDIATION_WORKER_STREAMS)
        segment = {
            "id": uuid.uuid4().hex,
            "index": segment_index,
            "started_at": started_at,
            "lineage_source_child_seed": SOURCE_CHILD_SEED,
            "initial_algorithm_seed": REMEDIATION_ALGORITHM_SEED,
            "initial_worker_streams": list(REMEDIATION_WORKER_STREAMS),
            "segment_algorithm_seed": segment_algorithm_seed,
            "segment_worker_streams": list(segment_worker_streams),
            "layout_sampler_preflight": [dict(record) for record in sampler_preflight],
            "start_lifetime_trained_actions": initial_trained,
            "start_child_trained_actions": child_trained,
            "start_remediation_trained_actions": remediation_trained,
            "remaining_remediation_actions": remaining,
            "resume_checkpoint": (str(resume.checkpoint) if resume is not None else None),
            "resume_checkpoint_sha256": (
                file_sha256(resume.checkpoint) if resume is not None else None
            ),
            "resume_model_state": (
                {
                    "policy_tensor_sha256": resume.policy_tensor_sha256,
                    "optimizer_state_sha256": resume.optimizer_state_sha256,
                }
                if resume is not None
                else None
            ),
            "resume_source_segment_index": (
                resume.segment_index - 1 if resume is not None else None
            ),
            "inherited_active_workers": abandoned_workers,
            "inherited_active_workers_abandoned": bool(abandoned_workers),
        }
        scheduler = lessons.TransitionDeficitScheduler(
            state,
            seed=segment_algorithm_seed + 90_000,
        )
        scheduler.load_state_dict(scheduler_state)
        storage_guard()
        carried_exams = (
            _carry_forward_exams(
                resume,
                run_directory=run_directory,
                storage_guard=storage_guard,
            )
            if resume is not None
            else []
        )

        evidence_wrappers: list[EpisodeEvidenceWrapper] = []
        factories: list[Callable[[], Any]] = []
        for worker_index, worker_stream in enumerate(segment_worker_streams):

            def factory(
                index: int = worker_index,
                stream: int = worker_stream,
            ) -> Any:
                environment = lessons.make_training_env(
                    scheduler=scheduler,
                    seed=stream,
                    size=args.size,
                    seed_access=seed_access,
                    forbidden_layout_hashes=forbidden_layout_hashes,
                    max_layout_resample_attempts=U2R_LAYOUT_RESAMPLE_ATTEMPTS,
                )
                recorder = EpisodeEvidenceWrapper(
                    environment,
                    worker_index=index,
                    worker_stream=stream,
                    ledger=run_directory / "episode-starts.jsonl",
                )
                evidence_wrappers.append(recorder)
                opened_environments.append(recorder)
                return recorder

            factories.append(factory)

        base_vector_environment = DummyVecEnv(factories)
        environment_to_close = base_vector_environment
        vector_environment = VecTransposeImage(base_vector_environment)
        environment_to_close = vector_environment
        model = RecurrentPPO.load(
            source_checkpoint,
            env=vector_environment,
            device=args.device,
        )
        _verify_loaded_model(
            model,
            args,
            expected_actions=initial_trained,
            expected_updates=expected_updates,
            expected_policy_sha256=(
                resume.policy_tensor_sha256 if resume is not None else SOURCE_POLICY_TENSOR_SHA256
            ),
            expected_optimizer_sha256=(
                resume.optimizer_state_sha256
                if resume is not None
                else SOURCE_OPTIMIZER_STATE_SHA256
            ),
        )
        model.set_random_seed(segment_algorithm_seed)

        atomic_write_json(
            run_directory / "manifest.json",
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "started_at": started_at,
                "continuation": True,
                "parent": parent.public_dict(),
                "qualification": qualification.public_dict(),
                "source": source,
                "external_preregistration": anchor.public_dict(),
                "runtime": runtime_snapshot(),
                "storage_mount": mount_evidence,
                "arguments": vars(args)
                | {
                    "parent": str(args.parent),
                    "confirmation_report": str(args.confirmation_report),
                    "qualification_report": str(args.qualification_report),
                    "resume": (str(args.resume) if args.resume is not None else None),
                    "storage_root": str(args.storage_root),
                    "run_root": str(args.run_root),
                    "media_directory": (
                        str(args.media_directory) if args.media_directory is not None else None
                    ),
                },
                "effective_config": config,
                "exclusions": exclusion_evidence.public_dict(),
                "carried_exams": carried_exams,
                "segment": segment,
                "information_boundary": (
                    "56x56x3 partial RGB pixels plus private 256-unit recurrent state"
                ),
                "online_model_calls": False,
                "demonstrations": False,
                "oracle_actions_used_for_training": False,
                "confirmation_seed_access": False,
                "terminal_only_selection": True,
            },
        )

        callback_type = _CallbackFactory.create(BaseCallback)
        callback = callback_type(
            active_workers=evidence_wrappers,
            exclusions=exclusion_evidence,
            storage_mount=mount_evidence,
            run_directory=run_directory,
            state=state,
            controller=controller,
            scheduler=scheduler,
            parent=parent,
            qualification=qualification,
            source=source,
            segment=segment,
            effective_config=config,
            exam_model_loader=lambda path: RecurrentPPO.load(path, device="cpu"),
            validation_access=seed_access,
            evaluation_interval=args.evaluation_every,
            frame_interval=args.frame_every,
            minimum_free_bytes=minimum_free_bytes,
            started_at=started_at,
            initial_trained_actions=initial_trained,
            initial_updates=expected_updates,
            child_start_actions=SOURCE_U1_INHERITED_ACTIONS,
            keep_rolling_exams=args.keep_rolling_exams,
            initial_completed_allocation=(
                resume.last_completed_allocation if resume is not None else None
            ),
            initial_completed_allocation_valid=(
                resume.last_completed_allocation_valid if resume is not None else None
            ),
            storage_guard=storage_guard,
            source_guard=source_guard,
            staging_directory=args.run_root.parent / ".u2r-staging",
        )
        print(f"Run artifacts: {run_directory}", flush=True)
        model.learn(
            total_timesteps=remaining,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        callback.finalize("completed")
    except KeyboardInterrupt:
        if callback is not None:
            _best_effort(
                "interruption record",
                lambda: callback.finalize_failure("interrupted"),
            )
        else:
            _best_effort(
                "early interruption status",
                lambda: _write_setup_terminal_status(
                    run_directory,
                    phase="interrupted",
                    started_at=started_at,
                    wall_start=setup_wall_start,
                    parent=parent,
                    qualification=qualification,
                    source=source,
                    exclusions=exclusion_evidence,
                    external_preregistration=anchor.public_dict(),
                    storage_mount=mount_evidence,
                    initial_trained_actions=initial_trained,
                    child_trained_actions=child_trained,
                    remediation_trained_actions=remediation_trained,
                    expected_updates=expected_updates,
                    segment=segment,
                ),
            )
        raise SystemExit(130) from None
    except BaseException:
        failure_traceback = traceback.format_exc()
        _best_effort(
            "crash report",
            lambda: atomic_write_json(
                run_directory / "crash.json",
                {
                    "timestamp": utc_now(),
                    "classification": (
                        "training_failure" if callback is not None else "pre_training_setup_failure"
                    ),
                    "setup_complete": callback is not None,
                    "traceback": failure_traceback,
                },
            ),
        )
        if callback is not None:
            _best_effort(
                "crash status",
                lambda: callback.finalize_failure("crashed"),
            )
        else:
            _best_effort(
                "early crash status",
                lambda: _write_setup_terminal_status(
                    run_directory,
                    phase="crashed",
                    started_at=started_at,
                    wall_start=setup_wall_start,
                    parent=parent,
                    qualification=qualification,
                    source=source,
                    exclusions=exclusion_evidence,
                    external_preregistration=anchor.public_dict(),
                    storage_mount=mount_evidence,
                    initial_trained_actions=initial_trained,
                    child_trained_actions=child_trained,
                    remediation_trained_actions=remediation_trained,
                    expected_updates=expected_updates,
                    segment=segment,
                ),
            )
        raise
    finally:
        if environment_to_close is not None:
            _best_effort("environment close", environment_to_close.close)
        else:
            for environment in reversed(opened_environments):
                _best_effort("environment close", environment.close)


if __name__ == "__main__":
    main()
