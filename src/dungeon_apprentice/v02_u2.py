"""Frozen warm-start controller for U2 Separated Unlock.

This module deliberately owns no layout generator.  It authenticates one confirmed
U1 parent and one already-written U2 qualification report, then coordinates the
four-lesson learner supplied by :mod:`dungeon_apprentice.v02_u2_lessons`.

The most important ordering guarantee is:

    optimize -> publish immutable exam archive/sidecar -> reload by digest ->
    grade -> decide -> publish a separate resumable archive

An exam therefore never grades mutable in-memory policy bytes, and its sidecar is
never rewritten with the curriculum decision that its score caused.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import traceback
import uuid
import zipfile
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dungeon_apprentice import v02_u1_confirm as u1_confirmation
from dungeon_apprentice import v02_u1_confirm_v2 as u1_confirmation_v2
from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice.artifacts import (
    append_jsonl,
    atomic_copy_file,
    atomic_model_save,
    atomic_write_json,
    create_run_directory,
    ensure_disk_space,
    file_sha256,
    git_snapshot,
    runtime_snapshot,
    utc_now,
)
from dungeon_apprentice.contracts import MINIMUM_GAMMA
from dungeon_apprentice.u2_storage import audit_u2_storage

PROTOCOL = "dungeon-apprentice-v0.2-u2"
CHECKPOINT_SCHEMA_VERSION = 4
CHILD_ACTION_BUDGET = 1_048_576
COHORT_ACTION_BUDGET = 3_145_728
EVALUATION_INTERVAL = 32_768
ROLLOUT_STEPS = 512
WORKERS = 4
ROLLOUT_TRANSITIONS = ROLLOUT_STEPS * WORKERS
PPO_EPOCHS = 4
EVALUATION_SEED_COUNT = 80
MINIMUM_FREE_GIB = 25.0
KEEP_ROLLING_EXAMS = 5
POLICY_KWARGS = {"lstm_hidden_size": 256, "n_lstm_layers": 1}

CANONICAL_U1_CONFIRMATION = Path(
    "/Volumes/T7 Developer/DungeonApprentice/confirmations/"
    "v0.2-u1-v2-20260723/report.json"
)
EXPECTED_U1_CONFIRMATION_SHA256 = (
    "6e577170050f6f14599b793a031776a19bf7c64eba0f243f457298da3193ae8f"
)
EXPECTED_U1_CONFIRMATION_SOURCE = "ebf064afd6e6296bb21524103c2a3c269a56e7a5"
EXPECTED_U1_ATTEMPT_PROTOCOL = "dungeon-apprentice-v0.2-u1-confirmation-v2-attempt"


@dataclass(frozen=True)
class FrozenParent:
    u1_child_seed: int
    archive: Path
    archive_sha256: str
    u2_child_seed: int
    worker_streams: tuple[int, int, int, int]
    inherited_baseline: Mapping[str, int]


FROZEN_PARENTS: Mapping[int, FrozenParent] = {
    20260737: FrozenParent(
        u1_child_seed=20260725,
        archive=Path(
            "/Volumes/T7 Developer/DungeonApprentice/u1-local-20260722/"
            "v02-u1-lead-seed-20260725/checkpoints/mastered-local-unlock.zip"
        ),
        archive_sha256=(
            "bcce9b8251e97ed4fddda32871c891c3783c057bbb1f89deedb3a3d32058102a"
        ),
        u2_child_seed=20260737,
        worker_streams=(20260737, 20260738, 20260739, 20260740),
        inherited_baseline={
            "navigate/full": 74,
            "unlock/u0-visible": 80,
            "unlock/u1-local": 72,
        },
    ),
    20260741: FrozenParent(
        u1_child_seed=20260729,
        archive=Path(
            "/Volumes/T7 Developer/DungeonApprentice/u1-local-replication-20260722/"
            "v02-u1-replication-seed-20260729/checkpoints/mastered-local-unlock.zip"
        ),
        archive_sha256=(
            "2a300927b48f966d5f6ddfeefe13d2e444da1e5c70bcd54e86abd6a9b2d1830b"
        ),
        u2_child_seed=20260741,
        worker_streams=(20260741, 20260742, 20260743, 20260744),
        inherited_baseline={
            "navigate/full": 77,
            "unlock/u0-visible": 80,
            "unlock/u1-local": 72,
        },
    ),
    20260745: FrozenParent(
        u1_child_seed=20260733,
        archive=Path(
            "/Volumes/T7 Developer/DungeonApprentice/u1-local-replication-20260722/"
            "v02-u1-replication-seed-20260733/checkpoints/mastered-local-unlock.zip"
        ),
        archive_sha256=(
            "3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104"
        ),
        u2_child_seed=20260745,
        worker_streams=(20260745, 20260746, 20260747, 20260748),
        inherited_baseline={
            "navigate/full": 75,
            "unlock/u0-visible": 80,
            "unlock/u1-local": 72,
        },
    ),
}


class U2ProtocolError(RuntimeError):
    """Raised when U2 evidence or resumable state violates the frozen protocol."""


class U2BundleRollbackError(U2ProtocolError):
    """Raised when recovery bytes must remain staged for manual repair."""


@dataclass(frozen=True)
class ParentProvenance:
    checkpoint: str
    checkpoint_sha256: str
    sidecar: str
    manifest: str
    source_commit: str
    u1_child_seed: int
    u1_parent_seed: int
    trained_timesteps: int
    n_updates: int
    u2_child_seed: int
    worker_streams: tuple[int, int, int, int]
    confirmation_report: str
    confirmation_sha256: str
    confirmation_protocol: str
    confirmation_verdict: str
    confirmation_completed_at: str
    confirmation_source_commit: str
    attempt_ledger: str
    attempt_ledger_sha256: str
    checksum_file: str

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["worker_streams"] = list(self.worker_streams)
        return value


@dataclass(frozen=True)
class QualificationProvenance:
    report: str
    report_sha256: str
    source_commit: str
    evidence: Mapping[str, Any]
    _verified_report: Mapping[str, Any] | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )
    _seed_access: Any = dataclass_field(default=None, repr=False, compare=False)

    def public_dict(self) -> dict[str, Any]:
        return {
            "report": self.report,
            "report_sha256": self.report_sha256,
            "source_commit": self.source_commit,
            "evidence": dict(self.evidence),
        }

    def verified_report(self) -> dict[str, Any]:
        """Return the already-authenticated snapshot without rereading its path."""

        if self._verified_report is None:
            raise U2ProtocolError("qualification has no authenticated report snapshot")
        return json.loads(json.dumps(self._verified_report))

    def seed_access(self) -> Any:
        """Return validation access bound to the same authenticated report bytes."""

        if self._seed_access is None:
            raise U2ProtocolError("qualification has no bound validation access")
        return self._seed_access


@dataclass
class ControllerState:
    """Decision state that belongs to the trainer rather than the lesson registry."""

    last_decision_child_actions: int | None = None
    last_normal_pass_child_actions: int | None = None
    first_pass_artifact: dict[str, Any] | None = None
    mastery_artifact: dict[str, Any] | None = None
    below_twenty_at_half_budget: bool = False

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ControllerState:
        allowed = {
            "last_decision_child_actions",
            "last_normal_pass_child_actions",
            "first_pass_artifact",
            "mastery_artifact",
            "below_twenty_at_half_budget",
        }
        if set(value) != allowed:
            raise U2ProtocolError("U2 controller sidecar fields are incomplete or unknown")
        if not isinstance(value.get("below_twenty_at_half_budget"), bool):
            raise U2ProtocolError(
                "below_twenty_at_half_budget must be a JSON boolean"
            )
        try:
            state = cls(
                last_decision_child_actions=(
                    None
                    if value["last_decision_child_actions"] is None
                    else int(value["last_decision_child_actions"])
                ),
                last_normal_pass_child_actions=(
                    None
                    if value["last_normal_pass_child_actions"] is None
                    else int(value["last_normal_pass_child_actions"])
                ),
                first_pass_artifact=(
                    dict(value["first_pass_artifact"])
                    if value["first_pass_artifact"] is not None
                    else None
                ),
                mastery_artifact=(
                    dict(value["mastery_artifact"])
                    if value["mastery_artifact"] is not None
                    else None
                ),
                below_twenty_at_half_budget=bool(
                    value["below_twenty_at_half_budget"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise U2ProtocolError(f"invalid U2 controller state: {error}") from error
        for name in ("last_decision_child_actions", "last_normal_pass_child_actions"):
            boundary = getattr(state, name)
            if boundary is not None and (
                boundary < 0 or boundary % EVALUATION_INTERVAL
            ):
                raise U2ProtocolError(f"{name} is not a complete U2 exam boundary")
        for name in ("first_pass_artifact", "mastery_artifact"):
            artifact = getattr(state, name)
            if artifact is None:
                continue
            if set(artifact) != {
                "path",
                "checkpoint_sha256",
                "child_trained_actions",
            }:
                raise U2ProtocolError(f"{name} has invalid identity fields")
            _require_sha256(artifact["checkpoint_sha256"], f"{name} digest")
            actions = int(artifact["child_trained_actions"])
            if actions <= 0 or actions % EVALUATION_INTERVAL:
                raise U2ProtocolError(f"{name} is not at an exam boundary")
        return state


@dataclass(frozen=True)
class DecisionOutcome:
    message: str
    first_pass: bool = False
    mastered: bool = False


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise U2ProtocolError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise U2ProtocolError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise U2ProtocolError(f"{label} must be a JSON object: {path}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise U2ProtocolError(f"{label} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise U2ProtocolError(f"{label} is not a SHA-256 digest") from error
    return value


def _require_exact_path(actual: Path, expected: Path, label: str) -> Path:
    resolved = actual.expanduser().resolve()
    if resolved != expected.expanduser().resolve():
        raise U2ProtocolError(f"{label} is not the frozen path: {resolved}")
    return resolved


def _checkpoint_entry_passed(
    entry: Mapping[str, Any], *, selection: FrozenParent
) -> bool:
    checkpoint = entry.get("checkpoint")
    integrity = entry.get("integrity")
    if not isinstance(checkpoint, Mapping) or not isinstance(integrity, Mapping):
        return False
    return (
        entry.get("passed") is True
        and entry.get("policy_updates") is False
        and integrity.get("passed") is True
        and int(checkpoint.get("child_algorithm_seed", -1))
        == selection.u1_child_seed
        and checkpoint.get("checkpoint_sha256") == selection.archive_sha256
        and checkpoint.get("checkpoint") == str(selection.archive)
        and entry.get("checkpoint_sha256_before_load") == selection.archive_sha256
        and entry.get("checkpoint_sha256_after_evaluation")
        == selection.archive_sha256
        and entry.get("policy_tensor_sha256_before")
        == entry.get("policy_tensor_sha256_after")
        and entry.get("optimizer_state_sha256_before")
        == entry.get("optimizer_state_sha256_after")
        and entry.get("model_num_timesteps_before")
        == entry.get("model_num_timesteps_after")
        and entry.get("model_updates_before") == entry.get("model_updates_after")
    )


def verify_parent(
    checkpoint: Path,
    confirmation_report: Path,
    *,
    child_seed: int,
    attempt_ledger: Path | None = None,
    checksum_file: Path | None = None,
) -> ParentProvenance:
    """Authenticate one exact U1 mastery archive and its confirmed report entry."""

    selection = FROZEN_PARENTS.get(int(child_seed))
    if selection is None:
        raise U2ProtocolError(f"no frozen U2 lineage exists for child seed {child_seed}")
    archive = _require_exact_path(checkpoint, selection.archive, "U1 parent archive")
    try:
        verified = u1_confirmation.verify_u1_checkpoint(archive)
    except u1_confirmation.U1ConfirmationError as error:
        raise U2ProtocolError(f"U1 parent verification failed: {error}") from error
    if (
        int(verified.child_algorithm_seed) != selection.u1_child_seed
        or verified.checkpoint_sha256 != selection.archive_sha256
        or str(Path(verified.checkpoint).resolve()) != str(archive)
        or verified.digest_verified is not True
        or verified.lineage_verified is not True
        or verified.configuration_verified is not True
        or verified.source_verified is not True
    ):
        raise U2ProtocolError("U1 parent does not match the frozen U2 lineage")
    try:
        with zipfile.ZipFile(archive) as zipped:
            if "policy.optimizer.pth" not in zipped.namelist():
                raise U2ProtocolError("U1 parent archive has no optimizer state")
    except zipfile.BadZipFile as error:
        raise U2ProtocolError("U1 parent is not a readable PPO archive") from error

    report_path = _require_exact_path(
        confirmation_report,
        CANONICAL_U1_CONFIRMATION,
        "U1 confirmation report",
    )
    report_digest = file_sha256(report_path)
    if report_digest != EXPECTED_U1_CONFIRMATION_SHA256:
        raise U2ProtocolError("U1 confirmation report digest changed")
    report = _read_json(report_path, "U1 confirmation report")
    source = report.get("source")
    if (
        report.get("schema_version") != 2
        or report.get("protocol") != u1_confirmation_v2.CONFIRMATION_PROTOCOL
        or report.get("verdict") != "confirmed"
        or report.get("policy_updates") is not False
        or report.get("checkpoint_scoring_performed") is not True
        or not isinstance(source, Mapping)
        or source.get("dirty") is not False
        or source.get("commit") != EXPECTED_U1_CONFIRMATION_SOURCE
        or report.get("selected_child_seeds")
        != list(u1_confirmation.EXPECTED_CHILD_SEEDS)
    ):
        raise U2ProtocolError("U1 confirmation is not the frozen no-update result")
    matching = [
        item
        for item in report.get("checkpoints", ())
        if isinstance(item, Mapping)
        and _checkpoint_entry_passed(item, selection=selection)
    ]
    if len(matching) != 1:
        raise U2ProtocolError("selected U1 parent did not individually pass confirmation")

    ledger_path = (
        attempt_ledger.expanduser().resolve()
        if attempt_ledger is not None
        else report_path.parent / "attempt.json"
    )
    checksum_path = (
        checksum_file.expanduser().resolve()
        if checksum_file is not None
        else report_path.parent / "report.json.sha256"
    )
    ledger = _read_json(ledger_path, "U1 confirmation attempt ledger")
    if (
        ledger.get("schema_version") != 1
        or ledger.get("protocol") != EXPECTED_U1_ATTEMPT_PROTOCOL
        or ledger.get("status") != "completed_with_report"
        or ledger.get("source_dirty") is not False
        or ledger.get("source_commit") != EXPECTED_U1_CONFIRMATION_SOURCE
        or ledger.get("launcher_exit_status") != 0
        or ledger.get("evaluator_exit_status") != 0
        or ledger.get("report_present") is not True
        or ledger.get("report_verdict") != "confirmed"
        or ledger.get("report_sha256") != report_digest
        or Path(str(ledger.get("output", ""))).resolve() != report_path
        or str(selection.archive) not in ledger.get("checkpoints", ())
    ):
        raise U2ProtocolError("U1 confirmation attempt ledger is incomplete or changed")
    try:
        checksum_digest = checksum_path.read_text(encoding="utf-8").split()[0]
    except (OSError, IndexError) as error:
        raise U2ProtocolError(f"cannot read U1 confirmation checksum: {error}") from error
    if checksum_digest != report_digest:
        raise U2ProtocolError("U1 confirmation external checksum does not match")

    return ParentProvenance(
        checkpoint=str(archive),
        checkpoint_sha256=selection.archive_sha256,
        sidecar=str(verified.sidecar),
        manifest=str(verified.manifest),
        source_commit=str(verified.source_commit),
        u1_child_seed=selection.u1_child_seed,
        u1_parent_seed=int(verified.parent_training_seed),
        trained_timesteps=int(verified.trained_timesteps),
        n_updates=int(verified.n_updates),
        u2_child_seed=selection.u2_child_seed,
        worker_streams=selection.worker_streams,
        confirmation_report=str(report_path),
        confirmation_sha256=report_digest,
        confirmation_protocol=str(report["protocol"]),
        confirmation_verdict=str(report["verdict"]),
        confirmation_completed_at=str(report["completed_at"]),
        confirmation_source_commit=str(source["commit"]),
        attempt_ledger=str(ledger_path),
        attempt_ledger_sha256=file_sha256(ledger_path),
        checksum_file=str(checksum_path),
    )


def verify_qualification(
    path: Path,
    *,
    expected_source_commit: str,
    anchor: Any,
) -> QualificationProvenance:
    """Authenticate one report against the externally published Git-tag anchor."""

    from dungeon_apprentice.v02_u2_qualify import verify_u2_qualification_report

    anchor_public = (
        anchor.public_dict() if hasattr(anchor, "public_dict") else asdict(anchor)
    )
    if (
        anchor_public.get("source_commit") != expected_source_commit
        or Path(str(anchor_public.get("report", ""))).expanduser().resolve()
        != path.expanduser().resolve()
    ):
        raise U2ProtocolError(
            "U2 qualification anchor differs from the active source or report"
        )
    evidence = verify_u2_qualification_report(
        path,
        expected_source_commit=expected_source_commit,
        expected_sha256=str(anchor_public.get("report_sha256", "")),
        expected_attempt_id=str(anchor_public.get("attempt_id", "")),
        expected_claim_id=str(anchor_public.get("claim_id", "")),
    )
    public = evidence.public_dict()
    report_path = Path(str(public.get("report", ""))).expanduser().resolve()
    digest = _require_sha256(
        public.get("report_sha256"),
        "U2 qualification report digest",
    )
    source_commit = str(public.get("source_commit", expected_source_commit))
    if (
        source_commit != expected_source_commit
        or digest != anchor_public.get("report_sha256")
        or public.get("report_byte_length")
        != anchor_public.get("report_byte_length")
        or public.get("attempt_id") != anchor_public.get("attempt_id")
        or public.get("attempt_sha256") != anchor_public.get("attempt_sha256")
        or public.get("claim_id") != anchor_public.get("claim_id")
        or public.get("claim_sha256") != anchor_public.get("claim_sha256")
        or public.get("generator_profile")
        != anchor_public.get("generator_profile")
        or public.get("generator_profile_version")
        != anchor_public.get("generator_profile_version")
    ):
        raise U2ProtocolError(
            "U2 qualification evidence differs from its external anchor"
        )
    public["external_anchor"] = anchor_public
    return QualificationProvenance(
        report=str(report_path),
        report_sha256=digest,
        source_commit=source_commit,
        evidence=public,
        _verified_report=evidence.verified_report(),
        _seed_access=evidence.qualified_seed_access(),
    )


def _target_profiles() -> dict[str, dict[str, float]]:
    public = getattr(lessons, "target_profiles_public", None)
    if public is not None:
        return public()
    profiles = getattr(lessons, "TARGET_PROFILES", None)
    if profiles is None:
        weak_sets: Sequence[tuple[Any, ...]] = (
            (),
            (lessons.LessonId.NAVIGATE,),
            (lessons.LessonId.VISIBLE_UNLOCK,),
            (lessons.LessonId.LOCAL_UNLOCK,),
            (lessons.LessonId.NAVIGATE, lessons.LessonId.VISIBLE_UNLOCK),
            (lessons.LessonId.NAVIGATE, lessons.LessonId.LOCAL_UNLOCK),
            (lessons.LessonId.VISIBLE_UNLOCK, lessons.LessonId.LOCAL_UNLOCK),
            (
                lessons.LessonId.NAVIGATE,
                lessons.LessonId.VISIBLE_UNLOCK,
                lessons.LessonId.LOCAL_UNLOCK,
            ),
        )
        return {
            ("normal" if not weak else "+".join(item.value for item in weak)): {
                lesson.value: float(share)
                for lesson, share in lessons.CurriculumState(
                    weak_prerequisites=weak
                ).targets().items()
            }
            for weak in weak_sets
        }
    result: dict[str, dict[str, float]] = {}
    for weak, target in profiles.items():
        if isinstance(weak, str):
            label = weak
        else:
            label = (
                "normal"
                if not weak
                else "+".join(
                    item.value if hasattr(item, "value") else str(item) for item in weak
                )
            )
        result[label] = {
            item.value if hasattr(item, "value") else str(item): float(share)
            for item, share in target.items()
        }
    return result


def effective_config(
    args: argparse.Namespace,
    *,
    parent: ParentProvenance,
    qualification: QualificationProvenance,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "warm_start": True,
        "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "parent_u1_child_seed": parent.u1_child_seed,
        "child_algorithm_seed": int(args.seed),
        "initial_worker_streams": list(parent.worker_streams),
        "child_action_budget": int(args.child_budget),
        "qualification_report_sha256": qualification.report_sha256,
        "environment": {
            "size": int(args.size),
            "workers": int(args.workers),
            "generator_profile_version": lessons.GENERATOR_PROFILE_VERSION,
            "observation_shape": [56, 56, 3],
            "action_count": 7,
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
        },
        "lessons": [lesson.value for lesson in lessons.LessonId],
        "lesson_specs": {
            lesson.value: {
                "generator_profile": lessons.LESSON_SPECS[
                    lesson
                ].generator_profile,
                "horizon": lessons.LESSON_SPECS[lesson].max_steps,
                "validation_seed_base": lessons.LESSON_SPECS[
                    lesson
                ].validation_seed_base,
            }
            for lesson in lessons.LessonId
        },
        "transition_target_profiles": _target_profiles(),
        "artifacts": {
            "keep_rolling_exams": int(args.keep_rolling_exams),
            "minimum_free_gib": float(args.minimum_free_gib),
            "lineage_cap_gib": 2,
            "cohort_cap_gib": 6,
            "optional_media_cap_gib": 10,
        },
        "evaluation": {
            "interval": int(args.evaluation_every),
            "seeds_per_lesson": int(args.evaluation_seeds),
            "panels": [40, 40],
            "consecutive_mastery_passes": 2,
            "consecutive_recovery_passes": 2,
            "allocation_tolerance": float(lessons.ALLOCATION_TOLERANCE),
            "thresholds": {
                lesson.value: {
                    "overall": int(lessons.LESSON_SPECS[lesson].overall_required),
                    "panel": int(lessons.LESSON_SPECS[lesson].panel_required),
                }
                for lesson in lessons.LessonId
            },
        },
    }


@dataclass(frozen=True)
class ResumeBundle:
    checkpoint: Path
    sidecar: Mapping[str, Any]
    curriculum: Any
    controller: ControllerState
    scheduler_state: Mapping[str, Any]
    lifetime_trained: int
    child_trained: int
    n_updates: int
    segment_index: int
    last_completed_allocation: Mapping[str, Any] | None
    last_completed_allocation_valid: bool | None


def _integrity_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".integrity.json")


def _bundle_paths(checkpoint: Path) -> tuple[Path, Path, Path]:
    return (
        checkpoint,
        checkpoint.with_suffix(".json"),
        _integrity_path(checkpoint),
    )


def _regular_file_bytes(path: Path) -> int:
    if path.is_symlink():
        raise U2ProtocolError(f"U2 bundle path cannot be a symlink: {path}")
    try:
        metadata = path.stat()
    except FileNotFoundError:
        return 0
    if not path.is_file():
        raise U2ProtocolError(f"U2 bundle path is not a regular file: {path}")
    return int(metadata.st_size)


def _sync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_integrity(checkpoint: Path, sidecar: Path) -> dict[str, Any]:
    integrity = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_sha256(checkpoint),
        "sidecar": sidecar.name,
        "sidecar_sha256": file_sha256(sidecar),
    }
    atomic_write_json(_integrity_path(checkpoint), integrity)
    return integrity


def _verify_integrity(checkpoint: Path, sidecar: Path) -> Mapping[str, Any]:
    integrity = _read_json(_integrity_path(checkpoint), "U2 checkpoint integrity record")
    if (
        integrity.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or integrity.get("protocol") != PROTOCOL
        or integrity.get("checkpoint") != checkpoint.name
        or integrity.get("sidecar") != sidecar.name
        or integrity.get("checkpoint_sha256") != file_sha256(checkpoint)
        or integrity.get("sidecar_sha256") != file_sha256(sidecar)
    ):
        raise U2ProtocolError("U2 checkpoint integrity record does not match its bundle")
    return integrity


def _verify_named_decision_artifact(
    resume_archive: Path,
    artifact: Mapping[str, Any] | None,
    *,
    label: str,
) -> None:
    if artifact is None:
        return
    run_directory = resume_archive.parent.parent
    relative = Path(str(artifact["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise U2ProtocolError(f"{label} path escapes its U2 run directory")
    archive = (run_directory / relative).resolve()
    try:
        archive.relative_to(run_directory.resolve())
    except ValueError as error:
        raise U2ProtocolError(f"{label} path escapes its U2 run directory") from error
    if not archive.is_file():
        raise U2ProtocolError(f"{label} archive is missing: {archive}")
    if file_sha256(archive) != artifact["checkpoint_sha256"]:
        raise U2ProtocolError(f"{label} archive digest changed")
    _verify_integrity(archive, archive.with_suffix(".json"))


def load_resume(
    checkpoint: Path,
    *,
    expected_config: Mapping[str, Any],
    parent: ParentProvenance,
    qualification: QualificationProvenance,
    expected_source_commit: str,
) -> ResumeBundle:
    """Load only a complete schema-4 post-decision artifact."""

    archive = checkpoint.expanduser().resolve()
    if archive.suffix != ".zip":
        archive = archive.with_suffix(".zip")
    sidecar_path = archive.with_suffix(".json")
    sidecar = _read_json(sidecar_path, "U2 resume sidecar")
    _verify_integrity(archive, sidecar_path)
    if (
        sidecar.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or sidecar.get("protocol") != PROTOCOL
        or sidecar.get("resume_eligible") is not True
        or sidecar.get("kind") not in {"initial", "resume", "latest"}
    ):
        raise U2ProtocolError("checkpoint is not a resumable schema-4 U2 artifact")
    if sidecar.get("checkpoint_sha256") != file_sha256(archive):
        raise U2ProtocolError("U2 resume archive digest does not match its sidecar")
    if sidecar.get("effective_config") != dict(expected_config):
        raise U2ProtocolError("U2 resume configuration differs from the frozen child")
    if sidecar.get("parent") != parent.public_dict():
        raise U2ProtocolError("U2 resume parent provenance changed")
    if sidecar.get("qualification") != qualification.public_dict():
        raise U2ProtocolError("U2 resume qualification provenance changed")
    source = sidecar.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("dirty") is not False
        or source.get("commit") != expected_source_commit
    ):
        raise U2ProtocolError("U2 resume source provenance changed")
    try:
        curriculum = lessons.CurriculumState.from_dict(sidecar["curriculum"])
        controller = ControllerState.from_dict(sidecar["controller"])
        scheduler_state = dict(sidecar["scheduler"])
        progress = sidecar["progress"]
        if set(progress) != {
            "collected_actions",
            "trained_actions",
            "lifetime_trained_actions",
            "inherited_trained_actions",
            "child_trained_actions",
            "remaining_child_actions",
            "segment_trained_actions",
            "optimizer_updates",
        }:
            raise U2ProtocolError("U2 resume progress fields do not match schema 4")
        lifetime = int(progress["lifetime_trained_actions"])
        trained = int(progress["trained_actions"])
        child = int(progress["child_trained_actions"])
        collected = int(progress["collected_actions"])
        inherited = int(progress["inherited_trained_actions"])
        remaining = int(progress["remaining_child_actions"])
        segment_trained = int(progress["segment_trained_actions"])
        n_updates = int(progress["optimizer_updates"])
        segment = sidecar["segment"]
        segment_index = int(segment["index"])
        segment_start_lifetime = int(segment["start_lifetime_trained_actions"])
        segment_start_child = int(segment["start_child_trained_actions"])
        declared_segment_remaining = int(segment["remaining_child_actions"])
        last_allocation = sidecar.get("last_completed_allocation")
        last_allocation_valid = sidecar.get("last_completed_allocation_valid")
    except U2ProtocolError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise U2ProtocolError(f"invalid U2 resume sidecar: {error}") from error
    if collected != lifetime or trained != lifetime:
        raise U2ProtocolError("U2 resume is not a fully optimized boundary")
    if (
        inherited != parent.trained_timesteps
        or lifetime != parent.trained_timesteps + child
    ):
        raise U2ProtocolError("U2 child and lifetime action counters disagree")
    if not 0 <= child <= CHILD_ACTION_BUDGET:
        raise U2ProtocolError("U2 resume lies outside the child action ceiling")
    if remaining != CHILD_ACTION_BUDGET - child:
        raise U2ProtocolError("U2 resume remaining action budget changed")
    if child % ROLLOUT_TRANSITIONS:
        raise U2ProtocolError("U2 resume is not on a complete vector rollout")
    expected_updates = parent.n_updates + child // ROLLOUT_TRANSITIONS * PPO_EPOCHS
    if n_updates != expected_updates:
        raise U2ProtocolError("U2 optimizer counter disagrees with trained actions")
    if (
        controller.last_decision_child_actions is not None
        and controller.last_decision_child_actions > child
    ):
        raise U2ProtocolError("U2 decision state is ahead of trained experience")
    expected_segment_seed = parent.u2_child_seed + segment_index * 100_000
    expected_worker_streams = [
        seed + segment_index * 100_000 for seed in parent.worker_streams
    ]
    if (
        segment_index < 0
        or int(segment.get("lineage_algorithm_seed", -1))
        != parent.u2_child_seed
        or segment.get("initial_worker_streams") != list(parent.worker_streams)
        or int(segment.get("segment_algorithm_seed", -1))
        != expected_segment_seed
        or segment.get("segment_worker_streams") != expected_worker_streams
        or segment_start_lifetime != parent.trained_timesteps + segment_start_child
        or not 0 <= segment_start_child <= child
        or segment_trained != lifetime - segment_start_lifetime
        or declared_segment_remaining
        != CHILD_ACTION_BUDGET - segment_start_child
        or (
            segment_index == 0
            and segment.get("resume_checkpoint") is not None
        )
        or (
            segment_index > 0
            and not isinstance(segment.get("resume_checkpoint"), str)
        )
    ):
        raise U2ProtocolError("U2 resume segment lineage or counters changed")
    if controller.last_decision_child_actions is not None:
        if (
            not isinstance(last_allocation, Mapping)
            or not isinstance(last_allocation_valid, bool)
        ):
            raise U2ProtocolError("U2 decision resume lacks completed allocation")
        transitions = last_allocation.get("window_transitions")
        targets = last_allocation.get("target_shares")
        realized = last_allocation.get("realized_shares")
        structurally_invalid = (
            not isinstance(transitions, Mapping)
            or not isinstance(targets, Mapping)
            or not isinstance(realized, Mapping)
            or sum(int(value) for value in transitions.values())
            != EVALUATION_INTERVAL
            or set(targets) != {lesson.value for lesson in lessons.LessonId}
        )
        computed_valid = not structurally_invalid and not any(
                abs(float(realized.get(lesson, -1.0)) - float(target))
                > lessons.ALLOCATION_TOLERANCE + 1e-12
                for lesson, target in targets.items()
            )
        if structurally_invalid or computed_valid is not last_allocation_valid:
            raise U2ProtocolError("U2 completed allocation evidence is invalid")
    elif last_allocation_valid is not None:
        raise U2ProtocolError("U2 resume has an allocation verdict without a decision")
    first_pass_actions = (
        int(controller.first_pass_artifact["child_trained_actions"])
        if controller.first_pass_artifact is not None
        else None
    )
    mastery_actions = (
        int(controller.mastery_artifact["child_trained_actions"])
        if controller.mastery_artifact is not None
        else None
    )
    if curriculum.mastered:
        if (
            curriculum.consecutive_passes != 2
            or controller.first_pass_artifact is None
            or controller.mastery_artifact is None
            or controller.last_normal_pass_child_actions
            != controller.last_decision_child_actions
            or mastery_actions != controller.last_normal_pass_child_actions
            or first_pass_actions is None
            or first_pass_actions + EVALUATION_INTERVAL
            != controller.last_normal_pass_child_actions
        ):
            raise U2ProtocolError(
                "mastered U2 resume lacks the adjacent first-pass/mastery pair"
            )
    elif curriculum.consecutive_passes == 1:
        if (
            controller.first_pass_artifact is None
            or controller.mastery_artifact is not None
            or controller.last_normal_pass_child_actions
            != controller.last_decision_child_actions
            or first_pass_actions != controller.last_normal_pass_child_actions
        ):
            raise U2ProtocolError(
                "U2 mastery streak lacks its bound first-pass artifact"
            )
    elif (
        curriculum.consecutive_passes != 0
        or controller.first_pass_artifact is not None
        or controller.mastery_artifact is not None
    ):
        raise U2ProtocolError("U2 resume retained a stale mastery candidate")
    if curriculum.recovery and controller.last_normal_pass_child_actions is not None:
        raise U2ProtocolError("U2 recovery cannot retain a mastery streak boundary")
    _verify_named_decision_artifact(
        archive,
        controller.first_pass_artifact,
        label="first-pass",
    )
    _verify_named_decision_artifact(
        archive,
        controller.mastery_artifact,
        label="mastery",
    )
    return ResumeBundle(
        checkpoint=archive,
        sidecar=sidecar,
        curriculum=curriculum,
        controller=controller,
        scheduler_state=scheduler_state,
        lifetime_trained=lifetime,
        child_trained=child,
        n_updates=n_updates,
        segment_index=segment_index + 1,
        last_completed_allocation=(
            dict(last_allocation) if isinstance(last_allocation, Mapping) else None
        ),
        last_completed_allocation_valid=last_allocation_valid,
    )


def _carry_forward_decision_artifacts(
    resume: ResumeBundle,
    *,
    run_directory: Path,
    storage_admission: Callable[[int], Mapping[str, Any] | None] | None = None,
    storage_audit: Callable[[], Mapping[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    """Copy permanent decision bundles so a resume chain stays self-contained."""

    source_run = resume.checkpoint.parent.parent.resolve()
    carried: list[dict[str, Any]] = []
    for field, label in (
        ("first_pass_artifact", "first-pass"),
        ("mastery_artifact", "mastery"),
    ):
        artifact = getattr(resume.controller, field)
        if artifact is None:
            continue
        relative = Path(str(artifact["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise U2ProtocolError(f"{label} path escapes its source run")
        source = (source_run / relative).resolve()
        try:
            source.relative_to(source_run)
        except ValueError as error:
            raise U2ProtocolError(f"{label} path escapes its source run") from error
        destination = run_directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_sidecar = source.with_suffix(".json")
        source_integrity = _integrity_path(source)
        destination_sidecar = destination.with_suffix(".json")
        destination_integrity = _integrity_path(destination)
        if any(
            path.exists() or path.is_symlink()
            for path in (
                destination,
                destination_sidecar,
                destination_integrity,
            )
        ):
            raise U2ProtocolError(
                f"refusing to overwrite carried {label} artifact"
            )
        _verify_integrity(source, source_sidecar)
        bundle_bytes = sum(
            _regular_file_bytes(path)
            for path in (source, source_sidecar, source_integrity)
        )
        if storage_admission is not None:
            storage_admission(bundle_bytes)
        published: list[Path] = []
        try:
            for source_path, destination_path in (
                (source, destination),
                (source_sidecar, destination_sidecar),
                (source_integrity, destination_integrity),
            ):
                atomic_copy_file(source_path, destination_path)
                published.append(destination_path)
            _verify_integrity(destination, destination_sidecar)
            if storage_audit is not None:
                storage_audit()
        except BaseException:
            for published_path in reversed(published):
                published_path.unlink(missing_ok=True)
            if storage_audit is not None:
                storage_audit()
            raise
        if file_sha256(destination) != artifact["checkpoint_sha256"]:
            raise U2ProtocolError(f"carried {label} archive digest changed")
        carried.append(
            {
                "kind": label,
                "path": str(relative),
                "checkpoint_sha256": file_sha256(destination),
                "sidecar_sha256": file_sha256(destination_sidecar),
            }
        )
    return carried


def _safe_frame(path: Path, observation: np.ndarray) -> None:
    frame = observation
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _U2Mastered(RuntimeError):
    """Internal clean stop raised only after the complete mastery rule."""


class _CallbackFactory:
    """Build a callback without importing the optional training stack at module import."""

    @staticmethod
    def create(base_callback: Any) -> type[Any]:
        class U2Callback(base_callback):
            def __init__(
                self,
                *,
                run_directory: Path,
                state: Any,
                controller: ControllerState,
                scheduler: Any,
                parent: ParentProvenance,
                qualification: QualificationProvenance,
                source: Mapping[str, Any],
                segment: Mapping[str, Any],
                effective_config: Mapping[str, Any],
                exam_model_loader: Callable[[Path], Any],
                validation_access: Any,
                evaluation_interval: int,
                frame_interval: int,
                minimum_free_bytes: int,
                started_at: str,
                initial_trained_actions: int,
                initial_updates: int,
                child_start_actions: int,
                keep_rolling_exams: int = KEEP_ROLLING_EXAMS,
                initial_completed_allocation: Mapping[str, Any] | None = None,
                initial_completed_allocation_valid: bool | None = None,
                storage_guard: (
                    Callable[[int], Mapping[str, Any] | None] | None
                ) = None,
                source_guard: Callable[[], None] | None = None,
                staging_directory: Path | None = None,
            ) -> None:
                super().__init__(verbose=0)
                self.run_directory = run_directory
                self.state = state
                self.controller = controller
                self.scheduler = scheduler
                self.parent = parent
                self.qualification = qualification
                self.source = dict(source)
                self.segment = dict(segment)
                self.effective_config = dict(effective_config)
                self.exam_model_loader = exam_model_loader
                self.validation_access = validation_access
                self.evaluation_interval = int(evaluation_interval)
                self.frame_interval = int(frame_interval)
                self.minimum_free_bytes = int(minimum_free_bytes)
                self.started_at = started_at
                self.trained_actions = int(initial_trained_actions)
                self.last_updates = int(initial_updates)
                self.child_start_actions = int(child_start_actions)
                self.segment_start_actions = int(initial_trained_actions)
                self.keep_rolling_exams = int(keep_rolling_exams)
                self.last_completed_allocation = (
                    dict(initial_completed_allocation)
                    if initial_completed_allocation is not None
                    else None
                )
                self.last_completed_allocation_valid = (
                    initial_completed_allocation_valid
                )
                self.storage_guard = storage_guard
                self.source_guard = source_guard
                self.staging_directory = (
                    staging_directory
                    if staging_directory is not None
                    else run_directory.parent / f".{run_directory.name}-staging"
                )
                self.wall_start = time.monotonic()
                self.next_evaluation = self._next_child_boundary(
                    self.trained_actions, self.evaluation_interval
                )
                self.next_status = self.trained_actions
                self.next_frame = self.trained_actions
                self.episodes = 0
                self.frame_revision = 0
                self.latest_evaluations: list[dict[str, Any]] = []
                self.latest_optimizer: dict[str, Any] = {}
                self.evaluation_history: deque[dict[str, Any]] = deque(maxlen=200)
                self.recent_outcomes: dict[str, deque[bool]] = defaultdict(
                    lambda: deque(maxlen=100)
                )
                self.recent_behavior: deque[dict[str, Any]] = deque(maxlen=100)
                self.latest_safe_checkpoint: Path | None = None
                self.latest_safe_trained_actions: int | None = None
                self.latest_exam_checkpoint: Path | None = None
                self.latest_evaluated_checkpoint: dict[str, Any] | None = None
                self._last_evaluation_child_actions: int | None = None

            @property
            def child_trained_actions(self) -> int:
                return self.trained_actions - self.child_start_actions

            def _next_child_boundary(self, current: int, interval: int) -> int:
                child = current - self.child_start_actions
                return self.child_start_actions + ((child // interval) + 1) * interval

            def _guard_storage(
                self,
                anticipated_lineage_bytes: int = 0,
            ) -> Mapping[str, Any] | None:
                if self.source_guard is not None:
                    self.source_guard()
                ensure_disk_space(self.run_directory, self.minimum_free_bytes)
                if self.storage_guard is None:
                    return None
                return self.storage_guard(int(anticipated_lineage_bytes))

            def _new_staging_directory(self) -> Path:
                root = self.staging_directory.expanduser()
                root.mkdir(parents=True, exist_ok=True)
                if root.is_symlink() or not root.is_dir():
                    raise U2ProtocolError(
                        f"U2 staging root is not a regular directory: {root}"
                    )
                if root.resolve() == self.run_directory.resolve() or (
                    self.run_directory.resolve() in root.resolve().parents
                ):
                    raise U2ProtocolError(
                        "U2 staging must remain outside the scientific lineage"
                    )
                if root.stat().st_dev != self.run_directory.stat().st_dev:
                    raise U2ProtocolError(
                        "U2 staging and scientific lineage are on different filesystems"
                    )
                stage = root / uuid.uuid4().hex
                stage.mkdir(mode=0o700)
                return stage

            def _publish_staged_bundle(
                self,
                staged_checkpoint: Path,
                destination: Path,
                *,
                replace: bool,
            ) -> None:
                staged_paths = _bundle_paths(staged_checkpoint)
                destination_paths = _bundle_paths(destination)
                if tuple(path.name for path in staged_paths) != tuple(
                    path.name for path in destination_paths
                ):
                    raise U2ProtocolError(
                        "staged and destination U2 bundle names differ"
                    )
                candidate_bytes = sum(
                    _regular_file_bytes(path) for path in staged_paths
                )
                if candidate_bytes <= 0:
                    raise U2ProtocolError("staged U2 bundle is empty")
                existing_bytes = sum(
                    _regular_file_bytes(path) for path in destination_paths
                )
                if not replace and existing_bytes:
                    raise U2ProtocolError(
                        f"refusing to overwrite immutable U2 artifact: {destination}"
                    )
                anticipated_growth = max(0, candidate_bytes - existing_bytes)
                self._guard_storage(anticipated_growth)

                destination.parent.mkdir(parents=True, exist_ok=True)
                backups = tuple(
                    staged_checkpoint.parent / f"previous-{path.name}"
                    for path in destination_paths
                )
                published: list[Path] = []
                backed_up: list[tuple[Path, Path]] = []
                preserve_failed_rollback_backups = False
                try:
                    for old, backup in zip(
                        destination_paths,
                        backups,
                        strict=True,
                    ):
                        if old.exists():
                            os.replace(old, backup)
                            backed_up.append((old, backup))
                    for staged, published_path in zip(
                        staged_paths,
                        destination_paths,
                        strict=True,
                    ):
                        os.replace(staged, published_path)
                        published.append(published_path)
                    _sync_directory(destination.parent)
                    self._guard_storage()
                except BaseException as publication_error:
                    rollback_errors: list[str] = []
                    for published_path in reversed(published):
                        try:
                            published_path.unlink(missing_ok=True)
                        except OSError as error:
                            rollback_errors.append(str(error))
                    for old, backup in reversed(backed_up):
                        try:
                            os.replace(backup, old)
                        except OSError as error:
                            rollback_errors.append(str(error))
                    _sync_directory(destination.parent)
                    if rollback_errors:
                        preserve_failed_rollback_backups = True
                        raise U2BundleRollbackError(
                            "U2 bundle publication failed and rollback was incomplete: "
                            + " | ".join(rollback_errors)
                            + "; preserved recovery backups: "
                            + ", ".join(
                                str(backup)
                                for _old, backup in backed_up
                                if backup.exists()
                            )
                        ) from publication_error
                    raise
                finally:
                    if not preserve_failed_rollback_backups:
                        for backup in backups:
                            backup.unlink(missing_ok=True)

            def _on_training_start(self) -> None:
                if int(self.model.num_timesteps) != self.trained_actions:
                    raise U2ProtocolError(
                        "model and U2 controller start-action boundaries differ"
                    )
                if int(self.model._n_updates) != self.last_updates:
                    raise U2ProtocolError(
                        "model and U2 controller optimizer boundaries differ"
                    )
                initial = self._publish_live_archive(
                    "initial",
                    kind="initial",
                    resume_eligible=True,
                    replace=False,
                )
                self.latest_safe_checkpoint = initial
                self.latest_safe_trained_actions = self.trained_actions
                self._evaluate_archive(
                    initial,
                    trigger="diagnostic_baseline",
                    decide=False,
                )
                self._write_status("training")

            def _on_rollout_start(self) -> None:
                self._guard_storage()
                if int(self.model._n_updates) > self.last_updates:
                    self._process_optimized_boundary()
                if self.state.mastered:
                    raise _U2Mastered

            def _on_step(self) -> bool:
                for done, info in zip(
                    self.locals.get("dones", ()),
                    self.locals.get("infos", ()),
                    strict=False,
                ):
                    if not done:
                        continue
                    lesson_id = str(
                        info.get(
                            "lesson_id",
                            lessons.LessonId.SEPARATED_UNLOCK.value,
                        )
                    )
                    success = bool(info.get("success", False))
                    self.recent_outcomes[lesson_id].append(success)
                    self.recent_behavior.append(
                        {
                            "lesson_id": lesson_id,
                            "success": success,
                            "terminal_reason": str(
                                info.get("terminal_reason") or "unknown"
                            ),
                            "elapsed_steps": int(
                                info.get("elapsed_steps", 0) or 0
                            ),
                            "coverage": float(
                                info.get(
                                    "coverage",
                                    info.get("unique_cells", 0),
                                )
                                or 0
                            ),
                            "collisions": int(info.get("collisions", 0) or 0),
                            "ineffective_interactions": int(
                                info.get("ineffective_interactions", 0) or 0
                            ),
                            "largest_action_share": float(
                                info.get("largest_action_share", 0.0) or 0.0
                            ),
                            "longest_repeated_action_run": int(
                                info.get("longest_repeated_action_run", 0) or 0
                            ),
                            "extrinsic_return": float(
                                info.get("extrinsic_return", 0.0) or 0.0
                            ),
                            "curiosity_return": float(
                                info.get("curiosity_return", 0.0) or 0.0
                            ),
                        }
                    )
                    self.episodes += 1
                    append_jsonl(
                        self.run_directory / "episodes.jsonl",
                        {
                            "timestamp": utc_now(),
                            "episode": self.episodes,
                            "collected_actions": int(self.model.num_timesteps),
                            "trained_actions": self.trained_actions,
                            "child_collected_actions": (
                                int(self.model.num_timesteps) - self.child_start_actions
                            ),
                            "child_trained_actions": self.child_trained_actions,
                            "lesson_id": lesson_id,
                            "lesson_label": info.get("lesson_label"),
                            "seed": info.get("seed"),
                            "layout_sha256": info.get("layout_sha256"),
                            "geometry_sha256": info.get("geometry_sha256"),
                            "success": success,
                            "terminal_reason": info.get("terminal_reason"),
                            "milestones": info.get("milestones"),
                            "key_visible_initially": info.get(
                                "initial_key_visible",
                                info.get("key_visible_initially"),
                            ),
                            "door_visible_initially": info.get(
                                "initial_door_visible",
                                info.get("door_visible_initially"),
                            ),
                            "path_actions": info.get("path_actions"),
                            "oracle_actions": info.get("oracle_actions"),
                            "coverage": info.get("coverage", info.get("unique_cells")),
                            "collisions": info.get("collisions"),
                            "ineffective_interactions": info.get(
                                "ineffective_interactions"
                            ),
                            "action_histogram": info.get("action_histogram"),
                            "largest_action_share": info.get("largest_action_share"),
                            "longest_repeated_action_run": info.get(
                                "longest_repeated_action_run"
                            ),
                            "extrinsic_return": info.get("extrinsic_return"),
                            "curiosity_return": info.get("curiosity_return"),
                        },
                    )
                current = int(self.num_timesteps)
                if current >= self.next_frame:
                    observations = self.locals.get("new_obs")
                    if observations is not None and len(observations):
                        _safe_frame(
                            self.run_directory / "frames" / "latest.png",
                            observations[0],
                        )
                        self.frame_revision += 1
                    self.next_frame = current + self.frame_interval
                if current >= self.next_status:
                    self._write_status("training")
                    self.next_status = current + 1_000
                return True

            def _process_optimized_boundary(self) -> None:
                collected = int(self.model.num_timesteps)
                updates = int(self.model._n_updates)
                delta_actions = collected - self.trained_actions
                delta_updates = updates - self.last_updates
                if (
                    delta_actions <= 0
                    or delta_actions % ROLLOUT_TRANSITIONS
                    or delta_updates
                    != delta_actions // ROLLOUT_TRANSITIONS * PPO_EPOCHS
                ):
                    raise U2ProtocolError(
                        "U2 observed an incomplete or inconsistent PPO update boundary"
                    )
                self.trained_actions = collected
                self.last_updates = updates
                if not 0 <= self.child_trained_actions <= CHILD_ACTION_BUDGET:
                    raise U2ProtocolError("U2 child crossed its frozen action ceiling")
                self._record_optimizer_metrics(delta_updates)
                if self.trained_actions >= self.next_evaluation:
                    if self.child_trained_actions % self.evaluation_interval:
                        raise U2ProtocolError(
                            "U2 exam was reached away from a frozen child boundary"
                        )
                    exam = self._publish_live_archive(
                        f"rolling/exam-{self.child_trained_actions:07d}",
                        kind="exam",
                        resume_eligible=False,
                        replace=False,
                    )
                    self.latest_exam_checkpoint = exam
                    self._evaluate_archive(exam, trigger="scheduled", decide=True)
                    self._prune_rolling_exams()
                    self.next_evaluation = self._next_child_boundary(
                        self.trained_actions, self.evaluation_interval
                    )
                latest = self._publish_live_archive(
                    "latest-safe",
                    kind="latest",
                    resume_eligible=True,
                    replace=True,
                )
                self.latest_safe_checkpoint = latest
                self.latest_safe_trained_actions = self.trained_actions
                self._write_status("mastered" if self.state.mastered else "training")

            def _sidecar(
                self,
                *,
                kind: str,
                checkpoint_sha256: str,
                resume_eligible: bool,
                exam_source: Mapping[str, Any] | None = None,
            ) -> dict[str, Any]:
                collected = int(self.model.num_timesteps)
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
                    "curriculum": self.state.public_dict(),
                    "controller": self.controller.public_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "last_completed_allocation": self.last_completed_allocation,
                    "last_completed_allocation_valid": (
                        self.last_completed_allocation_valid
                    ),
                    "progress": {
                        "collected_actions": collected,
                        "trained_actions": self.trained_actions,
                        "lifetime_trained_actions": self.trained_actions,
                        "inherited_trained_actions": self.child_start_actions,
                        "child_trained_actions": self.child_trained_actions,
                        "remaining_child_actions": (
                            CHILD_ACTION_BUDGET - self.child_trained_actions
                        ),
                        "segment_trained_actions": (
                            self.trained_actions - self.segment_start_actions
                        ),
                        "optimizer_updates": int(self.model._n_updates),
                    },
                    "segment": dict(self.segment),
                    "exam_source": dict(exam_source) if exam_source is not None else None,
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

            def _publish_live_archive(
                self,
                name: str,
                *,
                kind: str,
                resume_eligible: bool,
                replace: bool,
            ) -> Path:
                if int(self.model.num_timesteps) != self.trained_actions:
                    raise U2ProtocolError(
                        "refusing U2 publication outside a trained boundary"
                    )
                destination = self.run_directory / "checkpoints" / f"{name}.zip"
                if not replace and (
                    destination.exists()
                    or destination.with_suffix(".json").exists()
                    or _integrity_path(destination).exists()
                ):
                    raise U2ProtocolError(
                        f"refusing to overwrite immutable U2 artifact: {destination}"
                )
                stage = self._new_staging_directory()
                staged_checkpoint = stage / destination.name
                preserve_stage = False
                try:
                    atomic_model_save(self.model, staged_checkpoint)
                    self._write_bundle_sidecar(
                        staged_checkpoint,
                        kind=kind,
                        resume_eligible=resume_eligible,
                    )
                    self._publish_staged_bundle(
                        staged_checkpoint,
                        destination,
                        replace=replace,
                    )
                except U2BundleRollbackError:
                    preserve_stage = True
                    raise
                finally:
                    if not preserve_stage:
                        shutil.rmtree(stage, ignore_errors=True)
                self._record_checkpoint_event(destination, kind)
                return destination

            def _copy_exam_artifact(
                self,
                exam: Path,
                name: str,
                *,
                kind: str,
                resume_eligible: bool,
                replace: bool,
                exam_integrity: Mapping[str, Any],
            ) -> Path:
                destination = self.run_directory / "checkpoints" / f"{name}.zip"
                if not replace and (
                    destination.exists()
                    or destination.with_suffix(".json").exists()
                    or _integrity_path(destination).exists()
                ):
                    raise U2ProtocolError(
                        f"refusing to overwrite named U2 artifact: {destination}"
                )
                stage = self._new_staging_directory()
                staged_checkpoint = stage / destination.name
                preserve_stage = False
                atomic_copy_file(exam, staged_checkpoint)
                source_record = {
                    "checkpoint": str(exam.relative_to(self.run_directory)),
                    "checkpoint_sha256": exam_integrity["checkpoint_sha256"],
                    "sidecar": str(
                        exam.with_suffix(".json").relative_to(self.run_directory)
                    ),
                    "sidecar_sha256": exam_integrity["sidecar_sha256"],
                }
                try:
                    self._write_bundle_sidecar(
                        staged_checkpoint,
                        kind=kind,
                        resume_eligible=resume_eligible,
                        exam_source=source_record,
                    )
                    self._publish_staged_bundle(
                        staged_checkpoint,
                        destination,
                        replace=replace,
                    )
                except U2BundleRollbackError:
                    preserve_stage = True
                    raise
                finally:
                    if not preserve_stage:
                        shutil.rmtree(stage, ignore_errors=True)
                if file_sha256(destination) != file_sha256(exam):
                    raise U2ProtocolError("copied U2 artifact bytes changed")
                self._record_checkpoint_event(destination, kind)
                return destination

            def _record_checkpoint_event(self, path: Path, kind: str) -> None:
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": "checkpoint",
                        "kind": kind,
                        "trained_actions": self.trained_actions,
                        "child_trained_actions": self.child_trained_actions,
                        "sha256": file_sha256(path),
                        "sidecar_sha256": file_sha256(path.with_suffix(".json")),
                        "path": str(path.relative_to(self.run_directory)),
                    },
                )

            def _load_exam_model(
                self, exam: Path, exam_sidecar: Mapping[str, Any]
            ) -> Any:
                expected_digest = str(exam_sidecar["checkpoint_sha256"])
                if file_sha256(exam) != expected_digest:
                    raise U2ProtocolError("exam archive changed before reload")
                graded = self.exam_model_loader(exam)
                if (
                    int(graded.num_timesteps)
                    != int(exam_sidecar["progress"]["trained_actions"])
                    or int(graded._n_updates)
                    != int(exam_sidecar["progress"]["optimizer_updates"])
                ):
                    raise U2ProtocolError(
                        "reloaded exam model counters differ from its sidecar"
                    )
                return graded

            def _validation_seeds(self, lesson: Any) -> Sequence[int]:
                return lessons.validation_seeds(
                    lesson, access=self.validation_access
                )

            def _evaluate_archive(
                self,
                exam: Path,
                *,
                trigger: str,
                decide: bool,
            ) -> None:
                sidecar_path = exam.with_suffix(".json")
                exam_sidecar = _read_json(sidecar_path, "immutable U2 exam sidecar")
                integrity = _verify_integrity(exam, sidecar_path)
                archive_before = file_sha256(exam)
                sidecar_before = file_sha256(sidecar_path)
                graded_model = self._load_exam_model(exam, exam_sidecar)
                allocation_before = self.scheduler.snapshot()
                allocation_transitions = sum(
                    int(value)
                    for value in allocation_before.get(
                        "window_transitions", {}
                    ).values()
                )
                allocation_complete = (
                    not decide or allocation_transitions == EVALUATION_INTERVAL
                )
                allocation_ok = (
                    allocation_complete and self.scheduler.allocation_within()
                )
                evaluations: dict[Any, Any] = {}
                for lesson in lessons.LessonId:
                    frame_path = (
                        self.run_directory
                        / "frames"
                        / f"exam-{lesson.value.replace('/', '-')}.png"
                    )
                    result = lessons.evaluate_lesson(
                        graded_model,
                        lesson,
                        self._validation_seeds(lesson),
                        frame_path=frame_path,
                        seed_access=self.validation_access,
                    )
                    evaluations[lesson] = result
                    append_jsonl(
                        self.run_directory / "evaluations.jsonl",
                        {
                            "timestamp": utc_now(),
                            "trigger": trigger,
                            "counts_toward_gate": decide,
                            "trained_actions": self.trained_actions,
                            "child_trained_actions": self.child_trained_actions,
                            "optimizer_updates": int(self.model._n_updates),
                            "checkpoint": str(exam.relative_to(self.run_directory)),
                            "checkpoint_sha256": archive_before,
                            "sidecar": str(sidecar_path.relative_to(self.run_directory)),
                            "sidecar_sha256": sidecar_before,
                            "reloaded_for_grading": True,
                            "passed": lessons.lesson_passed(result),
                            **result.public_dict(),
                        },
                    )
                if (
                    file_sha256(exam) != archive_before
                    or file_sha256(sidecar_path) != sidecar_before
                ):
                    raise U2ProtocolError("immutable exam bundle changed during grading")
                if (
                    int(self.model.num_timesteps) != self.trained_actions
                    or int(self.model._n_updates) != self.last_updates
                ):
                    raise U2ProtocolError("grading changed the live training model")
                self.latest_evaluated_checkpoint = {
                    "path": str(exam.relative_to(self.run_directory)),
                    "checkpoint_sha256": archive_before,
                    "sidecar_sha256": sidecar_before,
                    "trained_actions": self.trained_actions,
                    "child_trained_actions": self.child_trained_actions,
                    "optimizer_updates": int(self.model._n_updates),
                    "trigger": trigger,
                    "counts_toward_gate": decide,
                }

                if not decide and self.child_trained_actions == 0:
                    expected = FROZEN_PARENTS[
                        self.parent.u2_child_seed
                    ].inherited_baseline
                    for lesson in (
                        lessons.LessonId.NAVIGATE,
                        lessons.LessonId.VISIBLE_UNLOCK,
                        lessons.LessonId.LOCAL_UNLOCK,
                    ):
                        observed = int(evaluations[lesson].successes)
                        required = int(expected[lesson.value])
                        if observed != required:
                            raise U2ProtocolError(
                                "U2 parent baseline does not reproduce its frozen "
                                f"boundary: {lesson.value}={observed}/80, "
                                f"expected {required}/80"
                            )

                decision = DecisionOutcome(
                    "Diagnostic four-lesson baseline; cannot count toward a gate"
                )
                if decide:
                    if (
                        self._last_evaluation_child_actions
                        == self.child_trained_actions
                    ):
                        raise U2ProtocolError(
                            "duplicate U2 decision at one trained boundary"
                        )
                    self.last_completed_allocation = dict(allocation_before)
                    self.last_completed_allocation_valid = allocation_ok
                    decision = self._apply_decision(
                        evaluations,
                        allocation_ok=allocation_ok,
                        child_actions=self.child_trained_actions,
                    )
                    self._last_evaluation_child_actions = self.child_trained_actions

                    if decision.first_pass:
                        candidate = (
                            self.run_directory
                            / "checkpoints"
                            / "first-pass-separated-unlock.zip"
                        )
                        self.controller.first_pass_artifact = {
                            "path": str(candidate.relative_to(self.run_directory)),
                            "checkpoint_sha256": archive_before,
                            "child_trained_actions": self.child_trained_actions,
                        }
                        candidate = self._copy_exam_artifact(
                            exam,
                            "first-pass-separated-unlock",
                            kind="first_pass",
                            resume_eligible=False,
                            replace=True,
                            exam_integrity=integrity,
                        )
                    if decision.mastered:
                        mastery = (
                            self.run_directory
                            / "checkpoints"
                            / "mastered-separated-unlock.zip"
                        )
                        self.controller.mastery_artifact = {
                            "path": str(mastery.relative_to(self.run_directory)),
                            "checkpoint_sha256": archive_before,
                            "child_trained_actions": self.child_trained_actions,
                        }
                        mastery = self._copy_exam_artifact(
                            exam,
                            "mastered-separated-unlock",
                            kind="mastery",
                            resume_eligible=False,
                            replace=False,
                            exam_integrity=integrity,
                        )
                    if not self.state.mastered:
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
                            "lesson_id": result.lesson_id,
                            "lesson_label": result.lesson_label,
                            "successes": result.successes,
                            "episodes": result.episodes,
                            "success_rate": result.success_rate,
                            "panel_successes": result.panel_successes,
                            "decision": (
                                decision.message
                                if lesson is lessons.LessonId.SEPARATED_UNLOCK
                                else "Retention exam"
                            ),
                            "counts_toward_gate": decide,
                            "checkpoint_sha256": archive_before,
                        }
                    )
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": (
                            "curriculum_decision" if decide else "baseline_evaluation"
                        ),
                        "trained_actions": self.trained_actions,
                        "child_trained_actions": self.child_trained_actions,
                        "decision": decision.message,
                        "checkpoint": str(exam.relative_to(self.run_directory)),
                        "checkpoint_sha256": archive_before,
                        "sidecar_sha256": sidecar_before,
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

            def _apply_decision(
                self,
                evaluations: Mapping[Any, Any],
                *,
                allocation_ok: bool,
                child_actions: int,
            ) -> DecisionOutcome:
                if child_actions <= 0 or child_actions % EVALUATION_INTERVAL:
                    raise U2ProtocolError(
                        "U2 gate decisions require a post-update exam boundary"
                    )
                previous = self.controller.last_decision_child_actions
                if previous is not None and child_actions - previous != EVALUATION_INTERVAL:
                    raise U2ProtocolError(
                        "U2 decision boundaries are not separated by one complete window"
                    )
                self.controller.last_decision_child_actions = child_actions
                weak = tuple(lessons.weak_prerequisites(evaluations))
                in_recovery = bool(tuple(self.state.weak_prerequisites))

                if child_actions == CHILD_ACTION_BUDGET // 2:
                    u2_result = evaluations[lessons.LessonId.SEPARATED_UNLOCK]
                    self.controller.below_twenty_at_half_budget = (
                        int(u2_result.successes) < 20
                    )

                if in_recovery:
                    self.state.consecutive_passes = 0
                    self.controller.last_normal_pass_child_actions = None
                    self.controller.first_pass_artifact = None
                    if weak:
                        changed = tuple(self.state.weak_prerequisites) != weak
                        self.state.weak_prerequisites = weak
                        self.state.recovery_passes = 0
                        self.state.change()
                        prefix = "changed" if changed else "continues"
                        return DecisionOutcome(
                            "Recovery "
                            f"{prefix}: {','.join(item.value for item in weak)} below gate"
                        )
                    if not allocation_ok:
                        self.state.recovery_passes = 0
                        self.state.change()
                        return DecisionOutcome(
                            "Recovery held: transition allocation outside tolerance"
                        )
                    self.state.recovery_passes += 1
                    self.state.change()
                    if self.state.recovery_passes < 2:
                        return DecisionOutcome(
                            "First clean recovery boundary; confirmation required"
                        )
                    self.state.weak_prerequisites = ()
                    self.state.recovery_passes = 0
                    self.state.consecutive_passes = 0
                    return DecisionOutcome(
                        "Prerequisites recovered; normal U2 practice resumes"
                    )

                if weak:
                    self.state.weak_prerequisites = weak
                    self.state.recovery_passes = 0
                    self.state.consecutive_passes = 0
                    self.controller.last_normal_pass_child_actions = None
                    self.controller.first_pass_artifact = None
                    self.state.change()
                    return DecisionOutcome(
                        "Recovery started: "
                        f"{','.join(item.value for item in weak)} below gate"
                    )

                u2_passed = lessons.lesson_passed(
                    evaluations[lessons.LessonId.SEPARATED_UNLOCK]
                )
                if not allocation_ok or not u2_passed:
                    self.state.consecutive_passes = 0
                    self.controller.last_normal_pass_child_actions = None
                    self.controller.first_pass_artifact = None
                    self.state.change()
                    return DecisionOutcome(
                        "Held: transition allocation outside tolerance"
                        if not allocation_ok
                        else "Held: Separated Unlock below gate"
                    )

                prior_pass = self.controller.last_normal_pass_child_actions
                if (
                    self.state.consecutive_passes
                    and (
                        prior_pass is None
                        or child_actions - prior_pass != EVALUATION_INTERVAL
                    )
                ):
                    raise U2ProtocolError(
                        "U2 mastery streak lacks one full intervening training window"
                    )
                if self.state.consecutive_passes:
                    candidate = self.controller.first_pass_artifact
                    candidate_actions = (
                        int(candidate.get("child_trained_actions", -1))
                        if isinstance(candidate, Mapping)
                        else -1
                    )
                    if (
                        candidate_actions != prior_pass
                        or child_actions - candidate_actions != EVALUATION_INTERVAL
                    ):
                        raise U2ProtocolError(
                            "U2 mastery candidate is not the immediately preceding "
                            "normal exam boundary"
                        )
                elif self.controller.first_pass_artifact is not None:
                    raise U2ProtocolError(
                        "U2 controller retained a stale first-pass candidate"
                    )
                self.state.consecutive_passes += 1
                self.controller.last_normal_pass_child_actions = child_actions
                self.state.change()
                if self.state.consecutive_passes == 1:
                    return DecisionOutcome(
                        "First cumulative U2 pass; confirmation required",
                        first_pass=self.controller.first_pass_artifact is None,
                    )
                if self.state.consecutive_passes != 2:
                    raise U2ProtocolError("U2 mastery streak exceeded its frozen rule")
                self.state.mastered = True
                return DecisionOutcome(
                    "U2 Separated Unlock mastered with Navigate, U0, and U1 retained",
                    mastered=True,
                )

            def _record_optimizer_metrics(self, updates_since_boundary: int) -> None:
                values = getattr(self.model.logger, "name_to_value", {})
                names = (
                    "entropy_loss",
                    "policy_gradient_loss",
                    "value_loss",
                    "approx_kl",
                    "clip_fraction",
                    "explained_variance",
                    "loss",
                )
                self.latest_optimizer = {
                    name: (
                        float(values[f"train/{name}"])
                        if values.get(f"train/{name}") is not None
                        else None
                    )
                    for name in names
                }
                append_jsonl(
                    self.run_directory / "optimizer.jsonl",
                    {
                        "timestamp": utc_now(),
                        "collected_actions": int(self.model.num_timesteps),
                        "trained_actions": self.trained_actions,
                        "child_trained_actions": self.child_trained_actions,
                        "optimizer_updates": int(self.model._n_updates),
                        "updates_since_boundary": updates_since_boundary,
                        **self.latest_optimizer,
                    },
                )

            def _prune_rolling_exams(self) -> None:
                rolling = self.run_directory / "checkpoints" / "rolling"
                archives = sorted(rolling.glob("exam-*.zip"))
                excess = max(0, len(archives) - self.keep_rolling_exams)
                for archive in archives[:excess]:
                    archive.unlink(missing_ok=True)
                    archive.with_suffix(".json").unlink(missing_ok=True)
                    _integrity_path(archive).unlink(missing_ok=True)

            def _storage_status(self) -> Mapping[str, Any] | None:
                try:
                    return (
                        self.storage_guard(0)
                        if self.storage_guard is not None
                        else None
                    )
                except BaseException as error:
                    return {"error": str(error)}

            def _behavior_status(self) -> dict[str, Any]:
                records = list(self.recent_behavior)
                terminal_reasons: dict[str, int] = {}
                for record in records:
                    reason = str(record["terminal_reason"])
                    terminal_reasons[reason] = terminal_reasons.get(reason, 0) + 1

                def mean(field: str) -> float | None:
                    if not records:
                        return None
                    return sum(float(record[field]) for record in records) / len(
                        records
                    )

                return {
                    "window_episodes": len(records),
                    "successes": sum(
                        1 for record in records if record["success"]
                    ),
                    "success_rate": (
                        sum(1 for record in records if record["success"])
                        / len(records)
                        if records
                        else None
                    ),
                    "mean_elapsed_steps": mean("elapsed_steps"),
                    "mean_coverage": mean("coverage"),
                    "mean_collisions": mean("collisions"),
                    "mean_ineffective_interactions": mean(
                        "ineffective_interactions"
                    ),
                    "mean_largest_action_share": mean("largest_action_share"),
                    "longest_repeated_action_run": (
                        max(
                            int(record["longest_repeated_action_run"])
                            for record in records
                        )
                        if records
                        else None
                    ),
                    "mean_extrinsic_return": mean("extrinsic_return"),
                    "mean_curiosity_return": mean("curiosity_return"),
                    "terminal_reasons": terminal_reasons,
                }

            def _write_status(self, phase: str) -> None:
                elapsed = time.monotonic() - self.wall_start
                collected = int(self.model.num_timesteps)
                recent = {
                    lessons.LESSON_SPECS[lesson].label: {
                        "episodes": len(self.recent_outcomes[lesson.value]),
                        "success_rate": (
                            sum(self.recent_outcomes[lesson.value])
                            / len(self.recent_outcomes[lesson.value])
                            if self.recent_outcomes[lesson.value]
                            else None
                        ),
                    }
                    for lesson in lessons.LessonId
                }
                atomic_write_json(
                    self.run_directory / "status.json",
                    {
                        "protocol": PROTOCOL,
                        "phase": phase,
                        "active_lineage": self.parent.u2_child_seed,
                        "parent": self.parent.public_dict(),
                        "qualification": self.qualification.public_dict(),
                        "source": dict(self.source),
                        "parent_checkpoint_sha256": self.parent.checkpoint_sha256,
                        "qualification_sha256": self.qualification.report_sha256,
                        "source_commit": self.source.get("commit"),
                        "started_at": self.started_at,
                        "updated_at": utc_now(),
                        "elapsed_seconds": elapsed,
                        "actions_per_second": (
                            (collected - self.segment_start_actions) / elapsed
                            if elapsed
                            else 0.0
                        ),
                        "fps": (
                            (collected - self.segment_start_actions) / elapsed
                            if elapsed
                            else 0.0
                        ),
                        "collected_actions": collected,
                        "collected_timesteps": collected,
                        "trained_actions": self.trained_actions,
                        "trained_timesteps": self.trained_actions,
                        "inherited_trained_actions": self.child_start_actions,
                        "child_collected_actions": (
                            collected - self.child_start_actions
                        ),
                        "child_collected_timesteps": (
                            collected - self.child_start_actions
                        ),
                        "child_trained_actions": self.child_trained_actions,
                        "child_trained_timesteps": self.child_trained_actions,
                        "remaining_child_actions": (
                            CHILD_ACTION_BUDGET - self.child_trained_actions
                        ),
                        "remaining_action_budget": (
                            CHILD_ACTION_BUDGET - self.child_trained_actions
                        ),
                        "action_cap": CHILD_ACTION_BUDGET,
                        "lifetime_trained_actions": self.trained_actions,
                        "lifetime_trained_timesteps": self.trained_actions,
                        "optimizer_updates": int(self.model._n_updates),
                        "episodes": self.episodes,
                        "current_lesson_id": (
                            lessons.LessonId.SEPARATED_UNLOCK.value
                        ),
                        "current_lesson_label": (
                            lessons.LESSON_SPECS[
                                lessons.LessonId.SEPARATED_UNLOCK
                            ].label
                        ),
                        "curriculum": self.state.public_dict(),
                        "controller": self.controller.public_dict(),
                        "practice_allocation": self.scheduler.snapshot(),
                        "last_completed_practice_allocation": (
                            self.last_completed_allocation
                        ),
                        "last_completed_allocation_valid": (
                            self.last_completed_allocation_valid
                        ),
                        "recent_training": recent,
                        "recent_behavior": self._behavior_status(),
                        "latest_optimizer": self.latest_optimizer,
                        "next_evaluation": self.next_evaluation,
                        "latest_evaluations": self.latest_evaluations,
                        "evaluation_history": list(self.evaluation_history),
                        "lesson_frames": {
                            lesson.value: (
                                f"frames/exam-{lesson.value.replace('/', '-')}.png"
                            )
                            for lesson in lessons.LessonId
                        },
                        "frame_revision": self.frame_revision,
                        "latest_exam_checkpoint": (
                            str(
                                self.latest_exam_checkpoint.relative_to(
                                    self.run_directory
                                )
                            )
                            if self.latest_exam_checkpoint
                            else None
                        ),
                        "latest_evaluated_checkpoint": (
                            dict(self.latest_evaluated_checkpoint)
                            if self.latest_evaluated_checkpoint is not None
                            else None
                        ),
                        "latest_safe_checkpoint": (
                            str(
                                self.latest_safe_checkpoint.relative_to(
                                    self.run_directory
                                )
                            )
                            if self.latest_safe_checkpoint
                            else None
                        ),
                        "latest_safe_trained_actions": (
                            self.latest_safe_trained_actions
                        ),
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

            def finalize(self, phase: str) -> None:
                if int(self.model._n_updates) > self.last_updates:
                    self._process_optimized_boundary()
                if int(self.model.num_timesteps) != self.trained_actions:
                    raise U2ProtocolError(
                        "U2 run ended with an untrained partial rollout"
                    )
                if (
                    not self.state.mastered
                    and self.child_trained_actions != CHILD_ACTION_BUDGET
                ):
                    raise U2ProtocolError(
                        "U2 scientific run stopped before mastery or its action ceiling"
                    )
                if (
                    self.latest_safe_checkpoint is None
                    or self.latest_safe_trained_actions != self.trained_actions
                ):
                    raise U2ProtocolError("U2 has no post-decision resumable boundary")
                terminal = self._copy_exam_artifact(
                    self.latest_safe_checkpoint,
                    "terminal",
                    kind="terminal",
                    resume_eligible=False,
                    replace=False,
                    exam_integrity=_verify_integrity(
                        self.latest_safe_checkpoint,
                        self.latest_safe_checkpoint.with_suffix(".json"),
                    ),
                )
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": "run_terminal",
                        "phase": "mastered" if self.state.mastered else phase,
                        "terminal_checkpoint": str(
                            terminal.relative_to(self.run_directory)
                        ),
                        "child_trained_actions": self.child_trained_actions,
                    },
                )
                self._write_status("mastered" if self.state.mastered else phase)

            def _pending_optimizer_delta(self) -> tuple[int, int, str]:
                delta_actions = int(self.model.num_timesteps) - self.trained_actions
                delta_updates = int(self.model._n_updates) - self.last_updates
                if delta_actions == 0 and delta_updates == 0:
                    classification = "none"
                elif (
                    delta_actions > 0
                    and delta_actions % ROLLOUT_TRANSITIONS == 0
                    and delta_updates
                    == delta_actions // ROLLOUT_TRANSITIONS * PPO_EPOCHS
                ):
                    classification = "complete"
                elif delta_actions >= 0 and delta_updates >= 0:
                    classification = "partial"
                else:
                    classification = "counter_regression"
                return delta_actions, delta_updates, classification

            def finalize_failure(self, phase: str) -> None:
                (
                    pending_actions,
                    pending_updates,
                    boundary_classification,
                ) = self._pending_optimizer_delta()
                recovered_complete_actions = 0
                recovered_complete_updates = 0
                recovery_error: str | None = None
                if boundary_classification == "complete":
                    try:
                        self._process_optimized_boundary()
                    except BaseException as error:
                        recovery_error = f"{type(error).__name__}: {error}"
                        boundary_classification = "complete_recovery_failed"
                    else:
                        recovered_complete_actions = pending_actions
                        recovered_complete_updates = pending_updates
                        boundary_classification = "complete_recovered"
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": f"run_{phase}",
                        "collected_actions": int(self.model.num_timesteps),
                        "trained_actions": self.trained_actions,
                        "child_trained_actions": self.child_trained_actions,
                        "pending_optimizer_boundary": boundary_classification,
                        "recovered_complete_actions": recovered_complete_actions,
                        "recovered_complete_optimizer_updates": (
                            recovered_complete_updates
                        ),
                        "discarded_partial_actions": (
                            max(0, pending_actions)
                            if boundary_classification
                            in {"partial", "counter_regression"}
                            else 0
                        ),
                        "discarded_partial_optimizer_updates": (
                            max(0, pending_updates)
                            if boundary_classification
                            in {"partial", "counter_regression"}
                            else 0
                        ),
                        "unpublished_complete_actions": (
                            max(0, pending_actions)
                            if boundary_classification == "complete_recovery_failed"
                            else 0
                        ),
                        "unpublished_complete_optimizer_updates": (
                            max(0, pending_updates)
                            if boundary_classification == "complete_recovery_failed"
                            else 0
                        ),
                        "recovery_error": recovery_error,
                        "latest_safe_checkpoint": (
                            str(
                                self.latest_safe_checkpoint.relative_to(
                                    self.run_directory
                                )
                            )
                            if self.latest_safe_checkpoint
                            else None
                        ),
                        "latest_safe_trained_actions": (
                            self.latest_safe_trained_actions
                        ),
                    },
                )
                self._write_status(phase)

        return U2Callback


def _environment_factory(
    *,
    scheduler: Any,
    seed: int,
    size: int,
    seed_access: Any,
    forbidden_layout_hashes: Mapping[Any, frozenset[str]],
) -> Callable[[], Any]:
    return lambda: lessons.make_training_env(
        scheduler=scheduler,
        seed=seed,
        size=size,
        seed_access=seed_access,
        forbidden_layout_hashes=forbidden_layout_hashes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path)
    parser.add_argument(
        "--confirmation-report",
        type=Path,
        default=CANONICAL_U1_CONFIRMATION,
    )
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=20260737)
    parser.add_argument("--child-budget", type=int, default=CHILD_ACTION_BUDGET)
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
    parser.add_argument(
        "--evaluation-seeds", type=int, default=EVALUATION_SEED_COUNT
    )
    parser.add_argument("--frame-every", type=int, default=2_048)
    parser.add_argument(
        "--keep-rolling-exams", type=int, default=KEEP_ROLLING_EXAMS
    )
    parser.add_argument("--minimum-free-gib", type=float, default=MINIMUM_FREE_GIB)
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=Path("/Volumes/T7 Developer"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "/Volumes/T7 Developer/DungeonApprentice/u2-separated-20260723"
        ),
    )
    parser.add_argument("--media-directory", type=Path)
    parser.add_argument("--run-name")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    selection = FROZEN_PARENTS.get(int(args.seed))
    if selection is None:
        raise SystemExit(
            f"U2 child seed must be one of {tuple(FROZEN_PARENTS)}, got {args.seed}"
        )
    positive = {
        "child action budget": args.child_budget,
        "workers": args.workers,
        "rollout steps": args.rollout_steps,
        "batch size": args.batch_size,
        "epochs": args.n_epochs,
        "evaluation interval": args.evaluation_every,
        "evaluation cases": args.evaluation_seeds,
        "frame interval": args.frame_every,
        "rolling exam retention": args.keep_rolling_exams,
    }
    for label, value in positive.items():
        if int(value) <= 0:
            raise SystemExit(f"{label} must be positive")
    rollout = int(args.workers) * int(args.rollout_steps)
    if rollout % int(args.batch_size):
        raise SystemExit("workers x rollout steps must be divisible by batch size")
    if int(args.child_budget) % rollout:
        raise SystemExit("child action budget must align to a complete vector rollout")
    if int(args.child_budget) % int(args.evaluation_every):
        raise SystemExit("child action budget must end on an exam boundary")
    if not math.isfinite(float(args.minimum_free_gib)) or float(
        args.minimum_free_gib
    ) < MINIMUM_FREE_GIB:
        raise SystemExit("U2 requires a free-space reserve of at least 25 GiB")
    if not math.isfinite(float(args.gamma)) or not MINIMUM_GAMMA <= float(
        args.gamma
    ) <= 1.0:
        raise SystemExit("gamma violates reward dominance")
    if not math.isfinite(float(args.learning_rate)) or float(
        args.learning_rate
    ) <= 0:
        raise SystemExit("learning rate must be finite and positive")
    if int(args.keep_rolling_exams) > KEEP_ROLLING_EXAMS:
        raise SystemExit("U2 permits at most five ordinary rolling exams")
    frozen = {
        "child action budget": (args.child_budget, CHILD_ACTION_BUDGET),
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
        "device": (args.device, "cpu"),
    }
    deviations = [
        f"{label}={actual!r} (requires {expected!r})"
        for label, (actual, expected) in frozen.items()
        if actual != expected
    ]
    if deviations:
        raise SystemExit(
            "frozen U2 configuration changed:\n  - " + "\n  - ".join(deviations)
        )
    if args.parent is not None and args.parent.expanduser().resolve() != (
        selection.archive.expanduser().resolve()
    ):
        raise SystemExit("the requested parent does not match the frozen U2 child")


def _verify_loaded_model(
    model: Any,
    args: argparse.Namespace,
    *,
    expected_actions: int,
    expected_updates: int,
) -> None:
    if int(model.num_timesteps) != int(expected_actions):
        raise U2ProtocolError(
            f"model/sidecar action mismatch: {model.num_timesteps} != "
            f"{expected_actions}"
        )
    if int(model._n_updates) != int(expected_updates):
        raise U2ProtocolError(
            f"model/sidecar optimizer mismatch: {model._n_updates} != "
            f"{expected_updates}"
        )
    optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
    if optimizer is None or not optimizer.state:
        raise U2ProtocolError("loaded U1/U2 archive has no restored optimizer state")
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
        raise U2ProtocolError(
            "loaded archive does not match the frozen U2 optimization contract"
        )


def _best_effort(label: str, operation: Callable[[], Any]) -> None:
    try:
        operation()
    except BaseException as error:
        print(f"Warning: {label} failed during finalization: {error}", flush=True)


def _exit_after_interruption(callback: Any) -> None:
    """Record an incomplete segment and stop a sequential launcher."""

    _best_effort(
        "interruption record",
        lambda: callback.finalize_failure("interrupted"),
    )
    raise SystemExit(130)


def _write_setup_terminal_status(
    run_directory: Path,
    *,
    phase: str,
    started_at: str,
    wall_start: float,
    parent: ParentProvenance,
    qualification: QualificationProvenance,
    source: Mapping[str, Any],
    initial_trained_actions: int,
    child_trained_actions: int,
    expected_updates: int,
    segment: Mapping[str, Any] | None,
) -> None:
    """Leave a dashboard-readable terminal record before a callback exists."""

    now = utc_now()
    atomic_write_json(
        run_directory / "status.json",
        {
            "protocol": PROTOCOL,
            "phase": phase,
            "setup_complete": False,
            "active_lineage": parent.u2_child_seed,
            "parent": parent.public_dict(),
            "qualification": qualification.public_dict(),
            "source": dict(source),
            "parent_checkpoint_sha256": parent.checkpoint_sha256,
            "qualification_sha256": qualification.report_sha256,
            "source_commit": source.get("commit"),
            "started_at": started_at,
            "updated_at": now,
            "elapsed_seconds": max(0.0, time.monotonic() - wall_start),
            "collected_actions": initial_trained_actions,
            "collected_timesteps": initial_trained_actions,
            "trained_actions": initial_trained_actions,
            "trained_timesteps": initial_trained_actions,
            "inherited_trained_actions": parent.trained_timesteps,
            "child_collected_actions": child_trained_actions,
            "child_collected_timesteps": child_trained_actions,
            "child_trained_actions": child_trained_actions,
            "child_trained_timesteps": child_trained_actions,
            "remaining_child_actions": CHILD_ACTION_BUDGET
            - child_trained_actions,
            "remaining_action_budget": CHILD_ACTION_BUDGET
            - child_trained_actions,
            "action_cap": CHILD_ACTION_BUDGET,
            "lifetime_trained_actions": initial_trained_actions,
            "lifetime_trained_timesteps": initial_trained_actions,
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
            "optimizer_updates": expected_updates,
        },
    )


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    selection = FROZEN_PARENTS[int(args.seed)]
    parent_path = args.parent or selection.archive

    repository = Path(__file__).resolve().parents[2]
    source = git_snapshot(repository)
    if (
        source.get("dirty") is not False
        or not isinstance(source.get("commit"), str)
        or not source["commit"]
    ):
        raise SystemExit("U2 requires one clean identified source commit")

    # All provenance and qualification checks precede child directory or policy load.
    from dungeon_apprentice.u2_qualification_anchor import verify_external_anchor

    qualification_anchor = verify_external_anchor(repository)
    parent = verify_parent(
        parent_path,
        args.confirmation_report,
        child_seed=args.seed,
    )
    qualification = verify_qualification(
        args.qualification_report,
        expected_source_commit=str(source["commit"]),
        anchor=qualification_anchor,
    )
    seed_access = qualification.seed_access()
    qualification_report = qualification.verified_report()
    forbidden_layout_hashes = lessons.reserved_training_layout_hashes(
        qualification_report,
        access=seed_access,
    )

    config = effective_config(
        args,
        parent=parent,
        qualification=qualification,
    )
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
    initial_trained = (
        resume.lifetime_trained if resume is not None else parent.trained_timesteps
    )
    child_trained = resume.child_trained if resume is not None else 0
    if resume is not None and resume.curriculum.mastered:
        raise SystemExit("the selected U2 child is already mastered")
    remaining = CHILD_ACTION_BUDGET - child_trained
    if remaining <= 0:
        raise SystemExit("the U2 child action budget is already exhausted")

    minimum_free_bytes = int(float(args.minimum_free_gib) * 1024**3)
    ensure_disk_space(args.run_root, minimum_free_bytes)
    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    except ImportError as error:
        raise SystemExit(
            'Install training dependencies with: pip install -e ".[train]"'
        ) from error

    run_directory = create_run_directory(args.run_root, args.run_name)
    started_at = utc_now()
    setup_wall_start = time.monotonic()
    expected_updates = resume.n_updates if resume is not None else parent.n_updates
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
                Path(parent.confirmation_report),
                Path(parent.attempt_ledger),
                Path(parent.checksum_file),
                Path(qualification.report),
            ),
            anticipated_lineage_bytes=anticipated_lineage_bytes,
        ).as_dict()

    def source_guard() -> None:
        current = git_snapshot(repository)
        if (
            current.get("dirty") is not False
            or current.get("commit") != source["commit"]
        ):
            raise U2ProtocolError(
                "U2 source became dirty or changed during the active child"
            )

    try:
        state = (
            resume.curriculum if resume is not None else lessons.CurriculumState()
        )
        controller = (
            resume.controller if resume is not None else ControllerState()
        )
        segment_index = resume.segment_index if resume is not None else 0
        segment_offset = segment_index * 100_000
        segment_algorithm_seed = int(args.seed) + segment_offset
        segment_worker_streams = tuple(
            int(seed) + segment_offset for seed in parent.worker_streams
        )
        segment = {
            "id": uuid.uuid4().hex,
            "index": segment_index,
            "started_at": started_at,
            "lineage_algorithm_seed": int(args.seed),
            "initial_worker_streams": list(parent.worker_streams),
            "segment_algorithm_seed": segment_algorithm_seed,
            "segment_worker_streams": list(segment_worker_streams),
            "start_lifetime_trained_actions": initial_trained,
            "start_child_trained_actions": child_trained,
            "remaining_child_actions": remaining,
            "resume_checkpoint": (
                str(resume.checkpoint) if resume is not None else None
            ),
        }
        scheduler = lessons.TransitionDeficitScheduler(
            state,
            seed=segment_algorithm_seed + 90_000,
        )
        if resume is not None:
            scheduler.load_state_dict(resume.scheduler_state)

        storage_guard()
        carried_decision_artifacts = (
            _carry_forward_decision_artifacts(
                resume,
                run_directory=run_directory,
                storage_admission=storage_guard,
                storage_audit=lambda: storage_guard(0),
            )
            if resume is not None
            else []
        )
        factories = [
            _environment_factory(
                scheduler=scheduler,
                seed=seed,
                size=args.size,
                seed_access=seed_access,
                forbidden_layout_hashes=forbidden_layout_hashes,
            )
            for seed in segment_worker_streams
        ]

        def tracked_factory(factory: Callable[[], Any]) -> Callable[[], Any]:
            def create() -> Any:
                environment = factory()
                opened_environments.append(environment)
                return environment

            return create

        base_vector_environment = DummyVecEnv(
            [tracked_factory(factory) for factory in factories]
        )
        environment_to_close = base_vector_environment
        vector_environment = VecTransposeImage(base_vector_environment)
        environment_to_close = vector_environment
        source_checkpoint = (
            resume.checkpoint
            if resume is not None
            else Path(parent.checkpoint)
        )
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
        )
        model.set_random_seed(segment_algorithm_seed)

        atomic_write_json(
            run_directory / "manifest.json",
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "started_at": started_at,
                "warm_start": True,
                "lineage_order": tuple(FROZEN_PARENTS).index(int(args.seed)) + 1,
                "parent": parent.public_dict(),
                "qualification": qualification.public_dict(),
                "source": source,
                "runtime": runtime_snapshot(),
                "arguments": vars(args)
                | {
                    "parent": str(parent_path),
                    "confirmation_report": str(args.confirmation_report),
                    "qualification_report": str(args.qualification_report),
                    "resume": (
                        str(args.resume) if args.resume is not None else None
                    ),
                    "storage_root": str(args.storage_root),
                    "run_root": str(args.run_root),
                    "media_directory": (
                        str(args.media_directory)
                        if args.media_directory is not None
                        else None
                    ),
                },
                "effective_config": config,
                "carried_decision_artifacts": carried_decision_artifacts,
                "segment": segment,
                "information_boundary": (
                    "56x56x3 partial RGB pixels plus private 256-unit recurrent state"
                ),
                "online_model_calls": False,
                "demonstrations": False,
                "oracle_actions_used_for_training": False,
            },
        )

        callback_type = _CallbackFactory.create(BaseCallback)
        callback = callback_type(
            run_directory=run_directory,
            state=state,
            controller=controller,
            scheduler=scheduler,
            parent=parent,
            qualification=qualification,
            source=source,
            segment=segment,
            effective_config=config,
            exam_model_loader=lambda path: RecurrentPPO.load(
                path,
                device="cpu",
            ),
            validation_access=seed_access,
            evaluation_interval=args.evaluation_every,
            frame_interval=args.frame_every,
            minimum_free_bytes=minimum_free_bytes,
            started_at=started_at,
            initial_trained_actions=initial_trained,
            initial_updates=expected_updates,
            child_start_actions=parent.trained_timesteps,
            keep_rolling_exams=args.keep_rolling_exams,
            initial_completed_allocation=(
                resume.last_completed_allocation if resume is not None else None
            ),
            initial_completed_allocation_valid=(
                resume.last_completed_allocation_valid
                if resume is not None
                else None
            ),
            storage_guard=storage_guard,
            source_guard=source_guard,
            staging_directory=args.run_root.parent / ".u2-staging",
        )
        print(f"Run artifacts: {run_directory}", flush=True)

        phase = "completed"
        try:
            model.learn(
                total_timesteps=remaining,
                callback=callback,
                reset_num_timesteps=False,
                progress_bar=False,
            )
        except _U2Mastered:
            phase = "mastered"
        callback.finalize(phase)
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
                    initial_trained_actions=initial_trained,
                    child_trained_actions=child_trained,
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
                        "training_failure"
                        if callback is not None
                        else "pre_training_setup_failure"
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
                    initial_trained_actions=initial_trained,
                    child_trained_actions=child_trained,
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
