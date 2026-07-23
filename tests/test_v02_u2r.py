from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import gymnasium as gym
import numpy as np
import pytest

from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v02_u2r as u2r
from dungeon_apprentice.artifacts import (
    append_jsonl,
    atomic_write_json,
    file_sha256,
)


def _sha(label: str) -> str:
    return u2r.hashlib.sha256(label.encode()).hexdigest()


def _write_parent_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[u2r.U2rParent, dict[str, Path], dict]:
    lineage = tmp_path / "lineage"
    checkpoint = lineage / "checkpoints" / "mastered-separated-unlock.zip"
    checkpoint.parent.mkdir(parents=True)
    with ZipFile(checkpoint, "w") as zipped:
        zipped.writestr("policy.optimizer.pth", b"complete optimizer")
        zipped.writestr("policy.pth", b"policy")
    monkeypatch.setattr(u2r, "SOURCE_ARCHIVE", checkpoint.resolve())
    monkeypatch.setattr(u2r, "SOURCE_ARCHIVE_SHA256", file_sha256(checkpoint))
    monkeypatch.setattr(
        u2r,
        "SOURCE_POLICY_MEMBER_SHA256",
        u2r.hashlib.sha256(b"policy").hexdigest(),
    )

    source = {
        "commit": u2r.SOURCE_TRAINING_COMMIT,
        "dirty": False,
    }
    effective_config = {"frozen": True}
    sidecar = {
        "schema_version": u2r.frozen_u2.CHECKPOINT_SCHEMA_VERSION,
        "protocol": u2r.frozen_u2.PROTOCOL,
        "kind": "mastery",
        "resume_eligible": False,
        "checkpoint_sha256": u2r.SOURCE_ARCHIVE_SHA256,
        "source": source,
        "effective_config": effective_config,
        "parent": {"u2_child_seed": u2r.SOURCE_CHILD_SEED},
        "progress": {
            "child_trained_actions": u2r.SOURCE_CHILD_ACTIONS,
            "lifetime_trained_actions": u2r.SOURCE_LIFETIME_ACTIONS,
            "inherited_trained_actions": u2r.SOURCE_U1_INHERITED_ACTIONS,
            "optimizer_updates": u2r.SOURCE_OPTIMIZER_UPDATES,
        },
        "curriculum": {"mastered": True},
        "controller": {
            "mastery_artifact": {
                "checkpoint_sha256": u2r.SOURCE_ARCHIVE_SHA256,
            }
        },
        "scheduler": {"rng_state": {"bit_generator": "PCG64"}},
    }
    sidecar_path = checkpoint.with_suffix(".json")
    atomic_write_json(sidecar_path, sidecar)
    monkeypatch.setattr(u2r, "SOURCE_SIDECAR_SHA256", file_sha256(sidecar_path))

    integrity_path = checkpoint.with_suffix(".integrity.json")
    atomic_write_json(
        integrity_path,
        {
            "schema_version": u2r.frozen_u2.CHECKPOINT_SCHEMA_VERSION,
            "protocol": u2r.frozen_u2.PROTOCOL,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": u2r.SOURCE_ARCHIVE_SHA256,
            "sidecar": sidecar_path.name,
            "sidecar_sha256": u2r.SOURCE_SIDECAR_SHA256,
        },
    )
    monkeypatch.setattr(
        u2r,
        "SOURCE_INTEGRITY_SHA256",
        file_sha256(integrity_path),
    )

    manifest_path = lineage / "manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "protocol": u2r.frozen_u2.PROTOCOL,
            "source": source,
            "effective_config": effective_config,
        },
    )
    monkeypatch.setattr(u2r, "SOURCE_MANIFEST_SHA256", file_sha256(manifest_path))

    evaluations_path = lineage / "evaluations.jsonl"
    evaluations_path.write_text(
        "".join(
            json.dumps(
                {
                    "child_trained_actions": u2r.SOURCE_CHILD_ACTIONS,
                    "counts_toward_gate": True,
                    "lesson_id": lesson_id,
                    "checkpoint_sha256": u2r.SOURCE_ARCHIVE_SHA256,
                    "successes": expected["successes"],
                    "panel_successes": list(expected["panel_successes"]),
                    "mean_ineffective_interactions": expected["mean_ineffective_interactions"],
                },
                sort_keys=True,
            )
            + "\n"
            for lesson_id, expected in u2r.SOURCE_BASELINE.items()
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(u2r, "SOURCE_EVALUATIONS", evaluations_path.resolve())
    monkeypatch.setattr(
        u2r,
        "SOURCE_EVALUATIONS_SHA256",
        file_sha256(evaluations_path),
    )

    confirmation = tmp_path / "confirmation"
    confirmation.mkdir()
    report_path = confirmation / "report.json"
    entry = {
        "checkpoint": {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": u2r.SOURCE_ARCHIVE_SHA256,
            "child_seed": u2r.SOURCE_CHILD_SEED,
            "child_trained_actions": u2r.SOURCE_CHILD_ACTIONS,
            "lifetime_trained_actions": u2r.SOURCE_LIFETIME_ACTIONS,
            "optimizer_updates": u2r.SOURCE_OPTIMIZER_UPDATES,
            "artifact_sha256s": {
                "evaluations": u2r.SOURCE_EVALUATIONS_SHA256,
            },
        },
        "policy_updates": False,
        "passed": False,
        "policy_tensor_sha256_before": u2r.SOURCE_POLICY_TENSOR_SHA256,
        "policy_tensor_sha256_after": u2r.SOURCE_POLICY_TENSOR_SHA256,
        "optimizer_state_sha256_before": u2r.SOURCE_OPTIMIZER_STATE_SHA256,
        "optimizer_state_sha256_after": u2r.SOURCE_OPTIMIZER_STATE_SHA256,
        "model_num_timesteps_before": u2r.SOURCE_LIFETIME_ACTIONS,
        "model_num_timesteps_after": u2r.SOURCE_LIFETIME_ACTIONS,
        "model_updates_before": u2r.SOURCE_OPTIMIZER_UPDATES,
        "model_updates_after": u2r.SOURCE_OPTIMIZER_UPDATES,
        "artifact_sha256s_before": {
            "checkpoint": u2r.SOURCE_ARCHIVE_SHA256,
        },
        "artifact_sha256s_after": {
            "checkpoint": u2r.SOURCE_ARCHIVE_SHA256,
        },
        "evaluations": [
            {
                "lesson_id": lessons.LessonId.SEPARATED_UNLOCK.value,
                "successes": 169,
                "panel_successes": [84, 85],
                "gate": {"passed": False},
            }
        ],
    }
    report = {
        "schema_version": 1,
        "protocol": "dungeon-apprentice-v0.2-u2-confirmation",
        "verdict": "capability_failed",
        "policy_updates": False,
        "checkpoint_scoring_performed": True,
        "source": {
            "commit": u2r.U2_CONFIRMATION_SOURCE_COMMIT,
            "dirty": False,
        },
        "source_after": {
            "commit": u2r.U2_CONFIRMATION_SOURCE_COMMIT,
            "dirty": False,
        },
        "checkpoints": [entry],
    }
    atomic_write_json(report_path, report)
    monkeypatch.setattr(u2r, "U2_CONFIRMATION_REPORT", report_path.resolve())
    monkeypatch.setattr(
        u2r,
        "U2_CONFIRMATION_REPORT_SHA256",
        file_sha256(report_path),
    )

    attempt_path = confirmation / "attempt.json"
    atomic_write_json(
        attempt_path,
        {
            "status": "completed_with_report",
            "report_present": True,
            "report_verdict": "capability_failed",
            "report_sha256": u2r.U2_CONFIRMATION_REPORT_SHA256,
            "source_commit": u2r.U2_CONFIRMATION_SOURCE_COMMIT,
            "source_dirty": False,
            "evaluator_exit_status": 1,
        },
    )
    monkeypatch.setattr(u2r, "U2_CONFIRMATION_ATTEMPT", attempt_path.resolve())
    monkeypatch.setattr(
        u2r,
        "U2_CONFIRMATION_ATTEMPT_SHA256",
        file_sha256(attempt_path),
    )
    parent = u2r.verify_u2r_parent(
        checkpoint,
        confirmation_report=report_path,
    )
    return (
        parent,
        {
            "checkpoint": checkpoint,
            "sidecar": sidecar_path,
            "integrity": integrity_path,
            "manifest": manifest_path,
            "evaluations": evaluations_path,
            "report": report_path,
            "attempt": attempt_path,
        },
        report,
    )


def _write_exclusion_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict,
    set[str],
    dict[lessons.LessonId, set[str]],
    dict[str, Path],
]:
    prior_u1_path = tmp_path / "u1-v2-confirmation.json"
    prior_u1_layouts: set[str] = set()
    prior_u1_lessons: dict[str, dict] = {}
    prior_u1_by_lesson: dict[lessons.LessonId, set[str]] = {}
    prior_u1_selections = []
    for lesson in (
        lessons.LessonId.NAVIGATE,
        lessons.LessonId.VISIBLE_UNLOCK,
        lessons.LessonId.LOCAL_UNLOCK,
    ):
        values = {_sha(f"u1-v2-{lesson.value}-{case}") for case in range(200)}
        prior_u1_layouts.update(values)
        prior_u1_by_lesson[lesson] = values
        prior_u1_lessons[lesson.value] = {
            "unique_layouts": 200,
            "set_sha256": u2r._hash_set_sha256(values),
        }
        prior_u1_selections.append(
            {
                "lesson_id": lesson.value,
                "result": "passed",
                "cases": [{"layout_sha256": value} for value in sorted(values)],
            }
        )
    atomic_write_json(
        prior_u1_path,
        {
            "protocol": "dungeon-apprentice-v0.2-u1-confirmation-v2",
            "verdict": "confirmed",
            "checkpoint_scoring_performed": True,
            "policy_updates": False,
            "layout_qualification": prior_u1_selections,
        },
    )
    monkeypatch.setattr(
        u2r.frozen_u2,
        "CANONICAL_U1_CONFIRMATION",
        prior_u1_path.resolve(),
    )
    monkeypatch.setattr(
        u2r.frozen_u2,
        "EXPECTED_U1_CONFIRMATION_SHA256",
        file_sha256(prior_u1_path),
    )

    histories: dict[str, dict] = {}
    history_layouts: set[str] = set()
    history_by_lesson: dict[lessons.LessonId, set[str]] = {
        lesson: set() for lesson in lessons.LessonId
    }
    history_paths: dict[str, Path] = {}
    for child in ("20260737", "20260741", "20260745"):
        path = tmp_path / f"episodes-{child}.jsonl"
        records = []
        lesson_evidence = {}
        for lesson in lessons.LessonId:
            layout = _sha(f"history-{child}-{lesson.value}")
            records.append(
                {
                    "lesson_id": lesson.value,
                    "layout_sha256": layout,
                }
            )
            lesson_evidence[lesson.value] = {
                "unique_layouts": 1,
                "set_sha256": u2r._hash_set_sha256({layout}),
            }
            history_layouts.add(layout)
            history_by_lesson[lesson].add(layout)
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        histories[child] = {
            "path": str(path.resolve()),
            "file_sha256": file_sha256(path),
            "lessons": lesson_evidence,
        }
        history_paths[child] = path

    confirmation = tmp_path / "confirmation"
    journal_root = confirmation / "selection-journal"
    journal_root.mkdir(parents=True)
    monkeypatch.setattr(
        u2r,
        "U2_CONFIRMATION_REPORT",
        confirmation / "report.json",
    )
    identities: list[dict] = []
    accepted_layouts: set[str] = set()
    inspectable_layouts: set[str] = set()
    inspectable_by_lesson: dict[lessons.LessonId, set[str]] = {
        lesson: set() for lesson in lessons.LessonId
    }
    index = 0
    for lesson in lessons.LessonId:
        for case in range(200):
            layout = _sha(f"accepted-{lesson.value}-{case}")
            relative = Path(f"{index:04d}-outcome.json")
            path = journal_root / relative
            atomic_write_json(
                path,
                {
                    "status": "accepted",
                    "lesson_id": lesson.value,
                    "layout_sha256": layout,
                },
            )
            identities.append(
                {
                    "path": str(relative),
                    "sha256": file_sha256(path),
                    "byte_length": path.stat().st_size,
                }
            )
            accepted_layouts.add(layout)
            inspectable_layouts.add(layout)
            inspectable_by_lesson[lesson].add(layout)
            index += 1
    for rejected in range(19):
        layout = _sha(f"rejected-{rejected}")
        relative = Path(f"{index:04d}-outcome.json")
        path = journal_root / relative
        atomic_write_json(
            path,
            {
                "status": "rejected",
                "lesson_id": lessons.LessonId.VISIBLE_UNLOCK.value,
                "layout_sha256": layout,
            },
        )
        identities.append(
            {
                "path": str(relative),
                "sha256": file_sha256(path),
                "byte_length": path.stat().st_size,
            }
        )
        inspectable_layouts.add(layout)
        inspectable_by_lesson[lessons.LessonId.VISIBLE_UNLOCK].add(layout)
        index += 1
    while len(identities) < 1_643:
        relative = Path(f"{len(identities):04d}-opened.json")
        path = journal_root / relative
        atomic_write_json(path, {"status": "opened"})
        identities.append(
            {
                "path": str(relative),
                "sha256": file_sha256(path),
                "byte_length": path.stat().st_size,
            }
        )

    base = {lesson: frozenset({_sha(f"base-{lesson.value}")}) for lesson in lessons.LessonId}
    monkeypatch.setattr(
        u2r,
        "_qualification_and_validation_hashes_by_lesson",
        lambda _report, *, access: (
            base,
            frozenset().union(*base.values()),
        ),
    )
    report = {
        "verdict": "capability_failed",
        "policy_updates": False,
        "reference_exclusions": {
            "u2_child_histories": histories,
            "prior_u1_confirmation": {
                "path": str(prior_u1_path.resolve()),
                "sha256": file_sha256(prior_u1_path),
                "lessons": prior_u1_lessons,
            },
        },
        "selection_journal": {
            "identities": identities,
            "files": len(identities),
            "file_set_sha256": u2r._canonical_sha256(identities),
        },
        "selected_layout_identity_recheck": {
            lesson.value: {"passed": True, "cases": 200} for lesson in lessons.LessonId
        },
    }
    expected = (
        set().union(*base.values()) | history_layouts | inspectable_layouts | prior_u1_layouts
    )
    expected_applied: dict[lessons.LessonId, set[str]] = {}
    for lesson in lessons.LessonId:
        values = set(base[lesson])
        if lesson is not lessons.LessonId.VISIBLE_UNLOCK:
            values.update(history_by_lesson[lesson])
            values.update(inspectable_by_lesson[lesson])
            values.update(prior_u1_by_lesson.get(lesson, set()))
        expected_applied[lesson] = values
    return report, expected, expected_applied, history_paths


def _qualification(tmp_path: Path) -> u2r.frozen_u2.QualificationProvenance:
    report = tmp_path / "qualification.json"
    report.write_text("sealed qualification", encoding="utf-8")
    return u2r.frozen_u2.QualificationProvenance(
        report=str(report),
        report_sha256=file_sha256(report),
        source_commit="c" * 40,
        evidence={"sealed": True},
        _seed_access=object(),
    )


def _write_parent_baseline_fixture(
    source_run: Path,
    *,
    source_commit: str,
) -> None:
    evaluations, cases, _expected_seeds = _exam_case_evidence()
    ineffective_totals = {
        lessons.LessonId.NAVIGATE: 2,
        lessons.LessonId.VISIBLE_UNLOCK: 524,
        lessons.LessonId.LOCAL_UNLOCK: 379,
        lessons.LessonId.SEPARATED_UNLOCK: 523,
    }
    offset = 0
    for lesson in lessons.LessonId:
        lesson_cases = cases[offset : offset + u2r.EVALUATION_SEED_COUNT]
        total = ineffective_totals[lesson]
        quotient, remainder = divmod(total, u2r.EVALUATION_SEED_COUNT)
        for index, record in enumerate(lesson_cases):
            record["ineffective_interactions"] = quotient + (index < remainder)
        evaluations[lesson] = lessons.aggregate_u2_case_evidence(
            lesson,
            lesson_cases,
            timestamp="2026-07-23T00:00:00+00:00",
        )
        offset += u2r.EVALUATION_SEED_COUNT
    checkpoint = source_run / "checkpoints" / "initial.zip"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"initial parent baseline")
    sidecar = checkpoint.with_suffix(".json")
    atomic_write_json(
        sidecar,
        {
            "kind": "initial",
            "resume_eligible": True,
            "model_state": {
                "policy_tensor_sha256": u2r.SOURCE_POLICY_TENSOR_SHA256,
                "optimizer_state_sha256": u2r.SOURCE_OPTIMIZER_STATE_SHA256,
            },
            "source": {"commit": source_commit, "dirty": False},
        },
    )
    u2r._write_integrity(checkpoint, sidecar)
    summary = u2r.verify_exam_case_evidence(evaluations, cases)
    cases_path = u2r._case_evidence_path(checkpoint)
    atomic_write_json(
        cases_path,
        {
            "schema_version": u2r.CASE_EVIDENCE_SCHEMA_VERSION,
            "protocol": u2r.PROTOCOL,
            "kind": "immutable_exam_cases",
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_sidecar_sha256": file_sha256(sidecar),
            "child_trained_actions": u2r.SOURCE_CHILD_ACTIONS,
            "summary": summary,
            "records": cases,
        },
    )
    binding = summary | {
        "path": str(cases_path.relative_to(source_run)),
        "file_sha256": file_sha256(cases_path),
        "checkpoint": str(checkpoint.relative_to(source_run)),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_sidecar_sha256": file_sha256(sidecar),
        "child_trained_actions": u2r.SOURCE_CHILD_ACTIONS,
    }
    lines = []
    for lesson in lessons.LessonId:
        lines.append(
            json.dumps(
                {
                    "trigger": "diagnostic_parent_baseline",
                    "counts_toward_gate": False,
                    "checkpoint": str(checkpoint.relative_to(source_run)),
                    "case_evidence": binding,
                    **evaluations[lesson].public_dict(),
                },
                sort_keys=True,
            )
        )
    (source_run / "evaluations.jsonl").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_interruption_fixture(
    source_run: Path,
    *,
    checkpoint: Path,
    sidecar: dict,
    active_workers: list[dict],
) -> None:
    timestamp = "2026-07-23T01:00:00+00:00"
    ledger = source_run / "episode-starts.jsonl"
    for worker in active_workers:
        append_jsonl(
            ledger,
            {
                "timestamp": worker["episode_started_at"],
                "type": "episode_start",
                **worker,
            },
        )
        append_jsonl(
            ledger,
            {
                "timestamp": timestamp,
                "type": "episode_abandoned_at_interruption",
                "interruption_phase": "interrupted",
                **worker,
            },
        )
    progress = sidecar["progress"]
    trained = int(progress["trained_actions"])
    child = int(progress["child_trained_actions"])
    updates = int(progress["optimizer_updates"])
    document = {
        "schema_version": u2r.CHECKPOINT_SCHEMA_VERSION,
        "protocol": u2r.PROTOCOL,
        "kind": "process_interruption_evidence",
        "created_at": timestamp,
        "phase": "interrupted",
        "resume_permitted": True,
        "resume_blocker": None,
        "source": sidecar["source"],
        "effective_config": sidecar["effective_config"],
        "parent": sidecar["parent"],
        "qualification": sidecar["qualification"],
        "exclusions": sidecar["exclusions"],
        "external_preregistration": sidecar["external_preregistration"],
        "storage_mount": sidecar["storage_mount"],
        "segment": sidecar["segment"],
        "at_signal": {
            "timestamp": timestamp,
            "collected_actions": trained,
            "trained_actions": trained,
            "child_trained_actions": child,
            "optimizer_updates": updates,
            "pending_optimizer_boundary": "none",
            "untrained_collected_actions": 0,
            "untrained_transitions_per_worker": 0,
        },
        "resolution": {
            "pending_optimizer_boundary": "none",
            "recovered_complete_actions": 0,
            "recovered_complete_optimizer_updates": 0,
            "discarded_post_safe_rollout_actions": 0,
            "unpublished_trained_actions": 0,
            "untrained_collected_actions": 0,
            "unpublished_complete_actions": 0,
            "recovery_error": None,
            "resolved_trained_actions": trained,
            "resolved_child_trained_actions": child,
            "resolved_optimizer_updates": updates,
            "safe_boundary_trained_actions": trained,
            "active_environment_state_disposition": ("discarded_at_process_boundary"),
            "policy_recurrent_state_disposition": ("discarded_at_process_boundary"),
        },
        "active_workers": active_workers,
        "active_workers_sha256": u2r._canonical_json_sha256(active_workers),
        "episode_start_ledger": {
            "path": ledger.name,
            "sha256": file_sha256(ledger),
            **u2r._jsonl_inventory(ledger),
        },
        "resume_boundary": {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "sidecar": str(checkpoint.with_suffix(".json").resolve()),
            "sidecar_sha256": file_sha256(checkpoint.with_suffix(".json")),
            "integrity": str(checkpoint.with_suffix(".integrity.json").resolve()),
            "integrity_sha256": file_sha256(checkpoint.with_suffix(".integrity.json")),
            "model_state": sidecar["model_state"],
            "progress": progress,
        },
    }
    interruption = source_run / "interruption.json"
    integrity = source_run / "interruption.integrity.json"
    atomic_write_json(interruption, document)
    atomic_write_json(
        integrity,
        {
            "schema_version": u2r.CHECKPOINT_SCHEMA_VERSION,
            "protocol": u2r.PROTOCOL,
            "interruption": interruption.name,
            "interruption_sha256": file_sha256(interruption),
        },
    )
    binding = {
        "path": interruption.name,
        "sha256": file_sha256(interruption),
        "integrity": integrity.name,
        "integrity_sha256": file_sha256(integrity),
        "phase": "interrupted",
        "active_workers_sha256": document["active_workers_sha256"],
    }
    atomic_write_json(
        source_run / "status.json",
        {
            "protocol": u2r.PROTOCOL,
            "phase": "interrupted",
            "interruption": binding,
        },
    )


def _write_resume_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mid_window_rollouts: int = 0,
) -> tuple[
    u2r.ResumeBundle,
    dict,
    Path,
    u2r.U2rParent,
    u2r.frozen_u2.QualificationProvenance,
]:
    parent, _paths, _report = _write_parent_fixture(
        tmp_path / "parent",
        monkeypatch,
    )
    qualification = _qualification(tmp_path)
    source_commit = "d" * 40
    remediation = u2r.EVALUATION_INTERVAL + mid_window_rollouts * u2r.ROLLOUT_TRANSITIONS
    child = u2r.SOURCE_CHILD_ACTIONS + remediation
    lifetime = u2r.SOURCE_LIFETIME_ACTIONS + remediation
    updates = u2r.SOURCE_OPTIMIZER_UPDATES + remediation // u2r.ROLLOUT_TRANSITIONS * u2r.PPO_EPOCHS
    decision_child = u2r.SOURCE_CHILD_ACTIONS + u2r.EVALUATION_INTERVAL
    resume_evaluations, resume_cases, resume_seeds = _exam_case_evidence()
    monkeypatch.setattr(
        lessons,
        "validation_seeds",
        lambda lesson, *, access: resume_seeds[lessons.LessonId(lesson)],
    )
    run_root = tmp_path / "runs"
    monkeypatch.setattr(u2r, "DEFAULT_RUN_ROOT", run_root)
    source_run = run_root / u2r._segment_run_name(0)
    rolling = source_run / "checkpoints" / "rolling"
    exam_checkpoint = _write_exam_bundle(rolling, decision_child)
    exam_sidecar = exam_checkpoint.with_suffix(".json")
    case_summary = u2r.verify_exam_case_evidence(
        resume_evaluations,
        resume_cases,
        expected_seeds=resume_seeds,
    )
    case_path = u2r._case_evidence_path(exam_checkpoint)
    atomic_write_json(
        case_path,
        {
            "schema_version": u2r.CASE_EVIDENCE_SCHEMA_VERSION,
            "protocol": u2r.PROTOCOL,
            "kind": "immutable_exam_cases",
            "checkpoint": exam_checkpoint.name,
            "checkpoint_sha256": file_sha256(exam_checkpoint),
            "checkpoint_sidecar_sha256": file_sha256(exam_sidecar),
            "child_trained_actions": decision_child,
            "summary": case_summary,
            "records": resume_cases,
        },
    )
    case_binding = case_summary | {
        "path": str(case_path.relative_to(source_run)),
        "file_sha256": file_sha256(case_path),
        "checkpoint": str(exam_checkpoint.relative_to(source_run)),
        "checkpoint_sha256": file_sha256(exam_checkpoint),
        "checkpoint_sidecar_sha256": file_sha256(exam_sidecar),
        "child_trained_actions": decision_child,
    }
    exam_record = {
        "child_trained_actions": decision_child,
        "remediation_trained_actions": u2r.EVALUATION_INTERVAL,
        "allocation_valid": True,
        "recovery": False,
        "evaluations": {
            lesson.value: result.public_dict() for lesson, result in resume_evaluations.items()
        },
        "case_evidence": case_binding,
    }
    controller = u2r.ControllerState(
        last_decision_child_actions=decision_child,
        exam_records=[exam_record],
    )
    state = lessons.CurriculumState(consecutive_passes=1, revision=1)
    scheduler = lessons.TransitionDeficitScheduler(state, seed=55)
    pending_transitions = mid_window_rollouts * u2r.ROLLOUT_TRANSITIONS
    lifetime_counts = {
        lessons.LessonId.NAVIGATE: child // 2,
        lessons.LessonId.VISIBLE_UNLOCK: child * 3 // 40,
        lessons.LessonId.LOCAL_UNLOCK: child * 3 // 40,
    }
    lifetime_counts[lessons.LessonId.SEPARATED_UNLOCK] = child - sum(lifetime_counts.values())
    pending_counts = {
        lessons.LessonId.NAVIGATE: pending_transitions // 2,
        lessons.LessonId.VISIBLE_UNLOCK: pending_transitions * 3 // 40,
        lessons.LessonId.LOCAL_UNLOCK: pending_transitions * 3 // 40,
    }
    pending_counts[lessons.LessonId.SEPARATED_UNLOCK] = pending_transitions - sum(
        pending_counts.values()
    )
    scheduler.window_transitions.update(pending_counts)
    scheduler.lifetime_transitions.update(lifetime_counts)
    active_workers = [
        {
            "worker_index": index,
            "worker_stream": u2r.REMEDIATION_WORKER_STREAMS[index],
            "episode_ordinal": index + 1,
            "seed": 100 + index,
            "lesson_id": lessons.LessonId.SEPARATED_UNLOCK.value,
            "layout_sha256": _sha(f"active-layout-{index}"),
            "geometry_sha256": _sha(f"active-geometry-{index}"),
            "elapsed_steps": index + 2,
            "active": True,
            "worker_transition_at_start": index * 10,
            "diagnostics": {},
            "seed_role": "training",
            "reset_provenance": "curriculum_training_rng",
            "generator_profile_version": lessons.GENERATOR_PROFILE_VERSION,
            "generator_profile": lessons.GENERATOR_PROFILE,
            "episode_started_at": "2026-07-23T00:00:00+00:00",
            "untrained_transitions": 0,
            "exact_environment_resume_supported": False,
            "environment_state_disposition": "discarded_on_process_resume",
            "environment_rng_state_disposition": (
                "discarded_on_process_resume; next segment uses a frozen fresh stream"
            ),
            "policy_recurrent_state_disposition": ("discarded_on_process_resume"),
            "checkpoint_source_commit": source_commit,
            "checkpoint_lifetime_trained_actions": lifetime,
            "checkpoint_child_trained_actions": child,
        }
        for index in range(u2r.WORKERS)
    ]
    expected_config = {
        "frozen": True,
        "exclusions": {"fixture": "bound"},
        "external_preregistration": {"fixture": "bound"},
    }
    checkpoint = source_run / "checkpoints" / "latest-safe.zip"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"policy and optimizer")
    sidecar = {
        "schema_version": u2r.CHECKPOINT_SCHEMA_VERSION,
        "protocol": u2r.PROTOCOL,
        "kind": "latest",
        "resume_eligible": True,
        "checkpoint_sha256": file_sha256(checkpoint),
        "effective_config": expected_config,
        "exclusions": expected_config["exclusions"],
        "external_preregistration": expected_config["external_preregistration"],
        "storage_mount": {"fixture": "bound"},
        "parent": parent.public_dict(),
        "qualification": qualification.public_dict(),
        "source": {"commit": source_commit, "dirty": False},
        "curriculum": state.public_dict(),
        "controller": controller.public_dict(),
        "scheduler": scheduler.state_dict(),
        "progress": {
            "collected_actions": lifetime,
            "trained_actions": lifetime,
            "lifetime_trained_actions": lifetime,
            "child_trained_actions": child,
            "remediation_trained_actions": remediation,
            "remaining_remediation_actions": (u2r.ADDITIONAL_ACTION_BUDGET - remediation),
            "optimizer_updates": updates,
        },
        "segment": {
            "index": 0,
            "lineage_source_child_seed": u2r.SOURCE_CHILD_SEED,
            "segment_algorithm_seed": u2r.REMEDIATION_ALGORITHM_SEED,
            "segment_worker_streams": list(u2r.REMEDIATION_WORKER_STREAMS),
            "resume_checkpoint": None,
            "resume_checkpoint_sha256": None,
            "resume_model_state": None,
            "resume_source_segment_index": None,
        },
        "last_completed_allocation": {
            "window_transitions": {
                lessons.LessonId.NAVIGATE.value: 16_384,
                lessons.LessonId.VISIBLE_UNLOCK.value: 2_458,
                lessons.LessonId.LOCAL_UNLOCK.value: 2_458,
                lessons.LessonId.SEPARATED_UNLOCK.value: 11_468,
            },
        },
        "last_completed_allocation_valid": True,
        "latest_exam_case_evidence": case_binding,
        "active_workers": active_workers,
        "resume_semantics": {
            "untrained_transitions": 0,
            "exact_environment_resume_supported": False,
            "environment_state_disposition": "discarded_on_process_resume",
            "environment_rng_state_disposition": (
                "discarded_on_process_resume; next segment uses a frozen fresh stream"
            ),
            "policy_recurrent_state_disposition": ("discarded_on_process_resume"),
        },
        "model_state": {
            "policy_tensor_sha256": _sha("resume-policy"),
            "optimizer_state_sha256": _sha("resume-optimizer"),
        },
    }
    atomic_write_json(checkpoint.with_suffix(".json"), sidecar)
    u2r._write_integrity(checkpoint, checkpoint.with_suffix(".json"))
    _write_parent_baseline_fixture(
        source_run,
        source_commit=source_commit,
    )
    _write_interruption_fixture(
        source_run,
        checkpoint=checkpoint,
        sidecar=sidecar,
        active_workers=active_workers,
    )
    loaded = u2r.load_resume(
        checkpoint,
        expected_config=expected_config,
        parent=parent,
        qualification=qualification,
        expected_source_commit=source_commit,
    )
    return loaded, sidecar, checkpoint, parent, qualification


def _write_exam_bundle(directory: Path, child_actions: int) -> Path:
    checkpoint = directory / f"exam-{child_actions:07d}.zip"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"exam at {child_actions}".encode())
    sidecar = checkpoint.with_suffix(".json")
    atomic_write_json(
        sidecar,
        {
            "protocol": u2r.PROTOCOL,
            "child_trained_actions": child_actions,
        },
    )
    u2r._write_integrity(checkpoint, sidecar)
    return checkpoint


def _write_exam_evidence(
    run_directory: Path,
    child_actions: int,
    evaluations: dict[lessons.LessonId, lessons.LessonEvaluation],
    cases: list[dict],
    expected_seeds: dict[lessons.LessonId, tuple[int, ...]],
) -> dict:
    checkpoint = _write_exam_bundle(
        run_directory / "checkpoints" / "rolling",
        child_actions,
    )
    sidecar = checkpoint.with_suffix(".json")
    summary = u2r.verify_exam_case_evidence(
        evaluations,
        cases,
        expected_seeds=expected_seeds,
    )
    case_path = u2r._case_evidence_path(checkpoint)
    atomic_write_json(
        case_path,
        {
            "schema_version": u2r.CASE_EVIDENCE_SCHEMA_VERSION,
            "protocol": u2r.PROTOCOL,
            "kind": "immutable_exam_cases",
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_sidecar_sha256": file_sha256(sidecar),
            "child_trained_actions": child_actions,
            "summary": summary,
            "records": cases,
        },
    )
    binding = summary | {
        "path": str(case_path.relative_to(run_directory)),
        "file_sha256": file_sha256(case_path),
        "checkpoint": str(checkpoint.relative_to(run_directory)),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_sidecar_sha256": file_sha256(sidecar),
        "child_trained_actions": child_actions,
    }
    return {
        "child_trained_actions": child_actions,
        "remediation_trained_actions": (child_actions - u2r.SOURCE_CHILD_ACTIONS),
        "allocation_valid": True,
        "recovery": False,
        "evaluations": {
            lesson.value: result.public_dict() for lesson, result in evaluations.items()
        },
        "case_evidence": binding,
    }


def _evaluation(
    lesson: lessons.LessonId,
    successes: int,
    *,
    panels: tuple[int, int] | None = None,
    ineffective: float = 2.0,
) -> lessons.LessonEvaluation:
    panel_successes = panels or (successes // 2, successes - successes // 2)
    return lessons.LessonEvaluation(
        protocol=u2r.PROTOCOL,
        timestamp="2026-07-23T00:00:00+00:00",
        lesson_id=lesson.value,
        lesson_label=lessons.LESSON_SPECS[lesson].label,
        episodes=80,
        successes=successes,
        success_rate=successes / 80,
        panel_successes=panel_successes,
        panel_success_rates=tuple(value / 40 for value in panel_successes),
        mean_steps=20.0,
        mean_ineffective_interactions=ineffective,
    )


def _stable_exam() -> dict[lessons.LessonId, lessons.LessonEvaluation]:
    return {
        lessons.LessonId.NAVIGATE: _evaluation(
            lessons.LessonId.NAVIGATE,
            72,
            ineffective=0.0,
        ),
        lessons.LessonId.VISIBLE_UNLOCK: _evaluation(
            lessons.LessonId.VISIBLE_UNLOCK,
            72,
        ),
        lessons.LessonId.LOCAL_UNLOCK: _evaluation(
            lessons.LessonId.LOCAL_UNLOCK,
            72,
        ),
        lessons.LessonId.SEPARATED_UNLOCK: _evaluation(
            lessons.LessonId.SEPARATED_UNLOCK,
            74,
        ),
    }


def _case_records(
    lesson: lessons.LessonId,
    *,
    panel_successes: tuple[int, int],
    seed_base: int,
    ineffective: int = 2,
) -> list[dict]:
    records: list[dict] = []
    for case_index in range(u2r.EVALUATION_SEED_COUNT):
        panel = 0 if case_index < 40 else 1
        panel_case_index = case_index if panel == 0 else case_index - 40
        success = panel_case_index < panel_successes[panel]
        steps = 8 + case_index % 3
        oracle_actions = 4 + case_index % 2
        reachable_cells = 20
        unique_cells = 8 + case_index % 4
        visibility_stratum = None
        if lesson is lessons.LessonId.SEPARATED_UNLOCK:
            visibility_stratum = (
                "key_hidden_door_hidden",
                "key_hidden_door_visible",
                "key_visible_door_hidden",
                "key_visible_door_visible",
            )[case_index % 4]
        records.append(
            {
                "lesson_id": lesson.value,
                "lesson_label": lessons.LESSON_SPECS[lesson].label,
                "case_index": case_index,
                "panel": panel,
                "panel_case_index": panel_case_index,
                "seed": seed_base + case_index,
                "layout_sha256": _sha(f"{lesson.value}-layout-{case_index}"),
                "geometry_sha256": _sha(f"{lesson.value}-geometry-{case_index}"),
                "success": success,
                "terminal_reason": "success" if success else "timeout",
                "steps": steps,
                "milestones": {
                    "key_picked_up": success,
                    "door_opened": success,
                    "success": success,
                },
                "collisions": case_index % 2,
                "ineffective_interactions": ineffective,
                "unique_cells": unique_cells,
                "reachable_cells": reachable_cells,
                "coverage": unique_cells / reachable_cells,
                "oracle_actions": oracle_actions,
                "path_actions_per_oracle_action": steps / oracle_actions,
                "action_histogram": {
                    "0": steps,
                    "1": 0,
                    "2": 0,
                    "3": 0,
                    "4": 0,
                    "5": 0,
                    "6": 0,
                },
                "longest_repeated_action_run": steps,
                "extrinsic_return": 1.0 if success else 0.0,
                "curiosity_return": 0.0,
                "initial_key_visible": None,
                "initial_door_visible": None,
                "visibility_stratum": visibility_stratum,
            }
        )
    return records


def _exam_case_evidence() -> tuple[
    dict[lessons.LessonId, lessons.LessonEvaluation],
    list[dict],
    dict[lessons.LessonId, tuple[int, ...]],
]:
    panel_scores = {
        lessons.LessonId.NAVIGATE: (39, 40),
        lessons.LessonId.VISIBLE_UNLOCK: (37, 39),
        lessons.LessonId.LOCAL_UNLOCK: (38, 39),
        lessons.LessonId.SEPARATED_UNLOCK: (36, 36),
    }
    evaluations: dict[lessons.LessonId, lessons.LessonEvaluation] = {}
    records: list[dict] = []
    expected_seeds: dict[lessons.LessonId, tuple[int, ...]] = {}
    for lesson in lessons.LessonId:
        seed_base = lessons.LESSON_SPECS[lesson].validation_seed_base
        cases = _case_records(
            lesson,
            panel_successes=panel_scores[lesson],
            seed_base=seed_base,
        )
        records.extend(cases)
        expected_seeds[lesson] = tuple(
            seed_base + index for index in range(u2r.EVALUATION_SEED_COUNT)
        )
        evaluations[lesson] = lessons.aggregate_u2_case_evidence(
            lesson,
            cases,
            timestamp="2026-07-23T00:00:00+00:00",
        )
    return evaluations, records, expected_seeds


def _write_terminal_report_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    monkeypatch.setattr(
        u2r,
        "_verify_loaded_model",
        lambda *_args, **_kwargs: None,
    )
    run_root = tmp_path / "terminal-runs"
    monkeypatch.setattr(u2r, "DEFAULT_RUN_ROOT", run_root)
    run_directory = run_root / u2r._segment_run_name(0)
    run_directory.mkdir(parents=True)
    source = {"commit": "d" * 40, "dirty": False}
    effective_config = {
        "frozen": True,
        "exclusions": {"fixture": "terminal"},
        "external_preregistration": {"fixture": "terminal"},
    }
    parent = {
        "source_baseline": {
            lesson: {
                "successes": int(value["successes"]),
                "panel_successes": list(value["panel_successes"]),
                "mean_ineffective_interactions": float(value["mean_ineffective_interactions"]),
            }
            for lesson, value in u2r.SOURCE_BASELINE.items()
        }
    }
    qualification = {"fixture": "terminal"}
    exclusions = effective_config["exclusions"]
    anchor = effective_config["external_preregistration"]
    storage_mount = {"fixture": "terminal"}
    segment = {
        "id": "terminal-fixture",
        "index": 0,
        "lineage_source_child_seed": u2r.SOURCE_CHILD_SEED,
        "initial_algorithm_seed": u2r.REMEDIATION_ALGORITHM_SEED,
        "initial_worker_streams": list(u2r.REMEDIATION_WORKER_STREAMS),
        "segment_algorithm_seed": u2r.REMEDIATION_ALGORITHM_SEED,
        "segment_worker_streams": list(u2r.REMEDIATION_WORKER_STREAMS),
        "start_lifetime_trained_actions": u2r.SOURCE_LIFETIME_ACTIONS,
        "start_child_trained_actions": u2r.SOURCE_CHILD_ACTIONS,
        "start_remediation_trained_actions": 0,
        "remaining_remediation_actions": u2r.ADDITIONAL_ACTION_BUDGET,
        "resume_checkpoint": None,
        "resume_checkpoint_sha256": None,
        "resume_model_state": None,
        "resume_source_segment_index": None,
        "inherited_active_workers": [],
        "inherited_active_workers_abandoned": False,
    }
    atomic_write_json(
        run_directory / "manifest.json",
        {
            "schema_version": u2r.CHECKPOINT_SCHEMA_VERSION,
            "protocol": u2r.PROTOCOL,
            "source": source,
            "effective_config": effective_config,
            "parent": parent,
            "qualification": qualification,
            "exclusions": exclusions,
            "external_preregistration": anchor,
            "storage_mount": storage_mount,
            "segment": segment,
        },
    )
    _write_parent_baseline_fixture(
        run_directory,
        source_commit=str(source["commit"]),
    )
    for filename, record in (
        ("episodes.jsonl", {"type": "episode"}),
        ("events.jsonl", {"type": "run_terminal"}),
        ("optimizer.jsonl", {"type": "optimizer"}),
    ):
        append_jsonl(run_directory / filename, record)
    active_workers = []
    for index, stream in enumerate(u2r.REMEDIATION_WORKER_STREAMS):
        worker = {
            "worker_index": index,
            "worker_stream": stream,
            "episode_ordinal": 1,
            "seed": 200 + index,
            "lesson_id": lessons.LessonId.SEPARATED_UNLOCK.value,
            "layout_sha256": _sha(f"terminal-worker-layout-{index}"),
            "geometry_sha256": _sha(f"terminal-worker-geometry-{index}"),
            "elapsed_steps": 5,
            "active": True,
            "worker_transition_at_start": index * 10,
            "diagnostics": {},
            "seed_role": "training",
            "reset_provenance": "curriculum_training_rng",
            "generator_profile_version": lessons.GENERATOR_PROFILE_VERSION,
            "generator_profile": lessons.GENERATOR_PROFILE,
            "episode_started_at": "2026-07-23T00:00:00+00:00",
            "untrained_transitions": 0,
            "exact_environment_resume_supported": False,
            "environment_state_disposition": "discarded_on_process_resume",
            "environment_rng_state_disposition": (
                "discarded_on_process_resume; next segment uses a frozen fresh stream"
            ),
            "policy_recurrent_state_disposition": ("discarded_on_process_resume"),
            "checkpoint_source_commit": source["commit"],
            "checkpoint_lifetime_trained_actions": (u2r.TERMINAL_LIFETIME_ACTIONS),
            "checkpoint_child_trained_actions": (u2r.TERMINAL_CHILD_ACTIONS),
        }
        active_workers.append(worker)
        append_jsonl(
            run_directory / "episode-starts.jsonl",
            {"type": "episode_start", **worker},
        )
    evaluations, cases, expected_seeds = _exam_case_evidence()
    records = []
    for index in range(1, u2r.KEEP_ALL_EXAMS + 1):
        boundary = u2r.SOURCE_CHILD_ACTIONS + index * u2r.EVALUATION_INTERVAL
        records.append(
            _write_exam_evidence(
                run_directory,
                boundary,
                evaluations,
                cases,
                expected_seeds,
            )
        )
    decision = u2r.grade_stability_pair(
        records[-2]["evaluations"],
        records[-1]["evaluations"],
        penultimate_allocation_valid=True,
        terminal_allocation_valid=True,
        penultimate_recovery=False,
        terminal_recovery=False,
        penultimate_child_actions=records[-2]["child_trained_actions"],
        terminal_child_actions=records[-1]["child_trained_actions"],
    )
    terminal_pair = {
        "penultimate_child_actions": records[-2]["child_trained_actions"],
        "terminal_child_actions": records[-1]["child_trained_actions"],
        **decision.public_dict(),
    }
    controller = u2r.ControllerState(
        last_decision_child_actions=u2r.TERMINAL_CHILD_ACTIONS,
        exam_records=records,
        terminal_eligible=decision.eligible,
        terminal_reasons=list(decision.reasons),
        terminal_pair=terminal_pair,
    )
    final_exam = (
        run_directory / "checkpoints" / "rolling" / f"exam-{u2r.TERMINAL_CHILD_ACTIONS:07d}.zip"
    )
    final_exam_sidecar = final_exam.with_suffix(".json")
    terminal = run_directory / "checkpoints" / "terminal.zip"
    terminal.write_bytes(final_exam.read_bytes())
    terminal_sidecar = terminal.with_suffix(".json")
    terminal_progress = {
        "collected_actions": u2r.TERMINAL_LIFETIME_ACTIONS,
        "trained_actions": u2r.TERMINAL_LIFETIME_ACTIONS,
        "lifetime_trained_actions": u2r.TERMINAL_LIFETIME_ACTIONS,
        "child_trained_actions": u2r.TERMINAL_CHILD_ACTIONS,
        "remediation_trained_actions": u2r.ADDITIONAL_ACTION_BUDGET,
        "remaining_remediation_actions": 0,
        "optimizer_updates": (
            u2r.SOURCE_OPTIMIZER_UPDATES
            + u2r.ADDITIONAL_ACTION_BUDGET // u2r.ROLLOUT_TRANSITIONS * u2r.PPO_EPOCHS
        ),
    }
    atomic_write_json(
        terminal_sidecar,
        {
            "schema_version": u2r.CHECKPOINT_SCHEMA_VERSION,
            "protocol": u2r.PROTOCOL,
            "kind": ("terminal_eligible" if decision.eligible else "terminal_failed"),
            "resume_eligible": False,
            "checkpoint_sha256": file_sha256(terminal),
            "source": source,
            "effective_config": effective_config,
            "parent": parent,
            "qualification": qualification,
            "exclusions": exclusions,
            "external_preregistration": anchor,
            "storage_mount": storage_mount,
            "segment": segment,
            "controller": controller.public_dict(),
            "progress": terminal_progress,
            "model_state": {
                "policy_tensor_sha256": _sha("terminal-policy"),
                "optimizer_state_sha256": _sha("terminal-optimizer"),
            },
            "active_workers": active_workers,
            "exam_source": {
                "checkpoint": str(final_exam.relative_to(run_directory)),
                "checkpoint_sha256": file_sha256(final_exam),
                "sidecar": str(final_exam_sidecar.relative_to(run_directory)),
                "sidecar_sha256": file_sha256(final_exam_sidecar),
            },
        },
    )
    u2r._write_integrity(terminal, terminal_sidecar)
    exam_inventory = []
    for record in records:
        boundary = record["child_trained_actions"]
        checkpoint = run_directory / "checkpoints" / "rolling" / f"exam-{boundary:07d}.zip"
        sidecar = checkpoint.with_suffix(".json")
        integrity = checkpoint.with_suffix(".integrity.json")
        case_path = u2r._case_evidence_path(checkpoint)
        binding = record["case_evidence"]
        exam_inventory.append(
            {
                "child_trained_actions": boundary,
                "checkpoint": str(checkpoint.relative_to(run_directory)),
                "checkpoint_sha256": file_sha256(checkpoint),
                "sidecar": str(sidecar.relative_to(run_directory)),
                "sidecar_sha256": file_sha256(sidecar),
                "integrity": str(integrity.relative_to(run_directory)),
                "integrity_sha256": file_sha256(integrity),
                "case_evidence": str(case_path.relative_to(run_directory)),
                "case_evidence_sha256": file_sha256(case_path),
                "case_set_sha256": binding["case_set_sha256"],
                "aggregate_sha256s": {
                    lesson: evidence["aggregate_sha256"]
                    for lesson, evidence in binding["per_lesson"].items()
                },
            }
        )
    terminal_integrity = terminal.with_suffix(".integrity.json")
    report = {
        "schema_version": u2r.CHECKPOINT_SCHEMA_VERSION,
        "protocol": u2r.PROTOCOL,
        "kind": "terminal_stability_eligibility_report",
        "verdict": "eligible" if decision.eligible else "failed",
        "eligible_for_fresh_confirmation": decision.eligible,
        "fresh_confirmation_opened": False,
        "policy_updates_during_reporting": False,
        "source": source,
        "external_preregistration": anchor,
        "effective_config": effective_config,
        "parent": parent,
        "qualification": qualification,
        "exclusions": exclusions,
        "storage_mount": storage_mount,
        "storage": {"fixture": "terminal"},
        "segment": segment,
        "progress": {
            "source_child_actions": u2r.SOURCE_CHILD_ACTIONS,
            "remediation_trained_actions": u2r.ADDITIONAL_ACTION_BUDGET,
            "child_trained_actions": u2r.TERMINAL_CHILD_ACTIONS,
            "lifetime_trained_actions": u2r.TERMINAL_LIFETIME_ACTIONS,
            "optimizer_updates": terminal_progress["optimizer_updates"],
        },
        "parent_baseline": parent["source_baseline"],
        "controller": controller.public_dict(),
        "terminal_pair": terminal_pair,
        "exams": exam_inventory,
        "terminal_artifact": {
            "checkpoint": str(terminal.relative_to(run_directory)),
            "checkpoint_sha256": file_sha256(terminal),
            "sidecar": str(terminal_sidecar.relative_to(run_directory)),
            "sidecar_sha256": file_sha256(terminal_sidecar),
            "integrity": str(terminal_integrity.relative_to(run_directory)),
            "integrity_sha256": file_sha256(terminal_integrity),
            "model_state": json.loads(terminal_sidecar.read_text(encoding="utf-8"))["model_state"],
        },
        "lineage_history": u2r._build_lineage_history(
            run_directory,
            terminal_sidecar=json.loads(terminal_sidecar.read_text(encoding="utf-8")),
        ),
        "evaluations": {
            "path": "evaluations.jsonl",
            "sha256": file_sha256(run_directory / "evaluations.jsonl"),
        },
    }
    report_path = run_directory / "report.json"
    atomic_write_json(report_path, report)
    atomic_write_json(
        run_directory / "report.integrity.json",
        {
            "schema_version": u2r.CHECKPOINT_SCHEMA_VERSION,
            "protocol": u2r.PROTOCOL,
            "report": report_path.name,
            "report_sha256": file_sha256(report_path),
        },
    )
    return run_directory


def _rehash_terminal_fixture(run_directory: Path) -> None:
    report = run_directory / "report.json"
    atomic_write_json(
        run_directory / "report.integrity.json",
        {
            "schema_version": u2r.CHECKPOINT_SCHEMA_VERSION,
            "protocol": u2r.PROTOCOL,
            "report": report.name,
            "report_sha256": file_sha256(report),
        },
    )


def _verify_terminal_fixture(run_directory: Path) -> dict:
    return u2r.verify_terminal_report(
        run_directory,
        model_loader=lambda _path: object(),
    )


def test_terminal_report_round_trip_remains_verifiable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = _write_terminal_report_fixture(tmp_path, monkeypatch)

    first = _verify_terminal_fixture(run_directory)
    second = _verify_terminal_fixture(run_directory)

    assert first == second
    assert first["eligible_for_fresh_confirmation"] is True
    assert len(first["exams"]) == u2r.KEEP_ALL_EXAMS


def test_terminal_report_authenticates_deserialized_model_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = _write_terminal_report_fixture(tmp_path, monkeypatch)
    sentinel = object()
    measured: dict[str, object] = {}

    def verify_model(model: object, _args: object, **kwargs: object) -> None:
        measured["model"] = model
        measured.update(kwargs)

    monkeypatch.setattr(u2r, "_verify_loaded_model", verify_model)
    u2r.verify_terminal_report(
        run_directory,
        model_loader=lambda _path: sentinel,
    )

    assert measured["model"] is sentinel
    assert measured["expected_actions"] == u2r.TERMINAL_LIFETIME_ACTIONS
    assert measured["expected_updates"] == 3_584
    assert measured["expected_policy_sha256"] == _sha("terminal-policy")
    assert measured["expected_optimizer_sha256"] == _sha("terminal-optimizer")


def test_terminal_report_rejects_non_boolean_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = _write_terminal_report_fixture(tmp_path, monkeypatch)
    report_path = run_directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["eligible_for_fresh_confirmation"] = 1
    atomic_write_json(report_path, report)
    _rehash_terminal_fixture(run_directory)

    with pytest.raises(u2r.U2rProtocolError, match="identity"):
        _verify_terminal_fixture(run_directory)


def test_terminal_report_recomputes_raw_stability_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = _write_terminal_report_fixture(tmp_path, monkeypatch)
    report_path = run_directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fabricated_pair = deepcopy(report["terminal_pair"])
    fabricated_pair["eligible"] = False
    fabricated_pair["reasons"] = ["fabricated_failure"]
    report["verdict"] = "failed"
    report["eligible_for_fresh_confirmation"] = False
    report["terminal_pair"] = fabricated_pair
    report["controller"]["terminal_eligible"] = False
    report["controller"]["terminal_reasons"] = ["fabricated_failure"]
    report["controller"]["terminal_pair"] = fabricated_pair
    terminal = run_directory / "checkpoints" / "terminal.zip"
    terminal_sidecar_path = terminal.with_suffix(".json")
    terminal_sidecar = json.loads(terminal_sidecar_path.read_text(encoding="utf-8"))
    terminal_sidecar["kind"] = "terminal_failed"
    terminal_sidecar["controller"] = report["controller"]
    atomic_write_json(terminal_sidecar_path, terminal_sidecar)
    u2r._write_integrity(terminal, terminal_sidecar_path)
    report["terminal_artifact"]["sidecar_sha256"] = file_sha256(terminal_sidecar_path)
    report["terminal_artifact"]["integrity_sha256"] = file_sha256(
        terminal.with_suffix(".integrity.json")
    )
    atomic_write_json(report_path, report)
    _rehash_terminal_fixture(run_directory)

    with pytest.raises(u2r.U2rProtocolError, match="recompute"):
        _verify_terminal_fixture(run_directory)


def test_terminal_report_rejects_terminal_exam_source_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = _write_terminal_report_fixture(tmp_path, monkeypatch)
    report_path = run_directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    terminal = run_directory / "checkpoints" / "terminal.zip"
    terminal_sidecar_path = terminal.with_suffix(".json")
    terminal_sidecar = json.loads(terminal_sidecar_path.read_text(encoding="utf-8"))
    terminal_sidecar["exam_source"]["checkpoint_sha256"] = _sha("fabricated-final-exam")
    atomic_write_json(terminal_sidecar_path, terminal_sidecar)
    u2r._write_integrity(terminal, terminal_sidecar_path)
    report["terminal_artifact"]["sidecar_sha256"] = file_sha256(terminal_sidecar_path)
    report["terminal_artifact"]["integrity_sha256"] = file_sha256(
        terminal.with_suffix(".integrity.json")
    )
    atomic_write_json(report_path, report)
    _rehash_terminal_fixture(run_directory)

    with pytest.raises(u2r.U2rProtocolError, match="artifact binding"):
        _verify_terminal_fixture(run_directory)


def test_terminal_report_rejects_self_consistent_wrong_exam_seed_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = _write_terminal_report_fixture(tmp_path, monkeypatch)
    report_path = run_directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    exam_item = report["exams"][0]
    cases_path = run_directory / exam_item["case_evidence"]
    case_document = json.loads(cases_path.read_text(encoding="utf-8"))
    case_document["records"][0]["seed"] = 9_999_999
    evaluations = report["controller"]["exam_records"][0]["evaluations"]
    summary = u2r.verify_exam_case_evidence(
        evaluations,
        case_document["records"],
    )
    case_document["summary"] = summary
    atomic_write_json(cases_path, case_document)
    old_binding = report["controller"]["exam_records"][0]["case_evidence"]
    binding = summary | {
        key: old_binding[key]
        for key in (
            "path",
            "checkpoint",
            "checkpoint_sha256",
            "checkpoint_sidecar_sha256",
            "child_trained_actions",
        )
    }
    binding["file_sha256"] = file_sha256(cases_path)
    report["controller"]["exam_records"][0]["case_evidence"] = binding
    exam_item["case_evidence_sha256"] = file_sha256(cases_path)
    exam_item["case_set_sha256"] = binding["case_set_sha256"]
    exam_item["aggregate_sha256s"] = {
        lesson: evidence["aggregate_sha256"] for lesson, evidence in binding["per_lesson"].items()
    }
    terminal = run_directory / "checkpoints" / "terminal.zip"
    terminal_sidecar_path = terminal.with_suffix(".json")
    terminal_sidecar = json.loads(terminal_sidecar_path.read_text(encoding="utf-8"))
    terminal_sidecar["controller"] = report["controller"]
    atomic_write_json(terminal_sidecar_path, terminal_sidecar)
    u2r._write_integrity(terminal, terminal_sidecar_path)
    report["terminal_artifact"]["sidecar_sha256"] = file_sha256(terminal_sidecar_path)
    report["terminal_artifact"]["integrity_sha256"] = file_sha256(
        terminal.with_suffix(".integrity.json")
    )
    atomic_write_json(report_path, report)
    _rehash_terminal_fixture(run_directory)

    with pytest.raises(u2r.U2rProtocolError, match="seed order"):
        _verify_terminal_fixture(run_directory)


def test_terminal_report_rejects_parent_baseline_case_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = _write_terminal_report_fixture(tmp_path, monkeypatch)
    baseline_cases = run_directory / "checkpoints" / "initial.cases.json"
    baseline_cases.write_bytes(baseline_cases.read_bytes() + b"\n")

    with pytest.raises(u2r.U2rProtocolError, match=r"case|baseline"):
        _verify_terminal_fixture(run_directory)


def test_terminal_report_rejects_bound_history_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = _write_terminal_report_fixture(tmp_path, monkeypatch)
    append_jsonl(
        run_directory / "events.jsonl",
        {"type": "late_unbound_event"},
    )

    with pytest.raises(u2r.U2rProtocolError, match="events ledger"):
        _verify_terminal_fixture(run_directory)


def test_terminal_report_rejects_active_worker_without_episode_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = _write_terminal_report_fixture(tmp_path, monkeypatch)
    report_path = run_directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    terminal = run_directory / "checkpoints" / "terminal.zip"
    terminal_sidecar_path = terminal.with_suffix(".json")
    terminal_sidecar = json.loads(terminal_sidecar_path.read_text(encoding="utf-8"))
    terminal_sidecar["active_workers"][0]["layout_sha256"] = _sha("fabricated-active-layout")
    atomic_write_json(terminal_sidecar_path, terminal_sidecar)
    u2r._write_integrity(terminal, terminal_sidecar_path)
    report["terminal_artifact"]["sidecar_sha256"] = file_sha256(terminal_sidecar_path)
    report["terminal_artifact"]["integrity_sha256"] = file_sha256(
        terminal.with_suffix(".integrity.json")
    )
    workers = terminal_sidecar["active_workers"]
    report["lineage_history"]["terminal_active_workers"] = {
        "count": u2r.WORKERS,
        "records": workers,
        "records_sha256": u2r._canonical_json_sha256(workers),
    }
    atomic_write_json(report_path, report)
    _rehash_terminal_fixture(run_directory)

    with pytest.raises(u2r.U2rProtocolError, match="episode start"):
        _verify_terminal_fixture(run_directory)


def _grade(
    penultimate: dict[lessons.LessonId, lessons.LessonEvaluation],
    terminal: dict[lessons.LessonId, lessons.LessonEvaluation],
    **overrides: object,
) -> u2r.StabilityDecision:
    arguments = {
        "penultimate_allocation_valid": True,
        "terminal_allocation_valid": True,
        "penultimate_recovery": False,
        "terminal_recovery": False,
        "penultimate_child_actions": (u2r.TERMINAL_CHILD_ACTIONS - u2r.EVALUATION_INTERVAL),
        "terminal_child_actions": u2r.TERMINAL_CHILD_ACTIONS,
    }
    arguments.update(overrides)
    return u2r.grade_stability_pair(penultimate, terminal, **arguments)


class _EvidenceEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (3, 56, 56), dtype=np.uint8)
    action_space = gym.spaces.Discrete(7)

    def __init__(self, *, valid_hashes: bool = True) -> None:
        super().__init__()
        self.valid_hashes = valid_hashes
        self.steps = 0

    def reset(self, **_kwargs):
        self.steps = 0
        return np.zeros((3, 56, 56), dtype=np.uint8), {
            "seed": 42,
            "lesson_id": lessons.LessonId.SEPARATED_UNLOCK.value,
            "lesson_label": "Separated Unlock",
            "layout_sha256": _sha("layout") if self.valid_hashes else None,
            "geometry_sha256": _sha("geometry"),
            "milestones": {},
        }

    def step(self, _action):
        self.steps += 1
        terminated = self.steps == 2
        return (
            np.zeros((3, 56, 56), dtype=np.uint8),
            1.0 if terminated else -0.001,
            terminated,
            False,
            {
                "elapsed_steps": self.steps,
                "success": terminated,
                "terminal_reason": "success" if terminated else None,
                "milestones": {"key_picked_up": self.steps >= 1},
                "ineffective_interactions": 1,
                "action_histogram": {"5": self.steps},
            },
        )


def test_u2r_is_bound_to_the_failed_parent_and_only_its_unspent_budget() -> None:
    assert u2r.PROTOCOL == "dungeon-apprentice-v0.2-u2r-stability-r1"
    assert u2r.SOURCE_CHILD_SEED == 20260745
    assert (
        Path(
            "/Volumes/T7 Developer/DungeonApprentice/u2-separated-20260723/"
            "v02-u2-seed-20260745/checkpoints/mastered-separated-unlock.zip"
        )
        == u2r.SOURCE_ARCHIVE
    )
    assert (
        u2r.SOURCE_ARCHIVE_SHA256
        == "56dc459fb94110f41a14b2304425f572fc235c8cfc07e1152306cbf77ac33dee"
    )
    assert u2r.SOURCE_CHILD_ACTIONS == 688_128
    assert u2r.ADDITIONAL_ACTION_BUDGET == 360_448
    assert u2r.TERMINAL_CHILD_ACTIONS == 1_048_576
    assert u2r.ADDITIONAL_ACTION_BUDGET == 11 * u2r.EVALUATION_INTERVAL
    assert u2r.SOURCE_CHILD_ACTIONS + u2r.ADDITIONAL_ACTION_BUDGET == u2r.TERMINAL_CHILD_ACTIONS


def test_parent_verifier_binds_archive_optimizer_counters_and_failed_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, _paths, _report = _write_parent_fixture(tmp_path, monkeypatch)

    assert parent.checkpoint_sha256 == u2r.SOURCE_ARCHIVE_SHA256
    assert parent.sidecar_sha256 == u2r.SOURCE_SIDECAR_SHA256
    assert parent.integrity_sha256 == u2r.SOURCE_INTEGRITY_SHA256
    assert parent.manifest_sha256 == u2r.SOURCE_MANIFEST_SHA256
    assert parent.evaluations_sha256 == u2r.SOURCE_EVALUATIONS_SHA256
    assert parent.source_baseline == u2r.SOURCE_BASELINE
    assert parent.child_trained_actions == 688_128
    assert parent.trained_timesteps == 1_474_560
    assert parent.n_updates == 2_880
    assert parent.confirmation_verdict == "capability_failed"
    assert parent.confirmation_u2_score == 169
    assert parent.confirmation_u2_panels == (84, 85)
    assert parent.scheduler_state == {"rng_state": {"bit_generator": "PCG64"}}


@pytest.mark.parametrize(
    "artifact",
    [
        "checkpoint",
        "sidecar",
        "integrity",
        "manifest",
        "evaluations",
        "report",
        "attempt",
    ],
)
def test_parent_verifier_rejects_any_changed_bound_artifact(
    artifact: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _parent, paths, _report = _write_parent_fixture(tmp_path, monkeypatch)
    with paths[artifact].open("ab") as handle:
        handle.write(b"\nchanged")

    with pytest.raises(u2r.U2rProtocolError):
        u2r.verify_u2r_parent(
            paths["checkpoint"],
            confirmation_report=paths["report"],
        )


def test_parent_verifier_rejects_rehashed_source_baseline_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _parent, paths, report = _write_parent_fixture(tmp_path, monkeypatch)
    records = [
        json.loads(line) for line in paths["evaluations"].read_text(encoding="utf-8").splitlines()
    ]
    records[0]["successes"] += 1
    records[0]["panel_successes"][0] += 1
    paths["evaluations"].write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    new_evaluations_sha = file_sha256(paths["evaluations"])
    monkeypatch.setattr(u2r, "SOURCE_EVALUATIONS_SHA256", new_evaluations_sha)

    rehashed_report = deepcopy(report)
    rehashed_report["checkpoints"][0]["checkpoint"]["artifact_sha256s"]["evaluations"] = (
        new_evaluations_sha
    )
    atomic_write_json(paths["report"], rehashed_report)
    new_report_sha = file_sha256(paths["report"])
    monkeypatch.setattr(u2r, "U2_CONFIRMATION_REPORT_SHA256", new_report_sha)

    attempt = json.loads(paths["attempt"].read_text(encoding="utf-8"))
    attempt["report_sha256"] = new_report_sha
    atomic_write_json(paths["attempt"], attempt)
    monkeypatch.setattr(
        u2r,
        "U2_CONFIRMATION_ATTEMPT_SHA256",
        file_sha256(paths["attempt"]),
    )

    with pytest.raises(
        u2r.U2rProtocolError,
        match="source U2 baseline changed",
    ):
        u2r.verify_u2r_parent(
            paths["checkpoint"],
            confirmation_report=paths["report"],
        )


def test_failed_confirmation_entry_requires_exact_no_update_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _parent, _paths, report = _write_parent_fixture(tmp_path, monkeypatch)
    mutations = (
        lambda entry: entry.update(policy_updates=True),
        lambda entry: entry.update(passed=True),
        lambda entry: entry["checkpoint"].update(child_trained_actions=688_127),
        lambda entry: entry["checkpoint"].update(checkpoint_sha256="0" * 64),
        lambda entry: entry.update(policy_tensor_sha256_after="0" * 64),
        lambda entry: entry.update(optimizer_state_sha256_after="0" * 64),
        lambda entry: entry.update(model_num_timesteps_after=1_474_561),
        lambda entry: entry.update(model_updates_after=2_881),
        lambda entry: entry.update(artifact_sha256s_after={"checkpoint": "0" * 64}),
        lambda entry: entry["evaluations"][0].update(successes=170),
        lambda entry: entry["evaluations"][0].update(panel_successes=[85, 85]),
        lambda entry: entry["evaluations"][0]["gate"].update(passed=True),
    )
    for mutate in mutations:
        tampered = deepcopy(report)
        mutate(tampered["checkpoints"][0])
        with pytest.raises(
            u2r.U2rProtocolError,
            match="no-update identity",
        ):
            u2r._matching_failed_confirmation_entry(tampered)


def test_training_exclusions_union_all_histories_and_inspectable_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, expected_inventory, expected_applied, history_paths = _write_exclusion_fixture(
        tmp_path,
        monkeypatch,
    )

    forbidden, evidence = u2r.build_u2r_forbidden_layout_hashes(
        {},
        report,
        access=object(),
    )

    assert evidence.qualification_and_validation_layouts == 4
    assert evidence.completed_history_layouts == 12
    assert evidence.accepted_confirmation_layouts == 800
    assert evidence.inspectable_confirmation_layouts == 819
    assert evidence.prior_u1_confirmation_layouts == 600
    assert evidence.selection_journal_files == 1_643
    assert evidence.exact_layouts == len(expected_inventory) == 1_435
    assert evidence.exact_layout_set_sha256 == u2r._hash_set_sha256(expected_inventory)
    assert {
        lesson: set(forbidden[lesson])
        for lesson in lessons.LessonId
    } == expected_applied
    assert set(forbidden[lessons.LessonId.VISIBLE_UNLOCK]) == {
        _sha(f"base-{lessons.LessonId.VISIBLE_UNLOCK.value}")
    }
    assert evidence.applied_by_lesson == {
        lesson.value: {
            "exact_layouts": len(expected_applied[lesson]),
            "exact_layout_set_sha256": u2r._hash_set_sha256(expected_applied[lesson]),
            "rule": u2r.U2R_APPLIED_EXCLUSION_RULES[lesson],
        }
        for lesson in lessons.LessonId
    }
    assert evidence.applied_mapping_sha256 == u2r._canonical_sha256(
        {
            lesson.value: sorted(expected_applied[lesson])
            for lesson in lessons.LessonId
        }
    )

    history_paths["20260745"].write_text("changed\n", encoding="utf-8")
    with pytest.raises(
        u2r.U2rProtocolError,
        match="completed history changed",
    ):
        u2r.build_u2r_forbidden_layout_hashes(
            {},
            report,
            access=object(),
        )


def test_sampler_preflight_is_deterministic_and_can_accept_after_128(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_stream = 812_337

    class FakeLessonEnv:
        def __init__(self, *, lesson):
            self.lesson = lessons.LessonId(lesson)

        def reset(self, *, seed, options=None):
            del options
            digest = _sha(f"{self.lesson.value}:{seed}")
            return np.zeros((1,), dtype=np.uint8), {"layout_sha256": digest}

        def close(self):
            return None

    monkeypatch.setattr(lessons, "U2LessonEnv", FakeLessonEnv)
    generator = np.random.default_rng(worker_stream)
    first_hashes: dict[lessons.LessonId, set[str]] = {
        lesson: set() for lesson in lessons.LessonId
    }
    first_seeds = [
        int(generator.integers(0, lessons.TRAINING_SEED_LIMIT))
        for _ in range(128)
    ]
    for lesson in lessons.LessonId:
        first_hashes[lesson] = {
            _sha(f"{lesson.value}:{seed}") for seed in first_seeds
        }

    records = u2r.preflight_u2r_training_layout_sampler(
        {
            lesson: frozenset(values)
            for lesson, values in first_hashes.items()
        },
        seed_access=object(),
        worker_streams=(worker_stream,),
        max_attempts=256,
    )

    assert len(records) == len(lessons.LessonId)
    assert all(record["accepted_attempt"] > 128 for record in records)
    assert records == u2r.preflight_u2r_training_layout_sampler(
        {
            lesson: frozenset(values)
            for lesson, values in first_hashes.items()
        },
        seed_access=object(),
        worker_streams=(worker_stream,),
        max_attempts=256,
    )


def test_sampler_preflight_fails_closed_for_an_impossible_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = _sha("always-blocked")

    class ImpossibleLessonEnv:
        def __init__(self, *, lesson):
            self.lesson = lessons.LessonId(lesson)

        def reset(self, *, seed, options=None):
            del seed, options
            return np.zeros((1,), dtype=np.uint8), {"layout_sha256": blocked}

        def close(self):
            return None

    monkeypatch.setattr(lessons, "U2LessonEnv", ImpossibleLessonEnv)
    with pytest.raises(
        u2r.U2rProtocolError,
        match="within 3 attempts",
    ):
        u2r.preflight_u2r_training_layout_sampler(
            {
                lesson: frozenset({blocked})
                for lesson in lessons.LessonId
            },
            seed_access=object(),
            worker_streams=(91,),
            max_attempts=3,
        )


def test_episode_evidence_is_durable_at_reset_and_checkpoint_ready(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "episode-starts.jsonl"
    wrapped = u2r.EpisodeEvidenceWrapper(
        _EvidenceEnv(),
        worker_index=2,
        worker_stream=20260751,
        ledger=ledger,
    )
    assert wrapped.evidence_state() == {
        "worker_index": 2,
        "worker_stream": 20260751,
        "episode_ordinal": 0,
        "active": False,
        "not_started": True,
        "untrained_transitions": 0,
        "exact_environment_resume_supported": False,
        "environment_state_disposition": "discarded_on_process_resume",
        "environment_rng_state_disposition": (
            "discarded_on_process_resume; next segment uses a frozen fresh stream"
        ),
        "policy_recurrent_state_disposition": "discarded_on_process_resume",
    }

    wrapped.reset()
    starts = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(starts) == 1
    assert starts[0]["type"] == "episode_start"
    assert starts[0]["worker_index"] == 2
    assert starts[0]["worker_stream"] == 20260751
    assert starts[0]["episode_ordinal"] == 1
    assert starts[0]["worker_transition_at_start"] == 0
    assert starts[0]["seed"] == 42
    assert starts[0]["lesson_id"] == lessons.LessonId.SEPARATED_UNLOCK.value
    assert starts[0]["layout_sha256"] == _sha("layout")
    assert starts[0]["geometry_sha256"] == _sha("geometry")
    assert starts[0]["active"] is True

    wrapped.step(5)
    active = wrapped.evidence_state()
    assert active["active"] is True
    assert active["elapsed_steps"] == 1
    assert active["diagnostics"]["ineffective_interactions"] == 1
    assert active["diagnostics"]["milestones"]["key_picked_up"] is True

    wrapped.step(5)
    terminal = wrapped.evidence_state()
    assert terminal["active"] is False
    assert terminal["elapsed_steps"] == 2
    assert terminal["diagnostics"]["success"] is True

    wrapped.reset()
    starts = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(starts) == 2
    assert starts[-1]["episode_ordinal"] == 2
    assert starts[-1]["worker_transition_at_start"] == 2


def test_episode_start_refuses_missing_layout_identity(tmp_path: Path) -> None:
    ledger = tmp_path / "episode-starts.jsonl"
    wrapped = u2r.EpisodeEvidenceWrapper(
        _EvidenceEnv(valid_hashes=False),
        worker_index=0,
        worker_stream=20260749,
        ledger=ledger,
    )

    with pytest.raises(u2r.U2rProtocolError, match="active worker layout"):
        wrapped.reset()
    assert not ledger.exists()
    assert wrapped.evidence_state()["not_started"] is True


def test_resume_preserves_optimizer_counter_remaining_cap_and_active_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded, _sidecar, _checkpoint, _parent, _qualification = _write_resume_fixture(
        tmp_path, monkeypatch
    )

    assert loaded.child_trained == 720_896
    assert loaded.remediation_trained == 32_768
    assert u2r.ADDITIONAL_ACTION_BUDGET - loaded.remediation_trained == 327_680
    assert loaded.lifetime_trained == 1_507_328
    assert loaded.n_updates == 2_944
    assert loaded.segment_index == 1
    assert len(loaded.controller.exam_records) == 1
    assert len(loaded.active_workers) == 4
    assert [item["worker_index"] for item in loaded.active_workers] == [
        0,
        1,
        2,
        3,
    ]
    assert all(item["active"] is True for item in loaded.active_workers)
    assert loaded.policy_tensor_sha256 == _sha("resume-policy")
    assert loaded.optimizer_state_sha256 == _sha("resume-optimizer")


def test_process_resume_durably_abandons_each_active_episode_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded, _sidecar, _checkpoint, _parent, _qualification = _write_resume_fixture(
        tmp_path, monkeypatch
    )
    run_directory = tmp_path / "resumed"
    run_directory.mkdir()
    ledger = run_directory / "episode-starts.jsonl"
    ledger.write_text(
        json.dumps({"type": "prior_episode_start"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    abandoned = u2r._record_abandoned_resume_workers(run_directory, loaded)
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]

    assert records[0] == {"type": "prior_episode_start"}
    assert len(abandoned) == u2r.WORKERS
    assert len(records[1:]) == u2r.WORKERS
    assert [record["worker_index"] for record in records[1:]] == [0, 1, 2, 3]
    for source, durable, returned in zip(
        loaded.active_workers,
        records[1:],
        abandoned,
        strict=True,
    ):
        for field in (
            "worker_index",
            "worker_stream",
            "episode_ordinal",
            "seed",
            "lesson_id",
            "layout_sha256",
            "geometry_sha256",
            "elapsed_steps",
        ):
            assert durable[field] == source[field]
            assert returned[field] == source[field]
        assert durable["type"] == "episode_abandoned_on_resume"
        assert durable["abandoned_on_resume"] is True
        assert durable["abandonment_reason"]
        assert durable["abandoned_at"]


def test_resumed_worker_abandonment_verifies_manifest_and_ledger_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded, _sidecar, checkpoint, _parent, _qualification = _write_resume_fixture(
        tmp_path, monkeypatch
    )
    run_directory = tmp_path / "resumed"
    run_directory.mkdir()
    abandoned = u2r._record_abandoned_resume_workers(run_directory, loaded)
    segment = {
        "inherited_active_workers": abandoned,
        "inherited_active_workers_abandoned": True,
    }
    source_interruption = checkpoint.parent.parent / "interruption.json"

    u2r._verify_resumed_worker_abandonment(
        segment=segment,
        ledger_path=run_directory / "episode-starts.jsonl",
        source_workers=loaded.active_workers,
        source_interruption=source_interruption,
        source_interruption_sha256=file_sha256(source_interruption),
        source_active_workers_sha256=loaded.interruption_evidence["active_workers_sha256"],
    )

    tampered = deepcopy(segment)
    tampered["inherited_active_workers"][0]["source_interruption_sha256"] = "0" * 64
    with pytest.raises(
        u2r.U2rProtocolError,
        match="worker 0 abandonment changed",
    ):
        u2r._verify_resumed_worker_abandonment(
            segment=tampered,
            ledger_path=run_directory / "episode-starts.jsonl",
            source_workers=loaded.active_workers,
            source_interruption=source_interruption,
            source_interruption_sha256=file_sha256(source_interruption),
            source_active_workers_sha256=loaded.interruption_evidence["active_workers_sha256"],
        )


def test_resumed_worker_abandonment_rejects_missing_ledger_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded, _sidecar, checkpoint, _parent, _qualification = _write_resume_fixture(
        tmp_path, monkeypatch
    )
    run_directory = tmp_path / "resumed"
    run_directory.mkdir()
    abandoned = u2r._record_abandoned_resume_workers(run_directory, loaded)
    ledger_path = run_directory / "episode-starts.jsonl"
    records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    ledger_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records[:-1]) + "\n",
        encoding="utf-8",
    )
    source_interruption = checkpoint.parent.parent / "interruption.json"

    with pytest.raises(
        u2r.U2rProtocolError,
        match="invalid abandonment-ledger count",
    ):
        u2r._verify_resumed_worker_abandonment(
            segment={
                "inherited_active_workers": abandoned,
                "inherited_active_workers_abandoned": True,
            },
            ledger_path=ledger_path,
            source_workers=loaded.active_workers,
            source_interruption=source_interruption,
            source_interruption_sha256=file_sha256(source_interruption),
            source_active_workers_sha256=loaded.interruption_evidence["active_workers_sha256"],
        )


def test_resume_carries_the_exact_contiguous_eleven_exam_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded, _sidecar, checkpoint, _parent, _qualification = _write_resume_fixture(
        tmp_path, monkeypatch
    )
    terminal_resume = replace(
        loaded,
        lifetime_trained=u2r.TERMINAL_LIFETIME_ACTIONS,
        child_trained=u2r.TERMINAL_CHILD_ACTIONS,
        remediation_trained=u2r.ADDITIONAL_ACTION_BUDGET,
        n_updates=(
            u2r.SOURCE_OPTIMIZER_UPDATES
            + u2r.ADDITIONAL_ACTION_BUDGET // u2r.ROLLOUT_TRANSITIONS * u2r.PPO_EPOCHS
        ),
    )
    expected_boundaries = [
        u2r.SOURCE_CHILD_ACTIONS + index * u2r.EVALUATION_INTERVAL
        for index in range(1, u2r.KEEP_ALL_EXAMS + 1)
    ]
    evaluations, cases, expected_seeds = _exam_case_evidence()
    source_run = checkpoint.parent.parent
    records = [deepcopy(loaded.controller.exam_records[0])]
    records.extend(
        _write_exam_evidence(
            source_run,
            boundary,
            evaluations,
            cases,
            expected_seeds,
        )
        for boundary in expected_boundaries[1:]
    )
    terminal_resume = replace(
        terminal_resume,
        controller=u2r.ControllerState(
            last_decision_child_actions=u2r.TERMINAL_CHILD_ACTIONS,
            exam_records=records,
        ),
    )

    run_directory = tmp_path / "continued"
    carried = u2r._carry_forward_exams(
        terminal_resume,
        run_directory=run_directory,
        storage_guard=lambda _requested: {},
    )

    assert [item["child_trained_actions"] for item in carried] == (expected_boundaries)
    assert len(carried) == u2r.KEEP_ALL_EXAMS == 11
    assert all((run_directory / item["path"]).is_file() for item in carried)
    assert all(u2r._case_evidence_path(run_directory / item["path"]).is_file() for item in carried)


def test_resume_rejects_an_off_boundary_exam_even_when_archive_count_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded, _sidecar, checkpoint, _parent, _qualification = _write_resume_fixture(
        tmp_path, monkeypatch
    )
    terminal_resume = replace(
        loaded,
        lifetime_trained=u2r.TERMINAL_LIFETIME_ACTIONS,
        child_trained=u2r.TERMINAL_CHILD_ACTIONS,
        remediation_trained=u2r.ADDITIONAL_ACTION_BUDGET,
    )
    expected_boundaries = [
        u2r.SOURCE_CHILD_ACTIONS + index * u2r.EVALUATION_INTERVAL
        for index in range(1, u2r.KEEP_ALL_EXAMS + 1)
    ]
    expected_boundaries[5] += u2r.ROLLOUT_TRANSITIONS
    evaluations, cases, expected_seeds = _exam_case_evidence()
    source_run = checkpoint.parent.parent
    records = [deepcopy(loaded.controller.exam_records[0])]
    records.extend(
        _write_exam_evidence(
            source_run,
            boundary,
            evaluations,
            cases,
            expected_seeds,
        )
        for boundary in expected_boundaries[1:]
    )
    terminal_resume = replace(
        terminal_resume,
        controller=u2r.ControllerState(
            last_decision_child_actions=u2r.TERMINAL_CHILD_ACTIONS,
            exam_records=records,
        ),
    )

    with pytest.raises(
        u2r.U2rProtocolError,
        match="exact contiguous completed prefix",
    ):
        u2r._carry_forward_exams(
            terminal_resume,
            run_directory=tmp_path / "continued",
            storage_guard=lambda _requested: {},
        )


@pytest.mark.parametrize("tamper", ["modify", "remove"])
def test_two_hop_resume_carries_and_reverifies_immutable_case_evidence(
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded, original_sidecar, _checkpoint, parent, qualification = _write_resume_fixture(
        tmp_path, monkeypatch
    )
    segment_run = u2r.DEFAULT_RUN_ROOT.resolve() / u2r._segment_run_name(loaded.segment_index)
    carried = u2r._carry_forward_exams(
        loaded,
        run_directory=segment_run,
        storage_guard=lambda _requested: {},
    )
    assert len(carried) == 1
    carried_exam = segment_run / carried[0]["path"]
    carried_cases = u2r._case_evidence_path(carried_exam)
    assert file_sha256(carried_cases) == carried[0]["case_evidence_sha256"]

    next_checkpoint = segment_run / "checkpoints" / "latest-safe.zip"
    next_checkpoint.write_bytes(b"policy and optimizer")
    next_sidecar = deepcopy(original_sidecar)
    next_sidecar["checkpoint_sha256"] = file_sha256(next_checkpoint)
    next_sidecar["segment"]["index"] = loaded.segment_index
    next_sidecar["segment"]["segment_algorithm_seed"] = (
        u2r.REMEDIATION_ALGORITHM_SEED + loaded.segment_index * u2r.SEGMENT_SEED_OFFSET
    )
    next_streams = [
        stream + loaded.segment_index * u2r.SEGMENT_SEED_OFFSET
        for stream in u2r.REMEDIATION_WORKER_STREAMS
    ]
    next_sidecar["segment"]["segment_worker_streams"] = next_streams
    next_sidecar["segment"]["resume_checkpoint"] = str(loaded.checkpoint)
    next_sidecar["segment"]["resume_checkpoint_sha256"] = file_sha256(loaded.checkpoint)
    next_sidecar["segment"]["resume_model_state"] = {
        "policy_tensor_sha256": loaded.policy_tensor_sha256,
        "optimizer_state_sha256": loaded.optimizer_state_sha256,
    }
    next_sidecar["segment"]["resume_source_segment_index"] = loaded.segment_index - 1
    for worker, stream in zip(
        next_sidecar["active_workers"],
        next_streams,
        strict=True,
    ):
        worker["worker_stream"] = stream
    atomic_write_json(next_checkpoint.with_suffix(".json"), next_sidecar)
    u2r._write_integrity(
        next_checkpoint,
        next_checkpoint.with_suffix(".json"),
    )
    _write_interruption_fixture(
        segment_run,
        checkpoint=next_checkpoint,
        sidecar=next_sidecar,
        active_workers=next_sidecar["active_workers"],
    )

    loaded_again = u2r.load_resume(
        next_checkpoint,
        expected_config=next_sidecar["effective_config"],
        parent=parent,
        qualification=qualification,
        expected_source_commit="d" * 40,
    )
    assert loaded_again.segment_index == loaded.segment_index + 1
    assert loaded_again.controller.exam_records == loaded.controller.exam_records

    if tamper == "modify":
        carried_cases.write_bytes(carried_cases.read_bytes() + b"\n")
    else:
        carried_cases.unlink()
    with pytest.raises(
        u2r.U2rProtocolError,
        match=r"case-evidence|case evidence",
    ):
        u2r.load_resume(
            next_checkpoint,
            expected_config=next_sidecar["effective_config"],
            parent=parent,
            qualification=qualification,
            expected_source_commit="d" * 40,
        )


def test_mid_window_latest_safe_resume_binds_pending_scheduler_transitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded, sidecar, _checkpoint, _parent, _qualification = _write_resume_fixture(
        tmp_path,
        monkeypatch,
        mid_window_rollouts=1,
    )

    assert loaded.controller.last_decision_child_actions == 720_896
    assert loaded.child_trained == 722_944
    assert loaded.remediation_trained == 34_816
    assert loaded.n_updates == 2_948
    assert sum(sidecar["scheduler"]["window_transitions"].values()) == u2r.ROLLOUT_TRANSITIONS


@pytest.mark.parametrize(
    "tamper",
    ["scheduler_delta", "controller_delta", "action_delta"],
)
def test_mid_window_resume_rejects_controller_scheduler_action_disagreement(
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _loaded, sidecar, checkpoint, parent, qualification = _write_resume_fixture(
        tmp_path,
        monkeypatch,
        mid_window_rollouts=1,
    )
    if tamper == "scheduler_delta":
        sidecar["scheduler"]["window_transitions"][lessons.LessonId.NAVIGATE.value] += 1
    elif tamper == "controller_delta":
        second = deepcopy(sidecar["controller"]["exam_records"][0])
        second["child_trained_actions"] += u2r.EVALUATION_INTERVAL
        second["remediation_trained_actions"] += u2r.EVALUATION_INTERVAL
        sidecar["controller"]["exam_records"].append(second)
        sidecar["controller"]["last_decision_child_actions"] = second["child_trained_actions"]
    else:
        progress = sidecar["progress"]
        progress["collected_actions"] += u2r.ROLLOUT_TRANSITIONS
        progress["trained_actions"] += u2r.ROLLOUT_TRANSITIONS
        progress["lifetime_trained_actions"] += u2r.ROLLOUT_TRANSITIONS
        progress["child_trained_actions"] += u2r.ROLLOUT_TRANSITIONS
        progress["remediation_trained_actions"] += u2r.ROLLOUT_TRANSITIONS
        progress["remaining_remediation_actions"] -= u2r.ROLLOUT_TRANSITIONS
        progress["optimizer_updates"] += u2r.PPO_EPOCHS
    atomic_write_json(checkpoint.with_suffix(".json"), sidecar)
    u2r._write_integrity(checkpoint, checkpoint.with_suffix(".json"))

    with pytest.raises(u2r.U2rProtocolError):
        u2r.load_resume(
            checkpoint,
            expected_config=sidecar["effective_config"],
            parent=parent,
            qualification=qualification,
            expected_source_commit="d" * 40,
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "remaining",
        "optimizer",
        "child",
        "worker",
        "source",
        "policy_digest",
        "optimizer_digest",
    ],
)
def test_resume_rejects_reset_caps_counters_or_worker_evidence(
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _loaded, sidecar, checkpoint, parent, qualification = _write_resume_fixture(
        tmp_path, monkeypatch
    )
    if tamper == "remaining":
        sidecar["progress"]["remaining_remediation_actions"] = u2r.ADDITIONAL_ACTION_BUDGET
    elif tamper == "optimizer":
        sidecar["progress"]["optimizer_updates"] += 1
    elif tamper == "child":
        sidecar["progress"]["child_trained_actions"] = sidecar["progress"][
            "remediation_trained_actions"
        ]
    elif tamper == "worker":
        sidecar["active_workers"].pop()
    elif tamper == "policy_digest":
        sidecar["model_state"]["policy_tensor_sha256"] = "not-a-digest"
    elif tamper == "optimizer_digest":
        del sidecar["model_state"]["optimizer_state_sha256"]
    else:
        sidecar["source"]["dirty"] = True
    atomic_write_json(checkpoint.with_suffix(".json"), sidecar)
    u2r._write_integrity(checkpoint, checkpoint.with_suffix(".json"))

    with pytest.raises(u2r.U2rProtocolError):
        u2r.load_resume(
            checkpoint,
            expected_config=sidecar["effective_config"],
            parent=parent,
            qualification=qualification,
            expected_source_commit="d" * 40,
        )


def test_resume_rejects_rehashed_interruption_safe_bundle_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _loaded, sidecar, checkpoint, parent, qualification = _write_resume_fixture(
        tmp_path, monkeypatch
    )
    source_run = checkpoint.parent.parent
    interruption_path = source_run / "interruption.json"
    interruption = json.loads(interruption_path.read_text(encoding="utf-8"))
    interruption["resume_boundary"]["checkpoint_sha256"] = _sha("fabricated-safe-checkpoint")
    atomic_write_json(interruption_path, interruption)
    interruption_integrity = source_run / "interruption.integrity.json"
    atomic_write_json(
        interruption_integrity,
        {
            "schema_version": u2r.CHECKPOINT_SCHEMA_VERSION,
            "protocol": u2r.PROTOCOL,
            "interruption": interruption_path.name,
            "interruption_sha256": file_sha256(interruption_path),
        },
    )
    status_path = source_run / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["interruption"]["sha256"] = file_sha256(interruption_path)
    status["interruption"]["integrity_sha256"] = file_sha256(interruption_integrity)
    atomic_write_json(status_path, status)

    with pytest.raises(u2r.U2rProtocolError, match="resume boundary"):
        u2r.load_resume(
            checkpoint,
            expected_config=sidecar["effective_config"],
            parent=parent,
            qualification=qualification,
            expected_source_commit="d" * 40,
        )


def test_resume_model_must_match_the_digests_bound_in_its_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded, _sidecar, _checkpoint, _parent, _qualification = _write_resume_fixture(
        tmp_path, monkeypatch
    )
    args = u2r.build_parser().parse_args([])
    model = SimpleNamespace(
        num_timesteps=loaded.lifetime_trained,
        _n_updates=loaded.n_updates,
        policy=SimpleNamespace(
            optimizer=SimpleNamespace(state={"restored": True}),
            lstm_actor=SimpleNamespace(hidden_size=256, num_layers=1),
        ),
        learning_rate=2.5e-4,
        n_steps=u2r.ROLLOUT_STEPS,
        batch_size=256,
        n_epochs=u2r.PPO_EPOCHS,
        gamma=0.995,
        gae_lambda=0.98,
        ent_coef=0.01,
        action_space=SimpleNamespace(n=7),
        observation_space=SimpleNamespace(shape=(3, 56, 56)),
    )
    monkeypatch.setattr(
        u2r.state_digests,
        "policy_tensor_sha256",
        lambda _model: loaded.policy_tensor_sha256,
    )
    monkeypatch.setattr(
        u2r.state_digests,
        "optimizer_state_sha256",
        lambda _model: loaded.optimizer_state_sha256,
    )
    u2r._verify_loaded_model(
        model,
        args,
        expected_actions=loaded.lifetime_trained,
        expected_updates=loaded.n_updates,
        expected_policy_sha256=loaded.policy_tensor_sha256,
        expected_optimizer_sha256=loaded.optimizer_state_sha256,
    )

    monkeypatch.setattr(
        u2r.state_digests,
        "optimizer_state_sha256",
        lambda _model: "f" * 64,
    )
    with pytest.raises(u2r.U2rProtocolError, match="optimizer state differs"):
        u2r._verify_loaded_model(
            model,
            args,
            expected_actions=loaded.lifetime_trained,
            expected_updates=loaded.n_updates,
            expected_policy_sha256=loaded.policy_tensor_sha256,
            expected_optimizer_sha256=loaded.optimizer_state_sha256,
        )


def test_loaded_model_requires_exact_parent_counters_and_optimizer_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = u2r.build_parser().parse_args([])
    policy = SimpleNamespace(
        optimizer=SimpleNamespace(state={"restored": True}),
        lstm_actor=SimpleNamespace(hidden_size=256, num_layers=1),
    )
    model = SimpleNamespace(
        num_timesteps=u2r.SOURCE_LIFETIME_ACTIONS,
        _n_updates=u2r.SOURCE_OPTIMIZER_UPDATES,
        policy=policy,
        learning_rate=2.5e-4,
        n_steps=u2r.ROLLOUT_STEPS,
        batch_size=256,
        n_epochs=u2r.PPO_EPOCHS,
        gamma=0.995,
        gae_lambda=0.98,
        ent_coef=0.01,
        action_space=SimpleNamespace(n=7),
        observation_space=SimpleNamespace(shape=(3, 56, 56)),
    )
    monkeypatch.setattr(
        u2r.state_digests,
        "policy_tensor_sha256",
        lambda _model: u2r.SOURCE_POLICY_TENSOR_SHA256,
    )
    monkeypatch.setattr(
        u2r.state_digests,
        "optimizer_state_sha256",
        lambda _model: u2r.SOURCE_OPTIMIZER_STATE_SHA256,
    )

    u2r._verify_loaded_model(
        model,
        args,
        expected_actions=u2r.SOURCE_LIFETIME_ACTIONS,
        expected_updates=u2r.SOURCE_OPTIMIZER_UPDATES,
        expected_policy_sha256=u2r.SOURCE_POLICY_TENSOR_SHA256,
        expected_optimizer_sha256=u2r.SOURCE_OPTIMIZER_STATE_SHA256,
    )
    model.num_timesteps += 1
    with pytest.raises(u2r.U2rProtocolError, match="action mismatch"):
        u2r._verify_loaded_model(
            model,
            args,
            expected_actions=u2r.SOURCE_LIFETIME_ACTIONS,
            expected_updates=u2r.SOURCE_OPTIMIZER_UPDATES,
            expected_policy_sha256=u2r.SOURCE_POLICY_TENSOR_SHA256,
            expected_optimizer_sha256=u2r.SOURCE_OPTIMIZER_STATE_SHA256,
        )
    model.num_timesteps -= 1
    model.policy.optimizer.state = {}
    with pytest.raises(u2r.U2rProtocolError, match="optimizer state"):
        u2r._verify_loaded_model(
            model,
            args,
            expected_actions=u2r.SOURCE_LIFETIME_ACTIONS,
            expected_updates=u2r.SOURCE_OPTIMIZER_UPDATES,
            expected_policy_sha256=u2r.SOURCE_POLICY_TENSOR_SHA256,
            expected_optimizer_sha256=u2r.SOURCE_OPTIMIZER_STATE_SHA256,
        )
    model.policy.optimizer.state = {"restored": True}
    monkeypatch.setattr(
        u2r.state_digests,
        "policy_tensor_sha256",
        lambda _model: "0" * 64,
    )
    with pytest.raises(u2r.U2rProtocolError, match="policy tensors"):
        u2r._verify_loaded_model(
            model,
            args,
            expected_actions=u2r.SOURCE_LIFETIME_ACTIONS,
            expected_updates=u2r.SOURCE_OPTIMIZER_UPDATES,
            expected_policy_sha256=u2r.SOURCE_POLICY_TENSOR_SHA256,
            expected_optimizer_sha256=u2r.SOURCE_OPTIMIZER_STATE_SHA256,
        )
    monkeypatch.setattr(
        u2r.state_digests,
        "policy_tensor_sha256",
        lambda _model: u2r.SOURCE_POLICY_TENSOR_SHA256,
    )
    monkeypatch.setattr(
        u2r.state_digests,
        "optimizer_state_sha256",
        lambda _model: "0" * 64,
    )
    with pytest.raises(u2r.U2rProtocolError, match="optimizer state differs"):
        u2r._verify_loaded_model(
            model,
            args,
            expected_actions=u2r.SOURCE_LIFETIME_ACTIONS,
            expected_updates=u2r.SOURCE_OPTIMIZER_UPDATES,
            expected_policy_sha256=u2r.SOURCE_POLICY_TENSOR_SHA256,
            expected_optimizer_sha256=u2r.SOURCE_OPTIMIZER_STATE_SHA256,
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--parent", "/tmp/not-the-parent.zip"], "parent"),
        (["--additional-budget", "327680"], "configuration changed"),
        (["--workers", "2"], "configuration changed"),
        (["--evaluation-every", "65536"], "configuration changed"),
    ],
)
def test_u2r_rejects_parent_budget_and_training_overrides(
    arguments: list[str],
    message: str,
) -> None:
    args = u2r.build_parser().parse_args(arguments)
    with pytest.raises(SystemExit, match=message):
        u2r._validate_args(args)


def test_only_the_exact_terminal_pair_can_be_stability_eligible() -> None:
    penultimate = _stable_exam()
    terminal = _stable_exam()

    decision = _grade(penultimate, terminal)

    assert decision.eligible is True
    assert not decision.reasons
    assert all(decision.checks.values())

    early = _grade(
        penultimate,
        terminal,
        penultimate_child_actions=(u2r.TERMINAL_CHILD_ACTIONS - 2 * u2r.EVALUATION_INTERVAL),
        terminal_child_actions=(u2r.TERMINAL_CHILD_ACTIONS - u2r.EVALUATION_INTERVAL),
    )
    nonadjacent = _grade(
        penultimate,
        terminal,
        penultimate_child_actions=(u2r.TERMINAL_CHILD_ACTIONS - 2 * u2r.EVALUATION_INTERVAL),
    )
    assert early.eligible is False
    assert nonadjacent.eligible is False
    assert early.reasons
    assert nonadjacent.reasons


@pytest.mark.parametrize(
    ("lesson", "panels"),
    [
        (lessons.LessonId.NAVIGATE, (39, 40)),
        (lessons.LessonId.VISIBLE_UNLOCK, (37, 39)),
        (lessons.LessonId.LOCAL_UNLOCK, (38, 39)),
        (lessons.LessonId.SEPARATED_UNLOCK, (36, 36)),
    ],
)
def test_case_aggregator_uses_exactly_eighty_cases_and_two_fixed_panels(
    lesson: lessons.LessonId,
    panels: tuple[int, int],
) -> None:
    cases = _case_records(
        lesson,
        panel_successes=panels,
        seed_base=12_000_000 + list(lessons.LessonId).index(lesson) * 100,
    )

    aggregate = lessons.aggregate_u2_case_evidence(
        lesson,
        cases,
        timestamp="2026-07-23T00:00:00+00:00",
    )

    assert len(cases) == aggregate.episodes == u2r.EVALUATION_SEED_COUNT
    assert aggregate.successes == sum(panels)
    assert aggregate.panel_successes == panels
    assert aggregate.panel_success_rates == (
        panels[0] / 40,
        panels[1] / 40,
    )
    assert aggregate.mean_ineffective_interactions == 2.0
    assert [case["panel"] for case in cases[:40]] == [0] * 40
    assert [case["panel"] for case in cases[40:]] == [1] * 40


def test_per_case_sink_is_observational_only_for_lesson_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_timestamp = "2026-07-23T00:00:00+00:00"
    monkeypatch.setattr(lessons, "utc_now", lambda: fixed_timestamp)
    model = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(3, 56, 56)),
        predict=lambda *_args, **_kwargs: (np.asarray(0), None),
    )
    seeds = (11, 12)

    without_sink = lessons.evaluate_u2_lesson(
        model,
        lessons.LessonId.NAVIGATE,
        seeds,
    )
    captured: list[dict] = []
    with_sink = lessons.evaluate_u2_lesson(
        model,
        lessons.LessonId.NAVIGATE,
        seeds,
        case_evidence_sink=lambda record: captured.append(dict(record)),
    )

    assert with_sink.public_dict() == without_sink.public_dict()
    assert len(captured) == len(seeds)
    assert (
        lessons.aggregate_u2_case_evidence(
            lessons.LessonId.NAVIGATE,
            captured,
            timestamp=fixed_timestamp,
        ).public_dict()
        == with_sink.public_dict()
    )


@pytest.mark.parametrize(
    "tamper",
    ["case_index", "panel", "seed", "histogram", "coverage"],
)
def test_case_aggregator_rejects_malformed_atomic_evidence(tamper: str) -> None:
    cases = _case_records(
        lessons.LessonId.SEPARATED_UNLOCK,
        panel_successes=(36, 36),
        seed_base=12_000_000,
    )
    if tamper == "case_index":
        cases[0]["case_index"] = 1
    elif tamper == "panel":
        cases[0]["panel"] = 1
    elif tamper == "seed":
        cases[1]["seed"] = cases[0]["seed"]
    elif tamper == "histogram":
        cases[0]["action_histogram"]["0"] += 1
    else:
        cases[0]["coverage"] += 0.01

    with pytest.raises(ValueError, match="case evidence"):
        lessons.aggregate_u2_case_evidence(
            lessons.LessonId.SEPARATED_UNLOCK,
            cases,
        )


def test_exam_case_evidence_binds_all_320_cases_aggregates_and_seed_panels() -> None:
    evaluations, cases, expected_seeds = _exam_case_evidence()

    evidence = u2r.verify_exam_case_evidence(
        evaluations,
        list(reversed(cases)),
        expected_seeds=expected_seeds,
    )

    assert evidence["schema_version"] == u2r.CASE_EVIDENCE_SCHEMA_VERSION
    assert evidence["protocol"] == u2r.PROTOCOL
    assert evidence["case_records"] == 320
    assert len(evidence["case_records_sha256"]) == 64
    assert len(evidence["case_set_sha256"]) == 64
    assert set(evidence["per_lesson"]) == {lesson.value for lesson in lessons.LessonId}
    for lesson in lessons.LessonId:
        per_lesson = evidence["per_lesson"][lesson.value]
        assert per_lesson["cases"] == 80
        assert per_lesson["panel_cases"] == [40, 40]
        assert len(per_lesson["seed_sequence_sha256"]) == 64
        assert len(per_lesson["case_records_sha256"]) == 64
        assert len(per_lesson["case_set_sha256"]) == 64
        assert len(per_lesson["aggregate_sha256"]) == 64

    repeated = u2r.verify_exam_case_evidence(
        evaluations,
        cases,
        expected_seeds=expected_seeds,
    )
    assert repeated == evidence


@pytest.mark.parametrize(
    "tamper",
    [
        "aggregate",
        "case_success",
        "case_ineffective",
        "case_missing",
        "panel",
        "seed",
        "layout",
    ],
)
def test_exam_case_evidence_rejects_any_aggregate_or_atomic_case_tamper(
    tamper: str,
) -> None:
    evaluations, cases, expected_seeds = _exam_case_evidence()
    public_evaluations = {lesson: result.public_dict() for lesson, result in evaluations.items()}
    changed_cases = deepcopy(cases)
    changed_evaluations = deepcopy(public_evaluations)
    if tamper == "aggregate":
        changed_evaluations[lessons.LessonId.NAVIGATE]["successes"] -= 1
    elif tamper == "case_success":
        changed_cases[0]["success"] = not changed_cases[0]["success"]
    elif tamper == "case_ineffective":
        changed_cases[80]["ineffective_interactions"] += 1
    elif tamper == "case_missing":
        changed_cases.pop()
    elif tamper == "panel":
        changed_cases[0]["panel"] = 1
    elif tamper == "seed":
        changed_cases[0]["seed"], changed_cases[1]["seed"] = (
            changed_cases[1]["seed"],
            changed_cases[0]["seed"],
        )
    else:
        changed_cases[0]["layout_sha256"] = "not-a-sha"

    with pytest.raises(u2r.U2rProtocolError):
        u2r.verify_exam_case_evidence(
            changed_evaluations,
            changed_cases,
            expected_seeds=expected_seeds,
        )


def test_case_set_digest_changes_for_a_coherently_recomputed_case_change() -> None:
    evaluations, cases, expected_seeds = _exam_case_evidence()
    original = u2r.verify_exam_case_evidence(
        evaluations,
        cases,
        expected_seeds=expected_seeds,
    )
    changed_cases = deepcopy(cases)
    changed_cases[80]["ineffective_interactions"] += 1
    changed_evaluations = dict(evaluations)
    visible_cases = [
        case for case in changed_cases if case["lesson_id"] == lessons.LessonId.VISIBLE_UNLOCK.value
    ]
    changed_evaluations[lessons.LessonId.VISIBLE_UNLOCK] = lessons.aggregate_u2_case_evidence(
        lessons.LessonId.VISIBLE_UNLOCK,
        visible_cases,
        timestamp=evaluations[lessons.LessonId.VISIBLE_UNLOCK].timestamp,
    )

    changed = u2r.verify_exam_case_evidence(
        changed_evaluations,
        changed_cases,
        expected_seeds=expected_seeds,
    )

    assert changed["case_records_sha256"] != original["case_records_sha256"]
    assert changed["case_set_sha256"] != original["case_set_sha256"]
    assert (
        changed["per_lesson"][lessons.LessonId.VISIBLE_UNLOCK.value]["aggregate_sha256"]
        != original["per_lesson"][lessons.LessonId.VISIBLE_UNLOCK.value]["aggregate_sha256"]
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"penultimate_allocation_valid": False},
        {"terminal_allocation_valid": False},
        {"penultimate_recovery": True},
        {"terminal_recovery": True},
    ],
)
def test_invalid_allocation_or_recovery_blocks_terminal_eligibility(
    overrides: dict[str, object],
) -> None:
    decision = _grade(_stable_exam(), _stable_exam(), **overrides)
    assert decision.eligible is False
    assert decision.reasons


@pytest.mark.parametrize(
    "lesson",
    [
        lessons.LessonId.VISIBLE_UNLOCK,
        lessons.LessonId.LOCAL_UNLOCK,
        lessons.LessonId.SEPARATED_UNLOCK,
    ],
)
def test_interaction_ceiling_is_independent_for_every_unlock_lesson(
    lesson: lessons.LessonId,
) -> None:
    at_limit = _stable_exam()
    at_limit[lesson] = replace(
        at_limit[lesson],
        mean_ineffective_interactions=3.0,
    )
    above_limit = dict(at_limit)
    above_limit[lesson] = replace(
        at_limit[lesson],
        mean_ineffective_interactions=3.000_001,
    )

    assert _grade(at_limit, at_limit).eligible is True
    assert _grade(at_limit, above_limit).eligible is False


@pytest.mark.parametrize(
    ("successes", "panels"),
    [
        (71, (35, 36)),
        (72, (33, 39)),
        (72, (39, 33)),
    ],
)
def test_u2_has_stricter_stability_floors_than_ordinary_mastery(
    successes: int,
    panels: tuple[int, int],
) -> None:
    terminal = _stable_exam()
    terminal[lessons.LessonId.SEPARATED_UNLOCK] = _evaluation(
        lessons.LessonId.SEPARATED_UNLOCK,
        successes,
        panels=panels,
    )

    assert _grade(_stable_exam(), terminal).eligible is False


@pytest.mark.parametrize("lesson", list(lessons.LessonId))
@pytest.mark.parametrize("weaken_penultimate", [False, True])
def test_each_final_exam_must_retain_every_existing_lesson_gate(
    lesson: lessons.LessonId,
    weaken_penultimate: bool,
) -> None:
    penultimate = _stable_exam()
    terminal = _stable_exam()
    weakened = _evaluation(lesson, 67, panels=(34, 33))
    if weaken_penultimate:
        penultimate[lesson] = weakened
    else:
        terminal[lesson] = weakened

    assert _grade(penultimate, terminal).eligible is False


@pytest.mark.parametrize("lesson", list(lessons.LessonId))
def test_no_lesson_may_regress_by_more_than_two_terminal_successes(
    lesson: lessons.LessonId,
) -> None:
    penultimate = _stable_exam()
    terminal = _stable_exam()
    terminal_successes = terminal[lesson].successes
    three_lower_panels = (
        (terminal_successes + 3) // 2,
        terminal_successes + 3 - (terminal_successes + 3) // 2,
    )
    penultimate[lesson] = replace(
        penultimate[lesson],
        successes=terminal_successes + 3,
        success_rate=(terminal_successes + 3) / 80,
        panel_successes=three_lower_panels,
        panel_success_rates=tuple(value / 40 for value in three_lower_panels),
    )

    assert _grade(penultimate, terminal).eligible is False

    two_lower_panels = (
        (terminal_successes + 2) // 2,
        terminal_successes + 2 - (terminal_successes + 2) // 2,
    )
    penultimate[lesson] = replace(
        penultimate[lesson],
        successes=terminal_successes + 2,
        success_rate=(terminal_successes + 2) / 80,
        panel_successes=two_lower_panels,
        panel_success_rates=tuple(value / 40 for value in two_lower_panels),
    )
    assert _grade(penultimate, terminal).eligible is True


def test_stability_grading_accepts_persisted_public_evaluation_records() -> None:
    penultimate = {lesson: result.public_dict() for lesson, result in _stable_exam().items()}
    terminal = {lesson.value: result.public_dict() for lesson, result in _stable_exam().items()}

    assert _grade(penultimate, terminal).eligible is True  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "tamper",
    ["missing_metric", "nonfinite_metric", "wrong_episode_count", "panel_total"],
)
def test_stability_grading_fails_closed_on_incomplete_aggregate_evidence(
    tamper: str,
) -> None:
    terminal = {lesson.value: result.public_dict() for lesson, result in _stable_exam().items()}
    record = terminal[lessons.LessonId.VISIBLE_UNLOCK.value]
    if tamper == "missing_metric":
        del record["mean_ineffective_interactions"]
    elif tamper == "nonfinite_metric":
        record["mean_ineffective_interactions"] = float("nan")
    elif tamper == "wrong_episode_count":
        record["episodes"] = 79
    else:
        record["panel_successes"] = [35, 35]

    decision = _grade(_stable_exam(), terminal)  # type: ignore[arg-type]
    assert decision.eligible is False
    assert decision.reasons
