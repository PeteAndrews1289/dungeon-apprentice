from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from dungeon_apprentice import v02_u2_lessons as lesson_specs
from dungeon_apprentice import v03_action_effect_train as trainer
from dungeon_apprentice.artifacts import atomic_write_json, file_sha256

REPOSITORY = Path(__file__).resolve().parents[1]
HELPER = REPOSITORY / "scripts" / "v03_action_effect_manifest.py"


def _load_helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "v03_action_effect_manifest",
        HELPER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _qualification(
    helper: ModuleType,
    *,
    root: Path,
    media: Path,
    source: str,
    tag_object: str,
    directory: Path,
) -> tuple[Path, dict[str, object]]:
    directory.mkdir()
    report = directory / "report.json"
    failed_attempt = {
        "disposition": "operationally_incomplete",
        "resume_authorized": False,
        "reuse_authorized": False,
    }
    _write_json(
        report,
        {
            "schema_version": 1,
            "protocol": helper.PROTOCOL,
            "kind": "sealed_stage_a_preflight_qualification",
            "verdict": "qualified",
            "created_at": "2026-07-24T00:00:00+00:00",
            "claim": {
                "schema_version": 1,
                "protocol": helper.PROTOCOL,
                "kind": "qualification_attempt_claim",
                "claim_id": "fixture",
                "source_commit": source,
                "tag": helper.TAG_NAME,
                "tag_object": tag_object,
                "created_at": "2026-07-24T00:00:00+00:00",
            },
            "source": {"commit": source, "dirty": False},
            "protocol_document": {
                "path": helper.v03.PROTOCOL_DOCUMENT,
                "sha256": "1" * 64,
            },
            "parent": {},
            "predecessors": {},
            "failed_stage_a_attempt": failed_attempt,
            "guards": {},
            "sampler_preflight": [],
            "architecture_contract": {},
            "protected_partitions": [],
            "smoke_evidence": {
                "_full_report": {
                    "protocol": helper.SMOKE_PROTOCOL,
                    "verdict": "passed",
                },
                "protocol": helper.SMOKE_PROTOCOL,
                "verdict": "passed",
                "pre_action_rng_identical": True,
                "pre_update_behavior_identical": True,
                "first_rollout_trajectory_identical": True,
                "first_rollout_policy_outputs_identical": True,
                "post_rollout_pre_optimizer_rng_identical": True,
                "learning_divergence_begins_after_first_update": True,
                "temporary_root_removed": True,
            },
            "storage_caps": {
                "per_arm_bytes": helper.LINEAGE_CAP_BYTES,
                "scientific_cohort_bytes": (
                    helper.COHORT_SCIENTIFIC_CAP_BYTES
                ),
                "media_bytes": helper.MEDIA_CAP_BYTES,
                "combined_bytes": helper.COMBINED_PLANNED_CAP_BYTES,
            },
            "storage_preflight": {},
            "restrictions": {
                "arms_run_sequentially": True,
                "stage_a_resume_supported": False,
                "failed_attempt_resume_authorized": False,
                "failed_attempt_root_reuse_authorized": False,
                "replacement_restarts_both_arms_from_confirmed_u1": True,
                "development_checkpoint_reuse_authorized": False,
                "u3_authorized": False,
            },
        },
    )
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    checksum = report.with_name("report.json.sha256")
    checksum.write_text(
        f"{digest}  report.json\n",
        encoding="utf-8",
    )
    public: dict[str, object] = {
        "report": str(report),
        "report_sha256": digest,
        "checksum": str(checksum),
        "source_commit": source,
        "tag": helper.TAG_NAME,
        "tag_object": tag_object,
        "tag_payload_sha256": "2" * 64,
        "verdict": "qualified",
        "protocol_document_sha256": "1" * 64,
        "guard_mapping_sha256": "3" * 64,
        "sampler_preflight_sha256": "4" * 64,
        "architecture_contract_sha256": "5" * 64,
        "smoke_evidence_sha256": "6" * 64,
        "protected_partitions_sha256": "7" * 64,
        "failed_stage_a_attempt_sha256": helper._canonical_sha256(
            failed_attempt
        ),
        "storage_caps": {
            "per_arm_bytes": helper.LINEAGE_CAP_BYTES,
            "scientific_cohort_bytes": (
                helper.COHORT_SCIENTIFIC_CAP_BYTES
            ),
            "media_bytes": helper.MEDIA_CAP_BYTES,
            "combined_bytes": helper.COMBINED_PLANNED_CAP_BYTES,
        },
    }
    return report, public


def _exam(
    helper: ModuleType,
    boundary: int,
    *,
    eligible: bool = True,
) -> dict:
    lessons = {
        lesson_id: {
            "lesson_id": lesson_id,
            "successes": (
                71
                if lesson_id == "unlock/u2-separated" and not eligible
                else 76
            ),
            "episodes": 80,
            "panel_successes": (
                [33, 38]
                if lesson_id == "unlock/u2-separated" and not eligible
                else [38, 38]
            ),
            "mean_ineffective_interactions": 1.0,
        }
        for lesson_id in helper.LESSON_IDS
    }
    return {
        "child_trained_actions": boundary,
        "allocation_valid": True,
        "practice_profile": "normal",
        "lessons": lessons,
        "case_diagnostics": {
            "max_case_ineffective_interactions": 3,
            "cases_with_ineffective_at_least_10": 0,
            "max_repeated_identical_interaction_run": 3,
        },
    }


def _valid_exam_case_evidence(
    helper: ModuleType,
    *,
    eligible: bool,
) -> tuple[dict[str, dict], list[dict]]:
    records: list[dict] = []
    evaluations: dict[str, dict] = {}
    for lesson in lesson_specs.LessonId:
        lesson_records: list[dict] = []
        for case_index in range(helper.v03.EVALUATION_SEED_COUNT):
            panel = 0 if case_index < 40 else 1
            panel_case_index = case_index % 40
            success_limit = (
                33
                if (
                    lesson is lesson_specs.LessonId.SEPARATED_UNLOCK
                    and not eligible
                )
                else 36
            )
            success = panel_case_index < success_limit
            specification = lesson_specs.LESSON_SPECS[lesson]
            oracle_range = specification.oracle_action_range
            oracle_actions = (
                oracle_range[0] if oracle_range is not None else 4
            )
            steps = min(
                oracle_actions + 5,
                specification.max_steps,
            )
            visible_effects = (
                trainer.frozen_u2s.visible_interaction_effect_evidence(
                    [],
                    [],
                    [],
                    penalty_enabled=False,
                )
            )
            lesson_records.append(
                {
                    "lesson_id": lesson.value,
                    "lesson_label": specification.label,
                    "case_index": case_index,
                    "panel": panel,
                    "panel_case_index": panel_case_index,
                    "seed": (
                        specification.validation_seed_base + case_index
                    ),
                    "layout_sha256": trainer._canonical_sha256(
                        [lesson.value, case_index, "layout"]
                    ),
                    "geometry_sha256": trainer._canonical_sha256(
                        [lesson.value, case_index, "geometry"]
                    ),
                    "success": success,
                    "terminal_reason": (
                        "success" if success else "time_limit"
                    ),
                    "steps": steps,
                    "milestones": {
                        "key_picked_up": success,
                        "door_opened": success,
                        "relic_picked_up": False,
                        "entrance_revisited": False,
                        "success": success,
                    },
                    "collisions": 0,
                    "ineffective_interactions": 0,
                    "unique_cells": 5,
                    "reachable_cells": 10,
                    "coverage": 0.5,
                    "oracle_actions": oracle_actions,
                    "path_actions_per_oracle_action": (
                        steps / oracle_actions
                    ),
                    "action_histogram": {
                        str(action): steps if action == 0 else 0
                        for action in range(7)
                    },
                    "longest_repeated_action_run": steps,
                    "longest_identical_visible_no_effect_streak": 0,
                    "longest_repeated_identical_interaction_run": 0,
                    "visible_no_effect_transition_count": 0,
                    "visible_interaction_effects": visible_effects,
                    "action_trace_sha256": trainer._canonical_sha256(
                        [lesson.value, case_index, "actions"]
                    ),
                    "visible_no_effect_streak_sha256": (
                        trainer._canonical_sha256([])
                    ),
                    "interaction_run_sha256": (
                        trainer._canonical_sha256([])
                    ),
                    "extrinsic_return": 1.0 if success else 0.0,
                    "curiosity_return": 0.0,
                    "initial_key_visible": (
                        False
                        if lesson
                        is lesson_specs.LessonId.SEPARATED_UNLOCK
                        else None
                    ),
                    "initial_door_visible": (
                        False
                        if lesson
                        is lesson_specs.LessonId.SEPARATED_UNLOCK
                        else None
                    ),
                    "visibility_stratum": (
                        "key_hidden_door_hidden"
                        if lesson
                        is lesson_specs.LessonId.SEPARATED_UNLOCK
                        else None
                    ),
                }
            )
        result = lesson_specs.aggregate_u2_case_evidence(
            lesson,
            lesson_records,
            timestamp="2026-07-24T00:00:00+00:00",
        )
        evaluations[lesson.value] = (
            trainer.frozen_u2s._augmented_lesson_evidence(
                result,
                lesson_records,
                penalty_enabled=False,
            )
        )
        records.extend(lesson_records)
    return evaluations, records


def _first_rollout(helper: ModuleType, arm: str) -> dict:
    return {
        "transitions": helper.v03.ROLLOUT_TRANSITIONS,
        "first_optimizer_boundary": helper.v03.ROLLOUT_TRANSITIONS,
        "pre_update_identity_verified": True,
        "divergence_before_first_optimizer": False,
        "first_divergence": (
            "not_observed" if arm == "sham" else "after_first_optimizer"
        ),
        "pre_action_rng_sha256": "1" * 64,
        "trajectory_sha256": "2" * 64,
        "policy_output_sha256": "3" * 64,
        "post_rollout_rng_sha256": "4" * 64,
    }


def _first_rollout_identity(helper: ModuleType) -> dict:
    core = {
        "trajectory_identity": [
            {
                "worker_index": index,
                "transitions": helper.v03.ROLLOUT_STEPS,
                "trajectory_sha256": f"{index + 10:064x}",
            }
            for index in range(helper.v03.WORKERS)
        ],
        "policy_output_sha256": "3" * 64,
        "post_rollout_rng_identity": {
            "phase": "post_rollout_pre_optimizer",
            "aggregate_sha256": "4" * 64,
        },
        "episode_ledger_normalized_sha256": "5" * 64,
    }
    return {
        **core,
        "aggregate_sha256": helper._canonical_sha256(core),
    }


def _arm_terminal(
    helper: ModuleType,
    *,
    root: Path,
    source: str,
    arm: str,
    eligible: bool,
) -> None:
    contract_digest = hashlib.sha256(
        (root / "cohort-contract.json").read_bytes()
    ).hexdigest()
    contract = json.loads(
        (root / "cohort-contract.json").read_text(encoding="utf-8")
    )
    directory = root / arm
    rolling = directory / "checkpoints" / "rolling"
    rolling.mkdir(parents=True)
    evaluations, cases = _valid_exam_case_evidence(
        helper,
        eligible=eligible,
    )
    exams: list[dict] = []
    case_files: list[dict] = []
    source_identity = {"commit": source, "dirty": False}
    for index in range(1, helper.EXAM_COUNT + 1):
        boundary = index * helper.EVALUATION_EVERY
        checkpoint = rolling / f"exam-{boundary:07d}.zip"
        checkpoint.write_bytes(f"exam {boundary}".encode())
        relative_checkpoint = str(checkpoint.relative_to(directory))
        exam_sidecar = {
            "schema_version": trainer.SCHEMA_VERSION,
            "protocol": helper.PROTOCOL,
            "kind": "exam",
            "arm": arm,
            "cohort_id": helper.COHORT_ID,
            "cohort_contract_sha256": contract_digest,
            "resume_eligible": False,
            "resume_authorized": False,
            "promotable": False,
            "development_checkpoint_reuse_authorized": False,
            "checkpoint_sha256": file_sha256(checkpoint),
            "source": source_identity,
            "qualification": contract["qualification"],
            "progress": {
                "child_trained_actions": boundary,
                "lifetime_trained_actions": (
                    helper.v03.PARENT_LIFETIME_ACTIONS + boundary
                ),
                "optimizer_updates": (
                    helper.v03.PARENT_OPTIMIZER_UPDATES
                    + boundary
                    // helper.v03.ROLLOUT_TRANSITIONS
                    * helper.v03.PPO_EPOCHS
                ),
            },
        }
        atomic_write_json(
            checkpoint.with_suffix(".json"),
            exam_sidecar,
        )
        trainer._write_integrity(
            checkpoint,
            checkpoint.with_suffix(".json"),
        )
        diagnostics = (
            trainer.frozen_u2s._case_diagnostics_from_records(cases)
        )
        cases_path = checkpoint.with_suffix(".cases.json")
        case_document = {
            "schema_version": trainer.CASE_EVIDENCE_SCHEMA_VERSION,
            "protocol": helper.PROTOCOL,
            "arm": arm,
            "checkpoint": relative_checkpoint,
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_sidecar_sha256": file_sha256(
                checkpoint.with_suffix(".json")
            ),
            "child_trained_actions": boundary,
            "case_diagnostics": diagnostics,
            "records": cases,
        }
        atomic_write_json(cases_path, case_document)
        relative_cases = str(cases_path.relative_to(directory))
        cases_sha256 = file_sha256(cases_path)
        case_files.append(
            {
                "path": relative_cases,
                "sha256": cases_sha256,
                "records": len(cases),
            }
        )
        exams.append(
            {
                "child_trained_actions": boundary,
                "allocation_valid": True,
                "practice_profile": "normal",
                "checkpoint": relative_checkpoint,
                "checkpoint_sha256": file_sha256(checkpoint),
                "sidecar_sha256": file_sha256(
                    checkpoint.with_suffix(".json")
                ),
                "lessons": evaluations,
                "case_diagnostics": {
                    **diagnostics,
                    "path": relative_cases,
                    "sha256": cases_sha256,
                },
            }
        )
    terminal = directory / "checkpoints" / "terminal.zip"
    terminal.write_bytes(b"fixture terminal checkpoint")
    sidecar = {
        "schema_version": trainer.SCHEMA_VERSION,
        "protocol": helper.PROTOCOL,
        "kind": "terminal",
        "arm": arm,
        "cohort_id": helper.COHORT_ID,
        "cohort_contract_sha256": contract_digest,
        "resume_eligible": False,
        "resume_authorized": False,
        "promotable": False,
        "development_checkpoint_reuse_authorized": False,
        "checkpoint_sha256": file_sha256(terminal),
        "progress": {
            "child_trained_actions": helper.ACTION_CAP,
        },
    }
    atomic_write_json(terminal.with_suffix(".json"), sidecar)
    trainer._write_integrity(terminal, terminal.with_suffix(".json"))
    grade = helper.v03.grade_terminal(arm, exams)
    initial_encoder = {
        "child_trained_actions": 0,
        "optimizer_updates": helper.v03.PARENT_OPTIMIZER_UPDATES,
        "weight_l2_norm": 0.0,
        "weight_max_abs": 0.0,
        "weight_nonzero_parameters": 0,
        "update_l2_norm": 0.0,
        "update_max_abs": 0.0,
        "update_nonzero_parameters": 0,
    }
    terminal_encoder = {
        "child_trained_actions": helper.ACTION_CAP,
        "optimizer_updates": helper.TERMINAL_OPTIMIZER_UPDATES,
        "weight_l2_norm": 0.0 if arm == "sham" else 1.25,
        "weight_max_abs": 0.0 if arm == "sham" else 0.25,
        "weight_nonzero_parameters": 0 if arm == "sham" else 16,
        "update_l2_norm": 0.0 if arm == "sham" else 0.01,
        "update_max_abs": 0.0 if arm == "sham" else 0.005,
        "update_nonzero_parameters": 0 if arm == "sham" else 16,
    }
    context_metrics = {
        "transitions": helper.ACTION_CAP,
        "active_contexts": (
            0 if arm == "sham" else helper.ACTION_CAP - 1
        ),
        "activation_rate": (
            0.0 if arm == "sham" else (helper.ACTION_CAP - 1) / helper.ACTION_CAP
        ),
        "changed": helper.ACTION_CAP // 2,
        "unchanged": helper.ACTION_CAP // 2,
        "changed_rate": 0.5,
        "by_action": {},
        "by_lesson": {},
    }
    context_summary = {**context_metrics, "unchanged_rate": 0.5}
    context_encoder = {
        "initial_weight_norm": 0.0,
        "initial_nonzero_parameters": 0,
        "residual_zero_before_action_one": True,
        "terminal_weight_norm": terminal_encoder["weight_l2_norm"],
        "terminal_update_norm": terminal_encoder["weight_l2_norm"],
        "terminal_nonzero_parameters": terminal_encoder[
            "weight_nonzero_parameters"
        ],
        "first_nonzero_child_actions": (
            None if arm == "sham" else helper.v03.ROLLOUT_TRANSITIONS
        ),
    }
    report = {
        "schema_version": 1,
        "protocol": helper.PROTOCOL,
        "cohort_id": helper.COHORT_ID,
        "cohort_contract_sha256": contract_digest,
        "arm": arm,
        "verdict": (
            "architecture_candidate_passed" if eligible else "arm_failed"
        ),
        "development_only": True,
        "architecture_definition_eligible": (
            arm == "action-effect" and eligible
        ),
        "sham_calibration_only": arm == "sham",
        "successor_checkpoint_authorized": False,
        "resume_authorized": False,
        "promotable": False,
        "u3_authorized": False,
        "source": source_identity,
        "source_clean_at_closeout": True,
        "qualification": contract["qualification"],
        "qualification_sha256": contract["qualification"]["report_sha256"],
        "development_checkpoint_reuse_authorized": False,
        "progress": {
            "child_trained_actions": helper.ACTION_CAP,
            "lifetime_trained_actions": helper.TERMINAL_LIFETIME_ACTIONS,
            "optimizer_updates": helper.TERMINAL_OPTIMIZER_UPDATES,
            "exam_count": helper.EXAM_COUNT,
        },
        "grade": grade.public_dict(),
        "controller": {
            "exam_records": exams,
            "practice_decisions": [],
        },
        "first_rollout_identity": _first_rollout_identity(helper),
        "context_metrics": context_metrics,
        "encoder_history": [initial_encoder, terminal_encoder],
        "terminal_encoder": terminal_encoder,
        "case_evidence": {
            "files": case_files,
            "file_count": helper.EXAM_COUNT,
            "record_count": helper.EXPECTED_CASE_COUNT,
            "inventory_sha256": trainer._canonical_sha256(case_files),
        },
        "terminal_checkpoint": {
            "path": str(terminal.relative_to(directory)),
            "sha256": file_sha256(terminal),
            "sidecar_sha256": file_sha256(terminal.with_suffix(".json")),
            "integrity_sha256": file_sha256(
                trainer._integrity_path(terminal)
            ),
            "promotable": False,
        },
        "case_count": helper.EXPECTED_CASE_COUNT,
        "exam_records": exams,
        "terminal_eligible": eligible,
        "first_rollout": _first_rollout(helper, arm),
        "context_encoder": context_encoder,
        "context_summary": context_summary,
    }
    report_path = directory / "report.json"
    _write_json(report_path, report)
    report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
    _write_json(
        directory / "report.integrity.json",
        {
            "schema_version": 1,
            "protocol": helper.PROTOCOL,
            "cohort_id": helper.COHORT_ID,
            "arm": arm,
            "cohort_contract_sha256": contract_digest,
            "report": "report.json",
            "report_sha256": report_sha256,
            "terminal_checkpoint_sha256": file_sha256(terminal),
            "terminal_sidecar_sha256": file_sha256(
                terminal.with_suffix(".json")
            ),
            "terminal_integrity_sha256": file_sha256(
                trainer._integrity_path(terminal)
            ),
        },
    )
    _write_json(
        directory / "status.json",
        {
            "protocol": helper.PROTOCOL,
            "cohort_id": helper.COHORT_ID,
            "phase": "completed",
            "arm": arm,
            "source": {"commit": source, "dirty": False},
            "qualification_sha256": contract["qualification"][
                "report_sha256"
            ],
            "cohort_contract_sha256": contract_digest,
            "parent_checkpoint_sha256": (
                helper.v03.PARENT_CHECKPOINT_SHA256
            ),
            "action_cap": helper.ACTION_CAP,
            "child_trained_actions": helper.ACTION_CAP,
            "collected_actions": helper.ACTION_CAP,
            "child_collected_actions": helper.ACTION_CAP,
            "remaining_action_budget": 0,
            "optimizer_updates": helper.TERMINAL_OPTIMIZER_UPDATES,
            "exam_count": helper.EXAM_COUNT,
            "report_sha256": report_sha256,
            "updated_at": "2026-07-24T01:00:00+00:00",
        },
    )


def _cohort(
    tmp_path: Path,
) -> tuple[ModuleType, Path, Path, str, str]:
    helper = _load_helper()
    root = tmp_path / "cohort"
    media = tmp_path / "media"
    root.mkdir()
    media.mkdir()
    source = "a" * 40
    tag_object = "b" * 40
    qualification, qualification_public = _qualification(
        helper,
        root=root,
        media=media,
        source=source,
        tag_object=tag_object,
        directory=tmp_path / "qualification",
    )

    def verify_fixture(
        report_path: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        if hashlib.sha256(report_path.read_bytes()).hexdigest() != (
            qualification_public["report_sha256"]
        ):
            raise helper.V03ManifestError(
                "v0.3 qualification report changed"
            )
        return dict(qualification_public)

    helper._verified_qualification_binding = verify_fixture
    helper.create_cohort(
        root,
        media_root=media,
        qualification_report=qualification,
        source_commit=source,
        tag_object=tag_object,
    )
    return helper, root, media, source, tag_object


def _clear_process_rows() -> list[dict[str, object]]:
    return [
        {"pid": 1, "ppid": 0, "command": "/sbin/launchd"},
        {
            "pid": 420,
            "ppid": 1,
            "command": (
                "python -m "
                "dungeon_apprentice.v03_action_effect_dashboard "
                "--run-root /fixture --port 8789"
            ),
        },
    ]


def _seal_process_closeout(
    helper: ModuleType,
    *,
    root: Path,
    source: str,
    tag_object: str,
) -> dict[str, object]:
    helper._gather_process_rows = _clear_process_rows
    return helper.seal_process_closeout(
        root,
        source_commit=source,
        tag_object=tag_object,
    )


def test_manifest_identity_is_v03_fresh_only_and_has_no_resume() -> None:
    helper = _load_helper()
    assert helper.PROTOCOL == (
        "dungeon-apprentice-v0.3-action-effect-architecture"
    )
    assert helper.COHORT_ID == "v0.3-action-effect-stage-a-r1-20260724"
    assert helper.TAG_NAME == (
        "action-effect-architecture-v0.3-stage-a-r1-20260724"
    )
    assert helper.DEFAULT_DASHBOARD_PORT == 8789
    assert helper.ARM_ORDER == ("sham", "action-effect")
    parser = helper.build_parser()
    parsed = parser.parse_args(
        [
            "seal-process-closeout",
            "--source-commit",
            "a" * 40,
            "--tag-object",
            "b" * 40,
        ]
    )
    assert parsed.command == "seal-process-closeout"
    source = HELPER.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        parser.parse_args(["resume"])
    assert "def resume" not in source
    assert "U2-S" not in source


def test_manifest_consumes_real_qualifier_public_api_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dungeon_apprentice import v03_action_effect_qualify as qualifier

    helper = _load_helper()
    root = tmp_path / "cohort"
    media = tmp_path / "media"
    report = tmp_path / "qualification" / "report.json"
    report.parent.mkdir()
    report.write_text('{"fixture":true}\n', encoding="utf-8")
    source = "a" * 40
    tag_object = "b" * 40
    failed_attempt = {
        "disposition": "operationally_incomplete",
        "resume_authorized": False,
        "reuse_authorized": False,
    }
    public = {
        "report": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "checksum": str(report.with_name("report.json.sha256")),
        "source_commit": source,
        "tag": helper.TAG_NAME,
        "tag_object": tag_object,
        "tag_payload_sha256": "1" * 64,
        "verdict": "qualified",
        "protocol_document_sha256": "2" * 64,
        "guard_mapping_sha256": "3" * 64,
        "sampler_preflight_sha256": "4" * 64,
        "architecture_contract_sha256": "5" * 64,
        "smoke_evidence_sha256": "6" * 64,
        "protected_partitions_sha256": "7" * 64,
        "failed_stage_a_attempt_sha256": helper._canonical_sha256(
            failed_attempt
        ),
        "storage_caps": {
            "per_arm_bytes": helper.LINEAGE_CAP_BYTES,
            "scientific_cohort_bytes": (
                helper.COHORT_SCIENTIFIC_CAP_BYTES
            ),
            "media_bytes": helper.MEDIA_CAP_BYTES,
            "combined_bytes": helper.COMBINED_PLANNED_CAP_BYTES,
        },
    }
    verified_report = {
        "protocol": helper.PROTOCOL,
        "kind": qualifier.KIND,
        "verdict": "qualified",
        "smoke_evidence": {"_full_report": {"verdict": "passed"}},
        "failed_stage_a_attempt": failed_attempt,
        "restrictions": {
            "arms_run_sequentially": True,
            "stage_a_resume_supported": False,
            "failed_attempt_resume_authorized": False,
            "failed_attempt_root_reuse_authorized": False,
            "replacement_restarts_both_arms_from_confirmed_u1": True,
            "development_checkpoint_reuse_authorized": False,
            "u3_authorized": False,
        },
    }

    class Evidence:
        def public_dict(self) -> dict[str, object]:
            return public

        def verified_report(self) -> dict[str, object]:
            return verified_report

    called: dict[str, object] = {}

    def verify(path: Path, **kwargs: object) -> Evidence:
        called["path"] = path
        called.update(kwargs)
        return Evidence()

    monkeypatch.setattr(qualifier, "CANONICAL_REPORT", report)
    monkeypatch.setattr(qualifier, "CANONICAL_COHORT_ROOT", root)
    monkeypatch.setattr(qualifier, "CANONICAL_MEDIA_ROOT", media)
    monkeypatch.setattr(qualifier, "verify_action_effect_qualification", verify)
    binding = helper._verified_qualification_binding(
        report,
        source_commit=source,
        tag_object=tag_object,
        root=root,
        media_root=media,
    )
    assert binding == public
    assert called["path"] == report
    assert called["expected_source_commit"] == source
    assert called["expected_tag_object"] == tag_object


def test_manifest_enforces_fixed_order_concurrency_and_fresh_targets(
    tmp_path: Path,
) -> None:
    helper, root, _media, source, tag_object = _cohort(tmp_path)
    assert helper.next_plan(
        root,
        source_commit=source,
        tag_object=tag_object,
    )["arm"] == "sham"
    helper.start_arm(
        root,
        source_commit=source,
        tag_object=tag_object,
        arm_id="sham",
    )
    with pytest.raises(helper.V03ManifestError, match="already active"):
        helper.start_arm(
            root,
            source_commit=source,
            tag_object=tag_object,
            arm_id="action-effect",
        )
    with pytest.raises(helper.V03ManifestError, match="already active"):
        helper.next_plan(
            root,
            source_commit=source,
            tag_object=tag_object,
        )
    _arm_terminal(
        helper,
        root=root,
        source=source,
        arm="sham",
        eligible=True,
    )
    helper.finish_arm(
        root,
        source_commit=source,
        tag_object=tag_object,
        arm_id="sham",
        outcome="completed",
        trainer_exit_code=0,
    )
    assert helper.next_plan(
        root,
        source_commit=source,
        tag_object=tag_object,
    )["arm"] == "action-effect"
    with pytest.raises(helper.V03ManifestError, match="fresh-only"):
        helper.start_arm(
            root,
            source_commit=source,
            tag_object=tag_object,
            arm_id="sham",
        )


def test_manifest_terminal_selection_is_candidate_only_and_nonpromotable(
    tmp_path: Path,
) -> None:
    helper, root, _media, source, tag_object = _cohort(tmp_path)
    for arm, eligible in (("sham", False), ("action-effect", True)):
        helper.start_arm(
            root,
            source_commit=source,
            tag_object=tag_object,
            arm_id=arm,
        )
        _arm_terminal(
            helper,
            root=root,
            source=source,
            arm=arm,
            eligible=eligible,
        )
        helper.finish_arm(
            root,
            source_commit=source,
            tag_object=tag_object,
            arm_id=arm,
            outcome="completed",
            trainer_exit_code=0,
        )
    with pytest.raises(
        helper.V03ManifestError,
        match="has not been sealed",
    ):
        helper.finalize_cohort(
            root,
            source_commit=source,
            tag_object=tag_object,
        )
    helper._gather_process_rows = lambda: [
        {
            "pid": 77,
            "ppid": 1,
            "command": (
                "python scripts/u2_trainer_supervisor.py --state fixture"
            ),
        }
    ]
    with pytest.raises(helper.V03ManifestError, match="supervisor:77"):
        helper.seal_process_closeout(
            root,
            source_commit=source,
            tag_object=tag_object,
        )
    binding = _seal_process_closeout(
        helper,
        root=root,
        source=source,
        tag_object=tag_object,
    )
    assert binding["verdict"] == "clear"
    assert binding["dashboard_pids"] == [420]
    helper._gather_process_rows = lambda: [
        {
            "pid": 88,
            "ppid": 1,
            "command": (
                "python -m "
                "dungeon_apprentice.v03_action_effect_train train-arm"
            ),
        }
    ]
    with pytest.raises(helper.V03ManifestError, match="trainer:88"):
        helper.finalize_cohort(
            root,
            source_commit=source,
            tag_object=tag_object,
        )
    helper._gather_process_rows = lambda: [
        {"pid": 2, "ppid": 0, "command": "/sbin/kernel_task"},
    ]
    terminal = helper.finalize_cohort(
        root,
        source_commit=source,
        tag_object=tag_object,
    )
    assert terminal["verdict"] == "architecture_selected"
    assert terminal["selected_architecture"] == "action-effect"
    assert terminal["stage_a_checkpoint_reuse_authorized"] is False
    assert terminal["u3_authorized"] is False
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert report["paired_first_rollout"][
        "identical_before_first_optimizer"
    ]
    assert report["selection"]["sham_is_calibration_only"]
    assert report["selection"]["replication_protocol_authorized"]
    assert report["checkpoint_rule"]["successor_checkpoint"] is None
    assert report["process_closeout"] == binding
    assert report["process_finalization_recheck"]["observed_rows"] == 1
    integrity = json.loads(
        (root / "report.integrity.json").read_text(encoding="utf-8")
    )
    assert integrity["process_closeout_sha256"] == binding["sha256"]
    assert integrity["process_inventory_sha256"] == binding[
        "inventory_sha256"
    ]


def test_interruption_is_terminal_and_cannot_resume(
    tmp_path: Path,
) -> None:
    helper, root, _media, source, tag_object = _cohort(tmp_path)
    helper.start_arm(
        root,
        source_commit=source,
        tag_object=tag_object,
        arm_id="sham",
    )
    contract = json.loads(
        (root / "cohort-contract.json").read_text(encoding="utf-8")
    )
    directory = root / "sham"
    directory.mkdir()
    _write_json(
        directory / "status.json",
        {
            "protocol": helper.PROTOCOL,
            "cohort_id": helper.COHORT_ID,
            "phase": "interrupted",
            "arm": "sham",
            "source": {"commit": source, "dirty": False},
            "qualification_sha256": contract["qualification"][
                "report_sha256"
            ],
            "cohort_contract_sha256": hashlib.sha256(
                (root / "cohort-contract.json").read_bytes()
            ).hexdigest(),
            "parent_checkpoint_sha256": (
                helper.v03.PARENT_CHECKPOINT_SHA256
            ),
            "action_cap": helper.ACTION_CAP,
        },
    )
    result = helper.finish_arm(
        root,
        source_commit=source,
        tag_object=tag_object,
        arm_id="sham",
        outcome="interrupted",
        trainer_exit_code=130,
    )
    assert result["phase"] == "operationally_incomplete"
    with pytest.raises(helper.V03ManifestError, match="non-resumable"):
        helper.next_plan(
            root,
            source_commit=source,
            tag_object=tag_object,
        )


def test_sealed_process_evidence_rejects_tamper_and_missing_file(
    tmp_path: Path,
) -> None:
    helper, root, _media, source, tag_object = _cohort(tmp_path)
    state_path = root / "cohort.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["phase"] = "awaiting_closeout"
    for arm in state["arms"]:
        arm["state"] = "completed"
    _write_json(state_path, state)
    _seal_process_closeout(
        helper,
        root=root,
        source=source,
        tag_object=tag_object,
    )
    evidence_path = root / helper.PROCESS_CLOSEOUT_NAME
    original = evidence_path.read_bytes()
    evidence = json.loads(original)
    evidence["inventory"][0]["command_sha256"] = "0" * 64
    _write_json(evidence_path, evidence)
    with pytest.raises(
        helper.V03ManifestError,
        match="process closeout",
    ):
        helper.next_plan(
            root,
            source_commit=source,
            tag_object=tag_object,
        )
    evidence_path.write_bytes(original)
    evidence_path.unlink()
    with pytest.raises(helper.V03ManifestError, match="missing"):
        helper.next_plan(
            root,
            source_commit=source,
            tag_object=tag_object,
        )


def test_qualification_byte_drift_blocks_every_later_operation(
    tmp_path: Path,
) -> None:
    helper, root, _media, source, tag_object = _cohort(tmp_path)
    contract = json.loads(
        (root / "cohort-contract.json").read_text(encoding="utf-8")
    )
    qualification = Path(contract["qualification"]["report"])
    qualification.write_text("{}\n", encoding="utf-8")
    with pytest.raises(helper.V03ManifestError, match="qualification"):
        helper.next_plan(
            root,
            source_commit=source,
            tag_object=tag_object,
        )


def test_first_rollout_mismatch_fails_terminal_closeout(
    tmp_path: Path,
) -> None:
    helper, root, _media, source, tag_object = _cohort(tmp_path)
    for arm in helper.ARM_ORDER:
        helper.start_arm(
            root,
            source_commit=source,
            tag_object=tag_object,
            arm_id=arm,
        )
        _arm_terminal(
            helper,
            root=root,
            source=source,
            arm=arm,
            eligible=True,
        )
        if arm == "action-effect":
            path = root / arm / "report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            identity = report["first_rollout_identity"]
            identity["policy_output_sha256"] = "9" * 64
            identity["aggregate_sha256"] = helper._canonical_sha256(
                {
                    key: value
                    for key, value in identity.items()
                    if key != "aggregate_sha256"
                }
            )
            _write_json(path, report)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            integrity = json.loads(
                (root / arm / "report.integrity.json").read_text(
                    encoding="utf-8"
                )
            )
            integrity["report_sha256"] = digest
            _write_json(root / arm / "report.integrity.json", integrity)
            status = json.loads(
                (root / arm / "status.json").read_text(encoding="utf-8")
            )
            status["report_sha256"] = digest
            _write_json(root / arm / "status.json", status)
        helper.finish_arm(
            root,
            source_commit=source,
            tag_object=tag_object,
            arm_id=arm,
            outcome="completed",
            trainer_exit_code=0,
        )
    _seal_process_closeout(
        helper,
        root=root,
        source=source,
        tag_object=tag_object,
    )
    with pytest.raises(helper.V03ManifestError, match="closeout failed"):
        helper.finalize_cohort(
            root,
            source_commit=source,
            tag_object=tag_object,
        )
    state = json.loads((root / "cohort.json").read_text(encoding="utf-8"))
    assert state["phase"] == "integrity_failed"
