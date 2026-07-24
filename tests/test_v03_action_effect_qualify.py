from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import pytest

from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v02_u2s
from dungeon_apprentice import v03_action_effect as v03
from dungeon_apprentice import v03_action_effect_qualify as qualify
from dungeon_apprentice import v03_action_effect_smoke as smoke
from dungeon_apprentice.action_effect import ActionEffectMode


def _digest(character: str) -> str:
    return character * 64


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm_report(
    root: Path,
    name: str,
    *,
    layout_character: str,
) -> tuple[dict[str, object], dict[str, object]]:
    arm = root / name
    arm.mkdir()
    records = [
        {
            "type": "episode_start",
            "lesson_id": lesson.value,
            "layout_sha256": _digest(layout_character),
        }
        for lesson in lessons.LessonId
    ]
    ledger = arm / "episode-starts.jsonl"
    ledger.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    active = [
        {
            "worker_index": index,
            "lesson_id": lesson.value,
            "layout_sha256": _digest(layout_character),
        }
        for index, lesson in enumerate(lessons.LessonId)
    ]
    report = {
        "arm": name,
        "successor_checkpoint_authorized": False,
        "resume_authorized": False,
        "progress": {
            "child_trained_actions": v02_u2s.CHILD_ACTION_BUDGET,
            "exam_count": v02_u2s.EXAM_COUNT,
        },
        "training_episode_evidence": {
            "episode_starts": {
                "path": ledger.name,
                "sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
                "record_count": len(records),
            },
            "terminal_active_workers": active,
            "terminal_active_workers_sha256": qualify._canonical_sha256(
                active
            ),
        },
    }
    report_path = arm / "report.json"
    report_sha256 = _write_json(report_path, report)
    integrity = {
        "schema_version": 1,
        "protocol": v02_u2s.PROTOCOL,
        "report": "report.json",
        "report_sha256": report_sha256,
    }
    integrity_path = arm / "report.integrity.json"
    integrity_sha256 = _write_json(integrity_path, integrity)
    binding: dict[str, object] = {
        "report": f"{name}/report.json",
        "report_sha256": report_sha256,
        "report_integrity": f"{name}/report.integrity.json",
        "report_integrity_sha256": integrity_sha256,
        "child_trained_actions": v02_u2s.CHILD_ACTION_BUDGET,
        "exam_count": v02_u2s.EXAM_COUNT,
        "mechanism_selection_eligible": False,
        "verdict": "arm_failed",
    }
    return binding, report


def _terminal_fixture(
    root: Path,
) -> tuple[dict[str, object], str, str]:
    bindings: dict[str, object] = {}
    for index, arm in enumerate(v02_u2s.ARM_PRIORITY, start=1):
        binding, _report = _arm_report(
            root,
            arm.value,
            layout_character=str(index),
        )
        bindings[arm.value] = binding
    report: dict[str, object] = {
        "protocol": v02_u2s.PROTOCOL,
        "verdict": "ablation_failed",
        "selected_configuration": None,
        "source": {"commit": qualify.U2S_TERMINAL_SOURCE_COMMIT},
        "qualification": {
            "report_sha256": qualify.U2S_QUALIFICATION_REPORT_SHA256
        },
        "selection": {
            "verdict": "ablation_failed",
            "selected_arm": None,
            "selected_mechanism": None,
            "successor_cohort_authorized": False,
            "ablation_checkpoint_reuse_authorized": False,
        },
        "checkpoint_rule": {
            "ablation_checkpoint_reuse_authorized": False,
            "successor_checkpoint": None,
            "successor_cohort_authorized": False,
        },
        "arm_evidence": bindings,
    }
    report_sha256 = _write_json(root / "report.json", report)
    integrity = {
        "protocol": v02_u2s.PROTOCOL,
        "report": "report.json",
        "report_sha256": report_sha256,
    }
    integrity_sha256 = _write_json(
        root / "report.integrity.json",
        integrity,
    )
    return report, report_sha256, integrity_sha256


def test_failed_stage_a_attempt_authenticates_partial_sham_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "stage-a-r1"
    media = tmp_path / "stage-a-r1-media"
    qualification = tmp_path / "stage-a-r1-qualification"
    root.mkdir()
    media.mkdir()
    (media / "sham").mkdir()
    qualification.mkdir()
    expected_directories = {
        ".v03-staging",
        "sham",
        "sham/checkpoints",
        "sham/checkpoints/rolling",
        "sham/frames",
    }
    for relative in sorted(expected_directories):
        (root / relative).mkdir(parents=True, exist_ok=True)
    expected_files = {
        "cohort-contract.json",
        "cohort.json",
        "dashboard.log",
        "dashboard.pid",
        "launcher-sham-attempt-0.json",
        "sham/checkpoints/initial.cases.json",
        "sham/checkpoints/initial.integrity.json",
        "sham/checkpoints/initial.json",
        "sham/checkpoints/initial.zip",
        "sham/checkpoints/latest-observed.integrity.json",
        "sham/checkpoints/latest-observed.json",
        "sham/checkpoints/latest-observed.zip",
        "sham/checkpoints/rolling/exam-0032768.cases.json",
        "sham/checkpoints/rolling/exam-0032768.integrity.json",
        "sham/checkpoints/rolling/exam-0032768.json",
        "sham/checkpoints/rolling/exam-0032768.zip",
        "sham/episode-starts.jsonl",
        "sham/episodes.jsonl",
        "sham/evaluations.jsonl",
        "sham/events.jsonl",
        "sham/first-rollout.json",
        "sham/frames/exam-navigate-full.png",
        "sham/frames/exam-unlock-u0-visible.png",
        "sham/frames/exam-unlock-u1-local.png",
        "sham/frames/exam-unlock-u2-separated.png",
        "sham/frames/latest.png",
        "sham/manifest.json",
        "sham/optimizer.jsonl",
        "sham/status.json",
    }
    source = "a" * 40
    tag = "stage-a-r1-tag"
    tag_object = "b" * 40
    tag_payload_sha256 = _digest("9")
    first_rollout_sha256 = _digest("7")
    initial_rng_sha256 = _digest("8")
    prior_attempt = {
        "attempt_id": "v0.3-action-effect-stage-a-20260724-attempt-0",
        "disposition": "operationally_incomplete",
        "cohort": {
            "classification": "pre_arm_dashboard_health_failure",
            "recorded_child_actions": 0,
        },
        "resume_authorized": False,
        "reuse_authorized": False,
    }
    prior_attempt_sha256 = qualify._canonical_sha256(prior_attempt)

    claim = {
        "schema_version": 1,
        "protocol": qualify.PROTOCOL,
        "kind": "qualification_attempt_claim",
        "source_commit": source,
        "tag": tag,
        "tag_object": tag_object,
    }
    claim_sha256 = _write_json(qualification / "claim.json", claim)
    report = {
        "schema_version": 1,
        "protocol": qualify.PROTOCOL,
        "kind": qualify.KIND,
        "verdict": "qualified",
        "claim": claim,
        "source": {
            "commit": source,
            "dirty": False,
            "tag": tag,
            "tag_object": tag_object,
            "tag_payload_sha256": tag_payload_sha256,
        },
        "failed_stage_a_attempt": prior_attempt,
    }
    report_sha256 = _write_json(qualification / "report.json", report)
    checksum = qualification / "report.json.sha256"
    checksum.write_text(
        f"{report_sha256}  report.json\n",
        encoding="ascii",
    )
    contract = {
        "schema_version": 1,
        "protocol": qualify.PROTOCOL,
        "cohort_id": "v0.3-action-effect-stage-a-r1-20260724",
        "source": {"commit": source, "dirty": False},
        "preregistration": {
            "tag": tag,
            "tag_object": tag_object,
            "peeled_commit": source,
        },
        "qualification": {
            "report_sha256": report_sha256,
            "tag_payload_sha256": tag_payload_sha256,
            "failed_stage_a_attempt_sha256": prior_attempt_sha256,
        },
        "roots": {"cohort": str(root), "media": str(media)},
        "replacement": {
            "failed_attempt_evidence_sha256": prior_attempt_sha256,
            "failed_attempt_resume_authorized": False,
            "failed_attempt_root_reuse_authorized": False,
            "restarts_both_arms_from_confirmed_u1": True,
        },
    }
    contract_sha256 = _write_json(root / "cohort-contract.json", contract)
    cohort = {
        "schema_version": 1,
        "protocol": qualify.PROTOCOL,
        "cohort_id": "v0.3-action-effect-stage-a-r1-20260724",
        "contract_sha256": contract_sha256,
        "source_commit": source,
        "tag": tag,
        "tag_object": tag_object,
        "qualification_sha256": report_sha256,
        "phase": "operationally_incomplete",
        "active_arm": None,
        "terminal_report": None,
        "process_closeout": None,
        "arms": [
            {
                "id": "sham",
                "state": "crashed",
                "attempts": [
                    {
                        "index": 0,
                        "state": "crashed",
                        "trainer_exit_code": 1,
                    }
                ],
                "terminal": None,
            },
            {
                "id": "action-effect",
                "state": "pending",
                "attempts": [],
                "terminal": None,
            },
        ],
    }
    cohort_sha256 = _write_json(root / "cohort.json", cohort)

    first_exam = root / "sham/checkpoints/rolling/exam-0032768.zip"
    first_exam.write_bytes(b"portable first frozen exam\n")
    first_exam_sha256 = hashlib.sha256(first_exam.read_bytes()).hexdigest()
    first_exam_sidecar = (
        root / "sham/checkpoints/rolling/exam-0032768.json"
    )
    first_exam_sidecar_sha256 = _write_json(
        first_exam_sidecar,
        {
            "protocol": qualify.PROTOCOL,
            "kind": "exam",
            "child_trained_actions": 32_768,
        },
    )
    _write_json(
        root / "sham/checkpoints/rolling/exam-0032768.integrity.json",
        {
            "protocol": qualify.PROTOCOL,
            "checkpoint_sha256": first_exam_sha256,
            "sidecar_sha256": first_exam_sidecar_sha256,
        },
    )

    latest_observed = root / "sham/checkpoints/latest-observed.zip"
    latest_observed.write_bytes(b"portable latest observed checkpoint\n")
    latest_observed_sha256 = hashlib.sha256(
        latest_observed.read_bytes()
    ).hexdigest()
    _write_json(
        root / "sham/first-rollout.json",
        {
            "arm": "sham",
            "captured_before_first_optimizer": True,
            "checkpoint_reuse_authorized": False,
            "identity": {
                "aggregate_sha256": first_rollout_sha256,
                "trajectory_identity": [
                    {"worker_index": index, "transitions": 512}
                    for index in range(4)
                ],
            },
        },
    )
    _write_json(
        root / "sham/status.json",
        {
            "protocol": qualify.PROTOCOL,
            "cohort_id": "v0.3-action-effect-stage-a-r1-20260724",
            "cohort_contract_sha256": contract_sha256,
            "qualification_sha256": report_sha256,
            "arm": "sham",
            "source": {"commit": source, "dirty": False},
            "phase": "training",
            "child_collected_actions": 40_004,
            "child_trained_actions": 38_912,
            "lifetime_collected_actions": 826_436,
            "lifetime_trained_actions": 825_344,
            "optimizer_updates": 1_612,
            "exam_count": 1,
            "exams_completed": 1,
            "initial_rng_identity_sha256": initial_rng_sha256,
            "first_rollout_identity_sha256": first_rollout_sha256,
            "first_rollout_verified": True,
            "latest_observed_checkpoint_sha256": latest_observed_sha256,
            "latest_safe_checkpoint": None,
            "latest_safe_checkpoint_sha256": None,
            "latest_evaluated_checkpoint": {
                "child_trained_actions": 32_768,
                "checkpoint_sha256": first_exam_sha256,
                "counts_toward_architecture_gate": True,
            },
            "report_sha256": None,
            "resume_authorized": False,
            "replacement_requires_both_fresh_arms": True,
        },
    )
    _write_json(
        root / "launcher-sham-attempt-0.json",
        {
            "state": "exited",
            "forwarded_signal": "SIGTERM",
            "exit_status": -15,
            "supervisor_pid": 1234,
            "trainer_pid": 1235,
        },
    )
    (root / "sham/episodes.jsonl").write_text(
        ("{}\n" * 1_919)
        + json.dumps(
            {
                "child_collected_actions": 40_960,
                "child_trained_actions": 38_912,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "sham/evaluations.jsonl").write_text(
        "{}\n" * 8,
        encoding="utf-8",
    )
    (root / "sham/optimizer.jsonl").write_text(
        "{}\n" * 19,
        encoding="utf-8",
    )
    dashboard_log = root / "dashboard.log"
    dashboard_log.write_text(
        "r1 dashboard and trainer remained healthy until launcher rejection\n",
        encoding="utf-8",
    )
    (root / "dashboard.pid").write_text("1233\n", encoding="ascii")

    for relative in sorted(expected_files):
        path = root / relative
        if not path.exists():
            path.write_bytes(f"portable fixture: {relative}\n".encode())
    root_hashes = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in sorted(expected_files)
    }

    launcher_log = tmp_path / "launcher.log"
    launcher_log.write_text(
        "\n".join(
            [
                "v0.3 Stage-A dashboard: http://127.0.0.1:8789/",
                "Starting v0.3 Stage-A arm sham (fresh matched twin).",
                (
                    "v0.3 sham artifacts: /Volumes/T7 Developer/"
                    "DungeonApprentice/v03-action-effect-stage-a-r1-20260724/sham"
                ),
                "v0.3 does not have exactly one neural trainer",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert launcher_log.stat().st_size == 250
    launcher_log_sha256 = hashlib.sha256(launcher_log.read_bytes()).hexdigest()
    artifact_map_sha256 = qualify._canonical_sha256(
        {
            "cohort": root_hashes,
            "qualification": {
                "report.json": report_sha256,
                "claim.json": claim_sha256,
                "report.json.sha256": hashlib.sha256(checksum.read_bytes()).hexdigest(),
            },
            "launcher_log": {str(launcher_log): launcher_log_sha256},
        }
    )

    replacements = {
        "FAILED_STAGE_A_ATTEMPT_ROOT": root,
        "FAILED_STAGE_A_ATTEMPT_MEDIA_ROOT": media,
        "FAILED_STAGE_A_ATTEMPT_QUALIFICATION_DIRECTORY": qualification,
        "FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG": launcher_log,
        "FAILED_STAGE_A_ATTEMPT_TAG": tag,
        "FAILED_STAGE_A_ATTEMPT_TAG_OBJECT": tag_object,
        "FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT": source,
        "FAILED_STAGE_A_ATTEMPT_QUALIFICATION_REPORT_SHA256": (
            report_sha256
        ),
        "FAILED_STAGE_A_ATTEMPT_QUALIFICATION_CLAIM_SHA256": (
            claim_sha256
        ),
        "FAILED_STAGE_A_ATTEMPT_QUALIFICATION_CHECKSUM_SHA256": (
            hashlib.sha256(checksum.read_bytes()).hexdigest()
        ),
        "FAILED_STAGE_A_ATTEMPT_CONTRACT_SHA256": contract_sha256,
        "FAILED_STAGE_A_ATTEMPT_COHORT_SHA256": cohort_sha256,
        "FAILED_STAGE_A_ATTEMPT_PREDECESSOR_SHA256": (
            prior_attempt_sha256
        ),
        "FAILED_STAGE_A_ATTEMPT_TAG_PAYLOAD_SHA256": (
            tag_payload_sha256
        ),
        "FAILED_STAGE_A_ATTEMPT_FIRST_ROLLOUT_SHA256": (
            first_rollout_sha256
        ),
        "FAILED_STAGE_A_ATTEMPT_INITIAL_RNG_SHA256": initial_rng_sha256,
        "FAILED_STAGE_A_ATTEMPT_LATEST_OBSERVED_SHA256": (
            latest_observed_sha256
        ),
        "FAILED_STAGE_A_ATTEMPT_FIRST_EXAM_SHA256": first_exam_sha256,
        "FAILED_STAGE_A_ATTEMPT_ARTIFACT_MAP_SHA256": artifact_map_sha256,
        "FAILED_STAGE_A_ATTEMPT_ROOT_FILE_SHA256": root_hashes,
        "FAILED_STAGE_A_ATTEMPT_ROOT_DIRECTORIES": expected_directories,
        "FAILED_STAGE_A_ATTEMPT_DASHBOARD_LOG_SHA256": (
            hashlib.sha256(dashboard_log.read_bytes()).hexdigest()
        ),
        "FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG_SHA256": launcher_log_sha256,
    }
    for name, value in replacements.items():
        monkeypatch.setattr(qualify, name, value)

    evidence = qualify.authenticate_failed_stage_a_attempt()

    assert evidence["disposition"] == "operationally_incomplete"
    assert evidence["cohort"]["classification"] == (
        "launcher_process_cardinality_false_positive"
    )
    assert (
        evidence["qualification"]["predecessor_attempt_sha256"]
        == prior_attempt_sha256
    )
    training = evidence["recorded_training_evidence"]
    assert training["recorded_child_actions"] == 38_912
    assert training["status_collected_actions"] == 40_004
    assert training["episode_ledger_collected_actions"] == 40_960
    assert training["optimizer_updates_total"] == 1_612
    assert training["optimizer_updates_inherited"] == 1_536
    assert training["optimizer_updates_new"] == 76
    assert training["rollout_boundaries"] == 19
    assert training["frozen_exams"] == 1
    assert training["evaluation_rows"] == 8
    assert training["evaluation_cases"] == 640
    assert training["trainer_started"] is True
    assert evidence["cohort"]["manifest_arm_outcome"]["exit_code"] == 1
    assert evidence["operational_failure"]["worker_stop"] == {
        "supervisor_forwarded_signal": "SIGTERM",
        "worker_exit_status": -15,
        "status_remained_phase": "training",
    }
    assert evidence["launcher_log"]["bytes"] == 250
    assert evidence["resume_authorized"] is False
    assert evidence["reuse_authorized"] is False

    (root / "sham/unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(
        qualify.ActionEffectQualificationError,
        match="inventory changed",
    ):
        qualify.authenticate_failed_stage_a_attempt()


def test_compact_tree_seal_binds_files_directories_and_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sealed"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "alpha.bin").write_bytes(b"alpha")
    (nested / "beta.bin").write_bytes(b"beta")

    file_map, seal = qualify._sealed_tree_inventory(root, "fixture")

    assert file_map == {
        "alpha.bin": hashlib.sha256(b"alpha").hexdigest(),
        "nested/beta.bin": hashlib.sha256(b"beta").hexdigest(),
    }
    assert seal == {
        "regular_file_count": 2,
        "regular_file_bytes": 9,
        "directory_count_including_root": 2,
        "directory_list_sha256": qualify._canonical_sha256([".", "nested"]),
        "regular_file_map_sha256": qualify._canonical_sha256(file_map),
    }

    (nested / "beta.bin").write_bytes(b"changed")
    changed_map, changed_seal = qualify._sealed_tree_inventory(root, "fixture")
    assert changed_map != file_map
    assert changed_seal != seal


def test_r2_failure_contract_has_compact_complete_seals() -> None:
    seals = qualify.FAILED_STAGE_A_R2_ATTEMPT_TREE_SEALS
    critical = qualify.FAILED_STAGE_A_R2_ATTEMPT_CRITICAL_SHA256

    assert sum(
        int(seal["regular_file_count"]) for seal in seals.values()
    ) == qualify.FAILED_STAGE_A_R2_ATTEMPT_COMBINED_FILE_COUNT
    assert sum(
        int(seal["regular_file_bytes"]) for seal in seals.values()
    ) == qualify.FAILED_STAGE_A_R2_ATTEMPT_COMBINED_FILE_BYTES
    assert {
        "cohort/cohort-contract.json",
        "cohort/cohort.json",
        "cohort/sham/status.json",
        "cohort/sham/first-rollout.json",
        "cohort/sham/report.json",
        "cohort/sham/report.integrity.json",
        "cohort/sham/checkpoints/terminal.zip",
        "qualification/report.json",
        "qualification/report.json.sha256",
        "launch-recovery/v03-action-effect-stage-a-r2-20260724-screen.log",
    } <= set(critical)
    assert (
        qualify.FAILED_STAGE_A_R2_ATTEMPT_FIRST_ROLLOUT_SHA256
        != qualify.FAILED_STAGE_A_R2_ATTEMPT_FIRST_ROLLOUT_NO_LF_SHA256
    )
    assert set(qualify.TAG_FIELDS) >= {
        "failed_stage_a_attempt",
        "failed_stage_a_r2_attempt",
    }
    assert set(qualify._REPORT_FIELDS) >= {
        "failed_stage_a_attempt",
        "failed_stage_a_r2_attempt",
    }


def test_terminal_u2s_authentication_requires_no_selection_or_reuse(
    tmp_path: Path,
) -> None:
    report, report_sha256, integrity_sha256 = _terminal_fixture(tmp_path)

    evidence = qualify.authenticate_u2s_terminal(
        root=tmp_path,
        expected_report_sha256=report_sha256,
        expected_integrity_sha256=integrity_sha256,
    )

    assert evidence["verdict"] == "ablation_failed"
    assert evidence["selected_configuration"] is None
    assert evidence["successor_cohort_authorized"] is False
    assert evidence["checkpoint_reuse_authorized"] is False
    assert set(evidence["arms"]) == {
        arm.value for arm in v02_u2s.ARM_PRIORITY
    }

    report["selection"]["successor_cohort_authorized"] = True
    report_sha256 = _write_json(tmp_path / "report.json", report)
    integrity = json.loads((tmp_path / "report.integrity.json").read_text())
    integrity["report_sha256"] = report_sha256
    integrity_sha256 = _write_json(
        tmp_path / "report.integrity.json",
        integrity,
    )
    with pytest.raises(
        qualify.ActionEffectQualificationError,
        match="no-selection",
    ):
        qualify.authenticate_u2s_terminal(
            root=tmp_path,
            expected_report_sha256=report_sha256,
            expected_integrity_sha256=integrity_sha256,
        )


def test_stage_a_guard_extends_history_but_keeps_finite_u0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, _report_sha256, _integrity_sha256 = _terminal_fixture(tmp_path)
    monkeypatch.setattr(
        qualify,
        "U2S_TERMINAL_REPORT_SHA256",
        _digest("f"),
    )
    base = {
        lesson: frozenset({_digest(str(index + 5))})
        for index, lesson in enumerate(lessons.LessonId)
    }

    mapping, evidence = (
        qualify.extend_action_effect_forbidden_layout_hashes(
            base,
            report,
            root=tmp_path,
        )
    )

    assert mapping[lessons.LessonId.VISIBLE_UNLOCK] == base[
        lessons.LessonId.VISIBLE_UNLOCK
    ]
    assert len(mapping[lessons.LessonId.NAVIGATE]) > 1
    assert len(mapping[lessons.LessonId.LOCAL_UNLOCK]) > 1
    assert len(mapping[lessons.LessonId.SEPARATED_UNLOCK]) > 1
    assert evidence["frozen_before_sham"] is True
    assert evidence["sham_history_added_before_action_effect"] is False
    assert evidence["guard_sets"][
        lessons.LessonId.VISIBLE_UNLOCK.value
    ]["rule"] == "development_only_history_overlap_diagnostic"


def _smoke_report(
    source_commit: str,
    guard_sha256: str = "8" * 64,
) -> dict[str, object]:
    before = {
        "trained_timesteps": v03.PARENT_LIFETIME_ACTIONS,
        "optimizer_updates": v03.PARENT_OPTIMIZER_UPDATES,
        "policy_tensor_sha256": _digest("1"),
        "optimizer_state_sha256": _digest("2"),
    }
    def simple_space(kind: str) -> dict[str, object]:
        return {"type": kind, "self_sha256": _digest("3")}

    observation_space = {
        "type": "Dict",
        "self_sha256": _digest("4"),
        "children": {
            "image": simple_space("Box"),
            "action_effect": simple_space("Box"),
        },
    }
    rng_workers = [
        {
            "worker_index": index,
            "worker_stream": v03.WORKER_STREAMS[index],
            "curriculum_sha256": _digest("5"),
            "base_environment_sha256": _digest("6"),
            "action_space": simple_space("Discrete"),
            "observation_space": observation_space,
        }
        for index in range(v03.WORKERS)
    ]
    rng_components = {
        "python_random_sha256": _digest("3"),
        "numpy_global_sha256": _digest("4"),
        "torch_cpu_sha256": _digest("5"),
        "model_action_space": simple_space("Discrete"),
        "scheduler_sha256": _digest("6"),
        "workers": rng_workers,
    }

    def rng(phase: str) -> dict[str, object]:
        return {
            "phase": phase,
            "components": rng_components,
            "aggregate_sha256": qualify._canonical_sha256(rng_components),
        }

    workers = [
        {
            "worker_index": index,
            "transitions": v03.ROLLOUT_STEPS,
            "episodes_started": 1,
            "episode_starts_sha256": _digest("4"),
            "action_counts": {
                str(action): v03.ROLLOUT_STEPS if action == 0 else 0
                for action in range(7)
            },
            "lesson_transition_counts": {
                lesson.value: (
                    v03.ROLLOUT_STEPS
                    if lesson is lessons.LessonId.NAVIGATE
                    else 0
                )
                for lesson in lessons.LessonId
            },
            "reward_total_hex": "0x0.0p+0",
            "extrinsic_total_hex": "0x0.0p+0",
            "curiosity_total_hex": "0x0.0p+0",
            "penalty_total_hex": "0x0.0p+0",
            "penalty_events": 0,
            "penalty_eligible_events": 0,
            "maximum_episode_penalty_count": 0,
            "trajectory_sha256": _digest("5"),
            "reward_evidence_sha256": _digest("6"),
        }
        for index in range(v03.WORKERS)
    ]
    trajectory = smoke._trajectory_identity(workers)
    seed_evidence = {
        "episode_starts": v03.WORKERS,
        "minimum_seed": 1,
        "maximum_seed": 4,
        "training_range_only": True,
        "separated_unlock_roles": [],
        "protected_roles": sorted(
            role.value for role in smoke.frozen_smoke._PROTECTED_ROLES
        ),
        "protected_seed_hits": [],
        "confirmation_or_final_seed_generated": False,
    }
    sampler = [
        {
            "lesson_id": lesson.value,
            "worker_stream": stream,
            "accepted_seed": index + 1,
            "accepted_attempt": 1,
            "layout_sha256": _digest("7"),
            "forbidden_layouts": 1,
            "max_attempts": 8_192,
        }
        for index, (lesson, stream) in enumerate(
            (lesson, stream)
            for lesson in lessons.LessonId
            for stream in v03.WORKER_STREAMS
        )
    ]
    sampler_sha256 = qualify._canonical_sha256(sampler)
    transplant = {
        "architecture_version": v03.ARCHITECTURE_VERSION,
        "transplant": {
            "missing_optimizer_state_names": [],
            "effect_encoder_zero": True,
        },
        "zero_context_equivalence": {
            "batch": 4,
            "features_exact": True,
            "actions_exact": True,
            "values_exact": True,
            "log_probabilities_exact": True,
            "recurrent_states_exact": True,
        },
        "policy_contract": v03.public_policy_contract(),
    }

    def arm(name: str, nonzero: int) -> dict[str, object]:
        return {
            "arm": name,
            "before": before,
            "after": {
                "trained_timesteps": (
                    v03.PARENT_LIFETIME_ACTIONS
                    + v03.ROLLOUT_TRANSITIONS
                ),
                "optimizer_updates": (
                    v03.PARENT_OPTIMIZER_UPDATES + v03.PPO_EPOCHS
                ),
                "policy_tensor_sha256": _digest("a"),
                "optimizer_state_sha256": _digest("b"),
            },
            "effect_projection_nonzero_parameters": nonzero,
            "transplant": transplant,
            "workers": workers,
            "trajectory_identity": trajectory,
            "pre_action_rng_identity": rng("post_reset_pre_action_one"),
            "policy_output_sha256": _digest("c"),
            "post_rollout_rng_identity": rng(
                "post_rollout_pre_optimizer"
            ),
            "episode_ledger": {
                "records": v03.WORKERS,
                "normalized_sha256": _digest("e"),
            },
            "seed_evidence": seed_evidence,
            "updated_archive_reloaded_exactly": True,
            "development_checkpoint_reuse_authorized": False,
        }

    return {
        "schema_version": smoke.SCHEMA_VERSION,
        "protocol": smoke.SMOKE_PROTOCOL,
        "completed_at": "2026-07-24T00:00:00+00:00",
        "verdict": "passed",
        "source": {
            "commit": source_commit,
            "branch": "test-v03",
            "dirty": False,
        },
        "parent_checkpoint": str(v03.PARENT_CHECKPOINT),
        "parent_checkpoint_sha256": v03.PARENT_CHECKPOINT_SHA256,
        "worker_streams": list(v03.WORKER_STREAMS),
        "guard_mapping_sha256": guard_sha256,
        "sampler_preflight": sampler,
        "sampler_preflight_sha256": sampler_sha256,
        "actions_per_arm": v03.ROLLOUT_TRANSITIONS,
        "pre_action_rng_identical": True,
        "pre_update_behavior_identical": True,
        "first_rollout_trajectory_identical": True,
        "first_rollout_policy_outputs_identical": True,
        "post_rollout_pre_optimizer_rng_identical": True,
        "first_rollout_episode_ledger_identical": True,
        "learning_divergence_begins_after_first_update": True,
        "arms": {
            ActionEffectMode.SHAM.value: arm(
                ActionEffectMode.SHAM.value,
                0,
            ),
            ActionEffectMode.ACTION_EFFECT.value: arm(
                ActionEffectMode.ACTION_EFFECT.value,
                7,
            ),
        },
        "canonical_predecessor_roots_unchanged": True,
        "temporary_root_removed": True,
        "scientific_evidence": False,
        "checkpoint_reuse_authorized": False,
    }


def test_disposable_smoke_binds_first_rollout_and_rng_identity() -> None:
    source_commit = "a" * 40
    guard_sha256 = _digest("8")
    report = _smoke_report(source_commit)
    sampler_sha256 = str(report["sampler_preflight_sha256"])

    evidence = qualify.validate_disposable_smoke(
        report,
        source_commit=source_commit,
        guard_mapping_sha256=guard_sha256,
        sampler_preflight_sha256=sampler_sha256,
    )

    assert evidence["matched_through_first_rollout"] is True
    assert evidence["candidate_context_projection_updated"] is True
    assert evidence["sham_context_projection_remained_zero"] is True
    assert (
        evidence["first_rollout_digest_profile"]
        == v03.FIRST_ROLLOUT_DIGEST_PROFILE
    )
    assert (
        evidence["first_rollout_policy_output_sha256"]
        == _digest("c")
    )
    assert len(evidence["pre_action_rng_identity_sha256"]) == 64


def test_disposable_smoke_rejects_preupdate_divergence() -> None:
    source_commit = "a" * 40
    guard_sha256 = _digest("8")
    report = _smoke_report(source_commit)
    sampler_sha256 = str(report["sampler_preflight_sha256"])
    report["arms"][ActionEffectMode.ACTION_EFFECT.value][
        "policy_output_sha256"
    ] = _digest("0")

    with pytest.raises(
        qualify.ActionEffectQualificationError,
        match="diverged",
    ):
        qualify.validate_disposable_smoke(
            report,
            source_commit=source_commit,
            guard_mapping_sha256=guard_sha256,
            sampler_preflight_sha256=sampler_sha256,
        )


def test_disposable_smoke_rejects_guard_mapping_drift() -> None:
    source_commit = "a" * 40
    report = _smoke_report(source_commit)
    sampler_sha256 = str(report["sampler_preflight_sha256"])

    with pytest.raises(
        qualify.ActionEffectQualificationError,
        match="contract changed",
    ):
        qualify.validate_disposable_smoke(
            report,
            source_commit=source_commit,
            guard_mapping_sha256=_digest("7"),
            sampler_preflight_sha256=sampler_sha256,
        )


def test_disposable_smoke_rejects_sampler_transcript_drift() -> None:
    source_commit = "a" * 40
    report = _smoke_report(source_commit)
    sampler_sha256 = str(report["sampler_preflight_sha256"])
    report["sampler_preflight"].pop()

    with pytest.raises(
        qualify.ActionEffectQualificationError,
        match="sampler preflight",
    ):
        qualify.validate_disposable_smoke(
            report,
            source_commit=source_commit,
            guard_mapping_sha256=_digest("8"),
            sampler_preflight_sha256=sampler_sha256,
        )


def test_disposable_smoke_rejects_incomplete_worker_evidence() -> None:
    source_commit = "a" * 40
    report = _smoke_report(source_commit)
    sampler_sha256 = str(report["sampler_preflight_sha256"])
    report["arms"][ActionEffectMode.SHAM.value]["workers"].pop()

    with pytest.raises(
        qualify.ActionEffectQualificationError,
        match=r"diverged|workers",
    ):
        qualify.validate_disposable_smoke(
            report,
            source_commit=source_commit,
            guard_mapping_sha256=_digest("8"),
            sampler_preflight_sha256=sampler_sha256,
        )


def test_disposable_smoke_rejects_incomplete_rng_evidence() -> None:
    source_commit = "a" * 40
    report = _smoke_report(source_commit)
    sampler_sha256 = str(report["sampler_preflight_sha256"])
    rng = report["arms"][ActionEffectMode.SHAM.value][
        "pre_action_rng_identity"
    ]
    rng["components"].pop("workers")
    rng["aggregate_sha256"] = qualify._canonical_sha256(rng["components"])

    with pytest.raises(
        qualify.ActionEffectQualificationError,
        match=r"diverged|components",
    ):
        qualify.validate_disposable_smoke(
            report,
            source_commit=source_commit,
            guard_mapping_sha256=_digest("8"),
            sampler_preflight_sha256=sampler_sha256,
        )


def test_disposable_smoke_rejects_protected_seed_evidence() -> None:
    source_commit = "a" * 40
    report = _smoke_report(source_commit)
    sampler_sha256 = str(report["sampler_preflight_sha256"])
    seed_evidence = report["arms"][ActionEffectMode.SHAM.value][
        "seed_evidence"
    ]
    seed_evidence["training_range_only"] = False
    seed_evidence["protected_seed_hits"] = [
        {"seed": 20_000_000, "role": "final_test"}
    ]
    seed_evidence["confirmation_or_final_seed_generated"] = True

    with pytest.raises(
        qualify.ActionEffectQualificationError,
        match=r"diverged|seed evidence",
    ):
        qualify.validate_disposable_smoke(
            report,
            source_commit=source_commit,
            guard_mapping_sha256=_digest("8"),
            sampler_preflight_sha256=sampler_sha256,
        )


def test_recorded_fresh_storage_remains_verifiable_after_bound_roots_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = {
        "volume": "/Volumes/T7 Developer",
        "dungeon_root": "/Volumes/T7 Developer/DungeonApprentice",
        "separate_mounted_device": True,
        "volume_device": 10,
        "parent_device": 9,
        "managed_roots": ["qualification", "cohort", "media"],
        "cohort_and_media_roots_absent": True,
        "bound_managed_roots_permitted": False,
        "qualification_root_absent_before_claim": True,
        "protected_artifacts_outside_managed_roots": True,
        "minimum_free_bytes": 25 * 1024**3,
        "measured_free_bytes": 100 * 1024**3,
        "measured_total_bytes": 1_000 * 1024**3,
        "passed": True,
        "measured_at": datetime.now(UTC).isoformat(),
    }
    live = {
        **recorded,
        "cohort_and_media_roots_absent": False,
        "bound_managed_roots_permitted": True,
        "qualification_root_absent_before_claim": False,
    }
    calls: list[dict[str, bool]] = []

    def storage(**kwargs: bool) -> dict[str, object]:
        calls.append(kwargs)
        return live

    monkeypatch.setattr(qualify, "_storage_preflight", storage)

    qualify._verify_storage_preflight(recorded)

    assert calls == [
        {
            "require_qualification_absent": False,
            "permit_bound_managed_roots": True,
        }
    ]


def test_source_tag_requires_clean_published_annotated_tag(
    tmp_path: Path,
) -> None:
    head = "a" * 40
    tag_object = "b" * 40
    payload = {field: None for field in qualify.TAG_FIELDS}
    payload.update(
        {
            "schema_version": qualify.TAG_SCHEMA_VERSION,
            "kind": qualify.TAG_KIND,
            "protocol": qualify.PROTOCOL,
            "tag": qualify.QUALIFIED_TAG,
            "source_commit": head,
        }
    )
    tag_message = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def runner(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        arguments = command[1:]
        ref = f"refs/tags/{qualify.QUALIFIED_TAG}"
        outputs = {
            ("rev-parse", "--verify", "HEAD^{commit}"): head,
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
            ("cat-file", "-t", ref): "tag",
            ("rev-parse", "--verify", ref): tag_object,
            ("rev-list", "-n", "1", ref): head,
            ("remote", "get-url", qualify.QUALIFIED_REMOTE): (
                qualify.EXPECTED_ORIGIN_URL
            ),
            (
                "ls-remote",
                "--tags",
                qualify.QUALIFIED_REMOTE,
                ref,
            ): f"{tag_object}\t{ref}",
            ("for-each-ref", "--format=%(contents)", ref): tag_message,
        }
        output = outputs[tuple(arguments)]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=output + "\n",
            stderr="",
        )

    source = qualify.verify_source_tag(
        tmp_path,
        expected_source_commit=head,
        expected_tag_object=tag_object,
        expected_tag_payload=payload,
        runner=runner,
    )

    assert source["commit"] == head
    assert source["tag_object"] == tag_object
    assert source["dirty"] is False
    assert source["tag_payload_sha256"] == qualify._canonical_sha256(
        payload
    )


def test_failed_attempt_tag_remains_exact_on_origin(tmp_path: Path) -> None:
    reference = f"refs/tags/{qualify.FAILED_STAGE_A_ATTEMPT_TAG}"

    def runner(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        arguments = command[1:]
        outputs = {
            ("cat-file", "-t", reference): "tag",
            ("rev-parse", "--verify", reference): (
                qualify.FAILED_STAGE_A_ATTEMPT_TAG_OBJECT
            ),
            ("rev-list", "-n", "1", reference): (
                qualify.FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT
            ),
            ("remote", "get-url", qualify.QUALIFIED_REMOTE): (
                qualify.EXPECTED_ORIGIN_URL
            ),
            (
                "ls-remote",
                "--tags",
                qualify.QUALIFIED_REMOTE,
                reference,
            ): (
                f"{qualify.FAILED_STAGE_A_ATTEMPT_TAG_OBJECT}"
                f"\t{reference}"
            ),
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=outputs[tuple(arguments)] + "\n",
            stderr="",
        )

    qualify._verify_failed_stage_a_tag(tmp_path, runner=runner)


def test_tag_payload_binds_runtime_and_cpu_device_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen protocol\n", encoding="utf-8")
    monkeypatch.setattr(qualify, "PROTOCOL_DOCUMENT", Path("protocol.md"))
    parent = {
        key: _digest(str(index % 10))
        for index, key in enumerate(
            (
                "checkpoint_sha256",
                "sidecar_sha256",
                "manifest_sha256",
                "policy_tensor_sha256",
                "optimizer_state_sha256",
                "confirmation_sha256",
            ),
            start=1,
        )
    }
    parent.update({"trained_timesteps": 786_432, "n_updates": 1_536})
    terminal = {
        "report_sha256": _digest("7"),
        "report_integrity_sha256": _digest("8"),
        "verdict": "ablation_failed",
        "selected_configuration": None,
        "successor_cohort_authorized": False,
        "checkpoint_reuse_authorized": False,
    }
    monkeypatch.setattr(
        qualify,
        "_collect_static_inputs",
        lambda _repository: (
            parent,
            None,
            {},
            {"applied_mapping_sha256": _digest("9")},
            {"u2s_terminal": terminal},
            [],
        ),
    )
    monkeypatch.setattr(qualify, "_protected_seed_partitions", lambda: [])
    monkeypatch.setattr(qualify, "_architecture_contract", lambda: {})
    failed_attempt = {
        "disposition": "operationally_incomplete",
        "resume_authorized": False,
        "reuse_authorized": False,
    }
    monkeypatch.setattr(
        qualify,
        "authenticate_failed_stage_a_attempt",
        lambda **_kwargs: failed_attempt,
    )
    failed_r2_attempt = {
        "disposition": "integrity_failed",
        "resume_authorized": False,
        "reuse_authorized": False,
    }
    monkeypatch.setattr(
        qualify,
        "authenticate_failed_stage_a_r2_attempt",
        lambda **_kwargs: failed_r2_attempt,
    )
    snapshot = {
        "python": "3.12-test",
        "platform": "macOS-test",
        "machine": "arm64",
        "packages": {"torch": "test"},
    }
    monkeypatch.setattr(qualify, "runtime_snapshot", lambda: snapshot)

    payload = qualify.build_action_effect_tag_payload(
        tmp_path,
        source_commit="a" * 40,
    )

    assert payload["runtime_contract"] == {
        "snapshot": snapshot,
        "training_device": "cpu",
        "qualification_smoke_device": "cpu",
    }
    assert "runtime_contract" in qualify.TAG_FIELDS
    assert payload["failed_stage_a_attempt"] == failed_attempt
    assert payload["failed_stage_a_r2_attempt"] == failed_r2_attempt
    assert payload["resume_rule"]["failed_attempts"] == [
        "v0.3-action-effect-stage-a-r1-20260724-attempt-0",
        "v0.3-action-effect-stage-a-r2-20260724-attempt-0",
    ]
    assert payload["resume_rule"]["replacement_attempt"] == (
        "v0.3-action-effect-stage-a-r3-20260724"
    )


def test_architecture_and_protected_partition_contracts_are_exact() -> None:
    contract = qualify._architecture_contract()
    protected = qualify._protected_seed_partitions()

    assert contract["arms"] == ["sham", "action-effect"]
    assert contract["transplant"]["sole_new_parameter_shape"] == [512, 9]
    assert contract["transplant"]["sole_new_parameter_initialization"] == (
        "exact_zero"
    )
    assert contract["equivalence"]["first_rollout_digest_profile"] == (
        v03.FIRST_ROLLOUT_DIGEST_PROFILE
    )
    assert contract["randomness"]["worker_streams"] == [
        20260757,
        20260758,
        20260759,
        20260760,
    ]
    assert contract["selection"]["development_checkpoint_reuse_authorized"] is False
    assert len(protected) == 10
    assert any(
        value["role"] == "sealed_qualification" for value in protected
    )
    assert any(value["role"] == "final_test" for value in protected)


class _BaseQualification:
    def __init__(self, token: object) -> None:
        self.token = token

    def seed_access(self) -> object:
        return self.token


def test_evidence_exposes_verified_report_seed_access_and_copy_of_guards() -> None:
    token = object()
    report = {"protocol": qualify.PROTOCOL, "verdict": "qualified"}
    report_bytes = qualify._canonical_json_bytes(report)
    mapping = {
        lesson: frozenset({_digest(str(index + 1))})
        for index, lesson in enumerate(lessons.LessonId)
    }
    evidence = qualify.ActionEffectQualificationEvidence(
        report="/report.json",
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        checksum="/report.json.sha256",
        source_commit="a" * 40,
        tag=qualify.QUALIFIED_TAG,
        tag_object="b" * 40,
        tag_payload_sha256=_digest("c"),
        verdict="qualified",
        protocol_document_sha256=_digest("d"),
        guard_mapping_sha256=_digest("e"),
        sampler_preflight_sha256=_digest("f"),
        architecture_contract_sha256=_digest("0"),
        smoke_evidence_sha256=_digest("1"),
        protected_partitions_sha256=_digest("2"),
        failed_stage_a_attempt_sha256=_digest("3"),
        failed_stage_a_r2_attempt_sha256=_digest("4"),
        storage_caps=MappingProxyType(qualify._storage_caps()),
        _report_bytes=report_bytes,
        _base_qualification=_BaseQualification(token),
        _forbidden_layouts=MappingProxyType(mapping),
    )

    assert evidence.verified_report() == report
    assert evidence.seed_access() is token
    assert evidence.forbidden_layout_hashes() == mapping
    assert isinstance(evidence.forbidden_layout_hashes(), MappingProxyType)
    assert evidence.public_dict()["failed_stage_a_r2_attempt_sha256"] == _digest("4")
