from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import pytest

from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v04_ineffective_trace as v04
from dungeon_apprentice import v04_ineffective_trace_qualify as qualify
from dungeon_apprentice import v04_ineffective_trace_smoke as smoke
from dungeon_apprentice.action_streak import IneffectiveTraceMode


def _digest(character: str) -> str:
    return character * 64


def _write(path: Path, value: object) -> str:
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rng_identity(phase: str) -> dict:
    workers = [
        {
            "worker_index": index,
            "worker_stream": stream,
            "curriculum_sha256": _digest("1"),
            "base_environment_sha256": _digest("2"),
            "action_space": {"type": "Discrete", "self_sha256": _digest("3")},
            "observation_space": {
                "type": "Dict",
                "self_sha256": _digest("4"),
                "children": {},
            },
        }
        for index, stream in enumerate(v04.WORKER_STREAMS)
    ]
    components = {
        "python_random_sha256": _digest("5"),
        "numpy_global_sha256": _digest("6"),
        "torch_cpu_sha256": _digest("7"),
        "model_action_space": {
            "type": "Discrete",
            "self_sha256": _digest("8"),
        },
        "scheduler_sha256": _digest("9"),
        "workers": workers,
    }
    return {
        "phase": phase,
        "components": components,
        "aggregate_sha256": qualify._canonical_sha256(components),
    }


def _worker(index: int) -> dict:
    actions = {str(action): 0 for action in range(7)}
    actions["0"] = v04.ROLLOUT_STEPS
    transitions = {lesson.value: 0 for lesson in lessons.LessonId}
    transitions[lessons.LessonId.NAVIGATE.value] = v04.ROLLOUT_STEPS
    return {
        "worker_index": index,
        "transitions": v04.ROLLOUT_STEPS,
        "episodes_started": 1,
        "episode_starts_sha256": _digest(str(index + 1)),
        "action_counts": actions,
        "lesson_transition_counts": transitions,
        "reward_total_hex": "0x0.0p+0",
        "extrinsic_total_hex": "0x0.0p+0",
        "curiosity_total_hex": "0x0.0p+0",
        "penalty_total_hex": "0x0.0p+0",
        "penalty_events": 0,
        "penalty_eligible_events": 0,
        "maximum_episode_penalty_count": 0,
        "trajectory_sha256": _digest(chr(ord("a") + index)),
        "reward_evidence_sha256": _digest(str(5 + index)),
    }


def _fingerprint(
    *,
    timesteps: int,
    updates: int,
    policy: str,
    optimizer: str,
) -> dict:
    return {
        "trained_timesteps": timesteps,
        "optimizer_updates": updates,
        "policy_tensor_sha256": _digest(policy),
        "optimizer_state_sha256": _digest(optimizer),
        "optimizer_has_state": True,
    }


def _smoke_arm(mode: IneffectiveTraceMode) -> dict:
    workers = [_worker(index) for index in range(v04.WORKERS)]
    enabled = mode is IneffectiveTraceMode.STREAK_TRACE
    trace = {
        "transitions": v04.ROLLOUT_TRANSITIONS,
        "visible_changed": 1_800,
        "visible_unchanged": 248,
        "raw_trace_nonzero_transitions": 248,
        "raw_trace_max": 9,
        "raw_trace_histogram": {"0": 1_800, "1": 200, "2": 48},
        "exposed_trace_nonzero_transitions": 248 if enabled else 0,
        "exposed_trace_max": 1.0 if enabled else 0.0,
        "trace_enabled": enabled,
        "descriptive_only": True,
    }
    update = {
        "weight_nonzero_parameters": 512 if enabled else 0,
        "gradient_observations": 32,
        "gradient_nonzero_observations": 32 if enabled else 0,
        "gradient_nonzero_parameters_max": 512 if enabled else 0,
        "adam_state_present": True,
        "adam_exp_avg_nonzero_parameters": 512 if enabled else 0,
        "adam_exp_avg_sq_nonzero_parameters": 512 if enabled else 0,
        "weight_exact_zero": not enabled,
        "adam_moments_exact_zero": not enabled,
    }
    episode_starts = sum(worker["episodes_started"] for worker in workers)
    seed_evidence = {
        "episode_starts": episode_starts,
        "minimum_seed": 1,
        "maximum_seed": 999,
        "training_range_only": True,
        "separated_unlock_roles": [],
        "protected_roles": sorted(
            role.value for role in smoke.frozen_smoke._PROTECTED_ROLES
        ),
        "protected_seed_hits": [],
        "confirmation_or_final_seed_generated": False,
    }
    return {
        "arm": mode.value,
        "before": _fingerprint(
            timesteps=v04.PARENT_LIFETIME_ACTIONS,
            updates=v04.PARENT_OPTIMIZER_UPDATES,
            policy="a",
            optimizer="b",
        ),
        "after": _fingerprint(
            timesteps=v04.PARENT_LIFETIME_ACTIONS
            + v04.ROLLOUT_TRANSITIONS,
            updates=v04.PARENT_OPTIMIZER_UPDATES + v04.PPO_EPOCHS,
            policy="c" if enabled else "d",
            optimizer="e" if enabled else "f",
        ),
        "trace_projection_nonzero_parameters": (
            update["weight_nonzero_parameters"]
        ),
        "trace_exercise": trace,
        "encoder_update": update,
        "transplant": {
            "architecture_version": v04.ARCHITECTURE_VERSION,
            "transplant": {
                "inherited_parameter_names": list(
                    qualify.EXPECTED_U1_PARAMETER_NAMES
                ),
                "new_parameter_names": [v04.TRACE_ENCODER_PARAMETER],
                "inherited_optimizer_state_names": list(
                    qualify.EXPECTED_U1_PARAMETER_NAMES
                ),
                "trace_encoder_zero": True,
                "missing_optimizer_state_names": [],
                "inherited_timesteps": v04.PARENT_LIFETIME_ACTIONS,
                "inherited_optimizer_updates": v04.PARENT_OPTIMIZER_UPDATES,
            },
            "zero_context_equivalence": {
                "batch": 4,
                "features_exact": True,
                "actions_exact": True,
                "values_exact": True,
                "log_probabilities_exact": True,
                "recurrent_states_exact": True,
            },
            "policy_contract": v04.public_policy_contract(),
        },
        "workers": workers,
        "trajectory_identity": smoke._trajectory_identity(workers),
        "pre_action_rng_identity": _rng_identity(
            "post_reset_pre_action_one"
        ),
        "policy_output_sha256": _digest("0"),
        "post_rollout_rng_identity": _rng_identity(
            "post_rollout_pre_optimizer"
        ),
        "episode_ledger": {
            "records": episode_starts,
            "normalized_sha256": _digest("1"),
        },
        "seed_evidence": seed_evidence,
        "updated_archive_reloaded_exactly": True,
        "development_checkpoint_reuse_authorized": False,
        "updated_archive_destroyed": True,
    }


def _smoke_report(source_commit: str) -> dict:
    sham = _smoke_arm(IneffectiveTraceMode.ZERO_TRACE)
    candidate = _smoke_arm(IneffectiveTraceMode.STREAK_TRACE)
    # The two policies and all pre-update evidence are exact twins.
    for key in (
        "before",
        "workers",
        "trajectory_identity",
        "pre_action_rng_identity",
        "policy_output_sha256",
        "post_rollout_rng_identity",
        "episode_ledger",
        "seed_evidence",
    ):
        candidate[key] = json.loads(json.dumps(sham[key]))
    sampler: list[dict] = []
    return {
        "schema_version": smoke.SCHEMA_VERSION,
        "protocol": smoke.SMOKE_PROTOCOL,
        "completed_at": datetime.now(UTC).isoformat(),
        "verdict": "passed",
        "source": {
            "commit": source_commit,
            "branch": "test",
            "dirty": False,
        },
        "parent_checkpoint": str(v04.PARENT_CHECKPOINT),
        "parent_checkpoint_sha256": v04.PARENT_CHECKPOINT_SHA256,
        "worker_streams": list(v04.WORKER_STREAMS),
        "guard_mapping_sha256": _digest("2"),
        "sampler_preflight": sampler,
        "sampler_preflight_sha256": qualify._canonical_sha256(sampler),
        "actions_per_arm": v04.ROLLOUT_TRANSITIONS,
        "pre_action_rng_identical": True,
        "pre_update_behavior_identical": True,
        "first_rollout_trajectory_identical": True,
        "first_rollout_policy_outputs_identical": True,
        "post_rollout_pre_optimizer_rng_identical": True,
        "first_rollout_episode_ledger_identical": True,
        "learning_divergence_begins_after_first_update": True,
        "raw_trace_evidence_identical": True,
        "arms": {
            IneffectiveTraceMode.ZERO_TRACE.value: sham,
            IneffectiveTraceMode.STREAK_TRACE.value: candidate,
        },
        "canonical_predecessor_roots_unchanged": True,
        "temporary_root_removed": True,
        "updated_archives_reloaded_exactly": True,
        "updated_archives_destroyed": True,
        "scientific_evidence": False,
        "checkpoint_reuse_authorized": False,
        "r3_checkpoint_or_state_reused": False,
    }


def test_architecture_contract_is_one_scalar_and_never_retains_v03_input() -> None:
    contract = qualify._architecture_contract()

    assert contract["arms"] == ["trace-sham", "ineffective-trace"]
    assert contract["ineffective_trace"]["shape"] == [1]
    assert contract["ineffective_trace"]["formula"] == "min(count,9)/9"
    assert contract["ineffective_trace"]["action_identity_exposed"] is False
    assert (
        contract["ineffective_trace"][
            "v03_nine_dimensional_input_retained"
        ]
        is False
    )
    assert contract["transplant"]["sole_new_parameter_shape"] == [512, 1]
    assert contract["randomness"]["worker_streams"] == list(
        v04.WORKER_STREAMS
    )
    assert contract["selection"]["development_checkpoint_reuse_authorized"] is False


def test_disposable_smoke_authenticates_trace_learning_and_exact_zero_sham() -> None:
    source_commit = "a" * 40
    report = _smoke_report(source_commit)
    result = qualify.validate_disposable_smoke(
        report,
        source_commit=source_commit,
        guard_mapping_sha256=_digest("2"),
        sampler_preflight_sha256=qualify._canonical_sha256([]),
    )

    assert result["candidate_trace_exercised"]
    assert result["candidate_encoder_learned_after_optimizer"]
    assert result["sham_encoder_and_adam_exact_zero"]
    assert result["updated_archives_destroyed"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (
            (
                "arms",
                IneffectiveTraceMode.ZERO_TRACE.value,
                "encoder_update",
                "weight_nonzero_parameters",
            ),
            1,
        ),
        (
            (
                "arms",
                IneffectiveTraceMode.STREAK_TRACE.value,
                "trace_exercise",
                "exposed_trace_nonzero_transitions",
            ),
            0,
        ),
        (
            (
                "arms",
                IneffectiveTraceMode.STREAK_TRACE.value,
                "updated_archive_destroyed",
            ),
            False,
        ),
    ],
)
def test_disposable_smoke_rejects_encoder_trace_and_archive_tampering(
    path: tuple[str, ...],
    value: object,
) -> None:
    report = _smoke_report("a" * 40)
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(qualify.IneffectiveTraceQualificationError):
        qualify.validate_disposable_smoke(
            report,
            source_commit="a" * 40,
            guard_mapping_sha256=_digest("2"),
            sampler_preflight_sha256=qualify._canonical_sha256([]),
        )


def test_disposable_smoke_rejects_preupdate_twin_divergence() -> None:
    report = _smoke_report("a" * 40)
    report["arms"]["ineffective-trace"]["policy_output_sha256"] = _digest("f")

    with pytest.raises(
        qualify.IneffectiveTraceQualificationError,
        match="diverged",
    ):
        qualify.validate_disposable_smoke(
            report,
            source_commit="a" * 40,
            guard_mapping_sha256=_digest("2"),
            sampler_preflight_sha256=qualify._canonical_sha256([]),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("log_probabilities_exact", False),
        ("batch", 3),
    ),
)
def test_disposable_smoke_rejects_incomplete_zero_context_equivalence(
    field: str,
    value: object,
) -> None:
    report = _smoke_report("a" * 40)
    report["arms"]["ineffective-trace"]["transplant"][
        "zero_context_equivalence"
    ][field] = value

    with pytest.raises(
        qualify.IneffectiveTraceQualificationError,
        match="transplant evidence",
    ):
        qualify.validate_disposable_smoke(
            report,
            source_commit="a" * 40,
            guard_mapping_sha256=_digest("2"),
            sampler_preflight_sha256=qualify._canonical_sha256([]),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("missing_optimizer_state_names", ["action_net.bias"]),
        ("new_parameter_names", ["wrong.weight"]),
        ("inherited_timesteps", v04.PARENT_LIFETIME_ACTIONS - 1),
    ),
)
def test_disposable_smoke_rejects_inexact_transplant_and_optimizer_coverage(
    field: str,
    value: object,
) -> None:
    report = _smoke_report("a" * 40)
    report["arms"]["ineffective-trace"]["transplant"]["transplant"][
        field
    ] = value

    with pytest.raises(
        qualify.IneffectiveTraceQualificationError,
        match="transplant evidence",
    ):
        qualify.validate_disposable_smoke(
            report,
            source_commit="a" * 40,
            guard_mapping_sha256=_digest("2"),
            sampler_preflight_sha256=qualify._canonical_sha256([]),
        )


def test_complete_tree_inventory_rejects_symlink_and_detects_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "a.json").write_text("one", encoding="utf-8")
    _mapping, before = qualify._sealed_tree_inventory(root, "fixture")
    (root / "a.json").write_text("two", encoding="utf-8")
    _mapping, after = qualify._sealed_tree_inventory(root, "fixture")
    assert before["regular_file_map_sha256"] != after["regular_file_map_sha256"]

    (root / "unsafe").symlink_to(root / "a.json")
    with pytest.raises(
        qualify.IneffectiveTraceQualificationError,
        match="unsafe file",
    ):
        qualify._sealed_tree_inventory(root, "fixture")


def _arm_fixture(root: Path, arm_name: str) -> dict[str, str]:
    arm = root / arm_name
    rolling = arm / "checkpoints" / "rolling"
    rolling.mkdir(parents=True)
    exams = []
    case_files = []
    final_checkpoint_bytes = b""
    for index in range(1, v04.EXAM_COUNT + 1):
        boundary = index * v04.EVALUATION_INTERVAL
        stem = rolling / f"exam-{boundary:07d}"
        checkpoint_bytes = f"checkpoint-{index}".encode()
        final_checkpoint_bytes = checkpoint_bytes
        checkpoint_sha = _write(stem.with_suffix(".zip"), checkpoint_bytes)
        sidecar_sha = _write(stem.with_suffix(".json"), {"index": index})
        case_path = stem.with_suffix(".cases.json")
        case_sha = _write(case_path, [{"case": index}])
        relative_case = case_path.relative_to(arm).as_posix()
        exams.append(
            {
                "child_trained_actions": boundary,
                "checkpoint": stem.with_suffix(".zip").relative_to(arm).as_posix(),
                "checkpoint_sha256": checkpoint_sha,
                "sidecar_sha256": sidecar_sha,
                "case_diagnostics": {
                    "path": relative_case,
                    "sha256": case_sha,
                    "case_count": 320,
                },
            }
        )
        case_files.append(
            {"path": relative_case, "sha256": case_sha, "records": 320}
        )
    terminal = arm / "checkpoints" / "terminal.zip"
    terminal_sha = _write(terminal, final_checkpoint_bytes)
    terminal_sidecar_sha = _write(
        arm / "checkpoints" / "terminal.json",
        {"terminal": True},
    )
    terminal_integrity_sha = _write(
        arm / "checkpoints" / "terminal.integrity.json",
        {"terminal": True},
    )
    report = {
        "protocol": qualify.frozen_v03.PROTOCOL,
        "cohort_id": qualify.R3_COHORT_ID,
        "arm": arm_name,
        "verdict": "arm_failed",
        "progress": {
            "child_trained_actions": v04.CHILD_ACTION_BUDGET,
            "lifetime_trained_actions": (
                v04.PARENT_LIFETIME_ACTIONS + v04.CHILD_ACTION_BUDGET
            ),
            "optimizer_updates": (
                v04.PARENT_OPTIMIZER_UPDATES
                + (v04.CHILD_ACTION_BUDGET // v04.ROLLOUT_TRANSITIONS)
                * v04.PPO_EPOCHS
            ),
            "exam_count": v04.EXAM_COUNT,
        },
        "child_trained_actions": v04.CHILD_ACTION_BUDGET,
        "lifetime_trained_actions": (
            v04.PARENT_LIFETIME_ACTIONS + v04.CHILD_ACTION_BUDGET
        ),
        "optimizer_updates": (
            v04.PARENT_OPTIMIZER_UPDATES
            + (v04.CHILD_ACTION_BUDGET // v04.ROLLOUT_TRANSITIONS)
            * v04.PPO_EPOCHS
        ),
        "exam_count": v04.EXAM_COUNT,
        "case_count": v04.EXAM_COUNT * 320,
        "architecture_definition_eligible": False,
        "promotable": False,
        "development_checkpoint_reuse_authorized": False,
        "successor_checkpoint_authorized": False,
        "resume_authorized": False,
        "u3_authorized": False,
        "exam_records": exams,
        "case_evidence": {
            "file_count": v04.EXAM_COUNT,
            "record_count": v04.EXAM_COUNT * 320,
            "files": case_files,
        },
        "terminal_checkpoint": {
            "path": "checkpoints/terminal.zip",
            "sha256": terminal_sha,
            "sidecar_sha256": terminal_sidecar_sha,
            "integrity_sha256": terminal_integrity_sha,
            "promotable": False,
        },
    }
    report_sha = _write(arm / "report.json", report)
    report_integrity_sha = _write(
        arm / "report.integrity.json",
        {
            "report_sha256": report_sha,
            "terminal_checkpoint_sha256": terminal_sha,
            "terminal_sidecar_sha256": terminal_sidecar_sha,
            "terminal_integrity_sha256": terminal_integrity_sha,
        },
    )
    status_sha = _write(arm / "status.json", {"phase": "completed"})
    return {
        "report": report_sha,
        "report_integrity": report_integrity_sha,
        "status": status_sha,
        "terminal_checkpoint": terminal_sha,
        "terminal_sidecar": terminal_sidecar_sha,
        "terminal_integrity": terminal_integrity_sha,
    }


def test_r3_arm_authenticates_all_exam_case_and_terminal_bindings(
    tmp_path: Path,
) -> None:
    expected = _arm_fixture(tmp_path, "sham")

    result = qualify._verify_r3_arm(tmp_path, "sham", expected)
    assert result["exam_count"] == 32
    assert result["case_count"] == 10_240
    assert result["promotable"] is False

    case = (
        tmp_path
        / "sham"
        / "checkpoints"
        / "rolling"
        / "exam-0032768.cases.json"
    )
    case.write_text("tampered", encoding="utf-8")
    with pytest.raises(
        qualify.IneffectiveTraceQualificationError,
        match="case checksum",
    ):
        qualify._verify_r3_arm(tmp_path, "sham", expected)


def test_source_tag_requires_clean_published_annotated_tag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    message = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    claim = tmp_path / "claim.json"
    claim.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(qualify, "CANONICAL_CLAIM", claim)
    scans: list[bool] = []

    def build_payload(
        _repository: Path,
        *,
        source_commit: str,
        scan_fresh_identities: bool,
    ) -> dict:
        assert source_commit == head
        scans.append(scan_fresh_identities)
        return payload

    monkeypatch.setattr(
        qualify,
        "build_ineffective_trace_tag_payload",
        build_payload,
    )

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
            ("for-each-ref", "--format=%(contents)", ref): message,
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=outputs[tuple(arguments)] + "\n",
            stderr="",
        )

    source = qualify.verify_source_tag(
        tmp_path,
        expected_source_commit=head,
        expected_tag_object=tag_object,
        runner=runner,
    )
    assert source["tag_object"] == tag_object
    assert source["tag_payload_sha256"] == qualify._canonical_sha256(payload)
    assert scans == [False]


def test_fresh_identity_scan_rejects_consumed_worker_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualify, "EVIDENCE_ROOT", tmp_path)
    (tmp_path / "old-run.json").write_text(
        '{"worker_stream":20260762}',
        encoding="utf-8",
    )
    with pytest.raises(
        qualify.IneffectiveTraceQualificationError,
        match="already consumed",
    ):
        qualify._fresh_identity_preflight()
    (tmp_path / "old-run.json").unlink()
    assert qualify._fresh_identity_preflight()["identities_unconsumed"]


class _BaseQualification:
    def __init__(self, token: object) -> None:
        self.token = token

    def seed_access(self) -> object:
        return self.token


def test_evidence_exposes_exact_public_api_and_read_only_guards() -> None:
    token = object()
    report = {"protocol": qualify.PROTOCOL, "verdict": "qualified"}
    report_bytes = qualify._canonical_json_bytes(report)
    mapping = MappingProxyType(
        {
            lesson: frozenset({_digest(str(index + 1))})
            for index, lesson in enumerate(lessons.LessonId)
        }
    )
    evidence = qualify.IneffectiveTraceQualificationEvidence(
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
        r3_terminal_evidence_sha256=_digest("3"),
        storage_caps=MappingProxyType(qualify._storage_caps()),
        _report_bytes=report_bytes,
        _base_qualification=_BaseQualification(token),
        _forbidden_layouts=mapping,
    )

    assert set(evidence.public_dict()) == {
        "report",
        "report_sha256",
        "checksum",
        "source_commit",
        "tag",
        "tag_object",
        "tag_payload_sha256",
        "verdict",
        "protocol_document_sha256",
        "guard_mapping_sha256",
        "sampler_preflight_sha256",
        "architecture_contract_sha256",
        "smoke_evidence_sha256",
        "protected_partitions_sha256",
        "r3_terminal_evidence_sha256",
        "storage_caps",
    }
    assert evidence.verified_report() == report
    assert evidence.seed_access() is token
    assert evidence.forbidden_layout_hashes() == mapping
    assert isinstance(evidence.forbidden_layout_hashes(), MappingProxyType)


def test_failed_one_shot_collection_retains_durable_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification = tmp_path / "qualification"
    report = qualification / "report.json"
    checksum = qualification / "report.json.sha256"
    claim = qualification / "claim.json"
    monkeypatch.setattr(qualify, "CANONICAL_QUALIFICATION_DIRECTORY", qualification)
    monkeypatch.setattr(qualify, "CANONICAL_REPORT", report)
    monkeypatch.setattr(qualify, "CANONICAL_CHECKSUM", checksum)
    monkeypatch.setattr(qualify, "CANONICAL_CLAIM", claim)
    source = {
        "commit": "a" * 40,
        "tag_object": "b" * 40,
        "tag_payload_sha256": _digest("c"),
    }
    monkeypatch.setattr(qualify, "verify_source_tag", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(
        qualify,
        "_storage_preflight",
        lambda **_kwargs: {"passed": True},
    )
    base = _BaseQualification(object())
    mapping = MappingProxyType(
        {lesson: frozenset() for lesson in lessons.LessonId}
    )
    monkeypatch.setattr(
        qualify,
        "_collect_static_inputs",
        lambda _repository: (
            {},
            base,
            mapping,
            {"applied_mapping_sha256": _digest("d")},
            {},
            [],
            {},
            [],
            {},
            {},
        ),
    )

    def fail_smoke(**_kwargs: object) -> dict:
        raise RuntimeError("smoke failed")

    with pytest.raises(RuntimeError, match="smoke failed"):
        qualify.collect_ineffective_trace_qualification(
            repository=tmp_path,
            output=report,
            smoke_runner=fail_smoke,
        )
    assert claim.is_file()
    assert not report.exists()
    with pytest.raises(qualify.IneffectiveTraceQualificationError):
        qualify.collect_ineffective_trace_qualification(
            repository=tmp_path,
            output=report,
            smoke_runner=fail_smoke,
        )
