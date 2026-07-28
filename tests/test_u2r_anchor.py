from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from dungeon_apprentice import u2r_anchor as anchor
from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v02_u2r


def _repository(tmp_path: Path) -> Path:
    protocol = tmp_path / anchor.PROTOCOL_REPOSITORY_PATH
    protocol.parent.mkdir(parents=True)
    protocol.write_text("# Prospective U2r protocol\n", encoding="utf-8")
    return tmp_path


def _exclusions() -> dict[str, object]:
    return {
        "exact_layout_set_sha256": "e" * 64,
        "exact_layouts": 12_345,
        "qualification_and_validation_layouts": 2_319,
        "completed_history_layouts": 10_000,
        "accepted_confirmation_layouts": 800,
        "inspectable_confirmation_outcomes": 819,
        "inspectable_confirmation_layouts": 807,
        "prior_u1_confirmation_layouts": 600,
        "unavailable_original_terminal_active_worker_layouts_upper_bound": 12,
        "unavailable_original_terminal_active_worker_layouts_status": (
            "unauthenticated_unavailable_not_in_static_exclusion_set"
        ),
        "prior_u1_confirmation_report": "/evidence/u1-report.json",
        "prior_u1_confirmation_report_sha256": "1" * 64,
        "prior_u1_confirmation_set_sha256": "2" * 64,
        "history_files": [
            {
                "child_seed": seed,
                "path": f"/evidence/{seed}/episodes.jsonl",
                "sha256": str(index) * 64,
                "completed_exact_layouts": 3_000 + index,
            }
            for index, seed in enumerate((20260737, 20260741, 20260745), start=3)
        ],
        "selection_journal_files": 1_643,
        "selection_journal_sha256": "6" * 64,
        "applied_by_lesson": {
            lesson: {
                "exact_layouts": 100 + index,
                "exact_layout_set_sha256": f"{index + 7:x}" * 64,
                "rule": rule,
            }
            for index, (lesson, rule) in enumerate(anchor.APPLIED_GUARD_RULES.items())
        },
        "applied_mapping_sha256": "b" * 64,
    }


def _verification_contract(repository: Path) -> dict[str, object]:
    return {
        "expected_source_commit": "a" * 40,
        "expected_protocol_sha256": hashlib.sha256(
            (repository / anchor.PROTOCOL_REPOSITORY_PATH).read_bytes()
        ).hexdigest(),
        "expected_exclusions": _exclusions(),
    }


def test_payload_freezes_parent_budget_stability_and_closed_confirmation(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    payload = anchor.build_anchor_payload(
        repository,
        source_commit="a" * 40,
        exclusions=_exclusions(),
    )

    assert set(payload) == anchor.ANCHOR_FIELDS
    assert payload["source_commit"] == "a" * 40
    assert payload["training_protocol"] == v02_u2r.PROTOCOL
    assert payload["protocol_document"] == anchor.PROTOCOL_REPOSITORY_PATH
    assert (
        payload["protocol_document_sha256"]
        == hashlib.sha256((repository / anchor.PROTOCOL_REPOSITORY_PATH).read_bytes()).hexdigest()
    )
    assert payload["parent"] == {
        "checkpoint": str(v02_u2r.SOURCE_ARCHIVE),
        "checkpoint_sha256": v02_u2r.SOURCE_ARCHIVE_SHA256,
        "sidecar": str(v02_u2r.SOURCE_ARCHIVE.with_suffix(".json")),
        "sidecar_sha256": v02_u2r.SOURCE_SIDECAR_SHA256,
        "integrity": str(v02_u2r.SOURCE_ARCHIVE.with_suffix(".integrity.json")),
        "integrity_sha256": v02_u2r.SOURCE_INTEGRITY_SHA256,
        "manifest": str(v02_u2r.SOURCE_ARCHIVE.parent.parent / "manifest.json"),
        "manifest_sha256": v02_u2r.SOURCE_MANIFEST_SHA256,
        "training_source_commit": v02_u2r.SOURCE_TRAINING_COMMIT,
        "policy_member_sha256": anchor.PARENT_POLICY_MEMBER_SHA256,
        "policy_tensor_sha256": v02_u2r.SOURCE_POLICY_TENSOR_SHA256,
        "optimizer_state_sha256": anchor.PARENT_OPTIMIZER_STATE_SHA256,
    }
    assert payload["failed_confirmation"] == {
        "report": str(v02_u2r.U2_CONFIRMATION_REPORT),
        "report_sha256": v02_u2r.U2_CONFIRMATION_REPORT_SHA256,
        "attempt": str(v02_u2r.U2_CONFIRMATION_ATTEMPT),
        "attempt_sha256": v02_u2r.U2_CONFIRMATION_ATTEMPT_SHA256,
        "source_commit": v02_u2r.U2_CONFIRMATION_SOURCE_COMMIT,
        "verdict": "capability_failed",
        "u2_score": 169,
        "u2_panels": [84, 85],
    }
    expected_inventory = {
        key: value
        for key, value in _exclusions().items()
        if key in anchor.STATIC_INVENTORY_FIELDS
    }
    assert payload["static_exclusions"] == expected_inventory
    assert payload["failed_r0_launch"] == anchor.FAILED_R0_EVIDENCE
    assert payload["applied_training_guards"] == {
        "mapping_sha256": "b" * 64,
        "by_lesson": _exclusions()["applied_by_lesson"],
        "full_static_inventory_sha256": "e" * 64,
        "full_static_inventory_layouts": 12_345,
        "unavailable_original_terminal_active_worker_layouts_upper_bound": 12,
        "unavailable_original_terminal_active_worker_layouts_status": (
            "unauthenticated_unavailable_not_in_static_exclusion_set"
        ),
    }
    assert payload["budget"] == {
        "source_child_actions": 688_128,
        "additional_action_budget": 360_448,
        "terminal_child_actions": 1_048_576,
        "source_lifetime_actions": 1_474_560,
        "terminal_lifetime_actions": 1_835_008,
        "source_optimizer_updates": 2_880,
        "terminal_optimizer_updates": 3_584,
        "evaluation_interval": 32_768,
        "complete_windows": 11,
    }
    assert payload["seeds"] == {
        "algorithm_seed": 20260749,
        "worker_streams": [20260749, 20260750, 20260751, 20260752],
        "training_allocation": [0, 999_999],
        "segment_seed_offset": 100_000,
        "initial_run_name": "v02-u2r-r1-seed-20260745",
        "resume_run_name_pattern": "v02-u2r-r1-seed-20260745-resume-N",
        "segment_algorithm_seed_formula": ("algorithm_seed + segment_index * segment_seed_offset"),
        "segment_worker_stream_formula": (
            "initial_worker_stream + segment_index * segment_seed_offset"
        ),
    }
    assert payload["stability_rule"] == anchor.STABILITY_RULE
    assert payload["stability_rule"]["terminal_pair_child_actions"] == [
        1_015_808,
        1_048_576,
    ]
    assert payload["stability_rule"]["u2_stability_overall"] == 72
    assert payload["stability_rule"]["u2_stability_panel"] == 34
    assert payload["stability_rule"]["mean_ineffective_interactions_max"] == 3.0
    assert payload["stability_rule"]["terminal_success_decline_max"] == 2
    assert payload["confirmation"] == {
        "consumed_ranges": anchor.CONSUMED_CONFIRMATION_RANGES,
        "fresh_reserved_ranges": anchor.FRESH_CONFIRMATION_RANGES,
        "available_during_training": False,
        "requires_separate_post_training_preregistration": True,
    }


def test_message_is_one_canonical_json_line(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    message = anchor.anchor_message(
        repository,
        source_commit="a" * 40,
        exclusions=_exclusions(),
    )
    assert "\n" not in message
    assert message == json.dumps(
        json.loads(message),
        sort_keys=True,
        separators=(",", ":"),
    )

    first = anchor.build_anchor_payload(
        repository,
        source_commit="a" * 40,
        exclusions=_exclusions(),
    )
    first["stability_rule"]["u2_stability_overall"] = 0
    first["confirmation"]["fresh_reserved_ranges"]["separated_u2"][0] = 0
    second = anchor.build_anchor_payload(
        repository,
        source_commit="a" * 40,
        exclusions=_exclusions(),
    )
    assert second["stability_rule"]["u2_stability_overall"] == 72
    assert second["confirmation"]["fresh_reserved_ranges"]["separated_u2"][0] == 15_240_000


def test_failed_r0_launch_verifier_requires_exact_zero_action_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "u2r-stability-20260723"
    run = root / "v02-u2r-seed-20260745"
    run.mkdir(parents=True)
    values = {
        run / "manifest.json": {
            "protocol": "dungeon-apprentice-v0.2-u2r-stability",
            "source": {
                "commit": anchor.FAILED_R0_SOURCE_COMMIT,
                "dirty": False,
            },
        },
        run / "crash.json": {
            "classification": "training_failure",
            "setup_complete": True,
            "traceback": (
                "RuntimeError: could not sample a training layout outside "
                "reserved evidence sets"
            ),
        },
        root / ".launcher-v02-u2r-seed-20260745.json": {
            "state": "exited",
            "exit_status": 1,
            "forwarded_signal": None,
        },
    }
    for path, value in values.items():
        path.write_text(json.dumps(value), encoding="utf-8")
    episode = {
        "type": "episode_start",
        "active": True,
        "elapsed_steps": 0,
        "worker_transition_at_start": 0,
        "diagnostics": {"action_histogram": {}},
    }
    (run / "episode-starts.jsonl").write_text(
        json.dumps(episode) + "\n",
        encoding="utf-8",
    )
    evidence = json.loads(json.dumps(anchor.FAILED_R0_EVIDENCE))
    evidence["root"] = str(root)
    evidence["run"] = str(run)
    artifact_paths = {
        "manifest": run / "manifest.json",
        "crash": run / "crash.json",
        "episode_starts": run / "episode-starts.jsonl",
        "supervisor": root / ".launcher-v02-u2r-seed-20260745.json",
    }
    for label, path in artifact_paths.items():
        evidence["artifacts"][label] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    monkeypatch.setattr(anchor, "FAILED_R0_EVIDENCE", evidence)

    assert anchor.verify_failed_r0_launch() == evidence
    (run / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(anchor.U2rAnchorError, match="inventory changed"):
        anchor.verify_failed_r0_launch()


def test_payload_rejects_unsafe_protocol_and_invalid_exclusions(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    protocol = repository / anchor.PROTOCOL_REPOSITORY_PATH
    protocol.unlink()
    target = tmp_path / "outside.md"
    target.write_text("outside", encoding="utf-8")
    protocol.symlink_to(target)
    with pytest.raises(anchor.U2rAnchorError, match="missing or unsafe"):
        anchor.build_anchor_payload(
            repository,
            source_commit="a" * 40,
            exclusions=_exclusions(),
        )

    protocol.unlink()
    protocol.write_text("# protocol\n", encoding="utf-8")
    invalid_digest = _exclusions()
    invalid_digest["exact_layout_set_sha256"] = "not-a-digest"
    with pytest.raises(anchor.U2rAnchorError, match="SHA-256"):
        anchor.build_anchor_payload(
            repository,
            source_commit="a" * 40,
            exclusions=invalid_digest,
        )
    invalid_count = _exclusions()
    invalid_count["exact_layouts"] = 0
    with pytest.raises(anchor.U2rAnchorError, match="positive integer"):
        anchor.build_anchor_payload(
            repository,
            source_commit="a" * 40,
            exclusions=invalid_count,
        )
    wrong_limitation = _exclusions()
    wrong_limitation["unavailable_original_terminal_active_worker_layouts_upper_bound"] = 11
    with pytest.raises(anchor.U2rAnchorError, match="limitation changed"):
        anchor.build_anchor_payload(
            repository,
            source_commit="a" * 40,
            exclusions=wrong_limitation,
        )


def _runner(
    *,
    message: str,
    head: str = "a" * 40,
    status: str = "",
    tag_type: str = "tag",
    tag_object: str = "b" * 40,
    peeled: str = "a" * 40,
    remote_url: str = anchor.EXPECTED_ORIGIN_URL,
    remote_object: str = "b" * 40,
    failed_r0_object: str = anchor.FAILED_R0_TAG_OBJECT,
):
    outputs = {
        ("rev-parse", "--verify", "HEAD^{commit}"): head,
        ("status", "--porcelain=v1", "--untracked-files=all"): status,
        ("remote", "get-url", anchor.ANCHOR_REMOTE): remote_url,
        (
            "cat-file",
            "-t",
            f"refs/tags/{anchor.FAILED_R0_TAG}",
        ): "tag",
        (
            "rev-parse",
            "--verify",
            f"refs/tags/{anchor.FAILED_R0_TAG}",
        ): failed_r0_object,
        (
            "rev-list",
            "-n",
            "1",
            f"refs/tags/{anchor.FAILED_R0_TAG}",
        ): anchor.FAILED_R0_SOURCE_COMMIT,
        (
            "ls-remote",
            "--tags",
            anchor.ANCHOR_REMOTE,
            f"refs/tags/{anchor.FAILED_R0_TAG}",
        ): (
            f"{failed_r0_object}\t"
            f"refs/tags/{anchor.FAILED_R0_TAG}"
        ),
        ("cat-file", "-t", f"refs/tags/{anchor.ANCHOR_TAG}"): tag_type,
        (
            "rev-parse",
            "--verify",
            f"refs/tags/{anchor.ANCHOR_TAG}",
        ): tag_object,
        ("rev-list", "-n", "1", f"refs/tags/{anchor.ANCHOR_TAG}"): peeled,
        (
            "ls-remote",
            "--tags",
            anchor.ANCHOR_REMOTE,
            f"refs/tags/{anchor.ANCHOR_TAG}",
        ): f"{remote_object}\trefs/tags/{anchor.ANCHOR_TAG}",
        (
            "for-each-ref",
            "--format=%(contents)",
            f"refs/tags/{anchor.ANCHOR_TAG}",
        ): message,
    }

    def run(command, **_kwargs):
        assert command[0] == "git"
        key = tuple(command[1:])
        if key not in outputs:
            raise AssertionError(f"unexpected git call: {command}")
        return subprocess.CompletedProcess(command, 0, outputs[key] + "\n", "")

    return run


def test_verifier_requires_clean_exact_annotated_object_on_origin(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    message = anchor.anchor_message(
        repository,
        source_commit="a" * 40,
        exclusions=_exclusions(),
    )
    verified = anchor.verify_external_anchor(
        repository,
        **_verification_contract(repository),
        runner=_runner(message=message),
    )
    assert verified.tag == anchor.ANCHOR_TAG
    assert verified.tag_object == "b" * 40
    assert verified.static_exclusion_set_sha256 == "e" * 64
    assert verified.static_exclusion_layouts == 12_345
    assert verified.static_exclusion_inventory["completed_history_layouts"] == 10_000
    assert verified.applied_training_guard_mapping_sha256 == "b" * 64
    assert set(verified.applied_training_guards) == set(anchor.APPLIED_GUARD_RULES)
    assert verified.failed_r0_launch == anchor.FAILED_R0_EVIDENCE
    assert verified.additional_action_budget == 360_448
    assert verified.terminal_child_actions == 1_048_576
    assert verified.terminal_lifetime_actions == 1_835_008
    assert verified.terminal_optimizer_updates == 3_584
    assert verified.worker_streams == (20260749, 20260750, 20260751, 20260752)

    with pytest.raises(anchor.U2rAnchorError, match="clean committed"):
        anchor.verify_external_anchor(
            repository,
            **_verification_contract(repository),
            runner=_runner(message=message, status=" M README.md"),
        )
    with pytest.raises(anchor.U2rAnchorError, match="r0 annotated tag identity changed"):
        anchor.verify_external_anchor(
            repository,
            **_verification_contract(repository),
            runner=_runner(message=message, failed_r0_object="c" * 40),
        )
    with pytest.raises(anchor.U2rAnchorError, match="annotated"):
        anchor.verify_external_anchor(
            repository,
            **_verification_contract(repository),
            runner=_runner(message=message, tag_type="commit"),
        )
    with pytest.raises(anchor.U2rAnchorError, match="frozen on origin"):
        anchor.verify_external_anchor(
            repository,
            **_verification_contract(repository),
            runner=_runner(message=message, remote_object="c" * 40),
        )


def test_verifier_rejects_payload_drift_and_unknown_fields(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    expected = anchor.build_anchor_payload(
        repository,
        source_commit="a" * 40,
        exclusions=_exclusions(),
    )
    drifted = dict(expected)
    drifted["budget"] = dict(expected["budget"])
    drifted["budget"]["additional_action_budget"] = 360_449
    with pytest.raises(anchor.U2rAnchorError, match="payload differs"):
        anchor.verify_external_anchor(
            repository,
            **_verification_contract(repository),
            runner=_runner(
                message=json.dumps(
                    drifted,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        )

    unknown = dict(expected)
    unknown["extra"] = True
    with pytest.raises(anchor.U2rAnchorError, match="fields"):
        anchor.verify_external_anchor(
            repository,
            **_verification_contract(repository),
            runner=_runner(
                message=json.dumps(
                    unknown,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        )


def test_publisher_creates_pushes_and_then_verifies_fixed_tag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    calls: list[tuple[str, ...]] = []
    remote_exists = False
    sentinel = object()

    def fake_git(_repository, arguments, **_kwargs):
        nonlocal remote_exists
        command = tuple(arguments)
        calls.append(command)
        if command[:3] == ("rev-parse", "--verify", "HEAD^{commit}"):
            return "a" * 40
        if command[0] == "status":
            return ""
        if command[:2] == ("remote", "get-url"):
            return anchor.EXPECTED_ORIGIN_URL
        if command[0] == "ls-remote":
            return f"{'b' * 40}\trefs/tags/{anchor.ANCHOR_TAG}" if remote_exists else ""
        if command[0] == "tag":
            assert command[-1] == "a" * 40
            payload = json.loads(command[4])
            assert payload["static_exclusions"]["exact_layouts"] == 12_345
            return ""
        if command[0] == "push":
            remote_exists = True
            return ""
        raise AssertionError(command)

    def runner(command, **_kwargs):
        assert command[:3] == ["git", "show-ref", "--verify"]
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(anchor, "_git", fake_git)
    monkeypatch.setattr(
        anchor,
        "verify_external_anchor",
        lambda _repository, **_kwargs: sentinel,
    )
    result = anchor.publish_external_anchor(
        repository,
        source_commit="a" * 40,
        exclusions=_exclusions(),
        runner=runner,
    )
    assert result is sentinel
    assert any(command[0] == "tag" for command in calls)
    assert any(command[0] == "push" for command in calls)


def test_anchor_module_never_authorizes_confirmation_streams() -> None:
    source = Path(anchor.__file__).read_text(encoding="utf-8")
    assert "authorize_u2_seed" not in source
    assert "_post_training_confirmation_seed_access" not in source
    assert "U2R_U2_CONFIRMATION" not in source


def _write_resume_segment_legacy(
    root: Path,
    *,
    index: int,
    prior: Path | None = None,
    commit: str = "a" * 40,
    phase: str = "interrupted",
    remediation_actions: int = 32_768,
) -> Path:
    name = v02_u2r.DEFAULT_RUN_NAME if index == 0 else f"{v02_u2r.DEFAULT_RUN_NAME}-resume-{index}"
    segment = root / name
    checkpoint = segment / "checkpoints" / "latest-safe.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"checkpoint-{index}".encode())
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    child_actions = v02_u2r.SOURCE_CHILD_ACTIONS + remediation_actions
    lifetime_actions = v02_u2r.SOURCE_LIFETIME_ACTIONS + remediation_actions
    optimizer_updates = (
        v02_u2r.SOURCE_OPTIMIZER_UPDATES
        + remediation_actions // v02_u2r.ROLLOUT_TRANSITIONS * v02_u2r.PPO_EPOCHS
    )
    exclusions = {
        "exact_layout_set_sha256": "d" * 64,
        "exact_layouts": 123,
    }
    external_preregistration = {
        "tag": anchor.ANCHOR_TAG,
        "tag_object": "e" * 40,
    }
    frozen_identity = {
        "effective_config": {
            "contract": "frozen",
            "exclusions": exclusions,
            "external_preregistration": external_preregistration,
        },
        "parent": {"checkpoint_sha256": "b" * 64},
        "qualification": {"report_sha256": "c" * 64},
        "exclusions": exclusions,
        "external_preregistration": external_preregistration,
        "storage_mount": {
            "path": "/Volumes/T7 Developer",
            "is_mount": True,
            "distinct_from_system": True,
        },
    }
    if prior is None:
        start_child_actions = v02_u2r.SOURCE_CHILD_ACTIONS
        start_remediation_actions = 0
        start_lifetime_actions = v02_u2r.SOURCE_LIFETIME_ACTIONS
        resume_checkpoint = None
        prior_remediation_actions = 0
    else:
        prior_status = json.loads((prior / "status.json").read_text(encoding="utf-8"))
        prior_checkpoint = (prior / str(prior_status["latest_safe_checkpoint"])).resolve()
        prior_sidecar = json.loads(
            prior_checkpoint.with_suffix(".json").read_text(encoding="utf-8")
        )
        prior_progress = prior_sidecar["progress"]
        start_child_actions = int(prior_progress["child_trained_actions"])
        start_remediation_actions = int(prior_progress["remediation_trained_actions"])
        start_lifetime_actions = int(prior_progress["lifetime_trained_actions"])
        resume_checkpoint = str(prior_checkpoint)
        prior_remediation_actions = start_remediation_actions
    segment_identity = {
        "index": index,
        "lineage_source_child_seed": v02_u2r.SOURCE_CHILD_SEED,
        "initial_algorithm_seed": v02_u2r.REMEDIATION_ALGORITHM_SEED,
        "initial_worker_streams": list(v02_u2r.REMEDIATION_WORKER_STREAMS),
        "segment_algorithm_seed": (v02_u2r.REMEDIATION_ALGORITHM_SEED + index * 100_000),
        "segment_worker_streams": [
            seed + index * 100_000 for seed in v02_u2r.REMEDIATION_WORKER_STREAMS
        ],
        "start_child_trained_actions": start_child_actions,
        "start_remediation_trained_actions": start_remediation_actions,
        "start_lifetime_trained_actions": start_lifetime_actions,
        "resume_checkpoint": resume_checkpoint,
    }
    carried: list[dict[str, object]] = []
    for boundary in range(
        v02_u2r.EVALUATION_INTERVAL,
        prior_remediation_actions + 1,
        v02_u2r.EVALUATION_INTERVAL,
    ):
        exam = (
            segment
            / "checkpoints"
            / "rolling"
            / f"exam-{v02_u2r.SOURCE_CHILD_ACTIONS + boundary:07d}.zip"
        )
        exam.parent.mkdir(parents=True, exist_ok=True)
        exam.write_bytes(f"exam-{boundary}".encode())
        exam_digest = hashlib.sha256(exam.read_bytes()).hexdigest()
        exam_sidecar = exam.with_suffix(".json")
        exam_sidecar.write_text(
            json.dumps(
                {
                    "schema_version": v02_u2r.CHECKPOINT_SCHEMA_VERSION,
                    "protocol": v02_u2r.PROTOCOL,
                    "kind": "exam",
                    "resume_eligible": False,
                    "checkpoint_sha256": exam_digest,
                    "source": {"dirty": False, "commit": commit},
                    **{
                        key: frozen_identity[key]
                        for key in (
                            "effective_config",
                            "parent",
                            "qualification",
                            "exclusions",
                            "external_preregistration",
                            "storage_mount",
                        )
                    },
                    "progress": {
                        "child_trained_actions": (v02_u2r.SOURCE_CHILD_ACTIONS + boundary)
                    },
                }
            ),
            encoding="utf-8",
        )
        exam_sidecar_digest = hashlib.sha256(exam_sidecar.read_bytes()).hexdigest()
        exam_integrity = exam.with_suffix(".integrity.json")
        exam_integrity.write_text(
            json.dumps(
                {
                    "schema_version": v02_u2r.CHECKPOINT_SCHEMA_VERSION,
                    "protocol": v02_u2r.PROTOCOL,
                    "checkpoint": exam.name,
                    "checkpoint_sha256": exam_digest,
                    "sidecar": exam_sidecar.name,
                    "sidecar_sha256": exam_sidecar_digest,
                }
            ),
            encoding="utf-8",
        )
        carried.append(
            {
                "path": str(exam.relative_to(segment)),
                "child_trained_actions": (v02_u2r.SOURCE_CHILD_ACTIONS + boundary),
                "checkpoint_sha256": exam_digest,
                "sidecar_sha256": exam_sidecar_digest,
                "integrity_sha256": hashlib.sha256(exam_integrity.read_bytes()).hexdigest(),
            }
        )
    (segment / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": v02_u2r.CHECKPOINT_SCHEMA_VERSION,
                "protocol": v02_u2r.PROTOCOL,
                "source": {"dirty": False, "commit": commit},
                **frozen_identity,
                "segment": segment_identity,
                "carried_exams": carried,
            }
        ),
        encoding="utf-8",
    )
    sidecar = checkpoint.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": v02_u2r.CHECKPOINT_SCHEMA_VERSION,
                "protocol": v02_u2r.PROTOCOL,
                "kind": "latest",
                "resume_eligible": True,
                "checkpoint_sha256": checkpoint_sha256,
                "source": {"dirty": False, "commit": commit},
                **{
                    key: frozen_identity[key]
                    for key in (
                        "effective_config",
                        "parent",
                        "qualification",
                        "exclusions",
                        "external_preregistration",
                        "storage_mount",
                    )
                },
                "curriculum": {"mastered": False},
                "controller": {
                    "terminal_eligible": None,
                    "exam_records": [
                        {"child_trained_actions": (v02_u2r.SOURCE_CHILD_ACTIONS + boundary)}
                        for boundary in range(
                            v02_u2r.EVALUATION_INTERVAL,
                            remediation_actions + 1,
                            v02_u2r.EVALUATION_INTERVAL,
                        )
                    ],
                },
                "progress": {
                    "child_trained_actions": child_actions,
                    "remediation_trained_actions": remediation_actions,
                    "lifetime_trained_actions": lifetime_actions,
                    "collected_actions": lifetime_actions,
                    "trained_actions": lifetime_actions,
                    "remaining_remediation_actions": (
                        v02_u2r.ADDITIONAL_ACTION_BUDGET - remediation_actions
                    ),
                    "optimizer_updates": optimizer_updates,
                },
                "segment": segment_identity,
            }
        ),
        encoding="utf-8",
    )
    integrity = checkpoint.with_suffix(".integrity.json")
    integrity.write_text(
        json.dumps(
            {
                "schema_version": v02_u2r.CHECKPOINT_SCHEMA_VERSION,
                "protocol": v02_u2r.PROTOCOL,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": checkpoint_sha256,
                "sidecar": sidecar.name,
                "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    (segment / "status.json").write_text(
        json.dumps(
            {
                "protocol": v02_u2r.PROTOCOL,
                "phase": phase,
                "source": {"dirty": False, "commit": commit},
                "latest_safe_checkpoint": "checkpoints/latest-safe.zip",
                "latest_safe_checkpoint_sha256": checkpoint_sha256,
                "child_trained_actions": child_actions,
                "remediation_trained_actions": remediation_actions,
                "lifetime_trained_actions": lifetime_actions,
                "optimizer_updates": optimizer_updates,
            }
        ),
        encoding="utf-8",
    )
    return segment


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _fixture_exam_cases(
    boundary: int,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    panel_scores = {
        lessons.LessonId.NAVIGATE: (39, 40),
        lessons.LessonId.VISIBLE_UNLOCK: (37, 39),
        lessons.LessonId.LOCAL_UNLOCK: (38, 39),
        lessons.LessonId.SEPARATED_UNLOCK: (36, 36),
    }
    ineffective_totals = {
        lessons.LessonId.NAVIGATE: 2,
        lessons.LessonId.VISIBLE_UNLOCK: 524,
        lessons.LessonId.LOCAL_UNLOCK: 379,
        lessons.LessonId.SEPARATED_UNLOCK: 523,
    }
    evaluations: dict[str, dict[str, object]] = {}
    records: list[dict[str, object]] = []
    for lesson in lessons.LessonId:
        selected: list[dict[str, object]] = []
        for case_index in range(v02_u2r.EVALUATION_SEED_COUNT):
            panel = 0 if case_index < 40 else 1
            panel_case_index = case_index % 40
            success = panel_case_index < panel_scores[lesson][panel]
            steps = 8 + case_index % 3
            oracle_actions = 4 + case_index % 2
            visibility_stratum = None
            if lesson is lessons.LessonId.SEPARATED_UNLOCK:
                visibility_stratum = (
                    "key_hidden_door_hidden",
                    "key_hidden_door_visible",
                    "key_visible_door_hidden",
                    "key_visible_door_visible",
                )[case_index % 4]
            ineffective_base, ineffective_remainder = divmod(
                ineffective_totals[lesson],
                v02_u2r.EVALUATION_SEED_COUNT,
            )
            selected.append(
                {
                    "lesson_id": lesson.value,
                    "lesson_label": lessons.LESSON_SPECS[lesson].label,
                    "case_index": case_index,
                    "panel": panel,
                    "panel_case_index": panel_case_index,
                    "seed": (lessons.LESSON_SPECS[lesson].validation_seed_base + case_index),
                    "layout_sha256": _sha(f"{boundary}-{lesson.value}-layout-{case_index}"),
                    "geometry_sha256": _sha(f"{boundary}-{lesson.value}-geometry-{case_index}"),
                    "success": success,
                    "terminal_reason": "success" if success else "timeout",
                    "steps": steps,
                    "milestones": {
                        "key_picked_up": success,
                        "door_opened": success,
                        "success": success,
                    },
                    "collisions": case_index % 2,
                    "ineffective_interactions": (
                        ineffective_base + int(case_index < ineffective_remainder)
                    ),
                    "unique_cells": 8 + case_index % 4,
                    "reachable_cells": 20,
                    "coverage": (8 + case_index % 4) / 20,
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
        records.extend(selected)
        evaluations[lesson.value] = lessons.aggregate_u2_case_evidence(
            lesson,
            selected,
            timestamp="2026-07-23T00:00:00+00:00",
        ).public_dict()
    return evaluations, records


def _write_resume_segment(
    root: Path,
    *,
    index: int,
    prior: Path | None = None,
    commit: str = "a" * 40,
    phase: str = "interrupted",
    remediation_actions: int = 32_768,
) -> Path:
    name = v02_u2r.DEFAULT_RUN_NAME if index == 0 else f"{v02_u2r.DEFAULT_RUN_NAME}-resume-{index}"
    segment = root / name
    checkpoints = segment / "checkpoints"
    checkpoints.mkdir(parents=True)
    child_actions = v02_u2r.SOURCE_CHILD_ACTIONS + remediation_actions
    lifetime_actions = v02_u2r.SOURCE_LIFETIME_ACTIONS + remediation_actions
    optimizer_updates = (
        v02_u2r.SOURCE_OPTIMIZER_UPDATES
        + remediation_actions // v02_u2r.ROLLOUT_TRANSITIONS * v02_u2r.PPO_EPOCHS
    )
    exclusions = {
        "exact_layout_set_sha256": "d" * 64,
        "exact_layouts": 123,
    }
    external_preregistration = {
        "tag": anchor.ANCHOR_TAG,
        "tag_object": "e" * 40,
    }
    frozen_identity = {
        "effective_config": {
            "contract": "frozen",
            "exclusions": exclusions,
            "external_preregistration": external_preregistration,
        },
        "parent": {"checkpoint_sha256": "b" * 64},
        "qualification": {"report_sha256": "c" * 64},
        "exclusions": exclusions,
        "external_preregistration": external_preregistration,
        "storage_mount": {
            "path": "/Volumes/T7 Developer",
            "is_mount": True,
            "distinct_from_system": True,
        },
    }
    if prior is None:
        start_child_actions = v02_u2r.SOURCE_CHILD_ACTIONS
        start_remediation_actions = 0
        start_lifetime_actions = v02_u2r.SOURCE_LIFETIME_ACTIONS
        resume_checkpoint = None
        resume_checkpoint_sha256 = None
        resume_model_state = None
        resume_source_segment_index = None
        inherited_active_workers: list[dict[str, object]] = []
        initial_model_state = {
            "policy_tensor_sha256": v02_u2r.SOURCE_POLICY_TENSOR_SHA256,
            "optimizer_state_sha256": v02_u2r.SOURCE_OPTIMIZER_STATE_SHA256,
        }
    else:
        prior_status = json.loads((prior / "status.json").read_text(encoding="utf-8"))
        prior_checkpoint = (prior / str(prior_status["latest_safe_checkpoint"])).resolve()
        prior_sidecar = json.loads(
            prior_checkpoint.with_suffix(".json").read_text(encoding="utf-8")
        )
        prior_progress = prior_sidecar["progress"]
        start_child_actions = int(prior_progress["child_trained_actions"])
        start_remediation_actions = int(prior_progress["remediation_trained_actions"])
        start_lifetime_actions = int(prior_progress["lifetime_trained_actions"])
        resume_checkpoint = str(prior_checkpoint)
        resume_checkpoint_sha256 = hashlib.sha256(prior_checkpoint.read_bytes()).hexdigest()
        resume_model_state = dict(prior_sidecar["model_state"])
        resume_source_segment_index = index - 1
        initial_model_state = dict(resume_model_state)
        prior_interruption_path = prior / "interruption.json"
        prior_interruption = json.loads(prior_interruption_path.read_text(encoding="utf-8"))
        abandoned_at = "2026-07-23T02:00:00+00:00"
        inherited_active_workers = [
            {
                **worker,
                "abandoned_on_resume": True,
                "abandoned_at": abandoned_at,
                "abandonment_reason": v02_u2r.RESUME_ABANDONMENT_REASON,
                "active_worker_evidence_source": "interruption.json",
                "source_interruption": str(prior_interruption_path.resolve()),
                "source_interruption_sha256": hashlib.sha256(
                    prior_interruption_path.read_bytes()
                ).hexdigest(),
                "source_interruption_active_workers_sha256": (
                    prior_interruption["active_workers_sha256"]
                ),
            }
            for worker in prior_interruption["active_workers"]
        ]
    segment_identity = {
        "index": index,
        "lineage_source_child_seed": v02_u2r.SOURCE_CHILD_SEED,
        "initial_algorithm_seed": v02_u2r.REMEDIATION_ALGORITHM_SEED,
        "initial_worker_streams": list(v02_u2r.REMEDIATION_WORKER_STREAMS),
        "segment_algorithm_seed": (
            v02_u2r.REMEDIATION_ALGORITHM_SEED + index * v02_u2r.SEGMENT_SEED_OFFSET
        ),
        "segment_worker_streams": [
            seed + index * v02_u2r.SEGMENT_SEED_OFFSET
            for seed in v02_u2r.REMEDIATION_WORKER_STREAMS
        ],
        "start_child_trained_actions": start_child_actions,
        "start_remediation_trained_actions": start_remediation_actions,
        "start_lifetime_trained_actions": start_lifetime_actions,
        "resume_checkpoint": resume_checkpoint,
        "resume_checkpoint_sha256": resume_checkpoint_sha256,
        "resume_model_state": resume_model_state,
        "resume_source_segment_index": resume_source_segment_index,
        "inherited_active_workers": inherited_active_workers,
        "inherited_active_workers_abandoned": bool(inherited_active_workers),
    }
    carried: list[dict[str, object]] = []
    exam_records: list[dict[str, object]] = []
    for boundary in range(
        v02_u2r.EVALUATION_INTERVAL,
        remediation_actions + 1,
        v02_u2r.EVALUATION_INTERVAL,
    ):
        boundary_child_actions = v02_u2r.SOURCE_CHILD_ACTIONS + boundary
        exam = checkpoints / "rolling" / f"exam-{boundary_child_actions:07d}.zip"
        exam.parent.mkdir(parents=True, exist_ok=True)
        exam.write_bytes(f"exam-{index}-{boundary}".encode())
        exam_digest = hashlib.sha256(exam.read_bytes()).hexdigest()
        exam_sidecar = exam.with_suffix(".json")
        exam_sidecar.write_text(
            json.dumps(
                {
                    "schema_version": v02_u2r.CHECKPOINT_SCHEMA_VERSION,
                    "protocol": v02_u2r.PROTOCOL,
                    "kind": "exam",
                    "resume_eligible": False,
                    "checkpoint_sha256": exam_digest,
                    "source": {"dirty": False, "commit": commit},
                    **{
                        key: frozen_identity[key]
                        for key in (
                            "effective_config",
                            "parent",
                            "qualification",
                            "exclusions",
                            "external_preregistration",
                            "storage_mount",
                        )
                    },
                    "progress": {"child_trained_actions": boundary_child_actions},
                }
            ),
            encoding="utf-8",
        )
        exam_sidecar_digest = hashlib.sha256(exam_sidecar.read_bytes()).hexdigest()
        exam_integrity = exam.with_suffix(".integrity.json")
        exam_integrity.write_text(
            json.dumps(
                {
                    "schema_version": v02_u2r.CHECKPOINT_SCHEMA_VERSION,
                    "protocol": v02_u2r.PROTOCOL,
                    "checkpoint": exam.name,
                    "checkpoint_sha256": exam_digest,
                    "sidecar": exam_sidecar.name,
                    "sidecar_sha256": exam_sidecar_digest,
                }
            ),
            encoding="utf-8",
        )
        evaluations, records = _fixture_exam_cases(boundary)
        summary = v02_u2r.verify_exam_case_evidence(
            evaluations,
            records,
            expected_seeds=None,
        )
        cases = exam.with_suffix(".cases.json")
        cases.write_text(
            json.dumps(
                {
                    "schema_version": v02_u2r.CASE_EVIDENCE_SCHEMA_VERSION,
                    "protocol": v02_u2r.PROTOCOL,
                    "kind": "immutable_exam_cases",
                    "checkpoint": exam.name,
                    "checkpoint_sha256": exam_digest,
                    "checkpoint_sidecar_sha256": exam_sidecar_digest,
                    "child_trained_actions": boundary_child_actions,
                    "summary": summary,
                    "records": records,
                }
            ),
            encoding="utf-8",
        )
        case_digest = hashlib.sha256(cases.read_bytes()).hexdigest()
        binding = summary | {
            "path": str(cases.relative_to(segment)),
            "file_sha256": case_digest,
            "checkpoint": str(exam.relative_to(segment)),
            "checkpoint_sha256": exam_digest,
            "checkpoint_sidecar_sha256": exam_sidecar_digest,
            "child_trained_actions": boundary_child_actions,
        }
        exam_records.append(
            {
                "child_trained_actions": boundary_child_actions,
                "remediation_trained_actions": boundary,
                "allocation_valid": True,
                "recovery": False,
                "evaluations": evaluations,
                "case_evidence": binding,
            }
        )
        if boundary <= start_remediation_actions:
            carried.append(
                {
                    "path": str(exam.relative_to(segment)),
                    "child_trained_actions": boundary_child_actions,
                    "checkpoint_sha256": exam_digest,
                    "sidecar_sha256": exam_sidecar_digest,
                    "integrity_sha256": hashlib.sha256(exam_integrity.read_bytes()).hexdigest(),
                    "case_evidence_sha256": case_digest,
                    "case_set_sha256": summary["case_set_sha256"],
                }
            )
    manifest = {
        "schema_version": v02_u2r.CHECKPOINT_SCHEMA_VERSION,
        "protocol": v02_u2r.PROTOCOL,
        "source": {"dirty": False, "commit": commit},
        **frozen_identity,
        "segment": segment_identity,
        "carried_exams": carried,
    }
    (segment / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    def write_checkpoint(
        filename: str,
        *,
        kind: str,
        progress_remediation_actions: int,
        model_state: dict[str, str],
        controller_records: list[dict[str, object]],
    ) -> tuple[Path, str]:
        checkpoint = checkpoints / filename
        checkpoint.write_bytes(f"{filename}-{index}-{progress_remediation_actions}".encode())
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        progress_child_actions = v02_u2r.SOURCE_CHILD_ACTIONS + progress_remediation_actions
        progress_lifetime_actions = v02_u2r.SOURCE_LIFETIME_ACTIONS + progress_remediation_actions
        progress_optimizer_updates = (
            v02_u2r.SOURCE_OPTIMIZER_UPDATES
            + progress_remediation_actions // v02_u2r.ROLLOUT_TRANSITIONS * v02_u2r.PPO_EPOCHS
        )
        sidecar = checkpoint.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": v02_u2r.CHECKPOINT_SCHEMA_VERSION,
                    "protocol": v02_u2r.PROTOCOL,
                    "kind": kind,
                    "resume_eligible": True,
                    "checkpoint_sha256": checkpoint_sha256,
                    "source": {"dirty": False, "commit": commit},
                    **{
                        key: frozen_identity[key]
                        for key in (
                            "effective_config",
                            "parent",
                            "qualification",
                            "exclusions",
                            "external_preregistration",
                            "storage_mount",
                        )
                    },
                    "model_state": model_state,
                    "curriculum": {"mastered": False},
                    "controller": {
                        "terminal_eligible": None,
                        "exam_records": controller_records,
                    },
                    "progress": {
                        "child_trained_actions": progress_child_actions,
                        "remediation_trained_actions": (progress_remediation_actions),
                        "lifetime_trained_actions": progress_lifetime_actions,
                        "collected_actions": progress_lifetime_actions,
                        "trained_actions": progress_lifetime_actions,
                        "remaining_remediation_actions": (
                            v02_u2r.ADDITIONAL_ACTION_BUDGET - progress_remediation_actions
                        ),
                        "optimizer_updates": progress_optimizer_updates,
                    },
                    "segment": segment_identity,
                }
            ),
            encoding="utf-8",
        )
        integrity = checkpoint.with_suffix(".integrity.json")
        integrity.write_text(
            json.dumps(
                {
                    "schema_version": v02_u2r.CHECKPOINT_SCHEMA_VERSION,
                    "protocol": v02_u2r.PROTOCOL,
                    "checkpoint": checkpoint.name,
                    "checkpoint_sha256": checkpoint_sha256,
                    "sidecar": sidecar.name,
                    "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        return checkpoint, checkpoint_sha256

    initial_records = [
        record
        for record in exam_records
        if int(record["remediation_trained_actions"]) <= start_remediation_actions
    ]
    initial_checkpoint, initial_checkpoint_sha256 = write_checkpoint(
        "initial.zip",
        kind="initial",
        progress_remediation_actions=start_remediation_actions,
        model_state=initial_model_state,
        controller_records=initial_records,
    )
    if index == 0:
        baseline_evaluations, baseline_records = _fixture_exam_cases(0)
        baseline_summary = v02_u2r.verify_exam_case_evidence(
            baseline_evaluations,
            baseline_records,
            expected_seeds=None,
        )
        initial_sidecar_path = initial_checkpoint.with_suffix(".json")
        initial_sidecar_sha256 = hashlib.sha256(initial_sidecar_path.read_bytes()).hexdigest()
        baseline_cases_path = initial_checkpoint.with_suffix(".cases.json")
        baseline_cases_path.write_text(
            json.dumps(
                {
                    "schema_version": v02_u2r.CASE_EVIDENCE_SCHEMA_VERSION,
                    "protocol": v02_u2r.PROTOCOL,
                    "kind": "immutable_exam_cases",
                    "checkpoint": initial_checkpoint.name,
                    "checkpoint_sha256": initial_checkpoint_sha256,
                    "checkpoint_sidecar_sha256": initial_sidecar_sha256,
                    "child_trained_actions": v02_u2r.SOURCE_CHILD_ACTIONS,
                    "summary": baseline_summary,
                    "records": baseline_records,
                }
            ),
            encoding="utf-8",
        )
        baseline_binding = baseline_summary | {
            "path": str(baseline_cases_path.relative_to(segment)),
            "file_sha256": hashlib.sha256(baseline_cases_path.read_bytes()).hexdigest(),
            "checkpoint": str(initial_checkpoint.relative_to(segment)),
            "checkpoint_sha256": initial_checkpoint_sha256,
            "checkpoint_sidecar_sha256": initial_sidecar_sha256,
            "child_trained_actions": v02_u2r.SOURCE_CHILD_ACTIONS,
        }
        evaluations_path = segment / "evaluations.jsonl"
        evaluations_path.write_text(
            "".join(
                json.dumps(
                    {
                        "trigger": "diagnostic_parent_baseline",
                        "counts_toward_gate": False,
                        "trained_actions": v02_u2r.SOURCE_LIFETIME_ACTIONS,
                        "child_trained_actions": v02_u2r.SOURCE_CHILD_ACTIONS,
                        "remediation_trained_actions": 0,
                        "optimizer_updates": v02_u2r.SOURCE_OPTIMIZER_UPDATES,
                        "checkpoint": str(initial_checkpoint.relative_to(segment)),
                        "checkpoint_sha256": initial_checkpoint_sha256,
                        "sidecar": str(initial_sidecar_path.relative_to(segment)),
                        "sidecar_sha256": initial_sidecar_sha256,
                        "case_evidence": baseline_binding,
                        "reloaded_for_grading": True,
                        **evaluation,
                    }
                )
                + "\n"
                for evaluation in baseline_evaluations.values()
            ),
            encoding="utf-8",
        )
    latest_model_state = (
        initial_model_state
        if remediation_actions == start_remediation_actions
        else {
            "policy_tensor_sha256": _sha(f"policy-{index}-{remediation_actions}"),
            "optimizer_state_sha256": _sha(f"optimizer-{index}-{remediation_actions}"),
        }
    )
    checkpoint, checkpoint_sha256 = write_checkpoint(
        "latest-safe.zip",
        kind="latest",
        progress_remediation_actions=remediation_actions,
        model_state=latest_model_state,
        controller_records=exam_records,
    )
    interruption_binding = None
    if phase in {"interrupted", "crashed"}:
        interruption_time = "2026-07-23T03:00:00+00:00"
        workers = [
            {
                "worker_index": worker_index,
                "worker_stream": (
                    v02_u2r.REMEDIATION_WORKER_STREAMS[worker_index]
                    + index * v02_u2r.SEGMENT_SEED_OFFSET
                ),
                "checkpoint_source_commit": commit,
                "checkpoint_lifetime_trained_actions": lifetime_actions,
                "checkpoint_child_trained_actions": child_actions,
                "episode_ordinal": worker_index + 1,
                "seed": 100 + index * 10 + worker_index,
                "lesson_id": lessons.LessonId.SEPARATED_UNLOCK.value,
                "layout_sha256": _sha(f"layout-{index}-{worker_index}"),
                "geometry_sha256": _sha(f"geometry-{index}-{worker_index}"),
                "elapsed_steps": worker_index + 2,
                "active": True,
                "worker_transition_at_start": worker_index * 10,
                "diagnostics": {},
                "seed_role": "training",
                "reset_provenance": "curriculum_training_rng",
                "generator_profile_version": (lessons.GENERATOR_PROFILE_VERSION),
                "generator_profile": lessons.GENERATOR_PROFILE,
                "episode_started_at": (f"2026-07-23T03:0{worker_index}:00+00:00"),
                "untrained_transitions": 0,
                "exact_environment_resume_supported": False,
                "environment_state_disposition": ("discarded_on_process_resume"),
                "environment_rng_state_disposition": (
                    "discarded_on_process_resume; next segment uses a frozen fresh stream"
                ),
                "policy_recurrent_state_disposition": ("discarded_on_process_resume"),
            }
            for worker_index in range(v02_u2r.WORKERS)
        ]
        ledger_path = segment / "episode-starts.jsonl"
        ledger_records = [
            {
                "timestamp": record["abandoned_at"],
                "type": "episode_abandoned_on_resume",
                **record,
            }
            for record in inherited_active_workers
        ]
        for worker in workers:
            ledger_records.extend(
                [
                    {
                        "timestamp": worker["episode_started_at"],
                        "type": "episode_start",
                        **worker,
                    },
                    {
                        "timestamp": interruption_time,
                        "type": "episode_abandoned_at_interruption",
                        "interruption_phase": phase,
                        **worker,
                    },
                ]
            )
        ledger_path.write_text(
            "".join(json.dumps(record) + "\n" for record in ledger_records),
            encoding="utf-8",
        )
        latest_sidecar_path = checkpoint.with_suffix(".json")
        latest_integrity_path = checkpoint.with_suffix(".integrity.json")
        latest_sidecar = json.loads(latest_sidecar_path.read_text(encoding="utf-8"))
        interruption_path = segment / "interruption.json"
        interruption_path.write_text(
            json.dumps(
                {
                    "schema_version": v02_u2r.CHECKPOINT_SCHEMA_VERSION,
                    "protocol": v02_u2r.PROTOCOL,
                    "kind": "process_interruption_evidence",
                    "phase": phase,
                    "resume_permitted": True,
                    "resume_blocker": None,
                    "source": {"dirty": False, "commit": commit},
                    **{
                        key: frozen_identity[key]
                        for key in (
                            "effective_config",
                            "parent",
                            "qualification",
                            "exclusions",
                            "external_preregistration",
                            "storage_mount",
                        )
                    },
                    "segment": segment_identity,
                    "at_signal": {
                        "timestamp": interruption_time,
                        "collected_actions": lifetime_actions,
                        "trained_actions": lifetime_actions,
                        "child_trained_actions": child_actions,
                        "optimizer_updates": optimizer_updates,
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
                        "resolved_trained_actions": lifetime_actions,
                        "resolved_child_trained_actions": child_actions,
                        "resolved_optimizer_updates": optimizer_updates,
                        "safe_boundary_trained_actions": lifetime_actions,
                        "active_environment_state_disposition": ("discarded_at_process_boundary"),
                        "policy_recurrent_state_disposition": ("discarded_at_process_boundary"),
                    },
                    "active_workers": workers,
                    "active_workers_sha256": _canonical_sha(workers),
                    "episode_start_ledger": {
                        "path": ledger_path.name,
                        "sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                        "record_count": len(ledger_records),
                        "episode_start_count": v02_u2r.WORKERS,
                        "interruption_abandonment_count": v02_u2r.WORKERS,
                        "resume_abandonment_count": len(inherited_active_workers),
                    },
                    "resume_boundary": {
                        "checkpoint": str(checkpoint.resolve()),
                        "checkpoint_sha256": checkpoint_sha256,
                        "sidecar": str(latest_sidecar_path.resolve()),
                        "sidecar_sha256": hashlib.sha256(
                            latest_sidecar_path.read_bytes()
                        ).hexdigest(),
                        "integrity": str(latest_integrity_path.resolve()),
                        "integrity_sha256": hashlib.sha256(
                            latest_integrity_path.read_bytes()
                        ).hexdigest(),
                        "model_state": latest_sidecar["model_state"],
                        "progress": latest_sidecar["progress"],
                    },
                }
            ),
            encoding="utf-8",
        )
        interruption_integrity_path = segment / "interruption.integrity.json"
        interruption_integrity_path.write_text(
            json.dumps(
                {
                    "schema_version": v02_u2r.CHECKPOINT_SCHEMA_VERSION,
                    "protocol": v02_u2r.PROTOCOL,
                    "interruption": interruption_path.name,
                    "interruption_sha256": hashlib.sha256(
                        interruption_path.read_bytes()
                    ).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        interruption_binding = {
            "path": interruption_path.name,
            "sha256": hashlib.sha256(interruption_path.read_bytes()).hexdigest(),
            "integrity": interruption_integrity_path.name,
            "integrity_sha256": hashlib.sha256(
                interruption_integrity_path.read_bytes()
            ).hexdigest(),
            "phase": phase,
            "active_workers_sha256": _canonical_sha(workers),
        }
    (segment / "status.json").write_text(
        json.dumps(
            {
                "protocol": v02_u2r.PROTOCOL,
                "phase": phase,
                "source": {"dirty": False, "commit": commit},
                "latest_safe_checkpoint": str(checkpoint.relative_to(segment)),
                "latest_safe_checkpoint_sha256": checkpoint_sha256,
                "child_trained_actions": child_actions,
                "remediation_trained_actions": remediation_actions,
                "lifetime_trained_actions": lifetime_actions,
                "optimizer_updates": optimizer_updates,
                "interruption": interruption_binding,
            }
        ),
        encoding="utf-8",
    )
    return segment


def _refresh_interruption_binding(segment: Path) -> None:
    status_path = segment / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    interruption_path = segment / "interruption.json"
    interruption = json.loads(interruption_path.read_text(encoding="utf-8"))
    checkpoint = segment / str(status["latest_safe_checkpoint"])
    sidecar_path = checkpoint.with_suffix(".json")
    integrity_path = checkpoint.with_suffix(".integrity.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    interruption["resume_boundary"] = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "sidecar": str(sidecar_path.resolve()),
        "sidecar_sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
        "integrity": str(integrity_path.resolve()),
        "integrity_sha256": hashlib.sha256(integrity_path.read_bytes()).hexdigest(),
        "model_state": sidecar["model_state"],
        "progress": sidecar["progress"],
    }
    interruption_path.write_text(
        json.dumps(interruption),
        encoding="utf-8",
    )
    interruption_integrity_path = segment / "interruption.integrity.json"
    interruption_integrity = json.loads(interruption_integrity_path.read_text(encoding="utf-8"))
    interruption_integrity["interruption_sha256"] = hashlib.sha256(
        interruption_path.read_bytes()
    ).hexdigest()
    interruption_integrity_path.write_text(
        json.dumps(interruption_integrity),
        encoding="utf-8",
    )
    status["interruption"]["sha256"] = hashlib.sha256(interruption_path.read_bytes()).hexdigest()
    status["interruption"]["integrity_sha256"] = hashlib.sha256(
        interruption_integrity_path.read_bytes()
    ).hexdigest()
    status_path.write_text(json.dumps(status), encoding="utf-8")


def _refresh_interruption_ledger_binding(segment: Path) -> None:
    interruption_path = segment / "interruption.json"
    interruption = json.loads(interruption_path.read_text(encoding="utf-8"))
    ledger_path = segment / "episode-starts.jsonl"
    records = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    interruption["episode_start_ledger"] = {
        "path": ledger_path.name,
        "sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "record_count": len(records),
        "episode_start_count": sum(record.get("type") == "episode_start" for record in records),
        "interruption_abandonment_count": sum(
            record.get("type") == "episode_abandoned_at_interruption" for record in records
        ),
        "resume_abandonment_count": sum(
            record.get("type") == "episode_abandoned_on_resume" for record in records
        ),
    }
    interruption_path.write_text(
        json.dumps(interruption),
        encoding="utf-8",
    )
    _refresh_interruption_binding(segment)


def _rewrite_segment_identity(
    segment: Path,
    updates: dict[str, object],
) -> None:
    manifest_path = segment / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["segment"].update(updates)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for filename in ("initial.zip", "latest-safe.zip"):
        checkpoint = segment / "checkpoints" / filename
        sidecar_path = checkpoint.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["segment"].update(updates)
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        integrity_path = checkpoint.with_suffix(".integrity.json")
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        integrity["sidecar_sha256"] = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
        integrity_path.write_text(json.dumps(integrity), encoding="utf-8")
    interruption_path = segment / "interruption.json"
    if interruption_path.is_file():
        interruption = json.loads(interruption_path.read_text(encoding="utf-8"))
        interruption["segment"].update(updates)
        interruption_path.write_text(
            json.dumps(interruption),
            encoding="utf-8",
        )
        _refresh_interruption_binding(segment)


def test_resume_planner_selects_only_contiguous_verified_tip(
    tmp_path: Path,
) -> None:
    root = tmp_path / "u2r"
    root.mkdir()
    first = _write_resume_segment(root, index=0, remediation_actions=32_768)
    _write_resume_segment(
        root,
        index=1,
        prior=first,
        remediation_actions=65_536,
    )

    plan = anchor.select_resume_plan(
        root,
        expected_source_commit="a" * 40,
    )
    assert plan.source_segment_index == 1
    assert plan.next_segment_index == 2
    assert plan.next_run_name == f"{v02_u2r.DEFAULT_RUN_NAME}-resume-2"
    assert plan.checkpoint.endswith(
        f"{v02_u2r.DEFAULT_RUN_NAME}-resume-1/checkpoints/latest-safe.zip"
    )
    assert plan.remediation_trained_actions == 65_536
    assert plan.child_trained_actions == 753_664
    assert plan.lifetime_trained_actions == 1_540_096
    assert plan.optimizer_updates == 3_008


def test_resume_planner_requires_episode_origins_and_successor_abandonment(
    tmp_path: Path,
) -> None:
    missing_start_root = tmp_path / "missing-start"
    missing_start_root.mkdir()
    interrupted = _write_resume_segment(
        missing_start_root,
        index=0,
    )
    ledger_path = interrupted / "episode-starts.jsonl"
    records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    removed = False
    retained = []
    for record in records:
        if not removed and record.get("type") == "episode_start":
            removed = True
            continue
        retained.append(record)
    ledger_path.write_text(
        "".join(json.dumps(record) + "\n" for record in retained),
        encoding="utf-8",
    )
    _refresh_interruption_ledger_binding(interrupted)
    with pytest.raises(
        anchor.U2rAnchorError,
        match="full resume semantics",
    ):
        anchor.select_resume_plan(
            missing_start_root,
            expected_source_commit="a" * 40,
        )

    missing_abandonment_root = tmp_path / "missing-abandonment"
    missing_abandonment_root.mkdir()
    first = _write_resume_segment(
        missing_abandonment_root,
        index=0,
    )
    second = _write_resume_segment(
        missing_abandonment_root,
        index=1,
        prior=first,
        remediation_actions=65_536,
    )
    second_ledger = second / "episode-starts.jsonl"
    second_records = [
        json.loads(line) for line in second_ledger.read_text(encoding="utf-8").splitlines()
    ]
    removed = False
    retained = []
    for record in second_records:
        if not removed and record.get("type") == "episode_abandoned_on_resume":
            removed = True
            continue
        retained.append(record)
    second_ledger.write_text(
        "".join(json.dumps(record) + "\n" for record in retained),
        encoding="utf-8",
    )
    _refresh_interruption_ledger_binding(second)
    with pytest.raises(
        anchor.U2rAnchorError,
        match="inherited-worker abandonment",
    ):
        anchor.select_resume_plan(
            missing_abandonment_root,
            expected_source_commit="a" * 40,
        )


def test_resume_planner_refuses_gaps_ambiguity_terminal_and_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "gap"
    root.mkdir()
    _write_resume_segment(root, index=0)
    _write_resume_segment(root, index=2)
    with pytest.raises(anchor.U2rAnchorError, match="not contiguous"):
        anchor.select_resume_plan(root, expected_source_commit="a" * 40)

    ambiguous = tmp_path / "ambiguous"
    ambiguous.mkdir()
    _write_resume_segment(ambiguous, index=0)
    (ambiguous / f"{v02_u2r.DEFAULT_RUN_NAME}-resume-final").mkdir()
    with pytest.raises(anchor.U2rAnchorError, match="unrecognized"):
        anchor.select_resume_plan(
            ambiguous,
            expected_source_commit="a" * 40,
        )

    terminal = tmp_path / "terminal"
    terminal.mkdir()
    _write_resume_segment(terminal, index=0, phase="eligible")
    with pytest.raises(anchor.U2rAnchorError, match="interrupted/crashed"):
        anchor.select_resume_plan(terminal, expected_source_commit="a" * 40)

    tampered = tmp_path / "tampered"
    tampered.mkdir()
    segment = _write_resume_segment(tampered, index=0)
    (segment / "checkpoints" / "latest-safe.zip").write_bytes(b"changed")
    with pytest.raises(anchor.U2rAnchorError, match="digest changed"):
        anchor.select_resume_plan(tampered, expected_source_commit="a" * 40)


def test_resume_planner_refuses_symlinked_segment_or_checkpoint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "symlink"
    root.mkdir()
    real = tmp_path / "real-segment"
    real.mkdir()
    (root / v02_u2r.DEFAULT_RUN_NAME).symlink_to(
        real,
        target_is_directory=True,
    )
    with pytest.raises(anchor.U2rAnchorError, match="missing or unsafe"):
        anchor.select_resume_plan(root, expected_source_commit="a" * 40)

    root2 = tmp_path / "checkpoint-link"
    root2.mkdir()
    segment = _write_resume_segment(root2, index=0)
    checkpoint = segment / "checkpoints" / "latest-safe.zip"
    target = tmp_path / "real.zip"
    target.write_bytes(checkpoint.read_bytes())
    checkpoint.unlink()
    checkpoint.symlink_to(target)
    with pytest.raises(anchor.U2rAnchorError, match="missing or unsafe"):
        anchor.select_resume_plan(root2, expected_source_commit="a" * 40)


def test_resume_planner_authenticates_link_anchor_and_carried_exam_order(
    tmp_path: Path,
) -> None:
    broken_link = tmp_path / "broken-link"
    broken_link.mkdir()
    first = _write_resume_segment(
        broken_link,
        index=0,
        remediation_actions=32_768,
    )
    second = _write_resume_segment(
        broken_link,
        index=1,
        prior=first,
        remediation_actions=65_536,
    )
    manifest_path = second / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["segment"]["resume_checkpoint"] = str(tmp_path / "forged.zip")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint = second / "checkpoints" / "latest-safe.zip"
    sidecar_path = checkpoint.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["segment"]["resume_checkpoint"] = str(tmp_path / "forged.zip")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    integrity_path = checkpoint.with_suffix(".integrity.json")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity["sidecar_sha256"] = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    integrity_path.write_text(json.dumps(integrity), encoding="utf-8")
    initial = second / "checkpoints" / "initial.zip"
    initial_sidecar_path = initial.with_suffix(".json")
    initial_sidecar = json.loads(initial_sidecar_path.read_text(encoding="utf-8"))
    initial_sidecar["segment"]["resume_checkpoint"] = str(tmp_path / "forged.zip")
    initial_sidecar_path.write_text(
        json.dumps(initial_sidecar),
        encoding="utf-8",
    )
    initial_integrity_path = initial.with_suffix(".integrity.json")
    initial_integrity = json.loads(initial_integrity_path.read_text(encoding="utf-8"))
    initial_integrity["sidecar_sha256"] = hashlib.sha256(
        initial_sidecar_path.read_bytes()
    ).hexdigest()
    initial_integrity_path.write_text(
        json.dumps(initial_integrity),
        encoding="utf-8",
    )
    _rewrite_segment_identity(
        second,
        {"resume_checkpoint": str(tmp_path / "forged.zip")},
    )
    with pytest.raises(
        anchor.U2rAnchorError,
        match=r"not continuous|full resume semantics",
    ):
        anchor.select_resume_plan(
            broken_link,
            expected_source_commit="a" * 40,
        )

    anchor_drift = tmp_path / "anchor-drift"
    anchor_drift.mkdir()
    segment = _write_resume_segment(anchor_drift, index=0)
    manifest_path = segment / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["external_preregistration"]["tag_object"] = "f" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(anchor.U2rAnchorError, match="sidecar is not resumable"):
        anchor.select_resume_plan(
            anchor_drift,
            expected_source_commit="a" * 40,
        )

    carried_order = tmp_path / "carried-order"
    carried_order.mkdir()
    first = _write_resume_segment(
        carried_order,
        index=0,
        remediation_actions=65_536,
    )
    second = _write_resume_segment(
        carried_order,
        index=1,
        prior=first,
        remediation_actions=98_304,
    )
    manifest_path = second / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["carried_exams"].reverse()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(anchor.U2rAnchorError, match="path identity changed"):
        anchor.select_resume_plan(
            carried_order,
            expected_source_commit="a" * 40,
        )


def test_resume_planner_binds_controller_exam_sequence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "controller"
    root.mkdir()
    segment = _write_resume_segment(
        root,
        index=0,
        remediation_actions=65_536,
    )
    checkpoint = segment / "checkpoints" / "latest-safe.zip"
    sidecar_path = checkpoint.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["controller"]["exam_records"][1]["child_trained_actions"] += 1
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    integrity_path = checkpoint.with_suffix(".integrity.json")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity["sidecar_sha256"] = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    integrity_path.write_text(json.dumps(integrity), encoding="utf-8")
    with pytest.raises(anchor.U2rAnchorError, match="exam sequence"):
        anchor.select_resume_plan(root, expected_source_commit="a" * 40)


def test_resume_planner_recomputes_raw_cases_and_rejects_missing_evidence(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing-cases"
    missing_root.mkdir()
    missing_segment = _write_resume_segment(
        missing_root,
        index=0,
        remediation_actions=32_768,
    )
    missing_case = next((missing_segment / "checkpoints" / "rolling").glob("*.cases.json"))
    missing_case.unlink()
    with pytest.raises(anchor.U2rAnchorError, match="missing or unsafe"):
        anchor.select_resume_plan(
            missing_root,
            expected_source_commit="a" * 40,
        )

    tampered_root = tmp_path / "tampered-cases"
    tampered_root.mkdir()
    tampered_segment = _write_resume_segment(
        tampered_root,
        index=0,
        remediation_actions=32_768,
    )
    cases_path = next((tampered_segment / "checkpoints" / "rolling").glob("*.cases.json"))
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    cases["records"][0]["ineffective_interactions"] = 99
    cases_path.write_text(json.dumps(cases), encoding="utf-8")
    latest = tampered_segment / "checkpoints" / "latest-safe.zip"
    sidecar_path = latest.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["controller"]["exam_records"][0]["case_evidence"]["file_sha256"] = hashlib.sha256(
        cases_path.read_bytes()
    ).hexdigest()
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    integrity_path = latest.with_suffix(".integrity.json")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity["sidecar_sha256"] = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    integrity_path.write_text(json.dumps(integrity), encoding="utf-8")
    with pytest.raises(
        anchor.U2rAnchorError,
        match="do not reproduce the controller evaluation",
    ):
        anchor.select_resume_plan(
            tampered_root,
            expected_source_commit="a" * 40,
        )


def test_resume_planner_rejects_multihop_model_and_checkpoint_rebinding(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model-rebinding"
    root.mkdir()
    first = _write_resume_segment(
        root,
        index=0,
        remediation_actions=32_768,
    )
    second = _write_resume_segment(
        root,
        index=1,
        prior=first,
        remediation_actions=65_536,
    )
    forged_model_state = {
        "policy_tensor_sha256": "1" * 64,
        "optimizer_state_sha256": "2" * 64,
    }
    _rewrite_segment_identity(
        second,
        {
            "resume_checkpoint_sha256": "3" * 64,
            "resume_model_state": forged_model_state,
        },
    )
    initial = second / "checkpoints" / "initial.zip"
    initial_sidecar_path = initial.with_suffix(".json")
    initial_sidecar = json.loads(initial_sidecar_path.read_text(encoding="utf-8"))
    initial_sidecar["model_state"] = forged_model_state
    initial_sidecar_path.write_text(
        json.dumps(initial_sidecar),
        encoding="utf-8",
    )
    initial_integrity_path = initial.with_suffix(".integrity.json")
    initial_integrity = json.loads(initial_integrity_path.read_text(encoding="utf-8"))
    initial_integrity["sidecar_sha256"] = hashlib.sha256(
        initial_sidecar_path.read_bytes()
    ).hexdigest()
    initial_integrity_path.write_text(
        json.dumps(initial_integrity),
        encoding="utf-8",
    )
    with pytest.raises(
        anchor.U2rAnchorError,
        match=r"not continuous|full resume semantics",
    ):
        anchor.select_resume_plan(
            root,
            expected_source_commit="a" * 40,
        )


def test_resume_planner_rejects_rehashed_interruption_worker_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interruption-tamper"
    root.mkdir()
    segment = _write_resume_segment(
        root,
        index=0,
        remediation_actions=32_768,
    )
    interruption_path = segment / "interruption.json"
    interruption = json.loads(interruption_path.read_text(encoding="utf-8"))
    interruption["active_workers"][0]["worker_stream"] += 999
    workers_sha256 = _canonical_sha(interruption["active_workers"])
    interruption["active_workers_sha256"] = workers_sha256
    interruption_path.write_text(
        json.dumps(interruption),
        encoding="utf-8",
    )
    interruption_integrity_path = segment / "interruption.integrity.json"
    interruption_integrity = json.loads(interruption_integrity_path.read_text(encoding="utf-8"))
    interruption_integrity["interruption_sha256"] = hashlib.sha256(
        interruption_path.read_bytes()
    ).hexdigest()
    interruption_integrity_path.write_text(
        json.dumps(interruption_integrity),
        encoding="utf-8",
    )
    status_path = segment / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["interruption"]["sha256"] = hashlib.sha256(interruption_path.read_bytes()).hexdigest()
    status["interruption"]["integrity_sha256"] = hashlib.sha256(
        interruption_integrity_path.read_bytes()
    ).hexdigest()
    status["interruption"]["active_workers_sha256"] = workers_sha256
    status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(
        anchor.U2rAnchorError,
        match="active-worker provenance changed",
    ):
        anchor.select_resume_plan(
            root,
            expected_source_commit="a" * 40,
        )


def test_resume_planner_requires_completed_segment_zero_baseline(
    tmp_path: Path,
) -> None:
    root = tmp_path / "incomplete-baseline"
    root.mkdir()
    segment = _write_resume_segment(
        root,
        index=0,
        remediation_actions=0,
    )
    (segment / "evaluations.jsonl").unlink()
    with pytest.raises(
        anchor.U2rAnchorError,
        match="authenticated parent baseline",
    ):
        anchor.select_resume_plan(
            root,
            expected_source_commit="a" * 40,
        )
