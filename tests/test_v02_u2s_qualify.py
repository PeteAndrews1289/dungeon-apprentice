from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v02_u2s
from dungeon_apprentice import v02_u2s_qualify as qualify
from dungeon_apprentice import v02_u2s_smoke as smoke


def _digest(character: str) -> str:
    return character * 64


def _initial_rng_identity() -> dict[str, object]:
    components: dict[str, object] = {
        name: f"{index + 1:064x}"
        for index, name in enumerate(
            v02_u2s.INITIAL_RNG_COMPONENT_DIGESTS
        )
    }
    components["torch_mps_sha256"] = None
    components["torch_cuda_sha256"] = None
    components["workers"] = [
        {
            "worker_index": index,
            "worker_stream": stream,
            **{
                name: f"{20 + index * 3 + offset:064x}"
                for offset, name in enumerate(
                    v02_u2s.INITIAL_RNG_WORKER_COMPONENT_DIGESTS
                )
            },
        }
        for index, stream in enumerate(v02_u2s.WORKER_STREAMS)
    ]
    return {
        "captured_before_action_one": True,
        "algorithm_seed": v02_u2s.ALGORITHM_SEED,
        "components": components,
        "aggregate_sha256": v02_u2s._canonical_sha256(components),
    }


def _terminal_report(root: Path, records: list[dict[str, object]]) -> dict[str, object]:
    active = [
        {key: value for key, value in record.items() if key != "type"}
        for record in records[:4]
    ]
    return {
        "verdict": "failed",
        "lineage_history": {
            "cohort_root": str(root),
            "segments": [
                {
                    "episode_start_ledger": {
                        "path": "run/episode-starts.jsonl",
                        "sha256": "",
                        "record_count": len(records),
                        "episode_start_count": len(records),
                    }
                }
            ],
            "terminal_active_workers": {
                "count": 4,
                "records": active,
                "records_sha256": qualify._canonical_sha256(active),
            },
        },
    }


def test_extend_guards_keeps_u0_development_only(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / "run"
    run.mkdir()
    records = [
        {
            "type": "episode_start",
            "active": True,
            "lesson_id": lesson.value,
            "layout_sha256": _digest(character),
        }
        for lesson, character in zip(
            lessons.LessonId,
            ("5", "6", "7", "8"),
            strict=True,
        )
    ]
    ledger = run / "episode-starts.jsonl"
    ledger.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    ledger_sha256 = hashlib.sha256(ledger.read_bytes()).hexdigest()
    monkeypatch.setattr(qualify, "U2R_EPISODE_STARTS_SHA256", ledger_sha256)
    report = _terminal_report(tmp_path, records)
    monkeypatch.setattr(
        qualify,
        "U2R_TERMINAL_ACTIVE_RECORDS_SHA256",
        report["lineage_history"]["terminal_active_workers"]["records_sha256"],
    )
    report["lineage_history"]["segments"][0]["episode_start_ledger"][
        "sha256"
    ] = ledger_sha256
    base = {
        lesson: frozenset({_digest(str(index))})
        for index, lesson in enumerate(lessons.LessonId, start=1)
    }

    mapping, evidence = qualify.extend_u2s_forbidden_layout_hashes(
        base,
        report,
        run_directory=run,
    )

    assert mapping[lessons.LessonId.VISIBLE_UNLOCK] == base[
        lessons.LessonId.VISIBLE_UNLOCK
    ]
    assert _digest("5") in mapping[lessons.LessonId.NAVIGATE]
    assert _digest("7") in mapping[lessons.LessonId.LOCAL_UNLOCK]
    assert _digest("8") in mapping[lessons.LessonId.SEPARATED_UNLOCK]
    assert evidence["episode_start_records"] == 4
    assert evidence["applied_by_lesson"]["unlock/u0-visible"][
        "rule"
    ] == "development_only_history_overlap_diagnostic"


def test_extend_guards_rejects_a_changed_ledger(
    tmp_path: Path, monkeypatch
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    ledger = run / "episode-starts.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "type": "episode_start",
                "lesson_id": lessons.LessonId.NAVIGATE.value,
                "layout_sha256": _digest("5"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(qualify, "U2R_EPISODE_STARTS_SHA256", _digest("f"))
    records = [
        {
            "type": "episode_start",
            "active": True,
            "lesson_id": lesson.value,
            "layout_sha256": _digest(character),
        }
        for lesson, character in zip(
            lessons.LessonId,
            ("5", "6", "7", "8"),
            strict=True,
        )
    ]
    report = _terminal_report(tmp_path, records)
    monkeypatch.setattr(
        qualify,
        "U2R_TERMINAL_ACTIVE_RECORDS_SHA256",
        report["lineage_history"]["terminal_active_workers"]["records_sha256"],
    )
    report["lineage_history"]["segments"][0]["episode_start_ledger"][
        "sha256"
    ] = _digest("f")
    base = {lesson: frozenset() for lesson in lessons.LessonId}

    with pytest.raises(qualify.U2SQualificationError, match="digest changed"):
        qualify.extend_u2s_forbidden_layout_hashes(
            base,
            report,
            run_directory=run,
        )


def test_arm_contract_freezes_schedule_pixels_and_reward_dominance() -> None:
    contract = qualify._arm_contract()

    assert contract["arm_order"] == [
        "control",
        "conservative",
        "no-effect",
        "combined",
    ]
    assert contract["pixels_only_probe"] == {
        "streaks": [1, 2, 0, 1, 2],
        "expected": [1, 2, 0, 1, 2],
        "passed": True,
    }
    assert contract["conservative_schedule"]["values"] == {
        "child_0": pytest.approx(2.5e-4),
        "child_half": pytest.approx(1.375e-4),
        "child_terminal": pytest.approx(2.5e-5),
    }
    assert contract["reward_dominance"]["passed"] is True
    assert contract["selection"]["ablation_checkpoint_may_be_successor"] is False


def test_verify_source_tag_requires_clean_published_annotated_tag(
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
        outputs = {
            ("rev-parse", "--verify", "HEAD^{commit}"): head,
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
            ("cat-file", "-t", f"refs/tags/{qualify.QUALIFIED_TAG}"): "tag",
            (
                "rev-parse",
                "--verify",
                f"refs/tags/{qualify.QUALIFIED_TAG}",
            ): tag_object,
            (
                "rev-list",
                "-n",
                "1",
                f"refs/tags/{qualify.QUALIFIED_TAG}",
            ): head,
            ("remote", "get-url", qualify.QUALIFIED_REMOTE): (
                qualify.EXPECTED_ORIGIN_URL
            ),
            (
                "ls-remote",
                "--tags",
                qualify.QUALIFIED_REMOTE,
                f"refs/tags/{qualify.QUALIFIED_TAG}",
            ): f"{tag_object}\trefs/tags/{qualify.QUALIFIED_TAG}",
            (
                "for-each-ref",
                "--format=%(contents)",
                f"refs/tags/{qualify.QUALIFIED_TAG}",
            ): tag_message,
        }
        output = outputs[tuple(arguments)]
        return subprocess.CompletedProcess(command, 0, stdout=output + "\n", stderr="")

    source = qualify.verify_source_tag(
        tmp_path,
        expected_source_commit=head,
        expected_tag_object=tag_object,
        expected_tag_payload=payload,
        runner=runner,
    )

    assert source == {
        "commit": head,
        "dirty": False,
        "tag": qualify.QUALIFIED_TAG,
        "tag_object": tag_object,
        "remote": qualify.QUALIFIED_REMOTE,
        "remote_url": qualify.EXPECTED_ORIGIN_URL,
        "tag_payload": payload,
        "tag_payload_sha256": qualify._canonical_sha256(payload),
    }


def test_verify_source_tag_rejects_dirty_tree(tmp_path: Path) -> None:
    def runner(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        arguments = command[1:]
        output = (
            "a" * 40
            if arguments[:2] == ["rev-parse", "--verify"]
            else " M protocol.md"
        )
        return subprocess.CompletedProcess(command, 0, stdout=output + "\n", stderr="")

    with pytest.raises(qualify.U2SQualificationError, match="clean repository"):
        qualify.verify_source_tag(tmp_path, runner=runner)


def test_storage_caps_reuse_frozen_u2_limits() -> None:
    assert qualify._storage_caps() == {
        "per_arm_bytes": 2 * 1024**3,
        "scientific_cohort_bytes": 6 * 1024**3,
        "media_bytes": 10 * 1024**3,
        "combined_bytes": 16 * 1024**3,
    }


def _valid_disposable_smoke_report() -> dict[str, object]:
    trajectory = _digest("1")
    episode_ledger = _digest("2")
    initial_rng_identity = _initial_rng_identity()
    arms = []
    for arm in v02_u2s.ARM_PRIORITY:
        spec = v02_u2s.ARM_SPECS[arm]
        epochs = spec.n_epochs
        learning_rate = (
            v02_u2s.ChildActionLinearSchedule().for_child_actions(
                v02_u2s.ROLLOUT_TRANSITIONS
            )
            if spec.conservative_ppo
            else v02_u2s.ORIGINAL_LEARNING_RATE
        )
        workers = [
            {
                "worker_index": worker,
                "transitions": v02_u2s.ROLLOUT_STEPS,
                "episodes_started": 1,
                "episode_starts_sha256": _digest("3"),
                "action_counts": {"0": v02_u2s.ROLLOUT_STEPS},
                "lesson_transition_counts": {
                    lessons.LessonId.NAVIGATE.value: (
                        v02_u2s.ROLLOUT_STEPS
                    )
                },
                "reward_total_hex": (0.0).hex(),
                "extrinsic_total_hex": (0.0).hex(),
                "curiosity_total_hex": (0.0).hex(),
                "penalty_total_hex": (0.0).hex(),
                "penalty_events": 0,
                "penalty_eligible_events": 0,
                "maximum_episode_penalty_count": 0,
                "trajectory_sha256": _digest("4"),
                "reward_evidence_sha256": _digest("5"),
            }
            for worker in range(v02_u2s.WORKERS)
        ]
        arms.append(
            {
                "arm": arm.value,
                "configuration": spec.public_dict(),
                "initial_rng_identity": initial_rng_identity,
                "parent_copy_sha256": v02_u2s.PARENT_CHECKPOINT_SHA256,
                "before": {
                    "trained_timesteps": v02_u2s.PARENT_LIFETIME_ACTIONS,
                    "optimizer_updates": v02_u2s.PARENT_OPTIMIZER_UPDATES,
                    "policy_tensor_sha256": (
                        v02_u2s.PARENT_POLICY_TENSOR_SHA256
                    ),
                    "optimizer_state_sha256": (
                        v02_u2s.PARENT_OPTIMIZER_STATE_SHA256
                    ),
                    "optimizer_has_state": True,
                },
                "after": {
                    "trained_timesteps": (
                        v02_u2s.PARENT_LIFETIME_ACTIONS
                        + v02_u2s.ROLLOUT_TRANSITIONS
                    ),
                    "optimizer_updates": (
                        v02_u2s.PARENT_OPTIMIZER_UPDATES + epochs
                    ),
                    "policy_tensor_sha256": _digest("6"),
                    "optimizer_state_sha256": _digest("7"),
                    "optimizer_has_state": True,
                },
                "reloaded_matches_after": True,
                "child_actions": v02_u2s.ROLLOUT_TRANSITIONS,
                "rollouts": 1,
                "optimizer": {
                    "epochs_planned": epochs,
                    "epochs_completed": epochs,
                    "epochs_skipped": 0,
                    "target_kl": (
                        float(spec.target_kl).hex()
                        if spec.target_kl is not None
                        else None
                    ),
                    "kl_stop_triggered": False,
                    "approx_kl_hex": (0.001).hex(),
                    "learning_rate_hex": learning_rate.hex(),
                    "expected_learning_rate_hex": learning_rate.hex(),
                },
                "workers": workers,
                "worker_evidence_sha256": _digest("8"),
                "trajectory_sha256": trajectory,
                "episode_ledger": {
                    "records": v02_u2s.WORKERS,
                    "normalized_sha256": episode_ledger,
                    "active_workers": v02_u2s.WORKERS,
                    "active_workers_sha256": _digest("9"),
                },
                "seed_evidence": {
                    "episode_starts": v02_u2s.WORKERS,
                    "minimum_seed": 0,
                    "maximum_seed": 3,
                    "training_range_only": True,
                    "separated_unlock_roles": ["training"],
                    "protected_roles": [],
                    "protected_seed_hits": [],
                    "confirmation_or_final_seed_generated": False,
                },
                "reward_evidence": {
                    "reward_total_hex": (0.0).hex(),
                    "extrinsic_total_hex": (0.0).hex(),
                    "curiosity_total_hex": (0.0).hex(),
                    "penalty_total_hex": (0.0).hex(),
                    "penalty_events": 0,
                    "maximum_episode_penalty_count": 0,
                    "penalty_enabled": spec.no_effect_penalty,
                    "reward_components_reconciled": True,
                },
                "updated_only_in_disposable_copy": True,
                "promotable": False,
                "capability_claim": False,
            }
        )
    return {
        "schema_version": smoke.SCHEMA_VERSION,
        "protocol": smoke.SMOKE_PROTOCOL,
        "kind": "disposable_engineering_integration_only",
        "result": "passed",
        "source": {"commit": "a" * 40, "dirty": False},
        "parent": {
            "checkpoint_sha256": v02_u2s.PARENT_CHECKPOINT_SHA256,
            "policy_tensor_sha256": v02_u2s.PARENT_POLICY_TENSOR_SHA256,
            "optimizer_state_sha256": (
                v02_u2s.PARENT_OPTIMIZER_STATE_SHA256
            ),
            "trained_timesteps": v02_u2s.PARENT_LIFETIME_ACTIONS,
            "optimizer_updates": v02_u2s.PARENT_OPTIMIZER_UPDATES,
        },
        "integration": {
            "workers": v02_u2s.WORKERS,
            "rollout_steps_per_worker": v02_u2s.ROLLOUT_STEPS,
            "transitions_per_arm": v02_u2s.ROLLOUT_TRANSITIONS,
            "arms": [arm.value for arm in v02_u2s.ARM_PRIORITY],
            "algorithm_seed": v02_u2s.ALGORITHM_SEED,
            "worker_streams": list(v02_u2s.WORKER_STREAMS),
            "guard_mapping_sha256": (
                v02_u2s.QUALIFIED_GUARD_MAPPING_SHA256
            ),
            "sampler_preflight_sha256": _digest("0"),
            "matched_initial_model_state": True,
            "matched_initial_rng_identity": True,
            "initial_rng_identity": initial_rng_identity,
            "matched_rollout_trajectory": True,
            "trajectory_sha256": trajectory,
            "episode_ledger_sha256": episode_ledger,
        },
        "arms": arms,
        "canonical_boundaries_before": [
            {"path": "/canonical", "exists": False}
        ],
        "canonical_boundaries_after": [
            {"path": "/canonical", "exists": False}
        ],
        "canonical_or_claim_policy_updates": False,
        "engineering_smoke_copy_updated": True,
        "canonical_roots_untouched": True,
        "disposable_artifacts_removed": True,
        "evaluation_performed": False,
        "promotion_decision_performed": False,
        "promotable": False,
        "capability_claim": False,
    }


def test_disposable_smoke_output_distinguishes_copy_updates_from_claim() -> None:
    report = _valid_disposable_smoke_report()
    stdout = qualify._canonical_json_bytes(report)
    summary = qualify._verify_disposable_smoke_output(stdout)

    assert summary["report"]["canonical_or_claim_policy_updates"] is False
    assert summary["report"]["engineering_smoke_copy_updated"] is True
    assert summary["stdout_sha256"] == hashlib.sha256(stdout).hexdigest()


def test_disposable_smoke_output_rejects_a_canonical_policy_update() -> None:
    report = _valid_disposable_smoke_report()
    report["canonical_or_claim_policy_updates"] = True

    with pytest.raises(
        qualify.U2SQualificationError,
        match="contract changed",
    ):
        qualify._verify_disposable_smoke_output(
            qualify._canonical_json_bytes(report)
        )
