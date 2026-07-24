from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v02_u2s_smoke as frozen_smoke
from dungeon_apprentice import v03_action_effect as v03
from dungeon_apprentice import v03_action_effect_smoke as v03_smoke
from dungeon_apprentice import v03_action_effect_train as trainer
from dungeon_apprentice.action_effect import (
    ACTION_EFFECT_DIM,
    ACTION_EFFECT_KEY,
    IMAGE_KEY,
    ActionEffectMode,
    ActionEffectPredictAdapter,
)
from dungeon_apprentice.artifacts import (
    append_jsonl,
    atomic_write_json,
    file_sha256,
)


def _stable_exam(index: int) -> dict:
    return {
        "child_trained_actions": index * v03.EVALUATION_INTERVAL,
        "allocation_valid": True,
        "practice_profile": "normal",
        "lessons": {
            lesson.value: {
                "successes": 80,
                "panel_successes": [40, 40],
                "mean_ineffective_interactions": 0.0,
            }
            for lesson in lessons.LessonId
        },
        "case_diagnostics": {
            "max_case_ineffective_interactions": 0,
            "cases_with_ineffective_at_least_10": 0,
            "max_repeated_identical_interaction_run": 0,
        },
    }


def _stable_exams() -> list[dict]:
    return [_stable_exam(index) for index in range(1, v03.EXAM_COUNT + 1)]


def _valid_exam_case_evidence() -> tuple[dict[str, dict], list[dict]]:
    records: list[dict] = []
    evaluations: dict[str, dict] = {}
    for lesson in lessons.LessonId:
        lesson_records: list[dict] = []
        for case_index in range(v03.EVALUATION_SEED_COUNT):
            panel = 0 if case_index < 40 else 1
            panel_case_index = case_index % 40
            success = panel_case_index < 36
            oracle_range = lessons.LESSON_SPECS[lesson].oracle_action_range
            oracle_actions = oracle_range[0] if oracle_range is not None else 4
            steps = min(
                oracle_actions + 5,
                lessons.LESSON_SPECS[lesson].max_steps,
            )
            visible_effects = trainer.frozen_u2s.visible_interaction_effect_evidence(
                [],
                [],
                [],
                penalty_enabled=False,
            )
            lesson_records.append(
                {
                    "lesson_id": lesson.value,
                    "lesson_label": lessons.LESSON_SPECS[lesson].label,
                    "case_index": case_index,
                    "panel": panel,
                    "panel_case_index": panel_case_index,
                    "seed": (lessons.LESSON_SPECS[lesson].validation_seed_base + case_index),
                    "layout_sha256": trainer._canonical_sha256(
                        [lesson.value, case_index, "layout"]
                    ),
                    "geometry_sha256": trainer._canonical_sha256(
                        [lesson.value, case_index, "geometry"]
                    ),
                    "success": success,
                    "terminal_reason": ("success" if success else "time_limit"),
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
                    "path_actions_per_oracle_action": (steps / oracle_actions),
                    "action_histogram": {
                        str(action): steps if action == 0 else 0 for action in range(7)
                    },
                    "longest_repeated_action_run": steps,
                    "longest_identical_visible_no_effect_streak": 0,
                    "longest_repeated_identical_interaction_run": 0,
                    "visible_no_effect_transition_count": 0,
                    "visible_interaction_effects": visible_effects,
                    "action_trace_sha256": trainer._canonical_sha256(
                        [lesson.value, case_index, "actions"]
                    ),
                    "visible_no_effect_streak_sha256": (trainer._canonical_sha256([])),
                    "interaction_run_sha256": (trainer._canonical_sha256([])),
                    "extrinsic_return": 1.0 if success else 0.0,
                    "curiosity_return": 0.0,
                    "initial_key_visible": (
                        False if lesson is lessons.LessonId.SEPARATED_UNLOCK else None
                    ),
                    "initial_door_visible": (
                        False if lesson is lessons.LessonId.SEPARATED_UNLOCK else None
                    ),
                    "visibility_stratum": (
                        "key_hidden_door_hidden"
                        if lesson is lessons.LessonId.SEPARATED_UNLOCK
                        else None
                    ),
                }
            )
        result = lessons.aggregate_u2_case_evidence(
            lesson,
            lesson_records,
            timestamp="2026-07-24T00:00:00+00:00",
        )
        evaluations[lesson.value] = trainer.frozen_u2s._augmented_lesson_evidence(
            result,
            lesson_records,
            penalty_enabled=False,
        )
        records.extend(lesson_records)
    return evaluations, records


def test_extract_image_observation_requires_canonical_dict() -> None:
    images = np.zeros((2, 3, 56, 56), dtype=np.uint8)
    contexts = np.zeros((2, ACTION_EFFECT_DIM), dtype=np.float32)
    selected = trainer.extract_image_observation(
        {
            IMAGE_KEY: images,
            ACTION_EFFECT_KEY: contexts,
        },
        worker_index=1,
    )
    assert selected.shape == (3, 56, 56)
    assert selected.dtype == np.uint8
    selected[0, 0, 0] = 1
    assert images[1, 0, 0, 0] == 0

    with pytest.raises(
        trainer.ActionEffectTrainingError,
        match="noncanonical",
    ):
        trainer.extract_image_observation({IMAGE_KEY: images})


def test_dict_rng_snapshot_restore_includes_every_child() -> None:
    space = gym.spaces.Dict(
        {
            IMAGE_KEY: gym.spaces.Box(
                0,
                255,
                shape=(3, 4, 4),
                dtype=np.uint8,
            ),
            ACTION_EFFECT_KEY: gym.spaces.Box(
                0.0,
                1.0,
                shape=(ACTION_EFFECT_DIM,),
                dtype=np.float32,
            ),
        }
    )
    seeded = trainer.seed_space_tree(space, 42)
    assert set(seeded["children"]) == {IMAGE_KEY, ACTION_EFFECT_KEY}
    snapshot = trainer.snapshot_space_rng(space)
    expected = space.sample()
    trainer.restore_space_rng(space, snapshot)
    actual = space.sample()
    assert np.array_equal(expected[IMAGE_KEY], actual[IMAGE_KEY])
    assert np.array_equal(
        expected[ACTION_EFFECT_KEY],
        actual[ACTION_EFFECT_KEY],
    )


class _TraceEnv(gym.Env):
    observation_space = gym.spaces.Box(
        0,
        255,
        shape=(2, 2, 3),
        dtype=np.uint8,
    )
    action_space = gym.spaces.Discrete(7)

    def __init__(self) -> None:
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.zeros((2, 2, 3), dtype=np.uint8), {
            "seed": 17,
            "u2_seed_role": "training",
            "lesson_id": lessons.LessonId.NAVIGATE.value,
            "layout_sha256": "a" * 64,
            "geometry_sha256": "b" * 64,
        }

    def step(self, action):
        self.step_count += 1
        observation = np.full(
            (2, 2, 3),
            self.step_count % 256,
            dtype=np.uint8,
        )
        return (
            observation,
            -0.001,
            False,
            False,
            {
                "lesson_id": lessons.LessonId.NAVIGATE.value,
                "extrinsic_reward": -0.001,
                "curiosity_reward": 0.0,
            },
        )


class _BoundaryTraceEnv(_TraceEnv):
    def __init__(self, terminate_at: int) -> None:
        super().__init__()
        self.terminate_at = int(terminate_at)

    def step(self, action):
        observation, reward, _, truncated, info = super().step(action)
        return (
            observation,
            reward,
            self.step_count == self.terminate_at,
            truncated,
            info,
        )


def test_first_rollout_wrapper_is_bounded_and_deterministic() -> None:
    wrapper = trainer.FirstRolloutEvidenceWrapper(
        _TraceEnv(),
        worker_index=0,
        transition_limit=2,
    )
    wrapper.reset(seed=1)
    wrapper.step(1)
    wrapper.step(2)
    expected = wrapper.public_dict()
    wrapper.step(3)
    assert wrapper.public_dict() == expected
    assert expected["transitions"] == 2
    assert expected["action_counts"]["1"] == 1
    assert expected["action_counts"]["2"] == 1
    assert len(expected["trajectory_sha256"]) == 64
    assert len(expected["reward_evidence_sha256"]) == 64


def test_first_rollout_matches_frozen_smoke_at_terminal_boundary(
    tmp_path: Path,
) -> None:
    production = trainer.FirstRolloutEvidenceWrapper(
        _BoundaryTraceEnv(trainer.ROLLOUT_STEPS),
        worker_index=0,
        transition_limit=trainer.ROLLOUT_STEPS,
    )
    qualification = frozen_smoke.TransitionEvidenceWrapper(
        _BoundaryTraceEnv(trainer.ROLLOUT_STEPS),
        worker_index=0,
    )
    for wrapper in (production, qualification):
        initial, _ = wrapper.reset(seed=1)
        assert initial.shape == (2, 2, 3)
        for transition in range(1, trainer.ROLLOUT_STEPS + 1):
            _, _, terminated, _, _ = wrapper.step(transition % 7)
        assert transition == 512
        assert terminated
        # DummyVecEnv performs this auto-reset immediately after the terminal
        # transition, including when it is exactly the rollout boundary.
        wrapper.reset()

    production_evidence = production.public_dict()
    qualification_evidence = qualification.public_dict()
    identity_keys = (
        "worker_index",
        "transitions",
        "episodes_started",
        "episode_starts_sha256",
        "action_counts",
        "lesson_transition_counts",
        "trajectory_sha256",
        "reward_evidence_sha256",
    )
    assert {key: production_evidence[key] for key in identity_keys} == {
        key: qualification_evidence[key] for key in identity_keys
    }
    assert production_evidence["transitions"] == 512
    assert production_evidence["episodes_started"] == 2

    # Once the one boundary auto-reset has been captured, later resets cannot
    # alter the sealed first-rollout evidence.
    production.reset()
    assert production.public_dict() == production_evidence

    observation = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert trainer._observation_sha256(observation) == frozen_smoke._observation_sha256(observation)
    assert trainer._qualification_canonical_sha256(
        production.resets
    ) == frozen_smoke._canonical_sha256(qualification.resets)
    assert trainer._normalized_trajectory_identity(
        [production_evidence]
    ) == v03_smoke._trajectory_identity([qualification_evidence])

    ledger = tmp_path / "episode-starts.jsonl"
    append_jsonl(
        ledger,
        {
            "timestamp": "first",
            "worker_index": 0,
            "seed": 17,
        },
    )
    append_jsonl(
        ledger,
        {
            "timestamp": "second",
            "worker_index": 0,
            "seed": 18,
        },
    )
    normalized = frozen_smoke._read_normalized_episode_ledger(ledger)
    assert trainer._qualification_episode_ledger_sha256(ledger) == frozen_smoke._canonical_sha256(
        normalized
    )


def test_context_metrics_separates_activation_effect_and_action() -> None:
    metrics = trainer.ContextMetrics()
    metrics.observe(
        action=4,
        lesson_id=lessons.LessonId.LOCAL_UNLOCK.value,
        changed=False,
        active=True,
    )
    metrics.observe(
        action=4,
        lesson_id=lessons.LessonId.LOCAL_UNLOCK.value,
        changed=True,
        active=False,
    )
    public = metrics.public_dict()
    assert public["transitions"] == 2
    assert public["activation_rate"] == 0.5
    assert public["changed_rate"] == 0.5
    assert public["by_action"]["4"] == {
        "changed": 1,
        "unchanged": 1,
    }


def test_encoder_metrics_reports_exact_update_norms() -> None:
    torch = pytest.importorskip("torch")
    linear = torch.nn.Linear(
        ACTION_EFFECT_DIM,
        512,
        bias=False,
    )
    torch.nn.init.zeros_(linear.weight)
    model = SimpleNamespace(
        policy=SimpleNamespace(features_extractor=SimpleNamespace(action_effect_encoder=linear))
    )
    initial, snapshot = trainer.encoder_metrics(model, None)
    assert initial["weight_nonzero_parameters"] == 0
    assert initial["update_l2_norm"] == 0.0
    with torch.no_grad():
        linear.weight[0, 0] = 3.0
        linear.weight[0, 1] = 4.0
    updated, _ = trainer.encoder_metrics(model, snapshot)
    assert updated["weight_nonzero_parameters"] == 2
    assert updated["weight_l2_norm"] == 5.0
    assert updated["update_l2_norm"] == 5.0


def test_optimizer_count_is_exact_and_includes_parent() -> None:
    assert trainer.optimizer_update_count_valid(
        0,
        v03.PARENT_OPTIMIZER_UPDATES,
    )
    assert trainer.optimizer_update_count_valid(
        v03.ROLLOUT_TRANSITIONS,
        v03.PARENT_OPTIMIZER_UPDATES + v03.PPO_EPOCHS,
    )
    assert trainer.optimizer_update_count_valid(
        v03.CHILD_ACTION_BUDGET,
        3_584,
    )
    assert not trainer.optimizer_update_count_valid(
        v03.CHILD_ACTION_BUDGET,
        3_583,
    )


def test_stage_a_controller_never_early_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = trainer.StageAController()
    state = lessons.CurriculumState()
    monkeypatch.setattr(
        lessons,
        "weak_prerequisites",
        lambda _evaluations: (),
    )
    decision = controller.update_recovery(
        state,
        {},
        allocation_valid=True,
    )
    assert decision == "normal_practice_continues"
    assert not state.mastered
    assert state.consecutive_passes == 0

    monkeypatch.setattr(
        lessons,
        "weak_prerequisites",
        lambda _evaluations: (lessons.LessonId.NAVIGATE,),
    )
    decision = controller.update_recovery(
        state,
        {},
        allocation_valid=True,
    )
    assert decision == "recovery_started"
    assert state.weak_prerequisites == (lessons.LessonId.NAVIGATE,)
    assert not state.mastered


def test_evaluation_always_installs_predict_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    cases = [{"case": 1}]

    def fake_evaluate(model, lesson, seeds, **kwargs):
        assert isinstance(model, ActionEffectPredictAdapter)
        assert model.mode is ActionEffectMode.ACTION_EFFECT
        assert lesson is lessons.LessonId.NAVIGATE
        assert tuple(seeds) == (1, 2)
        assert kwargs["penalty_enabled"] is False
        return sentinel, cases

    monkeypatch.setattr(
        trainer.frozen_u2s,
        "evaluate_lesson_with_visible_no_effect",
        fake_evaluate,
    )
    raw = SimpleNamespace(
        observation_space=gym.spaces.Dict(
            {
                IMAGE_KEY: gym.spaces.Box(
                    0,
                    255,
                    (3, 56, 56),
                    np.uint8,
                ),
                ACTION_EFFECT_KEY: gym.spaces.Box(
                    0.0,
                    1.0,
                    (ACTION_EFFECT_DIM,),
                    np.float32,
                ),
            }
        ),
        action_space=gym.spaces.Discrete(7),
    )
    result = trainer.evaluate_architecture_lesson(
        raw,
        mode=ActionEffectMode.ACTION_EFFECT,
        lesson=lessons.LessonId.NAVIGATE,
        seeds=(1, 2),
        seed_access=object(),
    )
    assert result == (sentinel, cases)


def test_normalized_first_rollout_excludes_arm_specific_fields() -> None:
    workers = [
        {
            "worker_index": index,
            "transitions": 512,
            "episodes_started": 2,
            "episode_starts_sha256": "a" * 64,
            "action_counts": {str(action): 0 for action in range(7)},
            "lesson_transition_counts": {lesson.value: 0 for lesson in lessons.LessonId},
            "trajectory_sha256": "b" * 64,
            "reward_evidence_sha256": "c" * 64,
            "arm_only": index,
        }
        for index in range(4)
    ]
    arm = {
        "arm": "sham",
        "workers": workers,
        "policy_output_sha256": "d" * 64,
        "post_rollout_rng_identity": {
            "phase": "post_rollout_pre_optimizer",
            "components": {},
            "aggregate_sha256": "e" * 64,
        },
        "episode_ledger": {"normalized_sha256": "f" * 64},
    }
    identity = trainer.normalized_first_rollout_identity(arm)
    assert "arm" not in identity
    assert "arm_only" not in identity["trajectory_identity"][0]
    assert len(identity["aggregate_sha256"]) == 64


def _write_terminal_arm(
    root: Path,
    *,
    arm: ActionEffectMode,
    source_commit: str,
) -> None:
    run = root / arm.value
    checkpoints = run / "checkpoints" / "rolling"
    checkpoints.mkdir(parents=True)
    cohort_id = "v0.3-action-effect-stage-a-r2-20260724"
    contract_sha256 = "a" * 64
    source = {"commit": source_commit, "dirty": False}
    qualification = {"report_sha256": "b" * 64}
    evaluations, cases = _valid_exam_case_evidence()
    records: list[dict] = []
    case_files: list[dict] = []
    for index in range(1, v03.EXAM_COUNT + 1):
        boundary = index * v03.EVALUATION_INTERVAL
        checkpoint = checkpoints / f"exam-{boundary:07d}.zip"
        checkpoint.write_bytes(f"exam {boundary}".encode())
        relative_checkpoint = str(checkpoint.relative_to(run))
        checkpoint_sidecar = {
            "schema_version": trainer.SCHEMA_VERSION,
            "protocol": trainer.PROTOCOL,
            "kind": "exam",
            "arm": arm.value,
            "cohort_id": cohort_id,
            "cohort_contract_sha256": contract_sha256,
            "resume_eligible": False,
            "resume_authorized": False,
            "promotable": False,
            "development_checkpoint_reuse_authorized": False,
            "checkpoint_sha256": file_sha256(checkpoint),
            "source": source,
            "qualification": qualification,
            "progress": {
                "child_trained_actions": boundary,
                "lifetime_trained_actions": (v03.PARENT_LIFETIME_ACTIONS + boundary),
                "optimizer_updates": (
                    v03.PARENT_OPTIMIZER_UPDATES
                    + boundary // v03.ROLLOUT_TRANSITIONS * v03.PPO_EPOCHS
                ),
            },
        }
        atomic_write_json(
            checkpoint.with_suffix(".json"),
            checkpoint_sidecar,
        )
        trainer._write_integrity(
            checkpoint,
            checkpoint.with_suffix(".json"),
        )
        diagnostics = trainer.frozen_u2s._case_diagnostics_from_records(cases)
        cases_path = checkpoint.with_suffix(".cases.json")
        case_document = {
            "schema_version": trainer.CASE_EVIDENCE_SCHEMA_VERSION,
            "protocol": trainer.PROTOCOL,
            "arm": arm.value,
            "checkpoint": relative_checkpoint,
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_sidecar_sha256": file_sha256(checkpoint.with_suffix(".json")),
            "child_trained_actions": boundary,
            "case_diagnostics": diagnostics,
            "records": cases,
        }
        atomic_write_json(cases_path, case_document)
        relative_cases = str(cases_path.relative_to(run))
        cases_sha256 = file_sha256(cases_path)
        case_files.append(
            {
                "path": relative_cases,
                "sha256": cases_sha256,
                "records": len(cases),
            }
        )
        records.append(
            {
                "child_trained_actions": boundary,
                "allocation_valid": True,
                "practice_profile": "normal",
                "checkpoint": relative_checkpoint,
                "checkpoint_sha256": file_sha256(checkpoint),
                "sidecar_sha256": file_sha256(checkpoint.with_suffix(".json")),
                "lessons": evaluations,
                "case_diagnostics": {
                    **diagnostics,
                    "path": relative_cases,
                    "sha256": cases_sha256,
                },
            }
        )

    terminal = run / "checkpoints" / "terminal.zip"
    terminal.write_bytes(b"test checkpoint")
    grade = v03.grade_terminal(arm, records)
    sidecar = {
        "schema_version": trainer.SCHEMA_VERSION,
        "protocol": trainer.PROTOCOL,
        "kind": "terminal",
        "arm": arm.value,
        "cohort_id": cohort_id,
        "cohort_contract_sha256": contract_sha256,
        "resume_eligible": False,
        "resume_authorized": False,
        "promotable": False,
        "development_checkpoint_reuse_authorized": False,
        "checkpoint_sha256": file_sha256(terminal),
        "progress": {
            "child_trained_actions": v03.CHILD_ACTION_BUDGET,
        },
    }
    atomic_write_json(terminal.with_suffix(".json"), sidecar)
    trainer._write_integrity(terminal, terminal.with_suffix(".json"))
    report = {
        "schema_version": trainer.REPORT_SCHEMA_VERSION,
        "protocol": trainer.PROTOCOL,
        "cohort_id": cohort_id,
        "cohort_contract_sha256": contract_sha256,
        "arm": arm.value,
        "development_only": True,
        "successor_checkpoint_authorized": False,
        "development_checkpoint_reuse_authorized": False,
        "resume_authorized": False,
        "promotable": False,
        "u3_authorized": False,
        "source": source,
        "qualification": qualification,
        "qualification_sha256": qualification["report_sha256"],
        "progress": {
            "child_trained_actions": v03.CHILD_ACTION_BUDGET,
            "lifetime_trained_actions": (v03.PARENT_LIFETIME_ACTIONS + v03.CHILD_ACTION_BUDGET),
            "optimizer_updates": 3_584,
            "exam_count": v03.EXAM_COUNT,
        },
        "terminal_checkpoint": {
            "path": str(terminal.relative_to(run)),
            "sha256": file_sha256(terminal),
            "sidecar_sha256": file_sha256(terminal.with_suffix(".json")),
            "integrity_sha256": file_sha256(trainer._integrity_path(terminal)),
            "promotable": False,
        },
        "controller": {
            "exam_records": records,
            "practice_decisions": [],
        },
        "exam_records": records,
        "case_count": (v03.EXAM_COUNT * len(lessons.LessonId) * v03.EVALUATION_SEED_COUNT),
        "case_evidence": {
            "files": case_files,
            "file_count": v03.EXAM_COUNT,
            "record_count": (v03.EXAM_COUNT * len(lessons.LessonId) * v03.EVALUATION_SEED_COUNT),
            "inventory_sha256": trainer._canonical_sha256(case_files),
        },
        "grade": grade.public_dict(),
    }
    report_path = run / "report.json"
    atomic_write_json(report_path, report)
    atomic_write_json(
        run / "status.json",
        {
            "protocol": trainer.PROTOCOL,
            "cohort_id": cohort_id,
            "cohort_contract_sha256": contract_sha256,
            "qualification_sha256": qualification["report_sha256"],
            "phase": "completed",
            "arm": arm.value,
            "child_trained_actions": v03.CHILD_ACTION_BUDGET,
            "remaining_action_budget": 0,
            "optimizer_updates": 3_584,
            "exam_count": v03.EXAM_COUNT,
            "report_sha256": file_sha256(report_path),
        },
    )
    atomic_write_json(
        run / "report.integrity.json",
        {
            "schema_version": trainer.REPORT_SCHEMA_VERSION,
            "protocol": trainer.PROTOCOL,
            "cohort_id": cohort_id,
            "cohort_contract_sha256": contract_sha256,
            "arm": arm.value,
            "report": report_path.name,
            "report_sha256": file_sha256(report_path),
            "terminal_checkpoint_sha256": file_sha256(terminal),
            "terminal_sidecar_sha256": file_sha256(terminal.with_suffix(".json")),
            "terminal_integrity_sha256": file_sha256(trainer._integrity_path(terminal)),
        },
    )


def test_terminal_arm_reports_are_authenticated_and_nonpromotable(
    tmp_path: Path,
) -> None:
    source_commit = "1" * 40
    for arm in v03.ARM_ORDER:
        _write_terminal_arm(
            tmp_path,
            arm=arm,
            source_commit=source_commit,
        )
        verified = trainer.verify_arm_terminal_report(
            tmp_path / arm.value,
            expected_arm=arm,
            expected_source_commit=source_commit,
        )
        assert verified["eligible"]
        assert verified["grade"]["eligible"]
        assert verified["case_count"] == 10_240
        assert len(verified["case_evidence_files"]) == v03.EXAM_COUNT
        assert len(verified["exam_bundles"]) == v03.EXAM_COUNT


def test_terminal_arm_rejects_missing_exam_archive(
    tmp_path: Path,
) -> None:
    source_commit = "1" * 40
    arm = ActionEffectMode.SHAM
    _write_terminal_arm(
        tmp_path,
        arm=arm,
        source_commit=source_commit,
    )
    run = tmp_path / arm.value
    first_exam = next((run / "checkpoints" / "rolling").glob("exam-*.zip"))
    first_exam.unlink()

    with pytest.raises(
        trainer.ActionEffectTrainingError,
        match="checkpoint bundle",
    ):
        trainer.verify_arm_terminal_report(
            run,
            expected_arm=arm,
            expected_source_commit=source_commit,
        )


def test_terminal_arm_rejects_tampered_case_file(
    tmp_path: Path,
) -> None:
    source_commit = "1" * 40
    arm = ActionEffectMode.SHAM
    _write_terminal_arm(
        tmp_path,
        arm=arm,
        source_commit=source_commit,
    )
    run = tmp_path / arm.value
    first_cases = next((run / "checkpoints" / "rolling").glob("exam-*.cases.json"))
    first_cases.write_bytes(first_cases.read_bytes() + b"\n")

    with pytest.raises(
        trainer.ActionEffectTrainingError,
        match="case evidence changed",
    ):
        trainer.verify_arm_terminal_report(
            run,
            expected_arm=arm,
            expected_source_commit=source_commit,
        )


def test_terminal_arm_recomputes_exam_aggregate_from_cases(
    tmp_path: Path,
) -> None:
    source_commit = "1" * 40
    arm = ActionEffectMode.SHAM
    _write_terminal_arm(
        tmp_path,
        arm=arm,
        source_commit=source_commit,
    )
    run = tmp_path / arm.value
    report_path = run / "report.json"
    status_path = run / "status.json"
    integrity_path = run / "report.integrity.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    first_exam = report["controller"]["exam_records"][0]
    cases_path = run / first_exam["case_diagnostics"]["path"]
    document = json.loads(cases_path.read_text(encoding="utf-8"))
    record = document["records"][0]
    record["success"] = not record["success"]
    record["terminal_reason"] = "success" if record["success"] else "time_limit"
    record["milestones"]["success"] = record["success"]
    atomic_write_json(cases_path, document)
    changed_sha256 = file_sha256(cases_path)

    for collection in (
        report["controller"]["exam_records"],
        report["exam_records"],
    ):
        collection[0]["case_diagnostics"]["sha256"] = changed_sha256
    case_entry = next(
        item
        for item in report["case_evidence"]["files"]
        if item["path"] == first_exam["case_diagnostics"]["path"]
    )
    case_entry["sha256"] = changed_sha256
    report["case_evidence"]["inventory_sha256"] = trainer._canonical_sha256(
        report["case_evidence"]["files"]
    )
    atomic_write_json(report_path, report)
    report_sha256 = file_sha256(report_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["report_sha256"] = report_sha256
    atomic_write_json(status_path, status)
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity["report_sha256"] = report_sha256
    atomic_write_json(integrity_path, integrity)

    with pytest.raises(
        trainer.ActionEffectTrainingError,
        match="aggregate disagrees",
    ):
        trainer.verify_arm_terminal_report(
            run,
            expected_arm=arm,
            expected_source_commit=source_commit,
        )


def test_public_cli_has_no_seed_budget_or_resume_override() -> None:
    parser = trainer.build_parser()
    option_strings = {
        option
        for action in parser._subparsers._group_actions[0].choices["train-arm"]._actions
        for option in action.option_strings
    }
    assert "--seed" not in option_strings
    assert "--child-budget" not in option_strings
    assert "--resume" not in option_strings


def test_trainer_consumes_launcher_owned_paths_without_mutating_them(
    tmp_path: Path,
) -> None:
    cohort = tmp_path / "cohort"
    media_root = tmp_path / "media"
    run = cohort / "sham"
    arm_media = media_root / "sham"
    run.mkdir(parents=True)
    arm_media.mkdir(parents=True)
    source = {
        "commit": "1" * 40,
        "branch": "agent/test",
        "dirty": False,
    }
    qualification = SimpleNamespace(
        public_dict=lambda: {
            "report_sha256": "2" * 64,
            "tag_object": "3" * 40,
        }
    )
    contract = {
        "schema_version": 1,
        "protocol": trainer.PROTOCOL,
        "cohort_id": "v0.3-action-effect-stage-a-r2-20260724",
        "source": {
            "commit": source["commit"],
            "dirty": False,
        },
        "qualification": {
            "report_sha256": "2" * 64,
            "tag_object": "3" * 40,
        },
        "parent": {
            "checkpoint_sha256": v03.PARENT_CHECKPOINT_SHA256,
            "policy_tensor_sha256": v03.PARENT_POLICY_TENSOR_SHA256,
            "optimizer_state_sha256": (v03.PARENT_OPTIMIZER_STATE_SHA256),
        },
        "roots": {
            "cohort": str(cohort),
            "media": str(media_root),
        },
        "matched_design": {
            "arm_order": [
                ActionEffectMode.SHAM.value,
                ActionEffectMode.ACTION_EFFECT.value,
            ],
            "action_cap_per_arm": v03.CHILD_ACTION_BUDGET,
            "evaluation_every": v03.EVALUATION_INTERVAL,
            "fresh_only": True,
            "resumable": False,
            "checkpoint_promotable": False,
        },
        "arms": [
            {
                "id": ActionEffectMode.SHAM.value,
                "directory": "sham",
                "media_directory": "sham",
            },
            {
                "id": ActionEffectMode.ACTION_EFFECT.value,
                "directory": "action-effect",
                "media_directory": "action-effect",
            },
        ],
    }
    contract_path = cohort / "cohort-contract.json"
    atomic_write_json(contract_path, contract)
    before = sorted(path.name for path in cohort.iterdir())
    verified, digest, verified_root, verified_media = trainer.verify_cohort_contract(
        path=contract_path,
        arm=ActionEffectMode.SHAM,
        run_directory=run,
        media_directory=arm_media,
        expected_cohort_root=cohort,
        expected_media_root=media_root,
        source=source,
        qualification=qualification,
    )
    assert verified == contract
    assert digest == file_sha256(contract_path)
    assert verified_root == cohort
    assert verified_media == media_root
    assert sorted(path.name for path in cohort.iterdir()) == before


def test_trainer_rejects_noncanonical_roots_and_arm_directories(
    tmp_path: Path,
) -> None:
    cohort = tmp_path / "cohort"
    media_root = tmp_path / "media"
    run = cohort / "sham"
    arm_media = media_root / "sham"
    run.mkdir(parents=True)
    arm_media.mkdir(parents=True)
    source = {"commit": "1" * 40, "dirty": False}
    qualification = SimpleNamespace(
        public_dict=lambda: {
            "report_sha256": "2" * 64,
            "tag_object": "3" * 40,
        }
    )
    contract = {
        "schema_version": 1,
        "protocol": trainer.PROTOCOL,
        "cohort_id": "v0.3-action-effect-stage-a-r2-20260724",
        "source": source,
        "qualification": {
            "report_sha256": "2" * 64,
            "tag_object": "3" * 40,
        },
        "parent": {
            "checkpoint_sha256": v03.PARENT_CHECKPOINT_SHA256,
            "policy_tensor_sha256": v03.PARENT_POLICY_TENSOR_SHA256,
            "optimizer_state_sha256": v03.PARENT_OPTIMIZER_STATE_SHA256,
        },
        "roots": {"cohort": str(cohort), "media": str(media_root)},
        "matched_design": {
            "arm_order": [arm.value for arm in v03.ARM_ORDER],
            "action_cap_per_arm": v03.CHILD_ACTION_BUDGET,
            "evaluation_every": v03.EVALUATION_INTERVAL,
            "fresh_only": True,
            "resumable": False,
            "checkpoint_promotable": False,
        },
        "arms": [
            {"id": "sham", "directory": "sham", "media_directory": "sham"},
            {
                "id": "action-effect",
                "directory": "action-effect",
                "media_directory": "action-effect",
            },
        ],
    }
    contract_path = cohort / "cohort-contract.json"
    atomic_write_json(contract_path, contract)

    with pytest.raises(
        trainer.ActionEffectTrainingError,
        match="cohort contract is missing or unsafe",
    ):
        trainer.verify_cohort_contract(
            path=contract_path,
            arm=ActionEffectMode.SHAM,
            run_directory=run,
            media_directory=arm_media,
            expected_cohort_root=tmp_path / "immutable-r1",
            expected_media_root=tmp_path / "immutable-r1-media",
            source=source,
            qualification=qualification,
        )

    contract["arms"][0]["directory"] = ".v03-staging"
    atomic_write_json(contract_path, contract)
    with pytest.raises(
        trainer.ActionEffectTrainingError,
        match="cohort contract changed",
    ):
        trainer.verify_cohort_contract(
            path=contract_path,
            arm=ActionEffectMode.SHAM,
            run_directory=run,
            media_directory=arm_media,
            expected_cohort_root=cohort,
            expected_media_root=media_root,
            source=source,
            qualification=qualification,
        )
