from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest
from minigrid.core.actions import Actions

from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v02_u2s as u2s
from dungeon_apprentice.artifacts import atomic_write_json, file_sha256


class _ObservationSequenceEnv(gym.Env):
    metadata: ClassVar[dict] = {}

    def __init__(
        self,
        observations: list[np.ndarray],
        *,
        terminate_at: int | None = None,
    ) -> None:
        super().__init__()
        self.observations = [np.array(value, copy=True) for value in observations]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.observations[0].shape,
            dtype=self.observations[0].dtype,
        )
        self.action_space = gym.spaces.Discrete(7)
        self.index = 0
        self.terminate_at = terminate_at

    def reset(self, **_kwargs: object) -> tuple[np.ndarray, dict]:
        self.index = 0
        return np.array(self.observations[0], copy=True), {}

    def step(self, _action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.index += 1
        observation = self.observations[min(self.index, len(self.observations) - 1)]
        return (
            np.array(observation, copy=True),
            0.5,
            self.terminate_at is not None and self.index >= self.terminate_at,
            False,
            {"ineffective_interactions": -999},
        )


def _stable_exam(index: int) -> dict:
    return {
        "child_trained_actions": index * u2s.EVALUATION_INTERVAL,
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
            "max_identical_visible_no_effect_streak": 0,
            "max_repeated_identical_interaction_run": 0,
        },
    }


def _stable_arm() -> list[dict]:
    return [_stable_exam(index) for index in range(1, u2s.EXAM_COUNT + 1)]


def _valid_initial_rng_identity() -> dict:
    components = {
        field: f"{index + 1:064x}"
        for index, field in enumerate(u2s.INITIAL_RNG_COMPONENT_DIGESTS)
    }
    components["torch_mps_sha256"] = None
    components["torch_cuda_sha256"] = None
    components["workers"] = [
        {
            "worker_index": index,
            "worker_stream": stream,
            **{
                field: f"{20 + index * 3 + offset:064x}"
                for offset, field in enumerate(
                    u2s.INITIAL_RNG_WORKER_COMPONENT_DIGESTS
                )
            },
        }
        for index, stream in enumerate(u2s.WORKER_STREAMS)
    ]
    return {
        "captured_before_action_one": True,
        "algorithm_seed": u2s.ALGORITHM_SEED,
        "components": components,
        "aggregate_sha256": u2s._canonical_sha256(components),
    }


def _u2s_exam_case_evidence(
    *,
    arm: u2s.ArmName = u2s.ArmName.CONTROL,
) -> tuple[dict, list[dict]]:
    records: list[dict] = []
    evaluations: dict[str, dict] = {}
    penalty_enabled = u2s.ARM_SPECS[arm].no_effect_penalty
    for lesson in lessons.LessonId:
        lesson_records: list[dict] = []
        for case_index in range(u2s.EVALUATION_SEED_COUNT):
            panel = 0 if case_index < 40 else 1
            panel_case_index = case_index % 40
            success = panel_case_index < 36
            oracle_range = lessons.LESSON_SPECS[lesson].oracle_action_range
            oracle_actions = oracle_range[0] if oracle_range is not None else 4
            steps = min(oracle_actions + 5, lessons.LESSON_SPECS[lesson].max_steps)
            visibility = (
                "key_hidden_door_hidden"
                if lesson is lessons.LessonId.SEPARATED_UNLOCK
                else None
            )
            visible_effects = u2s.visible_interaction_effect_evidence(
                [],
                [],
                [],
                penalty_enabled=penalty_enabled,
            )
            lesson_records.append(
                {
                    "lesson_id": lesson.value,
                    "lesson_label": lessons.LESSON_SPECS[lesson].label,
                    "case_index": case_index,
                    "panel": panel,
                    "panel_case_index": panel_case_index,
                    "seed": (
                        lessons.LESSON_SPECS[lesson].validation_seed_base
                        + case_index
                    ),
                    "layout_sha256": u2s._canonical_sha256(
                        [lesson.value, case_index, "layout"]
                    ),
                    "geometry_sha256": u2s._canonical_sha256(
                        [lesson.value, case_index, "geometry"]
                    ),
                    "success": success,
                    "terminal_reason": "success" if success else "time_limit",
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
                    "path_actions_per_oracle_action": steps / oracle_actions,
                    "action_histogram": {
                        str(action): steps if action == int(Actions.left) else 0
                        for action in range(7)
                    },
                    "longest_repeated_action_run": steps,
                    "longest_identical_visible_no_effect_streak": 0,
                    "longest_repeated_identical_interaction_run": 0,
                    "visible_no_effect_transition_count": 0,
                    "visible_interaction_effects": visible_effects,
                    "action_trace_sha256": u2s._canonical_sha256(
                        [lesson.value, case_index, "actions"]
                    ),
                    "visible_no_effect_streak_sha256": u2s._canonical_sha256([]),
                    "interaction_run_sha256": u2s._canonical_sha256([]),
                    "extrinsic_return": 1.0 if success else 0.0,
                    "curiosity_return": 0.0,
                    "initial_key_visible": (
                        False
                        if lesson is lessons.LessonId.SEPARATED_UNLOCK
                        else None
                    ),
                    "initial_door_visible": (
                        False
                        if lesson is lessons.LessonId.SEPARATED_UNLOCK
                        else None
                    ),
                    "visibility_stratum": visibility,
                }
            )
        result = lessons.aggregate_u2_case_evidence(
            lesson,
            lesson_records,
            timestamp="2026-07-23T00:00:00+00:00",
        )
        evaluations[lesson.value] = u2s._augmented_lesson_evidence(
            result,
            lesson_records,
            penalty_enabled=penalty_enabled,
        )
        records.extend(lesson_records)
    return {"lessons": evaluations}, records


def test_arm_matrix_changes_only_prospective_dimensions() -> None:
    control = u2s.ARM_SPECS[u2s.ArmName.CONTROL]
    conservative = u2s.ARM_SPECS[u2s.ArmName.CONSERVATIVE]
    penalty = u2s.ARM_SPECS[u2s.ArmName.NO_EFFECT]
    combined = u2s.ARM_SPECS[u2s.ArmName.COMBINED]

    assert not control.conservative_ppo and not control.no_effect_penalty
    assert conservative.conservative_ppo and not conservative.no_effect_penalty
    assert not penalty.conservative_ppo and penalty.no_effect_penalty
    assert combined.conservative_ppo and combined.no_effect_penalty
    assert conservative.n_epochs == combined.n_epochs == 2
    assert control.n_epochs == penalty.n_epochs == 4
    assert conservative.clip_range == combined.clip_range == 0.10
    assert conservative.target_kl == combined.target_kl == 0.015


def test_conservative_schedule_is_exactly_child_action_based() -> None:
    schedule = u2s.ChildActionLinearSchedule()
    raw_start = schedule.raw_start_progress_remaining

    assert schedule(raw_start) == pytest.approx(2.5e-4)
    assert schedule(raw_start / 2) == pytest.approx(1.375e-4)
    assert schedule(0.0) == pytest.approx(2.5e-5)
    assert schedule.for_child_actions(0) == pytest.approx(2.5e-4)
    assert schedule.for_child_actions(u2s.CHILD_ACTION_BUDGET // 2) == pytest.approx(1.375e-4)
    assert schedule.for_child_actions(u2s.CHILD_ACTION_BUDGET) == pytest.approx(2.5e-5)
    # SB3's inherited-lifetime start is below one; values above it must not
    # accidentally increase the prospective starting rate.
    assert schedule(1.0) == pytest.approx(2.5e-4)


def test_optimizer_update_bounds_allow_target_kl_early_stop_only_where_declared() -> None:
    actions = 3 * u2s.ROLLOUT_TRANSITIONS
    assert u2s.optimizer_update_bounds("control", actions) == (12, 12)
    assert u2s.optimizer_update_bounds("no-effect", actions) == (12, 12)
    assert u2s.optimizer_update_bounds("conservative", actions) == (3, 6)
    assert u2s.optimizer_update_bounds("combined", actions, parent_updates=10) == (
        13,
        16,
    )
    assert u2s.optimizer_update_count_valid("conservative", actions, 3)
    assert u2s.optimizer_update_count_valid("conservative", actions, 6)
    assert not u2s.optimizer_update_count_valid("conservative", actions, 2)
    assert not u2s.optimizer_update_count_valid("control", actions, 11)
    with pytest.raises(u2s.U2SProtocolError):
        u2s.optimizer_update_bounds("control", actions + 1)


def test_penalty_requires_two_consecutive_identical_visible_no_effects() -> None:
    zeros = np.zeros((4, 4, 3), dtype=np.uint8)
    ones = np.ones((4, 4, 3), dtype=np.uint8)
    wrapped = u2s.NoEffectInteractionPenalty(
        _ObservationSequenceEnv([zeros, ones, ones, ones, ones])
    )
    wrapped.reset()

    # First toggle changes pixels: it cannot seed an eligible streak.
    _, reward, *_rest, info = wrapped.step(int(Actions.toggle))
    assert reward == pytest.approx(0.5)
    assert info["u2s_identical_interaction_run"] == 0

    # First no-effect toggle begins the streak but is not penalized.
    _, reward, *_rest, info = wrapped.step(int(Actions.toggle))
    assert reward == pytest.approx(0.5)
    assert info["u2s_identical_interaction_run"] == 1

    # The second consecutive identical no-effect transition is penalized.
    _, reward, *_rest, info = wrapped.step(int(Actions.toggle))
    assert reward == pytest.approx(0.49)
    assert info["u2s_no_effect_penalty"] == pytest.approx(-0.01)
    assert info["u2s_identical_interaction_run"] == 2


def test_penalty_resets_on_different_action_and_ignores_privileged_info() -> None:
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    wrapped = u2s.NoEffectInteractionPenalty(_ObservationSequenceEnv([pixels] * 6))
    wrapped.reset()
    wrapped.step(int(Actions.pickup))
    _, reward, *_rest, info = wrapped.step(int(Actions.toggle))
    assert reward == pytest.approx(0.5)
    assert info["u2s_identical_interaction_run"] == 1
    _, reward, *_rest, info = wrapped.step(int(Actions.toggle))
    assert reward == pytest.approx(0.49)
    # The wrapped env deliberately reports nonsense privileged diagnostics;
    # the pixels-only wrapper neither reads nor trusts them.
    assert info["ineffective_interactions"] == -999


def test_penalty_is_bounded_below_success_signal_and_resets_each_episode() -> None:
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    wrapped = u2s.NoEffectInteractionPenalty(_ObservationSequenceEnv([pixels] * 20))
    wrapped.reset()
    rewards = [wrapped.step(int(Actions.toggle))[1] for _ in range(15)]
    assert sum(0.5 - reward for reward in rewards) == pytest.approx(u2s.NO_EFFECT_EPISODE_CAP)
    wrapped.reset()
    assert wrapped.step(int(Actions.toggle))[1] == pytest.approx(0.5)


def test_penalty_cap_is_exactly_ten_integer_counted_events() -> None:
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    wrapped = u2s.NoEffectInteractionPenalty(_ObservationSequenceEnv([pixels] * 25))
    wrapped.reset()
    transitions = [wrapped.step(int(Actions.toggle)) for _ in range(20)]
    applied = [transition[4]["u2s_no_effect_penalty"] for transition in transitions]
    assert applied == [0.0, *([-0.01] * 10), *([0.0] * 9)]
    final_info = transitions[-1][4]
    assert final_info["u2s_no_effect_applied_penalty_count"] == 10
    assert final_info["u2s_no_effect_episode_total"] == -0.1


def test_penalty_terminal_info_survives_internal_episode_reset() -> None:
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    wrapped = u2s.NoEffectInteractionPenalty(
        _ObservationSequenceEnv(
            [pixels] * 5,
            terminate_at=3,
        )
    )
    wrapped.reset()
    wrapped.step(int(Actions.toggle))
    wrapped.step(int(Actions.toggle))
    _observation, _reward, terminated, _truncated, info = wrapped.step(int(Actions.toggle))
    assert terminated
    assert info["u2s_no_effect_applied_penalty_count"] == 2
    assert info["u2s_no_effect_episode_total"] == -0.02
    assert wrapped._applied_penalty_count == 0
    assert wrapped._episode_penalty == 0.0
    assert wrapped._previous_action is None


def test_episode_evidence_wrapper_records_every_reset_identity(
    tmp_path: Path,
) -> None:
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)

    class EvidenceEnv(_ObservationSequenceEnv):
        def reset(self, **kwargs: object) -> tuple[np.ndarray, dict]:
            observation, _info = super().reset(**kwargs)
            return observation, {
                "seed": 123,
                "lesson_id": lessons.LessonId.SEPARATED_UNLOCK.value,
                "layout_sha256": "a" * 64,
                "geometry_sha256": "b" * 64,
            }

    ledger = tmp_path / "episode-starts.jsonl"
    wrapper = u2s.EpisodeEvidenceWrapper(
        EvidenceEnv([pixels] * 3),
        worker_index=2,
        worker_stream=20260755,
        ledger=ledger,
    )
    wrapper.reset()
    wrapper.step(int(Actions.forward))
    wrapper.reset()
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert [record["episode_ordinal"] for record in records] == [1, 2]
    assert all(record["type"] == "episode_start" for record in records)
    assert wrapper.evidence_state() == {
        "worker_index": 2,
        "worker_stream": 20260755,
        "episode_ordinal": 2,
        "episode_seed": 123,
        "lesson_id": lessons.LessonId.SEPARATED_UNLOCK.value,
        "layout_sha256": "a" * 64,
        "geometry_sha256": "b" * 64,
        "worker_transition_at_start": 1,
        "elapsed_steps": 0,
        "active": True,
    }


def test_visible_no_effect_diagnostic_resets_on_any_pixel_change() -> None:
    zeros = np.zeros((2, 2, 3), dtype=np.uint8)
    ones = np.ones((2, 2, 3), dtype=np.uint8)
    actions = [int(Actions.toggle)] * 4
    streaks = u2s.visible_no_effect_streaks(
        actions,
        [zeros, ones, ones, zeros],
        [ones, ones, zeros, zeros],
    )
    assert streaks == (0, 1, 0, 1)


def test_generic_interaction_run_is_independent_of_visible_effect() -> None:
    actions = [
        int(Actions.toggle),
        int(Actions.toggle),
        int(Actions.toggle),
        int(Actions.forward),
        int(Actions.pickup),
        int(Actions.pickup),
    ]
    assert u2s.repeated_identical_interaction_runs(actions) == (
        1,
        2,
        3,
        0,
        1,
        2,
    )


def test_ineffective_tail_summary_preserves_concentration() -> None:
    summary = u2s.ineffective_tail_summary([0, 1, 3, 10, 32])
    assert summary["ineffective_threshold_counts"] == {
        "at_least_1": 4,
        "at_least_3": 3,
        "at_least_10": 2,
        "at_least_32": 1,
    }
    assert summary["worst_case_shares"]["worst_1"] == pytest.approx(32 / 46)
    assert summary["worst_case_shares"]["worst_2"] == pytest.approx(42 / 46)
    assert summary["ineffective_quantiles"]["p50"] == pytest.approx(3)


def test_terminal_grade_requires_all_three_final_exams() -> None:
    exams = _stable_arm()
    grade = u2s.grade_arm_terminal("control", exams)
    assert grade.eligible
    assert grade.final_exam_actions == (
        983_040,
        1_015_808,
        1_048_576,
    )

    exams[-2]["case_diagnostics"]["max_case_ineffective_interactions"] = 10
    exams[-2]["case_diagnostics"]["cases_with_ineffective_at_least_10"] = 1
    failed = u2s.grade_arm_terminal("control", exams)
    assert not failed.eligible
    assert any("terminal_2:cases" in reason for reason in failed.reasons)


def test_terminal_grade_rejects_visible_loop_tail_and_prerequisite_regression() -> None:
    exams = _stable_arm()
    exams[-1]["case_diagnostics"]["max_repeated_identical_interaction_run"] = 10
    exams[-3]["lessons"][lessons.LessonId.LOCAL_UNLOCK.value]["successes"] = 67
    grade = u2s.grade_arm_terminal("combined", exams)
    assert not grade.eligible
    assert not grade.checks["terminal_1:unlock/u1-local:existing_gate"]
    assert not grade.checks["terminal_3:cases:repeated_identical_interaction_run_below_10"]


def test_mechanism_selection_is_fixed_priority_and_never_returns_checkpoint() -> None:
    records = {arm: _stable_arm() for arm in u2s.ArmName}
    selected = u2s.select_mechanism(records)
    assert selected["selected_arm"] == "control"
    assert selected["successor_cohort_authorized"]
    assert not selected["ablation_checkpoint_reuse_authorized"]
    assert "selected_checkpoint" not in selected
    assert "checkpoint" not in selected["selected_mechanism"]

    records[u2s.ArmName.CONTROL][-1]["case_diagnostics"][
        "max_repeated_identical_interaction_run"
    ] = 10
    selected = u2s.select_mechanism(records)
    assert selected["selected_arm"] == "conservative"


def test_no_qualifying_arm_fails_without_authorizing_successor() -> None:
    records = {arm: _stable_arm() for arm in u2s.ArmName}
    for exams in records.values():
        exams[-1]["lessons"][lessons.LessonId.SEPARATED_UNLOCK.value]["successes"] = 71
    selected = u2s.select_mechanism(records)
    assert selected["verdict"] == "ablation_failed"
    assert selected["selected_mechanism"] is None
    assert not selected["successor_cohort_authorized"]


def test_cohort_contract_binds_complete_matched_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit = "a" * 40
    cohort = tmp_path / "cohort"
    media = tmp_path / "media"
    cohort.mkdir()
    media.mkdir()
    monkeypatch.setattr(u2s, "CANONICAL_COHORT_ROOT", cohort)
    monkeypatch.setattr(u2s, "CANONICAL_MEDIA_ROOT", media)
    qualification = {
        "report": str(u2s.DEFAULT_QUALIFICATION_REPORT),
        "report_sha256": "1" * 64,
        "checksum": f"{u2s.DEFAULT_QUALIFICATION_REPORT}.sha256",
        "source_commit": source_commit,
        "tag": u2s.TRAINING_TAG,
        "tag_object": "b" * 40,
        "tag_payload_sha256": "2" * 64,
        "protocol_document_sha256": "3" * 64,
        "verdict": "qualified",
        "sampler_preflight_sha256": "4" * 64,
        "arm_contract_sha256": "5" * 64,
        "protected_partitions_sha256": "6" * 64,
        "resume_contract_sha256": "7" * 64,
        "storage_preflight_sha256": "8" * 64,
        "initial_rng_identity": _valid_initial_rng_identity(),
        "guard_sets": {
            lesson.value: {
                "exact_layouts": index + 1,
                "exact_layout_set_sha256": f"{index + 4:064x}",
                "rule": f"guard-{index}",
            }
            for index, lesson in enumerate(lessons.LessonId)
        },
        "storage_caps": {
            "per_arm_bytes": u2s.LINEAGE_CAP_BYTES,
            "scientific_cohort_bytes": u2s.COHORT_SCIENTIFIC_CAP_BYTES,
            "media_bytes": u2s.MEDIA_CAP_BYTES,
            "combined_bytes": u2s.COMBINED_PLANNED_CAP_BYTES,
        },
    }
    path = cohort / "cohort-contract.json"
    expected = u2s._expected_cohort_contract(
        source_commit=source_commit,
        qualification=qualification,
    )
    atomic_write_json(path, expected)
    verified = u2s.verify_cohort_contract(
        path,
        source_commit=source_commit,
        qualification=qualification,
    )
    assert verified is not None
    assert verified["path"] == str(path.resolve())
    assert len(verified["sha256"]) == 64
    assert expected["matched_design"]["initial_rng_identity"] == (
        u2s._initial_rng_identity_contract()
    )
    assert expected["qualification"]["initial_rng_identity"] == (
        qualification["initial_rng_identity"]
    )
    assert {
        key: expected["qualification"][key]
        for key in (
            "sampler_preflight_sha256",
            "arm_contract_sha256",
            "protected_partitions_sha256",
            "resume_contract_sha256",
            "storage_preflight_sha256",
        )
    } == {
        key: qualification[key]
        for key in (
            "sampler_preflight_sha256",
            "arm_contract_sha256",
            "protected_partitions_sha256",
            "resume_contract_sha256",
            "storage_preflight_sha256",
        )
    }

    mutations = (
        lambda value: value["parent"].__setitem__("optimizer_state_sha256", "f" * 64),
        lambda value: value["qualification"]["guard_sets"][
            lessons.LessonId.SEPARATED_UNLOCK.value
        ].__setitem__("count", 0),
        lambda value: value["matched_design"]["interruption"].__setitem__("resumable", True),
        lambda value: value["matched_design"]["initial_rng_identity"].__setitem__(
            "captured_before_action_one", False
        ),
        lambda value: value["arms"][3]["intervention"].__setitem__("clip_range", 0.2),
    )
    for mutate in mutations:
        value = json.loads(json.dumps(expected))
        mutate(value)
        atomic_write_json(path, value)
        with pytest.raises(u2s.U2SProtocolError):
            u2s.verify_cohort_contract(
                path,
                source_commit=source_commit,
                qualification=qualification,
            )


@pytest.mark.parametrize(
    "component",
    (
        *u2s.INITIAL_RNG_COMPONENT_DIGESTS,
        *u2s.INITIAL_RNG_WORKER_COMPONENT_DIGESTS,
    ),
)
def test_initial_rng_identity_fails_closed_for_component_perturbation(
    component: str,
) -> None:
    expected = _valid_initial_rng_identity()
    changed = json.loads(json.dumps(expected))
    if component in u2s.INITIAL_RNG_COMPONENT_DIGESTS:
        changed["components"][component] = "f" * 64
    else:
        changed["components"]["workers"][2][component] = "f" * 64
    changed["aggregate_sha256"] = u2s._canonical_sha256(
        changed["components"]
    )

    assert u2s.verify_initial_rng_identity(expected) == expected
    with pytest.raises(
        u2s.U2SProtocolError,
        match="frozen expectation",
    ):
        u2s.verify_initial_rng_identity(changed, expected=expected)


def test_initial_rng_identity_rejects_an_unrecomputed_aggregate() -> None:
    identity = _valid_initial_rng_identity()
    identity["components"]["python_random_sha256"] = "f" * 64

    with pytest.raises(u2s.U2SProtocolError, match="aggregate"):
        u2s.verify_initial_rng_identity(identity)


def test_diagnostic_rng_guard_restores_the_actual_pre_action_state(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    random.seed(u2s.ALGORITHM_SEED)
    np.random.seed(u2s.ALGORITHM_SEED)
    torch.manual_seed(u2s.ALGORITHM_SEED)
    state = lessons.CurriculumState()
    scheduler = lessons.TransitionDeficitScheduler(
        state,
        seed=u2s.ALGORITHM_SEED + 90_000,
    )
    workers = []
    for index, stream in enumerate(u2s.WORKER_STREAMS):
        worker = u2s.EpisodeEvidenceWrapper(
            u2s._make_training_environment(
                scheduler=scheduler,
                seed=stream,
                seed_access=None,
                forbidden_layout_hashes={},
                penalize_no_effect=False,
            ),
            worker_index=index,
            worker_stream=stream,
            ledger=tmp_path / "episode-starts.jsonl",
        )
        workers.append(worker)
    model = SimpleNamespace(action_space=workers[0].action_space)
    u2s.seed_initial_rng_spaces(workers)
    for worker, stream in zip(workers, u2s.WORKER_STREAMS, strict=True):
        worker.reset(seed=stream)
    before = u2s.capture_initial_rng_identity(
        model=model,
        scheduler=scheduler,
        active_workers=workers,
    )
    snapshot = u2s._snapshot_training_rng_state(
        model=model,
        scheduler=scheduler,
        active_workers=workers,
    )

    random.random()
    np.random.random()
    torch.rand(2)
    scheduler._rng.random()
    for worker in workers:
        curriculum = u2s._curriculum_environment(worker)
        curriculum._rng.random()
        curriculum.unwrapped.np_random.random()
        worker.action_space.sample()
        worker.observation_space.sample()

    u2s._restore_training_rng_state(
        snapshot,
        model=model,
        scheduler=scheduler,
        active_workers=workers,
    )
    after = u2s.capture_initial_rng_identity(
        model=model,
        scheduler=scheduler,
        active_workers=workers,
    )
    assert after == before


@pytest.mark.parametrize(
    "tamper",
    (
        "summary",
        "panel",
        "seed",
        "missing",
        "unexpected_lesson",
        "duplicate",
        "schema",
    ),
)
def test_exam_case_verifier_rejects_fabricated_or_malformed_evidence(
    tamper: str,
) -> None:
    exam, records = _u2s_exam_case_evidence()
    if tamper == "summary":
        exam["lessons"][lessons.LessonId.SEPARATED_UNLOCK.value][
            "successes"
        ] += 1
    elif tamper == "panel":
        records[0]["panel"] = 1
    elif tamper == "seed":
        records[1]["seed"] = records[0]["seed"]
    elif tamper == "missing":
        records.pop()
    elif tamper == "unexpected_lesson":
        records[0]["lesson_id"] = "unlock/forged"
    elif tamper == "duplicate":
        records[1] = json.loads(json.dumps(records[0]))
    else:
        records[0]["forged"] = True

    with pytest.raises(u2s.U2SProtocolError):
        u2s._verify_exam_case_records(exam, records, arm=u2s.ArmName.CONTROL)


def test_exam_case_verifier_accepts_navigate_without_declared_oracle_range() -> None:
    assert lessons.LESSON_SPECS[lessons.LessonId.NAVIGATE].oracle_action_range is None
    exam, records = _u2s_exam_case_evidence()

    u2s._verify_exam_case_records(exam, records, arm=u2s.ArmName.CONTROL)


def test_deep_case_audit_recomputes_summary_and_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exam, records = _u2s_exam_case_evidence()
    exam.update(
        {
            "child_trained_actions": u2s.EVALUATION_INTERVAL,
            "checkpoint": "checkpoints/exam.zip",
            "checkpoint_sha256": "a" * 64,
            "sidecar_sha256": "b" * 64,
        }
    )
    diagnostics = u2s._case_diagnostics_from_records(records)
    case_path = tmp_path / "checkpoints" / "exam.cases.json"
    case_path.parent.mkdir()
    atomic_write_json(
        case_path,
        {
            "schema_version": u2s.CASE_EVIDENCE_SCHEMA_VERSION,
            "protocol": u2s.PROTOCOL,
            "arm": u2s.ArmName.CONTROL.value,
            "checkpoint": exam["checkpoint"],
            "checkpoint_sha256": exam["checkpoint_sha256"],
            "checkpoint_sidecar_sha256": exam["sidecar_sha256"],
            "child_trained_actions": exam["child_trained_actions"],
            "case_diagnostics": diagnostics,
            "records": records,
        },
    )
    exam["case_diagnostics"] = {
        **diagnostics,
        "path": str(case_path.relative_to(tmp_path)),
        "sha256": file_sha256(case_path),
    }
    monkeypatch.setattr(u2s, "EXAM_COUNT", 1)

    evidence = u2s._audit_case_files(
        tmp_path,
        u2s.ArmName.CONTROL,
        [exam],
    )
    assert evidence["record_count"] == 320

    exam["lessons"][lessons.LessonId.SEPARATED_UNLOCK.value][
        "successes"
    ] += 1
    with pytest.raises(u2s.U2SProtocolError, match="immutable cases"):
        u2s._audit_case_files(tmp_path, u2s.ArmName.CONTROL, [exam])


def test_u2s_resume_is_unconditionally_unavailable() -> None:
    with pytest.raises(u2s.U2SProtocolError, match="non-resumable"):
        u2s.load_resume()
    parser = u2s.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "train-arm",
                "--arm",
                "control",
                "--run-dir",
                "/tmp/control",
                "--cohort-contract",
                "/tmp/contract.json",
                "--resume",
                "/tmp/latest.zip",
            ]
        )


def test_optimizer_epoch_evidence_is_exact_by_arm() -> None:
    assert u2s.optimizer_epoch_accounting(
        "control",
        u2s.ROLLOUT_TRANSITIONS,
        4,
        approx_kl=0.01,
    ) == {
        "epochs_planned": 4,
        "epochs_completed": 4,
        "epochs_skipped": 0,
        "target_kl": None,
        "kl_stop_triggered": False,
        "approx_kl": 0.01,
    }
    assert u2s.optimizer_epoch_accounting(
        "conservative",
        u2s.ROLLOUT_TRANSITIONS,
        1,
        approx_kl=0.03,
    ) == {
        "epochs_planned": 2,
        "epochs_completed": 1,
        "epochs_skipped": 1,
        "target_kl": 0.015,
        "kl_stop_triggered": True,
        "approx_kl": 0.03,
    }


def test_visible_effect_and_penalty_evidence_recomputes_exactly() -> None:
    zeros = np.zeros((2, 2, 3), dtype=np.uint8)
    ones = np.ones((2, 2, 3), dtype=np.uint8)
    actions = [
        int(Actions.pickup),
        int(Actions.pickup),
        int(Actions.pickup),
        int(Actions.toggle),
        int(Actions.toggle),
    ]
    evidence = u2s.visible_interaction_effect_evidence(
        actions,
        [zeros, zeros, zeros, zeros, ones],
        [zeros, zeros, zeros, ones, ones],
        penalty_enabled=True,
    )
    assert evidence["per_action"]["pickup"] == {
        "action": int(Actions.pickup),
        "attempts": 3,
        "visible_effects": 0,
        "visible_no_effects": 3,
        "visible_effect_rate": 0.0,
        "visible_no_effect_rate": 1.0,
    }
    assert evidence["per_action"]["toggle"]["visible_effects"] == 1
    assert evidence["per_action"]["toggle"]["visible_no_effects"] == 1
    assert evidence["penalty_diagnostic"] == {
        "enabled_for_arm": True,
        "eligible_events": 2,
        "applied_count": 2,
        "capped_return": -0.02,
        "episode_cap": 0.1,
        "evaluation_reward_changed": False,
    }
