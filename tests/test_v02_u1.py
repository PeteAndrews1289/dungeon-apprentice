from types import SimpleNamespace

import numpy as np
import pytest

from dungeon_apprentice.v02_lessons import (
    LESSON_SPECS,
    PROTOCOL,
    CurriculumState,
    LessonEnv,
    LessonEvaluation,
    LessonId,
    RecoveryCause,
    TransitionDeficitScheduler,
    make_training_env,
    qualify_local_unlock,
)
from dungeon_apprentice.v02_sentinel import (
    LessonId as SentinelLessonId,
)
from dungeon_apprentice.v02_sentinel import (
    V02SentinelEnv,
)
from dungeon_apprentice.v02_u1 import ParentProvenance, _CallbackFactory


def _layout_core(layout):
    return (
        layout.start,
        layout.start_direction,
        layout.goal,
        layout.key,
        layout.door,
        layout.key_color,
        layout.walls,
    )


@pytest.mark.parametrize("seed", [0, 1, 9, 12345, 999_999, 11_000_079])
def test_u0_profile_is_pixel_exact_with_the_confirmed_sentinel(seed) -> None:
    frozen = V02SentinelEnv(lesson=SentinelLessonId.VISIBLE_UNLOCK)
    child = LessonEnv(lesson=LessonId.VISIBLE_UNLOCK)
    try:
        frozen.reset(seed=seed)
        child.reset(seed=seed)
        assert frozen.layout is not None
        assert child.layout is not None
        assert _layout_core(child.layout) == _layout_core(frozen.layout)
        assert np.array_equal(child.grid.encode(), frozen.grid.encode())
    finally:
        frozen.close()
        child.close()


def test_local_unlock_is_deterministic_visible_hidden_and_nontrivial() -> None:
    env = LessonEnv(lesson=LessonId.LOCAL_UNLOCK)
    try:
        _, info = env.reset(seed=5_100_042)
        first = env.layout
        first_plan = env.pure_oracle_actions
        assert first is not None
        assert first_plan is not None
        assert env.agent_sees(*first.key)
        assert not env.agent_sees(*first.door)
        assert 9 <= len(first_plan) <= 18
        assert info["protocol"] == PROTOCOL
        assert info["geometry_sha256"] == env.geometry_sha256

        env.reset(seed=5_100_042)
        assert env.layout == first
        assert env.pure_oracle_actions == first_plan
    finally:
        env.close()


def test_local_unlock_development_qualification() -> None:
    report = qualify_local_unlock(100)
    assert report["result"] == "passed"
    assert report["solved"] == 100
    assert report["full_unique_layouts"] == 100
    assert report["geometry_unique_layouts"] >= 95
    assert report["minimum_oracle_actions"] == 9
    assert report["maximum_oracle_actions"] == 18


def _simulate_scheduler(
    scheduler: TransitionDeficitScheduler, steps: int = 32_768
) -> dict[str, float]:
    collected = 0
    lengths = {
        LessonId.NAVIGATE: 30,
        LessonId.VISIBLE_UNLOCK: 12,
        LessonId.LOCAL_UNLOCK: 128,
    }
    while collected < steps:
        lesson, reservation = scheduler.assign()
        for _ in range(lengths[lesson]):
            scheduler.record_step(lesson)
        scheduler.release(lesson, reservation)
        collected += lengths[lesson]
    return scheduler.snapshot()["realized_shares"]


@pytest.mark.parametrize(
    ("cause", "expected"),
    [
        (
            None,
            {
                LessonId.NAVIGATE: 0.50,
                LessonId.VISIBLE_UNLOCK: 0.15,
                LessonId.LOCAL_UNLOCK: 0.35,
            },
        ),
        (
            RecoveryCause.NAVIGATE,
            {
                LessonId.NAVIGATE: 0.75,
                LessonId.VISIBLE_UNLOCK: 0.10,
                LessonId.LOCAL_UNLOCK: 0.15,
            },
        ),
        (
            RecoveryCause.VISIBLE_UNLOCK,
            {
                LessonId.NAVIGATE: 0.50,
                LessonId.VISIBLE_UNLOCK: 0.35,
                LessonId.LOCAL_UNLOCK: 0.15,
            },
        ),
        (
            RecoveryCause.BOTH,
            {
                LessonId.NAVIGATE: 0.65,
                LessonId.VISIBLE_UNLOCK: 0.25,
                LessonId.LOCAL_UNLOCK: 0.10,
            },
        ),
    ],
)
def test_scheduler_tracks_each_normal_and_recovery_mix(cause, expected) -> None:
    scheduler = TransitionDeficitScheduler(
        CurriculumState(recovery_cause=cause), seed=72
    )
    shares = _simulate_scheduler(scheduler)
    for lesson, target in expected.items():
        assert shares[lesson.value] == pytest.approx(target, abs=0.01)
    assert scheduler.allocation_within()


def test_scheduler_state_round_trips_without_losing_counts_or_rng() -> None:
    state = CurriculumState()
    first = TransitionDeficitScheduler(state, seed=73)
    _simulate_scheduler(first, 5_000)
    saved = first.state_dict()
    restored = TransitionDeficitScheduler(state, seed=999)
    restored.load_state_dict(saved)

    assert restored.snapshot() == first.snapshot()
    assert restored.assign() == first.assign()


def test_training_reset_rejects_an_exact_reserved_visual_layout() -> None:
    state = CurriculumState()
    first_scheduler = TransitionDeficitScheduler(state, seed=81)
    probe = make_training_env(scheduler=first_scheduler, seed=82)
    try:
        _, first_info = probe.reset(seed=83)
    finally:
        probe.close()

    second_state = CurriculumState()
    second_scheduler = TransitionDeficitScheduler(second_state, seed=81)
    selected_lesson = LessonId(first_info["lesson_id"])
    guarded = make_training_env(
        scheduler=second_scheduler,
        seed=82,
        forbidden_layout_hashes={
            selected_lesson: frozenset({first_info["layout_sha256"]})
        },
    )
    try:
        _, guarded_info = guarded.reset(seed=83)
    finally:
        guarded.close()

    assert guarded_info["reserved_layout_rejections"] == 1
    assert guarded_info["layout_sha256"] != first_info["layout_sha256"]


class _FakeBaseCallback:
    def __init__(self, *, verbose: int) -> None:
        self.verbose = verbose


def _parent() -> ParentProvenance:
    return ParentProvenance(
        checkpoint="parent.zip",
        checkpoint_sha256="a" * 64,
        sidecar="parent.json",
        manifest="manifest.json",
        source_commit="b" * 40,
        training_seed=20260725,
        trained_timesteps=491_520,
        n_updates=960,
        confirmation_report="report.json",
        confirmation_sha256="c" * 64,
        confirmation_protocol="confirmation",
        confirmation_verdict="passed",
        confirmation_completed_at="2026-07-22T00:00:00Z",
    )


def _evaluation(lesson: LessonId, successes: int) -> LessonEvaluation:
    panel_successes = (successes // 2, successes - successes // 2)
    return LessonEvaluation(
        protocol=PROTOCOL,
        timestamp="2026-07-22T00:00:00Z",
        lesson_id=lesson.value,
        lesson_label=LESSON_SPECS[lesson].label,
        episodes=80,
        successes=successes,
        success_rate=successes / 80,
        panel_successes=panel_successes,
        panel_success_rates=tuple(value / 40 for value in panel_successes),
        mean_steps=10.0,
    )


def _callback(tmp_path, state: CurriculumState):
    scheduler = TransitionDeficitScheduler(state, seed=74)
    callback_type = _CallbackFactory.create(_FakeBaseCallback)
    callback = callback_type(
        run_directory=tmp_path,
        state=state,
        scheduler=scheduler,
        parent=_parent(),
        segment={},
        effective_config={},
        evaluation_interval=32_768,
        checkpoint_interval=32_768,
        frame_interval=2_048,
        minimum_free_bytes=0,
        started_at="2026-07-22T00:00:00Z",
        initial_trained_timesteps=491_520,
        initial_updates=960,
        child_start_timesteps=491_520,
        keep_checkpoints=5,
    )
    callback.model = SimpleNamespace(num_timesteps=491_520, _n_updates=960)
    return callback


def test_u0_forgetting_starts_targeted_recovery_instead_of_passing_u1(tmp_path) -> None:
    state = CurriculumState(consecutive_passes=1)
    callback = _callback(tmp_path, state)
    evaluations = {
        LessonId.NAVIGATE: _evaluation(LessonId.NAVIGATE, 80),
        LessonId.VISIBLE_UNLOCK: _evaluation(LessonId.VISIBLE_UNLOCK, 60),
        LessonId.LOCAL_UNLOCK: _evaluation(LessonId.LOCAL_UNLOCK, 80),
    }

    decision = callback._apply_decision(evaluations, allocation_ok=True)

    assert "Recovery started" in decision
    assert state.recovery_cause is RecoveryCause.VISIBLE_UNLOCK
    assert state.consecutive_passes == 0
    assert state.targets()[LessonId.VISIBLE_UNLOCK] == 0.35


def test_u1_mastery_requires_two_full_cumulative_passes(tmp_path) -> None:
    state = CurriculumState()
    callback = _callback(tmp_path, state)
    callback._checkpoint = lambda *_args: tmp_path / "mastery.zip"
    evaluations = {lesson: _evaluation(lesson, 80) for lesson in LessonId}

    first = callback._apply_decision(evaluations, allocation_ok=True)
    second = callback._apply_decision(evaluations, allocation_ok=True)

    assert "confirmation required" in first
    assert "mastered" in second
    assert state.mastered


def test_recovery_requires_two_clean_anchor_exams(tmp_path) -> None:
    state = CurriculumState(recovery_cause=RecoveryCause.BOTH)
    callback = _callback(tmp_path, state)
    evaluations = {lesson: _evaluation(lesson, 80) for lesson in LessonId}

    first = callback._apply_decision(evaluations, allocation_ok=True)
    second = callback._apply_decision(evaluations, allocation_ok=True)

    assert "confirmation required" in first
    assert "normal U1 practice resumes" in second
    assert state.recovery_cause is None
    assert state.consecutive_passes == 0
