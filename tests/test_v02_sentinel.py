from types import SimpleNamespace

import numpy as np
import pytest

import dungeon_apprentice.v02_sentinel as sentinel_module
from dungeon_apprentice.contracts import PIXEL_SHAPE, STEP_REWARD, SUCCESS_REWARD
from dungeon_apprentice.oracle import DungeonOracle
from dungeon_apprentice.v02_sentinel import (
    PROTOCOL,
    LessonEvaluation,
    LessonId,
    SentinelCurriculumState,
    TransitionDeficitScheduler,
    V02SentinelEnv,
    _CallbackFactory,
    make_sentinel_pixel_env,
    qualify_sentinel,
)


def test_visible_unlock_is_deterministic_visible_and_complete() -> None:
    env = V02SentinelEnv(lesson=LessonId.VISIBLE_UNLOCK)
    try:
        _, info = env.reset(seed=12345)
        first = env.layout
        assert first is not None
        assert env.agent_sees(*first.key)
        assert env.agent_sees(*first.door)
        result = DungeonOracle(env).solve()
        assert 5 <= result.steps <= 10
        assert result.success
        assert result.total_reward == pytest.approx(
            STEP_REWARD * (result.steps - 1) + SUCCESS_REWARD
        )

        env.reset(seed=12345)
        assert env.layout == first
        assert info["protocol"] == PROTOCOL
        assert info["lesson_id"] == LessonId.VISIBLE_UNLOCK.value
    finally:
        env.close()


def test_sentinel_policy_observation_remains_pixels_only() -> None:
    env = make_sentinel_pixel_env(lesson=LessonId.VISIBLE_UNLOCK)
    try:
        observation, info = env.reset(seed=9)
    finally:
        env.close()

    assert isinstance(observation, np.ndarray)
    assert observation.shape == PIXEL_SHAPE
    assert observation.dtype == np.uint8
    assert info["lesson_id"] == LessonId.VISIBLE_UNLOCK.value


def _simulate_scheduler(scheduler: TransitionDeficitScheduler, steps: int) -> dict[str, float]:
    collected = 0
    while collected < steps:
        lesson, reservation = scheduler.assign()
        episode_length = 30 if lesson is LessonId.NAVIGATE else 128
        for _ in range(episode_length):
            scheduler.record_step(lesson)
        scheduler.release(lesson, reservation)
        collected += episode_length
    return scheduler.snapshot()["realized_shares"]


def test_scheduler_balances_transitions_not_unequal_episode_counts() -> None:
    state = SentinelCurriculumState(active_lesson=LessonId.VISIBLE_UNLOCK)
    scheduler = TransitionDeficitScheduler(state, seed=4)
    shares = _simulate_scheduler(scheduler, 32_768)

    assert shares[LessonId.NAVIGATE.value] == pytest.approx(0.5, abs=0.01)
    assert shares[LessonId.VISIBLE_UNLOCK.value] == pytest.approx(0.5, abs=0.01)
    assert scheduler.allocation_within()


def test_scheduler_switches_to_declared_recovery_targets() -> None:
    state = SentinelCurriculumState(
        active_lesson=LessonId.VISIBLE_UNLOCK,
        recovery=True,
    )
    scheduler = TransitionDeficitScheduler(state, seed=5)
    shares = _simulate_scheduler(scheduler, 32_768)

    assert shares[LessonId.NAVIGATE.value] == pytest.approx(0.75, abs=0.01)
    assert shares[LessonId.VISIBLE_UNLOCK.value] == pytest.approx(0.25, abs=0.01)


def test_sentinel_generator_qualification_covers_both_lessons() -> None:
    report = qualify_sentinel(100)
    assert report["result"] == "passed"
    assert report["solved_levels"] == 200
    assert {item["lesson_id"] for item in report["lessons"]} == {
        LessonId.NAVIGATE.value,
        LessonId.VISIBLE_UNLOCK.value,
    }


class _FakeBaseCallback:
    def __init__(self, *, verbose: int) -> None:
        self.verbose = verbose


@pytest.mark.parametrize(
    ("last_evaluation_step", "expected_triggers"),
    [(32_768, []), (None, ["final"])],
)
def test_finalize_does_not_duplicate_an_exam_at_the_same_trained_boundary(
    tmp_path, last_evaluation_step, expected_triggers
) -> None:
    state = SentinelCurriculumState(mastered=True)
    scheduler = TransitionDeficitScheduler(state, seed=17)
    callback_type = _CallbackFactory.create(_FakeBaseCallback)
    callback = callback_type(
        run_directory=tmp_path,
        state=state,
        scheduler=scheduler,
        evaluation_interval=32_768,
        checkpoint_interval=32_768,
        evaluation_seed_count=80,
        frame_interval=2_048,
        minimum_free_bytes=0,
        started_at="2026-07-22T00:00:00Z",
        effective_config={},
    )
    callback.model = SimpleNamespace(_n_updates=0, num_timesteps=32_768)
    callback.trained_timesteps = 32_768
    callback._last_evaluation_step = last_evaluation_step
    triggers = []
    callback._evaluate_and_advance = triggers.append
    callback._checkpoint = lambda *_args: tmp_path / "final.zip"
    callback._write_status = lambda _phase: None

    callback.finalize("completed")

    assert triggers == expected_triggers


def test_mastery_preserves_the_completed_practice_allocation(
    tmp_path, monkeypatch
) -> None:
    state = SentinelCurriculumState(
        active_lesson=LessonId.VISIBLE_UNLOCK,
        consecutive_passes=1,
    )
    scheduler = TransitionDeficitScheduler(state, seed=18)
    for _ in range(100):
        scheduler.record_step(LessonId.NAVIGATE)
        scheduler.record_step(LessonId.VISIBLE_UNLOCK)
    callback_type = _CallbackFactory.create(_FakeBaseCallback)
    callback = callback_type(
        run_directory=tmp_path,
        state=state,
        scheduler=scheduler,
        evaluation_interval=32_768,
        checkpoint_interval=32_768,
        evaluation_seed_count=80,
        frame_interval=2_048,
        minimum_free_bytes=0,
        started_at="2026-07-22T00:00:00Z",
        effective_config={},
    )
    callback.model = SimpleNamespace(_n_updates=1, num_timesteps=32_768)
    callback.trained_timesteps = 32_768
    checkpoint = tmp_path / "checkpoints" / "step.zip"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"trained parameters")
    callback._evaluation_checkpoint = lambda: checkpoint
    callback._checkpoint = lambda *_args: checkpoint

    def passing_evaluation(_model, lesson, _seeds, **_kwargs):
        return LessonEvaluation(
            protocol=PROTOCOL,
            timestamp="2026-07-22T00:00:00Z",
            lesson_id=lesson.value,
            lesson_label=lesson.label,
            episodes=80,
            successes=80,
            success_rate=1.0,
            panel_success_rates=(1.0, 1.0),
            mean_steps=10.0,
        )

    monkeypatch.setattr(sentinel_module, "evaluate_lesson", passing_evaluation)

    callback._evaluate_and_advance("scheduled")

    allocation = scheduler.snapshot()
    assert state.mastered
    assert allocation["window_transitions"] == {
        LessonId.NAVIGATE.value: 100,
        LessonId.VISIBLE_UNLOCK.value: 100,
    }
    assert allocation["realized_shares"] == {
        LessonId.NAVIGATE.value: 0.5,
        LessonId.VISIBLE_UNLOCK.value: 0.5,
    }
