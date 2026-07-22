import numpy as np
import pytest

from dungeon_apprentice.contracts import PIXEL_SHAPE, STEP_REWARD, SUCCESS_REWARD
from dungeon_apprentice.oracle import DungeonOracle
from dungeon_apprentice.v02_sentinel import (
    PROTOCOL,
    LessonId,
    SentinelCurriculumState,
    TransitionDeficitScheduler,
    V02SentinelEnv,
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
