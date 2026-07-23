from types import SimpleNamespace

import numpy as np
import pytest

import dungeon_apprentice.v02_u2_lessons as u2
from dungeon_apprentice.contracts import (
    CURIOSITY_BUDGET,
    DEFAULT_CURIOSITY_SCALE,
    MINIMUM_GAMMA,
    STEP_REWARD,
    SUCCESS_REWARD,
)
from dungeon_apprentice.oracle import DungeonOracle, OracleFailure
from dungeon_apprentice.u2_seed_guard import (
    U2SeedAccessError,
    U2SeedRole,
    _new_verified_qualification_binding,
    engineering_seed_access,
    qualified_training_seed_access,
)
from dungeon_apprentice.v02_lessons import LessonEnv as FrozenU1LessonEnv
from dungeon_apprentice.v02_lessons import LessonId as FrozenU1LessonId

ENGINEERING_START = 5_200_000
FUTURE_U2_START = 15_200_000
FINAL_TEST_START = 20_000_000


@pytest.fixture
def engineering_access():
    return engineering_seed_access()


def test_u2_horizon_preserves_raw_and_discounted_reward_dominance() -> None:
    horizon = u2.LESSON_SPECS[u2.LessonId.SEPARATED_UNLOCK].max_steps
    assert horizon == 160

    raw_failure = horizon * STEP_REWARD + CURIOSITY_BUDGET
    raw_success = (horizon - 1) * STEP_REWARD + SUCCESS_REWARD
    remaining_curiosity = CURIOSITY_BUDGET
    discounted_failure = 0.0
    for step in range(horizon):
        bonus = min(DEFAULT_CURIOSITY_SCALE, remaining_curiosity)
        remaining_curiosity -= bonus
        discounted_failure += (
            STEP_REWARD + bonus
        ) * MINIMUM_GAMMA**step
    discounted_success = sum(
        STEP_REWARD * MINIMUM_GAMMA**step
        for step in range(horizon - 1)
    ) + SUCCESS_REWARD * MINIMUM_GAMMA ** (horizon - 1)

    assert raw_failure < 0.0 < raw_success
    assert raw_success > raw_failure
    assert discounted_success > discounted_failure


@pytest.mark.parametrize(
    ("u2_lesson", "frozen_lesson"),
    [
        (u2.LessonId.NAVIGATE, FrozenU1LessonId.NAVIGATE),
        (u2.LessonId.VISIBLE_UNLOCK, FrozenU1LessonId.VISIBLE_UNLOCK),
        (u2.LessonId.LOCAL_UNLOCK, FrozenU1LessonId.LOCAL_UNLOCK),
    ],
)
@pytest.mark.parametrize("seed", [0, 19, 83, 999_999])
def test_inherited_lessons_delegate_byte_exactly(u2_lesson, frozen_lesson, seed) -> None:
    frozen = FrozenU1LessonEnv(lesson=frozen_lesson)
    successor = u2.U2LessonEnv(lesson=u2_lesson)
    try:
        frozen_observation, frozen_info = frozen.reset(seed=seed)
        successor_observation, successor_info = successor.reset(seed=seed)

        assert successor.layout == frozen.layout
        assert successor.geometry_sha256 == frozen.geometry_sha256
        assert np.array_equal(successor.grid.encode(), frozen.grid.encode())
        assert successor_observation.keys() == frozen_observation.keys()
        for key in successor_observation:
            if isinstance(successor_observation[key], np.ndarray):
                assert np.array_equal(
                    successor_observation[key],
                    frozen_observation[key],
                )
            else:
                assert successor_observation[key] == frozen_observation[key]
        assert successor_info == frozen_info

        successor_result = DungeonOracle(successor).solve()
        frozen_result = DungeonOracle(frozen).solve()
        assert successor_result == frozen_result
    finally:
        frozen.close()
        successor.close()


def test_separated_unlock_is_deterministic_and_pixels_only(engineering_access) -> None:
    wrapped = u2.make_u2_pixel_env(lesson=u2.LessonId.SEPARATED_UNLOCK)
    env = wrapped.unwrapped
    try:
        observation, info = wrapped.reset(
            seed=ENGINEERING_START,
            options={
                "u2_seed_role": U2SeedRole.ENGINEERING.value,
                "u2_seed_access": engineering_access,
            },
        )
        first_layout = env.layout
        first_plan = env.pure_oracle_actions
        first_grid = env.grid.encode().copy()

        assert isinstance(observation, np.ndarray)
        assert observation.shape == (56, 56, 3)
        assert observation.dtype == np.uint8
        assert int(wrapped.action_space.n) == 7
        assert env.max_steps == 160
        assert info["lesson_id"] == u2.LessonId.SEPARATED_UNLOCK.value
        assert info["u2_seed_role"] == U2SeedRole.ENGINEERING.value
        assert first_layout is not None
        assert first_plan is not None
        assert 17 <= len(first_plan) <= 26

        repeated, repeated_info = wrapped.reset(
            seed=ENGINEERING_START,
            options={
                "u2_seed_role": U2SeedRole.ENGINEERING.value,
                "u2_seed_access": engineering_access,
            },
        )
        assert env.layout == first_layout
        assert env.pure_oracle_actions == first_plan
        assert np.array_equal(env.grid.encode(), first_grid)
        assert np.array_equal(repeated, observation)
        assert repeated_info["layout_sha256"] == info["layout_sha256"]
    finally:
        wrapped.close()


def test_engineering_seed_requires_issued_access_before_generation() -> None:
    env = u2.U2LessonEnv(lesson=u2.LessonId.SEPARATED_UNLOCK)
    try:
        with pytest.raises(U2SeedAccessError, match="engineering seed access"):
            env.reset(
                seed=ENGINEERING_START,
                options={"u2_seed_role": U2SeedRole.ENGINEERING.value},
            )
        with pytest.raises(U2SeedAccessError, match="reserved for engineering"):
            env.reset(
                seed=ENGINEERING_START,
                options={"u2_seed_role": U2SeedRole.TRAINING.value},
            )
        assert env.layout is None
    finally:
        env.close()


@pytest.mark.parametrize(
    "lesson",
    [
        u2.LessonId.NAVIGATE,
        u2.LessonId.VISIBLE_UNLOCK,
        u2.LessonId.LOCAL_UNLOCK,
    ],
)
@pytest.mark.parametrize(
    "seed",
    [
        FUTURE_U2_START,
        FINAL_TEST_START,
        5_201_000,
    ],
)
def test_inherited_env_rejects_future_final_and_unrecognized_before_generation(
    monkeypatch,
    lesson,
    seed,
) -> None:
    generation_attempted = False

    def forbidden_generation(*_args, **_kwargs):
        nonlocal generation_attempted
        generation_attempted = True
        raise AssertionError("protected layout generation was reached")

    monkeypatch.setattr(u2.U2LessonEnv, "_gen_grid", forbidden_generation)
    env = u2.U2LessonEnv(lesson=lesson)
    try:
        with pytest.raises(U2SeedAccessError):
            env.reset(seed=seed)
        assert generation_attempted is False
        assert env.layout is None
    finally:
        env.close()


def test_inherited_env_rejects_validation_without_access_and_wrong_lesson_role(
    monkeypatch,
) -> None:
    generation_attempted = False

    def forbidden_generation(*_args, **_kwargs):
        nonlocal generation_attempted
        generation_attempted = True
        raise AssertionError("protected layout generation was reached")

    monkeypatch.setattr(u2.U2LessonEnv, "_gen_grid", forbidden_generation)
    env = u2.U2LessonEnv(lesson=u2.LessonId.NAVIGATE)
    binding = _new_verified_qualification_binding(
        source_commit="a" * 40,
        report_sha256="b" * 64,
        report_byte_length=123,
        attempt_id="u2-preflight-v0.2-u2-20260723-attempt-1",
        attempt_sha256="c" * 64,
        claim_id="d" * 64,
        claim_sha256="e" * 64,
    )
    qualified_access = qualified_training_seed_access(binding)
    try:
        with pytest.raises(U2SeedAccessError, match="qualified seed access"):
            env.reset(
                seed=10_000_000,
                options={
                    "u2_seed_role": U2SeedRole.NAVIGATE_VALIDATION.value,
                },
            )
        with pytest.raises(U2SeedAccessError, match="cannot instantiate"):
            env.reset(
                seed=11_000_000,
                options={
                    "u2_seed_role": U2SeedRole.U0_VALIDATION.value,
                    "u2_seed_access": qualified_access,
                },
            )
        assert generation_attempted is False
        assert env.layout is None
    finally:
        env.close()


@pytest.mark.parametrize(
    "lesson",
    [
        u2.LessonId.NAVIGATE,
        u2.LessonId.VISIBLE_UNLOCK,
        u2.LessonId.LOCAL_UNLOCK,
    ],
)
@pytest.mark.parametrize("seed", [FUTURE_U2_START, FINAL_TEST_START])
def test_evaluator_rejects_inherited_future_and_final_before_generation(
    monkeypatch,
    lesson,
    seed,
) -> None:
    generation_attempted = False

    def forbidden_generation(*_args, **_kwargs):
        nonlocal generation_attempted
        generation_attempted = True
        raise AssertionError("protected layout generation was reached")

    monkeypatch.setattr(u2.U2LessonEnv, "_gen_grid", forbidden_generation)
    with pytest.raises(U2SeedAccessError):
        u2.evaluate_u2_lesson(object(), lesson, [seed])
    assert generation_attempted is False


@pytest.mark.parametrize(
    "lesson",
    [
        u2.LessonId.NAVIGATE,
        u2.LessonId.VISIBLE_UNLOCK,
        u2.LessonId.LOCAL_UNLOCK,
    ],
)
@pytest.mark.parametrize("seed", [FUTURE_U2_START, FINAL_TEST_START])
def test_oracle_helper_rejects_inherited_future_and_final_before_generation(
    monkeypatch,
    lesson,
    seed,
) -> None:
    generation_attempted = False

    def forbidden_generation(*_args, **_kwargs):
        nonlocal generation_attempted
        generation_attempted = True
        raise AssertionError("protected layout generation was reached")

    monkeypatch.setattr(u2.U2LessonEnv, "_gen_grid", forbidden_generation)
    with pytest.raises(U2SeedAccessError):
        u2._oracle_action_count(
            lesson,
            seed,
            size=9,
            seed_access=None,
        )
    assert generation_attempted is False


def test_generator_contract_spans_geometry_visibility_and_color(engineering_access) -> None:
    cases = [
        u2.generate_u2_case_evidence(
            ENGINEERING_START + offset,
            seed_role=U2SeedRole.ENGINEERING,
            access=engineering_access,
        )
        for offset in range(80)
    ]

    assert all(case["planner_oracle_counts_match"] for case in cases)
    assert all(17 <= case["live_oracle_actions"] <= 26 for case in cases)
    assert all(case["post_key_pre_door_turns"] >= 1 for case in cases)
    assert all(case["extra_wall_count"] == 2 for case in cases)
    assert all(not case["goal_reachable_while_locked"] for case in cases)
    assert all(case["success"] for case in cases)
    assert all(case["terminal_reason"] == "success" for case in cases)
    assert {case["divider_orientation"] for case in cases} == {
        "horizontal",
        "vertical",
    }
    assert {case["divider_index"] for case in cases} == {2, 3, 4, 5, 6}
    assert {case["approach_from_low"] for case in cases} == {False, True}
    assert {case["key_color"] for case in cases} == {
        "blue",
        "red",
        "yellow",
        "purple",
    }
    assert {case["visibility_stratum"] for case in cases} == {
        "key_hidden_door_hidden",
        "key_hidden_door_visible",
        "key_visible_door_hidden",
        "key_visible_door_visible",
    }
    assert len({case["layout_sha256"] for case in cases}) == len(cases)
    assert len({case["geometry_sha256"] for case in cases}) == len(cases)
    assert all(1 <= case["generation_attempts"] <= u2.MAX_GENERATION_ATTEMPTS for case in cases)


def test_generator_fails_at_bounded_rejection_limit(monkeypatch) -> None:
    attempts = 0

    def always_invalid(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise OracleFailure("forced rejection")

    monkeypatch.setattr(u2, "_validate_separated_candidate", always_invalid)
    env = u2.U2LessonEnv(lesson=u2.LessonId.SEPARATED_UNLOCK)
    try:
        with pytest.raises(RuntimeError, match="within 2048 attempts"):
            env.reset(
                seed=1,
                options={"u2_seed_role": U2SeedRole.TRAINING.value},
            )
    finally:
        env.close()
    assert attempts == u2.MAX_GENERATION_ATTEMPTS


def test_engineering_qualification_uses_explicit_block_only(engineering_access) -> None:
    report = u2.qualify_separated_unlock(
        range(ENGINEERING_START, ENGINEERING_START + 32),
        seed_role=U2SeedRole.ENGINEERING,
        access=engineering_access,
    )

    assert report["result"] == "passed"
    assert report["requested"] == 32
    assert report["solved"] == 32
    assert report["full_unique_layouts"] == 32
    assert report["geometry_unique_layouts"] == 32
    assert report["minimum_oracle_actions"] >= 17
    assert report["maximum_oracle_actions"] <= 26
    assert sum(report["visibility_strata"].values()) == 32
    assert report["validation_exact_overlap"] == 0


EXPECTED_PROFILES = {
    (): (0.50, 0.075, 0.075, 0.35),
    (u2.LessonId.NAVIGATE,): (0.70, 0.10, 0.10, 0.10),
    (u2.LessonId.VISIBLE_UNLOCK,): (0.45, 0.35, 0.10, 0.10),
    (u2.LessonId.LOCAL_UNLOCK,): (0.45, 0.10, 0.35, 0.10),
    (
        u2.LessonId.NAVIGATE,
        u2.LessonId.VISIBLE_UNLOCK,
    ): (0.55, 0.25, 0.10, 0.10),
    (
        u2.LessonId.NAVIGATE,
        u2.LessonId.LOCAL_UNLOCK,
    ): (0.55, 0.10, 0.25, 0.10),
    (
        u2.LessonId.VISIBLE_UNLOCK,
        u2.LessonId.LOCAL_UNLOCK,
    ): (0.40, 0.25, 0.25, 0.10),
    u2.PREREQUISITE_LESSONS: (0.45, 0.225, 0.225, 0.10),
}


@pytest.mark.parametrize(("weak", "expected"), EXPECTED_PROFILES.items())
def test_all_normal_and_recovery_profiles_are_exact(weak, expected) -> None:
    state = u2.CurriculumState(weak_prerequisites=weak)
    targets = state.targets()

    assert tuple(targets) == tuple(u2.LessonId)
    assert tuple(targets.values()) == expected
    assert sum(targets.values()) == pytest.approx(1.0)
    assert u2.TARGET_PROFILES[weak] == targets


def _simulate_scheduler(
    scheduler: u2.TransitionDeficitScheduler,
    transitions: int = 32_768,
) -> None:
    lengths = {
        u2.LessonId.NAVIGATE: 31,
        u2.LessonId.VISIBLE_UNLOCK: 11,
        u2.LessonId.LOCAL_UNLOCK: 79,
        u2.LessonId.SEPARATED_UNLOCK: 137,
    }
    collected = 0
    while collected < transitions:
        lesson, reservation = scheduler.assign()
        length = min(lengths[lesson], transitions - collected)
        for _ in range(length):
            scheduler.record_step(lesson)
        scheduler.release(lesson, reservation)
        collected += length


@pytest.mark.parametrize("weak", EXPECTED_PROFILES)
def test_scheduler_tracks_every_profile_by_transitions(weak) -> None:
    state = u2.CurriculumState(weak_prerequisites=weak)
    scheduler = u2.TransitionDeficitScheduler(state, seed=612)
    _simulate_scheduler(scheduler)

    snapshot = scheduler.snapshot()
    assert sum(snapshot["window_transitions"].values()) == 32_768
    assert scheduler.allocation_within()
    for lesson, target in state.targets().items():
        assert snapshot["realized_shares"][lesson.value] == pytest.approx(
            target,
            abs=0.01,
        )


def test_curriculum_and_scheduler_round_trip_rng_and_counts() -> None:
    state = u2.CurriculumState(
        weak_prerequisites=(
            u2.LessonId.NAVIGATE,
            u2.LessonId.LOCAL_UNLOCK,
        ),
        recovery_passes=1,
        revision=3,
    )
    scheduler = u2.TransitionDeficitScheduler(state, seed=991)
    _simulate_scheduler(scheduler, transitions=4_096)
    state_json = state.public_dict()
    scheduler_json = scheduler.state_dict()

    restored_state = u2.CurriculumState.from_dict(state_json)
    restored = u2.TransitionDeficitScheduler(restored_state, seed=1)
    restored.load_state_dict(scheduler_json)

    assert restored_state.public_dict() == state_json
    assert restored.snapshot() == scheduler.snapshot()
    assert restored.assign() == scheduler.assign()


def _evaluation(lesson: u2.LessonId, successes: int) -> u2.LessonEvaluation:
    panels = (successes // 2, successes - successes // 2)
    return u2.LessonEvaluation(
        protocol=u2.PROTOCOL,
        timestamp="2026-07-23T00:00:00Z",
        lesson_id=lesson.value,
        lesson_label=u2.LESSON_SPECS[lesson].label,
        episodes=80,
        successes=successes,
        success_rate=successes / 80,
        panel_successes=panels,
        panel_success_rates=tuple(value / 40 for value in panels),
        mean_steps=10.0,
    )


def test_weak_prerequisites_ignores_u2_only_failure() -> None:
    evaluations = {
        lesson: _evaluation(lesson, 80)
        for lesson in u2.LessonId
    }
    evaluations[u2.LessonId.SEPARATED_UNLOCK] = _evaluation(
        u2.LessonId.SEPARATED_UNLOCK,
        0,
    )
    assert u2.weak_prerequisites(evaluations) == ()

    evaluations[u2.LessonId.NAVIGATE] = _evaluation(
        u2.LessonId.NAVIGATE,
        60,
    )
    evaluations[u2.LessonId.LOCAL_UNLOCK] = _evaluation(
        u2.LessonId.LOCAL_UNLOCK,
        60,
    )
    assert u2.weak_prerequisites(evaluations) == (
        u2.LessonId.NAVIGATE,
        u2.LessonId.LOCAL_UNLOCK,
    )


class _RepeatDoneModel:
    observation_space = SimpleNamespace(shape=(56, 56, 3))

    def predict(self, _observation, *, state, episode_start, deterministic):
        assert deterministic is True
        assert episode_start.shape == (1,)
        return np.asarray(6), state


def test_evaluation_reports_visibility_wilson_actions_and_returns(
    engineering_access,
) -> None:
    result = u2.evaluate_u2_lesson(
        _RepeatDoneModel(),
        u2.LessonId.SEPARATED_UNLOCK,
        range(ENGINEERING_START, ENGINEERING_START + 4),
        seed_access=engineering_access,
    )

    assert result.episodes == 4
    assert result.successes == 0
    assert result.panel_successes == (0, 0)
    assert result.success_wilson_interval[0] == 0.0
    assert result.success_wilson_interval[1] > 0.0
    assert sum(
        item["episodes"] for item in result.visibility_strata.values()
    ) == 4
    assert result.action_histogram == {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 640,
    }
    assert result.largest_action_share == 1.0
    assert result.longest_repeated_action_run == 160
    assert result.mean_steps == 160
    assert result.mean_extrinsic_return == pytest.approx(-0.160)
    assert result.mean_curiosity_return == 0.0
    assert result.mean_path_actions_per_oracle_action > 1.0
    assert 0.0 < result.mean_coverage <= 1.0
