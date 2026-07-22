import numpy as np
import pytest

from dungeon_apprentice.contracts import PIXEL_SHAPE, STEP_REWARD, SUCCESS_REWARD, DungeonTier
from dungeon_apprentice.env import (
    CurriculumEnv,
    CurriculumState,
    DungeonApprenticeEnv,
    EpisodicPixelCuriosity,
    make_pixel_env,
)
from dungeon_apprentice.oracle import DungeonOracle


@pytest.mark.parametrize("tier", list(DungeonTier))
def test_same_seed_generates_same_layout(tier: DungeonTier) -> None:
    env = DungeonApprenticeEnv(tier=tier)
    try:
        env.reset(seed=12345)
        first = env.layout
        env.reset(seed=12345)
        second = env.layout
    finally:
        env.close()

    assert first == second


@pytest.mark.parametrize("tier", list(DungeonTier))
def test_seeded_generator_produces_layout_diversity(tier: DungeonTier) -> None:
    env = DungeonApprenticeEnv(tier=tier)
    try:
        hashes = set()
        for seed in range(20):
            env.reset(seed=seed)
            assert env.layout is not None
            hashes.add(env.layout.layout_sha256)
    finally:
        env.close()

    assert len(hashes) == 20


def test_policy_observation_is_pixels_only() -> None:
    env = make_pixel_env(tier=DungeonTier.RETRIEVE)
    try:
        observation, info = env.reset(seed=7)
    finally:
        env.close()

    assert isinstance(observation, np.ndarray)
    assert observation.shape == PIXEL_SHAPE
    assert observation.dtype == np.uint8
    assert info["seed"] == 7


@pytest.mark.parametrize("tier", list(DungeonTier))
def test_only_success_receives_positive_reward(tier: DungeonTier) -> None:
    env = DungeonApprenticeEnv(tier=tier)
    try:
        env.reset(seed=91)
        result = DungeonOracle(env).solve()
    finally:
        env.close()

    rewards_before_success = result.total_reward - SUCCESS_REWARD
    assert result.success
    assert rewards_before_success == pytest.approx(STEP_REWARD * (result.steps - 1))


def test_curriculum_uses_only_training_seed_partition() -> None:
    state = CurriculumState(max_tier=DungeonTier.RETRIEVE)
    env = CurriculumEnv(DungeonApprenticeEnv(), seed=4, state=state)
    try:
        seen_tiers = set()
        for _ in range(100):
            _, info = env.reset()
            seen_tiers.add(info["tier"])
            assert 0 <= info["seed"] < 1_000_000
    finally:
        env.close()

    assert seen_tiers == {0, 1, 2}


def test_retrieve_requires_carrying_relic_at_entrance() -> None:
    env = DungeonApprenticeEnv(tier=DungeonTier.RETRIEVE)
    try:
        env.reset(seed=11)
        assert env.layout is not None
        env.agent_pos = env.layout.goal
        _, reward, terminated, _, info = env.step(int(env.actions.done))
    finally:
        env.close()

    assert not terminated
    assert not info["success"]
    assert reward == STEP_REWARD


def test_curiosity_uses_pixels_without_changing_observation() -> None:
    plain = make_pixel_env(tier=DungeonTier.NAVIGATE)
    curious = EpisodicPixelCuriosity(
        make_pixel_env(tier=DungeonTier.NAVIGATE),
        scale=0.01,
    )
    try:
        plain_observation, _ = plain.reset(seed=77)
        curious_observation, _ = curious.reset(seed=77)
        np.testing.assert_array_equal(plain_observation, curious_observation)
        plain_next, plain_reward, *_ = plain.step(6)
        curious_next, curious_reward, _, _, curious_info = curious.step(6)
    finally:
        plain.close()
        curious.close()

    np.testing.assert_array_equal(plain_next, curious_next)
    assert curious_info["extrinsic_reward"] == plain_reward
    assert curious_info["curiosity_reward"] > 0.0
    assert curious_reward == pytest.approx(
        curious_info["extrinsic_reward"] + curious_info["curiosity_reward"]
    )
