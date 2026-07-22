import hashlib

import gymnasium as gym
import numpy as np
import pytest

from dungeon_apprentice.contracts import (
    CURIOSITY_BUDGET,
    DEFAULT_CURIOSITY_SCALE,
    PIXEL_SHAPE,
    STEP_REWARD,
    SUCCESS_REWARD,
    TIER_RULES,
    DungeonTier,
)
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


@pytest.mark.parametrize(
    "tier,layout_hash,pixel_hash",
    [
        (
            DungeonTier.NAVIGATE,
            "8057ac6608171ac6b206e4f9cee4c3704383ca77c0bf591adcf2cee2a166d62e",
            "e043627b4fa34e6bf52035e8180fd303829c0c2ffb64919c5b6c336cd00761ca",
        ),
        (
            DungeonTier.UNLOCK,
            "fc376c22fdbf2ba547971007328021c908ecc1e3ac9c3914d55c5202b65f659d",
            "4951edc50dabcae81bc9d3f23db14f23ff1e3aa0f4cf0503fd01650223d6887a",
        ),
        (
            DungeonTier.RETRIEVE,
            "8117db40c64422a8c53080ec577a5b26bfa3219f6e1e647a4513efb45ff0e81a",
            "4951edc50dabcae81bc9d3f23db14f23ff1e3aa0f4cf0503fd01650223d6887a",
        ),
    ],
)
def test_seeded_layout_and_pixels_match_v01_golden_contract(
    tier: DungeonTier,
    layout_hash: str,
    pixel_hash: str,
) -> None:
    raw = DungeonApprenticeEnv(tier=tier)
    pixels = make_pixel_env(tier=tier)
    try:
        raw.reset(seed=12345)
        observation, _ = pixels.reset(seed=12345)
        assert raw.layout is not None
        assert raw.layout.layout_sha256 == layout_hash
        assert hashlib.sha256(observation.tobytes()).hexdigest() == pixel_hash
    finally:
        raw.close()
        pixels.close()


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


def test_noop_view_earns_no_curiosity_and_does_not_change_observation() -> None:
    plain = make_pixel_env(tier=DungeonTier.NAVIGATE)
    curious = EpisodicPixelCuriosity(make_pixel_env(tier=DungeonTier.NAVIGATE))
    try:
        plain_observation, _ = plain.reset(seed=77)
        curious_observation, reset_info = curious.reset(seed=77)
        np.testing.assert_array_equal(plain_observation, curious_observation)
        no_op = int(curious.unwrapped.actions.done)
        plain_next, plain_reward, *_ = plain.step(no_op)
        curious_next, curious_reward, _, _, curious_info = curious.step(no_op)
    finally:
        plain.close()
        curious.close()

    np.testing.assert_array_equal(plain_next, curious_next)
    np.testing.assert_array_equal(curious_observation, curious_next)
    assert reset_info["curiosity_budget_used"] == 0.0
    assert reset_info["curiosity_budget_remaining"] == CURIOSITY_BUDGET
    assert curious_info["extrinsic_reward"] == plain_reward
    assert curious_info["curiosity_reward"] == 0.0
    assert not curious_info["curiosity_observation_novel"]
    assert curious_info["curiosity_budget_used"] == 0.0
    assert curious_info["curiosity_budget_remaining"] == CURIOSITY_BUDGET
    assert curious_reward == pytest.approx(
        curious_info["extrinsic_reward"] + curious_info["curiosity_reward"]
    )


def test_a_repeated_spin_cannot_farm_curiosity() -> None:
    env = EpisodicPixelCuriosity(make_pixel_env(tier=DungeonTier.NAVIGATE))
    try:
        env.reset(seed=77)
        turn_right = int(env.unwrapped.actions.right)
        first_cycle = [env.step(turn_right)[4] for _ in range(4)]
        second_cycle = [env.step(turn_right)[4] for _ in range(4)]
    finally:
        env.close()

    for info in first_cycle:
        expected = DEFAULT_CURIOSITY_SCALE if info["curiosity_observation_novel"] else 0.0
        assert info["curiosity_reward"] == pytest.approx(expected)
    assert all(info["curiosity_reward"] == 0.0 for info in second_cycle)
    assert all(not info["curiosity_observation_novel"] for info in second_cycle)


class _UniquePixelEnv(gym.Env):
    """Minimal source of a different pixel observation on every step."""

    observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 1, 1), dtype=np.uint8)
    action_space = gym.spaces.Discrete(1)

    def __init__(self) -> None:
        super().__init__()
        self._value = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._value = 0
        return np.array([[[self._value]]], dtype=np.uint8), {}

    def step(self, action: int):
        self._value += 1
        observation = np.array([[[self._value]]], dtype=np.uint8)
        return observation, STEP_REWARD, False, False, {}


def test_curiosity_budget_is_a_hard_cap_including_partial_final_payment() -> None:
    env = EpisodicPixelCuriosity(_UniquePixelEnv(), scale=0.002, budget=0.005)
    try:
        _, reset_info = env.reset()
        transitions = [env.step(0) for _ in range(6)]
    finally:
        env.close()

    rewards = [transition[4]["curiosity_reward"] for transition in transitions]
    final_info = transitions[-1][4]
    assert reset_info["curiosity_budget_used"] == 0.0
    assert rewards == pytest.approx([0.002, 0.002, 0.001, 0.0, 0.0, 0.0])
    assert sum(rewards) == pytest.approx(0.005)
    assert final_info["curiosity_budget_used"] == pytest.approx(0.005)
    assert final_info["curiosity_budget_remaining"] == 0.0
    assert final_info["curiosity_unique_observations"] == 7


@pytest.mark.parametrize(
    "scale,budget",
    [(-0.001, 0.1), (0.001, -0.1), (0.001, CURIOSITY_BUDGET + 0.001)],
)
def test_curiosity_rejects_unsafe_reward_parameters(scale: float, budget: float) -> None:
    with pytest.raises(ValueError):
        EpisodicPixelCuriosity(_UniquePixelEnv(), scale=scale, budget=budget)


@pytest.mark.parametrize("tier", list(DungeonTier))
def test_oracle_completion_records_trainer_only_milestones(tier: DungeonTier) -> None:
    env = DungeonApprenticeEnv(tier=tier)
    try:
        env.reset(seed=203)
        DungeonOracle(env).solve()
        info = env._evidence_info(success=True)
    finally:
        env.close()

    milestones = info["milestones"]
    assert milestones["success"]
    assert info["unique_cells"] >= 2
    if tier >= DungeonTier.UNLOCK:
        assert milestones["key_picked_up"]
        assert milestones["door_opened"]
    if tier is DungeonTier.RETRIEVE:
        assert milestones["relic_picked_up"]
        assert milestones["entrance_revisited"]


def test_collision_and_ineffective_interaction_telemetry_stays_out_of_observation() -> None:
    env = DungeonApprenticeEnv(tier=DungeonTier.NAVIGATE)
    try:
        observation, _ = env.reset(seed=19)
        env.agent_pos = (1, 1)
        env.agent_dir = 2
        next_observation, _, _, _, collision_info = env.step(int(env.actions.forward))
        _, _, _, _, interaction_info = env.step(int(env.actions.pickup))
    finally:
        env.close()

    assert isinstance(observation, dict)
    assert isinstance(next_observation, dict)
    assert "collisions" not in observation
    assert "milestones" not in next_observation
    assert collision_info["collisions"] == 1
    assert interaction_info["ineffective_interactions"] == 1


def test_action_horizon_is_terminal_failure_not_bootstrapped_truncation() -> None:
    env = DungeonApprenticeEnv(tier=DungeonTier.NAVIGATE)
    try:
        env.reset(seed=19)
        transition = None
        for _ in range(TIER_RULES[DungeonTier.NAVIGATE].max_steps):
            transition = env.step(int(env.actions.done))
    finally:
        env.close()

    assert transition is not None
    _, reward, terminated, truncated, info = transition
    assert reward == STEP_REWARD
    assert terminated is True
    assert truncated is False
    assert info["success"] is False
    assert info["terminal_reason"] == "time_limit"
