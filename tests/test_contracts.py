import pytest

from dungeon_apprentice.contracts import (
    CURIOSITY_BUDGET,
    DEFAULT_CURIOSITY_SCALE,
    FINAL_SEED_BASE,
    MINIMUM_GAMMA,
    PIXEL_SHAPE,
    PROTOCOL,
    STEP_REWARD,
    SUCCESS_REWARD,
    TIER_RULES,
    TRAINING_SEED_LIMIT,
    VALIDATION_SEED_BASE,
    DungeonTier,
    discounted_reward_bounds,
    evaluation_seeds,
)


def test_seed_partitions_do_not_overlap() -> None:
    validation = {
        seed for tier in DungeonTier for seed in evaluation_seeds(tier, 100, final=False)
    }
    final = {seed for tier in DungeonTier for seed in evaluation_seeds(tier, 100, final=True)}

    assert max(range(TRAINING_SEED_LIMIT)) < VALIDATION_SEED_BASE
    assert min(validation) >= VALIDATION_SEED_BASE
    assert min(final) >= FINAL_SEED_BASE
    assert validation.isdisjoint(final)


def test_pixel_contract_is_stable() -> None:
    assert PIXEL_SHAPE == (56, 56, 3)


def test_evaluation_seed_allocations_cannot_overlap_adjacent_tiers() -> None:
    with pytest.raises(ValueError, match="tier allocation"):
        evaluation_seeds(DungeonTier.NAVIGATE, 100_001)


def test_discounted_success_dominates_front_loaded_curiosity_failure() -> None:
    for tier in DungeonTier:
        success, failure = discounted_reward_bounds(tier, MINIMUM_GAMMA)
        assert success > failure, (tier, success, failure)

    unsafe_success, unsafe_failure = discounted_reward_bounds(
        DungeonTier.RETRIEVE, 0.99
    )
    assert unsafe_success < unsafe_failure


def test_v01_reward_contract_is_stable() -> None:
    assert PROTOCOL == "dungeon-apprentice-v0.1"
    assert DEFAULT_CURIOSITY_SCALE == 0.002
    assert CURIOSITY_BUDGET == 0.1


def test_reward_bounds_make_failure_unprofitable_and_success_dominant() -> None:
    """Even maximally novel failure loses; even zero-novelty success wins."""

    for rules in TIER_RULES.values():
        failed_timeout_maximum = rules.max_steps * STEP_REWARD + CURIOSITY_BUDGET
        successful_minimum = (rules.max_steps - 1) * STEP_REWARD + SUCCESS_REWARD

        assert failed_timeout_maximum < 0.0
        assert successful_minimum > 0.0
        assert successful_minimum > failed_timeout_maximum
