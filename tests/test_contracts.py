from dungeon_apprentice.contracts import (
    FINAL_SEED_BASE,
    PIXEL_SHAPE,
    TRAINING_SEED_LIMIT,
    VALIDATION_SEED_BASE,
    DungeonTier,
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

