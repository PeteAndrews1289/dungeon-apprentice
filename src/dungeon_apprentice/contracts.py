"""Frozen Version-0.1 constants shared by the game, trainer, and evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

PROTOCOL = "dungeon-apprentice-v0.1"
CHECKPOINT_SCHEMA_VERSION = 1
OBSERVATION_SIZE = 7
PIXEL_TILE_SIZE = 8
PIXEL_SHAPE = (OBSERVATION_SIZE * PIXEL_TILE_SIZE,) * 2 + (3,)
SUCCESS_REWARD = 1.0
STEP_REWARD = -0.001
VALIDATION_SEED_BASE = 10_000_000
FINAL_SEED_BASE = 20_000_000
TRAINING_SEED_LIMIT = 1_000_000
EVALUATION_TIER_STRIDE = 100_000
PROMOTION_THRESHOLD = 0.90
RETENTION_THRESHOLD = 0.80
NEWEST_TIER_FRACTION = 0.70
DEFAULT_CURIOSITY_SCALE = 0.002
CURIOSITY_BUDGET = 0.1
MINIMUM_GAMMA = 0.995


class DungeonTier(IntEnum):
    """The cumulative Version-0 capability ladder."""

    NAVIGATE = 0
    UNLOCK = 1
    RETRIEVE = 2

    @property
    def label(self) -> str:
        return {
            self.NAVIGATE: "Navigate",
            self.UNLOCK: "Unlock",
            self.RETRIEVE: "Retrieve",
        }[self]

    @property
    def objective(self) -> str:
        return {
            self.NAVIGATE: "Reach the green exit",
            self.UNLOCK: "Find the key, unlock the door, and reach the exit",
            self.RETRIEVE: "Recover the relic and return it to the entrance",
        }[self]


@dataclass(frozen=True)
class TierRules:
    tier: DungeonTier
    max_steps: int
    obstacle_count: int


TIER_RULES = {
    DungeonTier.NAVIGATE: TierRules(DungeonTier.NAVIGATE, max_steps=128, obstacle_count=7),
    DungeonTier.UNLOCK: TierRules(DungeonTier.UNLOCK, max_steps=256, obstacle_count=5),
    DungeonTier.RETRIEVE: TierRules(DungeonTier.RETRIEVE, max_steps=384, obstacle_count=5),
}


def evaluation_seeds(tier: DungeonTier, count: int, *, final: bool = False) -> tuple[int, ...]:
    """Return the immutable validation or untouched-final seed partition."""

    if count <= 0:
        raise ValueError("evaluation seed count must be positive")
    if count > EVALUATION_TIER_STRIDE:
        raise ValueError(
            "evaluation seed count cannot exceed the "
            f"{EVALUATION_TIER_STRIDE}-seed tier allocation"
        )
    base = FINAL_SEED_BASE if final else VALIDATION_SEED_BASE
    return tuple(base + int(tier) * EVALUATION_TIER_STRIDE + offset for offset in range(count))


def discounted_reward_bounds(
    tier: DungeonTier, gamma: float
) -> tuple[float, float]:
    """Return worst late-success and best early-curiosity failure from episode start."""

    horizon = TIER_RULES[tier].max_steps
    success = sum(STEP_REWARD * gamma**step for step in range(horizon - 1))
    success += SUCCESS_REWARD * gamma ** (horizon - 1)

    remaining_curiosity = CURIOSITY_BUDGET
    failure = 0.0
    for step in range(horizon):
        bonus = min(DEFAULT_CURIOSITY_SCALE, remaining_curiosity)
        remaining_curiosity -= bonus
        failure += (STEP_REWARD + bonus) * gamma**step
    return success, failure
