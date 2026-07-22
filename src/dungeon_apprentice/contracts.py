"""Frozen Version-0 constants shared by the game, trainer, and evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

PROTOCOL = "dungeon-apprentice-v0"
OBSERVATION_SIZE = 7
PIXEL_TILE_SIZE = 8
PIXEL_SHAPE = (OBSERVATION_SIZE * PIXEL_TILE_SIZE,) * 2 + (3,)
SUCCESS_REWARD = 1.0
STEP_REWARD = -0.001
VALIDATION_SEED_BASE = 10_000_000
FINAL_SEED_BASE = 20_000_000
TRAINING_SEED_LIMIT = 1_000_000
PROMOTION_THRESHOLD = 0.90
RETENTION_THRESHOLD = 0.80
NEWEST_TIER_FRACTION = 0.70
DEFAULT_CURIOSITY_SCALE = 0.01


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
    base = FINAL_SEED_BASE if final else VALIDATION_SEED_BASE
    return tuple(base + int(tier) * 100_000 + offset for offset in range(count))
