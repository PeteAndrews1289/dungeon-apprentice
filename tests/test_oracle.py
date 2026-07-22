import pytest

from dungeon_apprentice.contracts import DungeonTier
from dungeon_apprentice.oracle import solve_seed


@pytest.mark.parametrize("tier", list(DungeonTier))
def test_oracle_solves_qualified_levels_deterministically(tier: DungeonTier) -> None:
    first_pass = [solve_seed(tier, seed) for seed in range(25)]
    second_pass = [solve_seed(tier, seed) for seed in range(25)]

    assert all(result.success for result in first_pass)
    assert [result.actions for result in first_pass] == [result.actions for result in second_pass]
    assert [result.layout_sha256 for result in first_pass] == [
        result.layout_sha256 for result in second_pass
    ]
    assert len({result.layout_sha256 for result in first_pass}) == 25

