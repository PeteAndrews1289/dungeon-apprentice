"""Command-line generator qualification."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass

from dungeon_apprentice.contracts import PROTOCOL, DungeonTier
from dungeon_apprentice.oracle import OracleFailure, solve_seed


@dataclass(frozen=True)
class TierQualification:
    tier: int
    label: str
    seeds: int
    successes: int
    minimum_actions: int
    median_actions: float
    maximum_actions: int
    unique_layouts: int


def qualify(seed_count: int, *, start_seed: int = 0, size: int = 9) -> dict[str, object]:
    if seed_count <= 0:
        raise ValueError("seed count must be positive")
    tiers: list[TierQualification] = []
    failures: list[dict[str, object]] = []
    for tier in DungeonTier:
        results = []
        for offset in range(seed_count):
            seed = start_seed + int(tier) * seed_count + offset
            try:
                results.append(solve_seed(tier, seed, size=size))
            except (OracleFailure, RuntimeError) as error:
                failures.append({"tier": int(tier), "seed": seed, "error": str(error)})
        action_counts = [result.steps for result in results]
        tiers.append(
            TierQualification(
                tier=int(tier),
                label=tier.label,
                seeds=seed_count,
                successes=len(results),
                minimum_actions=min(action_counts) if action_counts else 0,
                median_actions=statistics.median(action_counts) if action_counts else 0.0,
                maximum_actions=max(action_counts) if action_counts else 0,
                unique_layouts=len({result.layout_sha256 for result in results}),
            )
        )
    return {
        "protocol": PROTOCOL,
        "result": "passed" if not failures else "failed",
        "requested_levels": seed_count * len(DungeonTier),
        "solved_levels": sum(item.successes for item in tiers),
        "tiers": [asdict(item) for item in tiers],
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100, help="levels per tier")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=9)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = qualify(args.seeds, start_seed=args.start_seed, size=args.size)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    raise SystemExit(0 if result["result"] == "passed" else 1)


if __name__ == "__main__":
    main()

