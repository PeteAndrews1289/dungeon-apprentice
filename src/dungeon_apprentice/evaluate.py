"""Frozen, no-update evaluation on held-out procedural seeds."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dungeon_apprentice.artifacts import utc_now
from dungeon_apprentice.contracts import PROTOCOL, DungeonTier, evaluation_seeds
from dungeon_apprentice.env import make_pixel_env


@dataclass(frozen=True)
class EpisodeEvaluation:
    seed: int
    success: bool
    steps: int
    reward: float
    terminal_reason: str | None


@dataclass(frozen=True)
class TierEvaluation:
    protocol: str
    timestamp: str
    tier: int
    tier_label: str
    final_suite: bool
    episodes: int
    successes: int
    success_rate: float
    mean_steps: float
    mean_reward: float
    results: tuple[EpisodeEvaluation, ...]

    def public_dict(self, *, include_episodes: bool = True) -> dict[str, Any]:
        value = asdict(self)
        if not include_episodes:
            value.pop("results")
        return value


def _policy_observation(observation: np.ndarray, model: Any) -> np.ndarray:
    expected = tuple(model.observation_space.shape)
    if tuple(observation.shape) == expected:
        return observation
    channels_first = np.transpose(observation, (2, 0, 1))
    if tuple(channels_first.shape) == expected:
        return channels_first
    raise ValueError(f"model expects {expected}, but the game produced {observation.shape}")


def evaluate_policy(
    model: Any,
    tier: DungeonTier | int,
    seeds: Iterable[int],
    *,
    size: int = 9,
    final_suite: bool = False,
    frame_path: Path | None = None,
) -> TierEvaluation:
    """Measure one frozen policy without updating it or exposing privileged state."""

    selected_tier = DungeonTier(tier)
    results: list[EpisodeEvaluation] = []
    latest_frame: np.ndarray | None = None
    for seed in seeds:
        env = make_pixel_env(tier=selected_tier, size=size)
        try:
            observation, _ = env.reset(seed=int(seed))
            recurrent_state = None
            episode_start = np.ones((1,), dtype=bool)
            total_reward = 0.0
            steps = 0
            info: dict[str, Any] = {}
            while True:
                policy_input = _policy_observation(observation, model)
                action, recurrent_state = model.predict(
                    policy_input,
                    state=recurrent_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                episode_start[:] = False
                observation, reward, terminated, truncated, info = env.step(
                    int(np.asarray(action).item())
                )
                total_reward += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            latest_frame = observation
            results.append(
                EpisodeEvaluation(
                    seed=int(seed),
                    success=bool(info.get("success", False)),
                    steps=steps,
                    reward=total_reward,
                    terminal_reason=info.get("terminal_reason"),
                )
            )
        finally:
            env.close()

    if not results:
        raise ValueError("evaluation requires at least one seed")
    if frame_path is not None and latest_frame is not None:
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(latest_frame).save(frame_path)
    successes = sum(result.success for result in results)
    return TierEvaluation(
        protocol=PROTOCOL,
        timestamp=utc_now(),
        tier=int(selected_tier),
        tier_label=selected_tier.label,
        final_suite=bool(final_suite),
        episodes=len(results),
        successes=successes,
        success_rate=successes / len(results),
        mean_steps=sum(result.steps for result in results) / len(results),
        mean_reward=sum(result.reward for result in results) / len(results),
        results=tuple(results),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tier", choices=("all", "0", "1", "2"), default="all")
    parser.add_argument("--seeds", type=int, default=100, help="held-out levels per tier")
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument(
        "--final",
        action="store_true",
        help="consume the untouched final suite; do this only after model selection",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError as error:
        raise SystemExit('Install training dependencies with: pip install -e ".[train]"') from error

    model = RecurrentPPO.load(args.checkpoint, device="auto")
    tiers = list(DungeonTier) if args.tier == "all" else [DungeonTier(int(args.tier))]
    report = {
        "protocol": PROTOCOL,
        "checkpoint": str(args.checkpoint.resolve()),
        "final_suite": args.final,
        "tiers": [
            evaluate_policy(
                model,
                tier,
                evaluation_seeds(tier, args.seeds, final=args.final),
                size=args.size,
                final_suite=args.final,
            ).public_dict()
            for tier in tiers
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
