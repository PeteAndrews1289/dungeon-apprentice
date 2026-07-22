"""Frozen, no-update evaluation on held-out procedural seeds."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dungeon_apprentice.artifacts import atomic_write_text, file_sha256, utc_now
from dungeon_apprentice.contracts import (
    CHECKPOINT_SCHEMA_VERSION,
    PROTOCOL,
    DungeonTier,
    evaluation_seeds,
)
from dungeon_apprentice.env import make_pixel_env


@dataclass(frozen=True)
class EpisodeEvaluation:
    seed: int
    success: bool
    steps: int
    reward: float
    terminal_reason: str | None
    milestones: dict[str, bool] = field(default_factory=dict)
    unique_cells: int = 0
    collisions: int = 0
    ineffective_interactions: int = 0


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
    milestone_rates: dict[str, float] = field(default_factory=dict)
    mean_unique_cells: float = 0.0
    mean_collisions: float = 0.0
    mean_ineffective_interactions: float = 0.0

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


def _atomic_frame(path: Path, observation: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.png")
    try:
        Image.fromarray(observation).save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
                    milestones={
                        str(name): bool(reached)
                        for name, reached in info.get("milestones", {}).items()
                    },
                    unique_cells=int(info.get("unique_cells", 0)),
                    collisions=int(info.get("collisions", 0)),
                    ineffective_interactions=int(
                        info.get("ineffective_interactions", 0)
                    ),
                )
            )
        finally:
            env.close()

    if not results:
        raise ValueError("evaluation requires at least one seed")
    if frame_path is not None and latest_frame is not None:
        _atomic_frame(frame_path, latest_frame)
    successes = sum(result.success for result in results)
    milestone_names = sorted(
        {name for result in results for name in result.milestones}
    )
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
        milestone_rates={
            name: sum(result.milestones.get(name, False) for result in results)
            / len(results)
            for name in milestone_names
        },
        mean_unique_cells=sum(result.unique_cells for result in results) / len(results),
        mean_collisions=sum(result.collisions for result in results) / len(results),
        mean_ineffective_interactions=(
            sum(result.ineffective_interactions for result in results) / len(results)
        ),
    )


def _load_checkpoint_bundle(checkpoint: Path) -> tuple[Path, dict[str, Any], str]:
    """Verify the model archive/sidecar pair before assigning it evaluation results."""

    archive = checkpoint.expanduser().resolve()
    if archive.suffix != ".zip":
        archive = archive.with_suffix(".zip")
    sidecar_path = archive.with_suffix(".json")
    if not archive.is_file():
        raise SystemExit(f"checkpoint does not exist: {archive}")
    if not sidecar_path.is_file():
        raise SystemExit(f"evaluation requires the checkpoint sidecar: {sidecar_path}")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read checkpoint sidecar {sidecar_path}: {error}") from error
    if sidecar.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise SystemExit(
            "checkpoint schema mismatch: "
            f"expected {CHECKPOINT_SCHEMA_VERSION}, "
            f"found {sidecar.get('schema_version')!r}"
        )
    if sidecar.get("protocol") != PROTOCOL:
        raise SystemExit(
            f"checkpoint protocol mismatch: expected {PROTOCOL!r}, "
            f"found {sidecar.get('protocol')!r}"
        )
    digest = file_sha256(archive)
    if sidecar.get("checkpoint_sha256") != digest:
        raise SystemExit(
            "checkpoint hash does not match its sidecar: "
            f"archive={digest}, sidecar={sidecar.get('checkpoint_sha256')!r}"
        )
    progress = sidecar.get("progress", {})
    collected = int(progress.get("collected_timesteps", -1))
    trained = int(progress.get("trained_timesteps", -2))
    if collected < 0 or collected != trained:
        raise SystemExit(
            "evaluation requires a fully trained checkpoint boundary: "
            f"collected={collected}, trained={trained}"
        )
    return archive, sidecar, digest


def _evaluation_size(args: argparse.Namespace, sidecar: dict[str, Any]) -> int:
    try:
        training_size = int(sidecar["effective_config"]["environment"]["size"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit("checkpoint sidecar does not contain a valid training size") from error
    requested_size = training_size if args.size is None else int(args.size)
    if requested_size != training_size and not args.allow_size_override:
        raise SystemExit(
            f"checkpoint was trained at size {training_size}, but evaluation requested "
            f"size {requested_size}; add --allow-size-override for a declared transfer test"
        )
    return requested_size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tier", choices=("all", "0", "1", "2"), default="all")
    parser.add_argument("--seeds", type=int, default=100, help="held-out levels per tier")
    parser.add_argument(
        "--size",
        type=int,
        help="dungeon size; defaults to the size recorded in the checkpoint sidecar",
    )
    parser.add_argument(
        "--allow-size-override",
        action="store_true",
        help="declare an intentional cross-size transfer evaluation",
    )
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

    checkpoint, sidecar, checkpoint_digest = _load_checkpoint_bundle(args.checkpoint)
    size = _evaluation_size(args, sidecar)
    training_size = int(sidecar["effective_config"]["environment"]["size"])
    model = RecurrentPPO.load(checkpoint, device="auto")
    tiers = list(DungeonTier) if args.tier == "all" else [DungeonTier(int(args.tier))]
    report = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_progress": sidecar.get("progress"),
        "checkpoint_curriculum": sidecar.get("curriculum"),
        "training_size": training_size,
        "evaluation_size": size,
        "size_override": size != training_size,
        "final_suite": args.final,
        "tiers": [
            evaluate_policy(
                model,
                tier,
                evaluation_seeds(tier, args.seeds, final=args.final),
                size=size,
                final_suite=args.final,
            ).public_dict()
            for tier in tiers
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        atomic_write_text(args.output, rendered)
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
