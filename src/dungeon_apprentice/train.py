"""Unattended recurrent-PPO curriculum runner."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dungeon_apprentice.artifacts import (
    append_jsonl,
    atomic_copy_file,
    atomic_model_save,
    atomic_write_json,
    create_run_directory,
    ensure_disk_space,
    file_sha256,
    git_snapshot,
    runtime_snapshot,
    utc_now,
)
from dungeon_apprentice.contracts import (
    CHECKPOINT_SCHEMA_VERSION,
    CURIOSITY_BUDGET,
    DEFAULT_CURIOSITY_SCALE,
    EVALUATION_TIER_STRIDE,
    MINIMUM_GAMMA,
    NEWEST_TIER_FRACTION,
    PROMOTION_THRESHOLD,
    PROTOCOL,
    RETENTION_THRESHOLD,
    DungeonTier,
    evaluation_seeds,
)
from dungeon_apprentice.dashboard import start_dashboard
from dungeon_apprentice.env import CurriculumState, make_curriculum_pixel_env
from dungeon_apprentice.evaluate import TierEvaluation, evaluate_policy
from dungeon_apprentice.qualify import qualify

POLICY_KWARGS = {"lstm_hidden_size": 256, "n_lstm_layers": 1}


class _CurriculumMastered(RuntimeError):
    """Stop between rollouts, before another untrained action is collected."""


def _require_training_packages() -> tuple[Any, Any, Any, Any]:
    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    except ImportError as error:
        raise SystemExit('Install training dependencies with: pip install -e ".[train]"') from error
    return RecurrentPPO, BaseCallback, DummyVecEnv, VecTransposeImage


def _safe_frame(path: Path, observation: np.ndarray) -> None:
    frame = observation
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    frame = np.asarray(frame, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    Image.fromarray(frame).save(temporary, format="PNG")
    os.replace(temporary, path)


def _checkpoint_path(path: Path) -> Path:
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _sidecar_path(checkpoint: Path) -> Path:
    return _checkpoint_path(checkpoint).with_suffix(".json")


def _requested_effective_config(args: argparse.Namespace) -> dict[str, Any]:
    """The exact policy/environment configuration that must survive a resume."""

    return {
        "algorithm": "RecurrentPPO",
        "policy_class": "RecurrentActorCriticCnnPolicy",
        "environment": {
            "workers": int(args.workers),
            "size": int(args.size),
            "curiosity_scale": float(args.curiosity_scale),
            "curiosity_budget": float(CURIOSITY_BUDGET),
        },
        "optimization": {
            "rollout_steps": int(args.rollout_steps),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "n_epochs": int(args.n_epochs),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "ent_coef": 0.01,
        },
        "policy_kwargs": dict(POLICY_KWARGS),
        "curriculum": {
            "newest_tier_fraction": float(NEWEST_TIER_FRACTION),
            "stop_when_mastered": not bool(args.continue_after_mastery),
        },
        "evaluation": {
            "interval": int(args.evaluation_every),
            "seeds_per_tier": int(args.evaluation_seeds),
            "promotion_threshold": float(PROMOTION_THRESHOLD),
            "retention_threshold": float(RETENTION_THRESHOLD),
            "deterministic": True,
        },
        "seed": int(args.seed),
    }


def _actual_effective_config(model: Any, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "algorithm": type(model).__name__,
        "policy_class": model.policy_class.__name__,
        "environment": {
            "workers": int(model.n_envs),
            "size": int(args.size),
            "curiosity_scale": float(args.curiosity_scale),
            "curiosity_budget": float(CURIOSITY_BUDGET),
        },
        "optimization": {
            "rollout_steps": int(model.n_steps),
            "batch_size": int(model.batch_size),
            "learning_rate": float(model.learning_rate),
            "n_epochs": int(model.n_epochs),
            "gamma": float(model.gamma),
            "gae_lambda": float(model.gae_lambda),
            "ent_coef": float(model.ent_coef),
        },
        "policy_kwargs": {
            key: model.policy_kwargs[key] for key in POLICY_KWARGS
        },
        "curriculum": {
            "newest_tier_fraction": float(NEWEST_TIER_FRACTION),
            "stop_when_mastered": not bool(args.continue_after_mastery),
        },
        "evaluation": {
            "interval": int(args.evaluation_every),
            "seeds_per_tier": int(args.evaluation_seeds),
            "promotion_threshold": float(PROMOTION_THRESHOLD),
            "retention_threshold": float(RETENTION_THRESHOLD),
            "deterministic": True,
        },
        "seed": int(model.seed),
    }


def _config_differences(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[str] = []
        for key in sorted(expected.keys() | actual.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in expected:
                differences.append(f"{path}: unexpected value {actual[key]!r}")
            elif key not in actual:
                differences.append(f"{path}: missing (expected {expected[key]!r})")
            else:
                differences.extend(_config_differences(expected[key], actual[key], path))
        return differences
    if expected != actual:
        return [f"{prefix}: expected {expected!r}, found {actual!r}"]
    return []


def _load_resume_metadata(
    requested_checkpoint: Path, expected_config: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    checkpoint = _checkpoint_path(requested_checkpoint.expanduser().resolve())
    sidecar_path = _sidecar_path(checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"resume checkpoint does not exist: {checkpoint}")
    if not sidecar_path.is_file():
        raise SystemExit(
            f"resume requires the v0.1 checkpoint sidecar: {sidecar_path}"
        )
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read resume sidecar {sidecar_path}: {error}") from error

    if sidecar.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise SystemExit(
            "resume checkpoint schema is incompatible: "
            f"expected {CHECKPOINT_SCHEMA_VERSION}, found {sidecar.get('schema_version')!r}"
        )
    if sidecar.get("protocol") != PROTOCOL:
        raise SystemExit(
            f"resume protocol mismatch: expected {PROTOCOL!r}, "
            f"found {sidecar.get('protocol')!r}"
        )
    recorded_hash = sidecar.get("checkpoint_sha256")
    actual_hash = file_sha256(checkpoint)
    if recorded_hash != actual_hash:
        raise SystemExit(
            f"resume checkpoint hash mismatch: sidecar has {recorded_hash!r}, "
            f"archive is {actual_hash}"
        )
    differences = _config_differences(expected_config, sidecar.get("effective_config"))
    if differences:
        rendered = "\n  - ".join(differences)
        raise SystemExit(f"resume configuration is incompatible:\n  - {rendered}")

    progress = sidecar.get("progress", {})
    collected = int(progress.get("collected_timesteps", -1))
    trained = int(progress.get("trained_timesteps", -2))
    updates = int(progress.get("n_updates", -1))
    if collected < 0 or trained < 0 or updates < 0 or collected != trained:
        raise SystemExit(
            "resume checkpoint is not a fully trained boundary: "
            f"collected={collected}, trained={trained}, n_updates={updates}"
        )
    curriculum = sidecar.get("curriculum", {})
    try:
        tier = DungeonTier(int(curriculum.get("max_tier", -1)))
    except (TypeError, ValueError) as error:
        raise SystemExit("resume checkpoint has an invalid curriculum tier") from error
    mastered = curriculum.get("mastered", False)
    if not isinstance(mastered, bool):
        raise SystemExit("resume checkpoint has an invalid mastered flag")
    if mastered and tier is not DungeonTier.RETRIEVE:
        raise SystemExit("resume checkpoint claims mastery before the final tier")
    return checkpoint, sidecar


def _json_scalar(value: Any) -> int | float | str | bool | None:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    return None


class _CallbackFactory:
    """Delay the SB3 base-class import until training is actually requested."""

    @staticmethod
    def create(base_callback: Any) -> type[Any]:
        class CurriculumCallback(base_callback):
            def __init__(
                self,
                *,
                run_directory: Path,
                curriculum: CurriculumState,
                size: int,
                evaluation_interval: int,
                evaluation_seed_count: int,
                checkpoint_interval: int,
                frame_interval: int,
                stop_when_mastered: bool,
                started_at: str,
                effective_config: dict[str, Any],
                segment: dict[str, Any],
                initial_trained_timesteps: int,
                initial_mastered: bool,
                keep_checkpoints: int,
                minimum_free_bytes: int,
            ) -> None:
                super().__init__(verbose=0)
                self.run_directory = run_directory
                self.curriculum = curriculum
                self.size = size
                self.evaluation_interval = evaluation_interval
                self.evaluation_seed_count = evaluation_seed_count
                self.checkpoint_interval = checkpoint_interval
                self.frame_interval = frame_interval
                self.stop_when_mastered = stop_when_mastered
                self.started_at = started_at
                self.effective_config = effective_config
                self.segment = segment
                self.keep_checkpoints = keep_checkpoints
                self.minimum_free_bytes = minimum_free_bytes
                self.wall_start = time.monotonic()
                self.segment_start_collected = int(
                    segment.get("start_collected_timesteps", 0)
                )
                self.segment_start_trained = int(initial_trained_timesteps)
                self.trained_timesteps = int(initial_trained_timesteps)
                self.last_processed_updates = 0
                self.next_evaluation = evaluation_interval
                self.next_checkpoint = checkpoint_interval
                self.next_status = 0
                self.next_frame = 0
                self.episode_count = 0
                self.recent_outcomes: dict[int, deque[bool]] = defaultdict(
                    lambda: deque(maxlen=100)
                )
                self.latest_evaluations: list[dict[str, Any]] = []
                self.evaluation_history: deque[dict[str, Any]] = deque(maxlen=100)
                self.frame_revision = 0
                self.mastered = bool(initial_mastered)
                self._last_curriculum_transition_at: int | None = None
                self._episode_totals: list[dict[str, Any]] = []
                self._last_saved_trained_timesteps: int | None = None
                self._last_saved_path: Path | None = None
                self.latest_safe_checkpoint: Path | None = None

            def _on_training_start(self) -> None:
                current = int(self.model.num_timesteps)
                if current != self.segment_start_collected:
                    raise RuntimeError(
                        "model/segment start mismatch: "
                        f"model={current}, segment={self.segment_start_collected}"
                    )
                if current != self.trained_timesteps:
                    raise RuntimeError(
                        "a segment may start only from a fully trained checkpoint"
                    )
                self.last_processed_updates = int(self.model._n_updates)
                self.next_evaluation = self._next_boundary(
                    self.trained_timesteps, self.evaluation_interval
                )
                self.next_checkpoint = self._next_boundary(
                    self.trained_timesteps, self.checkpoint_interval
                )
                self.next_status = current
                self.next_frame = current
                self._episode_totals = [self._empty_episode() for _ in range(self.model.n_envs)]
                self._checkpoint("initial", kind="initial")
                self._write_status("training")

            @staticmethod
            def _empty_episode() -> dict[str, Any]:
                return {
                    "combined_return": 0.0,
                    "extrinsic_return": 0.0,
                    "curiosity_return": 0.0,
                    "length": 0,
                    "seed": None,
                    "layout_sha256": None,
                    "tier": None,
                    "tier_label": None,
                }

            @staticmethod
            def _next_boundary(current: int, interval: int) -> int:
                return ((current // interval) + 1) * interval

            def _on_rollout_start(self) -> None:
                # SB3 calls this after the previous rollout's optimizer pass and before
                # collecting another action. This is the first safe post-update hook.
                if int(self.model._n_updates) > self.last_processed_updates:
                    self._process_trained_boundary()
                if self.mastered and self.stop_when_mastered:
                    raise _CurriculumMastered

            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                dones = self.locals.get("dones", [])
                rewards = self.locals.get("rewards", [])
                for worker, (done, info) in enumerate(zip(dones, infos, strict=False)):
                    if worker >= len(self._episode_totals):
                        continue
                    reward = float(rewards[worker]) if worker < len(rewards) else 0.0
                    episode = self._episode_totals[worker]
                    episode["combined_return"] += reward
                    episode["extrinsic_return"] += float(
                        info.get("extrinsic_reward", reward)
                    )
                    episode["curiosity_return"] += float(info.get("curiosity_reward", 0.0))
                    episode["length"] += 1
                    for key in ("seed", "layout_sha256", "tier", "tier_label"):
                        if info.get(key) is not None:
                            episode[key] = info[key]
                    if bool(done):
                        self._record_episode(worker, episode, info)
                        self._episode_totals[worker] = self._empty_episode()

                current = int(self.num_timesteps)
                if current >= self.next_frame:
                    observations = self.locals.get("new_obs")
                    if observations is not None and len(observations):
                        _safe_frame(
                            self.run_directory / "frames" / "latest.png",
                            observations[0],
                        )
                        self.frame_revision += 1
                    self.next_frame = current + self.frame_interval

                if current >= self.next_status:
                    self._write_status("training")
                    self.next_status = current + 1_000
                return True

            def _record_episode(
                self, worker: int, episode: dict[str, Any], info: dict[str, Any]
            ) -> None:
                tier = int(info.get("tier", episode.get("tier") or 0))
                success = bool(info.get("success", False))
                self.recent_outcomes[tier].append(success)
                self.episode_count += 1
                append_jsonl(
                    self.run_directory / "episodes.jsonl",
                    {
                        "timestamp": utc_now(),
                        "segment_id": self.segment["id"],
                        "episode": self.episode_count,
                        "worker": worker,
                        "collected_timesteps": int(self.model.num_timesteps),
                        "tier": tier,
                        "tier_label": info.get("tier_label", episode.get("tier_label")),
                        "seed": info.get("seed", episode.get("seed")),
                        "layout_sha256": info.get(
                            "layout_sha256", episode.get("layout_sha256")
                        ),
                        "success": success,
                        "terminal_reason": info.get("terminal_reason"),
                        "time_limit_truncated": bool(info.get("TimeLimit.truncated", False)),
                        "length": int(episode["length"]),
                        "combined_return": float(episode["combined_return"]),
                        "extrinsic_return": float(episode["extrinsic_return"]),
                        "curiosity_return": float(episode["curiosity_return"]),
                        "curiosity_unique_observations": info.get(
                            "curiosity_unique_observations"
                        ),
                        "curiosity_budget_used": info.get("curiosity_budget_used"),
                        "milestones": info.get("milestones"),
                        "unique_cells": info.get("unique_cells"),
                        "collisions": info.get("collisions"),
                        "ineffective_interactions": info.get(
                            "ineffective_interactions"
                        ),
                    },
                )

            def _process_trained_boundary(self) -> None:
                updates = int(self.model._n_updates)
                if updates <= self.last_processed_updates:
                    return
                self.trained_timesteps = int(self.model.num_timesteps)
                updates_since_boundary = updates - self.last_processed_updates
                self.last_processed_updates = updates
                self._record_optimizer_metrics(updates_since_boundary)

                if self.trained_timesteps >= self.next_checkpoint:
                    self._checkpoint(
                        f"step-{self.trained_timesteps:012d}", kind="step"
                    )
                    self.next_checkpoint = self._next_boundary(
                        self.trained_timesteps, self.checkpoint_interval
                    )
                if self.trained_timesteps >= self.next_evaluation:
                    self.next_evaluation = self._next_boundary(
                        self.trained_timesteps, self.evaluation_interval
                    )
                    self._evaluate_and_maybe_promote(trigger="scheduled")
                self._write_status("mastered" if self.mastered else "training")

            def _record_optimizer_metrics(self, updates_since_boundary: int) -> None:
                logger_values = getattr(self.model.logger, "name_to_value", {})
                metrics = {
                    key: scalar
                    for key, value in logger_values.items()
                    if key.startswith("train/") and (scalar := _json_scalar(value)) is not None
                }
                append_jsonl(
                    self.run_directory / "optimizer.jsonl",
                    {
                        "timestamp": utc_now(),
                        "segment_id": self.segment["id"],
                        "collected_timesteps": int(self.model.num_timesteps),
                        "trained_timesteps": self.trained_timesteps,
                        "segment_trained_timesteps": (
                            self.trained_timesteps - self.segment_start_trained
                        ),
                        "n_updates": int(self.model._n_updates),
                        "updates_since_boundary": updates_since_boundary,
                        "metrics": metrics,
                    },
                )

            def _evaluate_and_maybe_promote(self, *, trigger: str) -> None:
                checkpoint = self._evaluation_checkpoint()
                checkpoint_digest = file_sha256(checkpoint)
                checkpoint_reference = str(checkpoint.relative_to(self.run_directory))
                current_tier = self.curriculum.max_tier
                evaluations: list[TierEvaluation] = []
                for tier_value in range(int(current_tier) + 1):
                    tier = DungeonTier(tier_value)
                    result = evaluate_policy(
                        self.model,
                        tier,
                        evaluation_seeds(tier, self.evaluation_seed_count),
                        size=self.size,
                        frame_path=self.run_directory / "frames" / "latest.png",
                    )
                    evaluations.append(result)
                    append_jsonl(
                        self.run_directory / "evaluations.jsonl",
                        {
                            "segment_id": self.segment["id"],
                            "collected_timesteps": int(self.model.num_timesteps),
                            "trained_timesteps": self.trained_timesteps,
                            "n_updates": int(self.model._n_updates),
                            "trigger": trigger,
                            "checkpoint": checkpoint_reference,
                            "checkpoint_sha256": checkpoint_digest,
                            **result.public_dict(),
                        },
                    )

                current_result = evaluations[-1]
                retained = all(
                    result.success_rate >= RETENTION_THRESHOLD
                    for result in evaluations[:-1]
                )
                passed = current_result.success_rate >= PROMOTION_THRESHOLD and retained
                transition_available = (
                    self._last_curriculum_transition_at != self.trained_timesteps
                )
                previous_mastered = self.mastered
                previous_transition = self._last_curriculum_transition_at
                if passed and not transition_available:
                    decision = "Held: one curriculum transition per trained boundary"
                elif passed and current_tier < DungeonTier.RETRIEVE:
                    promoted = DungeonTier(int(current_tier) + 1)
                    decision = f"Promoted to {promoted.label}"
                    self.curriculum.max_tier = promoted
                    self._last_curriculum_transition_at = self.trained_timesteps
                    try:
                        self._checkpoint(
                            f"promotion-tier-{int(promoted)}", kind="promotion"
                        )
                    except BaseException:
                        self.curriculum.max_tier = current_tier
                        self._last_curriculum_transition_at = previous_transition
                        raise
                elif passed:
                    decision = "Version 0 mastered"
                    self.mastered = True
                    self._last_curriculum_transition_at = self.trained_timesteps
                    try:
                        self._checkpoint("mastered-v0", kind="mastery")
                    except BaseException:
                        self.mastered = previous_mastered
                        self._last_curriculum_transition_at = previous_transition
                        raise
                elif not retained:
                    decision = "Held: earlier skill retention below 80%"
                else:
                    decision = "Held: current lesson below 90%"
                if current_tier is DungeonTier.RETRIEVE and not passed:
                    self.mastered = False

                self.latest_evaluations = [
                    result.public_dict(include_episodes=False) for result in evaluations
                ]
                for result in evaluations:
                    self.evaluation_history.append(
                        {
                            "trained_timesteps": self.trained_timesteps,
                            "tier": result.tier,
                            "tier_label": result.tier_label,
                            "success_rate": result.success_rate,
                            "trigger": trigger,
                            "checkpoint": checkpoint_reference,
                            "checkpoint_sha256": checkpoint_digest,
                            "decision": (
                                decision
                                if result.tier == int(current_tier)
                                else "Retention exam"
                            ),
                        }
                    )
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": "curriculum_decision",
                        "collected_timesteps": int(self.model.num_timesteps),
                        "trained_timesteps": self.trained_timesteps,
                        "n_updates": int(self.model._n_updates),
                        "trigger": trigger,
                        "transition_available": transition_available,
                        "checkpoint": checkpoint_reference,
                        "checkpoint_sha256": checkpoint_digest,
                        "decision": decision,
                        "evaluated_tier": int(current_tier),
                        "next_tier": int(self.curriculum.max_tier),
                    },
                )
                self.frame_revision += 1

            def _evaluation_checkpoint(self) -> Path:
                """Return a retained bundle containing the exact policy an exam grades."""

                if (
                    self._last_saved_trained_timesteps == self.trained_timesteps
                    and self._last_saved_path is not None
                    and self._last_saved_path.is_file()
                ):
                    return self._last_saved_path
                return self._checkpoint(
                    f"step-{self.trained_timesteps:012d}", kind="exam"
                )

            def _checkpoint(self, name: str, *, kind: str) -> Path:
                collected = int(self.model.num_timesteps)
                if collected != self.trained_timesteps:
                    raise RuntimeError(
                        f"refusing unsafe checkpoint: collected={collected}, "
                        f"trained={self.trained_timesteps}"
                    )
                ensure_disk_space(self.run_directory, self.minimum_free_bytes)
                destination = self.run_directory / "checkpoints" / f"{name}.zip"
                if (
                    self._last_saved_trained_timesteps == self.trained_timesteps
                    and self._last_saved_path is not None
                    and self._last_saved_path.is_file()
                ):
                    atomic_copy_file(self._last_saved_path, destination)
                else:
                    atomic_model_save(self.model, destination)

                digest = file_sha256(destination)
                sidecar = self._checkpoint_sidecar(
                    name=name,
                    kind=kind,
                    checkpoint_sha256=digest,
                )
                atomic_write_json(_sidecar_path(destination), sidecar)

                latest = destination.parent / "latest.zip"
                if destination != latest:
                    atomic_copy_file(destination, latest)
                    latest_sidecar = dict(sidecar)
                    latest_sidecar["checkpoint_name"] = "latest"
                    latest_sidecar["kind"] = "latest"
                    latest_sidecar["alias_of"] = destination.name
                    atomic_write_json(_sidecar_path(latest), latest_sidecar)
                self.latest_safe_checkpoint = destination
                self._last_saved_trained_timesteps = self.trained_timesteps
                self._last_saved_path = destination
                if kind in {"step", "exam"}:
                    self._prune_numbered_checkpoints()
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": "checkpoint",
                        "kind": kind,
                        "collected_timesteps": collected,
                        "trained_timesteps": self.trained_timesteps,
                        "n_updates": int(self.model._n_updates),
                        "sha256": digest,
                        "path": str(destination.relative_to(self.run_directory)),
                    },
                )
                return destination

            def _checkpoint_sidecar(
                self, *, name: str, kind: str, checkpoint_sha256: str
            ) -> dict[str, Any]:
                return {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "created_at": utc_now(),
                    "checkpoint_name": name,
                    "kind": kind,
                    "checkpoint_sha256": checkpoint_sha256,
                    "effective_config": self.effective_config,
                    "curriculum": {
                        "max_tier": int(self.curriculum.max_tier),
                        "max_tier_label": self.curriculum.max_tier.label,
                        "mastered": self.mastered,
                    },
                    "progress": {
                        "collected_timesteps": int(self.model.num_timesteps),
                        "trained_timesteps": self.trained_timesteps,
                        "segment_collected_timesteps": (
                            int(self.model.num_timesteps) - self.segment_start_collected
                        ),
                        "segment_trained_timesteps": (
                            self.trained_timesteps - self.segment_start_trained
                        ),
                        "n_updates": int(self.model._n_updates),
                    },
                    "segment": self.segment,
                }

            def _prune_numbered_checkpoints(self) -> None:
                checkpoints = self.run_directory / "checkpoints"
                numbered = sorted(checkpoints.glob("step-*.zip"))
                for checkpoint in numbered[: -self.keep_checkpoints]:
                    checkpoint.unlink(missing_ok=True)
                    _sidecar_path(checkpoint).unlink(missing_ok=True)

            def _write_status(self, phase: str) -> None:
                elapsed = time.monotonic() - self.wall_start
                collected = int(self.model.num_timesteps)
                segment_collected = collected - self.segment_start_collected
                recent = {
                    DungeonTier(tier).label: {
                        "episodes": len(values),
                        "success_rate": sum(values) / len(values) if values else None,
                    }
                    for tier, values in self.recent_outcomes.items()
                }
                atomic_write_json(
                    self.run_directory / "status.json",
                    {
                        "protocol": PROTOCOL,
                        "phase": phase,
                        "started_at": self.started_at,
                        "updated_at": utc_now(),
                        "elapsed_seconds": elapsed,
                        "total_timesteps": collected,
                        "collected_timesteps": collected,
                        "trained_timesteps": self.trained_timesteps,
                        "segment_collected_timesteps": segment_collected,
                        "segment_trained_timesteps": (
                            self.trained_timesteps - self.segment_start_trained
                        ),
                        "discarded_partial_timesteps": max(
                            0, collected - self.trained_timesteps
                        ),
                        "n_updates": int(self.model._n_updates),
                        "fps": segment_collected / elapsed if elapsed else 0.0,
                        "episodes": self.episode_count,
                        "current_tier": int(self.curriculum.max_tier),
                        "current_tier_label": self.curriculum.max_tier.label,
                        "recent_training": recent,
                        "next_evaluation": self.next_evaluation,
                        "next_checkpoint": self.next_checkpoint,
                        "latest_evaluations": self.latest_evaluations,
                        "evaluation_history": list(self.evaluation_history),
                        "frame_revision": self.frame_revision,
                        "segment": self.segment,
                        "latest_safe_checkpoint": (
                            str(self.latest_safe_checkpoint.relative_to(self.run_directory))
                            if self.latest_safe_checkpoint is not None
                            else None
                        ),
                    },
                )

            def finalize_success(self, phase: str) -> None:
                if int(self.model._n_updates) > self.last_processed_updates:
                    self._process_trained_boundary()
                collected = int(self.model.num_timesteps)
                if collected != self.trained_timesteps:
                    raise RuntimeError(
                        f"successful run ended with an untrained partial rollout: "
                        f"collected={collected}, trained={self.trained_timesteps}"
                    )
                # Every completed policy receives an explicit frozen exam, even when
                # the same boundary happened to trigger a scheduled exam.
                self._evaluate_and_maybe_promote(trigger="final")
                final_phase = "mastered" if self.mastered else phase
                self._checkpoint(final_phase, kind="final")
                self._write_status(final_phase)

            def finalize_failure(self, phase: str) -> None:
                collected = int(self.model.num_timesteps)
                discarded = max(0, collected - self.trained_timesteps)
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": f"run_{phase}",
                        "collected_timesteps": collected,
                        "trained_timesteps": self.trained_timesteps,
                        "discarded_partial_timesteps": discarded,
                        "latest_safe_checkpoint": (
                            str(self.latest_safe_checkpoint.relative_to(self.run_directory))
                            if self.latest_safe_checkpoint is not None
                            else None
                        ),
                    },
                )
                self._write_status(phase)

        return CurriculumCallback


def _environment_factory(
    *, seed: int, state: CurriculumState, size: int, curiosity_scale: float
) -> Any:
    def create() -> Any:
        return make_curriculum_pixel_env(
            seed=seed,
            state=state,
            size=size,
            curiosity_scale=curiosity_scale,
        )

    return create


def _activate_segment_seed(model: Any, segment_seed: int) -> None:
    """Start a declared RNG stream for policy sampling and the next vector reset."""

    model.set_random_seed(int(segment_seed))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--curiosity-scale", type=float, default=DEFAULT_CURIOSITY_SCALE)
    parser.add_argument("--evaluation-every", type=int, default=50_000)
    parser.add_argument("--evaluation-seeds", type=int, default=40)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--keep-checkpoints", type=int, default=5)
    parser.add_argument("--minimum-free-gib", type=float, default=2.0)
    parser.add_argument("--frame-every", type=int, default=1_000)
    parser.add_argument("--qualification-seeds", type=int, default=100)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8780)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--continue-after-mastery", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "total timesteps": args.total_timesteps,
        "workers": args.workers,
        "rollout steps": args.rollout_steps,
        "batch size": args.batch_size,
        "epochs": args.n_epochs,
        "evaluation interval": args.evaluation_every,
        "evaluation seeds": args.evaluation_seeds,
        "checkpoint interval": args.checkpoint_every,
        "checkpoint retention": args.keep_checkpoints,
        "frame interval": args.frame_every,
        "qualification seeds": args.qualification_seeds,
    }
    for label, value in positive.items():
        if value <= 0:
            raise SystemExit(f"{label} must be positive")
    rollout_size = args.workers * args.rollout_steps
    if rollout_size % args.batch_size:
        raise SystemExit("workers x rollout-steps must be divisible by batch-size")
    if args.size < 7:
        raise SystemExit("dungeon size must be at least 7")
    if args.evaluation_seeds > EVALUATION_TIER_STRIDE:
        raise SystemExit(
            f"evaluation seeds cannot exceed {EVALUATION_TIER_STRIDE} per tier"
        )
    if not math.isfinite(args.curiosity_scale):
        raise SystemExit("curiosity scale must be finite")
    if args.curiosity_scale != DEFAULT_CURIOSITY_SCALE:
        raise SystemExit(
            f"protocol {PROTOCOL} freezes curiosity scale at "
            f"{DEFAULT_CURIOSITY_SCALE}"
        )
    if not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib < 0.0:
        raise SystemExit("minimum free GiB must be finite and non-negative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise SystemExit("learning rate must be finite and positive")
    if not math.isfinite(args.gamma) or not MINIMUM_GAMMA <= args.gamma <= 1.0:
        raise SystemExit(
            f"gamma must be in [{MINIMUM_GAMMA}, 1] to preserve discounted "
            "reward dominance"
        )
    if not math.isfinite(args.gae_lambda) or not 0.0 < args.gae_lambda <= 1.0:
        raise SystemExit("GAE lambda must be in (0, 1]")


def _best_effort(label: str, operation: Callable[[], Any]) -> None:
    """Run cleanup/reporting without replacing an already-active failure."""

    try:
        operation()
    except BaseException as error:
        print(f"Warning: {label} failed during finalization: {error}", file=sys.stderr)


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    expected_config = _requested_effective_config(args)
    resume_checkpoint: Path | None = None
    resume_sidecar: dict[str, Any] | None = None
    if args.resume:
        resume_checkpoint, resume_sidecar = _load_resume_metadata(
            args.resume, expected_config
        )

    minimum_free_bytes = int(args.minimum_free_gib * (1024**3))
    ensure_disk_space(args.run_root, minimum_free_bytes)
    RecurrentPPO, BaseCallback, DummyVecEnv, VecTransposeImage = _require_training_packages()

    repository = Path(__file__).resolve().parents[2]
    run_directory = create_run_directory(args.run_root, args.run_name)
    started_at = utc_now()
    qualification = qualify(args.qualification_seeds, size=args.size)
    atomic_write_json(run_directory / "qualification.json", qualification)
    if qualification["result"] != "passed":
        raise SystemExit(f"Generator qualification failed; inspect {run_directory}")

    prior_segment = resume_sidecar.get("segment", {}) if resume_sidecar else {}
    segment_index = int(prior_segment.get("index", -1)) + 1
    initial_progress = resume_sidecar.get("progress", {}) if resume_sidecar else {}
    initial_collected = int(initial_progress.get("collected_timesteps", 0))
    initial_trained = int(initial_progress.get("trained_timesteps", 0))
    worker_seed_base = args.seed + segment_index * 10_000
    segment = {
        "id": uuid.uuid4().hex,
        "index": segment_index,
        "started_at": started_at,
        "run_directory": str(run_directory),
        "start_collected_timesteps": initial_collected,
        "start_trained_timesteps": initial_trained,
        "worker_seed_base": worker_seed_base,
        "algorithm_seed": worker_seed_base,
        "parent": (
            {
                "checkpoint": str(resume_checkpoint),
                "checkpoint_sha256": resume_sidecar["checkpoint_sha256"],
                "segment_id": prior_segment.get("id"),
                "collected_timesteps": initial_collected,
                "trained_timesteps": initial_trained,
            }
            if resume_sidecar is not None
            else None
        ),
    }

    curriculum = CurriculumState()
    if resume_sidecar:
        curriculum.max_tier = DungeonTier(
            int(resume_sidecar.get("curriculum", {}).get("max_tier", 0))
        )

    factories = [
        _environment_factory(
            seed=worker_seed_base + worker,
            state=curriculum,
            size=args.size,
            curiosity_scale=args.curiosity_scale,
        )
        for worker in range(args.workers)
    ]
    vector_environment = VecTransposeImage(DummyVecEnv(factories))
    if resume_checkpoint:
        model = RecurrentPPO.load(
            resume_checkpoint,
            env=vector_environment,
            device=args.device,
        )
        reset_num_timesteps = False
        if int(model.num_timesteps) != initial_collected:
            raise SystemExit(
                "resume model/sidecar timestep mismatch: "
                f"model={model.num_timesteps}, sidecar={initial_collected}"
            )
        if int(model._n_updates) != int(initial_progress.get("n_updates", -1)):
            raise SystemExit(
                "resume model/sidecar optimizer update mismatch: "
                f"model={model._n_updates}, sidecar={initial_progress.get('n_updates')!r}"
            )
    else:
        model = RecurrentPPO(
            "CnnLstmPolicy",
            vector_environment,
            learning_rate=args.learning_rate,
            n_steps=args.rollout_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=0.01,
            seed=args.seed,
            device=args.device,
            verbose=1,
            policy_kwargs=POLICY_KWARGS,
        )
        reset_num_timesteps = True

    # Loading an SB3 model re-stages its original seed on the vector environment.
    # Re-seed after both construction paths so child segments do not replay segment 0.
    _activate_segment_seed(model, worker_seed_base)

    effective_config = _actual_effective_config(model, args)
    differences = _config_differences(expected_config, effective_config)
    if differences:
        rendered = "\n  - ".join(differences)
        raise SystemExit(f"effective model configuration mismatch:\n  - {rendered}")

    manifest = {
        "protocol": PROTOCOL,
        "started_at": started_at,
        "arguments": vars(args)
        | {
            "run_root": str(args.run_root),
            "resume": str(resume_checkpoint) if resume_checkpoint else None,
        },
        "effective_config": effective_config,
        "segment": segment,
        "git": git_snapshot(repository),
        "runtime": runtime_snapshot(),
        "information_boundary": "partial RGB pixels plus recurrent state only",
        "online_model_calls": False,
    }
    atomic_write_json(run_directory / "manifest.json", manifest)

    callback_type = _CallbackFactory.create(BaseCallback)
    callback = callback_type(
        run_directory=run_directory,
        curriculum=curriculum,
        size=args.size,
        evaluation_interval=args.evaluation_every,
        evaluation_seed_count=args.evaluation_seeds,
        checkpoint_interval=args.checkpoint_every,
        frame_interval=args.frame_every,
        stop_when_mastered=not args.continue_after_mastery,
        started_at=started_at,
        effective_config=effective_config,
        segment=segment,
        initial_trained_timesteps=initial_trained,
        initial_mastered=bool(
            resume_sidecar.get("curriculum", {}).get("mastered", False)
        )
        if resume_sidecar
        else False,
        keep_checkpoints=args.keep_checkpoints,
        minimum_free_bytes=minimum_free_bytes,
    )

    dashboard = None
    if not args.no_dashboard:
        dashboard_error = None
        try:
            dashboard = start_dashboard(
                run_directory,
                host=args.dashboard_host,
                port=args.dashboard_port,
            )
        except OSError as error:
            dashboard_error = str(error)
            try:
                dashboard = start_dashboard(
                    run_directory, host=args.dashboard_host, port=0
                )
            except OSError as fallback_error:
                dashboard_error = f"{dashboard_error}; fallback failed: {fallback_error}"
        if dashboard is not None:
            print(
                f"Dashboard: http://{args.dashboard_host}:{dashboard.server_port}/",
                flush=True,
            )
        if dashboard_error:
            append_jsonl(
                run_directory / "events.jsonl",
                {
                    "timestamp": utc_now(),
                    "type": "dashboard_warning",
                    "message": dashboard_error,
                },
            )
    print(f"Run artifacts: {run_directory}", flush=True)

    phase = "completed"
    try:
        try:
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=callback,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=False,
            )
        except _CurriculumMastered:
            phase = "mastered"
        except KeyboardInterrupt:
            phase = "interrupted"

        if phase == "interrupted":
            _best_effort(
                "interruption record", lambda: callback.finalize_failure(phase)
            )
        else:
            callback.finalize_success(phase)
    except BaseException:
        phase = "crashed"
        crash = {"timestamp": utc_now(), "traceback": traceback.format_exc()}
        _best_effort(
            "crash report", lambda: atomic_write_json(run_directory / "crash.json", crash)
        )
        _best_effort("crash status", lambda: callback.finalize_failure(phase))
        raise
    finally:
        _best_effort("environment close", vector_environment.close)
        if dashboard is not None:
            _best_effort("dashboard shutdown", dashboard.shutdown)
            _best_effort("dashboard close", dashboard.server_close)


if __name__ == "__main__":
    main()
