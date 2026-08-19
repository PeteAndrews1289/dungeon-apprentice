"""The Stable-Baselines3 callback factory that drives a training run."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from dungeon_apprentice.artifacts import (
    append_jsonl,
    atomic_copy_file,
    atomic_model_save,
    atomic_write_json,
    ensure_disk_space,
    file_sha256,
    utc_now,
)
from dungeon_apprentice.contracts import (
    CHECKPOINT_SCHEMA_VERSION,
    PROMOTION_THRESHOLD,
    PROTOCOL,
    RETENTION_THRESHOLD,
    DungeonTier,
    evaluation_seeds,
)
from dungeon_apprentice.env import CurriculumState, make_curriculum_pixel_env
from dungeon_apprentice.evaluate import TierEvaluation, evaluate_policy
from dungeon_apprentice.training.config import (
    _CurriculumMastered,
    _json_scalar,
    _safe_frame,
    _sidecar_path,
)


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
