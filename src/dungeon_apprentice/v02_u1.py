"""Warm-start U1 development child: retain Navigate/U0 while learning Local Unlock."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
import uuid
import zipfile
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
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
from dungeon_apprentice.contracts import MINIMUM_GAMMA
from dungeon_apprentice.dashboard import start_dashboard
from dungeon_apprentice.v02_confirm import (
    CONFIRMATION_PROTOCOL,
    ConfirmationError,
    verify_checkpoint,
)
from dungeon_apprentice.v02_lessons import (
    ALLOCATION_TOLERANCE,
    EVALUATION_SEED_COUNT,
    GENERATOR_PROFILE_VERSION,
    LESSON_SPECS,
    PROTOCOL,
    QUALIFICATION_SEED_COUNT,
    CurriculumState,
    LessonEvaluation,
    LessonId,
    RecoveryCause,
    TransitionDeficitScheduler,
    evaluate_lesson,
    lesson_passed,
    make_training_env,
    qualify_local_unlock,
    recovery_cause,
    reserved_training_layout_hashes,
    validation_seeds,
)

CHECKPOINT_SCHEMA_VERSION = 3
CHILD_ACTION_BUDGET = 524_288
EVALUATION_INTERVAL = 32_768
EXPECTED_PARENT_TRAINING_SEED = 20260725
EXPECTED_PARENT_SHA256 = (
    "2b235ed54746429e737af2a6037069821fdaf81a936edf8f74ce86cc766b40a2"
)
EXPECTED_PARENT_TRAINED_TIMESTEPS = 491_520
EXPECTED_CONFIRMATION_SHA256 = (
    "f43610252943fce9c0169ac0231724fc2829686d1899f5f88bd0c250a913b398"
)
DEFAULT_PARENT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/sentinels/v0.2-u0-20260722/"
    "v02-u0-seed-20260725/checkpoints/mastered-visible-unlock.zip"
)
DEFAULT_CONFIRMATION = Path(
    "/Volumes/T7 Developer/DungeonApprentice/confirmations/"
    "v0.2-u0-20260722/report.json"
)
POLICY_KWARGS = {"lstm_hidden_size": 256, "n_lstm_layers": 1}


class U1ProtocolError(RuntimeError):
    """Raised when provenance or a resumable U1 artifact violates the protocol."""


@dataclass(frozen=True)
class ParentProvenance:
    checkpoint: str
    checkpoint_sha256: str
    sidecar: str
    manifest: str
    source_commit: str
    training_seed: int
    trained_timesteps: int
    n_updates: int
    confirmation_report: str
    confirmation_sha256: str
    confirmation_protocol: str
    confirmation_verdict: str
    confirmation_completed_at: str

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise U1ProtocolError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise U1ProtocolError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise U1ProtocolError(f"{label} must be a JSON object: {path}")
    return value


def verify_parent(
    checkpoint: Path, confirmation_report: Path
) -> ParentProvenance:
    """Require the exact selected, confirmed U0 development parent."""

    try:
        verified = verify_checkpoint(checkpoint)
    except ConfirmationError as error:
        raise U1ProtocolError(f"U0 parent verification failed: {error}") from error
    if verified.training_seed != EXPECTED_PARENT_TRAINING_SEED:
        raise U1ProtocolError(
            f"U1 lead child requires parent seed {EXPECTED_PARENT_TRAINING_SEED}"
        )
    if verified.checkpoint_sha256 != EXPECTED_PARENT_SHA256:
        raise U1ProtocolError("U1 parent checkpoint does not match its frozen digest")
    if verified.trained_timesteps != EXPECTED_PARENT_TRAINED_TIMESTEPS:
        raise U1ProtocolError("U1 parent has the wrong trained boundary")
    try:
        with zipfile.ZipFile(verified.checkpoint) as archive:
            if "policy.optimizer.pth" not in archive.namelist():
                raise U1ProtocolError("U0 parent archive does not contain optimizer state")
    except zipfile.BadZipFile as error:
        raise U1ProtocolError("U0 parent is not a readable PPO archive") from error

    report_path = confirmation_report.expanduser().resolve()
    digest = file_sha256(report_path)
    if digest != EXPECTED_CONFIRMATION_SHA256:
        raise U1ProtocolError("U0 confirmation report does not match its frozen digest")
    report = _read_json(report_path, "U0 confirmation report")
    if (
        report.get("protocol") != CONFIRMATION_PROTOCOL
        or report.get("verdict") != "passed"
        or report.get("policy_updates") is not False
    ):
        raise U1ProtocolError("U0 confirmation report is not the passed no-update result")
    matching = [
        item
        for item in report.get("checkpoints", [])
        if item.get("checkpoint", {}).get("checkpoint_sha256") == EXPECTED_PARENT_SHA256
    ]
    if len(matching) != 1 or matching[0].get("passed") is not True:
        raise U1ProtocolError("selected parent did not individually pass confirmation")

    sidecar_path = Path(verified.sidecar)
    sidecar = _read_json(sidecar_path, "U0 parent sidecar")
    n_updates = int(sidecar.get("progress", {}).get("n_updates", -1))
    if n_updates != 960:
        raise U1ProtocolError(f"U0 parent has an unexpected optimizer boundary: {n_updates}")
    return ParentProvenance(
        checkpoint=verified.checkpoint,
        checkpoint_sha256=verified.checkpoint_sha256,
        sidecar=verified.sidecar,
        manifest=verified.manifest,
        source_commit=verified.source_commit,
        training_seed=verified.training_seed,
        trained_timesteps=verified.trained_timesteps,
        n_updates=n_updates,
        confirmation_report=str(report_path),
        confirmation_sha256=digest,
        confirmation_protocol=str(report["protocol"]),
        confirmation_verdict=str(report["verdict"]),
        confirmation_completed_at=str(report["completed_at"]),
    )


def _effective_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "warm_start": True,
        "parent_training_seed": EXPECTED_PARENT_TRAINING_SEED,
        "child_algorithm_seed": args.seed,
        "child_action_budget": args.child_budget,
        "environment": {
            "size": args.size,
            "workers": args.workers,
            "generator_profile_version": GENERATOR_PROFILE_VERSION,
        },
        "optimization": {
            "rollout_steps": args.rollout_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "ent_coef": 0.01,
            "policy_kwargs": POLICY_KWARGS,
        },
        "lessons": [lesson.value for lesson in LessonId],
        "normal_transition_targets": {
            lesson.value: share for lesson, share in CurriculumState().targets().items()
        },
        "recovery_transition_targets": {
            cause.value: {
                lesson.value: share
                for lesson, share in CurriculumState(recovery_cause=cause).targets().items()
            }
            for cause in RecoveryCause
        },
        "evaluation": {
            "interval": args.evaluation_every,
            "seeds_per_lesson": args.evaluation_seeds,
            "panels": [40, 40],
            "consecutive_mastery_passes": 2,
            "consecutive_recovery_passes": 2,
            "allocation_tolerance": ALLOCATION_TOLERANCE,
            "thresholds": {
                lesson.value: {
                    "overall": LESSON_SPECS[lesson].overall_threshold,
                    "panel": LESSON_SPECS[lesson].panel_threshold,
                }
                for lesson in LessonId
            },
        },
    }


@dataclass(frozen=True)
class ResumeBundle:
    checkpoint: Path
    sidecar: dict[str, Any]
    curriculum: CurriculumState
    scheduler_state: dict[str, Any]
    lifetime_trained: int
    child_trained: int
    n_updates: int
    segment_index: int
    last_completed_allocation: dict[str, Any] | None


def load_resume(
    checkpoint: Path,
    *,
    expected_config: Mapping[str, Any],
    parent: ParentProvenance,
) -> ResumeBundle:
    archive = checkpoint.expanduser().resolve()
    if archive.suffix != ".zip":
        archive = archive.with_suffix(".zip")
    sidecar_path = archive.with_suffix(".json")
    sidecar = _read_json(sidecar_path, "U1 checkpoint sidecar")
    if sidecar.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise U1ProtocolError("U1 resume checkpoint has an incompatible schema")
    if sidecar.get("protocol") != PROTOCOL:
        raise U1ProtocolError("U1 resume checkpoint has the wrong protocol")
    if file_sha256(archive) != sidecar.get("checkpoint_sha256"):
        raise U1ProtocolError("U1 resume checkpoint digest does not match its sidecar")
    if sidecar.get("effective_config") != dict(expected_config):
        raise U1ProtocolError("U1 resume configuration differs from the frozen child")
    if sidecar.get("parent") != parent.public_dict():
        raise U1ProtocolError("U1 resume checkpoint has different parent provenance")
    try:
        curriculum = CurriculumState.from_dict(sidecar["curriculum"])
        scheduler_state = dict(sidecar["scheduler"])
        progress = sidecar["progress"]
        lifetime_trained = int(progress["trained_timesteps"])
        child_trained = int(progress["child_trained_timesteps"])
        collected = int(progress["collected_timesteps"])
        n_updates = int(progress["n_updates"])
        segment_index = int(sidecar["segment"]["index"]) + 1
        last_completed = sidecar.get("last_completed_allocation")
    except (KeyError, TypeError, ValueError) as error:
        raise U1ProtocolError(f"invalid U1 resume state: {error}") from error
    if collected != lifetime_trained:
        raise U1ProtocolError("U1 resume checkpoint is not a fully trained boundary")
    if lifetime_trained - parent.trained_timesteps != child_trained:
        raise U1ProtocolError("U1 child/lifetime progress counters disagree")
    if not 0 <= child_trained <= int(expected_config["child_action_budget"]):
        raise U1ProtocolError("U1 resume progress is outside the declared child budget")
    return ResumeBundle(
        checkpoint=archive,
        sidecar=sidecar,
        curriculum=curriculum,
        scheduler_state=scheduler_state,
        lifetime_trained=lifetime_trained,
        child_trained=child_trained,
        n_updates=n_updates,
        segment_index=segment_index,
        last_completed_allocation=(
            dict(last_completed) if isinstance(last_completed, dict) else None
        ),
    )


def _safe_frame(path: Path, observation: np.ndarray) -> None:
    frame = observation
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(temporary, format="PNG")
    os.replace(temporary, path)


class _U1Mastered(RuntimeError):
    pass


class _CallbackFactory:
    @staticmethod
    def create(base_callback: Any) -> type[Any]:
        class U1Callback(base_callback):
            def __init__(
                self,
                *,
                run_directory: Path,
                state: CurriculumState,
                scheduler: TransitionDeficitScheduler,
                parent: ParentProvenance,
                segment: dict[str, Any],
                effective_config: dict[str, Any],
                evaluation_interval: int,
                checkpoint_interval: int,
                frame_interval: int,
                minimum_free_bytes: int,
                started_at: str,
                initial_trained_timesteps: int,
                initial_updates: int,
                child_start_timesteps: int,
                keep_checkpoints: int,
                initial_completed_allocation: dict[str, Any] | None = None,
            ) -> None:
                super().__init__(verbose=0)
                self.run_directory = run_directory
                self.state = state
                self.scheduler = scheduler
                self.parent = parent
                self.segment = segment
                self.effective_config = effective_config
                self.evaluation_interval = evaluation_interval
                self.checkpoint_interval = checkpoint_interval
                self.frame_interval = frame_interval
                self.minimum_free_bytes = minimum_free_bytes
                self.started_at = started_at
                self.trained_timesteps = initial_trained_timesteps
                self.last_updates = initial_updates
                self.child_start_timesteps = child_start_timesteps
                self.segment_start_timesteps = initial_trained_timesteps
                self.keep_checkpoints = keep_checkpoints
                self.wall_start = time.monotonic()
                self.next_evaluation = self._next_child_boundary(
                    initial_trained_timesteps, evaluation_interval
                )
                self.next_checkpoint = self._next_child_boundary(
                    initial_trained_timesteps, checkpoint_interval
                )
                self.next_status = initial_trained_timesteps
                self.next_frame = initial_trained_timesteps
                self.episodes = 0
                self.frame_revision = 0
                self.latest_evaluations: list[dict[str, Any]] = []
                self.latest_optimizer: dict[str, Any] = {}
                self.last_completed_allocation = initial_completed_allocation
                self.evaluation_history: deque[dict[str, Any]] = deque(maxlen=150)
                self.recent_outcomes: dict[str, deque[bool]] = defaultdict(
                    lambda: deque(maxlen=100)
                )
                self.latest_safe_checkpoint: Path | None = None
                self._last_saved_step: int | None = None
                self._last_saved_path: Path | None = None
                self._last_evaluation_step: int | None = None
                self._baseline_recorded = False

            def _next_child_boundary(self, current: int, interval: int) -> int:
                child = current - self.child_start_timesteps
                return self.child_start_timesteps + ((child // interval) + 1) * interval

            def _on_training_start(self) -> None:
                if int(self.model.num_timesteps) != self.trained_timesteps:
                    raise RuntimeError("model and child callback start boundaries differ")
                if int(self.model._n_updates) != self.last_updates:
                    raise RuntimeError("model and child optimizer boundaries differ")
                self._checkpoint("segment-initial", "initial")
                self._evaluate_all("segment_baseline", decide=False)
                self._baseline_recorded = True
                self._write_status("training")

            def _on_rollout_start(self) -> None:
                if int(self.model._n_updates) > self.last_updates:
                    self._process_boundary()
                if self.state.mastered:
                    raise _U1Mastered

            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                dones = self.locals.get("dones", [])
                for done, info in zip(dones, infos, strict=False):
                    if not done:
                        continue
                    lesson_id = str(info.get("lesson_id", LessonId.LOCAL_UNLOCK.value))
                    success = bool(info.get("success", False))
                    self.recent_outcomes[lesson_id].append(success)
                    self.episodes += 1
                    append_jsonl(
                        self.run_directory / "episodes.jsonl",
                        {
                            "timestamp": utc_now(),
                            "episode": self.episodes,
                            "collected_timesteps": int(self.model.num_timesteps),
                            "trained_timesteps": self.trained_timesteps,
                            "child_collected_timesteps": (
                                int(self.model.num_timesteps) - self.child_start_timesteps
                            ),
                            "lesson_id": lesson_id,
                            "lesson_label": info.get("lesson_label"),
                            "seed": info.get("seed"),
                            "layout_sha256": info.get("layout_sha256"),
                            "geometry_sha256": info.get("geometry_sha256"),
                            "success": success,
                            "terminal_reason": info.get("terminal_reason"),
                            "milestones": info.get("milestones"),
                            "unique_cells": info.get("unique_cells"),
                            "collisions": info.get("collisions"),
                            "ineffective_interactions": info.get(
                                "ineffective_interactions"
                            ),
                        },
                    )
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

            def _process_boundary(self) -> None:
                previous_updates = self.last_updates
                self.trained_timesteps = int(self.model.num_timesteps)
                self.last_updates = int(self.model._n_updates)
                self._record_optimizer_metrics(self.last_updates - previous_updates)
                if self.trained_timesteps >= self.next_checkpoint:
                    self._checkpoint(
                        f"step-{self.trained_timesteps:012d}", "step"
                    )
                    self.next_checkpoint = self._next_child_boundary(
                        self.trained_timesteps, self.checkpoint_interval
                    )
                    self._prune_numbered_checkpoints()
                if self.trained_timesteps >= self.next_evaluation:
                    self.next_evaluation = self._next_child_boundary(
                        self.trained_timesteps, self.evaluation_interval
                    )
                    self._evaluate_all("scheduled", decide=True)
                self._write_status("mastered" if self.state.mastered else "training")

            def _evaluation_checkpoint(self) -> Path:
                if (
                    self._last_saved_step == self.trained_timesteps
                    and self._last_saved_path is not None
                    and self._last_saved_path.is_file()
                ):
                    return self._last_saved_path
                return self._checkpoint(
                    f"step-{self.trained_timesteps:012d}", "exam"
                )

            def _evaluate_all(self, trigger: str, *, decide: bool) -> None:
                allocation_before = self.scheduler.snapshot()
                allocation_ok = self.scheduler.allocation_within()
                checkpoint = self._evaluation_checkpoint()
                digest = file_sha256(checkpoint)
                evaluations: dict[LessonId, LessonEvaluation] = {}
                for lesson in LessonId:
                    frame_path = (
                        self.run_directory
                        / "frames"
                        / f"exam-{lesson.value.replace('/', '-')}.png"
                    )
                    result = evaluate_lesson(
                        self.model,
                        lesson,
                        validation_seeds(lesson),
                        frame_path=frame_path,
                    )
                    evaluations[lesson] = result
                    append_jsonl(
                        self.run_directory / "evaluations.jsonl",
                        {
                            "trigger": trigger,
                            "counts_toward_gate": decide,
                            "trained_timesteps": self.trained_timesteps,
                            "child_trained_timesteps": (
                                self.trained_timesteps - self.child_start_timesteps
                            ),
                            "n_updates": int(self.model._n_updates),
                            "checkpoint": str(checkpoint.relative_to(self.run_directory)),
                            "checkpoint_sha256": digest,
                            "passed": lesson_passed(result),
                            **result.public_dict(),
                        },
                    )

                decision = "Diagnostic baseline; cannot count toward a gate"
                if decide:
                    self.last_completed_allocation = allocation_before
                    decision = self._apply_decision(evaluations, allocation_ok)
                    self._last_evaluation_step = self.trained_timesteps
                    if not self.state.mastered:
                        self.scheduler.start_window()
                        # The graded archive bytes are unchanged, but its resumable sidecar must
                        # reflect the exam decision and the newly opened allocation window.
                        self._checkpoint(
                            f"step-{self.trained_timesteps:012d}", "post-exam"
                        )

                self.latest_evaluations = [
                    evaluations[lesson].public_dict()
                    | {"passed": lesson_passed(evaluations[lesson])}
                    for lesson in LessonId
                ]
                for lesson in LessonId:
                    result = evaluations[lesson]
                    self.evaluation_history.append(
                        {
                            "trained_timesteps": self.trained_timesteps,
                            "child_trained_timesteps": (
                                self.trained_timesteps - self.child_start_timesteps
                            ),
                            "tier_label": result.lesson_label,
                            "lesson_id": result.lesson_id,
                            "success_rate": result.success_rate,
                            "panel_successes": result.panel_successes,
                            "decision": (
                                decision
                                if lesson is LessonId.LOCAL_UNLOCK
                                else "Retention exam"
                            ),
                            "counts_toward_gate": decide,
                        }
                    )
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": "curriculum_decision" if decide else "baseline_evaluation",
                        "trained_timesteps": self.trained_timesteps,
                        "child_trained_timesteps": (
                            self.trained_timesteps - self.child_start_timesteps
                        ),
                        "decision": decision,
                        "lesson_passes": {
                            lesson.value: lesson_passed(result)
                            for lesson, result in evaluations.items()
                        },
                        "curriculum": self.state.public_dict(),
                        "allocation": allocation_before,
                        "allocation_within_tolerance": allocation_ok,
                    },
                )
                self.frame_revision += 1

            def _record_optimizer_metrics(self, updates_since_boundary: int) -> None:
                values = getattr(self.model.logger, "name_to_value", {})
                fields = {
                    "entropy_loss": values.get("train/entropy_loss"),
                    "policy_gradient_loss": values.get("train/policy_gradient_loss"),
                    "value_loss": values.get("train/value_loss"),
                    "approx_kl": values.get("train/approx_kl"),
                    "clip_fraction": values.get("train/clip_fraction"),
                    "explained_variance": values.get("train/explained_variance"),
                    "loss": values.get("train/loss"),
                }
                self.latest_optimizer = {
                    key: float(value) if value is not None else None
                    for key, value in fields.items()
                }
                append_jsonl(
                    self.run_directory / "optimizer.jsonl",
                    {
                        "timestamp": utc_now(),
                        "collected_timesteps": int(self.model.num_timesteps),
                        "trained_timesteps": self.trained_timesteps,
                        "child_trained_timesteps": (
                            self.trained_timesteps - self.child_start_timesteps
                        ),
                        "n_updates": int(self.model._n_updates),
                        "updates_since_boundary": updates_since_boundary,
                        **self.latest_optimizer,
                    },
                )

            def _apply_decision(
                self,
                evaluations: Mapping[LessonId, LessonEvaluation],
                allocation_ok: bool,
            ) -> str:
                weak_anchor = recovery_cause(evaluations)
                if self.state.recovery_cause is not None:
                    self.state.consecutive_passes = 0
                    if weak_anchor is not None:
                        changed = weak_anchor is not self.state.recovery_cause
                        self.state.recovery_cause = weak_anchor
                        self.state.recovery_passes = 0
                        if changed:
                            self.state.change()
                        return f"Recovery continues: {weak_anchor.value} below gate"
                    if not allocation_ok:
                        self.state.recovery_passes = 0
                        return "Recovery held: transition allocation outside tolerance"
                    self.state.recovery_passes += 1
                    if self.state.recovery_passes < 2:
                        return "First recovered prerequisite exam; confirmation required"
                    self.state.recovery_cause = None
                    self.state.recovery_passes = 0
                    self.state.change()
                    return "Prerequisites recovered; normal U1 practice resumes"

                if weak_anchor is not None:
                    self.state.recovery_cause = weak_anchor
                    self.state.recovery_passes = 0
                    self.state.consecutive_passes = 0
                    self.state.change()
                    return f"Recovery started: {weak_anchor.value} below gate"

                u1_pass = lesson_passed(evaluations[LessonId.LOCAL_UNLOCK])
                full_pass = u1_pass and allocation_ok
                self.state.consecutive_passes = (
                    self.state.consecutive_passes + 1 if full_pass else 0
                )
                if not allocation_ok:
                    return "Held: transition allocation outside tolerance"
                if not u1_pass:
                    return "Held: Local Unlock below gate"
                if self.state.consecutive_passes < 2:
                    return "First cumulative U1 pass; confirmation required"
                self.state.mastered = True
                self._checkpoint("mastered-local-unlock", "mastery")
                return "U1 Local Unlock mastered with Navigate and U0 retained"

            def _checkpoint(self, name: str, kind: str) -> Path:
                if int(self.model.num_timesteps) != self.trained_timesteps:
                    raise RuntimeError("refusing U1 checkpoint outside a trained boundary")
                ensure_disk_space(self.run_directory, self.minimum_free_bytes)
                destination = self.run_directory / "checkpoints" / f"{name}.zip"
                if (
                    self._last_saved_step == self.trained_timesteps
                    and self._last_saved_path is not None
                    and self._last_saved_path.is_file()
                ):
                    atomic_copy_file(self._last_saved_path, destination)
                else:
                    atomic_model_save(self.model, destination)
                digest = file_sha256(destination)
                sidecar = {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "created_at": utc_now(),
                    "kind": kind,
                    "checkpoint_sha256": digest,
                    "effective_config": self.effective_config,
                    "parent": self.parent.public_dict(),
                    "curriculum": self.state.public_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "last_completed_allocation": self.last_completed_allocation,
                    "progress": {
                        "collected_timesteps": int(self.model.num_timesteps),
                        "trained_timesteps": self.trained_timesteps,
                        "child_trained_timesteps": (
                            self.trained_timesteps - self.child_start_timesteps
                        ),
                        "segment_trained_timesteps": (
                            self.trained_timesteps - self.segment_start_timesteps
                        ),
                        "n_updates": int(self.model._n_updates),
                    },
                    "segment": self.segment,
                }
                atomic_write_json(destination.with_suffix(".json"), sidecar)
                latest = destination.parent / "latest.zip"
                if destination != latest:
                    atomic_copy_file(destination, latest)
                    atomic_write_json(latest.with_suffix(".json"), sidecar | {"kind": "latest"})
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": "checkpoint",
                        "kind": kind,
                        "trained_timesteps": self.trained_timesteps,
                        "child_trained_timesteps": (
                            self.trained_timesteps - self.child_start_timesteps
                        ),
                        "sha256": digest,
                        "path": str(destination.relative_to(self.run_directory)),
                    },
                )
                self.latest_safe_checkpoint = destination
                self._last_saved_step = self.trained_timesteps
                self._last_saved_path = destination
                return destination

            def _prune_numbered_checkpoints(self) -> None:
                checkpoints = self.run_directory / "checkpoints"
                numbered = sorted(checkpoints.glob("step-*.zip"))
                for checkpoint in numbered[: -self.keep_checkpoints]:
                    checkpoint.unlink(missing_ok=True)
                    checkpoint.with_suffix(".json").unlink(missing_ok=True)

            def _write_status(self, phase: str) -> None:
                elapsed = time.monotonic() - self.wall_start
                collected = int(self.model.num_timesteps)
                segment_collected = collected - self.segment_start_timesteps
                recent = {
                    LESSON_SPECS[lesson].label: {
                        "episodes": len(self.recent_outcomes[lesson.value]),
                        "success_rate": (
                            sum(self.recent_outcomes[lesson.value])
                            / len(self.recent_outcomes[lesson.value])
                            if self.recent_outcomes[lesson.value]
                            else None
                        ),
                    }
                    for lesson in LessonId
                }
                atomic_write_json(
                    self.run_directory / "status.json",
                    {
                        "protocol": PROTOCOL,
                        "phase": phase,
                        "warm_start": True,
                        "parent_checkpoint_sha256": self.parent.checkpoint_sha256,
                        "started_at": self.started_at,
                        "updated_at": utc_now(),
                        "elapsed_seconds": elapsed,
                        "total_timesteps": collected,
                        "collected_timesteps": collected,
                        "trained_timesteps": self.trained_timesteps,
                        "child_collected_timesteps": (
                            collected - self.child_start_timesteps
                        ),
                        "child_trained_timesteps": (
                            self.trained_timesteps - self.child_start_timesteps
                        ),
                        "segment_collected_timesteps": segment_collected,
                        "segment_trained_timesteps": (
                            self.trained_timesteps - self.segment_start_timesteps
                        ),
                        "n_updates": int(self.model._n_updates),
                        "fps": segment_collected / elapsed if elapsed else 0.0,
                        "episodes": self.episodes,
                        "current_tier_label": "Local Unlock",
                        "current_lesson_id": LessonId.LOCAL_UNLOCK.value,
                        "current_lesson_label": "Local Unlock",
                        "curriculum": self.state.public_dict(),
                        "practice_allocation": self.scheduler.snapshot(),
                        "last_completed_practice_allocation": (
                            self.last_completed_allocation
                        ),
                        "allocation_within_tolerance": self.scheduler.allocation_within(),
                        "recent_training": recent,
                        "latest_optimizer": self.latest_optimizer,
                        "next_evaluation": self.next_evaluation,
                        "next_checkpoint": self.next_checkpoint,
                        "latest_evaluations": self.latest_evaluations,
                        "evaluation_history": list(self.evaluation_history),
                        "lesson_frames": {
                            lesson.value: f"frames/exam-{lesson.value.replace('/', '-')}.png"
                            for lesson in LessonId
                        },
                        "frame_revision": self.frame_revision,
                        "latest_safe_checkpoint": (
                            str(self.latest_safe_checkpoint.relative_to(self.run_directory))
                            if self.latest_safe_checkpoint
                            else None
                        ),
                        "segment": self.segment,
                    },
                )

            def finalize(self, phase: str) -> None:
                if int(self.model._n_updates) > self.last_updates:
                    self._process_boundary()
                if int(self.model.num_timesteps) != self.trained_timesteps:
                    raise RuntimeError("U1 run ended with an untrained partial rollout")
                if self._last_evaluation_step != self.trained_timesteps:
                    self._evaluate_all("final", decide=True)
                final_phase = "mastered" if self.state.mastered else phase
                self._checkpoint(final_phase, "final")
                self._write_status(final_phase)

            def finalize_failure(self, phase: str) -> None:
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": f"run_{phase}",
                        "collected_timesteps": int(self.model.num_timesteps),
                        "trained_timesteps": self.trained_timesteps,
                        "discarded_partial_timesteps": max(
                            0, int(self.model.num_timesteps) - self.trained_timesteps
                        ),
                    },
                )
                self._write_status(phase)

        return U1Callback


def _environment_factory(
    *,
    scheduler: TransitionDeficitScheduler,
    seed: int,
    size: int,
    forbidden_layout_hashes: Mapping[LessonId, frozenset[str]],
) -> Callable[[], gym.Env]:
    return lambda: make_training_env(
        scheduler=scheduler,
        seed=seed,
        size=size,
        forbidden_layout_hashes=forbidden_layout_hashes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--confirmation-report", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--child-budget", type=int, default=CHILD_ACTION_BUDGET)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--evaluation-every", type=int, default=EVALUATION_INTERVAL)
    parser.add_argument("--evaluation-seeds", type=int, default=EVALUATION_SEED_COUNT)
    parser.add_argument("--checkpoint-every", type=int, default=EVALUATION_INTERVAL)
    parser.add_argument("--keep-checkpoints", type=int, default=5)
    parser.add_argument("--frame-every", type=int, default=2_048)
    parser.add_argument("--qualification-seeds", type=int, default=QUALIFICATION_SEED_COUNT)
    parser.add_argument("--minimum-free-gib", type=float, default=25.0)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name")
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8784)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--engineering-smoke", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "child budget": args.child_budget,
        "workers": args.workers,
        "rollout steps": args.rollout_steps,
        "batch size": args.batch_size,
        "epochs": args.n_epochs,
        "evaluation interval": args.evaluation_every,
        "checkpoint interval": args.checkpoint_every,
        "checkpoint retention": args.keep_checkpoints,
        "frame interval": args.frame_every,
        "qualification seeds": args.qualification_seeds,
    }
    for label, value in positive.items():
        if value <= 0:
            raise SystemExit(f"{label} must be positive")
    if args.size != 9:
        raise SystemExit("the U1 child is qualified only at 9 x 9")
    rollout = args.workers * args.rollout_steps
    if rollout % args.batch_size:
        raise SystemExit("workers x rollout steps must be divisible by batch size")
    if args.child_budget % rollout:
        raise SystemExit("child budget must align to a complete vector rollout")
    if not math.isfinite(args.gamma) or not MINIMUM_GAMMA <= args.gamma <= 1.0:
        raise SystemExit("gamma violates reward dominance")
    if not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib < 0:
        raise SystemExit("minimum free GiB must be finite and non-negative")
    frozen = {
        "workers": (args.workers, 4),
        "rollout steps": (args.rollout_steps, 512),
        "batch size": (args.batch_size, 256),
        "epochs": (args.n_epochs, 4),
        "learning rate": (args.learning_rate, 2.5e-4),
        "gamma": (args.gamma, 0.995),
        "GAE lambda": (args.gae_lambda, 0.98),
        "evaluation interval": (args.evaluation_every, EVALUATION_INTERVAL),
        "evaluation seeds": (args.evaluation_seeds, EVALUATION_SEED_COUNT),
        "checkpoint interval": (args.checkpoint_every, EVALUATION_INTERVAL),
        "child budget": (args.child_budget, CHILD_ACTION_BUDGET),
        "qualification seeds": (args.qualification_seeds, QUALIFICATION_SEED_COUNT),
    }
    if not args.engineering_smoke:
        deviations = [
            f"{label}={actual!r} (requires {expected!r})"
            for label, (actual, expected) in frozen.items()
            if actual != expected
        ]
        if deviations:
            raise SystemExit("frozen U1 configuration changed:\n  - " + "\n  - ".join(deviations))


def _best_effort(label: str, operation: Callable[[], Any]) -> None:
    try:
        operation()
    except BaseException as error:
        print(f"Warning: {label} failed during finalization: {error}")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    parent = verify_parent(args.parent, args.confirmation_report)
    effective_config = _effective_config(args)
    resume = (
        load_resume(args.resume, expected_config=effective_config, parent=parent)
        if args.resume
        else None
    )
    initial_trained = resume.lifetime_trained if resume else parent.trained_timesteps
    child_trained = resume.child_trained if resume else 0
    remaining = args.child_budget - child_trained
    if remaining <= 0:
        raise SystemExit("the U1 child budget is already exhausted")

    minimum_free_bytes = int(args.minimum_free_gib * 1024**3)
    ensure_disk_space(args.run_root, minimum_free_bytes)
    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    except ImportError as error:
        raise SystemExit('Install training dependencies with: pip install -e ".[train]"') from error

    repository = Path(__file__).resolve().parents[2]
    source = git_snapshot(repository)
    if not args.engineering_smoke and source.get("dirty") is not False:
        raise SystemExit("the frozen U1 run requires a clean committed worktree")
    run_directory = create_run_directory(args.run_root, args.run_name)
    started_at = utc_now()
    qualification = qualify_local_unlock(args.qualification_seeds, size=args.size)
    atomic_write_json(run_directory / "qualification.json", qualification)
    if qualification["result"] != "passed":
        raise SystemExit(f"Local Unlock qualification failed; inspect {run_directory}")
    forbidden_layout_hashes = reserved_training_layout_hashes(qualification)

    state = resume.curriculum if resume else CurriculumState()
    segment_index = resume.segment_index if resume else 0
    segment_seed = args.seed + segment_index * 100_000
    scheduler = TransitionDeficitScheduler(state, seed=segment_seed + 99_000)
    if resume:
        scheduler.load_state_dict(resume.scheduler_state)
    factories = [
        _environment_factory(
            scheduler=scheduler,
            seed=segment_seed + worker,
            size=args.size,
            forbidden_layout_hashes=forbidden_layout_hashes,
        )
        for worker in range(args.workers)
    ]
    vector_environment = VecTransposeImage(DummyVecEnv(factories))
    source_checkpoint = resume.checkpoint if resume else Path(parent.checkpoint)
    model = RecurrentPPO.load(
        source_checkpoint,
        env=vector_environment,
        device=args.device,
    )
    expected_updates = resume.n_updates if resume else parent.n_updates
    if int(model.num_timesteps) != initial_trained:
        raise SystemExit(
            f"model/sidecar timestep mismatch: {model.num_timesteps} != {initial_trained}"
        )
    if int(model._n_updates) != expected_updates:
        raise SystemExit(
            f"model/sidecar optimizer mismatch: {model._n_updates} != {expected_updates}"
        )
    if not model.policy.optimizer.state:
        raise SystemExit("loaded PPO archive has no restored optimizer state")
    if (
        model.n_steps != args.rollout_steps
        or model.batch_size != args.batch_size
        or model.n_epochs != args.n_epochs
        or model.gamma != args.gamma
        or model.gae_lambda != args.gae_lambda
        or model.ent_coef != 0.01
        or float(model.learning_rate) != args.learning_rate
        or int(model.action_space.n) != 7
        or tuple(model.observation_space.shape) != (3, 56, 56)
        or int(model.policy.lstm_actor.hidden_size) != 256
        or int(model.policy.lstm_actor.num_layers) != 1
    ):
        raise SystemExit("loaded PPO archive does not match the frozen optimization contract")
    model.set_random_seed(segment_seed)

    segment = {
        "id": uuid.uuid4().hex,
        "index": segment_index,
        "started_at": started_at,
        "algorithm_seed": segment_seed,
        "worker_seed_base": segment_seed,
        "start_trained_timesteps": initial_trained,
        "start_child_trained_timesteps": child_trained,
        "remaining_child_budget": remaining,
        "resume_checkpoint": str(resume.checkpoint) if resume else None,
    }
    atomic_write_json(
        run_directory / "manifest.json",
        {
            "protocol": PROTOCOL,
            "started_at": started_at,
            "development_child": True,
            "engineering_smoke": bool(args.engineering_smoke),
            "warm_start": True,
            "parent": parent.public_dict(),
            "arguments": vars(args)
            | {
                "run_root": str(args.run_root),
                "parent": str(args.parent),
                "confirmation_report": str(args.confirmation_report),
                "resume": str(args.resume) if args.resume else None,
            },
            "effective_config": effective_config,
            "segment": segment,
            "git": source,
            "runtime": runtime_snapshot(),
            "information_boundary": "partial RGB pixels plus recurrent state only",
            "online_model_calls": False,
            "demonstrations": False,
            "oracle_actions_used_for_training": False,
        },
    )

    callback_type = _CallbackFactory.create(BaseCallback)
    callback = callback_type(
        run_directory=run_directory,
        state=state,
        scheduler=scheduler,
        parent=parent,
        segment=segment,
        effective_config=effective_config,
        evaluation_interval=args.evaluation_every,
        checkpoint_interval=args.checkpoint_every,
        frame_interval=args.frame_every,
        minimum_free_bytes=minimum_free_bytes,
        started_at=started_at,
        initial_trained_timesteps=initial_trained,
        initial_updates=expected_updates,
        child_start_timesteps=parent.trained_timesteps,
        keep_checkpoints=args.keep_checkpoints,
        initial_completed_allocation=(
            resume.last_completed_allocation if resume else None
        ),
    )
    dashboard = None
    if not args.no_dashboard:
        dashboard_error = None
        try:
            dashboard = start_dashboard(
                run_directory, host=args.dashboard_host, port=args.dashboard_port
            )
        except OSError as error:
            dashboard_error = str(error)
            try:
                dashboard = start_dashboard(
                    run_directory, host=args.dashboard_host, port=0
                )
            except OSError as fallback:
                dashboard_error = f"{dashboard_error}; fallback failed: {fallback}"
        if dashboard is not None:
            print(f"Dashboard: http://{args.dashboard_host}:{dashboard.server_port}/", flush=True)
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
                total_timesteps=remaining,
                callback=callback,
                reset_num_timesteps=False,
                progress_bar=False,
            )
        except _U1Mastered:
            phase = "mastered"
        except KeyboardInterrupt:
            phase = "interrupted"
        if phase == "interrupted":
            _best_effort("interruption record", lambda: callback.finalize_failure(phase))
        else:
            callback.finalize(phase)
    except BaseException:
        phase = "crashed"
        _best_effort(
            "crash report",
            lambda: atomic_write_json(
                run_directory / "crash.json",
                {"timestamp": utc_now(), "traceback": traceback.format_exc()},
            ),
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
