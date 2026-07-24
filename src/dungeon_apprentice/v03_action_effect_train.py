"""Non-resumable Stage-A trainer for the v0.3 action-effect study.

This module deliberately does not qualify or launch the experiment.  A
separately anchored :mod:`dungeon_apprentice.v03_action_effect_qualify`
release supplies the immutable source, tag, roots, guards, and disposable
first-rollout evidence.  The trainer consumes that evidence and fails closed
if the canonical arm does not reproduce it.

The implementation reuses the v0.2 U2 callback's atomic checkpoint publication
machinery, but owns every scientific decision that changed in v0.3:

* Dict observations are handled explicitly;
* evaluation uses :class:`ActionEffectPredictAdapter`;
* both arms run to the exact full budget, without an early-mastery stop;
* prerequisite recovery remains active;
* every one of the 32 post-update exams is retained;
* context and encoder diagnostics are recorded but never affect actions;
* all checkpoints are non-resumable and non-promotable; and
* interruption invalidates the complete matched Stage-A root.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from dungeon_apprentice import v02_u1_confirm as state_digests
from dungeon_apprentice import v02_u2 as frozen_u2
from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v02_u2r as frozen_u2r
from dungeon_apprentice import v02_u2s as frozen_u2s
from dungeon_apprentice import v02_u2s_smoke as frozen_smoke
from dungeon_apprentice import v03_action_effect as v03
from dungeon_apprentice import v03_action_effect_smoke as frozen_v03_smoke
from dungeon_apprentice.action_effect import (
    ACTION_EFFECT_DIM,
    ACTION_EFFECT_KEY,
    IMAGE_KEY,
    ActionEffectMode,
    ActionEffectObservation,
    ActionEffectPredictAdapter,
)
from dungeon_apprentice.artifacts import (
    append_jsonl,
    atomic_write_json,
    ensure_disk_space,
    file_sha256,
    git_snapshot,
    runtime_snapshot,
    utc_now,
)
from dungeon_apprentice.u2_storage import (
    StorageCapExceededError,
    enforce_directory_cap,
    ensure_directory_growth,
    validate_directory_path,
)

PROTOCOL = v03.PROTOCOL
SCHEMA_VERSION = 1
CASE_EVIDENCE_SCHEMA_VERSION = 1
FIRST_ROLLOUT_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1

PARENT_CHILD_SEED = v03.PARENT_CHILD_SEED
PARENT_LIFETIME_ACTIONS = v03.PARENT_LIFETIME_ACTIONS
PARENT_OPTIMIZER_UPDATES = v03.PARENT_OPTIMIZER_UPDATES
ALGORITHM_SEED = v03.ALGORITHM_SEED
WORKER_STREAMS = v03.WORKER_STREAMS
WORKERS = v03.WORKERS
ROLLOUT_STEPS = v03.ROLLOUT_STEPS
ROLLOUT_TRANSITIONS = v03.ROLLOUT_TRANSITIONS
PPO_EPOCHS = v03.PPO_EPOCHS
CHILD_ACTION_BUDGET = v03.CHILD_ACTION_BUDGET
EVALUATION_INTERVAL = v03.EVALUATION_INTERVAL
EVALUATION_SEED_COUNT = v03.EVALUATION_SEED_COUNT
EXAM_COUNT = v03.EXAM_COUNT
MINIMUM_FREE_GIB = v03.MINIMUM_FREE_GIB
KEEP_ROLLING_EXAMS = EXAM_COUNT


class ActionEffectTrainingError(RuntimeError):
    """Raised when the frozen Stage-A execution boundary is violated."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ActionEffectTrainingError(f"{label} is not a SHA-256 digest")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ActionEffectTrainingError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ActionEffectTrainingError(f"{label} must be a JSON object")
    return value


def extract_image_observation(
    observations: Mapping[str, Any],
    *,
    worker_index: int = 0,
) -> np.ndarray:
    """Return one CHW image from a vectorized v0.3 Dict observation."""

    if not isinstance(observations, Mapping) or set(observations) != {
        IMAGE_KEY,
        ACTION_EFFECT_KEY,
    }:
        raise ActionEffectTrainingError("v0.3 callback received a noncanonical Dict observation")
    images = np.asarray(observations[IMAGE_KEY])
    contexts = np.asarray(observations[ACTION_EFFECT_KEY])
    if (
        images.ndim != 4
        or tuple(images.shape[1:]) != (3, 56, 56)
        or images.dtype != np.uint8
        or contexts.ndim != 2
        or contexts.shape[0] != images.shape[0]
        or contexts.shape[1] != ACTION_EFFECT_DIM
        or contexts.dtype != np.float32
    ):
        raise ActionEffectTrainingError("v0.3 vector observation shape or dtype changed")
    index = int(worker_index)
    if not 0 <= index < images.shape[0]:
        raise IndexError("worker index is outside the vector observation")
    return np.array(images[index], copy=True)


def _raw_generator_state(owner: Any, label: str) -> dict[str, Any]:
    generator = getattr(owner, "np_random", None)
    bit_generator = getattr(generator, "bit_generator", None)
    state = getattr(bit_generator, "state", None)
    if not isinstance(state, Mapping):
        raise ActionEffectTrainingError(f"{label} has no NumPy RNG state")
    return copy.deepcopy(dict(state))


def _space_children(space: gym.Space) -> Mapping[str, gym.Space]:
    if isinstance(space, gym.spaces.Dict):
        return space.spaces
    return {}


def seed_space_tree(space: gym.Space, seed: int) -> dict[str, Any]:
    """Seed a Gym space and prove that every Dict child received state."""

    space.seed(int(seed))
    return snapshot_space_rng(space)


def snapshot_space_rng(space: gym.Space) -> dict[str, Any]:
    """Capture a complete recursively restorable RNG tree."""

    state = _raw_generator_state(space, f"{type(space).__name__} space")
    children = {
        str(name): snapshot_space_rng(child) for name, child in _space_children(space).items()
    }
    return {
        "type": type(space).__name__,
        "state": state,
        "state_sha256": frozen_u2s._rng_state_sha256(state),
        "children": children,
    }


def restore_space_rng(space: gym.Space, snapshot: Mapping[str, Any]) -> None:
    """Restore a space RNG tree, rejecting key or type drift."""

    children = _space_children(space)
    raw_children = snapshot.get("children")
    if (
        snapshot.get("type") != type(space).__name__
        or not isinstance(snapshot.get("state"), Mapping)
        or not isinstance(raw_children, Mapping)
        or set(raw_children) != set(children)
    ):
        raise ActionEffectTrainingError("space RNG snapshot topology changed")
    space.np_random.bit_generator.state = copy.deepcopy(snapshot["state"])
    for name, child in children.items():
        raw_child = raw_children[name]
        if not isinstance(raw_child, Mapping):
            raise ActionEffectTrainingError("space RNG child snapshot is invalid")
        restore_space_rng(child, raw_child)
    if snapshot_space_rng(space) != dict(snapshot):
        raise ActionEffectTrainingError("space RNG restoration was not exact")


def _space_rng_identity(space: gym.Space) -> dict[str, Any]:
    snapshot = snapshot_space_rng(space)
    return {
        "type": snapshot["type"],
        "state_sha256": snapshot["state_sha256"],
        "children": {
            name: _identity_from_space_snapshot(child)
            for name, child in snapshot["children"].items()
        },
    }


def _smoke_space_rng_identity(space: gym.Space) -> dict[str, Any]:
    """Use the disposable qualification smoke's exact public schema."""

    value: dict[str, Any] = {
        "type": type(space).__name__,
        "self_sha256": frozen_u2s._space_rng_state_sha256(
            space,
            f"v0.3 {type(space).__name__}",
        ),
    }
    if isinstance(space, gym.spaces.Dict):
        value["children"] = {
            key: _smoke_space_rng_identity(child) for key, child in space.spaces.items()
        }
    return value


def _identity_from_space_snapshot(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    children = snapshot.get("children")
    if not isinstance(children, Mapping):
        raise ActionEffectTrainingError("space RNG identity is incomplete")
    return {
        "type": str(snapshot["type"]),
        "state_sha256": _require_sha256(
            snapshot["state_sha256"],
            "space RNG state",
        ),
        "children": {
            str(name): _identity_from_space_snapshot(child) for name, child in children.items()
        },
    }


@dataclass(frozen=True)
class WorkerHandle:
    """The three wrappers whose state must remain bound for one worker."""

    episode: Any
    context: ActionEffectObservation
    first_rollout: FirstRolloutEvidenceWrapper | None = None


def _ordered_workers(
    workers: Sequence[WorkerHandle],
) -> tuple[WorkerHandle, ...]:
    ordered = tuple(sorted(workers, key=lambda item: int(item.episode.worker_index)))
    if (
        len(ordered) != WORKERS
        or [int(item.episode.worker_index) for item in ordered] != list(range(WORKERS))
        or [int(item.episode.worker_stream) for item in ordered] != list(WORKER_STREAMS)
    ):
        raise ActionEffectTrainingError("v0.3 requires the four frozen worker streams")
    return ordered


def seed_initial_rng_spaces(workers: Sequence[WorkerHandle]) -> None:
    """Eliminate lazy operating-system entropy before the first reset."""

    for handle in _ordered_workers(workers):
        episode = handle.episode
        if (
            int(episode.episode_ordinal) != 0
            or int(episode.local_transition_count) != 0
            or episode._active is not None
        ):
            raise ActionEffectTrainingError(
                "v0.3 spaces must be seeded before the first worker reset"
            )
        stream = int(episode.worker_stream)
        seed_space_tree(handle.context.action_space, stream)
        seed_space_tree(handle.context.observation_space, stream)


def _torch_rng_identity() -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency message
        raise ActionEffectTrainingError(
            'install training dependencies with: pip install -e ".[train]"'
        ) from error
    mps: str | None = None
    if bool(
        getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available()
    ):
        mps = frozen_u2s._torch_rng_state_sha256(torch.mps.get_rng_state())
    cuda: str | None = None
    if torch.cuda.is_available():
        cuda = _canonical_sha256(
            [frozen_u2s._torch_rng_state_sha256(state) for state in torch.cuda.get_rng_state_all()]
        )
    return {
        "torch_cpu_sha256": frozen_u2s._torch_rng_state_sha256(torch.random.get_rng_state()),
        "torch_mps_sha256": mps,
        "torch_cuda_sha256": cuda,
    }


def capture_rng_identity(
    *,
    model: Any,
    scheduler: lessons.TransitionDeficitScheduler,
    workers: Sequence[WorkerHandle],
    phase: str,
) -> dict[str, Any]:
    """Seal the complete stochastic identity at a declared boundary."""

    ordered = _ordered_workers(workers)
    worker_components: list[dict[str, Any]] = []
    for handle in ordered:
        episode = handle.episode
        curriculum = frozen_u2s._curriculum_environment(episode)
        worker_components.append(
            {
                "worker_index": int(episode.worker_index),
                "worker_stream": int(episode.worker_stream),
                "curriculum_sha256": frozen_u2s._rng_state_sha256(
                    curriculum._rng.bit_generator.state
                ),
                "base_environment_sha256": frozen_u2s._rng_state_sha256(
                    curriculum.unwrapped.np_random.bit_generator.state
                ),
                "episode_position_sha256": _canonical_sha256(episode.evidence_state()),
                "action_space": _space_rng_identity(handle.context.action_space),
                "observation_space": _space_rng_identity(handle.context.observation_space),
            }
        )
    components = {
        "python_random_sha256": frozen_u2s._rng_state_sha256(random.getstate()),
        "numpy_global_sha256": frozen_u2s._rng_state_sha256(np.random.get_state()),
        **_torch_rng_identity(),
        "model_action_space": _space_rng_identity(model.action_space),
        "model_observation_space": _space_rng_identity(model.observation_space),
        "scheduler_sha256": frozen_u2s._rng_state_sha256(scheduler.state_dict()),
        "workers": worker_components,
    }
    value = {
        "schema_version": 1,
        "phase": str(phase),
        "algorithm_seed": ALGORITHM_SEED,
        "components": components,
        "aggregate_sha256": _canonical_sha256(components),
    }
    return verify_rng_identity(value)


def capture_qualified_rng_identity(
    *,
    model: Any,
    scheduler: lessons.TransitionDeficitScheduler,
    workers: Sequence[WorkerHandle],
    phase: str,
) -> dict[str, Any]:
    """Reproduce the qualification smoke's matched RNG identity exactly."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency message
        raise ActionEffectTrainingError("v0.3 qualification RNG identity requires Torch") from error
    values: list[dict[str, Any]] = []
    for handle in _ordered_workers(workers):
        episode = handle.episode
        curriculum = frozen_u2s._curriculum_environment(episode)
        values.append(
            {
                "worker_index": int(episode.worker_index),
                "worker_stream": int(episode.worker_stream),
                "curriculum_sha256": frozen_u2s._rng_state_sha256(
                    curriculum._rng.bit_generator.state
                ),
                "base_environment_sha256": frozen_u2s._rng_state_sha256(
                    curriculum.unwrapped.np_random.bit_generator.state
                ),
                "action_space": _smoke_space_rng_identity(handle.context.action_space),
                "observation_space": _smoke_space_rng_identity(handle.context.observation_space),
            }
        )
    components = {
        "python_random_sha256": frozen_u2s._rng_state_sha256(random.getstate()),
        "numpy_global_sha256": frozen_u2s._rng_state_sha256(np.random.get_state()),
        "torch_cpu_sha256": frozen_u2s._torch_rng_state_sha256(torch.random.get_rng_state()),
        "model_action_space": _smoke_space_rng_identity(model.action_space),
        "scheduler_sha256": frozen_u2s._rng_state_sha256(scheduler._rng.bit_generator.state),
        "workers": values,
    }
    return {
        "phase": str(phase),
        "components": components,
        "aggregate_sha256": frozen_u2s._canonical_sha256(components),
    }


def verify_rng_identity(
    value: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a v0.3 RNG identity and optionally require exact equality."""

    if set(value) != {
        "schema_version",
        "phase",
        "algorithm_seed",
        "components",
        "aggregate_sha256",
    }:
        raise ActionEffectTrainingError("v0.3 RNG identity schema changed")
    components = value.get("components")
    if (
        value.get("schema_version") != 1
        or not isinstance(value.get("phase"), str)
        or not value["phase"]
        or value.get("algorithm_seed") != ALGORITHM_SEED
        or not isinstance(components, Mapping)
        or value.get("aggregate_sha256") != _canonical_sha256(components)
    ):
        raise ActionEffectTrainingError("v0.3 RNG identity is invalid")
    _require_sha256(value["aggregate_sha256"], "RNG aggregate")
    normalized = json.loads(json.dumps(value))
    if expected is not None and normalized != verify_rng_identity(expected):
        raise ActionEffectTrainingError("v0.3 RNG identity differs from qualification")
    return normalized


def snapshot_training_rng_state(
    *,
    model: Any,
    scheduler: lessons.TransitionDeficitScheduler,
    workers: Sequence[WorkerHandle],
) -> dict[str, Any]:
    """Capture every RNG component that diagnostic evaluation must not move."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency message
        raise ActionEffectTrainingError("v0.3 RNG preservation requires Torch") from error
    worker_states: list[dict[str, Any]] = []
    for handle in _ordered_workers(workers):
        curriculum = frozen_u2s._curriculum_environment(handle.episode)
        worker_states.append(
            {
                "curriculum": copy.deepcopy(curriculum._rng.bit_generator.state),
                "base_environment": copy.deepcopy(
                    curriculum.unwrapped.np_random.bit_generator.state
                ),
                "action_space": snapshot_space_rng(handle.context.action_space),
                "observation_space": snapshot_space_rng(handle.context.observation_space),
            }
        )
    return {
        "python_random": random.getstate(),
        "numpy_global": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_mps": (
            torch.mps.get_rng_state().clone()
            if bool(
                getattr(getattr(torch, "backends", None), "mps", None)
                and torch.backends.mps.is_available()
            )
            else None
        ),
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
        "model_action_space": snapshot_space_rng(model.action_space),
        "model_observation_space": snapshot_space_rng(model.observation_space),
        "scheduler": copy.deepcopy(scheduler._rng.bit_generator.state),
        "workers": worker_states,
    }


def restore_training_rng_state(
    snapshot: Mapping[str, Any],
    *,
    model: Any,
    scheduler: lessons.TransitionDeficitScheduler,
    workers: Sequence[WorkerHandle],
) -> None:
    """Restore the complete pre-diagnostic RNG snapshot."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency message
        raise ActionEffectTrainingError("v0.3 RNG restoration requires Torch") from error
    ordered = _ordered_workers(workers)
    raw_workers = snapshot.get("workers")
    if not isinstance(raw_workers, list) or len(raw_workers) != len(ordered):
        raise ActionEffectTrainingError("v0.3 RNG restoration worker inventory changed")
    random.setstate(snapshot["python_random"])
    np.random.set_state(snapshot["numpy_global"])
    torch.random.set_rng_state(snapshot["torch_cpu"])
    if snapshot.get("torch_mps") is not None:
        torch.mps.set_rng_state(snapshot["torch_mps"])
    if snapshot.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(snapshot["torch_cuda"])
    restore_space_rng(model.action_space, snapshot["model_action_space"])
    restore_space_rng(
        model.observation_space,
        snapshot["model_observation_space"],
    )
    scheduler._rng.bit_generator.state = copy.deepcopy(snapshot["scheduler"])
    for handle, raw in zip(ordered, raw_workers, strict=True):
        if not isinstance(raw, Mapping):
            raise ActionEffectTrainingError("v0.3 worker RNG restoration record changed")
        curriculum = frozen_u2s._curriculum_environment(handle.episode)
        curriculum._rng.bit_generator.state = copy.deepcopy(raw["curriculum"])
        curriculum.unwrapped.np_random.bit_generator.state = copy.deepcopy(raw["base_environment"])
        restore_space_rng(handle.context.action_space, raw["action_space"])
        restore_space_rng(
            handle.context.observation_space,
            raw["observation_space"],
        )


def _observation_sha256(value: Any) -> str:
    """Use the qualification smoke's byte-for-byte observation identity."""

    return frozen_smoke._observation_sha256(value)


def _qualification_canonical_sha256(value: Any) -> str:
    """Hash first-rollout evidence with the frozen smoke primitive."""

    return frozen_smoke._canonical_sha256(value)


def _qualification_episode_ledger_sha256(path: Path) -> str:
    """Normalize and hash a ledger exactly as the qualification smoke does."""

    records = frozen_smoke._read_normalized_episode_ledger(path)
    return _qualification_canonical_sha256(records)


class FirstRolloutEvidenceWrapper(gym.Wrapper):
    """Record exactly one worker's bounded first-rollout trajectory."""

    def __init__(
        self,
        env: gym.Env,
        *,
        worker_index: int,
        transition_limit: int = ROLLOUT_STEPS,
    ) -> None:
        super().__init__(env)
        self.worker_index = int(worker_index)
        self.transition_limit = int(transition_limit)
        self.transitions = 0
        self.resets: list[dict[str, Any]] = []
        self._trajectory: list[dict[str, Any]] = []
        self._rewards: list[dict[str, str]] = []
        self._previous_observation: np.ndarray | None = None
        self.action_counts: Counter[int] = Counter()
        self.lesson_counts: Counter[str] = Counter()
        self._boundary_reset_pending = False

    @property
    def complete(self) -> bool:
        return self.transitions == self.transition_limit

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        record_boundary_reset = (
            self.transitions == self.transition_limit and self._boundary_reset_pending
        )
        if self.transitions < self.transition_limit or record_boundary_reset:
            self.resets.append(
                {
                    "worker_index": self.worker_index,
                    "episode_ordinal": len(self.resets) + 1,
                    "seed": int(info["seed"]),
                    "seed_role": (
                        str(info["u2_seed_role"]) if info.get("u2_seed_role") is not None else None
                    ),
                    "lesson_id": str(info["lesson_id"]),
                    "layout_sha256": str(info["layout_sha256"]),
                    "geometry_sha256": str(info["geometry_sha256"]),
                    "worker_transition_at_start": self.transitions,
                }
            )
            self._previous_observation = np.array(
                observation,
                copy=True,
            )
            if record_boundary_reset:
                self._boundary_reset_pending = False
        return observation, info

    def step(
        self,
        action: Any,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        selected = int(np.asarray(action).item())
        observation, reward, terminated, truncated, info = self.env.step(selected)
        if self.transitions < self.transition_limit:
            if self._previous_observation is None:
                raise ActionEffectTrainingError("first-rollout worker stepped before reset")
            extrinsic = float(info.get("extrinsic_reward", reward))
            curiosity = float(info.get("curiosity_reward", 0.0))
            if not all(math.isfinite(value) for value in (float(reward), extrinsic, curiosity)):
                raise ActionEffectTrainingError("first-rollout reward is non-finite")
            self.transitions += 1
            self.action_counts[selected] += 1
            lesson_id = str(info["lesson_id"])
            self.lesson_counts[lesson_id] += 1
            self._trajectory.append(
                {
                    "worker_index": self.worker_index,
                    "worker_transition": self.transitions,
                    "action": selected,
                    "lesson_id": lesson_id,
                    "before_sha256": _observation_sha256(self._previous_observation),
                    "after_sha256": _observation_sha256(observation),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            self._rewards.append(
                {
                    "reward": frozen_smoke._float_hex(
                        reward,
                        "first-rollout reward",
                    ),
                    "extrinsic": frozen_smoke._float_hex(
                        extrinsic,
                        "first-rollout extrinsic reward",
                    ),
                    "curiosity": frozen_smoke._float_hex(
                        curiosity,
                        "first-rollout curiosity reward",
                    ),
                    "penalty": frozen_smoke._float_hex(
                        0.0,
                        "first-rollout no-effect penalty",
                    ),
                }
            )
            self._previous_observation = np.array(observation, copy=True)
            self._boundary_reset_pending = bool(
                self.transitions == self.transition_limit and (terminated or truncated)
            )
        elif self._boundary_reset_pending:
            # A reset immediately follows a terminal VecEnv transition.  Any
            # subsequent step proves that boundary reset did not occur.
            self._boundary_reset_pending = False
        return observation, reward, terminated, truncated, info

    def public_dict(self) -> dict[str, Any]:
        if not self.complete:
            raise ActionEffectTrainingError("first-rollout evidence is incomplete")
        return {
            "worker_index": self.worker_index,
            "transitions": self.transitions,
            "episodes_started": len(self.resets),
            "episode_starts_sha256": _qualification_canonical_sha256(self.resets),
            "action_counts": {
                str(action): int(self.action_counts.get(action, 0)) for action in range(7)
            },
            "lesson_transition_counts": {
                lesson.value: int(self.lesson_counts.get(lesson.value, 0))
                for lesson in lessons.LessonId
            },
            "trajectory_sha256": _qualification_canonical_sha256(self._trajectory),
            "reward_evidence_sha256": _qualification_canonical_sha256(self._rewards),
        }


def _update_array_digest(digest: Any, label: str, value: Any) -> None:
    frozen_v03_smoke._update_array_digest(digest, label, value)


@dataclass
class StageAController:
    """Full-budget practice controller; it never marks the arm mastered."""

    exam_records: list[dict[str, Any]] = field(default_factory=list)
    practice_decisions: list[dict[str, Any]] = field(default_factory=list)

    def public_dict(self) -> dict[str, Any]:
        return {
            "exam_records": json.loads(json.dumps(self.exam_records)),
            "practice_decisions": json.loads(json.dumps(self.practice_decisions)),
        }

    def update_recovery(
        self,
        state: lessons.CurriculumState,
        evaluations: Mapping[lessons.LessonId, Any],
        *,
        allocation_valid: bool,
    ) -> str:
        """Apply frozen prerequisite recovery without early stopping."""

        weak = tuple(lessons.weak_prerequisites(evaluations))
        in_recovery = bool(tuple(state.weak_prerequisites))
        state.mastered = False
        state.consecutive_passes = 0
        if in_recovery:
            if weak:
                changed = tuple(state.weak_prerequisites) != weak
                state.weak_prerequisites = weak
                state.recovery_passes = 0
                decision = "recovery_changed" if changed else "recovery_continues"
            elif not allocation_valid:
                state.recovery_passes = 0
                decision = "recovery_held_allocation_invalid"
            else:
                state.recovery_passes += 1
                if state.recovery_passes < 2:
                    decision = "recovery_first_clean_boundary"
                else:
                    state.weak_prerequisites = ()
                    state.recovery_passes = 0
                    decision = "recovery_completed"
        elif weak:
            state.weak_prerequisites = weak
            state.recovery_passes = 0
            decision = "recovery_started"
        else:
            state.recovery_passes = 0
            decision = "normal_practice_continues"
        state.change()
        if state.mastered or state.consecutive_passes:
            raise ActionEffectTrainingError("Stage A controller attempted an early mastery stop")
        return decision


@dataclass
class ContextMetrics:
    """Training-only descriptive context counts."""

    transitions: int = 0
    active_contexts: int = 0
    changed: int = 0
    unchanged: int = 0
    by_action: dict[str, dict[str, int]] = field(
        default_factory=lambda: {str(action): {"changed": 0, "unchanged": 0} for action in range(7)}
    )
    by_lesson: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            lesson.value: {"changed": 0, "unchanged": 0} for lesson in lessons.LessonId
        }
    )

    def observe(
        self,
        *,
        action: int,
        lesson_id: str,
        changed: bool,
        active: bool,
    ) -> None:
        selected = int(action)
        if not 0 <= selected < 7 or lesson_id not in self.by_lesson:
            raise ActionEffectTrainingError("context metric received an invalid action or lesson")
        label = "changed" if changed else "unchanged"
        self.transitions += 1
        self.active_contexts += int(bool(active))
        self.changed += int(bool(changed))
        self.unchanged += int(not changed)
        self.by_action[str(selected)][label] += 1
        self.by_lesson[lesson_id][label] += 1

    def public_dict(self) -> dict[str, Any]:
        return {
            "transitions": self.transitions,
            "active_contexts": self.active_contexts,
            "activation_rate": (
                self.active_contexts / self.transitions if self.transitions else 0.0
            ),
            "changed": self.changed,
            "unchanged": self.unchanged,
            "changed_rate": (self.changed / self.transitions if self.transitions else 0.0),
            "by_action": json.loads(json.dumps(self.by_action)),
            "by_lesson": json.loads(json.dumps(self.by_lesson)),
        }


def encoder_metrics(model: Any, previous: Any | None) -> tuple[dict[str, Any], Any]:
    """Measure the context projection without feeding it back into learning."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency message
        raise ActionEffectTrainingError("v0.3 encoder metrics require Torch") from error
    weight = model.policy.features_extractor.action_effect_encoder.weight.detach().cpu()
    if tuple(weight.shape) != (512, ACTION_EFFECT_DIM):
        raise ActionEffectTrainingError("v0.3 encoder shape changed")
    prior = torch.zeros_like(weight) if previous is None else previous
    if tuple(prior.shape) != tuple(weight.shape):
        raise ActionEffectTrainingError("v0.3 prior encoder snapshot shape changed")
    delta = weight - prior
    result = {
        "weight_l2_norm": float(torch.linalg.vector_norm(weight).item()),
        "weight_max_abs": float(weight.abs().max().item()),
        "weight_nonzero_parameters": int(torch.count_nonzero(weight).item()),
        "update_l2_norm": float(torch.linalg.vector_norm(delta).item()),
        "update_max_abs": float(delta.abs().max().item()),
        "update_nonzero_parameters": int(torch.count_nonzero(delta).item()),
    }
    if not all(
        math.isfinite(float(value)) for key, value in result.items() if "parameters" not in key
    ):
        raise ActionEffectTrainingError("v0.3 encoder metric is non-finite")
    return result, weight.clone()


def optimizer_update_count_valid(
    child_actions: int,
    updates: int,
    *,
    parent_updates: int = PARENT_OPTIMIZER_UPDATES,
) -> bool:
    actions = int(child_actions)
    return (
        actions >= 0
        and actions % ROLLOUT_TRANSITIONS == 0
        and int(updates) == int(parent_updates) + actions // ROLLOUT_TRANSITIONS * PPO_EPOCHS
    )


def audit_stage_a_storage(
    *,
    storage_root: Path,
    cohort_directory: Path,
    lineage_directory: Path,
    media_directory: Path,
    protected_paths: Sequence[Path] = (),
    anticipated_lineage_bytes: int = 0,
) -> dict[str, Any]:
    """Apply the smaller, v0.3-specific 2/4/6/10 GiB ceilings."""

    if anticipated_lineage_bytes < 0:
        raise ValueError("anticipated v0.3 growth cannot be negative")
    protected = tuple(protected_paths)
    cohort = validate_directory_path(
        cohort_directory,
        root=storage_root,
        protected_paths=protected,
    )
    lineage = validate_directory_path(
        lineage_directory,
        root=storage_root,
        protected_paths=protected,
    )
    media = validate_directory_path(
        media_directory,
        root=storage_root,
        protected_paths=protected,
    )
    if lineage == cohort or cohort not in lineage.parents:
        raise ActionEffectTrainingError("v0.3 lineage is not inside its cohort")
    if media == cohort or cohort in media.parents or media in cohort.parents:
        raise ActionEffectTrainingError("v0.3 media root overlaps scientific evidence")
    lineage_usage = enforce_directory_cap(
        lineage,
        cap_bytes=v03.LINEAGE_CAP_BYTES,
        label="v0.3 arm scientific artifacts",
    )
    cohort_usage = enforce_directory_cap(
        cohort,
        cap_bytes=v03.COHORT_SCIENTIFIC_CAP_BYTES,
        label="v0.3 cohort scientific artifacts",
    )
    media_usage = enforce_directory_cap(
        media,
        cap_bytes=v03.MEDIA_CAP_BYTES,
        label="v0.3 narrative media",
    )
    ensure_directory_growth(
        lineage_usage,
        anticipated_lineage_bytes,
    )
    ensure_directory_growth(
        cohort_usage,
        anticipated_lineage_bytes,
    )
    projected = cohort_usage.used_bytes + anticipated_lineage_bytes + media_usage.used_bytes
    if projected > v03.COMBINED_PLANNED_CAP_BYTES:
        raise StorageCapExceededError(
            "v0.3 projected scientific and media footprint exceeds its 10 GiB combined cap"
        )
    return {
        "lineage": lineage_usage.as_dict(),
        "cohort": cohort_usage.as_dict(),
        "media": media_usage.as_dict(),
        "anticipated_lineage_bytes": anticipated_lineage_bytes,
        "projected_combined_used_bytes": projected,
        "caps": {
            "lineage_bytes": v03.LINEAGE_CAP_BYTES,
            "cohort_scientific_bytes": (v03.COHORT_SCIENTIFIC_CAP_BYTES),
            "media_bytes": v03.MEDIA_CAP_BYTES,
            "combined_planned_bytes": (v03.COMBINED_PLANNED_CAP_BYTES),
        },
        "within_caps": True,
    }


def _integrity_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".integrity.json")


def _write_integrity(checkpoint: Path, sidecar: Path) -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_sha256(checkpoint),
        "sidecar": sidecar.name,
        "sidecar_sha256": file_sha256(sidecar),
    }
    atomic_write_json(_integrity_path(checkpoint), value)
    return value


def _verify_integrity(
    checkpoint: Path,
    sidecar: Path,
) -> dict[str, Any]:
    integrity_path = _integrity_path(checkpoint)
    if any(path.is_symlink() for path in (checkpoint, sidecar, integrity_path)) or not all(
        path.is_file() for path in (checkpoint, sidecar, integrity_path)
    ):
        raise ActionEffectTrainingError("v0.3 checkpoint bundle is missing or unsafe")
    value = _read_json(integrity_path, "v0.3 checkpoint integrity")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_sha256(checkpoint),
        "sidecar": sidecar.name,
        "sidecar_sha256": file_sha256(sidecar),
    }
    if value != expected:
        raise ActionEffectTrainingError("v0.3 checkpoint integrity record changed")
    return value


def _normalized_trajectory_identity(
    workers: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return frozen_v03_smoke._trajectory_identity(list(workers))


def normalized_first_rollout_identity(
    arm_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the arm-independent qualification-smoke identity."""

    workers = arm_evidence.get("workers")
    post_rng = arm_evidence.get("post_rollout_rng_identity")
    policy_output = arm_evidence.get("policy_output_sha256")
    episode_ledger = arm_evidence.get("episode_ledger")
    if (
        not isinstance(workers, list)
        or len(workers) != WORKERS
        or not isinstance(post_rng, Mapping)
        or not isinstance(policy_output, str)
        or not isinstance(episode_ledger, Mapping)
    ):
        raise ActionEffectTrainingError("qualification first-rollout evidence is incomplete")
    _require_sha256(policy_output, "first-rollout policy output")
    ledger_digest = _require_sha256(
        episode_ledger.get("normalized_sha256"),
        "first-rollout episode ledger",
    )
    identity = {
        "trajectory_identity": _normalized_trajectory_identity(workers),
        "policy_output_sha256": policy_output,
        "post_rollout_rng_identity": json.loads(json.dumps(post_rng)),
        "episode_ledger_normalized_sha256": ledger_digest,
    }
    identity["aggregate_sha256"] = _qualification_canonical_sha256(identity)
    return identity


def _qualification_smoke_arm(
    qualification_report: Mapping[str, Any],
    arm: ActionEffectMode,
) -> Mapping[str, Any]:
    smoke = qualification_report.get("smoke_evidence")
    if not isinstance(smoke, Mapping):
        raise ActionEffectTrainingError("v0.3 qualification lacks disposable smoke evidence")
    # The qualifier may retain either the complete smoke report or a
    # normalized nested copy.  Both forms keep the arm evidence explicit.
    arms = smoke.get("arms")
    if not isinstance(arms, Mapping):
        full_report = smoke.get("_full_report")
        arms = full_report.get("arms") if isinstance(full_report, Mapping) else None
    if not isinstance(arms, Mapping):
        result = smoke.get("result")
        arms = result.get("arms") if isinstance(result, Mapping) else None
    selected = arms.get(arm.value) if isinstance(arms, Mapping) else None
    if not isinstance(selected, Mapping):
        raise ActionEffectTrainingError(f"v0.3 qualification lacks {arm.value} smoke evidence")
    return selected


def _qualification_public(qualification: Any) -> dict[str, Any]:
    value = qualification.public_dict()
    if not isinstance(value, Mapping):
        raise ActionEffectTrainingError("v0.3 qualification public evidence is invalid")
    return json.loads(json.dumps(value))


def _qualification_report(qualification: Any) -> dict[str, Any]:
    value = qualification.verified_report()
    if not isinstance(value, Mapping):
        raise ActionEffectTrainingError("v0.3 qualification report is invalid")
    return json.loads(json.dumps(value))


def _case_diagnostics(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return frozen_u2s._case_diagnostics_from_records(cases)


def evaluate_architecture_lesson(
    model: Any,
    *,
    mode: ActionEffectMode | str,
    lesson: lessons.LessonId | str,
    seeds: Sequence[int],
    seed_access: Any,
    frame_path: Path | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Run a frozen raw-pixel exam through the context reconstruction adapter."""

    adapter = ActionEffectPredictAdapter(model, mode=mode)
    return frozen_u2s.evaluate_lesson_with_visible_no_effect(
        adapter,
        lesson,
        seeds,
        seed_access=seed_access,
        penalty_enabled=False,
        frame_path=frame_path,
    )


class _CallbackFactory:
    """Create a Stage-A callback while keeping training dependencies optional."""

    @staticmethod
    def create(base_callback: Any) -> type[Any]:
        frozen_base = frozen_u2._CallbackFactory.create(base_callback)

        class ActionEffectCallback(frozen_base):
            def __init__(
                self,
                *,
                arm: ActionEffectMode,
                manifest: Mapping[str, Any],
                workers: Sequence[WorkerHandle],
                transplant: Mapping[str, Any],
                qualification_report: Mapping[str, Any],
                cohort_contract: Mapping[str, Any],
                cohort_contract_sha256: str,
                expected_pre_action_rng: Mapping[str, Any],
                expected_first_rollout: Mapping[str, Any],
                **kwargs: Any,
            ) -> None:
                self.arm = ActionEffectMode(arm)
                self.manifest = json.loads(json.dumps(manifest))
                self.workers = _ordered_workers(workers)
                self.transplant = json.loads(json.dumps(transplant))
                self.qualification_report = json.loads(json.dumps(qualification_report))
                self.cohort_contract = json.loads(json.dumps(cohort_contract))
                self.cohort_contract_sha256 = _require_sha256(
                    cohort_contract_sha256,
                    "v0.3 cohort contract",
                )
                self.cohort_id = str(self.cohort_contract.get("cohort_id", ""))
                if not self.cohort_id:
                    raise ActionEffectTrainingError("v0.3 cohort contract lacks an ID")
                self.expected_pre_action_rng = json.loads(json.dumps(expected_pre_action_rng))
                self.expected_first_rollout = json.loads(json.dumps(expected_first_rollout))
                self.initial_rng_identity: dict[str, Any] | None = None
                self.extended_initial_rng_identity: dict[str, Any] | None = None
                self.first_rollout_identity: dict[str, Any] | None = None
                self._first_policy_digest = hashlib.sha256()
                self._first_rollout_steps = 0
                self._first_rollout_closed = False
                self._encoder_snapshot: Any | None = None
                self.encoder_history: list[dict[str, Any]] = []
                self.context_metrics = ContextMetrics()
                super().__init__(**kwargs)
                if not isinstance(self.controller, StageAController):
                    raise ActionEffectTrainingError("v0.3 callback requires StageAController")
                self._last_evaluation_child_actions = (
                    int(self.controller.exam_records[-1]["child_trained_actions"])
                    if self.controller.exam_records
                    else None
                )

            def _on_training_start(self) -> None:
                if (
                    int(self.model.num_timesteps) != self.trained_actions
                    or int(self.model._n_updates) != self.last_updates
                    or self.child_trained_actions != 0
                    or int(self.segment.get("index", -1)) != 0
                ):
                    raise ActionEffectTrainingError(
                        "v0.3 arm did not begin at the exact parent boundary"
                    )
                qualified = capture_qualified_rng_identity(
                    model=self.model,
                    scheduler=self.scheduler,
                    workers=self.workers,
                    phase="post_reset_pre_action_one",
                )
                if qualified != self.expected_pre_action_rng:
                    raise ActionEffectTrainingError(
                        "canonical pre-action RNG differs from qualification"
                    )
                self.initial_rng_identity = qualified
                self.extended_initial_rng_identity = capture_rng_identity(
                    model=self.model,
                    scheduler=self.scheduler,
                    workers=self.workers,
                    phase="post_reset_pre_action_one",
                )
                initial_encoder, self._encoder_snapshot = encoder_metrics(
                    self.model,
                    None,
                )
                if initial_encoder["weight_nonzero_parameters"] != 0:
                    raise ActionEffectTrainingError(
                        "v0.3 context encoder was nonzero before action one"
                    )
                self.encoder_history.append(
                    {
                        "child_trained_actions": 0,
                        "optimizer_updates": int(self.model._n_updates),
                        **initial_encoder,
                    }
                )
                self.manifest["initial_rng_identity"] = qualified
                self.manifest["extended_initial_rng_identity"] = self.extended_initial_rng_identity
                atomic_write_json(
                    self.run_directory / "manifest.json",
                    self.manifest,
                )
                rng_snapshot = snapshot_training_rng_state(
                    model=self.model,
                    scheduler=self.scheduler,
                    workers=self.workers,
                )
                try:
                    initial = self._publish_live_archive(
                        "initial",
                        kind="initial",
                        resume_eligible=False,
                        replace=False,
                    )
                    self.latest_safe_checkpoint = initial
                    self.latest_safe_trained_actions = self.trained_actions
                    self._evaluate_archive(
                        initial,
                        trigger="diagnostic_parent_baseline",
                        decide=False,
                    )
                finally:
                    restore_training_rng_state(
                        rng_snapshot,
                        model=self.model,
                        scheduler=self.scheduler,
                        workers=self.workers,
                    )
                restored = capture_qualified_rng_identity(
                    model=self.model,
                    scheduler=self.scheduler,
                    workers=self.workers,
                    phase="post_reset_pre_action_one",
                )
                if restored != qualified:
                    raise ActionEffectTrainingError("baseline evaluation moved training RNG state")
                self._write_status("training")

            def _on_rollout_start(self) -> None:
                self._guard_storage()
                if int(self.model._n_updates) > self.last_updates:
                    self._process_optimized_boundary()
                # Deliberately no state.mastered stop.  Both arms consume the
                # full fixed budget even when an intermediate exam passes.

            def _record_first_policy_outputs(self) -> None:
                if self._first_rollout_closed:
                    return
                self._first_rollout_steps += 1
                for label in ("actions", "values", "log_probs"):
                    _update_array_digest(
                        self._first_policy_digest,
                        label,
                        self.locals[label],
                    )
                states = self.locals["lstm_states"]
                for branch_name, branch in (
                    ("pi", states.pi),
                    ("vf", states.vf),
                ):
                    _update_array_digest(
                        self._first_policy_digest,
                        f"{branch_name}_hidden",
                        branch[0],
                    )
                    _update_array_digest(
                        self._first_policy_digest,
                        f"{branch_name}_cell",
                        branch[1],
                    )

            def _on_step(self) -> bool:
                self._record_first_policy_outputs()
                actions = np.asarray(self.locals.get("actions")).reshape(-1)
                infos = tuple(self.locals.get("infos", ()))
                if len(actions) != len(infos):
                    raise ActionEffectTrainingError("v0.3 vector actions and infos diverged")
                for action, info in zip(actions, infos, strict=True):
                    lesson_id = str(info.get("lesson_id"))
                    self.context_metrics.observe(
                        action=int(action),
                        lesson_id=lesson_id,
                        changed=bool(info.get("action_effect_visible_changed")),
                        active=bool(info.get("action_effect_context_active")),
                    )

                observations = self.locals.get("new_obs")
                if not isinstance(observations, Mapping):
                    raise ActionEffectTrainingError("v0.3 callback lost its Dict observation")
                original = observations
                self.locals["new_obs"] = np.asarray(original[IMAGE_KEY])
                try:
                    result = bool(super()._on_step())
                finally:
                    self.locals["new_obs"] = original
                return result

            def _on_rollout_end(self) -> None:
                if self._first_rollout_closed:
                    return
                if (
                    self._first_rollout_steps != ROLLOUT_STEPS
                    or int(self.model._n_updates) != PARENT_OPTIMIZER_UPDATES
                    or int(self.model.num_timesteps)
                    != PARENT_LIFETIME_ACTIONS + ROLLOUT_TRANSITIONS
                ):
                    raise ActionEffectTrainingError(
                        "v0.3 first rollout crossed its pre-update boundary"
                    )
                worker_records = [
                    handle.first_rollout.public_dict()
                    for handle in self.workers
                    if handle.first_rollout is not None
                ]
                if (
                    len(worker_records) != WORKERS
                    or sum(int(record["transitions"]) for record in worker_records)
                    != ROLLOUT_TRANSITIONS
                ):
                    raise ActionEffectTrainingError(
                        "v0.3 first-rollout worker evidence is incomplete"
                    )
                post_rng = capture_qualified_rng_identity(
                    model=self.model,
                    scheduler=self.scheduler,
                    workers=self.workers,
                    phase="post_rollout_pre_optimizer",
                )
                identity = {
                    "trajectory_identity": (_normalized_trajectory_identity(worker_records)),
                    "policy_output_sha256": (self._first_policy_digest.hexdigest()),
                    "post_rollout_rng_identity": post_rng,
                    "episode_ledger_normalized_sha256": (
                        _qualification_episode_ledger_sha256(
                            self.run_directory / "episode-starts.jsonl"
                        )
                    ),
                }
                identity["aggregate_sha256"] = _qualification_canonical_sha256(identity)
                if identity != self.expected_first_rollout:
                    raise ActionEffectTrainingError(
                        "canonical first rollout differs from qualification"
                    )
                self.first_rollout_identity = identity
                atomic_write_json(
                    self.run_directory / "first-rollout.json",
                    {
                        "schema_version": FIRST_ROLLOUT_SCHEMA_VERSION,
                        "protocol": PROTOCOL,
                        "arm": self.arm.value,
                        "captured_before_first_optimizer": True,
                        "identity": identity,
                        "qualification_identity_sha256": (identity["aggregate_sha256"]),
                        "checkpoint_reuse_authorized": False,
                    },
                )
                self._first_rollout_closed = True

            def _sidecar(
                self,
                *,
                kind: str,
                checkpoint_sha256: str,
                resume_eligible: bool,
                exam_source: Mapping[str, Any] | None = None,
            ) -> dict[str, Any]:
                if resume_eligible:
                    raise ActionEffectTrainingError("v0.3 Stage A is non-resumable")
                model_state = {
                    "policy_tensor_sha256": (state_digests.policy_tensor_sha256(self.model)),
                    "optimizer_state_sha256": (state_digests.optimizer_state_sha256(self.model)),
                }
                return {
                    "schema_version": SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "created_at": utc_now(),
                    "kind": kind,
                    "arm": self.arm.value,
                    "cohort_id": self.cohort_id,
                    "cohort_contract_sha256": (self.cohort_contract_sha256),
                    "resume_eligible": False,
                    "resume_authorized": False,
                    "promotable": False,
                    "development_checkpoint_reuse_authorized": False,
                    "restart_scope_after_interruption": ("entire_matched_stage_a_root"),
                    "checkpoint_sha256": checkpoint_sha256,
                    "source": dict(self.source),
                    "effective_config": dict(self.effective_config),
                    "parent": self.parent.public_dict(),
                    "qualification": _qualification_public(self.qualification),
                    "transplant": self.transplant,
                    "initial_rng_identity": self.initial_rng_identity,
                    "extended_initial_rng_identity": (self.extended_initial_rng_identity),
                    "first_rollout_identity": self.first_rollout_identity,
                    "model_state": model_state,
                    "curriculum": self.state.public_dict(),
                    "controller": self.controller.public_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "last_completed_allocation": (self.last_completed_allocation),
                    "last_completed_allocation_valid": (self.last_completed_allocation_valid),
                    "context_metrics": self.context_metrics.public_dict(),
                    "encoder_latest": (self.encoder_history[-1] if self.encoder_history else None),
                    "progress": {
                        "collected_actions": int(self.model.num_timesteps),
                        "trained_actions": self.trained_actions,
                        "lifetime_trained_actions": self.trained_actions,
                        "inherited_trained_actions": self.child_start_actions,
                        "child_trained_actions": self.child_trained_actions,
                        "remaining_child_actions": (
                            CHILD_ACTION_BUDGET - self.child_trained_actions
                        ),
                        "optimizer_updates": int(self.model._n_updates),
                    },
                    "segment": dict(self.segment),
                    "exam_source": (dict(exam_source) if exam_source is not None else None),
                }

            def _write_bundle_sidecar(
                self,
                checkpoint: Path,
                *,
                kind: str,
                resume_eligible: bool,
                exam_source: Mapping[str, Any] | None = None,
            ) -> Mapping[str, Any]:
                sidecar = checkpoint.with_suffix(".json")
                atomic_write_json(
                    sidecar,
                    self._sidecar(
                        kind=kind,
                        checkpoint_sha256=file_sha256(checkpoint),
                        resume_eligible=resume_eligible,
                        exam_source=exam_source,
                    ),
                )
                return _write_integrity(checkpoint, sidecar)

            def _record_optimizer_metrics(
                self,
                updates_since_boundary: int,
            ) -> None:
                values = getattr(self.model.logger, "name_to_value", {})
                names = (
                    "entropy_loss",
                    "policy_gradient_loss",
                    "value_loss",
                    "approx_kl",
                    "clip_fraction",
                    "explained_variance",
                    "loss",
                )
                optimizer = {
                    name: (
                        float(values[f"train/{name}"])
                        if values.get(f"train/{name}") is not None
                        else None
                    )
                    for name in names
                }
                measured, self._encoder_snapshot = encoder_metrics(
                    self.model,
                    self._encoder_snapshot,
                )
                encoder_record = {
                    "child_trained_actions": self.child_trained_actions,
                    "optimizer_updates": int(self.model._n_updates),
                    **measured,
                }
                self.encoder_history.append(encoder_record)
                self.latest_optimizer = optimizer | {"encoder": encoder_record}
                append_jsonl(
                    self.run_directory / "optimizer.jsonl",
                    {
                        "timestamp": utc_now(),
                        "collected_actions": int(self.model.num_timesteps),
                        "trained_actions": self.trained_actions,
                        "child_trained_actions": (self.child_trained_actions),
                        "optimizer_updates": int(self.model._n_updates),
                        "updates_since_boundary": updates_since_boundary,
                        **optimizer,
                        "encoder": encoder_record,
                    },
                )

            def _process_optimized_boundary(self) -> None:
                collected = int(self.model.num_timesteps)
                updates = int(self.model._n_updates)
                delta_actions = collected - self.trained_actions
                delta_updates = updates - self.last_updates
                if (
                    delta_actions <= 0
                    or delta_actions % ROLLOUT_TRANSITIONS
                    or delta_updates != delta_actions // ROLLOUT_TRANSITIONS * PPO_EPOCHS
                ):
                    raise ActionEffectTrainingError(
                        "v0.3 observed an incomplete optimizer boundary"
                    )
                self.trained_actions = collected
                self.last_updates = updates
                if not (
                    0 <= self.child_trained_actions <= CHILD_ACTION_BUDGET
                    and optimizer_update_count_valid(
                        self.child_trained_actions,
                        updates,
                    )
                ):
                    raise ActionEffectTrainingError(
                        "v0.3 action or optimizer counter crossed its ceiling"
                    )
                self._record_optimizer_metrics(delta_updates)
                if self.trained_actions >= self.next_evaluation:
                    if self.child_trained_actions % EVALUATION_INTERVAL:
                        raise ActionEffectTrainingError("v0.3 exam is off its frozen boundary")
                    exam = self._publish_live_archive(
                        f"rolling/exam-{self.child_trained_actions:07d}",
                        kind="exam",
                        resume_eligible=False,
                        replace=False,
                    )
                    self.latest_exam_checkpoint = exam
                    self._evaluate_archive(
                        exam,
                        trigger="scheduled",
                        decide=True,
                    )
                    self.next_evaluation = self._next_child_boundary(
                        self.trained_actions,
                        EVALUATION_INTERVAL,
                    )
                if self.child_trained_actions < CHILD_ACTION_BUDGET:
                    latest = self._publish_live_archive(
                        "latest-observed",
                        kind="latest",
                        resume_eligible=False,
                        replace=True,
                    )
                    self.latest_safe_checkpoint = latest
                else:
                    self.latest_safe_checkpoint = self.latest_exam_checkpoint
                self.latest_safe_trained_actions = self.trained_actions
                self._write_status("training")

            def _publish_case_evidence(
                self,
                exam: Path,
                *,
                sidecar_sha256: str,
                cases: Sequence[Mapping[str, Any]],
            ) -> dict[str, Any]:
                path = exam.with_suffix(".cases.json")
                if path.exists() or path.is_symlink():
                    raise ActionEffectTrainingError(
                        f"refusing to overwrite v0.3 exam cases: {path}"
                    )
                diagnostics = _case_diagnostics(cases)
                document = {
                    "schema_version": CASE_EVIDENCE_SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "arm": self.arm.value,
                    "checkpoint": str(exam.relative_to(self.run_directory)),
                    "checkpoint_sha256": file_sha256(exam),
                    "checkpoint_sidecar_sha256": sidecar_sha256,
                    "child_trained_actions": self.child_trained_actions,
                    "case_diagnostics": diagnostics,
                    "records": [dict(case) for case in cases],
                }
                atomic_write_json(path, document)
                return diagnostics | {
                    "path": str(path.relative_to(self.run_directory)),
                    "sha256": file_sha256(path),
                }

            def _evaluate_archive(
                self,
                exam: Path,
                *,
                trigger: str,
                decide: bool,
            ) -> None:
                sidecar = exam.with_suffix(".json")
                exam_sidecar = _read_json(
                    sidecar,
                    "v0.3 exam sidecar",
                )
                _verify_integrity(exam, sidecar)
                archive_before = file_sha256(exam)
                sidecar_before = file_sha256(sidecar)
                graded_model = self._load_exam_model(exam, exam_sidecar)
                allocation = self.scheduler.snapshot()
                allocation_transitions = sum(
                    int(value)
                    for value in allocation.get(
                        "window_transitions",
                        {},
                    ).values()
                )
                allocation_valid = not decide or (
                    allocation_transitions == EVALUATION_INTERVAL
                    and self.scheduler.allocation_within()
                )
                practice_profile = str(allocation.get("profile", "unknown"))
                evaluations: dict[
                    lessons.LessonId,
                    dict[str, Any],
                ] = {}
                evaluation_objects: dict[lessons.LessonId, Any] = {}
                all_cases: list[dict[str, Any]] = []
                for lesson in lessons.LessonId:
                    result, cases = evaluate_architecture_lesson(
                        graded_model,
                        mode=self.arm,
                        lesson=lesson,
                        seeds=self._validation_seeds(lesson),
                        frame_path=(
                            self.run_directory
                            / "frames"
                            / (f"exam-{lesson.value.replace('/', '-')}.png")
                        ),
                        seed_access=self.validation_access,
                    )
                    evaluation_objects[lesson] = result
                    all_cases.extend(cases)
                    public = frozen_u2s._augmented_lesson_evidence(
                        result,
                        cases,
                        penalty_enabled=False,
                    )
                    evaluations[lesson] = public
                    append_jsonl(
                        self.run_directory / "evaluations.jsonl",
                        {
                            "timestamp": utc_now(),
                            "trigger": trigger,
                            "counts_toward_architecture_gate": decide,
                            "arm": self.arm.value,
                            "trained_actions": self.trained_actions,
                            "child_trained_actions": (self.child_trained_actions),
                            "optimizer_updates": int(self.model._n_updates),
                            "checkpoint": str(exam.relative_to(self.run_directory)),
                            "checkpoint_sha256": archive_before,
                            "sidecar_sha256": sidecar_before,
                            "allocation_valid": allocation_valid,
                            "practice_profile": practice_profile,
                            **public,
                        },
                    )
                case_diagnostics = self._publish_case_evidence(
                    exam,
                    sidecar_sha256=sidecar_before,
                    cases=all_cases,
                )
                if (
                    file_sha256(exam) != archive_before
                    or file_sha256(sidecar) != sidecar_before
                    or int(self.model.num_timesteps) != self.trained_actions
                    or int(self.model._n_updates) != self.last_updates
                ):
                    raise ActionEffectTrainingError(
                        "v0.3 evaluation changed model or evidence bytes"
                    )
                if not decide and self.child_trained_actions == 0:
                    expected = frozen_u2.FROZEN_PARENTS[PARENT_CHILD_SEED].inherited_baseline
                    for lesson in (
                        lessons.LessonId.NAVIGATE,
                        lessons.LessonId.VISIBLE_UNLOCK,
                        lessons.LessonId.LOCAL_UNLOCK,
                    ):
                        if int(evaluations[lesson]["successes"]) != int(expected[lesson.value]):
                            raise ActionEffectTrainingError(
                                "v0.3 zero-context baseline did not "
                                "reproduce the confirmed U1 parent"
                            )
                record = {
                    "child_trained_actions": self.child_trained_actions,
                    "practice_profile": practice_profile,
                    "checkpoint": str(exam.relative_to(self.run_directory)),
                    "checkpoint_sha256": archive_before,
                    "sidecar_sha256": sidecar_before,
                    "allocation": allocation,
                    "allocation_transitions": allocation_transitions,
                    "allocation_valid": allocation_valid,
                    "lessons": {lesson.value: evaluations[lesson] for lesson in lessons.LessonId},
                    "case_diagnostics": case_diagnostics,
                }
                if decide:
                    if (
                        self._last_evaluation_child_actions is not None
                        and self.child_trained_actions - self._last_evaluation_child_actions
                        != EVALUATION_INTERVAL
                    ):
                        raise ActionEffectTrainingError(
                            "v0.3 exams are not a contiguous fixed prefix"
                        )
                    decision = self.controller.update_recovery(
                        self.state,
                        evaluation_objects,
                        allocation_valid=allocation_valid,
                    )
                    record["practice_decision"] = decision
                    self.controller.exam_records.append(record)
                    self.controller.practice_decisions.append(
                        {
                            "child_trained_actions": (self.child_trained_actions),
                            "window_profile": practice_profile,
                            "allocation_valid": allocation_valid,
                            "weak_prerequisites": [
                                lesson.value
                                for lesson in lessons.weak_prerequisites(evaluation_objects)
                            ],
                            "decision": decision,
                            "next_profile": self.state.recovery_profile,
                        }
                    )
                    self._last_evaluation_child_actions = self.child_trained_actions
                    self.last_completed_allocation = dict(allocation)
                    self.last_completed_allocation_valid = allocation_valid
                    self.scheduler.start_window()
                self.latest_evaluations = [evaluations[lesson] for lesson in lessons.LessonId]
                self.latest_evaluated_checkpoint = {
                    "path": str(exam.relative_to(self.run_directory)),
                    "checkpoint_sha256": archive_before,
                    "child_trained_actions": (self.child_trained_actions),
                    "trigger": trigger,
                    "counts_toward_architecture_gate": decide,
                    "case_diagnostics": case_diagnostics,
                }
                self.evaluation_history.append(record)
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": ("architecture_exam" if decide else "baseline_evaluation"),
                        "arm": self.arm.value,
                        "cohort_id": self.cohort_id,
                        "child_trained_actions": (self.child_trained_actions),
                        "checkpoint_sha256": archive_before,
                        "allocation_valid": allocation_valid,
                        "practice_profile": practice_profile,
                        "practice_decision": record.get("practice_decision"),
                        "case_diagnostics": case_diagnostics,
                        "checks": (frozen_u2s.exam_stability_checks(record)),
                    },
                )
                self.frame_revision += 1

            def _write_status(self, phase: str) -> None:
                elapsed = max(0.0, time.monotonic() - self.wall_start)
                collected = int(self.model.num_timesteps)
                records = list(self.controller.exam_records)
                grade = (
                    v03.grade_terminal(self.arm, records).public_dict()
                    if len(records) == EXAM_COUNT
                    else None
                )
                atomic_write_json(
                    self.run_directory / "status.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "protocol": PROTOCOL,
                        "phase": phase,
                        "arm": self.arm.value,
                        "cohort_id": self.cohort_id,
                        "started_at": self.started_at,
                        "updated_at": utc_now(),
                        "elapsed_seconds": elapsed,
                        "actions_per_second": (
                            (collected - self.segment_start_actions) / elapsed if elapsed else 0.0
                        ),
                        "child_collected_actions": (collected - PARENT_LIFETIME_ACTIONS),
                        "child_trained_actions": (self.child_trained_actions),
                        "lifetime_collected_actions": collected,
                        "lifetime_trained_actions": self.trained_actions,
                        "remaining_action_budget": (
                            CHILD_ACTION_BUDGET - self.child_trained_actions
                        ),
                        "action_cap": CHILD_ACTION_BUDGET,
                        "optimizer_updates": int(self.model._n_updates),
                        "parent_checkpoint_sha256": (self.parent.checkpoint_sha256),
                        "qualification_sha256": (self.qualification.report_sha256),
                        "cohort_contract_sha256": (self.cohort_contract_sha256),
                        "source": dict(self.source),
                        "resume_authorized": False,
                        "promotable": False,
                        "cohort_outcome_if_interrupted": ("operationally_incomplete"),
                        "replacement_requires_both_fresh_arms": True,
                        "initial_rng_identity_sha256": (
                            self.initial_rng_identity["aggregate_sha256"]
                            if self.initial_rng_identity
                            else None
                        ),
                        "first_rollout_identity_sha256": (
                            self.first_rollout_identity["aggregate_sha256"]
                            if self.first_rollout_identity
                            else None
                        ),
                        "first_rollout_verified": (self.first_rollout_identity is not None),
                        "exam_count": len(records),
                        "exams_completed": len(records),
                        "evaluations": records,
                        "latest_evaluation": (records[-1] if records else None),
                        "terminal_grade": grade,
                        "terminal_eligible": (grade["eligible"] if grade is not None else None),
                        "next_evaluation": self.next_evaluation,
                        "latest_observed_checkpoint": (
                            str(self.latest_safe_checkpoint.relative_to(self.run_directory))
                            if self.latest_safe_checkpoint is not None
                            else None
                        ),
                        "latest_observed_checkpoint_sha256": (
                            file_sha256(self.latest_safe_checkpoint)
                            if self.latest_safe_checkpoint is not None
                            and self.latest_safe_checkpoint.is_file()
                            else None
                        ),
                        "latest_safe_checkpoint": None,
                        "latest_safe_checkpoint_sha256": None,
                        "latest_evaluated_checkpoint": (self.latest_evaluated_checkpoint),
                        "recent_behavior": self._behavior_status(),
                        "context_metrics": (self.context_metrics.public_dict()),
                        "encoder_latest": (
                            self.encoder_history[-1] if self.encoder_history else None
                        ),
                        "latest_optimizer": self.latest_optimizer,
                        "storage": self._storage_status(),
                        "segment": dict(self.segment),
                        "report_sha256": (
                            file_sha256(self.run_directory / "report.json")
                            if phase == "completed"
                            and (self.run_directory / "report.json").is_file()
                            else None
                        ),
                    },
                )

            def _pending_optimizer_delta(self) -> tuple[int, int, str]:
                delta_actions = int(self.model.num_timesteps) - self.trained_actions
                delta_updates = int(self.model._n_updates) - self.last_updates
                complete = (
                    delta_actions > 0
                    and delta_actions % ROLLOUT_TRANSITIONS == 0
                    and delta_updates == delta_actions // ROLLOUT_TRANSITIONS * PPO_EPOCHS
                )
                if delta_actions == 0 and delta_updates == 0:
                    classification = "none"
                elif complete:
                    classification = "complete"
                elif delta_actions >= 0 and delta_updates >= 0:
                    classification = "partial"
                else:
                    classification = "counter_regression"
                return delta_actions, delta_updates, classification

            def finalize_failure(self, phase: str) -> None:
                pending_actions, pending_updates, classification = self._pending_optimizer_delta()
                if classification == "complete":
                    self._process_optimized_boundary()
                    classification = "complete_recovered"
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": f"run_{phase}",
                        "arm": self.arm.value,
                        "pending_optimizer_boundary": classification,
                        "pending_actions": pending_actions,
                        "pending_optimizer_updates": pending_updates,
                        "cohort_outcome": "operationally_incomplete",
                        "resume_authorized": False,
                    },
                )
                self._write_status(phase)

            def _terminal_episode_evidence(self) -> dict[str, Any]:
                workers = [handle.episode.evidence_state() for handle in self.workers]
                if len(workers) != WORKERS or not all(
                    worker.get("active") is True for worker in workers
                ):
                    raise ActionEffectTrainingError(
                        "v0.3 terminal active-worker evidence is incomplete"
                    )
                ledger = self.run_directory / "episode-starts.jsonl"
                return {
                    "ledger": str(ledger.relative_to(self.run_directory)),
                    "ledger_sha256": file_sha256(ledger),
                    "terminal_active_workers": workers,
                    "terminal_active_workers_sha256": (_canonical_sha256(workers)),
                }

            def _case_inventory(self) -> dict[str, Any]:
                files: list[dict[str, Any]] = []
                total_records = 0
                for path in sorted(
                    (self.run_directory / "checkpoints" / "rolling").glob("exam-*.cases.json")
                ):
                    document = _read_json(path, "v0.3 exam cases")
                    records = document.get("records")
                    if (
                        document.get("protocol") != PROTOCOL
                        or document.get("arm") != self.arm.value
                        or not isinstance(records, list)
                        or len(records) != len(lessons.LessonId) * EVALUATION_SEED_COUNT
                    ):
                        raise ActionEffectTrainingError("v0.3 case evidence inventory changed")
                    total_records += len(records)
                    files.append(
                        {
                            "path": str(path.relative_to(self.run_directory)),
                            "sha256": file_sha256(path),
                            "records": len(records),
                        }
                    )
                if (
                    len(files) != EXAM_COUNT
                    or total_records != EXAM_COUNT * len(lessons.LessonId) * EVALUATION_SEED_COUNT
                ):
                    raise ActionEffectTrainingError("v0.3 terminal case evidence is incomplete")
                return {
                    "files": files,
                    "file_count": len(files),
                    "record_count": total_records,
                    "inventory_sha256": _canonical_sha256(files),
                }

            def finalize(self, _phase: str) -> None:
                if int(self.model._n_updates) > self.last_updates:
                    self._process_optimized_boundary()
                if (
                    int(self.model.num_timesteps) != self.trained_actions
                    or self.child_trained_actions != CHILD_ACTION_BUDGET
                    or self.trained_actions != PARENT_LIFETIME_ACTIONS + CHILD_ACTION_BUDGET
                    or not optimizer_update_count_valid(
                        CHILD_ACTION_BUDGET,
                        int(self.model._n_updates),
                    )
                    or len(self.controller.exam_records) != EXAM_COUNT
                    or self.latest_exam_checkpoint is None
                    or self.first_rollout_identity is None
                ):
                    raise ActionEffectTrainingError(
                        "v0.3 arm did not finish its exact trained ceiling"
                    )
                grade = v03.grade_terminal(
                    self.arm,
                    self.controller.exam_records,
                )
                terminal = self._copy_exam_artifact(
                    self.latest_exam_checkpoint,
                    "terminal",
                    kind="terminal",
                    resume_eligible=False,
                    replace=False,
                    exam_integrity=_verify_integrity(
                        self.latest_exam_checkpoint,
                        self.latest_exam_checkpoint.with_suffix(".json"),
                    ),
                )
                if self.source_guard is not None:
                    self.source_guard()
                storage = self.storage_guard(0) if self.storage_guard is not None else None
                terminal_sidecar = _read_json(
                    terminal.with_suffix(".json"),
                    "v0.3 terminal checkpoint sidecar",
                )
                case_evidence = self._case_inventory()
                context_public = self.context_metrics.public_dict()
                context_summary = {
                    **context_public,
                    "unchanged_rate": (
                        context_public["unchanged"] / context_public["transitions"]
                        if context_public["transitions"]
                        else 0.0
                    ),
                }
                initial_encoder = self.encoder_history[0]
                terminal_encoder = self.encoder_history[-1]
                first_nonzero = next(
                    (
                        int(record["child_trained_actions"])
                        for record in self.encoder_history[1:]
                        if int(record["weight_nonzero_parameters"]) > 0
                    ),
                    None,
                )
                context_encoder = {
                    "initial_weight_norm": initial_encoder["weight_l2_norm"],
                    "initial_nonzero_parameters": initial_encoder["weight_nonzero_parameters"],
                    "residual_zero_before_action_one": True,
                    "terminal_weight_norm": terminal_encoder["weight_l2_norm"],
                    "terminal_update_norm": terminal_encoder["update_l2_norm"],
                    "terminal_nonzero_parameters": terminal_encoder["weight_nonzero_parameters"],
                    "first_nonzero_child_actions": first_nonzero,
                }
                first_rollout = {
                    "pre_action_rng_sha256": (
                        _qualification_canonical_sha256(self.initial_rng_identity)
                    ),
                    "trajectory_sha256": (
                        _qualification_canonical_sha256(
                            self.first_rollout_identity["trajectory_identity"]
                        )
                    ),
                    "policy_output_sha256": (self.first_rollout_identity["policy_output_sha256"]),
                    "post_rollout_rng_sha256": (
                        _qualification_canonical_sha256(
                            self.first_rollout_identity["post_rollout_rng_identity"]
                        )
                    ),
                    "transitions": ROLLOUT_TRANSITIONS,
                    "first_optimizer_boundary": ROLLOUT_TRANSITIONS,
                    "pre_update_identity_verified": True,
                    "divergence_before_first_optimizer": False,
                    "first_divergence": (
                        "after_first_optimizer" if first_nonzero is not None else "not_observed"
                    ),
                }
                report = {
                    "schema_version": REPORT_SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "cohort_id": self.cohort_id,
                    "cohort_contract_sha256": (self.cohort_contract_sha256),
                    "arm": self.arm.value,
                    "verdict": (
                        "architecture_candidate_passed" if grade.eligible else "arm_failed"
                    ),
                    "development_only": True,
                    "architecture_definition_eligible": (
                        self.arm is ActionEffectMode.ACTION_EFFECT and grade.eligible
                    ),
                    "sham_calibration_only": (self.arm is ActionEffectMode.SHAM),
                    "successor_checkpoint_authorized": False,
                    "development_checkpoint_reuse_authorized": False,
                    "resume_authorized": False,
                    "promotable": False,
                    "u3_authorized": False,
                    "grade": grade.public_dict(),
                    "terminal_eligible": grade.eligible,
                    "child_trained_actions": (self.child_trained_actions),
                    "lifetime_trained_actions": self.trained_actions,
                    "optimizer_updates": int(self.model._n_updates),
                    "exam_count": len(self.controller.exam_records),
                    "case_count": case_evidence["record_count"],
                    "qualification_sha256": (self.qualification.report_sha256),
                    "progress": {
                        "child_trained_actions": (self.child_trained_actions),
                        "lifetime_trained_actions": self.trained_actions,
                        "optimizer_updates": int(self.model._n_updates),
                        "exam_count": len(self.controller.exam_records),
                    },
                    "source": dict(self.source),
                    "source_clean_at_closeout": True,
                    "parent": self.parent.public_dict(),
                    "qualification": _qualification_public(self.qualification),
                    "effective_config": dict(self.effective_config),
                    "transplant": self.transplant,
                    "initial_rng_identity": self.initial_rng_identity,
                    "extended_initial_rng_identity": (self.extended_initial_rng_identity),
                    "first_rollout_identity": (self.first_rollout_identity),
                    "first_rollout": first_rollout,
                    "controller": self.controller.public_dict(),
                    "exam_records": self.controller.exam_records,
                    "context_metrics": (self.context_metrics.public_dict()),
                    "context_summary": context_summary,
                    "context_encoder": context_encoder,
                    "encoder_history": self.encoder_history,
                    "terminal_encoder": (self.encoder_history[-1]),
                    "training_episode_evidence": (self._terminal_episode_evidence()),
                    "case_evidence": case_evidence,
                    "protected_seed_access": {
                        "confirmation_or_final_partitions_opened": False,
                        "bound_by_qualification": True,
                    },
                    "storage": storage,
                    "terminal_checkpoint": {
                        "path": str(terminal.relative_to(self.run_directory)),
                        "sha256": file_sha256(terminal),
                        "sidecar_sha256": file_sha256(terminal.with_suffix(".json")),
                        "integrity_sha256": file_sha256(_integrity_path(terminal)),
                        "model_state": terminal_sidecar["model_state"],
                        "promotable": False,
                    },
                    "completed_at": utc_now(),
                }
                report_path = self.run_directory / "report.json"
                atomic_write_json(report_path, report)
                report_integrity = {
                    "schema_version": REPORT_SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "cohort_id": self.cohort_id,
                    "arm": self.arm.value,
                    "cohort_contract_sha256": (self.cohort_contract_sha256),
                    "report": report_path.name,
                    "report_sha256": file_sha256(report_path),
                    "terminal_checkpoint_sha256": file_sha256(terminal),
                    "terminal_sidecar_sha256": file_sha256(terminal.with_suffix(".json")),
                    "terminal_integrity_sha256": file_sha256(_integrity_path(terminal)),
                }
                atomic_write_json(
                    self.run_directory / "report.integrity.json",
                    report_integrity,
                )
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": "run_completed",
                        "arm": self.arm.value,
                        "verdict": report["verdict"],
                        "report_sha256": file_sha256(report_path),
                        "resume_authorized": False,
                        "promotable": False,
                    },
                )
                self._write_status("completed")

        return ActionEffectCallback


def _verify_terminal_exam_evidence(
    run: Path,
    *,
    arm: ActionEffectMode,
    report: Mapping[str, Any],
    exam_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Deep-authenticate every scheduled exam and immutable case transcript."""

    expected_boundaries = [index * EVALUATION_INTERVAL for index in range(1, EXAM_COUNT + 1)]
    boundaries = [int(exam.get("child_trained_actions", -1)) for exam in exam_records]
    if (
        len(exam_records) != EXAM_COUNT
        or boundaries != expected_boundaries
        or report.get("exam_records") != list(exam_records)
    ):
        raise ActionEffectTrainingError("v0.3 terminal exam inventory is incomplete")

    verified_case_files: list[dict[str, Any]] = []
    verified_exam_bundles: list[dict[str, Any]] = []
    total_cases = 0
    for exam, boundary in zip(
        exam_records,
        expected_boundaries,
        strict=True,
    ):
        relative_checkpoint = exam.get("checkpoint")
        if not isinstance(relative_checkpoint, str):
            raise ActionEffectTrainingError("v0.3 exam checkpoint identity is missing")
        checkpoint = (run / relative_checkpoint).resolve()
        try:
            checkpoint.relative_to(run)
        except ValueError as error:
            raise ActionEffectTrainingError("v0.3 exam checkpoint escapes its arm") from error
        sidecar = checkpoint.with_suffix(".json")
        integrity_path = _integrity_path(checkpoint)
        _verify_integrity(checkpoint, sidecar)
        sidecar_value = _read_json(sidecar, "v0.3 exam sidecar")
        expected_updates = PARENT_OPTIMIZER_UPDATES + boundary // ROLLOUT_TRANSITIONS * PPO_EPOCHS
        if (
            file_sha256(checkpoint) != exam.get("checkpoint_sha256")
            or file_sha256(sidecar) != exam.get("sidecar_sha256")
            or sidecar_value.get("kind") != "exam"
            or sidecar_value.get("arm") != arm.value
            or sidecar_value.get("resume_eligible") is not False
            or sidecar_value.get("resume_authorized") is not False
            or sidecar_value.get("promotable") is not False
            or sidecar_value.get("development_checkpoint_reuse_authorized") is not False
            or sidecar_value.get("checkpoint_sha256") != exam.get("checkpoint_sha256")
            or sidecar_value.get("source") != report.get("source")
            or sidecar_value.get("cohort_id") != report.get("cohort_id")
            or sidecar_value.get("cohort_contract_sha256") != report.get("cohort_contract_sha256")
            or sidecar_value.get("qualification") != report.get("qualification")
            or sidecar_value.get("progress", {}).get("child_trained_actions") != boundary
            or sidecar_value.get("progress", {}).get("lifetime_trained_actions")
            != PARENT_LIFETIME_ACTIONS + boundary
            or sidecar_value.get("progress", {}).get("optimizer_updates") != expected_updates
        ):
            raise ActionEffectTrainingError("v0.3 exam checkpoint bundle changed")

        diagnostics = exam.get("case_diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise ActionEffectTrainingError("v0.3 exam case diagnostics are missing")
        relative_cases = diagnostics.get("path")
        case_sha256 = diagnostics.get("sha256")
        if not isinstance(relative_cases, str) or not isinstance(
            case_sha256,
            str,
        ):
            raise ActionEffectTrainingError("v0.3 exam case-file identity is missing")
        cases_path = (run / relative_cases).resolve()
        try:
            cases_path.relative_to(run)
        except ValueError as error:
            raise ActionEffectTrainingError("v0.3 exam case file escapes its arm") from error
        if (
            cases_path != checkpoint.with_suffix(".cases.json")
            or cases_path.is_symlink()
            or not cases_path.is_file()
        ):
            raise ActionEffectTrainingError("v0.3 exam case evidence is missing or unsafe")
        document = _read_json(cases_path, "v0.3 exam case evidence")
        cases = document.get("records")
        if (
            set(document)
            != {
                "schema_version",
                "protocol",
                "arm",
                "checkpoint",
                "checkpoint_sha256",
                "checkpoint_sidecar_sha256",
                "child_trained_actions",
                "case_diagnostics",
                "records",
            }
            or file_sha256(cases_path) != case_sha256
            or document.get("schema_version") != CASE_EVIDENCE_SCHEMA_VERSION
            or document.get("protocol") != PROTOCOL
            or document.get("arm") != arm.value
            or document.get("checkpoint") != relative_checkpoint
            or document.get("checkpoint_sha256") != exam.get("checkpoint_sha256")
            or document.get("checkpoint_sidecar_sha256") != exam.get("sidecar_sha256")
            or document.get("child_trained_actions") != boundary
            or not isinstance(cases, list)
            or len(cases) != len(lessons.LessonId) * EVALUATION_SEED_COUNT
            or not frozen_u2s._numbers_are_finite(cases)
        ):
            raise ActionEffectTrainingError("v0.3 exam case evidence changed")
        try:
            # The v0.3 arms retain the original U2 control reward: no
            # no-effect penalty.  Reusing the frozen control validator gives
            # us its exact seed panels, record schema, aggregate
            # recomputation, visible-effect checks, and duplicate detection.
            frozen_u2s._verify_exam_case_records(
                exam,
                cases,
                arm=frozen_u2s.ArmName.CONTROL,
            )
            recomputed_diagnostics = frozen_u2s._case_diagnostics_from_records(cases)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ActionEffectTrainingError(
                "v0.3 exam aggregate disagrees with immutable cases"
            ) from error
        expected_diagnostics = {
            **recomputed_diagnostics,
            "path": relative_cases,
            "sha256": case_sha256,
        }
        if (
            document.get("case_diagnostics") != recomputed_diagnostics
            or dict(diagnostics) != expected_diagnostics
        ):
            raise ActionEffectTrainingError("v0.3 case diagnostics disagree with immutable cases")

        total_cases += len(cases)
        verified_case_files.append(
            {
                "path": relative_cases,
                "sha256": case_sha256,
                "records": len(cases),
            }
        )
        verified_exam_bundles.append(
            {
                "path": relative_checkpoint,
                "checkpoint_sha256": file_sha256(checkpoint),
                "sidecar_sha256": file_sha256(sidecar),
                "integrity_sha256": file_sha256(integrity_path),
                "child_trained_actions": boundary,
            }
        )

    expected_cases = EXAM_COUNT * len(lessons.LessonId) * EVALUATION_SEED_COUNT
    inventory = report.get("case_evidence")
    expected_inventory = {
        "files": verified_case_files,
        "file_count": EXAM_COUNT,
        "record_count": expected_cases,
        "inventory_sha256": _canonical_sha256(verified_case_files),
    }
    if (
        total_cases != expected_cases
        or inventory != expected_inventory
        or report.get("case_count") != expected_cases
    ):
        raise ActionEffectTrainingError("v0.3 terminal case inventory changed")
    return {
        **expected_inventory,
        "exam_bundles": verified_exam_bundles,
        "exam_bundles_sha256": _canonical_sha256(verified_exam_bundles),
    }


def verify_arm_terminal_report(
    run_directory: Path,
    *,
    expected_arm: ActionEffectMode | str,
    expected_source_commit: str | None = None,
    expected_cohort_id: str | None = None,
    expected_contract_sha256: str | None = None,
    expected_qualification_sha256: str | None = None,
) -> dict[str, Any]:
    """Authenticate one terminal arm report without making it promotable."""

    run = run_directory.expanduser().resolve()
    arm = ActionEffectMode(expected_arm)
    report_path = run / "report.json"
    integrity_path = run / "report.integrity.json"
    status_path = run / "status.json"
    status = _read_json(status_path, "v0.3 arm status")
    report = _read_json(report_path, "v0.3 arm report")
    integrity = _read_json(
        integrity_path,
        "v0.3 arm report integrity",
    )
    terminal_record = report.get("terminal_checkpoint")
    progress = report.get("progress")
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("protocol") != PROTOCOL
        or report.get("arm") != arm.value
        or status.get("phase") != "completed"
        or status.get("protocol") != PROTOCOL
        or status.get("arm") != arm.value
        or status.get("report_sha256") != file_sha256(report_path)
        or int(status.get("child_trained_actions", -1)) != CHILD_ACTION_BUDGET
        or int(status.get("remaining_action_budget", -1)) != 0
        or int(status.get("optimizer_updates", -1)) != 3_584
        or int(status.get("exam_count", -1)) != EXAM_COUNT
        or report.get("development_only") is not True
        or report.get("successor_checkpoint_authorized") is not False
        or report.get("development_checkpoint_reuse_authorized") is not False
        or report.get("resume_authorized") is not False
        or report.get("promotable") is not False
        or report.get("u3_authorized") is not False
        or not isinstance(progress, Mapping)
        or int(progress.get("child_trained_actions", -1)) != CHILD_ACTION_BUDGET
        or int(progress.get("lifetime_trained_actions", -1))
        != PARENT_LIFETIME_ACTIONS + CHILD_ACTION_BUDGET
        or not optimizer_update_count_valid(
            CHILD_ACTION_BUDGET,
            int(progress.get("optimizer_updates", -1)),
        )
        or int(progress.get("exam_count", -1)) != EXAM_COUNT
        or not isinstance(terminal_record, Mapping)
        or terminal_record.get("promotable") is not False
        or (
            expected_cohort_id is not None
            and (
                report.get("cohort_id") != expected_cohort_id
                or status.get("cohort_id") != expected_cohort_id
                or integrity.get("cohort_id") != expected_cohort_id
            )
        )
        or (
            expected_contract_sha256 is not None
            and (
                report.get("cohort_contract_sha256") != expected_contract_sha256
                or status.get("cohort_contract_sha256") != expected_contract_sha256
                or integrity.get("cohort_contract_sha256") != expected_contract_sha256
            )
        )
        or (
            expected_qualification_sha256 is not None
            and (
                report.get("qualification_sha256") != expected_qualification_sha256
                or status.get("qualification_sha256") != expected_qualification_sha256
            )
        )
    ):
        raise ActionEffectTrainingError("v0.3 terminal arm report contract changed")
    source = report.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("dirty") is not False
        or (expected_source_commit is not None and source.get("commit") != expected_source_commit)
    ):
        raise ActionEffectTrainingError("v0.3 terminal arm source identity changed")
    terminal = (run / str(terminal_record.get("path", ""))).resolve()
    try:
        terminal.relative_to(run)
    except ValueError as error:
        raise ActionEffectTrainingError("v0.3 terminal checkpoint escapes its arm") from error
    sidecar = terminal.with_suffix(".json")
    _verify_integrity(terminal, sidecar)
    sidecar_value = _read_json(sidecar, "v0.3 terminal sidecar")
    if (
        terminal_record.get("sha256") != file_sha256(terminal)
        or terminal_record.get("sidecar_sha256") != file_sha256(sidecar)
        or terminal_record.get("integrity_sha256") != file_sha256(_integrity_path(terminal))
        or sidecar_value.get("kind") != "terminal"
        or sidecar_value.get("resume_eligible") is not False
        or sidecar_value.get("promotable") is not False
        or sidecar_value.get("arm") != arm.value
        or sidecar_value.get("progress", {}).get("child_trained_actions") != CHILD_ACTION_BUDGET
    ):
        raise ActionEffectTrainingError("v0.3 terminal checkpoint binding changed")
    if (
        integrity.get("schema_version") != REPORT_SCHEMA_VERSION
        or integrity.get("protocol") != PROTOCOL
        or integrity.get("arm") != arm.value
        or integrity.get("report") != report_path.name
        or integrity.get("report_sha256") != file_sha256(report_path)
        or integrity.get("terminal_checkpoint_sha256") != file_sha256(terminal)
        or integrity.get("terminal_sidecar_sha256") != file_sha256(sidecar)
        or integrity.get("terminal_integrity_sha256") != file_sha256(_integrity_path(terminal))
    ):
        raise ActionEffectTrainingError("v0.3 arm report integrity changed")
    controller = report.get("controller")
    records = controller.get("exam_records") if isinstance(controller, Mapping) else None
    if not isinstance(records, list) or not all(isinstance(record, Mapping) for record in records):
        raise ActionEffectTrainingError("v0.3 arm report lacks its exam records")
    case_evidence = _verify_terminal_exam_evidence(
        run,
        arm=arm,
        report=report,
        exam_records=records,
    )
    grade = v03.grade_terminal(arm, records)
    if grade.public_dict() != report.get("grade"):
        raise ActionEffectTrainingError("v0.3 arm grade does not recompute")
    return {
        "arm": arm.value,
        "cohort_id": report.get("cohort_id"),
        "cohort_contract_sha256": report.get("cohort_contract_sha256"),
        "qualification_sha256": report.get("qualification_sha256"),
        "status_sha256": file_sha256(status_path),
        "report": str(report_path),
        "report_sha256": file_sha256(report_path),
        "report_integrity_sha256": file_sha256(integrity_path),
        "terminal_checkpoint_sha256": file_sha256(terminal),
        "eligible": grade.eligible,
        "grade": grade.public_dict(),
        "exam_records": records,
        "first_rollout_identity": report.get("first_rollout_identity"),
        "context_metrics": report.get("context_metrics"),
        "terminal_encoder": report.get("terminal_encoder"),
        "case_count": case_evidence["record_count"],
        "case_evidence_sha256": case_evidence["inventory_sha256"],
        "case_evidence_files": case_evidence["files"],
        "exam_bundles_sha256": case_evidence["exam_bundles_sha256"],
        "exam_bundles": case_evidence["exam_bundles"],
    }


def _best_effort(label: str, operation: Callable[[], Any]) -> None:
    try:
        operation()
    except BaseException as error:
        print(
            f"Warning: {label} failed during v0.3 finalization: {error}",
            flush=True,
        )


def verify_cohort_contract(
    *,
    path: Path,
    arm: ActionEffectMode,
    run_directory: Path,
    media_directory: Path,
    expected_cohort_root: Path,
    expected_media_root: Path,
    source: Mapping[str, Any],
    qualification: Any,
) -> tuple[dict[str, Any], str, Path, Path]:
    """Bind one trainer invocation to the launcher-owned immutable contract."""

    contract_path = path.expanduser().resolve()
    canonical_cohort_root = expected_cohort_root.expanduser().resolve()
    canonical_media_root = expected_media_root.expanduser().resolve()
    canonical_contract_path = canonical_cohort_root / "cohort-contract.json"
    if (
        contract_path != canonical_contract_path
        or contract_path.is_symlink()
        or not contract_path.is_file()
    ):
        raise ActionEffectTrainingError("v0.3 cohort contract is missing or unsafe")
    contract = _read_json(contract_path, "v0.3 cohort contract")
    digest = file_sha256(contract_path)
    roots = contract.get("roots")
    arms = contract.get("arms")
    matched = contract.get("matched_design")
    parent = contract.get("parent")
    qualification_public = _qualification_public(qualification)
    qualification_binding = contract.get("qualification")
    contract_source = contract.get("source")
    expected_arm_directories = [
        (
            ActionEffectMode.SHAM.value,
            ActionEffectMode.SHAM.value,
            ActionEffectMode.SHAM.value,
        ),
        (
            ActionEffectMode.ACTION_EFFECT.value,
            ActionEffectMode.ACTION_EFFECT.value,
            ActionEffectMode.ACTION_EFFECT.value,
        ),
    ]
    if (
        contract.get("schema_version") != 1
        or contract.get("protocol") != PROTOCOL
        or contract.get("cohort_id") != "v0.3-action-effect-stage-a-r2-20260724"
        or not isinstance(contract_source, Mapping)
        or contract_source.get("commit") != source.get("commit")
        or contract_source.get("dirty") is not False
        or source.get("dirty") is not False
        or not isinstance(roots, Mapping)
        or not isinstance(arms, list)
        or [
            (
                value.get("id"),
                value.get("directory"),
                value.get("media_directory"),
            )
            for value in arms
            if isinstance(value, Mapping)
        ]
        != expected_arm_directories
        or not isinstance(matched, Mapping)
        or not isinstance(parent, Mapping)
        or not isinstance(qualification_binding, Mapping)
        or qualification_binding.get("report_sha256") != qualification_public.get("report_sha256")
        or qualification_binding.get("tag_object") != qualification_public.get("tag_object")
        or parent.get("checkpoint_sha256") != v03.PARENT_CHECKPOINT_SHA256
        or parent.get("policy_tensor_sha256") != v03.PARENT_POLICY_TENSOR_SHA256
        or parent.get("optimizer_state_sha256") != v03.PARENT_OPTIMIZER_STATE_SHA256
        or matched.get("arm_order") != [value.value for value in v03.ARM_ORDER]
        or matched.get("action_cap_per_arm") != CHILD_ACTION_BUDGET
        or matched.get("evaluation_every") != EVALUATION_INTERVAL
        or matched.get("fresh_only") is not True
        or matched.get("resumable") is not False
        or matched.get("checkpoint_promotable") is not False
    ):
        raise ActionEffectTrainingError("v0.3 cohort contract changed")
    cohort_root = Path(str(roots.get("cohort", ""))).expanduser().resolve()
    media_root = Path(str(roots.get("media", ""))).expanduser().resolve()
    run = run_directory.expanduser().resolve()
    media = media_directory.expanduser().resolve()
    selected = next(
        (value for value in arms if isinstance(value, Mapping) and value.get("id") == arm.value),
        None,
    )
    if (
        cohort_root != canonical_cohort_root
        or media_root != canonical_media_root
        or not isinstance(selected, Mapping)
        or run != (cohort_root / str(selected.get("directory", ""))).resolve()
        or media != (media_root / str(selected.get("media_directory", ""))).resolve()
        or run.is_symlink()
        or not run.is_dir()
        or media.is_symlink()
        or not media.is_dir()
        or any(run.iterdir())
        or any(media.iterdir())
    ):
        raise ActionEffectTrainingError("v0.3 arm paths are not the launcher-owned fresh targets")
    return contract, digest, cohort_root, media_root


def train_arm(args: argparse.Namespace) -> None:
    """Run one canonical arm after reauthenticating the qualified release."""

    arm = ActionEffectMode(args.arm)
    repository = Path(__file__).resolve().parents[2]
    source = git_snapshot(repository)
    if (
        source.get("dirty") is not False
        or not isinstance(source.get("commit"), str)
        or not source["commit"]
    ):
        raise SystemExit("v0.3 Stage A requires one clean source commit")
    if (
        not v03.PARENT_CHECKPOINT.is_file()
        or v03.PARENT_CHECKPOINT.is_symlink()
        or file_sha256(v03.PARENT_CHECKPOINT) != v03.PARENT_CHECKPOINT_SHA256
    ):
        raise SystemExit("v0.3 confirmed U1 parent changed")

    from dungeon_apprentice.v03_action_effect_qualify import (
        CANONICAL_COHORT_ROOT,
        CANONICAL_MEDIA_ROOT,
        verify_action_effect_qualification,
    )

    qualification_path = Path(args.qualification_report)
    qualification = verify_action_effect_qualification(
        qualification_path,
        expected_source_commit=str(source["commit"]),
        repository=repository,
    )
    report = _qualification_report(qualification)
    seed_access = qualification.seed_access()
    forbidden_layout_hashes = qualification.forbidden_layout_hashes()
    parent = frozen_u2.verify_parent(
        v03.PARENT_CHECKPOINT,
        frozen_u2.CANONICAL_U1_CONFIRMATION,
        child_seed=PARENT_CHILD_SEED,
    )
    if (
        parent.u1_child_seed != v03.PARENT_U1_SEED
        or parent.checkpoint_sha256 != v03.PARENT_CHECKPOINT_SHA256
        or parent.trained_timesteps != PARENT_LIFETIME_ACTIONS
        or parent.n_updates != PARENT_OPTIMIZER_UPDATES
    ):
        raise SystemExit("v0.3 parent identity changed")

    run_directory = args.run_dir.expanduser().resolve()
    media_directory = args.media_dir.expanduser().resolve()
    (
        cohort_contract,
        cohort_contract_sha256,
        cohort_root,
        media_root,
    ) = verify_cohort_contract(
        path=args.cohort_contract,
        arm=arm,
        run_directory=run_directory,
        media_directory=media_directory,
        expected_cohort_root=CANONICAL_COHORT_ROOT,
        expected_media_root=CANONICAL_MEDIA_ROOT,
        source=source,
        qualification=qualification,
    )
    minimum_free_bytes = int(MINIMUM_FREE_GIB * 1024**3)
    ensure_disk_space(run_directory, minimum_free_bytes)

    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            VecTransposeImage,
        )
    except ImportError as error:
        raise SystemExit('Install training dependencies with: pip install -e ".[train]"') from error

    state = lessons.CurriculumState()
    controller = StageAController()
    scheduler = lessons.TransitionDeficitScheduler(
        state,
        seed=ALGORITHM_SEED + 90_000,
    )
    segment = {
        "id": uuid.uuid4().hex,
        "index": 0,
        "started_at": utc_now(),
        "algorithm_seed": ALGORITHM_SEED,
        "worker_streams": list(WORKER_STREAMS),
        "start_child_trained_actions": 0,
        "remaining_child_actions": CHILD_ACTION_BUDGET,
        "resume_authorized": False,
        "resume_checkpoint": None,
    }
    config = v03.effective_config(arm)

    def source_guard() -> None:
        current = git_snapshot(repository)
        if current.get("dirty") is not False or current.get("commit") != source["commit"]:
            raise ActionEffectTrainingError("v0.3 source became dirty or changed")

    def storage_guard(
        anticipated_lineage_bytes: int = 0,
    ) -> Mapping[str, Any]:
        return audit_stage_a_storage(
            storage_root=cohort_root.parent,
            cohort_directory=cohort_root,
            lineage_directory=run_directory,
            media_directory=media_root,
            protected_paths=(
                Path(parent.checkpoint),
                Path(parent.confirmation_report),
                Path(qualification.report),
            ),
            anticipated_lineage_bytes=int(anticipated_lineage_bytes),
        )

    sampler = frozen_u2r.preflight_u2r_training_layout_sampler(
        forbidden_layout_hashes,
        seed_access=seed_access,
        worker_streams=WORKER_STREAMS,
        max_attempts=frozen_u2s.LAYOUT_RESAMPLE_ATTEMPTS,
    )
    storage_guard()
    started_at = utc_now()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "arm": arm.value,
        "started_at": started_at,
        "source": source,
        "runtime": runtime_snapshot(),
        "parent": parent.public_dict(),
        "qualification": _qualification_public(qualification),
        "cohort_id": cohort_contract["cohort_id"],
        "cohort_contract_sha256": cohort_contract_sha256,
        "sampler_preflight": [dict(value) for value in sampler],
        "effective_config": config,
        "segment": segment,
        "information_boundary": (
            "56x56x3 partial RGB pixels, preceding self-selected "
            "action, visible changed/unchanged outcome, and private "
            "256-unit recurrent state"
        ),
        "online_model_calls": False,
        "demonstrations": False,
        "oracle_actions_used_for_training": False,
        "confirmation_or_final_seed_access": False,
        "resume_authorized": False,
        "promotable": False,
    }

    handles: list[WorkerHandle] = []
    factories: list[Callable[[], gym.Env]] = []
    for worker_index, stream in enumerate(WORKER_STREAMS):

        def factory(
            index: int = worker_index,
            worker_stream: int = stream,
        ) -> gym.Env:
            environment = lessons.make_training_env(
                scheduler=scheduler,
                seed=worker_stream,
                size=9,
                seed_access=seed_access,
                forbidden_layout_hashes=forbidden_layout_hashes,
                max_layout_resample_attempts=(frozen_u2s.LAYOUT_RESAMPLE_ATTEMPTS),
            )
            episode = frozen_u2s.EpisodeEvidenceWrapper(
                environment,
                worker_index=index,
                worker_stream=worker_stream,
                ledger=run_directory / "episode-starts.jsonl",
            )
            first_rollout = FirstRolloutEvidenceWrapper(
                episode,
                worker_index=index,
            )
            context = ActionEffectObservation(
                first_rollout,
                mode=arm,
            )
            handles.append(
                WorkerHandle(
                    episode=episode,
                    context=context,
                    first_rollout=first_rollout,
                )
            )
            return context

        factories.append(factory)

    callback: Any | None = None
    vector_environment: Any | None = None
    setup_start = time.monotonic()
    try:
        vector_environment = VecTransposeImage(DummyVecEnv(factories))
        model, transplant = v03.build_transplanted_model(
            vector_environment,
            parent_checkpoint=v03.PARENT_CHECKPOINT,
            device=args.device,
        )
        smoke_arm = _qualification_smoke_arm(report, arm)
        expected_pre_action = json.loads(json.dumps(smoke_arm["pre_action_rng_identity"]))
        expected_first = normalized_first_rollout_identity(smoke_arm)
        callback_type = _CallbackFactory.create(BaseCallback)
        callback = callback_type(
            arm=arm,
            manifest=manifest,
            workers=handles,
            transplant=transplant,
            qualification_report=report,
            cohort_contract=cohort_contract,
            cohort_contract_sha256=cohort_contract_sha256,
            expected_pre_action_rng=expected_pre_action,
            expected_first_rollout=expected_first,
            run_directory=run_directory,
            state=state,
            controller=controller,
            scheduler=scheduler,
            parent=parent,
            qualification=qualification,
            source=source,
            segment=segment,
            effective_config=config,
            exam_model_loader=lambda path: RecurrentPPO.load(
                path,
                device="cpu",
            ),
            validation_access=seed_access,
            evaluation_interval=EVALUATION_INTERVAL,
            frame_interval=2_048,
            minimum_free_bytes=minimum_free_bytes,
            started_at=started_at,
            initial_trained_actions=PARENT_LIFETIME_ACTIONS,
            initial_updates=PARENT_OPTIMIZER_UPDATES,
            child_start_actions=PARENT_LIFETIME_ACTIONS,
            keep_rolling_exams=KEEP_ROLLING_EXAMS,
            storage_guard=storage_guard,
            source_guard=source_guard,
            staging_directory=(cohort_root / ".v03-staging"),
        )
        model.set_random_seed(ALGORITHM_SEED)
        seed_initial_rng_spaces(handles)
        print(
            f"v0.3 {arm.value} artifacts: {run_directory}",
            flush=True,
        )
        model.learn(
            total_timesteps=CHILD_ACTION_BUDGET,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        callback.finalize("completed")
    except KeyboardInterrupt:
        if callback is not None:
            _best_effort(
                "interruption evidence",
                lambda: callback.finalize_failure("interrupted"),
            )
        raise SystemExit(130) from None
    except BaseException:
        failure_traceback = traceback.format_exc()
        _best_effort(
            "crash report",
            lambda: atomic_write_json(
                run_directory / "crash.json",
                {
                    "timestamp": utc_now(),
                    "protocol": PROTOCOL,
                    "arm": arm.value,
                    "classification": (
                        "training_failure" if callback is not None else "pre_training_setup_failure"
                    ),
                    "setup_elapsed_seconds": (time.monotonic() - setup_start),
                    "traceback": failure_traceback,
                    "resume_authorized": False,
                    "cohort_outcome": "operationally_incomplete",
                },
            ),
        )
        if callback is not None:
            _best_effort(
                "crash status",
                lambda: callback.finalize_failure("crashed"),
            )
        raise
    finally:
        if vector_environment is not None:
            _best_effort("environment close", vector_environment.close)
        else:
            for handle in reversed(handles):
                _best_effort(
                    "environment close",
                    handle.context.close,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train-arm")
    train.add_argument(
        "--arm",
        choices=[arm.value for arm in v03.ARM_ORDER],
        required=True,
    )
    train.add_argument(
        "--qualification-report",
        type=Path,
        required=True,
    )
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--media-dir", type=Path, required=True)
    train.add_argument(
        "--cohort-contract",
        type=Path,
        required=True,
    )
    train.add_argument("--device", choices=("cpu",), default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command != "train-arm":
        raise SystemExit(f"unsupported v0.3 command: {args.command}")
    train_arm(args)


if __name__ == "__main__":
    main()
