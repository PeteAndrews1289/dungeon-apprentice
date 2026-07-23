"""Prospective U2 stability-mechanism ablation.

U2-S is a development experiment, not a retry of the consumed U2
confirmation.  Four matched arms start from the same authenticated, confirmed
U1 policy *and optimizer*, receive the same training streams, and consume the
same fixed U2 action budget:

``control``
    The original U2 PPO and reward.
``conservative``
    A child-action-linear learning-rate decay, smaller PPO clip, two epochs,
    and a KL stop.
``no-effect``
    Original PPO plus a bounded training-only penalty for the second and later
    identical interaction action whose next visible pixel observation is
    unchanged.
``combined``
    Both prospective interventions.

The policy still receives only 56 x 56 x 3 partial RGB pixels and its private
recurrent state.  The penalty wrapper is intentionally outside the lesson
environment: it may inspect only the selected primitive action and consecutive
policy-visible observations.  It is never installed for deterministic exams.

Every arm runs to the exact action ceiling.  The final three deterministic
development exams, rather than a best-looking checkpoint, determine whether
the mechanism is stable.  A later cohort launcher may select only the
mechanism/configuration according to :func:`select_mechanism`; this module
never authorizes reusing an ablation checkpoint as a scientific successor.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import gymnasium as gym
import numpy as np
from minigrid.core.actions import Actions
from PIL import Image

from dungeon_apprentice import v02_u1_confirm as state_digests
from dungeon_apprentice import v02_u2 as frozen_u2
from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v02_u2r as frozen_u2r
from dungeon_apprentice.artifacts import (
    append_jsonl,
    atomic_write_json,
    ensure_disk_space,
    file_sha256,
    git_snapshot,
    runtime_snapshot,
    utc_now,
)
from dungeon_apprentice.u2_storage import audit_u2_storage

PROTOCOL = "dungeon-apprentice-v0.2-u2s-stability-ablation"
SCHEMA_VERSION = 1
CASE_EVIDENCE_SCHEMA_VERSION = 1

PARENT_CHILD_SEED = 20260745
PARENT_U1_SEED = 20260733
PARENT_CHECKPOINT = frozen_u2.FROZEN_PARENTS[PARENT_CHILD_SEED].archive
PARENT_CHECKPOINT_SHA256 = "3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104"
PARENT_SIDECAR_SHA256 = "268f89361521dd855ba637254b236592186992718ad37ac374e85e5dbf408a07"
PARENT_MANIFEST = (
    "/Volumes/T7 Developer/DungeonApprentice/u1-local-replication-20260722/"
    "v02-u1-replication-seed-20260733/manifest.json"
)
PARENT_MANIFEST_SHA256 = "cd808ba8fea6b79225895b455d975122f85d66f1451569bd04a69737f8a6dfba"
PARENT_POLICY_TENSOR_SHA256 = "e555d3f7e2364f74e3b43371938c2f25e2ddbf62558a810038509a70868e6888"
PARENT_OPTIMIZER_STATE_SHA256 = "cc07791b374620d680e5cbb3602d59ab83eb55f195d26808abc0adf5f0e3bc5f"
CONFIRMATION_REPORT = (
    "/Volumes/T7 Developer/DungeonApprentice/confirmations/v0.2-u1-v2-20260723/report.json"
)
CONFIRMATION_REPORT_SHA256 = "6e577170050f6f14599b793a031776a19bf7c64eba0f243f457298da3193ae8f"
CONFIRMATION_ATTEMPT = (
    "/Volumes/T7 Developer/DungeonApprentice/confirmations/v0.2-u1-v2-20260723/attempt.json"
)
CONFIRMATION_ATTEMPT_SHA256 = "62147b2b556fe51e83876f2be35467fb488a4fb3b9c6347a600cf969e2bc86eb"
CONFIRMATION_CHECKSUM = (
    "/Volumes/T7 Developer/DungeonApprentice/confirmations/v0.2-u1-v2-20260723/report.json.sha256"
)
CONFIRMATION_CHECKSUM_SHA256 = "0be93cbbaf6581c4f6188ff6c4c82dee630f28da1cce462fa2acdb9caebed71a"
PARENT_LIFETIME_ACTIONS = 786_432
PARENT_OPTIMIZER_UPDATES = 1_536

ALGORITHM_SEED = 20260753
WORKER_STREAMS = (20260753, 20260754, 20260755, 20260756)
WORKERS = 4
INITIAL_RNG_COMPONENT_DIGESTS = (
    "python_random_sha256",
    "numpy_global_sha256",
    "torch_cpu_sha256",
    "torch_mps_sha256",
    "torch_cuda_sha256",
    "model_action_space_sha256",
    "scheduler_sha256",
)
INITIAL_RNG_NULLABLE_COMPONENT_DIGESTS = (
    "torch_mps_sha256",
    "torch_cuda_sha256",
)
INITIAL_RNG_WORKER_COMPONENT_DIGESTS = (
    "environment_np_random_sha256",
    "action_space_sha256",
    "observation_space_sha256",
)
ROLLOUT_STEPS = 512
ROLLOUT_TRANSITIONS = WORKERS * ROLLOUT_STEPS
BATCH_SIZE = 256
CHILD_ACTION_BUDGET = 1_048_576
EVALUATION_INTERVAL = 32_768
EVALUATION_SEED_COUNT = 80
EXAM_COUNT = CHILD_ACTION_BUDGET // EVALUATION_INTERVAL
FINAL_STABILITY_EXAMS = 3
MINIMUM_FREE_GIB = 25.0
LAYOUT_RESAMPLE_ATTEMPTS = frozen_u2r.U2R_LAYOUT_RESAMPLE_ATTEMPTS

ORIGINAL_LEARNING_RATE = 2.5e-4
CONSERVATIVE_FINAL_LEARNING_RATE = 2.5e-5
ORIGINAL_CLIP_RANGE = 0.2
CONSERVATIVE_CLIP_RANGE = 0.10
ORIGINAL_EPOCHS = 4
CONSERVATIVE_EPOCHS = 2
CONSERVATIVE_TARGET_KL = 0.015
ENTROPY_COEFFICIENT = 0.01
GAMMA = 0.995
GAE_LAMBDA = 0.98

NO_EFFECT_PENALTY = -0.01
NO_EFFECT_EPISODE_CAP = 0.10
NO_EFFECT_MAX_APPLIED_PER_EPISODE = round(NO_EFFECT_EPISODE_CAP / abs(NO_EFFECT_PENALTY))
INTERACTION_ACTIONS = frozenset(
    {
        int(Actions.pickup),
        int(Actions.drop),
        int(Actions.toggle),
    }
)
CASE_RECORD_FIELDS = frozenset(
    {
        "lesson_id",
        "lesson_label",
        "case_index",
        "panel",
        "panel_case_index",
        "seed",
        "layout_sha256",
        "geometry_sha256",
        "success",
        "terminal_reason",
        "steps",
        "milestones",
        "collisions",
        "ineffective_interactions",
        "unique_cells",
        "reachable_cells",
        "coverage",
        "oracle_actions",
        "path_actions_per_oracle_action",
        "action_histogram",
        "longest_repeated_action_run",
        "longest_identical_visible_no_effect_streak",
        "longest_repeated_identical_interaction_run",
        "visible_no_effect_transition_count",
        "visible_interaction_effects",
        "action_trace_sha256",
        "visible_no_effect_streak_sha256",
        "interaction_run_sha256",
        "extrinsic_return",
        "curiosity_return",
        "initial_key_visible",
        "initial_door_visible",
        "visibility_stratum",
    }
)

INEFFECTIVE_MEAN_MAX = 3.0
MAX_CASE_INEFFECTIVE_EXCLUSIVE = 10
MAX_IDENTICAL_INTERACTION_RUN_EXCLUSIVE = 10
U2_STABILITY_SUCCESSES = 72
U2_STABILITY_PANEL_SUCCESSES = 34

DEFAULT_QUALIFICATION_REPORT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.2-u2s-r1-20260723/report.json"
)
TRAINING_TAG = "u2s-stability-ablation-v0.2-u2s-r1-20260723"
COHORT_ID = "v0.2-u2s-ablation-r1-20260723"
PROTOCOL_DOCUMENT = "docs/protocol-v0.2-u2s-stability-ablation.md"
LINEAGE_CAP_BYTES = 2 * 1024**3
COHORT_SCIENTIFIC_CAP_BYTES = 6 * 1024**3
MEDIA_CAP_BYTES = 10 * 1024**3
COMBINED_PLANNED_CAP_BYTES = 16 * 1024**3
QUALIFIED_GUARD_MAPPING_SHA256 = "cf468f599737220224c696c796d1305c78453359537b80d7b32716ae53e1c2ef"
U2R_TERMINAL_ACTIVE_RECORDS_SHA256 = (
    "93663439a363a4c47152cc4e320654a1f2807a708637777b1fe71b358817e522"
)
CANONICAL_COHORT_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/u2s-ablation-r1-20260723"
)
CANONICAL_MEDIA_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/u2s-ablation-r1-media-20260723"
)


class U2SProtocolError(RuntimeError):
    """Raised when U2-S identity, state, or evidence fails closed."""


def _public_evidence(value: Any) -> dict[str, Any]:
    if hasattr(value, "public_dict"):
        public = value.public_dict()
    elif isinstance(value, Mapping):
        public = dict(value)
    else:
        raise U2SProtocolError("U2-S evidence has no public representation")
    return json.loads(json.dumps(public))


class ArmName(StrEnum):
    CONTROL = "control"
    CONSERVATIVE = "conservative"
    NO_EFFECT = "no-effect"
    COMBINED = "combined"


@dataclass(frozen=True)
class ArmSpec:
    name: ArmName
    conservative_ppo: bool
    no_effect_penalty: bool
    n_epochs: int
    learning_rate_start: float
    learning_rate_end: float
    clip_range: float
    target_kl: float | None

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "conservative_ppo": self.conservative_ppo,
            "no_effect_penalty": self.no_effect_penalty,
            "n_epochs": self.n_epochs,
            "learning_rate": {
                "schedule": (
                    "linear_by_new_u2_child_actions" if self.conservative_ppo else "constant"
                ),
                "start": self.learning_rate_start,
                "end": self.learning_rate_end,
            },
            "clip_range": self.clip_range,
            "target_kl": self.target_kl,
            "no_effect_reward": {
                "amount": NO_EFFECT_PENALTY if self.no_effect_penalty else 0.0,
                "episode_cap": (NO_EFFECT_EPISODE_CAP if self.no_effect_penalty else 0.0),
                "starts_on_identical_action_number": 2,
                "interaction_actions": sorted(INTERACTION_ACTIONS),
                "requires_next_visible_pixels_unchanged": True,
                "privileged_state": False,
                "enabled_during_evaluation": False,
                "reset_on_pixel_change": True,
                "reset_on_action_change": True,
                "reset_on_non_interaction": True,
                "reset_on_environment_reset": True,
                "reset_on_episode_end": True,
            },
        }


ARM_SPECS: Mapping[ArmName, ArmSpec] = MappingProxyType(
    {
        ArmName.CONTROL: ArmSpec(
            name=ArmName.CONTROL,
            conservative_ppo=False,
            no_effect_penalty=False,
            n_epochs=ORIGINAL_EPOCHS,
            learning_rate_start=ORIGINAL_LEARNING_RATE,
            learning_rate_end=ORIGINAL_LEARNING_RATE,
            clip_range=ORIGINAL_CLIP_RANGE,
            target_kl=None,
        ),
        ArmName.CONSERVATIVE: ArmSpec(
            name=ArmName.CONSERVATIVE,
            conservative_ppo=True,
            no_effect_penalty=False,
            n_epochs=CONSERVATIVE_EPOCHS,
            learning_rate_start=ORIGINAL_LEARNING_RATE,
            learning_rate_end=CONSERVATIVE_FINAL_LEARNING_RATE,
            clip_range=CONSERVATIVE_CLIP_RANGE,
            target_kl=CONSERVATIVE_TARGET_KL,
        ),
        ArmName.NO_EFFECT: ArmSpec(
            name=ArmName.NO_EFFECT,
            conservative_ppo=False,
            no_effect_penalty=True,
            n_epochs=ORIGINAL_EPOCHS,
            learning_rate_start=ORIGINAL_LEARNING_RATE,
            learning_rate_end=ORIGINAL_LEARNING_RATE,
            clip_range=ORIGINAL_CLIP_RANGE,
            target_kl=None,
        ),
        ArmName.COMBINED: ArmSpec(
            name=ArmName.COMBINED,
            conservative_ppo=True,
            no_effect_penalty=True,
            n_epochs=CONSERVATIVE_EPOCHS,
            learning_rate_start=ORIGINAL_LEARNING_RATE,
            learning_rate_end=CONSERVATIVE_FINAL_LEARNING_RATE,
            clip_range=CONSERVATIVE_CLIP_RANGE,
            target_kl=CONSERVATIVE_TARGET_KL,
        ),
    }
)

ARM_PRIORITY = (
    ArmName.CONTROL,
    ArmName.CONSERVATIVE,
    ArmName.NO_EFFECT,
    ArmName.COMBINED,
)


def optimizer_update_bounds(
    arm: ArmName | str,
    trained_child_actions: int,
    *,
    parent_updates: int = 0,
) -> tuple[int, int]:
    selected = ArmName(arm)
    actions = int(trained_child_actions)
    if actions < 0 or actions % ROLLOUT_TRANSITIONS:
        raise U2SProtocolError("optimizer update bounds require a complete vector rollout")
    rollouts = actions // ROLLOUT_TRANSITIONS
    spec = ARM_SPECS[selected]
    minimum = int(parent_updates) + rollouts
    maximum = int(parent_updates) + rollouts * spec.n_epochs
    if not spec.conservative_ppo:
        minimum = maximum
    return minimum, maximum


def optimizer_update_count_valid(
    arm: ArmName | str,
    trained_child_actions: int,
    actual_updates: int,
    *,
    parent_updates: int = 0,
) -> bool:
    minimum, maximum = optimizer_update_bounds(
        arm,
        trained_child_actions,
        parent_updates=parent_updates,
    )
    return minimum <= int(actual_updates) <= maximum


def optimizer_epoch_accounting(
    arm: ArmName | str,
    trained_child_actions: int,
    completed_updates: int,
    *,
    approx_kl: float | None,
) -> dict[str, Any]:
    """Describe planned versus completed PPO epochs at one safe boundary."""

    selected = ArmName(arm)
    actions = int(trained_child_actions)
    completed = int(completed_updates)
    if actions <= 0 or actions % ROLLOUT_TRANSITIONS:
        raise U2SProtocolError("optimizer epoch accounting requires complete vector rollouts")
    if not optimizer_update_count_valid(selected, actions, completed):
        raise U2SProtocolError("optimizer epoch accounting has invalid updates")
    planned = actions // ROLLOUT_TRANSITIONS * ARM_SPECS[selected].n_epochs
    skipped = planned - completed
    return {
        "epochs_planned": planned,
        "epochs_completed": completed,
        "epochs_skipped": skipped,
        "target_kl": ARM_SPECS[selected].target_kl,
        "kl_stop_triggered": (ARM_SPECS[selected].target_kl is not None and skipped > 0),
        "approx_kl": (float(approx_kl) if approx_kl is not None else None),
    }


@dataclass(frozen=True)
class ChildActionLinearSchedule:
    """Map SB3 lifetime progress to an exact new-child-action LR schedule."""

    inherited_actions: int = PARENT_LIFETIME_ACTIONS
    child_budget: int = CHILD_ACTION_BUDGET
    start: float = ORIGINAL_LEARNING_RATE
    end: float = CONSERVATIVE_FINAL_LEARNING_RATE

    @property
    def terminal_lifetime_actions(self) -> int:
        return self.inherited_actions + self.child_budget

    @property
    def raw_start_progress_remaining(self) -> float:
        return self.child_budget / self.terminal_lifetime_actions

    def for_child_actions(self, child_actions: int | float) -> float:
        fraction = min(1.0, max(0.0, float(child_actions) / self.child_budget))
        return self.start + (self.end - self.start) * fraction

    def __call__(self, progress_remaining: float) -> float:
        raw = min(
            self.raw_start_progress_remaining,
            max(0.0, float(progress_remaining)),
        )
        child_actions = self.terminal_lifetime_actions * (1.0 - raw) - self.inherited_actions
        return self.for_child_actions(child_actions)

    def public_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "raw_start_progress_remaining": self.raw_start_progress_remaining,
            "values": {
                "child_0": self.for_child_actions(0),
                "child_half": self.for_child_actions(self.child_budget // 2),
                "child_terminal": self.for_child_actions(self.child_budget),
            },
        }


@dataclass(frozen=True)
class ConstantSchedule:
    value: float

    def __call__(self, _progress_remaining: float) -> float:
        return self.value


class NoEffectInteractionPenalty(gym.Wrapper):
    """Bounded pixels-only feedback for repeated ineffective interactions.

    No environment state or ``info`` field participates in the decision.  A
    penalty is paid only when all of these policy-visible facts are true:

    * the current selected primitive is pickup, drop, or toggle;
    * it is identical to the immediately preceding selected primitive; and
    * the current observation is byte-identical to the preceding observation.

    The first action in a run cannot be penalized.  The streak and budget reset
    on every episode.  Evaluation environments never install this wrapper.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        penalty: float = NO_EFFECT_PENALTY,
        episode_cap: float = NO_EFFECT_EPISODE_CAP,
    ) -> None:
        super().__init__(env)
        if not math.isfinite(float(penalty)) or float(penalty) >= 0:
            raise ValueError("no-effect penalty must be finite and negative")
        if not math.isfinite(float(episode_cap)) or not 0 < float(episode_cap) < 1:
            raise ValueError("no-effect episode cap must be finite and below success")
        self.penalty = float(penalty)
        self.episode_cap = float(episode_cap)
        self._previous_observation: np.ndarray | None = None
        self._previous_action: int | None = None
        self._identical_interaction_run = 0
        self._episode_penalty = 0.0
        self._eligible_event_count = 0
        self._applied_penalty_count = 0

    def _clear_episode_state(self) -> None:
        self._previous_observation = None
        self._previous_action = None
        self._identical_interaction_run = 0
        self._episode_penalty = 0.0
        self._eligible_event_count = 0
        self._applied_penalty_count = 0

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._clear_episode_state()
        self._previous_observation = np.array(observation, copy=True)
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        selected = int(np.asarray(action).item())
        observation, reward, terminated, truncated, info = self.env.step(selected)
        current = np.asarray(observation)
        unchanged = (
            self._previous_observation is not None
            and current.shape == self._previous_observation.shape
            and current.dtype == self._previous_observation.dtype
            and current.tobytes() == self._previous_observation.tobytes()
        )
        interaction = selected in INTERACTION_ACTIONS
        identical = interaction and self._previous_action == selected
        eligible_no_effect = interaction and unchanged
        if eligible_no_effect and identical and self._identical_interaction_run > 0:
            self._identical_interaction_run += 1
        elif eligible_no_effect:
            self._identical_interaction_run = 1
        else:
            self._identical_interaction_run = 0

        applied = 0.0
        maximum_applied = math.floor((self.episode_cap / abs(self.penalty)) + 1e-12)
        if self._identical_interaction_run >= 2:
            self._eligible_event_count += 1
            if self._applied_penalty_count < maximum_applied:
                self._applied_penalty_count += 1
                applied = self.penalty
                # Derive the cumulative value from the integer event count.
                # This avoids a binary-float residual authorizing an eleventh
                # microscopic penalty after ten -0.01 events.
                self._episode_penalty = max(
                    -self.episode_cap,
                    self._applied_penalty_count * self.penalty,
                )
        shaped_reward = float(reward) + applied
        result_info = dict(info)
        result_info.update(
            {
                "u2s_no_effect_penalty": applied,
                "u2s_no_effect_episode_total": self._episode_penalty,
                "u2s_no_effect_eligible_event_count": (self._eligible_event_count),
                "u2s_no_effect_applied_penalty_count": (self._applied_penalty_count),
                "u2s_identical_interaction_run": self._identical_interaction_run,
                "u2s_visible_pixels_unchanged": bool(unchanged),
            }
        )
        self._previous_observation = np.array(current, copy=True)
        self._previous_action = selected
        if terminated or truncated:
            self._clear_episode_state()
        return observation, shaped_reward, terminated, truncated, result_info


class EpisodeEvidenceWrapper(gym.Wrapper):
    """Append-only identity evidence for every U2-S training episode."""

    def __init__(
        self,
        env: gym.Env,
        *,
        worker_index: int,
        worker_stream: int,
        ledger: Path,
    ) -> None:
        super().__init__(env)
        self.worker_index = int(worker_index)
        self.worker_stream = int(worker_stream)
        self.ledger = ledger
        self.episode_ordinal = 0
        self.local_transition_count = 0
        self._active: dict[str, Any] | None = None

    @staticmethod
    def _digest(value: Any, label: str) -> str:
        digest = str(value)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise U2SProtocolError(f"{label} is not a SHA-256 digest")
        return digest

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.episode_ordinal += 1
        episode_seed = int(info["seed"])
        self._active = {
            "worker_index": self.worker_index,
            "worker_stream": self.worker_stream,
            "episode_ordinal": self.episode_ordinal,
            "episode_seed": episode_seed,
            "lesson_id": str(info["lesson_id"]),
            "layout_sha256": self._digest(
                info.get("layout_sha256"),
                "U2-S training layout",
            ),
            "geometry_sha256": self._digest(
                info.get("geometry_sha256"),
                "U2-S training geometry",
            ),
            "worker_transition_at_start": self.local_transition_count,
            "elapsed_steps": 0,
            "active": True,
        }
        append_jsonl(
            self.ledger,
            {
                "timestamp": utc_now(),
                "type": "episode_start",
                **self._active,
            },
        )
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.local_transition_count += 1
        if self._active is None:
            raise U2SProtocolError("U2-S worker stepped before episode-start evidence")
        self._active["elapsed_steps"] = int(
            info.get("elapsed_steps", self._active["elapsed_steps"] + 1)
            or self._active["elapsed_steps"] + 1
        )
        self._active["active"] = not bool(terminated or truncated)
        return observation, reward, terminated, truncated, info

    def evidence_state(self) -> dict[str, Any]:
        if self._active is None:
            raise U2SProtocolError("U2-S worker has no active episode identity")
        return json.loads(json.dumps(self._active))


def visible_no_effect_streaks(
    actions: Sequence[int],
    observations_before: Sequence[np.ndarray],
    observations_after: Sequence[np.ndarray],
) -> tuple[int, ...]:
    """Return the eligible streak after each observation-aware transition."""

    if not (len(actions) == len(observations_before) == len(observations_after)):
        raise ValueError("actions and visible observations must align")
    streaks: list[int] = []
    current = 0
    previous_eligible_action: int | None = None
    for raw_action, before, after in zip(
        actions,
        observations_before,
        observations_after,
        strict=True,
    ):
        action = int(raw_action)
        unchanged = (
            np.asarray(before).shape == np.asarray(after).shape
            and np.asarray(before).dtype == np.asarray(after).dtype
            and np.asarray(before).tobytes() == np.asarray(after).tobytes()
        )
        if action in INTERACTION_ACTIONS and unchanged:
            current = current + 1 if previous_eligible_action == action else 1
            previous_eligible_action = action
        else:
            current = 0
            previous_eligible_action = None
        streaks.append(current)
    return tuple(streaks)


def repeated_identical_interaction_runs(
    actions: Sequence[int],
) -> tuple[int, ...]:
    """Return the generic identical interaction run after every action."""

    runs: list[int] = []
    current = 0
    previous: int | None = None
    for raw_action in actions:
        action = int(raw_action)
        current = (
            current + 1
            if action in INTERACTION_ACTIONS and previous == action
            else int(action in INTERACTION_ACTIONS)
        )
        runs.append(current)
        previous = action
    return tuple(runs)


def visible_interaction_effect_evidence(
    actions: Sequence[int],
    observations_before: Sequence[np.ndarray],
    observations_after: Sequence[np.ndarray],
    *,
    penalty_enabled: bool,
) -> dict[str, Any]:
    """Return recomputable per-action visible effects and penalty diagnostics."""

    if not (len(actions) == len(observations_before) == len(observations_after)):
        raise ValueError("actions and visible observations must align")
    streaks = visible_no_effect_streaks(
        actions,
        observations_before,
        observations_after,
    )
    labels = {
        int(Actions.pickup): "pickup",
        int(Actions.drop): "drop",
        int(Actions.toggle): "toggle",
    }
    per_action: dict[str, dict[str, Any]] = {}
    for action, label in labels.items():
        indices = [index for index, selected in enumerate(actions) if int(selected) == action]
        visible_no_effects = sum(streaks[index] > 0 for index in indices)
        attempts = len(indices)
        visible_effects = attempts - visible_no_effects
        per_action[label] = {
            "action": action,
            "attempts": attempts,
            "visible_effects": visible_effects,
            "visible_no_effects": visible_no_effects,
            "visible_effect_rate": (visible_effects / attempts if attempts else None),
            "visible_no_effect_rate": (visible_no_effects / attempts if attempts else None),
        }
    eligible_events = sum(streak >= 2 for streak in streaks)
    applied_count = (
        min(eligible_events, NO_EFFECT_MAX_APPLIED_PER_EPISODE) if penalty_enabled else 0
    )
    return {
        "per_action": per_action,
        "total_interaction_attempts": sum(int(value["attempts"]) for value in per_action.values()),
        "total_visible_effects": sum(
            int(value["visible_effects"]) for value in per_action.values()
        ),
        "total_visible_no_effects": sum(
            int(value["visible_no_effects"]) for value in per_action.values()
        ),
        "penalty_diagnostic": {
            "enabled_for_arm": bool(penalty_enabled),
            "eligible_events": eligible_events,
            "applied_count": applied_count,
            "capped_return": applied_count * NO_EFFECT_PENALTY,
            "episode_cap": NO_EFFECT_EPISODE_CAP,
            "evaluation_reward_changed": False,
        },
    }


def aggregate_visible_interaction_effect_evidence(
    cases: Sequence[Mapping[str, Any]],
    *,
    penalty_enabled: bool,
) -> dict[str, Any]:
    """Aggregate exact case evidence without averaging per-case rates."""

    labels = ("pickup", "drop", "toggle")
    per_action: dict[str, dict[str, Any]] = {}
    for label in labels:
        records = [case["visible_interaction_effects"]["per_action"][label] for case in cases]
        attempts = sum(int(record["attempts"]) for record in records)
        effects = sum(int(record["visible_effects"]) for record in records)
        no_effects = sum(int(record["visible_no_effects"]) for record in records)
        per_action[label] = {
            "action": int(records[0]["action"])
            if records
            else {
                "pickup": int(Actions.pickup),
                "drop": int(Actions.drop),
                "toggle": int(Actions.toggle),
            }[label],
            "attempts": attempts,
            "visible_effects": effects,
            "visible_no_effects": no_effects,
            "visible_effect_rate": effects / attempts if attempts else None,
            "visible_no_effect_rate": no_effects / attempts if attempts else None,
        }
    eligible = sum(
        int(case["visible_interaction_effects"]["penalty_diagnostic"]["eligible_events"])
        for case in cases
    )
    applied = sum(
        int(case["visible_interaction_effects"]["penalty_diagnostic"]["applied_count"])
        for case in cases
    )
    return {
        "case_count": len(cases),
        "per_action": per_action,
        "penalty_diagnostic": {
            "enabled_for_arm": bool(penalty_enabled),
            "eligible_events": eligible,
            "applied_count": applied,
            "capped_return": applied * NO_EFFECT_PENALTY,
            "evaluation_reward_changed": False,
        },
    }


def ineffective_tail_summary(values: Sequence[int]) -> dict[str, Any]:
    normalized = tuple(int(value) for value in values)
    if any(value < 0 for value in normalized):
        raise ValueError("ineffective-interaction counts cannot be negative")
    ordered = sorted(normalized, reverse=True)
    total = sum(ordered)

    def worst_share(count: int) -> float:
        return sum(ordered[:count]) / total if total else 0.0

    return {
        "cases": len(normalized),
        "ineffective_quantiles": {
            "p50": float(np.percentile(normalized, 50)) if normalized else 0.0,
            "p90": float(np.percentile(normalized, 90)) if normalized else 0.0,
            "p95": float(np.percentile(normalized, 95)) if normalized else 0.0,
            "p99": float(np.percentile(normalized, 99)) if normalized else 0.0,
        },
        "ineffective_threshold_counts": {
            "at_least_1": sum(value >= 1 for value in normalized),
            "at_least_3": sum(value >= 3 for value in normalized),
            "at_least_10": sum(value >= 10 for value in normalized),
            "at_least_32": sum(value >= 32 for value in normalized),
        },
        "worst_case_shares": {
            "worst_1": worst_share(1),
            "worst_2": worst_share(2),
            "worst_5": worst_share(5),
        },
    }


def _augmented_lesson_evidence(
    result: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    penalty_enabled: bool,
) -> dict[str, Any]:
    public = result.public_dict()
    ineffective_values = sorted(
        (int(case["ineffective_interactions"]) for case in cases),
        reverse=True,
    )
    public["passed"] = lessons.lesson_passed(result)
    public["max_ineffective_interactions"] = max(ineffective_values, default=0)
    public["max_identical_visible_no_effect_streak"] = max(
        (int(case["longest_identical_visible_no_effect_streak"]) for case in cases),
        default=0,
    )
    public["max_repeated_identical_interaction_run"] = max(
        (int(case["longest_repeated_identical_interaction_run"]) for case in cases),
        default=0,
    )
    public["repeated_identical_interaction_run_counts"] = {
        f"at_least_{threshold}": sum(
            int(case["longest_repeated_identical_interaction_run"]) >= threshold
            for case in cases
        )
        for threshold in (3, 10, 32)
    }
    public["visible_no_effect_streak_counts"] = {
        f"at_least_{threshold}": sum(
            int(case["longest_identical_visible_no_effect_streak"]) >= threshold
            for case in cases
        )
        for threshold in (3, 10, 32)
    }
    public["ineffective_tail"] = ineffective_tail_summary(ineffective_values)
    public["visible_interaction_effects"] = aggregate_visible_interaction_effect_evidence(
        cases,
        penalty_enabled=penalty_enabled,
    )
    public["top_two_ineffective_share"] = public["ineffective_tail"][
        "worst_case_shares"
    ]["worst_2"]
    return public


def _case_diagnostics_from_records(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ineffective_values = [int(case["ineffective_interactions"]) for case in cases]
    visible_streaks = [
        int(case["longest_identical_visible_no_effect_streak"]) for case in cases
    ]
    interaction_runs = [
        int(case["longest_repeated_identical_interaction_run"]) for case in cases
    ]
    return {
        "case_count": len(cases),
        "max_case_ineffective_interactions": max(ineffective_values, default=0),
        "cases_with_ineffective_at_least_10": sum(
            value >= MAX_CASE_INEFFECTIVE_EXCLUSIVE for value in ineffective_values
        ),
        "max_identical_visible_no_effect_streak": max(visible_streaks, default=0),
        "max_repeated_identical_interaction_run": max(interaction_runs, default=0),
        "cases_with_repeated_identical_interaction_run_at_least_10": sum(
            value >= MAX_IDENTICAL_INTERACTION_RUN_EXCLUSIVE
            for value in interaction_runs
        ),
        "cases_with_identical_visible_no_effect_streak_at_least_10": sum(
            value >= MAX_IDENTICAL_INTERACTION_RUN_EXCLUSIVE
            for value in visible_streaks
        ),
        "ineffective_tail": ineffective_tail_summary(ineffective_values),
    }


def _validate_case_visible_effects(
    case: Mapping[str, Any],
    *,
    penalty_enabled: bool,
) -> None:
    evidence = case.get("visible_interaction_effects")
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "per_action",
        "total_interaction_attempts",
        "total_visible_effects",
        "total_visible_no_effects",
        "penalty_diagnostic",
    }:
        raise U2SProtocolError("U2-S case visible-effect evidence is incomplete")
    per_action = evidence.get("per_action")
    if not isinstance(per_action, Mapping) or set(per_action) != {
        "pickup",
        "drop",
        "toggle",
    }:
        raise U2SProtocolError("U2-S case visible-effect actions changed")
    action_histogram = case["action_histogram"]
    expected_actions = {
        "pickup": int(Actions.pickup),
        "drop": int(Actions.drop),
        "toggle": int(Actions.toggle),
    }
    attempts_total = 0
    effects_total = 0
    no_effects_total = 0
    for label, action in expected_actions.items():
        record = per_action[label]
        if not isinstance(record, Mapping) or set(record) != {
            "action",
            "attempts",
            "visible_effects",
            "visible_no_effects",
            "visible_effect_rate",
            "visible_no_effect_rate",
        }:
            raise U2SProtocolError("U2-S case visible-effect record changed")
        raw_counts = (
            record.get("attempts"),
            record.get("visible_effects"),
            record.get("visible_no_effects"),
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or int(value) < 0
            for value in raw_counts
        ):
            raise U2SProtocolError("U2-S case visible-effect counts are invalid")
        attempts, effects, no_effects = (int(value) for value in raw_counts)
        expected_effect_rate = effects / attempts if attempts else None
        expected_no_effect_rate = no_effects / attempts if attempts else None
        if (
            isinstance(record.get("action"), bool)
            or not isinstance(record.get("action"), int)
            or record.get("action") != action
            or attempts != int(action_histogram[str(action)])
            or effects + no_effects != attempts
            or record.get("visible_effect_rate") != expected_effect_rate
            or record.get("visible_no_effect_rate") != expected_no_effect_rate
        ):
            raise U2SProtocolError("U2-S case visible-effect arithmetic changed")
        attempts_total += attempts
        effects_total += effects
        no_effects_total += no_effects
    raw_totals = (
        evidence.get("total_interaction_attempts"),
        evidence.get("total_visible_effects"),
        evidence.get("total_visible_no_effects"),
    )
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_totals
        )
        or
        evidence.get("total_interaction_attempts") != attempts_total
        or evidence.get("total_visible_effects") != effects_total
        or evidence.get("total_visible_no_effects") != no_effects_total
        or int(case["visible_no_effect_transition_count"]) != no_effects_total
    ):
        raise U2SProtocolError("U2-S case visible-effect totals changed")
    penalty = evidence.get("penalty_diagnostic")
    if not isinstance(penalty, Mapping) or set(penalty) != {
        "enabled_for_arm",
        "eligible_events",
        "applied_count",
        "capped_return",
        "episode_cap",
        "evaluation_reward_changed",
    }:
        raise U2SProtocolError("U2-S case penalty diagnostic changed")
    eligible = penalty.get("eligible_events")
    applied = penalty.get("applied_count")
    if (
        isinstance(eligible, bool)
        or not isinstance(eligible, int)
        or eligible < 0
        or isinstance(applied, bool)
        or not isinstance(applied, int)
        or applied < 0
    ):
        raise U2SProtocolError("U2-S case penalty counts are invalid")
    expected_applied = (
        min(eligible, NO_EFFECT_MAX_APPLIED_PER_EPISODE) if penalty_enabled else 0
    )
    if (
        penalty.get("enabled_for_arm") is not penalty_enabled
        or applied != expected_applied
        or penalty.get("capped_return") != expected_applied * NO_EFFECT_PENALTY
        or penalty.get("episode_cap") != NO_EFFECT_EPISODE_CAP
        or penalty.get("evaluation_reward_changed") is not False
    ):
        raise U2SProtocolError("U2-S case penalty diagnostic is inconsistent")


def _verify_exam_case_records(
    exam: Mapping[str, Any],
    case_records: Sequence[Mapping[str, Any]],
    *,
    arm: ArmName | str,
) -> None:
    selected_arm = ArmName(arm)
    evaluations = exam.get("lessons")
    if not isinstance(evaluations, Mapping) or set(evaluations) != {
        lesson.value for lesson in lessons.LessonId
    }:
        raise U2SProtocolError("U2-S exam lesson summaries are incomplete")
    records = [dict(record) for record in case_records]
    if len(records) != len(lessons.LessonId) * EVALUATION_SEED_COUNT:
        raise U2SProtocolError("U2-S exam does not contain exactly 320 cases")
    ordered_records: list[dict[str, Any]] = []
    all_case_digests: list[str] = []
    penalty_enabled = ARM_SPECS[selected_arm].no_effect_penalty
    for lesson in lessons.LessonId:
        selected = [
            record for record in records if record.get("lesson_id") == lesson.value
        ]
        selected.sort(key=lambda record: int(record.get("case_index", -1)))
        if len(selected) != EVALUATION_SEED_COUNT:
            raise U2SProtocolError(
                f"U2-S {lesson.value} case evidence is not exactly 80 cases"
            )
        expected_seeds = tuple(
            lessons.LESSON_SPECS[lesson].validation_seed_base + index
            for index in range(EVALUATION_SEED_COUNT)
        )
        if tuple(int(record.get("seed", -1)) for record in selected) != expected_seeds:
            raise U2SProtocolError(
                f"U2-S {lesson.value} case evidence seed order changed"
            )
        for index, record in enumerate(selected):
            if set(record) != CASE_RECORD_FIELDS:
                raise U2SProtocolError(
                    f"U2-S {lesson.value} case record schema changed"
                )
            panel = 0 if index < EVALUATION_SEED_COUNT // 2 else 1
            panel_index = index % (EVALUATION_SEED_COUNT // 2)
            if (
                any(
                    isinstance(record.get(field_name), bool)
                    or not isinstance(record.get(field_name), int)
                    for field_name in (
                        "case_index",
                        "panel",
                        "panel_case_index",
                        "seed",
                    )
                )
                or record.get("lesson_label")
                != lessons.LESSON_SPECS[lesson].label
                or record.get("case_index") != index
                or record.get("panel") != panel
                or record.get("panel_case_index") != panel_index
                or record.get("seed") != expected_seeds[index]
                or not isinstance(record.get("success"), bool)
                or record.get("terminal_reason")
                != ("success" if record["success"] else "time_limit")
                or not isinstance(record.get("milestones"), Mapping)
                or not all(
                    isinstance(value, bool)
                    for value in record["milestones"].values()
                )
                or record["milestones"].get("success") is not record["success"]
            ):
                raise U2SProtocolError(
                    f"U2-S {lesson.value} case identity changed"
                )
            for field_name in (
                "layout_sha256",
                "geometry_sha256",
                "action_trace_sha256",
                "visible_no_effect_streak_sha256",
                "interaction_run_sha256",
            ):
                digest = record.get(field_name)
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in digest
                    )
                ):
                    raise U2SProtocolError(
                        f"U2-S {lesson.value} case digest changed"
                    )
            steps = int(record.get("steps", -1))
            oracle_actions = int(record.get("oracle_actions", -1))
            oracle_range = lessons.LESSON_SPECS[lesson].oracle_action_range
            longest_values = (
                record.get("longest_repeated_action_run"),
                record.get("longest_identical_visible_no_effect_streak"),
                record.get("longest_repeated_identical_interaction_run"),
                record.get("visible_no_effect_transition_count"),
            )
            if (
                steps <= 0
                or steps > lessons.LESSON_SPECS[lesson].max_steps
                or oracle_actions <= 0
                or (
                    oracle_range is not None
                    and not (
                        oracle_range[0]
                        <= oracle_actions
                        <= oracle_range[1]
                    )
                )
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value <= steps
                    for value in longest_values
                )
            ):
                raise U2SProtocolError(
                    f"U2-S {lesson.value} case action diagnostics changed"
                )
            _validate_case_visible_effects(
                record,
                penalty_enabled=penalty_enabled,
            )
        public = evaluations[lesson.value]
        if not isinstance(public, Mapping) or not isinstance(
            public.get("timestamp"), str
        ):
            raise U2SProtocolError(
                f"U2-S {lesson.value} aggregate is incomplete"
            )
        try:
            result = lessons.aggregate_u2_case_evidence(
                lesson,
                selected,
                timestamp=str(public["timestamp"]),
            )
            recomputed = _augmented_lesson_evidence(
                result,
                selected,
                penalty_enabled=penalty_enabled,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise U2SProtocolError(
                f"U2-S {lesson.value} case evidence is invalid: {error}"
            ) from error
        if _canonical_sha256(recomputed) != _canonical_sha256(public):
            raise U2SProtocolError(
                f"U2-S {lesson.value} aggregate disagrees with immutable cases"
            )
        case_digests = [_canonical_sha256(record) for record in selected]
        if len(set(case_digests)) != EVALUATION_SEED_COUNT:
            raise U2SProtocolError(
                f"U2-S {lesson.value} case evidence is duplicated"
            )
        all_case_digests.extend(case_digests)
        ordered_records.extend(selected)
    if records != ordered_records:
        raise U2SProtocolError("U2-S exam case record order changed")
    if len(set(all_case_digests)) != len(all_case_digests):
        raise U2SProtocolError("U2-S exam case records are duplicated across lessons")


def evaluate_lesson_with_visible_no_effect(
    model: Any,
    lesson: lessons.LessonId | str,
    seeds: Sequence[int],
    *,
    seed_access: Any,
    penalty_enabled: bool = False,
    frame_path: Path | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Run the frozen deterministic exam while retaining visible transitions."""

    selected = lessons.LessonId(lesson)
    seed_values = tuple(int(seed) for seed in seeds)
    midpoint = len(seed_values) // 2
    outcomes: list[dict[str, Any]] = []
    latest: np.ndarray | None = None
    for case_index, seed in enumerate(seed_values):
        environment = lessons.make_u2_pixel_env(lesson=selected, size=9)
        try:
            observation, reset_info = lessons._reset_lesson_env(
                environment,
                selected,
                seed,
                seed_access=seed_access,
            )
            base = environment.unwrapped
            if not isinstance(base, lessons.U2LessonEnv):
                raise U2SProtocolError("U2-S pixel wrapper lost its frozen lesson environment")
            recurrent_state = None
            episode_start = np.ones((1,), dtype=bool)
            actions: list[int] = []
            observations_before: list[np.ndarray] = []
            observations_after: list[np.ndarray] = []
            extrinsic_return = 0.0
            curiosity_return = 0.0
            info: dict[str, Any] = {}
            while True:
                action, recurrent_state = model.predict(
                    lessons._policy_observation(observation, model),
                    state=recurrent_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                episode_start[:] = False
                selected_action = int(np.asarray(action).item())
                before = np.array(observation, copy=True)
                observation, reward, terminated, truncated, info = environment.step(selected_action)
                actions.append(selected_action)
                observations_before.append(before)
                observations_after.append(np.array(observation, copy=True))
                extrinsic_return += float(info.get("extrinsic_reward", reward))
                curiosity_return += float(info.get("curiosity_reward", 0.0))
                if terminated or truncated:
                    break
            latest = observation
            streaks = visible_no_effect_streaks(
                actions,
                observations_before,
                observations_after,
            )
            interaction_runs = repeated_identical_interaction_runs(actions)
            oracle_actions = lessons._oracle_action_count(
                selected,
                seed,
                size=9,
                seed_access=seed_access,
            )
            reachable = lessons._reachable_cell_count(base)
            key_visible = reset_info.get("initial_key_visible")
            door_visible = reset_info.get("initial_door_visible")
            per_case_actions = Counter(actions)
            effect_evidence = visible_interaction_effect_evidence(
                actions,
                observations_before,
                observations_after,
                penalty_enabled=penalty_enabled,
            )
            outcome = {
                "lesson_id": selected.value,
                "lesson_label": lessons.LESSON_SPECS[selected].label,
                "case_index": case_index,
                "panel": 0 if case_index < midpoint else 1,
                "panel_case_index": (
                    case_index if case_index < midpoint else case_index - midpoint
                ),
                "seed": seed,
                "layout_sha256": str(reset_info.get("layout_sha256", "")),
                "geometry_sha256": str(reset_info.get("geometry_sha256", "")),
                "success": bool(info.get("success", False)),
                "terminal_reason": info.get("terminal_reason"),
                "steps": len(actions),
                "milestones": dict(info.get("milestones", {})),
                "collisions": int(info.get("collisions", 0)),
                "ineffective_interactions": int(info.get("ineffective_interactions", 0)),
                "unique_cells": int(info.get("unique_cells", 0)),
                "reachable_cells": int(reachable),
                "coverage": (int(info.get("unique_cells", 0)) / reachable if reachable else 0.0),
                "oracle_actions": int(oracle_actions),
                "path_actions_per_oracle_action": len(actions) / oracle_actions,
                "action_histogram": {
                    str(action): int(per_case_actions.get(action, 0)) for action in range(7)
                },
                "longest_repeated_action_run": lessons._longest_action_run(actions),
                "longest_identical_visible_no_effect_streak": max(streaks, default=0),
                "longest_repeated_identical_interaction_run": max(
                    interaction_runs,
                    default=0,
                ),
                "visible_no_effect_transition_count": sum(streak > 0 for streak in streaks),
                "visible_interaction_effects": effect_evidence,
                "action_trace_sha256": _canonical_sha256(actions),
                "visible_no_effect_streak_sha256": _canonical_sha256(streaks),
                "interaction_run_sha256": _canonical_sha256(interaction_runs),
                "extrinsic_return": float(extrinsic_return),
                "curiosity_return": float(curiosity_return),
                "initial_key_visible": (bool(key_visible) if key_visible is not None else None),
                "initial_door_visible": (bool(door_visible) if door_visible is not None else None),
                "visibility_stratum": (
                    lessons._visibility_stratum(key_visible, door_visible)
                    if selected is lessons.LessonId.SEPARATED_UNLOCK
                    else None
                ),
            }
            outcomes.append(outcome)
        finally:
            environment.close()
    if not outcomes:
        raise U2SProtocolError("U2-S evaluation requires at least one case")
    if frame_path is not None and latest is not None:
        temporary = frame_path.with_name(f".{frame_path.name}.tmp")
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(latest).save(temporary, format="PNG")
        os.replace(temporary, frame_path)
    result = lessons.aggregate_u2_case_evidence(
        selected,
        outcomes,
        timestamp=utc_now(),
    )
    return result, outcomes


@dataclass
class AblationController:
    exam_records: list[dict[str, Any]] = field(default_factory=list)
    practice_decisions: list[dict[str, Any]] = field(default_factory=list)

    def public_dict(self) -> dict[str, Any]:
        return {
            "exam_records": json.loads(json.dumps(self.exam_records)),
            "practice_decisions": json.loads(json.dumps(self.practice_decisions)),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AblationController:
        records = value.get("exam_records")
        if not isinstance(records, list) or not all(
            isinstance(record, Mapping) for record in records
        ):
            raise U2SProtocolError("U2-S controller exam records are invalid")
        normalized = [json.loads(json.dumps(record)) for record in records]
        actions = [int(record.get("child_trained_actions", -1)) for record in normalized]
        expected = [index * EVALUATION_INTERVAL for index in range(1, len(normalized) + 1)]
        if actions != expected:
            raise U2SProtocolError("U2-S exam history is not a contiguous prefix")
        decisions = value.get("practice_decisions", [])
        if (
            not isinstance(decisions, list)
            or not all(isinstance(decision, Mapping) for decision in decisions)
            or len(decisions) != len(normalized)
        ):
            raise U2SProtocolError("U2-S practice-decision history is invalid")
        return cls(
            exam_records=normalized,
            practice_decisions=[json.loads(json.dumps(decision)) for decision in decisions],
        )


@dataclass(frozen=True)
class ArmGrade:
    arm: ArmName
    eligible: bool
    checks: Mapping[str, bool]
    reasons: tuple[str, ...]
    final_exam_actions: tuple[int, ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "eligible": self.eligible,
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
            "final_exam_actions": list(self.final_exam_actions),
        }


def _lesson_metric(exam: Mapping[str, Any], lesson: lessons.LessonId) -> Mapping[str, Any]:
    lesson_values = exam.get("lessons")
    value = lesson_values.get(lesson.value) if isinstance(lesson_values, Mapping) else None
    if not isinstance(value, Mapping):
        raise U2SProtocolError(f"U2-S exam lacks {lesson.value}")
    return value


def existing_lesson_gate_passed(
    lesson: lessons.LessonId,
    value: Mapping[str, Any],
) -> bool:
    spec = lessons.LESSON_SPECS[lesson]
    panels = value.get("panel_successes")
    return (
        int(value.get("successes", -1)) >= int(spec.overall_required)
        and isinstance(panels, (list, tuple))
        and len(panels) == 2
        and all(int(panel) >= int(spec.panel_required) for panel in panels)
    )


def exam_stability_checks(exam: Mapping[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {
        "allocation_valid": exam.get("allocation_valid") is True,
        "practice_profile_normal": exam.get("practice_profile") == "normal",
    }
    for lesson in lessons.LessonId:
        value = _lesson_metric(exam, lesson)
        checks[f"{lesson.value}:existing_gate"] = existing_lesson_gate_passed(lesson, value)
    u2 = _lesson_metric(exam, lessons.LessonId.SEPARATED_UNLOCK)
    u2_panels = u2.get("panel_successes")
    checks["u2:successes_at_least_72"] = int(u2.get("successes", -1)) >= U2_STABILITY_SUCCESSES
    checks["u2:panels_at_least_34"] = (
        isinstance(u2_panels, (list, tuple))
        and len(u2_panels) == 2
        and all(int(panel) >= U2_STABILITY_PANEL_SUCCESSES for panel in u2_panels)
    )
    for lesson in (
        lessons.LessonId.VISIBLE_UNLOCK,
        lessons.LessonId.LOCAL_UNLOCK,
        lessons.LessonId.SEPARATED_UNLOCK,
    ):
        value = _lesson_metric(exam, lesson)
        ineffective = float(value.get("mean_ineffective_interactions", math.nan))
        checks[f"{lesson.value}:mean_ineffective_at_most_3"] = (
            math.isfinite(ineffective) and 0 <= ineffective <= INEFFECTIVE_MEAN_MAX
        )
    diagnostics = exam.get("case_diagnostics")
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    checks["cases:max_ineffective_below_10"] = (
        int(diagnostics.get("max_case_ineffective_interactions", -1))
        < MAX_CASE_INEFFECTIVE_EXCLUSIVE
        and int(diagnostics.get("max_case_ineffective_interactions", -1)) >= 0
    )
    checks["cases:no_ineffective_tail"] = (
        int(diagnostics.get("cases_with_ineffective_at_least_10", -1)) == 0
    )
    checks["cases:repeated_identical_interaction_run_below_10"] = (
        0
        <= int(
            diagnostics.get(
                "max_repeated_identical_interaction_run",
                -1,
            )
        )
        < MAX_IDENTICAL_INTERACTION_RUN_EXCLUSIVE
    )
    return checks


def grade_arm_terminal(
    arm: ArmName | str,
    exam_records: Sequence[Mapping[str, Any]],
) -> ArmGrade:
    selected = ArmName(arm)
    checks: dict[str, bool] = {
        "exact_exam_count": len(exam_records) == EXAM_COUNT,
        "terminal_boundary": bool(exam_records)
        and int(exam_records[-1].get("child_trained_actions", -1)) == CHILD_ACTION_BUDGET,
        "three_terminal_exams_available": len(exam_records) >= FINAL_STABILITY_EXAMS,
    }
    boundaries = [int(record.get("child_trained_actions", -1)) for record in exam_records]
    checks["contiguous_exam_boundaries"] = boundaries == [
        index * EVALUATION_INTERVAL for index in range(1, len(exam_records) + 1)
    ]
    terminal = tuple(exam_records[-FINAL_STABILITY_EXAMS:])
    for index, exam in enumerate(terminal, start=1):
        for label, passed in exam_stability_checks(exam).items():
            checks[f"terminal_{index}:{label}"] = passed
    reasons = tuple(label for label, passed in checks.items() if not passed)
    return ArmGrade(
        arm=selected,
        eligible=not reasons,
        checks=MappingProxyType(checks),
        reasons=reasons,
        final_exam_actions=tuple(
            int(record.get("child_trained_actions", -1)) for record in terminal
        ),
    )


def select_mechanism(
    arm_exam_records: Mapping[ArmName | str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Select only a prospective mechanism, never an ablation checkpoint."""

    normalized = {ArmName(name): tuple(records) for name, records in arm_exam_records.items()}
    if set(normalized) != set(ArmName):
        raise U2SProtocolError("mechanism selection requires all four terminal arms")
    grades = {arm: grade_arm_terminal(arm, normalized[arm]) for arm in ArmName}
    selected = next((arm for arm in ARM_PRIORITY if grades[arm].eligible), None)
    return {
        "protocol": PROTOCOL,
        "selection_kind": "mechanism_configuration_only",
        "selected_arm": selected.value if selected is not None else None,
        "selected_mechanism": (ARM_SPECS[selected].public_dict() if selected is not None else None),
        "ablation_checkpoint_reuse_authorized": False,
        "successor_cohort_authorized": selected is not None,
        "priority": [arm.value for arm in ARM_PRIORITY],
        "grades": {arm.value: grades[arm].public_dict() for arm in ARM_PRIORITY},
        "verdict": "mechanism_selected" if selected is not None else "ablation_failed",
    }


def effective_config(
    arm: ArmName | str,
    *,
    parent: frozen_u2.ParentProvenance,
    qualification: Any,
    exclusions: Any,
) -> dict[str, Any]:
    selected = ArmName(arm)
    spec = ARM_SPECS[selected]
    lr_schedule = ChildActionLinearSchedule()
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "development_only": True,
        "matched_ablation": True,
        "arm": spec.public_dict(),
        "parent": {
            "source_protocol": "dungeon-apprentice-v0.2-u1",
            "u1_child_seed": PARENT_U1_SEED,
            "checkpoint": parent.checkpoint,
            "checkpoint_sha256": parent.checkpoint_sha256,
            "optimizer_inherited": True,
            "lifetime_actions": parent.trained_timesteps,
        },
        "paired_randomness": {
            "algorithm_seed": ALGORITHM_SEED,
            "worker_streams": list(WORKER_STREAMS),
            "same_across_all_arms": True,
        },
        "budget": {
            "new_child_actions": CHILD_ACTION_BUDGET,
            "terminal_lifetime_actions": (PARENT_LIFETIME_ACTIONS + CHILD_ACTION_BUDGET),
            "rollout_transitions": ROLLOUT_TRANSITIONS,
            "exam_interval": EVALUATION_INTERVAL,
            "exam_count": EXAM_COUNT,
            "terminal_only_comparison": True,
        },
        "environment": {
            "size": 9,
            "workers": WORKERS,
            "generator_profile_version": lessons.GENERATOR_PROFILE_VERSION,
            "layout_resample_attempts": LAYOUT_RESAMPLE_ATTEMPTS,
            "mechanics_changed_from_u2": False,
        },
        "observation": {
            "kind": "egocentric_partial_rgb",
            "shape": [56, 56, 3],
            "recurrent_state_units": 256,
            "lesson_metadata": False,
            "privileged_state": False,
        },
        "reward": {
            "success": 1.0,
            "ordinary_step": -0.001,
            "pixel_novelty": 0.002,
            "episodic_curiosity_cap": 0.1,
            "no_effect": spec.public_dict()["no_effect_reward"],
            "timeout_is_terminal_failure": True,
            "curiosity_during_exams": False,
        },
        "optimization": {
            "rollout_steps": ROLLOUT_STEPS,
            "batch_size": BATCH_SIZE,
            "n_epochs": spec.n_epochs,
            "learning_rate": (
                lr_schedule.public_dict()
                if spec.conservative_ppo
                else {
                    "schedule": "constant",
                    "value": ORIGINAL_LEARNING_RATE,
                }
            ),
            "clip_range": spec.clip_range,
            "target_kl": spec.target_kl,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "ent_coef": ENTROPY_COEFFICIENT,
            "policy_kwargs": dict(frozen_u2.POLICY_KWARGS),
        },
        "lessons": [lesson.value for lesson in lessons.LessonId],
        "transition_target_profiles": lessons.target_profiles_public(),
        "training_layout_exclusions": _public_evidence(exclusions),
        "qualification_report_sha256": qualification.report_sha256,
        "evaluation": {
            "seeds_per_lesson": EVALUATION_SEED_COUNT,
            "panels": [40, 40],
            "final_stability_exams": FINAL_STABILITY_EXAMS,
            "existing_lesson_gates_required": True,
            "u2_successes_required": U2_STABILITY_SUCCESSES,
            "u2_panel_successes_required": U2_STABILITY_PANEL_SUCCESSES,
            "mean_ineffective_interactions_max": INEFFECTIVE_MEAN_MAX,
            "max_case_ineffective_interactions_exclusive": (MAX_CASE_INEFFECTIVE_EXCLUSIVE),
            "max_identical_interaction_run_exclusive": (MAX_IDENTICAL_INTERACTION_RUN_EXCLUSIVE),
        },
        "selection": {
            "priority": [arm.value for arm in ARM_PRIORITY],
            "mechanism_only": True,
            "checkpoint_reuse": False,
            "none_qualify_disposition": "no_successor_cohort_authorized",
        },
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise U2SProtocolError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise U2SProtocolError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise U2SProtocolError(f"{label} is missing or unsafe")
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise U2SProtocolError(f"{label} line {line_number} is not an object")
            records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise U2SProtocolError(f"cannot read {label}: {error}") from error
    return records


def _numbers_are_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_numbers_are_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_numbers_are_finite(item) for item in value)
    return True


def _audit_episode_ledger(run_directory: Path) -> dict[str, Any]:
    path = run_directory / "episode-starts.jsonl"
    records = _read_jsonl(path, "U2-S episode-start ledger")
    counts = Counter()
    for record in records:
        worker = int(record.get("worker_index", -1))
        if (
            record.get("type") != "episode_start"
            or worker not in range(WORKERS)
            or record.get("lesson_id") not in {lesson.value for lesson in lessons.LessonId}
            or not isinstance(record.get("episode_seed"), int)
            or not isinstance(record.get("episode_ordinal"), int)
            or int(record["episode_ordinal"]) < 1
        ):
            raise U2SProtocolError("U2-S episode-start record is invalid")
        for key in ("layout_sha256", "geometry_sha256"):
            digest = record.get(key)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise U2SProtocolError("U2-S episode-start digest is invalid")
        counts[worker] += 1
    if not records or set(counts) != set(range(WORKERS)):
        raise U2SProtocolError("U2-S episode ledger does not cover every worker")
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "record_count": len(records),
        "worker_record_counts": {str(worker): counts[worker] for worker in range(WORKERS)},
        "finite": _numbers_are_finite(records),
    }


def _audit_optimizer_ledger(
    run_directory: Path,
    arm: ArmName | str,
) -> dict[str, Any]:
    selected = ArmName(arm)
    path = run_directory / "optimizer.jsonl"
    records = _read_jsonl(path, "U2-S optimizer ledger")
    expected_count = CHILD_ACTION_BUDGET // ROLLOUT_TRANSITIONS
    expected_child_actions = [(index + 1) * ROLLOUT_TRANSITIONS for index in range(expected_count)]
    if (
        len(records) != expected_count
        or [int(record.get("child_trained_actions", -1)) for record in records]
        != expected_child_actions
        or not _numbers_are_finite(records)
    ):
        raise U2SProtocolError("U2-S optimizer ledger boundaries changed")
    total_completed = 0
    total_planned = 0
    for record in records:
        planned = int(record.get("epochs_planned", -1))
        completed = int(record.get("epochs_completed", -1))
        skipped = int(record.get("epochs_skipped", -1))
        if (
            planned != ARM_SPECS[selected].n_epochs
            or planned - completed != skipped
            or not optimizer_update_count_valid(
                selected,
                ROLLOUT_TRANSITIONS,
                completed,
            )
            or record.get("target_kl") != ARM_SPECS[selected].target_kl
            or record.get("kl_stop_triggered")
            != (ARM_SPECS[selected].target_kl is not None and skipped > 0)
        ):
            raise U2SProtocolError("U2-S optimizer epoch evidence changed")
        total_completed += completed
        total_planned += planned
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "record_count": len(records),
        "epochs_planned": total_planned,
        "epochs_completed": total_completed,
        "epochs_skipped": total_planned - total_completed,
        "epoch_counter_semantics": (
            "sb3_epoch_loops_entered; a target-KL loop may stop before all minibatches"
        ),
        "finite": True,
    }


def _audit_reward_ledger(
    run_directory: Path,
    arm: ArmName | str,
) -> dict[str, Any]:
    selected = ArmName(arm)
    path = run_directory / "reward-components.jsonl"
    records = _read_jsonl(path, "U2-S reward-component ledger")
    if not records or not _numbers_are_finite(records):
        raise U2SProtocolError("U2-S reward-component ledger is empty or nonfinite")
    penalty_enabled = ARM_SPECS[selected].no_effect_penalty
    for record in records:
        eligible = int(record.get("no_effect_eligible_event_count", -1))
        applied = int(record.get("no_effect_applied_penalty_count", -1))
        penalty_return = float(record.get("no_effect_penalty_return", math.nan))
        expected = max(
            -NO_EFFECT_EPISODE_CAP,
            applied * NO_EFFECT_PENALTY,
        )
        if (
            int(record.get("worker_index", -1)) not in range(WORKERS)
            or eligible < 0
            or not 0 <= applied <= NO_EFFECT_MAX_APPLIED_PER_EPISODE
            or applied > eligible
            or not math.isclose(
                penalty_return,
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or (not penalty_enabled and (eligible != 0 or applied != 0 or penalty_return != 0.0))
        ):
            raise U2SProtocolError("U2-S reward-component record is inconsistent")
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "record_count": len(records),
        "finite": True,
        "penalty_enabled": penalty_enabled,
        "eligible_events": sum(int(record["no_effect_eligible_event_count"]) for record in records),
        "applied_penalties": sum(
            int(record["no_effect_applied_penalty_count"]) for record in records
        ),
        "penalty_return": sum(float(record["no_effect_penalty_return"]) for record in records),
    }


def _audit_case_files(
    run_directory: Path,
    arm: ArmName | str,
    exam_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = ArmName(arm)
    files: list[dict[str, Any]] = []
    total_cases = 0
    for exam in exam_records:
        diagnostics = exam.get("case_diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise U2SProtocolError("U2-S exam case diagnostics are missing")
        relative = diagnostics.get("path")
        digest = diagnostics.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise U2SProtocolError("U2-S exam case-file identity is missing")
        path = (run_directory / relative).resolve()
        try:
            path.relative_to(run_directory.resolve())
        except ValueError as error:
            raise U2SProtocolError("U2-S case file escapes its run") from error
        document = _read_json(path, "U2-S exam case evidence")
        records = document.get("records")
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
            or
            file_sha256(path) != digest
            or document.get("schema_version") != CASE_EVIDENCE_SCHEMA_VERSION
            or document.get("protocol") != PROTOCOL
            or document.get("arm") != selected.value
            or document.get("checkpoint") != exam.get("checkpoint")
            or document.get("checkpoint_sha256") != exam.get("checkpoint_sha256")
            or document.get("checkpoint_sidecar_sha256")
            != exam.get("sidecar_sha256")
            or document.get("child_trained_actions") != exam.get("child_trained_actions")
            or not isinstance(records, list)
            or len(records) != len(lessons.LessonId) * EVALUATION_SEED_COUNT
            or not _numbers_are_finite(records)
        ):
            raise U2SProtocolError("U2-S exam case evidence changed")
        _verify_exam_case_records(exam, records, arm=selected)
        recomputed_diagnostics = _case_diagnostics_from_records(records)
        if (
            document.get("case_diagnostics") != recomputed_diagnostics
            or diagnostics
            != {
                **recomputed_diagnostics,
                "path": relative,
                "sha256": digest,
            }
        ):
            raise U2SProtocolError(
                "U2-S exam case diagnostics disagree with immutable cases"
            )
        total_cases += len(records)
        files.append(
            {
                "path": relative,
                "sha256": digest,
                "record_count": len(records),
            }
        )
    expected_cases = EXAM_COUNT * len(lessons.LessonId) * EVALUATION_SEED_COUNT
    if len(files) != EXAM_COUNT or total_cases != expected_cases:
        raise U2SProtocolError("U2-S terminal case inventory is incomplete")
    return {
        "file_count": len(files),
        "record_count": total_cases,
        "files": files,
        "inventory_sha256": _canonical_sha256(files),
        "finite": True,
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


def _verify_integrity(checkpoint: Path, sidecar: Path) -> Mapping[str, Any]:
    integrity_path = _integrity_path(checkpoint)
    if any(path.is_symlink() for path in (checkpoint, sidecar, integrity_path)):
        raise U2SProtocolError("U2-S checkpoint bundle may not contain symlinks")
    if not all(path.is_file() for path in (checkpoint, sidecar, integrity_path)):
        raise U2SProtocolError("U2-S checkpoint bundle is incomplete")
    value = _read_json(integrity_path, "U2-S checkpoint integrity")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol") != PROTOCOL
        or value.get("checkpoint") != checkpoint.name
        or value.get("checkpoint_sha256") != file_sha256(checkpoint)
        or value.get("sidecar") != sidecar.name
        or value.get("sidecar_sha256") != file_sha256(sidecar)
    ):
        raise U2SProtocolError("U2-S checkpoint integrity record does not match")
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _initial_rng_identity_contract() -> dict[str, Any]:
    """Return the frozen proof required for matched random numbers."""

    return {
        "captured_before_action_one": True,
        "algorithm_seed": ALGORITHM_SEED,
        "identical_across_all_arms": True,
        "required_component_digests": list(INITIAL_RNG_COMPONENT_DIGESTS),
        "nullable_component_digests": list(
            INITIAL_RNG_NULLABLE_COMPONENT_DIGESTS
        ),
        "workers": {
            "count": WORKERS,
            "indices": list(range(WORKERS)),
            "streams": list(WORKER_STREAMS),
            "required_component_digests": list(
                INITIAL_RNG_WORKER_COMPONENT_DIGESTS
            ),
        },
        "aggregate_sha256_required": True,
    }


def _rng_json_value(value: Any) -> Any:
    """Convert implementation RNG state into stable, finite JSON evidence."""

    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "type": "ndarray",
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "data_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return _rng_json_value(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _rng_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_rng_json_value(item) for item in value],
        }
    if isinstance(value, list):
        return [_rng_json_value(item) for item in value]
    if isinstance(value, bytes | bytearray | memoryview):
        raw = bytes(value)
        return {
            "type": "bytes",
            "length": len(raw),
            "data_sha256": hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(value, float):
        if not math.isfinite(value):
            raise U2SProtocolError("U2-S RNG evidence contains a non-finite float")
        return {"type": "float", "hex": value.hex()}
    if value is None or isinstance(value, str | int | bool):
        return value
    raise U2SProtocolError(
        f"U2-S RNG evidence has unsupported state type: {type(value).__name__}"
    )


def _rng_state_sha256(value: Any) -> str:
    return _canonical_sha256(_rng_json_value(value))


def _torch_rng_state_sha256(value: Any) -> str:
    try:
        array = value.detach().cpu().numpy()
    except (AttributeError, RuntimeError, TypeError) as error:
        raise U2SProtocolError("U2-S Torch RNG state is unreadable") from error
    return _rng_state_sha256(array)


def _space_rng_state_sha256(space: Any, label: str) -> str:
    generator = getattr(space, "np_random", None)
    bit_generator = getattr(generator, "bit_generator", None)
    state = getattr(bit_generator, "state", None)
    if not isinstance(state, Mapping):
        raise U2SProtocolError(f"U2-S {label} has no auditable NumPy RNG state")
    return _rng_state_sha256(state)


def seed_initial_rng_spaces(
    active_workers: Sequence[EpisodeEvidenceWrapper],
) -> None:
    """Remove Gymnasium's lazy operating-system entropy before first reset."""

    workers = sorted(active_workers, key=lambda worker: worker.worker_index)
    if (
        len(workers) != WORKERS
        or [worker.worker_index for worker in workers] != list(range(WORKERS))
        or [worker.worker_stream for worker in workers] != list(WORKER_STREAMS)
        or any(
            worker.episode_ordinal != 0
            or worker.local_transition_count != 0
            or worker._active is not None
            for worker in workers
        )
    ):
        raise U2SProtocolError(
            "U2-S RNG spaces must be seeded before the first worker reset"
        )
    for worker, stream in zip(workers, WORKER_STREAMS, strict=True):
        worker.action_space.seed(stream)
        worker.observation_space.seed(stream)


def _curriculum_environment(worker: EpisodeEvidenceWrapper) -> Any:
    current: Any = worker
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, lessons.CurriculumEnv):
            return current
        current = getattr(current, "env", None)
    raise U2SProtocolError("U2-S worker has no curriculum environment RNG")


def capture_initial_rng_identity(
    *,
    model: Any,
    scheduler: lessons.TransitionDeficitScheduler,
    active_workers: Sequence[EpisodeEvidenceWrapper],
) -> dict[str, Any]:
    """Seal all stochastic state after initialization and before action one."""

    workers = sorted(active_workers, key=lambda worker: worker.worker_index)
    if (
        len(workers) != WORKERS
        or [worker.worker_index for worker in workers] != list(range(WORKERS))
        or [worker.worker_stream for worker in workers] != list(WORKER_STREAMS)
        or any(
            worker.episode_ordinal != 1
            or worker.local_transition_count != 0
            or worker._active is None
            or worker._active.get("active") is not True
            or worker._active.get("elapsed_steps") != 0
            or worker._active.get("worker_transition_at_start") != 0
            for worker in workers
        )
    ):
        raise U2SProtocolError(
            "U2-S initial RNG identity was not captured before worker action one"
        )

    try:
        import torch
    except ImportError as error:
        raise U2SProtocolError(
            "U2-S initial RNG identity requires the training runtime"
        ) from error

    mps_state: str | None = None
    if bool(
        getattr(getattr(torch, "backends", None), "mps", None)
        and torch.backends.mps.is_available()
    ):
        try:
            mps_state = _torch_rng_state_sha256(torch.mps.get_rng_state())
        except (AttributeError, RuntimeError) as error:
            raise U2SProtocolError(
                "U2-S could not capture the available MPS RNG state"
            ) from error

    cuda_state: str | None = None
    if torch.cuda.is_available():
        try:
            cuda_state = _canonical_sha256(
                [
                    _torch_rng_state_sha256(state)
                    for state in torch.cuda.get_rng_state_all()
                ]
            )
        except RuntimeError as error:
            raise U2SProtocolError(
                "U2-S could not capture the available CUDA RNG state"
            ) from error

    worker_components: list[dict[str, Any]] = []
    for worker in workers:
        curriculum = _curriculum_environment(worker)
        generator = getattr(curriculum, "_rng", None)
        bit_generator = getattr(generator, "bit_generator", None)
        environment_state = getattr(bit_generator, "state", None)
        if not isinstance(environment_state, Mapping):
            raise U2SProtocolError(
                "U2-S worker environment has no auditable NumPy RNG state"
            )
        base_generator = getattr(curriculum.unwrapped, "np_random", None)
        base_bit_generator = getattr(base_generator, "bit_generator", None)
        base_environment_state = getattr(base_bit_generator, "state", None)
        if not isinstance(base_environment_state, Mapping):
            raise U2SProtocolError(
                "U2-S worker base environment has no auditable RNG state"
            )
        worker_components.append(
            {
                "worker_index": worker.worker_index,
                "worker_stream": worker.worker_stream,
                "environment_np_random_sha256": _rng_state_sha256(
                    {
                        "curriculum_generator": environment_state,
                        "base_environment_generator": (
                            base_environment_state
                        ),
                        "episode_position": worker.evidence_state(),
                    }
                ),
                "action_space_sha256": _space_rng_state_sha256(
                    worker.action_space,
                    f"worker {worker.worker_index} action space",
                ),
                "observation_space_sha256": _space_rng_state_sha256(
                    worker.observation_space,
                    f"worker {worker.worker_index} observation space",
                ),
            }
        )

    components = {
        "python_random_sha256": _rng_state_sha256(random.getstate()),
        "numpy_global_sha256": _rng_state_sha256(np.random.get_state()),
        "torch_cpu_sha256": _torch_rng_state_sha256(
            torch.random.get_rng_state()
        ),
        "torch_mps_sha256": mps_state,
        "torch_cuda_sha256": cuda_state,
        "model_action_space_sha256": _space_rng_state_sha256(
            model.action_space,
            "model action space",
        ),
        "scheduler_sha256": _rng_state_sha256(scheduler.state_dict()),
        "workers": worker_components,
    }
    identity = {
        "captured_before_action_one": True,
        "algorithm_seed": ALGORITHM_SEED,
        "components": components,
        "aggregate_sha256": _canonical_sha256(components),
    }
    return verify_initial_rng_identity(identity)


def verify_initial_rng_identity(
    value: Any,
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and return the complete canonical pre-action RNG identity."""

    if not isinstance(value, Mapping) or set(value) != {
        "captured_before_action_one",
        "algorithm_seed",
        "components",
        "aggregate_sha256",
    }:
        raise U2SProtocolError("U2-S initial RNG identity schema changed")
    components = value.get("components")
    expected_component_fields = set(INITIAL_RNG_COMPONENT_DIGESTS) | {
        "workers"
    }
    if (
        value.get("captured_before_action_one") is not True
        or value.get("algorithm_seed") != ALGORITHM_SEED
        or not isinstance(components, Mapping)
        or set(components) != expected_component_fields
    ):
        raise U2SProtocolError("U2-S initial RNG identity is incomplete")

    nullable = set(INITIAL_RNG_NULLABLE_COMPONENT_DIGESTS)
    for component_name in INITIAL_RNG_COMPONENT_DIGESTS:
        digest = components.get(component_name)
        if digest is None and component_name in nullable:
            continue
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise U2SProtocolError(
                f"U2-S initial RNG component is invalid: {component_name}"
            )

    raw_workers = components.get("workers")
    expected_worker_fields = {
        "worker_index",
        "worker_stream",
        *INITIAL_RNG_WORKER_COMPONENT_DIGESTS,
    }
    if not isinstance(raw_workers, list) or len(raw_workers) != WORKERS:
        raise U2SProtocolError("U2-S initial worker RNG inventory changed")
    for index, worker in enumerate(raw_workers):
        if (
            not isinstance(worker, Mapping)
            or set(worker) != expected_worker_fields
            or worker.get("worker_index") != index
            or worker.get("worker_stream") != WORKER_STREAMS[index]
        ):
            raise U2SProtocolError("U2-S initial worker RNG identity changed")
        for component_name in INITIAL_RNG_WORKER_COMPONENT_DIGESTS:
            digest = worker.get(component_name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in digest
                )
            ):
                raise U2SProtocolError(
                    "U2-S initial worker RNG component is invalid: "
                    f"{component_name}"
                )
    aggregate = value.get("aggregate_sha256")
    if (
        not isinstance(aggregate, str)
        or len(aggregate) != 64
        or any(character not in "0123456789abcdef" for character in aggregate)
        or aggregate != _canonical_sha256(components)
    ):
        raise U2SProtocolError("U2-S initial RNG aggregate changed")
    normalized = json.loads(json.dumps(value))
    if expected is not None:
        expected_normalized = verify_initial_rng_identity(expected)
        if normalized != expected_normalized:
            raise U2SProtocolError(
                "U2-S initial RNG identity differs from its frozen expectation"
            )
    return normalized


def _raw_space_rng_state(space: Any, label: str) -> dict[str, Any]:
    generator = getattr(space, "np_random", None)
    bit_generator = getattr(generator, "bit_generator", None)
    state = getattr(bit_generator, "state", None)
    if not isinstance(state, Mapping):
        raise U2SProtocolError(f"U2-S {label} has no restorable RNG state")
    return copy.deepcopy(dict(state))


def _snapshot_training_rng_state(
    *,
    model: Any,
    scheduler: lessons.TransitionDeficitScheduler,
    active_workers: Sequence[EpisodeEvidenceWrapper],
) -> dict[str, Any]:
    """Hold every captured RNG component across diagnostic evaluation."""

    try:
        import torch
    except ImportError as error:
        raise U2SProtocolError(
            "U2-S RNG preservation requires the training runtime"
        ) from error
    workers = sorted(active_workers, key=lambda worker: worker.worker_index)
    worker_states: list[dict[str, Any]] = []
    for worker in workers:
        curriculum = _curriculum_environment(worker)
        worker_states.append(
            {
                "curriculum": copy.deepcopy(
                    curriculum._rng.bit_generator.state
                ),
                "base_environment": copy.deepcopy(
                    curriculum.unwrapped.np_random.bit_generator.state
                ),
                "action_space": _raw_space_rng_state(
                    worker.action_space,
                    f"worker {worker.worker_index} action space",
                ),
                "observation_space": _raw_space_rng_state(
                    worker.observation_space,
                    f"worker {worker.worker_index} observation space",
                ),
            }
        )
    mps_state = (
        torch.mps.get_rng_state().clone()
        if bool(
            getattr(getattr(torch, "backends", None), "mps", None)
            and torch.backends.mps.is_available()
        )
        else None
    )
    cuda_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    return {
        "python_random": random.getstate(),
        "numpy_global": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_mps": mps_state,
        "torch_cuda": cuda_states,
        "model_action_space": _raw_space_rng_state(
            model.action_space,
            "model action space",
        ),
        "scheduler": copy.deepcopy(scheduler._rng.bit_generator.state),
        "workers": worker_states,
    }


def _restore_training_rng_state(
    snapshot: Mapping[str, Any],
    *,
    model: Any,
    scheduler: lessons.TransitionDeficitScheduler,
    active_workers: Sequence[EpisodeEvidenceWrapper],
) -> None:
    """Restore the complete pre-action RNG snapshot after diagnostics."""

    try:
        import torch
    except ImportError as error:
        raise U2SProtocolError(
            "U2-S RNG restoration requires the training runtime"
        ) from error
    workers = sorted(active_workers, key=lambda worker: worker.worker_index)
    raw_workers = snapshot.get("workers")
    if not isinstance(raw_workers, list) or len(raw_workers) != len(workers):
        raise U2SProtocolError("U2-S RNG restoration worker inventory changed")
    random.setstate(snapshot["python_random"])
    np.random.set_state(snapshot["numpy_global"])
    torch.random.set_rng_state(snapshot["torch_cpu"])
    if snapshot.get("torch_mps") is not None:
        torch.mps.set_rng_state(snapshot["torch_mps"])
    if snapshot.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(snapshot["torch_cuda"])
    model.action_space.np_random.bit_generator.state = copy.deepcopy(
        snapshot["model_action_space"]
    )
    scheduler._rng.bit_generator.state = copy.deepcopy(snapshot["scheduler"])
    for worker, raw in zip(workers, raw_workers, strict=True):
        if not isinstance(raw, Mapping):
            raise U2SProtocolError("U2-S RNG restoration worker state changed")
        curriculum = _curriculum_environment(worker)
        curriculum._rng.bit_generator.state = copy.deepcopy(
            raw["curriculum"]
        )
        curriculum.unwrapped.np_random.bit_generator.state = copy.deepcopy(
            raw["base_environment"]
        )
        worker.action_space.np_random.bit_generator.state = copy.deepcopy(
            raw["action_space"]
        )
        worker.observation_space.np_random.bit_generator.state = (
            copy.deepcopy(raw["observation_space"])
        )


def _common_learner_contract() -> dict[str, Any]:
    return {
        "environment": {
            "size": 9,
            "workers": WORKERS,
            "observation": {
                "kind": "egocentric_partial_rgb",
                "shape": [56, 56, 3],
                "recurrent_state_units": 256,
                "recurrent_layers": 1,
                "privileged_state": False,
            },
            "actions": 7,
        },
        "optimization": {
            "rollout_steps": ROLLOUT_STEPS,
            "rollout_transitions": ROLLOUT_TRANSITIONS,
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "ent_coef": ENTROPY_COEFFICIENT,
        },
        "reward": {
            "success": 1.0,
            "ordinary_step": -0.001,
            "pixel_novelty": 0.002,
            "episodic_curiosity_cap": 0.1,
            "timeout_is_terminal_failure": True,
            "curiosity_during_exams": False,
        },
        "normal_transition_targets": {
            "navigate/full": 0.5,
            "unlock/u0-visible": 0.075,
            "unlock/u1-local": 0.075,
            "unlock/u2-separated": 0.35,
        },
        "terminal_rule": {
            "final_exam_boundaries": [
                CHILD_ACTION_BUDGET - (FINAL_STABILITY_EXAMS - index - 1) * EVALUATION_INTERVAL
                for index in range(FINAL_STABILITY_EXAMS)
            ],
            "u2_successes_minimum": U2_STABILITY_SUCCESSES,
            "u2_panel_successes_minimum": U2_STABILITY_PANEL_SUCCESSES,
            "u0_u1_u2_mean_ineffective_maximum": INEFFECTIVE_MEAN_MAX,
            "max_case_ineffective_exclusive": MAX_CASE_INEFFECTIVE_EXCLUSIVE,
            "max_repeated_identical_interaction_run_exclusive": (
                MAX_IDENTICAL_INTERACTION_RUN_EXCLUSIVE
            ),
            "visible_no_effect_streak": ("reported_diagnostic_not_terminal_gate"),
        },
    }


def _contract_qualification(qualification: Any) -> dict[str, Any]:
    public = _public_evidence(qualification)
    guards = public.get("guard_sets")
    storage = public.get("storage_caps")
    initial_rng_identity = public.get("initial_rng_identity")
    deep_digest_fields = (
        "sampler_preflight_sha256",
        "arm_contract_sha256",
        "protected_partitions_sha256",
        "resume_contract_sha256",
        "storage_preflight_sha256",
    )
    if (
        public.get("report") != str(DEFAULT_QUALIFICATION_REPORT)
        or public.get("source_commit") is None
        or public.get("tag") != TRAINING_TAG
        or public.get("verdict") != "qualified"
        or not isinstance(public.get("tag_object"), str)
        or not isinstance(public.get("tag_payload_sha256"), str)
        or not isinstance(public.get("protocol_document_sha256"), str)
        or any(
            not isinstance(public.get(field), str)
            or len(public[field]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in public[field]
            )
            for field in deep_digest_fields
        )
        or not isinstance(guards, Mapping)
        or not isinstance(storage, Mapping)
        or not isinstance(initial_rng_identity, Mapping)
    ):
        raise U2SProtocolError("U2-S qualification public binding is incomplete")
    qualified_rng_identity = verify_initial_rng_identity(
        initial_rng_identity
    )
    normalized_guards: dict[str, dict[str, Any]] = {}
    expected_lessons = {lesson.value for lesson in lessons.LessonId}
    if set(guards) != expected_lessons:
        raise U2SProtocolError("U2-S qualification guard lessons changed")
    for lesson_id, raw in guards.items():
        if not isinstance(raw, Mapping):
            raise U2SProtocolError("U2-S qualification guard is invalid")
        count = raw.get("count", raw.get("exact_layouts"))
        digest = raw.get("sha256", raw.get("exact_layout_set_sha256"))
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise U2SProtocolError("U2-S qualification guard identity changed")
        normalized_guards[str(lesson_id)] = {
            "count": count,
            "sha256": digest,
            "rule": raw.get("rule"),
        }
    expected_caps = {
        "lineage_cap_bytes": LINEAGE_CAP_BYTES,
        "cohort_scientific_cap_bytes": COHORT_SCIENTIFIC_CAP_BYTES,
        "media_cap_bytes": MEDIA_CAP_BYTES,
        "combined_planned_cap_bytes": COMBINED_PLANNED_CAP_BYTES,
    }
    normalized_caps = {
        "lineage_cap_bytes": storage.get("lineage_cap_bytes", storage.get("per_arm_bytes")),
        "cohort_scientific_cap_bytes": storage.get(
            "cohort_scientific_cap_bytes",
            storage.get("scientific_cohort_bytes"),
        ),
        "media_cap_bytes": storage.get("media_cap_bytes", storage.get("media_bytes")),
        "combined_planned_cap_bytes": storage.get(
            "combined_planned_cap_bytes", storage.get("combined_bytes")
        ),
    }
    if normalized_caps != expected_caps:
        raise U2SProtocolError("U2-S qualification storage caps changed")
    return {
        "report": str(DEFAULT_QUALIFICATION_REPORT),
        "report_sha256": public["report_sha256"],
        "checksum": public["checksum"],
        "source_commit": public["source_commit"],
        "tag": TRAINING_TAG,
        "tag_object": public["tag_object"],
        "tag_payload_sha256": public["tag_payload_sha256"],
        "verdict": "qualified",
        "protocol_document": PROTOCOL_DOCUMENT,
        "protocol_document_sha256": public["protocol_document_sha256"],
        "guard_sets": {
            lesson_id: normalized_guards[lesson_id] for lesson_id in sorted(normalized_guards)
        },
        "guard_mapping_sha256": QUALIFIED_GUARD_MAPPING_SHA256,
        "u2r_terminal_active_records_sha256": (U2R_TERMINAL_ACTIVE_RECORDS_SHA256),
        "sampler_preflight_sha256": public["sampler_preflight_sha256"],
        "arm_contract_sha256": public["arm_contract_sha256"],
        "protected_partitions_sha256": public["protected_partitions_sha256"],
        "resume_contract_sha256": public["resume_contract_sha256"],
        "storage_preflight_sha256": public["storage_preflight_sha256"],
        "initial_rng_identity": qualified_rng_identity,
        "storage_caps": expected_caps,
    }


def _expected_cohort_contract(
    *,
    source_commit: str,
    qualification: Any,
) -> dict[str, Any]:
    binding = _contract_qualification(qualification)
    if binding["source_commit"] != source_commit:
        raise U2SProtocolError("U2-S contract source and qualification differ")
    labels = {
        ArmName.CONTROL: "A · Current PPO",
        ArmName.CONSERVATIVE: "B · Conservative PPO",
        ArmName.NO_EFFECT: "C · No-effect feedback",
        ArmName.COMBINED: "D · Conservative + feedback",
    }
    arms = []
    for arm in ARM_PRIORITY:
        conservative = ARM_SPECS[arm].conservative_ppo
        arms.append(
            {
                "id": arm.value,
                "label": labels[arm],
                "directory": arm.value,
                "media_directory": arm.value,
                "ppo_profile": "conservative" if conservative else "current",
                "no_effect_penalty": ARM_SPECS[arm].no_effect_penalty,
                "algorithm_seed": ALGORITHM_SEED,
                "worker_streams": list(WORKER_STREAMS),
                "action_cap": CHILD_ACTION_BUDGET,
                "intervention": ARM_SPECS[arm].public_dict(),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "cohort_id": COHORT_ID,
        "source": {"commit": source_commit, "dirty": False},
        "preregistration": {
            "tag": TRAINING_TAG,
            "tag_object": binding["tag_object"],
            "peeled_commit": source_commit,
            "tag_payload_sha256": binding["tag_payload_sha256"],
        },
        "parent": {
            "checkpoint": str(PARENT_CHECKPOINT),
            "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "sidecar": str(PARENT_CHECKPOINT.with_suffix(".json")),
            "sidecar_sha256": PARENT_SIDECAR_SHA256,
            "manifest": PARENT_MANIFEST,
            "manifest_sha256": PARENT_MANIFEST_SHA256,
            "policy_tensor_sha256": PARENT_POLICY_TENSOR_SHA256,
            "optimizer_state_sha256": PARENT_OPTIMIZER_STATE_SHA256,
            "lifetime_trained_actions": PARENT_LIFETIME_ACTIONS,
            "optimizer_updates": PARENT_OPTIMIZER_UPDATES,
        },
        "confirmation": {
            "report": CONFIRMATION_REPORT,
            "report_sha256": CONFIRMATION_REPORT_SHA256,
            "attempt": CONFIRMATION_ATTEMPT,
            "attempt_sha256": CONFIRMATION_ATTEMPT_SHA256,
            "checksum": CONFIRMATION_CHECKSUM,
            "checksum_sha256": CONFIRMATION_CHECKSUM_SHA256,
            "protocol": "dungeon-apprentice-v0.2-u1-confirmation-v2",
            "verdict": "confirmed",
        },
        "qualification": binding,
        "roots": {
            "cohort": str(CANONICAL_COHORT_ROOT),
            "media": str(CANONICAL_MEDIA_ROOT),
        },
        "matched_design": {
            "algorithm_seed": ALGORITHM_SEED,
            "worker_streams": list(WORKER_STREAMS),
            "initial_rng_identity": _initial_rng_identity_contract(),
            "action_cap_per_arm": CHILD_ACTION_BUDGET,
            "total_action_cap": len(ARM_PRIORITY) * CHILD_ACTION_BUDGET,
            "evaluation_every": EVALUATION_INTERVAL,
            "exam_episodes": EVALUATION_SEED_COUNT,
            "arm_order": [arm.value for arm in ARM_PRIORITY],
            "selection_priority": [arm.value for arm in ARM_PRIORITY],
            "terminal_stability_exams": FINAL_STABILITY_EXAMS,
            "checkpoint_promotable": False,
            "interruption": {
                "resumable": False,
                "cohort_disposition": "operationally_incomplete",
                "replacement_requires_all_four_fresh_arms": True,
                "replacement_requires_new_source_commit": True,
                "replacement_requires_new_annotated_tag": True,
                "replacement_requires_new_protocol_attempt_id": True,
                "replacement_requires_new_cohort_root": True,
            },
            "common_learner": _common_learner_contract(),
            "storage_caps": {
                "lineage_cap_bytes": LINEAGE_CAP_BYTES,
                "cohort_scientific_cap_bytes": COHORT_SCIENTIFIC_CAP_BYTES,
                "media_cap_bytes": MEDIA_CAP_BYTES,
                "combined_planned_cap_bytes": COMBINED_PLANNED_CAP_BYTES,
                "minimum_free_gib": int(MINIMUM_FREE_GIB),
            },
        },
        "arms": arms,
    }


def verify_cohort_contract(
    path: Path | None,
    *,
    source_commit: str,
    qualification: Any,
) -> dict[str, Any] | None:
    if path is None:
        return None
    contract_path = path.expanduser().resolve()
    expected_path = CANONICAL_COHORT_ROOT / "cohort-contract.json"
    if contract_path != expected_path or contract_path.is_symlink() or not contract_path.is_file():
        raise U2SProtocolError("U2-S cohort contract path is not canonical")
    value = _read_json(contract_path, "U2-S cohort contract")
    expected = _expected_cohort_contract(
        source_commit=source_commit,
        qualification=qualification,
    )
    if value != expected:
        raise U2SProtocolError("U2-S cohort contract differs from the complete frozen design")
    return {
        "path": str(contract_path),
        "sha256": file_sha256(contract_path),
        "contract": value,
    }


def verify_cohort_arm_paths(
    cohort_contract: Mapping[str, Any],
    *,
    arm: ArmName | str,
    run_directory: Path,
    media_directory: Path | None,
) -> None:
    selected = ArmName(arm)
    contract = cohort_contract.get("contract")
    if not isinstance(contract, Mapping):
        raise U2SProtocolError("U2-S cohort contract snapshot is missing")
    roots = contract.get("roots")
    arms = contract.get("arms")
    if not isinstance(roots, Mapping) or not isinstance(arms, list):
        raise U2SProtocolError("U2-S cohort roots or arms are missing")
    definition = next(
        (
            value
            for value in arms
            if isinstance(value, Mapping) and value.get("id") == selected.value
        ),
        None,
    )
    if not isinstance(definition, Mapping):
        raise U2SProtocolError("U2-S selected arm is absent from its contract")
    expected_run = (
        (Path(str(roots.get("cohort", ""))) / str(definition.get("directory", "")))
        .expanduser()
        .resolve()
    )
    expected_media = (
        (Path(str(roots.get("media", ""))) / str(definition.get("media_directory", "")))
        .expanduser()
        .resolve()
    )
    if run_directory.expanduser().resolve() != expected_run:
        raise U2SProtocolError("U2-S arm directory differs from its cohort contract")
    if media_directory is None or media_directory.expanduser().resolve() != expected_media:
        raise U2SProtocolError("U2-S media directory differs from its cohort contract")


def load_resume(
    *_args: Any,
    **_kwargs: Any,
) -> None:
    """Reject continuation of one arm in a matched factorial cohort.

    Resuming only the interrupted arm would move that arm onto a different
    environment and policy RNG trajectory while the other arms retain the
    original trajectory.  That destroys the controlled 2x2 comparison.  A
    durable interruption therefore invalidates this canonical cohort and a
    replacement attempt must restart all four arms from the confirmed parent.
    """

    raise U2SProtocolError("U2-S is non-resumable; restart all four arms in a new cohort root")


def _apply_arm_optimization(model: Any, arm: ArmName) -> None:
    spec = ARM_SPECS[arm]
    schedule: Callable[[float], float]
    if spec.conservative_ppo:
        schedule = ChildActionLinearSchedule()
    else:
        schedule = ConstantSchedule(ORIGINAL_LEARNING_RATE)
    model.learning_rate = schedule
    model.lr_schedule = schedule
    model.clip_range = ConstantSchedule(spec.clip_range)
    model.n_epochs = spec.n_epochs
    model.target_kl = spec.target_kl
    current_child_actions = min(
        CHILD_ACTION_BUDGET,
        max(0, int(model.num_timesteps) - PARENT_LIFETIME_ACTIONS),
    )
    current_rate = (
        schedule.for_child_actions(current_child_actions)
        if isinstance(schedule, ChildActionLinearSchedule)
        else schedule(0.0)
    )
    for group in model.policy.optimizer.param_groups:
        group["lr"] = current_rate


def _verify_arm_model(
    model: Any,
    arm: ArmName,
    *,
    expected_actions: int,
    expected_updates: int,
    expected_model_state: Mapping[str, Any] | None = None,
) -> None:
    spec = ARM_SPECS[arm]
    clip = model.clip_range(1.0) if callable(model.clip_range) else model.clip_range
    if (
        int(model.num_timesteps) != expected_actions
        or int(model._n_updates) != expected_updates
        or int(model.n_steps) != ROLLOUT_STEPS
        or int(model.batch_size) != BATCH_SIZE
        or int(model.n_epochs) != spec.n_epochs
        or float(model.gamma) != GAMMA
        or float(model.gae_lambda) != GAE_LAMBDA
        or float(model.ent_coef) != ENTROPY_COEFFICIENT
        or float(clip) != spec.clip_range
        or model.target_kl != spec.target_kl
        or int(model.action_space.n) != 7
        or tuple(model.observation_space.shape) != (3, 56, 56)
    ):
        raise U2SProtocolError("loaded model differs from the selected U2-S arm")
    if not getattr(model.policy.optimizer, "state", None):
        raise U2SProtocolError("U2-S model lost its inherited optimizer state")
    if expected_model_state is not None:
        measured = {
            "policy_tensor_sha256": state_digests.policy_tensor_sha256(model),
            "optimizer_state_sha256": state_digests.optimizer_state_sha256(model),
        }
        if measured != dict(expected_model_state):
            raise U2SProtocolError(
                "U2-S serialized model state differs from its sidecar"
            )


def _make_training_environment(
    *,
    scheduler: lessons.TransitionDeficitScheduler,
    seed: int,
    seed_access: Any,
    forbidden_layout_hashes: Mapping[lessons.LessonId, frozenset[str]],
    penalize_no_effect: bool,
) -> gym.Env:
    environment = lessons.make_training_env(
        scheduler=scheduler,
        seed=seed,
        size=9,
        seed_access=seed_access,
        forbidden_layout_hashes=forbidden_layout_hashes,
        max_layout_resample_attempts=LAYOUT_RESAMPLE_ATTEMPTS,
    )
    if penalize_no_effect:
        environment = NoEffectInteractionPenalty(environment)
    return environment


class _CallbackFactory:
    @staticmethod
    def create(base_callback: Any) -> type[Any]:
        frozen_base = frozen_u2._CallbackFactory.create(base_callback)

        class U2SCallback(frozen_base):
            def __init__(
                self,
                *,
                arm: ArmName,
                exclusions: Any,
                cohort_contract: Mapping[str, Any] | None,
                expected_initial_rng_identity: Mapping[str, Any],
                manifest: Mapping[str, Any],
                active_workers: Sequence[EpisodeEvidenceWrapper],
                **kwargs: Any,
            ) -> None:
                self.arm = arm
                self.arm_spec = ARM_SPECS[arm]
                self.exclusions = exclusions
                self.cohort_contract = (
                    dict(cohort_contract) if cohort_contract is not None else None
                )
                self.cohort_contract_sha256 = (
                    str(self.cohort_contract["sha256"])
                    if self.cohort_contract is not None
                    else None
                )
                self.expected_initial_rng_identity = (
                    verify_initial_rng_identity(
                        expected_initial_rng_identity
                    )
                )
                self.initial_rng_identity: dict[str, Any] | None = None
                self.manifest = json.loads(json.dumps(manifest))
                if "initial_rng_identity" in self.manifest:
                    raise U2SProtocolError(
                        "U2-S manifest claimed RNG identity before reset"
                    )
                self.active_workers = tuple(active_workers)
                if len(self.active_workers) != WORKERS or {
                    worker.worker_index for worker in self.active_workers
                } != set(range(WORKERS)):
                    raise U2SProtocolError("U2-S requires four episode-evidence workers")
                super().__init__(**kwargs)
                self._last_evaluation_child_actions = (
                    int(self.controller.exam_records[-1]["child_trained_actions"])
                    if self.controller.exam_records
                    else None
                )

            def _on_training_start(self) -> None:
                if (
                    int(self.model.num_timesteps) != self.trained_actions
                    or int(self.model._n_updates) != self.last_updates
                ):
                    raise U2SProtocolError("model and U2-S controller start boundaries differ")
                segment_index = int(self.segment["index"])
                if segment_index != 0 or self.child_trained_actions != 0:
                    raise U2SProtocolError(
                        "U2-S arms may only start once from the confirmed parent"
                    )
                measured_identity = capture_initial_rng_identity(
                    model=self.model,
                    scheduler=self.scheduler,
                    active_workers=self.active_workers,
                )
                verify_initial_rng_identity(
                    measured_identity,
                    expected=self.expected_initial_rng_identity,
                )
                self.initial_rng_identity = measured_identity
                self.manifest["initial_rng_identity"] = measured_identity
                atomic_write_json(
                    self.run_directory / "manifest.json",
                    self.manifest,
                )
                rng_snapshot = _snapshot_training_rng_state(
                    model=self.model,
                    scheduler=self.scheduler,
                    active_workers=self.active_workers,
                )
                try:
                    initial_archive = self._publish_live_archive(
                        "initial",
                        kind="initial",
                        resume_eligible=False,
                        replace=False,
                    )
                    self.latest_safe_checkpoint = initial_archive
                    self.latest_safe_trained_actions = self.trained_actions
                    self._evaluate_archive(
                        initial_archive,
                        trigger="diagnostic_parent_baseline",
                        decide=False,
                    )
                finally:
                    _restore_training_rng_state(
                        rng_snapshot,
                        model=self.model,
                        scheduler=self.scheduler,
                        active_workers=self.active_workers,
                    )
                post_diagnostic_identity = capture_initial_rng_identity(
                    model=self.model,
                    scheduler=self.scheduler,
                    active_workers=self.active_workers,
                )
                verify_initial_rng_identity(
                    post_diagnostic_identity,
                    expected=measured_identity,
                )
                self._write_status("training")

            def _on_rollout_start(self) -> None:
                self._guard_storage()
                if int(self.model._n_updates) > self.last_updates:
                    self._process_optimized_boundary()

            def _on_step(self) -> bool:
                keep_training = super()._on_step()
                for worker_index, (done, info) in enumerate(
                    zip(
                        self.locals.get("dones", ()),
                        self.locals.get("infos", ()),
                        strict=False,
                    )
                ):
                    if not done:
                        continue
                    extrinsic = float(info.get("extrinsic_return", 0.0) or 0.0)
                    curiosity = float(info.get("curiosity_return", 0.0) or 0.0)
                    penalty_return = float(info.get("u2s_no_effect_episode_total", 0.0) or 0.0)
                    eligible_count = int(
                        info.get(
                            "u2s_no_effect_eligible_event_count",
                            0,
                        )
                        or 0
                    )
                    applied_count = int(
                        info.get(
                            "u2s_no_effect_applied_penalty_count",
                            0,
                        )
                        or 0
                    )
                    expected_penalty = max(
                        -NO_EFFECT_EPISODE_CAP,
                        applied_count * NO_EFFECT_PENALTY,
                    )
                    if (
                        not all(
                            math.isfinite(value)
                            for value in (
                                extrinsic,
                                curiosity,
                                penalty_return,
                            )
                        )
                        or eligible_count < 0
                        or not 0 <= applied_count <= NO_EFFECT_MAX_APPLIED_PER_EPISODE
                        or applied_count > eligible_count
                        or (
                            not self.arm_spec.no_effect_penalty
                            and (eligible_count != 0 or applied_count != 0 or penalty_return != 0.0)
                        )
                        or (
                            self.arm_spec.no_effect_penalty
                            and not math.isclose(
                                penalty_return,
                                expected_penalty,
                                rel_tol=0.0,
                                abs_tol=1e-12,
                            )
                        )
                    ):
                        raise U2SProtocolError("U2-S reward-component evidence is invalid")
                    append_jsonl(
                        self.run_directory / "reward-components.jsonl",
                        {
                            "timestamp": utc_now(),
                            "arm": self.arm.value,
                            "worker_index": worker_index,
                            "child_collected_actions": (
                                int(self.model.num_timesteps) - PARENT_LIFETIME_ACTIONS
                            ),
                            "lesson_id": info.get("lesson_id"),
                            "seed": info.get("seed"),
                            "extrinsic_return": extrinsic,
                            "curiosity_return": curiosity,
                            "no_effect_penalty_return": expected_penalty,
                            "no_effect_eligible_event_count": eligible_count,
                            "no_effect_applied_penalty_count": applied_count,
                        },
                    )
                return keep_training

            def _sidecar(
                self,
                *,
                kind: str,
                checkpoint_sha256: str,
                resume_eligible: bool,
                exam_source: Mapping[str, Any] | None = None,
            ) -> dict[str, Any]:
                if resume_eligible:
                    raise U2SProtocolError("U2-S checkpoints cannot authorize single-arm resume")
                return {
                    "schema_version": SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "created_at": utc_now(),
                    "kind": kind,
                    "resume_eligible": False,
                    "resume_authorized": False,
                    "restart_scope_after_interruption": ("entire_four_arm_cohort"),
                    "checkpoint_sha256": checkpoint_sha256,
                    "arm": self.arm.value,
                    "source": dict(self.source),
                    "effective_config": dict(self.effective_config),
                    "parent": self.parent.public_dict(),
                    "qualification": self.qualification.public_dict(),
                    "exclusions": _public_evidence(self.exclusions),
                    "cohort_contract": self.cohort_contract,
                    "cohort_contract_sha256": self.cohort_contract_sha256,
                    "initial_rng_identity": self.initial_rng_identity,
                    "model_state": {
                        "policy_tensor_sha256": (state_digests.policy_tensor_sha256(self.model)),
                        "optimizer_state_sha256": (
                            state_digests.optimizer_state_sha256(self.model)
                        ),
                    },
                    "curriculum": self.state.public_dict(),
                    "controller": self.controller.public_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "last_completed_allocation": self.last_completed_allocation,
                    "last_completed_allocation_valid": (self.last_completed_allocation_valid),
                    "progress": {
                        "collected_actions": int(self.model.num_timesteps),
                        "trained_actions": self.trained_actions,
                        "lifetime_trained_actions": self.trained_actions,
                        "inherited_trained_actions": self.child_start_actions,
                        "child_trained_actions": self.child_trained_actions,
                        "remaining_child_actions": (
                            CHILD_ACTION_BUDGET - self.child_trained_actions
                        ),
                        "segment_trained_actions": (
                            self.trained_actions - self.segment_start_actions
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
                    "clip_fraction",
                    "explained_variance",
                    "loss",
                )
                self.latest_optimizer = {
                    name: (
                        float(values[f"train/{name}"])
                        if values.get(f"train/{name}") is not None
                        else None
                    )
                    for name in names
                }
                approx_kl = (
                    float(values["train/approx_kl"])
                    if values.get("train/approx_kl") is not None
                    else None
                )
                delta_actions = self.trained_actions - self.latest_safe_trained_actions
                epoch_evidence = optimizer_epoch_accounting(
                    self.arm,
                    delta_actions,
                    updates_since_boundary,
                    approx_kl=approx_kl,
                )
                self.latest_optimizer.update(epoch_evidence)
                append_jsonl(
                    self.run_directory / "optimizer.jsonl",
                    {
                        "timestamp": utc_now(),
                        "collected_actions": int(self.model.num_timesteps),
                        "trained_actions": self.trained_actions,
                        "child_trained_actions": self.child_trained_actions,
                        "optimizer_updates": int(self.model._n_updates),
                        "updates_since_boundary": updates_since_boundary,
                        **self.latest_optimizer,
                    },
                )

            def _process_optimized_boundary(self) -> None:
                collected = int(self.model.num_timesteps)
                updates = int(self.model._n_updates)
                delta_actions = collected - self.trained_actions
                delta_updates = updates - self.last_updates
                updates_valid = (
                    delta_actions > 0
                    and not delta_actions % ROLLOUT_TRANSITIONS
                    and optimizer_update_count_valid(
                        self.arm,
                        delta_actions,
                        delta_updates,
                    )
                )
                if delta_actions <= 0 or delta_actions % ROLLOUT_TRANSITIONS or not updates_valid:
                    raise U2SProtocolError("U2-S observed an incomplete optimizer boundary")
                self.trained_actions = collected
                self.last_updates = updates
                if not 0 <= self.child_trained_actions <= CHILD_ACTION_BUDGET:
                    raise U2SProtocolError("U2-S crossed its fixed action ceiling")
                self._record_optimizer_metrics(delta_updates)
                at_exam = self.trained_actions >= self.next_evaluation
                if at_exam:
                    if self.child_trained_actions % EVALUATION_INTERVAL:
                        raise U2SProtocolError("U2-S exam occurred away from its fixed boundary")
                    exam = self._publish_live_archive(
                        f"rolling/exam-{self.child_trained_actions:07d}",
                        kind="exam",
                        resume_eligible=False,
                        replace=False,
                    )
                    self.latest_exam_checkpoint = exam
                    self._evaluate_archive(exam, trigger="scheduled", decide=True)
                    self.next_evaluation = self._next_child_boundary(
                        self.trained_actions, EVALUATION_INTERVAL
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
                    raise U2SProtocolError(f"refusing to overwrite U2-S exam cases: {path}")
                diagnostics = _case_diagnostics_from_records(cases)
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

            def _update_recovery_controller(
                self,
                evaluations: Mapping[lessons.LessonId, Any],
                *,
                allocation_valid: bool,
            ) -> str:
                """Retain U2 prerequisite recovery without early mastery."""

                weak = tuple(lessons.weak_prerequisites(evaluations))
                in_recovery = bool(tuple(self.state.weak_prerequisites))
                self.state.mastered = False
                self.state.consecutive_passes = 0
                if in_recovery:
                    if weak:
                        changed = tuple(self.state.weak_prerequisites) != weak
                        self.state.weak_prerequisites = weak
                        self.state.recovery_passes = 0
                        decision = "recovery_changed" if changed else "recovery_continues"
                    elif not allocation_valid:
                        self.state.recovery_passes = 0
                        decision = "recovery_held_allocation_invalid"
                    else:
                        self.state.recovery_passes += 1
                        if self.state.recovery_passes < 2:
                            decision = "recovery_first_clean_boundary"
                        else:
                            self.state.weak_prerequisites = ()
                            self.state.recovery_passes = 0
                            decision = "recovery_completed"
                elif weak:
                    self.state.weak_prerequisites = weak
                    self.state.recovery_passes = 0
                    decision = "recovery_started"
                else:
                    self.state.recovery_passes = 0
                    decision = "normal_practice_continues"
                self.state.change()
                return decision

            def _evaluate_archive(
                self,
                exam: Path,
                *,
                trigger: str,
                decide: bool,
            ) -> None:
                sidecar = exam.with_suffix(".json")
                exam_sidecar = _read_json(sidecar, "U2-S exam sidecar")
                _verify_integrity(exam, sidecar)
                archive_before = file_sha256(exam)
                sidecar_before = file_sha256(sidecar)
                graded_model = self._load_exam_model(exam, exam_sidecar)
                allocation = self.scheduler.snapshot()
                allocation_transitions = sum(
                    int(value) for value in allocation.get("window_transitions", {}).values()
                )
                allocation_valid = not decide or (
                    allocation_transitions == EVALUATION_INTERVAL
                    and self.scheduler.allocation_within()
                )
                practice_profile = str(allocation.get("profile", "unknown"))
                evaluations: dict[lessons.LessonId, dict[str, Any]] = {}
                evaluation_objects: dict[lessons.LessonId, Any] = {}
                all_cases: list[dict[str, Any]] = []
                for lesson in lessons.LessonId:
                    result, cases = evaluate_lesson_with_visible_no_effect(
                        graded_model,
                        lesson,
                        self._validation_seeds(lesson),
                        frame_path=(
                            self.run_directory
                            / "frames"
                            / f"exam-{lesson.value.replace('/', '-')}.png"
                        ),
                        seed_access=self.validation_access,
                        penalty_enabled=self.arm_spec.no_effect_penalty,
                    )
                    evaluation_objects[lesson] = result
                    all_cases.extend(cases)
                    public = _augmented_lesson_evidence(
                        result,
                        cases,
                        penalty_enabled=self.arm_spec.no_effect_penalty,
                    )
                    evaluations[lesson] = public
                    append_jsonl(
                        self.run_directory / "evaluations.jsonl",
                        {
                            "timestamp": utc_now(),
                            "trigger": trigger,
                            "counts_toward_arm_stability": decide,
                            "arm": self.arm.value,
                            "trained_actions": self.trained_actions,
                            "child_trained_actions": self.child_trained_actions,
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
                    raise U2SProtocolError("U2-S evaluation changed immutable or live model state")
                if not decide and self.child_trained_actions == 0:
                    expected = frozen_u2.FROZEN_PARENTS[PARENT_CHILD_SEED].inherited_baseline
                    for lesson in (
                        lessons.LessonId.NAVIGATE,
                        lessons.LessonId.VISIBLE_UNLOCK,
                        lessons.LessonId.LOCAL_UNLOCK,
                    ):
                        if int(evaluations[lesson]["successes"]) != int(expected[lesson.value]):
                            raise U2SProtocolError("U2-S parent baseline failed exact reproduction")
                exam_record = {
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
                        raise U2SProtocolError("U2-S decision exams are not contiguous")
                    practice_decision = self._update_recovery_controller(
                        evaluation_objects,
                        allocation_valid=allocation_valid,
                    )
                    exam_record["practice_decision"] = practice_decision
                    self.controller.exam_records.append(exam_record)
                    self.controller.practice_decisions.append(
                        {
                            "child_trained_actions": self.child_trained_actions,
                            "window_profile": practice_profile,
                            "allocation_valid": allocation_valid,
                            "weak_prerequisites": [
                                lesson.value
                                for lesson in lessons.weak_prerequisites(evaluation_objects)
                            ],
                            "decision": practice_decision,
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
                    "child_trained_actions": self.child_trained_actions,
                    "trigger": trigger,
                    "counts_toward_arm_stability": decide,
                    "case_diagnostics": case_diagnostics,
                }
                self.evaluation_history.append(exam_record)
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": ("stability_exam" if decide else "baseline_evaluation"),
                        "arm": self.arm.value,
                        "child_trained_actions": self.child_trained_actions,
                        "checkpoint_sha256": archive_before,
                        "allocation_valid": allocation_valid,
                        "practice_profile": practice_profile,
                        "practice_decision": exam_record.get("practice_decision"),
                        "case_diagnostics": case_diagnostics,
                        "checks": exam_stability_checks(exam_record),
                    },
                )
                self.frame_revision += 1

            def _storage_status(self) -> Mapping[str, Any] | None:
                try:
                    return self.storage_guard(0) if self.storage_guard else None
                except BaseException as error:
                    return {"error": str(error)}

            def _write_status(self, phase: str) -> None:
                elapsed = max(0.0, time.monotonic() - self.wall_start)
                collected = int(self.model.num_timesteps)
                records = list(self.controller.exam_records)
                rolling_grade = (
                    grade_arm_terminal(self.arm, records).public_dict()
                    if len(records) == EXAM_COUNT
                    else None
                )
                schedule = ChildActionLinearSchedule()
                atomic_write_json(
                    self.run_directory / "status.json",
                    {
                        "protocol": PROTOCOL,
                        "phase": phase,
                        "arm": self.arm.value,
                        "started_at": self.started_at,
                        "updated_at": utc_now(),
                        "elapsed_seconds": elapsed,
                        "actions_per_second": (
                            (collected - self.segment_start_actions) / elapsed if elapsed else 0.0
                        ),
                        "collected_actions": (collected - PARENT_LIFETIME_ACTIONS),
                        "trained_actions": self.child_trained_actions,
                        "child_trained_actions": self.child_trained_actions,
                        "lifetime_collected_actions": collected,
                        "lifetime_trained_actions": self.trained_actions,
                        "remaining_action_budget": (
                            CHILD_ACTION_BUDGET - self.child_trained_actions
                        ),
                        "action_cap": CHILD_ACTION_BUDGET,
                        "optimizer_updates": int(self.model._n_updates),
                        "learning_rate": (
                            schedule.for_child_actions(self.child_trained_actions)
                            if self.arm_spec.conservative_ppo
                            else ORIGINAL_LEARNING_RATE
                        ),
                        "parent_checkpoint_sha256": self.parent.checkpoint_sha256,
                        "qualification_sha256": self.qualification.report_sha256,
                        "source": dict(self.source),
                        "source_commit": self.source.get("commit"),
                        "resume_authorized": False,
                        "cohort_outcome_if_interrupted": ("operationally_incomplete"),
                        "replacement_requires_all_four_fresh_arms": True,
                        "cohort_contract": self.cohort_contract,
                        "cohort_contract_sha256": self.cohort_contract_sha256,
                        "initial_rng_identity_sha256": (
                            self.initial_rng_identity["aggregate_sha256"]
                        ),
                        "exams_completed": len(records),
                        "evaluations": records,
                        "latest_evaluations": self.latest_evaluations,
                        "latest_evaluation": (records[-1] if records else None),
                        "evaluation_history": list(self.evaluation_history),
                        "terminal_grade": rolling_grade,
                        "terminal_eligible": (
                            rolling_grade["eligible"] if rolling_grade is not None else None
                        ),
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
                        "latest_optimizer": self.latest_optimizer,
                        "storage": self._storage_status(),
                        "segment": dict(self.segment),
                    },
                )

            def _pending_optimizer_delta(self) -> tuple[int, int, str]:
                delta_actions = int(self.model.num_timesteps) - self.trained_actions
                delta_updates = int(self.model._n_updates) - self.last_updates
                complete = (
                    delta_actions > 0
                    and not delta_actions % ROLLOUT_TRANSITIONS
                    and optimizer_update_count_valid(
                        self.arm,
                        delta_actions,
                        delta_updates,
                    )
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

            def _terminal_training_evidence(self) -> dict[str, Any]:
                workers = sorted(
                    (worker.evidence_state() for worker in self.active_workers),
                    key=lambda value: int(value["worker_index"]),
                )
                if (
                    len(workers) != WORKERS
                    or [int(worker["worker_index"]) for worker in workers] != list(range(WORKERS))
                    or not all(worker.get("active") is True for worker in workers)
                ):
                    raise U2SProtocolError("U2-S terminal active-worker evidence is incomplete")
                return {
                    "episode_starts": _audit_episode_ledger(self.run_directory),
                    "terminal_active_workers": workers,
                    "terminal_active_workers_sha256": _canonical_sha256(workers),
                }

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
                        "latest_observed_checkpoint": (
                            str(self.latest_safe_checkpoint.relative_to(self.run_directory))
                            if self.latest_safe_checkpoint
                            else None
                        ),
                    },
                )
                self._write_status(phase)

            def finalize(self, _phase: str) -> None:
                if int(self.model._n_updates) > self.last_updates:
                    self._process_optimized_boundary()
                updates = int(self.model._n_updates)
                updates_valid = optimizer_update_count_valid(
                    self.arm,
                    CHILD_ACTION_BUDGET,
                    updates,
                    parent_updates=self.parent.n_updates,
                )
                if (
                    int(self.model.num_timesteps) != self.trained_actions
                    or self.child_trained_actions != CHILD_ACTION_BUDGET
                    or self.trained_actions != PARENT_LIFETIME_ACTIONS + CHILD_ACTION_BUDGET
                    or not updates_valid
                    or len(self.controller.exam_records) != EXAM_COUNT
                    or self.latest_exam_checkpoint is None
                ):
                    raise U2SProtocolError("U2-S did not finish its exact optimized arm ceiling")
                grade = grade_arm_terminal(self.arm, self.controller.exam_records)
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
                self.source_guard()
                storage = self.storage_guard(0)
                training_evidence = self._terminal_training_evidence()
                optimizer_evidence = _audit_optimizer_ledger(
                    self.run_directory,
                    self.arm,
                )
                reward_evidence = _audit_reward_ledger(
                    self.run_directory,
                    self.arm,
                )
                case_evidence = _audit_case_files(
                    self.run_directory,
                    self.arm,
                    self.controller.exam_records,
                )
                terminal_sidecar = _read_json(
                    terminal.with_suffix(".json"),
                    "U2-S terminal checkpoint sidecar",
                )
                report = {
                    "schema_version": SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "arm": self.arm.value,
                    "verdict": ("stable_mechanism_candidate" if grade.eligible else "arm_failed"),
                    "development_only": True,
                    "successor_checkpoint_authorized": False,
                    "resume_authorized": False,
                    "restart_scope_after_interruption": ("entire_four_arm_cohort"),
                    "mechanism_selection_eligible": grade.eligible,
                    "grade": grade.public_dict(),
                    "progress": {
                        "child_trained_actions": self.child_trained_actions,
                        "lifetime_trained_actions": self.trained_actions,
                        "optimizer_updates": updates,
                        "exam_count": len(self.controller.exam_records),
                    },
                    "source": dict(self.source),
                    "source_clean_at_closeout": True,
                    "parent": self.parent.public_dict(),
                    "qualification": self.qualification.public_dict(),
                    "protected_seed_access": {
                        "confirmation_or_final_partitions_opened": False,
                        "bound_by_qualification": True,
                    },
                    "cohort_contract": self.cohort_contract,
                    "cohort_contract_sha256": self.cohort_contract_sha256,
                    "initial_rng_identity": self.initial_rng_identity,
                    "effective_config": dict(self.effective_config),
                    "optimizer_evidence": optimizer_evidence,
                    "reward_component_evidence": reward_evidence,
                    "training_episode_evidence": training_evidence,
                    "case_evidence": case_evidence,
                    "storage": storage,
                    "terminal_checkpoint": {
                        "path": str(terminal.relative_to(self.run_directory)),
                        "sha256": file_sha256(terminal),
                        "sidecar_sha256": file_sha256(terminal.with_suffix(".json")),
                        "integrity_sha256": file_sha256(_integrity_path(terminal)),
                        "model_state": terminal_sidecar["model_state"],
                    },
                    "exam_records": list(self.controller.exam_records),
                }
                report_path = self.run_directory / "report.json"
                atomic_write_json(report_path, report)
                atomic_write_json(
                    self.run_directory / "report.integrity.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "protocol": PROTOCOL,
                        "report": report_path.name,
                        "report_sha256": file_sha256(report_path),
                    },
                )
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": "run_terminal",
                        "arm": self.arm.value,
                        "phase": "completed",
                        "terminal_checkpoint_sha256": file_sha256(terminal),
                        "report_sha256": file_sha256(report_path),
                        "mechanism_selection_eligible": grade.eligible,
                    },
                )
                self.latest_safe_checkpoint = terminal
                self.latest_safe_trained_actions = self.trained_actions
                self._write_status("completed")

        return U2SCallback


def verify_terminal_report(
    run_directory: Path,
    *,
    expected_arm: ArmName | str,
    expected_source_commit: str,
    expected_cohort_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Deep-authenticate one completed arm before the cohort may advance."""

    selected = ArmName(expected_arm)
    run = run_directory.expanduser().resolve()
    if run.is_symlink() or not run.is_dir():
        raise U2SProtocolError("U2-S terminal run directory is unsafe")
    manifest = _read_json(run / "manifest.json", "U2-S arm manifest")
    report_path = run / "report.json"
    report_integrity_path = run / "report.integrity.json"
    report = _read_json(report_path, "U2-S terminal report")
    report_integrity = _read_json(
        report_integrity_path,
        "U2-S terminal report integrity",
    )
    if (
        report_integrity.get("schema_version") != SCHEMA_VERSION
        or report_integrity.get("protocol") != PROTOCOL
        or report_integrity.get("report") != report_path.name
        or report_integrity.get("report_sha256") != file_sha256(report_path)
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("protocol") != PROTOCOL
        or manifest.get("arm") != selected.value
        or report.get("schema_version") != SCHEMA_VERSION
        or report.get("protocol") != PROTOCOL
        or report.get("arm") != selected.value
        or report.get("development_only") is not True
        or report.get("successor_checkpoint_authorized") is not False
        or report.get("resume_authorized") is not False
        or report.get("restart_scope_after_interruption") != "entire_four_arm_cohort"
        or report.get("source_clean_at_closeout") is not True
        or report.get("source") != manifest.get("source")
        or report.get("source", {}).get("commit") != expected_source_commit
        or report.get("parent") != manifest.get("parent")
        or report.get("qualification") != manifest.get("qualification")
        or report.get("effective_config") != manifest.get("effective_config")
        or report.get("cohort_contract") != manifest.get("cohort_contract")
        or report.get("cohort_contract_sha256") != manifest.get("cohort_contract", {}).get("sha256")
        or report.get("initial_rng_identity")
        != manifest.get("initial_rng_identity")
    ):
        raise U2SProtocolError("U2-S terminal report provenance changed")
    initial_rng_identity = verify_initial_rng_identity(
        manifest.get("initial_rng_identity")
    )
    qualification_rng_identity = manifest.get("qualification", {}).get(
        "initial_rng_identity"
    )
    if (
        not isinstance(qualification_rng_identity, Mapping)
        or initial_rng_identity
        != verify_initial_rng_identity(qualification_rng_identity)
    ):
        raise U2SProtocolError(
            "U2-S terminal initial RNG identity differs from qualification"
        )
    contract_digest = str(report["cohort_contract_sha256"])
    if (
        expected_cohort_contract_sha256 is not None
        and contract_digest != expected_cohort_contract_sha256
    ):
        raise U2SProtocolError("U2-S terminal cohort-contract digest changed")
    contract_snapshot = report["cohort_contract"]
    contract_path = Path(str(contract_snapshot.get("path", "")))
    verified_contract = verify_cohort_contract(
        contract_path,
        source_commit=expected_source_commit,
        qualification=manifest["qualification"],
    )
    if verified_contract != contract_snapshot:
        raise U2SProtocolError("U2-S terminal cohort-contract snapshot changed")
    matched_design = contract_snapshot.get("contract", {}).get(
        "matched_design"
    )
    if (
        not isinstance(matched_design, Mapping)
        or matched_design.get("initial_rng_identity")
        != _initial_rng_identity_contract()
    ):
        raise U2SProtocolError(
            "U2-S terminal initial RNG contract changed"
        )
    verify_cohort_arm_paths(
        contract_snapshot,
        arm=selected,
        run_directory=run,
        media_directory=(CANONICAL_MEDIA_ROOT / selected.value),
    )

    progress = report.get("progress")
    if (
        not isinstance(progress, Mapping)
        or int(progress.get("child_trained_actions", -1)) != CHILD_ACTION_BUDGET
        or int(progress.get("lifetime_trained_actions", -1))
        != PARENT_LIFETIME_ACTIONS + CHILD_ACTION_BUDGET
        or int(progress.get("exam_count", -1)) != EXAM_COUNT
        or not optimizer_update_count_valid(
            selected,
            CHILD_ACTION_BUDGET,
            int(progress.get("optimizer_updates", -1)),
            parent_updates=PARENT_OPTIMIZER_UPDATES,
        )
    ):
        raise U2SProtocolError("U2-S terminal counters changed")
    exams = report.get("exam_records")
    if not isinstance(exams, list) or len(exams) != EXAM_COUNT:
        raise U2SProtocolError("U2-S terminal exam inventory is incomplete")
    boundaries = [
        int(exam.get("child_trained_actions", -1)) for exam in exams if isinstance(exam, Mapping)
    ]
    if boundaries != [(index + 1) * EVALUATION_INTERVAL for index in range(EXAM_COUNT)]:
        raise U2SProtocolError("U2-S terminal exams are not ordered")
    for exam in exams:
        checkpoint = (run / str(exam.get("checkpoint", ""))).resolve()
        try:
            checkpoint.relative_to(run)
        except ValueError as error:
            raise U2SProtocolError("U2-S exam checkpoint escapes its run") from error
        sidecar_path = checkpoint.with_suffix(".json")
        _verify_integrity(checkpoint, sidecar_path)
        sidecar = _read_json(sidecar_path, "U2-S exam checkpoint sidecar")
        if (
            file_sha256(checkpoint) != exam.get("checkpoint_sha256")
            or file_sha256(sidecar_path) != exam.get("sidecar_sha256")
            or sidecar.get("kind") != "exam"
            or sidecar.get("resume_eligible") is not False
            or sidecar.get("resume_authorized") is not False
            or sidecar.get("progress", {}).get("child_trained_actions")
            != exam.get("child_trained_actions")
            or sidecar.get("source") != report.get("source")
            or sidecar.get("cohort_contract_sha256") != contract_digest
            or sidecar.get("initial_rng_identity")
            != initial_rng_identity
        ):
            raise U2SProtocolError("U2-S exam checkpoint bundle changed")
    recomputed_grade = grade_arm_terminal(selected, exams).public_dict()
    if (
        report.get("grade") != recomputed_grade
        or report.get("mechanism_selection_eligible") is not recomputed_grade["eligible"]
        or report.get("verdict")
        != ("stable_mechanism_candidate" if recomputed_grade["eligible"] else "arm_failed")
    ):
        raise U2SProtocolError("U2-S terminal grade does not recompute")

    case_evidence = _audit_case_files(run, selected, exams)
    optimizer_evidence = _audit_optimizer_ledger(run, selected)
    reward_evidence = _audit_reward_ledger(run, selected)
    episode_evidence = _audit_episode_ledger(run)
    if (
        report.get("case_evidence") != case_evidence
        or report.get("optimizer_evidence") != optimizer_evidence
        or report.get("reward_component_evidence") != reward_evidence
        or int(optimizer_evidence["epochs_completed"])
        != int(progress["optimizer_updates"]) - PARENT_OPTIMIZER_UPDATES
    ):
        raise U2SProtocolError("U2-S terminal ledger binding changed")
    training = report.get("training_episode_evidence")
    active = training.get("terminal_active_workers") if isinstance(training, Mapping) else None
    if (
        not isinstance(training, Mapping)
        or training.get("episode_starts") != episode_evidence
        or not isinstance(active, list)
        or len(active) != WORKERS
        or training.get("terminal_active_workers_sha256") != _canonical_sha256(active)
        or [int(worker.get("worker_index", -1)) for worker in active] != list(range(WORKERS))
    ):
        raise U2SProtocolError("U2-S terminal active-worker evidence changed")
    start_records = _read_jsonl(
        run / str(episode_evidence["path"]),
        "U2-S episode-start ledger",
    )
    identity_fields = (
        "worker_index",
        "worker_stream",
        "episode_ordinal",
        "episode_seed",
        "lesson_id",
        "layout_sha256",
        "geometry_sha256",
    )
    for worker in active:
        if worker.get("active") is not True or not any(
            all(record.get(field) == worker.get(field) for field in identity_fields)
            for record in start_records
        ):
            raise U2SProtocolError("U2-S terminal worker is absent from its episode ledger")

    terminal_record = report.get("terminal_checkpoint")
    if not isinstance(terminal_record, Mapping):
        raise U2SProtocolError("U2-S terminal checkpoint identity is missing")
    terminal = (run / str(terminal_record.get("path", ""))).resolve()
    try:
        terminal.relative_to(run)
    except ValueError as error:
        raise U2SProtocolError("U2-S terminal checkpoint escapes its run") from error
    sidecar_path = terminal.with_suffix(".json")
    integrity_path = _integrity_path(terminal)
    _verify_integrity(terminal, sidecar_path)
    sidecar = _read_json(sidecar_path, "U2-S terminal checkpoint sidecar")
    if (
        file_sha256(terminal) != terminal_record.get("sha256")
        or file_sha256(sidecar_path) != terminal_record.get("sidecar_sha256")
        or file_sha256(integrity_path) != terminal_record.get("integrity_sha256")
        or sidecar.get("kind") != "terminal"
        or sidecar.get("resume_eligible") is not False
        or sidecar.get("resume_authorized") is not False
        or sidecar.get("arm") != selected.value
        or sidecar.get("source") != report.get("source")
        or sidecar.get("parent") != report.get("parent")
        or sidecar.get("qualification") != report.get("qualification")
        or sidecar.get("effective_config") != report.get("effective_config")
        or sidecar.get("cohort_contract_sha256") != contract_digest
        or sidecar.get("initial_rng_identity") != initial_rng_identity
        or sidecar.get("progress", {}).get("child_trained_actions") != CHILD_ACTION_BUDGET
        or sidecar.get("progress", {}).get("lifetime_trained_actions")
        != PARENT_LIFETIME_ACTIONS + CHILD_ACTION_BUDGET
        or sidecar.get("model_state") != terminal_record.get("model_state")
    ):
        raise U2SProtocolError("U2-S terminal checkpoint bundle changed")
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError as error:
        raise U2SProtocolError(
            "U2-S terminal model verifier requires training dependencies"
        ) from error
    model = RecurrentPPO.load(terminal, device="cpu")
    try:
        _apply_arm_optimization(model, selected)
        _verify_arm_model(
            model,
            selected,
            expected_actions=PARENT_LIFETIME_ACTIONS + CHILD_ACTION_BUDGET,
            expected_updates=int(progress["optimizer_updates"]),
            expected_model_state=terminal_record["model_state"],
        )
    finally:
        del model

    protected = report.get("protected_seed_access")
    repository = Path(__file__).resolve().parents[2]
    closing_source = git_snapshot(repository)
    if (
        not isinstance(protected, Mapping)
        or protected.get("confirmation_or_final_partitions_opened") is not False
        or protected.get("bound_by_qualification") is not True
        or not isinstance(report.get("storage"), Mapping)
        or closing_source.get("dirty") is not False
        or closing_source.get("commit") != expected_source_commit
    ):
        raise U2SProtocolError("U2-S terminal operational audit is incomplete")
    return {
        "arm": selected.value,
        "verdict": report["verdict"],
        "mechanism_selection_eligible": bool(report["mechanism_selection_eligible"]),
        "child_trained_actions": CHILD_ACTION_BUDGET,
        "lifetime_trained_actions": (PARENT_LIFETIME_ACTIONS + CHILD_ACTION_BUDGET),
        "optimizer_updates": int(progress["optimizer_updates"]),
        "exam_count": EXAM_COUNT,
        "case_count": int(case_evidence["record_count"]),
        "case_evidence_sha256": case_evidence["inventory_sha256"],
        "case_evidence_files": list(case_evidence["files"]),
        "initial_rng_identity": initial_rng_identity,
        "qualification_sha256": report["qualification"]["report_sha256"],
        "terminal_checkpoint_sha256": terminal_record["sha256"],
        "terminal_sidecar_sha256": terminal_record["sidecar_sha256"],
        "terminal_integrity_sha256": terminal_record["integrity_sha256"],
        "report_sha256": file_sha256(report_path),
        "report_integrity_sha256": file_sha256(report_integrity_path),
    }


def _best_effort(label: str, operation: Callable[[], Any]) -> None:
    try:
        operation()
    except BaseException as error:
        print(f"Warning: {label} failed during U2-S finalization: {error}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train-arm")
    train.add_argument("--arm", choices=[arm.value for arm in ArmName], required=True)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--media-dir", type=Path)
    train.add_argument("--cohort-contract", type=Path, required=True)
    train.add_argument(
        "--qualification-report",
        type=Path,
        default=DEFAULT_QUALIFICATION_REPORT,
    )
    train.add_argument("--device", choices=("cpu",), default="cpu")
    return parser


def _prepare_run_directory(path: Path) -> Path:
    run = path.expanduser().resolve()
    if run.exists() or run.is_symlink():
        raise U2SProtocolError("refusing to reuse a U2-S arm directory")
    run.mkdir(parents=True)
    return run


def _validate_parent_model_before_arm(model: Any, parent: frozen_u2.ParentProvenance) -> None:
    learning_rate = model.learning_rate
    if callable(learning_rate):
        learning_rate = learning_rate(1.0)
    clip = model.clip_range(1.0) if callable(model.clip_range) else model.clip_range
    if (
        int(model.num_timesteps) != parent.trained_timesteps
        or int(model._n_updates) != parent.n_updates
        or int(model.n_steps) != ROLLOUT_STEPS
        or int(model.batch_size) != BATCH_SIZE
        or int(model.n_epochs) != ORIGINAL_EPOCHS
        or float(learning_rate) != ORIGINAL_LEARNING_RATE
        or float(clip) != ORIGINAL_CLIP_RANGE
        or model.target_kl is not None
        or not getattr(model.policy.optimizer, "state", None)
    ):
        raise U2SProtocolError("authenticated U1 parent does not carry the expected PPO optimizer")


def train_arm(args: argparse.Namespace) -> None:
    arm = ArmName(args.arm)
    if getattr(args, "resume", None) is not None:
        raise SystemExit("U2-S is non-resumable; a replacement cohort must restart all four arms")
    repository = Path(__file__).resolve().parents[2]
    source = git_snapshot(repository)
    if (
        source.get("dirty") is not False
        or not isinstance(source.get("commit"), str)
        or not source["commit"]
    ):
        raise SystemExit("U2-S requires one clean identified source commit")
    if (
        PARENT_CHECKPOINT.expanduser().resolve().is_symlink()
        or file_sha256(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256
    ):
        raise SystemExit("U2-S confirmed U1 parent checkpoint changed")

    parent = frozen_u2.verify_parent(
        PARENT_CHECKPOINT,
        frozen_u2.CANONICAL_U1_CONFIRMATION,
        child_seed=PARENT_CHILD_SEED,
    )
    if (
        parent.u1_child_seed != PARENT_U1_SEED
        or parent.checkpoint_sha256 != PARENT_CHECKPOINT_SHA256
        or parent.trained_timesteps != PARENT_LIFETIME_ACTIONS
    ):
        raise SystemExit("U2-S parent identity differs from its prospective contract")

    from dungeon_apprentice.v02_u2s_qualify import (
        verify_u2s_qualification,
    )

    qualification = verify_u2s_qualification(
        args.qualification_report,
        expected_source_commit=str(source["commit"]),
        repository=repository,
    )
    seed_access = qualification.seed_access()
    forbidden_layout_hashes = qualification.forbidden_layout_hashes()
    qualification_public = qualification.public_dict()
    qualification_report = qualification.verified_report()
    exclusion_evidence = qualification_report["guards"]
    if qualification_public.get("guard_sets") != exclusion_evidence.get("applied_by_lesson"):
        raise SystemExit("U2-S qualification guard inventory changed")
    sampler_preflight = frozen_u2r.preflight_u2r_training_layout_sampler(
        forbidden_layout_hashes,
        seed_access=seed_access,
        worker_streams=WORKER_STREAMS,
        max_attempts=LAYOUT_RESAMPLE_ATTEMPTS,
    )
    config = effective_config(
        arm,
        parent=parent,
        qualification=qualification,
        exclusions=exclusion_evidence,
    )
    cohort_contract = verify_cohort_contract(
        args.cohort_contract,
        source_commit=str(source["commit"]),
        qualification=qualification,
    )
    if cohort_contract is None:
        raise SystemExit("U2-S requires its immutable cohort contract")
    media_directory = args.media_dir.expanduser().resolve() if args.media_dir is not None else None
    verify_cohort_arm_paths(
        cohort_contract,
        arm=arm,
        run_directory=args.run_dir,
        media_directory=media_directory,
    )
    run_directory = _prepare_run_directory(args.run_dir)
    child_trained = 0
    remaining = CHILD_ACTION_BUDGET
    minimum_free_bytes = int(MINIMUM_FREE_GIB * 1024**3)
    ensure_disk_space(run_directory, minimum_free_bytes)

    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    except ImportError as error:
        raise SystemExit('Install training dependencies with: pip install -e ".[train]"') from error

    source_checkpoint = Path(parent.checkpoint)
    model = RecurrentPPO.load(source_checkpoint, device=args.device)
    _validate_parent_model_before_arm(model, parent)
    _apply_arm_optimization(model, arm)
    expected_actions = parent.trained_timesteps
    expected_updates = parent.n_updates
    _verify_arm_model(
        model,
        arm,
        expected_actions=expected_actions,
        expected_updates=expected_updates,
        expected_model_state=None,
    )
    del model

    state = lessons.CurriculumState()
    controller = AblationController()
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
        "start_child_trained_actions": child_trained,
        "remaining_child_actions": remaining,
        "resume_authorized": False,
        "resume_checkpoint": None,
        "resume_checkpoint_sha256": None,
    }

    def storage_guard(anticipated_lineage_bytes: int = 0) -> Mapping[str, Any]:
        return audit_u2_storage(
            storage_root=run_directory.parents[2],
            cohort_directory=run_directory.parent,
            lineage_directory=run_directory,
            media_directory=media_directory,
            protected_paths=(
                Path(parent.checkpoint),
                Path(parent.confirmation_report),
                Path(qualification.report),
                frozen_u2r.U2_CONFIRMATION_REPORT,
            ),
            anticipated_lineage_bytes=anticipated_lineage_bytes,
        ).as_dict()

    def source_guard() -> None:
        current = git_snapshot(repository)
        if current.get("dirty") is not False or current.get("commit") != source["commit"]:
            raise U2SProtocolError("U2-S source became dirty or changed")

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
        "qualification": qualification.public_dict(),
        "exclusions": _public_evidence(exclusion_evidence),
        "layout_sampler_preflight": [dict(item) for item in sampler_preflight],
        "cohort_contract": cohort_contract,
        "effective_config": config,
        "segment": segment,
        "information_boundary": (
            "56x56x3 partial RGB pixels plus private 256-unit recurrent state"
        ),
        "online_model_calls": False,
        "demonstrations": False,
        "oracle_actions_used_for_training": False,
        "confirmation_seed_access": False,
        "ablation_checkpoint_reuse_authorized": False,
        "resume_authorized": False,
        "restart_scope_after_interruption": "entire_four_arm_cohort",
    }

    evidence_environments: list[EpisodeEvidenceWrapper] = []
    factories: list[Callable[[], gym.Env]] = []
    for worker_index, stream in enumerate(WORKER_STREAMS):

        def factory(
            index: int = worker_index,
            worker_stream: int = stream,
        ) -> gym.Env:
            environment = _make_training_environment(
                scheduler=scheduler,
                seed=worker_stream,
                seed_access=seed_access,
                forbidden_layout_hashes=forbidden_layout_hashes,
                penalize_no_effect=ARM_SPECS[arm].no_effect_penalty,
            )
            recorder = EpisodeEvidenceWrapper(
                environment,
                worker_index=index,
                worker_stream=worker_stream,
                ledger=run_directory / "episode-starts.jsonl",
            )
            evidence_environments.append(recorder)
            return recorder

        factories.append(factory)

    callback: Any | None = None
    vector_environment: Any | None = None
    setup_start = time.monotonic()
    try:
        vector_environment = VecTransposeImage(DummyVecEnv(factories))
        model = RecurrentPPO.load(
            source_checkpoint,
            env=vector_environment,
            device=args.device,
        )
        _apply_arm_optimization(model, arm)
        _verify_arm_model(
            model,
            arm,
            expected_actions=expected_actions,
            expected_updates=expected_updates,
            expected_model_state=None,
        )
        qualified_rng_identity = qualification.public_dict().get(
            "initial_rng_identity"
        )
        if not isinstance(qualified_rng_identity, Mapping):
            raise U2SProtocolError(
                "U2-S qualification has no initial RNG identity"
            )
        expected_initial_rng_identity = verify_initial_rng_identity(
            qualified_rng_identity
        )
        callback_type = _CallbackFactory.create(BaseCallback)
        callback = callback_type(
            arm=arm,
            exclusions=exclusion_evidence,
            cohort_contract=cohort_contract,
            expected_initial_rng_identity=expected_initial_rng_identity,
            manifest=manifest,
            active_workers=evidence_environments,
            run_directory=run_directory,
            state=state,
            controller=controller,
            scheduler=scheduler,
            parent=parent,
            qualification=qualification,
            source=source,
            segment=segment,
            effective_config=config,
            exam_model_loader=lambda path: RecurrentPPO.load(path, device="cpu"),
            validation_access=seed_access,
            evaluation_interval=EVALUATION_INTERVAL,
            frame_interval=2_048,
            minimum_free_bytes=minimum_free_bytes,
            started_at=started_at,
            initial_trained_actions=expected_actions,
            initial_updates=expected_updates,
            child_start_actions=PARENT_LIFETIME_ACTIONS,
            keep_rolling_exams=EXAM_COUNT,
            storage_guard=storage_guard,
            source_guard=source_guard,
            staging_directory=run_directory.parent / ".u2s-staging",
        )
        # This is the final RNG-mutating initialization before SB3 resets all
        # workers and invokes the pre-action callback.  Callback construction
        # is deliberately outside the matched-random-number boundary.
        model.set_random_seed(ALGORITHM_SEED)
        seed_initial_rng_spaces(evidence_environments)
        print(f"U2-S {arm.value} artifacts: {run_directory}", flush=True)
        model.learn(
            total_timesteps=remaining,
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
                    "setup_elapsed_seconds": time.monotonic() - setup_start,
                    "traceback": failure_traceback,
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
            for environment in reversed(evidence_environments):
                _best_effort("environment close", environment.close)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train-arm":
        train_arm(args)
        return
    raise SystemExit(f"unsupported U2-S command: {args.command}")


if __name__ == "__main__":
    main()
