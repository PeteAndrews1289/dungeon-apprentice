"""Frozen no-update confirmation for the three mastered Local Unlock policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dungeon_apprentice.artifacts import (
    atomic_write_json,
    file_sha256,
    git_snapshot,
    runtime_snapshot,
    utc_now,
)
from dungeon_apprentice.contracts import PIXEL_SHAPE, STEP_REWARD, SUCCESS_REWARD
from dungeon_apprentice.oracle import DungeonOracle, OracleFailure
from dungeon_apprentice.v02_lessons import (
    ALLOCATION_TOLERANCE,
    FULL_LAYOUT_UNIQUENESS_FLOOR,
    GENERATOR_PROFILE_VERSION,
    GEOMETRY_UNIQUENESS_FLOOR,
    PROTOCOL,
    CurriculumState,
    LessonEnv,
    LessonEvaluation,
    LessonId,
    RecoveryCause,
    evaluate_lesson,
    make_pixel_env,
    qualify_local_unlock,
    validation_seeds,
)
from dungeon_apprentice.v02_u1 import (
    CHECKPOINT_SCHEMA_VERSION,
    CHILD_ACTION_BUDGET,
    EVALUATION_INTERVAL,
    ParentProvenance,
    U1ProtocolError,
    verify_parent,
)

CONFIRMATION_PROTOCOL = "dungeon-apprentice-v0.2-u1-confirmation"
CONFIRMATION_SCHEMA_VERSION = 1
CONFIRMATION_CASES = 200
CONFIRMATION_PANEL_SIZE = 100
U1_CONFIRMATION_BASE = 15_020_000
NAVIGATE_CONFIRMATION_BASE = 15_030_000
VISIBLE_UNLOCK_CONFIRMATION_BASE = 15_040_000
ANCHOR_UNIQUENESS_FLOOR = 0.80
U1_FULL_LAYOUT_UNIQUENESS_FLOOR = 0.99
U1_GEOMETRY_UNIQUENESS_FLOOR = 0.95
CONFIRMATION_GATES: Mapping[LessonId, tuple[float, float]] = {
    LessonId.NAVIGATE: (0.85, 0.85),
    LessonId.VISIBLE_UNLOCK: (0.85, 0.80),
    LessonId.LOCAL_UNLOCK: (0.85, 0.80),
}
FROZEN_SEED_PARTITIONS: tuple[dict[str, Any], ...] = (
    {"name": "training", "start": 0, "end": 999_999, "role": "training"},
    {
        "name": "u1_engineering_qualification",
        "start": 5_100_000,
        "end": 5_100_999,
        "role": "qualification",
    },
    {
        "name": "navigate_validation",
        "start": 10_000_000,
        "end": 10_000_079,
        "role": "validation",
    },
    {
        "name": "u0_validation",
        "start": 11_000_000,
        "end": 11_000_079,
        "role": "validation",
    },
    {
        "name": "u1_validation",
        "start": 11_100_000,
        "end": 11_100_079,
        "role": "validation",
    },
    {
        "name": "prior_u0_navigate_confirmation",
        "start": 15_000_000,
        "end": 15_000_199,
        "role": "prior_confirmation",
    },
    {
        "name": "prior_u0_visible_unlock_confirmation",
        "start": 15_010_000,
        "end": 15_010_199,
        "role": "prior_confirmation",
    },
    {
        "name": "new_u1_confirmation",
        "start": U1_CONFIRMATION_BASE,
        "end": U1_CONFIRMATION_BASE + CONFIRMATION_CASES - 1,
        "role": "new_confirmation",
    },
    {
        "name": "new_navigate_confirmation",
        "start": NAVIGATE_CONFIRMATION_BASE,
        "end": NAVIGATE_CONFIRMATION_BASE + CONFIRMATION_CASES - 1,
        "role": "new_confirmation",
    },
    {
        "name": "new_u0_confirmation",
        "start": VISIBLE_UNLOCK_CONFIRMATION_BASE,
        "end": VISIBLE_UNLOCK_CONFIRMATION_BASE + CONFIRMATION_CASES - 1,
        "role": "new_confirmation",
    },
    {
        "name": "untouched_final_boundary",
        "start": 20_000_000,
        "end": None,
        "role": "final",
    },
)

MASTERY_DECISION = "U1 Local Unlock mastered with Navigate and U0 retained"
FIRST_PASS_DECISION = "First cumulative U1 pass; confirmation required"

EXPECTED_CHECKPOINTS: dict[int, dict[str, Any]] = {
    20260725: {
        "sha256": "bcce9b8251e97ed4fddda32871c891c3783c057bbb1f89deedb3a3d32058102a",
        "source_commit": "e2e765131bc24d22dc8a328ff0c462afc9a74c79",
        "parent_training_seed": 20260725,
        "parent_sha256": "2b235ed54746429e737af2a6037069821fdaf81a936edf8f74ce86cc766b40a2",
        "child_trained_timesteps": 393_216,
        "n_updates": 1_728,
        "artifact_sha256s": {
            "manifest.json": "fc9a9056de311572b5027993e10613ba62a1a332419f879ccca164e24ca6d7e3",
            "events.jsonl": "687ecd081d48516bf41547321a7bc617746d2d2a8fa2d77cb0e70fdb15552def",
            "evaluations.jsonl": "3ba1b2c39e1dafce354aba6f4e6cf003d25f7a07628b0ad76d7a7f8c29e2350a",
            "episodes.jsonl": "dff7ed6a87c79c26cfda9d4e164f434c9790847769b9bf5d5b577e9e74e207a4",
            "qualification.json": (
                "093ebf6a1521f98e070171dda406acff53b1336303b6c48563d9bfde53e063d6"
            ),
            "checkpoints/mastered-local-unlock.json": (
                "44d2429e33be459d7a19a8876550709a2ff86b88a61e191419e9b7219da5943c"
            ),
        },
    },
    20260729: {
        "sha256": "2a300927b48f966d5f6ddfeefe13d2e444da1e5c70bcd54e86abd6a9b2d1830b",
        "source_commit": "2bd274e6b89b8e20f36fb7831256d6683e961ffc",
        "parent_training_seed": 20260726,
        "parent_sha256": "86c5bd42cd36209cd23e27569da4684229d6c52d09082708b7d1fda380826be4",
        "child_trained_timesteps": 393_216,
        "n_updates": 1_728,
        "artifact_sha256s": {
            "manifest.json": "2dad053d4cfaff1781730550b9710eb426ac0b8a5864b1c6278ce0c59f024f2a",
            "events.jsonl": "70a844ab5c8088f6794ca9229112f6b71780e2bf93c22992450cf1715dead887",
            "evaluations.jsonl": "10f3361aa2a576400cce1ac552284bf0d0c080b6b49fd13f0387dfa411a0f3ac",
            "episodes.jsonl": "142e7e2e09d7f7489e961f50d98a100cc4b3b9d351188f30b24334c0af4a2229",
            "qualification.json": (
                "093ebf6a1521f98e070171dda406acff53b1336303b6c48563d9bfde53e063d6"
            ),
            "checkpoints/mastered-local-unlock.json": (
                "5b5409bef6767fc28e200086862417438b6e662f458e99a94a763a56bb025456"
            ),
        },
    },
    20260733: {
        "sha256": "3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104",
        "source_commit": "2bd274e6b89b8e20f36fb7831256d6683e961ffc",
        "parent_training_seed": 20260727,
        "parent_sha256": "aba4d8693ed56c9e3fee57bcee21a33f682b23bc8dad4978dce0f765d620e11b",
        "child_trained_timesteps": 294_912,
        "n_updates": 1_536,
        "artifact_sha256s": {
            "manifest.json": "cd808ba8fea6b79225895b455d975122f85d66f1451569bd04a69737f8a6dfba",
            "events.jsonl": "e8664507a8263cba714a9c8b166bd7f8bc6a106c7807017c17fad796fcbc50ea",
            "evaluations.jsonl": "88db82e16615bfee147c3dfffab352309322db7ef1aa08e3035b90405d8c5ca3",
            "episodes.jsonl": "b1070d55922530646a536240708c08b50e16a7dbec926315f5536162ec241fa8",
            "qualification.json": (
                "093ebf6a1521f98e070171dda406acff53b1336303b6c48563d9bfde53e063d6"
            ),
            "checkpoints/mastered-local-unlock.json": (
                "268f89361521dd855ba637254b236592186992718ad37ac374e85e5dbf408a07"
            ),
        },
    },
}
EXPECTED_CHILD_SEEDS = tuple(EXPECTED_CHECKPOINTS)
EXPECTED_PARENT_SEEDS = tuple(
    int(EXPECTED_CHECKPOINTS[seed]["parent_training_seed"]) for seed in EXPECTED_CHILD_SEEDS
)


class U1ConfirmationError(RuntimeError):
    """Raised when evidence cannot support the frozen U1 confirmation."""


@dataclass(frozen=True)
class AllocationVerification:
    windows: int
    profile_windows: dict[str, int]
    maximum_deviation: float
    final_target_shares: dict[str, float]
    final_window_transitions: dict[str, int]
    final_realized_shares: dict[str, float]
    whole_child_transitions: dict[str, int]
    whole_child_shares: dict[str, float]
    verified: bool = True


@dataclass(frozen=True)
class VerifiedU1Checkpoint:
    checkpoint: str
    checkpoint_sha256: str
    expected_checkpoint_sha256: str
    digest_verified: bool
    sidecar: str
    manifest: str
    child_algorithm_seed: int
    parent_training_seed: int
    parent_checkpoint_sha256: str
    parent_manifest: str
    parent_confirmation_sha256: str
    source_commit: str
    expected_source_commit: str
    source_verified: bool
    inherited_trained_timesteps: int
    child_trained_timesteps: int
    trained_timesteps: int
    n_updates: int
    artifact_sha256s: dict[str, str]
    allocation: AllocationVerification
    lineage_verified: bool = True
    configuration_verified: bool = True


def confirmation_seeds(lesson: LessonId, count: int = CONFIRMATION_CASES) -> tuple[int, ...]:
    """Return the three explicit, newly opened U1 confirmation blocks."""

    if count != CONFIRMATION_CASES:
        raise ValueError(f"the frozen U1 confirmation requires exactly {CONFIRMATION_CASES} cases")
    bases = {
        LessonId.NAVIGATE: NAVIGATE_CONFIRMATION_BASE,
        LessonId.VISIBLE_UNLOCK: VISIBLE_UNLOCK_CONFIRMATION_BASE,
        LessonId.LOCAL_UNLOCK: U1_CONFIRMATION_BASE,
    }
    base = bases[LessonId(lesson)]
    return tuple(base + offset for offset in range(count))


def seed_partition_audit() -> dict[str, Any]:
    """Prove the newly opened numerical blocks do not touch prior evidence."""

    new_partitions = [
        partition for partition in FROZEN_SEED_PARTITIONS if partition["role"] == "new_confirmation"
    ]
    reference_partitions = [
        partition
        for partition in FROZEN_SEED_PARTITIONS
        if partition["role"] not in {"new_confirmation", "final"}
    ]
    final_start = next(
        int(partition["start"])
        for partition in FROZEN_SEED_PARTITIONS
        if partition["role"] == "final"
    )
    checks: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    for index, selected in enumerate(new_partitions):
        comparators = [*reference_partitions, *new_partitions[index + 1 :]]
        for other in comparators:
            overlap_start = max(int(selected["start"]), int(other["start"]))
            overlap_end = min(int(selected["end"]), int(other["end"]))
            overlapping = overlap_start <= overlap_end
            check = {
                "left": selected["name"],
                "right": other["name"],
                "overlap": overlapping,
                "overlap_start": overlap_start if overlapping else None,
                "overlap_end": overlap_end if overlapping else None,
            }
            checks.append(check)
            if overlapping:
                collisions.append(check)
        if int(selected["end"]) >= final_start:
            collisions.append(
                {
                    "left": selected["name"],
                    "right": "untouched_final_boundary",
                    "overlap": True,
                    "overlap_start": final_start,
                    "overlap_end": int(selected["end"]),
                }
            )
    return {
        "partitions": [dict(partition) for partition in FROZEN_SEED_PARTITIONS],
        "pairwise_checks": checks,
        "collisions": collisions,
        "passed": not collisions,
    }


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise U1ConfirmationError(f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise U1ConfirmationError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise U1ConfirmationError(f"{description} must contain a JSON object: {path}")
    return value


def _lesson_float_map(values: Mapping[str, Any], description: str) -> dict[str, float]:
    expected = {lesson.value for lesson in LessonId}
    if set(values) != expected:
        raise U1ConfirmationError(f"{description} must contain exactly {sorted(expected)}")
    try:
        result = {lesson: float(value) for lesson, value in values.items()}
    except (TypeError, ValueError) as error:
        raise U1ConfirmationError(f"{description} contains a non-numeric value") from error
    if any(not math.isfinite(value) for value in result.values()):
        raise U1ConfirmationError(f"{description} contains a non-finite value")
    return result


def _lesson_count_map(values: Mapping[str, Any], description: str) -> dict[str, int]:
    expected = {lesson.value for lesson in LessonId}
    if set(values) != expected:
        raise U1ConfirmationError(f"{description} must contain exactly {sorted(expected)}")
    result: dict[str, int] = {}
    for lesson, value in values.items():
        if isinstance(value, bool):
            raise U1ConfirmationError(f"{description} contains a Boolean count")
        try:
            count = int(value)
        except (TypeError, ValueError) as error:
            raise U1ConfirmationError(f"{description} contains a non-integer count") from error
        if count != value or count < 0:
            raise U1ConfirmationError(f"{description} contains an invalid count")
        result[lesson] = count
    return result


def _target_profiles() -> dict[str, dict[str, float]]:
    profiles = {
        "normal": CurriculumState().targets(),
        **{
            f"recovery_{cause.value}": CurriculumState(recovery_cause=cause).targets()
            for cause in RecoveryCause
        },
    }
    return {
        label: {lesson.value: float(share) for lesson, share in targets.items()}
        for label, targets in profiles.items()
    }


TARGET_PROFILES = _target_profiles()


def _profile_name(targets: Mapping[str, float]) -> str:
    for name, expected in TARGET_PROFILES.items():
        if dict(targets) == expected:
            return name
    raise U1ConfirmationError(f"allocation uses an undeclared target profile: {dict(targets)}")


def _all_lessons_passed(event: Mapping[str, Any]) -> bool:
    values = event.get("lesson_passes")
    if not isinstance(values, Mapping) or set(values) != {lesson.value for lesson in LessonId}:
        return False
    return all(values[lesson.value] is True for lesson in LessonId)


def verify_transition_history(
    events_path: Path,
    *,
    child_trained_timesteps: int,
    final_allocation: Mapping[str, Any],
    scheduler_state: Mapping[str, Any],
) -> AllocationVerification:
    """Recompute every U1 practice window and verify the two-boundary mastery history."""

    if not events_path.is_file():
        raise U1ConfirmationError(f"missing U1 curriculum event history: {events_path}")
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise U1ConfirmationError(f"cannot read U1 curriculum events: {events_path}") from error

    windows: list[dict[str, Any]] = []
    maximum_deviation = 0.0
    profile_counts: Counter[str] = Counter()
    prior_lifetime = {lesson.value: 0 for lesson in LessonId}
    prior_profile: str | None = None
    prior_revision = 0
    for line_number, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise U1ConfirmationError(
                f"invalid curriculum event JSON at {events_path}:{line_number}"
            ) from error
        if event.get("type") != "curriculum_decision":
            continue
        allocation = event.get("allocation")
        if not isinstance(allocation, Mapping):
            raise U1ConfirmationError(f"curriculum event has no allocation at line {line_number}")
        targets = _lesson_float_map(
            allocation.get("target_shares", {}), f"target shares at line {line_number}"
        )
        profile = _profile_name(targets)
        transitions = _lesson_count_map(
            allocation.get("window_transitions", {}),
            f"window transitions at line {line_number}",
        )
        total = sum(transitions.values())
        if total != EVALUATION_INTERVAL:
            raise U1ConfirmationError(
                f"curriculum window at line {line_number} has {total} transitions, "
                f"expected {EVALUATION_INTERVAL}"
            )
        realized_recorded = _lesson_float_map(
            allocation.get("realized_shares", {}), f"realized shares at line {line_number}"
        )
        realized = {lesson: count / total for lesson, count in transitions.items()}
        for lesson in realized:
            if not math.isclose(
                realized[lesson], realized_recorded[lesson], rel_tol=0.0, abs_tol=1e-12
            ):
                raise U1ConfirmationError(
                    f"recorded realized share disagrees with counts at line {line_number}"
                )
            deviation = abs(realized[lesson] - targets[lesson])
            maximum_deviation = max(maximum_deviation, deviation)
            if deviation > ALLOCATION_TOLERANCE + 1e-12:
                raise U1ConfirmationError(
                    f"practice allocation exceeded tolerance at line {line_number}: "
                    f"{lesson} deviation={deviation:.6f}"
                )
        if event.get("allocation_within_tolerance") is not True:
            raise U1ConfirmationError(
                f"curriculum event does not record valid allocation at line {line_number}"
            )
        child_boundary = int(event.get("child_trained_timesteps", -1))
        expected_boundary = (len(windows) + 1) * EVALUATION_INTERVAL
        if child_boundary != expected_boundary:
            raise U1ConfirmationError(
                f"curriculum boundary sequence broke at line {line_number}: "
                f"{child_boundary} != {expected_boundary}"
            )
        lifetime = _lesson_count_map(
            allocation.get("lifetime_transitions", {}),
            f"lifetime transitions at line {line_number}",
        )
        expected_lifetime = {
            lesson: prior_lifetime[lesson] + transitions[lesson] for lesson in prior_lifetime
        }
        if lifetime != expected_lifetime or sum(lifetime.values()) != child_boundary:
            raise U1ConfirmationError(
                f"lifetime transition counts disagree with the window history at line {line_number}"
            )
        revision = int(allocation.get("revision", -1))
        expected_revision = (
            0 if prior_profile is None else prior_revision + int(profile != prior_profile)
        )
        if revision != expected_revision:
            raise U1ConfirmationError(
                f"scheduler revision sequence broke at line {line_number}: "
                f"{revision} != {expected_revision}"
            )
        windows.append(
            {
                "event": event,
                "allocation": {
                    "revision": revision,
                    "target_shares": targets,
                    "window_transitions": transitions,
                    "realized_shares": realized,
                    "lifetime_transitions": lifetime,
                },
                "profile": profile,
            }
        )
        profile_counts[profile] += 1
        prior_lifetime = lifetime
        prior_profile = profile
        prior_revision = revision

    expected_windows = child_trained_timesteps // EVALUATION_INTERVAL
    if (
        child_trained_timesteps <= 0
        or child_trained_timesteps % EVALUATION_INTERVAL
        or len(windows) != expected_windows
    ):
        raise U1ConfirmationError(
            f"curriculum history has {len(windows)} windows for "
            f"{child_trained_timesteps} child actions"
        )

    mastery = [
        index
        for index, item in enumerate(windows)
        if item["event"].get("decision") == MASTERY_DECISION
    ]
    if mastery != [len(windows) - 1]:
        raise U1ConfirmationError("curriculum history must end with exactly one mastery decision")
    if len(windows) < 2:
        raise U1ConfirmationError("U1 mastery requires two consecutive curriculum boundaries")
    first_pass = windows[-2]
    mastery_window = windows[-1]
    if (
        first_pass["event"].get("decision") != FIRST_PASS_DECISION
        or first_pass["profile"] != "normal"
        or not _all_lessons_passed(first_pass["event"])
        or mastery_window["profile"] != "normal"
        or not _all_lessons_passed(mastery_window["event"])
    ):
        raise U1ConfirmationError("mastery was not preceded by two valid normal cumulative passes")
    for index in range(1, len(windows) - 1):
        earlier = windows[index - 1]
        later = windows[index]
        if (
            earlier["profile"] == "normal"
            and later["profile"] == "normal"
            and _all_lessons_passed(earlier["event"])
            and _all_lessons_passed(later["event"])
        ):
            raise U1ConfirmationError(
                "curriculum history contains an earlier two-exam mastery boundary"
            )

    authoritative = mastery_window["allocation"]
    if dict(final_allocation) != authoritative:
        raise U1ConfirmationError("mastery sidecar allocation differs from curriculum history")
    for key, value in authoritative.items():
        if scheduler_state.get(key) != value:
            raise U1ConfirmationError(
                f"mastery scheduler field {key!r} differs from curriculum history"
            )
    if sum(authoritative["lifetime_transitions"].values()) != child_trained_timesteps:
        raise U1ConfirmationError("final scheduler lifetime does not equal child trained actions")

    return AllocationVerification(
        windows=len(windows),
        profile_windows=dict(sorted(profile_counts.items())),
        maximum_deviation=maximum_deviation,
        final_target_shares=dict(authoritative["target_shares"]),
        final_window_transitions=dict(authoritative["window_transitions"]),
        final_realized_shares=dict(authoritative["realized_shares"]),
        whole_child_transitions=dict(authoritative["lifetime_transitions"]),
        whole_child_shares={
            lesson: count / child_trained_timesteps
            for lesson, count in authoritative["lifetime_transitions"].items()
        },
    )


def _expected_effective_config(child_seed: int, parent_training_seed: int) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "warm_start": True,
        "parent_training_seed": parent_training_seed,
        "child_algorithm_seed": child_seed,
        "child_action_budget": CHILD_ACTION_BUDGET,
        "environment": {
            "size": 9,
            "workers": 4,
            "generator_profile_version": GENERATOR_PROFILE_VERSION,
        },
        "optimization": {
            "rollout_steps": 512,
            "batch_size": 256,
            "n_epochs": 4,
            "learning_rate": 0.00025,
            "gamma": 0.995,
            "gae_lambda": 0.98,
            "ent_coef": 0.01,
            "policy_kwargs": {"lstm_hidden_size": 256, "n_lstm_layers": 1},
        },
        "lessons": [lesson.value for lesson in LessonId],
        "normal_transition_targets": TARGET_PROFILES["normal"],
        "recovery_transition_targets": {
            cause.value: TARGET_PROFILES[f"recovery_{cause.value}"] for cause in RecoveryCause
        },
        "evaluation": {
            "interval": EVALUATION_INTERVAL,
            "seeds_per_lesson": 80,
            "panels": [40, 40],
            "thresholds": {
                lesson.value: {
                    "overall": CONFIRMATION_GATES[lesson][0],
                    "panel": CONFIRMATION_GATES[lesson][1],
                }
                for lesson in LessonId
            },
            "consecutive_mastery_passes": 2,
            "consecutive_recovery_passes": 2,
            "allocation_tolerance": ALLOCATION_TOLERANCE,
        },
    }


def verify_u1_checkpoint(
    checkpoint: Path,
    *,
    expected_checkpoints: Mapping[int, Mapping[str, Any]] = EXPECTED_CHECKPOINTS,
) -> VerifiedU1Checkpoint:
    """Verify a frozen U1 mastery archive, its lineage, and every practice window."""

    archive = checkpoint.expanduser().resolve()
    if archive.suffix != ".zip":
        archive = archive.with_suffix(".zip")
    if archive.name != "mastered-local-unlock.zip" or not archive.is_file():
        raise U1ConfirmationError(
            f"confirmation requires an existing mastered-local-unlock.zip: {archive}"
        )
    sidecar_path = archive.with_suffix(".json")
    sidecar = _read_json(sidecar_path, "U1 mastery sidecar")
    if sidecar.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise U1ConfirmationError(f"U1 checkpoint schema mismatch: {archive}")
    if sidecar.get("protocol") != PROTOCOL or sidecar.get("kind") != "mastery":
        raise U1ConfirmationError(f"checkpoint is not a U1 mastery artifact: {archive}")
    digest = file_sha256(archive)
    if sidecar.get("checkpoint_sha256") != digest:
        raise U1ConfirmationError(f"checkpoint digest does not match sidecar: {archive}")

    config = sidecar.get("effective_config")
    if not isinstance(config, Mapping):
        raise U1ConfirmationError(f"U1 sidecar has no effective configuration: {archive}")
    try:
        child_seed = int(config["child_algorithm_seed"])
        parent_seed = int(config["parent_training_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise U1ConfirmationError(f"U1 sidecar has invalid seed lineage: {archive}") from error
    selection = expected_checkpoints.get(child_seed)
    if selection is None:
        raise U1ConfirmationError(f"unselected U1 child seed: {child_seed}")
    if digest != selection.get("sha256") or parent_seed != int(
        selection.get("parent_training_seed", -1)
    ):
        raise U1ConfirmationError(f"checkpoint does not match the frozen U1 selection: {archive}")

    curriculum = sidecar.get("curriculum", {})
    if (
        curriculum.get("active_lesson") != LessonId.LOCAL_UNLOCK.value
        or curriculum.get("mastered") is not True
        or int(curriculum.get("consecutive_passes", 0)) < 2
        or curriculum.get("recovery") is not False
        or set(curriculum.get("passed_lessons", ()))
        != {LessonId.NAVIGATE.value, LessonId.VISIBLE_UNLOCK.value}
    ):
        raise U1ConfirmationError(f"sidecar does not describe cumulative U1 mastery: {archive}")
    if config != _expected_effective_config(child_seed, parent_seed):
        raise U1ConfirmationError(f"U1 effective configuration changed: {archive}")

    parent_data = sidecar.get("parent")
    if not isinstance(parent_data, Mapping):
        raise U1ConfirmationError(f"U1 sidecar has no parent provenance: {archive}")
    if int(parent_data.get("training_seed", -1)) != parent_seed or parent_data.get(
        "checkpoint_sha256"
    ) != selection.get("parent_sha256"):
        raise U1ConfirmationError(f"U1 parent lineage differs from the frozen selection: {archive}")
    try:
        parent = verify_parent(
            Path(str(parent_data["checkpoint"])),
            Path(str(parent_data["confirmation_report"])),
            child_seed=child_seed,
        )
    except (KeyError, U1ProtocolError) as error:
        raise U1ConfirmationError(f"U1 parent verification failed: {error}") from error
    if not isinstance(parent, ParentProvenance) or parent.public_dict() != dict(parent_data):
        raise U1ConfirmationError(f"U1 sidecar parent provenance is not reproducible: {archive}")

    progress = sidecar.get("progress", {})
    try:
        child_trained = int(progress["child_trained_timesteps"])
        trained = int(progress["trained_timesteps"])
        collected = int(progress["collected_timesteps"])
        segment_trained = int(progress["segment_trained_timesteps"])
        n_updates = int(progress["n_updates"])
    except (KeyError, TypeError, ValueError) as error:
        raise U1ConfirmationError(f"U1 sidecar has invalid progress counters: {archive}") from error
    if (
        child_trained != int(selection.get("child_trained_timesteps", -1))
        or n_updates != int(selection.get("n_updates", -1))
        or not 0 < child_trained <= CHILD_ACTION_BUDGET
        or child_trained % EVALUATION_INTERVAL
        or trained != collected
        or trained != parent.trained_timesteps + child_trained
        or segment_trained != child_trained
    ):
        raise U1ConfirmationError(f"U1 checkpoint is not at its frozen trained boundary: {archive}")
    rollout_size = 4 * 512
    expected_updates = parent.n_updates + child_trained // rollout_size * 4
    if n_updates != expected_updates:
        raise U1ConfirmationError(f"U1 optimizer counter disagrees with trained actions: {archive}")

    segment = sidecar.get("segment", {})
    if (
        int(segment.get("index", -1)) != 0
        or segment.get("resume_checkpoint") is not None
        or int(segment.get("algorithm_seed", -1)) != child_seed
        or int(segment.get("worker_seed_base", -1)) != child_seed
        or int(segment.get("start_trained_timesteps", -1)) != parent.trained_timesteps
        or int(segment.get("start_child_trained_timesteps", -1)) != 0
    ):
        raise U1ConfirmationError(f"U1 segment lineage differs from the frozen run: {archive}")
    try:
        with zipfile.ZipFile(archive) as zipped:
            if "policy.optimizer.pth" not in zipped.namelist():
                raise U1ConfirmationError(f"U1 archive has no optimizer state: {archive}")
    except zipfile.BadZipFile as error:
        raise U1ConfirmationError(f"U1 archive is not readable: {archive}") from error

    run_directory = archive.parent.parent
    manifest_path = run_directory / "manifest.json"
    manifest = _read_json(manifest_path, "U1 run manifest")
    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("effective_config") != dict(config)
        or manifest.get("parent") != dict(parent_data)
        or manifest.get("segment") != dict(segment)
    ):
        raise U1ConfirmationError(f"U1 manifest and mastery sidecar disagree: {archive}")
    expected_artifacts = selection.get("artifact_sha256s")
    if not isinstance(expected_artifacts, Mapping):
        raise U1ConfirmationError(
            f"frozen supporting-artifact digests are missing for child {child_seed}"
        )
    measured_artifacts: dict[str, str] = {}
    for relative_path, expected_digest in expected_artifacts.items():
        artifact_path = run_directory / str(relative_path)
        if not artifact_path.is_file():
            raise U1ConfirmationError(f"frozen supporting artifact is missing: {artifact_path}")
        measured = file_sha256(artifact_path)
        measured_artifacts[str(relative_path)] = measured
        if measured != expected_digest:
            raise U1ConfirmationError(f"supporting artifact digest changed: {artifact_path}")
    source = manifest.get("git", {})
    source_commit = source.get("commit")
    if source.get("dirty") is not False or source_commit != selection.get("source_commit"):
        raise U1ConfirmationError(
            f"U1 checkpoint source does not match its frozen commit: {archive}"
        )

    final_allocation = sidecar.get("last_completed_allocation")
    scheduler = sidecar.get("scheduler")
    if not isinstance(final_allocation, Mapping) or not isinstance(scheduler, Mapping):
        raise U1ConfirmationError(f"U1 mastery sidecar lacks scheduler evidence: {archive}")
    allocation = verify_transition_history(
        run_directory / "events.jsonl",
        child_trained_timesteps=child_trained,
        final_allocation=final_allocation,
        scheduler_state=scheduler,
    )
    return VerifiedU1Checkpoint(
        checkpoint=str(archive),
        checkpoint_sha256=digest,
        expected_checkpoint_sha256=str(selection["sha256"]),
        digest_verified=True,
        sidecar=str(sidecar_path),
        manifest=str(manifest_path),
        child_algorithm_seed=child_seed,
        parent_training_seed=parent_seed,
        parent_checkpoint_sha256=parent.checkpoint_sha256,
        parent_manifest=parent.manifest,
        parent_confirmation_sha256=parent.confirmation_sha256,
        source_commit=str(source_commit),
        expected_source_commit=str(selection["source_commit"]),
        source_verified=True,
        inherited_trained_timesteps=parent.trained_timesteps,
        child_trained_timesteps=child_trained,
        trained_timesteps=trained,
        n_updates=n_updates,
        artifact_sha256s=measured_artifacts,
        allocation=allocation,
    )


def _duplicate_evidence(cases: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = {}
    for case in cases:
        grouped.setdefault(str(case[field]), []).append(int(case["seed"]))
    return [
        {
            field: digest,
            "occurrences": len(seeds),
            "seeds": seeds,
        }
        for digest, seeds in sorted(grouped.items())
        if len(seeds) > 1
    ]


def _panel_uniqueness(
    cases: Sequence[Mapping[str, Any]],
    *,
    full_floor: float,
    geometry_floor: float | None,
) -> list[dict[str, Any]]:
    panels: list[dict[str, Any]] = []
    for index in range(2):
        panel = [
            case
            for case_index, case in enumerate(cases)
            if int(
                case.get(
                    "panel",
                    case_index // CONFIRMATION_PANEL_SIZE,
                )
            )
            == index
        ]
        full_unique = len({str(case["layout_sha256"]) for case in panel})
        geometry_unique = len({str(case["geometry_sha256"]) for case in panel})
        required_full = math.ceil(CONFIRMATION_PANEL_SIZE * full_floor)
        required_geometry = (
            math.ceil(CONFIRMATION_PANEL_SIZE * geometry_floor)
            if geometry_floor is not None
            else None
        )
        full_passed = full_unique >= required_full
        geometry_passed = (
            geometry_unique >= required_geometry if required_geometry is not None else None
        )
        panels.append(
            {
                "panel": index,
                "cases": len(panel),
                "expected_cases": CONFIRMATION_PANEL_SIZE,
                "full_unique_layouts": full_unique,
                "required_full_unique_layouts": required_full,
                "geometry_unique_layouts": geometry_unique,
                "required_geometry_unique_layouts": required_geometry,
                "geometry_affects_verdict": geometry_floor is not None,
                "full_layout_duplicates": _duplicate_evidence(panel, "layout_sha256"),
                "geometry_duplicates": _duplicate_evidence(panel, "geometry_sha256"),
                "full_layouts_passed": full_passed,
                "geometry_passed": geometry_passed,
                "passed": full_passed
                and (geometry_passed is True if geometry_floor is not None else True),
            }
        )
    return panels


def _qualify_anchor_block(lesson: LessonId, seeds: tuple[int, ...]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    actions: list[int] = []
    for case_index, seed in enumerate(seeds):
        wrapped = make_pixel_env(lesson=lesson, size=9)
        env = wrapped.unwrapped
        assert isinstance(env, LessonEnv)
        try:
            observation, _ = wrapped.reset(seed=seed)
            assert env.layout is not None
            if observation.shape != PIXEL_SHAPE or observation.dtype != np.uint8:
                raise OracleFailure("confirmation observation contract changed")
            key_visible = env.agent_sees(*env.layout.key) if env.layout.key is not None else None
            door_visible = env.agent_sees(*env.layout.door) if env.layout.door is not None else None
            if lesson is LessonId.VISIBLE_UNLOCK and (
                key_visible is not True or door_visible is not True
            ):
                raise OracleFailure("Visible Unlock confirmation visibility contract failed")
            result = DungeonOracle(env).solve()
            if lesson is LessonId.VISIBLE_UNLOCK and not 5 <= result.steps <= 10:
                raise OracleFailure(f"Visible Unlock solution length {result.steps} outside 5..10")
            expected_reward = STEP_REWARD * (result.steps - 1) + SUCCESS_REWARD
            if not math.isclose(result.total_reward, expected_reward, abs_tol=1e-9):
                raise OracleFailure("confirmation reward contract changed")
            if env.geometry_sha256 is None:
                raise OracleFailure("confirmation geometry hash is missing")
            actions.append(result.steps)
            cases.append(
                {
                    "seed": seed,
                    "panel": case_index // CONFIRMATION_PANEL_SIZE,
                    "panel_label": "A" if case_index < CONFIRMATION_PANEL_SIZE else "B",
                    "layout_sha256": result.layout_sha256,
                    "geometry_sha256": env.geometry_sha256,
                    "oracle_actions": result.steps,
                    "visibility": {
                        "key_visible": key_visible,
                        "door_visible": door_visible,
                    },
                }
            )
        except (AssertionError, OracleFailure, RuntimeError, ValueError) as error:
            failures.append({"seed": seed, "error": str(error)})
        finally:
            wrapped.close()
    panels = _panel_uniqueness(
        cases,
        full_floor=ANCHOR_UNIQUENESS_FLOOR,
        geometry_floor=None,
    )
    full_unique = len({case["layout_sha256"] for case in cases})
    geometry_unique = len({case["geometry_sha256"] for case in cases})
    passed = (
        not failures
        and len(cases) == CONFIRMATION_CASES
        and full_unique >= math.ceil(CONFIRMATION_CASES * ANCHOR_UNIQUENESS_FLOOR)
        and all(panel["passed"] for panel in panels)
    )
    return {
        "protocol": CONFIRMATION_PROTOCOL,
        "lesson_id": lesson.value,
        "seed_partition": {"start": seeds[0], "end": seeds[-1]},
        "requested": len(seeds),
        "solved": len(actions),
        "full_unique_layouts": full_unique,
        "geometry_unique_layouts": geometry_unique,
        "full_layout_uniqueness_floor": ANCHOR_UNIQUENESS_FLOOR,
        "geometry_uniqueness_floor": None,
        "geometry_affects_verdict": False,
        "minimum_oracle_actions": min(actions) if actions else None,
        "maximum_oracle_actions": max(actions) if actions else None,
        "panel_qualification": panels,
        "full_layout_duplicates": _duplicate_evidence(cases, "layout_sha256"),
        "geometry_duplicates": _duplicate_evidence(cases, "geometry_sha256"),
        "cases": cases,
        "failures": failures,
        "result": "passed" if passed else "failed",
    }


def qualify_confirmation_block(lesson: LessonId) -> dict[str, Any]:
    """Qualify one frozen block without exposing oracle actions to a policy."""

    seeds = confirmation_seeds(lesson)
    if lesson is not LessonId.LOCAL_UNLOCK:
        return _qualify_anchor_block(lesson, seeds)
    if not math.isclose(
        FULL_LAYOUT_UNIQUENESS_FLOOR,
        U1_FULL_LAYOUT_UNIQUENESS_FLOOR,
        rel_tol=0.0,
        abs_tol=0.0,
    ) or not math.isclose(
        GEOMETRY_UNIQUENESS_FLOOR,
        U1_GEOMETRY_UNIQUENESS_FLOOR,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise U1ConfirmationError(
            "runtime U1 qualification floors differ from the frozen confirmation floors"
        )
    report = qualify_local_unlock(
        CONFIRMATION_CASES,
        seed_base=U1_CONFIRMATION_BASE,
        size=9,
    )
    cases = [
        dict(case)
        | {
            "panel": (int(case["seed"]) - U1_CONFIRMATION_BASE) // CONFIRMATION_PANEL_SIZE,
            "panel_label": (
                "A" if int(case["seed"]) < U1_CONFIRMATION_BASE + CONFIRMATION_PANEL_SIZE else "B"
            ),
            "visibility": {"key_visible": True, "door_visible": False},
        }
        for case in report.get("cases", [])
    ]
    panels = _panel_uniqueness(
        cases,
        full_floor=U1_FULL_LAYOUT_UNIQUENESS_FLOOR,
        geometry_floor=U1_GEOMETRY_UNIQUENESS_FLOOR,
    )
    failures = list(report.get("failures", []))
    passed = (
        int(report.get("solved", -1)) == CONFIRMATION_CASES
        and int(report.get("full_unique_layouts", -1))
        >= math.ceil(CONFIRMATION_CASES * U1_FULL_LAYOUT_UNIQUENESS_FLOOR)
        and int(report.get("geometry_unique_layouts", -1))
        >= math.ceil(CONFIRMATION_CASES * U1_GEOMETRY_UNIQUENESS_FLOOR)
        and not failures
        and all(panel["passed"] for panel in panels)
    )
    return dict(report) | {
        "confirmation_protocol": CONFIRMATION_PROTOCOL,
        "cases": cases,
        "panel_qualification": panels,
        "full_layout_duplicates": _duplicate_evidence(cases, "layout_sha256"),
        "geometry_duplicates": _duplicate_evidence(cases, "geometry_sha256"),
        "validation_disjointness_affects_verdict": True,
        "result": "passed" if passed else "failed",
    }


def _read_episode_layouts(path: Path) -> dict[LessonId, list[dict[str, Any]]]:
    if not path.is_file():
        raise U1ConfirmationError(f"missing episode history: {path}")
    records = {lesson: [] for lesson in LessonId}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise U1ConfirmationError(f"cannot read episode history: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
            lesson = LessonId(str(value["lesson_id"]))
        except (json.JSONDecodeError, KeyError, ValueError) as error:
            raise U1ConfirmationError(f"invalid episode history at {path}:{line_number}") from error
        layout_hash = value.get("layout_sha256")
        seed = value.get("seed")
        if not isinstance(layout_hash, str) or not layout_hash or seed is None:
            raise U1ConfirmationError(
                f"episode history lacks layout evidence at {path}:{line_number}"
            )
        records[lesson].append(
            {
                "seed": int(seed),
                "layout_sha256": layout_hash,
                "geometry_sha256": value.get("geometry_sha256"),
            }
        )
    return records


def _confirmation_cases_by_lesson(
    qualifications: Sequence[Mapping[str, Any]],
) -> dict[LessonId, list[dict[str, Any]]]:
    by_lesson: dict[LessonId, list[dict[str, Any]]] = {}
    for report in qualifications:
        lesson = LessonId(str(report["lesson_id"]))
        cases = [dict(case) for case in report.get("cases", ())]
        if len(cases) != CONFIRMATION_CASES:
            raise U1ConfirmationError(
                f"confirmation qualification for {lesson.value} lacks all cases"
            )
        by_lesson[lesson] = cases
    if set(by_lesson) != set(LessonId):
        raise U1ConfirmationError("confirmation qualification is missing a lesson")
    return by_lesson


def _overlap_summary(
    confirmation_cases: Sequence[Mapping[str, Any]],
    history_hashes: set[str],
    *,
    field: str,
    affects_verdict: bool = False,
) -> dict[str, Any]:
    overlapping_cases = [
        {
            "seed": int(case["seed"]),
            field: str(case[field]),
        }
        for case in confirmation_cases
        if str(case[field]) in history_hashes
    ]
    overlap_hashes = sorted({str(case[field]) for case in overlapping_cases})
    return {
        "history_unique_hashes": len(history_hashes),
        "overlap_unique_hashes": len(overlap_hashes),
        "overlapping_confirmation_cases": len(overlapping_cases),
        "overlap_hashes": overlap_hashes,
        "cases": overlapping_cases,
        "affects_verdict": affects_verdict,
        "passed": not overlapping_cases if affects_verdict else None,
    }


def _geometry_matrix(
    confirmation: Mapping[LessonId, Sequence[Mapping[str, Any]]],
    history: Mapping[LessonId, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, int]]]:
    matrix: dict[str, dict[str, dict[str, int]]] = {}
    for confirmation_lesson, cases in confirmation.items():
        confirmation_hashes = {str(case["geometry_sha256"]) for case in cases}
        row: dict[str, dict[str, int]] = {}
        for history_lesson, records in history.items():
            history_hashes = {
                str(record["geometry_sha256"])
                for record in records
                if record.get("geometry_sha256")
            }
            row[history_lesson.value] = {
                "confirmation_unique_geometries": len(confirmation_hashes),
                "history_unique_geometries": len(history_hashes),
                "overlap_unique_geometries": len(confirmation_hashes & history_hashes),
            }
        matrix[confirmation_lesson.value] = row
    return matrix


def _reconstruct_parent_geometry(
    parent_manifest: Path,
) -> dict[LessonId, list[dict[str, Any]]]:
    parent_run = parent_manifest.parent
    parent_records = _read_episode_layouts(parent_run / "episodes.jsonl")
    reconstructed = {lesson: [] for lesson in LessonId}
    cache: dict[tuple[LessonId, int], str] = {}
    for lesson in (LessonId.NAVIGATE, LessonId.VISIBLE_UNLOCK):
        for record in parent_records[lesson]:
            seed = int(record["seed"])
            key = (lesson, seed)
            if key not in cache:
                env = LessonEnv(lesson=lesson, size=9)
                try:
                    env.reset(seed=seed)
                    if env.geometry_sha256 is None:
                        raise U1ConfirmationError(
                            f"could not reconstruct parent geometry for {lesson.value}/{seed}"
                        )
                    cache[key] = env.geometry_sha256
                finally:
                    env.close()
            reconstructed[lesson].append(
                {
                    "seed": seed,
                    "layout_sha256": record["layout_sha256"],
                    "geometry_sha256": cache[key],
                }
            )
    return reconstructed


def _reference_hashes(
    run_directory: Path,
) -> dict[LessonId, dict[str, set[str]]]:
    qualification = _read_json(run_directory / "qualification.json", "U1 qualification report")
    qualification_hashes = {str(case["layout_sha256"]) for case in qualification.get("cases", ())}
    references: dict[LessonId, dict[str, set[str]]] = {}
    for lesson in LessonId:
        validation_hashes: set[str] = set()
        for seed in validation_seeds(lesson):
            env = LessonEnv(lesson=lesson, size=9)
            try:
                env.reset(seed=seed)
                assert env.layout is not None
                validation_hashes.add(env.layout.layout_sha256)
            finally:
                env.close()
        references[lesson] = {"validation": validation_hashes}
    references[LessonId.LOCAL_UNLOCK]["engineering_qualification"] = qualification_hashes
    return references


def _fixed_geometry_cases(lesson: LessonId, seeds: Sequence[int]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for seed in seeds:
        env = LessonEnv(lesson=lesson, size=9)
        try:
            env.reset(seed=int(seed))
            assert env.layout is not None
            if env.geometry_sha256 is None:
                raise U1ConfirmationError(
                    f"missing reconstructed geometry for {lesson.value}/{seed}"
                )
            cases.append(
                {
                    "seed": int(seed),
                    "layout_sha256": env.layout.layout_sha256,
                    "geometry_sha256": env.geometry_sha256,
                }
            )
        finally:
            env.close()
    return cases


def _prior_u0_confirmation_geometry() -> dict[LessonId, list[dict[str, Any]]]:
    return {
        LessonId.NAVIGATE: _fixed_geometry_cases(LessonId.NAVIGATE, range(15_000_000, 15_000_200)),
        LessonId.VISIBLE_UNLOCK: _fixed_geometry_cases(
            LessonId.VISIBLE_UNLOCK, range(15_010_000, 15_010_200)
        ),
        LessonId.LOCAL_UNLOCK: [],
    }


def collision_diagnostics(
    qualifications: Sequence[Mapping[str, Any]],
    verified: Sequence[VerifiedU1Checkpoint],
) -> dict[str, Any]:
    """Report procedural repeats without allowing them to alter a frozen verdict."""

    confirmation = _confirmation_cases_by_lesson(qualifications)
    prior_confirmation = _prior_u0_confirmation_geometry()
    per_policy: list[dict[str, Any]] = []
    contract_violations: list[dict[str, Any]] = []
    for checkpoint in verified:
        run_directory = Path(checkpoint.manifest).parent
        child_history = _read_episode_layouts(run_directory / "episodes.jsonl")
        same_protocol_exact = {
            lesson.value: _overlap_summary(
                confirmation[lesson],
                {str(record["layout_sha256"]) for record in child_history[lesson]},
                field="layout_sha256",
            )
            for lesson in LessonId
        }
        references = _reference_hashes(run_directory)
        parent_history = _reconstruct_parent_geometry(Path(checkpoint.parent_manifest))
        reference_overlap = {
            lesson.value: {
                label: _overlap_summary(
                    confirmation[lesson],
                    hashes,
                    field="layout_sha256",
                    affects_verdict=True,
                )
                for label, hashes in lesson_references.items()
            }
            for lesson, lesson_references in references.items()
        }
        for lesson, lesson_overlaps in reference_overlap.items():
            for label, overlap in lesson_overlaps.items():
                if overlap["overlapping_confirmation_cases"]:
                    contract_violations.append(
                        {
                            "child_algorithm_seed": checkpoint.child_algorithm_seed,
                            "lesson_id": lesson,
                            "reference_partition": label,
                            "overlapping_confirmation_cases": overlap[
                                "overlapping_confirmation_cases"
                            ],
                        }
                    )
        per_policy.append(
            {
                "child_algorithm_seed": checkpoint.child_algorithm_seed,
                "parent_training_seed": checkpoint.parent_training_seed,
                "same_protocol_exact_training_overlap": same_protocol_exact,
                "child_training_geometry_matrix": _geometry_matrix(confirmation, child_history),
                "parent_history_protocol_independent_geometry_matrix": _geometry_matrix(
                    confirmation, parent_history
                ),
                "u1_reference_overlap": reference_overlap,
            }
        )
    return {
        "metric": ("Same-protocol exact layout SHA-256 and protocol-independent geometry SHA-256"),
        "ordinary_procedural_repeats_affect_verdict": False,
        "declared_reference_partition_overlaps_affect_verdict": True,
        "contract_violations": contract_violations,
        "passed": not contract_violations,
        "new_confirmation_geometry_matrix": _geometry_matrix(confirmation, confirmation),
        "prior_u0_confirmation_geometry_matrix": _geometry_matrix(confirmation, prior_confirmation),
        "policies": per_policy,
    }


def grade_evaluation(result: LessonEvaluation) -> dict[str, Any]:
    lesson = LessonId(result.lesson_id)
    if (
        result.protocol != PROTOCOL
        or result.episodes != CONFIRMATION_CASES
        or len(result.panel_successes) != 2
        or len(result.panel_success_rates) != 2
        or sum(result.panel_successes) != result.successes
    ):
        raise U1ConfirmationError(
            f"confirmation evaluation has invalid counts: {result.public_dict()}"
        )
    expected_rate = result.successes / CONFIRMATION_CASES
    if not math.isclose(result.success_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12):
        raise U1ConfirmationError("confirmation overall rate disagrees with success count")
    for successes, rate in zip(result.panel_successes, result.panel_success_rates, strict=True):
        if not math.isclose(rate, successes / CONFIRMATION_PANEL_SIZE, rel_tol=0.0, abs_tol=1e-12):
            raise U1ConfirmationError("confirmation panel rate disagrees with success count")
    overall_threshold, panel_threshold = CONFIRMATION_GATES[lesson]
    overall_required = math.ceil(CONFIRMATION_CASES * overall_threshold)
    panel_required = math.ceil(CONFIRMATION_PANEL_SIZE * panel_threshold)
    overall_passed = result.successes >= overall_required
    panels_passed = all(value >= panel_required for value in result.panel_successes)
    return {
        "overall_threshold": overall_threshold,
        "panel_threshold": panel_threshold,
        "overall_required": overall_required,
        "panel_required": panel_required,
        "overall_passed": overall_passed,
        "panels_passed": panels_passed,
        "passed": overall_passed and panels_passed,
    }


def policy_tensor_sha256(model: Any) -> str:
    """Digest every policy parameter and buffer in stable name order."""

    digest = hashlib.sha256()
    state = model.policy.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        metadata = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _update_structured_digest(digest: Any, value: Any, *, path: tuple[str, ...] = ()) -> None:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu().contiguous()
        metadata = {
            "path": list(path),
            "kind": "tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        digest.update(tensor.numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(f"mapping:{path}".encode())
        for key in sorted(value, key=lambda item: str(item)):
            _update_structured_digest(digest, value[key], path=(*path, str(key)))
        return
    if isinstance(value, (list, tuple)):
        digest.update(f"sequence:{path}:{len(value)}".encode())
        for index, item in enumerate(value):
            _update_structured_digest(digest, item, path=(*path, str(index)))
        return
    digest.update(
        json.dumps(
            {
                "path": list(path),
                "kind": type(value).__name__,
                "value": value,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def optimizer_state_sha256(model: Any) -> str:
    """Digest optimizer tensors, counters, and parameter-group settings."""

    optimizer = getattr(model.policy, "optimizer", None)
    if optimizer is None:
        raise U1ConfirmationError("loaded policy has no optimizer")
    digest = hashlib.sha256()
    _update_structured_digest(digest, optimizer.state_dict())
    return digest.hexdigest()


def _verify_loaded_model_contract(model: Any) -> None:
    expected_observation = (PIXEL_SHAPE[2], PIXEL_SHAPE[0], PIXEL_SHAPE[1])
    observation_shape = tuple(getattr(model.observation_space, "shape", ()))
    action_count = int(getattr(model.action_space, "n", -1))
    lstm = getattr(model.policy, "lstm_actor", None)
    optimizer = getattr(model.policy, "optimizer", None)
    if observation_shape != expected_observation or action_count != 7:
        raise U1ConfirmationError("loaded model observation or action contract changed")
    if (
        lstm is None
        or int(getattr(lstm, "hidden_size", -1)) != 256
        or int(getattr(lstm, "num_layers", -1)) != 1
    ):
        raise U1ConfirmationError("loaded model LSTM contract changed")
    if optimizer is None or not optimizer.state:
        raise U1ConfirmationError("loaded model has no trained optimizer state")


def milestone_evidence(result: LessonEvaluation) -> dict[str, Any]:
    """Convert evaluation rates back to auditable counts and conditional rates."""

    counts: dict[str, int] = {}
    for name, rate in result.milestone_rates.items():
        count = round(float(rate) * result.episodes)
        if (
            not math.isfinite(float(rate))
            or not 0 <= count <= result.episodes
            or not math.isclose(
                float(rate),
                count / result.episodes,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise U1ConfirmationError(
                f"milestone rate {name!r} cannot be represented as an episode count"
            )
        counts[name] = count
    key_count = counts.get("key_picked_up", 0)
    door_count = counts.get("door_opened", 0)
    success_count = counts.get("success", result.successes)
    return {
        "counts": counts,
        "key_picked_up": key_count,
        "door_opened": door_count,
        "quest_completed": success_count,
        "door_given_key": door_count / key_count if key_count else None,
        "success_given_door": success_count / door_count if door_count else None,
    }


def _evaluate_selected_checkpoint(
    verified: VerifiedU1Checkpoint, model_type: Any
) -> dict[str, Any]:
    archive = Path(verified.checkpoint)
    archive_digest_before = file_sha256(archive)
    if archive_digest_before != verified.checkpoint_sha256:
        raise U1ConfirmationError("checkpoint archive changed before policy evaluation")
    model = model_type.load(verified.checkpoint, device="cpu")
    _verify_loaded_model_contract(model)
    before_digest = policy_tensor_sha256(model)
    before_optimizer_digest = optimizer_state_sha256(model)
    before_timesteps = int(model.num_timesteps)
    before_updates = int(model._n_updates)
    if before_timesteps != verified.trained_timesteps or before_updates != verified.n_updates:
        raise U1ConfirmationError(
            "loaded model counters disagree with the verified mastery sidecar"
        )
    evaluations: list[dict[str, Any]] = []
    import torch

    with torch.inference_mode():
        for lesson in LessonId:
            result = evaluate_lesson(model, lesson, confirmation_seeds(lesson), size=9)
            evaluations.append(
                result.public_dict()
                | {
                    "gate": grade_evaluation(result),
                    "milestone_evidence": milestone_evidence(result),
                }
            )
    after_digest = policy_tensor_sha256(model)
    after_optimizer_digest = optimizer_state_sha256(model)
    after_timesteps = int(model.num_timesteps)
    after_updates = int(model._n_updates)
    archive_digest_after = file_sha256(archive)
    if (
        after_digest != before_digest
        or after_optimizer_digest != before_optimizer_digest
        or after_timesteps != before_timesteps
        or after_updates != before_updates
    ):
        raise U1ConfirmationError("confirmation evaluation changed policy or optimizer counters")
    if archive_digest_after != archive_digest_before:
        raise U1ConfirmationError("checkpoint archive changed during policy evaluation")
    return {
        "checkpoint": asdict(verified),
        "policy_tensor_sha256_before": before_digest,
        "policy_tensor_sha256_after": after_digest,
        "optimizer_state_sha256_before": before_optimizer_digest,
        "optimizer_state_sha256_after": after_optimizer_digest,
        "checkpoint_sha256_before_load": archive_digest_before,
        "checkpoint_sha256_after_evaluation": archive_digest_after,
        "model_num_timesteps_before": before_timesteps,
        "model_num_timesteps_after": after_timesteps,
        "model_updates_before": before_updates,
        "model_updates_after": after_updates,
        "policy_updates": False,
        "evaluations": evaluations,
        "passed": all(item["gate"]["passed"] for item in evaluations),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="selected mastered-local-unlock.zip; provide exactly three",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(args.checkpoint) != len(EXPECTED_CHILD_SEEDS):
        raise SystemExit("U1 confirmation requires exactly three selected checkpoints")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite U1 confirmation report: {output}")
    try:
        repository = Path(__file__).resolve().parents[2]
        source = git_snapshot(repository)
        if source.get("dirty") is not False:
            raise U1ConfirmationError("U1 confirmation requires a clean committed source tree")
        started_at = utc_now()
        wall_started = time.monotonic()
        partition_audit = seed_partition_audit()
        if not partition_audit["passed"]:
            raise U1ConfirmationError(
                f"confirmation seed partitions overlap: {partition_audit['collisions']}"
            )
        base_report = {
            "schema_version": CONFIRMATION_SCHEMA_VERSION,
            "protocol": CONFIRMATION_PROTOCOL,
            "started_at": started_at,
            "source": source,
            "runtime": runtime_snapshot(),
            "report_path": str(output),
            "policy_updates": False,
            "online_model_calls": False,
            "action_selection": "deterministic",
            "recurrent_state_reset_each_case": True,
            "interruption": None,
            "selected_child_seeds": list(EXPECTED_CHILD_SEEDS),
            "selected_parent_seeds": list(EXPECTED_PARENT_SEEDS),
            "seed_partition_audit": partition_audit,
            "seed_partitions": {
                lesson.value: {
                    "start": confirmation_seeds(lesson)[0],
                    "end": confirmation_seeds(lesson)[-1],
                    "cases": CONFIRMATION_CASES,
                    "panels": [CONFIRMATION_PANEL_SIZE, CONFIRMATION_PANEL_SIZE],
                    "newly_opened": True,
                }
                for lesson in LessonId
            },
        }

        verified = [verify_u1_checkpoint(path) for path in args.checkpoint]
        child_seeds = [item.child_algorithm_seed for item in verified]
        parent_seeds = [item.parent_training_seed for item in verified]
        if sorted(child_seeds) != sorted(EXPECTED_CHILD_SEEDS):
            raise U1ConfirmationError(
                f"selected child seeds must be {EXPECTED_CHILD_SEEDS}, found {child_seeds}"
            )
        if sorted(parent_seeds) != sorted(EXPECTED_PARENT_SEEDS):
            raise U1ConfirmationError(
                f"selected parent seeds must be {EXPECTED_PARENT_SEEDS}, found {parent_seeds}"
            )
        if len({item.checkpoint_sha256 for item in verified}) != len(verified):
            raise U1ConfirmationError("selected U1 checkpoints are not distinct policies")
        verified.sort(key=lambda item: item.child_algorithm_seed)

        qualifications = [qualify_confirmation_block(lesson) for lesson in LessonId]
        if not all(item["result"] == "passed" for item in qualifications):
            report = base_report | {
                "completed_at": utc_now(),
                "elapsed_seconds": time.monotonic() - wall_started,
                "layout_qualification": qualifications,
                "collision_diagnostics": None,
                "checkpoint_scoring_performed": False,
                "checkpoints": [],
                "verdict": "qualification_failed",
            }
            atomic_write_json(output, report)
            sys.stdout.write(f"{report['verdict']}: {output}\n")
            raise SystemExit(1)
        diagnostics = collision_diagnostics(qualifications, verified)
        if diagnostics["contract_violations"]:
            report = base_report | {
                "completed_at": utc_now(),
                "elapsed_seconds": time.monotonic() - wall_started,
                "layout_qualification": qualifications,
                "collision_diagnostics": diagnostics,
                "checkpoint_scoring_performed": False,
                "checkpoints": [],
                "verdict": "qualification_failed",
            }
            atomic_write_json(output, report)
            sys.stdout.write(f"{report['verdict']}: {output}\n")
            raise SystemExit(1)
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError as error:
            raise U1ConfirmationError(
                'install training dependencies with: pip install -e ".[train]"'
            ) from error

        checkpoints = [_evaluate_selected_checkpoint(item, RecurrentPPO) for item in verified]
        if len({item["policy_tensor_sha256_before"] for item in checkpoints}) != len(checkpoints):
            raise U1ConfirmationError(
                "selected checkpoints do not contain three distinct policy tensors"
            )
        report = base_report | {
            "completed_at": utc_now(),
            "elapsed_seconds": time.monotonic() - wall_started,
            "layout_qualification": qualifications,
            "collision_diagnostics": diagnostics,
            "checkpoint_scoring_performed": True,
            "checkpoints": checkpoints,
            "verdict": (
                "confirmed" if all(item["passed"] for item in checkpoints) else "capability_failed"
            ),
        }
        atomic_write_json(output, report)
    except U1ConfirmationError as error:
        raise SystemExit(str(error)) from error
    sys.stdout.write(f"{report['verdict']}: {output}\n")
    if report["verdict"] != "confirmed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
