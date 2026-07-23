"""Isolated lesson layer for protocol v0.2 U2: Separated Unlock.

The confirmed U1 implementation is deliberately imported as a frozen dependency.
For Navigate, Visible Unlock, and Local Unlock this module delegates generation to
``v02_lessons.LessonEnv``.  Only the new Separated Unlock lesson has new behavior.

Oracle plans, seed roles, curriculum labels, and all other privileged fields in
this module are trainer-side evidence.  The policy-facing environment remains a
56 x 56 x 3 partial RGB image with the seven primitive MiniGrid actions.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import gymnasium as gym
import numpy as np
from minigrid.core.world_object import Door, Key, Wall
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from PIL import Image

from dungeon_apprentice.artifacts import utc_now
from dungeon_apprentice.contracts import (
    CURIOSITY_BUDGET,
    DEFAULT_CURIOSITY_SCALE,
    PIXEL_SHAPE,
    PIXEL_TILE_SIZE,
    STEP_REWARD,
    SUCCESS_REWARD,
    TRAINING_SEED_LIMIT,
    DungeonTier,
)
from dungeon_apprentice.env import (
    KEY_COLORS,
    EpisodicPixelCuriosity,
    LayoutMetadata,
    Position,
)
from dungeon_apprentice.oracle import DungeonOracle, OracleFailure
from dungeon_apprentice.u2_seed_guard import (
    U2AccessPhase,
    U2SeedAccess,
    U2SeedAccessError,
    U2SeedRole,
    authorize_u2_seed,
    classify_u2_seed,
)
from dungeon_apprentice.v02_lessons import (
    LessonEnv as FrozenU1LessonEnv,
)
from dungeon_apprentice.v02_lessons import (
    LessonId as FrozenU1LessonId,
)
from dungeon_apprentice.v02_lessons import (
    geometry_sha256,
    plan_unlock_candidate,
)

PROTOCOL = "dungeon-apprentice-v0.2-u2"
SCHEMA_VERSION = 4
GENERATOR_PROFILE_VERSION = 1
GENERATOR_PROFILE = "v0.2-u2-separated-v1"
EVALUATION_PANEL_SIZE = 40
EVALUATION_SEED_COUNT = EVALUATION_PANEL_SIZE * 2
ALLOCATION_TOLERANCE = 0.05
MAX_GENERATION_ATTEMPTS = 2_048
FULL_LAYOUT_UNIQUENESS_FLOOR = 0.99
GEOMETRY_UNIQUENESS_FLOOR = 0.95

ENGINEERING_SEED_BASE = 5_200_000
ENGINEERING_SEED_COUNT = 1_000
SEALED_QUALIFICATION_SEED_BASE = 5_210_000
SEALED_QUALIFICATION_SEED_COUNT = 2_000
U2_VALIDATION_SEED_BASE = 11_200_000


def validate_u2_seed_role(
    seed: int,
    seed_role: U2SeedRole | str,
    *,
    access: U2SeedAccess | None = None,
) -> U2SeedRole:
    """Require an explicit, exact role for every instantiated U2 seed."""

    selected = U2SeedRole(seed_role)
    value = int(seed)
    authorize_u2_seed(value, expected_role=selected, access=access)
    return selected


def _role_for_seed(seed: int) -> U2SeedRole:
    role = classify_u2_seed(int(seed))
    if role not in {
        U2SeedRole.TRAINING,
        U2SeedRole.ENGINEERING,
        U2SeedRole.SEALED_QUALIFICATION,
        U2SeedRole.U2_VALIDATION,
    }:
        raise ValueError(
            f"seed {seed} has no permitted U2 role; future confirmation and final "
            "partitions are intentionally unavailable"
        )
    return role


class U2LessonId(StrEnum):
    """Stable trainer-visible identifiers for the four cumulative lessons."""

    NAVIGATE = "navigate/full"
    VISIBLE_UNLOCK = "unlock/u0-visible"
    LOCAL_UNLOCK = "unlock/u1-local"
    SEPARATED_UNLOCK = "unlock/u2-separated"


# Compatibility name used by the U2 trainer.  It is a separate enum from the
# frozen U1 enum, so adding U2 cannot change any U1 iteration or registry.
LessonId = U2LessonId

PREREQUISITE_LESSONS = (
    U2LessonId.NAVIGATE,
    U2LessonId.VISIBLE_UNLOCK,
    U2LessonId.LOCAL_UNLOCK,
)


@dataclass(frozen=True)
class U2LessonSpec:
    lesson_id: U2LessonId
    label: str
    tier: DungeonTier
    prerequisites: tuple[U2LessonId, ...]
    generator_profile: str
    max_steps: int
    validation_seed_base: int
    overall_threshold: float
    panel_threshold: float
    overall_required: int
    panel_required: int
    oracle_action_range: tuple[int, int] | None
    key_visible: bool | None = None
    door_visible: bool | None = None


U2_LESSON_SPECS: Mapping[U2LessonId, U2LessonSpec] = MappingProxyType(
    {
        U2LessonId.NAVIGATE: U2LessonSpec(
            lesson_id=U2LessonId.NAVIGATE,
            label="Navigate",
            tier=DungeonTier.NAVIGATE,
            prerequisites=(),
            generator_profile="v0.1-navigate",
            max_steps=128,
            validation_seed_base=10_000_000,
            overall_threshold=0.85,
            panel_threshold=0.85,
            overall_required=68,
            panel_required=34,
            oracle_action_range=None,
        ),
        U2LessonId.VISIBLE_UNLOCK: U2LessonSpec(
            lesson_id=U2LessonId.VISIBLE_UNLOCK,
            label="Visible Unlock",
            tier=DungeonTier.UNLOCK,
            prerequisites=(U2LessonId.NAVIGATE,),
            generator_profile="v0.2-u0-visible-frozen",
            max_steps=128,
            validation_seed_base=11_000_000,
            overall_threshold=0.85,
            panel_threshold=0.80,
            overall_required=68,
            panel_required=32,
            oracle_action_range=(5, 10),
            key_visible=True,
            door_visible=True,
        ),
        U2LessonId.LOCAL_UNLOCK: U2LessonSpec(
            lesson_id=U2LessonId.LOCAL_UNLOCK,
            label="Local Unlock",
            tier=DungeonTier.UNLOCK,
            prerequisites=(
                U2LessonId.NAVIGATE,
                U2LessonId.VISIBLE_UNLOCK,
            ),
            generator_profile="v0.2-u1-local-v1",
            max_steps=128,
            validation_seed_base=11_100_000,
            overall_threshold=0.85,
            panel_threshold=0.80,
            overall_required=68,
            panel_required=32,
            oracle_action_range=(9, 18),
            key_visible=True,
            door_visible=False,
        ),
        U2LessonId.SEPARATED_UNLOCK: U2LessonSpec(
            lesson_id=U2LessonId.SEPARATED_UNLOCK,
            label="Separated Unlock",
            tier=DungeonTier.UNLOCK,
            prerequisites=PREREQUISITE_LESSONS,
            generator_profile=GENERATOR_PROFILE,
            max_steps=160,
            validation_seed_base=U2_VALIDATION_SEED_BASE,
            overall_threshold=0.85,
            panel_threshold=0.80,
            overall_required=68,
            panel_required=32,
            oracle_action_range=(17, 26),
        ),
    }
)

# Compatibility name expected by the trainer.
LESSON_SPECS = U2_LESSON_SPECS

_LEGACY_LESSONS: Mapping[U2LessonId, FrozenU1LessonId] = MappingProxyType(
    {
        U2LessonId.NAVIGATE: FrozenU1LessonId.NAVIGATE,
        U2LessonId.VISIBLE_UNLOCK: FrozenU1LessonId.VISIBLE_UNLOCK,
        U2LessonId.LOCAL_UNLOCK: FrozenU1LessonId.LOCAL_UNLOCK,
    }
)

_LESSON_SEED_ROLES: Mapping[U2LessonId, frozenset[U2SeedRole]] = MappingProxyType(
    {
        U2LessonId.NAVIGATE: frozenset(
            {
                U2SeedRole.TRAINING,
                U2SeedRole.NAVIGATE_VALIDATION,
                U2SeedRole.FUTURE_NAVIGATE_CONFIRMATION,
            }
        ),
        U2LessonId.VISIBLE_UNLOCK: frozenset(
            {
                U2SeedRole.TRAINING,
                U2SeedRole.U0_VALIDATION,
                U2SeedRole.FUTURE_U0_CONFIRMATION,
            }
        ),
        U2LessonId.LOCAL_UNLOCK: frozenset(
            {
                U2SeedRole.TRAINING,
                U2SeedRole.U1_VALIDATION,
                U2SeedRole.FUTURE_U1_CONFIRMATION,
            }
        ),
        U2LessonId.SEPARATED_UNLOCK: frozenset(
            {
                U2SeedRole.TRAINING,
                U2SeedRole.ENGINEERING,
                U2SeedRole.SEALED_QUALIFICATION,
                U2SeedRole.U2_VALIDATION,
                U2SeedRole.FUTURE_U2_CONFIRMATION,
            }
        ),
    }
)


def _authorize_lesson_seed(
    lesson: U2LessonId | str,
    seed: int,
    *,
    requested_role: U2SeedRole | str | None = None,
    access: U2SeedAccess | None = None,
) -> U2SeedRole:
    """Fail before generation unless the seed role belongs to this exact lesson."""

    selected_lesson = U2LessonId(lesson)
    candidate = int(seed)
    actual = classify_u2_seed(candidate)
    if actual is None:
        raise U2SeedAccessError(
            f"seed {candidate} is outside every frozen U2 partition"
        )
    selected_role = actual if requested_role is None else U2SeedRole(requested_role)
    authorize_u2_seed(
        candidate,
        expected_role=selected_role,
        access=access,
    )
    if selected_role not in _LESSON_SEED_ROLES[selected_lesson]:
        raise U2SeedAccessError(
            f"{selected_role.value} seeds cannot instantiate {selected_lesson.value}"
        )
    return selected_role


def _canonical_weak_lessons(
    lessons: Iterable[U2LessonId | str],
) -> tuple[U2LessonId, ...]:
    selected = {U2LessonId(lesson) for lesson in lessons}
    if not selected <= set(PREREQUISITE_LESSONS):
        raise ValueError("only Navigate, U0, and U1 can own U2 recovery")
    return tuple(lesson for lesson in PREREQUISITE_LESSONS if lesson in selected)


NORMAL_TARGETS: Mapping[U2LessonId, float] = MappingProxyType(
    {
        U2LessonId.NAVIGATE: 0.50,
        U2LessonId.VISIBLE_UNLOCK: 0.075,
        U2LessonId.LOCAL_UNLOCK: 0.075,
        U2LessonId.SEPARATED_UNLOCK: 0.35,
    }
)

RECOVERY_TARGETS: Mapping[frozenset[U2LessonId], Mapping[U2LessonId, float]] = (
    MappingProxyType(
        {
            frozenset({U2LessonId.NAVIGATE}): MappingProxyType(
                {
                    U2LessonId.NAVIGATE: 0.70,
                    U2LessonId.VISIBLE_UNLOCK: 0.10,
                    U2LessonId.LOCAL_UNLOCK: 0.10,
                    U2LessonId.SEPARATED_UNLOCK: 0.10,
                }
            ),
            frozenset({U2LessonId.VISIBLE_UNLOCK}): MappingProxyType(
                {
                    U2LessonId.NAVIGATE: 0.45,
                    U2LessonId.VISIBLE_UNLOCK: 0.35,
                    U2LessonId.LOCAL_UNLOCK: 0.10,
                    U2LessonId.SEPARATED_UNLOCK: 0.10,
                }
            ),
            frozenset({U2LessonId.LOCAL_UNLOCK}): MappingProxyType(
                {
                    U2LessonId.NAVIGATE: 0.45,
                    U2LessonId.VISIBLE_UNLOCK: 0.10,
                    U2LessonId.LOCAL_UNLOCK: 0.35,
                    U2LessonId.SEPARATED_UNLOCK: 0.10,
                }
            ),
            frozenset(
                {
                    U2LessonId.NAVIGATE,
                    U2LessonId.VISIBLE_UNLOCK,
                }
            ): MappingProxyType(
                {
                    U2LessonId.NAVIGATE: 0.55,
                    U2LessonId.VISIBLE_UNLOCK: 0.25,
                    U2LessonId.LOCAL_UNLOCK: 0.10,
                    U2LessonId.SEPARATED_UNLOCK: 0.10,
                }
            ),
            frozenset(
                {
                    U2LessonId.NAVIGATE,
                    U2LessonId.LOCAL_UNLOCK,
                }
            ): MappingProxyType(
                {
                    U2LessonId.NAVIGATE: 0.55,
                    U2LessonId.VISIBLE_UNLOCK: 0.10,
                    U2LessonId.LOCAL_UNLOCK: 0.25,
                    U2LessonId.SEPARATED_UNLOCK: 0.10,
                }
            ),
            frozenset(
                {
                    U2LessonId.VISIBLE_UNLOCK,
                    U2LessonId.LOCAL_UNLOCK,
                }
            ): MappingProxyType(
                {
                    U2LessonId.NAVIGATE: 0.40,
                    U2LessonId.VISIBLE_UNLOCK: 0.25,
                    U2LessonId.LOCAL_UNLOCK: 0.25,
                    U2LessonId.SEPARATED_UNLOCK: 0.10,
                }
            ),
            frozenset(PREREQUISITE_LESSONS): MappingProxyType(
                {
                    U2LessonId.NAVIGATE: 0.45,
                    U2LessonId.VISIBLE_UNLOCK: 0.225,
                    U2LessonId.LOCAL_UNLOCK: 0.225,
                    U2LessonId.SEPARATED_UNLOCK: 0.10,
                }
            ),
        }
    )
)

TARGET_PROFILES: Mapping[
    tuple[U2LessonId, ...],
    Mapping[U2LessonId, float],
] = MappingProxyType(
    {
        (): NORMAL_TARGETS,
        **{
            _canonical_weak_lessons(weak): targets
            for weak, targets in RECOVERY_TARGETS.items()
        },
    }
)


def target_profiles_public() -> dict[str, dict[str, float]]:
    """Return every frozen practice profile in sidecar-friendly form."""

    return {
        (
            "normal"
            if not weak
            else "recovery:" + "+".join(lesson.value for lesson in weak)
        ): {
            lesson.value: float(share)
            for lesson, share in targets.items()
        }
        for weak, targets in TARGET_PROFILES.items()
    }


@dataclass
class CurriculumState:
    """Four-lesson U2 curriculum with explicit prerequisite recovery."""

    active_lesson: U2LessonId = U2LessonId.SEPARATED_UNLOCK
    passed_lessons: tuple[U2LessonId, ...] = PREREQUISITE_LESSONS
    consecutive_passes: int = 0
    weak_prerequisites: tuple[U2LessonId, ...] = ()
    recovery_passes: int = 0
    revision: int = 0
    mastered: bool = False

    def __post_init__(self) -> None:
        self.active_lesson = U2LessonId(self.active_lesson)
        self.passed_lessons = tuple(U2LessonId(value) for value in self.passed_lessons)
        self.weak_prerequisites = _canonical_weak_lessons(self.weak_prerequisites)
        if self.active_lesson is not U2LessonId.SEPARATED_UNLOCK:
            raise ValueError("U2 curriculum has the wrong active lesson")
        if set(self.passed_lessons) != set(PREREQUISITE_LESSONS):
            raise ValueError("U2 curriculum must preserve all three prerequisites")
        for name in ("consecutive_passes", "recovery_passes", "revision"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")

    @property
    def recovery(self) -> bool:
        return bool(self.weak_prerequisites)

    @property
    def recovery_profile(self) -> str:
        if not self.weak_prerequisites:
            return "normal"
        return "recovery:" + "+".join(lesson.value for lesson in self.weak_prerequisites)

    def targets(self) -> dict[U2LessonId, float]:
        if not self.weak_prerequisites:
            return dict(NORMAL_TARGETS)
        return dict(RECOVERY_TARGETS[frozenset(self.weak_prerequisites)])

    def change(self) -> None:
        self.revision += 1

    def public_dict(self) -> dict[str, Any]:
        return {
            "active_lesson": self.active_lesson.value,
            "active_lesson_label": U2_LESSON_SPECS[self.active_lesson].label,
            "passed_lessons": [lesson.value for lesson in self.passed_lessons],
            "consecutive_passes": int(self.consecutive_passes),
            "recovery": self.recovery,
            "weak_prerequisites": [
                lesson.value for lesson in self.weak_prerequisites
            ],
            "recovery_cause": (
                [lesson.value for lesson in self.weak_prerequisites]
                if self.weak_prerequisites
                else None
            ),
            "recovery_profile": self.recovery_profile,
            "recovery_passes": int(self.recovery_passes),
            "revision": int(self.revision),
            "mastered": bool(self.mastered),
            "targets": {
                lesson.value: share for lesson, share in self.targets().items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CurriculumState:
        weak = value.get("weak_prerequisites")
        if weak is None:
            weak = value.get("recovery_cause") or ()
        if isinstance(weak, str):
            weak = (weak,)
        return cls(
            active_lesson=U2LessonId(str(value["active_lesson"])),
            passed_lessons=tuple(
                U2LessonId(str(lesson))
                for lesson in value.get("passed_lessons", ())
            ),
            consecutive_passes=int(value.get("consecutive_passes", 0)),
            weak_prerequisites=tuple(U2LessonId(str(lesson)) for lesson in weak),
            recovery_passes=int(value.get("recovery_passes", 0)),
            revision=int(value.get("revision", 0)),
            mastered=bool(value.get("mastered", False)),
        )


class TransitionDeficitScheduler:
    """Balance complete episodes against four-lesson transition targets."""

    def __init__(self, state: CurriculumState, *, seed: int) -> None:
        self.state = state
        self._rng = np.random.default_rng(seed)
        self._revision = state.revision
        self.window_transitions: dict[U2LessonId, int] = defaultdict(int)
        self.lifetime_transitions: dict[U2LessonId, int] = defaultdict(int)
        self._reservations: dict[U2LessonId, int] = defaultdict(int)

    def _sync(self) -> None:
        if self._revision == self.state.revision:
            return
        self._revision = self.state.revision
        self.window_transitions = defaultdict(int)
        self._reservations = defaultdict(int)

    def assign(self) -> tuple[U2LessonId, int]:
        self._sync()
        candidates: list[tuple[float, float, U2LessonId]] = []
        for lesson, target in self.state.targets().items():
            effective = self.window_transitions[lesson] + self._reservations[lesson]
            candidates.append((effective / target, float(self._rng.random()), lesson))
        _, _, lesson = min(candidates)
        reservation = U2_LESSON_SPECS[lesson].max_steps
        self._reservations[lesson] += reservation
        return lesson, reservation

    def record_step(self, lesson: U2LessonId | str) -> None:
        self._sync()
        selected = U2LessonId(lesson)
        self.window_transitions[selected] += 1
        self.lifetime_transitions[selected] += 1

    def release(self, lesson: U2LessonId | str, reservation: int) -> None:
        selected = U2LessonId(lesson)
        self._reservations[selected] = max(
            0,
            self._reservations[selected] - max(0, int(reservation)),
        )

    def snapshot(self) -> dict[str, Any]:
        self._sync()
        targets = self.state.targets()
        total = sum(self.window_transitions.values())
        return {
            "revision": self._revision,
            "profile": self.state.recovery_profile,
            "target_shares": {
                lesson.value: share for lesson, share in targets.items()
            },
            "window_transitions": {
                lesson.value: int(self.window_transitions.get(lesson, 0))
                for lesson in U2LessonId
            },
            "realized_shares": {
                lesson.value: (
                    self.window_transitions.get(lesson, 0) / total if total else 0.0
                )
                for lesson in U2LessonId
            },
            "lifetime_transitions": {
                lesson.value: int(self.lifetime_transitions.get(lesson, 0))
                for lesson in U2LessonId
            },
        }

    def state_dict(self) -> dict[str, Any]:
        return self.snapshot() | {"rng_state": self._rng.bit_generator.state}

    @staticmethod
    def _validated_count_map(
        value: Mapping[str, Any],
        *,
        description: str,
    ) -> dict[U2LessonId, int]:
        expected = {lesson.value for lesson in U2LessonId}
        if set(value) != expected:
            raise ValueError(f"{description} must contain exactly all four lessons")
        counts: dict[U2LessonId, int] = {}
        for lesson, raw in value.items():
            if isinstance(raw, bool) or int(raw) != raw or int(raw) < 0:
                raise ValueError(f"{description} contains an invalid count")
            counts[U2LessonId(lesson)] = int(raw)
        return counts

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if int(value.get("revision", -1)) != self.state.revision:
            raise ValueError("scheduler and curriculum revisions differ")
        expected_targets = {
            lesson.value: share for lesson, share in self.state.targets().items()
        }
        if value.get("target_shares") != expected_targets:
            raise ValueError("scheduler target shares do not match curriculum state")
        if value.get("profile") not in {None, self.state.recovery_profile}:
            raise ValueError("scheduler target profile does not match curriculum state")
        window = self._validated_count_map(
            value.get("window_transitions", {}),
            description="window transitions",
        )
        lifetime = self._validated_count_map(
            value.get("lifetime_transitions", {}),
            description="lifetime transitions",
        )
        if any(window[lesson] > lifetime[lesson] for lesson in U2LessonId):
            raise ValueError("window transitions cannot exceed lifetime transitions")
        rng_state = value.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise ValueError("scheduler state has no RNG state")
        self._revision = self.state.revision
        self.window_transitions = defaultdict(int, window)
        self.lifetime_transitions = defaultdict(int, lifetime)
        self._reservations = defaultdict(int)
        try:
            self._rng.bit_generator.state = dict(rng_state)
        except (TypeError, ValueError) as error:
            raise ValueError("scheduler RNG state is invalid") from error

    def start_window(self) -> None:
        self.window_transitions = defaultdict(int)
        self._reservations = defaultdict(int)

    def allocation_within(self, tolerance: float = ALLOCATION_TOLERANCE) -> bool:
        snapshot = self.snapshot()
        total = sum(snapshot["window_transitions"].values())
        if not total:
            return True
        return all(
            abs(snapshot["realized_shares"][lesson] - target) <= tolerance
            for lesson, target in snapshot["target_shares"].items()
        )


def _opposite_sides(
    candidate: Mapping[str, Any],
    *,
    position: Position,
) -> bool:
    divider_index = int(candidate["divider_index"])
    approach_from_low = bool(candidate["approach_from_low"])
    axis = position[0] if candidate["divider_orientation"] == "vertical" else position[1]
    return axis < divider_index if approach_from_low else axis > divider_index


def _locked_goal_reachable(
    candidate: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> bool:
    blocked = set(candidate["walls"]) | {tuple(candidate["door"])}
    start = tuple(candidate["start"])
    goal = tuple(candidate["goal"])
    frontier = [start]
    visited = {start}
    while frontier:
        x, y = frontier.pop()
        if (x, y) == goal:
            return True
        for neighbor in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            if (
                0 < neighbor[0] < width - 1
                and 0 < neighbor[1] < height - 1
                and neighbor not in blocked
                and neighbor not in visited
            ):
                visited.add(neighbor)
                frontier.append(neighbor)
    return False


def _sample_separated_unlock_candidate(
    env: FrozenU1LessonEnv,
    width: int,
    height: int,
) -> dict[str, Any]:
    vertical = bool(env.np_random.integers(0, 2))
    orientation = "vertical" if vertical else "horizontal"
    divider_index = int(env.np_random.integers(2, 7))
    door_lane = int(env.np_random.integers(1, 8))
    approach_from_low = bool(env.np_random.integers(0, 2))

    if vertical:
        door = (divider_index, door_lane)
        divider_walls = {
            (divider_index, y)
            for y in range(1, height - 1)
            if y != door_lane
        }
        approach = [
            (x, y)
            for x in (
                range(1, divider_index)
                if approach_from_low
                else range(divider_index + 1, width - 1)
            )
            for y in range(1, height - 1)
        ]
        far = [
            (x, y)
            for x in (
                range(divider_index + 1, width - 1)
                if approach_from_low
                else range(1, divider_index)
            )
            for y in range(1, height - 1)
        ]
    else:
        door = (door_lane, divider_index)
        divider_walls = {
            (x, divider_index)
            for x in range(1, width - 1)
            if x != door_lane
        }
        approach = [
            (x, y)
            for x in range(1, width - 1)
            for y in (
                range(1, divider_index)
                if approach_from_low
                else range(divider_index + 1, height - 1)
            )
        ]
        far = [
            (x, y)
            for x in range(1, width - 1)
            for y in (
                range(divider_index + 1, height - 1)
                if approach_from_low
                else range(1, divider_index)
            )
        ]

    start, key = env._sample(approach, 2)
    goal = env._choice(far)
    occupied = {start, key, goal, door} | divider_walls
    wall_pool = [
        (x, y)
        for x in range(1, width - 1)
        for y in range(1, height - 1)
        if (x, y) not in occupied
    ]
    extra_walls = set(env._sample(wall_pool, 2))
    return {
        "start": start,
        "start_direction": int(env.np_random.integers(0, 4)),
        "goal": goal,
        "key": key,
        "door": door,
        "relic": None,
        "key_color": env._choice(list(KEY_COLORS)),
        "walls": divider_walls | extra_walls,
        "divider_walls": divider_walls,
        "extra_walls": extra_walls,
        "divider_orientation": orientation,
        "divider_index": divider_index,
        "door_lane": door_lane,
        "approach_from_low": approach_from_low,
    }


def _validate_separated_candidate(
    candidate: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> tuple[Any, bool]:
    divider = set(candidate["divider_walls"])
    extras = set(candidate["extra_walls"])
    walls = set(candidate["walls"])
    door = tuple(candidate["door"])
    objects = {
        tuple(candidate["start"]),
        tuple(candidate["key"]),
        tuple(candidate["goal"]),
        door,
    }
    if len(extras) != 2 or divider & extras or walls != divider | extras:
        raise OracleFailure("Separated Unlock requires exactly two non-divider walls")
    if extras & objects or walls & objects:
        raise OracleFailure("wall overlaps a required object")
    if candidate["divider_orientation"] == "vertical":
        expected_divider = {
            (int(candidate["divider_index"]), y)
            for y in range(1, height - 1)
            if y != int(candidate["door_lane"])
        }
        expected_door = (
            int(candidate["divider_index"]),
            int(candidate["door_lane"]),
        )
    else:
        expected_divider = {
            (x, int(candidate["divider_index"]))
            for x in range(1, width - 1)
            if x != int(candidate["door_lane"])
        }
        expected_door = (
            int(candidate["door_lane"]),
            int(candidate["divider_index"]),
        )
    if divider != expected_divider or door != expected_door:
        raise OracleFailure("door is not the sole opening in a complete divider")
    if not _opposite_sides(candidate, position=tuple(candidate["start"])):
        raise OracleFailure("start is not on the approach side")
    if not _opposite_sides(candidate, position=tuple(candidate["key"])):
        raise OracleFailure("key is not on the approach side")
    if _opposite_sides(candidate, position=tuple(candidate["goal"])):
        raise OracleFailure("goal is not on the far side")
    locked_goal_reachable = _locked_goal_reachable(
        candidate,
        width=width,
        height=height,
    )
    if locked_goal_reachable:
        raise OracleFailure("goal is reachable while the door is locked")
    plan = plan_unlock_candidate(candidate, width=width, height=height)
    if not 17 <= len(plan.actions) <= 26:
        raise OracleFailure("Separated Unlock solution is outside 17..26 actions")
    if plan.post_key_turns < 1:
        raise OracleFailure("Separated Unlock has no post-key pre-door turn")
    return plan, locked_goal_reachable


class U2LessonEnv(FrozenU1LessonEnv):
    """Frozen inherited lessons plus the isolated Separated Unlock generator."""

    def __init__(
        self,
        *,
        lesson: U2LessonId | str = U2LessonId.SEPARATED_UNLOCK,
        size: int = 9,
        render_mode: str | None = None,
    ) -> None:
        selected = U2LessonId(lesson)
        self.u2_lesson = selected
        self.u2_seed_role: U2SeedRole | None = None
        self.u2_seed_access: U2SeedAccess | None = None
        self.generation_attempts = 0
        self.initial_key_visible: bool | None = None
        self.initial_door_visible: bool | None = None
        self.divider_orientation: str | None = None
        self.divider_index: int | None = None
        self.door_lane: int | None = None
        self.approach_from_low: bool | None = None
        self.extra_walls: tuple[Position, ...] = ()
        frozen = _LEGACY_LESSONS.get(selected, FrozenU1LessonId.LOCAL_UNLOCK)
        super().__init__(lesson=frozen, size=size, render_mode=render_mode)
        self.max_steps = U2_LESSON_SPECS[selected].max_steps

    def set_lesson(self, lesson: U2LessonId | str) -> None:
        selected = U2LessonId(lesson)
        self.u2_lesson = selected
        if selected in _LEGACY_LESSONS:
            super().set_lesson(_LEGACY_LESSONS[selected])
        else:
            # Keep the frozen parent's internal enum valid while the overridden
            # grid generator owns all new U2 behavior.
            super().set_lesson(FrozenU1LessonId.LOCAL_UNLOCK)
            self.max_steps = U2_LESSON_SPECS[selected].max_steps

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        clean_options = dict(options or {})
        requested_role = clean_options.pop("u2_seed_role", None)
        requested_access = clean_options.pop("u2_seed_access", None)
        if seed is None:
            raise ValueError("U2 lesson environments require an explicit layout seed")
        if requested_access is not None and not isinstance(
            requested_access, U2SeedAccess
        ):
            raise TypeError("u2_seed_access must be an issued U2SeedAccess capability")
        self.u2_seed_access = requested_access
        self.u2_seed_role = _authorize_lesson_seed(
            self.u2_lesson,
            int(seed),
            requested_role=requested_role,
            access=requested_access,
        )
        return super().reset(
            seed=seed,
            options=clean_options or None,
        )

    def _record_u2_layout(self, candidate: Mapping[str, Any]) -> None:
        digest = hashlib.sha256()
        digest.update(self.grid.encode().tobytes())
        digest.update(bytes((*self.agent_pos, self.agent_dir, int(self.tier))))
        digest.update(PROTOCOL.encode())
        digest.update(str(GENERATOR_PROFILE_VERSION).encode())
        digest.update(U2LessonId.SEPARATED_UNLOCK.value.encode())
        self.geometry_sha256 = geometry_sha256(candidate, size=self._size)
        self.layout = LayoutMetadata(
            protocol=PROTOCOL,
            tier=int(self.tier),
            tier_label=self.tier.label,
            seed=self._reset_seed,
            size=self._size,
            start=tuple(self.agent_pos),
            start_direction=int(self.agent_dir),
            goal=tuple(candidate["goal"]),
            key=tuple(candidate["key"]),
            door=tuple(candidate["door"]),
            relic=None,
            key_color=str(candidate["key_color"]),
            walls=tuple(sorted(candidate["walls"])),
            layout_sha256=digest.hexdigest(),
        )

    def _gen_grid(self, width: int, height: int) -> None:
        if self.u2_lesson is not U2LessonId.SEPARATED_UNLOCK:
            super()._gen_grid(width, height)
            return
        if width != 9 or height != 9:
            raise ValueError("Separated Unlock is qualified only at 9 x 9")
        self.pure_oracle_actions = None
        self.generation_attempts = 0
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            candidate = _sample_separated_unlock_candidate(self, width, height)
            try:
                plan, _ = _validate_separated_candidate(
                    candidate,
                    width=width,
                    height=height,
                )
            except OracleFailure:
                continue
            self._install_local_candidate(candidate, width=width, height=height)
            self.pure_oracle_actions = plan.actions
            self.generation_attempts = attempt
            self.divider_orientation = str(candidate["divider_orientation"])
            self.divider_index = int(candidate["divider_index"])
            self.door_lane = int(candidate["door_lane"])
            self.approach_from_low = bool(candidate["approach_from_low"])
            self.extra_walls = tuple(sorted(candidate["extra_walls"]))
            self.initial_key_visible = bool(self.agent_sees(*candidate["key"]))
            self.initial_door_visible = bool(self.agent_sees(*candidate["door"]))
            self._record_u2_layout(candidate)
            return
        raise RuntimeError(
            f"failed to generate Separated Unlock within {MAX_GENERATION_ATTEMPTS} attempts"
        )

    def _evidence_info(self, *, success: bool) -> dict[str, Any]:
        info = super()._evidence_info(success=success)
        if self.u2_lesson is not U2LessonId.SEPARATED_UNLOCK:
            return info
        info.update(
            {
                "protocol": PROTOCOL,
                "lesson_id": self.u2_lesson.value,
                "lesson_label": U2_LESSON_SPECS[self.u2_lesson].label,
                "generator_profile": GENERATOR_PROFILE,
                "generator_profile_version": GENERATOR_PROFILE_VERSION,
                "geometry_sha256": self.geometry_sha256,
                "u2_seed_role": (
                    self.u2_seed_role.value if self.u2_seed_role else None
                ),
                "generation_attempts": self.generation_attempts,
                "initial_key_visible": self.initial_key_visible,
                "initial_door_visible": self.initial_door_visible,
                "visibility_stratum": _visibility_stratum(
                    self.initial_key_visible,
                    self.initial_door_visible,
                ),
                "divider_orientation": self.divider_orientation,
                "divider_index": self.divider_index,
                "door_lane": self.door_lane,
                "approach_from_low": self.approach_from_low,
                "extra_wall_count": len(self.extra_walls),
            }
        )
        return info


# Compatibility alias requested by the trainer/preflight layers.
LessonEnv = U2LessonEnv


def make_u2_pixel_env(
    *,
    lesson: U2LessonId | str,
    size: int = 9,
    render_mode: str | None = "rgb_array",
) -> gym.Env:
    env: gym.Env = U2LessonEnv(
        lesson=U2LessonId(lesson),
        size=size,
        render_mode=render_mode,
    )
    env = RGBImgPartialObsWrapper(env, tile_size=PIXEL_TILE_SIZE)
    return ImgObsWrapper(env)


make_pixel_env = make_u2_pixel_env


class CurriculumEnv(gym.Wrapper):
    """Assign complete episodes while rejecting reserved exact layouts."""

    def __init__(
        self,
        env: U2LessonEnv,
        *,
        scheduler: TransitionDeficitScheduler,
        seed: int,
        seed_access: U2SeedAccess | None = None,
        forbidden_layout_hashes: (
            Mapping[U2LessonId, frozenset[str]] | frozenset[str] | None
        ) = None,
    ) -> None:
        super().__init__(env)
        self.scheduler = scheduler
        self._rng = np.random.default_rng(seed)
        self._seed_access = seed_access
        self._lesson: U2LessonId | None = None
        self._reservation = 0
        self._forbidden_layout_hashes = forbidden_layout_hashes or {}

    def _forbidden_for(self, lesson: U2LessonId) -> frozenset[str]:
        if isinstance(self._forbidden_layout_hashes, Mapping):
            return self._forbidden_layout_hashes.get(lesson, frozenset())
        return self._forbidden_layout_hashes

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        if self._lesson is not None:
            self.scheduler.release(self._lesson, self._reservation)
        requested_seed = kwargs.pop("seed", None)
        if requested_seed is not None:
            self._rng = np.random.default_rng(requested_seed)
        self._lesson, self._reservation = self.scheduler.assign()
        self.unwrapped.set_lesson(self._lesson)
        forbidden = self._forbidden_for(self._lesson)
        for rejected in range(128):
            episode_seed = int(self._rng.integers(0, TRAINING_SEED_LIMIT))
            options = dict(kwargs.pop("options", {}) or {})
            if self._lesson is U2LessonId.SEPARATED_UNLOCK:
                options["u2_seed_role"] = U2SeedRole.TRAINING.value
                if self._seed_access is not None:
                    options["u2_seed_access"] = self._seed_access
            observation, info = super().reset(
                seed=episode_seed,
                options=options or None,
                **kwargs,
            )
            if info.get("layout_sha256") not in forbidden:
                info["reserved_layout_rejections"] = rejected
                return observation, info
        raise RuntimeError("could not sample a training layout outside reserved evidence sets")

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        if self._lesson is None:
            raise RuntimeError("curriculum step occurred before lesson assignment")
        self.scheduler.record_step(self._lesson)
        if terminated or truncated:
            self.scheduler.release(self._lesson, self._reservation)
            self._lesson = None
            self._reservation = 0
        return observation, reward, terminated, truncated, info


def make_training_env(
    *,
    scheduler: TransitionDeficitScheduler,
    seed: int,
    size: int = 9,
    render_mode: str | None = "rgb_array",
    seed_access: U2SeedAccess | None = None,
    forbidden_layout_hashes: (
        Mapping[U2LessonId, frozenset[str]] | frozenset[str] | None
    ) = None,
) -> gym.Env:
    base = U2LessonEnv(
        lesson=U2LessonId.SEPARATED_UNLOCK,
        size=size,
        render_mode=render_mode,
    )
    env: gym.Env = CurriculumEnv(
        base,
        scheduler=scheduler,
        seed=seed,
        seed_access=seed_access,
        forbidden_layout_hashes=forbidden_layout_hashes,
    )
    env = RGBImgPartialObsWrapper(env, tile_size=PIXEL_TILE_SIZE)
    env = ImgObsWrapper(env)
    return EpisodicPixelCuriosity(
        env,
        scale=DEFAULT_CURIOSITY_SCALE,
        budget=CURIOSITY_BUDGET,
    )


@dataclass(frozen=True)
class U2LessonEvaluation:
    protocol: str
    timestamp: str
    lesson_id: str
    lesson_label: str
    episodes: int
    successes: int
    success_rate: float
    panel_successes: tuple[int, int]
    panel_success_rates: tuple[float, float]
    mean_steps: float
    success_wilson_interval: tuple[float, float] = (0.0, 1.0)
    panel_wilson_intervals: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 1.0),
        (0.0, 1.0),
    )
    milestone_rates: dict[str, float] = field(default_factory=dict)
    milestone_counts: dict[str, int] = field(default_factory=dict)
    door_given_key: float | None = None
    success_given_door: float | None = None
    visibility_strata: dict[str, dict[str, Any]] = field(default_factory=dict)
    mean_path_actions_per_oracle_action: float = 0.0
    mean_coverage: float = 0.0
    mean_collisions: float = 0.0
    mean_ineffective_interactions: float = 0.0
    action_histogram: dict[str, int] = field(default_factory=dict)
    largest_action_share: float = 0.0
    longest_repeated_action_run: int = 0
    mean_extrinsic_return: float = 0.0
    mean_curiosity_return: float = 0.0

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


LessonEvaluation = U2LessonEvaluation


def _wilson_interval(successes: int, episodes: int) -> tuple[float, float]:
    if episodes <= 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    proportion = successes / episodes
    denominator = 1 + z**2 / episodes
    center = (proportion + z**2 / (2 * episodes)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / episodes
            + z**2 / (4 * episodes**2)
        )
        / denominator
    )
    return (max(0.0, center - radius), min(1.0, center + radius))


def _policy_observation(observation: np.ndarray, model: Any) -> np.ndarray:
    expected = tuple(model.observation_space.shape)
    if tuple(observation.shape) == expected:
        return observation
    channels_first = np.transpose(observation, (2, 0, 1))
    if tuple(channels_first.shape) == expected:
        return channels_first
    raise ValueError(f"model expects {expected}, game produced {observation.shape}")


def _visibility_stratum(
    key_visible: bool | None,
    door_visible: bool | None,
) -> str:
    return (
        f"key_{'visible' if key_visible else 'hidden'}_"
        f"door_{'visible' if door_visible else 'hidden'}"
    )


def _reachable_cell_count(env: FrozenU1LessonEnv) -> int:
    count = 0
    for x in range(1, env.width - 1):
        for y in range(1, env.height - 1):
            if not isinstance(env.grid.get(x, y), Wall):
                count += 1
    return count


def _longest_action_run(actions: Sequence[int]) -> int:
    longest = 0
    current = 0
    prior: int | None = None
    for action in actions:
        if action == prior:
            current += 1
        else:
            prior = action
            current = 1
        longest = max(longest, current)
    return longest


def _reset_lesson_env(
    env: gym.Env,
    lesson: U2LessonId,
    seed: int,
    *,
    seed_access: U2SeedAccess | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    role = _authorize_lesson_seed(
        lesson,
        seed,
        access=seed_access,
    )
    return env.reset(
        seed=seed,
        options={
            "u2_seed_role": role.value,
            "u2_seed_access": seed_access,
        },
    )


def _oracle_action_count(
    lesson: U2LessonId,
    seed: int,
    *,
    size: int,
    seed_access: U2SeedAccess | None,
) -> int:
    oracle_env = U2LessonEnv(lesson=lesson, size=size)
    try:
        role = _authorize_lesson_seed(
            lesson,
            seed,
            access=seed_access,
        )
        oracle_env.reset(
            seed=seed,
            options={
                "u2_seed_role": role.value,
                "u2_seed_access": seed_access,
            },
        )
        return DungeonOracle(oracle_env).solve().steps
    finally:
        oracle_env.close()


def aggregate_u2_case_evidence(
    lesson: U2LessonId | str,
    cases: Sequence[Mapping[str, Any]],
    *,
    timestamp: str | None = None,
) -> U2LessonEvaluation:
    """Recompute one lesson evaluation solely from immutable per-case evidence."""

    selected = U2LessonId(lesson)
    outcomes = [dict(case) for case in cases]
    if not outcomes:
        raise ValueError("evaluation requires at least one case")
    midpoint = len(outcomes) // 2
    expected_histogram_keys = {str(action) for action in range(7)}
    seeds: set[int] = set()
    for index, outcome in enumerate(outcomes):
        if outcome.get("lesson_id") != selected.value:
            raise ValueError("case evidence lesson identity changed")
        if int(outcome.get("case_index", -1)) != index:
            raise ValueError("case evidence index is not contiguous")
        expected_panel = 0 if index < midpoint else 1
        if int(outcome.get("panel", -1)) != expected_panel:
            raise ValueError("case evidence panel assignment changed")
        seed = int(outcome.get("seed", -1))
        if seed < 0 or seed in seeds:
            raise ValueError("case evidence seeds are invalid or duplicated")
        seeds.add(seed)
        if not isinstance(outcome.get("success"), bool):
            raise ValueError("case evidence success is not boolean")
        for name in (
            "steps",
            "oracle_actions",
            "collisions",
            "ineffective_interactions",
            "unique_cells",
            "reachable_cells",
            "longest_repeated_action_run",
        ):
            raw = outcome.get(name)
            if isinstance(raw, bool) or int(raw) != raw or int(raw) < 0:
                raise ValueError(f"case evidence {name} is invalid")
        if int(outcome["steps"]) <= 0 or int(outcome["oracle_actions"]) <= 0:
            raise ValueError("case evidence action counts must be positive")
        if int(outcome["reachable_cells"]) <= 0:
            raise ValueError("case evidence reachable-cell count must be positive")
        milestones = outcome.get("milestones")
        if not isinstance(milestones, Mapping):
            raise ValueError("case evidence milestones are missing")
        histogram = outcome.get("action_histogram")
        if not isinstance(histogram, Mapping) or set(histogram) != expected_histogram_keys:
            raise ValueError("case evidence action histogram is incomplete")
        histogram_values = []
        for raw in histogram.values():
            if isinstance(raw, bool) or int(raw) != raw or int(raw) < 0:
                raise ValueError("case evidence action histogram is invalid")
            histogram_values.append(int(raw))
        if sum(histogram_values) != int(outcome["steps"]):
            raise ValueError("case evidence action histogram disagrees with steps")
        for name in (
            "path_actions_per_oracle_action",
            "coverage",
            "extrinsic_return",
            "curiosity_return",
        ):
            value = float(outcome.get(name, math.nan))
            if not math.isfinite(value):
                raise ValueError(f"case evidence {name} is not finite")
        expected_ratio = int(outcome["steps"]) / int(outcome["oracle_actions"])
        if not math.isclose(
            float(outcome["path_actions_per_oracle_action"]),
            expected_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("case evidence path ratio disagrees with action counts")
        expected_coverage = int(outcome["unique_cells"]) / int(
            outcome["reachable_cells"]
        )
        if not math.isclose(
            float(outcome["coverage"]),
            expected_coverage,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("case evidence coverage disagrees with cell counts")

    panels = (outcomes[:midpoint], outcomes[midpoint:])
    panel_successes = tuple(
        sum(bool(item["success"]) for item in panel) for panel in panels
    )
    successes = sum(bool(item["success"]) for item in outcomes)
    milestone_names = sorted(
        {
            str(name)
            for outcome in outcomes
            for name in outcome["milestones"]
        }
    )
    milestone_counts = {
        name: sum(
            bool(outcome["milestones"].get(name, False))
            for outcome in outcomes
        )
        for name in milestone_names
    }
    key_count = milestone_counts.get("key_picked_up", 0)
    door_count = milestone_counts.get("door_opened", 0)
    success_count = milestone_counts.get("success", successes)
    strata: dict[str, dict[str, Any]] = {}
    if selected is U2LessonId.SEPARATED_UNLOCK:
        for label in (
            "key_hidden_door_hidden",
            "key_hidden_door_visible",
            "key_visible_door_hidden",
            "key_visible_door_visible",
        ):
            stratum_cases = [
                item
                for item in outcomes
                if item.get("visibility_stratum") == label
            ]
            stratum_successes = sum(
                bool(item["success"]) for item in stratum_cases
            )
            strata[label] = {
                "episodes": len(stratum_cases),
                "successes": stratum_successes,
                "success_rate": (
                    stratum_successes / len(stratum_cases)
                    if stratum_cases
                    else None
                ),
                "wilson_interval": (
                    _wilson_interval(stratum_successes, len(stratum_cases))
                    if stratum_cases
                    else None
                ),
            }
    action_counts: Counter[int] = Counter()
    for outcome in outcomes:
        action_counts.update(
            {
                int(action): int(count)
                for action, count in outcome["action_histogram"].items()
            }
        )
    total_actions = sum(action_counts.values())
    return U2LessonEvaluation(
        protocol=PROTOCOL,
        timestamp=timestamp or utc_now(),
        lesson_id=selected.value,
        lesson_label=U2_LESSON_SPECS[selected].label,
        episodes=len(outcomes),
        successes=successes,
        success_rate=successes / len(outcomes),
        panel_successes=panel_successes,
        panel_success_rates=tuple(
            count / len(panel) if panel else 0.0
            for count, panel in zip(panel_successes, panels, strict=True)
        ),
        mean_steps=sum(int(item["steps"]) for item in outcomes) / len(outcomes),
        success_wilson_interval=_wilson_interval(successes, len(outcomes)),
        panel_wilson_intervals=tuple(
            _wilson_interval(count, len(panel))
            for count, panel in zip(panel_successes, panels, strict=True)
        ),
        milestone_rates={
            name: count / len(outcomes)
            for name, count in milestone_counts.items()
        },
        milestone_counts=milestone_counts,
        door_given_key=(door_count / key_count if key_count else None),
        success_given_door=(success_count / door_count if door_count else None),
        visibility_strata=strata,
        mean_path_actions_per_oracle_action=(
            sum(
                float(item["path_actions_per_oracle_action"])
                for item in outcomes
            )
            / len(outcomes)
        ),
        mean_coverage=(
            sum(float(item["coverage"]) for item in outcomes) / len(outcomes)
        ),
        mean_collisions=(
            sum(int(item["collisions"]) for item in outcomes) / len(outcomes)
        ),
        mean_ineffective_interactions=(
            sum(int(item["ineffective_interactions"]) for item in outcomes)
            / len(outcomes)
        ),
        action_histogram={
            str(action): int(action_counts.get(action, 0))
            for action in range(7)
        },
        largest_action_share=(
            max(action_counts.values(), default=0) / total_actions
            if total_actions
            else 0.0
        ),
        longest_repeated_action_run=max(
            int(item["longest_repeated_action_run"]) for item in outcomes
        ),
        mean_extrinsic_return=(
            sum(float(item["extrinsic_return"]) for item in outcomes)
            / len(outcomes)
        ),
        mean_curiosity_return=(
            sum(float(item["curiosity_return"]) for item in outcomes)
            / len(outcomes)
        ),
    )


def evaluate_u2_lesson(
    model: Any,
    lesson: U2LessonId | str,
    seeds: Iterable[int],
    *,
    size: int = 9,
    frame_path: Path | None = None,
    seed_access: U2SeedAccess | None = None,
    case_evidence_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> U2LessonEvaluation:
    """Evaluate pixels-only actions with privileged diagnostics kept out of policy input."""

    selected = U2LessonId(lesson)
    seed_values = tuple(int(raw_seed) for raw_seed in seeds)
    midpoint = len(seed_values) // 2
    outcomes: list[dict[str, Any]] = []
    latest: np.ndarray | None = None
    for case_index, seed in enumerate(seed_values):
        env = make_u2_pixel_env(lesson=selected, size=size)
        try:
            observation, reset_info = _reset_lesson_env(
                env,
                selected,
                seed,
                seed_access=seed_access,
            )
            base = env.unwrapped
            if not isinstance(base, U2LessonEnv):
                raise TypeError("U2 pixel wrapper lost its lesson environment")
            recurrent_state = None
            episode_start = np.ones((1,), dtype=bool)
            actions: list[int] = []
            extrinsic_return = 0.0
            curiosity_return = 0.0
            info: dict[str, Any] = {}
            while True:
                action, recurrent_state = model.predict(
                    _policy_observation(observation, model),
                    state=recurrent_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                episode_start[:] = False
                selected_action = int(np.asarray(action).item())
                actions.append(selected_action)
                observation, reward, terminated, truncated, info = env.step(
                    selected_action
                )
                extrinsic_return += float(info.get("extrinsic_reward", reward))
                curiosity_return += float(info.get("curiosity_reward", 0.0))
                if terminated or truncated:
                    break
            latest = observation
            oracle_actions = _oracle_action_count(
                selected,
                seed,
                size=size,
                seed_access=seed_access,
            )
            reachable = _reachable_cell_count(base)
            key_visible = reset_info.get("initial_key_visible")
            door_visible = reset_info.get("initial_door_visible")
            per_case_actions = Counter(actions)
            outcome = {
                "lesson_id": selected.value,
                "lesson_label": U2_LESSON_SPECS[selected].label,
                "case_index": case_index,
                "panel": 0 if case_index < midpoint else 1,
                "panel_case_index": (
                    case_index if case_index < midpoint else case_index - midpoint
                ),
                "seed": seed,
                "layout_sha256": str(reset_info.get("layout_sha256", "")),
                "geometry_sha256": str(
                    reset_info.get("geometry_sha256", "")
                ),
                "success": bool(info.get("success", False)),
                "terminal_reason": info.get("terminal_reason"),
                "steps": len(actions),
                "milestones": dict(info.get("milestones", {})),
                "collisions": int(info.get("collisions", 0)),
                "ineffective_interactions": int(
                    info.get("ineffective_interactions", 0)
                ),
                "unique_cells": int(info.get("unique_cells", 0)),
                "reachable_cells": int(reachable),
                "coverage": (
                    int(info.get("unique_cells", 0)) / reachable
                    if reachable
                    else 0.0
                ),
                "oracle_actions": int(oracle_actions),
                "path_actions_per_oracle_action": (
                    len(actions) / oracle_actions
                ),
                "action_histogram": {
                    str(action): int(per_case_actions.get(action, 0))
                    for action in range(7)
                },
                "longest_repeated_action_run": _longest_action_run(actions),
                "extrinsic_return": float(extrinsic_return),
                "curiosity_return": float(curiosity_return),
                "initial_key_visible": (
                    bool(key_visible) if key_visible is not None else None
                ),
                "initial_door_visible": (
                    bool(door_visible) if door_visible is not None else None
                ),
                "visibility_stratum": (
                    _visibility_stratum(key_visible, door_visible)
                    if selected is U2LessonId.SEPARATED_UNLOCK
                    else None
                ),
            }
            outcomes.append(outcome)
            if case_evidence_sink is not None:
                case_evidence_sink(dict(outcome))
        finally:
            env.close()
    if not outcomes:
        raise ValueError("evaluation requires at least one seed")
    if frame_path is not None and latest is not None:
        temporary = frame_path.with_name(f".{frame_path.name}.tmp")
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(latest).save(temporary, format="PNG")
        os.replace(temporary, frame_path)

    return aggregate_u2_case_evidence(
        selected,
        outcomes,
        timestamp=utc_now(),
    )


evaluate_lesson = evaluate_u2_lesson


def u2_lesson_passed(result: U2LessonEvaluation) -> bool:
    spec = U2_LESSON_SPECS[U2LessonId(result.lesson_id)]
    return result.success_rate >= spec.overall_threshold and all(
        rate >= spec.panel_threshold for rate in result.panel_success_rates
    )


lesson_passed = u2_lesson_passed


def weak_prerequisites(
    evaluations: Mapping[U2LessonId, U2LessonEvaluation],
) -> tuple[U2LessonId, ...]:
    return tuple(
        lesson
        for lesson in PREREQUISITE_LESSONS
        if not u2_lesson_passed(evaluations[lesson])
    )


def u2_validation_seeds(
    count: int = EVALUATION_SEED_COUNT,
    *,
    access: U2SeedAccess,
) -> tuple[int, ...]:
    if count != EVALUATION_SEED_COUNT:
        raise ValueError(
            f"U2 freezes exactly {EVALUATION_SEED_COUNT} validation cases"
        )
    seeds = tuple(U2_VALIDATION_SEED_BASE + offset for offset in range(count))
    for seed in seeds:
        authorize_u2_seed(
            seed,
            expected_role=U2SeedRole.U2_VALIDATION,
            access=access,
        )
    return seeds


def validation_seeds(
    lesson: U2LessonId | str,
    count: int = EVALUATION_SEED_COUNT,
    *,
    access: U2SeedAccess,
) -> tuple[int, ...]:
    if count != EVALUATION_SEED_COUNT:
        raise ValueError(
            f"U2 freezes exactly {EVALUATION_SEED_COUNT} cases per lesson"
        )
    selected = U2LessonId(lesson)
    base = U2_LESSON_SPECS[selected].validation_seed_base
    role = {
        U2LessonId.NAVIGATE: U2SeedRole.NAVIGATE_VALIDATION,
        U2LessonId.VISIBLE_UNLOCK: U2SeedRole.U0_VALIDATION,
        U2LessonId.LOCAL_UNLOCK: U2SeedRole.U1_VALIDATION,
        U2LessonId.SEPARATED_UNLOCK: U2SeedRole.U2_VALIDATION,
    }[selected]
    seeds = tuple(base + offset for offset in range(count))
    for seed in seeds:
        authorize_u2_seed(seed, expected_role=role, access=access)
    return seeds


def reserved_training_layout_hashes(
    qualification_report: Mapping[str, Any],
    *,
    access: U2SeedAccess,
) -> Mapping[U2LessonId, frozenset[str]]:
    """Build the global exact-layout exclusion set after qualified access exists.

    Calling this function intentionally instantiates all four frozen validation
    suites.  It therefore requires a seed capability bound to an authenticated
    qualification report and has no unguarded/default path.
    """

    if access.phase is not U2AccessPhase.QUALIFIED_TRAINING:
        raise ValueError(
            "reserved training hashes require qualified-training seed access"
        )
    partition = qualification_report.get("seed_partition")
    if (
        qualification_report.get("protocol") != PROTOCOL
        or qualification_report.get("lesson_id")
        != U2LessonId.SEPARATED_UNLOCK.value
        or not isinstance(partition, Mapping)
        or partition.get("role") != U2SeedRole.SEALED_QUALIFICATION.value
        or qualification_report.get("result") != "passed"
    ):
        raise ValueError("qualification report is not a passed sealed U2 report")
    cases = qualification_report.get("cases")
    if not isinstance(cases, list) or len(cases) != SEALED_QUALIFICATION_SEED_COUNT:
        raise ValueError("sealed qualification report must contain 2,000 cases")
    reserved = {
        str(case["layout_sha256"])
        for case in cases
        if isinstance(case, Mapping)
    }
    if len(cases) != sum(isinstance(case, Mapping) for case in cases):
        raise ValueError("sealed qualification report contains an invalid case")

    for lesson in U2LessonId:
        for seed in validation_seeds(lesson, access=access):
            env = U2LessonEnv(lesson=lesson)
            try:
                role = {
                    U2LessonId.NAVIGATE: U2SeedRole.NAVIGATE_VALIDATION,
                    U2LessonId.VISIBLE_UNLOCK: U2SeedRole.U0_VALIDATION,
                    U2LessonId.LOCAL_UNLOCK: U2SeedRole.U1_VALIDATION,
                    U2LessonId.SEPARATED_UNLOCK: U2SeedRole.U2_VALIDATION,
                }[lesson]
                env.reset(
                    seed=seed,
                    options={
                        "u2_seed_role": role.value,
                        "u2_seed_access": access,
                    },
                )
                if env.layout is None:
                    raise RuntimeError("validation environment generated no layout")
                reserved.add(env.layout.layout_sha256)
            finally:
                env.close()
    frozen = frozenset(reserved)
    # A global union follows the protocol literally: an exact layout reserved
    # by any suite is rejected from every training lesson.
    return MappingProxyType({lesson: frozen for lesson in U2LessonId})


def generate_u2_case_evidence(
    seed: int,
    *,
    seed_role: U2SeedRole | str,
    access: U2SeedAccess | None = None,
    size: int = 9,
) -> dict[str, Any]:
    """Generate and independently live-solve one U2 case without action export."""

    role = validate_u2_seed_role(seed, seed_role, access=access)
    inspection_pixel = make_u2_pixel_env(
        lesson=U2LessonId.SEPARATED_UNLOCK,
        size=size,
    )
    inspection = inspection_pixel.unwrapped
    if not isinstance(inspection, U2LessonEnv):
        inspection_pixel.close()
        raise TypeError("U2 pixel wrapper lost its lesson environment")
    live = U2LessonEnv(
        lesson=U2LessonId.SEPARATED_UNLOCK,
        size=size,
    )
    mechanics = U2LessonEnv(
        lesson=U2LessonId.SEPARATED_UNLOCK,
        size=size,
    )
    try:
        observation, info = inspection_pixel.reset(
            seed=int(seed),
            options={
                "u2_seed_role": role.value,
                "u2_seed_access": access,
            },
        )
        live.reset(
            seed=int(seed),
            options={
                "u2_seed_role": role.value,
                "u2_seed_access": access,
            },
        )
        mechanics.reset(
            seed=int(seed),
            options={
                "u2_seed_role": role.value,
                "u2_seed_access": access,
            },
        )
        if inspection.layout is None or live.layout is None:
            raise OracleFailure("Separated Unlock generated no layout")
        if inspection.layout != live.layout:
            raise OracleFailure("Separated Unlock is not deterministic")
        if inspection.pure_oracle_actions is None:
            raise OracleFailure("Separated Unlock has no pure planner result")
        pure_count = len(inspection.pure_oracle_actions)
        live_result = DungeonOracle(live).solve()
        if pure_count != live_result.steps:
            raise OracleFailure("pure planner and live oracle action counts differ")
        if not 17 <= live_result.steps <= 26:
            raise OracleFailure("live oracle action count is outside 17..26")
        reward_contract = math.isclose(
            live_result.total_reward,
            STEP_REWARD * (live_result.steps - 1) + SUCCESS_REWARD,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        if not reward_contract:
            raise OracleFailure("Separated Unlock reward contract changed")
        observation_contract = (
            observation.shape == PIXEL_SHAPE
            and observation.dtype == np.uint8
            and int(inspection.action_space.n) == 7
        )
        if not observation_contract:
            raise OracleFailure("Separated Unlock observation contract changed")
        door = live.grid.get(*live.layout.door)
        if (
            not isinstance(door, Door)
            or live.layout.key_color != door.color
            or not live_result.success
        ):
            raise OracleFailure("Separated Unlock matching-door mechanics changed")
        if mechanics.layout is None or mechanics.layout.door is None:
            raise OracleFailure("mechanics check generated no door")
        if mechanics.divider_orientation == "vertical":
            mechanics.agent_pos = (
                mechanics.divider_index - 1
                if mechanics.approach_from_low
                else mechanics.divider_index + 1,
                mechanics.door_lane,
            )
            mechanics.agent_dir = 0 if mechanics.approach_from_low else 2
        else:
            mechanics.agent_pos = (
                mechanics.door_lane,
                mechanics.divider_index - 1
                if mechanics.approach_from_low
                else mechanics.divider_index + 1,
            )
            mechanics.agent_dir = 1 if mechanics.approach_from_low else 3
        mechanics.carrying = None
        mechanics.step(int(mechanics.actions.toggle))
        mechanics_door = mechanics.grid.get(*mechanics.layout.door)
        no_key_rejected = bool(
            isinstance(mechanics_door, Door)
            and mechanics_door.is_locked
            and not mechanics_door.is_open
        )
        wrong_color = next(
            color
            for color in KEY_COLORS
            if color != mechanics.layout.key_color
        )
        mechanics.carrying = Key(wrong_color)
        mechanics.step(int(mechanics.actions.toggle))
        wrong_key_rejected = bool(
            isinstance(mechanics_door, Door)
            and mechanics_door.is_locked
            and not mechanics_door.is_open
        )
        door_requires_matching_key = no_key_rejected and wrong_key_rejected
        if not door_requires_matching_key:
            raise OracleFailure("locked door accepted no key or the wrong key")
        plan = plan_unlock_candidate(
            {
                "start": inspection.layout.start,
                "start_direction": inspection.layout.start_direction,
                "goal": inspection.layout.goal,
                "key": inspection.layout.key,
                "door": inspection.layout.door,
                "walls": inspection.layout.walls,
            },
            width=size,
            height=size,
        )
        extra_walls = tuple(sorted(inspection.extra_walls))
        divider_walls = tuple(
            sorted(set(inspection.layout.walls) - set(extra_walls))
        )
        qualification_topology = {
            "size": int(size),
            "max_steps": int(live.max_steps),
            "start": list(inspection.layout.start),
            "key": list(inspection.layout.key),
            "goal": list(inspection.layout.goal),
            "door": list(inspection.layout.door),
            "walls": [list(position) for position in inspection.layout.walls],
            "divider_walls": [list(position) for position in divider_walls],
            "extra_walls": [list(position) for position in extra_walls],
            "divider_orientation": inspection.divider_orientation,
            "divider_index": inspection.divider_index,
            "door_lane": inspection.door_lane,
            "approach_from_low": inspection.approach_from_low,
        }
        locked_goal_reachable = _locked_goal_reachable(
            {
                "start": inspection.layout.start,
                "goal": inspection.layout.goal,
                "door": inspection.layout.door,
                "walls": inspection.layout.walls,
            },
            width=size,
            height=size,
        )
        return {
            "protocol": PROTOCOL,
            "generator_profile": GENERATOR_PROFILE,
            "generator_profile_version": GENERATOR_PROFILE_VERSION,
            "seed": int(seed),
            "seed_role": role.value,
            "layout_sha256": inspection.layout.layout_sha256,
            "geometry_sha256": inspection.geometry_sha256,
            "pure_planner_actions": pure_count,
            "live_oracle_actions": live_result.steps,
            "planner_oracle_counts_match": pure_count == live_result.steps,
            "planner_live_action_match": pure_count == live_result.steps,
            "post_key_pre_door_turns": plan.post_key_turns,
            "post_key_turn_present": plan.post_key_turns >= 1,
            "initial_key_visible": inspection.initial_key_visible,
            "initial_door_visible": inspection.initial_door_visible,
            "visibility_stratum": info["visibility_stratum"],
            "divider_orientation": inspection.divider_orientation,
            "divider_index": inspection.divider_index,
            "door_lane": inspection.door_lane,
            "approach_from_low": inspection.approach_from_low,
            "key_color": inspection.layout.key_color,
            "extra_wall_count": len(extra_walls),
            "exactly_two_extra_walls": len(extra_walls) == 2,
            "goal_reachable_while_locked": locked_goal_reachable,
            "goal_blocked_while_locked": not locked_goal_reachable,
            "door_requires_matching_key": door_requires_matching_key,
            "observation_shape": list(observation.shape),
            "observation_dtype": str(observation.dtype),
            "action_count": int(inspection.action_space.n),
            "observation_contract": observation_contract,
            "reward_contract": reward_contract,
            "terminal_reason": live_result.terminal_reason,
            "terminated": (
                live_result.success and live_result.terminal_reason == "success"
            ),
            "truncated": live_result.terminal_reason == "time_limit",
            "elapsed_steps": int(live.step_count),
            "success": live_result.success,
            "ordered_objective_completed": live_result.success,
            "generation_attempts": inspection.generation_attempts,
            "max_generation_attempts": MAX_GENERATION_ATTEMPTS,
            "qualification_topology": qualification_topology,
        }
    finally:
        inspection_pixel.close()
        live.close()
        mechanics.close()


def qualify_separated_unlock(
    seeds: Iterable[int],
    *,
    seed_role: U2SeedRole | str,
    access: U2SeedAccess | None = None,
    validation_layout_hashes: frozenset[str] | None = None,
    size: int = 9,
) -> dict[str, Any]:
    """Qualify an explicit U2 seed block without choosing or opening a block."""

    role = U2SeedRole(seed_role)
    selected_seeds = tuple(int(seed) for seed in seeds)
    if not selected_seeds:
        raise ValueError("qualification requires at least one explicit seed")
    for seed in selected_seeds:
        validate_u2_seed_role(seed, role, access=access)
    if role is U2SeedRole.SEALED_QUALIFICATION:
        if len(selected_seeds) != SEALED_QUALIFICATION_SEED_COUNT:
            raise ValueError("sealed U2 qualification requires exactly 2,000 seeds")
        if validation_layout_hashes is None:
            raise ValueError(
                "sealed qualification requires the already authorized U2 "
                "validation exact-layout set"
            )

    cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed in selected_seeds:
        try:
            cases.append(
                generate_u2_case_evidence(
                    seed,
                    seed_role=role,
                    access=access,
                    size=size,
                )
            )
        except (AssertionError, OracleFailure, RuntimeError, ValueError) as error:
            failures.append({"seed": seed, "error": str(error)})
    exact_hashes = {
        str(case["layout_sha256"])
        for case in cases
    }
    geometry_hashes = {
        str(case["geometry_sha256"])
        for case in cases
    }
    visibility = Counter(str(case["visibility_stratum"]) for case in cases)
    required_exact = math.ceil(
        len(selected_seeds) * FULL_LAYOUT_UNIQUENESS_FLOOR
    )
    required_geometry = math.ceil(
        len(selected_seeds) * GEOMETRY_UNIQUENESS_FLOOR
    )
    if len(exact_hashes) < required_exact:
        failures.append(
            {
                "seed": None,
                "error": (
                    f"only {len(exact_hashes)}/{len(selected_seeds)} exact layouts "
                    f"were unique; requires {required_exact}"
                ),
            }
        )
    if len(geometry_hashes) < required_geometry:
        failures.append(
            {
                "seed": None,
                "error": (
                    f"only {len(geometry_hashes)}/{len(selected_seeds)} geometries "
                    f"were unique; requires {required_geometry}"
                ),
            }
        )
    overlap = (
        exact_hashes & set(validation_layout_hashes)
        if validation_layout_hashes is not None
        else set()
    )
    if overlap:
        failures.append(
            {
                "seed": None,
                "error": (
                    f"qualification overlaps {len(overlap)} exact validation layouts"
                ),
            }
        )
    return {
        "protocol": PROTOCOL,
        "generator_profile": GENERATOR_PROFILE,
        "generator_profile_version": GENERATOR_PROFILE_VERSION,
        "lesson_id": U2LessonId.SEPARATED_UNLOCK.value,
        "seed_role": role.value,
        "seed_partition": {
            "start": min(selected_seeds),
            "end": max(selected_seeds),
            "count": len(selected_seeds),
        },
        "requested": len(selected_seeds),
        "solved": len(cases),
        "full_unique_layouts": len(exact_hashes),
        "required_full_unique_layouts": required_exact,
        "geometry_unique_layouts": len(geometry_hashes),
        "required_geometry_unique_layouts": required_geometry,
        "minimum_oracle_actions": (
            min(int(case["live_oracle_actions"]) for case in cases)
            if cases
            else None
        ),
        "maximum_oracle_actions": (
            max(int(case["live_oracle_actions"]) for case in cases)
            if cases
            else None
        ),
        "visibility_strata": dict(sorted(visibility.items())),
        "validation_exact_overlap": len(overlap),
        "cases": cases,
        "failures": failures,
        "result": "passed" if not failures else "failed",
    }
