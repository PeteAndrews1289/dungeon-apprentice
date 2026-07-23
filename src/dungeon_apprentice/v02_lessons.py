"""Declarative lessons and generators for the warm-start U1 curriculum.

This module deliberately lives beside, rather than inside, ``v02_sentinel``.  The
confirmed U0 implementation stays frozen while this child protocol can grow into
an arbitrary lesson graph.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import gymnasium as gym
import numpy as np
from minigrid.core.constants import DIR_TO_VEC
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from PIL import Image

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
    DungeonApprenticeEnv,
    EpisodicPixelCuriosity,
    LayoutMetadata,
    Position,
)
from dungeon_apprentice.oracle import DungeonOracle, OracleFailure

PROTOCOL = "dungeon-apprentice-v0.2-u1"
GENERATOR_PROFILE_VERSION = 1
QUALIFICATION_SEED_BASE = 5_100_000
QUALIFICATION_SEED_COUNT = 1_000
U1_CONFIRMATION_SEED_BASE = 15_020_000
EVALUATION_PANEL_SIZE = 40
EVALUATION_SEED_COUNT = EVALUATION_PANEL_SIZE * 2
ALLOCATION_TOLERANCE = 0.05
FULL_LAYOUT_UNIQUENESS_FLOOR = 0.99
GEOMETRY_UNIQUENESS_FLOOR = 0.95


class LessonId(StrEnum):
    """Stable identifiers are policy-invisible and trainer-visible only."""

    NAVIGATE = "navigate/full"
    VISIBLE_UNLOCK = "unlock/u0-visible"
    LOCAL_UNLOCK = "unlock/u1-local"


class RecoveryCause(StrEnum):
    NAVIGATE = "navigate"
    VISIBLE_UNLOCK = "visible_unlock"
    BOTH = "navigate_and_visible_unlock"


@dataclass(frozen=True)
class LessonSpec:
    lesson_id: LessonId
    label: str
    tier: DungeonTier
    prerequisites: tuple[LessonId, ...]
    generator_profile: str
    max_steps: int
    validation_seed_base: int
    overall_threshold: float
    panel_threshold: float
    oracle_action_range: tuple[int, int] | None
    key_visible: bool | None = None
    door_visible: bool | None = None


LESSON_SPECS: Mapping[LessonId, LessonSpec] = MappingProxyType(
    {
        LessonId.NAVIGATE: LessonSpec(
            lesson_id=LessonId.NAVIGATE,
            label="Navigate",
            tier=DungeonTier.NAVIGATE,
            prerequisites=(),
            generator_profile="v0.1-navigate",
            max_steps=128,
            validation_seed_base=10_000_000,
            overall_threshold=0.85,
            panel_threshold=0.85,
            oracle_action_range=None,
        ),
        LessonId.VISIBLE_UNLOCK: LessonSpec(
            lesson_id=LessonId.VISIBLE_UNLOCK,
            label="Visible Unlock",
            tier=DungeonTier.UNLOCK,
            prerequisites=(LessonId.NAVIGATE,),
            generator_profile="v0.2-u0-visible-frozen",
            max_steps=128,
            validation_seed_base=11_000_000,
            overall_threshold=0.85,
            panel_threshold=0.80,
            oracle_action_range=(5, 10),
            key_visible=True,
            door_visible=True,
        ),
        LessonId.LOCAL_UNLOCK: LessonSpec(
            lesson_id=LessonId.LOCAL_UNLOCK,
            label="Local Unlock",
            tier=DungeonTier.UNLOCK,
            prerequisites=(LessonId.NAVIGATE, LessonId.VISIBLE_UNLOCK),
            generator_profile="v0.2-u1-local-v1",
            max_steps=128,
            validation_seed_base=11_100_000,
            overall_threshold=0.85,
            panel_threshold=0.80,
            oracle_action_range=(9, 18),
            key_visible=True,
            door_visible=False,
        ),
    }
)


@dataclass(frozen=True)
class UnlockPlan:
    actions: tuple[int, ...]
    pickup_index: int
    toggle_index: int
    post_key_turns: int


def _advance_pose(
    state: tuple[int, int, int], actions: Iterable[int]
) -> tuple[int, int, int]:
    x, y, direction = state
    for action in actions:
        if action == 0:
            direction = (direction - 1) % 4
        elif action == 1:
            direction = (direction + 1) % 4
        elif action == 2:
            dx, dy = (int(value) for value in DIR_TO_VEC[direction])
            x += dx
            y += dy
    return x, y, direction


def _shortest_pose_plan(
    *,
    start: tuple[int, int, int],
    width: int,
    height: int,
    blocked: set[Position],
    target: Position,
    face_target: bool,
) -> tuple[int, ...]:
    """Return a deterministic shortest primitive-action navigation plan."""

    def complete(state: tuple[int, int, int]) -> bool:
        x, y, direction = state
        if not face_target:
            return (x, y) == target
        dx, dy = (int(value) for value in DIR_TO_VEC[direction])
        return (x + dx, y + dy) == target

    queue = deque([start])
    previous: dict[
        tuple[int, int, int], tuple[tuple[int, int, int], int] | None
    ] = {start: None}
    final: tuple[int, int, int] | None = None
    while queue:
        state = queue.popleft()
        if complete(state):
            final = state
            break
        x, y, direction = state
        candidates = (
            ((x, y, (direction - 1) % 4), 0),
            ((x, y, (direction + 1) % 4), 1),
        )
        for next_state, action in candidates:
            if next_state not in previous:
                previous[next_state] = (state, action)
                queue.append(next_state)
        dx, dy = (int(value) for value in DIR_TO_VEC[direction])
        forward = (x + dx, y + dy)
        if (
            0 < forward[0] < width - 1
            and 0 < forward[1] < height - 1
            and forward not in blocked
        ):
            next_state = (*forward, direction)
            if next_state not in previous:
                previous[next_state] = (state, 2)
                queue.append(next_state)

    if final is None:
        raise OracleFailure("pure planner could not reach the requested pose")
    actions: list[int] = []
    cursor = final
    while previous[cursor] is not None:
        prior, action = previous[cursor]
        actions.append(action)
        cursor = prior
    actions.reverse()
    return tuple(actions)


def plan_unlock_candidate(
    layout: Mapping[str, Any], *, width: int = 9, height: int = 9
) -> UnlockPlan:
    """Solve immutable candidate geometry without mutating a live environment."""

    start = (*layout["start"], int(layout["start_direction"]))
    key = tuple(layout["key"])
    door = tuple(layout["door"])
    goal = tuple(layout["goal"])
    walls = set(layout["walls"])

    to_key = _shortest_pose_plan(
        start=start,
        width=width,
        height=height,
        blocked=walls | {key, door},
        target=key,
        face_target=True,
    )
    after_key = _advance_pose(start, to_key)
    to_door = _shortest_pose_plan(
        start=after_key,
        width=width,
        height=height,
        blocked=walls | {door},
        target=door,
        face_target=True,
    )
    after_door = _advance_pose(after_key, to_door)
    to_goal = _shortest_pose_plan(
        start=after_door,
        width=width,
        height=height,
        blocked=walls,
        target=goal,
        face_target=False,
    )
    actions = (*to_key, 3, *to_door, 5, *to_goal)
    pickup_index = len(to_key)
    toggle_index = len(to_key) + 1 + len(to_door)
    return UnlockPlan(
        actions=tuple(actions),
        pickup_index=pickup_index,
        toggle_index=toggle_index,
        post_key_turns=sum(action in {0, 1} for action in to_door),
    )


def geometry_sha256(layout: Mapping[str, Any], *, size: int) -> str:
    """Hash geometry and pose while deliberately excluding protocol and key color."""

    payload = {
        "size": int(size),
        "start": list(layout["start"]),
        "start_direction": int(layout["start_direction"]),
        "goal": list(layout["goal"]),
        "key": list(layout["key"]) if layout.get("key") is not None else None,
        "door": list(layout["door"]) if layout.get("door") is not None else None,
        "walls": [list(position) for position in sorted(layout["walls"])],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _visible_unlock_layout(env: DungeonApprenticeEnv, width: int, height: int) -> dict[str, Any]:
    """The confirmed U0 profile, copied byte-for-byte in behavior from its frozen source."""

    vertical = bool(env.np_random.integers(0, 2))
    split = int(env.np_random.integers(3, 6))
    lane = int(env.np_random.integers(1, 8))
    approach_from_low = bool(env.np_random.integers(0, 2))

    if approach_from_low:
        start_axis = int(env.np_random.integers(1, split - 1))
        key_axis = int(env.np_random.integers(start_axis + 1, split))
        goal_axis = int(env.np_random.integers(split + 1, 8))
        direction = 0 if vertical else 1
    else:
        start_axis = int(env.np_random.integers(split + 2, 8))
        key_axis = int(env.np_random.integers(split + 1, start_axis))
        goal_axis = int(env.np_random.integers(1, split))
        direction = 2 if vertical else 3

    if vertical:
        start = (start_axis, lane)
        key = (key_axis, lane)
        door = (split, lane)
        goal = (goal_axis, lane)
        barrier = {(split, y) for y in range(1, height - 1) if y != lane}
    else:
        start = (lane, start_axis)
        key = (lane, key_axis)
        door = (lane, split)
        goal = (lane, goal_axis)
        barrier = {(x, split) for x in range(1, width - 1) if x != lane}

    return {
        "start": start,
        "start_direction": direction,
        "goal": goal,
        "key": key,
        "door": door,
        "relic": None,
        "key_color": env._choice(list(KEY_COLORS)),
        "walls": barrier,
    }


def _sample_local_unlock_candidate(
    env: DungeonApprenticeEnv, width: int, height: int
) -> dict[str, Any] | None:
    vertical = bool(env.np_random.integers(0, 2))
    split = int(env.np_random.integers(3, 6))
    door_lane = int(env.np_random.integers(1, 8))
    approach_from_low = bool(env.np_random.integers(0, 2))

    if vertical:
        barrier = {(split, y) for y in range(1, height - 1) if y != door_lane}
        door = (split, door_lane)
        approach = [
            (x, y)
            for x in (
                range(1, split) if approach_from_low else range(split + 1, width - 1)
            )
            for y in range(1, height - 1)
        ]
        far = [
            (x, y)
            for x in (
                range(split + 1, width - 1)
                if approach_from_low
                else range(1, split)
            )
            for y in range(1, height - 1)
        ]
    else:
        barrier = {(x, split) for x in range(1, width - 1) if x != door_lane}
        door = (door_lane, split)
        approach = [
            (x, y)
            for x in range(1, width - 1)
            for y in (
                range(1, split) if approach_from_low else range(split + 1, height - 1)
            )
        ]
        far = [
            (x, y)
            for x in range(1, width - 1)
            for y in (
                range(split + 1, height - 1)
                if approach_from_low
                else range(1, split)
            )
        ]

    key_candidates = [
        position
        for position in approach
        if position[0] != door[0]
        and position[1] != door[1]
        and DungeonApprenticeEnv._manhattan(position, door) >= 4
    ]
    if not key_candidates:
        return None
    key = env._choice(key_candidates)
    headings = list(range(4))
    # Randomize the iteration itself so orientation is not coupled to rejection order.
    headings = [int(value) for value in env.np_random.permutation(headings)]
    distances = [int(value) for value in env.np_random.permutation([1, 2, 3])]
    starts: list[tuple[Position, int]] = []
    approach_set = set(approach)
    for direction in headings:
        dx, dy = (int(value) for value in DIR_TO_VEC[direction])
        for distance in distances:
            start = (key[0] - dx * distance, key[1] - dy * distance)
            sight_line = {
                (start[0] + dx * step, start[1] + dy * step)
                for step in range(distance + 1)
            }
            if start in approach_set and sight_line <= approach_set:
                starts.append((start, direction))
    if not starts:
        return None
    start, direction = starts[int(env.np_random.integers(0, len(starts)))]
    goal = env._choice(far)
    candidate = {
        "start": start,
        "start_direction": direction,
        "goal": goal,
        "key": key,
        "door": door,
        "relic": None,
        "key_color": env._choice(list(KEY_COLORS)),
        "walls": barrier,
    }
    try:
        plan = plan_unlock_candidate(candidate, width=width, height=height)
    except OracleFailure:
        return None
    if not 9 <= len(plan.actions) <= 18:
        return None
    if plan.post_key_turns < 1:
        return None
    return candidate


class LessonEnv(DungeonApprenticeEnv):
    """Pixel-identical legacy lessons plus the qualified Local Unlock profile."""

    def __init__(
        self,
        *,
        lesson: LessonId = LessonId.LOCAL_UNLOCK,
        size: int = 9,
        render_mode: str | None = None,
    ) -> None:
        self.lesson = LessonId(lesson)
        self.geometry_sha256: str | None = None
        self.pure_oracle_actions: tuple[int, ...] | None = None
        spec = LESSON_SPECS[self.lesson]
        super().__init__(tier=spec.tier, size=size, render_mode=render_mode)
        self.max_steps = spec.max_steps

    def set_lesson(self, lesson: LessonId | str) -> None:
        self.lesson = LessonId(lesson)
        spec = LESSON_SPECS[self.lesson]
        self.tier = spec.tier
        self.max_steps = spec.max_steps

    def set_tier(self, tier: DungeonTier | int) -> None:
        selected = DungeonTier(tier)
        if selected is DungeonTier.NAVIGATE:
            self.set_lesson(LessonId.NAVIGATE)
        elif selected is DungeonTier.UNLOCK:
            self.set_lesson(LessonId.LOCAL_UNLOCK)
        else:
            raise ValueError("the U1 child contains only Navigate and Unlock lessons")

    def _generate_quest(self, width: int, height: int) -> dict[str, Any]:
        if self.lesson is LessonId.VISIBLE_UNLOCK:
            return _visible_unlock_layout(self, width, height)
        return super()._generate_quest(width, height)

    def _install_local_candidate(
        self, candidate: Mapping[str, Any], *, width: int, height: int
    ) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        for position in candidate["walls"]:
            self.grid.set(*position, Wall())
        color = str(candidate["key_color"])
        self.grid.set(*candidate["key"], Key(color))
        self.grid.set(*candidate["door"], Door(color, is_locked=True))
        self.grid.set(*candidate["goal"], Goal())
        self.agent_pos = candidate["start"]
        self.agent_dir = int(candidate["start_direction"])
        self.mission = self.tier.objective

    def _record_layout(self, candidate: Mapping[str, Any]) -> None:
        digest = hashlib.sha256()
        digest.update(self.grid.encode().tobytes())
        digest.update(bytes((*self.agent_pos, self.agent_dir, int(self.tier))))
        digest.update(PROTOCOL.encode())
        digest.update(str(GENERATOR_PROFILE_VERSION).encode())
        digest.update(self.lesson.value.encode())
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
            key=tuple(candidate["key"]) if candidate.get("key") is not None else None,
            door=tuple(candidate["door"]) if candidate.get("door") is not None else None,
            relic=None,
            key_color=(
                str(candidate["key_color"]) if candidate.get("key_color") is not None else None
            ),
            walls=tuple(sorted(candidate["walls"])),
            layout_sha256=digest.hexdigest(),
        )

    def _gen_grid(self, width: int, height: int) -> None:
        if width != 9 or height != 9:
            raise ValueError("the U1 child is qualified only at 9 x 9")
        self.pure_oracle_actions = None
        if self.lesson is not LessonId.LOCAL_UNLOCK:
            super()._gen_grid(width, height)
            assert self.layout is not None
            candidate = {
                "start": self.layout.start,
                "start_direction": self.layout.start_direction,
                "goal": self.layout.goal,
                "key": self.layout.key,
                "door": self.layout.door,
                "walls": self.layout.walls,
            }
            self.geometry_sha256 = geometry_sha256(candidate, size=self._size)
            digest = hashlib.sha256()
            digest.update(self.layout.layout_sha256.encode())
            digest.update(PROTOCOL.encode())
            digest.update(str(GENERATOR_PROFILE_VERSION).encode())
            digest.update(self.lesson.value.encode())
            self.layout = replace(
                self.layout,
                protocol=PROTOCOL,
                layout_sha256=digest.hexdigest(),
            )
            return

        for _ in range(2_048):
            candidate = _sample_local_unlock_candidate(self, width, height)
            if candidate is None:
                continue
            self._install_local_candidate(candidate, width=width, height=height)
            if not self.agent_sees(*candidate["key"]):
                continue
            if self.agent_sees(*candidate["door"]):
                continue
            plan = plan_unlock_candidate(candidate, width=width, height=height)
            self.pure_oracle_actions = plan.actions
            self._record_layout(candidate)
            return
        raise RuntimeError("failed to generate a qualified Local Unlock dungeon")

    def _evidence_info(self, *, success: bool) -> dict[str, Any]:
        info = super()._evidence_info(success=success)
        info.update(
            {
                "protocol": PROTOCOL,
                "lesson_id": self.lesson.value,
                "lesson_label": LESSON_SPECS[self.lesson].label,
                "generator_profile": LESSON_SPECS[self.lesson].generator_profile,
                "geometry_sha256": self.geometry_sha256,
            }
        )
        return info


def make_pixel_env(
    *,
    lesson: LessonId,
    size: int = 9,
    render_mode: str | None = "rgb_array",
) -> gym.Env:
    env: gym.Env = LessonEnv(lesson=lesson, size=size, render_mode=render_mode)
    env = RGBImgPartialObsWrapper(env, tile_size=PIXEL_TILE_SIZE)
    return ImgObsWrapper(env)


@dataclass
class CurriculumState:
    """Cumulative U1 state with explicit prerequisite-recovery ownership."""

    active_lesson: LessonId = LessonId.LOCAL_UNLOCK
    passed_lessons: tuple[LessonId, ...] = (
        LessonId.NAVIGATE,
        LessonId.VISIBLE_UNLOCK,
    )
    consecutive_passes: int = 0
    recovery_cause: RecoveryCause | None = None
    recovery_passes: int = 0
    revision: int = 0
    mastered: bool = False

    def targets(self) -> dict[LessonId, float]:
        if self.recovery_cause is RecoveryCause.NAVIGATE:
            return {
                LessonId.NAVIGATE: 0.75,
                LessonId.VISIBLE_UNLOCK: 0.10,
                LessonId.LOCAL_UNLOCK: 0.15,
            }
        if self.recovery_cause is RecoveryCause.VISIBLE_UNLOCK:
            return {
                LessonId.NAVIGATE: 0.50,
                LessonId.VISIBLE_UNLOCK: 0.35,
                LessonId.LOCAL_UNLOCK: 0.15,
            }
        if self.recovery_cause is RecoveryCause.BOTH:
            return {
                LessonId.NAVIGATE: 0.65,
                LessonId.VISIBLE_UNLOCK: 0.25,
                LessonId.LOCAL_UNLOCK: 0.10,
            }
        return {
            LessonId.NAVIGATE: 0.50,
            LessonId.VISIBLE_UNLOCK: 0.15,
            LessonId.LOCAL_UNLOCK: 0.35,
        }

    def change(self) -> None:
        self.revision += 1

    def public_dict(self) -> dict[str, Any]:
        return {
            "active_lesson": self.active_lesson.value,
            "active_lesson_label": LESSON_SPECS[self.active_lesson].label,
            "passed_lessons": [lesson.value for lesson in self.passed_lessons],
            "consecutive_passes": self.consecutive_passes,
            "recovery": self.recovery_cause is not None,
            "recovery_cause": self.recovery_cause.value if self.recovery_cause else None,
            "recovery_passes": self.recovery_passes,
            "revision": self.revision,
            "mastered": self.mastered,
            "targets": {lesson.value: share for lesson, share in self.targets().items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CurriculumState:
        state = cls(
            active_lesson=LessonId(str(value["active_lesson"])),
            passed_lessons=tuple(
                LessonId(str(lesson)) for lesson in value.get("passed_lessons", ())
            ),
            consecutive_passes=int(value.get("consecutive_passes", 0)),
            recovery_cause=(
                RecoveryCause(str(value["recovery_cause"]))
                if value.get("recovery_cause")
                else None
            ),
            recovery_passes=int(value.get("recovery_passes", 0)),
            revision=int(value.get("revision", 0)),
            mastered=bool(value.get("mastered", False)),
        )
        if state.active_lesson is not LessonId.LOCAL_UNLOCK:
            raise ValueError("U1 sidecar has the wrong active lesson")
        if set(state.passed_lessons) != {LessonId.NAVIGATE, LessonId.VISIBLE_UNLOCK}:
            raise ValueError("U1 sidecar does not preserve both prerequisites")
        return state


class TransitionDeficitScheduler:
    """Balance complete episodes against N-lesson transition targets."""

    def __init__(self, state: CurriculumState, *, seed: int) -> None:
        self.state = state
        self._rng = np.random.default_rng(seed)
        self._revision = state.revision
        self.window_transitions: dict[LessonId, int] = defaultdict(int)
        self.lifetime_transitions: dict[LessonId, int] = defaultdict(int)
        self._reservations: dict[LessonId, int] = defaultdict(int)

    def _sync(self) -> None:
        if self._revision == self.state.revision:
            return
        self._revision = self.state.revision
        self.window_transitions = defaultdict(int)
        self._reservations = defaultdict(int)

    def assign(self) -> tuple[LessonId, int]:
        self._sync()
        candidates: list[tuple[float, float, LessonId]] = []
        for lesson, target in self.state.targets().items():
            effective = self.window_transitions[lesson] + self._reservations[lesson]
            candidates.append((effective / target, float(self._rng.random()), lesson))
        _, _, lesson = min(candidates)
        reservation = LESSON_SPECS[lesson].max_steps
        self._reservations[lesson] += reservation
        return lesson, reservation

    def record_step(self, lesson: LessonId) -> None:
        self._sync()
        self.window_transitions[lesson] += 1
        self.lifetime_transitions[lesson] += 1

    def release(self, lesson: LessonId, reservation: int) -> None:
        self._reservations[lesson] = max(
            0, self._reservations[lesson] - max(0, int(reservation))
        )

    def snapshot(self) -> dict[str, Any]:
        self._sync()
        targets = self.state.targets()
        total = sum(self.window_transitions.values())
        return {
            "revision": self._revision,
            "target_shares": {lesson.value: share for lesson, share in targets.items()},
            "window_transitions": {
                lesson.value: int(self.window_transitions.get(lesson, 0))
                for lesson in targets
            },
            "realized_shares": {
                lesson.value: (
                    self.window_transitions.get(lesson, 0) / total if total else 0.0
                )
                for lesson in targets
            },
            "lifetime_transitions": {
                lesson.value: int(self.lifetime_transitions.get(lesson, 0))
                for lesson in LessonId
            },
        }

    def state_dict(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        return snapshot | {"rng_state": self._rng.bit_generator.state}

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if int(value.get("revision", -1)) != self.state.revision:
            raise ValueError("scheduler and curriculum revisions differ")
        self._revision = self.state.revision
        self.window_transitions = defaultdict(
            int,
            {
                LessonId(str(lesson)): int(count)
                for lesson, count in value.get("window_transitions", {}).items()
            },
        )
        self.lifetime_transitions = defaultdict(
            int,
            {
                LessonId(str(lesson)): int(count)
                for lesson, count in value.get("lifetime_transitions", {}).items()
            },
        )
        self._reservations = defaultdict(int)
        self._rng.bit_generator.state = dict(value["rng_state"])
        expected = {lesson.value: share for lesson, share in self.state.targets().items()}
        if value.get("target_shares") != expected:
            raise ValueError("scheduler target shares do not match curriculum state")

    def start_window(self) -> None:
        self.window_transitions = defaultdict(int)
        self._reservations = defaultdict(int)

    def allocation_within(self, tolerance: float = ALLOCATION_TOLERANCE) -> bool:
        snapshot = self.snapshot()
        if not sum(snapshot["window_transitions"].values()):
            return True
        return all(
            abs(snapshot["realized_shares"][lesson] - target) <= tolerance
            for lesson, target in snapshot["target_shares"].items()
        )


class CurriculumEnv(gym.Wrapper):
    def __init__(
        self,
        env: LessonEnv,
        *,
        scheduler: TransitionDeficitScheduler,
        seed: int,
        forbidden_layout_hashes: Mapping[LessonId, frozenset[str]] | None = None,
    ) -> None:
        super().__init__(env)
        self.scheduler = scheduler
        self._rng = np.random.default_rng(seed)
        self._lesson: LessonId | None = None
        self._reservation = 0
        self._forbidden_layout_hashes = forbidden_layout_hashes or {}

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        if self._lesson is not None:
            self.scheduler.release(self._lesson, self._reservation)
        requested_seed = kwargs.pop("seed", None)
        if requested_seed is not None:
            self._rng = np.random.default_rng(requested_seed)
        self._lesson, self._reservation = self.scheduler.assign()
        self.unwrapped.set_lesson(self._lesson)
        forbidden = self._forbidden_layout_hashes.get(self._lesson, frozenset())
        for rejected in range(128):
            episode_seed = int(self._rng.integers(0, TRAINING_SEED_LIMIT))
            observation, info = super().reset(seed=episode_seed, **kwargs)
            if info.get("layout_sha256") not in forbidden:
                info["reserved_layout_rejections"] = rejected
                return observation, info
        raise RuntimeError("could not sample a training layout outside reserved evidence sets")

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        assert self._lesson is not None
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
    forbidden_layout_hashes: Mapping[LessonId, frozenset[str]] | None = None,
) -> gym.Env:
    base = LessonEnv(lesson=LessonId.LOCAL_UNLOCK, size=size, render_mode=render_mode)
    env: gym.Env = CurriculumEnv(
        base,
        scheduler=scheduler,
        seed=seed,
        forbidden_layout_hashes=forbidden_layout_hashes,
    )
    env = RGBImgPartialObsWrapper(env, tile_size=PIXEL_TILE_SIZE)
    env = ImgObsWrapper(env)
    return EpisodicPixelCuriosity(
        env, scale=DEFAULT_CURIOSITY_SCALE, budget=CURIOSITY_BUDGET
    )


@dataclass(frozen=True)
class LessonEvaluation:
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
    milestone_rates: dict[str, float] = field(default_factory=dict)
    mean_collisions: float = 0.0
    mean_ineffective_interactions: float = 0.0

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    from dungeon_apprentice.artifacts import utc_now

    return utc_now()


def _policy_observation(observation: np.ndarray, model: Any) -> np.ndarray:
    expected = tuple(model.observation_space.shape)
    if tuple(observation.shape) == expected:
        return observation
    channels_first = np.transpose(observation, (2, 0, 1))
    if tuple(channels_first.shape) == expected:
        return channels_first
    raise ValueError(f"model expects {expected}, game produced {observation.shape}")


def validation_seeds(
    lesson: LessonId, count: int = EVALUATION_SEED_COUNT
) -> tuple[int, ...]:
    if count != EVALUATION_SEED_COUNT:
        raise ValueError(f"the U1 child freezes exactly {EVALUATION_SEED_COUNT} exam cases")
    base = LESSON_SPECS[LessonId(lesson)].validation_seed_base
    return tuple(base + offset for offset in range(count))


def reserved_training_layout_hashes(
    qualification: Mapping[str, Any],
) -> Mapping[LessonId, frozenset[str]]:
    """Prevent exact validation/qualification images from reappearing in training."""

    reserved: dict[LessonId, set[str]] = {lesson: set() for lesson in LessonId}
    for lesson in LessonId:
        for seed in validation_seeds(lesson):
            env = LessonEnv(lesson=lesson)
            try:
                env.reset(seed=seed)
                assert env.layout is not None
                reserved[lesson].add(env.layout.layout_sha256)
            finally:
                env.close()
    if qualification.get("lesson_id") != LessonId.LOCAL_UNLOCK.value:
        raise ValueError("qualification report is not for Local Unlock")
    for case in qualification.get("cases", []):
        reserved[LessonId.LOCAL_UNLOCK].add(str(case["layout_sha256"]))
    return MappingProxyType(
        {lesson: frozenset(hashes) for lesson, hashes in reserved.items()}
    )


def evaluate_lesson(
    model: Any,
    lesson: LessonId,
    seeds: Iterable[int],
    *,
    size: int = 9,
    frame_path: Path | None = None,
) -> LessonEvaluation:
    outcomes: list[dict[str, Any]] = []
    latest: np.ndarray | None = None
    for seed in seeds:
        env = make_pixel_env(lesson=lesson, size=size)
        try:
            observation, _ = env.reset(seed=int(seed))
            recurrent_state = None
            episode_start = np.ones((1,), dtype=bool)
            steps = 0
            info: dict[str, Any] = {}
            while True:
                action, recurrent_state = model.predict(
                    _policy_observation(observation, model),
                    state=recurrent_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                episode_start[:] = False
                observation, _, terminated, truncated, info = env.step(
                    int(np.asarray(action).item())
                )
                steps += 1
                if terminated or truncated:
                    break
            latest = observation
            outcomes.append(
                {
                    "success": bool(info.get("success", False)),
                    "steps": steps,
                    "milestones": dict(info.get("milestones", {})),
                    "collisions": int(info.get("collisions", 0)),
                    "ineffective": int(info.get("ineffective_interactions", 0)),
                }
            )
        finally:
            env.close()
    if not outcomes:
        raise ValueError("evaluation requires at least one seed")
    if frame_path is not None and latest is not None:
        temporary = frame_path.with_name(f".{frame_path.name}.tmp")
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(latest).save(temporary, format="PNG")
        os.replace(temporary, frame_path)
    midpoint = len(outcomes) // 2
    panels = (outcomes[:midpoint], outcomes[midpoint:])
    panel_successes = tuple(sum(item["success"] for item in panel) for panel in panels)
    milestone_names = sorted(
        {name for result in outcomes for name in result["milestones"]}
    )
    successes = sum(result["success"] for result in outcomes)
    return LessonEvaluation(
        protocol=PROTOCOL,
        timestamp=_utc_now(),
        lesson_id=lesson.value,
        lesson_label=LESSON_SPECS[lesson].label,
        episodes=len(outcomes),
        successes=successes,
        success_rate=successes / len(outcomes),
        panel_successes=panel_successes,
        panel_success_rates=tuple(
            count / len(panel) if panel else 0.0
            for count, panel in zip(panel_successes, panels, strict=True)
        ),
        mean_steps=sum(result["steps"] for result in outcomes) / len(outcomes),
        milestone_rates={
            name: sum(result["milestones"].get(name, False) for result in outcomes)
            / len(outcomes)
            for name in milestone_names
        },
        mean_collisions=sum(result["collisions"] for result in outcomes) / len(outcomes),
        mean_ineffective_interactions=(
            sum(result["ineffective"] for result in outcomes) / len(outcomes)
        ),
    )


def lesson_passed(result: LessonEvaluation) -> bool:
    spec = LESSON_SPECS[LessonId(result.lesson_id)]
    return result.success_rate >= spec.overall_threshold and all(
        rate >= spec.panel_threshold for rate in result.panel_success_rates
    )


def recovery_cause(
    evaluations: Mapping[LessonId, LessonEvaluation]
) -> RecoveryCause | None:
    navigate_failed = not lesson_passed(evaluations[LessonId.NAVIGATE])
    u0_failed = not lesson_passed(evaluations[LessonId.VISIBLE_UNLOCK])
    if navigate_failed and u0_failed:
        return RecoveryCause.BOTH
    if navigate_failed:
        return RecoveryCause.NAVIGATE
    if u0_failed:
        return RecoveryCause.VISIBLE_UNLOCK
    return None


def qualify_local_unlock(
    seed_count: int = QUALIFICATION_SEED_COUNT,
    *,
    seed_base: int = QUALIFICATION_SEED_BASE,
    size: int = 9,
) -> dict[str, Any]:
    if seed_count <= 0:
        raise ValueError("qualification seed count must be positive")
    if size != 9:
        raise ValueError("Local Unlock is qualified only at size 9")
    failures: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    full_hashes: list[str] = []
    geometry_hashes: list[str] = []
    actions: list[int] = []
    for seed in range(seed_base, seed_base + seed_count):
        wrapped = make_pixel_env(lesson=LessonId.LOCAL_UNLOCK, size=size)
        env = wrapped.unwrapped
        assert isinstance(env, LessonEnv)
        try:
            observation, info = wrapped.reset(seed=seed)
            assert env.layout is not None
            key_visible = env.agent_sees(*env.layout.key)
            door_visible = env.agent_sees(*env.layout.door)
            if not key_visible or door_visible:
                raise OracleFailure("U1 visibility contract failed")
            result = DungeonOracle(env).solve()
            if not 9 <= result.steps <= 18:
                raise OracleFailure(f"U1 solution length {result.steps} outside 9..18")
            if env.pure_oracle_actions is None or len(env.pure_oracle_actions) != result.steps:
                raise OracleFailure("pure and live oracle action counts differ")
            if observation.shape != PIXEL_SHAPE or observation.dtype != np.uint8:
                raise OracleFailure("policy observation contract changed")
            expected_reward = STEP_REWARD * (result.steps - 1) + SUCCESS_REWARD
            if not math.isclose(result.total_reward, expected_reward, abs_tol=1e-9):
                raise OracleFailure("U1 reward contract changed")
            if info.get("lesson_id") != LessonId.LOCAL_UNLOCK.value:
                raise OracleFailure("trainer evidence has the wrong lesson ID")
            full_hashes.append(result.layout_sha256)
            assert env.geometry_sha256 is not None
            geometry_hashes.append(env.geometry_sha256)
            actions.append(result.steps)
            cases.append(
                {
                    "seed": seed,
                    "layout_sha256": result.layout_sha256,
                    "geometry_sha256": env.geometry_sha256,
                    "oracle_actions": result.steps,
                }
            )
        except (AssertionError, OracleFailure, RuntimeError, ValueError) as error:
            failures.append({"seed": seed, "error": str(error)})
        finally:
            wrapped.close()

    full_unique = len(set(full_hashes))
    geometry_unique = len(set(geometry_hashes))
    full_floor = math.ceil(seed_count * FULL_LAYOUT_UNIQUENESS_FLOOR)
    if full_unique < full_floor:
        failures.append(
            {
                "seed": None,
                "error": (
                    f"only {full_unique}/{seed_count} full layouts were unique; "
                    f"requires at least {full_floor}"
                ),
            }
        )
    geometry_floor = math.ceil(seed_count * GEOMETRY_UNIQUENESS_FLOOR)
    if geometry_unique < geometry_floor:
        failures.append(
            {
                "seed": None,
                "error": (
                    f"only {geometry_unique}/{seed_count} geometries were unique; "
                    f"requires at least {geometry_floor}"
                ),
            }
        )
    validation_full_hashes: set[str] = set()
    validation_geometry_hashes: set[str] = set()
    for seed in validation_seeds(LessonId.LOCAL_UNLOCK):
        env = LessonEnv(lesson=LessonId.LOCAL_UNLOCK, size=size)
        try:
            env.reset(seed=seed)
            assert env.layout is not None
            assert env.geometry_sha256 is not None
            validation_full_hashes.add(env.layout.layout_sha256)
            validation_geometry_hashes.add(env.geometry_sha256)
        finally:
            env.close()
    full_overlap = sorted(set(full_hashes) & validation_full_hashes)
    geometry_overlap = sorted(set(geometry_hashes) & validation_geometry_hashes)
    if full_overlap:
        failures.append(
            {
                "seed": None,
                "error": (
                    f"qualification and validation share {len(full_overlap)} exact visual layouts"
                ),
            }
        )
    return {
        "protocol": PROTOCOL,
        "generator_profile_version": GENERATOR_PROFILE_VERSION,
        "lesson_id": LessonId.LOCAL_UNLOCK.value,
        "seed_partition": {
            "start": seed_base,
            "end": seed_base + seed_count - 1,
        },
        "requested": seed_count,
        "solved": len(actions),
        "full_unique_layouts": full_unique,
        "full_layout_uniqueness_floor": FULL_LAYOUT_UNIQUENESS_FLOOR,
        "geometry_unique_layouts": geometry_unique,
        "geometry_uniqueness_floor": GEOMETRY_UNIQUENESS_FLOOR,
        "minimum_oracle_actions": min(actions) if actions else None,
        "maximum_oracle_actions": max(actions) if actions else None,
        "visibility": {"key_visible": True, "door_visible": False},
        "validation_disjointness": {
            "validation_cases": EVALUATION_SEED_COUNT,
            "validation_unique_full_layouts": len(validation_full_hashes),
            "validation_unique_geometries": len(validation_geometry_hashes),
            "exact_visual_overlap": len(full_overlap),
            "geometry_overlap_diagnostic": len(geometry_overlap),
        },
        "cases": cases,
        "failures": failures,
        "result": "passed" if not failures else "failed",
    }
