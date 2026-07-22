"""Procedural Dungeon Apprentice environments."""

from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

from dungeon_apprentice.contracts import (
    DEFAULT_CURIOSITY_SCALE,
    OBSERVATION_SIZE,
    PIXEL_TILE_SIZE,
    PROTOCOL,
    STEP_REWARD,
    SUCCESS_REWARD,
    TIER_RULES,
    TRAINING_SEED_LIMIT,
    DungeonTier,
)

Position = tuple[int, int]
KEY_COLORS = ("blue", "red", "yellow", "purple")


@dataclass(frozen=True)
class LayoutMetadata:
    """Trainer-only evidence describing one generated level."""

    protocol: str
    tier: int
    tier_label: str
    seed: int | None
    size: int
    start: Position
    start_direction: int
    goal: Position
    key: Position | None
    door: Position | None
    relic: Position | None
    key_color: str | None
    walls: tuple[Position, ...]
    layout_sha256: str

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


class DungeonApprenticeEnv(MiniGridEnv):
    """Three-tier procedural quest environment with a stable action/observation contract."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 8,
    }

    def __init__(
        self,
        *,
        tier: DungeonTier | int = DungeonTier.NAVIGATE,
        size: int = 9,
        render_mode: str | None = None,
    ) -> None:
        if size < 7:
            raise ValueError("dungeon size must be at least 7")
        self.tier = DungeonTier(tier)
        self._size = int(size)
        self.layout: LayoutMetadata | None = None
        self._reset_seed: int | None = None
        mission_space = MissionSpace(mission_func=lambda: "complete the dungeon quest")
        super().__init__(
            mission_space=mission_space,
            grid_size=self._size,
            max_steps=TIER_RULES[self.tier].max_steps,
            see_through_walls=False,
            agent_view_size=OBSERVATION_SIZE,
            render_mode=render_mode,
        )

    def set_tier(self, tier: DungeonTier | int) -> None:
        """Select the tier used by the next reset without changing spaces."""

        self.tier = DungeonTier(tier)
        self.max_steps = TIER_RULES[self.tier].max_steps

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._reset_seed = seed
        observation, info = super().reset(seed=seed, options=options)
        info.update(self._evidence_info(success=False))
        return observation, info

    def _gen_grid(self, width: int, height: int) -> None:
        if self.tier is DungeonTier.NAVIGATE:
            layout = self._generate_navigation(width, height)
        else:
            layout = self._generate_quest(width, height)

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        for position in layout["walls"]:
            self.grid.set(*position, Wall())

        if self.tier is not DungeonTier.NAVIGATE:
            color = str(layout["key_color"])
            self.grid.set(*layout["key"], Key(color))
            self.grid.set(*layout["door"], Door(color, is_locked=True))

        if self.tier is DungeonTier.RETRIEVE:
            self.grid.set(*layout["relic"], Ball("yellow"))
            self.grid.set(*layout["goal"], Goal())
        else:
            self.grid.set(*layout["goal"], Goal())

        self.agent_pos = layout["start"]
        self.agent_dir = int(layout["start_direction"])
        self.mission = self.tier.objective

        digest = hashlib.sha256()
        digest.update(self.grid.encode().tobytes())
        digest.update(bytes((*self.agent_pos, self.agent_dir, int(self.tier))))
        self.layout = LayoutMetadata(
            protocol=PROTOCOL,
            tier=int(self.tier),
            tier_label=self.tier.label,
            seed=self._reset_seed,
            size=self._size,
            start=self.agent_pos,
            start_direction=self.agent_dir,
            goal=layout["goal"],
            key=layout.get("key"),
            door=layout.get("door"),
            relic=layout.get("relic"),
            key_color=layout.get("key_color"),
            walls=tuple(sorted(layout["walls"])),
            layout_sha256=digest.hexdigest(),
        )

    def _generate_navigation(self, width: int, height: int) -> dict[str, Any]:
        interior = self._interior_positions(width, height)
        for _ in range(256):
            start = self._choice(interior)
            far = [
                pos
                for pos in interior
                if pos != start and self._manhattan(start, pos) >= max(4, self._size // 2)
            ]
            goal = self._choice(far)
            available = [pos for pos in interior if pos not in {start, goal}]
            walls = set(self._sample(available, TIER_RULES[self.tier].obstacle_count))
            if self._path_exists(start, goal, walls, width, height):
                return {
                    "start": start,
                    "start_direction": int(self.np_random.integers(0, 4)),
                    "goal": goal,
                    "walls": walls,
                }
        raise RuntimeError("failed to generate a solvable navigation dungeon")

    def _generate_quest(self, width: int, height: int) -> dict[str, Any]:
        for _ in range(256):
            split_x = int(self.np_random.integers(3, width - 3))
            door_y = int(self.np_random.integers(1, height - 1))
            door = (split_x, door_y)
            barrier = {(split_x, y) for y in range(1, height - 1) if y != door_y}
            left = [(x, y) for x in range(1, split_x) for y in range(1, height - 1)]
            right = [(x, y) for x in range(split_x + 1, width - 1) for y in range(1, height - 1)]

            start, key = self._sample(left, 2)
            target = self._choice(right)
            goal = start if self.tier is DungeonTier.RETRIEVE else target
            relic = target if self.tier is DungeonTier.RETRIEVE else None
            occupied = {start, key, target, door}
            obstacle_pool = [
                pos for pos in (*left, *right) if pos not in occupied and pos not in barrier
            ]
            extra_walls = set(
                self._sample(obstacle_pool, TIER_RULES[self.tier].obstacle_count)
            )
            walls = barrier | extra_walls
            closed = walls | {door}

            if not self._has_reachable_neighbor(start, key, closed, width, height):
                continue
            if not self._path_exists(start, (split_x - 1, door_y), closed, width, height):
                continue
            if self.tier is DungeonTier.UNLOCK:
                if not self._path_exists(
                    (split_x + 1, door_y), target, walls, width, height
                ):
                    continue
            else:
                if not self._has_reachable_neighbor(
                    (split_x + 1, door_y), target, walls, width, height
                ):
                    continue
                if not self._path_exists(
                    (split_x + 1, door_y), start, walls, width, height
                ):
                    continue

            return {
                "start": start,
                "start_direction": int(self.np_random.integers(0, 4)),
                "goal": goal,
                "key": key,
                "door": door,
                "relic": relic,
                "key_color": self._choice(list(KEY_COLORS)),
                "walls": walls,
            }
        raise RuntimeError("failed to generate a solvable quest dungeon")

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        front = tuple(int(value) for value in self.front_pos)
        front_cell = self.grid.get(*front)
        opening_matching_door = (
            int(action) == int(self.actions.toggle)
            and isinstance(front_cell, Door)
            and isinstance(self.carrying, Key)
            and self.carrying.color == front_cell.color
            and not front_cell.is_open
        )

        observation, _reward, terminated, truncated, info = super().step(action)

        if opening_matching_door and isinstance(front_cell, Door) and front_cell.is_open:
            # Dungeon Apprentice keys are single-use, keeping Tier 2's inventory rule explicit.
            self.carrying = None

        on_goal = self.layout is not None and tuple(self.agent_pos) == self.layout.goal
        carrying_relic = isinstance(self.carrying, Ball)
        if self.tier is DungeonTier.RETRIEVE:
            success = bool(terminated and on_goal and carrying_relic)
            if terminated and not success:
                terminated = False
        else:
            success = bool(terminated and on_goal)

        reward = SUCCESS_REWARD if success else STEP_REWARD
        info.update(self._evidence_info(success=success))
        info["terminal_reason"] = (
            "success" if success else "time_limit" if truncated else None
        )
        return observation, reward, terminated, truncated, info

    def _evidence_info(self, *, success: bool) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "tier": int(self.tier),
            "tier_label": self.tier.label,
            "success": bool(success),
            "seed": self._reset_seed,
            "layout_sha256": self.layout.layout_sha256 if self.layout else None,
        }

    @staticmethod
    def _interior_positions(width: int, height: int) -> list[Position]:
        return [(x, y) for x in range(1, width - 1) for y in range(1, height - 1)]

    def _choice(self, values: list[Any]) -> Any:
        if not values:
            raise ValueError("cannot choose from an empty collection")
        return values[int(self.np_random.integers(0, len(values)))]

    def _sample(self, values: list[Any], count: int) -> list[Any]:
        if count > len(values):
            raise ValueError("sample is larger than its source collection")
        indices = self.np_random.choice(len(values), size=count, replace=False)
        return [values[int(index)] for index in np.atleast_1d(indices)]

    @staticmethod
    def _manhattan(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @classmethod
    def _has_reachable_neighbor(
        cls,
        start: Position,
        target: Position,
        walls: set[Position],
        width: int,
        height: int,
    ) -> bool:
        return any(
            cls._path_exists(start, neighbor, walls, width, height)
            for neighbor in cls._neighbors(target)
            if 0 < neighbor[0] < width - 1 and 0 < neighbor[1] < height - 1
        )

    @classmethod
    def _path_exists(
        cls,
        start: Position,
        goal: Position,
        walls: set[Position],
        width: int,
        height: int,
    ) -> bool:
        if start in walls or goal in walls:
            return False
        frontier = deque([start])
        visited = {start}
        while frontier:
            position = frontier.popleft()
            if position == goal:
                return True
            for neighbor in cls._neighbors(position):
                x, y = neighbor
                if (
                    0 < x < width - 1
                    and 0 < y < height - 1
                    and neighbor not in walls
                    and neighbor not in visited
                ):
                    visited.add(neighbor)
                    frontier.append(neighbor)
        return False

    @staticmethod
    def _neighbors(position: Position) -> tuple[Position, ...]:
        x, y = position
        return ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1))


@dataclass
class CurriculumState:
    """Mutable trainer-owned curriculum state shared by local environments."""

    max_tier: DungeonTier = DungeonTier.NAVIGATE


class CurriculumEnv(gym.Wrapper):
    """Sample the newest unlocked tier while retaining earlier-tier practice."""

    def __init__(
        self,
        env: DungeonApprenticeEnv,
        *,
        seed: int,
        state: CurriculumState | None = None,
        newest_tier_fraction: float = 0.70,
    ) -> None:
        super().__init__(env)
        if not 0.0 < newest_tier_fraction <= 1.0:
            raise ValueError("newest tier fraction must be in (0, 1]")
        self.state = state if state is not None else CurriculumState()
        self._rng = np.random.default_rng(seed)
        self._newest_tier_fraction = newest_tier_fraction

    def set_max_tier(self, tier: DungeonTier | int) -> None:
        self.state.max_tier = DungeonTier(tier)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        requested_seed = kwargs.pop("seed", None)
        if requested_seed is not None:
            self._rng = np.random.default_rng(requested_seed)
        max_tier = self.state.max_tier
        if max_tier is DungeonTier.NAVIGATE or self._rng.random() < self._newest_tier_fraction:
            tier = max_tier
        else:
            tier = DungeonTier(int(self._rng.integers(0, int(max_tier))))
        episode_seed = int(self._rng.integers(0, TRAINING_SEED_LIMIT))
        self.unwrapped.set_tier(tier)
        return super().reset(seed=episode_seed, **kwargs)


class EpisodicPixelCuriosity(gym.Wrapper):
    """Reward novel pixel observations without any task or map knowledge."""

    def __init__(self, env: gym.Env, *, scale: float) -> None:
        super().__init__(env)
        if scale < 0.0:
            raise ValueError("curiosity scale cannot be negative")
        self.scale = float(scale)
        self._counts: dict[bytes, int] = {}

    @staticmethod
    def _fingerprint(observation: np.ndarray) -> bytes:
        return hashlib.blake2b(observation.tobytes(), digest_size=16).digest()

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = super().reset(**kwargs)
        self._counts = {self._fingerprint(observation): 1}
        return observation, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, extrinsic_reward, terminated, truncated, info = super().step(action)
        fingerprint = self._fingerprint(observation)
        count = self._counts.get(fingerprint, 0) + 1
        self._counts[fingerprint] = count
        curiosity_reward = self.scale / math.sqrt(count)
        info["extrinsic_reward"] = float(extrinsic_reward)
        info["curiosity_reward"] = curiosity_reward
        return (
            observation,
            float(extrinsic_reward) + curiosity_reward,
            terminated,
            truncated,
            info,
        )


def make_pixel_env(
    *,
    tier: DungeonTier | int = DungeonTier.NAVIGATE,
    size: int = 9,
    render_mode: str | None = "rgb_array",
) -> gym.Env:
    """Create the pixels-only policy view; mission, direction, and coordinates are removed."""

    env = DungeonApprenticeEnv(tier=tier, size=size, render_mode=render_mode)
    env = RGBImgPartialObsWrapper(env, tile_size=PIXEL_TILE_SIZE)
    return ImgObsWrapper(env)


def make_curriculum_pixel_env(
    *,
    seed: int,
    state: CurriculumState,
    size: int = 9,
    curiosity_scale: float = DEFAULT_CURIOSITY_SCALE,
    render_mode: str | None = "rgb_array",
) -> gym.Env:
    """Create a training environment whose policy still receives pixels alone."""

    base = DungeonApprenticeEnv(size=size, render_mode=render_mode)
    env: gym.Env = CurriculumEnv(base, seed=seed, state=state)
    env = RGBImgPartialObsWrapper(env, tile_size=PIXEL_TILE_SIZE)
    env = ImgObsWrapper(env)
    return EpisodicPixelCuriosity(env, scale=curiosity_scale)
