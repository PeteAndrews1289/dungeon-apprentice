"""A full-state solver used only to qualify generated levels."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass

from minigrid.core.constants import DIR_TO_VEC
from minigrid.core.world_object import Ball, Door, Key, Wall

from dungeon_apprentice.contracts import DungeonTier
from dungeon_apprentice.env import DungeonApprenticeEnv, Position


class OracleFailure(RuntimeError):
    """Raised when a generated dungeon has no mechanically valid solution."""


@dataclass(frozen=True)
class OracleResult:
    tier: int
    seed: int
    success: bool
    actions: tuple[int, ...]
    steps: int
    total_reward: float
    layout_sha256: str
    terminal_reason: str | None

    def public_dict(self) -> dict[str, object]:
        return asdict(self)


class DungeonOracle:
    """Pathfind with privileged state without exposing a demonstration to learning."""

    def __init__(self, env: DungeonApprenticeEnv) -> None:
        if env.layout is None:
            raise OracleFailure("environment must be reset before oracle qualification")
        self.env = env
        self.actions: list[int] = []
        self.total_reward = 0.0
        self.terminal_reason: str | None = None
        self.success = False

    def solve(self) -> OracleResult:
        layout = self.env.layout
        assert layout is not None

        if self.env.tier is DungeonTier.NAVIGATE:
            self._move_to(layout.goal)
        else:
            assert layout.key is not None and layout.door is not None
            self._face(layout.key)
            self._act(int(self.env.actions.pickup))
            if not isinstance(self.env.carrying, Key):
                raise OracleFailure("oracle could not pick up the key")

            self._face(layout.door)
            self._act(int(self.env.actions.toggle))
            door = self.env.grid.get(*layout.door)
            if not isinstance(door, Door) or not door.is_open:
                raise OracleFailure("oracle could not unlock the door")
            if self.env.carrying is not None:
                raise OracleFailure("single-use key was not consumed")

            if self.env.tier is DungeonTier.UNLOCK:
                self._move_to(layout.goal)
            else:
                assert layout.relic is not None
                self._face(layout.relic)
                self._act(int(self.env.actions.pickup))
                if not isinstance(self.env.carrying, Ball):
                    raise OracleFailure("oracle could not pick up the relic")
                self._move_to(layout.goal)

        if not self.success:
            raise OracleFailure("oracle action sequence ended without success")
        return OracleResult(
            tier=int(self.env.tier),
            seed=int(layout.seed if layout.seed is not None else -1),
            success=True,
            actions=tuple(self.actions),
            steps=len(self.actions),
            total_reward=self.total_reward,
            layout_sha256=layout.layout_sha256,
            terminal_reason=self.terminal_reason,
        )

    def _move_to(self, target: Position) -> None:
        plan = self._plan(lambda state: state[:2] == target)
        for action in plan:
            self._act(action)

    def _face(self, target: Position) -> None:
        def facing(state: tuple[int, int, int]) -> bool:
            x, y, direction = state
            dx, dy = (int(value) for value in DIR_TO_VEC[direction])
            return (x + dx, y + dy) == target

        plan = self._plan(facing)
        for action in plan:
            self._act(action)

    def _plan(
        self,
        goal: Callable[[tuple[int, int, int]], bool],
    ) -> tuple[int, ...]:
        start = (*tuple(int(value) for value in self.env.agent_pos), int(self.env.agent_dir))
        queue = deque([start])
        previous: dict[tuple[int, int, int], tuple[tuple[int, int, int], int] | None] = {
            start: None
        }

        final: tuple[int, int, int] | None = None
        while queue:
            state = queue.popleft()
            if goal(state):
                final = state
                break
            for next_state, action in self._successors(state):
                if next_state not in previous:
                    previous[next_state] = (state, action)
                    queue.append(next_state)

        if final is None:
            raise OracleFailure("no legal navigation plan reaches the requested state")

        actions: list[int] = []
        cursor = final
        while previous[cursor] is not None:
            prior, action = previous[cursor]
            actions.append(action)
            cursor = prior
        actions.reverse()
        return tuple(actions)

    def _successors(
        self, state: tuple[int, int, int]
    ) -> tuple[tuple[tuple[int, int, int], int], ...]:
        x, y, direction = state
        left = (x, y, (direction - 1) % 4)
        right = (x, y, (direction + 1) % 4)
        successors = [
            (left, int(self.env.actions.left)),
            (right, int(self.env.actions.right)),
        ]
        dx, dy = (int(value) for value in DIR_TO_VEC[direction])
        forward = (x + dx, y + dy)
        if self._can_enter(forward):
            successors.append(((*forward, direction), int(self.env.actions.forward)))
        return tuple(successors)

    def _can_enter(self, position: Position) -> bool:
        cell = self.env.grid.get(*position)
        if cell is None:
            return True
        if isinstance(cell, Wall | Key | Ball):
            return False
        if isinstance(cell, Door):
            return bool(cell.is_open)
        return bool(cell.can_overlap())

    def _act(self, action: int) -> None:
        if self.success:
            raise OracleFailure("oracle attempted an action after terminal success")
        _, reward, terminated, _truncated, info = self.env.step(action)
        self.actions.append(int(action))
        self.total_reward += float(reward)
        self.terminal_reason = info.get("terminal_reason")
        self.success = bool(info.get("success", False))
        if self.terminal_reason == "time_limit" and not self.success:
            raise OracleFailure("oracle exceeded the level action limit")
        if terminated and not self.success:
            raise OracleFailure("oracle reached an unexpected terminal state")


def solve_seed(tier: DungeonTier | int, seed: int, *, size: int = 9) -> OracleResult:
    """Generate and solve one level without returning observations or teaching data."""

    env = DungeonApprenticeEnv(tier=tier, size=size)
    try:
        env.reset(seed=seed)
        return DungeonOracle(env).solve()
    finally:
        env.close()
