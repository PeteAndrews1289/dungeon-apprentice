"""Protocol-v0.2 development sentinel: Navigate then a visible full Unlock chain."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import time
import traceback
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
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
    CURIOSITY_BUDGET,
    DEFAULT_CURIOSITY_SCALE,
    MINIMUM_GAMMA,
    PIXEL_TILE_SIZE,
    TRAINING_SEED_LIMIT,
    DungeonTier,
)
from dungeon_apprentice.dashboard import start_dashboard
from dungeon_apprentice.env import (
    KEY_COLORS,
    DungeonApprenticeEnv,
    EpisodicPixelCuriosity,
)
from dungeon_apprentice.oracle import DungeonOracle, OracleFailure

PROTOCOL = "dungeon-apprentice-v0.2-sentinel"
CHECKPOINT_SCHEMA_VERSION = 2
PROMOTION_THRESHOLD = 0.90
RETENTION_THRESHOLD = 0.85
STAGE_THRESHOLD = 0.85
PANEL_PROMOTION_THRESHOLD = 0.85
PANEL_STAGE_THRESHOLD = 0.80
EVALUATION_PANEL_SIZE = 40
NAVIGATION_VALIDATION_BASE = 10_000_000
VISIBLE_UNLOCK_VALIDATION_BASE = 11_000_000
POLICY_KWARGS = {"lstm_hidden_size": 256, "n_lstm_layers": 1}


class LessonId(StrEnum):
    NAVIGATE = "navigate/full"
    VISIBLE_UNLOCK = "unlock/u0-visible"

    @property
    def label(self) -> str:
        return {
            self.NAVIGATE: "Navigate",
            self.VISIBLE_UNLOCK: "Visible Unlock",
        }[self]

    @property
    def tier(self) -> DungeonTier:
        return DungeonTier.NAVIGATE if self is LessonId.NAVIGATE else DungeonTier.UNLOCK

    @property
    def max_steps(self) -> int:
        return 128


@dataclass
class SentinelCurriculumState:
    active_lesson: LessonId = LessonId.NAVIGATE
    consecutive_passes: int = 0
    recovery: bool = False
    recovery_passes: int = 0
    revision: int = 0
    mastered: bool = False

    def targets(self) -> dict[LessonId, float]:
        if self.active_lesson is LessonId.NAVIGATE:
            return {LessonId.NAVIGATE: 1.0}
        if self.recovery:
            return {LessonId.NAVIGATE: 0.75, LessonId.VISIBLE_UNLOCK: 0.25}
        return {LessonId.NAVIGATE: 0.50, LessonId.VISIBLE_UNLOCK: 0.50}

    def change(self) -> None:
        self.revision += 1

    def public_dict(self) -> dict[str, Any]:
        return {
            "active_lesson": self.active_lesson.value,
            "active_lesson_label": self.active_lesson.label,
            "consecutive_passes": self.consecutive_passes,
            "recovery": self.recovery,
            "recovery_passes": self.recovery_passes,
            "revision": self.revision,
            "mastered": self.mastered,
            "targets": {lesson.value: value for lesson, value in self.targets().items()},
        }


class TransitionDeficitScheduler:
    """Assign complete episodes while targeting shares of actual transitions."""

    def __init__(self, state: SentinelCurriculumState, *, seed: int) -> None:
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
        targets = self.state.targets()
        candidates: list[tuple[float, float, LessonId]] = []
        for lesson, target in targets.items():
            effective = self.window_transitions[lesson] + self._reservations[lesson]
            normalized = effective / target
            candidates.append((normalized, float(self._rng.random()), lesson))
        _, _, selected = min(candidates)
        reservation = selected.max_steps
        self._reservations[selected] += reservation
        return selected, reservation

    def record_step(self, lesson: LessonId) -> None:
        self._sync()
        self.window_transitions[lesson] += 1
        self.lifetime_transitions[lesson] += 1

    def release(self, lesson: LessonId, reservation: int) -> None:
        self._reservations[lesson] = max(0, self._reservations[lesson] - max(0, int(reservation)))

    def snapshot(self) -> dict[str, Any]:
        self._sync()
        total = sum(self.window_transitions.values())
        targets = self.state.targets()
        return {
            "revision": self._revision,
            "target_shares": {lesson.value: value for lesson, value in targets.items()},
            "window_transitions": {
                lesson.value: int(self.window_transitions.get(lesson, 0)) for lesson in targets
            },
            "realized_shares": {
                lesson.value: (self.window_transitions.get(lesson, 0) / total if total else 0.0)
                for lesson in targets
            },
            "lifetime_transitions": {
                lesson.value: int(self.lifetime_transitions.get(lesson, 0)) for lesson in LessonId
            },
        }

    def start_window(self) -> None:
        self.window_transitions = defaultdict(int)
        self._reservations = defaultdict(int)

    def allocation_within(self, tolerance: float = 0.05) -> bool:
        snapshot = self.snapshot()
        if not sum(snapshot["window_transitions"].values()):
            return True
        return all(
            abs(snapshot["realized_shares"][lesson] - target) <= tolerance
            for lesson, target in snapshot["target_shares"].items()
        )


class V02SentinelEnv(DungeonApprenticeEnv):
    """V0.1 mechanics with a procedurally varied, visible end-to-end Unlock lesson."""

    def __init__(
        self,
        *,
        lesson: LessonId = LessonId.NAVIGATE,
        size: int = 9,
        render_mode: str | None = None,
    ) -> None:
        self.lesson = LessonId(lesson)
        super().__init__(tier=self.lesson.tier, size=size, render_mode=render_mode)
        self.max_steps = self.lesson.max_steps

    def set_lesson(self, lesson: LessonId | str) -> None:
        self.lesson = LessonId(lesson)
        self.tier = self.lesson.tier
        self.max_steps = self.lesson.max_steps

    def set_tier(self, tier: DungeonTier | int) -> None:
        selected = DungeonTier(tier)
        if selected is DungeonTier.NAVIGATE:
            self.set_lesson(LessonId.NAVIGATE)
        elif selected is DungeonTier.UNLOCK:
            self.set_lesson(LessonId.VISIBLE_UNLOCK)
        else:
            raise ValueError("the v0.2 sentinel contains only Navigate and Visible Unlock")

    def _gen_grid(self, width: int, height: int) -> None:
        super()._gen_grid(width, height)
        assert self.layout is not None
        digest = hashlib.sha256()
        digest.update(self.layout.layout_sha256.encode())
        digest.update(self.lesson.value.encode())
        self.layout = replace(
            self.layout,
            protocol=PROTOCOL,
            layout_sha256=digest.hexdigest(),
        )

    def _generate_quest(self, width: int, height: int) -> dict[str, Any]:
        if self.lesson is not LessonId.VISIBLE_UNLOCK:
            return super()._generate_quest(width, height)
        if width != 9 or height != 9:
            raise ValueError("the v0.2 visible lesson is qualified only at 9 x 9")

        vertical = bool(self.np_random.integers(0, 2))
        split = int(self.np_random.integers(3, 6))
        lane = int(self.np_random.integers(1, 8))
        approach_from_low = bool(self.np_random.integers(0, 2))

        if approach_from_low:
            start_axis = int(self.np_random.integers(1, split - 1))
            key_axis = int(self.np_random.integers(start_axis + 1, split))
            goal_axis = int(self.np_random.integers(split + 1, 8))
            direction = 0 if vertical else 1
        else:
            start_axis = int(self.np_random.integers(split + 2, 8))
            key_axis = int(self.np_random.integers(split + 1, start_axis))
            goal_axis = int(self.np_random.integers(1, split))
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
            "key_color": self._choice(list(KEY_COLORS)),
            "walls": barrier,
        }

    def _evidence_info(self, *, success: bool) -> dict[str, Any]:
        info = super()._evidence_info(success=success)
        info.update(
            {
                "protocol": PROTOCOL,
                "lesson_id": self.lesson.value,
                "lesson_label": self.lesson.label,
            }
        )
        return info


class SentinelCurriculumEnv(gym.Wrapper):
    def __init__(
        self,
        env: V02SentinelEnv,
        *,
        scheduler: TransitionDeficitScheduler,
        seed: int,
    ) -> None:
        super().__init__(env)
        self.scheduler = scheduler
        self._rng = np.random.default_rng(seed)
        self._lesson: LessonId | None = None
        self._reservation = 0

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        if self._lesson is not None:
            self.scheduler.release(self._lesson, self._reservation)
        requested_seed = kwargs.pop("seed", None)
        if requested_seed is not None:
            self._rng = np.random.default_rng(requested_seed)
        self._lesson, self._reservation = self.scheduler.assign()
        self.unwrapped.set_lesson(self._lesson)
        episode_seed = int(self._rng.integers(0, TRAINING_SEED_LIMIT))
        return super().reset(seed=episode_seed, **kwargs)

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        assert self._lesson is not None
        self.scheduler.record_step(self._lesson)
        if terminated or truncated:
            self.scheduler.release(self._lesson, self._reservation)
            self._lesson = None
            self._reservation = 0
        return observation, reward, terminated, truncated, info


def make_sentinel_pixel_env(
    *,
    lesson: LessonId,
    size: int = 9,
    render_mode: str | None = "rgb_array",
) -> gym.Env:
    env: gym.Env = V02SentinelEnv(lesson=lesson, size=size, render_mode=render_mode)
    env = RGBImgPartialObsWrapper(env, tile_size=PIXEL_TILE_SIZE)
    return ImgObsWrapper(env)


def make_sentinel_training_env(
    *,
    scheduler: TransitionDeficitScheduler,
    seed: int,
    size: int = 9,
    render_mode: str | None = "rgb_array",
) -> gym.Env:
    base = V02SentinelEnv(size=size, render_mode=render_mode)
    env: gym.Env = SentinelCurriculumEnv(base, scheduler=scheduler, seed=seed)
    env = RGBImgPartialObsWrapper(env, tile_size=PIXEL_TILE_SIZE)
    env = ImgObsWrapper(env)
    return EpisodicPixelCuriosity(env, scale=DEFAULT_CURIOSITY_SCALE, budget=CURIOSITY_BUDGET)


@dataclass(frozen=True)
class LessonEvaluation:
    protocol: str
    timestamp: str
    lesson_id: str
    lesson_label: str
    episodes: int
    successes: int
    success_rate: float
    panel_success_rates: tuple[float, float]
    mean_steps: float
    milestone_rates: dict[str, float] = field(default_factory=dict)
    mean_collisions: float = 0.0
    mean_ineffective_interactions: float = 0.0

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def _policy_observation(observation: np.ndarray, model: Any) -> np.ndarray:
    expected = tuple(model.observation_space.shape)
    if tuple(observation.shape) == expected:
        return observation
    channels_first = np.transpose(observation, (2, 0, 1))
    if tuple(channels_first.shape) == expected:
        return channels_first
    raise ValueError(f"model expects {expected}, game produced {observation.shape}")


def validation_seeds(lesson: LessonId, count: int = 80) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("evaluation seed count must be positive")
    base = (
        NAVIGATION_VALIDATION_BASE
        if lesson is LessonId.NAVIGATE
        else VISIBLE_UNLOCK_VALIDATION_BASE
    )
    return tuple(base + offset for offset in range(count))


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
        env = make_sentinel_pixel_env(lesson=lesson, size=size)
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
    milestone_names = sorted({name for result in outcomes for name in result["milestones"]})
    successes = sum(result["success"] for result in outcomes)
    return LessonEvaluation(
        protocol=PROTOCOL,
        timestamp=utc_now(),
        lesson_id=lesson.value,
        lesson_label=lesson.label,
        episodes=len(outcomes),
        successes=successes,
        success_rate=successes / len(outcomes),
        panel_success_rates=tuple(
            sum(result["success"] for result in panel) / len(panel) if panel else 0.0
            for panel in panels
        ),
        mean_steps=sum(result["steps"] for result in outcomes) / len(outcomes),
        milestone_rates={
            name: sum(result["milestones"].get(name, False) for result in outcomes) / len(outcomes)
            for name in milestone_names
        },
        mean_collisions=sum(result["collisions"] for result in outcomes) / len(outcomes),
        mean_ineffective_interactions=(
            sum(result["ineffective"] for result in outcomes) / len(outcomes)
        ),
    )


def qualify_sentinel(seed_count: int, *, size: int = 9) -> dict[str, Any]:
    lessons: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for lesson_index, lesson in enumerate(LessonId):
        results = []
        for offset in range(seed_count):
            seed = lesson_index * seed_count + offset
            env = V02SentinelEnv(lesson=lesson, size=size)
            try:
                env.reset(seed=seed)
                if lesson is LessonId.VISIBLE_UNLOCK:
                    assert env.layout is not None
                    if not env.agent_sees(*env.layout.key) or not env.agent_sees(*env.layout.door):
                        raise OracleFailure("visible lesson hid the key or door at reset")
                result = DungeonOracle(env).solve()
                if lesson is LessonId.VISIBLE_UNLOCK and not 5 <= result.steps <= 10:
                    raise OracleFailure(f"visible solution length {result.steps} outside 5..10")
                results.append(result)
            except (AssertionError, OracleFailure, RuntimeError, ValueError) as error:
                failures.append({"lesson": lesson.value, "seed": seed, "error": str(error)})
            finally:
                env.close()
        actions = [result.steps for result in results]
        unique_layouts = len({result.layout_sha256 for result in results})
        if unique_layouts < math.ceil(seed_count * 0.80):
            failures.append(
                {
                    "lesson": lesson.value,
                    "seed": None,
                    "error": (
                        f"only {unique_layouts}/{seed_count} layouts were unique; "
                        "sentinel requires at least 80%"
                    ),
                }
            )
        lessons.append(
            {
                "lesson_id": lesson.value,
                "lesson_label": lesson.label,
                "requested": seed_count,
                "successes": len(results),
                "minimum_actions": min(actions) if actions else 0,
                "maximum_actions": max(actions) if actions else 0,
                "unique_layouts": unique_layouts,
            }
        )
    return {
        "protocol": PROTOCOL,
        "result": "passed" if not failures else "failed",
        "requested_levels": seed_count * len(LessonId),
        "solved_levels": sum(item["successes"] for item in lessons),
        "lessons": lessons,
        "failures": failures,
    }


class _SentinelMastered(RuntimeError):
    pass


def _safe_frame(path: Path, observation: np.ndarray) -> None:
    frame = observation
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(temporary, format="PNG")
    os.replace(temporary, path)


class _CallbackFactory:
    @staticmethod
    def create(base_callback: Any) -> type[Any]:
        class SentinelCallback(base_callback):
            def __init__(
                self,
                *,
                run_directory: Path,
                state: SentinelCurriculumState,
                scheduler: TransitionDeficitScheduler,
                evaluation_interval: int,
                checkpoint_interval: int,
                evaluation_seed_count: int,
                frame_interval: int,
                minimum_free_bytes: int,
                started_at: str,
                effective_config: dict[str, Any],
            ) -> None:
                super().__init__(verbose=0)
                self.run_directory = run_directory
                self.state = state
                self.scheduler = scheduler
                self.evaluation_interval = evaluation_interval
                self.checkpoint_interval = checkpoint_interval
                self.evaluation_seed_count = evaluation_seed_count
                self.frame_interval = frame_interval
                self.minimum_free_bytes = minimum_free_bytes
                self.started_at = started_at
                self.effective_config = effective_config
                self.wall_start = time.monotonic()
                self.trained_timesteps = 0
                self.last_updates = 0
                self.next_evaluation = evaluation_interval
                self.next_checkpoint = checkpoint_interval
                self.next_status = 0
                self.next_frame = 0
                self.frame_revision = 0
                self.episodes = 0
                self.latest_evaluations: list[dict[str, Any]] = []
                self.evaluation_history: deque[dict[str, Any]] = deque(maxlen=100)
                self.recent_outcomes: dict[str, deque[bool]] = defaultdict(
                    lambda: deque(maxlen=100)
                )
                self.latest_safe_checkpoint: Path | None = None
                self._last_saved_step: int | None = None
                self._last_saved_path: Path | None = None
                self._last_transition_step: int | None = None
                self._last_evaluation_step: int | None = None

            def _on_training_start(self) -> None:
                self.last_updates = int(self.model._n_updates)
                self._checkpoint("initial", "initial")
                self._write_status("training")

            def _on_rollout_start(self) -> None:
                if int(self.model._n_updates) > self.last_updates:
                    self._process_boundary()
                if self.state.mastered:
                    raise _SentinelMastered

            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                dones = self.locals.get("dones", [])
                for done, info in zip(dones, infos, strict=False):
                    if done:
                        lesson_id = str(info.get("lesson_id", LessonId.NAVIGATE.value))
                        success = bool(info.get("success", False))
                        self.recent_outcomes[lesson_id].append(success)
                        self.episodes += 1
                        append_jsonl(
                            self.run_directory / "episodes.jsonl",
                            {
                                "timestamp": utc_now(),
                                "episode": self.episodes,
                                "collected_timesteps": int(self.model.num_timesteps),
                                "lesson_id": lesson_id,
                                "lesson_label": info.get("lesson_label"),
                                "tier": info.get("tier"),
                                "seed": info.get("seed"),
                                "layout_sha256": info.get("layout_sha256"),
                                "success": success,
                                "terminal_reason": info.get("terminal_reason"),
                                "milestones": info.get("milestones"),
                                "unique_cells": info.get("unique_cells"),
                                "collisions": info.get("collisions"),
                                "ineffective_interactions": info.get("ineffective_interactions"),
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

            @staticmethod
            def _next_boundary(current: int, interval: int) -> int:
                return ((current // interval) + 1) * interval

            def _process_boundary(self) -> None:
                self.trained_timesteps = int(self.model.num_timesteps)
                self.last_updates = int(self.model._n_updates)
                if self.trained_timesteps >= self.next_checkpoint:
                    self._checkpoint(f"step-{self.trained_timesteps:012d}", "step")
                    self.next_checkpoint = self._next_boundary(
                        self.trained_timesteps, self.checkpoint_interval
                    )
                if self.trained_timesteps >= self.next_evaluation:
                    self.next_evaluation = self._next_boundary(
                        self.trained_timesteps, self.evaluation_interval
                    )
                    self._evaluate_and_advance("scheduled")
                self._write_status("mastered" if self.state.mastered else "training")

            def _evaluation_checkpoint(self) -> Path:
                if (
                    self._last_saved_step == self.trained_timesteps
                    and self._last_saved_path is not None
                    and self._last_saved_path.is_file()
                ):
                    return self._last_saved_path
                return self._checkpoint(f"step-{self.trained_timesteps:012d}", "exam")

            def _evaluate_and_advance(self, trigger: str) -> None:
                allocation_before = self.scheduler.snapshot()
                allocation_ok = self.scheduler.allocation_within()
                checkpoint = self._evaluation_checkpoint()
                digest = file_sha256(checkpoint)
                lessons = [LessonId.NAVIGATE]
                if self.state.active_lesson is LessonId.VISIBLE_UNLOCK:
                    lessons.append(LessonId.VISIBLE_UNLOCK)
                evaluations = [
                    evaluate_lesson(
                        self.model,
                        lesson,
                        validation_seeds(lesson, self.evaluation_seed_count),
                        frame_path=self.run_directory / "frames" / "latest.png",
                    )
                    for lesson in lessons
                ]
                for result in evaluations:
                    append_jsonl(
                        self.run_directory / "evaluations.jsonl",
                        {
                            "trigger": trigger,
                            "trained_timesteps": self.trained_timesteps,
                            "n_updates": int(self.model._n_updates),
                            "checkpoint": str(checkpoint.relative_to(self.run_directory)),
                            "checkpoint_sha256": digest,
                            **result.public_dict(),
                        },
                    )

                navigate = evaluations[0]
                current = evaluations[-1]
                transition_available = self._last_transition_step != self.trained_timesteps
                new_evaluation_boundary = self._last_evaluation_step != self.trained_timesteps
                decision = "Measured"
                navigate_pass = (
                    navigate.success_rate >= RETENTION_THRESHOLD
                    and min(navigate.panel_success_rates) >= RETENTION_THRESHOLD
                )

                if self.state.active_lesson is LessonId.NAVIGATE:
                    current_pass = (
                        current.success_rate >= PROMOTION_THRESHOLD
                        and min(current.panel_success_rates) >= PANEL_PROMOTION_THRESHOLD
                    )
                    if new_evaluation_boundary:
                        self.state.consecutive_passes = (
                            self.state.consecutive_passes + 1 if current_pass else 0
                        )
                    if current_pass and self.state.consecutive_passes < 2:
                        decision = "First stable Navigate pass; confirmation required"
                    elif current_pass and transition_available and allocation_ok:
                        self.state.active_lesson = LessonId.VISIBLE_UNLOCK
                        self.state.consecutive_passes = 0
                        self.state.change()
                        self._last_transition_step = self.trained_timesteps
                        decision = "Promoted to Visible Unlock"
                        self._checkpoint("promotion-visible-unlock", "promotion")
                    elif not allocation_ok:
                        decision = "Held: transition allocation outside tolerance"
                    else:
                        decision = "Held: Navigate stability gate"
                else:
                    if not navigate_pass:
                        if not self.state.recovery:
                            self.state.recovery = True
                            self.state.recovery_passes = 0
                            self.state.consecutive_passes = 0
                            self.state.change()
                        decision = "Recovery: Navigate below 85%"
                    elif self.state.recovery:
                        if new_evaluation_boundary:
                            self.state.recovery_passes += 1
                        if self.state.recovery_passes >= 2:
                            self.state.recovery = False
                            self.state.recovery_passes = 0
                            self.state.change()
                            decision = "Navigate recovered; resuming Visible Unlock"
                        else:
                            decision = "Navigate recovery confirmation required"
                    else:
                        current_pass = (
                            current.success_rate >= STAGE_THRESHOLD
                            and min(current.panel_success_rates) >= PANEL_STAGE_THRESHOLD
                        )
                        if new_evaluation_boundary:
                            self.state.consecutive_passes = (
                                self.state.consecutive_passes + 1 if current_pass else 0
                            )
                        if current_pass and self.state.consecutive_passes < 2:
                            decision = "First Visible Unlock pass; confirmation required"
                        elif current_pass and transition_available and allocation_ok:
                            self.state.mastered = True
                            self._last_transition_step = self.trained_timesteps
                            decision = "V0.2 sentinel mastered"
                            self._checkpoint("mastered-visible-unlock", "mastery")
                        elif not allocation_ok:
                            decision = "Held: transition allocation outside tolerance"
                        else:
                            decision = "Held: Visible Unlock below 85%"

                self.latest_evaluations = [result.public_dict() for result in evaluations]
                for result in evaluations:
                    self.evaluation_history.append(
                        {
                            "trained_timesteps": self.trained_timesteps,
                            "tier_label": result.lesson_label,
                            "lesson_id": result.lesson_id,
                            "success_rate": result.success_rate,
                            "decision": (
                                decision
                                if result.lesson_id == current.lesson_id
                                else "Retention exam"
                            ),
                        }
                    )
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": "curriculum_decision",
                        "trained_timesteps": self.trained_timesteps,
                        "decision": decision,
                        "curriculum": self.state.public_dict(),
                        "allocation": allocation_before,
                        "allocation_within_tolerance": allocation_ok,
                    },
                )
                self._last_evaluation_step = self.trained_timesteps
                if not self.state.mastered:
                    self.scheduler.start_window()
                self.frame_revision += 1

            def _checkpoint(self, name: str, kind: str) -> Path:
                if int(self.model.num_timesteps) != self.trained_timesteps:
                    raise RuntimeError("refusing checkpoint outside a trained boundary")
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
                    "curriculum": self.state.public_dict(),
                    "allocation": self.scheduler.snapshot(),
                    "progress": {
                        "collected_timesteps": int(self.model.num_timesteps),
                        "trained_timesteps": self.trained_timesteps,
                        "n_updates": int(self.model._n_updates),
                    },
                }
                atomic_write_json(destination.with_suffix(".json"), sidecar)
                latest = destination.parent / "latest.zip"
                if destination != latest:
                    atomic_copy_file(destination, latest)
                    atomic_write_json(latest.with_suffix(".json"), sidecar | {"kind": "latest"})
                self.latest_safe_checkpoint = destination
                self._last_saved_step = self.trained_timesteps
                self._last_saved_path = destination
                return destination

            def _write_status(self, phase: str) -> None:
                elapsed = time.monotonic() - self.wall_start
                collected = int(self.model.num_timesteps)
                recent = {
                    lesson.label: {
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
                        "started_at": self.started_at,
                        "updated_at": utc_now(),
                        "elapsed_seconds": elapsed,
                        "total_timesteps": collected,
                        "collected_timesteps": collected,
                        "trained_timesteps": self.trained_timesteps,
                        "n_updates": int(self.model._n_updates),
                        "fps": collected / elapsed if elapsed else 0.0,
                        "episodes": self.episodes,
                        "current_tier_label": self.state.active_lesson.label,
                        "current_lesson_id": self.state.active_lesson.value,
                        "current_lesson_label": self.state.active_lesson.label,
                        "curriculum": self.state.public_dict(),
                        "practice_allocation": self.scheduler.snapshot(),
                        "allocation_within_tolerance": self.scheduler.allocation_within(),
                        "recent_training": recent,
                        "next_evaluation": self.next_evaluation,
                        "next_checkpoint": self.next_checkpoint,
                        "latest_evaluations": self.latest_evaluations,
                        "evaluation_history": list(self.evaluation_history),
                        "frame_revision": self.frame_revision,
                        "latest_safe_checkpoint": (
                            str(self.latest_safe_checkpoint.relative_to(self.run_directory))
                            if self.latest_safe_checkpoint
                            else None
                        ),
                    },
                )

            def finalize(self, phase: str) -> None:
                if int(self.model._n_updates) > self.last_updates:
                    self._process_boundary()
                if int(self.model.num_timesteps) != self.trained_timesteps:
                    raise RuntimeError("run ended with an untrained partial rollout")
                if self._last_evaluation_step != self.trained_timesteps:
                    self._evaluate_and_advance("final")
                final_phase = "mastered" if self.state.mastered else phase
                self._checkpoint(final_phase, "final")
                self._write_status(final_phase)

        return SentinelCallback


def _environment_factory(
    *, scheduler: TransitionDeficitScheduler, seed: int, size: int
) -> Callable[[], gym.Env]:
    return lambda: make_sentinel_training_env(scheduler=scheduler, seed=seed, size=size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=524_288)
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
    parser.add_argument("--evaluation-every", type=int, default=32_768)
    parser.add_argument("--evaluation-seeds", type=int, default=80)
    parser.add_argument("--checkpoint-every", type=int, default=32_768)
    parser.add_argument("--frame-every", type=int, default=2_048)
    parser.add_argument("--qualification-seeds", type=int, default=1_000)
    parser.add_argument("--minimum-free-gib", type=float, default=25.0)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name")
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8781)
    parser.add_argument("--no-dashboard", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.total_timesteps <= 0 or args.workers <= 0 or args.rollout_steps <= 0:
        raise SystemExit("timesteps, workers, and rollout steps must be positive")
    if args.size != 9:
        raise SystemExit("the v0.2 sentinel is qualified only at size 9")
    rollout = args.workers * args.rollout_steps
    if rollout % args.batch_size:
        raise SystemExit("workers x rollout steps must be divisible by batch size")
    if args.total_timesteps % rollout:
        raise SystemExit("total timesteps must align to a complete vector rollout")
    if args.evaluation_seeds != 80:
        raise SystemExit("the v0.2 sentinel freezes evaluation at 80 seeds")
    if args.gamma < MINIMUM_GAMMA or args.gamma > 1.0:
        raise SystemExit("gamma violates reward dominance")
    if not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib < 0:
        raise SystemExit("minimum free GiB must be finite and non-negative")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    except ImportError as error:
        raise SystemExit('Install training dependencies with: pip install -e ".[train]"') from error

    minimum_free_bytes = int(args.minimum_free_gib * 1024**3)
    ensure_disk_space(args.run_root, minimum_free_bytes)
    run_directory = create_run_directory(args.run_root, args.run_name)
    qualification = qualify_sentinel(args.qualification_seeds, size=args.size)
    atomic_write_json(run_directory / "qualification.json", qualification)
    if qualification["result"] != "passed":
        raise SystemExit(f"sentinel qualification failed; inspect {run_directory}")

    state = SentinelCurriculumState()
    scheduler = TransitionDeficitScheduler(state, seed=args.seed + 99_000)
    factories = [
        _environment_factory(scheduler=scheduler, seed=args.seed + worker, size=args.size)
        for worker in range(args.workers)
    ]
    vector_environment = VecTransposeImage(DummyVecEnv(factories))
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
    effective_config = {
        "protocol": PROTOCOL,
        "seed": args.seed,
        "environment": {"size": args.size, "workers": args.workers},
        "optimization": {
            "rollout_steps": args.rollout_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "ent_coef": 0.01,
        },
        "lessons": [lesson.value for lesson in LessonId],
        "evaluation": {
            "interval": args.evaluation_every,
            "seeds": args.evaluation_seeds,
            "promotion_threshold": PROMOTION_THRESHOLD,
            "retention_threshold": RETENTION_THRESHOLD,
            "consecutive_passes": 2,
        },
    }
    repository = Path(__file__).resolve().parents[2]
    started_at = utc_now()
    atomic_write_json(
        run_directory / "manifest.json",
        {
            "protocol": PROTOCOL,
            "started_at": started_at,
            "arguments": vars(args) | {"run_root": str(args.run_root)},
            "effective_config": effective_config,
            "git": git_snapshot(repository),
            "runtime": runtime_snapshot(),
            "information_boundary": "partial RGB pixels plus recurrent state only",
            "online_model_calls": False,
            "development_sentinel": True,
        },
    )

    callback_type = _CallbackFactory.create(BaseCallback)
    callback = callback_type(
        run_directory=run_directory,
        state=state,
        scheduler=scheduler,
        evaluation_interval=args.evaluation_every,
        checkpoint_interval=args.checkpoint_every,
        evaluation_seed_count=args.evaluation_seeds,
        frame_interval=args.frame_every,
        minimum_free_bytes=minimum_free_bytes,
        started_at=started_at,
        effective_config=effective_config,
    )
    dashboard = None
    if not args.no_dashboard:
        dashboard = start_dashboard(
            run_directory, host=args.dashboard_host, port=args.dashboard_port
        )
        print(
            f"Dashboard: http://{args.dashboard_host}:{dashboard.server_port}/",
            flush=True,
        )
    print(f"Run artifacts: {run_directory}", flush=True)

    phase = "completed"
    try:
        try:
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=callback,
                reset_num_timesteps=True,
                progress_bar=False,
            )
        except _SentinelMastered:
            phase = "mastered"
        callback.finalize(phase)
    except BaseException:
        atomic_write_json(
            run_directory / "crash.json",
            {"timestamp": utc_now(), "traceback": traceback.format_exc()},
        )
        raise
    finally:
        vector_environment.close()
        if dashboard is not None:
            dashboard.shutdown()
            dashboard.server_close()


if __name__ == "__main__":
    main()
