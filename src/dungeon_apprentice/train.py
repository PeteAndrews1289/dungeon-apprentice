"""Unattended recurrent-PPO curriculum runner."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dungeon_apprentice.artifacts import (
    append_jsonl,
    atomic_model_save,
    atomic_write_json,
    create_run_directory,
    git_snapshot,
    runtime_snapshot,
    utc_now,
)
from dungeon_apprentice.contracts import (
    DEFAULT_CURIOSITY_SCALE,
    PROMOTION_THRESHOLD,
    PROTOCOL,
    RETENTION_THRESHOLD,
    DungeonTier,
    evaluation_seeds,
)
from dungeon_apprentice.dashboard import start_dashboard
from dungeon_apprentice.env import CurriculumState, make_curriculum_pixel_env
from dungeon_apprentice.evaluate import TierEvaluation, evaluate_policy
from dungeon_apprentice.qualify import qualify


def _require_training_packages() -> tuple[Any, Any, Any, Any]:
    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    except ImportError as error:
        raise SystemExit('Install training dependencies with: pip install -e ".[train]"') from error
    return RecurrentPPO, BaseCallback, DummyVecEnv, VecTransposeImage


def _safe_frame(path: Path, observation: np.ndarray) -> None:
    frame = observation
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    frame = np.asarray(frame, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    Image.fromarray(frame).save(temporary, format="PNG")
    os.replace(temporary, path)


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
                self.wall_start = time.monotonic()
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
                self.mastered = False

            def _on_training_start(self) -> None:
                current = int(self.model.num_timesteps)
                self.next_evaluation = self._next_boundary(current, self.evaluation_interval)
                self.next_checkpoint = self._next_boundary(current, self.checkpoint_interval)
                self.next_status = current
                self.next_frame = current
                self._write_status("training")

            @staticmethod
            def _next_boundary(current: int, interval: int) -> int:
                return ((current // interval) + 1) * interval

            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                dones = self.locals.get("dones", [])
                for done, info in zip(dones, infos, strict=False):
                    if bool(done):
                        tier = int(info.get("tier", 0))
                        self.recent_outcomes[tier].append(bool(info.get("success", False)))
                        self.episode_count += 1

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

                if current >= self.next_checkpoint:
                    self._checkpoint(f"step-{current:012d}")
                    self.next_checkpoint = self._next_boundary(current, self.checkpoint_interval)

                keep_training = True
                if current >= self.next_evaluation:
                    self.next_evaluation = self._next_boundary(
                        current, self.evaluation_interval
                    )
                    keep_training = self._evaluate_and_maybe_promote()

                if current >= self.next_status:
                    self._write_status("training" if keep_training else "mastered")
                    self.next_status = current + 1_000
                return keep_training

            def _evaluate_and_maybe_promote(self) -> bool:
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
                            "timesteps": int(self.num_timesteps),
                            **result.public_dict(),
                        },
                    )

                current_result = evaluations[-1]
                retained = all(
                    result.success_rate >= RETENTION_THRESHOLD for result in evaluations[:-1]
                )
                passed = current_result.success_rate >= PROMOTION_THRESHOLD and retained
                if passed and current_tier < DungeonTier.RETRIEVE:
                    promoted = DungeonTier(int(current_tier) + 1)
                    decision = f"Promoted to {promoted.label}"
                    self.curriculum.max_tier = promoted
                    self._checkpoint(f"promotion-tier-{int(promoted)}")
                elif passed:
                    decision = "Version 0 mastered"
                    self.mastered = True
                    self._checkpoint("mastered-v0")
                elif not retained:
                    decision = "Held: earlier skill retention below 80%"
                else:
                    decision = "Held: current lesson below 90%"

                self.latest_evaluations = [
                    result.public_dict(include_episodes=False) for result in evaluations
                ]
                for result in evaluations:
                    self.evaluation_history.append(
                        {
                            "timesteps": int(self.num_timesteps),
                            "tier": result.tier,
                            "tier_label": result.tier_label,
                            "success_rate": result.success_rate,
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
                        "timesteps": int(self.num_timesteps),
                        "decision": decision,
                        "evaluated_tier": int(current_tier),
                        "next_tier": int(self.curriculum.max_tier),
                    },
                )
                self.frame_revision += 1
                self._write_status("mastered" if self.mastered else "training")
                return not (self.mastered and self.stop_when_mastered)

            def _checkpoint(self, name: str) -> Path:
                destination = atomic_model_save(
                    self.model, self.run_directory / "checkpoints" / name
                )
                atomic_model_save(self.model, self.run_directory / "checkpoints" / "latest")
                append_jsonl(
                    self.run_directory / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "type": "checkpoint",
                        "timesteps": int(self.num_timesteps),
                        "path": str(destination.relative_to(self.run_directory)),
                    },
                )
                return destination

            def _write_status(self, phase: str) -> None:
                elapsed = time.monotonic() - self.wall_start
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
                        "total_timesteps": int(self.num_timesteps),
                        "fps": int(self.num_timesteps) / elapsed if elapsed else 0.0,
                        "episodes": self.episode_count,
                        "current_tier": int(self.curriculum.max_tier),
                        "current_tier_label": self.curriculum.max_tier.label,
                        "recent_training": recent,
                        "next_evaluation": self.next_evaluation,
                        "latest_evaluations": self.latest_evaluations,
                        "evaluation_history": list(self.evaluation_history),
                        "frame_revision": self.frame_revision,
                    },
                )

            def finalize(self, phase: str) -> None:
                self._checkpoint(phase)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--curiosity-scale", type=float, default=DEFAULT_CURIOSITY_SCALE)
    parser.add_argument("--evaluation-every", type=int, default=50_000)
    parser.add_argument("--evaluation-seeds", type=int, default=40)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--frame-every", type=int, default=1_000)
    parser.add_argument("--qualification-seeds", type=int, default=100)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8780)
    parser.add_argument("--continue-after-mastery", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "total timesteps": args.total_timesteps,
        "workers": args.workers,
        "rollout steps": args.rollout_steps,
        "batch size": args.batch_size,
        "evaluation interval": args.evaluation_every,
        "evaluation seeds": args.evaluation_seeds,
        "checkpoint interval": args.checkpoint_every,
        "frame interval": args.frame_every,
        "qualification seeds": args.qualification_seeds,
    }
    for label, value in positive.items():
        if value <= 0:
            raise SystemExit(f"{label} must be positive")
    rollout_size = args.workers * args.rollout_steps
    if rollout_size % args.batch_size:
        raise SystemExit("workers x rollout-steps must be divisible by batch-size")
    if args.curiosity_scale < 0.0:
        raise SystemExit("curiosity scale cannot be negative")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    RecurrentPPO, BaseCallback, DummyVecEnv, VecTransposeImage = _require_training_packages()

    repository = Path(__file__).resolve().parents[2]
    run_directory = create_run_directory(args.run_root, args.run_name)
    started_at = utc_now()
    qualification = qualify(args.qualification_seeds, size=args.size)
    atomic_write_json(run_directory / "qualification.json", qualification)
    if qualification["result"] != "passed":
        raise SystemExit(f"Generator qualification failed; inspect {run_directory}")

    manifest = {
        "protocol": PROTOCOL,
        "started_at": started_at,
        "arguments": vars(args) | {
            "run_root": str(args.run_root),
            "resume": str(args.resume) if args.resume else None,
        },
        "git": git_snapshot(repository),
        "runtime": runtime_snapshot(),
        "information_boundary": "partial RGB pixels plus recurrent state only",
        "online_model_calls": False,
    }
    atomic_write_json(run_directory / "manifest.json", manifest)

    curriculum = CurriculumState()
    if args.resume:
        resume_status = args.resume.resolve().parents[1] / "status.json"
        if resume_status.exists():
            prior = json.loads(resume_status.read_text(encoding="utf-8"))
            curriculum.max_tier = DungeonTier(int(prior.get("current_tier", 0)))

    factories = [
        _environment_factory(
            seed=args.seed + worker,
            state=curriculum,
            size=args.size,
            curiosity_scale=args.curiosity_scale,
        )
        for worker in range(args.workers)
    ]
    vector_environment = VecTransposeImage(DummyVecEnv(factories))
    if args.resume:
        model = RecurrentPPO.load(
            args.resume,
            env=vector_environment,
            device=args.device,
        )
        reset_num_timesteps = False
    else:
        model = RecurrentPPO(
            "CnnLstmPolicy",
            vector_environment,
            learning_rate=args.learning_rate,
            n_steps=args.rollout_steps,
            batch_size=args.batch_size,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            seed=args.seed,
            device=args.device,
            verbose=1,
            policy_kwargs={"lstm_hidden_size": 256, "n_lstm_layers": 1},
        )
        reset_num_timesteps = True

    callback_type = _CallbackFactory.create(BaseCallback)
    callback = callback_type(
        run_directory=run_directory,
        curriculum=curriculum,
        size=args.size,
        evaluation_interval=args.evaluation_every,
        evaluation_seed_count=args.evaluation_seeds,
        checkpoint_interval=args.checkpoint_every,
        frame_interval=args.frame_every,
        stop_when_mastered=not args.continue_after_mastery,
        started_at=started_at,
    )

    dashboard = None
    dashboard_error = None
    try:
        dashboard = start_dashboard(
            run_directory,
            host=args.dashboard_host,
            port=args.dashboard_port,
        )
    except OSError as error:
        dashboard_error = str(error)
        try:
            dashboard = start_dashboard(run_directory, host=args.dashboard_host, port=0)
        except OSError as fallback_error:
            dashboard_error = f"{dashboard_error}; fallback failed: {fallback_error}"
    if dashboard is not None:
        print(f"Dashboard: http://{args.dashboard_host}:{dashboard.server_port}/", flush=True)
    if dashboard_error:
        append_jsonl(
            run_directory / "events.jsonl",
            {"timestamp": utc_now(), "type": "dashboard_warning", "message": dashboard_error},
        )
    print(f"Run artifacts: {run_directory}", flush=True)

    phase = "completed"
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
        if callback.mastered:
            phase = "mastered"
    except KeyboardInterrupt:
        phase = "interrupted"
    except Exception:
        phase = "crashed"
        atomic_write_json(
            run_directory / "crash.json",
            {"timestamp": utc_now(), "traceback": traceback.format_exc()},
        )
        raise
    finally:
        callback.finalize(phase)
        vector_environment.close()
        if dashboard is not None:
            dashboard.shutdown()
            dashboard.server_close()


if __name__ == "__main__":
    main()
