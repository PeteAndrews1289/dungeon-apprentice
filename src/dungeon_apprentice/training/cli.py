"""Argument parsing, validation, and the training entry point."""

from __future__ import annotations

import argparse
import math
import sys
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dungeon_apprentice.artifacts import (
    append_jsonl,
    atomic_write_json,
    create_run_directory,
    ensure_disk_space,
    git_snapshot,
    runtime_snapshot,
    utc_now,
)
from dungeon_apprentice.contracts import (
    DEFAULT_CURIOSITY_SCALE,
    EVALUATION_TIER_STRIDE,
    MINIMUM_GAMMA,
    PROTOCOL,
    DungeonTier,
)
from dungeon_apprentice.dashboard import start_dashboard
from dungeon_apprentice.env import CurriculumState
from dungeon_apprentice.qualify import qualify
from dungeon_apprentice.training.callbacks import (
    _activate_segment_seed,
    _CallbackFactory,
    _environment_factory,
)
from dungeon_apprentice.training.config import (
    POLICY_KWARGS,
    _actual_effective_config,
    _config_differences,
    _CurriculumMastered,
    _load_resume_metadata,
    _requested_effective_config,
    _require_training_packages,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--curiosity-scale", type=float, default=DEFAULT_CURIOSITY_SCALE)
    parser.add_argument("--evaluation-every", type=int, default=50_000)
    parser.add_argument("--evaluation-seeds", type=int, default=40)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--keep-checkpoints", type=int, default=5)
    parser.add_argument("--minimum-free-gib", type=float, default=2.0)
    parser.add_argument("--frame-every", type=int, default=1_000)
    parser.add_argument("--qualification-seeds", type=int, default=100)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8780)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--continue-after-mastery", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "total timesteps": args.total_timesteps,
        "workers": args.workers,
        "rollout steps": args.rollout_steps,
        "batch size": args.batch_size,
        "epochs": args.n_epochs,
        "evaluation interval": args.evaluation_every,
        "evaluation seeds": args.evaluation_seeds,
        "checkpoint interval": args.checkpoint_every,
        "checkpoint retention": args.keep_checkpoints,
        "frame interval": args.frame_every,
        "qualification seeds": args.qualification_seeds,
    }
    for label, value in positive.items():
        if value <= 0:
            raise SystemExit(f"{label} must be positive")
    rollout_size = args.workers * args.rollout_steps
    if rollout_size % args.batch_size:
        raise SystemExit("workers x rollout-steps must be divisible by batch-size")
    if args.size < 7:
        raise SystemExit("dungeon size must be at least 7")
    if args.evaluation_seeds > EVALUATION_TIER_STRIDE:
        raise SystemExit(
            f"evaluation seeds cannot exceed {EVALUATION_TIER_STRIDE} per tier"
        )
    if not math.isfinite(args.curiosity_scale):
        raise SystemExit("curiosity scale must be finite")
    if args.curiosity_scale != DEFAULT_CURIOSITY_SCALE:
        raise SystemExit(
            f"protocol {PROTOCOL} freezes curiosity scale at "
            f"{DEFAULT_CURIOSITY_SCALE}"
        )
    if not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib < 0.0:
        raise SystemExit("minimum free GiB must be finite and non-negative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise SystemExit("learning rate must be finite and positive")
    if not math.isfinite(args.gamma) or not MINIMUM_GAMMA <= args.gamma <= 1.0:
        raise SystemExit(
            f"gamma must be in [{MINIMUM_GAMMA}, 1] to preserve discounted "
            "reward dominance"
        )
    if not math.isfinite(args.gae_lambda) or not 0.0 < args.gae_lambda <= 1.0:
        raise SystemExit("GAE lambda must be in (0, 1]")


def _best_effort(label: str, operation: Callable[[], Any]) -> None:
    """Run cleanup/reporting without replacing an already-active failure."""

    try:
        operation()
    except BaseException as error:
        print(f"Warning: {label} failed during finalization: {error}", file=sys.stderr)


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    expected_config = _requested_effective_config(args)
    resume_checkpoint: Path | None = None
    resume_sidecar: dict[str, Any] | None = None
    if args.resume:
        resume_checkpoint, resume_sidecar = _load_resume_metadata(
            args.resume, expected_config
        )

    minimum_free_bytes = int(args.minimum_free_gib * (1024**3))
    ensure_disk_space(args.run_root, minimum_free_bytes)
    RecurrentPPO, BaseCallback, DummyVecEnv, VecTransposeImage = _require_training_packages()

    repository = Path(__file__).resolve().parents[2]
    run_directory = create_run_directory(args.run_root, args.run_name)
    started_at = utc_now()
    qualification = qualify(args.qualification_seeds, size=args.size)
    atomic_write_json(run_directory / "qualification.json", qualification)
    if qualification["result"] != "passed":
        raise SystemExit(f"Generator qualification failed; inspect {run_directory}")

    prior_segment = resume_sidecar.get("segment", {}) if resume_sidecar else {}
    segment_index = int(prior_segment.get("index", -1)) + 1
    initial_progress = resume_sidecar.get("progress", {}) if resume_sidecar else {}
    initial_collected = int(initial_progress.get("collected_timesteps", 0))
    initial_trained = int(initial_progress.get("trained_timesteps", 0))
    worker_seed_base = args.seed + segment_index * 10_000
    segment = {
        "id": uuid.uuid4().hex,
        "index": segment_index,
        "started_at": started_at,
        "run_directory": str(run_directory),
        "start_collected_timesteps": initial_collected,
        "start_trained_timesteps": initial_trained,
        "worker_seed_base": worker_seed_base,
        "algorithm_seed": worker_seed_base,
        "parent": (
            {
                "checkpoint": str(resume_checkpoint),
                "checkpoint_sha256": resume_sidecar["checkpoint_sha256"],
                "segment_id": prior_segment.get("id"),
                "collected_timesteps": initial_collected,
                "trained_timesteps": initial_trained,
            }
            if resume_sidecar is not None
            else None
        ),
    }

    curriculum = CurriculumState()
    if resume_sidecar:
        curriculum.max_tier = DungeonTier(
            int(resume_sidecar.get("curriculum", {}).get("max_tier", 0))
        )

    factories = [
        _environment_factory(
            seed=worker_seed_base + worker,
            state=curriculum,
            size=args.size,
            curiosity_scale=args.curiosity_scale,
        )
        for worker in range(args.workers)
    ]
    vector_environment = VecTransposeImage(DummyVecEnv(factories))
    if resume_checkpoint:
        model = RecurrentPPO.load(
            resume_checkpoint,
            env=vector_environment,
            device=args.device,
        )
        reset_num_timesteps = False
        if int(model.num_timesteps) != initial_collected:
            raise SystemExit(
                "resume model/sidecar timestep mismatch: "
                f"model={model.num_timesteps}, sidecar={initial_collected}"
            )
        if int(model._n_updates) != int(initial_progress.get("n_updates", -1)):
            raise SystemExit(
                "resume model/sidecar optimizer update mismatch: "
                f"model={model._n_updates}, sidecar={initial_progress.get('n_updates')!r}"
            )
    else:
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
        reset_num_timesteps = True

    # Loading an SB3 model re-stages its original seed on the vector environment.
    # Re-seed after both construction paths so child segments do not replay segment 0.
    _activate_segment_seed(model, worker_seed_base)

    effective_config = _actual_effective_config(model, args)
    differences = _config_differences(expected_config, effective_config)
    if differences:
        rendered = "\n  - ".join(differences)
        raise SystemExit(f"effective model configuration mismatch:\n  - {rendered}")

    manifest = {
        "protocol": PROTOCOL,
        "started_at": started_at,
        "arguments": vars(args)
        | {
            "run_root": str(args.run_root),
            "resume": str(resume_checkpoint) if resume_checkpoint else None,
        },
        "effective_config": effective_config,
        "segment": segment,
        "git": git_snapshot(repository),
        "runtime": runtime_snapshot(),
        "information_boundary": "partial RGB pixels plus recurrent state only",
        "online_model_calls": False,
    }
    atomic_write_json(run_directory / "manifest.json", manifest)

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
        effective_config=effective_config,
        segment=segment,
        initial_trained_timesteps=initial_trained,
        initial_mastered=bool(
            resume_sidecar.get("curriculum", {}).get("mastered", False)
        )
        if resume_sidecar
        else False,
        keep_checkpoints=args.keep_checkpoints,
        minimum_free_bytes=minimum_free_bytes,
    )

    dashboard = None
    if not args.no_dashboard:
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
                dashboard = start_dashboard(
                    run_directory, host=args.dashboard_host, port=0
                )
            except OSError as fallback_error:
                dashboard_error = f"{dashboard_error}; fallback failed: {fallback_error}"
        if dashboard is not None:
            print(
                f"Dashboard: http://{args.dashboard_host}:{dashboard.server_port}/",
                flush=True,
            )
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
                total_timesteps=args.total_timesteps,
                callback=callback,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=False,
            )
        except _CurriculumMastered:
            phase = "mastered"
        except KeyboardInterrupt:
            phase = "interrupted"

        if phase == "interrupted":
            _best_effort(
                "interruption record", lambda: callback.finalize_failure(phase)
            )
        else:
            callback.finalize_success(phase)
    except BaseException:
        phase = "crashed"
        crash = {"timestamp": utc_now(), "traceback": traceback.format_exc()}
        _best_effort(
            "crash report", lambda: atomic_write_json(run_directory / "crash.json", crash)
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
