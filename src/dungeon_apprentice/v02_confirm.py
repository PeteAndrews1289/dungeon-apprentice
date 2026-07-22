"""Preregistered post-training confirmation for the three mastered v0.2 U0 policies."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dungeon_apprentice.artifacts import (
    atomic_write_json,
    file_sha256,
    git_snapshot,
    runtime_snapshot,
    utc_now,
)
from dungeon_apprentice.oracle import DungeonOracle, OracleFailure
from dungeon_apprentice.v02_sentinel import (
    CHECKPOINT_SCHEMA_VERSION,
    PROTOCOL,
    LessonEvaluation,
    LessonId,
    V02SentinelEnv,
    evaluate_lesson,
)

CONFIRMATION_PROTOCOL = "dungeon-apprentice-v0.2-u0-confirmation"
CONFIRMATION_SCHEMA_VERSION = 1
CONFIRMATION_CASES = 200
NAVIGATE_CONFIRMATION_BASE = 15_000_000
VISIBLE_UNLOCK_CONFIRMATION_BASE = 15_010_000
EXPECTED_TRAINING_SEEDS = (20260725, 20260726, 20260727)
EXPECTED_CHECKPOINTS: dict[int, dict[str, str]] = {
    20260725: {
        "sha256": "2b235ed54746429e737af2a6037069821fdaf81a936edf8f74ce86cc766b40a2",
        "source_commit": "d27070f76a0bf19b64b82a9b15fac567751a0339",
    },
    20260726: {
        "sha256": "86c5bd42cd36209cd23e27569da4684229d6c52d09082708b7d1fda380826be4",
        "source_commit": "fefcd08c8c3127214b10620978cc4b1f8409678c",
    },
    20260727: {
        "sha256": "aba4d8693ed56c9e3fee57bcee21a33f682b23bc8dad4978dce0f765d620e11b",
        "source_commit": "fefcd08c8c3127214b10620978cc4b1f8409678c",
    },
}
NAVIGATE_OVERALL_THRESHOLD = 0.85
NAVIGATE_PANEL_THRESHOLD = 0.85
VISIBLE_UNLOCK_OVERALL_THRESHOLD = 0.85
VISIBLE_UNLOCK_PANEL_THRESHOLD = 0.80
MINIMUM_UNIQUE_LAYOUT_RATE = 0.80
ALLOCATION_TOLERANCE = 0.05


class ConfirmationError(RuntimeError):
    """Raised when a selected artifact cannot support the frozen confirmation claim."""


@dataclass(frozen=True)
class VerifiedCheckpoint:
    checkpoint: str
    checkpoint_sha256: str
    expected_checkpoint_sha256: str
    digest_verified: bool
    sidecar: str
    manifest: str
    training_seed: int
    source_commit: str
    expected_source_commit: str
    source_verified: bool
    trained_timesteps: int
    final_window_target_shares: dict[str, float]
    final_window_transitions: dict[str, int]
    final_window_realized_shares: dict[str, float]
    allocation_windows: int
    maximum_allocation_deviation: float
    allocation_verified: bool


def confirmation_seeds(lesson: LessonId, count: int = CONFIRMATION_CASES) -> tuple[int, ...]:
    if count != CONFIRMATION_CASES:
        raise ValueError(f"the frozen confirmation requires exactly {CONFIRMATION_CASES} cases")
    base = (
        NAVIGATE_CONFIRMATION_BASE
        if lesson is LessonId.NAVIGATE
        else VISIBLE_UNLOCK_CONFIRMATION_BASE
    )
    return tuple(base + offset for offset in range(count))


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ConfirmationError(f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfirmationError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfirmationError(f"{description} must contain a JSON object: {path}")
    return value


def _verify_allocation(events_path: Path) -> tuple[int, float]:
    if not events_path.is_file():
        raise ConfirmationError(f"missing curriculum event history: {events_path}")
    windows = 0
    maximum_deviation = 0.0
    mastery_seen = False
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ConfirmationError(f"cannot read curriculum events {events_path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ConfirmationError(
                f"invalid curriculum event JSON at {events_path}:{line_number}"
            ) from error
        if event.get("type") != "curriculum_decision":
            continue
        mastery_seen = mastery_seen or event.get("decision") == "V0.2 sentinel mastered"
        allocation = event.get("allocation", {})
        targets = allocation.get("target_shares", {})
        transitions = allocation.get("window_transitions", {})
        realized = allocation.get("realized_shares", {})
        if LessonId.VISIBLE_UNLOCK.value not in targets:
            continue
        total = sum(int(value) for value in transitions.values())
        if total <= 0:
            continue
        windows += 1
        for lesson, target in targets.items():
            deviation = abs(float(realized.get(lesson, 0.0)) - float(target))
            maximum_deviation = max(maximum_deviation, deviation)
            if deviation > ALLOCATION_TOLERANCE + 1e-12:
                raise ConfirmationError(
                    f"practice allocation exceeded tolerance in {events_path}: "
                    f"{lesson} deviation={deviation:.6f}"
                )
    if not mastery_seen:
        raise ConfirmationError(f"curriculum history has no mastery decision: {events_path}")
    if windows == 0:
        raise ConfirmationError(f"curriculum history has no completed U0 windows: {events_path}")
    return windows, maximum_deviation


def verify_checkpoint(
    checkpoint: Path,
    *,
    expected_checkpoints: Mapping[int, Mapping[str, str]] = EXPECTED_CHECKPOINTS,
) -> VerifiedCheckpoint:
    archive = checkpoint.expanduser().resolve()
    if archive.suffix != ".zip":
        archive = archive.with_suffix(".zip")
    if not archive.is_file():
        raise ConfirmationError(f"checkpoint does not exist: {archive}")
    sidecar_path = archive.with_suffix(".json")
    sidecar = _read_json(sidecar_path, "checkpoint sidecar")
    if sidecar.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ConfirmationError(
            f"checkpoint schema mismatch for {archive}: {sidecar.get('schema_version')!r}"
        )
    if sidecar.get("protocol") != PROTOCOL:
        raise ConfirmationError(
            f"checkpoint protocol mismatch for {archive}: {sidecar.get('protocol')!r}"
        )
    digest = file_sha256(archive)
    if sidecar.get("checkpoint_sha256") != digest:
        raise ConfirmationError(f"checkpoint digest does not match sidecar: {archive}")
    if sidecar.get("kind") != "mastery":
        raise ConfirmationError(f"confirmation requires the mastery artifact: {archive}")

    curriculum = sidecar.get("curriculum", {})
    if (
        curriculum.get("mastered") is not True
        or curriculum.get("active_lesson") != LessonId.VISIBLE_UNLOCK.value
        or int(curriculum.get("consecutive_passes", 0)) < 2
    ):
        raise ConfirmationError(f"checkpoint is not a mastered U0 policy: {archive}")
    progress = sidecar.get("progress", {})
    collected = int(progress.get("collected_timesteps", -1))
    trained = int(progress.get("trained_timesteps", -2))
    if collected != 491_520 or collected != trained:
        raise ConfirmationError(
            f"checkpoint is not a fully trained boundary: collected={collected}, trained={trained}"
        )
    try:
        training_seed = int(sidecar["effective_config"]["seed"])
        size = int(sidecar["effective_config"]["environment"]["size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ConfirmationError(
            f"checkpoint has invalid effective configuration: {archive}"
        ) from error
    if training_seed not in EXPECTED_TRAINING_SEEDS:
        raise ConfirmationError(f"checkpoint uses an unselected training seed: {training_seed}")
    if size != 9:
        raise ConfirmationError(
            f"checkpoint was not trained on the frozen 9 x 9 environment: {size}"
        )

    run_directory = archive.parent.parent
    manifest_path = run_directory / "manifest.json"
    manifest = _read_json(manifest_path, "run manifest")
    if manifest.get("protocol") != PROTOCOL:
        raise ConfirmationError(f"run manifest protocol mismatch: {manifest_path}")
    if manifest.get("effective_config") != sidecar.get("effective_config"):
        raise ConfirmationError(f"manifest and sidecar configurations differ: {archive}")
    source = manifest.get("git", {})
    source_commit = source.get("commit")
    if source.get("dirty") is not False or not isinstance(source_commit, str) or not source_commit:
        raise ConfirmationError(
            f"checkpoint was not produced from a clean identified commit: {archive}"
        )

    expected = expected_checkpoints.get(training_seed)
    if expected is None:
        raise ConfirmationError(f"no frozen selection exists for training seed {training_seed}")
    if digest != expected.get("sha256"):
        raise ConfirmationError(
            f"checkpoint does not match the preregistered digest for seed {training_seed}"
        )
    if source_commit != expected.get("source_commit"):
        raise ConfirmationError(
            f"checkpoint does not match the preregistered source for seed {training_seed}"
        )

    target = {
        LessonId.NAVIGATE.value: 0.5,
        LessonId.VISIBLE_UNLOCK.value: 0.5,
    }
    allocation = sidecar.get("allocation", {})
    if allocation.get("target_shares") != target:
        raise ConfirmationError(f"mastery sidecar has the wrong transition target: {archive}")
    transitions = allocation.get("window_transitions", {})
    realized = allocation.get("realized_shares", {})
    if sum(int(value) for value in transitions.values()) <= 0:
        raise ConfirmationError(f"mastery sidecar has no completed allocation window: {archive}")
    if any(
        abs(float(realized.get(lesson, 0.0)) - share) > ALLOCATION_TOLERANCE + 1e-12
        for lesson, share in target.items()
    ):
        raise ConfirmationError(f"mastery sidecar allocation exceeds tolerance: {archive}")

    allocation_windows, maximum_deviation = _verify_allocation(run_directory / "events.jsonl")
    return VerifiedCheckpoint(
        checkpoint=str(archive),
        checkpoint_sha256=digest,
        expected_checkpoint_sha256=str(expected["sha256"]),
        digest_verified=True,
        sidecar=str(sidecar_path),
        manifest=str(manifest_path),
        training_seed=training_seed,
        source_commit=source_commit,
        expected_source_commit=str(expected["source_commit"]),
        source_verified=True,
        trained_timesteps=trained,
        final_window_target_shares={
            lesson: float(share) for lesson, share in target.items()
        },
        final_window_transitions={
            lesson: int(transitions[lesson]) for lesson in target
        },
        final_window_realized_shares={
            lesson: float(realized[lesson]) for lesson in target
        },
        allocation_windows=allocation_windows,
        maximum_allocation_deviation=maximum_deviation,
        allocation_verified=True,
    )


def inspect_layouts(lesson: LessonId, seeds: tuple[int, ...], *, size: int = 9) -> dict[str, Any]:
    hashes: list[str] = []
    seed_layout_hashes: list[dict[str, Any]] = []
    actions: list[int] = []
    failures: list[dict[str, Any]] = []
    for seed in seeds:
        env = V02SentinelEnv(lesson=lesson, size=size)
        try:
            env.reset(seed=seed)
            if lesson is LessonId.VISIBLE_UNLOCK:
                assert env.layout is not None
                if not env.agent_sees(*env.layout.key) or not env.agent_sees(*env.layout.door):
                    raise OracleFailure("visible confirmation case hid the key or door")
            result = DungeonOracle(env).solve()
            if lesson is LessonId.VISIBLE_UNLOCK and not 5 <= result.steps <= 10:
                raise OracleFailure(
                    f"visible confirmation solution length {result.steps} outside 5..10"
                )
            hashes.append(result.layout_sha256)
            seed_layout_hashes.append({"seed": seed, "layout_sha256": result.layout_sha256})
            actions.append(result.steps)
        except (AssertionError, OracleFailure, RuntimeError, ValueError) as error:
            failures.append({"seed": seed, "error": str(error)})
        finally:
            env.close()
    unique = len(set(hashes))
    unique_rate = unique / len(seeds)
    return {
        "lesson_id": lesson.value,
        "seed_cases": len(seeds),
        "solved_cases": len(actions),
        "minimum_oracle_actions": min(actions) if actions else None,
        "maximum_oracle_actions": max(actions) if actions else None,
        "unique_layouts": unique,
        "unique_rate": unique_rate,
        "seed_layout_hashes": seed_layout_hashes,
        "failures": failures,
        "qualification_floor": MINIMUM_UNIQUE_LAYOUT_RATE,
        "passed": len(actions) == len(seeds)
        and unique_rate >= MINIMUM_UNIQUE_LAYOUT_RATE,
    }


def grade_evaluation(result: LessonEvaluation) -> dict[str, Any]:
    lesson = LessonId(result.lesson_id)
    if result.episodes != CONFIRMATION_CASES or len(result.panel_success_rates) != 2:
        raise ConfirmationError(
            f"confirmation evaluation has the wrong case or panel count: {result.public_dict()}"
        )
    if lesson is LessonId.NAVIGATE:
        overall_threshold = NAVIGATE_OVERALL_THRESHOLD
        panel_threshold = NAVIGATE_PANEL_THRESHOLD
    else:
        overall_threshold = VISIBLE_UNLOCK_OVERALL_THRESHOLD
        panel_threshold = VISIBLE_UNLOCK_PANEL_THRESHOLD
    return {
        "overall_threshold": overall_threshold,
        "panel_threshold": panel_threshold,
        "overall_passed": result.success_rate >= overall_threshold,
        "panels_passed": all(
            value >= panel_threshold for value in result.panel_success_rates
        ),
        "passed": result.success_rate >= overall_threshold
        and all(value >= panel_threshold for value in result.panel_success_rates),
    }


def _evaluate_selected_checkpoint(
    verified: VerifiedCheckpoint, model_type: Any
) -> dict[str, Any]:
    model = model_type.load(verified.checkpoint, device="cpu")
    evaluations = []
    for lesson in LessonId:
        result = evaluate_lesson(model, lesson, confirmation_seeds(lesson), size=9)
        panel_size = result.episodes // 2
        evaluations.append(
            result.public_dict()
            | {
                "panel_successes": [
                    round(rate * panel_size) for rate in result.panel_success_rates
                ],
                "gate": grade_evaluation(result),
            }
        )
    return {
        "checkpoint": asdict(verified),
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
        help="selected mastered.zip; provide exactly three",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(args.checkpoint) != len(EXPECTED_TRAINING_SEEDS):
        raise SystemExit("confirmation requires exactly three selected checkpoints")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite confirmation report: {output}")
    try:
        verified = [verify_checkpoint(path) for path in args.checkpoint]
        training_seeds = [item.training_seed for item in verified]
        if sorted(training_seeds) != list(EXPECTED_TRAINING_SEEDS):
            raise ConfirmationError(
                f"selected seeds must be {EXPECTED_TRAINING_SEEDS}, found {training_seeds}"
            )
        if len(set(item.checkpoint_sha256 for item in verified)) != len(verified):
            raise ConfirmationError("selected checkpoints are not three distinct policies")
        verified.sort(key=lambda item: item.training_seed)
        layouts = [
            inspect_layouts(lesson, confirmation_seeds(lesson), size=9) for lesson in LessonId
        ]
        if not all(item["passed"] for item in layouts):
            raise ConfirmationError(f"confirmation layout qualification failed: {layouts}")
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError as error:
            raise ConfirmationError(
                'install training dependencies with: pip install -e ".[train]"'
            ) from error
        started_at = utc_now()
        checkpoints = [
            _evaluate_selected_checkpoint(item, RecurrentPPO) for item in verified
        ]
        repository = Path(__file__).resolve().parents[2]
        source = git_snapshot(repository)
        report = {
            "schema_version": CONFIRMATION_SCHEMA_VERSION,
            "protocol": CONFIRMATION_PROTOCOL,
            "started_at": started_at,
            "completed_at": utc_now(),
            "source": source,
            "runtime": runtime_snapshot(),
            "policy_updates": False,
            "online_model_calls": False,
            "action_selection": "deterministic",
            "recurrent_state_reset_each_case": True,
            "interruption": None,
            "selected_training_seeds": list(EXPECTED_TRAINING_SEEDS),
            "seed_partitions": {
                LessonId.NAVIGATE.value: {
                    "start": NAVIGATE_CONFIRMATION_BASE,
                    "end": NAVIGATE_CONFIRMATION_BASE + CONFIRMATION_CASES - 1,
                },
                LessonId.VISIBLE_UNLOCK.value: {
                    "start": VISIBLE_UNLOCK_CONFIRMATION_BASE,
                    "end": VISIBLE_UNLOCK_CONFIRMATION_BASE + CONFIRMATION_CASES - 1,
                },
            },
            "layout_qualification": layouts,
            "checkpoints": checkpoints,
            "verdict": "passed" if all(item["passed"] for item in checkpoints) else "failed",
        }
        atomic_write_json(output, report)
    except ConfirmationError as error:
        raise SystemExit(str(error)) from error
    sys.stdout.write(f"{report['verdict']}: {output}\n")
    if report["verdict"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
