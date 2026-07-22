import json
from pathlib import Path

import pytest

from dungeon_apprentice.artifacts import file_sha256
from dungeon_apprentice.v02_confirm import (
    CONFIRMATION_CASES,
    EXPECTED_TRAINING_SEEDS,
    NAVIGATE_CONFIRMATION_BASE,
    VISIBLE_UNLOCK_CONFIRMATION_BASE,
    ConfirmationError,
    confirmation_seeds,
    grade_evaluation,
    inspect_layouts,
    verify_checkpoint,
)
from dungeon_apprentice.v02_sentinel import (
    CHECKPOINT_SCHEMA_VERSION,
    NAVIGATION_VALIDATION_BASE,
    PROTOCOL,
    VISIBLE_UNLOCK_VALIDATION_BASE,
    LessonEvaluation,
    LessonId,
)


def _bundle(tmp_path: Path, *, seed: int = EXPECTED_TRAINING_SEEDS[0]) -> Path:
    run = tmp_path / f"run-{seed}"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "mastered.zip"
    checkpoint.write_bytes(f"policy-{seed}".encode())
    config = {
        "protocol": PROTOCOL,
        "seed": seed,
        "environment": {"size": 9, "workers": 4},
    }
    sidecar = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "kind": "mastery",
        "checkpoint_sha256": file_sha256(checkpoint),
        "effective_config": config,
        "curriculum": {
            "active_lesson": LessonId.VISIBLE_UNLOCK.value,
            "consecutive_passes": 2,
            "mastered": True,
        },
        "progress": {"collected_timesteps": 491_520, "trained_timesteps": 491_520},
        "allocation": {
            "target_shares": {
                LessonId.NAVIGATE.value: 0.5,
                LessonId.VISIBLE_UNLOCK.value: 0.5,
            },
            "window_transitions": {
                LessonId.NAVIGATE.value: 100,
                LessonId.VISIBLE_UNLOCK.value: 100,
            },
            "realized_shares": {
                LessonId.NAVIGATE.value: 0.5,
                LessonId.VISIBLE_UNLOCK.value: 0.5,
            },
        },
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "effective_config": config,
                "git": {"commit": "a" * 40, "branch": "test", "dirty": False},
            }
        ),
        encoding="utf-8",
    )
    (run / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "curriculum_decision",
                "decision": "V0.2 sentinel mastered",
                "allocation": {
                    "target_shares": {
                        LessonId.NAVIGATE.value: 0.5,
                        LessonId.VISIBLE_UNLOCK.value: 0.5,
                    },
                    "window_transitions": {
                        LessonId.NAVIGATE.value: 100,
                        LessonId.VISIBLE_UNLOCK.value: 100,
                    },
                    "realized_shares": {
                        LessonId.NAVIGATE.value: 0.5,
                        LessonId.VISIBLE_UNLOCK.value: 0.5,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint


def test_confirmation_seed_blocks_are_frozen_and_disjoint() -> None:
    navigate = set(confirmation_seeds(LessonId.NAVIGATE))
    unlock = set(confirmation_seeds(LessonId.VISIBLE_UNLOCK))

    assert len(navigate) == len(unlock) == CONFIRMATION_CASES
    assert min(navigate) == NAVIGATE_CONFIRMATION_BASE
    assert min(unlock) == VISIBLE_UNLOCK_CONFIRMATION_BASE
    assert navigate.isdisjoint(unlock)
    assert min(navigate) >= 15_000_000
    assert navigate.isdisjoint(range(NAVIGATION_VALIDATION_BASE, NAVIGATION_VALIDATION_BASE + 80))
    assert unlock.isdisjoint(
        range(VISIBLE_UNLOCK_VALIDATION_BASE, VISIBLE_UNLOCK_VALIDATION_BASE + 80)
    )


def test_confirmation_seed_count_cannot_be_changed() -> None:
    with pytest.raises(ValueError, match="exactly 200"):
        confirmation_seeds(LessonId.NAVIGATE, 199)


def test_checkpoint_verification_requires_mastery_digest_and_allocation(tmp_path: Path) -> None:
    checkpoint = _bundle(tmp_path)
    digest = file_sha256(checkpoint)
    selection = {
        EXPECTED_TRAINING_SEEDS[0]: {
            "sha256": digest,
            "source_commit": "a" * 40,
        }
    }
    verified = verify_checkpoint(checkpoint, expected_checkpoints=selection)

    assert verified.training_seed == EXPECTED_TRAINING_SEEDS[0]
    assert verified.allocation_windows == 1
    assert verified.maximum_allocation_deviation == 0.0

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ConfirmationError, match="digest"):
        verify_checkpoint(checkpoint, expected_checkpoints=selection)


def test_checkpoint_verification_rejects_out_of_tolerance_allocation(tmp_path: Path) -> None:
    checkpoint = _bundle(tmp_path)
    selection = {
        EXPECTED_TRAINING_SEEDS[0]: {
            "sha256": file_sha256(checkpoint),
            "source_commit": "a" * 40,
        }
    }
    events = checkpoint.parent.parent / "events.jsonl"
    value = json.loads(events.read_text(encoding="utf-8"))
    value["allocation"]["realized_shares"] = {
        LessonId.NAVIGATE.value: 0.4,
        LessonId.VISIBLE_UNLOCK.value: 0.6,
    }
    events.write_text(json.dumps(value) + "\n", encoding="utf-8")

    with pytest.raises(ConfirmationError, match="exceeded tolerance"):
        verify_checkpoint(checkpoint, expected_checkpoints=selection)


@pytest.mark.parametrize(
    ("lesson", "overall", "panels", "passed"),
    [
        (LessonId.NAVIGATE, 0.85, (0.85, 0.85), True),
        (LessonId.NAVIGATE, 0.85, (0.84, 0.86), False),
        (LessonId.VISIBLE_UNLOCK, 0.85, (0.80, 0.90), True),
        (LessonId.VISIBLE_UNLOCK, 0.84, (0.84, 0.84), False),
    ],
)
def test_confirmation_gate_includes_overall_and_panel_floors(
    lesson: LessonId, overall: float, panels: tuple[float, float], passed: bool
) -> None:
    result = LessonEvaluation(
        protocol=PROTOCOL,
        timestamp="2026-07-22T00:00:00Z",
        lesson_id=lesson.value,
        lesson_label=lesson.label,
        episodes=CONFIRMATION_CASES,
        successes=int(overall * CONFIRMATION_CASES),
        success_rate=overall,
        panel_success_rates=panels,
        mean_steps=10.0,
    )

    assert grade_evaluation(result)["passed"] is passed


def test_layout_inspection_uses_non_confirmation_test_seeds() -> None:
    report = inspect_layouts(LessonId.VISIBLE_UNLOCK, (101, 102, 103))
    assert report["seed_cases"] == 3
    assert 1 <= report["unique_layouts"] <= 3
