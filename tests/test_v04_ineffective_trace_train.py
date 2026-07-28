from __future__ import annotations

import inspect
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v04_ineffective_trace as v04
from dungeon_apprentice import v04_ineffective_trace_train as trainer
from dungeon_apprentice.action_streak import (
    IMAGE_KEY,
    INEFFECTIVE_TRACE_DIM,
    INEFFECTIVE_TRACE_KEY,
    IneffectiveTraceMode,
)
from dungeon_apprentice.artifacts import atomic_write_json


def test_extract_image_observation_requires_one_scalar_trace() -> None:
    images = np.zeros((2, 3, 56, 56), dtype=np.uint8)
    traces = np.zeros((2, INEFFECTIVE_TRACE_DIM), dtype=np.float32)
    selected = trainer.extract_image_observation(
        {
            IMAGE_KEY: images,
            INEFFECTIVE_TRACE_KEY: traces,
        },
        worker_index=1,
    )

    assert selected.shape == (3, 56, 56)
    assert selected.dtype == np.uint8
    selected[0, 0, 0] = 1
    assert images[1, 0, 0, 0] == 0

    with pytest.raises(
        trainer.IneffectiveTraceTrainingError,
        match="noncanonical",
    ):
        trainer.extract_image_observation({IMAGE_KEY: images})

    with pytest.raises(
        trainer.IneffectiveTraceTrainingError,
        match="shape or dtype",
    ):
        trainer.extract_image_observation(
            {
                IMAGE_KEY: images,
                INEFFECTIVE_TRACE_KEY: np.zeros(
                    (2, 9),
                    dtype=np.float32,
                ),
            }
        )


def test_dict_rng_snapshot_restore_includes_trace_child() -> None:
    space = gym.spaces.Dict(
        {
            IMAGE_KEY: gym.spaces.Box(
                0,
                255,
                shape=(3, 4, 4),
                dtype=np.uint8,
            ),
            INEFFECTIVE_TRACE_KEY: gym.spaces.Box(
                0.0,
                1.0,
                shape=(INEFFECTIVE_TRACE_DIM,),
                dtype=np.float32,
            ),
        }
    )
    seeded = trainer.seed_space_tree(space, 42)
    assert set(seeded["children"]) == {
        IMAGE_KEY,
        INEFFECTIVE_TRACE_KEY,
    }
    snapshot = trainer.snapshot_space_rng(space)
    expected = space.sample()
    trainer.restore_space_rng(space, snapshot)
    actual = space.sample()
    assert np.array_equal(expected[IMAGE_KEY], actual[IMAGE_KEY])
    assert np.array_equal(
        expected[INEFFECTIVE_TRACE_KEY],
        actual[INEFFECTIVE_TRACE_KEY],
    )


def test_trace_metrics_are_descriptive_and_retain_raw_distribution() -> None:
    metrics = trainer.TraceMetrics()
    metrics.observe(
        lesson_id=lessons.LessonId.NAVIGATE.value,
        changed=True,
        raw_trace_count=0,
        trace_enabled=True,
    )
    metrics.observe(
        lesson_id=lessons.LessonId.NAVIGATE.value,
        changed=False,
        raw_trace_count=3,
        trace_enabled=True,
    )
    metrics.observe(
        lesson_id=lessons.LessonId.VISIBLE_UNLOCK.value,
        changed=False,
        raw_trace_count=3,
        trace_enabled=False,
    )

    public = metrics.public_dict()
    assert public == {
        "transitions": 3,
        "visible_changed": 1,
        "visible_unchanged": 2,
        "visible_changed_rate": pytest.approx(1 / 3),
        "trace_enabled_transitions": 2,
        "trace_nonzero_transitions": 1,
        "trace_activation_rate": pytest.approx(1 / 3),
        "raw_trace_sum": 6,
        "raw_trace_mean": 2.0,
        "raw_trace_max": 3,
        "raw_trace_histogram": {"0": 1, "3": 2},
        "by_lesson": {
            lessons.LessonId.NAVIGATE.value: {
                "transitions": 2,
                "visible_changed": 1,
                "visible_unchanged": 1,
                "trace_nonzero_transitions": 1,
                "raw_trace_max": 3,
            },
            lessons.LessonId.VISIBLE_UNLOCK.value: {
                "transitions": 1,
                "visible_changed": 0,
                "visible_unchanged": 1,
                "trace_nonzero_transitions": 0,
                "raw_trace_max": 3,
            },
            lessons.LessonId.LOCAL_UNLOCK.value: {
                "transitions": 0,
                "visible_changed": 0,
                "visible_unchanged": 0,
                "trace_nonzero_transitions": 0,
                "raw_trace_max": 0,
            },
            lessons.LessonId.SEPARATED_UNLOCK.value: {
                "transitions": 0,
                "visible_changed": 0,
                "visible_unchanged": 0,
                "trace_nonzero_transitions": 0,
                "raw_trace_max": 0,
            },
        },
        "descriptive_only": True,
        "affects_learning_or_selection": False,
    }

    with pytest.raises(
        trainer.IneffectiveTraceTrainingError,
        match="invalid count",
    ):
        metrics.observe(
            lesson_id=lessons.LessonId.NAVIGATE.value,
            changed=False,
            raw_trace_count=v04.TRACE_CAP + 1,
            trace_enabled=True,
        )


class _Qualification:
    report_sha256 = "f" * 64

    @staticmethod
    def public_dict() -> dict[str, object]:
        return {
            "protocol": v04.PROTOCOL,
            "report_sha256": _Qualification.report_sha256,
        }


def _valid_contract(
    *,
    cohort_root: Path,
    media_root: Path,
    source: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol": v04.PROTOCOL,
        "cohort_id": "v0.4-ineffective-trace-stage-a-20260724",
        "source": source,
        "roots": {
            "cohort": str(cohort_root),
            "media": str(media_root),
        },
        "arms": [
            {
                "id": mode.value,
                "directory": mode.value,
                "media_directory": mode.value,
            }
            for mode in v04.ARM_ORDER
        ],
        "parent": {
            "checkpoint_sha256": v04.PARENT_CHECKPOINT_SHA256,
            "policy_tensor_sha256": v04.PARENT_POLICY_TENSOR_SHA256,
            "optimizer_state_sha256": v04.PARENT_OPTIMIZER_STATE_SHA256,
        },
        "qualification": _Qualification.public_dict(),
        "matched_design": {
            "arm_order": [mode.value for mode in v04.ARM_ORDER],
            "first_rollout_digest_profile": (
                v04.FIRST_ROLLOUT_DIGEST_PROFILE
            ),
            "action_cap_per_arm": v04.CHILD_ACTION_BUDGET,
            "evaluation_every": v04.EVALUATION_INTERVAL,
            "fresh_only": True,
            "resumable": False,
            "checkpoint_promotable": False,
            "r3_state_reuse_authorized": False,
        },
    }


def test_cohort_contract_binds_fresh_v04_arms_and_rejects_r3_reuse(
    tmp_path: Path,
) -> None:
    cohort_root = tmp_path / "cohort"
    media_root = tmp_path / "media"
    cohort_root.mkdir()
    media_root.mkdir()
    for mode in v04.ARM_ORDER:
        (cohort_root / mode.value).mkdir()
        (media_root / mode.value).mkdir()
    source = {"commit": "a" * 40, "dirty": False}
    contract_path = cohort_root / "cohort-contract.json"
    contract = _valid_contract(
        cohort_root=cohort_root,
        media_root=media_root,
        source=source,
    )
    atomic_write_json(contract_path, contract)

    verified, digest, actual_cohort, actual_media = (
        trainer.verify_cohort_contract(
            path=contract_path,
            arm=IneffectiveTraceMode.ZERO_TRACE,
            run_directory=cohort_root
            / IneffectiveTraceMode.ZERO_TRACE.value,
            media_directory=media_root
            / IneffectiveTraceMode.ZERO_TRACE.value,
            expected_cohort_root=cohort_root,
            expected_media_root=media_root,
            source=source,
            qualification=_Qualification(),
        )
    )
    assert verified == contract
    assert len(digest) == 64
    assert actual_cohort == cohort_root
    assert actual_media == media_root

    contract["matched_design"]["r3_state_reuse_authorized"] = True
    atomic_write_json(contract_path, contract)
    with pytest.raises(
        trainer.IneffectiveTraceTrainingError,
        match="cohort contract changed",
    ):
        trainer.verify_cohort_contract(
            path=contract_path,
            arm=IneffectiveTraceMode.ZERO_TRACE,
            run_directory=cohort_root
            / IneffectiveTraceMode.ZERO_TRACE.value,
            media_directory=media_root
            / IneffectiveTraceMode.ZERO_TRACE.value,
            expected_cohort_root=cohort_root,
            expected_media_root=media_root,
            source=source,
            qualification=_Qualification(),
        )


def test_trainer_uses_lazy_v04_qualification_api_and_no_legacy_switch() -> None:
    parameters = inspect.signature(
        trainer.verify_arm_terminal_report
    ).parameters
    assert "authenticate_frozen_r2_legacy_envelope" not in parameters
    parsed = trainer.build_parser().parse_args(
        [
            "train-arm",
            "--arm",
            IneffectiveTraceMode.STREAK_TRACE.value,
            "--qualification-report",
            "qualification.json",
            "--run-dir",
            "run",
            "--media-dir",
            "media",
            "--cohort-contract",
            "contract.json",
        ]
    )
    assert parsed.arm == IneffectiveTraceMode.STREAK_TRACE.value
    assert parsed.device == "cpu"


def test_only_candidate_can_be_architecture_eligible() -> None:
    assert not trainer.architecture_definition_eligible(
        IneffectiveTraceMode.ZERO_TRACE,
        gate_passed=True,
    )
    assert not trainer.architecture_definition_eligible(
        IneffectiveTraceMode.STREAK_TRACE,
        gate_passed=False,
    )
    assert trainer.architecture_definition_eligible(
        IneffectiveTraceMode.STREAK_TRACE,
        gate_passed=True,
    )


def test_latest_frame_binding_uses_newest_canonical_frame(
    tmp_path: Path,
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    latest = frames / "latest.png"
    exam = frames / "exam-unlock-u2-separated.png"
    latest.write_bytes(b"\x89PNG\r\n\x1a\nlatest")
    exam.write_bytes(b"\x89PNG\r\n\x1a\nexam")
    os.utime(latest, ns=(1_000, 1_000))
    os.utime(exam, ns=(2_000, 2_000))

    binding = trainer.latest_frame_binding(tmp_path)

    assert binding is not None
    assert binding["path"] == "frames/exam-unlock-u2-separated.png"
    assert binding["bytes"] == len(b"\x89PNG\r\n\x1a\nexam")
    assert len(binding["sha256"]) == 64


def test_latest_frame_binding_rejects_unsafe_canonical_frame(
    tmp_path: Path,
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    target = tmp_path / "outside.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\noutside")
    (frames / "latest.png").symlink_to(target)

    with pytest.raises(
        trainer.IneffectiveTraceTrainingError,
        match="unsafe",
    ):
        trainer.latest_frame_binding(tmp_path)
