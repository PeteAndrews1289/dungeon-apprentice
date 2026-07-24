from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dungeon_apprentice import v04_ineffective_trace as v04
from dungeon_apprentice import v04_ineffective_trace_smoke as smoke
from dungeon_apprentice.action_streak import IneffectiveTraceMode
from dungeon_apprentice.artifacts import file_sha256


def _trace_exercise(mode: IneffectiveTraceMode) -> dict[str, object]:
    return {
        "transitions": v04.ROLLOUT_TRANSITIONS,
        "visible_changed": 1_000,
        "visible_unchanged": 1_048,
        "raw_trace_nonzero_transitions": 1_048,
        "raw_trace_max": 9,
        "raw_trace_histogram": {"0": 1_000, "1": 1_048},
        "exposed_trace_nonzero_transitions": (
            0 if mode is IneffectiveTraceMode.ZERO_TRACE else 1_048
        ),
        "exposed_trace_max": (
            0.0 if mode is IneffectiveTraceMode.ZERO_TRACE else 1 / 9
        ),
        "trace_enabled": mode is IneffectiveTraceMode.STREAK_TRACE,
        "descriptive_only": True,
    }


def _encoder_update(mode: IneffectiveTraceMode) -> dict[str, object]:
    candidate = mode is IneffectiveTraceMode.STREAK_TRACE
    return {
        "weight_nonzero_parameters": 512 if candidate else 0,
        "gradient_observations": 16,
        "gradient_nonzero_observations": 16 if candidate else 0,
        "gradient_nonzero_parameters_max": 512 if candidate else 0,
        "adam_state_present": True,
        "adam_exp_avg_nonzero_parameters": 512 if candidate else 0,
        "adam_exp_avg_sq_nonzero_parameters": 512 if candidate else 0,
        "weight_exact_zero": not candidate,
        "adam_moments_exact_zero": not candidate,
    }


def _arm_result(mode: IneffectiveTraceMode) -> dict[str, object]:
    return {
        "arm": mode.value,
        "before": {"policy": "identical", "optimizer": "identical"},
        "after": {"policy": mode.value},
        "trace_projection_nonzero_parameters": (
            0 if mode is IneffectiveTraceMode.ZERO_TRACE else 512
        ),
        "trace_exercise": _trace_exercise(mode),
        "encoder_update": _encoder_update(mode),
        "transplant": {"zero_context_equivalence": True},
        "workers": [],
        "trajectory_identity": [{"trajectory_sha256": "a" * 64}],
        "pre_action_rng_identity": {"aggregate_sha256": "b" * 64},
        "policy_output_sha256": "c" * 64,
        "post_rollout_rng_identity": {"aggregate_sha256": "d" * 64},
        "episode_ledger": {
            "records": 10,
            "normalized_sha256": "e" * 64,
        },
        "seed_evidence": {"training_range_only": True},
        "updated_archive_reloaded_exactly": True,
        "development_checkpoint_reuse_authorized": False,
    }


def test_trace_exercise_records_raw_and_exposed_values() -> None:
    candidate = smoke.TraceExerciseMetrics()
    candidate.observe(
        infos=(
            {
                "ineffective_trace_raw_count": 2,
                "ineffective_trace_visible_changed": False,
                "ineffective_trace_context_active": True,
            },
            {
                "ineffective_trace_raw_count": 0,
                "ineffective_trace_visible_changed": True,
                "ineffective_trace_context_active": True,
            },
        ),
        observations={
            "ineffective_trace": np.asarray(
                [[2 / 9], [0.0]],
                dtype=np.float32,
            )
        },
        mode=IneffectiveTraceMode.STREAK_TRACE,
    )
    assert candidate.public_dict(
        mode=IneffectiveTraceMode.STREAK_TRACE
    ) == {
        "transitions": 2,
        "visible_changed": 1,
        "visible_unchanged": 1,
        "raw_trace_nonzero_transitions": 1,
        "raw_trace_max": 2,
        "raw_trace_histogram": {"0": 1, "2": 1},
        "exposed_trace_nonzero_transitions": 1,
        "exposed_trace_max": pytest.approx(2 / 9),
        "trace_enabled": True,
        "descriptive_only": True,
    }

    sham = smoke.TraceExerciseMetrics()
    sham.observe(
        infos=(
            {
                "ineffective_trace_raw_count": 2,
                "ineffective_trace_visible_changed": False,
                "ineffective_trace_context_active": False,
                "terminal_observation": {
                    "ineffective_trace": np.asarray(
                        [0.0],
                        dtype=np.float32,
                    )
                },
            },
        ),
        observations={
            "ineffective_trace": np.asarray(
                [[0.0]],
                dtype=np.float32,
            )
        },
        mode=IneffectiveTraceMode.ZERO_TRACE,
    )
    assert sham.exposed_trace_nonzero_transitions == 0
    assert sham.raw_trace_nonzero_transitions == 1

    with pytest.raises(
        smoke.IneffectiveTraceSmokeError,
        match="frozen encoding",
    ):
        sham.observe(
            infos=(
                {
                    "ineffective_trace_raw_count": v04.TRACE_CAP + 1,
                    "ineffective_trace_visible_changed": False,
                    "ineffective_trace_context_active": False,
                },
            ),
            observations={
                "ineffective_trace": np.asarray(
                    [[0.0]],
                    dtype=np.float32,
                )
            },
            mode=IneffectiveTraceMode.ZERO_TRACE,
        )


@pytest.mark.parametrize(
    ("mode", "nonzero"),
    [
        (IneffectiveTraceMode.ZERO_TRACE, False),
        (IneffectiveTraceMode.STREAK_TRACE, True),
    ],
)
def test_encoder_update_reports_weight_gradient_and_adam_moments(
    mode: IneffectiveTraceMode,
    nonzero: bool,
) -> None:
    torch = pytest.importorskip("torch")
    weight = torch.nn.Parameter(torch.zeros((512, 1)))
    if nonzero:
        with torch.no_grad():
            weight.fill_(0.25)
    exp_avg = torch.zeros_like(weight)
    exp_avg_sq = torch.zeros_like(weight)
    if nonzero:
        exp_avg.fill_(0.1)
        exp_avg_sq.fill_(0.01)
    model = SimpleNamespace(
        policy=SimpleNamespace(
            features_extractor=SimpleNamespace(
                ineffective_trace_encoder=SimpleNamespace(weight=weight)
            ),
            optimizer=SimpleNamespace(
                state={
                    weight: {
                        "exp_avg": exp_avg,
                        "exp_avg_sq": exp_avg_sq,
                    }
                }
            ),
        )
    )
    gradients = smoke.TraceGradientEvidence(
        observations=4,
        nonzero_observations=4 if nonzero else 0,
        nonzero_parameters_max=512 if nonzero else 0,
    )

    evidence = smoke._trace_encoder_update_evidence(
        model,
        gradients=gradients,
    )
    assert evidence["weight_exact_zero"] is (not nonzero)
    assert evidence["adam_moments_exact_zero"] is (not nonzero)
    assert evidence["gradient_nonzero_observations"] == (
        4 if nonzero else 0
    )


def _prepare_mocked_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent.zip"
    parent.write_bytes(b"confirmed U1")
    monkeypatch.setattr(v04, "PARENT_CHECKPOINT", parent)
    monkeypatch.setattr(
        v04,
        "PARENT_CHECKPOINT_SHA256",
        file_sha256(parent),
    )
    monkeypatch.setattr(
        smoke,
        "git_snapshot",
        lambda _repository: {"commit": "a" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        smoke.frozen_smoke,
        "_load_protocol_inputs",
        lambda _repository: SimpleNamespace(
            forbidden_layout_hashes={
                lesson: frozenset() for lesson in smoke.lessons.LessonId
            },
            seed_access=object(),
        ),
    )
    monkeypatch.setattr(
        smoke.frozen_u2r,
        "preflight_u2r_training_layout_sampler",
        lambda *_args, **_kwargs: ({"stream": v04.WORKER_STREAMS[0]},),
    )
    monkeypatch.setattr(
        smoke.frozen_smoke,
        "_canonical_root_snapshots",
        lambda: [{"unchanged": True}],
    )


def test_run_smoke_seals_matched_trace_learning_and_destruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_mocked_smoke(tmp_path, monkeypatch)
    monkeypatch.setattr(
        smoke,
        "_run_arm",
        lambda mode, **_kwargs: _arm_result(mode),
    )

    result = smoke.run_smoke(repository=tmp_path)

    assert result["verdict"] == "passed"
    assert result["pre_update_behavior_identical"]
    assert result["raw_trace_evidence_identical"]
    assert result["updated_archives_reloaded_exactly"]
    assert result["updated_archives_destroyed"]
    assert result["temporary_root_removed"]
    assert all(
        arm["updated_archive_destroyed"]
        for arm in result["arms"].values()
    )


def test_run_smoke_rejects_raw_trace_divergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_mocked_smoke(tmp_path, monkeypatch)

    def divergent(
        mode: IneffectiveTraceMode,
        **_kwargs: object,
    ) -> dict[str, object]:
        result = _arm_result(mode)
        if mode is IneffectiveTraceMode.STREAK_TRACE:
            result["trace_exercise"]["raw_trace_max"] = 8
        return result

    monkeypatch.setattr(smoke, "_run_arm", divergent)
    with pytest.raises(
        smoke.IneffectiveTraceSmokeError,
        match="diverged before",
    ):
        smoke.run_smoke(repository=tmp_path)


def test_smoke_contract_uses_one_real_rollout_and_fixed_arm_order() -> None:
    assert smoke.TOTAL_TRANSITIONS == 2_048
    assert smoke.TOTAL_TRANSITIONS == v04.WORKERS * v04.ROLLOUT_STEPS
    assert v04.ARM_ORDER == (
        IneffectiveTraceMode.ZERO_TRACE,
        IneffectiveTraceMode.STREAK_TRACE,
    )
    assert vars(smoke.build_parser().parse_args([])) == {
        "repository": Path.cwd(),
        "allow_dirty": False,
    }
