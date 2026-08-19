import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dungeon_apprentice import train
from dungeon_apprentice.artifacts import atomic_write_json, file_sha256
from dungeon_apprentice.contracts import PROTOCOL
from dungeon_apprentice.env import CurriculumState
from dungeon_apprentice.evaluate import TierEvaluation
from dungeon_apprentice.training import callbacks as training_callbacks


class _FakeBaseCallback:
    def __init__(self, verbose: int = 0) -> None:
        self.verbose = verbose
        self.model: Any = None
        self.locals: dict[str, Any] = {}

    @property
    def num_timesteps(self) -> int:
        return int(self.model.num_timesteps)


class _FakeModel:
    def __init__(self) -> None:
        self.num_timesteps = 0
        self._n_updates = 0
        self.n_envs = 1
        self.logger = SimpleNamespace(name_to_value={"train/loss": 0.25})
        self.save_calls = 0

    def save(self, destination: Path) -> None:
        self.save_calls += 1
        Path(destination).write_bytes(
            f"steps={self.num_timesteps},updates={self._n_updates}".encode()
        )


def _callback(tmp_path: Path) -> Any:
    callback_type = train._CallbackFactory.create(_FakeBaseCallback)
    segment = {
        "id": "segment-0",
        "index": 0,
        "started_at": "2026-07-22T00:00:00+00:00",
        "run_directory": str(tmp_path),
        "start_collected_timesteps": 0,
        "start_trained_timesteps": 0,
        "worker_seed_base": 1,
        "algorithm_seed": 1,
        "parent": None,
    }
    callback = callback_type(
        run_directory=tmp_path,
        curriculum=CurriculumState(),
        size=9,
        evaluation_interval=2,
        evaluation_seed_count=1,
        checkpoint_interval=2,
        frame_interval=1_000,
        stop_when_mastered=True,
        started_at=segment["started_at"],
        effective_config={"test": True},
        segment=segment,
        initial_trained_timesteps=0,
        initial_mastered=False,
        keep_checkpoints=2,
        minimum_free_bytes=0,
    )
    callback.model = _FakeModel()
    return callback


def _failed_evaluation(_model: Any, tier: Any, _seeds: Any, **_kwargs: Any) -> TierEvaluation:
    return TierEvaluation(
        protocol=PROTOCOL,
        timestamp="2026-07-22T00:00:01+00:00",
        tier=int(tier),
        tier_label=tier.label,
        final_suite=False,
        episodes=1,
        successes=0,
        success_rate=0.0,
        mean_steps=1.0,
        mean_reward=-0.001,
        results=(),
    )


def _passed_evaluation(_model: Any, tier: Any, _seeds: Any, **_kwargs: Any) -> TierEvaluation:
    return TierEvaluation(
        protocol=PROTOCOL,
        timestamp="2026-07-22T00:00:01+00:00",
        tier=int(tier),
        tier_label=tier.label,
        final_suite=False,
        episodes=1,
        successes=1,
        success_rate=1.0,
        mean_steps=1.0,
        mean_reward=1.0,
        results=(),
    )


def test_checkpoint_and_exam_wait_for_optimizer_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training_callbacks, "evaluate_policy", _failed_evaluation)
    callback = _callback(tmp_path)
    callback._on_training_start()
    assert callback.model.save_calls == 1

    callback.model.num_timesteps = 2
    callback.locals = {
        "infos": [
            {
                "tier": 0,
                "tier_label": "Navigate",
                "seed": 17,
                "layout_sha256": "abc",
                "success": False,
                "terminal_reason": "time_limit",
                "extrinsic_reward": -0.001,
                "curiosity_reward": 0.051,
                "curiosity_unique_observations": 8,
                "curiosity_budget_used": 0.051,
                "milestones": {"success": False},
                "unique_cells": 4,
                "collisions": 2,
                "ineffective_interactions": 1,
            }
        ],
        "dones": [True],
        "rewards": [0.05],
    }
    assert callback._on_step()
    assert not (tmp_path / "checkpoints" / "step-000000000002.zip").exists()
    assert not (tmp_path / "evaluations.jsonl").exists()

    callback.model._n_updates = 4
    callback._on_rollout_start()

    step = tmp_path / "checkpoints" / "step-000000000002.zip"
    assert step.exists()
    assert (tmp_path / "evaluations.jsonl").exists()
    sidecar = json.loads(step.with_suffix(".json").read_text())
    assert sidecar["progress"] == {
        "collected_timesteps": 2,
        "n_updates": 4,
        "segment_collected_timesteps": 2,
        "segment_trained_timesteps": 2,
        "trained_timesteps": 2,
    }
    assert sidecar["checkpoint_sha256"] == file_sha256(step)

    episode = json.loads((tmp_path / "episodes.jsonl").read_text().splitlines()[0])
    assert episode["combined_return"] == pytest.approx(0.05)
    assert episode["extrinsic_return"] == pytest.approx(-0.001)
    assert episode["curiosity_return"] == pytest.approx(0.051)
    assert episode["milestones"] == {"success": False}
    assert episode["unique_cells"] == 4
    assert episode["collisions"] == 2
    assert episode["ineffective_interactions"] == 1
    optimizer = json.loads((tmp_path / "optimizer.jsonl").read_text().splitlines()[0])
    assert optimizer["trained_timesteps"] == 2
    assert optimizer["metrics"]["train/loss"] == 0.25

    callback.finalize_success("completed")
    assert (tmp_path / "checkpoints" / "completed.zip").exists()
    assert callback.model.save_calls == 2  # initial + step; final/latest reuse bytes
    final_records = [
        json.loads(line)
        for line in (tmp_path / "evaluations.jsonl").read_text().splitlines()
    ]
    assert [record["trigger"] for record in final_records] == ["scheduled", "final"]
    assert all(
        record["checkpoint_sha256"] == sidecar["checkpoint_sha256"]
        for record in final_records
    )
    assert all(
        record["checkpoint"] == "checkpoints/step-000000000002.zip"
        for record in final_records
    )


def test_partial_rollout_is_never_saved_as_a_success(tmp_path: Path) -> None:
    callback = _callback(tmp_path)
    callback._on_training_start()
    callback.model.num_timesteps = 1

    with pytest.raises(RuntimeError, match="untrained partial rollout"):
        callback.finalize_success("completed")

    callback.finalize_failure("interrupted")
    assert callback.model.save_calls == 1
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["discarded_partial_timesteps"] == 1
    assert status["latest_safe_checkpoint"] == "checkpoints/initial.zip"


def test_policy_cannot_cross_two_curriculum_gates_without_another_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training_callbacks, "evaluate_policy", _passed_evaluation)
    callback = _callback(tmp_path)
    callback._on_training_start()
    callback.model.num_timesteps = 2
    callback.model._n_updates = 1

    callback._on_rollout_start()
    assert callback.curriculum.max_tier.value == 1

    callback.finalize_success("completed")
    assert callback.curriculum.max_tier.value == 1
    decisions = [
        json.loads(line)["decision"]
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if json.loads(line).get("type") == "curriculum_decision"
    ]
    assert decisions == [
        "Promoted to Unlock",
        "Held: one curriculum transition per trained boundary",
    ]


def test_failed_promotion_checkpoint_rolls_back_live_curriculum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training_callbacks, "evaluate_policy", _passed_evaluation)
    callback = _callback(tmp_path)
    callback._on_training_start()
    callback.model.num_timesteps = 2
    callback.model._n_updates = 1
    original_checkpoint = callback._checkpoint

    def fail_promotion(name: str, *, kind: str) -> Path:
        if kind == "promotion":
            raise OSError("simulated publication failure")
        return original_checkpoint(name, kind=kind)

    monkeypatch.setattr(callback, "_checkpoint", fail_promotion)
    with pytest.raises(OSError, match="publication failure"):
        callback._on_rollout_start()

    assert callback.curriculum.max_tier.value == 0
    assert callback._last_curriculum_transition_at is None


def test_failed_final_tier_exam_clears_current_mastery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training_callbacks, "evaluate_policy", _failed_evaluation)
    callback = _callback(tmp_path)
    callback.curriculum.max_tier = train.DungeonTier.RETRIEVE
    callback.mastered = True
    callback._on_training_start()

    callback._evaluate_and_maybe_promote(trigger="scheduled")

    assert callback.mastered is False


def test_numbered_checkpoint_retention_keeps_only_newest(tmp_path: Path) -> None:
    callback = _callback(tmp_path)
    callback._on_training_start()

    for step in (2, 4, 6):
        callback.model.num_timesteps = step
        callback.model._n_updates += 1
        callback.trained_timesteps = step
        callback._checkpoint(f"step-{step:012d}", kind="step")

    numbered = sorted(path.name for path in (tmp_path / "checkpoints").glob("step-*.zip"))
    assert numbered == ["step-000000000004.zip", "step-000000000006.zip"]
    assert (tmp_path / "checkpoints" / "initial.zip").exists()
    assert (tmp_path / "checkpoints" / "latest.zip").exists()


def test_resume_requires_hash_protocol_config_and_trained_boundary(tmp_path: Path) -> None:
    args = train.build_parser().parse_args([])
    config = train._requested_effective_config(args)
    checkpoint = tmp_path / "latest.zip"
    checkpoint.write_bytes(b"safe model")
    sidecar = {
        "schema_version": train.CHECKPOINT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "checkpoint_sha256": file_sha256(checkpoint),
        "effective_config": config,
        "curriculum": {"max_tier": 1},
        "progress": {
            "collected_timesteps": 100,
            "trained_timesteps": 100,
            "n_updates": 4,
        },
        "segment": {"id": "parent", "index": 0},
    }
    atomic_write_json(checkpoint.with_suffix(".json"), sidecar)

    resolved, loaded = train._load_resume_metadata(checkpoint, config)
    assert resolved == checkpoint.resolve()
    assert loaded["curriculum"]["max_tier"] == 1

    incompatible = dict(config)
    incompatible["seed"] = config["seed"] + 1
    with pytest.raises(SystemExit, match="configuration is incompatible"):
        train._load_resume_metadata(checkpoint, incompatible)

    sidecar["protocol"] = "different-protocol"
    atomic_write_json(checkpoint.with_suffix(".json"), sidecar)
    with pytest.raises(SystemExit, match="protocol mismatch"):
        train._load_resume_metadata(checkpoint, config)

    sidecar["protocol"] = PROTOCOL
    sidecar["progress"]["collected_timesteps"] = 101
    atomic_write_json(checkpoint.with_suffix(".json"), sidecar)
    with pytest.raises(SystemExit, match="not a fully trained boundary"):
        train._load_resume_metadata(checkpoint, config)


def test_segment_seed_is_reactivated_after_model_load() -> None:
    class SeedRecorder:
        def __init__(self) -> None:
            self.seeds: list[int] = []

        def set_random_seed(self, seed: int) -> None:
            self.seeds.append(seed)

    model = SeedRecorder()
    train._activate_segment_seed(model, 22_345)

    assert model.seeds == [22_345]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--evaluation-seeds", "100001"], "cannot exceed"),
        (["--size", "6"], "at least 7"),
        (["--learning-rate", "nan"], "finite and positive"),
        (["--curiosity-scale", "0.003"], "freezes curiosity scale"),
        (["--gamma", "0.99"], "discounted reward dominance"),
    ],
)
def test_invalid_run_parameters_fail_during_preflight(
    arguments: list[str], message: str
) -> None:
    args = train.build_parser().parse_args(arguments)
    with pytest.raises(SystemExit, match=message):
        train._validate_args(args)
