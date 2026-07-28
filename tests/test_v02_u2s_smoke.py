from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v02_u2s
from dungeon_apprentice import v02_u2s_smoke as smoke
from dungeon_apprentice.u2_seed_guard import U2SeedRole


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _initial_rng_identity() -> dict[str, Any]:
    components: dict[str, Any] = {
        name: f"{index + 1:064x}"
        for index, name in enumerate(
            v02_u2s.INITIAL_RNG_COMPONENT_DIGESTS
        )
    }
    components["torch_mps_sha256"] = None
    components["torch_cuda_sha256"] = None
    components["workers"] = [
        {
            "worker_index": index,
            "worker_stream": stream,
            **{
                name: f"{20 + index * 3 + offset:064x}"
                for offset, name in enumerate(
                    v02_u2s.INITIAL_RNG_WORKER_COMPONENT_DIGESTS
                )
            },
        }
        for index, stream in enumerate(v02_u2s.WORKER_STREAMS)
    ]
    return {
        "captured_before_action_one": True,
        "algorithm_seed": v02_u2s.ALGORITHM_SEED,
        "components": components,
        "aggregate_sha256": v02_u2s._canonical_sha256(components),
    }


def test_protocol_is_one_real_production_rollout_per_arm() -> None:
    assert smoke.TOTAL_TRANSITIONS == 2_048
    assert smoke.TOTAL_TRANSITIONS == v02_u2s.WORKERS * v02_u2s.ROLLOUT_STEPS
    assert [arm.value for arm in v02_u2s.ARM_PRIORITY] == [
        "control",
        "conservative",
        "no-effect",
        "combined",
    ]
    assert vars(smoke.build_parser().parse_args([])) == {}
    with pytest.raises(SystemExit):
        smoke.build_parser().parse_args(["--actions", "4096"])


class _EvidenceEnv(gym.Env[np.ndarray, int]):
    def __init__(self) -> None:
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(2, 2, 3),
            dtype=np.uint8,
        )
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del seed, options
        self.steps = 0
        return np.zeros((2, 2, 3), dtype=np.uint8), {
            "seed": 123,
            "lesson_id": lessons.LessonId.SEPARATED_UNLOCK.value,
            "u2_seed_role": U2SeedRole.TRAINING.value,
            "layout_sha256": "a" * 64,
            "geometry_sha256": "b" * 64,
        }

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del action
        self.steps += 1
        return (
            np.full((2, 2, 3), self.steps, dtype=np.uint8),
            0.001,
            self.steps == 2,
            False,
            {
                "lesson_id": lessons.LessonId.SEPARATED_UNLOCK.value,
                "extrinsic_reward": -0.001,
                "curiosity_reward": 0.002,
                "u2s_no_effect_penalty": 0.0,
                "u2s_no_effect_eligible_event_count": 0,
                "u2s_identical_interaction_run": 0,
                "u2s_no_effect_applied_penalty_count": 0,
            },
        )


def test_transition_evidence_reconciles_real_reward_components() -> None:
    wrapped = smoke.TransitionEvidenceWrapper(_EvidenceEnv(), worker_index=0)
    wrapped.reset()
    wrapped.step(0)
    wrapped.step(1)
    evidence = wrapped.public_dict()
    assert evidence["transitions"] == 2
    assert evidence["episodes_started"] == 1
    assert evidence["action_counts"]["0"] == 1
    assert evidence["action_counts"]["1"] == 1
    assert evidence["reward_total_hex"] == (0.002).hex()
    assert evidence["extrinsic_total_hex"] == (-0.002).hex()
    assert evidence["curiosity_total_hex"] == (0.004).hex()
    assert evidence["penalty_events"] == 0
    assert len(evidence["trajectory_sha256"]) == 64


def test_seed_evidence_rejects_protected_and_accepts_training() -> None:
    accepted = smoke._validate_seed_evidence(
        [
            {
                "seed": 55,
                "seed_role": U2SeedRole.TRAINING.value,
                "lesson_id": lessons.LessonId.SEPARATED_UNLOCK.value,
            }
        ]
    )
    assert accepted["training_range_only"] is True
    assert accepted["protected_seed_hits"] == []
    with pytest.raises(smoke.U2SSmokeError, match="non-training"):
        smoke._validate_seed_evidence(
            [
                {
                    "seed": 15_240_000,
                    "seed_role": U2SeedRole.U2R_U2_CONFIRMATION.value,
                    "lesson_id": lessons.LessonId.SEPARATED_UNLOCK.value,
                }
            ]
        )


def test_single_rollout_limiter_stops_only_after_one_completed_training_phase() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.collections = 0
            self.training_phases = 0

        def collect_rollouts(self, *_args: Any, **_kwargs: Any) -> bool:
            self.collections += 1
            return True

        def learn(self) -> None:
            while self.collect_rollouts():
                self.training_phases += 1

    model = FakeModel()
    count, restore = smoke._install_single_rollout_limit(model)
    model.learn()
    assert count() == 1
    assert model.collections == 1
    assert model.training_phases == 1
    restore()
    assert model.collect_rollouts() is True
    assert model.collections == 2


def test_run_smoke_uses_and_removes_one_disposable_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_roots = (
        tmp_path / "canonical-qualification",
        tmp_path / "canonical-cohort",
        tmp_path / "canonical-media",
    )
    parent = tmp_path / "parent.zip"
    parent.write_bytes(b"exact confirmed parent")
    parent_sha256 = _sha(parent.read_bytes())
    monkeypatch.setattr(smoke, "CANONICAL_ROOTS", canonical_roots)
    monkeypatch.setattr(v02_u2s, "PARENT_CHECKPOINT", parent)
    monkeypatch.setattr(
        v02_u2s,
        "PARENT_CHECKPOINT_SHA256",
        parent_sha256,
    )
    monkeypatch.setattr(
        smoke,
        "_require_clean_source",
        lambda _repository: {"commit": "a" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        smoke,
        "_load_protocol_inputs",
        lambda _repository: smoke._ProtocolInputs(
            parent=SimpleNamespace(),
            seed_access=object(),
            forbidden_layout_hashes={},
            guard_mapping_sha256="b" * 64,
            sampler_preflight_sha256="c" * 64,
        ),
    )
    disposable_paths: list[Path] = []

    def fake_arm(
        arm: v02_u2s.ArmName,
        *,
        arm_directory: Path,
        inputs: smoke._ProtocolInputs,
    ) -> dict[str, Any]:
        del inputs
        arm_directory.mkdir()
        (arm_directory / "changed-policy.zip").write_bytes(arm.value.encode())
        disposable_paths.append(arm_directory)
        return {
            "arm": arm.value,
            "trajectory_sha256": "d" * 64,
            "episode_ledger": {"normalized_sha256": "e" * 64},
            "before": {"policy": "same", "optimizer": "same"},
            "initial_rng_identity": _initial_rng_identity(),
            "updated_only_in_disposable_copy": True,
        }

    monkeypatch.setattr(smoke, "_run_arm", fake_arm)
    report = smoke.run_smoke(repository=tmp_path)
    assert [value["arm"] for value in report["arms"]] == [
        arm.value for arm in v02_u2s.ARM_PRIORITY
    ]
    assert report["canonical_or_claim_policy_updates"] is False
    assert report["engineering_smoke_copy_updated"] is True
    assert report["integration"]["matched_initial_rng_identity"] is True
    assert report["disposable_artifacts_removed"] is True
    assert report["canonical_boundaries_before"] == report[
        "canonical_boundaries_after"
    ]
    assert all(not path.exists() for path in disposable_paths)
    assert parent.read_bytes() == b"exact confirmed parent"


def test_cli_emits_one_canonical_json_document(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "schema_version": 1,
        "result": "passed",
        "canonical_or_claim_policy_updates": False,
        "engineering_smoke_copy_updated": True,
    }
    monkeypatch.setattr(smoke, "run_smoke", lambda: report)
    monkeypatch.setattr(sys, "argv", ["dungeon-smoke-v02-u2s"])
    smoke.main()
    output = capsys.readouterr().out
    assert json.loads(output) == report
    assert output == smoke._canonical_json_bytes(report).decode("utf-8")
