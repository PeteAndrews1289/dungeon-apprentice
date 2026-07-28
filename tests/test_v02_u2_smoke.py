from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from dungeon_apprentice import v02_u2_smoke as smoke
from dungeon_apprentice.u2_seed_guard import (
    U2SeedAccessError,
    U2SeedRole,
    classify_u2_seed,
    engineering_seed_access,
)

sb3_contrib = pytest.importorskip("sb3_contrib")
vec_env = pytest.importorskip("stable_baselines3.common.vec_env")


class _AttestingEnv(gym.Env[np.ndarray, int]):
    observation_space = gym.spaces.Box(0, 255, shape=(56, 56, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(7)

    def __init__(self) -> None:
        self.resets: list[dict[str, Any]] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.resets.append({"seed": seed, "options": options})
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {
            "u2_seed_role": U2SeedRole.ENGINEERING.value,
            "lesson_id": smoke.lessons.U2LessonId.SEPARATED_UNLOCK.value,
            "layout_sha256": "a" * 64,
            "geometry_sha256": "b" * 64,
        }

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return (
            np.zeros(self.observation_space.shape, dtype=np.uint8),
            0.0,
            False,
            False,
            {},
        )


def _fingerprint(
    *,
    timesteps: int,
    updates: int,
    digest: str,
    optimizer_digest: str = "0" * 64,
    optimizer: bool = True,
) -> smoke.ModelFingerprint:
    return smoke.ModelFingerprint(
        trained_timesteps=timesteps,
        optimizer_updates=updates,
        policy_tensor_sha256=digest,
        optimizer_state_sha256=optimizer_digest,
        optimizer_has_state=optimizer,
    )


def test_protocol_has_one_fixed_engineering_rollout_and_no_seed_controls() -> None:
    assert smoke.WORKERS == 4
    assert smoke.ROLLOUT_STEPS == 16
    assert smoke.TOTAL_TRANSITIONS == 64
    assert smoke.BATCH_SIZE == 64
    assert smoke.N_EPOCHS == 1
    assert smoke.LEAD_CHILD_SEED == 20260737
    assert smoke.MINIMUM_FREE_GIB == 25.0

    args = smoke.build_parser().parse_args(["--run-name", "engineering-smoke"])
    assert args.run_name == "engineering-smoke"
    with pytest.raises(SystemExit):
        smoke.build_parser().parse_args(
            [
                "--run-name",
                "engineering-smoke",
                "--seed",
                "5210000",
            ]
        )


def test_forced_environment_uses_only_its_fixed_engineering_slice() -> None:
    base = _AttestingEnv()
    audit: list[dict[str, Any]] = []
    access = engineering_seed_access()
    env = smoke.ForcedSeparatedEngineeringEnv(
        base,
        worker_index=2,
        access=access,
        seed_audit=audit,
    )

    for _ in range(3):
        env.reset()

    expected = [
        smoke.lessons.ENGINEERING_SEED_BASE + 2,
        smoke.lessons.ENGINEERING_SEED_BASE + 6,
        smoke.lessons.ENGINEERING_SEED_BASE + 10,
    ]
    assert [item["seed"] for item in base.resets] == expected
    assert [item["seed"] for item in audit] == expected
    assert all(classify_u2_seed(seed) is U2SeedRole.ENGINEERING for seed in expected)
    assert all(
        item["options"]["u2_seed_role"] == U2SeedRole.ENGINEERING.value
        for item in base.resets
    )
    assert all(item["options"]["u2_seed_access"] is access for item in base.resets)


def test_protected_seed_override_is_ignored_and_cannot_generate_protected_layout() -> None:
    base = _AttestingEnv()
    audit: list[dict[str, Any]] = []
    env = smoke.ForcedSeparatedEngineeringEnv(
        base,
        worker_index=0,
        access=engineering_seed_access(),
        seed_audit=audit,
    )

    env.reset(seed=5_210_000)
    assert base.resets[0]["seed"] == smoke.lessons.ENGINEERING_SEED_BASE
    assert audit[0]["requested_seed_ignored"] == 5_210_000
    assert classify_u2_seed(base.resets[0]["seed"]) is U2SeedRole.ENGINEERING
    with pytest.raises(U2SeedAccessError, match="options"):
        env.reset(options={"u2_seed_role": U2SeedRole.SEALED_QUALIFICATION.value})

    assert len(base.resets) == 1


def test_all_four_mock_workers_generate_no_protected_seed() -> None:
    access = engineering_seed_access()
    audit: list[dict[str, Any]] = []
    bases: list[_AttestingEnv] = []
    for worker_index in range(smoke.WORKERS):
        base = _AttestingEnv()
        bases.append(base)
        env = smoke.ForcedSeparatedEngineeringEnv(
            base,
            worker_index=worker_index,
            access=access,
            seed_audit=audit,
        )
        for _ in range(5):
            env.reset()

    generated = [int(item["seed"]) for base in bases for item in base.resets]
    assert len(generated) == 20
    assert min(generated) == smoke.lessons.ENGINEERING_SEED_BASE
    assert max(generated) < (
        smoke.lessons.ENGINEERING_SEED_BASE
        + smoke.lessons.ENGINEERING_SEED_COUNT
    )
    assert {classify_u2_seed(seed) for seed in generated} == {
        U2SeedRole.ENGINEERING
    }
    smoke.validate_seed_audit(audit, access=access)


def test_seed_audit_rejects_a_mock_protected_episode() -> None:
    access = engineering_seed_access()
    audit = [
        {
            "worker_index": worker,
            "seed": smoke.lessons.ENGINEERING_SEED_BASE + worker,
            "seed_role": U2SeedRole.ENGINEERING.value,
            "lesson_id": smoke.lessons.U2LessonId.SEPARATED_UNLOCK.value,
        }
        for worker in range(smoke.WORKERS)
    ]
    audit[3]["seed"] = 11_200_000
    audit[3]["seed_role"] = U2SeedRole.U2_VALIDATION.value

    with pytest.raises(U2SeedAccessError):
        smoke.validate_seed_audit(audit, access=access)


def test_optimizer_transition_requires_timesteps_updates_changed_parameters_and_reload() -> None:
    before = _fingerprint(timesteps=1_000, updates=10, digest="a" * 64)
    after = _fingerprint(
        timesteps=1_064,
        updates=11,
        digest="b" * 64,
        optimizer_digest="1" * 64,
    )
    smoke.verify_optimizer_transition(before, after, after)

    with pytest.raises(smoke.U2EngineeringSmokeError, match="parameters"):
        smoke.verify_optimizer_transition(
            before,
            _fingerprint(
                timesteps=1_064,
                updates=11,
                digest="a" * 64,
                optimizer_digest="1" * 64,
            ),
            _fingerprint(
                timesteps=1_064,
                updates=11,
                digest="a" * 64,
                optimizer_digest="1" * 64,
            ),
        )
    with pytest.raises(smoke.U2EngineeringSmokeError, match="64-transition"):
        smoke.verify_optimizer_transition(
            before,
            _fingerprint(
                timesteps=1_063,
                updates=11,
                digest="b" * 64,
                optimizer_digest="1" * 64,
            ),
            _fingerprint(
                timesteps=1_063,
                updates=11,
                digest="b" * 64,
                optimizer_digest="1" * 64,
            ),
        )
    with pytest.raises(smoke.U2EngineeringSmokeError, match="single frozen epoch"):
        smoke.verify_optimizer_transition(
            before,
            _fingerprint(
                timesteps=1_064,
                updates=12,
                digest="b" * 64,
                optimizer_digest="1" * 64,
            ),
            _fingerprint(
                timesteps=1_064,
                updates=12,
                digest="b" * 64,
                optimizer_digest="1" * 64,
            ),
        )
    with pytest.raises(smoke.U2EngineeringSmokeError, match="reloaded"):
        smoke.verify_optimizer_transition(
            before,
            after,
            _fingerprint(
                timesteps=1_064,
                updates=11,
                digest="c" * 64,
                optimizer_digest="1" * 64,
            ),
        )
    with pytest.raises(smoke.U2EngineeringSmokeError, match="reloaded"):
        smoke.verify_optimizer_transition(
            before,
            after,
            _fingerprint(
                timesteps=1_064,
                updates=11,
                digest="b" * 64,
                optimizer_digest="2" * 64,
            ),
        )

    with pytest.raises(smoke.U2EngineeringSmokeError, match="optimizer state"):
        unchanged_optimizer = _fingerprint(
            timesteps=1_064,
            updates=11,
            digest="b" * 64,
        )
        smoke.verify_optimizer_transition(
            before,
            unchanged_optimizer,
            unchanged_optimizer,
        )


def test_parent_architecture_check_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _fingerprint(timesteps=900, updates=12, digest="a" * 64)
    monkeypatch.setattr(smoke, "model_fingerprint", lambda _model: expected)
    model = SimpleNamespace(
        action_space=SimpleNamespace(n=7),
        observation_space=SimpleNamespace(shape=(3, 56, 56)),
        policy=SimpleNamespace(
            lstm_actor=SimpleNamespace(hidden_size=256, num_layers=1)
        ),
    )
    parent = SimpleNamespace(trained_timesteps=900, n_updates=12)

    assert smoke.verify_parent_architecture(model, parent=parent) == expected
    model.policy.lstm_actor.hidden_size = 128
    with pytest.raises(smoke.U2EngineeringSmokeError, match="recurrent"):
        smoke.verify_parent_architecture(model, parent=parent)


def test_output_directory_is_exclusive_fixed_and_checksummed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "DungeonApprentice" / "u2-engineering-smokes"
    monkeypatch.setattr(smoke, "STORAGE_ROOT", tmp_path)
    monkeypatch.setattr(smoke, "RUN_ROOT", run_root)
    monkeypatch.setattr(smoke, "ensure_disk_space", lambda *_args: 100)

    output = smoke._prepare_output("fixed-smoke", protected_paths=())
    assert output == run_root / "fixed-smoke"
    assert output.is_dir()
    with pytest.raises(FileExistsError):
        smoke._prepare_output("fixed-smoke", protected_paths=())
    with pytest.raises(smoke.U2EngineeringSmokeError, match="one new directory"):
        smoke._prepare_output("../escape", protected_paths=())

    manifest = output / "manifest.json"
    report = output / "report.json"
    manifest.write_text("manifest", encoding="utf-8")
    report.write_text("report", encoding="utf-8")
    checksums = smoke._publish_checksums(
        output,
        ("manifest.json", "report.json"),
    )
    assert set(checksums) == {"manifest.json", "report.json"}
    assert (output / "manifest.json.sha256").is_file()
    assert (output / "report.json.sha256").is_file()
    assert (output / "SHA256SUMS").read_text(encoding="utf-8").count("\n") == 2


def test_keyboard_interrupt_writes_failure_report_then_reraises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "lead.zip"
    archive.write_bytes(b"lead-parent")
    selection = SimpleNamespace(
        archive=archive,
        archive_sha256=smoke.file_sha256(archive),
    )
    parent = SimpleNamespace(
        sidecar=str(tmp_path / "parent.json"),
        manifest=str(tmp_path / "parent-manifest.json"),
        confirmation_report=str(tmp_path / "confirmation.json"),
        attempt_ledger=str(tmp_path / "attempt.json"),
        checksum_file=str(tmp_path / "confirmation.json.sha256"),
        public_dict=lambda: {"checkpoint": str(archive)},
    )
    source = {
        "commit": "d" * 40,
        "branch": "agent/v02-u2-separated",
        "dirty": False,
    }
    output = tmp_path / "smoke-output"
    output.mkdir()
    closed = {"value": False}

    class _FakeVectorEnvironment:
        def close(self) -> None:
            closed["value"] = True

    class _InterruptedModel:
        def set_random_seed(self, seed: int) -> None:
            assert seed == smoke.ALGORITHM_SEED

        def learn(self, **_kwargs: Any) -> None:
            raise KeyboardInterrupt

    class _FakeRecurrentPPO:
        @classmethod
        def load(cls, *_args: Any, **_kwargs: Any) -> _InterruptedModel:
            return _InterruptedModel()

    monkeypatch.setattr(
        smoke.v02_u2,
        "FROZEN_PARENTS",
        {smoke.LEAD_CHILD_SEED: selection},
    )
    monkeypatch.setattr(smoke.v02_u2, "verify_parent", lambda *_args, **_kwargs: parent)
    monkeypatch.setattr(smoke, "_require_clean_source", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(smoke, "_prepare_output", lambda *_args, **_kwargs: output)
    monkeypatch.setattr(
        smoke,
        "verify_parent_architecture",
        lambda *_args, **_kwargs: _fingerprint(
            timesteps=1_000,
            updates=10,
            digest="a" * 64,
        ),
    )
    monkeypatch.setattr(sb3_contrib, "RecurrentPPO", _FakeRecurrentPPO)
    monkeypatch.setattr(
        vec_env,
        "DummyVecEnv",
        lambda _factories: _FakeVectorEnvironment(),
    )
    monkeypatch.setattr(vec_env, "VecTransposeImage", lambda environment: environment)

    with pytest.raises(KeyboardInterrupt):
        smoke.run_smoke(run_name="interrupted")

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["result"] == "failed"
    assert report["error_type"] == "KeyboardInterrupt"
    assert report["capability_claim"] is False
    assert (output / "report.json.sha256").is_file()
    assert closed["value"] is True
