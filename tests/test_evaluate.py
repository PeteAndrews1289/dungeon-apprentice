import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dungeon_apprentice.artifacts import file_sha256
from dungeon_apprentice.contracts import (
    CHECKPOINT_SCHEMA_VERSION,
    PIXEL_SHAPE,
    PROTOCOL,
    DungeonTier,
)
from dungeon_apprentice.evaluate import (
    _evaluation_size,
    _load_checkpoint_bundle,
    build_parser,
    evaluate_policy,
)


def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"policy")
    sidecar = checkpoint.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "checkpoint_sha256": file_sha256(checkpoint),
                "effective_config": {"environment": {"size": 9}},
                "progress": {
                    "collected_timesteps": 64,
                    "trained_timesteps": 64,
                },
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, sidecar


def test_evaluation_requires_matching_checkpoint_bundle(tmp_path: Path) -> None:
    checkpoint, sidecar = _bundle(tmp_path)
    archive, metadata, digest = _load_checkpoint_bundle(checkpoint)

    assert archive == checkpoint.resolve()
    assert metadata["progress"]["trained_timesteps"] == 64
    assert digest == file_sha256(checkpoint)

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="hash does not match"):
        _load_checkpoint_bundle(checkpoint)

    checkpoint.write_bytes(b"policy")
    sidecar.unlink()
    with pytest.raises(SystemExit, match="requires the checkpoint sidecar"):
        _load_checkpoint_bundle(checkpoint)


def test_evaluation_reports_partial_progress_without_privileged_observations() -> None:
    class WaitPolicy:
        observation_space = SimpleNamespace(shape=PIXEL_SHAPE)

        @staticmethod
        def predict(
            _observation: np.ndarray,
            *,
            state: object,
            episode_start: np.ndarray,
            deterministic: bool,
        ) -> tuple[np.ndarray, object]:
            del episode_start, deterministic
            return np.asarray(6), state

    result = evaluate_policy(WaitPolicy(), DungeonTier.NAVIGATE, [10_000_000])

    assert result.success_rate == 0.0
    assert result.milestone_rates["success"] == 0.0
    assert result.mean_unique_cells == 1.0
    assert result.mean_collisions == 0.0
    assert result.mean_ineffective_interactions == 0.0


def test_evaluation_size_comes_from_bundle_and_overrides_are_explicit() -> None:
    metadata = {"effective_config": {"environment": {"size": 11}}}
    parser = build_parser()

    derived = parser.parse_args(["model.zip"])
    assert _evaluation_size(derived, metadata) == 11

    mismatch = parser.parse_args(["model.zip", "--size", "9"])
    with pytest.raises(SystemExit, match="allow-size-override"):
        _evaluation_size(mismatch, metadata)

    declared = parser.parse_args(
        ["model.zip", "--size", "9", "--allow-size-override"]
    )
    assert _evaluation_size(declared, metadata) == 9
