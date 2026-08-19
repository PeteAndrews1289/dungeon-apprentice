"""Effective-configuration capture, resume metadata, and run paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dungeon_apprentice.artifacts import (
    file_sha256,
)
from dungeon_apprentice.contracts import (
    CHECKPOINT_SCHEMA_VERSION,
    CURIOSITY_BUDGET,
    NEWEST_TIER_FRACTION,
    PROMOTION_THRESHOLD,
    PROTOCOL,
    RETENTION_THRESHOLD,
    DungeonTier,
)

POLICY_KWARGS = {"lstm_hidden_size": 256, "n_lstm_layers": 1}




class _CurriculumMastered(RuntimeError):
    """Stop between rollouts, before another untrained action is collected."""


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


def _checkpoint_path(path: Path) -> Path:
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _sidecar_path(checkpoint: Path) -> Path:
    return _checkpoint_path(checkpoint).with_suffix(".json")


def _requested_effective_config(args: argparse.Namespace) -> dict[str, Any]:
    """The exact policy/environment configuration that must survive a resume."""

    return {
        "algorithm": "RecurrentPPO",
        "policy_class": "RecurrentActorCriticCnnPolicy",
        "environment": {
            "workers": int(args.workers),
            "size": int(args.size),
            "curiosity_scale": float(args.curiosity_scale),
            "curiosity_budget": float(CURIOSITY_BUDGET),
        },
        "optimization": {
            "rollout_steps": int(args.rollout_steps),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "n_epochs": int(args.n_epochs),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "ent_coef": 0.01,
        },
        "policy_kwargs": dict(POLICY_KWARGS),
        "curriculum": {
            "newest_tier_fraction": float(NEWEST_TIER_FRACTION),
            "stop_when_mastered": not bool(args.continue_after_mastery),
        },
        "evaluation": {
            "interval": int(args.evaluation_every),
            "seeds_per_tier": int(args.evaluation_seeds),
            "promotion_threshold": float(PROMOTION_THRESHOLD),
            "retention_threshold": float(RETENTION_THRESHOLD),
            "deterministic": True,
        },
        "seed": int(args.seed),
    }


def _actual_effective_config(model: Any, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "algorithm": type(model).__name__,
        "policy_class": model.policy_class.__name__,
        "environment": {
            "workers": int(model.n_envs),
            "size": int(args.size),
            "curiosity_scale": float(args.curiosity_scale),
            "curiosity_budget": float(CURIOSITY_BUDGET),
        },
        "optimization": {
            "rollout_steps": int(model.n_steps),
            "batch_size": int(model.batch_size),
            "learning_rate": float(model.learning_rate),
            "n_epochs": int(model.n_epochs),
            "gamma": float(model.gamma),
            "gae_lambda": float(model.gae_lambda),
            "ent_coef": float(model.ent_coef),
        },
        "policy_kwargs": {
            key: model.policy_kwargs[key] for key in POLICY_KWARGS
        },
        "curriculum": {
            "newest_tier_fraction": float(NEWEST_TIER_FRACTION),
            "stop_when_mastered": not bool(args.continue_after_mastery),
        },
        "evaluation": {
            "interval": int(args.evaluation_every),
            "seeds_per_tier": int(args.evaluation_seeds),
            "promotion_threshold": float(PROMOTION_THRESHOLD),
            "retention_threshold": float(RETENTION_THRESHOLD),
            "deterministic": True,
        },
        "seed": int(model.seed),
    }


def _config_differences(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[str] = []
        for key in sorted(expected.keys() | actual.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in expected:
                differences.append(f"{path}: unexpected value {actual[key]!r}")
            elif key not in actual:
                differences.append(f"{path}: missing (expected {expected[key]!r})")
            else:
                differences.extend(_config_differences(expected[key], actual[key], path))
        return differences
    if expected != actual:
        return [f"{prefix}: expected {expected!r}, found {actual!r}"]
    return []


def _load_resume_metadata(
    requested_checkpoint: Path, expected_config: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    checkpoint = _checkpoint_path(requested_checkpoint.expanduser().resolve())
    sidecar_path = _sidecar_path(checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"resume checkpoint does not exist: {checkpoint}")
    if not sidecar_path.is_file():
        raise SystemExit(
            f"resume requires the v0.1 checkpoint sidecar: {sidecar_path}"
        )
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read resume sidecar {sidecar_path}: {error}") from error

    if sidecar.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise SystemExit(
            "resume checkpoint schema is incompatible: "
            f"expected {CHECKPOINT_SCHEMA_VERSION}, found {sidecar.get('schema_version')!r}"
        )
    if sidecar.get("protocol") != PROTOCOL:
        raise SystemExit(
            f"resume protocol mismatch: expected {PROTOCOL!r}, "
            f"found {sidecar.get('protocol')!r}"
        )
    recorded_hash = sidecar.get("checkpoint_sha256")
    actual_hash = file_sha256(checkpoint)
    if recorded_hash != actual_hash:
        raise SystemExit(
            f"resume checkpoint hash mismatch: sidecar has {recorded_hash!r}, "
            f"archive is {actual_hash}"
        )
    differences = _config_differences(expected_config, sidecar.get("effective_config"))
    if differences:
        rendered = "\n  - ".join(differences)
        raise SystemExit(f"resume configuration is incompatible:\n  - {rendered}")

    progress = sidecar.get("progress", {})
    collected = int(progress.get("collected_timesteps", -1))
    trained = int(progress.get("trained_timesteps", -2))
    updates = int(progress.get("n_updates", -1))
    if collected < 0 or trained < 0 or updates < 0 or collected != trained:
        raise SystemExit(
            "resume checkpoint is not a fully trained boundary: "
            f"collected={collected}, trained={trained}, n_updates={updates}"
        )
    curriculum = sidecar.get("curriculum", {})
    try:
        tier = DungeonTier(int(curriculum.get("max_tier", -1)))
    except (TypeError, ValueError) as error:
        raise SystemExit("resume checkpoint has an invalid curriculum tier") from error
    mastered = curriculum.get("mastered", False)
    if not isinstance(mastered, bool):
        raise SystemExit("resume checkpoint has an invalid mastered flag")
    if mastered and tier is not DungeonTier.RETRIEVE:
        raise SystemExit("resume checkpoint claims mastery before the final tier")
    return checkpoint, sidecar


def _json_scalar(value: Any) -> int | float | str | bool | None:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    return None
