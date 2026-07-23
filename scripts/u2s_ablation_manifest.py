#!/usr/bin/env python3
"""Create and advance the fail-closed U2-S matched-ablation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import uuid
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROTOCOL = "dungeon-apprentice-v0.2-u2s-stability-ablation"
COHORT_ID = "v0.2-u2s-ablation-20260723"
TAG_NAME = "u2s-stability-ablation-v0.2-u2s-20260723"
ACTION_CAP = 1_048_576
TOTAL_ACTION_CAP = 4 * ACTION_CAP
EVALUATION_EVERY = 32_768
EXAM_EPISODES = 80
EXAM_COUNT = ACTION_CAP // EVALUATION_EVERY
LESSON_IDS = (
    "navigate/full",
    "unlock/u0-visible",
    "unlock/u1-local",
    "unlock/u2-separated",
)
EXPECTED_CASE_COUNT = EXAM_COUNT * EXAM_EPISODES * len(LESSON_IDS)
PARENT_LIFETIME_ACTIONS = 786_432
TERMINAL_LIFETIME_ACTIONS = PARENT_LIFETIME_ACTIONS + ACTION_CAP
ALGORITHM_SEED = 20_260_753
WORKER_STREAMS = (20_260_753, 20_260_754, 20_260_755, 20_260_756)
PARENT_CHECKPOINT = (
    "/Volumes/T7 Developer/DungeonApprentice/u1-local-replication-20260722/"
    "v02-u1-replication-seed-20260733/checkpoints/mastered-local-unlock.zip"
)
PARENT_CHECKPOINT_SHA256 = (
    "3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104"
)
PARENT_SIDECAR = str(Path(PARENT_CHECKPOINT).with_suffix(".json"))
PARENT_SIDECAR_SHA256 = (
    "268f89361521dd855ba637254b236592186992718ad37ac374e85e5dbf408a07"
)
PARENT_MANIFEST = (
    "/Volumes/T7 Developer/DungeonApprentice/u1-local-replication-20260722/"
    "v02-u1-replication-seed-20260733/manifest.json"
)
PARENT_MANIFEST_SHA256 = (
    "cd808ba8fea6b79225895b455d975122f85d66f1451569bd04a69737f8a6dfba"
)
PARENT_POLICY_TENSOR_SHA256 = (
    "e555d3f7e2364f74e3b43371938c2f25e2ddbf62558a810038509a70868e6888"
)
PARENT_OPTIMIZER_STATE_SHA256 = (
    "cc07791b374620d680e5cbb3602d59ab83eb55f195d26808abc0adf5f0e3bc5f"
)
CONFIRMATION_REPORT = (
    "/Volumes/T7 Developer/DungeonApprentice/confirmations/"
    "v0.2-u1-v2-20260723/report.json"
)
CONFIRMATION_REPORT_SHA256 = (
    "6e577170050f6f14599b793a031776a19bf7c64eba0f243f457298da3193ae8f"
)
CONFIRMATION_ATTEMPT = (
    "/Volumes/T7 Developer/DungeonApprentice/confirmations/"
    "v0.2-u1-v2-20260723/attempt.json"
)
CONFIRMATION_ATTEMPT_SHA256 = (
    "62147b2b556fe51e83876f2be35467fb488a4fb3b9c6347a600cf969e2bc86eb"
)
CONFIRMATION_CHECKSUM = (
    "/Volumes/T7 Developer/DungeonApprentice/confirmations/"
    "v0.2-u1-v2-20260723/report.json.sha256"
)
CONFIRMATION_CHECKSUM_SHA256 = (
    "0be93cbbaf6581c4f6188ff6c4c82dee630f28da1cce462fa2acdb9caebed71a"
)
QUALIFICATION_REPORT = (
    "/Volumes/T7 Developer/DungeonApprentice/qualifications/"
    "v0.2-u2s-20260723/report.json"
)
QUALIFIED_GUARD_MAPPING_SHA256 = (
    "cf468f599737220224c696c796d1305c78453359537b80d7b32716ae53e1c2ef"
)
U2R_TERMINAL_ACTIVE_RECORDS_SHA256 = (
    "93663439a363a4c47152cc4e320654a1f2807a708637777b1fe71b358817e522"
)
QUALIFIED_GUARD_COUNTS = {
    "navigate/full": 76_565,
    "unlock/u0-visible": 79,
    "unlock/u1-local": 7_419,
    "unlock/u2-separated": 15_368,
}
PROTOCOL_DOCUMENT = "docs/protocol-v0.2-u2s-stability-ablation.md"
LINEAGE_CAP_BYTES = 2 * 1024**3
COHORT_SCIENTIFIC_CAP_BYTES = 6 * 1024**3
MEDIA_CAP_BYTES = 10 * 1024**3
COMBINED_PLANNED_CAP_BYTES = 16 * 1024**3
MAX_JSON_BYTES = 8 * 1024 * 1024
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
OBJECT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
ARM_ORDER = ("control", "conservative", "no-effect", "combined")
FACTORIAL_EFFECT_ORDER = (
    "conservative_main",
    "no_effect_main",
    "interaction",
)
FACTORIAL_METRIC_ORDER = (
    "u2_successes_mean",
    "u0_mean_ineffective_interactions",
    "u1_mean_ineffective_interactions",
    "u2_mean_ineffective_interactions",
    "ineffective_cases_at_least_1",
    "ineffective_cases_at_least_3",
    "ineffective_cases_at_least_10",
    "ineffective_cases_at_least_32",
    "worst_case_ineffective_interactions",
    "worst_repeated_identical_interaction_run",
    "worst_identical_visible_no_effect_streak",
)
INEFFECTIVE_TAIL_THRESHOLDS = (
    "at_least_1",
    "at_least_3",
    "at_least_10",
    "at_least_32",
)
ARMS: tuple[dict[str, Any], ...] = (
    {
        "id": "control",
        "label": "A · Current PPO",
        "directory": "control",
        "media_directory": "control",
        "ppo_profile": "current",
        "no_effect_penalty": False,
    },
    {
        "id": "conservative",
        "label": "B · Conservative PPO",
        "directory": "conservative",
        "media_directory": "conservative",
        "ppo_profile": "conservative",
        "no_effect_penalty": False,
    },
    {
        "id": "no-effect",
        "label": "C · No-effect feedback",
        "directory": "no-effect",
        "media_directory": "no-effect",
        "ppo_profile": "current",
        "no_effect_penalty": True,
    },
    {
        "id": "combined",
        "label": "D · Conservative + feedback",
        "directory": "combined",
        "media_directory": "combined",
        "ppo_profile": "conservative",
        "no_effect_penalty": True,
    },
)
ARM_STATES = frozenset(
    {
        "pending",
        "training",
        "completed",
        "interrupted",
        "crashed",
        "integrity_failed",
    }
)
COHORT_PHASES = frozenset(
    {
        "ready",
        "training",
        "awaiting_closeout",
        "completed",
        "operationally_incomplete",
        "integrity_failed",
    }
)


class U2sManifestError(RuntimeError):
    """Raised when the U2-S cohort contract or state would drift."""


def _arm_intervention(arm_id: str) -> dict[str, Any]:
    conservative = arm_id in {"conservative", "combined"}
    penalty = arm_id in {"no-effect", "combined"}
    return {
        "name": arm_id,
        "conservative_ppo": conservative,
        "no_effect_penalty": penalty,
        "n_epochs": 2 if conservative else 4,
        "learning_rate": {
            "schedule": (
                "linear_by_new_u2_child_actions"
                if conservative
                else "constant"
            ),
            "start": 0.00025,
            "end": 0.000025 if conservative else 0.00025,
        },
        "clip_range": 0.1 if conservative else 0.2,
        "target_kl": 0.015 if conservative else None,
        "no_effect_reward": {
            "amount": -0.01 if penalty else 0.0,
            "episode_cap": 0.1 if penalty else 0.0,
            "starts_on_identical_action_number": 2,
            "interaction_actions": [3, 4, 5],
            "requires_next_visible_pixels_unchanged": True,
            "privileged_state": False,
            "enabled_during_evaluation": False,
            "reset_on_pixel_change": True,
            "reset_on_action_change": True,
            "reset_on_non_interaction": True,
            "reset_on_environment_reset": True,
            "reset_on_episode_end": True,
        },
    }


def _common_learner_contract() -> dict[str, Any]:
    return {
        "environment": {
            "size": 9,
            "workers": 4,
            "observation": {
                "kind": "egocentric_partial_rgb",
                "shape": [56, 56, 3],
                "recurrent_state_units": 256,
                "recurrent_layers": 1,
                "privileged_state": False,
            },
            "actions": 7,
        },
        "optimization": {
            "rollout_steps": 512,
            "rollout_transitions": 2048,
            "batch_size": 256,
            "gamma": 0.995,
            "gae_lambda": 0.98,
            "ent_coef": 0.01,
        },
        "reward": {
            "success": 1.0,
            "ordinary_step": -0.001,
            "pixel_novelty": 0.002,
            "episodic_curiosity_cap": 0.1,
            "timeout_is_terminal_failure": True,
            "curiosity_during_exams": False,
        },
        "normal_transition_targets": {
            "navigate/full": 0.5,
            "unlock/u0-visible": 0.075,
            "unlock/u1-local": 0.075,
            "unlock/u2-separated": 0.35,
        },
        "terminal_rule": {
            "final_exam_boundaries": [983_040, 1_015_808, 1_048_576],
            "u2_successes_minimum": 72,
            "u2_panel_successes_minimum": 34,
            "u0_u1_u2_mean_ineffective_maximum": 3.0,
            "max_case_ineffective_exclusive": 10,
            "max_repeated_identical_interaction_run_exclusive": 10,
            "visible_no_effect_streak": "reported_diagnostic_not_terminal_gate",
        },
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _encoded(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise U2sManifestError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _exclusive_write(path: Path, payload: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(_encoded(payload))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(_encoded(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()


def _regular_directory(path: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        raise U2sManifestError(f"{label} must be absolute")
    try:
        metadata = candidate.lstat()
    except FileNotFoundError as error:
        raise U2sManifestError(f"{label} does not exist: {candidate}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise U2sManifestError(f"{label} must be a regular directory: {candidate}")
    return candidate.resolve(strict=True)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise U2sManifestError(f"{label} is missing: {path}") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 0
        or metadata.st_size > MAX_JSON_BYTES
    ):
        raise U2sManifestError(f"{label} is not a safe regular JSON file: {path}")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise U2sManifestError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise U2sManifestError(f"{label} must contain a JSON object")
    return payload


def _validate_object_id(value: str, *, label: str, pattern: re.Pattern[str]) -> str:
    normalized = value.strip().lower()
    if pattern.fullmatch(normalized) is None:
        raise U2sManifestError(f"{label} must be a full hexadecimal Git object ID")
    return normalized


def _arm_definition(arm_id: str) -> dict[str, Any]:
    try:
        return next(arm for arm in ARMS if arm["id"] == arm_id)
    except StopIteration as error:
        raise U2sManifestError(f"unknown U2-S arm: {arm_id}") from error


def _verified_qualification_binding(
    *,
    source_commit: str,
    tag_object: str,
) -> dict[str, Any]:
    try:
        from dungeon_apprentice.v02_u2s_qualify import verify_u2s_qualification
    except ImportError as error:
        raise U2sManifestError("U2-S qualification verifier is unavailable") from error
    repository = Path(__file__).resolve().parents[1]
    report_path = Path(QUALIFICATION_REPORT)
    try:
        evidence = verify_u2s_qualification(
            report_path,
            expected_source_commit=source_commit,
            expected_tag_object=tag_object,
            repository=repository,
        )
        public = evidence.public_dict()
        verified_report = evidence.verified_report()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise U2sManifestError(f"U2-S qualification is invalid: {error}") from error
    if not isinstance(public, dict):
        raise U2sManifestError("U2-S qualification verifier returned no public evidence")
    guard_sets = public.get("guard_sets")
    storage_caps = public.get("storage_caps")
    deep_digest_fields = (
        "guard_mapping_sha256",
        "u2r_terminal_active_records_sha256",
        "sampler_preflight_sha256",
        "arm_contract_sha256",
        "protected_partitions_sha256",
        "resume_contract_sha256",
        "storage_preflight_sha256",
    )
    if (
        public.get("report") != QUALIFICATION_REPORT
        or not isinstance(public.get("report_sha256"), str)
        or public.get("source_commit") != source_commit
        or public.get("tag") != TAG_NAME
        or public.get("tag_object") != tag_object
        or not isinstance(public.get("tag_payload_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", public["tag_payload_sha256"]) is None
        or public.get("verdict") != "qualified"
        or not isinstance(public.get("protocol_document_sha256"), str)
        or any(
            not isinstance(public.get(field), str)
            or re.fullmatch(r"[0-9a-f]{64}", public[field]) is None
            for field in deep_digest_fields
        )
        or public.get("guard_mapping_sha256")
        != QUALIFIED_GUARD_MAPPING_SHA256
        or public.get("u2r_terminal_active_records_sha256")
        != U2R_TERMINAL_ACTIVE_RECORDS_SHA256
        or not isinstance(guard_sets, Mapping)
        or set(guard_sets)
        != {
            "navigate/full",
            "unlock/u0-visible",
            "unlock/u1-local",
            "unlock/u2-separated",
        }
        or not isinstance(storage_caps, Mapping)
    ):
        raise U2sManifestError("U2-S qualification public binding is incomplete")
    normalized_guards: dict[str, dict[str, Any]] = {}
    for lesson_id, guard in guard_sets.items():
        count = (
            guard.get("count", guard.get("exact_layouts"))
            if isinstance(guard, Mapping)
            else None
        )
        digest = (
            guard.get("sha256", guard.get("exact_layout_set_sha256"))
            if isinstance(guard, Mapping)
            else None
        )
        if (
            not isinstance(guard, Mapping)
            or int(count if count is not None else -1) < 1
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise U2sManifestError(
                f"U2-S qualification guard identity is invalid for {lesson_id}"
            )
        normalized_guards[lesson_id] = {
            "count": int(count),
            "sha256": digest,
            "rule": guard.get("rule"),
        }
    if {
        lesson: value["count"] for lesson, value in normalized_guards.items()
    } != QUALIFIED_GUARD_COUNTS:
        raise U2sManifestError("U2-S qualification guard counts changed")
    normalized_caps = {
        "lineage_cap_bytes": storage_caps.get(
            "lineage_cap_bytes",
            storage_caps.get("per_arm_bytes"),
        ),
        "cohort_scientific_cap_bytes": storage_caps.get(
            "cohort_scientific_cap_bytes",
            storage_caps.get("scientific_cohort_bytes"),
        ),
        "media_cap_bytes": storage_caps.get(
            "media_cap_bytes",
            storage_caps.get("media_bytes"),
        ),
        "combined_planned_cap_bytes": storage_caps.get(
            "combined_planned_cap_bytes",
            storage_caps.get("combined_bytes"),
        ),
    }
    expected_caps = {
        "lineage_cap_bytes": LINEAGE_CAP_BYTES,
        "cohort_scientific_cap_bytes": COHORT_SCIENTIFIC_CAP_BYTES,
        "media_cap_bytes": MEDIA_CAP_BYTES,
        "combined_planned_cap_bytes": COMBINED_PLANNED_CAP_BYTES,
    }
    if any(
        int(normalized_caps.get(key) or -1) != value
        for key, value in expected_caps.items()
    ):
        raise U2sManifestError("U2-S qualification storage caps differ from the frozen caps")
    protocol_path = repository / PROTOCOL_DOCUMENT
    if (
        protocol_path.is_symlink()
        or not protocol_path.is_file()
        or _sha256(protocol_path) != public["protocol_document_sha256"]
    ):
        raise U2sManifestError("qualified protocol-document digest changed")
    report_guards = (
        verified_report.get("guards")
        if isinstance(verified_report, Mapping)
        else None
    )
    extension = (
        report_guards.get("u2r_r1_extension")
        if isinstance(report_guards, Mapping)
        else None
    )
    if (
        not isinstance(report_guards, Mapping)
        or report_guards.get("applied_mapping_sha256")
        != QUALIFIED_GUARD_MAPPING_SHA256
        or not isinstance(extension, Mapping)
        or extension.get("terminal_active_records_sha256")
        != U2R_TERMINAL_ACTIVE_RECORDS_SHA256
    ):
        raise U2sManifestError("U2-S qualification extended guard anchor changed")
    initial_rng_identity = _validated_initial_rng_identity(
        public.get("initial_rng_identity")
    )
    return {
        "report": QUALIFICATION_REPORT,
        "report_sha256": public["report_sha256"],
        "checksum": public.get("checksum"),
        "source_commit": source_commit,
        "tag": TAG_NAME,
        "tag_object": tag_object,
        "tag_payload_sha256": public["tag_payload_sha256"],
        "verdict": "qualified",
        "protocol_document": PROTOCOL_DOCUMENT,
        "protocol_document_sha256": public["protocol_document_sha256"],
        "guard_sets": {
            lesson_id: normalized_guards[lesson_id]
            for lesson_id in sorted(normalized_guards)
        },
        "guard_mapping_sha256": QUALIFIED_GUARD_MAPPING_SHA256,
        "u2r_terminal_active_records_sha256": (
            U2R_TERMINAL_ACTIVE_RECORDS_SHA256
        ),
        "sampler_preflight_sha256": public["sampler_preflight_sha256"],
        "arm_contract_sha256": public["arm_contract_sha256"],
        "protected_partitions_sha256": public[
            "protected_partitions_sha256"
        ],
        "resume_contract_sha256": public["resume_contract_sha256"],
        "storage_preflight_sha256": public["storage_preflight_sha256"],
        "initial_rng_identity": initial_rng_identity,
        "storage_caps": expected_caps,
    }


def _expected_contract(
    *,
    root: Path,
    media_root: Path,
    source_commit: str,
    tag_object: str,
    qualification: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "cohort_id": COHORT_ID,
        "source": {"commit": source_commit, "dirty": False},
        "preregistration": {
            "tag": TAG_NAME,
            "tag_object": tag_object,
            "peeled_commit": source_commit,
            "tag_payload_sha256": qualification["tag_payload_sha256"],
        },
        "parent": {
            "checkpoint": PARENT_CHECKPOINT,
            "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "sidecar": PARENT_SIDECAR,
            "sidecar_sha256": PARENT_SIDECAR_SHA256,
            "manifest": PARENT_MANIFEST,
            "manifest_sha256": PARENT_MANIFEST_SHA256,
            "policy_tensor_sha256": PARENT_POLICY_TENSOR_SHA256,
            "optimizer_state_sha256": PARENT_OPTIMIZER_STATE_SHA256,
            "lifetime_trained_actions": PARENT_LIFETIME_ACTIONS,
            "optimizer_updates": 1_536,
        },
        "confirmation": {
            "report": CONFIRMATION_REPORT,
            "report_sha256": CONFIRMATION_REPORT_SHA256,
            "attempt": CONFIRMATION_ATTEMPT,
            "attempt_sha256": CONFIRMATION_ATTEMPT_SHA256,
            "checksum": CONFIRMATION_CHECKSUM,
            "checksum_sha256": CONFIRMATION_CHECKSUM_SHA256,
            "protocol": "dungeon-apprentice-v0.2-u1-confirmation-v2",
            "verdict": "confirmed",
        },
        "qualification": dict(qualification),
        "roots": {"cohort": str(root), "media": str(media_root)},
        "matched_design": {
            "algorithm_seed": ALGORITHM_SEED,
            "worker_streams": list(WORKER_STREAMS),
            "action_cap_per_arm": ACTION_CAP,
            "total_action_cap": TOTAL_ACTION_CAP,
            "evaluation_every": EVALUATION_EVERY,
            "exam_episodes": EXAM_EPISODES,
            "arm_order": list(ARM_ORDER),
            "selection_priority": list(ARM_ORDER),
            "terminal_stability_exams": 3,
            "checkpoint_promotable": False,
            "initial_rng_identity": {
                "captured_before_action_one": True,
                "algorithm_seed": ALGORITHM_SEED,
                "identical_across_all_arms": True,
                "required_component_digests": [
                    "python_random_sha256",
                    "numpy_global_sha256",
                    "torch_cpu_sha256",
                    "torch_mps_sha256",
                    "torch_cuda_sha256",
                    "model_action_space_sha256",
                    "scheduler_sha256",
                ],
                "nullable_component_digests": [
                    "torch_mps_sha256",
                    "torch_cuda_sha256",
                ],
                "workers": {
                    "count": 4,
                    "indices": [0, 1, 2, 3],
                    "streams": list(WORKER_STREAMS),
                    "required_component_digests": [
                        "environment_np_random_sha256",
                        "action_space_sha256",
                        "observation_space_sha256",
                    ],
                },
                "aggregate_sha256_required": True,
            },
            "interruption": {
                "resumable": False,
                "cohort_disposition": "operationally_incomplete",
                "replacement_requires_all_four_fresh_arms": True,
                "replacement_requires_new_source_commit": True,
                "replacement_requires_new_annotated_tag": True,
                "replacement_requires_new_protocol_attempt_id": True,
                "replacement_requires_new_cohort_root": True,
            },
            "common_learner": _common_learner_contract(),
            "storage_caps": {
                "lineage_cap_bytes": LINEAGE_CAP_BYTES,
                "cohort_scientific_cap_bytes": COHORT_SCIENTIFIC_CAP_BYTES,
                "media_cap_bytes": MEDIA_CAP_BYTES,
                "combined_planned_cap_bytes": COMBINED_PLANNED_CAP_BYTES,
                "minimum_free_gib": 25,
            },
        },
        "arms": [
            {
                **dict(arm),
                "algorithm_seed": ALGORITHM_SEED,
                "worker_streams": list(WORKER_STREAMS),
                "action_cap": ACTION_CAP,
                "intervention": _arm_intervention(str(arm["id"])),
            }
            for arm in ARMS
        ],
    }


def _expected_state(contract: dict[str, Any]) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "cohort_id": COHORT_ID,
        "contract": "cohort-contract.json",
        "contract_sha256": hashlib.sha256(_encoded(contract)).hexdigest(),
        "source_commit": contract["source"]["commit"],
        "tag": TAG_NAME,
        "tag_object": contract["preregistration"]["tag_object"],
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "phase": "ready",
        "active_arm": None,
        "terminal_report": None,
        "created_at": now,
        "updated_at": now,
        "arms": [
            {
                **dict(arm),
                "state": "pending",
                "attempts": [],
                "terminal": None,
            }
            for arm in ARMS
        ],
        "history": [{"event": "cohort_created", "at": now}],
    }


def _load_verified(
    root: Path,
    *,
    source_commit: str,
    tag_object: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    resolved = _regular_directory(root, label="cohort root")
    contract = _read_json(resolved / "cohort-contract.json", label="cohort contract")
    state = _read_json(resolved / "cohort.json", label="cohort state")
    media_value = contract.get("roots", {}).get("media")
    if not isinstance(media_value, str):
        raise U2sManifestError("cohort contract has no media root")
    media_root = _regular_directory(Path(media_value), label="media root")
    qualification = _verified_qualification_binding(
        source_commit=source_commit,
        tag_object=tag_object,
    )
    expected = _expected_contract(
        root=resolved,
        media_root=media_root,
        source_commit=source_commit,
        tag_object=tag_object,
        qualification=qualification,
    )
    if contract != expected:
        raise U2sManifestError("immutable U2-S cohort contract differs from the fixed design")
    contract_digest = _sha256(resolved / "cohort-contract.json")
    if (
        state.get("schema_version") != 1
        or state.get("protocol") != PROTOCOL
        or state.get("cohort_id") != COHORT_ID
        or state.get("contract") != "cohort-contract.json"
        or state.get("contract_sha256") != contract_digest
        or state.get("source_commit") != source_commit
        or state.get("tag") != TAG_NAME
        or state.get("tag_object") != tag_object
        or state.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or state.get("phase") not in COHORT_PHASES
        or "terminal_report" not in state
        or not isinstance(state.get("arms"), list)
        or len(state["arms"]) != len(ARMS)
    ):
        raise U2sManifestError("mutable U2-S cohort state has invalid provenance")
    for actual, definition in zip(state["arms"], ARMS, strict=True):
        if (
            not isinstance(actual, dict)
            or any(actual.get(key) != value for key, value in definition.items())
            or actual.get("state") not in ARM_STATES
            or not isinstance(actual.get("attempts"), list)
        ):
            raise U2sManifestError("U2-S arm identity or state drifted")
    active = state.get("active_arm")
    training = [arm["id"] for arm in state["arms"] if arm["state"] == "training"]
    if (
        len(training) > 1
        or (training and active != training[0])
        or (not training and active is not None)
    ):
        raise U2sManifestError("U2-S cohort has an inconsistent active arm")
    states = [arm["state"] for arm in state["arms"]]
    first_not_completed = next(
        (index for index, value in enumerate(states) if value != "completed"),
        len(states),
    )
    if any(value == "completed" for value in states[first_not_completed + 1 :]):
        raise U2sManifestError("U2-S completed arms are not a fixed-order prefix")
    phase = state["phase"]
    if (
        (phase == "ready" and (active is not None or not any(
            value == "pending" for value in states
        )))
        or (phase == "training" and len(training) != 1)
        or (
            phase == "awaiting_closeout"
            and states != ["completed"] * len(ARMS)
        )
        or (
            phase == "completed"
            and (
                states != ["completed"] * len(ARMS)
                or not isinstance(state.get("terminal_report"), Mapping)
            )
        )
        or (
            phase in {"operationally_incomplete", "integrity_failed"}
            and active is not None
        )
    ):
        raise U2sManifestError("U2-S cohort phase and arm states are inconsistent")
    return resolved, contract, state


def _arm_state(state: dict[str, Any], arm_id: str) -> dict[str, Any]:
    return next(arm for arm in state["arms"] if arm["id"] == arm_id)


def _assert_order(state: dict[str, Any], arm_id: str) -> None:
    index = ARM_ORDER.index(arm_id)
    states = [arm["state"] for arm in state["arms"]]
    if any(value != "completed" for value in states[:index]):
        raise U2sManifestError("U2-S arms must run sequentially in their fixed order")
    if states[index] != "pending":
        raise U2sManifestError("a fresh U2-S arm must still be pending")
    if any(value != "pending" for value in states[index + 1 :]):
        raise U2sManifestError("later U2-S arms must remain unopened")


def _status_source_commit(status: dict[str, Any]) -> Any:
    source = status.get("source")
    return source.get("commit") if isinstance(source, dict) else source


def _validate_status_identity(
    status: dict[str, Any],
    *,
    arm_id: str,
    source_commit: str,
) -> None:
    if (
        status.get("protocol") != PROTOCOL
        or status.get("arm") != arm_id
        or _status_source_commit(status) != source_commit
        or status.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or int(status.get("action_cap", -1)) != ACTION_CAP
    ):
        raise U2sManifestError(f"{arm_id} status does not match the cohort contract")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validated_initial_rng_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise U2sManifestError("U2-S arm lacks its initial RNG identity")
    identity = json.loads(json.dumps(value))
    if (
        set(identity)
        != {
            "captured_before_action_one",
            "algorithm_seed",
            "components",
            "aggregate_sha256",
        }
        or identity.get("captured_before_action_one") is not True
        or identity.get("algorithm_seed") != ALGORITHM_SEED
        or not isinstance(identity.get("components"), Mapping)
        or not isinstance(identity.get("aggregate_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", identity["aggregate_sha256"]) is None
        or identity["aggregate_sha256"]
        != _canonical_sha256(identity["components"])
    ):
        raise U2sManifestError("U2-S initial RNG aggregate identity is invalid")
    components = identity["components"]
    digest_fields = (
        "python_random_sha256",
        "numpy_global_sha256",
        "torch_cpu_sha256",
        "model_action_space_sha256",
        "scheduler_sha256",
    )
    nullable_fields = ("torch_mps_sha256", "torch_cuda_sha256")
    if set(components) != {*digest_fields, *nullable_fields, "workers"}:
        raise U2sManifestError("U2-S initial RNG component inventory changed")
    if any(
        not isinstance(components.get(field), str)
        or re.fullmatch(r"[0-9a-f]{64}", components[field]) is None
        for field in digest_fields
    ):
        raise U2sManifestError("U2-S initial RNG component digest is invalid")
    if any(
        components.get(field) is not None
        and (
            not isinstance(components[field], str)
            or re.fullmatch(r"[0-9a-f]{64}", components[field]) is None
        )
        for field in nullable_fields
    ):
        raise U2sManifestError("U2-S optional RNG component digest is invalid")
    workers = components.get("workers")
    if not isinstance(workers, list) or len(workers) != len(WORKER_STREAMS):
        raise U2sManifestError("U2-S initial worker RNG inventory changed")
    worker_digest_fields = (
        "environment_np_random_sha256",
        "action_space_sha256",
        "observation_space_sha256",
    )
    for index, (worker, stream) in enumerate(
        zip(workers, WORKER_STREAMS, strict=True)
    ):
        if (
            not isinstance(worker, Mapping)
            or set(worker)
            != {"worker_index", "worker_stream", *worker_digest_fields}
            or worker.get("worker_index") != index
            or worker.get("worker_stream") != stream
            or any(
                not isinstance(worker.get(field), str)
                or re.fullmatch(r"[0-9a-f]{64}", worker[field]) is None
                for field in worker_digest_fields
            )
        ):
            raise U2sManifestError("U2-S initial worker RNG identity changed")
    return identity


def _relative_regular_file(
    root: Path,
    value: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(value, str):
        raise U2sManifestError(f"{label} has no relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise U2sManifestError(f"{label} path escapes its arm directory")
    candidate = root / relative
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        metadata = candidate.lstat()
    except OSError as error:
        raise U2sManifestError(f"cannot authenticate {label}: {error}") from error
    if (
        resolved.parent != resolved_root
        and resolved_root not in resolved.parents
    ):
        raise U2sManifestError(f"{label} path escapes its arm directory")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise U2sManifestError(f"{label} must be a regular file")
    current = candidate.parent
    while current != resolved_root:
        ancestor = current.lstat()
        if stat.S_ISLNK(ancestor.st_mode) or not stat.S_ISDIR(ancestor.st_mode):
            raise U2sManifestError(f"{label} has an unsafe ancestor")
        current = current.parent
    return resolved


def _tree_size(path: Path, *, label: str) -> int:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise U2sManifestError(f"cannot inspect {label}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise U2sManifestError(f"{label} must be a regular directory")
    total = 0
    try:
        children = tuple(path.iterdir())
    except OSError as error:
        raise U2sManifestError(f"cannot enumerate {label}: {error}") from error
    for child in children:
        child_metadata = child.lstat()
        if stat.S_ISLNK(child_metadata.st_mode):
            raise U2sManifestError(f"{label} contains a symlink: {child}")
        if stat.S_ISDIR(child_metadata.st_mode):
            total += _tree_size(child, label=label)
        elif stat.S_ISREG(child_metadata.st_mode):
            total += int(child_metadata.st_size)
        else:
            raise U2sManifestError(f"{label} contains a non-file artifact: {child}")
    return total


def _process_closeout_evidence(
    *,
    root: Path,
    dashboard_pid: int,
    dashboard_port: int = 8787,
) -> dict[str, Any]:
    if isinstance(dashboard_pid, bool) or int(dashboard_pid) <= 1:
        raise U2sManifestError("U2-S closeout requires the exact dashboard PID")
    expected_dashboard_pid = int(dashboard_pid)
    try:
        process_result = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid=,command="],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        listener_result = subprocess.run(
            [
                "/usr/sbin/lsof",
                "-nP",
                "-t",
                f"-iTCP:{dashboard_port}",
                "-sTCP:LISTEN",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise U2sManifestError(
            f"cannot audit U2-S terminal processes: {error}"
        ) from error
    rows: list[dict[str, Any]] = []
    for raw in process_result.stdout.splitlines():
        fields = raw.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        try:
            pid, parent_pid = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        rows.append(
            {
                "pid": pid,
                "parent_pid": parent_pid,
                "command": fields[2],
            }
        )
    dashboard = next(
        (row for row in rows if row["pid"] == expected_dashboard_pid),
        None,
    )
    if (
        dashboard is None
        or "dungeon_apprentice.v02_u2s_dashboard"
        not in dashboard["command"]
        or str(root) not in dashboard["command"]
    ):
        raise U2sManifestError(
            "U2-S dashboard process identity differs at closeout"
        )
    trainer_rows = [
        row
        for row in rows
        if re.search(
            r"(?:^|\s)-m\s+dungeon_apprentice\.v02_u2s\s+train-arm(?:\s|$)",
            row["command"],
        )
    ]
    supervisor_rows = [
        row for row in rows if "u2_trainer_supervisor.py" in row["command"]
    ]
    caffeinate_rows = [
        row
        for row in rows
        if re.search(r"(?:^|/)caffeinate(?:\s|$)", row["command"])
        and re.search(r"(?:^|\s)-ims(?:\s|$)", row["command"])
    ]
    if trainer_rows or supervisor_rows or caffeinate_rows:
        raise U2sManifestError(
            "U2-S trainer, supervisor, or caffeinate process remains at closeout"
        )
    try:
        listener_pids = sorted(
            {int(value) for value in listener_result.stdout.split()}
        )
    except ValueError as error:
        raise U2sManifestError("U2-S listener PID audit is invalid") from error
    if listener_pids != [expected_dashboard_pid]:
        raise U2sManifestError(
            "U2-S dashboard is not the sole fixed-port listener at closeout"
        )
    return {
        "checked_at": _utc_now(),
        "trainer_process_count": 0,
        "supervisor_process_count": 0,
        "caffeinate_process_count": 0,
        "orphan_free": True,
        "dashboard": {
            "pid": expected_dashboard_pid,
            "parent_pid": int(dashboard["parent_pid"]),
            "command_sha256": hashlib.sha256(
                dashboard["command"].encode()
            ).hexdigest(),
            "host": "127.0.0.1",
            "port": dashboard_port,
            "listener_pids": listener_pids,
            "explicitly_excepted": True,
        },
        "relevant_process_inventory_sha256": _canonical_sha256(
            [
                {
                    "pid": expected_dashboard_pid,
                    "parent_pid": int(dashboard["parent_pid"]),
                    "command_sha256": hashlib.sha256(
                        dashboard["command"].encode()
                    ).hexdigest(),
                    "kind": "dashboard_exception",
                }
            ]
        ),
    }


def _verified_arm_terminal(
    directory: Path,
    *,
    arm_id: str,
    source_commit: str,
    contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from dungeon_apprentice.v02_u2s import verify_terminal_report
    except ImportError as error:
        raise U2sManifestError("U2-S terminal verifier is unavailable") from error
    try:
        summary = verify_terminal_report(
            directory,
            expected_arm=arm_id,
            expected_source_commit=source_commit,
            expected_cohort_contract_sha256=contract_sha256,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise U2sManifestError(
            f"{arm_id} terminal evidence failed deep verification: {error}"
        ) from error
    if not isinstance(summary, Mapping):
        raise U2sManifestError(f"{arm_id} terminal verifier returned no evidence")
    normalized = json.loads(json.dumps(summary))
    normalized["initial_rng_identity"] = _validated_initial_rng_identity(
        normalized.get("initial_rng_identity")
    )
    if (
        normalized.get("arm") != arm_id
        or normalized.get("verdict")
        not in {"stable_mechanism_candidate", "arm_failed"}
        or not isinstance(normalized.get("mechanism_selection_eligible"), bool)
        or (
            normalized["mechanism_selection_eligible"]
            != (normalized["verdict"] == "stable_mechanism_candidate")
        )
        or int(normalized.get("child_trained_actions", -1)) != ACTION_CAP
        or int(normalized.get("lifetime_trained_actions", -1))
        != TERMINAL_LIFETIME_ACTIONS
        or int(normalized.get("exam_count", -1)) != EXAM_COUNT
        or int(normalized.get("case_count", -1)) != EXPECTED_CASE_COUNT
        or not isinstance(normalized.get("optimizer_updates"), int)
    ):
        raise U2sManifestError(
            f"{arm_id} terminal verifier returned incomplete matched evidence"
        )
    for key in ("terminal_checkpoint_sha256", "report_sha256"):
        value = normalized.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise U2sManifestError(f"{arm_id} terminal evidence lacks {key}")

    report_path = directory / "report.json"
    report = _read_json(report_path, label=f"{arm_id} terminal report")
    if _sha256(report_path) != normalized["report_sha256"]:
        raise U2sManifestError(f"{arm_id} terminal report changed after verification")
    integrity_path = directory / "report.integrity.json"
    integrity = _read_json(
        integrity_path,
        label=f"{arm_id} terminal report integrity",
    )
    if (
        integrity.get("schema_version") != 1
        or integrity.get("protocol") != PROTOCOL
        or integrity.get("report") != "report.json"
        or integrity.get("report_sha256") != normalized["report_sha256"]
    ):
        raise U2sManifestError(f"{arm_id} terminal report integrity changed")

    terminal = report.get("terminal_checkpoint")
    exams = report.get("exam_records")
    if not isinstance(terminal, Mapping) or not isinstance(exams, list):
        raise U2sManifestError(f"{arm_id} terminal report is incomplete")
    checkpoint = _relative_regular_file(
        directory,
        terminal.get("path"),
        label=f"{arm_id} terminal checkpoint",
    )
    sidecar = _relative_regular_file(
        directory,
        str(checkpoint.with_suffix(".json").relative_to(directory)),
        label=f"{arm_id} terminal checkpoint sidecar",
    )
    checkpoint_integrity = _relative_regular_file(
        directory,
        str(
            checkpoint.with_suffix(".integrity.json").relative_to(directory)
        ),
        label=f"{arm_id} terminal checkpoint integrity",
    )
    if (
        _sha256(checkpoint) != terminal.get("sha256")
        or terminal.get("sha256") != normalized["terminal_checkpoint_sha256"]
        or _sha256(sidecar) != terminal.get("sidecar_sha256")
        or terminal.get("sidecar_sha256")
        != normalized.get("terminal_sidecar_sha256")
        or _sha256(checkpoint_integrity) != terminal.get("integrity_sha256")
        or terminal.get("integrity_sha256")
        != normalized.get("terminal_integrity_sha256")
    ):
        raise U2sManifestError(f"{arm_id} terminal checkpoint bundle changed")

    case_evidence: list[dict[str, Any]] = []
    for index, exam in enumerate(exams, start=1):
        if (
            not isinstance(exam, Mapping)
            or int(exam.get("child_trained_actions", -1))
            != index * EVALUATION_EVERY
        ):
            raise U2sManifestError(f"{arm_id} terminal exam sequence changed")
        diagnostics = exam.get("case_diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise U2sManifestError(f"{arm_id} exam lacks case evidence")
        case_path = _relative_regular_file(
            directory,
            diagnostics.get("path"),
            label=f"{arm_id} exam case evidence",
        )
        case_sha256 = diagnostics.get("sha256")
        if (
            not isinstance(case_sha256, str)
            or _sha256(case_path) != case_sha256
            or int(diagnostics.get("case_count", -1))
            != EXAM_EPISODES * len(LESSON_IDS)
        ):
            raise U2sManifestError(f"{arm_id} exam case evidence changed")
        case_evidence.append(
            {
                "child_trained_actions": index * EVALUATION_EVERY,
                "path": str(case_path.relative_to(directory)),
                "sha256": case_sha256,
            }
        )
    if len(case_evidence) != EXAM_COUNT:
        raise U2sManifestError(f"{arm_id} terminal exam count changed")
    core_case_files = [
        {
            "path": value["path"],
            "sha256": value["sha256"],
            "record_count": EXAM_EPISODES * len(LESSON_IDS),
        }
        for value in case_evidence
    ]
    if (
        normalized.get("case_evidence_files") != core_case_files
        or normalized.get("case_evidence_sha256")
        != _canonical_sha256(core_case_files)
    ):
        raise U2sManifestError(f"{arm_id} case-evidence inventory changed")

    evidence = {
        **normalized,
        "report": str(report_path.relative_to(directory.parent)),
        "report_integrity": str(integrity_path.relative_to(directory.parent)),
        "report_integrity_sha256": _sha256(integrity_path),
        "terminal_checkpoint": str(checkpoint.relative_to(directory.parent)),
        "terminal_checkpoint_sidecar_sha256": normalized[
            "terminal_sidecar_sha256"
        ],
        "terminal_checkpoint_integrity_sha256": normalized[
            "terminal_integrity_sha256"
        ],
        "case_evidence": case_evidence,
    }
    return evidence, report


def _mark_integrity_failed(
    resolved: Path,
    state: dict[str, Any],
    *,
    arm_id: str | None,
    stage: str,
) -> None:
    now = _utc_now()
    if arm_id is not None:
        arm = _arm_state(state, arm_id)
        if arm.get("state") == "training" and arm.get("attempts"):
            attempt = arm["attempts"][-1]
            attempt["state"] = "integrity_failed"
            attempt["finished_at"] = now
            arm["state"] = "integrity_failed"
            arm["terminal"] = {
                "verified": False,
                "failure_kind": "terminal_evidence_integrity_failure",
                "resumable": False,
            }
    state["active_arm"] = None
    state["phase"] = "integrity_failed"
    state["updated_at"] = now
    state["history"].append(
        {
            "event": "evidence_integrity_failed",
            "arm": arm_id,
            "stage": stage,
            "at": now,
        }
    )
    _atomic_write(resolved / "cohort.json", state)


def create_cohort(
    root: Path,
    *,
    media_root: Path,
    source_commit: str,
    tag_object: str,
) -> dict[str, Any]:
    resolved = _regular_directory(root, label="cohort root")
    resolved_media = _regular_directory(media_root, label="media root")
    if any(resolved.iterdir()):
        raise U2sManifestError("fresh U2-S cohort root must be empty")
    if any(resolved_media.iterdir()):
        raise U2sManifestError("fresh U2-S media root must be empty")
    commit = _validate_object_id(source_commit, label="source commit", pattern=COMMIT_PATTERN)
    tag = _validate_object_id(tag_object, label="tag object", pattern=OBJECT_PATTERN)
    qualification = _verified_qualification_binding(
        source_commit=commit,
        tag_object=tag,
    )
    contract = _expected_contract(
        root=resolved,
        media_root=resolved_media,
        source_commit=commit,
        tag_object=tag,
        qualification=qualification,
    )
    state = _expected_state(contract)
    _exclusive_write(resolved / "cohort-contract.json", contract)
    _exclusive_write(resolved / "cohort.json", state)
    return state


def start_arm(
    root: Path,
    *,
    source_commit: str,
    tag_object: str,
    arm_id: str,
) -> dict[str, Any]:
    _arm_definition(arm_id)
    resolved, _contract, state = _load_verified(
        root,
        source_commit=source_commit,
        tag_object=tag_object,
    )
    if state["active_arm"] is not None:
        raise U2sManifestError("another U2-S arm is already active")
    _assert_order(state, arm_id)
    arm = _arm_state(state, arm_id)
    directory = resolved / str(arm["directory"])
    if directory.exists() or directory.is_symlink():
        raise U2sManifestError(f"fresh U2-S arm target already exists: {directory}")
    attempt = {
        "index": len(arm["attempts"]),
        "state": "training",
        "started_at": _utc_now(),
        "finished_at": None,
    }
    arm["attempts"].append(attempt)
    arm["state"] = "training"
    state["active_arm"] = arm_id
    state["phase"] = "training"
    state["updated_at"] = _utc_now()
    state["history"].append(
        {
            "event": "arm_started",
            "arm": arm_id,
            "attempt": attempt["index"],
            "at": state["updated_at"],
        }
    )
    _atomic_write(resolved / "cohort.json", state)
    return {"arm": arm_id, "attempt": attempt["index"]}


def finish_arm(
    root: Path,
    *,
    source_commit: str,
    tag_object: str,
    arm_id: str,
    outcome: str,
    trainer_exit_code: int,
) -> dict[str, Any]:
    if outcome not in {"completed", "interrupted", "crashed"}:
        raise U2sManifestError(f"invalid U2-S arm outcome: {outcome}")
    if isinstance(trainer_exit_code, bool):
        raise U2sManifestError("trainer exit code must be an integer")
    exit_code = int(trainer_exit_code)
    resolved, _contract, state = _load_verified(
        root,
        source_commit=source_commit,
        tag_object=tag_object,
    )
    arm = _arm_state(state, arm_id)
    if state["active_arm"] != arm_id or arm["state"] != "training":
        raise U2sManifestError("only the active U2-S arm may be finished")
    directory = resolved / str(arm["directory"])
    status_path = directory / "status.json"
    status: dict[str, Any] | None = None
    if status_path.exists() or status_path.is_symlink():
        status = _read_json(status_path, label="arm status")
        _validate_status_identity(status, arm_id=arm_id, source_commit=source_commit)
    if outcome == "completed":
        if exit_code != 0:
            raise U2sManifestError(
                "a nonzero trainer exit can never count as a completed U2-S arm"
            )
        if status is None or status.get("phase") != "completed":
            raise U2sManifestError("completed arm lacks a terminal trainer status")
        if (
            int(status.get("child_trained_actions", -1)) != ACTION_CAP
            or int(status.get("remaining_action_budget", -1)) != 0
        ):
            raise U2sManifestError("completed arm did not consume its exact matched budget")
        try:
            verified_terminal, _report = _verified_arm_terminal(
                directory,
                arm_id=arm_id,
                source_commit=source_commit,
                contract_sha256=state["contract_sha256"],
            )
        except (OSError, TypeError, ValueError, U2sManifestError) as error:
            _mark_integrity_failed(
                resolved,
                state,
                arm_id=arm_id,
                stage="arm_terminal_closeout",
            )
            raise U2sManifestError(
                f"{arm_id} cannot advance: terminal integrity verification failed"
            ) from error
    elif outcome == "interrupted":
        if status is None or status.get("phase") != "interrupted":
            raise U2sManifestError("interrupted arm lacks an interrupted trainer status")
    elif exit_code == 0:
        raise U2sManifestError("a crashed arm requires a nonzero trainer exit")
    attempt = arm["attempts"][-1]
    if attempt.get("state") != "training":
        raise U2sManifestError("active arm attempt is not training")
    attempt["state"] = outcome
    attempt["finished_at"] = _utc_now()
    attempt["trainer_exit_code"] = exit_code
    arm["state"] = outcome
    if outcome == "completed":
        arm["terminal"] = {
            "verified": True,
            "verified_at": _utc_now(),
            "status_sha256": _sha256(status_path),
            "resumable": False,
            **verified_terminal,
        }
    else:
        arm["terminal"] = (
            None
            if status is None
            else {
                "verified": False,
                "phase": status.get("phase"),
                "child_trained_actions": status.get("child_trained_actions"),
                "status_sha256": _sha256(status_path),
                "terminal_eligible": status.get("terminal_eligible"),
                "latest_safe_checkpoint_sha256": status.get(
                    "latest_safe_checkpoint_sha256"
                ),
                "resumable": False,
            }
        )
    state["active_arm"] = None
    state["phase"] = {
        "completed": (
            "awaiting_closeout"
            if all(item["state"] == "completed" for item in state["arms"])
            else "ready"
        ),
        "interrupted": "operationally_incomplete",
        "crashed": "operationally_incomplete",
    }[outcome]
    state["updated_at"] = _utc_now()
    state["history"].append(
        {
            "event": f"arm_{outcome}",
            "arm": arm_id,
            "attempt": attempt["index"],
            "trainer_exit_code": exit_code,
            "at": state["updated_at"],
        }
    )
    _atomic_write(resolved / "cohort.json", state)
    return {"arm": arm_id, "outcome": outcome, "phase": state["phase"]}


def _exam_curve(exams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    for exam in exams:
        lesson_values = exam.get("lessons")
        diagnostics = exam.get("case_diagnostics")
        if not isinstance(lesson_values, Mapping) or not isinstance(
            diagnostics,
            Mapping,
        ):
            raise U2sManifestError("verified U2-S exam summary is incomplete")
        ineffective_tail = diagnostics.get("ineffective_tail")
        if not isinstance(ineffective_tail, Mapping):
            raise U2sManifestError(
                "verified U2-S exam lacks its ineffective tail"
            )
        raw_thresholds = ineffective_tail.get(
            "ineffective_threshold_counts"
        )
        if not isinstance(raw_thresholds, Mapping):
            raise U2sManifestError(
                "verified U2-S exam lacks ineffective tail counts"
            )
        threshold_counts: dict[str, int] = {}
        for threshold in INEFFECTIVE_TAIL_THRESHOLDS:
            value = raw_thresholds.get(threshold)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise U2sManifestError(
                    "verified U2-S ineffective tail counts are invalid"
                )
            threshold_counts[threshold] = value
        direct_at_least_10 = diagnostics.get(
            "cases_with_ineffective_at_least_10"
        )
        if direct_at_least_10 != threshold_counts["at_least_10"]:
            raise U2sManifestError(
                "verified U2-S ineffective tail counts disagree"
            )
        row: dict[str, Any] = {
            "child_trained_actions": int(exam["child_trained_actions"]),
            "practice_profile": exam.get("practice_profile"),
            "allocation_valid": exam.get("allocation_valid") is True,
            "max_case_ineffective_interactions": int(
                diagnostics.get("max_case_ineffective_interactions", -1)
            ),
            "cases_with_ineffective_at_least_10": int(
                diagnostics.get("cases_with_ineffective_at_least_10", -1)
            ),
            "max_repeated_identical_interaction_run": int(
                diagnostics.get("max_repeated_identical_interaction_run", -1)
            ),
            "max_identical_visible_no_effect_streak": int(
                diagnostics.get(
                    "max_identical_visible_no_effect_streak",
                    -1,
                )
            ),
            "ineffective_threshold_counts": threshold_counts,
            "lessons": {},
        }
        for lesson_id in LESSON_IDS:
            lesson = lesson_values.get(lesson_id)
            if not isinstance(lesson, Mapping):
                raise U2sManifestError(
                    f"verified U2-S exam lacks {lesson_id}"
                )
            row["lessons"][lesson_id] = {
                "successes": int(lesson.get("successes", -1)),
                "panel_successes": [
                    int(value) for value in lesson.get("panel_successes", [])
                ],
                "mean_ineffective_interactions": float(
                    lesson.get("mean_ineffective_interactions", -1.0)
                ),
                "ineffective_tail": lesson.get("ineffective_tail"),
                "max_repeated_identical_interaction_run": int(
                    lesson.get(
                        "max_repeated_identical_interaction_run",
                        -1,
                    )
                ),
                "max_identical_visible_no_effect_streak": int(
                    lesson.get(
                        "max_identical_visible_no_effect_streak",
                        -1,
                    )
                ),
            }
        curve.append(row)
    return curve


def _mean(values: list[float]) -> float:
    if not values:
        raise U2sManifestError("cannot summarize an empty U2-S terminal window")
    return sum(values) / len(values)


def _terminal_summary(
    curve: list[dict[str, Any]],
    *,
    grade: Mapping[str, Any],
) -> dict[str, Any]:
    terminal = curve[-3:]
    if [row["child_trained_actions"] for row in terminal] != [
        983_040,
        1_015_808,
        1_048_576,
    ]:
        raise U2sManifestError("U2-S terminal stability window changed")
    u2_rows = [
        row["lessons"]["unlock/u2-separated"] for row in terminal
    ]
    return {
        "eligible": grade.get("eligible") is True,
        "grade": dict(grade),
        "boundaries": [row["child_trained_actions"] for row in terminal],
        "u2_successes_mean": _mean(
            [float(row["successes"]) for row in u2_rows]
        ),
        "u2_successes_minimum": min(int(row["successes"]) for row in u2_rows),
        "u2_mean_ineffective_interactions": _mean(
            [
                float(row["mean_ineffective_interactions"])
                for row in u2_rows
            ]
        ),
        "u0_u1_u2_mean_ineffective_interactions": {
            lesson_id: _mean(
                [
                    float(
                        row["lessons"][lesson_id][
                            "mean_ineffective_interactions"
                        ]
                    )
                    for row in terminal
                ]
            )
            for lesson_id in LESSON_IDS[1:]
        },
        "worst_case_ineffective_interactions": max(
            int(row["max_case_ineffective_interactions"]) for row in terminal
        ),
        "cases_with_ineffective_at_least_10": sum(
            int(row["cases_with_ineffective_at_least_10"])
            for row in terminal
        ),
        "ineffective_threshold_counts": {
            threshold: sum(
                int(row["ineffective_threshold_counts"][threshold])
                for row in terminal
            )
            for threshold in INEFFECTIVE_TAIL_THRESHOLDS
        },
        "worst_repeated_identical_interaction_run": max(
            int(row["max_repeated_identical_interaction_run"])
            for row in terminal
        ),
        "worst_identical_visible_no_effect_streak": max(
            int(row["max_identical_visible_no_effect_streak"])
            for row in terminal
        ),
        "terminal_exams": terminal,
    }


def _factorial_contrasts(
    terminal: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(terminal) != set(ARM_ORDER):
        raise U2sManifestError(
            "U2-S factorial contrasts require all four terminal cells"
        )

    def metric_values(metric: str) -> dict[str, float]:
        values: dict[str, float] = {}
        for arm_id in ARM_ORDER:
            summary = terminal[arm_id]
            if metric == "u2_successes_mean":
                value = summary.get("u2_successes_mean")
            elif metric.endswith("_mean_ineffective_interactions"):
                lesson_id = {
                    "u0_mean_ineffective_interactions": "unlock/u0-visible",
                    "u1_mean_ineffective_interactions": "unlock/u1-local",
                    "u2_mean_ineffective_interactions": "unlock/u2-separated",
                }.get(metric)
                means = summary.get(
                    "u0_u1_u2_mean_ineffective_interactions"
                )
                value = (
                    means.get(lesson_id)
                    if lesson_id is not None and isinstance(means, Mapping)
                    else None
                )
            elif metric.startswith("ineffective_cases_"):
                threshold = metric.removeprefix("ineffective_cases_")
                counts = summary.get("ineffective_threshold_counts")
                value = (
                    counts.get(threshold)
                    if isinstance(counts, Mapping)
                    else None
                )
            else:
                value = summary.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise U2sManifestError(
                    f"U2-S factorial metric {metric} is missing for {arm_id}"
                )
            normalized = float(value)
            if normalized != normalized or normalized in {
                float("-inf"),
                float("inf"),
            }:
                raise U2sManifestError(
                    f"U2-S factorial metric {metric} is nonfinite"
                )
            values[arm_id] = normalized
        return values

    def normalized_effect(value: float) -> float:
        return 0.0 if value == 0.0 else value

    metrics: dict[str, dict[str, Any]] = {}
    categories = {
        "u2_successes_mean": "success",
        "u0_mean_ineffective_interactions": "ineffective_mean",
        "u1_mean_ineffective_interactions": "ineffective_mean",
        "u2_mean_ineffective_interactions": "ineffective_mean",
        "ineffective_cases_at_least_1": "tail_count",
        "ineffective_cases_at_least_3": "tail_count",
        "ineffective_cases_at_least_10": "tail_count",
        "ineffective_cases_at_least_32": "tail_count",
        "worst_case_ineffective_interactions": "tail_maximum",
        "worst_repeated_identical_interaction_run": "streak_maximum",
        "worst_identical_visible_no_effect_streak": "streak_maximum",
    }
    for metric in FACTORIAL_METRIC_ORDER:
        cells = metric_values(metric)
        conservative_main = (
            (cells["conservative"] + cells["combined"]) / 2.0
            - (cells["control"] + cells["no-effect"]) / 2.0
        )
        no_effect_main = (
            (cells["no-effect"] + cells["combined"]) / 2.0
            - (cells["control"] + cells["conservative"]) / 2.0
        )
        interaction = (
            cells["combined"]
            - cells["conservative"]
            - cells["no-effect"]
            + cells["control"]
        )
        metrics[metric] = {
            "category": categories[metric],
            "cells": cells,
            "effects": {
                "conservative_main": normalized_effect(conservative_main),
                "no_effect_main": normalized_effect(no_effect_main),
                "interaction": normalized_effect(interaction),
            },
        }
    return {
        "schema_version": 1,
        "descriptive_only": True,
        "population_inference_authorized": False,
        "cell_order": list(ARM_ORDER),
        "metric_order": list(FACTORIAL_METRIC_ORDER),
        "effect_order": list(FACTORIAL_EFFECT_ORDER),
        "effect_definitions": {
            "conservative_main": (
                "mean(conservative,combined)-mean(control,no-effect)"
            ),
            "no_effect_main": (
                "mean(no-effect,combined)-mean(control,conservative)"
            ),
            "interaction": (
                "(combined-conservative)-(no-effect-control)"
            ),
        },
        "metrics": metrics,
    }


def _matched_initial_rng_identity(
    verified_arms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(verified_arms) != set(ARM_ORDER):
        raise U2sManifestError(
            "U2-S matched RNG proof requires all four arms"
        )
    initial_rng = verified_arms["control"].get("initial_rng_identity")
    if not isinstance(initial_rng, Mapping) or any(
        verified_arms[arm_id].get("initial_rng_identity") != initial_rng
        for arm_id in ARM_ORDER[1:]
    ):
        raise U2sManifestError(
            "U2-S arms did not start from one identical RNG identity"
        )
    normalized = _validated_initial_rng_identity(initial_rng)
    return {
        "identical_across_all_arms": True,
        "identity": normalized,
        "arm_aggregate_sha256": {
            arm_id: verified_arms[arm_id]["initial_rng_identity"][
                "aggregate_sha256"
            ]
            for arm_id in ARM_ORDER
        },
    }


def _cohort_report(
    *,
    resolved: Path,
    contract: dict[str, Any],
    verified_arms: Mapping[str, Mapping[str, Any]],
    arm_reports: Mapping[str, Mapping[str, Any]],
    storage: Mapping[str, Any],
    process_closeout: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        from dungeon_apprentice.v02_u2s import select_mechanism
    except ImportError as error:
        raise U2sManifestError("U2-S mechanism selector is unavailable") from error
    exam_records = {
        arm_id: arm_reports[arm_id]["exam_records"] for arm_id in ARM_ORDER
    }
    try:
        selection = select_mechanism(exam_records)
    except (RuntimeError, TypeError, ValueError) as error:
        raise U2sManifestError(
            f"U2-S fixed-priority selection failed: {error}"
        ) from error
    if (
        not isinstance(selection, Mapping)
        or selection.get("selected_arm") not in {*ARM_ORDER, None}
        or selection.get("ablation_checkpoint_reuse_authorized") is not False
        or selection.get("priority") != list(ARM_ORDER)
    ):
        raise U2sManifestError("U2-S selector returned an invalid decision")
    curves: dict[str, list[dict[str, Any]]] = {}
    terminal: dict[str, dict[str, Any]] = {}
    grades = selection.get("grades")
    if not isinstance(grades, Mapping):
        raise U2sManifestError("U2-S selector returned no arm grades")
    for arm_id in ARM_ORDER:
        exams = exam_records[arm_id]
        if not isinstance(exams, list):
            raise U2sManifestError(f"{arm_id} report has no exam records")
        curves[arm_id] = _exam_curve(exams)
        grade = grades.get(arm_id)
        if not isinstance(grade, Mapping):
            raise U2sManifestError(f"{arm_id} selection grade is missing")
        terminal[arm_id] = _terminal_summary(curves[arm_id], grade=grade)
    control = terminal["control"]
    contrasts = {
        arm_id: {
            "u2_successes_mean_delta": (
                terminal[arm_id]["u2_successes_mean"]
                - control["u2_successes_mean"]
            ),
            "u2_mean_ineffective_delta": (
                terminal[arm_id]["u2_mean_ineffective_interactions"]
                - control["u2_mean_ineffective_interactions"]
            ),
            "worst_case_ineffective_delta": (
                terminal[arm_id]["worst_case_ineffective_interactions"]
                - control["worst_case_ineffective_interactions"]
            ),
            "worst_repeated_interaction_run_delta": (
                terminal[arm_id]["worst_repeated_identical_interaction_run"]
                - control["worst_repeated_identical_interaction_run"]
            ),
        }
        for arm_id in ARM_ORDER[1:]
    }
    selected_arm = selection["selected_arm"]
    verdict = (
        "mechanism_selected"
        if selected_arm is not None
        else "ablation_failed"
    )
    matched_initial_rng = _matched_initial_rng_identity(verified_arms)
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "cohort_id": COHORT_ID,
        "verdict": verdict,
        "completed_at": _utc_now(),
        "development_only": True,
        "source": dict(contract["source"]),
        "preregistration": dict(contract["preregistration"]),
        "parent": dict(contract["parent"]),
        "qualification": dict(contract["qualification"]),
        "cohort_contract": "cohort-contract.json",
        "cohort_contract_sha256": _sha256(
            resolved / "cohort-contract.json"
        ),
        "matched_design": dict(contract["matched_design"]),
        "arm_evidence": {
            arm_id: dict(verified_arms[arm_id]) for arm_id in ARM_ORDER
        },
        "matched_initial_rng_identity": matched_initial_rng,
        "learning_curves": curves,
        "terminal_window": terminal,
        "contrasts_to_control": contrasts,
        "factorial_contrasts": _factorial_contrasts(terminal),
        "selection": dict(selection),
        "selected_configuration": selected_arm,
        "checkpoint_rule": {
            "ablation_checkpoint_reuse_authorized": False,
            "successor_checkpoint": None,
            "successor_must_restart_from_confirmed_u1_parent": True,
            "successor_cohort_authorized": selected_arm is not None,
        },
        "storage": dict(storage),
        "process_closeout": dict(process_closeout),
    }


def finalize_cohort(
    root: Path,
    *,
    source_commit: str,
    tag_object: str,
    dashboard_pid: int,
) -> dict[str, Any]:
    resolved, contract, state = _load_verified(
        root,
        source_commit=source_commit,
        tag_object=tag_object,
    )
    if (
        state["phase"] != "awaiting_closeout"
        or state["active_arm"] is not None
        or [arm["state"] for arm in state["arms"]]
        != ["completed"] * len(ARMS)
    ):
        raise U2sManifestError(
            "U2-S closeout requires four deeply verified terminal arms"
        )
    try:
        verified_arms: dict[str, dict[str, Any]] = {}
        arm_reports: dict[str, dict[str, Any]] = {}
        arm_sizes: dict[str, int] = {}
        for arm in state["arms"]:
            arm_id = str(arm["id"])
            directory = resolved / str(arm["directory"])
            evidence, report = _verified_arm_terminal(
                directory,
                arm_id=arm_id,
                source_commit=source_commit,
                contract_sha256=state["contract_sha256"],
            )
            recorded = arm.get("terminal")
            if (
                not isinstance(recorded, Mapping)
                or recorded.get("verified") is not True
            ):
                raise U2sManifestError(
                    f"{arm_id} lacks its first verified terminal binding"
                )
            for key in (
                "report_sha256",
                "report_integrity_sha256",
                "terminal_checkpoint_sha256",
                "terminal_checkpoint_sidecar_sha256",
                "terminal_checkpoint_integrity_sha256",
                "case_evidence_sha256",
                "case_evidence_files",
                "qualification_sha256",
                "initial_rng_identity",
            ):
                if recorded.get(key) != evidence.get(key):
                    raise U2sManifestError(
                        f"{arm_id} terminal evidence changed before closeout"
                    )
            size = _tree_size(directory, label=f"{arm_id} scientific artifacts")
            if size > LINEAGE_CAP_BYTES:
                raise U2sManifestError(f"{arm_id} exceeded its 2 GiB cap")
            arm_sizes[arm_id] = size
            verified_arms[arm_id] = evidence
            arm_reports[arm_id] = report
        media_root = _regular_directory(
            Path(contract["roots"]["media"]),
            label="media root",
        )
        scientific_bytes = _tree_size(
            resolved,
            label="U2-S scientific cohort",
        )
        media_bytes = _tree_size(media_root, label="U2-S media cohort")
        if scientific_bytes > COHORT_SCIENTIFIC_CAP_BYTES:
            raise U2sManifestError("U2-S scientific cohort exceeded its 6 GiB cap")
        if media_bytes > MEDIA_CAP_BYTES:
            raise U2sManifestError("U2-S media cohort exceeded its 10 GiB cap")
        if scientific_bytes + media_bytes > COMBINED_PLANNED_CAP_BYTES:
            raise U2sManifestError("U2-S combined artifacts exceeded their 16 GiB cap")
        storage = {
            "measured_at": _utc_now(),
            "arm_bytes": arm_sizes,
            "scientific_cohort_bytes_before_report": scientific_bytes,
            "media_bytes": media_bytes,
            "combined_bytes_before_report": scientific_bytes + media_bytes,
            "caps": dict(contract["matched_design"]["storage_caps"]),
            "within_caps": True,
        }
        process_closeout = _process_closeout_evidence(
            root=resolved,
            dashboard_pid=dashboard_pid,
        )
        report = _cohort_report(
            resolved=resolved,
            contract=contract,
            verified_arms=verified_arms,
            arm_reports=arm_reports,
            storage=storage,
            process_closeout=process_closeout,
        )
        report_path = resolved / "report.json"
        integrity_path = resolved / "report.integrity.json"
        if (
            report_path.exists()
            or report_path.is_symlink()
            or integrity_path.exists()
            or integrity_path.is_symlink()
        ):
            raise U2sManifestError(
                "refusing to replace existing U2-S cohort closeout evidence"
            )
        _exclusive_write(report_path, report)
        report_sha256 = _sha256(report_path)
        integrity = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "cohort_id": COHORT_ID,
            "report": "report.json",
            "report_sha256": report_sha256,
            "cohort_contract_sha256": state["contract_sha256"],
            "arm_report_sha256": {
                arm_id: verified_arms[arm_id]["report_sha256"]
                for arm_id in ARM_ORDER
            },
            "arm_case_evidence_sha256": {
                arm_id: verified_arms[arm_id]["case_evidence_sha256"]
                for arm_id in ARM_ORDER
            },
            "initial_rng_identity_sha256": _canonical_sha256(
                report["matched_initial_rng_identity"]
            ),
            "process_closeout_sha256": _canonical_sha256(
                report["process_closeout"]
            ),
            "factorial_contrasts_sha256": _canonical_sha256(
                report["factorial_contrasts"]
            ),
        }
        _exclusive_write(integrity_path, integrity)
        integrity_sha256 = _sha256(integrity_path)
        reread_report = _read_json(report_path, label="U2-S cohort report")
        reread_integrity = _read_json(
            integrity_path,
            label="U2-S cohort report integrity",
        )
        if (
            reread_report != report
            or reread_integrity != integrity
            or reread_integrity["report_sha256"] != _sha256(report_path)
        ):
            raise U2sManifestError("U2-S cohort closeout changed while sealing")
    except (OSError, RuntimeError, TypeError, ValueError, U2sManifestError) as error:
        _mark_integrity_failed(
            resolved,
            state,
            arm_id=None,
            stage="cohort_terminal_closeout",
        )
        raise U2sManifestError("U2-S cohort terminal closeout failed") from error

    now = _utc_now()
    state["phase"] = "completed"
    state["terminal_report"] = {
        "path": "report.json",
        "sha256": report_sha256,
        "integrity": "report.integrity.json",
        "integrity_sha256": integrity_sha256,
        "verdict": report["verdict"],
        "selected_configuration": report["selected_configuration"],
        "checkpoint_promotable": False,
        "initial_rng_identity_sha256": integrity[
            "initial_rng_identity_sha256"
        ],
        "process_closeout_sha256": integrity["process_closeout_sha256"],
        "factorial_contrasts_sha256": integrity[
            "factorial_contrasts_sha256"
        ],
    }
    state["updated_at"] = now
    state["history"].append(
        {
            "event": "cohort_terminal_closeout_verified",
            "verdict": report["verdict"],
            "selected_configuration": report["selected_configuration"],
            "at": now,
        }
    )
    _atomic_write(resolved / "cohort.json", state)
    return dict(state["terminal_report"])


def next_plan(
    root: Path,
    *,
    source_commit: str,
    tag_object: str,
) -> dict[str, Any]:
    resolved, contract, state = _load_verified(
        root,
        source_commit=source_commit,
        tag_object=tag_object,
    )
    states = [arm["state"] for arm in state["arms"]]
    if states == ["completed"] * len(ARMS):
        return {
            "done": True,
            "closeout_required": state["phase"] == "awaiting_closeout",
        }
    if any(
        value in {"crashed", "interrupted", "integrity_failed"}
        for value in states
    ):
        raise U2sManifestError(
            "interrupted U2-S cohorts are terminal operationally_incomplete; "
            "a replacement requires a new source commit, annotated tag, "
            "protocol-attempt identity, cohort root, and four fresh arms"
        )
    pending = [arm for arm in state["arms"] if arm["state"] == "pending"]
    if not pending:
        raise U2sManifestError("U2-S cohort has no legal next arm")
    arm = pending[0]
    _assert_order(state, str(arm["id"]))
    return {
        "done": False,
        "arm": arm["id"],
        "attempt": len(arm["attempts"]),
        "run_dir": str(resolved / arm["directory"]),
        "media_dir": str(Path(contract["roots"]["media"]) / arm["media_directory"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--media-root", type=Path, required=True)
    create.add_argument("--source-commit", required=True)
    create.add_argument("--tag-object", required=True)

    for name in ("start", "finish", "next-plan", "finalize"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--source-commit", required=True)
        command.add_argument("--tag-object", required=True)
        if name in {"start", "finish"}:
            command.add_argument("--arm", choices=ARM_ORDER, required=True)
        if name == "finish":
            command.add_argument(
                "--outcome",
                choices=("completed", "interrupted", "crashed"),
                required=True,
            )
            command.add_argument("--trainer-exit-code", type=int, required=True)
        if name == "finalize":
            command.add_argument("--dashboard-pid", type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "create":
            result = create_cohort(
                args.root,
                media_root=args.media_root,
                source_commit=args.source_commit,
                tag_object=args.tag_object,
            )
        elif args.command == "start":
            result = start_arm(
                args.root,
                source_commit=args.source_commit,
                tag_object=args.tag_object,
                arm_id=args.arm,
            )
        elif args.command == "finish":
            result = finish_arm(
                args.root,
                source_commit=args.source_commit,
                tag_object=args.tag_object,
                arm_id=args.arm,
                outcome=args.outcome,
                trainer_exit_code=args.trainer_exit_code,
            )
        elif args.command == "next-plan":
            result = next_plan(
                args.root,
                source_commit=args.source_commit,
                tag_object=args.tag_object,
            )
        else:
            result = finalize_cohort(
                args.root,
                source_commit=args.source_commit,
                tag_object=args.tag_object,
                dashboard_pid=args.dashboard_pid,
            )
    except (OSError, TypeError, ValueError, U2sManifestError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
