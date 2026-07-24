"""Fail-closed Stage-A qualification for the v0.3 action-effect study.

This module verifies a prospective release; it never creates the annotated
tag and it never launches either scientific arm.  The qualifier deliberately
restarts from the confirmed U1 parent and treats the completed U2-S ablation
as terminal evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dungeon_apprentice import v02_u2 as frozen_u2
from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v02_u2r as frozen_u2r
from dungeon_apprentice import v02_u2s as frozen_u2s
from dungeon_apprentice import v02_u2s_qualify as u2s_qualify
from dungeon_apprentice import v03_action_effect as v03
from dungeon_apprentice import v03_action_effect_smoke as smoke
from dungeon_apprentice.action_effect import (
    ACTION_EFFECT_DIM,
    ACTION_EFFECT_KEY,
    IMAGE_KEY,
    ActionEffectMode,
)
from dungeon_apprentice.artifacts import file_sha256, runtime_snapshot, utc_now
from dungeon_apprentice.u2_seed_guard import (
    SEED_PARTITIONS,
    U2SeedRole,
)

PROTOCOL = v03.PROTOCOL
SCHEMA_VERSION = 1
KIND = "sealed_stage_a_preflight_qualification"
VERDICT = "qualified"

QUALIFIED_TAG = "action-effect-architecture-v0.3-stage-a-r1-20260724"
QUALIFIED_REMOTE = "origin"
EXPECTED_ORIGIN_URL = "https://github.com/PeteAndrews1289/dungeon-apprentice.git"
PROTOCOL_DOCUMENT = Path(v03.PROTOCOL_DOCUMENT)
DASHBOARD_PORT = 8789

CANONICAL_QUALIFICATION_DIRECTORY = Path(
    "/Volumes/T7 Developer/DungeonApprentice/qualifications/"
    "v0.3-action-effect-stage-a-r1-20260724"
)
CANONICAL_REPORT = CANONICAL_QUALIFICATION_DIRECTORY / "report.json"
CANONICAL_CHECKSUM = CANONICAL_QUALIFICATION_DIRECTORY / "report.json.sha256"
CANONICAL_CLAIM = CANONICAL_QUALIFICATION_DIRECTORY / "claim.json"
CANONICAL_COHORT_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/"
    "v03-action-effect-stage-a-r1-20260724"
)
CANONICAL_MEDIA_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/"
    "v03-action-effect-stage-a-r1-media-20260724"
)

FAILED_STAGE_A_ATTEMPT_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/"
    "v03-action-effect-stage-a-20260724"
)
FAILED_STAGE_A_ATTEMPT_MEDIA_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/"
    "v03-action-effect-stage-a-media-20260724"
)
FAILED_STAGE_A_ATTEMPT_QUALIFICATION_DIRECTORY = Path(
    "/Volumes/T7 Developer/DungeonApprentice/qualifications/"
    "v0.3-action-effect-stage-a-20260724"
)
FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG = Path(
    "/Volumes/T7 Developer/DungeonApprentice/launch-recovery/"
    "v03-action-effect-stage-a-20260724-launcher.log"
)
FAILED_STAGE_A_ATTEMPT_TAG = (
    "action-effect-architecture-v0.3-stage-a-20260724"
)
FAILED_STAGE_A_ATTEMPT_TAG_OBJECT = (
    "9bd59e367b0ccb9890e4ddb5ad0144dfd4897c7c"
)
FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT = (
    "5b135a4e2953db9f14e83cdaba77fe219fecb160"
)
FAILED_STAGE_A_ATTEMPT_QUALIFICATION_REPORT_SHA256 = (
    "a3a50ecf91411a27a70e2c6d3e03b93183aa3b078f2f1104b5dd6884f2c3fc85"
)
FAILED_STAGE_A_ATTEMPT_QUALIFICATION_CLAIM_SHA256 = (
    "1ab64150f7db79735cdd4bb3192cfad2b9ad244a0944d76fae275a05d1e1c39b"
)
FAILED_STAGE_A_ATTEMPT_QUALIFICATION_CHECKSUM_SHA256 = (
    "f96599d4d813ff92897a70077f41a1de16a60ec2860e441eb70ab4af4bb25a4e"
)
FAILED_STAGE_A_ATTEMPT_CONTRACT_SHA256 = (
    "15a180d7a38af6dbc459870b0e4f06a565b291399971fc2165774bce91c6e72c"
)
FAILED_STAGE_A_ATTEMPT_COHORT_SHA256 = (
    "c35441207060ab9a5c55e82289924ab0530effc7f0e5e50fb11b2a0ab83f68eb"
)
FAILED_STAGE_A_ATTEMPT_DASHBOARD_LOG_SHA256 = (
    "8806979b29d3778f8730546c6bb2f577548844cf891cb06220cd935a18bd95f2"
)
FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)

U2S_TERMINAL_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/u2s-ablation-r1-20260723"
)
U2S_TERMINAL_REPORT_SHA256 = (
    "dfd288955bd2f8367ba3818e7242ff29c4a6e8e5a03d248df45e85f43b562e44"
)
U2S_TERMINAL_INTEGRITY_SHA256 = (
    "a9468a504f48911943c43a5a6165917c3dd05697c21abfd784efe9f882b7faa2"
)
U2S_TERMINAL_SOURCE_COMMIT = "2e2a91c9864720326a5fa8ba82212f116d9ead04"
U2S_QUALIFICATION_REPORT_SHA256 = (
    "7fd8fa009191c35ef608767a93e41dfc2c91aca20ae43493a5d1c32624cb467a"
)

PARENT_SIDECAR_SHA256 = u2s_qualify.PARENT_SIDECAR_SHA256
PARENT_MANIFEST_SHA256 = u2s_qualify.PARENT_MANIFEST_SHA256

TAG_SCHEMA_VERSION = 1
TAG_KIND = "v03_action_effect_stage_a_source_preregistration"
TAG_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "protocol",
        "tag",
        "remote",
        "remote_url",
        "source_commit",
        "protocol_document_sha256",
        "parent",
        "u2s_terminal",
        "failed_stage_a_attempt",
        "guard_mapping_sha256",
        "sampler_preflight_sha256",
        "architecture_contract_sha256",
        "protected_partitions_sha256",
        "runtime_contract",
        "roots",
        "dashboard_port",
        "storage_caps",
        "resume_rule",
    }
)

_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "kind",
        "verdict",
        "created_at",
        "claim",
        "source",
        "protocol_document",
        "parent",
        "predecessors",
        "failed_stage_a_attempt",
        "guards",
        "sampler_preflight",
        "architecture_contract",
        "protected_partitions",
        "smoke_evidence",
        "storage_caps",
        "storage_preflight",
        "restrictions",
    }
)
_SHA256 = frozenset("0123456789abcdef")
_GIT_OBJECT_LENGTHS = frozenset({40, 64})
_ARMS = tuple(mode.value for mode in v03.ARM_ORDER)
MAX_REPORT_BYTES = 8 * 1024**2
MAX_SMOKE_BYTES = 8 * 1024**2


class ActionEffectQualificationError(RuntimeError):
    """Raised when Stage A cannot prove its complete pre-action contract."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_set_sha256(values: Sequence[str] | set[str] | frozenset[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(values)):
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in _SHA256 for character in result):
        raise ActionEffectQualificationError(
            f"{label} is not a lowercase SHA-256 digest"
        )
    return result


def _require_git_object(value: Any, label: str) -> str:
    result = str(value)
    if (
        len(result) not in _GIT_OBJECT_LENGTHS
        or any(character not in _SHA256 for character in result)
    ):
        raise ActionEffectQualificationError(f"{label} is not a Git object ID")
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ActionEffectQualificationError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ActionEffectQualificationError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ActionEffectQualificationError(f"{label} is not a JSON object")
    return value


def _regular_file_bytes(path: Path, label: str, *, maximum: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ActionEffectQualificationError(f"{label} is missing or unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size < 1
            or metadata.st_size > maximum
        ):
            raise ActionEffectQualificationError(f"{label} has an invalid file bound")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                raise ActionEffectQualificationError(f"{label} changed while read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ActionEffectQualificationError(f"{label} grew while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _reject_symlink_chain(path: Path) -> None:
    current = path.expanduser().absolute()
    while True:
        if current.is_symlink():
            raise ActionEffectQualificationError(
                f"qualification path contains a symlink: {path}"
            )
        parent = current.parent
        if parent == current:
            return
        current = parent


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _git(
    repository: Path,
    arguments: Sequence[str],
    *,
    runner: Runner = subprocess.run,
) -> str:
    try:
        result = runner(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ActionEffectQualificationError(
            f"cannot verify v0.3 source with git {' '.join(arguments)}"
        ) from error
    return result.stdout.rstrip("\n")


def _storage_caps() -> dict[str, int]:
    return {
        "per_arm_bytes": v03.LINEAGE_CAP_BYTES,
        "scientific_cohort_bytes": v03.COHORT_SCIENTIFIC_CAP_BYTES,
        "media_bytes": v03.MEDIA_CAP_BYTES,
        "combined_bytes": v03.COMBINED_PLANNED_CAP_BYTES,
    }


def _resume_rule() -> dict[str, Any]:
    return {
        "resume_supported": False,
        "interruption_disposition": "whole_stage_a_operationally_incomplete",
        "single_arm_continuation": False,
        "failed_attempt": (
            "v0.3-action-effect-stage-a-20260724-attempt-0"
        ),
        "replacement_attempt": (
            "v0.3-action-effect-stage-a-r1-20260724"
        ),
        "replacement_requires_new_commit_tag_qualification_and_roots": True,
        "replacement_restarts_both_arms_from_confirmed_u1": True,
    }


def _protected_seed_partitions() -> list[dict[str, Any]]:
    protected = [
        partition.public_dict()
        for partition in SEED_PARTITIONS
        if (
            partition.role is U2SeedRole.SEALED_QUALIFICATION
            or "confirmation" in partition.role.value
            or partition.role is U2SeedRole.FINAL_TEST
        )
    ]
    if len(protected) != 10:
        raise ActionEffectQualificationError(
            "v0.3 protected seed inventory changed"
        )
    return protected


def _architecture_contract() -> dict[str, Any]:
    sham = v03.effective_config(ActionEffectMode.SHAM)
    candidate = v03.effective_config(ActionEffectMode.ACTION_EFFECT)
    for config in (sham, candidate):
        if (
            config["parent"]["checkpoint_sha256"]
            != v03.PARENT_CHECKPOINT_SHA256
            or config["parent"]["policy_tensor_sha256"]
            != v03.PARENT_POLICY_TENSOR_SHA256
            or config["parent"]["optimizer_state_sha256"]
            != v03.PARENT_OPTIMIZER_STATE_SHA256
            or config["optimization"]["changed_from_original_u2_control"]
            is not False
            or config["reward"]["changed_from_original_u2_control"] is not False
            or config["selection"]["development_checkpoint_reuse_authorized"]
            is not False
        ):
            raise ActionEffectQualificationError(
                "v0.3 effective configuration changed"
            )
    return {
        "arms": list(_ARMS),
        "same_dict_observation_space": True,
        "same_parameter_set_and_initialization": True,
        "image": {
            "key": IMAGE_KEY,
            "shape_hwc": [56, 56, 3],
            "dtype": "uint8",
        },
        "action_effect": {
            "key": ACTION_EFFECT_KEY,
            "shape": [ACTION_EFFECT_DIM],
            "dtype": "float32",
            "previous_action_one_hot": 7,
            "visible_outcome_one_hot": ["changed", "unchanged"],
            "reset": "all_zero",
            "sham": "always_zero",
        },
        "policy": v03.public_policy_contract(),
        "transplant": {
            "legacy_parameters_by_exact_name": True,
            "legacy_adam_moments_by_exact_parameter_name": True,
            "positional_optimizer_loading": False,
            "orthogonal_reinitialization": False,
            "sole_new_parameter": (
                "features_extractor.action_effect_encoder.weight"
            ),
            "sole_new_parameter_shape": [512, ACTION_EFFECT_DIM],
            "sole_new_parameter_initialization": "exact_zero",
        },
        "equivalence": {
            "zero_context_features_bit_exact": True,
            "zero_context_actions_values_log_probabilities_bit_exact": True,
            "zero_context_recurrent_state_bit_exact": True,
            "first_rollout_transitions": v03.ROLLOUT_TRANSITIONS,
            "behavioral_divergence_before_first_optimizer_phase": False,
        },
        "randomness": {
            "architecture_initialization_seed": (
                v03.ARCHITECTURE_INITIALIZATION_SEED
            ),
            "algorithm_seed": v03.ALGORITHM_SEED,
            "worker_streams": list(v03.WORKER_STREAMS),
        },
        "budget": {
            "child_actions_per_arm": v03.CHILD_ACTION_BUDGET,
            "evaluation_interval": v03.EVALUATION_INTERVAL,
            "exam_count_per_arm": v03.EXAM_COUNT,
            "terminal_lifetime_actions": (
                v03.PARENT_LIFETIME_ACTIONS + v03.CHILD_ACTION_BUDGET
            ),
            "terminal_optimizer_updates": (
                v03.PARENT_OPTIMIZER_UPDATES
                + (v03.CHILD_ACTION_BUDGET // v03.ROLLOUT_TRANSITIONS)
                * v03.PPO_EPOCHS
            ),
        },
        "selection": {
            "sham_selectable": False,
            "action_effect_only_selectable_definition": True,
            "complete_u2s_terminal_three_gate": True,
            "development_checkpoint_reuse_authorized": False,
            "u3_authorized": False,
        },
        "sham_config_sha256": _canonical_sha256(sham),
        "action_effect_config_sha256": _canonical_sha256(candidate),
    }


def _safe_child(root: Path, relative: Any, label: str) -> Path:
    candidate = Path(str(relative))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ActionEffectQualificationError(f"{label} escapes the U2-S root")
    path = (root / candidate).absolute()
    _reject_symlink_chain(path)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve())
    except (OSError, ValueError) as error:
        raise ActionEffectQualificationError(f"{label} is missing or unsafe") from error
    if not resolved.is_file():
        raise ActionEffectQualificationError(f"{label} is missing or unsafe")
    return resolved


def _regular_tree_inventory(root: Path, label: str) -> tuple[set[str], set[str]]:
    """Return regular relative files/directories without following links."""

    if root.is_symlink() or not root.is_dir():
        raise ActionEffectQualificationError(f"{label} is missing or unsafe")
    files: set[str] = set()
    directories: set[str] = set()
    for current, child_directories, child_files in os.walk(
        root,
        followlinks=False,
    ):
        current_path = Path(current)
        for name in child_directories:
            path = current_path / name
            if path.is_symlink() or not path.is_dir():
                raise ActionEffectQualificationError(
                    f"{label} contains an unsafe directory"
                )
            directories.add(path.relative_to(root).as_posix())
        for name in child_files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise ActionEffectQualificationError(
                    f"{label} contains an unsafe file"
                )
            files.add(path.relative_to(root).as_posix())
    return files, directories


def _verify_failed_stage_a_tag(
    repository: Path,
    *,
    runner: Runner = subprocess.run,
) -> None:
    """Require the superseded attempt-0 tag to remain exact on origin."""

    root = repository.expanduser().resolve()
    reference = f"refs/tags/{FAILED_STAGE_A_ATTEMPT_TAG}"
    if (
        _git(root, ["cat-file", "-t", reference], runner=runner) != "tag"
        or _git(root, ["rev-parse", "--verify", reference], runner=runner)
        != FAILED_STAGE_A_ATTEMPT_TAG_OBJECT
        or _git(root, ["rev-list", "-n", "1", reference], runner=runner)
        != FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT
        or _git(
            root,
            ["remote", "get-url", QUALIFIED_REMOTE],
            runner=runner,
        )
        != EXPECTED_ORIGIN_URL
    ):
        raise ActionEffectQualificationError(
            "failed Stage-A attempt annotated tag identity changed"
        )
    remote = _git(
        root,
        [
            "ls-remote",
            "--tags",
            QUALIFIED_REMOTE,
            reference,
        ],
        runner=runner,
    ).split()
    if remote != [FAILED_STAGE_A_ATTEMPT_TAG_OBJECT, reference]:
        raise ActionEffectQualificationError(
            "failed Stage-A attempt annotated tag is not exact on origin"
        )


def authenticate_failed_stage_a_attempt(
    *,
    repository: Path | None = None,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Authenticate the immutable, zero-recorded-action Stage-A attempt 0."""

    if repository is not None:
        _verify_failed_stage_a_tag(repository, runner=runner)

    root = FAILED_STAGE_A_ATTEMPT_ROOT
    media_root = FAILED_STAGE_A_ATTEMPT_MEDIA_ROOT
    qualification_root = FAILED_STAGE_A_ATTEMPT_QUALIFICATION_DIRECTORY
    contract_path = root / "cohort-contract.json"
    cohort_path = root / "cohort.json"
    report_path = qualification_root / "report.json"
    claim_path = qualification_root / "claim.json"
    checksum_path = qualification_root / "report.json.sha256"

    root_files, root_directories = _regular_tree_inventory(
        root,
        "failed Stage-A attempt root",
    )
    media_files, media_directories = _regular_tree_inventory(
        media_root,
        "failed Stage-A attempt media root",
    )
    qualification_files, qualification_directories = _regular_tree_inventory(
        qualification_root,
        "failed Stage-A attempt qualification root",
    )
    _reject_symlink_chain(FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG)
    if (
        root_files
        != {"cohort-contract.json", "cohort.json", "dashboard.log"}
        or root_directories
        or media_files
        or media_directories
        or qualification_files
        != {"claim.json", "report.json", "report.json.sha256"}
        or qualification_directories
        or not FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG.is_file()
        or FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG.stat().st_size != 0
    ):
        raise ActionEffectQualificationError(
            "failed Stage-A attempt artifact inventory changed"
        )
    measured = {
        "contract": file_sha256(contract_path),
        "cohort": file_sha256(cohort_path),
        "qualification_report": file_sha256(report_path),
        "qualification_claim": file_sha256(claim_path),
        "qualification_checksum": file_sha256(checksum_path),
        "dashboard_log": file_sha256(root / "dashboard.log"),
        "launcher_log": file_sha256(FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG),
    }
    expected = {
        "contract": FAILED_STAGE_A_ATTEMPT_CONTRACT_SHA256,
        "cohort": FAILED_STAGE_A_ATTEMPT_COHORT_SHA256,
        "qualification_report": (
            FAILED_STAGE_A_ATTEMPT_QUALIFICATION_REPORT_SHA256
        ),
        "qualification_claim": (
            FAILED_STAGE_A_ATTEMPT_QUALIFICATION_CLAIM_SHA256
        ),
        "qualification_checksum": (
            FAILED_STAGE_A_ATTEMPT_QUALIFICATION_CHECKSUM_SHA256
        ),
        "dashboard_log": FAILED_STAGE_A_ATTEMPT_DASHBOARD_LOG_SHA256,
        "launcher_log": FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG_SHA256,
    }
    if measured != expected:
        raise ActionEffectQualificationError(
            "failed Stage-A attempt checksum changed"
        )

    contract = _read_json(contract_path, "failed Stage-A cohort contract")
    cohort = _read_json(cohort_path, "failed Stage-A cohort state")
    report = _read_json(report_path, "failed Stage-A qualification report")
    claim = _read_json(claim_path, "failed Stage-A qualification claim")
    try:
        checksum_fields = checksum_path.read_text(encoding="ascii").split()
    except (OSError, UnicodeDecodeError) as error:
        raise ActionEffectQualificationError(
            "failed Stage-A qualification checksum is unreadable"
        ) from error

    arms = cohort.get("arms")
    sham = arms[0] if isinstance(arms, list) and len(arms) == 2 else None
    candidate = arms[1] if isinstance(arms, list) and len(arms) == 2 else None
    sham_attempts = sham.get("attempts") if isinstance(sham, Mapping) else None
    sham_attempt = (
        sham_attempts[0]
        if isinstance(sham_attempts, list) and len(sham_attempts) == 1
        else None
    )
    roots = contract.get("roots")
    preregistration = contract.get("preregistration")
    qualification = contract.get("qualification")
    report_source = report.get("source")
    if (
        checksum_fields
        != [
            FAILED_STAGE_A_ATTEMPT_QUALIFICATION_REPORT_SHA256,
            "report.json",
        ]
        or report.get("protocol") != PROTOCOL
        or report.get("verdict") != VERDICT
        or report.get("claim") != claim
        or not isinstance(report_source, Mapping)
        or report_source.get("commit")
        != FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT
        or report_source.get("tag") != FAILED_STAGE_A_ATTEMPT_TAG
        or report_source.get("tag_object")
        != FAILED_STAGE_A_ATTEMPT_TAG_OBJECT
        or claim.get("source_commit")
        != FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT
        or claim.get("tag") != FAILED_STAGE_A_ATTEMPT_TAG
        or claim.get("tag_object") != FAILED_STAGE_A_ATTEMPT_TAG_OBJECT
        or contract.get("protocol") != PROTOCOL
        or contract.get("cohort_id")
        != "v0.3-action-effect-stage-a-20260724"
        or contract.get("source")
        != {"commit": FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT, "dirty": False}
        or not isinstance(preregistration, Mapping)
        or preregistration.get("tag") != FAILED_STAGE_A_ATTEMPT_TAG
        or preregistration.get("tag_object")
        != FAILED_STAGE_A_ATTEMPT_TAG_OBJECT
        or preregistration.get("peeled_commit")
        != FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT
        or not isinstance(qualification, Mapping)
        or qualification.get("report_sha256")
        != FAILED_STAGE_A_ATTEMPT_QUALIFICATION_REPORT_SHA256
        or not isinstance(roots, Mapping)
        or roots.get("cohort") != str(root)
        or roots.get("media") != str(media_root)
        or cohort.get("protocol") != PROTOCOL
        or cohort.get("cohort_id")
        != "v0.3-action-effect-stage-a-20260724"
        or cohort.get("contract_sha256")
        != FAILED_STAGE_A_ATTEMPT_CONTRACT_SHA256
        or cohort.get("source_commit")
        != FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT
        or cohort.get("tag") != FAILED_STAGE_A_ATTEMPT_TAG
        or cohort.get("tag_object") != FAILED_STAGE_A_ATTEMPT_TAG_OBJECT
        or cohort.get("qualification_sha256")
        != FAILED_STAGE_A_ATTEMPT_QUALIFICATION_REPORT_SHA256
        or cohort.get("phase") != "operationally_incomplete"
        or cohort.get("active_arm") is not None
        or cohort.get("terminal_report") is not None
        or cohort.get("process_closeout") is not None
        or not isinstance(sham, Mapping)
        or sham.get("id") != "sham"
        or sham.get("state") != "crashed"
        or sham.get("terminal") is not None
        or not isinstance(sham_attempt, Mapping)
        or sham_attempt.get("index") != 0
        or sham_attempt.get("state") != "crashed"
        or sham_attempt.get("trainer_exit_code") != 130
        or not isinstance(candidate, Mapping)
        or candidate.get("id") != "action-effect"
        or candidate.get("state") != "pending"
        or candidate.get("attempts") != []
        or candidate.get("terminal") is not None
    ):
        raise ActionEffectQualificationError(
            "failed Stage-A attempt disposition changed"
        )

    return {
        "attempt_id": "v0.3-action-effect-stage-a-20260724-attempt-0",
        "protocol": PROTOCOL,
        "disposition": "operationally_incomplete",
        "source_commit": FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT,
        "tag": FAILED_STAGE_A_ATTEMPT_TAG,
        "tag_object": FAILED_STAGE_A_ATTEMPT_TAG_OBJECT,
        "qualification": {
            "root": str(qualification_root),
            "report_sha256": (
                FAILED_STAGE_A_ATTEMPT_QUALIFICATION_REPORT_SHA256
            ),
            "claim_sha256": (
                FAILED_STAGE_A_ATTEMPT_QUALIFICATION_CLAIM_SHA256
            ),
            "checksum_sha256": (
                FAILED_STAGE_A_ATTEMPT_QUALIFICATION_CHECKSUM_SHA256
            ),
            "verdict": "qualified",
        },
        "cohort": {
            "root": str(root),
            "media_root": str(media_root),
            "contract_sha256": FAILED_STAGE_A_ATTEMPT_CONTRACT_SHA256,
            "state_sha256": FAILED_STAGE_A_ATTEMPT_COHORT_SHA256,
            "phase": "operationally_incomplete",
            "classification": "pre_arm_dashboard_health_failure",
            "manifest_synthetic_arm_claim": {
                "arm": "sham",
                "attempt": 0,
                "state": "crashed",
                "exit_code": 130,
            },
            "dashboard_log_sha256": (
                FAILED_STAGE_A_ATTEMPT_DASHBOARD_LOG_SHA256
            ),
        },
        "recorded_training_evidence": {
            "arm_directories": 0,
            "media_entries": 0,
            "trainer_started": False,
            "trainer_status_files": 0,
            "trainer_supervisor_files": 0,
            "checkpoint_files": 0,
            "recorded_child_actions": 0,
            "action_one_reached": False,
        },
        "launcher_log": {
            "path": str(FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG),
            "sha256": FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG_SHA256,
            "bytes": 0,
        },
        "resume_authorized": False,
        "reuse_authorized": False,
        "replacement": {
            "tag": QUALIFIED_TAG,
            "qualification_root": str(CANONICAL_QUALIFICATION_DIRECTORY),
            "cohort_root": str(CANONICAL_COHORT_ROOT),
            "media_root": str(CANONICAL_MEDIA_ROOT),
            "dashboard_port": DASHBOARD_PORT,
            "restarts_both_arms_from_confirmed_u1": True,
        },
    }


def authenticate_u2s_terminal(
    *,
    root: Path = U2S_TERMINAL_ROOT,
    expected_report_sha256: str = U2S_TERMINAL_REPORT_SHA256,
    expected_integrity_sha256: str = U2S_TERMINAL_INTEGRITY_SHA256,
) -> dict[str, Any]:
    """Authenticate terminal U2-S outcome and its four arm report bindings."""

    terminal_root = root.expanduser().resolve()
    report_path = terminal_root / "report.json"
    integrity_path = terminal_root / "report.integrity.json"
    if (
        file_sha256(report_path) != expected_report_sha256
        or file_sha256(integrity_path) != expected_integrity_sha256
    ):
        raise ActionEffectQualificationError("terminal U2-S report bytes changed")
    report = _read_json(report_path, "terminal U2-S report")
    integrity = _read_json(integrity_path, "terminal U2-S report integrity")
    selection = report.get("selection")
    checkpoint_rule = report.get("checkpoint_rule")
    if (
        report.get("protocol") != frozen_u2s.PROTOCOL
        or report.get("verdict") != "ablation_failed"
        or report.get("selected_configuration") is not None
        or report.get("source", {}).get("commit") != U2S_TERMINAL_SOURCE_COMMIT
        or not isinstance(selection, Mapping)
        or selection.get("verdict") != "ablation_failed"
        or selection.get("selected_arm") is not None
        or selection.get("selected_mechanism") is not None
        or selection.get("successor_cohort_authorized") is not False
        or selection.get("ablation_checkpoint_reuse_authorized") is not False
        or not isinstance(checkpoint_rule, Mapping)
        or checkpoint_rule.get("ablation_checkpoint_reuse_authorized") is not False
        or checkpoint_rule.get("successor_checkpoint") is not None
        or checkpoint_rule.get("successor_cohort_authorized") is not False
        or integrity.get("protocol") != frozen_u2s.PROTOCOL
        or integrity.get("report") != report_path.name
        or integrity.get("report_sha256") != expected_report_sha256
    ):
        raise ActionEffectQualificationError(
            "terminal U2-S no-selection verdict changed"
        )
    qualification = report.get("qualification")
    arm_evidence = report.get("arm_evidence")
    if (
        not isinstance(qualification, Mapping)
        or qualification.get("report_sha256")
        != U2S_QUALIFICATION_REPORT_SHA256
        or not isinstance(arm_evidence, Mapping)
        or set(arm_evidence) != set(frozen_u2s.ARM_PRIORITY)
    ):
        raise ActionEffectQualificationError(
            "terminal U2-S qualification or arm inventory changed"
        )
    arms: dict[str, Any] = {}
    for arm in frozen_u2s.ARM_PRIORITY:
        name = arm.value
        binding = arm_evidence.get(name)
        if not isinstance(binding, Mapping):
            raise ActionEffectQualificationError(f"U2-S arm is missing: {name}")
        arm_report_path = _safe_child(
            terminal_root,
            binding.get("report"),
            f"U2-S {name} report",
        )
        arm_integrity_path = _safe_child(
            terminal_root,
            binding.get("report_integrity"),
            f"U2-S {name} report integrity",
        )
        if (
            file_sha256(arm_report_path) != binding.get("report_sha256")
            or file_sha256(arm_integrity_path)
            != binding.get("report_integrity_sha256")
            or int(binding.get("child_trained_actions", -1))
            != frozen_u2s.CHILD_ACTION_BUDGET
            or int(binding.get("exam_count", -1)) != frozen_u2s.EXAM_COUNT
            or binding.get("mechanism_selection_eligible") is not False
            or binding.get("verdict") != "arm_failed"
        ):
            raise ActionEffectQualificationError(
                f"terminal U2-S arm binding changed: {name}"
            )
        arm_report = _read_json(arm_report_path, f"U2-S {name} report")
        arm_integrity = _read_json(
            arm_integrity_path,
            f"U2-S {name} report integrity",
        )
        if (
            arm_report.get("arm") != name
            or arm_report.get("successor_checkpoint_authorized") is not False
            or arm_report.get("resume_authorized") is not False
            or arm_report.get("progress", {}).get("child_trained_actions")
            != frozen_u2s.CHILD_ACTION_BUDGET
            or arm_report.get("progress", {}).get("exam_count")
            != frozen_u2s.EXAM_COUNT
            or arm_integrity.get("report_sha256") != binding.get("report_sha256")
        ):
            raise ActionEffectQualificationError(
                f"terminal U2-S arm report changed: {name}"
            )
        arms[name] = {
            "report": str(arm_report_path),
            "report_sha256": str(binding["report_sha256"]),
            "report_integrity": str(arm_integrity_path),
            "report_integrity_sha256": str(
                binding["report_integrity_sha256"]
            ),
            "child_trained_actions": frozen_u2s.CHILD_ACTION_BUDGET,
            "exam_count": frozen_u2s.EXAM_COUNT,
        }
    return {
        "root": str(terminal_root),
        "report": str(report_path),
        "report_sha256": expected_report_sha256,
        "report_integrity": str(integrity_path),
        "report_integrity_sha256": expected_integrity_sha256,
        "source_commit": U2S_TERMINAL_SOURCE_COMMIT,
        "verdict": "ablation_failed",
        "selected_configuration": None,
        "successor_cohort_authorized": False,
        "checkpoint_reuse_authorized": False,
        "qualification_report_sha256": U2S_QUALIFICATION_REPORT_SHA256,
        "arms": arms,
        "_report": report,
    }


def extend_action_effect_forbidden_layout_hashes(
    base_mapping: Mapping[lessons.LessonId, frozenset[str]],
    terminal_report: Mapping[str, Any],
    *,
    root: Path = U2S_TERMINAL_ROOT,
) -> tuple[Mapping[lessons.LessonId, frozenset[str]], dict[str, Any]]:
    """Add authenticated U2-S training identities to one static Stage-A guard."""

    if set(base_mapping) != set(lessons.LessonId):
        raise ActionEffectQualificationError("base guard inventory is incomplete")
    arm_evidence = terminal_report.get("arm_evidence")
    if not isinstance(arm_evidence, Mapping) or set(arm_evidence) != set(
        frozen_u2s.ARM_PRIORITY
    ):
        raise ActionEffectQualificationError("U2-S arm inventory is incomplete")
    by_lesson: dict[lessons.LessonId, set[str]] = {
        lesson: set() for lesson in lessons.LessonId
    }
    arm_records: dict[str, dict[str, Any]] = {}
    for arm in frozen_u2s.ARM_PRIORITY:
        name = arm.value
        binding = arm_evidence[name]
        arm_report_path = _safe_child(root, binding["report"], f"{name} report")
        if file_sha256(arm_report_path) != binding["report_sha256"]:
            raise ActionEffectQualificationError(f"{name} report digest changed")
        arm_report = _read_json(arm_report_path, f"{name} report")
        training = arm_report.get("training_episode_evidence")
        ledger_binding = (
            training.get("episode_starts")
            if isinstance(training, Mapping)
            else None
        )
        if not isinstance(ledger_binding, Mapping):
            raise ActionEffectQualificationError(f"{name} ledger binding missing")
        ledger_path = _safe_child(
            arm_report_path.parent,
            ledger_binding.get("path"),
            f"{name} episode-start ledger",
        )
        if file_sha256(ledger_path) != ledger_binding.get("sha256"):
            raise ActionEffectQualificationError(f"{name} ledger digest changed")
        records = 0
        try:
            with ledger_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    value = json.loads(line)
                    if (
                        not isinstance(value, Mapping)
                        or value.get("type") != "episode_start"
                    ):
                        raise ActionEffectQualificationError(
                            f"{name} ledger contains an invalid record"
                        )
                    try:
                        lesson = lessons.LessonId(str(value.get("lesson_id")))
                    except ValueError as error:
                        raise ActionEffectQualificationError(
                            f"{name} ledger contains an unknown lesson"
                        ) from error
                    by_lesson[lesson].add(
                        _require_sha256(
                            value.get("layout_sha256"),
                            f"{name} layout",
                        )
                    )
                    records += 1
        except (OSError, json.JSONDecodeError) as error:
            raise ActionEffectQualificationError(
                f"cannot parse {name} episode-start ledger"
            ) from error
        if records != int(ledger_binding.get("record_count", -1)):
            raise ActionEffectQualificationError(f"{name} ledger count changed")
        active = training.get("terminal_active_workers")
        if (
            not isinstance(active, list)
            or len(active) != v03.WORKERS
            or _canonical_sha256(active)
            != training.get("terminal_active_workers_sha256")
        ):
            raise ActionEffectQualificationError(
                f"{name} terminal-active inventory changed"
            )
        for value in active:
            try:
                lesson = lessons.LessonId(str(value.get("lesson_id")))
            except (AttributeError, ValueError) as error:
                raise ActionEffectQualificationError(
                    f"{name} terminal-active lesson changed"
                ) from error
            by_lesson[lesson].add(
                _require_sha256(value.get("layout_sha256"), f"{name} active layout")
            )
        arm_records[name] = {
            "report_sha256": str(binding["report_sha256"]),
            "episode_start_ledger": str(ledger_path),
            "episode_start_ledger_sha256": str(ledger_binding["sha256"]),
            "episode_start_records": records,
            "terminal_active_workers": len(active),
            "terminal_active_workers_sha256": str(
                training["terminal_active_workers_sha256"]
            ),
        }

    applied: dict[lessons.LessonId, frozenset[str]] = {}
    guard_sets: dict[str, dict[str, Any]] = {}
    for lesson in lessons.LessonId:
        values = set(base_mapping[lesson])
        if lesson is not lessons.LessonId.VISIBLE_UNLOCK:
            values.update(by_lesson[lesson])
        applied[lesson] = frozenset(values)
        guard_sets[lesson.value] = {
            "count": len(values),
            "sha256": _hash_set_sha256(values),
            "rule": (
                "development_only_history_overlap_diagnostic"
                if lesson is lessons.LessonId.VISIBLE_UNLOCK
                else "same_lesson_history_through_terminal_u2s"
            ),
            "u2s_unique_layouts_seen": len(by_lesson[lesson]),
        }
    if applied[lessons.LessonId.VISIBLE_UNLOCK] != base_mapping[
        lessons.LessonId.VISIBLE_UNLOCK
    ]:
        raise ActionEffectQualificationError("Stage A changed the finite U0 guard")
    frozen = MappingProxyType(applied)
    mapping_sha256 = _canonical_sha256(
        {lesson.value: sorted(frozen[lesson]) for lesson in lessons.LessonId}
    )
    return frozen, {
        "base_mapping_sha256": _canonical_sha256(
            {
                lesson.value: sorted(base_mapping[lesson])
                for lesson in lessons.LessonId
            }
        ),
        "u2s_terminal_report_sha256": U2S_TERMINAL_REPORT_SHA256,
        "u2s_arms": arm_records,
        "guard_sets": guard_sets,
        "applied_mapping_sha256": mapping_sha256,
        "frozen_before_sham": True,
        "sham_history_added_before_action_effect": False,
    }


def _authenticate_parent_and_guards(
    repository: Path,
) -> tuple[
    dict[str, Any],
    Any,
    Mapping[lessons.LessonId, frozenset[str]],
    dict[str, Any],
    dict[str, Any],
]:
    (
        parent,
        base_qualification,
        failed_u2_parent,
        u2r_terminal,
        failed_u2r_r0,
    ) = u2s_qualify._authenticate_predecessors(repository)
    parent_public = u2s_qualify._verify_parent_model(parent)
    if (
        parent_public["checkpoint_sha256"] != v03.PARENT_CHECKPOINT_SHA256
        or parent_public["sidecar_sha256"] != PARENT_SIDECAR_SHA256
        or parent_public["manifest_sha256"] != PARENT_MANIFEST_SHA256
        or parent_public["policy_tensor_sha256"]
        != v03.PARENT_POLICY_TENSOR_SHA256
        or parent_public["optimizer_state_sha256"]
        != v03.PARENT_OPTIMIZER_STATE_SHA256
        or parent_public["u1_child_seed"] != v03.PARENT_U1_SEED
        or parent_public["trained_timesteps"] != v03.PARENT_LIFETIME_ACTIONS
        or parent_public["n_updates"] != v03.PARENT_OPTIMIZER_UPDATES
        or parent_public["confirmation_verdict"] != "confirmed"
    ):
        raise ActionEffectQualificationError(
            "confirmed U1 parent provenance or model state changed"
        )
    seed_access = base_qualification.seed_access()
    base_mapping, base_guard_evidence = (
        u2s_qualify.build_u2s_forbidden_layout_hashes(
            base_qualification.verified_report(),
            failed_u2_parent.confirmation_snapshot(),
            access=seed_access,
            terminal_report=u2r_terminal,
        )
    )
    u2s_terminal = authenticate_u2s_terminal()
    mapping, guards = extend_action_effect_forbidden_layout_hashes(
        base_mapping,
        u2s_terminal["_report"],
    )
    terminal_public = {
        key: value for key, value in u2s_terminal.items() if key != "_report"
    }
    predecessors = {
        "base_qualification": base_qualification.public_dict(),
        "u2_confirmation": failed_u2_parent.public_dict(),
        "u2r_terminal": {
            "report_sha256": u2s_qualify.U2R_REPORT_SHA256,
            "verdict": u2r_terminal["verdict"],
            "eligible_for_fresh_confirmation": u2r_terminal[
                "eligible_for_fresh_confirmation"
            ],
            "failed_r0_launch": failed_u2r_r0,
        },
        "u2s_terminal": terminal_public,
        "policy_updates_during_authentication": False,
    }
    guards["pre_u2s"] = base_guard_evidence
    return parent_public, base_qualification, mapping, guards, predecessors


def _sampler_preflight(
    mapping: Mapping[lessons.LessonId, frozenset[str]],
    *,
    seed_access: Any,
) -> list[dict[str, Any]]:
    records = frozen_u2r.preflight_u2r_training_layout_sampler(
        mapping,
        seed_access=seed_access,
        worker_streams=v03.WORKER_STREAMS,
        max_attempts=frozen_u2s.LAYOUT_RESAMPLE_ATTEMPTS,
    )
    public = [dict(record) for record in records]
    if len(public) != len(lessons.LessonId) * v03.WORKERS:
        raise ActionEffectQualificationError(
            "v0.3 sampler preflight is incomplete"
        )
    streams = {int(record.get("worker_stream", -1)) for record in public}
    if streams != set(v03.WORKER_STREAMS):
        raise ActionEffectQualificationError(
            "v0.3 sampler preflight worker streams changed"
        )
    return public


def _validate_rng_identity(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "phase",
        "components",
        "aggregate_sha256",
    }:
        raise ActionEffectQualificationError(f"{label} is missing")
    components = value.get("components")
    if (
        value.get("phase")
        not in {"post_reset_pre_action_one", "post_rollout_pre_optimizer"}
        or not isinstance(components, Mapping)
        or value.get("aggregate_sha256") != _canonical_sha256(components)
    ):
        raise ActionEffectQualificationError(f"{label} changed")
    expected_component_keys = {
        "python_random_sha256",
        "numpy_global_sha256",
        "torch_cpu_sha256",
        "model_action_space",
        "scheduler_sha256",
        "workers",
    }
    if set(components) != expected_component_keys:
        raise ActionEffectQualificationError(f"{label} components changed")
    for key in (
        "python_random_sha256",
        "numpy_global_sha256",
        "torch_cpu_sha256",
        "scheduler_sha256",
    ):
        _require_sha256(components[key], f"{label} {key}")
    _validate_space_rng_identity(
        components["model_action_space"],
        f"{label} model action space",
        expected_type="Discrete",
    )
    workers = components["workers"]
    if not isinstance(workers, list) or len(workers) != v03.WORKERS:
        raise ActionEffectQualificationError(f"{label} workers changed")
    for index, worker in enumerate(workers):
        if not isinstance(worker, Mapping) or set(worker) != {
            "worker_index",
            "worker_stream",
            "curriculum_sha256",
            "base_environment_sha256",
            "action_space",
            "observation_space",
        }:
            raise ActionEffectQualificationError(
                f"{label} worker schema changed"
            )
        if (
            worker.get("worker_index") != index
            or worker.get("worker_stream") != v03.WORKER_STREAMS[index]
        ):
            raise ActionEffectQualificationError(
                f"{label} worker identity changed"
            )
        _require_sha256(
            worker["curriculum_sha256"],
            f"{label} worker curriculum",
        )
        _require_sha256(
            worker["base_environment_sha256"],
            f"{label} worker environment",
        )
        _validate_space_rng_identity(
            worker["action_space"],
            f"{label} worker action space",
            expected_type="Discrete",
        )
        _validate_space_rng_identity(
            worker["observation_space"],
            f"{label} worker observation space",
            expected_type="Dict",
        )
    return json.loads(json.dumps(dict(value)))


def _validate_space_rng_identity(
    value: Any,
    label: str,
    *,
    expected_type: str | None = None,
) -> None:
    if not isinstance(value, Mapping):
        raise ActionEffectQualificationError(f"{label} is missing")
    kind = value.get("type")
    expected_keys = {"type", "self_sha256"}
    if kind == "Dict":
        expected_keys.add("children")
    if (
        set(value) != expected_keys
        or not isinstance(kind, str)
        or (expected_type is not None and kind != expected_type)
    ):
        raise ActionEffectQualificationError(f"{label} schema changed")
    _require_sha256(value["self_sha256"], f"{label} state")
    if kind == "Dict":
        children = value["children"]
        if not isinstance(children, Mapping) or set(children) != {
            IMAGE_KEY,
            ACTION_EFFECT_KEY,
        }:
            raise ActionEffectQualificationError(
                f"{label} Dict children changed"
            )
        _validate_space_rng_identity(
            children[IMAGE_KEY],
            f"{label} image child",
            expected_type="Box",
        )
        _validate_space_rng_identity(
            children[ACTION_EFFECT_KEY],
            f"{label} action-effect child",
            expected_type="Box",
        )


def _validate_sampler_preflight(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != (
        len(lessons.LessonId) * v03.WORKERS
    ):
        raise ActionEffectQualificationError(
            "v0.3 smoke sampler preflight is incomplete"
        )
    expected_pairs = [
        (lesson.value, stream)
        for lesson in lessons.LessonId
        for stream in v03.WORKER_STREAMS
    ]
    normalized: list[dict[str, Any]] = []
    for record, expected_pair in zip(value, expected_pairs, strict=True):
        if not isinstance(record, Mapping) or set(record) != {
            "lesson_id",
            "worker_stream",
            "accepted_seed",
            "accepted_attempt",
            "layout_sha256",
            "forbidden_layouts",
            "max_attempts",
        }:
            raise ActionEffectQualificationError(
                "v0.3 smoke sampler record schema changed"
            )
        pair = (record.get("lesson_id"), record.get("worker_stream"))
        if (
            pair != expected_pair
            or not isinstance(record.get("accepted_seed"), int)
            or not 0 <= int(record["accepted_seed"]) < 1_000_000
            or not isinstance(record.get("accepted_attempt"), int)
            or not 1 <= int(record["accepted_attempt"]) <= 8_192
            or not isinstance(record.get("forbidden_layouts"), int)
            or int(record["forbidden_layouts"]) < 0
            or record.get("max_attempts") != 8_192
        ):
            raise ActionEffectQualificationError(
                "v0.3 smoke sampler record changed"
            )
        _require_sha256(
            record["layout_sha256"],
            "v0.3 smoke sampler layout",
        )
        normalized.append(json.loads(json.dumps(dict(record))))
    return normalized


def _validate_smoke_workers(value: Any, arm_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != v03.WORKERS:
        raise ActionEffectQualificationError(
            f"v0.3 smoke workers changed: {arm_name}"
        )
    normalized: list[dict[str, Any]] = []
    expected_keys = {
        "worker_index",
        "transitions",
        "episodes_started",
        "episode_starts_sha256",
        "action_counts",
        "lesson_transition_counts",
        "reward_total_hex",
        "extrinsic_total_hex",
        "curiosity_total_hex",
        "penalty_total_hex",
        "penalty_events",
        "penalty_eligible_events",
        "maximum_episode_penalty_count",
        "trajectory_sha256",
        "reward_evidence_sha256",
    }
    for index, worker in enumerate(value):
        if not isinstance(worker, Mapping) or set(worker) != expected_keys:
            raise ActionEffectQualificationError(
                f"v0.3 smoke worker schema changed: {arm_name}"
            )
        actions = worker["action_counts"]
        lesson_counts = worker["lesson_transition_counts"]
        if (
            worker.get("worker_index") != index
            or worker.get("transitions") != v03.ROLLOUT_STEPS
            or not isinstance(worker.get("episodes_started"), int)
            or int(worker["episodes_started"]) < 1
            or not isinstance(actions, Mapping)
            or set(actions) != {str(action) for action in range(7)}
            or any(not isinstance(count, int) or count < 0 for count in actions.values())
            or sum(actions.values()) != v03.ROLLOUT_STEPS
            or not isinstance(lesson_counts, Mapping)
            or set(lesson_counts)
            != {lesson.value for lesson in lessons.LessonId}
            or any(
                not isinstance(count, int) or count < 0
                for count in lesson_counts.values()
            )
            or sum(lesson_counts.values()) != v03.ROLLOUT_STEPS
            or worker.get("penalty_events") != 0
            or worker.get("penalty_eligible_events") != 0
            or worker.get("maximum_episode_penalty_count") != 0
        ):
            raise ActionEffectQualificationError(
                f"v0.3 smoke worker totals changed: {arm_name}"
            )
        for key in (
            "episode_starts_sha256",
            "trajectory_sha256",
            "reward_evidence_sha256",
        ):
            _require_sha256(worker[key], f"{arm_name} worker {key}")
        for key in (
            "reward_total_hex",
            "extrinsic_total_hex",
            "curiosity_total_hex",
            "penalty_total_hex",
        ):
            try:
                measured = float.fromhex(str(worker[key]))
            except ValueError as error:
                raise ActionEffectQualificationError(
                    f"{arm_name} worker {key} is invalid"
                ) from error
            if not math.isfinite(measured):
                raise ActionEffectQualificationError(
                    f"{arm_name} worker {key} is not finite"
                )
        if float.fromhex(str(worker["penalty_total_hex"])) != 0.0:
            raise ActionEffectQualificationError(
                f"{arm_name} smoke used reward shaping"
            )
        normalized.append(json.loads(json.dumps(dict(worker))))
    return normalized


def _validate_seed_evidence(
    value: Any,
    *,
    episode_starts: int,
    arm_name: str,
) -> dict[str, Any]:
    expected_keys = {
        "episode_starts",
        "minimum_seed",
        "maximum_seed",
        "training_range_only",
        "separated_unlock_roles",
        "protected_roles",
        "protected_seed_hits",
        "confirmation_or_final_seed_generated",
    }
    expected_protected = sorted(
        role.value for role in smoke.frozen_smoke._PROTECTED_ROLES
    )
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_keys
        or value.get("episode_starts") != episode_starts
        or not isinstance(value.get("minimum_seed"), int)
        or not isinstance(value.get("maximum_seed"), int)
        or not 0 <= int(value["minimum_seed"])
        <= int(value["maximum_seed"]) < 1_000_000
        or value.get("training_range_only") is not True
        or value.get("separated_unlock_roles") not in ([], ["training"])
        or value.get("protected_roles") != expected_protected
        or value.get("protected_seed_hits") != []
        or value.get("confirmation_or_final_seed_generated") is not False
    ):
        raise ActionEffectQualificationError(
            f"v0.3 smoke seed evidence changed: {arm_name}"
        )
    return json.loads(json.dumps(dict(value)))


def validate_disposable_smoke(
    value: Any,
    *,
    source_commit: str,
    guard_mapping_sha256: str,
    sampler_preflight_sha256: str,
) -> dict[str, Any]:
    """Validate the destroyed matched rollout and return its bound summary."""

    if not isinstance(value, Mapping):
        raise ActionEffectQualificationError("v0.3 disposable smoke is missing")
    expected_top_level = {
        "schema_version",
        "protocol",
        "completed_at",
        "verdict",
        "source",
        "parent_checkpoint",
        "parent_checkpoint_sha256",
        "worker_streams",
        "guard_mapping_sha256",
        "sampler_preflight",
        "sampler_preflight_sha256",
        "actions_per_arm",
        "pre_action_rng_identical",
        "pre_update_behavior_identical",
        "first_rollout_trajectory_identical",
        "first_rollout_policy_outputs_identical",
        "post_rollout_pre_optimizer_rng_identical",
        "first_rollout_episode_ledger_identical",
        "learning_divergence_begins_after_first_update",
        "arms",
        "canonical_predecessor_roots_unchanged",
        "temporary_root_removed",
        "scientific_evidence",
        "checkpoint_reuse_authorized",
    }
    arms = value.get("arms")
    source = value.get("source")
    sampler = _validate_sampler_preflight(value.get("sampler_preflight"))
    if (
        set(value) != expected_top_level
        or value.get("schema_version") != smoke.SCHEMA_VERSION
        or value.get("protocol") != smoke.SMOKE_PROTOCOL
        or value.get("verdict") != "passed"
        or not isinstance(source, Mapping)
        or set(source) != {"commit", "branch", "dirty"}
        or source.get("commit") != source_commit
        or not isinstance(source.get("branch"), str)
        or source.get("dirty") is not False
        or value.get("parent_checkpoint") != str(v03.PARENT_CHECKPOINT)
        or value.get("parent_checkpoint_sha256")
        != v03.PARENT_CHECKPOINT_SHA256
        or value.get("worker_streams") != list(v03.WORKER_STREAMS)
        or value.get("guard_mapping_sha256")
        != guard_mapping_sha256
        or value.get("sampler_preflight_sha256")
        != sampler_preflight_sha256
        or value.get("sampler_preflight_sha256")
        != _canonical_sha256(sampler)
        or int(value.get("actions_per_arm", -1))
        != v03.ROLLOUT_TRANSITIONS
        or not isinstance(arms, Mapping)
        or set(arms) != set(_ARMS)
        or any(
            value.get(label) is not True
            for label in (
                "pre_action_rng_identical",
                "pre_update_behavior_identical",
                "first_rollout_trajectory_identical",
                "first_rollout_policy_outputs_identical",
                "post_rollout_pre_optimizer_rng_identical",
                "first_rollout_episode_ledger_identical",
                "learning_divergence_begins_after_first_update",
                "canonical_predecessor_roots_unchanged",
                "temporary_root_removed",
            )
        )
        or value.get("scientific_evidence") is not False
        or value.get("checkpoint_reuse_authorized") is not False
    ):
        raise ActionEffectQualificationError(
            "v0.3 disposable smoke contract changed"
        )
    sham = arms[ActionEffectMode.SHAM.value]
    candidate = arms[ActionEffectMode.ACTION_EFFECT.value]
    if not isinstance(sham, Mapping) or not isinstance(candidate, Mapping):
        raise ActionEffectQualificationError("v0.3 smoke arms are invalid")
    matched_fields = (
        "before",
        "workers",
        "trajectory_identity",
        "pre_action_rng_identity",
        "policy_output_sha256",
        "post_rollout_rng_identity",
        "episode_ledger",
        "seed_evidence",
    )
    if any(sham.get(field) != candidate.get(field) for field in matched_fields):
        raise ActionEffectQualificationError(
            "matched twins diverged before the first optimizer phase"
        )
    before = sham.get("before")
    if (
        not isinstance(before, Mapping)
        or before.get("trained_timesteps") != v03.PARENT_LIFETIME_ACTIONS
        or before.get("optimizer_updates") != v03.PARENT_OPTIMIZER_UPDATES
        or before.get("policy_tensor_sha256")
        == sham.get("after", {}).get("policy_tensor_sha256")
        or before.get("optimizer_state_sha256")
        == sham.get("after", {}).get("optimizer_state_sha256")
        or sham.get("effect_projection_nonzero_parameters") != 0
        or int(candidate.get("effect_projection_nonzero_parameters", 0)) <= 0
    ):
        raise ActionEffectQualificationError(
            "v0.3 smoke transplant/update boundary changed"
        )
    for name, arm in (
        (ActionEffectMode.SHAM.value, sham),
        (ActionEffectMode.ACTION_EFFECT.value, candidate),
    ):
        after = arm.get("after")
        transplant = arm.get("transplant")
        equivalence = (
            transplant.get("zero_context_equivalence")
            if isinstance(transplant, Mapping)
            else None
        )
        transplant_record = (
            transplant.get("transplant")
            if isinstance(transplant, Mapping)
            else None
        )
        workers = _validate_smoke_workers(arm.get("workers"), name)
        trajectory = arm.get("trajectory_identity")
        expected_trajectory = smoke._trajectory_identity(workers)
        ledger = arm.get("episode_ledger")
        episode_starts = sum(
            int(worker["episodes_started"]) for worker in workers
        )
        if (
            not isinstance(arm, Mapping)
            or set(arm)
            != {
                "arm",
                "before",
                "after",
                "effect_projection_nonzero_parameters",
                "transplant",
                "workers",
                "trajectory_identity",
                "pre_action_rng_identity",
                "policy_output_sha256",
                "post_rollout_rng_identity",
                "episode_ledger",
                "seed_evidence",
                "updated_archive_reloaded_exactly",
                "development_checkpoint_reuse_authorized",
            }
            or arm.get("arm") != name
            or not isinstance(after, Mapping)
            or after.get("trained_timesteps")
            != v03.PARENT_LIFETIME_ACTIONS + v03.ROLLOUT_TRANSITIONS
            or after.get("optimizer_updates")
            != v03.PARENT_OPTIMIZER_UPDATES + v03.PPO_EPOCHS
            or not isinstance(transplant, Mapping)
            or not isinstance(equivalence, Mapping)
            or any(
                equivalence.get(key) is not True
                for key in (
                    "features_exact",
                    "actions_exact",
                    "values_exact",
                    "log_probabilities_exact",
                    "recurrent_states_exact",
                )
            )
            or not isinstance(transplant_record, Mapping)
            or transplant_record.get("missing_optimizer_state_names")
            not in ([], ())
            or transplant_record.get("effect_encoder_zero") is not True
            or arm.get("updated_archive_reloaded_exactly") is not True
            or arm.get("development_checkpoint_reuse_authorized") is not False
            or trajectory != expected_trajectory
            or not isinstance(ledger, Mapping)
            or set(ledger) != {"records", "normalized_sha256"}
            or ledger.get("records") != episode_starts
        ):
            raise ActionEffectQualificationError(
                f"v0.3 smoke arm contract changed: {name}"
            )
        _require_sha256(
            ledger["normalized_sha256"],
            f"{name} first-rollout episode ledger",
        )
        _require_sha256(
            arm["policy_output_sha256"],
            f"{name} first-rollout policy output",
        )
        pre_action = _validate_rng_identity(
            arm["pre_action_rng_identity"],
            f"{name} pre-action RNG identity",
        )
        if pre_action["phase"] != "post_reset_pre_action_one":
            raise ActionEffectQualificationError(
                f"{name} pre-action RNG phase changed"
            )
        post_rollout = _validate_rng_identity(
            arm["post_rollout_rng_identity"],
            f"{name} post-rollout RNG identity",
        )
        if post_rollout["phase"] != "post_rollout_pre_optimizer":
            raise ActionEffectQualificationError(
                f"{name} post-rollout RNG phase changed"
            )
        _validate_seed_evidence(
            arm["seed_evidence"],
            episode_starts=episode_starts,
            arm_name=name,
        )
    pre_action = _validate_rng_identity(
        sham["pre_action_rng_identity"],
        "v0.3 pre-action RNG identity",
    )
    _require_sha256(sham["policy_output_sha256"], "first-rollout policy output")
    _require_sha256(
        sham["episode_ledger"]["normalized_sha256"],
        "first-rollout episode ledger",
    )
    return {
        "protocol": smoke.SMOKE_PROTOCOL,
        "verdict": "passed",
        "source_commit": source_commit,
        "guard_mapping_sha256": guard_mapping_sha256,
        "sampler_preflight_sha256": sampler_preflight_sha256,
        "actions_per_arm": v03.ROLLOUT_TRANSITIONS,
        "pre_action_rng_identity": pre_action,
        "pre_action_rng_identity_sha256": _canonical_sha256(pre_action),
        "first_rollout_policy_output_sha256": str(
            sham["policy_output_sha256"]
        ),
        "first_rollout_trajectory_sha256": _canonical_sha256(
            sham["trajectory_identity"]
        ),
        "first_rollout_episode_ledger_sha256": str(
            sham["episode_ledger"]["normalized_sha256"]
        ),
        "post_rollout_rng_identity_sha256": _canonical_sha256(
            sham["post_rollout_rng_identity"]
        ),
        "transplant_sha256": _canonical_sha256(sham["transplant"]),
        "matched_through_first_rollout": True,
        "optimizer_phase_completed_for_both": True,
        "candidate_context_projection_updated": True,
        "sham_context_projection_remained_zero": True,
        "temporary_root_removed": True,
        "checkpoint_reuse_authorized": False,
        "full_report_sha256": _canonical_sha256(value),
    }


def _storage_preflight(
    *,
    require_qualification_absent: bool,
    permit_bound_managed_roots: bool = False,
) -> dict[str, Any]:
    volume = Path("/Volumes/T7 Developer")
    dungeon_root = volume / "DungeonApprentice"
    _reject_symlink_chain(volume)
    _reject_symlink_chain(dungeon_root)
    try:
        volume_stat = volume.stat()
        parent_stat = volume.parent.stat()
        dungeon_stat = dungeon_root.stat()
    except OSError as error:
        raise ActionEffectQualificationError(
            "cannot inspect Stage-A storage"
        ) from error
    if (
        not stat.S_ISDIR(volume_stat.st_mode)
        or not stat.S_ISDIR(dungeon_stat.st_mode)
        or not os.path.ismount(volume)
        or volume_stat.st_dev == parent_stat.st_dev
        or dungeon_stat.st_dev != volume_stat.st_dev
    ):
        raise ActionEffectQualificationError(
            "T7 Developer is not a separate mounted volume"
        )
    fresh = (CANONICAL_COHORT_ROOT, CANONICAL_MEDIA_ROOT)
    managed_roots_absent = True
    for path in fresh:
        if path.is_symlink():
            raise ActionEffectQualificationError(
                "Stage-A cohort or media root is unsafe"
            )
        if path.exists():
            managed_roots_absent = False
            if not permit_bound_managed_roots:
                raise ActionEffectQualificationError(
                    "Stage-A cohort or media root is not fresh"
                )
            try:
                metadata = path.stat()
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise ActionEffectQualificationError(
                    "cannot authenticate a bound Stage-A managed root"
                ) from error
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or resolved != path
                or metadata.st_dev != volume_stat.st_dev
            ):
                raise ActionEffectQualificationError(
                    "Stage-A managed root changed device or type"
                )
    if require_qualification_absent and (
        CANONICAL_QUALIFICATION_DIRECTORY.exists()
        or CANONICAL_QUALIFICATION_DIRECTORY.is_symlink()
    ):
        raise ActionEffectQualificationError(
            "Stage-A qualification root is not fresh"
        )
    protected = (
        v03.PARENT_CHECKPOINT,
        frozen_u2.CANONICAL_U1_CONFIRMATION,
        U2S_TERMINAL_ROOT,
        u2s_qualify.U2R_RUN_DIRECTORY,
        FAILED_STAGE_A_ATTEMPT_QUALIFICATION_DIRECTORY,
        FAILED_STAGE_A_ATTEMPT_ROOT,
        FAILED_STAGE_A_ATTEMPT_MEDIA_ROOT,
    )
    managed = (
        CANONICAL_QUALIFICATION_DIRECTORY,
        CANONICAL_COHORT_ROOT,
        CANONICAL_MEDIA_ROOT,
    )
    for artifact in protected:
        resolved = artifact.expanduser().resolve()
        if any(resolved == root or root in resolved.parents for root in managed):
            raise ActionEffectQualificationError(
                "Stage-A roots overlap predecessor evidence"
            )
    usage = shutil.disk_usage(volume)
    minimum = int(v03.MINIMUM_FREE_GIB * 1024**3)
    if usage.free < minimum:
        raise ActionEffectQualificationError(
            "T7 Developer lacks the Stage-A free-space reserve"
        )
    return {
        "volume": str(volume),
        "dungeon_root": str(dungeon_root),
        "separate_mounted_device": True,
        "volume_device": int(volume_stat.st_dev),
        "parent_device": int(parent_stat.st_dev),
        "managed_roots": [str(path) for path in managed],
        "cohort_and_media_roots_absent": managed_roots_absent,
        "bound_managed_roots_permitted": permit_bound_managed_roots,
        "qualification_root_absent_before_claim": require_qualification_absent,
        "protected_artifacts_outside_managed_roots": True,
        "minimum_free_bytes": minimum,
        "measured_free_bytes": int(usage.free),
        "measured_total_bytes": int(usage.total),
        "passed": True,
        "measured_at": utc_now(),
    }


def _verify_storage_preflight(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ActionEffectQualificationError("storage preflight is missing")
    live = _storage_preflight(
        require_qualification_absent=False,
        permit_bound_managed_roots=True,
    )
    if (
        value.get("volume") != live["volume"]
        or value.get("dungeon_root") != live["dungeon_root"]
        or value.get("separate_mounted_device") is not True
        or value.get("volume_device") != live["volume_device"]
        or value.get("parent_device") != live["parent_device"]
        or value.get("managed_roots") != live["managed_roots"]
        or value.get("cohort_and_media_roots_absent") is not True
        or value.get("bound_managed_roots_permitted") is not False
        or value.get("qualification_root_absent_before_claim") is not True
        or value.get("protected_artifacts_outside_managed_roots") is not True
        or value.get("minimum_free_bytes") != live["minimum_free_bytes"]
        or not isinstance(value.get("measured_free_bytes"), int)
        or int(value["measured_free_bytes"]) < int(value["minimum_free_bytes"])
        or value.get("passed") is not True
    ):
        raise ActionEffectQualificationError("storage preflight changed")
    try:
        timestamp = datetime.fromisoformat(str(value["measured_at"]))
    except (KeyError, ValueError) as error:
        raise ActionEffectQualificationError(
            "storage preflight timestamp is invalid"
        ) from error
    if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(None):
        raise ActionEffectQualificationError(
            "storage preflight timestamp must be UTC"
        )


def _collect_static_inputs(
    repository: Path,
) -> tuple[
    dict[str, Any],
    Any,
    Mapping[lessons.LessonId, frozenset[str]],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    parent, base, mapping, guards, predecessors = (
        _authenticate_parent_and_guards(repository)
    )
    sampler = _sampler_preflight(mapping, seed_access=base.seed_access())
    return parent, base, mapping, guards, predecessors, sampler


def build_action_effect_tag_payload(
    repository: Path,
    *,
    source_commit: str,
) -> dict[str, Any]:
    """Build the canonical one-line preregistration tag payload."""

    commit = _require_git_object(source_commit, "v0.3 source commit")
    protocol_path = repository.expanduser().resolve() / PROTOCOL_DOCUMENT
    if protocol_path.is_symlink() or not protocol_path.is_file():
        raise ActionEffectQualificationError(
            "v0.3 protocol document is missing or unsafe"
        )
    parent, _base, _mapping, guards, predecessors, sampler = (
        _collect_static_inputs(repository)
    )
    protected = _protected_seed_partitions()
    contract = _architecture_contract()
    failed_attempt = authenticate_failed_stage_a_attempt(
        repository=repository,
    )
    payload = {
        "schema_version": TAG_SCHEMA_VERSION,
        "kind": TAG_KIND,
        "protocol": PROTOCOL,
        "tag": QUALIFIED_TAG,
        "remote": QUALIFIED_REMOTE,
        "remote_url": EXPECTED_ORIGIN_URL,
        "source_commit": commit,
        "protocol_document_sha256": file_sha256(protocol_path),
        "parent": {
            key: parent[key]
            for key in (
                "checkpoint_sha256",
                "sidecar_sha256",
                "manifest_sha256",
                "policy_tensor_sha256",
                "optimizer_state_sha256",
                "confirmation_sha256",
                "trained_timesteps",
                "n_updates",
            )
        },
        "u2s_terminal": {
            key: predecessors["u2s_terminal"][key]
            for key in (
                "report_sha256",
                "report_integrity_sha256",
                "verdict",
                "selected_configuration",
                "successor_cohort_authorized",
                "checkpoint_reuse_authorized",
            )
        },
        "failed_stage_a_attempt": failed_attempt,
        "guard_mapping_sha256": guards["applied_mapping_sha256"],
        "sampler_preflight_sha256": _canonical_sha256(sampler),
        "architecture_contract_sha256": _canonical_sha256(contract),
        "protected_partitions_sha256": _canonical_sha256(protected),
        "runtime_contract": {
            "snapshot": runtime_snapshot(),
            "training_device": "cpu",
            "qualification_smoke_device": "cpu",
        },
        "roots": {
            "qualification": str(CANONICAL_QUALIFICATION_DIRECTORY),
            "cohort": str(CANONICAL_COHORT_ROOT),
            "media": str(CANONICAL_MEDIA_ROOT),
        },
        "dashboard_port": DASHBOARD_PORT,
        "storage_caps": _storage_caps(),
        "resume_rule": _resume_rule(),
    }
    if set(payload) != TAG_FIELDS:
        raise ActionEffectQualificationError(
            "internal v0.3 tag payload is incomplete"
        )
    return payload


def verify_source_tag(
    repository: Path,
    *,
    expected_source_commit: str | None = None,
    expected_tag_object: str | None = None,
    expected_tag_payload: Mapping[str, Any] | None = None,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Verify one clean, pushed commit and its published annotated tag."""

    root = repository.expanduser().resolve()
    head = _require_git_object(
        _git(root, ["rev-parse", "--verify", "HEAD^{commit}"], runner=runner),
        "v0.3 source commit",
    )
    if (
        expected_source_commit is not None
        and head
        != _require_git_object(expected_source_commit, "expected source commit")
    ):
        raise ActionEffectQualificationError(
            "v0.3 source differs from qualification"
        )
    if _git(
        root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        runner=runner,
    ):
        raise ActionEffectQualificationError(
            "v0.3 qualification requires a clean repository"
        )
    ref = f"refs/tags/{QUALIFIED_TAG}"
    if _git(root, ["cat-file", "-t", ref], runner=runner) != "tag":
        raise ActionEffectQualificationError("v0.3 source tag must be annotated")
    tag_object = _require_git_object(
        _git(root, ["rev-parse", "--verify", ref], runner=runner),
        "v0.3 tag object",
    )
    if (
        expected_tag_object is not None
        and tag_object
        != _require_git_object(expected_tag_object, "expected tag object")
    ):
        raise ActionEffectQualificationError("v0.3 tag object changed")
    if _git(root, ["rev-list", "-n", "1", ref], runner=runner) != head:
        raise ActionEffectQualificationError("v0.3 tag does not point to HEAD")
    remote_url = _git(
        root,
        ["remote", "get-url", QUALIFIED_REMOTE],
        runner=runner,
    )
    if remote_url != EXPECTED_ORIGIN_URL:
        raise ActionEffectQualificationError("v0.3 origin changed")
    remote = _git(
        root,
        ["ls-remote", "--tags", QUALIFIED_REMOTE, ref],
        runner=runner,
    ).split()
    if remote != [tag_object, ref]:
        raise ActionEffectQualificationError(
            "v0.3 annotated tag is not published"
        )
    raw_message = _git(
        root,
        ["for-each-ref", "--format=%(contents)", ref],
        runner=runner,
    )
    if "\n" in raw_message:
        raise ActionEffectQualificationError(
            "v0.3 tag message must be one JSON line"
        )
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError as error:
        raise ActionEffectQualificationError(
            "v0.3 tag message is not JSON"
        ) from error
    expected = (
        build_action_effect_tag_payload(root, source_commit=head)
        if expected_tag_payload is None
        else json.loads(json.dumps(expected_tag_payload))
    )
    canonical = json.dumps(
        expected,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    if (
        not isinstance(payload, Mapping)
        or set(payload) != TAG_FIELDS
        or payload != expected
        or raw_message != canonical
    ):
        raise ActionEffectQualificationError(
            "v0.3 tag payload differs from live protocol inputs"
        )
    return {
        "commit": head,
        "dirty": False,
        "tag": QUALIFIED_TAG,
        "tag_object": tag_object,
        "remote": QUALIFIED_REMOTE,
        "remote_url": remote_url,
        "tag_payload": dict(payload),
        "tag_payload_sha256": _canonical_sha256(payload),
    }


def _expected_report(
    repository: Path,
    *,
    source: Mapping[str, Any],
    claim: Mapping[str, Any],
    smoke_report: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    Any,
    Mapping[lessons.LessonId, frozenset[str]],
]:
    parent, base, mapping, guards, predecessors, sampler = (
        _collect_static_inputs(repository)
    )
    sampler_digest = _canonical_sha256(sampler)
    smoke_evidence = validate_disposable_smoke(
        smoke_report,
        source_commit=str(source["commit"]),
        guard_mapping_sha256=str(guards["applied_mapping_sha256"]),
        sampler_preflight_sha256=sampler_digest,
    )
    protocol_path = repository / PROTOCOL_DOCUMENT
    failed_attempt = authenticate_failed_stage_a_attempt(
        repository=repository,
    )
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "kind": KIND,
            "verdict": VERDICT,
            "claim": dict(claim),
            "source": dict(source),
            "protocol_document": {
                "path": str(PROTOCOL_DOCUMENT),
                "sha256": file_sha256(protocol_path),
            },
            "parent": parent,
            "predecessors": predecessors,
            "failed_stage_a_attempt": failed_attempt,
            "guards": guards,
            "sampler_preflight": sampler,
            "architecture_contract": _architecture_contract(),
            "protected_partitions": _protected_seed_partitions(),
            "smoke_evidence": smoke_evidence,
            "storage_caps": _storage_caps(),
            "restrictions": {
                "arms_run_sequentially": True,
                "each_arm_starts_from_exact_confirmed_u1": True,
                "u2_u2r_u2s_checkpoint_loading_authorized": False,
                "stage_a_resume_supported": False,
                "failed_attempt_resume_authorized": False,
                "failed_attempt_root_reuse_authorized": False,
                "replacement_restarts_both_arms_from_confirmed_u1": True,
                "confirmation_or_final_seed_issuer_opened": False,
                "canonical_or_claim_policy_updates_during_qualification": False,
                "disposable_smoke_policies_destroyed": True,
                "development_checkpoint_reuse_authorized": False,
                "stage_b_authorized": False,
                "u3_authorized": False,
            },
        },
        base,
        mapping,
    )


@dataclass(frozen=True)
class ActionEffectQualificationEvidence:
    """Authenticated inputs exposed to the Stage-A trainer."""

    report: str
    report_sha256: str
    checksum: str
    source_commit: str
    tag: str
    tag_object: str
    tag_payload_sha256: str
    verdict: str
    protocol_document_sha256: str
    guard_mapping_sha256: str
    sampler_preflight_sha256: str
    architecture_contract_sha256: str
    smoke_evidence_sha256: str
    protected_partitions_sha256: str
    failed_stage_a_attempt_sha256: str
    storage_caps: Mapping[str, int]
    _report_bytes: bytes = field(repr=False, compare=False)
    _base_qualification: Any = field(repr=False, compare=False)
    _forbidden_layouts: Mapping[lessons.LessonId, frozenset[str]] = field(
        repr=False,
        compare=False,
    )

    def public_dict(self) -> dict[str, Any]:
        return {
            "report": self.report,
            "report_sha256": self.report_sha256,
            "checksum": self.checksum,
            "source_commit": self.source_commit,
            "tag": self.tag,
            "tag_object": self.tag_object,
            "tag_payload_sha256": self.tag_payload_sha256,
            "verdict": self.verdict,
            "protocol_document_sha256": self.protocol_document_sha256,
            "guard_mapping_sha256": self.guard_mapping_sha256,
            "sampler_preflight_sha256": self.sampler_preflight_sha256,
            "architecture_contract_sha256": self.architecture_contract_sha256,
            "smoke_evidence_sha256": self.smoke_evidence_sha256,
            "protected_partitions_sha256": (
                self.protected_partitions_sha256
            ),
            "failed_stage_a_attempt_sha256": (
                self.failed_stage_a_attempt_sha256
            ),
            "storage_caps": dict(self.storage_caps),
        }

    def verified_report(self) -> dict[str, Any]:
        value = json.loads(self._report_bytes)
        if not isinstance(value, dict):
            raise ActionEffectQualificationError(
                "verified v0.3 report bytes changed"
            )
        return value

    def seed_access(self) -> Any:
        return self._base_qualification.seed_access()

    def forbidden_layout_hashes(
        self,
    ) -> Mapping[lessons.LessonId, frozenset[str]]:
        return MappingProxyType(
            {
                lesson: frozenset(values)
                for lesson, values in self._forbidden_layouts.items()
            }
        )


def _assert_canonical_report(path: Path) -> Path:
    requested = path.expanduser().absolute()
    if requested != CANONICAL_REPORT:
        raise ActionEffectQualificationError(
            f"v0.3 qualification report must be {CANONICAL_REPORT}"
        )
    _reject_symlink_chain(requested)
    return requested


def _verify_checksum(path: Path, digest: str) -> None:
    try:
        fields = _regular_file_bytes(
            path,
            "v0.3 qualification checksum",
            maximum=256,
        ).decode("ascii").strip().split()
    except UnicodeDecodeError as error:
        raise ActionEffectQualificationError(
            "v0.3 checksum is not ASCII"
        ) from error
    if fields != [digest, CANONICAL_REPORT.name]:
        raise ActionEffectQualificationError(
            "v0.3 qualification checksum changed"
        )


def verify_action_effect_qualification(
    path: Path,
    *,
    expected_source_commit: str,
    expected_tag_object: str | None = None,
    repository: Path | None = None,
) -> ActionEffectQualificationEvidence:
    """Reauthenticate the canonical Stage-A qualification and live inputs."""

    report_path = _assert_canonical_report(path)
    repository = (
        Path(__file__).resolve().parents[2]
        if repository is None
        else repository.expanduser().resolve()
    )
    source = verify_source_tag(
        repository,
        expected_source_commit=expected_source_commit,
        expected_tag_object=expected_tag_object,
    )
    report_bytes = _regular_file_bytes(
        report_path,
        "v0.3 qualification report",
        maximum=MAX_REPORT_BYTES,
    )
    try:
        report = json.loads(report_bytes)
    except json.JSONDecodeError as error:
        raise ActionEffectQualificationError(
            "v0.3 qualification report is invalid JSON"
        ) from error
    if not isinstance(report, dict) or set(report) != _REPORT_FIELDS:
        raise ActionEffectQualificationError(
            "v0.3 qualification report fields changed"
        )
    digest = hashlib.sha256(report_bytes).hexdigest()
    _verify_checksum(CANONICAL_CHECKSUM, digest)
    claim = _read_json(CANONICAL_CLAIM, "v0.3 qualification claim")
    if report.get("claim") != claim:
        raise ActionEffectQualificationError(
            "v0.3 qualification claim binding changed"
        )
    smoke_report = report.get("smoke_evidence", {}).get("_full_report")
    if not isinstance(smoke_report, Mapping):
        raise ActionEffectQualificationError(
            "v0.3 report lacks its disposable smoke transcript"
        )
    expected, base, mapping = _expected_report(
        repository,
        source=source,
        claim=claim,
        smoke_report=smoke_report,
    )
    expected["storage_preflight"] = report.get("storage_preflight")
    expected["created_at"] = report.get("created_at")
    expected["smoke_evidence"]["_full_report"] = smoke_report
    if report != expected:
        raise ActionEffectQualificationError(
            "v0.3 qualification evidence differs from live inputs"
        )
    _verify_storage_preflight(report["storage_preflight"])
    try:
        created_at = datetime.fromisoformat(str(report["created_at"]))
    except ValueError as error:
        raise ActionEffectQualificationError(
            "v0.3 qualification timestamp is invalid"
        ) from error
    if created_at.tzinfo is None or created_at.utcoffset() != UTC.utcoffset(None):
        raise ActionEffectQualificationError(
            "v0.3 qualification timestamp must be UTC"
        )
    return ActionEffectQualificationEvidence(
        report=str(report_path),
        report_sha256=digest,
        checksum=str(CANONICAL_CHECKSUM),
        source_commit=str(source["commit"]),
        tag=QUALIFIED_TAG,
        tag_object=str(source["tag_object"]),
        tag_payload_sha256=str(source["tag_payload_sha256"]),
        verdict=VERDICT,
        protocol_document_sha256=str(report["protocol_document"]["sha256"]),
        guard_mapping_sha256=str(report["guards"]["applied_mapping_sha256"]),
        sampler_preflight_sha256=_canonical_sha256(
            report["sampler_preflight"]
        ),
        architecture_contract_sha256=_canonical_sha256(
            report["architecture_contract"]
        ),
        smoke_evidence_sha256=_canonical_sha256(
            {
                key: value
                for key, value in report["smoke_evidence"].items()
                if key != "_full_report"
            }
        ),
        protected_partitions_sha256=_canonical_sha256(
            report["protected_partitions"]
        ),
        failed_stage_a_attempt_sha256=_canonical_sha256(
            report["failed_stage_a_attempt"]
        ),
        storage_caps=MappingProxyType(dict(report["storage_caps"])),
        _report_bytes=report_bytes,
        _base_qualification=base,
        _forbidden_layouts=mapping,
    )


def collect_action_effect_qualification(
    *,
    repository: Path,
    output: Path = CANONICAL_REPORT,
    smoke_runner: Callable[..., Mapping[str, Any]] = smoke.run_smoke,
) -> ActionEffectQualificationEvidence:
    """Claim and collect one qualification; never create a tag or launch."""

    report_path = _assert_canonical_report(output)
    source = verify_source_tag(repository)
    storage_preflight = _storage_preflight(require_qualification_absent=True)
    (
        _parent,
        smoke_base,
        smoke_mapping,
        smoke_guards,
        _predecessors,
        _sampler,
    ) = _collect_static_inputs(repository)
    report_path.parent.mkdir(mode=0o700)
    claim = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "kind": "qualification_attempt_claim",
        "claim_id": uuid.uuid4().hex,
        "source_commit": source["commit"],
        "tag": QUALIFIED_TAG,
        "tag_object": source["tag_object"],
        "created_at": utc_now(),
    }
    try:
        with CANONICAL_CLAIM.open("xb") as handle:
            claim_bytes = _canonical_json_bytes(claim)
            handle.write(claim_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        smoke_report = smoke_runner(
            repository=repository,
            require_clean_source=True,
            seed_access=smoke_base.seed_access(),
            forbidden_layout_hashes=smoke_mapping,
        )
        if (
            smoke_report.get("guard_mapping_sha256")
            != smoke_guards["applied_mapping_sha256"]
        ):
            raise ActionEffectQualificationError(
                "v0.3 disposable smoke used a different static history guard"
            )
        expected, _base, _mapping = _expected_report(
            repository,
            source=source,
            claim=claim,
            smoke_report=smoke_report,
        )
        expected["smoke_evidence"]["_full_report"] = json.loads(
            json.dumps(smoke_report)
        )
        expected["storage_preflight"] = storage_preflight
        expected["created_at"] = utc_now()
        if set(expected) != _REPORT_FIELDS:
            raise ActionEffectQualificationError(
                "internal v0.3 report schema is incomplete"
            )
        report_bytes = _canonical_json_bytes(expected)
        with report_path.open("xb") as handle:
            handle.write(report_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        digest = hashlib.sha256(report_bytes).hexdigest()
        with CANONICAL_CHECKSUM.open("x", encoding="ascii") as handle:
            handle.write(f"{digest}  {report_path.name}\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # The durable claim is intentionally retained.  This attempt identity
        # is terminal and cannot silently be retried under the same root.
        raise
    return verify_action_effect_qualification(
        report_path,
        expected_source_commit=str(source["commit"]),
        expected_tag_object=str(source["tag_object"]),
        repository=repository,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", type=Path, default=CANONICAL_REPORT)
    parser.add_argument("--tag-payload-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = args.repository.expanduser().resolve()
    if args.tag_payload_only:
        head = _require_git_object(
            _git(repository, ["rev-parse", "--verify", "HEAD^{commit}"]),
            "v0.3 source commit",
        )
        if _git(
            repository,
            ["status", "--porcelain=v1", "--untracked-files=all"],
        ):
            raise SystemExit(
                "v0.3 tag payload requires a clean committed repository"
            )
        print(
            json.dumps(
                build_action_effect_tag_payload(
                    repository,
                    source_commit=head,
                ),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return
    evidence = collect_action_effect_qualification(
        repository=repository,
        output=args.output,
    )
    print(json.dumps(evidence.public_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
