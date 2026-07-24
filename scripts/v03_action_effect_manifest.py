#!/usr/bin/env python3
"""Fail-closed manifest state machine for the v0.3 Stage-A architecture study.

This helper never starts a process and has no resume operation.  It owns only
the immutable cohort contract, the small mutable cohort state, and terminal
closeout records.  A launcher may ask it which arm is legal next, but the
fixed order is always ``sham`` then ``action-effect``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dungeon_apprentice import v03_action_effect as v03

PROTOCOL = v03.PROTOCOL
QUALIFICATION_PROTOCOL = PROTOCOL
SMOKE_PROTOCOL = "dungeon-apprentice-v0.3-action-effect-disposable-smoke"
COHORT_ID = "v0.3-action-effect-stage-a-20260724"
TAG_NAME = "action-effect-architecture-v0.3-stage-a-20260724"
DEFAULT_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/"
    "v03-action-effect-stage-a-20260724"
)
DEFAULT_MEDIA_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/"
    "v03-action-effect-stage-a-media-20260724"
)
QUALIFICATION_REPORT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/qualifications/"
    "v0.3-action-effect-stage-a-20260724/report.json"
)
DEFAULT_DASHBOARD_PORT = 8788

ACTION_CAP = v03.CHILD_ACTION_BUDGET
TOTAL_ACTION_CAP = 2 * ACTION_CAP
EVALUATION_EVERY = v03.EVALUATION_INTERVAL
EXAM_COUNT = v03.EXAM_COUNT
EXAM_EPISODES = v03.EVALUATION_SEED_COUNT
EXPECTED_CASE_COUNT = EXAM_COUNT * EXAM_EPISODES * 4
PARENT_LIFETIME_ACTIONS = v03.PARENT_LIFETIME_ACTIONS
TERMINAL_LIFETIME_ACTIONS = PARENT_LIFETIME_ACTIONS + ACTION_CAP
TERMINAL_OPTIMIZER_UPDATES = (
    v03.PARENT_OPTIMIZER_UPDATES
    + ACTION_CAP // v03.ROLLOUT_TRANSITIONS * v03.PPO_EPOCHS
)
ARM_ORDER = ("sham", "action-effect")
ARM_LABELS = {
    "sham": "A · Zero-context sham",
    "action-effect": "B · Action-effect context",
}
LESSON_IDS = (
    "navigate/full",
    "unlock/u0-visible",
    "unlock/u1-local",
    "unlock/u2-separated",
)
LINEAGE_CAP_BYTES = v03.LINEAGE_CAP_BYTES
COHORT_SCIENTIFIC_CAP_BYTES = v03.COHORT_SCIENTIFIC_CAP_BYTES
MEDIA_CAP_BYTES = v03.MEDIA_CAP_BYTES
COMBINED_PLANNED_CAP_BYTES = v03.COMBINED_PLANNED_CAP_BYTES
MAX_JSON_BYTES = 64 * 1024 * 1024
PROCESS_CLOSEOUT_NAME = "process-closeout.json"
PROCESS_CLOSEOUT_MAX_BYTES = 4 * 1024 * 1024
PROCESS_INVENTORY_MAX_BYTES = 8 * 1024 * 1024
PROCESS_INVENTORY_MAX_ROWS = 4096
PROCESS_COMMAND_MAX_BYTES = 64 * 1024
PROCESS_SCAN_TIMEOUT_SECONDS = 10
PROCESS_SCAN_COMMAND = ("/bin/ps", "-axo", "pid=,ppid=,command=")
PROCESS_ROLES = frozenset(
    {"unrelated", "dashboard", "trainer", "supervisor", "caffeinate"}
)
PROHIBITED_PROCESS_ROLES = frozenset(
    {"trainer", "supervisor", "caffeinate"}
)
TRAINER_PROCESS_PATTERN = re.compile(
    r"(?:^|[\s/])(?:dungeon-train|"
    r"dungeon_apprentice\.(?:train|v02_sentinel|v02_u1|v02_u2|"
    r"v02_u2r|v02_u2s|v03_action_effect_train))(?=\s|$)"
)
SUPERVISOR_PROCESS_PATTERN = re.compile(
    r"(?:^|[\s/])u2_trainer_supervisor\.py(?=\s|$)"
)
DASHBOARD_PROCESS_PATTERN = re.compile(
    r"(?:^|[\s/])dungeon_apprentice\.v03_action_effect_dashboard"
    r"(?=\s|$)"
)
CAFFEINATE_PROCESS_PATTERN = re.compile(
    r"(?:^|\s)(?:/usr/bin/)?caffeinate(?=\s|$)"
)
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
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


class V03ManifestError(RuntimeError):
    """Raised when v0.3 provenance or sequential state would drift."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _encoded(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise V03ManifestError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _exclusive_write(path: Path, payload: Mapping[str, Any]) -> None:
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


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
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
        raise V03ManifestError(f"{label} must be absolute")
    try:
        metadata = candidate.lstat()
    except FileNotFoundError as error:
        raise V03ManifestError(f"{label} does not exist: {candidate}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise V03ManifestError(f"{label} must be a regular directory")
    return candidate.resolve(strict=True)


def _read_json(
    path: Path,
    *,
    label: str,
    maximum: int = MAX_JSON_BYTES,
) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise V03ManifestError(f"{label} is missing: {path}") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 0
        or metadata.st_size > maximum
    ):
        raise V03ManifestError(f"{label} is not a safe regular JSON file")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V03ManifestError(f"cannot read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise V03ManifestError(f"{label} must contain one JSON object")
    return payload


def _object_id(value: str, *, label: str) -> str:
    normalized = value.strip().lower()
    if COMMIT_PATTERN.fullmatch(normalized) is None:
        raise V03ManifestError(f"{label} must be a full Git object ID")
    return normalized


def _digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise V03ManifestError(f"{label} must be one SHA-256 digest")
    return value


def _source_commit(payload: Mapping[str, Any]) -> Any:
    source = payload.get("source")
    return source.get("commit") if isinstance(source, Mapping) else source


def _verified_qualification_binding(
    report_path: Path,
    *,
    source_commit: str,
    tag_object: str,
    root: Path,
    media_root: Path,
) -> dict[str, Any]:
    try:
        from dungeon_apprentice import v03_action_effect_qualify as qualifier
    except ImportError as error:
        raise V03ManifestError(
            "v0.3 qualification verifier is unavailable"
        ) from error
    repository = Path(__file__).resolve().parents[1]
    if (
        report_path.expanduser().absolute() != qualifier.CANONICAL_REPORT
        or root != qualifier.CANONICAL_COHORT_ROOT
        or media_root != qualifier.CANONICAL_MEDIA_ROOT
        or TAG_NAME != qualifier.QUALIFIED_TAG
    ):
        raise V03ManifestError(
            "v0.3 qualification, cohort, or media identity changed"
        )
    try:
        evidence = qualifier.verify_action_effect_qualification(
            report_path,
            expected_source_commit=source_commit,
            expected_tag_object=tag_object,
            repository=repository,
        )
        public = evidence.public_dict()
        report = evidence.verified_report()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise V03ManifestError(
            f"v0.3 qualification is invalid: {error}"
        ) from error
    restrictions = report.get("restrictions")
    smoke = report.get("smoke_evidence")
    expected_storage_caps = {
        "per_arm_bytes": LINEAGE_CAP_BYTES,
        "scientific_cohort_bytes": COHORT_SCIENTIFIC_CAP_BYTES,
        "media_bytes": MEDIA_CAP_BYTES,
        "combined_bytes": COMBINED_PLANNED_CAP_BYTES,
    }
    if (
        not isinstance(public, dict)
        or public.get("report") != str(qualifier.CANONICAL_REPORT)
        or public.get("source_commit") != source_commit
        or public.get("tag") != TAG_NAME
        or public.get("tag_object") != tag_object
        or public.get("verdict") != "qualified"
        or _sha256(qualifier.CANONICAL_REPORT)
        != public.get("report_sha256")
        or public.get("storage_caps") != expected_storage_caps
        or report.get("protocol") != PROTOCOL
        or report.get("kind") != qualifier.KIND
        or report.get("verdict") != "qualified"
        or not isinstance(smoke, Mapping)
        or not isinstance(smoke.get("_full_report"), Mapping)
        or not isinstance(restrictions, Mapping)
        or restrictions.get("arms_run_sequentially") is not True
        or restrictions.get("stage_a_resume_supported") is not False
        or restrictions.get(
            "development_checkpoint_reuse_authorized"
        )
        is not False
        or restrictions.get("u3_authorized") is not False
    ):
        raise V03ManifestError(
            "v0.3 qualification public binding is incomplete"
        )
    return json.loads(json.dumps(public))


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
        },
        "qualification": dict(qualification),
        "parent": {
            "protocol": "dungeon-apprentice-v0.2-u1",
            "u1_child_seed": v03.PARENT_U1_SEED,
            "checkpoint": str(v03.PARENT_CHECKPOINT),
            "checkpoint_sha256": v03.PARENT_CHECKPOINT_SHA256,
            "policy_tensor_sha256": v03.PARENT_POLICY_TENSOR_SHA256,
            "optimizer_state_sha256": v03.PARENT_OPTIMIZER_STATE_SHA256,
            "lifetime_actions": PARENT_LIFETIME_ACTIONS,
            "optimizer_updates": v03.PARENT_OPTIMIZER_UPDATES,
        },
        "roots": {"cohort": str(root), "media": str(media_root)},
        "process_closeout": {
            "required_before_finalization": True,
            "evidence_path": PROCESS_CLOSEOUT_NAME,
            "collector": list(PROCESS_SCAN_COMMAND),
            "maximum_rows": PROCESS_INVENTORY_MAX_ROWS,
            "maximum_inventory_bytes": PROCESS_INVENTORY_MAX_BYTES,
            "prohibited_roles": sorted(PROHIBITED_PROCESS_ROLES),
            "dashboard_may_remain": True,
        },
        "matched_design": {
            "arm_order": list(ARM_ORDER),
            "architecture_initialization_seed": (
                v03.ARCHITECTURE_INITIALIZATION_SEED
            ),
            "algorithm_seed": v03.ALGORITHM_SEED,
            "worker_streams": list(v03.WORKER_STREAMS),
            "action_cap_per_arm": ACTION_CAP,
            "total_action_cap": TOTAL_ACTION_CAP,
            "evaluation_every": EVALUATION_EVERY,
            "exam_count": EXAM_COUNT,
            "first_rollout_transitions": v03.ROLLOUT_TRANSITIONS,
            "first_optimizer_boundary": v03.ROLLOUT_TRANSITIONS,
            "fresh_only": True,
            "resumable": False,
            "checkpoint_promotable": False,
            "sham_selectable": False,
            "u3_authorized": False,
            "interruption_disposition": "operationally_incomplete",
            "storage_caps": {
                "lineage_cap_bytes": LINEAGE_CAP_BYTES,
                "cohort_scientific_cap_bytes": COHORT_SCIENTIFIC_CAP_BYTES,
                "media_cap_bytes": MEDIA_CAP_BYTES,
                "combined_planned_cap_bytes": COMBINED_PLANNED_CAP_BYTES,
                "minimum_free_gib": int(v03.MINIMUM_FREE_GIB),
            },
        },
        "observation": {
            "kind": "Dict",
            "image": {"shape_hwc": [56, 56, 3], "dtype": "uint8"},
            "action_effect": {
                "shape": [9],
                "dtype": "float32",
                "previous_action_one_hot": 7,
                "visible_outcome_one_hot": ["changed", "unchanged"],
                "reset_sentinel": "all_zero",
                "privileged_state": False,
            },
        },
        "arms": [
            {
                "id": arm,
                "label": ARM_LABELS[arm],
                "directory": arm,
                "media_directory": arm,
                "mode": arm,
                "context_enabled": arm == "action-effect",
                "selectable": arm == "action-effect",
                "action_cap": ACTION_CAP,
            }
            for arm in ARM_ORDER
        ],
    }


def _expected_state(contract: Mapping[str, Any]) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "cohort_id": COHORT_ID,
        "contract_sha256": "",
        "source_commit": contract["source"]["commit"],
        "tag": TAG_NAME,
        "tag_object": contract["preregistration"]["tag_object"],
        "qualification_sha256": contract["qualification"][
            "report_sha256"
        ],
        "phase": "ready",
        "active_arm": None,
        "created_at": now,
        "updated_at": now,
        "arms": [
            {
                "id": arm["id"],
                "directory": arm["directory"],
                "media_directory": arm["media_directory"],
                "state": "pending",
                "attempts": [],
                "terminal": None,
            }
            for arm in contract["arms"]
        ],
        "terminal_report": None,
        "process_closeout": None,
        "history": [{"event": "cohort_created", "at": now}],
    }


def _validate_contract(contract: Mapping[str, Any]) -> None:
    source = contract.get("source")
    preregistration = contract.get("preregistration")
    parent = contract.get("parent")
    matched = contract.get("matched_design")
    process_closeout = contract.get("process_closeout")
    observation = contract.get("observation")
    arms = contract.get("arms")
    if (
        contract.get("schema_version") != 1
        or contract.get("protocol") != PROTOCOL
        or contract.get("cohort_id") != COHORT_ID
        or not isinstance(source, Mapping)
        or source.get("dirty") is not False
        or COMMIT_PATTERN.fullmatch(str(source.get("commit", ""))) is None
        or not isinstance(preregistration, Mapping)
        or preregistration.get("tag") != TAG_NAME
        or preregistration.get("peeled_commit") != source.get("commit")
        or COMMIT_PATTERN.fullmatch(
            str(preregistration.get("tag_object", ""))
        )
        is None
        or not isinstance(parent, Mapping)
        or parent.get("checkpoint_sha256")
        != v03.PARENT_CHECKPOINT_SHA256
        or parent.get("policy_tensor_sha256")
        != v03.PARENT_POLICY_TENSOR_SHA256
        or parent.get("optimizer_state_sha256")
        != v03.PARENT_OPTIMIZER_STATE_SHA256
        or parent.get("lifetime_actions") != PARENT_LIFETIME_ACTIONS
        or parent.get("optimizer_updates")
        != v03.PARENT_OPTIMIZER_UPDATES
        or not isinstance(matched, Mapping)
        or matched.get("arm_order") != list(ARM_ORDER)
        or matched.get("action_cap_per_arm") != ACTION_CAP
        or matched.get("evaluation_every") != EVALUATION_EVERY
        or matched.get("fresh_only") is not True
        or matched.get("resumable") is not False
        or matched.get("checkpoint_promotable") is not False
        or matched.get("sham_selectable") is not False
        or matched.get("u3_authorized") is not False
        or not isinstance(process_closeout, Mapping)
        or process_closeout.get("required_before_finalization") is not True
        or process_closeout.get("evidence_path")
        != PROCESS_CLOSEOUT_NAME
        or process_closeout.get("collector") != list(PROCESS_SCAN_COMMAND)
        or process_closeout.get("maximum_rows")
        != PROCESS_INVENTORY_MAX_ROWS
        or process_closeout.get("maximum_inventory_bytes")
        != PROCESS_INVENTORY_MAX_BYTES
        or process_closeout.get("prohibited_roles")
        != sorted(PROHIBITED_PROCESS_ROLES)
        or process_closeout.get("dashboard_may_remain") is not True
        or not isinstance(observation, Mapping)
        or observation.get("kind") != "Dict"
        or not isinstance(observation.get("action_effect"), Mapping)
        or observation["action_effect"].get("shape") != [9]
        or observation["action_effect"].get("privileged_state") is not False
        or not isinstance(arms, list)
        or [arm.get("id") for arm in arms if isinstance(arm, Mapping)]
        != list(ARM_ORDER)
        or [arm.get("selectable") for arm in arms]
        != [False, True]
    ):
        raise V03ManifestError("immutable v0.3 cohort contract changed")


def _validate_state(
    state: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
) -> None:
    arms = state.get("arms")
    process_closeout = state.get("process_closeout")
    if (
        state.get("schema_version") != 1
        or state.get("protocol") != PROTOCOL
        or state.get("cohort_id") != COHORT_ID
        or state.get("contract_sha256") != contract_sha256
        or state.get("source_commit") != contract["source"]["commit"]
        or state.get("tag") != TAG_NAME
        or state.get("tag_object")
        != contract["preregistration"]["tag_object"]
        or state.get("qualification_sha256")
        != contract["qualification"]["report_sha256"]
        or state.get("phase") not in COHORT_PHASES
        or not isinstance(arms, list)
        or [arm.get("id") for arm in arms if isinstance(arm, Mapping)]
        != list(ARM_ORDER)
    ):
        raise V03ManifestError("mutable v0.3 state differs from its contract")
    training = []
    for arm in arms:
        if (
            arm.get("state") not in ARM_STATES
            or not isinstance(arm.get("attempts"), list)
            or len(arm["attempts"]) > 1
        ):
            raise V03ManifestError("v0.3 arm state is invalid or reused")
        if arm["state"] == "training":
            training.append(arm["id"])
    active = state.get("active_arm")
    phase = state["phase"]
    if (
        len(training) > 1
        or (phase == "training" and training != [active])
        or (phase != "training" and (training or active is not None))
        or (
            phase == "awaiting_closeout"
            and [arm["state"] for arm in arms] != ["completed", "completed"]
        )
        or (
            phase == "completed"
            and (
                [arm["state"] for arm in arms]
                != ["completed", "completed"]
                or not isinstance(state.get("terminal_report"), Mapping)
                or not isinstance(process_closeout, Mapping)
            )
        )
        or (
            process_closeout is not None
            and (
                not isinstance(process_closeout, Mapping)
                or phase
                not in {
                    "awaiting_closeout",
                    "completed",
                    "integrity_failed",
                }
            )
        )
    ):
        raise V03ManifestError("v0.3 phase permits no concurrent or skipped arm")


def _load_verified(
    root: Path,
    *,
    source_commit: str,
    tag_object: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    resolved = _regular_directory(root, label="v0.3 cohort root")
    contract_path = resolved / "cohort-contract.json"
    state_path = resolved / "cohort.json"
    contract = _read_json(contract_path, label="v0.3 cohort contract")
    state = _read_json(state_path, label="v0.3 cohort state")
    _validate_contract(contract)
    commit = _object_id(source_commit, label="source commit")
    tag = _object_id(tag_object, label="tag object")
    if (
        contract["source"]["commit"] != commit
        or contract["preregistration"]["tag_object"] != tag
    ):
        raise V03ManifestError("v0.3 source or annotated tag changed")
    roots = contract.get("roots")
    if (
        not isinstance(roots, Mapping)
        or roots.get("cohort") != str(resolved)
    ):
        raise V03ManifestError("v0.3 cohort root binding changed")
    media = _regular_directory(
        Path(str(roots.get("media"))),
        label="v0.3 media root",
    )
    qualification_path = Path(
        str(contract["qualification"].get("report"))
    )
    live_binding = _verified_qualification_binding(
        qualification_path,
        source_commit=commit,
        tag_object=tag,
        root=resolved,
        media_root=media,
    )
    if live_binding != contract["qualification"]:
        raise V03ManifestError("v0.3 qualification binding changed")
    contract_digest = _sha256(contract_path)
    _validate_state(
        state,
        contract=contract,
        contract_sha256=contract_digest,
    )
    if state.get("process_closeout") is not None:
        _verified_process_closeout(resolved, state=state)
    return resolved, contract, state


@contextmanager
def _mutation_lock(root: Path) -> Iterator[None]:
    lock = root / ".v03-manifest.lock"
    try:
        descriptor = os.open(
            lock,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        raise V03ManifestError(
            "another v0.3 manifest mutation is active or failed uncleanly"
        ) from error
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        yield
    finally:
        with suppress(FileNotFoundError):
            lock.unlink()


def _arm_state(state: dict[str, Any], arm_id: str) -> dict[str, Any]:
    if arm_id not in ARM_ORDER:
        raise V03ManifestError(f"unknown v0.3 arm: {arm_id}")
    return next(arm for arm in state["arms"] if arm["id"] == arm_id)


def _assert_fixed_order(state: Mapping[str, Any], arm_id: str) -> None:
    index = ARM_ORDER.index(arm_id)
    states = [arm["state"] for arm in state["arms"]]
    if any(value != "completed" for value in states[:index]):
        raise V03ManifestError("v0.3 arms cannot skip the fixed sham-first order")
    if states[index] != "pending":
        raise V03ManifestError("v0.3 arms are fresh-only and may start once")
    if any(value != "pending" for value in states[index + 1 :]):
        raise V03ManifestError("a later v0.3 arm changed before its turn")


def _finite_number(value: Any, *, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise V03ManifestError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise V03ManifestError(f"{label} must be numeric") from error
    if not math.isfinite(number) or number < minimum:
        raise V03ManifestError(f"{label} is outside its valid range")
    return number


def _first_rollout_evidence(
    value: Any,
    *,
    arm_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise V03ManifestError(f"{arm_id} first-rollout evidence is missing")
    trajectory = value.get("trajectory_identity")
    post_rng = value.get("post_rollout_rng_identity")
    aggregate = value.get("aggregate_sha256")
    if (
        not isinstance(trajectory, list)
        or len(trajectory) != v03.WORKERS
        or not isinstance(post_rng, Mapping)
        or _canonical_sha256(
            {key: item for key, item in value.items() if key != "aggregate_sha256"}
        )
        != aggregate
    ):
        raise V03ManifestError(
            f"{arm_id} first-rollout boundary is unauthenticated"
        )
    _digest(value.get("policy_output_sha256"), label=f"{arm_id} policy output")
    _digest(
        value.get("episode_ledger_normalized_sha256"),
        label=f"{arm_id} first-rollout episode ledger",
    )
    _digest(aggregate, label=f"{arm_id} first-rollout aggregate")
    return dict(value)


def _context_encoder_evidence(
    history_value: Any,
    terminal_value: Any,
    *,
    arm_id: str,
) -> dict[str, Any]:
    if (
        not isinstance(history_value, list)
        or not history_value
        or not all(isinstance(item, Mapping) for item in history_value)
        or not isinstance(terminal_value, Mapping)
        or terminal_value != history_value[-1]
    ):
        raise V03ManifestError(f"{arm_id} context-encoder evidence is missing")
    initial = history_value[0]
    terminal = terminal_value
    initial_norm = _finite_number(
        initial.get("weight_l2_norm"),
        label=f"{arm_id} initial encoder norm",
    )
    terminal_norm = _finite_number(
        terminal.get("weight_l2_norm"),
        label=f"{arm_id} terminal encoder norm",
    )
    update_norm = _finite_number(
        terminal.get("update_l2_norm"),
        label=f"{arm_id} terminal encoder update norm",
    )
    if (
        initial_norm != 0.0
        or int(initial.get("weight_nonzero_parameters", -1)) != 0
        or int(initial.get("child_trained_actions", -1)) != 0
        or (
            arm_id == "sham"
            and (
                terminal_norm != 0.0
                or update_norm != 0.0
                or int(terminal.get("weight_nonzero_parameters", -1)) != 0
            )
        )
    ):
        raise V03ManifestError(
            f"{arm_id} context-encoder boundary changed"
        )
    return {
        "history_records": len(history_value),
        "initial": dict(initial),
        "terminal": dict(terminal),
        "first_nonzero_child_trained_actions": next(
            (
                int(item["child_trained_actions"])
                for item in history_value
                if int(item.get("weight_nonzero_parameters", 0)) > 0
            ),
            None,
        ),
    }


def _context_summary(value: Any, *, arm_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise V03ManifestError(f"{arm_id} context summary is missing")
    activation = _finite_number(
        value.get("activation_rate"),
        label=f"{arm_id} context activation rate",
    )
    changed = _finite_number(
        value.get("changed_rate"),
        label=f"{arm_id} changed rate",
    )
    transitions = int(value.get("transitions", -1))
    unchanged_count = int(value.get("unchanged", -1))
    if transitions < 1 or unchanged_count < 0:
        raise V03ManifestError(f"{arm_id} context counts are invalid")
    unchanged_value = unchanged_count / transitions
    unchanged = _finite_number(
        unchanged_value,
        label=f"{arm_id} unchanged rate",
    )
    if any(number > 1.0 for number in (activation, changed, unchanged)):
        raise V03ManifestError(f"{arm_id} context rates exceed one")
    if arm_id == "sham" and activation != 0.0:
        raise V03ManifestError("sham cannot expose active context")
    return {**dict(value), "unchanged_rate": unchanged}


def _validate_status_identity(
    status: Mapping[str, Any],
    *,
    arm_id: str,
    contract: Mapping[str, Any],
    contract_sha256: str,
) -> None:
    if (
        status.get("protocol") != PROTOCOL
        or status.get("cohort_id") != COHORT_ID
        or status.get("arm") != arm_id
        or _source_commit(status) != contract["source"]["commit"]
        or status.get("qualification_sha256")
        != contract["qualification"]["report_sha256"]
        or status.get("cohort_contract_sha256") != contract_sha256
        or status.get("parent_checkpoint_sha256")
        != v03.PARENT_CHECKPOINT_SHA256
        or int(status.get("action_cap", -1)) != ACTION_CAP
    ):
        raise V03ManifestError(f"{arm_id} status provenance changed")


def _verified_arm_terminal(
    directory: Path,
    *,
    arm_id: str,
    contract: Mapping[str, Any],
    contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from dungeon_apprentice.v03_action_effect_train import (
            verify_arm_terminal_report,
        )
    except ImportError as error:
        raise V03ManifestError(
            "v0.3 terminal arm verifier is unavailable"
        ) from error
    status_path = directory / "status.json"
    report_path = directory / "report.json"
    integrity_path = directory / "report.integrity.json"
    status = _read_json(status_path, label=f"{arm_id} terminal status")
    report = _read_json(report_path, label=f"{arm_id} terminal report")
    integrity = _read_json(
        integrity_path,
        label=f"{arm_id} terminal report integrity",
    )
    _validate_status_identity(
        status,
        arm_id=arm_id,
        contract=contract,
        contract_sha256=contract_sha256,
    )
    report_sha256 = _sha256(report_path)
    try:
        verified = verify_arm_terminal_report(
            directory,
            expected_arm=arm_id,
            expected_source_commit=contract["source"]["commit"],
            expected_cohort_id=COHORT_ID,
            expected_contract_sha256=contract_sha256,
            expected_qualification_sha256=contract["qualification"][
                "report_sha256"
            ],
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise V03ManifestError(
            f"{arm_id} failed the trainer-owned terminal verifier"
        ) from error
    progress = report.get("progress")
    controller = report.get("controller")
    qualification = report.get("qualification")
    raw_exams = (
        controller.get("exam_records")
        if isinstance(controller, Mapping)
        else None
    )
    if (
        status.get("phase") != "completed"
        or int(status.get("child_trained_actions", -1)) != ACTION_CAP
        or int(status.get("remaining_action_budget", -1)) != 0
        or int(status.get("optimizer_updates", -1))
        != TERMINAL_OPTIMIZER_UPDATES
        or int(status.get("exam_count", -1)) != EXAM_COUNT
        or report.get("schema_version") != 1
        or report.get("protocol") != PROTOCOL
        or report.get("cohort_id") != COHORT_ID
        or report.get("arm") != arm_id
        or _source_commit(report) != contract["source"]["commit"]
        or qualification != contract["qualification"]
        or report.get("cohort_contract_sha256") != contract_sha256
        or report.get("development_checkpoint_reuse_authorized")
        is not False
        or integrity.get("schema_version") != 1
        or integrity.get("protocol") != PROTOCOL
        or integrity.get("cohort_id") != COHORT_ID
        or integrity.get("arm") != arm_id
        or integrity.get("report") != "report.json"
        or integrity.get("report_sha256") != report_sha256
        or integrity.get("cohort_contract_sha256") != contract_sha256
        or status.get("report_sha256") != report_sha256
        or verified.get("report_sha256") != report_sha256
        or verified.get("report_integrity_sha256")
        != _sha256(integrity_path)
        or verified.get("cohort_id") != COHORT_ID
        or verified.get("cohort_contract_sha256") != contract_sha256
        or verified.get("qualification_sha256")
        != contract["qualification"]["report_sha256"]
        or not isinstance(progress, Mapping)
    ):
        raise V03ManifestError(f"{arm_id} terminal report is not authentic")
    if (
        not isinstance(raw_exams, list)
        or len(raw_exams) != EXAM_COUNT
        or not all(isinstance(exam, Mapping) for exam in raw_exams)
    ):
        raise V03ManifestError(f"{arm_id} terminal exams are incomplete")
    try:
        grade = v03.grade_terminal(arm_id, raw_exams)
    except (RuntimeError, TypeError, ValueError) as error:
        raise V03ManifestError(
            f"{arm_id} terminal grade cannot be reproduced"
        ) from error
    first_rollout = _first_rollout_evidence(
        report.get("first_rollout_identity"),
        arm_id=arm_id,
    )
    context_encoder = _context_encoder_evidence(
        report.get("encoder_history"),
        report.get("terminal_encoder"),
        arm_id=arm_id,
    )
    context = _context_summary(
        report.get("context_metrics"),
        arm_id=arm_id,
    )
    case_inventory = report.get("case_evidence")
    case_count = (
        int(case_inventory.get("record_count", -1))
        if isinstance(case_inventory, Mapping)
        else -1
    )
    if (
        int(progress.get("child_trained_actions", -1)) != ACTION_CAP
        or int(progress.get("lifetime_trained_actions", -1))
        != TERMINAL_LIFETIME_ACTIONS
        or int(progress.get("optimizer_updates", -1))
        != TERMINAL_OPTIMIZER_UPDATES
        or int(progress.get("exam_count", -1)) != EXAM_COUNT
        or case_count != EXPECTED_CASE_COUNT
        or report.get("grade") != grade.public_dict()
        or verified.get("eligible") is not grade.eligible
        or verified.get("exam_records") != raw_exams
        or verified.get("case_count") != case_count
        or verified.get("first_rollout_identity") != first_rollout
        or verified.get("context_metrics") != report.get("context_metrics")
        or verified.get("terminal_encoder")
        != report.get("terminal_encoder")
    ):
        raise V03ManifestError(f"{arm_id} terminal counters or grade changed")
    return (
        {
            "arm": arm_id,
            "status_sha256": _sha256(status_path),
            "report_sha256": report_sha256,
            "report_integrity_sha256": _sha256(integrity_path),
            "child_trained_actions": ACTION_CAP,
            "lifetime_trained_actions": TERMINAL_LIFETIME_ACTIONS,
            "optimizer_updates": TERMINAL_OPTIMIZER_UPDATES,
            "exam_count": EXAM_COUNT,
            "case_count": case_count,
            "eligible": grade.eligible,
            "grade": grade.public_dict(),
            "first_rollout": first_rollout,
            "context_encoder": context_encoder,
            "context_summary": context,
        },
        report,
    )


def _tree_size(path: Path, *, label: str) -> int:
    total = 0
    for current, directories, files in os.walk(path, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *files]:
            item = current_path / name
            try:
                metadata = item.lstat()
            except OSError as error:
                raise V03ManifestError(f"cannot inspect {label}") from error
            if stat.S_ISLNK(metadata.st_mode):
                raise V03ManifestError(f"{label} contains a symbolic link")
            if stat.S_ISREG(metadata.st_mode):
                total += metadata.st_size
            elif not stat.S_ISDIR(metadata.st_mode):
                raise V03ManifestError(f"{label} contains an unsafe file")
    return total


def _gather_process_rows() -> list[dict[str, Any]]:
    """Capture one bounded system-wide process table without a shell."""

    try:
        completed = subprocess.run(
            PROCESS_SCAN_COMMAND,
            check=False,
            capture_output=True,
            timeout=PROCESS_SCAN_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise V03ManifestError(
            f"cannot gather the terminal process inventory: {error}"
        ) from error
    if (
        completed.returncode != 0
        or len(completed.stdout) > PROCESS_INVENTORY_MAX_BYTES
        or len(completed.stderr) > PROCESS_COMMAND_MAX_BYTES
    ):
        raise V03ManifestError(
            "terminal process inventory exceeded its bound or failed"
        )
    try:
        lines = completed.stdout.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise V03ManifestError(
            "terminal process inventory is not UTF-8"
        ) from error
    if len(lines) > PROCESS_INVENTORY_MAX_ROWS:
        raise V03ManifestError("terminal process inventory has too many rows")
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line in lines:
        match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s+(.+)", line)
        if match is None:
            raise V03ManifestError(
                "terminal process inventory contains an invalid row"
            )
        pid = int(match.group(1))
        ppid = int(match.group(2))
        command = match.group(3).strip()
        if (
            pid <= 0
            or ppid < 0
            or pid in seen
            or not command
            or len(command.encode()) > PROCESS_COMMAND_MAX_BYTES
        ):
            raise V03ManifestError(
                "terminal process inventory contains an unsafe identity"
            )
        seen.add(pid)
        rows.append({"pid": pid, "ppid": ppid, "command": command})
    if not rows:
        raise V03ManifestError("terminal process inventory is empty")
    return rows


def _process_role(command: str) -> str:
    if TRAINER_PROCESS_PATTERN.search(command):
        return "trainer"
    if SUPERVISOR_PROCESS_PATTERN.search(command):
        return "supervisor"
    if CAFFEINATE_PROCESS_PATTERN.search(command):
        return "caffeinate"
    if DASHBOARD_PROCESS_PATTERN.search(command):
        return "dashboard"
    return "unrelated"


def _process_closeout_evidence(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if (
        not rows
        or len(rows) > PROCESS_INVENTORY_MAX_ROWS
        or not all(isinstance(row, Mapping) for row in rows)
    ):
        raise V03ManifestError("terminal process inventory is invalid")
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        pid = row.get("pid")
        ppid = row.get("ppid")
        command = row.get("command")
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or isinstance(ppid, bool)
            or not isinstance(ppid, int)
            or ppid < 0
            or pid in seen
            or not isinstance(command, str)
            or not command
            or len(command.encode()) > PROCESS_COMMAND_MAX_BYTES
        ):
            raise V03ManifestError(
                "terminal process inventory contains an unsafe row"
            )
        seen.add(pid)
        normalized.append(
            {
                "pid": pid,
                "ppid": ppid,
                "command_sha256": hashlib.sha256(
                    command.encode()
                ).hexdigest(),
                "role": _process_role(command),
            }
        )
    normalized.sort(key=lambda item: item["pid"])
    prohibited = [
        dict(item)
        for item in normalized
        if item["role"] in PROHIBITED_PROCESS_ROLES
    ]
    if prohibited:
        identities = ", ".join(
            f"{item['role']}:{item['pid']}" for item in prohibited
        )
        raise V03ManifestError(
            "terminal process closeout still sees prohibited processes: "
            f"{identities}"
        )
    dashboards = [
        dict(item) for item in normalized if item["role"] == "dashboard"
    ]
    captured_at = _utc_now()
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "cohort_id": COHORT_ID,
        "kind": "terminal_process_closeout",
        "captured_at": captured_at,
        "collector": {
            "command": list(PROCESS_SCAN_COMMAND),
            "timeout_seconds": PROCESS_SCAN_TIMEOUT_SECONDS,
            "maximum_inventory_bytes": PROCESS_INVENTORY_MAX_BYTES,
            "maximum_rows": PROCESS_INVENTORY_MAX_ROWS,
            "observed_rows": len(normalized),
        },
        "policy": {
            "prohibited_roles": sorted(PROHIBITED_PROCESS_ROLES),
            "dashboard_may_remain": True,
            "commands_redacted_to_sha256": True,
        },
        "inventory": normalized,
        "inventory_sha256": _canonical_sha256(normalized),
        "prohibited_matches": [],
        "dashboard_processes": dashboards,
        "verdict": "clear",
    }


def _verified_process_closeout(
    root: Path,
    *,
    state: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = state.get("process_closeout")
    if not isinstance(binding, Mapping):
        raise V03ManifestError(
            "terminal process closeout evidence has not been sealed"
        )
    evidence_path = root / PROCESS_CLOSEOUT_NAME
    evidence = _read_json(
        evidence_path,
        label="v0.3 terminal process closeout",
        maximum=PROCESS_CLOSEOUT_MAX_BYTES,
    )
    inventory = evidence.get("inventory")
    dashboards = evidence.get("dashboard_processes")
    collector = evidence.get("collector")
    policy = evidence.get("policy")
    if (
        set(evidence)
        != {
            "schema_version",
            "protocol",
            "cohort_id",
            "kind",
            "captured_at",
            "collector",
            "policy",
            "inventory",
            "inventory_sha256",
            "prohibited_matches",
            "dashboard_processes",
            "verdict",
        }
        or evidence.get("schema_version") != 1
        or evidence.get("protocol") != PROTOCOL
        or evidence.get("cohort_id") != COHORT_ID
        or evidence.get("kind") != "terminal_process_closeout"
        or evidence.get("verdict") != "clear"
        or not isinstance(evidence.get("captured_at"), str)
        or not isinstance(collector, Mapping)
        or collector.get("command") != list(PROCESS_SCAN_COMMAND)
        or collector.get("timeout_seconds")
        != PROCESS_SCAN_TIMEOUT_SECONDS
        or collector.get("maximum_inventory_bytes")
        != PROCESS_INVENTORY_MAX_BYTES
        or collector.get("maximum_rows") != PROCESS_INVENTORY_MAX_ROWS
        or not isinstance(inventory, list)
        or not inventory
        or len(inventory) > PROCESS_INVENTORY_MAX_ROWS
        or collector.get("observed_rows") != len(inventory)
        or not isinstance(policy, Mapping)
        or policy.get("prohibited_roles")
        != sorted(PROHIBITED_PROCESS_ROLES)
        or policy.get("dashboard_may_remain") is not True
        or policy.get("commands_redacted_to_sha256") is not True
        or evidence.get("prohibited_matches") != []
        or not isinstance(dashboards, list)
        or _canonical_sha256(inventory)
        != evidence.get("inventory_sha256")
    ):
        raise V03ManifestError(
            "terminal process closeout contract changed"
        )
    seen: set[int] = set()
    expected_dashboards: list[dict[str, Any]] = []
    for item in inventory:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"pid", "ppid", "command_sha256", "role"}
            or isinstance(item.get("pid"), bool)
            or not isinstance(item.get("pid"), int)
            or item["pid"] <= 0
            or item["pid"] in seen
            or isinstance(item.get("ppid"), bool)
            or not isinstance(item.get("ppid"), int)
            or item["ppid"] < 0
            or item.get("role") not in PROCESS_ROLES
            or item.get("role") in PROHIBITED_PROCESS_ROLES
            or SHA256_PATTERN.fullmatch(
                str(item.get("command_sha256", ""))
            )
            is None
        ):
            raise V03ManifestError(
                "terminal process closeout inventory changed"
            )
        seen.add(item["pid"])
        if item["role"] == "dashboard":
            expected_dashboards.append(dict(item))
    if (
        [item["pid"] for item in inventory] != sorted(seen)
        or dashboards != expected_dashboards
    ):
        raise V03ManifestError(
            "terminal process closeout ordering changed"
        )
    digest = _sha256(evidence_path)
    expected_binding = {
        "path": PROCESS_CLOSEOUT_NAME,
        "sha256": digest,
        "inventory_sha256": evidence["inventory_sha256"],
        "captured_at": evidence["captured_at"],
        "observed_rows": len(inventory),
        "dashboard_pids": [item["pid"] for item in dashboards],
        "prohibited_count": 0,
        "verdict": "clear",
    }
    if dict(binding) != expected_binding:
        raise V03ManifestError(
            "terminal process closeout binding changed"
        )
    return expected_binding, evidence


def create_cohort(
    root: Path,
    *,
    media_root: Path,
    qualification_report: Path,
    source_commit: str,
    tag_object: str,
) -> dict[str, Any]:
    resolved = _regular_directory(root, label="v0.3 cohort root")
    resolved_media = _regular_directory(
        media_root,
        label="v0.3 media root",
    )
    commit = _object_id(source_commit, label="source commit")
    tag = _object_id(tag_object, label="tag object")
    with _mutation_lock(resolved):
        if any(
            entry.name != ".v03-manifest.lock"
            for entry in resolved.iterdir()
        ):
            raise V03ManifestError("fresh v0.3 cohort root must be empty")
        if any(resolved_media.iterdir()):
            raise V03ManifestError("fresh v0.3 media root must be empty")
        qualification = _verified_qualification_binding(
            qualification_report.expanduser().resolve(strict=True),
            source_commit=commit,
            tag_object=tag,
            root=resolved,
            media_root=resolved_media,
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
        state["contract_sha256"] = _sha256(
            resolved / "cohort-contract.json"
        )
        _exclusive_write(resolved / "cohort.json", state)
    return state


def start_arm(
    root: Path,
    *,
    source_commit: str,
    tag_object: str,
    arm_id: str,
) -> dict[str, Any]:
    if arm_id not in ARM_ORDER:
        raise V03ManifestError(f"unknown v0.3 arm: {arm_id}")
    resolved = _regular_directory(root, label="v0.3 cohort root")
    with _mutation_lock(resolved):
        resolved, contract, state = _load_verified(
            resolved,
            source_commit=source_commit,
            tag_object=tag_object,
        )
        if state["active_arm"] is not None:
            raise V03ManifestError("another v0.3 arm is already active")
        if state["phase"] != "ready":
            raise V03ManifestError("v0.3 cohort cannot start an arm now")
        _assert_fixed_order(state, arm_id)
        arm = _arm_state(state, arm_id)
        directory = resolved / arm["directory"]
        media_directory = Path(contract["roots"]["media"]) / arm[
            "media_directory"
        ]
        if (
            directory.exists()
            or directory.is_symlink()
            or media_directory.exists()
            or media_directory.is_symlink()
        ):
            raise V03ManifestError(
                f"fresh v0.3 arm target already exists: {arm_id}"
            )
        now = _utc_now()
        attempt = {
            "index": 0,
            "state": "training",
            "started_at": now,
            "finished_at": None,
        }
        arm["attempts"].append(attempt)
        arm["state"] = "training"
        state["active_arm"] = arm_id
        state["phase"] = "training"
        state["updated_at"] = now
        state["history"].append(
            {"event": "arm_started", "arm": arm_id, "at": now}
        )
        _atomic_write(resolved / "cohort.json", state)
    return {"arm": arm_id, "attempt": 0}


def _mark_integrity_failed(
    root: Path,
    state: dict[str, Any],
    *,
    arm_id: str | None,
    stage: str,
) -> None:
    now = _utc_now()
    if arm_id is not None:
        arm = _arm_state(state, arm_id)
        arm["state"] = "integrity_failed"
        if arm["attempts"]:
            arm["attempts"][-1].update(
                {"state": "integrity_failed", "finished_at": now}
            )
    state["active_arm"] = None
    state["phase"] = "integrity_failed"
    state["updated_at"] = now
    state["history"].append(
        {"event": "integrity_failed", "arm": arm_id, "stage": stage, "at": now}
    )
    _atomic_write(root / "cohort.json", state)


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
        raise V03ManifestError("invalid v0.3 arm outcome")
    if isinstance(trainer_exit_code, bool):
        raise V03ManifestError("trainer exit code must be an integer")
    exit_code = int(trainer_exit_code)
    resolved = _regular_directory(root, label="v0.3 cohort root")
    with _mutation_lock(resolved):
        resolved, contract, state = _load_verified(
            resolved,
            source_commit=source_commit,
            tag_object=tag_object,
        )
        arm = _arm_state(state, arm_id)
        if (
            state["active_arm"] != arm_id
            or arm["state"] != "training"
            or len(arm["attempts"]) != 1
        ):
            raise V03ManifestError(
                "only the one active fresh v0.3 arm may finish"
            )
        directory = resolved / arm["directory"]
        terminal: dict[str, Any] | None = None
        if outcome == "completed":
            if exit_code != 0:
                raise V03ManifestError(
                    "nonzero trainer exit cannot complete a v0.3 arm"
                )
            try:
                terminal, _report = _verified_arm_terminal(
                    directory,
                    arm_id=arm_id,
                    contract=contract,
                    contract_sha256=state["contract_sha256"],
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                _mark_integrity_failed(
                    resolved,
                    state,
                    arm_id=arm_id,
                    stage="arm_terminal_closeout",
                )
                raise V03ManifestError(
                    f"{arm_id} terminal evidence failed closed"
                ) from error
        elif outcome == "crashed" and exit_code == 0:
            raise V03ManifestError(
                "a crashed v0.3 arm requires nonzero trainer exit"
            )
        elif outcome == "interrupted":
            status = _read_json(
                directory / "status.json",
                label=f"{arm_id} interrupted status",
            )
            _validate_status_identity(
                status,
                arm_id=arm_id,
                contract=contract,
                contract_sha256=state["contract_sha256"],
            )
            if status.get("phase") != "interrupted":
                raise V03ManifestError(
                    "interrupted arm lacks interrupted trainer status"
                )
        now = _utc_now()
        arm["state"] = outcome
        arm["attempts"][-1].update(
            {
                "state": outcome,
                "finished_at": now,
                "trainer_exit_code": exit_code,
            }
        )
        arm["terminal"] = terminal
        state["active_arm"] = None
        if outcome == "completed":
            state["phase"] = (
                "awaiting_closeout"
                if all(item["state"] == "completed" for item in state["arms"])
                else "ready"
            )
        else:
            state["phase"] = "operationally_incomplete"
        state["updated_at"] = now
        state["history"].append(
            {
                "event": f"arm_{outcome}",
                "arm": arm_id,
                "trainer_exit_code": exit_code,
                "at": now,
            }
        )
        _atomic_write(resolved / "cohort.json", state)
    return {"arm": arm_id, "outcome": outcome, "phase": state["phase"]}


def seal_process_closeout(
    root: Path,
    *,
    source_commit: str,
    tag_object: str,
) -> dict[str, Any]:
    """Seal one post-training process snapshot before scientific closeout."""

    resolved = _regular_directory(root, label="v0.3 cohort root")
    with _mutation_lock(resolved):
        resolved, _contract, state = _load_verified(
            resolved,
            source_commit=source_commit,
            tag_object=tag_object,
        )
        if (
            state["phase"] != "awaiting_closeout"
            or state["active_arm"] is not None
            or [arm["state"] for arm in state["arms"]]
            != ["completed", "completed"]
        ):
            raise V03ManifestError(
                "process closeout requires both completed v0.3 arms"
            )
        evidence_path = resolved / PROCESS_CLOSEOUT_NAME
        if (
            state.get("process_closeout") is not None
            or evidence_path.exists()
            or evidence_path.is_symlink()
        ):
            raise V03ManifestError(
                "refusing to replace terminal process closeout evidence"
            )
        evidence = _process_closeout_evidence(_gather_process_rows())
        _exclusive_write(evidence_path, evidence)
        binding = {
            "path": PROCESS_CLOSEOUT_NAME,
            "sha256": _sha256(evidence_path),
            "inventory_sha256": evidence["inventory_sha256"],
            "captured_at": evidence["captured_at"],
            "observed_rows": len(evidence["inventory"]),
            "dashboard_pids": [
                item["pid"] for item in evidence["dashboard_processes"]
            ],
            "prohibited_count": 0,
            "verdict": "clear",
        }
        state["process_closeout"] = binding
        now = _utc_now()
        state["updated_at"] = now
        state["history"].append(
            {
                "event": "terminal_process_closeout_sealed",
                "sha256": binding["sha256"],
                "inventory_sha256": binding["inventory_sha256"],
                "at": now,
            }
        )
        _atomic_write(resolved / "cohort.json", state)
    return binding


def _paired_first_rollout(
    verified: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    sham = verified["sham"]["first_rollout"]
    candidate = verified["action-effect"]["first_rollout"]
    if sham != candidate:
        raise V03ManifestError(
            "v0.3 twins diverged before their first optimizer phase"
        )
    candidate_first_nonzero = verified["action-effect"][
        "context_encoder"
    ]["first_nonzero_child_trained_actions"]
    if (
        verified["sham"]["context_encoder"][
            "first_nonzero_child_trained_actions"
        ]
        is not None
        or (
            candidate_first_nonzero is not None
            and candidate_first_nonzero < v03.ROLLOUT_TRANSITIONS
        )
    ):
        raise V03ManifestError(
            "v0.3 context encoder diverged before its first optimizer phase"
        )
    return {
        "transitions": v03.ROLLOUT_TRANSITIONS,
        "identical_before_first_optimizer": True,
        "identity": dict(sham),
        "candidate_first_encoder_nonzero_actions": candidate_first_nonzero,
        "divergence_before_first_optimizer": False,
    }


def finalize_cohort(
    root: Path,
    *,
    source_commit: str,
    tag_object: str,
) -> dict[str, Any]:
    resolved = _regular_directory(root, label="v0.3 cohort root")
    with _mutation_lock(resolved):
        resolved, contract, state = _load_verified(
            resolved,
            source_commit=source_commit,
            tag_object=tag_object,
        )
        if (
            state["phase"] != "awaiting_closeout"
            or state["active_arm"] is not None
            or [arm["state"] for arm in state["arms"]]
            != ["completed", "completed"]
        ):
            raise V03ManifestError(
                "v0.3 closeout requires both complete arms in fixed order"
            )
        process_closeout, _process_evidence = (
            _verified_process_closeout(resolved, state=state)
        )
        # Re-scan immediately before closeout.  We intentionally do not
        # require unrelated PID equality with the earlier sealed snapshot;
        # only a newly appeared prohibited role blocks finalization.
        finalization_process_evidence = _process_closeout_evidence(
            _gather_process_rows()
        )
        process_finalization_recheck = {
            "captured_at": finalization_process_evidence["captured_at"],
            "inventory_sha256": finalization_process_evidence[
                "inventory_sha256"
            ],
            "observed_rows": len(
                finalization_process_evidence["inventory"]
            ),
            "dashboard_pids": [
                item["pid"]
                for item in finalization_process_evidence[
                    "dashboard_processes"
                ]
            ],
            "prohibited_count": 0,
            "verdict": "clear",
        }
        try:
            verified: dict[str, dict[str, Any]] = {}
            reports: dict[str, dict[str, Any]] = {}
            arm_sizes: dict[str, int] = {}
            for arm_id in ARM_ORDER:
                directory = resolved / arm_id
                evidence, report = _verified_arm_terminal(
                    directory,
                    arm_id=arm_id,
                    contract=contract,
                    contract_sha256=state["contract_sha256"],
                )
                recorded = _arm_state(state, arm_id).get("terminal")
                if (
                    not isinstance(recorded, Mapping)
                    or recorded != evidence
                ):
                    raise V03ManifestError(
                        f"{arm_id} evidence changed after arm closeout"
                    )
                size = _tree_size(
                    directory,
                    label=f"{arm_id} scientific artifacts",
                )
                if size > LINEAGE_CAP_BYTES:
                    raise V03ManifestError(
                        f"{arm_id} exceeds the v0.3 lineage cap"
                    )
                verified[arm_id] = evidence
                reports[arm_id] = report
                arm_sizes[arm_id] = size
            paired = _paired_first_rollout(verified)
            selection = v03.select_architecture(
                {
                    arm_id: reports[arm_id]["controller"][
                        "exam_records"
                    ]
                    for arm_id in ARM_ORDER
                }
            )
            if (
                selection.get("verdict")
                not in {"architecture_selected", "architecture_failed"}
                or selection.get(
                    "development_checkpoint_reuse_authorized"
                )
                is not False
                or selection.get("u3_authorized") is not False
            ):
                raise V03ManifestError(
                    "v0.3 selector violated the Stage-A stop rule"
                )
            media_root = _regular_directory(
                Path(contract["roots"]["media"]),
                label="v0.3 media root",
            )
            scientific_bytes = _tree_size(
                resolved,
                label="v0.3 cohort",
            )
            media_bytes = _tree_size(media_root, label="v0.3 media")
            if (
                scientific_bytes > COHORT_SCIENTIFIC_CAP_BYTES
                or media_bytes > MEDIA_CAP_BYTES
                or scientific_bytes + media_bytes
                > COMBINED_PLANNED_CAP_BYTES
            ):
                raise V03ManifestError("v0.3 storage cap exceeded")
            report = {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "cohort_id": COHORT_ID,
                "verdict": selection["verdict"],
                "completed_at": _utc_now(),
                "development_only": True,
                "source": dict(contract["source"]),
                "preregistration": dict(contract["preregistration"]),
                "qualification": dict(contract["qualification"]),
                "parent": dict(contract["parent"]),
                "cohort_contract": "cohort-contract.json",
                "cohort_contract_sha256": state["contract_sha256"],
                "process_closeout": process_closeout,
                "process_finalization_recheck": (
                    process_finalization_recheck
                ),
                "arm_evidence": verified,
                "paired_first_rollout": paired,
                "selection": selection,
                "selected_architecture": selection[
                    "selected_architecture"
                ],
                "checkpoint_rule": {
                    "stage_a_checkpoint_reuse_authorized": False,
                    "successor_checkpoint": None,
                    "replication_protocol_authorized": selection[
                        "replication_protocol_authorized"
                    ],
                    "u3_authorized": False,
                },
                "storage": {
                    "arm_bytes": arm_sizes,
                    "scientific_bytes_before_report": scientific_bytes,
                    "media_bytes": media_bytes,
                    "within_caps": True,
                    "caps": contract["matched_design"]["storage_caps"],
                },
            }
            report_path = resolved / "report.json"
            integrity_path = resolved / "report.integrity.json"
            if (
                report_path.exists()
                or report_path.is_symlink()
                or integrity_path.exists()
                or integrity_path.is_symlink()
            ):
                raise V03ManifestError(
                    "refusing to replace v0.3 terminal closeout evidence"
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
                    arm: verified[arm]["report_sha256"]
                    for arm in ARM_ORDER
                },
                "paired_first_rollout_sha256": _canonical_sha256(paired),
                "selection_sha256": _canonical_sha256(selection),
                "process_closeout_sha256": process_closeout["sha256"],
                "process_inventory_sha256": process_closeout[
                    "inventory_sha256"
                ],
                "process_finalization_recheck_sha256": (
                    _canonical_sha256(process_finalization_recheck)
                ),
            }
            _exclusive_write(integrity_path, integrity)
            integrity_sha256 = _sha256(integrity_path)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _mark_integrity_failed(
                resolved,
                state,
                arm_id=None,
                stage="cohort_terminal_closeout",
            )
            raise V03ManifestError(
                "v0.3 terminal closeout failed closed"
            ) from error
        now = _utc_now()
        state["phase"] = "completed"
        state["terminal_report"] = {
            "path": "report.json",
            "sha256": report_sha256,
            "integrity": "report.integrity.json",
            "integrity_sha256": integrity_sha256,
            "verdict": report["verdict"],
            "selected_architecture": (
                "action-effect"
                if report["selected_architecture"] is not None
                else None
            ),
            "stage_a_checkpoint_reuse_authorized": False,
            "u3_authorized": False,
            "process_closeout_sha256": process_closeout["sha256"],
        }
        state["updated_at"] = now
        state["history"].append(
            {
                "event": "cohort_terminal_closeout_verified",
                "verdict": report["verdict"],
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
    if state["phase"] in {
        "operationally_incomplete",
        "integrity_failed",
    }:
        raise V03ManifestError(
            "v0.3 is terminal and non-resumable; a replacement needs a new "
            "source, tag, attempt identity, root, and two fresh twins"
        )
    if state["active_arm"] is not None:
        raise V03ManifestError("a v0.3 arm is already active")
    states = [arm["state"] for arm in state["arms"]]
    if states == ["completed", "completed"]:
        return {
            "done": True,
            "closeout_required": state["phase"] == "awaiting_closeout",
        }
    pending = next(
        (arm for arm in state["arms"] if arm["state"] == "pending"),
        None,
    )
    if pending is None:
        raise V03ManifestError("v0.3 has no legal next arm")
    _assert_fixed_order(state, pending["id"])
    return {
        "done": False,
        "arm": pending["id"],
        "attempt": 0,
        "run_dir": str(resolved / pending["directory"]),
        "media_dir": str(
            Path(contract["roots"]["media"])
            / pending["media_directory"]
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def identity(command: argparse.ArgumentParser) -> None:
        command.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        command.add_argument("--source-commit", required=True)
        command.add_argument("--tag-object", required=True)

    create = subparsers.add_parser("create")
    identity(create)
    create.add_argument(
        "--media-root",
        type=Path,
        default=DEFAULT_MEDIA_ROOT,
    )
    create.add_argument(
        "--qualification-report",
        type=Path,
        default=QUALIFICATION_REPORT,
    )

    start = subparsers.add_parser("start-arm")
    identity(start)
    start.add_argument("--arm", choices=ARM_ORDER, required=True)

    finish = subparsers.add_parser("finish-arm")
    identity(finish)
    finish.add_argument("--arm", choices=ARM_ORDER, required=True)
    finish.add_argument(
        "--outcome",
        choices=("completed", "interrupted", "crashed"),
        required=True,
    )
    finish.add_argument("--trainer-exit-code", type=int, required=True)

    process_closeout = subparsers.add_parser("seal-process-closeout")
    identity(process_closeout)

    finalize = subparsers.add_parser("finalize")
    identity(finalize)

    plan = subparsers.add_parser("next")
    identity(plan)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "create":
        result = create_cohort(
            args.root,
            media_root=args.media_root,
            qualification_report=args.qualification_report,
            source_commit=args.source_commit,
            tag_object=args.tag_object,
        )
    elif args.command == "start-arm":
        result = start_arm(
            args.root,
            source_commit=args.source_commit,
            tag_object=args.tag_object,
            arm_id=args.arm,
        )
    elif args.command == "finish-arm":
        result = finish_arm(
            args.root,
            source_commit=args.source_commit,
            tag_object=args.tag_object,
            arm_id=args.arm,
            outcome=args.outcome,
            trainer_exit_code=args.trainer_exit_code,
        )
    elif args.command == "seal-process-closeout":
        result = seal_process_closeout(
            args.root,
            source_commit=args.source_commit,
            tag_object=args.tag_object,
        )
    elif args.command == "finalize":
        result = finalize_cohort(
            args.root,
            source_commit=args.source_commit,
            tag_object=args.tag_object,
        )
    else:
        result = next_plan(
            args.root,
            source_commit=args.source_commit,
            tag_object=args.tag_object,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
