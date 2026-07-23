"""External annotated preregistration anchor for U2r stability training.

The tag created by this module is the public, immutable boundary between U2r
implementation and U2r training.  It binds the clean implementation commit to
the exact failed U2 parent, its frozen confirmation failure, the measured
training exclusion set, the remaining action budget, and the terminal-only
stability rule.  It never grants access to a confirmation seed stream.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dungeon_apprentice import v02_u2r

ANCHOR_SCHEMA_VERSION = 1
ANCHOR_PROTOCOL = "dungeon-apprentice-v0.2-u2r-stability-anchor"
ANCHOR_TAG = "u2r-stability-v0.2-u2r-20260723"
ANCHOR_REMOTE = "origin"
EXPECTED_ORIGIN_URL = "https://github.com/PeteAndrews1289/dungeon-apprentice.git"
PROTOCOL_REPOSITORY_PATH = "docs/protocol-v0.2-u2r-stability-remediation.md"

PARENT_POLICY_MEMBER_SHA256 = "0783c6955ff1d62b87980d7801c8fbaedcb010174d640c4ee0f07f340371b6db"
PARENT_OPTIMIZER_STATE_SHA256 = "e821dec631c76e32ce3ac2a0fc3aac70ae55dbe4cd625ed29a11b5b35febaba2"
TERMINAL_OPTIMIZER_UPDATES = 3_584
UNAVAILABLE_ORIGINAL_ACTIVE_LAYOUTS_UPPER_BOUND = 12
UNAVAILABLE_ORIGINAL_ACTIVE_LAYOUTS_STATUS = (
    "unauthenticated_unavailable_not_in_static_exclusion_set"
)

CONSUMED_CONFIRMATION_RANGES = {
    "separated_u2": [15_200_000, 15_209_999],
    "navigate": [15_210_000, 15_219_999],
    "visible_u0": [15_220_000, 15_229_999],
    "local_u1": [15_230_000, 15_239_999],
}
FRESH_CONFIRMATION_RANGES = {
    "separated_u2": [15_240_000, 15_249_999],
    "navigate": [15_250_000, 15_259_999],
    "visible_u0": [15_260_000, 15_269_999],
    "local_u1": [15_270_000, 15_279_999],
}

STABILITY_RULE = {
    "terminal_only": True,
    "all_eleven_windows_required": True,
    "terminal_pair_child_actions": [1_015_808, 1_048_576],
    "normal_practice_required_on_both": True,
    "allocation_valid_required_on_both": True,
    "recovery_inactive_required_on_both": True,
    "existing_capability_floors": {
        "navigate/full": {"overall": 68, "panel": 34},
        "unlock/u0-visible": {"overall": 68, "panel": 32},
        "unlock/u1-local": {"overall": 68, "panel": 32},
        "unlock/u2-separated": {"overall": 68, "panel": 32},
    },
    "u2_stability_overall": 72,
    "u2_stability_panel": 34,
    "ineffective_interaction_lessons": [
        "unlock/u0-visible",
        "unlock/u1-local",
        "unlock/u2-separated",
    ],
    "mean_ineffective_interactions_max": 3.0,
    "terminal_success_decline_max": 2,
}

ANCHOR_FIELDS = frozenset(
    {
        "schema_version",
        "anchor_protocol",
        "training_protocol",
        "tag",
        "remote",
        "remote_url",
        "source_commit",
        "protocol_document",
        "protocol_document_sha256",
        "parent",
        "failed_confirmation",
        "static_exclusions",
        "budget",
        "seeds",
        "stability_rule",
        "confirmation",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT = re.compile(r"^[0-9a-f]{40,64}$")
Runner = Callable[..., subprocess.CompletedProcess[str]]


class U2rAnchorError(RuntimeError):
    """Raised when the prospective U2r anchor is absent or inconsistent."""


@dataclass(frozen=True)
class U2rAnchor:
    """Verified identity of the local and remote annotated U2r tag."""

    tag: str
    tag_object: str
    remote: str
    remote_url: str
    source_commit: str
    protocol_document: str
    protocol_document_sha256: str
    static_exclusion_set_sha256: str
    static_exclusion_layouts: int
    unavailable_original_active_layouts_upper_bound: int
    unavailable_original_active_layouts_status: str
    parent_checkpoint_sha256: str
    confirmation_report_sha256: str
    additional_action_budget: int
    terminal_child_actions: int
    terminal_lifetime_actions: int
    terminal_optimizer_updates: int
    algorithm_seed: int
    worker_streams: tuple[int, int, int, int]
    fresh_confirmation_ranges: Mapping[str, list[int]]

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["worker_streams"] = list(self.worker_streams)
        return value


@dataclass(frozen=True)
class U2rResumePlan:
    """One mechanically selected append-only U2r resume segment."""

    source_directory: str
    checkpoint: str
    checkpoint_sha256: str
    source_segment_index: int
    next_segment_index: int
    next_run_name: str
    child_trained_actions: int
    remediation_trained_actions: int
    lifetime_trained_actions: int
    optimizer_updates: int

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise U2rAnchorError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _require_commit(value: Any) -> str:
    if not isinstance(value, str) or _GIT_OBJECT.fullmatch(value) is None:
        raise U2rAnchorError("U2r source commit is invalid")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise U2rAnchorError(f"{label} must be a positive integer")
    return value


def _git(
    repository: Path,
    arguments: Sequence[str],
    *,
    runner: Runner,
) -> str:
    try:
        completed = runner(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise U2rAnchorError(f"cannot verify U2r anchor with git {' '.join(arguments)}") from error
    return completed.stdout.rstrip("\n")


def _clean_head(
    repository: Path,
    *,
    expected_source_commit: str,
    runner: Runner,
) -> str:
    expected = _require_commit(expected_source_commit)
    head = _git(
        repository,
        ["rev-parse", "--verify", "HEAD^{commit}"],
        runner=runner,
    )
    if head != expected:
        raise U2rAnchorError("active source differs from the U2r source commit")
    if _git(
        repository,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        runner=runner,
    ):
        raise U2rAnchorError("U2r preregistration requires clean committed source")
    return head


def protocol_document_sha256(repository: Path) -> str:
    """Measure the committed prospective protocol document without following a link."""

    path = repository.expanduser().resolve() / PROTOCOL_REPOSITORY_PATH
    if path.is_symlink() or not path.is_file():
        raise U2rAnchorError("U2r protocol document is missing or unsafe")
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise U2rAnchorError("cannot read the U2r protocol document") from error


def _regular_absolute_directory(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    if absolute.is_symlink() or not absolute.is_dir():
        raise U2rAnchorError(f"{label} is missing or unsafe: {absolute}")
    return absolute


def _read_safe_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise U2rAnchorError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise U2rAnchorError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise U2rAnchorError(f"{label} is not a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise U2rAnchorError(f"cannot hash U2r artifact: {path}") from error


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _safe_relative_artifact(
    root: Path,
    relative_value: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise U2rAnchorError(f"{label} path is missing")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise U2rAnchorError(f"{label} path escapes its segment")
    candidate = root.joinpath(relative)
    if candidate.is_symlink() or not candidate.is_file():
        raise U2rAnchorError(f"{label} is missing or unsafe: {candidate}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise U2rAnchorError(f"{label} path escapes its segment") from error
    return resolved


def _discover_resume_chain(root: Path) -> dict[int, Path]:
    base = v02_u2r.DEFAULT_RUN_NAME
    pattern = re.compile(rf"^{re.escape(base)}-resume-([1-9][0-9]*)$")
    indexed: dict[int, Path] = {}
    for child in root.iterdir():
        if child.name == base:
            index = 0
        else:
            match = pattern.fullmatch(child.name)
            if match is None:
                if child.name.startswith(base):
                    raise U2rAnchorError(f"unrecognized U2r lineage-chain entry: {child.name}")
                continue
            index = int(match.group(1))
        if child.is_symlink() or not child.is_dir():
            raise U2rAnchorError(f"U2r segment is missing or unsafe: {child}")
        if index in indexed:
            raise U2rAnchorError("U2r lineage chain has duplicate segment indexes")
        indexed[index] = child
    if not indexed or 0 not in indexed:
        raise U2rAnchorError("U2r resume requires the canonical initial segment")
    highest = max(indexed)
    if set(indexed) != set(range(highest + 1)):
        raise U2rAnchorError("U2r resume segment chain is not contiguous")
    return indexed


def _verify_exam_case_binding(
    directory: Path,
    archive: Path,
    sidecar: Path,
    record: Mapping[str, Any],
) -> Mapping[str, Any]:
    binding = record.get("case_evidence")
    cases = archive.with_suffix(".cases.json")
    if not isinstance(binding, Mapping):
        raise U2rAnchorError("U2r controller exam lacks case-evidence binding")
    if cases.is_symlink() or not cases.is_file():
        raise U2rAnchorError("U2r exam case-evidence file is missing or unsafe")
    measured_digest = _file_sha256(cases)
    document = _read_safe_json(cases, "U2r exam case evidence")
    summary = document.get("summary")
    records = document.get("records")
    evaluations = record.get("evaluations")
    checkpoint_sha256 = _file_sha256(archive)
    sidecar_sha256 = _file_sha256(sidecar)
    child_actions = int(record.get("child_trained_actions", -1))
    try:
        relative_cases = str(cases.resolve().relative_to(directory.resolve()))
        relative_checkpoint = str(archive.resolve().relative_to(directory.resolve()))
    except ValueError as error:
        raise U2rAnchorError("U2r exam case evidence escapes its segment") from error
    if (
        document.get("schema_version") != v02_u2r.CASE_EVIDENCE_SCHEMA_VERSION
        or document.get("protocol") != v02_u2r.PROTOCOL
        or document.get("kind") != "immutable_exam_cases"
        or document.get("checkpoint") != archive.name
        or document.get("checkpoint_sha256") != checkpoint_sha256
        or document.get("checkpoint_sidecar_sha256") != sidecar_sha256
        or int(document.get("child_trained_actions", -1)) != child_actions
        or not isinstance(records, list)
        or not all(isinstance(item, Mapping) for item in records)
        or not isinstance(summary, Mapping)
        or not isinstance(evaluations, Mapping)
    ):
        raise U2rAnchorError("U2r exam case-evidence document binding changed")
    try:
        recomputed_summary = v02_u2r.verify_exam_case_evidence(
            evaluations,
            records,
            expected_seeds={
                lesson: tuple(
                    v02_u2r.lessons.LESSON_SPECS[lesson].validation_seed_base + index
                    for index in range(v02_u2r.EVALUATION_SEED_COUNT)
                )
                for lesson in v02_u2r.lessons.LessonId
            },
        )
    except (v02_u2r.U2rProtocolError, KeyError, TypeError, ValueError) as error:
        raise U2rAnchorError("U2r exam cases do not reproduce the controller evaluation") from error
    if dict(summary) != recomputed_summary:
        raise U2rAnchorError("U2r exam case-evidence summary does not match its raw cases")
    expected_binding = recomputed_summary | {
        "path": relative_cases,
        "file_sha256": measured_digest,
        "checkpoint": relative_checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_sidecar_sha256": sidecar_sha256,
        "child_trained_actions": child_actions,
    }
    if dict(binding) != expected_binding:
        raise U2rAnchorError("U2r controller case-evidence binding changed")
    _require_sha256(
        binding.get("case_set_sha256"),
        "U2r exam case set",
    )
    return binding


def _verify_segment_initial_bundle(
    directory: Path,
    manifest: Mapping[str, Any],
) -> Mapping[str, str]:
    checkpoint = _safe_relative_artifact(
        directory,
        "checkpoints/initial.zip",
        label="U2r segment initial checkpoint",
    )
    sidecar_path = checkpoint.with_suffix(".json")
    integrity_path = checkpoint.with_suffix(".integrity.json")
    sidecar = _read_safe_json(sidecar_path, "U2r segment initial sidecar")
    integrity = _read_safe_json(
        integrity_path,
        "U2r segment initial integrity",
    )
    checkpoint_sha256 = _file_sha256(checkpoint)
    progress = sidecar.get("progress")
    model_state = sidecar.get("model_state")
    segment = manifest.get("segment")
    if not isinstance(segment, Mapping):
        raise U2rAnchorError("U2r segment initial identity is missing")
    try:
        start_child_actions = int(segment["start_child_trained_actions"])
        start_remediation_actions = int(segment["start_remediation_trained_actions"])
        start_lifetime_actions = int(segment["start_lifetime_trained_actions"])
    except (KeyError, TypeError, ValueError) as error:
        raise U2rAnchorError("U2r segment start counters are incomplete") from error
    expected_updates = (
        v02_u2r.SOURCE_OPTIMIZER_UPDATES
        + start_remediation_actions // v02_u2r.ROLLOUT_TRANSITIONS * v02_u2r.PPO_EPOCHS
    )
    if (
        sidecar.get("schema_version") != v02_u2r.CHECKPOINT_SCHEMA_VERSION
        or sidecar.get("protocol") != v02_u2r.PROTOCOL
        or sidecar.get("kind") != "initial"
        or sidecar.get("resume_eligible") is not True
        or sidecar.get("checkpoint_sha256") != checkpoint_sha256
        or sidecar.get("source") != manifest.get("source")
        or sidecar.get("effective_config") != manifest.get("effective_config")
        or sidecar.get("parent") != manifest.get("parent")
        or sidecar.get("qualification") != manifest.get("qualification")
        or sidecar.get("exclusions") != manifest.get("exclusions")
        or sidecar.get("external_preregistration") != manifest.get("external_preregistration")
        or sidecar.get("storage_mount") != manifest.get("storage_mount")
        or sidecar.get("segment") != segment
        or not isinstance(progress, Mapping)
        or int(progress.get("child_trained_actions", -1)) != start_child_actions
        or int(progress.get("remediation_trained_actions", -1)) != start_remediation_actions
        or int(progress.get("lifetime_trained_actions", -1)) != start_lifetime_actions
        or int(progress.get("optimizer_updates", -1)) != expected_updates
        or not isinstance(model_state, Mapping)
    ):
        raise U2rAnchorError("U2r segment initial bundle identity changed")
    normalized_model_state = {
        "policy_tensor_sha256": _require_sha256(
            model_state.get("policy_tensor_sha256"),
            "U2r segment initial policy",
        ),
        "optimizer_state_sha256": _require_sha256(
            model_state.get("optimizer_state_sha256"),
            "U2r segment initial optimizer",
        ),
    }
    if dict(model_state) != normalized_model_state:
        raise U2rAnchorError("U2r segment initial model-state fields changed")
    if (
        integrity.get("schema_version") != v02_u2r.CHECKPOINT_SCHEMA_VERSION
        or integrity.get("protocol") != v02_u2r.PROTOCOL
        or integrity.get("checkpoint") != checkpoint.name
        or integrity.get("checkpoint_sha256") != checkpoint_sha256
        or integrity.get("sidecar") != sidecar_path.name
        or integrity.get("sidecar_sha256") != _file_sha256(sidecar_path)
    ):
        raise U2rAnchorError("U2r segment initial integrity binding changed")
    return normalized_model_state


def _verify_interruption_evidence(
    directory: Path,
    manifest: Mapping[str, Any],
    status: Mapping[str, Any],
    checkpoint: Path,
    checkpoint_sidecar: Mapping[str, Any],
) -> Mapping[str, Any]:
    interruption_path = _safe_relative_artifact(
        directory,
        "interruption.json",
        label="U2r interruption evidence",
    )
    integrity_path = _safe_relative_artifact(
        directory,
        "interruption.integrity.json",
        label="U2r interruption integrity",
    )
    document = _read_safe_json(
        interruption_path,
        "U2r interruption evidence",
    )
    integrity = _read_safe_json(
        integrity_path,
        "U2r interruption integrity",
    )
    binding = status.get("interruption")
    workers = document.get("active_workers")
    if (
        not isinstance(binding, Mapping)
        or set(binding)
        != {
            "path",
            "sha256",
            "integrity",
            "integrity_sha256",
            "phase",
            "active_workers_sha256",
        }
        or binding.get("path") != interruption_path.name
        or binding.get("sha256") != _file_sha256(interruption_path)
        or binding.get("integrity") != integrity_path.name
        or binding.get("integrity_sha256") != _file_sha256(integrity_path)
        or binding.get("phase") != status.get("phase")
        or integrity.get("schema_version") != v02_u2r.CHECKPOINT_SCHEMA_VERSION
        or integrity.get("protocol") != v02_u2r.PROTOCOL
        or integrity.get("interruption") != interruption_path.name
        or integrity.get("interruption_sha256") != _file_sha256(interruption_path)
        or document.get("schema_version") != v02_u2r.CHECKPOINT_SCHEMA_VERSION
        or document.get("protocol") != v02_u2r.PROTOCOL
        or document.get("kind") != "process_interruption_evidence"
        or document.get("phase") != status.get("phase")
        or document.get("resume_permitted") is not True
        or document.get("resume_blocker") is not None
        or document.get("source") != manifest.get("source")
        or document.get("effective_config") != manifest.get("effective_config")
        or document.get("parent") != manifest.get("parent")
        or document.get("qualification") != manifest.get("qualification")
        or document.get("exclusions") != manifest.get("exclusions")
        or document.get("external_preregistration") != manifest.get("external_preregistration")
        or document.get("storage_mount") != manifest.get("storage_mount")
        or document.get("segment") != manifest.get("segment")
        or not isinstance(workers, list)
        or len(workers) != v02_u2r.WORKERS
        or not all(isinstance(worker, Mapping) for worker in workers)
        or {int(worker.get("worker_index", -1)) for worker in workers}
        != set(range(v02_u2r.WORKERS))
        or document.get("active_workers_sha256") != _canonical_json_sha256(workers)
        or binding.get("active_workers_sha256") != document.get("active_workers_sha256")
    ):
        raise U2rAnchorError("U2r interruption evidence binding changed")
    checkpoint_sidecar_path = checkpoint.with_suffix(".json")
    checkpoint_integrity_path = checkpoint.with_suffix(".integrity.json")
    expected_resume_boundary = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "sidecar": str(checkpoint_sidecar_path),
        "sidecar_sha256": _file_sha256(checkpoint_sidecar_path),
        "integrity": str(checkpoint_integrity_path),
        "integrity_sha256": _file_sha256(checkpoint_integrity_path),
        "model_state": checkpoint_sidecar.get("model_state"),
        "progress": checkpoint_sidecar.get("progress"),
    }
    resume_boundary = document.get("resume_boundary")
    if (
        not isinstance(resume_boundary, Mapping)
        or dict(resume_boundary) != expected_resume_boundary
    ):
        raise U2rAnchorError("U2r interruption resume boundary changed")
    ledger = document.get("episode_start_ledger")
    ledger_path = _safe_relative_artifact(
        directory,
        "episode-starts.jsonl",
        label="U2r episode-start ledger",
    )
    if (
        not isinstance(ledger, Mapping)
        or ledger.get("path") != ledger_path.name
        or ledger.get("sha256") != _file_sha256(ledger_path)
    ):
        raise U2rAnchorError("U2r interruption ledger binding changed")
    try:
        records = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise U2rAnchorError("cannot read U2r interruption ledger") from error
    if not all(isinstance(record, Mapping) for record in records):
        raise U2rAnchorError("U2r interruption ledger records are invalid")
    inventory = {
        "record_count": len(records),
        "episode_start_count": sum(record.get("type") == "episode_start" for record in records),
        "interruption_abandonment_count": sum(
            record.get("type") == "episode_abandoned_at_interruption" for record in records
        ),
        "resume_abandonment_count": sum(
            record.get("type") == "episode_abandoned_on_resume" for record in records
        ),
    }
    if {key: ledger.get(key) for key in inventory} != inventory or inventory[
        "interruption_abandonment_count"
    ] < v02_u2r.WORKERS:
        raise U2rAnchorError("U2r interruption ledger inventory changed")
    at_signal = document.get("at_signal")
    if not isinstance(at_signal, Mapping):
        raise U2rAnchorError("U2r interruption counters are missing")
    try:
        signal_collected = int(at_signal["collected_actions"])
        signal_trained = int(at_signal["trained_actions"])
        signal_child = int(at_signal["child_trained_actions"])
        signal_updates = int(at_signal["optimizer_updates"])
    except (KeyError, TypeError, ValueError) as error:
        raise U2rAnchorError("U2r interruption counters are incomplete") from error
    if (
        signal_collected < signal_trained
        or signal_trained < v02_u2r.SOURCE_LIFETIME_ACTIONS
        or signal_child != signal_trained - v02_u2r.SOURCE_U1_INHERITED_ACTIONS
        or signal_updates < v02_u2r.SOURCE_OPTIMIZER_UPDATES
    ):
        raise U2rAnchorError("U2r interruption counters changed")
    segment = manifest.get("segment")
    expected_streams = (
        segment.get("segment_worker_streams") if isinstance(segment, Mapping) else None
    )
    source = manifest.get("source")
    source_commit = source.get("commit") if isinstance(source, Mapping) else None
    if (
        not isinstance(expected_streams, list)
        or len(expected_streams) != v02_u2r.WORKERS
        or any(
            int(worker.get("worker_stream", -1)) != int(expected_streams[index])
            or worker.get("checkpoint_source_commit") != source_commit
            or int(worker.get("checkpoint_lifetime_trained_actions", -1)) != signal_trained
            or int(worker.get("checkpoint_child_trained_actions", -1)) != signal_child
            for index, worker in enumerate(workers)
        )
    ):
        raise U2rAnchorError("U2r interruption active-worker provenance changed")
    try:
        segment_index = int(segment.get("index", -1))
    except (TypeError, ValueError) as error:
        raise U2rAnchorError("U2r interruption segment index changed") from error
    if segment_index == 0:
        try:
            v02_u2r._build_parent_baseline_inventory(
                directory,
                cohort_root=directory.parent.resolve(),
            )
        except (v02_u2r.U2rProtocolError, OSError, ValueError) as error:
            raise U2rAnchorError(
                "U2r segment zero lacks its authenticated parent baseline"
            ) from error
    return resume_boundary


def _verify_carried_exams(
    directory: Path,
    manifest: Mapping[str, Any],
    *,
    expected_count: int,
    controller_records: Sequence[Mapping[str, Any]],
) -> None:
    carried = manifest.get("carried_exams")
    if not isinstance(carried, list) or len(carried) != expected_count:
        raise U2rAnchorError("U2r carried-exam inventory is incomplete")
    records_by_boundary = {
        int(record.get("child_trained_actions", -1)): record for record in controller_records
    }
    seen: set[str] = set()
    for position, raw in enumerate(carried, start=1):
        if not isinstance(raw, Mapping):
            raise U2rAnchorError("U2r carried-exam identity is invalid")
        expected_child_actions = (
            v02_u2r.SOURCE_CHILD_ACTIONS + position * v02_u2r.EVALUATION_INTERVAL
        )
        expected_name = f"exam-{expected_child_actions:07d}.zip"
        archive = _safe_relative_artifact(
            directory,
            raw.get("path"),
            label="U2r carried exam",
        )
        if (
            archive.name in seen
            or archive.parent != directory / "checkpoints" / "rolling"
            or archive.name != expected_name
            or int(raw.get("child_trained_actions", -1)) != expected_child_actions
        ):
            raise U2rAnchorError("U2r carried-exam path identity changed")
        seen.add(archive.name)
        sidecar = archive.with_suffix(".json")
        integrity = archive.with_suffix(".integrity.json")
        cases = archive.with_suffix(".cases.json")
        if (
            sidecar.is_symlink()
            or not sidecar.is_file()
            or integrity.is_symlink()
            or not integrity.is_file()
            or cases.is_symlink()
            or not cases.is_file()
            or raw.get("checkpoint_sha256") != _file_sha256(archive)
            or raw.get("sidecar_sha256") != _file_sha256(sidecar)
            or raw.get("integrity_sha256") != _file_sha256(integrity)
            or raw.get("case_evidence_sha256") != _file_sha256(cases)
        ):
            raise U2rAnchorError("U2r carried-exam bundle digest changed")
        record = records_by_boundary.get(expected_child_actions)
        if not isinstance(record, Mapping):
            raise U2rAnchorError("U2r carried exam has no controller decision record")
        binding = _verify_exam_case_binding(
            directory,
            archive,
            sidecar,
            record,
        )
        if raw.get("case_set_sha256") != binding.get("case_set_sha256"):
            raise U2rAnchorError("U2r carried-exam case-set identity changed")
        integrity_value = _read_safe_json(
            integrity,
            "U2r carried-exam integrity",
        )
        sidecar_value = _read_safe_json(
            sidecar,
            "U2r carried-exam sidecar",
        )
        progress = sidecar_value.get("progress")
        if (
            integrity_value.get("schema_version") != v02_u2r.CHECKPOINT_SCHEMA_VERSION
            or integrity_value.get("protocol") != v02_u2r.PROTOCOL
            or integrity_value.get("checkpoint") != archive.name
            or integrity_value.get("checkpoint_sha256") != raw.get("checkpoint_sha256")
            or integrity_value.get("sidecar") != sidecar.name
            or integrity_value.get("sidecar_sha256") != raw.get("sidecar_sha256")
            or sidecar_value.get("schema_version") != v02_u2r.CHECKPOINT_SCHEMA_VERSION
            or sidecar_value.get("protocol") != v02_u2r.PROTOCOL
            or sidecar_value.get("kind") != "exam"
            or sidecar_value.get("resume_eligible") is not False
            or sidecar_value.get("checkpoint_sha256") != raw.get("checkpoint_sha256")
            or sidecar_value.get("source") != manifest.get("source")
            or sidecar_value.get("effective_config") != manifest.get("effective_config")
            or sidecar_value.get("parent") != manifest.get("parent")
            or sidecar_value.get("qualification") != manifest.get("qualification")
            or sidecar_value.get("exclusions") != manifest.get("exclusions")
            or sidecar_value.get("external_preregistration")
            != manifest.get("external_preregistration")
            or sidecar_value.get("storage_mount") != manifest.get("storage_mount")
            or not isinstance(progress, Mapping)
            or int(progress.get("child_trained_actions", -1)) != expected_child_actions
        ):
            raise U2rAnchorError("U2r carried-exam sidecar/integrity binding changed")


def _verify_resume_segment(
    directory: Path,
    *,
    index: int,
    commit: str,
) -> dict[str, Any]:
    manifest = _read_safe_json(
        directory / "manifest.json",
        f"U2r segment {index} manifest",
    )
    status = _read_safe_json(
        directory / "status.json",
        f"U2r segment {index} status",
    )
    manifest_source = manifest.get("source")
    status_source = status.get("source")
    manifest_segment = manifest.get("segment")
    if status.get("phase") not in {"interrupted", "crashed"}:
        raise U2rAnchorError("U2r resume requires an interrupted/crashed segment tip")
    if (
        manifest.get("schema_version") != v02_u2r.CHECKPOINT_SCHEMA_VERSION
        or manifest.get("protocol") != v02_u2r.PROTOCOL
        or not isinstance(manifest_source, Mapping)
        or manifest_source.get("dirty") is not False
        or manifest_source.get("commit") != commit
        or not isinstance(status_source, Mapping)
        or status_source != manifest_source
        or status.get("protocol") != v02_u2r.PROTOCOL
        or not isinstance(manifest_segment, Mapping)
        or int(manifest_segment.get("index", -1)) != index
        or int(manifest_segment.get("lineage_source_child_seed", -1)) != v02_u2r.SOURCE_CHILD_SEED
        or int(manifest_segment.get("segment_algorithm_seed", -1))
        != (v02_u2r.REMEDIATION_ALGORITHM_SEED + index * v02_u2r.SEGMENT_SEED_OFFSET)
        or manifest_segment.get("segment_worker_streams")
        != [
            seed + index * v02_u2r.SEGMENT_SEED_OFFSET
            for seed in v02_u2r.REMEDIATION_WORKER_STREAMS
        ]
    ):
        raise U2rAnchorError("U2r segment manifest/status identity changed")
    checkpoint = _safe_relative_artifact(
        directory,
        status.get("latest_safe_checkpoint"),
        label=f"U2r segment {index} latest-safe checkpoint",
    )
    if checkpoint.suffix != ".zip":
        raise U2rAnchorError("U2r latest-safe checkpoint is not a ZIP archive")
    checkpoint_sha256 = _file_sha256(checkpoint)
    if status.get("latest_safe_checkpoint_sha256") != checkpoint_sha256:
        raise U2rAnchorError("U2r status/latest-safe checkpoint digest changed")
    sidecar_path = checkpoint.with_suffix(".json")
    integrity_path = checkpoint.with_suffix(".integrity.json")
    sidecar = _read_safe_json(sidecar_path, "U2r latest-safe sidecar")
    integrity = _read_safe_json(integrity_path, "U2r latest-safe integrity")
    progress = sidecar.get("progress")
    segment = sidecar.get("segment")
    controller = sidecar.get("controller")
    curriculum = sidecar.get("curriculum")
    sidecar_source = sidecar.get("source")
    model_state = sidecar.get("model_state")
    if (
        sidecar.get("schema_version") != v02_u2r.CHECKPOINT_SCHEMA_VERSION
        or sidecar.get("protocol") != v02_u2r.PROTOCOL
        or sidecar.get("resume_eligible") is not True
        or sidecar.get("kind") not in {"initial", "resume", "latest"}
        or sidecar.get("checkpoint_sha256") != checkpoint_sha256
        or sidecar_source != manifest_source
        or sidecar.get("effective_config") != manifest.get("effective_config")
        or sidecar.get("parent") != manifest.get("parent")
        or sidecar.get("qualification") != manifest.get("qualification")
        or sidecar.get("exclusions") != manifest.get("exclusions")
        or sidecar.get("external_preregistration") != manifest.get("external_preregistration")
        or sidecar.get("storage_mount") != manifest.get("storage_mount")
        or segment != manifest_segment
        or not isinstance(progress, Mapping)
        or not isinstance(controller, Mapping)
        or not isinstance(curriculum, Mapping)
        or not isinstance(model_state, Mapping)
        or controller.get("terminal_eligible") is not None
        or curriculum.get("mastered") is not False
    ):
        raise U2rAnchorError("U2r latest-safe sidecar is not resumable evidence")
    normalized_model_state = {
        "policy_tensor_sha256": _require_sha256(
            model_state.get("policy_tensor_sha256"),
            "U2r latest-safe policy",
        ),
        "optimizer_state_sha256": _require_sha256(
            model_state.get("optimizer_state_sha256"),
            "U2r latest-safe optimizer",
        ),
    }
    if dict(model_state) != normalized_model_state:
        raise U2rAnchorError("U2r latest-safe model-state fields changed")
    if (
        integrity.get("schema_version") != v02_u2r.CHECKPOINT_SCHEMA_VERSION
        or integrity.get("protocol") != v02_u2r.PROTOCOL
        or integrity.get("checkpoint") != checkpoint.name
        or integrity.get("checkpoint_sha256") != checkpoint_sha256
        or integrity.get("sidecar") != sidecar_path.name
        or integrity.get("sidecar_sha256") != _file_sha256(sidecar_path)
    ):
        raise U2rAnchorError("U2r latest-safe integrity binding changed")
    try:
        child_actions = int(progress["child_trained_actions"])
        remediation_actions = int(progress["remediation_trained_actions"])
        lifetime_actions = int(progress["lifetime_trained_actions"])
        collected_actions = int(progress["collected_actions"])
        trained_actions = int(progress["trained_actions"])
        remaining_actions = int(progress["remaining_remediation_actions"])
        optimizer_updates = int(progress["optimizer_updates"])
        start_child_actions = int(manifest_segment["start_child_trained_actions"])
        start_remediation_actions = int(manifest_segment["start_remediation_trained_actions"])
        start_lifetime_actions = int(manifest_segment["start_lifetime_trained_actions"])
    except (KeyError, TypeError, ValueError) as error:
        raise U2rAnchorError("U2r latest-safe counters are incomplete") from error
    exam_records = controller.get("exam_records")
    expected_exam_count = remediation_actions // v02_u2r.EVALUATION_INTERVAL
    if (
        not isinstance(exam_records, list)
        or len(exam_records) != expected_exam_count
        or not all(isinstance(record, Mapping) for record in exam_records)
        or [int(record.get("child_trained_actions", -1)) for record in exam_records]
        != [
            v02_u2r.SOURCE_CHILD_ACTIONS + position * v02_u2r.EVALUATION_INTERVAL
            for position in range(1, expected_exam_count + 1)
        ]
    ):
        raise U2rAnchorError("U2r latest-safe controller exam sequence is incomplete")
    for record in exam_records:
        boundary = int(record["child_trained_actions"])
        exam = directory / "checkpoints" / "rolling" / f"exam-{boundary:07d}.zip"
        exam_sidecar = exam.with_suffix(".json")
        exam_integrity = exam.with_suffix(".integrity.json")
        if (
            exam.is_symlink()
            or not exam.is_file()
            or exam_sidecar.is_symlink()
            or not exam_sidecar.is_file()
        ):
            raise U2rAnchorError("U2r controller exam bundle is missing or unsafe")
        integrity_value = _read_safe_json(
            exam_integrity,
            "U2r controller exam integrity",
        )
        if (
            integrity_value.get("schema_version") != v02_u2r.CHECKPOINT_SCHEMA_VERSION
            or integrity_value.get("protocol") != v02_u2r.PROTOCOL
            or integrity_value.get("checkpoint") != exam.name
            or integrity_value.get("checkpoint_sha256") != _file_sha256(exam)
            or integrity_value.get("sidecar") != exam_sidecar.name
            or integrity_value.get("sidecar_sha256") != _file_sha256(exam_sidecar)
        ):
            raise U2rAnchorError("U2r controller exam integrity binding changed")
        _verify_exam_case_binding(
            directory,
            exam,
            exam_sidecar,
            record,
        )
    expected_updates = (
        v02_u2r.SOURCE_OPTIMIZER_UPDATES
        + remediation_actions // v02_u2r.ROLLOUT_TRANSITIONS * v02_u2r.PPO_EPOCHS
    )
    if (
        remediation_actions != child_actions - v02_u2r.SOURCE_CHILD_ACTIONS
        or lifetime_actions != v02_u2r.SOURCE_LIFETIME_ACTIONS + remediation_actions
        or collected_actions != trained_actions
        or trained_actions != lifetime_actions
        or remaining_actions != v02_u2r.ADDITIONAL_ACTION_BUDGET - remediation_actions
        or not 0 <= remediation_actions < v02_u2r.ADDITIONAL_ACTION_BUDGET
        or remediation_actions % v02_u2r.ROLLOUT_TRANSITIONS
        or optimizer_updates != expected_updates
        or not v02_u2r.SOURCE_CHILD_ACTIONS <= start_child_actions <= child_actions
        or not 0 <= start_remediation_actions <= remediation_actions
        or not v02_u2r.SOURCE_LIFETIME_ACTIONS <= start_lifetime_actions <= lifetime_actions
        or int(status.get("child_trained_actions", -1)) != child_actions
        or int(status.get("remediation_trained_actions", -1)) != remediation_actions
        or int(status.get("lifetime_trained_actions", -1)) != lifetime_actions
        or int(status.get("optimizer_updates", -1)) != optimizer_updates
    ):
        raise U2rAnchorError("U2r resume counters or segment identity changed")
    initial_model_state = _verify_segment_initial_bundle(directory, manifest)
    interruption_boundary = _verify_interruption_evidence(
        directory,
        manifest,
        status,
        checkpoint,
        sidecar,
    )
    try:
        interruption_document, interruption_workers = v02_u2r._verify_interruption_for_resume(
            directory,
            checkpoint=checkpoint,
            checkpoint_sidecar=sidecar,
            expected_config=manifest["effective_config"],
            parent=manifest["parent"],
            qualification=manifest["qualification"],
            expected_source_commit=commit,
            segment_index=index,
        )
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        v02_u2r.U2rProtocolError,
    ) as error:
        raise U2rAnchorError("U2r interruption fails full resume semantics") from error
    if interruption_document.get("resume_boundary") != interruption_boundary:
        raise U2rAnchorError("U2r interruption verifiers disagree")
    return {
        "directory": directory,
        "manifest": manifest,
        "status": status,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "child_actions": child_actions,
        "remediation_actions": remediation_actions,
        "lifetime_actions": lifetime_actions,
        "optimizer_updates": optimizer_updates,
        "model_state": normalized_model_state,
        "initial_model_state": dict(initial_model_state),
        "interruption_boundary": dict(interruption_boundary),
        "interruption_workers": tuple(dict(worker) for worker in interruption_workers),
        "interruption_path": str((directory / "interruption.json").resolve()),
        "interruption_sha256": _file_sha256(directory / "interruption.json"),
        "interruption_active_workers_sha256": _require_sha256(
            interruption_document.get("active_workers_sha256"),
            "U2r interruption active workers",
        ),
        "controller_records": tuple(dict(record) for record in exam_records),
        "start_child_actions": start_child_actions,
        "start_remediation_actions": start_remediation_actions,
        "start_lifetime_actions": start_lifetime_actions,
    }


def select_resume_plan(
    run_root: Path,
    *,
    expected_source_commit: str,
) -> U2rResumePlan:
    """Authenticate one append-only segment chain and select only its tip."""

    commit = _require_commit(expected_source_commit)
    root = _regular_absolute_directory(run_root, "U2r run root")
    indexed = _discover_resume_chain(root)
    verified = [
        _verify_resume_segment(indexed[index], index=index, commit=commit)
        for index in range(len(indexed))
    ]
    initial = verified[0]
    initial_segment = initial["manifest"]["segment"]
    if (
        initial_segment.get("resume_checkpoint") is not None
        or initial_segment.get("resume_checkpoint_sha256") is not None
        or initial_segment.get("resume_model_state") is not None
        or initial_segment.get("resume_source_segment_index") is not None
        or initial_segment.get("inherited_active_workers") != []
        or initial_segment.get("inherited_active_workers_abandoned") is not False
        or initial["start_child_actions"] != v02_u2r.SOURCE_CHILD_ACTIONS
        or initial["start_remediation_actions"] != 0
        or initial["start_lifetime_actions"] != v02_u2r.SOURCE_LIFETIME_ACTIONS
        or initial["initial_model_state"]
        != {
            "policy_tensor_sha256": v02_u2r.SOURCE_POLICY_TENSOR_SHA256,
            "optimizer_state_sha256": v02_u2r.SOURCE_OPTIMIZER_STATE_SHA256,
        }
    ):
        raise U2rAnchorError("U2r initial segment does not begin at the frozen parent")
    _verify_carried_exams(
        initial["directory"],
        initial["manifest"],
        expected_count=0,
        controller_records=initial["controller_records"],
    )
    frozen_identity = {
        key: initial["manifest"].get(key)
        for key in (
            "effective_config",
            "parent",
            "qualification",
            "exclusions",
            "external_preregistration",
            "storage_mount",
        )
    }
    for index in range(1, len(verified)):
        prior = verified[index - 1]
        current = verified[index]
        manifest = current["manifest"]
        segment = manifest["segment"]
        prior_boundary = prior["interruption_boundary"]
        if any(manifest.get(key) != value for key, value in frozen_identity.items()):
            raise U2rAnchorError("U2r resume segment changed frozen provenance")
        if (
            segment.get("resume_checkpoint") != prior_boundary.get("checkpoint")
            or segment.get("resume_checkpoint_sha256") != prior_boundary.get("checkpoint_sha256")
            or segment.get("resume_model_state") != prior_boundary.get("model_state")
            or segment.get("resume_source_segment_index") != index - 1
            or current["initial_model_state"] != prior_boundary.get("model_state")
            or current["start_child_actions"] != prior["child_actions"]
            or current["start_remediation_actions"] != prior["remediation_actions"]
            or current["start_lifetime_actions"] != prior["lifetime_actions"]
            or current["child_actions"] < prior["child_actions"]
            or current["optimizer_updates"] < prior["optimizer_updates"]
        ):
            raise U2rAnchorError("U2r resume link is not continuous")
        try:
            v02_u2r._verify_resumed_worker_abandonment(
                segment=segment,
                ledger_path=current["directory"] / "episode-starts.jsonl",
                source_workers=prior["interruption_workers"],
                source_interruption=Path(prior["interruption_path"]),
                source_interruption_sha256=str(prior["interruption_sha256"]),
                source_active_workers_sha256=str(prior["interruption_active_workers_sha256"]),
            )
        except (OSError, TypeError, ValueError, v02_u2r.U2rProtocolError) as error:
            raise U2rAnchorError(
                "U2r successor segment changed inherited-worker abandonment"
            ) from error
        _verify_carried_exams(
            current["directory"],
            manifest,
            expected_count=(prior["remediation_actions"] // v02_u2r.EVALUATION_INTERVAL),
            controller_records=current["controller_records"],
        )
    tip = verified[-1]
    highest = len(verified) - 1
    next_index = highest + 1
    next_name = f"{v02_u2r.DEFAULT_RUN_NAME}-resume-{next_index}"
    next_path = root / next_name
    if next_path.exists() or next_path.is_symlink():
        raise U2rAnchorError("next U2r resume segment target already exists")
    return U2rResumePlan(
        source_directory=str(tip["directory"]),
        checkpoint=str(tip["checkpoint"]),
        checkpoint_sha256=str(tip["checkpoint_sha256"]),
        source_segment_index=highest,
        next_segment_index=next_index,
        next_run_name=next_name,
        child_trained_actions=int(tip["child_actions"]),
        remediation_trained_actions=int(tip["remediation_actions"]),
        lifetime_trained_actions=int(tip["lifetime_actions"]),
        optimizer_updates=int(tip["optimizer_updates"]),
    )


def _static_exclusion_fields(
    evidence: Any,
) -> tuple[str, int, int, str]:
    if hasattr(evidence, "public_dict"):
        public = evidence.public_dict()
    elif isinstance(evidence, Mapping):
        public = evidence
    else:
        raise U2rAnchorError("U2r exclusion evidence has no public mapping")
    if not isinstance(public, Mapping):
        raise U2rAnchorError("U2r exclusion evidence is not a mapping")
    unavailable = _require_positive_int(
        public.get("unavailable_original_terminal_active_worker_layouts_upper_bound"),
        "unavailable original U2 active-layout upper bound",
    )
    unavailable_status = public.get("unavailable_original_terminal_active_worker_layouts_status")
    if (
        unavailable != UNAVAILABLE_ORIGINAL_ACTIVE_LAYOUTS_UPPER_BOUND
        or unavailable_status != UNAVAILABLE_ORIGINAL_ACTIVE_LAYOUTS_STATUS
    ):
        raise U2rAnchorError("original U2 active-layout evidence limitation changed")
    return (
        _require_sha256(
            public.get("exact_layout_set_sha256"),
            "U2r static exclusion set",
        ),
        _require_positive_int(
            public.get("exact_layouts"),
            "U2r static exclusion layout count",
        ),
        unavailable,
        str(unavailable_status),
    )


def build_anchor_payload(
    repository: Path,
    *,
    source_commit: str,
    exclusions: Any,
) -> dict[str, Any]:
    """Build the exact deterministic annotated-tag message."""

    repo = repository.expanduser().resolve()
    commit = _require_commit(source_commit)
    (
        exclusion_sha256,
        exclusion_layouts,
        unavailable_active_layouts,
        unavailable_active_layouts_status,
    ) = _static_exclusion_fields(exclusions)
    protocol_sha256 = protocol_document_sha256(repo)
    parent_sidecar = v02_u2r.SOURCE_ARCHIVE.with_suffix(".json")
    parent_integrity = v02_u2r.SOURCE_ARCHIVE.with_suffix(".integrity.json")
    parent_manifest = v02_u2r.SOURCE_ARCHIVE.parent.parent / "manifest.json"
    return {
        "schema_version": ANCHOR_SCHEMA_VERSION,
        "anchor_protocol": ANCHOR_PROTOCOL,
        "training_protocol": v02_u2r.PROTOCOL,
        "tag": ANCHOR_TAG,
        "remote": ANCHOR_REMOTE,
        "remote_url": EXPECTED_ORIGIN_URL,
        "source_commit": commit,
        "protocol_document": PROTOCOL_REPOSITORY_PATH,
        "protocol_document_sha256": protocol_sha256,
        "parent": {
            "checkpoint": str(v02_u2r.SOURCE_ARCHIVE),
            "checkpoint_sha256": v02_u2r.SOURCE_ARCHIVE_SHA256,
            "sidecar": str(parent_sidecar),
            "sidecar_sha256": v02_u2r.SOURCE_SIDECAR_SHA256,
            "integrity": str(parent_integrity),
            "integrity_sha256": v02_u2r.SOURCE_INTEGRITY_SHA256,
            "manifest": str(parent_manifest),
            "manifest_sha256": v02_u2r.SOURCE_MANIFEST_SHA256,
            "training_source_commit": v02_u2r.SOURCE_TRAINING_COMMIT,
            "policy_member_sha256": PARENT_POLICY_MEMBER_SHA256,
            "policy_tensor_sha256": v02_u2r.SOURCE_POLICY_TENSOR_SHA256,
            "optimizer_state_sha256": PARENT_OPTIMIZER_STATE_SHA256,
        },
        "failed_confirmation": {
            "report": str(v02_u2r.U2_CONFIRMATION_REPORT),
            "report_sha256": v02_u2r.U2_CONFIRMATION_REPORT_SHA256,
            "attempt": str(v02_u2r.U2_CONFIRMATION_ATTEMPT),
            "attempt_sha256": v02_u2r.U2_CONFIRMATION_ATTEMPT_SHA256,
            "source_commit": v02_u2r.U2_CONFIRMATION_SOURCE_COMMIT,
            "verdict": "capability_failed",
            "u2_score": 169,
            "u2_panels": [84, 85],
        },
        "static_exclusions": {
            "exact_layout_set_sha256": exclusion_sha256,
            "exact_layouts": exclusion_layouts,
            "unavailable_original_terminal_active_worker_layouts_upper_bound": (
                unavailable_active_layouts
            ),
            "unavailable_original_terminal_active_worker_layouts_status": (
                unavailable_active_layouts_status
            ),
        },
        "budget": {
            "source_child_actions": v02_u2r.SOURCE_CHILD_ACTIONS,
            "additional_action_budget": v02_u2r.ADDITIONAL_ACTION_BUDGET,
            "terminal_child_actions": v02_u2r.TERMINAL_CHILD_ACTIONS,
            "source_lifetime_actions": v02_u2r.SOURCE_LIFETIME_ACTIONS,
            "terminal_lifetime_actions": v02_u2r.TERMINAL_LIFETIME_ACTIONS,
            "source_optimizer_updates": v02_u2r.SOURCE_OPTIMIZER_UPDATES,
            "terminal_optimizer_updates": TERMINAL_OPTIMIZER_UPDATES,
            "evaluation_interval": v02_u2r.EVALUATION_INTERVAL,
            "complete_windows": v02_u2r.KEEP_ALL_EXAMS,
        },
        "seeds": {
            "algorithm_seed": v02_u2r.REMEDIATION_ALGORITHM_SEED,
            "worker_streams": list(v02_u2r.REMEDIATION_WORKER_STREAMS),
            "training_allocation": [0, 999_999],
            "segment_seed_offset": v02_u2r.SEGMENT_SEED_OFFSET,
            "initial_run_name": v02_u2r.DEFAULT_RUN_NAME,
            "resume_run_name_pattern": f"{v02_u2r.DEFAULT_RUN_NAME}-resume-N",
            "segment_algorithm_seed_formula": (
                "algorithm_seed + segment_index * segment_seed_offset"
            ),
            "segment_worker_stream_formula": (
                "initial_worker_stream + segment_index * segment_seed_offset"
            ),
        },
        "stability_rule": json.loads(json.dumps(STABILITY_RULE)),
        "confirmation": {
            "consumed_ranges": {
                key: list(bounds) for key, bounds in CONSUMED_CONFIRMATION_RANGES.items()
            },
            "fresh_reserved_ranges": {
                key: list(bounds) for key, bounds in FRESH_CONFIRMATION_RANGES.items()
            },
            "available_during_training": False,
            "requires_separate_post_training_preregistration": True,
        },
    }


def anchor_message(
    repository: Path,
    *,
    source_commit: str,
    exclusions: Any,
) -> str:
    """Return the canonical one-line annotated-tag message."""

    return json.dumps(
        build_anchor_payload(
            repository,
            source_commit=source_commit,
            exclusions=exclusions,
        ),
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_payload(raw: str) -> Mapping[str, Any]:
    if "\n" in raw:
        raise U2rAnchorError("U2r anchor message must be one JSON line")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise U2rAnchorError("U2r anchor message is not JSON") from error
    if not isinstance(value, Mapping) or set(value) != ANCHOR_FIELDS:
        raise U2rAnchorError("U2r anchor fields are incomplete or unknown")
    return value


def verify_external_anchor(
    repository: Path,
    *,
    expected_source_commit: str,
    expected_protocol_sha256: str,
    expected_exclusion_set_sha256: str,
    expected_exclusion_layouts: int,
    runner: Runner = subprocess.run,
) -> U2rAnchor:
    """Verify one exact annotated tag object locally and on frozen origin."""

    repo = repository.expanduser().resolve()
    protocol_sha256 = _require_sha256(
        expected_protocol_sha256,
        "expected U2r protocol document",
    )
    exclusion_sha256 = _require_sha256(
        expected_exclusion_set_sha256,
        "expected U2r static exclusion set",
    )
    exclusion_layouts = _require_positive_int(
        expected_exclusion_layouts,
        "expected U2r static exclusion layout count",
    )
    measured_protocol_sha256 = protocol_document_sha256(repo)
    if measured_protocol_sha256 != protocol_sha256:
        raise U2rAnchorError("active U2r protocol document differs from the expected digest")
    expected_payload = build_anchor_payload(
        repo,
        source_commit=expected_source_commit,
        exclusions={
            "exact_layout_set_sha256": exclusion_sha256,
            "exact_layouts": exclusion_layouts,
            "unavailable_original_terminal_active_worker_layouts_upper_bound": (
                UNAVAILABLE_ORIGINAL_ACTIVE_LAYOUTS_UPPER_BOUND
            ),
            "unavailable_original_terminal_active_worker_layouts_status": (
                UNAVAILABLE_ORIGINAL_ACTIVE_LAYOUTS_STATUS
            ),
        },
    )
    head = _clean_head(
        repo,
        expected_source_commit=expected_source_commit,
        runner=runner,
    )
    if _git(repo, ["remote", "get-url", ANCHOR_REMOTE], runner=runner) != EXPECTED_ORIGIN_URL:
        raise U2rAnchorError("U2r anchor remote is not the frozen GitHub origin")
    if (
        _git(
            repo,
            ["cat-file", "-t", f"refs/tags/{ANCHOR_TAG}"],
            runner=runner,
        )
        != "tag"
    ):
        raise U2rAnchorError("U2r preregistration must be an annotated Git tag")
    tag_object = _git(
        repo,
        ["rev-parse", "--verify", f"refs/tags/{ANCHOR_TAG}"],
        runner=runner,
    )
    if _GIT_OBJECT.fullmatch(tag_object) is None:
        raise U2rAnchorError("U2r tag object ID is invalid")
    peeled = _git(
        repo,
        ["rev-list", "-n", "1", f"refs/tags/{ANCHOR_TAG}"],
        runner=runner,
    )
    if peeled != head:
        raise U2rAnchorError("U2r tag does not peel to the active source commit")
    remote_line = _git(
        repo,
        ["ls-remote", "--tags", ANCHOR_REMOTE, f"refs/tags/{ANCHOR_TAG}"],
        runner=runner,
    )
    remote_parts = remote_line.split()
    if (
        len(remote_parts) != 2
        or remote_parts[0] != tag_object
        or remote_parts[1] != f"refs/tags/{ANCHOR_TAG}"
    ):
        raise U2rAnchorError("local U2r tag object is not frozen on origin")
    payload = _parse_payload(
        _git(
            repo,
            [
                "for-each-ref",
                "--format=%(contents)",
                f"refs/tags/{ANCHOR_TAG}",
            ],
            runner=runner,
        )
    )
    if dict(payload) != expected_payload:
        raise U2rAnchorError("U2r tag payload differs from source, protocol, parent, or rules")
    static = expected_payload["static_exclusions"]
    budget = expected_payload["budget"]
    seeds = expected_payload["seeds"]
    confirmation = expected_payload["confirmation"]
    return U2rAnchor(
        tag=ANCHOR_TAG,
        tag_object=tag_object,
        remote=ANCHOR_REMOTE,
        remote_url=EXPECTED_ORIGIN_URL,
        source_commit=head,
        protocol_document=PROTOCOL_REPOSITORY_PATH,
        protocol_document_sha256=str(expected_payload["protocol_document_sha256"]),
        static_exclusion_set_sha256=str(static["exact_layout_set_sha256"]),
        static_exclusion_layouts=int(static["exact_layouts"]),
        unavailable_original_active_layouts_upper_bound=int(
            static["unavailable_original_terminal_active_worker_layouts_upper_bound"]
        ),
        unavailable_original_active_layouts_status=str(
            static["unavailable_original_terminal_active_worker_layouts_status"]
        ),
        parent_checkpoint_sha256=str(expected_payload["parent"]["checkpoint_sha256"]),
        confirmation_report_sha256=str(expected_payload["failed_confirmation"]["report_sha256"]),
        additional_action_budget=int(budget["additional_action_budget"]),
        terminal_child_actions=int(budget["terminal_child_actions"]),
        terminal_lifetime_actions=int(budget["terminal_lifetime_actions"]),
        terminal_optimizer_updates=int(budget["terminal_optimizer_updates"]),
        algorithm_seed=int(seeds["algorithm_seed"]),
        worker_streams=tuple(int(item) for item in seeds["worker_streams"]),
        fresh_confirmation_ranges={
            str(key): [int(bound) for bound in value]
            for key, value in confirmation["fresh_reserved_ranges"].items()
        },
    )


def publish_external_anchor(
    repository: Path,
    *,
    source_commit: str,
    exclusions: Any,
    runner: Runner = subprocess.run,
) -> U2rAnchor:
    """Create or recover the fixed tag, push it, then verify remote identity."""

    repo = repository.expanduser().resolve()
    _clean_head(
        repo,
        expected_source_commit=source_commit,
        runner=runner,
    )
    if _git(repo, ["remote", "get-url", ANCHOR_REMOTE], runner=runner) != EXPECTED_ORIGIN_URL:
        raise U2rAnchorError("U2r anchor remote is not the frozen GitHub origin")
    message = anchor_message(
        repo,
        source_commit=source_commit,
        exclusions=exclusions,
    )
    local = runner(
        ["git", "show-ref", "--verify", "--quiet", f"refs/tags/{ANCHOR_TAG}"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if local.returncode == 0:
        if (
            _git(
                repo,
                ["cat-file", "-t", f"refs/tags/{ANCHOR_TAG}"],
                runner=runner,
            )
            != "tag"
            or _git(
                repo,
                ["rev-list", "-n", "1", f"refs/tags/{ANCHOR_TAG}"],
                runner=runner,
            )
            != source_commit
            or _git(
                repo,
                [
                    "for-each-ref",
                    "--format=%(contents)",
                    f"refs/tags/{ANCHOR_TAG}",
                ],
                runner=runner,
            )
            != message
        ):
            raise U2rAnchorError("existing local U2r tag differs from this preregistration")
    elif local.returncode == 1:
        remote_before = _git(
            repo,
            ["ls-remote", "--tags", ANCHOR_REMOTE, f"refs/tags/{ANCHOR_TAG}"],
            runner=runner,
        )
        if remote_before:
            raise U2rAnchorError("U2r tag exists remotely but not as the verified local object")
        _git(
            repo,
            [
                "tag",
                "--annotate",
                ANCHOR_TAG,
                "--message",
                message,
                source_commit,
            ],
            runner=runner,
        )
    else:
        raise U2rAnchorError("cannot determine whether the local U2r tag already exists")
    remote = _git(
        repo,
        ["ls-remote", "--tags", ANCHOR_REMOTE, f"refs/tags/{ANCHOR_TAG}"],
        runner=runner,
    )
    if not remote:
        _git(
            repo,
            ["push", ANCHOR_REMOTE, f"refs/tags/{ANCHOR_TAG}"],
            runner=runner,
        )
    (
        exclusion_sha256,
        exclusion_layouts,
        _unavailable_active_layouts,
        _unavailable_active_layouts_status,
    ) = _static_exclusion_fields(exclusions)
    return verify_external_anchor(
        repo,
        expected_source_commit=source_commit,
        expected_protocol_sha256=protocol_document_sha256(repo),
        expected_exclusion_set_sha256=exclusion_sha256,
        expected_exclusion_layouts=exclusion_layouts,
        runner=runner,
    )
