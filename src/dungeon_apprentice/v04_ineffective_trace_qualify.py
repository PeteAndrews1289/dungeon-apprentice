"""Fail-closed one-shot qualification for the v0.4 ineffective-trace study.

Qualification is deliberately separate from release and training.  The
annotated source tag preregisters the prospective contract, this command
authenticates that published tag and every live input, and only then does it
write one durable qualification claim.  A claim is never removed after a
failure, so the same scientific identity cannot be retried.

The completed v0.3 r3 cohort is authenticated as immutable rationale only.
The sole policy parent remains the directly authenticated confirmed-U1
archive.  No r3 policy, optimizer, rollout, recurrent, RNG, or environment
state is exposed by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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

from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v02_u2r as frozen_u2r
from dungeon_apprentice import v02_u2s as frozen_u2s
from dungeon_apprentice import v03_action_effect as frozen_v03
from dungeon_apprentice import v03_action_effect_qualify as v03_qualify
from dungeon_apprentice import v04_ineffective_trace as v04
from dungeon_apprentice import v04_ineffective_trace_smoke as smoke
from dungeon_apprentice.action_streak import (
    IMAGE_KEY,
    INEFFECTIVE_TRACE_DIM,
    INEFFECTIVE_TRACE_KEY,
    IneffectiveTraceMode,
)
from dungeon_apprentice.artifacts import (
    file_sha256,
    runtime_snapshot,
    utc_now,
)

PROTOCOL = v04.PROTOCOL
SCHEMA_VERSION = 1
KIND = "sealed_stage_a_preflight_qualification"
VERDICT = "qualified"

QUALIFIED_TAG = "ineffective-trace-architecture-v0.4-stage-a-20260724"
QUALIFIED_REMOTE = "origin"
EXPECTED_ORIGIN_URL = "https://github.com/PeteAndrews1289/dungeon-apprentice.git"
PROTOCOL_DOCUMENT = Path(v04.PROTOCOL_DOCUMENT)
DASHBOARD_PORT = 8792

CANONICAL_QUALIFICATION_DIRECTORY = Path(
    "/Volumes/T7 Developer/DungeonApprentice/qualifications/"
    "v0.4-ineffective-trace-stage-a-20260724"
)
CANONICAL_REPORT = CANONICAL_QUALIFICATION_DIRECTORY / "report.json"
CANONICAL_CHECKSUM = CANONICAL_QUALIFICATION_DIRECTORY / "report.json.sha256"
CANONICAL_CLAIM = CANONICAL_QUALIFICATION_DIRECTORY / "claim.json"
CANONICAL_COHORT_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/"
    "v04-ineffective-trace-stage-a-20260724"
)
CANONICAL_MEDIA_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/"
    "v04-ineffective-trace-stage-a-media-20260724"
)

R3_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/"
    "v03-action-effect-stage-a-r3-20260724"
)
R3_QUALIFICATION_DIRECTORY = Path(
    "/Volumes/T7 Developer/DungeonApprentice/qualifications/"
    "v0.3-action-effect-stage-a-r3-20260724"
)
R3_MEDIA_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/"
    "v03-action-effect-stage-a-r3-media-20260724"
)
R3_SCREEN_LOG = Path(
    "/Volumes/T7 Developer/DungeonApprentice/launch-recovery/"
    "v03-action-effect-stage-a-r3-20260724-screen.log"
)
EVIDENCE_ROOT = Path("/Volumes/T7 Developer/DungeonApprentice")
R3_COHORT_ID = "v0.3-action-effect-stage-a-r3-20260724"
R3_SOURCE_COMMIT = "c4834b73dfed7d875c6f59887d3077319a305910"
R3_TAG = "action-effect-architecture-v0.3-stage-a-r3-20260724"
R3_TAG_OBJECT = "fc5eb2f82896b7f6d41030f2776a0028a70ea4c2"
R3_QUALIFICATION_REPORT_SHA256 = (
    "07151fecea179dfaedabe56dcadc009d3750798987973c17cb67b1888909c499"
)
R3_COHORT_CONTRACT_SHA256 = (
    "fea7b7eb10accec2395105630ad790d550993996c5bd35a2a937fdb8837d5ade"
)
R3_COHORT_STATE_SHA256 = (
    "fb99ab57244d1e1b2a6bc5f9cdfd7f1763038c8186b174587b9e4d95cecbc8db"
)
R3_REPORT_SHA256 = (
    "05f23509652e48a127be3728fb4ccb052b8285035792db9aaec7a956e021dcdb"
)
R3_REPORT_INTEGRITY_SHA256 = (
    "0e42c78e7a1b541d55c4286ccf606f1477a453914427cd74ebe59af8568b81a4"
)
R3_PROCESS_CLOSEOUT_SHA256 = (
    "1e8512fd2a20c3371d8c6207ee9ef6f10955fa1e14a8556d3d46f25ea79d2641"
)
R3_SCREEN_LOG_SHA256 = (
    "31cf55aba551c5b084d658161b51849db077ec225d22b7abb9bbda302eda169f"
)
R3_TREE_SEALS: Mapping[str, Mapping[str, int | str]] = MappingProxyType(
    {
        "cohort": MappingProxyType(
            {
                "regular_file_count": 315,
                "regular_file_bytes": 1_891_170_919,
                "directory_count_including_root": 10,
                "directory_list_sha256": (
                    "821152ab67fbbacf5c3008fa51384190df793d47225cec89eec6f0ae4ef9bc23"
                ),
                "regular_file_map_sha256": (
                    "39eb42ba11f3a56976def56cbdd7493c3202f79c756772feea9375240e6396c8"
                ),
            }
        ),
        "qualification": MappingProxyType(
            {
                "regular_file_count": 3,
                "regular_file_bytes": 96_402,
                "directory_count_including_root": 1,
                "directory_list_sha256": (
                    "19a3f6872c35183258369865a8dbf1c1d9153746792b755d177241c485a2544f"
                ),
                "regular_file_map_sha256": (
                    "83929801d494a2c3a0e1cdeb6b1b88877e9da04e8104546c99e89dc6ce39f6f7"
                ),
            }
        ),
        "media": MappingProxyType(
            {
                "regular_file_count": 0,
                "regular_file_bytes": 0,
                "directory_count_including_root": 3,
                "directory_list_sha256": (
                    "703e20d974855ed845056ebfe7cef8c389be426fa410353d57e5f0de962a5087"
                ),
                "regular_file_map_sha256": (
                    "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
                ),
            }
        ),
    }
)
R3_ARM_HASHES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "sham": MappingProxyType(
            {
                "report": (
                    "f6c363d532a7547ce5cc2b1bfddf939698742c85889a9b05ff7bba3468281202"
                ),
                "report_integrity": (
                    "090e26d3387d7b297c8e6af5080f02db88124de2bc88fa5fda535a9dc52f4bff"
                ),
                "status": (
                    "3fc7e4db2afa3e14616ea555612ffb2207792bf99c00b4fef85301f3d9d321b4"
                ),
                "terminal_checkpoint": (
                    "ffe5bcd5b5813984c2b8c02cdb8579ba85dc7c36168c4a91d6e90306dc284932"
                ),
                "terminal_sidecar": (
                    "f4e2226fca7997846e1d9abe4eb4706f289ed1a0d744ad1e0077a420480a0857"
                ),
                "terminal_integrity": (
                    "d2d3ba9ac90450d3105ffe8dfd428ba0752c05ca8658af212efc2eff01267abf"
                ),
            }
        ),
        "action-effect": MappingProxyType(
            {
                "report": (
                    "40e0cdbb11c15d8387da5f2a4b4576502eb57b9499085c3d8c338805395c1b14"
                ),
                "report_integrity": (
                    "24775cf45693d5583f0ded434173de77dc9da0e7488cbd531d984bd9bf1f027b"
                ),
                "status": (
                    "f1d5d49f9059b149e1dba37dc58951ab1b0c4e51ab77a307398c38d665bd3c94"
                ),
                "terminal_checkpoint": (
                    "930756ee9ced46675933a89d3205da8df9f8edcff549d127e5571f3aa973a3c5"
                ),
                "terminal_sidecar": (
                    "611da93e5cfbc8614131c1a01aeafa968343c3d7ee0b546490cf5f9d9e07e285"
                ),
                "terminal_integrity": (
                    "c363315ae810f193b9b11bf62c05135d1ab0d7f27d5ec8d101c3adac22ee6eed"
                ),
            }
        ),
    }
)

TAG_SCHEMA_VERSION = 1
TAG_KIND = "v04_ineffective_trace_stage_a_source_preregistration"
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
        "r3_terminal_evidence_sha256",
        "guard_mapping_sha256",
        "sampler_preflight_sha256",
        "architecture_contract_sha256",
        "protected_partitions_sha256",
        "fresh_identity_preflight_sha256",
        "runtime_contract",
        "roots",
        "dashboard_port",
        "storage_caps",
        "resume_rule",
    }
)
MAX_REPORT_BYTES = 16 * 1024**2
MAX_JSON_BYTES = 12 * 1024**2
_ARMS = tuple(mode.value for mode in v04.ARM_ORDER)
EXPECTED_U1_PARAMETER_NAMES = (
    "action_net.bias",
    "action_net.weight",
    "features_extractor.cnn.0.bias",
    "features_extractor.cnn.0.weight",
    "features_extractor.cnn.2.bias",
    "features_extractor.cnn.2.weight",
    "features_extractor.cnn.4.bias",
    "features_extractor.cnn.4.weight",
    "features_extractor.linear.0.bias",
    "features_extractor.linear.0.weight",
    "lstm_actor.bias_hh_l0",
    "lstm_actor.bias_ih_l0",
    "lstm_actor.weight_hh_l0",
    "lstm_actor.weight_ih_l0",
    "lstm_critic.bias_hh_l0",
    "lstm_critic.bias_ih_l0",
    "lstm_critic.weight_hh_l0",
    "lstm_critic.weight_ih_l0",
    "value_net.bias",
    "value_net.weight",
)
_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "kind",
        "verdict",
        "claim",
        "source",
        "protocol_document",
        "parent",
        "predecessors",
        "r3_terminal_evidence",
        "guards",
        "sampler_preflight",
        "architecture_contract",
        "protected_partitions",
        "fresh_identity_preflight",
        "smoke_evidence",
        "storage_caps",
        "restrictions",
        "storage_preflight",
        "created_at",
    }
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


class IneffectiveTraceQualificationError(RuntimeError):
    """Raised when any prospective v0.4 qualification gate changes."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)[:-1]).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise IneffectiveTraceQualificationError(f"{label} is not SHA-256")
    return text


def _require_git_object(value: Any, label: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-f]{40,64}", text):
        raise IneffectiveTraceQualificationError(f"{label} is not a Git object")
    return text


def _reject_symlink_chain(path: Path) -> None:
    requested = path.expanduser().absolute()
    current = Path(requested.anchor)
    for part in requested.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise IneffectiveTraceQualificationError(
                f"unsafe symlink in qualification path: {current}"
            )
        if not current.exists():
            break


def _regular_file_bytes(
    path: Path,
    label: str,
    *,
    maximum: int,
) -> bytes:
    _reject_symlink_chain(path)
    try:
        before = path.stat()
    except OSError as error:
        raise IneffectiveTraceQualificationError(
            f"{label} is missing or unsafe"
        ) from error
    if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
        raise IneffectiveTraceQualificationError(f"{label} is missing or unsafe")
    try:
        value = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise IneffectiveTraceQualificationError(
            f"{label} changed while read"
        ) from error
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in fields):
        raise IneffectiveTraceQualificationError(f"{label} changed while read")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            _regular_file_bytes(path, label, maximum=MAX_JSON_BYTES)
        )
    except json.JSONDecodeError as error:
        raise IneffectiveTraceQualificationError(
            f"{label} is invalid JSON"
        ) from error
    if not isinstance(value, dict):
        raise IneffectiveTraceQualificationError(f"{label} is not an object")
    return value


def _git(
    repository: Path,
    arguments: Sequence[str],
    *,
    runner: Runner = subprocess.run,
) -> str:
    result = runner(
        ["git", *arguments],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise IneffectiveTraceQualificationError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _safe_child(root: Path, relative: Any, label: str) -> Path:
    candidate = Path(str(relative))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise IneffectiveTraceQualificationError(f"{label} escapes its evidence root")
    path = (root / candidate).absolute()
    _reject_symlink_chain(path)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise IneffectiveTraceQualificationError(
            f"{label} is missing or unsafe"
        ) from error
    if not resolved.is_file():
        raise IneffectiveTraceQualificationError(
            f"{label} is missing or unsafe"
        )
    return resolved


def _stable_regular_sha256(path: Path, label: str) -> tuple[str, int]:
    _reject_symlink_chain(path)
    try:
        before = path.stat()
    except OSError as error:
        raise IneffectiveTraceQualificationError(
            f"{label} is missing or unsafe"
        ) from error
    if not stat.S_ISREG(before.st_mode):
        raise IneffectiveTraceQualificationError(f"{label} is not a regular file")
    digest = file_sha256(path)
    try:
        after = path.stat()
    except OSError as error:
        raise IneffectiveTraceQualificationError(
            f"{label} changed while hashed"
        ) from error
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in fields):
        raise IneffectiveTraceQualificationError(f"{label} changed while hashed")
    return digest, int(after.st_size)


def _sealed_tree_inventory(
    root: Path,
    label: str,
) -> tuple[dict[str, str], dict[str, int | str]]:
    """Hash every regular file in one immutable tree without following links."""

    _reject_symlink_chain(root)
    if root.is_symlink() or not root.is_dir():
        raise IneffectiveTraceQualificationError(f"{label} is missing or unsafe")
    files: list[str] = []
    directories: list[str] = ["."]
    for current, child_directories, child_files in os.walk(
        root,
        followlinks=False,
    ):
        current_path = Path(current)
        child_directories.sort()
        child_files.sort()
        for name in child_directories:
            path = current_path / name
            if path.is_symlink() or not path.is_dir():
                raise IneffectiveTraceQualificationError(
                    f"{label} contains an unsafe directory"
                )
            directories.append(path.relative_to(root).as_posix())
        for name in child_files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise IneffectiveTraceQualificationError(
                    f"{label} contains an unsafe file"
                )
            files.append(path.relative_to(root).as_posix())
    measured: dict[str, str] = {}
    byte_count = 0
    for relative in sorted(files):
        digest, size = _stable_regular_sha256(
            root / relative,
            f"{label} file {relative}",
        )
        measured[relative] = digest
        byte_count += size
    return measured, {
        "regular_file_count": len(measured),
        "regular_file_bytes": byte_count,
        "directory_count_including_root": len(directories),
        "directory_list_sha256": _canonical_sha256(sorted(directories)),
        "regular_file_map_sha256": _canonical_sha256(measured),
    }


def _verify_exact_file(
    path: Path,
    expected_sha256: str,
    label: str,
) -> None:
    digest, _size = _stable_regular_sha256(path, label)
    if digest != expected_sha256:
        raise IneffectiveTraceQualificationError(f"{label} bytes changed")


def _verify_r3_tag(
    repository: Path,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, str]:
    """Require the immutable r3 annotated tag locally and on origin."""

    ref = f"refs/tags/{R3_TAG}"
    if _git(repository, ["cat-file", "-t", ref], runner=runner) != "tag":
        raise IneffectiveTraceQualificationError("r3 tag is no longer annotated")
    tag_object = _require_git_object(
        _git(repository, ["rev-parse", "--verify", ref], runner=runner),
        "r3 tag object",
    )
    commit = _require_git_object(
        _git(repository, ["rev-list", "-n", "1", ref], runner=runner),
        "r3 tag commit",
    )
    remote_url = _git(
        repository,
        ["remote", "get-url", QUALIFIED_REMOTE],
        runner=runner,
    )
    remote = _git(
        repository,
        ["ls-remote", "--tags", QUALIFIED_REMOTE, ref],
        runner=runner,
    ).split()
    if (
        tag_object != R3_TAG_OBJECT
        or commit != R3_SOURCE_COMMIT
        or remote_url != EXPECTED_ORIGIN_URL
        or remote != [R3_TAG_OBJECT, ref]
    ):
        raise IneffectiveTraceQualificationError(
            "immutable r3 local or remote tag changed"
        )
    return {
        "tag": R3_TAG,
        "tag_object": tag_object,
        "source_commit": commit,
        "remote": QUALIFIED_REMOTE,
        "remote_url": remote_url,
    }


def _verify_r3_arm(
    root: Path,
    arm_name: str,
    expected: Mapping[str, str],
) -> dict[str, Any]:
    arm_root = root / arm_name
    report_path = arm_root / "report.json"
    integrity_path = arm_root / "report.integrity.json"
    status_path = arm_root / "status.json"
    for path, digest, label in (
        (report_path, expected["report"], f"r3 {arm_name} report"),
        (
            integrity_path,
            expected["report_integrity"],
            f"r3 {arm_name} report integrity",
        ),
        (status_path, expected["status"], f"r3 {arm_name} status"),
    ):
        _verify_exact_file(path, digest, label)
    report = _read_json(report_path, f"r3 {arm_name} report")
    integrity = _read_json(
        integrity_path,
        f"r3 {arm_name} report integrity",
    )
    progress = report.get("progress")
    terminal = report.get("terminal_checkpoint")
    case_evidence = report.get("case_evidence")
    exams = report.get("exam_records")
    if (
        report.get("protocol") != frozen_v03.PROTOCOL
        or report.get("cohort_id") != R3_COHORT_ID
        or report.get("arm") != arm_name
        or report.get("verdict") != "arm_failed"
        or not isinstance(progress, Mapping)
        or progress.get("child_trained_actions") != v04.CHILD_ACTION_BUDGET
        or progress.get("lifetime_trained_actions")
        != v04.PARENT_LIFETIME_ACTIONS + v04.CHILD_ACTION_BUDGET
        or progress.get("optimizer_updates")
        != v04.PARENT_OPTIMIZER_UPDATES
        + (v04.CHILD_ACTION_BUDGET // v04.ROLLOUT_TRANSITIONS) * v04.PPO_EPOCHS
        or progress.get("exam_count") != v04.EXAM_COUNT
        or report.get("child_trained_actions") != v04.CHILD_ACTION_BUDGET
        or report.get("lifetime_trained_actions")
        != v04.PARENT_LIFETIME_ACTIONS + v04.CHILD_ACTION_BUDGET
        or report.get("optimizer_updates")
        != v04.PARENT_OPTIMIZER_UPDATES
        + (v04.CHILD_ACTION_BUDGET // v04.ROLLOUT_TRANSITIONS) * v04.PPO_EPOCHS
        or report.get("exam_count") != v04.EXAM_COUNT
        or report.get("case_count") != v04.EXAM_COUNT * 320
        or report.get("architecture_definition_eligible") is not False
        or report.get("promotable") is not False
        or report.get("development_checkpoint_reuse_authorized") is not False
        or report.get("successor_checkpoint_authorized") is not False
        or report.get("resume_authorized") is not False
        or report.get("u3_authorized") is not False
        or not isinstance(exams, list)
        or len(exams) != v04.EXAM_COUNT
        or not isinstance(case_evidence, Mapping)
        or case_evidence.get("file_count") != v04.EXAM_COUNT
        or case_evidence.get("record_count") != v04.EXAM_COUNT * 320
        or not isinstance(case_evidence.get("files"), list)
        or len(case_evidence["files"]) != v04.EXAM_COUNT
        or not isinstance(terminal, Mapping)
        or terminal.get("promotable") is not False
    ):
        raise IneffectiveTraceQualificationError(
            f"r3 {arm_name} terminal contract changed"
        )
    expected_boundaries = [
        index * v04.EVALUATION_INTERVAL
        for index in range(1, v04.EXAM_COUNT + 1)
    ]
    if [
        int(value.get("child_trained_actions", -1))
        for value in exams
        if isinstance(value, Mapping)
    ] != expected_boundaries:
        raise IneffectiveTraceQualificationError(
            f"r3 {arm_name} exam boundaries changed"
        )
    listed_cases = {
        str(value.get("path")): value
        for value in case_evidence["files"]
        if isinstance(value, Mapping)
    }
    if len(listed_cases) != v04.EXAM_COUNT:
        raise IneffectiveTraceQualificationError(
            f"r3 {arm_name} case inventory changed"
        )
    for exam in exams:
        checkpoint = _safe_child(
            arm_root,
            exam.get("checkpoint"),
            f"r3 {arm_name} exam checkpoint",
        )
        if file_sha256(checkpoint) != exam.get("checkpoint_sha256"):
            raise IneffectiveTraceQualificationError(
                f"r3 {arm_name} exam checkpoint digest changed"
            )
        sidecar = checkpoint.with_suffix(".json")
        _verify_exact_file(
            sidecar,
            _require_sha256(
                exam.get("sidecar_sha256"),
                f"r3 {arm_name} exam sidecar digest",
            ),
            f"r3 {arm_name} exam sidecar",
        )
        diagnostic = exam.get("case_diagnostics")
        if not isinstance(diagnostic, Mapping):
            raise IneffectiveTraceQualificationError(
                f"r3 {arm_name} exam case binding changed"
            )
        relative = str(diagnostic.get("path"))
        listed = listed_cases.get(relative)
        case_path = _safe_child(
            arm_root,
            relative,
            f"r3 {arm_name} exam cases",
        )
        if (
            not isinstance(listed, Mapping)
            or listed.get("records") != 320
            or diagnostic.get("case_count") != 320
            or listed.get("sha256") != diagnostic.get("sha256")
            or file_sha256(case_path) != diagnostic.get("sha256")
        ):
            raise IneffectiveTraceQualificationError(
                f"r3 {arm_name} exam case checksum changed"
            )
    terminal_path = _safe_child(
        arm_root,
        terminal.get("path"),
        f"r3 {arm_name} terminal checkpoint",
    )
    terminal_sidecar = arm_root / "checkpoints" / "terminal.json"
    terminal_integrity = arm_root / "checkpoints" / "terminal.integrity.json"
    if (
        terminal.get("sha256") != expected["terminal_checkpoint"]
        or terminal.get("sidecar_sha256") != expected["terminal_sidecar"]
        or terminal.get("integrity_sha256") != expected["terminal_integrity"]
        or file_sha256(terminal_path) != expected["terminal_checkpoint"]
        or file_sha256(terminal_sidecar) != expected["terminal_sidecar"]
        or file_sha256(terminal_integrity) != expected["terminal_integrity"]
        or integrity.get("report_sha256") != expected["report"]
        or integrity.get("terminal_checkpoint_sha256")
        != expected["terminal_checkpoint"]
        or integrity.get("terminal_sidecar_sha256") != expected["terminal_sidecar"]
        or integrity.get("terminal_integrity_sha256")
        != expected["terminal_integrity"]
    ):
        raise IneffectiveTraceQualificationError(
            f"r3 {arm_name} terminal checkpoint binding changed"
        )
    return {
        "arm": arm_name,
        "verdict": "arm_failed",
        "eligible": False,
        "child_trained_actions": v04.CHILD_ACTION_BUDGET,
        "lifetime_trained_actions": (
            v04.PARENT_LIFETIME_ACTIONS + v04.CHILD_ACTION_BUDGET
        ),
        "optimizer_updates": (
            v04.PARENT_OPTIMIZER_UPDATES
            + (v04.CHILD_ACTION_BUDGET // v04.ROLLOUT_TRANSITIONS)
            * v04.PPO_EPOCHS
        ),
        "exam_count": v04.EXAM_COUNT,
        "case_count": v04.EXAM_COUNT * 320,
        "report_sha256": expected["report"],
        "report_integrity_sha256": expected["report_integrity"],
        "status_sha256": expected["status"],
        "terminal_checkpoint_sha256": expected["terminal_checkpoint"],
        "terminal_sidecar_sha256": expected["terminal_sidecar"],
        "terminal_integrity_sha256": expected["terminal_integrity"],
        "promotable": False,
    }


def authenticate_r3_terminal(
    repository: Path,
    *,
    tag_runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Fully authenticate r3 as immutable negative rationale, never a parent."""

    tag = _verify_r3_tag(repository, runner=tag_runner)
    measured: dict[str, dict[str, int | str]] = {}
    for label, root in (
        ("cohort", R3_ROOT),
        ("qualification", R3_QUALIFICATION_DIRECTORY),
        ("media", R3_MEDIA_ROOT),
    ):
        _mapping, seal = _sealed_tree_inventory(root, f"r3 {label}")
        if seal != dict(R3_TREE_SEALS[label]):
            raise IneffectiveTraceQualificationError(
                f"immutable r3 {label} tree changed"
            )
        measured[label] = seal
    _verify_exact_file(
        R3_SCREEN_LOG,
        R3_SCREEN_LOG_SHA256,
        "r3 screen log",
    )
    critical = {
        "qualification_report": (
            R3_QUALIFICATION_DIRECTORY / "report.json",
            R3_QUALIFICATION_REPORT_SHA256,
        ),
        "cohort_contract": (
            R3_ROOT / "cohort-contract.json",
            R3_COHORT_CONTRACT_SHA256,
        ),
        "cohort_state": (R3_ROOT / "cohort.json", R3_COHORT_STATE_SHA256),
        "report": (R3_ROOT / "report.json", R3_REPORT_SHA256),
        "report_integrity": (
            R3_ROOT / "report.integrity.json",
            R3_REPORT_INTEGRITY_SHA256,
        ),
        "process_closeout": (
            R3_ROOT / "process-closeout.json",
            R3_PROCESS_CLOSEOUT_SHA256,
        ),
    }
    for label, (path, digest) in critical.items():
        _verify_exact_file(path, digest, f"r3 {label}")
    cohort = _read_json(R3_ROOT / "cohort.json", "r3 cohort state")
    report = _read_json(R3_ROOT / "report.json", "r3 cohort report")
    integrity = _read_json(
        R3_ROOT / "report.integrity.json",
        "r3 report integrity",
    )
    closeout = _read_json(
        R3_ROOT / "process-closeout.json",
        "r3 process closeout",
    )
    selection = report.get("selection")
    checkpoint_rule = report.get("checkpoint_rule")
    cohort_arms = cohort.get("arms")
    if (
        cohort.get("protocol") != frozen_v03.PROTOCOL
        or cohort.get("cohort_id") != R3_COHORT_ID
        or cohort.get("phase") != "completed"
        or cohort.get("contract_sha256") != R3_COHORT_CONTRACT_SHA256
        or cohort.get("qualification_sha256")
        != R3_QUALIFICATION_REPORT_SHA256
        or cohort.get("source_commit") != R3_SOURCE_COMMIT
        or cohort.get("tag") != R3_TAG
        or cohort.get("tag_object") != R3_TAG_OBJECT
        or not isinstance(cohort_arms, list)
        or [value.get("id") for value in cohort_arms] != ["sham", "action-effect"]
        or any(value.get("state") != "completed" for value in cohort_arms)
        or report.get("protocol") != frozen_v03.PROTOCOL
        or report.get("cohort_id") != R3_COHORT_ID
        or report.get("verdict") != "architecture_failed"
        or report.get("selected_architecture") is not None
        or report.get("source")
        != {"commit": R3_SOURCE_COMMIT, "dirty": False}
        or report.get("cohort_contract_sha256") != R3_COHORT_CONTRACT_SHA256
        or not isinstance(selection, Mapping)
        or selection.get("verdict") != "architecture_failed"
        or selection.get("selected_architecture") is not None
        or selection.get("replication_protocol_authorized") is not False
        or selection.get("u3_authorized") is not False
        or not isinstance(checkpoint_rule, Mapping)
        or checkpoint_rule.get("stage_a_checkpoint_reuse_authorized") is not False
        or checkpoint_rule.get("replication_protocol_authorized") is not False
        or checkpoint_rule.get("successor_checkpoint") is not None
        or checkpoint_rule.get("u3_authorized") is not False
        or integrity.get("report") != "report.json"
        or integrity.get("report_sha256") != R3_REPORT_SHA256
        or integrity.get("cohort_contract_sha256")
        != R3_COHORT_CONTRACT_SHA256
        or integrity.get("process_closeout_sha256")
        != R3_PROCESS_CLOSEOUT_SHA256
        or closeout.get("protocol") != frozen_v03.PROTOCOL
        or closeout.get("cohort_id") != R3_COHORT_ID
        or closeout.get("verdict") != "clear"
        or closeout.get("prohibited_matches") != []
    ):
        raise IneffectiveTraceQualificationError(
            "immutable r3 terminal verdict or closeout changed"
        )
    arms = {
        name: _verify_r3_arm(R3_ROOT, name, R3_ARM_HASHES[name])
        for name in ("sham", "action-effect")
    }
    arm_bindings = report.get("arm_evidence")
    integrity_bindings = integrity.get("arm_report_sha256")
    if (
        not isinstance(arm_bindings, Mapping)
        or not isinstance(integrity_bindings, Mapping)
        or set(arm_bindings) != set(arms)
        or set(integrity_bindings) != set(arms)
    ):
        raise IneffectiveTraceQualificationError("r3 arm inventory changed")
    for name, evidence in arms.items():
        binding = arm_bindings[name]
        if (
            not isinstance(binding, Mapping)
            or binding.get("report_sha256") != evidence["report_sha256"]
            or binding.get("report_integrity_sha256")
            != evidence["report_integrity_sha256"]
            or binding.get("status_sha256") != evidence["status_sha256"]
            or binding.get("child_trained_actions") != v04.CHILD_ACTION_BUDGET
            or binding.get("exam_count") != v04.EXAM_COUNT
            or binding.get("case_count") != v04.EXAM_COUNT * 320
            or binding.get("eligible") is not False
            or integrity_bindings[name] != evidence["report_sha256"]
        ):
            raise IneffectiveTraceQualificationError(
                f"r3 {name} top-level binding changed"
            )
    return {
        "rationale_only": True,
        "policy_parent_authorized": False,
        "root": str(R3_ROOT),
        "qualification_root": str(R3_QUALIFICATION_DIRECTORY),
        "media_root": str(R3_MEDIA_ROOT),
        "screen_log": str(R3_SCREEN_LOG),
        **tag,
        "qualification_report_sha256": R3_QUALIFICATION_REPORT_SHA256,
        "cohort_contract_sha256": R3_COHORT_CONTRACT_SHA256,
        "cohort_state_sha256": R3_COHORT_STATE_SHA256,
        "report_sha256": R3_REPORT_SHA256,
        "report_integrity_sha256": R3_REPORT_INTEGRITY_SHA256,
        "process_closeout_sha256": R3_PROCESS_CLOSEOUT_SHA256,
        "screen_log_sha256": R3_SCREEN_LOG_SHA256,
        "tree_seals": measured,
        "arms": arms,
        "verdict": "architecture_failed",
        "selected_architecture": None,
        "checkpoint_reuse_authorized": False,
        "replication_protocol_authorized": False,
        "u3_authorized": False,
        "process_closeout_clear": True,
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
    """Reuse the audited pre-v0.3 history loader, then recheck v0.4 values."""

    try:
        parent, base, mapping, guards, predecessors = (
            v03_qualify._authenticate_parent_and_guards(repository)
        )
    except Exception as error:
        raise IneffectiveTraceQualificationError(
            "confirmed-U1 parent or static history guard failed authentication"
        ) from error
    if (
        parent.get("checkpoint_sha256") != v04.PARENT_CHECKPOINT_SHA256
        or parent.get("sidecar_sha256") != v03_qualify.PARENT_SIDECAR_SHA256
        or parent.get("manifest_sha256") != v03_qualify.PARENT_MANIFEST_SHA256
        or parent.get("policy_tensor_sha256")
        != v04.PARENT_POLICY_TENSOR_SHA256
        or parent.get("optimizer_state_sha256")
        != v04.PARENT_OPTIMIZER_STATE_SHA256
        or parent.get("u1_child_seed") != v04.PARENT_U1_SEED
        or parent.get("trained_timesteps") != v04.PARENT_LIFETIME_ACTIONS
        or parent.get("n_updates") != v04.PARENT_OPTIMIZER_UPDATES
        or parent.get("confirmation_verdict") != "confirmed"
        or set(mapping) != set(lessons.LessonId)
        or guards.get("applied_mapping_sha256")
        != _canonical_sha256(
            {
                lesson.value: sorted(mapping[lesson])
                for lesson in lessons.LessonId
            }
        )
    ):
        raise IneffectiveTraceQualificationError(
            "confirmed-U1 identity or frozen guard changed"
        )
    public_predecessors = json.loads(json.dumps(predecessors))
    public_predecessors.update(
        {
            "v03_r3_is_rationale_only": True,
            "v03_policy_or_optimizer_parent_authorized": False,
            "v04_parent_protocol": "dungeon-apprentice-v0.2-u1",
            "policy_updates_during_authentication": False,
        }
    )
    return parent, base, mapping, guards, public_predecessors


def _sampler_preflight(
    mapping: Mapping[lessons.LessonId, frozenset[str]],
    *,
    seed_access: Any,
) -> list[dict[str, Any]]:
    try:
        records = frozen_u2r.preflight_u2r_training_layout_sampler(
            mapping,
            seed_access=seed_access,
            worker_streams=v04.WORKER_STREAMS,
            max_attempts=frozen_u2s.LAYOUT_RESAMPLE_ATTEMPTS,
        )
    except Exception as error:
        raise IneffectiveTraceQualificationError(
            "v0.4 fresh sampler preflight failed"
        ) from error
    public = [dict(record) for record in records]
    if (
        len(public) != len(lessons.LessonId) * v04.WORKERS
        or {
            int(record.get("worker_stream", -1))
            for record in public
        }
        != set(v04.WORKER_STREAMS)
    ):
        raise IneffectiveTraceQualificationError(
            "v0.4 sampler preflight is incomplete"
        )
    return public


def _protected_seed_partitions() -> list[dict[str, Any]]:
    try:
        protected = v03_qualify._protected_seed_partitions()
    except Exception as error:
        raise IneffectiveTraceQualificationError(
            "protected seed inventory changed"
        ) from error
    if len(protected) != 10:
        raise IneffectiveTraceQualificationError(
            "protected seed inventory changed"
        )
    return json.loads(json.dumps(protected))


def _architecture_contract() -> dict[str, Any]:
    sham = v04.effective_config(IneffectiveTraceMode.ZERO_TRACE)
    candidate = v04.effective_config(IneffectiveTraceMode.STREAK_TRACE)
    for config in (sham, candidate):
        observation = config.get("observation")
        trace = (
            observation.get("ineffective_trace")
            if isinstance(observation, Mapping)
            else None
        )
        policy = config.get("policy")
        if (
            config.get("parent", {}).get("checkpoint_sha256")
            != v04.PARENT_CHECKPOINT_SHA256
            or config.get("parent", {}).get("policy_tensor_sha256")
            != v04.PARENT_POLICY_TENSOR_SHA256
            or config.get("parent", {}).get("optimizer_state_sha256")
            != v04.PARENT_OPTIMIZER_STATE_SHA256
            or config.get("optimization", {}).get(
                "changed_from_original_u2_control"
            )
            is not False
            or config.get("reward", {}).get("changed_from_original_u2_control")
            is not False
            or not isinstance(observation, Mapping)
            or observation.get("retains_v03_nine_dimensional_action_effect")
            is not False
            or not isinstance(trace, Mapping)
            or trace.get("shape") != [1]
            or trace.get("clip_streak_at") != 9
            or trace.get("normalization_divisor") != 9.0
            or not isinstance(policy, Mapping)
            or policy.get("ineffective_trace_features") != 1
            or policy.get("trace_encoder_parameter")
            != v04.TRACE_ENCODER_PARAMETER
            or config.get("selection", {}).get(
                "development_checkpoint_reuse_authorized"
            )
            is not False
        ):
            raise IneffectiveTraceQualificationError(
                "v0.4 effective architecture contract changed"
            )
    return {
        "architecture_version": v04.ARCHITECTURE_VERSION,
        "arms": list(_ARMS),
        "same_dict_observation_space": True,
        "same_parameter_set_and_initialization": True,
        "image": {
            "key": IMAGE_KEY,
            "shape_hwc": [56, 56, 3],
            "dtype": "uint8",
        },
        "ineffective_trace": {
            "key": INEFFECTIVE_TRACE_KEY,
            "shape": [INEFFECTIVE_TRACE_DIM],
            "dtype": "float32",
            "formula": "min(count,9)/9",
            "cap": 9,
            "normalization_divisor": 9.0,
            "episode_start": 0,
            "visible_change": 0,
            "first_unchanged_transition": 1,
            "unchanged_action_switch": 1,
            "same_action_unchanged": "min(previous_count+1,9)",
            "sham": "always_zero",
            "candidate": "truthful_bounded_trace",
            "action_identity_exposed": False,
            "v03_nine_dimensional_input_retained": False,
            "reward_or_info_read": False,
        },
        "policy": v04.public_policy_contract(),
        "transplant": {
            "legacy_parameters_by_exact_name": True,
            "legacy_adam_moments_by_exact_parameter_name": True,
            "positional_optimizer_loading": False,
            "orthogonal_reinitialization": False,
            "sole_new_parameter": v04.TRACE_ENCODER_PARAMETER,
            "sole_new_parameter_shape": [512, 1],
            "sole_new_parameter_initialization": "exact_zero",
            "new_parameter_inherited_adam_state": False,
        },
        "equivalence": {
            "zero_trace_features_bit_exact": True,
            "zero_trace_actions_values_log_probabilities_bit_exact": True,
            "zero_trace_recurrent_state_bit_exact": True,
            "first_rollout_digest_profile": (
                v04.FIRST_ROLLOUT_DIGEST_PROFILE
            ),
            "first_rollout_transitions": v04.ROLLOUT_TRANSITIONS,
            "behavioral_divergence_before_first_optimizer_phase": False,
        },
        "randomness": {
            "architecture_initialization_seed": (
                v04.ARCHITECTURE_INITIALIZATION_SEED
            ),
            "algorithm_seed": v04.ALGORITHM_SEED,
            "worker_streams": list(v04.WORKER_STREAMS),
        },
        "budget": {
            "child_actions_per_arm": v04.CHILD_ACTION_BUDGET,
            "evaluation_interval": v04.EVALUATION_INTERVAL,
            "exam_count_per_arm": v04.EXAM_COUNT,
            "terminal_lifetime_actions": (
                v04.PARENT_LIFETIME_ACTIONS + v04.CHILD_ACTION_BUDGET
            ),
            "terminal_optimizer_updates": (
                v04.PARENT_OPTIMIZER_UPDATES
                + (v04.CHILD_ACTION_BUDGET // v04.ROLLOUT_TRANSITIONS)
                * v04.PPO_EPOCHS
            ),
        },
        "selection": {
            "trace_sham_selectable": False,
            "ineffective_trace_only_selectable_definition": True,
            "complete_u2s_terminal_three_gate": True,
            "development_checkpoint_reuse_authorized": False,
            "replication_requires_separate_version": True,
            "u3_authorized": False,
        },
        "trace_sham_config_sha256": _canonical_sha256(sham),
        "ineffective_trace_config_sha256": _canonical_sha256(candidate),
    }


def _storage_caps() -> dict[str, int]:
    return {
        "per_arm_bytes": v04.LINEAGE_CAP_BYTES,
        "scientific_cohort_bytes": v04.COHORT_SCIENTIFIC_CAP_BYTES,
        "media_bytes": v04.MEDIA_CAP_BYTES,
        "combined_bytes": v04.COMBINED_PLANNED_CAP_BYTES,
    }


def _resume_rule() -> dict[str, Any]:
    return {
        "resume_supported": False,
        "interruption_disposition": "whole_stage_a_operationally_incomplete",
        "single_arm_continuation": False,
        "replacement_requires_new_commit_tag_qualification_roots_and_seeds": True,
        "replacement_restarts_both_arms_from_confirmed_u1": True,
        "r3_root_resume_or_reuse_authorized": False,
    }


_FRESH_IDENTITY_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        rb'"architecture_initialization_seed"\s*:\s*20260761',
        rb'"algorithm_seed"\s*:\s*20260762',
        rb'"worker_stream"\s*:\s*2026076[2-5]',
        rb'"worker_streams"\s*:\s*\[\s*20260762',
        rb"--seed(?:=|\s+)2026076[1-5](?:\D|$)",
        rb"v04-ineffective-trace-stage-a-20260724",
    )
)


def _fresh_identity_contract() -> dict[str, Any]:
    return {
        "evidence_root": str(EVIDENCE_ROOT),
        "excluded_managed_roots": [
            str(root)
            for root in (
                CANONICAL_QUALIFICATION_DIRECTORY,
                CANONICAL_COHORT_ROOT,
                CANONICAL_MEDIA_ROOT,
            )
        ],
        "architecture_initialization_seed": v04.ARCHITECTURE_INITIALIZATION_SEED,
        "algorithm_seed": v04.ALGORITHM_SEED,
        "worker_streams": list(v04.WORKER_STREAMS),
        "cohort_id": "v0.4-ineffective-trace-stage-a-20260724",
        "consumption_matches": [],
        "identities_unconsumed": True,
    }


def _validate_fresh_identity_contract(value: Any) -> dict[str, Any]:
    expected = _fresh_identity_contract()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise IneffectiveTraceQualificationError(
            "v0.4 recorded fresh-identity contract changed"
        )
    return expected


def _fresh_identity_preflight() -> dict[str, Any]:
    """Fail if a prior evidence root consumed the prospective v0.4 identity."""

    evidence_root = EVIDENCE_ROOT
    excluded = (
        CANONICAL_QUALIFICATION_DIRECTORY,
        CANONICAL_COHORT_ROOT,
        CANONICAL_MEDIA_ROOT,
    )
    _reject_symlink_chain(evidence_root)
    if not evidence_root.is_dir():
        raise IneffectiveTraceQualificationError(
            "DungeonApprentice evidence root is missing"
        )
    suffixes = {
        ".json",
        ".jsonl",
        ".log",
        ".txt",
        ".md",
        ".sha256",
    }
    hits: list[str] = []
    for current, directories, files in os.walk(evidence_root, followlinks=False):
        current_path = Path(current)
        directories[:] = [
            name
            for name in sorted(directories)
            if not any(
                (current_path / name).absolute() == root
                or root in (current_path / name).absolute().parents
                for root in excluded
            )
        ]
        for name in sorted(files):
            path = current_path / name
            absolute = path.absolute()
            if any(absolute == root or root in absolute.parents for root in excluded):
                continue
            if path.is_symlink():
                raise IneffectiveTraceQualificationError(
                    "evidence scan encountered an unsafe symlink"
                )
            if path.suffix.lower() not in suffixes:
                continue
            try:
                metadata = path.stat()
            except OSError as error:
                raise IneffectiveTraceQualificationError(
                    "cannot inspect preserved evidence during seed scan"
                ) from error
            if metadata.st_size > 64 * 1024**2:
                continue
            value = _regular_file_bytes(
                path,
                "fresh identity evidence scan",
                maximum=64 * 1024**2,
            )
            if any(pattern.search(value) for pattern in _FRESH_IDENTITY_PATTERNS):
                hits.append(path.relative_to(evidence_root).as_posix())
    if hits:
        raise IneffectiveTraceQualificationError(
            f"v0.4 seed or run identity was already consumed: {hits[:5]}"
        )
    return _fresh_identity_contract()


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
        raise IneffectiveTraceQualificationError(
            "cannot inspect v0.4 storage"
        ) from error
    if (
        not stat.S_ISDIR(volume_stat.st_mode)
        or not stat.S_ISDIR(dungeon_stat.st_mode)
        or not os.path.ismount(volume)
        or volume_stat.st_dev == parent_stat.st_dev
        or dungeon_stat.st_dev != volume_stat.st_dev
    ):
        raise IneffectiveTraceQualificationError(
            "T7 Developer is not a separate mounted volume"
        )
    managed = (
        CANONICAL_QUALIFICATION_DIRECTORY,
        CANONICAL_COHORT_ROOT,
        CANONICAL_MEDIA_ROOT,
    )
    managed_roots_absent = True
    for path in (CANONICAL_COHORT_ROOT, CANONICAL_MEDIA_ROOT):
        if path.is_symlink():
            raise IneffectiveTraceQualificationError(
                "v0.4 managed root is unsafe"
            )
        if path.exists():
            managed_roots_absent = False
            if not permit_bound_managed_roots:
                raise IneffectiveTraceQualificationError(
                    "v0.4 cohort or media root is not fresh"
                )
            metadata = path.stat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or path.resolve(strict=True) != path
                or metadata.st_dev != volume_stat.st_dev
            ):
                raise IneffectiveTraceQualificationError(
                    "v0.4 managed root changed type or device"
                )
    if require_qualification_absent and (
        CANONICAL_QUALIFICATION_DIRECTORY.exists()
        or CANONICAL_QUALIFICATION_DIRECTORY.is_symlink()
    ):
        raise IneffectiveTraceQualificationError(
            "v0.4 qualification identity is not fresh"
        )
    protected = (
        v04.PARENT_CHECKPOINT,
        R3_ROOT,
        R3_QUALIFICATION_DIRECTORY,
        R3_MEDIA_ROOT,
        R3_SCREEN_LOG,
    )
    for artifact in protected:
        resolved = artifact.expanduser().resolve()
        if any(
            resolved == root
            or root in resolved.parents
            or resolved in root.parents
            for root in managed
        ):
            raise IneffectiveTraceQualificationError(
                "v0.4 roots overlap immutable evidence"
            )
    usage = shutil.disk_usage(volume)
    minimum = int(v04.MINIMUM_FREE_GIB * 1024**3)
    if usage.free < minimum:
        raise IneffectiveTraceQualificationError(
            "T7 Developer lacks the v0.4 free-space reserve"
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
        raise IneffectiveTraceQualificationError("storage preflight is missing")
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
        raise IneffectiveTraceQualificationError(
            "recorded storage preflight changed"
        )
    try:
        timestamp = datetime.fromisoformat(str(value["measured_at"]))
    except (KeyError, ValueError) as error:
        raise IneffectiveTraceQualificationError(
            "storage preflight timestamp is invalid"
        ) from error
    if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(None):
        raise IneffectiveTraceQualificationError(
            "storage preflight timestamp must be UTC"
        )


def _validate_fingerprint(value: Any, label: str) -> dict[str, Any]:
    fields = {
        "trained_timesteps",
        "optimizer_updates",
        "policy_tensor_sha256",
        "optimizer_state_sha256",
        "optimizer_has_state",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or not isinstance(value.get("trained_timesteps"), int)
        or not isinstance(value.get("optimizer_updates"), int)
        or value.get("optimizer_has_state") is not True
    ):
        raise IneffectiveTraceQualificationError(
            f"{label} model fingerprint changed"
        )
    _require_sha256(value.get("policy_tensor_sha256"), f"{label} policy")
    _require_sha256(value.get("optimizer_state_sha256"), f"{label} optimizer")
    return json.loads(json.dumps(dict(value)))


def _validate_rng_identity(value: Any, label: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"phase", "components", "aggregate_sha256"}
        or value.get("phase")
        not in {"post_reset_pre_action_one", "post_rollout_pre_optimizer"}
        or not isinstance(value.get("components"), Mapping)
        or _canonical_sha256(value["components"])
        != value.get("aggregate_sha256")
    ):
        raise IneffectiveTraceQualificationError(
            f"{label} RNG identity changed"
        )
    components = value["components"]
    workers = components.get("workers")
    if (
        not isinstance(workers, list)
        or len(workers) != v04.WORKERS
        or [worker.get("worker_index") for worker in workers]
        != list(range(v04.WORKERS))
        or [worker.get("worker_stream") for worker in workers]
        != list(v04.WORKER_STREAMS)
    ):
        raise IneffectiveTraceQualificationError(
            f"{label} RNG worker identity changed"
        )
    for key in (
        "python_random_sha256",
        "numpy_global_sha256",
        "torch_cpu_sha256",
        "scheduler_sha256",
    ):
        _require_sha256(components.get(key), f"{label} {key}")
    return json.loads(json.dumps(dict(value)))


def _validate_smoke_workers(value: Any, arm_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != v04.WORKERS:
        raise IneffectiveTraceQualificationError(
            f"{arm_name} smoke worker evidence changed"
        )
    normalized: list[dict[str, Any]] = []
    for index, worker in enumerate(value):
        if (
            not isinstance(worker, Mapping)
            or worker.get("worker_index") != index
            or worker.get("transitions") != v04.ROLLOUT_STEPS
            or not isinstance(worker.get("episodes_started"), int)
            or int(worker["episodes_started"]) <= 0
            or set(worker.get("action_counts", {}))
            != {str(action) for action in range(7)}
            or sum(int(count) for count in worker["action_counts"].values())
            != v04.ROLLOUT_STEPS
            or set(worker.get("lesson_transition_counts", {}))
            != {lesson.value for lesson in lessons.LessonId}
            or sum(
                int(count)
                for count in worker["lesson_transition_counts"].values()
            )
            != v04.ROLLOUT_STEPS
            or worker.get("penalty_events") != 0
            or worker.get("maximum_episode_penalty_count") != 0
            or float.fromhex(str(worker.get("penalty_total_hex"))) != 0.0
        ):
            raise IneffectiveTraceQualificationError(
                f"{arm_name} smoke worker {index} changed"
            )
        for key in (
            "episode_starts_sha256",
            "trajectory_sha256",
            "reward_evidence_sha256",
        ):
            _require_sha256(worker.get(key), f"{arm_name} worker {key}")
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
        or not 0
        <= int(value["minimum_seed"])
        <= int(value["maximum_seed"])
        < 1_000_000
        or value.get("training_range_only") is not True
        or value.get("separated_unlock_roles") not in ([], ["training"])
        or value.get("protected_roles") != expected_protected
        or value.get("protected_seed_hits") != []
        or value.get("confirmation_or_final_seed_generated") is not False
    ):
        raise IneffectiveTraceQualificationError(
            f"{arm_name} smoke used non-training seed evidence"
        )
    return json.loads(json.dumps(dict(value)))


def _validate_trace_exercise(
    value: Any,
    *,
    mode: IneffectiveTraceMode,
) -> dict[str, Any]:
    fields = {
        "transitions",
        "visible_changed",
        "visible_unchanged",
        "raw_trace_nonzero_transitions",
        "raw_trace_max",
        "raw_trace_histogram",
        "exposed_trace_nonzero_transitions",
        "exposed_trace_max",
        "trace_enabled",
        "descriptive_only",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise IneffectiveTraceQualificationError(
            f"{mode.value} trace exercise schema changed"
        )
    histogram = value.get("raw_trace_histogram")
    enabled = mode is IneffectiveTraceMode.STREAK_TRACE
    histogram_valid = isinstance(histogram, Mapping) and all(
        isinstance(key, str)
        and key == str(int(key))
        and 0 <= int(key) <= v04.TRACE_CAP
        and isinstance(count, int)
        and count >= 0
        for key, count in histogram.items()
    )
    if (
        value.get("transitions") != v04.ROLLOUT_TRANSITIONS
        or int(value.get("visible_changed", -1))
        + int(value.get("visible_unchanged", -1))
        != v04.ROLLOUT_TRANSITIONS
        or not histogram_valid
        or sum(int(count) for count in histogram.values())
        != v04.ROLLOUT_TRANSITIONS
        or int(value.get("raw_trace_nonzero_transitions", 0)) <= 0
        or not 1 <= int(value.get("raw_trace_max", 0)) <= v04.TRACE_CAP
        or value.get("trace_enabled") is not enabled
        or value.get("descriptive_only") is not True
    ):
        raise IneffectiveTraceQualificationError(
            f"{mode.value} trace was not completely exercised"
        )
    if enabled:
        if (
            int(value.get("exposed_trace_nonzero_transitions", 0)) <= 0
            or not 0.0 < float(value.get("exposed_trace_max", 0.0)) <= 1.0
        ):
            raise IneffectiveTraceQualificationError(
                "candidate did not expose a bounded nonzero trace"
            )
    elif (
        value.get("exposed_trace_nonzero_transitions") != 0
        or float(value.get("exposed_trace_max", -1.0)) != 0.0
    ):
        raise IneffectiveTraceQualificationError(
            "trace-sham exposed a nonzero trace"
        )
    return json.loads(json.dumps(dict(value)))


def _validate_encoder_update(
    value: Any,
    *,
    mode: IneffectiveTraceMode,
) -> dict[str, Any]:
    fields = {
        "weight_nonzero_parameters",
        "gradient_observations",
        "gradient_nonzero_observations",
        "gradient_nonzero_parameters_max",
        "adam_state_present",
        "adam_exp_avg_nonzero_parameters",
        "adam_exp_avg_sq_nonzero_parameters",
        "weight_exact_zero",
        "adam_moments_exact_zero",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("adam_state_present") is not True
        or int(value.get("gradient_observations", 0)) <= 0
    ):
        raise IneffectiveTraceQualificationError(
            f"{mode.value} encoder update evidence changed"
        )
    counts = (
        int(value.get("weight_nonzero_parameters", -1)),
        int(value.get("adam_exp_avg_nonzero_parameters", -1)),
        int(value.get("adam_exp_avg_sq_nonzero_parameters", -1)),
    )
    if any(count < 0 or count > 512 for count in counts):
        raise IneffectiveTraceQualificationError(
            f"{mode.value} encoder update count changed"
        )
    if mode is IneffectiveTraceMode.ZERO_TRACE:
        if (
            counts != (0, 0, 0)
            or value.get("gradient_nonzero_observations") != 0
            or value.get("gradient_nonzero_parameters_max") != 0
            or value.get("weight_exact_zero") is not True
            or value.get("adam_moments_exact_zero") is not True
        ):
            raise IneffectiveTraceQualificationError(
                "trace-sham encoder or Adam moments changed from exact zero"
            )
    elif (
        any(count <= 0 for count in counts)
        or int(value.get("gradient_nonzero_observations", 0)) <= 0
        or int(value.get("gradient_nonzero_parameters_max", 0)) <= 0
        or value.get("weight_exact_zero") is not False
        or value.get("adam_moments_exact_zero") is not False
    ):
        raise IneffectiveTraceQualificationError(
            "candidate trace encoder did not learn after the optimizer"
        )
    return json.loads(json.dumps(dict(value)))


def _validate_smoke_transplant(
    value: Any,
    *,
    mode: IneffectiveTraceMode,
) -> dict[str, Any]:
    top_fields = {
        "architecture_version",
        "transplant",
        "zero_context_equivalence",
        "policy_contract",
    }
    transplant_fields = {
        "inherited_parameter_names",
        "new_parameter_names",
        "inherited_optimizer_state_names",
        "missing_optimizer_state_names",
        "inherited_timesteps",
        "inherited_optimizer_updates",
        "trace_encoder_zero",
    }
    equivalence_fields = {
        "batch",
        "features_exact",
        "actions_exact",
        "values_exact",
        "log_probabilities_exact",
        "recurrent_states_exact",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != top_fields
        or value.get("architecture_version") != v04.ARCHITECTURE_VERSION
        or value.get("policy_contract") != v04.public_policy_contract()
    ):
        raise IneffectiveTraceQualificationError(
            f"{mode.value} transplant envelope changed"
        )
    transplant = value.get("transplant")
    equivalence = value.get("zero_context_equivalence")
    if (
        not isinstance(transplant, Mapping)
        or set(transplant) != transplant_fields
        or transplant.get("inherited_parameter_names")
        != list(EXPECTED_U1_PARAMETER_NAMES)
        or transplant.get("new_parameter_names")
        != [v04.TRACE_ENCODER_PARAMETER]
        or transplant.get("inherited_optimizer_state_names")
        != list(EXPECTED_U1_PARAMETER_NAMES)
        or transplant.get("missing_optimizer_state_names") != []
        or transplant.get("inherited_timesteps")
        != v04.PARENT_LIFETIME_ACTIONS
        or transplant.get("inherited_optimizer_updates")
        != v04.PARENT_OPTIMIZER_UPDATES
        or transplant.get("trace_encoder_zero") is not True
        or not isinstance(equivalence, Mapping)
        or set(equivalence) != equivalence_fields
        or equivalence.get("batch") != 4
        or any(
            equivalence.get(field) is not True
            for field in equivalence_fields - {"batch"}
        )
    ):
        raise IneffectiveTraceQualificationError(
            f"{mode.value} transplant evidence changed"
        )
    return json.loads(json.dumps(dict(value)))


def validate_disposable_smoke(
    value: Any,
    *,
    source_commit: str,
    guard_mapping_sha256: str,
    sampler_preflight_sha256: str,
) -> dict[str, Any]:
    """Validate the real, destroyed two-arm rollout and optimizer smoke."""

    top_fields = {
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
        "raw_trace_evidence_identical",
        "arms",
        "canonical_predecessor_roots_unchanged",
        "temporary_root_removed",
        "updated_archives_reloaded_exactly",
        "updated_archives_destroyed",
        "scientific_evidence",
        "checkpoint_reuse_authorized",
        "r3_checkpoint_or_state_reused",
    }
    source = value.get("source") if isinstance(value, Mapping) else None
    arms = value.get("arms") if isinstance(value, Mapping) else None
    sampler = value.get("sampler_preflight") if isinstance(value, Mapping) else None
    required_true = (
        "pre_action_rng_identical",
        "pre_update_behavior_identical",
        "first_rollout_trajectory_identical",
        "first_rollout_policy_outputs_identical",
        "post_rollout_pre_optimizer_rng_identical",
        "first_rollout_episode_ledger_identical",
        "learning_divergence_begins_after_first_update",
        "raw_trace_evidence_identical",
        "canonical_predecessor_roots_unchanged",
        "temporary_root_removed",
        "updated_archives_reloaded_exactly",
        "updated_archives_destroyed",
    )
    if (
        not isinstance(value, Mapping)
        or set(value) != top_fields
        or value.get("schema_version") != smoke.SCHEMA_VERSION
        or value.get("protocol") != smoke.SMOKE_PROTOCOL
        or value.get("verdict") != "passed"
        or not isinstance(source, Mapping)
        or source.get("commit") != source_commit
        or source.get("dirty") is not False
        or value.get("parent_checkpoint") != str(v04.PARENT_CHECKPOINT)
        or value.get("parent_checkpoint_sha256")
        != v04.PARENT_CHECKPOINT_SHA256
        or value.get("worker_streams") != list(v04.WORKER_STREAMS)
        or value.get("guard_mapping_sha256") != guard_mapping_sha256
        or not isinstance(sampler, list)
        or value.get("sampler_preflight_sha256")
        != sampler_preflight_sha256
        or value.get("sampler_preflight_sha256")
        != _canonical_sha256(sampler)
        or value.get("actions_per_arm") != v04.ROLLOUT_TRANSITIONS
        or not isinstance(arms, Mapping)
        or set(arms) != set(_ARMS)
        or any(value.get(key) is not True for key in required_true)
        or value.get("scientific_evidence") is not False
        or value.get("checkpoint_reuse_authorized") is not False
        or value.get("r3_checkpoint_or_state_reused") is not False
    ):
        raise IneffectiveTraceQualificationError(
            "v0.4 disposable smoke contract changed"
        )
    try:
        completed = datetime.fromisoformat(str(value["completed_at"]))
    except ValueError as error:
        raise IneffectiveTraceQualificationError(
            "v0.4 smoke timestamp is invalid"
        ) from error
    if completed.tzinfo is None or completed.utcoffset() != UTC.utcoffset(None):
        raise IneffectiveTraceQualificationError(
            "v0.4 smoke timestamp must be UTC"
        )
    normalized_arms: dict[str, dict[str, Any]] = {}
    for mode in v04.ARM_ORDER:
        arm = arms.get(mode.value)
        expected_fields = {
            "arm",
            "before",
            "after",
            "trace_projection_nonzero_parameters",
            "trace_exercise",
            "encoder_update",
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
            "updated_archive_destroyed",
        }
        if (
            not isinstance(arm, Mapping)
            or set(arm) != expected_fields
            or arm.get("arm") != mode.value
            or arm.get("updated_archive_reloaded_exactly") is not True
            or arm.get("updated_archive_destroyed") is not True
            or arm.get("development_checkpoint_reuse_authorized") is not False
        ):
            raise IneffectiveTraceQualificationError(
                f"{mode.value} disposable arm evidence changed"
            )
        before = _validate_fingerprint(
            arm.get("before"),
            f"{mode.value} pre-update",
        )
        after = _validate_fingerprint(
            arm.get("after"),
            f"{mode.value} post-update",
        )
        if (
            before["trained_timesteps"] != v04.PARENT_LIFETIME_ACTIONS
            or before["optimizer_updates"] != v04.PARENT_OPTIMIZER_UPDATES
            or after["trained_timesteps"] - before["trained_timesteps"]
            != v04.ROLLOUT_TRANSITIONS
            or after["optimizer_updates"] - before["optimizer_updates"]
            != v04.PPO_EPOCHS
            or after["policy_tensor_sha256"] == before["policy_tensor_sha256"]
            or after["optimizer_state_sha256"]
            == before["optimizer_state_sha256"]
        ):
            raise IneffectiveTraceQualificationError(
                f"{mode.value} smoke optimizer boundary changed"
            )
        trace = _validate_trace_exercise(
            arm.get("trace_exercise"),
            mode=mode,
        )
        update = _validate_encoder_update(
            arm.get("encoder_update"),
            mode=mode,
        )
        if arm.get("trace_projection_nonzero_parameters") != update[
            "weight_nonzero_parameters"
        ]:
            raise IneffectiveTraceQualificationError(
                f"{mode.value} trace projection count changed"
            )
        transplant = _validate_smoke_transplant(
            arm.get("transplant"),
            mode=mode,
        )
        workers = _validate_smoke_workers(arm.get("workers"), mode.value)
        trajectory = arm.get("trajectory_identity")
        if trajectory != smoke._trajectory_identity(workers):
            raise IneffectiveTraceQualificationError(
                f"{mode.value} first-rollout trajectory digest changed"
            )
        pre_rng = _validate_rng_identity(
            arm.get("pre_action_rng_identity"),
            f"{mode.value} pre-action",
        )
        post_rng = _validate_rng_identity(
            arm.get("post_rollout_rng_identity"),
            f"{mode.value} post-rollout",
        )
        _require_sha256(
            arm.get("policy_output_sha256"),
            f"{mode.value} policy output",
        )
        ledger = arm.get("episode_ledger")
        episode_starts = sum(worker["episodes_started"] for worker in workers)
        if (
            not isinstance(ledger, Mapping)
            or set(ledger) != {"records", "normalized_sha256"}
            or ledger.get("records") != episode_starts
        ):
            raise IneffectiveTraceQualificationError(
                f"{mode.value} episode ledger changed"
            )
        _require_sha256(
            ledger.get("normalized_sha256"),
            f"{mode.value} episode ledger",
        )
        seed_evidence = _validate_seed_evidence(
            arm.get("seed_evidence"),
            episode_starts=episode_starts,
            arm_name=mode.value,
        )
        normalized_arms[mode.value] = {
            **json.loads(json.dumps(dict(arm))),
            "before": before,
            "after": after,
            "trace_exercise": trace,
            "encoder_update": update,
            "transplant": transplant,
            "workers": workers,
            "pre_action_rng_identity": pre_rng,
            "post_rollout_rng_identity": post_rng,
            "seed_evidence": seed_evidence,
        }
    sham = normalized_arms[IneffectiveTraceMode.ZERO_TRACE.value]
    candidate = normalized_arms[IneffectiveTraceMode.STREAK_TRACE.value]
    raw_fields = (
        "transitions",
        "visible_changed",
        "visible_unchanged",
        "raw_trace_nonzero_transitions",
        "raw_trace_max",
        "raw_trace_histogram",
    )
    if (
        sham["before"] != candidate["before"]
        or sham["pre_action_rng_identity"]
        != candidate["pre_action_rng_identity"]
        or sham["trajectory_identity"] != candidate["trajectory_identity"]
        or sham["policy_output_sha256"]
        != candidate["policy_output_sha256"]
        or sham["post_rollout_rng_identity"]
        != candidate["post_rollout_rng_identity"]
        or sham["episode_ledger"]["normalized_sha256"]
        != candidate["episode_ledger"]["normalized_sha256"]
        or any(
            sham["trace_exercise"][key]
            != candidate["trace_exercise"][key]
            for key in raw_fields
        )
    ):
        raise IneffectiveTraceQualificationError(
            "v0.4 twins diverged before the first optimizer"
        )
    return {
        "schema_version": smoke.SCHEMA_VERSION,
        "protocol": smoke.SMOKE_PROTOCOL,
        "verdict": "passed",
        "source_commit": source_commit,
        "parent_checkpoint_sha256": v04.PARENT_CHECKPOINT_SHA256,
        "worker_streams": list(v04.WORKER_STREAMS),
        "guard_mapping_sha256": guard_mapping_sha256,
        "sampler_preflight_sha256": sampler_preflight_sha256,
        "actions_per_arm": v04.ROLLOUT_TRANSITIONS,
        "pre_update_behavior_identical": True,
        "raw_trace_evidence_identical": True,
        "candidate_trace_exercised": True,
        "candidate_encoder_learned_after_optimizer": True,
        "sham_encoder_and_adam_exact_zero": True,
        "updated_archives_reloaded_exactly": True,
        "updated_archives_destroyed": True,
        "temporary_root_removed": True,
        "scientific_evidence": False,
        "checkpoint_reuse_authorized": False,
        "r3_checkpoint_or_state_reused": False,
        "arms": normalized_arms,
    }


def _collect_static_inputs(
    repository: Path,
    *,
    scan_fresh_identities: bool = True,
) -> tuple[
    dict[str, Any],
    Any,
    Mapping[lessons.LessonId, frozenset[str]],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    parent, base, mapping, guards, predecessors = (
        _authenticate_parent_and_guards(repository)
    )
    sampler = _sampler_preflight(
        mapping,
        seed_access=base.seed_access(),
    )
    r3_terminal = authenticate_r3_terminal(repository)
    protected = _protected_seed_partitions()
    fresh = (
        _fresh_identity_preflight()
        if scan_fresh_identities
        else _fresh_identity_contract()
    )
    architecture = _architecture_contract()
    return (
        parent,
        base,
        mapping,
        guards,
        predecessors,
        sampler,
        r3_terminal,
        protected,
        fresh,
        architecture,
    )


def build_ineffective_trace_tag_payload(
    repository: Path,
    *,
    source_commit: str,
    scan_fresh_identities: bool = True,
) -> dict[str, Any]:
    """Build the canonical one-line, pre-qualification tag payload."""

    root = repository.expanduser().resolve()
    commit = _require_git_object(source_commit, "v0.4 source commit")
    protocol_path = root / PROTOCOL_DOCUMENT
    _reject_symlink_chain(protocol_path)
    if protocol_path.is_symlink() or not protocol_path.is_file():
        raise IneffectiveTraceQualificationError(
            "v0.4 protocol document is missing or unsafe"
        )
    (
        parent,
        _base,
        _mapping,
        guards,
        _predecessors,
        sampler,
        r3_terminal,
        protected,
        fresh,
        architecture,
    ) = _collect_static_inputs(
        root,
        scan_fresh_identities=scan_fresh_identities,
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
        "r3_terminal_evidence_sha256": _canonical_sha256(r3_terminal),
        "guard_mapping_sha256": guards["applied_mapping_sha256"],
        "sampler_preflight_sha256": _canonical_sha256(sampler),
        "architecture_contract_sha256": _canonical_sha256(architecture),
        "protected_partitions_sha256": _canonical_sha256(protected),
        "fresh_identity_preflight_sha256": _canonical_sha256(fresh),
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
        raise IneffectiveTraceQualificationError(
            "internal v0.4 tag payload is incomplete"
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
    """Verify one clean pushed source and its published annotated v0.4 tag."""

    root = repository.expanduser().resolve()
    head = _require_git_object(
        _git(
            root,
            ["rev-parse", "--verify", "HEAD^{commit}"],
            runner=runner,
        ),
        "v0.4 source commit",
    )
    if (
        expected_source_commit is not None
        and head
        != _require_git_object(
            expected_source_commit,
            "expected v0.4 source commit",
        )
    ):
        raise IneffectiveTraceQualificationError(
            "v0.4 source differs from qualification"
        )
    if _git(
        root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        runner=runner,
    ):
        raise IneffectiveTraceQualificationError(
            "v0.4 qualification requires a clean repository"
        )
    ref = f"refs/tags/{QUALIFIED_TAG}"
    if _git(root, ["cat-file", "-t", ref], runner=runner) != "tag":
        raise IneffectiveTraceQualificationError(
            "v0.4 source tag must be annotated"
        )
    tag_object = _require_git_object(
        _git(root, ["rev-parse", "--verify", ref], runner=runner),
        "v0.4 tag object",
    )
    if (
        expected_tag_object is not None
        and tag_object
        != _require_git_object(
            expected_tag_object,
            "expected v0.4 tag object",
        )
    ):
        raise IneffectiveTraceQualificationError(
            "v0.4 tag object changed"
        )
    if _git(root, ["rev-list", "-n", "1", ref], runner=runner) != head:
        raise IneffectiveTraceQualificationError(
            "v0.4 tag does not point to HEAD"
        )
    remote_url = _git(
        root,
        ["remote", "get-url", QUALIFIED_REMOTE],
        runner=runner,
    )
    if remote_url != EXPECTED_ORIGIN_URL:
        raise IneffectiveTraceQualificationError("v0.4 origin changed")
    remote = _git(
        root,
        ["ls-remote", "--tags", QUALIFIED_REMOTE, ref],
        runner=runner,
    ).split()
    if remote != [tag_object, ref]:
        raise IneffectiveTraceQualificationError(
            "v0.4 annotated tag is not published"
        )
    raw_message = _git(
        root,
        ["for-each-ref", "--format=%(contents)", ref],
        runner=runner,
    )
    if "\n" in raw_message:
        raise IneffectiveTraceQualificationError(
            "v0.4 tag message must be one JSON line"
        )
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError as error:
        raise IneffectiveTraceQualificationError(
            "v0.4 tag message is not JSON"
        ) from error
    expected = (
        build_ineffective_trace_tag_payload(
            root,
            source_commit=head,
            # Before the durable claim, scan all preserved evidence to prove
            # the identity is unused. After claim, the canonical report binds
            # that historical fact; active launcher logs necessarily contain
            # the now-consumed identifiers and must not invalidate read-only
            # reauthentication by trainers or the manifest.
            scan_fresh_identities=not CANONICAL_CLAIM.is_file(),
        )
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
        raise IneffectiveTraceQualificationError(
            "v0.4 tag payload differs from live prospective inputs"
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


def _restrictions() -> dict[str, bool]:
    return {
        "arms_run_sequentially": True,
        "each_arm_starts_from_exact_confirmed_u1": True,
        "later_checkpoint_loading_authorized": False,
        "r3_parent_authorized": False,
        "stage_a_resume_supported": False,
        "replacement_restarts_both_arms_from_confirmed_u1": True,
        "confirmation_or_final_seed_issuer_opened": False,
        "canonical_or_claim_policy_updates_during_qualification": False,
        "disposable_smoke_policies_destroyed": True,
        "development_checkpoint_reuse_authorized": False,
        "replication_protocol_authorized": False,
        "u3_authorized": False,
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
    (
        parent,
        base,
        mapping,
        guards,
        predecessors,
        sampler,
        r3_terminal,
        protected,
        fresh,
        architecture,
    ) = _collect_static_inputs(
        repository,
        scan_fresh_identities=False,
    )
    sampler_sha256 = _canonical_sha256(sampler)
    smoke_evidence = validate_disposable_smoke(
        smoke_report,
        source_commit=str(source["commit"]),
        guard_mapping_sha256=str(guards["applied_mapping_sha256"]),
        sampler_preflight_sha256=sampler_sha256,
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
                "sha256": file_sha256(repository / PROTOCOL_DOCUMENT),
            },
            "parent": parent,
            "predecessors": predecessors,
            "r3_terminal_evidence": r3_terminal,
            "guards": guards,
            "sampler_preflight": sampler,
            "architecture_contract": architecture,
            "protected_partitions": protected,
            "fresh_identity_preflight": fresh,
            "smoke_evidence": smoke_evidence,
            "storage_caps": _storage_caps(),
            "restrictions": _restrictions(),
        },
        base,
        mapping,
    )


@dataclass(frozen=True)
class IneffectiveTraceQualificationEvidence:
    """Authenticated inputs exposed to the v0.4 launcher and trainer."""

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
    r3_terminal_evidence_sha256: str
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
            "architecture_contract_sha256": (
                self.architecture_contract_sha256
            ),
            "smoke_evidence_sha256": self.smoke_evidence_sha256,
            "protected_partitions_sha256": (
                self.protected_partitions_sha256
            ),
            "r3_terminal_evidence_sha256": (
                self.r3_terminal_evidence_sha256
            ),
            "storage_caps": dict(self.storage_caps),
        }

    def verified_report(self) -> dict[str, Any]:
        value = json.loads(self._report_bytes)
        if not isinstance(value, dict):
            raise IneffectiveTraceQualificationError(
                "verified v0.4 report bytes changed"
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
        raise IneffectiveTraceQualificationError(
            f"v0.4 qualification report must be {CANONICAL_REPORT}"
        )
    _reject_symlink_chain(requested)
    return requested


def _verify_checksum(path: Path, digest: str) -> None:
    try:
        fields = (
            _regular_file_bytes(
                path,
                "v0.4 qualification checksum",
                maximum=256,
            )
            .decode("ascii")
            .strip()
            .split()
        )
    except UnicodeDecodeError as error:
        raise IneffectiveTraceQualificationError(
            "v0.4 checksum is not ASCII"
        ) from error
    if fields != [digest, CANONICAL_REPORT.name]:
        raise IneffectiveTraceQualificationError(
            "v0.4 qualification checksum changed"
        )


def verify_ineffective_trace_qualification(
    path: Path,
    *,
    expected_source_commit: str,
    expected_tag_object: str | None = None,
    repository: Path | None = None,
) -> IneffectiveTraceQualificationEvidence:
    """Reauthenticate the one-shot canonical qualification and live inputs."""

    report_path = _assert_canonical_report(path)
    root = (
        Path(__file__).resolve().parents[2]
        if repository is None
        else repository.expanduser().resolve()
    )
    source = verify_source_tag(
        root,
        expected_source_commit=expected_source_commit,
        expected_tag_object=expected_tag_object,
    )
    report_bytes = _regular_file_bytes(
        report_path,
        "v0.4 qualification report",
        maximum=MAX_REPORT_BYTES,
    )
    try:
        report = json.loads(report_bytes)
    except json.JSONDecodeError as error:
        raise IneffectiveTraceQualificationError(
            "v0.4 qualification report is invalid JSON"
        ) from error
    if (
        not isinstance(report, dict)
        or set(report) != _REPORT_FIELDS
        or report_bytes != _canonical_json_bytes(report)
    ):
        raise IneffectiveTraceQualificationError(
            "v0.4 qualification report schema or canonical encoding changed"
        )
    digest = hashlib.sha256(report_bytes).hexdigest()
    _verify_checksum(CANONICAL_CHECKSUM, digest)
    claim_bytes = _regular_file_bytes(
        CANONICAL_CLAIM,
        "v0.4 qualification claim",
        maximum=16 * 1024,
    )
    try:
        claim = json.loads(claim_bytes)
    except json.JSONDecodeError as error:
        raise IneffectiveTraceQualificationError(
            "v0.4 qualification claim is invalid JSON"
        ) from error
    if (
        not isinstance(claim, dict)
        or claim_bytes != _canonical_json_bytes(claim)
    ):
        raise IneffectiveTraceQualificationError(
            "v0.4 qualification claim is not canonical"
        )
    if report.get("claim") != claim:
        raise IneffectiveTraceQualificationError(
            "v0.4 qualification claim binding changed"
        )
    smoke_report = report.get("smoke_evidence", {}).get("_full_report")
    if not isinstance(smoke_report, Mapping):
        raise IneffectiveTraceQualificationError(
            "v0.4 qualification lacks its disposable smoke transcript"
        )
    expected, base, mapping = _expected_report(
        root,
        source=source,
        claim=claim,
        smoke_report=smoke_report,
    )
    expected["smoke_evidence"]["_full_report"] = json.loads(
        json.dumps(smoke_report)
    )
    expected["storage_preflight"] = report.get("storage_preflight")
    expected["created_at"] = report.get("created_at")
    if report != expected:
        raise IneffectiveTraceQualificationError(
            "v0.4 qualification differs from live prospective inputs"
        )
    _verify_storage_preflight(report["storage_preflight"])
    try:
        created_at = datetime.fromisoformat(str(report["created_at"]))
    except ValueError as error:
        raise IneffectiveTraceQualificationError(
            "v0.4 qualification timestamp is invalid"
        ) from error
    if created_at.tzinfo is None or created_at.utcoffset() != UTC.utcoffset(None):
        raise IneffectiveTraceQualificationError(
            "v0.4 qualification timestamp must be UTC"
        )
    smoke_public = {
        key: value
        for key, value in report["smoke_evidence"].items()
        if key != "_full_report"
    }
    return IneffectiveTraceQualificationEvidence(
        report=str(report_path),
        report_sha256=digest,
        checksum=str(CANONICAL_CHECKSUM),
        source_commit=str(source["commit"]),
        tag=QUALIFIED_TAG,
        tag_object=str(source["tag_object"]),
        tag_payload_sha256=str(source["tag_payload_sha256"]),
        verdict=VERDICT,
        protocol_document_sha256=str(
            report["protocol_document"]["sha256"]
        ),
        guard_mapping_sha256=str(
            report["guards"]["applied_mapping_sha256"]
        ),
        sampler_preflight_sha256=_canonical_sha256(
            report["sampler_preflight"]
        ),
        architecture_contract_sha256=_canonical_sha256(
            report["architecture_contract"]
        ),
        smoke_evidence_sha256=_canonical_sha256(smoke_public),
        protected_partitions_sha256=_canonical_sha256(
            report["protected_partitions"]
        ),
        r3_terminal_evidence_sha256=_canonical_sha256(
            report["r3_terminal_evidence"]
        ),
        storage_caps=MappingProxyType(dict(report["storage_caps"])),
        _report_bytes=report_bytes,
        _base_qualification=base,
        _forbidden_layouts=mapping,
    )


def collect_ineffective_trace_qualification(
    *,
    repository: Path,
    output: Path = CANONICAL_REPORT,
    smoke_runner: Callable[..., Mapping[str, Any]] = smoke.run_smoke,
) -> IneffectiveTraceQualificationEvidence:
    """Claim and collect the only canonical v0.4 qualification attempt."""

    report_path = _assert_canonical_report(output)
    root = repository.expanduser().resolve()
    source = verify_source_tag(root)
    storage_preflight = _storage_preflight(
        require_qualification_absent=True
    )
    (
        _parent,
        smoke_base,
        smoke_mapping,
        smoke_guards,
        _predecessors,
        _sampler,
        _r3,
        _protected,
        _fresh,
        _architecture,
    ) = _collect_static_inputs(root)
    if report_path.parent.exists() or report_path.parent.is_symlink():
        raise IneffectiveTraceQualificationError(
            "v0.4 qualification identity was already claimed"
        )
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
            repository=root,
            require_clean_source=True,
            seed_access=smoke_base.seed_access(),
            forbidden_layout_hashes=smoke_mapping,
        )
        if (
            smoke_report.get("guard_mapping_sha256")
            != smoke_guards["applied_mapping_sha256"]
        ):
            raise IneffectiveTraceQualificationError(
                "v0.4 disposable smoke used a different static guard"
            )
        expected, _base, _mapping = _expected_report(
            root,
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
            raise IneffectiveTraceQualificationError(
                "internal v0.4 report schema is incomplete"
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
        # Keep claim.json forever. This exact source/tag/root identity cannot
        # be silently retried after any failure.
        raise
    return verify_ineffective_trace_qualification(
        report_path,
        expected_source_commit=str(source["commit"]),
        expected_tag_object=str(source["tag_object"]),
        repository=root,
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
            _git(
                repository,
                ["rev-parse", "--verify", "HEAD^{commit}"],
            ),
            "v0.4 source commit",
        )
        if _git(
            repository,
            ["status", "--porcelain=v1", "--untracked-files=all"],
        ):
            raise SystemExit(
                "v0.4 tag payload requires a clean committed repository"
            )
        print(
            json.dumps(
                build_ineffective_trace_tag_payload(
                    repository,
                    source_commit=head,
                ),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return
    evidence = collect_ineffective_trace_qualification(
        repository=repository,
        output=args.output,
    )
    print(json.dumps(evidence.public_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
