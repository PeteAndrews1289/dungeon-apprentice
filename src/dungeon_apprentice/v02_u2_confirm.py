"""Collision-safe, no-update confirmation for the three mastered U2 policies.

Importing this module does not open a confirmation seed.  Reserved access is
minted only after the canonical attempt claim has been durably written.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import secrets
import sys
import time
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from minigrid.core.world_object import Door

from dungeon_apprentice import v02_u1_confirm as model_integrity
from dungeon_apprentice import v02_u2 as frozen_training
from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice.artifacts import (
    atomic_write_json,
    file_sha256,
    git_snapshot,
    runtime_snapshot,
    utc_now,
)
from dungeon_apprentice.contracts import PIXEL_SHAPE, STEP_REWARD, SUCCESS_REWARD
from dungeon_apprentice.oracle import DungeonOracle, OracleFailure
from dungeon_apprentice.u2_confirmation_anchor import publish_external_anchor
from dungeon_apprentice.u2_qualification_anchor import (
    verify_external_anchor as verify_qualification_anchor,
)
from dungeon_apprentice.u2_seed_guard import (
    PARTITION_BY_ROLE,
    U2SeedAccess,
    U2SeedAccessError,
    U2SeedRole,
    _new_confirmation_launch_claim,
    _post_training_confirmation_seed_access,
    seed_partition_ledger,
)

CONFIRMATION_PROTOCOL = "dungeon-apprentice-v0.2-u2-confirmation"
CONFIRMATION_SCHEMA_VERSION = 1
ATTEMPT_PROTOCOL = f"{CONFIRMATION_PROTOCOL}-attempt"
ATTEMPT_SCHEMA_VERSION = 1
ATTEMPT_ID = "u2-confirmation-v0.2-u2-20260723-attempt-1"
ACKNOWLEDGEMENT = "OPEN U2 CONFIRMATION 15200000-15239999 ONCE"
CONFIRMATION_CASES = 200
CONFIRMATION_PANEL_SIZE = 100
CANDIDATE_COUNT = 10_000
TRAINING_SOURCE_COMMIT = "b7b5d361b0aa2eeabedc435fa0d4b9e1ffdd09db"
QUALIFICATION_SHA256 = (
    "61397579ec2754ae53c972e1553607d2db28184de462cd3073f845f3ca8ac955"
)
COHORT_SHA256 = "982b1fe6c58f5fa652dff1d2be0894f1918d2927e492b896b3fd102e52da7843"
U1_CONFIRMATION_SHA256 = (
    "6e577170050f6f14599b793a031776a19bf7c64eba0f243f457298da3193ae8f"
)

CANONICAL_COHORT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/u2-separated-20260723"
)
CANONICAL_COHORT_MANIFEST = CANONICAL_COHORT / "cohort.json"
CANONICAL_QUALIFICATION = Path(
    "/Volumes/T7 Developer/DungeonApprentice/qualifications/"
    "v0.2-u2-20260723/report.json"
)
CANONICAL_U1_CONFIRMATION = Path(
    "/Volumes/T7 Developer/DungeonApprentice/confirmations/"
    "v0.2-u1-v2-20260723/report.json"
)
CANONICAL_ATTEMPT_DIRECTORY = Path(
    "/Volumes/T7 Developer/DungeonApprentice/confirmations/v0.2-u2-20260723"
)

CANDIDATE_ROLES: Mapping[lessons.LessonId, U2SeedRole] = {
    lessons.LessonId.NAVIGATE: U2SeedRole.FUTURE_NAVIGATE_CONFIRMATION,
    lessons.LessonId.VISIBLE_UNLOCK: U2SeedRole.FUTURE_U0_CONFIRMATION,
    lessons.LessonId.LOCAL_UNLOCK: U2SeedRole.FUTURE_U1_CONFIRMATION,
    lessons.LessonId.SEPARATED_UNLOCK: U2SeedRole.FUTURE_U2_CONFIRMATION,
}

CAPABILITY_GATES: Mapping[lessons.LessonId, tuple[int, int]] = {
    lessons.LessonId.NAVIGATE: (170, 85),
    lessons.LessonId.VISIBLE_UNLOCK: (170, 80),
    lessons.LessonId.LOCAL_UNLOCK: (170, 80),
    lessons.LessonId.SEPARATED_UNLOCK: (170, 80),
}

GEOMETRY_GATES: Mapping[lessons.LessonId, tuple[int, int] | None] = {
    lessons.LessonId.NAVIGATE: None,
    lessons.LessonId.VISIBLE_UNLOCK: None,
    lessons.LessonId.LOCAL_UNLOCK: (190, 95),
    lessons.LessonId.SEPARATED_UNLOCK: (190, 95),
}


@dataclass(frozen=True)
class FrozenU2Checkpoint:
    child_seed: int
    u1_parent_seed: int
    child_trained_actions: int
    inherited_trained_actions: int
    lifetime_trained_actions: int
    optimizer_updates: int
    first_pass_actions: int
    first_pass_sha256: str
    first_pass_sidecar_sha256: str
    first_pass_integrity_sha256: str
    archive: Path
    archive_sha256: str
    sidecar_sha256: str
    integrity_sha256: str
    manifest_sha256: str
    evaluations_sha256: str
    events_sha256: str
    episodes_sha256: str
    optimizer_sha256: str

    @property
    def run_directory(self) -> Path:
        return self.archive.parent.parent


FROZEN_CHECKPOINTS: Mapping[int, FrozenU2Checkpoint] = {
    20260737: FrozenU2Checkpoint(
        child_seed=20260737,
        u1_parent_seed=20260725,
        child_trained_actions=688_128,
        inherited_trained_actions=884_736,
        lifetime_trained_actions=1_572_864,
        optimizer_updates=3_072,
        first_pass_actions=655_360,
        first_pass_sha256=(
            "6f8adfeba1897afcd0842a43d0762fd5bec20abb3d1ad9a6f35e401da3bc0f21"
        ),
        first_pass_sidecar_sha256=(
            "703f03379b179997a8cb27728a7e7b69b6806d1dee46efb17ab1069ac8bca2aa"
        ),
        first_pass_integrity_sha256=(
            "4218ad45e73b6cb26bd93b07024e97b90955dbc1731f3e6f9d129c0f26ec49a2"
        ),
        archive=CANONICAL_COHORT
        / "v02-u2-seed-20260737/checkpoints/mastered-separated-unlock.zip",
        archive_sha256=(
            "9d0fcec87309f7777b06fa14f4f70e633fcf4b0a73b30a8ba1a92f04b7530e14"
        ),
        sidecar_sha256=(
            "f525edf6565af4dcf3b012fe640aa80e3540d49e552cc3330c76422ef695ca6a"
        ),
        integrity_sha256=(
            "e27029c862404095c38776fe0c1a8345ba42e1faa470e0bdc3d313e6aa6229b8"
        ),
        manifest_sha256=(
            "c6e20f9be6bf9631e2602b299794bc19c59c6bf6b6403fbb6afa31539fb74f94"
        ),
        evaluations_sha256=(
            "00dbf99adde1fd18a326832691abceabd8cefa797d7d9833b89b5fcd8bb99cd1"
        ),
        events_sha256=(
            "1ce86372f8e29a92b648f92237362c128e5f0a9b3463fe4089748ebebf46f2a3"
        ),
        episodes_sha256=(
            "05899240805bd423a0c357c362b25a85da9bba51ad8a9a52c469a3360140ca8d"
        ),
        optimizer_sha256=(
            "121d892835ad642acd53511f22c208e3f86d5f38f276f4bf1f18a7a01c1ae4c5"
        ),
    ),
    20260741: FrozenU2Checkpoint(
        child_seed=20260741,
        u1_parent_seed=20260729,
        child_trained_actions=557_056,
        inherited_trained_actions=884_736,
        lifetime_trained_actions=1_441_792,
        optimizer_updates=2_816,
        first_pass_actions=524_288,
        first_pass_sha256=(
            "ef2967e8e625456042af2df31f39c790d67293823446f3027a7fb43eedf62dce"
        ),
        first_pass_sidecar_sha256=(
            "9d12ae47bf5d112217ac68ca0328725ca6ce88a450d601b7b3ba951488531743"
        ),
        first_pass_integrity_sha256=(
            "9bb97b8568c2b550998dc031896959d4de3d9350b5135b33c795d1ee548e156b"
        ),
        archive=CANONICAL_COHORT
        / "v02-u2-seed-20260741/checkpoints/mastered-separated-unlock.zip",
        archive_sha256=(
            "bbf4f17bf32f0879489d7ca555e723e2fa4c767efc32dcf1b88a94ce528f1bea"
        ),
        sidecar_sha256=(
            "95b872b605bf20829dfeb75c2ab2f135e6db120574e5e0f424ae132e4c0fb2af"
        ),
        integrity_sha256=(
            "48e18db09985c0b17833facf8567000a377b8a4508f64ec738fa12d5bd2297dd"
        ),
        manifest_sha256=(
            "d3f89c2f656757377cd17d359b8f507a926f54966535c405411f4f275fcbdd59"
        ),
        evaluations_sha256=(
            "c70c166bbba1ac52da50c913b3b8b35cf37a5ba6508804c9158fd203676ee33e"
        ),
        events_sha256=(
            "a98698a6a40cec7db74c53e3f6233d609e7c212d499f56acc8bc7e0f016da950"
        ),
        episodes_sha256=(
            "6bf772e58fd2c6a98b47f990cdc03981a47d86a7b516d29d167b982f6a4472aa"
        ),
        optimizer_sha256=(
            "a3280763cae840929769d085f70908053d6845b52d5e093e255995c43eba266f"
        ),
    ),
    20260745: FrozenU2Checkpoint(
        child_seed=20260745,
        u1_parent_seed=20260733,
        child_trained_actions=688_128,
        inherited_trained_actions=786_432,
        lifetime_trained_actions=1_474_560,
        optimizer_updates=2_880,
        first_pass_actions=655_360,
        first_pass_sha256=(
            "aad63e5dcba3bd3c28cbdd726161c1a38e05c1df1f777f53d48ab5b20c94bf2d"
        ),
        first_pass_sidecar_sha256=(
            "100ce750f3c2e8164710c89be8574c2492f7c70371d34fb4559d594500848366"
        ),
        first_pass_integrity_sha256=(
            "31fa498f371d3c610501291674e63d813f355e2adba99e773ddd3ece3289ac1c"
        ),
        archive=CANONICAL_COHORT
        / "v02-u2-seed-20260745/checkpoints/mastered-separated-unlock.zip",
        archive_sha256=(
            "56dc459fb94110f41a14b2304425f572fc235c8cfc07e1152306cbf77ac33dee"
        ),
        sidecar_sha256=(
            "68cfb84ec3fbfd4204cfa9679d5b8c1fe75b17bdf26d828ed1f92510b439cc29"
        ),
        integrity_sha256=(
            "250ebf0a8fe98ed5370d20c9b55531ec5e1be33df6821fc73b670628ddfb6151"
        ),
        manifest_sha256=(
            "4d37f72e2679441c1bd3eeff81c1c95a079b2f180d21ace136b5d80a1972e2ec"
        ),
        evaluations_sha256=(
            "d3ef6c0a59ff9b198076571ce78dc1a5d17b667cfa8aac92948a6616de3714d8"
        ),
        events_sha256=(
            "f4572ae55692bf66d9664f8f7ebcf901a8478e853a28798e8e4344957184f9bb"
        ),
        episodes_sha256=(
            "dd30982d02f536978a0913f4ed4f5a36be149c7d8136116550be4ea5e1b7cb6a"
        ),
        optimizer_sha256=(
            "6bc1ddfff177c2644ba811de27b60b7b540e86a6c203e7f27aa27ce295af8a7b"
        ),
    ),
}


class U2ConfirmationError(RuntimeError):
    """Raised before a result can count under the frozen confirmation."""


class CandidateRejected(RuntimeError):
    """One policy-blind candidate failed a frozen acceptance condition."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class VerifiedU2Checkpoint:
    frozen: FrozenU2Checkpoint
    sidecar: Mapping[str, Any]
    manifest: Mapping[str, Any]
    artifact_sha256s: Mapping[str, str]
    policy_member_sha256: str = ""

    def public_dict(self) -> dict[str, Any]:
        return {
            "child_seed": self.frozen.child_seed,
            "u1_parent_seed": self.frozen.u1_parent_seed,
            "checkpoint": str(self.frozen.archive),
            "checkpoint_sha256": self.frozen.archive_sha256,
            "sidecar": str(self.frozen.archive.with_suffix(".json")),
            "child_trained_actions": self.frozen.child_trained_actions,
            "inherited_trained_actions": self.frozen.inherited_trained_actions,
            "lifetime_trained_actions": self.frozen.lifetime_trained_actions,
            "optimizer_updates": self.frozen.optimizer_updates,
            "first_pass_actions": self.frozen.first_pass_actions,
            "first_pass_sha256": self.frozen.first_pass_sha256,
            "source_commit": TRAINING_SOURCE_COMMIT,
            "artifact_sha256s": dict(self.artifact_sha256s),
            "policy_member_sha256": self.policy_member_sha256,
        }


@dataclass(frozen=True)
class LessonReferenceExclusions:
    gating: Mapping[str, frozenset[str]]
    diagnostic: Mapping[str, frozenset[str]]
    sources: Mapping[str, Any]

    def public_dict(self) -> dict[str, Any]:
        return {
            "gating": {
                name: _hash_set_evidence(values)
                for name, values in self.gating.items()
            },
            "diagnostic": {
                name: _hash_set_evidence(values)
                for name, values in self.diagnostic.items()
            },
            "sources": dict(self.sources),
        }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise U2ConfirmationError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise U2ConfirmationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise U2ConfirmationError(f"{label} must be a JSON object: {path}")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise U2ConfirmationError(f"missing {label}: {path}")
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise U2ConfirmationError(
                        f"{label} line {line_number} is not an object"
                    )
                records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise U2ConfirmationError(f"cannot read {label} {path}: {error}") from error
    return records


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise U2ConfirmationError(f"{label} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise U2ConfirmationError(f"{label} is not a SHA-256 digest") from error
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _hash_set_sha256(values: Sequence[str] | set[str] | frozenset[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _hash_set_evidence(values: set[str] | frozenset[str]) -> dict[str, Any]:
    return {
        "unique_layouts": len(values),
        "set_sha256": _hash_set_sha256(values),
    }


def _seed_list_sha256(seeds: Sequence[int]) -> str:
    return _canonical_sha256([int(seed) for seed in seeds])


def _artifact_paths(frozen: FrozenU2Checkpoint) -> dict[str, Path]:
    run = frozen.run_directory
    first_pass = run / "checkpoints/first-pass-separated-unlock.zip"
    return {
        "checkpoint": frozen.archive,
        "sidecar": frozen.archive.with_suffix(".json"),
        "integrity": frozen.archive.with_suffix(".integrity.json"),
        "first_pass_checkpoint": first_pass,
        "first_pass_sidecar": first_pass.with_suffix(".json"),
        "first_pass_integrity": first_pass.with_suffix(".integrity.json"),
        "manifest": run / "manifest.json",
        "evaluations": run / "evaluations.jsonl",
        "events": run / "events.jsonl",
        "episodes": run / "episodes.jsonl",
        "optimizer": run / "optimizer.jsonl",
    }


def _expected_artifact_sha256s(frozen: FrozenU2Checkpoint) -> dict[str, str]:
    return {
        "checkpoint": frozen.archive_sha256,
        "sidecar": frozen.sidecar_sha256,
        "integrity": frozen.integrity_sha256,
        "first_pass_checkpoint": frozen.first_pass_sha256,
        "first_pass_sidecar": frozen.first_pass_sidecar_sha256,
        "first_pass_integrity": frozen.first_pass_integrity_sha256,
        "manifest": frozen.manifest_sha256,
        "evaluations": frozen.evaluations_sha256,
        "events": frozen.events_sha256,
        "episodes": frozen.episodes_sha256,
        "optimizer": frozen.optimizer_sha256,
    }


def _policy_archive_member_sha256(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            payload = archive.read("policy.pth")
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise U2ConfirmationError(
            f"cannot authenticate policy.pth in frozen archive {path}"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _measure_frozen_artifacts(frozen: FrozenU2Checkpoint) -> dict[str, str]:
    measured = {
        name: file_sha256(path)
        for name, path in _artifact_paths(frozen).items()
    }
    expected = _expected_artifact_sha256s(frozen)
    if measured != expected:
        changed = {
            name: {"expected": expected[name], "measured": measured.get(name)}
            for name in expected
            if measured.get(name) != expected[name]
        }
        raise U2ConfirmationError(
            f"frozen U2 lineage {frozen.child_seed} evidence changed: {changed}"
        )
    return measured


def _evaluation_records_at(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    child_actions: int,
    checkpoint_sha256: str,
) -> list[Mapping[str, Any]]:
    records = [
        item
        for item in evaluations
        if int(item.get("child_trained_actions", -1)) == child_actions
    ]
    if (
        len(records) != len(lessons.LessonId)
        or {item.get("lesson_id") for item in records}
        != {lesson.value for lesson in lessons.LessonId}
        or any(item.get("checkpoint_sha256") != checkpoint_sha256 for item in records)
        or any(item.get("counts_toward_gate") is not True for item in records)
    ):
        raise U2ConfirmationError(
            f"{child_actions} is not a complete passing four-lesson U2 exam"
        )
    for item in records:
        lesson = lessons.LessonId(str(item["lesson_id"]))
        spec = lessons.LESSON_SPECS[lesson]
        episodes = int(item.get("episodes", -1))
        successes = int(item.get("successes", -1))
        panel_successes = item.get("panel_successes")
        panel_rates = item.get("panel_success_rates")
        if (
            episodes != lessons.EVALUATION_SEED_COUNT
            or not isinstance(panel_successes, list)
            or len(panel_successes) != 2
            or not isinstance(panel_rates, list)
            or len(panel_rates) != 2
            or any(type(value) is not int for value in panel_successes)
            or sum(panel_successes) != successes
            or not math.isclose(
                float(item.get("success_rate", -1.0)),
                successes / episodes,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or any(
                not math.isclose(
                    float(rate),
                    count / lessons.EVALUATION_PANEL_SIZE,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for count, rate in zip(panel_successes, panel_rates, strict=True)
            )
        ):
            raise U2ConfirmationError(
                f"{child_actions}/{lesson.value} has inconsistent raw exam counts"
            )
        recomputed_pass = (
            successes >= spec.overall_required
            and all(count >= spec.panel_required for count in panel_successes)
        )
        if item.get("passed") is not recomputed_pass or not recomputed_pass:
            raise U2ConfirmationError(
                f"{child_actions}/{lesson.value} does not recompute as passing"
            )
    return records


def _verify_allocation_history(
    events: Sequence[Mapping[str, Any]],
    *,
    child_actions: int,
    final_allocation: Mapping[str, Any],
    scheduler: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every completed 32,768-action window from raw transition counts."""

    decisions = [item for item in events if item.get("type") == "curriculum_decision"]
    expected_windows = child_actions // frozen_training.EVALUATION_INTERVAL
    if (
        child_actions <= 0
        or child_actions % frozen_training.EVALUATION_INTERVAL
        or len(decisions) != expected_windows
    ):
        raise U2ConfirmationError("U2 allocation history has the wrong number of windows")
    lesson_names = {lesson.value for lesson in lessons.LessonId}
    profiles = lessons.target_profiles_public()
    prior_lifetime = {name: 0 for name in lesson_names}
    prior_profile: str | None = None
    maximum_deviation = 0.0
    for index, event in enumerate(decisions, start=1):
        allocation = event.get("allocation")
        if not isinstance(allocation, Mapping):
            raise U2ConfirmationError(f"U2 allocation window {index} is missing")
        profile = allocation.get("profile")
        targets = allocation.get("target_shares")
        transitions = allocation.get("window_transitions")
        recorded_realized = allocation.get("realized_shares")
        lifetime = allocation.get("lifetime_transitions")
        if (
            not isinstance(profile, str)
            or profile not in profiles
            or targets != profiles[profile]
            or not isinstance(transitions, Mapping)
            or set(transitions) != lesson_names
            or not isinstance(recorded_realized, Mapping)
            or set(recorded_realized) != lesson_names
            or not isinstance(lifetime, Mapping)
            or set(lifetime) != lesson_names
            or any(type(value) is not int or value < 0 for value in transitions.values())
            or any(type(value) is not int or value < 0 for value in lifetime.values())
        ):
            raise U2ConfirmationError(f"U2 allocation window {index} is malformed")
        total = sum(int(value) for value in transitions.values())
        if (
            total != frozen_training.EVALUATION_INTERVAL
            or int(event.get("allocation_transitions", -1)) != total
            or event.get("allocation_complete") is not True
        ):
            raise U2ConfirmationError(
                f"U2 allocation window {index} is not exactly 32,768 actions"
            )
        realized = {
            name: int(transitions[name]) / total
            for name in lesson_names
        }
        for name in lesson_names:
            if not math.isclose(
                float(recorded_realized[name]),
                realized[name],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise U2ConfirmationError(
                    f"U2 allocation window {index} recorded a false realized share"
                )
            deviation = abs(realized[name] - float(targets[name]))
            maximum_deviation = max(maximum_deviation, deviation)
            if deviation > lessons.ALLOCATION_TOLERANCE + 1e-12:
                raise U2ConfirmationError(
                    f"U2 allocation window {index} exceeded tolerance"
                )
        if event.get("allocation_within_tolerance") is not True:
            raise U2ConfirmationError(
                f"U2 allocation window {index} stores a false tolerance verdict"
            )
        boundary = int(event.get("child_trained_actions", -1))
        if boundary != index * frozen_training.EVALUATION_INTERVAL:
            raise U2ConfirmationError("U2 allocation boundary sequence is discontinuous")
        expected_lifetime = {
            name: prior_lifetime[name] + int(transitions[name])
            for name in lesson_names
        }
        if {name: int(lifetime[name]) for name in lesson_names} != expected_lifetime:
            raise U2ConfirmationError("U2 lifetime allocation counts do not add up")
        revision = int(allocation.get("revision", -1))
        expected_revision = index - 1
        if revision != expected_revision:
            raise U2ConfirmationError("U2 scheduler revision sequence changed")
        prior_lifetime = expected_lifetime
        prior_profile = profile
    authoritative = dict(decisions[-1]["allocation"])
    if dict(final_allocation) != authoritative:
        raise U2ConfirmationError(
            "U2 mastery sidecar allocation differs from recomputed history"
        )
    scheduler_window = scheduler.get("window_transitions")
    scheduler_realized = scheduler.get("realized_shares")
    if (
        scheduler.get("profile") != authoritative["profile"]
        or scheduler.get("target_shares") != authoritative["target_shares"]
        or scheduler.get("lifetime_transitions")
        != authoritative["lifetime_transitions"]
        or int(scheduler.get("revision", -1))
        != int(authoritative["revision"]) + 1
        or not isinstance(scheduler_window, Mapping)
        or set(scheduler_window) != lesson_names
        or any(value != 0 for value in scheduler_window.values())
        or not isinstance(scheduler_realized, Mapping)
        or set(scheduler_realized) != lesson_names
        or any(float(value) != 0.0 for value in scheduler_realized.values())
        or not isinstance(scheduler.get("rng_state"), Mapping)
    ):
        raise U2ConfirmationError(
            "U2 mastery scheduler is not the valid empty post-decision window"
        )
    if sum(prior_lifetime.values()) != child_actions:
        raise U2ConfirmationError("U2 lifetime allocation total changed")
    return {
        "windows": len(decisions),
        "maximum_deviation": maximum_deviation,
        "final_profile": prior_profile,
        "whole_child_transitions": dict(sorted(prior_lifetime.items())),
    }


def _decision_event_at(
    events: Sequence[Mapping[str, Any]],
    *,
    child_actions: int,
    checkpoint_sha256: str,
    mastered: bool,
) -> Mapping[str, Any]:
    matches = [
        item
        for item in events
        if item.get("type") == "curriculum_decision"
        and int(item.get("child_trained_actions", -1)) == child_actions
        and item.get("checkpoint_sha256") == checkpoint_sha256
    ]
    if len(matches) != 1:
        raise U2ConfirmationError(
            f"{child_actions} has no unique frozen U2 curriculum decision"
        )
    event = matches[0]
    curriculum = event.get("curriculum")
    passes = event.get("lesson_passes")
    allocation = event.get("allocation")
    if (
        event.get("allocation_complete") is not True
        or event.get("allocation_within_tolerance") is not True
        or not isinstance(allocation, Mapping)
        or allocation.get("profile") != "normal"
        or not isinstance(curriculum, Mapping)
        or curriculum.get("recovery") is not False
        or curriculum.get("recovery_profile") != "normal"
        or curriculum.get("weak_prerequisites") != []
        or curriculum.get("mastered") is not mastered
        or not isinstance(passes, Mapping)
        or set(passes) != {lesson.value for lesson in lessons.LessonId}
        or not all(value is True for value in passes.values())
    ):
        raise U2ConfirmationError(
            f"{child_actions} does not satisfy the frozen normal-practice gate"
        )
    return event


def verify_u2_checkpoint(path: Path) -> VerifiedU2Checkpoint:
    """Authenticate one exact first-mastery bundle and its complete frozen history."""

    archive = path.expanduser().resolve()
    selected = next(
        (
            item
            for item in FROZEN_CHECKPOINTS.values()
            if item.archive.expanduser().resolve() == archive
        ),
        None,
    )
    if selected is None:
        raise U2ConfirmationError(
            f"checkpoint is not one of the three frozen U2 archives: {archive}"
        )
    measured = _measure_frozen_artifacts(selected)
    sidecar = _read_json(archive.with_suffix(".json"), "U2 mastery sidecar")
    manifest = _read_json(selected.run_directory / "manifest.json", "U2 manifest")
    try:
        frozen_training._verify_integrity(archive, archive.with_suffix(".json"))
    except frozen_training.U2ProtocolError as error:
        raise U2ConfirmationError(f"U2 bundle integrity failed: {error}") from error

    source = sidecar.get("source")
    progress = sidecar.get("progress")
    curriculum = sidecar.get("curriculum")
    controller = sidecar.get("controller")
    parent = sidecar.get("parent")
    qualification = sidecar.get("qualification")
    if (
        sidecar.get("schema_version") != frozen_training.CHECKPOINT_SCHEMA_VERSION
        or sidecar.get("protocol") != frozen_training.PROTOCOL
        or sidecar.get("kind") != "mastery"
        or sidecar.get("resume_eligible") is not False
        or sidecar.get("checkpoint_sha256") != selected.archive_sha256
        or not isinstance(source, Mapping)
        or source.get("commit") != TRAINING_SOURCE_COMMIT
        or source.get("dirty") is not False
        or not isinstance(progress, Mapping)
        or int(progress.get("child_trained_actions", -1))
        != selected.child_trained_actions
        or int(progress.get("trained_actions", -1))
        != selected.lifetime_trained_actions
        or int(progress.get("optimizer_updates", -1))
        != selected.optimizer_updates
        or not isinstance(curriculum, Mapping)
        or curriculum.get("mastered") is not True
        or int(curriculum.get("consecutive_passes", -1)) != 2
        or curriculum.get("recovery") is not False
        or curriculum.get("recovery_profile") != "normal"
        or curriculum.get("weak_prerequisites") != []
        or not isinstance(controller, Mapping)
        or not isinstance(parent, Mapping)
        or int(parent.get("u2_child_seed", -1)) != selected.child_seed
        or int(parent.get("u1_child_seed", -1)) != selected.u1_parent_seed
        or not isinstance(qualification, Mapping)
        or qualification.get("report_sha256") != QUALIFICATION_SHA256
    ):
        raise U2ConfirmationError(
            f"U2 mastery sidecar contract failed for child {selected.child_seed}"
        )
    first_pass = controller.get("first_pass_artifact")
    mastery = controller.get("mastery_artifact")
    if (
        not isinstance(first_pass, Mapping)
        or first_pass.get("checkpoint_sha256") != selected.first_pass_sha256
        or int(first_pass.get("child_trained_actions", -1))
        != selected.first_pass_actions
        or not isinstance(mastery, Mapping)
        or mastery.get("checkpoint_sha256") != selected.archive_sha256
        or int(mastery.get("child_trained_actions", -1))
        != selected.child_trained_actions
        or selected.child_trained_actions - selected.first_pass_actions
        != frozen_training.EVALUATION_INTERVAL
        or int(controller.get("last_normal_pass_child_actions", -1))
        != selected.child_trained_actions
        or int(controller.get("last_decision_child_actions", -1))
        != selected.child_trained_actions
    ):
        raise U2ConfirmationError(
            f"U2 first-pass/mastery adjacency failed for child {selected.child_seed}"
        )
    first_pass_archive = selected.run_directory / str(first_pass.get("path", ""))
    if (
        first_pass_archive.resolve()
        != selected.run_directory
        / "checkpoints/first-pass-separated-unlock.zip"
        or file_sha256(first_pass_archive) != selected.first_pass_sha256
        or file_sha256(first_pass_archive.with_suffix(".json"))
        != selected.first_pass_sidecar_sha256
        or file_sha256(first_pass_archive.with_suffix(".integrity.json"))
        != selected.first_pass_integrity_sha256
    ):
        raise U2ConfirmationError(
            f"frozen first-pass artifact changed for child {selected.child_seed}"
        )
    try:
        frozen_training._verify_integrity(
            first_pass_archive,
            first_pass_archive.with_suffix(".json"),
        )
    except frozen_training.U2ProtocolError as error:
        raise U2ConfirmationError(f"first-pass bundle integrity failed: {error}") from error

    if (
        manifest.get("schema_version") != frozen_training.CHECKPOINT_SCHEMA_VERSION
        or manifest.get("protocol") != frozen_training.PROTOCOL
        or manifest.get("warm_start") is not True
        or manifest.get("demonstrations") is not False
        or manifest.get("online_model_calls") is not False
        or manifest.get("oracle_actions_used_for_training") is not False
        or manifest.get("source") != dict(source)
        or manifest.get("parent") != dict(parent)
        or manifest.get("qualification") != dict(qualification)
    ):
        raise U2ConfirmationError(
            f"U2 manifest provenance failed for child {selected.child_seed}"
        )
    try:
        verified_parent = frozen_training.verify_parent(
            Path(str(parent.get("checkpoint", ""))),
            Path(str(parent.get("confirmation_report", ""))),
            child_seed=selected.child_seed,
        )
    except (frozen_training.U2ProtocolError, OSError, ValueError) as error:
        raise U2ConfirmationError(
            f"U1 parent provenance failed for U2 child {selected.child_seed}: {error}"
        ) from error
    if verified_parent.public_dict() != dict(parent):
        raise U2ConfirmationError(
            f"U1 parent evidence is not reproducible for U2 child {selected.child_seed}"
        )

    evaluations = _read_jsonl(
        selected.run_directory / "evaluations.jsonl",
        "U2 evaluation history",
    )
    events = _read_jsonl(selected.run_directory / "events.jsonl", "U2 event history")
    _evaluation_records_at(
        evaluations,
        child_actions=selected.first_pass_actions,
        checkpoint_sha256=selected.first_pass_sha256,
    )
    _evaluation_records_at(
        evaluations,
        child_actions=selected.child_trained_actions,
        checkpoint_sha256=selected.archive_sha256,
    )
    _decision_event_at(
        events,
        child_actions=selected.first_pass_actions,
        checkpoint_sha256=selected.first_pass_sha256,
        mastered=False,
    )
    _decision_event_at(
        events,
        child_actions=selected.child_trained_actions,
        checkpoint_sha256=selected.archive_sha256,
        mastered=True,
    )
    final_allocation = sidecar.get("last_completed_allocation")
    scheduler = sidecar.get("scheduler")
    if (
        not isinstance(final_allocation, Mapping)
        or not isinstance(scheduler, Mapping)
        or sidecar.get("last_completed_allocation_valid") is not True
    ):
        raise U2ConfirmationError(
            f"U2 child {selected.child_seed} has no valid final allocation"
        )
    _verify_allocation_history(
        events,
        child_actions=selected.child_trained_actions,
        final_allocation=final_allocation,
        scheduler=scheduler,
    )
    return VerifiedU2Checkpoint(
        frozen=selected,
        sidecar=sidecar,
        manifest=manifest,
        artifact_sha256s=measured,
        policy_member_sha256=_policy_archive_member_sha256(selected.archive),
    )


def verify_frozen_cohort() -> dict[str, Any]:
    if file_sha256(CANONICAL_COHORT_MANIFEST) != COHORT_SHA256:
        raise U2ConfirmationError("completed U2 cohort manifest digest changed")
    cohort = _read_json(CANONICAL_COHORT_MANIFEST, "completed U2 cohort manifest")
    lineages = cohort.get("lineages")
    if (
        cohort.get("phase") != "completed"
        or cohort.get("active_lineage_id") is not None
        or not isinstance(lineages, list)
        or len(lineages) != 3
        or {int(item.get("seed", -1)) for item in lineages if isinstance(item, Mapping)}
        != set(FROZEN_CHECKPOINTS)
        or any(
            not isinstance(item, Mapping) or item.get("state") != "mastered"
            for item in lineages
        )
    ):
        raise U2ConfirmationError("cohort is not the frozen three-lineage terminal result")
    return {
        "path": str(CANONICAL_COHORT_MANIFEST),
        "sha256": COHORT_SHA256,
        "phase": "completed",
        "lineages": [int(item["seed"]) for item in lineages],
    }


def checkpoint_set_sha256(
    verified: Sequence[VerifiedU2Checkpoint],
) -> str:
    identities = [
        {
            "child_seed": item.frozen.child_seed,
            "path": str(item.frozen.archive),
            "sha256": item.frozen.archive_sha256,
            "policy_member_sha256": item.policy_member_sha256,
            "child_trained_actions": item.frozen.child_trained_actions,
        }
        for item in sorted(verified, key=lambda value: value.frozen.child_seed)
    ]
    return _canonical_sha256(identities)


def candidate_seeds(lesson: lessons.LessonId | str) -> range:
    role = CANDIDATE_ROLES[lessons.LessonId(lesson)]
    partition = PARTITION_BY_ROLE[role]
    if partition.stop - partition.start != CANDIDATE_COUNT:
        raise U2ConfirmationError(f"{role.value} is not the frozen 10,000-seed stream")
    return range(partition.start, partition.stop)


def candidate_partition_audit() -> dict[str, Any]:
    partitions = [
        {
            "lesson_id": lesson.value,
            "role": CANDIDATE_ROLES[lesson].value,
            "start": candidate_seeds(lesson).start,
            "end": candidate_seeds(lesson).stop - 1,
            "count": len(candidate_seeds(lesson)),
        }
        for lesson in lessons.LessonId
    ]
    collisions: list[dict[str, Any]] = []
    for index, left in enumerate(partitions):
        for right in partitions[index + 1 :]:
            start = max(int(left["start"]), int(right["start"]))
            end = min(int(left["end"]), int(right["end"]))
            if start <= end:
                collisions.append({"left": left["role"], "right": right["role"]})
    if any(
        int(item["count"]) != CANDIDATE_COUNT
        or int(item["start"]) < 15_200_000
        or int(item["end"]) > 15_239_999
        for item in partitions
    ):
        collisions.append({"error": "candidate streams escaped the frozen reservation"})
    return {
        "partitions": partitions,
        "seed_partition_ledger": list(seed_partition_ledger()),
        "untouched_final_start": 20_000_000,
        "collisions": collisions,
        "passed": not collisions,
    }


def _validation_hashes(
    lesson: lessons.LessonId,
    *,
    validation_access: U2SeedAccess,
) -> frozenset[str]:
    hashes: set[str] = set()
    for seed in lessons.validation_seeds(lesson, access=validation_access):
        env = lessons.U2LessonEnv(lesson=lesson)
        try:
            role = {
                lessons.LessonId.NAVIGATE: U2SeedRole.NAVIGATE_VALIDATION,
                lessons.LessonId.VISIBLE_UNLOCK: U2SeedRole.U0_VALIDATION,
                lessons.LessonId.LOCAL_UNLOCK: U2SeedRole.U1_VALIDATION,
                lessons.LessonId.SEPARATED_UNLOCK: U2SeedRole.U2_VALIDATION,
            }[lesson]
            env.reset(
                seed=seed,
                options={
                    "u2_seed_role": role.value,
                    "u2_seed_access": validation_access,
                },
            )
            if env.layout is None:
                raise U2ConfirmationError("validation reference generated no layout")
            hashes.add(_require_sha256(env.layout.layout_sha256, "validation layout"))
        finally:
            env.close()
    if not hashes:
        raise U2ConfirmationError(
            f"{lesson.value} validation references produced no exact layouts"
        )
    return frozenset(hashes)


def _history_hashes(
    verified: Sequence[VerifiedU2Checkpoint],
) -> tuple[
    dict[lessons.LessonId, frozenset[str]],
    dict[str, Any],
]:
    union: dict[lessons.LessonId, set[str]] = {
        lesson: set() for lesson in lessons.LessonId
    }
    evidence: dict[str, Any] = {}
    for checkpoint in verified:
        path = checkpoint.frozen.run_directory / "episodes.jsonl"
        records = _read_jsonl(path, "frozen U2 episode history")
        per_lesson: dict[str, Any] = {}
        for lesson in lessons.LessonId:
            hashes = {
                _require_sha256(record.get("layout_sha256"), "training layout")
                for record in records
                if record.get("lesson_id") == lesson.value
            }
            union[lesson].update(hashes)
            per_lesson[lesson.value] = _hash_set_evidence(hashes)
        evidence[str(checkpoint.frozen.child_seed)] = {
            "path": str(path),
            "file_sha256": checkpoint.frozen.episodes_sha256,
            "scope": "completed_logged_episodes_only",
            "partial_active_final_layouts_authenticated": False,
            "maximum_unauthenticated_active_layouts": 4,
            "lessons": per_lesson,
        }
    return (
        {lesson: frozenset(values) for lesson, values in union.items()},
        evidence,
    )


def _prior_confirmation_hashes() -> tuple[
    dict[lessons.LessonId, frozenset[str]],
    dict[str, Any],
]:
    if file_sha256(CANONICAL_U1_CONFIRMATION) != U1_CONFIRMATION_SHA256:
        raise U2ConfirmationError("frozen U1 confirmation report digest changed")
    report = _read_json(CANONICAL_U1_CONFIRMATION, "frozen U1 confirmation report")
    if (
        report.get("protocol")
        != "dungeon-apprentice-v0.2-u1-confirmation-v2"
        or report.get("verdict") != "confirmed"
        or report.get("checkpoint_scoring_performed") is not True
        or report.get("policy_updates") is not False
    ):
        raise U2ConfirmationError("prior U1 confirmation is not the frozen result")
    result: dict[lessons.LessonId, frozenset[str]] = {}
    for selection in report.get("layout_qualification", ()):
        if not isinstance(selection, Mapping):
            raise U2ConfirmationError("prior confirmation selection is invalid")
        try:
            lesson = lessons.LessonId(str(selection["lesson_id"]))
        except (KeyError, ValueError) as error:
            raise U2ConfirmationError("prior confirmation lesson is invalid") from error
        cases = selection.get("cases")
        if (
            selection.get("result") != "passed"
            or not isinstance(cases, list)
            or len(cases) != CONFIRMATION_CASES
        ):
            raise U2ConfirmationError("prior confirmation panel is incomplete")
        result[lesson] = frozenset(
            _require_sha256(case.get("layout_sha256"), "prior confirmation layout")
            for case in cases
            if isinstance(case, Mapping)
        )
        if len(cases) != sum(isinstance(case, Mapping) for case in cases):
            raise U2ConfirmationError("prior confirmation contains an invalid case")
    expected = {
        lessons.LessonId.NAVIGATE,
        lessons.LessonId.VISIBLE_UNLOCK,
        lessons.LessonId.LOCAL_UNLOCK,
    }
    if set(result) != expected:
        raise U2ConfirmationError("prior confirmation does not contain all inherited lessons")
    return (
        result,
        {
            "path": str(CANONICAL_U1_CONFIRMATION),
            "sha256": U1_CONFIRMATION_SHA256,
            "lessons": {
                lesson.value: _hash_set_evidence(values)
                for lesson, values in result.items()
            },
        },
    )


def _qualification_hashes(
    qualification_report: Mapping[str, Any],
) -> frozenset[str]:
    cases = qualification_report.get("cases")
    if not isinstance(cases, list) or len(cases) != 2_000:
        raise U2ConfirmationError("sealed U2 qualification does not contain 2,000 cases")
    hashes = frozenset(
        _require_sha256(case.get("layout_sha256"), "sealed qualification layout")
        for case in cases
        if isinstance(case, Mapping)
    )
    if len(cases) != sum(isinstance(case, Mapping) for case in cases):
        raise U2ConfirmationError("sealed U2 qualification contains an invalid case")
    if len(hashes) < 1_980:
        raise U2ConfirmationError("sealed U2 qualification no longer meets its exact floor")
    return hashes


def build_reference_exclusions(
    verified: Sequence[VerifiedU2Checkpoint],
    *,
    validation_access: U2SeedAccess,
    qualification_report: Mapping[str, Any],
) -> tuple[dict[lessons.LessonId, LessonReferenceExclusions], dict[str, Any]]:
    """Build every prospective set before confirmation candidate one is opened."""

    validation = {
        lesson: _validation_hashes(lesson, validation_access=validation_access)
        for lesson in lessons.LessonId
    }
    histories, history_evidence = _history_hashes(verified)
    prior, prior_evidence = _prior_confirmation_hashes()
    sealed = _qualification_hashes(qualification_report)
    exclusions: dict[lessons.LessonId, LessonReferenceExclusions] = {}
    for lesson in lessons.LessonId:
        gating: dict[str, frozenset[str]] = {
            "development_validation": validation[lesson],
        }
        diagnostic: dict[str, frozenset[str]] = {}
        if lesson in {
            lessons.LessonId.NAVIGATE,
            lessons.LessonId.LOCAL_UNLOCK,
            lessons.LessonId.SEPARATED_UNLOCK,
        }:
            gating["u2_child_training_history"] = histories[lesson]
        else:
            diagnostic["u2_child_training_history"] = histories[lesson]
        if lesson in {lessons.LessonId.NAVIGATE, lessons.LessonId.LOCAL_UNLOCK}:
            gating["prior_u1_confirmation"] = prior[lesson]
        elif lesson is lessons.LessonId.VISIBLE_UNLOCK:
            diagnostic["prior_u1_confirmation"] = prior[lesson]
        if lesson is lessons.LessonId.SEPARATED_UNLOCK:
            gating["sealed_u2_qualification"] = sealed
        exclusions[lesson] = LessonReferenceExclusions(
            gating=gating,
            diagnostic=diagnostic,
            sources={
                "validation_seed_sha256": _seed_list_sha256(
                    lessons.validation_seeds(lesson, access=validation_access)
                ),
                "u0_overlap_is_diagnostic": (
                    lesson is lessons.LessonId.VISIBLE_UNLOCK
                ),
            },
        )
    return (
        exclusions,
        {
            "validation": {
                lesson.value: _hash_set_evidence(values)
                for lesson, values in validation.items()
            },
            "u2_child_histories": history_evidence,
            "training_history_limitation": {
                "scope": "completed_logged_episodes_only",
                "unauthenticated_active_final_layouts_across_lineages": 12,
                "four_active_worker_layouts_per_lineage": True,
                "unlogged_final_vector_steps_by_child": {
                    "20260737": 3,
                    "20260741": 5,
                    "20260745": 8,
                },
                "collision_with_unlogged_partial_layouts_can_be_ruled_out": False,
                "u3_requirement": (
                    "persist per-worker episode-start seed, lesson, layout identity, "
                    "and active environment state at every checkpoint"
                ),
            },
            "prior_u1_confirmation": prior_evidence,
            "sealed_u2_qualification": _hash_set_evidence(sealed),
        },
    )


def _inspect_candidate(
    lesson: lessons.LessonId,
    seed: int,
    *,
    confirmation_access: U2SeedAccess,
) -> dict[str, Any]:
    """Qualify one candidate without deserializing or consulting a policy."""

    role = CANDIDATE_ROLES[lesson]
    if lesson is lessons.LessonId.SEPARATED_UNLOCK:
        try:
            evidence = lessons.generate_u2_case_evidence(
                int(seed),
                seed_role=role,
                access=confirmation_access,
                size=9,
            )
        except (AssertionError, OracleFailure, RuntimeError, TypeError, ValueError) as error:
            raise CandidateRejected("u2_mechanics_contract", str(error)) from error
        required_true = (
            "planner_oracle_counts_match",
            "post_key_turn_present",
            "exactly_two_extra_walls",
            "goal_blocked_while_locked",
            "door_requires_matching_key",
            "observation_contract",
            "reward_contract",
            "terminated",
            "success",
            "ordered_objective_completed",
        )
        if any(evidence.get(name) is not True for name in required_true):
            raise CandidateRejected(
                "u2_mechanics_contract",
                "one or more complete Separated Unlock evidence gates failed",
            )
        return {
            "seed": int(seed),
            "seed_role": role.value,
            "lesson_id": lesson.value,
            "layout_sha256": _require_sha256(
                evidence.get("layout_sha256"),
                "candidate exact layout",
            ),
            "geometry_sha256": _require_sha256(
                evidence.get("geometry_sha256"),
                "candidate geometry",
            ),
            "live_oracle_actions": int(evidence["live_oracle_actions"]),
            "pure_oracle_actions": int(evidence["pure_planner_actions"]),
            "planner_oracle_counts_match": True,
            "key_visible": bool(evidence["initial_key_visible"]),
            "door_visible": bool(evidence["initial_door_visible"]),
            "visibility_stratum": evidence["visibility_stratum"],
            "generation_attempts": int(evidence["generation_attempts"]),
            "contracts": {
                "mechanically_solved": True,
                "policy_interface_unchanged": True,
                "reward_unchanged": True,
                "terminal_semantics_unchanged": True,
                "ordered_objective": True,
                "matching_key_required": True,
                "goal_blocked_while_locked": True,
                "post_key_turn_present": True,
                "complete_objective": True,
                "action_length": True,
            },
        }
    wrapped = lessons.make_pixel_env(lesson=lesson, size=9)
    base = wrapped.unwrapped
    try:
        observation, info = wrapped.reset(
            seed=int(seed),
            options={
                "u2_seed_role": role.value,
                "u2_seed_access": confirmation_access,
            },
        )
        if not isinstance(base, lessons.U2LessonEnv) or base.layout is None:
            raise CandidateRejected(
                "environment_contract",
                "candidate did not expose one U2 lesson layout",
            )
        if (
            observation.shape != PIXEL_SHAPE
            or observation.dtype != np.uint8
            or int(wrapped.action_space.n) != 7
        ):
            raise CandidateRejected(
                "policy_interface_contract",
                "candidate changed the pixel or action interface",
            )
        if info.get("lesson_id") != lesson.value:
            raise CandidateRejected("lesson_contract", "candidate has the wrong lesson ID")
        key_visible = (
            bool(base.agent_sees(*base.layout.key))
            if base.layout.key is not None
            else None
        )
        door_visible = (
            bool(base.agent_sees(*base.layout.door))
            if base.layout.door is not None
            else None
        )
        matching_key = True
        if base.layout.door is not None:
            door = base.grid.get(*base.layout.door)
            matching_key = (
                isinstance(door, Door)
                and door.is_locked
                and base.layout.key_color == door.color
            )
            if not matching_key:
                raise CandidateRejected(
                    "matching_key_contract",
                    "unlock lesson did not begin with its matching locked door",
                )
        post_key_turn_present: bool | None = None
        divider_contract: bool | None = None
        if lesson is lessons.LessonId.LOCAL_UNLOCK:
            if (
                base.layout.key is None
                or base.layout.door is None
                or base.layout.key_color is None
            ):
                raise CandidateRejected(
                    "u1_topology_contract",
                    "Local Unlock is missing key or door topology",
                )
            candidate = {
                "start": base.layout.start,
                "start_direction": base.layout.start_direction,
                "goal": base.layout.goal,
                "key": base.layout.key,
                "door": base.layout.door,
                "walls": base.layout.walls,
            }
            try:
                plan = lessons.plan_unlock_candidate(
                    candidate,
                    width=9,
                    height=9,
                )
            except OracleFailure as error:
                raise CandidateRejected("u1_topology_contract", str(error)) from error
            post_key_turn_present = plan.post_key_turns >= 1
            door_x, door_y = base.layout.door
            walls = set(base.layout.walls)
            vertical = (
                len(walls) == 6
                and {x for x, _y in walls} == {door_x}
                and {y for _x, y in walls} | {door_y} == set(range(1, 8))
            )
            horizontal = (
                len(walls) == 6
                and {y for _x, y in walls} == {door_y}
                and {x for x, _y in walls} | {door_x} == set(range(1, 8))
            )
            start_side = (
                base.layout.start[0] < door_x
                if vertical
                else base.layout.start[1] < door_y
            )
            key_side = (
                base.layout.key[0] < door_x
                if vertical
                else base.layout.key[1] < door_y
            )
            goal_side = (
                base.layout.goal[0] < door_x
                if vertical
                else base.layout.goal[1] < door_y
            )
            door_count = sum(
                isinstance(base.grid.get(x, y), Door)
                for x in range(1, 8)
                for y in range(1, 8)
            )
            divider_contract = bool(
                (vertical or horizontal)
                and door_count == 1
                and start_side == key_side
                and goal_side != start_side
                and post_key_turn_present
            )
            if not divider_contract:
                raise CandidateRejected(
                    "u1_topology_contract",
                    "Local Unlock divider, sole-door, side, or post-key contract changed",
                )
        try:
            live = DungeonOracle(base).solve()
        except OracleFailure as error:
            raise CandidateRejected("oracle_contract", str(error)) from error
        spec = lessons.LESSON_SPECS[lesson]
        if not live.success or not 0 < live.steps <= spec.max_steps:
            raise CandidateRejected(
                "oracle_contract",
                "live oracle did not complete within the frozen horizon",
            )
        expected_reward = STEP_REWARD * (live.steps - 1) + SUCCESS_REWARD
        if (
            live.terminal_reason != "success"
            or not math.isclose(
                live.total_reward,
                expected_reward,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise CandidateRejected(
                "terminal_reward_contract",
                "oracle terminal reason or total reward changed",
            )
        if (
            spec.oracle_action_range is not None
            and not spec.oracle_action_range[0]
            <= live.steps
            <= spec.oracle_action_range[1]
        ):
            raise CandidateRejected(
                "action_length_contract",
                f"{live.steps} is outside {spec.oracle_action_range}",
            )
        pure_count = (
            len(base.pure_oracle_actions)
            if base.pure_oracle_actions is not None
            else None
        )
        if lesson in {
            lessons.LessonId.LOCAL_UNLOCK,
            lessons.LessonId.SEPARATED_UNLOCK,
        } and pure_count != live.steps:
            raise CandidateRejected(
                "planner_oracle_contract",
                f"pure/live counts differ: {pure_count}/{live.steps}",
            )
        if spec.key_visible is not None and key_visible is not spec.key_visible:
            raise CandidateRejected("visibility_contract", "key visibility changed")
        if spec.door_visible is not None and door_visible is not spec.door_visible:
            raise CandidateRejected("visibility_contract", "door visibility changed")
        if lesson is lessons.LessonId.SEPARATED_UNLOCK and (
            len(base.extra_walls) != 2
            or base.divider_orientation not in {"horizontal", "vertical"}
            or base.generation_attempts <= 0
        ):
            raise CandidateRejected(
                "separated_geometry_contract",
                "Separated Unlock divider or extra-wall evidence changed",
            )
        return {
            "seed": int(seed),
            "seed_role": role.value,
            "lesson_id": lesson.value,
            "layout_sha256": _require_sha256(
                base.layout.layout_sha256,
                "candidate exact layout",
            ),
            "geometry_sha256": _require_sha256(
                base.geometry_sha256,
                "candidate geometry",
            ),
            "live_oracle_actions": live.steps,
            "pure_oracle_actions": pure_count,
            "planner_oracle_counts_match": (
                pure_count == live.steps if pure_count is not None else None
            ),
            "key_visible": key_visible,
            "door_visible": door_visible,
            "visibility_stratum": info.get("visibility_stratum"),
            "generation_attempts": int(getattr(base, "generation_attempts", 0)),
            "contracts": {
                "mechanically_solved": True,
                "policy_interface_unchanged": True,
                "reward_unchanged": True,
                "terminal_semantics_unchanged": True,
                "ordered_objective": True,
                "matching_key_required": matching_key,
                "divider_contract": divider_contract,
                "post_key_turn_present": post_key_turn_present,
                "complete_objective": True,
                "action_length": True,
            },
        }
    finally:
        wrapped.close()


def _selection_result(
    lesson: lessons.LessonId,
    accepted: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    *,
    examined: int,
    exclusions: LessonReferenceExclusions,
) -> dict[str, Any]:
    accepted_hashes = {
        _require_sha256(item.get("layout_sha256"), "accepted exact layout")
        for item in accepted
    }
    geometry_hashes = {
        _require_sha256(item.get("geometry_sha256"), "accepted geometry")
        for item in accepted
    }
    panels: list[dict[str, Any]] = []
    geometry_gate = GEOMETRY_GATES[lesson]
    for index, start in enumerate((0, CONFIRMATION_PANEL_SIZE)):
        panel = list(accepted[start : start + CONFIRMATION_PANEL_SIZE])
        panel_exact = {str(item["layout_sha256"]) for item in panel}
        panel_geometry = {str(item["geometry_sha256"]) for item in panel}
        panel_visibility = Counter(
            str(item["visibility_stratum"])
            for item in panel
            if item.get("visibility_stratum") is not None
        )
        panel_passed = (
            len(panel) == CONFIRMATION_PANEL_SIZE
            and len(panel_exact) == CONFIRMATION_PANEL_SIZE
            and (
                geometry_gate is None
                or len(panel_geometry) >= geometry_gate[1]
            )
        )
        panels.append(
            {
                "panel": "A" if index == 0 else "B",
                "cases": len(panel),
                "exact_unique_layouts": len(panel_exact),
                "geometry_unique_layouts": len(panel_geometry),
                "required_exact_unique_layouts": CONFIRMATION_PANEL_SIZE,
                "required_geometry_unique_layouts": (
                    geometry_gate[1] if geometry_gate is not None else None
                ),
                "visibility_strata": dict(sorted(panel_visibility.items())),
                "passed": panel_passed,
            }
        )
    overlap = {
        name: {
            "overlap": len(accepted_hashes & values),
            "passed": not (accepted_hashes & values),
        }
        for name, values in exclusions.gating.items()
    }
    diagnostic_overlap = {
        name: len(accepted_hashes & values)
        for name, values in exclusions.diagnostic.items()
    }
    required_geometry = geometry_gate[0] if geometry_gate is not None else None
    passed = (
        len(accepted) == CONFIRMATION_CASES
        and len(accepted_hashes) == CONFIRMATION_CASES
        and all(item["passed"] for item in panels)
        and all(item["passed"] for item in overlap.values())
        and (
            required_geometry is None
            or len(geometry_hashes) >= required_geometry
        )
    )
    reason_counts = Counter(
        reason
        for item in rejected
        for reason in item.get("reasons", [item.get("reason")])
        if isinstance(reason, str)
    )
    visibility = Counter(
        str(item.get("visibility_stratum"))
        for item in accepted
        if item.get("visibility_stratum") is not None
    )
    seeds = [int(item["seed"]) for item in accepted]
    return {
        "lesson_id": lesson.value,
        "candidate_role": CANDIDATE_ROLES[lesson].value,
        "candidate_start": candidate_seeds(lesson).start,
        "candidate_end": candidate_seeds(lesson).stop - 1,
        "candidate_count": CANDIDATE_COUNT,
        "candidates_examined": examined,
        "candidates_unopened": CANDIDATE_COUNT - examined,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_seed_sha256": _seed_list_sha256(seeds),
        "exact_unique_layouts": len(accepted_hashes),
        "required_exact_unique_layouts": CONFIRMATION_CASES,
        "geometry_unique_layouts": len(geometry_hashes),
        "required_geometry_unique_layouts": required_geometry,
        "geometry_affects_verdict": geometry_gate is not None,
        "panel_qualification": panels,
        "gating_reference_overlap": overlap,
        "diagnostic_reference_overlap": diagnostic_overlap,
        "reference_sets": exclusions.public_dict(),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "visibility_strata": dict(sorted(visibility.items())),
        "cases": [dict(item) for item in accepted],
        "rejections": [dict(item) for item in rejected],
        "result": "passed" if passed else "failed",
    }


def select_confirmation_block(
    lesson: lessons.LessonId | str,
    exclusions: LessonReferenceExclusions,
    *,
    confirmation_access: U2SeedAccess,
    journal_directory: Path | None = None,
    journal_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected = lessons.LessonId(lesson)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted_hashes: set[str] = set()
    examined = 0
    for seed in candidate_seeds(selected):
        examined += 1
        journal_stem = f"{examined:05d}-{int(seed)}"
        if journal_directory is not None:
            _publish_immutable_json(
                journal_directory
                / selected.value.replace("/", "__")
                / f"{journal_stem}-opened.json",
                {
                    "schema_version": 1,
                    "protocol": CONFIRMATION_PROTOCOL,
                    "attempt_id": ATTEMPT_ID,
                    "lesson_id": selected.value,
                    "candidate_role": CANDIDATE_ROLES[selected].value,
                    "candidate_index": examined - 1,
                    "seed": int(seed),
                    "status": "opened_before_generation",
                    "opened_at": utc_now(),
                    "base_identity": dict(journal_identity or {}),
                },
            )
        try:
            candidate = _inspect_candidate(
                selected,
                seed,
                confirmation_access=confirmation_access,
            )
        except CandidateRejected as error:
            outcome = {
                "seed": int(seed),
                "status": "rejected",
                "reason": error.code,
                "reasons": [error.code],
                "detail": error.detail,
                "layout_sha256": None,
                "geometry_sha256": None,
                "live_oracle_actions": None,
                "visibility_stratum": None,
            }
            rejected.append(outcome)
            if journal_directory is not None:
                _publish_immutable_json(
                    journal_directory
                    / selected.value.replace("/", "__")
                    / f"{journal_stem}-outcome.json",
                    outcome,
                )
            continue
        except (AssertionError, OracleFailure, RuntimeError, TypeError, ValueError) as error:
            if isinstance(error, U2SeedAccessError):
                raise
            outcome = {
                "seed": int(seed),
                "status": "rejected",
                "reason": "bounded_candidate_generation_error",
                "reasons": ["bounded_candidate_generation_error"],
                "detail": f"{type(error).__name__}: {error}",
                "layout_sha256": None,
                "geometry_sha256": None,
                "live_oracle_actions": None,
                "visibility_stratum": None,
            }
            rejected.append(outcome)
            if journal_directory is not None:
                _publish_immutable_json(
                    journal_directory
                    / selected.value.replace("/", "__")
                    / f"{journal_stem}-outcome.json",
                    outcome,
                )
            continue
        except BaseException as error:
            if journal_directory is not None:
                _publish_immutable_json(
                    journal_directory
                    / selected.value.replace("/", "__")
                    / f"{journal_stem}-outcome.json",
                    {
                        "seed": int(seed),
                        "status": "inspection_failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
            raise
        layout_hash = str(candidate["layout_sha256"])
        reasons = [
            f"{name}_exact_overlap"
            for name, values in exclusions.gating.items()
            if layout_hash in values
        ]
        if layout_hash in accepted_hashes:
            reasons.append("duplicate_accepted_exact_layout")
        if reasons:
            outcome = dict(candidate) | {
                "status": "rejected",
                "reason": reasons[0],
                "reasons": reasons,
            }
            rejected.append(outcome)
            if journal_directory is not None:
                _publish_immutable_json(
                    journal_directory
                    / selected.value.replace("/", "__")
                    / f"{journal_stem}-outcome.json",
                    outcome,
                )
            continue
        index = len(accepted)
        outcome = dict(candidate) | {
            "status": "accepted",
            "accepted_index": index,
            "panel": "A" if index < CONFIRMATION_PANEL_SIZE else "B",
        }
        accepted.append(outcome)
        if journal_directory is not None:
            _publish_immutable_json(
                journal_directory
                / selected.value.replace("/", "__")
                / f"{journal_stem}-outcome.json",
                outcome,
            )
        accepted_hashes.add(layout_hash)
        if len(accepted) == CONFIRMATION_CASES:
            break
    result = _selection_result(
        selected,
        accepted,
        rejected,
        examined=examined,
        exclusions=exclusions,
    )
    if journal_directory is not None:
        _publish_immutable_json(
            journal_directory
            / selected.value.replace("/", "__")
            / "selection-summary.json",
            result,
        )
    return result


def selected_seed_lists(
    selections: Sequence[Mapping[str, Any]],
) -> dict[lessons.LessonId, tuple[int, ...]]:
    result: dict[lessons.LessonId, tuple[int, ...]] = {}
    for selection in selections:
        lesson = lessons.LessonId(str(selection["lesson_id"]))
        if lesson in result:
            raise U2ConfirmationError(f"selection repeats {lesson.value}")
        cases = selection.get("cases")
        if selection.get("result") != "passed" or not isinstance(cases, list):
            raise U2ConfirmationError(f"cannot score failed selection for {lesson.value}")
        seeds = tuple(int(item["seed"]) for item in cases)
        if (
            len(seeds) != CONFIRMATION_CASES
            or len(set(seeds)) != CONFIRMATION_CASES
            or _seed_list_sha256(seeds) != selection.get("accepted_seed_sha256")
        ):
            raise U2ConfirmationError(f"selected seed evidence changed for {lesson.value}")
        result[lesson] = seeds
    if set(result) != set(lessons.LessonId):
        raise U2ConfirmationError("selected seed evidence is missing a lesson")
    return result


def verify_selected_layout_identities(
    selections: Sequence[Mapping[str, Any]],
    *,
    confirmation_access: U2SeedAccess,
) -> dict[str, Any]:
    """Regenerate selected cases policy-blind and bind scoring to selected layouts."""

    evidence: dict[str, Any] = {}
    for selection in selections:
        lesson = lessons.LessonId(str(selection["lesson_id"]))
        cases = selection.get("cases")
        if not isinstance(cases, list) or len(cases) != CONFIRMATION_CASES:
            raise U2ConfirmationError("cannot reverify an incomplete selection")
        identities: list[dict[str, Any]] = []
        for case in cases:
            if not isinstance(case, Mapping):
                raise U2ConfirmationError("selected case identity is malformed")
            regenerated = _inspect_candidate(
                lesson,
                int(case["seed"]),
                confirmation_access=confirmation_access,
            )
            if (
                regenerated["layout_sha256"] != case.get("layout_sha256")
                or regenerated["geometry_sha256"] != case.get("geometry_sha256")
            ):
                raise U2ConfirmationError(
                    f"selected layout identity changed for {lesson.value}/{case['seed']}"
                )
            identities.append(
                {
                    "seed": int(case["seed"]),
                    "layout_sha256": str(case["layout_sha256"]),
                    "geometry_sha256": str(case["geometry_sha256"]),
                }
            )
        evidence[lesson.value] = {
            "cases": len(identities),
            "identity_sha256": _canonical_sha256(identities),
            "passed": True,
        }
    if set(evidence) != {lesson.value for lesson in lessons.LessonId}:
        raise U2ConfirmationError("selected layout revalidation missed a lesson")
    return evidence


def grade_confirmation_evaluation(
    result: lessons.U2LessonEvaluation,
) -> dict[str, Any]:
    try:
        lesson = lessons.LessonId(result.lesson_id)
    except ValueError as error:
        raise U2ConfirmationError("confirmation evaluation has an unknown lesson") from error
    if result.protocol != lessons.PROTOCOL:
        raise U2ConfirmationError("confirmation evaluation protocol changed")
    overall_required, panel_required = CAPABILITY_GATES[lesson]
    counts_valid = (
        result.episodes == CONFIRMATION_CASES
        and type(result.successes) is int
        and 0 <= result.successes <= result.episodes
        and len(result.panel_successes) == 2
        and all(
            type(value) is int and 0 <= value <= CONFIRMATION_PANEL_SIZE
            for value in result.panel_successes
        )
        and sum(result.panel_successes) == result.successes
        and len(result.panel_success_rates) == 2
        and math.isclose(
            result.success_rate,
            result.successes / result.episodes,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and all(
            math.isclose(
                rate,
                count / CONFIRMATION_PANEL_SIZE,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for count, rate in zip(
                result.panel_successes,
                result.panel_success_rates,
                strict=True,
            )
        )
    )
    if not counts_valid:
        raise U2ConfirmationError(
            f"{lesson.value} confirmation counts or rates are inconsistent"
        )
    passed = (
        result.successes >= overall_required
        and len(result.panel_successes) == 2
        and all(value >= panel_required for value in result.panel_successes)
    )
    return {
        "lesson_id": lesson.value,
        "overall_required": overall_required,
        "panel_required": panel_required,
        "overall_passed": result.successes >= overall_required,
        "panel_passed": [
            value >= panel_required for value in result.panel_successes
        ],
        "passed": passed,
    }


def _evaluate_checkpoint(
    verified: VerifiedU2Checkpoint,
    model_type: Any,
    seeds_by_lesson: Mapping[lessons.LessonId, Sequence[int]],
    *,
    confirmation_access: U2SeedAccess,
) -> dict[str, Any]:
    """Score one exact archive and prove every learned byte remains unchanged."""

    before_files = _measure_frozen_artifacts(verified.frozen)
    policy_member_before = _policy_archive_member_sha256(verified.frozen.archive)
    if (
        verified.policy_member_sha256
        and policy_member_before != verified.policy_member_sha256
    ):
        raise U2ConfirmationError(
            f"policy member identity changed for child {verified.frozen.child_seed}"
        )
    model = model_type.load(str(verified.frozen.archive), device="cpu")
    model_integrity._verify_loaded_model_contract(model)
    policy_before = model_integrity.policy_tensor_sha256(model)
    optimizer_before = model_integrity.optimizer_state_sha256(model)
    timesteps_before = int(model.num_timesteps)
    updates_before = int(model._n_updates)
    if (
        timesteps_before != verified.frozen.lifetime_trained_actions
        or updates_before != verified.frozen.optimizer_updates
    ):
        raise U2ConfirmationError(
            f"loaded counters changed for U2 child {verified.frozen.child_seed}"
        )

    import torch

    evaluations: list[dict[str, Any]] = []
    with torch.inference_mode():
        for lesson in lessons.LessonId:
            seeds = tuple(int(seed) for seed in seeds_by_lesson[lesson])
            if len(seeds) != CONFIRMATION_CASES:
                raise U2ConfirmationError(
                    f"{lesson.value} does not contain {CONFIRMATION_CASES} seeds"
                )
            result = lessons.evaluate_lesson(
                model,
                lesson,
                seeds,
                size=9,
                seed_access=confirmation_access,
            )
            evaluations.append(
                result.public_dict()
                | {
                    "seeds": list(seeds),
                    "seed_list_sha256": _seed_list_sha256(seeds),
                    "gate": grade_confirmation_evaluation(result),
                }
            )

    policy_after = model_integrity.policy_tensor_sha256(model)
    optimizer_after = model_integrity.optimizer_state_sha256(model)
    timesteps_after = int(model.num_timesteps)
    updates_after = int(model._n_updates)
    after_files = _measure_frozen_artifacts(verified.frozen)
    policy_member_after = _policy_archive_member_sha256(verified.frozen.archive)
    if (
        policy_before != policy_after
        or optimizer_before != optimizer_after
        or timesteps_before != timesteps_after
        or updates_before != updates_after
        or before_files != after_files
        or policy_member_before != policy_member_after
    ):
        raise U2ConfirmationError(
            f"confirmation changed learned state for child {verified.frozen.child_seed}"
        )
    return {
        "checkpoint": verified.public_dict(),
        "policy_tensor_sha256_before": policy_before,
        "policy_tensor_sha256_after": policy_after,
        "policy_archive_member_sha256_before": policy_member_before,
        "policy_archive_member_sha256_after": policy_member_after,
        "optimizer_state_sha256_before": optimizer_before,
        "optimizer_state_sha256_after": optimizer_after,
        "model_num_timesteps_before": timesteps_before,
        "model_num_timesteps_after": timesteps_after,
        "model_updates_before": updates_before,
        "model_updates_after": updates_after,
        "artifact_sha256s_before": before_files,
        "artifact_sha256s_after": after_files,
        "policy_updates": False,
        "evaluation_settings": {
            "device": "cpu",
            "environment_size": 9,
            "deterministic_actions": True,
            "recurrent_state_reset_each_case": True,
            "curiosity_enabled": False,
            "inference_mode": True,
        },
        "evaluations": evaluations,
        "passed": all(item["gate"]["passed"] for item in evaluations),
    }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_immutable_bytes(path: Path, payload: bytes) -> None:
    """Publish bytes once; an identical retry is allowed but replacement is not."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise U2ConfirmationError(f"immutable evidence already differs: {path}")
        return
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.pending"
    )
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise U2ConfirmationError(
                    f"concurrent immutable evidence differs: {path}"
                ) from None
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    if path.read_bytes() != payload:
        raise U2ConfirmationError(f"immutable evidence did not persist exactly: {path}")


def _publish_immutable_json(path: Path, value: Any) -> None:
    _publish_immutable_bytes(path, _json_bytes(value))


def _publish_terminal_bundle(
    directory: Path,
    *,
    report_bytes: bytes,
    checksum_bytes: bytes,
    terminal_bytes: bytes,
) -> Path:
    """Atomically publish the three-file terminal authority on one filesystem."""

    bundle = directory / "terminal-bundle"
    expected = {
        "report.json": report_bytes,
        "report.json.sha256": checksum_bytes,
        "terminal.json": terminal_bytes,
    }
    if bundle.exists():
        if not bundle.is_dir() or any(
            not (bundle / name).is_file()
            or (bundle / name).read_bytes() != payload
            for name, payload in expected.items()
        ):
            raise U2ConfirmationError("immutable terminal bundle already differs")
        return bundle
    staging = directory / (
        f".terminal-bundle.{os.getpid()}.{secrets.token_hex(8)}.pending"
    )
    staging.mkdir(mode=0o700)
    try:
        for name, payload in expected.items():
            path = staging / name
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        _sync_directory(staging)
        try:
            os.rename(staging, bundle)
        except FileExistsError:
            if not bundle.is_dir() or any(
                not (bundle / name).is_file()
                or (bundle / name).read_bytes() != payload
                for name, payload in expected.items()
            ):
                raise U2ConfirmationError(
                    "concurrent terminal bundle already differs"
                ) from None
        _sync_directory(directory)
    finally:
        if staging.exists():
            for child in staging.rglob("*"):
                if child.is_file():
                    child.unlink()
            for child in sorted(
                (path for path in staging.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                child.rmdir()
            staging.rmdir()
    return bundle


def finalize_attempt(
    directory: Path,
    report: Mapping[str, Any],
    *,
    evaluator_exit_status: int,
) -> dict[str, Any]:
    """Publish one immutable terminal bundle and reconcile its mutable ledger view."""

    report_path = directory / "report.json"
    report_bytes = _json_bytes(dict(report))
    digest = hashlib.sha256(report_bytes).hexdigest()
    byte_length = len(report_bytes)
    checksum_path = directory / "report.json.sha256"
    terminal = {
        "schema_version": 1,
        "protocol": f"{CONFIRMATION_PROTOCOL}-terminal-bundle",
        "attempt_id": ATTEMPT_ID,
        "evaluator_exit_status": int(evaluator_exit_status),
        "report": str(report_path),
        "report_sha256": digest,
        "report_byte_length": byte_length,
        "report_verdict": report.get("verdict"),
        "checksum": str(checksum_path),
    }
    terminal_bytes = _json_bytes(terminal)
    checksum_bytes = f"{digest}  report.json\n".encode("ascii")
    bundle = _publish_terminal_bundle(
        directory,
        report_bytes=report_bytes,
        checksum_bytes=checksum_bytes,
        terminal_bytes=terminal_bytes,
    )
    # Compatibility copies are immutable views; the atomically published bundle
    # above remains authoritative if interruption occurs during this projection.
    _publish_immutable_bytes(report_path, report_bytes)
    _publish_immutable_bytes(checksum_path, checksum_bytes)
    _publish_immutable_bytes(directory / "terminal.json", terminal_bytes)
    initial = _read_json(directory / "attempt.json", "U2 confirmation attempt")
    if initial.get("status") == "completed_with_report" and any(
        (
            initial.get("report_sha256") != digest,
            initial.get("report_byte_length") != byte_length,
            initial.get("report_verdict") != report.get("verdict"),
            initial.get("evaluator_exit_status") != int(evaluator_exit_status),
        )
    ):
        raise U2ConfirmationError("terminal attempt ledger cannot be replaced")
    final = dict(initial) | {
        "updated_at": utc_now(),
        "status": "completed_with_report",
        "evaluator_exit_status": int(evaluator_exit_status),
        "report_present": True,
        "report": str(report_path),
        "report_sha256": digest,
        "report_byte_length": byte_length,
        "report_verdict": report.get("verdict"),
        "checkpoint_scoring_performed": report.get(
            "checkpoint_scoring_performed",
            False,
        ),
    }
    atomic_write_json(directory / "attempt.json", final)
    return {
        "report": str(report_path),
        "report_sha256": digest,
        "report_byte_length": byte_length,
        "checksum": str(checksum_path),
        "terminal": str(directory / "terminal.json"),
        "terminal_bundle": str(bundle),
        "attempt": str(directory / "attempt.json"),
    }


def _claim_attempt(
    *,
    source: Mapping[str, Any],
    plan_path: Path,
    plan_sha256: str,
    checkpoint_set_digest: str,
    verified: Sequence[VerifiedU2Checkpoint],
    confirmation_anchor: Mapping[str, Any],
) -> tuple[U2SeedAccess, dict[str, Any]]:
    if CANONICAL_ATTEMPT_DIRECTORY.exists():
        raise U2ConfirmationError(
            f"refusing to reuse canonical confirmation attempt: "
            f"{CANONICAL_ATTEMPT_DIRECTORY}"
        )
    if not CANONICAL_ATTEMPT_DIRECTORY.parent.is_dir():
        raise U2ConfirmationError(
            f"missing canonical confirmation parent: "
            f"{CANONICAL_ATTEMPT_DIRECTORY.parent}"
        )
    launcher_token = secrets.token_hex(32)
    claim_id = secrets.token_hex(32)
    token_sha256 = hashlib.sha256(launcher_token.encode("ascii")).hexdigest()
    created_at = utc_now()
    claim_body = {
        "schema_version": 1,
        "protocol": CONFIRMATION_PROTOCOL,
        "attempt_id": ATTEMPT_ID,
        "source_commit": source["commit"],
        "source_dirty": source["dirty"],
        "plan": str(plan_path),
        "plan_sha256": plan_sha256,
        "checkpoint_set_sha256": checkpoint_set_digest,
        "external_preregistration": dict(confirmation_anchor),
        "claim_id": claim_id,
        "launcher_token_sha256": token_sha256,
        "created_at": created_at,
    }
    claim_sha256 = _canonical_sha256(claim_body)
    claim_record = claim_body | {"claim_sha256": claim_sha256}
    attempt = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "protocol": ATTEMPT_PROTOCOL,
        "attempt_id": ATTEMPT_ID,
        "status": "claimed_before_candidate_access",
        "created_at": created_at,
        "updated_at": created_at,
        "source_commit": source["commit"],
        "source_dirty": source["dirty"],
        "plan": str(plan_path),
        "plan_sha256": plan_sha256,
        "checkpoint_set_sha256": checkpoint_set_digest,
        "external_preregistration": dict(confirmation_anchor),
        "checkpoints": [
            str(item.frozen.archive)
            for item in sorted(verified, key=lambda value: value.frozen.child_seed)
        ],
        "claim": str(CANONICAL_ATTEMPT_DIRECTORY / "launch-claim.json"),
        "claim_id": claim_id,
        "claim_sha256": claim_sha256,
        "report_present": False,
        "checkpoint_scoring_performed": False,
        "candidate_streams_opened": False,
    }
    staging = CANONICAL_ATTEMPT_DIRECTORY.parent / (
        f".{CANONICAL_ATTEMPT_DIRECTORY.name}."
        f"{os.getpid()}.{secrets.token_hex(8)}.claim-pending"
    )
    staging.mkdir(mode=0o700)
    try:
        _publish_immutable_json(
            staging / "claim-opened.json",
            {
                "schema_version": 1,
                "protocol": f"{CONFIRMATION_PROTOCOL}-claim-opened",
                "attempt_id": ATTEMPT_ID,
                "claim_id": claim_id,
                "claim_sha256": claim_sha256,
                "created_at": created_at,
            },
        )
        _publish_immutable_json(staging / "launch-claim.json", claim_record)
        _publish_immutable_json(staging / "attempt.json", attempt)
        for lesson in lessons.LessonId:
            journal = (
                staging
                / "selection-journal"
                / lesson.value.replace("/", "__")
            )
            journal.mkdir(parents=True)
            _sync_directory(journal)
        _sync_directory(staging / "selection-journal")
        _sync_directory(staging)
        os.rename(staging, CANONICAL_ATTEMPT_DIRECTORY)
        _sync_directory(CANONICAL_ATTEMPT_DIRECTORY.parent)
    finally:
        if staging.exists():
            for child in staging.rglob("*"):
                if child.is_file():
                    child.unlink()
            for child in sorted(
                (path for path in staging.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                child.rmdir()
            staging.rmdir()
    persisted_claim = _read_json(
        CANONICAL_ATTEMPT_DIRECTORY / "launch-claim.json",
        "persisted U2 confirmation claim",
    )
    persisted_attempt = _read_json(
        CANONICAL_ATTEMPT_DIRECTORY / "attempt.json",
        "persisted U2 confirmation attempt",
    )
    if (
        persisted_claim != claim_record
        or persisted_attempt != attempt
        or _canonical_sha256(
            {key: persisted_claim[key] for key in claim_body}
        )
        != claim_sha256
    ):
        raise U2ConfirmationError("confirmation attempt claim did not persist exactly")
    opaque = _new_confirmation_launch_claim(
        source_commit=str(source["commit"]),
        clean_source=source.get("dirty") is False,
        protocol=CONFIRMATION_PROTOCOL,
        plan_sha256=plan_sha256,
        checkpoint_set_sha256=checkpoint_set_digest,
        anchor_tag=str(confirmation_anchor["tag"]),
        anchor_tag_object=str(confirmation_anchor["tag_object"]),
        anchor_remote_url=str(confirmation_anchor["remote_url"]),
        attempt_id=ATTEMPT_ID,
        claim_id=claim_id,
        claim_sha256=claim_sha256,
        launcher_token_sha256=token_sha256,
    )
    access = _post_training_confirmation_seed_access(
        claim=opaque,
        launcher_token=launcher_token,
    )
    return access, attempt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acknowledgement",
        required=True,
        help=f"must equal: {ACKNOWLEDGEMENT}",
    )
    return parser


def _verified_on_disk_claim(directory: Path) -> dict[str, Any] | None:
    """Return the durable attempt view only when all immutable claim links agree."""

    try:
        opened = _read_json(directory / "claim-opened.json", "claim-opened authority")
        claim = _read_json(directory / "launch-claim.json", "launch claim")
        attempt = _read_json(directory / "attempt.json", "attempt ledger")
    except U2ConfirmationError:
        return None
    body = {key: value for key, value in claim.items() if key != "claim_sha256"}
    if (
        opened.get("protocol") != f"{CONFIRMATION_PROTOCOL}-claim-opened"
        or opened.get("attempt_id") != ATTEMPT_ID
        or claim.get("protocol") != CONFIRMATION_PROTOCOL
        or claim.get("attempt_id") != ATTEMPT_ID
        or attempt.get("protocol") != ATTEMPT_PROTOCOL
        or attempt.get("attempt_id") != ATTEMPT_ID
        or not isinstance(claim.get("claim_sha256"), str)
        or _canonical_sha256(body) != claim["claim_sha256"]
        or opened.get("claim_id") != claim.get("claim_id")
        or opened.get("claim_sha256") != claim.get("claim_sha256")
        or attempt.get("claim_id") != claim.get("claim_id")
        or attempt.get("claim_sha256") != claim.get("claim_sha256")
    ):
        return None
    return attempt


def _authoritative_terminal(
    directory: Path,
) -> tuple[dict[str, Any], int] | None:
    bundle = directory / "terminal-bundle"
    if not bundle.exists():
        return None
    report_path = bundle / "report.json"
    checksum_path = bundle / "report.json.sha256"
    terminal_path = bundle / "terminal.json"
    report = _read_json(report_path, "terminal bundle report")
    terminal = _read_json(terminal_path, "terminal bundle identity")
    digest = file_sha256(report_path)
    try:
        checksum = checksum_path.read_text(encoding="ascii").split()[0]
    except (OSError, IndexError) as error:
        raise U2ConfirmationError("terminal bundle checksum is unreadable") from error
    if (
        terminal.get("protocol") != f"{CONFIRMATION_PROTOCOL}-terminal-bundle"
        or terminal.get("attempt_id") != ATTEMPT_ID
        or terminal.get("report_sha256") != digest
        or terminal.get("report_byte_length") != report_path.stat().st_size
        or terminal.get("report_verdict") != report.get("verdict")
        or checksum != digest
    ):
        raise U2ConfirmationError("authoritative terminal bundle is inconsistent")
    return report, int(terminal["evaluator_exit_status"])


def _selection_journal_evidence(directory: Path) -> dict[str, Any]:
    if not directory.exists():
        return {
            "path": str(directory),
            "files": 0,
            "opened_records": 0,
            "file_set_sha256": _canonical_sha256([]),
        }
    files = sorted(path for path in directory.rglob("*.json") if path.is_file())
    identities = [
        {
            "path": str(path.relative_to(directory)),
            "sha256": file_sha256(path),
            "byte_length": path.stat().st_size,
        }
        for path in files
    ]
    return {
        "path": str(directory),
        "files": len(files),
        "opened_records": sum(path.name.endswith("-opened.json") for path in files),
        "file_set_sha256": _canonical_sha256(identities),
        "identities": identities,
    }


def _global_reference_sha256s(plan_path: Path) -> dict[str, str]:
    return {
        "plan": file_sha256(plan_path),
        "cohort_manifest": file_sha256(CANONICAL_COHORT_MANIFEST),
        "u2_qualification": file_sha256(CANONICAL_QUALIFICATION),
        "u1_confirmation": file_sha256(CANONICAL_U1_CONFIRMATION),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.acknowledgement != ACKNOWLEDGEMENT:
        raise SystemExit("exact U2 confirmation acknowledgement is required")
    repository = Path(__file__).resolve().parents[2]
    plan_path = repository / "docs/v0.2-u2-confirmation-plan.md"
    attempt_parent = CANONICAL_ATTEMPT_DIRECTORY.parent
    if not attempt_parent.is_dir():
        raise SystemExit(f"missing canonical confirmation parent: {attempt_parent}")
    lock_handle = (attempt_parent / ".v0.2-u2-confirmation.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit("another U2 confirmation launcher holds the attempt lock") from error

    wall_started = time.monotonic()
    started_at = utc_now()
    report: dict[str, Any] | None = None
    base_report: dict[str, Any] = {}
    selections: list[dict[str, Any]] = []
    checkpoint_results: list[dict[str, Any]] = []
    scoring_started = False
    journal_directory = CANONICAL_ATTEMPT_DIRECTORY / "selection-journal"

    if CANONICAL_ATTEMPT_DIRECTORY.exists():
        authoritative = _authoritative_terminal(CANONICAL_ATTEMPT_DIRECTORY)
        if authoritative is not None:
            existing_report, existing_status = authoritative
            identity = finalize_attempt(
                CANONICAL_ATTEMPT_DIRECTORY,
                existing_report,
                evaluator_exit_status=existing_status,
            )
            sys.stdout.write(
                f"recovered_{existing_report['verdict']}: {identity['report']}\n"
            )
            if existing_status:
                raise SystemExit(existing_status)
            return
        existing_attempt = _verified_on_disk_claim(CANONICAL_ATTEMPT_DIRECTORY)
        if existing_attempt is None:
            raise SystemExit(
                "canonical U2 confirmation directory is malformed; refusing access"
            )
        interrupted = {
            "schema_version": CONFIRMATION_SCHEMA_VERSION,
            "protocol": CONFIRMATION_PROTOCOL,
            "attempt_id": ATTEMPT_ID,
            "started_at": existing_attempt.get("created_at"),
            "completed_at": utc_now(),
            "elapsed_seconds": None,
            "source": {
                "commit": existing_attempt.get("source_commit"),
                "dirty": existing_attempt.get("source_dirty"),
            },
            "plan": {
                "path": existing_attempt.get("plan"),
                "sha256": existing_attempt.get("plan_sha256"),
            },
            "checkpoint_set_sha256": existing_attempt.get(
                "checkpoint_set_sha256"
            ),
            "external_preregistration": existing_attempt.get(
                "external_preregistration"
            ),
            "claim": existing_attempt,
            "selection_journal": _selection_journal_evidence(journal_directory),
            "checkpoint_scoring_performed": False,
            "checkpoints": [],
            "policy_updates": False,
            "online_model_calls": False,
            "verdict": "interrupted",
            "error": {
                "type": "StartupRecovery",
                "message": (
                    "a durable claim existed without a terminal bundle; "
                    "sealed without minting candidate access"
                ),
            },
        }
        identity = finalize_attempt(
            CANONICAL_ATTEMPT_DIRECTORY,
            interrupted,
            evaluator_exit_status=1,
        )
        raise SystemExit(f"interrupted: {identity['report']}")

    try:
        source = git_snapshot(repository)
        if source.get("dirty") is not False:
            raise U2ConfirmationError(
                "U2 confirmation requires a clean committed source tree"
            )
        if not plan_path.is_file():
            raise U2ConfirmationError(f"missing frozen confirmation plan: {plan_path}")
        plan_sha256 = file_sha256(plan_path)
        partition_audit = candidate_partition_audit()
        if not partition_audit["passed"]:
            raise U2ConfirmationError(
                f"candidate partition audit failed: {partition_audit['collisions']}"
            )
        cohort = verify_frozen_cohort()
        verified = [
            verify_u2_checkpoint(FROZEN_CHECKPOINTS[seed].archive)
            for seed in sorted(FROZEN_CHECKPOINTS)
        ]
        policy_member_digests = [
            item.policy_member_sha256 for item in verified
        ]
        if (
            any(not digest for digest in policy_member_digests)
            or len(set(policy_member_digests)) != len(verified)
        ):
            raise U2ConfirmationError(
                "selected U2 archives do not contain three distinct policy members"
            )
        checkpoint_digest = checkpoint_set_sha256(verified)
        anchor = verify_qualification_anchor(
            repository,
            expected_source_commit=TRAINING_SOURCE_COMMIT,
        )
        qualification = frozen_training.verify_qualification(
            CANONICAL_QUALIFICATION,
            expected_source_commit=TRAINING_SOURCE_COMMIT,
            anchor=anchor,
        )
        if qualification.report_sha256 != QUALIFICATION_SHA256:
            raise U2ConfirmationError("U2 qualification identity changed")
        qualification_report = qualification.verified_report()
        exclusions, reference_evidence = build_reference_exclusions(
            verified,
            validation_access=qualification.seed_access(),
            qualification_report=qualification_report,
        )
        confirmation_anchor = publish_external_anchor(
            repository,
            source_commit=str(source["commit"]),
            plan_sha256=plan_sha256,
            checkpoint_set_sha256=checkpoint_digest,
        )
        confirmation_anchor_public = confirmation_anchor.public_dict()
        global_references_before = _global_reference_sha256s(plan_path)
        base_report = {
            "schema_version": CONFIRMATION_SCHEMA_VERSION,
            "protocol": CONFIRMATION_PROTOCOL,
            "attempt_id": ATTEMPT_ID,
            "started_at": started_at,
            "source": source,
            "runtime": runtime_snapshot(),
            "plan": {
                "path": str(plan_path),
                "sha256": plan_sha256,
            },
            "cohort": cohort,
            "qualification": qualification.public_dict(),
            "checkpoint_set_sha256": checkpoint_digest,
            "external_preregistration": confirmation_anchor_public,
            "selected_checkpoints": [item.public_dict() for item in verified],
            "pre_access_policy_member_sha256s": policy_member_digests,
            "candidate_partition_audit": partition_audit,
            "reference_exclusions": reference_evidence,
            "global_reference_sha256s_before": global_references_before,
            "policy_updates": False,
            "online_model_calls": False,
            "action_selection": "deterministic",
            "recurrent_state_reset_each_case": True,
            "interruption": None,
        }
        confirmation_access, attempt = _claim_attempt(
            source=source,
            plan_path=plan_path,
            plan_sha256=plan_sha256,
            checkpoint_set_digest=checkpoint_digest,
            verified=verified,
            confirmation_anchor=confirmation_anchor_public,
        )
        journal_identity = {
            "attempt_id": ATTEMPT_ID,
            "claim_id": attempt["claim_id"],
            "claim_sha256": attempt["claim_sha256"],
            "source_commit": source["commit"],
            "plan_sha256": plan_sha256,
            "checkpoint_set_sha256": checkpoint_digest,
            "anchor_tag_object": confirmation_anchor_public["tag_object"],
            "reference_exclusions_sha256": _canonical_sha256(reference_evidence),
        }
        _publish_immutable_json(
            journal_directory / "selection-session.json",
            {
                "schema_version": 1,
                "protocol": CONFIRMATION_PROTOCOL,
                "status": "candidate_access_about_to_open",
                "identity": journal_identity,
            },
        )
        atomic_write_json(
            CANONICAL_ATTEMPT_DIRECTORY / "attempt.json",
            attempt
            | {
                "updated_at": utc_now(),
                "candidate_streams_opened": True,
            },
        )
        for lesson in lessons.LessonId:
            selections.append(
                select_confirmation_block(
                    lesson,
                    exclusions[lesson],
                    confirmation_access=confirmation_access,
                    journal_directory=journal_directory,
                    journal_identity=journal_identity,
                )
            )
        if not all(item["result"] == "passed" for item in selections):
            report = base_report | {
                "completed_at": utc_now(),
                "elapsed_seconds": time.monotonic() - wall_started,
                "layout_qualification": selections,
                "selection_journal": _selection_journal_evidence(journal_directory),
                "checkpoint_scoring_performed": False,
                "checkpoints": [],
                "verdict": "qualification_failed",
            }
            identity = finalize_attempt(
                CANONICAL_ATTEMPT_DIRECTORY,
                report,
                evaluator_exit_status=1,
            )
            sys.stdout.write(f"qualification_failed: {identity['report']}\n")
            raise SystemExit(1)
        seeds_by_lesson = selected_seed_lists(selections)
        selected_identity_recheck = verify_selected_layout_identities(
            selections,
            confirmation_access=confirmation_access,
        )
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError as error:
            raise U2ConfirmationError(
                'install training dependencies with: pip install -e ".[train]"'
            ) from error
        scoring_started = True
        for item in verified:
            checkpoint_results.append(
                _evaluate_checkpoint(
                    item,
                    RecurrentPPO,
                    seeds_by_lesson,
                    confirmation_access=confirmation_access,
                )
            )
        if len(
            {item["policy_tensor_sha256_before"] for item in checkpoint_results}
        ) != len(checkpoint_results):
            raise U2ConfirmationError(
                "selected U2 checkpoints do not contain three distinct policies"
            )
        source_after = git_snapshot(repository)
        if source_after != source:
            raise U2ConfirmationError("source identity changed during confirmation")
        global_references_after = _global_reference_sha256s(plan_path)
        if global_references_after != global_references_before:
            raise U2ConfirmationError(
                "a frozen global reference changed during confirmation"
            )
        report = base_report | {
            "completed_at": utc_now(),
            "elapsed_seconds": time.monotonic() - wall_started,
            "source_after": source_after,
            "global_reference_sha256s_after": global_references_after,
            "layout_qualification": selections,
            "selected_layout_identity_recheck": selected_identity_recheck,
            "selection_journal": _selection_journal_evidence(journal_directory),
            "checkpoint_scoring_performed": True,
            "checkpoints": checkpoint_results,
            "verdict": (
                "confirmed"
                if all(item["passed"] for item in checkpoint_results)
                else "capability_failed"
            ),
        }
        exit_status = 0 if report["verdict"] == "confirmed" else 1
        identity = finalize_attempt(
            CANONICAL_ATTEMPT_DIRECTORY,
            report,
            evaluator_exit_status=exit_status,
        )
    except SystemExit:
        raise
    except BaseException as error:
        if CANONICAL_ATTEMPT_DIRECTORY.exists():
            authoritative = _authoritative_terminal(CANONICAL_ATTEMPT_DIRECTORY)
            if authoritative is not None:
                existing_report, existing_status = authoritative
                identity = finalize_attempt(
                    CANONICAL_ATTEMPT_DIRECTORY,
                    existing_report,
                    evaluator_exit_status=existing_status,
                )
                raise SystemExit(
                    f"recovered_{existing_report['verdict']}: "
                    f"{identity['report']}"
                ) from error
            persisted_attempt = _verified_on_disk_claim(
                CANONICAL_ATTEMPT_DIRECTORY
            )
            if persisted_attempt is None:
                raise SystemExit(
                    "canonical attempt exists but its durable claim is malformed; "
                    "refusing to overwrite evidence"
                ) from error
            verdict = (
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "integrity_failed"
            )
            failure_report = base_report | {
                "schema_version": CONFIRMATION_SCHEMA_VERSION,
                "protocol": CONFIRMATION_PROTOCOL,
                "attempt_id": ATTEMPT_ID,
                "started_at": started_at,
                "completed_at": utc_now(),
                "elapsed_seconds": time.monotonic() - wall_started,
                "claim": persisted_attempt,
                "layout_qualification": selections,
                "selection_journal": _selection_journal_evidence(journal_directory),
                "checkpoint_scoring_performed": scoring_started,
                "checkpoints": checkpoint_results,
                "policy_updates": False,
                "verdict": verdict,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
            identity = finalize_attempt(
                CANONICAL_ATTEMPT_DIRECTORY,
                failure_report,
                evaluator_exit_status=1,
            )
            raise SystemExit(f"{verdict}: {identity['report']}: {error}") from error
        raise SystemExit(str(error)) from error
    sys.stdout.write(f"{report['verdict']}: {identity['report']}\n")
    if report["verdict"] != "confirmed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
