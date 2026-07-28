"""One-shot sealed generator qualification and immutable U2 report verification.

There is one canonical qualification directory and one attempt identity.  The
qualifier is launched with a 256-bit token, atomically records its claim before
opening a protected layout, and cannot be rerun.  The public verifier never
opens a layout: it authenticates the canonical report against a mandatory
external digest and returns the exact bytes it parsed so training never rereads
mutable sibling files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dungeon_apprentice.artifacts import (
    atomic_write_json,
    atomic_write_text,
    git_snapshot,
    runtime_snapshot,
    utc_now,
)
from dungeon_apprentice.u2_seed_guard import (
    U2SeedAccess,
    U2SeedRole,
    _new_sealed_launch_claim,
    _new_verified_qualification_binding,
    _sealed_preflight_seed_access,
    qualified_training_seed_access,
    sealed_acknowledgement,
    seed_partition_ledger,
)
from dungeon_apprentice.v02_u2_lessons import (
    GENERATOR_PROFILE,
    GENERATOR_PROFILE_VERSION,
    MAX_GENERATION_ATTEMPTS,
    U2_LESSON_SPECS,
    U2LessonId,
)

# The one-shot qualification predates U2 confirmation consumption and the U2r
# successor reservation.  Its externally anchored report must remain
# verifiable against the ledger that was true when those bytes were written.
FROZEN_U2_QUALIFICATION_SEED_LEDGER: tuple[dict[str, Any], ...] = (
    {"role": "training", "start": 0, "opened_by_u2": True, "end": 999_999},
    {
        "role": "engineering",
        "start": 5_200_000,
        "opened_by_u2": True,
        "end": 5_200_999,
    },
    {
        "role": "sealed_qualification",
        "start": 5_210_000,
        "opened_by_u2": True,
        "end": 5_211_999,
    },
    {
        "role": "navigate_validation",
        "start": 10_000_000,
        "opened_by_u2": False,
        "end": 10_000_079,
    },
    {
        "role": "u0_validation",
        "start": 11_000_000,
        "opened_by_u2": False,
        "end": 11_000_079,
    },
    {
        "role": "u1_validation",
        "start": 11_100_000,
        "opened_by_u2": False,
        "end": 11_100_079,
    },
    {
        "role": "u2_validation",
        "start": 11_200_000,
        "opened_by_u2": True,
        "end": 11_200_079,
    },
    {
        "role": "future_u2_confirmation",
        "start": 15_200_000,
        "opened_by_u2": False,
        "end": 15_209_999,
    },
    {
        "role": "future_navigate_confirmation",
        "start": 15_210_000,
        "opened_by_u2": False,
        "end": 15_219_999,
    },
    {
        "role": "future_u0_confirmation",
        "start": 15_220_000,
        "opened_by_u2": False,
        "end": 15_229_999,
    },
    {
        "role": "future_u1_confirmation",
        "start": 15_230_000,
        "opened_by_u2": False,
        "end": 15_239_999,
    },
    {
        "role": "final_test",
        "start": 20_000_000,
        "opened_by_u2": False,
        "end": 20_299_999,
    },
)

PROTOCOL = "dungeon-apprentice-v0.2-u2"
QUALIFICATION_SCHEMA_VERSION = 2
QUALIFICATION_KIND = "sealed_preflight_qualification"
QUALIFICATION_ATTEMPT_SCHEMA_VERSION = 2
QUALIFICATION_ATTEMPT_PROTOCOL = "dungeon-apprentice-v0.2-u2-qualification-attempt"
QUALIFICATION_ATTEMPT_KIND = "canonical_sealed_preflight_attempt"
QUALIFICATION_CLAIM_SCHEMA_VERSION = 1
QUALIFICATION_CLAIM_PROTOCOL = "dungeon-apprentice-v0.2-u2-qualification-claim"
QUALIFICATION_CLAIM_KIND = "canonical_sealed_preflight_launch_claim"
QUALIFICATION_ATTEMPT_ID = "u2-preflight-v0.2-u2-20260723-attempt-1"
CANONICAL_QUALIFICATION_DIRECTORY = Path(
    "/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.2-u2-20260723"
)
CANONICAL_REPORT_NAME = "report.json"
CANONICAL_CHECKSUM_NAME = "report.json.sha256"
CANONICAL_ATTEMPT_NAME = "attempt.json"
CANONICAL_CLAIM_NAME = "launch-claim.json"

LESSON_ID = U2LessonId.SEPARATED_UNLOCK.value
QUALIFICATION_SEED_BASE = 5_210_000
QUALIFICATION_CASES = 2_000
QUALIFICATION_SEED_END = QUALIFICATION_SEED_BASE + QUALIFICATION_CASES - 1
U2_VALIDATION_SEED_BASE = 11_200_000
U2_VALIDATION_CASES = 80
FULL_UNIQUENESS_REQUIRED = 1_980
GEOMETRY_UNIQUENESS_REQUIRED = 1_900
ORACLE_ACTION_MINIMUM = 17
ORACLE_ACTION_MAXIMUM = 26
U2_HORIZON = U2_LESSON_SPECS[U2LessonId.SEPARATED_UNLOCK].max_steps
VISIBILITY_STRATA = (
    "key_hidden_door_hidden",
    "key_hidden_door_visible",
    "key_visible_door_hidden",
    "key_visible_door_visible",
)
RUNTIME_PACKAGES = frozenset(
    {"gymnasium", "minigrid", "numpy", "sb3-contrib", "stable-baselines3", "torch"}
)
CONTRACT_COUNT_FIELDS = frozenset(
    {
        "bounded_rejection_contract",
        "complete_divider_with_sole_door",
        "door_requires_matching_key",
        "exactly_two_extra_walls",
        "generator_identity_contract",
        "horizon_contract",
        "independent_goal_blocked_while_locked",
        "objects_on_declared_sides",
        "observation_contract",
        "ordered_objective_completed",
        "planner_live_action_match",
        "post_key_turn_present",
        "reward_contract",
        "success_terminated_not_truncated",
    }
)
CASE_FIELDS = frozenset(
    {
        "seed",
        "layout_sha256",
        "geometry_sha256",
        "oracle_actions",
        "key_visible",
        "door_visible",
        "generation_attempts",
        "max_steps",
        "elapsed_steps",
        "terminated",
        "truncated",
        "contracts",
    }
)
REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "kind",
        "result",
        "policy_updates",
        "attempt_id",
        "claim_id",
        "claim_sha256",
        "started_at",
        "completed_at",
        "duration_seconds",
        "source",
        "runtime",
        "generator_profile",
        "generator_profile_version",
        "lesson_id",
        "seed_partition",
        "seed_partition_ledger",
        "requested",
        "generated",
        "solved",
        "full_unique_layouts",
        "geometry_unique_layouts",
        "contract_counts",
        "visibility_strata",
        "validation_disjointness",
        "validation_reference",
        "cases",
        "failures",
    }
)
CLAIM_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "kind",
        "status",
        "claimed_at",
        "source",
        "attempt_id",
        "claim_id",
        "qualification_directory",
        "report",
        "attempt",
        "launcher_token_sha256",
        "acknowledgement_sha256",
        "seed_partition",
    }
)
ATTEMPT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "kind",
        "status",
        "attempt_id",
        "claim_id",
        "claim_sha256",
        "started_at",
        "completed_at",
        "source",
        "qualification_directory",
        "report",
        "claim",
        "seed_partition",
        "report_sha256",
        "report_result",
        "qualifier_exit_status",
    }
)
TOPOLOGY_FIELDS = frozenset(
    {
        "size",
        "max_steps",
        "start",
        "key",
        "goal",
        "door",
        "walls",
        "divider_walls",
        "extra_walls",
        "divider_orientation",
        "divider_index",
        "door_lane",
        "approach_from_low",
    }
)
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")
_IDENTITY = re.compile(r"^[a-z0-9][a-z0-9._:-]{7,127}$")
_SUSPICIOUS_SEQUENCE_KEYS = (
    "action_sequence",
    "action_trace",
    "actions_list",
    "demonstration",
    "oracle_path",
    "oracle_plan",
    "oracle_trajectory",
    "policy_trace",
    "move_sequence",
    "command_sequence",
)


class U2QualificationError(RuntimeError):
    """Raised when the sealed preflight or its immutable evidence is invalid."""


def canonical_qualification_paths() -> Mapping[str, Path]:
    """Return the single frozen qualification location."""

    directory = CANONICAL_QUALIFICATION_DIRECTORY.expanduser().absolute()
    return MappingProxyType(
        {
            "directory": directory,
            "report": directory / CANONICAL_REPORT_NAME,
            "checksum": directory / CANONICAL_CHECKSUM_NAME,
            "attempt": directory / CANONICAL_ATTEMPT_NAME,
            "claim": directory / CANONICAL_CLAIM_NAME,
        }
    )


@dataclass(frozen=True)
class U2QualificationEvidence:
    report: str
    report_sha256: str
    report_byte_length: int
    checksum: str
    attempt_ledger: str
    attempt_id: str
    attempt_sha256: str
    launch_claim: str
    claim_id: str
    claim_sha256: str
    source_commit: str
    generator_profile: str
    generator_profile_version: int
    lesson_id: str
    seed_start: int
    seed_end: int
    requested: int
    generated: int
    solved: int
    full_unique_layouts: int
    geometry_unique_layouts: int
    validation_generated: int
    validation_unique_layouts: int
    validation_exact_overlap: int
    _report_bytes: bytes = field(repr=False, compare=False)
    _report_snapshot: Mapping[str, Any] = field(repr=False, compare=False)
    _seed_binding: Any = field(repr=False, compare=False)

    def public_dict(self) -> dict[str, Any]:
        """Return provenance only; report contents and capabilities remain private."""

        return {
            "report": self.report,
            "report_sha256": self.report_sha256,
            "report_byte_length": self.report_byte_length,
            "checksum": self.checksum,
            "attempt_ledger": self.attempt_ledger,
            "attempt_id": self.attempt_id,
            "attempt_sha256": self.attempt_sha256,
            "launch_claim": self.launch_claim,
            "claim_id": self.claim_id,
            "claim_sha256": self.claim_sha256,
            "source_commit": self.source_commit,
            "generator_profile": self.generator_profile,
            "generator_profile_version": self.generator_profile_version,
            "lesson_id": self.lesson_id,
            "seed_start": self.seed_start,
            "seed_end": self.seed_end,
            "requested": self.requested,
            "generated": self.generated,
            "solved": self.solved,
            "full_unique_layouts": self.full_unique_layouts,
            "geometry_unique_layouts": self.geometry_unique_layouts,
            "validation_generated": self.validation_generated,
            "validation_unique_layouts": self.validation_unique_layouts,
            "validation_exact_overlap": self.validation_exact_overlap,
        }

    @property
    def report_snapshot(self) -> Mapping[str, Any]:
        """Immutable parse of the exact externally anchored report bytes."""

        return self._report_snapshot

    def verified_report(self) -> dict[str, Any]:
        """Return a fresh mutable copy without touching the filesystem."""

        value = json.loads(self._report_bytes)
        if not isinstance(value, dict):  # protected by verifier; defensive API boundary
            raise U2QualificationError("verified report bytes no longer encode an object")
        return value

    def qualified_seed_access(self) -> U2SeedAccess:
        """Issue validation access bound to this exact report/attempt/claim snapshot."""

        return qualified_training_seed_access(self._seed_binding)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _regular_file_bytes(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise U2QualificationError(f"cannot open regular {label} {path}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise U2QualificationError(f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _json_from_bytes(value: bytes, label: str, path: Path) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise U2QualificationError(f"cannot decode {label} {path}: {error}") from error
    if not isinstance(decoded, dict):
        raise U2QualificationError(f"{label} must be a JSON object: {path}")
    return decoded


def _read_json_once(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    value = _regular_file_bytes(path, label)
    return value, _json_from_bytes(value, label, path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HEX_DIGEST.fullmatch(value):
        raise U2QualificationError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _require_commit(value: Any, label: str = "source commit") -> str:
    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise U2QualificationError(f"{label} is not a hexadecimal commit")
    return value


def _require_identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTITY.fullmatch(value):
        raise U2QualificationError(f"{label} is not a canonical identity")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise U2QualificationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise U2QualificationError(f"{label} must be at least {minimum}")
    return value


def _require_fields(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    measured = set(value)
    if measured != expected:
        missing = sorted(expected - measured)
        extra = sorted(measured - expected)
        raise U2QualificationError(
            f"{label} fields changed (missing={missing}, extra={extra})"
        )


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise U2QualificationError(f"{label} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise U2QualificationError(f"{label} is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise U2QualificationError(f"{label} must include the UTC offset")
    return parsed


def _verify_runtime(report: Mapping[str, Any]) -> None:
    started = _parse_utc(report.get("started_at"), "qualification start")
    completed = _parse_utc(report.get("completed_at"), "qualification completion")
    duration = report.get("duration_seconds")
    if (
        type(duration) not in {int, float}
        or isinstance(duration, bool)
        or not math.isfinite(float(duration))
        or float(duration) <= 0.0
    ):
        raise U2QualificationError("qualification duration must be finite and positive")
    elapsed = (completed - started).total_seconds()
    if elapsed < 0 or abs(elapsed - float(duration)) > 3.0:
        raise U2QualificationError("qualification timestamps and duration disagree")
    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "python",
        "platform",
        "machine",
        "packages",
    }:
        raise U2QualificationError("qualification runtime snapshot fields changed")
    for field_name in ("python", "platform", "machine"):
        if not isinstance(runtime[field_name], str) or not runtime[field_name].strip():
            raise U2QualificationError(f"qualification runtime {field_name} is empty")
    packages = runtime["packages"]
    if not isinstance(packages, Mapping) or set(packages) != RUNTIME_PACKAGES:
        raise U2QualificationError("qualification package snapshot fields changed")
    for package, version in packages.items():
        if not isinstance(version, str) or not version.strip():
            raise U2QualificationError(f"qualification runtime package {package} is unresolved")


def _reject_oracle_demonstrations(value: Any, *, path: str = "report") -> None:
    """Reject action demonstrations even when encoded as strings or nested objects."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.lower().replace("-", "_").replace(" ", "_")
            if normalized == "oracle_actions":
                if type(child) is not int:
                    raise U2QualificationError(
                        f"{path}.{key} must be a scalar oracle action count"
                    )
            elif any(fragment in normalized for fragment in _SUSPICIOUS_SEQUENCE_KEYS):
                raise U2QualificationError(
                    f"{path}.{key} contains prohibited oracle demonstration material"
                )
            elif (
                isinstance(child, (list, tuple, bytes, bytearray))
                and ("oracle" in normalized or "action" in normalized)
            ):
                raise U2QualificationError(
                    f"{path}.{key} contains a prohibited oracle action sequence"
                )
            _reject_oracle_demonstrations(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_oracle_demonstrations(child, path=f"{path}[{index}]")


def _visibility_name(*, key_visible: bool, door_visible: bool) -> str:
    key = "visible" if key_visible else "hidden"
    door = "visible" if door_visible else "hidden"
    return f"key_{key}_door_{door}"


def _position(value: Any, label: str, *, size: int) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(type(component) is not int for component in value)
    ):
        raise U2QualificationError(f"{label} is not an integer grid position")
    position = (value[0], value[1])
    if not (1 <= position[0] < size - 1 and 1 <= position[1] < size - 1):
        raise U2QualificationError(f"{label} is outside the playable interior")
    return position


def _positions(value: Any, label: str, *, size: int) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise U2QualificationError(f"{label} must be a position sequence")
    positions = tuple(
        _position(position, f"{label}[{index}]", size=size)
        for index, position in enumerate(value)
    )
    if len(set(positions)) != len(positions):
        raise U2QualificationError(f"{label} contains duplicate positions")
    return positions


def _reachable_while_locked(
    *,
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: set[tuple[int, int]],
    size: int,
) -> bool:
    queue = deque([start])
    visited = {start}
    while queue:
        current = queue.popleft()
        if current == goal:
            return True
        x, y = current
        for candidate in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (
                1 <= candidate[0] < size - 1
                and 1 <= candidate[1] < size - 1
                and candidate not in blocked
                and candidate not in visited
            ):
                visited.add(candidate)
                queue.append(candidate)
    return False


def _independent_topology_contracts(case: Mapping[str, Any]) -> dict[str, bool]:
    topology = case.get("qualification_topology")
    if not isinstance(topology, Mapping):
        raise U2QualificationError("generator evidence lacks qualification topology")
    _require_fields(topology, TOPOLOGY_FIELDS, "qualification topology")
    size = _require_exact_int(topology["size"], "topology size", minimum=1)
    max_steps = _require_exact_int(topology["max_steps"], "topology horizon", minimum=1)
    start = _position(topology["start"], "topology start", size=size)
    key = _position(topology["key"], "topology key", size=size)
    goal = _position(topology["goal"], "topology goal", size=size)
    door = _position(topology["door"], "topology door", size=size)
    walls = set(_positions(topology["walls"], "topology walls", size=size))
    divider = set(
        _positions(topology["divider_walls"], "topology divider", size=size)
    )
    extras = set(_positions(topology["extra_walls"], "topology extras", size=size))
    orientation = topology["divider_orientation"]
    if orientation not in {"vertical", "horizontal"}:
        raise U2QualificationError("topology divider orientation changed")
    divider_index = _require_exact_int(
        topology["divider_index"], "topology divider index", minimum=1
    )
    door_lane = _require_exact_int(topology["door_lane"], "topology door lane", minimum=1)
    if type(topology["approach_from_low"]) is not bool:
        raise U2QualificationError("topology approach side is not boolean")
    approach_from_low = topology["approach_from_low"]
    if not (1 <= divider_index < size - 1 and 1 <= door_lane < size - 1):
        raise U2QualificationError("topology divider metadata is outside the interior")

    if orientation == "vertical":
        expected_door = (divider_index, door_lane)
        expected_divider = {
            (divider_index, lane)
            for lane in range(1, size - 1)
            if lane != door_lane
        }
        axis = 0
    else:
        expected_door = (door_lane, divider_index)
        expected_divider = {
            (lane, divider_index)
            for lane in range(1, size - 1)
            if lane != door_lane
        }
        axis = 1

    objects = {start, key, goal, door}
    complete_divider = (
        door == expected_door
        and divider == expected_divider
        and len(extras) == 2
        and not divider & extras
        and walls == divider | extras
        and len(objects) == 4
        and not walls & objects
    )

    def approach(position: tuple[int, int]) -> bool:
        coordinate = position[axis]
        return (
            coordinate < divider_index
            if approach_from_low
            else coordinate > divider_index
        )

    objects_on_sides = (
        approach(start)
        and approach(key)
        and not approach(goal)
        and all(position[axis] != divider_index for position in (start, key, goal))
    )
    locked_reachable = _reachable_while_locked(
        start=start,
        goal=goal,
        blocked=walls | {door},
        size=size,
    )
    return {
        "complete_divider_with_sole_door": complete_divider,
        "exactly_two_extra_walls": len(extras) == 2,
        "objects_on_declared_sides": objects_on_sides,
        "independent_goal_blocked_while_locked": not locked_reachable,
        "horizon_contract": size == 9 and max_steps == U2_HORIZON,
    }


def _case_contracts(
    case: Mapping[str, Any],
    *,
    expected_seed: int | None = None,
    expected_seed_role: U2SeedRole | None = None,
) -> dict[str, bool]:
    """Recompute every sealed contract from rich transient generator evidence."""

    topology_contracts = _independent_topology_contracts(case)
    shape = case.get("observation_shape")
    pure_actions = case.get("pure_planner_actions")
    live_actions = case.get("live_oracle_actions")
    generation_attempts = case.get("generation_attempts")
    max_generation_attempts = case.get("max_generation_attempts")
    elapsed_steps = case.get("elapsed_steps")
    numeric_counts = all(
        type(value) is int
        for value in (
            pure_actions,
            live_actions,
            generation_attempts,
            max_generation_attempts,
            elapsed_steps,
        )
    )
    planner_match = (
        numeric_counts
        and case.get("planner_oracle_counts_match") is True
        and case.get("planner_live_action_match") is True
        and pure_actions == live_actions
        and ORACLE_ACTION_MINIMUM <= live_actions <= ORACLE_ACTION_MAXIMUM
    )
    return {
        **topology_contracts,
        "bounded_rejection_contract": (
            numeric_counts
            and max_generation_attempts == MAX_GENERATION_ATTEMPTS
            and 1 <= generation_attempts <= max_generation_attempts
        ),
        "door_requires_matching_key": case.get("door_requires_matching_key") is True,
        "generator_identity_contract": (
            case.get("protocol") == PROTOCOL
            and case.get("generator_profile") == GENERATOR_PROFILE
            and case.get("generator_profile_version") == GENERATOR_PROFILE_VERSION
            and type(case.get("seed")) is int
            and (
                expected_seed is None
                or case.get("seed") == expected_seed
            )
            and (
                case.get("seed_role")
                in {
                    U2SeedRole.SEALED_QUALIFICATION.value,
                    U2SeedRole.U2_VALIDATION.value,
                }
                if expected_seed_role is None
                else case.get("seed_role") == expected_seed_role.value
            )
        ),
        "observation_contract": (
            shape == [56, 56, 3]
            and case.get("observation_dtype") == "uint8"
            and case.get("action_count") == 7
            and case.get("observation_contract") is True
        ),
        "ordered_objective_completed": (
            case.get("success") is True
            and case.get("ordered_objective_completed") is True
            and case.get("terminal_reason") == "success"
        ),
        "planner_live_action_match": planner_match,
        "post_key_turn_present": (
            type(case.get("post_key_pre_door_turns")) is int
            and case["post_key_pre_door_turns"] >= 1
            and case.get("post_key_turn_present") is True
        ),
        "reward_contract": case.get("reward_contract") is True,
        "success_terminated_not_truncated": (
            numeric_counts
            and case.get("terminated") is True
            and case.get("truncated") is False
            and case.get("terminal_reason") == "success"
            and elapsed_steps == live_actions
            and elapsed_steps < U2_HORIZON
        ),
    }


def _normalize_case(
    case: Mapping[str, Any],
    contracts: Mapping[str, bool],
) -> dict[str, Any]:
    """Strip transient topology and all oracle actions from public evidence."""

    _require_fields(contracts, CONTRACT_COUNT_FIELDS, "case contracts")
    try:
        if type(case["initial_key_visible"]) is not bool or type(
            case["initial_door_visible"]
        ) is not bool:
            raise TypeError("initial visibility must be boolean")
        for field_name in (
            "seed",
            "live_oracle_actions",
            "generation_attempts",
            "elapsed_steps",
        ):
            if type(case[field_name]) is not int:
                raise TypeError(f"{field_name} must be an integer")
        topology = case["qualification_topology"]
        if not isinstance(topology, Mapping) or type(topology["max_steps"]) is not int:
            raise TypeError("qualification topology horizon must be an integer")
        return {
            "seed": case["seed"],
            "layout_sha256": _require_digest(case["layout_sha256"], "layout"),
            "geometry_sha256": _require_digest(case["geometry_sha256"], "geometry"),
            "oracle_actions": case["live_oracle_actions"],
            "key_visible": case["initial_key_visible"],
            "door_visible": case["initial_door_visible"],
            "generation_attempts": case["generation_attempts"],
            "max_steps": topology["max_steps"],
            "elapsed_steps": case["elapsed_steps"],
            "terminated": case.get("terminated"),
            "truncated": case.get("truncated"),
            "contracts": {
                field_name: contracts[field_name]
                for field_name in sorted(CONTRACT_COUNT_FIELDS)
            },
        }
    except (KeyError, TypeError, ValueError) as error:
        raise U2QualificationError(f"invalid generator evidence: {error}") from error


def _verify_cases(
    report: Mapping[str, Any],
) -> tuple[int, int, Counter[str], Counter[str]]:
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != QUALIFICATION_CASES:
        raise U2QualificationError(
            f"qualification must contain exactly {QUALIFICATION_CASES} cases"
        )
    measured_seeds: list[int] = []
    full_hashes: set[str] = set()
    geometry_hashes: set[str] = set()
    visibility = Counter[str]()
    contract_counts = Counter[str]()
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise U2QualificationError(f"qualification case {index} is not an object")
        _require_fields(case, CASE_FIELDS, f"qualification case {index}")
        seed = _require_exact_int(case["seed"], f"case {index} seed", minimum=0)
        oracle_actions = _require_exact_int(
            case["oracle_actions"], f"case {index} oracle actions", minimum=0
        )
        generation_attempts = _require_exact_int(
            case["generation_attempts"], f"case {index} generation attempts", minimum=1
        )
        max_steps = _require_exact_int(
            case["max_steps"], f"case {index} max steps", minimum=1
        )
        elapsed_steps = _require_exact_int(
            case["elapsed_steps"], f"case {index} elapsed steps", minimum=1
        )
        if not ORACLE_ACTION_MINIMUM <= oracle_actions <= ORACLE_ACTION_MAXIMUM:
            raise U2QualificationError(
                f"qualification case {index} has oracle length {oracle_actions}"
            )
        if (
            generation_attempts > MAX_GENERATION_ATTEMPTS
            or max_steps != U2_HORIZON
            or elapsed_steps != oracle_actions
            or case["terminated"] is not True
            or case["truncated"] is not False
        ):
            raise U2QualificationError(
                f"qualification case {index} runtime contract changed"
            )
        layout_hash = _require_digest(case["layout_sha256"], f"case {index} layout")
        geometry_hash = _require_digest(case["geometry_sha256"], f"case {index} geometry")
        if type(case["key_visible"]) is not bool or type(case["door_visible"]) is not bool:
            raise U2QualificationError(f"qualification case {index} visibility is not boolean")
        contracts = case["contracts"]
        if not isinstance(contracts, Mapping):
            raise U2QualificationError(f"qualification case {index} contracts are not an object")
        _require_fields(contracts, CONTRACT_COUNT_FIELDS, f"case {index} contracts")
        if any(contracts[field_name] is not True for field_name in CONTRACT_COUNT_FIELDS):
            raise U2QualificationError(
                f"qualification case {index} did not satisfy every U2 contract"
            )
        contract_counts.update(CONTRACT_COUNT_FIELDS)
        measured_seeds.append(seed)
        full_hashes.add(layout_hash)
        geometry_hashes.add(geometry_hash)
        visibility[
            _visibility_name(
                key_visible=case["key_visible"],
                door_visible=case["door_visible"],
            )
        ] += 1
    expected_seeds = list(range(QUALIFICATION_SEED_BASE, QUALIFICATION_SEED_END + 1))
    if measured_seeds != expected_seeds:
        raise U2QualificationError("qualification cases do not use the exact sealed seed order")
    return len(full_hashes), len(geometry_hashes), visibility, contract_counts


def _verify_report_schema(
    report: Mapping[str, Any],
    *,
    expected_source_commit: str,
    expected_attempt_id: str,
    expected_claim_id: str,
) -> None:
    _reject_oracle_demonstrations(report)
    _require_fields(report, REPORT_FIELDS, "qualification report")
    source = report["source"]
    partition = report["seed_partition"]
    disjointness = report["validation_disjointness"]
    validation_reference = report["validation_reference"]
    if (
        report["schema_version"] != QUALIFICATION_SCHEMA_VERSION
        or report["protocol"] != PROTOCOL
        or report["kind"] != QUALIFICATION_KIND
        or report["result"] != "passed"
        or report["policy_updates"] is not False
        or report["lesson_id"] != LESSON_ID
        or report["generator_profile"] != GENERATOR_PROFILE
        or report["generator_profile_version"] != GENERATOR_PROFILE_VERSION
    ):
        raise U2QualificationError("report is not the frozen passed U2 preflight")
    if report["attempt_id"] != expected_attempt_id:
        raise U2QualificationError("qualification attempt identity changed")
    if report["claim_id"] != expected_claim_id:
        raise U2QualificationError("qualification claim identity changed")
    _require_digest(report["claim_sha256"], "qualification claim")
    if source != {"commit": expected_source_commit, "dirty": False}:
        raise U2QualificationError("qualification source does not match the clean training source")
    if partition != {
        "role": U2SeedRole.SEALED_QUALIFICATION.value,
        "start": QUALIFICATION_SEED_BASE,
        "end": QUALIFICATION_SEED_END,
        "count": QUALIFICATION_CASES,
    }:
        raise U2QualificationError("qualification seed partition changed")
    if json.dumps(
        report["seed_partition_ledger"],
        sort_keys=True,
        separators=(",", ":"),
    ) not in {
        json.dumps(
            list(seed_partition_ledger()),
            sort_keys=True,
            separators=(",", ":"),
        ),
        json.dumps(
            list(FROZEN_U2_QUALIFICATION_SEED_LEDGER),
            sort_keys=True,
            separators=(",", ":"),
        ),
    }:
        raise U2QualificationError("qualification seed partition ledger changed")
    for field_name, expected in {
        "requested": QUALIFICATION_CASES,
        "generated": QUALIFICATION_CASES,
        "solved": QUALIFICATION_CASES,
    }.items():
        if report[field_name] != expected or type(report[field_name]) is not int:
            raise U2QualificationError(f"qualification {field_name} must be {expected}")
    if report["failures"] != []:
        raise U2QualificationError("a qualification with failures cannot authorize training")
    _verify_runtime(report)

    full_unique, geometry_unique, visibility, measured_contracts = _verify_cases(report)
    if report["contract_counts"] != {
        field_name: measured_contracts[field_name]
        for field_name in sorted(CONTRACT_COUNT_FIELDS)
    }:
        raise U2QualificationError("qualification contract counts do not match its cases")
    expected_visibility = {name: visibility[name] for name in VISIBILITY_STRATA}
    if report["visibility_strata"] != expected_visibility:
        raise U2QualificationError("qualification visibility strata do not match its cases")
    if (
        report["full_unique_layouts"] != full_unique
        or full_unique < FULL_UNIQUENESS_REQUIRED
    ):
        raise U2QualificationError("qualification exact-layout uniqueness gate failed")
    if (
        report["geometry_unique_layouts"] != geometry_unique
        or geometry_unique < GEOMETRY_UNIQUENESS_REQUIRED
    ):
        raise U2QualificationError("qualification geometry uniqueness gate failed")

    if disjointness != {
        "u2_validation_start": U2_VALIDATION_SEED_BASE,
        "u2_validation_cases": U2_VALIDATION_CASES,
        "u2_validation_generated": U2_VALIDATION_CASES,
        "exact_visual_overlap": 0,
    }:
        raise U2QualificationError("qualification/validation disjointness changed")
    if not isinstance(validation_reference, Mapping) or set(validation_reference) != {
        "generated",
        "unique_exact_layouts",
        "unique_geometries",
        "exact_layout_set_sha256",
        "geometry_set_sha256",
    }:
        raise U2QualificationError("qualification validation reference fields changed")
    validation_generated = _require_exact_int(
        validation_reference["generated"],
        "validation generated count",
        minimum=0,
    )
    validation_unique = _require_exact_int(
        validation_reference["unique_exact_layouts"],
        "validation exact uniqueness",
        minimum=0,
    )
    validation_geometry_unique = _require_exact_int(
        validation_reference["unique_geometries"],
        "validation geometry uniqueness",
        minimum=0,
    )
    if (
        validation_generated != U2_VALIDATION_CASES
        or not 1 <= validation_unique <= validation_generated
        or not 1 <= validation_geometry_unique <= validation_generated
    ):
        raise U2QualificationError(
            "validation generated count and hash uniqueness are inconsistent"
        )
    _require_digest(
        validation_reference["exact_layout_set_sha256"],
        "validation exact-layout set",
    )
    _require_digest(
        validation_reference["geometry_set_sha256"],
        "validation geometry set",
    )


def _assert_canonical_report_path(path: Path) -> Path:
    expected = canonical_qualification_paths()["report"]
    supplied = path.expanduser().absolute()
    if supplied != expected:
        raise U2QualificationError(
            f"qualification report must be the canonical path {expected}"
        )
    if supplied.is_symlink():
        raise U2QualificationError("canonical qualification report cannot be a symlink")
    _reject_symlink_chain(supplied)
    return supplied


def _reject_symlink_chain(path: Path) -> None:
    current = path.expanduser().absolute()
    while True:
        if current.is_symlink():
            raise U2QualificationError(
                f"canonical qualification path contains a symlink: {current}"
            )
        if current == current.parent:
            return
        current = current.parent


def _verify_checksum_bytes(checksum_bytes: bytes, report_digest: str) -> None:
    expected = f"{report_digest}  {CANONICAL_REPORT_NAME}\n".encode("ascii")
    if checksum_bytes != expected:
        raise U2QualificationError(
            "qualification checksum sidecar does not exactly match the report"
        )


def _verify_claim(
    claim: Mapping[str, Any],
    *,
    expected_source_commit: str,
    expected_attempt_id: str,
    expected_claim_id: str,
) -> None:
    _require_fields(claim, CLAIM_FIELDS, "qualification launch claim")
    paths = canonical_qualification_paths()
    if (
        claim["schema_version"] != QUALIFICATION_CLAIM_SCHEMA_VERSION
        or claim["protocol"] != QUALIFICATION_CLAIM_PROTOCOL
        or claim["kind"] != QUALIFICATION_CLAIM_KIND
        or claim["status"] != "claimed_for_single_launch"
        or claim["source"] != {"commit": expected_source_commit, "dirty": False}
        or claim["attempt_id"] != expected_attempt_id
        or claim["claim_id"] != expected_claim_id
        or claim["qualification_directory"] != str(paths["directory"])
        or claim["report"] != str(paths["report"])
        or claim["attempt"] != str(paths["attempt"])
        or claim["seed_partition"]
        != {
            "start": QUALIFICATION_SEED_BASE,
            "end": QUALIFICATION_SEED_END,
            "count": QUALIFICATION_CASES,
        }
    ):
        raise U2QualificationError("qualification launch claim is inconsistent")
    _parse_utc(claim["claimed_at"], "qualification claim time")
    _require_digest(claim["launcher_token_sha256"], "qualification launcher token")
    expected_ack = hashlib.sha256(sealed_acknowledgement().encode("utf-8")).hexdigest()
    if claim["acknowledgement_sha256"] != expected_ack:
        raise U2QualificationError("qualification acknowledgement binding changed")


def _verify_attempt(
    attempt: Mapping[str, Any],
    *,
    expected_source_commit: str,
    expected_attempt_id: str,
    expected_claim_id: str,
    expected_claim_sha256: str,
    expected_report_sha256: str,
) -> None:
    _require_fields(attempt, ATTEMPT_FIELDS, "qualification attempt ledger")
    paths = canonical_qualification_paths()
    if (
        attempt["schema_version"] != QUALIFICATION_ATTEMPT_SCHEMA_VERSION
        or attempt["protocol"] != QUALIFICATION_ATTEMPT_PROTOCOL
        or attempt["kind"] != QUALIFICATION_ATTEMPT_KIND
        or attempt["status"] != "completed_with_report"
        or attempt["attempt_id"] != expected_attempt_id
        or attempt["claim_id"] != expected_claim_id
        or attempt["claim_sha256"] != expected_claim_sha256
        or attempt["source"] != {"commit": expected_source_commit, "dirty": False}
        or attempt["qualification_directory"] != str(paths["directory"])
        or attempt["report"] != str(paths["report"])
        or attempt["claim"] != str(paths["claim"])
        or attempt["seed_partition"]
        != {
            "start": QUALIFICATION_SEED_BASE,
            "end": QUALIFICATION_SEED_END,
            "count": QUALIFICATION_CASES,
        }
        or attempt["report_sha256"] != expected_report_sha256
        or attempt["report_result"] != "passed"
        or attempt["qualifier_exit_status"] != 0
    ):
        raise U2QualificationError("qualification attempt ledger is incomplete or inconsistent")
    started = _parse_utc(attempt["started_at"], "qualification attempt start")
    completed = _parse_utc(attempt["completed_at"], "qualification attempt completion")
    if completed < started:
        raise U2QualificationError("qualification attempt timestamps are reversed")


def verify_u2_qualification_report(
    path: Path,
    *,
    expected_source_commit: str,
    expected_sha256: str,
    expected_attempt_id: str,
    expected_claim_id: str,
) -> U2QualificationEvidence:
    """Authenticate the canonical evidence against caller-supplied immutable anchors."""

    source_commit = _require_commit(expected_source_commit)
    anchored_digest = _require_digest(expected_sha256, "expected qualification")
    attempt_id = _require_identity(expected_attempt_id, "expected attempt ID")
    claim_id = _require_digest(expected_claim_id, "expected claim ID")
    report_path = _assert_canonical_report_path(path)
    paths = canonical_qualification_paths()
    for evidence_path in paths.values():
        _reject_symlink_chain(evidence_path)

    # Each file is opened and read once.  All parsing, hashes, and returned
    # snapshots below derive from these bytes, never from a second path read.
    report_bytes, report = _read_json_once(report_path, "qualification report")
    measured_digest = _sha256_bytes(report_bytes)
    if measured_digest != anchored_digest:
        raise U2QualificationError(
            "qualification report digest differs from the external frozen digest"
        )
    checksum_bytes = _regular_file_bytes(paths["checksum"], "qualification checksum")
    _verify_checksum_bytes(checksum_bytes, measured_digest)
    claim_bytes, claim = _read_json_once(paths["claim"], "qualification launch claim")
    claim_sha256 = _sha256_bytes(claim_bytes)
    _verify_claim(
        claim,
        expected_source_commit=source_commit,
        expected_attempt_id=attempt_id,
        expected_claim_id=claim_id,
    )
    attempt_bytes, attempt = _read_json_once(
        paths["attempt"], "qualification attempt ledger"
    )
    attempt_sha256 = _sha256_bytes(attempt_bytes)
    _verify_attempt(
        attempt,
        expected_source_commit=source_commit,
        expected_attempt_id=attempt_id,
        expected_claim_id=claim_id,
        expected_claim_sha256=claim_sha256,
        expected_report_sha256=measured_digest,
    )
    if (
        report.get("claim_sha256") != claim_sha256
        or report.get("attempt_id") != attempt_id
        or report.get("claim_id") != claim_id
    ):
        raise U2QualificationError("qualification report identity does not match claim/attempt")
    _verify_report_schema(
        report,
        expected_source_commit=source_commit,
        expected_attempt_id=attempt_id,
        expected_claim_id=claim_id,
    )

    binding = _new_verified_qualification_binding(
        source_commit=source_commit,
        report_sha256=measured_digest,
        report_byte_length=len(report_bytes),
        attempt_id=attempt_id,
        attempt_sha256=attempt_sha256,
        claim_id=claim_id,
        claim_sha256=claim_sha256,
    )
    validation_reference = report["validation_reference"]
    return U2QualificationEvidence(
        report=str(report_path),
        report_sha256=measured_digest,
        report_byte_length=len(report_bytes),
        checksum=str(paths["checksum"]),
        attempt_ledger=str(paths["attempt"]),
        attempt_id=attempt_id,
        attempt_sha256=attempt_sha256,
        launch_claim=str(paths["claim"]),
        claim_id=claim_id,
        claim_sha256=claim_sha256,
        source_commit=source_commit,
        generator_profile=str(report["generator_profile"]),
        generator_profile_version=int(report["generator_profile_version"]),
        lesson_id=str(report["lesson_id"]),
        seed_start=QUALIFICATION_SEED_BASE,
        seed_end=QUALIFICATION_SEED_END,
        requested=int(report["requested"]),
        generated=int(report["generated"]),
        solved=int(report["solved"]),
        full_unique_layouts=int(report["full_unique_layouts"]),
        geometry_unique_layouts=int(report["geometry_unique_layouts"]),
        validation_generated=int(validation_reference["generated"]),
        validation_unique_layouts=int(validation_reference["unique_exact_layouts"]),
        validation_exact_overlap=int(
            report["validation_disjointness"]["exact_visual_overlap"]
        ),
        _report_bytes=report_bytes,
        _report_snapshot=_deep_freeze(report),
        _seed_binding=binding,
    )


def _hash_set_sha256(values: set[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def collect_sealed_qualification(
    *,
    access: U2SeedAccess,
    source_commit: str,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Open the exact sealed and U2-validation ranges once and grade every case."""

    from dungeon_apprentice.v02_u2_lessons import generate_u2_case_evidence

    if (
        access.qualification_attempt_id != QUALIFICATION_ATTEMPT_ID
        or access.qualification_claim_id is None
        or access.qualification_claim_sha256 is None
        or access.source_commit != source_commit
    ):
        raise U2QualificationError("sealed access is not bound to the canonical attempt")
    started_at = utc_now()
    wall_start = time.monotonic()
    failures: list[dict[str, Any]] = []
    validation_hashes: set[str] = set()
    validation_geometry_hashes: set[str] = set()
    validation_generated = 0
    for offset in range(U2_VALIDATION_CASES):
        seed = U2_VALIDATION_SEED_BASE + offset
        try:
            evidence = generate_u2_case_evidence(
                seed,
                seed_role=U2SeedRole.U2_VALIDATION,
                access=access,
            )
            validation_generated += 1
            contracts = _case_contracts(
                evidence,
                expected_seed=seed,
                expected_seed_role=U2SeedRole.U2_VALIDATION,
            )
            failed_contracts = sorted(
                field_name for field_name, passed in contracts.items() if not passed
            )
            if failed_contracts:
                raise U2QualificationError(
                    "validation case contract failed: " + ", ".join(failed_contracts)
                )
            validation_hashes.add(
                _require_digest(evidence["layout_sha256"], "validation layout")
            )
            validation_geometry_hashes.add(
                _require_digest(evidence["geometry_sha256"], "validation geometry")
            )
        except Exception as error:  # preserve every per-seed qualification failure
            failures.append(
                {
                    "phase": "u2_validation_reference",
                    "seed": seed,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    if progress is not None:
        progress(
            f"U2 validation reference: {validation_generated}/{U2_VALIDATION_CASES} generated"
        )

    public_cases: list[dict[str, Any]] = []
    contract_counts = Counter[str]()
    sealed_generated = 0
    sealed_solved = 0
    for offset in range(QUALIFICATION_CASES):
        seed = QUALIFICATION_SEED_BASE + offset
        try:
            evidence = generate_u2_case_evidence(
                seed,
                seed_role=U2SeedRole.SEALED_QUALIFICATION,
                access=access,
            )
            sealed_generated += 1
            if evidence.get("success") is True:
                sealed_solved += 1
            contracts = _case_contracts(
                evidence,
                expected_seed=seed,
                expected_seed_role=U2SeedRole.SEALED_QUALIFICATION,
            )
            contract_counts.update(
                field_name for field_name, passed in contracts.items() if passed
            )
            public_cases.append(_normalize_case(evidence, contracts))
            failed_contracts = sorted(
                field_name for field_name, passed in contracts.items() if not passed
            )
            if failed_contracts:
                raise U2QualificationError(
                    "case contract failed: " + ", ".join(failed_contracts)
                )
        except Exception as error:  # continue so the immutable report covers the range
            failures.append(
                {
                    "phase": "sealed_qualification",
                    "seed": seed,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        if progress is not None and (offset + 1) % 100 == 0:
            progress(
                f"Sealed U2 preflight: {offset + 1}/{QUALIFICATION_CASES} cases examined"
            )

    full_hashes = {case["layout_sha256"] for case in public_cases}
    geometry_hashes = {case["geometry_sha256"] for case in public_cases}
    exact_overlap = full_hashes & validation_hashes
    gates = (
        (
            validation_generated == U2_VALIDATION_CASES,
            "the complete 80-case U2 validation reference was not generated",
        ),
        (
            sealed_generated == QUALIFICATION_CASES,
            f"only {sealed_generated}/{QUALIFICATION_CASES} sealed cases generated",
        ),
        (
            sealed_solved == QUALIFICATION_CASES,
            f"only {sealed_solved}/{QUALIFICATION_CASES} sealed cases solved",
        ),
        (
            len(public_cases) == QUALIFICATION_CASES,
            f"only {len(public_cases)}/{QUALIFICATION_CASES} sealed cases recorded",
        ),
        (
            len(full_hashes) >= FULL_UNIQUENESS_REQUIRED,
            f"only {len(full_hashes)} exact layouts; {FULL_UNIQUENESS_REQUIRED} required",
        ),
        (
            len(geometry_hashes) >= GEOMETRY_UNIQUENESS_REQUIRED,
            (
                f"only {len(geometry_hashes)} geometries; "
                f"{GEOMETRY_UNIQUENESS_REQUIRED} required"
            ),
        ),
        (not exact_overlap, f"{len(exact_overlap)} sealed layouts overlap U2 validation"),
    )
    for passed, error_message in gates:
        if not passed:
            failures.append(
                {"phase": "qualification_gate", "seed": None, "error": error_message}
            )
    for field_name in CONTRACT_COUNT_FIELDS:
        if contract_counts[field_name] != QUALIFICATION_CASES:
            failures.append(
                {
                    "phase": "qualification_gate",
                    "seed": None,
                    "error": (
                        f"{field_name} passed "
                        f"{contract_counts[field_name]}/{QUALIFICATION_CASES} cases"
                    ),
                }
            )

    visibility = Counter(
        _visibility_name(
            key_visible=case["key_visible"],
            door_visible=case["door_visible"],
        )
        for case in public_cases
    )
    return {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "kind": QUALIFICATION_KIND,
        "result": "passed" if not failures else "failed",
        "policy_updates": False,
        "attempt_id": access.qualification_attempt_id,
        "claim_id": access.qualification_claim_id,
        "claim_sha256": access.qualification_claim_sha256,
        "started_at": started_at,
        "completed_at": utc_now(),
        "duration_seconds": time.monotonic() - wall_start,
        "source": {"commit": source_commit, "dirty": False},
        "runtime": runtime_snapshot(),
        "generator_profile": GENERATOR_PROFILE,
        "generator_profile_version": GENERATOR_PROFILE_VERSION,
        "lesson_id": LESSON_ID,
        "seed_partition": {
            "role": U2SeedRole.SEALED_QUALIFICATION.value,
            "start": QUALIFICATION_SEED_BASE,
            "end": QUALIFICATION_SEED_END,
            "count": QUALIFICATION_CASES,
        },
        "seed_partition_ledger": list(seed_partition_ledger()),
        "requested": QUALIFICATION_CASES,
        "generated": sealed_generated,
        "solved": sealed_solved,
        "full_unique_layouts": len(full_hashes),
        "geometry_unique_layouts": len(geometry_hashes),
        "contract_counts": {
            field_name: contract_counts[field_name]
            for field_name in sorted(CONTRACT_COUNT_FIELDS)
        },
        "visibility_strata": {name: visibility[name] for name in VISIBILITY_STRATA},
        "validation_disjointness": {
            "u2_validation_start": U2_VALIDATION_SEED_BASE,
            "u2_validation_cases": U2_VALIDATION_CASES,
            "u2_validation_generated": validation_generated,
            "exact_visual_overlap": len(exact_overlap),
        },
        "validation_reference": {
            "generated": validation_generated,
            "unique_exact_layouts": len(validation_hashes),
            "unique_geometries": len(validation_geometry_hashes),
            "exact_layout_set_sha256": _hash_set_sha256(validation_hashes),
            "geometry_set_sha256": _hash_set_sha256(validation_geometry_hashes),
        },
        "cases": public_cases,
        "failures": failures,
    }


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> bytes:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return encoded


def _validate_launcher_token(value: str) -> str:
    if not isinstance(value, str) or not _HEX_DIGEST.fullmatch(value):
        raise U2QualificationError("launcher token must be exactly 256 lowercase hex bits")
    return value


def _begin_canonical_attempt(
    *,
    source_commit: str,
    launcher_token: str,
    started_at: str,
) -> tuple[U2SeedAccess, dict[str, Any]]:
    """Atomically claim the one canonical attempt before issuing sealed access."""

    token = _validate_launcher_token(launcher_token)
    paths = canonical_qualification_paths()
    directory = paths["directory"]
    if directory.is_symlink() or directory.exists():
        raise U2QualificationError(
            f"refusing to reuse canonical U2 qualification directory: {directory}"
        )
    parent = directory.parent
    _reject_symlink_chain(parent)
    if parent.is_symlink() or not parent.is_dir():
        raise U2QualificationError(
            f"canonical qualification parent must already exist: {parent}"
        )
    directory.mkdir(mode=0o755)
    token_sha256 = hashlib.sha256(token.encode("ascii")).hexdigest()
    claim_id = hashlib.sha256(
        (
            f"{token_sha256}\n{source_commit}\n{QUALIFICATION_ATTEMPT_ID}\n"
            f"{directory}\n"
        ).encode()
    ).hexdigest()
    claim = {
        "schema_version": QUALIFICATION_CLAIM_SCHEMA_VERSION,
        "protocol": QUALIFICATION_CLAIM_PROTOCOL,
        "kind": QUALIFICATION_CLAIM_KIND,
        "status": "claimed_for_single_launch",
        "claimed_at": started_at,
        "source": {"commit": source_commit, "dirty": False},
        "attempt_id": QUALIFICATION_ATTEMPT_ID,
        "claim_id": claim_id,
        "qualification_directory": str(directory),
        "report": str(paths["report"]),
        "attempt": str(paths["attempt"]),
        "launcher_token_sha256": token_sha256,
        "acknowledgement_sha256": hashlib.sha256(
            sealed_acknowledgement().encode("utf-8")
        ).hexdigest(),
        "seed_partition": {
            "start": QUALIFICATION_SEED_BASE,
            "end": QUALIFICATION_SEED_END,
            "count": QUALIFICATION_CASES,
        },
    }
    claim_bytes = _exclusive_json(paths["claim"], claim)
    claim_sha256 = _sha256_bytes(claim_bytes)
    attempt = {
        "schema_version": QUALIFICATION_ATTEMPT_SCHEMA_VERSION,
        "protocol": QUALIFICATION_ATTEMPT_PROTOCOL,
        "kind": QUALIFICATION_ATTEMPT_KIND,
        "status": "started",
        "attempt_id": QUALIFICATION_ATTEMPT_ID,
        "claim_id": claim_id,
        "claim_sha256": claim_sha256,
        "started_at": started_at,
        "completed_at": None,
        "source": {"commit": source_commit, "dirty": False},
        "qualification_directory": str(directory),
        "report": str(paths["report"]),
        "claim": str(paths["claim"]),
        "seed_partition": {
            "start": QUALIFICATION_SEED_BASE,
            "end": QUALIFICATION_SEED_END,
            "count": QUALIFICATION_CASES,
        },
        "report_sha256": None,
        "report_result": None,
        "qualifier_exit_status": None,
    }
    _exclusive_json(paths["attempt"], attempt)
    opaque_claim = _new_sealed_launch_claim(
        source_commit=source_commit,
        clean_source=True,
        attempt_id=QUALIFICATION_ATTEMPT_ID,
        claim_id=claim_id,
        claim_sha256=claim_sha256,
        launcher_token_sha256=token_sha256,
    )
    access = _sealed_preflight_seed_access(
        claim=opaque_claim,
        launcher_token=token,
    )
    return access, attempt


def build_parser() -> argparse.ArgumentParser:
    """The production CLI has no seed, count, path, or rerun override."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch-token",
        required=True,
        help="A fresh 256-bit lowercase hex token generated by the canonical launcher",
    )
    parser.add_argument(
        "--acknowledge",
        required=True,
        help="Exact one-shot range acknowledgement printed by the launcher",
    )
    return parser


def _failed_report(
    *,
    source_commit: str,
    access: U2SeedAccess,
    started_at: str,
    wall_start: float,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "kind": QUALIFICATION_KIND,
        "result": "failed",
        "policy_updates": False,
        "attempt_id": access.qualification_attempt_id,
        "claim_id": access.qualification_claim_id,
        "claim_sha256": access.qualification_claim_sha256,
        "started_at": started_at,
        "completed_at": utc_now(),
        "duration_seconds": max(time.monotonic() - wall_start, 1e-9),
        "source": {"commit": source_commit, "dirty": False},
        "runtime": runtime_snapshot(),
        "generator_profile": GENERATOR_PROFILE,
        "generator_profile_version": GENERATOR_PROFILE_VERSION,
        "lesson_id": LESSON_ID,
        "seed_partition": {
            "role": U2SeedRole.SEALED_QUALIFICATION.value,
            "start": QUALIFICATION_SEED_BASE,
            "end": QUALIFICATION_SEED_END,
            "count": QUALIFICATION_CASES,
        },
        "seed_partition_ledger": list(seed_partition_ledger()),
        "requested": QUALIFICATION_CASES,
        "generated": 0,
        "solved": 0,
        "full_unique_layouts": 0,
        "geometry_unique_layouts": 0,
        "contract_counts": {
            field_name: 0 for field_name in sorted(CONTRACT_COUNT_FIELDS)
        },
        "visibility_strata": {name: 0 for name in VISIBILITY_STRATA},
        "validation_disjointness": {
            "u2_validation_start": U2_VALIDATION_SEED_BASE,
            "u2_validation_cases": U2_VALIDATION_CASES,
            "u2_validation_generated": 0,
            "exact_visual_overlap": 0,
        },
        "validation_reference": {
            "generated": 0,
            "unique_exact_layouts": 0,
            "unique_geometries": 0,
            "exact_layout_set_sha256": _hash_set_sha256(set()),
            "geometry_set_sha256": _hash_set_sha256(set()),
        },
        "cases": [],
        "failures": [
            {
                "phase": "qualifier_exception",
                "seed": None,
                "error": f"{type(error).__name__}: {error}",
            }
        ],
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.acknowledge != sealed_acknowledgement():
        raise SystemExit("the exact one-shot U2 sealed-range acknowledgement is required")
    try:
        launcher_token = _validate_launcher_token(args.launch_token)
    except U2QualificationError as error:
        raise SystemExit(str(error)) from error
    repository = Path(__file__).resolve().parents[2]
    source_before = git_snapshot(repository)
    if source_before.get("dirty") is not False or not source_before.get("commit"):
        raise SystemExit("sealed U2 qualification requires a clean committed source")

    started_at = utc_now()
    wall_start = time.monotonic()
    source_commit = str(source_before["commit"])
    try:
        access, attempt = _begin_canonical_attempt(
            source_commit=source_commit,
            launcher_token=launcher_token,
            started_at=started_at,
        )
    except U2QualificationError as error:
        raise SystemExit(str(error)) from error
    paths = canonical_qualification_paths()
    exit_status = 1
    report: dict[str, Any] | None = None
    report_digest: str | None = None
    try:
        report = collect_sealed_qualification(
            access=access,
            source_commit=source_commit,
            progress=lambda message: print(message, flush=True),
        )
        if report["result"] == "passed":
            try:
                _verify_report_schema(
                    report,
                    expected_source_commit=source_commit,
                    expected_attempt_id=QUALIFICATION_ATTEMPT_ID,
                    expected_claim_id=str(access.qualification_claim_id),
                )
            except U2QualificationError as error:
                report["failures"].append(
                    {
                        "phase": "self_verification",
                        "seed": None,
                        "error": str(error),
                    }
                )
                report["result"] = "failed"
                report["completed_at"] = utc_now()
                report["duration_seconds"] = time.monotonic() - wall_start
        source_after = git_snapshot(repository)
        if source_after != source_before:
            report["failures"].append(
                {
                    "phase": "source_integrity",
                    "seed": None,
                    "error": "source changed while the sealed range was open",
                }
            )
            report["result"] = "failed"
            report["completed_at"] = utc_now()
            report["duration_seconds"] = time.monotonic() - wall_start
        atomic_write_json(paths["report"], report)
        report_bytes = _regular_file_bytes(paths["report"], "written qualification report")
        report_digest = _sha256_bytes(report_bytes)
        atomic_write_text(
            paths["checksum"],
            f"{report_digest}  {CANONICAL_REPORT_NAME}\n",
        )
        exit_status = 0 if report["result"] == "passed" else 1
    except BaseException as error:
        report = _failed_report(
            source_commit=source_commit,
            access=access,
            started_at=started_at,
            wall_start=wall_start,
            error=error,
        )
        atomic_write_json(paths["report"], report)
        report_digest = _sha256_bytes(
            _regular_file_bytes(paths["report"], "written failed qualification report")
        )
        atomic_write_text(
            paths["checksum"],
            f"{report_digest}  {CANONICAL_REPORT_NAME}\n",
        )
        print(f"U2 qualification stopped: {error}", flush=True)
        exit_status = 1
    finally:
        attempt.update(
            {
                "status": (
                    "completed_with_report"
                    if report_digest is not None
                    else "failed_without_report"
                ),
                "completed_at": utc_now(),
                "report_sha256": report_digest,
                "report_result": report.get("result") if report is not None else None,
                "qualifier_exit_status": exit_status,
            }
        )
        atomic_write_json(paths["attempt"], attempt)

    print(f"U2 qualification report: {paths['report']}", flush=True)
    print(f"U2 qualification SHA-256: {report_digest}", flush=True)
    print(f"U2 qualification attempt ID: {attempt['attempt_id']}", flush=True)
    print(f"U2 qualification claim ID: {attempt['claim_id']}", flush=True)
    print(f"U2 qualification attempt ledger: {paths['attempt']}", flush=True)
    raise SystemExit(exit_status)


if __name__ == "__main__":
    main()
