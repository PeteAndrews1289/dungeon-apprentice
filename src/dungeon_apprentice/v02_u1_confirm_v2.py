"""Collision-safe, no-update confirmation for the three mastered U1 policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dungeon_apprentice import v02_u1_confirm as frozen
from dungeon_apprentice.artifacts import (
    atomic_write_json,
    file_sha256,
    git_snapshot,
    runtime_snapshot,
    utc_now,
)
from dungeon_apprentice.contracts import PIXEL_SHAPE, STEP_REWARD, SUCCESS_REWARD
from dungeon_apprentice.oracle import DungeonOracle, OracleFailure
from dungeon_apprentice.v02_lessons import (
    LESSON_SPECS,
    PROTOCOL,
    LessonEnv,
    LessonId,
    evaluate_lesson,
    make_pixel_env,
    validation_seeds,
)

CONFIRMATION_PROTOCOL = "dungeon-apprentice-v0.2-u1-confirmation-v2"
CONFIRMATION_SCHEMA_VERSION = 2
CONFIRMATION_CASES = frozen.CONFIRMATION_CASES
CONFIRMATION_PANEL_SIZE = frozen.CONFIRMATION_PANEL_SIZE
CANDIDATE_COUNT = 10_000
U1_CANDIDATE_BASE = 15_050_000
NAVIGATE_CANDIDATE_BASE = 15_060_000
VISIBLE_UNLOCK_CANDIDATE_BASE = 15_070_000
PRIOR_ATTEMPT_SHA256 = (
    "d2fa53308f7488cc08f5ee67b86a125ad91c6ba3790fcd9433c35aab1215b25c"
)
U1_ENGINEERING_CASES = 1_000
U1_GEOMETRY_REQUIRED = 190
U1_PANEL_GEOMETRY_REQUIRED = 95

CANDIDATE_BASES: Mapping[LessonId, int] = {
    LessonId.NAVIGATE: NAVIGATE_CANDIDATE_BASE,
    LessonId.VISIBLE_UNLOCK: VISIBLE_UNLOCK_CANDIDATE_BASE,
    LessonId.LOCAL_UNLOCK: U1_CANDIDATE_BASE,
}
REFERENCE_SEED_PARTITIONS: tuple[dict[str, Any], ...] = (
    {"name": "training", "start": 0, "end": 999_999},
    {
        "name": "u1_engineering_qualification",
        "start": 5_100_000,
        "end": 5_100_999,
    },
    {"name": "navigate_validation", "start": 10_000_000, "end": 10_000_079},
    {"name": "u0_validation", "start": 11_000_000, "end": 11_000_079},
    {"name": "u1_validation", "start": 11_100_000, "end": 11_100_079},
    {
        "name": "prior_u0_navigate_confirmation",
        "start": 15_000_000,
        "end": 15_000_199,
    },
    {
        "name": "prior_u0_visible_unlock_confirmation",
        "start": 15_010_000,
        "end": 15_010_199,
    },
    {
        "name": "u1_confirmation_v1_local_unlock",
        "start": frozen.U1_CONFIRMATION_BASE,
        "end": frozen.U1_CONFIRMATION_BASE + CONFIRMATION_CASES - 1,
    },
    {
        "name": "u1_confirmation_v1_navigate",
        "start": frozen.NAVIGATE_CONFIRMATION_BASE,
        "end": frozen.NAVIGATE_CONFIRMATION_BASE + CONFIRMATION_CASES - 1,
    },
    {
        "name": "u1_confirmation_v1_visible_unlock",
        "start": frozen.VISIBLE_UNLOCK_CONFIRMATION_BASE,
        "end": frozen.VISIBLE_UNLOCK_CONFIRMATION_BASE + CONFIRMATION_CASES - 1,
    },
)


class U1ConfirmationV2Error(RuntimeError):
    """Raised when evidence cannot support the collision-safe confirmation."""


class CandidateRejected(RuntimeError):
    """A generated candidate failed one declared pre-policy contract."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class LessonReferenceExclusions:
    """Exact-layout exclusions applied before a candidate can be accepted."""

    validation_exact: frozenset[str]
    prior_attempt_exact: frozenset[str]
    engineering_qualification_exact: frozenset[str] = frozenset()
    frozen_child_history_exact: frozenset[str] = frozenset()
    evidence: Mapping[str, Any] | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "validation_exact": _hash_set_evidence(self.validation_exact),
            "prior_attempt_exact": _hash_set_evidence(self.prior_attempt_exact),
            "engineering_qualification_exact": _hash_set_evidence(
                self.engineering_qualification_exact
            ),
            "frozen_child_history_exact": _hash_set_evidence(
                self.frozen_child_history_exact
            ),
            "sources": dict(self.evidence or {}),
        }


def candidate_seeds(lesson: LessonId) -> range:
    """Return the complete, immutable 10,000-seed candidate population."""

    base = CANDIDATE_BASES[LessonId(lesson)]
    return range(base, base + CANDIDATE_COUNT)


def candidate_partition_audit() -> dict[str, Any]:
    """Prove every candidate population is disjoint from every prior seed role."""

    partitions = [
        {
            "lesson_id": lesson.value,
            "start": seeds.start,
            "end": seeds.stop - 1,
            "candidates": len(seeds),
        }
        for lesson in LessonId
        for seeds in (candidate_seeds(lesson),)
    ]
    collisions: list[dict[str, Any]] = []
    pairwise_checks: list[dict[str, Any]] = []
    for index, left in enumerate(partitions):
        comparators = [
            *partitions[index + 1 :],
            *REFERENCE_SEED_PARTITIONS,
        ]
        for right in comparators:
            overlap_start = max(int(left["start"]), int(right["start"]))
            overlap_end = min(int(left["end"]), int(right["end"]))
            overlapping = overlap_start <= overlap_end
            check = {
                "candidate_lesson": left["lesson_id"],
                "reference": right.get("lesson_id", right.get("name")),
                "overlap": overlapping,
                "overlap_start": overlap_start if overlapping else None,
                "overlap_end": overlap_end if overlapping else None,
            }
            pairwise_checks.append(check)
            if overlapping:
                collisions.append(
                    {
                        "left": left["lesson_id"],
                        "right": right.get("lesson_id", right.get("name")),
                        "start": overlap_start,
                        "end": overlap_end,
                    }
                )
    for partition in partitions:
        if int(partition["end"]) >= 20_000_000:
            collisions.append(
                {
                    "left": partition["lesson_id"],
                    "right": "untouched_final_boundary",
                    "start": 20_000_000,
                    "end": partition["end"],
                }
            )
        if int(partition["candidates"]) != CANDIDATE_COUNT:
            collisions.append(
                {
                    "left": partition["lesson_id"],
                    "right": "candidate_count_contract",
                    "found": partition["candidates"],
                    "required": CANDIDATE_COUNT,
                }
            )
    return {
        "candidate_count_per_lesson": CANDIDATE_COUNT,
        "partitions": partitions,
        "reference_partitions": [dict(partition) for partition in REFERENCE_SEED_PARTITIONS],
        "pairwise_checks": pairwise_checks,
        "untouched_final_boundary": 20_000_000,
        "collisions": collisions,
        "passed": not collisions,
    }


def _hash_set_sha256(hashes: Sequence[str] | set[str] | frozenset[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in hashes):
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _hash_set_evidence(hashes: set[str] | frozenset[str]) -> dict[str, Any]:
    return {
        "unique_layouts": len(hashes),
        "set_sha256": _hash_set_sha256(hashes),
    }


def _seed_list_sha256(seeds: Sequence[int]) -> str:
    encoded = json.dumps(
        [int(seed) for seed in seeds],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(value: Any, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise U1ConfirmationV2Error(f"{description} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise U1ConfirmationV2Error(f"{description} is not a SHA-256 digest") from error
    return value


def load_prior_attempt(
    path: Path,
) -> tuple[
    dict[str, Any],
    dict[LessonId, frozenset[str]],
    dict[LessonId, frozenset[str]],
]:
    """Authenticate attempt 1 and recover every exact layout it exposed."""

    report_path = path.expanduser().resolve()
    if not report_path.is_file():
        raise U1ConfirmationV2Error(f"missing immutable attempt-1 report: {report_path}")
    measured = file_sha256(report_path)
    if measured != PRIOR_ATTEMPT_SHA256:
        raise U1ConfirmationV2Error(
            "attempt-1 report digest changed: "
            f"{measured} != {PRIOR_ATTEMPT_SHA256}"
        )
    report = frozen._read_json(report_path, "attempt-1 U1 confirmation report")
    if (
        report.get("protocol") != frozen.CONFIRMATION_PROTOCOL
        or report.get("checkpoint_scoring_performed") is not False
        or report.get("verdict") != "qualification_failed"
    ):
        raise U1ConfirmationV2Error(
            "attempt-1 report is not the frozen, unscored qualification failure"
        )
    qualifications = report.get("layout_qualification")
    if not isinstance(qualifications, list):
        raise U1ConfirmationV2Error("attempt-1 report has no layout qualifications")
    by_lesson: dict[LessonId, frozenset[str]] = {}
    geometry_by_lesson: dict[LessonId, frozenset[str]] = {}
    for qualification in qualifications:
        if not isinstance(qualification, Mapping):
            raise U1ConfirmationV2Error("attempt-1 qualification is not an object")
        try:
            lesson = LessonId(str(qualification["lesson_id"]))
        except (KeyError, ValueError) as error:
            raise U1ConfirmationV2Error(
                "attempt-1 qualification has an invalid lesson"
            ) from error
        if lesson in by_lesson:
            raise U1ConfirmationV2Error(
                f"attempt-1 report repeats lesson {lesson.value}"
            )
        cases = qualification.get("cases")
        if not isinstance(cases, list) or len(cases) != CONFIRMATION_CASES:
            raise U1ConfirmationV2Error(
                f"attempt-1 {lesson.value} qualification must contain "
                f"{CONFIRMATION_CASES} cases"
            )
        expected_seeds = frozen.confirmation_seeds(lesson)
        measured_seeds: list[int] = []
        hashes: set[str] = set()
        geometry_hashes: set[str] = set()
        for case in cases:
            if not isinstance(case, Mapping):
                raise U1ConfirmationV2Error(
                    f"attempt-1 {lesson.value} contains an invalid case"
                )
            measured_seeds.append(int(case.get("seed", -1)))
            hashes.add(
                _require_hash(
                    case.get("layout_sha256"),
                    f"attempt-1 {lesson.value} layout",
                )
            )
            geometry_hashes.add(
                _require_hash(
                    case.get("geometry_sha256"),
                    f"attempt-1 {lesson.value} geometry",
                )
            )
        if tuple(measured_seeds) != expected_seeds:
            raise U1ConfirmationV2Error(
                f"attempt-1 {lesson.value} seeds differ from its frozen block"
            )
        by_lesson[lesson] = frozenset(hashes)
        geometry_by_lesson[lesson] = frozenset(geometry_hashes)
    if set(by_lesson) != set(LessonId):
        raise U1ConfirmationV2Error("attempt-1 report does not contain all three lessons")
    return (
        {
            "path": str(report_path),
            "sha256": measured,
            "protocol": report["protocol"],
            "verdict": report["verdict"],
            "checkpoint_scoring_performed": False,
            "exact_layouts": {
                lesson.value: _hash_set_evidence(hashes)
                for lesson, hashes in by_lesson.items()
            },
            "protocol_independent_geometries": {
                lesson.value: _hash_set_evidence(hashes)
                for lesson, hashes in geometry_by_lesson.items()
            },
        },
        by_lesson,
        geometry_by_lesson,
    )


def _validation_exact_hashes(lesson: LessonId) -> frozenset[str]:
    cases = frozen._fixed_geometry_cases(lesson, validation_seeds(lesson))
    if len(cases) != 80:
        raise U1ConfirmationV2Error(
            f"{lesson.value} validation reference does not contain 80 cases"
        )
    return frozenset(
        _require_hash(case.get("layout_sha256"), f"{lesson.value} validation layout")
        for case in cases
    )


def _engineering_qualification_hashes(
    verified: Sequence[frozen.VerifiedU1Checkpoint],
) -> tuple[frozenset[str], list[dict[str, Any]]]:
    union: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for checkpoint in verified:
        path = Path(checkpoint.manifest).parent / "qualification.json"
        report = frozen._read_json(path, "frozen U1 engineering qualification")
        cases = report.get("cases")
        if (
            report.get("protocol") != PROTOCOL
            or report.get("lesson_id") != LessonId.LOCAL_UNLOCK.value
            or not isinstance(cases, list)
            or len(cases) != U1_ENGINEERING_CASES
        ):
            raise U1ConfirmationV2Error(
                f"invalid frozen U1 engineering qualification: {path}"
            )
        hashes = frozenset(
            _require_hash(case.get("layout_sha256"), "engineering qualification layout")
            for case in cases
            if isinstance(case, Mapping)
        )
        if len(cases) != sum(isinstance(case, Mapping) for case in cases):
            raise U1ConfirmationV2Error(
                f"invalid case in frozen U1 engineering qualification: {path}"
            )
        union.update(hashes)
        evidence.append(
            {
                "child_algorithm_seed": checkpoint.child_algorithm_seed,
                "path": str(path),
                "file_sha256": file_sha256(path),
                **_hash_set_evidence(hashes),
            }
        )
    return frozenset(union), evidence


def _frozen_child_u1_history_hashes(
    verified: Sequence[frozen.VerifiedU1Checkpoint],
) -> tuple[frozenset[str], list[dict[str, Any]]]:
    union: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for checkpoint in verified:
        path = Path(checkpoint.manifest).parent / "episodes.jsonl"
        records = frozen._read_episode_layouts(path)[LessonId.LOCAL_UNLOCK]
        hashes = frozenset(
            _require_hash(record.get("layout_sha256"), "frozen child U1 history layout")
            for record in records
        )
        union.update(hashes)
        evidence.append(
            {
                "child_algorithm_seed": checkpoint.child_algorithm_seed,
                "path": str(path),
                "file_sha256": file_sha256(path),
                "episodes": len(records),
                **_hash_set_evidence(hashes),
            }
        )
    return frozenset(union), evidence


def build_reference_exclusions(
    verified: Sequence[frozen.VerifiedU1Checkpoint],
    *,
    prior_attempt_report: Path,
) -> tuple[
    dict[LessonId, LessonReferenceExclusions],
    dict[str, Any],
    dict[LessonId, frozenset[str]],
]:
    """Build all declared exact-layout exclusions before candidate generation."""

    prior_evidence, prior_hashes, prior_geometries = load_prior_attempt(
        prior_attempt_report
    )
    validation = {lesson: _validation_exact_hashes(lesson) for lesson in LessonId}
    engineering, engineering_evidence = _engineering_qualification_hashes(verified)
    child_history, child_evidence = _frozen_child_u1_history_hashes(verified)
    exclusions = {
        lesson: LessonReferenceExclusions(
            validation_exact=validation[lesson],
            prior_attempt_exact=prior_hashes[lesson],
            engineering_qualification_exact=(
                engineering if lesson is LessonId.LOCAL_UNLOCK else frozenset()
            ),
            frozen_child_history_exact=(
                child_history if lesson is LessonId.LOCAL_UNLOCK else frozenset()
            ),
            evidence={
                "validation_cases": 80,
                "validation_seed_sha256": _seed_list_sha256(validation_seeds(lesson)),
            },
        )
        for lesson in LessonId
    }
    return (
        exclusions,
        {
            "prior_attempt": prior_evidence,
            "validation": {
                lesson.value: _hash_set_evidence(hashes)
                for lesson, hashes in validation.items()
            },
            "u1_engineering_qualification": {
                "union": _hash_set_evidence(engineering),
                "lineages": engineering_evidence,
            },
            "u1_frozen_child_history": {
                "union": _hash_set_evidence(child_history),
                "lineages": child_evidence,
            },
            "anchor_training_overlap_affects_verdict": False,
            "u1_training_overlap_affects_verdict": True,
        },
        prior_geometries,
    )


def _reject(code: str, detail: str) -> None:
    raise CandidateRejected(code, detail)


def _inspect_candidate(lesson: LessonId, seed: int) -> dict[str, Any]:
    """Generate and mechanically qualify one candidate without loading a policy."""

    lesson = LessonId(lesson)
    wrapped = make_pixel_env(lesson=lesson, size=9)
    env = wrapped.unwrapped
    try:
        if not isinstance(env, LessonEnv):
            _reject("environment_contract", "pixel wrapper does not contain LessonEnv")
        observation, info = wrapped.reset(seed=int(seed))
        if env.layout is None:
            _reject("generation_contract", "candidate has no layout metadata")
        layout = env.layout
        if observation.shape != PIXEL_SHAPE or observation.dtype != np.uint8:
            _reject(
                "observation_contract",
                f"expected {PIXEL_SHAPE}/uint8, found {observation.shape}/{observation.dtype}",
            )
        if info.get("lesson_id") != lesson.value:
            _reject("evidence_contract", "reset evidence has the wrong lesson ID")
        key_visible = env.agent_sees(*layout.key) if layout.key is not None else None
        door_visible = env.agent_sees(*layout.door) if layout.door is not None else None
        spec = LESSON_SPECS[lesson]
        if spec.key_visible is not None and key_visible is not spec.key_visible:
            _reject(
                "visibility_contract",
                f"key visibility {key_visible} != {spec.key_visible}",
            )
        if spec.door_visible is not None and door_visible is not spec.door_visible:
            _reject(
                "visibility_contract",
                f"door visibility {door_visible} != {spec.door_visible}",
            )
        result = DungeonOracle(env).solve()
        if not result.success or result.steps != len(result.actions):
            _reject("oracle_contract", "oracle did not produce a complete successful plan")
        if not 0 < result.steps <= spec.max_steps:
            _reject(
                "action_length_contract",
                f"oracle length {result.steps} outside 1..{spec.max_steps}",
            )
        if spec.oracle_action_range is not None:
            minimum, maximum = spec.oracle_action_range
            if not minimum <= result.steps <= maximum:
                _reject(
                    "action_length_contract",
                    f"oracle length {result.steps} outside {minimum}..{maximum}",
                )
        if lesson is LessonId.LOCAL_UNLOCK and (
            env.pure_oracle_actions is None
            or len(env.pure_oracle_actions) != result.steps
        ):
            _reject(
                "oracle_contract",
                "pure and live Local Unlock oracle lengths differ",
            )
        expected_reward = STEP_REWARD * (result.steps - 1) + SUCCESS_REWARD
        if not math.isclose(result.total_reward, expected_reward, abs_tol=1e-9):
            _reject(
                "reward_contract",
                f"oracle reward {result.total_reward} != {expected_reward}",
            )
        try:
            layout_hash = _require_hash(result.layout_sha256, "candidate layout")
        except U1ConfirmationV2Error as error:
            _reject("layout_contract", str(error))
        if layout_hash != layout.layout_sha256:
            _reject("layout_contract", "oracle and layout hashes differ")
        try:
            geometry_hash = _require_hash(env.geometry_sha256, "candidate geometry")
        except U1ConfirmationV2Error as error:
            _reject("geometry_contract", str(error))
        return {
            "seed": int(seed),
            "layout_sha256": layout_hash,
            "geometry_sha256": geometry_hash,
            "oracle_actions": result.steps,
            "visibility": {
                "key_visible": key_visible,
                "door_visible": door_visible,
            },
            "contracts": {
                "mechanically_solved": True,
                "observation": True,
                "visibility": True,
                "reward": True,
                "action_length": True,
            },
        }
    except CandidateRejected:
        raise
    except (AssertionError, OracleFailure, RuntimeError, ValueError) as error:
        raise CandidateRejected(
            "generation_or_oracle_failure",
            str(error),
        ) from error
    finally:
        wrapped.close()


def _reference_rejection_reasons(
    lesson: LessonId,
    layout_hash: str,
    exclusions: LessonReferenceExclusions,
    accepted_hashes: set[str],
) -> list[str]:
    reasons: list[str] = []
    if layout_hash in exclusions.validation_exact:
        reasons.append("validation_exact_overlap")
    if layout_hash in exclusions.prior_attempt_exact:
        reasons.append("prior_attempt_exact_overlap")
    if (
        lesson is LessonId.LOCAL_UNLOCK
        and layout_hash in exclusions.engineering_qualification_exact
    ):
        reasons.append("engineering_qualification_exact_overlap")
    if (
        lesson is LessonId.LOCAL_UNLOCK
        and layout_hash in exclusions.frozen_child_history_exact
    ):
        reasons.append("frozen_child_history_exact_overlap")
    if layout_hash in accepted_hashes:
        reasons.append("duplicate_accepted_exact_layout")
    return reasons


def _panel_summary(
    cases: Sequence[Mapping[str, Any]],
    *,
    geometry_affects_verdict: bool,
) -> list[dict[str, Any]]:
    panels: list[dict[str, Any]] = []
    for panel_index in range(2):
        start = panel_index * CONFIRMATION_PANEL_SIZE
        panel = list(cases[start : start + CONFIRMATION_PANEL_SIZE])
        exact_unique = len({str(case["layout_sha256"]) for case in panel})
        geometry_unique = len({str(case["geometry_sha256"]) for case in panel})
        panels.append(
            {
                "panel": panel_index,
                "panel_label": "A" if panel_index == 0 else "B",
                "cases": len(panel),
                "required_cases": CONFIRMATION_PANEL_SIZE,
                "exact_unique_layouts": exact_unique,
                "required_exact_unique_layouts": CONFIRMATION_PANEL_SIZE,
                "geometry_unique_layouts": geometry_unique,
                "required_geometry_unique_layouts": (
                    U1_PANEL_GEOMETRY_REQUIRED if geometry_affects_verdict else None
                ),
                "geometry_affects_verdict": geometry_affects_verdict,
                "exact_passed": exact_unique == CONFIRMATION_PANEL_SIZE,
                "geometry_passed": (
                    geometry_unique >= U1_PANEL_GEOMETRY_REQUIRED
                    if geometry_affects_verdict
                    else None
                ),
                "passed": (
                    len(panel) == CONFIRMATION_PANEL_SIZE
                    and exact_unique == CONFIRMATION_PANEL_SIZE
                    and (
                        geometry_unique >= U1_PANEL_GEOMETRY_REQUIRED
                        if geometry_affects_verdict
                        else True
                    )
                ),
            }
        )
    return panels


def _selection_result(
    lesson: LessonId,
    accepted: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    *,
    candidates_examined: int,
    exclusions: LessonReferenceExclusions,
) -> dict[str, Any]:
    geometry_affects_verdict = lesson is LessonId.LOCAL_UNLOCK
    accepted_hashes = {str(case["layout_sha256"]) for case in accepted}
    exact_unique = len(accepted_hashes)
    geometry_unique = len({str(case["geometry_sha256"]) for case in accepted})
    reference_sets = {
        "validation_exact": exclusions.validation_exact,
        "prior_attempt_exact": exclusions.prior_attempt_exact,
    }
    if lesson is LessonId.LOCAL_UNLOCK:
        reference_sets |= {
            "engineering_qualification_exact": (
                exclusions.engineering_qualification_exact
            ),
            "frozen_child_history_exact": exclusions.frozen_child_history_exact,
        }
    accepted_reference_overlap = {
        name: {
            "overlap_unique_layouts": len(accepted_hashes & hashes),
            "required_overlap_unique_layouts": 0,
            "passed": not (accepted_hashes & hashes),
        }
        for name, hashes in reference_sets.items()
    }
    reference_disjointness_passed = all(
        item["passed"] for item in accepted_reference_overlap.values()
    )
    panels = _panel_summary(
        accepted,
        geometry_affects_verdict=geometry_affects_verdict,
    )
    passed = (
        len(accepted) == CONFIRMATION_CASES
        and exact_unique == CONFIRMATION_CASES
        and reference_disjointness_passed
        and all(panel["passed"] for panel in panels)
        and (
            geometry_unique >= U1_GEOMETRY_REQUIRED
            if geometry_affects_verdict
            else True
        )
    )
    primary_reason_counts = Counter(str(item["reason"]) for item in rejected)
    reason_counts = Counter(
        str(reason)
        for item in rejected
        for reason in item.get("reasons", (item["reason"],))
    )
    seeds = [int(case["seed"]) for case in accepted]
    return {
        "protocol": CONFIRMATION_PROTOCOL,
        "lesson_id": lesson.value,
        "candidate_partition": {
            "start": candidate_seeds(lesson).start,
            "end": candidate_seeds(lesson).stop - 1,
            "candidates": CANDIDATE_COUNT,
        },
        "selection_order": "ascending_seed",
        "acceptance_target": CONFIRMATION_CASES,
        "candidates_examined": candidates_examined,
        "uninspected_candidates": CANDIDATE_COUNT - candidates_examined,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_seed_sha256": _seed_list_sha256(seeds),
        "full_unique_layouts": exact_unique,
        "required_full_unique_layouts": CONFIRMATION_CASES,
        "geometry_unique_layouts": geometry_unique,
        "required_geometry_unique_layouts": (
            U1_GEOMETRY_REQUIRED if geometry_affects_verdict else None
        ),
        "geometry_affects_verdict": geometry_affects_verdict,
        "panel_qualification": panels,
        "exclusions": exclusions.public_dict(),
        "accepted_vs_exclusion_overlap": accepted_reference_overlap,
        "accepted_reference_disjointness_passed": reference_disjointness_passed,
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "rejection_primary_reason_counts": dict(sorted(primary_reason_counts.items())),
        "cases": list(accepted),
        "rejections": list(rejected),
        "failures": (
            []
            if passed
            else [
                {
                    "error": (
                        f"accepted {len(accepted)}/{CONFIRMATION_CASES} after "
                        f"examining {candidates_examined}/{CANDIDATE_COUNT} candidates"
                    )
                }
            ]
        ),
        "result": "passed" if passed else "failed",
    }


def select_confirmation_block(
    lesson: LessonId,
    exclusions: LessonReferenceExclusions,
) -> dict[str, Any]:
    """Select the first 200 eligible cases in deterministic ascending seed order."""

    lesson = LessonId(lesson)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted_hashes: set[str] = set()
    examined = 0
    for seed in candidate_seeds(lesson):
        examined += 1
        try:
            candidate = _inspect_candidate(lesson, seed)
        except CandidateRejected as error:
            rejected.append(
                {
                    "seed": int(seed),
                    "status": "rejected",
                    "reason": error.code,
                    "reasons": [error.code],
                    "detail": error.detail,
                }
            )
            continue
        layout_hash = _require_hash(candidate.get("layout_sha256"), "candidate layout")
        reasons = _reference_rejection_reasons(
            lesson,
            layout_hash,
            exclusions,
            accepted_hashes,
        )
        if reasons:
            rejected.append(
                dict(candidate)
                | {
                    "seed": int(seed),
                    "status": "rejected",
                    "reason": reasons[0],
                    "reasons": reasons,
                }
            )
            continue
        accepted_index = len(accepted)
        accepted_case = dict(candidate) | {
            "status": "accepted",
            "accepted_index": accepted_index,
            "panel": accepted_index // CONFIRMATION_PANEL_SIZE,
            "panel_label": (
                "A" if accepted_index < CONFIRMATION_PANEL_SIZE else "B"
            ),
            "reference_contracts": {
                "validation_exact_novel": True,
                "prior_attempt_exact_novel": True,
                "engineering_qualification_exact_novel": (
                    True if lesson is LessonId.LOCAL_UNLOCK else None
                ),
                "frozen_child_history_exact_novel": (
                    True if lesson is LessonId.LOCAL_UNLOCK else None
                ),
                "accepted_set_exact_unique": True,
            },
        }
        accepted.append(accepted_case)
        accepted_hashes.add(layout_hash)
        if len(accepted) == CONFIRMATION_CASES:
            break
    return _selection_result(
        lesson,
        accepted,
        rejected,
        candidates_examined=examined,
        exclusions=exclusions,
    )


def selected_seed_lists(
    selections: Sequence[Mapping[str, Any]],
) -> dict[LessonId, tuple[int, ...]]:
    """Recover and verify the exact ordered seed lists used for policy scoring."""

    result: dict[LessonId, tuple[int, ...]] = {}
    for selection in selections:
        lesson = LessonId(str(selection["lesson_id"]))
        if lesson in result:
            raise U1ConfirmationV2Error(f"selection repeats lesson {lesson.value}")
        cases = selection.get("cases")
        if selection.get("result") != "passed" or not isinstance(cases, list):
            raise U1ConfirmationV2Error(
                f"cannot score failed selection for {lesson.value}"
            )
        seeds = tuple(int(case["seed"]) for case in cases)
        if (
            len(seeds) != CONFIRMATION_CASES
            or len(set(seeds)) != CONFIRMATION_CASES
            or _seed_list_sha256(seeds) != selection.get("accepted_seed_sha256")
        ):
            raise U1ConfirmationV2Error(
                f"selected seed evidence is invalid for {lesson.value}"
            )
        result[lesson] = seeds
    if set(result) != set(LessonId):
        raise U1ConfirmationV2Error("selected seed evidence is missing a lesson")
    return result


def attempt1_geometry_diagnostics(
    selections: Sequence[Mapping[str, Any]],
    prior_geometries: Mapping[LessonId, frozenset[str]],
) -> dict[str, Any]:
    """Compare v2 and authenticated v1 geometry across every lesson pairing."""

    selected: dict[LessonId, frozenset[str]] = {}
    for selection in selections:
        lesson = LessonId(str(selection["lesson_id"]))
        cases = selection.get("cases")
        if not isinstance(cases, list) or len(cases) != CONFIRMATION_CASES:
            raise U1ConfirmationV2Error(
                f"cannot compare incomplete geometry for {lesson.value}"
            )
        selected[lesson] = frozenset(
            _require_hash(case.get("geometry_sha256"), "selected candidate geometry")
            for case in cases
            if isinstance(case, Mapping)
        )
        if len(cases) != sum(isinstance(case, Mapping) for case in cases):
            raise U1ConfirmationV2Error(
                f"selected candidate geometry is invalid for {lesson.value}"
            )
    if set(selected) != set(LessonId) or set(prior_geometries) != set(LessonId):
        raise U1ConfirmationV2Error(
            "v1-v2 geometry comparison requires all three lessons"
        )
    matrix: dict[str, dict[str, dict[str, int]]] = {}
    for selected_lesson, selected_hashes in selected.items():
        row: dict[str, dict[str, int]] = {}
        for prior_lesson, prior_hashes in prior_geometries.items():
            row[prior_lesson.value] = {
                "v2_unique_geometries": len(selected_hashes),
                "attempt1_unique_geometries": len(prior_hashes),
                "overlap_unique_geometries": len(selected_hashes & prior_hashes),
            }
        matrix[selected_lesson.value] = row
    return {
        "metric": "protocol-independent geometry SHA-256",
        "attempt1_report_sha256": PRIOR_ATTEMPT_SHA256,
        "geometry_overlap_affects_verdict": False,
        "matrix": matrix,
    }


def _evaluate_selected_checkpoint(
    verified: frozen.VerifiedU1Checkpoint,
    model_type: Any,
    seeds_by_lesson: Mapping[LessonId, Sequence[int]],
) -> dict[str, Any]:
    """Score the exact accepted cases while proving all learned state is unchanged."""

    archive = Path(verified.checkpoint)
    archive_digest_before = file_sha256(archive)
    if archive_digest_before != verified.checkpoint_sha256:
        raise U1ConfirmationV2Error(
            "checkpoint archive changed before policy evaluation"
        )
    model = model_type.load(verified.checkpoint, device="cpu")
    frozen._verify_loaded_model_contract(model)
    before_policy = frozen.policy_tensor_sha256(model)
    before_optimizer = frozen.optimizer_state_sha256(model)
    before_timesteps = int(model.num_timesteps)
    before_updates = int(model._n_updates)
    if before_timesteps != verified.trained_timesteps or before_updates != verified.n_updates:
        raise U1ConfirmationV2Error(
            "loaded model counters disagree with the verified mastery sidecar"
        )

    import torch

    evaluations: list[dict[str, Any]] = []
    with torch.inference_mode():
        for lesson in LessonId:
            seeds = tuple(int(seed) for seed in seeds_by_lesson[lesson])
            if len(seeds) != CONFIRMATION_CASES:
                raise U1ConfirmationV2Error(
                    f"{lesson.value} evaluation does not have {CONFIRMATION_CASES} seeds"
                )
            result = evaluate_lesson(model, lesson, seeds, size=9)
            evaluations.append(
                result.public_dict()
                | {
                    "seeds": list(seeds),
                    "seed_list_sha256": _seed_list_sha256(seeds),
                    "gate": frozen.grade_evaluation(result),
                    "milestone_evidence": frozen.milestone_evidence(result),
                }
            )

    after_policy = frozen.policy_tensor_sha256(model)
    after_optimizer = frozen.optimizer_state_sha256(model)
    after_timesteps = int(model.num_timesteps)
    after_updates = int(model._n_updates)
    archive_digest_after = file_sha256(archive)
    if (
        after_policy != before_policy
        or after_optimizer != before_optimizer
        or after_timesteps != before_timesteps
        or after_updates != before_updates
    ):
        raise U1ConfirmationV2Error(
            "confirmation evaluation changed policy or optimizer counters"
        )
    if archive_digest_after != archive_digest_before:
        raise U1ConfirmationV2Error(
            "checkpoint archive changed during policy evaluation"
        )
    return {
        "checkpoint": asdict(verified),
        "integrity": {
            "passed": True,
            "archive_digest_verified": verified.digest_verified,
            "source_verified": verified.source_verified,
            "lineage_sidecars_and_allocation_verified": True,
            "loaded_model_contract_verified": True,
        },
        "policy_tensor_sha256_before": before_policy,
        "policy_tensor_sha256_after": after_policy,
        "optimizer_state_sha256_before": before_optimizer,
        "optimizer_state_sha256_after": after_optimizer,
        "checkpoint_sha256_before_load": archive_digest_before,
        "checkpoint_sha256_after_evaluation": archive_digest_after,
        "model_num_timesteps_before": before_timesteps,
        "model_num_timesteps_after": after_timesteps,
        "model_updates_before": before_updates,
        "model_updates_after": after_updates,
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


def _terminal_qualification_report(
    base_report: Mapping[str, Any],
    *,
    wall_started: float,
    selections: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return dict(base_report) | {
        "completed_at": utc_now(),
        "elapsed_seconds": time.monotonic() - wall_started,
        "layout_qualification": list(selections),
        "collision_diagnostics": diagnostics,
        "checkpoint_scoring_performed": False,
        "checkpoints": [],
        "verdict": "qualification_failed",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="selected mastered-local-unlock.zip; provide exactly three",
    )
    parser.add_argument(
        "--prior-attempt-report",
        type=Path,
        required=True,
        help="immutable report.json from U1 confirmation attempt 1",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(args.checkpoint) != len(frozen.EXPECTED_CHILD_SEEDS):
        raise SystemExit("U1 confirmation v2 requires exactly three selected checkpoints")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite U1 confirmation v2 report: {output}")
    try:
        repository = Path(__file__).resolve().parents[2]
        source = git_snapshot(repository)
        if source.get("dirty") is not False:
            raise U1ConfirmationV2Error(
                "U1 confirmation v2 requires a clean committed source tree"
            )
        started_at = utc_now()
        wall_started = time.monotonic()
        partition_audit = candidate_partition_audit()
        if not partition_audit["passed"]:
            raise U1ConfirmationV2Error(
                f"candidate partitions overlap: {partition_audit['collisions']}"
            )

        verified = [frozen.verify_u1_checkpoint(path) for path in args.checkpoint]
        child_seeds = [item.child_algorithm_seed for item in verified]
        parent_seeds = [item.parent_training_seed for item in verified]
        if sorted(child_seeds) != sorted(frozen.EXPECTED_CHILD_SEEDS):
            raise U1ConfirmationV2Error(
                f"selected child seeds must be {frozen.EXPECTED_CHILD_SEEDS}, "
                f"found {child_seeds}"
            )
        if sorted(parent_seeds) != sorted(frozen.EXPECTED_PARENT_SEEDS):
            raise U1ConfirmationV2Error(
                f"selected parent seeds must be {frozen.EXPECTED_PARENT_SEEDS}, "
                f"found {parent_seeds}"
            )
        if len({item.checkpoint_sha256 for item in verified}) != len(verified):
            raise U1ConfirmationV2Error(
                "selected U1 checkpoints are not distinct archives"
            )
        verified.sort(key=lambda item: item.child_algorithm_seed)

        exclusions, reference_evidence, prior_geometries = build_reference_exclusions(
            verified,
            prior_attempt_report=args.prior_attempt_report,
        )
        base_report = {
            "schema_version": CONFIRMATION_SCHEMA_VERSION,
            "protocol": CONFIRMATION_PROTOCOL,
            "started_at": started_at,
            "source": source,
            "runtime": runtime_snapshot(),
            "report_path": str(output),
            "policy_updates": False,
            "online_model_calls": False,
            "action_selection": "deterministic",
            "recurrent_state_reset_each_case": True,
            "evaluation_settings": {
                "device": "cpu",
                "environment_size": 9,
                "deterministic_actions": True,
                "recurrent_state_reset_each_case": True,
                "curiosity_enabled": False,
                "inference_mode": True,
            },
            "interruption": None,
            "selected_child_seeds": list(frozen.EXPECTED_CHILD_SEEDS),
            "selected_parent_seeds": list(frozen.EXPECTED_PARENT_SEEDS),
            "candidate_partition_audit": partition_audit,
            "reference_exclusions": reference_evidence,
        }
        selections = [
            select_confirmation_block(lesson, exclusions[lesson])
            for lesson in LessonId
        ]
        if not all(selection["result"] == "passed" for selection in selections):
            report = _terminal_qualification_report(
                base_report,
                wall_started=wall_started,
                selections=selections,
                diagnostics=None,
            )
            atomic_write_json(output, report)
            sys.stdout.write(f"{report['verdict']}: {output}\n")
            raise SystemExit(1)

        diagnostics = dict(frozen.collision_diagnostics(selections, verified)) | {
            "attempt1_protocol_independent_geometry": (
                attempt1_geometry_diagnostics(selections, prior_geometries)
            )
        }
        if not diagnostics["passed"]:
            report = _terminal_qualification_report(
                base_report,
                wall_started=wall_started,
                selections=selections,
                diagnostics=diagnostics,
            )
            atomic_write_json(output, report)
            sys.stdout.write(f"{report['verdict']}: {output}\n")
            raise SystemExit(1)
        seeds_by_lesson = selected_seed_lists(selections)

        try:
            from sb3_contrib import RecurrentPPO
        except ImportError as error:
            raise U1ConfirmationV2Error(
                'install training dependencies with: pip install -e ".[train]"'
            ) from error

        checkpoints = [
            _evaluate_selected_checkpoint(item, RecurrentPPO, seeds_by_lesson)
            for item in verified
        ]
        if len(
            {item["policy_tensor_sha256_before"] for item in checkpoints}
        ) != len(checkpoints):
            raise U1ConfirmationV2Error(
                "selected checkpoints do not contain three distinct policy tensors"
            )
        report = base_report | {
            "completed_at": utc_now(),
            "elapsed_seconds": time.monotonic() - wall_started,
            "layout_qualification": selections,
            "collision_diagnostics": diagnostics,
            "checkpoint_scoring_performed": True,
            "checkpoints": checkpoints,
            "verdict": (
                "confirmed"
                if all(item["passed"] for item in checkpoints)
                else "capability_failed"
            ),
        }
        atomic_write_json(output, report)
    except (frozen.U1ConfirmationError, U1ConfirmationV2Error) as error:
        raise SystemExit(str(error)) from error
    sys.stdout.write(f"{report['verdict']}: {output}\n")
    if report["verdict"] != "confirmed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
