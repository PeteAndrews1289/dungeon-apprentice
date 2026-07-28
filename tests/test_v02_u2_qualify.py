from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dungeon_apprentice import v02_u2_qualify as qualification
from dungeon_apprentice.u2_seed_guard import U2AccessPhase, seed_partition_ledger
from dungeon_apprentice.v02_u2_qualify import (
    CONTRACT_COUNT_FIELDS,
    GENERATOR_PROFILE,
    GENERATOR_PROFILE_VERSION,
    QUALIFICATION_ATTEMPT_ID,
    QUALIFICATION_ATTEMPT_KIND,
    QUALIFICATION_ATTEMPT_PROTOCOL,
    QUALIFICATION_ATTEMPT_SCHEMA_VERSION,
    QUALIFICATION_CASES,
    QUALIFICATION_CLAIM_KIND,
    QUALIFICATION_CLAIM_PROTOCOL,
    QUALIFICATION_CLAIM_SCHEMA_VERSION,
    QUALIFICATION_KIND,
    QUALIFICATION_SCHEMA_VERSION,
    QUALIFICATION_SEED_BASE,
    U2QualificationError,
    _case_contracts,
    build_parser,
    canonical_qualification_paths,
    verify_u2_qualification_report,
)

SOURCE_COMMIT = "a" * 40
CLAIM_ID = "b" * 64


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _runtime() -> dict:
    return {
        "python": "3.13.5 test",
        "platform": "test-platform",
        "machine": "arm64",
        "packages": {
            "gymnasium": "1.2.0",
            "minigrid": "3.0.0",
            "numpy": "2.3.0",
            "sb3-contrib": "2.7.0",
            "stable-baselines3": "2.7.0",
            "torch": "2.7.0",
        },
    }


def _case(index: int) -> dict:
    return {
        "seed": QUALIFICATION_SEED_BASE + index,
        "layout_sha256": _digest(f"layout-{index}"),
        "geometry_sha256": _digest(f"geometry-{index}"),
        "oracle_actions": 17 + index % 10,
        "key_visible": bool(index & 1),
        "door_visible": bool(index & 2),
        "generation_attempts": 1 + index % 15,
        "max_steps": 160,
        "elapsed_steps": 17 + index % 10,
        "terminated": True,
        "truncated": False,
        "contracts": {
            field_name: True for field_name in sorted(CONTRACT_COUNT_FIELDS)
        },
    }


def _report(*, claim_sha256: str) -> dict:
    cases = [_case(index) for index in range(QUALIFICATION_CASES)]
    return {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "protocol": "dungeon-apprentice-v0.2-u2",
        "kind": QUALIFICATION_KIND,
        "result": "passed",
        "policy_updates": False,
        "attempt_id": QUALIFICATION_ATTEMPT_ID,
        "claim_id": CLAIM_ID,
        "claim_sha256": claim_sha256,
        "started_at": "2026-07-23T00:00:00+00:00",
        "completed_at": "2026-07-23T00:00:10+00:00",
        "duration_seconds": 10.0,
        "source": {"commit": SOURCE_COMMIT, "dirty": False},
        "runtime": _runtime(),
        "generator_profile": GENERATOR_PROFILE,
        "generator_profile_version": GENERATOR_PROFILE_VERSION,
        "lesson_id": "unlock/u2-separated",
        "seed_partition": {
            "role": "sealed_qualification",
            "start": QUALIFICATION_SEED_BASE,
            "end": QUALIFICATION_SEED_BASE + QUALIFICATION_CASES - 1,
            "count": QUALIFICATION_CASES,
        },
        "seed_partition_ledger": list(seed_partition_ledger()),
        "requested": QUALIFICATION_CASES,
        "generated": QUALIFICATION_CASES,
        "solved": QUALIFICATION_CASES,
        "full_unique_layouts": QUALIFICATION_CASES,
        "geometry_unique_layouts": QUALIFICATION_CASES,
        "contract_counts": {
            field_name: QUALIFICATION_CASES
            for field_name in sorted(CONTRACT_COUNT_FIELDS)
        },
        "visibility_strata": {
            "key_hidden_door_hidden": 500,
            "key_hidden_door_visible": 500,
            "key_visible_door_hidden": 500,
            "key_visible_door_visible": 500,
        },
        "validation_disjointness": {
            "u2_validation_start": 11_200_000,
            "u2_validation_cases": 80,
            "u2_validation_generated": 80,
            "exact_visual_overlap": 0,
        },
        "validation_reference": {
            "generated": 80,
            "unique_exact_layouts": 79,
            "unique_geometries": 73,
            "exact_layout_set_sha256": _digest("validation layouts"),
            "geometry_set_sha256": _digest("validation geometry"),
        },
        "cases": cases,
        "failures": [],
    }


def _write_json(path: Path, value: dict) -> bytes:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    return encoded


def _publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    report_mutation=None,
    attempt_mutation=None,
    claim_mutation=None,
) -> tuple[Path, str]:
    canonical = tmp_path / "canonical"
    monkeypatch.setattr(
        qualification,
        "CANONICAL_QUALIFICATION_DIRECTORY",
        canonical,
    )
    canonical.mkdir()
    paths = canonical_qualification_paths()
    claim = {
        "schema_version": QUALIFICATION_CLAIM_SCHEMA_VERSION,
        "protocol": QUALIFICATION_CLAIM_PROTOCOL,
        "kind": QUALIFICATION_CLAIM_KIND,
        "status": "claimed_for_single_launch",
        "claimed_at": "2026-07-23T00:00:00+00:00",
        "source": {"commit": SOURCE_COMMIT, "dirty": False},
        "attempt_id": QUALIFICATION_ATTEMPT_ID,
        "claim_id": CLAIM_ID,
        "qualification_directory": str(paths["directory"]),
        "report": str(paths["report"]),
        "attempt": str(paths["attempt"]),
        "launcher_token_sha256": _digest("launch token"),
        "acknowledgement_sha256": hashlib.sha256(
            qualification.sealed_acknowledgement().encode()
        ).hexdigest(),
        "seed_partition": {
            "start": QUALIFICATION_SEED_BASE,
            "end": QUALIFICATION_SEED_BASE + QUALIFICATION_CASES - 1,
            "count": QUALIFICATION_CASES,
        },
    }
    if claim_mutation is not None:
        claim_mutation(claim)
    claim_sha256 = hashlib.sha256(_write_json(paths["claim"], claim)).hexdigest()
    report = _report(claim_sha256=claim_sha256)
    if report_mutation is not None:
        report_mutation(report)
    report_sha256 = hashlib.sha256(_write_json(paths["report"], report)).hexdigest()
    paths["checksum"].write_text(
        f"{report_sha256}  report.json\n",
        encoding="ascii",
    )
    attempt = {
        "schema_version": QUALIFICATION_ATTEMPT_SCHEMA_VERSION,
        "protocol": QUALIFICATION_ATTEMPT_PROTOCOL,
        "kind": QUALIFICATION_ATTEMPT_KIND,
        "status": "completed_with_report",
        "attempt_id": QUALIFICATION_ATTEMPT_ID,
        "claim_id": CLAIM_ID,
        "claim_sha256": claim_sha256,
        "started_at": "2026-07-23T00:00:00+00:00",
        "completed_at": "2026-07-23T00:00:11+00:00",
        "source": {"commit": SOURCE_COMMIT, "dirty": False},
        "qualification_directory": str(paths["directory"]),
        "report": str(paths["report"]),
        "claim": str(paths["claim"]),
        "seed_partition": {
            "start": QUALIFICATION_SEED_BASE,
            "end": QUALIFICATION_SEED_BASE + QUALIFICATION_CASES - 1,
            "count": QUALIFICATION_CASES,
        },
        "report_sha256": report_sha256,
        "report_result": "passed",
        "qualifier_exit_status": 0,
    }
    if attempt_mutation is not None:
        attempt_mutation(attempt)
    _write_json(paths["attempt"], attempt)
    return paths["report"], report_sha256


def _verify(path: Path, digest: str):
    return verify_u2_qualification_report(
        path,
        expected_source_commit=SOURCE_COMMIT,
        expected_sha256=digest,
        expected_attempt_id=QUALIFICATION_ATTEMPT_ID,
        expected_claim_id=CLAIM_ID,
    )


def test_verifier_returns_exact_immutable_snapshot_and_bound_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, digest = _publish(tmp_path, monkeypatch)

    evidence = _verify(path, digest)

    assert evidence.requested == 2_000
    assert evidence.seed_start == 5_210_000
    assert evidence.seed_end == 5_211_999
    assert evidence.validation_generated == 80
    assert evidence.validation_unique_layouts == 79
    assert evidence.public_dict()["report_sha256"] == digest
    assert evidence.report_snapshot["result"] == "passed"
    with pytest.raises(TypeError):
        evidence.report_snapshot["result"] = "failed"  # type: ignore[index]
    copy = evidence.verified_report()
    copy["result"] = "failed"
    assert evidence.report_snapshot["result"] == "passed"
    access = evidence.qualified_seed_access()
    assert access.phase is U2AccessPhase.QUALIFIED_TRAINING
    assert access.qualification_sha256 == digest
    assert access.qualification_byte_length == path.stat().st_size
    assert access.qualification_attempt_id == QUALIFICATION_ATTEMPT_ID
    assert access.qualification_claim_id == CLAIM_ID


def test_verifier_requires_external_digest_and_both_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, digest = _publish(tmp_path, monkeypatch)
    with pytest.raises(TypeError):
        verify_u2_qualification_report(  # type: ignore[call-arg]
            path,
            expected_source_commit=SOURCE_COMMIT,
        )
    with pytest.raises(U2QualificationError, match="external frozen digest"):
        _verify(path, "0" * 64)
    with pytest.raises(U2QualificationError, match="launch claim"):
        verify_u2_qualification_report(
            path,
            expected_source_commit=SOURCE_COMMIT,
            expected_sha256=digest,
            expected_attempt_id="different-attempt-id",
            expected_claim_id=CLAIM_ID,
        )


def test_verifier_refuses_a_valid_looking_report_at_an_arbitrary_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, digest = _publish(tmp_path, monkeypatch)
    other = tmp_path / "other.json"
    other.write_bytes(path.read_bytes())
    with pytest.raises(U2QualificationError, match="canonical path"):
        _verify(other, digest)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda report: report.update(result="failed"), "passed U2 preflight"),
        (
            lambda report: report["cases"][0].update(seed=5_210_001),
            "exact sealed seed order",
        ),
        (
            lambda report: report["cases"][0].update(oracle_actions=27),
            "oracle length",
        ),
        (
            lambda report: report.update(full_unique_layouts=1_999),
            "exact-layout uniqueness",
        ),
        (
            lambda report: report["cases"][0]["contracts"].update(
                reward_contract=False
            ),
            "did not satisfy every U2 contract",
        ),
        (
            lambda report: report.update(oracle_action_sequence="left,right"),
            "prohibited oracle demonstration",
        ),
        (
            lambda report: report.update(unexpected=True),
            "fields changed",
        ),
        (
            lambda report: report["runtime"].update(machine=""),
            "runtime machine",
        ),
        (
            lambda report: report["validation_reference"].update(generated=79),
            "generated count and hash uniqueness",
        ),
    ],
)
def test_verifier_rejects_tampered_or_demonstration_like_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    match: str,
) -> None:
    path, digest = _publish(
        tmp_path,
        monkeypatch,
        report_mutation=mutation,
    )

    with pytest.raises(U2QualificationError, match=match):
        _verify(path, digest)


def test_verifier_requires_exact_checksum_claim_and_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, digest = _publish(tmp_path, monkeypatch)
    canonical_qualification_paths()["checksum"].write_text("0" * 64 + "\n")
    with pytest.raises(U2QualificationError, match="checksum"):
        _verify(path, digest)

    other = tmp_path / "attempt"
    other.mkdir()
    path, digest = _publish(
        other,
        monkeypatch,
        attempt_mutation=lambda attempt: attempt.update(status="started"),
    )
    with pytest.raises(U2QualificationError, match="attempt ledger"):
        _verify(path, digest)

    third = tmp_path / "claim"
    third.mkdir()
    path, digest = _publish(
        third,
        monkeypatch,
        claim_mutation=lambda claim: claim.update(status="reusable"),
    )
    with pytest.raises(U2QualificationError, match="launch claim"):
        _verify(path, digest)


def _transient_case() -> dict:
    divider = [[4, lane] for lane in range(1, 8) if lane != 4]
    extras = [[2, 2], [6, 6]]
    return {
        "protocol": "dungeon-apprentice-v0.2-u2",
        "generator_profile": GENERATOR_PROFILE,
        "generator_profile_version": GENERATOR_PROFILE_VERSION,
        "seed": QUALIFICATION_SEED_BASE,
        "seed_role": "sealed_qualification",
        "pure_planner_actions": 20,
        "live_oracle_actions": 20,
        "planner_oracle_counts_match": True,
        "planner_live_action_match": True,
        "post_key_pre_door_turns": 2,
        "post_key_turn_present": True,
        "door_requires_matching_key": True,
        "observation_shape": [56, 56, 3],
        "observation_dtype": "uint8",
        "action_count": 7,
        "observation_contract": True,
        "reward_contract": True,
        "terminal_reason": "success",
        "terminated": True,
        "truncated": False,
        "elapsed_steps": 20,
        "success": True,
        "ordered_objective_completed": True,
        "generation_attempts": 2,
        "max_generation_attempts": 2_048,
        "qualification_topology": {
            "size": 9,
            "max_steps": 160,
            "start": [2, 4],
            "key": [3, 3],
            "goal": [6, 4],
            "door": [4, 4],
            "walls": divider + extras,
            "divider_walls": divider,
            "extra_walls": extras,
            "divider_orientation": "vertical",
            "divider_index": 4,
            "door_lane": 4,
            "approach_from_low": True,
        },
    }


def test_contract_grader_independently_checks_topology_horizon_and_termination() -> None:
    case = _transient_case()
    assert all(_case_contracts(case).values())

    case["qualification_topology"]["goal"] = [3, 5]
    contracts = _case_contracts(case)
    assert contracts["objects_on_declared_sides"] is False
    assert contracts["independent_goal_blocked_while_locked"] is False

    case = _transient_case()
    case["qualification_topology"]["divider_walls"].pop()
    assert _case_contracts(case)["complete_divider_with_sole_door"] is False

    case = _transient_case()
    case["truncated"] = True
    assert _case_contracts(case)["success_terminated_not_truncated"] is False


def test_cli_has_no_seed_count_or_output_path_override() -> None:
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}
    assert destinations == {"help", "launch_token", "acknowledge"}
