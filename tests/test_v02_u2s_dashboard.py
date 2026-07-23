from __future__ import annotations

import hashlib
import json
import socket
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from dungeon_apprentice import v02_u2s_dashboard as dashboard


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _contract(root: Path, *, source: str = "a" * 40) -> tuple[dict, str]:
    contract = {
        "schema_version": 1,
        "protocol": dashboard.PROTOCOL,
        "cohort_id": dashboard.COHORT_ID,
        "source": {"commit": source, "dirty": False},
        "preregistration": {
            "tag": "u2s-stability-ablation-v0.2-u2s-20260723",
            "tag_object": "b" * 40,
            "peeled_commit": source,
            "tag_payload_sha256": "e" * 64,
        },
        "parent": {"checkpoint_sha256": "c" * 64},
        "qualification": {
            "verdict": "qualified",
            "report_sha256": "d" * 64,
        },
        "matched_design": {
            "arm_order": list(dashboard.ARM_ORDER),
            "selection_priority": list(dashboard.ARM_ORDER),
            "checkpoint_promotable": False,
        },
        "arms": [
            {
                "id": arm,
                "label": arm.title(),
                "directory": arm,
                "ppo_profile": (
                    "conservative"
                    if arm in {"conservative", "combined"}
                    else "current"
                ),
                "no_effect_penalty": arm in {"no-effect", "combined"},
            }
            for arm in dashboard.ARM_ORDER
        ],
    }
    path = root / "cohort-contract.json"
    _write_json(path, contract)
    return contract, hashlib.sha256(path.read_bytes()).hexdigest()


def _lesson_row(
    lesson_id: str,
    *,
    boundary: int,
    fail: bool,
) -> dict:
    is_u2 = lesson_id == "unlock/u2-separated"
    successes = 71 if fail and is_u2 else (72 if is_u2 else 76)
    panels = [33, 38] if fail and is_u2 else ([34, 38] if is_u2 else [38, 38])
    return {
        "protocol": dashboard.PROTOCOL,
        "counts_toward_arm_stability": True,
        "child_trained_actions": boundary,
        "allocation_valid": True,
        "practice_profile": "normal",
        "lesson_id": lesson_id,
        "lesson_label": lesson_id,
        "successes": successes,
        "episodes": 80,
        "panel_successes": panels,
        "mean_ineffective_interactions": 1.25,
        "passed": True,
        "max_ineffective_interactions": 4,
        "max_identical_visible_no_effect_streak": 3,
        "max_repeated_identical_interaction_run": 5,
        "top_two_ineffective_share": 0.4,
        "ineffective_tail": {
            "percentiles": {"p50": 0, "p90": 2, "p95": 3, "p99": 4},
            "cases_at_least": {"1": 12, "3": 3, "10": 0, "32": 0},
            "worst_share": {"1": 0.25, "2": 0.4, "5": 0.7},
        },
    }


def _arm(
    root: Path,
    *,
    arm: str,
    source: str,
    contract_sha256: str,
    fail_terminal: bool,
) -> None:
    directory = root / arm
    directory.mkdir()
    boundaries = [index * 32_768 for index in range(1, 33)]
    exam_records: list[dict] = []
    with (directory / "evaluations.jsonl").open("w", encoding="utf-8") as stream:
        for boundary in boundaries:
            lessons: dict[str, dict] = {}
            for lesson_id in dashboard.LESSON_IDS:
                row = _lesson_row(
                    lesson_id,
                    boundary=boundary,
                    fail=fail_terminal and boundary >= 983_040,
                )
                lessons[lesson_id] = row
                stream.write(json.dumps(row) + "\n")
            exam_records.append(
                {
                    "child_trained_actions": boundary,
                    "allocation_valid": True,
                    "practice_profile": "normal",
                    "lessons": lessons,
                }
            )
    _write_json(
        directory / "report.json",
        {
            "protocol": dashboard.PROTOCOL,
            "arm": arm,
            "exam_records": exam_records,
        },
    )
    _write_json(
        directory / "status.json",
        {
            "protocol": dashboard.PROTOCOL,
            "phase": "completed",
            "arm": arm,
            "source": {"commit": source, "dirty": False},
            "parent_checkpoint_sha256": "c" * 64,
            "qualification_sha256": "d" * 64,
            "cohort_contract_sha256": contract_sha256,
            "action_cap": dashboard.ACTION_CAP,
            "child_trained_actions": dashboard.ACTION_CAP,
            "collected_actions": 1_835_008,
            "remaining_action_budget": 0,
            "optimizer_updates": 3_584,
            "updated_at": "2026-07-23T16:00:00+00:00",
        },
    )


def _cohort(root: Path) -> None:
    contract_value, digest = _contract(root)
    source = "a" * 40
    for arm in dashboard.ARM_ORDER:
        _arm(
            root,
            arm=arm,
            source=source,
            contract_sha256=digest,
            fail_terminal=arm == "control",
        )
    initial_rng_identity = {
        "captured_before_action_one": True,
        "algorithm_seed": 20_260_753,
        "components": {"fixture": "deep-verifier-owned"},
        "aggregate_sha256": "9" * 64,
    }
    arm_evidence = {
        arm: {
            "arm": arm,
            "child_trained_actions": dashboard.ACTION_CAP,
            "exam_count": 32,
            "case_count": 10_240,
            "report_sha256": hashlib.sha256(
                (root / arm / "report.json").read_bytes()
            ).hexdigest(),
            "case_evidence_sha256": f"{index + 11:064x}",
            "initial_rng_identity": initial_rng_identity,
        }
        for index, arm in enumerate(dashboard.ARM_ORDER)
    }
    grades = {
        arm: {"arm": arm, "eligible": arm != "control"}
        for arm in dashboard.ARM_ORDER
    }
    process_closeout = {
        "checked_at": "2026-07-23T20:00:00+00:00",
        "trainer_process_count": 0,
        "supervisor_process_count": 0,
        "caffeinate_process_count": 0,
        "orphan_free": True,
        "dashboard": {
            "pid": 41_001,
            "parent_pid": 1,
            "command_sha256": "8" * 64,
            "host": "127.0.0.1",
            "port": 8787,
            "listener_pids": [41_001],
            "explicitly_excepted": True,
        },
    }
    factorial_contrasts = {
        "schema_version": 1,
        "descriptive_only": True,
        "population_inference_authorized": False,
    }
    matched_rng = {
        "identical_across_all_arms": True,
        "identity": initial_rng_identity,
        "arm_aggregate_sha256": {
            arm: initial_rng_identity["aggregate_sha256"]
            for arm in dashboard.ARM_ORDER
        },
    }
    report = {
        "schema_version": 1,
        "protocol": dashboard.PROTOCOL,
        "cohort_id": dashboard.COHORT_ID,
        "verdict": "mechanism_selected",
        "development_only": True,
        "source": contract_value["source"],
        "preregistration": contract_value["preregistration"],
        "parent": contract_value["parent"],
        "qualification": contract_value["qualification"],
        "cohort_contract": "cohort-contract.json",
        "cohort_contract_sha256": digest,
        "arm_evidence": arm_evidence,
        "matched_initial_rng_identity": matched_rng,
        "process_closeout": process_closeout,
        "factorial_contrasts": factorial_contrasts,
        "selection": {
            "selected_arm": "conservative",
            "priority": list(dashboard.ARM_ORDER),
            "grades": grades,
            "ablation_checkpoint_reuse_authorized": False,
        },
        "selected_configuration": "conservative",
        "checkpoint_rule": {
            "ablation_checkpoint_reuse_authorized": False,
            "successor_checkpoint": None,
            "successor_must_restart_from_confirmed_u1_parent": True,
        },
    }
    report_path = root / "report.json"
    _write_json(report_path, report)
    report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    integrity = {
        "schema_version": 1,
        "protocol": dashboard.PROTOCOL,
        "cohort_id": dashboard.COHORT_ID,
        "report": "report.json",
        "report_sha256": report_digest,
        "cohort_contract_sha256": digest,
        "arm_report_sha256": {
            arm: arm_evidence[arm]["report_sha256"]
            for arm in dashboard.ARM_ORDER
        },
        "arm_case_evidence_sha256": {
            arm: arm_evidence[arm]["case_evidence_sha256"]
            for arm in dashboard.ARM_ORDER
        },
        "initial_rng_identity_sha256": dashboard._canonical_sha256(
            matched_rng
        ),
        "process_closeout_sha256": dashboard._canonical_sha256(
            process_closeout
        ),
        "factorial_contrasts_sha256": dashboard._canonical_sha256(
            factorial_contrasts
        ),
    }
    integrity_path = root / "report.integrity.json"
    _write_json(integrity_path, integrity)
    integrity_digest = hashlib.sha256(integrity_path.read_bytes()).hexdigest()
    _write_json(
        root / "cohort.json",
        {
            "schema_version": 1,
            "protocol": dashboard.PROTOCOL,
            "cohort_id": dashboard.COHORT_ID,
            "contract": "cohort-contract.json",
            "contract_sha256": digest,
            "source_commit": source,
            "tag": "u2s-stability-ablation-v0.2-u2s-20260723",
            "tag_object": "b" * 40,
            "parent_checkpoint_sha256": "c" * 64,
            "phase": "completed",
            "active_arm": None,
            "terminal_report": {
                "path": "report.json",
                "sha256": report_digest,
                "integrity": "report.integrity.json",
                "integrity_sha256": integrity_digest,
                "verdict": "mechanism_selected",
                "selected_configuration": "conservative",
                "checkpoint_promotable": False,
                "initial_rng_identity_sha256": integrity[
                    "initial_rng_identity_sha256"
                ],
                "process_closeout_sha256": integrity[
                    "process_closeout_sha256"
                ],
                "factorial_contrasts_sha256": integrity[
                    "factorial_contrasts_sha256"
                ],
            },
            "arms": [
                {
                    "id": arm,
                    "state": "completed",
                    "terminal": {
                        "verified": True,
                        **arm_evidence[arm],
                    },
                }
                for arm in dashboard.ARM_ORDER
            ],
        },
    )


def _mock_completed_authentication(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> None:
    contract = json.loads((root / "cohort-contract.json").read_text())
    report = json.loads((root / "report.json").read_text())
    monkeypatch.setattr(
        dashboard,
        "_authenticate_live_qualification",
        lambda **_kwargs: contract["qualification"],
    )
    monkeypatch.setattr(
        dashboard,
        "_authenticate_live_process_closeout",
        lambda _root, sealed: dict(sealed),
    )
    monkeypatch.setattr(
        dashboard,
        "_deep_verify_arm_terminal",
        lambda _directory, *, arm_id, **_kwargs: report["arm_evidence"][
            arm_id
        ],
    )
def test_snapshot_compares_all_arms_and_seals_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cohort"
    root.mkdir()
    _cohort(root)
    _mock_completed_authentication(monkeypatch, root)
    snapshot = dashboard.load_u2s_snapshot(root)
    assert snapshot["phase"] == "completed"
    assert snapshot["total_trained_actions"] == 4 * dashboard.ACTION_CAP
    assert [arm["id"] for arm in snapshot["arms"]] == list(dashboard.ARM_ORDER)
    assert snapshot["arms"][0]["eligible"] is False
    assert snapshot["arms"][1]["eligible"] is True
    assert snapshot["selected_mechanism"]["id"] == "conservative"
    assert snapshot["checkpoint_promotable"] is False
    terminal = snapshot["arms"][1]["terminal_checks"]
    assert [exam["boundary"] for exam in terminal] == [
        983_040,
        1_015_808,
        1_048_576,
    ]
    assert all(exam["passed"] for exam in terminal)
    u2 = terminal[-1]["lessons"][-1]
    assert u2["ineffective_quantiles"]["p95"] == 3
    assert terminal[-1]["case_threshold_counts"]["at_least_10"] == 0
    assert terminal[-1]["max_no_effect_streak"] == 3
    assert terminal[-1]["max_interaction_run"] == 5
    assert terminal[-1]["top_two_tail_share"] == 0.4


def test_fabricated_completed_claim_without_arm_bundles_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cohort"
    root.mkdir()
    _cohort(root)
    contract = json.loads((root / "cohort-contract.json").read_text())
    monkeypatch.setattr(
        dashboard,
        "_authenticate_live_qualification",
        lambda **_kwargs: contract["qualification"],
    )
    monkeypatch.setattr(
        dashboard,
        "_authenticate_live_process_closeout",
        lambda _root, sealed: dict(sealed),
    )
    with pytest.raises(
        dashboard.U2sDashboardError,
        match="deep terminal authentication",
    ):
        dashboard.load_u2s_snapshot(root)


def test_healthy_nonterminal_snapshot_does_not_claim_or_deep_authenticate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cohort"
    root.mkdir()
    _contract_value, digest = _contract(root)
    _write_json(
        root / "cohort.json",
        {
            "schema_version": 1,
            "protocol": dashboard.PROTOCOL,
            "cohort_id": dashboard.COHORT_ID,
            "contract": "cohort-contract.json",
            "contract_sha256": digest,
            "source_commit": "a" * 40,
            "tag": "u2s-stability-ablation-v0.2-u2s-20260723",
            "tag_object": "b" * 40,
            "parent_checkpoint_sha256": "c" * 64,
            "phase": "ready",
            "active_arm": None,
            "terminal_report": None,
            "arms": [
                {"id": arm, "state": "pending"}
                for arm in dashboard.ARM_ORDER
            ],
        },
    )

    def unexpected(**_kwargs: object) -> dict:
        raise AssertionError("nonterminal dashboard invoked terminal authentication")

    monkeypatch.setattr(dashboard, "_authenticate_live_qualification", unexpected)
    monkeypatch.setattr(
        dashboard,
        "_deep_verify_arm_terminal",
        lambda *_args, **_kwargs: unexpected(),
    )
    snapshot = dashboard.load_u2s_snapshot(root)
    assert snapshot["phase"] == "ready"
    assert snapshot["selected_mechanism"] is None
    assert snapshot["terminal_verdict"] is None


def test_snapshot_fails_closed_on_arm_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cohort"
    root.mkdir()
    _cohort(root)
    _mock_completed_authentication(monkeypatch, root)
    status_path = root / "combined" / "status.json"
    status = json.loads(status_path.read_text())
    status["source"]["commit"] = "d" * 40
    _write_json(status_path, status)
    with pytest.raises(dashboard.U2sDashboardError, match="differs"):
        dashboard.load_u2s_snapshot(root)


def test_dashboard_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cohort"
    root.mkdir()
    _cohort(root)
    _mock_completed_authentication(monkeypatch, root)
    with socket.socket() as probe:
        probe.bind((dashboard.DEFAULT_HOST, 0))
        port = int(probe.getsockname()[1])
    server = dashboard.start_u2s_dashboard(root, port=port)
    try:
        with urllib.request.urlopen(
            f"http://{dashboard.DEFAULT_HOST}:{port}/api/u2s.json",
            timeout=2,
        ) as response:
            payload = json.load(response)
        assert payload["selected_mechanism"]["id"] == "conservative"
        request = urllib.request.Request(
            f"http://{dashboard.DEFAULT_HOST}:{port}/api/u2s.json",
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=2)
        assert caught.value.code == 405
    finally:
        server.shutdown()
        server.server_close()


def test_dashboard_copy_explains_metrics_and_claim_boundary() -> None:
    for phrase in (
        "Can stability be learned?",
        "Fixed priority decision",
        "Terminal-window comparison",
        "Worst case",
        "Interaction run",
        "Visible no-effect streak",
        "No arm checkpoint is promotable",
        "there is no resume",
        "/api/u2s.json",
        "includes(d.phase)?60000:2000",
        "setTimeout(poll,pollDelay)",
    ):
        assert phrase in dashboard.U2S_DASHBOARD_HTML
