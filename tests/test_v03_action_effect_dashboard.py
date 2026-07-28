from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType

import pytest

from dungeon_apprentice import v03_action_effect_dashboard as dashboard

REPOSITORY = Path(__file__).resolve().parents[1]
HELPER_PATH = REPOSITORY / "scripts" / "v03_action_effect_manifest.py"
OPERATIONAL_FIXTURES = (
    REPOSITORY / "tests" / "test_v03_action_effect_operational.py"
)


def _helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "v03_action_effect_manifest_dashboard_fixture",
        HELPER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operational_fixtures() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "v03_action_effect_operational_fixtures",
        OPERATIONAL_FIXTURES,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _qualification(
    helper: ModuleType,
    *,
    root: Path,
    media: Path,
    source: str,
    tag_object: str,
    path: Path,
) -> tuple[Path, dict[str, object]]:
    path.parent.mkdir()
    _write_json(
        path,
        {
            "schema_version": 1,
            "protocol": helper.PROTOCOL,
            "kind": "sealed_stage_a_preflight_qualification",
            "verdict": "qualified",
            "created_at": "2026-07-24T00:00:00+00:00",
            "claim": {},
            "source": {"commit": source, "dirty": False},
            "protocol_document": {
                "path": helper.v03.PROTOCOL_DOCUMENT,
                "sha256": "1" * 64,
            },
            "parent": {},
            "predecessors": {},
            "guards": {},
            "sampler_preflight": [],
            "architecture_contract": {},
            "protected_partitions": [],
            "smoke_evidence": {"_full_report": {}},
            "storage_caps": {},
            "storage_preflight": {},
            "restrictions": {},
        },
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name("report.json.sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )
    public: dict[str, object] = {
        "report": str(path),
        "report_sha256": digest,
        "checksum": str(path.with_name("report.json.sha256")),
        "source_commit": source,
        "tag": helper.TAG_NAME,
        "tag_object": tag_object,
        "tag_payload_sha256": "2" * 64,
        "verdict": "qualified",
        "protocol_document_sha256": "1" * 64,
        "guard_mapping_sha256": "3" * 64,
        "sampler_preflight_sha256": "4" * 64,
        "architecture_contract_sha256": "5" * 64,
        "smoke_evidence_sha256": "6" * 64,
        "protected_partitions_sha256": "7" * 64,
        "failed_stage_a_attempt_sha256": "8" * 64,
        "failed_stage_a_r2_attempt_sha256": "9" * 64,
        "storage_caps": {
            "per_arm_bytes": helper.LINEAGE_CAP_BYTES,
            "scientific_cohort_bytes": (
                helper.COHORT_SCIENTIFIC_CAP_BYTES
            ),
            "media_bytes": helper.MEDIA_CAP_BYTES,
            "combined_bytes": helper.COMBINED_PLANNED_CAP_BYTES,
        },
    }
    return path, public


def _cohort(
    tmp_path: Path,
) -> tuple[ModuleType, Path, Path, str, str]:
    helper = _helper()
    root = tmp_path / "cohort"
    media = tmp_path / "media"
    root.mkdir()
    media.mkdir()
    source = "a" * 40
    tag_object = "b" * 40
    qualification, qualification_public = _qualification(
        helper,
        root=root,
        media=media,
        source=source,
        tag_object=tag_object,
        path=tmp_path / "qualification" / "report.json",
    )
    helper._verified_qualification_binding = (
        lambda *_args, **_kwargs: dict(qualification_public)
    )
    dashboard._authenticate_live_qualification = (
        lambda *_args, **_kwargs: dict(qualification_public)
    )
    helper.create_cohort(
        root,
        media_root=media,
        qualification_report=qualification,
        source_commit=source,
        tag_object=tag_object,
    )
    return helper, root, media, source, tag_object


def _training_status(
    helper: ModuleType,
    *,
    root: Path,
    source: str,
    arm: str,
    updated_at: str = "2026-07-24T04:00:00+00:00",
) -> dict:
    contract_path = root / "cohort-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    return {
        "protocol": helper.PROTOCOL,
        "cohort_id": helper.COHORT_ID,
        "phase": "training",
        "arm": arm,
        "source": {"commit": source, "dirty": False},
        "qualification_sha256": contract["qualification"][
            "report_sha256"
        ],
        "cohort_contract_sha256": hashlib.sha256(
            contract_path.read_bytes()
        ).hexdigest(),
        "parent_checkpoint_sha256": helper.v03.PARENT_CHECKPOINT_SHA256,
        "action_cap": helper.ACTION_CAP,
        "child_trained_actions": 65_536,
        "collected_actions": 67_584,
        "remaining_action_budget": helper.ACTION_CAP - 65_536,
        "optimizer_updates": helper.v03.PARENT_OPTIMIZER_UPDATES + 128,
        "exam_count": 2,
        "frame_revision": 4,
        "updated_at": updated_at,
        "encoder_latest": {
            "child_trained_actions": 65_536,
            "weight_l2_norm": 0.0,
            "weight_nonzero_parameters": 0,
        },
        "context_metrics": {
            "transitions": 65_536,
            "active_contexts": 0,
            "activation_rate": 0.0,
            "changed": 39_322,
            "unchanged": 26_214,
            "changed_rate": 0.6,
        },
        "first_rollout_identity_sha256": "1" * 64,
        "first_rollout_verified": True,
        "exam_records": [
            _exam(index * helper.EVALUATION_EVERY)
            for index in range(1, 3)
        ],
    }


def _exam(boundary: int) -> dict:
    lessons = {
        lesson_id: {
            "lesson_id": lesson_id,
            "successes": 76,
            "episodes": 80,
            "panel_successes": [38, 38],
            "mean_ineffective_interactions": 1.0,
        }
        for lesson_id in (
            "navigate/full",
            "unlock/u0-visible",
            "unlock/u1-local",
            "unlock/u2-separated",
        )
    }
    return {
        "child_trained_actions": boundary,
        "allocation_valid": True,
        "practice_profile": "normal",
        "lessons": lessons,
        "case_diagnostics": {
            "max_case_ineffective_interactions": 3,
            "cases_with_ineffective_at_least_10": 0,
            "max_repeated_identical_interaction_run": 3,
        },
    }


def test_empty_cohort_snapshot_is_v03_and_nonpromotable(
    tmp_path: Path,
) -> None:
    _helper_module, root, _media, source, _tag = _cohort(tmp_path)
    snapshot = dashboard.load_v03_snapshot(root)
    assert snapshot["protocol"] == dashboard.PROTOCOL
    assert snapshot["cohort_id"] == dashboard.COHORT_ID
    assert snapshot["source_commit"] == source
    assert snapshot["phase"] == "ready"
    assert snapshot["active_arm"] is None
    assert [arm["id"] for arm in snapshot["arms"]] == list(
        dashboard.ARM_ORDER
    )
    assert snapshot["checkpoint_promotable"] is False
    assert snapshot["u3_authorized"] is False
    assert snapshot["selected_architecture"] is None


def test_live_snapshot_shows_actions_exams_context_first_rollout_and_staleness(
    tmp_path: Path,
) -> None:
    helper, root, _media, source, tag_object = _cohort(tmp_path)
    helper.start_arm(
        root,
        source_commit=source,
        tag_object=tag_object,
        arm_id="sham",
    )
    directory = root / "sham"
    directory.mkdir()
    status = _training_status(
        helper,
        root=root,
        source=source,
        arm="sham",
        updated_at="2020-01-01T00:00:00+00:00",
    )
    frames = directory / "frames"
    frames.mkdir()
    png = b"\x89PNG\r\n\x1a\nfixture-pixels"
    (frames / "latest.png").write_bytes(png)
    status["latest_frame"] = {
        "path": "frames/latest.png",
        "sha256": hashlib.sha256(png).hexdigest(),
    }
    _write_json(directory / "status.json", status)
    snapshot = dashboard.load_v03_snapshot(root)
    sham = snapshot["arms"][0]
    assert snapshot["heartbeat_stale"]
    assert snapshot["active_arm"] == "sham"
    assert snapshot["total_trained_actions"] == 65_536
    assert snapshot["total_exams"] == 2
    assert sham["exams_completed"] == 2
    assert sham["context_encoder"]["weight_l2_norm"] == 0.0
    assert sham["context_summary"]["activation_rate"] == 0.0
    assert sham["first_rollout"]["verified"]
    assert sham["frame_url"] == "/api/frame?arm=sham"
    assert dashboard.resolve_v03_frame(root, arm_id="sham").read_bytes() == png


def test_dashboard_rejects_source_qualification_and_status_drift(
    tmp_path: Path,
) -> None:
    helper, root, _media, source, tag_object = _cohort(tmp_path)
    helper.start_arm(
        root,
        source_commit=source,
        tag_object=tag_object,
        arm_id="sham",
    )
    directory = root / "sham"
    directory.mkdir()
    status = _training_status(
        helper,
        root=root,
        source=source,
        arm="sham",
    )
    status["qualification_sha256"] = "0" * 64
    _write_json(directory / "status.json", status)
    with pytest.raises(dashboard.V03DashboardError, match="status differs"):
        dashboard.load_v03_snapshot(root)

    status["qualification_sha256"] = json.loads(
        (root / "cohort-contract.json").read_text(encoding="utf-8")
    )["qualification"]["report_sha256"]
    status["source"]["commit"] = "f" * 40
    _write_json(directory / "status.json", status)
    with pytest.raises(dashboard.V03DashboardError, match="status differs"):
        dashboard.load_v03_snapshot(root)


def test_safe_frame_refuses_traversal_symlink_wrong_type_and_oversize(
    tmp_path: Path,
) -> None:
    helper, root, _media, source, tag_object = _cohort(tmp_path)
    helper.start_arm(
        root,
        source_commit=source,
        tag_object=tag_object,
        arm_id="sham",
    )
    directory = root / "sham"
    directory.mkdir()
    status = _training_status(
        helper,
        root=root,
        source=source,
        arm="sham",
    )
    status["latest_frame"] = {"path": "../secret.png"}
    _write_json(directory / "status.json", status)
    with pytest.raises(dashboard.V03DashboardError, match="escapes"):
        dashboard.resolve_v03_frame(root, arm_id="sham")
    with pytest.raises(FileNotFoundError, match="unknown"):
        dashboard.resolve_v03_frame(root, arm_id="../qualification")

    outside = tmp_path / "outside.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\noutside")
    frames = directory / "frames"
    frames.mkdir()
    (frames / "latest.png").symlink_to(outside)
    status["latest_frame"] = {"path": "frames/latest.png"}
    _write_json(directory / "status.json", status)
    with pytest.raises(dashboard.V03DashboardError, match="unsafe"):
        dashboard.resolve_v03_frame(root, arm_id="sham")

    (frames / "latest.png").unlink()
    (frames / "latest.png").write_bytes(b"not-a-png-file")
    with pytest.raises(dashboard.V03DashboardError, match="not a PNG"):
        dashboard.resolve_v03_frame(root, arm_id="sham")

    (frames / "latest.png").write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + b"x" * (dashboard.FRAME_MAX_BYTES + 1)
    )
    with pytest.raises(dashboard.V03DashboardError, match="unsafe"):
        dashboard.resolve_v03_frame(root, arm_id="sham")


def test_http_server_exposes_only_dashboard_snapshot_and_safe_frame(
    tmp_path: Path,
) -> None:
    helper, root, _media, source, tag_object = _cohort(tmp_path)
    helper.start_arm(
        root,
        source_commit=source,
        tag_object=tag_object,
        arm_id="sham",
    )
    directory = root / "sham"
    directory.mkdir()
    status = _training_status(
        helper,
        root=root,
        source=source,
        arm="sham",
    )
    frames = directory / "frames"
    frames.mkdir()
    png = b"\x89PNG\r\n\x1a\nhttp-frame"
    (frames / "latest.png").write_bytes(png)
    status["latest_frame"] = {
        "path": "frames/latest.png",
        "sha256": hashlib.sha256(png).hexdigest(),
    }
    _write_json(directory / "status.json", status)

    server = dashboard.start_v03_dashboard(root, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/v03.json",
            timeout=2,
        ) as response:
            payload = json.loads(response.read())
            assert payload["protocol"] == dashboard.PROTOCOL
            assert response.headers["Cache-Control"] == "no-store"
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/frame?arm=sham",
            timeout=2,
        ) as response:
            assert response.headers["Content-Type"] == "image/png"
            assert response.read() == png
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(
                f"http://{host}:{port}/cohort-contract.json",
                timeout=2,
            )
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_server_deep_authenticates_once_then_polls_with_cached_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _helper_module, root, _media, _source, _tag = _cohort(tmp_path)
    contract = json.loads(
        (root / "cohort-contract.json").read_text(encoding="utf-8")
    )
    deep_auth_calls = 0

    def authenticate(*_args: object, **_kwargs: object) -> dict:
        nonlocal deep_auth_calls
        deep_auth_calls += 1
        return dict(contract["qualification"])

    monkeypatch.setattr(
        dashboard,
        "_authenticate_live_qualification",
        authenticate,
    )
    server = dashboard.start_v03_dashboard(root, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        assert deep_auth_calls == 1
        for _ in range(2):
            with urllib.request.urlopen(
                f"http://{host}:{port}/api/v03.json",
                timeout=2,
            ) as response:
                assert response.status == 200
        assert deep_auth_calls == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("target", ["report", "checksum"])
def test_cached_poll_rejects_qualification_file_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    _helper_module, root, _media, _source, _tag = _cohort(tmp_path)
    contract = json.loads(
        (root / "cohort-contract.json").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        dashboard,
        "_authenticate_live_qualification",
        lambda *_args, **_kwargs: dict(contract["qualification"]),
    )
    server = dashboard.start_v03_dashboard(root, port=0)
    try:
        qualification = contract["qualification"]
        path = Path(
            str(
                qualification[
                    "report" if target == "report" else "checksum"
                ]
            )
        )
        path.write_bytes(path.read_bytes() + b"tampered\n")
        with pytest.raises(
            dashboard.V03DashboardError,
            match=r"qualification .*changed",
        ):
            dashboard.load_v03_snapshot(
                root,
                authentication=server.authentication,
            )
    finally:
        server.server_close()


def test_dashboard_rejects_nonabsolute_or_symlinked_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="absolute"):
        dashboard.load_v03_snapshot(Path("relative"))
    _helper_module, root, _media, _source, _tag = _cohort(tmp_path)
    link = tmp_path / "cohort-link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="regular directory"):
        dashboard.load_v03_snapshot(link)


def test_terminal_dashboard_recomputes_candidate_only_selection(
    tmp_path: Path,
) -> None:
    fixtures = _operational_fixtures()
    helper, root, _media, source, tag_object = _cohort(tmp_path)
    contract = json.loads(
        (root / "cohort-contract.json").read_text(encoding="utf-8")
    )
    dashboard._authenticate_live_qualification = (
        lambda *_args, **_kwargs: dict(contract["qualification"])
    )
    for arm, eligible in (("sham", True), ("action-effect", True)):
        helper.start_arm(
            root,
            source_commit=source,
            tag_object=tag_object,
            arm_id=arm,
        )
        fixtures._arm_terminal(
            helper,
            root=root,
            source=source,
            arm=arm,
            eligible=eligible,
        )
        helper.finish_arm(
            root,
            source_commit=source,
            tag_object=tag_object,
            arm_id=arm,
            outcome="completed",
            trainer_exit_code=0,
        )
    process_binding = fixtures._seal_process_closeout(
        helper,
        root=root,
        source=source,
        tag_object=tag_object,
    )
    helper.finalize_cohort(
        root,
        source_commit=source,
        tag_object=tag_object,
    )
    snapshot = dashboard.load_v03_snapshot(root)
    assert snapshot["phase"] == "completed"
    assert snapshot["terminal_verdict"] == "architecture_selected"
    assert snapshot["selected_architecture"]["name"] == "action-effect"
    assert snapshot["arms"][0]["eligible"]
    assert snapshot["arms"][1]["eligible"]
    assert snapshot["checkpoint_promotable"] is False
    assert snapshot["u3_authorized"] is False
    assert snapshot["process_closeout"] == process_binding

    evidence_path = root / helper.PROCESS_CLOSEOUT_NAME
    original = evidence_path.read_bytes()
    evidence = json.loads(original)
    evidence["verdict"] = "tampered"
    _write_json(evidence_path, evidence)
    with pytest.raises(
        dashboard.V03DashboardError,
        match="process closeout",
    ):
        dashboard.load_v03_snapshot(root)
    evidence_path.write_bytes(original)
    assert dashboard.load_v03_snapshot(root)["phase"] == "completed"
    evidence_path.unlink()
    with pytest.raises(dashboard.V03DashboardError, match="missing"):
        dashboard.load_v03_snapshot(root)
