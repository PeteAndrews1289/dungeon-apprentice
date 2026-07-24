from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from test_v04_ineffective_trace_operational import _cohort

from dungeon_apprentice import v04_ineffective_trace_dashboard as dashboard


def test_v04_dashboard_ready_snapshot_is_bounded_and_read_only(
    tmp_path: Path,
) -> None:
    helper, root, _media, _source, _tag_object, public = _cohort(tmp_path)
    dashboard._authenticate_live_qualification = (
        lambda *_args, **_kwargs: json.loads(json.dumps(public))
    )

    snapshot = dashboard.load_v04_snapshot(root)
    assert snapshot["protocol"] == helper.PROTOCOL
    assert snapshot["cohort_id"] == helper.COHORT_ID
    assert snapshot["phase"] == "ready"
    assert [arm["id"] for arm in snapshot["arms"]] == [
        "trace-sham",
        "ineffective-trace",
    ]
    assert snapshot["checkpoint_promotable"] is False
    assert snapshot["u3_authorized"] is False

    server = dashboard.start_v04_dashboard(
        root,
        host="127.0.0.1",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/v04.json",
            timeout=5,
        ) as response:
            payload = json.load(response)
        assert payload["phase"] == "ready"
        assert payload["active_arm"] is None

        request = urllib.request.Request(
            f"http://{host}:{port}/api/v04.json",
            method="POST",
            data=b"{}",
        )
        with pytest.raises(urllib.error.HTTPError) as captured:
            urllib.request.urlopen(request, timeout=5)
        assert captured.value.code == 501
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_v04_dashboard_cached_authentication_rejects_contract_tamper(
    tmp_path: Path,
) -> None:
    _helper, root, _media, _source, _tag_object, public = _cohort(tmp_path)
    dashboard._authenticate_live_qualification = (
        lambda *_args, **_kwargs: json.loads(json.dumps(public))
    )
    authentication = dashboard._startup_authentication(root)
    path = root / "cohort-contract.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract["matched_design"]["resumable"] = True
    path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        dashboard.V04DashboardError,
        match="contract",
    ):
        dashboard.load_v04_snapshot(
            root,
            authentication=authentication,
        )


def test_v04_dashboard_html_and_api_are_versioned() -> None:
    assert "bounded ineffective-action trace" in dashboard.V04_DASHBOARD_HTML
    assert "/api/v04.json" in dashboard.V04_DASHBOARD_HTML
    assert "/api/v03.json" not in dashboard.V04_DASHBOARD_HTML
    assert dashboard.DEFAULT_PORT == 8792


def test_v04_dashboard_frame_is_status_bound_and_tamper_evident(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    arm = root / "trace-sham"
    frames = arm / "frames"
    frames.mkdir(parents=True)
    frame = frames / "exam-unlock-u2-separated.png"
    frame.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    digest = hashlib.sha256(frame.read_bytes()).hexdigest()
    status = {
        "latest_frame": {
            "path": "frames/exam-unlock-u2-separated.png",
            "sha256": digest,
            "bytes": frame.stat().st_size,
        }
    }
    (arm / "status.json").write_text(
        json.dumps(status) + "\n",
        encoding="utf-8",
    )
    assert dashboard.resolve_v04_frame(
        root,
        arm_id="trace-sham",
    ) == frame

    status["latest_frame"]["sha256"] = "0" * 64
    (arm / "status.json").write_text(
        json.dumps(status) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        dashboard.V04DashboardError,
        match="checksum",
    ):
        dashboard.resolve_v04_frame(root, arm_id="trace-sham")
