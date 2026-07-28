from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dungeon_apprentice import v02_u2r
from dungeon_apprentice import v02_u2r_dashboard as dashboard


def _segment(
    root: Path,
    *,
    index: int,
    phase: str,
    updated_at: str | None = None,
) -> Path:
    name = (
        v02_u2r.DEFAULT_RUN_NAME
        if index == 0
        else f"{v02_u2r.DEFAULT_RUN_NAME}-resume-{index}"
    )
    directory = root / name
    frames = directory / "frames"
    frames.mkdir(parents=True)
    (frames / "latest.png").write_bytes(b"\x89PNG\r\n")
    lesson_frames: dict[str, str] = {}
    evaluations: list[dict[str, object]] = []
    for lesson_id, label in dashboard.LESSONS:
        filename = f"exam-{lesson_id.replace('/', '-')}.png"
        (frames / filename).write_bytes(b"\x89PNG\r\n")
        lesson_frames[lesson_id] = f"frames/{filename}"
        evaluations.append(
            {
                "lesson_id": lesson_id,
                "lesson_label": label,
                "successes": 72,
                "episodes": 80,
                "success_rate": 0.9,
                "panel_successes": [36, 36],
                "mean_ineffective_interactions": 1.25,
                "passed": True,
                "private_debug": "must not leave the dashboard",
            }
        )
    status = {
        "protocol": v02_u2r.PROTOCOL,
        "phase": phase,
        "source": {"dirty": False, "commit": "a" * 40},
        "parent": {"checkpoint_sha256": "b" * 64},
        "exclusions": {"exact_layout_set_sha256": "c" * 64},
        "external_preregistration": {"tag_object": "d" * 40},
        "started_at": "2026-07-23T00:00:00+00:00",
        "updated_at": updated_at or datetime.now(UTC).isoformat(),
        "elapsed_seconds": 90.0,
        "actions_per_second": 321.5,
        "child_trained_actions": 753_664,
        "remediation_trained_actions": 65_536,
        "remaining_remediation_actions": 294_912,
        "lifetime_trained_actions": 1_540_096,
        "optimizer_updates": 3_008,
        "latest_evaluations": evaluations,
        "controller": {
            "terminal_eligible": None,
            "terminal_reasons": [],
            "terminal_pair": None,
        },
        "curriculum": {"recovery": False, "recovery_cause": None},
        "storage": {
            "lineage": {"used_bytes": 1234, "cap_bytes": 2**31},
            "combined_used_bytes": 4321,
        },
        "storage_mount": {
            "path": "/Volumes/T7 Developer",
            "is_mount": True,
            "distinct_from_system": True,
        },
        "frame_revision": 7,
        "lesson_frames": lesson_frames,
    }
    (directory / "status.json").write_text(
        json.dumps(status),
        encoding="utf-8",
    )
    return directory


def test_dashboard_html_explains_u2r_metrics_and_stability() -> None:
    page = dashboard.U2R_DASHBOARD_HTML
    for text in (
        "One apprentice, eleven windows",
        "Remediation learned",
        "Actions remaining",
        "Scientific storage",
        "Frozen four-lesson exam",
        "Stability verdict",
        "Evidence boundary",
        "Immutable segments",
        "What the policy sees",
        "/api/u2r.json",
    ):
        assert text in page


def test_snapshot_follows_active_resume_segment_and_exposes_four_exams(
    tmp_path: Path,
) -> None:
    root = tmp_path / "u2r"
    root.mkdir()
    _segment(root, index=0, phase="interrupted")
    _segment(root, index=1, phase="training")

    snapshot = dashboard.load_u2r_snapshot(root)
    assert snapshot["active_segment_index"] == 1
    assert [item["index"] for item in snapshot["segments"]] == [0, 1]
    active = snapshot["active_status"]
    assert active["phase"] == "training"
    assert active["remediation_trained_actions"] == 65_536
    assert active["remaining_remediation_actions"] == 294_912
    assert active["child_trained_actions"] == 753_664
    assert active["optimizer_updates"] == 3_008
    assert [item["lesson_id"] for item in active["latest_evaluations"]] == [
        lesson_id for lesson_id, _label in dashboard.LESSONS
    ]
    assert all(
        "private_debug" not in item for item in active["latest_evaluations"]
    )
    assert set(active["frame_urls"]) == {*dashboard.LESSON_IDS, "live"}
    assert snapshot["provenance"] == {
        "source_commit": "a" * 40,
        "parent_checkpoint_sha256": "b" * 64,
        "exclusion_set_sha256": "c" * 64,
        "anchor_tag_object": "d" * 40,
        "storage_mount_path": "/Volumes/T7 Developer",
    }
    assert active["storage"]["combined_used_bytes"] == 4321
    assert active["storage_mount"]["is_mount"] is True
    assert snapshot["heartbeat_stale"] is False


def test_snapshot_marks_old_training_heartbeat_stale(tmp_path: Path) -> None:
    root = tmp_path / "u2r"
    root.mkdir()
    _segment(
        root,
        index=0,
        phase="training",
        updated_at="2026-07-20T00:00:00+00:00",
    )
    snapshot = dashboard.load_u2r_snapshot(root)
    assert snapshot["heartbeat_stale"] is True
    assert snapshot["heartbeat_age_seconds"] > 180


def test_frame_resolution_is_status_declared_bounded_and_symlink_free(
    tmp_path: Path,
) -> None:
    root = tmp_path / "u2r"
    root.mkdir()
    directory = _segment(root, index=0, phase="training")

    live = dashboard.resolve_u2r_frame(root, segment_index=0, name="live")
    assert live == (directory / "frames" / "latest.png").resolve()
    lesson = dashboard.resolve_u2r_frame(
        root,
        segment_index=0,
        name="unlock/u2-separated",
    )
    assert lesson.name == "exam-unlock-u2-separated.png"
    with pytest.raises(FileNotFoundError, match="unknown"):
        dashboard.resolve_u2r_frame(
            root,
            segment_index=0,
            name="../../private",
        )

    target = tmp_path / "outside.png"
    target.write_bytes(b"\x89PNG\r\n")
    live.unlink()
    live.symlink_to(target)
    with pytest.raises(FileNotFoundError, match="missing or unsafe"):
        dashboard.resolve_u2r_frame(root, segment_index=0, name="live")


def test_dashboard_rejects_noncontiguous_or_symlinked_segment_chain(
    tmp_path: Path,
) -> None:
    root = tmp_path / "gap"
    root.mkdir()
    _segment(root, index=0, phase="interrupted")
    _segment(root, index=2, phase="training")
    with pytest.raises(dashboard.U2rDashboardError, match="non-contiguous"):
        dashboard.discover_u2r_segments(root)

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    (unsafe / v02_u2r.DEFAULT_RUN_NAME).symlink_to(
        real,
        target_is_directory=True,
    )
    with pytest.raises(dashboard.U2rDashboardError, match="unsafe"):
        dashboard.discover_u2r_segments(unsafe)


def test_dashboard_refuses_port_fallback(tmp_path: Path) -> None:
    root = tmp_path / "u2r"
    root.mkdir()
    _segment(root, index=0, phase="training")
    with pytest.raises(ValueError, match="between 1 and 65535"):
        dashboard.start_u2r_dashboard(root, port=0)


def test_dashboard_server_exposes_read_only_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "u2r"
    root.mkdir()
    _segment(root, index=0, phase="training")
    with socket.socket() as probe:
        probe.bind((dashboard.DEFAULT_HOST, 0))
        port = int(probe.getsockname()[1])
    server = dashboard.start_u2r_dashboard(root, port=port)
    try:
        with urllib.request.urlopen(
            f"http://{dashboard.DEFAULT_HOST}:{port}/api/u2r.json",
            timeout=2,
        ) as response:
            payload = json.load(response)
        assert payload["protocol"] == v02_u2r.PROTOCOL
        assert payload["active_segment_index"] == 0

        request = urllib.request.Request(
            f"http://{dashboard.DEFAULT_HOST}:{port}/api/u2r.json",
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=2)
        assert caught.value.code == 405
    finally:
        server.shutdown()
        server.server_close()
