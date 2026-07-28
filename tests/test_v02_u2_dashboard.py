from __future__ import annotations

import http.client
import json
import socket
from pathlib import Path
from typing import Any

import pytest

from dungeon_apprentice.v02_u2_dashboard import (
    COHORT_DASHBOARD_HTML,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DashboardDataError,
    UnsafeDashboardPathError,
    load_cohort_snapshot,
    resolve_frame,
    start_u2_dashboard,
)


def _manifest() -> dict[str, Any]:
    return {
        "protocol": "dungeon-apprentice-v0.2-u2",
        "cohort_id": "u2-test",
        "phase": "training",
        "active_lineage_id": "lineage-2",
        "started_at": "2026-07-23T04:00:00+00:00",
        "updated_at": "2026-07-23T05:00:00+00:00",
        "elapsed_seconds": 3600,
        "action_cap_per_lineage": 1_048_576,
        "storage": {
            "combined_used_bytes": 1024,
            "combined_cap_bytes": 16 * 1024**3,
        },
        "lineages": [
            {
                "id": f"lineage-{index}",
                "label": f"Apprentice {index}",
                "directory": f"lineage-{index}",
                "seed": 20_260_736 + index,
                "parent_seed": 20_260_724 + index,
                "state": "pending",
            }
            for index in range(1, 4)
        ],
    }


def _status(*, phase: str, trained: int) -> dict[str, Any]:
    return {
        "protocol": "dungeon-apprentice-v0.2-u2",
        "phase": phase,
        "active_lineage": 20260741,
        "parent_checkpoint_sha256": "a" * 64,
        "qualification_sha256": "b" * 64,
        "source_commit": "c" * 40,
        "started_at": "2026-07-23T04:10:00+00:00",
        "updated_at": "2026-07-23T05:00:00+00:00",
        "elapsed_seconds": 3000,
        "fps": 121.5,
        "child_collected_timesteps": trained + 2048,
        "child_trained_timesteps": trained,
        "inherited_trained_actions": 884_736,
        "child_trained_actions": trained,
        "lifetime_trained_actions": 884_736 + trained,
        "optimizer_updates": 1_728 + trained // 512,
        "remaining_action_budget": 1_048_576 - trained,
        "action_cap": 1_048_576,
        "current_lesson_id": "unlock/u2-separated",
        "current_lesson_label": "Separated Unlock",
        "frame_revision": 3,
        "latest_evaluations": [
            {
                "lesson_id": lesson_id,
                "lesson_label": label,
                "successes": 70 + index,
                "episodes": 80,
                "success_rate": (70 + index) / 80,
                "panel_successes": [35 + index, 35],
            }
            for index, (lesson_id, label) in enumerate(
                (
                    ("navigate/full", "Navigate"),
                    ("unlock/u0-visible", "Visible Unlock"),
                    ("unlock/u1-local", "Local Unlock"),
                    ("unlock/u2-separated", "Separated Unlock"),
                )
            )
        ],
        "curriculum": {
            "recovery": True,
            "recovery_cause": "unlock/u1-local",
            "mastered": False,
            "consecutive_passes": 0,
        },
        "controller": {
            "last_decision_child_actions": trained,
            "below_twenty_at_half_budget": False,
        },
        "last_completed_practice_allocation": {
            "target_shares": {
                "navigate/full": 0.1,
                "unlock/u0-visible": 0.1,
                "unlock/u1-local": 0.2,
                "unlock/u2-separated": 0.6,
            },
            "realized_shares": {
                "navigate/full": 0.1,
                "unlock/u0-visible": 0.1,
                "unlock/u1-local": 0.2,
                "unlock/u2-separated": 0.6,
            },
        },
        "last_completed_practice_allocation_valid": True,
        "recent_behavior": {
            "window_episodes": 100,
            "success_rate": 0.42,
            "mean_coverage": 0.61,
            "mean_collisions": 2.5,
            "mean_largest_action_share": 0.31,
        },
        "latest_optimizer": {"policy_gradient_loss": -0.002},
        "latest_evaluated_checkpoint": {
            "path": "checkpoints/rolling/exam.zip",
            "checkpoint_sha256": "d" * 64,
            "child_trained_actions": trained,
        },
        "latest_safe_checkpoint_sha256": "e" * 64,
        "lesson_frames": {
            "navigate/full": "frames/exam-navigate.png",
            "unlock/u0-visible": "frames/exam-u0.png",
            "unlock/u1-local": "frames/exam-u1.png",
            "unlock/u2-separated": "frames/exam-u2.png",
        },
        "storage": {
            "used_bytes": 512,
            "cap_bytes": 2 * 1024**3,
            "remaining_bytes": 2 * 1024**3 - 512,
        },
    }


def _write_cohort(root: Path) -> None:
    root.mkdir()
    (root / "cohort.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    for index in range(1, 4):
        lineage = root / f"lineage-{index}"
        frames = lineage / "frames"
        frames.mkdir(parents=True)
        status = _status(
            phase=("completed", "training", "pending")[index - 1],
            trained=index * 32_768,
        )
        (lineage / "status.json").write_text(json.dumps(status), encoding="utf-8")
        for frame_name in (
            "latest.png",
            "exam-navigate.png",
            "exam-u0.png",
            "exam-u1.png",
            "exam-u2.png",
        ):
            (frames / frame_name).write_bytes(b"\x89PNG\r\n\x1a\n")


def _request(
    server_port: int,
    path: str,
    *,
    method: str = "GET",
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection(DEFAULT_HOST, server_port, timeout=3)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def _available_port() -> int:
    probe = socket.socket()
    try:
        probe.bind((DEFAULT_HOST, 0))
        return int(probe.getsockname()[1])
    finally:
        probe.close()


def test_dashboard_page_is_approachable_mobile_and_covers_u2_evidence() -> None:
    assert '<meta name="viewport"' in COHORT_DASHBOARD_HTML
    assert "Three apprentices, one experiment" in COHORT_DASHBOARD_HTML
    assert "Frozen four-lesson exams" in COHORT_DASHBOARD_HTML
    assert "Panel A" in COHORT_DASHBOARD_HTML
    assert "Panel B" in COHORT_DASHBOARD_HTML
    assert "Practice, recovery, and mastery" in COHORT_DASHBOARD_HTML
    assert "Evidence chain" in COHORT_DASHBOARD_HTML
    assert "Learning and behavior" in COHORT_DASHBOARD_HTML
    assert "Heartbeat delayed" in COHORT_DASHBOARD_HTML
    assert "Closing this page never pauses or stops a trainer." in COHORT_DASHBOARD_HTML
    assert "@media(max-width:560px)" in COHORT_DASHBOARD_HTML
    assert DEFAULT_PORT == 8785


def test_snapshot_reads_all_three_statuses_and_active_lineage_dynamically(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cohort"
    _write_cohort(root)

    first = load_cohort_snapshot(root)
    assert len(first["lineages"]) == 3
    assert first["manifest"]["active_lineage_id"] == "lineage-2"
    assert [item["state"] for item in first["lineages"]] == [
        "completed",
        "training",
        "pending",
    ]
    assert first["lineages"][1]["status"]["child_trained_timesteps"] == 65_536
    assert first["lineages"][1]["status"]["inherited_trained_actions"] == 884_736
    assert first["lineages"][1]["status"]["optimizer_updates"] > 1_728
    assert (
        first["lineages"][1]["status"]["latest_evaluated_checkpoint"][
            "checkpoint_sha256"
        ]
        == "d" * 64
    )
    assert isinstance(first["lineages"][1]["status_age_seconds"], int)
    assert isinstance(first["lineages"][1]["status_stale"], bool)
    assert len(first["lineages"][1]["status"]["latest_evaluations"]) == 4
    assert first["lineages"][1]["status"]["latest_evaluations"][3]["panel_successes"] == [
        38,
        35,
    ]
    assert len(first["lineages"][1]["frame_urls"]) == 5

    updated = _status(phase="mastered", trained=98_304)
    (root / "lineage-2" / "status.json").write_text(json.dumps(updated), encoding="utf-8")
    second = load_cohort_snapshot(root)
    assert second["lineages"][1]["state"] == "mastered"
    assert second["lineages"][1]["status"]["child_trained_timesteps"] == 98_304


def test_snapshot_keeps_missing_future_lineage_visible(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    root.mkdir()
    (root / "cohort.json").write_text(json.dumps(_manifest()), encoding="utf-8")

    snapshot = load_cohort_snapshot(root)

    assert len(snapshot["lineages"]) == 3
    assert all(not item["status_available"] for item in snapshot["lineages"])
    assert all(item["state"] == "pending" for item in snapshot["lineages"])
    assert all("not written yet" in item["status_error"] for item in snapshot["lineages"])


@pytest.mark.parametrize(
    "directory",
    [
        "../outside",
        "/tmp/outside",
    ],
)
def test_manifest_lineage_cannot_escape_cohort(tmp_path: Path, directory: str) -> None:
    root = tmp_path / "cohort"
    root.mkdir()
    manifest = _manifest()
    manifest["lineages"][0]["directory"] = directory
    (root / "cohort.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(UnsafeDashboardPathError):
        load_cohort_snapshot(root)


def test_dashboard_rejects_lineage_and_frame_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "cohort.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    (root / "lineage-1").symlink_to(outside, target_is_directory=True)

    with pytest.raises(UnsafeDashboardPathError, match="symlink"):
        load_cohort_snapshot(root)

    (root / "lineage-1").unlink()
    _write_status_directory = root / "lineage-1"
    (_write_status_directory / "frames").mkdir(parents=True)
    status = _status(phase="training", trained=32_768)
    (_write_status_directory / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (_write_status_directory / "frames" / "latest.png").symlink_to(
        outside / "secret.png"
    )
    with pytest.raises(UnsafeDashboardPathError, match="symlink"):
        load_cohort_snapshot(root)


def test_frame_resolver_serves_only_declared_png_inside_lineage(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    _write_cohort(root)

    latest = resolve_frame(root, lineage_id="lineage-2", name="latest")
    exam = resolve_frame(root, lineage_id="lineage-2", name="unlock/u2-separated")

    assert latest == root / "lineage-2" / "frames" / "latest.png"
    assert exam == root / "lineage-2" / "frames" / "exam-u2.png"
    with pytest.raises(FileNotFoundError, match="unknown frame"):
        resolve_frame(root, lineage_id="lineage-2", name="../../cohort.json")

    status = _status(phase="training", trained=32_768)
    status["lesson_frames"]["unlock/u2-separated"] = "../../cohort.json"
    (root / "lineage-2" / "status.json").write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(UnsafeDashboardPathError):
        resolve_frame(root, lineage_id="lineage-2", name="unlock/u2-separated")


def test_server_is_read_only_dynamic_and_does_not_publish_files(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    _write_cohort(root)
    existing_files = {path.relative_to(root) for path in root.rglob("*") if path.is_file()}
    server = start_u2_dashboard(root, port=_available_port())
    try:
        status, headers, page = _request(server.server_port, "/")
        assert status == 200
        assert headers["Cache-Control"] == "no-store"
        assert b"Three apprentices, one experiment" in page

        status, _headers, payload = _request(server.server_port, "/api/cohort.json")
        assert status == 200
        snapshot = json.loads(payload)
        assert len(snapshot["lineages"]) == 3

        frame_url = snapshot["lineages"][1]["frame_urls"]["latest"]
        status, headers, frame = _request(server.server_port, frame_url)
        assert status == 200
        assert headers["Content-Type"] == "image/png"
        assert frame.startswith(b"\x89PNG")

        status, _headers, payload = _request(
            server.server_port,
            "/api/cohort.json",
            method="POST",
        )
        assert status == 405
        assert json.loads(payload)["error"] == "read-only dashboard"
    finally:
        server.shutdown()
        server.server_close()

    assert not (root / "index.html").exists()
    assert {path.relative_to(root) for path in root.rglob("*") if path.is_file()} == existing_files


def test_server_refuses_port_zero_and_occupied_port_without_fallback(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    _write_cohort(root)
    with pytest.raises(ValueError, match="port 0 fallback is forbidden"):
        start_u2_dashboard(root, port=0)

    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind((DEFAULT_HOST, 0))
    blocker.listen()
    occupied_port = int(blocker.getsockname()[1])
    try:
        with pytest.raises(OSError):
            start_u2_dashboard(root, port=occupied_port)
    finally:
        blocker.close()


def test_manifest_requires_exactly_three_unique_lineages(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    root.mkdir()
    manifest = _manifest()
    manifest["lineages"].pop()
    (root / "cohort.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DashboardDataError, match="exactly 3"):
        load_cohort_snapshot(root)

    manifest = _manifest()
    manifest["lineages"][2]["id"] = "lineage-2"
    (root / "cohort.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DashboardDataError, match="duplicate lineage id"):
        load_cohort_snapshot(root)
