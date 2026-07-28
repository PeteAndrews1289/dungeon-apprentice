from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dungeon_apprentice.v02_u2_dashboard import load_cohort_snapshot

REPOSITORY = Path(__file__).resolve().parents[1]
QUALIFICATION_LAUNCHER = REPOSITORY / "scripts" / "run_v02_u2_qualification.sh"
COHORT_LAUNCHER = REPOSITORY / "scripts" / "run_v02_u2_cohort.sh"
MANIFEST_HELPER = REPOSITORY / "scripts" / "u2_cohort_manifest.py"
TRAINER_SUPERVISOR = REPOSITORY / "scripts" / "u2_trainer_supervisor.py"


@pytest.mark.parametrize("launcher", (QUALIFICATION_LAUNCHER, COHORT_LAUNCHER))
def test_u2_launchers_have_valid_zsh_syntax_without_running(launcher: Path) -> None:
    zsh = shutil.which("zsh")
    if zsh is None:
        pytest.skip("zsh is unavailable on this host")
    subprocess.run(
        [zsh, "-n", str(launcher)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_qualification_launcher_is_exact_one_shot_and_fail_closed() -> None:
    source = QUALIFICATION_LAUNCHER.read_text(encoding="utf-8")
    assert "set -euo pipefail" in source
    assert 'output_directory="$qualification_parent/v0.2-u2-20260723"' in source
    assert "dungeon-qualify-v02-u2" in source
    assert "--output-directory" not in source
    assert '--launch-token "$launch_token"' in source
    assert "secrets.token_hex(32)" in source
    assert '--acknowledge "$acknowledgement"' in source
    assert "sealed_acknowledgement" in source
    assert "OPEN U2 SEALED PREFLIGHT" not in source
    assert "git status --porcelain=v1 --untracked-files=all" in source
    assert source.count("assert_source_unchanged") >= 3
    assert '[[ -e "$output_directory" ]]' in source
    assert "qualification parent must already exist" in source
    assert "accepts no overrides" in source
    assert "qualification_already_exists" in source
    assert "recovering only the external anchor" in source
    assert "qualification_evidence_for_anchor" in source
    assert "publish_external_anchor" in source
    assert "os.path.ismount(volume)" in source
    assert "metadata.st_dev == parent_metadata.st_dev" in source
    assert "assert_no_active_neural_trainer" in source
    assert "minimum_free_gib=25" in source
    assert '/usr/bin/caffeinate -ims "$qualifier"' in source


def test_cohort_launcher_freezes_all_three_independent_lineages() -> None:
    source = COHORT_LAUNCHER.read_text(encoding="utf-8")
    assert 'run_root="$dungeon_root/u2-separated-20260723"' in source
    assert 'media_root="$dungeon_root/u2-separated-media-20260723"' in source
    assert (
        'qualification_report="$dungeon_root/qualifications/v0.2-u2-20260723/report.json"' in source
    )
    for seed in (20260737, 20260741, 20260745):
        assert source.count(str(seed)) >= 2
    for digest in (
        "bcce9b8251e97ed4fddda32871c891c3783c057bbb1f89deedb3a3d32058102a",
        "2a300927b48f966d5f6ddfeefe13d2e444da1e5c70bcd54e86abd6a9b2d1830b",
        "3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104",
    ):
        assert digest in source
    assert "for (( index = start_index; index <= ${#seeds}; index++ ))" in source
    assert "/usr/bin/caffeinate -ims >/dev/null 2>&1 &" in source
    assert '"$repository/.venv/bin/python" "$trainer_supervisor"' in source
    assert "--workers 4" in source
    assert "--child-budget 1048576" in source
    assert "--evaluation-every 32768" in source
    assert "--evaluation-seeds 80" in source
    assert "minimum_free_gib=25" in source
    assert '--media-directory "$media_root"' in source
    assert "U2 media target" in source


def test_cohort_launcher_separates_dashboard_and_guards_protocol() -> None:
    source = COHORT_LAUNCHER.read_text(encoding="utf-8")
    assert "dashboard_port=8785" in source
    assert "subprocess.Popen" in source
    assert "start_new_session=True" in source
    assert "http://127.0.0.1:$dashboard_port/" in source
    assert 'manifest.get("cohort_id") != "v0.2-u2-separated-20260723"' in source
    assert 'manifest.get("source_commit") != source_commit' in source
    assert "active_neural_trainers" in source
    assert source.count("assert_no_active_neural_trainer") >= 4
    assert source.count("assert_source_unchanged") >= 4
    assert source.count("assert_free_space") >= 4
    assert '[[ -e "$run_root" || -L "$run_root" ]]' in source
    assert "verify_external_anchor" in source
    assert "verify_u2_qualification_report" in source
    assert "expected_source_commit=sys.argv[3]" in source
    assert "expected_sha256=anchor.report_sha256" in source
    assert "expected_attempt_id=anchor.attempt_id" in source
    assert "expected_claim_id=anchor.claim_id" in source
    assert "assert_regular_ancestor_chain" in source
    assert "accepts only an optional --resume" in source
    assert "os.path.ismount(volume)" in source
    assert "metadata.st_dev == parent_metadata.st_dev" in source
    assert "trap record_unexpected_exit EXIT" in source
    assert "trap request_launcher_stop INT TERM" in source
    assert "segment-start" in source
    assert "segment-finish" in source
    assert "resume-plan" in source
    assert "supervisor_pid=$!" in source
    assert "local status=" not in source
    assert "local path=" not in source


def _run_helper(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(MANIFEST_HELPER), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_manifest_helper_creates_and_atomically_advances_exact_cohort(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cohort"
    root.mkdir()
    commit = "a" * 40
    created = _run_helper(
        "create",
        "--root",
        str(root),
        "--source-commit",
        commit,
    )
    assert created.returncode == 0, created.stderr
    path = root / "cohort.json"
    first = json.loads(path.read_text(encoding="utf-8"))
    assert first["protocol"] == "dungeon-apprentice-v0.2-u2"
    assert first["schema_version"] == 2
    assert first["source_commit"] == commit
    assert first["active_lineage_id"] is None
    assert first["cohort_action_cap"] == 3_145_728
    assert [lineage["id"] for lineage in first["lineages"]] == [
        "lineage-1",
        "lineage-2",
        "lineage-3",
    ]
    assert [lineage["seed"] for lineage in first["lineages"]] == [
        20260737,
        20260741,
        20260745,
    ]
    assert [lineage["state"] for lineage in first["lineages"]] == [
        "pending",
        "pending",
        "pending",
    ]
    assert all(lineage["segments"] == [] for lineage in first["lineages"])
    assert first["history"][0]["event"] == "cohort_created"
    snapshot = load_cohort_snapshot(root)
    assert len(snapshot["lineages"]) == 3
    assert all(lineage["state"] == "pending" for lineage in snapshot["lineages"])

    updated = _run_helper(
        "update",
        "--root",
        str(root),
        "--source-commit",
        commit,
        "--phase",
        "training",
        "--active-lineage-id",
        "lineage-2",
        "--state",
        "completed",
        "--state",
        "training",
        "--state",
        "pending",
    )
    assert updated.returncode == 0, updated.stderr
    second = json.loads(path.read_text(encoding="utf-8"))
    assert second["active_lineage_id"] == "lineage-2"
    assert [lineage["state"] for lineage in second["lineages"]] == [
        "completed",
        "training",
        "pending",
    ]
    assert list(root.glob(".cohort.json.*.tmp")) == []
    assert second["history"][-1]["event"] == "cohort_state"


def test_manifest_helper_refuses_reuse_drift_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    root.mkdir()
    commit = "b" * 40
    assert (
        _run_helper(
            "create",
            "--root",
            str(root),
            "--source-commit",
            commit,
        ).returncode
        == 0
    )
    assert (
        _run_helper(
            "create",
            "--root",
            str(root),
            "--source-commit",
            commit,
        ).returncode
        != 0
    )
    assert (
        _run_helper(
            "update",
            "--root",
            str(root),
            "--source-commit",
            "c" * 40,
            "--phase",
            "training",
            "--state",
            "pending",
            "--state",
            "pending",
            "--state",
            "pending",
        ).returncode
        != 0
    )

    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    assert (
        _run_helper(
            "create",
            "--root",
            str(alias),
            "--source-commit",
            commit,
        ).returncode
        != 0
    )


def _write_safe_resume_bundle(
    root: Path,
    *,
    directory: str,
    commit: str,
    child_actions: int,
    kind: str = "resume",
) -> Path:
    checkpoint_directory = root / directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_directory / f"{kind}.zip"
    checkpoint.write_bytes(f"schema-4-{child_actions}".encode())
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    sidecar = checkpoint.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "protocol": "dungeon-apprentice-v0.2-u2",
                "resume_eligible": True,
                "kind": kind,
                "checkpoint_sha256": checkpoint_digest,
                "source": {"dirty": False, "commit": commit},
                "parent": {
                    "u2_child_seed": 20260737,
                    "checkpoint_sha256": (
                        "bcce9b8251e97ed4fddda32871c891c3783c057bbb1f89deedb3a3d32058102a"
                    ),
                },
                "qualification": {"source_commit": commit},
                "segment": {
                    "index": 0,
                    "lineage_algorithm_seed": 20260737,
                },
                "progress": {
                    "child_trained_actions": child_actions,
                    "trained_actions": 884_736 + child_actions,
                    "collected_actions": 884_736 + child_actions,
                    "remaining_child_actions": 1_048_576 - child_actions,
                },
            }
        ),
        encoding="utf-8",
    )
    sidecar_digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    checkpoint.with_suffix(".integrity.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "protocol": "dungeon-apprentice-v0.2-u2",
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": checkpoint_digest,
                "sidecar": sidecar.name,
                "sidecar_sha256": sidecar_digest,
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


def test_manifest_resume_plan_is_strict_append_only_and_uses_latest_safe_bundle(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cohort"
    root.mkdir()
    commit = "d" * 40
    assert (
        _run_helper(
            "create",
            "--root",
            str(root),
            "--source-commit",
            commit,
        ).returncode
        == 0
    )
    first_directory = "v02-u2-seed-20260737"
    started = _run_helper(
        "segment-start",
        "--root",
        str(root),
        "--source-commit",
        commit,
        "--lineage-id",
        "lineage-1",
        "--directory",
        first_directory,
    )
    assert started.returncode == 0, started.stderr
    initial = _write_safe_resume_bundle(
        root,
        directory=first_directory,
        commit=commit,
        child_actions=0,
        kind="initial",
    )
    latest = _write_safe_resume_bundle(
        root,
        directory=first_directory,
        commit=commit,
        child_actions=65_536,
    )
    (root / first_directory / "status.json").write_text(
        json.dumps(
            {
                "phase": "interrupted",
                "child_trained_timesteps": 65_536,
            }
        ),
        encoding="utf-8",
    )
    finished = _run_helper(
        "segment-finish",
        "--root",
        str(root),
        "--source-commit",
        commit,
        "--lineage-id",
        "lineage-1",
        "--directory",
        first_directory,
        "--segment-state",
        "interrupted",
    )
    assert finished.returncode == 0, finished.stderr

    planned = _run_helper(
        "resume-plan",
        "--root",
        str(root),
        "--source-commit",
        commit,
    )
    assert planned.returncode == 0, planned.stderr
    plan = json.loads(planned.stdout)
    assert plan["lineage_id"] == "lineage-1"
    assert plan["directory"] == "v02-u2-seed-20260737-segment-001"
    assert plan["resume_checkpoint"] == str(latest)
    assert plan["resume_checkpoint"] != str(initial)
    assert plan["child_trained_actions"] == 65_536

    successor = _run_helper(
        "segment-start",
        "--root",
        str(root),
        "--source-commit",
        commit,
        "--lineage-id",
        "lineage-1",
        "--directory",
        plan["directory"],
        "--resume-checkpoint",
        plan["resume_checkpoint"],
    )
    assert successor.returncode == 0, successor.stderr
    manifest = json.loads((root / "cohort.json").read_text(encoding="utf-8"))
    lineage = manifest["lineages"][0]
    assert lineage["directory"] == plan["directory"]
    assert [segment["state"] for segment in lineage["segments"]] == [
        "interrupted",
        "training",
    ]
    assert lineage["segments"][1]["resume_checkpoint"] == str(latest.relative_to(root))
    assert [event["event"] for event in manifest["history"]][-3:] == [
        "segment_started",
        "segment_finished",
        "segment_started",
    ]


def test_manifest_resume_plan_rejects_tampered_safe_bundle(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    root.mkdir()
    commit = "e" * 40
    assert (
        _run_helper(
            "create",
            "--root",
            str(root),
            "--source-commit",
            commit,
        ).returncode
        == 0
    )
    directory = "v02-u2-seed-20260737"
    assert (
        _run_helper(
            "segment-start",
            "--root",
            str(root),
            "--source-commit",
            commit,
            "--lineage-id",
            "lineage-1",
            "--directory",
            directory,
        ).returncode
        == 0
    )
    checkpoint = _write_safe_resume_bundle(
        root,
        directory=directory,
        commit=commit,
        child_actions=32_768,
    )
    (root / directory / "status.json").write_text(
        json.dumps(
            {
                "phase": "crashed",
                "child_trained_timesteps": 32_768,
            }
        ),
        encoding="utf-8",
    )
    assert (
        _run_helper(
            "segment-finish",
            "--root",
            str(root),
            "--source-commit",
            commit,
            "--lineage-id",
            "lineage-1",
            "--directory",
            directory,
            "--segment-state",
            "crashed",
        ).returncode
        == 0
    )
    checkpoint.write_bytes(b"tampered")
    planned = _run_helper(
        "resume-plan",
        "--root",
        str(root),
        "--source-commit",
        commit,
    )
    assert planned.returncode != 0
    assert "no schema-4 safe resume bundle" in planned.stderr


def test_trainer_supervisor_persists_child_pid_and_terminal_status(
    tmp_path: Path,
) -> None:
    state = tmp_path / "supervisor.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(TRAINER_SUPERVISOR),
            "--state-path",
            str(state),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 7
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert payload["state"] == "exited"
    assert type(payload["trainer_pid"]) is int
    assert payload["trainer_pid"] > 0
    assert payload["exit_status"] == 7


def test_trainer_supervisor_reaps_child_if_initial_state_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "u2_trainer_supervisor_fixture",
        TRAINER_SUPERVISOR,
    )
    assert spec is not None and spec.loader is not None
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    real_popen = supervisor.subprocess.Popen
    children = []

    def capture_child(*args: object, **kwargs: object) -> subprocess.Popen:
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    def fail_initial_state(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated supervisor-state I/O failure")

    monkeypatch.setattr(supervisor.subprocess, "Popen", capture_child)
    monkeypatch.setattr(supervisor, "_atomic_write", fail_initial_state)
    with pytest.raises(OSError, match="simulated"):
        supervisor.run_supervised(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            state_path=tmp_path / "supervisor.json",
        )

    assert len(children) == 1
    assert children[0].poll() is not None
