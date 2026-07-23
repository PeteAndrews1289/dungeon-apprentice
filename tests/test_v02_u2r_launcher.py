from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from dungeon_apprentice import u2r_anchor, v02_u2r

REPOSITORY = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY / "scripts" / "run_v02_u2r.sh"


def test_u2r_launcher_has_valid_zsh_syntax_without_running() -> None:
    zsh = shutil.which("zsh")
    if zsh is None:
        pytest.skip("zsh is unavailable on this host")
    subprocess.run(
        [zsh, "-n", str(LAUNCHER)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert os.access(LAUNCHER, os.X_OK)


def test_launcher_freezes_exact_storage_parent_and_terminal_budget() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "set -euo pipefail" in source
    assert (
        'run_root="$dungeon_root/u2r-stability-r1-20260723"' in source
    )
    assert (
        'media_root="$dungeon_root/u2r-stability-r1-media-20260723"' in source
    )
    assert 'initial_run_name="v02-u2r-r1-seed-20260745"' in source
    assert 'training_protocol="dungeon-apprentice-v0.2-u2r-stability-r1"' in source
    assert (
        "u2-separated-20260723/v02-u2-seed-20260745/checkpoints/"
        "mastered-separated-unlock.zip"
    ) in source
    for digest in (
        v02_u2r.SOURCE_ARCHIVE_SHA256,
        v02_u2r.SOURCE_SIDECAR_SHA256,
        v02_u2r.SOURCE_INTEGRITY_SHA256,
        v02_u2r.SOURCE_MANIFEST_SHA256,
        v02_u2r.U2_CONFIRMATION_REPORT_SHA256,
        v02_u2r.U2_CONFIRMATION_ATTEMPT_SHA256,
    ):
        assert digest in source
    for argument in (
        "--seed 20260749",
        "--additional-budget 360448",
        "--workers 4",
        "--rollout-steps 512",
        "--batch-size 256",
        "--n-epochs 4",
        "--evaluation-every 32768",
        "--evaluation-seeds 80",
        "--keep-rolling-exams 11",
        '--minimum-free-gib "$minimum_free_gib"',
    ):
        assert argument in source


def test_launcher_is_fail_closed_before_target_creation() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "accepts only an optional --resume" in source
    assert "git status --porcelain=v1 --untracked-files=all" in source
    assert source.count("assert_source_unchanged") >= 2
    assert "os.path.ismount(volume)" in source
    assert "metadata.st_dev == parent_metadata.st_dev" in source
    assert "assert_regular_ancestor_chain" in source
    assert source.count("assert_frozen_file") >= 7
    assert "assert_no_active_neural_trainer" in source
    assert source.count("assert_free_space") >= 3
    assert '[[ -e "$initial_target" || -L "$initial_target" ]]' in source
    assert '[[ -e "$target" || -L "$target" ]]' in source
    assert "minimum_free_gib=25" in source
    assert "u2_trainer_supervisor.py" in source
    assert 'trainer_module="dungeon_apprentice.v02_u2r"' in source
    assert '"$repository/.venv/bin/python" -m "$trainer_module"' in source
    assert "dungeon-train-v02-u2r" not in source
    assert "/usr/bin/caffeinate -ims" in source
    assert "trap record_launcher_exit EXIT" in source
    assert "trap request_launcher_stop INT TERM" in source

    anchor_check = source.index("verify_external_anchor(")
    sampler_preflight = source.index("preflight_u2r_training_layout_sampler(")
    resume_authentication = source.index("select_resume_plan(")
    run_root_creation = source.index('/bin/mkdir -m 0755 "$run_root"')
    trainer_start = source.index('"$repository/.venv/bin/python" "$supervisor"')
    assert anchor_check < sampler_preflight < run_root_creation < trainer_start
    assert resume_authentication < run_root_creation


def test_launcher_verifies_remote_preregistration_without_opening_seeds() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "protocol_document_sha256" in source
    assert "build_u2r_forbidden_layout_hashes" in source
    assert "expected_exclusions=exclusions" in source
    assert "verify_failed_r0_launch()" in source
    assert "preflight_u2r_training_layout_sampler" in source
    assert "U2R_LAYOUT_RESAMPLE_ATTEMPTS" in source
    assert "verify_u2r_parent()" in source
    assert "expected_source_commit=source_commit" in source
    assert u2r_anchor.ANCHOR_TAG not in source
    assert "publish_external_anchor" not in source
    assert "authorize_u2_seed" not in source
    assert "_post_training_confirmation_seed_access" not in source
    assert "15_240_000" not in source
    assert "15_279_999" not in source


def test_launcher_resume_has_no_arbitrary_checkpoint_or_name() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "select_resume_plan" in source
    assert '"checkpoint",' in source
    assert '"next_run_name",' in source
    assert '"next_segment_index",' in source
    assert 'resume_arguments=(--resume "$resume_checkpoint")' in source
    assert "--resume-checkpoint" not in source
    assert "--run-name" in source
    assert 'run_name=${resume_fields[3]}' in source
    assert 'expected_segment_index=${resume_fields[4]}' in source
    assert "resume-N" not in source
    assert (
        'next_name = f"{v02_u2r.DEFAULT_RUN_NAME}-resume-{next_index}"'
        in Path(u2r_anchor.__file__).read_text(encoding="utf-8")
    )
    assert (
        "v02_u2r.SEGMENT_SEED_OFFSET"
        in Path(u2r_anchor.__file__).read_text(encoding="utf-8")
    )
    assert (
        '"segment_seed_offset": v02_u2r.SEGMENT_SEED_OFFSET'
        in Path(u2r_anchor.__file__).read_text(encoding="utf-8")
    )
    planner_source = Path(u2r_anchor.__file__).read_text(encoding="utf-8")
    assert "interruption.json" in planner_source
    assert "_verify_interruption_evidence(" in planner_source


def test_launcher_starts_fixed_read_only_u2r_dashboard() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'dashboard_module="dungeon_apprentice.v02_u2r_dashboard"' in source
    assert "dashboard_port=8786" in source
    assert "start_new_session=True" in source
    assert "/api/u2r.json" in source
    assert "http://127.0.0.1:$dashboard_port/" in source
    assert "active_segment_index" in source
    assert 'active.get("source", {}).get("commit") == source_commit' in source
    assert 'print -r -- "$dashboard_pid" >"$target/dashboard.pid"' in source
    assert "stop_prior_dashboard_for_resume" in source
    assert "refusing to stop an unrecognized process" in source


def test_launcher_accepts_only_terminal_scientific_or_failure_phases() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert '{"eligible", "failed", "interrupted", "crashed"}' in source
    assert 'if [[ "$terminal_phase" == "interrupted" ]]' in source
    assert (
        'if [[ "$terminal_phase" == "eligible" || '
        '"$terminal_phase" == "failed" ]]'
        in source
    )
    assert "verify_terminal_report(run_directory)" in source
    assert (
        'if [[ "$terminal_phase" == "crashed" || '
        '"$terminal_evidence_verified" != true ]]'
        in source
    )
    assert 'if [[ "$trainer_status" -ne 0 ]]' in source
    assert "the terminal result remains valid" in source
    assert (
        '"$terminal_phase" == "crashed" || "$trainer_status" -ne 0'
        not in source
    )
    assert "U2r reached its exact terminal budget with verdict" in source
