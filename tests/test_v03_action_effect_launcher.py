from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY / "scripts" / "run_v03_action_effect_stage_a_r1.sh"


def test_v03_launcher_has_valid_zsh_syntax_without_running() -> None:
    zsh = shutil.which("zsh")
    if zsh is None:
        pytest.skip("zsh is unavailable")
    subprocess.run(
        [zsh, "-n", str(LAUNCHER)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert os.access(LAUNCHER, os.X_OK)


def test_v03_launcher_freezes_release_and_storage_identity() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for text in (
        "set -euo pipefail",
        'run_root="$dungeon_root/v03-action-effect-stage-a-r1-20260724"',
        'media_root="$dungeon_root/v03-action-effect-stage-a-r1-media-20260724"',
        "qualifications/v0.3-action-effect-stage-a-r1-20260724/report.json",
        'training_protocol="dungeon-apprentice-v0.3-action-effect-architecture"',
        'cohort_id="v0.3-action-effect-stage-a-r1-20260724"',
        'training_tag="action-effect-architecture-v0.3-stage-a-r1-20260724"',
        'expected_origin="https://github.com/PeteAndrews1289/dungeon-apprentice.git"',
        'trainer_module="dungeon_apprentice.v03_action_effect_train"',
        'dashboard_module="dungeon_apprentice.v03_action_effect_dashboard"',
        'manifest_helper="$repository/scripts/v03_action_effect_manifest.py"',
        "dashboard_port=8789",
        "minimum_free_gib=25",
        "git status --porcelain=v1 --untracked-files=all",
        "git cat-file -t",
        "git ls-remote --tags origin",
        "os.path.ismount(volume)",
        "metadata.st_dev == parent_metadata.st_dev",
        "verify_action_effect_qualification",
        "expected_tag_object=tag_object",
        "assert_frozen_parent",
        "assert_free_space",
        "assert_source_unchanged",
    ):
        assert text in source
    assert source.count("assert_source_unchanged") >= 3
    assert source.count("assert_free_space") >= 4
    assert "--resume" not in source


def test_v03_launcher_is_sequential_fresh_only_and_fail_closed() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for text in (
        "accepts no options; interruption makes both fresh twins terminal",
        "manifest_command next",
        "manifest_command start-arm",
        "manifest_command finish-arm",
        "manifest_command seal-process-closeout",
        "manifest_command finalize",
        '[[ "$active_arm" != sham && "$active_arm" != action-effect ]]',
        '[[ "$attempt" != 0 ]]',
        '[[ -e "$run_directory" || -L "$run_directory" ]]',
        '[[ -e "$media_directory" || -L "$media_directory" ]]',
        "--qualification-report",
        "--run-dir",
        "--media-dir",
        "--cohort-contract",
        "--device cpu",
        "u2_trainer_supervisor.py",
        "/usr/bin/caffeinate -ims",
        "assert_single_active_chain",
        "terminalize_inactive_cohort",
        "terminal operationally_incomplete",
        "can never be resumed or reused",
    ):
        assert text in source

    qualification = source.index("assert_v03_qualification")
    cohort_creation = source.index('/bin/mkdir -m 0755 "$run_root"')
    manifest_creation = source.index("manifest_command create")
    dashboard_start = source.index("start_dashboard", manifest_creation)
    arm_start = source.index("manifest_command start-arm")
    run_reuse_guard = source.index(
        '[[ -e "$run_directory" || -L "$run_directory" ]]'
    )
    run_creation = source.index('/bin/mkdir -m 0755 "$run_directory"')
    media_creation = source.index('/bin/mkdir -m 0755 "$media_directory"')
    trainer_start = source.index(
        '"$repository/.venv/bin/python" "$trainer_supervisor"'
    )
    finalization = source.rindex("manifest_command finalize")
    trap_release = source.rindex("trap - EXIT INT TERM")

    assert qualification < cohort_creation < manifest_creation < dashboard_start
    assert (
        arm_start
        < run_reuse_guard
        < run_creation
        < media_creation
        < trainer_start
        < finalization
        < trap_release
    )
    inactive_terminalizer = source.index("terminalize_inactive_cohort()")
    trap_terminalizer = source.index(
        'terminalize_inactive_cohort "$exit_code"'
    )
    assert inactive_terminalizer < trap_terminalizer < cohort_creation
    terminalizer_body = source[
        inactive_terminalizer : source.index(
            "stop_caffeine()",
            inactive_terminalizer,
        )
    ]
    assert "manifest_command finalize" not in terminalizer_body
    assert "awaiting closeout" in terminalizer_body
    assert (
        'manifest_command create \\\n'
        '  --root "$run_root" \\\n'
        '  --media-root "$media_root" \\\n'
        '  --qualification-report "$qualification_report"'
    ) in source
    assert (
        '"$repository/.venv/bin/python" -m "$trainer_module" train-arm \\\n'
        '      --arm "$active_arm" \\\n'
        '      --qualification-report "$qualification_report" \\\n'
        '      --run-dir "$run_directory" \\\n'
        '      --media-dir "$media_directory" \\\n'
        '      --cohort-contract "$contract"'
    ) in source


def test_v03_launcher_keeps_one_chain_and_read_only_dashboard() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for text in (
        "assert_no_competing_training_chain",
        "active_neural_trainers",
        "active_trainer_supervisors",
        "active_caffeinates",
        "trainer_parent=",
        '[[ "$trainer_parent" == "$supervisor_pid" ]]',
        '[[ "$trainers" == "$trainer_pid" ]]',
        '[[ "$supervisors" == "$supervisor_pid" ]]',
        '[[ "$caffeinates" == "$caffeine_pid" ]]',
        'lsof -nP -t -iTCP:"$dashboard_port" -sTCP:LISTEN',
        'f"http://127.0.0.1:{port}/api/v03.json"',
        '"--run-root"',
        '"--host"',
        '"127.0.0.1"',
        '"--port"',
        "start_new_session=True",
        "assert_dashboard_healthy",
        "Read-only dashboard remains available",
    ):
        assert text in source
    assert "kill \"$dashboard_pid\"" in source
    assert "stop_dashboard" not in source
    exit_trap = source[
        source.index("record_launcher_exit()") : source.index(
            "request_launcher_stop()"
        )
    ]
    assert "dashboard_pid" not in exit_trap


def test_v03_launcher_finalizes_only_after_both_arms() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    loop = source.index("while true; do")
    planner_done = source.index(
        'if [[ "${plan_fields[1]}" == done ]]',
        loop,
    )
    requires_closeout = source.index(
        '[[ "${plan_fields[2]}" != true ]]',
        planner_done,
    )
    arm_finish = source.index(
        "manifest_command finish-arm",
        source.index("terminal_phase="),
    )
    loop_end = source.index("\ndone\n", arm_finish)
    process_seal = source.rindex("manifest_command seal-process-closeout")
    finalization = source.rindex("manifest_command finalize")
    stop_caffeine = source.rindex(
        "stop_caffeine\nmanifest_command seal-process-closeout"
    )
    final_health = source.rindex("assert_dashboard_healthy")

    assert loop < planner_done < requires_closeout < arm_finish < loop_end
    assert (
        loop_end
        < stop_caffeine
        < process_seal
        < finalization
        < final_health
    )
    assert (
        'manifest_command seal-process-closeout \\\n'
        '  --root "$run_root" \\\n'
        '  --source-commit "$source_commit" \\\n'
        '  --tag-object "$tag_object" >/dev/null'
    ) in source
