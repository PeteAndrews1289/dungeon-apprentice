from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY / "scripts" / "run_v04_ineffective_trace_stage_a.sh"


def _function_block(first: str, following: str) -> str:
    source = LAUNCHER.read_text(encoding="utf-8")
    start = source.index(f"{first}() {{")
    end = source.index(f"{following}() {{", start)
    return source[start:end]


def test_v04_launcher_has_valid_zsh_syntax_without_running() -> None:
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


def test_v04_launcher_freezes_identity_order_and_no_resume() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for text in (
        'run_root="$dungeon_root/v04-ineffective-trace-stage-a-20260724"',
        'media_root="$dungeon_root/v04-ineffective-trace-stage-a-media-20260724"',
        "qualifications/v0.4-ineffective-trace-stage-a-20260724/report.json",
        'training_protocol="dungeon-apprentice-v0.4-ineffective-trace-architecture"',
        'cohort_id="v0.4-ineffective-trace-stage-a-20260724"',
        'training_tag="ineffective-trace-architecture-v0.4-stage-a-20260724"',
        'trainer_module="dungeon_apprentice.v04_ineffective_trace_train"',
        'dashboard_module="dungeon_apprentice.v04_ineffective_trace_dashboard"',
        'manifest_helper="$repository/scripts/v04_ineffective_trace_manifest.py"',
        "dashboard_port=8792",
        "minimum_free_gib=25",
        "verify_ineffective_trace_qualification",
        "r3_terminal_evidence_sha256",
        '{"trace-sham", "ineffective-trace"}',
        "--device cpu",
        "manifest_command seal-process-closeout",
        "manifest_command abort-closeout",
        "manifest_command finalize",
        "assert_single_active_chain",
        "assert_no_competing_training_chain",
        "dungeon-train-v04-ineffective-trace",
        "dungeon-train-v03-action-effect",
        "dungeon-train-v02-u2s",
    ):
        assert text in source
    assert "--resume" not in source
    assert "dungeon_apprentice.v03_action_effect_train" in source
    assert "/usr/bin/pgrep -f" not in source

    qualification = source.index(
        "  assert_v04_qualification",
        source.index("trap record_launcher_exit"),
    )
    root_creation = source.index('/bin/mkdir -m 0755 "$run_root"')
    manifest_creation = source.index("manifest_command create")
    dashboard_start = source.index("start_dashboard", manifest_creation)
    arm_claim = source.index("manifest_command start-arm", dashboard_start)
    trainer_start = source.index('"$repository/.venv/bin/python" "$trainer_supervisor"')
    finalization = source.rindex("manifest_command finalize")
    assert (
        qualification
        < root_creation
        < manifest_creation
        < dashboard_start
        < arm_claim
        < trainer_start
        < finalization
    )


@pytest.mark.parametrize("planner_output", ("", "not-json", "[]"))
def test_v04_inactive_terminalizer_ignores_malformed_plans(
    planner_output: str,
) -> None:
    zsh = shutil.which("zsh")
    if zsh is None:
        pytest.skip("zsh is unavailable")
    terminalizer = _function_block(
        "terminalize_inactive_cohort",
        "stop_caffeine",
    )
    harness = (
        "set -euo pipefail\n"
        f"repository={shlex.quote(str(REPOSITORY))}\n"
        'run_root="/unused/cohort"\n'
        'source_commit="source"\n'
        'tag_object="tag"\n'
        "cohort_created=true\n"
        "launcher_finalized=false\n"
        "manifest_command() {\n"
        '  [[ "$1" == next ]] || return 99\n'
        f"  print -r -- {shlex.quote(planner_output)}\n"
        "}\n"
        f"{terminalizer}\n"
        "terminalize_inactive_cohort 1\n"
    )
    result = subprocess.run(
        [zsh, "-c", harness],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        cwd=REPOSITORY,
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_v04_launcher_dashboard_is_read_only_and_versioned() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for text in (
        'f"http://127.0.0.1:{port}/api/v04.json"',
        '"--run-root"',
        '"--host"',
        '"127.0.0.1"',
        '"--port"',
        "start_new_session=True",
        "Read-only dashboard remains available",
    ):
        assert text in source
    assert "/api/v03.json" not in source
