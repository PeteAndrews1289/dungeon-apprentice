from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY / "scripts" / "run_v03_action_effect_stage_a_r2.sh"
TRAINER_SUPERVISOR = REPOSITORY / "scripts" / "u2_trainer_supervisor.py"


def _function_block(first: str, following: str) -> str:
    source = LAUNCHER.read_text(encoding="utf-8")
    start = source.index(f"{first}() {{")
    end = source.index(f"{following}() {{", start)
    return source[start:end]


def _wait_for_json(path: Path, *, timeout: float = 10.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_pid_exit(pid: int, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return
        time.sleep(0.02)
    raise AssertionError(f"process {pid} remained alive")


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
        'run_root="$dungeon_root/v03-action-effect-stage-a-r2-20260724"',
        'media_root="$dungeon_root/v03-action-effect-stage-a-r2-media-20260724"',
        "qualifications/v0.3-action-effect-stage-a-r2-20260724/report.json",
        'training_protocol="dungeon-apprentice-v0.3-action-effect-architecture"',
        'cohort_id="v0.3-action-effect-stage-a-r2-20260724"',
        'training_tag="action-effect-architecture-v0.3-stage-a-r2-20260724"',
        'expected_origin="https://github.com/PeteAndrews1289/dungeon-apprentice.git"',
        'trainer_module="dungeon_apprentice.v03_action_effect_train"',
        'dashboard_module="dungeon_apprentice.v03_action_effect_dashboard"',
        'manifest_helper="$repository/scripts/v03_action_effect_manifest.py"',
        "dashboard_port=8790",
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
    run_reuse_guard = source.index('[[ -e "$run_directory" || -L "$run_directory" ]]')
    run_creation = source.index('/bin/mkdir -m 0755 "$run_directory"')
    media_creation = source.index('/bin/mkdir -m 0755 "$media_directory"')
    trainer_start = source.index('"$repository/.venv/bin/python" "$trainer_supervisor"')
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
    trap_terminalizer = source.index('terminalize_inactive_cohort "$exit_code"')
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
        '"v0.3 could not create its cohort manifest" \\\n'
        "  manifest_command create \\\n"
        '    --root "$run_root" \\\n'
        '    --media-root "$media_root" \\\n'
        '    --qualification-report "$qualification_report"'
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
    assert "/bin/ps -axo pid=,command=" in source
    assert "is_trainer_module" in source
    assert "/usr/bin/pgrep -f" not in source
    assert "require_launcher_step" in source
    assert "cleanup_launcher" in source
    assert "abort_launcher" in source
    assert "shutdown_supervised_chain" in source
    assert "capture_owned_supervisor_identity" in source
    assert "capture_owned_trainer_identity" in source
    assert 'signal_owned_supervisor TERM' in source
    assert source.count("signal_owned_supervisor TERM") == 2
    assert '/bin/kill -KILL -- "-$trainer_pid"' in source
    assert "wait_for_supervisor_exit" in source
    assert (
        '"v0.3 active trainer chain identity check failed" \\\n'
        '      assert_single_active_chain "$supervisor_state"'
    ) in source
    assert 'kill "$dashboard_pid"' in source
    assert "stop_dashboard" not in source
    exit_trap = source[
        source.index("record_launcher_exit()") : source.index("request_launcher_stop()")
    ]
    assert "dashboard_pid" not in exit_trap


def test_process_discovery_counts_real_trainer_not_supervisor_argv(
    tmp_path: Path,
) -> None:
    zsh = shutil.which("zsh")
    if zsh is None:
        pytest.skip("zsh is unavailable")

    package = tmp_path / "dungeon_apprentice"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "v03_action_effect_train.py").write_text(
        "import time\ntry:\n    time.sleep(60)\nexcept KeyboardInterrupt:\n    pass\n",
        encoding="utf-8",
    )
    state_path = tmp_path / "supervisor.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)
    supervisor = subprocess.Popen(
        [
            sys.executable,
            str(TRAINER_SUPERVISOR),
            "--state-path",
            str(state_path),
            "--",
            sys.executable,
            "-m",
            "dungeon_apprentice.v03_action_effect_train",
        ],
        cwd=tmp_path,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    trainer_pid: int | None = None
    try:
        state = _wait_for_json(state_path)
        assert state["supervisor_pid"] == supervisor.pid
        trainer_pid = state["trainer_pid"]
        assert isinstance(trainer_pid, int)

        functions = _function_block(
            "active_neural_trainers",
            "assert_no_competing_training_chain",
        )
        trainers = subprocess.run(
            [zsh, "-c", f"{functions}\nactive_neural_trainers"],
            check=True,
            capture_output=True,
            text=True,
        )
        supervisors = subprocess.run(
            [zsh, "-c", f"{functions}\nactive_trainer_supervisors"],
            check=True,
            capture_output=True,
            text=True,
        )
        trainer_pids = [int(line) for line in trainers.stdout.splitlines() if line.strip()]
        supervisor_pids = [int(line) for line in supervisors.stdout.splitlines() if line.strip()]

        assert trainer_pids == [trainer_pid]
        assert supervisors.stderr == ""
        assert supervisor_pids == [supervisor.pid]
        assert supervisor.pid not in trainer_pids
    finally:
        if supervisor.poll() is None:
            supervisor.terminate()
            try:
                supervisor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                supervisor.kill()
                supervisor.wait(timeout=5)
        if trainer_pid is not None and _pid_exists(trainer_pid):
            os.killpg(trainer_pid, signal.SIGTERM)
            _wait_for_pid_exit(trainer_pid)


def test_explicit_abort_escalates_and_reaps_uncooperative_trainer(
    tmp_path: Path,
) -> None:
    zsh = shutil.which("zsh")
    if zsh is None:
        pytest.skip("zsh is unavailable")

    package = tmp_path / "dungeon_apprentice"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    ready_file = tmp_path / "trainer-ready"
    (package / "v03_action_effect_train.py").write_text(
        "import os\n"
        "import signal\n"
        "import time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        'Path(os.environ["V03_TEST_READY"]).write_text("ready")\n'
        "while True:\n"
        "    time.sleep(1)\n",
        encoding="utf-8",
    )
    state_path = tmp_path / "supervisor.json"
    process_file = tmp_path / "guarded-processes.txt"
    discovery_functions = _function_block(
        "active_neural_trainers",
        "assert_no_competing_training_chain",
    )
    shutdown_functions = _function_block(
        "read_supervised_trainer_pid",
        "assert_single_active_chain",
    )
    cleanup_functions = _function_block("stop_caffeine", "record_launcher_exit")
    harness = (
        "set -euo pipefail\n"
        f"repository={shlex.quote(str(REPOSITORY))}\n"
        "launcher_pid=$$\n"
        "shutdown_grace_polls=3\n"
        "shutdown_escalation_polls=3\n"
        "shutdown_kill_polls=25\n"
        "shutdown_poll_interval=0.02\n"
        f"{discovery_functions}\n"
        f"{shutdown_functions}\n"
        "finish_active_after_abnormal_exit() { return 0; }\n"
        "terminalize_inactive_cohort() { return 0; }\n"
        f"{cleanup_functions}\n"
        'supervisor_pid=""\n'
        'supervisor_identity=""\n'
        'trainer_pid=""\n'
        'trainer_identity=""\n'
        f"supervisor_state={shlex.quote(str(state_path))}\n"
        'caffeine_pid=""\n'
        'active_arm=""\n'
        "cohort_created=false\n"
        "launcher_finalized=false\n"
        f"{shlex.quote(sys.executable)} "
        f"{shlex.quote(str(TRAINER_SUPERVISOR))} "
        f"--state-path {shlex.quote(str(state_path))} -- "
        f"{shlex.quote(sys.executable)} -m "
        "dungeon_apprentice.v03_action_effect_train &\n"
        "supervisor_pid=$!\n"
        "capture_owned_supervisor_identity\n"
        "for _attempt in {1..500}; do\n"
        f"  [[ -f {shlex.quote(str(ready_file))} ]] && break\n"
        "  /bin/sleep 0.01\n"
        "done\n"
        f"[[ -f {shlex.quote(str(ready_file))} ]]\n"
        'trainer_pid=$(read_supervised_trainer_pid "$supervisor_state")\n'
        "capture_owned_trainer_identity\n"
        "/bin/sleep 60 &\n"
        "caffeine_pid=$!\n"
        f'print -r -- "$supervisor_pid $trainer_pid $caffeine_pid" >'
        f"{shlex.quote(str(process_file))}\n"
        "forced_process_guard_failure() { return 37; }\n"
        'require_launcher_step "forced process-guard failure" '
        "forced_process_guard_failure\n"
    )
    result = subprocess.run(
        [zsh, "-c", harness],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(tmp_path),
            "V03_TEST_READY": str(ready_file),
        },
    )

    assert result.returncode == 37
    assert "forced process-guard failure" in result.stderr
    assert "ignored graceful stop; escalating through supervisor" in result.stderr
    assert "ignored TERM; killing its authenticated process group" in result.stderr
    supervisor_pid, trainer_pid, caffeine_pid = (
        int(value) for value in process_file.read_text(encoding="utf-8").split()
    )
    _wait_for_pid_exit(supervisor_pid)
    _wait_for_pid_exit(trainer_pid)
    _wait_for_pid_exit(caffeine_pid)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["supervisor_pid"] == supervisor_pid
    assert state["trainer_pid"] == trainer_pid
    assert state["state"] == "exited"
    assert state["forwarded_signal"] == "SIGTERM"
    assert state["exit_status"] == -signal.SIGKILL


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
    stop_caffeine = source.rindex("stop_caffeine\nrequire_launcher_step")
    final_health = source.rindex("assert_dashboard_healthy")

    assert loop < planner_done < requires_closeout < arm_finish < loop_end
    assert loop_end < stop_caffeine < process_seal < finalization < final_health
    assert (
        '"v0.3 could not seal its process closeout" \\\n'
        "  manifest_command seal-process-closeout \\\n"
        '    --root "$run_root" \\\n'
        '    --source-commit "$source_commit" \\\n'
        '    --tag-object "$tag_object" >/dev/null'
    ) in source
