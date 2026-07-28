#!/bin/zsh
set -euo pipefail

repository=${0:A:h:h}
volume="/Volumes/T7 Developer"
dungeon_root="$volume/DungeonApprentice"
run_root="$dungeon_root/v03-action-effect-stage-a-r3-20260724"
media_root="$dungeon_root/v03-action-effect-stage-a-r3-media-20260724"
qualification_report="$dungeon_root/qualifications/v0.3-action-effect-stage-a-r3-20260724/report.json"
contract="$run_root/cohort-contract.json"
parent="$dungeon_root/u1-local-replication-20260722/v02-u1-replication-seed-20260733/checkpoints/mastered-local-unlock.zip"
parent_sha256="3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104"
training_protocol="dungeon-apprentice-v0.3-action-effect-architecture"
cohort_id="v0.3-action-effect-stage-a-r3-20260724"
training_tag="action-effect-architecture-v0.3-stage-a-r3-20260724"
expected_origin="https://github.com/PeteAndrews1289/dungeon-apprentice.git"
trainer_module="dungeon_apprentice.v03_action_effect_train"
dashboard_module="dungeon_apprentice.v03_action_effect_dashboard"
qualifier_module="dungeon_apprentice.v03_action_effect_qualify"
manifest_helper="$repository/scripts/v03_action_effect_manifest.py"
trainer_supervisor="$repository/scripts/u2_trainer_supervisor.py"
dashboard_port=8791
minimum_free_gib=25
shutdown_grace_polls=150
shutdown_escalation_polls=100
shutdown_kill_polls=50
shutdown_poll_interval=0.1

if (( $# != 0 )); then
  print -u2 "the fixed v0.3 Stage-A launcher accepts no options; interruption makes both fresh twins terminal"
  exit 2
fi

source_commit=""
tag_object=""
supervisor_pid=""
trainer_pid=""
supervisor_identity=""
trainer_identity=""
supervisor_state=""
caffeine_pid=""
dashboard_pid=""
active_arm=""
signal_requested=false
launcher_finalized=false
cohort_created=false
launcher_pid=$$

cd "$repository"

clean_source_commit() {
  local commit
  commit=$(git rev-parse --verify 'HEAD^{commit}' 2>/dev/null) || return 1
  [[ -n "$commit" ]] || return 1
  [[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || return 1
  print -r -- "$commit"
}

authenticate_published_training_tag() {
  local object_type peeled remote_url remote_line
  object_type=$(git cat-file -t "refs/tags/$training_tag" 2>/dev/null) || {
    print -u2 "required v0.3 annotated tag is missing: $training_tag"
    return 1
  }
  [[ "$object_type" == tag ]] || {
    print -u2 "v0.3 source tag must be an annotated tag object"
    return 1
  }
  tag_object=$(git rev-parse --verify "refs/tags/$training_tag")
  peeled=$(git rev-parse --verify "refs/tags/$training_tag^{commit}")
  [[ "$peeled" == "$source_commit" ]] || {
    print -u2 "v0.3 tag does not peel to the clean source commit"
    return 1
  }
  remote_url=$(git remote get-url origin)
  [[ "$remote_url" == "$expected_origin" ]] || {
    print -u2 "v0.3 origin differs from its preregistered repository"
    return 1
  }
  remote_line=$(git ls-remote --tags origin "refs/tags/$training_tag")
  [[ "$remote_line" == "$tag_object"$'\t'"refs/tags/$training_tag" ]] || {
    print -u2 "v0.3 annotated tag is not published unchanged on origin"
    return 1
  }
}

assert_source_unchanged() {
  local measured measured_tag peeled
  measured=$(clean_source_commit) || {
    print -u2 "v0.3 source became dirty or lost its committed HEAD"
    return 1
  }
  [[ "$measured" == "$source_commit" ]] || {
    print -u2 "v0.3 source commit changed after authentication"
    return 1
  }
  measured_tag=$(git rev-parse --verify "refs/tags/$training_tag" 2>/dev/null) || return 1
  peeled=$(git rev-parse --verify "refs/tags/$training_tag^{commit}" 2>/dev/null) || return 1
  [[ "$measured_tag" == "$tag_object" && "$peeled" == "$source_commit" ]] || {
    print -u2 "v0.3 annotated tag identity changed"
    return 1
  }
}

assert_actual_mounted_volume() {
  "$repository/.venv/bin/python" - "$volume" <<'PY'
import os
import stat
import sys
from pathlib import Path

volume = Path(sys.argv[1])
try:
    metadata = volume.lstat()
    parent_metadata = volume.parent.stat()
except OSError as error:
    raise SystemExit(f"cannot inspect T7 mount point {volume}: {error}") from error
if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
    raise SystemExit(f"T7 mount point is not a regular directory: {volume}")
if not os.path.ismount(volume):
    raise SystemExit(f"T7 Developer is not an active mounted volume: {volume}")
if metadata.st_dev == parent_metadata.st_dev:
    raise SystemExit(f"T7 Developer does not have its own mounted device: {volume}")
PY
}

assert_free_space() {
  "$repository/.venv/bin/python" - "$volume" "$minimum_free_gib" <<'PY'
import shutil
import sys

volume, minimum_gib = sys.argv[1:]
free = shutil.disk_usage(volume).free
required = int(float(minimum_gib) * 1024**3)
if free < required:
    raise SystemExit(
        f"v0.3 Stage A requires {minimum_gib} GiB free, but only "
        f"{free / 1024**3:.2f} GiB remains"
    )
PY
}

assert_regular_ancestor_chain() {
  local artifact_path=$1
  local stop=$2
  local current=${artifact_path:h}
  while true; do
    if [[ ! -d "$current" || -L "$current" ]]; then
      print -u2 "v0.3 artifact path has a missing or symlinked ancestor: $current"
      return 1
    fi
    [[ "$current" == "$stop" ]] && return 0
    if [[ "$current" == "/" ]]; then
      print -u2 "v0.3 artifact path escapes its fixed storage root"
      return 1
    fi
    current=${current:h}
  done
}

assert_frozen_parent() {
  if [[ ! -f "$parent" || -L "$parent" ]]; then
    print -u2 "confirmed U1 parent is missing or unsafe: $parent"
    return 1
  fi
  assert_regular_ancestor_chain "$parent" "$dungeon_root"
  local checksum_line
  checksum_line=$(/usr/bin/shasum -a 256 "$parent")
  [[ "${checksum_line%% *}" == "$parent_sha256" ]] || {
    print -u2 "confirmed U1 parent checksum changed"
    return 1
  }
}

assert_v03_qualification() {
  "$repository/.venv/bin/python" - \
    "$repository" "$qualification_report" "$source_commit" "$tag_object" <<'PY'
import sys
from pathlib import Path

from dungeon_apprentice.v03_action_effect_qualify import (
    CANONICAL_COHORT_ROOT,
    CANONICAL_MEDIA_ROOT,
    CANONICAL_REPORT,
    QUALIFIED_TAG,
    verify_action_effect_qualification,
)

repository, report, source_commit, tag_object = sys.argv[1:]
report_path = Path(report)
if (
    report_path != CANONICAL_REPORT
    or Path("/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r3-20260724")
    != CANONICAL_COHORT_ROOT
    or Path("/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r3-media-20260724")
    != CANONICAL_MEDIA_ROOT
    or QUALIFIED_TAG != "action-effect-architecture-v0.3-stage-a-r3-20260724"
):
    raise SystemExit("v0.3 canonical identities differ from the launcher")
evidence = verify_action_effect_qualification(
    report_path,
    expected_source_commit=source_commit,
    expected_tag_object=tag_object,
    repository=Path(repository),
)
public = evidence.public_dict()
if (
    public.get("verdict") != "qualified"
    or public.get("source_commit") != source_commit
    or public.get("tag") != QUALIFIED_TAG
    or public.get("tag_object") != tag_object
    or public.get("report") != str(CANONICAL_REPORT)
    or not public.get("report_sha256")
    or not public.get("architecture_contract_sha256")
    or not public.get("smoke_evidence_sha256")
):
    raise SystemExit("v0.3 qualification identity differs from this launch")
PY
}

active_neural_trainers() {
  # Classify the leading argv of each process instead of searching its whole
  # command line.  The fixed supervisor carries the complete child command
  # after `--`, so pgrep -f incorrectly classifies that supervisor as a second
  # trainer even though its executable argv launches u2_trainer_supervisor.py.
  /bin/ps -axo pid=,command= | /usr/bin/awk '
    function basename(path, pieces, count) {
      count = split(path, pieces, "/")
      return pieces[count]
    }
    function is_python(name, lowered) {
      lowered = tolower(name)
      return lowered ~ /^python([0-9]+([.][0-9]+)*)?$/
    }
    function is_trainer_module(name) {
      return name == "dungeon_apprentice.train" \
        || name == "dungeon_apprentice.v02_sentinel" \
        || name == "dungeon_apprentice.v02_u1" \
        || name == "dungeon_apprentice.v02_u2" \
        || name == "dungeon_apprentice.v02_u2r" \
        || name == "dungeon_apprentice.v02_u2s" \
        || name == "dungeon_apprentice.v03_action_effect_train"
    }
    {
      pid = $1
      executable = basename($2)
      first_argument = basename($3)
      if (executable == "dungeon-train") {
        print pid
      } else if (is_python(executable)) {
        if (first_argument == "dungeon-train") {
          print pid
        } else if ($3 == "-m" && is_trainer_module($4)) {
          print pid
        }
      }
    }
  '
}

active_trainer_supervisors() {
  # As above, only the process whose leading argv executes the supervisor
  # script is a supervisor.  A shell, test runner, or child command that merely
  # mentions the script later in argv must not be counted.
  /bin/ps -axo pid=,command= | /usr/bin/awk '
    function basename(path, pieces, count) {
      count = split(path, pieces, "/")
      return pieces[count]
    }
    function is_python(name, lowered) {
      lowered = tolower(name)
      return lowered ~ /^python([0-9]+([.][0-9]+)*)?$/
    }
    {
      pid = $1
      executable = basename($2)
      first_argument = basename($3)
      if (executable == "u2_trainer_supervisor.py" \
          || (is_python(executable) \
              && first_argument == "u2_trainer_supervisor.py")) {
        print pid
      }
    }
  '
}

active_caffeinates() {
  /usr/bin/pgrep -x caffeinate || true
}

assert_no_competing_training_chain() {
  local trainers supervisors caffeinates
  trainers=$(active_neural_trainers)
  supervisors=$(active_trainer_supervisors)
  caffeinates=$(active_caffeinates)
  if [[ -n "$trainers" || -n "$supervisors" || -n "$caffeinates" ]]; then
    print -u2 "refusing a competing neural trainer/supervisor/caffeinate chain"
    [[ -z "$trainers" ]] || print -u2 "trainer PIDs: ${trainers//$'\n'/, }"
    [[ -z "$supervisors" ]] || print -u2 "supervisor PIDs: ${supervisors//$'\n'/, }"
    [[ -z "$caffeinates" ]] || print -u2 "caffeinate PIDs: ${caffeinates//$'\n'/, }"
    return 1
  fi
}

dashboard_listener_pid() {
  /usr/sbin/lsof -nP -t -iTCP:"$dashboard_port" -sTCP:LISTEN 2>/dev/null \
    | /usr/bin/head -n 1 || true
}

assert_dashboard_healthy() {
  "$repository/.venv/bin/python" - \
    "$dashboard_port" "$source_commit" "$tag_object" <<'PY'
import json
import sys
import urllib.request

port, source, tag_object = sys.argv[1:]
with urllib.request.urlopen(
    f"http://127.0.0.1:{port}/api/v03.json",
    timeout=10,
) as response:
    payload = json.load(response)
if (
    payload.get("cohort_id")
    != "v0.3-action-effect-stage-a-r3-20260724"
    or payload.get("protocol")
    != "dungeon-apprentice-v0.3-action-effect-architecture"
    or payload.get("source_commit") != source
    or payload.get("tag")
    != "action-effect-architecture-v0.3-stage-a-r3-20260724"
    or payload.get("tag_object") != tag_object
    or payload.get("checkpoint_promotable") is not False
    or payload.get("u3_authorized") is not False
):
    raise SystemExit("dashboard identity differs from this v0.3 cohort")
PY
}

start_dashboard() {
  local listener
  listener=$(dashboard_listener_pid)
  if [[ -n "$listener" ]]; then
    print -u2 "v0.3 dashboard port $dashboard_port is already in use"
    return 1
  fi
  local dashboard_log="$run_root/dashboard.log"
  dashboard_pid=$(
    "$repository/.venv/bin/python" - \
      "$repository/.venv/bin/python" "$dashboard_module" "$run_root" \
      "$dashboard_port" "$dashboard_log" <<'PY'
import subprocess
import sys

python, module, root, port, log_path = sys.argv[1:]
with open(log_path, "ab", buffering=0) as log:
    process = subprocess.Popen(
        [
            python,
            "-m",
            module,
            "--run-root",
            root,
            "--host",
            "127.0.0.1",
            "--port",
            port,
        ],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        close_fds=True,
        start_new_session=True,
    )
print(process.pid)
PY
  )
  for _attempt in {1..300}; do
    if ! kill -0 "$dashboard_pid" 2>/dev/null; then
      break
    fi
    if assert_dashboard_healthy 2>/dev/null; then
      print -r -- "$dashboard_pid" >"$run_root/dashboard.pid"
      print "v0.3 Stage-A dashboard: http://127.0.0.1:$dashboard_port/"
      return 0
    fi
    /bin/sleep 0.1
  done
  kill "$dashboard_pid" 2>/dev/null || true
  print -u2 "v0.3 dashboard did not become healthy on /api/v03.json"
  return 1
}

manifest_command() {
  "$repository/.venv/bin/python" "$manifest_helper" "$@"
}

read_status_phase() {
  local status_path=$1
  "$repository/.venv/bin/python" - "$status_path" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    phase = json.load(stream).get("phase")
if phase not in {"completed", "interrupted", "crashed"}:
    raise SystemExit(f"unexpected v0.3 arm phase: {phase!r}")
print(phase)
PY
}

read_supervised_trainer_pid() {
  local state_path=$1
  "$repository/.venv/bin/python" - "$state_path" "$supervisor_pid" <<'PY'
import json
import sys

path, expected_supervisor = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    state = json.load(stream)
if (
    state.get("supervisor_pid") != int(expected_supervisor)
    or state.get("state") not in {"running", "stop_requested", "exited"}
    or not isinstance(state.get("trainer_pid"), int)
):
    raise SystemExit("v0.3 supervisor identity is invalid")
print(state["trainer_pid"])
PY
}

process_identity() {
  local pid=$1
  local identity
  identity=$(
    /bin/ps -ww -o ppid=,lstart=,command= -p "$pid" 2>/dev/null \
      | /usr/bin/awk '{$1=$1; print}'
  ) || return 1
  [[ -n "$identity" ]] || return 1
  print -r -- "$identity"
}

process_field() {
  local pid=$1
  local field=$2
  local value
  value=$(
    /bin/ps -o "${field}=" -p "$pid" 2>/dev/null \
      | /usr/bin/awk '{$1=$1; print}'
  ) || return 1
  [[ -n "$value" ]] || return 1
  print -r -- "$value"
}

capture_owned_supervisor_identity() {
  local measured previous parent supervisors recorded_trainer
  previous=""
  for _attempt in {1..1000}; do
    measured=$(process_identity "$supervisor_pid" 2>/dev/null || true)
    parent=$(process_field "$supervisor_pid" ppid 2>/dev/null || true)
    supervisors=$(active_trainer_supervisors)
    recorded_trainer=""
    if [[ -n "$supervisor_state" && -f "$supervisor_state" \
      && ! -L "$supervisor_state" ]]; then
      recorded_trainer=$(
        read_supervised_trainer_pid "$supervisor_state" 2>/dev/null || true
      )
    fi
    # Python on macOS can replace its displayed executable argv during early
    # framework startup. Require authenticated supervisor evidence and two
    # identical consecutive measurements before freezing the PID identity.
    if [[ -n "$measured" && "$measured" == "$previous" \
      && "$parent" == "$launcher_pid" \
      && "$supervisors" == "$supervisor_pid" \
      && -n "$recorded_trainer" ]]; then
      supervisor_identity=$measured
      return 0
    fi
    previous=$measured
    /bin/sleep 0.01
  done
  print -u2 "v0.3 could not capture its owned supervisor process identity"
  return 1
}

owned_supervisor_is_live() {
  local measured parent state
  [[ -n "$supervisor_pid" && -n "$supervisor_identity" ]] || return 1
  measured=$(process_identity "$supervisor_pid" 2>/dev/null) || return 1
  parent=$(process_field "$supervisor_pid" ppid 2>/dev/null) || return 1
  state=$(process_field "$supervisor_pid" state 2>/dev/null) || return 1
  [[ "$measured" == "$supervisor_identity" \
    && "$parent" == "$launcher_pid" \
    && "$state" != Z* ]]
}

capture_owned_trainer_identity() {
  local candidate measured parent group trainers
  [[ -n "$supervisor_state" && -f "$supervisor_state" \
    && ! -L "$supervisor_state" ]] || return 1
  candidate=$(read_supervised_trainer_pid "$supervisor_state") || return 1
  measured=$(process_identity "$candidate" 2>/dev/null) || return 1
  parent=$(process_field "$candidate" ppid 2>/dev/null) || return 1
  group=$(process_field "$candidate" pgid 2>/dev/null) || return 1
  trainers=$(active_neural_trainers)
  [[ "$parent" == "$supervisor_pid" \
    && "$group" == "$candidate" \
    && "$trainers" == "$candidate" ]] || return 1
  if [[ -n "$trainer_pid" && "$trainer_pid" != "$candidate" ]]; then
    return 1
  fi
  if [[ -n "$trainer_identity" && "$trainer_identity" != "$measured" ]]; then
    return 1
  fi
  trainer_pid=$candidate
  trainer_identity=$measured
}

reap_supervisor() {
  wait "$supervisor_pid" 2>/dev/null || true
  supervisor_pid=""
  supervisor_identity=""
  trainer_pid=""
  trainer_identity=""
}

wait_for_supervisor_exit() {
  local polls=$1
  local measured state
  for (( _poll = 0; _poll < polls; _poll++ )); do
    measured=$(process_identity "$supervisor_pid" 2>/dev/null || true)
    if [[ -z "$measured" ]]; then
      reap_supervisor
      return 0
    fi
    if [[ "$measured" != "$supervisor_identity" ]]; then
      print -u2 \
        "v0.3 refused to wait on a supervisor PID with a changed identity"
      return 2
    fi
    state=$(process_field "$supervisor_pid" state 2>/dev/null || true)
    if [[ "$state" == Z* ]]; then
      reap_supervisor
      return 0
    fi
    /bin/sleep "$shutdown_poll_interval"
  done
  return 1
}

signal_owned_supervisor() {
  local signal_name=$1
  if ! owned_supervisor_is_live; then
    print -u2 \
      "v0.3 refused to signal a supervisor without its captured identity"
    return 1
  fi
  kill -"$signal_name" "$supervisor_pid"
}

kill_owned_trainer_group() {
  local measured parent group
  capture_owned_trainer_identity || {
    print -u2 \
      "v0.3 refused to kill a trainer group without authenticated supervisor evidence"
    return 1
  }
  measured=$(process_identity "$trainer_pid" 2>/dev/null) || return 1
  parent=$(process_field "$trainer_pid" ppid 2>/dev/null) || return 1
  group=$(process_field "$trainer_pid" pgid 2>/dev/null) || return 1
  [[ "$measured" == "$trainer_identity" \
    && "$parent" == "$supervisor_pid" \
    && "$group" == "$trainer_pid" ]] || {
    print -u2 "v0.3 refused to kill a reused or unowned trainer process group"
    return 1
  }
  /bin/kill -KILL -- "-$trainer_pid"
}

shutdown_supervised_chain() {
  [[ -n "$supervisor_pid" ]] || return 0
  if ! owned_supervisor_is_live; then
    if wait_for_supervisor_exit 1; then
      return 0
    fi
    print -u2 "v0.3 cannot authenticate its supervisor for shutdown"
    return 1
  fi

  # The supervisor translates the first TERM into trainer SIGINT, preserving
  # the trainer's normal interrupted-status and checkpoint evidence when it
  # cooperates.
  signal_owned_supervisor TERM || return 1
  if wait_for_supervisor_exit "$shutdown_grace_polls"; then
    return 0
  fi

  # A second TERM is intentionally sent to the same authenticated supervisor;
  # its fixed handler escalates the still-running trainer from SIGINT to
  # SIGTERM while retaining the atomic supervisor state file.
  print -u2 "v0.3 trainer ignored graceful stop; escalating through supervisor"
  signal_owned_supervisor TERM || return 1
  if wait_for_supervisor_exit "$shutdown_escalation_polls"; then
    return 0
  fi

  # Only the trainer-owned session/process group is killed, and only while its
  # PID, parent, PGID, argv identity, and supervisor state all still agree.
  print -u2 "v0.3 trainer ignored TERM; killing its authenticated process group"
  kill_owned_trainer_group || return 1
  if wait_for_supervisor_exit "$shutdown_kill_polls"; then
    return 0
  fi

  # The child group is gone, so a supervisor that still failed to publish and
  # exit is finally reaped without ever signalling an unverified/reused PID.
  print -u2 "v0.3 supervisor did not reap the killed trainer; forcing supervisor exit"
  signal_owned_supervisor KILL || return 1
  wait_for_supervisor_exit "$shutdown_kill_polls"
}

assert_single_active_chain() {
  local supervisor_state=$1
  local trainer_parent trainer_group trainers supervisors caffeinates
  kill -0 "$supervisor_pid" 2>/dev/null || {
    print -u2 "v0.3 trainer supervisor disappeared"
    return 1
  }
  kill -0 "$caffeine_pid" 2>/dev/null || {
    print -u2 "v0.3 caffeinate process disappeared"
    return 1
  }
  trainer_pid=$(read_supervised_trainer_pid "$supervisor_state")
  kill -0 "$trainer_pid" 2>/dev/null || {
    print -u2 "v0.3 trainer process disappeared"
    return 1
  }
  trainer_parent=$(/bin/ps -o ppid= -p "$trainer_pid" | tr -d ' ')
  [[ "$trainer_parent" == "$supervisor_pid" ]] || {
    print -u2 "v0.3 trainer is not owned by its fixed supervisor"
    return 1
  }
  trainer_group=$(process_field "$trainer_pid" pgid)
  [[ "$trainer_group" == "$trainer_pid" ]] || {
    print -u2 "v0.3 trainer does not own its fixed process group"
    return 1
  }
  trainers=$(active_neural_trainers)
  supervisors=$(active_trainer_supervisors)
  caffeinates=$(active_caffeinates)
  [[ "$trainers" == "$trainer_pid" ]] || {
    print -u2 "v0.3 does not have exactly one neural trainer"
    return 1
  }
  [[ "$supervisors" == "$supervisor_pid" ]] || {
    print -u2 "v0.3 does not have exactly one trainer supervisor"
    return 1
  }
  [[ "$caffeinates" == "$caffeine_pid" ]] || {
    print -u2 "v0.3 does not have exactly one caffeinate process"
    return 1
  }
  trainer_identity=$(process_identity "$trainer_pid") || {
    print -u2 "v0.3 could not capture its trainer process identity"
    return 1
  }
}

finish_active_after_abnormal_exit() {
  local trainer_exit_code=${1:-1}
  local outcome=crashed
  if [[ -z "$active_arm" ]]; then
    return 0
  fi
  local status_path="$run_root/$active_arm/status.json"
  if [[ "$signal_requested" == true && -f "$status_path" && ! -L "$status_path" ]]; then
    local phase
    phase=$(read_status_phase "$status_path" 2>/dev/null || true)
    [[ "$phase" == interrupted ]] && outcome=interrupted
  fi
  manifest_command finish-arm \
    --root "$run_root" \
    --source-commit "$source_commit" \
    --tag-object "$tag_object" \
    --arm "$active_arm" \
    --outcome "$outcome" \
    --trainer-exit-code "$trainer_exit_code" >/dev/null 2>&1 || true
  active_arm=""
}

terminalize_inactive_cohort() {
  local exit_code=${1:-1}
  if [[ "$cohort_created" != true || "$launcher_finalized" == true ]]; then
    return 0
  fi
  (( exit_code == 0 )) && exit_code=1
  local plan_json disposition arm
  plan_json=$(
    manifest_command next \
      --root "$run_root" \
      --source-commit "$source_commit" \
      --tag-object "$tag_object" 2>/dev/null
  ) || return 0
  [[ -n "$plan_json" ]] || return 0
  disposition=$(
    "$repository/.venv/bin/python" - "$plan_json" <<'PY'
import json
import sys

try:
    plan = json.loads(sys.argv[1])
except (TypeError, ValueError):
    raise SystemExit(2) from None
if not isinstance(plan, dict):
    raise SystemExit(2)
if plan.get("done") is True and plan.get("closeout_required") is True:
    print("closeout")
elif plan.get("done") is False and plan.get("arm") in {"sham", "action-effect"}:
    print(f"abort:{plan['arm']}")
else:
    print("none")
PY
  ) || return 0
  if [[ "$disposition" == closeout ]]; then
    # Both arms are complete, but an abnormal launcher exit cannot fabricate
    # the required post-caffeinate process proof. Leave the root visibly
    # awaiting closeout; finalize must remain fail-closed.
    return 0
  fi
  if [[ "$disposition" == abort:* ]]; then
    arm=${disposition#abort:}
    # Claim the one legal pending arm and immediately record a crash. This
    # uses the manifest's normal state machine to make an otherwise-idle
    # interrupted cohort durably operationally_incomplete.
    if manifest_command start-arm \
      --root "$run_root" \
      --source-commit "$source_commit" \
      --tag-object "$tag_object" \
      --arm "$arm" >/dev/null 2>&1; then
      manifest_command finish-arm \
        --root "$run_root" \
        --source-commit "$source_commit" \
        --tag-object "$tag_object" \
        --arm "$arm" \
        --outcome crashed \
        --trainer-exit-code "$exit_code" >/dev/null 2>&1 || true
    fi
  fi
}

stop_caffeine() {
  if [[ -n "$caffeine_pid" ]]; then
    kill "$caffeine_pid" 2>/dev/null || true
    wait "$caffeine_pid" 2>/dev/null || true
    caffeine_pid=""
  fi
}

cleanup_launcher() {
  local exit_code=${1:-1}
  trap - EXIT INT TERM
  shutdown_supervised_chain || true
  finish_active_after_abnormal_exit "$exit_code" || true
  terminalize_inactive_cohort "$exit_code" || true
  stop_caffeine || true
}

abort_launcher() {
  local exit_code=${1:-1}
  shift || true
  (( exit_code == 0 )) && exit_code=1
  if (( $# > 0 )); then
    print -u2 "$*"
  fi
  cleanup_launcher "$exit_code"
  exit "$exit_code"
}

require_launcher_step() {
  local description=$1
  shift
  if "$@"; then
    return 0
  else
    local exit_code=$?
    (( exit_code == 0 )) && exit_code=1
    abort_launcher "$exit_code" "$description"
  fi
}

record_launcher_exit() {
  local exit_code=$?
  cleanup_launcher "$exit_code"
  exit "$exit_code"
}

request_launcher_stop() {
  signal_requested=true
  abort_launcher 130 \
    "v0.3 launcher stop requested; closing the supervised chain"
}

trap record_launcher_exit EXIT
trap request_launcher_stop INT TERM

require_launcher_step \
  "v0.3 could not authenticate the mounted storage volume" \
  assert_actual_mounted_volume
if [[ ! -d "$dungeon_root" || -L "$dungeon_root" ]]; then
  abort_launcher 1 \
    "Dungeon Apprentice storage root is missing or unsafe: $dungeon_root"
fi
require_launcher_step \
  "v0.3 storage root failed its regular-ancestor check" \
  assert_regular_ancestor_chain "$dungeon_root/placeholder" "$volume"
if [[ ! -x "$repository/.venv/bin/python" ]]; then
  abort_launcher 1 "project environment is missing its Python runtime"
fi
for helper in "$manifest_helper" "$trainer_supervisor"; do
  if [[ ! -f "$helper" || -L "$helper" ]]; then
    abort_launcher 1 "v0.3 launcher helper is missing or unsafe: $helper"
  fi
done
if ! "$repository/.venv/bin/python" -c \
  "import $trainer_module, $dashboard_module, $qualifier_module" >/dev/null 2>&1; then
  abort_launcher 1 "project environment cannot import the fixed v0.3 runtime"
fi

require_launcher_step \
  "v0.3 could not authenticate the frozen parent checkpoint" \
  assert_frozen_parent
source_commit=$(clean_source_commit) || {
  abort_launcher 1 \
    "refusing to launch v0.3 Stage A from dirty or uncommitted source"
}
require_launcher_step \
  "v0.3 could not authenticate its published training tag" \
  authenticate_published_training_tag
require_launcher_step \
  "v0.3 qualification authentication failed" \
  assert_v03_qualification
if [[ -e "$run_root" || -L "$run_root" || -e "$media_root" || -L "$media_root" ]]; then
  abort_launcher 1 \
    "refusing to reuse the canonical fresh v0.3 Stage-A roots"
fi
require_launcher_step "v0.3 free-space preflight failed" assert_free_space
require_launcher_step \
  "v0.3 found a competing training chain during preflight" \
  assert_no_competing_training_chain
[[ -z "$(dashboard_listener_pid)" ]] || {
  abort_launcher 1 "v0.3 dashboard port $dashboard_port is already in use"
}

require_launcher_step \
  "v0.3 could not create its canonical cohort root" \
  /bin/mkdir -m 0755 "$run_root"
require_launcher_step \
  "v0.3 could not create its canonical media root" \
  /bin/mkdir -m 0755 "$media_root"
require_launcher_step \
  "v0.3 cohort root failed its regular-ancestor check" \
  assert_regular_ancestor_chain "$run_root/placeholder" "$dungeon_root"
require_launcher_step \
  "v0.3 media root failed its regular-ancestor check" \
  assert_regular_ancestor_chain "$media_root/placeholder" "$dungeon_root"
require_launcher_step \
  "v0.3 could not create its cohort manifest" \
  manifest_command create \
    --root "$run_root" \
    --media-root "$media_root" \
    --qualification-report "$qualification_report" \
    --source-commit "$source_commit" \
    --tag-object "$tag_object" >/dev/null
cohort_created=true

require_launcher_step \
  "v0.3 source changed after cohort creation" \
  assert_source_unchanged
require_launcher_step "v0.3 free-space check failed after cohort creation" \
  assert_free_space
require_launcher_step \
  "v0.3 found a competing chain after cohort creation" \
  assert_no_competing_training_chain
require_launcher_step "v0.3 dashboard startup failed" start_dashboard

/usr/bin/caffeinate -ims >/dev/null 2>&1 &
caffeine_pid=$!
require_launcher_step \
  "v0.3 caffeinate process did not remain alive" \
  kill -0 "$caffeine_pid"

while true; do
  require_launcher_step \
    "v0.3 source changed before the next arm" \
    assert_source_unchanged
  require_launcher_step \
    "v0.3 free-space check failed before the next arm" \
    assert_free_space
  require_launcher_step \
    "v0.3 dashboard health failed before the next arm" \
    assert_dashboard_healthy
  [[ -z "$(active_neural_trainers)" ]] || {
    abort_launcher 1 "a neural trainer exists before the next v0.3 arm"
  }
  [[ -z "$(active_trainer_supervisors)" ]] || {
    abort_launcher 1 \
      "a trainer supervisor exists before the next v0.3 arm"
  }
  [[ "$(active_caffeinates)" == "$caffeine_pid" ]] || {
    abort_launcher 1 \
      "the v0.3 launcher lost its sole caffeinate identity"
  }
  plan_json=$(
    manifest_command next \
      --root "$run_root" \
      --source-commit "$source_commit" \
      --tag-object "$tag_object"
  ) || abort_launcher 1 "v0.3 next-arm planning failed"
  [[ -n "$plan_json" ]] || {
    abort_launcher 1 "v0.3 next-arm planner returned no JSON"
  }
  plan_fields=("${(@f)$(
    "$repository/.venv/bin/python" - "$plan_json" <<'PY'
import json
import sys

try:
    plan = json.loads(sys.argv[1])
except (TypeError, ValueError):
    raise SystemExit(2) from None
if not isinstance(plan, dict):
    raise SystemExit(2)
if plan.get("done") is True:
    print("done")
    print("true" if plan.get("closeout_required") is True else "false")
else:
    print("run")
    for key in ("arm", "attempt", "run_dir", "media_dir"):
        print(plan[key])
PY
  )}") || abort_launcher 1 "v0.3 next-arm plan parsing failed"
  if [[ "${plan_fields[1]}" == done ]]; then
    if (( ${#plan_fields} != 2 )) || [[ "${plan_fields[2]}" != true ]]; then
      abort_launcher 1 \
        "v0.3 planner reached done without a required closeout"
    fi
    break
  fi
  if (( ${#plan_fields} != 5 )) || [[ "${plan_fields[1]}" != run ]]; then
    abort_launcher 1 \
      "v0.3 next-arm planner returned an incomplete exact plan"
  fi
  active_arm=${plan_fields[2]}
  attempt=${plan_fields[3]}
  run_directory=${plan_fields[4]}
  media_directory=${plan_fields[5]}
  if [[ "$active_arm" != sham && "$active_arm" != action-effect ]] \
    || [[ "$attempt" != 0 ]] \
    || [[ "$run_directory" != "$run_root/$active_arm" ]] \
    || [[ "$media_directory" != "$media_root/$active_arm" ]]; then
    abort_launcher 1 "v0.3 planner changed its fixed arm identity"
  fi

  require_launcher_step \
    "v0.3 could not claim the next arm in its manifest" \
    manifest_command start-arm \
      --root "$run_root" \
      --source-commit "$source_commit" \
      --tag-object "$tag_object" \
      --arm "$active_arm" >/dev/null
  if [[ -e "$run_directory" || -L "$run_directory" ]] \
    || [[ -e "$media_directory" || -L "$media_directory" ]]; then
    abort_launcher 1 "refusing to reuse a fresh v0.3 arm target"
  fi
  require_launcher_step \
    "v0.3 could not create its fresh arm run directory" \
    /bin/mkdir -m 0755 "$run_directory"
  require_launcher_step \
    "v0.3 could not create its fresh arm media directory" \
    /bin/mkdir -m 0755 "$media_directory"
  require_launcher_step \
    "v0.3 arm run directory failed its regular-ancestor check" \
    assert_regular_ancestor_chain "$run_directory/placeholder" "$dungeon_root"
  require_launcher_step \
    "v0.3 arm media directory failed its regular-ancestor check" \
    assert_regular_ancestor_chain "$media_directory/placeholder" "$dungeon_root"

  supervisor_state="$run_root/launcher-$active_arm-attempt-$attempt.json"
  if [[ -e "$supervisor_state" || -L "$supervisor_state" ]]; then
    abort_launcher 1 \
      "refusing to replace v0.3 supervisor evidence: $supervisor_state"
  fi
  print "Starting v0.3 Stage-A arm $active_arm (fresh matched twin)."
  "$repository/.venv/bin/python" "$trainer_supervisor" \
    --state-path "$supervisor_state" \
    -- \
      "$repository/.venv/bin/python" -m "$trainer_module" train-arm \
      --arm "$active_arm" \
      --qualification-report "$qualification_report" \
      --run-dir "$run_directory" \
      --media-dir "$media_directory" \
      --cohort-contract "$contract" \
      --device cpu &
  supervisor_pid=$!
  supervisor_identity=""
  trainer_pid=""
  trainer_identity=""
  require_launcher_step \
    "v0.3 could not authenticate its newly launched supervisor" \
    capture_owned_supervisor_identity

  status_path="$run_directory/status.json"
  status_ready=false
  for _attempt in {1..3000}; do
    if [[ -f "$status_path" && ! -L "$status_path" ]]; then
      status_ready=true
      break
    fi
    if ! kill -0 "$supervisor_pid" 2>/dev/null; then
      break
    fi
    /bin/sleep 0.1
  done
  if [[ "$status_ready" == true ]] && kill -0 "$supervisor_pid" 2>/dev/null; then
    require_launcher_step \
      "v0.3 active trainer chain identity check failed" \
      assert_single_active_chain "$supervisor_state"
  fi
  set +e
  wait "$supervisor_pid"
  trainer_status=$?
  supervisor_pid=""
  supervisor_identity=""
  trainer_pid=""
  trainer_identity=""
  set -e
  if [[ "$status_ready" != true || ! -f "$status_path" || -L "$status_path" ]]; then
    require_launcher_step \
      "v0.3 could not record its pre-status trainer crash" \
      manifest_command finish-arm \
        --root "$run_root" --source-commit "$source_commit" \
        --tag-object "$tag_object" --arm "$active_arm" --outcome crashed \
        --trainer-exit-code 1 >/dev/null
    active_arm=""
    abort_launcher 1 \
      "v0.3 arm stopped before publishing safe status evidence"
  fi
  terminal_phase=$(read_status_phase "$status_path") \
    || abort_launcher 1 "v0.3 terminal arm status authentication failed"
  require_launcher_step \
    "v0.3 source changed after an arm exited" \
    assert_source_unchanged
  require_launcher_step \
    "v0.3 free-space check failed after an arm exited" \
    assert_free_space
  require_launcher_step \
    "v0.3 dashboard health failed after an arm exited" \
    assert_dashboard_healthy
  if [[ "$terminal_phase" == interrupted ]]; then
    require_launcher_step \
      "v0.3 could not record its interrupted arm outcome" \
      manifest_command finish-arm \
        --root "$run_root" --source-commit "$source_commit" \
        --tag-object "$tag_object" --arm "$active_arm" --outcome interrupted \
        --trainer-exit-code "$trainer_status" >/dev/null
    active_arm=""
    abort_launcher 130 \
      "v0.3 arm was interrupted; the complete matched root is terminal operationally_incomplete."
  fi
  if [[ "$terminal_phase" == crashed || "$trainer_status" -ne 0 ]]; then
    require_launcher_step \
      "v0.3 could not record its crashed arm outcome" \
      manifest_command finish-arm \
        --root "$run_root" --source-commit "$source_commit" \
        --tag-object "$tag_object" --arm "$active_arm" --outcome crashed \
        --trainer-exit-code "$trainer_status" >/dev/null
    active_arm=""
    abort_launcher 1 \
      "v0.3 arm failed; this root can never be resumed or reused."
  fi
  require_launcher_step \
    "v0.3 could not record its completed arm outcome" \
    manifest_command finish-arm \
      --root "$run_root" \
      --source-commit "$source_commit" \
      --tag-object "$tag_object" \
      --arm "$active_arm" \
      --outcome completed \
      --trainer-exit-code "$trainer_status" >/dev/null
  print "Completed v0.3 arm $active_arm at exactly 1,048,576 child actions."
  active_arm=""
done

require_launcher_step \
  "v0.3 source changed before cohort closeout" \
  assert_source_unchanged
require_launcher_step \
  "v0.3 free-space check failed before cohort closeout" \
  assert_free_space
[[ -z "$(active_neural_trainers)" ]] || {
  abort_launcher 1 "a neural trainer remained after both v0.3 arms"
}
[[ -z "$(active_trainer_supervisors)" ]] || {
  abort_launcher 1 "a trainer supervisor remained after both v0.3 arms"
}
require_launcher_step \
  "v0.3 dashboard health failed before process closeout" \
  assert_dashboard_healthy
stop_caffeine
require_launcher_step \
  "v0.3 could not seal its process closeout" \
  manifest_command seal-process-closeout \
    --root "$run_root" \
    --source-commit "$source_commit" \
    --tag-object "$tag_object" >/dev/null
require_launcher_step \
  "v0.3 could not finalize its matched cohort" \
  manifest_command finalize \
    --root "$run_root" \
    --source-commit "$source_commit" \
    --tag-object "$tag_object" >/dev/null
require_launcher_step \
  "v0.3 finalized dashboard health failed" \
  assert_dashboard_healthy
launcher_finalized=true
trap - EXIT INT TERM
print "v0.3 Stage A completed and sealed its matched architecture decision."
print "Read-only dashboard remains available: http://127.0.0.1:$dashboard_port/"
