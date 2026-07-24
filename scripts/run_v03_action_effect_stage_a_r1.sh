#!/bin/zsh
set -euo pipefail

repository=${0:A:h:h}
volume="/Volumes/T7 Developer"
dungeon_root="$volume/DungeonApprentice"
run_root="$dungeon_root/v03-action-effect-stage-a-r1-20260724"
media_root="$dungeon_root/v03-action-effect-stage-a-r1-media-20260724"
qualification_report="$dungeon_root/qualifications/v0.3-action-effect-stage-a-r1-20260724/report.json"
contract="$run_root/cohort-contract.json"
parent="$dungeon_root/u1-local-replication-20260722/v02-u1-replication-seed-20260733/checkpoints/mastered-local-unlock.zip"
parent_sha256="3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104"
training_protocol="dungeon-apprentice-v0.3-action-effect-architecture"
cohort_id="v0.3-action-effect-stage-a-r1-20260724"
training_tag="action-effect-architecture-v0.3-stage-a-r1-20260724"
expected_origin="https://github.com/PeteAndrews1289/dungeon-apprentice.git"
trainer_module="dungeon_apprentice.v03_action_effect_train"
dashboard_module="dungeon_apprentice.v03_action_effect_dashboard"
qualifier_module="dungeon_apprentice.v03_action_effect_qualify"
manifest_helper="$repository/scripts/v03_action_effect_manifest.py"
trainer_supervisor="$repository/scripts/u2_trainer_supervisor.py"
dashboard_port=8789
minimum_free_gib=25

if (( $# != 0 )); then
  print -u2 "the fixed v0.3 Stage-A launcher accepts no options; interruption makes both fresh twins terminal"
  exit 2
fi

source_commit=""
tag_object=""
supervisor_pid=""
trainer_pid=""
caffeine_pid=""
dashboard_pid=""
active_arm=""
signal_requested=false
launcher_finalized=false
cohort_created=false

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
    or Path("/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r1-20260724")
    != CANONICAL_COHORT_ROOT
    or Path("/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r1-media-20260724")
    != CANONICAL_MEDIA_ROOT
    or QUALIFIED_TAG != "action-effect-architecture-v0.3-stage-a-r1-20260724"
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
  /usr/bin/pgrep -f \
    '[d]ungeon-train|dungeon_apprentice\.(train|v02_sentinel|v02_u1|v02_u2|v02_u2r|v02_u2s|v03_action_effect_train)([[:space:]]|$)' \
    || true
}

active_trainer_supervisors() {
  /usr/bin/pgrep -f '[u]2_trainer_supervisor.py' || true
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
    != "v0.3-action-effect-stage-a-r1-20260724"
    or payload.get("protocol")
    != "dungeon-apprentice-v0.3-action-effect-architecture"
    or payload.get("source_commit") != source
    or payload.get("tag")
    != "action-effect-architecture-v0.3-stage-a-r1-20260724"
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

assert_single_active_chain() {
  local supervisor_state=$1
  local trainer_parent trainers supervisors caffeinates
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
  disposition=$(
    "$repository/.venv/bin/python" - "$plan_json" <<'PY'
import json
import sys

plan = json.loads(sys.argv[1])
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

record_launcher_exit() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "$supervisor_pid" ]]; then
    kill -TERM "$supervisor_pid" 2>/dev/null || true
    wait "$supervisor_pid" 2>/dev/null || true
    supervisor_pid=""
  fi
  finish_active_after_abnormal_exit "$exit_code"
  terminalize_inactive_cohort "$exit_code"
  stop_caffeine
  exit "$exit_code"
}

request_launcher_stop() {
  signal_requested=true
  if [[ -n "$supervisor_pid" ]]; then
    kill -TERM "$supervisor_pid" 2>/dev/null || true
  else
    exit 130
  fi
}

trap record_launcher_exit EXIT
trap request_launcher_stop INT TERM

assert_actual_mounted_volume
if [[ ! -d "$dungeon_root" || -L "$dungeon_root" ]]; then
  print -u2 "Dungeon Apprentice storage root is missing or unsafe: $dungeon_root"
  exit 1
fi
assert_regular_ancestor_chain "$dungeon_root/placeholder" "$volume"
if [[ ! -x "$repository/.venv/bin/python" ]]; then
  print -u2 "project environment is missing its Python runtime"
  exit 1
fi
for helper in "$manifest_helper" "$trainer_supervisor"; do
  if [[ ! -f "$helper" || -L "$helper" ]]; then
    print -u2 "v0.3 launcher helper is missing or unsafe: $helper"
    exit 1
  fi
done
if ! "$repository/.venv/bin/python" -c \
  "import $trainer_module, $dashboard_module, $qualifier_module" >/dev/null 2>&1; then
  print -u2 "project environment cannot import the fixed v0.3 runtime"
  exit 1
fi

assert_frozen_parent
source_commit=$(clean_source_commit) || {
  print -u2 "refusing to launch v0.3 Stage A from dirty or uncommitted source"
  exit 1
}
authenticate_published_training_tag
assert_v03_qualification
if [[ -e "$run_root" || -L "$run_root" || -e "$media_root" || -L "$media_root" ]]; then
  print -u2 "refusing to reuse the canonical fresh v0.3 Stage-A roots"
  exit 1
fi
assert_free_space
assert_no_competing_training_chain
[[ -z "$(dashboard_listener_pid)" ]] || {
  print -u2 "v0.3 dashboard port $dashboard_port is already in use"
  exit 1
}

/bin/mkdir -m 0755 "$run_root"
/bin/mkdir -m 0755 "$media_root"
assert_regular_ancestor_chain "$run_root/placeholder" "$dungeon_root"
assert_regular_ancestor_chain "$media_root/placeholder" "$dungeon_root"
manifest_command create \
  --root "$run_root" \
  --media-root "$media_root" \
  --qualification-report "$qualification_report" \
  --source-commit "$source_commit" \
  --tag-object "$tag_object" >/dev/null
cohort_created=true

assert_source_unchanged
assert_free_space
assert_no_competing_training_chain
start_dashboard

/usr/bin/caffeinate -ims >/dev/null 2>&1 &
caffeine_pid=$!
kill -0 "$caffeine_pid"

while true; do
  assert_source_unchanged
  assert_free_space
  assert_dashboard_healthy
  [[ -z "$(active_neural_trainers)" ]] || {
    print -u2 "a neural trainer exists before the next v0.3 arm"
    exit 1
  }
  [[ -z "$(active_trainer_supervisors)" ]] || {
    print -u2 "a trainer supervisor exists before the next v0.3 arm"
    exit 1
  }
  [[ "$(active_caffeinates)" == "$caffeine_pid" ]] || {
    print -u2 "the v0.3 launcher lost its sole caffeinate identity"
    exit 1
  }
  plan_json=$(
    manifest_command next \
      --root "$run_root" \
      --source-commit "$source_commit" \
      --tag-object "$tag_object"
  )
  plan_fields=("${(@f)$(
    "$repository/.venv/bin/python" - "$plan_json" <<'PY'
import json
import sys

plan = json.loads(sys.argv[1])
if plan.get("done") is True:
    print("done")
    print("true" if plan.get("closeout_required") is True else "false")
else:
    print("run")
    for key in ("arm", "attempt", "run_dir", "media_dir"):
        print(plan[key])
PY
  )}")
  if [[ "${plan_fields[1]}" == done ]]; then
    if (( ${#plan_fields} != 2 )) || [[ "${plan_fields[2]}" != true ]]; then
      print -u2 "v0.3 planner reached done without a required closeout"
      exit 1
    fi
    break
  fi
  if (( ${#plan_fields} != 5 )) || [[ "${plan_fields[1]}" != run ]]; then
    print -u2 "v0.3 next-arm planner returned an incomplete exact plan"
    exit 1
  fi
  active_arm=${plan_fields[2]}
  attempt=${plan_fields[3]}
  run_directory=${plan_fields[4]}
  media_directory=${plan_fields[5]}
  if [[ "$active_arm" != sham && "$active_arm" != action-effect ]] \
    || [[ "$attempt" != 0 ]] \
    || [[ "$run_directory" != "$run_root/$active_arm" ]] \
    || [[ "$media_directory" != "$media_root/$active_arm" ]]; then
    print -u2 "v0.3 planner changed its fixed arm identity"
    exit 1
  fi

  manifest_command start-arm \
    --root "$run_root" \
    --source-commit "$source_commit" \
    --tag-object "$tag_object" \
    --arm "$active_arm" >/dev/null
  if [[ -e "$run_directory" || -L "$run_directory" ]] \
    || [[ -e "$media_directory" || -L "$media_directory" ]]; then
    print -u2 "refusing to reuse a fresh v0.3 arm target"
    exit 1
  fi
  /bin/mkdir -m 0755 "$run_directory"
  /bin/mkdir -m 0755 "$media_directory"
  assert_regular_ancestor_chain "$run_directory/placeholder" "$dungeon_root"
  assert_regular_ancestor_chain "$media_directory/placeholder" "$dungeon_root"

  supervisor_state="$run_root/launcher-$active_arm-attempt-$attempt.json"
  if [[ -e "$supervisor_state" || -L "$supervisor_state" ]]; then
    print -u2 "refusing to replace v0.3 supervisor evidence: $supervisor_state"
    exit 1
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
    assert_single_active_chain "$supervisor_state"
  fi
  set +e
  wait "$supervisor_pid"
  trainer_status=$?
  supervisor_pid=""
  trainer_pid=""
  set -e
  if [[ "$status_ready" != true || ! -f "$status_path" || -L "$status_path" ]]; then
    manifest_command finish-arm \
      --root "$run_root" --source-commit "$source_commit" \
      --tag-object "$tag_object" --arm "$active_arm" --outcome crashed \
      --trainer-exit-code 1 >/dev/null
    active_arm=""
    print -u2 "v0.3 arm stopped before publishing safe status evidence"
    exit 1
  fi
  terminal_phase=$(read_status_phase "$status_path")
  assert_source_unchanged
  assert_free_space
  assert_dashboard_healthy
  if [[ "$terminal_phase" == interrupted ]]; then
    manifest_command finish-arm \
      --root "$run_root" --source-commit "$source_commit" \
      --tag-object "$tag_object" --arm "$active_arm" --outcome interrupted \
      --trainer-exit-code "$trainer_status" >/dev/null
    active_arm=""
    print -u2 "v0.3 arm was interrupted; the complete matched root is terminal operationally_incomplete."
    exit 130
  fi
  if [[ "$terminal_phase" == crashed || "$trainer_status" -ne 0 ]]; then
    manifest_command finish-arm \
      --root "$run_root" --source-commit "$source_commit" \
      --tag-object "$tag_object" --arm "$active_arm" --outcome crashed \
      --trainer-exit-code "$trainer_status" >/dev/null
    active_arm=""
    print -u2 "v0.3 arm failed; this root can never be resumed or reused."
    exit 1
  fi
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

assert_source_unchanged
assert_free_space
[[ -z "$(active_neural_trainers)" ]] || {
  print -u2 "a neural trainer remained after both v0.3 arms"
  exit 1
}
[[ -z "$(active_trainer_supervisors)" ]] || {
  print -u2 "a trainer supervisor remained after both v0.3 arms"
  exit 1
}
assert_dashboard_healthy
stop_caffeine
manifest_command seal-process-closeout \
  --root "$run_root" \
  --source-commit "$source_commit" \
  --tag-object "$tag_object" >/dev/null
manifest_command finalize \
  --root "$run_root" \
  --source-commit "$source_commit" \
  --tag-object "$tag_object" >/dev/null
assert_dashboard_healthy
launcher_finalized=true
trap - EXIT INT TERM
print "v0.3 Stage A completed and sealed its matched architecture decision."
print "Read-only dashboard remains available: http://127.0.0.1:$dashboard_port/"
