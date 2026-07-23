#!/bin/zsh
set -euo pipefail

repository=${0:A:h:h}
volume="/Volumes/T7 Developer"
dungeon_root="$volume/DungeonApprentice"
run_root="$dungeon_root/u2s-ablation-20260723"
media_root="$dungeon_root/u2s-ablation-media-20260723"
contract="$run_root/cohort-contract.json"
parent="$dungeon_root/u1-local-replication-20260722/v02-u1-replication-seed-20260733/checkpoints/mastered-local-unlock.zip"
parent_sidecar="${parent:r}.json"
parent_manifest="${parent:h:h}/manifest.json"
confirmation_report="$dungeon_root/confirmations/v0.2-u1-v2-20260723/report.json"
confirmation_attempt="${confirmation_report:h}/attempt.json"
confirmation_checksum="$confirmation_report.sha256"
qualification_report="$dungeon_root/qualifications/v0.2-u2s-20260723/report.json"
qualification_checksum="$qualification_report.sha256"
parent_sha256="3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104"
parent_sidecar_sha256="268f89361521dd855ba637254b236592186992718ad37ac374e85e5dbf408a07"
parent_manifest_sha256="cd808ba8fea6b79225895b455d975122f85d66f1451569bd04a69737f8a6dfba"
confirmation_sha256="6e577170050f6f14599b793a031776a19bf7c64eba0f243f457298da3193ae8f"
confirmation_attempt_sha256="62147b2b556fe51e83876f2be35467fb488a4fb3b9c6347a600cf969e2bc86eb"
confirmation_checksum_sha256="0be93cbbaf6581c4f6188ff6c4c82dee630f28da1cce462fa2acdb9caebed71a"
training_protocol="dungeon-apprentice-v0.2-u2s-stability-ablation"
cohort_id="v0.2-u2s-ablation-20260723"
training_tag="u2s-stability-ablation-v0.2-u2s-20260723"
trainer_module="dungeon_apprentice.v02_u2s"
dashboard_module="dungeon_apprentice.v02_u2s_dashboard"
manifest_helper="$repository/scripts/u2s_ablation_manifest.py"
trainer_supervisor="$repository/scripts/u2_trainer_supervisor.py"
dashboard_port=8787
minimum_free_gib=25

if (( $# != 0 )); then
  print -u2 "the fixed U2-S ablation launcher accepts no options; interrupted cohorts must restart all four arms"
  exit 2
fi

source_commit=""
tag_object=""
supervisor_pid=""
caffeine_pid=""
dashboard_pid=""
active_arm=""
signal_requested=false
launcher_finalized=false

cd "$repository"

clean_source_commit() {
  local commit
  commit=$(git rev-parse --verify 'HEAD^{commit}' 2>/dev/null) || return 1
  [[ -n "$commit" ]] || return 1
  [[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || return 1
  print -r -- "$commit"
}

assert_source_unchanged() {
  local measured
  measured=$(clean_source_commit) || {
    print -u2 "U2-S source became dirty or lost its committed HEAD"
    return 1
  }
  [[ "$measured" == "$source_commit" ]] || {
    print -u2 "U2-S source commit changed after its contract was authenticated"
    return 1
  }
  local measured_tag
  measured_tag=$(git rev-parse --verify "refs/tags/$training_tag" 2>/dev/null) || return 1
  [[ "$measured_tag" == "$tag_object" ]] || {
    print -u2 "U2-S annotated tag object changed"
    return 1
  }
}

authenticate_training_tag() {
  local object_type peeled
  object_type=$(git cat-file -t "refs/tags/$training_tag" 2>/dev/null) || {
    print -u2 "required U2-S annotated tag is missing: $training_tag"
    return 1
  }
  [[ "$object_type" == tag ]] || {
    print -u2 "U2-S training tag must be an annotated tag object"
    return 1
  }
  tag_object=$(git rev-parse --verify "refs/tags/$training_tag")
  peeled=$(git rev-parse --verify "refs/tags/$training_tag^{commit}")
  [[ "$peeled" == "$source_commit" ]] || {
    print -u2 "U2-S tag does not peel to the clean source commit"
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
        f"U2-S requires {minimum_gib} GiB free, but only "
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
      print -u2 "U2-S artifact path has a missing or symlinked ancestor: $current"
      return 1
    fi
    [[ "$current" == "$stop" ]] && return 0
    if [[ "$current" == "/" ]]; then
      print -u2 "U2-S artifact path escapes its fixed storage root"
      return 1
    fi
    current=${current:h}
  done
}

assert_frozen_file() {
  local path=$1
  local expected=$2
  local label=$3
  if [[ ! -f "$path" || -L "$path" ]]; then
    print -u2 "$label is missing or unsafe: $path"
    return 1
  fi
  assert_regular_ancestor_chain "$path" "$dungeon_root"
  local checksum_line
  checksum_line=$(/usr/bin/shasum -a 256 "$path")
  [[ "${checksum_line%% *}" == "$expected" ]] || {
    print -u2 "$label checksum changed: $path"
    return 1
  }
}

assert_u2s_qualification() {
  "$repository/.venv/bin/python" - \
    "$repository" "$qualification_report" "$source_commit" "$tag_object" <<'PY'
import sys
from pathlib import Path

from dungeon_apprentice.v02_u2s_qualify import verify_u2s_qualification

repository, report, source_commit, tag_object = sys.argv[1:]
evidence = verify_u2s_qualification(
    Path(report),
    expected_source_commit=source_commit,
    expected_tag_object=tag_object,
    repository=Path(repository),
)
public = evidence.public_dict()
if (
    public.get("verdict") != "qualified"
    or public.get("source_commit") != source_commit
    or public.get("tag_object") != tag_object
    or not public.get("tag_payload_sha256")
):
    raise SystemExit("U2-S qualification identity differs from this launch")
PY
}

active_neural_trainers() {
  /usr/bin/pgrep -f \
    '[d]ungeon-train|dungeon_apprentice\.(train|v02_sentinel|v02_u1|v02_u2|v02_u2r|v02_u2s)([[:space:]]|$)' \
    || true
}

assert_no_active_neural_trainer() {
  local active
  active=$(active_neural_trainers)
  if [[ -n "$active" ]]; then
    print -u2 "refusing concurrent neural training; active process IDs: ${active//$'\n'/, }"
    return 1
  fi
}

dashboard_listener_pid() {
  /usr/sbin/lsof -nP -t -iTCP:"$dashboard_port" -sTCP:LISTEN 2>/dev/null \
    | /usr/bin/head -n 1 || true
}

assert_dashboard_healthy() {
  "$repository/.venv/bin/python" - \
    "$dashboard_port" "$source_commit" "$tag_object" "$parent_sha256" <<'PY'
import json
import sys
import urllib.request

port, source, tag_object, parent = sys.argv[1:]
with urllib.request.urlopen(
    f"http://127.0.0.1:{port}/api/u2s.json",
    timeout=2,
) as response:
    payload = json.load(response)
if (
    payload.get("cohort_id") != "v0.2-u2s-ablation-20260723"
    or payload.get("protocol")
    != "dungeon-apprentice-v0.2-u2s-stability-ablation"
    or payload.get("source_commit") != source
    or payload.get("tag_object") != tag_object
    or payload.get("parent_checkpoint_sha256") != parent
):
    raise SystemExit("dashboard identity differs from this U2-S cohort")
PY
}

start_dashboard() {
  local listener
  listener=$(dashboard_listener_pid)
  if [[ -n "$listener" ]]; then
    print -u2 "U2-S dashboard port $dashboard_port is already in use"
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
        [python, "-m", module, root, "--port", port],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        close_fds=True,
        start_new_session=True,
    )
print(process.pid)
PY
  )
  for _attempt in {1..100}; do
    if ! kill -0 "$dashboard_pid" 2>/dev/null; then
      break
    fi
    if assert_dashboard_healthy 2>/dev/null; then
      print -r -- "$dashboard_pid" >"$run_root/dashboard.pid"
      print "U2-S matched-ablation dashboard: http://127.0.0.1:$dashboard_port/"
      return 0
    fi
    /bin/sleep 0.1
  done
  kill "$dashboard_pid" 2>/dev/null || true
  print -u2 "U2-S dashboard did not become healthy on its fixed endpoint"
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
    raise SystemExit(f"unexpected U2-S arm phase: {phase!r}")
print(phase)
PY
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
  manifest_command finish \
    --root "$run_root" \
    --source-commit "$source_commit" \
    --tag-object "$tag_object" \
    --arm "$active_arm" \
    --outcome "$outcome" \
    --trainer-exit-code "$trainer_exit_code" >/dev/null 2>&1 || true
  active_arm=""
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
    print -u2 "U2-S launcher helper is missing or unsafe: $helper"
    exit 1
  fi
done
if ! "$repository/.venv/bin/python" -c \
  "import $trainer_module, $dashboard_module" >/dev/null 2>&1; then
  print -u2 "project environment cannot import the fixed U2-S trainer and dashboard"
  exit 1
fi

assert_frozen_file "$parent" "$parent_sha256" "frozen U1 parent"
assert_frozen_file "$parent_sidecar" "$parent_sidecar_sha256" "frozen U1 parent sidecar"
assert_frozen_file "$parent_manifest" "$parent_manifest_sha256" "frozen U1 parent manifest"
assert_frozen_file "$confirmation_report" "$confirmation_sha256" "frozen U1 confirmation"
assert_frozen_file "$confirmation_attempt" "$confirmation_attempt_sha256" "frozen U1 confirmation attempt"
assert_frozen_file "$confirmation_checksum" "$confirmation_checksum_sha256" "frozen U1 confirmation checksum"
source_commit=$(clean_source_commit) || {
  print -u2 "refusing to launch U2-S from dirty or uncommitted source"
  exit 1
}
authenticate_training_tag
if [[ -e "$run_root" || -L "$run_root" || -e "$media_root" || -L "$media_root" ]]; then
  print -u2 "refusing to reuse the canonical fresh U2-S roots"
  exit 1
fi
assert_u2s_qualification
if [[ ! -f "$qualification_checksum" || -L "$qualification_checksum" ]]; then
  print -u2 "U2-S qualification checksum is missing or unsafe"
  exit 1
fi
assert_free_space
assert_no_active_neural_trainer

[[ -z "$(dashboard_listener_pid)" ]] || {
  print -u2 "U2-S dashboard port $dashboard_port is already in use"
  exit 1
}
/bin/mkdir -m 0755 "$run_root"
/bin/mkdir -m 0755 "$media_root"
assert_regular_ancestor_chain "$run_root/placeholder" "$dungeon_root"
assert_regular_ancestor_chain "$media_root/placeholder" "$dungeon_root"
manifest_command create \
  --root "$run_root" \
  --media-root "$media_root" \
  --source-commit "$source_commit" \
  --tag-object "$tag_object" >/dev/null

assert_source_unchanged
assert_free_space
assert_no_active_neural_trainer
start_dashboard

/usr/bin/caffeinate -ims >/dev/null 2>&1 &
caffeine_pid=$!

while true; do
  assert_source_unchanged
  assert_free_space
  assert_no_active_neural_trainer
  assert_dashboard_healthy
  plan_json=$(
    manifest_command next-plan \
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
else:
    print("run")
    for key in ("arm", "attempt", "run_dir", "media_dir"):
        print(plan[key])
PY
  )}")
  if [[ "${plan_fields[1]}" == done ]]; then
    break
  fi
  if (( ${#plan_fields} != 5 )); then
    print -u2 "U2-S next-arm planner returned an incomplete exact plan"
    exit 1
  fi
  active_arm=${plan_fields[2]}
  attempt=${plan_fields[3]}
  run_directory=${plan_fields[4]}
  media_directory=${plan_fields[5]}
  start_arguments=(
    start
    --root "$run_root"
    --source-commit "$source_commit"
    --tag-object "$tag_object"
    --arm "$active_arm"
  )
  manifest_command "${start_arguments[@]}" >/dev/null
  supervisor_state="$run_root/launcher-$active_arm-attempt-$attempt.json"
  if [[ -e "$supervisor_state" || -L "$supervisor_state" ]]; then
    print -u2 "refusing to replace U2-S supervisor evidence: $supervisor_state"
    exit 1
  fi
  print "Starting U2-S arm $active_arm (matched attempt $attempt)."
  set +e
  "$repository/.venv/bin/python" "$trainer_supervisor" \
    --state-path "$supervisor_state" \
    -- \
      "$repository/.venv/bin/python" -m "$trainer_module" train-arm \
      --arm "$active_arm" \
      --run-dir "$run_directory" \
      --media-dir "$media_directory" \
      --cohort-contract "$contract" \
      --qualification-report "$qualification_report" &
  supervisor_pid=$!
  set -e

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
  set +e
  wait "$supervisor_pid"
  trainer_status=$?
  supervisor_pid=""
  set -e
  if [[ "$status_ready" != true || ! -f "$status_path" || -L "$status_path" ]]; then
    manifest_command finish \
      --root "$run_root" --source-commit "$source_commit" \
      --tag-object "$tag_object" --arm "$active_arm" --outcome crashed \
      --trainer-exit-code 1 >/dev/null
    active_arm=""
    print -u2 "U2-S arm stopped before publishing safe status evidence"
    exit 1
  fi
  terminal_phase=$(read_status_phase "$status_path")
  assert_source_unchanged
  assert_free_space
  if [[ "$terminal_phase" == interrupted ]]; then
    manifest_command finish \
      --root "$run_root" --source-commit "$source_commit" \
      --tag-object "$tag_object" --arm "$active_arm" --outcome interrupted \
      --trainer-exit-code "$trainer_status" >/dev/null
    active_arm=""
    print -u2 "U2-S arm was interrupted; this cohort is terminal operationally_incomplete."
    exit 130
  fi
  if [[ "$terminal_phase" == crashed ]]; then
    manifest_command finish \
      --root "$run_root" --source-commit "$source_commit" \
      --tag-object "$tag_object" --arm "$active_arm" --outcome crashed \
      --trainer-exit-code "$trainer_status" >/dev/null
    active_arm=""
    print -u2 "U2-S arm ended with an engineering failure."
    exit 1
  fi
  if [[ "$trainer_status" -ne 0 ]]; then
    manifest_command finish \
      --root "$run_root" --source-commit "$source_commit" \
      --tag-object "$tag_object" --arm "$active_arm" --outcome crashed \
      --trainer-exit-code "$trainer_status" >/dev/null
    active_arm=""
    print -u2 "U2-S trainer exited nonzero; terminal-looking files cannot count."
    exit 1
  fi
  manifest_command finish \
    --root "$run_root" --source-commit "$source_commit" \
    --tag-object "$tag_object" --arm "$active_arm" --outcome completed \
    --trainer-exit-code "$trainer_status" >/dev/null
  print "Completed U2-S arm $active_arm at its exact 1,048,576-action boundary."
  active_arm=""
done

assert_source_unchanged
assert_free_space
assert_no_active_neural_trainer
assert_dashboard_healthy
stop_caffeine
manifest_command finalize \
  --root "$run_root" \
  --source-commit "$source_commit" \
  --tag-object "$tag_object" \
  --dashboard-pid "$dashboard_pid" >/dev/null
assert_dashboard_healthy
launcher_finalized=true
trap - EXIT INT TERM
print "U2-S matched ablation completed and cryptographically sealed its fixed-priority decision."
print "Read-only comparison dashboard remains available: http://127.0.0.1:$dashboard_port/"
