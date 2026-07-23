#!/bin/zsh
set -euo pipefail

repository=${0:A:h:h}
volume="/Volumes/T7 Developer"
dungeon_root="$volume/DungeonApprentice"
run_root="$dungeon_root/u2-separated-20260723"
media_root="$dungeon_root/u2-separated-media-20260723"
qualification_report="$dungeon_root/qualifications/v0.2-u2-20260723/report.json"
confirmation_report="$dungeon_root/confirmations/v0.2-u1-v2-20260723/report.json"
trainer="$repository/.venv/bin/dungeon-train-v02-u2"
dashboard="$repository/.venv/bin/dungeon-dashboard-v02-u2"
manifest_helper="$repository/scripts/u2_cohort_manifest.py"
trainer_supervisor="$repository/scripts/u2_trainer_supervisor.py"
dashboard_port=8785
minimum_free_gib=25

seeds=(20260737 20260741 20260745)
lineage_ids=(lineage-1 lineage-2 lineage-3)
run_names=(
  v02-u2-seed-20260737
  v02-u2-seed-20260741
  v02-u2-seed-20260745
)
parents=(
  "$dungeon_root/u1-local-20260722/v02-u1-lead-seed-20260725/checkpoints/mastered-local-unlock.zip"
  "$dungeon_root/u1-local-replication-20260722/v02-u1-replication-seed-20260729/checkpoints/mastered-local-unlock.zip"
  "$dungeon_root/u1-local-replication-20260722/v02-u1-replication-seed-20260733/checkpoints/mastered-local-unlock.zip"
)
parent_sha256=(
  bcce9b8251e97ed4fddda32871c891c3783c057bbb1f89deedb3a3d32058102a
  2a300927b48f966d5f6ddfeefe13d2e444da1e5c70bcd54e86abd6a9b2d1830b
  3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104
)
states=(pending pending pending)
mode=fresh
if (( $# == 1 )) && [[ "$1" == "--resume" ]]; then
  mode=resume
elif (( $# != 0 )); then
  print -u2 "the frozen U2 cohort launcher accepts only an optional --resume"
  exit 2
fi

manifest_initialized=false
launcher_finalized=false
signal_requested=false
active_index=0
active_lineage_id=""
active_run_name=""
supervisor_pid=""
caffeine_pid=""

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
    print -u2 "U2 source became dirty or lost its committed HEAD"
    return 1
  }
  if [[ "$measured" != "$source_commit" ]]; then
    print -u2 "U2 source changed between sequential lineages"
    return 1
  fi
}

assert_free_space() {
  "$repository/.venv/bin/python" - "$volume" "$minimum_free_gib" <<'PY'
import shutil
import sys

volume, minimum_gib = sys.argv[1:]
free = shutil.disk_usage(volume).free
minimum = int(float(minimum_gib) * 1024**3)
if free < minimum:
    raise SystemExit(
        f"U2 requires {minimum_gib} GiB free, but only {free / 1024**3:.2f} GiB remains"
    )
PY
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

active_neural_trainers() {
  /usr/bin/pgrep -f \
    '[d]ungeon-train|dungeon_apprentice\.(train|v02_sentinel|v02_u1|v02_u2)([[:space:]]|$)' \
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

assert_regular_ancestor_chain() {
  local artifact_path=$1
  local stop=$2
  local current=${artifact_path:h}
  while true; do
    if [[ ! -d "$current" || -L "$current" ]]; then
      print -u2 "U2 artifact path has a missing or symlinked ancestor: $current"
      return 1
    fi
    [[ "$current" == "$stop" ]] && return 0
    if [[ "$current" == "/" ]]; then
      print -u2 "U2 artifact path is outside its frozen storage root: $artifact_path"
      return 1
    fi
    current=${current:h}
  done
}

assert_dashboard_healthy() {
  if [[ -n "${dashboard_pid:-}" ]] && ! kill -0 "$dashboard_pid" 2>/dev/null; then
    print -u2 "the independent U2 dashboard process stopped unexpectedly"
    return 1
  fi
  "$repository/.venv/bin/python" - "$dashboard_port" "$source_commit" <<'PY'
import json
import sys
import urllib.request

port, source_commit = sys.argv[1:]
url = f"http://127.0.0.1:{port}/api/cohort.json"
try:
    with urllib.request.urlopen(url, timeout=2) as response:
        payload = json.load(response)
except Exception as error:
    raise SystemExit(f"the independent U2 dashboard is unavailable: {error}") from error
manifest = payload.get("manifest", {})
if (
    manifest.get("cohort_id") != "v0.2-u2-separated-20260723"
    or manifest.get("source_commit") != source_commit
    or [item.get("seed") for item in manifest.get("lineages", [])]
    != [20260737, 20260741, 20260745]
):
    raise SystemExit("dashboard endpoint does not belong to this frozen U2 cohort")
PY
}

update_manifest() {
  local phase=$1
  local active=${2:-}
  local arguments=(
    update
    --root "$run_root"
    --source-commit "$source_commit"
    --phase "$phase"
    --state "$states[1]"
    --state "$states[2]"
    --state "$states[3]"
  )
  if [[ -n "$active" ]]; then
    arguments+=(--active-lineage-id "$active")
  fi
  "$repository/.venv/bin/python" "$manifest_helper" "${arguments[@]}" >/dev/null
}

start_segment_history() {
  local lineage_id=$1
  local directory=$2
  local resume_checkpoint=${3:-}
  local arguments=(
    segment-start
    --root "$run_root"
    --source-commit "$source_commit"
    --lineage-id "$lineage_id"
    --directory "$directory"
  )
  if [[ -n "$resume_checkpoint" ]]; then
    arguments+=(--resume-checkpoint "$resume_checkpoint")
  fi
  "$repository/.venv/bin/python" "$manifest_helper" "${arguments[@]}" >/dev/null
}

finish_segment_history() {
  local terminal_state=$1
  if [[ "$manifest_initialized" != true || -z "$active_lineage_id" || -z "$active_run_name" ]]; then
    return 0
  fi
  "$repository/.venv/bin/python" "$manifest_helper" \
    segment-finish \
    --root "$run_root" \
    --source-commit "$source_commit" \
    --lineage-id "$active_lineage_id" \
    --directory "$active_run_name" \
    --segment-state "$terminal_state" >/dev/null
  states[$active_index]="$terminal_state"
  active_lineage_id=""
  active_run_name=""
  active_index=0
}

stop_caffeine() {
  if [[ -n "$caffeine_pid" ]]; then
    kill "$caffeine_pid" 2>/dev/null || true
    wait "$caffeine_pid" 2>/dev/null || true
    caffeine_pid=""
  fi
}

record_unexpected_exit() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "$supervisor_pid" ]]; then
    kill -TERM "$supervisor_pid" 2>/dev/null || true
    wait "$supervisor_pid" 2>/dev/null || true
    supervisor_pid=""
  fi
  stop_caffeine
  if [[ "$launcher_finalized" != true && "$manifest_initialized" == true && -n "$active_lineage_id" ]]; then
    local terminal_state=crashed
    if [[ "$signal_requested" == true || "$exit_code" -eq 130 || "$exit_code" -eq 143 ]]; then
      terminal_state=interrupted
    fi
    finish_segment_history "$terminal_state" 2>/dev/null || true
  fi
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

trap record_unexpected_exit EXIT
trap request_launcher_stop INT TERM

read_terminal_phase() {
  local status_path=$1
  "$repository/.venv/bin/python" - "$status_path" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    status = json.load(stream)
phase = status.get("phase")
if phase not in {"completed", "mastered", "interrupted", "crashed"}:
    raise SystemExit(f"unexpected or missing terminal trainer phase: {phase!r}")
print(phase)
PY
}

start_or_reuse_dashboard() {
  if /usr/sbin/lsof -nP -iTCP:"$dashboard_port" -sTCP:LISTEN >/dev/null 2>&1; then
    if [[ "$mode" != resume ]]; then
      print -u2 "U2 dashboard port $dashboard_port is already in use"
      return 1
    fi
    if ! assert_dashboard_healthy; then
      print -u2 "dashboard port $dashboard_port is occupied by another cohort or service"
      return 1
    fi
    print "Reusing the healthy U2 dashboard at http://127.0.0.1:$dashboard_port/"
    return 0
  fi

  dashboard_log="$run_root/dashboard.log"
  dashboard_pid=$(
    "$repository/.venv/bin/python" - \
      "$dashboard" "$run_root" "$dashboard_port" "$dashboard_log" <<'PY'
import subprocess
import sys

command, root, port, log_path = sys.argv[1:]
with open(log_path, "ab", buffering=0) as log:
    process = subprocess.Popen(
        [command, root, "--port", port],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        close_fds=True,
        start_new_session=True,
    )
print(process.pid)
PY
  )

  if ! "$repository/.venv/bin/python" - "$dashboard_pid" "$dashboard_port" <<'PY'
import os
import sys
import time
import urllib.request

pid, port = (int(value) for value in sys.argv[1:])
url = f"http://127.0.0.1:{port}/api/cohort.json"
for _attempt in range(50):
    try:
        os.kill(pid, 0)
        with urllib.request.urlopen(url, timeout=0.5) as response:
            if response.status == 200:
                raise SystemExit(0)
    except (OSError, TimeoutError):
        pass
    time.sleep(0.1)
raise SystemExit("U2 dashboard did not become healthy on its fixed endpoint")
PY
  then
    kill "$dashboard_pid" 2>/dev/null || true
    return 1
  fi

  print -r -- "$dashboard_pid" >"$run_root/dashboard.pid"
  print "U2 cohort dashboard: http://127.0.0.1:$dashboard_port/"
}

assert_actual_mounted_volume

if [[ ! -d "$dungeon_root" || -L "$dungeon_root" ]]; then
  print -u2 "U2 parent directory is missing or unsafe: $dungeon_root"
  exit 1
fi

if [[ ! -x "$repository/.venv/bin/python" || ! -x "$trainer" || ! -x "$dashboard" ]]; then
  print -u2 "project environment is missing a frozen U2 command"
  exit 1
fi

if [[ ! -f "$manifest_helper" || -L "$manifest_helper" ]] \
  || [[ ! -f "$trainer_supervisor" || -L "$trainer_supervisor" ]]; then
  print -u2 "U2 launcher helper is missing or unsafe"
  exit 1
fi

if [[ "$mode" == fresh ]]; then
  if [[ -e "$run_root" || -L "$run_root" ]]; then
    print -u2 "refusing to reuse the fresh U2 cohort target: $run_root"
    exit 1
  fi
  if [[ -e "$media_root" || -L "$media_root" ]]; then
    print -u2 "refusing to reuse the fresh U2 media target: $media_root"
    exit 1
  fi
elif [[ ! -d "$run_root" || -L "$run_root" ]]; then
  print -u2 "resume requires the existing regular U2 cohort root: $run_root"
  exit 1
elif [[ ! -d "$media_root" || -L "$media_root" ]]; then
  print -u2 "resume requires the existing regular U2 media root: $media_root"
  exit 1
fi

if [[ ! -f "$qualification_report" || -L "$qualification_report" ]]; then
  print -u2 "frozen U2 qualification report is missing or unsafe: $qualification_report"
  exit 1
fi
assert_regular_ancestor_chain "$qualification_report" "$dungeon_root"

if [[ ! -f "$confirmation_report" || -L "$confirmation_report" ]]; then
  print -u2 "frozen U1 confirmation report is missing or unsafe: $confirmation_report"
  exit 1
fi
assert_regular_ancestor_chain "$confirmation_report" "$dungeon_root"

source_commit=$(clean_source_commit) || {
  print -u2 "refusing to train U2 from dirty or uncommitted source"
  exit 1
}

for (( index = 1; index <= ${#seeds}; index++ )); do
  parent=${parents[$index]}
  if [[ ! -f "$parent" || -L "$parent" ]]; then
    print -u2 "frozen U1 parent is missing or unsafe: $parent"
    exit 1
  fi
  assert_regular_ancestor_chain "$parent" "$dungeon_root"
  checksum_line=$(/usr/bin/shasum -a 256 "$parent")
  if [[ "${checksum_line%% *}" != "$parent_sha256[$index]" ]]; then
    print -u2 "frozen U1 parent checksum changed: $parent"
    exit 1
  fi
done

assert_source_unchanged
assert_free_space
assert_no_active_neural_trainer

"$repository/.venv/bin/python" - "$repository" "$qualification_report" "$source_commit" <<'PY'
import sys
from pathlib import Path

from dungeon_apprentice.u2_qualification_anchor import verify_external_anchor
from dungeon_apprentice.v02_u2_qualify import verify_u2_qualification_report

anchor = verify_external_anchor(Path(sys.argv[1]))
evidence = verify_u2_qualification_report(
    Path(sys.argv[2]),
    expected_source_commit=sys.argv[3],
    expected_sha256=anchor.report_sha256,
    expected_attempt_id=anchor.attempt_id,
    expected_claim_id=anchor.claim_id,
)
public = evidence.public_dict()
anchored = anchor.public_dict()
for field in (
    "report_sha256",
    "report_byte_length",
    "attempt_id",
    "attempt_sha256",
    "claim_id",
    "claim_sha256",
    "source_commit",
    "generator_profile",
    "generator_profile_version",
):
    if public[field] != anchored[field]:
        raise SystemExit(f"qualification evidence differs from external anchor: {field}")
PY

assert_source_unchanged
assert_free_space
assert_no_active_neural_trainer

if [[ "$mode" == fresh ]]; then
  /bin/mkdir -m 0755 "$media_root"
  if ! /bin/mkdir -m 0755 "$run_root"; then
    /bin/rmdir "$media_root" 2>/dev/null || true
    exit 1
  fi
  if ! "$repository/.venv/bin/python" "$manifest_helper" \
    create \
    --root "$run_root" \
    --source-commit "$source_commit" \
    --media-directory "$media_root" >/dev/null
  then
    /bin/rmdir "$run_root" 2>/dev/null || true
    /bin/rmdir "$media_root" 2>/dev/null || true
    exit 1
  fi
fi
manifest_initialized=true

start_index=1
resume_checkpoint=""
resume_run_name=""
if [[ "$mode" == resume ]]; then
  resume_plan_json=$(
    "$repository/.venv/bin/python" "$manifest_helper" \
      resume-plan \
      --root "$run_root" \
      --source-commit "$source_commit"
  )
  resume_fields=("${(@f)$(
    "$repository/.venv/bin/python" - "$resume_plan_json" <<'PY'
import json
import sys

plan = json.loads(sys.argv[1])
fields = (
    plan["lineage_order"],
    plan["lineage_id"],
    plan["seed"],
    plan["directory"],
    plan["resume_checkpoint"],
    *plan["lineage_states"],
)
for value in fields:
    print(value)
PY
  )}")
  if (( ${#resume_fields} != 8 )); then
    print -u2 "U2 resume planner returned an incomplete frozen plan"
    exit 1
  fi
  start_index=${resume_fields[1]}
  if [[ "$resume_fields[2]" != "$lineage_ids[$start_index]" ]]; then
    print -u2 "U2 resume planner selected the wrong frozen lineage"
    exit 1
  fi
  if [[ "$resume_fields[3]" != "$seeds[$start_index]" ]]; then
    print -u2 "U2 resume planner selected the wrong child seed"
    exit 1
  fi
  resume_run_name=${resume_fields[4]}
  resume_checkpoint=${resume_fields[5]}
  states=("${resume_fields[6]}" "${resume_fields[7]}" "${resume_fields[8]}")
  print "Resuming ${lineage_ids[$start_index]} from its latest verified safe boundary."
fi

start_or_reuse_dashboard

for (( index = start_index; index <= ${#seeds}; index++ )); do
  seed=${seeds[$index]}
  lineage_id=${lineage_ids[$index]}
  parent=${parents[$index]}
  segment_resume=""
  if [[ "$mode" == resume && "$index" -eq "$start_index" ]]; then
    run_name="$resume_run_name"
    segment_resume="$resume_checkpoint"
  else
    run_name=${run_names[$index]}
  fi
  target="$run_root/$run_name"

  assert_source_unchanged
  assert_free_space
  assert_no_active_neural_trainer
  assert_dashboard_healthy
  if [[ -e "$target" || -L "$target" ]]; then
    print -u2 "fresh U2 lineage target unexpectedly exists: $target"
    states[$index]=crashed
    update_manifest operational_failure "$lineage_id"
    exit 1
  fi

  states[$index]=training
  start_segment_history "$lineage_id" "$run_name" "$segment_resume"
  active_index=$index
  active_lineage_id="$lineage_id"
  active_run_name="$run_name"
  print "Starting U2 ${lineage_ids[$index]} (child $seed)"

  resume_arguments=()
  if [[ -n "$segment_resume" ]]; then
    resume_arguments=(--resume "$segment_resume")
  fi
  supervisor_state="$run_root/launcher-$run_name.json"
  /usr/bin/caffeinate -ims >/dev/null 2>&1 &
  caffeine_pid=$!
  set +e
  "$repository/.venv/bin/python" "$trainer_supervisor" \
    --state-path "$supervisor_state" \
    -- \
      "$trainer" \
      --parent "$parent" \
      --confirmation-report "$confirmation_report" \
      --qualification-report "$qualification_report" \
      --seed "$seed" \
      --child-budget 1048576 \
      --workers 4 \
      --size 9 \
      --device cpu \
      --rollout-steps 512 \
      --batch-size 256 \
      --n-epochs 4 \
      --learning-rate 0.00025 \
      --gamma 0.995 \
      --gae-lambda 0.98 \
      --evaluation-every 32768 \
      --evaluation-seeds 80 \
      --frame-every 2048 \
      --keep-rolling-exams 5 \
      --minimum-free-gib "$minimum_free_gib" \
      --storage-root "$volume" \
      --run-root "$run_root" \
      --run-name "$run_name" \
      --media-directory "$media_root" \
      "${resume_arguments[@]}" &
  supervisor_pid=$!
  wait "$supervisor_pid"
  trainer_status=$?
  if [[ "$signal_requested" == true ]] && kill -0 "$supervisor_pid" 2>/dev/null; then
    wait "$supervisor_pid"
    trainer_status=$?
  fi
  supervisor_pid=""
  set -e
  stop_caffeine

  if [[ ! -f "$target/status.json" ]]; then
    if [[ "$signal_requested" == true ]]; then
      finish_segment_history interrupted
      print -u2 "U2 cohort was interrupted before the trainer published status"
      exit 130
    fi
    finish_segment_history crashed
    print -u2 "U2 child $seed stopped with an engineering failure"
    exit 1
  fi
  terminal_phase=$(read_terminal_phase "$target/status.json") || {
    finish_segment_history crashed
    exit 1
  }
  if [[ "$terminal_phase" == "interrupted" ]]; then
    finish_segment_history interrupted
    print -u2 "U2 cohort was interrupted during child $seed"
    exit 130
  fi
  if [[ "$terminal_phase" == "crashed" || "$trainer_status" -ne 0 ]]; then
    finish_segment_history crashed
    print -u2 "U2 child $seed returned a nonzero launcher status"
    exit 1
  fi
  finish_segment_history "$terminal_phase"
done

assert_source_unchanged
assert_free_space
assert_no_active_neural_trainer
assert_dashboard_healthy
update_manifest completed ""
launcher_finalized=true
trap - EXIT INT TERM
print "All three U2 lineages finished. The independent dashboard remains available:"
print "http://127.0.0.1:$dashboard_port/"
