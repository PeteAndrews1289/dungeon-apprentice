#!/bin/zsh
set -euo pipefail

repository=${0:A:h:h}
volume="/Volumes/T7 Developer"
dungeon_root="$volume/DungeonApprentice"
run_root="$dungeon_root/u2r-stability-r1-20260723"
media_root="$dungeon_root/u2r-stability-r1-media-20260723"
initial_run_name="v02-u2r-r1-seed-20260745"
initial_target="$run_root/$initial_run_name"
parent="$dungeon_root/u2-separated-20260723/v02-u2-seed-20260745/checkpoints/mastered-separated-unlock.zip"
parent_sidecar="${parent:r}.json"
parent_integrity="${parent:r}.integrity.json"
parent_manifest="${parent:h:h}/manifest.json"
qualification_report="$dungeon_root/qualifications/v0.2-u2-20260723/report.json"
confirmation_report="$dungeon_root/confirmations/v0.2-u2-20260723/report.json"
confirmation_attempt="${confirmation_report:h}/attempt.json"
trainer_module="dungeon_apprentice.v02_u2r"
supervisor="$repository/scripts/u2_trainer_supervisor.py"
dashboard_module="dungeon_apprentice.v02_u2r_dashboard"
dashboard_port=8786
training_protocol="dungeon-apprentice-v0.2-u2r-stability-r1"
minimum_free_gib=25

parent_sha256="56dc459fb94110f41a14b2304425f572fc235c8cfc07e1152306cbf77ac33dee"
parent_sidecar_sha256="68cfb84ec3fbfd4204cfa9679d5b8c1fe75b17bdf26d828ed1f92510b439cc29"
parent_integrity_sha256="250ebf0a8fe98ed5370d20c9b55531ec5e1be33df6821fc73b670628ddfb6151"
parent_manifest_sha256="4d37f72e2679441c1bd3eeff81c1c95a079b2f180d21ace136b5d80a1972e2ec"
confirmation_report_sha256="7522eb8742ed567577d1f02a2a9960d698981044128e0866daa262d36aaf1c69"
confirmation_attempt_sha256="4487b9b45bb6f599d9212beded7ce39b0b2f655073f375a641137c1f4ea63099"

mode=fresh
if (( $# == 1 )) && [[ "$1" == "--resume" ]]; then
  mode=resume
elif (( $# != 0 )); then
  print -u2 "the frozen U2r launcher accepts only an optional --resume"
  exit 2
fi

supervisor_pid=""
caffeine_pid=""
dashboard_pid=""
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
    print -u2 "U2r source became dirty or lost its committed HEAD"
    return 1
  }
  if [[ "$measured" != "$source_commit" ]]; then
    print -u2 "U2r source commit changed after preregistration verification"
    return 1
  fi
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
        f"U2r requires {minimum_gib} GiB free, but only "
        f"{free / 1024**3:.2f} GiB remains"
    )
PY
}

active_neural_trainers() {
  /usr/bin/pgrep -f \
    '[d]ungeon-train|dungeon_apprentice\.(train|v02_sentinel|v02_u1|v02_u2|v02_u2r)([[:space:]]|$)' \
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
      print -u2 "U2r artifact path has a missing or symlinked ancestor: $current"
      return 1
    fi
    [[ "$current" == "$stop" ]] && return 0
    if [[ "$current" == "/" ]]; then
      print -u2 "U2r artifact path is outside its frozen storage root: $artifact_path"
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
  if [[ "${checksum_line%% *}" != "$expected" ]]; then
    print -u2 "$label checksum changed: $path"
    return 1
  fi
}

dashboard_listener_pid() {
  /usr/sbin/lsof -nP -t -iTCP:"$dashboard_port" -sTCP:LISTEN 2>/dev/null \
    | /usr/bin/head -n 1 || true
}

stop_prior_dashboard_for_resume() {
  local listener
  listener=$(dashboard_listener_pid)
  [[ -n "$listener" ]] || return 0
  local pid_file="$resume_source_directory/dashboard.pid"
  if [[ ! -f "$pid_file" || -L "$pid_file" ]]; then
    print -u2 "dashboard port $dashboard_port is occupied without a safe U2r PID record"
    return 1
  fi
  local recorded
  recorded=$(<"$pid_file")
  if [[ "$recorded" != <-> || "$recorded" != "$listener" ]]; then
    print -u2 "dashboard PID record does not own fixed port $dashboard_port"
    return 1
  fi
  local command
  command=$(/bin/ps -p "$recorded" -o command= 2>/dev/null || true)
  if [[ "$command" != *"$dashboard_module"* \
    || "$command" != *"$run_root"* \
    || "$command" != *"--port $dashboard_port"* ]]; then
    print -u2 "refusing to stop an unrecognized process on dashboard port $dashboard_port"
    return 1
  fi
  kill -TERM "$recorded"
  for _attempt in {1..50}; do
    kill -0 "$recorded" 2>/dev/null || break
    /bin/sleep 0.1
  done
  if kill -0 "$recorded" 2>/dev/null; then
    print -u2 "the prior verified U2r dashboard did not stop"
    return 1
  fi
  [[ -z "$(dashboard_listener_pid)" ]] || {
    print -u2 "dashboard port $dashboard_port remained occupied"
    return 1
  }
}

start_dashboard() {
  if [[ -n "$(dashboard_listener_pid)" ]]; then
    print -u2 "dashboard port $dashboard_port is already in use"
    return 1
  fi
  local dashboard_log="$target/dashboard.log"
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
  if ! "$repository/.venv/bin/python" - \
      "$dashboard_pid" "$dashboard_port" "$source_commit" "$training_protocol" \
      "$expected_segment_index" <<'PY'
import json
import os
import sys
import time
import urllib.request

pid, port, source_commit, protocol, expected_segment = sys.argv[1:]
url = f"http://127.0.0.1:{port}/api/u2r.json"
for _attempt in range(100):
    try:
        os.kill(int(pid), 0)
        with urllib.request.urlopen(url, timeout=0.5) as response:
            payload = json.load(response)
        active = payload.get("active_status", {})
        if (
            payload.get("protocol") == protocol
            and int(payload.get("active_segment_index", -1))
            == int(expected_segment)
            and active.get("source", {}).get("commit") == source_commit
        ):
            raise SystemExit(0)
    except (OSError, TimeoutError, ValueError):
        pass
    time.sleep(0.1)
raise SystemExit("U2r dashboard did not become healthy on its frozen endpoint")
PY
  then
    kill "$dashboard_pid" 2>/dev/null || true
    return 1
  fi
  print -r -- "$dashboard_pid" >"$target/dashboard.pid"
  print "U2r live dashboard: http://127.0.0.1:$dashboard_port/"
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
  print -u2 "U2r storage root is missing or unsafe: $dungeon_root"
  exit 1
fi
assert_regular_ancestor_chain "$dungeon_root/placeholder" "$volume"

if [[ ! -x "$repository/.venv/bin/python" ]]; then
  print -u2 "project environment is missing its Python runtime"
  exit 1
fi
if ! "$repository/.venv/bin/python" -c \
  "import $trainer_module" >/dev/null 2>&1; then
  print -u2 "project environment cannot import the frozen U2r trainer"
  exit 1
fi
if [[ ! -f "$supervisor" || -L "$supervisor" ]]; then
  print -u2 "U2r trainer supervisor is missing or unsafe"
  exit 1
fi

assert_frozen_file "$parent" "$parent_sha256" "frozen U2 parent"
assert_frozen_file "$parent_sidecar" "$parent_sidecar_sha256" "frozen U2 parent sidecar"
assert_frozen_file "$parent_integrity" "$parent_integrity_sha256" "frozen U2 parent integrity"
assert_frozen_file "$parent_manifest" "$parent_manifest_sha256" "frozen U2 parent manifest"
assert_frozen_file "$confirmation_report" "$confirmation_report_sha256" "frozen U2 confirmation"
assert_frozen_file "$confirmation_attempt" "$confirmation_attempt_sha256" "frozen U2 confirmation attempt"
if [[ ! -f "$qualification_report" || -L "$qualification_report" ]]; then
  print -u2 "frozen U2 qualification report is missing or unsafe"
  exit 1
fi
assert_regular_ancestor_chain "$qualification_report" "$dungeon_root"

source_commit=$(clean_source_commit) || {
  print -u2 "refusing to train U2r from dirty or uncommitted source"
  exit 1
}
assert_free_space
assert_no_active_neural_trainer

run_name="$initial_run_name"
resume_checkpoint=""
resume_source_directory=""
expected_segment_index=0
if [[ "$mode" == fresh ]]; then
  if [[ -e "$initial_target" || -L "$initial_target" ]]; then
    print -u2 "refusing to reuse the canonical fresh U2r target: $initial_target"
    exit 1
  fi
  if [[ -e "$media_root" || -L "$media_root" ]]; then
    print -u2 "refusing to reuse the canonical fresh U2r media target: $media_root"
    exit 1
  fi
  if [[ -e "$run_root" && ( ! -d "$run_root" || -L "$run_root" ) ]]; then
    print -u2 "U2r run root exists but is not a regular directory"
    exit 1
  fi
  [[ -z "$(dashboard_listener_pid)" ]] || {
    print -u2 "dashboard port $dashboard_port is already in use"
    exit 1
  }
else
  if [[ ! -d "$run_root" || -L "$run_root" ]]; then
    print -u2 "U2r resume requires its existing regular run root"
    exit 1
  fi
  if [[ ! -d "$media_root" || -L "$media_root" ]]; then
    print -u2 "U2r resume requires its existing regular media root"
    exit 1
  fi
  resume_plan_json=$(
    "$repository/.venv/bin/python" - "$run_root" "$source_commit" <<'PY'
import json
import sys
from pathlib import Path

from dungeon_apprentice.u2r_anchor import select_resume_plan

plan = select_resume_plan(
    Path(sys.argv[1]),
    expected_source_commit=sys.argv[2],
)
print(json.dumps(plan.public_dict(), sort_keys=True, separators=(",", ":")))
PY
  )
  resume_fields=("${(@f)$(
    "$repository/.venv/bin/python" - "$resume_plan_json" <<'PY'
import json
import sys

plan = json.loads(sys.argv[1])
for key in (
    "checkpoint",
    "source_directory",
    "next_run_name",
    "next_segment_index",
):
    print(plan[key])
PY
  )}")
  if (( ${#resume_fields} != 4 )); then
    print -u2 "U2r resume planner returned an incomplete exact plan"
    exit 1
  fi
  resume_checkpoint=${resume_fields[1]}
  resume_source_directory=${resume_fields[2]}
  run_name=${resume_fields[3]}
  expected_segment_index=${resume_fields[4]}
  stop_prior_dashboard_for_resume
fi

target="$run_root/$run_name"
if [[ -e "$target" || -L "$target" ]]; then
  print -u2 "refusing to reuse the selected U2r segment target: $target"
  exit 1
fi

# This preflight verifies the external tag and exact exclusion-set identity.
# It uses only the already consumed U2 evidence and qualified validation access;
# no consumed or fresh confirmation candidate stream is generated.
"$repository/.venv/bin/python" - \
  "$repository" "$source_commit" "$qualification_report" <<'PY'
import sys
from pathlib import Path

from dungeon_apprentice import v02_u2 as frozen_u2
from dungeon_apprentice import v02_u2r
from dungeon_apprentice.u2_qualification_anchor import verify_external_anchor as verify_u2_anchor
from dungeon_apprentice.u2r_anchor import (
    protocol_document_sha256,
    verify_failed_r0_launch,
    verify_external_anchor,
)

repository = Path(sys.argv[1])
source_commit = sys.argv[2]
qualification_path = Path(sys.argv[3])
parent = v02_u2r.verify_u2r_parent()
qualification_anchor = verify_u2_anchor(
    repository,
    expected_source_commit=v02_u2r.SOURCE_TRAINING_COMMIT,
)
qualification = frozen_u2.verify_qualification(
    qualification_path,
    expected_source_commit=v02_u2r.SOURCE_TRAINING_COMMIT,
    anchor=qualification_anchor,
)
forbidden, exclusions = v02_u2r.build_u2r_forbidden_layout_hashes(
    qualification.verified_report(),
    parent.confirmation_snapshot(),
    access=qualification.seed_access(),
)
verify_failed_r0_launch()
verified = verify_external_anchor(
    repository,
    expected_source_commit=source_commit,
    expected_protocol_sha256=protocol_document_sha256(repository),
    expected_exclusions=exclusions,
)
sampler_preflight = v02_u2r.preflight_u2r_training_layout_sampler(
    forbidden,
    seed_access=qualification.seed_access(),
    worker_streams=v02_u2r.REMEDIATION_WORKER_STREAMS,
    max_attempts=v02_u2r.U2R_LAYOUT_RESAMPLE_ATTEMPTS,
)
print(
    "Verified U2r-r1 preregistration "
    f"{verified.tag} ({verified.tag_object[:12]}) with "
    f"{verified.static_exclusion_layouts:,} inventoried exact exclusions "
    f"and {len(sampler_preflight)} feasible lesson/worker pairs."
)
PY

assert_source_unchanged
assert_free_space
assert_no_active_neural_trainer

if [[ ! -d "$run_root" ]]; then
  /bin/mkdir -m 0755 "$run_root"
fi
assert_regular_ancestor_chain "$run_root/placeholder" "$dungeon_root"
if [[ "$mode" == fresh ]]; then
  /bin/mkdir -m 0755 "$media_root"
fi
assert_regular_ancestor_chain "$media_root/placeholder" "$dungeon_root"

supervisor_state="$run_root/.launcher-$run_name.json"
if [[ -e "$supervisor_state" || -L "$supervisor_state" ]]; then
  print -u2 "refusing to replace the U2r supervisor state: $supervisor_state"
  exit 1
fi
resume_arguments=()
if [[ -n "$resume_checkpoint" ]]; then
  resume_arguments=(--resume "$resume_checkpoint")
fi

print "Starting U2r segment $expected_segment_index from its frozen prospective boundary."
/usr/bin/caffeinate -ims >/dev/null 2>&1 &
caffeine_pid=$!
set +e
"$repository/.venv/bin/python" "$supervisor" \
  --state-path "$supervisor_state" \
  -- \
    "$repository/.venv/bin/python" -m "$trainer_module" \
    --parent "$parent" \
    --confirmation-report "$confirmation_report" \
    --qualification-report "$qualification_report" \
    --seed 20260749 \
    --additional-budget 360448 \
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
    --keep-rolling-exams 11 \
    --minimum-free-gib "$minimum_free_gib" \
    --storage-root "$volume" \
    --run-root "$run_root" \
    --run-name "$run_name" \
    --media-directory "$media_root" \
    "${resume_arguments[@]}" &
supervisor_pid=$!
set -e

status_ready=false
for _attempt in {1..3000}; do
  if [[ -f "$target/status.json" && ! -L "$target/status.json" ]]; then
    status_ready=true
    break
  fi
  if ! kill -0 "$supervisor_pid" 2>/dev/null; then
    break
  fi
  /bin/sleep 0.1
done
if [[ "$status_ready" == true ]]; then
  start_dashboard
fi

set +e
wait "$supervisor_pid"
trainer_status=$?
supervisor_pid=""
set -e
stop_caffeine

if [[ "$status_ready" != true || ! -f "$target/status.json" ]]; then
  print -u2 "U2r stopped before publishing a readable segment status"
  exit 1
fi
terminal_phase=$(
  "$repository/.venv/bin/python" - "$target/status.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    phase = json.load(stream).get("phase")
if phase not in {"eligible", "failed", "interrupted", "crashed"}:
    raise SystemExit(f"unexpected terminal U2r phase: {phase!r}")
print(phase)
PY
)

assert_source_unchanged
assert_free_space
terminal_evidence_verified=false
if [[ "$terminal_phase" == "eligible" || "$terminal_phase" == "failed" ]]; then
  "$repository/.venv/bin/python" - "$target" "$terminal_phase" <<'PY'
import sys
from pathlib import Path

from dungeon_apprentice.v02_u2r import verify_terminal_report

run_directory = Path(sys.argv[1])
phase = sys.argv[2]
report = verify_terminal_report(run_directory)
if report.get("verdict") != phase:
    raise SystemExit("verified U2r terminal report/status verdict mismatch")
if report.get("eligible_for_fresh_confirmation") is not (phase == "eligible"):
    raise SystemExit("verified U2r terminal eligibility binding mismatch")
PY
  terminal_evidence_verified=true
fi
if [[ "$terminal_phase" == "interrupted" ]]; then
  print -u2 "U2r segment was interrupted; its exact safe tip may be resumed with --resume."
  exit 130
fi
if [[ "$terminal_phase" == "crashed" || "$terminal_evidence_verified" != true ]]; then
  print -u2 "U2r segment stopped with an engineering failure."
  exit 1
fi
if [[ "$trainer_status" -ne 0 ]]; then
  print "Trainer exited after publishing verified terminal evidence; the terminal result remains valid."
fi

launcher_finalized=true
trap - EXIT INT TERM
print "U2r reached its exact terminal budget with verdict: $terminal_phase"
print "Read-only dashboard remains available: http://127.0.0.1:$dashboard_port/"
