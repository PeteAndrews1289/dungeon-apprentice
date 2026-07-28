#!/bin/zsh
set -euo pipefail

repository=${0:A:h:h}
volume="/Volumes/T7 Developer"
output="$volume/DungeonApprentice/confirmations/v0.2-u1-20260723/report.json"
confirmation_directory=${output:h}
attempt_ledger="$confirmation_directory/attempt.json"
checksum_path="$output.sha256"
evaluator_log="$confirmation_directory/evaluator.log"
checkpoints=(
  "$volume/DungeonApprentice/u1-local-20260722/v02-u1-lead-seed-20260725/checkpoints/mastered-local-unlock.zip"
  "$volume/DungeonApprentice/u1-local-replication-20260722/v02-u1-replication-seed-20260729/checkpoints/mastered-local-unlock.zip"
  "$volume/DungeonApprentice/u1-local-replication-20260722/v02-u1-replication-seed-20260733/checkpoints/mastered-local-unlock.zip"
)

cd "$repository"

typeset -i attempt_started=0
typeset -i attempt_finalized=0
typeset -i evaluator_status=-1
typeset attempt_status="launcher_interrupted"
typeset report_sha256=""
typeset report_verdict=""
typeset report_present="false"

create_attempt_ledger() {
  local source_commit=$1
  .venv/bin/python -c '
import datetime
import json
import os
import sys

path, source_commit, output, checksum, evaluator_log, *checkpoints = sys.argv[1:]
now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
payload = {
    "schema_version": 1,
    "protocol": "dungeon-apprentice-v0.2-u1-confirmation-attempt",
    "status": "started",
    "started_at": now,
    "completed_at": None,
    "source_commit": source_commit,
    "source_dirty": False,
    "output": output,
    "checksum_path": checksum,
    "evaluator_log": evaluator_log,
    "checkpoints": checkpoints,
    "evaluator_exit_status": None,
    "launcher_exit_status": None,
    "report_present": False,
    "report_sha256": None,
    "report_verdict": None,
}
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
with os.fdopen(descriptor, "wb") as stream:
    stream.write(encoded)
    stream.flush()
    os.fsync(stream.fileno())
directory = os.open(os.path.dirname(path), os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
' \
  "$attempt_ledger" \
  "$source_commit" \
  "$output" \
  "$checksum_path" \
  "$evaluator_log" \
  "${checkpoints[@]}"
}

finalize_attempt_ledger() {
  local launcher_status=$1
  if .venv/bin/python -c '
import datetime
import json
import os
import sys
import uuid

(
    path,
    status,
    evaluator_status,
    launcher_status,
    report_present,
    report_sha256,
    report_verdict,
) = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    payload = json.load(stream)
payload.update(
    {
        "status": status,
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "evaluator_exit_status": (
            None if evaluator_status == "-1" else int(evaluator_status)
        ),
        "launcher_exit_status": int(launcher_status),
        "report_present": report_present == "true",
        "report_sha256": report_sha256 or None,
        "report_verdict": report_verdict or None,
    }
)
temporary = f"{path}.{uuid.uuid4().hex}.tmp"
try:
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(os.path.dirname(path), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
finally:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
' \
    "$attempt_ledger" \
    "$attempt_status" \
    "$evaluator_status" \
    "$launcher_status" \
    "$report_present" \
    "$report_sha256" \
    "$report_verdict"; then
    attempt_finalized=1
  else
    print -u2 "could not finalize confirmation attempt ledger: $attempt_ledger"
    return 1
  fi
}

TRAPEXIT() {
  local launcher_status=$?
  if (( attempt_started == 1 && attempt_finalized == 0 )); then
    finalize_attempt_ledger "$launcher_status" || true
  fi
}

if [[ ! -d "$volume" ]]; then
  print -u2 "T7 Developer is not mounted at $volume"
  exit 1
fi

if [[ ! -x .venv/bin/dungeon-confirm-v02-u1 ]]; then
  print -u2 "project environment does not contain the U1 confirmation command"
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  print -u2 "refusing to open the confirmation partition from a dirty working tree"
  exit 1
fi

if [[ -e "$output" || -e "$checksum_path" || -e "$evaluator_log" ]]; then
  print -u2 "refusing to overwrite an existing confirmation report, checksum, or log"
  exit 1
fi

if [[ -e "$attempt_ledger" ]]; then
  print -u2 "refusing to overwrite the existing confirmation attempt ledger: $attempt_ledger"
  exit 1
fi

for checkpoint in "${checkpoints[@]}"; do
  if [[ ! -f "$checkpoint" ]]; then
    print -u2 "selected U1 mastery checkpoint is missing: $checkpoint"
    exit 1
  fi
done

mkdir -p "$confirmation_directory"
source_commit=$(git rev-parse HEAD)
create_attempt_ledger "$source_commit"
attempt_started=1

if .venv/bin/dungeon-confirm-v02-u1 \
    --checkpoint "${checkpoints[1]}" \
    --checkpoint "${checkpoints[2]}" \
    --checkpoint "${checkpoints[3]}" \
    --output "$output" > >(/usr/bin/tee "$evaluator_log") 2>&1; then
  evaluator_status=0
else
  evaluator_status=$?
fi

typeset -i launcher_status=$evaluator_status
if [[ -f "$output" ]]; then
  report_present="true"
  if checksum_line=$(/usr/bin/shasum -a 256 "$output"); then
    report_sha256=${checksum_line%% *}
    if ! print -r -- "$checksum_line" > "$checksum_path"; then
      attempt_status="report_checksum_sidecar_failed"
      launcher_status=1
    fi
  else
    attempt_status="report_checksum_failed"
    launcher_status=1
  fi
  if report_verdict=$(
    .venv/bin/python -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream).get("verdict", ""))
' "$output"
  ); then
    :
  else
    report_verdict="unreadable"
    attempt_status="report_parse_failed"
    launcher_status=1
  fi
  if [[ "$attempt_status" == "launcher_interrupted" ]]; then
    attempt_status="completed_with_report"
  fi
else
  attempt_status="failed_without_report"
  (( launcher_status == 0 )) && launcher_status=1
fi

finalize_attempt_ledger "$launcher_status"

print "confirmation attempt ledger: $attempt_ledger"
print "confirmation evaluator log: $evaluator_log"
if [[ "$report_present" == "true" ]]; then
  print "confirmation report: $output"
  print "confirmation checksum: $checksum_path"
fi
exit "$launcher_status"
