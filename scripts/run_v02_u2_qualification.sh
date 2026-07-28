#!/bin/zsh
set -euo pipefail

repository=${0:A:h:h}
volume="/Volumes/T7 Developer"
dungeon_root="$volume/DungeonApprentice"
qualification_parent="$dungeon_root/qualifications"
output_directory="$qualification_parent/v0.2-u2-20260723"
qualifier="$repository/.venv/bin/dungeon-qualify-v02-u2"
minimum_free_gib=25

cd "$repository"

if (( $# != 0 )); then
  print -u2 "the frozen U2 qualification launcher accepts no overrides"
  exit 2
fi

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
    print -u2 "U2 source commit changed after qualification preflight"
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
minimum = int(float(minimum_gib) * 1024**3)
if free < minimum:
    raise SystemExit(
        f"U2 qualification requires {minimum_gib} GiB free, "
        f"but only {free / 1024**3:.2f} GiB remains"
    )
PY
}

assert_no_active_neural_trainer() {
  local active
  active=$(
    /usr/bin/pgrep -f \
      '[d]ungeon-train|dungeon_apprentice\.(train|v02_sentinel|v02_u1|v02_u2)([[:space:]]|$)' \
      || true
  )
  if [[ -n "$active" ]]; then
    print -u2 "refusing qualification while neural training is active: ${active//$'\n'/, }"
    return 1
  fi
}

assert_actual_mounted_volume
assert_free_space
assert_no_active_neural_trainer

if [[ ! -x "$qualifier" || ! -x "$repository/.venv/bin/python" ]]; then
  print -u2 "project environment is missing the frozen U2 qualification command"
  exit 1
fi

if [[ ! -d "$dungeon_root" || -L "$dungeon_root" ]]; then
  print -u2 "Dungeon Apprentice storage root is missing or unsafe: $dungeon_root"
  exit 1
fi

if [[ ! -d "$qualification_parent" || -L "$qualification_parent" ]]; then
  print -u2 "qualification parent must already exist as a regular directory: $qualification_parent"
  exit 1
fi

qualification_already_exists=false
if [[ -L "$output_directory" ]]; then
  print -u2 "the one-shot U2 qualification target cannot be a symlink: $output_directory"
  exit 1
elif [[ -e "$output_directory" ]]; then
  if [[ ! -d "$output_directory" ]]; then
    print -u2 "the U2 qualification target is not a directory: $output_directory"
    exit 1
  fi
  qualification_already_exists=true
fi

source_commit=$(clean_source_commit) || {
  print -u2 "refusing to open sealed U2 seeds from dirty or uncommitted source"
  exit 1
}

acknowledgement=$(
  "$repository/.venv/bin/python" -c \
    'from dungeon_apprentice.u2_seed_guard import sealed_acknowledgement; print(sealed_acknowledgement())'
)
if [[ -z "$acknowledgement" ]]; then
  print -u2 "the frozen source did not provide its sealed-range acknowledgement"
  exit 1
fi

assert_source_unchanged
assert_free_space
assert_no_active_neural_trainer
if [[ "$qualification_already_exists" == false ]]; then
  launch_token=$(
    "$repository/.venv/bin/python" -c \
      'import secrets; print(secrets.token_hex(32))'
  )
  print "Opening the exact one-shot U2 sealed preflight from source commit $source_commit"
  set +e
  /usr/bin/caffeinate -ims "$qualifier" \
    --launch-token "$launch_token" \
    --acknowledge "$acknowledgement"
  qualifier_status=$?
  set -e
  unset launch_token
  if [[ "$qualifier_status" -ne 0 ]]; then
    print -u2 "the one-shot U2 qualification did not pass; training remains locked"
    exit "$qualifier_status"
  fi
else
  print "Qualification artifacts already exist; recovering only the external anchor"
fi

if ! assert_source_unchanged; then
  exit 1
fi

"$repository/.venv/bin/python" - "$repository" "$source_commit" <<'PY'
import sys
from pathlib import Path

from dungeon_apprentice.u2_qualification_anchor import (
    publish_external_anchor,
    qualification_evidence_for_anchor,
)

repository = Path(sys.argv[1])
source_commit = sys.argv[2]
evidence = qualification_evidence_for_anchor(
    repository,
    expected_source_commit=source_commit,
)
anchor = publish_external_anchor(
    repository,
    evidence,
    source_commit=source_commit,
)
print(f"U2 qualification externally anchored by {anchor.tag}")
print(f"U2 qualification report SHA-256: {anchor.report_sha256}")
print(f"U2 qualification attempt: {anchor.attempt_id}")
print(f"U2 qualification claim: {anchor.claim_id}")
PY

assert_source_unchanged
print "The U2 qualification passed and its fixed annotated tag is present on origin."
