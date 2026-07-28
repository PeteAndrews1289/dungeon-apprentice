#!/bin/sh
set -eu

RUN_ROOT="/Volumes/T7 Developer/DungeonApprentice/u1-local-20260722"
PARENT="/Volumes/T7 Developer/DungeonApprentice/sentinels/v0.2-u0-20260722/v02-u0-seed-20260725/checkpoints/mastered-visible-unlock.zip"
CONFIRMATION="/Volumes/T7 Developer/DungeonApprentice/confirmations/v0.2-u0-20260722/report.json"

exec .venv/bin/dungeon-train-v02-u1 \
  --parent "$PARENT" \
  --confirmation-report "$CONFIRMATION" \
  --run-root "$RUN_ROOT" \
  --run-name "v02-u1-lead-seed-20260725" \
  --dashboard-port 8784 \
  "$@"
