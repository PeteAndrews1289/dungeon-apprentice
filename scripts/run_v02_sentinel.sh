#!/bin/zsh
set -euo pipefail

repository=${0:A:h:h}
cd "$repository"

exec .venv/bin/python -m dungeon_apprentice.v02_sentinel \
  --total-timesteps 524288 \
  --workers 4 \
  --size 9 \
  --seed 20260725 \
  --device cpu \
  --rollout-steps 512 \
  --batch-size 256 \
  --n-epochs 4 \
  --learning-rate 0.00025 \
  --gamma 0.995 \
  --gae-lambda 0.98 \
  --evaluation-every 32768 \
  --evaluation-seeds 80 \
  --checkpoint-every 32768 \
  --frame-every 2048 \
  --qualification-seeds 1000 \
  --minimum-free-gib 25 \
  --run-root "/Volumes/T7 Developer/DungeonApprentice/sentinels/v0.2-u0-20260722" \
  --run-name "v02-u0-seed-20260725" \
  --dashboard-host 127.0.0.1 \
  --dashboard-port 8781
