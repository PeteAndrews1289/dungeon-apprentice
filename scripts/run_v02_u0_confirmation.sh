#!/bin/zsh
set -euo pipefail

repository=${0:A:h:h}
volume="/Volumes/T7 Developer"
output="$volume/DungeonApprentice/confirmations/v0.2-u0-20260722/report.json"

cd "$repository"

if [[ ! -d "$volume" ]]; then
  print -u2 "T7 Developer is not mounted at $volume"
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  print -u2 "project environment is missing: $repository/.venv/bin/python"
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  print -u2 "refusing to consume confirmation seeds from a dirty working tree"
  exit 1
fi

if [[ -e "$output" ]]; then
  print -u2 "refusing to overwrite confirmation report: $output"
  exit 1
fi

.venv/bin/python -m dungeon_apprentice.v02_confirm \
  --checkpoint "$volume/DungeonApprentice/sentinels/v0.2-u0-20260722/v02-u0-seed-20260725/checkpoints/mastered-visible-unlock.zip" \
  --checkpoint "$volume/DungeonApprentice/sentinels/v0.2-u0-replication-20260722/v02-u0-replication-seed-20260726/checkpoints/mastered-visible-unlock.zip" \
  --checkpoint "$volume/DungeonApprentice/sentinels/v0.2-u0-replication-20260722/v02-u0-replication-seed-20260727/checkpoints/mastered-visible-unlock.zip" \
  --output "$output"
