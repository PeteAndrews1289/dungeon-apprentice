#!/bin/zsh
set -euo pipefail

repository=${0:A:h:h}
volume="/Volumes/T7 Developer"
run_root="$volume/DungeonApprentice/u1-local-replication-20260722"
confirmation="$volume/DungeonApprentice/confirmations/v0.2-u0-20260722/report.json"
dashboard_port=8784
seeds=(20260729 20260733)
parents=(
  "$volume/DungeonApprentice/sentinels/v0.2-u0-replication-20260722/v02-u0-replication-seed-20260726/checkpoints/mastered-visible-unlock.zip"
  "$volume/DungeonApprentice/sentinels/v0.2-u0-replication-20260722/v02-u0-replication-seed-20260727/checkpoints/mastered-visible-unlock.zip"
)

cd "$repository"

if [[ ! -d "$volume" ]]; then
  print -u2 "T7 Developer is not mounted at $volume"
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  print -u2 "project environment is missing: $repository/.venv/bin/python"
  exit 1
fi

if [[ ! -f "$confirmation" ]]; then
  print -u2 "frozen U0 confirmation report is missing: $confirmation"
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  print -u2 "refusing to launch U1 replications from a dirty working tree"
  exit 1
fi

if /usr/sbin/lsof -nP -iTCP:"$dashboard_port" -sTCP:LISTEN >/dev/null 2>&1; then
  print -u2 "dashboard port $dashboard_port is already in use"
  exit 1
fi

for (( index = 1; index <= ${#seeds}; index++ )); do
  seed=${seeds[$index]}
  parent=${parents[$index]}
  target="$run_root/v02-u1-replication-seed-$seed"
  if [[ ! -f "$parent" ]]; then
    print -u2 "frozen U0 parent is missing: $parent"
    exit 1
  fi
  if [[ -e "$target" ]]; then
    print -u2 "fresh-run target already exists: $target"
    exit 1
  fi
done

source_commit=$(git rev-parse HEAD)
mkdir -p "$run_root"

for (( index = 1; index <= ${#seeds}; index++ )); do
  seed=${seeds[$index]}
  parent=${parents[$index]}

  if [[ "$(git rev-parse HEAD)" != "$source_commit" || -n "$(git status --porcelain)" ]]; then
    print -u2 "source changed between U1 replications; refusing to mix protocols"
    exit 1
  fi

  print "starting U1 replication child $seed on http://127.0.0.1:$dashboard_port/"
  /usr/bin/caffeinate -ims .venv/bin/dungeon-train-v02-u1 \
    --parent "$parent" \
    --confirmation-report "$confirmation" \
    --child-budget 524288 \
    --workers 4 \
    --size 9 \
    --seed "$seed" \
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
    --keep-checkpoints 5 \
    --frame-every 2048 \
    --qualification-seeds 1000 \
    --minimum-free-gib 25 \
    --run-root "$run_root" \
    --run-name "v02-u1-replication-seed-$seed" \
    --dashboard-host 127.0.0.1 \
    --dashboard-port "$dashboard_port"
done

final_run="$run_root/v02-u1-replication-seed-${seeds[-1]}"
print "both U1 replications completed; keeping the final dashboard available"
exec .venv/bin/dungeon-dashboard "$final_run" --host 127.0.0.1 --port "$dashboard_port"
