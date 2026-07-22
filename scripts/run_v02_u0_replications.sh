#!/bin/zsh
set -euo pipefail

repository=${0:A:h:h}
volume="/Volumes/T7 Developer"
run_root="$volume/DungeonApprentice/sentinels/v0.2-u0-replication-20260722"
seeds=(20260726 20260727)
ports=(8782 8783)

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
  print -u2 "refusing to launch replications from a dirty working tree"
  exit 1
fi

source_commit=$(git rev-parse HEAD)

for (( index = 1; index <= ${#seeds}; index++ )); do
  seed=${seeds[$index]}
  port=${ports[$index]}
  target="$run_root/v02-u0-replication-seed-$seed"

  if [[ -e "$target" ]]; then
    print -u2 "fresh-run target already exists: $target"
    exit 1
  fi

  if /usr/sbin/lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    print -u2 "dashboard port $port is already in use"
    exit 1
  fi
done

mkdir -p "$run_root"

for (( index = 1; index <= ${#seeds}; index++ )); do
  seed=${seeds[$index]}
  port=${ports[$index]}

  if [[ "$(git rev-parse HEAD)" != "$source_commit" || -n "$(git status --porcelain)" ]]; then
    print -u2 "source changed between replications; refusing to mix protocols"
    exit 1
  fi

  print "starting v0.2 U0 replication seed $seed on dashboard port $port"
  /usr/bin/caffeinate -ims .venv/bin/python -m dungeon_apprentice.v02_sentinel \
    --total-timesteps 524288 \
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
    --frame-every 2048 \
    --qualification-seeds 1000 \
    --minimum-free-gib 25 \
    --run-root "$run_root" \
    --run-name "v02-u0-replication-seed-$seed" \
    --dashboard-host 127.0.0.1 \
    --dashboard-port "$port"
done

print "both v0.2 U0 replications completed"
