#!/bin/zsh
set -eu

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_ROOT" >&2
  exit 2
fi

script_directory=${0:A:h}
repository=${script_directory:h}
run_root=$1
seeds=(20260722 20260723 20260724)

mkdir -p "$run_root"
cd "$repository"

for seed in $seeds; do
  .venv/bin/dungeon-train \
    --total-timesteps 262144 \
    --workers 4 \
    --size 9 \
    --rollout-steps 512 \
    --batch-size 256 \
    --n-epochs 4 \
    --learning-rate 0.00025 \
    --gamma 0.995 \
    --gae-lambda 0.98 \
    --curiosity-scale 0.002 \
    --evaluation-every 32768 \
    --evaluation-seeds 40 \
    --checkpoint-every 32768 \
    --keep-checkpoints 3 \
    --minimum-free-gib 25 \
    --frame-every 2048 \
    --qualification-seeds 100 \
    --seed "$seed" \
    --device cpu \
    --dashboard-port 8780 \
    --run-root "$run_root" \
    --run-name "navigate-canary-seed-$seed"
done
