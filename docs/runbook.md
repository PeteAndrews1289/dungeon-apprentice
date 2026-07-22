# Operations runbook

## First setup

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,train]"
.venv/bin/dungeon-qualify --seeds 100
```

Qualification generates and solves 100 levels in every tier. Training refuses to start if its own
qualification pass fails.

The project pins the audited simulation and training stack in `pyproject.toml`. It is tested on
macOS/Apple Silicon and Linux/CPU without platform-specific code. After an upgrade, rerun every local
check and begin a new protocol if behavior, numerics, or learned results could change.

## Preflight the external run volume

Long runs belong on the external T7 rather than the internal system disk. Connect it and verify its
mount and free space:

```bash
test -d "/Volumes/T7 Developer"
df -h "/Volumes/T7 Developer"
mkdir -p "/Volumes/T7 Developer/DungeonApprentice/runs"
```

Do not substitute a nearly full volume. The trainer also checks its configured free-space reserve
before starting and before checkpoint publication.

## Start a declared run

The first v0.1 capability run should begin from random parameters. Its intervals align with complete
four-worker rollouts (`4 × 512 = 2,048` steps), so labels never imply a partial PPO update:

```bash
caffeinate -ims .venv/bin/dungeon-train \
  --total-timesteps 1048576 \
  --workers 4 \
  --rollout-steps 512 \
  --batch-size 256 \
  --n-epochs 4 \
  --learning-rate 0.00025 \
  --gamma 0.995 \
  --gae-lambda 0.98 \
  --curiosity-scale 0.002 \
  --evaluation-every 65536 \
  --evaluation-seeds 40 \
  --checkpoint-every 65536 \
  --keep-checkpoints 5 \
  --minimum-free-gib 25 \
  --frame-every 2048 \
  --qualification-seeds 100 \
  --seed 20260722 \
  --device cpu \
  --run-root "/Volumes/T7 Developer/DungeonApprentice/runs" \
  --run-name v0.1-seed-20260722
```

The command prints both the run directory and dashboard address. The dashboard normally uses
`http://127.0.0.1:8780/`; if that port is busy, the runner selects a free local port and prints it.
Closing the dashboard tab does not stop training. Stopping the Terminal process does.

On macOS, `caffeinate -ims` prevents idle system and disk sleep while that command is alive and the
Mac is connected to power. It intentionally omits `-d`, so the display may sleep or be switched off.
Leave the Mac plugged in, keep the Terminal process open, and confirm the dashboard heartbeat once
before walking away. The learner itself is local and consumes no GPT/Codex allowance.

For a headless smoke test or a machine where no dashboard is wanted, add `--no-dashboard`.

## Artifacts

Each run directory is self-contained:

| Path | Meaning |
| --- | --- |
| `manifest.json` | Exact arguments, package versions, platform, and source revision |
| `qualification.json` | Generator solvability evidence recorded before learning |
| `status.json` | Atomic current state used by the dashboard |
| `episodes.jsonl` | Append-only trainer-only outcomes, milestones, returns, and curiosity telemetry |
| `optimizer.jsonl` | Append-only post-update PPO metrics and trained-step counters |
| `evaluations.jsonl` | Unseen-level results, milestone rates, and exact checkpoint digests |
| `events.jsonl` | Promotions, holds, checkpoints, and warnings |
| `checkpoints/latest.zip` | Most recently published reloadable policy |
| `checkpoints/latest.json` | Configuration, progress, parentage, and digest paired with `latest.zip` |
| `frames/latest.png` | Latest pixel view for the live dashboard |
| `crash.json` | Full diagnostic trace if the trainer exits unexpectedly |

Each named policy archive has a matching `.json` sidecar. The sidecar records effective environment,
PPO, and evaluation settings; collected/trained/update counters; curriculum tier; segment seed stream;
protocol; parentage; and the archive's SHA-256 digest. Each file is written atomically and the digest
detects a missing or torn pair. A power loss may make the newest alias unusable, but resume rejects it
rather than guessing; the most recent intact named checkpoint remains the recovery point.

`--keep-checkpoints 5` bounds ordinary rolling snapshots. `latest`, initial, promotion, mastery, and
terminal artifacts may be retained separately because they explain a decision. The 25 GiB reserve
is a refusal threshold, not an estimate of what one run will consume.

## Resume after interruption

Resume from an archive only when its matching state sidecar is present and passes digest/protocol
validation. A resume creates a new child run rather than rewriting the parent:

```bash
caffeinate -ims .venv/bin/dungeon-train \
  --resume "/Volumes/T7 Developer/DungeonApprentice/runs/v0.1-seed-20260722/checkpoints/latest.zip" \
  --total-timesteps 1048576 \
  --workers 4 \
  --size 9 \
  --rollout-steps 512 \
  --batch-size 256 \
  --n-epochs 4 \
  --learning-rate 0.00025 \
  --gamma 0.995 \
  --gae-lambda 0.98 \
  --curiosity-scale 0.002 \
  --evaluation-every 65536 \
  --evaluation-seeds 40 \
  --checkpoint-every 65536 \
  --keep-checkpoints 5 \
  --frame-every 2048 \
  --qualification-seeds 100 \
  --seed 20260722 \
  --device cpu \
  --run-root "/Volumes/T7 Developer/DungeonApprentice/runs" \
  --run-name v0.1-seed-20260722-segment-02 \
  --minimum-free-gib 25
```

On resume, `--total-timesteps` is the **additional budget for this segment**, not a replacement
lifetime target. The child preserves lifetime collected/trained counters while measuring its own
elapsed time and throughput. Structural PPO settings, the curriculum tier, and the evaluation gate
are checked against the checkpoint bundle; incompatible or weakened settings fail rather than
silently relabeling the restored model. Scheduling and retention flags are repeated explicitly above
as part of the declared experiment. The new manifest records its parent archive, digest, and distinct
segment RNG stream.

Stable Baselines rounds collection to complete vector rollouts. With four workers and 512 rollout
steps, segment totals therefore advance in multiples of 2,048. Use aligned budgets as above when an
exact boundary matters.

Pressing Control-C or a crash during collection does not serialize partially trained experience. The
status records the discarded partial-step count and points to the last fully trained safe checkpoint.
Resume from that checkpoint/sidecar pair; do not copy or rename an incomplete temporary artifact.

## Final evaluation

Validation exams are automatic. Do not use `--final` while selecting models. After training stops
and one checkpoint has been chosen without consulting final results:

```bash
.venv/bin/dungeon-evaluate \
  "/Volumes/T7 Developer/DungeonApprentice/runs/CHOSEN/checkpoints/latest.zip" \
  --final \
  --seeds 100 \
  --output "/Volumes/T7 Developer/DungeonApprentice/runs/CHOSEN/final-evaluation.json"
```

Running the final suite more than once should be disclosed. If its results influence another round
of design, those seeds are no longer an untouched final test for the successor protocol.

The evaluator derives dungeon size from the checkpoint sidecar. A deliberate cross-size transfer
test must name both `--size` and `--allow-size-override`; its report records training and evaluation
sizes separately.
