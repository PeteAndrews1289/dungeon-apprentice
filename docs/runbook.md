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

## Start a declared run

```bash
.venv/bin/dungeon-train \
  --total-timesteps 1000000 \
  --workers 4 \
  --evaluation-every 50000 \
  --evaluation-seeds 40 \
  --run-name v0-seed-20260722
```

The command prints both the run directory and dashboard address. The dashboard normally uses
`http://127.0.0.1:8780/`; if that port is busy, the runner selects a free local port and prints it.
Closing the dashboard tab does not stop training. Stopping the Terminal process does.

## Artifacts

Each run directory is self-contained:

| Path | Meaning |
| --- | --- |
| `manifest.json` | Exact arguments, package versions, platform, and source revision |
| `qualification.json` | Generator solvability evidence recorded before learning |
| `status.json` | Atomic current state used by the dashboard |
| `evaluations.jsonl` | Append-only per-seed unseen-level results |
| `events.jsonl` | Promotions, holds, checkpoints, and warnings |
| `checkpoints/latest.zip` | Most recently published reloadable policy |
| `frames/latest.png` | Latest pixel view for the live dashboard |
| `crash.json` | Full diagnostic trace if the trainer exits unexpectedly |

Checkpoint files are written under a temporary name and then atomically renamed. A power loss may
lose the newest unpublished write, but it should not corrupt the previous `latest.zip`.

## Resume after interruption

Start a new documented run from the last checkpoint:

```bash
.venv/bin/dungeon-train \
  --resume runs/v0-seed-20260722/checkpoints/latest.zip \
  --total-timesteps 500000 \
  --run-name v0-seed-20260722-resumed
```

The new manifest records its parent checkpoint and restores the last unlocked curriculum tier. The
new directory preserves a clear experimental chain instead of silently rewriting the original run.

## Final evaluation

Validation exams are automatic. Do not use `--final` while selecting models. After training stops
and one checkpoint has been chosen without consulting final results:

```bash
.venv/bin/dungeon-evaluate runs/CHOSEN/checkpoints/latest.zip \
  --final --seeds 100 --output runs/CHOSEN/final-evaluation.json
```

Running the final suite more than once should be disclosed. If its results influence another round
of design, those seeds are no longer an untouched final test for the successor protocol.

