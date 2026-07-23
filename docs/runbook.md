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

The preregistered three-seed Navigate canary suite can be launched sequentially with:

```bash
caffeinate -ims scripts/run_navigate_canaries.sh \
  "/Volumes/T7 Developer/DungeonApprentice/canaries/v0.1-navigate-20260722"
```

The same dashboard address is reused as each seed hands off to the next. Do not run another neural
trainer beside this suite on the audited 8 GB Mac.

## Start the U1 Local Unlock child

U1 is a warm-start development child, not a random initialization. Its launcher verifies the exact
confirmed parent archive and raw U0 confirmation report before doing any work:

```bash
caffeinate -ims scripts/run_v02_u1_lead.sh
```

The command uses the T7 path
`/Volumes/T7 Developer/DungeonApprentice/u1-local-20260722/v02-u1-lead-seed-20260725`,
serves the dashboard at `http://127.0.0.1:8784/`, and runs at most 524,288 new actions. The dashboard
separates those new child actions from 491,520 inherited parent actions and labels the parent digest.
Its initial three-lesson exam is diagnostic and cannot satisfy a gate.

Do not substitute a later-looking U0 checkpoint, another seed, a weights-only export, or a different
confirmation report. Do not run U1 beside another trainer on the 8 GB M1. The full frozen contract is
in [the U1 development protocol](protocol-v0.2-u1-development.md).

After the positive lead result, launch the two sequential preregistered U1 replications with:

```bash
scripts/run_v02_u1_replications.sh
```

The first child uses confirmed U0 parent `20260726` with fresh child stream `20260729`; the second
uses parent `20260727` with stream `20260733`. Both reuse `http://127.0.0.1:8784/`, and the launcher
keeps the final dashboard available after training. Exact lineages and the all-two decision rule are
in [the U1 replication plan](v0.2-u1-replication-plan.md).

## Confirm Local Unlock on disjoint cases

Historical record: after the lead and both replications mastered, attempt 1 was launched once from
a clean tree with:

```bash
scripts/run_v02_u1_confirmation.sh
```

Do **not** run that command again. Its launcher selected the three exact first-mastery archives,
refused an existing output, qualified all three 200-case seed blocks before scoring, and wrote the
raw report beneath
`/Volumes/T7 Developer/DungeonApprentice/confirmations/v0.2-u1-20260723`. It never trains a policy.
It creates `attempt.json` and `evaluator.log` before qualification, then writes a checksum sidecar
for any completed report—including a valid negative result. A preexisting ledger, report, log, or
checksum stops the launcher instead of silently beginning a second attempt. Do not open, substitute,
or preview the confirmation ranges outside this frozen command. The exact gates and all-three
verdict are in [the U1 confirmation plan](v0.2-u1-confirmation-plan.md).

Attempt 1 stopped at qualification and is immutable. After its result and the successor plan are
committed from a clean tree, launch the collision-safe successor once:

```bash
scripts/run_v02_u1_confirmation_v2.sh
```

The successor first verifies attempt 1's exact report SHA-256, then opens fresh candidate streams.
It selects cases in ascending seed order using only generator, oracle, uniqueness, and frozen
reference exclusions; no policy is loaded until all three accepted 200-case panels qualify. Its
separate ledger, log, report, and checksum live under
`/Volumes/T7 Developer/DungeonApprentice/confirmations/v0.2-u1-v2-20260723`. Exact rules are frozen
in [the v2 confirmation plan](v0.2-u1-confirmation-v2-plan.md).

If interrupted, resume from the latest intact child archive and sidecar with the same launcher
settings plus `--resume` and a new run name. The U1 runner interprets 524,288 as a cumulative child
ceiling: it subtracts already trained child actions instead of granting a fresh full budget. Parent
lineage, optimizer updates, recovery state, transition counts, scheduler random state, and the new
segment seed are all checked before learning continues.

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
| `frames/exam-*.png` | Latest frozen frame for each declared lesson |
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
