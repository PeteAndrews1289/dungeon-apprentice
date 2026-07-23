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

Attempt 1 stopped at qualification and is immutable. Historical record: after committing that
result and a prospective successor plan, confirmation v2 was launched once with:

```bash
scripts/run_v02_u1_confirmation_v2.sh
```

Do **not** run this command again. The successor verified attempt 1's exact report SHA-256, selected
cases in ascending seed order using only generator, oracle, uniqueness, and frozen reference
exclusions, and loaded no policy until all three accepted 200-case panels qualified. Navigate
accepted 200/200 examined candidates, U0 accepted 200 after 236 examined and 36 rejected, and U1
accepted 200 after 207 examined and seven rejected.

All three policies passed all three lessons and both panels: seed `20260725` scored
188/200 Navigate, 200/200 U0, and 188/200 U1; `20260729` scored 192/200, 200/200, and
192/200; `20260733` scored 188/200, 200/200, and 182/200. No update occurred. The report SHA-256 is
`6e577170050f6f14599b793a031776a19bf7c64eba0f243f457298da3193ae8f`.
Its ledger, log, report, and checksum live under
`/Volumes/T7 Developer/DungeonApprentice/confirmations/v0.2-u1-v2-20260723`. Exact rules are frozen
in [the v2 confirmation plan](v0.2-u1-confirmation-v2-plan.md), and the measured outcome is in
[the confirmation v2 result](results/v0.2-u1-confirmation-v2.md).

The next development boundary is [U2 Separated Unlock](protocol-v0.2-u2-separated-unlock.md).
Do not improvise a U2 command from the completed U1 launchers; use only the checkpoint lineage,
generator, curricula, gates, and launcher declared by that successor protocol.

Historical U1 recovery note: an interrupted child run resumes from its latest intact archive and
sidecar with the same launcher settings plus `--resume` and a new run name. The U1 runner interprets
524,288 as a cumulative child ceiling: it subtracts already trained child actions instead of
granting a fresh full budget. Parent lineage, optimizer updates, recovery state, transition counts,
scheduler random state, and the new segment seed are all checked before learning continues.

## Run U2 Separated Unlock

U2 has two accepted entry points and no hand-configurable training command. Do not adapt a U1
script, call the lower-level trainer directly, or generate a protected “preview.”

Before launch, the repository must be clean and committed, the exact source must already be pushed,
the T7 must be mounted at `/Volumes/T7 Developer`, the three frozen U1 parents and confirmation
record must verify, at least 25 GiB must remain free, no neural trainer may be active, and dashboard
port 8785 must be free. First run the fixed 64-transition engineering-only smoke on the disposable
5.2-million sandbox:

```bash
.venv/bin/dungeon-smoke-v02-u2 --run-name v0.2-u2-final-20260723
```

Its report must say `passed`, show a real optimizer and policy change, reload the same trained
state, list only engineering seed roles, and perform no evaluation, promotion, or capability claim.
Then run the one-shot qualification exactly once:

```bash
./scripts/run_v02_u2_qualification.sh
```

The launcher creates a fresh private 256-bit token, atomically claims the one canonical attempt,
and opens only seeds `5_210_000`–`5_211_999` plus the declared U2 validation reference. There is no
path, count, seed, or rerun override. A pass must solve all 2,000 cases, meet both diversity floors,
and have no exact U2-validation collision. The launcher then publishes the annotated tag
`u2-preflight-v0.2-u2-20260723` to the frozen GitHub origin. That tag binds the exact report bytes,
attempt, claim, generator profile, and source commit. If the network fails after a valid report is
written, rerunning this launcher performs anchor recovery only; it never reopens the sealed range.
A failed qualification permanently blocks this cohort.

With the same clean source commit still checked out, begin the three children:

```bash
./scripts/run_v02_u2_cohort.sh
```

The launcher re-verifies the remote tag and report, creates
`/Volumes/T7 Developer/DungeonApprentice/u2-separated-20260723`, starts the read-only dashboard at
`http://127.0.0.1:8785/`, and runs children `20260737`, `20260741`, and `20260745` sequentially.
Each inherits its exact confirmed U1 archive and optimizer, receives at most 1,048,576 new trained
actions, and runs regardless of an earlier valid child's learning outcome. Optional narrative
captures belong only in the separately measured
`/Volumes/T7 Developer/DungeonApprentice/u2-separated-media-20260723` directory; never place video
inside a scientific lineage.

If the Mac, app, or launcher is interrupted, do not delete or rename the cohort directory. Restore
the same repository commit and T7, ensure no trainer remains active, then run:

```bash
./scripts/run_v02_u2_cohort.sh --resume
```

Resume verifies every schema-4 archive, sidecar, digest, source, parent, qualification, curriculum,
optimizer counter, and action counter; selects the highest-action safe bundle; creates a new
immutable `-segment-NNN` directory; and spends only the unconsumed cumulative budget. Every
completed 2,048-action optimizer phase has a bounded `latest-safe` bundle, while genuinely partial
work is discarded and reported. Signals and unexpected launcher exits are durably reflected in the
cohort manifest rather than leaving a lineage indefinitely labeled `training`.

Preserve every terminal result. Select first-mastery checkpoints only by the frozen rule; do not use
the dashboard or a favorable-looking intermediate curve to reselect them. Only after all valid
children finish may a separate U2 confirmation protocol be authored. Confirmation must remain
no-update, collision-aware, and preregistered before opening a candidate.

The historical U2 launcher owned one read-only cohort dashboard at
`http://127.0.0.1:8785/`. Port 8785 being occupied was a launch failure, not permission to pick an
unrecorded fallback. The view read a small `cohort.json` and the three declared lineage
`status.json` files, showed pending/completed lineages as well as the active one, and could not
control training. Closing the browser had no effect on a trainer. The launcher used
`caffeinate -ims`, rejected another neural trainer, and left a durable terminal or crash record.

The historical U2 resource envelope on the audited M1 was:

- one four-worker CPU trainer at a time; never three concurrent children;
- approximately 8–10 hours for all three if they approached their full ceilings, based on measured
  U1 throughput plus U2's fourth exam;
- 2 GiB maximum per lineage and 6 GiB for all scientific run directories;
- five ordinary rolling checkpoints, with named decision artifacts retained separately;
- 10 GiB maximum for optional screen capture in a separate media directory;
- 16 GiB maximum planned addition and a 25 GiB free-space refusal reserve.

Those were ceilings, not targets or evidence. A storage stop was operational evidence, a passed
oracle qualification was engineering evidence, and a rising frozen exam curve was learning
evidence. Only the later no-update disjoint confirmation could establish confirmation evidence. At
the committed source-freeze entry, no protected U2 qualification, U2 validation,
candidate-confirmation, or final seed had been opened and no U2 child had trained. The completed
external ledgers, result document, and frozen dashboard artifacts now supersede that prospective
status.

## Run the bounded U2r stability successor

U2 confirmation is complete and remains `capability_failed`: two policies passed, while child
`20260745` missed the frozen U2 overall gate by one case. U2r does not retry that exam or replace
that verdict. It continues only the failed child's exact mastery archive and optimizer for the
360,448 actions left under its original 1,048,576-action ceiling.

After the clean implementation commit and annotated tag
`u2r-stability-v0.2-u2r-20260723` have been pushed, launch the sole canonical segment with:

```bash
./scripts/run_v02_u2r.sh
```

The launcher re-authenticates the parent, failed confirmation, qualification, static layout
exclusions, remote annotated tag, mounted T7, free-space reserve, clean source, and absence of
another neural trainer before it creates evidence. It writes the scientific lineage beneath
`/Volumes/T7 Developer/DungeonApprentice/u2r-stability-20260723`, keeps optional media beneath
`/Volumes/T7 Developer/DungeonApprentice/u2r-stability-media-20260723`, and serves the read-only
dashboard at `http://127.0.0.1:8786/`.

Do not choose a checkpoint or new name by hand after an interruption. If—and only if—the prior
segment durably published its fail-closed interruption record, restore the same source and T7,
verify no trainer remains, and run:

```bash
./scripts/run_v02_u2r.sh --resume
```

The resume planner authenticates the complete contiguous chain and chooses only its tip. Segment
zero is `v02-u2r-seed-20260745`; successors are exactly
`v02-u2r-seed-20260745-resume-N`. Each segment derives new algorithm and worker RNG streams by
adding `N × 100,000`, while retaining the same policy, optimizer, curriculum, completed exams, and
cumulative ceiling. The four partial episode identities captured at interruption are retained and
explicitly abandoned because active Gym and recurrent states are not serialized. A power loss that
prevents the interruption record from becoming durable is an operational failure, not permission
to guess at a recovery point.

All eleven 32,768-action windows run. Only the exact 1,048,576-child-action terminal artifact may
qualify, and only from the declared final two development exams. No launcher or trainer path opens
the fresh U2r confirmation reservation; a separately committed no-update evaluator is required
after a positive terminal stability report.

The seed ledger is a refusal list:

| Purpose | Allocation | Present status |
| --- | --- | --- |
| Ordinary policy training | `0`–`999_999` | Available only through the frozen trainer |
| Generator engineering sandbox | `5_200_000`–`5_200_999` | Engineering only; never evidence |
| One-shot sealed qualification | `5_210_000`–`5_211_999` | Opened only by the canonical claimed qualifier |
| Navigate retention | `10_000_000`–`10_000_079` | Existing frozen suite |
| U0 retention | `11_000_000`–`11_000_079` | Existing frozen suite |
| U1 retention | `11_100_000`–`11_100_079` | Existing frozen suite |
| U2 development validation | `11_200_000`–`11_200_079` | Opened only by qualification/training with bound access |
| Consumed U2 confirmation | `15_200_000`–`15_239_999` in four lesson-specific streams | Opened once; immutable and permanently unavailable to U2r |
| Fresh U2r confirmation reservation | `15_240_000`–`15_279_999` in four lesson-specific streams | Structurally sealed; no training access issuer exists |
| Complete-project final test | `20_000_000`–`20_299_999` | Untouched throughout U2 |

Numerical separation is not enough. The trainer must also reject exact layout hashes belonging to
qualification or any frozen validation suite. Never inspect the future confirmation streams to
debug training, and never substitute the final allocation for a missing confirmation plan.

## U2r artifacts

Each immutable U2r segment directory is self-contained, while a terminal report binds the complete
contiguous segment chain:

| Path | Meaning |
| --- | --- |
| `manifest.json` | Exact arguments, qualification provenance, package versions, platform, source revision, exclusions, anchor, and segment identity |
| `status.json` | Atomic current state used by the dashboard |
| `episodes.jsonl` | Append-only trainer-only outcomes, milestones, returns, and curiosity telemetry |
| `optimizer.jsonl` | Append-only post-update PPO metrics and trained-step counters |
| `evaluations.jsonl` | Unseen-level results, milestone rates, and exact checkpoint digests |
| `events.jsonl` | Promotions, holds, checkpoints, and warnings |
| `episode-starts.jsonl` | U2r episode identities written before the first action, including interruption/resume disposition |
| `checkpoints/initial.*` | Exact policy/optimizer state loaded at this segment's start; segment zero also binds the raw frozen-parent baseline |
| `checkpoints/latest-safe.*` | Most recent fully optimized intra-window recovery bundle, atomically replaced |
| `checkpoints/resume.*` | Exact post-exam recovery bundle, atomically replaced when another window remains |
| `checkpoints/rolling/exam-*.*` | All eleven immutable development exams, their sidecars, integrity records, and raw case evidence |
| `checkpoints/rolling/exam-*.cases.json` | U2r's immutable 320-case evidence behind each development aggregate |
| `checkpoints/terminal.*` | Exact full-budget terminal policy, sidecar, and integrity record; never resume-eligible |
| `interruption.json` | U2r's actual four active episodes and exact safe resume boundary, when durably interrupted |
| `interruption.integrity.json` | Digest binding for an interruption record |
| `report.json` | U2r's immutable terminal stability verdict, complete exam inventory, and lineage-history bindings |
| `report.integrity.json` | Digest binding for the terminal report |
| `frames/latest.png` | Latest pixel view for the live dashboard |
| `frames/exam-*.png` | Latest frozen frame for each declared lesson |
| `crash.json` | Full diagnostic trace if the trainer exits unexpectedly |

Each named policy archive has a matching `.json` sidecar. The sidecar records effective environment,
PPO, and evaluation settings; collected/trained/update counters; curriculum tier; segment seed stream;
protocol; parentage; and the archive's SHA-256 digest. Each file is written atomically and the digest
detects a missing or torn pair. A power loss may make the newest alias unusable, but resume rejects it
rather than guessing; the most recent intact named checkpoint remains the recovery point.

U2r retains exactly eleven rolling exam bundles because all eleven are part of its terminal
selection audit. Replaceable `latest-safe` and `resume` recovery aliases are separate from those
immutable exams. The 2 GiB lineage cap, 6 GiB cohort cap, optional 10 GiB media cap, and 25 GiB
free-space reserve are refusal thresholds rather than estimates of expected consumption.

## Resume a legacy v0/v0.1 run after interruption

This older generic procedure does not apply to U2r. For U2r, use only the authenticated
`./scripts/run_v02_u2r.sh --resume` path described above.

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
