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

### Do not resume launch attempt 1

The original tag `u2r-stability-v0.2-u2r-20260723` and root
`/Volumes/T7 Developer/DungeonApprentice/u2r-stability-20260723` identify a completed operational
failure. At `2026-07-23T15:38:10+00:00`, its initial vector reset stopped after one zero-step U1
episode start and 128 excluded U0 proposals. It took zero policy actions and made zero optimizer
updates. It has no safe checkpoint, interruption record, or resume-eligible state.

Do not delete, move, rename, append to, or pass that root to `--resume`. Its exact file inventory
and checksums are in
[the launch-attempt result](results/v0.2-u2r-launch-attempt-1.md).

### r1 terminal boundary

r1 is a distinct prospective release:

| Field | Required identity |
| --- | --- |
| Protocol | `dungeon-apprentice-v0.2-u2r-stability-r1` |
| Annotated tag | `u2r-stability-v0.2-u2r-r1-20260723` |
| Run root | `/Volumes/T7 Developer/DungeonApprentice/u2r-stability-r1-20260723` |
| Media root | `/Volumes/T7 Developer/DungeonApprentice/u2r-stability-r1-media-20260723` |
| Initial segment | `v02-u2r-r1-seed-20260745` |
| Dashboard | `http://127.0.0.1:8786/` |

The launcher did refuse action until the clean r1 source, amended protocol digest, external
annotated tag, four immutable attempt-1 digests, exact parent, complete historical inventory,
lesson-specific guard identities, mounted T7, and real four-worker reset preflight all qualified.
It then completed all eleven 32,768-action windows and selected only the exact
1,048,576-child-action terminal artifact.

That artifact retained 76/80 U2 capability but failed the frozen terminal interaction-stability
limit. The authenticated result is
[U2r-r1 Stability Remediation: Terminal Result](results/v0.2-u2r-r1-stability.md). The fresh U2r
confirmation reservation never opened.

Both U2r roots and tags are now terminal evidence. Do **not** run either the launch or `--resume`
form of `scripts/run_v02_u2r.sh`, choose the favorable penultimate checkpoint, rename a segment, or
append another continuation. The resume description formerly in this runbook applied only while r1
was active; terminal closeout permanently ended that authority.

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

Numerical separation is not enough. The r1 trainer must apply the lesson-specific guard bound by
its tag: broad historical novelty for Navigate, U1, and U2; U0 development-only refusal with all
other U0 history retained diagnostically. Never inspect the future confirmation streams to debug
training, and never substitute the final allocation for a missing confirmation plan.

## U2r artifacts

The original failed root contains only its launcher state, manifest, crash report, and one-line
episode-start ledger. The table below describes artifacts expected only after qualified r1 training
begins. Each immutable r1 segment directory is self-contained, while a terminal report binds the
complete contiguous segment chain:

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

## Run the U2-S matched stability ablation

U2-S is a four-arm development mechanism study, not another continuation and not a confirmation.
Its complete scientific contract is
[Protocol v0.2 U2-S](protocol-v0.2-u2s-stability-ablation.md).

> **Terminal historical procedure:** r1 completed all four arms and selected no eligible
> configuration. The cohort is immutable `ablation_failed` evidence. Do not run its qualification
> or launcher again, do not resume an arm, and do not reuse any U2-S checkpoint. The commands below
> document the already consumed procedure; they are not current operating authority.

| Field | Fixed identity |
| --- | --- |
| Protocol | `dungeon-apprentice-v0.2-u2s-stability-ablation` |
| Annotated tag | `u2s-stability-ablation-v0.2-u2s-r1-20260723` |
| Qualification | `/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.2-u2s-r1-20260723/report.json` |
| Cohort root | `/Volumes/T7 Developer/DungeonApprentice/u2s-ablation-r1-20260723` |
| Media root | `/Volumes/T7 Developer/DungeonApprentice/u2s-ablation-r1-media-20260723` |
| Dashboard | `http://127.0.0.1:8787/` |
| Parent | Confirmed U1 child `20260733` |
| Sequential arms | `control`, `conservative`, `no-effect`, `combined` |

The sole safe order is:

1. freeze the complete implementation and protocol in one clean commit;
2. push that commit and its exact one-line annotated tag;
3. run the claim-bearing qualification, which includes the full tests, mechanical oracle, and a
   real four-arm disposable rollout/update smoke;
4. verify the qualification report and checksum while both canonical run roots are still absent;
5. invoke the fixed launcher once; and
6. leave the repository, tag, qualification, parent, and protected historical evidence unchanged
   until the cohort becomes terminal.

The qualification command is:

```bash
.venv/bin/python -m dungeon_apprentice.v02_u2s_qualify
```

It fails closed unless the remote annotated tag points to clean `HEAD`, predecessor evidence and
the confirmed U1 policy/optimizer retain their frozen hashes, all four lesson guards reconstruct
exactly, confirmation/final partitions stay unavailable, the T7 is a distinct mounted device with
at least 25 GiB free, and the canonical qualification/cohort/media roots do not already exist. Its
disposable smoke updates temporary policy copies only; it cannot create a capability claim or a
promotable checkpoint.

After qualification succeeds, the only launcher command is:

```bash
./scripts/run_v02_u2s_ablation.sh
```

The launcher accepts no arm, seed, checkpoint, run-name, budget, or resume override. It creates the
cohort contract, starts the read-only dashboard, prevents sleep, and runs the four cells
sequentially. Each cell reloads the exact confirmed U1 policy and optimizer, proves the same
post-reset pre-action RNG identity, trains exactly 1,048,576 actions, and retains all 32 fixed
post-update exams.

Attempt 0 at `/Volumes/T7 Developer/DungeonApprentice/u2s-ablation-20260723` is immutable failed
infrastructure evidence. Its control arm stopped before status publication or action one because
the per-arm media directory had not been created. It must never be resumed or reused. The r1
launcher creates and validates that exact directory after the manifest opens an arm and before the
trainer process starts. See the
[authenticated incident narrative](results/v0.2-u2s-launch-attempt-0.md).

### U2-S interruption rule

U2-S has no safe resume command. If a trainer, launcher, machine, source tree, or storage boundary
interrupts any cell, the entire root closes as `operationally_incomplete`. Do not continue the
current cell and do not reuse already completed cells. A replacement requires a new prospective
source commit, annotated tag, protocol-attempt identity, and root, followed by all four fresh cells
from the confirmed U1 parent.

This strict rule exists because resetting only one arm would give it a different environment,
recurrent, scheduler, and random trajectory than the other factorial cells. A checkpoint may be
useful forensic evidence without being a valid continuation point.

### U2-S live and terminal evidence

The dashboard reads only bounded public evidence. During training it shows each arm's action
counter, update count, completed exam curve, U2 score, ineffective-interaction mean and tails,
generic repeated-interaction run, and visible no-effect streak. It never controls training.

Terminal closeout succeeds only after all four trainers exit normally and the launcher:

- deep-verifies every arm report, all 128 exam bundles, all 40,960 case records, terminal model and
  optimizer state, reward/optimizer/episode ledgers, and source/qualification bindings;
- proves all four initial RNG identities are identical;
- recomputes the fixed final-three-exam grades and simplest-intervention priority;
- publishes the conservative main, no-effect main, and interaction contrasts;
- records storage caps and zero trainer/supervisor/caffeinate orphans, with only the dashboard
  explicitly left running; and
- seals the cohort report and its integrity record before the dashboard may display a selected
  mechanism.

The ablation selects only a configuration. No arm checkpoint may be copied into U3, confirmation,
or a successor. A positive result permits writing a separate multi-lineage protocol that starts
fresh from confirmed U1 parents; it does not itself confirm U2.

The actual all-arm result was negative, so that conditional successor did not open. U3 remains
closed. The next prospective boundary is the
[v0.3 matched action-effect architecture study](protocol-v0.3-action-effect-architecture.md).

## Prepare the implemented v0.3 action-effect architecture study

v0.3 is not a continuation of U2-S and does not reuse an ablation checkpoint. It is a two-arm,
matched architecture study from the exact confirmed U1 child `20260733`:

| Arm | Observation context | Role |
| --- | --- | --- |
| `sham` | Same Dict policy, new pathway, and comparison work; context always zero | Calibration only |
| `action-effect` | Previous primitive action plus whether consecutive visible RGB bytes changed | Only selectable candidate |

Both twins inherit the U1 policy and Adam state by verified parameter name, add only one
zero-initialized `512 × 9` context weight, and use the original U2 reward, PPO, curriculum, full
1,048,576-action budget, 32-exam schedule, and terminal U2-S gate. They share algorithm seed
`20260757` and worker streams `20260757`–`20260760`. Their complete first 2,048-transition
pre-update rollout must be identical; the first optimizer update is the earliest valid divergence.

The Stage-A implementation is complete, but implementation alone is still **unfrozen and
unqualified**. The operational identities are assigned; at this preregistration checkpoint they
have not been created:

| Boundary | Assigned identity |
| --- | --- |
| Annotated source-preregistration tag | `action-effect-architecture-v0.3-stage-a-20260724` |
| Qualification root | `/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.3-action-effect-stage-a-20260724` |
| Cohort root | `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-20260724` |
| Media root | `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-media-20260724` |
| Dashboard | `http://127.0.0.1:8788/` |
| Sole launcher | `scripts/run_v03_action_effect_stage_a.sh` |

No canonical tag, qualification claim/report, cohort, media root, policy action, or result exists at
this source entry. Do not improvise a root, adapt a U2-S script, or call the lower-level trainer
directly. Scientific training remains blocked until one clean release:

1. freezes and pushes the exact source and protocol;
2. publishes the assigned annotated tag and its one-line source-preregistration payload;
3. authenticates the confirmed U1 parent and every immutable predecessor result;
4. proves the Dict observation and exact named tensor/Adam transplant;
5. proves bit-exact zero-context equivalence and matched first-rollout identity in a disposable
   four-worker smoke;
6. verifies unchanged reward/PPO/curriculum and the static layout guards;
7. proves zero confirmation/final seed access, a mounted T7 with the declared reserve, and no
   competing trainer; and
8. emits one claim-bearing qualification report bound to that exact tag object before action one.

The old sealed U2 qualification range may not be reopened. The v0.3 engineering smoke must use only
disposable engineering layouts, leave canonical and predecessor evidence unchanged, and destroy
its updated policy copies.

Stage A is non-resumable. An interruption closes the entire matched root as
`operationally_incomplete`; neither twin may continue or be reused. A valid full-budget result
selects the architecture definition only when the `action-effect` arm passes all three fixed
terminal exams. It never selects a Stage A checkpoint and never opens U3.

If the candidate passes, the only next action is to author a separate Stage B protocol for three
fresh lineages from confirmed U1 children `20260725`, `20260729`, and `20260733`, with fresh random
streams and no Stage A state. Even a successful Stage B would still require a separate,
prospectively frozen no-update confirmation before U3 could open.

### Release, qualify, and launch v0.3 Stage A

The release chain is intentionally non-circular:

1. the published annotated tag preregisters the exact source commit, protocol digest, parent,
   predecessor verdict, guard/sampler/architecture contracts, protected partitions, canonical
   roots, dashboard, storage caps, non-resume rule, CPU-only training/smoke devices, and runtime
   snapshot (Python, platform, machine, Gymnasium, MiniGrid, NumPy, sb3-contrib,
   Stable-Baselines3, and Torch);
2. the durable qualification claim and report then bind that already-published tag object and prove
   the real four-worker matched smoke; and
3. the cohort contract and manifest bind the exact qualification-report SHA-256 before either
   trainer may act.

Run the ordinary repository checks while the tree is still a working candidate:

```bash
.venv/bin/ruff check .
.venv/bin/pytest
.venv/bin/dungeon-qualify --seeds 100
git diff --check
```

Then review the complete diff, commit it, push that exact commit to the approved `origin`, and
confirm the repository is clean. Do not substitute a commit identifier in this document; the
qualifier derives and verifies the eventual release commit itself.

Generate the canonical one-line tag payload only from that clean commit, create the annotated tag
with that exact line as its complete message, and publish it:

```bash
V03_TAG_PAYLOAD="$(
  .venv/bin/python -m dungeon_apprentice.v03_action_effect_qualify \
    --tag-payload-only
)"
git tag -a action-effect-architecture-v0.3-stage-a-20260724 \
  -m "$V03_TAG_PAYLOAD"
git push origin HEAD
git push origin refs/tags/action-effect-architecture-v0.3-stage-a-20260724
```

Do not hand-edit the payload. The tag preregisters inputs; it does **not** contain a qualification
report digest that can only exist later.

With the T7 mounted, at least 25 GiB free, no competing trainer, all three assigned roots absent,
and the repository still clean at the tagged commit, create the one-shot qualification:

```bash
.venv/bin/python -m dungeon_apprentice.v03_action_effect_qualify
```

The qualifier creates the durable claim first, runs the disposable real four-worker rollout/update
smoke, destroys its updated policy copies, and writes `report.json` plus its SHA-256 sidecar in the
assigned qualification root. Failure or partial output permanently consumes that attempt identity;
do not delete the claim or retry it under the same tag and roots.

Only after the qualification report verifies may the fixed launcher be invoked once:

```bash
./scripts/run_v03_action_effect_stage_a.sh
```

The launcher owns cohort/media creation, contract sealing, fixed arm order (`sham`, then
`action-effect`), trainer supervision, sleep prevention, the read-only dashboard at
`http://127.0.0.1:8788/`, and terminal closeout. After both trainers exit it stops sleep prevention,
asks the manifest to gather and seal a command-redacted process inventory, and rescans immediately
before finalization. Any remaining trainer, supervisor, or `caffeinate` process blocks the terminal
report; the read-only dashboard may remain. The launcher accepts no arm, seed, parent, checkpoint,
budget, root, or resume override. Closing the dashboard tab does not stop training; stopping the
launcher, trainer, machine, or storage connection closes the non-resumable cohort as operationally
incomplete.

The v0.3 entry points declared in `pyproject.toml` are:

| Command | Role | May start canonical training? |
| --- | --- | --- |
| `dungeon-smoke-v03-action-effect` | Disposable engineering comparison used by qualification | No |
| `dungeon-qualify-v03-action-effect` | Tag-payload generation and one-shot qualification | No |
| `dungeon-train-v03-action-effect` | Lower-level single-arm trainer used by the fixed launcher | Never call directly |
| `dungeon-dashboard-v03-action-effect` | Read-only evidence server | No |

They are refreshed by the repository's editable-install command. The canonical instructions above
use the equivalent module entry point so their availability does not depend on a stale generated
shell wrapper.

The sole scientific start command remains `scripts/run_v03_action_effect_stage_a.sh`.

## Resume a legacy v0/v0.1 run after interruption

This older generic procedure does not apply to U2r, U2-S, or v0.3 Stage A. U2r and
U2-S are terminal/non-resumable under their current records, and v0.3 Stage A is declared
non-resumable; never adapt this generic command to any of those protocols.

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
