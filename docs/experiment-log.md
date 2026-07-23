# Experiment log

This log records failures and negative results as carefully as successes. A later polished account
should be able to reconstruct what we knew at each decision rather than rewriting the path after the
ending is known.

## July 22, 2026 — Foundation

### Decision

The Pokémon Red project demonstrated that a large sparse world could consume enormous experience
without producing a coherent, reusable skillset on one Mac. The successor experiment would not use
an online language model, human trajectories, savestate lessons, or a succession of hand-authored
game checkpoints. Instead, it would use an original procedural world with a fixed visual interface,
mechanically provable levels, and automatic capability gates.

The initial question became: can one randomly initialized recurrent pixel policy learn navigation,
interaction, memory, and backtracking as accumulating concepts—and use them on unseen dungeons?

### Engineering qualification

- 300 levels generated: 100 Navigate, 100 Unlock, and 100 Retrieve.
- 300 levels solved by the privileged qualification oracle.
- 300 unique layout signatures.
- Median shortest legal solutions: 8, 17.5, and 27 actions by tier.
- 18 automated contract and behavior tests passed.
- Lint passed with no findings.

The oracle does not supply actions to training. Its only conclusion is that every sampled failure is
meaningful: a legal solution existed.

### Smoke run 1 — useful failure

The first 64-step end-to-end run exposed an error in atomic checkpoint publication. The model was
correctly written to a temporary path, but the publisher expected the training library to append a
different suffix. The attempted rename failed. The error was reproduced, corrected to detect the
actual temporary archive, and retained here because it validates the reason for smoke runs.

### Smoke run 2 — complete pipeline

The repeated 64-step run exercised model initialization, rollout collection, checkpoint publication,
held-out evaluation, status history, live frame capture, model reload, and clean shutdown. A random
policy scored 0/2 on validation, as expected. This was an engineering result, not evidence of skill.

### 4,096-step v0 canary — engineering evidence only

The first multi-update canary used four environments, recurrent PPO, training-only pixel curiosity,
and CPU execution.

| Measurement | Result |
| --- | ---: |
| Wall time | 73.7 seconds |
| Training steps | 4,096 |
| PPO updates in scheduled 4,096-step checkpoint | 70 |
| PPO updates in terminal `completed.zip` | 80 |
| Completed training episodes | 32 |
| Stochastic training successes | 5/32 (15.6%) |
| Frozen validation at 2,048 | 0/10 |
| Scheduled frozen validation labeled 4,096 | 0/10, 70-update policy |
| Terminal 80-update policy evaluation | Not run |
| Curriculum decision | Remain on Navigate |

The agent sometimes reached the exit while exploring, and the scheduled policy did not reproduce
that behavior on unseen layouts. No promotion occurred. However, the result cannot support a claim
about the terminal 4,096-step model: evaluation and checkpoint publication occurred inside rollout
collection, before PPO applied the latest update. The terminal model had 80 optimizer updates and
was never evaluated.

An audit also found that v0 curiosity could make failure the optimizer's preferred outcome. The
training bonus was `0.01 / sqrt(visits)` against a `-0.001` step cost. Measured random failed Unlock
and Retrieve episodes could outscore efficient successful episodes. Promotion still used task
success, so it did not falsely declare mastery, but training was optimizing the wrong shaped return.

These findings supersede the earlier interpretation of the canary. It proves that the simulator,
optimizer, recurrent policy, artifact writer, and evaluator could run end to end. It does not prove
or disprove learnability. Protocol v0 is closed and will not be resumed for capability claims.

### July 22, 2026 — v0.1 remediation

Protocol v0.1 begins from random parameters and changes the experimental machinery before the first
long run:

- curiosity pays 0.002 only for a new pixel fingerprint and is capped at 0.1 per episode;
- unchanged/repeated views pay no intrinsic reward and the reset view begins marked as seen;
- failed timeouts remain negative while maximum-horizon successes remain positive;
- exams and public checkpoints move behind complete PPO optimizer phases;
- status distinguishes collected and trained timesteps;
- checkpoint archives receive matching reproducibility sidecars;
- resume becomes a new bounded segment with explicit parentage;
- CI trains, saves, resumes a child segment, reloads, and evaluates a real recurrent-PPO checkpoint.

The audit and disposition of every finding are preserved in [audit.md](audit.md).

### July 22, 2026 — local v0.1 engineering qualification

The hardened path passed 58 local tests and a fresh 300-level oracle qualification. A real recurrent
PPO acceptance pair trained 128 steps, saved a digest-linked policy, resumed for another 128 steps,
and evaluated the 256-step child checkpoint. Lifetime and segment counters were exact (`256` total,
`128` in the child), both policies had completed optimizer updates, and the child used a distinct
recorded RNG stream: its first training episode was seed `39161`, not the parent's `814135`.

Scheduled and final exams at the same 128-step boundary named the same checkpoint digest. A separate
all-pass regression proved that the unchanged policy could cross only one curriculum gate at that
boundary. Evaluation now records key, door, relic, coverage, collision, and ineffective-interaction
diagnostics even when success remains zero. These are engineering results, not evidence that Navigate
has been learned.

A deliberately interrupted grandchild stopped cleanly at a fully trained 4,864-step checkpoint.
The next child restored exactly 4,864 collected/trained steps, 38 optimizer updates, Navigate state,
and the parent digest before adding its own 128-step segment. No interrupted policy was mislabeled as
having learned from unsafely partial experience.

The published branch then passed both GitHub Actions jobs: engine/unit qualification in 21 seconds
and the independent train-save-resume-reload-evaluate smoke in 1 minute 22 seconds. The run is
[preserved in GitHub Actions](https://github.com/PeteAndrews1289/dungeon-apprentice/actions/runs/29940403120).
This completes the v0.1 engineering release gate; it does not complete the first capability claim.

### July 22, 2026 — preregistered Navigate canaries

The first capability probe is three sequential, independently initialized runs at seeds `20260722`,
`20260723`, and `20260724`. Each receives 262,144 training steps on CPU with four workers, 512-step
rollouts, four PPO epochs, gamma 0.995, and the frozen 0.002/0.1 curiosity contract. Frozen exams run
every 32,768 trained steps against 40 validation layouts per unlocked tier. Ordinary checkpoints use
the same interval and retain the newest three. Runs live on the T7 and execute sequentially because
parallel neural optimization was slower on the audited 8 GB M1.

No reward, architecture, seed suite, or hyperparameter may change between these three runs. Automatic
promotion remains enabled: crossing 90% Navigate success is itself a canary result, and later exams
continue to measure Navigate retention if the policy begins Unlock. Training diagnostics and partial
milestones may explain behavior, but held-out success is the authority. These runs may establish a
Navigate learning signal; they cannot establish full curriculum mastery or final-suite generalization.

After all three complete, compare their eight-exam trajectories before changing anything. If Navigate
remains flat across the suite, declare a small controlled exploration study without consulting the
untouched final seeds.

### July 22, 2026 — v0.1 canary outcome

All three fresh policies learned Navigate. Seeds `20260722` and `20260723` crossed the 90% gate at
163,840 trained actions; seed `20260724` crossed it at 196,608. Their mean held-out Navigate score
rose from 5.8% at 32,768 actions to 90.0% at 163,840. This is the first replicated capability result
in the project.

The same runs produced a replicated failure after promotion. Every final Navigate exam scored
exactly 75%, below the 80% retention gate. Full Unlock scored 0/120 across the final deterministic
exams. In 978 stochastic Unlock training attempts the policies acquired a key 532 times, opened a
door 23 times, and completed the quest 22 times. The conversion from key acquisition to door opening
was only 4.32%; once a door opened, completion was 95.65%.

A reproducible 200-episode uniform-random diagnostic then acquired keys on 61% of full-Unlock
layouts, opened doors on 3.5%, and completed none. The canaries' stochastic key and door counts do
not establish an Unlock fragment; the frozen 0/120 result remains authoritative.

The retention audit exposed a separate implementation assumption. The declared 70/30 curriculum
sampled episodes. Successful Navigate episodes took roughly 29–32 actions while failed Unlock
episodes usually took 256. Navigate therefore supplied only 4.4%, 4.8%, and 5.7% of post-promotion
transitions—not the intended 30%. The policy experienced an abrupt shift to roughly 95% Unlock.

The immutable tables and interpretation are in
[the v0.1 Navigate canary report](results/v0.1-navigate-canaries.md).

### July 22, 2026 — v0.2 design decision

The proposed successor changes the curriculum rather than simultaneously changing the policy and
reward. It keeps complete key-door-exit success as the only authored positive reward, inserts three
procedurally simplified but complete Unlock distributions before the original full distribution,
requires two consecutive panel-balanced exams for advancement, and allocates rehearsal by measured
transitions rather than episode counts. A retention failure automatically enters a 75% Navigate
recovery schedule.

Retrieve is withheld. Protocol v0.2 asks one falsifiable question: can frequent complete early
practice turn rare stochastic interaction into deterministic full Unlock without destroying
navigation? The full design, outcome categories, budget, and deviation rules are in
[the proposed v0.2 specification](protocol-v0.2-design.md). It remains a proposal until its code,
qualification, tests, commit, and commands are frozen.

### July 22, 2026 — preregistered v0.2 U0 development sentinel

The first implementation slice deliberately stops after `unlock/u0-visible`; it does not claim that
the complete U0–U3 design is finished. It tests the two riskiest changes before expanding the graph:

1. a transition-deficit scheduler must produce the declared experience mix despite unequal episode
   lengths;
2. a fresh recurrent policy must convert a short, visible, but still complete key-door-exit quest
   into deterministic held-out behavior while retaining Navigate.

The Visible Unlock generator uses 9 × 9 layouts, ordinary empty inventory starts, 128-step horizons,
matching colored keys, locked doors, and a green exit beyond the barrier. Both the key and door are
visible at reset. Oracle solutions take 5–8 actions. Success still requires reaching the exit; key
pickup and door opening receive no authored reward. This development generator currently produces
806 distinct layout hashes across its first 1,000 seeds, so the sentinel freezes an 80% uniqueness
qualification rather than pretending it already satisfies the future five-lesson specification's
stronger diversity target.

The run begins from random parameters at seed `20260725` and receives at most 524,288 trained actions.
It retains the v0.1 policy and PPO settings. Frozen exams run every 32,768 trained actions against 80
layouts per active lesson. Navigate requires two consecutive 90% exams with both 40-layout panels at
85%. After promotion, the scheduler targets 50% Navigate and 50% Visible Unlock transitions. Visible
Unlock requires two consecutive 85% exams, both panels at 80%, while Navigate remains at 85%. If
Navigate falls below 85%, advancement freezes and practice changes to 75% Navigate / 25% Visible
Unlock until two recovery exams pass. A measured transition share more than five percentage points
from target blocks advancement.

The sentinel is **positive** only if it reaches the two-exam Visible Unlock gate with retained
Navigate. It is **partial** if final Visible Unlock reaches at least 25% without mastery. It is an
**interaction failure** below 25%, an **allocation failure** if any completed practice window misses
its transition target, and an **engineering failure** if checkpoint, evaluation, dashboard, or
qualification invariants break. Milestone rates explain a result but never replace exit success.

The exact command is preserved in `scripts/run_v02_sentinel.sh`. It runs one CPU trainer, writes to
`/Volumes/T7 Developer/DungeonApprentice/sentinels/v0.2-u0-20260722`, keeps the existing 25 GiB free
space guard, serves the dashboard on port 8781, and makes no online model calls. No behavior-affecting
setting may change after launch. This sentinel is development evidence and is not pooled with the
future three-seed v0.2 confirmation.

### July 22, 2026 — v0.2 U0 sentinel result and replication preregistration

Seed `20260725` mastered the development sentinel after 491,520 trained actions and 54 minutes. Its
two qualifying Visible Unlock exams scored 74/80 and 79/80 while Navigate retention scored 73/80
and 74/80. The final 79/80 checkpoint result was independently reproduced after reloading the saved
model. Its final completed practice window measured 50.0031% Navigate and 49.9969% Visible Unlock;
all completed windows remained inside tolerance. This is a positive development result, not yet a
replicated U0 claim and not evidence for U1–U3.

The audit found that finalization repeated an already completed mastery exam, attached a false
“held” label to the duplicate, and reset the displayed allocation to an empty window. A
behavior-neutral correction now skips same-boundary final evaluation and preserves the final
completed allocation, with regression coverage for both cases. The frozen U0 validation block also
contains 79 distinct layouts across its 80 seed cases; this is disclosed and retained for direct
replication rather than silently changing the exam after seeing the result.

Two fresh random initializations, seeds `20260726` and `20260727`, are preregistered as sequential
replications under the identical 524,288-action ceiling and decision rules. Both must master U0 for
the result to count as replicated. Exact paths, ports, refusal guards, outcome categories, and the
post-run reporting contract are frozen in
[the U0 replication plan](v0.2-u0-replication-plan.md).

### July 22, 2026 — replicated U0 result and confirmation preregistration

Both preregistered policies mastered Visible Unlock, so the strict replication criterion passed.
Seeds `20260725`, `20260726`, and `20260727` all reached mastery at 491,520 trained actions. Their
final Navigate scores were 74/80, 75/80, and 73/80; their final U0 scores were 79/80, 80/80, and
80/80. Every final Navigate panel reached at least 85%, every U0 panel reached at least 80%, no
policy entered retention recovery, and every completed post-promotion practice window remained
within five percentage points of its 50/50 transition target.

The U0 curves were nonlinear. All three first crossed the qualifying threshold at 458,752 actions,
after long intervals at low or zero success, and all three required the next frozen exam to confirm
mastery. This validates the two-consecutive-exam rule and shows why an apparently flat early curve
was not enough to classify an interaction failure.

The result remains deliberately narrow. All policies were evaluated on the same 80 seeded U0 cases,
which represent 79 unique layouts. That suite was appropriate for comparing random initializations
without moving the goalposts, but it is not a disjoint confirmation of layout generalization. Exact
curves, panels, allocation records, artifact digests, limitations, and the frozen claim are in
[the v0.2 Visible Unlock replication report](results/v0.2-visible-unlock-replication.md).

Before inspecting any new confirmation case, a post-training protocol froze all three selected
mastery checkpoints and two disjoint 200-case blocks: Navigate seeds `15_000_000`–`15_000_199` and
U0 seeds `15_010_000`–`15_010_199`. Each block is split into two fixed 100-case panels and must report
at least 80% unique layouts. Every checkpoint must independently retain at least 85% Navigate overall
and in each panel, and reach at least 85% U0 overall with at least 80% in each U0 panel. All three
must pass; checkpoint digests and training-allocation evidence must verify before any score counts.
No checkpoint reselection, retuning, seed substitution, or threshold change is allowed after
evaluation. The complete preregistration is
[the U0 confirmation plan](v0.2-u0-confirmation-plan.md).

### July 22, 2026 — disjoint U0 confirmation passed

The confirmation was executed from clean commit `2b215e7` without policy updates. Both frozen
200-case blocks passed oracle qualification before scoring: Navigate contained 200 unique solvable
layouts, and U0 contained 193 unique solvable layouts. All three selected checkpoints passed every
overall and panel gate. Their Navigate scores were 189/200, 192/200, and 187/200; their U0 scores
were 187/200, 199/200, and 198/200.

The strict all-three verdict is positive. The raw 72 KiB report, including seed-to-layout hashes,
checkpoint and source verification, allocation evidence, and individual metrics, is stored on the
T7 with SHA-256 `f43610252943fce9c0169ac0231724fc2829686d1899f5f88bd0c250a913b398`.
The immutable human-readable result is in
[the U0 confirmation report](results/v0.2-u0-confirmation.md). U1 Local Unlock is now the next
declared capability question; this result does not pre-approve its generator or training protocol.

### July 22, 2026 — U1 Local Unlock child frozen

The next implementation does not alter the confirmed U0 sentinel or pretend that a warm start is a
fresh replication. It creates protocol `dungeon-apprentice-v0.2-u1`, verifies the exact confirmed
seed-`20260725` mastery archive and confirmation report, restores the full PPO optimizer, and begins
a new cumulative-learning segment. No trajectory, recurrent episode state, oracle action, or
demonstration crosses the boundary.

“Local” is now testable rather than rhetorical. The key is visible at reset, the door is hidden, the
two are non-collinear and at least four cells apart, a turn is required after key pickup, and the
complete shortest solution takes 9–18 actions. The first 1,000-case engineering qualification
solved all 1,000, produced 998 exact visual layouts and 991 geometry-only layouts, and had no exact
visual overlap with the fixed 80-case U1 validation suite. Training rejects exact visual hashes from
qualification and all validation suites.

The audit also closed a protocol loophole: the earlier proposal could pass U1 while forgetting U0.
The child now evaluates Navigate, Visible Unlock, and Local Unlock at every 32,768-action boundary.
Normal practice targets 50/15/35 percent of actual transitions. Navigate-only, U0-only, and combined
retention failures receive separate rehearsal mixes and need two recovery exams before U1 practice
can resume. U1 mastery requires two consecutive post-update boundaries where every lesson and both
panels pass and allocation remains valid.

An end-to-end 2,048-action engineering smoke loaded the parent's 491,520-step/960-update archive,
recorded a non-counting pre-update baseline of Navigate 74/80, U0 79/80, and U1 0/80, completed four
new PPO updates, then scored Navigate 71/80, U0 79/80, and U1 1/80. This is plumbing evidence only;
the tiny post-update U1 result is not a capability claim. The real child remains frozen at a
524,288-new-action ceiling. Exact parentage, generator rules, seed blocks, gates, recovery, resume,
and outcomes are in [the U1 development protocol](protocol-v0.2-u1-development.md).

A separate deliberate-interruption smoke then stopped segment 0 at 4,096 child actions: 495,616
lifetime trained actions, 968 optimizer updates, and lifetime practice counts of 1,977 Navigate,
657 U0, and 1,462 U1 transitions. Segment 1 loaded that exact schema-3 bundle and completed the
declared 8,192-child-action engineering ceiling at 499,712 lifetime actions and 976 updates. Its
practice counts continued to 3,920 Navigate, 1,360 U0, and 2,912 U1; parent digest, curriculum state,
and scheduler stream remained intact. The test discarded no trained progress and did not grant the
resumed segment a replacement budget.

### July 22, 2026 — U1 lead mastered and replication frozen

The Local Unlock lead mastered after 393,216 new actions and 51 minutes. Frozen U1 success rose from
0/80 before the first update to 73/80 and 72/80 at two consecutive mastery boundaries. Final
Navigate was 74/80 and Visible Unlock was 80/80, so the new capability did not replace its two
prerequisites. One temporary Navigate recovery began at 131,072 child actions and ended after two
clean anchor exams at 196,608, providing direct evidence that the automatic rehearsal controller
operated as declared.

This remains one selected lineage. Before either new child received an update, the project froze two
sequential replications from the other independently confirmed U0 parents. Fresh child streams
`20260729` and `20260733` do not overlap the lead's four worker streams or each other. They retain the
same generator, reward, optimizer, scheduler, exams, gates, recovery rules, and 524,288-action
ceiling. Both must master for Local Unlock learnability to count as replicated. See
[the U1 replication plan](v0.2-u1-replication-plan.md).

### July 22, 2026 — U1 replication passed

Both preregistered Local Unlock children mastered, so the frozen replication rule passed and the
lead plus replications produced three positive lineages out of three. Child `20260729` mastered at
393,216 new actions after combined Navigate/U0 recovery and a later U0-only recovery. Child
`20260733` mastered at 294,912 new actions without entering recovery. Their final U1 scores were
72/80 with panels 36/40 and 36/40, then 35/40 and 37/40. Final Navigate was 77/80 and 75/80; both
finished U0 at 80/80.

Every completed transition window matched its active normal or recovery target within 0.28
percentage points, far inside the five-point tolerance. The lead and first replication each used
393,216 child actions; the second used 294,912. All three final U1 scores happened to be 72/80, but
the verdict remains per policy rather than a pooled 216/240 statistic. Complete trajectories,
recovery windows, whole-child versus final-window allocation, clean source commits, qualification,
and mastery digests are frozen in
[the Local Unlock replication result](results/v0.2-local-unlock-replication.md).

### July 22, 2026 — U1 confirmation preregistered

Before generating a layout or policy score from any new confirmation block, the project froze the
three first-mastery checkpoint digests and three deterministic 200-case exams. Local Unlock uses its
previously reserved `15_020_000` block; new Navigate and U0 retention blocks begin at `15_030_000`
and `15_040_000`. Every checkpoint must pass all three overall gates and both fixed panels with no
weight updates. Qualification, provenance, allocation history, deterministic inference, and a
strict all-three decision are specified in
[the U1 confirmation plan](v0.2-u1-confirmation-plan.md).

### July 23, 2026 — U1 confirmation attempt 1 stopped at qualification

The preregistered evaluator opened all three 200-case blocks from clean source commit `19137fa`.
Every block was mechanically valid, oracle-solvable, and above its diversity floors. The numerical
range audit also passed. The stronger exact-layout audit then found eight U0 cases that repeated
fixed U0 validation layouts and one U1 case that repeated the engineering-qualification set.

The evaluator honored the frozen rule and stopped before importing or scoring a policy. The
attempt is therefore `qualification_failed`, not a negative capability result. Its raw report
SHA-256 is `d2fa53308f7488cc08f5ee67b86a125ad91c6ba3790fcd9433c35aab1215b25c`.
The complete result and all nine exact collisions are preserved in
[the immutable attempt-1 record](results/v0.2-u1-confirmation-attempt-1.md).

### July 23, 2026 — collision-safe successor designed

The failure exposed a concrete distinction: disjoint random seeds do not guarantee distinct
generated dungeons when a finite procedural generator maps multiple seeds to the same layout. The
successor does not delete cases or relax attempt 1. It prospectively reserves fresh 10,000-seed
candidate streams, then uses a deterministic policy-blind selector to accept the first 200
mechanically valid, exact-unique layouts outside the declared development references. It also
excludes same-lesson layouts from the opened-but-unscored first attempt, and makes U1 exact-layout
novel to all three frozen child histories. Only after selection and qualification complete may a
policy load. See [the successor confirmation plan](v0.2-u1-confirmation-v2-plan.md).

### July 23, 2026 — collision-safe U1 confirmation passed

Confirmation v2 ran once from clean source commit
`ebf064afd6e6296bb21524103c2a3c269a56e7a5`. The deterministic selector qualified all three exams
before importing a policy. Navigate accepted the first 200 candidates with zero rejections and
200 unique exact layouts/geometries. Visible Unlock examined 236 candidates, rejected 36, and
accepted 200 unique exact layouts with 180 unique geometries; those rejections removed repeats from
validation, attempt 1, and the already accepted set. Local Unlock examined 207, rejected seven, and
accepted 200 exact- and geometry-unique layouts; its rejected cases overlapped attempt 1 or the
union of frozen U1 training histories. No accepted case overlapped a declared exact-layout
exclusion.

Every frozen policy independently passed Navigate, Visible Unlock, and Local Unlock overall and in
both 100-case panels:

| U1 child seed | Navigate | Visible Unlock U0 | Local Unlock U1 |
| ---: | ---: | ---: | ---: |
| `20260725` | 188/200 (95 + 93) | 200/200 (100 + 100) | 188/200 (93 + 95) |
| `20260729` | 192/200 (96 + 96) | 200/200 (100 + 100) | 192/200 (95 + 97) |
| `20260733` | 188/200 (94 + 94) | 200/200 (100 + 100) | 182/200 (87 + 95) |

The strict all-three verdict is `confirmed`. Evaluation ran under inference mode and performed no
updates; each checkpoint's policy tensors, optimizer state, archive bytes, trained-timestep count,
and optimizer-update count were identical before and after its 600 episodes. The externally hashed
raw report is
`6e577170050f6f14599b793a031776a19bf7c64eba0f243f457298da3193ae8f`.
The complete qualification, collision, lineage, panel, milestone, and no-update evidence is in
[the confirmation v2 result](results/v0.2-u1-confirmation-v2.md).

This confirms replicated cumulative Local Unlock on the declared distribution; it does not prove
full Unlock or Retrieve. The next prospective development boundary is
[U2 Separated Unlock](protocol-v0.2-u2-separated-unlock.md).

### July 23, 2026 — U2 implementation boundary activated, with no run yet

The full disposable engineering range, seeds `5,200,000`–`5,200,999`, subsequently produced 1,000
contract-complete and 1,000 exact-unique Separated Unlock layouts. Privileged live-oracle solutions
covered the full 17–26-action range and all four initial key/door visibility strata. This is
generator engineering evidence only; no sealed, validation, future-confirmation, or final-test case
was opened. The full record and chart are in
[the U2 engineering sandbox report](results/v0.2-u2-engineering-sandbox.md).

The collision-safe U1 confirmation activated design and implementation work on U2; it did not
retroactively make U2 a learned capability. Each of the three exact confirmed U1 first-mastery
archives is frozen as the parent of one independent child. Policies, trajectories, replay,
recurrent episode state, and optimizer updates do not cross between lineages. Each child keeps only
its own parent's complete recurrent-PPO archive and optimizer.

The next task is Separated Unlock. A 9 × 9 map is divided by a complete wall whose only crossing is
one locked door. Start and matching key are on the approach side, the goal is on the far side, and
exactly two additional valid interior walls lengthen or redirect travel. Key and door visibility are
unconstrained. The pure planner and separately run live oracle must agree on a 17–26-action complete
key → door → goal solution, including at least one turn between key acquisition and opening the
door. Qualification actions are never retained as demonstrations.

This is a controlled geometry step. The `56 × 56 × 3` partial pixel observation, seven actions,
256-unit recurrent state, PPO architecture and hyperparameters, `+1.0` complete-success reward,
`-0.001` step cost, and bounded `0.002`/`0.1` training-only pixel curiosity remain unchanged. A
key pickup or opened door is still telemetry, not payment. The episode horizon increases to 160
because the mechanically qualified solutions are longer.

Every child must preserve four abilities: Navigate, Visible Unlock, Local Unlock, and the new
Separated Unlock. A four-lesson diagnostic baseline occurs before update one and cannot satisfy a
gate. Thereafter the exact post-optimizer bytes are checkpointed, hashed, and examined every 32,768
new trained actions. A prerequisite miss resets U2 progress and invokes a frozen
transition-balanced recovery profile. Recovery changes practice frequency only; it cannot change
reward, lower gates, roll weights back, or count as a U2 mastery pass.

The evidence stages are now named explicitly:

- **Engineering proof:** property tests, golden inherited behavior, save/reload/resume,
  interruption, storage, path confinement, and the one-shot 2,000-case planner/live-oracle
  qualification establish that the experiment is capable of measuring its question.
- **Learning evidence:** three sequential children show whether each confirmed lineage changes its
  four frozen development-exam scores within at most 1,048,576 new actions.
- **Confirmation evidence:** only a later separately preregistered, collision-aware, no-update
  evaluation of frozen selected checkpoints can establish generalization beyond development.

The U1 confirmation failure remains part of U2's design: seed roles are separated and exact-layout
hashes are tracked. Policy training remains below one million; disposable generator engineering is
`5_200_000`–`5_200_999`; one-shot qualification is `5_210_000`–`5_211_999`; frozen validation
occupies the existing 10–11.2-million lesson blocks; four future confirmation streams begin at
`15_200_000`; and the 20-million final allocation remains untouched.

The three children will run sequentially on the audited 8 GB M1 and share a read-only cohort view at
`127.0.0.1:8785`. Measured U1 throughput suggests roughly 8–10 hours if the cohort approaches its
full 3,145,728-action ceiling. Scientific evidence is capped at 2 GiB per lineage and 6 GiB for the
cohort; optional video lives separately under a 10 GiB cap, for a 16 GiB maximum planned addition.

The implementation audit then found several ways an otherwise valid long run could become
scientifically ambiguous: a completed optimizer update between exams could be lost on interruption;
the sequential script could launch fresh work but could not safely continue it; an old first-pass
candidate could outlive a broken mastery streak; and a stale internal directory could masquerade as
the external T7 mount. Those are engineering failures, not reasons to alter the learning task.

The frozen implementation addresses them without changing observations, rewards, lessons, gates, or
budgets. Every completed 2,048-action optimizer phase now publishes one bounded `latest-safe`
resume bundle. A continuation creates a new segment, restores the exact optimizer/curriculum/RNG
state, and spends only the remaining cumulative budget. The active mastery candidate must be the
immediately previous 32,768-action passing boundary. Launchers prove the real T7 mount, keep durable
cohort state through signals and failures, and verify one remote annotated Git tag that binds the
qualification report bytes, attempt, claim, generator profile, and source commit.

At this source-freeze entry, the protected one-shot qualification, U2 validation, future
confirmation, and final partitions remained unopened and no U2 child had trained. The accepted
commands now live in the runbook. Once they execute, runtime truth belongs to the immutable
qualification attempt, external tag, cohort manifest, segment sidecars, and status records. This
entry remains a frozen question and engineering decision record—not a capability result. The
check totals and complete audit disposition are preserved in the
[U2 launch-readiness record](results/v0.2-u2-launch-readiness.md).
