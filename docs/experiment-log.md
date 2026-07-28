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

### July 23, 2026 — U2 Separated Unlock replicated 3/3

The protected cohort ran once from frozen source
`b7b5d361b0aa2eeabedc435fa0d4b9e1ffdd09db`, bound to annotated qualification tag
`u2-preflight-v0.2-u2-20260723`. The sealed generator instrument solved all 2,000 declared cases
before any child update. Three confirmed U1 first-mastery archives then became three independent U2
children; no policy, trajectory, replay, recurrent state, or optimizer update crossed lineages.

All three children reached the complete two-consecutive-exam mastery rule:

| Child seed | U2 baseline | First 32,768 | Terminal N / U0 / U1 / U2 | Child actions |
| ---: | ---: | ---: | ---: | ---: |
| `20260737` | 32/80 | 40/80 | 78 / 80 / 80 / 75 | 688,128 |
| `20260741` | 33/80 | 44/80 | 74 / 80 / 79 / 74 | 557,056 |
| `20260745` | 24/80 | 29/80 | 79 / 76 / 77 / 72 | 688,128 |

The average U2 score rose from 37.1% at inheritance to 47.1% after one practice window and 92.1% at
mastery. Child `20260737` recovered Navigate once. Child `20260741` recovered Navigate and then U1.
Child `20260745` entered no prerequisite recovery. Earlier promising U2 passes that did not remain
adjacent were correctly discarded; each selected first-pass artifact is the immediately preceding
32,768-action boundary in the final mastery pair.

The sequential cohort finished in 13,286 seconds—3 hours, 41 minutes, and 26 seconds—using
1,933,312 of its possible 3,145,728 actions. Independent terminal verification recomputed 66 hashes
across 33 retained checkpoint/integrity pairs with no mismatch, authenticated the three U1 parents
and U1 confirmation, verified every promotion chain and allocation-valid boundary, found no orphaned
trainer, and measured 866,332,331 bytes of scientific evidence under the 6 GiB cohort cap.

This is a positive 3/3 development replication under the frozen U2 rule. It supports the narrow
claim that three cumulative pixel-only policies extended the learned key → door → exit ritual into
a longer, partially observed search while retaining Navigate, Visible Unlock, and Local Unlock. It
does not yet support disjoint U2 generalization. The four reserved 15.2-million candidate streams
remain closed until the separately committed
[U2 confirmation plan](v0.2-u2-confirmation-plan.md) freezes selection, collision exclusions,
panels, gates, checkpoint identities, and no-update verification.

Closeout found one bounded instrumentation gap before confirmation was opened. Each four-worker
trainer stopped on an optimizer boundary while four episodes were still active, so 12 layouts
contributed final transitions without producing completed-episode records. The frozen histories
therefore authenticate every completed logged episode but do not directly persist those 12 active
layouts. The prospective confirmation claim was narrowed accordingly rather than hiding the gap.
U3 must persist an episode-start identity and each worker's active seed, lesson, layout, elapsed
steps, RNG state, and scheduler reservation at every checkpoint.

The first two confirmation launch commands stopped before preregistration or candidate access. The
new console shortcut had not yet been installed into the existing virtual environment, so the first
command never entered Python. Direct module invocation then exposed a historical-anchor adapter bug:
it verified the frozen U2 training tag correctly but returned the active evaluator commit in its
public record. That failed the qualification binding before the confirmation tag, canonical claim,
or any `15.2`-million seed existed. The adapter now returns the anchored training commit when
historical verification is requested, with a regression test covering an evaluator running from a
newer clean commit. The full 299-test suite and 300-level mechanical qualification passed again
before another launch.

A second pre-claim stop found a portability edge in Git itself: on the installed Apple Git,
`show-ref --verify` returns `128` rather than the expected absent-ref code unless `--quiet` is
present. The tag publisher treated that as indeterminate and again stopped before creating a tag,
claim, or candidate layout. Both U2 tag publishers now use the portable quiet form.

### July 23, 2026 — U2 confirmation completes with a one-case strict failure

The canonical attempt then opened once from clean commit
`6c266e0a51cc951a9a37d98e611049f08b1e143f`, externally anchored by annotated tag
`u2-confirmation-v0.2-u2-20260723`. The policy-blind selector examined 819 candidates to accept four
200-case exams. Navigate and U2 accepted their first 200 candidates. U0 examined 215, rejecting four
development-validation exact overlaps and 11 duplicates of already accepted exact layouts. U1
examined 204, rejecting four exact overlaps with completed logged U2-child histories. All accepted
exams had 200 unique exact layouts; the numerical partition audit found no collision.

The three frozen checkpoints scored:

| Child seed | Navigate | Visible U0 | Local U1 | Separated U2 | Strict result |
| ---: | ---: | ---: | ---: | ---: | --- |
| `20260737` | 193/200 | 200/200 | 200/200 | 184/200 | Passed |
| `20260741` | 188/200 | 200/200 | 200/200 | 188/200 | Passed |
| `20260745` | 190/200 | 188/200 | 196/200 | **169/200** | Failed U2 overall by 1 |

Every one of the 24 panel gates passed. Child `20260745` retained all three prerequisite lessons
and produced U2 panels of 84/100 and 85/100, both above the 80-case panel floor. Its combined 169
was nevertheless below the preregistered 170/200 overall requirement. The strict all-three verdict
is `capability_failed`, not confirmed.

The failed policy acquired the U2 key in 188 cases, opened the door in 172, and completed 169. Once
the door opened, completion was 98.26%; door opening after key acquisition was 91.49%. It also
averaged 5.135 ineffective interactions and 2.526 policy actions per oracle action, versus
0.71/2.105 and 0.19/1.905 for the two passing policies. These are useful post-result diagnostics,
not new gates or permission to tune against the opened exam. A Binomial(200, 0.85) model assigns
45.1489% probability to 169 or fewer successes, so the result is not evidence of collapse; the
frozen discrete rule still makes it a protocol failure.

Evaluation made no update. For every checkpoint, policy tensors, optimizer state, archive bytes,
trained-action counters, update counters, and supporting artifact hashes matched before and after.
The evaluator used deterministic actions, reset recurrent state per case, made no online model
calls, and finished without interruption. The raw 1,388,908-byte report is frozen at SHA-256
`7522eb8742ed567577d1f02a2a9960d698981044128e0866daa262d36aaf1c69`; its external checksum,
attempt ledger, and terminal bundle agree.

The result preserves two claims at once. U2 remains a positive 3/3 development-replication result.
Its larger collision-aware generalization claim is only 2/3 and therefore fails the declared cohort
rule. The opened 15.2-million exams cannot be recycled, the threshold cannot be moved after seeing
169, and U3 remains blocked. Full selection, integrity, launch-stop, diagnostic, and limitation
evidence is in the [immutable confirmation result](results/v0.2-u2-confirmation.md).

### July 23, 2026 — U2r freezes one bounded stability-remediation path

The project did not reinterpret 169/200 as a pass, rerun the unchanged checkpoint, open U3 with
only the two strongest policies, or silently add replacement lineages until one passed. Instead it
froze [U2r Stability Remediation](protocol-v0.2-u2r-stability-remediation.md) before another policy
update or fresh protected case.

The successor continues only child `20260745` from exact mastery archive
`56dc459fb94110f41a14b2304425f572fc235c8cfc07e1152306cbf77ac33dee`, including its optimizer.
The child stopped at 688,128 of its original 1,048,576-action U2 ceiling, leaving exactly 360,448
actions, or eleven 32,768-action windows. U2r must spend all eleven under the unchanged learner,
reward, four-lesson curriculum, 50/7.5/7.5/35 normal mix, and existing recovery controller. Segment
algorithm seed `20260749`, worker streams `20260749`–`20260752`, and annotated training tag
`u2r-stability-v0.2-u2r-20260723` are frozen prospectively.

No favorable intermediate checkpoint may be selected. The only eligible artifact is the exact
1,048,576-child-action terminal checkpoint with 1,835,008 lifetime actions and optimizer counter
3,584. Its exams at 1,015,808 and 1,048,576 child actions must both follow allocation-valid normal
practice, with recovery inactive, and pass all original overall and panel gates. Both must also
score at least 72/80 U2 with panels at least 34/40, hold mean ineffective interactions to at most
3.0 per episode independently on U0/U1/U2, and avoid any terminal lesson decline greater than 2/80
from the penultimate exam.

This stronger rule is explicitly post-result but prospectively applied to a new artifact. It is
motivated by the broad interaction-loop regression seen between child `20260745`'s original first
pass and selected mastery boundary; it does not regrade either checkpoint.

Only a terminal-eligible artifact may face the one new confirmation allocation:

| Lesson | Fresh candidate stream |
| --- | --- |
| Separated U2 | `15_240_000`–`15_249_999` |
| Navigate | `15_250_000`–`15_259_999` |
| Visible U0 | `15_260_000`–`15_269_999` |
| Local U1 | `15_270_000`–`15_279_999` |

The consumed `15_200_000`–`15_239_999` ranges remain inaccessible, and the complete-project final
allocation beginning at 20 million remains unopened. The new evaluator gets one attempt, selects
200 accepted cases per lesson before loading the policy, preserves the existing 170/200 overall
and 80/85 panel gates, and performs no update.

A positive outcome may open a separately frozen U3 protocol only with the ancestry label **two
directly confirmed U2 policies plus one prospectively remediated U2r policy**. The original U2
confirmation remains 2/3 and `capability_failed` in every outcome. A valid U2r failure ends this
rescue path rather than authorizing another extension, checkpoint substitution, or confirmation
retry.

### July 23, 2026 — U2r implementation qualifies without starting training

The bounded successor is now implemented, qualified, and externally anchored under annotated tag
`u2r-stability-v0.2-u2r-20260723`. This release adds the sole-parent continuation trainer,
terminal-only stability grader, exact case evidence, static history exclusions, fail-closed
interruption/resume chain, storage guards, read-only dashboard, fixed launcher, and narrative
documentation. It does not add a confirmation evaluator or any route to U3.

The release boundary passed:

- **427/427** repository tests;
- **133/133** U2r-focused provenance, trainer, dashboard, launcher, anchor, and seed-guard tests;
- Ruff, Python compilation, launcher syntax, and patch-integrity checks;
- the existing mechanical oracle qualification on **300/300** generated levels; and
- a fresh read-only reconstruction of **84,218** exact static exclusion layouts with canonical
  set digest
  `4157869d218289eebdd6a04dd0bd4cd2c91e6717cbe6eb6d966fc96749cf0fb3`.

The exclusion evidence preserves the original protocol's explicit upper-bound limitation: up to 12
terminal active-worker layouts were never authenticated by the historical U2 logging and therefore
cannot be invented or added now. U2r closes that evidence gap prospectively by recording every
episode start and the four actual active workers at a durable interruption or terminal boundary.

Release verification found the mounted T7 with roughly 216 GiB free, no active neural trainer,
fixed dashboard port 8786 available, and neither the canonical U2r run root nor media root present.
The fresh `15_240_000`–`15_279_999` candidate streams remained structurally sealed and no issuer was
created. Accordingly this entry records **implementation readiness, not a learning result**:
trained U2r actions remain zero, all eleven windows remain pending, and U3 remains closed.

### July 23, 2026 — U2r launch attempt 1 stops before policy action one

The externally preregistered launcher started at `2026-07-23T15:38:06+00:00`. Four seconds later,
during Stable-Baselines3's initial vector-environment reset, it exited with status `1`. The policy
had taken **zero actions**, the remediation counter had advanced by **zero actions**, and the
optimizer had completed **zero updates**. The inherited counters remained 688,128 child actions,
1,474,560 lifetime actions, and 2,880 updates.

One worker durably logged a Local Unlock U1 episode start at seed `767339` and layout
`3bab367b4263d0638d9d98142686c1f34c7026dc3d83465d12ffba3c1c30784c`.
Its record shows zero elapsed steps, zero worker transitions, and an empty action histogram. The
next scheduler assignment was finite Visible Unlock U0. Worker stream `20260750` proposed 128
candidates—126 unique layouts—and every proposal hit the frozen exclusion union. The environment
then failed with `could not sample a training layout outside reserved evidence sets` before rollout
collection or policy inference.

The frozen implementation had correctly reconstructed 84,218 historical exact identities and
applied the entire union to every lesson. That was too strong for the finite U0 generator. Its exact
domain contains 2,800 layouts; 2,774 were hard-excluded, leaving only 26. The failure therefore
invalidates the launch rule, not the learner.

The failed evidence remains immutable:

| Artifact | SHA-256 |
| --- | --- |
| Launcher state | `99bde1ff17ea54b6d573305100f351e67829cd10320a5cebe326ab6a0f414ec5` |
| Manifest | `5d2b0a632f03ed6963c5eff496d4907f5b811d34efbc3ed102f63d4012c4fc16` |
| Crash report | `aca77af751f987c363f7deb292ffcba5b3ed919a2fdd2e381dda13405f47482a` |
| Episode-start ledger | `f5117f0bf22256774a2b802a99311a8a9fea3c02ff5faaaa9f85df4cd0ed063b` |

No status, progress ledger, completed episode, evaluation, checkpoint, interruption record,
integrity record, terminal report, or resumable state was produced. The parent archive remains
unchanged at
`56dc459fb94110f41a14b2304425f572fc235c8cfc07e1152306cbf77ac33dee`.
The original tag `u2r-stability-v0.2-u2r-20260723`, source, run root, empty media root, and four-file
bundle remain the permanent attempt-1 record. Full interpretation is in the
[launch result](results/v0.2-u2r-launch-attempt-1.md).

### July 23, 2026 — U2r r1 amendment qualifies before training

The project does not rewrite attempt 1 or invoke its nonexistent resume path. It prospectively
defines `dungeon-apprentice-v0.2-u2r-stability-r1`, annotated tag
`u2r-stability-v0.2-u2r-r1-20260723`, root
`/Volumes/T7 Developer/DungeonApprentice/u2r-stability-r1-20260723`, media root
`/Volumes/T7 Developer/DungeonApprentice/u2r-stability-r1-media-20260723`, and initial segment
`v02-u2r-r1-seed-20260745`.

r1 retains the exact policy and optimizer parent, remaining 360,448-action budget, algorithm seed
`20260749`, worker streams `20260749`–`20260752`, unchanged PPO/reward/curriculum, terminal-only
selection, stability gates, and still-sealed `15_240_000`–`15_279_999` confirmation allocation.
Using the same streams is not repeated learning because attempt 1 ended before action one or update
one; the independent root and tag preserve the launch history.

The only behavioral amendment is lesson-specific training exclusion. Navigate, U1, and U2 retain
the authenticated historical-novelty guard. U0 rejects only its 79 unique frozen development
layouts, leaving 2,721 legal layouts; earlier U0 training-history overlap remains mandatory
diagnostic evidence rather than a gating exclusion. The complete historical inventory and source
digests remain retained. The attempt-1 U1 start identity is separately bound by the failed-launch
evidence, but does not join the U1 refusal set because no policy action occurred in that episode.

r1 then passed its release boundary before a canonical run root existed:

- **435/435** repository tests and **200/200** focused r1 boundary tests;
- Ruff, Python compilation, launcher syntax, and patch-integrity checks;
- **300/300** generated levels solved by the mechanical oracle;
- exact authentication of the four-file r0 zero-action bundle;
- reconstruction of all **84,218** retained identities at digest
  `4157869d218289eebdd6a04dd0bd4cd2c91e6717cbe6eb6d966fc96749cf0fb3`;
- exact r1 guard mapping digest
  `d58e063841be32a36a3c6bdadb333da90401f282b7734c1d6307a2d097c49536`; and
- all **16/16** fixed lesson/worker sampler checks accepted on their first proposal.

The mounted T7 had roughly 216 GiB free, no trainer was active, and the r1 root did not exist.
Accordingly this is implementation qualification, not a learning result: no r1 action, checkpoint,
exam, or learning outcome existed at the freeze, the confirmation streams remained sealed, and U3
remained closed.

### July 23, 2026 — U2r-r1 completes with a valid terminal stability failure

The corrected r1 trainer began at `2026-07-23T16:09:08+00:00` from clean source
`86d3424a631a661966c763da3e1fabf27df798f1` and annotated tag
`u2r-stability-v0.2-u2r-r1-20260723`. It used the exact failed-lineage U2 checkpoint and optimizer,
algorithm seed `20260749`, worker streams `20260749`–`20260752`, and the prospectively amended
lesson-specific guards. It finished at `2026-07-23T16:53:18+00:00` without a crash.

Every declared boundary was exact:

- 360,448 remediation actions;
- 1,048,576 total U2 child actions;
- 1,835,008 lifetime actions;
- 3,584 optimizer updates;
- eleven 32,768-action post-update exams; and
- 3,520 authenticated deterministic case records.

All eleven windows were allocation-valid normal practice. The U2 trajectory was not a simple
failure to learn:

| Boundary | Navigate | U0 | U1 | U2; panels | U2 ineffective |
| --- | ---: | ---: | ---: | --- | ---: |
| Inherited source | 79 | 76 | 77 | 72; 36, 36 | 6.5375 |
| 32,768 | 73 | 78 | 76 | 70; 35, 35 | 11.8500 |
| 163,840 | 79 | 80 | 80 | 73; 37, 36 | 0.1250 |
| 196,608 | 80 | 80 | 80 | 77; 38, 39 | 2.1625 |
| 327,680 — penultimate | 78 | 80 | 80 | **76; 37, 39** | **0.1875** |
| 360,448 — terminal | 77 | 80 | 80 | **76; 37, 39** | **4.1625** |

The terminal-pair grader passed every original overall and panel gate, both stronger 72/80 U2
overall gates, all four 34/40 U2 panel gates, the decline limits, allocation checks, normal-practice
checks, and penultimate interaction checks. Its sole failed condition was
`unlock/u2-separated:terminal_ineffective_at_most_3`.

Case evidence explained the mean. The penultimate U2 exam contained only 15 ineffective
interactions and no case with ten. The terminal exam contained 333. Validation seed `11_200_025`
opened the door and then repeated toggle 145 times; seed `11_200_027` repeated pickup 153 times.
Together those two timeouts contributed 299 interactions, 89.8% of the U2 total. A separate Local
Unlock case produced 42 ineffective interactions and a 41-action repeated run while its lesson mean
remained only 0.5875. These tails showed why a lesson mean alone was not a sufficient stability
description.

The valid scientific verdict is **`failed`**, not an engineering crash and not capability
collapse. The policy still completed 76/80 U2 cases, acquired 79 keys, and opened 78 doors.
However, the exact terminal artifact exceeded the prospectively frozen mean limit and could not
enter confirmation. The equally capable, unusually clean penultimate checkpoint remains diagnostic
evidence only; selecting it after observing the terminal result would violate the terminal-only
rule.

The raw terminal report SHA-256 is
`dcfbcbc9fb3e042d44c1bb7762479f005a24a989a96611b85b102c34f955fcc2`.
The terminal checkpoint SHA-256 is
`11ce6b1d3a858bdc07182ed935ff8450baba4202b66bb663a56840dc10214818`.
Closeout authenticated every retained exam/checkpoint/case reference, the parent and source/tag
bindings, the unchanged r0 evidence bundle, and the no-update report boundary. The run used about
381.70 MiB, remained under every storage cap, left roughly 215.5 GiB free on the T7, and left no
orphaned trainer.

The fresh `15_240_000`–`15_279_999` confirmation candidates remained unopened. No confirmation
evaluator ran, no fresh case was selected, both U2r roots became terminal evidence, and U3 remained
closed. The complete record is
[the U2r-r1 terminal result](results/v0.2-u2r-r1-stability.md).

### July 23, 2026 — U2-S freezes a matched mechanism study before implementation

U2r-r1 exhausted the declared “more unchanged experience” path. The next decision does not continue
that policy, choose its penultimate checkpoint, lower the 3.0 threshold, or open another protected
exam. It restarts at the last confirmed boundary before U2 and tests two possible causes of the
loop-tail failure.

The [prospective U2-S protocol](protocol-v0.2-u2s-stability-ablation.md) fixes a 2 × 2 design:

| Sequential arm | PPO rule | Repeated visible no-effect penalty |
| --- | --- | --- |
| `control` | Existing U2 PPO | Off |
| `conservative` | Linear `2.5e-4`→`2.5e-5`, clip 0.10, two epochs, target KL 0.015 | Off |
| `no-effect` | Existing U2 PPO | On |
| `combined` | Conservative PPO | On |

All four children must load the exact confirmed U1 `20260733` parent checkpoint
`3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104`
and its optimizer. No U2, U2r, or earlier ablation policy may be a parent. The arms deliberately
share algorithm seed `20260753`, worker streams `20260753`–`20260756`, the same static
lesson-specific guard inventory, and the same 1,048,576-action U2 budget. This makes them matched
development trajectories rather than four independent replications.

The comparison is prospectively non-resumable. A reset continuation of only one arm would change
its environment, recurrent, penalty, and RNG trajectory relative to the other factorial cells.
Any interruption therefore closes the entire canonical cohort as `operationally_incomplete`; a
replacement requires a new source commit, annotated tag, protocol-attempt identity, and root before
all four arms restart from the confirmed parent.

The reward intervention is deliberately narrow. On pickup, drop, or toggle, the wrapper compares
only the visible pixel bytes before and after the chosen action. A streak exists only while the same
interaction repeats and every transition leaves the visible pixels byte-identical. Its first
transition is unpenalized; the second and later transitions receive `-0.01`. A pixel change,
different action, movement action, reset, or terminal resets the streak. Added penalty is capped at
`-0.10` per episode. The rule receives no coordinate, object, inventory, milestone, oracle, path,
or trainer-only state; it never masks an action, stays out of observation, and is disabled during
evaluation.

Every arm must run all 32 fixed 32,768-action windows. Only exams at 983,040, 1,015,808, and
1,048,576 actions may decide eligibility, and all three must independently:

- pass the existing prerequisite overall and panel gates;
- score U2 at least 72/80 with both panels at least 34/40;
- hold U0, U1, and U2 mean ineffective interactions to at most 3.0;
- contain no case with ten or more ineffective interactions; and
- contain no identical pickup/drop/toggle action run of length ten or more.

The report also freezes mean, median, upper quantiles, maxima, tail counts, and worst-case
concentration so two catastrophic cases cannot hide inside a clean average again.

If several arms qualify, the simplest-intervention priority is fixed as control, conservative,
no-effect, then combined. The study selects only the learner configuration. It never carries an arm
checkpoint into a claim-bearing cohort or U3. No eligible arm ends this learner-family experiment;
an eligible configuration merely permits a new separately preregistered multi-lineage cohort from
confirmed U1 parents.

The prospective scientific root is
`/Volumes/T7 Developer/DungeonApprentice/u2s-ablation-20260723`, and the read-only dashboard is
assigned to port 8787. This entry records a design decision only. At this point it does not claim an
implementation, clean source freeze, external tag, qualification, root creation, policy action, or
training result. All U2 confirmation, unopened U2r confirmation, and project-final ranges remain
unavailable to the ablation.

### July 23, 2026 — U2-S attempt 0 fails safely before action one

The first qualified U2-S launch created its cohort at `19:03:50 UTC`, opened the control cell at
`19:04:16 UTC`, and closed it as `crashed` at `19:04:42 UTC`. The trainer created the scientific
`control` directory, then its initial storage audit correctly refused to proceed because the
matching media directory did not exist. The exact exception was:

`FileNotFoundError: storage directory does not exist: .../u2s-ablation-media-20260723/control`

This was an infrastructure failure, not a failed learning result. The attempt produced:

| Evidence | Attempt-0 outcome |
| --- | --- |
| Policy actions | 0 |
| Optimizer updates | 0 |
| Frozen exams | 0 |
| Safe checkpoints | 0 |
| Arm status | Never published |
| Cohort phase | `operationally_incomplete` |
| Scientific reuse | Forbidden |

The failed root, contract, cohort history, and supervisor exit evidence remain untouched under
`/Volumes/T7 Developer/DungeonApprentice/u2s-ablation-20260723`. The dashboard process was stopped
only after the terminal state and evidence were verified.

The full incident record, exact timeline, checksums, and scientific interpretation are preserved in
[U2-S launch attempt 0](results/v0.2-u2s-launch-attempt-0.md).

The r1 correction does not alter the learner, curriculum, rewards, seeds, budgets, evaluation
cases, decision rule, or four-arm order. It adds one fail-closed launcher step: after the manifest
opens an arm and before the trainer starts, create the arm's exact media directory non-recursively
and validate its ancestor chain. Because attempt 0 is non-resumable, r1 receives a new source
commit, annotated tag `u2s-stability-ablation-v0.2-u2s-r1-20260723`, qualification root, cohort
root `/Volumes/T7 Developer/DungeonApprentice/u2s-ablation-r1-20260723`, and four entirely fresh
children from the confirmed U1 parent.

### July 23–24, 2026 — U2-S r1 completes; no mechanism qualifies

The corrected matched ablation ran all four cells sequentially from the byte-identical confirmed
U1 `20260733` parent. Every arm authenticated the same pre-action RNG digest
`dba44c0603a35614fd5f7b56706fe6d5abf83e893a086eca6f2fbaa4d1428030`,
spent exactly 1,048,576 new U2 actions, completed all 32 fixed post-update exams, and exited
normally. Across the cohort that is 4,194,304 actions, 128 exams, and 40,960 deterministic case
records.

The fixed terminal-three-exam comparison was:

| Arm | U2 successes | Mean U2 ineffective | 10+ cases | Worst loop | Frozen outcome |
| --- | --- | ---: | ---: | ---: | --- |
| `control` | 78, 76, 77 | 1.1083 | 6 | 156 | Capability passed; tails failed |
| `conservative` | 64, 64, 64 | 3.2542 | 5 | 155 | Capability and tails failed |
| `no-effect` | 78, 79, 76 | 0.3500 | 2 | 126 | One clean exam; later tails failed |
| `combined` | 70, 69, 71 | 0.1167 | 0 | 2 | Stability passed; capability failed |

Control showed that the existing learner can retain strong U2 capability while remaining
deterministically brittle. The bounded pixels-only no-effect signal improved both average
capability and tail frequency relative to control, but one 28-action loop and one 126-action loop
made its second and third terminal exams ineligible. Conservative PPO weakened U2 learning.
Combining conservative PPO with the no-effect signal removed the measured catastrophic loops, but
held the weaker U2 panel below 34/40 and all three overall scores below 72/80.

No arm made capability and stability coexist under every frozen terminal condition. The cohort
verdict is **`ablation_failed`**, `selected_configuration` is `null`, ablation checkpoint reuse is
forbidden, no successor cohort is authorized, and U3 remains closed. The report SHA-256 is
`dfd288955bd2f8367ba3818e7242ff29c4a6e8e5a03d248df45e85f43b562e44`;
the full authenticated closeout is in
[the U2-S r1 result](results/v0.2-u2s-r1-stability-ablation.md).

This closes the “rescue it with more PPO or a small reward adjustment” branch. The useful lesson is
architectural: the policy needs an internal representation of action and visible consequence,
rather than a stronger external force telling it not to repeat an action. Any such experiment must
be separately preregistered, restart at an authorized confirmed boundary, and may not load an
U2-S checkpoint.

### July 24, 2026 — v0.3 Stage-A implementation reaches qualification boundary

The next experiment is now implemented as a complete release candidate rather than another
reward/PPO adjustment. Two fresh twins reconstruct the exact confirmed U1 `20260733` policy and
Adam state by parameter name. Both receive the same Dict observation and new bias-free `9 → 512`
residual; sham receives zeros, while the candidate receives only its own previous primitive action
and whether its next visible frame changed. The projection starts exactly at zero, preserving the
inherited policy at the intervention boundary.

The implementation now includes the observation wrapper, named policy/optimizer transplant,
bit-exact zero-context checks, a disposable matched four-worker rollout/update smoke, one-shot
qualification, two sequential full-budget trainers, a fail-closed cohort state machine, bounded
artifacts, a read-only dashboard, and a fixed non-resumable launcher. The runtime contract fixes
both scientific training and qualification smoke to CPU and records Python, platform, machine,
Gymnasium, MiniGrid, NumPy, sb3-contrib, Stable-Baselines3, and Torch identities.

The assigned release identities are:

| Boundary | Assigned identity |
| --- | --- |
| Annotated tag | `action-effect-architecture-v0.3-stage-a-20260724` |
| Qualification root | `/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.3-action-effect-stage-a-20260724` |
| Cohort root | `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-20260724` |
| Media root | `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-media-20260724` |
| Dashboard | `http://127.0.0.1:8788/` |
| Launcher | `scripts/run_v03_action_effect_stage_a.sh` |

This entry records implementation readiness only and intentionally predates source freeze. The
annotated tag, durable qualification claim/report, cohort, media root, policy actions, and learning
result do not yet exist. The eventual release must preserve a non-circular chain: the
published tag first preregisters source and contracts; qualification then binds that exact tag
object; the cohort manifest finally binds the qualification-report SHA-256.

The scientific question is narrower and stronger than “did another tweak help?” The matched first
rollout must prove that the new pathway initially changes nothing. Only after the first PPO update
may the candidate learn to use its own action/outcome memory. If it passes all three fixed terminal
exams, Stage A selects only the architecture definition for a separate three-lineage replication.
It never promotes a checkpoint and never opens U3 by itself.

### July 24, 2026 — v0.3 launch attempt 0 closes before action one

Attempt 0 subsequently froze source commit
`5b135a4e2953db9f14e83cdaba77fe219fecb160`, published annotated tag
`action-effect-architecture-v0.3-stage-a-20260724`, and completed its one-shot qualification with
verdict `qualified`. The report SHA-256 is
`a3a50ecf91411a27a70e2c6d3e03b93183aa3b078f2f1104b5dd6884f2c3fc85`.

The launcher created and sealed the fresh cohort contract, but its dashboard startup check exposed
an operational defect. The server repeated the full source, qualification, parent, and predecessor
authentication on every request. That work exceeded the launcher's ten-second HTTP deadline.
Three clients disconnected before their payload writes completed, leaving three
`BrokenPipeError` traces in the authenticated dashboard log.

No arm directory, supervisor, trainer, status, action, optimizer update, checkpoint, evaluation, or
media artifact was created. On exit, the launcher's fail-closed trap claimed the pending `sham`
transition only long enough to record a synthetic crash with exit code `130`; it sealed the whole
root as `operationally_incomplete`. The manifest's apparent sham attempt is therefore operational
closeout, not a model run. The scientific action and update counts are both zero.

The exact chronology, frozen digests, and negative inventory are in the
[attempt-0 closeout](results/v0.3-action-effect-launch-attempt-0.md). Its tag, qualification,
cohort, and media roots are immutable historical evidence and may never be resumed or reused.

### July 24, 2026 — v0.3 Stage-A r1 replacement prepared

> Historical preparation entry: this section records the boundary before r1 was released. The next
> entry records its later partial-sham operational failure and superseding r2 decision.

r1 keeps the complete Stage-A scientific contract fixed and changes only the dashboard boundary.
The process now performs the expensive full authentication once at startup, before binding its
socket. Each live poll still rereads mutable cohort/status evidence and rehashes the immutable
cohort contract, qualification report, and checksum. The launcher determines health by
authenticating expected identity fields; the API does not return a literal `healthy: true` field.

The fresh r1 assignments are:

| Boundary | Assigned r1 identity |
| --- | --- |
| Annotated tag | `action-effect-architecture-v0.3-stage-a-r1-20260724` |
| Qualification root | `/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.3-action-effect-stage-a-r1-20260724` |
| Cohort root | `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r1-20260724` |
| Media root | `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r1-media-20260724` |
| Dashboard | `http://127.0.0.1:8789/` |
| Launcher | `scripts/run_v03_action_effect_stage_a_r1.sh` |

This entry records the implemented replacement boundary, not a launch. At that historical point no
r1 tag, qualification claim/report, cohort, media root, dashboard, trainer, policy action, or
learning result existed. The then-governing prospective document was the
[v0.3 Stage-A r1 protocol](protocol-v0.3-action-effect-architecture-r1.md).

### July 24, 2026 — v0.3 Stage-A r1 reaches sham, then closes operationally incomplete

r1 subsequently froze clean source
`b9dc80b4c5f95b4b0a8bf7a681b9431764c6f0ec`, published annotated tag object
`1f11b98fae6691dc9282682c44cdaa33ea55e5ab`, and passed its one-shot qualification. The
qualification report SHA-256 is
`c59c3033a892bb05fea437bc525090c4f958110c960494fc2366444526d2dcb7`; the cohort contract
SHA-256 is `647f8237c2dfcb8451a6ccb644e2b6b10f1087bd9a5b5d7a3dcc0b23426fe22d`.
The corrected dashboard opened normally on port `8789`, sham transplanted the confirmed U1 policy,
and the first complete 2,048-transition rollout was captured before optimization.

The launch then exposed a new operational defect. The launcher used whole-command-line `pgrep -f`
matching to count trainers. Its supervisor command embeds the complete child trainer command after
`--`, so the supervisor and actual child both matched as neural trainers. At
`2026-07-24T04:56:53+00:00` the launcher logged
`v0.3 does not have exactly one neural trainer` even though the owned chain contained one
supervisor and one trainer.

The failed assertion returned from a shell function under zsh `set -e`. That error path bypassed
the intended `EXIT` cleanup, so the background chain continued after the error. Explicit
`SIGTERM` was forwarded at `05:02:58`; the manifest sealed the cohort
`operationally_incomplete` at `05:03:39`.

The exact surviving counters are:

| Evidence | r1 terminal value |
|---|---:|
| Sham trained child actions | `38,912` |
| Sham collected actions in final status | `40,004` |
| Sham collected actions in episode ledger | `40,960` |
| New optimizer updates / phases | `76 / 19` |
| Scheduled exam boundaries | `1 / 32` |
| Evaluation rows / deterministic cases | `8 / 640` |
| Latest safe checkpoint | none |
| Action-effect actions | `0` |

The final status preceded the last four-worker collection completion, which is why its collected
counter is 40,004 while the append-only episode ledger reaches 40,960. Only 38,912 transitions
were optimized and checkpointed. The one scheduled boundary was at 32,768 child actions; together
with the inherited baseline it produced four lesson rows and 320 cases per boundary.

The latest-observed checkpoint SHA-256 is
`0124df33809e0b81ccef181a98060617130ae12ba7eff116c8675fb3a27ac943`, but
`latest_safe_checkpoint` is null. No r1 checkpoint is resumable, promotable, reusable, or
selectable. The action-effect arm has no directory or attempt. Therefore r1 did not perform the
matched comparison and supplies no scientific evidence for or against the architecture.

The exact timeline, inventories, and hashes are preserved in the
[r1 operational-failure result](results/v0.3-action-effect-stage-a-r1-operational-failure.md).

### July 24, 2026 — v0.3 Stage-A r2 process-safe replacement prepared

r2 keeps the entire scientific contract fixed. It changes only the operational mechanism that
misclassified the supervisor and failed to clean up:

- process roles are classified from leading executable arguments rather than arbitrary text later
  in the command line;
- the supervisor's child command no longer makes the supervisor a trainer;
- critical assertions use explicit cleanup-and-terminalization branches instead of relying on
  implicit zsh `errexit`; and
- operational tests must exercise a real supervisor-plus-child chain and prove that abnormal exit
  leaves no trainer, supervisor, or `caffeinate` process.

Both r2 arms restart fresh from confirmed U1 child `20260733`; no r1 checkpoint, optimizer,
rollout, recurrent, environment, or scheduler state may be loaded.

| Boundary | Prospective r2 identity |
|---|---|
| Cohort ID | `v0.3-action-effect-stage-a-r2-20260724` |
| Annotated tag | `action-effect-architecture-v0.3-stage-a-r2-20260724` |
| Qualification root | `/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.3-action-effect-stage-a-r2-20260724` |
| Cohort root | `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r2-20260724` |
| Media root | `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r2-media-20260724` |
| Dashboard | `http://127.0.0.1:8790/` |
| Launcher | `scripts/run_v03_action_effect_stage_a_r2.sh` |

This is a preparation entry, not a release or launch claim. No final r2 source commit, tag object,
tag payload, qualification digest, cohort contract, policy action, or result exists yet. The
governing prospective document is the
[v0.3 Stage-A r2 protocol](protocol-v0.3-action-effect-architecture-r2.md).

### July 24, 2026 — v0.3 Stage-A r2 completes sham, then fails terminal authentication

r2 froze clean source `01b1b910edbb676f5de7375fa55d1b6e3bc6a54c`, published annotated
tag object `87d4ce7d24bd13d3a5e3e182c29889c36db35d87`, passed one-shot qualification
`148bb469cf66753b3ab998299e5f595c94d4fb868c5c38d93c8ec0e9cd097a9e`, and created
cohort contract
`4114f1d03e5d7e84eab9106676b6dcb9a8b2b0f2b8376a765f599dc61b1b49fa`.

Sham started at `05:50:42Z` and its trainer completed at `07:49:41Z`:

| Counter | Terminal value |
|---|---:|
| Child actions collected/trained | 1,048,576 / 1,048,576 |
| New / lifetime optimizer updates | 2,048 / 3,584 |
| Frozen exams | 32 / 32 |
| Terminal U2 scores | 77, 78, 78 / 80 |
| Terminal U2 panel splits | 39/38, 39/39, 38/40 |
| Terminal global maximum ineffective/repeated run | 3, 85, 9 / 3, 85, 8 |

The middle terminal exam's one 85-action pathological case made sham terminal-ineligible. Sham is
calibration only, so it could not select the architecture in any outcome.

At `07:50:00Z`, terminal manifest closeout rejected the first-rollout boundary. Qualification and
training hash compact sorted JSON followed by `\n`; the manifest's generic helper omitted that
byte. The intended profile reproduces the recorded `fa4c7bda…` aggregate, while the manifest
recomputed `3b9ecf3a…`. The cohort stopped `integrity_failed`. Action-effect remained pending with
zero attempts and no directory.

r2 is an operational failure, not an architecture result. Its report, checkpoint, and every root
are non-resumable, non-promotable, and unusable as parents. The
[exact r2 result](results/v0.3-action-effect-stage-a-r2-operational-failure.md) records the full
timeline, hashes, and terminal-three evidence.

### July 24, 2026 — v0.3 Stage-A r3 canonical-evidence replacement designed

r3 keeps the scientific design and every training variable fixed. It gives the first-rollout
identity one named byte profile, `u2s-canonical-json-v1-lf`, shared by qualification, trainer, and
manifest. Terminal closeout must authenticate the early envelope and require exact equality across
all later copies.

Both arms restart from confirmed U1 child `20260733` in fixed sham-then-action-effect order. r2
layouts are not added to the history guard, and no r2 checkpoint, optimizer, RNG, recurrent,
scheduler, or environment state may cross the boundary.

| Boundary | Prospective r3 identity |
|---|---|
| Cohort ID | `v0.3-action-effect-stage-a-r3-20260724` |
| Annotated tag | `action-effect-architecture-v0.3-stage-a-r3-20260724` |
| Qualification root | `/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.3-action-effect-stage-a-r3-20260724` |
| Cohort root | `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r3-20260724` |
| Media root | `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r3-media-20260724` |
| Dashboard | `http://127.0.0.1:8791/` |
| Prospective launcher | `scripts/run_v03_action_effect_stage_a_r3.sh` |

This is a documentation and design entry. The r3 source hash, tag object, qualification, checksums,
roots, actions, and result remain **TBD**. The governing prospective document is the
[v0.3 Stage-A r3 protocol](protocol-v0.3-action-effect-architecture-r3.md).

### July 24, 2026 — v0.3 Stage-A r3 completes with `architecture_failed`

r3 froze clean source `c4834b73dfed7d875c6f59887d3077319a305910`, published annotated tag
`action-effect-architecture-v0.3-stage-a-r3-20260724` at tag object
`fc5eb2f82896b7f6d41030f2776a0028a70ea4c2`, and passed its one-shot qualification. The
qualification report SHA-256 is
`07151fecea179dfaedabe56dcadc009d3750798987973c17cb67b1888909c499`; the architecture
and cohort contract SHA-256 values are
`4059eba189bee3cdb13d9ea29c9befc908209b937c2f21715cf9cd6b8dcbeb11` and
`fea7b7eb10accec2395105630ad790d550993996c5bd35a2a937fdb8837d5ade`.

The fresh cohort was created at `09:08:14Z`. Sham ran from `09:09:27Z` to `11:07:45Z`;
action-effect ran from `11:08:34Z` to `13:10:41Z`. Each arm completed exactly 1,048,576 child
actions, 2,048 new optimizer updates, 32 exams, and 10,240 deterministic cases. The full cohort
therefore produced 2,097,152 child actions, 64 exams, and 20,480 cases.

The complete pre-update 2,048-transition rollouts were identical at aggregate SHA-256
`fa4c7bda99a261f8fa49741a49360cd1bfc6ab3081db51aeffc64266a109ce72`, with no
divergence before the first optimizer phase. Sham's projection remained exact zero. The candidate
projection first became nonzero at 2,048 actions and ended with all 4,608 weights nonzero, weight
L2 norm 2.826899, update L2 norm 0.080078, and maximum absolute weight 0.177197.

| Boundary | Sham U2 | Sham tails | Action-effect U2 | Action-effect tails |
|---|---:|---:|---:|---:|
| Inherited baseline | 24/80 | 51 ten-plus cases; worst 160/160 | 24/80 | 51 ten-plus cases; worst 160/160 |
| 32,768 actions | 25/80 | 60; worst 160/160 | 27/80 | 58; worst 160/160 |
| 983,040 actions | 77/80 | 0; worst 3/3 | 70/80 | 6; worst 160/160 |
| 1,015,808 actions | 78/80 | 1; worst 85/85 | 72/80 | 2; worst 124/124 |
| 1,048,576 actions | 78/80 | 0; worst 9/8 | 75/80 | 1; worst 152/149 |

Sham was terminal-ineligible because its middle deciding exam contained the 85-action tail; it was
also calibration-only. Action-effect learned substantial capability and used the new pathway, but
its first deciding exam missed the 72/80 U2 floor and the mean-ineffective ceiling, and all three
deciding exams contained forbidden ten-plus ineffective/repeated-action tails. The sealed verdict
is `architecture_failed`, `selected_architecture` is null, checkpoint reuse is false, Stage B is
not authorized, and U3 remains closed.

Process closeout sealed at `13:11:35Z`; finalization completed at `13:12:08Z`. The sealed process
inventory and immediate final scan found zero trainer, supervisor, or `caffeinate` process. The
cohort report and integrity SHA-256 values are
`05f23509652e48a127be3728fb4ccb052b8285035792db9aaec7a956e021dcdb` and
`0e42c78e7a1b541d55c4286ccf606f1477a453914427cd74ebe59af8568b81a4`. Every
storage cap passed.

The immutable [r3 result](results/v0.3-action-effect-stage-a-r3.md) contains the complete
authentication table and claim limits. The evidence supported one smallest prospective successor,
now frozen as the
[v0.4 matched ineffective-trace architecture study](protocol-v0.4-ineffective-trace-architecture.md).
It replaces the action identity plus one-step outcome with one bounded duration scalar, computed
identically in both arms and exposed truthfully only to the candidate. Both arms restart from
confirmed U1; reward, PPO, curriculum, action budget, frozen cases, and selection rule remain
unchanged. The source/tag/qualification/audit gates still precede any canonical action.

### July 24, 2026 — v0.4 ineffective-trace release candidate built

The bounded root-cause analysis localized all nine action-effect terminal tails to near-pure
same-action/no-visible-change loops. The failed nine-value input stayed constant inside each loop
and returned the action identity itself, creating a plausible repetition shortcut. v0.4 therefore
replaces—not extends—that pathway with one scalar:
`min(consecutive same-action byte-identical RGB transitions, 9) / 9`.

Both fresh twins compute the same capped counter. `trace-sham` always exposes `0.0`;
`ineffective-trace` exposes the truthful scalar. Neither arm sees the action ID. Both reconstruct
independently from confirmed U1 child `20260733`, copy every legacy policy tensor and Adam moment
by parameter name, and add exactly one zero-initialized bias-free 1→512 residual. Reward, PPO,
curriculum, worker count, random streams, 1,048,576-action budget, 32 exams, terminal-three rule,
and non-promotable checkpoint boundary remain fixed.

The release candidate now includes the wrapper and evaluation adapter, matched trainer and real
four-worker smoke, fail-closed one-shot qualifier, full r3-tree authentication, fresh-only cohort
manifest, read-only dashboard on port 8792, one-shot sequential launcher, tests, protocol,
architecture visual, runbook, and video narrative.

The first audit cycle found real pre-release defects:

- the raw diagnostic counter was initially unbounded even though its encoded value was capped;
- the qualifier did not reject three important transplant-evidence mutations;
- process classifiers missed versioned trainer executables;
- an abnormal exit after both arms could leave an unintended closeout-resume path;
- the shared supervisor could orphan a just-created trainer if its first ownership-state write
  failed; and
- one table misstated the exact r3 pickup tail lengths.

Each issue was corrected prospectively with regression tests. In particular, the raw counter now
itself caps at nine; the qualifier requires all 20 inherited U1 parameter names and Adam states,
the sole new trace weight, exact parent counters, and every zero-context equivalence field; the
manifest has a durable `abort-closeout` transition; and the supervisor kills and reaps an
unpublished child process group.

At that point this entry recorded a release candidate, not a launched experiment. The source
commit, tag object, qualification report SHA-256, cohort contract, action counts, and outcome were
still unknown pending the non-circular release chain recorded immediately below.

### July 24, 2026 — v0.4 ineffective-trace Stage A completed

The non-circular release chain completed from clean source
`0d417255a0b344f4863045862adf0617c49f51bf`. The annotated tag
`ineffective-trace-architecture-v0.4-stage-a-20260724` resolved to tag object
`754278095226c324e3191e39c70d2297730f8573`; one-shot qualification passed with report SHA-256
`6d23e925b3b0fdaaf9e621c69ceb8bf8afd57c96ed33262958885bf9bd33beb4`, and the canonical cohort
contract was `756eab356d962e0ca12beb23d47092dfd8c53e7cb314121dd1b394e35ca27393`.

The user-context detached Screen session started `trace-sham` at `14:48:55Z`, completed it at
`17:01:06Z`, started `ineffective-trace` at `17:01:59Z`, and completed it at `19:15:35Z`.
The cohort sealed a zero-orphan process closeout at `19:16:29Z` and finalized
`architecture_failed` at `19:17:02Z`. Each arm reached exactly 1,048,576 child actions,
1,835,008 lifetime actions, 2,048 new optimizer updates, 3,584 lifetime updates, 32 frozen exams,
and 10,240 deterministic cases. All 64 scheduled checkpoint, sidecar, integrity, and case bundles
authenticated.

The twins had the same initial RNG aggregate
`2b343c0d3454edc5a74354ed20bc6f69110e2ceace9083de9a18eed22043e846`
and the same complete 2,048-transition pre-update rollout:

| Matched-start evidence | SHA-256 |
| --- | --- |
| First-rollout aggregate | `e90a548764bc2bb490176a0e8d166d629f97f8e5a1dc008f3e90fae04a3f92d7` |
| Policy output | `60def45030935347f6527539a68cdab763073752415e0980d0a432415ba3de68` |
| Normalized episode ledger | `51a6cfe24f1869d8b832000f5826a2c99d98f369512e198f177a155efb4fe079` |
| Post-rollout RNG aggregate | `dbc2b81ed917fa6810463615817c0449e8b61e287387bf2a36479e739a7f0a6c` |

There was no divergence before the first optimizer phase. Sham's encoder remained exactly zero.
The candidate encoder first became nonzero at 2,048 actions and ended with all 512 weights nonzero,
weight L2 norm `1.078283`, and update L2 norm `0.023001`. Its trace was nonzero on 123,552 of
1,048,576 transitions (11.7828%). The intervention was exercised and learned; this was not a dead
input or failed transplant.

| Boundary | Trace sham U2 | Sham 10+ / worst ineffective-repeat | Candidate U2 | Candidate 10+ / worst ineffective-repeat |
| --- | ---: | ---: | ---: | ---: |
| Inherited baseline | 24/80 | 51 / 160-160 | 24/80 | 51 / 160-160 |
| 32,768 actions | 28/80 | 46 / 160-160 | 27/80 | 43 / 160-160 |
| 983,040 actions | 77/80 | 0 / 5-5 | 79/80 | 4 / 153-152 |
| 1,015,808 actions | 78/80 | 0 / 1-1 | 78/80 | 1 / 157-156 |
| 1,048,576 actions | 76/80 | 0 / 7-7 | 79/80 | 4 / 121-121 |

Sham passed every terminal behavioral check, but the protocol defined it as calibration-only and
nonselectable. The candidate averaged 78.67/80 U2, higher than sham's 77.0/80, and passed the
capability, panel, prerequisite-retention, and mean-ineffective checks. It failed the reliability
conjunction at all three deciding exams. Each boundary violated all three tail requirements:
no ten-plus ineffective case, maximum ineffective interactions below ten, and maximum repeated
identical interaction run below ten. Those are the nine frozen failed checks. The duration signal
improved ordinary behavior without eliminating rare catastrophic loops.

The sealed verdict is:

```text
architecture_failed
selected_architecture: null
development_checkpoint_reuse_authorized: false
replication_protocol_authorized: false
u3_authorized: false
```

The cohort report and integrity SHA-256 values are
`da966107e0c846c5b797ea0153805a159c0a59adced8ddf8f8affe07c47406f6` and
`dfbbdd7a4b50466c4f24de24682e6d084decd47e43b4c230180f13bb86436fa8`.
The exact [terminal result](results/v0.4-ineffective-trace-stage-a.md) is read alongside the
prospectively frozen [v0.4 protocol](protocol-v0.4-ineffective-trace-architecture.md). The macOS
operational lesson is distinct from the science: detached user-context Screen preserved T7 access
and launcher continuity through application restarts, while `caffeinate` prevented sleep. The
Screen launcher ended normally after closeout. No v0.5 protocol or cohort is authorized, and U3
remains closed.
