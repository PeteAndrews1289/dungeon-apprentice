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
