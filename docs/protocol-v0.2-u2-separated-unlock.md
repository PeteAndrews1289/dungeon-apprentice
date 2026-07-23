# Protocol v0.2 U2: Separated Unlock

> Status: frozen implementation and execution protocol. The immutable U1 successor confirmation
> satisfied the activation gate on July 23, 2026, and the disposable U2 engineering sandbox passed
> its 1,000-map acceptance sweep. At the source-freeze boundary, the one-shot protected
> qualification, development validation, future confirmation, and final-test ranges remained
> unopened and no U2 child had trained. Runtime truth after launch belongs to the canonical
> qualification ledger, external Git tag, cohort manifest, and child status records—not to a
> retrospective edit of these rules.

## Question and claim boundary

Can each independently learned and confirmed Local Unlock policy add a longer, less visually
convenient key → locked door → exit behavior while retaining Navigate, Visible Unlock, and Local
Unlock?

This is a cumulative-learning experiment. Every U2 child inherits one complete U1 recurrent-PPO
archive, including optimizer state, and learns only from its own new interaction experience. It
does not ask whether U2 can be learned from random initialization. It does not establish Full
Unlock, Retrieve, puzzle solving in general, planning in arbitrary worlds, or readiness for a
commercial multi-character game.

The narrow positive claim is:

> The unchanged pixel-only learner repeatedly added the preregistered Separated Unlock distribution
> to three independent confirmed U1 lineages while retaining every earlier frozen lesson.

That claim requires all three lineages to satisfy the complete rule below. Training episodes,
milestone counts, pooled evaluation episodes, or one successful child cannot substitute for it.

## Satisfied activation gate

U1 development scores alone did not authorize U2. The canonical successor report now exists at:

`/Volumes/T7 Developer/DungeonApprentice/confirmations/v0.2-u1-v2-20260723/report.json`

Its externally measured SHA-256 is
`6e577170050f6f14599b793a031776a19bf7c64eba0f243f457298da3193ae8f`.
The completed exclusive-attempt ledger records source commit
`ebf064afd6e6296bb21524103c2a3c269a56e7a5`, evaluator and launcher exit status zero, and the same
report digest.

The authenticated report:

- has the frozen U1 successor protocol and a `confirmed` verdict;
- records `policy_updates: false`;
- contains all three exact first-mastery U1 archives below;
- shows that every archive independently passed Navigate, U0, and U1 overall and panel gates; and
- has a matching external checksum and completed exclusive-attempt ledger.

That evidence activates design and implementation work under this frozen boundary. It does not
pre-approve engineering acceptance or training. Any later mismatch in the report, checksum,
ledger, or parent entries stops U2 before a child directory or policy update is created. This
provenance binding may not be used to change U2 geometry, seeds, budgets, gates, recovery,
optimization, or the replication rule after U2 evidence is observed.

## Frozen parent and child lineages

Each child inherits exactly one first-mastery U1 archive. A later checkpoint, best-looking policy,
weights-only export, cross-lineage optimizer, or replacement parent is prohibited.

| Order | Confirmed U1 parent | Parent training lineage | Parent archive | Parent SHA-256 | U2 child seed | Four worker streams |
| ---: | ---: | --- | --- | --- | ---: | --- |
| 1 | `20260725` | U0 `20260725` → U1 `20260725` | `/Volumes/T7 Developer/DungeonApprentice/u1-local-20260722/v02-u1-lead-seed-20260725/checkpoints/mastered-local-unlock.zip` | `bcce9b8251e97ed4fddda32871c891c3783c057bbb1f89deedb3a3d32058102a` | `20260737` | `20260737`–`20260740` |
| 2 | `20260729` | U0 `20260726` → U1 `20260729` | `/Volumes/T7 Developer/DungeonApprentice/u1-local-replication-20260722/v02-u1-replication-seed-20260729/checkpoints/mastered-local-unlock.zip` | `2a300927b48f966d5f6ddfeefe13d2e444da1e5c70bcd54e86abd6a9b2d1830b` | `20260741` | `20260741`–`20260744` |
| 3 | `20260733` | U0 `20260727` → U1 `20260733` | `/Volumes/T7 Developer/DungeonApprentice/u1-local-replication-20260722/v02-u1-replication-seed-20260733/checkpoints/mastered-local-unlock.zip` | `3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104` | `20260745` | `20260745`–`20260748` |

The protocol ID is `dungeon-apprentice-v0.2-u2`; its checkpoint sidecar schema is 4. The
runner must re-verify each parent archive, mastery sidecar, manifest, clean source, optimizer
boundary, allocation history, supporting-artifact digests, U0 ancestry, and U1 confirmation entry
before creating a child directory.

The initial deterministic N/U0/U1 baseline must exactly reproduce the parent's selected mastery
boundary:

| U1 parent | Navigate | U0 Visible Unlock | U1 Local Unlock |
| ---: | ---: | ---: | ---: |
| `20260725` | 74/80 | 80/80 | 72/80 |
| `20260729` | 77/80 | 80/80 | 72/80 |
| `20260733` | 75/80 | 80/80 | 72/80 |

A mismatch is an engineering or provenance failure, not permission to choose another parent.

## Information, action, and reward boundary

The U2 policy receives exactly:

- the existing `56 × 56 × 3` egocentric partial RGB image; and
- its private 256-unit recurrent state carried within an episode.

It does not receive lesson ID, coordinates, direction numbers, map, visibility flags, seed,
objective text, object labels, oracle distance, oracle action, action mask, milestone state,
training curriculum state, or an online model/LLM response. The action space remains the same seven
primitive MiniGrid actions. No demonstration, trajectory, recurrent state, replay buffer, or policy
output is transferred between lineages.

Every lesson retains success reward `+1.0`, ordinary step reward `-0.001`, and the existing bounded
pixel-novelty curiosity of `+0.002` with a hard `0.1` episodic budget. Key pickup and door opening
remain diagnostics and receive no authored reward. Curiosity is disabled during every baseline,
validation, and later confirmation case. Success still means completing the entire lesson; U2 does
not terminate or pay merely for finding the key or door.

At U2's 160-step horizon, a maximally novel failed timeout still has negative undiscounted combined
return: `0.1 - 0.160 = -0.060`. Tests must also prove that the latest permitted success has greater
discounted return than failure at gamma `0.995`. A timeout is a terminal failed quest and is never
treated as a bootstrap truncation.

## Lesson registry

The child expands the immutable registry from three lessons to four. Earlier generators and pixels
must remain byte-for-byte compatible with their frozen sources.

| Lesson | U2 role | Generator | Horizon | Frozen validation cases |
| --- | --- | --- | ---: | --- |
| `navigate/full` | retained anchor | exact v0.1 Navigate distribution | 128 | `10_000_000`–`10_000_079` |
| `unlock/u0-visible` | retained anchor | exact confirmed Visible Unlock profile | 128 | `11_000_000`–`11_000_079` |
| `unlock/u1-local` | retained anchor | exact confirmed Local Unlock v1 profile | 128 | `11_100_000`–`11_100_079` |
| `unlock/u2-separated` | active lesson | Separated Unlock v1, defined below | 160 | `11_200_000`–`11_200_079` |

Golden tests must compare the U2 runtime's Navigate, U0, and U1 layouts, encoded grids, initial
images, actions, rewards, termination, and oracle results with their frozen implementations across
fixed seeds. U2 may add a registry entry; it may not silently redefine a prerequisite.

## Exact boundary of Separated Unlock v1

Every U2 case is a 9 × 9 map with a 7 × 7 interior, empty starting inventory, one matching colored
key, one locked door, and one green goal. It preserves the complete ordered objective:

1. acquire the matching key;
2. open the locked door; and
3. cross to and reach the goal.

The locked door occupies the sole opening in a complete horizontal or vertical divider. The start
and key are on the approach side and the goal is on the far side. In addition to the divider, the
map contains exactly two valid interior wall cells. Those walls may lengthen or redirect travel but
may not overlap an object, start, door, border, or each other; create an unreachable required
object; create an alternate divider crossing; or remove the need to unlock the door.

Rotation, reflection, divider position, door lane, approach side, start, orientation, goal, key
position, key color, and valid extra-wall placement vary by seed. Start visibility is not an
acceptance shortcut: neither the key nor the door is guaranteed visible or hidden. Qualification
reports all four key-visible/door-visible strata, but visibility proportions do not select cases or
change reward.

A candidate is valid only when:

- a pure full-state planner finds the ordered key → door → goal solution;
- a separately executed live oracle completes the same objective;
- pure-planner and live-oracle action counts match;
- the complete solution is 17–26 actions inclusive;
- at least one turn occurs after key acquisition and before opening the door;
- opening the locked door requires possession of the matching key;
- the goal remains unreachable while the door is locked;
- the policy image and action/reward/termination contracts remain unchanged; and
- generation succeeds within a bounded rejection limit without relaxing any condition.

The episode horizon is 160. The planner and oracle are generator instruments only. Their action
sequences are never supplied to the policy, used as targets, added to experience, or stored as
demonstrations. Qualification may retain seeds, hashes, lengths, and pass/fail evidence, but not an
oracle-action dataset for learning.

This boundary intentionally changes geometry and distance, not mechanics or observation. Decoy
keys, multiple colors at once, inventory choice, enemies, levers, text, shaped milestone rewards,
larger maps, and full unrestricted Unlock remain outside U2.

## Seed and evidence partition ledger

Numerically disjoint seeds do not guarantee distinct generated layouts. U2 therefore records both
seed-role separation and exact-layout hashes.

| Role | Frozen allocation | May influence learning or design? |
| --- | --- | --- |
| Training layout seeds | `0`–`999_999` | Yes; policy experience only |
| U2 generator engineering sandbox | `5_200_000`–`5_200_999` | Generator engineering only; never a claim |
| Sealed U2 preflight qualification | `5_210_000`–`5_211_999` | Oracle qualification only |
| Navigate validation | `10_000_000`–`10_000_079` | Automatic retention |
| U0 validation | `11_000_000`–`11_000_079` | Automatic retention |
| U1 validation | `11_100_000`–`11_100_079` | Automatic retention |
| U2 validation | `11_200_000`–`11_200_079` | Automatic U2 mastery |
| Future U2 confirmation: U2 candidates | `15_200_000`–`15_209_999` | Unopened until a separate preregistration |
| Future U2 confirmation: Navigate candidates | `15_210_000`–`15_219_999` | Unopened until a separate preregistration |
| Future U2 confirmation: U0 candidates | `15_220_000`–`15_229_999` | Unopened until a separate preregistration |
| Future U2 confirmation: U1 candidates | `15_230_000`–`15_239_999` | Unopened until a separate preregistration |
| Complete-project final test | `20_000_000`–`20_299_999` | Never during U2 |

Algorithm and worker seeds in the lineage table choose independent policy, optimizer, scheduler,
and reset streams; they are not layout seed roles. Training layouts remain inside the existing
sub-million partition. Each training environment must reject an exact layout hash reserved by the
sealed U2 qualification or any of the four validation suites. Geometry-only repetition is recorded
but is not silently converted into an exact collision.

The engineering sandbox may be inspected while implementing the generator. It is permanently
ineligible for validation, confirmation, or final claims. The sealed qualification and new U2
validation ranges remain unopened until the generator, oracle, observation, reward, and evidence
code is complete on a clean commit. If sealed qualification fails, no training begins: the failure
is recorded and any successor must declare new sealed ranges before inspecting them.

The four 15.2-million candidate streams are reservations, not a post-training confirmation plan.
No case may be generated from them until the three first-mastery U2 checkpoints are selected and a
separate, committed, collision-aware confirmation protocol freezes acceptance, panels, gates,
integrity checks, and no-update protections. The 20-million final partition is not a substitute for
that confirmation.

## Engineering acceptance before training

Implementation work may begin only after the U1 activation gate. Before the first policy update,
all of the following must pass from a clean committed source:

1. Separated Unlock generator and pure-planner property tests across rotations, divider directions,
   walls, visibility states, colors, and bounded rejection.
2. Pixel/action/reward/termination golden tests for all three inherited lessons.
3. A no-lesson-metadata observation test and an explicit seven-action-space test.
4. Reward-invariant tests for every horizon, curiosity cap, timeout, and latest success.
5. Deterministic transition-deficit scheduling tests for four lessons and every recovery profile.
6. Full save/reload/resume tests for schema 4, including optimizer, recurrent architecture,
   curriculum, scheduler RNG, window counts, parent lineage, and child stream.
7. Post-update evaluation/checkpoint-order tests proving that an exam scores the exact digest named
   in its record.
8. Deliberate interruption tests proving completed optimizer phases are durably resumable while
   genuinely partial collection/optimization is discarded and never labeled trained.
9. Storage-cap, rolling-retention, T7 mount, free-space, and exclusive-run-directory tests.
10. A short forced-lesson engineering smoke on sandbox seeds. It proves plumbing only and is not
    capability evidence.

After those checks, the sealed 2,000-case U2 preflight qualification opens once. It requires:

- 2,000/2,000 generated and live-oracle-completed cases;
- the exact divider, two-extra-wall, ordered-mechanics, horizon, observation, reward, and
  17–26-action contracts above;
- matching pure-planner/live-oracle action counts in every case;
- at least 1,980/2,000 unique color-inclusive exact layouts (99%);
- at least 1,900/2,000 unique geometry-only layouts (95%);
- zero exact-layout overlap with the U2 validation suite; and
- an immutable report containing every seed, exact-layout hash, geometry hash, action count,
  visibility stratum, failure, source commit, and generator profile.

A qualification failure is an engineering result, not evidence that a policy cannot learn U2. The
source and report must be frozen before training. No behavior-affecting generator, oracle, reward,
observation, scheduler, or evaluation change is allowed between qualification and the cohort.

The canonical qualifier has no seed, count, path, or rerun override. Before protected access it
atomically claims the fixed attempt with a fresh 256-bit launcher token. A passed report is bound to
its exact bytes, attempt, claim, generator profile, and clean source commit by the fixed annotated
Git tag `u2-preflight-v0.2-u2-20260723`, pushed to the declared GitHub origin. Training independently
verifies that exact remote tag object, report digest, attempt identity, and claim identity. It uses
the already-authenticated in-memory report snapshot and seed capability; it may not reread a mutable
report path or mint validation access from a digest string.

## Optimization and cumulative inheritance

Each child loads its parent's entire PPO archive and optimizer. Architecture and optimization remain
unchanged:

- recurrent PPO on CPU with four workers;
- 512-step rollouts, so every update consumes 2,048 complete collected transitions;
- batch size 256 and four epochs;
- learning rate `0.00025`;
- gamma `0.995` and GAE lambda `0.98`;
- entropy coefficient `0.01`; and
- the existing convolutional encoder and one-layer, 256-unit LSTM.

Collection, optimization, checkpoint publication, and evaluation remain separate ordered phases.
Collected actions become trained actions only after all PPO epochs finish. Any due checkpoint and
sidecar are published after optimization; an exam then names and hashes those exact trained bytes.
At most one curriculum-state transition may occur at one boundary.

## Transition-balanced practice and recovery

Lesson assignment targets actual transitions, never episode counts. Normal U2 practice is:

| Lesson | Normal target |
| --- | ---: |
| Navigate | 50% |
| U0 Visible Unlock | 7.5% |
| U1 Local Unlock | 7.5% |
| U2 Separated Unlock | 35% |

Assignment changes only between complete episodes. Every 32,768-new-action window must finish
within five percentage points of every active target. The sidecar persists lifetime and current
window counts, target profile, deficit state, and scheduler RNG.

All three prerequisites are evaluated at every boundary. If any prerequisite misses either its
overall or panel gate, U2 mastery counting stops, the active streak resets to zero, and the next
complete window uses the exact profile for the currently weak set:

| Weak prerequisites | Navigate | U0 | U1 | U2 |
| --- | ---: | ---: | ---: | ---: |
| Navigate only | 70% | 10% | 10% | 10% |
| U0 only | 45% | 35% | 10% | 10% |
| U1 only | 45% | 10% | 35% | 10% |
| Navigate + U0 | 55% | 25% | 10% | 10% |
| Navigate + U1 | 55% | 10% | 25% | 10% |
| U0 + U1 | 40% | 25% | 25% | 10% |
| Navigate + U0 + U1 | 45% | 22.5% | 22.5% | 10% |

The weak set is recomputed only from a frozen exam. Recovery ends after two consecutive boundaries
where all three prerequisites pass and every intervening recovery window is allocation-valid.
Recovery exams cannot count toward U2 mastery. Normal practice resumes with a zero-length U2
mastery streak. A weak U2 score by itself is not a retention failure and keeps normal practice.

Training never rolls back to a better-looking checkpoint. Recovery changes only future practice
frequency; it does not change reward, gates, optimizer state, or the active lesson.

## Frozen exams and mastery

Before update one, every child receives a four-lesson diagnostic baseline. The three inherited
counts must match the parent table above; U2 measures immediate transfer but cannot satisfy a gate.
Post-update exams run every 32,768 newly trained child actions. Each lesson uses its fixed 80-case
validation suite split into panels A and B of 40 cases.

| Lesson | Overall gate | Panel A | Panel B |
| --- | ---: | ---: | ---: |
| Navigate | at least 68/80 (85%) | at least 34/40 (85%) | at least 34/40 (85%) |
| U0 Visible Unlock | at least 68/80 (85%) | at least 32/40 (80%) | at least 32/40 (80%) |
| U1 Local Unlock | at least 68/80 (85%) | at least 32/40 (80%) | at least 32/40 (80%) |
| U2 Separated Unlock | at least 68/80 (85%) | at least 32/40 (80%) | at least 32/40 (80%) |

A child masters U2 only after two consecutive post-update **normal-practice** boundaries at which:

- all four lessons pass overall and both panels;
- the just-completed normal transition window is within allocation tolerance;
- no recovery is active;
- the evaluated archive and sidecar pass their digest and provenance checks; and
- the two passes are separated by one full 32,768-action training window.

The first qualifying pass produces a permanent candidate artifact. The second produces
`mastered-separated-unlock.zip` and a matching immutable sidecar. Training success, reward, loss,
milestone rates, oracle efficiency, one panel, or an unchanged terminal remeasurement cannot
advance the policy.

## Action, storage, and stopping ceilings

Each child receives at most **1,048,576 new trained actions**. The cohort ceiling is therefore
3,145,728 new trained actions. Inherited lifetime actions are reported but do not reduce a child's
declared U2 budget. A resume receives only the unspent remainder; it cannot reset the ceiling.

Early stop is allowed only at the complete U2 mastery gate. A flat, favorable-looking, or
discouraging curve runs to its action ceiling. All three lineages run sequentially on the audited
8 GB M1 unless an engineering failure invalidates the cohort. A child result never determines
whether a later valid lineage is run.

Scientific artifacts are bounded as follows:

- maximum 2 GiB per lineage run directory and 6 GiB for the three-lineage cohort;
- at most five ordinary rolling checkpoints, plus initial, first-pass, mastery, terminal, and crash
  artifacts required to explain decisions;
- bounded latest and exam frames rather than unbounded per-step video;
- append-only JSONL evidence with no raw observation or oracle-action replay archive;
- a 25 GiB free-space refusal reserve checked before launch, rollout collection, and checkpoint
  publication; and
- optional narrative screen capture in a separate, non-training media directory capped at 10 GiB.

The maximum planned incremental footprint is therefore 16 GiB including optional media. Recoverable
rolling checkpoints may be pruned by the declared retention policy; named decision artifacts,
ledgers, manifests, reports, and checksums may not be silently deleted to stay under cap. A storage
stop is an operational result, not a negative learning result.

## Resume, interruption, and unattended operation

Every archive has a matching schema-4 sidecar containing:

- exact U1 parent and confirmation provenance;
- child algorithm and worker streams;
- collected, trained, child, and lifetime actions plus optimizer updates;
- four-lesson curriculum and mastery/recovery state;
- transition counts, active target profile, allocation result, and scheduler RNG;
- PPO, observation, reward, generator, qualification, validation, and gate configuration;
- source commit and supporting-artifact digests; and
- archive SHA-256.

Resume creates a new recorded segment, verifies every structural field and digest, restores the
scheduler and optimizer, uses a distinct declared segment RNG stream, and preserves the remaining
cumulative action/storage budgets. It never borrows a newer dashboard state. A partially collected
rollout after interruption is discarded and reported; every completed 2,048-action optimizer phase
atomically replaces one bounded `latest-safe` resume bundle, so completed learning between larger
32,768-action exams cannot be lost or charged twice.

The sequential launcher must refuse a dirty or changing source, missing T7, less than the free-space
reserve, an existing target directory, a non-confirmed parent, wrong baseline, occupied dashboard
port, or a concurrent neural trainer. It uses `caffeinate -ims` for unattended macOS operation and
must leave a durable terminal/crash record even if the dashboard disappears.

## Evidence, dashboard, and narrative record

The three sequential children share one read-only dashboard endpoint at
`http://127.0.0.1:8785/`. Closing the browser cannot stop training. The dashboard and append-only
records must show:

- active lineage, exact parent/child seeds and parent digest;
- wall-clock start, last heartbeat, elapsed time, actions per second, trained actions, remaining
  action budget, and storage used/remaining;
- inherited versus newly trained and lifetime actions;
- four simultaneous deterministic success curves with counts and Wilson intervals;
- panel A and B counts for every lesson;
- current normal/recovery profile, weak prerequisites, target versus realized transition shares,
  recovery entry/exit, and mastery streak;
- key-pickup → door-open → exit funnels for U0, U1, and U2;
- `door_given_key` and `success_given_door` conversion rates;
- U2 results stratified by initial key/door visibility as non-gating telemetry;
- path actions divided by oracle actions, coverage, collisions, ineffective interactions, action
  histogram, largest action share, and longest repeated-action run;
- extrinsic and curiosity returns separately;
- PPO entropy, approximate KL, clip fraction, explained variance, policy loss, and value loss;
- exact evaluated checkpoint path/SHA, source commit, and qualification digest; and
- a policy-view frame per lesson, clearly labeled training or frozen exam.

The narrative record must preserve failures and recoveries, not only the final curve. The central
visual comparison is:

1. U0: key and door visible;
2. U1: key visible and door hidden;
3. U2: visibility unconstrained, path longer, and two extra walls;
4. four retained-skill curves rising or falling together; and
5. the automatic practice mix responding when an earlier capability weakens.

Oracle footage, if shown, is labeled “generator qualification only” and never presented as agent
behavior. Percentages accompany primary counts; thousands of episodes from one policy are never
described as independent replications.

## Three-lineage decision rule

Each lineage is independently:

- **positive** if it reaches the full U2 mastery rule before its action ceiling with valid
  qualification, parentage, allocation, no-update exams, storage, and artifacts;
- **transfer without mastery** if U2 rises above baseline but misses mastery at the ceiling;
- **retention failure** if a prerequisite cannot regain its frozen gate through the declared
  recovery controller;
- **U2 learning failure** if U2 remains below 20/80 after 524,288 new actions and does not later
  master, reported as a diagnostic subtype rather than an early-stop rule; or
- **engineering failure** if source, generator, qualification, lineage, scheduler, allocation,
  checkpoint, evaluation, resume, storage, or evidence integrity fails.

Separated Unlock is replicated only with **three positive lineages out of three**. Two positives are
mixed evidence; one remains a single-lineage development signal; zero is a replicated negative
within this learner, distribution, and budget. Engineering-invalid lineages are not silently
replaced or counted as learning failures. Evaluation episodes are reported per policy and never
pooled into a synthetic 240-case or 960-case agent.

No behavior-affecting correction is allowed after the first child receives update one. A defect
invalidating shared behavior invalidates the cohort and requires a documented successor protocol,
new child streams, and new sealed U2 qualification/validation roles.

## Post-training confirmation and untouched final test

After all valid lineages finish, their first-mastery checkpoints and digests are frozen before any
15.2-million candidate is opened. A separate preregistered no-update confirmation must:

- qualify and select cases without policy behavior;
- exclude declared exact reference layouts prospectively;
- evaluate all retained lessons and U2 on fixed accepted panels;
- preserve before/after policy, optimizer, counter, and checkpoint digests; and
- require every selected lineage to pass every frozen lesson and panel gate.

This document reserves candidate ranges but deliberately does not invent that later confirmation's
case count, diversity floors, or pass thresholds. Those choices must be committed after the U2
checkpoints are selected but before a reserved candidate is generated.

The complete-project final allocation at `20_000_000`–`20_299_999` remains untouched throughout
U2 engineering, training, replication, debugging, reporting, and post-training confirmation. U2
success does not authorize opening it. The final suite is reserved for the eventual complete
Navigate–Unlock–Retrieve system after all intermediate protocols are frozen.

## What a result authorizes

- A 3/3 positive cohort plus a positive disjoint U2 confirmation authorizes **design work** on U3
  Full Unlock. It does not authorize claiming that U3 is already learned.
- Mixed or negative capability evidence authorizes analysis and a separately declared successor,
  not threshold changes, checkpoint reselection, extra actions, or favorable-case filtering.
- Engineering failure authorizes repair and rerun only under a new clean source/protocol record;
  it does not answer the learning question.
- No U2 outcome authorizes Retrieve, new mechanics, milestone shaping, a different observation, an
  online model, or use of the final partition without another explicit protocol.

The discipline is the point: U2 should reveal whether the same learned ritual survives greater
spatial separation, not whether enough unrecorded interventions can eventually make a curve rise.
