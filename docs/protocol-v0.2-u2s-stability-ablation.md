# Protocol v0.2 U2-S: Matched Interaction-Stability Ablation

> **Status:** prospective development protocol. The design below records the decision after the
> valid U2r-r1 terminal stability failure and before any U2-S policy action. Implementation,
> qualification, source commit, external tag, canonical root creation, and training are not
> claimed by this document.

## Decision in one sentence

Train four matched, full-budget U2 children from the same frozen confirmed U1 parent and cross
unchanged versus conservative PPO updates with absence versus presence of a pixels-only repeated
no-effect penalty; compare only the fixed terminal three-exam windows; select the simplest eligible
**learner configuration**, never an ablation checkpoint.

![U2-S matched interaction-stability ablation](assets/v0.2-u2s-stability-ablation.svg)

This is a development mechanism study. It is not U2 confirmation, another U2r continuation, a
fourth replacement confirmation lineage, or U3.

## Why this protocol exists

The corrected U2r-r1 run completed its exact budget without an engineering fault. Its penultimate
and terminal U2 scores were both 76/80 with panels 37/40 and 39/40. Yet mean ineffective
interactions rose from 0.1875 to 4.1625 on the terminal exam.

That regression was concentrated:

- two terminal U2 cases produced 299 of all 333 ineffective interactions;
- one repeated toggle 145 times after the key and door milestones;
- one repeated pickup 153 times without changing the visible situation; and
- a Local Unlock case separately produced a 41-action interaction run while its lesson mean
  remained below 1.0.

The [authenticated terminal result](results/v0.2-u2r-r1-stability.md) therefore distinguishes two
questions that earlier experiments had combined:

1. **Update stability:** are full-strength PPO updates pushing a capable deterministic policy into
   narrow behavioral loops?
2. **Missing experiential feedback:** does the learner need a small consequence for repeating an
   interaction that visibly does nothing?

A matched 2 × 2 design can identify whether either intervention, or their combination, is worth
carrying into a fresh multi-lineage U2 study. Another continuation from a favorable U2 or U2r
checkpoint could not answer that question cleanly.

## Claim boundary

This ablation may support only:

> Under one frozen confirmed U1 parent and one matched set of random streams, configuration X was
> the simplest tested learner that satisfied the fixed U2 capability and interaction-stability
> development rule.

It cannot support:

- independent replication across U1 ancestries;
- held-out confirmation;
- U3 activation;
- a claim that any ablation checkpoint is a confirmed U2 policy;
- a causal population estimate from one arm per factorial cell;
- use of an arm checkpoint as the parent of a later claim-bearing cohort; or
- revision of the immutable U2 or U2r-r1 verdicts.

Intermediate exams are diagnostics. Development validation suites influence the fixed terminal
decision and therefore cannot later be presented as fresh generalization evidence.

## Evidence that remains immutable

The original U2 confirmation remains `capability_failed`: children `20260737` and `20260741`
passed, while `20260745` completed 169/200 Separated Unlock cases against the frozen 170/200
requirement.

U2r-r1 remains a valid terminal stability failure:

| Evidence | Frozen identity |
| --- | --- |
| U2r-r1 source | `86d3424a631a661966c763da3e1fabf27df798f1` |
| U2r-r1 tag | `u2r-stability-v0.2-u2r-r1-20260723` |
| Terminal report SHA-256 | `dcfbcbc9fb3e042d44c1bb7762479f005a24a989a96611b85b102c34f955fcc2` |
| Terminal checkpoint SHA-256 | `11ce6b1d3a858bdc07182ed935ff8450baba4202b66bb663a56840dc10214818` |
| Scientific verdict | `failed` |
| Fresh U2r confirmation opened | No |

No U2-S result may reinterpret the favorable U2r-r1 penultimate exam as selected evidence, resume
either U2r root, or describe the mechanism study as completing the failed lineage.

## Exact sole parent

Every arm starts from the same confirmed U1 child:

| Field | Frozen value |
| --- | --- |
| U0 ancestry | U0 child `20260727` |
| U1 parent | U1 child `20260733` |
| Parent path | `/Volumes/T7 Developer/DungeonApprentice/u1-local-replication-20260722/v02-u1-replication-seed-20260733/checkpoints/mastered-local-unlock.zip` |
| Parent checkpoint SHA-256 | `3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104` |
| Policy-tensor SHA-256 | `e555d3f7e2364f74e3b43371938c2f25e2ddbf62558a810038509a70868e6888` |
| Optimizer-state SHA-256 | `cc07791b374620d680e5cbb3602d59ab83eb55f195d26808abc0adf5f0e3bc5f` |
| Parent lifetime trained actions | 786,432 |
| Parent optimizer updates | 1,536 |
| Collision-safe U1 confirmation | Navigate 188/200; U0 200/200; U1 182/200, all gates passed |

Every arm loads the exact parent policy **and optimizer state**. No arm may load, distill, imitate,
inspect for action selection, or otherwise inherit from a U2, U2r, or another U2-S policy.

“Fresh child” means a new U2 learning trajectory from this frozen U1 boundary. It does not mean
random policy initialization, and it does not discard the confirmed cumulative skills or Adam
moments.

## Fixed matched identities

All four arms run sequentially on the audited local machine. They deliberately share:

- algorithm seed `20260753`;
- worker streams `20260753`, `20260754`, `20260755`, and `20260756`;
- the same exact parent bytes and parent optimizer;
- the same training-layout partition and fixed development suites;
- the same action budget, worker count, rollout size, batch size, lesson mix, and exam schedule; and
- the same static lesson-specific history guards frozen before arm one.

These are **matched random numbers**, not four independent replications. Arm identity must not
perturb RNG initialization before the intervention-specific behavior begins. Each arm starts from
a newly loaded parent and freshly initialized environment/scheduler state; no optimizer, recurrent
state, rollout buffer, curriculum state, or episode history flows from one arm to another.
The disposable qualification smoke and every canonical arm must capture one exact RNG identity at
the same boundary: after SB3 has reset all four workers, but before diagnostics or action one. The
identity covers Python, global NumPy, Torch CPU and any available accelerator, the model action
space, the scheduler, and every worker's action space, observation space, curriculum/base
environment RNG, and deterministic first-episode position. The diagnostic parent-baseline exam is
surrounded by a complete RNG snapshot/restore, followed by an exact recapture, so the sealed
identity is the state that actually enters the first training rollout.

The canonical scientific root is prospectively assigned:

`/Volumes/T7 Developer/DungeonApprentice/u2s-ablation-20260723`

The read-only dashboard is prospectively assigned:

`http://127.0.0.1:8787/`

Canonical protocol ID:

`dungeon-apprentice-v0.2-u2s-stability-ablation`

The completed implementation and protocol must first be frozen in a clean commit and externally
anchored by an annotated tag. Claim-bearing qualification then authenticates that exact remote tag
before it may create only its sealed qualification record. Until qualification passes, no canonical
cohort root or arm may be launched.

## The 2 × 2 intervention matrix

The fixed sequential order is control, conservative, no-effect, then combined:

| Arm ID | PPO update rule | Repeated visible no-effect penalty | Interpretation |
| --- | --- | --- | --- |
| `control` | Existing U2 PPO | Off | Matched baseline |
| `conservative` | Conservative PPO below | Off | Update-stability intervention only |
| `no-effect` | Existing U2 PPO | On | Experience-feedback intervention only |
| `combined` | Conservative PPO below | On | Both interventions |

Canonical arm directories beneath the scientific root are `control`, `conservative`, `no-effect`,
and `combined`.

No fifth arm, replacement arm, altered seed, reordered intervention, or post-result hyperparameter
variant is part of this protocol.

## Unchanged PPO cell

The `control` and `no-effect` arms retain the U2 learner:

| PPO field | Frozen value |
| --- | ---: |
| Learning rate | constant `2.5e-4` |
| Clip range | `0.20` |
| Epochs per rollout | 4 |
| Target KL | none |
| Rollout | 512 steps × 4 workers = 2,048 transitions |
| Batch size | 256 |
| Gamma | 0.995 |
| GAE lambda | 0.98 |
| Entropy coefficient | 0.01 |
| Recurrent state | one 256-unit LSTM layer |

All PPO fields not named by the conservative intervention remain equal in every arm.

## Conservative PPO cell

The `conservative` and `combined` arms change exactly four PPO controls:

| PPO field | Conservative value |
| --- | ---: |
| Learning-rate schedule | linear `2.5e-4` at child action 0 to `2.5e-5` at action 1,048,576 |
| Clip range | `0.10` |
| Epochs per rollout | 2 |
| Target KL | `0.015` |

The linear learning-rate schedule uses trained child actions and reaches each endpoint exactly.
Target KL may stop later epochs within a rollout according to the underlying PPO implementation;
the skipped epochs and measured approximate KL must be logged. It does not change the collected
action budget.

There is no additional weight decay, gradient clipping change, entropy schedule, architecture
change, value-function change, or recurrent-state change.

## Experience-only repeated no-effect penalty

The `no-effect` and `combined` arms add one training-only reward component. The rule sees only
information the learner itself experiences:

1. the action just chosen;
2. the current `56 × 56 × 3` visible RGB bytes; and
3. the next `56 × 56 × 3` visible RGB bytes.

Pickup, drop, and toggle are the only eligible actions. A transition is **visibly ineffective**
when the chosen eligible action produces a byte-identical next visible image.

The wrapper tracks a streak only when consecutive transitions:

1. choose the same eligible interaction action; and
2. each produce a byte-identical next visible image.

The first visibly ineffective transition starts a streak and receives no added penalty. The second
and every later transition in that uninterrupted identical-action, visibly-no-effect streak adds
exactly **`-0.01`**. The streak resets on:

- any visible pixel change;
- a different action;
- a non-interaction action;
- environment reset; or
- episode termination.

The cumulative added penalty is capped at **`-0.10` per episode**. Once that cap is reached, later
qualifying actions remain diagnostic events but add no further reward change. This keeps the
experience signal bounded below the complete-quest reward.

Examples:

| Sequence | Added penalty |
| --- | ---: |
| One ineffective toggle, then movement | 0 |
| Two consecutive visibly ineffective toggles | `-0.01` |
| Five consecutive visibly ineffective pickups | `-0.04` |
| Ineffective pickup followed by ineffective toggle | 0 |
| Toggle that visibly opens a door | 0 |
| Toggle opens a door, then one repeated toggle does nothing | 0 |
| Toggle opens a door, then two repeated toggles do nothing | `-0.01` |
| Twenty consecutive visibly ineffective toggles | `-0.10`, the episode cap |

The rule may not inspect coordinates, map IDs, object types, inventory labels, milestones, oracle
state, shortest paths, solvability traces, or trainer-only `info` to decide the reward. It does not
mask or replace an action. It does not enter the observation. Evaluation disables it completely.

The existing `-0.001` ordinary step cost, `+1.0` complete-quest reward, and bounded pixels-only
curiosity remain unchanged. The report must separately record extrinsic return, curiosity return,
no-effect penalty count, and no-effect penalty return so their effects cannot be hidden inside one
aggregate reward.

## Unchanged task and curriculum

All arms retain:

- the same 9 × 9 Navigate, U0, U1, and U2 generators and horizons;
- the same `56 × 56 × 3` partial pixel observation;
- the same seven primitive actions;
- deterministic recurrent evaluation with state reset per case;
- the same task success signal and no milestone shaping;
- the same 50% Navigate, 7.5% U0, 7.5% U1, 35% U2 normal transition target;
- the same prerequisite recovery controller and recovery profiles; and
- timeout as a terminal failed quest rather than a bootstrap truncation.

The policy receives pixels and its recurrent state only. Trainer diagnostics grade outcomes but may
never choose or replace an action.

## Lesson-specific history guards

The ablation retains the scientifically workable r1 mapping:

| Lesson | Hard exact-layout training guard |
| --- | --- |
| Navigate | Same-lesson development, qualification/confirmation references, and authenticated completed historical Navigate identities |
| Visible Unlock U0 | Its 79 unique frozen development layouts only; other historical overlap is diagnostic |
| Local Unlock U1 | Same-lesson development/confirmation references and authenticated completed historical U1 identities |
| Separated Unlock U2 | Sealed qualification, development/confirmation references, and authenticated completed or active historical U2/U2r identities |

Before launch, qualification must reconstruct and digest-bind the complete pre-U2-S inventory,
including the now-complete U2r-r1 episode-start and active-worker evidence. The resulting four
static guard sets are frozen once and supplied identically to every arm.

An earlier arm's training history is recorded for later exclusion and audit, but it is **not** added
to the next arm's hard guard. Adding it sequentially would break the matched training distribution.
Within-arm layouts also need not be globally unique. U0 history overlap remains an explicit
diagnostic rather than a false claim of broad finite-domain novelty.

The guard operates before the policy sees a layout and never exposes a historical image, outcome,
seed role, score, or action trace to the learner.

## Seed boundary

`20260753`–`20260756` are algorithm and worker RNG stream identities. They do not grant direct use
of numerically similar final-test seeds. Environment layouts remain restricted to the ordinary
training partition `0`–`999_999`.

The fixed development validation suites at 10–11.2 million may be used only by scheduled no-update
exams. The ablation must not generate, qualify, inspect, select, or score:

- the consumed U2 confirmation streams `15_200_000`–`15_239_999`;
- the unopened U2r successor streams `15_240_000`–`15_279_999`; or
- any project-final seed beginning at 20 million.

No confirmation or final-test issuer belongs to this implementation. The unopened U2r ranges remain
sealed evidence of an exam that never became eligible; they are not silently recycled for U2-S.

## Exact budget and exam schedule

Each arm receives exactly **1,048,576 new U2 child actions** from the U1 parent. With four workers
and 2,048-transition rollouts, this is 512 optimizer phases per arm under the ordinary four-epoch
rule, subject to target-KL epoch stopping in conservative cells.

Every 32,768 trained child actions, after optimization, each arm runs the same frozen four-lesson
exam:

- 80 Navigate cases, panels 40 + 40;
- 80 Visible Unlock U0 cases, panels 40 + 40;
- 80 Local Unlock U1 cases, panels 40 + 40; and
- 80 Separated Unlock U2 cases, panels 40 + 40.

That produces 32 exams and 10,240 deterministic case records per arm. All intermediate exams are
retained for mechanism curves and debugging, but they cannot stop an arm, select a checkpoint, or
change another arm.

Every arm runs to action 1,048,576 even if it appears to master earlier. The only decision window is
the prospectively fixed last three post-update exams:

`983,040`, `1,015,808`, and `1,048,576` child actions.

The terminal artifact is always the exact 1,048,576-action checkpoint. No “best checkpoint” search
is permitted.

## Explicit loop-tail evidence

Means alone allowed two catastrophic U2 cases and one Local Unlock case to hide inside otherwise
clean lessons. Every exam must therefore preserve per-case records sufficient to recompute:

- successes, panel successes, milestones, steps, collisions, coverage, and action histograms;
- total ineffective interactions per case and per lesson;
- mean, median, p90, p95, p99, and maximum ineffective interactions per lesson;
- counts and rates of cases with at least 1, 3, 10, and 32 ineffective interactions;
- the longest repeated identical interaction-action run in every case;
- counts of cases whose longest such run is at least 3, 10, and 32;
- the longest run and count of actions that separately satisfy the pixels-only penalty condition;
- the fraction of each lesson's ineffective interactions contributed by its worst one, two, and
  five cases;
- pickup, drop, and toggle visible-effect rates independently;
- success conditional on key acquisition and on door opening; and
- for penalty arms, penalty event count and return by lesson and case.

The terminal gate uses the generic repeated-identical-interaction diagnostic: consecutive identical
pickup/drop/toggle actions whether or not every transition changes pixels. The separate
penalty-eligible streak requires every transition in its run to leave the visible pixels
byte-identical and resets on any visible change. Both measures remain in the report so update
instability and visible no-effect feedback are not conflated.

The public dashboard should show, for each arm, terminal-window eligibility, U2 successes, mean
ineffective interactions, worst-case ineffective interactions, maximum visible no-effect streak,
and the top-two tail share. Dashboard presentation cannot affect training or selection.

## Fixed terminal eligibility rule

An arm is eligible only if **each of its final three exams independently** satisfies every item
below:

1. the exam follows an allocation-valid normal-practice window;
2. Navigate is at least 68/80 overall and 34/40 in each panel;
3. Visible U0 is at least 68/80 overall and 32/40 in each panel;
4. Local U1 is at least 68/80 overall and 32/40 in each panel;
5. Separated U2 is at least **72/80** overall and **34/40** in each panel;
6. mean ineffective interactions are at most **3.0** independently on U0, U1, and U2;
7. no case in any of the four lessons has **10 or more** ineffective interactions; and
8. no case in any of the four lessons has a repeated identical pickup/drop/toggle action run of 10
   or more actions.

The inequalities are exact. A case with nine passes the two tail ceilings; a case with ten fails.
Three passing exams separated by an earlier failure are eligible because the terminal window was
fixed prospectively. An earlier three-exam streak cannot replace a failed terminal window.

No metric may be averaged across the three exams to rescue an individual failed boundary, and no
arm may borrow a score from another arm.

## Fixed arm decision rule

After all four valid terminal reports exist, choose the first eligible configuration in this
predeclared simplest-intervention order:

1. `control`;
2. `conservative`;
3. `no-effect`;
4. `combined`.

This priority deliberately prefers no behavioral change, then an optimizer-only change, then a
reward change, and uses both interventions only when neither simpler option qualifies.

The decision selects only the **configuration definition**. It never selects, copies, promotes,
confirms, or deploys an arm checkpoint. Even the winning arm's policy is development evidence and
must remain inside the ablation record.

The 2 × 2 report also presents descriptive paired contrasts for success, inefficient-interaction
means, tail counts, and streak maxima:

- conservative main contrast: average of `conservative` and `combined` versus average of `control`
  and `no-effect`;
- no-effect main contrast: average of `no-effect` and `combined` versus average of `control` and
  `conservative`; and
- interaction contrast: whether the combined change differs from the sum of the two individual
  changes.

With one trajectory per cell, these contrasts explain the observed matched study; they are not
population-level significance tests.

## Stop rule

The ablation ends after four valid full-budget terminal reports and one deterministic application
of the priority rule.

- If at least one arm is eligible, the selected configuration may be proposed in a **new,
  separately committed and qualified** multi-lineage U2-S cohort protocol.
- That successor must start fresh from frozen confirmed U1 parents. It may not inherit an ablation
  checkpoint.
- If no arm is eligible, the ablation verdict is `ablation_failed`. No successor U2-S cohort and
  no U3 work may begin from this study.
- A valid all-arm failure is the stopping point for this learner family. The next prospective
  decision must be architectural—for example, an explicitly tested action-effect prediction
  module—not another post-hoc reward amount, PPO setting, checkpoint, seed, or continuation.

U2-S is deliberately **non-resumable**. Continuing only the interrupted arm from a fresh
environment, recurrent state, or RNG position would break the matched four-cell comparison. Any
interruption or crash therefore makes the entire canonical cohort terminal
`operationally_incomplete`. No arm tip may continue and no completed arm may be reused. A future
replacement requires a new prospective source commit, annotated tag, protocol-attempt identity,
and root, and must restart all four cells from the exact confirmed U1 parent.

## Qualification before action one

A clean implementation must prove, before creating the canonical root:

1. the original U2, U2 confirmation, U2r attempt 1, and U2r-r1 reports and artifacts still match
   their frozen hashes;
2. all four arms load the exact U1 parent policy and optimizer, and no path accepts a U2/U2r parent;
3. the same algorithm seed, worker streams, parent state, scheduler state, static guard digests,
   and complete post-reset/pre-action RNG identity initialize every arm and exactly match the
   disposable smoke;
4. the intervention matrix changes only the declared PPO fields and/or penalty;
5. the linear learning-rate schedule reaches both exact endpoints and no public or launcher path
   can resume one arm;
6. target KL, epoch counts, clip range, and skipped-epoch accounting are tested;
7. the penalty fires only on the second and later identical consecutive eligible actions with
   byte-identical next pixels, stops changing reward at its exact `-0.10` episode cap, resets under
   every declared condition, stays out of policy observation, and is absent from evaluation;
8. reward components are separately logged and finite;
9. all 32 exam boundaries describe post-optimizer parameters and retain exact case evidence;
10. terminal eligibility and the simplest-intervention priority rule reject every off-by-one,
    missing-case, non-normal, allocation-invalid, intermediate-checkpoint, and cross-arm rescue;
11. training cannot access confirmation or final partitions;
12. any interruption or crash publishes terminal `operationally_incomplete` evidence, rejects
    continuation from every checkpoint, and requires any replacement to restart all four arms;
13. storage guards refuse launch without a 25 GiB reserve and bound each arm, the scientific root,
    and optional narrative media; and
14. lint, the complete automated suite, the mechanical oracle suite, fixed-worker sampler
    preflight, and a non-claim integration smoke all pass from the same clean source. The smoke
    loads four disposable copies of the exact parent, gives every arm one real 2,048-transition
    four-worker rollout and optimizer phase, and then deletes those copies. It must prove that the
    disposable policies and optimizers changed while no canonical or claim-bearing policy did.

The clean commit and annotated tag are created immediately before this claim-bearing qualification
and bind this protocol, the exact U1 parent, the four interventions, seeds, guard inventory, action
budget, decision rule, protected ranges, root, and dashboard assignment. Qualification must
authenticate that exact remote anchor before root creation or action one.

## Required terminal report

The terminal cohort report must include:

- source commit, external tag and tag object, protocol-document digest, and repository cleanliness;
- the exact U1 parent archive, policy tensor, optimizer state, sidecar, ancestry, and confirmation;
- per-arm intervention identity and a proof that only declared fields differ;
- algorithm/worker streams, the complete initial RNG-state identity, its aggregate digest, and
  exact equality to qualification and across all four arms;
- collected, trained, child, inherited, lifetime, rollout, epoch, KL-stop, and optimizer counters;
- target and realized transition allocations plus recovery history;
- all 128 exam records and all 40,960 case records with independent hashes;
- all loop-tail metrics and the fixed final-three eligibility checks;
- complete training-layout and active-worker identities;
- every selected terminal checkpoint, sidecar, optimizer, model-state, and integrity digest;
- fixed priority-rule inputs and output;
- storage-cap and free-space audits;
- proof that no confirmation/final issuer existed or protected candidate opened; and
- process closeout with no orphaned trainer.

The report must distinguish `mechanism_selected`, `ablation_failed`,
`operationally_incomplete`, and `integrity_failed`. Only the first permits writing—not
automatically launching—a separately preregistered successor cohort.

## Narrative boundary

The story is not “we kept tweaking until one AI passed.” It is:

1. the project froze a one-case U2 confirmation failure;
2. unchanged continuation showed that capability could coexist with unstable deterministic loops;
3. the attractive penultimate checkpoint was deliberately not selected;
4. two terminal U2 cases accounted for almost 90% of all ineffective interactions;
5. four new children restart from the last confirmed boundary before U2;
6. a matched 2 × 2 test separates gentler learning updates from feedback about visible no-effect
   repetition;
7. every child receives the same full budget and is judged only in the same fixed terminal window;
8. the simplest qualifying intervention wins; and
9. no ablation model itself advances the game.

Success means the project has identified a learner worth replicating honestly. Failure means the
current PPO-and-reward family has reached its declared stopping point. Either result is more useful
than another lucky checkpoint.
