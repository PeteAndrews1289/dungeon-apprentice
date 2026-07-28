# Architecture

```mermaid
flowchart LR
    Seed["Seeded dungeon generator"] --> World["MiniGrid world"]
    World --> Pixels["Partial RGB pixels"]
    Pixels --> Policy["One recurrent local policy"]
    Policy --> Buttons["Seven game actions"]
    Buttons --> World
    World --> Info["Trainer-only outcome record"]
    Info --> Curiosity["Bounded pixel-only curiosity"]
    Curiosity --> Rollout["Complete recurrent rollout"]
    Rollout --> PPO["PPO optimizer phase"]
    PPO --> Policy
    World --> Oracle["Solvability oracle"]
    Oracle --> Valid["Generator qualification only"]
    PPO --> Exam["Post-update frozen unseen-seed exams"]
    Exam --> Gate["Automatic promotion + retention gate"]
    PPO --> Bundle["Digest-linked model + JSON sidecar"]
```

## Separation of responsibilities

- **Game:** owns objects, transitions, rendering, rewards, success, and time limits.
- **Generator:** creates varied layouts and rejects any layout that violates the tier's reachability
  constraints.
- **Oracle:** proves a complete legal action sequence exists. It does not teach.
- **Policy:** receives pixels and produces actions.
- **Trainer:** updates the policy from ordinary experience and mixes retained tiers.
- **Curiosity:** pays only for novel policy-visible pixel views, has a hard episodic budget, and is
  absent from exams.
- **Evaluator:** freezes updates, resets recurrent state between episodes, and grades held-out seeds.
- **Run supervisor:** separates collected from trained timesteps and writes manifests, heartbeats,
  checkpoint bundles, frames, and terminal reasons after optimizer boundaries.
- **Dashboard:** displays evidence but cannot change the run.

## Present expansion boundary

Protocol v0.1 deliberately implements only three tiers. Their generation, reward, oracle recipe,
evaluation loop, and promotion order are still encoded in tier-specific branches. That is adequate
for the first learnability experiment, but it is not yet a plug-in mechanics architecture. Adding a
lever or enemy today would require coordinated edits across those components.

The completed v0.1 canaries also showed that episode-weighted practice does not control experience
when tier horizons differ: a nominal 30% Navigate episode share became only 4–6% of post-promotion
transitions. The proposed [v0.2 protocol](protocol-v0.2-design.md) introduces a declarative lesson
layer and a transition-deficit sampler before any new mechanic is added.

Before the first mechanics expansion, the framework will introduce declarative `MechanicSpec` and
`TaskSpec` registries, success predicates, procedural blueprints, and a capability dependency graph.
Each new mechanic must bring:

- placement and transition rules;
- a mechanical solvability proof;
- an isolated unseen-seed evaluation suite;
- at least one composed suite with earlier mechanics;
- a stable policy-visible objective representation if two tasks can share the same scene.

The observation shape and seven-action vocabulary should remain stable where practical so old
checkpoints can still be evaluated. A model is never credited merely because the generator or oracle
can solve a level.

## U2 cumulative lineage boundary

U2 is the first expansion built from three independently confirmed cumulative parents rather than
one selected development checkpoint. There is no policy merge and no cross-lineage replay:

```mermaid
flowchart LR
    P1["Confirmed U1 parent 20260725"] --> C1["U2 child 20260737"]
    P2["Confirmed U1 parent 20260729"] --> C2["U2 child 20260741"]
    P3["Confirmed U1 parent 20260733"] --> C3["U2 child 20260745"]
    Q["One-shot 2,000-layout oracle qualification"] --> A["Remote annotated evidence tag"]
    A --> C1
    A --> C2
    A --> C3
    C1 --> E["Four frozen post-update exams"]
    C2 --> E
    C3 --> E
    E --> R["Automatic prerequisite recovery or U2 mastery"]
    C1 --> D["Read-only cohort dashboard :8785"]
    C2 --> D
    C3 --> D
```

The children execute sequentially on the audited 8 GB M1, but the dashboard presents all three
lineage states from one small cohort manifest. It reads each lineage's atomic status and frame files;
it cannot choose an action, update a policy, or stop a trainer when the browser closes.

The new environment changes geometry and distance, not the agent interface. A complete divider has
one locked-door crossing, the key and start are on the approach side, the goal is beyond it, and
exactly two extra interior walls create a longer 17–26-action qualified route. The policy still sees
the same partial RGB pixels, carries the same 256-unit recurrent state, and emits the same seven
actions. PPO, reward, curiosity, and inherited optimizer state are unchanged.

The trainer now maintains a four-lesson capability graph:

```mermaid
flowchart LR
    N["Navigate"] --> U0["Visible Unlock"]
    U0 --> U1["Local Unlock"]
    U1 --> U2["Separated Unlock"]
    Exam["Every 32,768 new trained actions"] --> N
    Exam --> U0
    Exam --> U1
    Exam --> U2
    N --> Recovery["Transition-balanced recovery"]
    U0 --> Recovery
    U1 --> Recovery
    Recovery --> Exam
```

A prerequisite miss resets the U2 mastery streak and changes only future practice allocation. It
does not roll weights back, change reward, or lower a gate. Exams occur after optimization and name
the exact archive digest they score, so dashboard state can never substitute for checkpoint state.
Every completed 2,048-action optimizer phase also replaces one bounded `latest-safe` archive,
sidecar, and integrity record. An interruption may discard incomplete work, but it cannot erase or
double-charge a completed update; a resumed lineage continues in a new manifest segment.

Seed roles are architectural boundaries as well as numbers. Training uses the sub-million
partition; generator engineering has a disposable `5_200_000` sandbox; sealed qualification uses
`5_210_000`–`5_211_999`; and four fixed validation suites occupy their declared 10–11.2-million
blocks. At the U2 source freeze, four 15.2-million candidate streams were reserved for the later
collision-aware confirmation, while the 20-million final allocation remained untouched.
Exact-layout hashes defend against the lesson learned from U1 confirmation attempt 1: different seed
numbers can still generate the same dungeon.

Storage is part of correctness. U2 refuses symlinked or escaping run paths, caps each lineage at
2 GiB and the scientific cohort at 6 GiB, keeps optional media in a separate 10 GiB directory, and
checks a 16 GiB combined plan plus the free-space reserve. At U2 source freeze, no protected U2
seed had been opened and no U2 learning or confirmation claim existed. Later runtime claims cite
the external qualification, cohort, and confirmation ledgers rather than rewriting that prospective
boundary.

## U2r successor boundary

Runtime evidence has now answered the original U2 confirmation: two children passed, while child
`20260745` scored 169/200 against its 170/200 U2 gate. That immutable 2/3 failure does not disappear
from the architecture. [U2r Stability Remediation](protocol-v0.2-u2r-stability-remediation.md)
adds one bounded successor branch:

```mermaid
flowchart LR
    C1["20260737 direct pass"] --> A["Potential U3 activation set"]
    C2["20260741 direct pass"] --> A
    C3["20260745 failed 169/200"] --> R["Exact optimizer continuation"]
    R --> W["11 fixed windows"]
    W --> T["Terminal-only stability gate"]
    T --> F["Fresh one-shot confirmation"]
    F -->|"only if passed"| A
    A --> L["Label: two direct + one remediated"]
```

Only the failed branch changes. It cannot load either passing sibling, reset its optimizer, select
an intermediate checkpoint, exceed the original 1,048,576-child-action ceiling, or inspect a fresh
confirmation candidate while training. The last two fixed exams must pass both the original
capability gates and the new prospective stability diagnostics.

Process recovery is an append-only lineage rather than an overwrite. The initial directory has the
fixed child name; contiguous successors use `-resume-N`, add `N × 100,000` to their process RNG
streams, and authenticate the prior safe checkpoint plus policy/optimizer state. Completed exams
and their 320 raw case records are copied and reverified in every successor. The four genuinely
active episodes at interruption are preserved as abandoned identities, while a terminal report
hashes every segment manifest and episode-start ledger so later collision exclusion covers the
whole lineage.

The consumed `15_200_000`–`15_239_999` confirmation roles are permanently unavailable for
successor training, retries, or new selection; claim-bound historical verifier access remains so
the frozen result can still be authenticated. Four new structurally sealed roles occupy
`15_240_000`–`15_279_999`, but only a full-budget terminal-eligible U2r artifact can authorize
their one-shot evaluator. The `20_000_000` final
allocation remains closed. Episode-start and active-worker layout identity now become part of every
checkpoint record so a future collision-aware selector can exclude the complete successor history,
including episodes active at the terminal boundary.

## v0.3 action-effect architecture boundary

U2-S completed without an eligible reward/PPO configuration. Its four cells made the missing
relationship unusually clear: strong updates could learn capability but occasionally collapse into
hundred-action interaction loops; conservative updates could suppress those loops only by losing
capability; pixels-only negative feedback narrowed but did not close the gap. v0.3 therefore changes
the policy input architecture while holding reward and PPO fixed.

```mermaid
flowchart LR
    P["Confirmed U1 parent 20260733"] --> T["Named tensor + Adam transplant"]
    T --> S["Sham twin"]
    T --> E["Action-effect twin"]
    RGB["Current 56×56 RGB view"] --> S
    RGB --> E
    A["Previous self-selected action"] --> C["9-value context"]
    D["Visible pixels changed / unchanged"] --> C
    C --> E
    Z["All-zero context"] --> S
    S --> G["Same full U2-S gate"]
    E --> G
    G -->|"candidate passes"| R["Separate three-lineage replication protocol"]
    G -->|"candidate fails"| X["Architecture study stops"]
```

The vector is retrospective sensorimotor context, not a game hint. Seven coordinates identify the
policy's own immediately preceding primitive action. Two identify whether the next visible RGB
bytes changed or remained identical. Episode start is all zero. The wrapper never reads reward,
coordinates, inventory, objects, milestones, seed roles, oracle state, or trainer-only `info`.

The custom extractor directly subclasses the existing NatureCNN. Its inherited `cnn` and `linear`
parameter names and 512-feature output therefore remain unchanged. One new bias-free `9 → 512`
linear projection is added as a residual before the existing actor and critic LSTMs. It is zeroed
after complete policy construction, and the policy disables the default post-LSTM multilayer head:

```text
image ── NatureCNN (inherited) ── 512 features ──┐
                                                ├─ add ─ actor/critic LSTMs ─ heads
context ── zero-initialized 9→512 projection ───┘
```

Every old CNN, LSTM, action-head, and value-head tensor is copied by exact state-dict name and shape.
Every old Adam state is joined to its new parameter by that same name; positional optimizer loading
is forbidden because inserting one parameter would otherwise shift moment ownership. The new
projection begins with no Adam state. Its zero output makes the migrated policy exactly equivalent
to the parent at the boundary even though the archive now has a Dict observation space.

That equivalence is executable, not rhetorical. Qualification must compare visual features,
deterministic actions, values, log probabilities, and both recurrent-state branches on a fixed
pixel corpus. The two arms must then produce byte-identical action/environment evidence over their
first real 2,048-transition rollout. Only the subsequent PPO phase may use the nonzero candidate
context to change the projection and create behavioral divergence.

The study remains deliberately narrow:

- both twins restart from confirmed U1, never from U2, U2r, or U2-S;
- reward, curiosity, curriculum, horizon, PPO, worker streams, action budget, and exams are matched;
- the complete U2-S final-three capability-and-stability gate is reused without relaxation;
- sham is calibration and cannot be selected;
- a passing candidate selects only the architecture definition, never its development checkpoint;
- Stage A is non-resumable, and an interruption requires a new frozen attempt for both twins; and
- U3 remains closed until a later three-parent replication and untouched confirmation both pass.

This boundary also prepares the eventual party-management game. A shared local policy must know not
only what an adventurer sees, but what that adventurer just attempted and whether the world visibly
responded. Because the context is per-agent and contains no class or objective shortcut, the same
weights can later be evaluated independently for several party members while their equipment,
class, local view, and higher-level tactics remain explicit future mechanics.

Attempt 0 exercised the original release chain and qualified, but its dashboard repeated the full
predecessor walk on every request and exceeded the launcher's ten-second health deadline. It
closed at zero actions; the exact negative inventory is in the
[attempt-0 result](results/v0.3-action-effect-launch-attempt-0.md). The separately committed
[r1 protocol](protocol-v0.3-action-effect-architecture-r1.md) preserved the architecture experiment
and corrected that dashboard boundary. It reached sham training, but whole-command-line process
matching counted the supervisor's embedded child command as a second trainer. The failed assertion
also bypassed the zsh exit-cleanup path. Sham reached 38,912 trained actions; status recorded 40,004
collected while the later episode ledger reached 40,960. It completed 19 optimizer phases, 76 new
updates, and one scheduled exam. No safe checkpoint exists and action-effect never started, so r1
is [terminal operational evidence](results/v0.3-action-effect-stage-a-r1-operational-failure.md),
not an architecture comparison.

The
[r2 protocol](protocol-v0.3-action-effect-architecture-r2.md) kept r1's dashboard correction and
added leading-argument process classification plus explicit abnormal-exit cleanup. It completed
the full sham budget, but the terminal manifest hashed first-rollout evidence without the final
line-feed byte used by qualification and training. The cohort stopped before action-effect began;
the [r2 result](results/v0.3-action-effect-stage-a-r2-operational-failure.md) is operational
evidence, not an architecture comparison.

The frozen
[r3 protocol](protocol-v0.3-action-effect-architecture-r3.md) retained every learner boundary and
named one shared LF-terminated digest profile. r3 then completed the matched experiment:

```mermaid
flowchart LR
    F0["Frozen zero-action attempt 0"] --> C["Clean r3 source"]
    F1["Frozen partial-sham r1"] --> C
    F2["Frozen full-sham r2 closeout failure"] --> C
    C --> T["Annotated r3 preregistration tag"]
    T --> Q["Durable qualification + matched smoke"]
    Q --> M["Cohort contract binds report SHA-256"]
    M --> L["Fixed r3 launcher"]
    L --> S["Full sham arm"]
    S --> E["Full action-effect arm"]
    E --> G["Fixed terminal-three grade"]
    G --> X["architecture_failed"]
    X --> N["No checkpoint reuse; Stage B and U3 closed"]
```

The r3 source was `c4834b73dfed7d875c6f59887d3077319a305910`, the annotated tag object
was `fc5eb2f82896b7f6d41030f2776a0028a70ea4c2`, the qualification report SHA-256 was
`07151fecea179dfaedabe56dcadc009d3750798987973c17cb67b1888909c499`, and the
cohort-contract SHA-256 was
`fea7b7eb10accec2395105630ad790d550993996c5bd35a2a937fdb8837d5ade`.
Both arms restarted from confirmed U1 and completed 1,048,576 actions, 2,048 new updates, and 32
exams. Their complete pre-update rollout matched at `fa4c7bda…`, with no divergence before the first
optimizer phase.

The sham projection remained exact zero. The candidate projection first became nonzero at 2,048
actions and ended with all 4,608 weights nonzero, weight L2 norm 2.826899, and update L2 norm
0.080078. The pathway was therefore live and learned. Yet the candidate's terminal U2 scores were
70, 72, and 75/80, with 6, 2, and 1 forbidden ten-plus interaction tails. The architecture supplied
one-step causal context but did not reliably suppress persistence across time. The exact
[r3 result](results/v0.3-action-effect-stage-a-r3.md) is `architecture_failed`; no architecture or
checkpoint was selected, Stage B was not authorized, and U3 remains closed.

The separately frozen
[v0.4 matched ineffective-trace protocol](protocol-v0.4-ineffective-trace-architecture.md)
tested the smallest evidence-supported successor: a bias-free `Linear(1, 512)` residual whose sole
input was `min(consecutive same-action unchanged frames, 9) / 9`, without returning the action ID.
Both fresh confirmed-U1 twins completed 1,048,576 child actions and 32 frozen exams each. Their
complete 2,048-transition pre-update rollout matched exactly, so the comparison began without
behavioral or RNG divergence. The candidate feature was active on 11.7828% of training
transitions, became nonzero at the first optimizer boundary, and ended with all 512 encoder weights
nonzero.

Average capability improved, but terminal reliability did not. Sham scored 77, 78, and 76/80 U2
with worst ineffective tails of 5, 1, and 7 actions. The candidate scored a higher 79, 78, and
79/80, yet retained tails of 153, 157, and 121 actions and failed exactly nine frozen tail checks:
three checks in each deciding exam. The authenticated
[v0.4 result](results/v0.4-ineffective-trace-stage-a.md) is `architecture_failed`. Sham remained
calibration-only, no architecture or checkpoint was selected, checkpoint reuse was not authorized,
and U3 remains closed.
