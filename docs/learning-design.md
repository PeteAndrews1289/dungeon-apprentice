# How the apprentice learns

## The central idea

Dungeon Apprentice does not begin with a world as sprawling and irregular as Pokémon. It builds a
measurable staircase of concepts inside a fast original game. The same policy first learns to move
toward an exit, then to perform a key-door sequence, then to carry an object back to a place that is
no longer visible. Procedural layouts prevent it from memorizing a single button script.

This is different from the earlier Pokémon experiments in four decisive ways:

1. The game exposes a small, stable action and visual interface designed for learning.
2. Every generated problem can be mechanically proved solvable before training sees it.
3. Progress means success on unseen layouts, not discovering one more hand-authored checkpoint.
4. The curriculum advances automatically under frozen rules and keeps retesting old skills.

There are no human gameplay traces, savestate lessons, route scripts, online language-model calls,
or objective-specific action rules. A human defines the game and the exam, not the move sequence.

## One policy, three core components

The policy has a convolutional vision encoder, an LSTM memory, and an action/value head. The vision
encoder turns a 56 × 56 pixel view into features. The LSTM can retain evidence after an object or
location leaves view. The action head chooses among seven buttons. PPO changes all of these weights
using trajectories produced by the policy itself.

The completed v0.3 Stage-A r3 study added a small fourth pathway without replacing those learned
components. It told the same recurrent policy which primitive it selected one transition ago and
whether the next visible RGB frame changed or remained identical. This was not a key label, success
flag, route, or object detector. It was the digital equivalent of remembering “I just tried this,
and I saw nothing happen.” A matched sham had the same parameters and input shape but received
zeros. The pathway learned, but its one-transition memory did not satisfy the fixed reliability
gate.

At the beginning of the overall project, the model starts with random parameters and is not reset
when a new tier unlocks. v0.3 was a later architecture study: both matched arms restarted from the
exact confirmed U1 policy and optimizer, then added the new zero-initialized pathway. The point
remains whether one growing policy can acquire and retain a repertoire.

## Exploration without a walkthrough

Sparse success rewards make early random experience repetitive. During training only, the agent
receives a small intrinsic bonus the first time a pixel view appears in the current attempt. The
default is `0.002` for a novel view, zero for a repeated or unchanged view, and no more than `0.1`
over the entire episode. The initial reset view is pre-marked as seen.

This mechanism cannot say “pick up the key” or “go north.” It cannot see coordinates, the full map,
or object labels. A useless new view and a useful new view are equally novel. Its limited purpose is
to make looking around more attractive than staring at one wall. The cap is deliberately smaller
than the task-success signal and cannot turn a timeout into a positive return. Held-out exams remove
this bonus completely, so curiosity cannot manufacture a passing score.

The original v0 formula was not merely too generous; it changed what optimization preferred. A
failed random Retrieve attempt could earn more than an efficient success. Those runs remain useful
software and storytelling evidence, but they are not learning evidence. Protocol v0.1 begins a new
comparison from random parameters under the bounded formula.

## Automatic curriculum

Training begins with Navigate. Every fixed interval the policy is frozen and evaluated
deterministically on validation seeds it has never trained on. A score of 90% unlocks the next tier.
After promotion, 70% of attempts use the newest tier and 30% rehearse earlier tiers. Every later
exam requires at least 80% success on every earlier tier or progression is held.

This is curriculum learning, but it is not a set of demonstrated solutions. It changes which class
of procedural problem the agent experiences. The policy must still discover every action sequence.

The three v0.1 capability canaries later revealed that this ratio described episode counts, not
actual experience. Because successful Navigate episodes were much shorter than failed Unlock
timeouts, the 30% earlier-tier episode share became only 4–6% of post-promotion transitions. That
finding is preserved in [the canary report](results/v0.1-navigate-canaries.md). The proposed
[v0.2 design](protocol-v0.2-design.md) controls measured transitions and adds progressively harder
complete Unlock distributions; it does not retroactively change the frozen v0.1 contract.

The implemented cumulative controller now measures **actions**, not episode counts. Its Unlock
staircase is Navigate → U0 Visible Unlock → U1 Local Unlock → U2 Separated Unlock, with every
post-update boundary retesting all inherited skills. Practice uses a declared transition mix. If an
older skill misses either its overall or panel gate, the newest mastery streak is erased and the
next complete practice window reallocates actions toward the weak prerequisite. Recovery changes
future experience only: it never rolls weights back, lowers a gate, pays for a milestone, or counts
as mastery.

U1 and U2 were launched as versioned cumulative children because their exact rules were frozen only
after the previous checkpoint set passed disjoint confirmation. Policy weights, LSTM parameters,
value function, and optimizer state remain continuous within each lineage; trajectories and
recurrent episode state never cross between lineages. A later fresh-start experiment must still
show that one launcher can reproduce the entire staircase from random initialization without manual
stage stitching.

## Learning more versus testing again

U2 exposed an important difference between those two actions. All three policies passed adjacent
development exams, but the larger no-update confirmation passed only two of them. The third scored
169/200 against a frozen 170/200 U2 gate. Testing the unchanged checkpoint repeatedly would not
teach it anything; it would only create more chances for a borderline sample to cross the line.

[U2r Stability Remediation](protocol-v0.2-u2r-stability-remediation.md) therefore creates a new
learned artifact before opening another test. Only the failed lineage continues, its complete
optimizer remains intact, and it receives exactly the 360,448 actions left under its original U2
ceiling. It still gets pixels, sparse completion reward, the same four-lesson practice mixture, and
no knowledge of the future exam.

The distinction is visible in the selection rule. Eleven additional windows must run regardless of
intermediate scores, and only the full-budget terminal model can qualify. Its last two development
exams must not only pass the old success gates; they must also avoid the repeated-interaction
behavior diagnosed in the failed lineage. A qualifying terminal model then gets one fresh,
collision-aware, no-update confirmation. Thus:

- **training** may change the policy using ordinary self-generated experience;
- **development exams** may determine whether the fixed terminal artifact is stable enough to test;
- **confirmation** measures that already selected artifact and cannot change it; and
- the old failed confirmation remains failed in every future outcome.

This is what prevents “keep testing until it passes” from masquerading as learning.

## What action-effect context taught us

The terminal U2-S ablation tested whether ordinary optimization, conservative optimization, a
bounded pixels-only penalty, or both could make Separated Unlock simultaneously capable and
reliable. It did not produce a winner:

- control finished at 78, 76, and 77/80 U2 but still contained catastrophic interaction loops;
- conservative finished at 64/80 three times and still developed rare tails;
- no-effect finished at 78, 79, and 76/80 and reduced the tail, but did not eliminate it; and
- combined had no 10-plus-ineffective cases, yet finished at 70, 69, and 71/80.

That is useful negative evidence. The problem is not simply “too much learning” or “no penalty for
bad buttons.” With the v0.2 interface, the LSTM processes the current frame before the policy samples
an action. On the next frame it can notice that pixels did not change, but stochastic action
selection means its hidden state was never explicitly told which action actually occurred. A
hundred identical toggles can therefore look like a generic unchanged scene rather than a specific
failed cause-and-effect experiment.

The frozen
[v0.3 Stage-A r3 protocol](protocol-v0.3-action-effect-architecture-r3.md) supplied that missing
sensorimotor link while keeping the task signal untouched. Attempt 0 never reached action one; r1
reached only a partial sham arm before an operational process-control failure; r2 completed sham
but stopped at terminal authentication because two evidence components disagreed about one final
line-feed byte. r3 then tested the matched architecture question:

1. At episode start, context is all zero.
2. The policy selects one of the same seven actions.
3. The wrapper compares only the visible before/after pixel bytes.
4. The next observation contains the prior action one-hot and exactly one
   changed/unchanged outcome bit.
5. A learned `9 → 512` residual decides whether and how that context should alter the inherited
   visual features.

There is no hand-coded action ban. An unchanged pickup may be useless in one state and a necessary
failed probe in another; the policy must learn that distinction from later returns. Evaluation uses
the same context derivation but disables all parameter updates, just as it disables curiosity.

The migration did not erase what U1 had proved. The NatureCNN, actor and critic LSTMs, action head,
value head, and every associated Adam moment transferred by exact parameter name. The one new
projection started at zero. Sham and action-effect generated the same complete 2,048-transition
rollout, with aggregate SHA-256 `fa4c7bda99a261f8fa49741a49360cd1bfc6ab3081db51aeffc64266a109ce72`,
and did not diverge before the first optimizer phase.

The candidate encoder became nonzero exactly at 2,048 actions and ended with all 4,608 weights
nonzero. It learned U2 from 24/80 at inheritance to 75/80 at the final exam. This rules out the easy
explanations that the feature was dead, the transplant failed, or the model learned nothing.

The frozen terminal-three rule nevertheless rejected the architecture. Sham finished at 77, 78,
and 78/80 U2 but was calibration-only and contained one 85-action tail. Action-effect finished at
70, 72, and 75/80, and all three deciding exams contained forbidden ten-plus ineffective or
repeated-action tails. The exact result is
[**`architecture_failed`**](results/v0.3-action-effect-stage-a-r3.md): no architecture selected,
no Stage-A checkpoint reuse, no Stage B replication, and no U3 activation.

The lesson is sharper than “action-effect did not work.” A one-step action/outcome pair is useful
enough to train on, but it does not explicitly represent persistence: how long the same ineffective
experiment has continued. The next bounded direction is therefore the prospectively frozen
[v0.4 matched ineffective-trace study](protocol-v0.4-ineffective-trace-architecture.md). It exposes
only `min(consecutive same-action unchanged frames, 9) / 9`, never the action ID, through one
zero-initialized 1→512 residual. A zero-valued sham and truthful candidate start independently
from the same confirmed-U1 parent under unchanged reward, PPO, curriculum, budget, and frozen
terminal-three gate. It is still a release candidate, not a training authorization.

## Why timing matters

Recurrent PPO alternates between collecting a rollout and optimizing on it. An exam triggered while
the rollout is still being collected measures the previous policy, even if its label shows the new
step count. Protocol v0.1 schedules exams and checkpoint publication after optimization. It records
collected/trained timesteps, optimizer updates, and the exact checkpoint digest graded by every exam.

## What counts as evidence

Loss curves and training reward show whether optimization is alive. They do not prove competence.
The evidence ladder is:

- generated levels pass the independent solvability oracle;
- model parameters update and checkpoints reload;
- a restored checkpoint reproduces its own curriculum and counter state rather than borrowing state
  from a newer point in the run;
- a frozen policy passes unseen validation layouts;
- the same checkpoint retains earlier tiers;
- a selected checkpoint passes the untouched final suite.

Only the last item supports a generalization claim. Final-suite seeds are deliberately separated so
that repeated experimentation cannot quietly turn the test set into another training signal.
