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

## One policy, three components

The policy has a convolutional vision encoder, an LSTM memory, and an action/value head. The vision
encoder turns a 56 × 56 pixel view into features. The LSTM can retain evidence after an object or
location leaves view. The action head chooses among seven buttons. PPO changes all of these weights
using trajectories produced by the policy itself.

The model begins with random parameters. It is not reset when a new tier unlocks. The point of the
experiment is whether one growing policy can acquire and retain a repertoire.

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
