# Protocol v0.1 experiment contract

## Question

Can one randomly initialized recurrent pixel policy acquire three reusable capabilities—navigation,
key-and-door interaction, and relic retrieval with backtracking—and retain earlier capabilities as
the procedural task distribution grows?

## Frozen rules

| Setting | Protocol v0.1 value |
| --- | --- |
| Protocol | `dungeon-apprentice-v0.1` |
| Observation | Partial RGB pixels, 56 × 56 × 3 |
| Policy-visible text | None |
| Policy-visible coordinates/map | None |
| Actions | Turn left, turn right, move forward, pick up, drop, toggle, wait |
| Declared capability-run dungeon size | 9 × 9 |
| Success reward | +1.0 |
| Ordinary step reward | -0.001 |
| Intermediate authored rewards | None |
| Training-only intrinsic reward | +0.002 once per novel pixel view |
| Per-episode intrinsic budget | 0.1, hard maximum |
| Training initialization | Random parameters |
| Recurrent PPO rollout | 4 workers × 512 steps |
| PPO optimization | Batch 256, 4 epochs, learning rate 0.00025 |
| Credit assignment | Gamma 0.995 (minimum allowed 0.995), GAE lambda 0.98 |
| Imported gameplay | None |
| Validation seeds | 10,000,000 and above |
| Final-test seeds | 20,000,000 and above |
| Promotion | ≥90% current-tier success and ≥80% retention |
| Promotion authority | Frozen deterministic policy, no updates |
| Training mixture after promotion | 70% newest tier, 30% prior tiers |

Short engineering/CI runs may use smaller rollouts or fewer epochs to exercise the pipeline, but
their results are not pooled with declared capability runs.

The key is consumed when it opens its matching door. Tier 2 succeeds only when the agent carries
the relic back onto the entrance tile. The agent is not rewarded for seeing or collecting a key,
opening a door, collecting the relic, visiting a new cell, or moving toward an objective.
Exhausting a tier's action budget is a terminal failed quest, not a continuing-state truncation; PPO
therefore cannot bootstrap value beyond the reset boundary.

The intrinsic curiosity signal is computed only from a fingerprint of the current policy-visible
pixels, and its memory resets every episode. The reset observation is marked seen before the first
action. A later view pays `min(0.002, remaining budget)` exactly once if its fingerprint is new;
repeated and unchanged views pay zero. Total curiosity can never exceed 0.1 in an episode. It does
not know whether a view contains a useful object, where the agent is, or whether an action made task
progress. It is disabled in evaluation.

The bound preserves undiscounted reward dominance. Including ordinary step costs, a timeout without success can
return at most -0.028, -0.156, or -0.284 on Navigate, Unlock, and Retrieve respectively. A successful
episode at the maximum horizon remains positive. A second invariant front-loads the entire curiosity
budget against the latest possible success and proves success still has the larger discounted return
at gamma 0.995. Lower gamma values are rejected because the ordering can invert. Changing the
curiosity formula, scale, budget, or reset behavior creates a new declared protocol; it is not tuned
against the final suite.

## Optimization and publication order

Experience collection, optimization, evaluation, and publication are distinct phases:

1. collect one complete recurrent-PPO rollout;
2. apply all PPO optimizer epochs for that rollout;
3. update the trained-timestep counter;
4. publish any due rolling checkpoint and matching digest sidecar;
5. run any due frozen exam against those trained parameters, publishing a promotion or mastery
   artifact if warranted.

The status record exposes collected and trained timesteps separately. A checkpoint or evaluation is
never labeled with experience that its parameters have not yet learned from. A successfully completed
model receives an explicit final exam even when its last boundary is not a scheduled interval. An
interrupted or failed partial rollout is reported and discarded; it is never published as trained.
Every exam records the path and SHA-256 digest of the exact checkpoint bytes it measured. At most one
curriculum transition may occur at a trained boundary, so a scheduled exam and terminal remeasurement
cannot move an unchanged policy through two lessons.

## Evaluation separation

Validation seeds may affect automatic promotion. Final-test seeds may be used only after training
has stopped and the selected checkpoint has been frozen. Any accidental use of final-test seeds
during training invalidates the final generalization claim.

The scripted oracle uses complete state and pathfinding solely to reject broken generators. Its
actions are never added to a replay buffer, imitation dataset, reward calculation, observation, or
policy prompt.

## Checkpoint and resume semantics

A checkpoint consists of the model archive and a matching JSON sidecar. Together they identify the
effective model, environment, and evaluation configuration; curriculum tier; trained counters;
segment and worker seed-stream identity, protocol, parent artifact, and archive digest. Resuming an
older checkpoint uses its own sidecar; it never borrows the latest dashboard state from the parent
run. A child segment deliberately begins a distinct worker seed stream rather than pretending to
continue a half-finished environment episode.

`--total-timesteps` is a segment budget. On a resumed command it means additional experience to
collect in the new child run, not a new lifetime target. The new manifest records the parent while
preserving lifetime counters for charts. A resume with an incompatible protocol, structural PPO
setting, or weakened evaluation gate must fail clearly rather than silently relabeling the restored
model. Policy sampling, optimizer shuffling, and environment resets begin a new recorded RNG stream
in every child segment.

## Claim ladder

1. **Engine valid:** the oracle solves every sampled generated level.
2. **Training operational:** parameters update and checkpoints resume.
3. **Tier learned:** a frozen checkpoint passes its unseen validation threshold.
4. **Retention demonstrated:** that checkpoint still passes every earlier tier.
5. **Composition demonstrated:** the frozen checkpoint solves Tier 2 from pixels and recurrent
   memory with no restore.
6. **Generalization demonstrated:** the selected checkpoint passes the untouched final suite.

No lower level implies a higher one.
