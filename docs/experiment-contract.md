# Version 0 experiment contract

## Question

Can one randomly initialized recurrent pixel policy acquire three reusable capabilities—navigation,
key-and-door interaction, and relic retrieval with backtracking—and retain earlier capabilities as
the procedural task distribution grows?

## Frozen rules

| Setting | Version 0 value |
| --- | --- |
| Protocol | `dungeon-apprentice-v0` |
| Observation | Partial RGB pixels, 56 × 56 × 3 |
| Policy-visible text | None |
| Policy-visible coordinates/map | None |
| Actions | Turn left, turn right, move forward, pick up, drop, toggle, wait |
| Success reward | +1.0 |
| Ordinary step reward | -0.001 |
| Intermediate authored rewards | None |
| Training-only intrinsic reward | 0.01 / sqrt(pixel-view visit count) |
| Training initialization | Random parameters |
| Imported gameplay | None |
| Validation seeds | 10,000,000 and above |
| Final-test seeds | 20,000,000 and above |
| Promotion | ≥90% current-tier success and ≥80% retention |
| Promotion authority | Frozen deterministic policy, no updates |
| Training mixture after promotion | 70% newest tier, 30% prior tiers |

The key is consumed when it opens its matching door. Tier 2 succeeds only when the agent carries
the relic back onto the entrance tile. The agent is not rewarded for seeing or collecting a key,
opening a door, collecting the relic, visiting a new cell, or moving toward an objective.

The intrinsic curiosity signal is computed only from a hash of the current policy-visible pixels,
and its counts reset every episode. It does not know whether a view contains a useful object, where
the agent is, or whether an action made task progress. It is disabled in evaluation. Changing its
formula or scale creates a new declared run protocol; it is not tuned against the final suite.

## Evaluation separation

Validation seeds may affect automatic promotion. Final-test seeds may be used only after training
has stopped and the selected checkpoint has been frozen. Any accidental use of final-test seeds
during training invalidates the final generalization claim.

The scripted oracle uses complete state and pathfinding solely to reject broken generators. Its
actions are never added to a replay buffer, imitation dataset, reward calculation, observation, or
policy prompt.

## Claim ladder

1. **Engine valid:** the oracle solves every sampled generated level.
2. **Training operational:** parameters update and checkpoints resume.
3. **Tier learned:** a frozen checkpoint passes its unseen validation threshold.
4. **Retention demonstrated:** that checkpoint still passes every earlier tier.
5. **Composition demonstrated:** the frozen checkpoint solves Tier 2 from pixels and recurrent
   memory with no restore.
6. **Generalization demonstrated:** the selected checkpoint passes the untouched final suite.

No lower level implies a higher one.
