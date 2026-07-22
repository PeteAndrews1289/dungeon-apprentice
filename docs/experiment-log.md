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

### 4,096-step learning canary

The first multi-update canary used four environments, recurrent PPO, training-only pixel curiosity,
and CPU execution.

| Measurement | Result |
| --- | ---: |
| Wall time | 73.7 seconds |
| Training steps | 4,096 |
| PPO optimizer updates | 70 |
| Completed training episodes | 32 |
| Stochastic training successes | 5/32 (15.6%) |
| Frozen validation at 2,048 | 0/10 |
| Frozen validation at 4,096 | 0/10 |
| Curriculum decision | Remain on Navigate |

This is the intended distinction between discovery and mastery. The agent sometimes reached the
exit while exploring, but its deterministic policy could not yet reproduce that behavior on unseen
layouts. No promotion occurred.

### Next declared test

Run long enough to reveal a validation trend on Navigate. Do not modify the game in response to a
single evaluation. If Navigate remains flat across several adequately spaced exams, compare a small
set of declared exploration settings using the same validation seeds, then select without consulting
the untouched final suite.

