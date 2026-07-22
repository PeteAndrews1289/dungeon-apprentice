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

### Next declared test

Run independent Navigate learnability trials from random initialization. Do not modify the game in
response to one exam. If Navigate remains flat across several adequately spaced exams and seeds,
compare a small declared set of exploration settings on the same validation suite without consulting
the untouched final suite.
