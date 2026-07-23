# Dungeon Apprentice

**Can one visual agent learn the idea of a quest—and use it in a dungeon it has never seen?**

Dungeon Apprentice is an original procedural game and a cumulative-learning experiment built on
MiniGrid. A local recurrent policy begins with random parameters and receives only a small pixel
view. It must learn navigation, interaction, memory, backtracking, and eventually combinations of
those skills without demonstrations, walkthroughs, coordinates, online model decisions, or human
controller actions.

The project begins deliberately small. Protocol v0.1 has three capability tiers:

| Tier | Objective | New demand |
| ---: | --- | --- |
| 0 — Navigate | Reach the green exit | Vision, movement, collision |
| 1 — Unlock | Find a colored key, unlock its door, reach the exit | Interaction and ordering |
| 2 — Retrieve | Unlock the door, collect the relic, return to the entrance | Memory and backtracking |

Every layout is generated from a seed and mechanically checked for solvability. Training and
evaluation use disjoint seed partitions. A policy advances only through frozen no-update exams on
unseen levels, and every promotion retests earlier tiers to detect forgetting.

## What makes this our game

MiniGrid supplies fast grid simulation and rendering. This repository owns the dungeon generator,
game rules, key-consumption mechanic, objective tiers, reward contract, agent information boundary,
solvability oracle, curriculum, evaluation suites, training runner, dashboard, and future mechanics.

The game can later grow to include multiple keys, decoys, levers, traps, enemies, inventory limits,
movable blocks, light, multiple floors, and composed final adventures. Those additions will be new
declared protocol versions rather than mid-run patches.

## Information boundary

The policy receives a `56 × 56 × 3` partial RGB image. It does not receive:

- its coordinates or facing direction as numbers;
- the full map or a visited-cell map;
- object names, route distance, shortest paths, or mission text;
- the procedural seed;
- oracle actions, demonstrations, pretrained weights, or online language-model output.

Trainer-only information records the tier, seed, success, timeout, and generated-layout signature.
It grades behavior but cannot choose a button.

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,train]"
```

Prove the first 300 generated levels are solvable:

```bash
.venv/bin/dungeon-qualify --seeds 100
```

Launch the initial automatic curriculum:

```bash
.venv/bin/dungeon-train --total-timesteps 1000000
```

The training process writes a run manifest, atomic checkpoints, live status, evaluation history,
frames, and a local dashboard. Once launched from an ordinary Terminal session, gameplay and
learning are entirely local and consume no GPT or Codex usage.

The learner is recurrent PPO: a compact vision network interprets pixels, an LSTM carries memory
between steps, and PPO updates the policy from its own attempts. During training, bounded episodic
curiosity pays only for genuinely new pixel views. The bonus defaults to `0.002` per novel view and
can contribute at most `0.1` across an entire attempt; repeated or unchanged views pay zero.
Curiosity knows nothing about keys, doors, coordinates, routes, or objectives. Evaluation disables
it and measures game success alone.

## Current experimental status

The first v0 canary proved that the software could train and occasionally discover the exit, but it
did **not** establish learned capability. Its curiosity bonus could make unsuccessful wandering more
valuable than an efficient solution, and its scheduled exams ran before the newest PPO update. At
4,096 collected steps, the scheduled checkpoint had 70 optimizer updates; the final archive had 80
and was never evaluated.

Protocol v0.1 corrected those faults, treats v0 as engineering evidence only, and added reproducible
checkpoint/resume state plus a real train-save-resume-reload-evaluate CI check. Its local and GitHub
engineering gates passed. Three subsequent fresh v0.1 canaries then produced the first replicated
capability result: all three learned Navigate, reaching 90% held-out success after 163,840–196,608
trained actions.

The promotion exposed the next two problems. Every policy ended at exactly 75% Navigate retention,
and all three scored 0% on deterministic full Unlock. The supposed 30% Navigate rehearsal share was
only 4–6% of actual post-promotion transitions because short Navigate episodes and long Unlock
timeouts were sampled as equal episodes. See
[the immutable canary report](docs/results/v0.1-navigate-canaries.md).

The first [v0.2](docs/protocol-v0.2-design.md) implementation slice corrected that scheduler and
introduced a short but complete Visible Unlock quest. Three independently initialized policies all
mastered the key-to-door-to-exit sequence while retaining Navigate; their final U0 scores were
79/80, 80/80, and 80/80. The frozen
[replication report](docs/results/v0.2-visible-unlock-replication.md) records the full trajectories,
panels, allocations, artifact digests, and narrow claim.

Those replications shared one development validation suite, so a
[post-training confirmation](docs/v0.2-u0-confirmation-plan.md) was preregistered before inspecting
two disjoint 200-case blocks. All three selected checkpoints passed every overall and panel gate:
their U0 scores were 187/200, 199/200, and 198/200 while Navigate remained between 93.5% and 96.0%.
The frozen [confirmation report](docs/results/v0.2-u0-confirmation.md) supports advancing to U1
Local Unlock without claiming that unrestricted Unlock has already been learned.

The next declared experiment is now implemented and preregistered as a
[warm-start U1 child](docs/protocol-v0.2-u1-development.md). It inherits the exact confirmed
seed-`20260725` U0 policy and optimizer, then learns on maps where the key begins visible but the door
does not. Every boundary retests Navigate, U0, and U1; forgetting either earlier skill activates
targeted rehearsal. The child receives no trajectories or demonstrations, and its pre-update U1
baseline cannot count as learning. The exact launcher is `scripts/run_v02_u1_lead.sh`.

## Evidence standard

Training reward, loss, map coverage, and one lucky completion are diagnostics. The behavioral
authority is deterministic evaluation on held-out seeds:

- promotion threshold: at least 90% success on the current tier;
- retention threshold: at least 80% on every earlier tier;
- validation suite: seeds beginning at `10_000_000`;
- untouched final suite: seeds beginning at `20_000_000`.

Collected experience and trained experience are reported separately. Exams and public checkpoints
are emitted after optimization. Every exam record names the exact checkpoint path and SHA-256 digest
it graded, while milestone rates and movement diagnostics make partial progress visible without
changing the pixels supplied to the policy.

See [the experiment contract](docs/experiment-contract.md),
[learning design](docs/learning-design.md), [operations runbook](docs/runbook.md),
[experiment log](docs/experiment-log.md), [video notebook](docs/video-narrative.md),
[architecture](docs/architecture.md), [audit and remediation record](docs/audit.md), and
[roadmap](docs/roadmap.md). The completed v0.1 capability result is preserved in the
[Navigate canary report](docs/results/v0.1-navigate-canaries.md), and the next proposed protocol is
specified in [the v0.2 design](docs/protocol-v0.2-design.md). The currently executable successor is
the [U1 Local Unlock child](docs/protocol-v0.2-u1-development.md).
