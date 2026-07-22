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

Protocol v0.1 corrects those faults, treats v0 as engineering evidence only, and adds reproducible
checkpoint/resume state plus a real train-save-resume-reload-evaluate CI check. The next result
begins from random parameters under v0.1; it will not be presented as a continuation of v0.

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
[roadmap](docs/roadmap.md).
