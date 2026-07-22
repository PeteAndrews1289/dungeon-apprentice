# Dungeon Apprentice contributor instructions

## Project contract

- The learning agent receives pixels and its own recurrent state only. Do not add coordinates,
  map IDs, shortest paths, object labels, oracle actions, or mission text to policy observations.
- Trainer-visible `info` fields may grade outcomes and create reports, but may never select or
  replace an action.
- The scripted oracle proves generated levels are solvable. Its actions are never training data.
- Validation seeds begin at `10_000_000`; final-test seeds begin at `20_000_000`. Training code
  must not use either partition.
- Curriculum promotions come only from frozen deterministic evaluation. Do not promote from
  rollout reward, training loss, or a hand-observed dashboard frame.
- Do not silently change rules during a declared run. A mechanics, observation, reward, evaluation,
  or promotion change creates a new protocol version.

## Local checks

Run before committing:

```bash
.venv/bin/ruff check .
.venv/bin/pytest
.venv/bin/dungeon-qualify --seeds 100
```

Training smoke test:

```bash
.venv/bin/dungeon-train --total-timesteps 2048 --evaluation-episodes 2 --no-dashboard
```

## Artifact policy

- Generated models, frames, checkpoints, and run directories stay under ignored `runs/`.
- Public experiment records contain configuration, aggregate metrics, hashes, and original charts;
  they do not contain training artifacts unless a later decision explicitly adds a release format.
- Checkpoints and status files are written atomically.

