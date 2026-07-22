# Dungeon Apprentice contributor instructions

## Project contract

- The active experimental protocol is `dungeon-apprentice-v0.1`. Results from the original v0
  canary are engineering evidence only: its intrinsic reward and evaluation timing invalidate it as
  capability evidence.
- The learning agent receives pixels and its own recurrent state only. Do not add coordinates,
  map IDs, shortest paths, object labels, oracle actions, or mission text to policy observations.
- Trainer-visible `info` fields may grade outcomes and create reports, but may never select or
  replace an action.
- The scripted oracle proves generated levels are solvable. Its actions are never training data.
- Validation seeds begin at `10_000_000`; final-test seeds begin at `20_000_000`. Training code
  must not use either partition.
- Curriculum promotions come only from frozen deterministic evaluation. Do not promote from
  rollout reward, training loss, or a hand-observed dashboard frame.
- Training-only intrinsic reward must remain bounded below the task-success signal, pay nothing for
  an unchanged observation, and be absent from evaluation. Keep both raw and discounted
  reward-dominance tests green; timeout is a terminal failed quest, not a bootstrap truncation.
- Exams and public checkpoints must describe trained parameters. Schedule them only after a PPO
  optimizer phase, and record collected and trained timesteps separately.
- A resumable checkpoint is a bundle: policy archive plus matching sidecar state. Never infer the
  curriculum or counters from a run's latest dashboard status when resuming an older checkpoint.
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
.venv/bin/dungeon-train \
  --total-timesteps 64 --workers 1 --rollout-steps 64 --batch-size 64 \
  --evaluation-every 64 --evaluation-seeds 1 --checkpoint-every 64 \
  --frame-every 64 --qualification-seeds 1 --no-dashboard
```

## Artifact policy

- Generated models, frames, checkpoints, and run directories stay under ignored `runs/`.
- Public experiment records contain configuration, aggregate metrics, hashes, and original charts;
  they do not contain training artifacts unless a later decision explicitly adds a release format.
- Checkpoint archives, checkpoint sidecars, and status files are written atomically.
- Long-run checkpoint retention is bounded; initial, promotion, mastery, and final artifacts are the
  intentional exceptions. Refuse to begin a run when the target volume lacks the configured safety
  margin.
