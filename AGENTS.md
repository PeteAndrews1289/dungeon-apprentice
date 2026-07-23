# Dungeon Apprentice contributor instructions

## Project contract

- The one-shot post-training confirmation of `dungeon-apprentice-v0.2-u2` is terminal
  `capability_failed`: two policies passed, while child `20260745` scored 169/200 against the frozen
  170/200 U2 gate. The completed cohort, confirmation attempt, and U0/U1 predecessors are frozen
  evidence; do not modify their environments, reports, checkpoints, launchers, or declared results.
  U3 protected work remains closed. The prospective
  `dungeon-apprentice-v0.2-u2r-stability` successor is implemented, qualified, and externally
  anchored, but training has not begun.
  It may use only child `20260745`'s exact mastery archive
  `56dc459fb94110f41a14b2304425f572fc235c8cfc07e1152306cbf77ac33dee`, including its optimizer,
  and only the 360,448 actions left under that child's original 1,048,576-action U2 ceiling. Results
  from the original v0 canary remain engineering evidence only: its intrinsic reward and evaluation
  timing invalidate it as capability evidence.
- U2r must use annotated training tag `u2r-stability-v0.2-u2r-20260723`, algorithm seed
  `20260749`, and worker streams `20260749`–`20260752`. All eleven 32,768-action windows run; only
  the exact full-budget terminal artifact may qualify. Do not select an earlier or best-looking
  checkpoint, reset the optimizer, extend the budget, or imply that writing this protocol means
  training has started.
- A U2r interruption may resume only from the launcher's authenticated chain tip. Segment zero is
  `v02-u2r-seed-20260745`; successors are contiguous `-resume-N` directories and add
  `N × 100,000` to process RNG streams. Never accept a caller-selected checkpoint or run name,
  overwrite a segment, omit carried exam/case evidence, or continue without a durable record of the
  four actual active episodes and the exact safe policy/optimizer boundary.
- U2r training retains the unchanged U2 generators, pixels, actions, PPO, reward, normal
  50/7.5/7.5/35 transition mix, and recovery controller. It uses a static exact-layout exclusion set
  built from already exposed qualification, validation, completed three-lineage history, and
  original-confirmation journal evidence. Do **not** require layouts encountered within U2r
  training to be globally unique; that would change the training distribution. Persist every U2r
  episode-start and active-worker layout identity so a later confirmation can exclude the complete
  U2r history.
- The learning agent receives pixels and its own recurrent state only. Do not add coordinates,
  map IDs, shortest paths, object labels, oracle actions, or mission text to policy observations.
- Trainer-visible `info` fields may grade outcomes and create reports, but may never select or
  replace an action.
- The scripted oracle proves generated levels are solvable. Its actions are never training data.
- Validation seeds begin at `10_000_000`; final-test seeds begin at `20_000_000`. Training code
  must not use either partition.
- The four U2 confirmation streams at `15_200_000`–`15_239_999` were consumed by the terminal
  attempt. Never regenerate, reopen, rescore, retry, replace, resume, or reuse those cases;
  claim-bound read-only verification of their immutable stored journal and result evidence remains
  permitted. The prospective U2r streams at `15_240_000`–`15_279_999` remain structurally sealed
  during implementation, qualification, training, development grading, and terminal selection.
  They may open once only for an eligible full-budget terminal artifact under the separately
  anchored no-update evaluator.
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
