# Dungeon Apprentice contributor instructions

## Project contract

- The one-shot post-training confirmation of `dungeon-apprentice-v0.2-u2` is terminal
  `capability_failed`: two policies passed, while child `20260745` scored 169/200 against the frozen
  170/200 U2 gate. The completed cohort, confirmation attempt, and U0/U1 predecessors are frozen
  evidence; do not modify their environments, reports, checkpoints, launchers, or declared results.
  Results from the original v0 canary remain engineering evidence only: its intrinsic reward and
  evaluation timing invalidate it as capability evidence.
- The first externally anchored `dungeon-apprentice-v0.2-u2r-stability` launch is an immutable
  zero-action operational failure. The corrected
  `dungeon-apprentice-v0.2-u2r-stability-r1` successor then completed all eleven fixed windows and
  reached its exact 360,448-remediation-action terminal boundary. Its valid scientific verdict is
  `failed`: terminal U2 remained 76/80 with panels 37/40 and 39/40, but mean ineffective
  interactions were 4.1625 against the frozen maximum of 3.0. The authenticated report SHA-256 is
  `dcfbcbc9fb3e042d44c1bb7762479f005a24a989a96611b85b102c34f955fcc2`; see
  `docs/results/v0.2-u2r-r1-stability.md`.
- Both U2r roots and their tags are now terminal evidence. Never resume, rename, reuse, prune,
  overwrite, or continue either root. Never substitute r1's favorable 1,015,808-child-action
  penultimate checkpoint for its required 1,048,576-action terminal artifact. The reserved
  `15_240_000`–`15_279_999` confirmation candidates were not opened and may not be repurposed as a
  U2-S development exam. U3 remains closed.
- The next prospective decision is the matched
  `dungeon-apprentice-v0.2-u2s-stability-ablation` in
  `docs/protocol-v0.2-u2s-stability-ablation.md`. This branch contains its implementation and
  preflight tests, but source and documentation alone do not imply qualification, external
  anchoring, canonical root creation, or training. Before action one, a clean release must bind the
  protocol, exact parent, four interventions, seeds, lesson-specific guard digests, terminal rule,
  protected partitions, storage boundary, and dashboard identity.
- U2-S may load only confirmed U1 child `20260733`, archive SHA-256
  `3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104`, including its optimizer.
  It may not load any U2, U2r, or earlier U2-S policy. Four sequential arms—control, conservative,
  no-effect, and combined—share algorithm seed `20260753`, worker streams
  `20260753`–`20260756`, and exactly 1,048,576 new U2 actions apiece. Arms are matched development
  trajectories, not four independent replications.
- The conservative U2-S intervention changes only learning rate (`2.5e-4` to `2.5e-5` linearly),
  clip range (`0.10`), PPO epochs (2), and target KL (`0.015`). The experience-only intervention
  adds `-0.01` on the second and later consecutive transition in which the same pickup/drop/toggle
  action leaves the next visible pixel bytes unchanged, capped at `-0.10` per episode. Any pixel
  change, action change, noninteraction, reset, or termination resets the streak. The rule receives
  no coordinates, objects, milestones, oracle state, or trainer-only `info`, stays outside the
  observation, and is absent from evaluation.
- All U2-S arms must run the full budget with exams every 32,768 post-update actions. Only the fixed
  final three exams may decide the arm. Each must pass all existing gates, score U2 at least 72/80
  with both panels at least 34/40, keep U0/U1/U2 mean ineffective interactions at most 3.0, contain
  no case with 10 or more ineffective interactions, and contain no identical pickup/drop/toggle
  action run of length 10 or more. Eligible configurations use the frozen
  simplest-intervention priority `control > conservative > no-effect > combined`.
- U2-S selects a configuration only. Never promote or reuse an ablation checkpoint. If no arm is
  eligible, the mechanism study fails and neither a successor cohort nor U3 opens. If one is
  eligible, only a new separately committed protocol may train fresh multi-lineage children from
  confirmed U1 parents. The ablation itself cannot use confirmation or final seeds and cannot
  activate U3.
- U2-S is non-resumable because continuing only one factorial arm from reset environment, RNG, or
  recurrent state would break the matched comparison. Any interruption or crash makes the whole
  canonical cohort terminal `operationally_incomplete`. A replacement must use a separately
  committed and tagged protocol-attempt identity and root, and restart all four arms fresh from the
  exact confirmed U1 parent.
- U2-S retains r1's lesson-specific training guards: Navigate, U1, and U2 use same-lesson
  historical novelty, while finite U0 rejects only its 79 unique development layouts and reports
  other historical overlap diagnostically. Freeze one static pre-ablation guard inventory for all
  four arms; do not add an earlier arm's layouts to a later arm's hard guard. The trainer may never
  load old evidence images, scores, roles, seeds, or action traces, and within-arm layouts need not
  be globally unique. Persist every episode-start and active-worker layout identity.
- The learning agent receives pixels and its own recurrent state only. Do not add coordinates,
  map IDs, shortest paths, object labels, oracle actions, or mission text to policy observations.
- Trainer-visible `info` fields may grade outcomes and create reports, but may never select or
  replace an action.
- The scripted oracle proves generated levels are solvable. Its actions are never training data.
- Validation seeds begin at `10_000_000`; final-test seeds begin at `20_000_000`. Training code
  must not use either partition.
- The four U2 confirmation seed streams at `15_200_000`–`15_239_999` were consumed by the terminal
  attempt. Never regenerate those seeds, reopen or feed their stored cases to training, rescore,
  retry, replace, or resume that exam; claim-bound read-only verification of its immutable journal
  and result evidence remains permitted. The finite U0 generator may independently reproduce a
  consumed exact layout from an ordinary `0`–`999_999` training seed under the explicit r1
  diagnostic-overlap rule above. The prospective U2r streams at
  `15_240_000`–`15_279_999` remain structurally sealed during implementation, qualification,
  training, development grading, and terminal selection. They may open once only for an eligible
  full-budget terminal artifact under the separately anchored no-update evaluator.
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
