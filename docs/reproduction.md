# Reproduction and verification

Dungeon Apprentice is a concluded research project. Its public repository supports strong software
verification and fresh independent experiments, while the historical scientific claims remain
bound to immutable protocols, source tags, run identities, reports, and artifact digests.

This distinction prevents a clean clone from being mistaken for an exact replay of evidence that
was produced under preregistered, one-shot conditions.

## Three useful levels of verification

### 1. Verify the public software

From a clean checkout with Python 3.11 or 3.12:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,train]"
.venv/bin/ruff check .
.venv/bin/pytest
.venv/bin/dungeon-qualify --seeds 100
```

These checks cover the environment, procedural generation, solvability oracle, seed guards,
training and resume contracts, evaluation paths, dashboards, evidence authentication, and the
v0.2–v0.4 experiment machinery. GitHub Actions repeats the core checks on Python 3.12.

### 2. Exercise the training lifecycle

The bounded smoke command from `AGENTS.md` creates a new local run and exercises training,
checkpoint publication, evaluation, and shutdown:

```bash
.venv/bin/dungeon-train \
  --total-timesteps 64 --workers 1 --rollout-steps 64 --batch-size 64 \
  --evaluation-every 64 --evaluation-seeds 1 --checkpoint-every 64 \
  --frame-every 64 --qualification-seeds 1 --no-dashboard
```

This is an engineering verification, not a learning result. Its generated artifacts remain under
the ignored `runs/` directory.

### 3. Run a new independent scientific replication

An independent replication must be prospective. It needs a new protocol and run identity, fresh
development seeds, untouched evaluation partitions, new artifact roots, and a decision rule frozen
before training or evaluation. It must not reuse the names, seeds, checkpoints, confirmation
streams, or roots of the historical cohorts.

The historical launch scripts and runbook are preserved to document the original controls. They
must not be executed as though they were unused launch identities.

## Portable repository versus historical evidence

| Public and portable | Historical and intentionally immutable |
| --- | --- |
| Environment, generator, oracle, agent, trainers, evaluators, dashboards, and tests | Long-run checkpoints, optimizer state, recurrent state, and external run directories |
| Frozen protocols, readable result reports, aggregate metrics, charts, tags, and SHA-256 claims | Preregistered one-shot qualification roots and consumed confirmation candidates |
| Commands for linting, testing, qualification, and fresh local runs | Machine-specific external-volume paths and historical process topology |
| Enough code to audit the information boundary and reproduce the engineering approach | A promise that a new stochastic run will reproduce an exact historical trajectory |

The repository's [artifact policy](../AGENTS.md#artifact-policy) intentionally excludes generated
models, frames, checkpoints, and run directories from source control. Result reports retain the
configuration, aggregate measurements, original charts, identities, and cryptographic digests
needed to audit the stated claims.

## Claim map

For a short review, follow the evidence in this order:

1. [Navigate canaries](results/v0.1-navigate-canaries.md) — replicated initial learning after the
   invalid v0 engineering evidence was set aside.
2. [Visible Unlock replication](results/v0.2-visible-unlock-replication.md) and
   [confirmation](results/v0.2-u0-confirmation.md) — the first cumulative interaction skill.
3. [Local Unlock replication](results/v0.2-local-unlock-replication.md) and
   [collision-safe confirmation](results/v0.2-u1-confirmation-v2.md) — three confirmed inherited
   lineages.
4. [Separated Unlock replication](results/v0.2-u2-separated-unlock-replication.md) and
   [strict confirmation](results/v0.2-u2-confirmation.md) — replicated development success followed
   by a one-case cohort failure.
5. [U2r stability result](results/v0.2-u2r-r1-stability.md) and
   [U2-S ablation](results/v0.2-u2s-r1-stability-ablation.md) — high capability without a stable
   qualifying mechanism.
6. [v0.3 action-effect](results/v0.3-action-effect-stage-a-r3.md) and
   [v0.4 ineffective-trace](results/v0.4-ineffective-trace-stage-a.md) — active learned pathways
   that did not eliminate rare catastrophic tails.

## Reproduction limits

- The public repository does not contain the historical model archives or external evidence roots,
  so it cannot independently recompute every reported policy score from checkpoint bytes.
- Several canonical launchers target macOS, zsh, and an external APFS volume. The Python library and
  core CI checks are portable, but the historical operations layer is not presented as
  platform-neutral.
- Stochastic reinforcement-learning trajectories are not expected to match byte-for-byte under a
  new run. Reproduction means testing the frozen hypothesis and decision rule with fresh identities,
  not manufacturing the old trajectory.
- The final-test partition beginning at `20_000_000` remains untouched. Closing the project does not
  authorize opening it.

These limits are part of the result. They make clear which claims can be checked from a clone,
which require the preserved evidence inventory, and what a legitimate future replication would
need to do.
