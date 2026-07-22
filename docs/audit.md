# Framework audit and v0.1 remediation

## Verdict

The experiment is feasible on the development Mac at its present scale, but protocol v0 was not a
valid long-run capability test. The visual information boundary, recurrent policy, procedural engine,
seed separation, and independent solvability oracle survived review. Reward shaping, PPO lifecycle
timing, resume state, CI coverage, and future mechanics boundaries required correction.

This document preserves the problems as part of the result. “We found the experiment was grading the
wrong behavior and stopped” is better evidence than an uninterrupted but uninterpretable curve.

## Release-blocking findings

### A-001 — Intrinsic reward could dominate task success

**v0 behavior:** curiosity paid `0.01 / sqrt(visits)` while an ordinary step cost 0.001. It also paid
for repeated views. A long failed episode could accumulate more shaped return than a short successful
one.

Measured during the audit:

| Behavior | Mean or exact combined return |
| --- | ---: |
| Wait through Navigate | +0.075 |
| Random failed Unlock attempt | +1.034 mean |
| Random failed Retrieve attempt | +1.437 mean |
| Oracle Unlock success | +1.165 mean |
| Oracle Retrieve success | +1.264 mean |

**Consequence:** validation still required real success, but PPO was encouraged to wander rather than
discover and repeat the quest. v0 learning curves cannot establish capability.

**v0.1 disposition:** pay 0.002 exactly once per novel pixel fingerprint, pre-mark the reset view,
and enforce a hard 0.1 episode budget. Repeated/unchanged observations pay zero. The maximum failed
timeout return is negative in all three tiers, and maximum-horizon success remains positive. A
separate discounted-return bound front-loads failure's curiosity and keeps late success preferable at
the protocol's minimum gamma of 0.995; unsafe lower discounts are rejected before a run is created.

### A-002 — Scheduled exams measured stale policies

**v0 behavior:** checkpoint and evaluation callbacks fired inside rollout collection, before PPO
optimized on that rollout.

| Artifact | Collected steps in label | Optimizer updates |
| --- | ---: | ---: |
| `step-2048.zip` | 2,048 | 30 |
| `step-4096.zip` | 4,096 | 70 |
| `completed.zip` | 4,096 | 80 |

The completed model was not evaluated. The recorded 4,096-step score belonged to the 70-update model.

**Consequence:** artifact names and charts overstated which experience a policy had learned from.

**v0.1 disposition:** treat a complete rollout plus its optimizer phase as the indivisible scheduling
boundary. Record collected and trained timesteps separately. Run due exams and publish public
checkpoints only after updates, including a final terminal evaluation.

## Reproducibility and operations findings

### A-003 — Resume restored weights, not the complete experiment

The model archive retained structural PPO settings even when the next command requested different
ones, while the new manifest could record the requested values as though they were effective. An
older checkpoint could also inherit a newer curriculum tier from the parent run's latest status.
Lifetime throughput was divided by segment wall time, inflating the resumed dashboard rate.

**v0.1 disposition:** pair every public archive with a JSON sidecar containing effective
configuration, counters, curriculum tier, segment/worker seed-stream identity, protocol, parentage,
and archive digest. Resume from that exact pair. A resumed `--total-timesteps` is an additional
segment budget, and segment/lifetime rates are reported separately.

### A-004 — CI did not exercise the training stack

The original workflow installed development dependencies, ran unit tests, and qualified generated
levels, but never installed PyTorch or recurrent PPO. It could not catch save/reload, callback-order,
or training-only dependency failures.

**v0.1 disposition:** add a CPU job that installs `dev,train`, collects a complete short rollout,
optimizes it, publishes a checkpoint bundle, resumes a second segment, checks its lifetime and
segment counters, reloads the child archive, and evaluates an unseen seed. Direct runtime and
training dependencies are pinned to the audited versions.

### A-005 — Storage and unattended execution were unbounded

At roughly 22 MiB per policy archive, keeping one checkpoint every 50,000 steps would consume about
44 GiB over 100 million steps before logs and frames. Internal storage had approximately 26 GiB free;
the external T7 Developer volume had approximately 219 GiB free during the audit.

**v0.1 disposition:** long runs target the T7, preserve rolling recent checkpoints plus intentional
initial/promotion/mastery/final artifacts, and enforce a free-space safety margin. The runbook
includes macOS display sleep prevention without requiring the monitor to remain on.

## Capability risks that remain open

Fixing the framework does not prove PPO can learn the tasks. Random exploration found Navigate
success occasionally but produced 0/100 random successes on Unlock and Retrieve during the audit.
Those tiers are discovery cliffs.

The next capability work should introduce automatic procedural difficulty ramps that still begin
every episode from the normal start: initially visible/nearby key-door arrangements, then increasing
separation, occlusion, obstacles, and dead ends. Promotion must continue to use the full held-out tier,
not the simplified training distribution. This changes problem frequency, not action labels, rewards,
demonstrations, or savestates.

The framework is also not yet mechanically plug-in scalable. Tier-specific branches are acceptable
until basic learnability is established. Before adding levers, enemies, or floors, replace them with
task/mechanic registries, explicit success predicates, procedural blueprints, and a capability graph.

## Hardware envelope

The audited M1 iMac with 8 GB memory is suitable for one compact CPU trainer:

| Configuration or horizon | Observed/estimated result |
| --- | ---: |
| v0 10-epoch PPO | about 48–56 steps/second |
| v0.1 four-epoch benchmark | about 110 steps/second |
| 1 million steps | about 2.5–5 hours across those measurements |
| 10 million steps | about 1.1–2.1 days |
| 100 million steps | about 10.5–21 days |

MPS was slower in the audit, and adding in-process environments did not improve throughput because
optimization was the bottleneck. Protocol v0.1 therefore declares CPU, four workers, and four PPO
epochs, but faster throughput is not evidence of equal learning quality. Use sequential experimental
seeds and compare learnability before optimizing further. Estimates are planning numbers, not a
promise; each run records its measured segment rate.

## v0.1 release gate

Do not begin the first long capability run until all of the following are true:

- reward-dominance and unchanged-view curiosity tests pass;
- a post-update exam names the same digest as its checkpoint bundle;
- train → save → reload → resume → evaluate passes locally;
- the training smoke job passes in GitHub Actions;
- generator qualification passes all three tiers;
- a deliberately interrupted child run restores exact counters and curriculum state;
- storage retention and free-space refusal are exercised on a temporary run root;
- the public experiment log clearly labels v0 invalid for capability claims.

Local status on July 22, 2026: reward invariants, checkpoint digest linkage, real train/resume/evaluate,
300-level qualification, interrupted-child recovery, retention/free-space tests, and public v0
invalidation are complete. The published GitHub training smoke remains open.

Passing this gate establishes a trustworthy instrument. Learnability must still be demonstrated by
independent v0.1 runs and frozen unseen-seed exams.
