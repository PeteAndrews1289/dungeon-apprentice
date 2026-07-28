# v0.3 Stage-A r2: process-safe action-effect architecture replacement

Status: **implemented replacement in development; not yet frozen, tagged, qualified, or launched**

This document is the second operational replacement for the matched
`dungeon-apprentice-v0.3-action-effect-architecture` Stage-A study. It incorporates the scientific
contract in
[`protocol-v0.3-action-effect-architecture.md`](protocol-v0.3-action-effect-architecture.md)
and the dashboard amendment in
[`protocol-v0.3-action-effect-architecture-r1.md`](protocol-v0.3-action-effect-architecture-r1.md).
It does not change the learner, observation, parent transplant, reward, PPO, curriculum, seeds,
budgets, evaluation cases, arm order, or selection rule.

r2 changes only the process-control boundary exposed by the terminal
[r1 operational failure](results/v0.3-action-effect-stage-a-r1-operational-failure.md). It assigns
fresh identities and restarts both matched arms from the confirmed U1 parent. It is not a resume
and cannot load any r1 checkpoint.

## Why r2 exists

r1 corrected the attempt-0 dashboard timeout and successfully reached training. It authenticated
source commit `b9dc80b4c5f95b4b0a8bf7a681b9431764c6f0ec`, published annotated tag object
`1f11b98fae6691dc9282682c44cdaa33ea55e5ab`, passed qualification, opened the dashboard, and
started sham.

Its launcher then counted neural trainers by searching complete process command lines. The fixed
supervisor included its child trainer command after `--`, so both the supervisor and child matched
the trainer expression. The resulting false cardinality failure was returned from a zsh function
under `set -e`; that implicit error path did not run the intended `EXIT` cleanup trap. Sham
continued until explicitly stopped.

The preserved boundary is:

| r1 fact | Frozen value |
|---|---:|
| Trained sham child actions | `38,912` |
| Collected actions in final status | `40,004` |
| Collected actions in episode ledger | `40,960` |
| New optimizer updates | `76` across `19` phases |
| Scheduled exam boundaries | `1 / 32` |
| Evaluation rows / deterministic cases | `8 / 640` |
| Latest safe checkpoint | none |
| Action-effect arm | never started |
| Scientific comparison | never reached |

r1 is terminal `operationally_incomplete`. Its latest-observed archive and first-exam archive are
authentic, but neither is safe, promotable, resumable, reusable, or selectable.

## Immutable predecessor evidence

r2 qualification must authenticate all r1 evidence before it may create an r2 cohort:

| Evidence | Frozen r1 identity |
|---|---|
| Source commit | `b9dc80b4c5f95b4b0a8bf7a681b9431764c6f0ec` |
| Annotated tag | `action-effect-architecture-v0.3-stage-a-r1-20260724` |
| Tag object | `1f11b98fae6691dc9282682c44cdaa33ea55e5ab` |
| Tag-payload SHA-256 | `02531149bcac991fd4255897a6ea2729f751dc499ca8e7576b5221983f385a64` |
| Qualification report SHA-256 | `c59c3033a892bb05fea437bc525090c4f958110c960494fc2366444526d2dcb7` |
| Qualification claim SHA-256 | `86f6473d5610bbc80e5487db039cc99dc92a7a9558c0b7a7237315b306519e9e` |
| Qualification checksum-file SHA-256 | `3b12be77dee66a7cafdf8bfb86cbfc380c2688eeb9b33d057d559634004c35a2` |
| Cohort contract SHA-256 | `647f8237c2dfcb8451a6ccb644e2b6b10f1087bd9a5b5d7a3dcc0b23426fe22d` |
| Cohort-state SHA-256 | `d989b884d1ff63ed87194b196f87e4396673200762167fa1f9de0c735727b83b` |
| Launcher-log SHA-256 | `598df636ad3f08538636f25311b20933849dc51386335e8dbc65ae76f5a95d7e` |
| First-rollout identity SHA-256 | `fa4c7bda99a261f8fa49741a49360cd1bfc6ab3081db51aeffc64266a109ce72` |
| First-exam checkpoint SHA-256 | `bcc9342942c672f60e00d547899f6a4cfe1a87a58df01527a6e70f9f38948eb8` |
| Latest-observed checkpoint SHA-256 | `0124df33809e0b81ccef181a98060617130ae12ba7eff116c8675fb3a27ac943` |
| Canonical 33-file evidence-map SHA-256 | `eb529df93a43be803d47ba73069ac428bdf953065aa7258d34c2d1dd333bbdc2` |
| Canonical failed-r1 evidence-object SHA-256 | `5d8038681355264b51c51fd4b5c2b93d3e49804286bc78c8d89a1efd21265110` |

Qualification must also verify the exact inventories: 29 cohort files, an empty sham media
directory, no action-effect directory, eight evaluation rows, 1,920 episode rows, 19 optimizer
rows, the lagging status-versus-ledger collection counts, no safe checkpoint, and no terminal
scientific report or process-closeout file.

Never modify, rename, resume, prune, overwrite, or use as a parent:

- `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r1-20260724`
- `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r1-media-20260724`
- `/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.3-action-effect-stage-a-r1-20260724`

Attempt 0 remains independently frozen under its original identities. r2 must authenticate both
historical failures without conflating zero-action attempt 0 with partial-sham r1.

## Fresh r2 identity

| Item | Prospective r2 assignment |
|---|---|
| Cohort ID | `v0.3-action-effect-stage-a-r2-20260724` |
| Annotated source-preregistration tag | `action-effect-architecture-v0.3-stage-a-r2-20260724` |
| Qualification root | `/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.3-action-effect-stage-a-r2-20260724` |
| Cohort root | `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r2-20260724` |
| Media root | `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r2-media-20260724` |
| Dashboard | `http://127.0.0.1:8790/` |
| Sole launcher | `scripts/run_v03_action_effect_stage_a_r2.sh` |

These are assignments, not completed release claims. The r2 source commit, annotated tag object,
tag-payload SHA-256, qualification claim/report SHA-256 values, and cohort contract SHA-256 do not
yet exist and must not be guessed or copied from r1.

No r2 qualification, cohort, media root, dashboard process, trainer, policy action, checkpoint, or
result exists while the source is still under development.

## Scientific contract remains unchanged

Both arms reconstruct exact confirmed U1 child `20260733`, checkpoint SHA-256
`3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104`, including every compatible
legacy Adam moment by parameter name.

The fixed order remains:

1. `sham`: the identical Dict architecture with all nine action-effect values held at zero;
2. `action-effect`: the same architecture with the preceding self-selected action one-hot and the
   next visible-frame changed/unchanged one-hot.

Both arms retain:

- architecture initialization seed `20260756`;
- algorithm seed `20260757`;
- worker streams `20260757`–`20260760`;
- four workers and 512 rollout steps per worker;
- four PPO epochs and 2,048 transitions per optimizer phase;
- exactly 1,048,576 fresh child actions apiece;
- exams every 32,768 post-update actions, 32 exams per arm;
- the unchanged U2 control reward, PPO, curriculum, lesson mechanics, and history guards; and
- the complete U2-S terminal-three capability-and-stability gate.

The new residual starts exactly at zero. Before action one, legacy features, logits, values,
deterministic actions, and recurrent states must match the U1 parent exactly. The complete first
2,048-transition sham and action-effect trajectories must also match exactly. Behavioral
divergence may begin only after the declared first optimizer phase.

Sham remains calibration only. Only an independently eligible action-effect architecture
definition can authorize a separately committed three-lineage replication protocol. Stage A
cannot promote a development checkpoint or open U3.

## Corrected process boundary

### Leading-argument role classification

Process roles must be determined from their leading executable arguments:

- a trainer is a process whose executable or Python `-m` target is the trainer;
- a supervisor is a process whose executable or first Python script argument is
  `u2_trainer_supervisor.py`;
- `caffeinate` is classified by its executable name; and
- a dashboard is classified by its actual Python module target.

Text appearing later inside a supervisor's child-command arguments, a shell command, a test
command, or an evidence scanner must not change that process's role. The live supervised chain
must contain exactly one launcher, supervisor, trainer, and `caffeinate` process.

### Explicit abnormal-exit cleanup

The launcher must not rely on zsh implicit `errexit` to invoke cleanup. Every critical assertion
after process creation must use an explicit failure branch that:

1. records the triggering failure and active arm;
2. signals the exact supervisor-owned trainer, waits for a fixed grace period, escalates through
   `SIGTERM`, and, only after reauthenticating its PID, parent, process group, command, and
   supervisor state, kills that trainer's process group within a second fixed deadline;
3. stops and waits for sleep prevention;
4. leaves the read-only dashboard available when it can still serve the terminal failure evidence;
5. records the active arm as interrupted or crashed;
6. marks the whole non-resumable cohort `operationally_incomplete`; and
7. exits nonzero only after evidence has been durably written.

Cleanup must be idempotent so an ordinary signal and a simultaneous assertion failure cannot
double-finish an arm or overwrite evidence.

### Required operational tests

Before source freeze, tests must prove:

- a real supervisor command carrying the trainer command after `--` counts as one supervisor and
  one trainer, not two trainers;
- unrelated command lines containing module names are ignored;
- one duplicate trainer, supervisor, or `caffeinate` process fails closed;
- every post-spawn assertion failure stops the complete owned chain;
- a trainer that ignores both the graceful interrupt and `SIGTERM` is still reaped within the
  bounded escalation window without signalling an unrelated process;
- no trainer, supervisor, or `caffeinate` process remains after failure;
- the lower-level trainer rejects every noncanonical cohort, media, contract, and arm-directory
  mapping, including either immutable failed-attempt root;
- the root becomes terminal `operationally_incomplete`;
- no replacement reuses an r1 checkpoint or root; and
- normal completion still seals `process-closeout.json` before terminal finalization.

## Dashboard and storage boundary

r2 retains r1's corrected dashboard contract. The expensive complete source, tag, qualification,
parent, and predecessor authentication happens once before port `8790` binds. Each request then
rereads live cohort, status, exam, and frame evidence and rehashes immutable contract and
qualification bytes. The dashboard remains read-only and supplies no learner input.

The same fixed caps remain: 2 GiB per arm, 4 GiB scientific cohort, 6 GiB media, 10 GiB combined,
and at least 25 GiB free before launch and each arm.

## Release and launch order

The only authorized sequence is:

1. complete implementation, documentation, and operational tests;
2. commit and push one clean r2 source;
3. generate the canonical one-line r2 tag payload from that clean source;
4. create and publish annotated tag
   `action-effect-architecture-v0.3-stage-a-r2-20260724`;
5. run one-shot qualification into the fresh r2 qualification root;
6. authenticate r1 directly, authenticate attempt 0 transitively through r1's exact tag and
   qualification report, and independently recheck the immutable attempt-0 evidence before launch;
7. verify that the fresh r2 cohort and media roots are absent;
8. run `scripts/run_v03_action_effect_stage_a_r2.sh` exactly once;
9. let sham and action-effect finish sequentially; and
10. stop sleep prevention, seal process closeout, and finalize the matched report.

The chain remains non-circular: the published tag preregisters source and contracts; the later
qualification report binds that exact tag object; the cohort then binds the qualification-report
SHA-256. Never claim that the tag contains the later qualification digest.

r2 has no resume mode. Any interruption or crash makes its complete root terminal
`operationally_incomplete`. A further replacement would require another committed protocol-attempt
identity, annotated tag, one-shot qualification, and two entirely fresh arms from confirmed U1.
