# v0.3 Stage-A r1: action-effect architecture replacement

Status: **terminal `operationally_incomplete`; superseded by r2**

> **Historical protocol.** r1 was subsequently committed as
> `b9dc80b4c5f95b4b0a8bf7a681b9431764c6f0ec`, externally anchored by annotated tag object
> `1f11b98fae6691dc9282682c44cdaa33ea55e5ab`, qualified, and launched. Sham reached 38,912
> trained child actions, but a whole-command-line process-count false positive and a zsh
> `errexit` cleanup bypass closed the cohort `operationally_incomplete`. Final status recorded
> 40,004 collected actions; the episode ledger reached 40,960. There is no safe checkpoint, the
> action-effect arm never started, and no scientific comparison occurred. Preserve this frozen
> contract as historical context; the full incident is
> [documented separately](results/v0.3-action-effect-stage-a-r1-operational-failure.md), and the
> fresh replacement is [r2](protocol-v0.3-action-effect-architecture-r2.md).

This document is the separately committed replacement-attempt protocol for the matched
`dungeon-apprentice-v0.3-action-effect-architecture` Stage-A experiment. It incorporates the
scientific design in
[`protocol-v0.3-action-effect-architecture.md`](protocol-v0.3-action-effect-architecture.md)
without changing the model, observations, reward, PPO settings, curriculum, seeds, budgets,
evaluation, or selection rule. Its only protocol change is operational: it records the
zero-recorded-action failure of attempt 0, assigns fresh r1 identities, and corrects the dashboard
health contract that prevented the trainer from starting.

## Why r1 exists

Attempt 0 successfully completed source preregistration and one-shot qualification. Its launcher
then created the cohort contract and read-only dashboard. Each dashboard request repeated the full
qualification and predecessor-evidence authentication and took slightly longer than the launcher's
ten-second HTTP deadline. The clients disconnected before the valid response completed, producing
the preserved `BrokenPipeError` records. The launcher was stopped before creating either arm
directory or trainer supervisor. Its exit trap wrote a synthetic `sham` attempt-0 crash record, so
the cohort correctly became terminal `operationally_incomplete`.

This was an operational failure, not a learning result. No trainer status, supervisor record,
checkpoint, arm directory, media artifact, policy action, child action, or optimizer update exists.
The failed root cannot be resumed or reused.

The complete chronology, byte digests, and negative root inventory are preserved in
[`results/v0.3-action-effect-launch-attempt-0.md`](results/v0.3-action-effect-launch-attempt-0.md).

## Immutable attempt-0 evidence

| Evidence | Frozen identity |
|---|---|
| Source commit | `5b135a4e2953db9f14e83cdaba77fe219fecb160` |
| Annotated tag | `action-effect-architecture-v0.3-stage-a-20260724` |
| Tag object | `9bd59e367b0ccb9890e4ddb5ad0144dfd4897c7c` |
| Qualification claim SHA-256 | `1ab64150f7db79735cdd4bb3192cfad2b9ad244a0944d76fae275a05d1e1c39b` |
| Qualification report SHA-256 | `a3a50ecf91411a27a70e2c6d3e03b93183aa3b078f2f1104b5dd6884f2c3fc85` |
| Qualification checksum-file SHA-256 | `f96599d4d813ff92897a70077f41a1de16a60ec2860e441eb70ab4af4bb25a4e` |
| Cohort contract SHA-256 | `15a180d7a38af6dbc459870b0e4f06a565b291399971fc2165774bce91c6e72c` |
| Cohort state SHA-256 | `c35441207060ab9a5c55e82289924ab0530effc7f0e5e50fb11b2a0ab83f68eb` |
| Dashboard log SHA-256 | `8806979b29d3778f8730546c6bb2f577548844cf891cb06220cd935a18bd95f2` |
| Launcher log SHA-256 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Terminal phase | `operationally_incomplete` |
| Recorded policy/child actions | `0 / 0` |
| Recorded optimizer updates/checkpoints | `0 / 0` |

The r1 source-preregistration payload and r1 qualification both reauthenticate these bytes and
the exact negative inventory. Any new arm directory, media entry, status file, supervisor record,
checkpoint, or changed bound file makes r1 qualification fail closed.

Never modify, rename, resume, prune, overwrite, or use as a training parent:

- `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-20260724`
- `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-media-20260724`
- `/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.3-action-effect-stage-a-20260724`

## Fresh r1 identity

| Item | Frozen r1 assignment |
|---|---|
| Annotated source-preregistration tag | `action-effect-architecture-v0.3-stage-a-r1-20260724` |
| Qualification root | `/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.3-action-effect-stage-a-r1-20260724` |
| Cohort root | `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r1-20260724` |
| Media root | `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r1-media-20260724` |
| Dashboard | `http://127.0.0.1:8789/` |
| Sole launcher | `scripts/run_v03_action_effect_stage_a_r1.sh` |

These roots must not exist before the corresponding step creates them. Attempt 0 and r1 may not
share a qualification claim, tag object, cohort contract, mutable state, arm directory, media
directory, process chain, or checkpoint.

## Scientific contract remains unchanged

Both r1 arms restart from confirmed U1 child `20260733`, checkpoint SHA-256
`3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104`, including its exact legacy
Adam state.

The fixed order is:

1. `sham`: Dict observation and zero-initialized action-effect residual, with its nine context
   values held at zero;
2. `action-effect`: the same architecture, with the preceding self-selected action one-hot and
   visible changed/unchanged one-hot.

Both use architecture seed `20260756`, algorithm seed `20260757`, worker streams
`20260757`–`20260760`, four workers, 512 steps per worker, four PPO epochs, and exactly
1,048,576 child actions. The first complete 2,048-transition rollout and its pre-optimizer policy
outputs must match exactly. Reward, PPO, lesson mechanics, static history guards, exam seeds, exam
frequency, storage caps, and the U2-S final-three selection gate remain unchanged from attempt 0.

Sham is not selectable. An eligible action-effect definition can authorize only a separately
committed three-lineage replication protocol. Stage A cannot promote a checkpoint or authorize U3.

## Corrected dashboard boundary

The dashboard is an evidence reader, never a trainer input. Its health endpoint must:

- read the current r1 manifest and contract without modifying either;
- expose r1 cohort, source, tag object, and qualification identities;
- have its process complete one full source, tag, qualification, and predecessor authentication
  once at startup, before binding its listening socket;
- return the authenticated API payload within the launcher's bounded health deadline;
- re-read live cohort, arm, status, frame, and exam evidence on every request;
- re-hash the immutable cohort contract, qualification report, and qualification checksum on every
  request without repeating the expensive predecessor walk;
- tolerate ordinary client disconnects without terminating the server;
- prevent stale cached state after an atomic manifest replacement;
- and fail closed on immutable contract or qualification-byte tampering.

The launcher must authenticate the health endpoint before starting sleep prevention, claiming an
arm, creating arm directories, or spawning a supervisor. A dashboard failure before those steps
leaves zero scientific actions and still terminalizes that fresh root.

“Healthy” is the launcher's conclusion after validating the expected payload identities. The API
does not return, and the launcher does not depend on, a literal `healthy: true` field.

## Release and launch order

The only authorized order is:

1. commit and push a clean r1 source;
2. generate the canonical one-line r1 tag payload;
3. create and push the annotated r1 tag;
4. run the one-shot r1 qualification into its fresh qualification root;
5. independently authenticate the qualification and attempt-0 evidence;
6. run `scripts/run_v03_action_effect_stage_a_r1.sh` once;
7. let sham and action-effect finish sequentially;
8. seal process closeout and terminal evidence.

Qualification smoke artifacts are disposable and non-scientific. The launcher has no resume mode.
Any r1 interruption or crash makes the complete r1 root terminal `operationally_incomplete`; a
future replacement would require another committed protocol-attempt identity, tag, qualification,
and set of fresh roots.

That contingency occurred. r1 must never be resumed or reused. r2 supplies the separately
committed replacement boundary and restarts both arms from confirmed U1.
