# v0.3 Stage-A r3: canonical-evidence action-effect replacement

Status: **release candidate; annotated release, qualification, and launch are external gates**

This document is the third operational amendment to the matched
`dungeon-apprentice-v0.3-action-effect-architecture` Stage-A study. It incorporates the scientific
contract in
[`protocol-v0.3-action-effect-architecture.md`](protocol-v0.3-action-effect-architecture.md), the
dashboard correction in
[`protocol-v0.3-action-effect-architecture-r1.md`](protocol-v0.3-action-effect-architecture-r1.md),
and the process-control correction in
[`protocol-v0.3-action-effect-architecture-r2.md`](protocol-v0.3-action-effect-architecture-r2.md).

r3 changes no learner-facing or statistical variable. It assigns fresh identities, restarts both
matched arms from confirmed U1, and corrects the byte-level verifier disagreement recorded in the
[r2 operational-failure result](results/v0.3-action-effect-stage-a-r2-operational-failure.md).
It is not a resume and cannot load any r2 policy or state.

## Why r3 exists

r2 completed the entire sham budget and produced authentic first-rollout, exam, case, terminal
checkpoint, and report evidence. At arm closeout, the manifest recomputed the recorded
first-rollout aggregate with a different canonical JSON primitive:

- qualification and training used compact, sorted, ASCII JSON followed by one line-feed byte;
- the manifest used compact, sorted JSON without that final byte.

The qualified profile reproduces r2's recorded aggregate `fa4c7bda…`; the manifest's generic
profile produces `3b9ecf3a…`. The manifest correctly stopped when its recomputation disagreed, but
the disagreement was a verifier defect. Action-effect never began.

r3 names and shares the qualified profile across qualification, training, and closeout. It retains
r2's dashboard and process-control corrections.

## Immutable r2 predecessor

r3 qualification must authenticate the frozen r2 result before it may create an r3 cohort:

| Evidence | Frozen r2 identity |
|---|---|
| Source commit | `01b1b910edbb676f5de7375fa55d1b6e3bc6a54c` |
| Annotated tag | `action-effect-architecture-v0.3-stage-a-r2-20260724` |
| Tag object | `87d4ce7d24bd13d3a5e3e182c29889c36db35d87` |
| Qualification report SHA-256 | `148bb469cf66753b3ab998299e5f595c94d4fb868c5c38d93c8ec0e9cd097a9e` |
| Cohort-contract SHA-256 | `4114f1d03e5d7e84eab9106676b6dcb9a8b2b0f2b8376a765f599dc61b1b49fa` |
| Final cohort-state SHA-256 | `7cd16a4f3cd886d9036be33dca8dce7e23799b0620a7a7272bcc482b2169bacc` |
| Sham report SHA-256 | `c90a9b4728fcddbec7b602ac67fde9aad784bb26c3979c8d2608f7dfdcd4251c` |
| Sham terminal checkpoint SHA-256 | `a9f06093d07110fc2eb069f911984cf5b6e2746ea10463e783e0c58b4425ca06` |
| First-rollout envelope SHA-256 | `8cf663b50973e608a049be40f0217597af9d20a5f5e8fdd76b61df92cf037f2f` |
| Qualified first-rollout aggregate | `fa4c7bda99a261f8fa49741a49360cd1bfc6ab3081db51aeffc64266a109ce72` |
| Legacy no-LF recomputation | `3b9ecf3ac69c834ddc879d1a542e9f109d833f30aa2324e80c099f7a2195b81c` |
| Empty launcher wrapper log SHA-256 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Detached screen launcher log SHA-256 | `18efff19216bc33b86284b54bfb6be097842866a275a5d772a63049c039534d7` |

Qualification must verify r2's exact immutable inventory, full sham counters, 32 frozen exams,
complete case evidence, terminal report and checkpoint, empty action-effect attempt list, absent
action-effect directory, terminal `integrity_failed` cohort state, and the exact closeout exception.
It must recompute the preserved identity under both byte profiles and prove the documented
one-byte cause.

Attempt 0, r1, and r2 remain independent historical records. r3 may authenticate attempt 0 and r1
through the r2 qualification's exact predecessor bindings, but it must also directly authenticate
r2. Never modify, resume, prune, rename, overwrite, or use as a parent:

- `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r2-20260724`
- `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r2-media-20260724`
- `/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.3-action-effect-stage-a-r2-20260724`

## Fresh r3 identity

| Item | Prospective r3 assignment |
|---|---|
| Cohort ID | `v0.3-action-effect-stage-a-r3-20260724` |
| Annotated source-preregistration tag | `action-effect-architecture-v0.3-stage-a-r3-20260724` |
| Qualification root | `/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.3-action-effect-stage-a-r3-20260724` |
| Cohort root | `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r3-20260724` |
| Media root | `/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r3-media-20260724` |
| Dashboard | `http://127.0.0.1:8791/` |
| Sole launcher | `scripts/run_v03_action_effect_stage_a_r3.sh` |

These are implementation assignments, not release claims. Their exact values remain **TBD in this
prospective document** and must be read from the later external release evidence:

| Release evidence | Prospective value |
|---|---|
| Clean r3 source commit | `TBD` |
| Annotated tag object | `TBD` |
| Canonical tag-payload SHA-256 | `TBD` |
| Qualification claim SHA-256 | `TBD` |
| Qualification report SHA-256 | `TBD` |
| Qualification checksum-file SHA-256 | `TBD` |
| Cohort-contract SHA-256 | `TBD` |

This document alone does not establish an r3 qualification, cohort, media root, dashboard process,
policy action, checkpoint, or result.

## Scientific contract is unchanged

Both arms reconstruct exact confirmed U1 child `20260733` from checkpoint SHA-256
`3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104`, including every compatible
legacy Adam moment by parameter name.

The fixed order remains:

1. `sham`: the identical Dict architecture with all nine action-effect values held at zero;
2. `action-effect`: the same architecture with the preceding self-selected action one-hot and the
   next visible-frame changed/unchanged one-hot.

Both fresh arms retain:

- architecture-initialization seed `20260756`;
- algorithm seed `20260757`;
- worker streams `20260757`–`20260760`;
- four workers and 512 rollout steps per worker;
- four PPO epochs and 2,048 transitions per optimizer phase;
- exactly 1,048,576 fresh child actions apiece;
- exams every 32,768 post-update actions, 32 exams per arm;
- the unchanged U2 control reward, PPO, curriculum, lesson mechanics, horizons, and exam cases;
- the same frozen protected seed partitions and static history guard used by r2;
- the complete U2-S terminal-three capability-and-stability gate; and
- the original selection rule.

Do not add r2 sham layouts to the applied history guard. Do not issue new sampler partitions.
Repeating the frozen r2 seed and guard contract is necessary to keep r3 an operational replacement
rather than a new experiment.

The new residual starts exactly at zero. Before action one, legacy features, logits, values,
deterministic actions, and recurrent states must match the confirmed U1 parent exactly. The
complete first 2,048-transition sham and action-effect trajectories must also match exactly.
Behavioral divergence may begin only after the declared first optimizer phase.

Sham remains calibration only. Only an independently eligible action-effect architecture
definition can authorize a separately committed, three-lineage replication protocol. Stage A
cannot promote a development checkpoint or open U3.

The completed r2 sham scores are visible historical evidence. They may not influence r3's
hyperparameters, curriculum, budget, seeds, guard, tail gate, or selection rule. In particular, the
single 85-action terminal tail is not grounds to relax the fixed stability requirement.

## Named first-rollout digest correction

r3 defines the first-rollout identity profile:

`u2s-canonical-json-v1-lf`

Its bytes are exactly:

```text
UTF-8(
  json.dumps(value, sort_keys=True, separators=(",", ":"),
             ensure_ascii=True, allow_nan=False)
  + "\n"
)
```

This profile applies only to the nested first-rollout identity and every direct copy of that
identity. Existing digest contracts for manifests, reports, checkpoints, sidecars, inventories,
and other artifacts do not change.

Qualification, trainer, and manifest must call one shared, dependency-free implementation. The
terminal manifest must authenticate:

1. the early `first-rollout.json` envelope;
2. `captured_before_first_optimizer: true`;
3. the correct protocol, arm, qualification binding, and no-reuse declaration;
4. the nested aggregate recomputed with `u2s-canonical-json-v1-lf`;
5. exact identity equality across the envelope, live status, report, terminal sidecar, and verified
   report; and
6. the envelope file SHA-256 bound into terminal integrity evidence.

The evidence envelope may receive a new schema/profile field to make the byte contract explicit.
That metadata is operational only. The identity content, observation supplied to the policy, and
all learner behavior must remain unchanged.

## Mandatory implementation and test gates

Before source freeze, r3 must pass:

### Exact regression

- the preserved r2 identity recomputes to `fa4c7bda99a261f8fa49741a49360cd1bfc6ab3081db51aeffc64266a109ce72`
  under `u2s-canonical-json-v1-lf`;
- the legacy no-LF helper recomputes
  `3b9ecf3ac69c834ddc879d1a542e9f109d833f30aa2324e80c099f7a2195b81c`;
- only the first value authenticates this field; and
- JSON capture, write, reread, report, and manifest round trips preserve that value.

### Cross-component integration

- trainer- or qualifier-generated evidence, not a verifier-generated mock, passes manifest
  closeout;
- the complete disposable lifecycle covers cohort creation, sham start, sham finish,
  action-effect handoff, action-effect finish, process closeout, and terminal finalization;
- each identity copy and wrapper field is compared; and
- the test fixture cannot compute its expected value with the function under test.

### Negative authentication

Tampering with any of these must fail closed:

- digest-profile name;
- envelope capture flag, protocol, arm, or qualification binding;
- aggregate or any nested trajectory, policy-output, ledger, or post-rollout RNG component;
- status, report, terminal sidecar, or verified-report copy;
- report-integrity or envelope-file digest; and
- arm order, root mapping, source identity, or predecessor binding.

### Existing operational boundary

All r2 process and launcher tests remain mandatory:

- leading-argument role classification;
- exactly one launcher, supervisor, trainer, and `caffeinate` chain for the active arm;
- duplicate-role rejection;
- explicit, idempotent abnormal-exit cleanup;
- bounded authenticated escalation without signalling unrelated processes;
- no owned process after failure;
- read-only dashboard behavior;
- fixed sham-then-action-effect order;
- fresh-only, non-resumable roots; and
- sealed process closeout before terminal finalization.

### Independent science-diff audit

Before qualification, an audit must prove that r3 changes no observation content, policy
architecture, transplant, reward, PPO configuration, curriculum, action budget, evaluation case,
seed issuer, static guard, or terminal selection rule. Any difference outside evidence
authentication, operational identities, tests, and documentation stops r3 and requires a new
scientific protocol.

## Dashboard, process, and storage boundary

r3 retains r2's deep-auth-once dashboard, leading-argument process classification, and explicit
abnormal-exit cleanup. Port `8791` is new and the dashboard remains read-only.

The fixed storage caps remain:

| Scope | Cap |
|---|---:|
| Per-arm scientific artifacts | 2 GiB |
| Cohort scientific artifacts | 4 GiB |
| Narrative media | 6 GiB |
| Combined planned artifacts | 10 GiB |
| Minimum free space before launch and each arm | 25 GiB |

## Release and launch order

The only authorized sequence is:

1. preserve and document r2 without changing any r2 byte;
2. implement only the named digest correction, evidence plumbing, r3 identities, and required
   tests;
3. pass the complete repository suite, cross-component integration, exact r2 regression, link
   checks, and independent science-diff audit;
4. commit and push one clean r3 source;
5. derive the canonical one-line r3 tag payload from that exact source;
6. create and publish annotated tag
   `action-effect-architecture-v0.3-stage-a-r3-20260724`;
7. run one-shot qualification into the absent r3 qualification root;
8. authenticate the new tag object, qualification report, r2 predecessor, confirmed U1 parent,
   protected partitions, static guard, roots, storage, device, and process boundary;
9. verify the r3 cohort and media roots are absent;
10. invoke `scripts/run_v03_action_effect_stage_a_r3.sh` exactly once;
11. verify sham terminal closeout and authenticated handoff before action-effect records action one;
12. let action-effect finish its exact budget; and
13. stop sleep prevention, seal process closeout, and finalize the matched report.

The chain remains non-circular: the annotated tag preregisters source and contracts; the later
qualification report binds that exact tag object; the cohort contract then binds the report
SHA-256. Never claim that the tag contains a qualification digest created later.

r3 has no resume mode. Any interruption, crash, verifier disagreement, source drift, storage
failure, or process-boundary violation closes the complete root as operationally incomplete. A
replacement would require another prospective identity, source commit, annotated tag, one-shot
qualification, and two entirely fresh twins from confirmed U1.

## Allowed claims

Before r3 completes:

- the r2 learner completed sham and the r2 evidence verifier failed;
- the mismatch is exactly one terminal line-feed byte in the canonical digest profile;
- r3 preserves the scientific question and proposes an operational correction; and
- release, qualification, and run claims require their later external evidence rather than this
  prospective document.

Do not claim that r2 tested action-effect, that its sham terminal checkpoint is reusable, that r3
is released, or that Stage A has selected an architecture.
