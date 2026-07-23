# Protocol v0.2 U1: the Local Unlock warm-start child

> Status: positive development result. This is an accumulating-skill child of the confirmed U0
> policy, not a fresh-start replication and not evidence for unrestricted Unlock. Two additional
> parent lineages are frozen in the [separate replication plan](v0.2-u1-replication-plan.md).

## Question

Can the confirmed seed-`20260725` policy add a longer, partially observed key–door–exit behavior
while retaining both capabilities it already demonstrated: Navigate and Visible Unlock?

The experiment is intentionally cumulative. It asks whether an existing recurrent policy can grow
its repertoire from experience. It does **not** ask whether U1 can be learned from random parameters,
and its result may not be pooled with the three fresh U0 replications.

## Why this is a new child protocol

The original v0.2 proposal described fresh runs eventually climbing U0 through U3. U0 has now been
replicated and confirmed, while its implementation deliberately stops at U0 and has no compatible
resume path into a third lesson. Extending that frozen sentinel in place would blur which source
produced the confirmed result.

U1 therefore uses protocol ID `dungeon-apprentice-v0.2-u1` and checkpoint schema 3. The confirmed U0
code and archives remain unchanged. The child inherits the full PPO archive—including the optimizer,
not merely policy weights—then binds it to a new three-lesson environment. No trajectory, action,
demonstration, recurrent episode state, or oracle path is transferred.

## Frozen parentage

The lead child may start only from this exact artifact:

- training seed: `20260725`;
- checkpoint: `mastered-visible-unlock.zip`;
- checkpoint SHA-256:
  `2b235ed54746429e737af2a6037069821fdaf81a936edf8f74ce86cc766b40a2`;
- fully trained boundary: 491,520 actions and 960 PPO updates;
- U0 confirmation report SHA-256:
  `f43610252943fce9c0169ac0231724fc2829686d1899f5f88bd0c250a913b398`;
- confirmation verdict: passed with no policy updates.

The launcher re-verifies the archive, sidecar, clean parent source, manifest, allocation history,
mastery state, confirmation digest, and that policy's individual confirmation verdict before it
creates a child run directory. A mismatch stops the run.

## Information and reward boundary

The policy still receives only its `56 × 56 × 3` egocentric RGB image and private recurrent state.
It is not given lesson ID, coordinates, facing numbers, map, route, object labels, seed, objective
text, oracle actions, or an online model call. The same seven primitive MiniGrid actions remain.

All lessons retain success reward `+1.0`, ordinary step reward `-0.001`, and the bounded
pixel-novelty contract used by the U0 parent. Key acquisition and door opening are diagnostics and
pay no authored reward. Local Unlock succeeds only after the complete key → locked door → exit
sequence. Evaluation is deterministic, disables curiosity, resets recurrent state per case, and
never updates weights.

## Declarative lesson registry

The child replaces hard-coded two-state curriculum assumptions with immutable lesson specifications.
Each specification owns its stable ID, prerequisites, generator profile, horizon, validation block,
oracle constraint, and gate. The policy cannot see the registry.

| Lesson | Role in U1 | Generator | Frozen exam seeds |
| --- | --- | --- | --- |
| `navigate/full` | retained anchor | exact v0.1 Navigate pixels | `10_000_000`–`10_000_079` |
| `unlock/u0-visible` | retained anchor | pixel-exact confirmed U0 profile | `11_000_000`–`11_000_079` |
| `unlock/u1-local` | active lesson | Local Unlock v1 | `11_100_000`–`11_100_079` |

A golden regression compares the child U0 profile with the confirmed sentinel across fixed seeds.
The grid encodings, starts, directions, objects, colors, and walls must match exactly.

## Exact meaning of Local Unlock

Every U1 case is a 9 × 9 map with a 7 × 7 interior, empty starting inventory, one matching colored
key, one locked door in a complete horizontal or vertical divider, a goal beyond that divider, and
no additional walls. Rotation, reflection, approach side, divider position, door lane, key color,
start, orientation, and goal vary by seed.

Candidate geometry is rejected unless all of these hold:

- the key is visible in the first policy image;
- the door is **not** visible in the first policy image;
- key and door are non-collinear and at least four Manhattan cells apart;
- the post-key path requires a turn;
- the goal can be reached only by crossing the locked divider;
- a pure full-state planner and the live qualification oracle agree on a complete 9–18-action
  solution;
- horizon remains 128, preserving negative timeout return and discounted reward dominance.

The generator fails after a bounded search instead of relaxing a condition. Its pure planner exists
only to accept levels; the resulting actions are never stored as learning data.

## Generator acceptance

Before every declared run, the launcher qualifies seeds `5_100_000`–`5_100_999`. It requires
1,000/1,000 live-oracle solutions, correct visibility, action length, reward, horizon, pixel shape,
and lesson evidence. At least 99% of color-inclusive visual layouts and 95% of geometry-only hashes
must be unique. This distinction prevents random key color from disguising weak geometric variety.

The prelaunch engineering qualification produced:

| Check | Result |
| --- | ---: |
| Oracle solved | 1,000/1,000 |
| Exact visual layouts | 998/1,000 (99.8%) |
| Geometry-only layouts | 991/1,000 (99.1%) |
| Oracle actions | 9–18 |
| Key visible / door hidden | 1,000/1,000 |
| Exact visual overlap with 80 U1 validation cases | 0 |
| Geometry-only overlap diagnostic | 2 |

The 99% full-layout floor is deliberate. Exact 100% uniqueness would turn two repeated samples in a
constrained finite generator into an engineering failure even though geometry diversity remains far
above its declared floor. Counts and both hash types remain public. Training actively rejects any
exact visual hash reserved by qualification or any of the three validation suites. Geometry-only
overlap remains a disclosed diagnostic; no seed partition overlaps.

The future U1 confirmation block begins at `15_020_000`. It remains untouched during development.
The project-wide 20-million final allocations also remain untouched.

## Practice controller

Lesson selection targets actual policy actions, not episodes. Ordinary U1 practice is:

| Source | Target |
| --- | ---: |
| Navigate | 50% |
| Visible Unlock U0 | 15% |
| Local Unlock U1 | 35% |

Every 32,768-action window must finish within five percentage points of all targets. Assignment is
made only between complete episodes. The scheduler persists cumulative/window counts and its random
state in every schema-3 sidecar.

The earlier proposal protected Navigate but did not explicitly prevent U0 forgetting. This child
closes that loophole. All three lessons are examined at every boundary. If a prerequisite weakens,
advancement and the active-pass counter freeze and the next window uses:

| Weak prerequisite | Navigate | U0 | U1 |
| --- | ---: | ---: | ---: |
| Navigate only | 75% | 10% | 15% |
| U0 only | 50% | 35% | 15% |
| Both | 65% | 25% | 10% |

Recovery ends only after two consecutive exams in which both anchors pass and practice allocation is
valid. Those recovery exams cannot count toward U1 mastery. Normal 50/15/35 practice then resumes.

## Frozen exams and mastery

An initial N/U0/U1 baseline exam is recorded before the first child update. It measures immediate
transfer and can never count toward a gate. Afterward, exams run every 32,768 newly trained child
actions. Each lesson has two fixed 40-case panels.

| Lesson | Overall gate | Each-panel gate |
| --- | ---: | ---: |
| Navigate | 68/80 (85%) | 34/40 (85%) |
| Visible Unlock U0 | 68/80 (85%) | 32/40 (80%) |
| Local Unlock U1 | 68/80 (85%) | 32/40 (80%) |

U1 mastery requires two consecutive post-update boundaries where all three lessons pass and the
normal transition window is valid. One favorable panel, training success, milestone rate, or
dashboard observation cannot promote the policy.

## Run budget and stopping rule

The lead child receives at most 524,288 **new** actions—approximately 183,501 target U1 actions—on
four CPU workers. It retains the parent's recurrent PPO settings: 512-step rollouts, batch 256, four
epochs, learning rate 0.00025, gamma 0.995, GAE lambda 0.98, entropy coefficient 0.01, and the same
256-unit LSTM. Early stop is permitted only after the full two-exam cumulative mastery gate. A flat
curve runs to the ceiling.

The exact launch command is `scripts/run_v02_u1_lead.sh`. It writes to the T7, enforces a 25 GiB free
space reserve, serves the dashboard on port 8784, and refuses a dirty source tree. The dashboard
labels the run as a warm-start child, separates inherited lifetime actions from new U1 actions,
shows all three frozen exams and panels, reports target versus actual practice, names the recovery
cause, and keeps separate exam frames.

## Resume and interruption

Every checkpoint contains full policy and optimizer state plus a matching schema-3 sidecar with the
exact parent/confirmation lineage, curriculum state, recovery owner, practice counts, scheduler RNG,
child and lifetime actions, optimizer updates, source configuration, and segment seed. Resume creates
a new run segment, verifies all fields and the archive digest, restores the scheduler, and uses a new
declared random stream. Partially collected rollouts are not called trained experience.

## Outcomes

- **Positive U1 development:** cumulative two-exam mastery before the ceiling.
- **Transfer signal:** U1 rises materially above its pre-update baseline but does not master.
- **Interaction/transfer failure:** U1 remains below 25% after 262,144 child actions.
- **U0 retention failure:** U0 remains below gate for two recovery exams.
- **Navigate retention failure:** Navigate remains below gate for two recovery exams.
- **Allocation failure:** a completed window exceeds the five-point tolerance.
- **Engineering failure:** parentage, qualification, checkpoint, evaluation, resume, or evidence
  invariants fail.

Milestones explain an outcome but never substitute for complete exit success. If the lead child is
positive, replications from the other two confirmed U0 parents require a separate frozen plan. If it
is negative, the next design may change curriculum geometry or declare bounded potential shaping,
but not relabel this run after seeing its curve.

## Lead result

The lead was positive. It ran from 20:04:13 to 20:55:39 EDT on July 22, 2026 and mastered after
393,216 new trained actions. Its diagnostic baseline was Navigate 74/80, U0 79/80, and U1 0/80.
U1 first passed at 360,448 child actions with 73/80 and confirmed at 393,216 with 72/80; final
Navigate was 74/80 and final U0 was 80/80. Both U1 panels passed at 39/40 and 33/40.

Navigate briefly missed one panel floor at 131,072 actions. The declared recovery controller shifted
practice toward Navigate, required two clean anchor exams, and returned to normal U1 practice at
196,608 actions. The final completed practice window was 49.8871% Navigate, 15.1093% U0, and
35.0037% U1. Whole-child shares were 53.9747%, 14.2799%, and 31.7454% because the two recovery
windows deliberately increased Navigate rehearsal. The mastery archive SHA-256 is
`bcce9b8251e97ed4fddda32871c891c3783c057bbb1f89deedb3a3d32058102a`.

## Narrative value

U0 established that the agent could learn a tiny complete ritual. U1 asks the more interesting
question: can that ritual become a reusable concept when one required object disappears from view?
The visual story is now three simultaneous curves rather than one—new skill rising, first skill
retained, and the immediately preceding skill retained—plus an automatic rehearsal controller that
responds when memory erodes.
