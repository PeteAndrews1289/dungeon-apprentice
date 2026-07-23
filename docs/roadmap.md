# Roadmap

## Foundation

- [x] Deterministic procedural generator for Tiers 0–2
- [x] Pixels-only policy observation wrapper
- [x] Full-state solvability oracle kept outside learning
- [x] Seed-partitioned validation and final suites
- [x] Recurrent PPO training and atomic checkpoint publication
- [x] Automatic promotion and retention testing
- [x] Living local dashboard and append-only evaluation history
- [x] Short v0 engineering canary on Tier 0
- [x] Independent reward, scheduling, resume, and hardware audit
- [x] Complete local v0.1 train-save-reload-resume-evaluate qualification
- [x] Pass the published v0.1 training smoke in GitHub Actions
- [x] Establish Navigate learnability across multiple random seeds
- [x] Implement and qualify the transition-balanced Visible Unlock development slice
- [x] Replicate Visible Unlock mastery with retained Navigate across three random initializations
- [x] Confirm all three U0 mastery checkpoints on the preregistered disjoint 200-case suites
- [x] Implement, qualify, test, and preregister the U1 Local Unlock warm-start child
- [ ] Run the U1 lead child to its mastery gate or 524,288-action ceiling
- [ ] Implement and qualify the proposed v0.2 Unlock staircase
- [ ] Demonstrate full Unlock with retained Navigate across multiple random seeds

The first end-to-end 64-step smoke run completed on July 22, 2026. It exercised initialization,
experience collection, checkpoint publication, held-out evaluation, status history, frame capture,
model reload, and clean shutdown. It is an engineering test, not a learning result. A later audit
invalidated all v0 runs as capability evidence because curiosity could reward failure and scheduled
exams measured pre-update policies. Protocol v0.1 fixes both before any long experiment.

The local v0.1 qualification also covers strict digest/config resume, distinct child RNG streams,
discounted reward dominance, terminal timeout semantics, interrupted-child recovery, bounded storage,
and milestone-rich evaluations. Three subsequent fresh canaries established Navigate learnability
and exposed the next two gates: transition-balanced retention and full Unlock discovery.

The first v0.2 slice has now passed its replicated gate: all three fresh policies mastered the
complete Visible Unlock sequence while retaining Navigate, with measured post-promotion experience
held at approximately 50/50 by transitions. The frozen
[replication report](results/v0.2-visible-unlock-replication.md) records the evidence and its narrow
claim. A [post-training confirmation](v0.2-u0-confirmation-plan.md) was preregistered before any new
case was inspected. All three checkpoints passed every overall and panel gate; the frozen
[confirmation report](results/v0.2-u0-confirmation.md) now supports treating U1 as the next
capability step.

U1 is now an explicit child protocol rather than an in-place modification of the confirmed
sentinel. Its generator makes the key visible and door hidden, exact U0 pixels are protected by
golden regression, all three lessons are graded at every boundary, and separate recovery mixes own
Navigate and U0 forgetting. The complete frozen design and decision rule are in
[the U1 development protocol](protocol-v0.2-u1-development.md).

The lead and both preregistered U1 replications mastered Local Unlock. The three first-mastery U1
scores were all 72/80, while final Navigate ranged from 74/80 to 77/80 and every policy finished U0
at 80/80. One lineage used Navigate recovery, one used combined and U0-only recovery, and one needed
no recovery; every completed window remained within allocation tolerance. This passes the frozen
three-lineage replication rule. Exact evidence is in the
[Local Unlock replication result](results/v0.2-local-unlock-replication.md).

The next gate was the preregistered [U1 post-training confirmation](v0.2-u1-confirmation-plan.md):
three frozen policies, three 200-case lesson blocks, no updates, and a strict all-three verdict.
Attempt 1 stopped before policy scoring because nine numerically new seeds generated exact layouts
from declared development references. That immutable qualification failure is documented in
[the attempt-1 result](results/v0.2-u1-confirmation-attempt-1.md).

The current gate is a separately preregistered replacement instrument rather than a retroactive
repair.
[Confirmation v2](v0.2-u1-confirmation-v2-plan.md) reserves fresh candidate streams and selects the
first 200 mechanically valid, exact-unique, reference-excluded layouts without loading a policy.
Only a positive v2 confirmation supports implementing U2 Separated Unlock.

## Scalability gates

The project scales in two different senses, and both require evidence:

1. **Experimental scale:** bounded storage, disk guards, atomic digest-linked artifact files, strict
   resume validation, unattended shutdown, and reproducible multiple-seed comparisons.
2. **Mechanical scale:** declarative mechanics/tasks, a capability graph, isolated and composed
   suites, and objective disambiguation before adding multiple possible quests.

Do not add gameplay breadth merely because one overnight process stays alive. The three v0.1
canaries established repeatable Navigate learning but exposed both a full-Unlock discovery cliff and
post-promotion forgetting. The [v0.2 design](protocol-v0.2-design.md) therefore adds a
declarative lesson layer, progressively harder complete Unlock quests, and transition-balanced
retention recovery before Retrieve is attempted. Only after full Unlock is learned and Navigate is
retained should the project add the third capability or a new mechanic.

## Capability releases

| Release | Mechanics | Final question |
| --- | --- | --- |
| V0 | Navigation, one key/door, one relic, return | Can one policy learn and retain the basic quest? |
| V1 | Multiple colors, decoy keys, inventory choice | Does it learn matching rather than “pick up anything”? |
| V2 | Levers, remote state changes, one-way passages | Can it remember consequences outside its view? |
| V3 | Traps, health, limited-use tools | Can it trade risk against progress? |
| V4 | Moving enemies and safe zones | Can it plan under a changing world? |
| V5 | Multiple floors and composed objectives | Can learned skills become a complete unseen adventure? |

Every release retains all previous frozen suites. New mechanics are not added to an active declared
run, and a successor is compared with both its immediate predecessor and a fresh-start control.
