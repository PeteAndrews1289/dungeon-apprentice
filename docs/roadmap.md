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
- [x] Run the U1 lead and both preregistered replications to Local Unlock mastery
- [x] Confirm all three U1 mastery checkpoints with the collision-safe disjoint instrument
- [x] Implement, qualify, and run all three U2 Separated Unlock children to mastery
- [x] Run the preregistered collision-aware U2 confirmation and preserve its strict failed verdict
- [ ] Establish a three-policy U2 activation set; original attempt 1 remains a strict 2/3 failure
- [x] Freeze a non-post-hoc U2 successor decision before any new training or protected evaluation
- [x] Implement, qualify, and externally anchor the one-lineage U2r Stability Remediation continuation
- [ ] Run its fixed eleven-window budget and grade only the full-budget terminal artifact
- [ ] If terminal-eligible, run its one fresh no-update confirmation on `15_240_000`–`15_279_999`
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

The separately preregistered
[confirmation v2](v0.2-u1-confirmation-v2-plan.md) then passed. Navigate accepted 200/200 examined
candidates; U0 accepted 200 after examining 236 and rejecting 36; U1 accepted 200 after examining
207 and rejecting seven. Every accepted exam had 200 unique exact layouts and no accepted reference
collision. The three policies scored Navigate/U0/U1 totals of 188/200, 200/200, 188/200;
192/200, 200/200, 192/200; and 188/200, 200/200, 182/200. All overall and panel gates passed with
no policy or optimizer update. The raw report SHA-256 is
`6e577170050f6f14599b793a031776a19bf7c64eba0f243f457298da3193ae8f`; see the
[confirmation v2 result](results/v0.2-u1-confirmation-v2.md).

That positive result closes U1 and opens the next bounded question:
[U2 Separated Unlock](protocol-v0.2-u2-separated-unlock.md). U2 must make key-to-door search less
local while preserving cumulative Navigate, U0, and U1 retention; it does not skip directly to
Retrieve or unrestricted game complexity.

The U2 cohort is now a replicated positive development result. All three independently inherited
children mastered Separated Unlock after 688,128, 557,056, and 688,128 new actions. Their U2 scores
rose from 32/80, 33/80, and 24/80 at inheritance to 75/80, 74/80, and 72/80, while every terminal
Navigate, Visible Unlock, and Local Unlock score remained above its frozen gate. The
[immutable U2 result](results/v0.2-u2-separated-unlock-replication.md) preserves the evidence.

Because the 80-case development suites influenced curriculum recovery and stopping, the next gate
was not U3 training. The three first-mastery archives first faced the separately committed
[collision-aware U2 confirmation](v0.2-u2-confirmation-plan.md) with no updates. It excluded every
persisted reference and completed logged training layout, while explicitly retaining the bounded
12-layout terminal logging limitation discovered during closeout.

That confirmation is complete, and its strict all-three gate did **not** pass. Children `20260737`
and `20260741` passed all four lessons. Child `20260745` passed Navigate, U0, U1, and both U2
panels, but its U2 total was 169/200 against a frozen 170/200 requirement. All 24 panel gates passed;
11 of 12 overall lesson gates passed. The immutable verdict remains `capability_failed`, and the
[confirmation result](results/v0.2-u2-confirmation.md) records all selection, collision,
no-update, integrity, and diagnostic evidence.

U3 is therefore blocked. The opened 15.2-million confirmation cases cannot become a fresh test,
and the one-case margin cannot be rounded, retuned, or retried away after observation.

The prospective decision is now frozen as
[U2r Stability Remediation](protocol-v0.2-u2r-stability-remediation.md). Only failed child
`20260745` may continue, from its exact selected archive and complete inherited optimizer. It gets
exactly the 360,448 actions left under its original 1,048,576-action U2 ceiling, using the same
pixels, PPO, reward, curriculum, and 50/7.5/7.5/35 transition mix. All eleven windows must run, and
only the full-budget terminal artifact can qualify. Its last two exams must pass the original gates
plus prospectively stronger interaction-stability criteria.

If that terminal artifact qualifies, it receives one no-update confirmation on four fresh candidate
streams at `15_240_000`–`15_279_999`. A pass would open U3 only with the explicit ancestry label
**two directly confirmed U2 policies plus one prospectively remediated U2r policy**. It would not
rewrite the original U2 confirmation as 3/3. A valid U2r stability or confirmation failure leaves
U3 closed and cannot trigger another unplanned retry.

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
