# Video narrative notebook

## Working premise

**I tried to teach an AI Pokémon. It taught me that the world was the problem. So I built it a world
where learning could be measured one idea at a time.**

The emotional spine is not guaranteed success. It is the attempt to separate genuine learned skill
from random progress, hidden assistance, and wishful interpretation.

## Possible title directions

- I Built a Dungeon to Teach an AI How to Learn
- Can an AI Learn the Idea of an Adventure?
- Pokémon Was Too Big—So I Built an AI Its Own Game
- From Random Buttons to a Dungeon Master

## Act structure

### Act I — The honest failure

Open on the seductive promise of “AI plays Pokémon,” then show the actual problem: an enormous game,
rare rewards, repeated local traps, and human intervention turning one game into many small lessons.
The important failure was experimental, not personal. We could not tell whether the agent was
building a general understanding or merely surviving the latest prod.

### Act II — Build a fair test

Introduce Dungeon Apprentice and the rules:

- pixels and seven buttons only;
- no GPT calls during play;
- no maps, coordinates, walkthroughs, demonstrations, or savestates;
- every dungeon is procedural and independently proved solvable;
- success must repeat on unseen levels;
- every new lesson includes an exam on the old ones.

Visually reveal the tiers as a staircase: exit, key and door, relic and return. Explain that the same
memory-bearing policy must climb the entire staircase.

### Act III — Auditing the scoreboard

Use the first canary as a lesson in why the experiment itself must be tested. It succeeded in 5 of 32
noisy training attempts and its two scheduled validation exams scored zero—but the deeper audit found
two flaws. Failed wandering could earn more shaped return than success, and the 4,096-step exam
measured a 70-update checkpoint while the terminal model reached 80 updates and was never evaluated.

This is a stronger story beat than a conveniently clean first result: even a green dashboard can
answer the wrong question. Show the misleading reward comparison, the collected-versus-trained
timeline, and the decision to invalidate v0 rather than polish it into evidence. Then introduce v0.1:
bounded curiosity, post-update exams, and checkpoint sidecars.

### Act IV — The first real skill

The three v0.1 curves supply the first payoff. Begin with 0–15% success, then animate all three
independent lines rising until every policy crosses the Navigate gate. This is the moment the project
can finally say “it learned” without relying on one seed or a training reward.

Then let the staircase collapse. All three policies finish at exactly 75% Navigate retention and 0%
deterministic Unlock. A funnel reveals 978 stochastic Unlock attempts: 532 keys, 23 opened doors, 22
completions. The apparent 30% Navigate rehearsal share is unmasked as only 4–6% of actual actions
because short successes and long timeouts were counted as equal episodes.

This gives the failure a concrete cause and a memorable visual: a pie chart labeled “episodes” next
to the radically different pie chart labeled “experience.”

### Act V — Build a better staircase

Introduce v0.2 as a controlled answer, not another emergency tweak. Keep the same model, pixels,
actions, rewards, and three algorithm seeds. Replace the single Unlock cliff with progressively
harder complete key-door-exit dungeons, allocate rehearsal by measured actions, and require two
consecutive balanced exams before advancement.

The open dramatic question is now precise: can abundant complete early experience turn a rare
interaction into a deterministic skill without erasing navigation? Full success, transfer failure,
retention failure, and evidence that sparse reward is insufficient are all honest endings.

### Act VI — Can a skill become an idea?

U0 supplies the second real payoff: three agents independently learn the entire visible key → door →
exit ritual, then all three pass a larger disjoint confirmation. Do not frame this as “the dungeon is
solved.” Put the tiny straight-line U0 map beside U1, where the key is visible but the door begins
outside the image and the route must turn.

The U1 visual should carry three bars at every exam: Navigate, Visible Unlock, and Local Unlock. This
makes cumulative learning legible. When an old bar drops, show the practice mix changing by itself;
when it recovers, show ordinary U1 practice returning. The dramatic question is no longer only
“does the new line rise?” It is “can the brain add a behavior without erasing the behaviors beneath
it?” A pre-update 0/80 U1 baseline also makes later transfer measurable without pretending inherited
U0 experience was a fresh start.

U1 now supplies that payoff. The lead and both preregistered children mastered, and all three ended
at 72/80 on the shared frozen U1 exam while preserving U0 and Navigate. Their paths were visibly
different: one needed Navigate recovery, one needed combined and U0-only recovery, and one climbed
without recovery and finished earlier. That is better narrative evidence than three identical clean
curves. The controller was not cosmetic; it protected old skills under three different learning
histories. Follow this with the larger disjoint confirmation rather than treating 72/80 as the end
of the claim.

The first confirmation attempt creates a useful scientific reversal. Its seed ranges were clean,
all 600 dungeons were solvable, and every diversity floor passed—yet nine supposedly new cases were
exact generated-layout repeats from development evidence. The evaluator stopped before seeing a
single policy action. Visually contrast “different seed numbers” with “the same dungeon,” then show
the empty checkpoint-results array. This makes the collision-safe successor feel earned: the test
instrument learned from failure without giving the models a second chance at a score they never
received.

Then deliver the payoff. The successor does not waive the failed instrument or hand-pick easy
cases. It walks new candidate streams in order, accepting only mechanically valid,
reference-excluded exact layouts before any policy loads. Turn this into three simple funnels:
Navigate accepts 200 of 200 examined; Visible Unlock accepts 200 after examining 236 and rejecting
36 repeats; Local Unlock accepts 200 after examining 207 and rejecting seven prior layouts. The
accepted exams contain 200 unique exact layouts apiece.

Now reveal the three policy rows:

| U1 child seed | Navigate | Visible Unlock | Local Unlock |
| ---: | ---: | ---: | ---: |
| `20260725` | 188/200 | 200/200 | 188/200 |
| `20260729` | 192/200 | 200/200 | 192/200 |
| `20260733` | 188/200 | 200/200 | 182/200 |

Every overall bar and both panels pass for every policy. Most importantly, the evaluator performs
no learning: policy tensors, optimizer state, archive bytes, and counters are identical afterward.
The models are not being coached through the test; the test is measuring skill they already
acquired. Preserve the raw report identity on screen:
`6e577170050f6f14599b793a031776a19bf7c64eba0f243f457298da3193ae8f`.
The detailed evidence belongs in
[the confirmation v2 result](results/v0.2-u1-confirmation-v2.md).

This gives the chapter a satisfying but honest ending: the agent has learned and retained a
small quest concept across progressively harder unseen layouts. It has not beaten the whole
dungeon. The staircase can now grow one controlled step into
[U2 Separated Unlock](protocol-v0.2-u2-separated-unlock.md), where key and door search become less
local without skipping straight to Retrieve.

### Act VII — Can a ritual become a plan?

U2 should open with one clean visual transformation rather than another wall of settings. Show U0:
the key and door are both in the little window. Slide to U1: the key remains visible, but the door is
somewhere beyond the initial view. Then reveal U2: a divider cuts the dungeon in two, its locked door
is the only crossing, and exactly two extra walls bend the route. The key and door are no longer
guaranteed visible. The oracle-qualified journey has grown to 17–26 actions, but the agent still
gets the same pixels, the same seven buttons, the same complete-quest reward, and the same recurrent
brain architecture.

The central narrative beat is: **we are not teaching a new trick; we are asking whether a learned
ritual can stretch into a plan.** Three confirmed U1 policies each become their own U2 child. They do
not merge memories, copy trajectories, watch the oracle, or query GPT. Put three apprentice cards on
screen and run them one at a time. The shared dashboard keeps the completed and waiting lineages
visible while the active card shows four exam bars, both panels, actions remaining, storage, and the
current practice mix.

Make recovery emotionally legible. When Navigate, Visible Unlock, or Local Unlock falls, freeze the
U2 mastery streak and animate practice flowing back toward the weakened foundation. When both
recovery exams pass, let the normal mix return. The drama is not simply whether the purple U2 line
rises. It is whether four lines can remain standing together while the newest one rises.

The evidence labels should become recurring on-screen chapter cards:

1. **Engineering proof:** the generator and live oracle agree on all 2,000 sealed qualification
   maps, tests preserve the old lessons, and interruption/storage guards work. This proves the
   measuring instrument, not learning.
2. **Learning evidence:** each child changes its frozen development-exam performance through its own
   experience. Training reward and one lucky episode remain supporting footage.
3. **Confirmation evidence:** after three valid children finish, a separately preregistered
   collision-aware evaluator tests frozen selected checkpoints with no updates. This is the only
   stage that can confirm disjoint generalization.

At the source-freeze boundary, show the unopened envelope honestly: blank U2 curves, three parent
digests, the sealed qualification range, the fixed external anchor tag, and dashboard port 8785.
Then let the immutable qualification and cohort ledgers reveal what happened after launch. This
keeps the suspense without rewriting preregistration in success tense. If the cohort fails, preserve
that ending and explain whether it was a learning result or an engineering stop.

The envelope opened cleanly. The three inherited policies began U2 at 40.0%, 41.25%, and 30.0%.
After 557,056–688,128 new actions, all three reached the complete two-exam mastery rule and finished
at 93.75%, 92.5%, and 90.0% while retaining the three earlier skills. Animate the average U2 line as
three fixed chapter cards—37.1% at inheritance, 47.1% after the first 32,768 actions, and 92.1% at
mastery—rather than pooling their episodes into one imaginary policy.

The dramatic qualifier belongs immediately after the rise: these are frozen development exams that
steered recovery and stopping. The next scene should be a second unopened envelope: the separately
preregistered collision-aware confirmation. No new case may be shown and no result may be implied
until its selection rules and checkpoint hashes are committed.

### Act VIII — One completion short

The confirmation gives the story a better ending than a frictionless pass would have. Begin with
the instrument, not the score. Show the console shortcut that never entered Python, then the two
fail-closed launch defects: the historical tag adapter reporting the evaluator commit and Apple
Git's absent-ref status without the quiet form. Put “zero protected candidates opened” beside both
stops. The important beat is that the safeguards were inconvenient and therefore real.

Then animate the selector. Four streams feed four 200-case exams: 200/200 accepted for Navigate,
200/215 for U0, 200/204 for U1, and 200/200 for U2. Keep the 19 rejected layouts visible—four U0
validation collisions, 11 repeated U0 acceptances, and four U1 training-history collisions. This is
the payoff to the earlier U1 seed-versus-layout lesson: numerical novelty is no longer mistaken for
layout novelty.

Reveal each policy as its own card. Seeds `20260737` and `20260741` pass all four skills. Seed
`20260745` keeps Navigate, U0, and U1, and even passes both U2 panels at 84 and 85. Then let the
combined counter stop at **169/200** while the frozen line remains at **170**. Do not round it, pool
it, or call it “basically confirmed.” Put the protocol verdict on screen:
`capability_failed — U3 blocked`.

The emotional point is not that the model collapsed. It was one completion short, and a
Binomial(200, 0.85) model would produce 169 or fewer successes 45.1489% of the time. The scientific
point is that preregistration matters most when the result hurts. If the line moves now, none of the
earlier clean wins mean as much.

Use the milestone funnel to turn the miss into the next question without pretending it answers it.
The failed policy reached 188 keys, opened 172 doors, and finished 169 quests. Completion after an
opened door was 98.26%; the larger leak was between key and door, accompanied by many more
ineffective interactions than the passing peers. End the chapter on two adjacent truths:

- three lineages learned the development lesson;
- only two lineages confirmed it on the larger collision-aware exam.

The next act is not automatically U3. It is a prospectively frozen decision about how to strengthen
U2 without training on the already opened confirmation cases.

### Act IX — Eleven windows, no moving line

Open on four tempting shortcuts and cross them out one at a time: round 169 to 170, test the same
model again, continue until any checkpoint looks good, or take only the two strongest policies into
U3. Then reveal the bounded successor:
[U2r Stability Remediation](protocol-v0.2-u2r-stability-remediation.md).

The framing matters. The original exam stays red. U2r does not “fix” its historical score. It asks
a new question of a new artifact: can the exact failed lineage become more stable using only the
part of its original action budget it never spent?

Turn the budget into the scene's visual clock:

- the parent remains the exact `20260745` mastery archive and optimizer;
- 688,128 of 1,048,576 child actions were already spent;
- 360,448 remain;
- that remainder is exactly eleven 32,768-action windows; and
- every window must run, even if window three looks perfect or window eight looks hopeless.

Show eleven empty blocks filling from left to right. There is no “best model” cursor following the
curve. Only the eleventh block connects to the next gate.

The new stability gate should be explained as a response to the behavioral diagnosis, not an
invisible rule change. Rewind child `20260745` from its original first pass to mastery:

| Lesson | First pass | Mastery | Mean ineffective interactions |
| --- | ---: | ---: | ---: |
| Visible U0 | 80/80 | 76/80 | 0.00 → 6.55 |
| Local U1 | 80/80 | 77/80 | 0.04 → 4.74 |
| Separated U2 | 76/80 | 72/80 | 2.49 → 6.54 |

The second checkpoint still cleared the old success gates, but it had become much more likely to
repeat pickup or toggle actions. That makes the successor's title literal: it is looking for
deterministic stability, not merely another score over a line.

At both final U2r exams, the original capability gates remain. On top of them, U2 must score at
least 72/80 with both panels at least 34/40, U0/U1/U2 must each average no more than three
ineffective interactions per episode, and no lesson may fall more than two successes between the
penultimate and terminal exam. Recovery and an invalid practice allocation disqualify the pair.
These are openly post-result diagnostics turned into **prospective rules for the successor**. They
never regrade original U2.

If the terminal model is eligible, reveal a third envelope. Its four streams begin at 15.24 million,
after the permanently consumed 15.20–15.239-million attempt:

| Fresh lesson exam | Candidate stream |
| --- | --- |
| Separated U2 | `15_240_000`–`15_249_999` |
| Navigate | `15_250_000`–`15_259_999` |
| Visible U0 | `15_260_000`–`15_269_999` |
| Local U1 | `15_270_000`–`15_279_999` |

The terminal policy gets one no-update measurement on 200 accepted cases per skill. The frozen
170/200 overall line and panel floors do not move. The two directly passing policies are not tested
again; their original evidence remains closed.

End the act with the only honest U3 equation:

> U3 parent set = two directly confirmed U2 policies + one prospectively remediated and freshly
> confirmed U2r policy.

Even on success, never replace that sentence with “U2 passed 3/3.” On failure, do not make the
chapter a tragedy. The result will distinguish three different limits: unchanged experience could
not stabilize the terminal policy; development stability did not generalize; or one bounded
continuation did create a usable third parent. Each is a real answer.

### Act X — Four ways to miss

U2r's terminal failure should not cut directly to another tweak. First show what it taught us:
capability was still present, but two rare interaction loops could dominate an otherwise strong
exam. Then introduce U2-S as the moment the project stops rescuing a policy and tests the learning
mechanism itself.

Use the matched 2 × 2 grid in
[the U2-S result visual](assets/v0.2-u2s-stability-ablation.svg). Four fresh children begin from the
same confirmed U1 brain, optimizer, and random state. Across the top, the project either leaves
reward alone or adds a tiny pixels-only signal after repeated visibly ineffective interactions.
Down the side, it either keeps the existing PPO updates or makes them gentler. Every cell receives
the full 1,048,576-action budget. There is no race, early winner, or “best checkpoint” cursor.

Start the animation with the shared inherited U2 score—24/80—and reveal the first 32,768-action
exam: 21 for control, 31 for conservative, 33 for no-effect, and 36 for combined. That ordering is
tempting, then deliberately leave it unresolved. Early progress is not the decision.

At the fixed final three exams, split the screen into capability and reliability:

| Arm | Final-three U2 | 10+ cases | Worst loop | What it learned |
| --- | --- | ---: | ---: | --- |
| Control | 78, 76, 77 | 6 | 156 | The quest, but not when to stop |
| Conservative | 64, 64, 64 | 5 | 155 | Too cautiously to reach the gate |
| No-effect | 78, 79, 76 | 2 | 126 | Better restraint, still brittle |
| Combined | 70, 69, 71 | 0 | 2 | Restraint, but not enough quest |

This is the visual thesis of the act: one axis preserves capability, the other improves stability,
but no square contains both. Control and no-effect clear every U2 score and panel gate; their rare
tails disqualify them. Combined removes every 10+ terminal case, yet misses 72/80 in all three
exams. Conservative does neither.

Let the verdict land without softening it:

> `ablation_failed` — no configuration selected — no checkpoint advances — U3 closed

The negative result is the pivot, not an anticlimax. A penalty can make a behavior expensive
without giving the network a durable concept of **“I just tried this and nothing changed.”** The
next chapter changes the learner's internal vocabulary: it may observe only pixels plus its own
previous action and the visible consequence of that action, learned through experience. That keeps
the no-walkthrough, no-oracle, no-GPT premise while moving the experiment from reward engineering
to architecture.

Do not describe no-effect as the winning arm because its mean was attractive, and do not describe
combined as a stability success that earned continuation. The terminal rule required the same cell
to satisfy both. None did.

### Act XI — Give the apprentice a sense of consequence

The next chapter should not feel like a fifth rescue attempt. The four-cell ablation already
answered that question: changing how hard the learner is pushed can trade capability for stability,
but it did not give one policy both. The architectural pivot asks something more human:

> What if the apprentice could remember not only what it sees, but what it just tried and whether
> the visible world responded?

Show the v0.3 architecture visual as two identical brains growing the same small input. The sham
twin's nine values are always zero. The action-effect twin sees a one-hot copy of its own previous
button plus changed/unchanged visible pixels. Keep the exclusions on screen: no coordinates, object
labels, route, mission text, oracle action, demonstration, reward label, or online GPT call.

The first dramatic test is not a score. It is enforced equality. Both twins must reproduce the
confirmed U1 parent at zero context, then generate the same complete 2,048-transition first
rollout. Put the two action traces on top of each other until they look like one line. Only after
the first optimizer update is divergence allowed. That makes the experiment legible: before
learning, the intervention is behaviorally inert; afterward, only the candidate has useful context
from which to learn.

The release process can become a short credibility montage:

1. a clean source commit becomes the published one-line preregistration tag;
2. a durable qualification report binds that exact tag object and proves the real matched smoke;
3. the cohort manifest binds the report digest; and
4. one fixed launcher opens sham, then action-effect, while the read-only dashboard only reads
   evidence.

Do not depict the tag as containing the later qualification digest. The point is the opposite: each
step commits to what is knowable at that moment, without a circular or post-hoc claim.

Attempt 0 supplies a useful short operational beat. Qualification passed, but the dashboard tried
to reauthenticate the entire history on every poll. Its answer took longer than the ten-second
client deadline, three sockets closed, and the launcher correctly refused to start a trainer. Show
the empty arm/media inventory beside the synthetic manifest “sham crash”: the important reveal is
that this is **zero-action fail-closed evidence**, not a model failure.

r1 supplies a second, different operational beat. Its corrected evidence reader worked, sham
started, and the dashboard showed real learning activity. Then the launcher searched complete
command lines and counted the supervisor's embedded child command as another trainer. Put the two
process rows on screen and reveal that there is still only one neural child. A shell cleanup path
also failed to run, so sham continued until it was explicitly stopped.

Show the counters together without merging them: 40,004 collected in final status, 40,960 in the
later episode ledger, but only 38,912 trained and checkpointed. It completed 19 optimizer phases,
76 new updates, and one scheduled exam. The action-effect side of the split screen remains empty.
The latest-observed checkpoint is authentic but not safe. The conclusion is again operational,
not scientific: no matched comparison occurred.

Then introduce r2 as the same preregistered scientific question with both operational lessons
applied. It keeps r1's deep-auth-once dashboard design, classifies roles by leading executable
arguments, and explicitly cleans up on every assertion failure. Port `8790` and all r2 identities
are fresh; both arms restart from confirmed U1.

If Stage A eventually succeeds, frame it as permission to replicate the **idea**, not promotion of
the attractive model on screen. Both Stage-A checkpoints are discarded. Three fresh policies from
three independently confirmed U1 parents must learn the same architecture under a separately
frozen protocol, and untouched no-update confirmation must still pass before U3 can open.

### Ending options

- **Full success:** one policy retrieves the relic on unseen final levels and retains every prior
  skill. End by showing how the world can now grow.
- **Partial success:** identify exactly which concept became the wall and what the controlled evidence
  says. The project still answers a sharper question than the Pokémon attempt did.
- **Unexpected behavior:** center the strategy the agent discovered and determine whether it is robust
  or an exploit using new frozen tests.

## Capture checklist

- clean pixel-view recordings at each tier;
- dashboard time lapses with training and validation separated;
- procedural layout montages to establish that maps are unseen;
- the oracle solving a map, clearly labeled “qualification only”;
- promotion and retention events;
- the three-seed Navigate curve and synchronized 75% final retention bars;
- episode-share versus transition-share comparison after v0.1 promotion;
- key → door → exit conversion funnel;
- the v0.2 lesson staircase and target-versus-realized practice mix;
- U0's three-policy 200-case confirmation table;
- side-by-side U0 and U1 first-person frames: visible door versus hidden door;
- the U1 three-bar exam panel and an automatic U0- or Navigate-recovery transition;
- the three U1 trajectories converging on 72/80 through different recovery histories;
- the attempt-1 U1 confirmation stop: different seed numbers producing the same layouts, all 600
  oracle-qualified cases, nine reference collisions, and an empty policy-results array;
- the completed v2 selector funnels—200/200 Navigate, 200/236 U0, and 200/207 U1—followed by the
  three-policy confirmation table, clearly separated from both the failed measurement instrument
  and the shared development exam;
- a before/after digest card showing that v2 changed no policy or optimizer state;
- the visual transition from confirmed Local Unlock to the next bounded U2 Separated Unlock map;
- the U0 → U1 → U2 visual progression: both objects visible, hidden door, then the divided map with
  exactly two extra walls and a 17–26-action qualified route;
- the three U2 lineage cards sharing dashboard 8785 while only one CPU trainer runs;
- four simultaneous lesson bars with panel A/B scores and a visible recovery-practice shift;
- a clear engineering-proof → learning-evidence → confirmation-evidence title sequence;
- the unopened U2 seed ledger and blank result panel before any protected case or policy run;
- the three U2 baseline → first-exam → mastery trajectories and their exact action counts;
- a 3/3 mastery card paired with the explicit label “development replication; confirmation next”;
- the fail-closed U2 confirmation launch stops, each labeled “zero protected candidates opened”;
- the U2 selector funnels: 200/200 Navigate, 200/215 U0, 200/204 U1, and 200/200 U2;
- the three frozen U2 confirmation cards, ending with `169/200`, the untouched `170` gate, and
  `capability_failed — U3 blocked`;
- the failed lineage's 188 key → 172 door → 169 completion funnel beside the two passing peers;
- a before/after digest card showing zero U2 confirmation policy or optimizer updates;
- the U2r protocol diagram: immutable 2/3 result → eleven fixed windows → one sealed fresh exam;
- child `20260745` first-pass-versus-mastery ineffective-interaction comparison;
- an eleven-block action clock with no intermediate “best checkpoint” marker;
- the penultimate and terminal U2r stability cards, including the 3.0 ineffective-interaction
  ceilings and no-more-than-2/80 decline rule;
- the unopened `15_240_000`–`15_279_999` U2r confirmation envelope;
- if activated, the exact title card “two direct + one prospectively remediated,” never “original
  U2 3/3”;
- the U2-S matched 2 × 2 grid with one shared 24/80 inherited baseline;
- four synchronized 32-exam clocks ending at exactly 1,048,576 actions per arm;
- the final-three split screen: control/no-effect above the capability line, combined below the
  loop-tail line, and no cell satisfying both;
- the exact U2-S end card: `ablation_failed`, `selected_configuration: null`, no checkpoint reuse,
  U3 closed;
- the v0.3 twin diagram: identical parent/transplant, zero sham context versus previous
  action + visible effect;
- overlapping first-rollout traces through all 2,048 pre-update transitions, followed by a clear
  “learning may diverge here” marker at the first optimizer phase;
- the non-circular v0.3 evidence chain: source tag → qualification report → cohort manifest →
  fixed launcher, with the prospective r2 dashboard at 8790 labeled read-only;
- the attempt-0 zero-action card: three broken client pipes, no arm directory, no trainer, no
  action, and the trap's synthetic sham crash clearly labeled operational closeout;
- the r1 partial-sham card: status `40,004 collected / 38,912 trained`, episode ledger `40,960
  collected`, `19` optimizer phases, `76` new updates, one scheduled boundary, eight rows, 640
  cases, no safe checkpoint, and an empty action-effect column;
- the r1 process reveal: supervisor plus embedded child argv causing a false second-trainer match,
  followed by the missed shell cleanup and explicit later stop;
- the v0.3 Stage-A end card, whatever its eventual outcome, explicitly stating that no Stage-A
  checkpoint advances and U3 remains closed;
- one example of curiosity encouraging exploration without task knowledge;
- failed runs, including the first checkpoint-publication bug;
- the v0 audit, reward inversion, and stale-policy evaluation timeline;
- a v0.1 checkpoint sidecar tied to the exact evaluated model;
- milestone-rate visuals showing partial quest knowledge before full success;
- final frozen evaluation with seed list and no edits afterward.

## Claims to avoid

Do not call a training reward curve intelligence. Do not describe the oracle as part of the agent. Do
not imply the final suite was untouched if its results influenced later design. Do not call one lucky
episode mastery. Do not present the v0 canary as a capability result. The credibility of the video is
the experiment's most valuable output. Do not call planner/oracle qualification U2 learning, do not
pool three children into one synthetic policy, and do not describe development-exam mastery as
disjoint confirmation. Do not describe 169/200 as a pass, “round it” to 85%, or imply that two
passing policies satisfy an all-three preregistered rule. Equally, do not frame the one-case miss as
a total learning collapse. Do not present U2r as a rerun of the failed exam, hide that its stability
criteria were designed after observing the failure, select an intermediate checkpoint, or describe
a future U2r pass as retroactive 3/3 confirmation. Do not call no-effect an U2-S winner because its
averages were attractive, call combined eligible because it removed the measured tails, reuse an
ablation checkpoint, or imply that the failed mechanism study authorized U3.
