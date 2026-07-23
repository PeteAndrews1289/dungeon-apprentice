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
the experiment's most valuable output.
