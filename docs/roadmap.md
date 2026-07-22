# Roadmap

## Foundation

- [x] Deterministic procedural generator for Tiers 0–2
- [x] Pixels-only policy observation wrapper
- [x] Full-state solvability oracle kept outside learning
- [x] Seed-partitioned validation and final suites
- [x] Recurrent PPO training and atomic resume
- [x] Automatic promotion and retention testing
- [x] Living local dashboard and append-only evaluation history
- [x] Short real training canary on Tier 0

The first end-to-end 64-step smoke run completed on July 22, 2026. It exercised initialization,
experience collection, checkpoint publication, held-out evaluation, status history, frame capture,
model reload, and clean shutdown. It is an engineering test, not a learning result.

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
