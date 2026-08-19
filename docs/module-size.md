# Module size, and why most of it is not being fixed

## The situation

Thirty source modules exceed 1,000 lines. The largest, `v02_u2r.py`, is 5,888.
Across the `v02_*`, `v03_*`, and `v04_*` families there are sixteen separate
`main()` functions and sixteen `build_parser()` functions; of 1,060 total
definitions only 707 names are unique. That is not a design. It is what happens
when each study is implemented as a new generation of the previous one and the
old generation must keep working exactly as it did.

This is a real readability cost and it is worth naming plainly: a reviewer
opening `v02_u2r.py` cannot hold it in their head, and neither can the author.

## Why the experiment modules are not being split

`AGENTS.md` states the constraint the whole repository rests on: the completed
cohorts, confirmation attempts, and their predecessors are **frozen evidence**,
and their environments, reports, checkpoints, launchers, and declared results
must not be modified.

The value of this repository is a preserved negative result — a study that
stopped because a prospectively frozen gate failed, rather than one that weakened
the gate after seeing the data. That claim depends on the code that produced each
report not being rewritten afterwards. A structural refactor would not change any
result, but it would make "this code produced that report" a statement requiring
trust rather than inspection.

Given a choice between a documented readability weakness and an undocumented
integrity question, the readability weakness is the better trade. So the frozen
modules stay as they are, and this document exists so nobody has to guess whether
that was a decision or an oversight.

## What *is* split

`train.py` was 1,115 lines and is live infrastructure rather than frozen
evidence. It is now the `dungeon_apprentice.training` package:

| Module | Responsibility |
| --- | --- |
| `training/config.py` | Effective-configuration capture, resume metadata, run paths |
| `training/callbacks.py` | The callback factory that drives a run |
| `training/cli.py` | Argument parsing, validation, and `main` |

`train.py` remains as a re-export shim, so `dungeon-train` and every existing
import keep working. `tests/test_training.py` was updated in one place:
`evaluate_policy` is now patched on `training.callbacks`, the module that
actually resolves it, rather than on the shim.

## How the line is held

`tests/test_module_sizes.py` records every frozen module at its current size and
fails if any of them **grows**. Everything else — new modules, live
infrastructure — is capped at 700 lines. The frozen list is also capped in
length, so a new oversized module has to be split rather than grandfathered.

The distinction matters. The existing large files are evidence. A new one would
just be a large file.
