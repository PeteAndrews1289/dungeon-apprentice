# Security policy

## Project status and scope

Dungeon Apprentice is a concluded, local research project. It does not operate a hosted service,
accept untrusted network input, or publish historical model checkpoints. Security maintenance
therefore focuses on the current default branch, dependency hygiene, unsafe artifact loading, path
handling, and defects that could invalidate the documented experiment or evidence boundaries.

Historical protocols, results, tags, and artifact digests are immutable research evidence. A
security fix must not silently rewrite those records. If a vulnerability affects a historical
claim, the correction should be published as a new, clearly linked erratum.

## Reporting a vulnerability

Please use GitHub's private vulnerability-reporting flow for this repository when it is available.
If that option is not shown, contact the maintainer through
[Peter Andrews's GitHub profile](https://github.com/PeteAndrews1289) without posting exploit details
in a public issue.

Include:

- the affected file, command, or dependency;
- the impact and conditions required to reproduce it;
- a minimal proof of concept, if safe to share privately; and
- whether the issue could alter generated evidence or cross the policy's declared information
  boundary.

Please do not include real credentials, execute a test against systems you do not own, or publish a
working exploit before the report has been assessed.

## Supported version

Only the latest commit on the default branch is considered for security documentation fixes. Frozen
experiment tags are preserved for reproducibility and will not be retagged or rewritten.
