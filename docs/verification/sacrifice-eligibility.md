# Shared Discard and Sacrifice eligibility verification

## Runtime identity

The HPO6 preset selects `card-instance-simulator-v2-shared-20261125-sacrifice`.
The Simulator revision remains `6d50603dab3f6d2df34ea4e433bab7dca96f52c1`.
The source inventories, TestServer and formats are recorded in the
[Simulator release verification](pestle-mirage-simulator.md#release-identity).
The corrected profile has its own candidate rule signature and uses dedicated
`mlp-256-convergence-1000-sacrifice` storage for new convergence runs.

| Item | SHA-256 |
| --- | --- |
| Corrected rule signature | `a3256001cae7fa4fdb588a685e2fd37e0eac5107c3b674e01fb940aca064bbb7` |
| Native source inventory | `e9899d3c998302b9b4a6e0f440c273591734f9e6f16291f8275ef7e5cc96a3c9` |
| Native wheel | `052cd6e54a29b0a61c2bdc987d9538f75e95656fe85e030f6487d24ba21f5470` |
| Native extension | `07240a98b7d4d2b037cf6d87878d53bbeb9bcb7620a0d16714a09ee24016cd92` |
| Native supervisor | `3510b9bb6c1a63cb20403e327035bc8e054c3b8ebdb49d776bcee103c83ae1bc` |

## Archived failure

The archived seed-17 parent `83355bf4-1cf3-43c3-94be-9dae91de7290` committed
43 updates and 168,061 learner decisions. At 03:11 JST on November 25, 2026,
Player 2 in `batch-000043`, pair `p81`, selected Energy Armor and Dangerous
Mortar for Sacrifice at turn 150 and revision 215. The Simulator rejected
the selection with HTTP 422 `invalid_parameter`.

The archived HTTP log SHA-256 is
`5175cbd349c51c429fd1862384b52333be1d373a83445923989f2bc4ab655a37`.
The saved trace includes the acting player's same-revision ActionOptions with
`can_sacrifice=true`. That boolean gates the action family. The candidate
generator supplies the item pool using the confirmed shared eligibility rule.

## Candidate and compatibility checks

- **PASS** The [p81 fixture](../issues/cases/0018-sacrifice-protected-artifact.json)
  replayed all 215 accepted actions with the same Simulator for both Action
  profiles. The old Python and Rust searches generated the rejected pair.
  The repaired searches included the armor-only submission and excluded
  protected references. The Simulator accepted the armor-only submission at
  revision 216, removed the armor, and advanced to turn 151.
- **PASS** `python -m pytest -q -rA -n0 tests/test_hpo_dream_discard_reproduction.py`
  passed seven pinned replay cases. The opt-in p11 case was skipped because
  its approved source was not supplied for this run.
- **PASS** The focused Python suite covering CardInstance actions and rules,
  runtime receipts, HPO config and entry, and Native search and runtime
  passed 742 tests, with 56 skipped. The Native library suite passed 221 tests,
  with 13 ignored. `ruff check src tests`, `mypy src` on 288 files, and
  `git diff --check` passed. The full Python suite was **NOT RUN**.
- **PASS** Independent reviews verified that both profiles accept the same
  physical Simulator and reject each other's Action-contract receipts. The
  previous profile's rule signature remains
  `2b6c09bd00ba070b1fb1a90535aee5831b3407bc2edded94db288bcbc1f0da58`.
  The old healthy Pestle/Mirage smoke `0b87482b-0635-45c4-8a01-a1ba1dfc58d7`
  passed a fresh full Journal, Replay, and Python numeric audit with the
  new wheel. Its audit SHA-256 is
  `e49f7a4ebc5b60793eece1cc3c9ea572878a425661833e90ccf84527b3680519`.

## CUDA HPO6 smoke

**PASS** The wheel installed in the main worktree `.venv`, and the actual
default preset preflight checked the Native companion, CUDA, pinned Simulator,
and dedicated storage. Smoke `9afb390d-f483-4dce-9467-685ef0f96749` used seed 17,
an RTX 5070 Ti, CUDA Torch `2.14.1+cu132`, and eight games. All games ended
naturally, with 149 accepted actions and 77 recorded learner decisions. It
saved one PPO update and paused. The independent original Journal, Replay,
and Python numeric audit passed. The existing TensorBoard worktree's
`hpo watch --once` displayed the committed update. The run, preflight, and
audit records are in `runs/sacrifice-eligibility`.

Measured collection and update/save times were 4.59 s and 1.91 s. No
throughput comparison was run. The full convergence run was **NOT RUN**.
Smoke audit SHA-256 is
`956fd0d0545fdb920b2e1cdf7953bc48a8050b4fe1799cc5d3ac92749e12d412`.
