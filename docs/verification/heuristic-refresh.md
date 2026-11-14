# Refresh heuristic verification

Status: implementation in progress. This record distinguishes design readiness,
component correctness, runtime compatibility and actual-game/learning checks.
No strength or formal-source/learning qualification is claimed.

## Frozen scope

Accepted [design](../design/heuristic-refresh.md), [ADR0039](../adr/0039-refresh-heuristic-and-mixed-opponents.md),
and [progress](../design/heuristic-refresh-progress.md). Simulator91d5926,
corrected-v2/public-ja-v3; old data/defaults/formal kernels preserved.

## Evidence so far

- Persisted design architecture recheck: READY, no blocking findings.
- Baseline: `.venv/bin/python -m pytest -q tests/test_card_instance_rules.py
  tests/test_card_instance_actions.py tests/test_card_instance_baseline_runtime.py
  tests/test_card_instance_cli.py` — PASS176,3.46s, before implementation.
- New model/experiment preset TOML syntax: PASS; application parsing NOT RUN.
- Task1:44 new tests PASS;224 related legacy cases PASS; scoped Ruff/mypy PASS.
  Root freshly ran `.venv/bin/python -m pytest -q tests/test_refresh_heuristic.py`:
 44passed in0.44s. Independent correctness/quality and architecture conformance READY.
  Reviewer also independently checked2047subset masks and transactional retry.
- Retention diagnostic:2097152-byte principal array,2295560-byte tracked transform
  peak,0.011340s local; O(K) facts preallocated/excluded. No timing guarantee.
- Application/runtime/native/PPO and real games: NOT RUN at Task1 milestone.

## Predeclared functional protocol

Use fresh directories below `runs/heuristic-refresh/`; never reuse prior parents
or alter the existing approved Simulator executable. Record actual Bots source
state, Simulator receipt, configs, sampling seeds, labels/options and run roots.
Keep failed attempts as failed evidence. Truncation/abort/rejection is not a draw.

1. Actual baseline: new-vs-old and new-vs-Random, one fixed game-seed pair per
   condition with swapped seats and sampling seed17. Default new threshold12;
   use the full corrected action preset for these games. Require four natural
   healthy games, actor-local journals and original Replay closure/inventory.
2. Runtime selectability: exercise the new policy in both configured roles and
   two distinct thresholds in component/actual worker checks; verify labels and
   options persist through journal/collection readers. A second threshold need
   not supply a separate strength experiment.
3. Mixed16 functional collection: ordinary and model4/HPO paths use explicit new
   descriptor, R4/H2/refresh2/snapshot8 and healthy natural outcomes. A deliberately
   bounded small action/model smoke configuration is permitted only if frozen
   before launch and identified separately from default-action baseline games.
4. PPO functional continuation: exercise actual update/save/pause/resume, Python
   and source-matched native structural readers as supported. Component fixture
   numeric tests do not substitute for actual-game collections. Ordinary and HPO
   have separate acceptance evidence; no shortened run is a formal parent.
5. HPO production checks must retain actual GPU job/admission/Source ownership.
   Do not bypass production gates merely to call a synthetic check a real run.
   Record unavailable hardware/admission checks as NOT RUN, never PASS.

Before final acceptance, run affected/whole project static/tests, historical
proofs and source/package correspondence, inspect bounded worst-case retention
memory/time, then obtain independent correctness and architecture conformance.

## Results

Implementation and functional rows will be appended with actual commands,
counts, identities and material limitations. No results are implied by this
predeclared protocol.
