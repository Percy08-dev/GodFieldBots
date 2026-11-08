# Shared pytest PPO sources — 2026-11-09

Implementation and Root acceptance complete after both independent final reviews.
Design:
[ADR0025](../adr/0025-session-shared-pytest-ppo-sources.md).
The preceding equivalence and timing investigation is in
[pytest runtime analysis](pytest-runtime-analysis.md).

## Scope and design review

Global pytest discovery of `frozen_sources`, with once-per-session current-code
generation of the same three legacy CPU MLP source parents and validation runs.
All consumers receive isolated, path-rebound and normally validated copies.
No persistent dataset cache, optional dependency, runtime math/guard change or
Simulator change is included.

The independent architecture reviewer is READY for design after Root accepted
the original-template validation finding: full `build_dataset` must validate
untouched original data before clone resealing can change any source proof.
The two-stage private copied-chain construction was separately checked against
the existing offline validators. Final implementation reviews are READY below.

## Measured unchanged baseline

Baseline source: `8eb6a7ff20804da665a23a0827f674bb7d20a2b2`, no tracked code/test
changes at measurement. Same installed environment as the earlier runtime
analysis. Whole default-run evidence had seventeen actual fixture generations,
including preparation and native-fixed consumers; the earlier fifteen-generation
subtotal covered fixed-data and native-session files only.

Exact baseline command, repository root, HOST execution:

```sh
/usr/bin/time -v -o /tmp/godfieldbots-shared-fixtures-20261109/before-resource.txt timeout --signal=INT --kill-after=15s 600s uv run pytest tests/test_fixed_ppo_data.py tests/test_native_session.py tests/test_preparation.py tests/test_native_fixed.py tests/test_native_parallel_parse.py --durations=0 --durations-min=0 --junitxml=/tmp/godfieldbots-shared-fixtures-20261109/before-junit.xml > /tmp/godfieldbots-shared-fixtures-20261109/before.log 2>&1
```

PASS: **48 passed / 8 skipped**, exit0. Pytest **218.07s**, external wall
**219.08s**. The eight skips are the unchanged opt-in parallel-parse cases.
This includes original fixture generation and all selected test bodies.

## Independent implementation reviews

Correctness and architecture-conformance reviews report no blocking findings.
Independent checks PASS: exact AST identity of the moved helper bodies, clean
imports without Torch or collected test modules, and an actual finished clone
with 1,033 regular non-symlink files on distinct inodes. Its 1,015 files outside
the eighteen specified management records remain byte-identical, and no copied
file references the template root. Whitespace checks PASS after correction.

The original-template stale-source-proof regression exercises a coherent chain
before expecting source validation to reject, so resealing cannot heal a bad
generator proof. A nonblocking review suggestion requests the same coherent-chain
setup for the unsupported-layout regression. That refinement is implemented and
the targeted host check passes; the reviewer closed the optional finding.

The first eight-file scoped run was interrupted with exit130 under the sandbox
after partial progress in native fixed preparation. It is NOT a scoped PASS;
`implementer-scope.log` is retained. The unchanged command passes on the
HOST, matching the baseline environment. No production fix was made for the
sandbox-only interruption.

## Scoped regression and controlled timing

HOST scoped command, exit0:

```sh
timeout --signal=INT --kill-after=10s 600s .venv/bin/pytest -q tests/test_shared_ppo_sources.py tests/test_fixed_ppo_data.py tests/test_native_session.py tests/test_preparation.py tests/test_native_fixed.py tests/test_native_parallel_parse.py tests/test_ppo_lineage.py tests/test_journal.py --durations=15 > /tmp/godfieldbots-shared-fixtures-20261109/implementer-scope-host.log 2>&1
```

PASS **104 passed / 8 skipped in128.60s**. The strengthened coherent-version
test separately PASS one case in14.52s under the host's120s bound; log
`implementer-version-recheck.log`. The initial focused run's exception-expectation
failure is preserved in `implementer-shared.log`; the new test was corrected to
accept the existing journal reader's ValueError/ContractError behavior.

The exact controlled after command is the baseline five-file command above,
with only the three output names changed from `before` to `after`. PASS:
**48 passed / 8 skipped**, exit0, pytest71.50s, external wall**72.22s**.
JUnit node IDs and pass/skip statuses are identical in both populations.

| Scope | Before seconds | After seconds |
| --- | ---: | ---: |
| External wall, including all generation/copy/check work | 219.08 | 72.22 |
| Fixed-data file, JUnit phase total | 63.091 | 25.466 |
| Native-session file, JUnit phase total | 125.539 | 35.044 |
| Preparation file, JUnit phase total | 15.707 | 6.633 |
| Native-fixed file, JUnit phase total | 13.074 | 4.155 |

This one before/after pair saves146.86s (**67.03%** external wall;3.03x speed
ratio) on the same selected population. It includes fresh per-invocation template
generation, every copy/reseal and normal source validation; no persistent cache,
smaller data or changed failure assertions are used. Single-pair/cache/order
variation remains a limitation. Machine-readable arithmetic and population audit:
`/tmp/godfieldbots-shared-fixtures-20261109/comparison.json`.

Full mypy PASS157source files; `uv run ruff check src tests native` PASS.
`uv run ruff check .` FAILS only on four pre-existing findings in the user's
untracked `codex_analysis.py`; it is preserved unchanged. Changed-file Ruff,
helper AST preservation and whitespace checks PASS.

## Whole-suite and lifecycle acceptance

Exact HOST command, exit0:

```sh
/usr/bin/time -v -o /tmp/godfieldbots-shared-fixtures-20261109/full-resource.txt timeout --signal=INT --kill-after=15s 3600s uv run pytest --setup-show --durations=0 --durations-min=0 --junitxml=/tmp/godfieldbots-shared-fixtures-20261109/full-junit.xml > /tmp/godfieldbots-shared-fixtures-20261109/full.log 2>&1
```

PASS **3,978 passed / 247 skipped / 4 warnings**, no failures; pytest1657.36s,
external wall**1658.54s (27:38.54)**. All4,217previous cases retain pass/skip
statuses. One `test_http` parameter's displayed UUIDv1 changes because the test
calls `uuid.uuid1()` at collection; the population audit normalizes only that
known parameter label and records both raw labels. The only eight added cases
are the passing shared-source regressions.

The actual whole-run setup/teardown trace contains exactly one session template
and24function-scoped `frozen_sources` requests, including the seven requesting
new regressions. The retained master at
`/tmp/pytest-of-percy/pytest-147/shared-ppo-sources0` passes ordinary
`inspect_dataset` again after the whole run:48train/192validation rows, with
each of17/29/43complete at one update and one original learner invocation.
The whole-run receipt and trace counts are in
`/tmp/godfieldbots-shared-fixtures-20261109/full-acceptance.json`.

Root accepts the test-only implementation. No runtime math/guard/dependency or
Simulator changes were made. There is no cross-run cache. Current fixed-layout
support must be revisited explicitly when its fixture recipe/schema changes.
The full invocation includes eight added regressions and extra setup reporting;
the controlled speed claim is the unchanged five-file pair, not a same-population
whole-suite estimate. No model-quality or formal-source claim follows.
