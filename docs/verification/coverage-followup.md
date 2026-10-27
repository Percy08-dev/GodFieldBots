# Phase 0–3 coverage follow-up

Date: 2026-10-27. Scope: the user's request to strengthen the four gaps found
in the coverage assessment. This is test-only work under the accepted architecture
and ADR0002. No dependency, public interface, schema, information boundary or
simulator change is authorized by the test plan. A discovered production defect
returns to root for classification and a bounded fix before acceptance.

Historical schema-3 record: [ADR0005 and ActionOptions verification](action-options.md)
subsequently supersede the old input prohibition and storage boundary. Current
journals use schema 4 with authorized actor atomic hints. Percentages and artifacts
below remain the actual frozen coverage-follow-up measurements, not current counts.

## Baseline

The measured suite passed 123 ordinary tests and two opt-in real-binary tests.
Coverage used a temporary `coverage` installation without changing project
dependencies; data and HTML were written under `/tmp/gfb-coverage-audit`.

| Scope | Line coverage | Branch coverage |
| --- | ---: | ---: |
| All `src/` | 83.8% | 74.8% |
| Public candidate generation | 90.2% | 85.5% |
| Environment orchestration | 69.8% | 50.8% |
| Journals and derived data | 86.8% | 75.5% |

These percentages include both ordinary and integration tests, but do not
collect execution inside fresh worker subprocesses. Their bootstrap's measured
zero does not mean its behavior was untested. Manual wheel/CLI checks and the
12-game fixed evaluation are also outside these automated coverage counts.

## Required behavioral additions

1. Exercise the real episode orchestration through controlled I/O: rejection,
   abort/truncation, unsupported contract, unknown transport outcome, save
   failure and cancellation. Inspect persisted transitions, pending delivery,
   result/health and data eligibility; do not fabricate rewards or retry policy
   proposals. Keep existing natural-outcome semantics.
2. Audit both participants' observation/event/cursor/catalog routing and use
   deterministic event barriers to test active/queued concurrent groups,
   admission, cancellation and resource release. Avoid sleep-based ordering.
3. Handwrite small exact candidate sets from the pinned public specification,
   independently of the generator implementation. Include Flash, End Times,
   visible/perceived identity, physical instances/order, learned MP boundaries,
   basic choices and trade syntax within the declared public subset. Compare
   both search implementations against these oracles. Unclear public rules
   remain explicit questions, not guessed engine legality.
4. Test the CLI parser, overrides and exit codes, including both module entry
   points and successful/failed execution. Preserve lazy optional imports.

Real-binary additions may cover a bounded non-natural ending or external-owner
mode when deterministic fixtures permit; they do not attempt a training pilot
or reinterpret the unresolved four rejections in the fixed evaluation.

## Verification and acceptance

The implementer owns `tests/`; root owns this record. Independent correctness
review checks that assertions express the contract rather than mirror code,
that fakes do not replace the behavior under test, and that expected candidate
sets do not depend on the production action-space/scorer. New public boundaries
or dependencies require the architecture gate before dependent changes.

Run focused tests, the ordinary suite, Ruff/mypy, affected opt-in integration,
then repeat the same parent-process line/branch measurement. No arbitrary
percentage target substitutes for these behavioral checks. Record PASS/FAIL/
NOT RUN, remaining risks and known measurement limits below. Preserve existing
uncommitted work and do not commit or publish.

## Defects found during coverage expansion

The whole-episode regression exposed lost simulator truncation cause. The
test-only classification was escalated at that boundary: ADR0003 was written
and independently reviewed READY before the dependent production fix. It
preserves raw lifecycle Aborted while projecting exact turn-limit evidence
as nonterminal truncation, with a schema-3 compatibility boundary.

A separate active-plus-queued cancellation regression exposed CLI return before
owned-group cleanup. ADR0004 was independently reviewed READY before its final
late/repeated cancellation policy was implemented. Follow-up review also caught
queued admission immediately after a peer exception and loss of child TimeoutError
origin. The final coordinator closes admission before releasing the failed slot,
drains all owned tasks, and preserves the original child exception.

## Final results and scope

Root accepts this follow-up after final independent correctness and architecture
conformance reviews both returned **READY**. No blocking findings remain.

| Check | Result |
| --- | --- |
| Ordinary pytest suite under coverage | PASS: 200 passed, 3 opt-in skipped, 6.50s |
| Actual TestServer opt-in suite under appended coverage | PASS: 3 passed, 200 deselected, 2.01s |
| `uv run --no-sync ruff check .` | PASS |
| `uv run --no-sync mypy src` | PASS: 22 files |
| Independent correctness final subset | PASS: 22 group/budget tests, 2.92s; original two probes resolved |
| Independent architecture final subset | PASS: 14 ownership tests, 0.53s; previous 92 affected checks passed |
| Isolated installed wheel, no ML/data extras | PASS: two natural games, Replay/EOF/reap healthy; both module/console entry tests also pass |
| Installed wheel local cutoff | PASS: expected incomplete result/exit 1, two truncated games, no natural result or terminal reward |
| Schema-3 deterministic export and metadata separation | PASS: 32 natural rows eligible; two cutoff rows ineligible, four end records retained |

Final source was frozen during measurement. Code also grew to fix the discovered
defects, so the percentages compare two actual revisions, not a fixed denominator.

| Scope | Lines before → after | Branches before → after |
| --- | ---: | ---: |
| All `src/` | 83.8% → 88.0% | 74.8% → 79.4% |
| Environment orchestration | 69.8% → 86.3% | 50.8% → 72.7% |
| CLI composition/parser | 80.7% → 96.9% | 61.9% → 89.4% |
| Journals/export | 86.8% → 88.4% | 75.5% → 76.9% |
| Candidate generation | 90.2% → 90.2% | 85.5% → 85.5% |

Candidate coverage did not increase numerically: the new independent exact
expected sets strengthen the oracle for already-executed paths. Both enumerate
and wide beam must match hand-authored sets for order/physical references, MP
boundaries, Flash, End Times, trade, buy decisions and perceived Dream identities.
Learned defensive miracles remain outside the declared v1 subset when the
required incoming-attack information is unavailable; tests document the omission
rather than invent an engine-legality claim.

Whole-episode tests use the real HTTP client with scripted public transport,
real candidate generation, journals, finalization and persisted results.
Game IDs/hand references/cursors differ per Create. Each input retains its
viewer-private stream and legitimate opponent-public events, without response
events/options or control metadata. Concurrent tests hold explicit barriers to
prove no queued resource creation or parent return before journal/Replay/log/reap
ownership ends. CLI tests cover parser/overrides/exit codes, both module and
console entry points, and lazy optional imports.

## Reproducing the coverage measurement

No project dependency or threshold was added. A temporary cached coverage tool
was used; network/listener sandbox approval was needed for initial tool download
and live integration. Ordinary application dependencies and `uv.lock` are unchanged.

```console
mkdir -p /tmp/gfb-coverage-followup
UV_CACHE_DIR=/tmp/gfb-uv-cache COVERAGE_FILE=/tmp/gfb-coverage-followup/.coverage uv run --offline --no-sync --with coverage python -m coverage run --branch --source=src -m pytest -q
UV_CACHE_DIR=/tmp/gfb-uv-cache COVERAGE_FILE=/tmp/gfb-coverage-followup/.coverage GFB_RUN_INTEGRATION=1 uv run --offline --no-sync --with coverage python -m coverage run --append --branch --source=src -m pytest -q -m integration
UV_CACHE_DIR=/tmp/gfb-uv-cache COVERAGE_FILE=/tmp/gfb-coverage-followup/.coverage uv run --offline --no-sync --with coverage python -m coverage html -d /tmp/gfb-coverage-followup/html
```

Artifacts: `/tmp/gfb-coverage-followup/coverage.json`, `summary.json`,
`html/index.html`. Totals: 2,172/2,469 statements, 838/1,056 branches.
Worker and module-entry subprocess execution remains uncollected by this parent
measurement, although real subprocess behavior is tested. No full-coverage or
hard-real-time guarantee is claimed.

## Package/data evidence and compatibility

- Final wheel SHA-256:
  `9b84a943a1d4980c78844f5cfd1ac05f65641e1e45670922f87b61eb5abe3732`.
  All 22 packaged Python files match final source. Installed into
  `/tmp/gfb-wheel-verification`, run from `/tmp` with a moved prebuilt simulator
  binary and no local simulator source requirement. Torch/LightGBM/PyArrow were
  neither installed nor imported in that environment.
- Natural run: `runs/wheel/wheel-no-ml-90d56084-5f0`, two natural games,
  CLI total 1.593s. Manifest and all journals use schema 3 and
  `end_projection_version=simulator-closure-v1`.
- Local cutoff run: `runs/wheel/wheel-local-cutoff-67c3eb0f-fc3`, two planned
  games truncated by `max_decisions=1`, CLI total 0.350s, invalid comparison.
  Real simulator turn-limit projection was NOT RUN: minimum supported TestServer
  max turns is 150; its exact nonterminal projection is covered by closed public
  Replay traces. No unsupported max-turn=1 or simulator fixture was introduced.
- Export artifacts: `/tmp/gfb-coverage-followup/data/clean-{a,b}.parquet` are
  byte-identical, SHA-256
  `c2a18b8814777be87de65df4ce95ecec5f1e2dac4935fbff5381e0c8066b0579`.
  `cutoff.parquet` has two decision rows, zero eligible rows and four participant
  endings in its sidecar, without terminal rewards. Neither rows nor schema
  metadata carry GameSeed/fingerprint/URL/abort reason; diagnostics are sidecar-only.
- Historical schema 1/2 data was retained, explicitly rejected by the current
  reader, not converted/reinterpreted. The previous schema-2 wheel is retained at
  `dist/archive/9f312799b2bfdfeb2ab8668ce7cb9e51c8eef534899784f9848feb29729c06cb/`.

The simulator pin remains `44a26e3a16b808e826deb332fa72acfc5037d3ef`;
binary hash `d9c774d214e0e732fed040e7ef2e386979fc1f42e225429c1dbcfc642b49831f`
still has unknown build revision. Existing simulator README-only work was
preserved and no simulator file was changed.

The 4/12 rejections have subsequently been [diagnosed](../issues/0001-public-observation-legality-investigation.md)
and reproduced from a fixed public trace; public-contract clarifications remain
[separate Simulator tasks](../issues/0002-learning-contract-handoff.md).
The clean learning pilot, Phase 4–9 models, GPU validation and training gate
remain outside this coverage follow-up and unfulfilled.
Accepted synchronous OS/FS timing and dual-correction failure/quarantine limits
remain documented in architecture/ADRs. No commit or publication was performed.
