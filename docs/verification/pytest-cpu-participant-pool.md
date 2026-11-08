# Pytest CPU participant pool — 2026-11-09

Implementation and Root acceptance complete after final reviews and the
whole-invocation/test-only-delta checks below. The owner explicitly selected
pytest collection tests, not normal collection or CUDA pooling. Design:
[ADR0027](../adr/0027-pytest-cpu-participant-pool.md).

## Baseline and feasibility

The existing42-case CPU module's latest ordinary timing is455.41s; its later
observed breakdown is455.89s at code revision5d9824d/docs-only37ac83b. Exact
commands/conditions are in
[reference reuse](structured-ppo-test-reference-reuse.md) and
[collection detail](structured-ppo-collection-detail.md). Collection spends
71.45of75.70seconds with participant readiness waits active in the representative
GRU case. Normal helpers and the completed MLP reference must remain fresh.

A bounded CPU inference-only replay, before source implementation, compares the
same seven own-view GRU episodes (six distinct plus repeated first,14decisions):

| Mode | Processes | Wall seconds |
| --- | ---: | ---: |
| Fresh process and model each episode | 7 | 12.857 |
| Warm process, fresh model each episode | 1 | 1.771 |
| Warm process and cached model/cache arrays | 1 | 1.514 |

All selections match exactly across modes and the recorded source selections;
A...A also matches. Each reuse retains artifact/hash verification and resets
the policy and fresh event bank. This is one fixed-order offline replay, without
candidate regeneration, online collection, production protocol/lease/fault proof,
GPU evidence or model-strength authority. It establishes feasibility only.

Exact HOST command, exit0:

```sh
/usr/bin/time -v -o /tmp/godfieldbots-warm-policy-probe-20261109/resource.txt timeout --signal=INT --kill-after=15s 180s uv run python /tmp/godfieldbots-warm-policy-probe-20261109/probe.py > /tmp/godfieldbots-warm-policy-probe-20261109/probe.log 2>&1
```

The disposable script/results/child logs remain in that directory. No original
recorded data is edited; temporary Source paths may expire under pytest retention.

## Design review

Architecture is READY for design after pinning explicit viewer ContextVar,
stable physical cleanup owner, guarded prepublication callback plus unconditional
collector cleanup, and strict reset acknowledgements. Original bootstrap EOF or
worker_error return alone can never authorize a clean handoff. Physical process
and lease are real; no fake poll or early release.

The receipt is a bounded root `pytest_worker_lifecycle` diagnostic, outside
Journal5/features/targets. Current Python/native readers permit root extras and
hash the full original manifest, so there is no Source-reader/schema relaxation.
All component sources remain engineering-only/unqualified.

## Initial implementation evidence

The first39new pool checks PASS in77.85s, before final narrow cleanup fixes.
An affected constructor/reset/thread/deadline set PASS6in1.04s; helper fallback
checks PASS4in0.36s; cold baseline/timeout/three-model bootstrap checks PASS5in
5.99s. Logs are under `/tmp/godfieldbots-pytest-cpu-pool-20261109`.

These are scoped intermediate results, not final-source acceptance. Independent
AST checking confirms the default worker's original reservation/Popen/error block
is moved exactly to the protected spawn hook, with prior normal methods unchanged.

Initial conformance review was BLOCKED on two concrete findings, accepted by Root:
stale/closed logical loans must reject before changing a new borrower's state or
lease, and physical cleanup failure must not replace the original end/reset error.
Both are fixed with specific combined/stale-call regressions. Correctness review
also reproduced a50msI/O-thread retirement succeeding outside a20mslogical close
deadline; that third finding is fixed by retaining the first deadline through
thread retirement and memory observation, poisoning/reaping on overrun. HOST
13related checks PASS20.56s and six final affected checks PASS1.06s. Both final
reviewers rechecked the fixes and are READY, with no remaining blockers.

## Original target timing and physical lifecycle

HOST command, exit0:

```sh
PYTHONPATH="/tmp/godfieldbots-pytest-cpu-pool-20261109${PYTHONPATH:+:$PYTHONPATH}" /usr/bin/time -v -o /tmp/godfieldbots-pytest-cpu-pool-20261109/pooled-target-resource.txt timeout --signal=INT --kill-after=15s 900s uv run pytest tests/test_structured_ppo.py -p pytest_pool_timing_probe --setup-show --durations=0 --durations-min=0 --junitxml=/tmp/godfieldbots-pytest-cpu-pool-20261109/pooled-target-junit.xml > /tmp/godfieldbots-pytest-cpu-pool-20261109/pooled-target.log 2>&1
```

PASS**42tests**, pytest241.42s, external wall**241.87s (4:01.87)**. Compared with
the preserved fresh42-case baseline455.89s, this one observation saves214.02s
(**46.95%**). Identical original node IDs/statuses, game/update budgets, duration/
setup reporting and fixture observer are verified. The observer differs only in
module name/output directory. Baseline is before the default-preserving source
seams; single-pair/cache/order variation remains a limitation. Immutable baseline
copies, code fingerprints and arithmetic are in the log directory's
`fresh-baseline-*`, `target-measurement-metadata.json`, `target-comparison.json`.

The ten pooled comparison collections contain320logical participants,40actual
physical processes,320verified resets and zero remaining worker reservations;
each collection has32logical participants and at most4physical owners, split
permanently across viewers1/2. The shared cold reference remains fresh. Every
healthy receipt is recorded before manifest hashing. Audit: `target-pool-audit.json`.

Peak sampled pooled-owner sum PSS is1826.52MiB and sum RSS2895.52MiB. These cover
owners only, not learner/coordinator; RSS double-counts shared pages and sampling
can miss transients. There is no paired aggregate fresh-memory measurement, so
no claim of unchanged total memory. All current admission limits are preserved.

## Package and standard checks

`uv run ruff check src tests native` PASS; `uv run mypy src` PASS157source files;
collection PASS4280tests; whitespace checks PASS. The known four pre-existing
untracked-user-script Ruff findings remain outside this change.

`uv build --wheel --offline` PASS. All157wheel Python modules match current
source bytes; no tests module is packaged. Wheel SHA256:
`6eea33f2eaa9ca835b8275ab24c82457726ccc3d42d5b483cf263c7407cb6519`.
An isolated Python3.14environment installed this wheel and mandatory dependencies
offline, with no Torch/LightGBM/PyArrow/native/pytest. Installed normal structured
protocol2Random initialization and real close/reap pass with empty SAME ledger,
no tests/ML imports. `gf-bots --help` passes outside the repository.
Evidence: `wheel-correspondence.json`, `baseline-smoke.log`, `baseline-help.log`.

## Whole invocation and final test-only correction

Exact HOST command:

```sh
/usr/bin/time -v -o /tmp/godfieldbots-pytest-cpu-pool-20261109/full-resource.txt timeout --signal=INT --kill-after=15s 3600s uv run pytest --setup-show --durations=0 --durations-min=0 --junitxml=/tmp/godfieldbots-pytest-cpu-pool-20261109/full-junit.xml > /tmp/godfieldbots-pytest-cpu-pool-20261109/full.log 2>&1
```

The invocation ends exit1: **4,032passed / 247skipped / 1failed / 4warnings**,
pytest1389.06s, external wall1390.16s (23:10.16). It is NOT a fully green run.
All4,225cases from the prior whole invocation preserve their pass/skip statuses;
one generated UUIDv1parameter label is normalized only for that identity audit.
The existing42-case target is green within this whole invocation as well.

The sole failure is the NEW
`test_cleanup_deadline_and_owner_health_prevent_healthy_receipt[deadline]`:
the approved deadline enforcement correctly raises TimeoutError during logical
worker.close, before the test's later expected pool-close failure. The test
expectation had not followed that earlier rejection. Only this test branch is
changed to require the precise early TimeoutError; subsequent pool failure,
SAME close task/original deadline, actual reap/lease release/empty ledger and
emergency-grace assertions remain. No implementation/guard/deadline is relaxed.
The failed full.log/XML remain untouched.

Final HOST changed-area command, exit0:

```sh
.venv/bin/pytest -q tests/test_pooled_structured_worker.py --tb=short > /tmp/godfieldbots-pytest-cpu-pool-20261109/final48-after-whole-fix.log 2>&1
```

PASS **48pool tests in77.85s** after the expectation correction. Root and
independent architecture review verify every implementation/helper fingerprint
still matches the timed/frozen source. Reversing only that single expectation
restores the recorded test-file hash. Both final reviewers retain READY and
find no order/semantic concern requiring another unrelated23-minute invocation.
Acceptance uses the complete frozen-implementation run plus the corrected
48-case changed-area rerun; it does not claim a new all-green whole invocation.
Receipt: `final-acceptance-audit.json` in the log directory.

## Root acceptance and limits

Root accepts the explicitly test-only CPU pool after the original target42passes,
all three exact wire/model-reset checks, physical ownership/failure regressions,
both independent reviews, whole plus test-only-delta validation, standard checks,
source-matched wheel and mandatory-only normal-worker smoke. No normal runtime,
CUDA, native helper, simulator rule, model arithmetic or dependency changes are
introduced. The two production seams retain default behavior; the pool/bootstrap
remain outside the packaged wheel. Lifecycle receipts remain diagnostic and
component data unqualified.

The measured speed claim is the42-case target, not the whole project or the
offline probe's ratio. Actual CUDA pooling is outside scope; fresh CUDA paths
retain their opt-in gates. No model strength or formal Source claim follows.
