# Pytest concurrent participant readiness — 2026-11-09

Status: accepted; both independent reviews READY and final scoped checks PASS.

[ADR0028](../adr/0028-pytest-parallel-participant-readiness.md) keeps ADR0027's
collection-scoped physical pool, original worker initialization and real cleanup.
It changes scheduling only in explicitly opted-in CPU engineering pytest helpers.
The cold shared MLP reference keeps one fresh process per participant.

## Prior observation and alternatives

The retained original42-case run is42PASS, external241.87s; its test phase sum
is241.243932s. Source/observer metadata and raw reports are under
`/tmp/godfieldbots-pytest-cpu-pool-20261109`. That timing remains the same-target
baseline for the final implementation, not the earlier pre-pool455.89s run.

The post-pool diagnostic under
`/tmp/godfieldbots-pooled-collection-detail-20261109` passes the original fresh MLP
and GRU comparison cases:2PASS, pytest92.10s, external93.09s. GRU collection has
four batches/64games/256decisions/128logical participants/16physical processes,
23.064881s collection wall and16.251377s readiness union. Each first neural
preparation averages1.407323s, later preparations0.036434s. Memory diagnosis is
260original `smaps_rollup` observations/5.085228s elapsed, overlapping readiness
and selection. It is deliberately unchanged in this increment.

Cross-collection persistence is excluded: `ReservationLedger.begin_update`
rejects any remaining worker reservation, and the PPO coordinator requires an
empty worker ledger after collection. Removing that ownership boundary would
need a separate design, not an uncharged cache or early fake lease release.

## Disposable healthy-path experiment

The bounded runner changes only the parent scheduling of the existing per-game
initialize calls. It retains original source bytes, child commands/environment/
stderr/protocol, physical ownership, budgets and original test assertions. A
simple healthy-path gather proves feasibility; it is not the accepted
cancellation/failure implementation. The separately reviewed task-ownership
recipe in ADR0028 must govern the actual helper.

Exact HOST command from repository root:

```sh
/usr/bin/time -v -o /tmp/godfieldbots-parallel-ready-probe-20261109/resource.txt timeout --signal=INT --kill-after=15s 240s .venv/bin/python /tmp/godfieldbots-parallel-ready-probe-20261109/profile_collection.py > /tmp/godfieldbots-parallel-ready-probe-20261109/pytest.log 2>&1
```

Result:2PASS, pytest76.74s. All192ready records/128reset ACKs and unchanged
source hashes validate. The same parent observer/aggregation retains explicit
wall-union and overlap combinations:

| Measured collection scope | Sequential readiness s | Concurrent readiness s |
| --- | ---: | ---: |
| Fresh MLP reference:32games/64physical processes |37.211198|26.501274|
| Pooled GRU:64games/16physical processes |23.064881|21.791269|
| Fresh-reference readiness union |35.306378|24.737570|
| Pooled-GRU readiness union |16.251377|13.795613|

These are single instrumented observations with preloaded parent modules and
wrapping overhead, not an ordinary whole-module or whole-suite speed claim.
The reused processes stay within one collection; no update overlaps participants.

Independent Root array checks compare actual original NPZ files without identity
normalization. Initial/final weights and saved optimizer/RNG state of the fresh
MLP reference plus GRU continuous/resumed parents agree exactly:916arrays across
six artifact directories. Evidence:
`/tmp/godfieldbots-parallel-ready-probe-20261109/initial-array-equivalence.json`.
This is numerical equivalence for the observed fixture, not a training-quality
or real Source qualification result.

## Implementation acceptance

The implementation is limited to `tests/structured_ppo_support.py`, the CPU
wrapper in `tests/test_structured_ppo.py`, and21new behavioral regressions in
`tests/test_parallel_structured_ready.py`. Normal helper defaults remain serial;
the CPU module opts in independently of physical pooling, so its cold reference
still creates fresh participant processes. No existing test body, budget or
assertion changes. Correctness and architecture conformance both report READY
with no blockers. The correctness reviewer independently runs20pure cases and
checks the pre-existing functions by AST.

Final same42case command:

```sh
PYTHONPATH=/tmp/godfieldbots-parallel-ready-20261109 UV_CACHE_DIR=/tmp/godfieldbots-uv-cache /usr/bin/time -v -o /tmp/godfieldbots-parallel-ready-20261109/target-resource.txt timeout --signal=INT --kill-after=15s 900s uv run --no-sync pytest tests/test_structured_ppo.py -p pytest_pool_timing_probe --basetemp=/tmp/godfieldbots-parallel-ready-20261109/pytest-temp --setup-show --durations=0 --durations-min=0 --junitxml=/tmp/godfieldbots-parallel-ready-20261109/target-junit.xml > /tmp/godfieldbots-parallel-ready-20261109/target.log 2>&1
```

PASS42; pytest231.35s, external231.79s. Baseline241.87s to231.79s saves10.08s,
4.17%. Original node IDs and every pass status match. Same observer/setup reporting
is retained; final uses an explicit durable basetemp and `uv --no-sync` with a
writable cache to avoid modifying the dependency environment. These are single
observations at different revisions, not a distribution or whole-suite timing.
The observer's only source difference is its output directory.

| Scope | Before s | After s |
| --- | ---: | ---: |
| Test setup |52.918159|43.628607|
| Test calls |188.298011|187.359121|
| Test teardown |0.027762|0.027499|
| Other external wall |0.626068|0.774773|
| External total |241.870000|231.790000|
| Shared MLP reference collection, nested |36.957220|27.387731|
| Pooled MLP pause/resume collection, nested |11.859032|11.193012|
| Pooled GRU comparison collection, nested |23.898174|22.920339|
| Pooled Transformer comparison collection, nested |24.013111|23.669627|

Final raw evidence is `/tmp/godfieldbots-parallel-ready-20261109`: timed log/XML/
resource file, `timings.json`, `target-comparison.json`, `final-workflows.json`
and `measurement-metadata.json`. All five snapshotted helper files and157source
modules retain exact measured bytes through final verification. Git confirms no
application/native/config/dependency changes. The older default pytest160parent
directories expired under pytest retention; their earlier aggregate timing and
source-hash evidence remains retained. Do not imply those original parents remain
available. The two-case diagnostic uses durable explicit basetemp directories.

Root's final retained-before proof compares those actual original MLP reference
and GRU continuous/resumed parents with the final42case target. All916G0/G2
weights/optimizer/RNG arrays retain exact dtype/shape/values. All384actual schema5
decisions have nonempty frame/table/selection/search/command, exact candidate
column bytes with verified SHA-256, search/RNG fields, selection index/logprob/
value/policy_version, frame content and command actions under a bounded comparison
projection. Only episode/command UUIDs and linked frame/table/prefix hashes are
excluded; history cursors compare their exact viewer/position suffix; each own
hand ID and its matching usable hint/action reference map to the original hand
position. Three measured search timers are excluded. No original artifact is
rewritten or relabeled. This avoids a missing-legacy-input comparison and does
not grant Source authority. Details are in `final-equivalence.json`. Transformer
is covered by the unchanged exact continuous/resume assertions and existing
full-wire pool regressions, not a separately retained-before artifact comparison.

Related command:

```sh
UV_CACHE_DIR=/tmp/godfieldbots-uv-cache /usr/bin/time -v -o /tmp/godfieldbots-parallel-ready-20261109/related-resource.txt timeout --signal=INT --kill-after=15s 360s uv run --no-sync pytest -q tests/test_parallel_structured_ready.py tests/test_pooled_structured_worker.py tests/test_structured_protocol.py tests/test_structured_environment.py tests/test_structured_source.py tests/test_structured_ppo_source.py tests/test_structured_ppo_cuda.py --junitxml=/tmp/godfieldbots-parallel-ready-20261109/related-junit.xml > /tmp/godfieldbots-parallel-ready-20261109/related.log 2>&1
```

PASS279, SKIP6 in107.93s. Breakdown: new readiness21, pool48, protocol86,
Environment20, Source74, PPO Source30. The6CUDA cases retain their opt-in skips;
no actual CUDA execution is claimed. The21new cases sum0.108s in this invocation's
JUnit data; that is an observation, not a new whole-suite wall estimate.

Additional PASS:

```sh
UV_CACHE_DIR=/tmp/godfieldbots-uv-cache uv run --no-sync ruff check src tests native
UV_CACHE_DIR=/tmp/godfieldbots-uv-cache uv run --no-sync mypy src
UV_CACHE_DIR=/tmp/godfieldbots-uv-cache timeout --signal=INT --kill-after=5s 60s uv run --no-sync pytest --collect-only -q
git diff --check
```

Mypy covers157source files; full collection has4301cases. Full execution, global
Ruff over unrelated untracked user scripts, wheel rebuild and actual CUDA are
NOT RUN for this tests-only increment. Root accepts after scoped evidence and
both reviews. No whole-suite elapsed/green claim follows. The preceding4032PASS/
247SKIP/1new-expectation-failure whole run and corrected48case rerun remain
separately recorded in ADR0027's evidence, not reclassified.
