# Lazy CPU deterministic runtime — 2026-11-09

Status: accepted; both independent reviews READY and required scoped/full checks
PASS. No simulator or learning-quality acceptance is implied.

[ADR0029](../adr/0029-lazy-cpu-deterministic-runtime.md) adopts CPU-only compiler
initialization deferral in the existing shared Torch runtime. CUDA retains its
original public setting API. Application changes are limited to
`src/gf_models/torch_runtime.py`; four new subprocess regression cases live in
`tests/test_torch_runtime_lazy.py`. Callers, versions, schemas, NUMERICS,
dependencies, process ownership, budgets and existing assertions are unchanged.
Measured parent revision is79b1c8d with frozen candidate worktree bytes; original
artifact producer metadata is not rewritten after the final local commit.

## Determinism and import completion

Supported Torch2.14.1 config consumes persistent process environment
`TORCHINDUCTOR_DETERMINISTIC=1`. CPU setup uses public debug-error mode and, when
config is present, joins its actual normal import before setting its caller-context
deterministic override. Absent config remains unimported until needed. Runtime/
device checks retain early refusal; failed-import sentinels still raise. Original
IEEE/backend/thread/core settings remain. Focused subprocess checks PASS4in5.18s:
optional imports, fresh eager/late compiler behavior, repeated/preloaded/conflicting
flags, guard failures, original CUDA dispatch and actual in-progress import.

The first partial-import test incorrectly required cross-thread propagation of a
ContextVar override. An actual original-API counterpart proves helperTrue/mainFalse
under the same barrier. The corrected test checks True in the caller, real module
identity/completion and global core flags; production logic was not changed to
force propagation. Failed and counterpart evidence is preserved in
`implementation-in-progress-initial-failure.{py,log}`,
`implementation-original-api-context.log` and
`/tmp/godfieldbots-original-api-context-20261109.py`.

New late-import defaults can differ in unrelated fresh contexts. Supported model/
learner loops initialize and use Torch in their owned context. Actual compiled
kernels, unrelated threaded compiler configuration and a separately installed
CPU-only Torch wheel are NOT RUN. CPU execution uses2.14.1+cu132. Revisit the
pinned environment/config assumption before a new compiled/threaded consumer or
Torch upgrade; do not infer a global cross-context equivalence guarantee.

## Actual worker startup and first decisions

The original three CPU neural protocol tests run with parent timing wrappers and
child `-X importtime`. Baseline is the actual old157module source-matched wheel
in an isolated interpreter with the same dependency directory, not FakeTorch.
Both runs PASS3original cases; every child makes two original decisions.

| Model | Before startup s | After startup s | Before first request s | After first request s |
| --- | ---: | ---: | ---: | ---: |
| MLP |1.531286|0.912671|0.011837|0.011696|
| GRU |1.438814|0.852908|0.011542|0.011493|
| Transformer |1.288031|0.792558|0.011667|0.011293|

Compiler modules are absent at candidate readiness and after BOTH decisions for
all three models. Actual columns/search RNG/index/logprob/value agree exactly.
Inference manifests differ only at producer metadata, so policy IDs correctly
differ; each test validates its own binding. This diagnostic includes tracing
overhead and is not the module benchmark. See `probe_actual_workers.py`,
`worker-{before,after}/summary.json`, `actual-worker-comparison.json` and
`worker-policy-identity-differences.json`. An initial observer-only tuple `.tolist()`
error failed3probe cases; corrected serialization preserves original operations.
The failed `worker-before-v1/` logs/script remain retained.

## Same42case module timing

CPU command from repository root:

```sh
PYTHONPATH=/tmp/godfieldbots-lazy-cpu-runtime-20261109 UV_CACHE_DIR=/tmp/godfieldbots-uv-cache /usr/bin/time -v -o /tmp/godfieldbots-lazy-cpu-runtime-20261109/target-resource.txt timeout --signal=INT --kill-after=15s 900s uv run --no-sync pytest tests/test_structured_ppo.py -p pytest_pool_timing_probe --basetemp=/tmp/godfieldbots-lazy-cpu-runtime-20261109/pytest-temp --setup-show --durations=0 --durations-min=0 --junitxml=/tmp/godfieldbots-lazy-cpu-runtime-20261109/target-junit.xml > /tmp/godfieldbots-lazy-cpu-runtime-20261109/target.log 2>&1
```

PASS42; pytest195.42s, external195.84s versus231.79s:35.95s/15.51% saved in single
observations. Same42nodes/statuses, observer/setup/duration reporting; temporary/
output directories differ. Setup32.340765s, call162.705058s, teardown0.028088s,
other external0.766089s. No variance/distribution claim is made.

The same10comparison collections retain320logical participants/40real processes
and original reset/reap. Owner-only sampled peak summed PSS1828.419→1439.592MiB;
RSS2897.520→2279.816MiB. These omit learner/coordinator; RSS duplicates shared pages
and samples can miss transients. They are not whole-system memory quotas or a
repeated study. Receipts are compared in `target-owner-memory.json`.

## Actual state, wire and producer preservation

Six before/candidate normal parents (shared MLP, MLP resumed, GRU continuous/
resumed, Transformer continuous/resumed) retain exact dtype/shape/value equality
for2112actual G0/G2weight/optimizer/RNG arrays across12artifact directories.
Before originals are the durable prior target under
`/tmp/godfieldbots-parallel-ready-20261109/pytest-temp`; no old parent is relabeled
or resumed under changed code. See `saved-array-equivalence.json`.

All768actual schema5decisions have nonempty frame/table/selection/search/command.
Candidate bytes/hashes, search RNG/fields, numeric index/logprob/value, frame
content and actions agree under the bounded comparison projection in
`compare_wire.py` and `wire-equivalence.json`: omit opaque episode/command UUIDs
and linked frame/table/prefix hashes; compare exact cursor viewer/position suffix;
map each visible own-hand ID and matching usable hint/action only to its original
hand position; omit3measured search timers. No missing legacy input is compared.

All18G0/G1/G2inference manifests differ ONLY in producer metadata. Each actual
policy label and checkpoint SHA is independently validated against its own
manifest:576neural identities correctly change with producer bytes;192baseline
identities remain. They are not silently rewritten or claimed byte-equal.
Formats/NUMERICS and active-resume code/runtime refusal remain unchanged. Full
compatibility/regression checks pass. The earlier five full historical-corpus
receipts remain historical, not newly rerun proofs; no reader/parser/schema
change requires a corpus migration.

## Actual CUDA and package checks

HOST GPU command:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 GFB_RUN_STRUCTURED_PROTOCOL_CUDA=1 GFB_RUN_STRUCTURED_NUMERIC_CUDA=1 UV_CACHE_DIR=/tmp/godfieldbots-uv-cache /usr/bin/time -v -o /tmp/godfieldbots-lazy-cpu-runtime-20261109/cuda-resource.txt timeout --signal=INT --kill-after=15s 360s uv run --no-sync pytest -q tests/test_structured_protocol_cuda.py tests/test_structured_numeric_worker_cuda.py --junitxml=/tmp/godfieldbots-lazy-cpu-runtime-20261109/cuda-junit.xml > /tmp/godfieldbots-lazy-cpu-runtime-20261109/cuda.log 2>&1
```

PASS6in43.46s: actual MLP/GRU/Transformer inference and numerical two-update/save/
restore. A separate same-process CPU→CUDA check preserves core determinism/warn/
debug2/thread/IEEE/cuDNN flags and exact small GPU/CPU matrix results. Hardware is
RTX5070Ti, CUDA13.2, driver API13030 (not exact driver-build identity). HOST GPU is
available while the ordinary sandbox hides it; no fallback is reported as a pass.

PASS `uv build --wheel`: all157source modules match byte-for-byte, no tests
packaged. Installed in a separate mandatory-only Python3.14environment without
Torch/LightGBM/PyArrow/native/pytest, normal structured Random worker startup/real
reap/SAME empty ledger and `gf-bots --help` outside the repository PASS. Evidence:
`wheel-build.log`, `wheel-correspondence.json`, `baseline-{install,smoke,help}.log`.

## Whole invocation and acceptance

Single frozen-source HOST command:

```sh
UV_CACHE_DIR=/tmp/godfieldbots-uv-cache /usr/bin/time -v -o /tmp/godfieldbots-lazy-cpu-runtime-20261109/full-resource.txt timeout --signal=INT --kill-after=15s 2400s uv run --no-sync pytest --basetemp=/tmp/godfieldbots-lazy-cpu-runtime-20261109/full-pytest-temp --setup-show --durations=0 --durations-min=0 --junitxml=/tmp/godfieldbots-lazy-cpu-runtime-20261109/full-junit.xml > /tmp/godfieldbots-lazy-cpu-runtime-20261109/full.log 2>&1
```

PASS **4058passed/247skipped/4warnings/zero failures**,4305cases, pytest1176.98s,
external1177.96s (19:37.96), exit0. All4217initial pass/skip statuses remain;
report joining normalizes only one generated HTTP UUIDv1parameter. The preceding
4280case run's sole status change is its already-corrected deadline-test
expectation failure, nowPASS. Its failed log stays historical. Historical full
1390.16s→1177.96s saves212.20s/15.26%; initial1738.71s→1177.96s saves560.75s/32.25%.
Whole runs have different added tests/reporting/revisions: these are reference
comparisons, not a matched effect isolated to this runtime change.

| Whole-run module including setup/teardown | Earlier pool full s | Latest full s |
| --- | ---: | ---: |
| `test_structured_tree_dataset.py` |286.866|195.655|
| `test_structured_ppo.py` |238.872|191.108|
| `test_native_structured_session.py` |170.653|118.689|
| `test_pooled_structured_worker.py` |76.348|57.782|

PASS `UV_CACHE_DIR=/tmp/godfieldbots-uv-cache uv run --no-sync ruff check src tests native`;
PASS analogous `mypy src` for157files; PASS `git diff --check`. Scoped Ruff omits
unrelated untracked user scripts. Both final source reviews are READY. Independent
audits check raw counts, hashes, scope and actual identities without rerunning
tests. Root accepts after the successful full invocation and frozen-byte checks.
No formal learning, qualified Source, strength or future-context/compiled-kernel
acceptance follows.

All evidence is under `/tmp/godfieldbots-lazy-cpu-runtime-20261109`, including
`before-metadata.json`, baseline wheel correspondence, `measurement-metadata.json`,
`full-freeze.json`, logs/XML/resources, `full-acceptance.json` and the proofs named
above. Application/test source hashes match measured bytes through acceptance.
