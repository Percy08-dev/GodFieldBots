# Pytest preloaded CPU launch — 2026-11-09

Status: accepted; both independent reviews READY and scoped/full verification
PASS. The matched PPO target improves; total default pytest increases with the
54added lifecycle regressions. No whole-suite speedup is claimed for this increment.

[ADR0030](../adr/0030-pytest-preloaded-cpu-launch.md) implements an explicitly
opted-in Linux CPU pytest launcher. A clean-exec `-E -S` template imports Torch
once, then directly forks fresh actors and records their exact wait statuses.
Model/options/frames stay in original direct pipes. Template never holds a model,
optimizer, game/private policy state or application test imports. Existing pool
owners still die before updates; template alone persists in the module scope.

## Boundaries, reference and failure ownership

The only application change is `WorkerReservation.creation_pending=False` and
its guarded transitions. Normal Popen launch/CUDA/native/default/source helpers
retain existing behavior. Private test files own the control protocol, actor
admission stream, server, parent service/mixin and lazy module fixture. Both normal
initial packet and EVERY pooled begin check fixed CPU factory/model/binding before
the original factory import. All original bootstrap/generator/selection semantics,
game/update budgets and assertions remain.

The module fixture binds an unstarted owner. Only eligible CPU collection starts
it before server/game/actor activity; pure guards and incompatible configs do not
warm it. Eligibility is exact actions2/known engineering contract, Python backend/
structured-v1 reader and supported CPU model/owner context. Valid beam/cap changes
are not artificially prohibited. Explicit incompatible preload fails early.

The shared MLP reference is now explicitly fresh-FORKED, not cold-exec. Each model/
episode state remains new. Original three-model cold Popen protocol and old pool
wire/reset tests retain independent cold coverage. New normal/fork/pool A-B-A and
checkpoint comparisons supplement it. Retained cold full parents are compared
below without relabeling or updating them under changed code.

Pending ownership exists before any external send; uncertain creation cannot use
the base `process=None` cancellation path. Actual PID/wait receipts alone permit
ordinary release after I/O retirement. Lost status poisons/quarantines the parent,
holds unresolved leases and blocks update/publication even when physical group
containment succeeds. Containment and confirmed actor reap are distinct outcomes.

Template wait ownership uses WNOWAIT until group signalling is finished. Known
reap is recorded even if real Popen.wait re-raises after reaping; no later PID/group
signal is allowed. One retained close preserves its original error/absolute
deadline under repeated cancellation. The auxiliary sampled RSS soft ceiling is
1GiB, independent of actor/CUDA/managed tensor reservations; it is not a hard
whole-process memory guarantee. Unsupported platforms, compiled/CUDA preloading,
arbitrary factories, descendant/group escape and unresponsive OS are not supported.

## Design and review evidence

The advisor resolved the consequential architecture choice in favor of the
clean direct-fork template, not the experimental multiprocessing facade. Six
HOST containment barriers PASS in4.574s under
`/tmp/godfieldbots-template-containment-20261109`: before fork, after fork/before
registration, post-registration/pre-ACK, post-wait/pre-status, lost ACK/repeated
cancel, and actual known exit23. Actor audit PIDs were read only AFTER cleanup,
never used to select signals or authorize release. Exact-status and contained-
status-lost/unconfirmed states stay separate.

Both independent final code reviews are READY. Concrete findings were corrected
with regressions before Root timing: fresh publication's closed receipt; actual
nonzero template exit after closeACK; all startup FD ownership/cached failure;
post-attestation and post-reply deadlines; no-child cleanup original error/owner/
idempotence; health-timeout poisoning; pre-Torch import-path purity; exact early
eligibility; and real wait-reap-then-error without signalling a recycled PID.

Historical checks/failures are preserved in
`/tmp/godfieldbots-cpu-preload-focused-20261109`: early stdin-wrapper lifetime
smoke failure (fixed by retaining/restoring its TextIOWrapper); first fault12P5F
in tool transcript session48805 (no disk stdout was created);10P2teardown-error
delta; later26P/29P/32P frozen logs; publication/FD3P; A-H43P48.06s; final real-
wait-exception6P5.17s. Later bytes are not falsely claimed tested by earlier runs.
Abort close(timeout=0) with actual known actor exit7 may release that physically
reaped lease but fail retirement bookkeeping and poison the service. No extra
normal deadline is granted to make this already-failed parent healthy.

## Same42case benchmark and module reuse

Root HOST command from repository root:

```sh
PYTHONPATH=/tmp/godfieldbots-preloaded-cpu-integration-20261109 UV_CACHE_DIR=/tmp/godfieldbots-uv-cache /usr/bin/time -v -o /tmp/godfieldbots-preloaded-cpu-integration-20261109/target-resource.txt timeout --signal=INT --kill-after=15s 900s uv run --no-sync pytest tests/test_structured_ppo.py -p pytest_pool_timing_probe --basetemp=/tmp/godfieldbots-preloaded-cpu-integration-20261109/pytest-temp --setup-show --durations=0 --durations-min=0 --junitxml=/tmp/godfieldbots-preloaded-cpu-integration-20261109/target-junit.xml > /tmp/godfieldbots-preloaded-cpu-integration-20261109/target.log 2>&1
```

PASS42; pytest160.73s, external161.16s versus195.84s:34.68s/17.7083% reduction
in single observations. Same42nodes/statuses, original workload/assertions and
observer/setup/duration reporting. Different temporary paths/revisions; no
variance claim. Setup19.015869s, calls141.288778s, teardown0.247406s and other
external0.607947s. First collector warmup and final module template close are
included, not a free cross-run cache.

`template-reuse-audit.json` confirms the13selected normal collections share ONE
template PID, and each returns closed/healthy with zero unretired tickets. The
template RSS sampled peak is528.520MiB, below its1GiBsoft allowance. Launch counters
are cumulative module counters, not per-batch values to sum. This excludes learner
and independent normal-cold imports; it is not a whole-memory or exact whole-suite
Torch-import-count claim.

## Numerical, wire and normal-path checks

All2112actual G0/G2weight/optimizer/RNG arrays across six normal before/candidate
parents and12artifact directories are exact in dtype/shape/value for MLP/GRU/
Transformer. Before sources are the actual normal-cold target in
`/tmp/godfieldbots-lazy-cpu-runtime-20261109/pytest-temp`. They are never rewritten,
requalified or resumed under changed code. See `array-equivalence.json`.

All768actual schema5decision records have nonempty frame/table/selection/search/
command. Actual candidate files/hashes, search RNG, numeric choice/logprob/value,
frame/action content and own policy/checkpoint identity compare exactly. All18
inference manifests and768policy identities are unchanged (mathematical producer
source did not change). The bounded comparison-only projection removes opaque
episode/command IDs and linked hashes, compares cursor viewer/position suffixes,
maps visible own-hand IDs plus matching hints/actions only to original positions,
and omits3search timing fields. It never compares missing legacy inputs or
relabels an original identity. See `compare_wire.py`/`wire-equivalence.json`.

Related HOST command:

```sh
UV_CACHE_DIR=/tmp/godfieldbots-uv-cache timeout --signal=INT --kill-after=15s 480s uv run --no-sync pytest -q tests/test_cpu_preload.py tests/test_cpu_preload_review.py tests/test_pooled_structured_worker.py tests/test_parallel_structured_ready.py tests/test_structured_protocol.py tests/test_structured_environment.py tests/test_structured_source.py tests/test_structured_ppo_source.py tests/test_structured_ppo_cuda.py --junitxml=/tmp/godfieldbots-preloaded-cpu-integration-20261109/related-junit.xml
```

PASS333/SKIP6in146.06s. Explicit existing actualCUDA protocol/numerical opt-ins
`GFB_RUN_STRUCTURED_PROTOCOL_CUDA=1 GFB_RUN_STRUCTURED_NUMERIC_CUDA=1` with
`CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0` PASS6in43.86s on HOST.
No preload mode is used for CUDA. No Source qualification or model quality follows.

## Whole invocation: success with an explicit test-cost tradeoff

Root frozen-source HOST command:

```sh
UV_CACHE_DIR=/tmp/godfieldbots-uv-cache /usr/bin/time -v -o /tmp/godfieldbots-preloaded-cpu-integration-20261109/full-resource.txt timeout --signal=INT --kill-after=15s 2400s uv run --no-sync pytest --basetemp=/tmp/godfieldbots-preloaded-cpu-integration-20261109/full-pytest-temp --setup-show --durations=0 --durations-min=0 --junitxml=/tmp/godfieldbots-preloaded-cpu-integration-20261109/full-junit.xml > /tmp/godfieldbots-preloaded-cpu-integration-20261109/full.log 2>&1
```

PASS **4112passed/247skipped/4warnings/zero failures**,4359cases, pytest1204.42s,
external1205.43s (20:05.43), exit0. All previous4305pass/skip statuses remain;
report joining normalizes only the generated HTTP UUIDv1parameter display.

The full invocation INCREASES27.47s from1177.96s, because54new tests add59.768
JUnit case-seconds. Common4305case sums fall1173.980→1141.793s (32.187s saved).
These sums exclude runner overhead and are single full invocations, not the
matched standalone benchmark. Do not market this increment as a faster entire
default pytest suite. Current whole-file PPO sum191.108→159.742s; the new preload
modules contribute44.338s plus15.430s. Critical lifecycle coverage is intentionally
retained rather than hidden behind new opt-ins or shortened budgets.

PASS scoped `ruff check src tests native`, `mypy src`157files and diff checks.
PASS application wheel157source modules byte-exact/no tests packaged. Installed
mandatory-only Python3.14without Torch/LightGBM/PyArrow/native/pytest PASS normal
structured Random worker init/actual reap/SAME empty ledger and CLI help outside
the repository. Application imports never depend on the test launcher.

All evidence is under `/tmp/godfieldbots-preloaded-cpu-integration-20261109`:
baseline snapshots, `measurement-metadata.json`, `full-freeze.json`, raw logs/
XML/resources, package correspondence and proof files. All157application and
9scoped test hashes match measured bytes through full acceptance. Independent
audits verify those hashes/counts/identities/scopes without rerunning tests.
Root accepts after both final reviews and all checks. Unsupported platforms,
CPU-only Torch distribution, arbitrary compiler/threaded consumers and formal
Source/learning acceptance remain unclaimed. Existing numerical and compatibility
versions and active-resume code-identity refusal policy are unchanged.
