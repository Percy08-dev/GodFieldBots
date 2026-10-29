# Native preprocessing verification

Historical initial custom-parser implementation evidence. The production Serde
migration supersedes its JSON resource/parser implementation; current measurements
and audit are in [Serde adoption](serde-json-adoption.md), governed by ADR0012.

Design: [ADR0011](../adr/0011-native-preprocessing.md). Implementation at
`01665840720064050ffc54d2855a67dce4ec8c82`; final acceptance on 2026-10-30.
Independent correctness and implementation-conformance reviews are READY.
The six completed comparison fits and their frozen data remain unchanged.
This unit covers offline build/inspect/index/fixed-data preparation and PPO
post-collection ingestion. Live participant inference remains Python.

## Runtime and dependency identity

CPython3.14.9/GIL, NumPy2.5.4, Rust1.99.0, Maturin1.15.0, PyO3 0.30.1,
rust-numpy0.30.0 and ndarray0.17.2. Root retains uv_build; the optional native
workspace distribution is built separately. No Simulator source/build/venv changes.
Simulator revision `ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6`, catalog `public-ja-v2`.

- Reviewed native source SHA: `347104771a3a1a9c83da56c2f21de44eacc2512e06e653a7dfeedbd2fcd1c2f9`.
- Loaded extension SHA: `1a0b1746b5582df0f2403230a6e406efadd28a64c3babbd354dfa20437e7f34f`.
- Native wheel SHA: `920d0242e7b573b4a0934379000fcb2849cc506acf92a3ed9ca59a5ffac7daa9`.
- Cargo.lock SHA: `4c46379f389198c74d9e79ac082f8b2025c6d7058fd3b549722370a8ee8555f9`.
- Application wheel SHA: `280a7624b13e606524472a97b7913790e9fe7bdd9329c0bd68ba644f8a06101b`;
  all59Python files byte-match the reviewed implementation. Later rollout changes
  are experiment configuration/documentation and an explicit Python test fixture.

Separate installed baseline help/config PASS without native/Torch/LightGBM/PyArrow.
Separate installed native features equal Python and have readonly FloatOwner bases;
installed CPU MLP/LightGBM inference PASS. Dependency files were reused from verified
local environments, not freshly resolved from remote indexes.

## Correctness and safety checks

Final native-enabled `pytest -q`:751PASS,5opt-in skipped,252.81s.
`ruff check .`, `mypy src`:PASS (59source modules). Rust31tests PASS.
Final preset-rollout checks:28configuration and19PPO lineage tests PASS;
synthetic/reference fixtures explicitly retain Python. Both independent rollout
delta reviews are READY. No learning/native source changed after the full suite.
Additional independent60feature/DTO cases,92mutations and29,983random finite-float
canonical comparisons PASS. Reviewed deltas close strict source boolean mode,
explicit fit work/wall deadlines and control metadata admission/lifetime findings.

Checks cover journal/Replay/source reconciliation, failed-parent rejection,
actor visibility, exact candidate order, same-policy learner seats, both-view
reward validation, Python-owned labels/GAE/global advantage normalization,
byte-identical NPZ/shards, readonly pointer sharing and retained slice leases,
shared parallel budgets, stale sources, cancellation/deadlines, finalization failure,
explicit depth policy, small-stack Core retirement/GC and baseline optional imports.
The physical journal directory must match its header/manifest episode; accepting
cross-directory content swaps was explicitly tightened in both readers. Viewer
identity remains header-based and journal-only viewer swaps remain supported.

Sandboxed asyncio thread notifications fail because self-pipe socket send is denied
with EPERM. Independent diagnosis reproduced the environment restriction. Tests
requiring those notifications and live CUDA/listeners use approved local execution;
no application polling workaround was added. Cancellation joins owned work before
releasing authority, including completion races and repeated cancellation.

Control memory is pre-admitted as64*(S+G)+32MiB before control-summary handoff;
G bounds authorization. At first data/proof publication it becomes64*(S+P)+32MiB,
and final decoding reserves64*(S+2P)+32MiB through Core retirement until explicit
release. These are conservative managed reservations, not exact Python object
sizes or RSS quotas. Native buffers, reallocation overlap, retained NumPy exports
and declared Python numerical workspace share the configured byte budget.

## Full frozen-corpus equivalence

`runs/assessments/native-final-corpus-equivalence.json` records both83,783-row
recipes at2GiB/16MiB, all original labels/sample ordering and exact NPZ bytes:

- History NPZ: `c5c3f3c1ef4068766d81c7bd0cfa7a8957b140ad7ea1f207374fc725dcbcc94d`.
- Observation NPZ: `841c6f885e8d8f052abde2c5db788de761279947e36300e7bc87ff0254e7d0c8`.
- Dataset digest: `ab5d4913fda45795d1cd5aa02a4a9c4c3464268298c4d3ea49d56d4909df770f`.
- Sample identity: `cbc11f1bc2810cb7a7536c07f42c64893f92693056dc1e90dbf41f9773676151`.
- Rebuilt Parquet83,783rows, row-content SHA:
  `574b81ecf6178849c91bea20a9bc0d9906a0348f2c5357825140fc6fc8c618cf`.

The native index projects/orders/hashes/spools rows in Rust; Python passes typed
numeric/UTF8 buffers to Arrow and writes Parquet. No Python row-dictionary rebuild.
Original dataset/indexes are unchanged. Incidental correctness-run timings with
concurrent tests are excluded from speed acceptance.

## Paired complete preparation benchmark

Reference: clean `42ce5e6`, archived at `/tmp/gfb-native-reference-42ce5e6`.
Same frozen history data, interpreter/dependencies,2GiB/16MiB, workers1/inflight1,
OMP/MKL/OpenBLAS threads1. One warmup per backend then three alternating measured
pairs; no concurrent training/tests. Records:
`runs/performance/native-20261030-history/{results,equivalence}.json` and
`runs/assessments/native-paired-preparation-summary.json`.

| Complete command | Optimized Python | Rust |
| --- | ---: | ---: |
| Wall seconds, three trials |252.239 /256.486 /252.822|94.560 /93.707 /94.261|
| Median wall seconds |252.822|94.261|
| Median process-tree CPU seconds |252.792|94.148|
| Median sampled per-run peak RSS, MiB |454.34|376.66|
| Median sampled per-run peak PSS, MiB |426.80|366.81|
| Maximum sampled RSS, MiB |454.58|376.76|
| Maximum sampled scratch, MiB |890.00|1770.29|
| Median sampled kernel read bytes |0|0|
| Median sampled kernel write bytes, MiB |890.08|1853.04|
| Accepted actions per wall second |331.39|888.84|
| Amortized pipeline ms per accepted action |3.018|1.125|

Median speedup **2.682x**, wall reduction62.72%; the>=1.5x gate passes.
The per-action figure covers complete preprocessing of83,783accepted actions,
including source validation; it is not live policy latency. Exact output identity
passes for every trial, all four arrays and the numeric archive hash.

CPU time approximately equals elapsed time: one core stays busy (about6.25% of16
logical CPUs). This is a CPU workload and uses no GPU. Kernel read_bytes=0 with
logical reads means the warm source data were served from page cache; it does not
mean input was skipped. OS I/O counters do not expose WSL host physical traffic.
RSS/PSS/scratch are sampled, so short peaks may be missed; storage output includes
scratch and should not be added to it. The observer is excluded from workload CPU.

Example native trial worker phases, seconds: read/hash2.935, JSON/owned-tree decode62.972,
journal input validation2.951, Replay validation0.081, feature encoding6.715,
proof/staging4.371. These are instrumented worker phase totals, not a complete
additive partition of command wall; global controls/finalization/teardown also cost
time. `next_batch_inclusive` and `global_scan_and_encode` overlap these phases and
must not be added. JSON decoding and owned-tree construction are now the largest
measured phase. The [decoder breakdown](native-decode-profile.md) describes finer
diagnostic measurement; semantic DTO checks have their own validation timers.

Boundary evidence:56returned batches,57data calls including one explicit EOF;
784pointer-verified shared numerical exports (929,982,696exposed bytes),0numeric
FFI copy events/bytes. This is not end-to-end zero-copy. Controls copy1,751,865bytes
out;2,775,858input bytes are borrowed. Internal managed reallocation copies total
7,604,227,970bytes and extend_from_slice copies3,889,510,329bytes. Rust's raw
feature/state spools coexist with the compatible Python NPZ writer; scratch and
write traffic increase. This measured tradeoff is accepted under the8GiB scratch
budget. Revisit direct owned-buffer reuse/spooling if decode/copy/I/O becomes the
next practical bottleneck; preserve exact data and learning ownership.

Representative real16-game data, same2GiB/16MiB shared budget, warmup+3trials:
workers1/2/4 median0.2854/0.2691/0.2840s with identical output hashes. Four workers
increase CPU with little elapsed benefit. Keep default1; explicit1/2/4 remain
available. Record:`runs/assessments/native-worker-comparison.json`.

## Real CUDA and reference-fit acceptance

`runs/assessments/native-ppo-smoke-equivalence.json`: continuous2updates and
pause-after1/resume-to2 both complete, each32healthy natural games,379accepted
learner actions,16optimizer steps. Original7200/5400/7140second limits retained;
only max_updates=2 scopes this smoke. Wall54.226s continuous /56.586s resumed,
collection50.208/50.358s, update response1.509/1.678s. Final weights, Adam, CPU/CUDA
and shuffle RNG arrays and resume JSON are byte-identical; schedule RNG/pool and
both batch numerical hashes match. Largest CPU/GPU parity error5.96e-7 under
atol=rtol=1e-5. GPU peak allocated92,533,760 /reserved109,051,904bytes.
This is functional smoke evidence, not a training-quality or throughput benchmark.

The continuous run's live CPU MLP policy.select samples (565calls, after candidate
generation) have p50=1.521ms/p95=11.254ms. Candidate search across all policies
(746actions) has p50=0.502ms/p95=28.566ms. The32games contain746total accepted
actions;379are learner actions. These distributions include real source shapes
and warmup effects and are not an isolated device benchmark. Final numerical
weight archive681,538bytes; learner process-lifetime peak RSS2,221,092,864bytes
includes the Torch/CUDA runtime. The GPU is active only during the short updates;
collection accounts for50.208of54.226continuous-run seconds. The earlier
preprocessing1.125ms/action figure measures a separate offline operation.

One native history/seed17 reference fit completes with original7200s budget:
93.132s charged preprocessing/control,5.245s fit/save/reap,98.377s charged total;
98.567s observer wall /99.260s process-tree CPU. Input NPZ, fit request, model and
inference manifest SHA match the existing seed17 reference exactly. Best iteration95,
validation L2=0.59091264247396; model SHA
`2039c4f31f23d06999cc2f1ce3ddcf6846d1dfdca2c163d2596a59d739f7336f`.
Sampled peak RSS1,074,749,440 /PSS1,047,697,408bytes, output1,856,759,889bytes
including scratch. Complete-fit memory includes the LightGBM child and is distinct
from preparation-only memory. Record:
`runs/assessments/native-reference-fit-equivalence.json`. This is an equivalence
repeat, not an additional comparison cell or a paired fit-speed benchmark.

An initial sandboxed reference-fit attempt was interrupted before fitting because
asyncio self-notification was blocked. Its observer record remains in
`runs/native-reference-fit-20261030`; it is excluded from successful measurements.
The successful approved-local repeat is `runs/native-reference-fit-20261030-live`.

`runs/assessments/native-original-six-preservation.json`: all24recorded original
files retain size/mtime/inode; small training/model/inference files also retain
SHA. Large numeric archives were verified against original hashes in the corpus
gate; their original files were not rewritten. No new formal PPO campaign or
final-test evaluation ran. Shared historical generation cost remains reported
once, separately from each fixed-data fit.

## Reproduction and rollout

```sh
env -u VIRTUAL_ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/benchmark_workload.py paired --kind preparation \
  --comparison native --encoder history --workers 1 --max-in-flight 1 \
  --output runs/performance/native-new-measurement --seconds 1800 --execute
```

The CUDA/CPU PPO and two fixed-LightGBM experiment presets select native after
acceptance. Standalone dataset commands select it explicitly with
`--preprocessing-config configs/preprocessing/native.toml` or the measured
`native-low-memory.toml`. Library/default reference settings remain Python for
compatibility; native failures never fall back. Missing native dependencies fail
preflight. Backend/build identity changes reject ordinary PPO resume.
Optional standalone dataset RLIMIT_AS is explicit, virtual-address-space scoped,
and never inherited into training/CUDA. Budget configuration is retained.
