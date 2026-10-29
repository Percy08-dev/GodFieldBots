# Bulk numeric I/O and episode lookup verification

Status: accepted2026-10-30. Independent design/correctness/conformance READY;
runtime/corpus/wheel/measurement gates and main rollout PASS. Overall current-
corpus speed is essentially unchanged. Baseline80b4911/release952657c7..., frozen Python source under
/tmp/gfb-bulk-io-reference. No simulator/numerical environment changes.

## Confirmed costs and scope

The existing files already use32KiB BufWriter/BufReader. Scalar helpers still do
one Rust write_all/read_exact and conversion/push per f64/i64. Therefore bulk I/O
is intended to remove call/element overhead; it does not remove one prior OS
syscall per value. Files remain little-endian, and all consumers retain the same
array order, dtype, shape, quota and NumPy owner contracts.

A collection with E episodes repeatedly scans episodes/schedule/group-health in
projection.links, searches each original ordinal with position, checks directory
membership with any, and validates each scheduled identity with nested scans.
These are O(E²) per collection. Formal PPO collections have16games, validations64;
this is not quadratic in the total6,784games. Expected per-collection indexes and
sets are standard maps with explicit duplicate/missing/health checks and original
ordinal preservation. Indexes borrow immutable manifests in local phases.

## Required verification

Typed bulk exact bytes/bits, empty/chunk boundaries, portable endian conversion,
short/partial failed I/O, control/quota checks and cursor publication. Manifest
index duplicate/missing/schedule/health/ordinal equivalence and linear-operation
count/scaling. Full frozen both-recipe NPZ/index and saved PPO ingestion under
2GiB/16MiB, relevant full tests/wheels, microbenchmarks plus3alternating complete
preparation pairs with CPU/RSS/PSS/scratch/I/O and no concurrent tests/builds.
No model fitting, collection, optimizer updates or final-test evaluation.

## Frozen implementation

Native source b479c64928172ef3004eddce9a610e9408e7ab8860c5a3959b7491de09b3eaa4;
release extension3b12d7afc7e4054d3b3ae1ea53f11059a9294204bec788b5f2c97b2750163322;
wheel02afe22bd984b02ffbfe1755a9b5d8955490cdcbcaf92fe5edf07684e75cc2a0;
Cargo.lock d2a6f8d2034864b0266341f383e108783fde02b8dd2666530eee5a2173c7bb55.
Bytemuck1.25.2 is the only new dependency; other pins/parser contract unchanged.

Only four private typed entry points; no handwritten unsafe or uninitialized
typed buffer. Upfront capacity admission is followed by bounded initialization
and read blocks; errors roll back append lengths. The Stage cursor advances only
after a complete valid batch. Checked shifts preserve candidate-offset semantics.
Portable conversion fallback is tested on the current host as well as the native
little-endian view. Shared helpers also replace scalar export_staging reads.

Local indexes borrow immutable manifest data and hold one coarse capacity lease.
Original ordinals and ordered vectors drive output. Same-ID repeats within one
health record and nonstring/foreign memberships preserve old any() semantics;
different records claiming the same actual episode still fail. The old helper
did not reject repeated foreign IDs, so they are filtered before insertion.
Missing/duplicate/schedule/result/health/source checks stay authoritative.

## Correctness and bounded microbenchmarks

PASS: whole Python769tests,5opt-in skipped,249.38s; Rustdebug49/releasefocused35
(two microbenchmarks ignored in those runs), focused Python94; whole Ruff/mypy59
source files, Rustfmt/diff checks. The two release microbenchmarks were executed
explicitly and pass separately. Independent source reviews READY; no blockers.

Sequential synthetic helper observations:

| Release helper microbenchmark | Scalar/scans | Bulk/index |
| --- | ---: | ---: |
| 262,144numeric values x5, buffered I/O,s |0.008293177|0.000833128|
| 1,024episodes x3, lookup including index construction,s |0.072275714|0.001174116|

These are bounded local observations, not end-to-end speedup claims. Baseline I/O
was already buffered. The real source uses smaller16/64-episode collections.
Tests also count block calls / mapped entries without flaky timing thresholds.

`runs/assessments/bulk-io-corpus.json`: both complete83,783-row numeric recipes
retain original NPZ hashes c5c3f3c1... /841c6f885... and dataset ab5d4913....
Index content574b81ec... and Parquet bytes9bce50db... are identical under original
2GiB/16MiB settings. Correctness runs overlapped tests, so their times are not
performance evidence. `runs/assessments/bulk-io-ppo-ingestion.json` confirms the
old saved16-game features/GAE/targets7e2629b8... with1/2/4workers; no optimizer ran.

Separate installed native release wheel/CPU MLP+LightGBM inference checks PASS;
the existing application wheel's59Python files still byte-match current source.
Optional-free installed baseline help/config PASS. Main old extension is retained
until all paired old/new trials finish. No Python dependency versions changed.

## Complete preparation and interpretation

Same frozen83,783-row history preparation,2GiB/16MiB,workers1/inflight1, release
builds/dependencies, one warmup each then3alternating measured pairs. No concurrent
builds/tests/training. Every output dataset/sample/array/NPZ identity matches.
Records: runs/performance/bulk-io-20261030/{results,equivalence}.json and
runs/assessments/bulk-io-performance.json.

| Whole workload / resource metric | Before | Bulk/index |
| --- | ---: | ---: |
| Wall seconds,3trials |47.425 /46.226 /46.432|45.992 /46.053 /46.121|
| Median wall seconds |46.432|46.053|
| Median process-tree CPU seconds |46.338|45.928|
| Median sampled per-run peak RSS,bytes |424,144,896|424,017,920|
| Median sampled per-run peak PSS,bytes |400,083,968|408,123,392|
| Internal tracked scratch peak,bytes |1,856,048,056|1,856,048,056|
| Kernel-accounted read-byte lower bound |0|0|
| Median sampled write-byte lower bound |1,943,031,808|1,943,031,808|

Observed median difference0.818%, ratio1.00825x. This is below1% with small run
variation; it does NOT demonstrate a substantial/generalized whole-preparation
speedup. Local buffering was already present, and E is16/64 in actual collections,
so removing per-scalar calls and quadratic lookups has limited end-to-end impact
here. Benefits are fewer Rust call units, exact bit-preserving bulk operations
and expected-linear collection lookup scaling (O(E+schedule+health-memberships)
construction/expected-constant query), not a claim that the main bottleneck is gone.

OS RSS/PSS/storage are sampled peaks and may miss brief events; PSS and scratch
sample timing differ between runs. Output includes scratch; do not add both.
Read bytes are warm-cache OS accounting, not WSL host physical disk traffic.
About one CPU core occupied; GPU unused. Preserve micro and whole timings as
separate evidence, rather than infer overall gains from the synthetic ratios.

After paired runs the verified wheel was installed into main Bots with --no-deps;
actual loaded release SHA rechecked. Prior952657c7... native package is archived
under the frozen reference's native-site. Original six fit artifact groups retain
all24file identities/small-file hashes; original arrays/models unchanged. Simulator
remains untouched/clean. No live collection, optimizer update or refit occurred.
