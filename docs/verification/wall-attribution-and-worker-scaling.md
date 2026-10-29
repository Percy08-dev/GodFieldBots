# Reconciled wall attribution and decoder-worker scaling

Status: accepted 2026-10-30 under [ADR0015](../adr/0015-complete-wall-attribution-and-worker-scaling.md).
Independent design, correctness, code-conformance and measurement-arithmetic
reviews are READY. No blocking findings remain. This completion unit measures
CPU preprocessing; no games, fits, optimizer updates or final-test evaluation ran.

## Result and controlled scope

The previous44.477680s preparation envelope versus34.247827s legacy phase sum
left10.229853s without an attribution. Nested/summed worker counters did not
provide a complete wall partition. New exclusive coordinator and Python ledgers
now reconcile exactly. A large previously unlabelled operation is releasing
parsed episode/collection temporaries:6.020s in the representative new1-worker
scan. The old10.230s cannot be retrospectively divided exactly using a new run.

Default workers1/max_in_flight1 deliberately executes serially. Rust workers
read/hash/decode each episode's two participant journals and two Replays. Threads
join at each window, then validation, features and numeric spooling run in order.
Python's single owned FFI executor does not prevent detached-GIL Rust threads.
WorkersN with max_in_flight1 remains effectively one; both settings must increase.
Eight is newly allowed in closed config/native/benchmark CLI validation. Presets
remain1/1; no pool, pipeline overlap or parallel feature kernel was introduced.

Frozen83,783-action history preparation,6,784episodes, shared memory2048MiB,
cache512MiB, numeric chunk16MiB, scratch8192MiB, release builds and identical
numerical dependencies. Each trial is a fresh process; OMP/MKL/OPENBLAS threads1.
One warmup per variant, then three rotations of baseline1/1 and instrumented
1/1,2/2,4/4,8/8:20successful runs total. No concurrent builds/tests/training during
the matrix. All variants fit the original budget without retuning. Warm OS cache;
no cache flush or cold-storage claim. Linux/WSL2, Python3.14.9,16logical CPUs.

Raw records: `runs/performance/wall-workers-20261030/results.json` and each trial's
measurement/workload/timeline files. Summary:
`runs/assessments/wall-workers-performance.json`. These are local run artifacts.

## Whole-workload time and resources

Each entry is the median of three measured runs; memory columns are medians of
sampled per-run peaks. MiB=2^20bytes, GiB=2^30bytes. Wall includes the entire
command, archive/proof, final source checks, cleanup and process overhead.

| Workers / max_in_flight | Wall s | Process-tree CPU s | RSS MiB | PSS MiB | Sampled scratch GiB | Amortized ms/action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline1/1, old binary |47.952|47.837|405.0|396.7|1.555|0.572|
| Instrumented1/1 |49.213|49.081|406.2|396.9|1.395|0.587|
| Instrumented2/2 |36.852|34.676|469.5|444.4|1.407|0.440|
| Instrumented4/4 |32.322|35.207|580.4|552.2|1.729|0.386|
| Instrumented8/8 |31.181|38.671|749.3|736.9|1.729|0.372|

Amortized time divides the complete preprocessing workload by accepted actions;
it is neither a policy-inference latency nor an individual decision percentile.
Instrumented1->8 reduces elapsed time36.64%, about1.578x. Eight is fastest in
these trials. Four->eight saves1.14s(3.53%) while CPU grows9.84% and RSS29.09%;
four is a reasonable time/resource balance. This is an observation for this
corpus, not a general optimum or a silent change to preset defaults.

Active decoder overlap/effective slots reach1/2/4/8; all6,784jobs complete with
zero active jobs remaining. For2/4/8 there are3,392/1,696/848windows and6,784spawned
threads respectively, because threads are recreated per window. Logical slots
are aggregate positions reused across those threads, not persistent thread IDs.
The measured overlap proves concurrent jobs, not full utilization ofNcores.
CPU/wall averages about1.00/0.94/1.09/1.24occupied cores respectively. Serial
stages and window joins remain visible limits; their individual optimization is
outside this measurement unit.

The uninstrumented control retains source6067721/release3b12d7af... and is not
compared against a different memory budget. Instrumented1 is2.63% slower by median.
Report this as an observed source/build/instrumentation and run-variation
difference, not an exact per-timer overhead or a statistically established cost.
The three wall observations are49.067/47.899/47.952s for baseline and
49.492/49.213/47.443s for instrumented1. Avoid comparing old46.05s from a separate
campaign directly to this campaign's worker medians.

Internal tracked scratch peak is1,856,048,056bytes for all variants. Sampled
scratch peaks can miss short-lived files; observer output includes scratch and
must not be added to it. OS write-byte lower bounds are about1,943,031,808bytes
(4/8:1,943,023,616); read-byte lower bounds are0 except one4KiB observation.
Sources were read despite near-zero physical-read accounting: cache hits and WSL
host I/O are distinct. These counters do not establish physical-device traffic.
RSS sampled100ms, PSS/scratch1s can miss peaks; shared RSS may double-count pages.
Accounting reservations, RSS and JSON size estimates remain distinct; budgets
are soft application limits. GPU is not used by this workload.

## Complete single-worker attribution

Use the median-wall instrumented1 trial, `1-instrumented-w1`, for this entire
partition. These are exclusive coordinator elapsed categories from the same
`pipeline::scan` parent. Integer nanoseconds sum exactly to45,481,783,553;
residual0ns. Displayed decimals may round independently.

| Native scan category | Seconds |
| --- | ---: |
| Episode read/hash/decode execution, window joins and source merge |25.972227|
| Features and numeric spooling |7.926398|
| Episode/collection temporary release |6.020197|
| Journal validation |3.314427|
| Source metadata, linkage and canonical checks |0.696207|
| Episode facts, layout and buffer preparation |0.526359|
| Global identity, split and source-manifest checks |0.334518|
| Proof-ledger serialization |0.324235|
| Collection setup, indexes and inventory |0.313430|
| Collection source-proof check |0.052507|
| Other bookkeeping |0.001280|
| **Native scan parent** |**45.481784**|

Release includes destruction of borrowed journal/episode buffers and owned
Serde JSON trees at the original episode/collection boundaries. The timer does
not extend their lifetime or add a retention cache. This is ordinary Rust object
destruction/deallocation, distinct from Python GC, NumPy view release and the
configured memory reservation mechanism. A separate parse worker diagnostic is
20.587s across the data-stage snapshot, including metadata JSON outside the
episode-reader interval; it is not a strict sub-partition of25.972s and must not
be added to or subtracted from that interval to infer an exact remaining cost. Legacy
six-phase counters still include nested/repeated worker work and are explicitly
named auxiliary aggregated worker elapsed times, not CPU or a wall partition.

The25.972s is elapsed time around EpisodeReader::next, not OS iowait or a measured
blocked-I/O duration. With one worker, decoding runs inline on the calling thread;
with multiple workers the caller starts/joins a window and merges the source
evidence. File reads, hashing, parsing, allocation, scheduling and result handling
all contribute. Read/hash counters themselves also include CPU and metadata work.

Worker inputs and parsed documents/source maps are independently owned. Shared
objects include atomic memory-budget/cancellation/telemetry state and the process
allocator. The decoder error Mutex is used for error reporting, not around normal
parsing. Independence therefore does not imply an absence of shared-resource costs.
All window threads join before serial validation/features/release, and worker-
allocated documents are later freed by the coordinator. Existing overlap peaks
do not quantify allocator/atomic contention, per-thread CPU or barrier imbalance.
Those and the division of the6.020s release interval into JSON trees, journal
containers and collection cleanup remain explicit next investigation targets;
no allocator or scheduling cause is established by the current measurements.

The enclosing Python construction ledger also sums exactly to47,437,247,148ns,
with residual0ns:

| Python construction category | Seconds |
| --- | ---: |
| Owner/session setup |1.031646|
| next_batch wait, including native scan and handoff/proof decoding |45.739111|
| Proof and initialization |0.001458|
| Array/reward validation and labels |0.034885|
| Bounded NPY write |0.453246|
| Batch release |0.008310|
| Seal and stage snapshot |0.167235|
| Other |0.001355|
| **Python construction parent** |**47.437247**|

The native45.481784s parent is contained in Python's45.739111s next_batch wait.
Do not add the two ledgers. The benchmark preparation envelope47.437290s includes
that construction, followed by archive staging0.352241s, output proof0.459134s
and final checks/cleanup0.803973s. Including tiny import/bookkeeping scopes, its
internal workload is49.052662s. The observer's whole-command49.212615s also
includes startup/provenance/process-boundary overhead, about0.160s outside that
internal timer. This explicitly distinguishes all enclosing boundaries.

The legacy `phase_seconds.global_scan_and_encode` includes the first staged
numeric-batch read in the first Session::next_batch call. It is therefore not the
same boundary as the exact new `coordinator_scan_wall` scan parent. Data-stage
diagnostics are frozen before archive/finalize; finalization-later counters remain
separate. Neither snapshot is silently substituted for the other.

The remaining serial coordinator bottlenecks at8workers are features/spooling
7.724s, temporary release6.177s and journal validation4.549s, alongside decode
wait6.855s. Increasing worker count alone cannot parallelize those stages.

## Frozen identity, checks and rollout

Native source SHA256 `5e18ca4e3f0a371d79667b38c9fcf52b958e4da9907cd796b706bdf8e9f72fc5`;
release extension `32e3ad68f883087517a0d9b393764dd4d01a2543a24175d8b5eb912ded7f901f`;
native wheel `e6088ba3fd5e4b80d6ee392e07946ed0150976903d22b9fdc3066756350f8e19`;
Cargo.lock `d2a6f8d2034864b0266341f383e108783fde02b8dd2666530eee5a2173c7bb55`.
No new dependency. Serde contract`serde-json-v2`, ABI`cp314-gil`,
encoder`participant-action-v1` and JSON memory policy unchanged. Application wheel
SHA256`e09c8880ebc11ea1873847b325f3f40c00ea0ad035006e841757ea58b3240f07`;
all59packaged Python source files byte-match current source.

PASS checks:

- Rust debug52, focused release42; fixed ledger nesting/error restoration,
  exact sums, config8, decoder overlap/order/cancel/join/shared limits andN/1
  admission control. Commands: `cargo test --manifest-path native/Cargo.toml`;
  `cargo test --release --manifest-path native/Cargo.toml --lib --test health_truthiness --test json_budgets --test parallel --test retirement --test serde_depth --test telemetry`.
- Whole Python: `python -m pytest -q`:772passed,5opt-in skipped,250.97s.
  Focused Python139; sync/async writer/ledger10passed(1toy-fit deselected there);
  final ledger/CLI15passed. Whole Ruff, mypy59source files, cargo fmt/diff checks
  pass. Async tests ran with local socket permissions; no event-loop workaround.
- Full frozen83,783-row corpus both recipes and new Parquet index exact under
  original2GiB/16MiB. History NPZ`c5c3f3c1...`, observation NPZ`841c6f885...`,
  dataset`ab5d4913...`, index rows`574b81ec...`, Parquet bytes`9bce50db...`.
  `runs/assessments/wall-workers-corpus.json`; new index
  `runs/ppo-comparison-v2/index-wall-workers-20261030.parquet`. Those correctness
  runs overlapped tests/cold reads, so their elapsed times are not speed evidence.
- Saved real16-game PPO ingestion at1/2/4/8 retains features/GAE/targets identity
  `7e2629b8...`; `runs/assessments/wall-workers-ppo-ingestion.json`. No update ran.
- All20matrix outputs have exact dataset/sample/array/NPZ identities. Both new
  ledgers reconcile with zero residual; active decoder peaks match requestedN.
- Separate installed optional-free baseline help/config and installed CPU
  MLP/LightGBM inference pass. The runtime retains the verifiedcu132 stack; no
  dependency versions or GPU behavior changed. GPU tests are not part of this unit.

After all control runs, the old main native package was archived under
`/tmp/gfb-wall-workers-reference/native-site/gf_native`; the verified new wheel
was installed into Bots`.venv` with`--no-deps`, actual release SHA rechecked.
Original six fits preserve24file size/mtime/inode records and18small-file hashes;
large original NPZ files remain unchanged and their contents are covered by the
full-corpus equivalence checks. Failed historical parents remain wholly ineligible.
Simulator revision`ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6` remains clean/untouched.
