# 0015: Complete wall-time attribution and worker scaling

- Status: Accepted 2026-10-30. Independent design, correctness and implementation
  conformance READY; complete ledgers, output equivalence and controlled worker
  measurements PASS. Evidence: [verification](../verification/wall-attribution-and-worker-scaling.md).
- Owner: root; architecture gate before telemetry/config/concurrency-cap change.
- Baseline6067721, installed release3b12d7af..., unchanged frozen83,783-row corpus.

## Required outcome

Explain44.477680s preparation envelope versus34.247827s summed legacy worker
counters without pretending the10.229853s difference is a known single phase.
Provide reconciled, nonoverlapping coordinator wall categories and preserve old
worker counters only as nested/summed diagnostics. Then measure workers1/2/4/8
with identical settings, real parallelism evidence and exact outputs/resources.
No performance-architecture rewrite, training, refit or Simulator change.

## Parallel semantics retained

Default1/1 deliberately serial. Current Rust workers decode complete episode
journals/Replays; per-window threads all join before ordered serial validation,
features and spooling. Python's single FFI executor is an ownership mechanism and
does not prevent Rust threads while GIL is detached. Keep this fork/window/join
and publication behavior unchanged for this experiment. Do not add a pool or
parallelize learning/feature kernels before evidence.

Allow8in closed Python/native config and benchmark CLI validation/tests, keeping
presets1/1 until results justify a separate explicit choice. Compare1/1,2/2,4/4,8/8;
include a smallN/1control if useful to demonstrate admission-limited concurrency.
Track episode/window/thread counts, active decode peak and per-worker job/time
aggregates, without per-row logs or mutable policy inputs. Distinguish configured
workers from min(workers,window_paths) and from actually simultaneous work.
Per-worker aggregates are logical slots across recreated window threads, not
persistent thread IDs. Active decode peak is overlapping jobs, not CPU utilization.
All workers share original byte/scratch quota; no automatic memory/worker changes.
If8failsunder2GiB, record failure rather than raise its budget toclaimcomparison.

## Timing model

Use low-frequency elapsed scopes; no per-JSON-node/value clock probes, deep-size
walks, giant tracing events or new framework/dependency. A small fixed coordinator
phase ledger/stopwatch may account exclusive elapsed across sequential scopes,
including explicit Other/residual. It must measure its own parent total and
reconcile category sum within rounding. Short RAII borrows must never cross child
hooks; errors must restore accounting. Worker timers remain separate because
multiple decoder threads overlap and their sum is not elapsed wall time.

Cover native scan: collection setup/index/inventory, decode-window wait+source
merge, journal validation, source metadata/linkage/canonical checks, proof ledger
serialization, episode facts/layout/buffer preparation, features+numeric spooling,
JSON/episode temporary release, collection proof checking, global identity/split/
source/manifest checks, and remaining bookkeeping. Combine adjacent categories
when that avoids intrusive refactoring, but do not leave the previous large
residual unexplained or rename it as a guessed operation. Keep features/JSON
legacy diagnostics explicitly nested under corresponding coordinator phases.
Explicit parsed episode-data release timing may need a private lexical-scope
refactor so borrowed journals die before owned documents drop; no ownership change.

Cover outer Python NativePreparedInputs (sync and async if shared changes touch
both): owner/session setup, next_batch waits, proof/initialization, array/reward
validation and labels, bounded NPY consumption, batch release, seal and residual.
Report this disjoint parent envelope separately from native constructor/data/
scan/finalization child measurements. Benchmark startup, final archive/proof and
final source/cleanup retain existing disjoint outer scopes. Never mix data-stage
snapshot counters with finalization-later counters silently.

Instrument whole boundary and coordinator coarse scopes, then measure an explicit
unmodified-vs-instrumented1worker control. Report timer overhead/noise; do not infer
exact production sub-times from materially perturbed traces. Default fine loop
budgets and ndarray/base-owner/failure/publication behavior remain unchanged.

## Verification and measurement

Tests for ledger nesting, exclusive sums/residual/error restoration, worker config8,
actual order/shared limits/cancel/worker-error/cleanup and active peak>1under a
controlled fixture (no flaky elapsed speed assertions). Code review verifies no
changing source health/eligibility/row order or leaking diagnostic metadata.
Full corpus NPZ/index equality under2GiB/16MiB and savedPPO ingestion; appropriate
Rust/Python/lint/types/wheel checks. Three measured rotations across1/2/4/8after
warmups, no concurrent build/test/training; CPU/RSS/PSS/storage/I/O/output hashes,
worker CPU/elapsed semantics and phase totals. KeepGPUnotapplicable forCPUprep.
Root owns Cargo/TOML/docs/experiment decisions/commits; implementer owns scoped
code/tests after architecture READY, independent correctness/conformance required.

## Accepted observations and limits

The new representative single-worker scan accounts for all45.481783553s with
zero residual; temporary episode/collection release takes6.020197401s and Other
1.280205ms. This measures current costs rather than retrospectively assigning the
old10.229853s difference. The Python construction ledger also reconciles exactly
and contains the native scan interval; its parent must not be added to that scan.

Three measured rotations after warmups give whole-workload medians49.213/36.852/
32.322/31.181s for1/2/4/8, with matching in-flight settings and unchanged2GiB/
16MiB budgets. All outputs match. Active decoder overlap reaches the configured
count, while feature/validation/release work remains serial. Eight is fastest;
four uses less CPU/memory for a1.14s elapsed tradeoff. Presets remain1/1; changing
their default and replacing per-window threads require separate decisions.
The uninstrumented1-worker control is47.952s: the observed2.63% difference includes
build/instrumentation and run variation, not an isolated timer-cost measurement.
