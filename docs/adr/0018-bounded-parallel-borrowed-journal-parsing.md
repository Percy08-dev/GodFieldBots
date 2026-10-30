# 0018: Bounded parallel borrowed-Journal parsing

- Status: Accepted2026-10-30; independent design and implementation source
  correctness/conformance reviews READY, execution correctness gates pass.
  Controlled performance acceptance pending.
- Owner: root. Baseline code1d3b059/native source37cae919; opt-in mimalloc typed
  wheelbc455458. Supersedes only ADR0017's coordinator-local parsing decision.

## Requirements and scope

The user requests parallel Journal deserialization and an equal-condition
before/after benchmark. ADR0017's ordered Journal parse/validation stage takes
about10.15s at both1and4workers. Move typed parsing to bounded scoped Rust workers;
keep semantic validation, feature generation and numeric/proof publication on
the coordinator in the original episode/participant/record order.

Preserve the existing reader selectors, default legacy path, shared memory/cache/
chunk/file/record/scratch budgets, raw admission, source validation, cancellation
and failure handling. No learning changes, new source data, games/fits, simulator
changes, dependencies, persistent pool, custom unsafe borrowing or main/default
wheel installation. CPU work uses native threads; no async runtime is needed.

## Ownership, interfaces and data flow

Keep the existing bounded read/Replay-decode window. A typed window has two
stages: read/decode and join, then borrowed Journal parse and join. This first
implementation intentionally retains a barrier between those stages and before
ordered consumption; overlapping windows is a later measured decision.

The coordinator owns the stable raw buffers for one window. Scoped parsing
workers borrow those buffers and return concrete `Records<'raw>` values through
join handles. They never own a buffer together with references into themselves.
All parse jobs join before raw storage can move or be released. Results carry
their original episode index; consume in input order, regardless of finish order.
No global sort or whole-corpus buffer is needed. Drop DTOs before their raw owners.

Refine crate-private parallel/pipeline interfaces only as needed to expose a
bounded owned window and separately mutable source-evidence ownership. Keep
raw EpisodeData and its source-evidence fields disjoint so ordered source merge
can move evidence without moving borrowed raw buffers. A concrete parsed-episode
container may hold optional per-participant records; do not introduce a general
task framework or a new generic JSON view. Legacy decoding/acceptance remains
unchanged and bypasses typed parsing. One worker uses the same parse placement
without spawning a parse thread.

Shared admission precedes parse allocations and remains live for every retained
raw/DTO window; parsing output/index capacities and native thread stacks also
use existing Budget leases. Reuse max_in_flight for the resident episode count,
and workers for maximum active parse threads. No hidden worker or memory increase.
Window retention can increase resident raw/DTO memory versus the old consumer;
measure its real peak and retained memory under the unchanged budget.

## Failure, information and compatibility

Check cancellation/deadline before and during each parse. On resource, syntax,
panic, spawn or cancellation failure, retain the first recorded cause, cancel
siblings, join every started worker and release all scoped storage/leases.
No failed parse window may produce successful numeric inputs. Ordered downstream
source/semantic checks remain authoritative; valid parse is not source eligibility.
Keep NumPy array owners and existing final mutable-source revalidation unchanged.

No DTO/raw data crosses Python IPC or survives its window. Diagnostic fields
stay outside policy features. Preserve schema4 source/Replay contracts, both
feature recipes/signatures/values, sample/source/dataset identities, saved PPO
candidate order and current Simulator pinffcffa0966dbf6d7ce1d982f4ee6444e291d42d6.
Binary/source identity changes naturally reject ordinary cross-version resume;
do not add a bypass or reinterpret old metadata.

## Timing and diagnostics

Add an exclusive native coordinator category for typed parse-window dispatch/
join wait. Existing journal_validation then measures post-parse validation; old
artifacts retain their old scope. Keep read/Replay wait and ordered source merge
in its existing category. All coordinator categories must reconcile exactly in
integer nanoseconds. Worker parse time is nested elapsed diagnostic data, not an
additive wall partition or CPU-utilization estimate. Report separate read/parse
overlap counters so the existing decoder counters are not silently reinterpreted.
Update journal_parse_scope and expose actual typed-parse worker overlap/counts.
Additive diagnostic fields do not change hashed numeric/source dataset formats.

## Verification and measurement

Before timing, test shuffled completion with deterministic output order, actual
2/4/8 parse overlap, one-worker behavior, invalid/canceled/panicking jobs, quota
refusal/cleanup and raw/DTO lifetimes. Preserve legacy behavior and retained NumPy
views. Run relevant Rust/Python checks and whole regression once code is stable.
Verify full fixed-corpus history/observation arrays/NPZ/sample/source/index parity
and the saved real current-v2 sixteen-game PPO input at1and4workers. Parent-run
locks require all real-source jobs to run sequentially.

Build a separate release mimalloc candidate with the same opt-level3/debug0
conditions as the retained serial baseline. Freeze sources/binaries/conditions
before primary trials. Use the same83,783-row frozen history workload, memory
2048MiB/cache512MiB/chunk16MiB, max_in_flight=workers, native thread controls and
observer. Compare serial baseline versus parallel candidate at1/2/4/8workers:
one warmup and five rotating fresh-process trials per cell, all sequential with
no builds/tests/edits during timing. Check exact outputs in every run. Report
median/range elapsed and CPU, kernel peak RSS and sampled tree RSS/PSS, post-release
memory, storage/I/O lower-bound coverage and cost. Separate correctness/setup/
profiling cost. Never infer device traffic or negligible reads from partial I/O
samples. Extract exclusive stage attribution from actual median-total trials.

After primary trials, bounded release-symbols CPU traces at1and4workers can
compare the new function/parse-worker scopes with ADR0017's profile. Use64KiB
DWARF stack capture, distinguish CPU from wall, inclusive overlap and unresolved
vDSO callers; profiler resources must not replace ordinary benchmark resources.

Independent correctness and final architecture-conformance review plus executed
evidence are required for acceptance. Report any slowdown/resource increase;
do not raise budgets, drop sources or alter defaults to manufacture a gain.
