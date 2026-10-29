# 0014: Bulk numeric I/O and collection episode lookups

- Status: Accepted2026-10-30; user identified scalar numeric I/O and quadratic episode scans.
- Owner: root; independent design/correctness/conformance and runtime gates PASS.
  [Verification](../verification/bulk-numeric-io-and-episode-lookups.md) records
  strong local microbenchmarks but no substantial current-corpus whole-time gain.
- Baseline80b4911, release native952657c7..., same frozen83,783-row corpus.

## Scope

Replace per-f64/i64 read_exact/write_all loops with bounded bulk array I/O in
pipeline numeric spooling/consumption and lib.export_staging. Files are already
32KiB buffered: expected benefit is fewer Rust calls/conversions/pushes, not a
claim of one prior syscall per value. Preserve exact little-endian typed bytes,
including f64 signed zero/NaN bit patterns at the storage-helper level. Learning
still rejects inappropriate nonfinite values through its existing checks.

Use pinned bytemuck1.25.2 safe typed-slice byte views for little-endian runtime.
Do not add handwritten unsafe casts or use uninitialized typed elements. Initialize
an admitted destination slice before read_exact; add a small checked BudgetVec
resize/extend-zero primitive or equivalent safe owned-buffer operation. Reserve
capacity/growth overlap before allocation; failed/partial reads never publish a
batch or advance Stage.cursor. Keep bounded32KiB chunks with control checks before/
after blocks. Big-endian builds use a bounded standard to/from_le_bytes conversion
fallback; test that fallback on the current host rather than assume portability.
No giant full-array byte-copy staging or changed shard format/NumPy owner contract.

Share only a small private numeric-I/O helper module across pipeline and export;
not an extensible serialization framework. f64/i64 operations suffice. Offset
arrays are read in blocks then shifted by checked integer arithmetic; headers/
first offset handling remains exactly equivalent. Bounds/overflow/truncation/
allocation/control failures fail the entire preparation.

## Lookups

Build ordinary HashMap indexes once per collection for manifest episode ID ->
(original ordinal, episode), schedule episode ID -> (group,spec), and group-health
membership episode ID -> health record. Borrow manifest values/keys while the
source documents live; do not clone whole JSON trees. Preserve manifest and file
consumption order and canonical proof order. Use episode IDs as keys, not pair/group
IDs. Retain missing, duplicate episode/schedule/health and schedule/result/health
checks; malformed inputs may fail earlier but cannot become eligible. No reliance
on HashMap iteration order.

Replace projection.links repeated scans and pipeline ordinal.position scans with
indexed lookup. Replace participant directory membership .any(episode paths) with
one HashSet of expected paths. Inspect session.open_collection validation for the
same nested episode/schedule loops and reuse its validated index or build a local
map once, rather than validating quadratically then indexing later. Keep small
fixed3-parent loops; do not broaden into unrelated collection/lineage rewrites.

Index memory is admitted as coarse collection-level capacity/entry estimates in
the existing soft budget, held for its lifetime. No per-probe quota/cancellation
hooks. Build/query boundary checks and normal source/finalization gates remain.
Borrowed indexes remain local to validation/scan scopes; do not create self-
referential Collection storage. Validation and scan may each build their bounded
linear index; neither phase may retain a quadratic episode walk.
Current formal collections contain16/64games; complexity is sum(E_collection^2),
not the square of the entire6,784-game corpus. Speed benefit remains empirical.

## Verification

Independent correctness/conformance review; focused helper tests for empty/small/
chunk-edge arrays, exact bits and endian fallback, short/failed I/O, quota/control,
checked offsets and no partial cursor publication. Manifest index tests cover
permuted input, canonical ordinal, missing/duplicate/multiple health membership,
process/overflow failure flags and duplicate source rejection. Synthetic scaling
or operation-count evidence confirms index construction+lookup is linear/expected
constant query, avoiding unstable micro-timing assertions in tests.

Full frozen both-recipe NPZ/index equality under2GiB/16MiB; saved actual PPO ingestion
1/2/4workers; appropriate Rust/Python/lint/type/wheel checks. Compare release scalar
vs bulk and repeated scan vs index microbenchmarks plus3alternating complete
preparation pairs, with CPU/RSS/PSS/storage/I/O. No model fitting, live games or final
test. Root owns Cargo/lock/config/docs/measurements/commits; one implementer owns
scoped native code/tests after design READY. No simulator or numerical env changes.
