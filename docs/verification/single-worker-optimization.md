# Single-worker local optimization experiments

2026-10-31. [ADR0021](../adr/0021-measured-single-worker-optimization.md).
Artifacts: `runs/assessments/single-worker-optimization-20261031/`.

## Scope and review

The user requests sequential implementation and measurement, reports no other
heavy task, and requires notice before a major rewrite or custom-library-level
work. Root notified them that allocation-free duplicate Schema comparison needs
a recursive comparison deserializer, while typed Replay requires changes across
source, parallel, pipeline, projection, view and validation. Both are deferred;
no custom parser/allocator/arena or persisted format change is authorized here.

Stage 1's bounded design received independent architecture READY before the
implementer started. Final source conformance and independent correctness are
READY with no findings. The disposable benchmark
controller was independently reviewed; direct-import and exceptional-reporting
findings were fixed before execution. The bounded Record-only Stage 2 experiment
also received design READY; it cannot start before Stage 1's measured decision.

## Stage 1: decoded-key rejection instead of a separate Journal scan

Only dynamic DTO maps add a reserved-key comparison after normal Serde key
decoding. Closed structs/custom visitors already reject that key. Typed Journal
removes the separate lexical guard, retains framing/UTF-8/size/depth/numeric
contracts, and checks Control immediately before and after decoding. Generic
Value parsing retains its guard. No library, dependency, format, Config, worker
or learning change. Shared standalone owned/borrowed maps now receive the same
reserved-key restriction as production Journals; invalid-input error wording
and ordering are not preserved.

The implementer ran three fixed acceptance tests against the old guard/unmodified
DTO decoder before production edits. Expected outcomes cover escaped and nested
dynamic keys, fixed/custom objects, ordinary reserved-text values, duplicate and
unknown fields, sequence forms, Unicode, integer/float bounds and the actual
typed recursion boundary (123 nested items accepted,124 rejected in this fixture).
Final native tests:92 PASS,2 existing ignored; standalone DTO contract tests:
11 PASS. Both Cargo fmt checks, native Cargo check, Ruff and mypy59 PASS.

Isolated release opt3/debug0/mimalloc candidate source:
`f98b03122c548e6ce897e4ee8454822997a9ffc7449a390c34e6eaf0db6ac935`.
ELF:`38ad8a230a6a3601ec7228df6826971a75d4218d7993a5d27c9b9d15a7957f45`.
Baseline is current installed sourcefaf71b33/ELF2ed57a31, preserved separately.
The main wheel stays unchanged throughout correctness and timing comparisons.

The isolated wheel passes106 Python parser/reader/validation/parallel/execution/
session/source-binding/zero-decision/PPO tests in241.90s, with the optional pool
cases enabled (`GFB_TEST_PARALLEL_PARSE=1`). This is the relevant native subset,
not a new full826-test application run. Full frozen history/observation arrays,
NPZ/sample/dataset/source identities,83,783-row exact Parquet/sidecar and saved
current-v2 PPO1/4worker proofs all PASS. Both PPO probes preserve16episodes/
224decisions/all arrays, and recheck72 original files each; diagnostic clones
stay running with zero updates. The five command wall times total44.9695s,
excluding identity gates and checks. Two diagnostic checkers initially used the
wrong JSON projection; their errors are preserved separately and corrected.
Every actual workload exits0; no successful workload is rerun or relabelled.

All12same-campaign commands PASS with unchanged original source/output/model/
main/build/code and zero residual scratch checks. One warmup per variant is
excluded. Five measured before/after pairs all improve. Source inventory and24
protected files match before and after. Artifacts:`stage1-primary/` and
`stage1-verification/protocol.json`.

| Metric | Before | After |
| --- | ---: | ---: |
| Whole elapsed median (s) | 22.554416 | 21.609089 |
| Whole elapsed range (s) | 22.080481–23.028602 | 21.137658–22.320268 |
| Reported OS CPU median (s) | 22.418075 | 21.369977 |
| Journal parse job-phase median (s) | 8.167892 | 6.983809 |
| Kernel peak RSS median (MiB) | 339.042969 | 348.964844 |
| Kernel peak RSS maximum (MiB) | 348.160156 | 349.925781 |
| Post-release RSS median (MiB) | 299.878906 | 300.687500 |
| Sampled peak PSS median (MiB) | 318.500977 | 318.491211 |
| Managed peak maximum (MiB) | 1531.020025 | 1531.020025 |
| Owned scratch peak maximum (MiB) | 1770.065361 | 1770.065361 |
| Native ELF size (bytes) | 7,479,936 | 7,478,128 |

Whole elapsed reduction is4.191318%; Journal job-phase reduction is14.50%.
Whole ranges overlap: paired wins and phase/CPU changes support retaining this
bounded change for this corpus, without a statistical-significance or other-
worker speed claim. Memory is not improved: median peak RSS rises9.92MiB,
maximum1.77MiB and post-release0.81MiB. Budgets remain unchanged and are soft
accounting, not an OS RSS cap. Phase medians are independently selected values;
do not sum them into an exclusive whole-median timeline.

Logical stage read/write stay1,068,020,777/1,003,260,077bytes. Owned scratch
stays1,856,048,056bytes. The six protected models remain1,898,536bytes in total;
all24 protected files retain their hashes. Warm-cache logical traffic is not
physical disk activity or measured I/O wait. Campaign elapsed is278.279555s.
Independent correctness and conformance result audits are READY. Root accepts
Stage1 with the documented memory cost. Main installation remains2ed57a31 until
the local Record capacity experiment has been decided.

## Comparison policy

Twelve sequential fresh-process commands per candidate: one warmup excluded and
five measured alternating pairs,1 worker/in-flight episode,2GiB/cache512MiB/
chunk16MiB. Both variants use the same full frozen history dataset and complete
episode pipeline. Every trial validates original source/numeric/array/NPZ/sample/
dataset proofs, record counts, unchanged limits, integer timing ledgers and clean
joined owners. All29,814 source files and24 protected fit files are verified
before and after; actual main/private wheel/build/source identities are checked.
No own concurrent builds/tests/edits/source jobs during measurements.

Whole elapsed, reported OS CPU, kernel peak/post RSS, sampled PSS, managed and
scratch peaks, logical I/O and model/build sizes are retained. These have different
scopes: cumulative allocations are not peak memory; sampled inclusive CPU and
parallel phase sums are not exclusive wall time; guest snapshots do not prove an
isolated host or physical storage activity. Compare medians, ranges and paired
results. Retain only measured useful changes; preserve rejected experiments.
