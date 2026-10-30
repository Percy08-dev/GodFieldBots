# Combined borrowed-Journal stage and function profiles

Measured 2026-10-30 after [ADR0017 acceptance](combined-borrowed-journal.md).
This is a diagnostic of the accepted opt-in mimalloc/`typed-borrowed-v1` path.
Application/native source, main installation, defaults, datasets and saved fits
are unchanged. The earlier [function profile](function-profiling.md) describes
legacy/System and must not be presented as the current candidate's timings.

## Conditions and retained evidence

- Frozen current-v2 dataset: 415 collections, 6,784 games, 13,568 Journals,
  534,640 records and 83,783 learner rows, public-history encoder.
- Memory/chunk budgets remain 2,048/16 MiB; workers/in-flight are 1/1 or 4/4.
  Whole source scans run sequentially under the existing parent ownership locks.
- Python 3.14.10/GIL, NumPy 2.5.4, Rust 1.99.0, pinned Cargo lock
  `85a520ff1c12a186487d79ef2e7f1a4cc232fc05879b807ebd31377c4a2eafa7`.
- Native source identity is
  `37cae919a0291fb2fbbb28738d0e13a206f7f060887bf2bfe7a6658a465770ca`.
  The ordinary candidate ELF is `bc455458...`; the separate profiling ELF is
  `28ac21d2...`. Both are release/opt-level 3 with mimalloc. Only the latter
  adds `CARGO_PROFILE_RELEASE_DEBUG=2` and `CARGO_PROFILE_RELEASE_STRIP=none`.
  No frame-pointer/compiler optimization changes or main wheel installation.
- Stage extraction: `runs/assessments/combined-borrowed-reprofile-20261030/`
  `stage-attribution.json`, from the completed five-trial candidate matrix.
- New raw traces, commands, binary identities, proofs, resources, demangled
  stacks, per-function JSON, caller-filtered subtrees and embedded-name Speedscope
  exports: `runs/performance/combined-borrowed-profile-20261030/`.
  `assessment.json` binds the exact binaries and all six diagnostic commands.
  `postprocess.py` and `aggregate.py` retain the disposable offline aggregation.

All six commands complete and reproduce the original arrays, NPZ bytes, sample
identity, dataset content/manifest identity and row count exactly. NPZ SHA-256 is
`c5c3f3c1ef4068766d81c7bd0cfa7a8957b140ad7ea1f207374fc725dcbcc94d`.
Each reports 13,568 typed files/534,640 records and zero legacy Journal files.
The original main ELF remains installed; only temporary `PYTHONPATH` overlays
select the candidate. Replay and small provenance controls still use legacy Value.

## Exclusive elapsed stage times

These are the actual median-whole-wall trials from five successful warm trials:
1 worker trial 4 and 4 workers trial 2. They are not independent per-stage
medians, which would not necessarily add to a real run. Native and enclosing
Python integer ledgers each reconcile with exactly zero nanosecond residual.

| Exclusive stage | 1 worker s | 4 workers s |
| --- | ---: | ---: |
| Collection setup, indexes and inventory | 0.078 | 0.073 |
| Collection source proof checks | 0.039 | 0.040 |
| Decode-window wait and owned-result/source merge | 5.215 | 2.966 |
| Episode/collection temporary release | 0.709 | 1.478 |
| Episode facts, numeric layout and buffers | 0.289 | 0.294 |
| Feature conversion and numeric spooling | 4.541 | 4.696 |
| Global sample identity, split and source-manifest checks | 0.320 | 0.297 |
| Typed Journal parse and validation | 10.155 | 10.125 |
| Other native coordinator work | 0.001 | 0.001 |
| Proof-ledger serialization | 0.260 | 0.270 |
| Source metadata, linkage and canonical checks | 0.623 | 0.643 |
| Enclosing preparation outside the native scan | 1.647 | 1.715 |
| Numeric archive staging | 0.374 | 0.344 |
| Archive proof/hash | 0.418 | 0.420 |
| Final source check and cleanup | 0.525 | 0.508 |
| Outer residual across measured timing envelopes | 0.143 | 0.159 |
| **Whole command** | **25.337** | **24.030** |

Rounding can affect the displayed sum. Native scan parents are 22.230250387 /
20.883001909 s. The preparation-outside-scan row is the enclosing preparation
envelope minus that scan, not pure Python CPU. The final residual includes work
outside those envelopes; it is not a separately measured startup or GC duration.
Python `next_batch_wait` and summed worker timings are nested diagnostics and
are excluded from this partition.

Decode-window wait includes raw Journal reads/hashes, Replay JSON parsing,
fork/join wait and result merging. It is not an OS I/O-wait measurement. The
Journal row includes typed deserialization, shape/semantic and journal checks.
Journal parse/validation and feature processing execute on the ordered
coordinator, so adding decoder workers does not parallelize those stages.

## Function-level sampled user CPU

`perf 6.8.12 record -e cpu-clock:u -F 499 --call-graph dwarf,65528` captures
`deep-w1` and `deep-w4`; offline inline-aware unwinding uses `--max-stack 512`.
The initial 8-KiB `sampled-w*` traces truncated deep Serde ancestors: their
1.894-second typed-parse root was smaller than its deserialize descendant.
Those root timings are rejected. The retained 64-KiB traces restore that ancestry.
All tab-indented inline frames are included, even frames without a DSO suffix.
Rust symbols use `c++filt -s rust`; matching libc debug symbols are supplied.

Seconds below are period-weighted on-CPU estimates over all workload threads,
not elapsed times, syscall time, I/O wait or call counts. Inclusive values include
descendants and therefore overlap. They must not be summed into the stage table.
`Record`/`Value`/`PreparedDecision` names abbreviate the full symbols in JSON.

| Function or caller-filtered subtree, inclusive | 1 worker CPU s | 4 workers CPU s |
| --- | ---: | ---: |
| `typed_journal::journal::parse_admitted` | 8.068 | 8.054 |
| `Record::deserialize` within that parse | 6.076 | 6.234 |
| `json::reject_reserved_number_members` within that parse | 1.345 | 1.184 |
| `typed_journal::journal::validate` | 2.140 | 2.098 |
| Typed `Record::validate` (shape checks; nested above) | 1.721 | 1.721 |
| `typed_journal::validation::input_semantics` (nested above) | 0.263 | 0.228 |
| `json::parse_admitted` (Replay/control Value path) | 3.639 | 0.918 |
| `Value::deserialize` (global inclusive) | 3.156 | 0.850 |
| `feature::typed::PreparedDecision::new` | 1.743 | 1.617 |
| `feature::typed::calculate<NumericSink>` | 1.635 | 1.537 |
| `source::Sources::read` | 1.389 | 0.116 |
| `projection::source_metadata` | 0.327 | 0.361 |
| `json::canonical` | 0.198 | 0.192 |
| `budget::Control::check`, resolved caller stacks only | 1.080 | 1.078 |
| `budget::monotonic`, resolved caller stacks only | 0.846 | 0.834 |

Within the 1-worker typed-parse subtree, deserialize accounts for 6.076 s,
reserved-member scanning for 1.345 s, and the remaining sampled parse work for
0.647 s. This is a CPU subtree decomposition, not an additional wall partition.
Global reserved-member scanning is 1.828 / 1.251 s; the table deliberately reports
only its typed-parse share in that row.

Selected self estimates for 1 worker: reserved-member scanning 1.659 s,
`__memcpy_avx512_unaligned_erms` 0.810 s, `__memcmp_evex_movbe` 0.745 s,
`budget::monotonic` 0.713 s, UTF-8 validation 0.691 s and mimalloc's
`mi_page_malloc_zero` 0.455 s. Self means the innermost reported inline frame;
standard perf physical-frame self attribution can differ.

There are also 2.501 / 2.447 s of self samples unresolved by perf in `[vdso]`.
Every reported instruction offset falls inside the matching ELF's
`__vdso_clock_gettime` range `[0xa70, 0xdc2)`; BuildID
`db469fb3d3d1f916c08281b31108a5038f87d15b` matches the captured mappings.
`deep-w*-vdso-report.txt` retains the offsets. The clock function is identifiable,
but its callers cannot be recovered reliably. These samples are not assigned to
`Control::check`, Journal parsing or another caller. Resolved ancestor estimates
can consequently undercount work behind an unwind failure. No claim that all
clock work comes from resource checks is made.

The 4-worker Replay/read figures are sampled CPU in many short-lived decoder
threads, not decoder elapsed time or evidence of a fourfold parser speedup.
CPU scheduling, allocator state and sampling/unwinding affect such comparisons;
the exclusive elapsed ledger is the authority for stage-level speed comparisons.

## Capture cost and limitations

| Diagnostic command | Whole wall s | Observed user CPU s | Sampled user CPU s |
| --- | ---: | ---: | ---: |
| First unprofiled symbols control, observed storage reads | 51.858 | 22.476 | unavailable |
| Warm unprofiled symbols control, 1 worker | 25.543 | 21.709 | unavailable |
| Final deep profile, 1 worker | 25.804 | 21.812 | 21.928 |
| Final deep profile, 4 workers | 24.132 | 19.180 | 18.756 |

Deep traces contain 10,942 / 9,359 samples, with zero frameless samples and zero
lost samples reported by perf. Sample-period totals differ from observed user
CPU by +0.53% / -2.22%. Observed CPU includes the recorder when present, and
these totals do not prove complete ancestor attribution. Individual function
estimates have sampling error. One trace per condition does not measure variance.
The 1-worker deep capture is about 1.02% slower than its single warm control;
this is an approximate diagnostic overhead comparison, not a paired benchmark.

The first control captures at least 3,330,408,448 kernel-accounted `read_bytes`.
The warm control's recorded counters are consistent with cache-served reads,
but all observer I/O counters are sampled lower bounds. Coverage during profiling
is incomplete: deep-w1 captures only 28,026 `rchar`, far below the workload's
input volume. Its low recorded `read_bytes` therefore cannot establish negligible
storage reads or an identical cache state. JSON byte reads can be served from
the OS page cache while parsing still consumes CPU. These are Linux process
accounting counters, not physical device/WSL-host traffic or I/O-wait measures.
Storage writes and profiler data are retained separately in the raw artifacts;
incomplete I/O coverage remains explicit. Diagnostic resource figures include
the profiler and must not replace the ordinary candidate's RSS measurements.

The largest current ordered stage is Journal parsing/validation, followed by
feature conversion/spooling. Its approximately 10.15-second elapsed duration
persists at 4 workers and limits decoder-worker scaling. This report changes no
processing, safety, compatibility, validation or resource-limit policy.

## Review

Independent stage arithmetic/scope and function/output/documentation audits pass.
The reporting finding about incomplete sampled I/O coverage was resolved by
qualifying the cache inference above. This is a measurement-only change;
application test suites are not rerun without code changes.
