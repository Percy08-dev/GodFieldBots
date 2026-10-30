# Isolated Journal parse scaling: 8 / 16 / 32

Later [root-cause diagnostics](preprocessing-root-cause.md) investigate scheduling,
repeated work and CPU-time API sensitivity. Historical CPU fields below remain
OS-reported observations; they do not establish physical utilization or work savings.

Design accepted2026-10-30 with independent architecture READY, at the user's
request to vary parsing independently of reads. Harness correctness/fidelity and
implementation conformance reviews READY. All18primary commands and independent
measurement/statistics audit pass. Component-experiment acceptance is complete.
This is a disposable component experiment on ADR0018's exact parser/scheduler,
not a production Config/default change or an end-to-end speed claim.

## Bounded experiment design

Use an isolated copy of native source`e604697d...` from commit`fd706c1`, with a
release/opt3/debug0 mimalloc executable. Production files and installed wheels
remain untouched. The private copy changes only parser telemetry slot capacity
from8to32 and adds a narrow binary/harness entry. Decoder slots and existing
parse/window algorithms remain byte-identical. Use a normal binary, not cfg(test)
parsing hooks, and record exact source differences/build/binary identity.

The frozen current-v2 sources identify6,784episode directories/13,568Journals
and534,640records. Keep one fixed ordered list and resident windows of32episodes
across collection boundaries. Parsing alone does not interpret collection/game
semantics; this is not a source-eligibility or learning-data producer. Every cell
loads exactly the same bytes in exactly the same windows. Crossing collection
boundaries supplies32real episode jobs without changing task granularity or
duplicating inputs; actual production collections commonly contain only16games.

Fix read/Replay-decode workers at4, resident episodes32, memory2048MiB, file64MiB,
record8MiB, and the existing per-thread8MiB stack accounting. Vary only parse
workers8/16/32. Use existing Sources/EpisodeReader typed windows and parse_window;
retain parent-owned raw buffers, joined scoped workers and original-index output.
No persistent pool, artificial start barrier, new parser/DTO representation, unsafe borrowing,
dependency or budget increase. Quota refusal fails the condition; do not silently
change window size or workers. Any design change returns to root first.

Measure read/decode, parse dispatch/join, result hash/check, DTO release and raw/
Replay/evidence release separately. Parse-phase time contains the actual existing
parser plus worker dispatch/join, with no read or oracle work. Also report parse+
DTO-drop time. Process user/system CPU deltas around parsing are OS-reported
values; summed worker elapsed is nested, not CPU or additive wall time.

Stream typed records in original episode/participant order through Serde Serialize
to a SHA-256 writer outside the parse timer; do not rebuild Value trees. Bind
record/file counts, deterministic output digest and actual raw source proofs to
every trial. Include boundaries/ordinals in the output digest. All worker counts
must agree with the8-worker reference. The unchanged actual parser and pinned
input authority provide fidelity to the accepted implementation; this oracle is
not a replacement for full numerical/semantic pipeline validation.

Use the unchanged owned process observer for whole-command CPU, kernel peak RSS,
post-release RSS/PSS, sampled tree/storage and I/O lower bounds. Whole-command
resources include reads/oracle; do not call them parse-only memory or compare
them directly to the earlier Python pipeline's interpreter RSS. Preserve shared
budget/lease cleanup and report measured setup/output/drop costs separately.

One warmup and five rotating fresh-process trials per count (18commands), serial
execution with no concurrent heavy work. Freeze source/input/build/harness/binary
identities first and verify afterward. No source edits during primary timing.
Perform an untimed preflight of all three counts for capacity, order/hash,
actual configured/effective/active parse overlap,4-reader counters and zero final
active jobs/leases. Do not force simultaneous starts merely to inflate overlap.
Retain every failure, stop on failure, and do not reuse failed observations.

Root owns design/docs/orchestration; bot_implementer owns only experiment files
under`runs/assessments/parse-scaling-20261030/`. Independent architecture review
checks fidelity/scope before dependent experiment code, and independent
correctness/measurement review checks the final harness and results. No production
feature or learning acceptance is claimed from this disposable experiment.

## Results

Artifacts are under`runs/assessments/parse-scaling-20261030/`: `CLI.md`, fixed
`input.json`, build metadata/private-copy diff, three preflights, primary plan/
freeze/results/completion/summary and the disposable orchestration/summary scripts.
The machine has16logical CPUs and affinity to16CPUs. Production source remains
`e604697d4982d3160c1d74fe242f05c9d2a11927166042942e4c82e229d8ccb7`;
private source is`6aca8ddd496ca875ad532533d2924c9c50360b908c147084711b77f03b7de184`.
Normal release/opt3/debug0 mimalloc binary SHA-256 is
`8eca7869c30297f6e81d76fb52ac06a623b10fa222e3fe9c97dbfcec262f2287`.
No cfg(test) hooks or Python shared-library/linker overrides are used.

The exact source comparison leaves parser, scheduler, DTOs, source reader,
budgets, Cargo dependencies/lock and decoder slots unchanged. Private lib entry
and parser-slot capacity are the only edits to copied production files; two
new harness/binary files are listed and hashed. Production files are unmodified.

All18commands (3warmups+15measured trials) reproduce13,568Journals/534,640records
over6,784episodes/212windows. Output digest is
`b824c463a73e6e78f7947307ee84467aeb73274ddd9107c7ecdb5785447eb0ff`;
actual-source-evidence digest is
`13311b5f70cc5affac4ec03f1756a18eb0013442e7c426da4ae2acb3e0b60fd7`.
Those digests, input/build identities, resources, timing partition, completed jobs
and zero final managed leases are checked every trial. Final input/private-code/
binary and original29,814source/117production-code/24protected-fit/main-install
checks pass. No builds/tests/edits or other heavy jobs run during primary timing.

### Parsing only

Five fresh-process measured trials per count, warmups excluded. Reads remain4,
resident episodes32 and memory2GiB in every row. Parse time includes the unchanged
parser, dispatch and joins; read/Replay decode, oracle and drops are excluded.

| Parse workers | Parse median [range] s | Reduction versus8 | Parse+DTO-drop median s | Reported parse CPU median s |
| ---: | ---: | ---: | ---: | ---: |
|8|2.264 [2.248,2.339]|reference|2.948|7.794|
|16|1.926 [1.912,1.969]|14.94%|2.614|8.799|
|32|1.859 [1.851,1.979]|17.88%|2.551|9.813|

16to32 saves0.066472s/3.45%of parse median. The derived parse+DTO-drop sum improves
2.39%; serialization lies between those intervals and is deliberately excluded.
Reported parse CPU rises8.799to9.813s. CPU is process `RUSAGE_SELF` delta including
dispatch/join, not summed worker wall time or a count of processor work. Five
observations/ranges do not establish statistical significance or unlimited scaling.

All cells retain exactly4read jobs at peak,848read threads and212read windows.
Parser threads are1,696/3,392/6,784 for8/16/32; every case completes6,784episode
jobs with zero failures/remaining active jobs. Effective available parser slots
are8/16/32. Observed active parser-job peaks are8/16/20–27 (median24in the32case).
Configured32 does not mean32simultaneously executing CPUs. No artificial start
barrier is added. The old telemetry configured_workers field continues to show
the reader's4; top-level parse_workers and parser slot/job counters identify the
independent experimental setting.

### Component-process memory and total cost

| Parse workers | Whole-command maximum kernel peak RSS MiB | Median post-release RSS / PSS MiB | Median whole-command wall / CPU s |
| ---: | ---: | ---: | ---: |
|8|472.18|400.20 /398.48|10.426 /18.365|
|16|479.50|471.13 /469.41|10.128 /19.548|
|32|541.09|498.55 /496.83|10.123 /20.544|

Peaks are the maximum exit-rusage RSS among five whole-command observations;
they include read, oracle, runtime, allocator retention and diagnostic publication.
They are not parse-only memory or the earlier Python interpreter's pipeline RSS.
Post-release snapshots follow raw/DTO/input/control owner release, with zero
managed leases; resident allocator/runtime memory remains. Managed admission
high-water is1,091,296,841/1,158,408,841/1,292,632,841bytes, within the fixed2GiB.
Those are conservative accounted reservations, not observed resident bytes.
16to32 increases maximum RSS61.59MiB and median post-release RSS27.42MiB.

For each parse-median trial, the disjoint loop phases are:

| Exclusive elapsed phase |8workers s|16workers s|32workers s|
| --- | ---: | ---: | ---: |
| Read/Replay decode |1.771|1.767|1.776|
| Parse dispatch/join |2.264|1.926|1.859|
| Oracle hash and source checks |3.694|3.713|3.714|
| DTO drop |0.675|0.688|0.692|
| Raw/Replay/evidence drop |1.995|2.009|2.030|
| Other loop work |0.001|0.001|0.001|
| **Loop total** |**10.400**|**10.104**|**10.072**|

Integer nanosecond partitions reconcile exactly; rounding can alter displayed
sums. These are actual parse-median trials, not independent stage medians or
necessarily the median whole-command trial. Derived parse+drop is not added
again. Setup/final control release/publication are outside this loop and included
by the external whole-command observer. Component disk output consists of
diagnostics/digests, not NPZs or training data. Raw sampled I/O/storage counters
are retained and remain lower bounds, not physical device/host traffic or I/O wait.

Primary command-wall sum is184.632s and reported CPU sum350.754s; matrixwall is
185.753s and final verification2.937s. Initial original-source/code check takes
21.157s. Three correctness preflights and the offline release build are separate
setup; build and primary precheck durations were not separately clocked. The first
preflight's read phase is9.859s and is excluded from performance statistics;
each primary condition receives its own warmup. No application regression suite
is repeated because production runtime code has not changed.

### Interpretation and acceptance

Within this fixed component setup,16workers reduce parse time substantially
versus8. The32-worker median is shortest, with a smaller additional gain and
increased CPU/peak/retained memory. This measures a fixed32episode cross-collection
window; ordinary production collections commonly contain only16episodes, and
the old end-to-end comparisons used different resident windows. These numbers
cannot establish an end-to-end production gain or be substituted into prior
pipeline totals. No separate production parse-worker setting is adopted here.

Production public settings still support1/2/4/8shared workers; independent16/32
parser settings exist only in this isolated executable. Adoption would require
a separate Config/window design and full-pipeline verification.

Independent source/fidelity and architecture conformance reviews READY.
Independent measurement/statistics audit PASS with zero findings: all18conditions,
identities, digests, counts, lifecycle checks, integer loop partitions, derived
times, resource scopes, representative selections and recorded final checks agree.
No runtime jobs or payload hashes were rerun by reviewers. This closes the
component experiment; it does not change production acceptance or defaults.
