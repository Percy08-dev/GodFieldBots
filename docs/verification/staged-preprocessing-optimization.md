# Staged preprocessing optimization experiments

Accepted 2026-10-30 under [ADR0016](../adr/0016-staged-preprocessing-optimization.md).
Stages 0–2 are complete. Independent design, correctness, implementation and
measurement reviews are READY. No production allocator/reader was adopted;
no games, fits, PPO updates or final-test evaluation ran.

## Reference, coverage and correctness

Experiment implementation: clean `0b4d4a6`; measured freeze records 81 code hashes.
Python 3.14.10/GIL, NumPy 2.5.4, Rust 1.99.0, Linux/WSL2. Historical Python 3.14.9
measurements are not this experiment's controls. The original main native release
remains `32e3ad68f883087517a0d9b393764dd4d01a2543a24175d8b5eb912ded7f901f`,
source `5e18ca4e3f0a371d79667b38c9fcf52b958e4da9907cd796b706bdf8e9f72fc5`.
Simulator `ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6` remains clean and untouched.

Frozen dataset `ab5d4913fda45795d1cd5aa02a4a9c4c3464268298c4d3ea49d56d4909df770f`
has 415 collections, 6,784 games and 83,783 accepted learner actions (81,523 train,
2,260 validation). The existing native reader revalidated complete parent/source
eligibility; source-set identity remains `bef26c2bbf98e9240846b8f8d8414d7488240c0470204a14f2c43a05b6be0628`.

An independent census parsed all 13,568 participant journals, 534,640 records and
2,267,302,084 bytes with recursive duplicate-key rejection: zero failures, no
repair/exclusion, all files/records within 64 MiB/8 MiB. Record counts are 13,568
headers, 169,168 decisions, 169,168 acknowledgements, 169,168 transitions and
13,568 ends. The census includes full file hashes/origins in
`runs/assessments/staged-optimization-census.json`.

Both typed forms passed the whole corpus against unchanged `read_journal` and
exact Python canonical value/order equality, with zero newly rejected records.
Their canonical records SHA256 is
`c05bfca5cf9319b59cff52022c77441db30ac8607d00fad5ed9e669283c7d90c`.
Observer completion, actual binary, features and joined cleanup owner are bound
to each oracle. Current valid generated JSON therefore passed this experiment;
this does not prove future typed feature-array/learning equivalence.

PASS checks: 30 Python experiment controls; 12 Rust tests each with AP, no AP and
AP+counter; Ruff; mypy on three experiment Python files; rustfmt; deterministic
DTO regeneration. Tests cover missing/null/value, unknown/duplicate keys, direct
finite numbers, integer boundaries, Outcome, both defense shapes, actual nested
borrowed/unescaped and owned/escaped strings, overflow/budget refusal, frozen
inventory/binary authority, and process-tree cancellation/reaping.

Each final native wheel passed 40 existing native ownership, source/session,
cancellation and learning-input tests. All six allocator validation cells and
all 36 primary cells match the original numeric archive
`c5c3f3c1ef4068766d81c7bd0cfa7a8957b140ad7ea1f207374fc725dcbcc94d`
and sample identity `cbc11f1bc2810cb7a7536c07f42c64893f92693056dc1e90dbf41f9773676151`,
including all array identities. Original six fits retain all protected bytes and
file identities; source/model/main-native checks run before and after matrices.

## Allocator comparison: actual complete pipeline

Only the allocator recipe differs. System uses the original Cargo.lock unchanged;
common dependencies remain pinned for the other recipes. mimalloc 0.1.52 uses
`local_dynamic_tls`; tikv-jemallocator 0.7.0 uses `disable_initial_exec_tls`, retaining
default background-thread runtime support without enabling background threads.
There is no override/unprefixed malloc, LD_PRELOAD or main-environment install.
Allocator/runtime/compiler controls are sanitized case-insensitively; all builds
and actual loaded binary identities are retained.

Same full history preparation, release opt-level 3, memory 2,048 MiB, cache 512 MiB,
chunk 16 MiB, workers/max_in_flight 1/1 or 4/4, OMP/MKL/OPENBLAS threads 1. Each cell
has one warmup and five fresh-process rotating-order trials. All 36 succeeded.
No builds, tests, profiling or other heavy experiment work overlapped primary
measurements. Warm-cache observations; no cache-flush or cold-device claim.

Medians of five trials; RSS columns are medians of sampled per-run peaks or
explicit post-return/GC snapshots. MiB is 1,048,576 bytes.

| Allocator | Workers | Whole wall s | Process CPU s | Peak RSS MiB | Post-GC RSS MiB | Wall reduction vs System |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| System |1|46.363|46.142|403.3|384.5|—|
| mimalloc |1|33.056|32.926|357.1|299.2|28.70%|
| jemalloc |1|35.637|35.499|358.7|69.8|23.13%|
| System |4|30.830|33.915|576.5|404.9|—|
| mimalloc |4|27.669|29.165|405.8|298.2|10.25%|
| jemalloc |4|28.288|31.231|373.7|104.5|8.25%|

Wall min–max / inclusive IQR, seconds: System1 45.400–46.809 / 0.355;
mimalloc1 32.927–33.380 / 0.171; jemalloc1 35.531–35.702 / 0.033;
System4 30.445–31.610 / 0.282; mimalloc4 27.169–27.884 / 0.065;
jemalloc4 28.015–29.135 / 0.382. All individual observations and PSS metrics are
in `allocator-summary.json` and the raw results; five trials are observations,
not a population confidence interval.

mimalloc is the fastest measured recipe. jemalloc trades 0.62 s against mimalloc
at four workers for substantially lower post-release resident memory. Post-GC
snapshots include interpreter/imports, use no forced purge and do not measure
long-term retention. They are not proof of a live-object leak in another allocator.

Amortized full preprocessing cost per accepted learner action is about 0.553 ms
for System1 vs 0.395 ms for mimalloc1, and 0.368 vs 0.330 ms at four workers.
This includes both-view validation and final archive/proof/cleanup; it is not
individual policy-inference latency or a p50/p95.

Application-tracked scratch peak is 1,856,048,056 bytes in every cell. Sampled
scratch medians vary because one-second probes miss short-lived peaks; those
variations do not establish less disk usage. OS-accounted write lower bounds are
about 1.81 GiB per trial. Numeric/staging outputs are identical. Near-zero warm
read-byte accounting does not mean files were not read or measure host-device
traffic. Output includes scratch and must not be added to it. Budgets remain
soft application limits; no OS hard quota or GPU workload is claimed.

## Typed Journal comparison: standalone System benchmark

A = Value with AP; B = owned typed DTOs with AP; C = borrowed typed DTOs with AP;
D = Value without AP, diagnostic only. Separate feature builds/targets prove D
has no AP; all use System. AP and `float_roundtrip` retain pinned Serde versions.
The typed contract is `typed-journal-experiment-v1`, without generic Value or
Content subtrees. Legacy production parsing remains unchanged.

The harness processes one complete journal at a time. Common checks, read,
deserialize, DTO drop and input release are separately timed. A/B release raw
input after parsing; C retains it until DTO drop. Correctness conversion/canonical
oracles are separate processes. B/C additionally execute typed shape validation;
A/D do not execute full Journal semantic validation in their timed path. The
existing complete source/Journal validation was performed separately. These are
representation/lifecycle comparisons, not equivalent production feature pipelines.

All 24 primary runs succeeded: four warmups, then five trials for each variant.
All processed 13,568 files and 534,640 records under the same coarse admission;
managed-estimate peak was 323.94 MiB in every variant, below 2 GiB. That estimate
is distinct from the much smaller observed RSS and is not an exact heap measure.

| Variant | Whole command s | Deserialize s | DTO drop s | Typed validation s | Peak RSS MiB | Whole wall reduction vs A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: Value/AP |16.347|10.541|2.530|—|42.8|—|
| B: owned/AP |12.660|7.487|0.921|1.670|23.0|22.56%|
| C: borrowed/AP |11.095|6.564|0.385|1.639|18.8|32.13%|
| D: Value/no AP |13.877|8.681|2.136|—|37.2|15.11%|

All columns are five-trial medians and cannot be added to reconstruct a median
parent. `whole_natural_cycle` contains its phases; do not add it to them. Its
medians are 16.324/12.635/11.072/13.854 s. Common checks take about 2.16–2.24 s;
read about 1.05/0.38/0.31/0.78 s respectively. Input-release medians are below
0.017 s. Natural lifetime/allocator state can affect read-buffer allocation too;
no isolated cause for each difference is claimed. Whole measured CPU medians
are 16.346/12.659/11.095/13.876 s. Post-release RSS is 32.3/17.5/16.4/18.7 MiB.

Wall min–max / IQR, seconds: A 16.211–16.421 / 0.055; B 12.636–12.685 / 0.004;
C 11.082–11.105 / 0.016; D 13.865–13.940 / 0.048.

Separate single full-corpus counter runs (not performance trials) observed:

| Variant | Deserialize allocation requests | Reallocation requests | Cumulative requested GiB |
| --- | ---: | ---: | ---: |
| A |323,656,739|4,416,960|27.842|
| B |73,059,314|6,072,841|22.725|
| C |22,927,151|6,072,841|22.186|
| D |208,351,701|1,428,108|26.769|

Allocation requests exclude the separately shown realloc requests. Requested
bytes include realloc request sizes; they are not copied bytes, simultaneous
heap/RSS or a claim that the 2 GiB budget was exceeded. Against A, B/C reduce
allocation requests by 77.43%/92.92%, while realloc request counts increase.
Cumulative requested-byte reductions are much smaller than request-count
reductions. Report these as different resource measures.

A→D demonstrates measurable AP feature cost, including 115,305,038 fewer
allocation requests. It does not attribute every change exclusively to Number
strings. D's real-corpus parse counts match, but numerical equivalence is not
established: fixtures demonstrate out-of-u64 representation/precision changes and
`1e999` acceptance differences. D is not an adoption candidate. Allocator and
representation ratios must not be multiplied to predict a combined speedup.

## Artifacts, costs and decision

### Kernel RSS high-water supplement

The earlier RSS columns are sampled process-tree peaks. The observer also saves
kernel `wait4` ru_maxrss: the largest individual-process peak per trial, which
can capture brief peaks missed by sampling. The following are the **maximum of
five measured trials**, excluding warmups, in MiB. They are not sums of simultaneous
process-tree RSS; PSS remains a separately sampled measure.

| Allocator / complete preparation | 1 worker | 4 workers |
| --- | ---: | ---: |
| System |412.5|591.6|
| mimalloc |360.5|409.5|
| jemalloc |361.8|380.2|

| Standalone Journal representation, System | Kernel peak RSS MiB |
| --- | ---: |
| A: Value/AP |49.4|
| B: owned/AP |23.3|
| C: borrowed/AP |22.6|
| D: Value/no AP, diagnostic |46.3|

Raw per-trial values and medians are in `peak-memory-supplement.json`. These two
tables measure different workloads and must not be compared as if the standalone
numbers were the complete pipeline. No mimalloc+borrowed production combination
was measured; borrowed-input lifetime/worker ownership and exact feature-array
equivalence remain integration gates. The 2 GiB setting is the application budget,
not a measured peak or an OS hard limit.

All local evidence is under `runs/assessments/staged-preprocessing-20261030/`:
`measured-freeze`, `final-builds`, `final-typed-builds`, `allocator-validation`,
`typed-oracles/{B,C}`, `allocator-primary`, `typed-primary`, `typed-counters`,
three summary JSONs and `final-assessment.json`. The freeze contains input hashes,
protected original fits, code/build identities, settings and actual environment.
Each measured command, resource record, ownership completion and output proof is
retained. The CLI defaults to dry-run; implementation is under
`benchmarks/native_preprocessing/` and reuses the existing performance observer.

Cumulative observed command costs (not wall-clock elapsed of the whole agent
session): allocator matrix including warmups 1,211.47 wall / 1,252.61 CPU seconds;
typed primary including warmups 323.86 / 323.84; separate counters 55.99 / 55.98.
The 64 commands therefore total about 26.52 command-wall minutes and 27.21 CPU
minutes. Allocator correctness runs add 213.15 cumulative command-wall seconds;
two typed oracles ran concurrently at about 190 s each. Final seven build
intervals total 76.57 s, excluding earlier discarded probes/builds. Not every
setup/edit/check interval was instrumented; do not present these sums as the
complete experiment's elapsed cost.

Retained experiment artifacts, including build outputs/retries, were about
3,243 MiB logical / 2,493 MiB allocated with per-tree inode deduplication. Dedicated
Cargo/tool/uv caches are separately listed in final-assessment.json; summing
per-tree allocated values can double-count cross-tree hardlinks. This is retained
artifact storage, distinct from workload scratch peaks and kernel I/O counters.
No shared historical game-generation cost was regenerated or recharged here.

Recommendations for the next decision: mimalloc is promising for speed with a
small production change; jemalloc is promising when post-release resident memory
matters more. Typed Journal processing is promising, with borrowed DTOs best in
this standalone experiment, but production ownership/integration and feature-array
equivalence still require a new design/measurement unit. Existing model arrays,
reader defaults, main environment and simulator remain unchanged. No automatic
adoption or claim of training-quality improvement is made.
