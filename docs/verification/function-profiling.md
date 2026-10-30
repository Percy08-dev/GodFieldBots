# Function profiling of native preprocessing

Completed 2026-10-30. This is a bounded diagnostic experiment on unchanged source
`d05bc8f`, following [ADR0015](../adr/0015-complete-wall-attribution-and-worker-scaling.md).
Independent arithmetic/scope review: READY with the limitations below. No
application interface, dependency pin, dataset, model, simulator or scheduling
implementation changed. The main installed native binary remains unchanged.

## Method and interpretation

A separate Maturin release wheel uses opt-level3, `CARGO_PROFILE_RELEASE_DEBUG=2`
and `CARGO_PROFILE_RELEASE_STRIP=none`, without disabling inlining or changing
frame-pointer/compiler flags. Native source SHA256 remains
`5e18ca4e3f0a371d79667b38c9fcf52b958e4da9907cd796b706bdf8e9f72fc5`.
Diagnostic extension SHA256:
`32b3acc7f1236913a13d66ca001d45fa427d7f716d5f9763623e2311e35ccc7b`;
wheel SHA256 `f082a3d6ca69a159e2eb46fb8045fe7d20fcf5f237d73708e8b7174ce0328838`.

Ubuntu perf6.8.12 records `cpu-clock:u`,499Hz, DWARF stack8KiB, inherited worker
threads. It samples user-mode execution, not wall/blocked-I/O time or call counts.
perf reports, `perf script --inline`, GNU Rust demangling, and matching libc2.39
symbols resolve native/Serde/allocator functions. Tools and symbols were extracted
to `/tmp`; no system package installation or kernel-setting change. Samply0.13.1
imports the raw capture into Firefox format, but its own recording command refuses
the current `perf_event_paranoid=2`; perf's user-only recording works unchanged.

This follows the [Rust Performance Book's profiling guidance](https://nnethercote.github.io/perf-book/profiling.html)
for optimized builds with symbols. Raw captures cover the entire preparation
command. Tables select symbols on the reading/validation/provenance path before
`PreparedDecision::new`; they are not a timestamp-cropped interval. Shared helper
functions can occur under other callers. The allocator subtrees below explicitly
require the named caller in the sampled stack, avoiding feature/archive callers.

Inclusive time charges a sample once per distinct symbol in its stack, even with
recursion. Parent and child values overlap and must not be added. Saved JSON
`self_cpu_seconds_estimate` means the innermost reported frame, including inlined
functions. Standard perf may assign self to the physical containing function
instead; its Self column and the JSON's inline self are different views.
No per-function timers or per-node application instrumentation were added.

## Function estimates

Same frozen83,783-row history corpus,2048MiB shared memory budget,16MiB chunks,
512MiB cache, matching workers/max_in_flight, OMP/MKL/OPENBLAS threads1. Two complete
profiled commands at1/1 and4/4; both outputs exactly match the original dataset,
sample identities, numeric arrays and NPZ archive. No fitting, game or update ran.

The following are **inclusive sampled user-CPU estimates** for1 worker, rounded
seconds. They exclude blocked I/O and kernel work; short/absent samples do not
prove zero cost. Callers can overlap or be incomplete if unwinding stops early.

| Function | Estimated CPU s | Meaning |
| --- | ---: | --- |
| `Sources::journal` |14.98|Read and decode participant JSONL files|
| `Sources::json` |4.59|Read and decode Replays and metadata JSON|
| `Sources::read` |2.10|Bounded read, metadata, hash and source evidence|
| `json::parse_admitted` |19.12|Guard, Serde deserialization and JSON construction|
| `json::reject_reserved_number_members` |1.68|Existing reserved-number-key compatibility guard|
| `journal::validate` |2.92|Journal structure, identity and action/response checks|
| `projection::source_metadata` |0.37|Source/episode provenance projection|
| `json::canonical` |0.35|Canonical serialization, all callers in preparation|
| `projection::EpisodeIndex::new` |0.05|Collection index construction|
| `Sources::merge` |0.02|Merge worker source evidence|

The journal/JSON functions contain parse/read work; the same parse/read functions
appear under both. These rows therefore never form a disjoint total. Their
aggregate includes initialization metadata as well as episode input files.

Call ancestry gives these more specific estimates from
`sampled-w1-subtrees.json`:

| Ancestor required | Descendant | Estimated inclusive CPU s |
| --- | --- | ---: |
| `json::parse_admitted` |`__libc_malloc`|6.38|
| `json::parse_admitted` |`_int_malloc`|4.30|
| `json::parse_admitted` |reserved-number-key guard|1.68|
| `Sources::read` |SHA256 `digest_blocks`|1.30|
| EpisodeData drop |whole selected drop subtree|5.89|
| EpisodeData drop |`__libc_free`|4.55|
| EpisodeData drop |`_int_free`|3.53|
| EpisodeData drop |`malloc_consolidate`|1.32|

Each descendant is nested under its stated ancestor; malloc/free implementation
levels overlap too. The5.89s drop CPU estimate is consistent with this run's
5.95s coordinator release elapsed interval, without claiming identical boundaries.
Its native scan is44.278s; episode read/decode/joins/merge takes24.905s elapsed.
This is execution plus waiting, not24.905s of iowait. Inlining and ancestor-unwind
coverage mean the table is useful for hotspot attribution, not exact accounting.

Allocation during JSON construction and standard allocator work during document
release are substantial measured costs. Source-result merge is small in this
capture. This does not prove allocator contention, explain cross-thread-free
causality, or justify removing validation/guards. Changing data representation,
allocator or processing placement requires a separate measured design decision.

## Coverage, overhead and resource observations

| Capture / control | Command wall s | Process-tree CPU s | Sampled peak RSS MiB | Kernel read-byte lower bound |
| --- | ---: | ---: | ---: | ---: |
| First unprofiled1, cold source reads |74.955|52.850|404.2|3,330,285,568|
| Profiled1 |48.231|48.178|433.3|12,288|
| Profiled4 |32.161|34.979|638.4|0|
| Later unprofiled1, warm |47.246|47.152|404.1|0|

The warm single-pair observed difference is+2.084%; this is not an alternating,
replicated overhead benchmark. The cold first control is excluded from overhead
inference. Storage reads occurred despite near-zero warm-cache read accounting;
WSL host/device traffic is a different metric. Profiled CPU/RSS includes perf's
recording process and buffers. Postprocessing ran only after all timing controls.
Profile files are about177.6/145.4MiB, additional diagnostic storage outside the
application's managed scratch quota. OS counters are sampled lower bounds/peaks,
not hard limits or exact physical traffic; the existing observer limitations apply.

Perf reports zero lost samples.1 worker:21,301samples, period-weighted42.687374408
estimated CPU seconds vs measured user43.221104s(-1.23%).4 workers:16,686samples,
33.438877488 estimated vs measured user30.525171s(+9.55%). The4-worker estimate
therefore needs particular caution; do not use it to claim fine-grained CPU
speedups. No explanation for that discrepancy is established here. Aggregated
sample weights are estimates, not replacements for process rusage.

All samples have frames, but this does not prove complete stacks: Python frames,
8KiB stack truncation, optimized code and unwind termination can obscure callers.
The vDSO still has unresolved leaf names: about3.21s(7.52%) of1-worker sample
weight and3.27s(9.79%) of4-worker weight. Clock/check-related callers are visible
in some stacks, but those weights are not assigned entirely to a guessed function.
Other library symbols are resolved with matching installed-library debug files.
Small functions at this rate may have few/no samples. Worker4 has thousands of
short-lived thread identities; logical worker names do not mean persistent threads.

## Artifacts and reproduction

Local artifacts under `runs/performance/function-profile-20261030/`:

- `results.json`: four complete command/resource/output-identity records.
- `sampled-w{1,4}.data`: original perf captures, retained unchanged.
- `sampled-w{1,4}.demangled.txt`: symbolized stacks and source lines.
- `sampled-w{1,4}-functions.json`: inclusive and inline-self estimates and threads.
- `sampled-w{1,4}-subtrees.json`: estimates filtered by caller ancestry.
- `sampled-w{1,4}.speedscope.json`: embedded function names for interactive call
  trees/flamegraphs in [Speedscope](https://www.speedscope.app/). These files use
  its [standard sampled format](https://www.speedscope.app/file-format-schema.json);
  concurrent threads are aggregated CPU weight, not a wall timeline. Frame
  references, sample/weight lengths and total weights were checked.
- `sampled-w{1,4}.firefox.json.gz`: Samply imports, retaining thread timing;
  symbol/source lookup requires the diagnostic binaries and local symbol server.
- `capture.py`, `aggregate.py`, `subtrees.py`, `native-identity.json`,
  `native-build.log`: disposable experiment scripts and build provenance.

`runs/assessments/function-profiling.json` records capture settings, resource
summaries and SHA256s of profiles/exports/scripts. Application source/tests were
unchanged; regression tests were not rerun for this diagnostic-only unit.
Independent read-only review checked period sums, self/thread/function accounting,
recursion, all four output identities and the distinctions above.

Capture command, with full workload arguments preserved in `results.json`:

```sh
perf record -e cpu-clock:u -F 499 --call-graph dwarf,8192 --mmap-pages 1024 \
  -o sampled-w1.data -- .venv/bin/python tools/benchmark_workload.py \
  preparation --variant rust --encoder history --memory-mib 2048 --chunk-mib 16 \
  --workers 1 --max-in-flight 1 --seconds 300 --source /absolute/path/to/src \
  --native-sha256 32b3acc7f1236913a13d66ca001d45fa427d7f716d5f9763623e2311e35ccc7b
```

Use the separate diagnostic site in PYTHONPATH, explicit Bots Python,
`env -u VIRTUAL_ENV`, and the existing observer to provide its output/scratch
context. Keep imported native identity checks. The main production native package
was not replaced by the symbols build; original models/arrays remain untouched.
