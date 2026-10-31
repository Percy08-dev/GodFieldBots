# Single-worker parsing and allocation bottleneck analysis

## Scope and bounded design

User authorizes bottleneck analysis of the current1worker path, including function
calls, repeated scans/schema construction, allocation/reallocation, capacity
reservation and reuse. This is diagnosis, not production optimization. Do not
change application/native sources, dependencies, Config/defaults, stored data,
schemas, eligibility, checkpoints, models or the installed wheel. No games/fits/
optimizer updates. Root owns design/docs/orchestration; one implementer owns only
private diagnostic copies/harnesses after architecture review. Reviewers do not edit.

Current source faf71b3364a26feb3ff9bcf0605d065135a8fc1adc3f6571e3a252b30c8b79b6,
releaseopt3/debug0mimalloc ELF2ed57a3181bf3eae5ee6bf6cf214dcab6a32aa3c11878c7ff551dde71ca5f965,
commit476c13f. Data: original83,783history learner rows,415collections/6,784games,
13,568typed Journals/534,640records, unchanged current-v2 source inventory and
24protected fit files. Simulator/catalog/schema semantics remain pinned.

### Existing evidence and questions

Current representative1worker elapsed22.687734s includes read/hash/Replay5.204079s,
Journal parse8.190619s, validation2.071502s, feature/spool1.692052s, release0.632357s.
Old source-identical parser-function files have a pre-pool CPU profile; it can
locate functions but cannot be relabelled as current whole-runtime measurements.
Current call graph proves separate full UTF8, line-boundary, reserved-key and
Serde passes. Both schema fields still deserialize before equality/Arc sharing.

Raw source input already reserves file_length+1capacity. Records use Vec::new,
nested standard Vec deserialization has no JSON length hint, UniqueMap uses
BTreeMap nodes, Schema also uses Arc/Box, while numeric fields are inline and
unescaped Text strings borrow input. Coarse32*raw+64KiBadmission is accounting,
not backing-memory preallocation. Existing BudgetVec copy counters do not measure
Serde String/Vec/Map allocation, so they cannot resolve that hypothesis.

Questions: which scopes dominate allocator requests and CPU; how much growth is
Record collection versus nested Schema/Event/Catalog; what is capacity slack;
how much text is borrowed/owned; which apparent repeats are significant enough
to justify a small later experiment? Do not infer all allocation calls are costly,
all copies unnecessary, or that a single arena automatically fits standard types.

### Private allocation diagnostic

Create a disposable copy of current native source, releaseopt3/debug0mimalloc,
same locked dependencies. Instrumentation only: no parser/validator/canonical/
container-growth behavior changes, new allocation strategy or scratch reuse.
Use a tiny GlobalAlloc wrapper delegating alloc/alloc_zeroed/realloc/dealloc exactly
once to the existing MiMalloc. This is necessary because malloc-interposition
tools cannot be assumed to see this private statically linked allocator. Do not
override Python/libc allocation or install a process-wide LD_PRELOAD allocator.

Collect fixed-size thread-local counters with allocation-free const initialization,
no per-allocation logging, maps, locks, formatting, clock reads or unwinding.
Protect TLS teardown/reentrancy conservatively (skip/account unavailable counters,
never panic or recurse). Record allocation/zeroed/realloc/dealloc request counts,
requested bytes and fixed request-size bins by the currently active scope. Report
success/failure; for realloc separately count same-pointer/moved/failed outcomes
and old/new requested sizes. A pointer move's preserved-payload bound is not a
measurement of physical memcpy or memory bandwidth. Free-phase attribution is
not allocation-origin attribution; no pointer-origin map or exact live-heap claim.
Python/NumPy C/system allocations and mimalloc internal OS/page activity are outside
these Rust allocator-request counters. OS RSS/PSS remains separately measured.
The [GlobalAlloc contract](https://doc.rust-lang.org/std/alloc/trait.GlobalAlloc.html#the-global_allocator-attribute)
allows allocation elision/merging and forbids allocator unwinding. Counters describe
requests that actually reach this instrumented allocator, not guaranteed source-level
allocation operations. Fixtures must not assume an otherwise removable Box always
calls the allocator; test delegation explicitly or keep allocated memory observably
used. Store the old pointer address before realloc for comparison, without accessing
old allocation data after success.

Use a small static scope enum/hierarchy, with guards only at existing coarse
boundaries: source read/hash, Replay Value decode, Journal UTF8/line split/reserved
guard/Record deserialize, Schema fields, Event vector, Catalog map, Input from_wire,
Record-vector growth, validation, metadata/proof/identity, feature/spool, local
release and other/control. Child counters are exclusive leaf scopes; any derived
inclusive table must name its hierarchy and overlapping interpretation. Avoid
per-Event-element or per-Schema-node timer hooks. No clocks inside GlobalAlloc.

Serde field hooks, if used, must transparently delegate to exactly the original
Deserialize at the same decoder context, retaining recursion/unknown/duplicate/
numeric/lifetime behavior. Do not substitute RawValue, Value or custom JSON parsing.
Wrap from_wire at its call boundary so discarded-schema destruction is included
without changing drop order. Count selected Vec returned lengths/capacities and
Record push capacity changes; these are capacity/slack data, not heap traffic.
Borrowed/owned Text counters record visit paths/lengths without changing semantics.
If a desired subtype cannot be instrumented transparently, report it unavailable
instead of refactoring the parser to obtain a number.

Enable counting only for this one-worker scan in its actual native executor
thread; collect/retrieve totals after scope exit with counting disabled, so report
serialization does not count itself. A bounded fixed snapshot can be attached to
the private telemetry; no global persistent task runtime. Other threads/scopes
are excluded or explicitly reported. Keep instrumentation storage below16MiB,
with actual counter structure/capacity reported, separate from the2GiBsoft workload
budget. Report overflow/incomplete capture; never silently wrap counters.

### Current CPU profile and captures

Build a separate source-identical current symbols wheel opt3/debug2/strip-none,
with no instrumentation/optimization/frame-pointer changes and no main install.
Reuse verified perf6.8.12 cpu-clock:u499Hz/DWARF65528 and the owned observer,
matching tool/library environment. Attribute known relative target-to-leaf paths;
unknown within a required segment rejects it, upstream process entry is unnecessary.
Differentiate self/inclusive, unknown leaves, short-lived/unwound ancestry, standard
Serde work and allocator self. Never combine sampled CPU with elapsed ledgers or
claim exact complete CPU coverage. Source/buildIDs and current symbols must match.

After prior notice, run serially atworkers/in_flight1/1,2GiB/cache512/chunk16,
original full history workload and exact output proofs. At most seven initial
commands: one ordinary warmup; two interleaved ordinary/allocation-diagnostic pairs;
one unprofiled symbols control and one CPU capture. No own builds/tests/edits or
other source jobs overlap. Bracket immutable raw/protected/main/source/binary/code
checks. Capture actual guest load/pressure and record user load reports without
inferring isolated host. Allocation-diagnostic elapsed is overhead calibration,
not an optimization result; counts/size distributions remain usable only with
their explicit observation scope. Two pairs do not prove variance/significance.

If a diagnostic fails fidelity, retain it and authorize at most one narrow repair
and bounded retry after review; never retag failed identity/counter evidence. No
optimization A/B or extra worker scaling is included. An uninstrumented replay of
existing reference rows may validate harness logic without a workload run.

Keep raw perf trace4GiBand decoded512MiBas soft admission thresholds checked
before/after captures/decoding; report overshoot, not guaranteed quota. Prefer the
existing signal-aware owned observer for builds/offline decode so SIGTERM and
timeouts forward, reap descendants and retain partial artifacts. No sysctl,
affinity/quota changes or new profiler installation. If tools cannot resolve a
scope, record that limit, not a fabricated time/cost.

### Verification and acceptance

Architecture design review precedes diagnostic implementation. Implementer owns
small allocator-delegation/scope/reentrancy/known-size counter fixtures and source
instrumentation diff. Independent source/correctness review checks delegation,
fidelity and counter interpretation before full-data collection. Every successful
workload must preserve original arrays/NPZ/sample/dataset/native-owner cleanup,
reader and source/identity/count proof. Run exact original numerical/source guards,
not a new learned-data fixture. Main2ed57and production code remain unchanged.
Full application tests are unnecessary for private instrumentation alone; use
targeted meaningful diagnostics plus full actual-path equivalence. Source jobs
remain serial under existing parent locks; inherited Simulator VIRTUAL_ENV unset.

Final independent arithmetic/coverage and architecture conformance reviews close
the analysis. Distinguish measured costs, code facts, hypotheses, unavailable
evidence and estimated upper bounds. Rank future candidates by complete-path
potential and implementation/maintenance cost in Knuth's sense; no automatic
arena, preallocation, validation reduction or custom parser adoption. Record both
capacity opportunities and cases already adequately reserved/borrowed. Root owns
local documentation commit; no pushes or production-wheel replacement.

Artifacts: runs/assessments/single-worker-allocation-20261031/ and
runs/performance/single-worker-allocation-20261031/. Independent architecture
design review READY with no blocking findings. Private diagnostic source f57cb4d0, allocator safety and implementation-conformance
reviews are READY. Six focused private tests and12harness failure/metadata cases
pass; exception-masking and prior-decoded overwrite gaps were fixed before capture.
Production/native/main/config/dependencies remain unchanged. Execution follows.


## Executed fidelity and overhead

All seven commands PASS with exact original83,783-row arrays/NPZ/sample/dataset,
13,568typedfiles/534,640records, actual variant/ABI/source identities, one-worker
pipeline caps, both integer wall ledgers, complete counters and joined cleanup.
Counter overflow/reentry/TLS-unavailable deltas/failures are zero. The two diagnostic
passes agree exactly in counts, request-size histograms, vector and Text statistics.
All raw/protected/main/code/binary/dependency bracket checks pass. Three owned
offline CPU decode commands pass matching current symbols BuildID and frozen
relative-suffix policy. No extra workload, optimization experiment or main install.

| Pair/control | Ordinary wall s | Diagnostic/profile wall s | Scope |
| --- | ---: | ---: | --- |
|0|22.985|24.227|allocator counter overhead|
|1|22.391|23.508|allocator counter overhead|
|symbols|22.469|23.160|source-identical debug2 CPU sampling|

Counter overhead is approximately5.4%/5.0%; sampling pair approximately3.1%.
Two pairs/one profile do not isolate variance, prove significance, measure an
optimization, or establish that every count has equal time cost. OS/profile RSS
figures have separate scopes and cannot be mixed with ordinary workload peaks.

Fixed13,848bytes is per touched TLS bank.41,560bytes is one TLS bank plus retained/
incoming snapshots, not total process instrumentation memory: other touched
threads can instantiate disabled banks; static metadata/compiler stack copies/
report serialization add storage. These small structures are separate from the
unchanged2GiBworkload estimate, not an exact process-wide memory accounting claim.

## Actual allocation requests

All counts below are native scan-thread requests reaching the private MiMalloc
wrapper. Preflight/post-scan/later Stage/session teardown, Python/NumPy C/system
and allocator page/OS internals are excluded. Frees of preexisting allocations
are included; no allocation-origin or exact live-heap inference follows.

| Operation | Requests | Requested bytes |
| --- | ---: | ---: |
|alloc|132,412,122|25,591,516,060|
|alloc_zeroed|0|0|
|realloc|7,736,226|12,724,635,646(new size)|
|dealloc|132,300,274|31,932,745,189|

Realloc old requested bytes separately total6,362,198,292. All recorded successful
reallocs returned a different address; this does not measure physical memcpy.
Summing alloc/realloc/free requested sizes is cumulative request flow, neither
unique storage nor resident memory; no single buffer the size of cumulative request
flow is implied. Alloc+zeroed+realloc-new total38,316,151,706bytes (decimal38.316GB).

The denominator below is140,148,348allocation-producing requests
(alloc+alloc_zeroed+realloc), excluding dealloc. These are EXCLUSIVE leaf scopes.
Record-deserialize residual excludes the separately labelled Schema/Event/Catalog
hooks; CPU inclusive parent frames below do not use this partition.

| Scope | Allocation-producing requests | Share | Requested bytes including realloc-new |
| --- | ---: | ---: | ---: |
|Replay Value decode|99,903,401|71.28%|9,159,365,340|
|Record deserialize residual|10,132,077|7.23%|4,406,884,415|
|Observation Schema fields|9,288,624|6.63%|5,221,699,936|
|Input Schema field|9,261,488|6.61%|5,220,831,584|
|Metadata/proof/identity|6,276,604|4.48%|1,509,397,282|
|Event vector|801,406|0.57%|5,257,138,304|
|Outer Record vector growth|88,137|0.063%|2,991,657,312|

The two Schema scopes together have18,550,112requests. They are not all redundant
or removable: Observation hooks include13,568terminal observations (182,736visits
versus169,168Input visits). No factor-of-two speed/memory saving follows. Both
copies still construct before from_wire equality/share, a code-proven repeat.

110,785,034allocation-producing requests (79.05%) are at most63bytes;
119,193,885(85.05%) at most255bytes. The power-of-two histogram cannot recover
exact <=64/<=256counts because those thresholds split bins. Small requests are
measured, but request share is not CPU share or proof that the allocator dominates.
Typed Text paths total50,390,895borrowed visits/582,297,172bytes and zero owned
visitor visits. This excludes Replay Value/String and other owned strings, and
borrowed visitors can also reference already-owned Value strings. It does not
prove that every typed string points directly into the original raw file.

## Container capacity and reservation

| Returned vector scope | Observations | Length sum | Capacity sum | Slack sum |
| --- | ---: | ---: | ---: | ---: |
|Record|13,568|534,640|757,702|223,062|
|Event|169,168|10,659,454|15,259,820|4,600,366|
|Observation Schema|182,736|921,744|1,144,336|222,592|
|Input Schema|169,168|921,744|1,144,336|222,592|

These are aggregate returned slots across many files/Inputs, not simultaneous
memory. No element/BTreeMap node size is inferred. Record Vec makes13,568initial
allocations and74,569reallocations=88,137capacity changes. Event scope makes
162,384initial allocations and639,022reallocations; the returned-vector table
alone does not count per-event or recursive-schema growth. Raw source bytes already
reserve file_length+1before read. Ordinary JSON allocations are not backed by the
32*raw+64KiBbudget counter. BTreeMap/Box-backed structures cannot use a single
Vec::reserve without a representation/allocator-aware decode change. Existing
borrowed DTO lifetimes also prevent casually retaining Record objects across
replacement raw-file owners.

## Current CPU call graph

The source-identical currentfaf71b33symbols profile has9,471samples, zero frameless
samples,18.979959768estimated sampled user CPU seconds. Function rows
below are raw observed-ancestor inclusive sampled estimates, not the stricter
accepted known-suffix totals and not an additive wall partition; some leaf
symbols/ancestry remain unresolved. CPU profile includes later teardown and
other threads outside the scan-only allocation window. Never equate its totals
with those counter-window byte/request totals.

| Observed function/scope | Inclusive CPU estimate s | Self estimate s |
| --- | ---: | ---: |
|Journal parse_admitted|8.192|0.511|
|Record::deserialize|6.303|0.000|
|Vec<Schema>::deserialize|1.932|0.000|
|Vec<Event>::deserialize|1.359|0.000|
|reserved-number member guard(global)|1.637|1.497|
|Value::deserialize(global)|3.064|0.046|
|Input::from_wire|0.124|0.004|
|RawVecInner::finish_grow(global)|0.647|0.024|
|Vec<Record>::push|0.062|0.000|
|mi_realloc_aligned(global)|0.044|0.006|
|mi_page_malloc_zero(global)|not separate from ancestors|0.457|

Read-only caller filtering of the existing folded stacks finds2.907816s of the
global3.064128s Value-deserialize observations under parallel::read_episode and
process_episode. On this typed-Journal path those Value reads are the two Replay
documents; the remaining0.156313s is outside that episode-reader context. These
are observed-ancestor weights including unresolved descendants, not complete
known-suffix CPU estimates. No new capture or exact function cost is implied.

String scanning and UTF8 validation also appear (SliceRead::skip_to_escape0.675s
self; run_utf8_validation0.663s self), with overlaps and multiple callers. The
reserved-member guard has no allocation requests, so an arena/preallocation cannot
remove that CPU work. Its independent scan and Serde's parse are code-proven
separate passes. Full-file upfront UTF8 is much smaller in earlier exact-caller
inspection; do not assign all global UTF8 samples to that one check.

## Ranked conclusions and next experiment boundary

1. **Repeated parsing/representation is the strongest structural target.** Schema
   construction is visible in both CPU and allocation data, while Replay's generic
   Value path owns71.28%of allocation-producing requests; observed Replay-reader
   Value-deserialize ancestry covers about2.91s inclusive sampled CPU. Typed/
   borrowed Replay or duplicate-schema treatment may help, but
   neither count is an achievable saving. Preserve old source acceptance, exact
   canonical proofs and versioning; these need bounded separate designs.
2. **The reserved-key scan is a measured allocation-independent target.** Its
   roughly1.50s global self CPU plus an independent full pass supports checking
   whether existing Serde key handling can enforce the same rule. Escaped keys,
   value strings, duplicate/unknown/numeric/recursion rejection must remain. No
   deletion or replacement is authorized by this diagnosis.
3. **Capacity reservation is plausible, but outer Record reservation is not the
   dominant measured cost.** It is a small fraction of allocation-producing
   requests and about0.062s observed inclusive push CPU; incomplete sampling means
   this is not an upper bound on all growth cost. Event growth is larger;
   its1.359s deserialize scope also includes real parsing, not just growth. A
   narrow preallocation experiment can answer ROI; no broad arena rewrite follows.
4. **Typed strings/raw input are already efficient in the measured respects.**
   Text borrowing and raw capacity are confirmed; preserve them rather than
   replacing them based on aggregate copy counters.

This applies Knuth's critical-path reasoning: combine observed CPU with structural
work and maintenance cost, then test one narrow candidate against whole1worker
elapsed. Many small requests alone do not justify a custom allocator/parser or
arena overhaul. All diagnostic evidence is retained; no production optimization
or future speed improvement is claimed here.

Final independent arithmetic/interpretation audit PASS, method/evidence
architecture conformance and narrow final report review READY, no blockers.
All seven full-data
commands and three offline owners reconcile. No additional captures or optimization
experiments were run. Actual collection elapsed totals164.171582s and reported CPU
161.001907s; builds/setup/offline decoding are separately retained, not included
as ordinary workload runtime. Raw trace620,717,316bytes and decoded206,150,606bytes
remain below the soft thresholds without overshoot.

The current main wheel, production code, dependencies, Config, original source
data/models and resume guards remain unchanged. This closes the requested
bottleneck analysis; later production changes need their own scoped design and
complete-path before/after comparison. Evidence: collection/, analysis/summary.json
and the performance/single-worker-allocation-20261031/ profile artifacts.
