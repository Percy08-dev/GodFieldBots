# Final preprocessing worker scaling and serial-function profiles

## Authorized scope and measurement design

User authorizes comparing the final adopted runtime at1/2/4/8workers and
investigating remaining Journal validation and temporary-release costs.
On2026-10-31 the user reports that the separate heavy task is not running.
Notify before each performance campaign; do not infer controlled host load from
that report. Record guest CPU/pressure snapshots with the actual commands.

This is a diagnostic task. No application/native implementation, dependency,
Config preset, simulator, source dataset, model, eligibility or checkpoint change.
Root owns design/docs/orchestration/acceptance; one implementer may prepare only
disposable assessment/profile harnesses. Independent reviewers check measurement
integrity, interpretation and conformance. No games, fits or optimizer updates.

Sources of truth are architecture.md, ADR0019 and the final installation proof:
native source d5090c4199c7462d83b50c486057aa682b0e0808475a048d3d48621f288ed961,
releaseopt3/debug0mimalloc ELF97f1b50990b4b97100c922e0191de4dd447ecb2c27bbbd81615d0c75ce67b628.
Use the unchanged isolated accepted wheel; verify the installed main wheel has
the same ELF/build. Preserve the previous32e3comparison evidence unchanged.
All commands unset inherited Simulator VIRTUAL_ENV and select Bots Python3.14.

### 1. Whole-path scaling

Use the existing83,783-row current-v2 public-history frozen workload:415collections,
6,784games,13,568Journals and534,640records. Retain original raw inventory hashes,
NPZ/array/sample/source/dataset goldens and24protected original fit files.
Create a new freeze that explicitly records the now-installed97f1main identity;
do not edit/relabel the old32e3freeze. Preserve simulator revision
ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6/catalogv2/schema4semantics.

Fixed memory/cache/chunk2048/512/16MiB, scratch8192MiB, reader typed-borrowed-v1,
releaseopt3/debug0mimalloc, native thread controls and owned observer. Change the
existing workers/max_in_flight together to1/1,2/2,4/4,8/8. This is the effect of
the complete supported concurrency/window configuration: read workers, parse
workers and resident episodes change together. It is not isolated parse-worker
scaling and adds no independent worker Config.

One fresh-process warmup per condition, then five trials per condition with
rotating order:1/2/4/8,2/4/8/1,4/8/1/2,8/1/2/4,1/2/4/8. All24commands run
sequentially under existing source locks. No own builds/tests/profiles/edits or
other source jobs run during primary timing. Check original outputs, loaded build,
reader/record counts, task units/counts, worker caps, completed cleanup owners and
integer native/Python wall ledgers every command. Bracket source/protected-fit/
main/binary/code checks. Do not bypass final mutable-source validation.

Report medians and observed ranges for whole/stage wall and OS-reported CPU;
kernel workload-self peak RSS, sampled tree RSS/PSS, post-release RSS/PSS,
managed peak, logical staging and scratch ownership, sampled storage/I/O with
coverage limits. Extract a complete exclusive stage table from the actual
median-whole trial at each worker count, keeping outer subtraction/residual
scopes explicit. Never add nested worker CPU/wall to exclusive stage totals.
Keep current CPU-accounting uncertainty and host/device attribution limits.

A resource failure makes that configuration unusable under this fixed budget:
retain the failure, join cleanup, and skip remaining identical trials for that
cell; never raise budget or shrink sources automatically. Identity, source,
numerical, lifecycle or unexpected infrastructure failure stops the campaign
pending investigation. Do not pick a fast partial/failed run as a result.
Recommend settings from measured time/resource tradeoffs without changing defaults.

### 2. Serial-function profiling

Separate diagnostic captures from primary timing. Build a private source-identical
releaseopt3mimalloc wheel with debug2/strip-none only; no instrumentation,
frame-pointer/optimization change or main installation. Freeze source/dependency/
build identities and verify original output proofs in every capture.
Reuse the existing deep-profile workflow: perf cpu-clock:u499Hz, DWARF65528bytes,
inline symbols and matching debug/build IDs, Rust demangling. No sysctl, affinity,
quota or machine-wide tracing changes. If perf permissions fail, report the
unavailable path and use the permitted read-only capture path when available.

Profile1worker and the parallel condition with the lowest observed primary median
among2/4/8 (choose the smaller count on an exact tie). For each, take one
unprofiled symbols control and one profiled command, all sequential. These four
diagnostic commands calibrate approximate overhead and retain exact output proofs;
one observation per capture cannot establish variance or performance improvement.
Keep profiler time/memory/storage out of ordinary runtime resource comparisons.
Use4GiBadditional raw traces and512MiBdecoded summaries as soft admission
thresholds: check actual sizes before/after captures and decoding, stop further
work when exhausted and report any overshoot. These checks are not hard quotas.
Do not silently discard failed/incomplete traces or rerun an unbounded series.

Extract inclusive and self sampled CPU for JournalView::validate,
typed_journal::journal::validate, generated Record/Input/Observation validation,
input_semantics and their resolved subtrees. For release, separate caller/type-
anchored Records/Journal/EpisodeWindow/Value destruction, raw-buffer release and
allocator activity where resolvable. Global allocator/free/clock samples must not
all be charged to the temporary-release stage. Inline paths/unresolved ancestry
and short-lived worker sampling remain explicit; vdso clock self without a
resolved caller stays unattributed. Function CPU estimates are not function wall
times or an additive partition; report them alongside the unchanged exclusive
stage ledger. Do not infer physical I/O wait or hidden CPU utilization.

Reject truncated/unmatched/unwound root estimates rather than inventing function
times. Prefer the existing verified64KiBworkflow over new custom instrumentation.
If sampling cannot resolve a requested subtree, record the missing evidence and
design a separately bounded diagnostic before modifying any source.

### Acceptance and retained evidence

Architecture/measurement-design review precedes dependent harness work. Source
reviews are limited to disposable harness fidelity; no production test suite is
rerun for this measurement-only task. Each actual command still verifies original
data/numeric/build/lifecycle contracts. Independent arithmetic/resource review
and final design conformance precede a local documentation milestone commit.
Report executed PASS/FAIL/NOT RUN precisely and list remaining uncertainty.
Artifacts use runs/assessments/final-worker-scaling-20261031/ and
runs/performance/final-native-profile-20261031/. Future optimization requires a
separate design and comparison; no pool, caching or validation reduction follows
automatically from these diagnostics.

## Execution

Independent design review READY. Before primary execution, source review found
that a signal exit with terminal ResourceLimit text could be classified as an
unusable cell. The classifier now requires normal +1 workload/observer exits and
failed status;14synthetic fixtures pass and independent re-review is READY.

All24primary commands pass, with five measured trials plus one warmup per cell.
Independent metadata/arithmetic/resource/ledger audit PASS with no discrepancies.
All source/protected-fit/main/binary/code bracket checks pass. User reports no
separate heavy task; actual guest observations are retained, not proof of host
isolation. The following are measured trial medians/ranges, excluding warmups.

| Workers/in-flight | Whole median [range] s | OS CPU median s | Self peak RSS median/max MiB | Sampled tree PSS max MiB |
| --- | ---: | ---: | ---: | ---: |
|1/1|22.599 [22.500,23.009]|22.414|344.69 /345.71|336.44|
|2/2|19.242 [18.994,19.400]|14.033|376.09 /395.53|346.25|
|4/4|15.907 [15.822,15.922]|13.997|419.15 /431.95|404.54|
|8/8|14.362 [14.243,14.662]|15.043|489.05 /507.30|458.05|

8workers saves1.544921s/9.712005%against4, with separated observed whole ranges.
The tradeoff is+75.35MiBmaximum/+69.90MiBmedian self peak RSS and+7.47%reported
CPU median. Recommend8for preparation speed on this corpus/machine;4is the
lower-memory alternative. No default/preset is changed. Reported CPU, especially
1versus2, does not establish physical CPU efficiency because known accounting/API
uncertainty remains. Guest measured busy ranges1/2/4/8are approximately
7.00–8.03/9.64–9.90/11.52–12.01/14.08–14.42%; these include guest work.

Managed peak1/2/4is1,605,390,806bytes. At8, median/max is1,669,511,205bytes
(min1,669,310,643), within the unchanged2GiBcoarse policy, not an RSS quota.
Post-release RSS medians1/2/4/8are300.12/294.91/293.85/295.84MiB, and PSS
291.77/286.65/285.24/287.46MiB. Every row has identical logical staging read/write
1,068,020,777/1,003,260,077bytes and owned scratch peak1,856,048,056bytes.
Different sampled storage peaks are not storage savings. Sampled read_bytes
12–20KiBdoes not establish negligible physical/host input I/O.

All four actual-median trials have complete27-category disjoint whole ledgers;
native and Python integer partitions reconcile exactly. All24command wall totals
439.367s, reported CPU397.627s; bracket validation takes24.110+3.236s separately.
Artifacts: runs/assessments/final-worker-scaling-20261031/primary/.

### Profile-side review and phase boundary

Source review found two profile-only blockers: offline decoding could abandon
children on SIGTERM, and reused decoder/aggregator bytes needed pre-capture
identity binding. These do not affect the completed primary matrix. Before any
fix, all nine primary harness files were preserved as readonly byte copies in
primary/frozen-harness/with a SHA-checked manifest. Original primary artifacts
remain unchanged. Reused tool hashes were preregistered before any profile capture.

Architecture accepts a separate profile-phase freeze binding the completed
primary freeze/results/summary/completion/snapshot manifest, current approved
profile harness, same application/source/golden/main identities and reused tool
hashes. Original primary evidence must not be relabelled after profile fixes.
Offline work uses the existing signal-aware observer with bounded execution,
partial-output retention and descendant draining. Derived scripts get expected
byte hashes before execution. Both profile-only blockers are resolved after
source re-review. Six targeted synthetic checks cover partial failure output,
timeout, parent SIGTERM/TERM-ignoring descendant cleanup, mutation rejection,
dependency/derived-byte identities and unchanged primary evidence.

The source-identical opt3/debug2mimalloc private wheel builds successfully under
an owned observer (30.918s, setup cost only). Main remains97f1unchanged.
All four controls/captures pass original outputs, full identity and source/owner
checks. Controls and captures use matching library environment. Diagnostic wall
control/profile is24.240/25.556s at1worker and16.524/16.950s at8workers. These
single pairs calibrate approximate profiling cost only, not worker variance or
an isolated profiler-overhead claim. They do not replace debug0primary timings.

Both64KiBDWARF captures decode with matching BuildIDs and inline Rust symbols.
Six offline observer-owned commands finish and drain descendants. Raw trace
storage is1,130,740,080bytes and decoded artifacts341,909,830bytes, below the
explicit soft thresholds without overshoot. All source/protected-fit/main/build/
harness/dependency/derived-script postflight checks pass. Artifacts and command
logs are under runs/performance/final-native-profile-20261031/.

After all performance captures completed, the user reports that another heavy
task is now active. Its exact start time is unavailable. No further benchmark or
profile capture runs follow that message; only light saved-data analysis and
documentation continue. Host-load isolation is not inferred retrospectively.

### Versioned function attribution refinement

The initial conservative subtrees filter returns unresolved for every target:
it requires a process-entry frame such as _start/clone, while these callchains
begin at _PyEval_Vector and contain identifiable downstream PyO3/Rust functions.
That upstream truncation does not prevent attribution to a fully known observed
target-to-leaf segment. Preserve original v1NULL results and all captures unchanged.

Architecture accepts a separately versioned analysis-v2 directory using only the
existing folded stacks/functions JSON. It accepts known target-to-leaf segments;
destruction additionally requires a known pipeline->owner-type->leaf segment.
Unknown/raw-address frames inside the required segment and max-stack hits reject
that period. Unknown upstream process ancestry is irrelevant to this relative
scope; missing frames are never reconstructed. Keep matched/accepted/rejected
weights and overlapping inclusive versus self CPU distinct. Bind the new code
and exact input hashes, verify period totals and replay fixtures before reporting
resolved subsets. These are filtered sampled CPU estimates, not complete function
costs or wall time. No new capture is required. Analysis-v2 executes one streaming
pass over existing folded files (1.72s) after12replay fixtures pass. Original
period totals, all bound inputs/top-level scripts and v1outputs remain unchanged.
Independent arithmetic/method review PASS and final design-conformance READY,
no blocking findings. The following reporting qualifications are part of acceptance.

### Reconciled ordinary-runtime stage comparison

Actual median-whole trial IDs at1/2/4/8are2/2/3/2. Values below belong to those
complete trials, not independently chosen phase medians. Other native and outer
work are explicit groups; the full27categories remain in stage-attribution.json.

| Exclusive scope |1worker s|2workers s|4workers s|8workers s|
| --- | ---: | ---: | ---: | ---: |
|Read/decode wait and source merge|5.265757|4.085228|2.742059|1.926434|
|Typed parse dispatch/join|8.191638|5.114405|3.146126|2.279400|
|Journal validation|2.163958|2.236505|2.243912|2.304927|
|Features and numeric spooling|1.640759|1.615417|1.590385|1.541109|
|Temporary release|0.663004|1.601988|1.657890|1.816282|
|Other native scan|1.590780|1.602287|1.565669|1.574183|
|Python/archive/final-check/outer scopes|3.083409|2.985694|2.961295|2.920080|
|Whole elapsed|22.599306|19.241524|15.907336|14.362414|

### Function CPU findings from accepted relative segments

The following inclusive sampled CPU estimates use the separate debug2captures,
not the ordinary debug0wall trials above. Each row excludes its rejected unknown
segments. Inherited predicates identify symbol families: Input/Observation include
Presence wrappers and input_semantics includes related closures/monomorphizations.
Event is the exact generated-function target. Rows labelled with a function name
therefore denote its observed family/scope unless explicitly exact. Parent/child
rows overlap and must not be summed. Missing periods are not zero cost; one trace
per condition cannot establish variance.

| Function/owner scope |1worker CPU s|8workers CPU s|
| --- | ---: | ---: |
|JournalView::validate|1.709|1.994|
|typed journal::validate|1.707|1.994|
|Record::validate|1.323|1.643|
|Input::validate|0.934|1.182|
|Observation::validate|0.311|0.445|
|Event::validate|0.577|0.669|
|input_semantics|0.242|0.224|
|Records destruction|0.240|0.419|
|Journal destruction|0.022|0.012|
|EpisodeWindow destruction|0.591|2.164|

JournalView's rejected segments total0.449/0.487s. Event's accepted inclusive
0.577/0.669s includes only0.068/0.070s root self; its ensure descendant alone
has0.172/0.192s self. The resolved validation path spends time in required-field/
shape checks, presence validation, string comparisons and Schema validation.
Input includes Observation/Event/Schema work; these are not separate additive
validation stages. This locates work without removing a source validation gate.

The accepted destruction union is0.853707408/2.597194368s; owner groups overlap
and their sum is not that union. In the8worker Window subtree, Value destruction
has2.160321s inclusive. Notable self samples include BTreeMap node traversal
0.460922s, Value drop glue0.314629s, RawVec current-memory handling0.246493s,
mimalloc abandoned-page handling0.226453s and multithread free-block handling
0.198397s. This points to Replay/Value-tree and allocator release work, not just
freeing the raw Journal byte buffers. The broad byte/string category cannot
separate raw buffers from strings inside those objects; raw-buffer-only cost
remains unresolved. Outside accepted owner contexts, allocator self1.729/0.719s
and clock/vdso self0.557/0.489s remain unattributed.

Total sampled user CPU is20.781562960/13.693386664s,10,370/6,833samples with zero
frameless samples. Recorder/decode logs report no loss warning; that alone does
not prove complete sampling. Short-lived workers and unwind gaps limit coverage;
the differing sampled totals do not demonstrate reduced physical CPU work.
Accepted-target union2.563126232/4.591182328s leaves18.218436728/9.102204336s
outside these selected targets, including unrelated parsing/feature/hash work,
not merely unresolved time. Exact matched/accepted/rejected nanoseconds, self,
recursion-deduplicated descendants, coverage and input/code hashes are retained
under analysis-v2/.

The actual profiling ELF is
`a21355d99f3b58bb50e0df4964970a198dd960853a0b25f3a026c7eef996c93c`, BuildID
`866d523569d9f110dfcf55f1c5188189dc1ce730`, with source d5090c41. V2's frozen
source_identity field points to source-identical debug0baseline97f1; it is a
baseline/source identity, not the capture ELF. The bound symbols-build variant,
profile freeze and BuildID evidence identify the actual debug2capture. Preserve
those original fields rather than retagging evidence.

The two nested v2subtree outputs add1,081,541bytes to the original top-level
decoded count, giving342,991,371bytes combined, still below512MiB. The entire
v2directory including code/fixtures/provenance is1,114,574bytes at audit. The
8worker trace contains2,218sampled identities versus13,568spawned decoder/parse
threads; short-lived work can be missed even without reported recorder loss.
No complete per-thread CPU coverage or physical CPU efficiency is inferred.
The offline unknown-address predicate does not recognize explicit short addresses
such as0x1234. None occur in either current function table, so current figures are
unaffected. Before reusing this diagnostic with other traces, extend that predicate
and add a replay fixture; do not silently treat this harness as a general profiler.

### Interpretation and next experiment boundary

Increasing supported concurrency improves read/parse elapsed, while Journal
validation remains about2.2–2.3s and temporary release rises from0.66s to1.82s in
the ordinary representative trials. Those serial costs now limit further gains.
Replay/Value representation or lifetime and thread reuse are evidence-based
candidates to investigate independently; visible allocator paths do not prove
that a persistent pool would improve total time. Reusing validated event prefixes
is another separate candidate requiring exact equality and preserved acceptance.
No new optimization, dependency, default setting or source-format change is
implemented by this comparison. Additional performance work requires prior notice
and a suitable load window.

## Final acceptance

Both requested investigations are complete. Independent primary metadata/resource/
arithmetic audit, profile/v2method audit and final architecture conformance are
PASS/READY with no blocking findings. Final metadata and bound evidence are in
`runs/assessments/final-worker-scaling-20261031/final-assessment.json`.
Production source, installed97f1wheel, dependencies, Config defaults, original
datasets and six fitted models stay unchanged throughout this diagnostic task.
No application regression suite is repeated for documentation/disposable-harness
changes; every actual workload still verifies the original numerical/source/
reader/cleanup contracts. No new performance command follows the later heavy-task
notification. No required investigation remains pending.
