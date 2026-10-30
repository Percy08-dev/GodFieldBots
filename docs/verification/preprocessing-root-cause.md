# Preprocessing root-cause investigation

Design accepted2026-10-30 with independent architecture READY. Investigation
precedes Rayon, task-granularity, streaming,
caching or validation-frequency changes. Production code/config/dependencies,
input semantics, datasets/models and main installation remain unchanged.

## Questions and existing facts

- Full pipeline4worker median23.595->19.786s was an episode-task change, not
  row-streaming evidence. New parse dispatch/join is4.040s; other elapsed work
  is15.746s. Holding everything else fixed, even eliminating that stage entirely
  would save only20.42%of the current total. This is an upper bound, not a fix.
- Read/Replay decode, parse and ordered consumption are separated by whole-window
  joins. Worker assignment is static by episode. Threads are recreated per window.
  These are code facts; they do not quantify spawn cost, imbalance or I/O blocking.
- Prior sampled CPU traces undercount short-lived workers and have unresolved
  vDSO ancestry. They cannot explain off-CPU time or identify all clock callers.
- Input.typed_metadata and Observation.typed_action_schema are separate decoded
  trees and are required equal by the current contract. Repetition cost and payload
  share are unmeasured. Clock/deadline checks also occur throughout preprocessing.
- Read/decode/wait is not kernel I/O wait. Resource caps, real read-call latency,
  compute/hash time, scheduler delay and parser supply/tail waits need separation.
- Reported process rusage and direct thread CPU may disagree for short-lived
  threads. Validate the counters before inferring low CPU activity or extra work.

## Bounded diagnostic design

Use private copies of accepted native sourcee604697d and the previously verified
normal release mimalloc component harness. No production edits or new dependency.
Preserve original raw admission, parsing, static assignments, joins, validators,
features, guards, hashes and failure/deadline behavior. Add diagnostic hooks only.
Record exact source diffs, flags, binaries, input hashes and output golden checks.

1. Audit existing full/component medians and phase scopes, per-worker busy time,
   source sizes and host affinity/cgroup/scheduler/pressure availability read-only.
   Missing counters are unavailable, not zero or proof of no wait.
2. Collect bounded direct timestamps and thread CPU for actual windows, worker
   lifetimes and jobs. Record spawn request/return, thread entry/exit, job start/end,
   rows/bytes and coarse read/parse/consume stage boundaries. Compute stage-wide
   gaps with no active jobs, launch ramp, completion tail, job-size/duration skew
   and CPU-versus-wall differences. Worker elapsed overlaps; do not add it to wall.
3. Split file reading into metadata/open, actual read calls, buffer extension,
   SHA update and final evidence checks. Aggregate per worker; do not log every
   read or record with a shared lock. Read-call wall includes syscall compute and
   scheduler effects; wall minus thread CPU is off-CPU, not automatically I/O wait.
   Aggregate native numeric read/write call wall/count/bytes when local hooks
   suffice; otherwise retain their explicit unmeasured scope. Source reads alone
   cannot rule out waiting in numeric spooling or archive work.
4. Count budget::monotonic/Control checks using thread-local counters and map them
   to coarse stages. Do not remove, throttle or bypass any check. Use existing
   on-CPU profile evidence alongside counts; counts alone are not elapsed cost.
5. In a separate component payload-analysis mode, stream Serde field sizes/hashes
   outside parse timing. Quantify Input field sizes, the two schema copies/equality,
   distinct schema hashes/frequencies and repeated catalog payloads. These are
   normalized serialized bytes, not raw-file byte spans or an optimization oracle.
   Do not rebuild Value trees, cache/replace a subtree or change file formats.

Diagnostics must avoid a contended per-job/per-record global log lock: collect
thread-local bounded events and merge at worker/stage boundaries. Bound trace
memory (at most64MiB), account it within existing2GiB admission or explicitly
report separate diagnostic memory, and fail/report incomplete capture on overflow.
Use no artificial start barriers. Emit only ordinals, counts, timings and hashes,
not policy-visible features, raw private payloads or new training data.

Thread CPU uses CLOCK_THREAD_CPUTIME_ID or equivalent RUSAGE_THREAD, with voluntary/
involuntary context-switch counts if useful. Read Linux scheduler counters only
when available/enabled; do not change sysctls, quotas, affinity or system tracing.
Unresolved runnable-versus-blocked time stays explicitly unresolved. A profile or
trace requiring unavailable kernel facilities is optional, never fabricated.
Also record CLOCK_PROCESS_CPUTIME_ID and the Python wrapper's process_time_ns
alongside existing rusage. If accounting disagrees, use at most three short pairs
of identical fixed-work pthread runs toggling only per-thread CPU-clock reads.
This tests observability/API effects, not a worker-pool optimization.

## Execution and acceptance

Calibrate diagnostic overhead on fixed legitimate inputs: two interleaved
baseline/instrumented runs for the actual4worker full pipeline, and two pairs
for the16worker fixed-read4/window32 component. At most one additional component
trace each at8and32 and one payload-analysis run. If instrumentation materially
perturbs timing, reduce coverage and label its sampling scope before interpreting
absolute timings. No new worker-pool, row batching or optimization A/B is authorized.

All runs preserve original NPZ/array/sample/dataset proofs (full pipeline) or
parsed/source digests (component), counts, original ordering and cleanup. Use the
same2GiB/cache512MiB/chunk16MiB settings and the unchanged owned observer. Root
coordinates serial source jobs; no heavy work runs concurrently with captures.
Async Python verification needs permitted socketpair wakeups; synchronous
component/full-workload paths use their previously verified environment.

Root owns design/docs/orchestration and conclusions. Implementer owns only
`runs/assessments/root-cause-20261030/` diagnostic copy/harnesses. Independent
architecture review checks the instrumentation design/fidelity before dependent
code; independent correctness review checks traces, arithmetic, overhead and
claims. Report confirmed causes, quantified contributions/upper bounds, remaining
unknowns and the next smallest causal test. Do not present a proposed optimization
as a diagnosis or declare pool/streaming/root schema changes accepted here.

## Results

Read-only evidence is recorded in
`runs/assessments/root-cause-20261030/existing-evidence.json`. It confirms16CPU
affinity, current init.scope unlimited quota, current nr_throttled=0, disabled
kernel scheduler statistics and ext4 repository mount. These are current snapshots,
not retrospective attribution of earlier trial stalls. CPU samples do not measure
off-CPU time. The15.746snon-parse time is observed remaining elapsed work, not
a measured optimization benefit; eliminating parse would leave that lower bound.

The first payload-analysis attempt exhausted its diagnostic distinct-hash count
cap, not the workload budget. Retain it as failed diagnostic evidence; one compact
digest-key/shared64MiB-bound retry is authorized, with no source/cache/algorithm
change. Tiny signed-negative CPU complements in independent clock measurements
are unavailable and reported with count/magnitude rather than clamped to zero.

### Executed evidence

All11root-owned captures pass their original output proofs and final source/
production-code/model/main-install checks. Four actual full-pipeline captures
(two baseline/two traced) match the original83,783-row arrays, NPZ/sample/dataset
identities. Seven component controls/captures preserve all13,568Journals,
534,640records and parsed/source digests. Six completed traces pass interval and
arithmetic analysis. Artifacts, exact conditions, versions, raw data and summaries
are under`runs/assessments/root-cause-20261030/`, especially`captures/`,
`findings-summary.json`, `preflight-payload-v2.json` and`existing-evidence.json`.

Production native source remains`e604697d...`. Full diagnostic wheel is
`9bc01b36...`/source`4ef104f0...`; final component is`8fea80f8...`/source`a40b9e25...`.
The original component trace and its earlier binary/source remain bound under
`v1/`. Release opt3/debug0, mimalloc and Cargo dependencies remain unchanged.
No Rayon, task-granularity, cache, validation-frequency or production edit occurs.

Native scan calibration is17.491/16.976s baseline and17.524/17.312s traced:
median difference+1.07%, only two observations each. Component16parse off is
1.949/1.870s and on1.843/1.887s (-2.32%median difference); this small sample and
scheduling variation do not establish a benefit from tracing. First full baseline
has larger post-native work (whole23.426s versus19.872s for the second), so no
precise whole-command overhead claim is made. Flushes are separately timed after
the unchanged workload. Full traces retain about40.4MiBdiagnostic capacity,
separate from unchanged2GiBworkload admission, below64MiBdiagnostic cap.
All full-pipeline captures use the public-history feature recipe; the
observation-only recipe was not newly remeasured in this investigation.

### Finding1: whole-workload idle is not just the parser

The accepted uninstrumented4worker trial is19.786s, of which4.040s is parsing
dispatch/join and6.768s is ordered feature/spooling plus post-parse validation.
Other work is not overlapped with parsing. Faster parsing alone cannot remove
the15.746sremaining elapsed work. This is a conditional lower bound with other
work fixed; it does not rule out changing the pipeline graph in a later task.

The two new traces confirm1,696separate read windows and1,696parse windows,
followed by ordered consumption. Parser job-wall occupation corresponds to
2.168/2.156simultaneously active jobs on average out of4configured slots.
This is time spent within job spans, not physical CPU utilization.

| Parse scheduling observation | Full trace0 | Full trace1 |
| --- | ---: | ---: |
| Parse-stage wall s |4.074|4.069|
| No-active-job wall s |0.243|0.237|
| No-active-job fraction |5.96%|5.83%|
| Parent spawn-call wall sum s |0.332|0.329|
| Median window max-worker busy / mean-worker busy |1.525|1.519|
| Correlation of slot input bytes with busy wall |0.974|0.837|

Parent spawn calls overlap worker execution and are not additive startup cost or
a bound on a future pool's benefit. Most parse-stage time has at least one active
job, while many slots are unoccupied. Variable episode work and static assignment
are directly relevant; this does not quantify how much each could be removed.
At16/32component workers, mean active jobs are about6.96–6.99/7.62 and the median
max-worker/mean-worker busy ratio is1.91/2.87. More configured slots alone do not
keep those slots supplied. These component windows differ from production ones.

Full parse jobs range from7to493records and15,568to3,626,719input bytes, with
medians67records/245,602bytes. This is substantial variation in the existing job
unit. All bytes/rows cover both participant Journals for one episode.

### Finding2: warm source-read calls do not dominate the captured work

In full trace0,30,644Source::read attempts transfer3,237,466,907verified bytes.
The following are summed inclusive file-substep wall durations across threads,
not a partition of the full command or physical device traffic:

| Source read substep | Summed seconds |
| --- | ---: |
| Metadata/open |0.357|
| Actual Read::read calls (145,583calls) |0.186|
| Extend raw buffers |0.073|
| SHA updates |1.617|
| Final evidence checks |0.100|
| Other source-read work |0.118|

Actual read-call wall includes kernel compute and scheduling, so it cannot be
called pure I/O wait. Within those captured calls it is an upper bound on their
blocked waiting, not proof of all storage waiting. Metadata/open and other paths
can also block. Captured numeric buffered read/write logical calls total0.055/
0.585s; opens/flushes/proof/identity/Python I/O are outside that aggregate.
The captured warm source read calls cannot by themselves explain several seconds
of parser/consumer inactivity. The earlier cold preflight had much longer read
calls and is not substituted into these warm results. No global or host I/O-wait
attribution is claimed.

### Finding3: feature-stage control calls dominate; history is repeatedly scanned

Both full traces count143,785,761production Control::check calls and143,785,763
budget::monotonic calls. Feature/numeric-spooling code accounts for129,631,653
checks, about90.16%of all checks; Journal parsing accounts for2,101,971.
Diagnostic clock reads are excluded from these counts.

For the public-history recipe, the concrete source at
`native/src/typed_journal/features.rs` walks the same
bounded history once for each of24EVENTS kinds, calling control.check for each
event on each pass. Role aggregates and definition hashes add further passes.
Prepared common/history features are already reused per decision, so this is
not a claim that they are regenerated per candidate or feature names per row.
The24passes and high polling frequency are work-count facts. Ninety percent of
calls is not90%of CPU time, and counts alone do not measure clock cost. No check
was removed/throttled and no aggregation algorithm changed in this investigation.
Counters localize calls to the whole feature/numeric-spooling stage, not individual
call sites. The history loops are a confirmed contributor; their precise share
of those129,631,653calls was not separately measured.

### Finding4: serialized input carries substantial repeated structures

The corrected payload diagnostic analyzes169,168Inputs. Under direct normalized
Serde serialization:

| Field | Serialized bytes | Share of whole normalized Input |
| --- | ---: | ---: |
| Both schema copies together |642,874,656|40.00%|
| Events |595,388,011|37.05%|
| Catalog |132,445,557|8.24%|

Both schema copies are equal for all169,168Inputs, have identical ordered
size/digest streams and only3distinct values. Each copy alone occupies20.00%of
normalized Input bytes. The current decoder constructs both trees and shape
validation visits both. This is actual redundant representation/work, not a
measured40%runtime saving. Observation as a whole includes its schema and must
not be added to that schema size again. The catalog has99,406distinct whole-value
hashes, so it is not established as a tiny globally constant structure.

Read-only producer audit confirms a separate issue: the environment appends new
visible events to a per-player history, passes the accumulated history through
decision_input, clones it, and serializes it into every worker/journal decision.
See `src/gf_simulator_client/environment.py:154,174`,
`src/gf_common/validation.py:235`, `src/gf_simulator_client/workers.py:76` and
`src/gf_training/journal.py:80`. Repeated-prefix serialization is a code fact;
with growing histories it can grow quadratically in episode length. The actual
prefix-overlap/avoidable-byte fraction and scaling of this corpus were not
measured. `history_end` is the opaque feed cursor, not a validated history checksum.
Normalized field bytes are not original file-byte spans, CPU costs or permission
to omit required data/validation. Stored schema4 sources remain authoritative.

### CPU observability remains a separate uncertainty

The fixed-work C microprobe uses16short pthreads per window for32windows, with
identical compute/checksum in three fresh-process pairs. Toggling only worker
thread-CPU-clock endpoints changes reported process CPU from49.95–55.74ms to
137.12–137.56ms while wall remains36.58–38.22ms. RUSAGE_SELF and process CPU clock
agree within each mode. This is API-sensitive measurement under fixed compute;
it does not distinguish actual clock-call CPU cost, scheduling/frequency effects
and accounting behavior. No kernel bug, exact physical utilization or true I/O-wait
fraction is inferred. Wall-minus-CPU differences are not validated as physical
off-CPU fractions while this API sensitivity remains unresolved.

Three original component-oracle intervals have CPU exceeding wall by109–295ns;
their raw endpoints remain retained and complements are unavailable, not clamped.
The four newer component traces add25such unavailable complements,8–270ns,
in oracle/read-job intervals. Their valid wall intervals remain usable.
Full traces report no such discrepancy. Disabled scheduler statistics remain
disabled. Current quota/pressure snapshots cannot explain historical stalls or
separate runnable-versus-blocked time for every job.

### Corrections and conclusions

Review identified two diagnostic-only issues. Full-wheel DROP_DTO measured only
the emptied result-container release after real per-episode DTO destruction;
actual DTO destruction is inside ordered_consume and is not separately timed.
The component's DROP_DTO does measure its actual DTO destruction. Reports use
those per-target scopes rather than retagging intervals.

The captured Python wrapper could overwrite an active workload exception if
diagnostic publication itself failed. Happy captures are unaffected and retain
their exact old wrapper under`captures/workload_child.py` (eb8bf2e...). After
captures, a private finally-path fix (c4ac23ae...) passes24isolated fixtures and
Ruff; the independent reviewer confirms exact original-exception preservation.
No workload was recaptured or retroactively assigned the repaired wrapper hash.

Confirmed priorities for a subsequent decision are repeated history traversal/
polling, repeated schema/history representation, and coarse stage/task supply.
No Rayon, pool, row-streaming or data-format change is adopted as a remedy here.
Smallest causal optimization tests require a separate design: preserve exact
feature bits, source contracts, fail-before-publication and bounded ownership.
The current evidence identifies work and waiting structure; it does not allocate
the entire wall time to one cause or promise an optimization's benefit.

Independent numeric/causal-scope audit PASS and architectural closing conformance
READY. The reviewer-required distinction between whole-stage call counts and the
unmeasured history-loop share is incorporated. Both diagnostic source findings
are resolved with explicit DTO scope and the tested wrapper repair. No unresolved
review findings remain. This closes investigation evidence, not an optimization
or learning-quality acceptance gate.
