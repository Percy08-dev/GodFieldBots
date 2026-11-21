# Comparison telemetry: closed measurement contract

Status: scoped engineering implementation accepted after one repair round.
Both independent correctness/conformance rereviews are READY. Actual merged
measurements, profiling, calibration and default adoption remain separate gates.
Architecture READY on proposal SHA
f77fbe2a0f8e0140e4d12651c688d3a902f21e3976a458712ec80bb4de03ed60.
Implementation reviews, actual measurement and adoption remain separate gates.
Owner: root. Process observer and pure CUDA estimator are separately accepted.
No policy/learning/Source/termination authority changes. The approved workload,
run order, budgets and3% adoption rule remain unchanged.

## Enablement and dependency

Host opt-in is the validated pair GF_BOTS_PERFORMANCE_DIAGNOSTICS_DIR and
GF_BOTS_PERFORMANCE_OBSERVATION_ID. The observation ID is nonempty valid UTF-8
text under the existing text contract, preserved exactly as metadata. It never
forms a pathname; no ASCII-only grammar or arbitrary length cap is imposed.
Both absent disables all new histograms, timestamps and files. Partial/invalid pair refuses measurement setup BEFORE
Simulator/GPU work. The experiment launcher creates a fresh absolute owned
output directory; actual hosts/cold continuations reopen that same directory.
Path/ID never enter observations, candidate/feature tensors, seeds or policy
routing. Filter them from external Simulator and unnecessary child environments.

Add a pure gf_common/performance_metrics.py for fixed histograms/counters/scope
values; an I/O edge gf_simulator_client/performance_diagnostics.py parses opt-in,
writes exclusive diagnostic files and phase events. Training depends on the edge;
models receive only pure counters or an explicit optional diagnostic callback,
never import the client/training layer. Native player_runtime/metrics.rs stores
local game/dispatcher metrics without Python callbacks. No new dependency.

Legacy shared participant gets only an explicit diagnostic timing BOOL through
its existing sanitized worker environment/bootstrap, not path/ID. Add exactly
optional shared_numeric_ready_ns to the existing closed timing metadata validators.
It is CLOCK_MONOTONIC nanoseconds captured after numeric_row construction, before
channel.request. Parent records successful RTT only after its existing complete
identity-validated reply (including sampling) returns, before Act. Missing/invalid
timing makes diagnostics incomplete, not the action invalid. Native RTT starts
after ServingRow numeric buffers are ready, before dispatcher admission, and ends
when the game receives its accepted response before Act. Failures/expiry/cancel/
rejection have separate counts, never successful latency samples.

## Clock, phases and partitioning

Use CLOCK_MONOTONIC nanoseconds and recorded Linux boot ID in every producer.
No phase is inferred from Prepared (no timestamp) or Exit wall time. Explicit
start/end events delimit startup, collection, update, save, retirement, cold,
evaluation and idle. At the common coordinator, update brackets the actual
update RPC/Source.consume, including checkpoint production; save brackets
durable generation commit/retention. These are coordinator phase durations,
not numerical-kernel time or checkpoint-only file-save time. Existing
learner update_seconds, when present in actual saved metrics, is a separate
numerical-call wall measurement and is never inferred from the RPC interval.
Supervisor captures the original Child.wait monotonic instant, then may attach
the actual durable Exit SHA; never substitute a later poll time.
Observer actual-command wall remains independent of sampler/drain time.

Begin/end one collection scope only at actual positive quiescence; first start
must be clean too. Keep only current scope histograms plus one bounded flush,
write/reset before next collection. Failure flushes active scope as incomplete;
flush loss is missing evidence, no scientific authority or cleanup grace. Native
runtime measurement_begin(scope)/measurement_end(scope) control methods return
DATA snapshots outside policy inputs; begin/end require its existing actual wave/
request/producer quiet checks, not queue0 alone. Persistent Python stage counters
snapshot at the same collection boundary; idle poll calls remain lifetime/idle
counts, not falsely assigned request RTTs. No unbounded map of historical scopes.

Legacy SharedServingController.finish_collection uses actual generation/batch and
service counter window after quiescence. Add diagnostic BEGIN/END control-pipe
commands with that scope, no per-request scientific routing extension. Existing
METRICS_RESET positive-empty boundary remains the ownership basis. A diagnostic
processing/snapshot error on a healthy, fully framed channel marks observation
incomplete without cancelling valid policy work. Actual shared-transport/framing/
timeout/health uncertainty keeps its existing infrastructure quarantine, as
[the reviewed clarification](rust-player-runtime-diagnostic-failures.md) specifies.

## Metrics and histogram representation

Event latency is LOGICAL validated Events PAGE RPC, including its internal
transport retries/reattest; record retry/attempt/failure counts separately. Start
before page dispatch and finish after its existing protocol/page validation.
Eventless pages count. Both Rust and legacy use this definition; do not label
an outer retried page call as a single HTTP attempt or include other endpoints.
No duplicate view/token validation is added for instrumentation.

Histogram recipe ns-exp2-sub32-v1: bin0=zero; positive u64 x has e=floor(log2 x),
s=floor((x-2**e)*32/2**e), index1+32e+s. There are2049 fixed count slots; small
exponents use only reachable exact integer slots. Store
{recipe,count,counts,overflow_count,min_ns,max_ns,sum_seconds}. count includes
all observations, sum(counts)+overflow=count; min/max concern finite observations
or null when none. Rank=ceil(p*count) for p95/99. Report the chosen finite bin's
lower-inclusive/upper-exclusive bounds, final upper2**64. A rank in overflow
reports [2**64,infinity), never a fabricated finite quantile. Negative/invalid
clock intervals mark incomplete. Merge only equal recipe/unit/scope definitions.
No raw unbounded request-duration list or per-event file write/fsync.

Actual batch-size map counts once per native issued nonempty batch. For legacy,
retain FULL-window response-size counts Rb (not rolling4096 samples), derive
Fb=Rb/b ONLY with positive quiescent start/end, all authorized requests covered
exactly once, no foreign/duplicates and0 failures including cancellation/expiry/
rejection/transport/service. Divisibility is only a consistency check. Keep Rb,
Fb and coverage facts; missing facts make batch histogram incomplete. No new GPU
forward IDs or unbounded dedup ledger. Configured B or request-weighted samples
alone are not an actual batch-frequency histogram.

Storage measures producer byte-capacity wait and admitted-enqueue through actual
service dequeue/dispatch, plus queue/outstanding/in-service bytes/depth peaks.
Native/thread FIFO and legacy serialized parent-RPC are labeled distinct queue
models. Legacy parent dispatch age excludes pipe/child scheduling; never relabel
combined write/DB-ACK seconds as queue age. Also record the existing collection
serialization wait separately when it precedes writer admission. No capacity
formula or reservation margin changes.

PyO3 counts are actual gf_native player-runtime boundary TOTALS with an explicit
hot-method split: next_batch, submit_results and all OTHER methods aggregated.
Polls/failures count. Do not claim a complete per-method breakdown; the total is
the sum of those three supported counters. New-mode
existing assembly/CPU-share/H2D/forward-fence/sampling/return sums are synchronized
HOST wall stage timings, not CUDA event durations. Unsupported legacy stage
values are null, never0 or derived from RTT. Candidate counts/cutoff reasons come
from actual persisted J6 decisions and preserve participant/collection identity.

## Closed diagnostic records

metrics1/kind rust-player-phase-metrics-v1 required keys
{schema_version,kind,observation_id,owner,clock,scope,start_monotonic_ns,
end_monotonic_ns,complete,errors,counters,histograms,batch_rows,stage_seconds,queue}.
owner={pid,start_ticks,role}; clock={name,boot_id}, name=CLOCK_MONOTONIC.
scope={run_id,operation_id,invocation_id,batch_id,phase}; batch_id null outside
collection, IDs null only where genuinely not yet assigned. role is host,
native-runtime, legacy-coordinator, legacy-shared-broker, legacy-gpu-worker, legacy-participant,
native-supervisor, native-inference or writer. phase is startup/collection/update/save/retirement/cold/evaluation/idle.
errors is a string array; complete is bool. Counters are exact known keys with
nonnegative integer values or null when unavailable:
{event_page_successes,event_page_failures,event_transport_attempts,
inference_issued,inference_successes,inference_cancellations,inference_expiries,
inference_rejections,inference_failures,pyo3_next_batch,pyo3_submit_results,
pyo3_other,writer_admissions,writer_admission_failures}.
histograms has exactly {event_page,inference_roundtrip,storage_admission_wait,
storage_queue_age,collection_storage_serialization_wait}, value is histogram or
null. batch_rows={method,request_size_counts,batch_size_counts,issued_requests,
completed_responses,quiescent_start,quiescent_end,failures}; unavailable values
are null. method=native-issued-batches-v1 or complete-response-size-inversion-v1.
stage_seconds has {assembly,cpu_share,h2d,forward,sampling,return,native_wait,
native_submit}; each value is null or {method,seconds}, seconds finite>=0.
queue={model,queued_bytes_peak,in_service_bytes_peak,outstanding_bytes_peak,
depth_peak,failed_attempts}; model=thread-fifo/native-ingress/legacy-parent-rpc,
unavailable values null. Byte counts are observational ownership, not extra PSS.

Native-inference collection metrics can include `native_inference`. The field
records `recipe`, `batch_width`, `batched_groups`, `useful_rows`, `dummy_rows`,
`scalar_rows`, `useful_size_counts`, `cuda_peak_allocated_bytes`, and
`cuda_peak_reserved_bytes`. The histogram counts actual Torch forwards by
useful row count. The fixed width times the number of forwards equals the
sum of useful and dummy rows. Scalar summaries have null width and zero
batched counts. The CUDA peaks measure Torch allocation and reservation over
the collection scope. CPU summaries have null peaks. Diagnostic readers
accept historical metrics without this field.

For `batched-torch-graph-v1`, `native_inference` also requires
`graph_replays`, the number of successful CUDA Graph forwards included in
`batched_groups`. It is an integer from zero through `batched_groups`. A
positive count requires `batch_width = 32`. The scalar and earlier batched
recipes omit `graph_replays`.

phase1/kind rust-player-phase-event-v1 exact
{schema_version,kind,observation_id,owner,clock,scope,monotonic_ns,event,links}.
owner/clock/scope as above; event=phase_start/phase_end/host_reaped/prepared_written/
completion_written. links={operation_sha256,prepared_sha256,exit_sha256,
completion_sha256}, each actual known SHA or null. No backwards record rewrite.
Files are exclusive per actual owner/scope; native returns data through PyO3,
not the64KiB authority socket. Missing/faulted producers remain incomplete.
Normative run/proof/ACK schemas do not acquire arbitrary diagnostics fields.

## Separate GPU profile observation, never primary timing

Installed TorchRL source puts policy_factory in the spawned process entry thread,
then H2D/model/D2H on InferenceServer-worker. Starting profiler in the factory is
NOT sufficient. An optional diagnostic callback object is created at the client
edge and passed to ServingPolicy; its first post-forward invocation starts Torch
profiler on the ACTUAL worker thread, a later post-forward closes/exports it.
The bounded interval explicitly includes previous batch D2H, current batch H2D
and current forward. Tag model-forward ranges; report exact interval/counts, not
same-batch RTT or whole-invocation GPU totals. If the required callbacks never
occur, profile is incomplete. No third-party monkeypatch/private loop replacement.
New single-host mode uses equivalent labeled worker-thread intervals.

Paths/IDs reach only this explicit diagnostic callback, not network features.
Profiler/trace I/O errors do not confer authority; CUDA execution errors retain
existing failure/poison behavior. Profiled runs are a separate population with
matching configuration/source and their extra synchronization/CPU cost disclosed.
Primary24 timing runs use no profiler. No H2D/kernel value is inferred from sums,
CPU enqueue time, allocator snapshots or broker acceptance-to-forward samples.

## Observation bias and verification

Before any adoption, Root predeclares bounded enabled/disabled actual-workload
calibration for legacy and both schedulers at relevant concurrency. Preserve
sampling duty/sweep spans, observer CPU, application wall/collection rates,
candidate counts and cutoffs. No modeled subtraction or invented overhead cap.
Unresolved bias material to3% prevents adoption. Formal matrix/order/criteria stay
unchanged and failure populations remain separate.

Tests cover histogram/rank/overflow/merge, batch inversion FULL coverage,
zero/failed waits, scope drain/reset/idle attribution, actual ready-to-accepted
RTT/stale responses, correct page retries, diagnostics-disabled path, file/flush
errors, no policy/RNG/math effects, logical versus physical clocks and bounded
worker-thread profiler callbacks. Actual small W1/W2 observations confirm useful
samples before the full matrix; they do not replace it. Independent architecture
review is required before code, then correctness/conformance and all Task6/7
gates remain required for acceptance.

## Closed profiling enablement, callback and result

Primary runs explicitly record profile_recipe=off and leave
GF_BOTS_PERFORMANCE_PROFILE absent or off. A SEPARATE diagnostic run may set it
to forward-window-v1 ONLY with the valid base diagnostic directory/ID pair; other
values fail setup. The manifest records the explicit population/recipe. No
profiler object, imports, callbacks or trace file is created in off mode.

The callback data signature is after_forward(*, scored_rows: int, boundary: str).
It runs ONLY after a successful model-forward boundary, on its ACTUAL execution
thread, counts callback ordinals locally and never receives policy data. Boundary
is legacy-post-transport-forward (scored_rows=actual transport batch size) or
native-post-row-forward (scored_rows=1). These are distinct, labeled windows; do
not pretend they are the same-batch RTT. Target is the first collection scope.
After ordinal1 the callback starts profiler on that same worker; after ordinal3
it stops/exports. Thus the bounded recording interval includes the preceding
output transfer and the next two input/forward portions, with actual callback/
scored-row counts recorded. Do not change ordinals after seeing measurements.

The first callback binds actual worker PID/TID/clock/boot/scope. A changed execution
thread, model failure, callback/profiler/export failure or fewer than3 callbacks
makes the observation incomplete. Retain actual status in a bounded diagnostic
status object/shared block so the coordinator can emit an incomplete result after
positive owner reap. Never submit fake inference to obtain another callback,
stop a thread-local profiler from an unproved foreign thread or add cleanup grace.
The bounded diagnostic holder/shared block publishes explicit target-scope
BEGIN/END status to the callback. END before post3 is TERMINAL incomplete even
if later collections continue on the same worker. A later callback may only
stop/clean that INVALID active window on its actual thread; it cannot contribute
an ordinal, qualify a trace spanning update/publication, restart the target or
submit synthetic work. The scope-end rule applies immediately, not only at owner
reap. A still-active profiler when the actual owner exits is lost/incomplete
evidence.
Underlying CUDA execution failure still poisons the normal runtime; diagnostic
I/O failure alone does not upgrade or invalidate otherwise valid policy actions.

profile1/kind rust-player-forward-profile-v1 exact keys
{schema_version,kind,observation_id,owner,clock,scope,recipe,enabled,boundary,
worker_tid,start_after_forward,stop_after_forward,callbacks_seen,scored_rows,
start_monotonic_ns,end_monotonic_ns,complete,errors,trace_path,trace_sha256,
trace_bytes,trace_clock,torch_version}. Recipe=forward-window-v1, enabled=true,
start_after_forward=1/stop_after_forward=3. scored_rows is the THREE callback
boundary row counts [rows1,rows2,rows3] (null for missing callbacks): input/forward
portions cover rows2+rows3, output transfers cover rows1+rows2. Do not assume these
counts equal for dynamic transport batches. Timestamps/trace
fields are null when absent, not0. owner/clock/scope match the common families;
GPU process uses legacy-gpu-worker, same-host service uses native-inference.
trace_clock=torch-profiler-native: its raw event clock is NOT silently equated
to CLOCK_MONOTONIC; use the explicit host window/tagged ranges and event durations.

Trace output uses a fresh actual owned file descriptor/path with original-file
custody through export/hash; verify published type/inode before accepting it.
Do not reopen a replaceable symlink as evidence. The result stores actual relative
trace path/bytes/SHA, exact pinned Torch version and errors. No source/data/
qualification authority is inferred from this diagnostic family. A missing or
corrupt result cannot become a primary measurement or a zero GPU-stage value.
