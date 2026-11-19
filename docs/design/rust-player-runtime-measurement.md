# Rust player runtime comparison measurements

Status: bounded telemetry engineering implementation accepted under both
independent reviews; actual calibration/performance/adoption still pending.
Architecture READY on proposal SHA
6b8942492145779076a079fd2aa3982fd08a115348e452425d1f731fd220eb0e.
The accepted closed telemetry contract below supersedes initial open decisions.
Owner: root. Applies after Source/common-PPO and public-entry acceptance, never
as a substitute for those gates. The [approved plan](rust-player-runtime-approved-plan.md)
and [runtime design](rust-player-runtime.md) retain their workload and adoption rules.

## Objective and responsibility

Measure the actual public ordinary/HPO invocations under the predeclared C8,
C16 and deployed-concurrency comparison. Record games and all-seat decisions per
collection, whole invocation wall, real batch sizes, latency distributions,
PyO3 calls, CPU/threads, GPU stages, PSS and storage backpressure. Diagnostics
must preserve candidate order, clocks used by search, RNG, Act acceptance,
Source eligibility and learning mathematics. They grant no execution, Source,
physical-termination or scientific-parent capability. Failed observations stay
failed or incomplete; missing values never become zero.

Native HTTP/game/dispatcher owners collect their own counters without Python
callbacks. The existing inference owner records actual Python stage/call/batch
observations. Existing Python HTTP/shared-worker/writer owners collect equivalent
old-mode observations. A standalone std-only Rust observer under tools/ launches
and waits the actual public command and samples its process tree. It imports no
Python and adds no runtime/development dependency. Existing Python benchmark
observers remain historical tools; they cannot remain alive during new-mode
training. Root chooses configurations/order and interprets evidence; diagnostics
never adapt scheduler, budgets, admission, batch formation or workload.

## Collection and distributions

Diagnostics are explicitly enabled for a measurement invocation, with one fresh
absolute output directory and an observation ID recorded in the experiment
manifest. The host enablement is the pair GF_BOTS_PERFORMANCE_DIAGNOSTICS_DIR and
GF_BOTS_PERFORMANCE_OBSERVATION_ID, parsed once before game/GPU work. Missing both
disables observation; a partial or invalid pair fails measurement setup. Neither
is a policy setting, checkpoint value or training-recipe field. The directory is
created freshly by the experiment launcher; hosts reopen that SAME owned
directory, without requiring each continuation to create another fresh path. Normal unobserved execution avoids
histogram allocation and hot-path timestamp calls. Both controls and candidates
use the same enabled definitions and external observer schedule.

Durations use each owner's monotonic clock and are accumulated locally. There
is no per-event file write, fsync, cross-language callback, raw observation/view
logging or unbounded per-request sample list. A fixed integer histogram stores
nonnegative nanoseconds: zero has its own bin; positive u64 durations use a
power-of-two exponent and 32 equal subintervals within that power of two (exact
integer bins for exponents below five). Each observation increments count, sum,
min, max and its bin. A duration above representable u64 nanoseconds increments
an explicit overflow count. The record includes the exact bin recipe/version.
The summary reports nearest-rank p95/p99 as the selected bin's inclusive lower
and exclusive upper bounds, with count and overflow; it never claims exact
sample quantiles. This bounded diagnostic storage avoids workload-length growth.
A small exact batch-size frequency map records actual issued nonempty row counts;
its cardinality is bounded by the actual concurrent pending requests.

Metric definitions:

- event_http/event_page: one LOGICAL Events page RPC from dispatch through existing
  response/page validation, INCLUDING internal transport retries/reattest; retry/
  attempt/failure counts are separate. Eventless successful pages count. This is
  not an individual HTTP attempt, an entire multipage feed, time since a visible
  game event or another endpoint's latency. Legacy and Rust share this definition.
- inference_roundtrip: a participant request with its serving numeric input ready,
  immediately before dispatcher/shared-service admission through successful
  identity-validated accepted response available to the game, before Act. It
  includes queue admission, formation, transport/service and CPU sampling. Record
  cancellation/expiry/rejection counts separately and never place failed requests
  in the successful latency distribution. Sampling-state acceptance remains
  owned by the unchanged transaction. The precise old/new input-ready sites and
  any transport difference must be documented with source references.
- batch_rows: actual rows issued to the inference consumer, once per batch, not
  configured B and not one observation per row. Record empty polls separately.
  Rust's native dispatcher is its only batch owner; old mode uses its actual
  existing shared-serving batch dispatch, without adding a second batcher.
- storage_admission_wait: producer elapsed waiting for queue capacity, measured
  at the existing admission wait boundary; zero-wait attempts count. Track full
  attempted/successful/failed counts. storage_queue_age is successful enqueue
  through dequeue by the actual writer. Queue bytes include only their documented
  owner lifetime; queued and in-service bytes are separate. Record high-water
  queued bytes and queue depth, not a new restrictive capacity formula.
- GPU/CPU serving stages reuse actual assembly, CPU sharing, synchronous H2D,
  forward/fence, CPU sampling and return measurements. Name their scope as
  synchronized wall durations, not CUDA event kernel durations. If a separate
  CUDA-event diagnostic is necessary, identify its extra synchronization/overhead
  and keep it out of the primary timing population. No additional per-row fence.
- PyO3 counts are actual gf_native player-runtime boundary totals with named
  next_batch/submit_results and aggregated other-method counts, including polls
  and failures, not estimated from decisions or batch count. Search candidates,
  real-time cutoffs and reasons come from actual persisted per-decision J6 data,
  preserving first/second collection and participant roles.

At actual owner retirement, each producer writes one diagnostics-only JSON file
using an exclusive unique name and returns its file hash to the measurement
manifest. A missing or failed owner flush makes that owner's metrics incomplete;
no authority record is forged to repair it. Native snapshots return data through
the existing facade/control edge, never inside policy input. New diagnostic
record fieldsets and the exact accessor must be listed before implementation.
Existing normative run/proof/ACK schemas do not gain optional arbitrary fields.

## External observer

The std-only Rust tool spawns the actual .venv/bin/gf-bots command (or exact
approved installed entrance), then retains its original Child and measures
CLOCK_MONOTONIC wall from immediately before spawn until its actual wait result.
Public CLI exec into the Rust supervisor preserves this original PID. The tool
has no admission/lock/prepared/exit authority and no subreaper or process-group
signalling role. Existing run/supervisor absolute deadlines own termination.
The observer must not extend them or launch a Python orchestration parent.

Sample process identity as (PID, /proc start time), discover descendants through
current ancestry and retain previously observed identities while alive, so PID
reuse cannot inherit attribution. Sample CPU ticks and actual thread counts at
100 ms; sample smaps_rollup/PSS at 1 s, matching historical observer cadence.
Log actual sampling times and unavailable/racing reads. Treat each PSS read as part of a timestamped sweep with per-read/sweep span and
identity/race coverage, rather than a truly simultaneous snapshot. Sum the
proportional memory readings from the same sweep for the owned tree; do not add shared recording-account
bytes, RSS totals or per-process historical peaks. A sample with a missing PSS
member is explicitly partial and cannot certify a strictly lower whole-tree
peak. Report peak PSS, per-role samples, observer CPU/wall overhead and coverage.
Sampling may miss short processes or peaks; preserve that limitation.

Record executable/cmdline identity and observed TIDs/names for role attribution;
configured W/I/O/writer/Torch counts are separate from observed OS threads.
The exactly-one-Python claim applies during new-mode training, before actual
host reap. Cold audits/evaluation after reap are separate phases. The observer
itself is explicitly recorded and excluded from workload resources. Diagnostic
CLOCK_MONOTONIC phase events delimit collection, update, warm startup and cold
finalization. Accepted Prepared has no timestamp and Exit finished_unix_ns is wall
time; neither is used to infer monotonic phase boundaries. Events carry boot ID,
clock identity, invocation/operation/collection identity and applicable receipt
hash links. Normative receipts remain unchanged. Capture actual original-host
wait time at its existing Rust owner and attach the durable Exit hash afterwards;
do not substitute polling time. No Python observer is introduced. Whole command
wait time is captured independently of sampler polling/drain latency and includes
reentry and CPU cold validation.

A standalone observer is chosen instead of coupling resource sampling and failure
handling into the authority-bearing supervisor. Both old and new commands get
the same tool and schedule. The observer's source/binary digest, sampling cadence,
clock, kernel/HZ/page size and command are retained with raw observations. It may
write diagnostic failure when the original command fails; this never upgrades a
failed command. Interrupted/missing final observations are incomplete evidence.

## Compatibility, verification and acceptance

All output is a new explicitly versioned diagnostic family, without Source or
resume authority. Old/new policy data and format readers retain their semantics.
No Simulator changes, hidden-state features, rejection probing or GPU admission
queries are introduced. Per-owner counters are cheap and bounded; do not hold a
new shared lock across HTTP/GPU/queue waits. The observer owns only its output
and original command handle; it must not follow output symlinks into source
or scan unrelated run directories.

Focused verification must cover histogram edges/merge/rank bounds, actual batch
counts, zero/failed waits, queue backpressure, success/failure separation,
observation-disabled fast paths, stale responses not becoming successful RTT,
PID reuse/partial PSS/phase attribution and exact original-child wall boundary.
Use deterministic injected clocks for instrumentation tests, no timing-sensitive
sleep assertions. Compare instrumented and disabled fixed numerical/RNG outputs.
Run a bounded actual two-game W1/W2 diagnostic to confirm useful samples and one
Python; do not call that the full approved performance population.

The unchanged predeclaration is in task-7-brief.md and the durable protocol
archive. No timing data has been generated under this supplement. Architecture
acceptance of this design does not accept code, measured speed, memory capacity,
quality, any default switch or completion of the approved matrix.

## Closure of the initial architecture findings

Initial review blocked the first proposal on three P2 findings: compatible
monotonic phase events, bounded per-collection distribution partitioning and
actual observation-bias calibration. Root incorporated these corrections.
The exact accessor/record/stage decisions in the accepted
[closed telemetry contract](rust-player-runtime-telemetry-contract.md) passed
narrow architecture rereview before isolated implementation was delegated.
The following rationale retains the initial findings and accepted choices.

Persistent owners keep only the current collection's histogram state. Begin on
a positively quiescent clean boundary (including the first collection); after the
actual collection drains, snapshot to its explicit generation/batch identity,
flush and reset before the next collection. Do not key an unbounded map by every
past collection. On failure, flush the active window as incomplete; lost/failed
flush is missing data and never obtains cleanup grace or training authority.
Legacy SharedServingController.finish_collection already supplies the actual
plan generation/batch and service counter window after quiescence. Its broker's
existing METRICS_RESET requires no pending requests/active relays. Use that
established ownership boundary; do not add collection IDs to policy routing.

For legacy batch frequency, an independently accepted minimal choice is exact
response-size inversion. Keep FULL window response counts R_b grouped by actual
transport_batch_size emitted by the existing model; do not use the rolling
last4096 samples as full-run data. If each actual b-row forward contributes all
b responses exactly once in one collection, F_b=R_b/b. Establish complete request
coverage, no duplicate/foreign responses, quiescent start/end and zero failure
counts (including cancellation/expiry/rejection/transport/service failures).
Divisibility is only a consistency check, never proof of coverage. Retain R_b,
exact derived F_b and those coverage facts with method=response-size-inversion-v1.
Missing facts make the histogram incomplete. This avoids new GPU forward IDs,
unbounded dedup ledgers or a second batching owner. New mode records once per
actual native batch. No scientific response/ACK schema is relabeled.

For matched request-ready RTT, the accepted narrow choice is a diagnostics-only
numeric-ready monotonic timestamp in the legacy shared worker's closed timing
metadata; the parent ends the interval ONLY after its existing identity-validated
reply acceptance, including CPU sampling and return transport, before Act. The
optional timing key must be explicitly supported by every relevant closed timing
validator and remain outside policy inputs. Native mode starts after final
serving buffers are ready, before queue admission, and ends when the game receives
the accepted sample; do not misuse the later formation received timestamp. Both
sides use CLOCK_MONOTONIC nanoseconds on the same recorded Linux boot. Missing or
invalid timing marks metrics incomplete and does not affect action acceptance.
The exact key and enablement forwarding are defined by the accepted closed
telemetry contract, rather than inferred from existing timing fields.

Environment inspection shows old participant/learner/broker allowlists and clean
bootstraps remove host diagnostic variables; supervisor host/cold inherit them.
Keep path/observation ID host-side and exclude them from the external Simulator.
A worker-only diagnostic enable flag or existing timing metadata is sufficient
where timing must originate inside a legacy worker; no path/ID/private management
context belongs in its policy input. Broker diagnostics use their existing
control pipe, not policy features. No new learner process or controller appears.

Before adoption, run a bounded PREDECLARED enabled/disabled calibration of the
actual application for old and new topologies at relevant concurrency. Same
cadence alone does not establish equal observation cost: a larger old tree can
make smaps sweeps and instrumentation cost more. Retain sweep spans/duty, observer
CPU, application wall/collection rates and candidate/cutoff/count effects. This
calibration is separate from the unchanged formal matrix, never a replacement.
Root declares workload/order before any calibration result and does not invent
an overhead cap or subtract a modeled correction. If observation bias material
to the3% adoption criterion remains unresolved, withhold adoption.

The histogram's highest positive bin has exclusive upper bound2**64, represented
without u64 wrap; overflow observations remain explicit and cannot be assigned
a fabricated finite quantile. Rank/overflow/merge semantics, old/new stage gaps
and exact diagnostic accessors/fields are fixed by the accepted closed telemetry
contract. The [Root predeclaration](rust-player-runtime-observation-calibration.md)
fixes the bounded calibration workload/order before data. Design acceptance does
not establish implementation correctness, measured performance or adoption.
