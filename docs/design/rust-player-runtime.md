# Single-process Rust player runtime

Status: accepted design after independent architecture READY; implementation in progress. Baseline: `9680f06`.
Approved requirements: [owner plan](rust-player-runtime-approved-plan.md).
Research: [ownership/port map](rust-player-runtime-research.md),
[Tokio/PyO3 notes](rust-player-runtime-tokio-pyo3-notes.md).

## Requirements and scope

Run all concurrent MLP training games in one Rust runtime inside one Python
learning host. Rust owns external HTTP v2, routing, each participant's observation,
visible event bank, v5 candidates/search RNG, Random/Heuristic/Refresh state,
request acceptance and Act/ACK. Python owns unchanged FP32 MLP inference,
participant CPU Torch generators, PPO mathematics, optimizer, artifact save/restore,
and one dedicated SQLite writer thread. A Rust supervisor owns the original
host child handle and records actual exit. Compare Tokio-per-game (A) against
fixed-worker game assignment (B), then run Model4 HPO and ordinary Model3 through
fresh/save/pause/resume/audit. Preserve all existing readers and resume identities.
HPO evaluation, GRU, Transformer, quality qualification and final test are outside
this runtime port. The Simulator repository is read-only and remains external.

New modes stay explicit until the exact owner adoption protocol passes. A cheaper
smoke is never substituted for the owner's 32 games x 2 updates, width192,
seed17, epochs4/minibatch128 comparison. No performance claim follows from a
component benchmark or an unexecuted experiment.

## Pins and dependencies

Use Simulator `072dabbca1458c9526897781d2fd64edeeb4f9ca`, specification07 SHA256
`8e5223b45a6e8c4c80e272736ed97b9e271136c939eca74427a035ddfdbfcac2`, v2
Observation/typed Action/ActionOptions/events and public catalog2/public-ja-v3.
Keep existing adopted public-rule signatures including defense priority. Existing corrected-layout
frame/EventBank hashes intentionally retain their historical91d5926 layout
contract even for shared072dabb action/runtime settings. Preserve those actual
identities separately and test their binding; never relabel a frame hash to the
new runtime revision. Capture
actual external executable SHA256 and config for each run; never use an old
binary with a new normative label. Historical91d5926 data stays historical.

Add pinned Tokio and reqwest to native/Cargo.toml and lockfile after review.
Initial proposal: Tokio `=1.53.1` (locally available), reqwest `=0.13.5`, async
client with only json and runtime features needed for external local HTTP.
No Tokio/Python asyncio bridge, TorchRL server, Rayon pool or additional Python
runtime dependency is required by this mode. Existing optional extras remain
available for their old modes. Add a Rust supervisor binary target in the native
crate, without importing the Simulator engine. Verify upstream API/version and
lock resolution before relying on a dependency; a failed version lookup is a
build prerequisite to resolve rather than permission to silently change pins.

## Ownership and dependency direction

| Owner | Responsibility and dependencies |
| --- | --- |
| native/src/player_runtime/ | Pure own-input policy/search/RNG/tokens; game/HTTP actors; batching queues and lifetimes. Depends on existing native DTO/rule/token helpers and Rust crates, never calls upper Python modules from worker threads. |
| gf_common/rust_player.py | Lazy-free closed configuration, execution/producer/receipt schemas and routing value types. No HTTP/Torch/native imports. |
| gf_models/rust_player.py | Final-dtype buffer-to-Tensor views, immutable existing shared_registry/score_numeric_row integration and own-generator sampling transaction. No HTTP or management access. |
| gf_simulator_client/rust_player.py | Native runtime facade, supervisor communication and original process/task cleanup. May expose an orchestration facade to training, as ADR0053's narrow edge does. |
| gf_training/rust_player.py | Same-process learner and bounded dedicated writer thread; existing capture/journal/group-pack/lineage owners remain authoritative for accepted data and saved generations. |
| gf_cli | Config composition, external server startup, normal train/HPO training selection and supervisor exec/reentry. |
| native/src/bin/gf-player-supervisor.rs | Original Python host spawn/wait, bounded termination and durable actual exit evidence; no learning/data authority. |

The companion is packaged in the native wheel as `gf_native_runtime/gf-player-supervisor`
plus a generated manifest containing executable SHA256, native source/Cargo-lock
identity and build profile. `tools/build_rust_player_wheel.py` is the explicit
runtime-wheel build entry: cargo builds the pinned release binary, the helper
writes the manifest and Maturin includes both in the wheel. The verified include
pattern is `{ path = "gf_native_runtime/**/*", format = "wheel" }`, relative to
native/pyproject.toml, with generated staging only under native/gf_native_runtime/.
The helper cleans staging in finally; executable mode0755 and exact distribution
file paths are preserved. The ordinary Maturin build explicitly selects the
extension cdylib when the Cargo binary target exists, so it remains companion-free
unless the runtime-wheel helper stages the companion. Verify actual installed
wheel inventory rather than treating an editable install as release evidence. This does not require
another Python package dependency or a Simulator build. Plain offline-native
builds remain usable for old modes but may lack the companion; selecting the new
mode then fails with the exact runtime-wheel build instruction before GPU/game
activity. Never search arbitrary PATH or compile at runtime. Discovery uses the
installed native distribution's recorded file inventory, checks a regular
executable and its manifest hash, and runs its identity handshake against the
loaded extension's source/lock/profile/ABI. Verify source-tree and installed-wheel
launch, missing/tampered/mismatched companion refusal and optional-free old CLI.
Do not commit generated executable bytes; retain source and lock identities.

The new native online responsibility is explicit and separate from existing
offline Source admission/feature APIs. No offline capability is treated as online
execution authority. No policy trait can inspect a game router, another
participant, Replay, attestation, management seed or diagnostic error detail.

## Public boundary and configuration

The explicit new runtime schema is `rust-player-runtime-v1`, with scheduler
`tokio` or `fixed`, `game_concurrency`, `player_worker_threads`, `max_batch_size`
and `formation_wait_ms`. Accept only positive integral counts (never bool),
finite nonnegative wait and the existing contract's concrete bounds. Do not
invent a W<=C restriction or reuse max_participant_workers' process semantics.
I/O/supervisor/writer/Torch thread counts are separate observations. New effective
config and parent identity include the complete runtime and producer identity
`rust-player-producer-v1`; old shared and resident values keep their old meaning.
An old parent always resumes through its recorded runtime. Each new mode has
explicit strict source/receipt readers; do not relabel old receipts as thread joins.

Preserve Journal6 and participant DTO3 when their content/chronology is unchanged:
these fields attest own-view decision data, not Python actor process existence.
Its runtime_evidence remains the real external Simulator process receipt. Numeric6
may likewise remain unchanged when the existing Python canonical feature/numeric
encoder still produces its content. In that case `encoder=canonical-python` is
truthful for persisted numerics; it does not label the Rust online player or
native serving-row encoder as Python. Keep those producer/encoder roles distinct.
Native serving rows have their own Rust identity; changing persisted numeric
encoding to Rust would require a new exact encoder/reader branch, not relabeling.
New outer run/collection/proof/update/registry/import/completion discriminants must
bind Rust online/capture provenance and host/thread retirement explicitly. Do not
reuse old owned actor protocol9/10 as Rust custody, or construct a legacy process
receipt. Preserve existing inner data validators and meanings instead of globally
bumping unrelated data-format constants. Root agrees the exact outer-version
inventory before implementing those source/lineage branches.


Native facade operations (precise Python spelling may follow existing bindings):

- start(runtime_config, run_id, trial_id, accepted games and own-policy descriptors)
  starts the owned runtime and the common batch dispatcher.
- next_batch(deadline) releases the GIL while waiting and returns a batch lease,
  ordered request identities and final-dtype contiguous owned NumPy buffers with
  per-row offsets/shapes. Only Rust forms batches.
- submit_results(batch_id, ordered scores/value and provisional own samples)
  validates identity/revision/checkpoint/original deadline and active epoch before
  atomically adopting an own sample; invalid responses consume no committed RNG
  and produce no Act. Duplicate/foreign/missing responses fail or retire their
  original requests, never match by row position alone.
- next_storage(deadline) / storage_ack(id, accepted-or-failed) transfer immutable
  pending/ACK/terminal records plus one unique byte reservation to the writer.
- cancel(reason), quiesce(), join() stop admissions, wake all waiters and report
  original tasks, queues, buffers, checkpoint references and actual completion.

Use typed Rust core APIs for these operations, with PyO3 wrappers limited to
control and owned arrays. JSON remains acceptable for compact controls and
unaltered journal input; do not serialize per-request numeric tensors as JSON.

## Participant state, compatibility and data flow

Routing identity is (run, trial, actual Simulator game ID, player ID, epoch).
A game actor serializes mutation and has at most one outstanding inference.
Each participant owns history cursor/event bank, frame interval/decision counters,
search CPython RNG, policy-local state and last accepted request. Reusing a
worker/game ID cannot reuse a previous participant's state. New checkpoint
publication does not alter an already accepted request or a frozen opponent.

Before porting, capture v5 comparison cases from the baseline generator itself:
own input, full candidate report/order/hash, fixed exploration clock transcript,
search RNG before/after, baseline selection and baseline RNG after. Keep old
generator goldens at their original version. Cover defense/Rainbow, Refresh
history, duplicate instances, empty events, zero/positive MP dynamic weapon and
large candidate spaces. Implement CPython integer seed initialization,
MT19937 getrandbits, rejection-sampled randrange and shuffle/state serialization
exactly for used APIs. Derive search seed from the existing digest recipe.
Use existing public rules only; no rejection-driven legality or engine calls.

For MLP, Rust retains opaque committed CPU Torch generator state bytes per
routing participant; it never interprets those bytes. Python owns a temporary
participant CPU torch.Generator restored from those bytes, samples with the
existing torch.multinomial and returns the sample and provisional next-state
bytes in the same result. Native acceptance atomically adopts both, only after
matching routing/epoch/revision/checkpoint/original deadline. Rejected/expired/
duplicate results leave committed bytes unchanged. There is no separate Python
committed state or acknowledgement window. Validate Torch state at its Python
restore boundary, bind it to the recorded Torch/runtime version, and return it
only for the exact participant. Dedicated inference/sampling consumes native
batches while waits release the GIL. A cancellation after valid acceptance may
cancel Act but never creates unacknowledged experience. Tests cross the deadline
between sampling and native acceptance and inject duplicate/foreign responses.

Preserve exact candidate ordering and tie rules, public-rule signature, policy
labels and recorded RNG under a fixed clock. Wall-clock search cutoffs remain
real in games and are recorded separately (reason/count/candidate distribution).
Do not require independent live trajectories to be identical under real clocks;
compare saved-state restore with same-input numerical controls as well.

## Shared buffers, batching and inference

A single dispatcher accepts bounded inference requests. Start formation at the
first valid received request and use that absolute deadline; later arrivals do
not extend it. Emit at max_batch_size or expiry, excluding retired requests.
Close/cancel drains promises with explicit errors, including queued senders.

Rust publishes serving dtypes directly: float32 numerics, int64 discrete/index
fields, bool masks, including exact empty interval trailing dimensions. Ragged
rows use offsets and lengths, with no cross-row reductions or normalization.
The existing row-local score_numeric_row and immutable registry remain the
mathematical baseline (atol=rtol=1e-5). Reuse final-dtype owned arrays through
torch.from_numpy. The PyArray base owns the allocation; Rust does not mutate or
recycle an exported allocation. Every Tensor keeps its owner alive. Initial
H2D is ordinary synchronous transfer, with separately measured CPU sharing,
batch construction, transfer, forward and return; adding asynchronous/pinned
transfer requires an explicit completion-owner test. No global Torch RNG
initialization may overlap a learner update or another registry publication.

## A/B scheduling

A: one future/task per game on one Tokio multi-thread runtime configured with W
worker threads. B: fixed round-robin game assignment to W named workers, each
with a current-thread runtime/local task set and no work stealing. Its own
completion events resume games, so pending HTTP/inference/storage permits other
games on that worker to progress. Both reuse the same game transition function,
HTTP pool/configuration, generator, batch dispatcher, clocks and writer facade.
W counts threads that run participant CPU work; IO on those threads is observed
as such. Runtime control/batcher and Python service threads are recorded separately.
No per-game blocking recv, sleep, Python callback or spawn_blocking CPU pool.
CPU search cooperatively checks existing finite budgets; add another CPU pool
only after measured event-delivery obstruction justifies a new design decision.

HTTP uses external v2 only: strict sticky TestServer attestation initially,
before Create, after Create/pre-Join, reconnect and resend. Create is never
retried. Disable reqwest automatic retries and redirects explicitly; only the
owned client state machine may retry. Join/Act allow existing at-most-two same-command/same-payload transport
retries following fresh successful attestation. Rejection is a diagnostic stop,
never a candidate oracle. Preserve events pagination/cursor advance, viewer
scope, same-revision hints, original deadlines, Replay reconciliation and Delete.
GameSeed/attestation/Replay stay in management state, outside ParticipantInput.

## Writer, learner and generation boundaries

One named Python storage thread constructs and owns SQLite connections and
reuses existing group pack validation, seal/checkpoint and storage formats.
Bound its queue in bytes under the recorded storage allocation, plus existing
item capacity where needed; reserve actual concurrently retained bytes once,
including data held by blocked producers. Pending records are validated/encoded
before Act; only a matching accepted ACK commits experience. Duplicate commands
are idempotent and mismatched/late ACK never double-commits. Store zero-Act
terminal bank identities as the existing capture contract requires.

Acquire the recorded collector byte reservation before allocating/retaining each
recorder increment. This is the existing record/document accounting, not a new
physical-RAM guarantee. Pending/ACK data is retained in the game recorder; terminal
handoff transfers that immutable recorder and its SAME reservation without a
second bulk copy/charge. A completed game keeps its slot until the writer accepts
the handoff. The bounded terminal channel carries ownership handles; if full the
game retains its handle/slot and asynchronously waits, while other game futures
and the separate cancellation/control path progress. At most C such unaccepted
handles exist because no slot has retired. Thus ACK/terminal handling needs no
new unbounded bulk queue or speculative terminal-memory reserve.

Once writer custody accepts the immutable original plus reservation, its queue
accounts that reservation until last use/write acknowledgement; pending data and
blocked producers remain in collector accounting exactly once. Refund only on
actual last owner release, or retain on uncertain failure. If an increment or
single item cannot fit its configured budget, fail the affected collection/parent
explicitly before any subsequent Act; do not wait indefinitely for impossible
capacity or silently drop data. Queue saturation blocks new-game admission (and
completed-slot retirement), not control/ACK processing. Cancellation closes/wakes
all sends, and writer failure resolves every outstanding handoff as failure.
No extra reserve or max-Pack-per-game preallocation is introduced.

A game slot retires after complete immutable terminal/original data and its
reservation move to writer custody, not merely after enqueue of an unowned
reference. Collection finalization
waits for all natural/aborted outcomes, no pending commands, writer acknowledgements,
SQLite checkpoint and seal. Only then may PPO update run. Saved checkpoint and
optimizer/RNG/schedule pair commit through existing lineage before publishing G1.
Pause/resume occurs only at a committed update boundary with the original
remaining deadline. Writer/GPU failures invalidate the whole parent eligibility;
never synthesize a win/loss, bootstrap or completed update.

## Supervisor, termination and GPU admission

The normal CLI replaces itself with the Rust supervisor before starting a
Python campaign/ordinary host. Exactly one Python host is live during training.
Python retains campaign proposal/materialization decisions. Before training GPU
allocation it requests admission for that exact operation over inherited private
control descriptors. Rust acquires the existing per-user flock and quarantine
namespace, durably admits the exact operation and authorizes that same host.
Use an explicit supervisor-bound admission capability, not legacy pid/thread-bound
GpuJob reconstructed from JSON. Bind nonce, original child, run/config, deadline,
GPU budget and full producer identity; markers/PID absence/EOF are not authority.

The host collects/updates/saves, closes native tasks/writer/CUDA work and publishes
`rust-player-prepared-v1` with the final committed state and cooperative close
proof. This is prepared data, never physical termination, evaluation authority or
eligible completion. It exits. The supervisor waits its ORIGINAL child handle,
then durably records `rust-player-exit-v1` with actual status/signal, exact
admission and authenticated close acknowledgement. Only confirmed reap plus
durable evidence may resolve physical admission. Forced exit may establish reap
but leaves parent data ineligible. On the Linux target, supervisor enables
PR_SET_CHILD_SUBREAPER before spawning each host in its own isolated process
group/session and retains that original group identity. Existing ManagedServer
inherits the group and pinned Simulator072dabb starts no OS child processes.
Closing its stdin requests shutdown; EOF never proves exit. Cooperative host
cleanup still records actual Simulator wait/shutdown/log outcomes before prepared.
After original host wait, the supervisor also positively reaps adopted descendants
until no owned child remains, within existing cleanup/server deadlines. Stop/join
its own telemetry child helpers before this drain so one wait owner cannot consume
another helper's status. If needed, TERM/KILL only the owned host group and record
every actual adopted PID/wait status and whether cleanup signaled it. Exit evidence
includes this positive descendant-retirement result, separate from already-reaped
Simulator outcomes recorded by the host. Supervisor cleanup never invents natural
game termination, a clean server status, or qualified completion. Unknown/out-of-
group descendants, unconfirmed wait or durable receipt failure prevent continuation
and retain quarantine. This contract relies on the pinned process topology;
setsid/child-spawning Simulator changes require a new containment review.
Uncertain reap, supervisor loss or missing
durable evidence retains the same quarantine and existing explicit recovery
rules; recovery never creates completion evidence.

After reap, Rust starts a fresh Python continuation. It cold-validates the exact
prepared result, original Source, final policy and exact exit receipt, then
publishes `rust-player-training-completion-v1`. Keep legacy GpuJob,
TerminationReceipt, HpoRunResult and OwnedRunCompletion semantics intact; new
mode dispatch has explicit typed authority and versions. Old readers never
interpret a thread join or empty child history as legacy termination.

Campaign state gains an explicit versioned `awaiting-supervised-finalization`
transition. Only this exact protocol's recorded prepared result plus qualifying
exit evidence can finish it; historical incomplete intents remain ineligible.
Retain exclusive campaign custody across exit/spawn gaps with the original
lock descriptor owned by the supervisor and narrowly handed to the one
continuation. Preserve the original parent-run lock across the same handoff for
ordinary and HPO finalization as well; one continuation owns its descriptor and
two finalizers cannot race. Cold finalization validates CPU Source/artifacts
without GPU allocation or implicit admission. Test duplicate finalizers and a
crash during transfer explicitly. Persist the original invocation/session timing and absolute run
deadlines so reentry cannot extend them. Host live-completion objects never
cross process death; use the existing cold-import/Source-validation path extended
for the new producer. A prepared file or successful exit is never a Source proof.

Evaluation then uses its existing runtime and independently admitted legacy GPU
job. Its subprocesses are outside this training-only single-Python scope. After
evaluation the continuation may become the next learning host upon fresh exact
supervisor admission. This avoids a second live Python controller and a
campaign-wide training lease that would conflict with evaluation admission.
Ordinary train/pause/resume uses the same prepared→reap→cold-finalization boundary
without campaign proposal scheduling. Scope includes campaign entrance, not only
a standalone Model4 helper.

Normal cooperative close stops admission, retires pending requests, joins original
Rust workers/batcher, finishes writer/CUDA operations and releases buffers and
checkpoint references before publishing prepared state. Rust reaping the actual
host then ends CUDA context/allocator lifetime. Failed prepared publication,
exit, receipt publication or continuation finalization never publishes an
objective or automatic orphan recovery. A strict continuation finalization is
retryable only for the same immutable prepared/exit pair; no new training occurs
under a closed admission. Measure reentry/cold validation/startup inside whole-run
comparisons and test two successive operations with intervening existing evaluation.

## Memory and measurement

Count shared runtime/weights once, 2C participant histories, W search workspaces,
<=C outstanding requests, exported/in-flight tensors until actual last use,
and writer-retained bytes under their actual concurrent lifetimes. Keep admitted
estimates distinct from measured process PSS/RSS, CUDA allocated/reserved and
whole-device samples. Do not reuse per-process actor PSS coefficients for this
runtime. Report any planning uncertainty; no arbitrary safety multiplier.

The approved protocol is binding: C8/B8 each A/B W1/2/4/8 with bracketing legacy
controls; 1ms formation; width192 seed17 epochs4 minibatch128,32 games x2 updates.
Select fastest W, within3% prefer lower PSS then W. C16/B16 repeat candidates and
legacy in both orders, all with5GiB storage,24GiB RAM/12GiB CUDA. Keep failures
outside successful timing populations. Before defaults, alternate two legacy/new
runs for actual HPO C4/B4 and ordinary C2/B2; each paired collection decrease and
whole-run increase must be<=3%, and each new peak PSS lower. Split first/second
collection; record all-player decisions/s, games/s, batch distributions, p95/p99
event/inference latency, PyO3 calls, CPU/threads, GPU timing, PSS and writer waits.

## Verification and staged implementation

1. Root design/ADR and independent architecture gate; capture v5 reference cases.
2. Rust policies/search/RNG and token/own-view state parity, no runtime promotion
   until exact fixed-clock comparisons pass.
3. Owned buffers/batch dispatcher/Python immutable inference and transactional
   sampling; adversarial cross-seat/game/run independence, expiry and owner tests.
4. Common game/HTTP state machine with both real schedulers; W1/W2 two-game/four-
   participant tests, full queue/cancellation/delayed ACK/same-command retry.
5. Same-process writer/learner, supervisor/receipt and parent version support;
   writer failure/forced host exit; CUDA Model3/4 continuous2 and pause1/resume2,
   G0 pending/G1 published/frozen G0, same-input restore and strict original audits.
6. Normal CLI/HPO selection, package/static/source native checks and regression
   checks appropriate to changed boundaries; then approved physical comparison.
7. Independent correctness and architecture conformance over stable code and
   evidence; root records selected scheduler/W or refusal to adopt, actual
   measurement limits, coherent commits, and leaves merge/publish to owner.

Open evidence: full v5 port equivalence, real buffer sizes/cost, actual runtime
throughput/memory, final supervisor integration and full audit success. These
are required work, not assumed accepted facts. If an interface must change,
root updates this design and returns the affected boundary to architecture review.
