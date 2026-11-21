# MLP shared batch inference after Simulator issue72

Status: implemented; independent correctness/conformance READY and bounded
physical acceptance recorded in [verification](../verification/ppo-shared-inference.md).
Base: Bots c78f8a0 on feat/ppo-shared-inference. This continues the approved
resource work in [the earlier design](ppo-shared-inference.md).

## Requirements and adoption

Adopt Simulator072dabbca1458c9526897781d2fd64edeeb4f9ca, spec07 SHA-256
8e5223b45a6e8c4c80e272736ed97b9e271136c939eca74427a035ddfdbfcac2.
Section21.1 now authorizes a trusted shared executor subject to independent
participant computation, checkpoint fixation and exact response ownership.
Add an explicit contract profile/inventory; preserve existing91d5926 and07e
profiles and historical readers. APIv2, catalog public-ja-v3/schema2 and game
rules are unchanged. New runs record full normative inventory, actual source
revision/dirty state and executable hash. A checkout revision alone does not
prove the binary; use a disposable exported source tree for a corresponding
build without editing the Simulator repository.

Connect one common MLP scorer to producer-owned resident HPO Model4 and ordinary
Model3 PPO. Shared mode is explicit, CUDA inference is explicit, and optional
TorchRL0.15.0/TensorDict0.15.2 are lazily imported. Other modes retain their
recorded execution semantics. Shared mode is not enabled for sequence models,
formal/final-test workflows or evaluation paths lacking corresponding ownership
and contract verification. No learning-quality authority is added.

## Responsibilities and dependencies

The CLI composes serving and participant capabilities from the actual training
invocation. Training retains learner, committed generations, Source and registry
authority. The simulator-client layer owns clean participant processes, bounded
numeric IPC endpoints and real process retirement. The model layer owns pure
MLP numeric encoding/scoring and the TorchRL policy adapter; it imports neither
HTTP nor training controllers. Common holds closed contract/config metadata.
The coordinator remains free of Torch imports and private inference computation.

Each CPU participant retains only its own observation/history, generates its
ordered candidates, packages the same existing numeric inputs and samples with
its own RNG from returned per-candidate log probabilities/value. It constructs
the existing PolicyDecision/Act. No CUDA context, weights, optimizer or resume
state is loaded in CPU actors. Existing accepted-Act capture remains the sole
source of rollout transitions; no request/response creates experience itself.

A clean CPU broker is permitted to bridge the existing exec participant
bootstraps to official multiprocessing clients. It creates the official
InferenceServer/MPTransport clients before process startup, handles bounded
dedicated participant requests in relay threads and owns exactly one GPU
inference process using the process service backend. This avoids pickling
unsupported multiprocessing Queue/Event clients into clean exec actors. The
broker is trusted serving infrastructure; only the participant's own result or
redacted error returns over its endpoint. Endpoints are handed to actual owners
as private capabilities, not discovered from policy-supplied identity fields.
No new generic RPC service, policy-facing multi-view API or custom replacement
for TorchRL aggregation/dispatch is introduced.

The bridge uses participant-specific inherited socketpairs with closed numeric
messages and bounded framing. The broker starts before actors; it creates its
official clients before spawning the service. No policy receives a multi-client
registry or arbitrary endpoint selector. CPU actors may import CPU numerical
libraries when required by existing tokenization; they never initialize CUDA.

Extend the ledger with one distinct service reservation, outside participant
worker slots. It retains service residency and actual phase scratch. Shared-only
update admission requires a service quiescence token and drained/parked CPU
actors; service residency remains charged during optimization/save. Existing
worker/learner ownership and exact-process checks retain their meanings.

The broker's original private control channel registers invocation/job/service
IDs (job ID for HPO, invocation-local ID for ordinary), service epoch, broker PID
and original GPU child PID. Normal retirement
on that same channel binds the registration digest, drained requests/publication
counters, joined relays and original multiprocessing.Process join/exitcode.
Parent verifies the bound receipt while retaining original broker custody,
then waits its exact Popen object before issuing composite retirement proof.
Pending creation and before-child-start failure require actual no-child evidence.
Broker loss without child proof retains the lease/pins and leaves HPO GPU admission
unconfirmed/quarantined; parent-death signaling alone is not reap evidence.
There is no new subreaper or administrative recovery authority in this task.
Retain the original mp.Process before official shutdown: TorchRL may clear its
process reference after bounded terminate/join. Independently join/check that
original object and retire relay/queue I/O; library shutdown return is not proof.

Ordinary Model3 retains its existing invocation-local ReservationLedger rather
than acquiring HPO's durable global GPU gate. Its service receipt binds the
ordinary run/config/invocation IDs and is persisted in that parent's service
namespace. The ordinary coordinator closes serving after collection drains,
before publishing healthy completion/pause; real service+broker reap, empty
pending/IPC and released service pins join the existing learner/actor cleanup
checks. Failure marks the whole parent failed and retains unconfirmed local
custody diagnostics. It does not claim a global GPU-admission quarantine or
prevent a separate invocation through HPO's namespace. Paused resume revalidates
the persisted original service cleanup, frozen runtime/config/source identities
and mode before creating a new invocation/service; no live service is inherited.
This preserves ordinary ownership scope while making success require actual
shared-process cleanup. HPO alone mints the durable schema2 GPU-job termination.

## Request and batch interface

The closed opt-in runtime descriptor uses mode=shared-mlp-v1, max_batch_size
(positive integer, default resolved from game concurrency), formation_wait_ms=1.0,
cache_entries=6 and cache_weight_bytes=134217728. Ordinary configuration adds
run.inference_service only when present, using a dedicated RunConfigV2 subtype
so absent fields leave historical effective config/hashes unchanged. HPO uses
an explicit new campaign/run-spec schema5 with producer-owned-v1 origin and
canonical-python encoder, plus run_variant=hpo-ppo-owned-shared-v1. Do not pass
the descriptor through the legacy resident parser. Require MLP, explicit CUDA
inference and the new adopted contract, PPO training and Python preprocessing;
reject unsupported formal/evaluate/final-test, sequence and native routes.
Mode, resolved geometry and admission recipe join execution/resume identity.
Preserve the recorded CUDA model binding and NumericCosts; the CPU transport
capability does not relabel the model's inference device.

Every accepted request binds execution/trial UUID, game ID, participant, monotonic
request ID, source Observation state_revision, and immutable checkpoint digest.
Control identity is separate from numeric features. The serving owner obtains
the allowed execution/game/participant/checkpoint from actual admission and
episode capabilities; untrusted fields cannot choose another participant's
endpoint. A participant compares all response fields with its pending request
before exposing scores. A mismatched/expired response is discarded and produces
no action, RNG advance, journal transition or training sample.

One request can be outstanding per game; enforce at the collection/game owner,
not by summing worker slots. Transport max_batch_size is independent of game
concurrency and defaults to it. Set min_batch_size=max_batch_size and timeout=.001
in the official server so ready singleton requests may wait at most1ms for more.
An official transport batch may include several digests; the trusted adapter
partitions it into checkpoint-homogeneous scoring groups and scatters results
back to the original request order. Ragged
candidate outputs retain their original order and exact valid extent.

Preserve shared current MLP network mathematics and IEEE FP32 settings. Normalize
candidates within a row only. Private sequence/candidate lengths cannot choose
another row's arithmetic shape or reduction order. For operations where padding
or batching affects a reference row as other private lengths/order change,
execute that operation row by row with its isolated shape. Batch only operations
that pass independence checks. Telemetry distinguishes transport batch size from
tensor-operation batching; do not claim vectorization that was not performed.

## Checkpoint publication and retirement

### HPO campaign evaluation dispatch

Schema5 must compose the existing HPO campaign through final-policy inspection
and evaluation without handing the shared training descriptor to a legacy
resident parser. Add one explicit evaluation-runtime selector: schema5 selects
the existing fresh participant evaluation path (runtime=None); existing schemas
retain their current fresh/resident selection. Record that actual evaluation
mode in campaign conditions and bind it consistently at plan construction,
publication, saved verification and comparison. Training keeps its original
shared runtime, registry and service receipt; never relabel the training parent.

Complete-parent, final-policy and cold readers explicitly admit schema5/its
shared registry and schema2 training termination after validating those new
proofs. They retain existing exact decision-budget, healthy completion,
checkpoint, original-Source and acquired evaluation-reference requirements.
No ineligible two-update performance parent may supply an objective.
Evaluation/comparison uses existing per-participant model inference, its actual
resource admission, artifact retention and cleanup; no shared service or service
lease is created for evaluation. Preflight its real participant CUDA/worker
geometry under the existing budgets rather than applying shared training's
single-service estimate. This adds no new evaluation concurrency tuning or
automatic reduction. Unsupported formal/final-test and shared evaluation serving
remain excluded. Verify component training→final-reader→fresh evaluation and
saved/cold verification with truthful schemas; no full HPO campaign is authorized
solely to test this adapter.

The inference registry stores inference-only immutable checkpoints keyed by
digest and preserves versions referenced by frozen opponents or valid accepted
requests. Model3 and Model4 use their existing loaders into the same network;
the service never receives optimizer/resume bundles. At acceptance the request
pins its checkpoint; later publication cannot rewrite its weights. A serving
module dispatches by bound checkpoint, grouping compatible requests. The
TorchRL mutable policy_version counter is diagnostic and cannot replace the
checkpoint digest.

HPO obtains only committed registry snapshot actor capabilities and transfers
their inference banks/metadata using existing RamBankChannel ownership; no
resume bank. Model3 uses its existing retained inference manifest/weight sealing
and validation path, then shares the tensor-to-network importer. Do not invent
producer-owned provenance for ordinary artifacts. Bound the plan's complete
distinct neural inventory, including configurable frozen pool, current version
and publication overlap. Refuse insufficient configured cache entries/bytes
rather than evict a required or in-flight digest.

Add typed service-cache acquisitions/pins for both registries, carrying committed
generation/inference/policy/weights identities (plus owned snapshot identity),
service ID/key, broker/GPU identity and cache identity. They participate in the
existing retention-protection algorithm but never masquerade as a participant
viewer/worker. New shared registry/cache row versions explicitly select this
owner schema in Python cold/audit readers; old rows retain exact meaning. Release
only after exact immutable-registry eviction ACK or composite service reap.

### Complete service correspondence in cold records

Cold validation must derive required services independently of the editable
reference map. Add a versioned shared-only service_usage stamp to the existing
committed update/source-proof controls, protected by the generation's existing
update hash. Stamp the original live execution/service/job/registration/config/
runtime identity, collection generation, required neural checkpoint/token
mapping, and start/end cumulative request/publication counters from actual
quiescence. Do not infer this evidence from report.json or registry rows.

Extend existing shared-mode invocation rows with start/end generation and an
explicit actual created-service identity, registration/final-proof digests and
HPO global termination digest, or explicit never-created state. This covers
pause/resume and zero-update invocations without a new generic journal. Stamp
from original owners at lifecycle boundaries; an untrusted flag is not live
authority. No-service is valid only when independently consistent with no
committed service-using collection, including genuine pre-service generation0
or an invocation that exits before collection.

Readers already scan small committed update/generation/invocation controls.
Derive exact required invocation/service intervals and job bindings from those
anchors; derive mandatory checkpoints from update.policy_bindings and each
committed generation's current+snapshot_pool controls. Require exact service
namespace/reference coverage, one resolved acquisition per actual publication,
mandatory checkpoint coverage and original registration/retirement evidence.
Collection counter windows must be contiguous within an invocation, with no
reset/gap and final totals matching actual retirement. HPO requires its stamped
schema2 global proof for each created shared service/job; ordinary mode requires
its stamped local proof. A missing-all/partial reference map or service directory
cannot erase those requirements. Shared backing/counters are not per-request
logs and add no policy input.

Request windows reconcile continuously from invocation start to retirement.
Publication coverage separately accounts for startup preload, inter-collection
updates and the after-last-commit tail: the last collection's publication count
need not equal final retirement. Record accepted broker publication ordinals
tied to checkpoint/acquisition tokens at their actual acknowledgement and bind
their complete sequence to invocation finalization. Reconcile that sequence,
collection boundary counts, required checkpoint sets and retirement totals.
These ordinals are distinct from TorchRL's diagnostic policy_version, which may
also advance for other registry mutations. Do not create a per-request journal
or rewrite an already committed update merely to attach a later publication.

An invocation that drains a healthy collection but stops with empty_collection
before committing an update may retain one optional tail_usage stamp of the
same versioned owner-minted shape. Its generation equals invocation end
generation; it is absent for a true no-collection exit and cannot duplicate a
committed usage. Include it in request-window reconciliation without inventing
an update, transition, reward or qualification for zero learner endpoints.

Use an explicit shared stamp/record version branch when required fields change;
historical ordinary and HPO schemas retain exact meanings. Avoid reopening
historical SQLite/Replay/numeric bodies merely to recover this small new-mode
anchor. Verify missing-all/partial references/services/proofs, wrong execution/
job/checkpoint/counters, resumed multi-invocation parents, no-op invocation and
genuine pre-service generation0, alongside original compatibility readers.

### Preparation and independent cleanup

Validate manifest/registry/digest/row inputs before borrowing snapshot bindings
or building ordinary staging banks where possible. From the first acquisition,
a local pending owner must retain cleanup responsibility until it transfers to
the tracked service acquisition. A preparation exception releases unexposed
temporary bindings/banks. After registration/publication begins, ambiguous I/O
keeps custody through original channel/service retirement. Track ordinary
staging banks as well as owned snapshot bindings; close staging on confirmed
publish completion, or after actual retirement when transfer was ambiguous.

Service and learner cleanup are independent attempts. A service-close exception
must not skip learner close/reap. Preserve the initiating error and attach both
cleanup failures/remaining actual owners. Test metadata/registry/digest/row/write/
publish preparation failures with real binding/budget/bank effects and ordinary
failed/uncertain service close with learner retirement. These refinements change
no learning math or successful-path policy visibility.

After generation save and actual Source/registry commit, apply next committed
weights using ProcessInferenceServer.update_model_weights(), require its
accepted acknowledgement, and expose them to subsequent requests. At that
boundary the normal collector is drained; additionally verify the independent
case where valid old requests coexist with new publication. Preserve a retained
old module rather than updating it in place while it has references. Staging,
old/new overlap and frozen opponent versions join admission. If module-registry
publication through the official update API needs a changed interface, return
to root/architecture review before implementation. The pinned official adapter
calls model.load_policy_weights under its model lock. A bounded hook validates
the complete inference-only publication, constructs a fresh digest-keyed module
and atomically registers/selects it, without mutating referenced versions.

On terminal game, request failure/cancellation or episode processing-generation
retirement, remove pending authority, discard late replies, and clear private
inputs/intermediates before that endpoint can serve a later episode. MLP holds
no temporal hidden state. Publication of a checkpoint is distinct from retiring
an episode generation. No retries with altered actions or automatic fallback.

Broker/service death, ambiguous update acknowledgement, invalid scores, timeout
or failed retirement fail the owning invocation through the existing error and
cleanup path. Do not leak another request's exception contents, inputs or
outputs into participant diagnostics. Cancellation first invalidates requests,
then joins owned relay work and real processes. Retain RAM/CUDA leases and
checkpoint references until actual I/O retirement and confirmed reap. Existing
GPU-job admission/termination and Source failure rules remain authoritative.

## Resource and diagnostic ownership

### Deadline and exceptional-close protocol refinement

Carry the parent's original absolute monotonic request deadline in outer private
control, separate from routing keys and numeric policy features. The CPU channel
checks it before sending, after receiving and before sampling. Expiry discards
the result without RNG advance, action or experience. The supported same-host
processes share the monotonic clock; do not synthesize a renewed timeout.

Shared actor abort first retires broker request authority, then enters the
existing actor join/reap path. Broker metadata-pending and actually executing
relay work have separate counters. Normal quiescence requires both empty;
retiring metadata cannot certify that GPU work and private buffers have ended.

A failed invocation uses an explicit private abort-close operation: invalidate
all requests, stop participant sockets, shut down the official server and
independently join the retained original GPU child, then join relay threads and
retire their queue/IPC ownership before final broker reap. Preserve original
deadline and primary-error precedence. A positively reaped failure may release
operational leases and service pins, but never makes the failed parent/data
healthy. Broker loss or unconfirmed child/relay retirement retains quarantine
and references. Test deadline-before-send, expired reply/no RNG advance,
metadata-retired-but-computing quiescence refusal, cancellation ordering, and
failed close with actual positive versus uncertain retirement.

### CPU host admission refinement

The former learner-like worker envelope is not CPU actor RAM: it includes
six-times parameter/Adam/reference storage, CUDA context/workspace, differentiable
activation copies and full temporal tensors absent from the shared MLP actor.
Retain intrinsic NumericCosts/shape checks, but replace that physical host charge
with a separately versioned, calibrated soft CPU-serving recipe. This is an
estimate used for admission, not a Python heap quota or a hard process bound;
adding a new incremental object-budget framework is not required by this change.

Before enforcing the reduced recipe, audit and screen actual CPU lifetimes:
raw EventBank bytes/list references/prefix hashes through max_episode, the
simultaneous full decoded history used by defense_context, bounded candidate/
search graphs, current frame/interval/candidate tokens, serialization/framing,
returned scores and CPU RNG sampling. Derive event sizes from the closed
supported DTO fields/domains and actual Python representations, using distinct
objects when measuring; aggregate_window is not a retained-history bound.
Use the maximum concurrent phase rather than summing mutually exclusive scratch.
Size active actor scratch for2C unless the actual game-owner serialization spans
generation through reply and proves at mostC simultaneously active actors.

Measure warmed actual actor, broker and GPU-service host baselines in fresh
processes, with private/shared categories and group PSS. Broker queued/decoding/
CPU tensor ownership uses the admitted game-concurrency bound even when transport
max_batch_size is smaller; GPU inputs use the effective transport batch. Count
shared MP/sealed backing once, per-process mapping/metadata separately, and
inference publication/load copies at their actual overlapping lifetimes. Keep
Source and parent sublimits nested under their actual owner rather than summing
every configured cap. Record interpreter/allocator and representation assumptions
with the calibration evidence; unmodeled components stay explicit.

Keep the unaffected learner's existing host compatibility allowance explicitly
identified as such; the new CPU actor recipe does not redefine learning limits
or claim a calibrated whole-learner physical prediction. In the owned topology,
parent writerP is nested under SourceS, so do not addP as an independent full
allocation alongsideS. Source and collector can overlap during collection and
both must be included. External scan/config graphs and private numerical
receiving/export buffers remain separate where simultaneously live. The shared
preflight and actual owner admission use the same recipe and phase inventory.

Acceptance of this refinement requires the bounded max-history and candidate/
position/token/framing CPU screen plus fresh process baseline samples, compared
with the derived terms before lowering the gate. Actual HPO phase PSS/peaks then
check the recipe on the authorized matrix; missed components trigger a scoped
correction, not an automatic budget increase or concurrency reduction. Do not
retain an obsolete learner-model charge merely to disguise missing CPU evidence.

### Provisional physical calibration without changing default admission

The scoped advisor recommends using the first collection of the already
authorized concurrency16 point to resolve remaining physical occupancy and
allocator-retention uncertainty. Do not add another campaign or chase an
unattainable exhaustive Python-heap proof. Before this probe, correctness and
architecture review must clear the actual serving/Source implementation.

Use a disposable, frozen copy of that reviewed source, with a separately reviewed
experimental host-admission hunk and recipe identity. The repository's normal
host guard remains unchanged until calibration evidence is accepted. This copy
may supersede the inherited learner-like actor RAM refusal only for the approved
shared MLP measurement configuration: width192, seed17, epochs4, minibatch128,
collection32, concurrency16, two updates, Source2GiB, collector4GiB and global
RAM24GiB/CUDA12GiB. All input/search/logical collector, CUDA, Source, checkpoint,
failure and cleanup guards remain intact. Record exactly which old refusal is
superseded; no configured RAM increase, retry, fallback or concurrency reduction.

The provisional estimate screens measured warm role baselines, the actual
controller, the learner's retained compatibility allowance and admitted owner
capacities. Unmeasured history/search allocator effects remain explicit; do not
present this startup screen as a complete physical upper bound. Store the full
frozen source/patch/config/runtime identities with the experiment. A public
production bypass or new policy input is unnecessary.

The driver samples aggregate owned-process PSS, system available memory and
Linux memory-pressure counters. Cancel through the original coordinator cleanup
if sampled owned PSS reaches24GiB, memory pressure appears, or the monitor fails;
this is an observational stop condition, not a hard quota or an OOM guarantee.
Record interval/overshoot limitations. Capture actor post-search/idle/PARK
residency together with parent collector contents, Source charges and shared
backing. Freed Python graphs do not imply freed allocator pages.

The monitor samples every350ms (sample duration is additional). Read
/proc/meminfo MemAvailable and both /proc/pressure/memory some/full total
counters. Any increase in either pressure counter since the previous valid
sample triggers one cancellation of the original coordinator task; stale
averages are not a trigger. Failure to read root/system counters or a still-live
owned process is also a cancellation. A process that exited/reaped or became
a zombie during the sample is recorded as retired and contributes no current
RSS/PSS; ordinary PID-disappearance races are not monitor failure. Preserve the
existing process receipts as actual cleanup authority. Cancellation is sent once
via the original event loop; monitoring continues through owned cleanup.

After the first collection drains and retires its requests, inspect observed
capacity before continuing its prescribed update and second collection. Use
remaining explicit Source/numeric/save allowances to screen the next phases;
do not insert an unexplained margin. If observation cannot support continuation,
stop through existing failure/retirement semantics and keep the whole attempt
incomplete. A completed, eligible provisional point remains part of the approved
measurement evidence with its own recorded admission identity.

The first-drain calculation uses a fresh tree PSS observationP and the SAME live
registry Source account: Sfree=S.limit−S.used. The first collection must precede
the first update, so the child has no prior received rollout: Nfree equals its
configured numeric allowance. G1 save has not begun: SaveNew equals its configured
additional save allowance. Do not add nested parentP, the learner compatibility
allowance again, or a retired collector's full cap. Their already-resident
contents are in the observation; current shared Source banks are counted once.
For current Source-bank backing not represented by that PSS observation, add
U=max(0, unique bank inode lengths−summed owned mapping PSS for those inodes),
using one boundary-only smaps/FD observation. This is a conservative backing
allowance, not a measured resident-page count. Do not subtract receiver charges
which already exclude shared backing.

Continue only when P+Sfree+Nfree+SaveNew+U≤24GiB and
Sfree+Nfree+SaveNew+U≤observed MemAvailable, with no pressure/monitor stop,
healthy drained collection and positive request/actor retirement. Missing live
account or mapping evidence stops the probe rather than guessing. Persist all
terms and their measured versus managed scope. This screens headroom for the
next existing owners; private paging/allocator changes and sampling overshoot
remain uncertainty, so it is not a whole-update physical bound.

Root then derives a versioned calibrated soft production recipe using reachable
per-game phase combinations, persistent collector contents, allocator retention
and independently owned copies. Do not add a full collector logical reservation
again alongside the same physical graphs; do not assume actor search pages have
vanished when parent reply preparation begins. Independent evidence/conformance
review precedes normal enforcement. Any remaining unmeasured extreme shapes stay
outside a guaranteed-physical-capacity claim.

The shared CPU actor drain receipt has its own explicit version: CPU END/PARK
does not claim actor CUDA synchronization. Bind pending=0 and episode retirement
to the actual shared service quiescence/epoch proof before update/Source commit;
do not feed these records to legacy resident checked_ack. Preserve Journal,
numeric arrays and accepted-Act formats whose semantics remain unchanged.

TorchRL transfers its complete lazy input batch to policy_device before forward.
Charge all simultaneous input tensors for the effective maximum batch shape,
even when length-sensitive operations execute rowwise, plus live row workspace
and retained outputs until dispatch. Do not charge only the largest single row.

Use a separately identified shared execution/admission recipe. GPU reservations
cover the learner allowance, one serving CUDA context/workspace, each actual
retained checkpoint's weights/position storage, effective batch temporaries and
publication copies concurrently alive. CPU worker count does not multiply CUDA
weights/context. RAM accounts separately cover CPU participant history/search,
numeric IPC, broker staging and shared immutable backing; retain measured PSS
separate from estimates. Do not count the same sealed bank once per mapping.
Keep legacy NumericCosts/search limits and PPO logical minibatches/epochs.

Record first and second collection separately: games/sec, all-participant
decisions/sec, observed active game overlap, transport/tensor batch sizes,
request wait p95, collection/encoding/transfer phases, process-tree PSS and
GPU allocated/reserved/whole-device samples with honest scope. Expose phase
estimates, sampled maxima and differences for issue2. Evaluate TorchRL rollout
transition_chunk_size only if measured worker-to-driver transfer is limiting.

### Serving allocator diagnostics

A small model-layer serving factory wrapper may delegate unchanged registry
forward/load_policy_weights operations and snapshot Torch allocator counters
into a fixed broker-owned shared CPU diagnostic block. It does not replace
official aggregation, transfer or dispatch, and does not modify inputs, outputs,
weights, sampling or arithmetic. Keep this block outside module parameters/
registered buffers so moving the policy to CUDA cannot move it accidentally.

Read one memory_stats snapshot at the existing forward-return or publication
boundary; do not add a CUDA synchronization solely for telemetry. Label the
actual host-side boundary and process-lifetime/no-reset peak scope accurately,
rather than claiming GPU completion where the official CPU-output conversion
has not happened yet. CPU component mode reports CUDA unavailable without
initializing it. Broker QUIESCE/report reads the block; no counter, global batch
fact or shared-memory handle enters participant-visible replies/features.

Use a bounded coherent read or explicit unavailable/stale status; diagnostics
must never wait on a lock held by a dead writer. Optional observation failure
does not alter numerical execution or cleanup authority. Count fixed shared
backing once and private metadata separately. Allocated and reserved are
different views of the same allocator; do not add them as independent memory.
Verify actual factory/transport CUDA counters and own-row numerical invariance
through the wrapper before using its measurements.

## Verification and acceptance

Legacy GPU termination schema1/resources remain unchanged. Shared schema2 adds
one services entry with service key/ID, broker PID/outcome, GPU PID/outcome and
exitcode, registration/retirement SHA-256. Persist bounded original records in
the invocation's service namespace for those hashes. PID is null only with
actual never-started evidence; a child started without confirmed join cannot
be relabeled never-started. Shared Source/receipt readers require this schema,
execution/config/checkpoint bindings and request/publication epoch counters,
alongside separate CPU END/PARK evidence. Old modes reject new shared receipts.
Producer-owned shared HPO and ordinary new-profile preprocessing remain Python;
native readers reject unsupported shared/profile input explicitly. Do not add
unused native formats or change historical native fixtures.

Before shared activation, fixed-checkpoint IEEE FP32 tests compare isolated and
shared output with atol=rtol=1e-5. Fix one reference request and change another
participant's private values, history/event lengths, candidate count, placement
and processing order: both another seat of the same game and the same seat of
another game. Require reference outputs/value and same-participant RNG selection
to be independent; private dependence fails even inside numerical tolerance.
Cover reused game/seat/checkpoint across executions/trials, wrong response IDs,
state revisions/digests, old-valid requests at publication, ended/failed/retired
requests, service failure and real cleanup. Isolated comparison can tolerate
environment/shape rounding only when unrelated to another private request.

Run focused regressions, full project Python/static checks, optional-free import
and package checks and relevant native tests if native interfaces change.
Independent correctness and separate architecture-conformance reviews must pass.

Physical budget is the approved seven HPO runs384games: matched width192,
collection16/concurrency3/seed17/four epochs/two updates in legacy and shared
modes with identical new contract/opponents/budgets; shared concurrency1/2/4/8/16
each collection32/two updates. Then one short ordinary MLP Model3 run at
concurrency2 verifies actions, PPO update, next-generation inference and cleanup.
Keep PPO minibatches/epochs fixed. Preserve original failed attempts and exact
identities; repeat only when near-ties/external interference affect a decision.
Select preset adoption from matched3 and concurrency from measured throughput
and headroom. Do not promise a speedup or infer learning quality.

The matched3 pair retains the existing mixed-refresh-16-v1 schedule and2GiB
collector. The separate scaling panel uses the already-supported generic HPO
32-game schedule (four Random, four heuristic and eight frozen seat-swap pairs)
at every point. The immutable mixed16 recipe must not be silently reinterpreted
as32. No new opponent curriculum/reader is introduced for this experiment.

For all five scaling points explicitly configure both
operating.collector_budget_bytes=4294967296 and actual storage.budget_mib=4096,
preserving global24GiB RAM/12GiB CUDA and existing queue/Pack limits. Existing
parent reply/journal admission reserves209,780,736bytes per active game;
sixteen plus one nested maximum journal operation/columns and writer base need
3,708,944,416bytes before retained groups/archives.4GiB clears this known floor,
not a completion guarantee. These parent copies remain distinct from child
search; writer/queue aliases within the same account are counted once. Final
host admission and actual collector peaks still decide whether16can run. Keep
this panel's conditions separate from the matched preset; never inflate budgets
or lower concurrency automatically after refusal.

GRU then Transformer remain subsequent model-specific work: separate
execution/game/participant/checkpoint hidden state or KV/valid prefix and
interleaved-history isolation before connecting their owners. No speculative
sequence serving implementation is required for this MLP milestone.


### External numeric audit capacity correction

The first actual C16 collection exposes an existing audit ownership mismatch:
its recorded live numerical receiver allowance is 512 MiB, while reopening the
NPZ and reconstructing complete originals needs a 583,949,647-byte additional
managed envelope. Live owned training receives sealed RAM banks and completed
both updates within its unchanged receiver allowance. External audit owns the
originals plus an independently specified scan capacity; it must use that
capacity rather than impose the producer's different live receiver limit.

In the private owned-cohort audit adapter, derive the additional numerical
allowance from the remaining original/scan account and the existing model cap.
The existing joint reservation callback must still admit all concurrently live
originals, retained scan state and numerical reconstruction before arrays open.
Do not raise recorded training budgets, change NPZ/JSON inflation estimates,
rewrite data, weaken checksum/Source comparison, or alter live restore/training.
No new public argument or persisted schema is needed. Insufficient external
audit capacity still fails before decoding. Test sufficient and insufficient
external budgets with unchanged recorded live limits, then audit both actual
C16 generations using a separately recorded corrected reader identity.


### Production soft host recipe after C16 calibration

Promote the provisional ownership model to `shared-mlp-host-calibrated-v1`.
The input is the actual worker count, game concurrency, model/queue geometry,
cache capacity and owner budgets; no seed, path or configuration fingerprint
selects it. It remains a planning estimate for the measured MLP topology, not a
physical upper bound for every supported history/search shape.

Use private CPU participant residency of 368,377,856 bytes per participant: the
largest observed individual private high-water in the completed actual C16 run,
including startup and subsequent retained pages. Do not add the full history,
search and token component graphs again. Use 439,175,168 bytes for aggregate empirical nonlearner PSS minus private
pages. This is not a proven pure/disjoint runtime allocation: it may overlap
Source-backed mapped pages already represented by the Source allowance. Bank
file lengths and mappings sampled at another instant cannot justify subtracting
that overlap. Keep this explicit soft-model uncertainty; do not add a second
full bank term or multiply shared pages by worker count. Revisit attribution
with simultaneous mapping evidence if admission becomes marginal.
The warmed component service private residency is 1,410,199,552 bytes; the actual
broker private high-water is 462,737,408 bytes. Coordinator private preload is 56,020,992
bytes, tracker residency 6,672,384 bytes, and each concurrently active simulator
uses the measured private estimate of 12,427,264 bytes. These physically measured
terms exclude the learner and collector-owned graphs. Calibration sources are
`runs/shared-inference-components/v2/topology32-frozen.json`,
`v3/controller-preload.json` and `v3/actual-c16-role-attribution.json`.

The component topology published one checkpoint. The actual two-update run held
G0 and the current G1/G2 generation, not a filled six-checkpoint cache; six entries
and 128 MiB are configured capacity, not measured occupancy. Sealed checkpoint
backing is owned by Source S and is already included there. GPU checkpoint
weights/positions belong to the CUDA calculation. Do not add either again as a
generic host cache-growth term. Per-module private metadata remains part of the
empirical service residency; a filled-cache or larger-model host measurement is
not claimed. Official publication's separately live CPU staging copy belongs to
sync and scales with the actual weight/position payload. Broker numeric
input/output staging is only the raw tensor/input/output storage increment at
its actual simultaneous copy lifetimes. Decoded JSON messages remain live beside
CPU TensorDict/futures; their Python/framing growth is calibrated only for the
observed inputs and is not bounded by that raw-byte term. Report warm allocator
reuse and this missing extreme-shape evidence explicitly; do not label the term
a complete broker bound. GPU estimates retain the existing exact shape/count
formula. Expose calibration geometry, observed occupancy and any
extrapolation in admission diagnostics.

Persistent RAM is the private participant total, shared runtime estimate, broker,
service, coordinator/tracker, simulator estimate and the existing
learner host compatibility allowance. Reachable phase increments are:

- collection: Source + collector + broker numeric staging;
- update: Source + numerical receiving;
- save: Source + numerical receiving + additional save export;
- sync: Source + publication host staging.

Parent P is nested inside Source S. Collector already owns its reply/table/writer
storage; do not add a second copy from physical phase residuals. Coordinator
Source graphs belong to Source rather than a second measured coordinator peak.
An independently retained numerical/export buffer remains distinct. Preserve
all existing live logical sublimits and CUDA limits. Preflight and actual
service owners call the same formula. The validated C16 case must fit the
unchanged 24/12 GiB global limits without an identity exception.

The second collection observes allocator retention after the first collection
and update, but longer runs can grow further. Full-cap search with maximum
reference mix remains unmeasured; the 223.7 MB search figure is an observed-mix
extrapolation and graph traversal omits allocator costs. Sampling can miss short
peaks. The recipe makes no hard heap quota or universal fit claim. Revisit its
coefficients when runtime/dependency/topology changes or actual phase PSS exceeds
the estimate; new recurrent/Transformer owners need their own recipe. Verify
arithmetic, no nested/shared double counting, unchanged logical/CUDA refusals,
and the remaining approved matrix before final adoption.


### External audit capacity containment (advisor resolution)

The initial audit correction correctly exposes a second existing ownership
mismatch: C16's complete 64-participant original reconstruction leaves only
367,251,044 bytes in a 2 GiB pool, below the 583,949,647-byte numeric envelope.
An explicit 4 GiB offline caller must not enlarge the original pool beyond its
recorded 2 GiB preprocessing limit. The original and numeric copies are different
allocations under an outer audit allowance. Producer settings/data stay intact.

The advisor accepts a capacity-based private adapter. Let E be the caller/model
external allowance and O the minimum recorded preprocessing/storage allowance.
When E <= O, preserve existing flat behavior: the original pool has capacity E
and numeric peak charges that same joint pool. When E > O, a fresh independently
owned external scan reserves the original pool's full capacity O in an external
account before constructing that exact concrete pool at O. Numeric absolute peak
reserves separately in the remaining external account, increasing only after
successful admission. Do not add original.used, cached groups or child charges
again: the fixed-capacity loan represents all child growth through O. Here
2,147,483,648 + 583,949,647 = 2,731,433,295 bytes fits the explicit 4 GiB caller.
This is a managed capacity proof, not measured physical RAM.

Keep this construction in the private owned external audit adapter; reuse the
existing account primitives. Do not replace a PackedArtifactPool budget or
loosen PackedSourceOwner's type, identity or preprocessing checks. Configure the
nested owner before first pool creation. Existing or borrowed pools retain their
existing behavior; never resize them or attach a scan-owned loan to another
custodian. Nested borrowed custody is deliberately deferred until a concrete
caller requires its own ownership design. Existing small 2 GiB audits retain
the flat path, including the E == O boundary. No fallback or budget expansion.

Attach the original-capacity loan's concrete cleanup owner to the scan before
admission/allocation effects. Release it only after confirmed original owners,
pool/native executors, borrows and scan-control cleanup in scan.close(). A failed
cleanup retains the loan and retry owner. Numeric reservations attach to the
existing cohort-retirement owner before decoding; aliases and failure tracebacks
keep them until the established last-use/close-hold retirement succeeds. Release
each reservation once, including retry after a release/retirement failure.
Returning from the reader or audit function is not retirement.

_ChainScan.admit_pool_controls already charges scan.retained to the same concrete
pool. Remove audit_owned_samples' duplicate original.charge(scan.retained) only
for the proven identical controls, retaining any genuinely outside ownership.
Keep all checksum, original/raw-record, snapshot, scalar/token correspondence,
Source health, 32x guards and actual retirement checks unchanged. No persisted
schema or policy feature changes. Before arrays decode, refuse actual aggregate
over-capacity and preserve original custody on failure.

Verify flat/nested/equality branches, 4 GiB outer with 2 GiB original pool,
aggregate refusal before arrays, absolute callback growth, independent original
pool growth, borrowed-pool noninterference, retained aliases/tracebacks, failed
and retried pool/native cleanup and one-time refunds. Then rerun the same C16
originals with separately recorded audit reader identity. No training rerun or
new capacity campaign is needed. This amendment supersedes using the original
pool remainder as the additional numeric bound only for the new nested case.
