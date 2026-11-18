# MLP shared batch inference after Simulator issue72

Status: bounded preimplementation design; architecture review pending.
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

GRU then Transformer remain subsequent model-specific work: separate
execution/game/participant/checkpoint hidden state or KV/valid prefix and
interleaved-history isolation before connecting their owners. No speculative
sequence serving implementation is required for this MLP milestone.
