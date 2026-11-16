# Producer-owned resident HPO data

Status: user-approved plan; concrete design preimplementation review READY;
Task1 capture/RAM/FD foundation independently accepted, later integration pending.
Task2 direct numeric/snapshot/Wire7/actor9 components are independently accepted;
Task3 concrete coordinator binding preimplementation clarification is READY.
Task3 fresh integration is independently accepted after resource, retention and
startup ownership corrections. Task4 cold/completion implementation and the
subsequent audit-accounting and zero-Act identity corrections pass independent
correctness/conformance reviews. Whole CPU6868PASS/408SKIP and package checks
pass; fresh v2 physical acceptance remains pending.
Reviewer: `/root/owned_architecture`, 2026-11-16. Snapshot-bank visibility finding
resolved before dependent implementation; scoped final implementation reviews
are READY. Overall physical/evidence acceptance remains pending.
Date: 2026-11-16. Root owns design and acceptance.
Authority: `/home/percy/.codex/attachments/be1ed73b-89b8-419e-825e-8834eed8fa52/approved-plan.md`.
This supersedes the proposal choices, not historical evidence, in
[the investigation](hpo-producer-owned-data-investigation.md).

## Requirements and scope

Implement fresh and same-code cold pause/resume resident HPO for Model4 MLP,
GRU and Transformer. Its generated samples go directly from live capture to
training through immutable RAM banks. Archive, journals, numeric bundles and
checkpoint persistence remain required outputs. The normal new path performs
no historical sample replay, generated archive/numeric semantic readback, or
whole-batch CPU/device parity. Original strict readers remain explicit audit
and legacy execution paths. No Simulator, numerical recipe, RNG ordering,
dependency, Rust implementation or fixed numerical kernel changes.

Simulator remains pinned by the existing corrected-v2 contract; inspected HEAD
is `91d5926a7572db327eaea3751e7c24f428120996`, with section21.1 of
`07-external-interfaces-adapters.md` the information boundary. Existing contract,
catalog, layout and specification hashes remain authoritative. No new rule use.

## Responsibility and dependency direction

* `gf_common`: a small capture protocol, version constants and sealed RAM/FD
  transport primitives. No training import or Torch import.
* `gf_simulator_client`: invokes capture callbacks at actual decision, command
  acknowledgement and terminal-observation boundaries; its existing transport,
  retries, attestation and physical-process owners remain authoritative.
* `gf_training`: private capture builder, `OwnedHpoBatch`, `OwnedModelSnapshot`
  and `OwnedRunRegistry`; deterministic selection/encoding/GAE; ownership and
  transactions; separate learner Wire7 entry point and cold import adapter.
* `gf_models`: consumes frozen inference arrays through the actor's explicit
  owned-snapshot loader; policy receives existing own-view inputs only.
* `gf_cli`: explicitly selects producer origin/Python encoder and composes
  collector/campaign/evaluation. It does not manufacture live capabilities.

Resident pool binding must not import training snapshot types into the simulator
client. Use one concrete private actor-snapshot capability in `gf_common`, minted
only by the actual local learner export, successful Wire7 snapshot transaction,
or the dedicated validated cold-import transaction specified below.
It exposes immutable inference metadata/backings and owner-preserving liveness/
binding leases, with no resume bank, Adam/RNG/control or execution authority.
The pool accepts that exact capability; original owner identity and lifecycle
remain checked. It is not constructible from JSON/public dataclass flags, and
unbind cannot retire in-flight transfers or the origin's permanent ownership.

Use concrete private types with construction tokens/identity and consumed-state
checks. Public `EpisodeRecords`/`HpoRollout` instances remain strict components;
do not disable their validation or turn an arbitrary public object into an owner.
Pure token/GAE/array-layout construction can be factored for both paths.
Owned token handles also make their array/scalar/key/provenance bindings immutable;
sealed backing alone cannot prevent a mutable Python handle from shadowing input.
Legacy local update/restore cannot silently issue producer-owned generations;
the dedicated cold adapter owns validated restoration into that origin.

## Capture and episode state

The run coordinator creates one capture owner from its actual lock owner,
run ID, generation, immutable learner snapshot, exact CollectionPlan and explicit
learner seat per episode. Policy-name comparison never chooses the learner.
The environment gets only the lightweight injected capture interface. Each
same-seed group clone shares the collection capture but owns its episode handles.

After successful actual journal-header construction, the environment calls
`participant_created(episode_id, game_id, viewer, bank: EventBankView)` with that
participant's real readonly empty bank. Capture verifies the registered game,
actor, layout and genuine initial prefix, and reserves before cloning it. Both
participants register once. Later frames/deltas must match; seal cannot invent a
missing identity. This also preserves the real header identity when there is no
Act. Existing valid non-UUID bank identifiers remain supported. Direct nonempty
fixtures may derive identity from their actual original frame; zero-Act fixtures
must register their actual original view.

The explicit offline sample audit retains one decoded cohort at a time. Its
numeric callback admits positive growth in an absolute high-water request, not
the same peak repeatedly. After actual records/arrays and loop aliases retire,
the existing packed owner refunds decoded charges while retaining exported
control metadata and physical originals for final checks. A real pool close hold
keeps failure arrays/tracebacks charged through enclosing scan cleanup until last
use; the original error and exact cleanup owner survive. This audit returns data
and evidence only and does not issue a Source or live update grant. Its failure
does not authorize rewriting old files or relaxing raw correspondence.

Episode states are OPEN -> COMPLETE or FAILED; collection states are BUILDING ->
SEALED -> PERSISTED -> DISPATCHED -> UPDATED -> COMMITTED -> RETIRED, with FAILED
terminal from any stage. A failure after optimizer mutation cannot retry the batch.
Private history views preserve their shared admission through derived/nested
prefixes. Completion-send attempts are single use; uncertain send/receive/validation
poisons the channel while sender leases remain until actual child reap.

At selection, retain the same participant-local frame, event delta/bank prefix,
candidate table, selection/logprob/value, actual model identity and command UUID
used for journaling/Act. A command is PENDING until the actual positive Act ACK
or the existing positively verified idempotent retry resolves that exact command.
One ACK advances it once; retransmission cannot add a second transition.
Rejection/unknown delivery/failed health poisons eligibility; neither rejection
details nor replay enter features. Keep external response decoding/finite checks.

Complete an episode only after both real participant terminal observations agree
under the existing terminal projection, no pending command remains, and physical
episode health is successful. Handle natural termination without any Act. Keep
zero-decision episodes in population/order. Capture may retain both seats for
diagnostics/persistence, but learner records use only its own view. Terminal
outcomes are control labels with existing perspective, never policy features.
Unreplayed output explicitly records `replay_reconciled=false` and producer origin.

Seal in original plan episode order. Derive counts, endpoint selection, full
own-history tokens, candidate order, GAE/targets and numeric output from the same
retained graph. Preserve GRU/Transformer complete prefixes and configured detached
gradient suffix; actor hidden state/KV cache is never training input. Free capture
copies only when neither encoding nor required persistence can access them.

## RAM banks and transfer

The capture protocol's concrete operations are `open_episode(episode_id, game_id)`,
`participant_created(episode_id, game_id, viewer, bank)`,
`begin_decision(episode_id, viewer, frame, delta, table, selection, command)`,
`accept_decision(episode_id, viewer, command_id, acknowledgement)`,
`terminal_observation(episode_id, viewer, observation)` and
`close_episode(episode_id, result)`, followed by
`close_group(group, result)` from the actual group owner after server/client
retirement. These operations use an injected owner already
bound to the actual plan/run/snapshot; the caller cannot supply a new generation
or learner seat. The base environment calls an overridable ACK hook after actual
`client.act`, so verified retry follows the same single acceptance path. Final
observations are captured before journal completion, including no-Act games.
Episode completion and final batch admission also require successful group/server
cleanup; an early terminal observation alone never publishes a healthy batch.
`close_episode` runs after worker/Delete cleanup and before result persistence so
capture errors can be persisted as health failures. `close_group` rechecks the
actual episode health after result persistence, as well as server/client cleanup;
late persistence errors therefore invalidate earlier completion. Closure callback
failures poison capture and annotate health without replacing a known natural,
truncated or aborted outcome with a fabricated missing episode.

The private capture owner constructs the dedicated `OwnedHpoBatch`; a serializer
may factor existing packing calculations but cannot manufacture that owner from
an arbitrary `HpoRollout`. A bank builder copies provided arrays into its own
admitted backing. It never exposes a writable bank view; seal returns the new
readonly owner and consumes the builder. Direct learner views implement the
existing endpoint/vector protocol from token banks, without decoding historical
journals or rebuilding raw frame/event histories for validation.

Use Linux `memfd_create(MFD_ALLOW_SEALING)` and shared writable maps only during
construction. Reserve before growth/copy/tokenizer allocation. Close every writable
map/export before adding `F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL`.
Failure to seal is fatal. Expose only readonly NumPy views over sealed mappings.
Existing bank layout and dtypes stay unchanged; no disk spool or fallback.

Create a private Unix socket pair when spawning each owned child; inherit only
the intended endpoint alongside existing JSON control. Transfer FD with SCM_RIGHTS.
The existing participant `_spawn` may add private keyword-only
`pass_fds: tuple[int, ...] = ()`, forwarding a nonempty tuple to its SAME existing
Popen call. Ordinary calls retain their original keywords/lifecycle; do not copy
creation/cleanup/custody machinery into the resident worker. Only explicit actor9
supplies the owned endpoint, and bank FDs retain close-on-exec until transferred.
The channel has a random invocation identity and monotonic single-use transaction
IDs. Bind kind (batch/model/save), generation, model/snapshot identity, original
`fstat` backing identity, byte size, layout descriptors, dtypes and ranges to the
same transaction. Refuse arbitrary path loading and extra/truncated ancillary FDs.
Receiver validates external envelope/resource bounds/seals before mapping.
The sender retains its actual original FD and reservation until completion ACK
or actual child reap, and compares received ACK with its live transaction object.

Learner consumes readonly bank views directly. It must synchronize GPU work,
release tensors/NumPy exports and close mappings before a completion ACK.
Cancellation merely starts/join existing owned cleanup. Only actual ACK or reap
retires a transfer lease; uncertain transfer/reap retains owners and fails the job.
Tests use real socket/process ownership, not a Boolean claiming a child exited.

## Persistence, snapshots and live registry

Archive writer consumes captured journals/controls through existing durable writer
ownership. Persist complete archive and numeric outputs before dispatching update.
Generate their checksum/seal metadata once while writing, without semantic reread.
I/O, bounded writer completion and failure remain gates. Strict historical Source
qualification never follows from these producer records.
Factor `gf_models.sharded_arrays.write_bundle` into a private
`_write_producer_bundle` writer and its existing public inspect-after-write wrapper.
Only explicit owned writers call the private helper. Preserve finite checks,
planning, `save_arrays` and writer checksum generation; public signatures and
strict inspection remain unchanged. Do not duplicate the codec or add a public
skip/trusted flag.
For GroupPack publication, add a separate private producer seal command composed
only by the actual owned collection attachment. The owned writer derives the
descriptor from its original seal/control metadata and computes the physical DB
hash once after mutable writing retires. Parent publication/terminal handoff keeps
actual descriptor, inode, namespace, size and I/O/process-completion checks, but
does not semantically inspect or rehash its own generated DB again. Bind the
returned completion to the original writer/collection transaction; serialized
descriptor fields alone cannot issue producer authority. Existing public seal,
publication and external import/audit commands retain their strict inspections.
Keep current PlayerReplay HTTP fetching/persistence as diagnostic archive output,
with its existing external response decoder, limits, deadlines and I/O failure
gate. Use the already decoded `episode_end` once for interrupted abort/truncation
diagnostics; do not traverse it again to reconcile captured accepted actions.
Replay remains outside policy features and learning authority, and
`replay_reconciled=false` remains truthful. Its retained fetch/write cost belongs
in measurements. This clarification has independent architecture review READY.

After successful optimizer work, export actual learner parameters/Adam/RNG/control
once into an immutable `OwnedModelSnapshot`. It owns DISTINCT sealed inference-only
and resume-state banks: transferring a descriptor subrange cannot isolate data in
one FD's backing. Inference contains only current public inference parameters and
minimal model metadata; Adam, RNG and management controls stay in the separate
resume bank and never enter actor transport. Derive inference5 and resume5 from
that same snapshot; preserve current tensor layouts and numerical state. A save
transaction links the actual writer, generation, optimizer step, snapshot FD and
file map. Only its actual successful completion ACK authorizes commit. Partial
save, cancellation or post-Adam failure leaves the run failed; never rerun update.
The same writer retains the ACTUAL completed inference5 manifest/bundle descriptor
from its write result and includes it in the save ACK/Wire7 snapshot metadata.
Expose an immutable/copying `inference_manifest` view bound to the saved
`inference.json` hash and inference subset of the actual file map. Journal6
metadata and resident decoded-byte admission consume this actual writer output;
do not reopen the file or reconstruct its descriptor independently. Actor metadata
may include this inference-only manifest, never resume state/control. Admit its
retention/transfer before allocation; persisted inference5/resume5 formats stay exact.

`OwnedRunRegistry` is invocation-local and requires the actual parent lock owner.
It keeps current generations, controls, pins and retention transactions as they
are committed, without rebuilding the whole disk chain for each operation.
Publication remains complete save -> update/generation -> control/registry/latest.
Live reference mutations reconcile the exact actual pin/release/retention
transaction. Retention may delete only unprotected owned entries after publication.
Serialized JSON or an arbitrary same-valued dataclass cannot recreate authority.

`SYNC_OWNED` consumes only the inference bank of that same RAM snapshot via actor
protocol9; it never receives a resume-bank FD or optimizer/RNG/control metadata.
A transfer lease
ends at the synchronized ACK, while current/final/opponent-pool snapshot ownership
persists through its last use. Resident processes remain viewer-partitioned,
private episode state resets at existing boundaries, and actual reap owns cleanup.
Generation0 is also exported from the actual initialized learner, not reloaded
from its just-written checkpoint.
Actor9 explicitly distinguishes training use from evaluation-final-policy use.
Training keeps source-generation <= collection/control-generation. Evaluation
keeps its original custody/control generation0 and binds the selected committed
final source reference (parent, source generation, inference hash and snapshot
identity) through the actual `EvaluationPoolScope` and actor capability. The
receiver checks this evaluation-only wrapper against that fixed source and outer
owner/epoch; it does not rewrite model generation or enable future models in a
training process. The usage binding is management metadata, never policy input
or execution authority, and introduces no client-to-training import.
Freeze the permitted final-reference set in the actual scope; paired comparisons
may alternate two authorized entries across seat swaps. Every request/reuse binds
one selected capability, parent/source generation/inference/snapshot identity and
original viewer/epochs/control0; references outside that set are rejected.

## Limits and failure boundaries

Keep Source2GiB, numeric512MiB, collector2GiB and all existing configured GPU,
worker/save/storage/deadline limits. Reserve capture retention, token temporary
copies, growing memfds, simultaneous archive/numeric/save/transfer, model pool and
mapping descriptors before allocation. Charge each physical allocation once to
its existing owner; leases preserve charges until actual last use. No silent disk
spill, smaller workload, automatic native fallback or enlarged budget.
Keep finite input/output/loss/gradient/parameter guards and external decoders.
Process death, IPC uncertainty and writer errors poison the invocation.
Owned learner startup observes the original process-factory REQUEST before its
async creator can execute, with the observer initialized before socket allocation.
A never-requested original factory permits explicit unstarted endpoint cleanup;
PID absence alone never does. Guard actual creation-task/process/lease state and
do not use this shortcut for an unobserved alternate creator. Once creation is
requested, keep the existing exact creation/no-child/reap evidence and deadlines.
Advisor disposition after repeated startup refinements: both factory and actual
async body require the SAME active admitted learner reservation. The factory
records its exact one-shot coroutine; before endpoint export/environment work or
OS creation, the body requires execution by the original managed `creation_task`
running that exact coroutine and consumes its execution permission. Direct,
foreign-task and repeated body calls cannot spawn. An eager task may yield once
for original task registration before this exact check; never weaken identity.
Requested-but-unsettled close raises and retains ownership before inherited
settlement can infer anything from missing task/process fields. Use only original
task/failure/no-child/reap observations for release. This protects ordinary
private-method misuse, not arbitrary Python-internal mutation or unrelated OS spawn.
An owned pre-child validation failure has a separate operational cleanup edge.
Before any child-construction/start dispatch, retain the actual invocation's
untouched dispatch evidence, original audited job/ledger identity and empty
resource history. Retire all actual cold-import/control/FD/thread owners first.
Only while that original job remains held, with no pending creation or uncertain
cleanup, may it confirm operational termination and attach the actual receipt to
the original failure. This grants no training completion, does not reopen a job,
and never infers no-child from missing PID/resource rows alone. Already-dispatched
creation keeps the original lifecycle; uncertain cleanup remains unconfirmed.
Capture, local/remote snapshot parent ownership, numeric encoding and parent
Wire7 dispatch share the SAME actual invocation Source account, rather than
independent2GiB allowances. Numeric/save/parent phase limits still apply within
that aggregate. Known outstanding-view `BufferError` defers retirement and keeps
charges; it is distinct from an uncertain FD close and must retire after the
actual last buffer export releases.
Configured save-phase admission is distinct from the512MiB individual RAM-bank
capacity and numeric-phase bound. Preserve the original declared save allowance
(including the prescribed1GiB comparison setting) through the existing checked
phase-limit contract and actual Source/parent/model bounds. Never reject it merely
by applying a per-bank constant. For each received bank, cap size at
`min(individual_bank_limit, remaining_configured_phase_bytes)` while enforcing the
aggregate phase including controls; do not subtract aggregate usage after taking
the individual-bank minimum. No configured budget or physical-bank cap increases.
Parent metadata/provider/collection-manifest/control-writer reservations share
one phase admission backed by that same Source account. Check simultaneous
retained and temporary copies against the configured parent limit before any
serialization/copy. This is an implementation of the existing common admission/
lease interface, not another independent Source allowance.

## Compatibility and cold resume

### Concrete live coordinator binding

Keep the common `_run_ppo` loop and explicitly add the exact owned operations
class to its closed dispatch; do not copy the learning/experiment loop. Its owned
branch calls `initialize_owned`/`update_owned` and passes the actual
`RemoteModelSnapshot`/save ACK through commit. Legacy branches retain their
existing Source-issued projection, scanners and child methods.

`OwnedRunRegistry` is constructed under the actual `RunOwnership`, with the SAME
Source account used by its process, snapshots and capture. It stores actual
committed generation/control/file-map state, current/pool snapshots and real
pin/release/retention transactions. Initial commit consumes the actual Wire7 G0
save result; subsequent commit consumes one private `OwnedUpdateTransaction`.
That transaction binds the exact original plan/capture, actual completed archive
publication, persisted `OwnedNumericBatch`, actual dispatched process/current
snapshot, actual next save ACK and counters. It is single use and cannot be
reconstructed from an update JSON or an old Source projection.

Reuse the existing `ParkedAuthority` minted by actual
`ResidentPoolCustody.retire_collection(admission)` after END/PARK and physical
owner checks. Pair it with the new data transaction at update/commit; capture or
JSON never mints that physical authority. The same audited ledger's existing
`begin_resident_update` remains the concurrency gate.

The actual finalized producer collection carries its capture and the manifest/hash
produced by the archive writer. The common loop's owned branch consumes these
objects and the numeric writer's actual file map directly. Do not invent an empty
scanner to satisfy legacy `read_control`/`verified_digest` calls. Factor explicit
collection-result and update/commit branches/hooks where necessary; preserve old
readers and signatures for old callers. Policy bindings derive from live committed
snapshots. Owned resident SYNC consumes the same committed snapshot capability;
cache pins/evictions/unbinding and retention use that live registry, not full disk
chain validation. Storage usage, external runtime attestation and live owner/phase
guards remain required.

Exact new branch: HPO spec4, run variant `hpo-ppo-owned-v1`, Collection8, Proof5,
Numeric6, learner Wire7, actor protocol9, Update6. Model4, Journal6, inference5,
resume5 and array layouts stay unchanged. New spec explicitly records producer
origin and canonical Python encoder even if legacy preprocessing was Rust; no
native-reader rejection fallback. Old formats keep exact original validation.
Runtime parity is recorded as `not_run`, with no invented zero error or checked
endpoint counts. Strict readers/audits select the actual format explicitly.

Owned management records additionally use generation5, artifact-registry6 and
cache acquire/release transaction2. These explicit producer-origin forms keep old
generation4/registry5/cache transaction1 meanings unchanged. Generation5 binds
the actual snapshot identity, file map, counters, parent hash and schedule/pool
controls; registry6 owns per-file retention and reference states. New cache pins
use actor9's existing canonical `owned-inference-cache-v1` identity (policy SHA,
inference SHA and actual snapshot identity), while weights SHA remains diagnostic.
Record the actual source snapshot identity in the owned binding so cold metadata
validation does not depend on a transient live lookup. Cold readers dispatch on
these exact forms and never relabel old records.

Actor9 is an explicit new owned-snapshot dispatch, additive to the existing
participant protocol3/resident outer1. Legacy envelopes and Journal6's recorded
participant protocol3 retain exact meanings; new collection provenance separately
records owned actor9 transport. No existing actor8 implementation is presumed.

Cold adapter decodes required checkpoint arrays/control/RNG/optimizer/pool state
once within existing admission, validates external structure/hashes/version/code
identity and restores original deadlines. It imports metadata needed for lineage,
pins and retention without replaying samples not used by the next update.
The adapter issues new live owners only from this actual restore transaction and
lock, never from a persisted producer flag. Legacy resume uses its legacy route.

Cold import may read ALL generation/update/control JSON once to check ancestry,
counters, schedule/pool sequencing and reference/retention transactions. This is
control metadata, not sample replay. Read checkpoint arrays only for the current
resume pair and required pool inference snapshots; never open old journals,
collection samples or numeric batches to recreate learning authority. Preserve
required output descriptor/namespace/presence checks and reserve control decoding
before allocation. Full historical storage/sample audits remain separate.

Introduce private `OwnedColdImport` under the actual new lock and Source account.
Its one external load validates exact spec4/origin/code/runtime/device/deadlines,
current generation/file maps, inference5/resume5 keys/dtypes/shapes/finite values,
Adam steps and CPU/CUDA/PCG64 state. Factor existing private resume validators and
state application from their file I/O; public legacy readers/restorers stay strict.
The existing array decoder may add a private optional `_on_array` ownership
callback immediately after successful array decode, before subsequent fallible
validation. Cold import reserves its decode envelope BEFORE loading and uses this
callback only to bind actual buffer lifetime, including failures/tracebacks.
Validation is never skipped or weakened; default legacy calls behave identically.
The owner keeps admission until actual buffers retire, tracks provisional banks
before validation, and drains every owner while preserving the original error.

Wire7 gets an explicit restore transaction, not a legacy `restore()` call followed
by clearing an origin flag. The parent accepts only its actual `OwnedColdImport`,
sends admitted immutable model/resume banks through its original FD channel, and
binds request nonce/generation/steps/checkpoint identity to that live import.
The child stages network/Adam/shuffle state using the original application math,
publishes global Torch RNG only after validation/allocation succeed, synchronizes,
and issues a distinct `OwnedRestoreAck`. Failure poisons the invocation.

That ACK certifies actual restored learner state; it is NEVER a fabricated new
`RemoteSaveAck`. A restored current snapshot may carry this distinct origin ACK
and original validated file map. The registry imports its already-committed
generation without rewriting or recommitting it; the next real update emits the
normal new save ACK. Owned generation is issued by this private transaction, not
by a caller-supplied counter or JSON authority.

Required historical pool models use private inference-only cold snapshot owners,
loaded once under `OwnedColdImport`. The common actor capability may also be
minted by this actual validated import (in addition to local export/Wire7 save or
restore), with immutable data/lifetime checks and no execution authority. No
historical Adam/RNG bank is loaded for a pool-only model. Existing source versions
are not silently promoted into this producer origin.

Campaign -> evaluation uses a private completion object issued after actual final
commit and successful original job cleanup, linking final snapshot and completed
state. It authorizes a separate evaluation job and cannot reopen the closed
training job or derive execution authority from JSON. External evaluation imports
remain independently validated.
Prepare only an inference-data lease before registry closure; it preserves the
actual final inference backing/manifest and origin witness while resume banks,
optimizer/RNG/control payloads and ordinary registry ownership retire. Preparation
alone grants no completion. Mint the private completion only after actual physical
cleanup/termination evidence, final complete-state persistence and final original
limit checks all succeed. Failure closes the prepared lease and issues no success.
The final data scope must survive the old execution scope without treating that
scope/job as active; its last-use release remains positively checked.
Factor the existing received-bank retain implementation into
`ReadonlyRamBank.retain(budget)` with `ReceivedBank.retain` delegating. The same
admitted FD duplication/read-only mapping extends actual original backing
ownership without copying contents or retaining the full training snapshot.
Preserve all existing view, uncertainty, close and transfer-lease semantics.

Preserve the existing campaign session boundary between training and evaluation.
If evaluation is deferred to a later invocation, close any transient data lease;
the later call uses a dedicated validated external final-policy/receipt import
and a NEW actual evaluation admission. Persisted completion/termination records
are evidence to validate, never the private live completion or execution authority.
Version-aware external audit/receipt/reference dispatch supports owned management
forms without relabeling old records or claiming historical replay occurred.
All-owned external comparisons use validated inference-only owned imports and
actor9. Mixed legacy/owned comparisons retain the explicit strict external
inference5/RetainedGameArtifacts route, without promoting legacy origin or minting
producer capability for it. Existing incompatible-condition rejections remain;
this is deliberate profile dispatch, never a reader-rejection fallback. Its disk
validation is external comparison cost, outside the normal owned training loop.

## Implementation and verification sequence

1. Capture states, sealed RAM banks, FD transport and leases; tests for aliasing,
   wrong ownership/generation/seat, duplicate ACK, unknown delivery and reap.
2. Direct learner route, durable writes, snapshots, save ACK, SYNC and registry.
   Test partial save/post-Adam failure and cancellation with outstanding leases.
3. Remove replay/readback/parity from the new normal route only. Fixed captures
   for all3models must match strict features/order/endpoints/GAE/targets/gradients,
   RNG and optimizer results. Assert forbidden audit calls do not occur.
4. Cold adapter and campaign/evaluation composition; independent correctness and
   architecture-conformance review, whole project/static/package checks.
5. Thin benchmark: ADR0042 control retained in current code versus owned path;
   192width/2layers, seed17,4workers/2games,16games x2updates, unchanged budgets,
   3alternating matched pairs/model (18trials). No audits/tests during measurement.
   Report whole startup/collect/preparation/update/save/SYNC/cleanup/other/total,
   old parity/new capture-transfer costs, then collection5check positions/counts,
   capture/tokenize/seal/candidates/inference/IPC/archive queue-write distributions.
   Inclusive concurrent spans are distinct from exclusive parent wall sections.
   Preserve failed trials; no performance claim until measured and reviewed.

Legacy HPO's actual parity call may retain a private elapsed observation; the
existing optional diagnostic stderr envelope adds optional finite nonnegative
`parity_seconds` alongside `save_seconds`. Wire6 numerical metrics/meaning stay
unchanged. New owned execution records parity `not_run` and omits this duration,
never inventing zero or a successful check. Diagnostics supply measurement only,
never learning, commit or cleanup authority. Keep existing emitter byte bounds.

## Decisions and debt

No open requirement permits substituting actor caches, disk spool, replay-proof
labels or public unchecked dataclasses. Linux-only sealed memfd is deliberate;
unsupported systems fail early and keep the legacy explicit route available.
Existing local-filesystem soft deadline/durability debt remains; revisit on remote
filesystem support or hard real-time requirements. Captured Python object memory
uses conservative preallocation estimates plus existing retained accounting,
not an RSS hard quota; measure actual memory in acceptance.
