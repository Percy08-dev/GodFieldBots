# Producer-owned resident HPO data

Status: user-approved plan; concrete design preimplementation review READY;
Task1 capture/RAM/FD foundation independently accepted, later integration pending.
Task2 direct numeric/snapshot/Wire7/actor9 components are independently accepted;
Task3 concrete coordinator binding preimplementation clarification is READY.
Reviewer: `/root/owned_architecture`, 2026-11-16. Snapshot-bank visibility finding
resolved before dependent implementation; final implementation review is pending.
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
only by the actual local learner export or successful Wire7 snapshot transaction.
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

## Limits and failure boundaries

Keep Source2GiB, numeric512MiB, collector2GiB and all existing configured GPU,
worker/save/storage/deadline limits. Reserve capture retention, token temporary
copies, growing memfds, simultaneous archive/numeric/save/transfer, model pool and
mapping descriptors before allocation. Charge each physical allocation once to
its existing owner; leases preserve charges until actual last use. No silent disk
spill, smaller workload, automatic native fallback or enlarged budget.
Keep finite input/output/loss/gradient/parameter guards and external decoders.
Process death, IPC uncertainty and writer errors poison the invocation.
Capture, local/remote snapshot parent ownership, numeric encoding and parent
Wire7 dispatch share the SAME actual invocation Source account, rather than
independent2GiB allowances. Numeric/save/parent phase limits still apply within
that aggregate. Known outstanding-view `BufferError` defers retirement and keeps
charges; it is distinct from an uncertain FD close and must retire after the
actual last buffer export releases.

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

Campaign -> evaluation uses a private completion object issued after actual final
commit and successful original job cleanup, linking final snapshot and completed
state. It authorizes a separate evaluation job and cannot reopen the closed
training job or derive execution authority from JSON. External evaluation imports
remain independently validated.

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
