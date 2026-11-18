# Rust player outer records and authority

Status: accepted exact version/authority design after independent architecture
READY; implementation/conformance/verification remain required. Base: feec816.
Both original cache-reference and campaign-handoff omissions are resolved.
Requirements: [approved plan](rust-player-runtime-approved-plan.md),
[accepted runtime design](rust-player-runtime.md), ADR0054. Root owns this table.
Numbers below are per record kind, not a global version counter.

## Stable content and distinct online provenance

Keep actual unchanged Journal6/participantDTO3/candidate-columns-v2 content and
its external Simulator schema2/072dabb process receipt. Normal Model3 persisted
rollout remains schema4; HPO owned persisted rollout remains Numeric6. Existing
Python canonical feature/numeric encoding and Python checkpoint saver remain
responsible for those bytes. Their existing encoder/origin and checkpoint
producer fields keep their original meaning; Rust serving arrays are a separate
online encoding and do not falsely acquire canonical-python provenance.

Add one closed runtime binding containing the materialized runtime declaration
(mode/scheduler/C/W/B/formation), sampling-runtime identity and
rust-player-producer-v1. That producer records actual Python source digest,
native source/Cargo.lock/loaded extension hashes, compiler/target/profile/ABI/
features/flags and the verified supervisor artifact identity. Existing integration
identity separately pins Simulator/catalog/rules. Bind both in new parent identity
and its resume digest. Offline preprocessing remains a separate property; do not
use preprocessing.backend=rust as an online-runtime selector.

Per-invocation original supervisor/host identities, nonce, deadlines, actual
thread/IO observations, admission, joins and exit statuses are receipt facts,
not new random compatibility fields. They must not make pause/resume require the
same PID/nonce. Fresh invocation receipts link to the stable parent runtime
binding. Resume uses the recorded mode/configuration and exact compatible
producer/build; no implicit migration or cross-build compatibility is promised.
Historical audit verifies its recorded versions/bindings without relabeling them
as current producer semantics.

## Exact proposed outer inventory

| Record kind | New discriminator/version | Required binding/meaning |
| --- | --- | --- |
| HPO campaign config | schema6, runtime.mode=rust-player-runtime-v1 | Existing science/budget settings plus exact runtime; old config1–5 unchanged. |
| Materialized HPO spec | schema6 | Current owned-data fields with truthful producer-owned-v1/canonical-python numeric origin/encoder, plus explicit Rust runtime/player producer; no actor9/10 substitution. |
| Ordinary Model3 run | schema5, run_variant=ppo-rust-player-v1 | Packed storage, full runtime identity/execution; distinct from HPO Run5 and legacy/formal3/4. |
| HPO Model4 run | existing Run5, new run_variant=hpo-ppo-owned-rust-player-v1 | Strict new variant selects spec6/new records; old Run5 variants unchanged. |
| Ordinary generation | schema4 | Existing generation content plus runtime_binding_sha256. |
| HPO generation | schema6 | Owned snapshot/numeric origin+encoder content plus runtime_binding_sha256, under the new variant. |
| Ordinary collection | schema10 | Journal6/DTO3/derived6/candidate codec2 and exact runtime/player producer/capture authority. |
| HPO collection | schema11, actor_protocol_version11 | Numeric ownership/selection/snapshot facts plus exact runtime/player producer/capture authority; inner Journal6/DTO3 unchanged. |
| Ordinary original Source proof | schema4, kind=rust-player-source-proof-v1 | Existing original/content facts and qualification status plus runtime and actual collection-retirement binding. |
| HPO original Source proof | schema7, kind=rust-player-hpo-source-proof-v1 | Same plus HPO selection/recipe/snapshot facts, actual capture issuance. |
| Ordinary update | schema6 | Existing update/numeric/Source correspondence and runtime/collection-retirement binding. |
| HPO update | schema8 | Existing owned snapshot/save/selection facts and runtime/collection-retirement binding. |
| Ordinary artifact registry | schema5 | Existing artifact/ref facts plus explicit Rust runtime/cache-reference ownership. |
| HPO artifact registry | schema8 | Same with owned snapshot/HPO identity; no legacy actor/process ref spoof. |
| Inference cache acquisition | schema1, kind=rust-player-cache-acquisition-v1 | New same-host/service/source binding below; no broker_pid/gpu_pid. |
| Inference cache publication | schema1, kind=rust-player-cache-publication-v1 | Actual immutable publication acknowledgement linked to acquisition/source. |
| Inference cache release | schema1, kind=rust-player-cache-release-v1 | Actual eviction or joined host-cache retirement, not inferred PID absence. |
| Collection inference usage | schema1, kind=rust-player-inference-usage-v1 | Exact collection/producer/service/cache tokens and original request/publication boundaries. |
| Final parent validation | schema3, kind=hpo-rust-parent-metadata-validation | Actual cold-finalized parent/Source/final-D/artifact facts plus runtime/completion/exit binding. |
| Closed seed training | schema2, kind=hpo-rust-seed-training | New typed completion and physical termination3, plus exact proposal/seed/final-D/source facts. |
| Evaluation artifact pin/release | existing schema1 hpo-evaluation-reference-*-intent/receipt | Artifact-only generation/inference/registry transaction; extend parent reader to new mode, never reuse legacy service-cache refs. |
| Evaluation intent/linkage/seed result | existing schema1 hpo-evaluation-intent/hpo-evaluation-linkage/hpo-seed-evaluation | Same existing fresh evaluation semantics; training/source hashes point to explicitly checked new training/parent receipts. |

| HPO global admission | schema2, kind=rust-player-gpu-admission-v1 | Same per-user namespace/flock, exact context/ticket/budget plus original supervisor/host capability. |
| HPO physical termination | schema3, kind=rust-player-gpu-termination-v1 | Actual original-host/adopted-descendant exit receipt; never legacy resource PID list. |
| Namespace current index | existing schema1 | Same ticket/admission-hash pointer semantics; dispatch actual admission/termination format explicitly. |
| Local host admission | schema1, kind=rust-player-admission-v1 | Ordinary invocation-local authority; HPO body additionally binds its real global ticket. |
| Prepared result | schema1, kind=rust-player-prepared-v1 | Exact operation, committed update boundary, close acknowledgement and immutable data/artifact hashes; no physical termination/completion claim. |
| Original host exit | schema1, kind=rust-player-exit-v1 | Actual original wait status/nonce/admission/prepared digest and positive descendant retirement, cleanup diagnostics. |
| Cold invocation completion | schema1, kind=rust-player-training-completion-v1 | Exact prepared+exit pair, independent cold Source/saved-state validation, real stop state/eligibility. |
| Campaign handoff control | schema1, kind=rust-player-campaign-handoff-v1 | awaiting-supervised-finalization transition bound to exact intent/prepared/admission/exit; exclusive campaign/run custody and original deadlines. |

Existing Model3 checkpoint4 and Model4 checkpoint5, their closed producer object
{code_revision,source_digest}, weight/Adam/Torch/PCG64 layouts and mathematics do
not change. New resume recipes are rust-player-ppo-resume-v1 and
rust-player-hpo-resume-v1, hashing the complete effective config, identity with
runtime binding, execution compatibility, classification and storage binding.
A new selector alone is insufficient. Extra PID/nonce observations stay out of
that stable compatibility projection. Current producer/source checks remain
strict; failed historical parents never become eligible through a new reader.

## Same-host inference reference transactions

Replace the legacy shared-service cache receipt family for this mode; do not emit
shared-service-cache-reference rows with fictional broker/GPU PIDs. Registry5/8
uses its service_references token index with the new kinds above and persists
immutable acquire/publication/release records under that token's namespace.

The closed acquisition binding is {run_id, token, generation, inference_sha256,
generation_sha256, policy_sha256, weights_sha256, snapshot_identity,
runtime_binding_sha256, invocation_id, admission_sha256, service_id,
cache_identity}. Snapshot identity is null only for an actual validated external
inference source. service_id names the SAME real RustPlayerInference instance;
invocation/admission bind its real supervised host, not a compatibility PID.
The live acquisition is a private noncopyable owner retaining the actual source
snapshot/banks or validated staging, registry and service. A JSON row alone grants
no publication or memory-release authority. Persist acquire+registry ownership
before publication; failures keep a reachable owner until positive cleanup.

Publication record fields are {schema_version, kind, acquisition_sha256, ordinal,
checkpoint, generation, source_binding_sha256, ack, ack_sha256}. Its private
same-service acknowledgement is emitted only after actual immutable load from
that acquisition's inference-only snapshot/weights under serialized_torch, with
matching model/binding/checkpoint and source identity, and after synchronous
transfer completion. Ack fields are {schema_version:1,
kind:rust-player-cache-publication-ack-v1, invocation_id, service_id, checkpoint,
generation, ordinal, cache_identity, source_binding_sha256, weights_bytes}.
Never infer publication from a persisted intent or republish a mutable checkpoint.

Release record fields are {schema_version, kind, binding, acquisition_sha256,
publication_sha256, evidence, evidence_sha256}; publication_sha256 may be null
only when publication never completed. Evidence is exactly one of:

- {schema_version:1,kind:rust-player-cache-eviction-ack-v1,invocation_id,
  service_id,checkpoint,cache_identity,active_policy_references:0,
  native_pending:0,started_producers:0,consumer_settlement_sha256,entry_absent:true},
  minted from SAME service/controller after authorized game retains and original
  batch/result owners settle, actual eviction succeeds, and GPU work is fenced.
- {schema_version:1,kind:rust-player-cache-retirement-ack-v1,invocation_id,
  service_id,checkpoint,cache_identity,host_close_ack_sha256}, minted only after
  actual consumer/worker joins, buffer release, cache removal and GPU fencing in
  the real host-close acknowledgement. CUDA context/allocator lifetime may remain
  until host exit; this is cache retirement, not physical host termination.

No release is recorded on uncertain eviction/settlement; keep the real pin until
confirmed cleanup/host failure. Source/staging loans close only at actual last use.
A release-journal write failure invalidates parent eligibility while retaining
positive actual resource-cleanup facts. Cold registry readers check exact token,
acquisition/publication/release hashes and bound host/prepared/exit chain.

Collection inference usage fields are {schema_version,kind,run_id,invocation_id,
service_id,admission_sha256,runtime_binding_sha256,config_sha256,generation,
batch_id,start,end,bindings,collection_retirement_sha256}. start/end contain actual
request/publication/accepted/expired counters; bindings map each model policy to
its checkpoint and acquired token. Same-service quiescence and per-game owner
settlement establish the boundaries. No old registration/broker/GPU PID fields
or fabricated zero-child resource histories are introduced.

## Cold final parent and unchanged evaluation runtime

Define a private noncopyable ColdFinalizedRustParent data owner, minted only by
the exact prepared→original-exit→cold-Source finalizer. It retains the actual
parent-lock/scan/file/budget owners and immutable final state/generation/artifact
facts, with stable runtime and completion/exit hashes. It contains no active
training job, optimizer/resume bank exposure or legacy actor9/10 capability.
require() checks liveness, exact original backing/namespace and final-D identity.
A healthy paused/incomplete result uses a separate invocation-completion status
and cannot create this final-parent owner or a closed seed-training receipt.

Final parent-validation schema3 contains the existing exact scalar parent fields
(run/path/seed/config/state/generation/generation-chain/inference/used-decisions/
effective-settings), truthful numeric origin/encoder, and
{runtime_binding_sha256,completion_receipt_sha256,prepared_sha256,exit_sha256,
source_validation_sha256}. It is issued from that actual owner after full cold
original/Replay/numeric/saved-policy validation, not by converting an old
OwnedRunCompletion or trusting metadata/exit alone. Existing metadata-validation2
and its constructors remain strict for old modes.

Closed seed-training schema2 contains the existing proposal/seed/run/path/config/
state/generation/generation-hash/inference/used-decisions/source-receipt/GPU-receipt
fields and {runtime_binding_sha256,completion_receipt_sha256,prepared_sha256,
exit_sha256}. Its GPU reference is new physical-termination3 and its Source
reference is new parent-validation3. The schema6 campaign branch requires the
actual ColdFinalizedRustParent and typed Rust physical receipt and verifies exact
campaign/proposal/seed/settings/final-D equality. It never fabricates legacy
TerminationReceipt, OwnedRunCompletion or HpoRunResult instances. Legacy seed
training1 and parent-validation2 verification remain unchanged.

After this cold training receipt is durable, the existing evaluation operation
acquires its OWN real legacy GpuJob and artifact pin under the new parent's
strict registry/chain reader. Artifact-only evaluation reference intents/receipts
retain schema1 and their exact binding (run/token/generation/inference/generation
hash); no process identity is asserted there. Extend the reader's parent-mode
selection, not their data meaning or reconciliation authority.

A narrow new data handoff
ColdFinalizedRustParent.evaluation_artifacts(binding,label,*,evaluation_job,
budget,parent_budget,pin) supplies the existing RetainedGameArtifacts weights-only
fresh-evaluation provider with actual validated checkpoint4/5 inference files and
retained readonly source/backing custody in the NEW evaluation account. Factor
common data-only provider checks if needed; do not synthesize protocol9/10
snapshots or legacy completion/process authority. Retain the exact file/source
loan through real evaluator-worker/artifact retirement, charging shared backing
once; release or transfer actual parent-lock custody for the pin transaction,
never reacquire a conflicting lock against the supervisor-held original FD.
Cold external reopening must independently validate the new receipt/parent under
its real lock; a persisted receipt is not a permanent Source trust cache.

Schema6 evaluation explicitly selects the existing fresh runtime. Actual
participant protocol3, evaluation intent/linkage/result schema1 and evaluation
GPU termination1 remain unchanged and record their actual fresh execution.
Parent/training receipt validators branch on their new kinds/versions before
using hashes in existing evaluation linkage. Existing aggregate/objective math,
seed panels, final-D/final-policy conditions, failure/cleanup, and final-test
separation are unchanged. Thus legacy evaluation consumes truthful new-parent
inference data without any Rust training-capture relabeling or evaluation runtime
migration.

## Issuance and validation boundaries

Rust owns actual pending-beforeAct, matching ACK and terminal/original custody.
Its immutable native capability and unique reservation handoff are the live
capture witness. Python journal/numeric encoding may reuse content helpers, but
must not invoke a legacy Python process/capture authority after the fact as if
it had observed live Act. New collection Source/projection issuance checks the
SAME actual native/canonical-writer owners, complete natural/abort health,
original record hashes and real writer seal before publishing an optimizer batch.
Cold readers independently reopen originals/Replay/numerics under the new mode.
Never synthesize an old actor9/10, process receipt or admitted GpuJob from JSON.

Per-collection retirement is logical: actual game tasks/epochs, pending commands,
started admissions and original inference leases are settled, and immutable
originals are in the real writer/Source owner. The persistent Rust worker runtime
and immutable inference cache may remain alive across PPO updates. A Source proof
must not claim that those threads or the Python CUDA host have exited. Before
optimizer update, the Python consumer must also reach a real safe buffer/GPU
boundary under the shared Torch serialization protocol; queue-close/count0 alone
is not that proof. Saved-pair commit precedes G1 publication.

Physical host termination remains supervisor-owned after original wait and
adopted-child reap. Successful physical cleanup after forced/nonzero exit may
resolve operational admission while leaving data ineligible. Prepared result,
EOF, absent PID, closed queue, or manual admission release never qualifies
training. Ordinary local custody remains distinct from HPO global admission.

Legacy and new scheduling readers in the current release must interoperate in
BOTH directions on the same namespace, dispatching strict admission1/termination1–2
and new admission2/termination3 records. Never weaken legacy GpuJob live ownership
or construct empty legacy resource histories. Unknown/malformed/unconfirmed state
retains quarantine and existing explicit operator recovery. Benchmark controls
use the legacy execution mode on the SAME final source revision with these
scheduling readers, not an old executable unaware of new records. Preserve and
record any policy/math changes separately; none are intended by scheduling reuse.

Cold continuation requires exact prepared+exit+Source/artifact validation before
publishing completion. A healthy paused/incomplete invocation is distinct from
an exact final-D HPO parent; it grants no evaluation objective or qualified final
policy. Only the current exact-D/policy/parent gates may authorize existing HPO
evaluation. The handoff control can finalize only its new exact protocol intent;
old incomplete intents remain interrupted/ineligible. Retry is idempotent for
that one immutable pair and cannot extend original invocation/session deadlines.

## Verification

- New/legacy/malformed mode, variant, schema and producer mismatch/refusal tables;
  unchanged old bytes/readers/resume selectors, no global constant replacement.
- Unchanged Journal6/normalNumeric4/HpoNumeric6 content accepted only through
  truthful encoder and actual new outer authority; altered DTO/ACK/encoder rejects.
- Actual live capture and thread writer seal before Source/update; zero-Act,
  duplicate command/ACK, delayed result, failed writer and interrupted game cases.
- Native hash/config/producer mutation rejects new-mode resume; fresh PID/nonce
  under same compatible runtime does not invalidate a real paused checkpoint.
- Both directions legacy/new GPU admission contention, confirmed closure, orphan
  quarantine and unchanged explicit recovery; no manual release mints data proof.
- Prepared/publication/exit/reap/finalization fault windows, competing finalizers,
  paused versus complete final-D and actual installed CLI/evaluation handoff.
- Full Original/Replay/numeric/saved-state audits and approved measured protocol
  remain required; schema smokes do not claim learning quality or performance.
