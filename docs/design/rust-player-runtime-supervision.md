# Supervised host control protocol and lock transfer

Status: accepted exact task5b protocol after independent architecture READY;
implementation/conformance/kernel/installed-wheel verification remain required.
Base:de43f2e. Both initial immutable-record cycles are resolved. Root owns field choices. No numerical/Source completion
is granted by this component alone.

## Clock and identity

Use original absolute CLOCK_MONOTONIC deadlines from the existing checked caller
clock. Encode each positive finite f64 seconds value as exactly16lowercase hex
IEEE754 bits; Rust reconstructs it without rounding/reanchoring and checks actual
Duration/host-Instant representability. Include the actual boot_id. UNIX times
are observations only; do not schedule from rounded UNIXmilliseconds or impose
an unrelated2**53 numeric cap. Existing checked wall/monotonic anchors and rollback/
expiry policy stay caller-owned and recorded; supervisor cannot extend a budget.

Operation milestones are admission, execution, term, kill and finish, ordered
within the original session finish. Caller maps configured work/finalization/
server-cleanup allowances to them; no new grace or freshly renewed duration is
invented. On execution/stop request cleanup, then only original term/kill escalation
milestones permit signals; final wait/proof/cold work stays within original finish.
A new operation may bind its own original run milestones only after prior exact
pair finalization, under the unchanged original session budget. Resume of the
SAME operation cannot reset its stored deadlines.

supervision_id is stable for the original Rust supervisor/session. Each Python
host gets a fresh invocation_id and private32hex nonce, including cold continuation.
operation_id/run_id identify the actual operation/parent. These observations are
not stable checkpoint compatibility fields; new runtime binding remains separate.

## Transport and frames

Use an inherited Linux AF_UNIX SOCK_SEQPACKET endpoint with sendmsg/recvmsg.
Message boundaries keep SCM_RIGHTS ownership attached to one control frame and
avoid ambiguous partial stream ancillary-data association. Enable actual kernel
per-message credentials and verify original child UID/PID; socketpair SO_PEERCRED
alone reports creator credentials and is not proof of the post-fork child.
Each side has one outstanding request/ack; exact sequence/nonce/host binding is
required, unknown/out-of-order/truncated/extra-FD frames fail. EOF only requests
failure cleanup, never exit/healthy completion.

Frames are closed JSON packets <=64KiB; they carry compact identities/digests, not
originals, server histories, full source inventories or unbounded evidence lists.
Large prepared/close/exit evidence is immutable owned regular files referenced
by path+SHA under the original record namespace. This limit bounds control-frame
allocation and is not a new history/game/config setting. MSG_TRUNC and oversized
packets reject; packet byte limits do not truncate legitimate evidence files.

Common host frames contain {v:1,kind,supervision_id,invocation_id,nonce,seq} plus
only that kind's exact payload. Rust initial hello contains those fields and
{supervisor_pid,host_pid,host_start_ticks,build}. Host ready returns
{pid,start_ticks,python_abi,extension_sha256,build}. Compare actual original child,
expected Python/native ABI, loaded extension and matched source/lock/profile/
flags/target before admitting GPU. build is exactly {version,rustc,target,profile,
rustflags,cargo_lock_sha256,native_source_sha256,abi,allocator}; no arbitrary module
metadata. The runtime manifest is schema1/kind rust-player-supervisor-manifest-v1
with {executable_sha256,build,features,extension_features}; features is [] or
[allocator-mimalloc], extension_features exactly[pyo3/extension-module]. Actual
extension SHA is verified in launch/producer, not circularly embedded before the
extension build. Both artifacts must derive from the same frozen source/lock.

Closed request/ack kinds:

| Kind | Payload / effect |
| --- | --- |
| hello / ready | Exact original launch/ABI/build authentication above. |
| bind_operation / operation_bound | operation_path,operation_sha256,operation_id,run_id,config_sha256,runtime_binding_sha256,original operation deadlines; no active/unfinalized previous operation. |
| lock / lock_ack | Exactly one SCM_RIGHTS FD plus lock_kind(parent/campaign),device,inode; actual SAME open-file-description custody verified. |
| admit / admitted | Exact typed context,declared host resource budget,run/config/runtime binding and bound operation; returns actual local admission and HPO ticket when applicable. |
| prepared / prepared_ack | Owned prepared_path and prepared_sha256; verify exact bound record and durable ownership, ACK its digest. |
| stop / stop_ack | Reason code/request cleanup only; no termination/data claim. |
| cold / cold_ready | Exact previous prepared/exit path+digest pair and retained lock mapping under fresh host credentials; same original operation/session deadlines. |
| finalized / finalized_ack | Exact pair plus completion_path/completion_sha256 from fixed cold Source finalizer; does not itself implement or bypass that data finalizer. |
| release_parent / parent_released | Exact current parent FD identity after finalized data/custody transfer; no competing active operation. |

Compact reply payloads are fixed: operation_bound {operation_id,operation_sha256};
lock_ack {lock_kind,device,inode}; admitted {admission_path,admission_sha256,
global_ticket}; prepared_ack {prepared_sha256}; finalized_ack {completion_sha256};
parent_released {device,inode}. admit payload is {context,budget,producer,
operation_id,run_id,config_sha256,runtime_binding_sha256}, checked against the
bound operation. HPO context is the existing exact training JobContext;
ordinary context and expected_context are null (no fabricated HPO context).

## Launch and operation envelopes

Launch schema1/kind rust-player-launch-v1 is closed:
{supervision_id,python_executable,operation_path,operation_sha256,record_directory,
mode(ordinary|hpo),session_deadlines,clock_binding,binding,locks}.
clock_binding is exactly {boot_id,monotonic_before_bits,monotonic_after_bits,
unix_anchor_bits}: the three bit strings preserve the original checked caller
wall/monotonic observation, before<=after; wall is provenance, not scheduling.
session_deadlines is exactly {finish_bits}. Per-operation deadlines is exactly
{admission_bits,execution_bits,term_bits,kill_bits,finish_bits}; all values are
original finite positive16hex f64 monotonic seconds with ordered milestones and
finish<=sessionfinish. binding is {python_abi,extension_sha256,build}. Initial locks are optional parent and/or
campaign inherited FDs with {lock_kind,fd,device,inode}. Native verifies owned
regular inode, retains the FD and sets proper close-on-exec/inheritance per sole
host mapping. No arbitrary host command/entrypoint, namespace override or PATH
search. Production entry is fixed python -I -m gf_cli.rust_player_host.
The fixed host accepts only --control-fd N --launch-fd N --phase host|cold.
The launch FD is an original validated read-only regular/sealed descriptor held
by Rust; both sides use offset-independent reads and exact content/inode/hash
checks (sealed memfd where used), not an ambient mutable file path claim. It
supplies the initial operation reference and inherited lock mapping. Rust passes
only its actual mappings to the sole child; cold control contains the current
retained mapping, CURRENT bound operation path+SHA and exact previous pair;
it must not dispatch from an earlier launch campaign operation. These arguments select fixed bootstrap
phases, never arbitrary executables/callables. The cold completion content schema
is task5c-owned; 5b checks owned-file/hash/exact-pair linkage without qualifying
its numerical/Source content.


Operation schema1/kind rust-player-operation-v1 is a compact immutable control
file, checked against bind_operation, with {operation_id,run_id,mode,operation_type,
config_path,config_sha256,runtime_binding_sha256,deadlines,parent_directory,
expected_context}. Only campaign-continue may have null run_id/parent_directory/
expected_context before a parent is materialized; training kinds require actual
run/parent, and context follows ordinary-null/HPO-JobContext rules. operation_type is ordinary-train, hpo-train, cold-finalize or
campaign-continue, dispatched to fixed application functions. Missing task5c/6
handlers fail explicitly; no placeholder returns eligible data. Mutable campaign
scientific choices remain Python-owned, then bind exact operation before admission.

HPO may create/materialize a parent only after initial host launch. Therefore
SCM_RIGHTS parent-lock delivery after ready/bind and before admit is supported.
Ordinary admission requires one actual parent lock; HPO additionally requires its
actual campaign lock. Later operations may replace the parent only after previous
cold finalization/release; campaign/session custody remains continuously held.

## Actual flock ownership

Receiving an FD is not evidence that its lock remains held. Validate actual
inode/owner/type and the existing FLOCK ADVISORY WRITE entry for THAT descriptor
in Linux fdinfo, then require a separate-open competitor to be blocked. Calling
flock(LOCK_EX) on an unlocked received descriptor would silently reacquire it and
hide a custody gap, so it is not verification. Retain the SAME OFD without unlock.
Tests use a separate-open competitor to prove exclusion through admission,
prepared, original host reap and cold continuation. Host/continuation borrow the
retained mapping; no path reopen or JSON lock assertion substitutes for custody.

Existing run_lock/context finally calls LOCK_UN, which would unlock all SCM_RIGHTS
aliases. Task5c/6 must use an explicit authenticated transfer/detach seam: after
supervisor's actual lock_ack, a transferring host closes its local descriptor
WITHOUT LOCK_UN. New borrowed data owners never accidentally invoke legacy unlock.
Do not edit old ownership globally or fabricate a RunOwnership from JSON. Explicit
parent release follows actual Source/finalization/pin custody; all other copies
must follow the agreed close-only transfer contract. Unknown/unheld lock fails
admission/continuation, never a reconstructed exclusivity claim.

## Durable records and physical observations

All records include exact schema_version/kind from the accepted inventory.
Local admission1 contains {supervision_id,invocation_id,nonce,supervisor:{pid,
start_ticks},host:{pid,start_ticks,pgid},context,budget,operation_id,run_id,
config_sha256,runtime_binding_sha256,deadlines,producer,global_ticket}.
producer is exactly {schema_version:1,kind:rust-player-producer-v1,
python_source_digest,python_runtime_digest,extension_sha256,build,
supervisor_sha256,sampling_runtime_sha256}. python_source_digest is the unchanged
checkpoint producer digest scope (common/models/training). python_runtime_digest
covers .py sources in all seven application packages from root build configuration,
including client/CLI control code, with a fixed sorted relative-path/content recipe.
No source bodies or unbounded inventory is inlined. All digest fields are SHA256;
build is the exact nine-key object above. Git revision remains a separately
recorded observation, not a fresh PID/nonce-like compatibility field.

The closed budget records actual host declaration {managed_ram_bytes,
managed_cuda_bytes,game_concurrency,player_worker_threads,max_batch_size,
storage_bytes,source_workspace_bytes}; it does not relabel legacy process-worker
slots as Rust thread count. Ordinary global_ticket is null.

Global HPO admission2 retains ticket/context/boot_id/owner_pid/admitted_unix plus
host_admission; owner_pid is actual supervisor and context is the existing exact
JobContext. Host admission binds the SAME real global ticket. Strict known old/
new namespace readers retain immutable context/hash/current-pointer semantics.
Old live GpuJob PID/thread/resource authority remains unchanged.

Prepared1 contains {supervision_id,invocation_id,admission_sha256,operation_id,
run_id,config_sha256,runtime_binding_sha256,state(complete|paused|incomplete),
committed_update,generation,state_sha256,generation_sha256,artifacts_sha256,
source_manifest_sha256,close_ack_path,close_ack_sha256}.
The close acknowledgement file is schema1/kind rust-player-host-close-ack-v1,
closed fields {supervision_id,invocation_id,operation_id,run_id,
admission_sha256,runtime_binding_sha256,actors_joined,service_joined,writer_joined,
buffers_released,gpu_fenced,native_retirement_sha256,writer_retirement_sha256,
service_retirement_sha256,cache_retirement_sha256,servers_path,servers_sha256,
server_count}. Required positive close booleans are actual cooperative facts.
The separate server file is schema1/kind rust-player-server-retirement-v1 with
{supervision_id,invocation_id,operation_id,servers}; each entry is
{pid,start_ticks,exit_code,signal,logs_closed}. The count/digest match the actual
file. Large histories/evidence stay in original owned files, not the64KiB frame. These are cooperative facts, not supervisor
physical termination or Source qualification. Earlier failed/historical intents
cannot be relabelled as this protocol. Prepared has NO backwards handoff digest.
After prepared and physical exit are immutable, the HPO-only campaign handoff
control binds their exact pair plus the preexisting operation/intent/admission.
Ordinary mode has no campaign handoff record (explicit null in a shared cold
payload if that slot is present). This is a one-way DAG, never Prepared→handoff
→exit→Prepared.

Exit1 contains {supervision_id,invocation_id,nonce,admission_sha256,
prepared_sha256(nullable),host_original,wait:{exit_code,signal,core_dumped},
descendants,signals,descendants_reaped,cooperative,finished_unix_ns}. Each adopted
entry contains actual {pid,start_ticks,pgid,wait}; signals is a list of {pgid,signal,delivered}; delivered=false only for actual
ESRCH, while other signal errors prevent confirmed cleanup. It records actual owned
group signal observations. Positive original-handle wait/subreaper retirement
remains mandatory; WNOHANG0/EOF/PIDabsence are not exit. Stop/join own telemetry
child helpers before adopted-child drain; uncertain/outgroup/wait/durable error
retains quarantine/no continuation. Forced cleanup never qualifies data.

Global physical termination3 retains {ticket,admission_sha256,context_sha256,
boot_id,confirmed_unix,exit_sha256,exit_record}, with no fake legacy resources.
Write durable exact exit before termination and admission resolution. A failed
host can establish physical retirement only; its parent stays ineligible.

Before any admission exists, setup/control failure still reaps actual original
children, but does not mint an admission/exit-resolution record pretending an
admission existed. An optional diagnostic is schema1/kind
rust-player-startup-failure-v1 with {supervision_id,invocation_id,operation_sha256,
phase,error_code,host_original,wait,descendants,signals,descendants_reaped,
finished_unix_ns}; absent actual host/wait is null. It grants no data or admission
resolution authority. If any admission/durability step was partially published,
retain quarantine/no continuation rather than inventing clean rollback.

Cold continuation gets a fresh host identity plus exact pair/locks/deadlines.
Only wait0+cooperative/no forcedcleanup+positive descendants permits the successful
cold-data path; actual task5c Source/parent/final-D validator must mint completion.
A finalized frame only acknowledges that immutable typed finalizer output; it
cannot turn file existence, exit0 or five close booleans into training eligibility.
A paused/incomplete result never authorizes final-D evaluation/objective.

## Verification

Required actual kernel FD/credential/sequence/truncation, monotonic/wallrollback,
clock-representability, controlled lock-detach/competitor, original host normal/
forced/reap/adopted/EOF/escalation tests, both old/new namespace directions,
quarantine/recovery and durable proof failure. Verify new operation cannot reset
old deadline or run before exact prior pair finalization. Actual fixed dispatch,
installed companion/extension identity and onePython training tree remain task5b/6
checks; no scientific/model-quality/performance acceptance follows from protocol.
