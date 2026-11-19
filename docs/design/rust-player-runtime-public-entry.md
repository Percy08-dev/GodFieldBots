# Public ordinary and HPO entry through the Rust supervisor

Status: accepted bounded integration design. C1 control protocol passes independent
correctness/conformance review; Part2 application wiring and full acceptance remain pending.
Architecture R1 READY on proposal SHA
9e368d6324aaf7b5ed7c27e11b6a1f3fe3769adbd8ae8d5dd9b221a21ca6c219. Owner: root.
Extends accepted integration/records/supervision/operation2 designs. Device account,
artifact retention, healthy empty parity and merged telemetry must pass their
implementation gates before final public experiment acceptance. No new CLI command,
learning algorithm, dependency or legacy completion capability is introduced.

## Existing public surface and routing

Ordinary remains gf-bots train --config PATH --mode fresh|resume --run-dir PATH
with existing --pause-after-update/--set behavior. At the explicit new runtime
selector branch, compose canonical effective config and immutable operation2,
then exec the installed verified companion. Historical modes retain the existing
train_card_instance_ppo route and readers. Do not simply bypass require_legacy_
player_entry or silently fall back when native/companion/planning is unavailable.

HPO remains gf-bots hpo validate/run/inspect and existing recovery/watch/compare
commands. There is no public single-operation train command to invent. Config6
run selects the supervised campaign route; config1–5 retain their old scheduler.
The SAME internal per-seed Rust learning handler is used by the campaign and
bounded timing harness. A harness may prepare one genuinely materialized operation
and exec its supervisor without keeping a Python driver alive; it must identify
itself as train-operation timing, not a completed Optuna campaign/fitness result.

Public HPO validate6 invokes the new same-host planning preview and existing
storage/science checks. Its output is explicitly startup/known-load feasibility
plus runtime phase checks, never HostAdmission or an all-future completion
promise. No old participant-process envelope or fake legacy joint admission.
Canonical control is checked using its actual new-mode profile as well as trial
settings. Keep evaluation max_participant_workers relation for legacy fresh eval;
do not turn it into Rust W, CUDA contexts or training RAM process counts.

## Launch and original ownership

The CLI performs pure/config/file/producer/native identity checks before any game
or CUDA allocation. Open actual original RunOwnership/CampaignStore as appropriate,
write canonical config/operation/control files, and pass the actual lock FDs in
validated launch mappings to exec_supervisor. Linux exec preserves SAME OFDs
without running a legacy finally LOCK_UN. On exec failure, local original owners
perform their normal actual cleanup. Never serialize an FD/lock/capability into
JSON and reopen a different description as a handoff.

Use discover_companion and sealed_launch_fd/exec_supervisor exactly as accepted:
installed distribution inventory/hash/build/ABI handshake, sealed bounded launch,
verified executable FD exec, no PATH search/runtime compile/missing-companion
fallback. Fixed bootstrap has explicit ordinary-train/hpo-train/campaign-continue
handlers only; operation2 request binds fresh/resume/absolute pause count before
state mutation. Original monotonic bits/boot/session remain authority. Ordinary
resume preserves its original parent milestones. New distinct HPO operations may
bind only after previous finalization/release, within original session finish.
Do not create a cross-session budget-renewal policy in this integration.

Add actual borrowed CampaignStore construction from authenticated inherited SAME
campaign OFD, validating directory/file/stat/PID/thread as existing store owners.
Borrowed close is close-only, never LOCK_UN; caller registration/liveness is
retained to its actual last control consumer. Newly acquired parent locks use
existing transfer_run_lock and authenticated ACK before actual admission; sender
close-only and uncertain-handoff rules are unchanged. Separate-open competitors
stay excluded through host exit/cold continuation and control handoff.

## Campaign continuation and one Python during training

Implement fixed gf_cli.rust_player_campaign.run_campaign and a shared explicit
new-mode branch of the existing campaign composition. Reuse actual CampaignStudy,
proposal seed/history, canonical materialize_hpo_run and existing evaluation/Optuna
recipes; do not fork scientific sampler logic or create a fake hpo-campaign dict.
initialize_campaign config6 uses the genuine new planner instead of legacy joint
admission. Initial campaign-continue work is CPU/control, with no pretend GPU job.

Before training, materialize the real proposal/seed configuration and acquire
actual local parent run_lock. Bind operation2 BEFORE transferring that SAME parent
OFD (native bind rejects preexisting parent custody), then wait authenticated
transfer ACK and obtain real HostAdmission/global namespace
admission before either CUDA role. Execute the existing run_hpo/common PPO path.
The current Python host then exits after Prepared; it cannot continue to evaluation
while its CUDA context remains alive. The Rust supervisor actually reaps it and
starts a fresh CPU cold continuation. This cold process may later become the next
training host under the already accepted finalized→release_parent→bind protocol;
there is no simultaneous Python campaign controller during training.

Cold finalization obtains actual Prepared/Exit/full Raw/Replay/numeric/saved state.
A real exact-D ColdFinalizedRustParent plus metadata3 and actual physical
termination3 is required before closed seed-training2. A generic paused/incomplete
completion cannot create a closed seed or objective; preserve existing campaign
incomplete/interrupted handling rather than pretending a D10000/maxupdates2 timing
parent completed science. No implicit retry of interrupted legacy intents.

Use the existing evaluation runtime and its independently acquired REAL legacy
GpuJob only AFTER first training-host reap. Evaluation artifact pin/release uses
the new parent's actual validated inference files/data owners, not reconstructed
OwnedRunCompletion/HpoRunResult/actor9. Before releasing native parent custody, create and durably register the actual
evaluation pin while the genuine Parent/borrower is live. Positively copy or
transfer its exact validated inference manifest/readonly weights into an
independent retained evaluation DATA account, preserving SHA/seals/original
identity. If copied, charge BOTH real allocations during overlap, then retire
the old one; no uncharged alias or guessed loan refund. The receiving pin must
positively own those immutable bytes before Parent.close and release_parent.
It is not an independent run_lock acquired on either side of a gap. Evaluation
consumes this actual retained pin under its real independent GpuJob. Later
reference-record transactions may borrow authenticated SAME OFD while native
retains it, or use actual original ownership after the protected data handoff.
Do not let the CPU continuation
initialize an unadmitted CUDA context. Existing evaluation workers/cleanup/receipt
rules remain authoritative; new training runtime is not relabeled as eval runtime.

After evaluation's actual resources close and receipt verifies, continue the
existing proposal/seed planner. Bind the next distinct training operation only
under original session authority. Reopen proposal/receipt/data state across
processes; never serialize live study/completion objects. Prescribed two successive
training operations with intervening existing evaluation are mandatory acceptance.

## Versioned awaiting protocol and closed receipts

Use a NEW strict awaiting record, not an old interrupted training.intent value:
schema1/kind rust-player-training-awaiting-v1 exact
{schema_version,kind,campaign_id,proposal_no,learning_seed,run_id,run_path,
config_path,config_sha256,runtime_binding_sha256,operation_path,operation_sha256,
supervision_id,session_clock_sha256,created_unix}.
It is data written before training dispatch, immutable and SHA-pinned in the
campaign's mutable new supervision cursor. No PID/nonce is invented before admit.

Cursor schema1/kind rust-player-campaign-supervision-v1 exact
{schema_version,kind,campaign_id,supervision_id,session_clock_sha256,
awaiting_path,awaiting_sha256,current_invocation_id,phase}.
phase=control/awaiting-host/awaiting-cold/finalized, nullable awaiting/invocation
only where not assigned. Admission's actual invocation is recorded when known.
Cold continuation can finalize ONLY its matching new awaiting operation/pair;
matching byte/path/config/run/proposal/seed/session is mandatory. Old intent
absence/interrupt semantics stay unchanged. Idempotency is exact immutable pair,
not budget renewal, second training dispatch or future-state guess.

Closed seed-training2/parent-validation3/physical-termination3 are the exact
already accepted records.md families. New campaign constructors require actual
ColdParent/physical data owners, never accept a shape-equivalent dict as issuance
authority. Reading a previously closed receipt across continuation is independent
data verification, not recreating that old live owner. Existing evaluation
intent/linkage/result schemas keep their meanings and explicitly dispatch new
training/source hashes. Persist no objective before exact-D/policy/eval checks.

For a later campaign resume with an already closed seed, independently inspect
its new record variants/complete source and current namespace proof under actual
campaign/parent ownership. Create a DATA-only final-policy import/pin if needed,
not a ColdContinuation for an old exited host or another typed Parent from JSON.
Keep strict producer/config/mode/schema binding; no implicit migration.

## Failure, verification and scope

Preserve natural outcomes, failed/truncated/aborted records, unsupported rules,
healthy incomplete/paused states and infrastructure errors. An absent/foreign/
failed Prepared/Exit/cold/source/receipt pair never becomes objective. Keep native
namespace quarantine/explicit recovery and same-session clock boundaries; no
new cleanup grace or PID-absence inference.

Architecture review must close this entry/record/ownership proposal before code.
Root assigns exact CLI/campaign/store/evaluation/runtime files to one implementer;
project presets/defaults remain Root-owned and opt-in until the measured adoption
protocol passes. TDD: new/legacy dispatch, pure validate6 vsactual admission,
missing/tampered companion before external work, real inherited/transferred lock
contention, awaiting pair matching/idempotency/interrupted refusal, actual typed
parent versus dict, incomplete no objective, legacy eval after actual training
reap, competing continuation and two operations under one original session.

Actual installed-wheel ordinary/HPO CUDA continuous2 and pause1/resume2, independent
Raw/Replay/numeric/saved audits, actual one-Python training tree, two HPO training
operations with intervening evaluation and force/failure cases remain mandatory
Task6 evidence. The unchanged full24-run timing/default protocol follows only
after acceptance. No synthetic/unit smoke or bounded private component substitutes.

## Closure decisions: clocks, bootstrap and terminal control

The five protocol gaps from the first review are resolved by the accepted choices
below. The terminal-failure refinement and durable-acknowledgment correction also
passed scoped architecture R1. Interfaces remain unimplemented until their task
and actual verification gates pass.

Clock record schema1/kind rust-player-campaign-clock-v1 exact
{schema_version,kind,supervision_id,boot_id,monotonic_before_bits,
monotonic_after_bits,unix_anchor_bits,soft_dispatch_finish_bits,
hard_session_finish_bits,max_operation_wall_seconds,mapping_recipe}.
mapping_recipe=soft-dispatch-plus-max-operation-wall-v1. Pair original monotonic
before/after and UNIX provenance as existing launch.clock_binding. Let t be that
original monotonic_after, S=loaded campaign.session_seconds, M=maximum declared
TOTAL operation wall budget among this campaign's training and existing evaluation
operations (including their current finalization, not a guessed margin).
soft_dispatch=t+S; hard_session_finish=t+S+M. Persist exact f64 bits/boot once.
Launch.session_deadlines.finish_bits=hard_session_finish; its clock_binding is the
record's original four anchor fields. Per-operation milestones retain their own
original checked budgets and finish<=hard_session_finish. No dispatch starts at
or after the SOFT boundary, but a previously started operation gets its complete
original wall budget; the native hard finish is not the soft dispatch deadline.
No added cleanup grace or operation-clock extension. Control closes as soon as
work is complete/boundary reached, within the fixed hard authority.

Same-supervisor cold continuation reuses this exact clock/ID and can finalize its
matching actual pair. A NEW user hpo run --mode resume opens a NEW supervision
session/soft dispatch interval only for operations not already dispatched, after
revalidating closed campaign receipts. It never renews an old training operation.
An unclosed old awaiting record remains interrupted/ineligible; a new session
cannot turn its JSON into ColdContinuation or redispatch it. No implicit recovery.
Already closed seed data may be independently reverified across sessions.

Canonical paths relative to actual campaign directory:
supervision/{supervision_id}/session-clock.json,
supervision/{supervision_id}/bootstrap.json,
supervision/{supervision_id}/loaded-campaign.json,
supervision/{supervision_id}/seed-settings.json, and
supervision/{supervision_id}/awaiting/{operation_id}.json.
Mutable cursor is supervision/current.json.
All generated IDs/basenames are validated safe, data files bounded/canonical and
written under actual same-store ownership; no path inference from PID absence.

Bootstrap schema1/kind rust-player-campaign-bootstrap-v1 exact
{schema_version,kind,supervision_id,campaign_directory,mode,loaded_campaign_path,
loaded_campaign_sha256,seed_settings_path,seed_settings_sha256,progress,
session_clock_path,session_clock_sha256,cursor_path}. mode=fresh|resume, progress
uses existing auto|plain|off. Seed-settings is the existing validated decoded
imports/source-provenance snapshot captured before exec, not late unpinned reads.
Loaded-campaign snapshot schema1/kind rust-player-loaded-campaign-v1 exact
{schema_version,kind,campaign_effective,campaign_sha256,base_effective,base_sha256,inputs}.
Reconstruct only typed CONFIG DATA with existing pure parsers; verify canonical
asdict equality, base SHA and existing combined loaded campaign SHA. No live
store/study/job/completion object is serialized. The bootstrap's path/SHA are the
initial campaign-continue operation's config_path/config_sha256; the sealed launch
binds that initial operation. Training operations keep their real LoadedConfig
path/hash meanings. Ordinary and direct HPO train-operation launches have no
campaign bootstrap and never infer a campaign from an arbitrary directory.

Fixed cold dispatch purpose derives from the AUTHENTICATED initial launch
operation kind (whose path/SHA remains verified), separate from CURRENT training
operation dispatch. Initial ordinary-train => ordinary result; initial hpo-train
=> direct operation/timing result; initial campaign-continue => mandatory verified
bootstrap/cursor and campaign continuation. Missing/corrupt campaign data is an
error, never timing fallback. Ordinary/direct results close before release/exit.
Campaign consumes genuine current Parent for seed2/pin publication, positively
hands evaluation data off, closes old owner before native release, runs eligible
evaluation, then plans/binds new work or emits honest terminal control completion.

A first campaign control host has no old ColdContinuation. After actual bind it
uses its actual current registered run_lock owner/SAME-OFD transfer ACK/admission
route, verified campaign/bootstrap/context; never require or fabricate a previous
finalized Cold. Later cold-to-host reuse keeps existing actual-current-owner path.
Campaign OFD remains native-owned across binds; only parent DATA borrowers block
parent release/rebind, not the genuine long-lived campaign control-store borrower.

For an initial campaign-continue that finishes without any training/admission,
add authenticated control_complete/control_completed frame with exact body
{operation_sha256,result_path,result_sha256}. It points to immutable schema1/kind
rust-player-control-completion-v1 exact
{schema_version,kind,supervision_id,operation_sha256,session_clock_sha256,status},
status=complete|session_boundary|no_work. This is control DATA, no training/Source/
GPU/fitness proof. Accept only the actual current control-only operation with no
active parent/admission. Native retains the original child, waits genuine exit0
and actual descendants, then records schema1/kind rust-player-control-exit-v1
{schema_version,kind,supervision_id,operation_sha256,control_completion_sha256,
host_original,wait,descendants,descendants_reaped,signals}. Forced/unconfirmed/
nonzero exit fails control completion; no fabricated admission/Prepared. Existing
finalized-cold normal control exit remains valid. This precise new protocol arm
requires implementation tests; it is not a generic arbitrary command/callable.

Add CampaignStore.update_supervision_cursor(value, *, expected_sha256) as one
narrow actual-store-owned canonical atomic compare-and-swap seam for ONLY
supervision/current.json and its declared kind. Validate current owner/path/
namespace and expected current SHA (None only if absent), write bounded temporary
file, fsync/replace/fsync parent, then return actual new SHA. Existing put remains
immutable. Caller cannot use this method for arbitrary record overwrite.

At startup enumerate only this campaign's bounded supervision awaiting files and
verify their exact kinds/hashes/keys. An orphan awaiting file without a closed
matching seed receipt is interrupted, not dispatchable, even if cursor update
never happened. A stale cursor may advance/clear ONLY after independent verification
of an ALREADY closed seed2 receipt and its exact awaiting/config/operation/pair/
source/physical chain. Preserve immutable history; do not silently erase an intent
or replay an operation. Use expected cursor SHA to reject competing/stale writers.
The existing actual campaign OFD excludes other writers throughout.

## Terminal proposal failure without fabricated completion

The remaining review finding concerns legacy proposal-failure continuation.
Root adopts the source-based recommendation from bot_advisor
`/root/rust_campaign_lifecycle`: a separate failed-operation return to campaign
control. Physical retirement and scientific completion remain different states.
This failure-boundary extension passed architecture R1 after the durable
disposition-acknowledgment correction below. Implementation is separately gated.
The unchanged legacy `hpo.py` proposal catch writes failure, performs an exact
idempotent FAIL tell, and proceeds unless resource-blocked. A blanket new-mode
fail-stop would change that behavior; accepting every nonzero exit would erase
protocol uncertainty. Neither alternative is accepted.

Add one authenticated `training_failed` request with exact body
{failure_path,failure_sha256}, and `training_failed_ack` with exact body
{failure_sha256}. The file is canonical, immutable, bounded by the existing
control-record limit, under the actual control root at failure-{invocation_id}.json.
Its schema1/kind rust-player-training-failure-v1 has exact fields
{schema_version,kind,supervision_id,invocation_id,nonce,operation_sha256,
admission_sha256,failure_class,reason}. failure_class=application_error; reason
is a nonempty bounded diagnostic string, never policy input or eligibility.
Require the current actual admission and operation, no prior Prepared, no second
terminal arm, no attached FDs, and the authenticated host's own IDs/hash. After
ACK this host performs no more training or planning and exits. The report asserts
application failure only; it never asserts cleanup, Source, Parent or objective.
AdmissionBusy/RecoveryRequired, protocol corruption, uncertain admission, failed
control-record durability and cold-audit exceptions do not enter this arm.

Unexpected EOF before an accepted terminal arm remains protocol failure. EOF
AFTER the valid failure ACK is an expected channel ending; this closes the
existing wait-versus-EOF race without weakening arbitrary-exit handling. Native
still retires the SAME original Child and descendants and writes the truthful
existing Exit with cooperative=false, prepared_sha256=null. Confirm namespace
physical-termination3 using the SAME actual Retirement, not report contents.
Failure reentry requires: no protocol/durable failure; actual exited status with
no signal or forced cleanup; positively reaped original/descendants; actual
namespace retirement durable; retained original campaign and parent locks
successfully rechecked; and remaining OLD operation finish. A signal termination,
unannounced exit, failed retirement/write/recheck or partial admission stops with
the existing recovery/quarantine policy. No new cleanup allowance is added.

Only an authenticated initial campaign-continue bootstrap permits this alternate
return. Ordinary/direct timing failures remain failed operations and stop. Native
starts one fresh CPU campaign-control host after the failed training host's actual
reap, retaining the same supervisor/session/campaign/parent OFDs. Its explicit
campaign-return handshake binds the current operation/admission/failure/Exit and
physical receipt path/hash pairs plus actual lock mappings. These are control
DATA. Introduce a narrow failed-operation control state, separate from Cold,
Prepared and finalized. It neither issues nor reconstructs ColdContinuation,
ColdFinalizedRustParent, Source or legacy completion. The fixed application handler
reopens only the authenticated campaign bootstrap and exact current awaiting.
No live Python controller survives training and no unadmitted CUDA is initialized.

While holding the actual retained parent custody, the control host writes the
existing immutable proposal failure.json, performs CampaignStudy.fail with its
exact durable proposal/reason, verifies the actual Optuna FAIL/value=None and
exact tell.intent/tell.receipt transaction, then writes a terminal disposition.
Do not replace failure files on idempotent recovery: existing exact records must
match the same derived reason and hashes. Close all actual parent-data borrowers
positively before release. The failure path gets a separate authenticated
`campaign_disposition` request with exact body {disposition_path,disposition_sha256}
and `campaign_disposition_ack` with exact body
{disposition_sha256,ack_path,ack_sha256}. Native checks
the actual current failed-operation evidence/awaiting/session pair and rechecks
held locks; ACK enables release_parent for that failure state only. It does not
set finalized=true. Parent release accepts either genuine acknowledged cold
finalization or this genuine acknowledged failure disposition, never shape alone.
Only after release ACK may the cursor advance and a distinct operation bind.

For a healthy cold-finalized incomplete/paused/empty result, preserve its actual
status and genuine cold completion/release path. It cannot issue seed-training2
or an objective. Before release, write the same terminal proposal disposition
with classification=healthy_incomplete and exact completion/physical evidence,
then use the same campaign_disposition acknowledgment arm. In this variant native
requires its actual finalized Cold pair, matching incomplete/paused Prepared and
ineligible completion, plus exact current operation/session. Genuine cold
finalization remains the release authority; the disposition does not replace it.
Proposal FAIL means the prescribed final-D condition was not reached. Existing
exact-D completed seed/evaluation paths remain unchanged.

Canonical disposition path is supervision/{supervision_id}/terminal/{operation_id}.json.
The strict schema1/kind rust-player-campaign-terminal-disposition-v1 has exact
fields {schema_version,kind,campaign_id,proposal_no,proposal_sha256,learning_seed,
supervision_id,session_clock_sha256,awaiting_path,awaiting_sha256,operation_path,
operation_sha256,invocation_id,classification,evidence,proposal_failure_path,
proposal_failure_sha256,tell_intent_path,tell_intent_sha256,tell_receipt_path,
tell_receipt_sha256}. classification=training_failed|healthy_incomplete.
For training_failed, evidence has exact fields
{failure_path,failure_sha256,exit_path,exit_sha256,physical_path,physical_sha256}.
For healthy_incomplete, evidence has exact fields
{completion_path,completion_sha256,physical_path,physical_sha256}.
Paths identify the same already-authenticated historical invocation and current
awaiting/proposal/session; exact files, kinds, producer and hashes must verify.
Nullable/unrelated alternative evidence is forbidden. The record grants no retry,
evaluation pin, Source, Parent or objective. References form a forward DAG: no
new backreference is written into immutable awaiting, report, Exit or completion.

Update the preceding cursor rule: advancement/clearing requires either an
independently verified already-closed seed2 OR this exact terminal disposition,
its durable native acknowledgment and already durable proposal FAIL, plus current
actual custody and resource resolution. Awaiting files remain immutable and
permanently nondispatchable after terminal settlement. A crash before the durable
native disposition acknowledgment remains interrupted/ineligible, even if a
failure report, truthful failed Exit/physical chain, tell or disposition file
exists. These records alone do not prove native accepted an eligible return.
There is no automatic later-session completion of that earlier FAIL transaction.
Resource-blocked or uncertain custody stops; no JSON absence, PID absence or lock
release authorizes recovery.

A NEW explicit user resume may independently reverify an ALREADY acknowledged
terminal disposition and its complete immutable failure/healthy-cold/proposal/
FAIL-tell/physical evidence under actual new campaign/parent custody and current
namespace resolution. It may reconcile a stale cursor for that already closed
proposal as DATA-only bookkeeping under the NEW session clock. It performs no
old training, new FAIL tell/disposition, old release, old continuation/finalizer,
Source/Parent/objective issuance or redispatch. It cannot reconstruct a live old
capability. Missing/conflicting acknowledgment or evidence remains interrupted;
no renewal of the old operation is permitted. This precise data reconciliation
is distinct from same-session failure settlement and parent release.

Native writes a strict immutable acknowledgment BEFORE sending
campaign_disposition_ack, only after validating the actual accepted return (or
actual finalized incomplete Cold), current locks, exact disposition/FAIL evidence
and healthy protocol/durability. Canonical path under the actual control root is
disposition-ack-{training_invocation_id}.json. Schema1/kind
rust-player-campaign-disposition-ack-v1 exact fields are
{schema_version,kind,supervision_id,invocation_id,control_invocation_id,
operation_sha256,disposition_sha256,evidence_sha256}. invocation_id is the OLD
training invocation; control_invocation_id is the ACTUAL current CPU control
channel's invocation. evidence_sha256=digest(disposition.evidence), which binds
the exact report-or-completion, Exit and physical chain through the disposition.
Recheck and durably publish this exact record before ACK; the reply carries its
actual path/hash. This record is native terminal-settlement DATA, never Source,
Parent, cleanup authority or proof of parent release. A reader uses the exact
canonical historical control root and strict whole-chain verification, not a
caller-provided shape or path alone. No immutable record points backward to this
future ACK, so the one-way DAG remains intact.

A report written without training_failed acceptance, unexpected EOF, or protocol
failure AFTER a training_failed ACK must never reach this acknowledgment: native
first retires the original, checks all eligible-return conditions, then starts
control. No disposition acknowledgment exists for a rejected return. Publication
failure stops; no success acknowledgment or release is issued. Same-session
control can finish its existing idempotent FAIL transaction only while the old
operation remains authoritative. Later sessions are limited to the already-closed
DATA reconciliation above.

Failure disposition and parent release must finish before the OLD operation
finish. Only after that acknowledged release does interoperation control use the
original session finish. Session/resume controls never renew the failed operation.
The failure host and replacement control host identities/waits are separately
truthful; do not describe a failed training host as cooperative.

Required bounded actual CPU process coverage: explicit declared failure -> actual
reap -> fresh CPU control -> FAIL/disposition -> release -> next distinct operation;
healthy incomplete/empty -> proposal FAIL -> next proposal; competing run/campaign
locks across both returns; EOF/nonzero without arm, signal/forced cleanup, control
publication failure and missing/foreign evidence refusing reentry; crashes around
FAIL, disposition and cursor writes proving no redispatch or objective; and a
boundary case proving the failure return cannot extend the old deadline. Existing
cold-audit failures remain fail-stop unless their positive custody settlement is
separately designed and reviewed. No tests or performance results are claimed by
this source-based design decision.

The failure-return wire phase is exactly `campaign-return` (a third fixed internal
bootstrap phase, never a user command). It uses fresh actual control-host
invocation/nonce in the normal authenticated hello/ready headers. Native then
sends seq1 `campaign_return` with exact body {operation_path,operation_sha256,
failure_path,failure_sha256,exit_path,exit_sha256,admission,physical_path,
physical_sha256,locks,deadlines}. The operation/deadlines/admission and referenced
failure/Exit/physical records describe the OLD failed training invocation; header
invocation/nonce describe this NEW CPU control process. Require those separate
identities exactly. Response seq1 `campaign_return_ready` has exact body
{failure_sha256,exit_sha256,physical_sha256}, no attached FDs. Sequence2 begins
normal requests. All lock mappings are actual inherited SAME OFDs, validated as
in cold. This state consumes the old operation's hard finish until release.
Python exposes these as read-only return DATA only while this exact current
HostControl/phase is live; it cannot construct a ColdContinuation from them.
The bootstrap routes campaign-return directly to the fixed campaign handler after
verifying the authenticated INITIAL campaign-continue bootstrap. It must not
redispatch current operation_type=hpo-train into training. The original launch
and current failed training operation retain their different meanings.

If no further operation is planned after acknowledged failure disposition and
parent release, use the same explicit control_complete/control_completed arm.
Its existing operation_sha256 binds the current settled operation and its result
binds the original session clock, while the actual native phase must be Session,
with no active parent/admission and the acknowledged failure disposition retained.
This is the one additional accepted control-only completion context beyond an
initial no-work campaign-continue. Native records control-exit for the NEW CPU
control host's real original wait/descendants. Its success never changes the
failed training Exit. An unannounced control exit before the explicit completion
arm is still an error. If a distinct next operation is bound instead, the previous
failure-return state is consumed and cannot arm another release or terminal
completion. Current cold-finalized success control exit behavior stays intact.


Preimplementation rereview R1 adopts the minimal durable-acknowledgment option.
The earlier proposal incorrectly allowed report/Exit/physical DATA to authorize
later settlement without evidence of native terminal acceptance. Root accepted
that P2; the original Exit1 and physical3 formats remain unchanged. Earlier
crashes remain interrupted, matching the existing no-implicit-retry boundary.
Required tests additionally cover report-without-terminal-request and protocol
failure after valid failure ACK, both leaving no disposition ACK and refusing
later recovery; lost acknowledgment response after durable native ACK permits
only exact already-closed DATA/cursor reconciliation, never old release/renewal.
