# Local learning and Source composition

Status: accepted bounded task5c1 refinement after independent architecture READY;
implementation conformance remains required. Accepted main/integration/records/
storage/supervision contracts and ADR0054 remain authoritative. Pure config is
implemented at1472ae6; supervisor/native/writer/inference components are accepted.
The actual new training and Source paths remain to be implemented here.

## Management composition and actual configuration

GF CLI owns configuration parsing/composition. The supervisor operation binds
one immutable canonical effective-config JSON document by config_path and its
SHA; canonical no-newline bytes make that SHA the existing effective config
identity. The document records all actual settings, paths and materialized
runtime. Preserve it under the parent at a stable pathname through resume.
The host's LoadedConfig.inputs truthfully identifies this actual canonical input;
upstream TOML/preset inputs remain separately recorded diagnostic provenance,
without pretending the new host reread them. Do not silently reinterpret an old
parent or change old input/identity recipes. New-mode resume hashes the complete
recorded effective config/identity/runtime/producer as accepted in records.md.

The CLI bootstrap validates and reconstructs the concrete loaded Model3 or HPO6
configuration before calling training. Reuse existing pure model/action/training/
storage validators and dataclasses; factor a narrow normalized-data constructor
where required, without a second TOML parser or optional imports at pure loading.
Use configured_model_v4 and HPO-specific runtime-training/projection bounds for
HPO6; generic ordinary parsers cannot accept all valid Model4/HPO values. Every
serialized effective setting suffices for typed construction, but per-run study/
base/campaign hashes are references: do not claim independent upstream campaign
verification from a document that omits its search space/base inputs.
Do not fabricate a legacy campaign/proposal/process as an intermediate parser.
The data-only config has no admission or learning authority.

Refine the internal Python fixed handlers to accept keyword-only loaded config:
run_ordinary(control,operation,*,loaded), run_hpo(control,operation,*,loaded),
finalize_cold(control,cold,*,loaded). Training handlers return after actual Prepared
ACK; the cold handler returns an owned data-only finalized-invocation result,
with a complete-parent capability only when exact final-D applies. CLI owns
closing that result or consuming it for separate evaluation/campaign continuation.
The fixed module/function names and native
wire/launch/operation schema remain unchanged. The CLI bootstrap alone constructs
loaded; training never imports CLI to parse configuration. Initial campaign
composition remains in GF CLI. This is an explicit internal interface refinement,
not a plugin callback registry or arbitrary executor. Missing handlers/invalid
configuration still fail before admitted GPU use.

For fresh runs, management creates the actual parent directory and acquires its
real lock before writing canonical config/control artifacts; transfers the real
FD into the Rust supervisor (initial launch mapping or authenticated lock ACK).
The common coordinator uses that actual parent/run ID rather than creating a
second directory/UUID. Resume uses the same persisted config, original boot/
operation deadlines and actual parent lock. Fresh invocation/nonce is observational.

## Real local parent and numerical owners

Keep the common PPO coordinator and its closed concrete operation dispatch.
Add explicit Rust Model3/HPO operations; no caller-registered training loop.
The original run-lock context gains an explicit authenticated transfer state:
AFTER real transfer ACK, finally closes the local FD WITHOUT LOCK_UN. It may
retain its real FD until local mutation ends. A cold/inherited context registers
its genuine borrowed FD/stat/directory/PID/thread as an actual RunOwnership and
always closes only. It is not built from JSON/PID equality. Existing historical
run_lock still unlocks as before. Keep the actual registered owner throughout
all writes; stop local mutation/close borrower before releasing supervisor parent
custody. Separate-open competitors remain blocked across original host exit.

Local GameLearner and HpoLearner facades own actual same-process numerical state
and call unchanged update/save/restore APIs under serialized_torch. Retain
intentional learner RNG changes; inference construction/publication alone
preserves caller CPU/already-initialized CUDA RNG. Configure visibility/Torch
before concurrent GPU work, and require actual HostAdmission before production
CUDA allocation. CPU component work does not fake a global GPU job.

Model4 uses actual OwnedModelSnapshot/OwnedSaveAck and same Source budget/banks;
add explicit local-owner branches to existing registry/import/data validators
where they assume remote Wire7. Model3 likewise records the actual saved pair
and writer completion. No RemoteModelSnapshot/remote ACK/Popen/old GpuJob is
constructed. Checkpoints keep normal4/HPO5 formats and closed Python producer;
full Rust runtime producer is separate outer provenance. Preserve named weights,
Adam, CPU/CUDA RNG, PCG64 shuffle, counters, recipe and exact FP32 math/order.

## Native collection, Source and generation transaction

A concrete Rust collection owner composes ManagedServer and accepted native
runtime, inference and real thread writer. Native policies see only own typed
input/hints/perceived definitions; management seeds/Replay/receipts never become
features. Use the same existing paired curriculum/endpoint selection and reward
perspective, recording actual logical collection order independently from task
completion order. One outstanding inference/game, exact epoch/routing/RNG and
active/frozen checkpoint retains follow accepted components.

Native GameSpec/facade callers use the already accepted absolute cleanup_deadline
refinement, shared by HTTP/replay cleanup and terminal storage publication and
bounded by the original cooperative-close cutoff. No phase starts a new grace.

The new online Source owner derives from actual native pending/ACK/terminal
witnesses, same encoding contexts and real writer-seal/pending-descriptor custody.
Only exact accepted ACKs contribute transitions. New collection/proof/projection
issuance reuses existing Journal6/DTO3/codec2 and Python canonical numeric content
validators while binding the accepted runtime/player producer and real logical
retirement. The numerical batch is actual persisted/single-use and same-budget.
Do not recreate legacy actor9/10 capture, GPU history or old Source proof JSON to
enter existing private constructors. Factor data-only validators or add explicit
new owner branches; parent/Source/capture authority remains truthful.

The exact record versions/kinds are the accepted records.md table: normalRun5/
HPO5variant, gen4/6, collection10/11, proof4/7, update6/8, registry5/8 and actor11.
Do not globally bump old constants or weaken old readers. Root approves explicit
new fields needed for those meanings, and requires closed per-kind validators.
Full stable runtime binding combines materialized declaration, sampling identity
and actual Python/native/companion producer; all new outer facts link its digest.

Transaction: accepted originals→writer seal→Source and persisted numeric batch→
PPO update→same saved pair/actual ACK→generation commit→immutable G1 publication.
Source retirement means logical collection tasks/actual service consumers and
buffers/GPU work settled; persistent Rust W threads and cache may remain alive.
Queue count zero alone is insufficient. Retain G0/frozen references through actual
last service/actor use. Snapshot/Source/record aliases are charged once; release
loans at last consumer, preserving uncertain owners through actual join/reap.

## Accounting and cold completion

Use one exact native+external recording account, existing explicit Source/numeric/
save/parent phase budgets and truthful new local host admission. Do not carry old
per-process/IPC factors or sum mutually exclusive inference/update scratch. GPU
learner/cache parameters stay concurrently live; shared backing counts once.
Managed limits/soft planning versus measured allocated/reserved/PSS peaks remain
distinct. The explicit writer workspace loan comes from the SAME Source budget
remaining after actual retained snapshot/data owners; do not reserve a second
full2GiB or charge aliases again. Available planning allowance is not measured
usage or a new universal minimum. Any new restrictive estimate needs concrete existing allocation evidence
and root review; no guessed coefficient or safety margin. The limited v3 bank
calibration does not establish whole-host/C16 capacity or a hard Source RSS cap.

At a committed boundary, close original actor/service/writer/GPU consumers and
actual Source loans under original deadlines, then write the accepted cache/
loan/host-close DAG and Prepared. Live host status awaits supervised finalization;
it cannot issue final parent/evaluation eligibility. Rust actual original-host/
adopted waits produce Exit/physical termination, then cold Source validates the
exact pair plus complete originals/Replay/numerics/saved state before completion.
Cold validation is data-only CPU work before any separate evaluation admission:
validate stored CUDA/Adam/RNG bytes and original metadata without restoring a
CUDA learner, initializing device state or querying CUDA to create a context.
CPU inference inspection uses its own CPU binding without relabeling saved
training bindings. Actual CUDA restore belongs to an admitted training operation.
Only exact-final-D HPO obtains ColdFinalizedRustParent and the new parent/training
records. Healthy paused/incomplete data stays distinct and cannot grant objective.
Weights-only handoff uses a fresh existing evaluation Source budget/admission;
no legacy completion/termination/actor receipt is reconstructed. Historical
failed/incomplete intents remain ineligible. Original budgets never restart.

## Verification and scope

Require meaningful new/legacy/malformed-version/producer/runtime tests, actual
same-owner raw/canonical/SQL/numeric correspondence, zero-Act, aborted/GPU/writer/
failed-save and single-use behavior; real FD transfer exclusion; immutable G0/G1;
same-input local-vs-current Model3/4 CPU/CUDA numerical/saved-state controls.
Independent original/Replay/numeric/saved-state audits qualify any actual new
learning evidence. Real CUDA continuous2/pause1-resume2 and user CLI/campaign/
evaluation/process-tree acceptance remain6; approved scaling/adoption remains7.
Supply real internal callable composition, never eligible placeholders. Public
legacy entry guards remain until CLI supervisor activation; internal validated
handlers have actual HostAdmission/lock authority. No learning-quality claim.


## Parent-borrow lifetime refinement (accepted after architecture READY)

The accepted close-borrower-before-parent-release rule needs an actual local
HostControl check, not only caller convention. HostControl owns a concrete
private registration of its active/uncertain parent borrowers. The lineage
context registers the actual FD/owner lifetime and retires that registration only
after positive local FD close. release_parent and close refuse while such a
registration remains. An uncertain close retains a reachable owner/registration
for honest failure cleanup; no JSON counter or PID-absence claim clears it.

Keep dependency direction lineage→HostControl; supervision does not import
lineage to inspect a global owner map. Use one narrow private registration/token
for this actual parent-FD lifetime, not a generic resource registry/callback
framework. Existing registered RunOwnership identity/path/stat/PID/thread checks
remain. Old unsupervised contexts retain their original unlock behavior.

For outgoing transfer, disable legacy LOCK_UN once descriptor sending is
DISPATCHED, since an interrupted ACK may leave a real receiver alias. This is
uncertainty containment only: authenticated transfer/mutation authority is marked
only after the exact kernel-authenticated ACK. Dispatch failure cannot claim
transfer success, advance learning or release supervisor custody; keep reachable
uncertain ownership and quarantine. Positive local close uses no LOCK_UN and does
not assert a receiver owns the OFD when no ACK was received. Exact acknowledged
transfer and normal inherited borrow follow the same registration lifetime.

Verification: actual same-OFD separate-open exclusion through transfer/reentry;
release_parent/close reject a real live borrower; positive close permits release;
failed/uncertain FD close and ACK-loss cannot unregister or unlock aliases early;
wrong/copied/foreign registration cannot grant ownership. No wire/receipt version,
external dependency, numerical or scientific budget change is introduced.


## Pre-seal writer inventory (accepted after architecture READY)

Keep new collection10/11 manifest/report controls inside the same sealed SQLite
as existing Pack storage; do not introduce external controls that are absent
from the sealed descriptor. They need actual terminal/logical-retirement results
and exact raw-member inventory after collection, before final sealing.

Add one data-only inventory request to RustPlayerStorage's existing ordered
command queue and actual database thread. It runs after prior accepted bundles/
commits and refuses a complete snapshot until every expected group is committed.
Return the exact existing raw member-index rows from actual commit/database
state, without re-encoding candidate data or re-reading all raw payloads. Factor
existing index access if needed. Ordering/filtering excludes later control files
and manifest self-hashes; no hash cycle is created. The inventory is immutable
observational data, never Source or learning authority.

Caller composes actual report/manifest from that inventory and native logical
retirement, writes both through existing store_control RPC, then seals. The
actual descriptor therefore binds those controls with the unchanged canonical
Journal/columns/Replay. All RPC waits consume original deadlines; no new grace.
Single-thread database affinity, seal/admission fences, error/cancel/quarantine
and positive-close owner retention remain unchanged. No post-seal mutation.

Inventory object/metadata lifetimes belong to the existing explicit Source
workspace loan; serialized control bytes use the single exact recording account.
Share actual rows/backing rather than double charging aliases or keeping another
full payload. No new coefficient/margin/pool is approved. Tests cover real raw
index correspondence, ordered outstanding work, partial-pair refusal, control
persistence followed by actual seal/reopen, cancellation/error and last-use
retirement. Existing data-only writer remains insufficient for Source issuance;
new online authority still requires actual native/capture/service owners.


## Strict managed-Simulator cleanup (accepted after architecture READY)

The existing Bots ManagedServer legacy stop path gives KILL a fresh mandatory
one-second reap allowance. Preserve that default for historical callers, but the
new supervised collector must not start that allowance after its original cutoff.
Add a concrete constructor opt-in strict_cleanup_deadline=False; when enabled,
require the finite original absolute cleanup_deadline. Constructor ownership makes
startup-error/internal stop paths strict too, rather than relying on every caller
to pass a stop-only keyword. This changes the Bots adapter, never the Simulator.

All EOF/TERM/KILL and log-drain waits consume the remaining original absolute
cutoff; existing per-phase caps may shorten it, never extend it. Even at expiry,
an immediate owned KILL/nonblocking status observation may be attempted, followed
by truthful unconfirmed diagnostics when positive wait/log closure cannot finish.
Retain the actual ManagedServer/process/task/FD owner for original supervisor
containment; no PID-absence or timeout creates clean health. Cleanup errors keep
reachable owners and do not mint Prepared/Source or refund uncertain resources.
Do not block indefinitely on a live original process or add a fresh reap window.
Actual internal drainer cancellation contracts may be trusted; OS/filesystem
blocking limitations remain explicit and the Rust supervisor owns hard containment.

Existing default callers and their recorded overrun behavior remain unchanged.
New collection always enables the strict path with the authenticated operation's
cooperative-close bound. Tests cover actual normal cleanup, TERM-ignore/KILL,
spent-deadline/unconfirmed wait, startup error and cancellation/log-drain ownership,
and unchanged legacy default behavior. No engine semantics, persisted schema,
extra process/resource budget or new runtime dependency changes.


## Late binding of actual group contexts (accepted after architecture READY)

A collection fixes the full immutable paired-group/episode/policy plan before
work. Only up to C group servers are alive. When the collection exceeds C games,
the collector starts each group's games sequentially and obtains later process
receipts as it admits groups. When the full collection fits C, it starts the
servers to run the entire collection together, binds every original receipt, then
submits all planned games in one wave. Repeated-seed games have independent
identities and share the same assigned seed. The collector never invents receipts
or mutates a bound context. See [ADR0060](../adr/0060-rust-player-journal-handoff.md).

Add a narrow deferred_contexts=False constructor opt-in to the actual thread
writer. Default callers still supply every actual context up front. In deferred
mode the fixed complete plan is mandatory, and any initial contexts must already
be genuine; remaining complete group contexts arrive through one existing ordered
thread RPC after actual server startup and BEFORE that group's native game
admission. Bind only exact planned episode/run/config/policy identities and actual
validated executed runtime evidence. Publish the local bound state only after the
thread's actual ACK; no JSON response grants Source or action authority.

Duplicate, unplanned, changed or already-admitted late binding refuses. An
unbound group's raw bundle cannot be accepted for encoding; caller ordering and
the writer boundary must prevent ambiguous pending data/context replacement.
All waits share original deadlines. Cancellation/ACK uncertainty keeps reachable
actual owners; no successful seal follows incomplete/missing groups. Seal and
complete inventory still require every planned context and committed group.

Use the same database thread, single record account and explicit Source-workspace
lifetimes, without new admission/resource multipliers. Tests cover actual group
startup/bind/first-seat handoff/second-seat progression at C1, later-group binding,
exact receipt correspondence, duplicate/late/unplanned/unbound refusal, and
failure/cancellation ownership plus unchanged eager-context behavior. This is
metadata/data registration under a fixed plan, not a dynamic policy registry or
replacement for the new Source issuer.


## Scoped support implementation acceptance

Task5c1A support/data owners pass independent correctness and architecture READY
after five boundary corrections: saved metadata binds its actual ACK before
state mutation; normalized HPO retains all shared profile/materializer checks;
Mailbox enforces absolute publication/export/settlement cutoffs; strict drainer
failure remains failed health; deferred receipts validate before atomic binding.
The remaining codec/bootstrap/actions guard omissions close in a narrow second
review without new resource policy. Local numerical CPU controls and one actual
supervised synthetic CUDA comparison plus post-reap no-Torch saved-state audit
pass their explicit component scope. No new Source/outer-record/common-PPO/cold
completion issuer exists yet; task5c1B and user CLI/experiment6/7 remain required.


## Explicit invocation and settled report refinement

The [operation2/completed-wave/restored-owner refinement](rust-player-runtime-operation-v2.md)
passes independent preimplementation READY. It binds explicit fresh/resume/absolute
pause requests through actual operation SHA in immutable invocation state, keeps
computational status separate from cold eligibility without a hash cycle, queries
actual full-key settled reports without shutting persistent workers, and separates
original saved lineage from fresh-process readonly bank ownership. Old operation1
remains strict control-only compatibility; B production requires operation2.


## Mixed-participant cost and inference-only ownership refinements

The [participant-cost contract](rust-player-runtime-participant-costs.md) passes
preimplementation READY: GameSpec's default remains compatible while each policy
may declare its own effective costs. B uses each participant's existing settings
and Journal hash; no global maximum changes baseline admission. Saved-policy
inference-only retirement in [operation2](rust-player-runtime-operation-v2.md)
also passes READY: genuine full saved/committed facts precede positive unused
CPU resume-bank retirement, while actual inference/metadata owners survive through
last consumer. Neither refinement changes policy math or grants Source by itself.
