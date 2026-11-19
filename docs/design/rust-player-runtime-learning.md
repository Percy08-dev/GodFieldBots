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
