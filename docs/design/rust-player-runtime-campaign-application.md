# Supervised HPO campaign application

Accepted implementation refinement of the public entry and terminal disposition
design. Independent architecture review READY on draft
`daff5f1c95e911af6f2c078b118ea74cbf0a586461e9e20b70734bf7c3c265b6`. All record schemas and Native protocol remain
as specified there. The application uses the existing Optuna proposal and
evaluation logic and the accepted evaluation DATA bridge.

## Dispatch and ownership

The public `hpo run` selects the fixed exec factory only for explicit schema 6.
Schemas 1–5 use their current implementation. The host dispatches from the
authenticated initial purpose. `run_campaign` borrows the real CampaignStore
from current HostControl and loads the authenticated CONFIG, settings and
session snapshots. Use the existing proposal sequence. Extract its repeated
planning logic into a shared pure helper if necessary. Each invocation
reopens durable state, takes one action at a time and uses the exact proposal.
Soft dispatch applies immediately before each training or evaluation start.
Any newly started operation fits the original hard session finish. The
controller remains in the only Python host, so no controller survives the
training-to-cold process boundary.

For a new seed, acquire the actual exclusive parent context once. Under that
owner write the effective CONFIG and new Op2/clock1 under the authenticated
supervision root. Publish immutable awaiting and CAS `awaiting-host` before
bind. Bind precedes actual `transfer_run_lock`. Call the same `run_hpo` through
a private optional campaign-only parent context and admission notification.
This context yields the exact registered outgoing RunOwnership. Training
checks authenticated initial campaign purpose, actual current bound operation,
parent path and original owner, then transfers the OFD and admits through the
existing control. After actual admission ACK and before creating learner or
Source resources, notify the caller with the actual HostAdmission so it can
CAS the same cursor to `awaiting-cold` with the actual invocation ID. Notification
failure aborts before learning and retains the interrupted record. The local
OFD context retires before Prepared, leaving Native custody. Standalone training
continues through inherited ownership. No training-to-CLI dependency or new
capability based on a callback. Retain the original terminal/error ownership.

Declare `training_failed` only from an admitted operation's application error
before any Prepared or uncertain protocol/record publication, following the
accepted failure protocol. The arm grants no cleanup proof. Native original
retirement gates reentry. In a genuine failed return or healthy incomplete
Cold, bind the exact current awaiting, proposal and physical pair, write and
verify the existing FAIL tell under actual parent custody, and submit the
canonical terminal disposition. Retire borrowers before the Native release
request and advance the cursor only after release ACK. Uncertainty stops
without replay or another bind. Use the existing control-complete protocol for
its permitted initial and failed-return exits. The normal finalized Cold exit
covers a session that successfully finalized a training operation.

## Evaluation execution

After cold finalization of an exact-D seed, persist seed2 with the actual
Parent and physical3 before releasing the parent. If evaluation starts before
soft, use the existing evaluation recipe and actual legacy Fresh worker path.
Create the current evaluation intent and obtain exactly one real GpuJob with
its original ledger, after training reap and before the Parent goes away. The
receiving Eval pool is independent of the Parent pool. Use the existing
materialized config, evaluation schedule, limits, intent and linkage formats.
With real GpuJob and receiving pool/phase, use `RustEvaluationData.from_parent`
to pin and acquire actual model file custody under the same borrowed parent.
Close Parent and release Native parent only after the handoff succeeds. For
an already closed seed, use `open_closed_rust_parent` under real current
RunOwnership, then `from_completed` within the receiving pool. Retire the
completed audit and its parent lock before collecting games.

Use the exact existing HpoEvaluationPlan, RetainedGameArtifacts and collector.
Both existing collector storage guards require a private explicit Fresh
schema-6 evaluation route. It must check actual RustEvaluationData, same
provider, exact plan, job and loaded config, with a validation split. Reuse
the existing Model4 storage/split/resource checks. Keep the generic legacy
mode rejection and other collectors as they are. The training runtime is not
parsed as an evaluation execution selector. The controller only uses CPU
metadata and existing admitted workers initialize CUDA.

Use the existing original collection and cleanup observation. Verify core
retirement before asking the same job to issue the existing GPU termination
receipt. That receipt describes actual job resources and allows the DATA
owner to close. Retire the provider through the observation, then the DATA
owner and its pin. While the verifier phase is still unentered, check the
existing outer-retirement assertion against the original plan, job and
provider. Release the receiving metadata reservation and pool only after
all remaining observation, plan, provider and core-result aliases retire.
Keep the actual confirmed job open and carry bounded receipt/linkage scalars.

Then run the existing independently owned saved-evaluation Original/Replay
audit without attaching the retired collector observation. This verifier
returns only after positive closure of its own actual Source, pool and FDs.
Charge the reconstructed expected linkage with the existing extra_owned_bytes
and owned-size estimate. The two whole memory allowances do not overlap.
Recheck the same confirmed job and exact acquired/released pin evidence before
publishing the durable eligible evaluation receipt. The legacy route retains
its default combined observation checks. Uncertain owners and the original
failure remain held through process retirement. A failed audit cannot produce
an objective. Foreign or unresolved evaluation intent stops instead of
repeating games. This lifetime ordering passed a separate architecture review.

## History and compatibility

Historical awaiting records remain immutable. Resume preflight can structurally
classify only exact candidates before initial exec. Preserve every interrupted
record and reject anything without either complete seed receipt candidates or
an already-acknowledged terminal disposition. Before any new work, the actual
CPU controller must revalidate successful training/evaluation histories using
C2b full independent DATA readers and actual new parent owners. It must also
check current GPU namespace resolution under the existing exclusive lock/scan
without opening a new job. Only verified past work can be skipped or used for
cursor reconciliation, sampler history, or a score. For resume, keep the old cursor and its SHA unchanged through exec. Native and
HostControl bootstrap authenticate its canonical path and the new immutable
bootstrap, independent of the mutable cursor contents. The fixed new CPU
controller verifies the exact old cursor and full historical evidence under
actual new owners before CAS to the new control cursor. Failure leaves the old
cursor in place and prevents planning, ask or bind. Fresh preparation retains
its current new cursor publication. Check existing namespace resolution through
a narrow helper that acquires the actual global lock and uses the existing scan.
No job or receipt is created by that check. No expired clock is revived and no
old Parent is reconstructed.

`CampaignStudy` retains control of Optuna history and tells. Its existing strict
trial verifier may receive the actual current HostControl for schema 6 only.
The trial aggregate dispatches each seed through the existing C2b full
verification under one actual new parent lock and funded pool/phase at a time.
Do not accept caller-supplied scores, projections, or detached history as
proof. Existing schema 1–5 APIs and behavior retain their defaults. This
requires no change to sampling, rewards, simulator or model math. A new audit
uses the existing configured operation wall within the current hard session
limit. Report resource/account lifetimes and scope rather than asserting that
this succeeds under any load.

## Verification and file scope

Review the dispatch, exact owner/identity pairs, deadlines, failure retention
and existing compatibility before implementation. Use unit tests for request
selection, cursor ordering/failure, original authority refusal and legacy guards.
Actual tests after a matching build must include first bind from CPU, genuine
exact-D cold handoff and existing Fresh evaluation, next distinct bind after
reap and receipt, history reopen and original forced/failure controls. A full
CUDA experiment follows the approved workload and selection protocol.

Proposed ownership is CLI campaign/host/hpo entry, shared CampaignStudy planning
and verification, evaluation composition and two existing collector call sites,
and narrowly scoped training entry/namespace helpers plus focused tests.
The C2b bridge and operation launcher are separate prerequisites. No runtime
experiment should be inferred from unit tests or a design review.
