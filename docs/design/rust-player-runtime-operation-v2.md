# Explicit invocation and settled-wave observation

Status: accepted bounded5c1B refinement after independent architecture READY;
implementation conformance and actual-source evidence remain required. Existing
learning/records/supervision contracts remain, except exact refinements below.

## Immutable execution request in operation2

Scientific stop/resume cannot be inferred from a deadline that already expired.
Do not temporarily pass an unrecorded pause keyword or mutate effective config
max_updates between pause and resume. Add explicit operation schema2, kind
rust-player-operation-v2. Its closed fields are ALL operation1 fields plus
execution_request. For ordinary-train and hpo-train that value is exactly
{mode:fresh|resume,pause_after_update:positive_integer|null}; booleans refuse.
The pause count is absolute committed-update count, with existing coordinator
bounds (greater than current and no greater than configured maximum). For initial
campaign-continue or independent cold-finalize operations it is null; auto cold
reentry carries the CURRENT training operation2 unchanged and does not retrain.

The operation file SHA binds the request along with config/runtime/run/context
and original deadlines. Persist the CURRENT kernel-authenticated bound operation
path, operation_sha256 and exact execution_request in the matching Run5
invocations entry (identified by invocation_id) BEFORE freezing state-{inv}.json.
Prepared1.state_sha256 binds that snapshot; Exit binds Prepared and completion
binds the exact Prepared/Exit pair. Cold audits reopen the immutable operation
and require this SHA/request to equal the actual current operation/SHA supplied
by the supervisor cold payload, not the earlier campaign launch operation.
This named immutable chain binds operational controls without changing unrelated
admission1/Prepared1/Exit1 formats. Resume retains every prior snapshot/operation.
 Mode must agree with actual parent lifecycle: fresh has
no initialized run state, resume requires the verified prior paused completion.
SAME operation cannot change any original bytes/deadlines/request; reentry uses
fresh invocation identity without extending original run/session authority.
A distinct resume invocation may have a different pause target but uses the
parent's same original budget and compatible effective config/producer.

Keep strict operation1 validation for historical control-only/component evidence;
unknown fields/kinds/versions reject. Do not silently reinterpret v1 as a new
scientific operation: B production training requires explicit operation2. No old
receipt becomes qualified data. Native fixed executable/bootstrap argv remains
unchanged. Native/Python validation supports the same closed version distinction;
launch1 and other accepted receipt versions need no unrelated global bump.

The fixed Python handlers consume the bound request to call the SAME common PPO
mode/pause_after_update controls. Public6 later writes it from existing user flags;
B internal calls also use genuine HostControl and the immutable request. It is
management/diagnostic data, never a policy feature or a fresh budget.

## Frozen computational run state and separate final completion

New Run5 variants may retain status as the common loop's computational state
(running/paused/complete/incomplete/failed). At healthy live closure record
host_result=awaiting-supervised-finalization as a POINT-IN-TIME observation,
never final eligibility. Public/internal operation return remains awaiting until
the exact cold result exists. New readers require their explicit new variant and
exact completion/exit/full Original/Replay/numeric/saved-state qualification;
status=complete alone never grants new Source/parent/evaluation authority.

Before Prepared, preserve an immutable canonical state-{invocation_id}.json under
the actual record directory, byte-identical to the frozen current run.json. Its
SHA is Prepared.state_sha256. Cold validates this actual immutable snapshot, exact
admission/Prepared/Exit and current state before final completion. It never
rewrites that snapshot/run to add its own completion hash. No run→Prepared→
completion→run cycle. All old snapshot files remain for later original audits.

After a verified paused completion permits resume, run.json may evolve for the
next invocation; the previous frozen state/pair/completion remains unchanged.
Completed computational state plus an absent/failed/foreign completion stays
ineligible. Failed live closure cannot record a healthy pending result. All
accepted old statuses/versions retain their old meanings; new fields/issuer
checks are closed and explicit in B's field inventory.

## Actual completed-wave reports without runtime shutdown

The immutable terminal transport is captured before its handoff ACK, so its
GameReport.storage_accepted is a pre-ACK fact. Never rewrite it or turn queue0/
consumer ACK into a fabricated final report. Add read-only completed_reports
for an exact unique set of full issued keys (run,trial,episode) after actual task
wait/settlement. Episode text alone is not unique across runs/trials.
It returns the actual completed immutable GameReports from the runtime, including
real publisher result, storage_accepted, inference release and errors, in requested
logical order, wrapped with its full issued key. Retain that key in the internal
completion ledger when submitting the actual GameSpec, without changing persisted
GameReport/terminal shape or the old join output. Reject foreign/unissued/duplicate
full-key/incomplete sets; same episode text with different valid keys is supported. Data-only
observation alone grants no Source authority.

The query neither joins nor stops persistent Rust workers/service and does not
readmit games. Use the same actual issued/completion ledger, synchronized with
its original owners. The facade and native/gf_native.pyi expose the narrow API;
no policy input, transport/persisted report format or process topology changes.
New logical Source retirement binds both unchanged pre-ACK original and actual
settled report/consumer state. Failure reports remain failures even when task
count reached completion. Actual buffer/GPU settlement checks remain necessary.

Verification: strict operation1/2/unknown distinction; explicit fresh/resume/pause
and original deadline/request hash mismatch refusal; actual paused pair/resume
state snapshots with no post-Prepared mutation; complete-status-without-completion
ineligibility; real wave completion before ACK/late failed publisher/refused
incomplete or duplicate query; two successive waves keep the same workers alive.
No new resource estimate, dependency, teacher input or learning-quality claim.


## Actual restored-policy owner across process lifetimes

A saved generation's original snapshot identity may include its original memfd
backings. A fresh resume process loads different actual readonly banks and did
not perform the original save. Use an explicit RustRestoredPolicy data owner,
issued only from full current cold/original scan plus held original checkpoint
files and newly validated readonly inference/resume banks under the SAME budget.
The original generation/saved snapshot identity remains a lineage reference;
actual current bank identities and owner are tracked separately. Neither is
silently substituted for the other.

Local restore derives or binds resume metadata to the actual held resume.json
SHA before any state mutation, retaining F1's saved-state integrity rule. Reuse
unchanged tensor/content validators via an explicit actual-restored-owner branch,
never actor9 envelopes, RemoteModelSnapshot or fabricated OwnedSaveAck. Registry
and service acquisition bind original generation/inference hashes AND the actual
current owner/banks. Positive close/last-consumer budget rules still apply.
Fresh G1 export/save produces its genuine OwnedModelSnapshot/OwnedSaveAck as
usual. Do not duplicate/re-save unchanged prior artifacts merely to manufacture
a new ACK. Checkpoint formats and numerical math remain unchanged.

Tests require a genuinely fresh-process bank load, different actual backing
identities with preserved original lineage identity, hash/metadata/foreign-budget
refusal before mutation, correct restored weights/Adam/RNG/PCG64 and real frozen
policy retention. CPU cold loading itself creates no GPU context; admitted
training restore uses its original bounds. Data owner alone grants no new Source
or final-parent qualification.
