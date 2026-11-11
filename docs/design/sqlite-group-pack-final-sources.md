# Group Pack: formal controls, final64 and Dataset/Tree

Read-policy amendment: [ADR0035](../adr/0035-pack-checksum-read-policy.md) and
[the accepted Pack checksum design](pack-checksum-read-policy.md) supersede
eager/full-rehash reads and custody-v1-only comparisons. New custody2 and strict
historical correspondence compatibility preserve original Dataset4 bytes,
learning eligibility, ownership and all persisted format versions.

Date: 2026-11-10. Owner: root. Status: final integration design accepted after
independent architecture review; implementation/conformance remain pending.
Depends on the reviewed foundation/runtime/Python Source/Native3B designs in
ADR0034. This is storage adoption, not authorization for a formal campaign,
shorter experimental budgets, HPO, real-source fitting or final-test evaluation.

## Explicit formal storage variant

Preserve original Run3/formal-parent-v1, recipe871fc8d4 and every old declaration,
stop, Config and dataset. No retrofit, mixed-generation parent or migration.
Packed formal parents use Run4, `run_variant='formal-packed-parent-v1'`, and a
new frozen variant recipe referencing the complete old literal rather than
copying/changing its learning algorithm. The variant adds storage only; existing
model/device/backend/seed/path axes remain exactly those accepted by
`checked_config`. Frozen1000update,10GiB disk,1GiB admission and
7200/5400/7140/60second budgets remain literal. No short smoke can qualify.

The variant asset is canonical sorted compact UTF-8 JSON plus one newline, at
`gf_common/contracts/card-instance-formal-ppo-group-pack-v1.json` and its identical
docs/contracts copy. Exact object:

```json
{"algorithm_recipe_sha256":"318eab51e51ab242c629111634ff25e986c1a0a0ece0d246694fd72a27c90cc9","base_recipe_id":"card-instance-formal-ppo-v1","base_recipe_sha256":"871fc8d4591e62d302bb2f79d98d474da15cbdb19c86506e5a70557f4ecd67b2","config_projection_recipe":"formal-config-remove-exact-group-pack-storage-v1","declaration_schema_version":2,"kind":"card-instance-formal-ppo-storage-variant","path_fields":["storage.db_root","storage.ram_root"],"recipe_id":"card-instance-formal-ppo-group-pack-v1","run_schema_version":4,"schema_version":1,"stop_schema_version":2,"storage_defaults":{"budget_mib":2048,"format":"sqlite-group-pack-v1","max_pack_mib":256,"queue_mib":512}}
```

Bytes SHA256 is
`27853b108089667e3e5f3e8f9d54c202b2f76583334f40954f1a1ab2a6844a47`.
Root materializes these project-configuration assets only after this gate.
Validate both the variant and its pinned base recipe/kernel AST. Formal storage
table has exactly format/db_root/ram_root/budget_mib/queue_mib/max_pack_mib:
format/sizes equal this literal; roots are nonempty absolute configured paths.
Removing ONLY that exact validated storage table yields the logical Config;
`checked_config(logical,base_recipe)` must equal the entire frozen old variant.
Other field changes cannot be hidden by a projection. Full Config, input paths,
storage roots/binding and hardware/runtime identities remain independently bound.

Packed formal Run4 required/optional keys are Python3A run4 keys plus the same
FORMAL_RUN_FIELDS names. Declaration schema2/kind
`card-instance-formal-packed-parent-declaration` has exactly DECLARATION_FIELDS
plus `storage_binding,logical_config_sha256`; recipe ID/SHA refer to the new
variant, algorithm SHA remains318eab51. config_sha256 binds full Config;
logical_config_sha256 hashes the checked no-storage literal. Existing
identity_sha/config_inputs/execution/input-binding relations remain exact.
Declaration reference keeps REFERENCE_FIELDS and relative formal-declaration.json,
with the new recipe identity. Stop schema2/kind `card-instance-formal-packed-stop`
keeps STOP_FIELDS, new variant ID/SHA, and unchanged max_updates/admission_deadline/
storage_admission reasons/clock rules. owned_bytes includes the deterministic
external parent root and catalog as specified by Source3A. Existing normal-stop,
complete-parent/three-seed qualification checks remain mandatory.

Native capability adds `card_instance_packed_formal_controls=
'run4-formal-packed-parent-v1;declaration2;stop2'` without changing the old marker.
Require it plus Native3B provider capability before packed formal allocation.
Structural markers and plausible metadata cannot substitute for the actual
parent lock, original controls, Source and all accepted qualification checks.

## Pending final64 physical ownership

Use SAME `FormalGameParent`, `GameFinalValidationContext`,
`_IssuedFinalValidationCollection`, `CollectionPublication` and concrete packed
publication member. Packed context requires genuine complete packed formal
Run4/variant2, not a short/nonformal parent or a directory-parent projection.
Final Config still changes ONLY run.split train→validation, including an exactly
unchanged storage table. Invocation clock/finalization/source budget conditions,
64games/roles/seats/final policy/zero-endpoint inclusion are unchanged.

The DB contains original logical manifest.json/report.json once, seals and never
changes. Composite owner retains collection.pending.json plus a nonpublic
`collection.locator.json.tmp` inode in its logical envelope, whose exact pointer
bytes refer to that pending descriptor. Both are exclusively owned and live;
public collection.locator.json and collection.published.json remain absent.
The actual issued token grants private pending-reader access tied to SAME context
and publication object; public locator readers continue to reject pending.
Logical manifest name stays manifest.json for packed pending/public Sources;
do not invent a per-member manifest.json.tmp inode. A private name accessor
returns that logical name only under the actual issued capability.

Pending proof2 uses its actual temporary locator/pending descriptor/DB custody;
public proof2 later captures public locator/descriptor/DB plus the verified
pending predecessor. Both have the same stable logical source_set from the
unchanged DB and original member names, but distinct physical storage/custody
receipts. The concrete composite owner verifies SAME DB inode/seal/hash and the
deterministic relation: public descriptor bytes equal canonical pending fields
with only publication changed to published; public pointer targets that exact
public descriptor/hash, and the original private pointer targeted the exact
pending descriptor/hash. It never declares distinct inode snapshots equal or
accepts a serialized transition flag as authority. No additional transition
receipt format is introduced.

The private pending locator is an owned temporary, retired after Source/parent
close while the composite still owns publication; public envelope namespace is
only canonical collection.locator.json. Preserve the physical pending descriptor
as the runtime's checked predecessor. Proof2 custody explicitly includes that
optional predecessor as role `predecessor` (narrow Source3A closure extension),
so replacing even that diagnostic inode across reopen is detected. Public and
pending proof sets share logical names/DB identity, not physical snapshots.

Retirement order is the existing outer composition: accept SAME pending Source
→ read its checked facts → `_retire_for_owned_parent` confirms Source/actors/
provider loans/graph retirement while publication/exclusion stay owned → close
actual FormalGameParent and confirm cleanup → `_finish_after_owned_parent`
performs composite publication → retire actual token/reservations/parent borrow.
At every failure, invalidate owned pending/public locator FIRST, then owned
descriptor via held/verified recovery FDs, before further cleanup. Preserve first
error and uncertain owners; no raw-FD-number retry, late healthy return or catalog
promotion. Public locator linkage is the consumer commit point; after successful
outer completion no new fallible/reportable I/O occurs.

Public validation custody locks the actual immutable collection.locator.json
inode. It retains/rechecks actual descriptor/DB originals alongside this lock;
directory mode continues locking manifest.json. Deterministic parent/Source→
catalog ordering and actual-owner identity are unchanged. A physically sealed
DB without the successful logical envelope remains diagnostic. Recovery reports
pending/unpublished evidence, never recreates a lost live owner or publishes it.

## Native final64 and facts

### Final-original parent lifetime closure (architecture READY)

Native3B's actual pool-close hold remains unchanged. A final Source sharing that
pool cannot terminal-close original wrappers before the parent executor joins;
the parent already waits for the Source/slot to retire. Resolve this through
explicit original ownership, never a hold bypass or permissive close flag.

The SAME actual FormalGameParent preowns one concrete final-original slot before
admission/open effects: actual PackedSourceOwner (including locator FD), final-only
provider/index and their reservations. The Source borrows those originals; base
training registry stays immutable. Source.close retires use, episode/packet/DTO
graphs, native final slot and borrow/exclusion in the established order, but does
not close parent-owned wrappers/pool/scan or refund their metadata/working delta.
Source cleanup-complete describes its facade/slot/local use only and prevents
further exports. This predicate is distinct from whole-parent physical cleanup.

After Source/slot use retires, the parent closes its actual native session/worker
with the creator-aware thread gates, confirms join and releases its SAME hold.
Only then does it close its actual final provider/wrapper plus packed scan/pool/
control charges, once per real owner. Keep the parent lock and all uncertain
owners through these steps. Whole parent.cleanup_complete requires those new
physical owners and accounts to retire too, not merely an empty Rust final slot.
Partial final admission belongs to the parent from its first effects; no summary
or serialized transfer marker reconstructs that ownership.

Parent-held final metadata/index/provider/original charges remain counted after
Source close and through native join. Preserve cumulative parent+final metadata
and max(one-group need), retaining the admitted positive final working delta
until parent terminal cleanup as the simple conservative v1 rule. Source-local
graph/packet charges retire only with their actual loans/graphs. Do not double-
charge shared metadata or refund it with the Source facade. Publication/facts
reservations remain owned by their actual context/composite.

The order stays Source/slot/actors → parent native join/hold release → parent
originals/scan/pool/lock → whole-parent cleanup-complete → composite public facts/
pointer commit. On failure, invalidate owned locator first and preserve actual
owners/first error; no healthy publication, new session, manual hold reset or
postcommit reopen. The composite keeps its independent publication capabilities.

Apply the SAME rule to canonical PUBLIC final Sources and Dataset4 readers:
each Source borrows originals preowned by its actual retained parent, Sources
retire first, then all actual parents, then existing outer locator/root custody.
Fitting itself uses the healthy retained Source/parent owners. Successful Dataset/
fit completion and publication require complete checked retirement of every pair;
one failed or partially retired pair cannot refund others' live ownership or
grant new fit permission.

Tests must prove Source use is retired while the actual parent hold/FDs/charges
remain; then actual join → hold release → guarded originals/accounts/parent-lock
retirement; pending and public/Dataset reader variants; partial construction,
escaped loans, worker/join/original/lock failures and multiple pairs. Existing
directory behavior, qualification, no-postcommit-IO and whole joint caps remain.

Extend the SAME four-method native lifecycle with an explicit optional provider
argument to inspect_final_validation; schema1/None preserves old behavior,
schema2 requires the concrete final-only provider borrowed from the actual parent.
Parent base registry stays immutable; a separate final-slot registry contains
only this admitted final root and shares the SAME parent pool/allowance. Retain
it until close_final_validation confirms packet/loan/core retirement. No new
independent native parent/session is opened for final64.

Request2 kind `card-instance-packed-final-validation-request` keys are current
request keys plus `storage_binding,artifact_registry_sha256`. expected_plan keys
remain linkage/config_projection/validation_invocation/schedule/policy_bindings.
Manifest is exactly `{kind:'packed-canonical'}` or
`{kind:'packed-issued-pending',evidence:<proof2 custody>}`. Pending is validated
against the actual private provider's original temporary locator, pending
descriptor, DB and absent public outputs; no old pending manifest inode is used.
Public mode requires published descriptor/locator. Provider registry/describe,
loan, budget and no-reentry contracts are Native3B's exact ones; source membership
and original values remain Rust-validated.

Summary2 kind `card-instance-packed-final-validation-summary` has current
FINAL_SUMMARY_FIELDS plus `storage_binding,source_proof`; source_proof is full
proof2. manifest_kind is packed-canonical or packed-issued-pending. Existing
64ordered episode-row key sets/counts and qualified=False/not_performed are
unchanged. Recheck/close methods keep existing no-new-payload behavior.
Final64 episode packet stays schema1/kind
card-instance-final-validation-episode-payload and exact PACKET_KEYS/typed-bank
layouts, with summary_sha256 now binding actual summary2. Decoder receives
explicit packed context from actual parent/Source, not a freely supplied flag.

Packed Python facts are a new closed object: all existing final64 fact keys plus
schema_version2, kind='card-instance-packed-final-validation-facts',
storage_binding, storage, custody, source_set_sha256. Pending/private and later
public facts identify their actual snapshots. No qualification flag changes.
Native capability is `card_instance_packed_final_validation=
'final64-packed;provider1;request2;summary2;lease1'`; old final64 capability stays
unchanged. Activate only after both reviews/source-matched build and bounded
component success/failure checks. No genuine formal-positive claim is inferred.

### Concrete final-provider and pending-byte seams (architecture READY)

The actual parent packed executor may attach exactly one concrete final-only
OriginalArtifactProvider in addition to its immutable base provider. The parent
preowns the final-original slot first. Attachment requires that exact parent/slot,
SAME pool, idle Future/callback state and creator-aware IO retirement. It creates
no second session, executor, pool or hold; the SAME executor's existing hold owns
the interval. Require that idle/exclusive creator-retired state BEFORE final
original/provider opening, shared-pool retirement/query and allocation effects,
not only at the later attachment. The actual parent owns that admission interval
and prevents overlapping FFI/pool use. No arbitrary provider list or serialized
association is accepted.

Every ordinary/cleanup callback fence, active/uncertain lease check, creator IO
retirement and join gate includes BOTH attached providers. Native final-slot
close retires use but keeps the final provider/metadata parent-owned and attached
until actual parent executor retirement. Account cumulative retained metadata
and only max(0,final working need-base prepaid working need) before final FFI,
using the existing pre-argument admission seam. Keep that delta through parent
terminal cleanup; callbacks never query native budget/session. Partial attach or
slot construction retains actual owners, first error and charges.

Physical PackedSource gains only a private constructor keyword
_allow_pending:bool=False threaded to existing PackedOriginals.allow_pending.
This is a byte/diagnostic seam, never Source eligibility. All public locator-open
entrypoints retain their default pending refusal and do not forward a caller
flag. The concrete pending PackedSourceOwner may enable the seam ONLY after
validating the actual issued token against its SAME context/parent/publication,
temporary locator, pending descriptor and held DB/seal. A bool, path, request
manifest kind or DTO cannot pass those owner checks. Ordinary public Source
construction and strict original custody remain unchanged. Tests separate low-
level diagnostic bytes from rejected public/forged-pending Source access.

### Failed final-slot cleanup lane (architecture READY)

The packed executor permanently refuses normal calls after its first FFI/provider
failure. The final Source must nevertheless retire its native slot before the
parent joins that executor. Reusing normal call() for close_final_validation
creates a cycle: the refusal prevents slot retirement, which prevents whole
parent close. Whole-parent close must not be moved ahead of Source retirement.

Add one private fixed-operation cleanup lane on the SAME packed executor for
the SAME bound native session's close_final_validation only. It accepts no
caller-selected callback, registry, new source, payload or budget. Healthy final
closure first uses the ordinary checked call. The cleanup lane is available
after sticky failure/expiry or failure of that close, on the original caller
thread, with no concurrent normal submission. Any already-active Future is
cancelled/joined through existing ownership before this lane proceeds; uncertain
active work retains the owner and blocks the lane.

Before its native call, permanently fence BOTH actual providers and cancel the
existing Control. Native close must encounter that cancellation before any
original-byte callback or filesystem recheck. The fixed close may retire the
final slot and return its first guard error; a second SAME-session call may
confirm the already-retired slot, matching the existing bounded two-attempt
slot protocol. Escaped packet views or uncertain native retirement remain
failures; an empty Python collection is not retirement evidence. No arbitrary
method retry or raw-FD-number retry is introduced.

Use the actual creator worker for native cleanup and any worker-created SQLite
connection/loan retirement. Caller-created IO retires on the caller. A failed
caller close must retain its actual creator/owner and cannot transfer that IO to
the worker. Track/join the actual cleanup Future using the existing executor;
its completion includes checked IO retirement for both providers. No second
executor/session, join/hold release, parent-original refund, registry mutation,
new reads or re-admission occurs here. Whole-parent close still owns terminal
executor join, hold release and original/account retirement in the existing order.

Keep error identity and terminal progress distinct: actual successful native
close plus checked IO retirement can confirm the Source's slot/local cleanup
even while the original operation remains failed. The facade records that
concrete cleanup before propagating its first error; the executor/parent retain
the same sticky failure and actual owners. No successful final facts, fitting,
publication or resumed work follows this cleanup-only lane. Secondary native,
creator-IO, Future or cancellation errors attach to that first failure and keep
uncertain owners reachable. A cleanup return cannot mark the whole parent healthy.

Verification: reproduce FFI/callback failure followed by final Source close;
confirm fixed native close runs on the SAME worker, callbacks stay fenced and
Control is cancelled before recheck, both providers' actual loans/IO retire, and
Source slot retirement precedes parent join/hold/FD/account retirement. Cover
first close guard error then actual terminal confirmation, escaped packet,
caller-versus-worker close failures, cancelled/failed Future, pre-existing active
work refusal/join, retained first error, and public/pending publication refusal.
Existing directory final lifecycle and healthy packed closure remain unchanged.

### Pool-backed versus external caller growth (architecture READY)

FormalGameParent._sync_native currently combines retained scan growth with
external actor/Source growth into one caller_growth value. Packed shared-controls
credit may exclude ONLY the proven SAME-pool scan component, never that external
component. Split the private caller-growth interface into control_growth plus
keyword external_growth=0; existing one-argument ordinary calls retain their
meaning and old non-shared totals stay the sum. The actual parent passes
scan.retained-native_scan_base and max(0,external-entry_extra) separately.

When actual shared_controls cover the control scan under the existing identity/
charge checks, H(t) excludes that control component already counted by P(t), but
always includes the external component and the separately admitted positive
final working delta. Without shared controls, include both components. Preserve
pre-argument N2 admission and sticky shortage behavior; no arbitrary memory
credit, higher cap, callback native-budget query or new serialized field.
Tests cover independent scan/external growth, exact sufficient/insufficient
joint caps, no double scan charge and no lost external charge across final use.

### Numeric fit transport version closure (architecture READY)

Packed Dataset4 uses request/completion/provenance schema5 only. Numerical
structured-tree-fit-inputs and structured-tree-fit-result envelopes retain their
EXACT existing game schema4/key sets and numerical contents. Input fields are
schema_version,kind,action_config_hash,feature_signature,label_recipe,input_binding,
row_counts,data,source_kind,source_origin. Result fields are schema_version,kind,
request_sha256,input_files,metrics,inference_sha256,model_sha256,source_kind,
source_origin. Their source_kind/source_origin values equal the checked packed
Dataset4 context; storage_binding is NOT a numeric input/result field. Its full
binding is carried by request5/provenance5 and the immutable request/inference
hash relations. Inference remains schema4 with the already accepted closed packed
source subtype below (including storage_binding and provenance5); its numerical
parameter/feature/weight format is unchanged, not its old source-subtype key set.
Inputs4 has no request_sha256: request5.input_files hashes the exact inputs4,
avoiding a cycle. The worker first verifies immutable request5 against its
supplied SHA, then validates inputs4 against that request and its input_files.
Result4.request_sha256 binds the exact request5, and its inference hash binds
the actual inference4 packed subtype. The actual verified Dataset4 context
chooses the packed paired/provenance subtype. Keep separate
paired-control versus numeric/inference schema selectors internally so changing
the pair context cannot accidentally bump every envelope. Old contexts retain
all existing version/key/domain checks and reject new source kinds by default.
No new arbitrary version override, source flag or fit authority is introduced.

## Dataset4 and Tree3 consumers

### Shared storage/custody value checker (architecture READY)

Factor the existing proof2 storage/custody syntax checks into the pure common
`checked_storage_custody(storage,custody)->tuple[Json,Json]` in group_pack_game.py.
Return the same validated value objects without mutation or new authority.
checked_packed_proof and Dataset4 closed row validation call that exact helper;
proof-specific file/member/source-set checks stay in checked_packed_proof.
Preserve all existing fields, allowed roles, ordering, stat6/SHA/fingerprint/type
checks and old error/refusal semantics. Common depends only on pure common
modules, with no I/O/training/model dependency. This introduces no new persisted
format, clone/lifetime owner, qualification flag or permissive extension mapping.
Tests compare valid/invalid storage and custody through proof2 and Dataset4 and
retain proof/member cross-relations plus actual-owner eligibility separately.

Use a new persisted Dataset4 subtype so no Dataset3 reader silently accepts
changed source provenance. Numerical feature projection3, selected-label recipe,
seed slots17/29/43, split/population rules and original own-view DTOs are unchanged.
`schema_version=4`, kind stays card-instance-game-fixed-sources, source_kind is
`card-instance-formal-ppo-packed-originals`, and source_origin differs only in
qualification_recipe='three-formal-packed-parents-final64-v1'.
Top fields are existing FIELDS plus storage_binding. Parent rows are existing
PARENT_FIELDS plus storage_binding. Collection rows are existing COLLECTION_FIELDS
plus storage and custody; each is bound to actual proof2/public final facts.
Generation/role/count/index column schemas stay unchanged. Training correspondence
uses Python3A's packed recipe. Final correspondence uses
card-instance-packed-final-validation-correspondence-v1 with existing exact
correspondence fields; its manifest and ordered64episode hashes bind the SAME
public Source. Dataset source/content digests include all new provenance fields.

Add explicit `VerifiedGameTreeDataset4` (private shared factoring with Dataset3
allowed) and schema-directed build/inspect/fit dispatch. Dataset3 old class,
source kind/origin/field sets and old fixtures remain exact. Dataset4 requires all
three complete packed formal parents and all three prescribed PUBLIC packed
final64 Sources, with full Config/runtime/hardware/code/semantic homogeneity and
only the already approved per-parent exclusions. Storage tables/bindings must be
identical; do not drop them from homogeneity. Mixed storage modes reject in v1.
No conversion of old Dataset3, short parents or a fixture owner grants Dataset4.

Use actual retained parents/final Sources and their scoped group/episode loans;
joint E/S0/F/A/B/L admission includes all retained pools, indexes, native copies,
feature rows and fit-child overlap. Source owners remain through fitting and
checked final closure, exactly as existing Dataset3. Native RAW is the reviewed
own-episode provider path, never an extraction/reinterpretation adapter.
Fresh Tree fitting uses unchanged label/feature/Booster algorithms and numerical
Tree3/inference4 artifact formats. Paired request/completion/provenance gets an
explicit packed Dataset4 source subtype (Dataset4 content/source/storage hashes)
while old Dataset3 provenance remains exact; pair validation must not accept a
new source kind by ignoring unknown fields. Tree3 inference can opt into packed
collection after these consumers/guards are connected; policy still sees only
its own existing protocol3 input, not storage/provenance controls.

## Verification and limits

### Final contract closure

The successful CLI returns PUBLIC packed facts, never the earlier temporary
snapshot. Context captures pending semantic facts under its existing admission
before Source retirement. The composite owner receives that frozen summary,
keeps DB/proof hashes bound, and constructs the public storage/custody/facts
object while its actual new pointer/descriptor and retained originals are still
owned, after their final links/private-temp retirement/sync and stat/hash checks.
For D:9p's terminal-control handoff, prepare from the owned readonly verification
snapshot after controlled write-handle retirement, as the runtime amendment
specifies; never serialize the earlier transient ctime as public custody.
It rechecks the deterministic descriptor/pointer relation and unchanged DB /
logical source_set, pre-admits/clones the public facts, then performs checked
terminal FD retirement and final pathname checks before setting committed.
Context keeps the already constructed return object; `_finish_after_owned_parent`
returns/transfers it only after publication cleanup succeeds. No public Source
reopen, extra JSON parse or new reportable I/O follows successful release.
Private pending snapshots may remain diagnostic values but cannot be returned
as the public custody or treated as new Source permission. A late close failure
invalidates and fails rather than returning the prepared public facts.

Existing native `inspect_final_validation(request,retained_python_bytes=0)`
calls remain valid. New signature is
`inspect_final_validation(request,retained_python_bytes=0,artifact_provider=None)` with the
provider optional in that THIRD position; ordinary request1 requires None,
packed request2 requires the actual final-slot adapter. Recheck/export/close
signatures and packet leases are unchanged. Parent base registry is never
mutated to smuggle in a final root. Final-slot provider metadata/allowance is
explicitly attached/retired under the same session and no-session-reentry rule.

Control admission has two phases. Before collection, preserve `_control_bound`'s
eight32x+64KiB clone/project/schedule envelope and include the complete storage
table/binding/variant/paths via allocation-free wire-size arithmetic. After
physical sealing, before original-source/summary/facts/provider JSON allocation,
use retained descriptor length and SQL `index_admission` lengths to reserve
packed index/storage/custody/proof/registry bytes. No all-row fetch or JSON
serialization precedes this admission. Let W be an allocation-free wire upper
bound for the actual source index plus proof/storage/custody/registry fields;
add `8*(32*W+65536)` for overlapping Python control/encode/facts copies and the
existing parent/control reservations. Native final summary admission includes
the ACTUAL full proof2/registry/storage-binding bound in addition to current
manifest+episodes+64KiB; admit raw JSON and its ordinary native parse/copy leases
before constructing the summary. Do not retain the old schema1-only bound.

The parent and final adapter share one provider pool/group allowance: reserve
the maximum of their admitted group working needs, not a second full per-source
budget. If final need exceeds the parent's prepaid one-group allowance, reserve
only that positive difference before loading/copying. Retain both parent and
final metadata charges cumulatively. External held bytes passed to every FFI
call include all caller/index/provider/temporary-summary/escaped-packet costs;
packet decoder retains existing64x+32MiB transport overlap. Python/Rust source,
metadata, Stage, numeric arrays and actor/fit budget owners must all see their
same existing joint cap. Refuse before allocation on shortage; neither raise
caps nor omit parent metadata to make a fixture succeed.

Tree pairing uses a concrete singleton `TREE3_PACKED_GAME_RECORDS` and concrete
`PackedGameTrainedProvenanceV5`. It shares current Tree3/game numerical algorithms
and adds only the explicit source-storage context. `_records` accepts this exact
singleton alongside existing ones; old defaults remain exact. Request schema5
has kind structured-tree-fit-request, exact REQUEST_FIELDS plus
`source_kind,source_origin,storage_binding,shared_generation,managed_memory`.
Completion schema5/kind structured-tree-training has exact COMPLETION_FIELDS plus
`source_kind,source_origin,storage_binding`. Trained provenance schema5/kind
structured-fixed-dataset4-fit has exact PROVENANCE_FIELDS plus
`source_kind,source_origin,storage_binding`. All source values equal Dataset4's
checked source kind/origin/binding. Other field/domain/envelope/health/status
checks are existing Tree3/game checks; no new arbitrary extension dict.

Request dataset_manifest_sha256 binds exact immutable Dataset4 file bytes;
dataset_content_sha256 binds its canonical object excluding content_sha256;
source_set_sha256 includes all Dataset4 original/storage/custody provenance.
The immutable request/completion file hashes remain SHA256 of their existing
canonical-JSON-plus-newline bytes. Completion request_sha256 and all current
paired identity fields plus source/storage keys must equal the immutable request;
inference provenance must equal the schema5 provenance derived from that SAME
request/hash. `TrainingPair` directory/request/completion hash fields are unchanged.
No serialized source_origin/storage field alone authorizes fitting.

Numerical Tree3/inference4 metadata remains inference schema4; the new closed
source subtype adds storage_binding alongside existing source_kind/source_origin
and schema5 trained provenance. Its model/feature/label/creation/descriptor/weights
formats stay unchanged. Artifact/pair/participant/worker/RAW dispatch chooses the
exact packed record context only for that verified subtype; old source kinds
continue to use existing context/schema4 fit records and reject added fields.
`_PreparedTreeFit` accepts EXACT VerifiedGameTreeDataset4 for packed context and
EXACT VerifiedGameTreeDataset3 for old game context. VerifiedFixtureTreeDataset3
remains its distinct unqualified context. Row builders/sink reservations/Booster
work are shared; no widening from a metadata flag or arbitrary subclass.

Tests first: exact recipe hash/base/storage-only relation, unchanged literal
budgets, old Config/declaration/stop/Dataset3 proofs and early unsupported refusals;
pending no-publication and every Source/parent/FD/provider close failure boundary;
native request2/summary2/packet parity and escaped loans; public Source custody
replacement checks; Dataset4 homogeneity/qualification/split/provenance refusal;
same-original feature/label/numeric/Tree score and bounded fresh fixture fitting.
Exercise existing controlled fixtures with honest fixture provenance, never
fabricated genuine formal execution/attestation. There is currently no complete
eligible real formal parent/final64; genuine positive formal handoff/real-source
fit remains NOT RUN unless already available without a forbidden campaign.
All reachable engineering paths, refusals and bounded smokes must be checked;
document that limitation separately rather than retrofitting short parents.
Whole verification and alternating same-real-data/HDD plus simultaneous workload
comparison remain required before complete-plan acceptance or any default choice.

### Formal-completion namespace handoff (architecture READY)

FormalCompletion.publish also retains its O_RDWR temporary FD across mutable
replace. The actualHOST9p open-FD visibility evidence applies to this operation;
closing that FD without preserving completion ownership would break revocation
across the outer run-lock exit. This amendment changes only controlled terminal
publication, not formal eligibility, schemas, canonical bytes, algorithms or
original deadlines/budgets. Apply the same implementation to old Run3 and explicit
packed Run4; no filesystem detection or permissive Source mode.

Preown a distinct writer descriptor stage before its exclusive temporary open,
in addition to the existing parent directory and retained completion stage.
Write/fsync and verify exact canonical bytes, regular mode, held/path six-field
identity and namespace under the existing bound. Capture this actual complete
original's dev/inode/mode/length/mtime and full hash. Confirm one terminal writer
close BEFORE the one replace; uncertainty retains the actual consumed stage and
first error, forbids publication and never retries a raw number. Keep the
original expected identity separate from any subsequent opened capability.

The directory capability remains held. Set exposed before replace as today,
since rename can take effect before reporting failure. No FD to the temporary
original may remain open across rename. After namespace publication, open a NEW
O_RDWR|O_NONBLOCK|O_NOFOLLOW retained completion capability through the SAME held
directory. Bind it to the ALREADY controlled original by dev/inode/mode/length/
mtime and full canonical hash. Allow ctime handoff only for that confirmed owned
writer close and namespace operation; then retain strict checks. Never replace
expected original identity with facts from an unvalidated opened file. An open,
lookup or verification error is failed publication; actual capabilities remain
owned and existing known-inode revocation rules apply. A foreign opened inode
must never be written, and no polling/rename retry or reconstructed Source right
is introduced.

While the actual directory capability remains active, failed-publication recovery
is owned by this SAME FormalCompletion and checks exactly two fixed names through
that held directory: run.json and run.json.formal-complete.tmp. This handles both
replace-before-effect failure and replace-after-effect/fresh-open failure,
including substitution of the absolute parent path. Do not require the replaced
absolute namespace to be healthy before cleanup through the original directory.
Preown bounded recovery descriptor stages before open/stat effects; compare
the independently preserved expected original dev/inode before ANY write. A
foreign candidate is never invalidated or promoted to original ownership; retain
its actual opened identity separately for checked one-attempt closure. Absent or
foreign canonical candidate may lead only to the other fixed candidate, not a
search or repeated opens. A validated actual original capability uses the
UNCHANGED known-inode invalidator. Keep every uncertain opened/closed stage and
first error reachable; never overwrite a live/uncertain field with another FD.
After actual directory-capability retirement, preserve the existing strict
canonical-path known-inode recovery. Failure to recover/invalidate remains
unconfirmed cleanup and cannot set revoked/committed/healthy optimistically.

The fresh retained O_RDWR capability is used only for verification and, upon
failure, actual original invalidation. It is not written on the healthy path.
Its actual one-attempt close and the existing post-close final guard remain
strict; do not silently allow a second ctime change at that close. Keep the
existing five-field public snapshot representation, adding only private mode/
hash facts needed to check the controlled handoff. Whole FormalCompletion.finish
still follows outer run-lock retirement, wall/disk guards and actual descriptor
retirement before committed=True. Failure before/after rename, fresh open,
outer-lock retirement, final guard or close revokes the actual known original
and keeps first error/uncertainty reachable; no healthy completion from inner
publication alone. Existing fresh known-inode recovery after close is preserved,
not widened to an arbitrary current pathname or a foreign replacement.

Charge new descriptor/bookkeeping/hash buffers before effects inside the existing
formal admission reservation. Constructor/partial stages are owned from first
effect; release other certain stages while retaining uncertain stages. Do not
reset the writer's consumed close state to install the reader/revoker. No new
session, lock, caller capability, persisted transfer token, dependency or generic
publication framework is required.

Verification: controlled unqualified FormalCompletion component fixtures for old
and packed control variants; writer close before replace, strict retained-original
handoff and actual revocation across outer-lock failure; fault before/after each
close/rename/open/hash and parent/target/temp replacement; expected identity must
survive a foreign fresh-open result; no stale-number retry/snapshot overwrite;
first error and storage/deadline limits. Existing formal/control/terminal tests
and8frozen algorithm ASTs remain. After reviews, bounded control-only HOST D:/tmp
publish/finish/revoke checks must confirm the newly opened non-writing O_RDWR
capability closes without identity drift. If that check fails, report evidence
and return to design review rather than permitting further ctime drift. No
1000-update campaign, genuine formal parent or successful Source handoff is
constructed to exercise this component.

### Tree fit terminal ownership closure (architecture READY)

Two concrete gaps exist in the shared actual fit workflow. Its mutable
training.json completion keeps _publish's writable original across replace,
matching the DrvFS visibility failure. It also sets fit.closed after completion
commit and then calls fallible modern Dataset.release; that failure can leave a
healthy completion and be retried/masked in the exception path. This is within
current Tree2/fixture3/game3/packed4 responsibilities, not a new fitting gate.
No real-source qualification, numerical input/result/inference schema, model
parameters, rounds, provenance fields, budget or dependency changes.

Keep _CompletionPublication.commit's existing signature and terminal contract.
For its mutable completion only, replace the hidden returned-FD handoff with
explicit preowned writer, retained completion and fixed-candidate recovery
stages. Confirm canonical writer bytes/hash/stat/namespace/fsync and one-attempt
writer close before the ONE replace. Reacquire a NEW retained non-writing O_RDWR
capability through the SAME held output directory, bound to the independently
preserved controlled original dev/inode/mode/length/mtime/hash. Only that owned
writer-close/rename permits ctime handoff, followed by strict6 checks. Preserve
the publisher's existing5tuple representation, all output-only pair postchecks,
strict terminal close/guards and known-inode invalidation. Do not change generic
inference-artifact _publish or the separate immutable-copy completion.

After every actual terminal close succeeds and BEFORE committed=True, compare
the published path's full six-field identity with the verified retained snapshot
and recheck the owned output directory/ancestor namespace. There is no second
ctime handoff at the retained non-writing O_RDWR close. A late lookup/identity/
namespace/deadline failure uses existing known-inode recovery and stays failed;
do not set committed merely because the raw closes returned. Keep the original
cleanup deadline and output-only boundary for this post-close check.

Use the SAME recovery contract as FormalCompletion: while actual directory FD
remains active, only fixed training.json and training.json.complete.tmp candidates
through that directory, independently expected original inode before any write,
foreign candidate preserved, actual opened identity separately retained, bounded
preowned recovery stages and unchanged known-inode invalidator. After directory
retirement, existing canonical recovery remains. Partial close/open/replace/
postcheck/expiry errors retain first error and actual capabilities. No raw-number
retry, search, visibility polling, optimistic committed flag or foreign writes.

Modern Dataset.release MUST complete BEFORE any healthy completion is exposed.
After the existing Source/parent/row/child reportable close, copy only the checked
shared-cost scalar/JSON needed for the final record into fit/publisher-owned
bounded metadata, charged before allocation. Drop any remaining borrowed Dataset
metadata references before its memory release. Do not keep/reopen original
Sources to service output-pair postchecks. Completion uses the already verified
request, child/model outputs and that bounded cost copy; no mutation to its
persisted contents or schema is needed. Only then build/validate and commit the
healthy completion and set fit.closed. No fallible Dataset/Source work follows
that terminal commit. Legacy non-modern behavior stays compatible.

Treat Dataset release as an actual one-use stage on the fit owner. A failure,
including after effects, retains the actual Dataset/prepared/control owners and
first error, blocks healthy publication and cannot be retried automatically or
inferred complete from empty maps/closed flags. Context-exit/cleanup failures
must not mask the initiating fit error. Failed or uncertain release keeps
fit.closed false; physical Source/child retirement and memory-release completion
are distinct facts. Remaining cleanup may retire other certain actual owners,
but does not clear uncertainty or grant new fitting/publication permission.
Existing close/failure record paths must honor this order and preserve the
original failure rather than skipping committed publication or invoking release
again. There is no new public callback or serialized handoff authority.

Both actual modern Dataset.__exit__ implementations already invoke release.
Normal direct release and error-path context exit are mutually accounted routes
through the SAME fit-owned one-use stage, marked BEFORE either call. Confirm
only an actual successful route with the contractual closed/inactive state;
an exit error retains uncertainty even if closed/active flags or maps changed.
It suppresses any subsequent direct release or repeated context exit. Other
certain physical cleanup can proceed separately without retrying that stage.
Do not add a new public Dataset API or infer memory retirement from empty maps.

The legacy non-modern Dataset has no release/_closed API: its normal/error
__exit__ uses that SAME fit-owned one-use stage too. Retain its actual Source
before exit if needed; confirmation requires actual successful return, inactive
Dataset and that actual Source's closed predicate. Raised exits remain uncertain
even if those flags changed, and cannot be repeated by the handler. Preserve
legacy serialized records and Dataset API without adding a public release method.

Verification: exact-context old/new completion records and fitted output bytes;
controlled writer close before rename, strict reacquisition, held-dir recovery
before/after rename and under parent/target/temp substitution, foreign inode
refusal, actual close failures/expiry/budget and retained first error. Instrument
an actual unqualified fixture fit's release before/after effects: no complete
training.json can survive a failure, no second release, actual child/Source
owners remain inspectable, no false fit.closed. Healthy fixture500-round fit,
paired copy/load/full-Booster/score regression, all formal ASTs and static checks
remain; genuinely qualified Dataset4 fitting stays NOT RUN. After both reviews,
bounded control-only HOST D:/tmp completion success/revocation checks are required.

### Schema selection preflight closure (architecture READY)

Dataset schema selection is routing only and must preserve the concrete reader's
allocation/preflight boundary. The new cross-module selector is
verified_game_dataset_type(path, expected_digest, settings, *, max_owned_bytes,
deadline), where settings is the SAME caller PreparationConfig, not default or
reconstructed permissions. It returns only the closed Dataset3/4 class and grants
no Source, parent, fitting or ownership right. Existing public entrypoints and
Config serialization stay unchanged.

Run the existing closed _settings validation and _reader_preflight BEFORE any
preliminary manifest read. Apply the same existing regular-file and
min(64,settings.max_source_file_mib)MiB cap before selector bytes/JSON allocation,
with SealedScan's actual owned-file/hash/deadline/budget checks. The concrete
reader still repeats its complete original/ownership validation; no temporary
selector result or JSON is promoted into original custody or used as a default
on failure. Whole-input resident scans and unknown-schema fallback remain refused.

Keep this preliminary selector LIMITED to reading the integer schema_version
and choosing the exact closed class. Do not call game.manifest/packed_manifest,
canonicalize or clone the full manifest here. Those validators allocate a clone
and belong to the concrete reader's existing separately reserved validation
pass (additional32*stat_size+65536 before object/validator work). Selection thus
uses only its own bounded SealedScan parse reservation; it creates no second
unreserved full-validator allowance. Full body validation remains mandatory in
the actual concrete reader before any Source/fitting permission.

Dataset build likewise validates SAME settings/native preflight before its
private three-parent subtype selection. That pass only distinguishes homogeneous
original Run3 versus explicit packed Run4 declarations; it is not parent
qualification and the actual owners still perform all checks. Keep existing
reader versions, byte/metadata bounds and early failure behavior. Negative tests
must prove invalid settings/backend/native marker and over-cap manifest fail
before the preliminary read, not just before later row allocation. Root treats
the already isolated correction as unaccepted evidence until this narrow design
review; dependent adoption waits READY and final implementation conformance.

Establish ONE original effective absolute preprocessing deadline at caller entry,
before selection/preflight work: validate SAME closed settings with the existing
validator, then cap by caller deadline, original training deadline and original
entry_monotonic + settings.wall_seconds - settings.cleanup_seconds. Pass that
SAME absolute deadline to both selector and concrete reader, so the latter's
existing min() cannot restart preprocessing time after routing. A local reuse
of the existing closed validator is sufficient; do not add a deadline-bearing
selector result, new Source token or budget extension. The build selector also
consumes its already established original absolute budget. Invalid/expired
settings/deadlines fail before metadata reads; tests must advance a fake clock
during routing and prove no restarted allowance at reader construction.
