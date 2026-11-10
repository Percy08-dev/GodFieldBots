# Group Pack: original Source, lineage and native byte provider

Date: 2026-11-10. Owner: root. Status: Python Source/lineage/PPO and Native3B
designs accepted after independent architecture reviews;
formal/final64/Dataset design is separately accepted in
[final sources](sqlite-group-pack-final-sources.md). Implementation and final
conformance remain pending.
References: ADR0034, foundation/runtime designs, ADR0024 §§16–18.2.
This design completes the intended consumer scope; no temporary extraction is
used as original custody and no old identity is reinterpreted.

## Original identity and proof

Logical root stays the existing collection envelope path (e.g.
`parent/collections/batch-000000`), not a physical expansion directory. The
new locator binds that root to an immutable external DB/descriptor. Retain and
recheck all three actual original files (locator, descriptor, DB) and their
namespace/stat identities throughout Source/update/parent/Dataset use.

Packed proof has `schema_version=2`,
`kind='card-instance-packed-original-source-proof'`, all existing corrected
proof fields (classification, qualified=False, parent_qualification, counts,
manifest_sha256, source_set_sha256, files, closure_counts, missing_replays), plus
`storage`. `storage` has exactly `format='sqlite-group-pack-v1'`,
`locator_sha256`, `descriptor_sha256`, `database_sha256`, `seal_id`,
`logical_sha256`. These identify actual admitted bytes, not catalog state.
Source proof1 and its physical file identities remain exact for directory data.

Proof2 also has `custody`, distinct from stable logical member identity. It is
closed `{recipe:'packed-original-custody-v1',files,directories,fingerprint}`.
Files are path-sorted `{role,path,sha256,identity}` for exact locator, descriptor
and DB originals plus the verified optional pending-descriptor predecessor;
roles are `locator`, `descriptor`, `database`, `predecessor`, paths absolute,
identity exactly `[dev,ino,mode,size,mtime_ns,ctime_ns]`. Directories are
path-sorted `{path,device,inode}` for logical envelope and physical DB directory.
Fingerprint hashes that object without `fingerprint`. Reopening compares saved
custody as well as logical proof; a byte-identical DB/descriptor/locator inode
replacement across pause/resume rejects. Logical source_set stays independent of
access/caching, while persisted physical-custody identity preserves the old
cross-open replacement guarantee. No automatic relocation/migration is enabled.

Packed `files[name]` is `{sha256,bytes,identity}`. Identity is a closed object
`{format:'packed-member-v1',database_sha256,group_id,pack_sha256,name,offset,
length,sha256}`; group/pack/offset are null for DB controls, length equals bytes.
`source_set_sha256=digest(files)` uses canonical JSON and stable original logical
names/ranges/content, not temporary inode or access path. The physical custody
owner separately holds stat snapshots/open descriptors; never serialize those
as a fake member inode. Source proof includes both participants/diagnostics in
the existing closed inventory; policy/learner inputs remain own-view only.

Original manifest/report, all journals/columns/results/Replay remain exact DB
member/control bytes. Original Collection5's complete inventory still closes
over those logical names, excluding locator/descriptor/DB physical namespace.
Extra/missing members, partial journals, inconsistent group/episode index or
different hashes reject under the existing eligibility rules. Physical metadata
cannot turn incomplete/all-natural-but-unhealthy into eligible input.

## Python owner and shared pool

Add concrete `PackedSourceOwner` and an explicit game-only factory alongside
`_SourceOwner`; no modification of old filesystem identity semantics.
The factory accepts only the configured storage binding and correct locator,
never automatically guesses a packed reader for old Config/data. Its logical
`read`, inventory, contains, evidence and finalize methods operate on verified
DB members and preserve full-name scoping. `_inspect_component_collection`
chooses the byte-journal seam only for this concrete owner and requires logical
sidecars. Directory owner rejects logical_sidecars; packed owner rejects physical
sidecars. `remember` never converts tagged identity objects to stat tuples.

Extend snapshot/owner lifecycle with explicit `close`: directory owner has no
new retained resources; packed owner holds actual descriptor/DB custody until
final recheck, graph/loan retirement and checked close. Uncertain close poisons
Source and retains its cleanup owner; a second empty close is not success.
Reuse the existing SAME `_OnlineGameOwner`, source-issued projection and
learner process/parent commit checks. An ArtifactSource or proof value alone
cannot construct `IssuedGameProjection` or grant update/fit authority.

`PackedArtifactPool` is one concrete retained pool per parent/Source session,
shared across its collections. It holds one SQLite read connection and one
decompressed group at a time; closed/inactive collections retain physical
original FDs and minimal verified indexes, not per-collection SQLite caches or
raw parent data. Switch connection/group only after scoped loans/export copies
retire; reopen through the SAME held original and recheck its public path.
Recheck full DB/descriptor/locator bytes from SAME opened FDs without SQLite
connections or group decompression. Additional parent/collection metadata is
admitted before retention. Source-owned pool shares its budget with metadata,
decoded records and exports under min(storage budget, existing Source cap).

The foundation ArtifactSource gains only a concrete shared-pool composition
option for this consumer path; public one-source convenience behavior stays
valid. A detached `read()` copy remains conservatively charged; bulk original
readers use scoped group loans and transfer admission for decoded/exported data.
History/Dataset borrowing reloads only the requested group/episode, validates
against the SAME retained original proof, and retires before another group.
Whole-collection decoded PPO batches may retain their already bounded selected
16-game inputs; do not accumulate every parent batch's decoded records.

## Parent control and generation compatibility

Packed PPO's external persistent root is deterministically
`storage.db_root/<parent_run_id>/collections/<batch_id>/` (collection's own
UUID4 remains its DB run_id). This replaces the standalone `<run_id>/` allocation
for PPO only, so active/uncommitted/failed external bytes remain discoverable by
the SAME parent even before an envelope pointer exists. No external reservation
registry or orphan guessing is needed. A resumed parent derives exactly the same
root from its frozen Config and run_id; pointer targets must match the prescribed
batch root. Standalone/final64 collection UUID roots retain runtime design's
location and their invocation-owned accounting.

Extend `owned_bytes(directory,deadline,*,external_roots=())` with the default
empty tuple preserving old behavior. For packed `_GameOperations`, the common
coordinator's SAME admission/after-collection/after-update/finalization checks and
final artifact_bytes measurement count the parent plus that run-owned external
root, including live DB/WAL/SHM/journals, descriptors, pending/failed evidence and
owned temporaries. Deduplicate actual `(st_dev,st_ino)` files across roots and
refuse symlinks/foreign roots. Count the shared catalog and its live sidecars once
conservatively for the current run; never use catalog rows for ownership or
eligibility. The existing disk cap/admission reservation is not replaced by the
separate storage-memory budget. Active collector exposes these exact owned roots
before Writer allocation, and cancellation retains them in its cleanup owner.
Include external bytes in native scratch allowance and final64/Tree invocation
disk accounting; do not let envelope pointers hide persistent storage.

`SealedScan.files` remains actual physical files with actual stat tuples. Add
separate `logical_files` and retained packed owners to the training `_ChainScan`
and corrected control-read adapters. A common `verified_digest(path)` accessor
checks physical or explicitly registered logical evidence. Never put a logical
member in the existing physical `files` map or return fabricated os.stat_result.
`_ChainScan.object` can route collection controls through its registered
PackedSourceOwner; ordinary run/generation/numeric/artifact files use SealedScan
as today. `gf_models` does not import training/storage/CLI. Replace corrected
consumers' manifest `.files[path]` lookups with the explicit accessor; physical
post-operation sweeps stay physical, logical owners get separate final rechecks.

Packed run state uses `schema_version=4`, with `storage_binding` exactly
`{format:'sqlite-group-pack-v1',pack_format:'group-pack-v1',database_schema:1,
member_identity:'packed-member-v1',provider:'python-original-bytes-v1'}`.
Its effective Config must contain the explicit storage table and its Config SHA
and resume_identity include that table/binding. Directory run3 remains exact.
Packed update records use schema4 and proof2; old update3/proof1 remain exact.
Generation checkpoint/artifact/numeric schema numbers retain their current
meanings, with their parent/control hashes binding the packed run/update variant.
Mixed storage generations in a parent reject; storage settings cannot change on
resume through CLI override. New readers validate explicit subtype closure;
old readers naturally reject unknown version/fields instead of guessing.

Packed source correspondence uses recipe
`card-instance-packed-source-correspondence-v1` with the existing closed fields,
binding hash and ordered own-episode correspondence. Source fingerprint includes
run Config/identity/execution/plan plus proof2's storage and source_set hashes.
Source-issued projection recipe is `card-instance-packed-source-update-v1`.
These new tags are accepted only with the packed state/proof; legacy recipes
cannot refer to packed originals. Correspondence remains an ephemeral validated
binding as today, passed from SAME Source to numeric reader/worker; it is not
newly persisted inside Numeric4 or granted authority. Numeric4 retains its
original contents/own-view numerical inputs and reward semantics. Proof2 and
update4 persist the separate original custody/provenance.

PPO callback result gains an explicit `PackedFinalizedCollection` subtype with
locator, while directory FinalizedCollection serialization remains untouched.
The common coordinator still requires requested output, healthy report, original
control digest, actual same parent/ledger and Source issuer. `_GameOperations`
routes read_control, project, parent validation and commit by explicit variant,
not reader backend fallback. Pause/resume reopens physical originals and rebuilds
the same stable proof2; it never recreates a directory proof1 from staged files.

## Python increment contract closure

Packed run4 required keys are exactly existing `RUN_REQUIRED | GAME_EXTRA |
{'storage_binding'}`, with existing `RUN_OPTIONAL`; formal additions remain
behind the later formal/final64 gate. Packed update4 keys are exactly existing
`UPDATE_FIELDS | GAME_EXTRA | {'storage_binding'}`. Binding must equal the run's
checked binding. Proof2 keys are exactly existing game proof1 keys plus `storage`
and `custody`, changing only schema/kind/identity representation as specified.
Generation3, registry3 and artifact4 fields stay exact. Corrected classification
dispatch recognizes explicit run3 or run4, never infers model semantics from a
truthy/high schema integer. Run4 requires corrected actions3 and explicit storage;
run3 refuses packed storage/binding/proof. No mixed proof/update variant.

Resume digest is canonical digest of `{recipe:
'card-instance-packed-ppo-resume-identity-v1',config_sha256,identity,execution,
input_binding:<existing corrected classification>,storage_binding}`. Directory
resume recipe/object remains unchanged. Packed correspondence control fingerprint
is digest of `{recipe:'card-instance-packed-source-controls-v1',run_id,
config_sha256,identity,execution,plan:<frozen linkage>,files:<same original
generation JSON and inference.json relative-path→SHA map>,source_set_sha256,
storage:<proof2 storage>,custody:<proof2 custody>}`. Preserve the existing
immutable generation/inference hashes, not just live Config. Correspondence has
exact existing keys `recipe,control_fingerprint,semantics_sha256,episode_count,
episode_digests,binding_sha256`; only recipe changes. Its binding digest excludes
binding_sha256, as today. Projection has existing exact PROJECTION_KEYS, with
new recipe as above. `PackedFinalizedCollection` fields are exactly inherited
`directory,report` plus `locator:CollectionLocator,envelope_sha256:str,
storage_metrics:Json`; return
type/serialized flags do not mint Source authority.

Use learner wire5 for packed Source updates. `GameLearningProcess` gains an
explicit optional `storage_binding` parameter; absent keeps wire4/identity exact.
Wire5 identity has all wire4 fields plus exact checked `storage_binding`; command/
reply key sets are wire4's sets, with `protocol_version=5`. Worker reconstructs
the exact identity and requires both packed projection/correspondence recipes;
wire4 only accepts old recipes. Packed transaction digest domain is
`card-instance-packed-update-transaction-v1` over projection; wire4's domain is
unchanged. No policy input gains storage/management data: this is the numerical
delegate, whose learner role already receives Source-issued projection. Model,
optimizer, RNG, checkpoint schema and numerical behavior remain unchanged.
Expected packed correspondence is passed only through an explicitly selected
numeric-reader context; old default numeric readers reject the new recipe.

One actual `PackedBatchCustody` per online batch is held by `_OnlineGameOwner`.
Callback control scan, projection's controls/snapshot and commit scan borrow
that SAME owner/pool rather than opening independently owned copies. Parent
historical scans own a separate session pool unless explicitly borrowing the
actual retained parent's pool. Every one-shot scan/parent validator/binding
helper uses a context/finally that final-rechecks then checks close; retained
FormalGameParent/Dataset owners close their pools at existing explicit owner
retirement. A scan borrowed from the online batch releases only its borrow.
No destructor, implicit refcount drop or successfully cancelled coroutine proves
FD/SQLite retirement. `.close()` errors preserve primary failure and actual owner.

`IssuedGameProjection.prepare_commit` is the last reportable per-batch original
I/O boundary: recheck Source/control/custody, complete native precommit if used,
confirm all group/export/control loans retired, freeze canonical proof2 in
memory, close actual batch original FDs/SQLite/pool and all borrowing scans,
then set prepared. Commit's earlier `_update` scan borrows the same batch owner
and detaches its logical-owner rechecks at that retirement point; its ordinary
physical lineage/numeric checks remain unchanged. Per-batch Source/scan finally
after prepared does only confirmed in-memory retirement, never new hashes or
fallible closes. If close cannot be confirmed, prepare_commit fails before any
update/generation/latest/run publication and retains SAME cleanup owner.
Failure before prepare likewise final-rechecks/closes before ownership release.
The prepared projection exposes its frozen proof/expectation for commit without
reopening closed originals; future parent validation/resume reopens and compares
that persisted custody against actual originals. Keep long-lived parent locks
through commit as today. This does not relax the no-reportable-I/O-after-commit
rule or permit multiple updates from one issued projection.

## Native provider (separate dependent increment)

Rust gets no SQLite/zstd dependency. Add a generic Rust `LogicalArtifactProvider`
trait used by corrected `structured_session::Sources`; only the PyO3 bridge
adapts Python callbacks. Legacy physical Sources and plan3 ABI remain unchanged.
The explicit plan4 variant carries the packed storage_binding and registered
logical collection roots; session construction requires the matching byte
provider argument. Missing/mismatched provider refuses before Source inspection.
No automatic fallback to Python numerical processing or physical extraction.

Python `OriginalArtifactProvider` is owned by the SAME `_NativeGameInputs` /
retained parent, shares its PackedArtifactPool, and exposes only registered
root/member reads, logical indexes and rechecks. Rust receives original logical
names/proof2 identities separately from access; no temp paths enter proof or
fingerprint. A provider can supply diagnostic bytes but grants no Python Source,
formal parent, update, final validation or fitting rights.

Concrete callbacks (versioned adapter, fixed methods): `describe(root)->bytes`
returns canonical UTF-8 closed `{format:'python-original-bytes-v1',root,storage,
custody,files}` with proof2 storage/custody and name-sorted
`{name,bytes,sha256,identity}` rows; `borrow(root,name)->OriginalByteLease` returns
only that verified original member under a live generation-scoped loan;
`recheck(root)->None` checks actual originals. Unknown root/name rejects.
PyO3 adapter invokes callbacks with the GIL attached while the long FFI work
otherwise releases it, validates exact bytes/JSON type/limits, pre-admits the
Rust destination and copies directly from the loan's PyBytes once. Avoid an unbudgeted
intermediate Vec. Provider does not call back into its native session or acquire
parent/catalog locks, preventing GIL/session/budget reentry cycles. Python
records callback failure; native failure remains sticky and cleanup-only.

Pre-reserve provider metadata and the maximum one-group working allowance in
the existing native external-held account for each FFI operation. Provider
callbacks work inside that admitted allowance; Rust parses/stages/exports under
its ordinary Budget. Only one group across all registered sources is resident.
No callback attempts to query/lock session state during a native call. Published
arrays/packets keep their existing Rust base-owner leases; provider group
loan/copy reservations retire only after the actual Rust copy completes.

### Native increment closure (design accepted)

Plan4 fields are exactly plan3's `schema_version,mode,parents,collections` and
live-only `cohort_collection_index`, plus `storage_binding` and
`artifact_registry_sha256`. Modes/cardinality/parent/collection row shapes remain
plan3's; schema_version is4 and packed run4/proof2 is required throughout.
Plan JSON retains the existing64KiB cap; do not embed all raw/member indexes or
1000collection paths in it. Python creates the explicit registered-source index
before opening the native owner. `registry()->bytes` returns canonical closed
`{format:'python-original-bytes-v1',roots}` with roots sorted by logical root;
each root row is `{root,storage,custody}`. Its digest must equal the plan's
artifact_registry_sha256. Cap this registry/describe metadata at8MiB, pre-admit
its actual retained/copy/parser costs under both existing owners, and reject
unregistered or duplicate/foreign roots. Historical traversal derives original
collection roots from the same parent update chain and requires registry
membership. Live plan includes its currently issued batch plus committed roots.

Generic Rust trait methods are `registry`, `describe`, `read`, `recheck`;
registry/describe/read return budget-owned bytes under supplied Budget/Control
and fixed metadata/file limits. The PyO3 adapter implements trait.read via
Python.borrow, direct admitted byte copy and explicit loan close before returning
the Rust-owned bytes. In-memory Rust test providers implement the same trait
without Python/SQLite. Only the bridge imports PyO3 callback types, preserving
source/parser testability. `Sources` uses an explicit physical/logical access
variant; it never fabricates a File/Metadata for a logical member. Virtual own
episode roots derive from the registered collection root plus validated prefix;
they need not exist as filesystem directories. Physical parent/numeric/artifact
roots retain ordinary stat/open/inventory logic. Convert direct collection
read_dir/stat uses to this same explicit inventory API, rather than adding a
second incomplete journal-only path.

`CardInstanceNativeSession.from_artifacts(plan,provider,settings,deadline,
cancel_handle)` is the explicit new constructor; ordinary constructor/plan3
remain exact and reject plan4. Internal typed settings mirror existing constructor
limits; no new caller-chosen parser profile. The RAW diagnostic constructor is
`StructuredJournalTokens.card_instance_artifacts(logical_root,episode_id,
journal_file,provider,settings,deadline,cancel_handle)` and accepts only the
registered own episode/viewer with ordinary corrected Journal6 metadata checks.
Both require the new capability marker before production use; no Python fallback.

`OriginalByteLease` is a private concrete object held in the actual provider's
active-lease registry. `data()->bytes` checks provider/Source/group/generation and
returns immutable exact original bytes; `close()` retires its actual group loan
only after bridge copy. Closing twice cannot free a newer generation. Borrow
holds cache admission while Python data remains referenced by native; it does
not exit a context before the returned bytes are copied. Bridge explicitly
closes every admitted loan even on type/hash/length/copy failure, preserving the
initiating error. Close uncertainty leaves that actual lease in the provider's
reachable registry and the native operation becomes sticky cleanup-only; Drop
must not silently perform reportable I/O. Provider session close confirms FFI
executor/thread retirement and zero active/uncertain loans before pool/FD/budget
release. Native raw copies/arrays retain their ordinary Rust leases independently.
Callbacks do not query session mutex/budget methods or acquire parent/catalog
locks. Fixed external-held allowance covers maximum one-group raw/codec/copy
working storage and provider metadata, so callback allocation cannot borrow a
second unaccounted full budget.

Packed native control summaries use schema4 and kind
`card-instance-packed-source-controls`, with exactly current corrected-summary
keys plus `storage_binding,artifact_registry_sha256`; collection rows retain
existing keys and carry proof2. Their original-control fingerprint/correspondence
uses the same Python packed recipe including immutable generation/inference and
physical custody. Native authorization/prepare receipts are explicitly packed
variants with existing exact fields and changed schema/kind tags; their bindings
must match this session's plan/summary. Original16/restage and numeric payload
algorithms remain identical. Exact final64/qualified formal variants and their
retained-provider extension are still the next gate, not inferred from plan4.

Exact cross-language variants:

- Packed top summary keys are exactly current corrected summary keys
  (`schema_version,kind,reader_contract,mode,parent,collections,
  authorization_max_bytes,qualified,parent_qualification` plus existing corrected
  binding fields) plus `storage_binding,artifact_registry_sha256`; version4 and
  kind `card-instance-packed-source-controls`.
- Nested parent summary keys are exactly current corrected parent keys
  (`schema_version,kind,validation_scope,npz_validated,journal6_validated,qualified,
  parent_qualification,directory,state,clocks,model,execution,semantics,
  actions_sha256,identity_sha256,generations,control_fingerprint,parent_index,
  run_sha256,run_limits,hardware` plus corrected binding) plus `storage_binding`;
  version4, kind `card-instance-packed-parent-controls`. Scope/false qualification
  flags retain their present values, and state must be explicit packed run4.
- Packed authorization has NO kind, matching the current authorization pattern:
  exact keys `schema_version,summary_sha256,parent,collections,storage_binding,
  artifact_registry_sha256`, version4. Parent acknowledgement keys stay exactly
  `parent_index,config_sha256,resume_identity,sealed,controls_sha256`; collection
  acknowledgements stay `collection_index,manifest_sha256,
  expected_schedule_sha256,expected_bindings_sha256,source_set_sha256,
  correspondence_sha256,numeric_sha256`. Top hashes bind the entire packed
  summary/registry, including proof2 custody. No extra nested kind/version.
- Packed live prepared receipt has exact current keys `schema_version,kind,
  summary_sha256,collection_manifest_sha256,source_correspondence,
  numeric_manifest_sha256,numeric_files_sha256,optimizer_steps`, version2 and
  kind `card-instance-packed-source-prepared`.
- Packed historical finalized receipt has exact keys `schema_version,kind,
  summary_sha256,qualified,parent_qualification`, version2 and kind
  `card-instance-packed-source-read-finalized`; qualification stays false /
  not_performed. Checks compare every binding with the SAME retained session.
- Cohort/recorded-cohort payload schemas/tags and exact key sets stay UNCHANGED
  (schema2, `card-instance-cohort-payload` or
  `card-instance-recorded-cohort-payload`). Numerical banks/episode metadata have
  unchanged meaning; source_proof is proof2 and correspondence uses the packed
  recipe. Decoder gets explicit packed context from SAME plan4 owner and checks
  those paired variants; ordinary plan3 payload consumers reject them.
- Original16 packet stays schema1, kind `card-instance-recorded-episode-payload`,
  exact RECORDED_PACKET_KEYS and numerical layout unchanged. Its existing source
  hashes/correspondence must match SAME retained proof2/summary in packed context.
  final64 packet choice remains its later gate. `original_config_inputs` likewise
  stays schema3/kind `card-instance-original-config-inputs` and its exact existing
  keys; summary_sha256 and run_sha256 bind the actual packed summary/run4. RAW
  token diagnostic DTO tags/layout remain unchanged because they confer no
  Source authority and numerical input semantics do not change.

Route Sources length/hash/bytes/json/inventory/recheck/proof/fingerprint through
the explicit provider branch for registered logical roots, including direct
collection inventory code, Journal6 candidate reads, original16 restaging,
final64 restaging and standalone Tree3 RAW. All parse/semantic checks remain
Rust-owned. Explicit raw-token diagnostic constructor accepts the same logical
provider/registered own episode, cannot imply Source rights. Native new capability
marker is `card_instance_group_pack='provider1;proof2;plan4'`; publish it only
after both reviews, Python/native byte/feature/token/numeric parity and a
source-matched build. The absence rejects packed native use before runtime I/O.

## final64 and Dataset/Tree (separate dependent increment)

Keep SAME `GameFinalValidationContext`, `_IssuedFinalValidationCollection` and
`CollectionPublication`, whose optional concrete packed-publication member owns
the pending descriptor/DB and controls. DB stores logical `manifest.json` once;
pending name access maps that logical control without rewriting DB bytes.
The final-only issued owner exposes an internal capability tied to this exact
context/publication, which opens pending ArtifactSource for validation. Public
locator APIs continue to reject pending. The pending envelope locator/public
descriptor/canonical manifest are absent until retirement finishes; only that
actual owner may pass pending byte/evidence to Python/native final64 inspection.

Adapt final-only request/summary to an explicit packed variant binding proof2,
logical manifest hash and actual pending-owner evidence; no integer FD or flag
is treated as equivalent to the live Python context. Python validates same parent,
inference/schedule/roles/deadline and final Sources, native keeps its same
four-method inspect/recheck/export/close lifecycle and raw-bank loan retirement.
On any failure invalidate pending/public locator FIRST, before actor/Source/
parent cleanup; only confirmed healthy Source/actor retirement plus existing
parent-owner guards reaches composite publication. Physical sealing alone and
catalog state cannot pass this handoff. Parent custody closes in its established
order before public eligibility is accepted.

Validation custody locks the actual envelope locator inode for public packed
Sources instead of a nonexistent physical manifest; default directory custody
keeps its manifest flock. DB/descriptor originals remain retained/rechecked
alongside that lock. Lock order is unchanged: parent/Source roots in deterministic
order, then catalog; provider callbacks never obtain those locks.

Dataset3 retains the SAME complete three formal parents and three final64
Sources. It accepts homogeneous explicit packed parents/validations or existing
homogeneous directory sources; mixed physical storage modes reject in v1 rather
than silently mixing proof recipes. Parent qualification, all-final64 source
rules, split separation and prescribed labels/budgets remain unchanged. History
borrows and native RAW receive only the requested original own episode under
generation loans; source owners stay retained through fresh Tree fitting/final
rechecks. Tree3 inference uses the same composite output publication and existing
paired-model source custody. No real-data training gate is widened by storage.

## Implementation sequence and verification

1. Python packed Source/control/lineage/PPO with native/final64/Dataset guards
   still active. Tests first: stable proof2 across reopen, exact logical bytes and
   current own-view features/candidates/selection/numeric inputs, actual
   collect/update/pause/resume under bounded supported conditions, old proof1
   and configuration goldens, mutation/loan/deadline failure ownership.
2. Native provider and explicit capability/build: all physical direct-read sites
   accounted for; same raw input → same Rust/Python token/feature/numeric bytes;
   all original16, recheck/export/stale/live-loan failure paths. Rust unit tests
   use an in-memory generic provider without Python or SQLite.
3. Pending final64/Source custody/Dataset/Tree wiring and command guards removed
   only where supported. Exercise existing bounded component fixtures and owned
   runtime smokes; no fabricated genuine formal parent or expanded campaign.
   If genuine final64 cannot run without a forbidden formal campaign, report
   that exact runtime limitation separately from component path verification.
4. Separate correctness and architecture conformance for each stable increment,
   root exact-command acceptance and local commits. Whole project/CUDA/wheel/
   legacy proofs and same-HDD/concurrent-workload comparison remain required.
