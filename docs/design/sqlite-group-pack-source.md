# Group Pack: original Source, lineage and native byte provider

Read-policy amendment: [ADR0035](../adr/0035-pack-checksum-read-policy.md) and
[the accepted Pack checksum design](pack-checksum-read-policy.md) supersede
eager/repeated whole-DB read verification and custody-v1-only production.
New evidence uses custody2; historical bytes/evidence stay unchanged. All other
information, Source authority, ownership and format contracts remain applicable.

Date: 2026-11-10. Owner: root. Status: Python Source/lineage/PPO and Native3B
designs accepted after independent architecture reviews; Python Source/lineage/
ordinary PPO and Native3B implementation accepted within their bounded
verification scopes;
formal/final64/Dataset design is separately accepted in
[final sources](sqlite-group-pack-final-sources.md). Final integration
implementation and final conformance remain pending.
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

### Real-cohort admission closure (2026-11-11)

An actual sixteen-game collection exposed duplicated conservative decode
reservations: the manifest pass reread opaque members, the Source parsed journals
before the logical reader parsed them again, and both the byte reader and parser
reserved decoded graphs. The default2GiB budget remains unchanged. The bounded
remedy is architecture READY; independent implementation reviews remain required.

For initial packed inventory verification, compare every manifest name/length/SHA
against the index produced by the constructor's COMPLETE original DB/Pack/member
validation. Remember its exact logical identity with admitted evidence metadata.
Preserve `max_source_file_mib` for EVERY declared inventory member, including
opaque diagnostics; eliminating its read does not widen file-size admission.
Do not trust fresh SQL metadata or an unvalidated external index. Preserve the
complete-name closure, group/episode checks and final SAME-original DB hash/stat
rechecks. This removes redundant opaque reads without weakening byte custody.

For packed journals, the concrete logical reader performs the single parse and
descriptor admission; Source applies `_game_header` to that validated header
before projecting records. The single logical parse must preserve the outer
parser's duplicate-key and nonfinite-number refusal. Directory preflight remains
unchanged. The concrete
`OriginalArtifactReader` uses a narrow raw-only read path: immutable returned
bytes/aliases retain conservative raw-byte admission until explicit graph
retirement or close, while `admit_validation` separately covers actual JSON/table
graphs BEFORE allocation. Generic `owner.object()` still admits JSON decoding.
Evidence metadata is charged once per remembered logical entry; separate live
parses/copies must never be deduplicated merely because their filenames match.
Keep the existing records-plus-artifacts refusal and exact concrete-reader gate.

Verify the saved failed cohort structurally with allocation category totals,
unchanged proof/numeric outputs and corruption/replacement refusals. This conveys
no parent/update qualification and cannot rehabilitate the preserved failed run.
Fresh actual PPO evidence is required in new run directories after both reviews.

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

Phase clarification: existing `ops.retain` is a SEPARATE subsequent owned parent/
pruning validation, after generation commit in both directory and packed runs.
Its new scan must validate newly committed run/generation/registry controls before
destructive pruning; retired mutable batch snapshots cannot authorize it. This
is an allowed future-parent validation, not a prepared batch Source callback,
finally guard or deferred close. It may fail the overall parent while preserving
the true committed generation/update/optimizer history and ineligibility; never
invent a rollback. It must retire its OWN scan/pool/FDs on every outcome. The
prepared Source remains incapable of new original I/O. This distinction was
confirmed by both reviewers after a targeted retention-reopen failure probe.

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
custody,files,index}` with proof2 storage/custody and name-sorted
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

### Native index-integrity amendment (architecture READY, 2026-11-11)

Implementation found that the original closed describe shape exposed member
group IDs but omitted the validated SQL group/episode records. That prevents
Rust from checking the same full index/manifest correspondence as Python3A.
The bounded amendment passes independent architecture review; dependent index
implementation may proceed. Final correctness/conformance and capability
activation remain pending.

`index` has exactly `groups,episodes`. Groups are group_id-sorted closed
`{group_id,record}` rows. Episodes are episode_id-sorted closed
`{episode_id,group_id,game_id,record}` rows; physical null game_id stays null.
All values come ONLY from the SAME constructor-validated original index, bound
by the registry's DB/storage/custody. No fresh SQL or caller-trusted flags.

Rust checks exact group/episode sets against the original complete manifest,
canonical group record equality to the corresponding manifest health row,
episode-to-schedule group mapping, game_id equality, and episode record equality
to `{end_kind,failure}`. Existing member-to-schedule group checks remain. It
still parses all manifest/journal/Replay semantics itself. Diagnostic RAW gains
no Source rights. Python owns byte/index custody, not a duplicate semantic parser.

The whole describe, including index records, retains the existing8MiB cap and
pre-admission of actual retained/serialized/copied/parsed bytes in both owners.
Source counts and duplicate/unknown rows refuse before semantic work. No new
dependency, persisted schema, numerical input, policy feature or old plan3 ABI
changes. Provider1/plan4 are not yet released; close this unreleased contract
before capability activation. Tests must reject altered group/episode records,
game IDs, mappings and row sets with exact fixture parity and cleanup intact.

Alternative considered: Python transiently parses the original manifest and
calls validate_index before native admission. It preserves the old callback
shape but adds duplicate cross-language semantic ownership and managed-parse
lifetime. Root prefers the explicit bounded metadata closure so Rust retains
the agreed Source semantic responsibility; architecture review confirms this
bounded closure with the existing whole-describe cap/admission requirements.

### Native SQLite thread ownership amendment (architecture READY)

The first private full-Source fixture matched proof/arrays/Original16 but failed
final close: callbacks created SQLite on the FFI worker, then coordinator cleanup
ran after that worker had joined. Earlier caller control reads can also leave a
caller-owned connection. Root adopts the advisor's bounded per-operation handoff;
The creator-state/accounting closures below pass independent architecture review;
dependent retirement implementation may proceed. Final implementation reviews
and preserved-fixture closure remain pending.

A default protected executor factory in NativeStructuredSource preserves the
physical path. The concrete packed owner selects a private packed executor
adapter colocated with its provider; the generic executor imports no storage.
Before each submission, the caller retires its own active cache/connection and
admits current external growth. The worker executes the constructor/operation
and retires its actual IO BEFORE publishing completion. No caller pool access
overlaps that worker interval. Successful result/prepared authority requires
confirmed retirement; cleanup failure poisons ordinary work permanently.

Actual SQLite closure is attempted on its creating worker even when loan/cache
retirement fails. Retain cache, generation, leases and charges; clear pool.active
only after complete retirement, not merely a successful DB close. Standard-library
connection close may be retried on that SAME actual object/worker for cleanup;
retain it until success. Installed CPython3.14.10/SQLite3.54.1 same-thread double
close was observed to remain closed. This permits no raw-FD retry or Source
health recovery. No check_same_thread bypass, alternate connection or IO service.

Cleanup-only scheduling on the SAME still-live worker bypasses work health/
deadline gates, performs no new read/hash/open/admission and never reenters a
native budget from a callback. Preserve first error and actual owners. One
bounded attempt per explicit close invocation; unresolved state stays reachable
for later disposition. Do not join/shutdown the executor until the actual native
session, Future/callback, SQLite connection, provider cache and loans are retired.
Only after checked join may coordinator release pool/original FDs/parent locks.

OriginalByteLease needs staged retirement and an owner registered BEFORE borrow
side effects. Mark context/budget attempts before invoking them; a second empty
generator exit is never proof. Record success or direct supported evidence from
the SAME actual loan/context/account under the exclusive interval; otherwise
retain uncertainty without forcing loan counts or charges to zero. Every separate
data/cache/budget/registry stage must remain owned until proven complete.

Bind provider release to the actual executor lifetime. An outer provider context
cannot free metadata or pool allowance when nested native close failed, even if
byte leases are zero. Constructor failure before handoff retains caller cleanup;
after binding, only confirmed executor retirement permits provider release.

External accounting includes positive SAME-pool metadata/control growth beyond
the prepaid baseline, after confirmed caller cache retirement. Decompose baseline,
sealed/caller bytes and maximum one-group allowance without double counting.
Admission must precede argument construction/FFI allocation, including constructor
settings and summary calls, not only methods with explicit _held arguments.
Callbacks remain inside the prepaid allowance. Ordinary resume/parent validation
also derives plan4/provider from actual retained scans; native snapshot=None must
not take Python-only decoded-graph retirement branches.

Acceptance requires real alternating caller/worker SQLite thread checks, constructor/
operation/expiry/sync-interrupt/async-cancel tests, close errors before and after
actual SQLite close, uncertain lease/context/budget stages, outer-context cleanup
refusal, sufficient/insufficient growth and successful final close on the preserved
unqualified one-update fixture. Generic physical executor behavior and the frozen
physical benchmark files remain unchanged. No new dependency, persisted format,
policy input, numerical algorithm or capability activation is authorized here.

#### Creator states and storage-free cleanup (N1 closure)

| State | Actual IO owner | Permitted transition |
| --- | --- | --- |
| caller IO / later caller-retirement uncertainty | recorded creating caller thread | retire that actual cache/connection on caller; no ordinary worker submission until confirmed |
| submitted/running Future | exclusive recorded FFI worker interval | caller cannot touch pool; worker retires its own IO before completion |
| worker-retirement uncertainty after Future completion | same still-live worker | cleanup-only same-worker submission; caller never closes this connection |
| idle, no IO | neither thread has a connection/cache/loan | normal caller preflight or cleanup, unless any prior failure made work sticky |
| joined | confirmed native/session/IO/loan/Future retirement | caller may release provider metadata, pool/FDs and locks |

Ownership records the creating thread BEFORE connection-capable work, including
failure after binding. A failed caller handoff remains caller-cleanup-owned;
executor binding does not transfer it. A worker cleanup lane may retire the native
session while caller IO remains only if it is storage-free, and cannot then join/
release the provider until caller retirement is confirmed. Unknown creator or
partial ownership remains cleanup-only and reachable.

Cleanup entry permanently fences ordinary provider callbacks (registry, describe,
borrow, recheck) before invoking native close. It also cancels the independent
native Control first: current OriginalInputs.recheck starts with Control.check,
so close cannot hash physical parents before it reaches the provider. A callback
attempt must refuse without IO, never return fabricated revalidation success.
Existing native close retires its stage and records Closed even when returning
that failed check; a later close on the SAME native object may confirm retirement
while the original error remains primary. If that behavior cannot be established,
retain session/executor rather than treating an error as closure. Normal healthy
close keeps its existing final revalidation. Tests instrument ALL source read/
hash/open paths, not only logical callbacks.

#### Disjoint account and pre-argument builder (N2 closure)

Let L be the configured joint native cap. After provider construction and confirmed
caller cache/connection retirement, define these prepaid constants:

| Term | Ownership represented exactly once |
| --- | --- |
| C0 | original caller extra plus preflight binary evidence, excluding only a proven SAME _ChainScan pool-backed credit |
| B0 | existing native Python owner base/identity plus a prepaid constructor settings/plan encoding envelope |
| P0 | cache-free SAME-pool used bytes, including original indexes, registry/describe and actual pool-backed control charges |
| W | maximum one-group callback/cache working bytes plus maximum control-copy overlap and bounded callback/hash scratch |

Compute N = L-C0-B0-P0-W before constructor encoding; nonpositive N refuses.
Rust receives memory_bytes=N. Obtain actual compressed lengths for W through a
bounded caller-thread metadata query on the SAME already validated original DB,
checking exact group IDs and original rechecks before/after. Retire that query's
temporary allowance and connection BEFORE measuring P0. W is
max_groups(4*raw_bytes+2*compressed_bytes) + max_controls(2*bytes) + 1MiB.
No raw payload query or second full budget is used. Constructor wire scratch
is bounded allocation-free before encoding and added to B0, not silently
deducted again as dynamic caller bytes.

A private _NativeGameInputs shared_controls argument may reference only the
actual _ChainScan for this parent and SAME provider pool. Before crediting, require
pool-control charge to cover its current retained amount and that amount to be
included in original caller extra. Credit exactly that actual retained amount
once from C0; no arbitrary integer credit, filename deduplication or other credit
is accepted. Other callers keep conservative extra ownership. Existing disjoint
sealed/persistent/caller-growth accounting remains S(t); omit from S any exact
shared-control ownership already represented by P(t).

For each ordinary operation, retire caller IO first, then read cache-free P(t).
Set D(t)=max(0,P(t)-P0), H(t)=D(t)+S(t)+K(t), where K is the precomputed argument
encoding/copy envelope. Prepaid P0 and W never appear in H. Rust's current used
account includes its previous caller lease h_prev; on a healthy idle session
R(t)=used_bytes-h_prev. Require R(t)+H(t)<=N BEFORE building arguments. Track
h_prev only from known-successful calls that set the explicit retained argument;
after any failed/uncertain call, ordinary work is forbidden, so an unknown caller
lease is never used for later admission. A non-growing call replaces its caller
lease rather than charging prepaid or previous ownership twice.

Use one concrete protected preflight/builder seam on NativeStructuredSource
(physical default preserves behavior). Packed calls perform retirement, growth
and remaining-account checks, then invoke the supplied bounded argument builder,
then submit through the packed executor. No canonical(plan/auth/settings) or
other affected argument encoding occurs ahead of that seam. Constructor uses the
prepaid B0 envelope and N calculation before settings/plan encoding; control_summary
must explicitly receive H(t), as do authorize/recheck/export/restage/prepare/
finalize and other retained-argument calls. Scalar used_bytes is an idle-only
accounting query outside callbacks. The executor factory/submit hook alone is
not the admission seam. Tests use builder/encoding sentinels on constructor and
summary, populated-cache baselines, smaller positive metadata growth, exact
sufficient/insufficient caps and unchanged physical behavior.

#### Terminal original-pool close hold (architecture READY)

Outer _ChainScan/PackedBatchCustody exits can call PackedSourceOwner.close or
PackedArtifactPool.close independently after a nested native cleanup error.
Provider.close's executor guard alone does not protect those original FDs.
Proposed enforcement adds one actual low-level pool-close hold, owned by the
concrete packed executor before its first submission. Both owner.close and
pool.close refuse BEFORE any original-FD/account release while a hold remains.
_ChainScan.close also refuses before borrowed-custody detachment or refunding
its _pool_control_charge; borrowed scans otherwise bypass both owner/pool close.
Retain that actual scan/control ownership, rather than guessing its graph died
or transferring a serialized reservation. The N2 pool-backed credit must remain
represented until actual native retirement.
Pool code imports no native provider/executor; the actual hold is a reachable
capability with a retained cleanup owner, not a serialized flag or counter waiver.

The packed adapter releases that SAME hold only after actual native/session/IO/
loan/Future retirement and checked executor join. A partial constructor either
has made no worker effects and retires its hold, or retains its actual cleanup
owner and hold. Uncertain close/join or outer-context failure cannot release it.
Worker IO retirement is not terminal pool close and remains available while the
hold is active. Multiple shared-pool native owners each retain their actual hold;
no wrapper decrements another owner's hold or forces a zero counter. Admission
and release bookkeeping are included in the same pool's metadata account.

This touches only the non-frozen group_pack_pool.py, group_pack_game_source.py
and the assigned structured_lineage.py terminal guard, plus native adapters/tests.
No physical PackedSource/core/benchmark-file
change or third-party dependency is proposed. Tests must prove outer scan/custody
and direct pool close cannot release FDs/charges with an active/uncertain executor,
then confirm release after actual joined retirement, including partial binding,
multiple owners and primary-error preservation.

### Native logical-name parity amendment (architecture READY)

Python3.14.10 safe_name rejects all Unicode general-category C code points;
Rust char::is_control alone only covers control characters. Relative logical
names must retain the SAME validity across readers, including opaque names.
Do not narrow valid Unicode names to ASCII or add a dependency for this closure.

The accepted native helper uses a frozen Unicode16.0.0 category-C range table,
generated from the supported Python3.14 runtime, with version and exact bytes
recorded in source/build identity. The736disjoint ranges require5888bytes of
u32 endpoints, plus small static metadata. Decode UTF-8 strictly, retain existing
traversal/NUL/backslash/name checks, and reject membership in any C range.
The concrete Python adapter verifies the same UCD version before new packed
allocation. New metadata fields or policy inputs are unnecessary; plan3 remains
unchanged and the unreleased provider1 contract becomes exact before activation.

Root's disposable data artifact is `/tmp/gfb-ucd16-category-c-ranges.json`.
Acceptance requires exhaustive code-point/table agreement with Python and named
Cc/Cf/Co/Cn/surrogate/valid-Unicode cases, including escaped JSON keys/names,
without changing physical format/Config or accepting a permissive fallback.
Future Unicode-version changes require a separate compatibility review rather
than silently reinterpreting old name validity.

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

### Generation-zero packed parent (architecture READY)

Ordinary fresh PPO validates its run4 parent BEFORE the first collection exists.
That historical-parent plan has a real generation0/control owner, zero committed
collections and no original byte roots. Rejecting an empty provider prevents
fresh native PPO from reaching collection, despite a valid parent.

Permit a concrete empty registry over an explicitly supplied SAME PackedArtifactPool
owned by the actual retained _ChainScan. Create/admit that pool and its control
charges even when no PackedSourceOwner exists. OriginalArtifactProvider accepts
zero owners only with that actual pool; it must not infer a pool, fabricate a
Source owner, or accept an arbitrary root/descriptor flag. Zero-owner construction
also requires the exact retained shared_controls _ChainScan: validate its actual
pool identity, control-charge coverage and actual run4 parent binding BEFORE
empty-provider/native allocation. At the concrete _open_game_parent_inputs
factory, require the validated actual state/chain to have generation0 and no
committed roots, and plan.parent to equal that scan.root. shared_controls=None
cannot admit an empty provider. Preown the pool on this scan before shared
control admission; partial initialization/admission failures retain the actual
scan/pool owner and stage their reservations. This does not introduce a global
pool registry or a serialized ownership assertion. Registry remains the
exact closed format with roots=[], and its canonical digest binds plan4. Unknown
describe/borrow/recheck roots always refuse; no episode/source authority appears.
Metadata ownership, close holds, thread lanes and N1/N2 accounts remain actual;
the no-group/no-control W case still includes its bounded scratch allowance.

Rust derives expected roots from the SAME actual parent update chain: only a
historical-parent mode with zero committed collections may match the empty
registry. A live cohort requires its issued actual collection; any committed
source requires its exact registered root. Existing run/identity/config/generation0
checks remain mandatory. Do not fall back to plan3 for run4, waive a missing
Source, retrofit old parents or return a successful Original16/export for no data.

Tests must execute ordinary fresh validation-before-collection and later
nonempty/paused-resume dispatch through the real scan/provider factory, with
source-kind/authority/unknown-root refusals and balanced actual pool/control/
hold/FD cleanup. Diagnostic fixtures remain unqualified. This closes an
unreleased plan4 ordinary-path gap without a persisted-schema/old-plan3 change.

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

Dataset3 remains the existing directory-original format. The separate Dataset4
subtype in the accepted [final-sources design](sqlite-group-pack-final-sources.md)
retains the SAME complete three packed formal parents and three packed final64
Sources. Mixed physical storage modes reject in v1 rather than silently mixing
proof recipes. Parent qualification, all-final64 source rules, split separation
and prescribed labels/budgets remain unchanged. History
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

## Controlled lineage publication on DrvFS (architecture READY)

The first actual concurrent block fails before collection: _write keeps its
writable temporary FD open through os.replace and immediate path verification.
A persistence-only reproduction fails6/6on D: and passes6/6on /tmp. An independent
12case namespace probe on the actual HOST D:9p/DrvFS mount shows ENOENT for all
six renames with either a writable or readonly FD held; both close-before-rename
variants succeed6/6. Held descriptors remain valid, and closing them makes the
renamed name visible. This is not an eligibility/schema/config/monitor failure.
Raw evidence: /tmp/gfb-rename-visibility-diagnostic-v2.json and preserved first
block63f2b9bef0f34cfe97e5ea599ea9245e. No outcomes/updates are inferred or repaired.

Revise only owned _write control publication, preserving its signature, canonical
bytes, old Config/hash/format domains and immutable-versus-mutable semantics.
It is a controlled write-close/namespace/readonly handoff, not a Source-admission
relaxation or a retry for arbitrary missing files. Use the same contract on both
filesystems/formats; do not infer a filesystem type or add a configuration switch.

Before namespace publication, the actual owner exclusively creates its nonsymlink
sibling temporary, writes/flushes/fsyncs it, verifies held object versus temporary
path and exact bytes under the existing scan/deadline/budget, and captures device,
inode, regular mode, length, mtime and SHA256. Actual writer close must be confirmed
once BEFORE rename/link. An uncertain close retains its actual owner/capability
and first error, prevents publication and is never retried by raw FD number.

Then perform the existing single mutable replace, or immutable exclusive link and
owned temporary unlink, with no live FD to this original across these namespace
operations. All parent directories stay owned nonsymlink paths. Do not poll for
visibility, retry rename/link, overwrite an immutable target, reopen another
inode as a replacement, or treat eventual file appearance as successful cleanup.
Namespace/close/inspection failure leaves the write failed and its actual owners
reachable; no scan snapshot is advanced and no healthy caller result follows.

Open the final original readonly with O_NOFOLLOW|O_NONBLOCK. Bind it to that
ALREADY controlled original through exact device/inode/mode/length/mtime and full
SHA256. ctime may change ONLY across this owner's confirmed writer-close and
own rename/link/unlink operations, matching the accepted terminal-control
handoff. Capture the new ctime only after those operations, then require strict
six-field held/path identity and exact bytes throughout readonly verification,
checked close and final path verification. A same-size mutation, path/symlink or
inode replacement, final read/close failure or later ctime drift fails. Preserve
retained readonly owner/uncertainty on failure; do not reconstruct authority from
an arbitrary pathname or weaken ordinary SealedScan/Pack/DB Source checks.

Advance scan.files[path] to the verified post-readonly-close snapshot and same
canonical hash ONLY after every actual close and check succeeds. Handoff checks
include mode in their six-field identity; scan.files retains the EXISTING
_stat_identity five-field projection (device,inode,size,mtime,ctime), consumed
unchanged by SealedScan and _ChainScan.finalize. Do not widen those tuple formats
or relax their checks. Existing
charges and deadlines still apply before effects; new private owner bookkeeping
must also fit them. Keep original first errors and uncertain capabilities attached
to actual exceptions/owners. No new persisted format, dependency, public authority,
policy visibility, simulator change or learning condition is introduced.

Required verification: reproduce original open-FD rename failure without games;
old/new control bytes and _write signatures unchanged; actual close occurs before
both mutable replace and immutable link/unlink; mutation/replacement/symlink at
each handoff rejected; wrong inode/mode/length/mtime/hash and post-handoff ctime
change rejected; failed write/close/rename/link/unlink/read/readonly-close retains
actual owners and never advances snapshot; no raw close/rename retry; historical
lineage/Source/resume regressions, formal frozen algorithm ASTs and static checks.
After both reviews, run bounded persistence-only checks on ext4 and actual D:
for both publication modes, then a NEW matched four-trial concurrent block.
The original failed block stays failed and immutable.
