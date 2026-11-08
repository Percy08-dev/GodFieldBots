# Dataset source seams / recorded-episode packet — concrete proposal v1

Status: DESIGN REFINEMENT ONLY, awaiting Root plus architecture implementation GO. Based on accepted ADR0024§18.2, `docs/design/card-instance-game-tree-dataset-v2.md` and the child-local/global8 addendum. The frozen sixteen-file solo64 handoff is unchanged. This document adds no capability publication, Source permission, migration, training, fitting or runtime claim.

## 1. Scope and dependency placement

* NativeActor owner: existing native controls/source/session/journal/staging/FFI and pure DTO decoder; `card_instance_formal.py`, `card_instance_final_validation.py`, `card_instance_source.py`, `native_card_instance.py`; relevant bounded tests. Existing cohort/Actor APIs, ordinary16/legacy semantics and serialized bytes stay exact.
* `FinalSourceBudgetLoan` lives beside `FormalGameParent` in `gf_training.card_instance_formal`. It imports no Dataset, CLI, fitting or worker module. Custody types are imported locally/under TYPE_CHECKING; the locks leaf imports neither formal nor actor.
* Dataset owner retains NEW `gf_training.card_instance_source_locks.{GameSourceCustody,ValidationManifestOwnership}`. Reuse those actual classes, not a second lock/token. NativeActor does not edit that leaf independently.
* `OriginalGameEpisodeBorrow` lives in the existing `gf_training.card_instance_source` module with the original game Source helpers. Formal/actor import it locally at factory invocation. Its annotations use TYPE_CHECKING for formal/actor types; no top-level formal↔actor↔Dataset cycle.
* The private `_Final64OriginalContract` and ONE lower `_inspect_final64_originals` live in the final-validation module. Neither is a new public Config/Source constructor.

## 2. Additive historical-only native method and exact packet1

On SAME `CardInstanceNativeSession`:

`export_recorded_episode_payload(collection_index: int, episode_index: int, retained_python_bytes: int = 0) -> (metadata_bytes, uint8_banks, int64_banks, float32_banks)`

Use the current ABI's actual `retained_python_bytes` spelling. No plan/request schema changes and no extra constructor. Existing `export_recorded_cohort`, its payload and retirement transitions remain byte-identical. Native generation is collection linkage generation0..999; Python generation selector resolves to the actual committed collection index, never assumes index==generation.

Recorded packet EXACT top-level keys:

`schema_version,kind,summary_sha256,parent_run_sha256,collection_index,collection_generation,collection_manifest_sha256,source_set_sha256,episode_index,episode_id,viewer,episode_digest,binding,endpoint,bank,decisions,arrays,correspondence`.

Values: schema_version integer1; kind `card-instance-recorded-episode-payload`; summary_sha256 is digest of the SAME acknowledged native controls Summary3, never a new summary; parent_run_sha256 is the retained original parent's run hash. Collection fields equal its acknowledged compact collection3 facts/proof; generation equals original C5 linkage. Episode ordinal0..15 follows the original selected learner-seat schedule order. episode_id/viewer/digest equal that exact selected original; source_set_sha256 and manifest SHA equal acknowledged original Source evidence. There is no qualification, issuer, target, GAE, update or numeric-publication field.

Common DTO BODY is the frozen final64 packet body's EXACT keysets:

* `binding`: full existing five game-binding fields, unchanged original qualified_game classification, synthetic_capacity=false.
* `endpoint={end_kind,outcome,reason}`: observed natural result, reason=null, no invented reward; all sixteen selected endpoints, including real zero-decision endpoints, remain represented.
* `bank={actor,episode,initial_prefix_hash,event_records,terminal_prefix_hash}`; each event record `{array_key,prefix_hash}` points to canonical original own visible event bytes, in original order.
* `decisions[]`: `{frame_array_key,candidate_descriptor,candidate_array_key,selection}` in accepted-decision order. Frame bytes are exact current frame_dict whitelist. Candidate descriptor and ordered codec2 bytes are original, with full candidate order; Selection retains original full-precision JSON logprob/value and selected index.
* `arrays[]`: existing exact typed descriptor `{key,dtype,shape,bank_index,element_offset,element_count}`; exhaustive nonoverlap/inventory, readonly banks and64MiB shards/arrays. Packet-local token keys remain e.0; dtype/shape/value checks remain shared.
* `correspondence={episode_digest,frames}` uses the existing full five-binding `_episode_correspondence_digest` / ordered frame-table-Selection recipe. It must equal the acknowledged collection correspondence's episode_digests[episode_index].

Frozen final64 payload1 keeps its existing kind/top-level keyset/bytes. Share the BODY builder/decoder with a CLOSED internal packet-profile enum: Final64Packet1 versus Recorded16Packet1. Each profile validates its exact own envelope; no accepted arbitrary kind, unknown extra ordinal, nullable membership or caller-supplied authority expectation. The controller's expected binding is obtained through the actual retained parent/Source, not a public decoder-created Source.

Raw16 token arrays, when present, retain their ACTUAL original NN header window/signature. They are diagnostic source parity, not Tree features. Dataset derives any Tree window from genuine frame/event/table DTOs using the existing common kernels; no signature relabel, cached-history slicing, Num4 substitution or Python participant-journal fallback. Original opaque refs are never categorical IDs. No whole J6 command/search/RNG transaction bytes are exposed as feature data.

## 3. Native admission, phase, lease and retirement

Pure preconditions: corrected profile; SAME authorized historical parent/ack3; actual committed collection membership; all sixteen original own learner roles natural; ordinal0..15; no old exported cohort, current packet, numeric permit or live Stage. Parent actual model/registry/sealed/source guards remain required. An idle final64 slot at next64 is allowed while retaining its originals; a final packet/Stage excludes training export.

After pure preconditions, arm sticky failed state BEFORE caller lease growth, restaging or reportable work. Reuse SAME participant Sources and accepted parser/DTO whitelist, with before/after complete original rechecks and exact inspected episode/selection/correspondence equality. Restage ONLY the selected episode; never create another parent baseline or public Source coordinator. Reuse Stage, Budget, Control, bank publisher, ExportLease and prestarted retirement. Admission covers source/caller/control/current packet/Python decode overlap; record/source/array/chunk ceilings remain64MiB, no cap increase.

Successful export returns only after Stage disk close and retirement/join. It leaves the parent authorized for READ-ONLY operations and installs one readonly-episode packet lease; it never enters NumericBound/Updating/Prepared and creates no numeric append permit. No separate retire API is added: next readonly operation/recheck/healthy close drops the bridge's consumer only when existing NumPy base-owner refs prove all packet views retired. Zero-bank packets retain their metadata quota until the same boundary. One final/training packet or raw borrow per parent is allowed, including across final64 and recorded16. Repeated raw16 ordinals are not a permission shortcut: a new guarded borrow may restage originals only after the prior borrow/lease retired and all original guards pass. Final64 export ordinal semantics remain once-only.

Failure is cleanup-only and preserves first error/actual owner; late file repair or quota recovery cannot produce a packet/receipt. Cleanup bypasses active-use permission but never confirms an uncertain join or consumed descriptor from an empty retry. Actual Source/borrow/packet buffers and Stage retire before resource-local release; parent core/executor and owned lock retire in their existing order. Native RAM base owners remain charged after core retirement until views disappear. Old cohort path semantics stay exact.

## 4. Pre-entry external ownership: exact once

Signature: `FormalGameParent(directory, *, max_owned_bytes, deadline, ownership=None, extra_owned_bytes: int = 0)`; zero default preserves standalone behavior/bytes. Require exact nonnegative int E<M before reading/decoding. Python `_ChainScan` gets limit M-E. `_open_game_parent_inputs` receives max_owned_bytes=M, extra_owned_bytes=E and that SAME scan, enforcing scan.limit<=M-E. Its native owner already incorporates E once in initial external/control ownership; do not add E again in later caller_growth.

Pin accounting with explicit fields rather than guessed subtraction:

* E: native owner's initial external base, recorded at actual native construction; S0: SAME Python-scan amount already represented at that construction.
* F=B+L: current requested external total; B other-owner/Dataset/sink/full-child base, L this source/borrow's Python control/raw/copy high-water reservation.
* Native caller growth: `(current_scan_bytes-S0) + max(0,F-E)`. Initial native extra already contains S0+E. Existing native immutable budget may conservatively keep E until actual close.
* Actual native-declared external A=`max(E,F)` while healthy; if a failed resize/cleanup cannot lower its old lease, retain the last ACTUAL declared amount instead. Track it explicitly. `parent.resident_bytes = actual joint owned bytes - A` for native, and own scan/control bytes for Python. Never subtract logical F after a failed/omitted lease update. Existing E=0 healthy formula remains the accepted one.
* Before non-native allocation, enforce parent intrinsic+F+guard<=M (and any retained actual conservative A). Before native work, its SAME budget and caller lease enforce remaining M-A; no independent8GiB session. Conservative unused initial E is an external reservation, not intrinsic or Dataset-owned memory, and is not recycled without actual counter release.

This proposal explicitly accepts that immutable native initial E may remain conservative; it does not promise native budget expansion after healthy base retirement. Root/architecture may instead approve a bounded exact resizing mechanism, but implementation must not silently change the existing Budget limit.

## 5. Concrete resource loan, custody and exclusive counters

Public resource-only class/signatures:

```
FinalSourceBudgetLoan(parent: FormalGameParent, *, custodian: GameSourceCustody,
                      base_bytes: int, deadline: float)
loan.require(parent: FormalGameParent) -> None
loan.update_base(n: int, *, custodian: GameSourceCustody) -> None
loan.admit_local(n: int) -> None
loan.retire_local(actual_read_owner: GameFinalValidationSource) -> None
loan.base_bytes / source_bytes / declared_external_bytes -> guarded int
loan.custodian -> GameSourceCustody  # SAME actual resource custodian, guarded
loan.cleanup_complete -> bool  # physical/local retirement, safe failed/expired
```

Ctor requires exact concrete active parent and complete actual custody. custodian.ownership_for(parent.directory) must be the SAME RunOwnership borrowed by parent; no inode/path copy or user callbacks replace it. Dataset updater must pass that SAME actual custodian and require its complete six-root membership live. Loan conveys resources ONLY, never Source/Dataset/fit permission.

Before Source construction, loan itself owns the registered opening scope/contract/management temporary owners. Factory privately binds the SAME resulting Source (or reachable partial Source). Failed pre-entry retires that registered opening scope; it cannot accept a caller's cleanup_complete flag. `retire_local` normally accepts only the SAME privately bound Source after cleanup_complete and no unit borrow/export remains; a failed-before-Source opener uses the token's OWN confirmed opening retirement, not an arbitrary external owner.

Every source-local admission routes ONCE through parent.admit_external(B+L); no second Source-to-session growth call. Base updates preserve L. Local source/borrow reservation is a high-water bound while that Source remains open; whole-episode transient reservation may remain conservatively covered until Source close. There is no numeric lowering interface that can retire live buffers: only actual Source/opening retirement releases L. Unit borrow closure releases its concrete export/view owner, not the Source's persistent Python data reservation.

Source close first finishes original guards, raw/DTO/export retirement and native joins, releases its actual Actor resource-exclusion borrow, and sets its physical cleanup confirmation. That physical Source confirmation excludes this separate accounting token; otherwise checking Source.cleanup_complete while retiring its own token would create a cycle. A failed final guard is still reported. Then `retire_local(SAME source)` may inspect that actual physical confirmation.

After confirmed Source retirement on healthy parent, set L=0 and call the existing checked parent.admit_external(B), preserving others' base. On poisoned/expired parent, mark local graphs retired but keep the last actual declared external reservation; fail Dataset and retain owners/locks. Only the Dataset custodian, after ALL associated caller/base buffers retire, may use existing cleanup-only parent.admit_external(0). Zero never proves cleanup.

Counter arithmetic for Dataset is `T=sum(P_i)+sum(V_i)+C+H<=M<=8GiB`:

* P_i=parent.resident_bytes: all actual parent/native ownership, including native retained final originals and current native packet/Stage/banks; excludes actual external claims.
* V_i=final_source.resident_bytes: residual Python source/contract/management/immutable-record/high-water-copy ownership ONLY. Excludes P_i/native packet bytes, base B, borrowed-reference aliases and custodian's lock ownership.
* C=Dataset/index/controller rows/labels/packing/publication/caller/retokenization reservations, each once; H=FULL configured child managed cap, not RSS or a clipped capability/domain.
* Base for parent i is T-P_i-V_i, before its local admission. It includes other parents/sources and C+H once. A new native packet delta is predicted/admitted before operation, then included only in P_i; Python DTO/reference copies are V_i/local. Recompute others' bases before further allocation.

`source.resident_bytes` reports its own exclusive local reservation/counter, not parent.resident_bytes+source. `_PreparedTreeFit` rows and packing are C and retire before Dataset/source close; do not hide them behind closed parents. Readonly getters require live owner/custody, resource cleanup confirmations do not.

## 6. Parent-derived readonly contract and canonical opener

Exact entrance retained from v2:

```
open_parent_game_final_validation(parent, directory, *,
    manifest_ownership: ValidationManifestOwnership,
    budget_loan: FinalSourceBudgetLoan, max_owned_bytes: int, deadline: float)
    -> contextmanager[GameFinalValidationSource]
```

Require exact concrete complete custody's token: `budget_loan.custodian.ownership_for(directory) is manifest_ownership`, type is ValidationManifestOwnership, and require(directory) remains true. Canonical manifest only. Opening/failure/close BORROWS it; never closes its FD/token. Dataset complete custodian alone retires locks in reverse actual acquisition order. Parent owns/borrows its ordinary run lock unchanged. No new .lock file or inventory exception at validation root.

Within admitted local opening scope, derive private immutable `_Final64OriginalContract(parent,effective_bytes,input_origin_bytes,projected_effective_bytes,actions,settings,expected_facts_bytes)`. Expected facts exact keys: `parent_run_id,parent_run_path,seed,final_generation,final_binding,code,integration`. Use protected `canonical_control_sizes` BEFORE state/record/final-binding copies. Validate full original keysets and exact roundtrips for ActionConfigV3/PreparationConfig; never merge missing historical fields from defaults. Project raw effective JSON ONLY run.split train→validation, preserve input byte origins and recompute digest. Existing pure validation_groups/registry/linkage/projection kernels derive expectations; no CLI/import/load_config/fake Loaded proxy/native-summary derivation.

Both current caller/live adapters and new readonly adapter call ONE lower `_inspect_final64_originals(contract,parent,...,pending=None|actualIssuedOwner,budget_loan=None|actualLoan)`. Caller adapter keeps existing Loaded shape/hash/input equality and Actor budget rules. Live pending additionally keeps actual issued/context/publication evidence; readonly rejects pending. Historical persisted invocation clocks remain evidence; actual work uses fresh parent/caller Control deadline. Same lower Python/native selection preserves whole64 originals, exact report wall/elapsed order, final model, metadata/roles/Replay/closure and Source ownership. No ordinal or eligibility is derived from compact native summary.

## 7. Genuine original-history borrowing API

```
parent.borrow_original_training_episode(generation: int, episode_ordinal: int, *,
                                       budget_loan: FinalSourceBudgetLoan)
source.borrow_original_selected_episode(ordinal: int, *,
                                       budget_loan: FinalSourceBudgetLoan)
# Both return contextmanager[OriginalGameEpisodeBorrow]
```

Training resolves actual committed C5 generation/collection/source correspondence from SAME retained parent; natural16 and original selected seat checks are mandatory. Final resolves actual selected ordinal0..63 from SAME retained Source. Use a shared exclusive per-parent unit-borrow slot nested under that actual source/resource loan; do not acquire a second Actor source lifetime token while final Source already holds its parent. Source and loan must be actual bound owner references. No constructor accepts arrays, envelopes, summaries, permission callbacks, flags or a fake Parent.

Chosen common borrow surface is ORIGINAL IMMUTABLE DTOs, sufficient for all Tree windows: guarded episode_id/viewer/end_kind/outcome/decision_count/episode_digest, complete `EpisodeRecords` own bank/frame/table/Selection objects, `resident_bytes`, `export_bytes`, `recheck()` and `iter_decisions()` in fixed accepted order. Each advance yields a frozen `OriginalGameDecision(frame: DecisionFrame, bank: EventBankView, table: CandidateTable, selection: Selection)` defined beside the borrow. The bank is the exact own prefix at frame.boundary. The borrow owns/guards these actual original references; no array constructor supplies them. Its guarded `records` accessor returns the corresponding genuine EpisodeRecords, including immutable original payload, full logical columns/order and full-precision Selection. Hash/codec verification happens against actual original wire inside Source decoder, not by Dataset. No full command/search/RNG J6 transaction blobs are feature inputs.

For final64, prefer borrowing already validated immutable64 DTOs from the solo64 Source: no re-export/repeated native ordinal and no extra wire bytes invented from tables. The API does NOT promise retained original codec2 binary payload/descriptor or whole J6 line bytes after materialization; logical table DTOs are original facts verified by codec2/hash/correspondence. If the Dataset requires exact raw wire bytes rather than these DTOs, that is a separate additional gate, not a hidden reconstruction. For training Rust, new recorded-episode packet supplies actual original codec2 bytes and DTO whitelist, decoded once under its retained export lease. Python backend borrows SAME validated original Source records; no backend switch or Python journal fallback for Rust.

Context entry/each advance/end/exit rechecks parent/source/custody and fixed membership, admits wrappers/retokenization before allocation, and retains actual source/export owner through work. Zero endpoints expose real bank/closure and zero decisions. On close, all packet/views/current raw graph retire before next borrow; base/local custody remains. Dataset uses common corrected real18 frame/event/table/token kernels at its EXACT Tree encoder window, then PreparedTreeFrameV3. NN token caches, logprob/value/GAE never become features or selection teachers.

## 8. Required bounded evidence and remaining decisions

Implementation, candidate builds, schema fixtures and tests begin ONLY after Root+architecture GO on this exact proposal. Required: old cohort/Actor/default bytes; raw16 natural viewer/outcome/zero/full candidate/codec/Selection parity and Tree-window reprojection; wrong parent/cohort/ordinal/summary/source/lease/cancel/late repair rejection; six-root custody; extra-owned before first read/copy; B/L high-water and source-close preserving B; failed-parent conservative reservation; child FULL cap plus two other parents; actual cleanup/reap/lock ordering.

A genuine SHORT native17 historical parent can later exercise NEW recorded-episode positive FFI as diagnostic only, preserving original identity/flags and opaque canonical data. It supplies NO formal Source/qualification/Dataset/learning evidence. Solo64 four-method admitted-formal positive FFI stays NOT RUN until a genuine admitted parent exists. New Tree3 options/inference/J6 RAW adoption and runtime golden0279d5... are a FOLLOWING separately scoped parser/capability increment; no model/runtime/Source/pair claim comes from that controlled schema golden.

Root/architecture decisions to pin: recorded packet1 exact envelope above; residual-Python DTO borrow rather than an impossible promise of retained final codec wire; conservative initial-native-E floor/exclusive counter arithmetic; resource-loan high-water lifetime and actual opening/Source retirement association. No code/markers/tests or frozen sixteen files changed to prepare this document.
