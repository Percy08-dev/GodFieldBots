# Qualified game Dataset3/shared fresh fit — refinement v2

READ-ONLY proposal for Root/architecture gate; implementation is NOT authorized.
Read current ADR0024 §18.2 after the validation-lock question. Baseline Actor
acceptance is MAIN `c96252aa42e35a601a0626b42fe583c1f7313e53`.
V1 remains frozen at `/tmp/gfb-card-instance-game-tree-dataset-design-v1.md`, SHA
`d9f671aeed08877301276ebba06a1025a0a441761a6bcc6900b852b156a437b8`.
This v2 replaces v1's open choices with Root's pinned directions and specifies
missing seams. It does not edit code or execute tests/imports/Source/model/build.

## 1. Retained scope and exact ownership

Keep v1's minimal implementation split and shared-kernel reuse:

* Future Dataset implementer: new `gf_common/card_instance_game_tree.py` pure real
  records; `gf_training/card_instance_game_tree_data.py` concrete Dataset/originals
  custody and joint governor; narrow `card_instance_source_locks.py` manifest
  flock owner; relevant controlled tests.
* Shared fit/records implementation only after explicit ownership GO:
  `structured_tree_fit.py`, `structured_tree_worker.py`,
  `gf_models/structured_tree_pairing.py`, `structured_tree_artifacts.py`.
  Same row preparation, fresh child, feature kernels, full-Booster verification,
  pair4/copy/load and existing terminal completion publication; no cloned loop.
* NativeActor owner exclusively coordinates existing `card_instance_formal.py`,
  `card_instance_final_validation.py`, `native_card_instance.py` and any narrow
  delegation into `card_instance_source.py`. Native is currently editing its
  single-parent default path. All seams below are requested design, not assumed
  implemented APIs; no existing file may be overwritten by Dataset work.
* Root owns CLI/config/docs: `game-dataset build`, `game-dataset inspect`,
  `fit-game-tree`; corrected Tree3 dispatch precedes legacy defaults. Retain
  existing `TrainingConfig`,500/50/.05/min20/thread defaults and fresh starts.

`VerifiedGameTreeDataset3` retains SAME actual THREE `FormalGameParent` and THREE
completed `GameFinalValidationSource` objects ordered seeds17/29/43. It implements
`manifest` clones, actual `source`, single-pass `selected_inputs`, `reserve_sink`,
`retained_without_sink`, `verify_complete/finalize/close/release/cleanup_complete`.
Only its exact active concrete type authorizes the real fit context, never an
array, callback, edited origin, Dataset hash, contract value or native summary.
GameTree3 worker/Journal6/evaluation routing remains NEXT, not a completed stub.

Include every committed16-game training cohort generation0..U-1 and each parent's
exact final64 (192 validation games total), including all zero endpoints. Work
order is parent slot, ascending train generation, final validation, original
schedule/group/episode order, accepted own decision order. Select only recorded
learner/final-policy seats, including training self-play's explicit one seat.
Use observed terminal `journal._reward(outcome,viewer)` ±1/0 at every accepted
own decision, gamma1; no GAE/advantage/value supervision or invented empty row.

Three originals must be distinct, healthy positive complete formal declared
parents with all Run3/Gen3/update3/registry3/C5/J6/Num4/final inference4 originals.
Require one homogeneous full formal model/actions/input/execution/code/recipe
variant. Allow ONLY declared seed/name/path/run differences; approved Simulator
hashes stay equal. Verify exact Config and transitive input byte origins PER
PARENT, not equality of different seed TOML bytes. No current-default overlays.

## 2. Paths-only reopening: smallest parent-derived readonly seam

Problem: existing `open_game_final_validation(loaded,parent,directory,...)` needs
caller `LoadedConfig`; training cannot import `gf_cli.config`, guess a root TOML
or reconstruct an ExperimentConfig through today's defaults.

Proposed NativeActor-owned public entrance:

```
open_parent_game_final_validation(parent: FormalGameParent, directory: Path, *,
    manifest_ownership: ValidationManifestOwnership,
    budget_loan: FinalSourceBudgetLoan,
    max_owned_bytes: int, deadline: float) -> contextmanager[GameFinalValidationSource]
```

This entrance accepts NO caller Config/JSON/summary/contract. It first checks the
actual parent, complete custody token and resource loan, then pre-admits complete
protected-parent/control/parse/copy bounds before deriving any contract or reading
canonical C5 management JSON. No new native request mode or Source flag.

Inside the existing final-validation module, create a PRIVATE immutable
`_Final64OriginalContract` with exactly:

`parent` (actual reference), `effective_bytes`, `input_origin_bytes`,
`projected_effective_bytes`, `actions` (ActionConfigV3), `settings`
(PreparationConfig), `expected_facts_bytes`.

`expected_facts_bytes` has exact keys
`parent_run_id,parent_run_path,seed,final_generation,final_binding,code,integration`.
`final_binding` stores the exact label plus RuntimePolicyBinding from guarded
`parent.final_binding()`, including expected actual final inference-file SHA.
No ordinal or field comes from a native compact summary.

Factory order, all within the admitted resource loan:

1. Require SAME active original parent and `recheck`; use guarded
   `canonical_control_sizes` to reserve BEFORE `parent.state`/final-binding clones.
2. Read its validated ORIGINAL effective Config/input inventory once. Verify its
   recorded Config/input digests and formal declaration bindings remain exact;
   existing parent qualification already checks the literal recipe and full
   inactive settings. Store canonical bytes and derived immutable size ceilings.
3. Parse actions with existing exact-field `action_config_v3`. Require the FULL
   original `PreparationConfig` field keyset before `preparation_config`, then
   require round-trip canonical equality, so its merging helper cannot fill a
   missing historical field from defaults. Do the same exact-full-record checks
   for any other typed subrecord; use existing recorded parsers, not CLI Config.
4. Project raw effective JSON by changing ONLY `run.split=train→validation`;
   recompute effective digest, keep original input byte-origin inventory unchanged.
   Original run limits/execution bindings/configuration inputs are data from the
   parent's protected originals. Never load TOML or call `load_config` here.
5. Derive fixed64 groups/registry and expected projection/linkage with existing
   `validation_groups(seed,label,actions_version=3)`/pure record kernels. Contract
   methods require its SAME parent/loan live; exported facts are clones.

Factor ONE lower `_inspect_final64_originals(contract,parent,... )` body from the
existing `_open_original`: physical inventory, C5/projection/linkage, runtime,
J6/codec2/Replay/closure, exact final policy/seats, report timing, natural health,
selected endpoints and snapshot retention remain shared. The readonly wrapper
uses canonical manifest ONLY and does not require live collector issuance.

The EXISTING live actor and caller-based public opener keep signatures and
original Loaded equality checks unchanged. Their caller adapter validates the
original Loaded object/hash/effective/full input origins, then constructs the
same private contract and calls the same lower validator; it does not pass a
fake Loaded proxy. Pending live validation still additionally requires the SAME
actual publisher/context token. A contract alone grants no Source or fit right.
Repeated control/public getter growth must refuse before serializers, as in the
accepted Actor guard. Source/backend identity stays original; no fallback.

This removes the need for three caller Loaded objects and makes manifest
build/inspect/fit self-contained under actual original owners.

## 3. Six-root custody, now pinned

Acquire parent roots with existing `run_lock` UNCHANGED. Acquire validation roots
by READ-ONLY no-follow canonical manifest FD plus exclusive advisory
`flock(LOCK_EX|LOCK_NB)`, retaining actual token/device/inode. Never create or
ignore validation `.lock`, accept pending control, or lock via a Source summary.
Independent data read handles must not close the lock-owner FD.

Sort resolved nonsymlink path BYTES; deduplicate locks by actual root device/inode
using the least path-byte representative. Preserve all six logical memberships;
duplicate populations still fail qualification. Acquire all unique locks before
raw decoding/output, retire inputs/native exports/child first, release locks in
reverse order. Numeric FD leaves active state BEFORE one close; uncertainty is
latched, never inferred away by registry absence/empty retry/stale-FD probing.

Borrowing supports ONLY a COMPLETE already-ordered custody set, with SAME actual
resource tokens/inodes and retained owner, never a partial set/prefix/copied
string. Partial pre-held roots refuse. The lock owner is resource custody, not
Source authority. Use existing one-close helper; retain actual owner on failure.
Dataset index management parsing needed to discover paths is bounded/pre-admitted
control reading only; it cannot qualify originals or decode raw episodes.

## 4. Exact pre-entry and base-versus-final-loan accounting

ONE `M<=8GiB` applies to all retained owners and fit allocation. Before every
allocation, sum intrinsic parent/native ownership, residual Source Python/control
ownership, Dataset/index/caller/copy/guard bounds, current FULL cohort/episode/
token export and controller rows/packing/publication, PLUS the FULL configured
child cap. External reservations are not counted again as intrinsic.

Proposed NativeActor-owned pre-entry signature extension:

```
FormalGameParent(..., extra_owned_bytes: int = 0)
```

No change to `max_owned_bytes=M` meaning or standalone zero default. The known
other-owner/caller base must reach Python scan/native constructor/canonical-copy
admission BEFORE first read/decode. For parent2/3, already retained parent/index/
Source ownership is passed from the joint Dataset governor; a parent never
receives an uncharged independent8GiB. Intrinsic `resident_bytes` continues to
EXCLUDE its external amount. No private scan/session walk by Dataset code.

Proposed resource-only `FinalSourceBudgetLoan` (same actual parent + Dataset
custodian, no registry/permission callbacks) tracks distinct:

`base_bytes` = other owners/index/caller/sink charges;
`source_bytes` = this final64 management/contract/raw/export/copy loan.

The Dataset governor updates base before related allocations; Source admits
`base+source` through the SAME parent counter. For source i, implicit own-parent
intrinsic is excluded from base; parent/native bytes already containing this
Source are excluded from residual Source counters. Other-parent/source and sink
charges are included exactly once. Subtract only proven shared references.
The loan is established BEFORE canonical C5 stat-bound management JSON,
parent-derived contract creation, signatures and raw parsing, not after a Source
constructor. Preserve original immutable size ceilings and reject growth first.

Exact resource-token operations proposed for NativeActor review:
`update_base(n)` (Dataset custodian only), `admit_local(n)` (Source owner before
its local allocation), `retire_local(actual_read_owner)` (after concrete data/
export retirement). Active replacement calls ONE `parent.admit_external(base+
local)`; Source's old direct parent calls route through this token. Preserve a
separate recorded actual-declared external total when failed cleanup cannot lower
it. Do not subtract the logical retired loan from intrinsic native accounting
unless that lease/counter actually changed.

For parent i, base excludes its OWN parent intrinsic and own local Source loan,
and includes every other retained owner plus Dataset/sink/full child charges.
An active raw export already included in the parent native used counter is NOT
added again as Source intrinsic/local bytes; only additional Python views/copies
are local. Native allocation enforces remaining M-base-local internally;
caller growth is forwarded ONCE through SAME parent owner, not again by a second
Source-to-session path. Counters must report these exclusive quantities. Before
a native operation, the governor also proves its complete planned native growth
fits the joint peak; after measured ownership changes, recompute others' bases
before further allocation. No independent8GiB session budgets.

Closing/failing Source must use this concrete token on EVERY exit, including
failed pre-entry management/contract parsing when no Source object exists. First
retire actual local copies/exports/graphs; then retire ONLY `source_bytes`. Healthy
parent path may call existing checked `admit_external(base)` after retirement.
Do NOT call Source's standalone `admit_external(0)` and erase base. On failed/
expired parents, retain the old conservative reservation and fail the Dataset;
no new nonzero cleanup permission is needed. After ALL associated caller/Source
buffers retire, the Dataset custodian may invoke existing cleanup-only zero.
Native failed caller leases may remain conservative until real native close.
Zero reservation never proves cleanup; keep actual owners/tokens and locks.

If private controller rows/labels must remain after Source closure, the Dataset
governor still owns/charges them independently of closed parents. Prefer explicit
`_PreparedTreeFit` row/label/packing retirement after child/reap and numeric/full-
model validation, before Dataset close; existing lists otherwise retain arrays.
All lowering follows actual retirement, never iterator end/return alone.

New necessary guarded counters: final Source intrinsic ownership excluding
parent/base (and any actual native bytes already counted in parent), and concrete
borrowed export leases with retained/export byte bounds. Snapshot/contract/copy
reservations remain cumulative. Do not expose session objects as constructors.

## 5. Genuine original history and train16/final64 borrowing

Existing native17 cached NN-window token export is INSUFFICIENT when the Tree
aggregate_window differs. Never change its signature/window tag, slice a cached
aggregate, or use Num4 as original history. Root's requirement needs a genuine
original-history projection under the SAME retained parent.

Preferred smallest requested NativeActor seam:

```
parent.borrow_original_training_episode(generation, episode_ordinal, *, budget_loan)
source.borrow_original_selected_episode(ordinal, *, budget_loan)
```

Both yield a concrete readonly `OriginalGameEpisodeBorrow` from original J6/
codec2/own public-event history, after full membership/closure/selected-seat checks.
Its exact data surface: guarded immutable original participant journal/frame/
event bytes, original ordered column payloads, original selected index/Selection,
viewer, observed natural closure/outcome, episode identity, and retained/export
byte bounds. Iteration is fixed original accepted-decision order. No arbitrary
array/envelope constructor supplies this borrow or Dataset rights. Zero episodes
still yield their original closure and zero decisions.

Controller retokenizes these original own facts with the shared corrected
`EventBank`/frame/table/token kernels at EXACT requested Tree encoder window and
real18 positions; then uses `PreparedTreeFrameV3`. This Python feature projection
is not a Source-reader fallback: Rust originals remain validated/owned by the
SAME native17/Native64 owner. Python-recorded originals use the corresponding
actual Python Source owner. Full token signatures/window and selected/full-table
identity are independently checked. NN logprob/value/GAE are diagnostics only.

For Rust, the forwarding must use the actual parent's retained historical raw
custody and final64 slot. If current native17 preservation/export cannot provide
those complete raw facts, this is a REQUIRED bounded native original-history
export addition before Dataset implementation, not an existing API assumption.
An alternative strict native `export_tree_episode(encoder_descriptor)` may
retokenize from its originals internally, but must prove the same full-history
semantics and exact window; Root/Native must select ONE implementation. Neither
alternative widens Actor64's public request/mode or reopens the parent/session.

Dataset then yields v1's frozen `GameSelectedInput` from that actual borrow:
original own frame/event tokens, ordered candidates, selected index/Selection for
identity, observed label and diagnostic parent/cohort/episode/viewer/decision/
policy fields. Those diagnostic fields are never features.

At most one final/native packet/raw borrow is live. Historical training may use
the existing idle next64 owner only when no final packet/Stage is live; retire
actual export/views before advancing. Charge the WHOLE exported native cohort
while leased, even for one-episode projection, plus all retokenization/copy/chunk
scratch. Count actual retained Python records. Keep all three Source owners,
not just the currently projected episode, under the joint governor.

## 6. Pinned real DTOs/correspondence/cost/context

V1's exact manifest/parent/generation/collection/count/preparation fieldsets are
adopted (no additional qualification flags). Root-pinned constants:
`kind=card-instance-game-fixed-sources`,
`source_kind=card-instance-formal-ppo-originals`,
`selection=all-explicit-learner-accepted-natural-return-v3`,
`label_recipe=behavior-selected-natural-return-gamma1-v1`.
Source origin exact fields/values:
`{purpose:qualified_game,synthetic_capacity:false,
outcome_origin:observed-natural-simulator,
qualification_recipe:three-formal-parents-final64-v1}`.

Final64 `original_correspondence` EXACTLY:
`{recipe,input_binding,manifest_sha256,episode_count,episode_digests,binding_sha256}`.
Recipe `card-instance-final-validation-correspondence-v1`, count64, episode_digests
in fixed selected original episode order using existing
`_episode_correspondence_digest`, including empty episodes. Binding SHA is digest
of the other five fields. Same raw-derived proof for Python and Native64;
no synthetic Num4/update/GAE declaration. Training keeps its existing original↔
Num4 correspondence and numeric reference; validation `numeric=null`.

`shared_generation={seconds,cost_key,scope}` remains v1's exact once-only sum of
three complete parent elapsed costs +three canonical final64 C5 elapsed costs,
with ordered original identities/digests. Never fixture construction cost or
reconstructed post-cleanup usage. Per-fit preparation/child/verification/cleanup
costs remain separate and measured.

`TREE3_GAME_RECORDS` is a third identity-checked singleton with SAME Tree3 schema4
features/architecture/numeric kernels, real binding/origin/observed label and
`shared_generation`. Fixture/toy/Tree2 bytes/defaults remain exact. Request4 adds
`source_kind,source_origin,shared_generation,managed_memory` to existing base;
inputs/result/provenance/inference4 keep the v1 closed real source extension.
Completion costs include the exact original shared_generation request receipt.
Child discriminator is closed source kind/origin AND schema, not schema4 alone.
Distinct immutable game-trained origin rejects fixture/toy promotion. SAME fit
workflow accepts only exact active VerifiedGameTreeDataset3, no arbitrary arrays.

## 7. Pinned8GiB/2GiB resource placement

Game-only request `managed_memory` EXACTLY:
`{recipe:'joint-game-tree-fit-v1',total_bytes,controller_reserved_bytes,
child_limit_bytes,reserved_peak_bytes}`; envelopes, NOT RSS.

`total_bytes<=8GiB`. `child_limit_bytes` equals the FULL explicitly configured
model/child limit (default2GiB). Never `min(child,remaining)`, silently downscale,
trim rows/window/chunks or increase total. Before fit Source entry and spawn,
reserve full child cap in the same global governor. Fail if it cannot coexist.

Keep existing `raw`, `5*raw`, fixed/model and packing formula arithmetic. Proposed
controller envelope is all unrelated retained originals/index/caller/current views
PLUS the existing `5*raw+fixed+packing` fit envelope and bounded logs/publication/
copy overlap. `reserved_peak_bytes=controller_reserved_bytes+child_limit_bytes`
must fit total. The five-copy/fixed envelope and full child cap deliberately
CONSERVATIVELY overlap; name that duplicate envelope, never claim reduced RSS.
Old contexts retain existing checks. Local feature/model2GiB checks receive only
relevant active tokens/table/model/chunk workspace, excluding unrelated original
residency which remains charged globally. No collapse of source8GiB into local2.

Same worker validates exact managed_memory arithmetic and child-limit equality
with request execution resources BEFORE inputs/labels/Dataset/Booster allocation.
To know a configurable cap before request JSON decoding, proposed qualified-only
bootstrap handoff is one sanitized decimal resource argument to SAME fit child,
then exact equality with immutable request; old three-argument bootstrap remains
exact. A game request with missing/wrong cap or another context with that argument
refuses. This is resource control, never fit/Source permission. Pin this small
bootstrap detail explicitly before code; do not use the old hardcoded2GiB scan
as an implicit min for a different configured cap.

## 8. Terminal ownership, verification and remaining gates

Keep all custody/original snapshots through preparation, confirmed child reap,
numeric/request/full-Booster/pair checks, final Source/native guards and actual
reportable Source/parent/lock close. Existing `_CompletionPublication` exposes
healthy training completion only afterward; memory-only retirement follows.
Capture immutable metadata/costs while owners live. Preserve primary, guarded
formatting, one-close uncertainty and actual reachable owners; never qualify a
partial or failed build/fit from its metadata alone.

Verification remains controlled primitives/fail-closed public gates only:
manifest flock/no extra inventory, full-set token borrowing, deterministic alias
lock dedup/order/reverse uncertain cleanup; pre-entry management admission for
second/third owners; base surviving final-loan close/failed entry; all committed train/final192
memberships and zero counters via inert views; no serializer on growth; DTO/context/
origin crossover; exact arithmetic/child cap mismatch; Source mutation, cancel,
reap and post-release publication failures. No fabricated complete formal source
fixtures or fake runtime receipts. Genuine three parents/192actual validations/
real Source/fits/comparison/HPO/final test remain user-deferred.

Remaining concrete design gates (not invented implementations):

1. Architecture approval of Parent pre-entry base, concrete source-loan token,
   exclusive counter definitions and SAME-owner genuine original-history borrow.
   Native must confirm raw fact availability or implement/select the strict
   original-history export/retokenization alternative; cached NN tokens do not suffice.
2. NativeActor placement/names/lifetime of private readonly contract and lower
   validator factoring; all existing live/caller APIs and default zero cleanup
   must remain unchanged without the Dataset resource token.
3. Manifest-flock token plumbing and complete custody-set borrow implementation;
   no new native request or inventory exception.
4. Exact qualified child-cap bootstrap argument and resource-envelope validation
   wiring in shared worker. Root-pinned record fieldset itself is settled.
5. Native64 adoption/guarded stream implementation availability. Until then its
   path rejects early with no fallback; no permanent stub counts as migration.

Only this v2 design file was written. No MAIN/staged code/tests/production imports/Source
traversal/model/build execution occurred during the timing quiet window.
