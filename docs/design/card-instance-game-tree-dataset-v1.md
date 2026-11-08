# Qualified game Dataset3 and shared fresh Tree3 fit — bounded proposal v1

Status: READ-ONLY design for Root/architecture review, not implementation approval.
Baseline: accepted Actor64 MAIN `c96252aa42e35a601a0626b42fe583c1f7313e53`; current
`docs/architecture.md` and ADR0024 §18.2. No tests, imports of model runtimes,
Source traversal, build, actual formal parents, games or fits were executed for
this proposal. Only this `/tmp` file is written.

## 1. Scope and smallest ownership split

Implement the guarded production interfaces, not the deferred formal experiment.
A fixture, summary, hash, numeric bundle, native receipt or edited origin cannot
instantiate qualified fitting permission. GameTree3 worker/Journal6/evaluation
routing is the NEXT scope; do not claim it complete here or add a raising stub.

| Owner | Proposed files/responsibility |
|---|---|
| Dataset/fit implementer, after GO | NEW `gf_common/card_instance_game_tree.py`: stdlib closed real-source/label/cost records. NEW `gf_training/card_instance_game_tree_data.py`: real Dataset3 index, six-root custody, aggregate accounting, selected worklist and concrete `VerifiedGameTreeDataset3`. NEW narrow `gf_training/card_instance_source_locks.py`: canonical-manifest flock owner only. Relevant new tests. |
| Dataset/fit implementer, explicit shared-kernel edits | `gf_models/structured_tree_pairing.py`, `structured_tree_artifacts.py`, `gf_training/structured_tree_fit.py`, `structured_tree_worker.py`: one additional closed real Tree3 record context; SAME preparation, child, Booster checks and terminal pair publisher. No training-loop clone. |
| Native/Actor64 owner, coordinated seams only | Existing `card_instance_formal.py`, `card_instance_source.py`, `card_instance_final_validation.py`, `native_card_instance.py` and native wrappers: pre-entry external accounting, actual-parent train/validation borrowing, intrinsic counters and manifest-lock borrowing. This proposal does not edit or assume those private internals. |
| Root | CLI/config/presets/docs/acceptance: explicit `game-dataset build/inspect` and `fit-game-tree`, corrected Tree3 model parsing and existing fresh `TrainingConfig` routing. No ordinary `dataset`, `train`, fixture or legacy default changes. |

Reuse `TreeEncoderConfigV3`, `PreparedTreeFrameV3`, 2810 features at real18,
feature/category signatures, candidate slicing, sharded NumPy bundles and64MiB
array limit. Reuse existing500 rounds/50 early stop/.05/min20/thread defaults,
fresh initialization, `_PreparedTreeFit`, `_prepare_request`, `_TreeFitRun`,
`_OwnedTreeFitChild`, fit worker, full-Booster parameter/text validation,
`TrainingBinding`, pair4 inspect/copy and `_CompletionPublication`.

References: `card_instance_tree_data.py` supplies the existing Dataset owner
method surface; `structured_tree_fit.py:65–193,283–435` is the shared workflow;
`structured_tree_pairing.py:29–108` currently knows ONLY Tree2 and fixture Tree3.

## 2. Actual owners and public interfaces

Proposed public library entrances (names for Root to pin):

```
build_game_tree_dataset(parents: tuple[Path,Path,Path],
                        validations: tuple[Path,Path,Path], output: Path,
                        settings: PreparationConfig, *, encoder: TreeEncoderConfigV3,
                        max_owned_bytes: int = 8*GiB, deadline: float) -> Json
inspect_game_tree_dataset(path: Path, *, expected_digest: str, settings,
                          max_owned_bytes: int = 8*GiB, deadline: float) -> Json
VerifiedGameTreeDataset3(path, expected_digest, settings, *,
                         max_owned_bytes=8*GiB, deadline)
fit_game_dataset(path, *, expected_digest, directory, model: ConfiguredTreeModelV3,
                 training: TrainingConfig, settings, seed, deadline) -> Json
```

`VerifiedGameTreeDataset3` implements the existing concrete-owner surface:
`manifest` (cloned), `source` (SAME private three-parent originals owner),
`selected_inputs()`, `reserve_sink(total)`, `retained_without_sink()`,
`verify_complete()`, `finalize()`, `close()`, `release()` and `cleanup_complete`.
Only this exact active class authorizes `TREE3_GAME_RECORDS` preparation. It
retains exactly THREE actual `FormalGameParent` instances and THREE actual
completed `GameFinalValidationSource` instances, never reconstructed summaries.
One lifetime, one worklist, sticky failure and reachable cleanup remain required.

Build receives six paths directly. Inspect/fit first admit and hash/stat a bounded
Dataset index management read to discover paths; that claimed index supplies no
Source authority. Then resolve the six roots before any raw decoding/output,
acquire all custody, and qualify each original parent and reopen each canonical final64
Source under its SAME parent. Fit reopens originals independently of a past
build/callback; an index alone is never permission. Python-recorded parents use
the explicit Python originals path; Rust-recorded parents require the actual
reviewed native17/Native64 owners. No silent Python fallback or reader identity
rewrite. Until Native64 and the required seams exist, Rust Dataset3 rejects before
Source/output/row/child allocation; this is staged NOT READY, not migration done.

### Homogeneity and complete membership

Require seed slots exactly `[17,29,43]`, three distinct run/original identities,
full pre-G0 declaration1/literal recipe/algorithm SHA, healthy positive normal
completion, all generation3/update3/registry3 records, original Num4 correspondence
and final retained inference4. A healthy positive prefix need not reach1000.
Reject paused/failed/short/ordinary/component/missing/backfilled originals.

Compare the SAME formal variant: complete NN architecture, actions/layout/rules,
input/semantic recipes, learner/inference execution and visibility, actual
numerics/runtime/hardware identities, code/native-source identities and full
inactive Config fields. Normalize ONLY the already-declared seed slot, run/name/
output/path identities; require approved Simulator source/binary/config/spec
hashes to remain equal. Original Config and transitive TOML byte origins are
verified separately PER PARENT. Do not require seed17/29/43 TOML bytes/paths to
match one another; do not invent further exclusion keys.

Frozen work order: parent slot0..2; for each parent, ALL committed training
cohorts generation0..U-1 in order, then its exact final64 validation; within a
cohort use original schedule/group/episode order and accepted decision order.
Train games = `16*sum(U)`; validation games =192. Record every natural episode,
including zero selected endpoints. Each training cohort uses its recorded learner
seat map (also self-play with the same policy in both roles); validation uses its
exact final-policy selected seats. No opponent-private facts, management controls,
GameSeed, rejections, sequence hints or future information enter features.

## 3. Inventory-preserving six-root custody

Use existing parent `run_lock` unchanged. Never create `.lock` in a validation
root or ignore an arbitrary extra file. New `_ValidationManifestLock` opens the
existing canonical `manifest.json` via independently owned no-follow directory/
regular-file descriptors, checks regular type and path/FD device+inode identity,
then takes `flock(LOCK_EX|LOCK_NB)` on the READ-ONLY manifest FD. Recheck identity
after acquisition and retain actual token/inode through Source/fit cleanup.
Pending `manifest.json.tmp` is never accepted. Its lock is RESOURCE custody only.

Resolve nonsymlink roots, group physical aliases by `(st_dev,st_ino)`, choose the
lexicographically first `os.fsencode(resolved_path)` representative and acquire
all unique roots in that byte order. Preserve six logical memberships; physical
lock dedup does not excuse duplicate parent/validation populations. Release in
reverse acquisition order only AFTER all associated native/Python Sources,
exports and fit child have confirmed retirement.

Same-process reuse requires the SAME actual retained lock object/token/inode,
not a copied token string/path/hash. Default entrance acquires all six fresh;
borrowing is allowed only from a complete already-ordered custody set (or a
proven ordered prefix). Reject unsupported partial pre-held sets rather than
invert order/reacquire flock on an independent open description. Root must pin
that conservative borrowing rule. Independent read handles used by validators
MUST NOT close the lock-owner FD. Close retires the active numeric FD before one
`os.close`; reuse `gf_common.owned_completion._close_owned_descriptor`. A thrown
close latches uncertainty: never fstat/unlock/retry that stale number or infer
success from path/registry absence. Retain the actual custody owner and remaining
locks; notes/formatting cannot replace the primary failure.

Required Actor seam: canonical readonly Source opener accepts a concrete
`ValidationManifestOwnership` resource token, verifies its SAME root/inode and
borrows it. This adds no native request field/mode and supplies no Source permission.

## 4. Minimal raw/token borrowing seams — coordinate, do not private-walk

The current parent exposes guarded facts, `canonical_control_sizes`,
`resident_bytes/admit_external`; current final Source exposes facts,
`selected_records/recheck/close/cleanup_complete`. It does NOT expose a public
native session, intrinsic Source counter or streaming borrow. Native confirmed
that idle next64 can coexist with existing historical-cohort export on the SAME
parent, but those APIs need an accepted seam.

Request these narrowly, with immutable facts, one active export and preallocation:

* `FormalGameParent.borrow_training_cohort(generation, *, encoder, resource_loan)`:
  checks membership against protected original generation/update/Source records,
  forwards actual Python original-record handling or SAME native17
  `export_recorded_cohort(index)`, and retires the actual payload before advancing.
* `GameFinalValidationSource.borrow_selected_episode(ordinal, *, encoder,
  resource_loan)` plus guarded intrinsic `resident_bytes`: exact final64 membership,
  actual Python records or SAME native64 selected-episode export, including zero.
* Returned concrete borrow offers an iterator of frozen
  `GameSelectedInput{tokens,candidates,selected_index,original_selection,label,
  parent_index,cohort_index,episode_id,viewer,decision,policy_label}` and
  `selected_tokens()` with original-index/table/frame binding. Diagnostic fields
  never become feature columns. No array/summary constructor grants Dataset rights.

The Rust path reuses actual raw typed-bank decoding/token reconstruction
(`_payload_views` and `_raw_token_arrays`) and native17 original correspondence;
never use saved Num4/GAE arrays as the raw Source or call numeric publication.
Ignore logprob/value/advantage/return arrays as Tree inputs/labels. Charge an ENTIRE
exported cohort/bank while it remains leased, even if iteration is per episode.
Python may retain more original records than native; count those real graphs,
not an assumed metadata-only quantity. Tree window/token features derive afresh
from original own event banks; never substitute an NN cached window.

## 5. Proposed exact real Dataset3 DTOs (Root pin before code)

New constants:

* `source_kind = 'card-instance-formal-ppo-originals'`
* `source_origin = {purpose:'qualified_game', synthetic_capacity:false,
  outcome_origin:'observed-natural-simulator',
  qualification_recipe:'three-formal-parents-final64-v1'}`
* `selection = 'all-explicit-learner-accepted-natural-return-v3'`
* `label_recipe = 'behavior-selected-natural-return-gamma1-v1'`

No `qualified=true` metadata gate. Existing raw Source qualification/not_performed
facts remain unchanged; the concrete Dataset owner supplies fitting authority.
Labels use the existing `journal._reward(outcome, viewer)` natural±1/0 kernel,
repeated at every original accepted own decision (gamma1). Reject unknown or
interrupted outcomes; no fabricated endpoint for an empty episode. This is
OBSERVED outcome provenance, disjoint from declared component terminal labels.

Real manifest exact outer keys:
`schema_version=3,kind='card-instance-game-fixed-sources',selection,label_recipe,
source_kind,source_origin,input_binding,input_recipe,tree_projection,neural_cohort,
parents,collections,counts,source_set_sha256,shared_generation,content_sha256`.
`input_binding` is the exact existing corrected `input_binding_game_v3`;
`tree_projection={encoder,feature_layout,feature_signature,token_signature}` uses
existing Tree3 recipes. `neural_cohort={model,actions_sha256,execution,semantics,
recipe_id,recipe_sha256,algorithm_recipe_sha256,code}` binds the normalized common
formal variant; full per-parent originals remain separately bound.

Parent row exact keys:
`parent_index,directory,run_id,seed,run_sha256,formal_declaration_sha256,
config_sha256,config_inputs_sha256,resume_identity,registry_sha256,
final_generation,final_generation_sha256,final_inference_sha256,generations,
elapsed_seconds`.
Generation row exact keys:
`generation,generation_sha256,update_sha256,inference_sha256,resume_sha256,
artifact_files_sha256` (G0 update null). Include all records, no retained-only filter.
Collection row exact keys:
`parent_index,directory,collection_role,split,generation,manifest_sha256,
source_set_sha256,schedule_sha256,bindings_sha256,original_correspondence,
numeric,selected_roles,counts,elapsed_seconds`.
`numeric` is existing training `{directory,sha256,files}` correspondence only,
NULL for validation. `selected_roles` rows are
`{episode_id,viewer,policy_label,end_kind,outcome,accepted_count}` including zero.
Collection counts `{games,participants,selected_decisions,zero_endpoint_games}`;
outer counts maps exactly train/validation to that shape. Outcome is original
validated JSON, cloned/encoded; no extra label claims. `original_correspondence`
binds the existing train original↔Num4 proof; final uses the actual raw final64
closure/selected correspondence, not a fake numeric packet (exact representation
must be coordinated with Native64).

`source_set_sha256` covers ordered parent immutable references and every
collection membership/role/original correspondence. `content_sha256` covers the
whole manifest except itself. Build emits separate existing-style preparation
receipt `{schema_version:1,kind:'card-instance-game-dataset-preparation',
dataset_content_sha256,seconds,scope}`; it is measured preparation, not Source
permission and not semantic identity.

`shared_generation={seconds,cost_key,scope}`: seconds is sum of THREE original
complete parent `elapsed_seconds` plus THREE original validation C5
`elapsed_seconds`, each once. Fixed scope:
`whole PPO parents plus final validation once; excludes per-fit preparation/fitting`.
Cost key hashes ordered `{run_id,run_sha256,final_generation_sha256}` parents and
`{parent_index,manifest_sha256,source_set_sha256}` validations. No fixture
construction cost, cleanup storage inference, or reconstructed shortened budget.

## 6. ONE8GiB aggregate admission and base-versus-loan custody

`M=8*1024**3` governs the entire retained originals/controller/active export/
private fit rows/packing/pair publication AND declared fit workspace. Define
`R_i` as each parent's INTRINSIC retained ownership (no external reservation),
`V_i` as each validation Source's intrinsic ownership not already in its parent's
native/session counter, `D` Dataset/index/control/copy/guard ownership, `A` current
full episode/cohort/token/export leases and decoding overlap, and `S` all caller/
fit/numeric/packing/child/model/log/publication envelopes. Before EVERY allocation:

`D + sum(R_0..R_2) + sum(V_0..V_2) + A + S <= M`.

Count actual shared references once only where identity proves sharing; do not
sum parent external claims again as intrinsic. Do not give each parent/source8GiB
independently, sample successful-run maxima, cut the corpus, shrink windows/
chunks/thread policy, change arrays or silently raise the total.

The Dataset governor exists BEFORE index or canonical-C5 management parsing.
Admit stat-derived complete JSON/decode/copy/scratch bounds through the same M
before those reads, including index/open pre-entry scans. After locks, bind every
claimed reference back to actual originals; a small metadata reader is not an
independent full allowance. Before second/third Parent/Source entry, all previously
retained Python/native/index/source charges are already in that caller base.

Pre-entry gap: `FormalGameParent` currently allocates before a caller can invoke
`admit_external`. Request ctor resource-only `extra_owned_bytes` (or equivalent
closed pre-entry account), applied to scans/native/clone admission BEFORE entry
parsing. For parent i, pass all previously held owners/index plus complete next
operation scratch as external under the SAME M; update other parents before
further growth. This is a remaining-slice computation, not a smaller global cap.

Final Source gap: current `close()` releases `parent.admit_external(0)`, which
would erase other-parent/sink charges. Request one concrete resource-only
`FinalSourceBudgetLoan` token tied to SAME parent and Dataset custodian, with
separate `base_bytes` (other owners/index/sink) and `source_bytes` (this loan).
No global permission registry/callback/serialized flag. Source uses its token to
admit `base+source` BEFORE canonical-open management parse, Config/plan/signature
copies, raw decoding and later getters. Sizing uses protected encoded lengths,
existing size-only traversal/immutable ceilings; repeated mutable growth refuses
before serializers. The bare standalone path keeps its existing zero cleanup.

A healthy Source closes/retires its actual graphs/exports first, then the token
may lower ONLY its source component through existing checked
`parent.admit_external(base)`; base remains owned. On failed/expired parents,
keep the prior reservation conservatively charged rather than calling nonzero
cleanup or clearing base. Dataset fails and retains actual owners; after ALL
associated Sources/caller buffers are retired, the Dataset custodian performs
cleanup-only `admit_external(0)`. Native leases may remain conservatively charged
until actual close. A zero counter is never cleanup confirmation. Sink/fit arrays
still retained after Source close stay charged to the Dataset governor until
actually retired; they must not disappear merely because a parent closes.

Required counter/borrow seams: Source intrinsic bytes and resource-loan retirement
must exclude base/parent charges precisely, and Parent pre-entry must include base
before any parse. Do not infer these by walking private scans/session summaries.
All three Sources stay retained, but only one raw export/token view is active.
Use existing fit envelope arithmetic `5*raw + fixed + packing`; include ALL
originals as retained bytes and all current child/packing/publication overlap.
Scratch is separately cumulative under configured scratch limits; never count
originals as disposable scratch or assume compression/hardlinks release bytes.

Resource placement requiring Root pin: current fit compares the whole envelope
against model resources (default2GiB), and the child hardcodes a2GiB scan ceiling.
Do NOT silently turn the required8GiB aggregate into that local2GiB or permit an
independent8GiB child. Preferred new-game-only handoff records
`managed_memory={recipe:'joint-game-tree-fit-v1',total_bytes:8GiB,
controller_reserved_bytes,child_limit_bytes,peak_bytes}` in the immutable request;
reuse exact formula/counters to derive a child slice under remaining total and
its explicitly configured own limit, enforce it before child decoding/Dataset/
Booster allocation, and validate its arithmetic in the worker. Old context limits
and defaults stay exact. Root must select the qualified context's explicit model/
child resource profile and whether this closed field is needed; no implementation
may guess a scope split or reduce configured rows to pass admission.

## 7. Closed fit/artifact context, same workflow and terminal boundary

Add singleton `TREE3_GAME_RECORDS` beside existing two identity-checked contexts.
It shares Tree3 features/schema4/architecture parser but selects exact real
binding/origin, observed label and `shared_generation`; fixture/toy context stays
unchanged. Replace only fixture-specific `context.modern` source/cost branches
with closed context properties. `_PreparedTreeFit` accepts exact
`VerifiedGameTreeDataset3` for this context and no arrays/callback/summary.

Request4 keeps existing `REQUEST_FIELDS` plus exact
`source_kind,source_origin,shared_generation` (and the Root-pinned resource field).
Input4/result4 retain their current fields plus `source_kind,source_origin`.
Completion4 keeps `COMPLETION_FIELDS` plus those source fields; costs exact
`{preparation_seconds,child_seconds,final_validation_seconds,cleanup_seconds,
elapsed_seconds,shared_generation}`, with shared receipt canonically equal to
request. Provenance4 keeps `PROVENANCE_FIELDS` plus source fields, same
`kind='structured-fixed-dataset3-fit'`; source kind/origin disambiguates variants.
Inference4 retains existing complete fields/layout/runtime/model text/native
parameters and source fields; its semantics label uses observed natural recipe.
Introduce a distinct immutable game-trained origin DTO and context-aware parse
helper, since `TreeOriginV3` currently hardcodes fixture classification; no game
toy origin or relabeling of a trained fixture is permitted.

Child discriminator is schema AND closed source kind/origin, not schema4 alone.
Keep native fresh fit, numerical operations, model-envelope/full-Booster checks
and pair publication identical. The child receives only its immutable own
request/numeric inputs/resource plan; no raw Source paths/private views/Parent
handles. Add explicit game inspect/load/pair wrappers over the same kernels;
qualified game worker/factory remains next scope.

Capture immutable Dataset/provenance/cost metadata while owners are live. After
confirmed child consumption/reap and numeric/full-model checks, explicitly retire
controller private row/label arrays and packing views when no longer needed;
current `_PreparedTreeFit.blocks/labels` otherwise keep them alive. Do not release
their sink/base claim merely because Dataset Source iteration ended. If buffers
must survive to completion, the Dataset governor must continue owning/charging
them independently of closed parents until memory-only retirement.

Keep six locks/original snapshots through row preparation, child confirmed reap,
request/numeric/full-model/pair verification, terminal original sweeps and
reportable Source/native/lock cleanup. All such cleanup precedes healthy
`training.json` via SAME `_CompletionPublication`; after commit release is
memory-only. Any uncertainty preserves primary plus actual owners and prevents
completion. Reuse protected diagnostic formatting and one-attempt descriptor
helpers; no late healthy index/fit status may authorize data by itself.

## 8. Verification and unresolved decisions

Controlled evidence only: pure exact DTO/context cross-rejection, all-state
homogeneity normalizer, per-parent Config-origin checking, artificial membership
counters/zero episodes, actual temporary flock/token/inode ordering/physical
dedup, overlapping/foreign/closed token refusal, manifest replacement, no added
inventory file, reverse close and FD reuse. Logical near-cap owners must refuse
before second/third Parent/Source parse, token decoding, signatures/JSON, row/
child allocation; test base survives Source close, sink retirement, mutable growth
and combined parent/native accounting. Child/cancel/publication doubles must
prove original-owner retention and no healthy completion after late failure.
No fabricated complete formal parents, runtime receipts or Source-positive fits.
Public short/fixture/ordinary/missing-source gates must reject before output/
optional Booster allocation. Preserve Tree2 and fixture record bytes, default
settings, worker arrays and terminal publication regressions. Genuine3parent,
192actual validation games, Source qualification, real fits/comparison/HPO/test
remain user-deferred, not simulated successes.

Root/architecture must pin before code: (1) manifest-flock mechanism and complete
ordered-set borrowing rule; (2) exact proposed real strings/DTO fields and Native64
final correspondence representation; (3) pre-entry/base-versus-loan budget and
streaming API ownership; (4) qualified fit's explicit8GiB/child resource placement
and immutable resource handoff; (5) CLI verbs/preset routing. Native Actor64 and
required guarded raw/token seams are missing production dependencies, not a
reason to treat a permanent raising placeholder as completed integration.
