# 0023: 構造化encoderと予算付き手札探索の共同導入

- Status: Accepted for independent structured encoder/version work (2026-11-01).
  Full integrated completion and unresolved utility rows remain blocked on issue0005.
  Root owns design and acceptance.
- Source: user's integrated implementation plan, 2026-11-01.
- Review: bot_architecture_reviewer READY for §§1–9/11/12 and the confirmed public
  boundary; BLOCKED only for issue0005/full completion. No tests were run by that
  design review. Final correctness/conformance and implementation acceptance pending.

## 1. Context and completion boundary

ADR0022's MLP/GRU/Transformer PPO and normal-boundary resume are implemented.
Their hash-aggregate inputs and three-element candidate space cannot express the
requested detailed hand, visible action intervals and longer diverse proposals.
This ADR adds an explicitly versioned path while retaining the existing Config,
feature signatures, datasets, checkpoints and historical evidence unchanged.

One completion unit covers public candidate generation/pre-count/admission,
structured input, configurable trainable encoders, all three PPO models,
deterministic LightGBM features, worker/journal/numeric/artifact integration,
CPU/CUDA inference, safe persistence, resource admission and verification.
HPO, final strength comparisons, formal parent campaigns and final-test evaluation
are excluded. Synthetic fits never authorize real-data fitting or quality claims.
No new third-party dependency and no Simulator change are planned.

Simulator pin remains `ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6`, HTTP v1,
Observation `"1"`, typed Action schema1, `true-match-1v1-v1`, catalog schema1
`public-ja-v2`, hash `481ddfb0c068a1d2aaaf2dbcb6a5b4bd04396dfcc5c995e14e47b71222b6a51d`.
The reviewed document hashes in `gf_common.domain.REVIEWED_SPEC_HASHES` remain
the declarations; actual checkout/spec/binary identities are recorded separately.
Normative authority: Simulator spec07 §§12.1,13,21.1, spec05, spec09 §§9–18,
and accepted ADR0005/0007/0009. Engine/internal support/SequenceOptions/rejection
probes are forbidden as candidate or policy authority.

## 2. Responsibilities and dependency direction

- `gf_common`: closed v2 action settings, public coverage/completion/payment rules,
  pre-count and bounded search, exact reference tables, participant-local event
  bank/frame/table contracts, deterministic vocabulary/token semantics. No Torch,
  HTTP, learning target or parent management dependency.
- `gf_simulator_client`: protocol2 actor-local deltas, parent frame/table/index
  validation, Action reconstruction against the original own Observation, process
  lifecycle and inference-device reservations. Simulator I/O remains separate.
- `gf_models`: closed v2 model Config; deterministic model-neutral tokenization
  reference and LightGBM feature recipe; lazy Torch trainable state/card/hand/
  event/temporal encoders, factorized actor and candidate-independent critic;
  participant-local inference and safe inference manifests.
- `gf_training`: schema5 source validation/projection, schema3 numeric rollout,
  source/target/split authorization, E2E PPO/TBPTT, optimizer/RNG resume, fixed
  dataset manifest2, publication/deadlines. Reuse existing run and collection
  ownership; do not introduce a second generic trainer.
- `gf_native`: deterministic validation/tokenization and compact staging for the
  new recipe. Never owns learned embeddings/latents, legality, rewards or PPO.
  Its capability must be checked before collection with a native recipe.
- `gf_cli`: version dispatch, presets, safe runtime bindings, resource ledger,
  evaluation/performance composition. Root owns Config integration/presets/docs.

Common imports no upper layer. Models import no training or transport layer.
Native imports no Python application package. One code-writing implementer owns
assigned source/tests/native files at a time; root does not edit those concurrently.

## 3. Version and compatibility dispatch

New identifiers are `model.version="2"`, `actions.version="2"`, generator
`public-atomic-diverse-v4`, input recipe `participant-structured-v1`, worker
protocol2, journal/derived schema5, numeric rollout3, neural inference/resume3,
LightGBM inference/training3 and fixed dataset manifest2. The authorized policy
input version is unchanged. Schema/config dispatch is explicit and closed.

Legacy dataclass serialization/defaults and hashes must remain byte/meaning
compatible. Use distinct v2 Config/records or equivalent closed dispatch; do not
add new defaults to old serialized objects. Never replace the global v3 generator
constant used by old artifacts with v4. Select semantic identities from the
actual recipe. Old unsupported mixes fail before collection/fit/forward.
New recipes do not reinterpret frozen old datasets. Ordinary old-version resume
retains its existing exact code/environment/config guard.

Inference device/chunk/resource binding is separate from network architecture
identity for new artifacts. Record the actual binding in evaluation and parent
resume identity; changing it during ordinary resume rejects. Legacy artifacts
retain their CPU route. Random/heuristic can use actions2 without a neural model;
LightGBM2 uses the new deterministic features with categorical ID columns.

## 4. Participant records and information boundary

All records are validated immutable snapshots (or private owned buffers exposed
read-only) bound to participant and episode. No global cross-viewer cache.

| Record | Contract |
| --- | --- |
| Event bank | Append-only visible raw events, actor/episode, contiguous start/end offsets and prefix hash. Reject gaps, reorder, altered prefix, foreign viewer, cap overflow. |
| Decision frame | Original own Observation, typed metadata, same-revision hints, own perceived catalog definitions, all owned cards/learned miracles, public opponent learned ID list, interval and total event boundaries, actor/revision/frame hash. |
| Candidate table | Frame hash; compact action kind/target/numeric columns, sequence offsets and ordered own card-reference indexes, stable order and digest; no card-definition duplication per candidate. |
| Selection | Table digest and index, policy identity/logprob/value. Parent validates and reconstructs the typed Action from the original own Observation before Act/journal acknowledgement. |

`DecisionInput.events` keeps its old meaning (complete visible history).
Protocol2 deltas use a separate bank/view path; they must never masquerade as
legacy DecisionInput.events. First interval includes all visible events before
the first own decision; later intervals begin after the preceding own decision.
No retroactive Dream/perception rewrite. Raw names/effect text are retained but
are not language-model inputs. Public observed card events remain past facts,
never a claim of current opponent possession. No opponent hand or count input.
Opponent's published learned IDs are categories, not catalog lookups for extra
definitions; catalog interpretation remains limited by spec07 §21.1.

Hand plus learned list has maximum18 slots independent of current usability.
Use visible deterministic hand ordering and explicit source/learned-order fields.
Instance IDs resolve references only; never feed their numeric magnitude into a
model. Exact version-pinned definition/ability/attribute/kind/status/event
vocabularies have explicit padding/missing categories and reject unknown values.
No 32-bucket hashing in the new path. Vocabularies identify categories, without
supplying unobserved catalog values. Preserve missingness and signed numerics.

## 5. Closed model configuration and numerical recipe

Config is stdlib-only and validated before model/array construction. Initial
values below are implementation baselines, not tuned hyperparameters.

| Component | Settings / defaults |
| --- | --- |
| State | identity or MLP; hidden widths/layers, output width, activation; default hidden128/output64/ReLU |
| Card | definition embedding32, ability embedding16, shared card width64 |
| Hand | direct/deepsets/attention; latent128; shared MLP widths; sum/mean/max pooling |
| Hand attention | 2 layers,4 heads, FFN128, ReLU; no arbitrary slot position encoding |
| Event interval | GRU width32,1 layer; reset at each interval |
| Temporal | MLP current frame, or own-decision GRU/Transformer; configurable width/layers/heads/FFN, default128 recipe from ADR0022 |
| Heads | independent actor/value hidden widths/layers, default128 baseline |
| History | aggregate128, max decisions1024, TBPTT32; event interval4096/episode262144 |
| Inference | cpu or cuda:0; candidate chunk256; explicit resource limits |

Unused-kind configuration fields must be absent or validated as the documented
canonical inactive value; nondefault ignored values reject. Validate strict
integer types, shapes, attention divisibility, vocabulary sizes, parameter counts,
per-tensor/shard limits, complete model/optimizer/runtime envelopes before network
construction. Activation initially ReLU; dropout0, IEEE FP32, AMP/TF32 disabled,
deterministic backends and orthogonal initialization. Preserve established1e-5
parity tolerances; do not relax them to make a route pass.

Direct hand input is a fixed visible-order concatenation of exact one-hot
categories, numerics, missingness, membership and learned order; no learned hand
aggregation. Shared card embeddings still train on the candidate branch.
DeepSets maps shared cards with a shared MLP then masked sum/mean/max to hand
latent. Attention self-attends shared cards then uses that aggregation path.
Empty/masked hands/events have defined zero representations without NaNs.
Hand latent is consumed directly downstream; combinations are not an encoder
input requirement and no action decoder is added.

Each frame representation includes detailed current state/hand, current interval
GRU output, and the bounded128-event aggregate. MLP uses the current frame;
GRU/Transformer process own-decision frame history, capped at1024. All historical
raw/token state/cards/events are retained with their observation-time semantics.
Policy-visible facts and train-only targets remain in different fields/owners.

For E2E TBPTT, recompute prefix frames with current encoder weights under no_grad;
detach temporal hidden/KV, encode suffix frames with gradients through state,
hand, shared cards and interval GRU. Endpoint loss once, episode GAE and batch
advantage normalization retain ADR0022 semantics. No saved learned latent is an
authoritative rollout input or reused across optimizer updates. The value branch
receives no current candidate content/count/order, search cutoff or selection tags.

## 6. Candidate space, pre-count and selection

The public coverage matrix in section10 must be independently reviewed before
generator implementation. Completion is separate from extensibility: a support
prefix can extend but is not submittable alone. Own hand references never repeat;
learned repetitions obey explicit public per-occurrence rules and payment.
Use actor same-revision atomic hints per purpose without treating their Cartesian
product as legal combinations. Keep documented unverified combinations omitted.

| Budget | Default / allowed bounds |
| --- | --- |
| max sequence length |18;1..128; model position capacity must be >= search length |
| presented candidates |32768; maximum65536 |
| expansion attempts |262144 |
| temporary complete candidates |262144 |
| pre-count |50ms and16384 count states |
| search/selection |1 second |
| scoring chunk |256 |
| channels |fixed seven positive configurable weights, default all1 |

Estimate before constructing Actions: completed counts, prefixes, failed/pruned
attempts, targets, numeric arguments, total sequence elements, tokenization,
scoring and storage bytes. Simple domains use saturating math.perm/math.comb;
complex domains use public upper bounds or budgeted counting. Saturate beyond
admission thresholds; counting must not generate every Action. Record whether
bound/exact/unknown. Full enumeration is admitted only if every relevant budget
fits. Lazily enumerate accepted small spaces; otherwise bounded interleaved
family/start/length-band search must reach long sequences. Only actual completed
traversal records exhaustive coverage. Every attempted extension/prune and
numeric supplement consumes its applicable budget. Preserve termination reasons.

Basic actions and every distinct completed single-card action get mandatory
coverage (numeric actions retain the carrier, not all allocations). If mandatory
coverage cannot fit, fail explicitly. Remaining unique proposals are selected by
weighted round robin over seven channels: attribute/effect, low/mid/high attack,
resource consumption, hand refresh, remaining-hand differences, sequence length,
exploratory complement. Minimum quotas apply to channels, not every tag/mask.
Use endpoints/median breadth for numeric axes and cycle discovered categorical
groups. Reallocate empty quotas, never duplicate an Action via overlapping tags.
Exploration RNG is independent of policy sampling and its seed/state/order and
cutoff reasons are persisted. Wall-time cutoffs can change discovered/selected
proposals despite equal seeds; actual stored tables/cutoff evidence, not seed-only
regeneration, are authoritative for replay.

Mandatory singleton coverage includes each configured/publicly supported target
variant. Exchange reserves at least one concrete range-valid conserved allocation
per eligible carrier; a carrier-only table row is not an Action. Sell consumes
two references and is not a singleton. Support-only elements reserve no slot
until a completed action is formed. Attack completion tracks an attack-bearing
element independently of positive attack magnitude, including zero-valued Stick
with paid Aura/Mirage. This slice admits supported Aura/Mirage after an established
attack-bearing prefix; no newly inferred support-before-first-attack orderings.

Exchange uses a25-interval conditional two-axis grid under three resource
orderings, deduplicated; add current/balanced/HP1/public own miracle-MP-cost
neighborhood allocations. All supplements count. Preserve conservation/ranges,
HP0 public legality and separation between one carrier and all numeric arguments.

Actor first linear layer is one dense parameterization split algebraically into
shared context, position-by-card projections and numeric arguments. Precompute
each position/card projection once and gather/sum ordered references per chunk;
apply the same nonlinear head. Dense reference has identical parameters and
initialization. Chunk recomputation uses Torch nonreentrant checkpoint explicitly
`use_reentrant=False`, without mutable closure state/device migration. Concatenate
logits into one global softmax/logprob/entropy and preserve gradients through all
candidates. Never normalize separately by chunk.
Reference: https://docs.pytorch.org/docs/2.14/checkpoint.html (checked2026-11-01).

## 7. Persistence and failure ownership

Schema5 stores event-bank appends once and frame/table references with actor,
revision, offsets, lengths and digests; projection validates every link before
yielding accepted selected-action transitions. Native and Python routes must
agree on token/category/order bytes, selected targets and failure classification.
New native capability identifies both schema5 and participant-structured-v1;
missing capability rejects before collection, without Python fallback.

Numeric rollout3 retains raw deterministic tokens and banks, compact candidates,
endpoint mapping/selection/old policy values/GAE and closed semantic identities.
No repeated prefix/frame/card arrays per candidate and no learned latent storage.
No pickle. Each decoded shard total<=64MiB; manifest lists multiple hashed typed
shards and exact key/shape/dtype ownership. Reject a single tensor above64MiB
before allocation. New optimizer persistence uses stable parameter names plus
per-parameter step/moment presence; unused parameters remain uninitialized after
resume until first gradient. Resume includes all encoder weights, optimizer,
CPU/CUDA/shuffle/search/schedule RNG and binding identities. Reuse normal committed
batch-boundary resume and existing locks/deadlines; no mid-episode resume.

LightGBM3 features contain detailed current state, every own held/learned card,
ordered candidate references, public opponent learned IDs and fixed visible
history aggregate. Exact definition/ability/ID columns are declared categorical.
No frozen neural encoder. Fixed dataset2 still requires all three complete PPO
parents and prescribed final-policy validation sources; legacy inclusive fitting
retains its pilot/source/decision-budget gate. Old datasets are not relabeled.

## 8. Device, resource and timing ownership

Maintain one clean interpreter per participant, including CUDA. No shared service
that sees both players. Coordinator does not import Torch. A parent resource
ledger admits up to four participant workers plus resident learner against
default12GiB managed CUDA budget. Do not overlap updating and collection.
Before allocation estimate resident weights/gradients/Adam, model/cache/context,
CUDA context reserve/workspace, hand attention, interval GRU, history/prefix
recomputation, candidate chunk/checkpoint replay/global normalization and transient
CPU/device copies. Saturate overflow and reject a single endpoint that cannot
fit; never auto-change W/candidate limits/device. Managed bounds are soft
estimates; measured allocated/reserved/RSS remain separate.

On CUDA, synchronize and check finite outputs before probability transfer to
CPU, then sample with a CPU Generator. CPU/CUDA identical action sequences are
not promised. Worker timeout remains10s. Timeout/OOM/asynchronous failure kills
and reaps the owned worker before releasing its reservation; preserve errors and
do not publish an accepted transition/update for interrupted work.

Acceptance CPU p95<=3s for ordinary decisions includes candidate generation,
conversion, scoring, sampling and parent/worker communication. Model startup and
Simulator communication are separate. Freeze512 eligible train-side decisions,
two concurrent games, fixed model/config/runtime and record identities before
measurement. Report long-history/long-sequence/tens-of-thousands candidate stress
separately. GPU reports host transfer/forward/sync/return separately. Do not
predict speed differences. Move only measured bottlenecks to existing Rust if
necessary and reverify candidate order/content/features.

## 9. Verification and acceptance

Implementation proceeds in dependent slices but this ADR remains one completion
unit: (A) contracts/coverage/Config; (B) candidates and banks/tables; (C) encoders,
PPO/LightGBM and sharded persistence; (D) integration/device/resources/performance;
(E) independent final correctness/conformance and root acceptance/local commit.

Required evidence: small-space reference enumeration; mandatory/singletons,
18/128 elements, repeats/order/attributes/CutCost/dynamic payment/defense/exchange;
instance mapping/learned order/Fog/Dream/actor/future isolation; all3 hand modes
and all3 PPO temporal paths, empty/mask/candidate permutation/critic independence;
dense/chunk logits and gradients, encoder gradients, CPU/CUDA cache/full/split;
all encoder/optimizer/RNG boundary resume and incompatible-Config rejection;
old format/data regressions; Python/native differential and capability preflight;
failure/reap accounting; full pytest/Ruff/mypy/Rust/isolated wheel/optional-free
baseline startup. Check actual installed binary/build identity, not checkout alone.

Then fresh192-game baseline validation. Default attention: each PPO model crossed
with CPU/CUDA inference, continuous2 updates versus pause after1/update+resume,
16games/batch and original7200/5400/7140s budgets. Other hand modes: real MLP games
plus all-model numeric coverage. Freeze512-decision ordinary CPU panel p95 target;
stress/resources measured separately. Preserve all failed evidence and exclude
failed parents as a whole. Performance/contract shortfalls return to root; do not
shrink candidate/input/validation scope or loosen tolerances silently.

Design/final review findings and exact command PASS/FAIL/NOT RUN outcomes are
recorded in `docs/verification/structured-encoders-and-search.md`. Acceptance is
after blocking findings resolve and required runnable checks pass; environmental
limitations must remain explicit rather than being called PASS.

## 10. Public action coverage matrix (review required)

The following table was assembled by a read-only verifier from normative public
specifications and the pinned catalog. It still requires independent design
review; a supported row is a proposal domain, never an engine legality guarantee.

| Family / capability | Completion, target, repetition/payment | Public evidence / omissions |
| --- | --- | --- |
| Pray | Complete basic when actor can_pray; full possession allowed |09 §10.1; required atomic mode uses can_pray |
| Allow / purchase choice | Empty response when can_allow; both bool buy choices |09 §14/17.3; ADR0007 |
| Normal/extra attack | One normal base first; extra attack can stand alone/follow; hand refs unique; learned repeat only established extra/support rules |09 §9.4/16.1–16.3; preserve ordered effects/attributes (§15.2) |
| Magical Stick | Complete at MP0; pay actual miracle costs first, then all residual MP; self/opponent, omitted/null defaults opponent |09 §13.1.1; CutCost does not waive weapon payment |
| Wand | Attack-bearing additional weapon, complete singleton and mandatory; colors only following additional attacks, with Mirage reset override |09 §9.2/9.4/16.2 and register GF-ACT-005; confirmed project rule takes precedence over explicitly archived conflicting wording in03/08 |
| Aura / Mirage | Support needs completed attack; Aura amplifies accumulated attack, Mirage resets attribute/Wand and repeats rounds; Mirage targets opponent except special Pestle |09 §9.4/13.3/15.2/16.2; omit Mirage+attackTwice and MoonGod-alone Mirage explicitly |
| CutCost weapon | Spiritual Staff has complete ordinary attack; simultaneous miracles cost0 |03 §16.1,09 §13.1.1; pure utility still requires miracle; nonattack combinations need missing contract |
| Named hostile status miracles | Wind/Heaven Wind/Fog/Dream/Dark Cloud; explicit self and hostile targets, printed MP unless supported CutCost |09 §16.3–16.4; no inferred ring/reflect legality for their zero-damage response |
| Numeric/ability defense and rings | First classify public completion (ordinary numeric defense/ring/stop/reflect/bounce vs support-only), then gate components by response hints; Flash one element; armor+ring allowed; mirror/reflect/bounce mutually exclusive with armor/ring/each other; learned defensive miracle costs apply |03 §10.1,09 §8/14/15.1/16.2–16.4; hints alone certify neither singleton completion nor cardinality; unknown/pure CutCost support awaits issue0005 |
| Discard / Sacrifice | Nonempty nonweapon hand subset; no instance repetition; normal Discard no refill, Sacrifice refill |09 §11; retain ADR0005 Sun Amulet/Dangerous Mortar exclusion only for normal Discard |
| Exchange | One carrier, conserved total, each0..99, HP0 allowed, self only |09 §17.1; conditional grid and supplements in §6 |
| Sell / Buy | Sell exact two distinct hand refs; Buy one carrier; explicit self/opponent, separate target response and buyer bool choice |09 §17.2–17.3; no hidden offer features |
| Recovery / cleanse / guardian singletons | Public effect semantics available, direct-use target/default/completion/repetition not sufficiently published |Open [issue0005](../issues/0005-public-utility-action-semantics.md); requested scope must not be silently reduced |

Issue0005 is a blocking specification dependency for those generator rows and
full acceptance, not for independent structured encoder/version work. Root must
adopt additional public semantics before authorizing their implementation.

Review disposition: inherited accepted v3 proposal semantics permit repeated
learned additional-attack/Aura occurrences with per-occurrence payment; this is
not a new claim of independent per-definition legality proof. Root adopts bounded
Mirage repetition from09 §16.2's multiple-occurrence rule plus that same learned
reference/payment convention, excluding attackTwice. Do not extend this convention
to normal-base/status/defensive/utility miracles. Existing ordinary attack target
proposals retain their configured target/hint intersection without claiming that
the global target array proves each card×target product; new Mirage restricts to
its explicit opponent contract. The engine remains final validation authority.

## 11. Alternatives, consequences and open questions

Retaining hashed aggregate inputs would lose exact card identity and interaction
information requested here. Frozen neural features would not provide E2E updates.
A decoder or engine legality oracle is unnecessary and outside the information
contract. Enumerating first then checking budgets cannot bound preparation cost.
Separate new version paths cost maintenance but preserve historical meaning.

Open before dependent implementation: exact public target/completion coverage in
section10; any newly discovered mismatch between schema5 and current source
authorization must return to root. GPU availability under the execution sandbox
requires an actual runtime check. No unverified legality assumption or automatic
CPU fallback resolves these questions. OS/I/O soft-budget debt from preceding
ADRs remains, with revisit when hard quotas/durable crash recovery are required.

## 12. Concrete v2 interfaces and review clarifications

The v2 route uses these small closed contracts, without changing the legacy
`Policy`, `CandidateGenerator` or meaning of `DecisionInput`:

```text
EventBank.append(EventDelta(actor, episode, start, previous_hash, events))
    -> EventBoundary(end, prefix_hash)
DecisionFrame.from_input(complete DecisionInput, EventBank, interval_start)
    -> immutable frame bound to actor/episode/revision/event boundaries
PublicCandidateGeneratorV2.generate(frame, bank_view, search_rng)
    -> CandidateTable
CandidateTable.to_action(index, original_frame) -> typed Action
PolicyV2.reset(sampling_seed)
PolicyV2.select_frame(frame, bank_view, table) -> Selection
tokenize_frame(frame, bank_view, recipe) -> deterministic FrameTokens
tokenize_table(table, frame, recipe) -> deterministic CandidateTokens
```

V2 runtime policy options are a distinct closed safe binding of checkpoint path,
expected manifest digest and inference device/chunk/reservation settings (plus
explicit toy permission where already allowed). They never carry effective_config,
parent paths/management or Simulator seeds. Keep legacy validated_policy_options
serialization unchanged. The new binding hash belongs to evaluation/resume
provenance, while checkpoint architecture semantics exclude runtime binding.

The wire protocol carries a frame plus explicit EventDelta; it does not send a
delta as complete DecisionInput.events. Frame raw payload consists of observation,
typed metadata, own catalog/action_options/history_end and bank/interval binding.
Frame hash covers that payload and actor/episode/revision. Table canonical bytes
use closed fields for frame_hash, action kind, relative target (omitted/null/self/
opponent remain distinguishable), numeric arguments/presence, offsets and ordered
source/slot references; hash covers exact order and contents, not timing metrics.
Selection contains table_hash/index/policy_version/logprob/value. Parent checks
every reference and reconstructs against its original bound frame. Policy receives
only this participant's bank prefix through the current frame boundary.

`FrameTokens` has fixed-ordered state numerics/categories/missingness,18 card slots
(exact definition/ability/kind/attribute, numeric/missing, source, usable, learned
order/presence), opponent public learned ID slots, ordered interval event tokens
and aggregate history. `CandidateTokens` has kinds/target/presence/numerics plus
offsets/ordered card-slot references. Padding is separate from valid category0.
The model-neutral reference defines exact field ordering/vocab and its signature;
native reproduces it byte-for-byte. Every model2 consumes these fields. Instance
ID numeric values, management identity hashes and search diagnostics are excluded.

Dense actor reference concatenates context, candidate kind/relative target,
numeric/presence, and each position's card representation/source/slot presence/
learned-order semantics. Factorization partitions that same first weight matrix
into context, categorical/numeric and position-by-card blocks; it does not drop
the action kind or target. Both paths use the same parameter initialization/order.
V2 admission counts chunk activation slots, retained all-candidate logits/global
normalization and compact references. It does not inherit legacy4096 total-candidate
microbatch admission, which would reject the new32768 default before chunking.
The legacy limit and its recipe are unchanged.

Artifact3 inference/resume manifests enumerate hashed typed shard files and
stable tensor keys. New generation record schema2 has an explicit artifact_files
hash map and references those manifests; do not redefine old weights_sha256 to
mean a collection of shards. Existing generation schema1 retains its fixed-file
hashing/retention logic. New source/native compact authorization summaries carry
model_id, model_version, input recipe and generation schema; model_id-only native
capability is insufficient for v2. Retention validates the whole closed file map,
retains inference identity manifests, and deletes only enumerated owned shards.
Malformed/mixed/deleted/missing shard states fail closed before use.

Fixed dataset2 requires a homogeneous model2 cohort at ordered training seeds
17/29/43 (same model kind/architecture/encoder/action recipe, apart from permitted
seed/run identities), three healthy complete parents and each parent's prescribed
final-policy validation sources. MLP/GRU/Transformer are supported cohort kinds;
do not mix kinds or call the three kinds the three required seed parents. Old
dataset1 retains its MLP-v1 cohort contract. No new formal parents/fits are
required to claim implementation; eligibility is tested without relabeling old
parents as model2 evidence. Real fitting remains gated.

Resource estimates explicitly include parameter-shaped FP32 weight/gradient/
Adam moment bytes, per-parameter steps, CPU reference copies, configured caches,
attention/GRU/history tensors, checkpoint replay, chunk/full-normalization and
CPU/device packing. CUDA context/workspace reserve is explicit per admitted
process in addition to tensor bytes, and peak observations must test its adequacy.
Collection retains a resident learner reservation but never overlaps its update
with workers. Available physical memory and managed12GiB admission are distinct;
no outside process is preempted. Root runtime check2026-11-01 confirms pinned
Torch2.14.1+cu132 sees RTX5070Ti when GPU permission is available; sandbox-only
Torch reports unavailable and must fail without silently choosing CPU.
