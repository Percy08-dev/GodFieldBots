# 0023: 構造化encoderと予算付き手札探索の共同導入

- Status: Accepted for independent structured encoder/version work (2026-11-01).
  Full integrated completion and unresolved utility rows remain blocked on issue0005.
  Root owns design and acceptance.
- Source: user's integrated implementation plan, 2026-11-01.
- Review: bot_architecture_reviewer READY for §§1–9/11/12 and the confirmed public
  boundary; BLOCKED only for issue0005/full completion. No tests were run by that
  design review. Full integrated correctness/conformance and acceptance remain pending.

## Implementation status

LATEST OWNER STEERING: [2026-11-01 confirmation](../contracts/owner-20261101-utility-and-repetition.md)
disallows repeated physical/learned references within one Action and confirms
utility singleton/explicit-target assertions, while leaving several boundaries
unresolved and public Simulator specs unchanged. Earlier B readiness below is
historical component evidence under the previous proposal assumptions; it is
not acceptance of the corrected rule domain. Advisor B and architecture review
approved a separately identified engineering-only profile; its implementation
and current-boundary metadata propagation are now reviewed READY. Real-I/O and
qualified-source gates remain closed.1460Python/13separate CUDA cases and full
Ruff/mypy pass in the supported test environment. No real v2 source/fit/experiment
has run. Later engineering Source/Generation2, numerical wire3 and common PPO
coordinator increments are now accepted after separate reviews (through83c51e1);
their full scopes and exact checks are in the verification ledger. Native/tree
integration and qualified real-game acceptance remain incomplete. The following
initial slice counts/statuses are historical milestones, superseded by that ledger.

The first pure records/token/config/network slice is implemented in
`gf_common.structured`, `structured_tokens`, `structured_vocab` and
`gf_models.structured_config`, `structured_network`. Independent correctness and
conformance reviews are READY after fixing joint endpoint admission, event/cache
bounds, inherited initialization and fixed KV storage. Root accepts only this
slice:102CPU/6separateCUDA and110selected legacy checks pass, scoped Ruff/mypy pass.
The pure v4 generator slice is also implemented/accepted after independent
correctness/conformance review and166affected checks. The pure artifact3/resume3/
rollout3 and E2E PPO numeric/inference API slice is implemented/accepted after
both reviews,168affected CPU/4separate CUDA checks and scoped Ruff/mypy, including
1024 persistence and synthetic exact two-update/resume. Config/binding dispatch
is accepted after both scoped reviews,112checks, old12exact identities and6new
draft graph loads. Worker/application routing, source journal/native/generation
integration, LightGBM integration and
required real-game/performance acceptance remain incomplete. The standalone
candidate-columns-v1 codec/sidecar I/O is accepted after33checks and both reviews;
standalone protocol2/structural journal5 transactions are also accepted after
223affected CPU/3full-default CUDA checks and both reviews. Environment/CLI,
formal source qualification, Generation2/PPO orchestration and native decoding
remain pending.
Issue0005 still blocks full completion. See the verification ledger for scope.

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

Bounded v2 exception accepted after focused architecture design review: client worker/
bootstrap may import stdlib-only metadata helpers from
`gf_models.structured_binding`, `structured_config` and `structured_runtime` to
validate own closed settings/identities and derive reservation/cost dimensions.
There is no reverse dependency, Torch/native model import or inference call in
the coordinator; actual policy factory import/execution stays in the clean child.
Sharing these pure formulas avoids a second inconsistent copy or a generic
dispatch framework. Common remains independent of upper layers; client never
imports training. Revisit if metadata gains I/O/Torch or new model kinds require
client policy-behavior branches; move/inject the closed descriptor then. Focused
design review is READY (2026-11-01); actual runtime conformance still requires
the separate journal/process-invariant fixes and recheck.

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

Initialization retains ADR0022's numerical conventions: initialize each temporal
and interval-GRU gate independently orthogonal with gain1; hand/temporal attention
Q/K/V/output and category embeddings gain1; ReLU linear stages gain√2; actor output
gain.01 and value output1; bias0, LayerNorm weight1/bias0, independent draws per
layer. The dense/factorized actor shares the same initialized matrix. Transformer
inference appends KV into preallocated bounded participant storage; differentiable
full/split execution may concatenate prefix/suffix KV. Do not grow inference cache
by torch.cat on every decision.

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

The hand-refresh axis is a public potential acquisition/refill proxy, including
miracle-use acquisition opportunities (§12.4/16.1) and ordinary eligible refill/
Sacrifice. Cap it using visible possession headroom after known removals; a hand
miracle's transfer into learned storage does not itself free a possession slot,
and self-Sell retains the sold item. Normal Discard has no refill. This strategy
summary is not an assertion of future RNG draws, exact post-action state or a
learning target; no hidden/future result refines it.

The configured usage-reference length applies to every action, including fixed
two-reference Sell. With max_sequence_length1, Sell is excluded by configured
capacity (not declared publicly illegal); it is not a mandatory singleton. The
counter/admission domain and table position capacity must apply the same bound.

Implementation stage allocation reserves25% of the same configured search/
selection wall budget for selection/preparation and uses75% for discovery; it
adds no budget or ignored Config setting. Record actual stage and table-finalize
costs and check deadlines cooperatively. Channel minimum opportunities apply to
discovered nonempty channels within remaining distinct-action capacity; no
requirement allocates a separate slot to every overlapping tag or mask.

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

New C1 JSON manifests have a bounded64MiB ceiling, distinct from each decoded
numeric shard's64MiB total. Size planning includes the complete typed-bundle/
member metadata and final canonical manifest, not only episode metadata. Writer
and reader use the same bound; JSON expansion and simultaneous decoded/raw/token
ownership are admitted before parsing/reconstruction. This accommodates maximum
history across the16-game batch without lowering the1024decision limit merely
to fit an incidental metadata ceiling. Large inputs can still fail an explicit
active resource limit; no record truncation is allowed. Focused independent
design review of this ceiling/admission clarification is READY (2026-11-01);
implementation fixes and final conformance remain pending.

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

Additional defense-role review (READY): positive-defense Spiritual Hood/Sash/
Socks remain ordinary numeric singleton/armor/ring proposals without miracle
companions; pure Doll remains issue0005. Fever Mask/Dreaming Hat preserve numeric
composition and ordered response effects (§14/18.4.1). Rainbow Curtain supports a
hinted filter singleton and filter-plus-ordinary-armor/ring composition (§15.1),
without deriving hidden incoming kind. Reflect/bounce armor remains exclusive
despite printed defense (§14.2); Wall/Turbulence are singleton-only here. Every
component is inside same-revision actor hints. This does not establish CutCost
plus miracle placement or other special combinations.

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
PublicCandidateGeneratorV2.generate(frame, bank_view, search_rng, numeric_costs)
    -> GeneratedCandidates(table: CandidateTable, report: SearchReport)
CandidateTable.to_action(index, original_frame) -> typed Action
PolicyV2.reset(sampling_seed)
PolicyV2.select_frame(frame, bank_view, table) -> Selection
tokenize_frame(frame, bank_view, recipe) -> deterministic FrameTokens
tokenize_table(table, frame, recipe) -> deterministic CandidateTokens
```

`GeneratedCandidates` separates the immutable semantic table from closed search
diagnostics. `SearchReport` records generator/config identity, pre-count exact/
upper-bound/unknown and saturation, estimated complete/prefix/attempt/element and
preparation/scoring/storage costs, admission decision, actual attempts/prunes/
discovered/selected counts, stage timings, independent search RNG before/after,
completion flags and cutoff reasons. Traversal completion and presentation
completeness are separate; presenting a subset is never called a full candidate
set. The report never enters encoder/policy/critic features. Parent journals it
alongside the table binding and selection.
Report identity binds the returned table digest and exact numeric-cost descriptor
hash. Independent design review of this addition is READY (2026-11-01); executable
checks and final implementation conformance remain required.

`numeric_costs` is a small closed model-neutral numeric cost descriptor derived
by the worker from the active model/binding's already admitted resource envelope;
baseline scoring has zero neural costs. It contains dimensions/byte limits, not
model objects, hidden values or callbacks into upper packages. Common saturating
counting combines these costs with proposal/element counts before constructing
Actions. The exact descriptor and report schemas are frozen with generator
implementation; do not add a generic plugin cost framework.

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

## 13. Application integration design (review before dependent implementation)

Actions2 uses explicit baseline registry/schedule/journal/Selection labels
`random-structured-v1` and `heuristic-structured-v1`; their nominal configured
model IDs/version1 remain separate closed identities. Actions1 labels stay
unchanged. New collection/source authorization derives expected baseline labels
explicitly, never infers learner role from label equality or accepts an unknown
model merely from a matching prefix.

Initial protocol2 private timing counters label host wall scopes accurately:
input packing/transfer host time, forward enqueue host time, explicit sync host
time, CPU return and CPU sampling separately, plus tokenization/search/IPC/startup
and storage/Simulator scopes outside policy math. Tensor-constructor timing can
include waiting for prior stream work; do not call it pure H2D or GPU kernel time.
Counters are per-select diagnostics only, never features or critic input. Final
GPU performance acceptance still needs CUDA-event/isolated transfer and forward
evidence; host counters alone do not establish a CPU/GPU bottleneck or speedup.

ConfiguredNeuralModelV2 also has a sibling `cuda_visible_devices: str|None`,
removed by exact architecture conversion. None means unset in a clean CUDA
child, without implicit coordinator-environment inheritance; a validated explicit
index/GPU-UUID selector is forwarded as the sole visibility setting. CPU children
always use empty visibility. The selector also applies to a CUDA learner when
inference is CPU. Record actual visibility in execution/resume identity and hash
active resources, pure inference binding and actual visibility consistently in
policy/CLI/source labels. A direct C1 host API records its actual environment;
production guarantees use the clean child. No coordinator mutation or service
sharing is introduced.

Pinned Torch2.14 Adam's accelerator graph-health check may query CUDA availability
and current stream even for CPU parameters. Keep standard Adam. Production CPU
numerical/policy workers set `CUDA_VISIBLE_DEVICES=""` inside their clean child
bootstrap before any Torch import; never mutate the coordinator or an already
initialized shared process. CUDA workers retain configured visibility and charged
contexts. Pure direct CPU learner APIs guarantee CPU math, not absence of standard
Torch probes in an arbitrary GPU-visible host; strict context ownership requires
the clean CPU child boundary. Constructor/save/restore avoid unnecessary CUDA
queries; do not assert that standard Adam never performs them or override private
Torch internals to force that assertion.

V2 Selection logprob/value become optional to preserve baseline `None` values;
validate finite/nonpositive values when present. Do not invent baseline critic
values. PPO source authorization/ingestion explicitly requires both finite fields
on every learner row before parity/update, rejecting missing values without zero
imputation. This v2-only clarification is independently design-reviewed READY;
require baseline-null roundtrip and PPO-null rejection tests.

Independent design review of this section is READY (2026-11-01). Implementation
and final conformance remain pending. Partial Config validation/default merging
must dispatch by version before old field checking, not just at final model parse.
Collection run-manifest schemas stay3(non-PPO)/4(PPO); actions2 uses
journal_schema_version5 and feature identity participant-structured-v1. Preserve
schema4's whole-PPO-parent authorization path when reading journal5—new journal
format alone never bypasses failed/incomplete-parent exclusion.

Participant residency (weights, CUDA context and full cache) is reserved before
spawn/initialize and held through confirmed worker reap, including startup and
preflight workers. A reply releases only scoped decision temporaries. The same
four-worker/global budget includes startup workers and resident learner weights/
Adam throughout collection; no separate uncharged preflight pool exists.

V2 policy/binding labels include model kind/version, full inference-manifest
digest and full runtime-binding digest. The same checkpoint on CPU and CUDA can
therefore coexist as distinct evaluation labels without overwriting/rejecting
one binding. Selection/source/native authorization checks that complete identity;
artifact/network identity remains independent of device. Do not change legacy
label syntax. Ordinary resume still requires the exact original binding.

Saved artifact3 `model.resources` records creation-time admission defaults and
provenance. Active configured resource limits govern preallocation/construction;
network compatibility compares `architecture_identity` (which excludes resources
and runtime binding), not full dataclass equality. With no explicit override,
use and validate saved defaults. A changed resource limit is permitted for a new
inference binding only when all active checks pass; ordinary resume rejects it
through full Config/binding identity. No weakened shape/tensor/runtime validation.

The pure `ModelConfigV2` is network configuration and contains no checkpoint or
inference binding. For application Config use a distinct closed
`ConfiguredNeuralModelV2` dataclass extending those flattened fields with optional
checkpoint and `InferenceBindingV2`. Its explicit `architecture()` conversion
returns the exact pure base type for network/token/resource APIs; never weaken
their exact-type validation to accept arbitrary subclasses. `model.version="2"`
still appears at the existing Config model level. Artifact3 stores the pure
network recipe, while actual safe runtime binding is separately recorded/hashed.
No new runtime fields are inserted into v1 dataclass serialization.

LightGBM2 has a separate closed configured model type: deterministic structured
encoder recipe/max positions/aggregate window, native tree parameters,
initialization and CPU chunk/resource binding. It has no trainable Hand encoder
settings. `model_config` dispatches on version and model kind before old parsing.
Actions2 can accompany v1 Random/heuristic through explicit structured baseline
policies; old neural/LightGBM feature artifacts cannot bind to actions2.

Model2 uses its one existing `model.temporal.sequence_length` setting for TBPTT
(default32). Do not duplicate it as a second training setting. New PPO training
presets use the base PPO settings for all three kinds; legacy sequence-model
training.sequence_length serialization retains its original behavior. A mismatched
old training preset on model2 rejects before I/O. Ordinary resume binds all model,
action, window, resource and actual inference-device settings unchanged.

An actions2-specific run Config adds global CUDA managed admission budget12GiB
without changing legacy RunConfig fields/hashes. New runtime supports at most
two simultaneous games/four participant workers and retains resident-learner
reservation. Each endpoint reserves its computed envelope, not the entire global
ceiling. Resource failures do not trigger automatic candidate/window/device edits.

CLI chooses the protocol2 factory/worker explicitly for actions2. Startup has a
separate initialize/ready exchange so model load is timed outside decision latency.
Worker initial messages carry only factory/model ID, sampling seed, action settings
and closed checkpoint/runtime binding; no effective_config, URL or GameSeed.
Protocol2 request carries actor-local EventDelta plus bound raw frame. Worker
validates/appends once, generates and selects; response contains compact table,
search report, bound Selection and scoped timings. Parent validates hashes and
references, reconstructs the Action against its original frame, then journals and
submits. CPU probabilities are sampled only after CUDA synchronization/finite
checks. Baseline policies do not materialize all candidates as legacy Action JSON.

Schema5 journal writes append events once and immutable frame/table/search/
selection references, preserving existing positive-acknowledgement and closure
semantics. Explicit reader dispatch validates5 independently from4; no input
history or semantic version is inferred. New run/source summaries carry model
version/input recipe/generation schema. Native capability checks require schema5
and participant-structured-v1 tokenization before a native collection starts;
model-ID-only capability cannot authorize model2. Existing v1 source/preprocessing
recipes and all old proof bytes remain unchanged.

PPO numeric rollout3 stores deterministic state/card/event tokens and compact
candidate references, episode/decision offsets and original frozen policy values,
with train-only GAE targets separate. Raw event tokens are banked once per episode;
frames reference intervals and own-card state at observation time. Sharded typed
arrays are enumerated in the manifest, decoded total<=64MiB per shard. Training
packs only admitted endpoints and computes encoder outputs with current weights;
it never persists learned latents. Existing PPO math, 128 logical endpoints,
minibatch shuffle, clip/Adam and four-epoch semantics are reused, with all-candidate
checkpoint chunks rather than the old4096 total-candidate cap.

Artifact3 named optimizer state distinguishes uninitialized parameters from
initialized moments, persists each parameter's step, and restores identical
optimizer membership/order. This matters for empty interval GRU or direct-hand
unused branches. Generation2's explicit file map governs hash checks and
retention, including all shard sidecars and deletion receipts; schema1 keeps its
old fixed-file contract. Native authorization/projection and fixed dataset2
validate these new identities rather than accepting an old model solely by ID.

These are application-level extensions of the accepted ownership, not a new
trainer, engine, serving service or generic dispatch framework. Required gate
evidence remains in §§7–9/12; real fitting/campaigns remain separately gated.

## 14. Native additive boundary (review before dependent implementation)

Extend the existing closed preprocessing reader-contract enum with
`structured-v1`, without changing old fields/defaults or serialized values.
Require that contract for structured-input/journal5 preparation and reject it
for legacy-input/schema4 paths; gate by input recipe, including valid baseline
model1 with actions2, rather than nominal model version alone. New presets select
it explicitly. No auto-mapping from legacy/typed-borrowed-v1 or backend fallback.
Native uses a separate owned typed journal5 path with existing thread/memory
limits; an explicitly chosen Python reference backend identifies its own
journal5 reference implementation, never claims Rust processing.

Independent architecture design review: READY (2026-11-01). Native code/build/
parity and final conformance remain pending. Preserve existing summary/IPC bounds
and lease/cancellation ownership; derive actual window/signature from validated
source/model semantics rather than echoing an untrusted request.

Read-only C2 investigation maps the existing boundary through Python journal/
native_authorization/native_ppo and Rust session/controls/projection/journal/
typed_journal/pipeline/feature. The old NativeBatch/feature.rs hashed encoder and
schema4 typed reader must not be overloaded with new token semantics.

Keep existing base `recipe_identity=native-preprocessing-v1`, legacy reader
contracts and v1 fingerprints unchanged. Append exact build-info strings:
`structured_preprocessing_contract=participant-structured-v1;journal=5;rollout=3;generation=2;model=2`
and `structured_default_signature=<Python signature(aggregate_window=128)>`.
The latter binds the complete ordered layout/vocabulary at the default window;
each request/proof additionally binds its actual configured window/signature.
For model2 require both markers before collection/fixed preparation. Existing
`ppo_policy_models` gates apply to neural learners and PPO source-parent IDs,
not Random/heuristic counterparts or the LightGBM2 consumer. Those roles have
their own closed model checks plus the structured schema/token capability;
LightGBM2's PPO source parents still require supported neural IDs. No missing-marker
MLP fallback is permitted for2; historical fallback remains only for1. Header
strings are not legality data.

Add a structured native session/batch path, rather than changing NativeBatch's
existing positional payload. It validates schema5 bank/frame/table records and
returns raw category/numeric/presence banks/offsets with compact candidate refs
and structural proof. Python still owns source authorization, GAE/labels,
optimization, Arrow publication and interpretation of natural/failed outcomes.
Native never calls Python model code or stores learned latents. Existing scoped
ownership/cancellation/lease/finalization patterns remain, with no fallback.

V2 parent/generation/collection compact summaries and proofs explicitly surface
model_id/model_version/input_recipe/journal_schema/feature identity/generation
schema, and rollout schema3 where applicable, plus actual token signature. Bind
these into a domain-separated v2 control fingerprint together with original
source file digests and authorized policy-binding labels. A metadata string
carried only by an untrusted request is insufficient. Collection3/4 still selects
non-PPO/whole-PPO-parent qualification independently of journal5. Baseline opponent
Config version1 with actions2 remains valid; parent/learner model2 identity must
not be incorrectly imposed on Random/heuristic counterpart artifacts.

Rust generation2 validates explicit artifact file maps/manifests and semantic
identity; generation1's fixed five-file retention/hash behavior is preserved.
Python native authorization consumes the small v2 summary without rereading or
copying full effective_config into IPC. Native preprocessing identity in parent
resume includes actual binary/build/capability/recipe/signature identities.

Required evidence: Python/native raw token/category/presence/order/hash equality
on same original owned inputs, all invalid prefix/actor/version/missing policy
values/shard cases, and exact old corpus/index/saved-PPO proofs using fresh output
directories. Build/install a fresh locked offline wheel and verify loaded ELF/
build_info/source digests; checkout HEAD alone is not build provenance. Existing
saved-data probe is `tests/native_v2_ppo_probe.py` (its name denotes the historical
catalog-v2 fixture, not this new schema5/model2 contract). No native tests/builds
or new-runtime parity were run by the investigation.

## 15. Journal5 candidate-column storage (design review required)

Focused independent architecture design review: READY (2026-11-01), including
raw-code convention and publication/failure ordering. Codec implementation and
all executable compatibility/resource/performance checks remain pending.

Raw kind/target codes are explicitly zero-based ACTION_KINDS/TARGETS indices.
PAD/MISSING offsets belong to tokenization only and never appear in these raw
table columns.

Publish each sidecar at a unique actor+decision-owned basename: no-overwrite
temporary write, hash/close, then atomic owned publication before the referencing
JSONL row. Submit no Act if either sidecar or journal-row write fails. Preserve
failed/orphan evidence without inventing an accepted transition, and retain the
existing positive-acknowledgement/closure qualification rules.

A bounded size investigation uses synthetic32768-row tables, not observed game
counts or legality evidence. With18zero-valued references per row the full current
CandidateTable JSON is3,565,775bytes; two participant journals with20such decisions
each cause4.25GiB candidate-only admission under existing whole-file32×JSON policy.
At128references it is10,804,995bytes/table. No throughput/RSS/normal-game inference
is drawn. Raising the existing2GiB preprocessing budget or dropping proposals to
hide this representation cost would violate the plan's baseline comparison.

Schema5 therefore stores table columns in an owned uncompressed typed sidecar;
the JSONL row retains frame/table hashes, a closed sidecar descriptor, search
report and Selection, without duplicating the table vectors. Parent owns writes
after validating worker output; no journal paths or management state go to workers.
In-memory CandidateTable and token semantics do not change. Protocol2 remains
separately bounded JSON initially; measure its actual512-panel communication cost
before considering a transport optimization.

Codec `candidate-columns-v1` has a fixed closed column order: kinds uint8,
relative targets uint8, numerics int64[N,4], presence uint8[N,4], offsets
int64[N+1], slots uint8[S]. Codes use the existing exact ACTION_KINDS/TARGETS
order, not a new feature vocabulary. N1..65536, S<=128*N; each slot0..17,
presence0/1, canonical absent numerics0, offsets start0/endS and row lengths
respect128/active capacity. Integer data is little-endian; int64 regions align
to8bytes with checked zero padding. Descriptor includes codec/schema, N/S,
column dtype/shape/offset/byte counts, exact owned relative basename, file length,
SHA-256 and semantic table/frame hashes. Reject unknown fields/layouts, overlap,
path traversal/symlinks, bad lengths/codes and trailing data before use.

The complete default single-table payload is at most about11MiB at65536×128,
including fixed columns; nevertheless enforce decoded-shard total<=64MiB before
allocation/write. Smaller configured resource limits may reject explicitly.
No pickle, compression, new crate, general serialization framework or model code
in native. Reuse existing bounded read_i64/write_i64, add the corresponding tiny
chunked u8 I/O with cancellation/reservation/rollback, and existing typed-owner
leases. Fixed layout and hashes identify bytes; it is not an engine-rule parser.

Journal/source inventories include referenced sidecar files and verified hashes,
not merely the JSONL path. Native derives and validates those links from original
schema5 records; include actual files/semantic identities in v2 fingerprint and
final source recheck. An altered/missing/reordered/foreign frame/table sidecar
rejects the source. Old journal4 JSON/body/inventory/fingerprint remain unchanged.
Candidate sidecars use newly owned safe publication; do not change legacy array
writers as a side effect. Numeric rollout3/model artifact3 keep their separately
reviewed NumPy shard formats and per-shard decoded64MiB bound.

Verification: exact Python/raw-column/native reconstruction and canonical hashes,
edge columns/targets/presence/order, all corruption/ownership/64MiB cases and
source-inventory mutation; default2GiB/16MiB preprocessing presets remain. Size
calculation is only design evidence; build, differential, real-game and512-panel
performance acceptance are still required. This new storage detail needs focused
architecture review before dependent implementation.

## 16. Structured LightGBM and fixed-data2 design (review required)

Focused independent architecture design review: READY (2026-11-01). Preserve
installed audited LightGBM4.7.0; the4.6 primary documentation below establishes
the supported API shape, and installed4.7.0 code confirms its implementation.
No dependency downgrade/update is authorized or required. Categorical values
must exactly roundtrip FP32 as integers in their closed vocabulary (current
codes<300), not merely fit int32. Keep category columns out of signed-log numeric
transforms and set zero-as-missing false so PAD0 is not merged with missing1.
Implementation/fits/builds and final conformance remain pending.

LightGBM2 uses its own closed configured model/encoder/native-tree/resource/CPU
binding types, not inactive neural Hand/State/Temporal settings. Default position
capacity18, aggregate window128, chunk256 and existing tree/fit hyperparameters
remain implementation baselines. Inference/training schema3 binds structured
input/feature signatures, exact ordered feature names, categorical column indices,
catalog/spec/reward/generator identities, native library/model digest and actual
execution resource settings. V1 LightGBM configs/features/artifacts remain intact.

The deterministic tree recipe consumes current detailed state, every own held/
learned card and learned order/usability, ordered candidate references/arguments,
opponent's public learned IDs, and the fixed visible history aggregate. It has
no learned encoder or decision-series hidden state. Exact category codes have
distinct pad/missing meanings and are explicitly passed as categorical columns;
instance ID magnitudes, source/control metadata and training labels never become
features. Own card values use observation-time perceived definitions only;
opponent/past public IDs do not trigger extra catalog lookups or possession claims.
Candidate permutation only permutes scores. Record the tree's bounded history
scope separately from neural own-decision history.

Produce contiguous FP32 feature-row blocks whose individual arrays and decoded
shards fit64MiB, preserving exact integer category codes (all below int32 max).
Training passes a list of equally shaped-column NumPy matrices to the supported
LightGBM Dataset API rather than concatenating a giant tensor. Admit aggregate
resident blocks, native Dataset/model/workspace and validation data before work;
native internals remain a measured soft estimate. Inference creates only admitted
candidate chunks. Native Dataset construction may retain/release raw blocks only
under explicit ownership; no pickle or frozen Torch encoder is added.
Primary evidence: [LightGBM4.6 Dataset](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.Dataset.html)
and installed `basic.py` list-matrix/LGBM_DatasetCreateFromMats implementation.

Fixed manifest2 requires ordered seeds17/29/43 in one homogeneous model2 PPO
cohort and prescribed final-policy validation, as§12. It binds actual action/
feature/runtime conditions and original source sidecar hashes. Revalidate all
parents/closures/ancestry/splits/generation2 file maps and reject failed or partial
parents as a whole. Dataset1/model1 cannot be silently retokenized as2. Export/
index flags are not fit authority. Source qualification reads sealed metadata and
declared supported semantics without executing neural policies or requiring Torch
in a LightGBM-only environment. Active inference/resume checks remain separate.

Legacy inclusive mode retains its original pilot/source/decision-budget gate.
Any model2 inclusive source must be explicitly schema5/new-recipe with its own
validated pilot/conditions; old data never acquires new semantics by dispatch.
Fixed-data2 fitting needs all three complete prescribed parents and validation.
Both costs/recipes remain distinct, generation cost deduplicated as before.
Fresh fits only, no LightGBM resume. Real model2 fits/formal parent campaigns are
not run as part of component acceptance. Tiny explicit synthetic fits validate
categorical interpretation/roundtrip/chunking/gates only, not quality or feasibility.

Required checks include exact deterministic feature/categorical signatures,
all held/learned/target/order/missingness/public-history cases, large position/chunk
admission, native categorical fit/inference and safe schema3 publication, old
dataset/artifact regressions and complete-parent/split/seed/budget negative cases.
No production preset advertises LightGBM2 before these paths are implemented.

## 17. Owner clarification and engineering-only proposal adoption

Root disposition after `bot_advisor`: adopt alternative B, an explicit
engineering-only proposal profile. Assertions authorize correcting new Bots
proposals, not claiming conformance of the current binary or completing missing
coverage. Adopted subset/omissions are in
[owner-proposal-engineering-v1.json](../contracts/owner-proposal-engineering-v1.json),
profile SHA-256 `4f18089923ab8b6b4b6346d5adfbe0ca5fba401ead3a6d7c186c9596327c7a16`.
The rule/omission/verification-claim digest is
`f4132843bb700947d51e19b57bed1b3bc9bcaf5f63c0cfa60998f1ddc3d6d978`.
Focused independent architecture review is READY for engineering-only
implementation (2026-11-01), including canonical hashes, explicit raw Config
discrimination and all real-I/O guards. This records root's reviewed decision,
not implementation/runtime/learning acceptance.

One closed `action_contract` descriptor contains profile_id, owner_confirmation_
sha256, adopted_rules_sha256 and eligibility=engineering_only. Bind the same
descriptor in actions2 effective Config/SearchReport, run/Journal5, artifact3/
resume3/Generation2, rollout3, native source-derived summaries/proofs and
Dataset2/LightGBM3. It is compatibility/control metadata, never a feature. Preserve
all baseline simulator/catalog/reviewed-spec fields separately and leave the
global v1 constants/readers unchanged. Missing descriptors on pre-adoption v2
artifacts reject current inference/update/resume/qualification; no implicit
default or automatic upgrade. Pure tensor/layout signatures need not change.

Raw actions2 Config must explicitly contain that descriptor or inherit it from
an explicit versioned engineering actions preset. Default merging must not fill
it into pre-adoption v2 TOMLs/effective configs. Partial overrides may omit it
only if their resolved reference already declared it. One authoritative Config
descriptor propagates to every boundary; readers closed-require/compare it,
never independently infer a missing value from current globals. A small central
validator can recognize the adopted descriptor; no generic rule framework is
needed. New synthetic component construction is explicit about contract/purpose.

CandidateTable additionally closed-requires purpose=component|qualified_game and
synthetic_capacity boolean, both bound in semantic hash/sidecar/wire metadata.
Repeated physical refs or rows above18 are allowed only in a component fixture
explicitly marked synthetic_capacity=true. Every generator/protocol/Journal5
game-proposal path rejects that marker and requires unique<=18 even while the
engineering profile itself is component-only. Qualified-game purpose requires a
qualified contract and still conveys no Source authority. Thus FakeHTTP and
frozen real-input replays do not acquire a false “synthetic128” provenance, and
a marker/flag cannot upgrade any engineering source. Numeric column layout and
network feature signatures remain unchanged.

Artifact/resume/rollout/Learner/Policy component APIs require the same explicit
contract, purpose and synthetic_capacity classification. For artifacts this is
data/provenance/allowed-use metadata, not maximum tensor position capacity: an
unupdated false-classified component model can still have128position parameters
while receiving unique<=18game-style tables. True synthetic-capacity artifacts
cannot be silently relabeled or loaded as false/game-proposal policies; resource
overrides never change classification. No flag alone grants formal Source
eligibility. New Frame reference maps also reject duplicate learned IDs/hand IDs
so distinct slots cannot alias one source+ID physical reference; v1 DTO validation
remains unchanged and distinct physical hand instances/sources are preserved.

Engineering-only permits component/FakeHTTP/synthetic numerical/storage tests and
offline fixed-input performance work. Real collection/evaluation, the192-game
feasibility gate, real PPO acceptance, qualified-source/fit/strength/full acceptance
remain blocked until required public coverage and runtime conformance are adopted.
Qualified game records require unique physical refs and the reachable18slot
bound.128-reference structural fixtures remain explicit synthetic components,
never qualified trajectories. Persist component-vs-qualified purpose explicitly
for new rollout/schema readers; source authority must still revalidate originals,
not trust a purpose flag alone. No budgets/counts/models/panel requirements shrink.

The real-I/O gate applies before any Simulator request/spawn in every supported
actions2 entry: collect/evaluate/train/smoke/real-game benchmark and direct
supported Environment use, not only two CLI command names. There is no
user-controlled component flag that can authorize formal sources. FakeHTTP tests
use a dedicated internal test path/double and remain component-only; original
source/profile revalidation always rejects this engineering profile for qualified
learning, regardless of a claimed purpose/eligibility field. Offline participant
inference with frozen visible inputs is not a Simulator experiment.

For utility proposals, the profile uses explicit targets; Tear/Herb currently
adopt self only pending the target/effect discrepancy. Summons require explicit
public guardian absence; soap requires a visible learned list with at least2;
broom adopts self with at least3remaining own hand refs after its carrier, never
an opponent private-count predicate. These are pending-coverage exclusions, not
public illegality claims or accepted scope reduction. Mixed utilities/CutCost
placement and the same-normal-miracle-copy exception remain pending. Reflection
decision flow remains entirely simulator-driven.

Material external-state update: read-only checks now find dirty Simulator07/09/
verification-register changes plus a new utility investigation, while HEAD is
still ffcffa. The observed document SHA-256 values are
07=`66d18191c596a7e371d963c0fa66a8e040cbef7b60433a2ede8acf9bccff650f`,
09=`dc363d47bb27f145e7dd7fbc95f75c8bd9f705fc229766fc56ad35b8c797f18b`,
register=`d38dda2eadabd36e6615892ea5ed4609005c21ebda630a95c0d44fe1026ba71a`.
They differ from the initial clean/declared baseline and were not modified by
Bots agents. The working copy reconciles some copy/reflect text, but17.5.4 is
narrower than the user summary for per-card target selection. No new committed
revision, full document adoption or binary-conformance proof is asserted.

The user's reply is recorded as immutable owner assertions with canonical hash
`a0e6acb7fdc708ce4bd1b11f23d5dd3f0e41373cc1c1c1bef0a43a3e3efc2cb2`.
It is not a newly published Simulator revision/spec hash. Do not overwrite the
baseline ffcffa/public-ja-v2 identities or reinterpret historical artifacts.

New proposals must stop relying on one learned reference repeated in an Action.
Distinct physical copies and across-turn learned reuse are different rules.
With18total possession slots, unique actual references cannot realize a128-element
game proposal; retain128position/config/shard/network capacity tests as synthetic
structural/numerical evidence, alongside real-domain uniqueness/18slot coverage.
No experiment budget, input facts or final-test policy changes follow from this.

Owner-confirmed utilities may use explicit self/opponent target syntax, avoiding
unknown None defaults, but chosen target must not be reinterpreted as effect
recipient for user-fixed effects. Unconfirmed mixed utility/CutCost placement,
guardian/removal boundaries and opponent utility responses remain unknown. Same-
definition normal-miracle copies and reflection exchanges conflict with broad
current normative statements; the Simulator owns resulting decision/state flow.
Never synthesize reflection opportunities or probe rejections to fill the gap.

Root considered these alternatives: wait for Simulator normative adoption for every
dependent rule; adopt only a separately versioned owner-proposal profile with
unchanged baseline pins and strict experiment gates; or claim full contract
readiness from the owner reply alone. The third alternative lacks implementation/
normative reconciliation evidence. Advisor review recommended the versioned
engineering-only subset adopted above; root owns the disposition and required
architecture review before dependent rule edits. Independent FakeHTTP dispatch,
storage, numerical and cleanup work can proceed within its actual scope.

## 18. Source/Generation2/PPO integration under the engineering gate

Architecture review accepted this bounded design before dependent implementation.
Implementation proceeds in separately reviewed Source, Generation2 and coordinator
increments; design readiness does not imply implemented or qualified execution.

Continue application implementation without treating an engineering profile as a
qualified game contract. Preserve schema4/model1 sources and Generation1 paths
byte/meaning unchanged; version dispatch precedes decoding.

New structural collection projection reads original schema5 manifests/journals/
referenced columns, verifies identities, caps, closure/ack evidence, inventories
and final hashes, and reconstructs actor-local EpisodeRecords. It can return
structural component evidence with explicit non-qualified purpose. Public Source/
export/fit and normal real-game PPO entries independently require a qualified
contract and original whole-parent/Replay/split qualification. No producer,
completed status, purpose flag or structural reader supplies that authority.
Current engineering profile always fails qualified use.

All-natural cohort projection requires matching original Replays and a successful
linked report independently of the manifest status. Mixed incomplete diagnostics
may retain known natural outcomes beside interrupted None outcomes, but preserve
every original learner episode, including zero-endpoint interrupted ones, so
healthy-row salvage cannot authorize a cohort update. Interrupted component
diagnostics may omit a Replay only when the
original episode health explicitly records replay/finalization failure. Preserve
accepted endpoints, exact closure reason, absent outcome and diagnostics; do not
infer rewards, bootstrap values or qualification from the missing Replay. Every
present Replay/manifest/sidecar conflict is rejected. Before Generation2 parent
proof is implemented, structural Source reports qualified=false and
parent_qualification=not_performed even for a collection4 envelope.
Compare closures from present Replays against each other and the original known
cause. An allowed missing view is absent evidence, not a contradictory closure;
it does not authorize reconstructing that participant's missing history.

Before candidate decoding, reject duplicate original game/participant identities
and compare each journal's own sampling seed and capacities with the original
schedule and saved policy/action scalar limits. These are internal consistency
checks, requiring no Torch, current hardware or current resource-estimator cost
formula. Generation2 separately reconstructs the parent-authorized schedule.
The Source work deadline is bounded by both the caller deadline and the configured
preprocessing wall allowance with its cleanup reserve; an infinite caller value
does not disable the configured budget.

Failed startup must retain a process owner as well as its budget reservation when
the first kill/wait cannot confirm reap. A private strong owner reference in the
existing WorkerReservation is registered before Popen and cleared only on confirmed
reap or a no-process rollback. The ledger retains references and checks release
conditions; it does not orchestrate process shutdown.
A narrow private exception-owner handoff preserves the initiating exception type
and lets StructuredEnvironment register the failed actor in its existing workers
cleanup set. Type/ledger identity is checked. Existing bounded finalization retries
close; an unregistered PID may roll back only after that owned Popen is confirmed
reaped. If retry still fails, retain both owner and reservation and report
infrastructure failure. No generic process registry or lifecycle framework is
introduced. Architecture review accepted this bounded ownership amendment before
implementation; failure-injection and conformance review are required for acceptance.

Generation2 enumerates exact owned inference/resume manifests/shards and hashes,
model/version/input/action-contract/classification identity, counters and chain
links. Retention validates/deletes only listed owned files, preserving inference
identity manifests/deletion receipts. Generation1's fixed-file protocol remains.
Sealed historical model2 metadata validation requires no Torch or equality to
current hardware/code; active policy loading and ordinary resume remain strict.

Reuse the existing PPO coordinator's schedules, locks, deadlines/storage,
16-game batches, explicit learner-viewer roles, snapshot cadence and committed-
boundary pause/resume. Dispatch model2 to C1 numeric worker/Learner/rollout3;
old model1 math/serializers stay intact. Owned current/pool bindings use committed
Generation2 file maps and exact actual execution identity. Formal binding needs
a qualified healthy complete parent's final generation, never bare weights.

For model2 the coordinator creates/adopts one ReservationLedger and charges the
computed resident learner envelope BEFORE numerical-child Popen. Keep that charge
through collection and release only after learner reap. Use a bounded collector-
factory closure receiving that SAME ledger; the composed run_experiment/structured
Environment accepts the injected ledger rather than creating a fresh zero-learner
ledger. CollectionPlan remains a serializable value with no runtime object/default
added to v1. Standalone collect may create its own zero-learner ledger; owned PPO
collection must reject absence/mismatch. Begin updates only after all participant
leases are gone. Test object identity and charge lifetimes, not merely reported
worker-count flags. No general resource framework is introduced.

Test now through a dedicated PRIVATE component coordinator entry/internal driver
sharing the same coordinator with an injected synthetic/FakeHTTP collector. It
hard-binds component purpose/profile in parent, collection, numeric and generation
evidence. No CLI real-I/O bypass is exposed; completed component parents/models
remain rejected by formal Source/dataset/binding paths. Public entries keep the
real-I/O gate before any Simulator activity. No new trainer framework or
user-elevatable authority token is introduced.

This Journal5/Source/Generation2 component path always uses purpose=component,
synthetic_capacity=false and unique<=18proposals even with a synthetic/FakeHTTP
collector.128/Truecapacity fixtures remain in separate C1 numerical/structural
tests, not a Source bypass. Model2 training_groups supplies explicit structured
Random/heuristic labels while retaining16-game schedule, seed derivation and
snapshot cadence. Learner role is always the explicit viewer, never label text.

Component projection preserves accepted endpoints, natural reward perspective/
GAE and interrupted closure diagnostics without guessed targets. Verify all three
model2 kinds, exact two-updates versus1+pause/resume arrays/Adam/RNG/source,
retention, role binding and missing metadata/values; late whole-parent failure,
manifest/Replay/sidecar conflicts, locks, changed config/runtime/backend/visibility/
deadlines and cleanup/publication failures. Qualified engineering negatives are
mandatory. Native selection fails early without new capabilities until separate
native implementation; explicit Python reference stays truthful, no fallback.

After this path stabilizes, both reviews and old-format/data proofs are required.
These components do not fulfill blocked real192-game/PPO/performance acceptance.

S1 accepts one bounded dependency debt: the legacy qualified predispatch guard
currently lazily imports the component Source reader, whose Journal5 dependency
also uses the legacy journal module. No observed import failure or eager Torch
import results, but the guard loads more parsing code and carries circular
coupling. Revisit a shallow guard in the existing common contract module when
Source/Generation2 dispatch stabilizes, or immediately if optional-free/legacy
guard import checks fail. Do not add a framework or a new layer for this guard.

## 19. Bounded Generation2 metadata increment

Architecture review accepted this bounded design before implementation, including
the exact current Resume3 inference-link semantics. This is design readiness;
implementation/conformance and later Source authority remain separate gates.

Implement sealed retained/pruned artifact inspection and component generation
chains before coordinator integration. Own two additive modules:
`gf_models.structured_sealed` and `gf_training.structured_lineage`, plus their
focused tests. Extract only pure bundle-metadata validation from the existing
sharded-array inspector; its retained-file behavior, outputs and legacy callers
must remain unchanged. Do not edit Generation1, Source or coordinator dispatch in
this increment. Do not claim original whole-parent/source qualification from a
structurally valid component chain.

Sealed inspection validates artifact3/resume3 declarations, exact named tensor
and Adam-state descriptors, paired metadata identities, full bounded file maps
and regular owned inventories. Process shards individually, each decoded at
most64MiB. It needs only stdlib/NumPy: no Torch import, installed-runtime metadata
lookup, `semantics()`, current hardware or current producer equality. Validate
recorded runtime/numerics/producer shapes and mutual consistency instead. Active
C1 inference/resume remains strict. Pruned inspection uses the preserved
inference's inline bundle metadata, original file map and completed deletion
receipt; it never reconstructs deleted arrays or provides an active binding.
Full paired resume/Adam declaration validation applies only to retained artifacts.
Resume3 references network parameters through its inference_digest and the
inference's weights bundle; it does not duplicate parameter arrays. Verify that
exact link, paired model/classification/semantics, named Adam ownership/shapes and
generation counters. Pruned inspection validates preserved inference and the
exact recorded hashes/deletion receipt; it cannot revalidate deleted resume
contents.
Account cumulative retained JSON/maps/descriptors and one decoded shard/scratch
under a single caller-owned budget and deadline. A per-document cap or64MiBshard
cap does not authorize an unbounded collection of metadata. Verify regular-file
snapshots and final inventories/hashes across the scan before returning results
or deleting, without importing training code into models.

Run2 has the legacy run1 core fields unchanged in name:
`schema_version, algorithm, run_id, status, config_sha256, effective_config,
config_inputs, identity, seed, device, created_unix, last_unix,
wall_deadline_unix, admission_deadline_unix, execution_deadline_unix, updates,
accepted_decisions, optimizer_steps, generation_sha256, collection_seconds,
update_seconds, invocations, completed_updates`.
Add required `model_id, model_version, input_recipe, action_contract, purpose,
synthetic_capacity, resume_identity, execution`. Schema is2; the model is2;
component purpose and False-capacity are mandatory. Enumerate only the existing
closing diagnostics as optional fields: `hardware, elapsed_seconds,
artifact_bytes, peak_coordinator_memory_bytes, coordinator_memory_scope,
timing_scope, failure, quarantine, cleanup_failure, budget_overrun_seconds`.
Committed states require recorded hardware; it is historical metadata, not a
comparison to the inspecting machine. Running/paused/complete states retain the
original health, counters,7200/5400/7140deadline and storage limits.
Use the existing `torch_runtime.hardware()` declaration shape: exactly `device,
name, cuda_runtime, device_count, cuda_driver_api_version, exact_driver_build`.
Validate recorded types, finite values and declared-device consistency; do not
query the inspecting machine or infer an attested driver build from API version.
The C1 learner needs no new hardware method; S3's child can reuse the existing
runtime helper for its active acknowledgement.

`execution` has exactly `learner` and `inference`; each contains a closed
InferenceBindingV2 under `binding` and the actual `cuda_visible_devices` value.
Resume identity is the domain-separated digest of original Config hash, recorded
code/runtime/input identity, these execution bindings and input classification.
Pin the exact digest object to `{recipe: "structured-ppo-resume-identity-v1",
config_sha256, identity, execution, input_binding}`; identity is the entire
recorded identity object, and input_binding is the existing closed3field record.
Actual hardware equality is checked by the active child acknowledgement at normal
resume, separately from sealed historical inspection. No active hardware query
is moved into the coordinator.

Generation2's exact fields are `schema_version, run_id, generation,
parent_sha256, model_id, model_version, input_recipe, action_contract, purpose,
synthetic_capacity, inference_sha256, resume_sha256, artifact_files,
accepted_decisions, optimizer_steps, update_sha256, parent_controls`.
`artifact_files` is the exact full original inference/resume manifests and every
NPZ/JSON sidecar returned by C1, not a single weights.npz shortcut. Controls keep
the closed PCG64 schedule RNG, snapshot_pool and collection_position contract.

Update2 keeps the existing update1 fields and meanings: `schema_version, status,
run_id, generation, parent_sha256, batch_id, collection,
collection_manifest_sha256, numeric, numeric_sha256, accepted_decisions,
source_proof, metrics, planned_games, actual_games, health, policy_bindings,
learner_seats, collection_seconds, update_seconds`.
Add required `model_id, model_version, input_recipe, action_contract, purpose,
synthetic_capacity, numeric_files`. The exact numeric3 file map includes
rollout.json and every declared NPZ/JSON sidecar; verify its original inventory.
Update accepted_decisions is the learner-only numeric decision_count, not S1's
both-view accepted count. Keep16planned/actual original games and every learner
role, including zero-endpoint games. S2 verifies structural count/hash/model/
classification links; S3 separately reconstructs authorized schedules and original
Source-to-rollout correspondence. Component proofs remain explicitly unqualified.

`require_retained=true` means every generation must retain its full original
artifacts. Default historical chain inspection permits only verified completed
pruning receipts. An owned component binding requires the existing run lock and
the specifically requested retained generation with exact file map; it cannot
bind a pruned artifact or authorize formal use. No new public qualification
helper that merely wraps an unconditional failure is needed in this increment.
Derive owned binding model, resources, inference binding and visibility solely
from the frozen run Config/execution; accept no caller overrides. Actual active
policy construction still performs C1 admission in the selected runtime.

Keep Generation1's retention cadence:0, multiples of10 and the current generation
remain retained; policy selection uses the existing last5snapshot pool. Before
deletion, verify exact live inventory/hashes and publish an intent listing every
owned original file except inference.json. Delete only those verified listed
files; publish a completed receipt only after all targets are absent. Preserve
immutable generation/update records and original maps. Partial pruning remains
an explicit incomplete intent and is rejected by healthy chain validation;
never call it completed. Foreign files, symlinks and mixed versions reject.

Full S2 review found a verify-to-unlink replacement race. Root adopts the advisor's
bounded staging protocol, accepted by architecture review before implementation:
public artifact basenames are untrusted; retention exclusively owns its new,
unpredictable mode0700staging namespace. Committed files have no concurrent content
writer through an existing FD/hard-link alias. Detect observed changes and fail
closed, without claiming isolation from an arbitrary same-UID actor that can also
mutate private staging entries. The run lock, parent-directory guards and held
directory FDs preserve this ownership; no generic deletion framework is added.

Create staging under the fixed owned `artifacts` parent and retain its dirfd.
Registry2's deletion_intent is exactly `{files, staging_basename}`; basename is a
validated unpredictable local name, never a supplied path. Publish status=deleting
and this intent before moving any target. Keep a verified original FD open (a
local dup may outlive SealedScan.file's unchanged-path context), atomically rename
the public entry to an unused name inside staging, and compare the moved regular
object's device/inode and expected hash against that FD. Verify stable metadata
during that check, accounting for legitimate rename ctime changes. Identical
bytes on a different inode still reject. Unlink ONLY the private staged basename
relative to its held dirfd; never unlink the public artifact pathname.

On mismatch, interruption or failed cleanup, stop and preserve the staged entry,
including any captured foreign file/symlink/directory, with incomplete intent and
failure diagnostics. Do not recursively clean it or automatically restore a public
name; restoration could overwrite a replacement. No automatic incomplete-intent
recovery is added. Earlier verified deletions may remain deleted. Healthy chain
inspection/binding rejects the incomplete state.

Publish the completed receipt only after every listed target has been verified
and privately unlinked, the artifact directory contains exactly hash-verified
inference.json, and the empty owned staging directory is identity-checked and
removed through the held parent dirfd. A replacement inserted at the public name
before private unlink survives and prevents completion. Receipt remains exactly
`{files: original_deletion_map}`; preserved intent names the now-absent staging
directory. Keep immutable maps and generation/update records unchanged. This adds
no power-loss durability claim beyond the repository's existing publication scope.

Required regressions cover pre-rename replacement (different and identical bytes/
different inode), post-rename public replacement, captured symlink/directory,
interruption after rename/one deletion, staging collision/substitution and ordinary
success/cadence. Retained model/resources/binding/RNG declarations must also match
the frozen run at commit and inspection. Carry a caller-owned snapshot scanner
through binding and pre-deletion/pre-publication validation; do not discard its
verified parent records. Required persisted digests are64hex before I/O. Publication
guards every generated parent as an owned nonsymlink directory before temp creation.

Verify all3model kinds with real C1 sharded component artifacts, no-Torch/no-
installed-version inspection, retained/pruned exact maps, corruption/extra files,
broken chains/counters/controls/clocks/health, lock ownership and failed deletion.
Retain all legacy checks. This bounded increment adds no numeric protocol,
coordinator loop, native session, LightGBM or real experiment.

## 20. PPO2 composition and process envelopes

Architecture design review accepted this bounded composition before implementation.
S2 conformance and later physical resource/experiment acceptance remain separate.

After S2 metadata stabilizes, connect its formats to the existing coordinator's
single commit/update loop. Keep public actions2 real-I/O/qualified-profile gates
before Simulator activity. A dedicated private component entry supplies only a
synthetic/FakeHTTP collector and fixes component/False-capacity classification;
it cannot authorize formal Source or expose a CLI bypass. Normal Environment
guards still apply if a caller attempts real I/O through that entry.

Use version-selected concrete helpers for model binding, numerical initialization,
collection projection, rollout publication and generation commit/retention inside
the common loop. Do not duplicate its schedules, clocks, storage admission, locks,
cleanup or pause/resume logic into another trainer. Preserve model1 default paths
and CollectionPlan serialization. A v2 collector receives `(plan, ledger)` and
passes that SAME runtime ledger to run_experiment/StructuredEnvironment; an owned
collection rejects an absent/mismatched ledger. Test actual object identity and
charge lifetime, not a producer flag. Standalone collect still has no learner.

Numerical worker wire3 dispatches before legacy decoding. Its closed initialize,
update/save and acknowledgement records carry model2 architecture, input binding,
action hash, resume identity and actual execution identity; parent validates them
before publication. Actor wire2 remains separate. Select child CPU/CUDA visibility
before importing Torch. Reuse C1 Learner/rollout3 and existing hardware() helper;
ordinary active resume retains strict Config/code/runtime/execution/hardware
equality. W comes only from model.temporal.sequence_length. No new latent cache is
stored in source or rollout, and the whole-batch frozen-policy parity check remains
before the first Adam step.

Each batch keeps16games, explicit learner roles, structured Random/heuristic
labels, existing seed namespace and schedule RNG,10update snapshot cadence and
last5selection pool. Check original schedule/policy bindings/parent generation
and every Source role against the coordinator's frozen plan. Keep zero-endpoint
episodes and reject whole interrupted cohorts; do not salvage healthy rows.
Build C1 raw EpisodeRecords/rollout3 from original component Source frames,
tables and values. Historical component-parent inspection reconstructs these
links rather than treating S2 scalar/hash validation as Source authority.
S1's own proof still says parent_qualification=not_performed; the composing
validator performs the extra checks itself. Engineering purpose remains
unqualified even for a complete healthy component run.
Historical correspondence checks retain the no-Torch/no-current-runtime boundary:
validate recorded declarations and original raw frame/bank/table links. Do not use
the active C1 `semantics()`/resume environment checks as historical Source authority;
keep active worker decoding strict and factor only the needed deterministic reader
validation if reuse requires it.

CUDA allocator caches can survive logical temporary-scope release. Use the full
configured managed endpoint envelope as a conservative process-lifetime charge:
worker `resource_estimate(..., suffix_decisions=1)` and learner full configured W
envelope. Reserve before Popen, keep through collection/replies and release only
after confirmed reap/no-process rollback. CPU contributes0CUDA bytes. Keep scoped
decision/update ownership even when its additional CUDA charge is0. The update
must begin with no participant leases. Do not force empty_cache after each action,
auto-change W/candidates/device, or interpret the conservative bytes as measured
GPU allocation. Record `structured-process-envelope-v1` in parent identity.
Ordinary v2 resume requires that exact recorded recipe; do not default-upgrade a
missing recipe. Model1's identity remains unchanged.
Default derived totals for4workers plus learner are about7.22/7.23/7.58GiB for
MLP/GRU/2-layer Transformer, under the unchanged12GiBbudget. Measure actual cached
allocator/context behavior and persistent reserved high-water across repeated
shapes and collection/update phases during later resource/stress verification; retain
explicit failure/cleanup for unavailable capacity and asynchronous errors.

Preserve7200wall/5400admission/7140execution seconds, max1000updates,10GiBowned
storage and1GiBadmission reserve, existing normal-boundary pause and no collection/
update overlap. New native capabilities remain a separate prerequisite: explicit
Python reference is truthful; missing Rust v2 support fails before collection.
No fallback or real-rule coverage reduction follows.

Verify same-binding two continuous updates against1+pause/resume for all3kinds,
including parameter arrays, individual Adam/RNG, schedules, artifacts and Source
links. Use actual configured defaults (Transformer2layers), original16-game
component batches and unchanged original limits. Cover CPU/CUDA inference,
cache/full/split tolerance, whole-cohort failures, mismatched config/visibility/
bindings, worker/learner cleanup, shared ledger/no-overlap, publication failures
and old model1 coordinator regression. These components establish integration and
resume behavior, not game feasibility, learning strength or final-test acceptance.

S3a first implements only numerical wire3, low-level process ownership, lifetime
envelopes and explicit version2 schedule labels; S3b connects the common loop and
historical Source correspondence. Root chooses one small concrete learner lease
in the existing ledger (not a process framework): reserve/spawn/confirmed-reap/
no-process rollback, no participant slot consumption, one learner at a time.
Attach its process owner before asynchronous child creation and preserve it across
cancellation. A null process pointer alone cannot authorize rollback while a
shielded creation task may still return a child. Keep the existing static
learner_bytes constructor behavior for old callers/tests; new v2 composition uses
the explicit lease and cannot double-charge that static amount.
In v3, a failed/incomplete close during pending creation is not terminal: once
creation resolves, a later close must retain a reachable single-flight cleanup
attempt under its caller-supplied bounded deadline, preserving cancellation and
exactly-once release. A successful terminal close forbids a subsequent start
before any reservation/spawn. Keep model1's legacy close defaults unchanged.
Root accepts the bounded training→client.structured_resources dependency needed
for this same concrete owner. That module stays stdlib/common only, with no HTTP,
environment, transport or training imports; dependency direction stays acyclic.
Revisit package ownership if it gains client I/O or additional unrelated consumers.
This avoids duplicating the ledger or moving process checks into pure common.

Reuse LearningProcess's cancellation-shielded async subprocess creation, drains
and close; add only private default-preserving hooks needed by its explicit v3
adapter. Wire3's closed identity contains model_id/version, architecture_identity,
action_config_hash, resume_identity, input binding, actual execution and the
existing execution_digest(model, binding, visibility), which also binds resource
settings excluded from architecture_identity. Initialize carries model/seed/output/
explicit nullable restore; update/save carries batch path/digest/frozen policy
version/output; close carries identity. Every request/reply has builtin integer3
protocol_version. Replies validate command/identity, full C1 artifact maps/digests,
optimizer counters, fixed declared hardware and finite timings/metrics before
publication. Unknown/malformed/error replies cannot advance state. Numeric wire3
dispatch precedes old decoding, and actual visibility is checked before Torch.

Keep the existing training_groups/validation_groups algorithm in
gf_training.collection and ordinary schedule algorithm in gf_common.schedule;
select structured baseline labels by explicit actions version with1default.
Preserve seed draws, pair order, roles and16games; do not copy the algorithm.
S3a tests actual CPU/CUDA child lifecycle, exact C1 continuous/resume state,
pre-import visibility, malformed acknowledgements, creation cancellation/cleanup,
lifetime/no-overlap and legacy schedule/process defaults. It adds no coordinator
loop, Source authority or real-I/O bypass.

S3b first factors only decoder/correspondence and original-Source snapshot
ownership. Keep public active read_rollout's signature and exact installed-runtime
checks; put its existing deterministic decode body behind a private core. A
separate private historical wrapper accepts S2-validated generation inference
declarations and exact original EpisodeRecords. Compare recorded rollout
semantics/runtime with that historical declaration, not the inspecting machine.
Validate every original bank/event byte, frame payload/revision/interval/hash,
full table column/hash, Selection and closure in ordered lockstep, including
zero-frame/interrupted records. Existing deterministic token/GAE reconstruction
then establishes correspondence; do not tokenize Source twice or compare latents.
Reuse original records after a successful per-episode comparison rather than
retaining a second complete raw Source copy.

A narrow private Source snapshot handle/context retains the existing reader owner
through numeric decoding/publication and reruns its original final checks afterward;
never replace the original identities with a fresh baseline. Keep the existing
inspect_component_collection return/proof behavior for old component callers.
Joint admission includes retained Source objects plus decoder arrays/scratch under
one deadline/budget, not independent full allowances. Structural correspondence
preserves every closure; the learning caller separately rejects interrupted
cohorts. This checkpoint adds no common runner or Source qualification.
Where shared sharded-array loops otherwise lack a caller deadline, add only an
optional private `_check` callable through the existing bundle inspect/read/write
and underlying array save/load/chunked-hash helpers, with None as the unchanged
default. Poll per file/shard, chunked hashing and archive-member/load/save
boundaries so one padded on-disk NPZ or a whole multi-shard call cannot bypass the
caller deadline;64MiB is a decoded cap, not an encoded-file bound. The callback is only an
injected check; it adds no upper-package import or artifact-format change, and
is not a native per-batch Python callback or a general I/O framework.
Similarly keep public write_rollout unchanged and factor only its existing private
writer core to accept extra retained Source bytes plus caller cap/check. Admission
covers planning/packing/publication under min(recorded model cap, caller cap),
including Source-owned objects not already counted by rollout residency. Do not
relabel model/resources to force a lower budget or count shared objects twice;
shared subtraction requires actual reused object identity, not equal contents.
Injected deadline TimeoutError must remain a deadline failure even where existing
array loading catches OSError; preserve the old None/default exception behavior.
The private historical decoder also pre-admits encoded archive/central-directory
scratch from verified regular-file sizes (conservative32times largest encoded
NPZ, consistent with S2) alongside retained Source and decoded/reconstruction
ownership, before ZipFile/NumPy opens. A64MiBdecoded bound alone is insufficient
for a padded archive. Preserve the active reader's existing default behavior and
reject size/path changes across this admission; do not adopt a new baseline.

For the later two-argument collector, derive owned runtime context from the locked
parent and frozen plan plus the supplied ledger's concrete learner owner/identity.
Reject missing/mismatched ownership before collection and pass that exact ledger
through CLI composition. CollectionPlan remains a pure serializable value; no
ownership token or runtime object reaches participant settings. Common-loop
composition starts only after this correspondence checkpoint stabilizes.
When the composing historical validator already retains sealed parent metadata,
pass its ownership as private extra_owned_bytes (default0) into Source snapshot
admission BEFORE the first Source JSON parse. Apply the one caller cap/deadline
without relabeling resources or double-counting shared objects. For live updates,
check originals before Adam and retain/recheck Source through update completion
before Generation2 commit; a later mismatch invalidates the whole parent and
leaves the new generation uncommitted, never salvaged as a healthy resume point.
Component collection evidence must never patch central qualification guards to
no-ops. Keep public run_experiment and public/direct Environment constructors
unchanged and guarded, including the existing run_group and _episode wrappers.
Factor only private initialization/episode/group/common collection bodies
and private server/client construction hooks; a test-owned subclass initializes
through that private path and supplies fixed FakeHTTP/scripted doubles. The
private test collector injects that subclass into the shared private collection
runner. Injecting a normal Environment still invokes its ordinary guarded
constructor. No public component flag, authority token or real-I/O bypass is added.
Fresh integration runs under this unpatched setup replace earlier patched-gate
functional smokes as acceptance evidence.

## 21. Native structured controls and staging increments

This is the bounded implementation decomposition of §§14–15, following S3b2.
Independent architecture design review: READY for C2a and the C2b direction
(2026-11-02). The concrete C2b batch ABI still requires its shape/dtype/owner review.
It preserves the existing parser, reader and recipe identities and all legacy
NativeSession/NativeBatch payloads. No candidate-rule port, new crate, Torch
dependency or Simulator access enters either increment. The full structured
build markers stay ABSENT until controls, Journal5 tokens and actual export
work end to end; merely parsing a model2 record cannot activate the Rust reader.

C2a implements original Run2/Generation2/Update2/Registry2 validation in additive
structured_controls/structured_session Rust modules, with a small Python owner in
native_structured. Rooted nonsymlink inputs, original file/inventory snapshots,
bounded parser/scratch ownership, cancellation and final rechecks follow the
existing Sources/Budget/Control/NativeExecution patterns. Retain snapshots across
Python authorization and every later batch; close/cancel joins native work before
releasing locks. Cross-version validation dispatches before legacy code and does
not weaken legacy equality or fingerprints. V2 requires exact integer JSON types,
closed maps and explicit classification, not Python-style bool/float equivalence.

Rust checks the original full Config/hash, recorded provenance, model/resource/
execution/semantic declarations, healthy run state/clocks, generation/update chain,
counters, complete explicit artifact maps and exact retention intent/receipt.
It compares extracted values with their frozen Config originals. Do not serialize
effective_config into IPC or have Python reopen it. The diagnostic C2a owner
exposes only an explicitly named structured-parent-controls result and final
recheck; it does not claim J5 projection, source-set verification or qualification.
Do not add placeholder authorization/export methods or advertise the final reader.

C2b completes the separate StructuredSession/structured batch path. One bounded
original Source scan validates Journal5 append banks, frames, tables, positive
acknowledgements, closures and sidecar inventories, then retains typed scratch for
summary→authorization→export. It produces exact raw category/numeric/presence
banks and ordered compact references, reusing IntegerOwner/ByteOwner and bounded
little-endian numeric I/O. It never calls a Python model or stores learned latents.
Existing FloatOwner is f64; an FP32 bank requires an additive correctly typed
owner, never reinterpretation of that owner or legacy NativeBatch.
Python retains GAE/targets and rollout3 publication; factor only the necessary
existing C1 publication primitive to avoid reconstructing/retokenizing a second
complete Source copy. Source roles/outcomes and baseline-null versus neural-finite
policy values remain explicit and are not inferred from syntax.

The final closed summary2 contains schema/kind/reader/scope, explicit action
contract/purpose/capacity and qualified=False/parent_qualification=not_performed,
domain-separated control fingerprint, bounded authorization-wire size and parent/
collection arrays. Each parent contains its directory; compact state/counters/
absolute clocks/config and resume digests; one bounded pure model/resource
descriptor; recorded learner/inference binding and visibility; actual semantic
declarations/action hash; and generation summaries with original record/link/
artifact-map hashes, retention status, counters and parent_controls. A pruned
generation additionally carries its bounded original_files map and exact deletion
intent/receipt, sufficient for S2's existing pruned inspector; their ownership and
wire sizes count toward the same control cap. Each collection
contains its directory/parent index, split/kind/generation/batch/run identity,
explicit model/input/journal/feature/generation/rollout versions, actual window/
token signature, original schedule/binding/source-set hashes and episode/both-view/
learner counts. No full Config or unbounded row arrays cross that control boundary.
Derive token signatures and counts from validated originals, not request labels.

Rust does not acquire a ZIP/NPY implementation to duplicate C1's sealed artifact
inspector. Before authorization, Python runs existing NumPy-only S2 validation on
referenced artifact directories, checking exact declarations/file maps against
the compact Rust result. Admit the Rust session's retained owned bytes, Python
summary/control ownership and S2 parse/encoded/decoded scratch under ONE caller
cap/deadline. Expose a concrete retained-owned-byte observation on the paused
session if needed; a separate full allowance for each owner is invalid. Both
owners retain/recheck their original identities through final export. When native
work resumes, keep Python summary/sealed-snapshot ownership charged by the existing
retained_python/ControlMemory/StagingLease mechanism (or an equivalent concrete
retained-byte argument to authorize/export/finalize). An observation of paused
used_bytes alone cannot release Python ownership or create a second full allowance.
The closed
authorization binds successful sealed results and schedule decisions to the
exact original control fingerprint. This is an explicit division of validation:
Rust checks metadata/maps/bytes, Python's existing S2 checks NPZ layout/finite/
Adam details. No check is skipped or represented as a native numeric check.

Python authorization also uses validated seed/model/controls for PCG64 seed
anchoring, snapshot-pool evolution and unchanged schedule reconstruction; Rust
checks that extracted seed equals frozen Config. No Rust RNG/SeedSequence port or
callback from a native per-record loop is introduced. Neither authorization nor
successful numerical validation upgrades engineering data eligibility. Public
real-I/O/qualified-source gates remain unchanged throughout.

Before activating markers, verify native/Python original metadata, raw tokens,
presence/order/hash/counts and rollout3 differentials on owned component inputs.
Cover changed original inventories across batches, broken chains/config/provenance/
PCG, retained/pruned/partial artifacts, malformed sidecars, padded JSON/NPZ,
deadlines, symlinks/FIFOs/replacements and cancellation/lease cleanup. Keep actual
18-reference records separate from synthetic128-capacity tests. Run old Rust/
corpus/index/saved-PPO proofs in fresh directories and build an offline locked
wheel, preserving and identifying the previous loaded binary. Source HEAD alone
is not evidence that the tested wheel contains it. Final design and conformance
reviews apply to each concrete increment; C2a alone is not native completion.
