# 0024: Adopt the card-instance Simulator v2 contract

- Status: Accepted design for staged direction and stage1 contract/wire/Config/
  direct HTTP (2026-11-03). Later dependent stages retain their scoped gates.
  Prior ADR0023 engineering evidence is historical.
- Owner: root. Source: the user's server-v2 migration and acceptance plan.
- Simulator changes: none. No new third-party dependency.

## 1. Requirements and authority

Adopt Simulator `07e8185f98c503aa41200a80ea554b4fd21088ad`, rule set
`true-match-1v1-v2`, Gameplay `/v2`, catalog schema2 / `public-ja-v3`, catalog
SHA256 `f51a3c73feb8b3f01837e65768aafb5a3aa805a33483308130244b33b16a8ad6`.
The canonical path-to-content-SHA256 inventory of the thirteen normative files
(spec00–10, rule-verification register, classification-v2) has SHA256
`8381079eae8a68378d8419c7f7245c62358833f0dd9045667021e112a1e9bad5`.
Record the actual executable SHA separately; source HEAD alone cannot attest a
previously built executable. Use an existing verified TestServer executable and
new run-owned data directories, never rebuild/modify the Simulator repository.

Normative v2 override sections supersede historical examples: spec05§27,
spec07§47 and §21.1, spec09 and `card-definition-classification-v2.md`.
Public DTO code is evidence for representation only. Engine internals,
SequenceOptions and rejection details are not rule or policy authority.

Preserve v1 and engineering actions/model2 default objects, serialized bytes,
hashes, meanings, records and artifacts. No old weights or data acquire new
semantics. Formal multi-seed comparison, HPO and final-test evaluation are future
work. All acceptance failures remain evidence; never shorten an acceptance
budget, alter a failed seed or narrow the candidate domain to claim success.

## 2. Responsibilities and dispatch

The package dependency direction in ADR0023§2 remains. `gf_common` owns pure
contract validation, participant records, candidate semantics and deterministic
tokens. Client owns attested transport and participant process lifecycle;
models own encoders/inference; training owns Source authorization, learning and
publication; CLI owns composition. Native owns deterministic validation and
staging only. The existing resource ledger, PPO coordinator, neural architectures,
GAE/Adam, sharded numerical storage and Tree fitting workflow are reused.

A frozen common `StructuredLayout` is selected from the exact closed actions /
model / contract tuple at configuration load and explicitly carried to consumers.
It describes recipe, vocabulary/codebook, token dimensions/signature and format
versions. It does not contain I/O, learned state or simulator legality. Preserve
legacy globals as aliases for the old layout; no process-global switching,
fake numeric card IDs, source rewriting or v2-to-v1 Observation normalization.
Add layout arguments to shared numerical kernels rather than copy a learning
stack. A consumer rejects a mismatched identity before model/array/process
construction. Native support remains undiscoverable until its complete reader,
tokenization, correspondence, staging and persistence implementation is verified.

New closed `CardInstanceContract` metadata contains the exact profile
`card-instance-simulator-v2-20261103`, simulator revision, rule set, API,
card-reference scheme, catalog schema/version/hash, normative inventory hash,
policy-input version and structured-input recipe. All fields must match the pin;
there is no user-editable eligibility flag. `ActionConfigV3` is a distinct type
with a required explicit contract and the existing search budgets. New server
and neural/Tree model configuration types dispatch before merging version-specific
defaults. Do not add default fields to old Config dataclasses. New baseline
random/heuristic policies may retain their algorithm Config identity while their
worker/input/action identities bind the v2 layout.

The exact contract fields are `profile_id`, `simulator_revision`,
`rule_set_version`, `gameplay_api`, `card_ref_scheme`, `catalog_schema_version`,
`catalog_version`, `catalog_sha256`, `spec_inventory_sha256`,
`policy_input_version`, `input_recipe`. API is `v2`, card-ref scheme is
`aes256-card-ref-v1`. Preserve the public representations rather than coercing
version fields:

| Public field | Representation |
| --- | --- |
| Observation / ActionOptions / public-event schema versions | string `"2"` |
| HTTP typed Action schema / catalog schema version | integer `2` |
| RL Dataset episode Action schema version | string `"2"` |
| management `/debug/v1` info schema | integer `1` |

`ServerConfigV2(ServerConfig)` overrides only the revision default and adds
`gameplay_api="v2"`. Require that discriminator and revision explicitly in raw
or inherited server settings, and match the actions3 contract. The full contract
is not duplicated in server settings. Legacy defaults and conversions remain
unchanged. Stage1 leaves Python/legacy preprocessing settings inert and rejects
all actions3 collection/evaluation/training/worker/persistence entry points before
allocation or I/O; only Config validation and direct attested v2 HTTP APIs are
available. Do not advertise a native or structured-v2 reader before it exists.

| Boundary | New identity |
| --- | --- |
| actions / neural / Tree Config | `3` |
| policy input / structured recipe | `player-card-refs-v2` / `participant-structured-v2` |
| generator | `public-card-instance-diverse-v5` |
| participant protocol / candidate columns | `3` / `candidate-columns-v2` |
| journal / derived / collection | `6` / `6` / `5` |
| run / generation / update / registry | `3` / `3` / `3` / `3` |
| numeric rollout / learner wire | `4` / `4` |
| neural inference / resume | `4` / `4` |
| Tree inference / training / request | `4` / `4` / `4` |
| fixed Dataset manifest | `3` |

NumPy bundle1, bank dtypes, file hashes and named Optimizer encoding retain their
versions where meaning is identical. New input semantics require new signatures.

## 3. Participant data flow and information visibility

`own.cards` is the only possession sequence, in observed order, at most18 slots.
Preserve artifact/miracle kind and unused/learned state, same-definition distinct
instances, and ordered duplicate opponent learned IDs. Frame-local reference
tables map opaque `card_ref` strings to slots; selected actions reconstruct the
original refs from the immutable actor frame. Ref bytes/magnitudes/hashes never
enter features. Position is observed possession position, not acquisition time;
attention receives no arbitrary position encoding. Public classification/use
roles and kind/state become explicit token fields. Definition interpretation
uses only perceived definitions allowed by the pinned own-view catalog contract.

Keep all25 public Event kinds including `safe`, observation-time definitions,
participant-local append-only prefixes and original raw facts. Never rewrite a
past frame after Dream changes. Events lacking a ref do not gain a synthetic ref.
`candidate_groups` is retained in raw records, but arbitrary group strings are
not semantic features: role rules, flat `usable_cards` and choice-specific fields
are the public candidate basis. Actor/revision/reference/definition checks repeat
at process and persisted-source boundaries. Another viewer's data, management
fields, GameSeed/RNG and future facts remain outside policy input.

## 4. Public candidate rules

Use one pure role/payment/order implementation for prefix pruning, completion
checks and seven-channel diversity summaries. CostCut selects only the immediately
preceding miracle instance and suppresses the support card's other roles for
that selection. A nominally unaffordable miracle prefix may extend via immediate
CostCut; only complete sequences require total actual estimated MP payment.
Do not reuse the old whole-sequence-free predicate. Cover distinct normal weapons,
additional attacks, same-definition separate miracle instances, wands, Aura,
Mirage and attackTwice using normative public rules.

Freeze a reviewed rule/Action-family matrix before generator implementation:
utility singleton, status miracle+adjacent CostCut, numeric defense, ring and
exclusive response, Discard/Sacrifice, trade, self-resource exchange and phenomenon.
`play_sequence`, `sell`, `buy` have explicit non-null targets; exchange has none;
responses use `respond`; `invoke_phenomenon` is separate. Server owns automatic
effects, HP0, refill and RNG. Unanswered public rules remain open questions and
cannot be resolved by rejected actions or internal handlers.

Reuse saturating pre-count, bounded length-band search, mandatory basics and
singletons, seven diversity channels and25-interval exchange grid. Budgets remain
18 game references/model positions (128 only explicit synthetic stress),32768
candidates,262144 expansions/temporary proposals,50ms/16384 counting states,
1s search/selection, scoring chunk256,10s participant timeout. Resource guards
fail explicitly rather than silently resize the experiment.

## 5. Transport, storage and failures

Reuse `/debug/v1` attestation, sticky poison, deadlines, command UUID ownership,
bounded resend and cleanup. Explicitly select the Gameplay v2 adapter; it uses
the new pure validators and `/v2` routes while the v1 adapter keeps its contract.
Validate management identity, public info schema and full pinned catalog before
Create; repeat the management fence after Create before Join. A concurrent
attestation failure invalidates a pending successful response. Serialize and
retain the same ordered Action payload, command ID and expected revision for
resends; never regenerate an Action from already-consumed cards.

HTTP Observation, Replay Observation and RL Dataset Observation are distinct
closed representation validators sharing only appropriate field validators.
Replay uses `decisions`, `visible_action`, and aborted closure; Dataset declares
Action schema as a string. Preserve natural, truncated, aborted, unsupported and
infrastructure outcomes. No synthetic transition/return for interrupted runs.

Every Source/artifact/evaluation binds the full contract, layout/token/candidate
signatures and actual source hashes. Component versus qualified source authority
is checked from complete original evidence, never a mutable flag or the mere
new contract. Retain existing private/source/process ownership, original-file
rechecks, admission, cancellation and inode-owned final-publication revocation.
All encoders, named Adam state and RNG states resume at normal update boundaries.

## 6. Stages and verification

Each dependent increment needs design READY before implementation and independent
correctness/conformance reviews before acceptance. Root owns Config/presets/docs
and commits. One implementer owns assigned source/native/tests at a time.

1. **Contract/wire/Config:** distinct immutable contract/actions3/server-v2 types,
   pure public validators and explicit shared HTTP adapter. In this first slice,
   random/heuristic Config only; model3 and collection/worker3 fail explicitly
   until subsequent implementation. This is a partial milestone, never full
   migration acceptance. Pin public fixtures, malformed/enum/omission tests,
   stale/foreign refs, late-success/poison and unchanged v1 HTTP/config proofs.
   Then existing-runtime read-only info/catalog checks and owned Create/Join/
   Observe smoke. Later slices replace temporary explicit unsupported gates.
2. **Candidates/frames/tokens:** reviewed coverage matrix, opaque ref frames,
   layout/codebook2, Python reference tokens and independent small-space enumerator.
3. **Models/workers/journals/native:** shared-layout neural/Tree features,
   original-frame reconstruction, protocol3/journal6, exact Python/native parity
   and admission before publishing native capabilities.
4. **Source/learning/persistence/evaluation:** common PPO orchestration and Tree
   workflow; paired source/numeric facts, full resume, mixed-version rejection,
   failures/reap/publication and old-data proofs.
5. **Runtime acceptance:** fresh dedicated runs;192games with the fixed32 GameSeeds
   (hex suffix1..20), sampling17/29/43 and seat swaps; seed17 attention PPO for
   MLP/GRU/Transformer × CPU/CUDA inference, CUDA learner,16games/update with
   continuous2 versus pause1/resume2. CPU learner numerical checks+MLP smoke;
   direct/DeepSets real16game MLP each and all-model numerical checks. Preserve
   7200/5400/7140s budgets. Component Tree fixture uses all3 complete parents and
   prescribed validation records, fresh fit/paired artifact/worker/RAW, without
   acquiring qualified Source status. Preselect512train-side natural-acceptance
   decisions with seed17 and measure all3 NN+Tree on identical actor inputs.
   CPU ordinary p95<=3s;2games/4workers; separate startup/HTTP/stress. Keep12GiB
   GPU and64MiB decoded array/shard limits. Full pytest/Ruff/mypy/Rust and needed
   CUDA checks, old evidence, wheel source/binary proof, isolated baseline startup
   without optional dependencies, local verified install and coherent commits.

Acceptance ledger must record exact commands, PASS/FAIL/NOT RUN, source/runtime
identities, limitations and failed artifacts. Synthetic/engineering evidence
cannot be presented as real acceptance or formal learning parents.

## 7. Alternatives, consequences and open questions

Rejected: switching legacy constants; normalizing refs into integer IDs;
duplicating PPO/model stacks; permitting old native readers; upgrading engineering
records by flags. Explicit layouts add plumbing but isolate semantics and preserve
historical reproduction. Keep wrappers only at genuine format boundaries.

Before dependent generator work, review the exact public role/target/order table
and distinguish normative uncertainty from field representation differences.
Fog `ascended` follows public DTO; invisible normal status is not filled in.
Before runtime acceptance, verify existing TestServer binary/public contract;
binary/source correspondence is a separate recorded evidence question. Before
layout/model work, freeze exact new widths/signature payloads against public
fixtures. Before native activation, prove complete new recipe support. These
questions are gates for their dependent stages, not invented public rules.

## 8. Review and acceptance dispositions

- Initial direction and stage1 architecture review: READY after correcting a
  blocking blanket integer-schema assumption to the explicit public version
  representation matrix in §2. Root accepts the correction. Reviewer confirms
  exact13-file inventory and closed server/actions dispatch. No runtime check was
  performed by the design review; production conformance remains separate.
- Stage1 implementation accepted at scoped contract/wire/Config/direct-HTTP
  milestone. Both final correctness and conformance reviews are READY after
  resolving direct environment/worker pre-allocation gates and preserving raw
  rejected Dataset attempts separately from submission constraints. The later
  test-only checkout-origin normalization keeps all12original hashes and passes
  independent review. Scoped tests/static/source-wheel/runtime-smoke evidence
  and the interrupted whole-suite limitation are in the verification ledger.
- Stage2a §9 DESIGN READY after explicit validated Delta initialization,
  old-codec/default-layout gates and pinned own-catalog subset authority.
  Root accepts those corrections. Implementation/conformance remain pending.
- Stage2b generator and full runtime acceptance remain blocked by issue0007.

## 9. Stage2a: shared layout, original frames and deterministic tokens (proposed)

This section is a separate pre-implementation gate. It does not enable a worker,
model, native capability, collection or Source consumer. `gf_common` owns the
layout and pure records/tokens; no upper-layer or third-party dependency is added.

Use one frozen `StructuredLayout` with two closed active instances (legacy
structured-v1 and card-instance-v2). It contains recipe/policy-input/generator,
catalog/revision/contract identity, exact categorical vocabularies, numeric field
tuples, Action kind/target codebooks, history recipe, card membership/order field
indexes and computed widths. It also names the format versions from §2. The
historical62 aggregate remains recorded-only under its existing signature.
Only exact known descriptors are accepted; arbitrary caller-supplied widths or
semantic tables cannot bypass validation. Pure `layout_for_actions` and
`layout_for_model_identity` perform closed dispatch. No global runtime switch.

Preserve the old serialized record and Config fields. New record construction
must explicitly carry the selected immutable layout through EventDelta/Bank/View,
DecisionFrame, compact table and tokenization. Use thin internal frozen dataclass
subclasses for new EventDelta/EventBankView/DecisionFrame, with no new fields and
an unannotated read-only `layout` property. Existing concrete classes report the
old layout. This retains the exact field set/asdict/old equality semantics;
new concrete equality and ordinary Frame/View `dataclasses.replace` preserve
layout. The EventDelta subclass must use `init=False` to inherit the validated
public `events` constructor, rather than generate an unchecked `_events`
initializer. Delta replacement stays unsupported, matching the existing custom
constructor; rebuild a delta through `event_delta` instead. No new raw-bytes
construction bypass is added. Do not use
InitVar: it enters `__dataclass_fields__`, is lost by serialization, and replace
can silently restore a legacy default on a new record. Bank itself is an ordinary
owner holding an explicit immutable layout. Every append/prefix/view/frame/table
boundary checks same known layout; View.prefix and Bank.view preserve the proper
concrete type. Existing calls default to old layout and keep exact old hashes.
New event-bank initial hashes use the new recipe. New frame hashes bind
`input_recipe` and `policy_input_version` as well as original actor/episode/
revision/payload/event prefix. No new frame can be reconstructed from a
v1-normalized Observation.

Pure factories are `event_delta(..., layout=old)`,
`DecisionFrame.from_input(..., layout=old)`, `parse_delta(raw, layout=old)` and
`parse_frame(raw, layout=old)`; `EventBank(..., layout=old)` selects its View
factory. CandidateTable needs no extra subclass/field: its exact contract union
selects the layout. `tokenize_frame/table` and raw-token validators take an
expected layout (old default), ensuring model2 cannot silently accept new widths.
The bottom descriptor module imports only pure vocab/history/domain/contract
data, never structured records/tokens/models. Its closed selector accepts scalar
identity fields/contract; config-level actions/model adapters perform object-type
dispatch above it. Do not introduce a layout↔record/token dependency cycle.

New frame references are ordered original `card_ref` strings from `own.cards`.
Validate original metadata, same-revision actor hints and the catalog definitions
actually visible in that frame. Perceived definition and card kind/state must
be consistent with the pinned public catalog classification. Store raw
`candidate_groups` without semantic interpretation. Opponent learned IDs remain
an ordered duplicate-preserving sequence. Keep original complete-event-input
versus event-delta meanings distinct, advancing intervals only at own decisions.

Pure per-definition authority is an immutable ID→canonical-definition-SHA256
table generated from the verified public-ja-v3 catalog and stored with common
vocabulary data. It stores no hidden identity or unobserved numerical definition
values. A participant catalog must contain exactly the distinct IDs visible in
that original own.cards, each full supplied perceived definition matching its
pinned digest. After that check, validate own kind=`miracle` iff catalog kind is
`miracles`, every own definition has class=`owned_card`, and miracle-only state
is present/valid. The digest table/checks are validation-only; validated public
classification/subclass/use-role fields remain the declared features below.
Only those supplied visible definitions feed token values. The whole-catalog
HTTP validator is not misused on an own-visible subset. Historical Event IDs
need only the public categorical vocabulary and do not fetch extra definitions.

Compact tables derive the layout from the exact contract type, preserving the
old table dataclass fields/content hash. Codebook2 uses
`pray, play_sequence, respond, discard, sacrifice, exchange, sell, buy,
buy_decision, invoke_phenomenon` in that exact order; relative target categories
remain `omitted,null,self,opponent`, with null/omitted rejected for targeted v2
families. New reconstruction returns the original refs and exact public v2 field
names. Selection uses the immutable original frame and table digest. Separate
syntax/ref validation from public candidate legality estimates.

Gameplay rows have at most18 distinct references. An explicit component-only
synthetic table may stress128 positions with repeated slot indexes, but new v2
synthetic tables cannot reconstruct a submit-ready Action. Worker/qualified
paths reject synthetic tables before selection/Act. This does not alter old
engineering synthetic semantics.

Before extending the table contract union, guard every unchanged codec1 consumer:
`candidate_columns.encode_candidate_columns` must reject a new-layout table before
byte allocation. The future codec2 is not advertised/implemented by stage2a.
Audit raw token/wire consumers for the same accidental old-codebook use. Pure
frame/delta/table conversions take an expected layout (legacy default); the new
outer protocol/journal version will explicitly pass the new layout later. No
new table may become codec1 bytes because its common Action kinds happen to
exist in both codebooks.

Token tuple container shapes can be reused; new exact signatures identify their
changed widths. Reference strings never enter token buffers. PAD0/MISSING1,
signed-log1p FP32, masks and numeric presence remain unchanged. The full public
v3 catalog has the same296definition,60ability and8category identifiers as the
old vocabulary (independently checked against the actual public catalog); the
new signature binds the new catalog/contract regardless of identifier equality.

| Token section | New exact recipe |
| --- | --- |
| state | existing26numeric fields/categories/18ordered opponent learned slots; width5499 |
| card numeric | `attack,defense,ability_value,price,mp_cost,hit_rate,additional_attack,usable,possession_position,exchange,sell,buy,phenomenon`, followed by13role flags below;26fields |
| card category | first6old categories (definition,ability,catalog kind,attribute,disease,guardian), then instance kind,miracle state,definition class,miracle subclass,membership;11columns |
| interval Event | existing24kind order plus `safe` at end; all other public fields unchanged; width395 |
| history | `public-exact-card-refs-v3`:25event counts,16actor-role counts,22numeric sum/presence values, existing314exact definition/window/actor fields; width377 |
| candidate | new Action codebook2 and unchanged target/numeric fields; width26 |

The sorted role flag order is `additional_attack,armor,cost_cut,
miracle:additional_weapon,miracle:attack_support,miracle:defense,miracle:status,
miracle:utility,miracle:weapon,numeric_defense,trade,utility,weapon`.
Flags reflect catalog `use_roles` only, never unknown candidate-group keys.
Instance-kind vocabulary is `artifact,miracle`; miracle-state is `unused,learned`
(missing on artifacts); definition-class is `owned_card,automatic,operation`;
subclass is `weapon,additional_weapon,attack_support,defense,status,utility`;
membership is `artifact,miracle_unused,miracle_learned`. Preserve the existing
category vocabularies/order for the first6columns. Card direct width is478.
Observed possession position is one-based `own.cards` index, present for every
owned slot; it replaces old learned-order semantics at numeric index8. It is
neither inferred acquisition time nor an attention position embedding.

New token signatures include the complete contract, exact tuples/codebooks,
history field list/actor fallback,18slots, numeric transform and masks/presence.
Validation receives the expected layout or resolves an exact known signature
with its declared aggregate window, then checks agreement with the caller's
layout before arrays. No unknown-signature fallback. Models/admission will use
these same dimensions in stage3; they remain closed in this slice.

Omitting the expected layout always means strict legacy permission, including
its recorded-only62 path; signature discovery never grants new-layout permission.
Test new tokens rejected under omitted layout, new table rejected by codec1
before allocation, and new synthetic `to_action(..., validate=False)` refusal.

Verification: small original-v2 fixtures, duplicate definitions with distinct
opaque refs, unused→learned same-ref movement across own frames,7/12learned and
18slots, Fog/Dream missingness, unknown/foreign/stale frames/events, `safe`,
historical perceived definitions and duplicate opponent order; exact new
signature/width/Python-token goldens, candidate-order equivariance and Action
round trips. Compare old EventBank/frame/table/token bytes and signatures to
frozen prior fixtures. No native, Torch or real-game eligibility claim follows.

## 10. Stage2b: public sequence matrix (proposed, review still required)

Use a single pure sequence-analysis result for extension, completion, payment and
diversity summaries. The shared bounded search machinery retains all budgets;
only its version-specific public space/completion/summary policy changes.
CostCut is assigned before counting effective weapon/defense roles. A prefix
whose only unpaid cost is its last miracle remains extendable when a distinct
hinted adjacent CostCut can still fit; complete rows must pay the actual total.
The last miracle cannot waive an earlier nonadjacent cost. Never probe the
server to settle a sequence predicate.

| Family | Supported v2 public domain / completion | Authority |
| --- | --- | --- |
| base offense | one normal weapon then additional attacks; additional-only; same-definition distinct normal miracle instances, with no ordinary weapon or different normal miracle; preserve order | spec09§§9.1/9.4/16.1 |
| support | Aura doubles accumulated attack; Wand affects later additional attacks; Mirage resets attribute/Wand and multiplies base rounds by max(1,count); attackTwice+Mirage permitted | spec09§§9.2/9.4/16.2 |
| adjacent CostCut | only previous miracle cost0; support card contributes no ordinary role/value/effect; Staff/armor retain ordinary role when not selected as support; Doll alone incomplete | classification§§3–4; spec09§17.5.3 |
| dynamic weapon | miracle payments first; Stick attack2×remaining MP and consumes remaining MP, including valid MP0; self/opponent explicit | spec09§13.1.1 |
| utility | each listed scoped utility singleton; miracle singleton may append adjacent CostCut; allow empty/full resource/zero-removal effect; guardian summon may overwrite | spec09§§17.5.2–5 |
| status | exactly one status miracle plus optional immediate CostCut; explicit self/opponent; hostile zero-damage status has matching response | spec09§§16.4/17.5.5 |
| defense | numeric armor, rings and reviewed numeric-defense weapon roles; exclusive stop/reflect/deflect excludes other defense effects but permits adjacent CostCut; Flash restriction applies; respond ordered refs only | spec09§14; classification§4 |
| discard/sacrifice | nonweapon instance subsets, no duplicate refs; normal Discard no refill, End Times Sacrifice one replacement per card | spec09§11 |
| trade | Sell exactly carrier+distinct eligible sold ref with explicit target; Buy carrier+explicit target then separate buyer boolean choice; exclude learned miracles from sold/bought candidates | spec09§§16.1/17.2–3 |
| exchange | own HP+MP+money total conserved, all resulting values0..99, HP0 allowed, no target; existing25interval grid | spec09§17.1 |
| phenomenon/pray/allow | explicit hinted phenomenon card with no selected outcome; hinted Pray; empty respond when can_allow; no invented fallback | spec07§47; spec09§§10/20 |

The analysis estimates only current public effects and potential acquisition;
server retains automatic HP0/refill/RNG/outcomes. Remaining-hand and refresh
summaries respect learned instances staying in18slots and role-suppressed support
cards. No old whole-sequence-free predicate is shared into new analysis.

Open questions to resolve before generator implementation: exact Flash counting
for miracle+CostCut, learned-card eligibility in Discard/Sacrifice, and whether
every scoped utility accepts both explicit targets under the public direct-use
contract. Inspect only public normative text/DTOs; an unanswered public rule
requires a separately reported Simulator contract task rather than an invented
rule. Moon-God standalone Mirage remains server-owned, outside Player candidate
generation; its unresolved base attack does not authorize a fabricated action.

Independent advisor review confirms the three predicates are not determined by
the pinned public contract. Root accepts that finding and records the required
separate Simulator clarification in [issue0007](../issues/0007-card-instance-v2-public-candidate-contract.md).
Stage2b and dependent complete real-game acceptance are BLOCKED on that explicit
contract adoption; independent stage2a remains unaffected. No conservative
candidate-domain reduction or private-rule lookup is an acceptance workaround.
