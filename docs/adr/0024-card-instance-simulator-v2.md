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
  Root accepts those corrections. Implementation and both final source reviews
  are READY, including a call-local layout reuse fix restoring table validation
  cost without fields/caches/semantic changes. Root accepts the scoped stage2a
  milestone after2181whole HOST passes/55skips, static/Rust/old-data proofs,
  source-matching wheel and isolated optional-free baseline checks. Exact
  conditions and limitations are in the verification ledger.
- Stage2b generator and full runtime acceptance remain blocked by issue0007.
- The later owner confirmation supersedes that ambiguity gate: section11's
  pure Stage2b design is READY after independent review. Root accepts explicit
  rule-signature separation, possession-only discard selection and pre-extraction
  legacy goldens. Real qualification remains closed on issue0008's independently
  confirmed mismatch; no source/worker/model/native gate is lifted.
- Stage2b pure component implementation passes independent correctness and
  conformance reviews. Root accepts390affected regressions, exact old-git
  generator goldens,32independent small-space cases, source-matching125file wheel
  and optional-free installed smoke. Rule signature is
  `1480f832b98c0b78b894a823681848b1dc11093b3a6a2aa462dd2069ed579a36`.
  This does not constitute full migration or corrected-runtime acceptance.
- Later server-fix verification (2026-11-04): issue0008's exact failing case now
  passes on both corrected debug/release binaries at91d5926. Independent runtime
  audit is READY and focused server tests pass. This resolves the external
  runtime defect, while Bots'07e8185pin and all incomplete integration gates
  remain unchanged. Explicit revised contract/signature compatibility adoption
  is the next prerequisite; old rejected records remain historical/incompatible.

## 9. Stage2a: shared layout, original frames and deterministic tokens (accepted)

This section passed its separate design and implementation gates. It enables no worker,
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

The raw HTTP ActionOptions wrapper remains `{state_revision,action_options}`.
The participant `DecisionInput` envelope additionally carries the explicit
`viewer_player_id` bound from that requested actor path and original Observation,
as in the existing participant envelope convention. Its validator rechecks that
outer viewer, inner actor and Observation actor agree. This is an envelope binding,
not a synthesized HTTP Observation field or a feature. Preserve raw endpoint
payloads separately in transport diagnostics when recording them.

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
That was the initial public-text finding. The owner subsequently resolved all
three predicates in the [2026-11-03 confirmation](../contracts/owner-20261103-card-instance-candidates.md).
Section11 supersedes the ambiguity gate; a separately demonstrated runtime
mismatch in [issue0008](../issues/0008-learned-miracle-discard-runtime.md) now
keeps real-game acceptance closed. Never reduce the confirmed candidate domain
to work around that mismatch.

## 11. Stage2b owner-rule generator (design accepted; component scope)

### Authority and compatibility

The owner confirms Flash≤1physical selection (including miracles/CostCut),
Discard/Sacrifice permission for every miracle subclass in both states, and
both explicit targets for every utility in spec09§17.5.2. The exact owner record
SHA is `3f3090f521041293bcb6de2a3cbac7e74ffc635b4acc0f7f02c957bbcf0e44ee`.
Public normative revision/inventory/catalog pins stay unchanged and are recorded
as such; the owner record is supplemental authority, not a fabricated Simulator
specification update. Test-source inspection corroborates named utility cases,
not every newly confirmed case. No Engine-handler/rejection inference is used.

Implement the pure generator against this declared rule domain. Both existing
debug/release TestServers reject a currently owned learned Mudflow Discard;
independent audit confirms current actor/revision/ref/wire correctness. Keep
public collection/worker/Source/fit entry points closed until issue0008 is fixed
and an actual corrected source/binary contract is explicitly adopted. The pure
stage emits component evidence only, never qualified Sources. A failed runtime
case remains diagnostic; do not turn it into an omission, mask or teacher rule.

Preserve ActionConfigV2/V3 fields/defaults/hash, old report field set, old token
signatures and all existing paths. Candidate legality authority is separately
versioned; no input feature shape/meaning changes in this stage. There was no
implemented generator-v5 or eligible v5 Source before this increment. Earlier
new-contract frames/tables are raw syntax/component evidence and cannot acquire
completed-generator or Source authority by adding a flag.

### Responsibilities and interfaces

- New stdlib-only `gf_common.card_instance_rules` owns frozen closed
  `CardInstanceRules`, `card_instance_rules()`, `rule_signature(contract,rules)`,
  and one pure `SequenceAnalysis` for ordered roles/payment/completion/summary.
  Its exact rules descriptor is version=`1`, profile_id=
  `owner-card-instance-candidates-20261103-v1`, and
  `owner_confirmation_sha256` above. Reject changed/missing/unknown descriptor
  values; the implementation does no runtime file I/O.
  It is below diverse_actions in dependency direction: return only primitive
  immutable analysis facts/owned pure types, never import _Proposal/_Summary or
  search machinery back into the rules module.
- Rule signature is the canonical digest of generator-v5, input-recipe-v2,
  full existing CardInstanceContract and the exact rules descriptor. It is
  compatibility/provenance metadata, not a policy feature.
- `PublicCandidateGeneratorV3(config: ActionConfigV3, *, rules:
  CardInstanceRules, clock=...)` requires explicit rules before proposal work.
  `generate(frame, bank_view, search_rng, numeric_costs)` requires the new layout,
  same participant prefix, independent Random and existing closed NumericCosts.
- `diverse_actions` holds version wrappers/spaces and a shared private
  `_run_search(space,frame,rng,costs,clock)`. Reuse traversal, saturation, resource
  math, quotas, length bands, seven-channel selection, compact table and RNG
  ownership. Version spaces supply prefix viability, completed-target count,
  mandatory/fixed rows, conservative count bounds and summaries. Retain old
  `_Space` semantics and exported aliases/private test seams.
- `SearchReportV3` is a distinct frozen DTO sharing common validated fields,
  with required keyword-only `rule_signature` and the new CardInstanceContract.
  Its generator is v5. `GeneratedCandidatesV3` binds that report to the exact
  table/counts/contract and verifies real unique18slot rows. Old report/generated
  DTOs remain closed and reject these variants. Factor common validation rather
  than copy the whole report implementation.
- `structured_wire` adds explicit v3 report/generated conversions requiring
  expected contract/rules, with no unknown-rule default. Old conversions reject
  new reports before serialization. New parse checks rule signature, generator,
  hashes, counts, table digest/layout/purpose and raw column bounds.

Future protocol3/journal6/artifact4/Source/Dataset3 consumers must bind the rule
signature explicitly together with Config hash and input/token signatures, and
validate it before acceptance/model loading. A bare manual table is not that
proof. This increment implements none of those future formats/capabilities.
NumericCosts1 remains the same closed dimensions/budget encoding; only future
validated model3/layout composition may supply actual model resource formulas.

### One public analysis for pruning and diversity

Analyze refs in original order using own perceived definitions, public catalog
classification/use_roles, flat same-revision usable_cards and choice-specific
carrier lists. Arbitrary candidate_groups strings are raw-only. Do not infer
another participant's possession or fetch unobserved numerical definitions.

Bind adjacent CostCut before counting effective roles: it waives only the prior
miracle and suppresses its own ordinary attack/defense/effects. No leading pure
Doll or whole-sequence-free predicate. Return both prefix viability and complete
submission/payment facts. A prefix can be viable with zero completions: only a
last unpaid miracle can still be fixed by an available distinct adjacent CostCut
within length/Flash limits. Earlier unpaid costs cannot be repaired later.

Offense permits an initial ordinary weapon plus additions, additional-only
attacks, or distinct same-definition normal-miracle instances plus additional
miracles/support. A normal-miracle root excludes all effective artifact attack
roles, including additional artifact weapons/armor; different normal miracles
are excluded. Bound CostCut artifacts are exempt because their ordinary roles
are suppressed. This is the explicit spec09§9.4 public matrix, not a rule
derived from rejection. Preserve Wand/Aura
position effects, Mirage attribute/Wand reset and base-round×max(1,Mirage count),
including two-hit weapons. Dynamic Stick pays miracle costs first then consumes
remaining MP; zero attack/MP remains supported. Utility/status roots complete
alone when affordable and may carry their adjacent CostCut; exclude unrelated
utility mixtures and multiple status miracles. Utility target choices are both
configured/hinted targets, including empty-effect/removal/guardian overwrite.

Response combines hinted public numeric-defense/ring roles or one exclusive
stop/reflect/deflect with only its bound CostCut. Flash allows≤1physical ref,
so the general CostCut combination permission creates no Flash exception.
Discard/Sacrifice includes every miracle in either state and artifact nonweapons;
the artifact weapon exclusion stays. Start from original own.cards plus the
explicit operation flag; do not reuse direct-use usable_cards or MP payment as
a discard mask. Include MP0/absent-from-usable cases for every miracle subclass
and state. Sell excludes learned miracles from sold
items, supports distinct carrier/item, and obeys configured length. Buy/Sell
have explicit targets; Exchange has none and conserves own resources with0..99
allocations including HP0. Phenomenon carrier uses invoke_phenomenon; empty
response/Pray/buy decisions use their explicit flags/choice. Server owns RNG,
automatic effects, HP0, actual future state and refill outcomes.

Use this same analysis in public pruning and summary payment/attack/effect axes.
Support cards do not regain ordinary roles in diversity scoring. Remaining-card
and refresh proxies account for learned miracles retaining possession slots,
known removals, self-Sell retention and full-possession Pray's one random removal,
without predicting its identity. Discard has zero refill; Sacrifice frees selected
slots. Preserve all seven channels and every mandatory completed singleton/basic.

### Counting, budgets and verification

Counting must distinguish viable prefixes from completed rows. Call a pure
completion-count hook rather than assume every valid prefix submits, and share
the exact same target/payment predicate with discovery. Counts/upper bounds
include all family/carrier/numeric supplements and saturate without materializing
all Actions. Legacy hook implementations preserve original count/order/RNG facts.
Keep16384states/50ms,262144expansions/temporary rows,32768presented candidates,
1ssearch/selection,25interval exchange grid and supplied tensor/storage limits.
No budget expansion or automatic domain reduction.

Required checks: independent small-space enumeration; exact count/discovery
agreement; unpaid prefix→CostCut completion with earlier-cost negative cases;
role suppression shared by pruning and summaries; same-definition distinct
miracles in both states; Wand/Aura/Mirage/two-hit ordering; all utility targets;
Flash physical counting; all six miracle subclasses in both Discard/Sacrifice
states; rings/exclusive responses; HP0/conserved exchange; trade/phenomenon;
mandatory coverage, long bands, saturation, deadlines and numeric admission;
wrong/missing rule signatures; old report/wire/Config/token/generator goldens.
No synthetic fit, game-strength or corrected-runtime acceptance follows.

Before extraction, freeze legacy generator outputs from214b1fd with deterministic
fake clocks for exact enumeration, bounded search and count-upper-bound cases.
Preserve table row order/hash, pre-count/search accounting, cutoff/mandatory
coverage and full RNG before/after state. Independently reproduce those goldens
from the committed old module during final review; wall timing is not the oracle.

## 12. Corrected-runtime adoption (accepted)

Independent architecture DESIGN READY after clarifying standalone Delta parsing
versus bank-chain attestation; root accepts that scope. Source conformance and
verification remain required before milestone acceptance.

Implementation and both final reviews are READY. Root accepts551affected tests,
old13Config/default/signature/golden preservation, corrected direct runtime
frame/token/generator/wire conformance,125source wheel and optional-free baseline.
Current corrected token/rule signatures are respectively
`72db3a64f6d087a5f889c4e3cf6b4eb46a8595fd334e1249ca2d51f76496e1ad` and
`3f55f044225fb20c63d683145cdceb397c36f0b7161c7b16793f72210db3e86d`.
This adopts identity and pure diagnostics, not the still-unimplemented qualified
collection/model/worker/native/Source paths.

### Scope and authority

Explicitly adopt verified Simulator
`91d5926a7572db327eaea3751e7c24f428120996`, normative13inventory
`131890dfe579bc6951f5189e460971d33658339106e219a2e00b1a289a0534d0`.
Rule set/API/schema/catalog/ref scheme remain unchanged; catalog SHA is still
`f51a3c73feb8b3f01837e65768aafb5a3aa805a33483308130244b33b16a8ad6`.
The verified release TestServer SHA is
`76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d`;
debug SHA is `b3670dac3eb099ba4363058f8d92a44976ebaaf1dfdc187df5a9fe333ec0e4a1`.
The handoff change inventory matches committed source/test/spec content; no new
reproducible compiler build is claimed. No Simulator edits or dependencies.

Spec03§34.1/spec06§31.1/spec09§11.3 govern the correction: historical rejected
Attempts can now disagree with the engine and must fail strict reconstruction;
rebuilt ActionOptions and full Replay JSON need not remain byte-identical.
Do not rerun old data under new semantics, rewrite an old rejection, flip source
eligibility, or transfer old weights automatically. Keep earlier evidence and
all v1/engineering2/07e-component hashes/meanings unchanged.

This first adoption slice adds the closed corrected identity and pure/Config/
direct-HTTP paths. It does NOT publish model3/protocol3/journal6/native/Source
support or open public collection/training. Those require the following scoped
integration gates and final acceptance. All new runs use dedicated91d5926 paths.

### Closed identity and dependency direction

Keep the SAME exact frozen CardInstanceContract class and eleven serialized
fields. Its defaults and card_instance_contract() remain the original07eprofile.
Replace single-default validation with a closed immutable registry of TWO exact
full tuples, selected by profile_id; mixed pins, missing fields, changed types
and unknown profiles reject. This avoids subclass/type-guard proliferation while
retaining exact historical construction/serialization. Add explicit
corrected_card_instance_contract() and corrected revision/spec constants/map.
The new profile is `card-instance-simulator-v2-20261104`. No partial/default
upgrade of raw Config or stored records is permitted.

Existing CARD_INSTANCE_LAYOUT stays07e. Add CORRECTED_CARD_INSTANCE_LAYOUT with
identical field tuples/codebooks/dimensions/format versions and corrected full
contract/revision. Only known singleton layouts pass require_layout. A closed
is_card_instance_layout(layout) identifies schema behavior for either card-ref
profile; it is not permission to mix their records. layout_for_contract dispatches
the exact registered contract to its matching singleton. Keep exact expected
layout identity checks in every bank/frame/table/token/wire boundary.

The layout module remains below records/tokens and imports no upper package.
Rules1/owner confirmation3f3090…e44ee stay unchanged; rule_signature's full-contract
input yields a new corrected signature. Existing07erule signature1480f832…79a36
and token signatures remain reproducible under their explicit original layout.
Legacy default arguments continue to mean strict LEGACY_LAYOUT.

### Record, signature and serialization behavior

Add three thin frozen no-field corrected EventDelta/View/DecisionFrame classes;
Delta uses init=False to retain validated construction. Existing classes stay
unchanged. Explicit factories select the matching known class; View.prefix and
dataclasses.replace preserve it. No InitVar or serialized runtime-layout field.

For corrected banks ONLY, bind the full corrected contract into the initial
prefix digest. For corrected frames ONLY, include that full contract in the
frame-hash binding as well as recipe/actor/revision/payload/history. Old initial
hashes and frame bindings stay exact. Corrected Frame/View binding verification
rejects cross-profile reinterpretation even when event/payload bytes coincide.
Raw Delta has no standalone profile discriminator: its parser constructs the
explicit caller-expected variant, and append must verify previous_hash against
the profile-bound bank chain. Parsing a Delta alone is not profile attestation.
The new token signature already binds the full contract/revision; preserve the
original payload keys/values for old layout signature computation.

All schema branches currently written as `is CARD_INSTANCE_LAYOUT` must handle
the closed card-ref family. All SAME-profile checks remain exact, including
generator config↔frame↔bank, table↔frame, tokens↔model expectation and v3 wire
expected contract/rules. CandidateTable derives layout from its stored contract;
neither its field set nor old hash changes. New bank/frame/table/token/report
hashes are distinct and explicitly recorded. No codec2/native marker is added;
codec1 and old numeric consumers continue rejecting BOTH card-ref profiles.

### Config, I/O and failure ownership

Root adds a corrected ServerConfigV2 variant with the same field set, overridden
revision and release-binary default, preserving original ServerConfigV2 defaults.
Dispatch the concrete defaults before merging from the explicit raw contract and
server revision; require their exact agreement. Keep raw API/revision/contract
declarations mandatory, path origins correct and all old Config hashes stable.
Add new corrected actions/experiment presets; do not overwrite07epresets. Fresh
output root is `runs/card-instance-v2-91d5926`. Baseline-only Config validation
and direct attested HTTP remain available; incomplete-pipeline/model3 guards stay.

HTTP dispatch uses the validated passed contract and unchanged v2 public wire
validators. Debug attestation, complete public info/catalog before Create,
sticky poison, UUID/payload retries, ownership/deadlines/reap stay shared and
unchanged. Metadata parsing can recognize the historical profile without granting
qualified-use authority; future public Source/coordinator preflight must require
the corrected profile AND verified complete original runtime/source evidence.
Neither a contract flag nor a manually constructed qualified-purpose DTO suffices.

The unchanged /v2/info carries no Git revision. Future qualified execution must
bind actual locally managed executable/source evidence separately: exact accepted
binary SHA, configured source revision/spec inventory and recorded handoff change
inventory must agree before Create. Preserve that evidence in run diagnostics,
never policy input. An external URL lacking equivalent code evidence cannot
qualify merely by returning matching wire/catalog metadata. No such qualified
execution path is enabled in this adoption slice.

### Verification and milestone acceptance

Before mutation, freeze hashes/bytes for all existing Config presets, original
CardInstanceContract/ActionConfigV3 defaults,07e token/rule signatures and exact
legacy/card-ref goldens. Preserve current owner record and all old fixtures.
Verify both registry entries round-trip; all mixed/unknown tuples reject; new
Config dispatch/relative paths work; old defaults/Config hashes stay exact.
Check distinct corrected initial prefixes/frame/table/token/rule hashes,
cross-profile Frame/View parser and bank/table/wire/token rejection before arrays,
explicit Delta-context plus append-chain mismatch rejection, no-field
dataclass/replace guarantees, and all unchanged rule/small-space/RNG goldens.

Use fresh diagnostic games on the already verified release binary, full own
Events/current Observation/same-revision hints, corrected frames, generator and
wire; submit an original-ref learned-Mudflow Discard and verify accepted removal,
resources/order, replay reconciliation and clean owner shutdown. This is direct
contract conformance, not192pilot or qualified Source. Both final correctness and
architecture-conformance reviews, relevant tests/Ruff/mypy and package baseline
checks precede root acceptance. Full migration/learning remains subsequent work.

## 13. Model3 neural kernels, inference and safe artifacts (implementation reviewed)

Independent architecture DESIGN READY after fixing exact owned RNG scope,
per-manifest/private-namespace publication semantics, pre-change initialization
goldens, immutable action-config binding and component-only policy selection.
Root accepts those boundaries. Both independent implementation reviews are now
READY; the original-layout CPU/actual CUDA matrix passes. The later §14.1 repair
supersedes its corrected-profile event/history widths and requires fresh numeric
acceptance. Original07e and legacy dimensions remain as recorded below.

This gate covers closed neural Config3, the SAME learned neural implementation
under an explicit selected layout, participant-local standalone inference and
inference/resume format4. It adds no runtime dependency. Tree3, participant
protocol3/journal6, numerical rollout4/learner wire4, Source/coordinator/native
integration are following scopes; none is implied by this numerical milestone.

### Config and shared kernels

Add distinct frozen EventConfigV3/ModelConfigV3/ConfiguredNeuralModelV3 types.
Reuse existing state/card/hand/temporal/head/resource/inference component DTOs;
event default is the card-reference aggregate, version3/input-recipe2 identify
input semantics, initialization/PPO algorithms remain unchanged. Old dataclass
fields/defaults, historical62 descriptors and pure/recorded parsers stay strict.
Model3 Config metadata never duplicates the full action contract in every preset.

Pure model3 functions require an explicit known card-instance layout. Support
either registered card profile for explicit component numerics/read-only evidence;
all active experiment/model3 Config combinations require the corrected profile.
Omitting layout retains LEGACY_LAYOUT and therefore rejects Model3 before Torch
or arrays. V2/historical functions reject card layouts. Root Config dispatches
from raw actions/model/version before defaults and invokes the model parser with
that exact layout; wrong profile/recipe/version/capacity fails before allocation.
Add neural3 presets only when standalone model3 inference is implemented; keep
the public run/collection guard until the complete execution path exists.

Parameterize shared dimensions/parameter_shapes/resource_estimate/make_network
and architecture/token identity with keyword-only layout=LEGACY_LAYOUT. Validate
the closed model/layout pair once, carry the known layout through all inner
packing, validators, caches, history, full/split and dense/chunked methods. Do not
fake a ModelConfigV2 or change module globals to process Model3. The same named
ActorCritic/state/hand/interval/GRU/Transformer/head modules and initialization
draw order remain for legacy calls. No second neural/PPO stack.

Use layout-derived category/field counts, including membership vocabulary at
membership_index and numeric order_field. Remove literal2*12 and old source-vector
assumptions from shared formulas. Default old→new widths: card_input116→166,
card_actor70→71, frame600→601, actor1414→1432; state5499 unchanged, card_direct
428→478, event394→395, aggregate376→377. These were the SAME for both
card profiles before §14.1; that repair changes only corrected-profile event/
history-dependent dimensions. Identities remain distinct through token/contract
pins.
Hand attention still has no arbitrary position embedding. Critic receives no
candidate content/count/order/search report. PreserveIEEEFP32, deterministic
initialization/backends, noAMP/TF32 and existing1e-5parity tolerances.

### Input binding and standalone policy

Add distinct safe-options/input-binding/network-binding v3 DTOs; never add fields
to the existing v2 serialized DTOs. The v3 binding explicitly carries full
CardInstanceContract, rule_signature, purpose and synthetic_capacity. Rules are
the reviewed rules1; computed signature must match the supplied contract before
arrays/model loading. Check model↔network↔active binding/layout and parameter
versions; frozen input identity cannot be relabeled or changed on resave.

The policy and private immutable network binding also require an explicit
action_config_hash from authoritative passed Actions Config (fresh construction)
or the externally checked artifact/load binding. Generated report.config_hash
must equal it before tokenization or cache advance. Manifest4 stores this hash
explicitly; first binding freezes it on the network, and subsequent policy/save/
resave/load cannot substitute another value even if contract/rules/shapes agree.
Architecture identity stays model/layout-only; full policy identity binds the
manifest/execution/input/action-config context. No Config dataclass field is added.

Fresh binding declares component inference-use scope, not Source authority.
This milestone permits component numerics under either explicit card profile;
qualified_game binding and selection remain rejected/deferred until a separately
reviewed qualified generated-table/Source path exists. Current GeneratedCandidatesV3
requires component+nonsynthetic tables and is not relaxed here. Synthetic capacity
can support explicit component numerical/artifact tests, but standalone policy
selection requires exact nonsynthetic purpose/capacity agreement. A component
network/artifact cannot be relabeled on resave. New labels
name version3 and include full manifest/execution/input identities.

StructuredPolicyV3 uses the shared inference/CPU sampling/cache mechanics with
an explicit expected layout/rules binding. Its select_frame validates a
GeneratedCandidatesV3 report+table against expected rule signature/contract and
the immutable expected action-config hash,
then tokenizes only policy-visible frame/table values. Report/cutoff metadata
are compatibility checks and diagnostics, never features. Before token/array
work check all scalar caps and exact profile, then retain the existing admission,
chunking and participant cache ownership. Empty events/masked hands stay finite.

### Inference/resume4 and metadata isolation

Expose new format4 save/inspect/load/read/restore wrappers with explicit expected
model/layout/contract/rules/execution/action-config binding. Factor private
version/context helpers around existing sharded storage, FP32 extraction,
named Adam moments/steps and the exact existing owned RNG set: Torch CPU RNG,
all CUDA RNG states visible to the dedicated learner process when using CUDA
(none on CPU), and the explicitly supplied NumPy Generator's PCG64 shuffle state.
Do not capture Python random/global NumPy state or implicit future search,
schedule or sampling state. Those are separately owned by future coordinator/
participant formats, with their own later resume verification.
Old format3 APIs/keysets/defaults/semantic helpers stay exact and reject format4;
new wrappers reject old format/unknown keys or mismatched profile/signature.
Do not duplicate optimizer serialization or an entire persistence module.

Format4 semantic metadata binds the selected profile's actual revision and13
normative hashes, catalog2, recipe/generator/policy-input versions, token signature,
candidate rule signature, transform/reward/numerics and architecture. Preserve
lower NumPy-bundle1/bank dtypes/shard hash/64MiB decoded-tensor limits. Producer
source identity and runtime execution metadata stay explicit and separate from
architecture. Add distinct sealed metadata keysets/inspectors for4; do not mutate
global legacy semantic/classification sets. Metadata inspection remains below
training and lazy with respect to Torch, never imports training orchestration.

The format4 record is numerical/inference compatibility evidence. It cannot
authorize a generation, complete parent or qualified Source. On failed validation,
timeout/cancel/save/reload, retain original failure and existing owned-file/
directory cleanup behavior. This milestone retains the existing per-manifest
commit semantics: validate/copy/write shards before publishing that manifest;
do not add reportable post-commit work or claim late-failure revocation. Inference
and resume are separate commits: a later failed resume save may leave its earlier
healthy inference loadable, but no healthy resume or parent-generation completion
is thereby implied. Pre-manifest injected failures must leave the affected output
unloadable. Output storage in this milestone assumes an exclusively caller-owned
private namespace and rejects preexisting foreign entries. Existing lstat/
write/replace helpers do not protect against concurrent hostile pathname
substitution; no such guarantee or C3b-style FD protection is claimed. Revisit
this assumption before supporting shared/untrusted output namespaces, with
exclusive FD/publication remedies and their tests rather than another preflight.
A late OS flush/
publication error may leave a readable manifest with uncertain durability; it is
failed/diagnostic and cannot authorize Source or parent commit. Accepted debt:
this scope has no owned-FD post-publication revocation; revisit before adding
post-save deadline/reload failures to these APIs or requiring durable standalone
failure revocation. The later coordinator must commit its generation only after
both validated records succeed. Ordinary
resume keeps strict configuration/code/execution/input/RNG identity; seed alone
does not establish determinism. New profiles never auto-load old trained weights.

### Verification

Before shared model-kernel edits, freeze named V2 state_dict shape/dtype/value
hashes plus post-initialization Torch CPU RNG for nine pinned small CPU
hand3×model3 constructions. Independently reproduce from committed pre-Model3
code and compare after changes; deterministic repetition alone is insufficient
to prove unchanged draw order. CPU construction covers the shared CPU initializer
before device transfer, while actual CUDA checks remain separate.

Hand3×model3 CPU numerical matrix and required actual CUDA counterparts: parameter
shape estimates versus state_dict, empty/masks, original order/duplicate instances,
encoder gradients, candidate permutation, critic independence, dense/chunked
logits AND gradients, cached/full/split equivalence and cache invalidation.
Test preallocation failure, wrong-profile/equal-width signatures and resource
envelopes. CPU/CUDA comparisons retain established tolerances.

Round-trip inference4 and all named Adam/RNG state at a normal synthetic update
boundary; compare continuous2 versus save1/load/update2 across encoders/models
where applicable. Keep inference-only artifact evidence separate from later
real PPO/coordinator resume acceptance. Check missing/extra/modified shards,
input signatures/rules, wrong execution/model, synthetic/qualification relabel,
old3/new4 mismatches and old configured/saved artifact corpus unchanged. The
verified host GPU is RTX5070Ti with16GiBphysical VRAM; preserve the12GiBmanaged
global limit and64MiBarray/shard limits, failing oversized settings explicitly.

## 14. Corrected baseline end-to-end collection (accepted baseline scope)

Independent architecture DESIGN READY after pinning executed-code verification,
closed approved runtime tuples, final publication/invalidation order, pure
receipt dependency direction and new heuristic facts. Root accepts this scope.
Implementation/conformance/pilot remain required. Root permits two disjoint
implementers for §13 model modules and §14 new common/client/journal modules
while their reviewed interfaces are fixed; only root edits shared CLI/Config.
Neither implementer edits the other's modules; global checks wait for stability.

Both independent implementation reviews are now READY after the raw HTTP hint
viewer-binding and terminal publication-close findings were resolved. The
two-game real smoke passes. The first exact192-game pilot completes all192but
fails with180natural/12incomplete (11Discard rejections and1event validation
failure); the whole failed population is preserved and ineligible.

After §14.1/14.2 input corrections and both independent incremental reviews,
the third exact192pilot completes192natural/zero failures/96complete pairs in
297.091s. Independent audit validates all384original journals/own Replay streams,
3244accepted decisions, all4492inventory hashes, exact schedule/runtime bindings
and cleanup health. Root accepts baseline provisional feasibility only, not
neural/Source/learning/Tree acceptance or strength. Both earlier failed populations
remain ineligible. Current134module wheel bytes match the frozen source and pass
mandatory-only installed protocol3 baseline checks; actual pilot ran those source
bytes, not an asserted wheel installation. Final combined acceptance remains.

Root's public-rule audit disposition: restore ADR0005's already-authorized
Sun Amulet/Dangerous Mortar exclusion for normal Discard, explicitly retained
by ADR0023's action matrix. Owner confirmation and the v2 override extend miracle
permission; they do not rescind this artifact exception. Sacrifice does not
acquire that exclusion. `_CardInstanceSpace` dropped the prior condition; this
is a local implementation conformance fix for BOTH v2card-instance profiles,
using perceived definitions. It restores the declared rules rather than deriving
a new mask from rejection. Keep all contract/token/rule/Config/schema identities
and saved bytes unchanged; record the corrected code revision/source digest and
require a fresh complete pilot. Exact old buggy-generation reproduction uses its
recorded code, never rewrites historical outputs or grants Source eligibility.
Independent offline exact-subset tests cover both profiles/normal-vs-EndTimes
and all miracle states/subclasses before the next full pilot.

This is the next execution/persistence gate after the model component scope.
It integrates corrected-profile Random/Heuristic only: protocol3, codec2,
journal6, collection manifest5, real Replay reconciliation and the fixed192game
pilot. It does not enable model3 qualified selection, neural training/rollout4,
Tree3, new native capability or fitting. Existing protocol2/codec1/journal5/
engineering/private-test paths and serialized meanings stay unchanged.

### Qualified-use context and evidence

Keep current GeneratedCandidatesV3 component-only. Add a distinct explicit
GameGeneratedCandidatesV3/serialization path requiring corrected contract,
purpose=qualified_game and nonsynthetic rows; its report still binds the same
reviewed Rule signature and actual Actions Config hash. Provide an explicit
game-generation entry on the shared generator/search engine, never mutate an
existing component table's purpose or relax its current DTO. Like a bare table,
this new DTO is use-context metadata, NOT Source qualification authority.

Actual public collection requires a locally managed verified runtime evidence
owner before allocating processes or creating output. The cross-module pure
receipt/pin schema and validators live in gf_common below journal consumers;
filesystem/live-process ownership lives in gf_simulator_client and composition
in CLI. Journal6 imports no client transport/lifecycle or CLI Config. The
evidence descriptor binds corrected contract, expected full source revision/
13spec inventory, approved executable hash and handoff source-change inventory,
plus actual binary stat/hash and source read checks. /v2/info is not a revision
attestation. Compare configured executable/source bytes before startup/Create
and recheck after owned shutdown, BEFORE healthy collection publication. Changes
invalidate run health. Approval is a CLOSED release-only tuple for this scope:
corrected profile/revision91d5926/normative131890…534d0/change inventory
e34b716…70346/release76aff19…d0f7d plus exact approved file→SHA maps. Paths may
vary; callers cannot supply arbitrary approval hashes. After startup and BEFORE
Create, open/hash the owned live process's /proc/<pid>/exe (or equivalent
executed-FD facility), verify owned-process liveness/identity around the read,
and retain the actual executed-code receipt. Pre/post pathname hashes alone
cannot detect swap-and-restore. Missing evidence fails before Create. Other
builds remain diagnostic pending an explicitly reviewed approval entry.
Keep debug management identity/seed/effective Config attestation in addition.
Reject corrected qualified external-URL collection absent equivalent explicit
build evidence; diagnostic HTTP remains separate. Policy workers receive none
of this source/path/seed/URL evidence.

New versioned bindings/header metadata must bind expected Config/contract/layout/
token/rules/purpose/caps/model identity explicitly. Only corrected profile can
enter this real-game path. Historical profiles remain readable in their original
scopes and cannot be upgraded by changing eligibility flags or DTO purpose.

### Shared episode ownership and clean participants

Reuse SimulatorEnvironment's one group/server ownership, one original accepted
command record, cursor histories, episode deadline/finalization/reap and outcome
handling. Extract small version-specific hooks for client creation, DecisionInput
building, Replay reconciliation/closure and participant adapter rather than copy
the entire episode loop. Old hooks/defaults reproduce old behavior.

Add a corrected baseline environment adapter using explicit runtime evidence,
new contract/layout/result identity and the existing shared ReservationLedger.
Preserve reserve-before-Popen, startup failure ownership, shielded cancellation,
worker≤4/game≤2 and source/process lifetime boundaries. The adapter gathers only
the current actor's own HTTP Observation, dedicated full visible Events prefix,
typed metadata, own-visible perceived catalog and same-revision bound hints.
It builds corrected deltas/frames in original possession order. Never use Act
events/next-observation/options or Replay as policy history.

Protocol3 uses a separate explicit bootstrap/DTO route; old protocol2 remains
strict. Bootstrap receives only safe own policy identifier/settings, sampling
seed, Actions/rules/input bindings and participant-local messages. Scrub argv/env
as currently implemented; no full experiment/server Config, management evidence,
GameSeed/RNG, other viewer, canonical history or Snapshot enters the policy.
Send once-only actor event deltas with prefix hashes and raw frames; child
validates exact profile, Config hash/rules/caps/continuity before candidate work.

Baseline Random/Heuristic implementations reuse selection/tie/penalty mechanics
with explicit corrected context and new labels. Heuristic payment/attack/
effective-role facts MUST come from new SequenceAnalysis: no old sequence-free
CostCut or suppressed armor defense is ported. They emit no PPO
logprob/value. Keep independent search and sampling RNG ownership/versioned
seed derivation. Parent validates full GameGeneratedCandidatesV3 and Selection,
then reconstructs the typed Action from its ORIGINAL own frame before journaling
or Act. Retain exact command UUID/revision/ordered payload on bounded resend;
consumed references are never regenerated. A rejection stops with diagnostics
and no re-proposal, fabricated transition or changed legality predicate.

### Codec2 and journal6

Add codec2 using the same numeric section encodings/shape/limits and a separate
closed action codebook/descriptor identity. Header binds full corrected contract,
frame/table digest, purpose/capacity, lengths and exact section bytes/hash. The
old codec1 API/descriptor/keysets stay exact and reject card-instance tables.
New codec2 accepts only the explicitly expected known layout; corrupt/mixed
descriptors/offsets/refs/numerics reject before arrays. Existing lower array/shard
64MiB limits and file hashing remain unchanged.

Factor private byte-layout and sidecar I/O helpers where shared; keep distinct
public old/new codec/storage wrappers. Preserve original relative sidecar paths,
exclusive ownership, fsync/hash/stat checks and publication BEFORE the journal
decision row references it. File failure remains a pending/failed decision,
never an accepted Action. Do not grant semantic/Source authority to codec data.

Journal6 is an additive closed writer/reader with original frame/prefix/raw input,
rule-bound generated report and codec2 table, selection, command identity,
acknowledgement, policy execution/timing and terminal metadata. Header includes
corrected revision/13spec/catalog/input/token/rule/Config identities, viewer/run/
game/episode/bank, explicit model1 baseline and protocol3/feature2. Preserve
once-only delta continuity and revision advance; raw refs/history semantics do
not get normalized to v1. Check report.config_hash/rules, frame/table/selection,
caps and exact reconstructed command before publishing a decision.

Only positive validated acknowledgement resolves pending commands. Preserve
pending rows on interrupted delivery. Reuse existing finish/close-incomplete and
no-overwrite journal publication ownership. Natural selected-actor reward±1/0 is
assigned only after known natural result and reconciled closure. Truncation,
abort, unsupported rules and infrastructure failures retain separate metadata;
no invented final Observation/bootstrap/return. Old schema5 readers cannot read6.

### Replay and collection completion

Add pure v2 Replay reconciliation/closure functions separate from legacy ones.
Validate both full player DTOs, compare each actor's accepted journal Action
sequence to own visible_action entries in decision order, and require shared
terminal/aborted closure identity and known result agreement. Concealed other
actions are not policy input or an own acceptance record. Aborted/no-action
decisions cannot become accepted journal transitions. Keep exact turn-limit
projection; reconciliation never invents an unknown natural result.

CLI opens ONLY corrected baseline collect/evaluate/smoke after complete runtime
preflight and all new adapters are ready; old07e/actions2 and neural/tree3 paths
remain guarded. Add collection5 manifest/report binding actual runtime evidence,
schedule, seeds/splits, original file inventory/hashes, policy labels and all
semantic versions. Reuse scheduling, deadlines, pair/seed-cluster statistics,
incomplete-run retention and late-report invalidation. Do not infer success from
an exit code alone; all192games/pairs/closures/process/journal health must pass.
Current unused preprocessing fields remain explicitly inert for collection;
new offline Source/native/learning readers are not advertised until implemented.

Finish all owned worker/server cleanup, actual-executable/path/source evidence
checks, journal/Replay health and original file inventory before healthy report/
manifest commit. No reportable evidence recheck follows that commit. Final
publication owns exclusively created report/manifest temporary files and retains
their read-write FDs through publication and the final work-deadline check. If a
late publication/deadline check fails, invalidate only the held owned inodes
before releasing them, using the existing cleanup allowance, preserving the
initiating exception and any foreign substituted paths. Reuse the C3b small
owned-invalidation mechanism where practical; do not duplicate a persistence
stack or reset deadlines. Unconfirmed invalidation retains its reachable owner
and reports quarantine; never return healthy collection success. After successful
FD release, perform no further operation that can reclassify the healthy commit
as collection failure. Private run namespace and ordinary OS/I/O limitations
remain explicit; no revocation of foreign copies is claimed.

Terminal-release clarification (root accepts advisor B): closing descriptors is
itself inside the commit failure boundary; set committed only after all release
operations succeed. Keep names/inode/directory ownership separate from FD
liveness. A close error may occur AFTER descriptor release; EBADF is not proof
of successful persistence, and blindly retrying its number risks a reused foreign
FD. On failure, invalidate both outputs independently: use unquestionably held
FDs, otherwise freshly open the recorded name without create/truncate/symlink
following and verify opened type/dev/inode before any write. Reopen/verify the
owned directory identity if necessary. Never use an uncertain stale FD number,
search renamed originals or modify a foreign replacement. Recover within the
original cleanup deadline; inaccessible original, recovery/sync failure or
unresolved resource release retains the reachable quarantine owner and original
exception. Test before/after-actual-close failures for report, manifest and
directory, FD-number reuse, foreign file/directory replacement and bounded
independent recovery failures. This adds no foreign-copy revocation guarantee.

Baseline qualified-use binding/headers are distinct from §13's component neural3
binding. Do not widen Model3/Tree/artifact selection to enable baselines. Test
GameGenerated refusal by component policy/DTOs and all old protocol/codec/journal
paths. Neither purpose flag grants Source authority: only original live-runtime
ownership, exact health/records and the later complete Source verification can.

### Verification and acceptance

Test code evidence mismatches before output/process/Create, no management-data
leak at import/reset/select, malformed/stale/foreign deltas and selections,
correct original-ref recovery, exact retries, poisoned late success, rollback/
pending ack/cancel/reap/deadlines, sidecar/journal publication and mixed versions.
Keep current component GeneratedV3 rejection and old codec/journal/Config proofs.
Test learned-only and duplicate miracles, all Action families and corrected
Replay's decisions/visible_action/aborted semantics, including zero decisions.

Run a fresh bounded real corrected baseline E2E smoke first, then the exact
fixed192game pilot (32hexsuffix1..20GameSeeds×17/29/43sampling×seat swap) under
unchanged7200/5400/7140s and existing resource/search limits. No seed replacement,
budget shortening or rejection-derived rule changes. Preserve any failed run;
fix concrete client/declared-contract defects with independent reviews before
new complete acceptance attempts. Report pilot feasibility separately from
strength/learning. Root accepts after both independent reviews, relevant checks
and original-data/package/optional-free baseline proofs.

## 14.1 Guardian event namespace and corrected input repair (design accepted)

Independent architecture DESIGN READY; root accepts the bounded corrected-only
repair, historical-data disposition and mixed-code handshake protection.
Implementation and both independent final reviews are now READY, with803CPU
and31actual HOST CUDA tests passing. Whole-project/package checks and the
complete fixed pilot remain required. Public catalog aggregation verifies all ten known
Guardian IDs in nested guardian.id references, with no overlap with card IDs.

Public Player Replay diagnostics from the retained failed pilot show
`guardian_set`/`guardian_left` with `definition_id="earth"`. Guardian IDs form
the existing public ten-value GUARDIANS namespace, separate from the296card
catalog. The event validator, interval tokenizer and exact definition-count
aggregate currently assume all event IDs are card IDs. This is a Bots input
omission, not evidence of an invalid Simulator transition. Replay diagnoses the
missing value only: do not reconstruct the failed original journal, infer an
Action mask, add unseen information or promote that population.

Correct only the newly adopted91profile's event-ID vocabulary to the ordered
card296plus guardian10 tuple. Preserve LEGACY_LAYOUT and original07e layout,
their recipes, bytes, Config/defaults and all hashes exactly. Keep the corrected
contract's Simulator/public schema/catalog pins; this repairs the already
declared public-events input implementation, not Simulator semantics. The
corrected token signature, exact dimensions/architecture identity and dependent
new artifact bindings necessarily change and must be recorded. Do not invent
new state, definition IDs or card refs; raw event strings remain unchanged.

The corrected validator checks namespaces by event family: guardian_set/left/
acted require a known public Guardian ID; card-ID events retain the pinned
catalog whitelist. Unknown and cross-namespace IDs still fail closed. Use public
DTO/test representation only to confirm field meaning, never Engine resolution
or rejection results. The existing observation Guardian vocabulary provides the
closed ID values; do not convert Guardian IDs to card definitions or hashes.

Pass the explicit layout's event-ID vocabulary into the shared interval token
and exact-history functions. Default/shared legacy calls retain the exact
296-card behavior. Corrected history appends all306definition counts in stable
order before the existing window/actor sums; preserve the prior card count
indices. Guardian interval categories likewise receive distinct observed values,
not PAD/MISSING or a fabricated card identity. Resource/parameter formulae derive
from the explicit layout; do not special-case model types or relax tolerances.

Protocol3's current game binding must also carry the exact repaired layout
signature, checked by parent/child at initialization and every response before
candidate work/acceptance. Use a fixed documented128window signature as this
structural layout identity; the actual model/Journal token signature additionally
binds its configured aggregate window. Journal6 already binds that actual token
signature. Keep this new game-binding field distinct from live runtime evidence;
derive it in pure common code without importing client/I/O or changing component
bindings. Historical old91protocol messages without this field reject. This
closes equal-profile/mixed-code handshakes instead of waiting for a Guardian
event to reveal the mismatch.

All preceding91component artifacts/healthy smoke/failed pilot receipts retain
their original bytes and producer-code identity. They are historical evidence
for the preceding incomplete layout, not current input/numerical acceptance or
eligible Sources. The previous corrected token signature must reject on current
load/worker/journal/Source paths BEFORE allocation or mutation; do not reinterpret
old files by width, profile flag, padding or new signature. Reproduce historical
records with their recorded code. No old91trained parent or qualified Source
exists. Metadata inspection may fail closed for unsupported old91signature;
legacy/engineering/original07compatibility remains fully supported and exact.
Do not silently turn earlier27CUDA/CPU numerical results into current acceptance;
rerun the full affected Model3 matrix and actual CUDA cases after this repair.

Architecture review precedes implementation because this changes persisted input
meaning/layout, although public rules and Config hashes remain unchanged.
Verification covers all ten Guardian IDs × three event families with both
viewer-relative encodings, event-time identity/order/prefix preservation, exact
306counts/interval categories, unknown/cross-namespace refusal, no same-width
signature coercion, old-layout/token/model2 goldens and new parameter/resource
shape checks. Baseline J6/protocol3 replay/candidate tests and the exact full192
pilot must pass after the normalDiscard fix. Final numerical/package acceptance
uses the repaired input, never a reduced candidate or event domain.

## 14.2 Guardian action-definition representation (design accepted)

Independent architecture DESIGN READY after correcting the catalog subset count;
root accepts this validator/fixture correction. Implementation and both
independent reviews are now READY, with551CPU and31actual HOST CUDA checks
passing. The whole unchanged third pilot and current source-matched/installed
wheel checks now pass; final combined migration acceptance remains pending.

The second unchanged192pilot completes191natural/1incomplete and no rejections.
Both public Player Replay DTOs for s10-m1-r0-seat0 expose guardian_acted with
definition_id="seafood-soup". The pinned catalog explicitly classifies that
definition as kind=guardians/automatic and links it to guardian.id=neptune.
GuardianSet/Left use the Guardian identity, while GuardianActed identifies an
action definition. The public DTO uses a generic String and does not promise
the uniform Guardian-ID domain assumed by §14.1. That assumption was too narrow;
the second failed population remains preserved/ineligible and is not repaired.

Amend §14.1's validator partition only: guardian_set/guardian_left accept the ten
public Guardian IDs; guardian_acted uses the pinned296card-definition namespace
like other definition-ID events. Do not invent a narrower kind-specific Action
constraint from an event name. Preserve the same corrected306-ID token vocabulary,
all existing indices, widths, token/layout/rule/Config/schema signatures and raw
event strings. No conversion from the action definition to Guardian ID, missing
past-event reconstruction or candidate/legality predicate changes. This corrects
public representation support; rejected Action details are not evidence for it.
The public catalog plus Player-visible output/DTO is the evidence.

Replace the incorrect GuardianActed-with-Guardian-ID synthetic fixtures; retain
original recorded fixture/artifact bytes with their producer identity. Test all
ten IDs for Set/Left and all42pinned guardian-kind action definitions for Acted,
plus representative other catalog definitions under the declared generic card-ID
support, both viewers, exact categories/counts/order/prefix, and unknown/cross-
namespace refusal. The54entries with nested Guardian refs also include8phenomena,
2armor and2miracles; that is not the Guardian-kind action-definition set.
Other observed catalog definition IDs keep the existing generic card-ID support.
Run affected token/worker/J6 and all hand3×model3 CPU/actual CUDA numerical checks
with a real catalog Guardian action ID. No dimension/signature change is expected.
Independent correctness/conformance reviews and another complete same-condition
192pilot are required. The preceding2554whole pytest/wheel results remain
preceding-validator evidence; final whole-project/package checks must use the
final code. Do not repeat unrelated lengthy checks solely to count passes.

## 15. Corrected neural game inference (accepted inference-only scope)

Independent architecture DESIGN READY after pinning the closed initialization
union, whole concurrent memory admission and retained external-artifact owner.
Root accepts this inference-only scope under the repaired §14.1 input.
Implementation/reviews and actual smokes remain required; this is not Source,
generation or learning acceptance. Do not start dependent implementation until
the §14.1 shared input/handshake interfaces are stable and reviewed.

Implementation and both final reviews are now READY. A blocking child CPU
visibility-canonicalization finding is fixed before Torch/factory import and
rechecked. Focused540CPU/43actual CUDA cases pass, plus14CPU/3actual CUDA worker
rechecks. All six real initial-policy inference conditions complete12natural
games, and an independent24journal/Replay/retained-input audit passes. Current
135module source-matched wheel and mandatory-only baseline checks pass. Root
accepts inference only; Source/numeric/learner/generation/native/Tree authority
remains closed until its own implementation and acceptance.

This bounded increment joins the reviewed Model3 kernels/artifact4 to the
corrected protocol3/journal6/collection5 real-game owner. It enables explicit
MLP/GRU/Transformer inference on CPU/CUDA with corrected game candidates, followed
by actual game records. It does not implement or qualify Source, numeric
rollout4, learner wire4, PPO updates, native-v2 or Tree. Those remain required
subsequent work; a natural inference smoke does not satisfy their acceptance.
The independent pilot defects in §14 must be resolved before real NN smoke.

### Explicit use binding and shared model code

Keep all existing component-only entry points strict and their serialized bytes
unchanged. Add named game entry points on the shared Model3 implementation,
requiring the closed corrected contract, existing Rules1/action Config hash,
purpose=qualified_game, synthetic_capacity=false and positions≤18 before model
or process allocation. Share private input-binding/artifact/policy helpers with
an explicit checked use context; no module globals, fake Model2 configuration,
copied neural network, or relabelled component table/artifact. A fresh network
can receive one immutable use binding; a bound component network cannot become
a game network, or conversely. Historical07e/game bindings remain rejected.

Add explicit game inference/resume4 save/inspect/load wrappers with the same
shard encodings/limits, named Adam state and RNG ownership as §13. Manifest
purpose and full input/rule/action Config identities distinguish game artifacts.
All component wrappers/readers continue to reject game files and game wrappers
reject component files, before allocation or mutation. Static recorded metadata
parsing remains NumPy-only, version/context explicit and independent of current
Torch availability. Sealed artifact metadata is compatibility/provenance, never
Source eligibility or proof of completed training. Only an independently owned
future coordinator can publish a completed generation after both paired saves.
The ordinary private-namespace publication guarantee/debt in §13 is unchanged.

Add GameStructuredPolicyV3 (or equivalently explicit named factory) that accepts
ONLY GameGeneratedCandidatesV3 and the exact original corrected frame/prefix.
Reuse the same token encoders, cache, actor/critic and chunked scoring. Validate
report Config/rules and exact context before tokenization/cache activity.
Sampling yields finite old logprob/value using the existing Selection contract;
critic values and search summaries are not actor features. Component policy
continues accepting only its existing GeneratedCandidatesV3 component type.

### Worker, journal and execution ownership

Extend protocol3 initialization with a closed baseline/neural tagged union.
Existing baseline payloads remain exact. Neural init carries only own Model3
safe inference options, manifest identity, action/rules/use bindings, caps and
sampling seed. No experiment/server config, source/revision paths, GameSeed,
management evidence, other-view observations, Snapshot or Replay enters child.
Fixed bootstrap sets permitted CUDA visibility BEFORE any Torch/factory import,
uses the reviewed model factory, then clears unused launch inputs. Arbitrary
factory/module selection is not enabled. Reuse original parent-bank continuity,
search RNG, framed I/O, exception/cancellation/reap and lease ownership.

The existing model_id is the discriminator: random/heuristic retain the exact
repaired baseline init key set and forbid policy_options; mlp/gru/transformer
require that same common set plus a mandatory closed policy_options object.
Unknown IDs, missing options, None options for NN and extra baseline options
reject before factory/Torch import. There is no caller-selected factory/module.
Ready/select wire shapes stay common and bind repaired layout_signature; neural
policy_version must equal the independently computed sealed manifest/execution
label, and baseline policy labels remain unchanged.

Reuse Model3 resource estimates and existing ReservationLedger: validate artifact
and configuration metadata, caps, CUDA visibility and numeric generation costs
before Popen; reserve before spawn; account actual inference lifetime plus
temporary decision memory under the unchanged global12GiB budget. Parent/child
independently agree on safe options, costs/caps and policy label. Initialization,
partial construction, timeout and cancellation preserve the original cleanup
owner. Baselines keep zero managed CUDA reservation. Maximum2games/4workers and
10sworker timeout remain; oversized configurations fail before allocation.

Before output or any Popen, CLI must admit the maximum concurrently live worker
set for the configured game concurrency, including both player roles per game,
and any already-owned learner reservation. Use the existing full configured
worker peak as lifetime charge where it already includes decision temporaries;
otherwise add the maximum overlapping temporary scopes (one acting worker per
live game). Do not double-count a covered scope or assume that individually
fitting models compose. With two simultaneous games, reject a two-role GPU
configuration whose four lifetime/covered-peak charges exceed the shared12GiB
budget before starting either game. Dynamic ledger checks remain mandatory and
use the same formulas; they do not replace this whole-configuration admission.

Inference artifacts use an explicit retained external-input contract for this
scope. Before output/process allocation, a bounded sealed input owner validates
the expected original manifest and every referenced regular shard, full hashes,
stat identities, context and metadata under existing64MiBper-array/shard limits.
Reuse SealedScan/snapshot accounting instead of a new persistence stack. Keep
that owner and exact file inventory through all child loads and reaps; children
load the same expected manifest digest and validate shard hashes independently.
Recheck the complete owned original inventory after reap and before healthy
collection publication. Changed/missing/replaced inputs or a late I/O/deadline
failure invalidate the collection; no silent copy/rebind/reload is attempted.
Collection5 records exact sealed metadata/file hashes and the external retained
input contract, distinguishing them from run-owned journal inventory. The owner
can release retained metadata after final recheck without further reportable I/O.
This is identity/compatibility integrity with the ordinary private-namespace/OS
limits already stated, not Source qualification or hostile-copy revocation.

Journal6 retains its baseline header/selection validation. Add a separate closed
neural header branch identified by model_version3, containing sealed Model3,
execution/artifact manifest identity and expected policy label/caps/numeric costs.
Validate it without loading Torch. Accept finite logprob/value only for the
neural branch; baseline still requires both None. The raw original Observation,
once-only event prefix, codec2 candidates, original-ref Action, acknowledgement
and natural terminal reward ownership stay identical. Reading either branch
never grants Source authority. Reader must validate neural policy/candidate/
selection context, not trust a model_version or purpose flag.

Reuse the same corrected environment, live executed-code attestation, Replay
reconciliation, original-journal checks, inventory and final collection owner.
Extend its allowed policy bindings explicitly and share per-group evidence
ownership. CLI validates corrected actions3 plus neural3 checkpoint/metadata and
resource envelope before output/process allocation. Root owns Config/presets/CLI
composition; implementer owns assigned model/client/journal modules/tests.
Collection5 retains exact baseline records and includes the actual sealed neural
bindings for new runs. Do not enter PPO collector/coordinator or fixed-data Source
from this inference-only increment.

### Verification

Retain old Model2 initialization/Config/artifact hashes and Model3 component
round trips. Test both directions of component/game/old-profile mismatches,
bound-network immutability, descriptor corruption, wrong action Config, candidate
permutation, cache/full/split/logprob/value equivalence and preallocation refusal.
Run CPU and actual CUDA inference/artifact/worker tests for all three models;
reuse the established numerical tolerances and resource budgets. Test malformed
safe init, import/env information isolation, fail-spawn, reserve/reap, timeout,
cancel and crossed journal/policy binding including empty-input handling.

After §14's full pilot passes, run fresh bounded real-game inference smokes for
the three NN models with CPU and CUDA inference, recording exact initial artifact
identity, original own-input journals and Replay closure. These are inference
smokes only: no result becomes a formal training parent, 512train performance
panel or PPO/resume acceptance by relabelling. Independent correctness and
implementation-conformance reviews remain required before this increment opens.

## 16. Corrected Python Source, numeric4 and shared PPO (accepted runtime scope)

Independent architecture DESIGN READY after pinning online ownership, exact
reader/PPO-collection variants, Source-issued wire expectation, RNG-neutral
recorded-output parity, early combined admission and compatibility/pruning
verification. Implementation now passes both independent reviews and all six
real continuous2/pause1→resume2 conditions plus the CPU-learner/direct/DeepSets
smokes. Ten final independent audit receipts pass: all12main Source parents,
six saved-state reports and three extras;432natural games/27actual updates.
Same-input G0/G2 saved arrays match exactly (MLP60/234, GRU64/250,
Transformer92/362). The earlier independent-trajectory comparison remains FAIL;
§16.1 defines the distinct, predeclared same-input restoration claim. Derived
buffers/scalars are checked by the reviewed helpers; final NumPy-only auditors
independently rehash saved arrays and original snapshots without replaying them.
Root accepts this bounded Python runtime scope. Final receipt
`python-ppo-acceptance-final-1.json` SHA d8b99b5a…3540202 is in the verification
ledger. Native/Tree and final combined checks remain separate gates. These short
parents are nonformal and gain no declaration or Source qualification by this
acceptance. The original137Python source freeze is complete and lifted.

After §15, connect the corrected neural game path to the existing single
`ppo_run._run_ppo` lifecycle. Implement the explicitly selected Python reference
reader first; rust selection stays rejected until the separate new native
capability is complete. This is an execution backend choice, not a reduction of
candidate/input/source domains or acceptance conditions. No new dependency.
Legacy PPO and engineering/component2 remain exact and distinct; formal
multi-seed learning/HPO/final-test remain outside the requested migration scope.

### Original-source authority and visibility

A new closed card-instance Source composition requires the complete original
collection5/Journal6/codec2 inventory, repaired token/layout/rule/action Config
bindings, corrected runtime receipts, unchanged original files and healthy natural
game/Replay closure. It additionally binds the actual owning PPO run, batch,
generation, schedule/learner seats, frozen registry and previous inference4
manifest from the committed lineage. A purpose flag, copied metadata, component
fixture, baseline pilot or standalone artifact cannot supply this authority.
Fresh NN inference collections remain inference records until all of this
source/lineage correspondence is checked. Failed parents/populations never gain
eligibility through a flag update or later passing run.

Online admission additionally requires the SAME live parent run-lock token,
wire4 learner process and lease, shared reservation ledger, exact expected
optimizer/generation steps and frozen control/schedule/registry owners BEFORE
collection output, Simulator or update activity. Mirror the existing
validate_owned_collection live-owner checks, not just JSON linkage equality.
Copied valid controls or a validated complete offline parent cannot authorize
an active run. Keep completed-parent/offline read-only qualification a separate
entry with its complete-state/source/split requirements; it creates no live
learner/lock authority.

Training explicitly selects preprocessing.backend=python and
preprocessing.reader=participant-structured-v2. Its identity records backend,
reader_contract=participant-structured-v2,
implementation=python-reference-player-card-refs-v2 and the repaired full input/
token/rule bindings. Reject inert legacy/structured-v1 readers and rust/old native
before output/allocation; no fallback. Baseline/inference preprocessing remains
inert as documented and cannot qualify a training Source.

Add a closed PPO-batch collection5 variant through a separate composition entry,
with the existing shared inventory/health/publication owner. Preserve baseline
and standalone-inference collection5 records. The PPO variant adds a required
collection_role=ppo-batch and closed linkage containing parent_run_id,
parent_run_path, batch_id, collection_generation, learner_seats, frozen schedule
digest and policy registry/binding digest. Its full linked controls/original
Source are checked against the live owner; none of those serialized fields is
authority by itself. Missing/mixed/extra linkage fields reject. Metadata alone
still declares source_eligibility=not-authorized-by-collection-metadata.

Concrete handoff before implementation: the PPO collection adds exactly
collection_role and linkage; linkage's exact key set is parent_run_id,
parent_run_path, batch_id, collection_generation, learner_seats, schedule_digest,
policy_registry_digest. Digests use canonical serialize_schedule(...,4) and the
frozen label→serialized RuntimePolicyBinding registry respectively. Per-batch
run_id remains the actual collection ID. No lock token, live process object or
Source authorization is serialized into this metadata.

The collector uses a separately named RetainedGameArtifacts.for_registry entry
over the same sealed owner/storage stack. It admits at most6NNidentities (current
learner plus active pool5) and the two canonical baseline bindings. The standalone
§15constructor keeps its two-input bound; concurrent games/workers remain2/4.
The registry entry reconstructs configured neural3 only from closed validated
safe options (pure model, checkpoint, inference and visibility), then requires
exact original options, registry key/policy label and expected manifest digest
agreement before returning or output/Popen. Unknown/noncanonical baselines and a
seventh NN reject. A valid but different current artifact cannot silently rebind.
This is inference-file ownership only; the SAME live parent/learner/ledger check
still precedes it and authorizes online use. No permissive owner flag or generic
capacity override. Independent architecture review accepts this amendment after
both reviewers identify the otherwise failing generation11 three-artifact case.
Test that case, all six current+five artifacts, and seventh refusal through the
actual owner/collector path, not only registry/pruning metadata fixtures.

Reuse existing owned Source snapshot/file inventory, retention accounting and
deadline mechanisms with explicit closed layout/reader context, rather than
copying the Source lifecycle. New public wrappers accept actions3/model3 and
repaired corrected context only; old component wrappers reject these formats.
Read all original participants for ownership/closure validation, but project
only each declared learner seat's own original frame/visible-event prefix,
candidate table, selected original slot/refs and recorded logprob/value. Never
feed opponent-private fields, Replay, management/RNG/rejection data to either
actor or critic. Recreate deterministic tokens from original perceived facts;
do not reconstruct missing original events or recompute old policy outputs.

Here "do not recompute" forbids replacing or inventing the recorded values.
Preserve read-only frozen-policy parity verification of recorded logprob/value
against the original selected row before the first optimizer or owned-RNG
mutation. Reuse existing CPU/cache/full/learner-device/split parity and unchanged
tolerances; preserve CPU/CUDA/shuffle states while constructing comparison
networks and checking. A mismatch invalidates the whole batch, rather than
repairing its recorded outputs or treating current policy outputs as old values.

Own every original parent/control/journal/sidecar until learner completion and
final recheck before generation publication. Recheck path/inode/hash identities
and linked policy manifests; an after-update Source mutation fails the run and
publishes no completed generation. Cleanup preserves the first failure and all
reachable resource owners. No claim of rollback of an already executed optimizer
step, hostile-copy revocation or cryptographic attestation is added.

Through-commit phase is explicit: immediately before commit, recheck the original
immutable collection/numeric inputs and OLD mutable parent/control snapshots.
Finish any per-batch cleanup that can report I/O failure before publication,
while retaining the issued handle, immutable in-memory proof and borrowed live
parent-lock/learner/ledger identity through commit. Retire the old mutable-control
snapshot checks after that successful pre-commit guard; commit deliberately
replaces run/registry names using its existing owned compare/publication logic.
After successful publication, retire the projection without reportable I/O or
an automatic context-exit recheck against legitimately superseded parent names.
Retain immutable-input metadata through commit, but do not append a new post-
publication failure check. Preparation/cleanup failure publishes no generation;
commit failure uses the existing owned invalidation/failure boundary. Shared
parent run-lock and learner leases are borrowed and remain owned by the outer
coordinator. Legacy/component context-exit behavior stays exactly unchanged.

### Numeric rollout and learner wire4

Add explicit episode/rollout4 wrappers on shared raw-token/GAE/normalization/
numeric-shard code with the repaired layout and game binding. Preserve NumPy
bundle1, numeric dtype, optimizer/PPO algorithms and each64MiBarray/shard decode
limit. Validate the whole cohort and conservative raw+token+endpoint memory
envelope before tokenization/update; do not decode an unbounded source first.
Serialized context binds schema4, full contract, rule/action Config/token
signatures, source correspondence and exact Model3 architecture/semantics.
Schema3/component/model2 and old91signature are rejected before arrays/Torch.

Original↔numeric proof covers every learner frame/event boundary+prefix, codec2
table/content/order, selected row/reference identity, recorded old logprob/value,
natural terminal reward, viewer, role/generation and original file inventory.
Keep correspondence verification inside the original Source owner; a self-hash
on a numeric bundle alone grants no learning permission. No learned latent,
cached past-update encoder output or candidate search report becomes a feature.

The Source owner issues the expected raw↔numeric correspondence/file/control
fingerprint through its live projection context to the owned parent process.
Bind that exact expectation and current expected optimizer/generation step in
the wire4 update transaction. The learner verifies numeric4 against it before
parity, optimizer or owned-RNG mutation, and the parent retains/finalizes the
issuing Source owner through completion. An arbitrary proof dictionary or
numeric self-hash cannot issue permission. Reuse the owned projection/process
handoff; do not create a second training framework or authority from DTO flags.

The private issued projection retains actual Source/run-lock/process/lease/ledger
references and an active owner state. Its serializable expectation has the exact
keys recipe, parent_run_id, batch_id, collection_generation,
expected_optimizer_steps, expected_policy_version, collection_manifest_sha256,
numeric_manifest_sha256, numeric_files_sha256, source_correspondence. Recipe is
card-instance-source-update-v1; numeric_files_sha256 covers canonical sorted
relative file→SHA256 entries validated by the Source owner. The correspondence
uses the existing closed fields recipe, control_fingerprint, semantics_sha256,
episode_count, episode_digests, binding_sha256, with distinct recipe
structured-source-cohort-correspondence-v2 and full game semantic declaration.
Its control fingerprint binds the owned original controls/files/source plan;
these serialized hashes never replace the live issuer check.

Wire4 update preserves the familiar request fields protocol_version, command,
identity, batch, batch_digest, expected_policy_version, output, adding required
transaction_id and projection. Projection is the exact expectation above;
numeric_manifest_sha256 equals batch_digest. Parent's typed update method accepts
only the live issued handle; a raw request/proof dictionary cannot bypass its
active owner/expected-step checks. Child checks the expectation and original
numeric correspondence/file identities before parity or optimizer/RNG mutation,
and reply binds the same transaction and generation/step advancement. Existing
initialize/close shapes retain their meaning with version4/game identity. Do not
introduce a new learned/random stream for transaction identity.

Natural±1/0 reward uses the selected learner viewer and only the reconciled
terminal outcome. Interrupted/unsupported/truncated/failed games invalidate the
whole requested16-game update cohort. Empty natural learner episodes remain
valid original episodes, but cannot invent decisions or bootstrap values; an
empty overall learner batch fails explicitly. GAE/Adam/shuffle/clip/minibatch/
suffix-gradient behavior and numerical tolerances remain the existing ones.

Add a strict Model3/game learner wrapper sharing the existing numerical update
implementation through an explicit immutable layout/context. Preserve legacy
Learner calls exactly; do not coerce Model3 into Model2. Wire4 parent/child carries
closed model/execution/use/source identities, numerical file hashes and resource
caps. Reuse process framing, one learner's lifetime CUDA reservation, owned
update transaction, cleanup/reap and paired artifact acknowledgements. Validate
CPU/CUDA visibility before Torch import and the whole batch before optimizer/RNG
mutation. Keep global12GiB and Source+rollout+endpoint accounting unchanged.

### One coordinator and version3 lineage

Refactor only the concrete operations supplied to the existing coordinator;
do not add a second training loop. Replace its Model2-specific labels/schedule/
record literals with explicit values from the selected known operations context,
keeping all legacy defaults and component2 serialized values exact. Add a closed
corrected-game operations implementation and explicit train entry dispatch.
Validation occurs before output, learner or Simulator allocation. Unsupported
rust/old native and mixed Config/model/action/readers fail without fallback.

For the corrected fresh entry, explicitly run closed approved Simulator runtime/
source metadata preflight and whole-configuration memory admission BEFORE the
existing run_lock(create=True) point. Admit known planned learner residency plus
the worst concurrently live NN player roles/covered peak or uncovered temporary
scopes using §15's same formulas. Do not first allocate a learner and rely on the
collector's later remaining-budget check to discover a predictable overcommit.
The collector still checks the SAME actually live ledger/learner owner, and
normal resume performs recorded identity validation before spawning anything.
Keep legacy/component2 ordering/outputs unchanged; this is concrete corrected
entry preflight, not a new resource manager or trainer loop.

Independent dependency review accepts one narrow additional control-plane
exception: training may reuse the stdlib/common-only
gf_simulator_client.card_instance_runtime owner for bounded pre-lock filesystem/
source metadata preflight. No HTTP/environment/CLI dependency or new process
ownership follows; the collector still verifies its actual executed PID before
Create, and a receipt alone is not Source authority. This avoids duplicating
attestation or moving the established owner for one consumer. Revisit the module
location/boundary if it gains HTTP/orchestration imports or broader consumers.

Run/generation/update/registry3 bind repaired input, full execution/Config/code/
Simulator/preprocessing/reward identity, original source proof and numeric4
hashes. Shared file ownership/lineage validation/pair save/retention are reused
with explicit closed version context. Publish completed generation only after
healthy inference4+resume4 acknowledgements, source final recheck and linked
numeric/update metadata validation; a standalone inference file is insufficient.
Maintain the existing run lock, single coordinator,16games/update, snapshot pool
(at most5, every10updates), worker≤4/game≤2, storage and7200/5400/7140s caps.
Snapshot-pool eviction is distinct from artifact retention: preserve the existing
legacy/component2 policy retaining every ten-update milestone artifact, including
those no longer in the active five-entry pool. Reuse ordinary-generation pruning;
do not introduce deletion of inactive milestone artifacts through this migration.

Resume checks all code/config/input/execution/RNG identities before allocation.
Restore all named encoder/actor/critic/Adam state, Torch CPU/visible CUDA RNG,
explicit PCG64 learner shuffle and existing coordinator schedule/opponent-pool/
generation controls. Keep sampling/search streams versioned and independent.
Pause only at a completed normal update boundary; retain original elapsed/wall
budget semantics. A previous failed/ineligible parent cannot become complete
by resuming into new semantics. Root owns new Config/CLI/presets/docs and final
acceptance; delegated implementer owns explicitly assigned source/learner/lineage
modules and tests. These scope boundaries do not authorize native or Tree.

### Verification and requested real acceptance

Test exact original↔numeric parity, source/control/sidecar/role mutation before
and after update, full cohort refusal, reward perspective, no missing-event
repair, mixed formats/old signatures, numeric corruption/limits, cancel/fail-spawn/
reap and late generation publication failures. Check legacy corpus/config/
checkpoint identity and all named encoder/optimizer/RNG resume equivalence.
Before coordinator literal/context edits, freeze bounded legacy/component2
control, schedule and update semantic outputs under fixed owned fixture inputs.
Compare exact versioned key sets, labels, input/reward/numeric identities,
schedule/seat choices, pool and RNG controls after the refactor. Any normalized
timing/path/producer-derived variable must be listed explicitly; do not mask
semantic/version fields with broad hash normalization. Independently reproduce
the pre-edit outputs from the saved code/receipt, rather than trusting a newly
generated same-code golden.

Exercise registry3 retention/pruning with bounded complete fixture chains across
the generation10snapshot boundary and later50/60pool eviction. Verify inactive
ten-update artifacts remain retained and ordinary generations prune correctly.
Verify retained
generation ownership and historical pruned-generation read-only validation,
including missing/foreign/replaced files and interrupted pruning. Two-update
actual smokes do not cover this boundary; no long formal training is required
to exercise it.
Component fixtures remain component evidence and cannot pass a qualified Source
gate solely through test flags. Independent correctness and conformance reviews
must examine learning/evaluation validity as well as byte/schema handling.

Then run the exact requested actual server acceptance: seed17/default attention,
MLP/GRU/Transformer × CPU/CUDA inference, CUDA learner,16games/update, continuous2
updates versus pause after1/resume through2 at normal boundaries. Record complete
source/numeric/lineage correspondence and compare all saved state, retaining
failures. CPU learner gets numerical verification and a real MLP smoke; direct/
DeepSets get real16-game MLP checks and every-model numerical verification.
Existing tolerance, budgets and input/candidate bounds are not loosened. These
acceptance runs are not shortened formal training parents or qualified real-data
three-parent Tree fitting evidence. New native-v2, Tree3/fixed Dataset3 component
fit, fixed512train-decision performance and whole-project/package checks remain
required subsequent migration acceptance.

## 16.1 Actual-input resume verification distinction (design accepted)

Independent advisor recommendation and architecture DESIGN READY; Root accepts
this bounded verification distinction. Helper conformance/results remain
required; the independent-trajectory equality failure is never relabelled PASS.

The first real MLP/CPU-inference/CUDA-learner continuous and pause/resume parents
both complete32natural games/two updates/419learner decisions/16Adam steps and
pass complete original Source/lineage validation. Their independent-trajectory
state-equality comparison FAILS and remains retained as such: final weights/Adam
moments differ, despite exact initial state, coordinator controls and final RNG.
Public records show different candidate tables before the first update at equal
visible inputs/search RNG; different ordered responses then change event order.
Adaptive wall-clock search can produce different cutoff positions. This is not
evidence that restoring an identical learner state is wrong, and seed alone never
promised identical adaptive real trajectories.

Independent advisor recommends keeping real execution health and same-input
state restoration separate. Root proposes the following bounded verification
addendum, with no production Python/source-authority change, reduced input domain,
budget change, tolerance relaxation or replacement of either original parent.

Predeclare the existing actual paused/resumed parent
runs/card-instance-v2-91d5926/ppo-acceptance/mlp-cpu-resume-1 as the first reference.
For every remaining prescribed model/device condition, use its actual paused/
resumed parent as the reference, chosen before diagnostic results. Independently
validate that complete parent's originals, both numeric4 cohorts, lineage and
paired G0/G2 artifacts; preserve all bytes, candidates, recorded values and
correspondence. Do not reconstruct pruned G1 and call it original evidence.

In an isolated assessment-only numerical process, restore that reference's EXACT
G0 paired state into the unchanged GameLearner, then apply its verified numeric
batch0 and batch1 continuously through existing parity/update kernels. Compare
the diagnostic final state against the actual restarted reference G2: every named
parameter/buffer, Adam moment/step/configuration, Torch CPU/all owned CUDA RNG,
PCG64 shuffle and learner counters. Keep current numerical tolerances, require
exact discrete state and separately report exact array equality. Coordinator
schedule/pool/generation/control restoration remains the real/coordinator test
evidence; process elapsed time and run-specific provenance are not learner state.

Retain the ORIGINAL complete-parent, numeric0/1 and G0/G2 manifest/shard/control
identity snapshots through replay and comparison, then recheck them without
resetting a replacement baseline. Use the recorded learner binding, CUDA
visibility and pinned runtime under the same bounded memory/array limits.
Immediately after diagnostic update2, capture all diagnostic tensors/counters,
Torch CPU/owned CUDA RNG and PCG64 state BEFORE loading any expected-G2 model or
constructing comparison networks that could mutate global RNG. Expected state
should be read-only metadata/arrays. A changed original or contaminated capture
fails the diagnostic rather than silently taking a new expected baseline.

The helper consumes originals only through the existing READ-ONLY complete-parent
Source/numeric-correspondence path. It cannot issue an online projection, invoke
a public update with forged authority, patch owner checks, create a PPO run/
generation, register its output as policy or claim Source qualification. Prefer
in-memory comparison; any scratch artifact is explicitly diagnostic, outside
parent publication/eligibility. Full input hashes and results are retained.

The two real runs establish execution/Source health only. The original independent
equality check remains FAIL; a successful fixed-input diagnostic can establish
scoped normal-boundary learner-state restoration, not identical independent
adaptive-game trajectories or strength. Apply this rule consistently across all
six conditions. If the same-input control fails, stop acceptance and localize
the first state difference; search timing is no explanation once inputs are
identical. A separate demand for deterministic successful search would require
its own design rather than shrinking the migration acceptance domain.

## 17. Corrected native Source/token/staging parity (design accepted)

Independent architecture DESIGN READY; Root accepts the bounded native design.
Implementation remains gated on stable accepted §16 Python interfaces and exact
record-shape confirmation. No native capability or Tree scope is enabled by
this design review, and no native runtime test is claimed here.

This increment follows stable accepted §16 Python Source/rollout4/lineage3
interfaces. It adds the explicitly selected rust backend for the SAME corrected
input and owned PPO lifecycle; it adds no learner, game-rule engine, model stack,
new dependency or Tree authorization. Python remains an explicit reference path,
never an automatic fallback after a native failure. Parent preprocessing identity
records the selected implementation; resume cannot silently change backend.

### Closed recipe and compatibility

Preserve existing native legacy/typed-borrowed-v1/structured-v1 markers, entry
points, token values, Source/protocol behavior and binary identity evidence.
Add a separately named card_instance_contract marker:
participant-structured-v2;journal6;rollout4;generation3;model3, a separate
card_instance_signature for the fixed128ed0c6115…68002f layout, and explicit
card_instance_models support for mlp/gru/transformer. They are capabilities,
not Source qualification. Old binaries cannot satisfy them through Python flags
or a changed recipe string. Reject wrong/missing markers, source/binary identity
mismatches, versions or full corrected contract before staging/collection/model
allocation. New capabilities stay absent until validation, tokenization,
persistence/correspondence and the tests below are complete.

Rust uses one explicit closed recipe context for old engineering versus corrected
card-instance layouts. Parameterize reusable pure token/count/array kernels;
retain separate public schema entry points and closed DTO variants. Do not
reinterpret integer hand IDs as card refs, infer a definition-level learned ref,
or serialize opaque ref strings into numerical features. New raw possession
order/kind/unused-learned state, duplicate instances,18slots, classification/roles,
25events and original participant prefixes match the Python reference exactly.
Set/Left events use Guardian IDs; Acted uses catalog definition IDs. Corrected
event306-ID and history387 layout, exact actor/window counts and signature are
the §14.1/14.2 definitions. Legacy history376/296indices and old recipes stay exact.

### Original Source and numeric ownership

Add closed J6/C5/codec2/Model3/control3 inspection on the existing Sources,
CohortSession, typed staging and retirement infrastructure. The concrete key
sets are the accepted §15/16 records; confirm them against those implementations
before native edits, returning incompatible assumptions to Root. Validate whole
original inventory/runtime/role/label/semantic bindings and natural closure.
Policy tokens contain only the acting learner's own original frames/events and
selected row/recorded outputs. Management/RNG/other-private/canonical/rejection
data remains control/diagnostic only; opaque server files may be hashed for
inventory but never parsed into features. No SequenceOptions, regenerated
missing history, rejection teacher or new legality oracle.

Reuse §16's SAME live online parent/learner/lease/ledger admission and Source-issued
update expectation. Native original-file and staged-output owners remain active
through numerical use and final pre-commit guard. Source controls/files are
rechecked before publication; mutable old controls retire and post-commit release
follows §16's no-reportable-I/O phase. A capsule/summary/DTO flag or numeric self-
hash cannot mint a live Source permission. Completed offline parent reading and
component fixtures retain separate authority. Native model3 staging emits the
same numeric4 bundle1 payload, exact per-episode correspondence and file/control
fingerprints; the learner does not accept a weaker native proof variant.

Do not duplicate the codec or whole source lifecycle. Reuse the existing private
array writers, source snapshots, budgets, cancellation checks, retirement worker,
exported-payload leases and no-overwrite publication. Preserve64MiBper-array/shard
decode, record/source/chunk/memory/scratch limits and cumulative admission before
reading. The GIL may be released only while all borrowed data has an owned native
lifetime. No callback into mutable Python policy/Source state while detached.
Timeout/cancel/late success drains or quarantines the reachable owner, preserves
the initiating failure, and publishes no healthy numeric/generation completion.
No earlier private-spool deletion while an exported batch/learner still needs it.

The Python adapter selects a concrete corrected native operations context for
the same _run_ppo loop and game lineage3; legacy/component operations remain
exact. Match original↔numeric facts and full metadata, not just row counts or
floating output samples. Native TreeDataset/RAW remains a separate following
gate because its three-parent/validation roles and Dataset3 authority differ.

### Verification and capability publication

First exercise diagnostic validation/token/staging without publishing the new
capability. Exhaustive/small-space Python-reference comparisons include all
Action families, ref renaming/duplicate definitions, learned7/12/18, state/order,
empty masks, all event vocabularies/Guardian namespaces, perceived historical IDs,
Fog omission and stale/foreign/future prefix refusal. Ref renaming must preserve
numerical tokens while original correspondence still binds the changed raw refs.
Compare exact raw arrays/dtypes/shapes/offsets/source correspondence and persisted
numeric4 round trips; retain established FP32 tolerances where transformation
actually uses floating arithmetic.

Test mixed versions/old binaries, malformed/corrupt metadata/sidecars, source and
control mutation before/after staging/update, crossed roles/issuers/transactions,
over-budget decoding, cancellation, repeated finalization, late publication,
retirement and exported-array lifetime under the same resource limits. Keep all
old native Rust/Python cases and original corpus/index/saved-PPO proofs. Build
and record exact Rust sources/cargo/toolchain/wheel/ELF/capability identity;
installed artifacts must match the reviewed source, not a stale extension.

Only after both reviews and complete parity/ownership checks publish the new
markers and allow reader=participant-structured-v2/backend=rust. Run actual
corrected native common-coordinator update/resume checks with CPU and CUDA
inference/CUDA learner, holding the exact Source→numeric correspondence and
comparing state to the Python reference conditions. These supplement the
requested full §16 six-condition real acceptance; they cannot relabel a synthetic
fit, source fixture or short smoke as a formal parent. Final combined package/
whole-project/performance acceptance and Tree3 remain required.

### 17.1 Concrete native provider and private ABI (design accepted)

Independent architecture DESIGN READY after the numeric append permit and
live/historical plan cardinalities are explicit. Root accepts this exact handoff
after the accepted private DTO diagnostic. Implementation and conformance review
remain required before ABI/capability activation.
Preserve `_GameOperations`, the SAME `_GameBatchSource` and exact
`IssuedGameProjection`; no operations subclass, second public Source issuer or
relaxed process/wire4 authority check. Add one mutually exclusive concrete native
inputs representation behind the existing Source. It supplies verified proof,
ordered episode digests, learner decision count, original rechecks and final
preparation; it never fabricates empty Python EpisodeRecords/_SourceSnapshot.
Replace only record-dependent count/digest access with these explicit accessors.
The existing common numeric4 reader and issued expectation remain authoritative.

The private `CardInstanceNativeSession` constructor fixes the corrected context;
it accepts no caller recipe/qualification selector or Python callback. Plan3 is
closed: schema_version3, mode live_cohort|historical_parent, parents
[{directory,require_complete}], collections[{directory,parent_index}], and
cohort_collection_index only in live mode. This increment requires exactly one
parent and one selected16-game current cohort in live mode (collections length1,
parent_index0, cohort_collection_index0). Historical mode requires collections[]
and no cohort_collection_index; it derives all committed collections in generation
order from the single parent. It may read a complete or still-running/paused
committed prefix with require_complete false for ordinary resume validation;
it cannot issue a live update. No standalone
unlinked collection, baseline or Tree export. Settings keep the existing exact
memory_bytes/scratch_bytes/max_source_file_bytes/max_record_bytes/
chunk_target_bytes budgets and cumulative admission/cancellation infrastructure.

Methods: control_summary, original_config_inputs, authorize, export_cohort,
retire_payload, bind_numeric, begin_update, prepare_commit, finalize_read,
recheck, used_bytes, peak_bytes and close. Retained-Python-byte accounting follows
the existing concrete FFI ownership. `authorize` and `begin_update` only validate
originals; neither grants learner permission. Summary3 is closed with kind
card-instance-source-controls, reader_contract participant-structured-v2, mode,
full five-field binding, parent, collections, authorization_max_bytes,
qualified false and parent_qualification not_performed. The five game-binding
fields are FLAT at summary top level, with no nested binding key. Parent uses existing
compact-parent fields/lineage3/model3/artifact4 descriptors. Each collection binds
index/directory/generation/manifest/config/linkage/schedule/bindings digests,
source_proof/semantics/learner_label/learner_decisions/correspondence and numeric.
Numeric is null for unpublished current projection, otherwise exact
directory/sha256/files. Preserve Python's §16 exact correspondence fingerprint;
do not reuse the old native composite fingerprint.
The compact parent replaces journal5_validated with literal
journal6_validated=false: control metadata alone cannot claim whole-journal
validation; collection validation has its own separately checked evidence.

The acknowledgement3 closes schema_version, summary_sha256, parent and collections.
Python independently checks retained/pruned sealed game artifacts, recorded
execution, schedule/pool/PCG/cadence, registry labels and prior numeric links using
existing game lineage/snapshot kernels. It then acknowledges the full summary.
No summary flag substitutes for those checks. The complete corrected Source
validator must validate all J6/C5/codec2/runtime/role/label/closure semantics; the
earlier DiagnosticHeaderExpected remains explicitly partial and cannot be used
as its completion proof.

Export metadata2 closes schema_version/kind card-instance-cohort-payload/
collection_index/metadata/arrays/source_proof/correspondence. Metadata is the
existing prepared rollout4 metadata with SEMANTIC_KEYS_V3/full game binding;
array descriptors stay key/dtype/shape/bank_index/element_offset/element_count.
Return the existing metadata-bytes plus readonly uint8/int64/float32 bank tuple.
Preserve full JSON selection precision for shared GAE/correspondence; typed
old-logprob/value remain FP32. Reuse `_payload_views`, `_attach_gae`,
`_publish_payload` and the existing prepared writer via a closed rollout context,
keeping old wrappers/serialized outputs exact. Do not parse native-owned original
journals again in Python. A native path still performs common Python numeric4
readback and exact correspondence checks.

Lifecycle: Scanned→Authorized→Exported→PayloadRetired→NumericBound→Updating→
Prepared. Numeric publication targets only the derived fresh parent
numeric/batch-NNNNNN path. Original inventory checks must explicitly distinguish
this one owned append from all original files/directories. Before returning any
payload, `export_cohort` requires the derived target absent, snapshots the already
existing numeric-root directory identity and its complete prior membership/
subtree identities, and opens a one-use permit for that exact target only.
The common coordinator creates numeric-root during initial parent allocation;
the native session cannot create/replace it. During publication the root's own
membership/mtime may reflect only that single append; its identity and every
prior subtree remain pinned. No existing, replaced or mutated prior subtree is
ignored. `bind_numeric` closes the permit, requires exact new target membership
and retains the target directory plus every file identity/hash. Its input has
exactly directory/manifest_sha256/files; it verifies derived path, manifest4/context/
file map and snapshots every persisted file before common readback or issue.
Release exported array views and retire their native payload before that handoff.
`begin_update` rechecks all originals/numeric and permits only the existing
next-artifact append rule. Snapshot/issuer checks repeat before wire/Adam/RNG use.

At the existing precommit hook verify completed transaction and exact optimizer
advancement, native originals/numeric, Python sealed/server/control owners and the
one next artifact append. Complete spool retirement, native core handoff/join and
FFI executor shutdown before marking the same projection prepared. Its small
immutable proof/expectation/receipt and borrowed live parent/process/lease/ledger
remain through lineage publication. No native close, task submission, join,
rehash or old-control check follows commit; release is memory-only. Pre-prepare
failures keep a reachable cleanup owner and preserve the initiating error. Old
component finalize/close stays exact.

Rust-only internal full validators/session/staging/tests may proceed while Python
is frozen after this design gate; no lib.rs export/marker/installed ELF change.
Later adapter files are card_instance_source.py, native_structured_session.py,
structured_rollout.py, card_instance_ppo.py, structured_lineage.py and
native_backend.py, with Root's CLI/config integration. No change to numerical
learner/wire4/PPO coordinator is expected. Final preflight requires exact new
capability/signature/model set plus existing binary identity and Rust-source
identity evidence. Recorded preprocessing identity includes backend rust,
reader_contract participant-structured-v2, implementation
rust-player-card-refs-v2, full game binding and native source/binary identity;
resume cannot switch backend or binary/source identity. Capability remains closed
until complete Python/native arrays/metadata/ownership parity and both reviews.
The exact identity keys are backend, reader_contract, implementation, binary
and the existing five game-binding keys. `binary` is the unchanged binary_identity
record (backend/extension_path/extension_sha256/build/python_abi/gil_disabled);
build.native_source_sha256 supplies its source hash. Historical native inspection
may read a recorded Python-reference parent as that closed original identity;
this does not authorize changing its backend on resume. Live native composition
requires the original recorded Rust variant.

### 17.2 Read-only recorded-cohort export for bridge diagnostics (design accepted)

Independent architecture DESIGN READY; Root accepts the bounded recorded-export
interface and its explicit restaging/retirement handoff. Implementation and
conformance remain required before using its positive bridge tests as evidence.

The internal Rust tests can stage historical cohorts directly, but the approved
FFI's export_cohort is live-only. Genuine completed Python-reference parents
cannot enter a Rust-live plan; they must not be relabelled or rewritten merely
to exercise the Python array/GAE/persistence bridge before capability activation.
Root proposes one bounded read-only method on the SAME concrete native session:
export_recorded_cohort(collection_index, retained_python_bytes=0).

It requires historical_parent, successful existing summary3/ack3 authorization,
an exact derived committed-collection index and no outstanding exported payload.
Reuse the same token/staging/bank/export-lifetime/budget kernels; add no plan mode,
callback, Source owner or issuer. Return payload2 with distinct kind
card-instance-recorded-cohort-payload plus the unchanged ordered metadata/arrays/
source proof/correspondence. Live provider rejects this diagnostic packet kind.
Original source files, path and recorded preprocessing identity stay unchanged.
Historical open retains metadata/digests rather than a resident staged cohort.
On export, re-read J6/columns into a bounded temporary Stage under the SAME
original source snapshot owners, with rechecks before/after. Compare the staged
proof, ordered episode digests, semantics and correspondence to the acknowledged
collection; do not open/adopt a fresh identity baseline. Source/record/scratch/
chunk limits are immutable open-time limits, not per-export narrowed or refreshed
settings. Retain original identity snapshots while freeing each export's
descriptors/banks after view release and successful reportable Stage cleanup,
before returning Authorized or allowing the next collection export.

After pure mode/phase/index/view checks enter Failed before reportable work;
first success enters a distinct RecordedExported phase. Only after every array
view is released may retire_payload perform reportable cleanup (Failed first),
returning to historical Authorized on success so the next committed cohort can
be inspected. Failure permits cleanup only, never a repaired retry. finalize_read
requires no outstanding payload and retains existing final checks/retirement.
Historical bind_numeric/begin_update/prepare_commit remain forbidden before I/O;
there is no numeric-target append permit and no live projection permission.

The diagnostic Python bridge may persist an ordinary numeric4 comparison ONLY
to a fresh owned assessment path outside every original parent/collection. It
retains the same original snapshot owners through NumPy/GAE/write/readback and
final recheck. Exact original correspondence and array/metadata equality remain
required. The output is comparison evidence, not an eligible Source: the same
live issuer/path/lease/ledger checks still refuse it. No private authority flag,
fake runtime chain or metadata-based Source promotion is added. Capability and
installed ELF stay closed until full §17 acceptance. Review this interface before
implementation; positive admitted live update/precommit remains a later test.

## 18. Corrected Tree3 and component fixed-data acceptance (design accepted)

Independent architecture DESIGN READY after the explicit policy IDs, declared
viewer label recipe, loader-purpose split, fixture cleanup phase and separate
token entry/capability are pinned. Root accepts this bounded design. Production
Python edits were gated on §16's source-frozen acceptance, now completed; dependent
implementation follows the separate native gate and explicit file ownership.
Formal real
three-parent fitting, multiple-seed comparisons, HPO and final-test remain outside
this migration acceptance. A component fit never satisfies those gates.

### Configuration, feature and artifact responsibility

Add closed Tree Model3 configuration dispatched before old Config construction;
do not add default fields to Model1/2. Use recipe
`participant-structured-tree-v2`, corrected player-card-refs-v2/participant-
structured-v2 input, corrected action/rule signatures and an ordered feature
signature. Keep existing CPU-only LightGBM parameters, fresh-fit algorithm,
prediction kernels, resource estimates and no-overwrite publishers shared.
Parameterize pure feature/layout and artifact/fit kernels with an explicit closed
old/corrected context rather than copying the whole Tree workflow. Old Tree2
defaults/feature names/transform operation order/artifact bytes remain exact.

Tree3 derives all widths/codebooks from the corrected layout. Preserve signed
FP32 log1p transforms, numeric/presence/category/mask order and original candidate
order: state fields,18ordered possession cards,18opponent learned IDs, corrected
history, then each candidate's kind/target/resources/presence/length/18slot-use
counts/ordered selected-card positions. No opaque refs are encoded numerically.
Current corrected card width64, context1629 and candidate-fixed29 imply
feature width1658+64*positions (2810at18); these are assertions over the derived
recipe, not separately maintained architecture constants. Feature metadata binds
exact names, transforms, categorical indices/limits, input/window/rule signatures.
Selected-row features must equal the same row from full ordered candidate scoring.
Prepared Tree3 context/cards must remain bound to the exact validated tokens and
model from which they were derived. A public dataclass replacement or direct
constructor may not attach another actor's tokens or another window/model to an
old cache. The bounded corrective design is a factory-created, immutable concrete
V3 prepared value with one private token/model/cache binding and readonly public
accessors; direct cached-state construction and dataclasses.replace are refused.
The factory validates closed recipe/window/signature and admits bytes before
allocation; iteration verifies the retained binding and candidate frame identity
before allocating row chunks. Preserve cumulative resident/model/array byte
accounting explicitly when the V3 container is no longer a dataclass. Existing Tree2 constructor/dataclass behavior stays
exact. No new Source/fit authority or general cache framework is introduced.

New inference/training/request/result format4, Fixed Dataset manifest3 and
version3 worker options are distinct from Tree2 formats. NumPy bundle1, per-array/
shard64MiB limits, native LightGBM model representation and existing bounded
fresh-fit process remain shared. A paired trained artifact requires the actual
owned request/completion/model hashes and immutable provenance. `fit_origin`
trained means a real fresh fit occurred; it does not imply qualified game data.
Resaving/relabeling cannot replace origin. Source and pair readers fail before
allocating a model on mixed contract/layout/version or missing pair evidence.
Production qualified-game readers/factories additionally reject fixture origin;
explicitly component readers/workers and the scores-only diagnostic below require
the exact verified component-fixture origin instead.

### A real component fixture, without fabricated execution evidence

The current hand-authored §16 test records are not executed sources and cannot
be promoted with flags or copied runtime receipts. Introduce a narrowly typed
`TreeFixtureSource` owner and a distinct closed component-original envelope.
It stores immutable raw participant frames, original own event prefixes,
codec2 candidate columns in original order, selected indices, declared terminal
outcomes from that viewer's perspective and exact declared policy/artifact
identities. Its distinct label recipe is
`card-instance-declared-terminal-v1`: a closed declared result names winner
participant1|2 or draw, using the SAME participant IDs as frame.actor. Each
endpoint gets+1for that viewer winning,-1for that viewer losing,0for draw.
No implicit zero-based slot conversion is allowed.
It contains no invented server acknowledgements, natural-game
execution receipts, PPO updates, Adam history, C5/J6 runtime publication or live
online-learner authority. Outcome origin is explicitly
`declared-component-fixture`, not observed Simulator termination.

For acceptance freeze before construction: three homogeneous MLP populations
with construction seeds17/29/43, declared train-policy generations0and1,
16train episodes per generation per population and64validation episodes using
declared final-policy generation2 per
population (96train/192validation, nine cohorts). Preserve the full population,
including predeclared zero-endpoint episodes; do not drop empty games, role/seat
swaps or missing files. Construction seeds are not claimed as Simulator GameSeeds.
Fixture generation counts are a component verification fixture, not a shortened
formal parent. Bind the same corrected contract/action/model/input/execution
descriptors across the three populations; policies and final validation identities
must match their declared generation. Fixtures exercise all Action families,
duplicate/unused/learned instances, historical perceived definitions and both
viewer reward perspectives. Typed own-view frame/column/token validators and
selected-row projection are reused without simulating game rules.

The concrete active fixture owner snapshots the complete original inventory,
manifest/population/policy/role/ordering/identity and checks all cohort closure.
It admits cumulative decoded/persisted bytes before reading and stays open through
feature preparation, fitting, pair validation and reportable cleanup. Reuse
existing bounded filesystem snapshots/array writers/fit-child ownership and
completion publication; no second PPO lineage or whole Source lifecycle.
A summary, JSON flag or numeric checksum cannot construct that owner. Its closed
origin propagates through Dataset3, fit request/completion, paired artifact4 and
component worker/RAW receipts: purpose component, qualified false,
parent_qualification not_performed and synthetic_capacity false. The latter only
describes the18-position capacity; it does not claim a game occurred.
All fixture-original final guards and reportable owner close complete BEFORE
paired complete publication; later context release is memory-only.

Qualified game Source composition and public real fitting continue to require
all three complete formally eligible parents and prescribed final-policy
validation sources. They explicitly reject TreeFixtureSource and fixture-origin
Dataset/artifacts before source collection, fitting or real-game worker creation.
Online `_GameBatchSource`/`IssuedGameProjection` cannot be issued by this owner.
The requested component fresh-fit→paired artifact→component worker/RAW path is
private and explicitly typed. Native token/RAW support uses a separate bounded
per-episode diagnostic entry, `CardInstanceFixtureTokens`, over an explicitly
component envelope1. Its separate marker
`card_instance_fixture_tokens=component-envelope1;participant-structured-v2;codec2`
and corrected token signature attest only this exact validation/token/export
entry. §17's J6Source marker alone cannot attest the fixture envelope. Reuse pure
corrected Rust DTO/token/column kernels and owned-array/export/cancel machinery;
do not add a second Sources/CohortSession/PPO-lineage stack. Python's SAME concrete
TreeFixtureSource retains all original/policy/role snapshots, validates ordered
hashes/selected rows and compares native arrays against the Python reference.
This token entry returns structural data only and grants no Source/fit permission.
Codec validation uses a closed corrected component context; fixture tables are
never rewritten to qualified_game. Old native cannot process this entry by
changing a recipe. Capability/extension installation remains gated on both
reviews, complete token parity and source/binary packaging checks.

### Performance panel with independently bound actual inputs

Predeclare a512endpoint seed17 train-side panel from normally completed corrected
acceptance parents before timing; retain every endpoint's original ordered
candidates and complete own recurrent prefix. Selection records parent/cohort/
episode/participant/decision and original input hashes, includes no failed
population, validation/test records, hidden server state or future event.
Use the exact same endpoint inputs for all three NN models and Tree.

A separate read-only diagnostic scorer may evaluate the component-trained paired
Tree on these verified actual inputs. It checks artifact component origin and
input contract/layout/rules/feature signatures independently, returns only scores/
timings and exposes no Selection, sendable Action or game policy factory. It does
not change either input or artifact purpose, promote a Source or weaken the
production Tree purpose check. Compare its scores with component RAW/worker
results on identical tokens. Report that timing used component weights; no
learning strength or real-data fitting claim follows. Count full candidate and
history work, cache/startup separately, and preserve CPU ordinary-decision
p95≤3seconds,2games/4workers,12GiB GPU and all existing per-array limits.

### Acceptance and compatibility

Require exact Python-reference feature names/categories/values and selected/full
row correspondence, all three populations and all192validation records, zero-row
cohort accounting, label-perspective checks, fresh fit and healthy paired format4
artifact, component worker/RAW score equality and immutable-origin round trips.
Refuse missing episodes, crossed policy/split/viewer/roles, mixed versions,
over-budget arrays and original/numeric/request/pair mutation before/after child
execution. Exercise cancellation, timeout/reap, rollback and final cleanup before
completion; no fallible work follows committed complete publication.
Keep all Tree2/native old-data proofs, no-new-dependency startup and old Config
hashes. Independent correctness/conformance reviews, final Rust/Python/CUDA/static/
wheel checks and the pinned512decision panel remain required. Passing this scope
must leave formal real-data three-parent fitting and test evaluation NOT RUN.

### 18.1 Performance population and timing boundary (design accepted)

Independent architecture DESIGN READY; Root accepts this bounded measurement
design. Harness implementation/conformance and actual results remain required.

Existing history376/Tree2 panels include generation, conversion, scoring,
sampling and parent/worker communication in ordinary-decision latency. Their
old component inputs/results cannot stand in for this corrected panel. A
fixed-table score-only timing cannot satisfy ordinary CPU p95≤3seconds.

Before collecting timings, freeze a panel recipe and its complete eligible input
pool: the six predeclared §16 pause/resume parents, ordered mlp-cpu, mlp-cuda,
gru-cpu, gru-cuda, transformer-cpu, transformer-cuda, each complete with normally
ended32train games and passing same-input resume control. No replacement parent
after failures. Within each parent, order committed generations0then1, original
manifest schedule order, exact learner seat, then original decision order. Select
512distinct endpoint identities without replacement using PCG64seed17, persist
the selected identities/indices/hashes before any timing, and retain all earlier
own-decision inputs/events needed by each selected endpoint. Do not claim input
uniqueness merely because different source identities happen to have equal bytes.
All source owners remain open/read-only through extraction and final rechecks;
only public acting-view frames and original learner rows enter model input.

Freeze one final G2 artifact from each model's CPU-inference reference parent for
NN comparison, retaining exact digests/execution bindings. Use the first healthy
default fresh component Tree fit and paired artifact for Tree; no model selection
or refit based on measured latency. All four scorers receive the same512acting
inputs/full recurrent prefixes. The Tree diagnostic independently verifies its
component artifact and actual input binding and still returns only scores/timings.

Report two explicitly separate measurements:

1. **Original-table diagnostic.** All four models score every candidate in the
   exact same original order at each endpoint, with the same own input/event
   prefix. Compare raw feature/token identities and score parity where applicable.
   Include tokenization/history/cache work; report cold/full-prefix and cached
   behavior separately. This is a fixed-input scoring comparison, not the ordinary
   decision acceptance statistic. Never substitute a selected-row-only or smaller
   candidate sample.
2. **Ordinary pipeline.** Reuse the actual corrected worker generation/selection
   path for NN, and an isolated assessment worker for the scores-only Tree
   diagnostic. For each identical original acting input, run the real generator
   with unchanged18positions/32768candidates/262144expansions/50mscount/1ssearch
   and exact model resource admission. Include input validation/frame construction,
   generator work, conversion, full scoring, sampling/argmax calculation and
   parent/worker round trip. The Tree assessment may calculate/discard an index
   for timing, but its scorer/worker exposes no Selection, sendable Action or game
   policy factory. Submit nothing to Simulator. Retain full generated tables and
   counts rather than forcing them to equal the original table or reducing them
   to meet latency. Adaptive search can yield different tables between models/
   trials; record that limit and do not use it to replace the original-table
   diagnostic. Replay complete original own prefixes to establish each recurrent
   cache and charge work at the same real decision boundary.

Run at most2concurrent episode streams/4workers under the unchanged management
budgets and pinned CPU/thread settings. Record all512ordinary endpoint samples;
no failed/slow endpoint replacement. Startup, Simulator communication (absent in
this offline panel), storage/publication and explicit128position stress are
separate. Worker/cache startup and prefix warmup costs are separately reported,
not silently erased. Final ordinary p95 must include all specified decision work
and meet3seconds for each model. Failure, timeout or budget overflow is retained
and addressed, never converted to a pass by dropping rows or changing the panel.

The ordinary assessment uses round barriers across its two episode lanes:
construct/start both workers before any timed round, await both round results
before excluded table serialization/storage/hash work, and retire both before
admitting replacements. Source/startup/storage work cannot run on the event loop
inside a sibling's ordinary timer. Bind the frozen request/completion hashes to
every Tree worker, not just an outer before/after artifact check. Interrupted
assessments retain completed samples and explicit failed/unexecuted rows for all
512identities. Unconfirmed reap stops further admission and retains the SAME
Source, artifact, transport and buffer owners until confirmed cleanup.

Root adopts the advisor's CPU ownership interpretation:8GiB is the joint
Source/controller allowance, not a newly imposed aggregate CPU-process cap.
Original records/controls, independent retained artifact scans, both controller/
transport slices and caller-created frame/event/table/result/publication buffers
share that allowance. Child processes retain their existing validated limits
separately. Record the combined declared envelopes as limits, not measured RSS.
Reserve fixed, complete owner slices before Source/spawn; Tree's existing2GiB
controller allowance per live facade is a conservative starting bound. A bounded
assessment-only NN reply hook may check the announced size before body/JSON
decode under its reserved slice while retaining the64MiB protocol maximum.
Outgoing construction, decoded/parsed overlap and caller copies still require
separate preallocation admission. Derive bounds from all frozen prefixes and
typed candidate maxima before timing; do not substitute observed successful-run
maxima. Insufficient complete admission fails explicitly without domain/weight/
cache/concurrency changes or silently increased limits. This interpretation
changes no production model, inference setting or GPU12GiB policy.

### 18.2 Reachable formal-parent, validation and game-Tree interfaces (design accepted)

Independent architecture DESIGN READY after the literal algorithm recipe,
closed declaration/stop fields, finalization reserve, separate validation budget,
joint multi-owner admission and GameTree init/journal handoffs are pinned. Root
accepts this bounded design. Implementation/conformance and the stated controlled
checks remain required; no formal campaign or qualified real-data fit is run.

Read-only implementation mapping finds a real production gap: old
FinalValidationContext is actions2/Model2/engineering-only; the corrected Source
accepts only16-game PPO batches; corrected64-game validation/Dataset3/public fit
and game-Tree3 routing are absent. Component fixture success alone cannot finish
the migration. Implement the following guarded interfaces while leaving actual
formal three-parent generation/fitting, comparison, HPO and final test NOT RUN.
The independent advisor recommends the small combination of a fixed formal
recipe and an immutable pre-G0 declaration; Root adopts that direction for design
review. No campaign manager or second coordinator is introduced.

#### Formal recipe and declaration

The literal [formal recipe](../contracts/card-instance-formal-ppo-v1.json), SHA256
871fc8d4591e62d302bb2f79d98d474da15cbdb19c86506e5a70557f4ecd67b2,
freezes actual full model/action/optimizer/schedule/reward/retention/preparation/
storage/admission settings. It uses max_updates1000,16games/update,
7200/5400/7140/60second budgets, corrected signatures and default attention.
Closed supported variants are the three NN models, CPU|cuda:0 inference,
CUDA learner and explicit Python|Rust readers. Rust remains subject to §17
capability/preflight acceptance. CPU-learner and direct/DeepSets smokes do not
belong to this formal recipe. Matching three arbitrarily shortened configurations
is insufficient. Runtime readers use the frozen literal, never mutable defaults.
Its closed algorithm recipe has canonical content SHA
318eab51e51ab242c629111634ff25e986c1a0a0ece0d246694fd72a27c90cc9:
Adam3e-4/betas.9,.999/eps1e-8 and all captured named optimizer options; natural
GAE gamma1/lambda.95/float64; whole-batch population-standard-deviation advantage
normalization with added1e-8 and FP32 output; clipped ratio.8..1.2, value.5 and
entropy.01; four epochs/logical128/grad norm.5; exact UUID/PCG/role-stream and
paired schedule recipes; retain0/all every10milestones/latest and newest5active
milestones. Eight unchanged shared learning/schedule function AST digests freeze
their existing operation order without hashing the coordinator's new formal
control branch. The recipe declares the exact Python3.14 AST digest method.
Formal preflight checks those kernel digests and the planned optimizer recipe;
existing initialization/save checks compare the actual optimizer metadata before
G0 publication. Source qualification checks the originals against the literal recipe. Changing
these semantics requires a new formal recipe, not updated mutable defaults.

The recipe lists the only relocatable/identity fields: run name/output path,
Simulator source/binary/config paths and the single training seed slot. Actual
paths/inputs remain recorded and source-owned; relocation does not relax approved
Simulator binary/spec/config hashes. Require exactly one declared seed17|29|43.
Model inference device and reader backend may vary only over their explicit
closed recipe variants; all three Dataset parents must use the same variant.
Compare every other field, including inactive canonical settings and capacities.
Materialize a variant by selecting its literal model profile, changing only its
inference.device and base preprocessing.backend to the explicitly listed values,
and supplying the checked seed/path identity fields. This is a closed comparison,
not a general overlay or fallback to current defaults.

Add an explicit corrected-only formal training entry over `_run_ppo` and the
same `_GameOperations`. Before G0 write a closed declaration1 binding recipe ID/
literal SHA, seed slot, original effective-config SHA/input-inventory SHA and
full input/execution identity. Bind its digest into original run/resume/lineage
identity. Use a distinct closed formal Run3/control variant; current ordinary
Run3, Model2/legacy formats/defaults/hashes stay exact. A declaration is intended
use plus immutable bindings, not a formal=true flag or sufficient authority.
Missing/backfilled declarations, old profiles, fixtures, paused/failed parents
and shortened configurations cannot be formal parents. Existing accepted two-
update records remain read-only-valid and permanently nonformal; do not rewrite
or retrofit them. Resume must use the same declaration/recipe/seed/original
deadlines and ownership. Pure preflight refuses mismatches before child/model/
collection allocation. The formal entry does not execute during this acceptance.
The pre-G0 execution identity is the planned canonical binding/visibility,
numerics/code/native/runtime integration metadata available before any learner
allocation. Actual child hardware is recorded and checked separately by the
existing initialization/Source controls; do not postpone declaration until after
G0 weights exist in order to obtain it.

Declaration1 has exactly schema_version, kind, recipe_id, recipe_sha256,
algorithm_recipe_sha256, seed_slot, config_sha256, config_inputs_sha256,
input_binding, execution_identity_sha256 and identity_sha256. Its kind is
card-instance-formal-parent-declaration; all identity/config digests use the
existing canonical digest, and recipe_sha256 is the literal file-byte SHA.
identity_sha256 covers the pre-formal code/runtime/integration/preprocessing
identity; execution_identity_sha256 covers the planned execution object.
The formal Run3 variant adds exactly run_variant=formal-parent-v1,
formal_declaration and formal_stop; ordinary Run3 has none of them. The closed
formal_declaration reference stores relative_path=formal-declaration.json,
file-byte sha256, recipe_id, recipe_sha256, algorithm_recipe_sha256 and seed_slot.
The SAME reference is included in identity.formal_parent before calculating
resume_identity and creating G0. Verify the sidecar, its base identity and its
G0/resume/generation binding; a later-added sidecar/reference is insufficient.
formal_stop is null while running/paused and a closed receipt when complete.

For this new variant only, persist a closed successful-stop receipt at the
existing between-update stop branch: recipe/declaration identity, reason,
committed generation, observed elapsed/admission state and owned-storage usage/
reservation/limit. Eligible reasons are max_updates, admission_deadline or
storage_admission; preserve the existing branch priority/meaning. Bind the
receipt into healthy run completion and verify its arithmetic against original
limits and the last committed generation. It is not inferred later from cleanup
storage size. Do not require1000completed updates or invent a minimum update
count: any positive healthy committed prefix under the full recipe may complete
normally at admission/storage reservation. Execution/wall/storage overruns,
interrupted batches, bad cleanup and zero updates fail. Pauses remain resumable,
not complete. Historical coordinator traces must remain unchanged; no new fields
or extra semantic behavior in ordinary/legacy branches.
The stop receipt keys are schema_version, kind, recipe_id, recipe_sha256,
declaration_sha256, reason, generation, clock, elapsed_seconds, owned_bytes,
admission_reservation_bytes, max_owned_bytes and max_updates; kind is
card-instance-formal-stop. Clock keys are invocation_started_unix,
invocation_started_monotonic, observed_monotonic, admission_deadline_monotonic,
execution_deadline_monotonic and wall_deadline_monotonic. Derive each deadline
using the existing exact operation order
(invocation_started_monotonic + original_deadline_unix) - invocation_started_unix.
Derive elapsed at stop from that invocation's anchor and original created_unix,
so earlier paused time is included. Preserve the existing rollback guard and
clock tolerances; do not substitute post-cleanup last_unix/storage measurements.
At the captured decision point: execution/wall/actual-storage overrun fails first;
otherwise select max_updates iff generation>=1000, else admission_deadline iff
observed_monotonic>=admission_deadline_monotonic, else storage_admission iff
owned_bytes+admission_reservation_bytes>max_owned_bytes. All earlier predicates
must be false for a later reason; otherwise there is no stop receipt. Require
positive generation equal to final committed updates and original recipe/limits.
Work and the captured stop decision must finish before the original execution
deadline. Subsequent confirmed reap/reportable finalization may use the original
60second reserve up to the wall deadline, under the existing cleanup/storage
bounds. Do not require cleanup itself before7140or discard that reserve.

#### Concrete original owners and final validation

A concrete FormalGameParent owner first validates declaration, literal recipe,
complete original Run3/generation3/update3/registry3 and every committed C5/J6/
numeric4 correspondence under existing lock/snapshot/native ownership. Require
healthy completion, positive updates and the exact final retained inference4
artifact. Keep original qualification metadata not_performed untouched. This
owner allows final-validation collection; it does not yet qualify a Dataset.

Corrected actor-only final validation borrows that SAME parent owner and the
same empty actor ledger, with no learner/lease/update permission. Reuse
validation_groups(seed, final_label, actions_version=3):32pairs/64games, final
policy versus prescribed opponents, seats and final policy bound exactly. Caller
configuration/input origins must equal the parent's originals, apart from the
explicit projection run.split=train→validation. This equality is PER PARENT,
not equality between the seed17/29/43 input files. A freshly admitted validation
invocation gets its OWN bounded7200/5400/7140/60second wall/admission/execution/
finalization deadlines, capped by its caller's fresh deadline. Those same budget
amounts do not reuse already-spent training deadlines. Training creation/deadlines/
stop receipt remain immutable. Record validation start/deadlines/cost separately;
never reset a deadline per game/group. Use existing corrected runtime/Environment/
policy sealing/worker admission.
Hold parent/final artifact/server-config snapshots through collection/reap and
final source validation. No callback or flag may bypass the normal Environment.

Collection5 gains a distinct closed final-validation variant, not a changed
batch cardinality parameter. Keep collection_role ppo-batch and its seven-field
linkage exact. The new role ppo-final-validation binds parent run ID/path,
batch_id validation-final, exact final_generation/final inference SHA/policy
label, learner-seat map, schedule digest and registry digest. It requires split
validation and all64natural healthy games, permits final generation1000, has no
invented PPO numeric update, and uses Journal6/protocol3/codec2 unchanged.
The exact final-validation linkage key set is parent_run_id, parent_run_path,
batch_id, final_generation, final_inference_sha256, policy_label, learner_seats,
schedule_digest and policy_registry_digest. batch_id is literally
validation-final. final_inference_sha256 names the original final inference.json
file, not the generation record; policy_label is its exact retained neural
binding. learner_seats maps every scheduled episode to its selected final-policy
participant1|2, without implying a resident learner. Parent-original snapshot/
declaration ownership remains with the same FormalGameParent; linkage supplies
no authority alone. Do not accept collection_generation or extra/missing keys
in this variant, or add any final-validation keys to the ordinary batch variant.
Share raw inventory/runtime/Replay/frame/event/candidate/selection validators
through a closed collection-role context. The original16batch path and old
engineering validation remain exact. Returned/persisted report timing ordering,
late success, cancellation, reap and final cleanup retain the existing owners.
Native receipts are structural evidence, not formal or Dataset authority.

Formal-control completion additionally owns a retained invalidation boundary.
The explicit formal training entry keeps one private completion owner around
the SAME coordinator call, including its post-persistence wall/storage checks,
run-lock exit and reportable final release. Publishing formal complete uses a
known owned temporary descriptor/inode and atomic replacement, retaining that
ownership/snapshot until every reportable terminal operation has succeeded.
A late failure first invalidates that owned completion and retains any uncertain
cleanup owner before returning the initiating error; a best-effort failed-state
rewrite alone is insufficient. Close errors obey§14's one-attempt rule; never
inspect/retry an uncertain numeric descriptor. Recover only through a fresh
O_NOFOLLOW writable regular descriptor matching the known original inode.
Ordinary/legacy completion paths and traces remain exact. This is one closed
formal JSON-control owner, not another training loop or publisher framework.
The existing two terminal-descriptor/invalidation helpers may move byte-exact
to a stdlib-only common leaf, with Tree's existing private names retained as
aliases. Share that low-level behavior without importing Tree workflow code
into PPO or changing old successful bytes. Architecture review of this concrete
handoff is required before dependent formal-control implementation.

A concrete VerifiedGameTreeDataset3 composes exactly the ordered formal seeds
17/29/43 and their three complete final64validation Sources. Require distinct
original parent identities, homogeneous model/action/input/execution/recipe/code
conditions, allowing only explicit seed/name/path/run identities. Include ALL
committed16-game training cohorts and all192validation games, including empty
endpoint games. Project only the original selected learner-view facts, preserving
row order and terminal±1/0 perspective. Observed natural-outcome label provenance
is distinct from the component declared-terminal recipe. Retain originals through
row preparation, fitting/reap, numeric/pair checks and reportable final cleanup.
Only this concrete verified owner authorizes qualified fresh fitting; no summary,
dataset hash, component owner or edited origin flag can substitute for it.
Acquire original parent/validation locks in deterministic order by resolved
filesystem path bytes, deduplicating physical roots and borrowing only a SAME
already-held lock token; release in reverse order. No raw decoding or output
allocation precedes the required ownership. Admit the cumulative native/Python
metadata, active episode/tokens, arrays/shards, fit rows and child workspace over
ALL retained owners under one caller allowance. Sequential staging does not
permit charging every parent its own full budget; reuse existing shared-budget/
retained-byte mechanisms and stream bounded episode/cohort views. Keep lock and
snapshot ownership through reportable close before completion publication.

Concrete validation-lock mechanism remains an implementation design question:
creating a root .lock would violate final64's exact original inventory. Root
proposes resource-only read locking through an independently opened no-follow
canonical manifest descriptor with exclusive advisory flock and retained
device/inode evidence. This preserves the original inventory; parent roots keep
the existing run_lock. Any same-process borrow must use the SAME actual retained
lock token, and all six roots still acquire in deterministic resolved-path byte
order and release in reverse order. The manifest lock is resource ownership,
never Source authority. Architecture review must settle this mechanism and its
one-attempt close/uncertainty handling before Dataset3 implementation; do not
silently ignore arbitrary .lock files or widen the Native Actor64 request.

Root's initial concrete Dataset/fit proposal for architecture review was
`../design/card-instance-game-tree-dataset-v1.md`, SHA
d9f671aeed08877301276ebba06a1025a0a441761a6bcc6900b852b156a437b8.
Production implementation was withheld at v1 pending the complete v2 seams
below. Root pins the following directions:

- Keep all six logical memberships, deterministic physical-root lock dedup and
  reverse retirement. Support borrowing only a COMPLETE already-ordered custody
  set, never an arbitrary subset or copied token. Validation uses the independently
  owned canonical manifest flock; parent run_lock stays unchanged.
- Real Dataset3 kind is card-instance-game-fixed-sources, source_kind is
  card-instance-formal-ppo-originals, selection is
  all-explicit-learner-accepted-natural-return-v3 and label_recipe remains
  behavior-selected-natural-return-gamma1-v1. Real source_origin has exactly
  purpose=qualified_game, synthetic_capacity=false,
  outcome_origin=observed-natural-simulator and
  qualification_recipe=three-formal-parents-final64-v1. No qualified=true flag
  supplies authority. Adopt the proposal's closed parent/generation/collection/
  counts/source/cost fields subject to final correspondence below.
- Final64 correspondence has exactly recipe, input_binding, manifest_sha256,
  episode_count, episode_digests and binding_sha256. Recipe is
  card-instance-final-validation-correspondence-v1; count is64 and digests are
  ordered selected original _episode_correspondence_digest values, including
  empty episodes. binding_sha256 hashes the other fields. Training keeps its
  existing original-versus-Num4 correspondence. No validation numeric update or
  artificial GAE/declaration is introduced.
- The effective Dataset/controller/fit allowance is at most8GiB and includes
  ALL retained owners, active views, controller rows/packing/publication and
  the full declared fit-child allowance. Keep the default child/model2GiB cap;
  an explicit configured cap is never silently lowered to the remaining space.
  Refuse before allocation if it cannot fit. A game-only request managed_memory
  object has exactly recipe=joint-game-tree-fit-v1, total_bytes,
  controller_reserved_bytes, child_limit_bytes and reserved_peak_bytes. These
  are conservative envelopes, not RSS. Preserve the existing5*raw/fixed/packing
  arithmetic, name conservative overlap and enforce it in the same fit child.
  Local feature/model checks charge their relevant live objects; unrelated
  Source residency remains charged to the joint governor, avoiding accidental
  collapse of the8GiB source allowance into the local2GiB model cap.
- Use explicit game-dataset build/inspect and fit-game-tree composition, with
  corrected Tree3 configuration dispatch before legacy defaults. Reuse current
  training settings and add no second learner. GameTree3 runtime routing is the
  following required increment, not a claimed implemented stub.

Paths-only Dataset reopening cannot import CLI Config types or reload historical
configuration through mutable defaults. Prefer a guarded parent-derived readonly
contract/opener over the SAME lower final64 validator: derive exact config,
actions/settings/projection/input origins from the actual retained formal
parent. Live actor collection still checks its original caller LoadedConfig and
keeps its existing API. This seam, pre-entry other-owner accounting, base-versus-
loan release and guarded train/final raw borrowing must pass architecture review
before dependent Dataset code. A hash/summary/contract value alone remains no
Source or fitting permission.

Refinement v2, accepted after independent architecture DESIGN READY, is
`../design/card-instance-game-tree-dataset-v2.md`, SHA
2a0d227eff97129999105696976db7dd300027a7c766772d0daee40081b4b8c1.
Root selects complete original-history DTO borrowing, using the SAME retained
native Sources/restaging/whitelist, rather than reusing an NN-window aggregate.
An additive historical-only export_recorded_episode_payload takes collection
and selected-episode ordinals plus retained Python ownership; the existing
cohort API stays exact. It may be checked against genuine short historical
parents as a read-only diagnostic without granting formal/Dataset eligibility.
Private actual-parent forwarding supplies source membership and custody.
Exact training-packet metadata and new gate remain subject to the native seam
design, not a reason for a Python journal fallback or narrowed Tree window.

The game-only fit bootstrap receives a fourth sanitized decimal child-cap
argument, admitted before request decoding/heavy library initialization and
checked exactly against request/model resources. The old three-argument path
and bytes stay exact; another context with that extra argument rejects. This
is resource control, not Source/fit permission. V2's full-keyset/roundtrip
parent-derived readonly contract, pre-entry other-owner charge, base/local
resource loan, exclusive intrinsic counters and complete custody mechanism
are accepted for implementation. Exact native training-packet metadata remains
a separate pre-code gate before that dependent ABI implementation. Root assigns
new pure Dataset/lock/shared-Tree files separately from the native owner's
existing parent/final/source seams, in isolated copies with selective handoffs.
No missing dependency or permanent stub counts as completed migration.

Independent v2 review additionally pins a distinct early child-local check:
controller_reserved+full child cap fitting8GiB does not prove the preserved
child5*raw/fixed/packing/own metadata/decode/log/model envelope fits its configured
cap. The controller derives and refuses that bound from VERIFIED row counts and
settings BEFORE private row preparation/spawn, and the SAME worker repeats it
before numeric decoding/Dataset/Booster allocation. Keep both equations, original
formula/default2GiB and explicit conservative overlap; no new field, automatic
cap adjustment or row reduction is permitted.

The concrete original-history/resource handoff is proposed in
`../design/card-instance-dataset-native-source-seams-v1.md`, SHA
ee2f924eddbc4bf6ae6ce330b338d86c573ca3cfaadb19a65427ee14f3bff050,
accepted after independent architecture DESIGN READY. Its explicit recorded16
envelope has18keys; the enumerated keyset governs without an extra filler field.
It pins the closed recorded16 packet1,
unchanged final64 body, actual logical DTO borrowing, initial-external native
floor and exclusive actual-declared counters, base/local custody and physical
opening/Source retirement. Root chooses native preflight marker
card_instance_original_history=journal6;codec2;episode-dto1;recorded16 with the
existing corrected signature/ABI/new-method checks; publication is separately
gated on validation/parity and both final reviews. It grants no Source permission.

The following required runtime increment has a separate bounded Root proposal,
`../design/card-instance-game-tree-runtime-v3.md`, accepted after architecture
review. It reuses the corrected participant/environment/
generator and Tree kernels, with actual qualified-pair custody in the controller,
strict source-free own-policy declaration/private child handle, null Tree
selection statistics, distinct Journal6/Collection5 metadata and native RAW
branch. Public Tree3 Config dispatch and exact compatibility stay in scope.
Genuine qualified Tree gameplay cannot be claimed from component fit/scoring;
formal sources/real fitting and their positive game execution remain deferred.
The v2 dependency is acyclic: request/provenance binds inference/model,
completion hashes those outputs, then the runtime declaration/journal/collection
binds completion. Inference never embeds completion_sha256. Only the controller
validates actual paired completion; the child treats that supplied digest as an
opaque binding and checks local inference request/source/model/features.
Tree-only sanitized argv capacity admits the first frame before JSON decoding;
baseline/NN argv and defaults stay exact. Journal6's tree branch and Collection5
tree_inputs use the closed four-field own external metadata record, with the
exact12-field own-policy options. Native processing additionally requires
card_instance_tree_raw=journal6;lightgbm3;inference4;codec2 plus the existing
corrected signature/ABI/API checks. Python gameplay gains no native dependency.
The marker remains unpublished until RAW checks and both final reviews pass.
The additive corrected RAW entry in
`../design/card-instance-journal-raw-entry-v1.md` passes architecture DESIGN READY.
StructuredJournalTokens.card_instance selects the existing corrected parser
under the same owner/export path; the old constructor stays Journal5-exact.
Corrected-only sticky diagnostic/recheck failures and actual close confirmation
are mandatory; core removal or empty retry cannot erase retirement uncertainty.
Root adopts the bounded advisor recommendation for new RAW allocation scope:
portable semantic/execution/cap/cost/identity checks stay exact, while Native
RAW's actual parse/copy/Stage/export quota is charged to its existing budget.
RAW allocates no Booster and never attests Python's CPython graph/alias-specific
allocation admission. Python inference's local guard is unchanged. Inputs
admitted by both must have semantic/token/output parity; physical boundary
admission may differ explicitly. This clarification requires narrow architecture
confirmation before final freeze; no cap increase, guessed reservation or new
persisted field is permitted while the new capability remains unpublished.

The bounded activation design in
`../design/card-instance-native-capability-activation-v1.md` now passes
architecture DESIGN READY and Root accepts its scoped implementation GO.
The five existing literal markers declare structural support; all actual
parent/Source/original-file/custody/pair guards remain mandatory. Root accepts
explicit coverage debt for successful genuine formal-parent/final64 four-method
FFI, which remains NOT RUN under the user-deferred formal execution scope.
The first genuine formal parent must exercise all four methods against complete
originals, zero endpoints and retained exports before any formal end-to-end
acceptance claim. Complete64 parser/DTO/owner unit controls do not replace that
future proof. Activation still requires marked positive preflights/refusals,
affected regressions, both delta reviews and final package/install verification.
Those activation conditions now pass: both delta reviews READY, final whole
pytest4034PASS/183explicit skips, Rust147PASS/4ignored,43actual CUDA cases, five
complete legacy proofs and source/package/installed identity checks. Main's
release/MIMALLOC wheel exposes the exact five literals and passes47installed
preflight/refusal cases. Root accepts this structural adoption; the genuine
formal four-method coverage debt and user-deferred experiments above remain.
See `../verification/card-instance-v2-adoption.md` for preserved failed attempts,
scope, exact artifacts and rollback evidence.

Reuse the shared Dataset/Tree3 features, fit child, paired format4 publication,
copy/load and inference kernels with a closed real-source context. Add reachable
CLI/config/library routing for formal declaration, actor validation, Dataset3,
fresh fit and qualified game Tree3 worker/evaluation. GameTree3 requires the
verified real-source trained paired origin; component/toy/diagnostic origins
refuse before allocation. Fixture and real-source Dataset3 remain disjoint
closed variants. This supplies future production interfaces without generating
formal parents or promoting the acceptance fixtures.

Qualified GameTree3 uses the SAME corrected worker/environment/transport loop,
with a closed lightgbm/version3 init/options discriminator and retained paired-
artifact owner. The controller checks qualified real-source trained provenance
and seals the complete pair before model/worker allocation; retain it through
reap/final publication. Send the child only own model/config/feature descriptor,
artifact hashes and small paired digest references, never raw Dataset/parent
controls, GameSeed/RNG, other-private originals or training feature arrays.
The child validates that bounded own-policy declaration against its sealed model
files; it cannot issue Source/fit permission. Journal6 adds a distinct closed
Tree metadata branch (model_id lightgbm, model_version3, tree own-policy metadata,
logprob/value both None). Existing baseline and finite-valued NN branches remain
exact. Tree metadata binds safe options/inference4/paired digest references/
owned file identities without copying raw Source controls into journals or
features. The pure recipe/cost/feature validators and native RAW parser must
recognize this separate branch. Formal NN/PPO Source continues to require an
explicit NN learner and rejects Tree as learner. Component/diagnostic workers
use their separate input/origin contract; no second game or learning loop.

#### Remaining concrete fixture choices and checks

Use one explicit selected viewer per fixture envelope, with both viewers covered
across the frozen populations. Declared policy IDs are the digest of the fixed
fixture-policy recipe, fixture/population IDs, declared generation and actual
component inference-manifest SHA; no fake PPO update or elapsed generation cost.
Tree3 leaf feature-name components percent-encode UTF8 bytes outside ASCII
letters/digits/_/- as uppercase%HH, then join known hierarchy segments with dots.
This is injective, keeps colon-bearing role fields legal for LightGBM, and leaves
Tree2 names exact. Bind raw-field mapping and ordered escaped names in the new
feature signature. Component worker/RAW accepts original ordered tables without
new search; ordinary performance generation remains the separate §18.1 path.

Before any formal execution test exact full-recipe admission; shortened budgets/
max_updates2, absent/backfilled declaration, wrong seed/profile/origin/caller
refusal; every normal stop/overrun/pause branch; exact final policy/64membership/
generation1000; actor-only ownership and mutations; three-parent completeness/
all train rows/split-role integrity; fixture rejection and paired-origin
preservation. Controlled I/O positives verify interfaces/predicates only and
remain component evidence. Independent architecture review is required before
this addendum's dependent implementation; formal production collection, Dataset3
qualification and real-data fitting remain NOT RUN.

Native formal-control adoption reuses CardInstanceNativeSession and the SAME
Sources/CohortSession traversal. A closed formal Run3 discriminator validates
the original declaration sidecar through Sources with byte hash/stat identity,
the SAME run/identity/G0 reference, base identity excluding ONLY formal_parent,
the packaged literal recipe/full materialized configuration and normal-stop
clock/limit/counter rules, including the original.1second allowance. Native
numbers retain their Python-compatible representation/arithmetic order.
The compact summary adds only the three formal fields inside parent.state for
the formal variant; ordinary and legacy summary bytes remain exact. It remains
structural evidence without qualification authority.

One private _open_game_parent_inputs context in card_instance_source extracts
the existing complete-parent traversal. It yields protected state/records and
the actual retained _NativeGameInputs when backend is Rust. Existing
validate_game_parent enters/exits it with the same return/ordinary-close
behavior. FormalGameParent enters it through its own ExitStack and retains the
SAME native owner through all use; recheck/final_binding require its live
original/numeric guards plus protected Python originals and lock. Joint caller/
sidecar/materialization ownership is admitted before native allocation. Final
guards, native final-read/retirement, executor join and reportable close precede
lock release, with sticky failure and reachable cleanup ownership. No ownership
is reconstructed from a compact summary.
A single ExitStack would continue to release the lock after input-close failure.
Use separate private input and lock stacks inside the SAME parent: finish native
final-read/retire/join/close first, release the existing lock stack only after
concrete cleanup confirmation. Otherwise retain both SAME owners through the
cleanup exception for retry. A narrow guarded native cleanup_complete accessor
reports actual closed/session/executor state, never a summary flag. Preserve the
successful Python-only path and avoid a new Source or lifecycle framework.
The separate native marker card_instance_formal_controls has exact value
run3-formal-parent-v1;declaration1;stop1 and stays absent until parser/sidecar/
source-retention parity and both reviews pass. It attests structural support,
never formal/Dataset eligibility. Same-backend/binary resume and early capability
refusal remain; no Python fallback. Actor64 role adoption is separate and may
borrow this seam without widening the16game PPO role. Architecture confirmation
of this concrete handoff precedes dependent implementation; no genuine formal
Source, learning run or game is invented for verification.

Actor64's concrete handoff uses a distinct GameFinalValidationContext/Plan and
retained GameFinalValidationSource. A live SAME FormalGameParent supplies its
protected final binding and originals, with an empty SAME actor ledger and no
learner/issuer/update role. Reuse the existing split-only LoadedConfig projection:
replace run.split with validation, recompute effective JSON/config digest, retain
the original input byte-origin inventory unchanged. CLI is explicitly
validate-formal-policy; ordinary validation/training commands remain exact.

Only the closed final-validation C5 variant adds config_projection,
validation_invocation and elapsed_seconds. The nine linkage keys stay exact.
config_projection has exactly recipe, parent_config_sha256,
parent_config_inputs_sha256 and projected_config_sha256; recipe is
run-split-train-to-validation-v1. validation_invocation has exactly started_unix,
started_monotonic, caller_deadline_monotonic, admission_deadline_monotonic,
execution_deadline_monotonic and wall_deadline_monotonic. Derive the fresh capped
7200/5400/7140/60 budgets once; caller duration insufficient for finalization
reserve rejects before allocation. Preserve training clocks unchanged.
Reuse the report wall_seconds/C5 elapsed_seconds cost kernel and ordering:
persisted report wall <= manifest elapsed <= returned report wall <= actual
capped invocation duration, every other report field equal. Returned final-role
report may advance only wall_seconds after collection publication. This records
a validation invocation cost, not a rewritten TOML or training cost.
Capture final-role C5 elapsed_seconds inside the existing publication after the
report write/hash and before the manifest write, using the frozen invocation
start. Do not substitute a precommit value that omits report storage cost.
The ordinary/engineering commit branches stay byte/control-order exact.
The final-role collection must retain the SAME existing publication/invalidation
owner through the returned-wall budget check, live complete-Source acceptance,
parent/original final guards and reportable actor-context cleanup/reap. A private
nonserialized finalized-collection token may carry that actual owner to the
concrete context; a caller callback or persisted flag cannot replace it. The
parent lock and resource-exclusion slot stay held through terminal retirement.
On late failure, invalidate the owned complete manifest before propagating the
primary error, preserving uncertain cleanup ownership and§14 descriptor rules.
The ordinary collection commit/retirement path remains exact. This extends the
existing closed final-role lifetime only; it adds no publisher framework,
sidecar or authority path. Readonly reopening must never revive a failed live
completion. Architecture confirmation precedes this terminal-lifetime hook.
Canonical manifest.json must remain absent while final-role acceptance is
pending. Use existing manifest.json.tmp as the actual owned pending control
file. Live validation borrows its exact descriptor/inode/bytes through the SAME
nonserialized publisher token and applies the shared raw validator; no caller
JSON, permission flag or fabricated16-game view is substituted. Readonly
completed-source opening accepts only canonical manifest.json. Expose that name
exclusively after Source/context guards and cleanup. CLI also completes its
OWNED formal-parent release before publisher.finish; a library borrowed-parent
path never closes its caller's parent and finishes only its own loan/control/
Source lifetime plus final parent guards. Retain owned invalidation through
publication's last reportable steps. Exposing canonical healthy C5 before owned
parent release leaves a lock-gap reader race and is not accepted. Architecture
confirmation precedes this concrete pending-control/terminal composition.

Common pure closed role descriptors carry16/seven-key PPO batch or64/nine-key
final validation structure only, never authority. Share existing raw/header/
Replay/runtime/health validators under the explicit closed role; preserve the
old ordinary/engineering branches. New validation modules and structured_source
may compose this role without editing the native parent's retained-context files.
A readonly completed-validation opener must borrow the SAME active formal parent
and verify original role/schedule/final-policy/config/health/inventory closure;
later Dataset3 cannot depend on a past live collection callback or a metadata flag.
It exposes cloned facts/selected own records and recheck/close only, with no PPO
numeric/update or Dataset qualification authority. Collection acceptance and
readonly reopening share that one original validator. Pure/controlled positives
remain unadmitted evidence; no real formal64 games are executed here. Native
actor64 role adoption is a following separate parser/retained-source increment,
without widening PPO16. Split-only projection retains its original reader
backend. Until distinct native64 support is adopted, a Rust-reader Actor64 entry
must reject BEFORE runtime/output/model/actor allocation, not after64games or
by silently checking Python under a Rust identity. Its separate planned marker
is card_instance_final_validation=collection5-final64;journal6;codec2 with the
existing corrected signature; it stays absent until closed-role parsing,
retained collection ownership, parity and both reviews pass. Formal-control
capability alone never attests this role. Python-reader Actor64 may proceed.
Architecture review of this concrete handoff precedes
dependent implementation.

Actor borrowing additionally needs a guarded parent budget seam, proposed before
implementation. FormalGameParent.resident_bytes reports intrinsic retained
Python/native/control/sidecar/canonical-original and bounded guard ownership,
excluding only its separately tracked external reservation; it is not RSS or
qualification. admit_external(n) sets that nonnegative external reservation and
checks intrinsic+n+guard scratch BEFORE caller allocation. Callers may lower or
release it only after the previously charged buffers/owners have retired. For
Rust, a narrow NativeGameInputs joint-owned counter combines existing extra/base
and actual session used_bytes; subtract only the known tracked external amount,
never original sidecar/control charges. Updating native caller growth includes
original scan/materialization changes plus n, then rechecks SAME originals.
After caller buffers have retired, strict admit_external(0) is a cleanup-only
release of the tracked external reservation even on failed/expired parents.
It performs no guard/native I/O, clears no poison and restores no operational
permission. A failed native caller lease may remain conservatively charged until
actual native close. Nonzero admission still requires an active owner; releasing
the actor's exclusion slot additionally requires confirmed associated cleanup.
Failures poison the same parent and preserve cleanup ownership; neither summary
reconstruction nor a larger original allowance is permitted. Architecture
confirmation precedes these cross-module accessors.

Actor64 implementation review finds that an exact reservation after building
the plan is too late. Root and architecture accept a narrow guarded read-only
byte-size accessor derived directly from protected canonical parent originals,
without parsing, mutation or additional authority. Before parent-state clones,
projected Config, the64-game schedule/registry and signature serialization,
admit their complete conservative simultaneous-copy/scratch bound through the
SAME parent.admit_external. Retain the later exact checks and unchanged limits;
a near-cap refusal must precede those materializations. Diagnostic formatting
also belongs inside its best-effort guard: attach the SAME cleanup owner and
preserve/invalidate from the first failure before formatting any secondary
exception. A throwing __str__ or add_note must not skip pending-publication
invalidation or replace the initiating error. Both initial blocking findings
are now independently reproduced and closed. A further repeated-validation
growth finding is also closed: retain original size-only ceilings and refuse
mutated internal/caller controls before every signature/asdict/JSON/copy or
re-admission, without enlarging reservations. Exact equality remains mandatory.
Both final reviews are READY;351MAIN scoped checks, Ruff and mypy152pass.
This accepts the Python interface, not genuine formal Source/game execution.

To avoid a second collection validator, Actor64 may extract _game_manifest to
new card_instance_collection_source with a closed role argument and unchanged
PPO16 default behavior. The native-parent writer keeps card_instance_source
ownership and merges only a small alias/removal patch after its retained-parent
seam is stable. Actor code imports the shared leaf directly meanwhile. Final
production has one validator, no normalized/fabricated16-record view and no
caller-provided validation callback. Architecture confirmation precedes this
narrow responsibility extraction.

#### Native final64 retained-source handoff (design accepted)

Root adopts the bounded proposal after independent architecture DESIGN READY:
`../design/card-instance-native-actor64-v1.md`, SHA
85ab7d3521a0957e76ab552a16281986915e469f4cc2d7c8ab97413623b343f2.
Implementation and both final reviews remain required. Extend the SAME corrected
native
parent/session with one closed final-validation slot; existing PPO16 and legacy
defaults stay exact. Four private methods inspect_final_validation,
export_final_validation(episode_index), recheck_final_validation and
close_final_validation retain the original Sources/budget/Control, use ordered
per-episode exports and grant no numeric/update/issuer/Dataset authority.

The closed schema1 request binds retained parent run SHA, directory, exact
expected linkage/projection/invocation/schedule/registry and optional returned
report. Recorded invocation deadlines are historical evidence on canonical
reopen; active inspection/export uses the retained Control/caller deadline.
Canonical mode reads manifest.json. Issued-pending mode reads only the
SAME publisher's actual manifest.json.tmp, requires canonical absence, and
independently checks fixed directory/file identities, size and SHA evidence
obtained through a separate no-follow read handle. Serialized evidence supplies
no publisher authority; Python retains the actual issued token. The native
slot validates original equality/parent/final artifact while Python computes
the prescribed schedule. Neither side implements a second scheduler or reader.
Canonical forwarding may use the existing retained SourceOwner/SealedScan for
the bounded management-only C5 manifest read needed to recover its historical
invocation and prescribed plan. Admit stat-sized decode/control ownership before
reading or parsing, retain original hash/stat, and require its SHA to equal the
native summary through final cleanup. Native alone traverses J6/codec/Replay/raw
inputs; this management read is not a Python journal fallback. Pending forwarding
also retains the actual issued token and independently owned fixed-name evidence.
The existing Python-compatible pure numeric arithmetic helper may be reused
crate-internally with unchanged operations and tie behavior.

Existing raw token banks omit original frame/codec documents. Reuse the existing
SAME-Sources Rust journal restaging pattern for one selected episode at a time,
bracketed by original guards and exact inspected correspondence. At its existing
accepted-decision staging point, additionally spool only original own event
bytes/prefixes, frame_dict bytes, exact codec2 descriptor/columns, full-precision
Selection and natural endpoint. Export those plus existing typed token banks
under one bounded descriptor inventory. Reconstruct genuine immutable64
EpisodeRecords through common constructors, comparing correspondence and
tokens; do not add a Python journal reader, dummy records or padded PPO16 views.

Admit request/control/decode overlap, all earlier retained own DTOs, current
packet/Stage/export lease and next DTO growth jointly before allocation under
the existing parent allowance. Source/record/array/shard64MiB bounds remain.
Export each ordinal once; only one packet may be live, and its views must retire
before the next export or healthy slot close. Stage retirement/join precedes
packet return. An admitted failure latches cleanup-only state; final original
guards and confirmed slot retirement precede external reservation/borrow/parent
release and pending publication. Uncertain close retains the SAME reachable
owners and primary failure. Separate final64 capability stays absent until
parser/DTO/raw parity, fault checks and both independent reviews pass. Genuine
formal parents/games remain NOT RUN during this implementation acceptance.

### 18.3 Component artifact4 and fresh-fit record handoff (design accepted)

Independent architecture DESIGN READY; Root accepts these concrete details for
the shared Tree2/Tree3 artifact, pairing and fixture-fit kernels. Old public
wrappers/defaults/field sets/bytes, native prediction/text parsing and fit-child/
publisher algorithms remain exact. New explicit *_v3 wrappers use a closed stdlib
Tree context (schema3|4, exact model/config classes, architecture parser/identity,
feature/category layout and input/label/provenance validators); no second workflow.

Tree3 inference/request/input/result/training records retain existing kinds with
schema_version4 and add exact source_kind/source_origin fields. Actually trained
fixture records require source_kind card-instance-tree-fixture-originals and the
exact immutable fixture origin, bound to Dataset3 manifest/content/source-set
hashes and the actual request. TrainedProvenanceV4 uses kind
structured-fixed-dataset3-fit and retains source kind/origin through request SHA
and paired save/copy/load. A toy component artifact has BOTH source fields null,
fit_origin toy, provenance null and no training binding. It cannot be resaved or
renamed into a Source-backed trained artifact. Trained fixture loading requires
the actual healthy exact pair before Booster allocation; purpose flags or numeric
hashes cannot authorize fitting or change origin. No qualified-game toy/factory
is introduced by this increment.

Artifact4 also binds rule_signature/layout_signature and the FULL accepted
feature_layout (raw-field mapping, escaped names, transforms, category limits/
indices, token/rule/layout/window signatures). Closed semantics4 keys are
input_recipe, policy_input_version, generator_version, simulator_revision,
catalog, reviewed_spec_hashes, spec_inventory_sha256, rule_signature,
layout_signature, token_signature, numeric_transform and label_recipe.
numeric_transform is TREE_NUMERICS, the actual Tree feature transform. Trained
fixture label_recipe is card-instance-declared-terminal-v1; toy label_recipe is
component-toy-labels-v1. Do not add natural-outcome/reward/end-projection claims
to declared fixture labels. Future real-game source semantics are a separate
closed following context.

The same _PreparedTreeFit/_prepare_request/_TreeFitRun/worker/paired publisher
accept only the actual active concrete VerifiedFixtureTreeDataset3 for this new
private fit entry. Preserve fresh initialization, no init_model/warmstart, and
500rounds/50early-stop/.05learning-rate/min20/thread defaults. Derive admission
from Tree3 layout widths instead of legacy1042/32 constants. Child sees owned
request/numeric shards only, never raw Dataset/Source originals. Hold SAME original
Source/index through preparation, fitting/reap, numeric/request/pair checks and
reportable final close BEFORE complete publication. Capture immutable manifest/
provenance before Source release clears metadata; after commit release is
memory-only. Preserve cancellation, late-child, rollback and reachable cleanup
owners; no failing postcommit I/O.

New completion costs use shared_fixture_construction {seconds,cost_key,scope},
deduplicated from the immutable original fixture-manifest SHA and its actual
construction cost; no PPO elapsed time. Old shared_generation stays unchanged.
Request4 also carries the exact shared_fixture_construction receipt derived from
the SAME active Dataset before close; completion costs must canonically equal
that request receipt. Request SHA transitively binds the stable construction cost
into paired provenance. New metadata cannot supply a different cost after fitting.
Per-fit preparation/child/validation/cleanup costs remain completion measurements,
not Dataset semantic identity. ComponentStructuredTreePolicyV3 may share scoring
through PreparedTreeFrameV3 with a component-only label lightgbm-tree-v3; no
registry/HTTP/game worker or qualified-fit stub is claimed complete here.
Durable3population originals→Dataset3→fresh fit→paired artifact→worker/RAW and
formal/game interfaces remain required following this bounded implementation.

### 18.4 Native component-envelope token diagnostic (design accepted)

Independent architecture DESIGN READY after the zero-decision/record bounds and
constructor/export failure-ownership handoffs are explicit. Root accepts this
bounded diagnostic design. Implementation/parity/conformance and later separate
fixture capability publication remain required.

Root accepts the proposed responsibility split for design review: native checks
only the exact envelope's structural/internal facts; the SAME TreeFixtureSource
owns frozen declaration/cohort membership, actual artifact-policy IDs, complete
inventories and fitting permission. Envelope bytes cannot establish those external
facts. Add one bounded CardInstanceFixtureTokens class, no second Sources/
CohortSession/PPO stack and no Python production changes in this native increment.

Constructor accepts owned envelope bytes, ordered codec2 payload tuple, the
existing closed five-key settings bytes, deadline/optional CancelHandle, explicit
aggregate_window(default128) and retained_python_bytes. Methods are export_tokens,
used_bytes, peak_bytes, control_reserved_bytes, cancel_handle and close. Individual
envelope/payload/record/array limits remain≤64MiB; cumulative original input,
decoded/staging/caller/scratch/exported ownership is charged before allocation.
All borrowed Python bytes retain owned lifetimes during GIL detachment.

Input is exactly existing component envelope1: schema_version, kind, fixture_id,
declaration_sha256, population_id, cohort_id, episode, viewer, split, generation,
roles, selected_policy_id, outcome, outcome_origin, events and decisions. Decisions
are exactly frame/columns/selected_index; the payload tuple matches descriptor
filenames/order/sizes/hashes and all original candidate columns.
The tuple is immutable bytes, exactly one payload per ordered decision; a
declared zero-decision episode uses an empty tuple and correctly empty decision/
token arrays. The whole envelope is one JSON record, so max_record_bytes also
applies in addition to source-file/cumulative limits.
Validate component
kind/origin/identifiers/SHA syntax/train-validation generation/internal roles/
viewer/declaration consistency; viewer-relative declared±1/0; corrected frame
revision and contiguous own intervals; every prefix and final bank closure;
real18/codebook2/selected bounds and complete corrected raw tokens. Reject J6,
HTTP/game Source envelopes, foreign/future prefixes and wrong purposes.
Introduce a distinct corrected-component codec/Stage context, leaving corrected
game and engineering branches exact. Never relabel component tables qualified.

Output reuses readonly metadata-bytes/uint8/int64/float32 banks and existing
array descriptors. Metadata1 kind is card-instance-component-token-payload;
exact keys are schema_version, kind, envelope_sha256, fixture_id,
declaration_sha256, population_id, cohort_id, episode, viewer, split, generation,
roles, selected_policy_id, outcome, outcome_origin, label_recipe, label,
input_binding, layout_signature, aggregate_window, token_signature, boundary,
decisions and arrays. Each ordered decision has decision, revision,
interval_start, interval_start_hash, boundary, frame_hash, table_hash,
columns_file, columns_sha256, selected_index, token_prefix and event_token_prefix.
Original table_hash+selected_index binds the selected row; invent no new selection
hash. Raw state/card/history/interval/candidate shapes/dtypes are the accepted
corrected layout, full original candidate order. No qualification/Source proof/
issuer/update/fit-permission field is returned.

Reuse Stage, shared token/event-prefix kernels, budgets/cancellation/retirement
and ndarray ExportLease. Constructor validates/stages one episode. export_tokens
is one-use: complete banks, retire spool and parsed-core ownership/join BEFORE
returning readonly arrays. Exported base-owner leases independently retain their
bytes/budgets after object close/drop. Any admitted failure is cleanup-only;
before-export close completes reportable cleanup, after successful export close
releases memory only. No second payload-retirement cycle is introduced.
Constructor-time validation/staging failure must retire/join every partial owned
Stage/core, or preserve a reachable cleanup owner/quarantine if confirmed close
is unavailable before returning the error; silent Drop is insufficient. An
export-side Python wrapper/quota failure after core retirement exposes no partial
arrays and remains cleanup-only. Reuse existing reachable retirement ownership.

Private candidate builds may expose this diagnostic for tests. Separate marker
card_instance_fixture_tokens=component-envelope1;participant-structured-v2;codec2
and corrected signature remain absent until full structural/token/export/
persistence parity and both reviews. Existing three corrected-game markers and
all old markers stay exact; do not replace the release ELF used by actual native
PPO. Require Python-reference full arrays/metadata/dtype/shape/bytes, all action/
state/viewer/label/Guardian/Fog/ref-order cases, malformed/mixed/prefix/resource/
cancel/lifetime tests and old codec/Stage goldens. No fitting, GPU, Simulator or
new dependency is part of this increment. Architecture review precedes code.

### 18.5 Component Tree scoring process handoff (design accepted)

Independent architecture DESIGN READY after single-operation ownership,
pre-spawn packet admission and complete reply checks are pinned below. Root
accepts this bounded private component-worker and actual-input performance
scorer design. It does not introduce a game-policy registry route or authorize
a component artifact for game inference. Implementation/parity and both final
reviews remain required.

Use a small scores-only facade over the existing framed exchange, owned process,
reservation, cancellation and reap machinery. Reuse those lifecycle methods with
a private transport owner; do not expose inherited participant selection methods
on the scoring facade or clone the worker lifecycle. A fixed clean bootstrap
accepts a closed protocol3 scoring discriminator and a closed initialization
payload. The facade exposes initialize, score-original, generate-and-score and
close, plus health/timing data. It never returns Selection, Action, logprob or
critic value and is absent from production game factories.
Allow only one in-flight initialize/score/generate operation. Reject an overlap
before touching the event bank, RNG, lease or pipe. Internal lazy initialization
belongs to that same admitted operation. Close may interrupt a pending request:
first forbid new operations, cancel the pending operation and use the same fixed
cleanup budget/owner to reap/join; no pending or late reply may escape as success.
Any admitted operational failure is sticky cleanup-only; an overlap rejected
before admission does not poison the existing operation. Successful state/RNG/
prefix updates happen only after the entire reply validates.

The controller retains the actual healthy paired component artifact4 and its
original file guards through startup, all scoring and confirmed worker close.
Require exact fixture source kind/origin and full architecture/feature/input/
rule bindings before reserving/spawning. Keep the actual pair/source path checks
on the controller. The child receives only owned bounded inference metadata,
native model text and CPU inference/resource/action settings, not raw Dataset,
parent controls, labels, private other-player records or source directories.
Both metadata and model text are at most64MiB individually and cumulatively
admitted with retained host/model/caller buffers before copying or spawning.
Also admit the entire actual encoded initialization JSON, including metadata,
escaped model text and controls, under framing.MAX_FRAME_BYTES=64MiB BEFORE
Popen. Conservatively reserve encoding/copy/decoded peak before constructing
that packet; if its actual encoded size exceeds the unchanged framing limit,
reject before spawn. Do not introduce chunked initialization or increase limits.
Child parsing/loading reuses pure artifact/text/native prediction kernels and
the corrected actual-Booster binding. Its private in-memory loading path grants
only diagnostic scoring, not a paired artifact or Source/fit permission.

Fix the input scope at construction: component-originals or actual-own-input
diagnostic. Both require the corrected contract/layout/rule recipe and real18
capacity. Component scope accepts the original component table; diagnostic scope
accepts the original qualified-game table without rewriting its purpose or the
artifact's component origin. Only the latter may request normal candidate
generation. Each actor process binds one participant/episode, exact increasing
revisions and contiguous event deltas; it checks the whole own prefix and all
frame/table hashes before scoring. Candidate columns/order, masks and full prior
own history remain intact. No packet contains hidden simulation state or another
participant's private input.
Keep artifact action-config identity and caller generation action-config identity
separate. They must match for fixture-original scoring. Actual-input diagnostic
scoring may use the original full-budget game configuration independently of
the fixture's64-candidate construction configuration; record and check each
exactly without relabeling either. Contract/layout/rule/feature recipe and real18
capacity must still agree, and inference must admit the full original/generated
table. This exception yields scores only and grants no game-policy authority.

Score-original takes the full original ordered table and returns finite scores
in that order, its exact table hash and measured token/prediction time.
Generate-and-score reuses PublicCandidateGeneratorV3, the existing sampling-
derived search stream and Tree numeric-cost model with all original18/32768/
262144/50ms/16384/1s/256 budgets. It returns the full generated table/report,
finite scores and generation/conversion/scoring timings; the assessment computes
argmax as an index only. It does not create a sendable action. Framing/decoded
input, interval/event/decision, per-array64MiB and cumulative resource bounds
precede allocation. Timeout10seconds and shared lease/reap behavior remain.
Ready and every reply must match the controller's immutable artifact/inference
digest, full model/window/feature identity, input scope, original action settings,
rule/layout and CPU execution identity. Score-original replies bind the exact
request frame/table hash, row count/order and finite scores. Generation replies
also bind original generation config and Tree numeric-cost hashes, the parent's
expected search RNG-before and all candidate/search bounds. Validate the whole
reply before advancing parent RNG, event-prefix or decision state. A malformed,
crossed or late admitted result is cleanup-only; framing alone is insufficient.

The SAME active TreeFixtureSource stays retained by the component acceptance
caller while its original episode/ordered-table facts are scored. The separate
native fixture-token diagnostic validates and exports that same envelope; a
bounded acceptance adapter compares all metadata/raw arrays with the Python
reference, persists and reloads the shared NumPy bundle1 representation, then
compares selected/full Tree features and native CPU scores. No cached token or
diagnostic output substitutes for the concrete fixture owner or changes its
fitting permission. Formal GameTree3/Journal6/RAW routing remains the following
separate production increment under§18.2.

Verify healthy fresh fixture pair→clean child→scores and native/persisted raw
reference equality across all nine cohorts, including zero-endpoint episodes;
mixed origins/contracts, changed windows/text/metadata, wrong actors/prefixes,
reordered tables, timeout/cancellation/startup failure and cleanup ownership.
Check ordinary diagnostic generation uses unchanged budgets, returns no action
or Selection and cannot enter a game factory. Preserve old worker packets,
factories, journals, artifacts and numeric outputs. No new dependency, Simulator
execution, real-data fit or performance claim is part of implementation tests.

Root accepts the concrete§18.5 packet handoff subject to architecture review of
two narrow sharing hooks: numeric_costs_v3 shares the existing Tree2 arithmetic
kernel using derived Tree3 widths and actual descriptor, while old Tree2 outputs
remain exact; StructuredParticipantWorker gains one private _read_reply hook
whose default is the existing read_frame call. Only the scoring transport
overrides that hook to admit declared wire bytes and conservative decoded/copy
ownership before reading/decoding the reply. Share framing validation rather than
copying exchange/reap or increasing MAX_FRAME_BYTES. The fixed bootstrap applies
the same predecode admission to initialization and requests. Scorer caps retain
Tree2 max_decisions=None, interval4096, episode262144 and real18 positions;
candidate capacity comes from the exact caller configuration. Do not introduce
a1024-decision cutoff into the stateless Tree scorer.
The fixed bootstrap receives exactly one sanitized decimal CPU managed-byte
limit as a control argument, derived from the controller's validated configured
resources. Parse its1..2^63-1bound before reading any frame, then clear argv to
the fixed scorer name and sanitize the environment. That limit admits the
initial header/body/encoding/decoded peak before JSON decoding. After decoding,
require exact equality with initialization's model.resources.max_owned_bytes
before loading a native model; a mismatch is cleanup-only. Include that same
initial peak in the controller's admission. No private state, labels or seed
is passed through this resource argument.
The scorer constructor also requires an explicit finite absolute monotonic
deadline for the original session/pair owner. Retain that cap through startup,
every operation and close; do not construct a retained pair scan with now+10
or extend its original deadline. A fresh per-operation/exchange10second budget
is clipped to the same original session cap. Checks may narrow an operation's
budget, but never raise the base cap; reject late results before committing
state. Close/reap retains its first cleanup budget and original session cap.
This argument stays on the controller and changes no packet or resource argv.

### 18.6 SAME fixture-owner native parity adapter (design accepted)

Independent architecture DESIGN READY after outer acceptance-commit ownership
is pinned. Root accepts the remaining native/persistence/feature adapter below;
implementation, full parity and both final reviews remain required.
The existing TreeFixtureSource owns all declaration, policy, cohort, role and
original-file authority. Add diagnostic_episode_count and a Source-owned
borrow_diagnostic_episode(ordinal, extra_owned_bytes=0) context. Only Source
resolves ordinal membership. Callers never walk its private scan/declaration/
policy tables. Do not change ordinary selected_inputs streaming, allocation,
single-pass completion or fit permission.

A borrow exposes guarded immutable envelope bytes, ordered tuple[bytes,...]
column payloads, closed Source-expected metadata, retained_bytes, guard and
admit_external. A proposed guarded readonly actions accessor returns the exact
immutable ActionConfigV3 constructed inside Source from its already-validated
declaration, allowing the scores facade to check artifact/caller configuration
without private-table access or inferred caps. It changes no completion/count/
permission state; bounded architecture confirmation precedes its implementation.
Its reference_decisions iterator streams the original frame,
bounded own bank view, full table, selected index/viewer label and existing
FixtureSelectedInput one decision at a time. Factor existing envelope/column
validation narrowly; do not decode/retain every table for Python-only fitting.
Read/hash/admit all files through SAME snapshots, with no open descriptor across
yield. Reject overlap with ordinary iteration or another borrow. Exit invalidates
borrow operations; admitted errors poison operational use while preserving
cleanup, original error and final Source guards. Diagnostic borrowing changes
neither _complete, population counts nor fitting permission; the original
selected-input worklist remains required for complete Source verification.
Source-expected metadata consists of§18.4's exact nonstorage metadata fields:
exclude schema_version, kind and arrays, and exclude token_prefix and
event_token_prefix from each decision. The adapter independently validates the
diagnostic schema/kind and storage-prefix/array descriptors against the shared
staging/token recipe. Source supplies original semantic facts and common token
signatures, not native bank offsets or a native permission object.

A new private native_card_instance_fixture module composes the diagnostic,
persistence and parity report. Default native loading requires the separate
fixture marker/signature, exact API/cancel ABI and binary identity BEFORE any
borrow or output allocation. Explicit isolated diagnostic-module injection may
inspect e390 with its marker absent; this grants no Source authority. Hold its
actual binary/file identity through verification. Process all288ordinals/nine
cohorts in original order, including declared zero-endpoint episodes.

Joint admission includes Source/borrow/caller ownership, original bytes, native
staging/export leases, reference tokens, encoding, reload buffers, model storage
and feature chunks BEFORE allocation. Neither Python nor native receives a
separate full allowance that can exceed the caller's total. Account cumulative
persisted bytes against scratch; write only fresh paths outside all originals.
Compare closed native metadata and every array/dtype/shape/byte against
Source-owned Python references before persistence. Persist native metadata plus
shared NumPy bundle1, release exported views/native owner before reload buffers,
then reconstruct immutable FrameTokens/CandidateTokens and run existing corrected
layout/real18 validators. Extract only low-level tuple/mask reconstruction from
the legacy native_tree_dataset helper; its selected-only/observed-outcome wrapper
and successful output stay exact. Never fabricate EpisodeRecords or observed
Selection/outcomes for the fixture.

Bundle1 rejects an empty array mapping. A zero-endpoint diagnostic stores its
real empty decision/array metadata with bundle=null, without sentinel arrays or
a bundle-schema change. Nonempty array mappings retain ordinary bundle1 rules,
including correctly zero-length slot arrays. The closed diagnostic container
keys are schema_version=1, kind=card-instance-fixture-token-diagnostic,
source_manifest_sha256, native_binary, metadata and bundle. This container is
structural evidence only; it cannot construct a Dataset or grant fitting rights.

Compare selected/full2810-column features using existing token/Tree kernels,
chunked in original candidate order; compare bound native CPU predictor scores
and§18.5 facade replies for the SAME paired model/frame/table/selected identity.
Source and borrow guards bracket export, persistence, reload, feature/scoring and
diagnostic result construction. Per-episode persisted containers are structural
evidence only, with no healthy final-acceptance status. The adapter returns
guarded diagnostic results under the live Source; it does not publish the final
acceptance report. The outer acceptance caller owns that final commit: consume/
verify the entire ordinary Source worklist, finish all original/binary/pair guards,
close all borrows, and complete reportable scorer/pair/Source close and reap
BEFORE publishing a healthy final report. Later release is memory-only. Reuse
existing no-overwrite/invalidation helpers if publication itself fails; no new
publisher framework. Native fixture
capability publication stays closed until this full pipeline and both reviews
pass. Require zero/multiple decisions, both viewers, labels, duplicates/order/
Guardian/Fog, changed originals/policies/metadata/shards, malformed arrays,
overlap/after-exit/close/error, cancellation and cumulative-budget failures.
No new dependency, Source lineage, PPO authority, real fitting, Simulator or
performance claim belongs to this increment.

Concrete facade composition remains synchronous: create one owned event loop
for this diagnostic session, reject an already-running caller loop BEFORE a
borrow/output allocation, and initialize/close/reap one component worker per
nonempty original episode. Zero-endpoint episodes spawn no worker. No facade
or framing loop is duplicated. Reserve the child's full managed runtime limit
jointly with Source, parent/native buffers, model and feature ownership. A
declared execution-only resource override may bound that child limit by the
remaining original total; record it explicitly, retain all architecture/weights/
inference/action identities and full candidate tables, and fail if the required
workspace cannot fit. Never increase the original total or trim inputs to fit.
Architecture confirmation of these bounded composition details precedes their
dependent code.

The first durable outer run reveals an ownership-scope error, not permission to
raise budgets. The diagnostic's joint limit is min(caller limit, Source limit),
8GiB for this acceptance. Do not reduce that whole limit to the2GiB model
runtime limit. Keep the parent predictor/feature runtime, facade controller and
facade child at their original2GiB limits; reserve the full controller+child4GiB
jointly before construction, using the unchanged model. Local feature admission
counts its predictor/tokens/prepared/chunk/score overlap under2GiB. Global
Source/borrow admission additionally includes original/control/caller/native/
reload/persistence/facade ownership under8GiB. Neither check replaces the other.
Retain conservative duplicate headroom unless a specific allocation's owner is
identified; no arbitrary subtraction from Source retention or bypass flag.
The outer caller's predictor envelope and adapter estimate must identify any
intentional overlap explicitly. All64MiB limits remain.
Before a fresh run, a metadata-only worksheet must list original model/inference
identities, caller reservation breakdown, Source intrinsic charge, native/export/
reload/local-feature overlap, full4GiB facade reservation and actual unchanged
initialization formula. Explain each retained duplicate or its replacement
owner. Architecture review and fitting/one-byte-over admission tests precede
another fresh acceptance attempt. The failed first run and its two containers
remain preserved with no healthy report, refit, reduced input or runtime-cap
change. Ordinary performance retains its separate Source/controller-plus-child
interpretation; this paragraph concerns the explicitly joint fixture diagnostic.
