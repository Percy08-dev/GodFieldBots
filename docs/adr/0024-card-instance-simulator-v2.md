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

## 15. Corrected neural game inference (design accepted)

Independent architecture DESIGN READY after pinning the closed initialization
union, whole concurrent memory admission and retained external-artifact owner.
Root accepts this inference-only scope under the repaired §14.1 input.
Implementation/reviews and actual smokes remain required; this is not Source,
generation or learning acceptance. Do not start dependent implementation until
the §14.1 shared input/handshake interfaces are stable and reviewed.

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

## 16. Qualified Python Source, numeric4 and shared PPO (design accepted)

Independent architecture DESIGN READY after pinning online ownership, exact
reader/PPO-collection variants, Source-issued wire expectation, RNG-neutral
recorded-output parity, early combined admission and compatibility/pruning
verification. Root accepts this bounded Python scope. Dependent implementation,
independent reviews and the real acceptance matrix remain required; native and
Tree remain separate gates. One implementer may capture bounded assessment-only
pre-edit compatibility evidence while current production source is frozen.

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

Run/generation/update/registry3 bind repaired input, full execution/Config/code/
Simulator/preprocessing/reward identity, original source proof and numeric4
hashes. Shared file ownership/lineage validation/pair save/retention are reused
with explicit closed version context. Publish completed generation only after
healthy inference4+resume4 acknowledgements, source final recheck and linked
numeric/update metadata validation; a standalone inference file is insufficient.
Maintain the existing run lock, single coordinator,16games/update, snapshot pool
(at most5, every10updates), worker≤4/game≤2, storage and7200/5400/7140s caps.

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
the generation10snapshot boundary and later pool eviction/pruning. Verify retained
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
