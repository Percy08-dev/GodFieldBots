# Owner confirmed defense priority adoption

## Requirements and scope

Implement the owner's 2026-11-17 defense clarification in
`docs/contracts/owner-20261117-defense-priority.md`, SHA256
`bdd1792cde50eae2338b5f068436e40ca87cfbdf0c276ad096651bc0ff27375c`.
The concrete failure is documented in issue0010. Angel Armor must not be combined
with Rainbow Curtain against the visible Mudflow miracle. Numerically defending
armor remains combinable when its special effect does not take priority.
Rainbow Curtain must be the first physical card and cannot join a prioritized
stop/reflect/deflect defense. Preserve existing Flash, MP and bound CostCut rules.

This is a gated change: the pure cross-module sequence-analysis interface and
persisted rule identity change. Simulator source, binary, revision, catalog,
normative hashes, PPO math, budgets and training/evaluation schedules stay
unchanged. No automatic games, learning, HPO rerun or failed-parent continuation.
The separate collection diagnostic-masking problem remains tracked in issue0010.

## Approach and alternatives

Use one current rules-v2 descriptor and shared pure public response context.
The owner explicitly removed the need to support historical rules-v1 readers
or executable behavior. Reject their identities instead of reinterpreting them.
Keep the existing
generator and structured layout: supplemental rule_signature already identifies
candidate semantics independently of numerical layout under ADR0024.

A blanket ban on every positive-defense stop/reflect/deflect card would remove
the owner's numerical-defense combinations. An HPO-only patch would leave the
same shared predicate defect in other policies. Accepting old signatures under
new predicates or accepting either signature independently at nested nodes would
reinterpret or mix historical evidence. A new layout would unnecessarily change
numerical identity. Historical support was initially considered, then removed
under the owner's explicit compatibility scope; a dual-version abstraction is
not part of the current design.

## Responsibilities and dependency direction

- `gf_common.card_instance_rules` owns the closed current rules-v2 descriptor,
  owner hash, signature and pure sequence analysis. It has no I/O,
  Simulator, Torch or training dependency.
- A small common public-context helper, kept in this module or one focused
  adjacent module, receives the current DecisionFrame and its verified
  EventBankView prefix. It owns public causal classification only, not engine
  state reconstruction or a game simulator.
- `gf_common.diverse_actions` supplies the selected rules/context to generation.
  Common baseline and refresh scoring supply the same selected facts when
  calling shared analysis. They do not independently invent effect-priority
  rules. Refresh currently reanalyzes offense; keep that distinction explicit.
- Python journal/Source/artifact readers require the exact current binding and
  reject unsupported historical identities.
- Native readers validate the same exact current signature and invariant layout;
  they do not implement effect priority, public causal inference or policy math.
- CLI/current runtime composes active rules-v2; root owns authority/design,
  project presets, historical acceptance policy and final verification.

## Main interfaces and data flow

1. Use the closed current rules-v2 descriptor with the amended authority hash.
   Reject rules-v1, missing and unknown descriptors/signatures. Do not retain
   a legacy predicate implementation or introduce historical reader selection.
2. Existing signature checks require one exact current rule identity throughout
   each Python/native read. No retry with another rule, rehash or migration.
3. Pass the current `rules` and an immutable `DefenseContext` to shared sequence
   analysis. Context minimally distinguishes proven miracle, proven weapon and
   unknown. Add fields only when necessary to establish the public cause.
4. Extract context only for defense from `frame.bank_view(bank_view)` through
   the frame boundary. Use the complete needed public prefix, not just the last
   decision interval. Relate public accepted-action/card-use/miracle-use and
   attack-declaration events to the current defense. A later utility, Guardian,
   targeted effect, reflection chain, mixed attack, stale target or obscured
   causal sequence must not be mistaken for an unrelated earlier attack.
5. Public `artifact_used` and `miracle_used` identities may be correlated within
   the same visible cause; do not expand static catalog access beyond own
   permitted perceived definitions. Mere opponent learned-miracle ownership,
   the presence of some earlier miracle, or absence in an incomplete causal
   block never proves the current incoming source.
   For a known weapon, require a fully matched direct cause whose used
   definition is already in this actor's permitted own-visible frame catalog
   and classified as an ordinary weapon. An absent catalog ID, additional-only
   role, mixed cause or merely absent miracle event remains unknown. Do not
   replace catalog expansion with a hardcoded list of opponent weapon IDs.
   Match accepted-action player, use player, attacker and current actor target
   within the same causal block. The architecture reviewer retains DESIGN READY
   for this stricter proof and its explicit coverage limitation.
6. Rules-v2 selects the effective defense role first. Matching miracle stop,
   reflection and deflection on positive-defense armor are exclusive against
   proven miracles; their numerical role can combine against proven weapons.
   Existing purely exclusive defenses retain their matching atomic-availability
   boundary. No currently published positive-defense BlockWeapon definition is
   present in the pinned catalog; do not invent unseen attribute semantics.
7. Apply exclusion to effective defenses after existing CostCut suppression.
   Apply Rainbow-first to physical selected order. Reject any non-first selected
   filter card, including reverse-order combinations. Ordinary numerical/ring
   combinations remain subject to the existing rules.

## Unknown context and bounded information

Unknown is explicit and never converted to weapon/non-miracle. A singleton
admitted by acting-player atomic hints remains available. A combination whose
validity depends on an unresolved conditional special effect is not certified
as a completed public proposal; retain the distinction between unknown and an
established illegal combination in the analysis interface. Do not probe the
engine or retry a different action to resolve this state.

This conservative candidate-completeness debt can reduce available combinations
for ambiguous public histories, including used weapon IDs absent from the
actor's own permitted catalog. Record its scope and revisit if ordinary real
histories repeatedly produce unknown, or when the public interface provides
more precise incoming context. It must not be described as an engine legality
guarantee. The extractor must demonstrate the saved issue0010 history as a
known miracle and ordinary visible weapon fixtures as known weapon before
adoption. Unsupported causal patterns remain unknown rather than guessed.

Full prefix access remains actor/episode/boundary checked and bounded by existing
event caps. Derive once per frame/search or scoring call; avoid per-candidate
history scans. No other participant view, Replay, snapshot, RNG/GameSeed,
rejection detail, future event or unpermitted catalog definition enters analysis.

## Compatibility and native activation

Rule signature, not generator/layout substitution, distinguishes v1 and v2.
Only current v2 is supported by affected readers and execution. Old, missing,
unknown or mixed nested rule identities reject explicitly. No dual-version
selector, validation-context extension or historical analyzer is required.
Do not change stored records, rehash old evidence or rewrite frozen metadata.
Existing fixed numerical kernels and historical identities remain unchanged;
old rule-bound artifacts are unsupported rather than silently upgraded.
Preserve the two failed pilot parents and all their eligibility/deadline facts.

Native control readers require current v2 pins. Keep existing invariant layout
and exact nested-signature checks. Add an explicit native capability for rules-v2
signature validation so an older installed extension fails preflight. Native
Tree feature signatures include the rule binding and therefore change with v2;
their columns, numerical transforms and kernels remain unchanged. Current
code-owned pin/recipe assets update coherently, while frozen formal recipes and
captured evidence are not rewritten and their old bindings are unsupported.
Native
source matching and activation are required before a user reruns an affected
native workflow; no Python fallback. Build/install does not imply game or
learning verification. If native activation is unavailable, report that limit
and the concrete manual build requirement instead of weakening admission.

## Failure handling and verification

Input/context validation is pure and occurs before completed candidates are
published. New logic does not change positive action acknowledgement, natural
outcome rules, poisoned-pool cleanup or failure eligibility. Missing/mixed rule
identities fail explicitly rather than silently adopting a current default. Separate
diagnostic masking is outside this bounded predicate adoption.

Required evidence before acceptance:

- Preserve original failed data hashes and add focused rejection checks for old,
  missing, unknown and mixed rule identities. Do not add historical execution
  support or preservation tests for invalid candidate behavior.
- RED/GREEN pure and full-generator regressions for the recorded miracle case,
  all three conditional special abilities with positive defense, numerical
  weapon use, both Rainbow orderings, singleton, Flash and matching CostCut.
- Context tests for same actor/prefix, no future/foreign view, normal weapon,
  mixed weapon/miracle, reflected/Guardian/targeted/ambiguous sequences and
  opponent learned-miracle-only input. Unknown cases must remain unknown.
- Baseline/refresh and generation agree on current rules/context; corrected
  layouts, fixed numerical kernels and original data bytes stay exact.
- Current v2 fixtures pass Python/source-matched native validators. Reject old,
  unknown and mixed nested signatures, old artifacts and old native capability.
- Relevant pytest, Ruff/mypy, Rust checks for changed readers, package/import
  smokes and separate correctness/conformance reviews. Bound all verification;
  do not start training or Simulator gameplay. Record exact PASS/FAIL/NOT RUN.

Any assumption change in public causal classification, compatibility policy
or native ownership returns to root and architecture review before
dependent implementation continues.
