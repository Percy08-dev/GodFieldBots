# 0005: Use authorized acting-player ActionOptions as atomic hints

- Status: Accepted
- Date: 2026-10-27
- Decision owner: root
- Review: `bot_architecture_reviewer`, design READY, 2026-10-27. Normative hashes
  and raw wire DTO verified; no blocking findings. Final conformance is required.
- Implementation: complete, including the reviewed v3 singleton amendment.
  Independent correctness/conformance READY; [verification](../verification/action-options.md).
- Contract amendment: [ADR0007](0007-staged-buy-contract-adoption.md) supersedes
  this record's original Simulator pin/spec snapshots for corrected Buy control.
  The authorization, input/storage formats and candidate rules remain in force.
- Follow-up 2026-10-28: the formal PPO attempt exposed an unverified zero-MP
  dynamic-weapon target case beyond the v3 amendment's MP-10 evidence. See
  [the separate contract clarification](../issues/0004-zero-mp-dynamic-weapon-target-contract.md).
  V3, its historical pilot and the wholly failed PPO parent remain unchanged;
  no new legality rule is inferred from the rejection.
- Follow-up 2026-11-25: [ADR0072](0072-sacrifice-eligibility.md) records the
  owner's confirmation that Sacrifice and Discard share item eligibility.

## Context and authority

The user explicitly authorizes ActionOptions and has updated the Simulator
normative learning-input contract. This instruction supersedes earlier Bots
prohibitions. Current spec 07 §§13/21.1 permits optional actor ActionOptions as
features, atomic masks, candidates and teacher signals; SequenceOptions and
rejection-derived features remain excluded. Dream uses actual availability and
perceived display under its public use-context compatibility rules. This is an
authorized input boundary, not a new hidden-state inference from diagnostics.

Simulator Git HEAD remains `44a26e3a16b808e826deb332fa72acfc5037d3ef` with user
worktree documentation changes. Do not modify/build the Simulator. Reviewed
document snapshots (record these in run provenance; HEAD alone is insufficient):

| File | SHA-256 |
| --- | --- |
| `docs/spec/00-specification-overview.md` | `a2ef3bbf2d8a1e88300d94f75bccfd1956a9b0ea18c14dce2dc0c1c41845a03d` |
| `docs/spec/07-external-interfaces-adapters.md` | `057577a3a3678cdc76a427bf578bd50295433fc0f096e50be3352a1f6e38062b` |
| `docs/spec/08-test-requirements.md` | `179d18cf95f1d8e8f4c925dc1998ae367a9e86fd32effbf9fe69272d359d48b7` |

HTTP `/v1`, rule set, Observation `"1"`, typed metadata `1`, info `1`, catalog
version/hash and binary remain unchanged. Binary build revision stays unknown.

## Requirements and scope

Use ActionOptions by default for Random/heuristic candidate construction. Keep
enumerate/beam, public construction rules, finite budgets, candidate order and
reserved basics. Do not interpret atomic hints as complete sequence legality.
Retain a deliberate config opt-out for controlled observation-only experiments;
there is no automatic fallback when hints are requested but fail validation.
No new dependencies, training/model stack, simulator code or legal-action oracle.

The old public candidate subset need not suddenly cover every card/effect.
Apply authoritative atomic masks to that supported space and report its
remaining omissions. Do not admit support-only Aura or other incomplete prefixes
as completed actions merely because their elements appear in hints.

## Interface, flow and ownership

1. Add `ActionConfig.use_action_options: bool = true`; both supplied search
   presets enable it. The effective config/hash includes the mode.
2. After obtaining the actionable actor Observation, fetch
   `GET /v1/games/{game}/action-options/{actor}` through the attested client.
   Raw response is exactly `state_revision` plus `action_options`; the nested DTO
   has required_choice, usable hand/learned arrays, exchange/sell/buy carriers,
   valid_targets, can_pray/can_discard/can_sacrifice/can_allow booleans.
3. Validate closed fields, exact types/domains (including bool-as-int rejection),
   duplicate-free IDs, required choice and the same Observation revision.
   All hand/carrier IDs must be in that participant's own hand; learned IDs must
   be in its learned collection; targets are visible Player IDs. Require that
   the requested viewer is current actor and Observation is actionable.
4. Normalize into an actor-scoped envelope containing `viewer_player_id`,
   `state_revision`, and the unchanged nested DTO. The added viewer tag is bound
   by the transport owner to the requested path, not trusted from the server.
   No Game ID, URL, seed or management configuration is added to policy input.
5. Extend `DecisionInput` with nullable `action_options`. With hints enabled it
   must contain that validated envelope; with the explicit opt-out it is null.
   Revalidate in the fresh policy worker against its own Observation before
   generation/selection. Keep typed metadata unchanged and catalog restricted to
   own visible/perceived definitions. The exact delivered options/order are
   journaled, not reconstructed from later state/Replay.
6. Only this dedicated same-revision fetch supplies hints. Continue ignoring
   unbound action_options/events/next_observation in Act responses and Replay
   for policy input. Events still come exclusively from the event API.

Mismatch, missing/unknown fields, foreign references, unavailable endpoint,
timeout or cancellation stops the decision before selection/submission. Reuse
existing attestation/recovery/deadline/cleanup ownership; no policy re-proposal,
cross-participant query or fabricated pass. Do not use SequenceOptions.

## Candidate construction by purpose

- Turn play elements: intersect supported offense references with
  `usable_hand_elements` / `usable_learned_miracles`.
- Defense elements: intersect supported defense references with those same
  actor/choice-scoped usable arrays. Do not infer compatibility from printed
  defense alone or obtain opponent definitions outside the catalog contract.
- Pray: use can_pray when hints are enabled, rather than guessing usability from
  presence of a weapons-kind card. Defense's empty response uses can_allow.
- Exchange/Sell/Buy carrier references use their dedicated arrays. Intersect
  configured explicit targets with valid_targets where an action has a target.
- Discard/Sacrifice: use the corresponding boolean for the action family, plus
  public discard rules on own visible cards. The usable-play list is **not** a
  per-card discard mask: ordinary armor may be discardable without being usable
  in a turn play. Likewise a sold item need not be in a usable-play/carrier list.
- The user's primary developer announcement confirms Sun Amulet and Dangerous
  Mortar cannot be selected for normal Discard. Apply this public restriction
  to Discard; do not claim it was inferred from a rejection/atomic list. The post
  does not explicitly specify Sacrifice, so do not invent an extension of that
  rule. Keep any unconfirmed limitation explicit.
- Buy accept/reject still offers both typed boolean decisions. can_allow is not
  a buy-decision permission flag.

Prefix expansion, completion, public pruning, approximate beam selection and
exact typed-action deduplication stay separate. Count every generated attempt,
including pruned ones. Atomic reference preselection may avoid generating a
branch; do not fabricate expansion counts for references never expanded.

## Compatibility and provenance

- New `POLICY_INPUT_VERSION = player-atomic-options-v1` and feature version
  `participant-json-v2`. Record the version in run/header/derived provenance.
- Journal/derived schema 4, run manifest schema 3; old schemas are retained and
  explicitly rejected, never filled with invented historical hints.
- Generator initially `public-atomic-subset-v2`, superseded by the accepted
  singleton amendment's v3 below; scorer, reward, end-projection, catalog
  and Simulator wire versions retain their unchanged meanings.
- Run manifest records the reviewed document hashes above and, when an optional
  local source_dir exists, actual local document hashes/dirty source provenance.
  Without a checkout, retain the reviewed snapshot declaration and mark actual
  local source hashes unknown. No source-layout dependency for installed wheels.
  This is provenance, not policy-visible configuration.
- Header records the hint mode as well as config hash. Reader/export verify
  input presence/mode, actor/revision/reference validity, candidate provenance
  and linked run semantics. Atomic hints are permitted row inputs; URLs/seeds/
  server info/Replay/rejection details remain diagnostic sidecar-only.
- All model comparisons use the same fixed hints mode and search configuration.
  Observation-only versus hints is a separate controlled experiment axis.

## Alternatives

- Continue refusing hints: contradicts the user's explicit updated requirement.
- Reuse options from an arbitrary response or another revision/viewer: can bind
  the wrong decision/participant and invalidates behavioral data.
- Treat usable-play IDs as discard/sold-item eligibility: excludes legal armor
  and conflates independent action purposes.
- Delegate whole-sequence probing to SequenceOptions or repeated rejected acts:
  outside the permitted atomic-hint contract.
- Silently upgrade historical journals: cannot reconstruct original masks.

## Verification and acceptance

Contract tests cover exact wire/envelope validation, same revision/current actor,
foreign IDs, separate learned/hand semantics, malformed flags/duplicates and
failure before policy/Act. Worker/journal/export tests cover snapshot roundtrip,
mode mismatch, stale/foreign hints, old-schema rejection and allowed/forbidden
information. Dream tests retain perceived definitions with the authorized mask.

Search goldens cover masked incompatible defenses, prayer flags, trade carriers
versus sold objects, discard flags versus play masks, protected normal discard,
empty/basic behavior, support-only prefixes and enumerate/wide-beam agreement.
Preserve finite/lazy budget and order/instance tests.

Run ordinary checks and opt-in real TestServer cases. Repeat the same 12-game
schedule as a new schema-4 experiment without replacing old failed evidence.
If clean and within budget, run the planned fixed-policy feasibility pilot
(64 games × 3 sampling seeds) with explicit accounting; any non-natural or
unhealthy result leaves learning gate unmet. Do not turn diagnostic success or
an optional mask into a claim of complete legality or model quality.

Independent correctness and architecture-conformance reviews follow stable code.
Root owns instructions/project presets/docs/acceptance; one implementer owns
src/tests; no agent edits the Simulator repository.

## Pilot-driven bounded candidate amendment (accepted)

The first fixed pilot (`baseline-feasibility-pilot-354f64ce-747`) completed all
192 scheduled games: 182 natural, 9 unsupported buy-decision metadata contracts,
and one empty supported candidate set; no rejected actions. Keep that whole
failed run ineligible, not just its ten failed episodes. No successful episodes
are substituted and no old snapshots are changed.

The empty case has a single own-visible Magical Stick (ID 10), MP 10, and valid
same-revision hints listing that exact hand element with targets 1/2. Its pinned
public definition says `ability=atkBy2xMP` and describes current-MP-dependent
attack, so no fixed `attack` field exists. The old positive-printed-attack subset
omits it even though its authorized atomic availability is explicit.

Accepted smallest extension: with hints enabled, construct singleton
`play_sequence` candidates for an own visible weapons-kind element with the
public `atkBy2xMP` ability when listed as usable. Use configured targets
intersected with valid_targets. Do not add this element to multi-card prefix
search or invent resource/order semantics; do not synthesize Pray. The existing
fixed heuristic may retain its approximate default score for an absent printed
attack; no scorer recipe/version change is required for this availability fix.
Observation-only mode retains its declared unsupported subset.

Publish generator `public-atomic-subset-v3` for the extended candidate space.
Input/journal/derived formats and policy-input/reward/scorer/projection contracts
are unchanged; existing version checks reject unsupported generator-v2 sources
explicitly rather than regenerating their candidates. Historical artifacts remain.

Verify exact independent singleton sets, target/mask/mode gates, no empty fallback,
no invented combined sequences, journal/version consistency and a fresh bounded
live evaluation. A new evaluation after the change is a separate run, not a
replacement for the failed pilot. The resolving/buy metadata issue remains a
separate Simulator task; never fabricate actionable metadata or submit a choice
against its explicit false scheduling flag.

Architecture amendment review: `bot_architecture_reviewer`, design READY,
2026-10-27, before dependent code changes. Require actionable turn_action /
play_sequence syntax, own hinted hand reference and allowed configured target;
count each emitted target-specific singleton within existing search budgets.
Final correctness/conformance checks remain required.
