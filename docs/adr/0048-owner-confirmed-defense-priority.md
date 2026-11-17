# 0048 Owner confirmed defense priority and current rules

- Status: Accepted, implemented and verified; both final independent reviews READY.
- Date: 2026-11-17. Owner: root.
- Authority: [owner defense clarification](../contracts/owner-20261117-defense-priority.md).
- Design: [bounded adoption](../design/defense-priority-owner-adoption.md).
- Baseline: `f4cdb4a`; previous experiment source `812a88a`.

## Context

The first manual D10,000 MLP probe fails at collection13 in both canonical and
width192 parents. Their common predicate admits Angel Armor plus Rainbow Curtain
against a publicly observed Mudflow miracle. The owner confirms that miracle
stop takes precedence over numerical defense and excludes another effective
defense. Rainbow Curtain must be selected first and only joins numerical defense.
The earlier uncertainty in issue0010 is resolved by this supplemental authority,
without changing Simulator source or pretending its normative hashes changed.

## Decision

Adopt current rules-v2 through the shared pure analyzer and an actor-visible
response context. The owner explicitly removes historical rules-v1 compatibility
from scope because the affected inputs are incorrect. Reject old identities
rather than reinterpret them. Keep generator/layout
identities unchanged; the existing supplemental rule_signature distinguishes
the new behavior, as permitted by ADR0024.

Require the exact current binding at each Python/native boundary and every
nested record; no dual-version selector or deadline/cancellation-control
extension is required. Current readers and execution require v2. Add a native
capability for the new rule-signature validation. Preserve old
failed parents and their eligibility/deadline state.

The context extractor accepts only a fully matched supported public cause from
the acting participant's verified event prefix. It does not reconstruct a game
engine. Missing cause IDs, reflection/Guardian/targeted effects or ambiguous
suffixes remain unknown. No private view, Replay, snapshot, rejected action,
GameSeed or unpermitted catalog expansion enters policy analysis.

## Alternatives and consequences

A global ban on positive-defense special armor would remove numerical use when
the special effect does not take priority. An HPO-only workaround would fork
common candidate meaning. Replacing old identities or allowing mixed signatures
would undermine dataset compatibility. A new numerical layout is unnecessary.

Unknown context can reduce completed combinations for conditional-effect armor;
singleton atomic candidates remain available. This is explicitly conservative
candidate completeness, not a new legality assertion. Revisit when ordinary
public histories frequently remain unknown or public cause metadata improves.
Historical-binding support was initially considered, then removed under the
owner's explicit scope correction. A source-matched native activation remains
required. The separate error-masking bug remains issue0010.

## Verification

Require failing/passing defense/order/context tests, current-v2 Python/native
validation, old/unknown/mixed-signature rejection, old-native preflight rejection, applicable
static/package/project checks, and independent correctness/conformance reviews.
No training, Simulator gameplay or HPO experiment is authorized by this adoption
verification. [Final evidence](../verification/defense-priority-owner-adoption.md)
records the source-matched native activation,232-file package/baseline checks,
Rust174PASS/13IGNORE, full Python7007PASS/586SKIP/2test-harness failures and
their isolated40PASS correction, plus root68PASS targeted confirmation and
Ruff/mypy232. Root accepts that explicitly documented whole-plus-delta evidence;
it is not a new fully green whole invocation or a learning-quality gate.
Captured current formal/final64 acceptance remains unavailable, and unrelated
baseline cargo-format debt remains disclosed. Original failed data is unchanged.
