# 0069 Combined Moon God and Dream Simulator contract

Status is Accepted. Independent design, correctness, and implementation
conformance reviews cleared. [Verification](../verification/moon-dream-simulator.md)

## Context

At HPO6 batch `000016`, p83 discarded `sell` at revision 70. Moon God
selected Mirage during the turn-end resolution. Simulator `9e47d4c`
returned `UNVERIFIED_RULE` and rolled back both state and RNG. Sixteen
updates were committed in parent `cd05a376-93f2-44b4-8516-913c96f49518`.
The pinned run remains failed and quarantined. [Recorded p83 actions](../issues/cases/0016-moon-god-standalone-mirage.json)

## Decision

The local combined Simulator revision `f394a7b57b0ffb12c6b5782ff52fdb8842ec7534`
has parents `9e47d4c5f8cd3f60437f9aeabc95076c1dc04321` and
`0eca2d6fe70fcf19bd0753fe400a7eff4422b955`. It retains Strength Powder and
Dream compatibility and applies the updated Simulator §19.4. Moon God
combines Mirage or Aura with Full Moon Blade using a single miracle draw.
The Simulator owns weapon defense, attack amount, event projection, and RNG.

The shared Action profile `card-instance-simulator-v2-shared-20261125` binds
the clean source, normative inventory, changes since their common release base,
and the release TestServer. It uses Gameplay API v2, public event schema 2,
catalog schema 2, the existing v5 search rules, Journal 6, and candidate-columns-v2.
The existing `GuardianActed` namespace admits `full-moon-blade`. Policy history
uses the acting participant's `/events` feed and same-revision Observation and
ActionOptions under Simulator §21.1. Concealed draw omission follows ADR0068.

Python and Rust validate the new pins independently. The fixed operation and
campaign entry points require a wheel with the combined profile and concealed
draw support before live ownership. The default 1,000-update preset selects
the combined profile and its own original, numeric, and diagnostic destinations.

## Alternatives and consequences

Adopting `0eca2d6` directly reproduced p41 on its earlier Dream projection.
The separate local Simulator merge combines both fixes and preserves both
parent commits. Earlier action profiles, artifacts and audit identities retain
their release pins. The changed Moon God effects can change future game
traces and outcomes. New convergence runs start at generation 0 with new IDs.

Independent architecture review cleared this design before implementation. The
[verification record](../verification/moon-dream-simulator.md) records the
release, fixed replays and HPO6 acceptance.
