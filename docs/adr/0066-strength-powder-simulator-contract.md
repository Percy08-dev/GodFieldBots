# 0066 Strength Powder Simulator contract

The [p11 investigation](../issues/0013-hpo-dream-discard-hidden-weapon.md)
traced a Discard rejection to a weapon displayed as Strength Powder under
Dream. The owner confirmed that Strength Powder is a damage-adding sundry
and requires a weapon in the same ordered attack.

## Decision

Simulator revision `3172da407e37a4ad57ac82e70a96910c3ac6fbb8` separates
damage-adding sundries from the Dream groups for additional weapons. The engine
requires an effective actual weapon when Strength Powder is selected. It checks
this before target resolution, random draws, payment, or consumption. §9.1
and §18.4 of the revised Simulator gameplay spec record the rules.

The shared Action profile `card-instance-simulator-v2-shared-20261123` binds
this revision and its reviewed spec inventory. It uses Gameplay API v2,
catalog schema 2 and `public-ja-v3`, the v5 search recipe, Journal 6 and
`candidate-columns-v2`. The existing `rule_signature` already hashes the
complete Action contract and identifies candidate behavior for recorded data.
Launch and stored runtime validation use the profile's original release
identity. The earlier shared profile and its candidate rules remain available
for historical p11, datasets and checkpoints.

Python and Rust use the same public completion condition during search and
candidate analysis. An offense with a visible Strength Powder needs an
effective card whose visible catalog kind is `weapons`. Role binding excludes
weapons selected only for CostCut. A prefix can still grow to include a later
additional weapon. Both implementations reuse the offense analysis and its
existing search budgets. The input remains the actor's perceived definitions
from the pinned catalog, Observation and same-revision atomic hints under
Simulator spec 07 §21.1.

## Alternatives and consequences

A Bots-only filter would leave incorrect Dream displays in the authoritative
runtime and could still propose Discard for a disguised weapon. Reinterpreting
the previous shared contract would change the meaning of frozen candidate
tables and checkpoints. Adoption uses an additional closed release pin while
the Simulator continues to own legality.

The revised candidate pool can shift subsequent random draws. Acceptance uses
controlled positions for the revised rule, and the original p11 fixture uses
its recorded source and binary. The fresh HPO template selects the revised
profile and uses a dedicated output namespace. The older probe config remains
available for reproducing its original runs.

Independent design review cleared the public information boundary and historical
compatibility design before implementation. Release identities, checks and run
evidence are recorded in [Strength Powder verification](../verification/strength-powder-simulator-contract.md).
