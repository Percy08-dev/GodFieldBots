# 0072 Shared Discard and Sacrifice eligibility

Status is Accepted. Architecture review cleared before implementation.
Independent implementation-conformance and correctness reviews cleared.
[Verification](../verification/sacrifice-eligibility.md)

## Context

Run `83355bf4-1cf3-43c3-94be-9dae91de7290` committed 43 updates. In batch
`000043`, pair `p81`, Player 2 selected Energy Armor and Dangerous Mortar
for Sacrifice at turn 150 and revision 215. Simulator `6d50603` returned
`422 INVALID_ACTION / invalid_parameter` and the HPO parent was quarantined.

The owner confirmed that a card has the same selection eligibility for
Discard and Sacrifice. Artifact weapons, Sun Amulet, and Dangerous Mortar
are protected in both phases. The permission for all owned miracles,
including attack and learned miracles, follows Simulator §11.3.
This confirmation resolves the Sacrifice ambiguity recorded in [ADR0005](0005-authorized-atomic-action-options.md).
The pinned Simulator already uses the same item validation for both actions.

## Decision

The shared Action profile `card-instance-simulator-v2-shared-20261125-sacrifice`
inherits the exact Simulator source, specs, and TestServer binding from
[ADR0070](0070-pestle-mirage-simulator.md). Python and Rust use a shared
item pool for normal Discard and End Times Sacrifice in this profile. The
phase and the matching acting-player ActionOptions boolean continue to gate
which action is proposed. Candidate counting and emission use that same pool.

The policy uses only the acting player's visible definitions and references.
The two protected passive artifacts can appear as one another under Dream,
and both visible definitions are excluded. An armor or miracle may be
eligible even when absent from the atomic list of playable cards. The
Simulator owns final legality, card removal, and replacement draws.

The Simulator revision, API v2, public event schema 2, catalog schema 2,
Journal 6, `candidate-columns-v2`, and HPO6 rollout 8 retain their versions.
The new Action profile produces a separate rule signature and a matching
Native wheel capability. Mirage's opponent submission and the Strength
Powder weapon requirement carry forward.

## Alternatives and consequences

Applying the correction under the previous profile would regenerate a
different candidate table for historical records. A distinct profile keeps
the old binding and search behavior available and gives new convergence runs
a fresh storage stage at generation zero. The failed parent stays quarantined
under its recorded profile.
