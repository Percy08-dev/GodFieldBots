# 0079 Simulator v3 direct defense and Fog event adoption

The architecture review approved this design before implementation and checked
the implementation. Independent correctness review cleared the reviewed changes.
[P46 reproduction](../issues/0019-hpo-defense-costcut-role.md) ·
[Verification](../verification/simulator-v3-direct-defense.md)

## Context

The v2 ActionOptions flat list advertised Spiritual Hood as a possible CostCut
support for Turbulence in the P46 defense decision. Both candidate generators
also proposed Hood as the only defense against Wood Big Tree. The Simulator
rejected the single-card response and the HPO parent was quarantined.

The Simulator release `997a0de59a456dd84699605c178f12ec74addf84` extends the
combined release `6d50603d` with the actor's direct-response hints and v3 Fog
attack events. The approved release pins include Gameplay API v3, rule set
`true-match-1v1-v3`, public event schema 3, and catalog `public-ja-v4`.
Observation, typed Action, and ActionOptions use schema 2.

## Decision

Use Action profile `card-instance-simulator-v3-shared-20261127-defense` for
new convergence runs. Admit the pinned API, catalog metadata and definitions,
source inventory, normative inventory, TestServer and Native capability.
Python and Rust select the protocol from the verified contract. A v3 event
bank, frame, and persisted reader bind the complete v3 raw identity. Existing
v2 records use the original hash preimages and their own release. Acquired
frames, transport, actor reconstruction and Journal use the enclosing actor
and revision authority. Raw records retain ordered events, and Replay remains
an independent verification input.

Use the same numerical layout and `participant-structured-v2` recipe. In
Fogged target events the actor channels use the existing missing-attacker
encoding. The target, attack value, and event order contribute to the existing
features. The new ordered source definitions and resolved attribute are retained
in the raw event. The raw event never supplies an inferred attacker. The
decoder distinguishes the two closed v3 event forms, validates the recipient
for anonymous forms, and keeps event-time visibility across later decisions.

Build the group membership once per frame from the current actor's exact
`defense:direct` references. Candidate counting and emission share sequence
analysis. First bind a CostCut card immediately after its miracle. An effective
CostCut armor acting as numerical defense must be directly eligible, with the
existing first Rainbow Curtain rule accounting for its attribute change. The
full sequence still follows MP, Flash, and existing defense compatibility
rules, with final acceptance owned by the Simulator. The new profile inherits
Strength Powder, Mirage target, and Discard/Sacrifice corrections.

The v2 candidate-group namespace stays opaque. Under v2, even that string
keeps the same raw-only treatment as other unknown groups. The v3 profile
defines and validates its use as a candidate hint.

## Alternatives and consequences

Filtering all CutCost armor would erase valid neutral-attack responses, and
looking up the opponent's attack definition would exceed the old contract.
Consuming the Simulator's authorized atomic group preserves both ordinary
responses and support while keeping rules and visibility with the Simulator.
The new profile keeps changed candidates and wire provenance separate from
historical data. Its convergence preset uses a dedicated storage stage and a
fresh parent with the original 1,000-update budget.
