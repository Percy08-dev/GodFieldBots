# Participant-specific numeric cost declarations

Status: accepted narrow5c1B correction after independent architecture READY;
implementation conformance and matching build checks remain. No Simulator/rules change.

The current native GameSpec has one numeric_costs declaration used by both seats.
Existing own_game_worker_settings_v3 assigns default NumericCosts to baselines
and model/binding geometry, working-memory and operation costs to MLP. A mixed
MLP/baseline game must preserve each participant's existing generator admission
and actual Journal numeric_costs_hash; selecting a global maximum or applying
MLP costs to the baseline changes the policy recipe/provenance.

Add optional PolicySpec.numeric_costs (default None; omit its serialized key when
None so old component inputs/transport retain their shape). GameSpec.numeric_costs
remains the explicit DEFAULT declaration. Validate every participant's effective
costs: its override when present, otherwise the default. Resolve and use the
acting seat's cost object for its candidate generator/report only; no other-seat
private data enters policy. Existing search math/order, recipe and limits stay.
B production collection supplies each seat's actual own_game_worker_settings_v3
costs explicitly, including baseline defaults, under its immutable fixed plan.

Pending transport already contains the actual PolicySpec and generated report,
including numeric costs/hash; bind those effective values to that participant's
actual journal metadata. Do not add misleading new global effective values.
Terminal transport's numeric_costs remains the DEFAULT declaration and its policy
array includes any explicit overrides. The canonical encoding context retains
that default plus per-policy native/journal declarations and verifies exact
correspondence before Journal writes/Source issuance. For fallback None, journal
costs must equal the applicable default. Same-account raw bundle identity, context
plan/receipt checks and all chronology rules remain unchanged.

Inner Journal6/DTO3/codec2 and its per-participant cost/hash semantics do not change.
Native transport v1 remains the same intended content contract with an optional
policy declaration; do not relabel old transport hashes or rewrite originals.
All captures/contexts record actual per-seat declarations. New Source must reject
wrong-seat cost provenance rather than silently recomputing/replacing it.

Scope: native game.rs PolicySpec/seat resolution, capture.rs only serialization/
fixture support, client facade and actual Python encoding context/collector,
plus relevant tests/stub if its signature needs it. No new dependency, resource
coefficient or preflight guard. Counts/costs are existing policy configuration,
never a complete legal-action oracle or hidden legality feature.

Verification: fixed own-input/clock generator parity for MLP versus baseline/Refresh
with their own costs; actual mixed MLP–Random, MLP–Heuristic and Refresh games/
Journal replay/candidate provenance at W1/W2 as appropriate; different costs
cannot contaminate the other participant; wrong/invalid override refusal; old
homogeneous/fallback/transport fixtures unchanged. Component traces do not grant
learning quality or default adoption. Final native/companion build identities must
match after these Rust changes.
