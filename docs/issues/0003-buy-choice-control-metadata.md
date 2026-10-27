# Separate Simulator task: actionable metadata for a pending BuyDecision

- Status: Simulator correction and Bots adoption verified on 2026-10-28;
  a new complete pilot is pending. Historical failure is retained below.
- Source pin: `44a26e3a16b808e826deb332fa72acfc5037d3ef` plus reviewed dirty
  documentation hashes in ADR0005. Binary build revision remains unknown.
- Binary SHA-256:
  `d9c774d214e0e732fed040e7ef2e386979fc1f42e225429c1dbcfc642b49831f`.

## Correction verification — 2026-10-28

Simulator commit `860d8b92db4ce82012e040510f221b52241f28aa` introduces the
target's Allow/Mirror response before the offer, then permits the current buyer's
`resolving / buy_accept_reject` input with the existing `buy_decision` descriptor.
The normative CLI contract documents that narrow exception.

The updated prebuilt TestServer passed the saved six-action case plus an Allow
response and both buyer decisions in separate same-seed games. Actor metadata,
nonactor false/empty metadata, same-revision ActionOptions, card/money effects,
stable repeated reads and clean process shutdown passed. A separate existing
prebuilt HTTP regression also passed. Source checkout identity does not prove
either executable's build revision.

At the initial confirmation, Bots still pinned the old contract and rejected the new actionable
resolving Observation with `typed metadata mismatch`. No production validator,
pin, dataset, checkpoint or pilot has been changed by this confirmation.
Bots adoption has since passed the architecture gate, a narrowly scoped validator
update and fresh integration checks under [ADR0007](../adr/0007-staged-buy-contract-adoption.md).
The complete new pilot remains pending; the old failed pilot stays ineligible.
See the [initial confirmation](../verification/simulator-buy-fix.md) and
[current adoption verification](../verification/buy-contract-adoption.md).

## Observable mismatch

The first ActionOptions feasibility pilot completed 192 games. Nine stopped at
an actor Observation with `phase=resolving`, `required_choice=buy_accept_reject`,
`current_player=viewer`, `actionable=false` and `typed_action_schema=[]`.
No buy-decision proposal was submitted against this contradictory control state.
This is separate from ActionOptions authorization and attribute compatibility.

Shortest saved prefix: [six-action public case](cases/0003-buy-choice-control.json),
SHA-256 `37bbc376476cf26fc9a2703049a1ff41c5157e7b2c4b5713a85fd7814fd00be4`.
All six actions return HTTP 200. The last is Buy by P2 at revision 5; the next
P2 Observation at revision 6 asks for buy acceptance but disables input metadata.
The fixed prefix was reproduced on the recorded binary in 0.099s at
`runs/diagnostics/buy-contract-a9d2723895e3`, followed by owned deletion and clean
stdin-EOF exit 0. No descriptor was fabricated and no action followed false.

## Historical cause and handoff at the original pin

- At the original revision, [economy.rs](../../../GodFieldSimurator/src/engine/resolution/economy.rs),
  lines 201–211, creates pending buy with `GamePhase::Resolving`,
  `RequiredChoice::BuyAcceptReject` and the buyer as current player.
- At the original revision, [dto.rs](../../../GodFieldSimurator/src/server/transport/http/dto.rs),
  lines 724–750, emits action metadata only in AwaitingAction/AwaitingResponse.
  Its BuyAcceptReject mapping therefore does not handle actual pending buys.
- The protocol already declares `buy_decision` with boolean `accept`.
- Spec 10 §3.1 both maps buy_accept_reject to that descriptor and states every
  resolving Observation is non-actionable. Those statements need reconciliation.
- Existing DTO tests exercise BuyAcceptReject in AwaitingResponse, not through
  an actual buy-offer transition into Resolving.

Define the current actor's pending-buy choice as an explicit exception while
preserving false/empty metadata for automatic resolving states. Emit the existing
buy_decision schema only when the pending-buy/actor invariants hold; update the
normative scheduling rule and tests through a real Engine→Application→DTO buy
transition. The Simulator owner chooses/pins the resulting compatibility/revision.

A subsequent Bots integration update must adopt that reviewed contract and its
revision/hash, including its Observation actionable predicate for the specific
Resolving/Buy choice. Do not simply weaken all resolving-state checks. No such
source/contract bypass is included in ADR0005's current implementation.

## Evidence, boundaries and acceptance

Source pilot: `runs/pilot/baseline-feasibility-pilot-354f64ce-747`.
Manifest SHA-256 `2450c3cc3c72c81e27985081a55df723e308705b95a98bc03fd9d88e38969238`;
report SHA-256 `a17108ea5810d919a38389644cd7ed59e2df496991df7332ceaecb0b611ffd4a`.
The JSON case retains source public-wire hashes/request IDs. It contains no
same-revision ActionOptions reply because Bots stopped before fetching it.
It contains no hidden dump, canonical state, rejection-derived mask or opponent
private Observation. It is diagnostic-only, not a teacher dataset.

Acceptance: pending actor can accept/reject after a real Buy offer; nonactor and
other resolving states remain false/empty; reads do not advance State/RNG; raw
positive actions and episode outcome remain authoritative. Re-run fixed case,
new actor-hints validation and the complete frozen feasibility pilot. Preserve
this failed pilot; do not substitute successful episodes. No Simulator source
or binary was edited/built in the Bots task.
