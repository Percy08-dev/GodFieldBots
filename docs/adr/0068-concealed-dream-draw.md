# 0068 Concealed Dream draw events

Status is Accepted. Independent design, correctness and conformance reviews
cleared. [Verification record](../verification/concealed-dream-draw.md)

## Context

Simulator `9e47d4c5f8cd3f60437f9aeabc95076c1dc04321` projects a draw as
`artifact_drawn` with `definition_id: "concealed"` when Dream prevents it
from resolving the public display. Pair p65 stopped in HPO6 batch `000007`
after the Simulator accepted the seventh action. Player 1's projection
contained eight of these draws. [Recorded case](../issues/cases/0015-concealed-dream-draw.json)

## Decision

- Python and Rust accept the exact marker on a CardInstance `artifact_drawn`.
  A marker on any other event type and unknown definition IDs are rejected.
  The acting participant's `/events` feed remains the history source. Act
  validates the acknowledgement. Simulator spec 07 §21.1 governs policy inputs.
- EventBank and Journal 6 preserve the original events, interval positions,
  limits and prefix hashes. Replay compares the same actor-visible sequence.
- Feature generation omits each concealed draw from the interval and every
  history statistic. It scans backwards through the original prefix to find
  the most recent 128 retained events and whether a 129th exists. The scan
  uses the configured history size where a model specifies another window.
  Rust accumulates bounded integer counts and sums during the reverse scan
  so it can finish without a temporary history buffer.
- Serving and training size the three interval arrays from the retained rows.
  Readers check matching dimensions and original-backed reads regenerate the
  arrays and require exact equality. An empty persisted interval keeps its
  existing shape. Ordinary draws and prior marker-free records retain their
  token and hash identities.
- The corrected Native wheel advertises this behavior in `build_info`. Current
  Dream HPO6 launch and the corrected Native reader check
  `card_instance_concealed_draw = raw-preserved;feature-omitted-v1`.
  Historical audits derive the capability requirement from the producer
  identity the run recorded.

Mapping the marker to the existing missing category would still add a draw
and an actor count, and could push an earlier event out of the history window.
Dropping it before persistence would change the prefix hash and break Replay
correspondence. Omission at feature generation preserves both boundaries.

## Compatibility and review

The Simulator source, gameplay and public event schema 2, catalog schema 2,
shared Action profile `card-instance-simulator-v2-shared-20261124`, Journal 6,
candidate-columns-v2, and persisted version identifiers stay pinned. The p65
parent remains failed and quarantined. Live resume retains its recorded binary
identity requirement.

The default convergence preset selects the corrected Dream release, the
1.5 GiB numeric budget, and the established Dream output paths for new runs.
Historical audit and replay use the settings and artifacts recorded at launch.

The architecture reviewer cleared this design before implementation after
checking the Simulator and ADRs 0062, 0065 and 0067. The review identified both
prepared-array shape gates and the Native serving count as points to verify.
Implementation review restored the independent raw interval cap on public
frame tokens and checked that recorded build evidence reaches historical
Source, final validation and Tree readers before they open originals. The
correctness review checked Python and Rust parity, prepared and owned interval
shapes, and rejection of changed arrays during original-backed regeneration.
Both reviewers cleared the resulting implementation.
