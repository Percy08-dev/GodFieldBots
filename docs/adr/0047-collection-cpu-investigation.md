# 0047: Bounded collection CPU and transfer investigation

- Status: Accepted design; implementation/evidence pending
- Date: 2026-11-17
- Owner: root
- Baseline: `f3aeca2`
- Preimplementation architecture review: READY, including explicit P trust
  exclusions,96MiB arena admission and shared-pilot/confirmation accounting

## Context

Collection now dominates the short two-update workflow. The user excludes GPU
model computation changes but explicitly includes actor CPU and transfer counts,
parent validation/recording, HTTP and candidate generation. CPU inference wins
only an isolated small-model screen; it is not a general HPO recommendation.
HTTP tails overlap parent CPU callbacks, while body/JSON/compression are small.
Earlier typed transfers reduce copy counts but do not establish whole-run gains.

## Decision

Execute the [accepted design](../design/collection-cpu-optimization.md) through
tools-only P/T/H installers and original owners/Source/receipts. Fixed learner
replays screen transfer implementations; actual32-game exploratory blocks select
two candidates, followed by independent confirmation and combination trials.
Maximum54performance plus3passive actual runs. Keep production defaults unchanged.

The user chose current192/two-layer conditions only and explicitly chose shared
exploratory control plus two fresh singleton confirmations to stay within54,
rather than66runs with three fully independent singleton confirmations. Report
that selection bias; do not silently describe three independent pairs.

## Alternatives and consequences

Ranking HTTP/parent scheduling from isolated actor replay would compare different
scopes. Switching actors to CPU generalizes beyond evidence. Broad thread
migration creates unreviewed owner/failure changes. Testing every small variant
on actual training expands cost without the requested impact ordering.

This adds bounded experimental tooling and retention/admission complexity, not a
new default or qualification authority. Review later if a measured candidate is
proposed for adoption; remove/reconsider temporary installers when superseded.
No dependency, Simulator, training mathematics or persisted version changes.

## Verification

Focused mechanism/regression/static tests, separate profiler populations, original
actual-source/file/receipt audits and independent final correctness/conformance.
Record results in `docs/verification/collection-cpu-optimization.md`. No quality,
formal-HPO or final-test acceptance follows from the experiment.
