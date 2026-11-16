# 0043: Producer-owned resident HPO data

- Status: Accepted, implemented and verified; root acceptance after both FINAL READY reviews
- Date: 2026-11-16
- Decision owner: root
- Preimplementation architecture review: `/root/owned_architecture` READY, with
  inference/resume bank isolation resolved; final implementation correctness and
  conformance reviews READY

## Context

ADR0042 retains historical validation between three precollection checks. The
resident loop still reconstructs internally generated samples from archives,
reopens numeric/checkpoint output and performs whole-batch parity at every update.
The approved next plan makes correctness follow from actual generation ownership.

## Decision

Adopt the explicit [producer-owned design](../design/hpo-producer-owned-data.md):
capture actual accepted commands and terminal views, retain complete own history,
seal RAM banks with kernel write seals and transfer owned FDs to the learner.
Persist output before update, derive saves and actor SYNC from one actual immutable
learner snapshot, and maintain live registry transactions under the parent lock.
Use explicit producer-origin formats and truthful omitted-parity status. Cold
resume validates its external restore boundary once without replaying unused data.

## Alternatives and consequences

Repeated full audits remain the legacy/control path and explicit audit tools.
Unchecked public dataclasses, a general trust cache, disk spool and actor hidden
state reuse fail the requested ownership or computation contract. The chosen
path requires Linux memfd/Unix FD transport and explicit lifetimes, while retaining
existing budgets, model math, persisted checkpoint layouts and information limits.
No speed or memory improvement is presumed before the prescribed measurements.

## Verification

Ownership/failure tests, strict-reference all-model numerical comparisons,
fresh/resume and campaign/evaluation checks, full project/static/package checks,
independent correctness/conformance reviews and18matched physical timing trials.
The four implementation milestones pass both independent review roles. The final
cold-import milestone includes actual pruned-parent pause/resume/full-audit and
ownership-failure regressions. Real two-update validation exposes audit-accounting
and zero-Act identity defects; their scoped repairs also pass both reviews.
Capture registers the actual journal bank before any decision; the offline audit
uses absolute high-water growth and actual last-use retirement with pool holds.
No reader/format/budget is relaxed. Whole CPU6868PASS/408SKIP and exact225module
packaging pass. First-set C data remains historical/ineligible; fresh strict-audited
v2 physical verification now passes: all-model fresh/resume and CLI/evaluation,
18paired trials,30full-original parent audits and40actual cleanup receipts.
Whole-helper median reductions are64.61%/63.44%/63.22% under the prescribed short
MLP/GRU/Transformer conditions. Both final whole-branch/evidence reviews are
FINAL READY; root accepts this approved functional/performance scope. Limits and
the ineligible first-set history stay explicit in the verification record.
No memory-reduction, training-quality or final-test claim follows.
