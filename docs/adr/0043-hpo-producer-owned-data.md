# 0043: Producer-owned resident HPO data

- Status: Accepted implementation design; implementation/verification pending
- Date: 2026-11-16
- Decision owner: root
- Preimplementation architecture review: `/root/owned_architecture` READY, with
  inference/resume bank isolation resolved; final implementation reviews pending

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
Implementation and measured acceptance are pending.
