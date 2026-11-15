# 0042: Reuse still-owned historical validation before resident collection

- Status: Accepted design, implementation and measured verification
- Date: 2026-11-15
- Decision owner: root
- Review: `bot_architecture_reviewer` preimplementation READY, including terminal
  expiry, partial-construction ownership, metadata accounting and cleanup allowance
- Implementation: independent correctness/spec/quality and conformance READY;
  whole CPU6557PASS/408SKIP on2026-11-16. All10matched physical trials and full
  Source/numeric/receipt audits PASS. Final correctness/evidence and architecture
  conformance are FINAL READY; root accepts scoped implementation/measurement.

## Context

The measured second resident MLP collection spends29.130 of43.731seconds in five
complete parent validations. Four happen before games, but mkdir separates the
third and fourth, and gameplay changes cache references. Removing all five checks
or caching returned dictionaries would lose ownership and mutation guarantees.

## Decision

Adopt the [bounded design](../design/hpo-precollection-validation.md). Retain the
actual full-validation scan across the first three checks, preserve all fresh
owner/learner/phase/plan guards and native dependency coverage, and close it before
mkdir. Later checks retain full validation. Enable only the known resident CLI
composition; direct/custom callers default off. No format/math/dependency change.

Charge retained evidence within existing Source admission and the already-admitted
simultaneous update bound. Actual native execution retires before retained original
custody; uncertain cleanup blocks confirmation. An invalid carrier never triggers
fallback, and normal expiry cannot restart the same scope.

## Alternatives and consequences

Keeping all full replays preserves repeated work. Reusing throughout collection
could remove more work but requires separate mutable registry/reference validation.
A global trust cache adds stale-state and lifecycle risks. The selected bounded
reuse keeps the smallest mutation-free interval; it still rehashes physical files
and cannot promise a particular speedup. Revisit extension beyond mkdir only after
measured acceptance and a separately reviewed exact mutation/reconciliation design.

## Verification

Lifecycle, mutation, original native coverage, resource/failure/cancellation and
compatibility tests; full CPU/static checks; independent correctness/conformance;
matched unchanged-budget2update timings with original Source/receipt audits.
The [verification record](../verification/hpo-precollection-validation.md) records
scoped47case lifecycle/mutation/native coverage, whole CPU6557PASS/408SKIP,
Ruff/mypy,8fixed-kernel ASTs and209wheel/sdist/source correspondence. All24protected
historical files remain byte/stat identical. All10same-code trials and complete
original audits pass after the user frees the GPU. MLP3pair medians reduce
second-collect40.840->30.082s (26.34%) and whole2update119.429->107.762s (9.77%).
GRU/Transformer1pair each support descriptive results only. No formal-HPO,
strength or final-test claim follows. Both final independent reviews are FINAL
READY without blocking findings; root accepts the declared measured scope.
