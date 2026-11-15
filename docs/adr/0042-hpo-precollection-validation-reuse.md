# 0042: Reuse still-owned historical validation before resident collection

- Status: Accepted design; implementation/acceptance pending
- Date: 2026-11-15
- Decision owner: root
- Review: `bot_architecture_reviewer` preimplementation READY, including terminal
  expiry, partial-construction ownership, metadata accounting and cleanup allowance

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
Results will be recorded separately; design review alone is not runtime evidence.
