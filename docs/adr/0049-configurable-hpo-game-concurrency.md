# 0049: Configurable HPO game concurrency

- Status: Accepted, implemented and verified. Independent correctness and
  implementation-conformance READY; root acceptance complete.
- Evidence: [verification](../verification/hpo-game-concurrency.md).
- Date: 2026-11-18. Owner: root.
- Baseline: `b61a603`.
- Authority: the owner's explicit request for more than two simultaneous HPO
  games and acceptance of responsibility for OOM-related configuration tuning.

## Context

HPO exposes a game count but limits it to two, alongside four participant
workers. Independent repetitions of the same limit exist in configuration,
resource admission, lineage/native controls, resident physical registration and
logical-retirement readers. Changing only TOML validation would fail later in
the real collection or original Source validation path.

## Decision

Implement [the bounded design](../design/hpo-configurable-game-concurrency.md).
HPO game and worker counts accept exact positive signed-63-bit integers, with
workers at least twice games. Keep default 2/4 and one trial at a time. Expand
only HPO entry points and its audited ledger, keeping the shared ordinary and
frozen formal limits unchanged.

Use the same declared ledger capacity for live resident partitions and the
bound original configuration for Python/Rust retirement validation. The receipt
cannot authorize its own capacity. No new schema field, dependency, policy
input, simulator contract or numerical recipe is introduced.

Managed RAM/CUDA admission remains caller-configured and checked. These are
estimates, not physical fit guarantees. The caller selects concurrency and
memory settings; actual OOM remains a failed invocation with original custody
and cleanup rules, no automatic reduction/retry or invented outcomes.

## Alternatives and consequences

A replacement ceiling of four/eight only postpones the same restriction.
Removing admission or silently changing concurrency after OOM changes declared
experiment conditions and resource ownership. Reusing the existing ledger and
count formulas is the smallest change that meets the request.

Larger resident pools consume more contexts, weights, cache, staging and host
resources. No universal fit, throughput gain or learning-quality claim follows.
Revisit the estimates if measured caller workloads reveal missing accounting;
do not label admission as an OS quota. Larger counts may be rejected by older
readers; existing 2/4 records keep their meanings and resume identity stays strict.

## Verification

Require preimplementation architecture READY, focused expanded-count/reader/
admission tests, four concurrent fake/CPU games/eight role owners with positive
retirement and failure coverage, eight-owner Python/Rust receipt checks, ordinary
limit regressions, static checks, matching installed native build and preset
validation. Independent correctness and conformance reviews precede root
acceptance. Real larger-concurrency training remains manual and NOT RUN here.

Initial design review correctly identified missing custody and persisted
retirement limits. Root accepts that blocker and extends the design to propagate
actual ledger/original configuration capacity through all such boundaries.
Independent re-review is DESIGN READY. Implementation passes both final reviews,
expanded-count/CPU owner/Rust checks, static checks, matching native activation
and actual CLI validation for all3models at2/4 and4/8 (CUDA management16GiB).
The final23-case module also covers8/16 materialization/lineage/resource phases.
Root accepts this functional scope; real larger-concurrency learning remains
manual and NOT RUN. No speed or physical-capacity guarantee follows.
