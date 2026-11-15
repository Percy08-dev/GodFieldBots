# 0041: HPO resident participant workers and diagnostic progress

- Status: Implemented; whole correctness/conformance reviews READY; final hardware acceptance pending.
- Progress: Task3 scoped implementation accepted after fix1/69coveringPASS,
  both independent re-reviews READY. Actual CUDA/Simulator/performance and final
  project acceptance remain pending; diagnostics never qualify data.
- Runtime: Task2 scoped implementation accepted after fix1/173coveringPASS and
  both independent re-reviews READY. Actual CUDA/Simulator/performance and
  final project acceptance remain pending.
- Foundation: Task1 contracts/readers/resources/references accepted after fix1,
  covering165PASS and both independent scoped re-reviews READY. Execution was
  guarded at that milestone; Task2 subsequently activated the explicit runtime.
- Date: 2026-11-15. Owner: root.
- Authority: user-approved plan and explicit implementation request.
- Architecture review: `/root/resident_inference_design_check`, READY after
  explicit lifetime, Registry5 cache-reference and Source/commit-order resolutions.

## Context

Fresh HPO participant interpreters repeat model/runtime initialization. The
existing CPU diagnostic attributes most of its1.393s mean load interval to
Torch/Inductor imports, not current CUDA transfer. Small model4 weights permit a
bounded persistent pool, but live-worker-empty Source/update/termination checks
require explicit new ownership. Saved progress exists, while inspection takes
the execution lock and cannot serve live observation.

## Decision

Adopt [the reviewed design/plan](../design/hpo-resident-workers.md): HPO-only,
viewer-partitioned physical owners, full per-episode reset, immutable bounded
inference cache, SAME actual job/ledger and post-commit weight synchronization.
Idle residency stays charged; parked is distinct from reaped. Keep actor-local
generation/selection, isolated learner mutation and actual final termination.
New campaign/spec3+variant selects Collection7/Proof4/Registry5/native plan6 and
an additive native marker; old contracts/numeric math retain their meanings.
Source/commit precedes cache-reference mutations; pins have separate durable
transactions/protection. Progress is bounded diagnostic JSON, stderr rendering
and lock-free read-only watch; never Source/score/cleanup authority.

## Alternatives

Fresh processes repeat measured initialization. A shared GPU service could
reduce contexts/enable batching but expands cross-participant execution/failure
ownership. Choose resident workers first and measure. ADR0027 is test-only CPU
precedent, not production CUDA authority; this ADR amends fresh-process isolation
only for the explicitly versioned new HPO path. No new dependency/Simulator edit.

## Consequences and accepted debt

Reset/park/pins/reap are separate proofs. Missing ACK/uncertain cleanup fail
closed. Accept small duplicated weights/serialized requests initially; revisit
with measured GPU/context/IPC need. Progress may stale during blocking work;
revisit stronger observation only for demonstrated needs. Managed VRAM limits
are soft, not physical guarantees. No speedup/strength result is preaccepted.
Loss of the entire invoker retains pending cache pins and failed-parent storage;
JSON/PID/admission release cannot recreate original release authority. Revisit a
separate administrative reclamation design when that retained storage needs
recovery. Normal failure/pause cleanup with surviving custody must still release
references after actual eviction/reap.

## Verification

All3model equivalence, isolation/fault/reference/admission tests; old format/
kernel/native proofs; optional-free live watch; whole project checks; predeclared
actual CUDA/Simulator update/resume/evaluation/comparison and paired timing;
both final independent reviews. Scoped and whole-project code evidence is in
[the verification record](../verification/hpo-resident-workers.md); remaining
actual resident hardware checks and final acceptance stay explicit pending work.
