# 0045: Experimental PPO update execution comparison

- Status: Accepted, implemented; physical comparison and final acceptance pending
- Date: 2026-11-17
- Decision owner: root
- Baseline: `1364748`
- Preimplementation review: `bot_architecture_reviewer` READY after explicit
  aggregate concurrent-prefix/event activation accounting amendment
- Implementation: independent correctness/spec/quality and conformance READY
  after bounded core and harness fixes; physical evidence remains pending

## Context

The approved plan asks for measured cumulative reductions of host loops, small
transfers, synchronization and duplicate history encoding, keeping PPO, TBPTT,
FP32 execution, all ownership and budgets unchanged. The
[preparation investigation](../verification/learning-update-optimization-preparation.md)
pins six batches and their correct G0/G1/Adam/RNG start states, but establishes
no speed improvement or implementation approval.

## Decision

Apply the bounded [update comparison design](../design/learning-update-optimization.md).
Keep the existing trainer/network and baseline arithmetic path. Experimental
metrics, immutable packing, admitted tensor microbatches and short-history
sharing are opt-in private execution choices. Original coordinator, creator,
worker module, Wire7 and Source capabilities remain authoritative. Explicit
hashed worker settings and separate experiment sidecars identify executions;
normal config and checkpoint schemas do not change.

## Alternatives and consequences

A separate trainer would duplicate PPO and network semantics. A reusable
inference buffer could be overwritten during backward. Sharing different TBPTT
cuts would alter gradient reach; batching the complete logical minibatch could
violate memory and single-tensor limits. The selected approach adds bounded
grouping/admission and diagnostic tooling maintenance while preserving step and
information boundaries. It promises no performance improvement. Revisit this
experimental debt on production adoption or protocol retirement.

## Verification

Preimplementation architecture review, focused grouping/mask/gradient/lifetime/
failure tests, existing numerical/resume/ownership/frozen-kernel checks, 84
fixed-state CUDA updates and 18 alternating actual runs with post-timing strict
original/numeric/termination audits. Both final independent reviews are required.
Results, missing evidence and the defaults recommendation belong in
[the verification record](../verification/learning-update-optimization.md).

The code milestone passes350original host regressions,67final focused checks
with1inapplicable skip, scoped Ruff/mypy and optional-free current-source CLI
help. Both independent implementation reviews are READY. Source/result
population validation, actual observer compatibility and profiler scope are
tested independently of GPU admission. This does not establish physical
performance, learning quality or completion of the approved comparison.

## Actual telemetry follow-up

The84core/25profile/18actual population and strict audits complete successfully.
Both final reviewers hold whole-plan acceptance because actual learner allocator,
processCPU/PSS/RSS and isolated retirement duration were not retained. No earlier
core-only waiver exists. Root accepts the advisor's minimum remedy: a separately
frozen six-run diagnostic supplement under the original conditions, with bounded
observers and strict audits; original18 timing rows remain immutable and are not
pooled with diagnostics. Six is an evidence-coverage decision, not an originally
prescribed count. The [bounded amendment](../design/learning-update-telemetry-supplement.md)
passes preimplementation architecture review after an explicit held16MiB
observation-lease amendment. Observational implementation passes independent
correctness/conformance,119affected host checks, final targeted checks and one
gated actual-CUDA worker/receipt smoke. Six-run evidence is pending. No numerical, default,
Wire7 or Source-authority change is authorized by this remedy.
