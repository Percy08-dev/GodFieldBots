# 0044: Experimental inference execution comparison

- Status: Accepted,implemented and verified; root acceptance after both FINAL READY reviews
- Date: 2026-11-16
- Decision owner: root
- Preimplementation review: `bot_architecture_reviewer` READY after explicit
  private-state reset and all-cache memory-accounting amendments
- Implementation: independent correctness/spec/quality and architecture
  conformance FINAL READY after the control/compiler/measurement/lifetime fixes

## Context

The current CUDA actor spends measurable host time constructing many small
tensors and submitting forward calculations. Existing policy timing includes
CPU submission and synchronization; it does not establish GPU busy time or
Tensor Core use. The approved plan asks for comparisons and a recommendation,
with architecture/training held fixed and learning quality deferred to HPO.

## Decision

Implement the bounded [comparison design](../design/inference-performance-comparison.md).
Share the existing tensor mathematics, keep variants explicitly opt-in, and
compare packing/reuse, chunk sizes, temporary padding, compilation and TF32.
Fixed-input replays screen options before alternating-order actual resident
trials. Compilation applies to helpers the actor actually invokes. Code caches
may survive reset; participant-bearing tensors must clear before acknowledgements.
Charge the complete experimental cache/transition envelope under existing limits.

The tooling binds execution options/code/inputs in independent experimental
metadata and labels production identities as base identities. Actual trials
remain fresh-only, incomplete at their two-update cap, and provide no objective,
training-quality, formal-source or final-test authority. New defaults and learner
precision changes require a subsequent decision.

## Alternatives and consequences

Changing all defaults before measurements would presume a speed/quality result.
A separate encoder or trainer would duplicate numerical responsibility. Compiling
an unused module wrapper would provide no evidence about the real inference path.
The selected approach adds private helper/experimental tooling maintenance and
compiler/cache resource costs; reject unavailable modes rather than hiding eager
fallback or enlarging budgets. Revisit the temporary tooling if a measured option
is later proposed for normal use, with corresponding configuration/identity and
learning-quality evidence.

## Verification

Focused implementation-mechanics checks and existing network/frozen-kernel
regressions; separate independent correctness/conformance reviews; sequential
fixed-checkpoint/input measurements and profiler captures; three alternating
pairs per model under the unchanged short protocol; original Source/termination
audits after timing. Record exact commands, individual results and limitations in
the [verification record](../verification/inference-performance-comparison.md).
No new output/action equality or custom error-threshold gate is introduced.

Final evidence passes the scoped checks and all physical comparisons:100covering
checks,57affected policy checks,final30focused checks,Ruff/mypy9;24paired-screen
and12follow-up replays,14separate CUDA profiles,actual cache-hit proof,18actual
trials/576natural games and18strict original/numeric/receipt audits.1,620physical
file instances retain their SHA/stat identity. Both final independent reviews
are FINAL READY and root accepts this comparison scope.

Reusable input storage improves isolated inference9.6–14.7%,but actual paired
whole-helper changes are−0.09% MLP,+2.21% GRU and−0.92% Transformer. Keep normal
defaults. Scorer compilation has no finite measured payback; observed TF32
TensorOp kernels do not yield a useful whole-route gain. Broader compiled regions,
maximum-capacity stress,hardware utilization and learning quality remain outside
this evidence. Revisit adoption only with consistent benefit in the intended
parallel workflow and any necessary quality/identity acceptance.
