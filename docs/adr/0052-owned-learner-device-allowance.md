# 0052: Bind a distinct device allowance to the owned CUDA learner

Status: accepted for normal Model4 producer-owned resident CUDA training; independent
implementation correctness and conformance reviews are READY. Baseline: 9e6cb5d.

## Context

A learner starts before its future participant pool is populated. The prior
managed reservation also served as the worker's logical planner cap, which
includes host rollout storage and conservative packing/checkpoint estimates.
Reserving that entire mixed quantity as CUDA bytes, regardless of the actual
declared pool, needlessly refuses some runnable configurations. Subtracting
host bytes without updating the allocation checks would break the relationship
between the worker's executed computation and its reserved capacity.

## Decision

Keep the logical allowance L and the existing input, candidate and tensor
checks. Freeze an additional device allowance D at invocation start from the
declared GPU budget minus all authorized future actor leases and simultaneous
import scratch, capped by L. Refuse a pool that exhausts the budget or a D
that cannot cover the independently estimated initialization, cold restore,
optimizer-step and export minima. Keep the explicit context/workspace reserves.
D is a maximum managed allocation budget, not a physical usage prediction.

The creator checks the original live lease; the child checks the descriptor
before allocating and retains L and D independently. The optimizer's pure
planning and final preallocation check use the existing grouped estimate plus
host residency against L, and the grouped estimate plus the position storage
against D. Check singletons before mutation, using the existing grouping/staging
mechanisms when larger physical groups need refinement. Preserve the logical
minibatch, endpoint order/weights and optimizer-step schedule. Physically
different grouping may change floating-point reduction order.

Bind the allowance/recipe to the normal new invocation identity. CPU, historical
and explicitly experimental executions keep their original launch contract.
Array formats and frozen Source authority are separate from launch admission.

## Alternatives, evidence and follow-up

A new free-form operator override would duplicate the existing global budget.
Automatically raising a limit after OOM changes the recorded execution. The
frozen future-pool subtraction uses existing configuration, and the checked
device guard is applied before device work. Keep historical records readable
and fail incompatible resume as an identity mismatch. Measure physical RAM,
CUDA allocated/reserved and shared-device sampling separately. Future shared
serving must use its own approved physical ownership, including retained
checkpoint identities. See the [design](../design/ppo-shared-inference.md) and
[verification record](../verification/ppo-shared-inference.md).
