# 0059: Fixed-shape CUDA inference for Rust player collection

Status — Accepted and implemented for explicit CUDA selection on the measured
Model3 and Model4 workloads.

## Context

[ADR0054](0054-single-process-rust-player-runtime.md) moved request formation
and result acceptance into Rust. The Python service still made a full set of
GPU transfers and a PyTorch forward for every request. Model3 and Model4 can
reuse inference operations within the same process. Simulator `072dabbc`
§21.1 requires that neighboring participants cannot influence a request's
outputs, sampling state, or checkpoint selection.

## Decision

Select `batched-torch-v1` explicitly in the ordinary `[run.player_runtime]`
or HPO `[runtime]` table. The effective config and runtime identity bind the
choice. Omission keeps existing scalar inference. HPO uses its campaign's
recipe. An ordinary base runtime still has to match the HPO scheduling fields.

Use Rust's existing formation window and fixed `min(game_concurrency,
max_batch_size)` width. Group by checkpoint and each request's own capped
power-of-two candidate and event shape, with a separate zero-event shape.
Use deterministic dummies, row-local candidate normalization, and a fixed
GRU that selects the last own event. Use the existing per-participant CPU
sampler and restore original order before Rust validates and commits replies.

Bind group capacity once after publishing scheduled checkpoints. The global
allowance counts the actual learner and cache residency and the shared host
charge. The model and binding limits cover the checkpoint's inference
residency and group. The tensor cap applies to individual group allocations.
Capacity decline uses scalar serving for the whole shape. Runtime failure
keeps partial transfers and related owners until retirement and GPU settlement.

## Alternatives and consequences

Padding to the largest request present could increase batch occupancy, but
lets neighboring requests change the computation shape. Splitting work into a
second Python queue duplicates formation and adds another custody boundary.
Fixed shapes avoid those problems at the cost of padding and more forwards
when requests land in different buckets. Updates keep the existing PPO math.

The [design](../design/rust-player-cuda-batch-inference.md) specifies the
geometry, failure boundary, and numeric comparisons. The
[verification record](../verification/rust-player-cuda-batch-inference.md)
records CUDA independence, capacity, lifecycle and matched benchmark evidence.
On the RTX 5070 Ti at 256 games and concurrency 16, the median collection
time fell 7.6% for Model3 and 7.7% for Model4. Pooled p95 fell in both models.
The largest Torch reserved peak was 170 MiB under the 12 GiB managed
allowance. Different models or narrower budgets use the same pre-transfer
admission and require their own performance measurements.

## CUDA inference temporary allowance

Accepted on 2026-11-22. The batch planner maps the normalized default
model tensor limit of 64 MiB to an allowance of 128 MiB for temporary
CUDA batch allocations. Lower model limits remain binding. The aggregate
same-host, model and device reservations apply, including the configured
workspace that covers the observed cuBLAS allocation. Weight, learning
and saved data validation continue to use their existing limits.

The 128 MiB comparison eliminated 351 and 321 scalar fallbacks from the
two measured workloads. The widened shapes passed scalar-reference and
neighbor-independence tests and the ordinary and HPO standard operation
verifiers. See the [verification record](../verification/rust-player-cuda-batch-inference.md).

## Direct CUDA Graph for a single collection shape

The operator can select `batched-torch-graph-v1` in the same runtime table.
Its candidate buckets start at 64 and its positive-event buckets at 16,
subject to the existing caps. Width and zero-event handling follow the fixed
batch design. The earlier Eager recipe keeps its fixed geometry and identity.
An omitted worker count in a fresh Tokio declaration materializes as 16.
Explicit counts remain valid, and the fixed scheduler requires a count.

The collection captures `(32, 64, 16)` for the current immutable learner
checkpoint before starting actors. Eligibility is limited to the measured
MLP compute configurations and the tested Torch, CUDA and allocator
combination on SM120. The same recipe runs its Eager forward for other
checkpoints, shapes, deployments and capacity declines. Only the forward
is captured. Input copies, result copies, per-participant CPU sampling, and
the original full Rust batch submission follow their existing boundaries.
The device budget charges the Graph inputs and private pool once. The
collection releases them after producer settlement and the device fence,
before the next PPO update. CUDA failures use the existing failed-operation
owner and terminal retirement. A later checkpoint gets its own capture.

The matched 256-game tests improved median collection time by 5.5% and
4.8%. Model3's pooled response p95 rose from 4.56–4.70 to 4.83–4.97 seconds.
Model4's p95 fell from 4.97–5.10 to 4.83–4.97 seconds. The capture's measured
persistent cost was 44.7 and 54.7 MiB. Reusing a capture across multiple
collection waves would avoid only about 0.09 seconds of already-warm
preparation per hypothetical 1,024-game update. Additional shape slots and
cross-collection reuse require their own throughput and residency evidence.

The [production design and matched results](../superpowers/plans/2026-11-21-rust-cuda-inference-performance.md)
record the acceptance conditions and timing boundaries.
