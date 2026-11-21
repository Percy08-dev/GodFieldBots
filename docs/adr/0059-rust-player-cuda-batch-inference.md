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
