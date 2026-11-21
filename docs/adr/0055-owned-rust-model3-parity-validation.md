# 0055: Validate frozen Model3 policy parity outside owned Rust updates

Accepted and implemented on 2026-11-21 in `f64fa3e`.
Independent architecture and correctness reviewers accepted the implementation.

This decision narrows the live verification obligation of [ADR0024 §16](0024-card-instance-simulator-v2.md)
for the ordinary Rust Model3 path specified by [ADR0054](0054-single-process-rust-player-runtime.md).
[ADR0043](0043-hpo-producer-owned-data.md) provides a precedent for reporting
a deliberately omitted comparison on data held by its actual producer.

## Context

Ordinary Rust Model3 already binds the current checkpoint, accepted commands,
participant journals, Replay and numerical input to a single-use Source.
The common learner nevertheless recalculates saved, cache, full, device and
split prediction parity before every optimizer update. The proposed change
came from the [parity-validation design](../design/rust-model3-parity-validation.md).

## Decision

The local learner admits the unchanged numerical rollout from the same pending
ordinary Source during its synchronous update transition. Preflight checks the
rollout and its resource bounds before shuffle or Adam. The private scope
reports `{"status":"not_run"}` for the frozen kernel's comparison on that
rollout. The common PPO kernel, explicit parity validation, legacy and HPO
paths, checkpoint version and original-data audit retain their existing
contracts.

This retains the trusted producer's recorded log probabilities and values.
A numerical divergence that would only be detected by replaying the separate
engines must be checked in explicit validation and regression controls for
this path. Revisit the decision if the recorded-output producer, checkpoint
binding or inference numerics change. Scope and actual Source ownership are mandatory. A configuration
toggle would permit omission without that binding. A copy of the PPO loop
would put its frozen numerical semantics at risk.

The information boundary is Simulator `072dabbca1458c9526897781d2fd64edeeb4f9ca`
section 21.1. A paused parent continues to bind its exact recorded producer,
configuration and deadlines.

## Verification

The evidence records explicit comparison, same-input update and owned-source
rejection tests, a matched bounded CUDA observation and current-build resume.
See the [verification record](../verification/rust-player-runtime.md).
