# 0057 Rust-owned PPO training data and sealed collection verification

Status — Accepted and implemented in `1268781` for the measured Model3 and
Model4 configurations. The mode is selected by its explicit configuration.

## Context

The Model3 and Model4 same-process Rust player routes previously reconstructed
learner inputs from saved Journal and Replay data. They repeated that work
during cold completion, resume and imported Model4 parent verification. The
game actor already produces the acting-view tokens, typed candidate table and
the serving results needed to construct a PPO trajectory.

## Decision

Build PPO trajectories in the native game task. The original writer checks
the learner seat and ordered accepted selections against its Journal and
Replay data on first custody. The collection owner authorizes immutable array
views after all games have committed and the seal is verified. Rust owns GAE,
normalization and grouped CUDA padding. Python owns the existing Model4 PCG64
budget selection, PPO and checkpoint operations. The original recording
account charges native capacity through the last array alias.

Pin the training-data and Source-audit recipes in the effective configuration
and the artifacts that identify their use. New normal readers verify the
descriptor and physical database SHA-256 against committed writer, Source and
generation evidence. Historical runs retain strict original reconstruction.
The operation verifier provides `--full-source-audit` for complete original
and numerical comparison. A past run resumes with its saved config and
producer identity.

The selector is `[run.ppo_source]` for Model3 and `[ppo_source]`
for a schema 6 Model4 campaign. Both take `training_data = "rust-ppo-training-v1"`
and `audit = "sealed-pack-sha256-v1"`. Model4 also declares
`encoder = "canonical-rust"` at the campaign root.

## Alternatives and consequences

Persisting and reading another training representation would preserve a
process boundary but would add write/read and copying costs in the same
process. Building tensors in the original writer would duplicate serving
tokenization or require transferring all inference operands to Python. Native
capture can share the acting-view token kernels and preserve the intended
scope of data access. Pack verification relies on the checked original writer
and immutable seal. The explicit full audit can independently check their
output against the Python reference.

The reproducibility contract permits the specified tolerances for GAE and PPO
comparison across encoders. Pause/resume under one recorded encoder requires
the existing exact state checks. Journal 6, Replay, Pack 1 and stored
checkpoint and rollout array layouts retain their established representations.

## Verification

The [design](../design/rust-ppo-training-data.md) records the information and
ownership boundaries. The [verification record](../verification/rust-ppo-training-data.md)
records parity, operational checks and the comparison against `aab5f35`.
Independent correctness and conformance reviews cleared the implementation.
Both workloads passed the median whole-invocation time gate within their
original resource limits.
