# 0074 Model4 sequence inference in the Rust HPO6 runtime

Status — Accepted design. The verification record tracks implementation and
physical evidence by model.

## Context

HPO6 already gives Rust custody of the game, inference FIFO, response identity,
and admitted sampling state. Its service can score independent Model4 MLP
rows. GRU and Transformer need own-decision state across accepted requests.
Simulator `6d50603dab3f6d2df34ea4e433bab7dca96f52c1` §21.1 permits reuse of
that state within the exact run, game, seat, and checkpoint. Fixed row shapes
guard each result against changes in neighboring private values or lengths.

## Decision

`batched-torch-sequence-v1` is an explicit HPO6 CUDA selector for Model4 GRU
and Transformer with `canonical-python` Source. Rust keeps one original
dispatcher and provides ordered row acceptance and game retirement tickets.
PyTorch holds the committed own-decision caches and updates their valid
state after Rust accepts the row. Cache retirement completes before Rust
releases the game slot. On failure, the original producers, checkpoint
references and GPU storage settle before device credit returns.

The scalar route uses the existing temporal step. Fixed-width groups bind
the published checkpoint and own candidate/event geometry. Transformer also
binds own committed history length. Groups operate inside a funded
collection allowance with separately accounted persistent caches and
transient storage. The existing Numeric6 Python projection and sequence
PPO consume sealed Journal 6 evidence at the original update boundary.
Existing MLP selectors and a paused parent's exact producer binding carry
forward.

Select the recipe in an HPO6 campaign's `[runtime]` table. Set the campaign
`model_id` to `gru` or `transformer`, use the matching corrected Model3 base,
and set `encoder = "canonical-python"`. The sequence path uses the existing
Numeric6 budget ceiling of 536,870,912 bytes. The verified batch and resource
conditions are recorded with each run.

## Alternatives and consequences

Keeping independent Python actor workers preserves a familiar lifecycle but
duplicates checkpoint residency and managed CUDA capacity. Reconstructing a
full decision prefix on every request repeats temporal work as history grows.
Allowing dynamic group shape based on neighboring lengths permits a private
row to influence another participant's rounding and action sample. The
selected route pays for per-seat cache residency and fixed-width padding
while preserving independent computations and the existing Rust actor path.

The [design](../design/rust-player-sequence-hpo.md) specifies the lifetime,
acceptance and capacity boundaries. The [verification record](../verification/rust-player-sequence-hpo.md)
records the common-path measurements, same-model parity and adoption evidence.
