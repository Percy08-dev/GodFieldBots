# 0056: Version the ordinary Rust Model3 PPO update recipe

Status: design accepted, implementation verification in progress.
Date: 2026-11-21. Base: `4912a9f`.

## Context

The ordinary Rust Model3 learner in [ADR0054](0054-single-process-rust-player-runtime.md)
and [ADR0055](0055-owned-rust-model3-parity-validation.md) used fixed 16-game
cohorts, 128-decision logical batches, four full shuffles and scalar execution.
The approved plan makes each of those settings explicit and starts with
256 games, width 64, four epochs, all-decision shuffling and a CUDA cap of 16.

## Decision

The schema 1 `training.ppo_update` table participates in the effective config
hash and existing resume identity. Table-less effective JSON retains the
historical recipe and its recorded hash. A pause/resume invocation uses the
saved recipe and original code, runtime, input receipts and deadlines. The
original array formats retain their meaning.

The ordinary paired schedule and actor11 audit use the same configured game
count, and readback derives Adam counts from reconstructed learner decisions.
Decision order is selected after the existing GAE and normalization. On CUDA,
Model3 can use the Model4 packing/planning mechanisms and still runs Model3
forward and the same PPO objective. Each logical batch makes one Adam step.
The learner's configured, managed and physical limits apply before packing.
Actual errors follow the original terminal Source cleanup.

## Alternatives and consequences

Putting settings beside the checkpoint would require a second source of
resume authority. Embedding the versioned table in the existing effective
config keeps the consistency check at its established boundary. The scalar
path remains the reference and the width-one execution, including CPU.
Grouped floating-point reductions can change the gradient and optimizer
trajectory. The numerical comparison measures those differences on the same
input. A later HPO experiment is required to assess learning quality.

The existing formal preflight pins the AST of the paired schedule and
`Learner.update`. A shared parameterized loop would invalidate that deployed
recipe. The configurable ordinary path carries separate batch-control and
schedule construction and shares the fixed PPO loss and model operations.
Corresponding changes will need verification against both loops. Revisit
sharing when the formal recipe is revised or the PPO objective changes.

See the [reviewed design](../design/ordinary-rust-ppo-update.md) for the
interfaces, capacity limits and assumptions.

## Verification

Independent preimplementation review accepted the bounded design. The
[verification record](../verification/ordinary-rust-ppo-update.md) records
implementation reviews, compatibility checks and platform observations.
