# 0063 PPO HPO diagnostics and checkpoint evaluation

Implemented for the HPO paths described below. The
[verification record](../verification/hpo-learning-diagnostics.md) lists
executed tests and remaining runtime evidence.

## Context

ADR0037 qualifies HPO objectives on fixed validation conditions and a healthy
final-D training run. The learner already records losses, timings, and
decisions on each committed update. Learning stability and performance
trends also need clip fraction, explained variance, the applied learning
rate, gradient norm, and observations of the policy at several checkpoints.

## Decision

Add the versioned diagnostic block to existing HPO update metrics for all
Model4 architectures. Reuse the existing ratio, rollout Critic predictions,
PPO return targets, and pre-clip norm. A legacy native training reader
advertises support before newly written metrics reach its closed record
reader. Historical updates can omit the new block.

Opt-in schedules support owned campaign schemas 4–6. A scheduled boundary
closes the original training resources, evaluates the saved policy through
the existing paired validation collector, and resumes from the saved pair
while the original deadlines still admit work. The Rust supervisor positively
checks each paused operation's match to its checkpoint and physical retirement
before releasing parent custody. The supervisor and original clocks remain
the same across training slices of one seed. An opted-in initial pause uses
a distinct Prepared and Completion with an explicit absent Source hash.
Historic records keep their existing schemas. A matched runtime wheel
advertises this protocol before an enabled Rust campaign opens storage.

Checkpoint panels have their own linkage and receipt, based on verified
inference weights and exact validation conditions. Preserve each natural
outcome and pair identity. Retain the specified inference weights and
select the earlier update on equal scores. A scheduled final-D checkpoint
can reference the usual qualifying evaluation. The Optuna objective remains
the equal-training-seed final-D score. Holdout seeds never run in checkpoint
or objective validation.

## Alternatives and consequences

An evaluator inside a live trainer would share its device lifetimes and
isolation boundaries. Post-training evaluation would delay the information
until the end of learning. Serial execution preserves the existing global
GPU reservation and restart state but adds restore and sealed-history
verification work. The present Rust supervision contract can reopen complete
seed history across processes. Continuing an interrupted unfinished seed
under a new supervisor would require a separate authority transfer.

See the [design](../design/hpo-learning-diagnostics.md) for phase bounds and
the [operator guide](../hpo.md#チェックポイントの定期評価) for configuration and
output paths. Independent architecture review accepted the numerical and
scheduled designs before dependent implementation. The implementation
reviews checked compatibility, ownership and failure settlement separately
from numerical correctness.
