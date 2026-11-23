# PPO HPO learning diagnostics implementation plan

**Goal.** Make per-update diagnostics and fixed-panel checkpoint outcomes
available in existing HPO records for the supported Model4 configurations.

**Design.** [PPO HPO learning diagnostics](../../design/hpo-learning-diagnostics.md).
Review numerical records and checkpoint scheduling independently before each
implementation starts. Use the existing Python learner and Rust reader, the
serial GPU ledger, campaign store, and latest-checkpoint restore contract.

1. Add focused CPU metric tests. Cover a clipped endpoint, ragged selected
   groups, constant returns, actual Adam LR and pre-clip norm. Check exact
   model/optimizer equality on the three existing Model4 fixture paths.
2. Extend the learner and closed Python/Rust record readers. Check old
   diagnostics-free records and new malformed record rejection. Add the
   legacy native training-only capability check and verify the relevant
   Python and Rust readers. Measure identical fixed-rollout CPU updates
   before and after and record whether CUDA evaluation is available.
3. Add the opt-in checkpoint schedule and holdout seed split, preserving the
   effective representation of historical configurations. Exercise old and
   new config round-trips and persisted checkpoint retention.
4. Use the existing isolated HPO evaluator at closed generation boundaries.
   Test the initial and committed-update milestones, full seat-pair coverage,
   deterministic best selection, pause/retirement/resume, and equivalence to
   fixed-data uninterrupted optimization. Verify failure and completion
   remain distinct from an objective-eligible final-D training parent.
5. Document options and read the generated JSON, then obtain independent
   correctness and conformance review. Record actual test results and
   unmeasured GPU or long-run behavior.
