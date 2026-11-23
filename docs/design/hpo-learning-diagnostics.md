# PPO HPO learning diagnostics and checkpoint evaluation

The numerical diagnostics and the checkpoint evaluation changes have separate
implementation and acceptance gates.

## Scope

The Model4 HPO learner is shared by MLP, GRU and Transformer, including the
Rust HPO6 MLP Source path. Ordinary PPO recipes, the PPO objective and Adam
steps keep their current definitions. Existing HPO diagnostic keys and the
final-D Optuna objective retain their meanings.

Relevant contracts are ADR0037 (HPO, equal unique-decision budgets and final
evaluation), ADR0046 (bounded PPO execution), ADR0054 (Rust host custody), and
ADR0057 (Rust PPO Source). The existing HPO6 MLP config pins Simulator
`072dabbca1458c9526897781d2fd64edeeb4f9ca`, external interfaces §21.1, API v2,
catalog schema 2 and Journal 6. Evaluation reuses the configured profile and
its normal player-visible policy inputs.

## Per-update diagnostics

Keep per-update output in the committed update's `metrics`. Append an
optional, closed `diagnostics` object (`schema_version = 1` for this object).
Existing updates without the object are valid historical observations. Fresh
HPO learner updates produce all four diagnostics. Wire6 and Wire7 validate
the block in live HPO update acknowledgements. Wire7 initialization/restore
keeps the empty metrics object. The common Python validation also accepts the
historical absence when reading persisted results, including owned restore.
The native persisted reader validates the optional block independently.

| Key | Meaning |
| --- | --- |
| `clip_fraction` | Count of selected endpoint evaluations whose existing FP32 ratio satisfies `abs(ratio - 1) > clip_range`, divided by all selected endpoints evaluated across epochs. Padding belongs to the action or context axis, never to the endpoint axis. |
| `explained_variance` | `1 - Var(target - old_value) / Var(target)` over the unique selected loss endpoints. Use their exact PPO GAE return targets and the frozen rollout Critic predictions from before optimization. Use FP64 population moments and JSON null if target variance is at most `1e-12` or the estimate is nonfinite. |
| `pre_clip_grad_norm_mean`, `pre_clip_grad_norm_max` | Mean and maximum of the norms returned by existing clipping, one per optimizer step after all accumulation for that step. |
| `learning_rate` | Actual LR of the current single Adam parameter group, sampled before the last optimizer step. There is currently no scheduler and the Actor and Critic use this group. |

Extend the existing grouped detached tensor metric reduction with the clip
indicator. Reuse the existing logical step boundary for gradient norm
accumulation on device and include both summaries in the final metric transfer.
The legacy singleton loop may reuse its already required synchronization.
Keep the standard scalar loss helper's existing return shape when the
extra endpoint terms are not requested. Compute explained variance once from
the existing rollout vector API with streaming centered moments and constant
additional memory. This avoids collecting model outputs, adding tensors to the
persisted numerical protocol or repeating any forward. Optional distribution
summaries are outside the initial implementation unless existing measurements
can be reused without widening numerical or collection buffers.

The diagnostic block is part of the existing committed update hash and
participates in normal persistence. Its presence grants no Source authority.
Old legacy native readers that enforce closed metric keys require a matched
package for new records. Advertise a separate additive
`card_instance_hpo_update_diagnostics = "ppo-update-diagnostics-v1"` build
capability and require it when creating a legacy HPO native training
admission. Historical read-only access does not require the new marker.
Same-host HPO6 uses its normal producer identity and Python-owned record
validation. Keep existing frozen ordinary PPO kernels and historical native
capability names meaningful.

Verification includes handcrafted clipped and unclipped selected endpoint
examples with a padded action axis, weighted groups and epochs, nonconstant
and degenerate returns, the pre-clip return from actual clipping, effective
optimizer LR, same gradients/checkpoint outputs on fixed input and three-model
CPU update checks. Exercise accepted historical records and the closed new
diagnostics shape in both Python and Rust. Measure fixed-rollout before/after
update wall time and endpoint throughput. Record CUDA results or the hardware
limitation separately.

## Scheduled checkpoint evaluation

Evaluate at the committed checkpoint boundaries, with the training job closed
before the evaluation job and its GPU admission start. This reuses the
existing resume pair, the serial GPU admission, fixed two-seat heuristic
schedule, original/Replay verification, and the campaign's authenticated Rust
host reentry. Running an evaluator inside the live trainer would introduce
overlapping device lifetimes and a second party to its RNG and source
boundaries. Evaluating all checkpoints only after training would delay the
feedback until after a potentially multi-day parent.

The campaign opts in with a closed optional `[operating.checkpoint_evaluation]` TOML
table. Omission or explicit `enabled = false` disables it. When enabled,
`updates` defaults to the requested 0/100/250/500/1000/2000/3000/5000/7500/10000
series, restricted to the configured update cap if the update list is
omitted. An explicit list is ordered, distinct and bounded by the cap.
`retain_updates` is an optional explicit list of other generations whose
inference and weights survive retention. Existing latest resume artifacts
continue to carry the actual Adam and RNG state. Optional
`final_test_game_seeds` is a separately identified, disjoint reserve, never
used in checkpoint selection or Optuna scoring. No automatic final-test
execution is added. Bind active settings and the seed split to campaign and
effective-run identities; configurations without the table preserve their
historical normalized representation. An enabled table declares the
per-panel `wall_seconds` budget. The evaluator uses the existing evaluation
phase reserves within this budget. This budget is a ceiling within the
original run/session limits, with measured evaluation time recorded in
the corresponding receipt.

The opt-in execution is supported on the producer-owned campaign schemas
4 (MLP, GRU and Transformer), 5 (shared MLP) and 6 (supervised Rust MLP).
Schemas 1–3 keep legacy campaign execution and receive the per-update
diagnostics. Their historic native Source contract does not gain a
second checkpoint retention policy. Configuration validation rejects a
scheduled feature on those older campaign entry points before work.

For every HPO proposal and training seed, the campaign orchestrator binds
one original run ID, configuration and set of absolute training deadlines.
Fresh and resume slices use the already committed generation counters, with an
explicit fresh HPO initialization boundary for G0 when scheduled. G0 contains
the original initial weights, Adam and RNG state. The operation and paused
parent validators allow this boundary only for configured HPO evaluations,
with authentic initial commit and physical retirement; ordinary PPO and
historical operations retain their pause/resume semantics. Positive
milestones use the existing absolute committed-update pause mechanism.
After a clean slice, evaluation executes before training dispatches the next
slice. D, cap and deadlines are checked against the original values and
cannot be reset by evaluation, new host processes or resumed sessions.
Evaluate a scheduled terminal generation when training reaches D or stops
healthily at the update cap. Skipped milestones past termination remain
absent. A final-D checkpoint's matching evaluation uses the existing
independently budgeted objective operation and is referenced once in the
checkpoint index. The intermediate panel budget applies while a training
slice can continue, and to a healthy update-cap diagnostic panel.
Training checks D and the update cap before deciding to pause at a committed
milestone. Progress evaluations, including cap-only parents, have a distinct
linkage purpose and cannot qualify an objective.

For Rust, use one supervisor root throughout a seed's training slices.
Existing session soft dispatch applies to every slice. Before launching an
enabled seed, the original run wall budget must fit before the session's
soft finish, since resume into a new supervisor root is unsupported.
When the remaining session time cannot fit that budget, stop at the
seed boundary. Validate that a fresh session is long enough to dispatch
one such seed. The original run admission, execution and finish limits
apply throughout, including evaluation. The existing authenticated cold host
closes each paused invocation, releases its borrowed parent, and starts the
next invocation under the same supervisor. The fresh operation under
that root supplies the only run clock. Each slice has a distinct operation
ID and awaiting record. Its immutable settled receipt binds its own
awaiting, admission, Prepared, Exit, completion and original physical
termination, with the same parent/run/proposal/seed identity. A verified
settled receipt enables the next slice. The existing Native parent-release
gate reads this receipt at its canonical location and checks the exact
current Awaiting, acknowledged Completion and physical retirement before
releasing the paused parent's custody. A missing/inconsistent receipt or
unconfirmed physical exit uses existing interruption/terminal policy.
Campaign history recognizes these exact slice receipts before requiring a
seed's final training receipt. This preserves RustResumeAudit's same-root
ownership invariant. A campaign session can normally stop between seeds.
An externally interrupted Rust seed cannot continue into a different
supervisor root. Legacy uses the same logical sequencing with the current
durable GPU lease and an immutable slice intent/closure pair.

Authenticate initial pause via the fresh scheduled HPO operation and the
committed generation-zero inference/resume pair, the schedule controls and
verified cleanup. At that boundary no PPO update, Source manifest, or loss
endpoint exists. Allow a null Source manifest hash in authenticated G0
Prepared and matching Completion only. Persist the operation receipt used to verify the G0 boundary
when resuming. Treat successful end/cap before admission of another slice
as a terminal result. If a panel cannot fit the remaining original
evaluation window, record a skipped status and reason without scoring it.
A failed panel with confirmed cleanup can be recorded as diagnostic failure,
with continuation subject to the original budget. Unconfirmed custody stops
the sequence under the existing GPU quarantine contract.

Use one immutable per-seed, per-generation evaluation result under
`proposals/<proposal>/seeds/<seed>/checkpoint-evaluations/<update>/evaluation.json`.
Its intent and finished receipt bind the original update/generation and
inference digests, exact evaluation conditions and seed panel, count of
committed training games and used learner decisions at the checkpoint, and
evaluation elapsed time. The progress linkage uses a checkpoint receipt
digest in place of the final Source receipt digest. Its purpose and
explicit version distinguish the two types. Validate checkpoints after
the original training owner returned and the pinned inference weights
match the committed generation. The progress proof has no status or digest
accepted as final-D authority. Reuse the two-seat heuristic schedule,
collection, artifact admission and natural-result projection. Retain the
per-GameSeed seat pair outcomes. Include W/D/L,
win rate (wins / games) and score ((wins + .5 draws) / games). These allow
paired uncertainty analysis and avoid treating two seats as independent
seed draws. Only a complete verified panel ranks. Select best inference per
training seed by score, breaking ties by earlier update. Expose its identity
in a small per-seed index derived from verified committed receipts. Pin the
inference artifacts for scheduled and explicitly retained updates using the
common HPO retention recipe, so the best remains available. Keep the latest
resume and terminal weights under the established checkpoint contract.

The campaign drives irreversible operations through immutable intents and
receipts. A completed checkpoint panel is not run twice on resume. A partial
panel has no score and preserves the existing failure and cleanup semantics;
no retry or apparent complete record can hide failure. Rust HPO6 keeps the
original supervisor, parent lock, cold completion and cursor handoff across
each slice. Neither project creates a service, queue, new policy inputs or
cross-boundary learner tensor ownership. Separately record training collection
and update durations (existing files) and evaluation time (each receipt).
The Rust cold/restore passes still verify original seals and saved files.
The default ten-point schedule through G10000 can entail about 48,700
batch visits in cold and restore combined versus 10,000 final cold visits
without intermediate evaluation. Benchmark a bounded run and report the
cost of scans and games separately. Reuse the current cold audit for
checkpoint authorization so evaluation adds no extra full training-history
scan. Reserve no unmeasured capacity margin.

Verification: opt-out preserves legacy projection hashes and operations;
closed options reject impossible explicit schedules and validation/test
overlap; validate/schedule exact 0/1/2 boundaries with a bounded panel;
compare continuous versus evaluation-interrupted fixed-input optimizer,
shuffle and RNG state; check model/checkpoint/condition identities, paired
terminal completeness, draw and best tie treatment, source separation,
storage protection and safe recovery at a closed boundary. No ten-thousand
update run is required. Hardware/actual-runtime checks are reported
separately from injected scheduling fixtures.
