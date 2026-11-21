# Rust Model3: move frozen-policy parity out of ordinary training

The original proposal was written at
`55cd288507dd0cdcf98ed237f356006d32abd23f` (Rust player runtime).
The scoped architecture review accepted the design against integration base
`742493745ff7f518998b2419e7b7b2b57bd0cc19`.
This is a follow-up to ADR0054 and the accepted ordinary Rust Source composition. It narrows
the ADR0024 runtime-parity requirement for that owner path only. ADR0043 is a
precedent for omitted, explicitly reported parity on owned data, not a grant to
Model3. No experimental batching or Pack/SQLite optimization is in scope.

## Requirement and cost

The same-policy saved/cache/full/device/split comparison is a correctness test,
not part of the PPO objective. Move it out of ordinary Rust Model3 updates.
Keep the full comparison, tolerances and failure behavior callable explicitly in
the numerical validation/tests and on the existing legacy path. Do not add an
opt-out flag, repeat a smaller inference check at every update, rewrite stored
values, or claim the omitted check passed. The 64-game/4-update diagnostic on
RTX 5070 Ti measured 17.795 s in parity out of 62.528 s in the update boundary
(a prior equivalent uninstrumented run took 57.871 s). Measure the change rather
than treating this time as a guaranteed end-to-end reduction.

## Ownership and flow

1. Keep `RustPlayerSource.consume` and its `PERSISTED → VALIDATING → UPDATING`
   transition, exact registry/current parent and original collection owner,
   on-policy snapshot/selection/ACK/Replay checks, immutable numeric inputs and
   published-original rechecks. Projection still owns the existing Model3
   layout/version-4 numeric input and actor-local facts. A submitted rollout,
   serialized proof/metric, external data reader or a configuration flag is
   **not** update authority.
   Review the full correspondence, not just the saved optimizer step count:
   `RustEmptyCollection._inspect` binds the actual capture/manifest and plan
   and projects only the learner seat after raw/Replay inspection. If the
   reviewer identifies a policy/input binding enforced only by the existing
   parity test, resolve it before enabling omission rather than adding another
   inference check by default.
2. Only that synchronous ordinary source transition passes its actual
   `RustPlayerSource` to a private local-learner entry point. Check the exact
   source, pending/current registry slot, local learner, ordinary route and
   `UPDATING` state, and take the rollout from the source, not from a caller
   argument. No repeat of physical file hashing or raw validation at this handoff.
   Keep `LocalGameLearner.update(rollout)` as the existing full-verification
   component path. HPO continues to call its own existing method.
3. The local owner continues to serialize Torch and own the existing update
   device phase/account (including CPU learner/CUDA serving). In
   `structured_numeric`, a private Model3 owned-call entry uses the existing
   nonnested per-learner operation scope with the actual allowance, or no device
   cap for a CPU learner. Before shuffle/Adam, keep exactly one input
   classification, rollout liveness and endpoint/position capacity preflight
   (`admit`). The frozen `Learner.update` AST and constructor remain unchanged:
   its pre-update `parity` call yields `{"status":"not_run"}` only for the same
   rollout in the owned call's ephemeral scope; explicit `Learner.parity`
   outside that scope and the existing `update_with_device_limit` retain the
   full check. The scope is cleared on exception and never beyond the numerical
   call. Do not add another PPO body, alter the loss, clip, epochs, minibatches,
   optimizer, RNG, or transfer another participant's information.
4. Keep source poisoning, account failure custody, commit/save/restore and
   cold original validation unchanged. A full numerical mismatch in explicit
   validation still fails. The normal owned path instead relies on recorded
   accepted events, selected actions, and old-policy values under the bound
   actual checkpoint; it intentionally does not detect a numerical divergence
   by independently rerunning the model at every update. The cold audit still
   checks raw/numeric/artifact correspondence, not reruns of all four engines.

## Alternatives

- A global parity toggle or accepting arbitrary `StructuredRollout` input for
  omission makes the runtime trust boundary depend on caller configuration
  rather than the actual Source. Keep neither.
- A separate copy of the PPO loop avoids the small operation-scope change, but
  risks drifting from the frozen numerical kernel; do not duplicate it.
- Batched or sampled online parity still spends time on inference in normal
  training and changes coverage. Keep the existing full comparison as the
  explicit test instead; the live producer's own binding checks are not sampled.

## Compatibility

The ordinary Rust Update6 envelope already records diagnostic JSON `metrics`.
Keep the envelope; document the concrete omitted report as a distinct result
(`{"status":"not_run"}`), never as a change of meaning for historical reports of
passed comparisons. Do not feed this status back into trust or eligibility.
Existing legacy worker wire/results (which require the checked shape), Rust
Model4/HPO, historical data, checkpoints, version pins/source identity checks
and restore rules stay unchanged. `resume_identity` still binds exact producer
and Python source; no cross-code resume exemption is requested. This is a
host-side change, not a simulator integration/schema change; compare on recorded
Simulator `072dabb` and Model3
checkpoint4/numeric4, retaining current Simulator section 21.1 information
boundaries. No CLI, persisted configuration, dependency or native change.

## Verification and acceptance

- Prove the **ordinary actual Source** dispatch skips numerical comparisons
  while explicit/legacy tests still run them and reject a deliberately wrong
  recorded value. Prove false/unowned/mismatched dispatch still cannot update.
- Prove preflight runs before shuffle, parameters or Adam mutate on the owned
  path (CPU/input classification and constrained CUDA allowance), with original
  terminal/restore paths, account failure custody and no leaked operation
  scope. Cover both old/new tagged metrics without relaxing legacy readers.
- With identical valid rollout and initial state, compare the scalar PPO
  outputs excluding only the parity report, optimizer, tensors and shuffle/
  Torch RNG. Keep all eight frozen-kernel checks and focused Rust Source,
  audit, lineage and Model3 regression tests. Avoid changing history semantics.
  Start with `tests/test_rust_player_learning.py`,
  `tests/test_rust_player_source.py`, `tests/test_rust_player_lineage.py`,
  `tests/test_card_instance_learning.py`, and
  `tests/test_hpo_numeric.py::test_all_eight_frozen_kernels_remain_exact`.
- Run an actual bounded CUDA Rust Model3 update through the supervised ordinary
  command and strict cold audit. Compare before/after update time separately
  from whole wall time on an equivalent fresh workload, without adopting a new
  default runtime or running HPO. State exact scope, run variance and skipped
  evidence. Production acceptance requires independent correctness and
  implementation-conformance review after tests.

### Pre-implementation evidence

On the unchanged source, these three focused CPU baseline checks passed
(`CUDA_VISIBLE_DEVICES=''`, CUDA-wheel test environment, 3 passed):

```sh
python -m pytest -q \
  tests/test_card_instance_learning.py::test_saved_output_mismatch_fails_before_rng_optimizer_mutation \
  tests/test_rust_player_learning.py::test_model3_operation_limit_refuses_before_parity_and_preserves_binding \
  tests/test_hpo_numeric.py::test_all_eight_frozen_kernels_remain_exact
```

The design-review request reached a new Codex thread but returned a usage-limit
error before a custom-role response; it is not a review disposition. Production
patch, paired live CUDA verification and final reviews are not run.

Test-only fixed-data CUDA replay was allowed before the gate. Four pairs on
the original 64 games/677 judgments, separately restoring each original
pre-update checkpoint, used the same frozen scalar update and exactly one
preflight. Full parity took 19.830 s across the four preflight/parity calls;
admission-only took 0.034 s. Total numerical time was 64.636 → 45.350 s; all
eight invocations reproduced every original checkpoint hash and all other
update metrics. This does **not** assess the live provenance change or
authorize implementation. Evidence:
`/tmp/opencode/gfb-rust-trusted-candidates-55cd288/runs/update-profile-64/test-only-parity-omission/summary.json`.


## Integrated result

[ADR0055](../adr/0055-owned-rust-model3-parity-validation.md) records the accepted
decision. The three scoped production files and regression tests are in
`f64fa3e`. Independent correctness and implementation-conformance reviews
accepted the exact implementation. The [actual verification](../verification/rust-player-runtime.md#ordinary-model3-owned-parity-follow-up)
records fresh CUDA before/after, same-source resume and the focused CPU results.
