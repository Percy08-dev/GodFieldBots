# PPO HPO diagnostics and checkpoint evaluation verification

## Compared inputs

The baseline is Bots `b8a71cc`. The Simulator contract is
`072dabbca1458c9526897781d2fd64edeeb4f9ca`, API v2, catalog schema 2,
Journal 6 and candidate codec 2. The approved Simulator binary has SHA-256
`76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d`.
Evaluation continues to use the acting participant's public inputs and the
existing heuristic and seat-swapped validation schedule.

## Diagnostics checks

Focused fixed-input tests cover weighted clip fraction, padded action
columns, pre-update explained variance and JSON null on degenerate targets,
the original pre-clip norm, and the current Adam learning rate. The same
CPU fixture exercises MLP, GRU and Transformer. The staged Rust reader run
passed 217 library tests with 13 ignored. The replacement extension
passed 37 native candidate checks plus 15 matched bootstrap checks. Wire7
passed its 13 tests with inherited Unix sockets enabled.

An actual CPU Model4 G0→G1→G2 pause/resume fixture finished with byte-identical
final parameter, Adam-state and Torch RNG arrays and the same shuffle state
as the continuous fixed-input run. A bounded checkpoint panel via controlled
Simulator endpoints verifies the existing collector, replay, checkpoint
provenance and immutable receipt publication. The separate scheduler checks
cover opt-out, the cap, objective identity, clean diagnostic failure and
refusal to resume with uncertain retirement or altered prior evidence.

The integrated Python regression command ran 622 tests with the archived
Torch dependencies and current Python source. CUDA was disabled for this
run. Scoped Ruff, mypy (24 production source files), `git diff --check`
and the original 256-game probe's config validation passed.

```sh
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' /tmp/gf-hpo-diagnostics-venv/bin/python \
  -m pytest -n 0 -q --tb=short \
  tests/test_hpo_{numeric,checkpoint_config,checkpoint_evaluation,checkpoint_schedule,lineage,owned_wire,wire,config}.py \
  tests/test_rust_player_{lineage,campaign_factory,campaign_controller,campaign_training,campaign_evaluation,evaluation}.py
cargo test --offline --locked --manifest-path native/Cargo.toml \
  --bin gf-player-supervisor checkpoint_pause_tests --quiet
```

The native pause test checks missing, mismatched and correct original
pause bindings. A separate native operation-record test accepts the
fresh HPO G0 request and rejects it on the ordinary or resume entry point.
An offline release build of the extension and supervisor passed exact
source/lock hash matching. The built wheel advertises both new capabilities.
The released candidate passed 136 tests from `test_hpo_native.py`,
`test_rust_player_config.py` and `test_rust_player_campaign_factory.py`.

After the Strength Powder adoption, the combined CPU regression rerun passed
707 Python tests with 30 skips. The matching native library passed 218 tests
with 13 ignored. The supervisor passed 27 tests with 14 ignored when run
with local Unix sockets and child processes available. Scoped Ruff and
`python -m mypy src` passed. The installed extension matched the native
source and Cargo.lock hashes.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python \
  -m pytest -n 0 -q --tb=short \
  tests/test_hpo_{numeric,checkpoint_config,checkpoint_evaluation,checkpoint_schedule,lineage,owned_wire,wire,config,native}.py \
  tests/test_rust_player_{lineage,config,campaign_factory,campaign_controller,campaign_training,campaign_evaluation,evaluation}.py
cargo test --offline --locked --manifest-path native/Cargo.toml --lib --quiet
cargo test --offline --locked --manifest-path native/Cargo.toml --bin gf-player-supervisor --quiet
```

The Rust test binaries use the documented [CPython link settings](rust-player-runtime.md).

## Timing

The CPU comparison alternated the baseline and the updated source for
13 runs per model. The first three pairs were warmup. Each update used
16 selected endpoints, two epochs and a fixed seed. The original metrics,
gradients, parameter bytes and Adam state matched on all comparison pairs.
Median update wall time was 31.639→31.522 ms for MLP, 38.540→38.083 ms for
GRU and 41.508→40.375 ms for Transformer. The median paired changes were
+0.49%, −0.86% and −1.15%.

A separate CUDA check on RTX 5070 Ti and Torch 2.14.1+cu132 used 48 selected
endpoints, two epochs, minibatch 16 and five repetitions per variant. A
parameter, Adam, RNG and shuffle digest matched between before/after in all
three model cases. In the four post-warmup runs the median update time was
142.4→133.2 ms for MLP, 1071.2→1009.7 ms for GRU, and 923.4→830.6 ms for
Transformer. This small, non-interleaved sample demonstrates numerical
parity. Its variation and fixture size limit throughput conclusions.

Evaluation performs a separate collection and reuses the existing sealed
readers during Rust pause/resume. The cost of repeated cold/restore audits
and full 200-game panels remains to be measured. The comparisons above
measure numerical updates only.

## Supervised runtime evidence

The initial bounded Rust HPO6 test under
`runs/hpo-checkpoint-verification/parents/campaign-a01` retired its G0 learner
and wrote G0 Prepared and Completion. The attempt then stopped on the old
supervisor requirement for a terminal disposition before releasing a paused
campaign parent. The fixed supervisor checks a durable checkpoint pause
against the original Awaiting, acknowledged Completion and physical
retirement. The first attempt left its evaluation admission unresolved.
It remains ineligible, and a new real GPU job requires the repository's
operator recovery procedure. A post-fix supervised Simulator campaign
and the production 256-game / 10,000-update workload remain unverified.

## Changed implementation areas

| Area | Files |
| --- | --- |
| Per-update diagnostics and compatible readers | `gf_common/hpo.py`, `gf_training/hpo_numeric.py`, `structured_numeric_worker.py`, `hpo_owned_process.py`, `hpo_owned_restore.py`, `hpo_lineage.py`, `hpo_ppo.py`, `native_backend.py`, `rust_player_completion.py`, `native/src/hpo_controls.rs`, `native/src/lib.rs` |
| TOML, checkpoint retention and paired panel proofs | `gf_common/hpo_checkpoint_evaluation.py`, `card_instance_collection.py`, `gf_cli/hpo_config.py`, `hpo_evaluation.py`, `card_instance.py`, `rust_player_evaluation.py`, `gf_training/rust_player_evaluation.py`, `rust_player_retention.py` |
| Committed boundary scheduling and resume | `gf_cli/hpo.py`, `rust_player_campaign.py`, `rust_player_entry.py`, `gf_training/ppo_run.py`, `hpo_owned_run.py`, `hpo_owned_worker.py`, `rust_player_{learning,lineage,restore,supervision}.py`, `gf_common/rust_player.py`, `native/src/bin/gf_player_supervisor/{records,session}.rs`, `tools/rust_player_verify_operation.py` |
| Tests | `tests/test_hpo_checkpoint_{config,evaluation,schedule}.py`, `test_hpo_{lineage,native,numeric,owned_wire,wire}.py`, `test_rust_player_{campaign_factory,lineage}.py` |
