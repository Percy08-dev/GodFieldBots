# Phase 4: native CPU LightGBM

Date: 2026-10-27. Root accepts the implementation in ADR0006 after independent
correctness and architecture-conformance reviews returned **READY**, with no
unresolved blocking findings. Model implementation and synthetic verification
are complete. **Real-data learning and model-quality gates are unmet.**

## Scope and identities

- Native LightGBM `Q(o,a)` regression, current-observation and public-event
  aggregate encoders, deterministic candidate-content inference, native
  checkpoints and strict `train` CLI/configuration.
- Whole-run data validation, positively acknowledged executed-only labels,
  train/validation/pilot lineage checks, recomputed clean-pilot evidence,
  explicit accepted-decision target/overshoot, per-seed original budgets.
- One owned native-fit child; metadata-only parent checks, bounded clean-worker
  native preflight, staging/publication and formal completion qualification.
- No new dependency, Simulator change, PPO adapter, GPU test or resume support.
- Encoder `participant-action-v1`, model/train manifest schema 1. Existing raw
  journal/derived schema 4, run manifest 3, input `player-atomic-options-v1`,
  generator `public-atomic-subset-v3`, reward `natural-terminal-v1`, projection
  `simulator-closure-v1` and catalog identities are unchanged.
- Simulator revision remains `44a26e3a16b808e826deb332fa72acfc5037d3ef`, with the
  user's existing documentation edits. The prebuilt binary is unchanged; no
  claim is made about its build revision.

The default experiment is `configs/experiments/lightgbm.toml`, effective hash
`cbab756d8551e87f02a52546d7fefc275c41dd941b6e850e7740d8404602234e`.
It selects 31 leaves, learning rate .05, 500 rounds, early stopping 50, one CPU
thread, model seeds 17/29/43 and 7,200 seconds per training run. It deliberately
contains no invented dataset, successful pilot or decision target.

## Final checks

Commands below use `env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/gfb-uv-cache`; the
inherited shell environment pointed to the Simulator venv. `--no-sync` preserves
the project environment containing the already-selected data/LightGBM extras.

| Check | Result |
| --- | --- |
| `uv run --no-sync pytest -q` | PASS: 353 tests, 3 opt-in skipped, 27.40s |
| Same ordinary suite under temporary parent coverage | PASS: 353 tests, 3 opt-in skipped, 27.52s |
| `GFB_RUN_INTEGRATION=1 uv run --no-sync pytest -q -m integration` | PASS: 3 tests, 353 deselected, 2.22s |
| Same live suite appended to coverage | PASS: 3 tests, 353 deselected, 2.15s |
| `uv run --no-sync ruff check .` | PASS |
| `uv run --no-sync mypy src` | PASS: 33 source files |
| Independent final correctness subset | PASS: 22 data/cleanup tests, 7.21s; no blockers |
| Independent final architecture subset | PASS: 61 cleanup/data/publication/native tests, 20.07s; conformant |
| `uv build --offline` | PASS: wheel and sdist |
| Wheel in separate Python 3.14 environment, without extras | PASS: installed-package imports, CLI help, smoke/baseline/LightGBM config loading; optional packages absent |
| Wheel in separate data/LightGBM environment | PASS: native toy load/prediction and owned clean-worker initialization/cleanup |

Tests include feature values/missingness, ordered references, instance-ID
invariance, candidate permutation, perceived definitions, input isolation,
executed-only targets and reward perspective, split/ancestry conflicts, source
integrity, failed/malformed/incomplete pilot guards, config merging/path origins,
native early stopping and exact save/load, tamper/runtime incompatibility, two
checkpoint identities in swapped seats, and toy/unready rejection before startup.
Failure tests cover timeout/cancellation/log limits/reaping, partial invalidation,
delayed writes, provenance cost and first-deadline preservation.

The live tests use the existing baseline: repeated create independence,
natural-end journal/Replay reconciliation and shutdown, plus a nonnatural
accepted-decision cutoff without fabricated terminal labels. They do not run a
LightGBM comparison or authorize real-data fitting.

## Synthetic component run

Reproducer and machine-readable evidence:
`runs/lightgbm-toy/phase4-b2ded98dd7ab/{verify.py,verification.json,fit/}`.
The copied script can be rerun from the repository's installed environment;
it creates a new uniquely named toy run, retaining prior artifacts.

The script constructs allowed public fixture inputs with two weapon candidates
and 32 synthetic HP contexts. A deliberately invented preference target assigns
the two cards -1/+1; these are **not observed game returns**. There are 48 fit
rows, 16 validation rows and 280 features. The owned native CPU child completed
40 rounds, validation L2 `4.053621024660071e-13`. This proves numerical execution
on an easy synthetic task, not playing strength or generalization.

Exact save/load predictions, reversed-candidate score correspondence, dedicated
worker selection, process/thread/stream cleanup, public-history encoding and
formal rejection of `purpose=toy` all passed. Scores were approximately
`[-0.99999936, 0.99999936]`. Low-level toy fit code provenance is explicitly
unknown; the script digest is recorded separately. The CLI real-training path
records current-fit provenance under its preparation budget instead.

The separate wheel ML environment is `/tmp/gfb-phase4-wheel-ml`: Python 3.14.9,
LightGBM 4.7.0, NumPy 2.5.4, PyArrow 26.0.0. It reloaded the same toy artifact,
reproduced its validation L2 and initialized/reaped a clean policy worker.
The baseline wheel environment `/tmp/gfb-wheel-verification` has no PyArrow,
LightGBM or Torch. Neither environment imports application modules from the
source checkout. The first offline ML install lacked cached registry metadata;
the subsequent normal install succeeded. GPU remains NOT RUN.

Wheel: `dist/godfieldbots-0.1.0-py3-none-any.whl`, 83,305 bytes,
SHA-256 `0220e4e1d3ef0f2189a7057c72972c69c2d409d32f78ac3ec84578e894c4010f`.
The previous baseline wheel is preserved under its existing SHA-256 archive.

## Review findings and dispositions

All consequential findings were resolved and independently rechecked:

1. Native loading in the coordinator could bypass timeout ownership. Parent
   inspection is now metadata-only; clean owned children perform native loading
   and staged-model validation before Simulator startup/promotion.
2. A ready artifact could become visible before child/log cleanup and successful
   completion persistence. Staging remains unready until cleanup completes;
   linked training completion is persisted before readiness is published last.
3. Source collection identity did not identify the current fit code. Training
   records current Git/package/module provenance and exact request/array hashes,
   kept outside policy features and inference metadata.
4. A failed invalidation write could leave one store looking successful. Formal
   qualification requires both artifact and linked healthy completion; either
   durable failure blocks use. Delayed final writes are included in returned
   cost, with persisted fields explicitly identified as pre-write snapshots.
5. Cancellation during worker shutdown could return before reclaiming a resistant
   child. One shielded owned close task is shared across concurrent/repeated
   close calls without resetting its deadline. Actual TERM-resistant tests
   preserve the original cancellation even if cleanup subsequently fails.
6. Source-provenance hashing occurred after shared preparation cost was frozen.
   It now runs under the original deadline before the final charge; a delayed
   final provenance call proves that expired preparation starts no native fit.
   Later model seeds keep independent original budgets.

## Coverage and limits

Parent-process line coverage is **3,212/3,664 = 87.7%**, branch coverage
**1,212/1,522 = 79.6%**, including ordinary and three actual integration tests.
Encoder: 90.1%/81.1%; native model adapter: 88.8%/73.2%; training controller:
87.0%/76.2%; source validation: 93.3%/83.3%; worker owner: 91.2%/88.0%
(line/branch). These are measurements, not a full-coverage guarantee.

Fresh subprocess coverage is not collected. In particular, the native-fit worker
shows zero parent coverage while its actual success/failure lifecycle is tested.
Manual CLI/wheel/component checks are outside these percentages. No permanent
coverage dependency or threshold was added. Artifacts are
`/tmp/gfb-phase4-coverage/{coverage.json,summary.json,html/index.html}`.

Reproduce using the temporary cached tool, then append the opt-in integration
suite with `--append` and `GFB_RUN_INTEGRATION=1`:

```console
COVERAGE_FILE=/tmp/gfb-phase4-coverage/.coverage uv run --offline --no-sync --with coverage python -m coverage run --branch --source=src -m pytest -q
```

## Gate and next work

`gf-bots train --config configs/experiments/lightgbm.toml` exits 2 with a missing
clean-pilot error. Selecting the retained failed pilot exits 2 with a finalized
healthy-run error. `--mode resume` exits 2 before work. None produces a successful
checkpoint, silently replaces episodes or grants a new budget.

The existing 192-game pilot remains failed/ineligible due to the
[pending-buy control metadata defect](../issues/0003-buy-choice-control-metadata.md).
No real-data fit, new feasibility pilot, LightGBM live model comparison or
learning-quality evaluation was run. Fixing/verifying that Simulator contract is
a separate task; then a complete newly identified clean pilot and preregistered
train/validation conditions are required. Model/toy implementation of later
phases may proceed under the user's plan, but cannot claim the learning gate.

Accepted OS/filesystem hard-deadline and dual-storage failure/quarantine limits
remain in ADR0004/0006. Formal use currently requires the checkpoint's training
run bundle; portable inference export is deferred to Phase 9. No push or
publication is part of this acceptance.
