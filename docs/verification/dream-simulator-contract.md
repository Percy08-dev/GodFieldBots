# Dream Simulator contract verification

## Release identity

| Item | Value |
| --- | --- |
| Simulator source | `9e47d4c5f8cd3f60437f9aeabc95076c1dc04321` |
| Spec inventory SHA-256 | `afefd6f3b9113b813c90aaae7bb7b18f1e6299e52bf814ef34ec22112362ab5e` |
| Changed file inventory SHA-256 | `eb56add45d4b4e89af3baf44ec25b0a909886229ed0dffe6d41e20b9c0019951` |
| TestServer SHA-256 | `f95fadbc8eaad23fbe63e17ac9bdb0ebeca4b237956dedeab09d86f30cfd94bc` |
| Action profile | `card-instance-simulator-v2-shared-20261124` |
| Candidate rule signature | `0334e53e7b7081f1cbb5b0771a9b091b49dac9c9d77d8035b07251916c11ff12` |
| Native wheel SHA-256 | `bf9fb21db106a0130dc52384ce24a310566ed470274642e33f4d14af035de2c8` |
| Native source inventory SHA-256 | `e7771d25d115d7f69a11094a151bd3bd024c6fd0debfd0877aeae6aa52c5bf0b` |

The clean isolated Simulator checkout and release binary are under
`runs/dream-compat/simulator`. The original 3172da4 release stays in its
original directory for historical diagnostics.

## Simulator

- **PASS** `cargo fmt --all -- --check` and
  `cargo test --offline --locked --features test-support --no-fail-fast`.
  511 tests passed. The ignored case is the 10,000-seed nightly soak.
  All 96 games of the fixed corpus ended naturally. Ten game digests changed
  and the policy RNG baseline matched all 96 games.
- **PASS** The ordered native Dream candidate matrices for all 296 definitions
  matched the planned delta. 106 definitions changed and 458 directed
  pairs disappeared with survivor order preserved. New component tests
  cover both directions, exact CostCut armor membership, draw and shuffle
  RNG, and Ice Age with a genuine exempt Staff at MP 0.
- **PASS** `cargo build --offline --locked --release --features test-support --bin gf-test-server`.
  A read-only cross-repository review cleared the rule diff and independently
  checked the ordered matrix. The Simulator's dedicated `spec_guardian` role
  was unavailable in the Bots session.

## Bots

- **PASS** 494 focused Python tests with 21 skips for optional or other pinned
  runtimes. Both preflights accepted the exact old and new release and refused
  the reciprocal source and executable swaps. Journal tests covered reciprocal
  profile errors at pre-ACK and read time. The native v5 differential suite
  checked the new-profile visible CostCut and Strength Powder cases.
- **PASS** The pinned 3172da4 TestServer and new wheel reconstructed p41's 46
  accepted actions and the original HTTP 422 using
  `tests/test_hpo_dream_discard_reproduction.py`. The explicit old pins were
  supplied through `GFB_P41_TEST_SOURCE`, `GFB_P41_TEST_SERVER` and
  `GFB_HTTP_DIAGNOSTICS_TEST_CONFIG`. The new release storage tests used
  `GFB_DREAM_COMPAT_TEST_SOURCE` and `GFB_DREAM_COMPAT_TEST_SERVER`.
- **PASS** 40 native player-runtime tests, Ruff on `src tests`, and mypy on
  288 source files. The Rust test executable linked the selected CPython
  3.14 shared library explicitly. The extension and supervisor were also
  built into the matched release wheel and loaded from the isolated environment.
- **PASS** Independent architecture conformance and correctness reviews.
  All three versioned HPO6 presets loaded and selected the same new release.

## Physical HPO6 verification

These runs used learning seed 17, CUDA Torch 2.14.1+cu132, Python 3.14.10,
and an RTX 5070 Ti. Collection staged on tmpfs and the closed original Packs
were saved to the designated HDD ext4 mount. The matching runtime environment
and wheel are under `runs/dream-compat/venv` and `runs/dream-compat/wheels`.

- **PASS** Smoke run `aa9b754d-ba86-4bcb-93d2-fdc2b12a5037` completed eight
  natural games, 149 accepted actions and one committed update with 77 learner
  decisions. The independent full Journal, Replay and numeric audit passed.
  Evidence is under `runs/dream-compat/acceptance/smoke`.
- **PASS** Campaign `96a32171-5c62-409b-aa2d-171ba7d9f894` completed its
  checkpoint evaluation path. Run `5481af06-1afc-4455-86eb-91ef5127fd86`
  settled both generations 0 and 1 with four seat-swapped games per panel.
  The first smoke's shorter wall envelope caused panel skips. The retry used
  an evaluation-specific wall envelope with the same game and model settings.
  Evidence is under `runs/dream-compat/acceptance/evaluation-retry`.
- The first 256-game convergence run
  `4c082241-f7da-4759-a3ca-b930819f505a` committed five updates. Its
  sixth Pack has 256 natural games and 3,454 learner decisions. The host
  recorded a generic `ContractError` before writing numeric output.
  Offline reconstruction matched the fifth batch's persisted metadata and
  demonstrated that the sixth batch's existing numeric admission guard
  required at least 1,076,260,266 bytes, exceeding the 1 GiB budget by
  2,518,442 bytes. The lower bound uses metadata reservation, resident array
  bytes, native array descriptors, and the space needed to write the largest
  observed candidate column. It describes the guard's calculation. The
  diagnostic script and original TOML are under `runs/dream-compat`.
- **PASS** Fresh convergence run `ab9b6c5c-0753-43fe-bc2f-01edb232f390`
  used the 1.5 GiB numeric budget in `54dc0a1`. Overall RAM, Source, CUDA,
  and learning budgets used the original preset values. It committed the same
  3,454 learner decisions in update 6, then paused after update 7 with 20,272
  learner decisions and 1,272 optimizer steps. All 1,792 games ended
  naturally, including both seat assignments of batch 6 pair p41. The
  observed original command exited 0 and the independent sealed-Pack and
  saved-pair audit passed. Evidence is under
  `runs/dream-compat/acceptance/stage7-retry`.
- **PASS** The independent full Journal, Replay and Python numeric comparison
  passed for all seven batches using the read-only `audit-by-batch.py`
  adapter. The combined `--full-source-audit` attempt reached the default
  12 GiB accounting limit. The adapter kept the original full comparison
  and opened each batch under the exact locked parent and committed hashes.
  It used an 18 GiB external raw-audit allowance for five batches and 21 GiB
  for the final two, subject to the observed available host memory. The
  numeric reader used its existing 12 GiB codec limit. These are audit
  allowances. Each full result matched its sealed-Pack audit identity.
  No HTTP error archive was produced by the successful training run.

| Stage 7 audit artifact | SHA-256 |
| --- | --- |
| `sealed-audit.json` | `13cafbb6581b8d27e0184245af138f296842fa83a5f79250ce3cdc805598a100` |
| `full-audit-by-batch.json` | `f3b5e5216075a00ee09df4e154d69580574925fedb209575416fd4abbe67b9b0` |
| `audit-by-batch.py` | `133ebdef3eba376037715e05e4c36a4f7efe8625fd1d20d01da9a853829c2389` |
