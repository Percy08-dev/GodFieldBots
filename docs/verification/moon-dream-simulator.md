# Combined Moon God and Dream Simulator verification

## Release identity

| Item | Value |
| --- | --- |
| Combined Simulator revision | `f394a7b57b0ffb12c6b5782ff52fdb8842ec7534` |
| Simulator parents | `9e47d4c5f8cd3f60437f9aeabc95076c1dc04321`, `0eca2d6fe70fcf19bd0753fe400a7eff4422b955` |
| Common change inventory base | `59e2e009af0846063d09963dc9da08e91cf6ab42` |
| Spec inventory SHA-256 | `c2a0e37607bff98e342b618c5057887cdf6adca981b81731245adf5853cdfc78` |
| Changed file inventory SHA-256 | `2ac3dd61dd077420b404decede74aaee17d16ec5d5185b5fac0068725d3f663e` |
| TestServer SHA-256 | `15137c72e670a90a64fd474030ee9024c91b97b1282c8d103db895f54a2bfa37` |
| Shared Action profile | `card-instance-simulator-v2-shared-20261125` |
| Wire and saved formats | Gameplay API v2, public event schema 2, catalog schema 2, Journal 6, `candidate-columns-v2`, HPO6 rollout 8 |
| Native source inventory SHA-256 | `1b54cac5f3c238c3a0fb05524baac9679164d3179edf6adef170d5893c1441f6` |
| Native wheel SHA-256 | `f0115a9ae33519f95ab27831985ed43acb32b7cdb5aa4dc902df4ba32018ed8c` |
| Native extension SHA-256 | `4e19c5e3e1d3a37d20a03de053424987186bf8fcfdfc3b5f9cbb091552cc46f1` |

The source merge was made in the separate `runs/moon-dream-compat/simulator`
checkout. The original Simulator repository was used as a read-only source.
The local `runs/moon-dream-compat/simulator-release.bundle` contains the exact
combined commit and both sides of the merge above the common base. Its SHA-256
is `a64a76e8417802be814484e869123c0e0a2b8d6c666b285be67f092d45acedbf`.
Fetch `refs/heads/fix/dream-compat` from that bundle into a separate clone
containing `59e2e00`, then check out `f394a7b` to recover the approved source.
The complete file hashes are recorded in `runs/moon-dream-compat/release-pins.json`.

## Simulator and Bots checks

- **PASS** `cargo fmt --manifest-path runs/moon-dream-compat/simulator/Cargo.toml
  --all -- --check` and `cargo test --offline --locked --release --manifest-path
  runs/moon-dream-compat/simulator/Cargo.toml --features test-support --no-fail-fast`.
  513 passed. The 10,000-seed nightly soak was ignored. The 96-game corpus
  matched its existing digests and policy RNG. The ordered Dream candidate
  rows for all 296 definitions and the Moon God selection pool matched
  `9e47d4c` exactly. The fixed Full Moon Blade Mirage and Aura tests passed.
  A final `cargo build --offline --locked --release --features test-support
  --bin gf-test-server` from the combined checkout produced the release SHA above.
- **PASS** The pinned p83 case replayed 70 accepted actions. The original
  revision 70 discard returned HTTP 200 at revision 71. Its actor feed and
  Replay recorded Player 2's `full-moon-blade` and weapon attack 10 against
  Player 1. The Native candidate generator offered the fixed defense from
  actor-visible inputs, and the Simulator accepted it at revision 72.
  [Recorded case](../issues/cases/0016-moon-god-standalone-mirage.json)
- **PASS** On the combined TestServer the p65 replay retained eight
  concealed draws in Act, actor feed, and Replay. The fixed exchange was
  accepted at revision 8. The archived p11 and p41 failures reproduced on
  their own historically pinned releases. The old p65 case, Native
  concealment parity, and marker-free saved fixtures passed with the new wheel.
- **PASS** Focused Python regression run returned 564 passed and 77 skipped.
  It covered runtime receipts, reciprocal old/new pins, native candidate
  and storage validation, HPO6 entry and campaign, fixed replays, and saved
  token compatibility. The four pinned replay cases also passed directly
  through `uv run --no-sync python -m pytest -q -rA -n0
  tests/test_hpo_dream_discard_reproduction.py` with the matching Simulator
  environment variables. Native library tests passed 221 cases with 13
  ignored. `ruff check src tests`, `mypy src` on 288 sources, and
  `git diff --check` passed. The full Python suite was **NOT RUN**.
- **PASS** Independent design, implementation conformance and correctness
  review cleared. A full read-only audit passed for the earlier Dream smoke
  `31edf132-841e-4c26-945d-fec91795c902` using its recorded identity.
  The prior wheel failed the combined profile capability check. The verified
  wheel was installed in `.venv` with `uv pip install --offline --no-deps
  --reinstall --python .venv/bin/python`. `uv run --no-sync` then passed the
  default launch preflight, including the matched companion, CUDA, runtime
  attestation and the dedicated `mlp-256-convergence-1000-moon-dream` storage.

## CUDA HPO6 smoke

**PASS** Run `50d90624-1f57-46cb-a8f3-4cb591ef81c2` used learning seed 17,
an RTX 5070 Ti, CUDA Torch `2.14.1+cu132`, tmpfs staging, and dedicated ext4
output. All eight games ended naturally, with 149 accepted actions and 77
learner decisions. The operation saved one PPO update and paused at the
requested boundary. The independent `--full-source-audit` matched Journal,
Replay and Python-regenerated numeric inputs. The test config, exact command
and outputs are in `runs/moon-dream-compat/smoke`. The full audit SHA-256 is
`bd3f972bb2f974f893e2c8e531b7bec6436d2c24fab06c1c16feda63eb81ed12`.
The 256-game convergence run was **NOT RUN**.
