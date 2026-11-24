# Dangerous Pestle and Mirage patch verification

## Release identity

| Item | Value |
| --- | --- |
| Simulator revision | `6d50603dab3f6d2df34ea4e433bab7dca96f52c1` |
| Combined base | `f394a7b57b0ffb12c6b5782ff52fdb8842ec7534` |
| Upstream correction | `4fcd7c2290a07649f3d051742beb49deaed1c873` |
| Change inventory base | `59e2e009af0846063d09963dc9da08e91cf6ab42` |
| Spec inventory SHA-256 | `05b68761d1c23a3784d04093b417875b2b5893739a1e6c73367a969dc83db753` |
| Changed file inventory SHA-256 | `eb801942ca482a6d6e381442faff2857fa45f70c16432ad517ac190724785342` |
| TestServer SHA-256 | `500d55cd8eb03126f65d3c24fe58b6091bf8acdb2fc20aed76622a8679f4000d` |
| Action profile | `card-instance-simulator-v2-shared-20261125-pestle` |
| Formats | Gameplay API v2, public event schema 2, catalog schema 2, Journal 6, `candidate-columns-v2`, HPO6 rollout 8 |
| Native source inventory SHA-256 | `f94a03194e3fe31fc077fd972a3acb2143517d93d11072f697a14ca7be2f7b67` |
| Native wheel SHA-256 | `134e651afc61f95f87da6890bbec2bda63e28ef09fd870e1f443134b993bcc2e` |
| Native extension SHA-256 | `85b8f85d71025a5e294bbe14bbae5fe823649e12ac8d0766d55d4eb131f33f66` |
| Native supervisor SHA-256 | `c7eb304fe0d2cdfc8f6b15462c863e7784fdc4aee686c771c13c7a045bad51e7` |

The release checkout is `runs/pestle-mirage-compat/simulator`. The local
`runs/pestle-mirage-compat/simulator-release.bundle` contains the release and
both sides of the prior merge above the change inventory base. Its SHA-256 is
`264ea2b1043c920a2683911ed3e569d8f5bc4060646483cc2d4cd6513ddac6ae`.
Fetch `refs/heads/fix/pestle-mirage-hpo` from the bundle into a separate
checkout containing `59e2e00`, then check out `6d50603`. The file inventory is
in `runs/pestle-mirage-compat/release-pins.json`.

## Simulator checks

- **PASS** `cargo fmt --manifest-path runs/pestle-mirage-compat/simulator/Cargo.toml
  --all -- --check`.
- **PASS** `cargo test --offline --locked --release --manifest-path
  runs/pestle-mirage-compat/simulator/Cargo.toml --features test-support
  --no-fail-fast`. 515 passed. The 10,000-seed nightly soak was ignored.
  Loopback permission was enabled for the HTTP tests. The standalone
  `cargo build --offline --locked --release --features test-support --bin
  gf-test-server` yielded the TestServer SHA-256 listed above.

## Bots and archived replay

- **PASS** Independent design, implementation-conformance, and correctness
  reviews. The reviews verified reciprocal profile pin rejection, prepared
  HPO entry checks, joint target counting and emission, and the acting-player
  input boundary. The native library suite passed 221 tests with 13 ignored.
  `ruff check src tests`, `mypy src` on 288 source files, and `git diff --check`
  passed.
- **PASS** Focused Python tests covered CardInstance actions and rules,
  runtime receipts, HPO config and entry, native search and runtime, and HPO
  progress. 795 passed, 56 skipped, and 1 was deselected. The excluded match
  progress fixture stalled both on the working tree and in an extracted
  `29bc8f7` baseline using stub runtime and writer objects. The full Python
  suite was **NOT RUN**.
- **PASS** The [p0 fixture](../issues/cases/0017-pestle-mirage-target.json)
  replayed the 11 recorded accepted actions on both pinned releases. At
  revision 11, the archived Python and Rust searches generated both targets
  and the recorded self submission returned HTTP 422 `invalid_target`. The
  repaired searches generated the opponent submission, which the patched
  Simulator accepted at revision 12. The actual draw selected Player 1 and
  the pending attack was neutral with strength 30. The simulator-only
  attribute check ran after the candidate searches and the submission.
- **PASS** `python -m pytest -q -rA -n0 tests/test_hpo_dream_discard_reproduction.py`
  passed both p0 cases and the pinned p41, p65, and p83 cases with matching
  TestServer environment variables. The opt-in p11 case was skipped because
  its approved source was not supplied for this run.
- **PASS** A read-only full Journal, Replay, and Python numeric audit of the
  archived Moon/Dream smoke `50d90624-1f57-46cb-a8f3-4cb591ef81c2` passed with
  the new wheel and its original Simulator pin. Audit SHA-256 is
  `7c966f28d2fc3e56929de9906f159b672ee0e36c3bf3f57653b99890cc801266`.
- **PASS** The installed wheel and companion passed the default HPO preset
  preflight with the pinned source, CUDA, and dedicated storage. The default
  still specifies 256 games per update and 1,000 updates. The launch command
  remains at the top of the [preset](../../configs/hpo/mlp-256-convergence-1000.toml).

## CUDA HPO6 smoke

**PASS** Run `0b87482b-0635-45c4-8a01-a1ba1dfc58d7`, learning seed 17, used
the RTX 5070 Ti with CUDA Torch `2.14.1+cu132`, tmpfs staging, and dedicated
ext4 storage. All eight games ended naturally. The run accepted 149 actions
and used 77 learner decisions. It saved one PPO update and paused at the
requested boundary. Recorded collection and update/save times were 6.71 s
and 5.28 s. No throughput comparison was run. The independent full-source
audit compared the Journals, Replays, and Python-regenerated numeric inputs.
The smoke configuration, command, and results are in
`runs/pestle-mirage-compat/smoke`.
The audit SHA-256 is `08b18e73649d0af17d6ff2c1017c3023e0ba1202bba78dd02552a03438d65a72`.
The 256-game full convergence run was **NOT RUN**.
