# Strength Powder contract verification

## Release identity

| Item | Value |
| --- | --- |
| Simulator commit | `3172da407e37a4ad57ac82e70a96910c3ac6fbb8` |
| Reviewed spec inventory SHA-256 | `6403eef771b64c34e78eb653c2baefe65ceca5308051b506a7092151e9316c59` |
| Simulator gameplay spec 09 SHA-256 | `b6cf7e55c078df3eb893312e30215e58127117792ebce1cc0ce250b51fcbaefa` |
| Simulator change inventory SHA-256 | `fca9be8a30ae4969047f1136211acba2ee427a0f15df35aa8212f17afa36bf81` |
| TestServer SHA-256 | `45ac711b2f172dfd89c23edf30ba0045ab0704c0215a6995d4e8626af1cfb923` |
| Action contract | `card-instance-simulator-v2-shared-20261123` |
| Candidate rule signature | `5adf1550f7e4bd28b3e69b752695c46971cafa9c49a7adb1a93d285d3adc4d3f` |
| Native wheel SHA-256 | `443fb7e1bdde9790f05ce5d4379a3aeb668fb9702484f09b9d722c9499f4a2db` |
| Native source inventory SHA-256 | `082a9b6289c1f91dc532d1bf896d3c2d59c51d776010abcdf4fbbcf45b5b3804` |

The separate local Simulator checkout and TestServer are under
`runs/strength-powder/simulator`. The Native wheel is under
`runs/strength-powder/wheels` and contains Maturin's extension and the
matched supervisor. It was built from the working tree. The build included
the owner's pending HPO checkpoint-evaluation changes in that tree.

## Rules and compatibility checks

- **PASS** Simulator `cargo fmt --all -- --check` and
  `cargo test --offline --locked --features test-support --no-fail-fast`.
  507 tests passed and the 10,000-seed nightly soak was skipped. All 96
  games in the PR corpus ended naturally. The two fixed Dream outcomes
  and affected corpus entries use the newly verified candidate pool.
- **PASS** Simulator `cargo build --offline --locked --release --features test-support --bin gf-test-server`
  and application preflight against the clean pinned source and release binary.
- **PASS** Python and Native controlled candidate tests cover powder alone
  or with a second powder, attack miracles, a suppressed CostCut Staff,
  attack-capable armor, normal weapon ordering, additional weapons and
  extendable prefixes. The Simulator test covers Dream acquisition and RNG.
  A separate public-input fixture covers the visible Dream wand's Discard
  exclusion and equal completed candidate tables.
- **PASS** The new Native wheel reproduced the 39 accepted p11 actions and
  the recorded HTTP 422 with the historical Simulator profile and binary.
  `python -m pytest -q -n0 tests/test_hpo_dream_discard_reproduction.py`
  used the pinned runtime variables in the [p11 record](../issues/0013-hpo-dream-discard-hidden-weapon.md).
- **PASS** Negative cases reject mixed Action profiles before native
  Journal ACK and when reading standalone Journals or the cold Source.
  The final focused suite passed 501 Python tests with two skips and 40
  Native player-runtime tests. Ruff and `python -m mypy src` passed in
  the matched build environment.
  Independent Bots correctness and architecture reviews cleared the change.
  The Simulator had an independent cross-repository review of the contract
  and RNG. The Simulator-specific `spec_guardian` role was unavailable
  in the Bots session.

## Fresh HPO6 acceptance

Run `216a370b-1d92-4cce-8f05-0c86d7a56fff` used seed 17, Model 4,
eight concurrent games, the Tokio scheduler, graph inference, the RTX 5070 Ti
and CUDA Torch 2.14.1+cu132 on Python 3.14.10. The runtime staged collection
on tmpfs and published the closed SQLite Pack on the designated HDD ext4
mount `/mnt/gf_data`. It recorded eight natural outcomes and 149 accepted
decisions across both players. The learner used 77 decisions and committed
one PPO update containing eight optimizer steps, then paused as requested.
The supervised command exited 0 and issued the cold completion and audit.

| Observation | Value |
| --- | --- |
| Collection wall time | 3.97 s |
| Collection throughput | 2.02 games/s |
| PPO update wall time | 1.81 s |
| Whole observed command | 27.40 s |
| Independent full Source audit | 2.11 s |
| Durable Pack bytes | 643,072 |
| Durable Pack SHA-256 | `5fe653eaeb7a95f28cb51f0a3f8f086aa803c240c240c9bb6ea5ca30f1f493a0` |

`tools/rust_player_operation.py --mode fresh --learning-seed 17 --pause-after-update 1`
used the frozen `runs/strength-powder/acceptance/hpo.toml`. The independent
`tools/rust_player_verify_operation.py --lane native-hpo --full-source-audit`
verified the original Pack, Journals, Replay transitions, Python numerical
comparison and full saved inference and resume pair. The observer receipt and
audit JSON are under `runs/strength-powder/acceptance/final`. The complete
originals are under `/mnt/gf_data/ppo-hpo/strength-powder-acceptance`.

One earlier invocation reached a saved update but stopped during cold
finalization after Python source changed during collection. The original
records remain separate. A fresh diagnostic retry passed, and the final
invocation above ran after the reviewed source settled.

The timings describe a bounded functional run. The 256- and 1,024-game
comparison, performance delta and learning-quality acceptance remain separate
from these results.
