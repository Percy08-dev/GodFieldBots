# Dream Simulator contract verification

## Release identity

| Item | Value |
| --- | --- |
| Simulator source | `9e47d4c5f8cd3f60437f9aeabc95076c1dc04321` |
| Spec inventory SHA-256 | `afefd6f3b9113b813c90aaae7bb7b18f1e6299e52bf814ef34ec22112362ab5e` |
| Changed file inventory SHA-256 | `eb56add45d4b4e89af3baf44ec25b0a909886229ed0dffe6d41e20b9c0019951` |
| TestServer SHA-256 | `f95fadbc8eaad23fbe63e17ac9bdb0ebeca4b237956dedeab09d86f30cfd94bc` |
| Action profile | `card-instance-simulator-v2-shared-20261124` |

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
