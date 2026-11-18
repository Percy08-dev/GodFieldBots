# Rust player runtime verification

Status: implementation in progress; no runtime, performance, adoption or learning
acceptance claimed. Branch feat/rust-player-runtime starts at9680f06. Approved
[owner protocol](../design/rust-player-runtime-approved-plan.md) and proposed
[design](../design/rust-player-runtime.md) govern remaining checks.

## Baseline environment and focused checks

2026-11-19, isolated managed worktree
`/home/percy/.codex/worktrees/37cb/GodFieldBots`. `uv sync --frozen --extra torch-cuda
--extra shared-inference --extra native --extra hpo --extra data --extra lightgbm`
PASS, source-matched release native built. Python3.14.10 GIL enabled,
Torch2.14.1+cu132, NumPy2.5.4, TorchRL0.15.0, TensorDict0.15.2.
Scoped escalated GPU probe PASS: RTX5070Ti,16303MiB,driver610.88,
Torch CUDA available with one device. A sandbox-denied probe is not physical
GPU unavailability. Simulator release exists at
`/tmp/gfb-shared-simulator-072dabb/target/release/gf-test-server`; verify its full
identity again before actual game experiments.

Baseline command (before runtime implementation):

```sh
timeout 300s .venv/bin/python -m pytest -q tests/test_card_instance_actions.py tests/test_defense_priority.py tests/test_refresh_heuristic.py tests/test_shared_inference_numeric.py tests/test_shared_reply_retirement.py tests/test_hpo_owned_capture.py tests/test_shared_gpu_termination.py tests/test_shared_normal_run.py
```

PASS:200passed,7skipped,7.32s,exit0 (independent verifier). Raw output retained at
`/tmp/gfb-rust-player-baseline.log`. This is a focused baseline, not a whole-suite
claim. ADR0053's recorded whole-suite result remains historical51FAIL/7158PASS/
574SKIP with documented triage; it is not silently converted to whole-green.

## Native dependency foundation

Reviewed pins Tokio1.53.1 and reqwest0.13.5 resolve without upgrading existing
locked packages; reqwest has default features disabled and JSON enabled.
`cargo check --manifest-path native/Cargo.toml` PASS. Native tests initially fail
to link `_Py_Dealloc` with plain cargo test because extension-module builds do
not add the interpreter library. The existing project's documented test recipe
resolves this without a code change:

```sh
env RUSTFLAGS='-L /home/percy/.local/share/uv/python/cpython-3.14.10-linux-x86_64-gnu/lib -C link-arg=-lpython3.14' LD_LIBRARY_PATH=/home/percy/.local/share/uv/python/cpython-3.14.10-linux-x86_64-gnu/lib PYO3_PYTHON=/home/percy/.codex/worktrees/37cb/GodFieldBots/.venv/bin/python cargo test --offline --locked --manifest-path native/Cargo.toml --lib --quiet
```

PASS:177passed,13ignored,3.11s test execution. Extension-wheel builds must unset
these test-only linker flags. This verifies the dependency foundation, not a
player runtime or installed-companion acceptance.

## Current-v5 reference originals

PASS: nine own-view episodes/twelve decisions, exact fixed-clock reports and
34,285 clock reads, ordered tables/actions/hashes, CPython search/baseline RNG
and sequential/repeated Refresh behavior. Largest capture has2048candidates and
12visible cards. Fixture SHA256:
`ecf073cdb848e97752711e4f9b0a39f6f5d1b2206206a7acb293d25047d080f3`.
All47source hashes match baseline9680f06. Explicit recapture is byte-identical;
implicit overwrite refuses. Scoped regression333PASS and Ruff check/format PASS.
Independent `/root/v5_reference_review`: spec READY, task quality READY; optional
README public-cause wording corrected. This supplies comparison originals only.
Full18-card/maximum-row extremes, every trade/allocation family and broad RNG API
boundaries need separate port tests. See [fixture scope](../../tests/fixtures/rust_player_v5/README.md).

## Required evidence still outstanding

- Exact Rust candidate/report/order/hash, selections and RNG state compatibility.
- Logical participant independence, transactional RNG adoption, expired replies,
  immutable checkpoint publication and final-dtype buffer lifetime.
- Both actual schedulers/common HTTP contract, W1/W2 four-participant failure tests.
- Single Python process, bounded writer custody and original-handle supervisor/
  same-namespace GPU admission termination, source/producer version checks.
- Model3/4 actual CUDA continuous2 and pause1/resume2, same-input restore and
  independent full Original/Replay/numeric audits.
- Exact approved C8 W sweep, C16 forward/reverse and per-entrance C4/C2 two-pair
  comparison; no adoption decision without all predeclared evidence.
- Independent final correctness and architecture conformance, root disposition.
