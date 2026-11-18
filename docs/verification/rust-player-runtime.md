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

## Native v5 policy core

Accepted bounded core after independent correctness/spec-quality and architecture
conformance READY, including one scoped repair/re-review round. New modules
`native/src/player_runtime/` implement public rules, bounded v5 search, CPython
used RNG APIs, Random/Heuristic and participant-local Refresh. Whole-call
GIL-detached diagnostics are `player_search_v5`, `player_select_v5` and
`player_rng_v5`; later actors must use the typed core and incremental bank
custody, rather than repeatedly rebuild full-prefix diagnostic JSON.

Original broad comparison:679PythonPASS, including346new collected tests;
237owned perceived definitions across four modes,40mixed seeded searches,
65,536rows/18cards, resource/allocation/sequence rules and Refresh histories.
This broad run preceded the two final fixes. Independent review found deadline
truncation could return an empty table, and baseline capacity18 was not enforced
separately from search capacity128. Both are corrected with meaningful RED→GREEN
coverage; search capacity is preserved and no candidate is synthesized.

Final repair evidence:11coveringPythonPASS,245affectedPythonPASS,3focusedRustPASS.
Root's full native suite after fixes:180PASS/13IGNORE in3.30s; root fresh11covering
checks PASS and loaded release/lock hashes verified. Both independent re-reviews
are READY with bothP2findings ADDRESSED and no remaining findings. The unchanged
broad679selection was not rerun after these two local fixes; no final whole-Python
green claim follows. Source-matched fixed release native SHA256:
`9bfc13beae4c1d308ae0267c0a9d0815fb08d0d74e5f42f193901d28938a536a`;
Cargo.lock SHA256:
`bb8e9e3cae6eab6c2877016c18d58256ea46f5f5807f8b41a6d475642bae692d`.
Extension RUSTFLAGS are empty. Saved reference fixture hash remains unchanged.

This accepts the policy core only. Live HTTP/state ingestion, Tensor ownership,
sampling retirement, complete runtime isolation, training/resume and measured
speed/memory remain required work. Diagnostic Refresh state is replay data,
not a new admitted checkpoint format or Source proof. Existing Python production
and historical readers/goldens are unchanged.

## Owned inference boundary

The bounded array-serving/dispatcher stage passes independent correctness and
architecture conformance READY, including one scoped repair/re-review round.
`ServingRow` reuses existing corrected native token kernels and transfers final
float32/int64/bool allocations into NumPy; torch.from_numpy aliases CPU storage
and last-reference tests preserve its lifetime. One shared Dispatcher supports
blocking adapters and direct Tokio admission/result waits without Python callbacks
or numeric JSON in the live path. Formation starts at first valid admission's
absolute timestamp, including aged queues and first-row retirement.

Rust owns committed opaque participant Torch bytes and atomically accepts only
matching valid request/sample/next-state tuples. Invalid/expired/retired results
adopt no RNG. Exported canceled leases retain quota until original service
settlement; logical queue close does not prove Tensor/GPU completion. Game actors
must bind actual Simulator gameID separately from planned frame.episode, retain
active/frozen checkpoints before descriptor admission, release after actual leases
settle, and use the same Torch serialization lock for learner operations. Registry
publication additionally restores CPU and already-initialized CUDA RNG; learning
retains its intentional RNG changes under the lock.

Original component evidence:8RustPASS,49inferencePASS including real CUDA,
470broaderPythonPASS/18existing explicit opt-in skips. Review found retirement
missed capacity-blocked/unseen admissions, and unsupported finite timers could
panic/poison cleanup. Both corrected with RED→GREEN tests: pre-capacity epoch
ownership/tombstones, drop-safe started-producer registration, and checked actual
Duration/host-clock representation with no arbitrary short cap.

Final repair evidence:12focusedRustPASS,55inferencePASS/0SKIP including real CUDA;
root fresh6coveringPASS and loaded native source/lock match. Root full native after
fixes:189PASS/13IGNORE in3.24s. Both independent re-reviews mark bothP2findings
ADDRESSED, no remaining findings. The unchanged broader470selection was not
rerun after this local repair; its result is historical, not a final whole suite.
Fixed native release SHA256:
`e62691d7fa8fa4eb61cfe6b4b9558fc62ad64c4260792d629c560b066b25f999`;
Cargo.lock remains `bb8e9e3cae6eab6c2877016c18d58256ea46f5f5807f8b41a6d475642bae692d`.
Extension flags empty; Python adapter Ruff/format/mypy pass.

Six-row CPU/CUDA mechanism timings separate assembly, sharing, synchronous
transfer, fenced forward and return. They include cold first forward and belong
to original pre-fix sourceaa9e5a74, not final corrected source or an end-to-end
speed comparison. Full actors/storage/learning/process-tree/audits and approved
physical matrix remain required before any complete-runtime/adoption claim.

## Companion/descendant feasibility

Independent preimplementation architecture supplement READY for concrete installed
wheel paths and Linux subreaper/isolated-group descendant retirement. A disposable
Maturin1.15probe PASS confirms `gf_native_runtime/gf-player-supervisor` and manifest
paths in Distribution.files plus executable0755. It is not the actual companion
wheel. Current ManagedServer inherits its host group and pinned Simulator starts
no OS child processes; stdinEOF requests shutdown and does not prove exit. Actual
original-host/adopted-child waits, failure injection, installed companion identity
and host reentry still require implementation/verification.

## Required evidence still outstanding

- Integrate the accepted typed policy core with live incremental own-view state.
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
