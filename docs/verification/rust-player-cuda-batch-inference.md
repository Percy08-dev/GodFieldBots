# Rust player CUDA batch inference verification

## Inputs

Bots base `49c8cc10a2a935b49f4c427e2e1e536325a7b573`. Both lanes use the
same Python source and native runtime, with the inference recipe selected
in each trial's effective configuration. The Python runtime digest is
`4b0b59728288041b567576158419101cab012f68ca71fd263b1f0b62f3930f90`.
The native source SHA-256 is
`3bc9c92fe0436ab83cb7c09c3691312a380f4a89153b01064ac08388099bff8e`.
The trial environment uses Python 3.14.10 with the GIL, NumPy 2.5.4,
Torch 2.14.1+cu132 and TensorDict 0.15.2. The host is an RTX 5070 Ti with
16,303 MiB total VRAM and NVIDIA driver 610.88.

Simulator `072dabbca1458c9526897781d2fd64edeeb4f9ca` supplies API v2,
`true-match-1v1-v2`, ActionConfig 3 and the `public-ja-v3` schema 2 catalog.
External interfaces §21.1 SHA-256 is
`8e5223b45a6e8c4c80e272736ed97b9e271136c939eca74427a035ddfdbfcac2`.
The test server SHA-256 is
`76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d`.

## Matched comparison protocol

The ordinary Model3 path uses its public training command. Model4 uses the
existing supervised schema 6 HPO operation with a fixed proposal to keep
training and collection inputs paired. Each trial uses seed 17, a fresh
parent, 256 games, Tokio with 16 game slots and 16 workers, Rust cap 64,
formation wait 1 ms, and a 12 GiB managed CUDA allowance. The batched
recipe therefore has a leading width of 16. Three fresh trials per recipe
alternate scalar and batched for each model. Initial weight hashes are
compared across recipes and repetitions. Instrumentation is on for both
lanes and Torch profiling is off.

Collection time comes from the committed update. The native
`inference_roundtrip` histogram measures actor request-to-accepted-response
wall time. Report per-run and pooled nearest-rank p95 bounds at the
recorded histogram resolution. Actual forwards and useful-size counts come
from the Python service. Rust's formation histogram measures which requests
arrived together. Scoped Torch allocator peaks report live allocated storage
and reserved segments. Admission accounts for context and workspace through
separate configured charges. Cursor and Source checks run after the original
process exits and outside the collection timing.

## Functional and device verification

The preimplementation architecture review accepted the fixed-geometry
design. The independent correctness and conformance reviews cleared the
implementation after adding accounting for CUDA's expanded candidate indices.
An isolated width-16 CUDA probe measured 884,736 bytes of concurrent int64
index copies for noncontiguous references across chunks. The regression fixes
the remaining transient allowance at 6 MiB and verifies whole-shape fallback
before any batch copy.

The final focused CUDA command passed 24 cases in 6.88 s on the RTX 5070 Ti.
It checks Model3 and Model4 at the actual workload widths, including full
groups, zero events, large candidate tables and mixed checkpoints. Logs and
values match the scalar CUDA and frozen CPU references within `atol=rtol=1e-5`.
Changes to another request's inputs, order, position and batch occupancy
preserve the tested request's exact logs, value, sample and next RNG state.
The coverage includes seat and trial isolation, stale responses, an old
request completing across checkpoint publication and fencing a partial
transfer failure before releasing its allocations.

```sh
env CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  runs/cuda-batch-inference.kYXk2q/env/bin/python -m pytest -o addopts='' -q \
  tests/test_rust_player_batch_inference.py
```

The same source passed supervised CUDA save and resume through two updates
for each model. These controls use width four and 16 natural games per
collection. Model3 advanced from 16 to 28 Adam steps, and Model4 from 8 to
12. All four invocations passed `--full-source-audit` with complete Journal,
Replay, Python numeric and saved-pair checks. The run directory retains both
the initiating commands and post-exit verifier outputs.

```sh
runs/cuda-batch-inference.kYXk2q/env/bin/python \
  runs/cuda-batch-inference.kYXk2q/experiment.py resume-control
```

Both paused parents rejected a resume request with their recipe removed.
Their saved state bytes matched the pre-attempt state.

```sh
env CUDA_VISIBLE_DEVICES='' runs/cuda-batch-inference.kYXk2q/env/bin/python \
  runs/cuda-batch-inference.kYXk2q/check_recipe_resume.py
```

## CPU regressions and static checks

The CPU selection passed 483 cases with six conditional skips and 19
CUDA-name deselections. It includes all eight frozen PPO kernel identity
checks. Ruff passed on the changed Python files, mypy passed on the ten
changed source/tool files, and `git diff --check` passed.

```sh
env CUDA_VISIBLE_DEVICES='' runs/cuda-batch-inference.kYXk2q/env/bin/python \
  -m pytest -q -o addopts='' tests/test_rust_player_config.py \
  tests/test_rust_player_inference.py tests/test_rust_player_batch_inference.py \
  tests/test_rust_player_device.py tests/test_rust_player_diagnostics.py \
  tests/test_rust_player_verify_operation.py tests/test_hpo_config.py \
  tests/test_hpo_numeric.py::test_all_eight_frozen_kernels_remain_exact -k 'not cuda'
```

Five additional live Simulator lifecycle cases passed with CUDA hidden
on the host. The host rerun resolved the earlier sandbox socket failures.

```sh
env CUDA_VISIBLE_DEVICES='' runs/cuda-batch-inference.kYXk2q/env/bin/python \
  -m pytest -q -o addopts='' \
  tests/test_rust_player_runtime.py::test_game_requires_original_absolute_cleanup_deadline \
  tests/test_rust_player_runtime.py::test_pinned_simulator_two_games_four_participants
```

## Matched CUDA results

All 12 commands and post-exit checks passed. Each trial completed 256 natural
games. Every Model3 trial had 4,928 accepted decisions and 3,861 neural
responses. Every Model4 trial had 5,160 accepted decisions and 4,063 neural
responses. Initial weight arrays matched across all six trials within each
model. Every serving row was accounted for by the diagnostic totals.

| Model | Scalar median s | Batch median s | Change | Scalar pooled p95 ms | Batch pooled p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 66.75 | 61.65 | -7.6% | 243.27–247.46 | 218.10–222.30 |
| 4 | 71.87 | 66.32 | -7.7% | 268.44–276.82 | 243.27–247.46 |

The table below reports every measured trial. Useful and dummy rows count
actual batched forward slots. Allocated and reserved values are peaks from
the collection's Torch allocator.

| Model | Trial | Collection s | Response p95 ms | Batch forwards | Useful rows | Dummy rows | Scalar rows | Allocated MiB | Reserved MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | scalar-1 | 67.84 | 247.46–251.66 | 0 | 0 | 0 | 3,861 | 45.99 | 60.00 |
| 3 | batch-1 | 61.88 | 213.91–218.10 | 3,241 | 3,861 | 47,995 | 0 | 122.80 | 170.00 |
| 3 | scalar-2 | 66.75 | 247.46–251.66 | 0 | 0 | 0 | 3,861 | 45.99 | 60.00 |
| 3 | batch-2 | 61.65 | 218.10–222.30 | 3,269 | 3,861 | 48,443 | 0 | 122.80 | 170.00 |
| 3 | scalar-3 | 66.37 | 230.69–234.88 | 0 | 0 | 0 | 3,861 | 45.99 | 60.00 |
| 3 | batch-3 | 60.85 | 222.30–226.49 | 3,274 | 3,861 | 48,523 | 0 | 122.80 | 170.00 |
| 4 | scalar-1 | 72.23 | 276.82–285.21 | 0 | 0 | 0 | 4,063 | 51.70 | 60.00 |
| 4 | batch-1 | 66.78 | 255.85–260.05 | 3,463 | 4,063 | 51,345 | 0 | 129.88 | 144.00 |
| 4 | scalar-2 | 71.82 | 264.24–268.44 | 0 | 0 | 0 | 4,063 | 51.70 | 60.00 |
| 4 | batch-2 | 66.32 | 243.27–247.46 | 3,456 | 4,063 | 51,233 | 0 | 129.88 | 144.00 |
| 4 | scalar-3 | 71.87 | 268.44–276.82 | 0 | 0 | 0 | 4,063 | 51.70 | 60.00 |
| 4 | batch-3 | 66.20 | 239.08–243.27 | 3,423 | 4,063 | 50,705 | 0 | 129.88 | 144.00 |

The largest batch forwards used six useful rows in Model3 and five in
Model4. Singleton groups accounted for most forwards. All batched trials
fit their CUDA allowance with zero scalar fallback rows. Both models meet
the acceptance criteria on the measured workload. `batched-torch-v1` is
accepted for explicit selection, with scalar inference as the default.

The workspace retains the commands, per-run verifier outputs, scoped
telemetry, and `summary.json` under `runs/cuda-batch-inference.kYXk2q/`.

```sh
runs/cuda-batch-inference.kYXk2q/env/bin/python \
  runs/cuda-batch-inference.kYXk2q/experiment.py comparison
```

## 128 MiB for CUDA batch temporaries

The widened planner passed 36 focused RTX 5070 Ti tests with the tracked
Python source and the paired installed native extension. Both models matched
scalar CPU and CUDA references at `atol=rtol=1e-5` in width-32 groups above
the old ceiling. Neighbor changes and position changes preserved the exact
own logs, value, draw and next RNG state. Lower per-tensor and existing
aggregate allowance controls passed. The worker comparison's 256-game
ordinary and HPO trials also passed the standard operation verifier after
the change. These trials use the recorded full-wave launch and shape floors.

See the [follow-up plan](../superpowers/plans/2026-11-21-rust-cuda-inference-performance.md)
for the comparison conditions and the subsequent Graph experiment.

## Product CUDA Graph and Tokio defaults

`batched-torch-graph-v1` captures the tested `(32,64,16)` learner forward
and preserves the original Rust dispatch and full response. The configuration
tests cover the default of 16 workers for fresh Tokio declarations, explicit
worker counts, and materialized worker counts in saved effective ordinary
and HPO declarations. All 31 archived ordinary and HPO effective configs
used by the measurements still loaded. Graph accounting tests cover the
actual capture pool, capacity decline, and reserve headroom for the current
checkpoint's scalar fallbacks as well as its admitted Eager groups.

Both ordinary PPO and HPO completed two real 32-game updates through the
normal entry and the source-matched native extension. Each operation
produced three distinct checkpoint manifests and reported positive Graph
replay counts for both collection generations. The config omitted the
worker count and the saved effective runtime materialized 16. The standard
verifier validated sealed collection originals and saved update artifacts.
The focused live CUDA cases compared both models' results and request-local
RNG, checked the state committed by Rust on the next request, and observed
old private pool retirement. A replay failure test held the Graph until
the original consumer and CUDA work were settled.

The full-concurrency performance comparisons use the previously recorded
full-wave Simulator launch overlay and the paired RTX 5070 Ti environment.
The same Torch geometry and checkpoint flow are shared by both lanes.
The Eager control sets the Graph support decision to false in its disposable
Python source copy. The production comparison and diagnostic overlays use
different native wheels. The production wheel's native source digest
matches the tracked native source. The timing wheel adds observation at
request admission, queue readiness and dequeue.

The 256-game checks observed 51.46 → 50.39 seconds for Model3 and
56.31 → 55.41 seconds for Model4. Each condition ran once. All four
operations passed the standard verifier and reached 256 simultaneously
active games. The paired journals had identical submitted action payloads
and FP32 log probabilities and values. These performance copies preceded
the last two guards for owning-checkpoint scalar admission and exact saved
ordinary config validation. The final source passed two new ordinary updates
and two new HPO updates with both guards in place. The replays were 191/132
and 208/122 by collection generation. The actual second-shape experiment
also passed the standard verifier and matched the one-shape journal.

The final scoped CPU command passed 390 tests with 6 skips and 26 CUDA
tests deselected. The complete batch inference test file passed 47 tests
with CUDA enabled. Ruff, scoped mypy across 11 source files, and
`git diff --check` passed. The normal-entry final run records are under
`graph-collection.MMtWST/product-check/refresh-final/`, within the run tree
linked from the [follow-up plan](../superpowers/plans/2026-11-21-rust-cuda-inference-performance.md).
That plan records the queue timings, Graph shape experiment and trade-offs.

## Prepared CPU sampling from the grouped CUDA path

The production grouped path now passes the completed CPU FP32 probability
slice directly to Torch multinomial. Torch enforces the weight contract.
The scalar route keeps its conversion and validation. Each draw creates and
seeds or restores the same participant-local generator. Rust commits only
the accepted original full batch and keeps the state bytes opaque.

The focused tests compare sample indices and generator bytes through repeated
restore, verify invalid weights and runtime/state errors, and check global RNG
isolation. The combined inference and batch inference files passed 108 tests
with CUDA enabled on the final Python source. The runtime and wheel were
unchanged.

The [performance record](../superpowers/plans/2026-11-21-rust-cuda-inference-performance.md#cpu抽選とeager形状別の比較)
contains the two alternating 256-game comparisons of the grouped CPU
optimization, the Tokio sampler prototype and the existing route. Across
the paired trial journals, the acknowledged actions and selected FP32
log probabilities and values matched. The isolated Rust sampler matched all
7,924 captured output rows and their 5,056-byte successor RNG states. The
deferred dispatcher prototype additionally matched every one of 1,769 states
re-exported by actual actors during two successive 32-game updates of both
ordinary PPO and HPO. The Rust sampler prototype depended on a private Torch
state format and used an experimental settlement boundary. The production
route keeps the existing Torch sampler and checkpoint lifecycle.

The final-source A/B used the source-matched native wheel and confirmed
256 simultaneous games in all four trials. Model3 collection was 49.50 →
49.06 seconds and Model4 was 54.20 → 53.13 seconds. Actions and selected
FP32 scores matched across the final pairs. Model4's response p95 and Torch
peak reservation increased in that pair. Both final-source routes then
completed two successive 32-game updates, with three distinct checkpoints,
positive Graph replay in each collection, and standard validation of
sealed originals and saved update artifacts. The results are retained in
`runs/cuda-sampling.vtoWD3/`.
