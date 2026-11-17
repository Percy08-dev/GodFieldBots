# HPO configurable game concurrency verification

Date: 2026-11-18. Baseline: `b61a603`.
Status: implemented and accepted; independent correctness and conformance READY.

## Scope and authority

The owner requests more than two simultaneous HPO games and accepts tuning
configuration when increased concurrency causes OOM. [ADR0049](../adr/0049-configurable-hpo-game-concurrency.md)
and [the design](../design/hpo-configurable-game-concurrency.md) retain managed
memory admission, single-job GPU ownership, one trial at a time, source/config
identity and positive process/I/O retirement. Only HPO count domains expand;
default2games/4workers, ordinary/frozen limits, mixed16games and storage remain.

Architecture preimplementation review initially BLOCKED on omitted custody and
Python/Rust retirement caps. Root adopted the finding and propagated expected
capacity from the original ledger/configuration, not a receipt's own claim.
Independent re-review is DESIGN READY; implementation/final reviews are separate.

## Executed checks

- TDD RED (implementer):
  `UV_CACHE_DIR=/tmp/gfb-hpo-uv-cache uv run --no-sync pytest -q tests/test_hpo_game_concurrency.py`
  gives6FAIL/7PASS in1.31s against old code. Rejections are the old fixed caps
  and absent explicit retirement capacity. This is expected failing-test evidence,
  not acceptance of the changed implementation.
- Root Python standard-library TOML assertions: all3budget-probe presets parse,
  explicit defaults equal2/4, and originals remain
  `/mnt/d/godfield/ppo-budget-probe-v2/<model>/d10000/originals`: PASS.
- Root local Markdown link checks for the new design/ADR/plan: PASS.

Python commands below use `UV_CACHE_DIR=/tmp/gfb-hpo-uv-cache uv run --no-sync`.

| Command suffix | Result | Scope |
| --- | --- | --- |
| `pytest -q tests/test_hpo_game_concurrency.py` | PASS23,1.34s | Final4/8 and8/16 materialization/lineage/admission, receipts, owner/failure boundaries |
| `pytest -q tests/test_hpo_game_concurrency.py tests/test_resident_evaluation.py tests/test_hpo_gpu_lease.py` | PASS76,7.23s | Earlier new-module revision; actual4-game overlap/eight CPU PIDs and corrected HPO worker-cap expectation |
| `pytest -q tests/test_hpo_game_concurrency.py tests/test_resident_pool.py tests/test_hpo_bootstrap_validation.py tests/test_hpo_eval_failure_cleanup.py` | PASS102,43.02s | Before final8/16 parametrization; all other files/source unchanged afterwards |
| `pytest -q tests/test_hpo_receive_custody.py` | PASS11,10.78s | Approved host rerun resolves inherited Unix-socket sandbox EPERM |
| `mypy src` | PASS232source files | Source unchanged after this check |
| `ruff check src tests/test_hpo_game_concurrency.py tests/test_hpo_gpu_lease.py tests/test_resident_evaluation.py` | PASS | Final changed source/tests |
| `git diff --check` | PASS | No whitespace errors |

The first broad selection was:

```sh
UV_CACHE_DIR=/tmp/gfb-hpo-uv-cache uv run --no-sync pytest -q \
  tests/test_hpo_config.py tests/test_hpo_phase_limits.py \
  tests/test_hpo_lineage.py tests/test_hpo_gpu_lease.py \
  tests/test_hpo_resident_custody.py tests/test_hpo_resident_contracts.py \
  tests/test_hpo_evaluation_role.py tests/test_hpo_eval_collector.py \
  tests/test_hpo_admission_release.py tests/test_hpo_receive_custody.py
```

It had440PASS/5FAIL in26.74s. One test still expected HPO worker capacity5 to be
invalid; its corrected signed63 overflow assertion passes in the76-case rerun.
The other4failures were sandbox `PermissionError` at inherited Unix sockets;
all11receive-custody tests pass with the approved host rerun. These are staged
selection-plus-delta results, not a new green invocation of the full445selection
or the whole project suite. Final23covers the last test-only8/16 additions.

Full offline Rust library checks pass177/13ignored/0FAIL in3.57s:

```sh
env RUSTFLAGS='-L /home/percy/.local/share/uv/python/cpython-3.14.10-linux-x86_64-gnu/lib -C link-arg=-lpython3.14' \
  LD_LIBRARY_PATH=/home/percy/.local/share/uv/python/cpython-3.14.10-linux-x86_64-gnu/lib \
  PYO3_PYTHON=/home/percy/develop/GodFieldBots/.venv/bin/python \
  cargo test --offline --locked --manifest-path native/Cargo.toml --lib
```

New native cases cover expanded spec/run projection, eight-owner retirement and
original-parent capacity versus manifest substitution. Ignored external/heavy
checks remain NOT RUN; the native manifest case intentionally reaches the next
independent Source check without manufacturing complete original game evidence.

Actual CLI validation passes for every current preset:

```sh
.venv/bin/gf-bots hpo validate --config configs/hpo/budget-probe-mlp-d10000.toml
.venv/bin/gf-bots hpo validate --config configs/hpo/budget-probe-gru-d10000.toml
.venv/bin/gf-bots hpo validate --config configs/hpo/budget-probe-transformer-d10000.toml
```

Temporary configs under `/tmp/gfb-hpo-concurrency-configs-f_7hms79` resolve the
same base/storage paths and set4games/8workers with16GiB managed CUDA; the same
CLI validate command also passes for all3. Both proposal and canonical control
report8workers. Raw results are `/tmp/gfb-hpo-concurrency-<model>-validate.json`
and `/tmp/gfb-hpo-concurrency-<model>-4-validate.json`.

The default12GiB CUDA estimate correctly refuses4/8. A pure calculation with
large temporary management budgets, without allocation, gives these current
proposal estimates; the16GiB example passes, while physical fit remains untested:

| Model | Managed RAM estimate GiB | Managed CUDA estimate GiB |
| --- | ---: | ---: |
| MLP | 18.342 | 14.842 |
| GRU | 18.424 | 14.924 |
| Transformer | 18.454 | 14.954 |

## Native activation

Before activation, root copied the old installed ELF and its exact identity to
`/tmp/gfb-hpo-concurrency-native-backup-ceexjk9s`. Old source digest:
`5f3b3d441da3ab5d503dc64b2ba93f13d35fa4126ebe75e31d246b8e368abd0d`;
old extension SHA256:
`78830a048eb709829d817b7b40a4f068a8c6954fc776b9e907b8006e4a594a5c`.
Rebuild/install and current source-digest comparison PASS:

```sh
env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS CARGO_NET_OFFLINE=true \
  /home/percy/.cache/uv/archive-v0/ZagxeqeShGVnUsQq/bin/maturin build \
  --release --locked --manifest-path native/Cargo.toml \
  --features pyo3/extension-module,allocator-mimalloc \
  --interpreter .venv/bin/python --out /tmp/gfb-hpo-concurrency-wheel
UV_CACHE_DIR=/tmp/gfb-hpo-uv-cache uv pip install --offline --no-deps --reinstall \
  --python .venv/bin/python \
  /tmp/gfb-hpo-concurrency-wheel/godfieldbots_native-0.1.0-cp314-cp314-manylinux_2_34_x86_64.whl
```

Release compilation takes9.47s and retains existing warnings in unchanged code.
Root independently hashes57Rust/build/asset inputs using build.rs's source
inventory and compares to the actual loaded extension: PASS.

- Current native source: `2f94bc85247b193ff4b37afa43b77616a32e73ef7825f087120a8998982484e4`.
- Current ELF: `b24f35f5a2aefa70114beb36c4ea33a0d4617b35c00ba654b137eef626b5e533`.
- Release, mimalloc, empty RUSTFLAGS, cp314-gil and GIL enabled: PASS.
- Actual `hpo_reader_settings(..., packed=True, resident=True)` for all3preset
  base readers, plus no Torch/Optuna import: PASS.

Identity checks are not bypassed; installation replaces only the native0.1.0
wheel with `--no-deps`. Full identity is saved in
`/tmp/gfb-hpo-concurrency-native-verified.json`.

## Limits and acceptance

Real HPO, Simulator games, CUDA capacity/OOM experiments and throughput
measurement: NOT RUN, by the owner's manual-execution preference. CPU/fake
concurrency and receipt tests establish functionality, not a safe machine-wide
count, physical memory guarantee or learning-quality improvement. No new whole
pytest invocation or package-wide performance result is claimed.

Correctness and architecture-conformance final reviews are READY without
blockers. Correctness's optional8/16 coverage note is resolved by the final
test-only extension and narrow READY re-review. Root accepts the functional
scope. No old parent, dataset, checkpoint or failed objective is rewritten.
Following the owner's feedback about excessive checking, no further redundant
tests/reviews are added after this evidence; only documentation/local commit
remain. This is not an automatic recurring verification policy.
