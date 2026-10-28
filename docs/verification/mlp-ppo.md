# MLP PPO implementation and experiment evidence

Date: 2026-10-28. Requirements and accepted design:
[ADR0008](../adr/0008-mlp-ppo-generations-and-fixed-data.md).
Full implementation, final independent review, and real learning acceptance are
in progress. No trained-policy quality or final-test claim is made.

## Stage A: optional CPU/CUDA dependencies

Independent architecture design and dependency conformance: READY.
Independent correctness review of dependency metadata/lock: READY, including
the final recheck after restoring nine general PyPI packages to their original
HEAD versions (avoiding CUDA-install carryover downgrades).
Baseline dependencies are httpx and NumPy 2.5.4. Mutually exclusive `torch`
(CPU) and `torch-cuda` extras select exact Torch 2.14.1 and torchvision 0.29.1
from explicit official indexes. Existing cu132 installation is preserved.
Configuration follows the
[official uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
Precision settings follow
[PyTorch 2.14 CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html).

Before alignment, targeted native/encoder tests reproduced the runtime pin
failure: 3 failed, 14 passed, 16 errors with NumPy 2.5.2. After sync,
`tests/test_lightgbm_native.py tests/test_lightgbm_encoder.py tests/test_runtime.py`
passed: 49 tests, 7.48s (independent verifier).

Commands used explicit Bots project environment or `env -u VIRTUAL_ENV` and
`UV_CACHE_DIR=/tmp/gfb-uv-cache` after the environment incident below:

```console
uv lock
uv sync --locked --extra torch-cuda --extra lightgbm --extra data
.venv/bin/python -m pytest -q tests/test_lightgbm_native.py tests/test_lightgbm_encoder.py tests/test_runtime.py
```

The root's hardware smoke used an approved GPU-capable command, a minimal
environment (PATH/locale/one-thread BLAS and `CUBLAS_WORKSPACE_CONFIG=:4096:8`),
and a fresh Python child. Torch 2.14.1+cu132, NumPy 2.5.4, CUDA 13.2,
RTX 5070 Ti, deterministic algorithms and IEEE FP32 passed exact CPU/CUDA
comparison of an 8x8 FP32 matrix product after synchronization. Peak allocated
33,555,456 bytes; reserved 35,651,584 bytes. This is dependency/device evidence,
not full-batch PPO parity or learning acceptance. GPU access is blocked in the
normal sandbox; GPU checks use approved escalated execution.

Independent verifier reports isolated installed-wheel baseline and CPU-extra
help/import checks passed in `/tmp/gfb-20261028-stageA-baseline` and
`/tmp/gfb-20261028-stageA-cpu`. CPU identities: Torch 2.14.1+cpu,
torchvision 0.29.1+cpu, NumPy 2.5.4. Both extras together are rejected.
Interim Stage-A wheel SHA-256:
`c58043c545a21806609ffb7af8f90d2403f6ce755fe8b4fac82858a12a3ceb5a`.
The final implementation requires another installed-wheel check.

## Environment incident

The dependency verifier mistakenly used `uv sync --active` with an inherited
`VIRTUAL_ENV=/home/percy/develop/GodFieldSimurator/.venv`. Two commands modified
that separate Python environment: simulator verifier/dev packages were removed,
baseline packages changed, and CPU Torch/torchvision were installed. Simulator
source was not changed. Root stopped further writes there and requested exact
delta evidence; restoring that environment requires a separately authorized
repair. Bots `.venv` remains the intended cu132 build. Subsequent isolated
checks explicitly unset VIRTUAL_ENV and use unique absolute `/tmp` targets.

Recorded incident delta: removed annotated-types 0.8.0, certifi 2026.7.22,
gf-current-client-verifier 0.1.0, idna 3.20, iniconfig 2.3.0, NumPy 2.5.4,
packaging 26.3, pluggy 1.6.0, pydantic 2.13.5, pydantic-core 2.46.5,
pytest 8.4.2, pytest-asyncio 1.4.0, pyyaml 6.0.3, quickjs-ng 0.17.0.1,
respx 0.23.1 and typing-inspection 0.4.4; reinstalled NumPy 2.5.4,
certifi 2022.12.7 and idna 3.4. The second command added filelock 3.32.3,
fsspec 2026.7.0, jinja2 3.1.6, markupsafe 3.0.3, mpmath 1.3.0,
networkx 3.6.1, pillow 12.3.0, setuptools 78.1.0, sympy 1.14.0,
Torch 2.14.1+cpu and torchvision 0.29.1+cpu. The original exhaustive removal
list contains neither Torch distribution. No restoration has been performed;
the user has been asked for separate authorization because AGENTS.md excludes
Simulator modification from this Bots task.

## Preregistered experiment conditions

After implementation acceptance and a successful real two-update smoke, run
learning seeds 17, 29 and 43 sequentially with `configs/experiments/ppo.toml`.
Each has original 7200s wall, 5400s admission, 7140s execution, 1000-update cap,
10 GiB soft storage and 1 GiB admission reserve. Concurrency is globally two
games. Preserve whole failed runs, with no offline reuse or replacement games.

Freeze all learner train trajectories from all three complete parents without
outcome selection. Each final policy collects 32 paired validation seeds (the
existing 32), pairs 0–15 versus Random, 16–31 versus heuristic, 64 games per
policy and 192 total. Both views are saved/revalidated; only learner-viewer
accepted actions and final natural returns are fit labels. No final test runs.

The fixed dataset is shared by observation-only and public-history LightGBM,
each at fitting seeds 17,29,43. Dataset generation cost is deduplicated and
reported separately; each fit's own validation/encoding/fit/save/cleanup has a
7200s budget. This is not an equal-total-cost PPO comparison. Model strength
and speed are measured results, not implementation acceptance criteria.
