# PPO learning update optimization

Date: 2026-11-17. Approved scope: experimental application, comparison and a
recommendation under [ADR0045](../adr/0045-learning-update-optimization.md) and
[the reviewed design](../design/learning-update-optimization.md). Normal defaults
remain unchanged. This record distinguishes design approval, code verification
and measurements; no learning-quality, HPO or final-test acceptance is implied.

## Current acceptance

Preimplementation architecture review: READY after explicitly accounting for
all concurrent batched-prefix/event activations in the joint estimate. Design
milestone: `89d50e3`. Both independent implementation reviews are READY after
the scoped fixes. Physical comparisons and final experiment acceptance remain
pending.

## Baseline and environment

Initial source: `13647484e5d40182a6fe461dc6bbeab0e258aef5` on the existing
`feat/heuristic-refresh` branch. Pre-existing untracked analysis/session files
and fixture `.lock` are outside this change. The simulator is read-only and its
queried HEAD is `91d5926a7572db327eaea3751e7c24f428120996`.

PASS before implementation:

```text
UV_CACHE_DIR=/tmp/gfb-hpo-baseline-uv-cache uv run --no-sync pytest -q tests/test_hpo_numeric.py tests/test_hpo_network.py tests/test_hpo_owned_numeric.py
92 passed in 7.78s
```

The first attempt using uv's default cache failed before pytest because that
cache is read-only in the sandbox; the retry above changes only cache location.
Python3.14.10/GIL, NumPy2.5.4 and Torch2.14.1+cu132 are present. CUDA/NVML are
unavailable inside the sandbox. The host read-only query through the approved
execution route succeeds: RTX5070Ti16,303MiB, driver610.88,2,521MiB used and9%
utilization at that instant. This is no reservation or exclusive-device claim;
each physical population must record fresh conditions.

## Fixed input and state contract

The [preparation record](learning-update-optimization-preparation.md) and
`runs/learning-update-preparation-20261117/input-state-inventory.json` pin the
three round1 baseline parents and all six numeric6 inputs, inference5/resume5
states and file hashes. Their recorded producer source is
`8b08e56e821c8ff72fabf892383c144021b0e8885dfdbe38218e21f38e25f59a`.
G0/step0 pairs with batch0; G1/step2 pairs with batch1. Diagnostic import must
verify those historical identities and restore every state before each run.
The original parents remain incomplete and unqualified; they are never resumed
or rewritten. A diagnostic adapter supplies no owned batch capability.

## Pending evidence

The first scoped core review found missing effective tensor shapes/dtypes in
admission. The fixes cover int64 one-hot intermediates, hand feedforward and
all actor layers, with pure rejection/splitting regressions. The second scoped
wave strengthened endpoint-card and valid empty-hand fixtures. The independent
core reviewer reports READY; a deliberate final-frame-cards substitution is
rejected by both sequence-model tests.

Reported implementation checks on the held source:

- PASS350 original numerical/network/artifact/owned startup/wire/phase-limit/
  resume/structured regressions on the host,247.00s. The first sandbox attempt
  produced309PASS/41FAIL due inherited Unix socket EPERM; the same host selection
  passes without a production change for that environment limitation.
- PASS50 new mechanism checks/1inapplicable MLP shared-prefix skip,14.14s;
  final diagnostic-label subset7PASS,1.59s.
- PASS scoped Ruff and mypy22sourcefiles, plus `git diff --check`.

These precede the final harness fixes and are not a physical-performance claim.
Both full independent reviews then found the same four integration blockers:
numeric bundle basenames resolved from the wrong directory; a historical
observer request-field mismatch; profiler windows beginning after selected
packing; incomplete summaries accepting an invented baseline. Root reproduced
the preparation failure before any output manifest or training. A bounded
integration fix/re-review resolves all four. Both independent roles report READY
with no blocking implementation findings. The final covering selection passes
67checks/1inapplicable skip in14.20s; scoped Ruff and mypy11files pass. Real
preparation then reaches manifest publication after checking all six saved
inputs, but its first repaired invocation fails because root omitted creation
of the output parent directory. No manifest is published by that failed attempt;
root corrects directory setup before retry. This is not a numeric/data failure.

Final covering command:

```text
UV_CACHE_DIR=/tmp/gfb-update-uv-cache uv run --no-sync pytest -q tests/test_learning_update_benchmarks.py tests/test_learning_update_diagnostics.py tests/test_learning_update_execution.py tests/test_learning_update_admission.py tests/test_learning_update_boundaries.py
```

The current-source CLI help also passes in a fresh interpreter with imports of
Torch/torchvision/LightGBM/PyArrow/Optuna/native blocked. This is an import smoke,
not a new installed-wheel verification. No dependency or native source changes
require rebuilding those components for this bounded comparison.

NOT RUN: completed real manifest preparation; bounded CUDA profiler diagnostics;
the 84 fixed-state measured CUDA updates; candidate selection; the 18 fresh
actual runs; post-timing complete-original Source/numeric/receipt audits; final
evidence reviews and recommendation. Measurements must run sequentially without
tests, audits or agent reviews in parallel. Failed attempts remain evidence and
must not be overwritten or relabeled.
