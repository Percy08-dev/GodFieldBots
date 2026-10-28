# MLP PPO implementation and experiment evidence

Date: 2026-10-28. Requirements and accepted design:
[ADR0008](../adr/0008-mlp-ppo-generations-and-fixed-data.md).
Implementation, independent code reviews and the real two-update/resume smoke
are complete. The formal comparison is blocked by an unresolved zero-MP dynamic
weapon target contract; see [the separate Simulator task](../issues/0004-zero-mp-dynamic-weapon-target-contract.md).
No trained-policy quality or final-test claim is made.

## Implemented code verification

The complete implementation has independent correctness and architecture
conformance READY, including the bounded final provenance correction. No actual
PPO experiment has yet been accepted by these code reviews.

| Check | Result |
| --- | --- |
| Root final full ordinary suite | PASS: 479 passed, four opt-in skipped, 127.31s |
| Final private provenance/lineage correction | PASS: 22 focused tests; final ancestor-Git guard four tests, 0.13s |
| Ruff / mypy | PASS, 47 source files |
| Existing real TestServer integration | PASS: four tests, 2.65s |
| Whole-parent offline, final binding and cleanup/deadline guards | PASS in independent focused probes/regressions |
| Both learner seats and nonzero natural returns | PASS, including generation-zero same-policy identity and actual-opponent index counts |
| Installed baseline wheel | PASS: help/config imports without optional ML/data dependencies |
| Installed CPU/cu132 wheel, CPU inference | PASS: 420 candidates, NPZ exact load and clean isolated worker; no CUDA initialization |
| Owned CUDA child | PASS: minimal environment, generation-zero save, driver API 13030, clean exit; parent never imported Torch |
| Installed CUDA wheel numerical update/restore | PASS: 140 synthetic decisions, full pre-step parity, exact resumed next update |

The ordinary suite command was
`env -u VIRTUAL_ENV .venv/bin/python -m pytest -q`; opt-in integration used
`GFB_RUN_INTEGRATION=1` and approved listener access. Static checks used
`.venv/bin/python -m ruff check .` and `.venv/bin/python -m mypy src`.

The installed-wheel synthetic numerical benchmark measured CPU 1.29135s and
CUDA 1.14565s for the same 140-decision update, including full parity and four
epochs, excluding initialization/checkpoint save. It retained the 1.01328e-6
maximum pre-update CPU/device error and exact checkpoint continuation. This is
one component workload, not an end-to-end speedup or learned-policy result.
Final wheel SHA-256:
`7057693fe30574a413683852d9099e3a9d272b40da3d01b4517fd3eff1999d3a`,
121,979 bytes. Every one of 47 Python source files matches the wheel. The final
479-test ordinary suite includes the provenance and CLI metadata corrections;
the earlier 472-test snapshot also passed. Installed baseline, CPU and
cu132 environments use absolute `/tmp/gfb-20261028-*` targets and `python -I`
from `/tmp`. Provenance now hashes exactly seven owned packages/47 files;
installed wheels/source archives cannot inherit an ancestor repository's Git
identity. The original installed-wheel probe reproduced 5,840 hashed files,
including unrelated Torch files, before this correction.

Blocking findings resolved during independent review: preallocation NPZ header
validation; failed/running-parent data escaping through generic offline paths;
spoofed final-policy validation binding; external fixed-seed validation; adverse
child health being published complete; controllable work after execution expiry;
same-path Simulator binary/config drift. Numeric metrics now weight decisions
across uneven microbatches. Default offline projection requires whole-parent
success; online PPO needs a live ownership token bound to exact current batch.
No unresolved blocking code finding is deliberately accepted.

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

Stage C interim component evidence (not real-data learning acceptance): root
tested 140 synthetic numerical decisions with 1/2/7/64/512 candidate counts and
690 public-aggregate features. Full frozen CPU/CUDA comparison passed with
maximum absolute error `1.0132789611816406e-6` at atol=rtol=1e-5; the first
four-epoch update made eight Adam steps. NPZ restore followed by a second
update was bit-identical to continuous CUDA weights. Peak allocated/reserved:
92,488,192/109,051,904 bytes. Disposable verification script:
`/tmp/gfb-ppo-numeric-smoke.py`. This preceded the final RNG-seeding and loader
hardening; the final numeric/hardware checks remain required.

Common encoder migration was independently compared with source revision
`8ed132a` on 100 actual corrected-pilot decisions, every stored candidate, and
both encoder recipes: names/order/signatures and every float64 value matched
exactly. Signatures remain `b6045e464c1488807c76fc5a298008e692bca603e922b7d604e0e023b2c2de14`
(observation) and `befca464acc87b893d1b5fadf47db3bbebbd93dfa206545a216f9fa312d3dedf`
(public history). The existing 192-game pilot was recomputed after the schedule
serializer change and retained source-set hash
`5ff2af59443bb258c17203791568785e67adf876294835071a6769c680b4bb04`.

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

Final read-only inventory still shows the verifier package absent, CPU Torch
present, certifi 2022.12.7 and idna 3.4. The separately proposed repair is now
concrete and dry-run checked: undo exactly the 11 added Torch-related packages,
then restore the Simulator lock **with its dev extra** while preserving known
pre-existing NumPy 2.5.4 through `--inexact`. Plain default sync would drop that
manual NumPy and omit the original dev tools. No actual repair was performed.

```console
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/gfb-uv-cache uv pip uninstall --python /home/percy/develop/GodFieldSimurator/.venv/bin/python filelock fsspec jinja2 markupsafe mpmath networkx pillow setuptools sympy torch torchvision
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/gfb-uv-cache UV_PROJECT_ENVIRONMENT=/home/percy/develop/GodFieldSimurator/.venv uv sync --project /home/percy/develop/GodFieldSimurator --locked --extra dev --inexact
```

Only these commands' `--dry-run` variants were executed: the uninstall plans
exactly 11 removals; restoration plans two version replacements and 16 installs,
including the verifier, pytest/respx/quickjs tools and their lock dependencies.
Unknown pre-incident manual environment details cannot be proven from the lock.

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

Formal source directories, fixed before collection, are
`runs/ppo-comparison/train-seed17`, `train-seed29`, `train-seed43`, in that order.
Final validation directories will be `runs/ppo-comparison/validation17`,
`validation29`, `validation43`. Frozen dataset destination is
`runs/ppo-comparison/dataset.json`. The engineering smoke below is excluded
from that dataset and its shared generation charge.

## Real two-update and pause/resume acceptance

Code revision `a3b7030e081699a0547fdadf69455d64b62b2531`, clean worktree.
Both runs completed all 32 games naturally and healthily, with 379 accepted
learner actions each. All four collection bundles were revalidated against
original journals/Replay/report and immutable source proofs after completion.

| Run | Result | Elapsed snapshot | Coordinator peak RSS | Owned files |
| --- | --- | --- | --- | --- |
| `runs/ppo-smoke/continuous-20261028` | two updates, complete | 70.535s | 144,973,824 bytes | 61,864,026 bytes |
| `runs/ppo-smoke/resume-20261028` | pause after update 1, resume to 2, complete | 73.601s including pause gap | 125,857,792 bytes | 61,864,084 bytes |

Final weights, every Adam array, CPU/CUDA RNG, shuffle state and parent
schedule/opponent RNG are exactly equal across continuous and resumed runs.
Final weight NPZ SHA-256:
`a5fd49ca62e0a3f0fdb7a69e12ffd0cd245af4ed0be9148f21f205f66c3aed61`.
Generation 1 arrays were intentionally deleted after successor commit while
its inference manifest/hash/record remained. Generation 0 and latest remain.

For continuous execution, batch 0 collected in 30.595s, with 224 learner
actions; native update/save response took 1.358s, including 0.0169s save.
Batch 1 collected in 26.614s, with 155 learner actions; update/save took .327s.
The real first batch's maximum CPU/saved/device difference was 5.96046e-7.
GPU allocated/reserved peak was 92,392,448/109,051,904 bytes; learner process
peak RSS was 2,204,463,104 bytes. Coordinator and learner memory scopes differ.
Batch-0 MLP policy-select p50/p95 was 5.618/126.353ms, covering both learner and
same-generation snapshot participants; the report explicitly separates mixed
global metrics and per-policy metrics. These are engineering measurements,
not a policy-quality or general GPU-speed claim.
Final smoke inference weights occupy 681,538 bytes and inference JSON 19,483
bytes. Whole-run throughput is 746 all-policy decisions / 70.535s = 10.576/s,
including collection, encoding, updates and saving; it is not native-only
prediction throughput. Full run artifacts and memory scopes are given above.

Assessment: `runs/assessments/ppo-smoke-20261028.json`, SHA-256
`1de55f66ade2f6a61433bb9d5dacdd78f5e4cb7f46073a3c2f6225282eb1ad2d`.
The smoke ran the CLI with `training.max_updates=2`; the second path used
`--pause-after-update 1`, then `--mode resume` with identical config. Original
deadlines were retained. Child exit codes were zero with no cleanup warnings.
Formal three-seed experiments retain the normal 1000-update/7200s preset.

Independent post-smoke correctness/evidence review is READY: both complete
chains and all four original source proofs pass, including all 224 first-batch
frozen CPU forwards (max error 5.96e-7). Each run has 746 accepted actions across
both views, 379 learner actions and 16 Adam steps. Twelve weight arrays and 38
optimizer/RNG arrays match exactly across continuous/resumed execution.

After this smoke, a metadata-only correction gives schema-4 batch/validation
manifests their actual distinct sampling-recipe descriptions instead of the
legacy baseline wording. The stored numeric streams, complete schedules and
parent RNG controls were already correct and were independently compared.
Historical smoke manifests are retained unchanged. Baseline/schema-3 text is
unchanged; 25 focused CLI tests pass, including three pre-startup provenance
cases. No numerical, opponent-selection or scheduling behavior changed.

## Formal experiment attempt: blocked

The prescribed seed-17 parent ran at clean code
`9faeac5d14b279b3ea4be5c57dc0c8b958b5ad37` with unchanged default budgets.
Its generation-2 weights exactly matched the engineering smoke, confirming the
metadata-only correction did not change that trajectory/update sequence.
It committed 24 updates and 3,800 learner actions, then batch 24 had 15 natural
games and one heuristic `invalid_target` rejection: perceived Magical Stick,
MP 0, self target. Total: 399 natural of 400 attempted games, elapsed 761.961s.

The entire parent is failed/quarantined, including all 24 healthy earlier
batches. Root checked the actual retained source: early-batch ingestion,
generic export projection, formal checkpoint qualification and failed-run
resume each reject it. Generation/update 25 and failed-batch numeric input do
not exist. No CPU fallback, repaired probability, substituted game or offline
reuse was performed. Seeds 29/43, 192-game final validation, fixed dataset and
six real LightGBM fits are NOT RUN because the three-parent gate is unmet.

The advisor and architecture investigator agree that current public normative
sources do not establish the legal MP-0 singleton target shape. Root accepts
recommendation C: [separate Simulator clarification](../issues/0004-zero-mp-dynamic-weapon-target-contract.md),
then a reviewed version-pinned Bots change and wholly fresh formal parents.
No production legality rule is inferred from rejection details or typed syntax.
No Simulator source or contract is modified by this Bots task.

Actual failure assessment:
`runs/assessments/ppo-seed17-failure-20261028.json`, SHA-256
`83d40a2e8ed2bd6d63608d37d4e670b7d79c3d54ee04300af4b69c8a5d312e98`;
failed parent `run.json` SHA-256
`d34c28e02a93ca5cc89ab861d891be25bdeb34a3d8ba60335a74dc1faa3cd6b9`.

Independent failure-handling evidence review is READY. It verified all 24
immutable commits/counters (184 Adam steps), the missing generation/update/input
for the failed batch, source rejection without acknowledgement/terminal label,
and default offline/formal/resume-admission rejection. It did not run additional
PPO or a legality probe. This validates failure handling, not the blocked
three-parent comparison.
