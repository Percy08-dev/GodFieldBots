# PPO HPO foundation verification

Implementation in progress. This is functional foundation acceptance, not formal
hyperparameter search or evidence of strength.

- Bots baseline: `040184223a4bdcde85875f556daeb4a160ee7027`.
- Simulator source: `91d5926a7572db327eaea3751e7c24f428120996`.
- Design: [PPO HPO](../design/ppo-hpo.md), [ADR0037](../adr/0037-ppo-hpo-foundation.md).
- Pre-existing untracked analysis/session files and fixture lock are excluded.
- Baseline PASS: `.venv/bin/python -m pytest tests/test_config.py -q` —12passed.
- Baseline PASS: `.venv/bin/python -m pytest tests/test_group_pack_formal_contract.py
  tests/test_card_instance_contract.py -q` —125passed, including fixed ASTs and
  historical configuration contracts.
- Protected-original baseline PASS: bounded300s readonly inventory scan,
  29,814frozen inputs and24protected fit files match frozen SHA declarations;
  per-file SHA/dev/ino/size/mtime/ctime captured in
  `/tmp/hpo-originals-before.json` (25.57s). A final post-change comparison remains
  pending. No source artifact or original was rewritten.
- Optional dependency PASS: `UV_CACHE_DIR=/tmp/godfieldbots-hpo-uv-cache uv lock`
  resolved Optuna5.0.0 and seven new optional packages. Existing locked package
  versions are preserved. Installed with `uv pip install --python .venv/bin/python
  optuna==5.0.0` for verification. `import gf_cli` leaves both Torch and Optuna
  absent from `sys.modules`.
- Runtime: `.venv/bin/python`3.14.10, NumPy2.5.4, Torch2.14.1+cu132, installed
  native0.1.0. Ordinary sandbox CUDA/NVML access is blocked; an approved escalated
  `nvidia-smi -L` sees RTX5070Ti. Actual CUDA verification is still pending.
- Current-context architecture mapping, implementation and final acceptance:
  design READY; later implementation units and final acceptance remain pending.
  No unexecuted check is claimed passing.

## Unit1 configuration and admission

The dedicated HPO loader requires explicit ranges/trial count/D/training seeds/
evaluation GameSeeds before training. It preserves the base experiment hash and
old loader semantics. Model4 has activation-aware initialization identity and a
temporal descriptor without TBPTT. Separate training/search records normalize
inactive fields; categorical/joint proposals use stable choice indices.

Initial scoped158tests/Ruff/mypy5 passed. Independent correctness review found
two blockers: pool curriculum was absent from study identity, and search-domain
validation used defaults instead of overridden controls. Both are reproduced by
new regression tests and corrected. Study identity now binds pool interval/size;
search reachability uses the actual normalized immutable control, active branches
and coupled rows, with a bounded65,536-state admission limit per invariant.
Independent re-review is READY, passes168focused checks/Ruff/mypy5 and confirms
both blockers addressed. Root accepts the scoped unit1 implementation; overall
integration remains pending.

No network/artifact/Source/campaign runtime or GPU/game acceptance follows from
these configuration checks.

## Unit2 model, numeric and artifact components

Both independent correctness and architecture-conformance reviews are READY,
without blocking findings. Shared network kernels support model4 activations and
depths while preserving default ordered initialization and all eight fixed ASTs.
Whole natural episodes supply GAE/context; selected endpoints alone supply
normalization and loss. Adam is one canonical ordered group with every option
validated. Inference5 excludes resources/reward/PPO/TBPTT; resume5 binds the full
recipe, resources, execution, named optimizer state and RNG. Resource-free recorded
architecture parsing uses technical ceilings only; allocation requires active
caller admission. The custom >16M-parameter metadata regression allocates no
large network.

Root's fresh affected CPU command (19modules listed in task2's execution report):
`.venv/bin/python -m pytest tests/test_hpo_config.py tests/test_hpo_network.py
tests/test_hpo_numeric.py tests/test_hpo_artifacts.py tests/test_hpo_cuda.py
tests/test_model3_config.py tests/test_model3_network.py tests/test_model3_policy.py
tests/test_model3_artifacts.py tests/test_structured_config.py
tests/test_structured_application_config.py tests/test_structured_network.py
tests/test_structured_learning.py tests/test_structured_sealed_metadata.py
tests/test_structured_history_compat.py tests/test_card_instance_learning.py
tests/test_card_instance_registry_artifacts.py tests/test_card_instance_baseline_runtime.py
tests/test_torch_runtime_lazy.py -q --disable-warnings --maxfail=1 --tb=short`
—576PASS/56explicit SKIP in55.73s.

Independent correctness ran84new tests/10explicit skips and336historical tests/
46skips; independent architecture reran all8AST hashes. These scopes overlap and
are not additional unique test counts. Root project Ruff
(`.venv/bin/ruff check src tests benchmarks native`) passes; full mypy
(`.venv/bin/mypy src`) passes179source files. Bare `ruff check .` finds four
pre-existing errors in the user's untracked `codex_analysis.py`; that unrelated
file is preserved and excluded from project validation.

Implementer actual-host command `GF_RUN_CUDA=1 .venv/bin/python -m pytest
tests/test_hpo_cuda.py -q --disable-warnings --maxfail=1 --tb=short` passes10cases
in9.87s, including all3models, ReLU/Tanh, full/cache/split/CPU parity,1024context,
selected update/resume/RNG and clean GPU participant fixture inference. Root's
independent identical-command rerun passes10cases in9.27s. This is numerical/component evidence; real games,
Source/native/lineage/campaign/evaluation/retention acceptance remain pending.

The old retained registry's6neural/8total guard remains exact. Unit3 adds a
separate bounded HPO registry route for a configurable pool above5. No Tree,
formal-parent/final64/Dataset authority is extended by model4 inference.
