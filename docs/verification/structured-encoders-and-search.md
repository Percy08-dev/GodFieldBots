# Structured encoders and diverse search verification

Date:2026-11-01. Design:[ADR0023](../adr/0023-structured-encoders-and-diverse-search.md).
This ledger records actual checks; planned coverage is not a passed test.

## Initial state

- Bot HEAD:`2d3f9618c4e0997f51b35a80117048ac7b253697`; clean at task start.
- Simulator HEAD:`ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6`; clean in read-only
  inspection. Simulator code is not modified or used as a legality oracle.
- Catalog:`public-ja-v2`, schema1,296 definitions, SHA-256
  `481ddfb0c068a1d2aaaf2dbcb6a5b4bd04396dfcc5c995e14e47b71222b6a51d`.
- Existing MLP/GRU/Transformer-v1 implementation/evidence is retained, including
  old fixed datasets and all failed historical parents.

## Environment checks

| Result | Command | Evidence / limit |
| --- | --- | --- |
| PASS | `git status --short` | No changes before this task |
| PASS | `git -C ../GodFieldSimurator rev-parse HEAD` | Pin above |
| PASS | `uv --version` |0.13.6 |
| FAIL (sandbox limitation) | `nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader` | Operating system blocks NVML in sandbox |
| PASS (GPU permission) | Same `nvidia-smi` command | RTX5070Ti,16303MiB; transient external usage6725MiB, not an available-memory guarantee |
| PASS | `.venv/bin/python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'` |2.14.1+cu132;False under sandbox; no fallback authorized |
| PASS (GPU permission) | `.venv/bin/python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.mem_get_info())'` |True;RTX5070Ti;15738077184free/17094475776total bytes at check time |

GPU-enabled execution permission is required for CUDA checks. A passed device
discovery check is not a numerical/training/performance result.

## Design and public contracts

Read-only `bot_verifier` investigated normative spec03/05/07/09, public rule
register and pinned catalog; it did not read engine/internal support or run
rejection probes. Supported and missing rows are in ADR0023 §10. Public catalog
effect semantics are permitted; missing direct-use completion/target/sequence
rules are [issue0005](../issues/0005-public-utility-action-semantics.md).

Independent architecture design review: READY for ADR0023 §§1–9/11/12 and
confirmed public boundaries. Root resolves four provisional findings through
§12: closed cross-module APIs; generation2/shard and version-aware native
authorization; homogeneous dataset2 cohort; chunk-based admission. Full scope
and unresolved utility rows remain BLOCKED on issue0005. Seed-only wall-time
search replay is explicitly disclaimed. Design review ran no tests.
Implementation, correctness review, conformance review and full completion are
not yet accepted.

Follow-up independent design review of §§6/10: READY for confirmed candidate
rows, after documenting component-vs-completion, Wand singleton/current-rule
precedence, zero-valued attack completion, inherited additional/Aura repeats and
root-adopted Mirage repeats, target variants/concrete Exchange mandatory coverage.
Issue0005 still blocks new utility and pure CutCost placement. No advisor was
needed: the reviewer withdrew its too-broad concern against inherited accepted
repetition semantics after distinguishing them from new per-definition rules.

Documentation check PASS: `git diff --check` and a stdlib pathlib/regex check of
relative Markdown links in the three new documents. No implementation acceptance
follows from those documentation checks.

## Frozen performance source selection

Read-only production source/lineage validation selects first512 accepted learner
TRAIN decisions from completed `runs/ppo-comparison-v2/train-seed17`, update/source
order, batches000000–000003 (224+155+119+first14). Parent run.json SHA-256
`bef3dc92834e8665bacd084ea35434835d1a64deb906f3d52ca711e9b987ec60`, run ID
`63e9b1fd-fb59-4169-935c-67f3fceff76c`, Config SHA-256
`3a5161815788a08ae2be225f9433c3db5d547b6965eb313e94387a4e31a5e416`.
Verifier's canonical identity+original-input panel digest is
`ca0a1e76c80ec7876815e38b15786607819d646c4baf907a55250edce350813c`.

Each of the four batches has32/32 eligible participant sources. The bounded check
used `lineage.validate_chain`, `q_data.load_run(...,'train')`, `validate_bundles`,
and `labeled_samples(...,'train')`, matching each immutable update's source proof.
All selected sources retain original own Observation/Events/metadata/catalog/hints;
no validation/final-test source enters this panel. No benchmark was run yet.
New-recipe benchmark tokenization is ephemeral; it does not rewrite or relabel
old dataset/journals/index. Retained dataset manifest file hash is
`1bfe1607dc80ee3def7ce4ab4ae24a807164dce8d09e67a8b5f2a9a75f3ece55`.

## First pure implementation slice

Source: five new modules `gf_common/{structured,structured_tokens,structured_vocab}`
and `gf_models/{structured_config,structured_network}`. Legacy production source,
Simulator, dependencies and historical datasets/artifacts are unchanged.
These APIs are not yet connected to CLI/transport/learning or inference artifacts.

Independent `bot_reviewer` correctness and `bot_architecture_reviewer` conformance
reviews are READY. Concrete findings were fixed and rechecked: oversized dense/
chunk override admission; actual suffix events and jointly admitted history plus
effective scoring chunk; first/contiguous/episode-bounded cache intervals;
per-gate/attention initialization; preallocated reusable KV. Root accepts this
bounded slice only. No advisor escalation was needed; no unresolved disagreement
or twice-surviving fix remained.

Root final-source commands:

```sh
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync pytest -q tests/test_structured_records.py tests/test_structured_config.py tests/test_structured_tokens.py tests/test_structured_network.py tests/test_structured_cuda.py
CUBLAS_WORKSPACE_CONFIG=:4096:8 GFB_RUN_STRUCTURED_CUDA=1 UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync pytest -q tests/test_structured_cuda.py
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync ruff check src/gf_common/structured.py src/gf_common/structured_tokens.py src/gf_common/structured_vocab.py src/gf_models/structured_config.py src/gf_models/structured_network.py tests/test_structured_records.py tests/test_structured_config.py tests/test_structured_tokens.py tests/test_structured_network.py tests/test_structured_cuda.py
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync mypy src/gf_common/structured.py src/gf_common/structured_tokens.py src/gf_common/structured_vocab.py src/gf_models/structured_config.py src/gf_models/structured_network.py
```

Results:102PASS/6opt-in SKIPPED in4.43s; separate GPU-permission6PASS in9.14s;
Ruff PASS; mypy PASS(5modules). GPU suite covers default attention×3temporal
models×18/128positions, CPU/CUDA full/split/cache and dense/chunk encoder/all
parameter gradients at unchanged1e-5, plus fixed-buffer reset/reuse. It does not
measure latency/peaks or demonstrate gameplay/training quality.

Implementer selected-legacy command PASS110 in17.67s:
`.venv/bin/pytest -q tests/test_actions.py tests/test_action_options.py tests/test_catalog_v2_models.py tests/test_gru_ppo.py tests/test_transformer_ppo.py`.
An earlier command named nonexistent `tests/test_encoder.py` and ran0tests; the
corrected command above is the evidence. Reviewer's full targeted95PASS preceded
final localized changes; later network57PASS and focused admission7PASS covered
fixes; root's102-case sweep is the final combined evidence.

uv warned that the interpreter is3.14.10 whereas this existing .venv was created
with3.14.9. No interpreter/package change was made. Default uv cache is read-only,
so commands use the disposable writable cache path and `--no-sync` to preserve
the installed environment. Old normal-resume environment guards remain intact.

## Pure candidate generator slice B

Implemented `gf_common.diverse_actions` with compact proposals/table, saturating
pre-count/cost admission, fair bounded family/start/length exploration, mandatory
coverage and seven-channel distinct selection. Reports bind table/config/cost/RNG
and separate traversal from presentation completion. Supported scope is the
confirmed ADR0023 matrix; issue0005 omissions remain explicit. No application
factory/worker/journal/training path uses this generator yet.

Both independent correctness and architecture-conformance reviews are READY after
one review/fix cycle. Resolved: positive additional armor/sundry coverage; fixed
Sell length capacity; zero-mandatory Sell-only domains; channel minimums before
skewed weights; Exchange/Sell fair scheduling alongside long sequences; attempt
checks before proposal work; padded19-row projection/transient and configurable
head-tail costs. Refresh is a visible potential-acquisition proxy including
miracles/capacity, not a future draw claim. Call-local table validation decodes
one immutable frame context and preserves original row/hash/ownership semantics.

Root final-source checks PASS:

```sh
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync pytest -q tests/test_diverse_actions.py tests/test_structured_records.py tests/test_structured_tokens.py tests/test_structured_config.py tests/test_structured_network.py
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync ruff check src/gf_common/diverse_actions.py src/gf_common/structured.py src/gf_models/structured_config.py tests/test_diverse_actions.py tests/test_structured_records.py tests/test_structured_config.py
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync mypy src/gf_common/diverse_actions.py src/gf_common/structured.py src/gf_models/structured_config.py
```

Results:166PASS in8.32s, Ruff PASS, mypy PASS(3modules). This overlaps the pure A
suite and must not be added to102as distinct cases. Correctness reviewer ran the
same166cases and original edge repros; architecture reviewer ran22targeted
generator/record and5refresh/preallocation checks. No fresh GPU/native/wheel/
real-game/performance result follows; no numeric Torch path changed in this slice.

## Remaining full-unit acceptance evidence

Currently NOT RUN: integrated E2E PPO numerics/gradients,
sharded artifacts/optimizer/RNG resume, worker2/journal5/
native differential,192-game baseline, attention3models×2inference devices×
continuous/pause-resume real2updates, other hand-mode MLP games, fixed512train
decision CPU p95<=3s panel and separate stress/GPU attribution, pytest/Ruff/mypy/
Rust, old-data proofs, isolated wheel and optional-dependency-free startup.
Do not turn these entries into PASS without exact commands and actual evidence.
