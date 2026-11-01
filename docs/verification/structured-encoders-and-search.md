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

## Pure persistence and E2E numeric slice C1

New model modules: `sharded_arrays`, `structured_artifacts`, `structured_policy`;
training modules: `structured_rollout`, `structured_numeric`. MLP's private
history path validates original history but encodes only its current frame.
No CLI/worker/journal/native/Generation2/formal-source path is connected yet.

Artifact/resume3 and rollout3 preserve raw own snapshots/event banks/compact
tables and deterministic tokens, without learned latents. Typed shards are
pickle-free with decoded total<=64MiB; separate bounded JSON manifests<=64MiB
have complete preplanned metadata sizes. Standard Adam remains unchanged; named
state preserves missing moments and per-parameter steps/RNG. CPU processes need
the clean-child CUDA-visibility boundary for strict physical ownership; pure CPU
math does not promise that standard Torch Adam never probes available GPUs.

Both independent reviews are READY after a coherent fix cycle A–F. Resolved:
actual network/wrapper/save semantic and runtime binding consistency; joint
raw/token/decoded/JSON and per-candidate ownership before allocations; full actual
canonical Adam options/order/ownership; impossible negative variance; preflight
of every owned final/temp path including dangling symlinks; full bundle/manifest
planning and identical reader/writer size bounds. Failed restore preserves live
weights/optimizer/RNG. No advisor escalation was needed: two overbroad CPU-Adam
test sweeps were corrected against pinned Torch behavior, not repeated failed
production fixes. Real source qualification remains a separate C2 responsibility.

Root final-source commands:

```sh
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync pytest -q tests/test_structured_learning.py tests/test_structured_network.py tests/test_structured_config.py tests/test_structured_learning_cuda.py
CUBLAS_WORKSPACE_CONFIG=:4096:8 GFB_RUN_STRUCTURED_LEARNING_CUDA=1 UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync pytest -q tests/test_structured_learning_cuda.py
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync ruff check src/gf_models/sharded_arrays.py src/gf_models/structured_artifacts.py src/gf_models/structured_policy.py src/gf_training/structured_rollout.py src/gf_training/structured_numeric.py src/gf_models/structured_network.py tests/test_structured_learning.py tests/test_structured_learning_cuda.py
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync mypy src/gf_models/sharded_arrays.py src/gf_models/structured_artifacts.py src/gf_models/structured_policy.py src/gf_training/structured_rollout.py src/gf_training/structured_numeric.py src/gf_models/structured_network.py
```

Results:168PASS/4opt-in SKIPPED in22.96s; final separate GPU4PASS in24.16s,
reaped exit0; Ruff PASS; mypy PASS(6modules). Earlier GPU4PASS9.79s preceded the
review fixes and is superseded by that final result; these elapsed test times
are not controlled performance comparisons. C1 overlaps A tests and must not be
summed as distinct cases. The independent correctness reviewer ran87C1 cases and
the original corruption/ownership reproductions; architecture ran18focused cases.

Evidence includes all3temporal numeric/resume paths, all3hand mode late-state
behavior, whole-batch saved/cache/full/learner/split parity at unchanged1e-5,
129-endpoint logical stepping,10,001-candidate global normalization, negative
state/token/manifest cases and actual1024-frame sharded roundtrip/1025early
rejection. Independent P2-loss/P1-win/P2-draw probes verify target perspective
and whole-batch normalization. Synthetic data is not a feasibility/quality gate.

## C2 configuration baseline and pending integration

Read-only protected-artifact check PASS at this checkpoint: authoritative
`runs/assessments/sequence-ppo-20261031/baseline.json` (SHA-256
`d09ba36c3f5310ce4b90c4a0913b74c650efd81ced3dd85c237e5af45fe5a2c9`)
lists24original fit files; all24current file hashes match. Frozen old dataset
manifest file SHA remains
`1bfe1607dc80ee3def7ce4ab4ae24a807164dce8d09e67a8b5f2a9a75f3ece55`.
This bounded hash verification reads no model values and is not the later full
corpus/index/numeric proof. Recheck after integration/build milestones.

Before new Config dispatch, a read-only verifier loaded all12git-tracked v1
experiment TOMLs using committed cfcb068 code from an isolated `/tmp` snapshot,
while retaining original workspace input paths. All28unique raw input/preset
files matched commit bytes;65ordered input entries, effective/LoadedConfig hashes
were captured, and Torch was not imported. Proof:
`/tmp/gfb-legacy-config-cfcb068/proof.json`, SHA-256
`cb6ee02993282fd7508087745a4d17e2c0c52863a9b3a036b463705013148db8`.
Root's new-parser comparison PASS: all12LoadedConfig/effective hashes and ordered
input path/kind/sha256 entries match exactly. The first root comparison mistakenly
included proof-only commit_sha256/matches_commit fields in the input equality;
after comparing the original three config fields, all identities match. No parser
or old preset change was needed. Six new draft experiment graphs covering11new
TOMLs load semantically, preserving worker10s and7200/5400/7140run budgets, exact
pure architecture conversion and only the model TBPTT setting. No Torch import
or Simulator I/O occurs. Draft files remain in `/tmp` until runtime wiring works.

Independent correctness and conformance reviews of this Config/binding slice are
READY. The existing full uint64 sampling contract is restored in Policy.reset,
including2**64-1, without changing parent derivation or initialization seeds.
Actual execution identity binds active resources/inference/visibility; baseline
None values remain explicit and PPO missing values reject before tokenization.

Root scoped checks PASS:

```sh
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync pytest -q tests/test_structured_application_config.py tests/test_config.py tests/test_preprocessing_config.py tests/test_lightgbm_config.py tests/test_structured_records.py
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync ruff check src/gf_cli/config.py src/gf_common/preprocessing.py src/gf_common/structured.py src/gf_models/config.py src/gf_models/structured_binding.py src/gf_models/structured_policy.py src/gf_training/structured_rollout.py tests/test_structured_application_config.py tests/test_structured_learning.py
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync mypy src/gf_cli/config.py src/gf_common/preprocessing.py src/gf_common/structured.py src/gf_models/config.py src/gf_models/structured_binding.py src/gf_models/structured_policy.py src/gf_training/structured_rollout.py
```

Results:112PASS in5.45s; Ruff PASS; mypy PASS(7modules). Correctness independently
ran82application/legacy/preprocessing cases and2affected inference-label cases;
architecture checked40application cases plus7uint64boundaries. These overlapping
counts are not a full-project test total. No bootstrap/protocol/codec/source/
lineage/native implementation or real-game acceptance follows from this slice.

## C2 candidate-column codec and owned sidecar I/O

The pure common `candidate_columns` codec and parent-side training
`candidate_storage` implement ADR0023 §15. The maximum65536×128layout is
11,403,272bytes. Decoding checks exact six-column layout,0-based raw codes,
little-endian integers/alignment/zero padding, lengths/missingness/hashes and
original frame/own references. Publication uses exclusive owned temporary files
and links; only validated descriptors can be returned. Journal/Act/source/native
integration is not yet connected.

Both independent reviews are READY. Three concrete I/O issues were fixed:
inventory digest/identity from different snapshots; temp/link object substitution;
and blocking non-regular reads. A residual final-hash EOF mutation was fixed in
the next bounded cycle, then its original reproduction rejected; the same issue
did not survive two fix/recheck cycles, so advisor escalation was not triggered.
The implementation retains verified FDs, rehashes/checks the same snapshot, and
uses nonblocking/no-follow regular-file checks. The related new C1 JSON reader
uses actual-FD checks too; legacy writers/formats remain unchanged. Existing soft
OS/filesystem wall guarantees are not upgraded to hard quotas.

Root final-source checks PASS:

```sh
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync pytest -q tests/test_candidate_columns.py
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync ruff check src/gf_common/candidate_columns.py src/gf_training/candidate_storage.py src/gf_models/sharded_arrays.py tests/test_candidate_columns.py
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync mypy src/gf_common/candidate_columns.py src/gf_training/candidate_storage.py src/gf_models/sharded_arrays.py
```

Results:33PASS in0.60s; Ruff PASS; mypy PASS(3modules). Independent correctness
ran33cases plus original/normal snapshot reproductions; architecture ran11focused
cases after the residual fix. Child FIFO/symlink probes were bounded and reaped.
Earlier32-case evidence preceded the final EOF guard and is superseded. The
implementer also ran66affected C1/codec I/O checks,53deselected, in14.88s before
that final localized publisher fix. No GPU math, native or performance claim.

## C2 standalone protocol2 worker and journal5 transactions

Implemented pure wire DTOs, compact Random/heuristic adapters, own caps/cost
derivation, global reservation primitives, fresh structured bootstrap/worker,
structural journal5 writer/reader and private host timing counters. This is a
component boundary; Environment/CLI selection and formal Source/Replay/split/
whole-parent qualification are not connected yet. Reader documentation makes
that structural-only evidence explicit. No real Simulator game was run here.

Both independent reviews are READY after the initial fixes and one residual
cap recheck. Fixed startup-thread construction/start and all initialization
validation failures (including cancellation/overflow), preserving original errors
and reaping before lease release; strict J5 schema2-vs-None typing, advertised
candidate/decision/interval/episode/position caps before side effects, neural
value requirements, closed outcomes, canonical UUID/integer command validation
and no-overwrite verified final-journal publication. Interval writer/reader
consistency was the residual; it rejects before event decoding/bank/files change.
No same issue remained after two bounded fix/recheck cycles, so no advisor was
needed. Legacy journal4 publication/serialization remains unchanged.

The documented stdlib-only model-metadata/cost helper dependency from the v2
client is independently design-approved; common stays pure, coordinator imports
no Torch and model execution remains inside the clean child. Actual CPU/CUDA
visibility, artifact/action hash, per-view bank/frame/table references and separate
search/CPU sampling RNG are bound. Model loading is timed outside decisions;
host pack/enqueue/sync/return/sampling counters do not claim GPU kernel time.

Root final-source CPU checks:

```sh
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync pytest -q tests/test_structured_protocol.py tests/test_structured_protocol_cuda.py tests/test_structured_network.py tests/test_structured_learning.py
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync ruff check src/gf_common/structured_wire.py src/gf_models/structured_baselines.py src/gf_models/structured_runtime.py src/gf_simulator_client/structured_resources.py src/gf_simulator_client/structured_worker.py src/gf_simulator_client/structured_bootstrap.py src/gf_training/structured_journal.py src/gf_models/structured_network.py src/gf_models/structured_policy.py tests/structured_worker_fixtures.py tests/test_structured_protocol.py tests/test_structured_protocol_cuda.py
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync mypy src/gf_common/structured_wire.py src/gf_models/structured_baselines.py src/gf_models/structured_runtime.py src/gf_simulator_client/structured_resources.py src/gf_simulator_client/structured_worker.py src/gf_simulator_client/structured_bootstrap.py src/gf_training/structured_journal.py src/gf_models/structured_network.py src/gf_models/structured_policy.py
```

Results:223PASS/3opt-in SKIPPED in29.63s, Ruff PASS, mypy PASS(9modules).
The implementer additionally ran legacy `tests/test_journal.py`:28PASS0.69s.
Correctness independently ran79protocol cases and original failure repros;
architecture ran44focused prior boundaries and7final cap regressions. Counts
overlap earlier stages and are not additive full-project totals.

Root separate CUDA command:
`CUBLAS_WORKSPACE_CONFIG=:4096:8 GFB_RUN_STRUCTURED_PROTOCOL_CUDA=1 UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync pytest -q tests/test_structured_protocol_cuda.py`.
Final startup-fixed GPU result3PASS18.92s/reaped exit0; subsequent changes were
only journal interval-cap checks unused by this GPU path. Earlier3PASS25.64s
predated startup/J5 fixes and is historical. Full documented default attention/
model widths are used for all3kinds, real clean CUDA bootstrap/2frames vs CPU
full reference, same-seed fresh-worker CPUGenerator replay, finite host metrics
and reservation release after reap. These are not controlled speed/peak-memory,
real-game or final source-qualification evidence.

## C2 Environment/CLI mechanical dispatch checkpoint

Six frozen files implement base-Environment private hooks with unchanged v1
defaults, a structured client adapter and CLI composition, and additive native
capability checks. CLI injects journal/factory/one shared ledger; client imports
no training. Opaque actor-local banks and original-frame Action reconstruction
precede J5 transaction/Act. Formal bare neural2 use remains rejected pending
Generation2 source qualification. New real rule-dependent collection was not run.

Both independent reviews are READY for this mechanical/component scope.
Command: `UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync pytest -q tests/test_structured_environment.py tests/test_episode_behavior.py tests/test_runtime.py tests/test_journal.py`.
Implementer69PASS4.84s; correctness independently69PASS5.19s; architecture12focused
FakeHTTP cases PASS. Scoped Ruff6files/mypy5sources/diff checks PASS. Tests cover
two-seat baseline labels, actual CPU children, markers/inventories/natural test
closures, two concurrent games/four slots sharing one ledger, cleanup and no-Act
on sidecar/row failure, early old-native/NN2 qualification denial and no Torch
import for metadata dispatch. They use ordinary single physical weapon fixtures,
not old learned-repetition legality or actual Simulator behavior.

The subsequent user clarification changes proposal semantics. The new required
action_contract/engineering-only real-I/O gates are a separate reviewed amendment
and are NOT implemented by this checkpoint. The dispatch result cannot count as
feasibility, Source eligibility, real PPO, native processing or performance.

## Owner clarification/adoption gate

The full user assertions and root's proposed adopted subset are recorded in
`docs/contracts/`. Advisor recommends B(engineering-only); root adopts B and
architecture design is READY. Owner assertion hash a0e6acb7…; full profile4f180899…;
adopted rule/omission/claim hash f4132843… (full values in ADR0023 §17/JSON).
Current live Simulator working-tree document hashes differ from initial clean
declarations while HEAD remains ffcffa; no Bots agent edited them and no binary
conformance is inferred. Target wording and several coverage rows remain pending.

Previous B generator repeat-domain READY is historical evidence only. New actions2
must use unique physical references, real-domain18, explicit contract propagation
and synthetic-only128 provenance. Real-I/O/feasibility/qualified learning/full
acceptance stays blocked; component/FakeHTTP/offline/synthetic work continues.
New pre-adoption v2 artifacts missing contract fields must reject current use;
v1 configurations/readers/pins and protected data remain untouched.

## Owner-contract implementation and current full regression

The engineering profile is implemented across current Config, generator/report/
table/codec/wire, runtime/safe-options/J5, artifact/resume3 and rollout/Learner
boundaries. Every metadata reader requires explicit expected contract/purpose/
synthetic classification; no missing-field upgrade. Unique physical proposals
are bounded18, aliasing learned IDs rejects at new Frame creation, distinct
physical copies/sources remain distinct.128fixtures explicitly mark synthetic
capacity and cannot enter game-proposal or qualified-source paths. V1 semantics,
global simulator/spec/catalog declarations and original artifacts remain frozen.

Both independent core and metadata reviews are READY. The one metadata finding
was normal-API True-artifact→loaded-network→False Policy/resave relabeling. The
immutable private network input-binding tag now persists loaded/trained/staged
provenance and rejects incompatible wrapping/save/update/restore before relevant
allocation or mutation. This is not tensor taint tracking and changes no feature,
weight shape or state_dict/layout signature. Central Source qualification remains
mandatory future work; profile/purpose/producer flags are not authority.

Scoped engineering command before the tag fix was623PASS/13opt-in SKIPPED; final
affected tag/resume/runtime regressions278PASS/7SKIPPED plus metadata60PASS.
Independent core review267PASS/architecture85PASS and final metadata60PASS/
8selected numerical+10architecture tag cases are overlapping evidence, not totals.
Root final GPU command:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 GFB_RUN_STRUCTURED_CUDA=1 GFB_RUN_STRUCTURED_LEARNING_CUDA=1 GFB_RUN_STRUCTURED_PROTOCOL_CUDA=1 UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync pytest -q tests/test_structured_cuda.py tests/test_structured_learning_cuda.py tests/test_structured_protocol_cuda.py
```

Final post-tag result13PASS25.77s/reaped exit0; earlier13PASS33.39s was pre-tag
and remains historical. No controlled performance inference follows.

Full-project validation exposed two environment/fixture boundaries, not weakened
contracts. Default-sandbox full pytest was interrupted(exit130) after failures
and a long native case; it is NOT a passed full run. First fail-fast result was
374PASS+1ERROR: synthetic fixed-data tests inherited a real source_dir whose
external public specs now differ from frozen v1 hashes. Test-only
`tests/ppo_support.py` supplies inert non-executable temporary binary identity and
source_dir=None only to synthetic collectors. Six affected test sites share it;
model/settings/counts/update/resume budgets are unchanged. A controlled temporary
spec drift test still exercises the unchanged production hash rejection.
Independent test-isolation review is READY.

The next fail-fast result419PASS/8SKIP/1FAIL confirmed the same inline synthetic
coupling. A bounded native probe then timed out while FFI was idle: async future
callbacks progressed only on25s timer wakeups. A minimal same-process test showed
sandbox self-pipe send PermissionError(errno1)/future timeout0.704s, while identical
code with local-IPC permission returned in0.051s. NativeExecution was NOT changed
to hide this environment limitation. Owned diagnostic process groups were reaped;
one root duplicate probe was promptly stopped, then diagnostics remained single.

With test-only isolation and appropriate local-IPC permission, the original4
GRU/Transformer/native coordinator two-update/resume cases plus drift guard pass
5/5 in24.31s. Root full check, in that supported environment:

```sh
timeout --signal=INT --kill-after=15s 600s env UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync pytest -q --tb=short
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync ruff check .
UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync mypy src
```

Results:1460PASS/46SKIP in350.21s; full Ruff PASS; mypy PASS(100source files).
GPU13cases above were separately enabled; skipped cases retain their original
opt-in/environment scopes. No blanket claim that every skipped check passed.
Post-run all24protected original fit hashes and frozen dataset manifest hash
remain equal to the recorded baseline. New explicit-profile draft6graphs and
old12Config/effective/input identities pass without Torch import.

These checks establish current engineering implementation/regression scope.
They do not satisfy real feasibility, qualified Source, Generation2/PPO orchestration,
new native differential, LightGBM2 or final performance/full-plan acceptance.

## Current Python wheel and optional-free environment

Root built the committed b31043e Python package offline with
`UV_CACHE_DIR=/tmp/gf-structured-uv uv build --wheel --package godfieldbots --out-dir /tmp/gfb-owner-wheel-20261102 --offline --no-sources`.
Wheel SHA-256 `41fa4fa6e9af67b8cdca0f8c4adcc85768c687c1be20c763992a09b14bffeb67`;
all100packaged Python files exactly match source bytes. This is the current
Python milestone, not a later native/final package build.

A fresh3.14.10 GIL-enabled venv `/tmp/gfb-owner-wheel-env-b31043e` installed only
the app wheel with `uv pip --no-deps --offline --no-index`. Verified local baseline
NumPy/httpx and transitive distribution files were copied into that disposable
environment with source/destination hashes checked; no optional ML/data/native
distribution was copied or installed. Package imports originate in the isolated
wheel environment, with no repository src path.

Under an import blocker for Torch/torchvision/LightGBM/PyArrow/gf_native, help,
all12oldConfig identities,6explicit engineering graphs,2structured baselines and
a fresh protocol2 Random worker initialize/ready/close/reap PASS. Worker leases
are gone and optional imports/distributions remain absent. No Simulator request
or model learning occurs. Evidence files `/tmp/gfb-owner-wheel-20261102/copy-proof.json`
and `runtime-proof.json` preserve the bounded scope. Rebuild/reverify after later
Source/native/tree implementation; do not carry this wheel result forward to
untested source bytes.

## Offline CPU latency panel: engineering components

Root retained the frozen512 TRAIN inputs and a separate dual-view workload under
`runs/assessments/structured-latency-20261102`. Dual-view extraction verifies the
same four original collections and adds512peer/tail decisions:1024accepted inputs
across49games, with exactly the original512identities marked as panel members.
Each participant child receives only its own original six-field DecisionInput;
the original datasets/journals are not rewritten. Panel input canonical digest
remains `ca0a1e76c80ec7876815e38b15786607819d646c4baf907a55250edce350813c`;
dual-view JSONL SHA-256 is
`619c6b707ad4c29b8409da5c5b4f97050cf3d1060824ae34ec1a8837b601d98c`.

The retained harness uses full default attention architectures and action budgets,
fixed seed17 untrained component weights, CPU inference,2concurrent games and
at most4independent participant workers. Startup is outside the decision clock.
The decision clock includes original-input validation, frame/event construction,
candidate generation, tokenization, scoring, CPU sampling, worker communication,
parent validation and Action reconstruction. No Simulator requests or journal
publication occur. Worker processes/reservations are closed after each game.
All512panel rows and the separate512peer/tail rows complete for each model.

| CPU model | Panel p50 | Panel p95 | Panel maximum | Startup p95 |
| --- | ---: | ---: | ---: | ---: |
| MLP | 14.37ms | 92.68ms | 1171.39ms | 1607.93ms |
| GRU | 14.39ms | 92.38ms | 1079.24ms | 1551.74ms |
| Transformer | 14.34ms | 87.97ms | 339.04ms | 1565.40ms |

Startup has98observations per model. This panel reaches2809candidates, sequence
length8 and240visible events; it does not cover the separate tens-of-thousands,
18/128-position or1024-decision stress conditions. Each model is a single run;
small differences are not evidence of a model speed ordering. The action profile
is the explicit engineering-only subset, not a qualified complete game contract.
These measurements are below3seconds within their measured component scope;
they do not complete full-plan performance or real-game acceptance.

Exact command, repeated sequentially for `mlp`, `gru`, `transformer` while the
implementer paused heavy checks:

```sh
timeout --signal=INT --kill-after=10s 300s env CUDA_VISIBLE_DEVICES= CUBLAS_WORKSPACE_CONFIG=:4096:8 UV_CACHE_DIR=/tmp/gf-structured-uv uv run --no-sync python /tmp/gfb-structured-latency-20261102.py --kind mlp --input /tmp/gfb-structured-performance-dual-view-20261102.jsonl --output /tmp/gfb-structured-latency-mlp-panel-20261102
```

All commands exit0. Retained harness SHA-256
`80397e86002a8b3b3ea4b1a62ab8f3b9383fc08945736df54273fb0d88e83bb8`;
evidence inventory/source snapshot `proof.json` SHA-256
`8ed8f463a3451a194c742670b7b488428e991f041f6a3ef71cb73dc107ea59a9`.
Result SHA-256 values are MLP
`f0c2434b828fceab6b1755ae4fe32f93ae3e92a242c7643186eb2afb1b41b427`, GRU
`d8a7b2710ac20baf238a5c8f55c864a048c246dd9a277443edb1534ccd986d78`,
Transformer `870b6ccbf4229b08921fd484869e0e3df6e37e68da427c08ba5d47deb4ad53d4`.
Independent read-only evidence/arithmetic audit PASS: all17evidence file hashes,
37inference/worker source hashes, exact512panel/512supplemental identities and all
reported quantiles match. The successful-run empty-ledger gate implies confirmed
worker reap before result publication. The audit ran no tests, Torch, Simulator,
GPU or repeated timings. A separate correctness assessment reproduced a P1
constructor-exception cleanup gap: exceptions after reservation/before Popen leave
a lease, and exceptions between returned Popen and registration leave a child
and lease. Four bounded MemoryError/CancelledError probes were explicitly
killed/reaped/closed and reservations recovered. This violates agreed failure
ownership and must be fixed before accepting the next runtime milestone. It is
not evidence of a leak in these successful measured runs. No new GPU attribution
result.

## Remaining integrated acceptance

Currently NOT RUN: Environment/CLI/qualified collection E2E PPO integration,
journal5/native differential,192-game baseline, attention3models×2inference devices×
continuous/pause-resume real2updates, other hand-mode MLP games, fixed512train
decision CPU p95<=3s acceptance under the complete qualified contract, separate
stress/GPU attribution and new Rust/native checks. Repeat old-data proofs, wheel
and optional-dependency-free startup after the remaining implementation.
Do not turn these entries into PASS without exact commands and actual evidence.
