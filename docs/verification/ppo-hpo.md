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

## Inactive-field conformance correction

A later root/architecture check found two additional configuration gaps before
campaign implementation: MLP TBPTT was treated as active even though MLP consumes
only its current frame, and event refusal caps were exposed to search. Both are
corrected. `normalize_training_for_model` canonicalizes MLP TBPTT32 before
effective identity and new campaign/run materialization. GRU/Transformer retain
active TBPTT. Event max_interval/max_episode remain fixed configuration facts,
outside search/conditions; aggregate_window remains searchable within those caps.
The model-independent recipe parser preserves honest numeric-component metadata.

Implementer198affected checks/static pass. Independent architecture re-review
is READY and runs110HPO cases successfully. These replace the earlier inactive/
event-cap search guidance, without changing old recipes or saved-data semantics.

## Structural lineage prerequisite4a

Source/native need actual new-version lineage readers, so this bounded reader
prerequisite precedes unit3; the coordinator and retention producer still follow.
Run5/Generation4/Update5/Registry4 readers reuse the same scan, bundle and sealed
artifact machinery. Both independent correctness and conformance reviews are
READY without blockers. Source→chain dependency direction remains explicit.

Root fresh command:
`.venv/bin/python -m pytest tests/test_hpo_lineage.py tests/test_hpo_config.py
tests/test_structured_lineage.py tests/test_card_instance_lineage.py
tests/test_lineage_control_publication.py tests/test_ppo_lineage.py
tests/test_hpo_numeric.py::test_all_eight_frozen_kernels_remain_exact
-q --disable-warnings --maxfail=1 --tb=short` —341PASS in106.72s.
This includes89new structural cases,110config cases and142legacy/AST cases.
Root project Ruff passes; full mypy passes180source files. Independent correctness
also runs the89new cases (21.04s), Ruff and mypy1 successfully; architecture review
does not claim additional test execution.

Fixtures save actual CPU HpoLearner artifacts and cover G0/updates, all/one/partial
selection declarations, both storage variants, exact counters/steps/seed/policy
registries/deadlines, rehashed corruption and per-file deletion receipts with
G0/latest2/pool/periodic/evaluation dependency protection. Structural CPU complete
metadata is explicitly unqualified for production HPO. Actual original Collection6
resolution/replay, selection RNG/GAE reconstruction and exact prefix costs remain
unit3's work. No original Source, GPU campaign, coordinator, deletion producer or
Optuna evaluation authority is granted by these reader tests.

## Python Source3a and Wire6

Source3a is the Python scheduling/selection/complete-original Source/Numeric5/
Wire6 milestone. Actual native support remains Source3b; current installed
native HPO dispatch is strictly refused, with no fallback or advertised marker.
The same Source, custody, scan, numerical loop and process owners are reused.
Journal6 now explicitly binds own neural model4 metadata; Collection6 publication
checks its closed HPO role/context. Old versions retain their exact meanings.

Proof3 preserves both-view accepted-original counts; learner collected/used counts
are separate. Original-backed selection replay rejects even a rehashed alternative
valid subset. Complete episodes supply targets/context, selected rows supply loss/
normalization. Wire6 validates actual saved state/counters/costs and retirement
freezes236collected/1used/delta3 in the controlled case while retaining originals.
Historical inference deletion is accepted only with SAME-chain successful receipts,
immutable generation hashes and actual journal-carried inference5 metadata. The
original proof/correspondence remains identical; active policies require weights.

Independent correctness initially found two P2 blockers: legacy999cap on HPO
generation linkage and missing seed-derived G0 opponent RNG validation. Both were
reproduced and fixed. HPO supports its technical domain/canonical minimum-width
names including1000/1,000,000; old PPO keeps999. Full parent/live planning now
checks exact `PCG64(learning_seed)` initial controls before original/native or
production work. Independent scoped re-review passes5cases and static checks;
architecture rechecks the cap fix and is READY for the Python milestone.

Root fresh host command (local preload sockets require host access):
`.venv/bin/python -m pytest tests/test_hpo_selection.py tests/test_hpo_source.py
tests/test_hpo_wire.py tests/test_hpo_native.py tests/test_card_instance_source.py
tests/test_structured_numeric_worker.py tests/test_group_pack_decoded_retirement.py
tests/test_card_instance_learning.py tests/test_card_instance_collection_publication.py
tests/test_card_instance_registry_artifacts.py tests/test_hpo_lineage.py
-q --disable-warnings --maxfail=1 --tb=short` —268PASS/25unchanged CUDA SKIP
in110.56s. Root full project Ruff passes; mypy passes185source files.
Earlier implementer affected322PASS, host254PASS/25SKIP and post-registry14PASS
are overlapping scoped evidence. Independent initial review ran26new cases.

Controlled cases include24-game schedule order past pair10, same learner/snapshot
label with explicit own viewer, GAE/token/scalar/normalization tampering, historical
fully pruned controls, pool>5 through SAME custody, and actual CPU Wire6 update/
restore/bad-ack reap. No actual game, production CUDA Source/coordinator, native
positive dispatch, deletion producer, Optuna objective or strength claim follows.
The [native authorization design](../design/ppo-hpo-native-handshake.md) is
architecture READY; its implementation/positive candidate checks remain pending.

## Management5a: strict scores and durable GPU admission

Implemented in an isolated worktree and integrated as local commit38444a9 after
both independent fix re-reviews. Pure evaluation binds complete seed/GameSeed/
seat panels and actual terminal results, final-D checkpoint/Source identities
and healthy projections; it grants no Source authority itself. The fixed
per-uid persistent GPU namespace spans campaigns/checkouts and ignores HOME/XDG
overrides. Actual audited reservation history, durable job admission and exact
termination receipts replace PID/flock/time assumptions. Restart recovery is an
explicit operator confirmation and never rewrites run data/deadlines.

Initial broad correctness checks passed, but architecture review reproduced two
blockers: absent/replaced creation barriers could produce never-started evidence,
and retry after parent fsync failure skipped uncertain ancestry durability. Both
were fixed using the separately reviewed
[positive creation observations](../design/ppo-hpo-creation-proof.md) and ancestry
resync on every admission. Existing owners retain actual LOCAL Future/Popen,
original Event and same observed stdlib failure; observer errors preserve actual
adoption/reaping and primary errors while withholding receipt authority. Legacy
resource bounds/accounting and eight ASTs remain exact.

Independent conformance runs15targeted fixes; correctness runs79cases/65existing
optional skips and21additional lifecycle cases, plus static checks. Both report
ADDR/READY. Root fresh isolated command runs255PASS/65existing optional skips
in11.80s, with Ruff9files/mypy7source passing. Root post-integration host command
`.venv/bin/python -m pytest tests/test_hpo_evaluation.py tests/test_hpo_gpu_lease.py
tests/test_hpo_wire.py tests/test_hpo_source.py tests/test_worker_cleanup.py
tests/test_hpo_artifacts.py tests/test_structured_numeric_worker.py
-q --disable-warnings --maxfail=1 --tb=short` passes167cases in42.79s.
Full project Ruff and mypy187source pass.

Cases exercise actual owned CPU processes, default stdlib missing executables,
delayed creation/cancellation, task-creation failure, observer failure before/
after forwarding, mismatched/forked capabilities, orphan admissions, corrupted
state, exact receipt retry, ancestry and atomic/fsync faults, and complete-panel
scoring/failure refusal. No GPU, Simulator, Optuna campaign or concrete Source/
coordinator wiring was executed in this component scope. The Source exact-ledger
allowance, complete resource projection and actual pipeline cleanup integration
remain required before production acceptance.

## Native Source3b private-candidate acceptance

Both independent correctness and implementation-conformance reviews are READY
with no blocking findings. Native uses exact Hpo/PackedHpo plan5, pending/final
controls5 and additive authorize_hpo on SAME Source/provider/session/custody.
Python independently derives schedule/subset/GAE/normalization; native only
validates facts and structural correspondence. Selected/collected/optimizer
deltas remain distinct through bind/begin/prepare. Test-only CPU admission is
cfg(test), with no release API or flag. Historical deletion receipts and active
policy dependencies retain their exact meaning.

The generation-tagged numeric guard lives inside existing Budget/Lease. Source
refunds cannot credit numeric allocations; mixed shrink is conservative, stale/
nested guards refuse, and actual NumPy base owners keep charges after session
close. Private positive cases cover all3Model4 kinds, plain/packed complete
originals, partial/all/one/empty decisions, every regenerated numeric array,
authorization faults, source mutation/cancellation, pruning, quotas and views.

Independent correctness runs20selected private-release FFI cases,5guard and3held-
original/live Rust cases, scoped static checks and package hashes. Architecture
inspects phase/math/ownership boundaries and independently rehashes binaries.
Root fresh private release command
`GFB_HPO_NATIVE_CANDIDATE=/tmp/gfb-hpo-native-wheel-extracted/gf_native/gf_native.cpython-314-x86_64-linux-gnu.so
.venv/bin/python -m pytest tests/test_hpo_native.py -q -k
'public_hpo_admission or three_models_partial or view_keeps'
--disable-warnings --maxfail=1 --tb=short` passes5cases in25.17s.
Root `cargo test --manifest-path native/Cargo.toml --lib --offline` with the
documented test-only libpython link flags passes172checks/7explicit opt-in ignored
in39.08s. Root full project Ruff passes; full mypy passes188source files.

Private wheel SHA7295d1c779f80a2d550ab7590d68df4fcbbe7665f04b0b2f912ea666cbd9cb79;
ELF SHA8164d5c07969e95d7a501ced62e31cdecd618dde1080b4b85ad79f5218ee9e85;
embedded source5db7cf1519722ba5df1aaac9b5a806cce1e6b0d5374720dfb5b38532d993d4b0.
RECORD/stub methods/Cargo.lock/profile/allocator/current source checks match.
The private capability is exact; the installed old extension remains
SHA9b1377daac7c93c97a4f267d5f20c2204eb37401a53daa1275019614d1b6767f
with no HPO marker. No install or heavyweight old-data proof was run for this
scoped acceptance. Genuine integrated CUDA/native/live/common-loop/Simulator
and whole-HPO acceptance remain pending runtime4/5b integration and final freeze.
