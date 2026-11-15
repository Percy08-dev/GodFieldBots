# HPO resident workers and progress verification

2026-11-15. [ADR0041](../adr/0041-hpo-resident-workers-and-progress.md),
[accepted plan](../design/hpo-resident-workers.md). Implementation and final
acceptance IN PROGRESS; no performance/strength result is preaccepted.

## Starting evidence

- Baseline71fbe49; accepted design/docs commit744feb6. Conversational and persisted
  architecture reviews READY, `/root/resident_inference_design_check`.
- `.venv/bin/python -m pytest -q tests/test_hpo_config.py tests/test_hpo_gpu_lease.py
  tests/test_hpo_retention.py tests/test_hpo_cli.py --tb=short`: PASS170/12.66s,
  exit0; no GPU/Simulator execution.
- CPython3.14.10/GIL, NumPy2.5.4, Torch2.14.1+cu132, Optuna5.0.0, native0.1.0.
  Read-only RTX5070Ti16303MiB/driver610.88 observation showed1891MiB in use;
  no model allocation or hardware-exclusivity claim follows.
- Simulator HEAD91d5926a7572db327eaea3751e7c24f428120996 and expected prebuilt
  executable exist; runtime will recheck source/spec/binary correspondence.

## Predeclared actual checks

Use fresh assessment/parent/campaign/DB roots under runs/hpo-resident-workers/.
Preserve old evidence/failures. Freeze actual source/config/native/package/
Simulator/hardware identities and exact commands before launch. No project
edits/tests/builds during timing; outside desktop/device load remains a limitation.

1. All3models: agreed widths192/main depth2; within each fresh/resident pair use
   the same seed17, recipe, opponents, action conditions/budgets/original deadlines.
   Explicitly record new representation/config/runtime identities.
2. Timing probes: D1,000,000/max_updates2, EXPECT incomplete/update_cap/no objective.
   Run3pairs/model in AB/BA/AB order. Include startup/reset/artifact checks/save/
   sync/retention/cleanup in total timing; measure physical/logical starts/loads,
   allocated/reserved VRAM and separate device-wide observer samples.
3. Separately pause1/resume2 with unchanged source/config/clocks; reap on pause,
   rebuild on resume. Compare arrays/schedules/state where applicable without
   claiming seed-only determinism.
4. Complete D64/seed17 CLI campaigns include canonical control and fixed proposal,
   paired heuristic evaluation and actual Optuna tell. Neural direct comparisons
   exercise both sides. Distinguish natural/truncated/aborted/infrastructure results.
5. Independently validate Source/numeric/manifest/pins/lifecycle/GPU receipts.
   Inject lost ACK/partial sync/reset/cancel/reference/retirement failures; never
   qualify failed data or release unconfirmed resources. Unsupported native refuses.
6. Observer tests cover held locks, readonly/missing/bad/stale snapshots,
   optional-free CLI, stdout, bounds/ETA/pause and diagnostic-write failure.
7. Fixed-input all3model equivalence covers episodes/checkpoints/A-B-A. Historical
   formats/8kernelAST/native/Source proofs and whole Python/Rust/static checks pass
   before final acceptance. Record explicit skips and environment limitations.

Candidate-generation wall cutoffs/host scheduling can change candidates despite
equal seeds. Record cutoffs; fixed-input policy parity is separate. Never silently
change actions to force equality or pool failed attempts into accepted timings.
No actual execution rows or final implementation reviews are complete yet.

## Task2 scoped runtime acceptance

Real CPU component fixtures exercise all3model fresh/resident equivalence,
read-only artifacts/A-B-A/partial load, actual subprocess setup/I/O ownership,
packed/plain Source→Wire6 update/commit→pin/SYNC through G2 with frozenG0, and
actual evaluation collector producer/runtime cross-products. Their one explicit
CUDA/data qualification bypass remains UNQUALIFIED engineering evidence.

Broad affected run812PASS39SKIP96.28s precedes final narrow refinements. Gated
setup9PASS, lifecycle44PASS, actual two-update/eval5PASS cover later transport
changes; native candidate50PASS1SKIP plus focused4G0 cases overlap existing cases.
No Rust source changes or new release build in this task.

Independent initial reviews found4blockers: incomplete parent-pin coverage,
mutable physical execution visibility, acquired refs untracked on observation
failure, and sealed normal-termination replay rejected after local publication
failure. Each was reproduced and fixed. Final specifically affected command:

```console
GF_RUN_CUDA=0 UV_CACHE_DIR=/tmp/gfb-uv-cache uv run pytest tests/test_resident_episode.py tests/test_resident_pool.py tests/test_resident_training.py tests/test_resident_evaluation.py tests/test_hpo_resident_custody.py tests/test_hpo_eval_failure_cleanup.py tests/test_hpo_gpu_lease.py tests/test_hpo_retention.py -q
```

PASS173/no skips78.92s, one existing fork-from-multithread deprecation warning.
Source mypy206/scoped Ruff/diffPASS; both scoped independent fix1 re-reviews READY.
Root accepts this scoped implementation, not actual CUDA/Simulator/performance
or final learning acceptance. Task3 and actual verification remain pending.

## Task3 scoped progress acceptance

Bounded latest snapshots, stderr auto/plain/off and lock-free optional-free watch,
actual committed/phase/learner/pool callbacks and observed allocator channels are
implemented. Product does not measure simultaneous whole-device peak; unavailable
values remain unavailable. Raw restored float32 payload counts are not PCIe bus
overhead; per-process last-sync samples/maxima are not summed as device peak.

Covering run192PASS183.74s precedes final narrow refinements. Subsequent changed
67/CLI50/late-save actuallearner19 checks overlap. Initial independent reviews
found nested nonfinite JSONwatch crashes, unbounded historical worker rows and
optional heartbeat setup escaping into execution. Root also adopted full inner
protocol3 validation before publishing pool facts. All reproduced/fixed; final:

```console
UV_CACHE_DIR=/tmp/gfb-uv-cache uv run pytest tests/test_hpo_progress.py tests/test_resident_pool.py tests/test_hpo_cli.py tests/test_hpo_campaign.py tests/test_hpo_coordinator.py::test_progress_observes_actual_commits_and_cleanup_without_authority -q
```

PASS69/no skips27.44s, seven existing Optuna experimental warnings; scopedRuff5/
mypy208/diffPASS. Both scoped independent fix1 re-reviews READY; root accepts
progress implementation only. Actual Source/learning with writer OSError is
healthy under explicit unqualified CPU fixtures; no GPU/game/quality inference.

## Acceptance

Task1 foundation reported418PythonPASS/27explicitSKIP, final affected125PASS,
source mypy201/Ruff/old-contract hash comparison/8AST checks PASS. Rust HPO subset
11PASS/3IGNORE plus explicit resident-original reader1PASS. These are scoped
foundation checks; pool/progress execution is still guarded off.

Root built the frozen private source-matched release/MIMALLOC/emptyRUSTFLAGS
wheel offline with Maturin1.15.0 in19.49s. Wheel SHA
814538268075f04d007e8c5f9867476601634cf63b8c889efd1e0b948226cae1,
ELF SHA0d5d0b242b1a08aafa22039f6c7c5921f2555cfcbf7c801c899e14573054aeda,
native sourcec7a357060c26661a23f54cf9de4bfa1bc914694400601b70cc8a573eee5a54cc.
Both old/new HPO capability markers, release profile, mimalloc and empty rustflags
were checked on the actual ELF. It remains a private candidate, not installed.

Root freshly ran:

```console
GFB_HPO_NATIVE_CANDIDATE=/tmp/gfb-resident-task1-site/gf_native/gf_native.cpython-314-x86_64-linux-gnu.so .venv/bin/python -m pytest -q tests/test_hpo_native.py tests/test_hpo_resident_contracts.py tests/test_hpo_resident_custody.py --tb=short
```

PASS71/46.91s. No GPU model allocation/gameplay or performance claim follows.
Both independent Task1 reviews are NOT READY pending fixes for tracked-I/O reuse/
termination and original-parent-lock cache API ownership. Code is not accepted;
the passing suites did not cover those independently reproduced failures.

Task1 fix1 adds10meaningful RED→GREEN regressions; final affected165PASS/no skips
in15.52s, scoped Ruff and source mypy201PASS. Both independent scoped re-reviews
are READY: tracked I/O blocks reuse/termination until explicit original retirement,
and cache transactions reuse the original coordinator lock. Old fresh branches
remain. Root accepts this guarded foundation only; actual pool/reset/readiness/
evaluation integration, progress and final hardware acceptance remain pending.

Append exact commands/results, failed attempts and independent correctness/
conformance reviews. Root accepts after blocking findings resolve and required
checks pass. Formal HPO sizing/model quality/final-test evaluation remain later.
