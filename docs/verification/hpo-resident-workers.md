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

## Whole-project and package checks before hardware probes

Installed reviewed offline native release/MIMALLOC/emptyRUSTFLAGS ELF matches
0d5d0b...54aeda and sourcec7a357...a54cc, with old/new capability markers checked.
Original installed44a13f...bb9f77 package/dist-info rollback remains in
`runs/hpo-resident-workers/assessments/native-before`.
Python wheel/sdist match all208source modules byte-for-byte. Independent
mandatory-only venv has only base dependencies: CLI/watch and all3config/storage/
admission checks PASS without Torch/Optuna/native. Its actual baselinevsRandom
pair completes2natural games/0failures in4.901s, functional only.

Whole CPU Python attempt1 FAIL:3959PASS146SKIP before an object.__new__ cleanup
fixture omitted new optional resident_runtime initialization. Verified production
constructor setsNone; fix is1test-onlyline, preserving the uncertain-cleanup
assertion. Affected73PASS55.43s and independent reviewREADY. The failed first
run/log/XML remain, not promoted into a pass.

```console
CUDA_VISIBLE_DEVICES='' GF_RUN_CUDA=0 .venv/bin/python -m pytest -q --ff --maxfail=1 --tb=short -o faulthandler_timeout=60 --junitxml=runs/hpo-resident-workers/assessments/final-checks/whole-cpu-02.xml
```

HOST/localIPC retry PASS6490/SKIP408/32warnings1220.74s. Skips are explicit
optional/CUDA/heavy engineering gates, not executed hardware proof. Tracked
Python Ruff503filesPASS, source mypy208PASS. Native full library with test-only
libpython linking flags PASS177/IGNORE8/0FAIL35.84s; installed release flags stay
empty. Final checks logs and package proof are in assessment final-checks.

Protected historical model24files match all old baseline hashes; before snapshot
records byte/stat identity for final comparison. Actual GPU/Simulator learning,
model comparison, allocator high-water and paired timing remain pending.

## Task1 chronological foundation evidence

Task1 foundation reported418PythonPASS/27explicitSKIP, final affected125PASS,
source mypy201/Ruff/old-contract hash comparison/8AST checks PASS. Rust HPO subset
11PASS/3IGNORE plus explicit resident-original reader1PASS. These are scoped
foundation checks. At that milestone pool/progress execution remained guarded
off; Task2 subsequently activated the explicit resident runtime.

Root built the frozen private source-matched release/MIMALLOC/emptyRUSTFLAGS
wheel offline with Maturin1.15.0 in19.49s. Wheel SHA
814538268075f04d007e8c5f9867476601634cf63b8c889efd1e0b948226cae1,
ELF SHA0d5d0b242b1a08aafa22039f6c7c5921f2555cfcbf7c801c899e14573054aeda,
native sourcec7a357060c26661a23f54cf9de4bfa1bc914694400601b70cc8a573eee5a54cc.
Both old/new HPO capability markers, release profile, mimalloc and empty rustflags
were checked on the actual ELF. At that milestone it remained a private
candidate; the later Task4 activation above installed the same reviewed ELF.

Root freshly ran:

```console
GFB_HPO_NATIVE_CANDIDATE=/tmp/gfb-resident-task1-site/gf_native/gf_native.cpython-314-x86_64-linux-gnu.so .venv/bin/python -m pytest -q tests/test_hpo_native.py tests/test_hpo_resident_contracts.py tests/test_hpo_resident_custody.py --tb=short
```

PASS71/46.91s. No GPU model allocation/gameplay or performance claim follows.
Initial independent Task1 reviews were NOT READY pending fixes for tracked-I/O reuse/
termination and original-parent-lock cache API ownership. Code was not accepted then;
the passing suites did not cover those independently reproduced failures.

Task1 fix1 adds10meaningful RED→GREEN regressions; final affected165PASS/no skips
in15.52s, scoped Ruff and source mypy201PASS. Both independent scoped re-reviews
were READY: tracked I/O blocks reuse/termination until explicit original retirement,
and cache transactions reuse the original coordinator lock. Old fresh branches
remain. Root accepted that guarded foundation only. Pool/reset/readiness/
evaluation integration and progress were later implemented in Tasks2/3; final
hardware acceptance remains pending.

Append exact commands/results, failed attempts and independent correctness/
conformance reviews. Root accepts after blocking findings resolve and required
checks pass. Formal HPO sizing/model quality/final-test evaluation remain later.

## Task4 current-source legacy audit

Independent verifier reports all5PASS on frozen2eae4c8/nativec7a357/ELF0d5d.
Corpus history12.613s/observation12.022s match the full expected projection and
all numeric arrays. Later nested finalization reports cleanup complete,
active0/admitted-completed-committed6784/failed0/four pool threads exited; the
explicitly pre-finalization data-stage snapshot remains historical pending.
Index19.051s matches83783rows and exact Parquet/dataset/content hashes. HOST
PPO readers1/4 complete in1.377s/1.268s, match3reader array proofs,16episodes/
224decisions/G0/zero optimizer updates; all72original source stat/hash identities
remain unchanged. Only declared diagnostic-clone manifest paths are rewritten.
Source320/320hashes match. Exact commands and full expected/result audits:
`runs/hpo-resident-workers/assessments/legacy-proof-a01/result-summary.json` and
`runs/assessments/hpo-resident-legacy-ppo-host-a03/*.result-check.json`.

Preserved nonpasses: obsolete explicitELF corpus argv NOTRUN, first PPO output
root guard rejection, unsupported default-sandbox asyncio interruption130, and
HOST retry collision with that partial directory. Corrected uniqueHOSTa03 is the
accepted PPO evidence. No old output overwritten, optimizer/GPU work or games.

## Initial hardware attempts (not matched performance acceptance)

First direct-helper a01 stopped before admission: diagnostic evidence directory
was incorrectly passed as the campaign storage namespace. The ignored helper
now passes the genuine configured parent/run namespace to the unchanged guard,
records both paths separately, and retains originalhelper/request/result. Real
storage RED→PASS and scoped independent review READY; corrected helper SHA
4012dc3dbc9f4ca14533572b819c6090c9b05f5349fad8c6b068423f10816ae7.

MLPfresha02 actual CUDA/Simulator run satisfies the declared diagnostic stop:
incomplete/update_cap,2updates,309used/collected,144.906s internal. The actual
original termination receipt verifies64worker reaps and1learner reap; last learner
process-lifetime allocator maxima allocated103799808/reserved121634816bytes.
Fresh actors expose no load/cache/allocator facts. This is functional evidence,
not a matched new-source timing or learning-quality result.

First residentMLPa02 fails after admission but before parent/model/learner/worker
creation. Bootstrap validation constructs a transient hpo_spec state without
run_variant; the strict reader consequently rejects schema3/runtime. Narrow
production-entry correction now passes scoped review; persisted formats stay strict.
Admission39dfd3df76ed4af980ca28d6573e0259 has no normal termination receipt.
Original invocation exited and actual Popen.wait was observed; this does not
manufacture termination. Explicit user approval for existing admission-only
operator release is pending. All failed evidence remains; no further GPU jobs
or recovery mutations occur before approval. Matched measurements will restart
with fresh identities after the fixed source is accepted/frozen.

Real-preset regression preparation also found the default512MiB additional-save
allowance too small for192/depth2GRU(638891136bytes) andTransformer(686931840),
whileMLP requires520242624. Root explicitly sets1GiB in all3functional presets
and all6new matched probe profiles (`probe-configs-v2`); model, decision/update/
wall/session budgets and other learning/action settings stay unchanged. Old
profiles remain intact. Genuine materialization of control+proposal for9profiles
PASS18cases, maximum686931840bytes; resident maximum managed phases RAM
13007871232bytes(12.11GiB), CUDA9518210304bytes(8.86GiB) within24/12GiB limits.
This is conservative admission evidence, not actual device peak.


The bootstrap correction supplies the existing explicit resident variant only
in schema3 transient validation context. Exact old method vs genuine corrected
presets reproduces6failures; new15cases pass for all3models/old1/2/hybrids and
strict persisted discriminator refusal. Covering reader/config suite PASS167,
no skips20.12s; mypy208/scopedRuff/diffPASS. Independent correctness review READY.
No cleanup/recovery or numerical/native change. Source whole6490CPU passes
precede this correction;167covering checks are its subsequent verification.
Post-fix Python wheel+sdist match all208modules byte-for-byte, proof
`final-checks/python-package-proof-v2.json`. Protected24historicalfiles retain
all content/size/device/inode/mtime/ctime identities after initial probes,
`assessments/protected-after-bootstrap.json` (2.063s). Remaining hardware and
final full implementation acceptance are pending, not implied by these passes.


## Current implementation review and remaining acceptance

Whole implementation71fbe49..01e5b7e independently READY for code correctness
and architecture conformance, with no blocking code findings/unresolved design
decisions. Correctness independently ran range `git diff --check` PASS; neither
review ran additional suites/hardware or granted recovery/data rights. Historical
Task1 guarded/private-candidate wording above and inADR0041 is now explicitly
past tense. Plan Task4 pre-hardware checks/package activation are complete;
actual resident all-model updates/resume/campaigns/comparisons, matched timings
and device-memory acceptance remain unchecked.

At that milestone the pending operator decision was admission-only release of
39dfd3df76ed4af980ca28d6573e0259. The user subsequently approved it and the
existing `hpo recover` command recorded release successfully. No original normal
termination receipt or data qualification was created; the new execution below
retains that distinction.
The failed parent was never created; prior requests/results and admission remain
protected. Prepared exact release command is in ignored assessment
`operator-release-39df-plan.json`; unused fresh21-row/24-invocation declaration
is `probe-declaration-v4.json`, pending post-doc launch freeze revalidation.
Read-only watch/product CLI usage is in [the HPO manual](../hpo.md). Formal HPO
search/model quality/final-test evaluation have not been launched or accepted.


## Actual a03 continuation and native payload correction

User approval for39df admission-only release received; command exit0 and receipt
`operator-release-39df-result.json` retained. All373source/profile6/helper/launcher
hashes and conditions match frozen5a364d0. Hardware RTX5070Ti16303MiB/driver610.88,
idle1976MiB/4% observed. No competing project checks/builds ran during timing.
Actual MLPfresha03 completes declared2update/incomplete/update_cap with verified
original cleanup in144.274sexternal. The nextresident run fails23.641s after
4actor starts/28physical reuses/16games, before the first committed update.
Native HPO packed payload projection omitted removal of Proof4.logical_retirement
from its temporary physical Proof2 view, causing the unchanged strict validator
to reject an extra field. Its original full Proof4 must remain intact.

Failedparent6e1f54be-cc30-4a0b-93e9-493c99c54ee7 remainsfailed/updates0/ineligible.
Actual invocation Popen.wait observed; cleanup records native cancellation and
uncertain executor/provider-original holds, so no normal termination receipt is
invented. User then explicitly approved admission-only releases for failed/stopped
jobs within this functional verification. Existing recover command1fd1ae48e26b4682b48c6dd1a84b25ad exits0, receipt
`operator-release-1fd1-result.json`; neither termination nor data rights follow.
The scope does not include unrelated jobs or reconstructing cache-release custody.

Original implementer reproduced the exact failure with installed-native packed
Collection7/plan6/provider/cohort conversion (FAIL1/7.44s, identical cleanup notes),
without conversion mocks. Approved narrow correction adds explicit Proof3/4
handling and validates new retirement/context while projecting only known HPO
fields for physical validation. Unknown/old hybrids stay strict and original
metadata/proof authority stays unchanged. Native all3modelG0→G1→G2 tests and both scoped independent reviews now pass;
all remaining hardware checks stay pending.
Failure measurements are not accepted timings. Batcha03 exited1 and owned NVML
sampler20377 was stopped/reaped. All attempts remain, no budgets shortened.


Native payload correction accepted after meaningful installed-native RED1 and
current-source all3packed16 G0→G1→G2 PASS3/48.84s, real recorded Proof3/4 mutation/
exact-array/retired-provider-hold PASS2/22.21s, neighboring compatibility checks
PASS64/2integrationdeselected39.99s. Source mypy208/scopedRuff/diffPASS. Both
independent correctness and architecture-conformance reviews READY, no blockers.
Only private Python physical projection changed; native ELF/Rust, full original
proof, numerical math and cleanup/recovery contracts remain exact. Whole6490CPU
checks precede this correction; new covering evidence is stated separately.
Root next executes declared D64 complete CLI campaigns/evaluation/watch before
fresh matched all-model measurements; no formal search/quality result follows.
