# Resident HPO precollection validation reuse: verification

Status: accepted scoped implementation and measured verification,2026-11-16.
Both final independent reviews are FINAL READY; root accepts ADR0042 scope.
Implementation revision
`fd77c8648577198a6aedbb771e0affdd6dd1bf9e`. Baseline `050713d`; design milestone
`ceb82ef` records [ADR0042](../adr/0042-hpo-precollection-validation-reuse.md), the
[accepted design](../design/hpo-precollection-validation.md) and execution plan.
Independent preimplementation and final implementation correctness/spec/quality
and conformance reviews are READY. No formal HPO or strength acceptance follows.

## Scope and required behavior

Actual original historical validation is retained only across resident HPO's
first three precollection checks. Fresh ownership/learner/phase/plan/dependency
checks remain, evidence closes before mkdir, and later full validations remain.
Private runtime opt-in defaults false; standard resident CLI opts in. No schema,
policy input, learning math, native Rust, Simulator or dependency changes.

## Initial checks

Implementer baseline52PASS covers bootstrap/resident-contract/phase-limit/CLI;
RED12fails detects the absent carrier/private interfaces. Scoped conformance
review identified three blocking lifetime/accounting findings: enclosing scope
restart after expiry, closing an unrelated existing carrier after failed attach,
and temporary scratch refund uncertainty. The active-scope/prior-owner guard and
retained high-water scratch/actual zero-pool close resolve these; scoped production
conformance is READY for that snapshot.

Independent correctness review then found a fresh-guard omission for Python
preprocessing with mixed opponents. RED3/PASS1 reproduces successful stale reuse
while the original full validator rejects. The minimal fix moves the existing
mixed capability guard before the Rust backend gate, with scratch preadmission.
GREEN4 and the new44+Source/refresh/native subset PASS142/SKIP29 in68.54s.
Three further same-byte inode/cache/evaluation-namespace mutation cases PASS3;
new suite count47. Scoped correctness/spec/quality and architecture-conformance
re-reviews are READY with P2 addressed and no remaining blocking findings.

Other scoped evidence: affected coordinator/CLI134PASS/175.54s; resident custody/
lease/pool/native153PASS/29SKIP; mypy209PASS. Sandbox numeric CPU preload produced
20startup errors in a broader command; unchanged Source tests rerun HOST pass33/
skip10 in60.51s, without a fixture bypass. Root all-tracked/new Ruff507PASS is
repeated successfully on the final Python mixed-guard fix and47-case test file.

Root intentionally interrupts wholeCPU01 for the reviewed fix. The interrupted
run includes KeyboardInterrupt/pytest temporary-fixture teardown error and is
not a passing suite. WholeCPU02 stops at1FAIL/4021PASS/146SKIP/32warnings in576.90s:
an existing no-I/O `object.__new__(_HpoOperations)` test stub lacks the new lifecycle
defaults and raises AttributeError before its intended cleanup assertion.
The test-only fix initializes those constructor defaults without changing any
assertion or production guard. Project/evaluation-failure cleanup43PASS/28.34s;
independent test-delta review READY, all production hashes unchanged.
WholeCPU03 PASS6557/SKIP408/32warnings in1662.78s, exit0; XML6965cases/0failures/
0errors. It runs frozen reviewed source/tests on HOST with CUDA disabled,
preserving ordinary collection/fixture semantics. The existing slow Tree fixture
produces a60s faulthandler diagnostic; execution continues and the suite passes.
Earlier attempts are retained separately and never relabeled as passing evidence.

```console
CUDA_VISIBLE_DEVICES='' GF_RUN_CUDA=0 .venv/bin/python -m pytest -q --ff --maxfail=1 --tb=short -o faulthandler_timeout=60 --junitxml=runs/hpo-resident-workers/assessments/precollection-v1/checks/whole-cpu-03.xml
```

Root verifies final9source/test hashes, all24protected historical hash/stat
observations, exact209module wheel/sdist/source Python bytes, and the existing
8frozen numeric kernel AST verifier. Offline package build makes no dependency or
installed-environment change. Checks/logs are under `precollection-v1/checks/`.

The ignored helper's stdlib tests pass original-once sync/async delegation,
primary outcome preservation, context/patch restoration and injected start/
finish/metadata/output failures. Independent helper prelaunch source review is
READY; those checks create no actual job or Source authority.

Root independently verifies all24protected historical files' hashes and complete
size/device/inode/mtime/ctime observations against the existing protection list.
New snapshot is under `runs/hpo-resident-workers/assessments/precollection-v1/`.
Pre-existing user untracked files remain outside task staging.

## Physical comparison protocol

Preparation-only declaration and3configs are under the ignored
`runs/hpo-resident-workers/assessments/precollection-v1/`; new run/storage roots
are `runs/hpo-precollection-validation/v1/`. No previous producer data is reused
as a new run. Freeze candidate source/native/Simulator/schema identity, actual
helper hashes and config before admission; declare the private toggle explicitly.

Use identical candidate code/settings with `_reuse_precollection=False` and
`True`, learning seed17, configured192/main2,4physicalworkers/2concurrentgames,
16games per collection and2updates. Preserve the existing D1000000/max_updates2,
600s run wall, epoch1, cache6/128MiB, parent/save1GiB and other original bounds.
The expected outcome is healthy `incomplete/update_cap`, not a completed HPO
objective. Ten runs are declared: MLP B1/C1,C2/B2,B3/C3; GRU C1/B1; Transformer
B1/C1. Do not run CPU suites/audits concurrently with timing.

Thin timers call original functions exactly once, preserve returns/exceptions
and restore wrappers. Measure full validation, retained recheck, close and collect
separately; inclusive nested spans must not be added. No global profiling.
Each accepted run needs its actual original termination receipt, expected
4worker/1learner retirement and empty reference maps, followed by independent
complete-original Source/numeric reopening. Preserve failures separately, with
their actual cleanup status; no JSON/PID-derived authority or deadline extension.

Require first-three full replay3->1, original later full checks, unchanged budget
enforcement/cleanup and lower median MLP second-collect wall time. GRU/Transformer
single pairs are functional/timing checks without distributional claims. Timing
overhead and cutoff/trajectory effects remain; seeds alone do not guarantee
bitwise training determinism. No formal HPO, strength or final-test evidence.

## Actual physical evidence

The initial device was busy (up to14896MiB/71%utilization), so no job was admitted
then. After the user's GPU-free confirmation, HOST observations show
RTX5070Ti16303MiB/driver610.88,13283MiBfree/3%utilization. Root rechecks HEAD,
382frozen source/config/script files, all3runtime/native/Simulator condition sets,
and all10new report paths before launch. The3profiles remain within the original
24GiB RAM/12GiB CUDA managed limits; no budget is raised. Frozen native ELF is
`0d5d0b242b1a08aafa22039f6c7c5921f2555cfcbf7c801c899e14573054aeda`, native source
`c7a357060c26661a23f54cf9de4bfa1bc914694400601b70cc8a573eee5a54cc`, Simulator
`91d5926a7572db327eaea3751e7c24f428120996`, public-ja-v3/catalog2.

All10declared trials execute in their original order on the same implementation
revision/settings: MLP3pairs, GRU1pair and Transformer1pair. All return exit0,
healthy `incomplete/update_cap2`; MLP/GRU use314decisions and Transformer310.
There are320naturally completed games in total. Every original production receipt
confirms4workers+1learner reaped (40workers+10learners total), and every final
cache/evaluation-reference map is empty. No failed trial, operator release,
recovery, budget reduction or deadline extension is used. Remaining-trial
pre-start observations show13115–13124MiBfree and2–3%utilization; no timing run
overlaps a CPU suite or full Source audit.

The first MLP pair passes actual Source/numeric/receipt audit before continuing.
After all trials, `all10-audit.json` independently reopens every complete original
Source and stored numeric rollout using the normal production reader and original
2GiB Source cap. All10audits PASS with exact unchanged run/registry hash/stat
snapshots and original post-run SHA/deadlines. The aggregate is
`final_aggregation_ready=true`. No receipt, Source/update or qualification
authority is reconstructed from the analysis JSON.

Raw timers prove baseline first-five full counts `[1,1,1,1,1]` and candidate
`[1,0,0,1,1]` in BOTHgenerations, with ordinal6 outside collection. Each candidate
has exactly one ready->closed transition inside call3, followed by an idempotent
closed->closed cleanup call. Actual detach/scan-close/zero Source bytes are
recorded; G1 additionally has actual pool closed/used0 (G0 has no original pool).
Timers/diagnostics have0drops/errors/unavailability. Independent raw-metadata
verification confirms these transitions/counts for all10runs.

## Timing results

MLP values below are medians of3baseline/candidate trials, with all3declared pairs
retained. GRU/Transformer are each1pair and descriptive only. "Whole" is the
original helper's elapsed time through original job cleanup, excluding its final
result serialization and interpreter startup; external GNUtime is retained
separately. "Second collect" is the original collect->source phase interval,
including coordinator call1, not just the shorter collector callback.

| Model | Pairs | Second collect before -> after (s) | Reduction | Whole before -> after (s) | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 3 | 40.840 -> 30.082 | 26.34% | 119.429 -> 107.762 | 9.77% |
| GRU | 1 | 41.238 -> 30.391 | 26.30% | 158.165 -> 146.938 | 7.10% |
| Transformer | 1 | 41.628 -> 30.712 | 26.22% | 159.987 -> 149.044 | 6.84% |

MLP individual second-collect pairs are40.840->30.082,40.971->31.158,
40.826->29.822seconds; each improves. Complete full-validation G1 totals have
median27.396->16.461seconds (39.91%less), consistent with5->3complete replays.
Candidate G1parent-recheck inclusive median is0.117s and actual close0.000255s;
these nested labels are **not** added to the full/phase totals. First-collect
MLPmedian18.550->18.267s is retained as well; the historical G1case is the main
declared opportunity.

These bounded2update observations satisfy the declared MLP median/call-count
acceptance without predicting a fixed saving for larger histories. Initial
warmup, observer overhead, timed cutoffs and training nondeterminism remain
limitations; no randomized observer-overhead study or strength/final-test/HPO
objective evaluation is performed. Keeping calls4/5and Source issuance fresh
continues to incur historical validation work.

## Integrity, review and local evidence

Root's postmeasurement check passes all382frozen files, HEAD/worktree consistency
and24protected historical content/size/device/inode/mtime/ctime observations.
Source, math, native and configuration files are unchanged through timing and
audits. Root makes no edits to the pre-existing user files; they remain outside
task staging. Only tracked documentation is updated after those checks.

Ignored evidence root: `runs/hpo-resident-workers/assessments/precollection-v1/`:

- `declaration.json`,3configs,`code-freeze.json`,`analysis-freeze-v2.json`,
  `launch-ready-01.json`; prepared records are not admission authority.
- Original10report folders, stdout/stderr/GNUtime files and
  `remaining-batch-events.jsonl`; original requests/results/actual termination,
  diagnostics and thin timer rows remain immutable.
- `pair01-audit.json`,`all10-audit.json`,`final-measurement-summary.json`,
  `postmeasurement-protection.json`; summary-only precursor remains unqualified.
- CPU/static/package/AST evidence under `checks/`.

Analysis preparation includes62stdlib synthetic checks. Independent review fixed
one analyzer gap by binding actual run SHA and all4saved deadlines to the probe's
original after observation before auditing. Actual first-pair and all10bundles
then pass without schema adaptation.

Both final independent reviews are FINAL READY with no blocking findings:
`bot_architecture_reviewer` confirms design boundaries and declared acceptance;
`bot_reviewer` independently recomputes all model statistics from the10audited
rows and checks actual fixed-namespace receipts, current controls, original
post-run hashes/deadlines, raw transitions,320games/20updates and per-model
settings after UUID normalization. Reviewers rerun no tests or workloads. Root
accepts the implemented and measured scope, preserving the stated limitations.
