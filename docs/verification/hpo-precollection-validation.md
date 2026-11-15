# Resident HPO precollection validation reuse: verification

Status: Python implementation/CPU milestone verified,2026-11-16; physical
comparison and overall measured acceptance pending. No GPU performance acceptance
is claimed. Baseline `050713d`; design milestone
`ceb82ef` records [ADR0042](../adr/0042-hpo-precollection-validation-reuse.md), the
[accepted design](../design/hpo-precollection-validation.md) and execution plan.
Independent preimplementation and final implementation correctness/spec/quality
and conformance reviews are READY. Physical evidence review remains required.

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

## Remaining physical acceptance work

- Bounded paired physical runs, original Source/numeric/receipt audits and
  independent measured-evidence acceptance.
- Postmeasurement protection/identity checks and final documentation acceptance.

The3model static/runtime-evidence preflights pass without device allocation,
job admission or gameplay. They pin the existing native and Simulator contract
and fit the original24GiB RAM/12GiB CUDA managed limits. The current device is
shared with external activity: early observation13838/16303MiB and95%utilization;
later8198MiB and21%. No physical comparison is launched while availability/fairness
is unresolved; the user confirms another GPU task is running and will free it
before measurement. Prepared declarations
and preflight JSON are not launch approval or actual termination evidence.
