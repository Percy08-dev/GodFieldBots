# Bounded parallel borrowed-Journal verification

[ADR0018](../adr/0018-bounded-parallel-borrowed-journal-parsing.md) has independent
pre-implementation architecture READY. Implementation and independent source
correctness/conformance reviews are READY; execution gates below pass.
Controlled performance acceptance is pending.

The comparison changes only typed Journal parse placement. Root retains the
serial mimalloc candidate `bc455458...`/native source `37cae919...`; both variants
use `typed-borrowed-v1`. Default legacy/main installation and original sources/
models remain unchanged. Simulator revision remains
`ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6` with catalog v2/schema4 evidence.

## Evidence and conditions

Artifacts: `runs/assessments/parallel-borrowed-20261030/`. Before implementation,
`input-freeze.json` verifies all29,814 original source files and all24protected
original-fit files (including bytes/stat identity). It binds the retained serial
wheel, unchanged installed main native, source inventory and observer/workload.
Preflight validation takes22.895s and is setup cost, not a speed trial.

The full fixed history workload has83,783rows from415collections/6,784games;
13,568Journals contain534,640records. Compare serial/parallel at1/2/4/8workers
with max_in_flight equal workers, memory/cache/chunk2048/512/16MiB, one warmup
and five fresh-process trials per cell. Rotate worker groups between trials and
alternate variant order. No source jobs run concurrently, and no builds/tests/
source edits run during primary measurements. Correctness gates precede timing.

The disposable `primary.py` reuses the unchanged benchmark observer, workload,
strict environment sanitization and cleanup-owner helper. It checks frozen
numeric/archive/sample/dataset identities, actual loaded binary and typed route
counts in every trial, plus exact exclusive coordinator ledger reconciliation.
Source/protected-file and code-freeze checks bracket the matrix. Main/default
installation is not replaced; isolated PYTHONPATH overlays select each wheel.

## Executed correctness gates

Candidate native source is
`e604697d4982d3160c1d74fe242f05c9d2a11927166042942e4c82e229d8ccb7`;
release mimalloc ELF is
`5eba67105c051d1931066c98f1a88014ce04b0840cea06721a03b1df7003d939`.
`candidate-v1/variant.json` records isolated source/wheel/flags and9.990sbuild.
Separate release-symbols build takes11.644s; it is not a primary timing binary.

- Rust System full suite:68PASS/2ignored microbenchmarks; mimalloc library:
  37PASS/2ignored. Formatting and diff checks pass. Mypy src passes59files;
  changed-test Ruff passes. Relevant old-wheel Python checks pass11, with8new
  candidate-only checks skipped until the final candidate is selected.
- Full candidate Python regression with `GFB_TEST_PARALLEL_PARSE=1`:
  **826PASS/5existing conditional skips**,397.37s. This includes both readers
  at1/2/4/8workers, exact public arrays/order, retained read-only views, cleanup,
  independent parser/decoder counters and exact12-category wall reconciliation.
- Complete history and observation corpus: exact83,783-row arrays, NPZ,
  dataset/sample/source proofs and13,568typedfiles/534,640records. No exclusions
  or repairs. Parquet index: original83,783rows/content and file bytes match.
- Actual current-v2 sixteen-game saved PPO probe at1and4workers: Python, legacy
  and typed numeric/provenance match all original arrays. There are224learner
  decisions and8same-policy-IDepisodes covering learner seats1and2. No game,
  optimizer update, eligible parent or ordinary resume was created.

The first sandboxed PPO verification was interrupted at411.5s; a second diagnostic
was also stopped after identifying the environment cause. A minimal socketpair
send reproduces sandbox EPERM; stacks show the asyncio selector waiting while
the FFI worker is idle. The completion self-wakeup cannot be delivered there.
Those interrupted attempts are retained and are not candidate success/failure
or speed trials. Fresh permitted executions, matching the full regression's
environment, pass in1.402/1.392s. `verification/ppo-permitted-summary.json` and
their actual result files are authoritative; earlier `execution-summary.json`
retains the interrupted attempt. Application code was not changed to bypass this.

Independent source reviews have no blocking findings. An optional uneven
max_in_flight>workers case with failure on a worker's second assigned episode is
not newly fault-injected. The reviewed stride loop and inline/multiworker tests
cover the current equal-workers/window comparison. Add that targeted case before
recommending unequal worker/window tuning. Real spawn-refusal and mid-record
deadline code paths were inspected; current tests inject partial stack refusal
and preexpired deadline rather than those exact external events.

## Performance results

NOT RUN: complete before/after matrix and profiling.
Report median/range wall/CPU, kernel peak and retained RSS, available PSS,
storage/I/O scope/coverage and separate setup/validation/measurement costs here.
Stage attribution must distinguish the new typed parse-window wait from
post-parse validation; historical journal_validation includes both.

No throughput, memory or learning-quality improvement is claimed before those
measurements and final reviews complete.
