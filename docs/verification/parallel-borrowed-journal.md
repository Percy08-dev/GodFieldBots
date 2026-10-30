# Bounded parallel borrowed-Journal verification

[ADR0018](../adr/0018-bounded-parallel-borrowed-journal-parsing.md) has independent
pre-implementation architecture READY. Implementation and independent source
correctness/conformance reviews are READY; execution gates below pass.
The48-command controlled comparison and independent measurement audit pass.

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

All48commands pass:8warmups plus5measured trials for each of8cells. The measured
code freeze is clean commit`ad605c9`,117code files. Every command reproduces the
frozen arrays/NPZ/sample/dataset identity and actual selected binary. The final
29,814source/24protected-fit and code checks pass. Raw plan, conditions, complete
observations, summary and actual-median stage extraction are under`primary/`.
No builds/tests/source edits run during primary timing.

| Workers | Serial wall median [range] s | Parallel wall median [range] s | Wall reduction | Serial / parallel CPU median s |
| ---: | ---: | ---: | ---: | ---: |
| 1 |25.139 [25.059,25.248]|25.426 [25.404,25.615]|-1.14%|25.001 /25.278|
| 2 |25.226 [25.163,25.262]|23.190 [23.051,23.291]|8.07%|22.322 /17.310|
| 4 |23.595 [23.531,23.872]|19.786 [19.621,19.943]|16.14%|22.350 /17.459|
| 8 |22.625 [22.547,22.840]|18.017 [17.862,19.099]|20.37%|22.559 /18.491|

The1-worker path is1.14%slower. Eight workers give the shortest median in this
corpus, with more peak memory. CPU is Linux wait4-reported workload-tree user+
system CPU, excluding the observer. It is neither wall time nor device usage;
these totals do not identify which function used or saved CPU time.

### Memory and storage

| Workers | Serial / parallel maximum kernel peak RSS MiB | Serial / parallel median post-release RSS MiB | Serial / parallel median post-release PSS MiB |
| ---: | ---: | ---: | ---: |
|1|348.32 /378.25|299.89 /301.81|291.51 /293.09|
|2|357.40 /381.61|298.97 /294.06|290.59 /286.08|
|4|365.16 /420.44|296.79 /293.53|288.37 /285.19|
|8|407.98 /501.64|298.99 /296.05|290.60 /287.87|

Kernel peaks are the maximum of five workload-interpreter self `VmHWM` values,
read in its post-release probe. They include Python/imports and preprocessing,
not only native-managed bytes or the complete tree. Post-release follows owner
drop and explicit GC; interpreter and allocator-retained pages remain. Sampled
tree RSS is separate; exact per-trial peaks, PSS availability and raw snapshots
are retained. In8workers, peak RSS increases93.66MiB, while post-release RSS is
similar. All use the unchanged2GiB Config soft budget; it is not an OS hard quota.

Native-owned logical scratch high-water is exactly1,856,048,056bytes for every
measured variant/worker case. The native8GiB scratch limit is unchanged. The
observer's sampled allocated output maximum is about1,783MiB, varying with its
1s sampling and file lifetime. These scopes overlap and must not be added.
The final NPZ is byte-identical at463,153,426bytes. No storage improvement is
established by differing sampled peaks.

Recorded median `write_bytes` is1,943,023,616 for every cell; recorded `rchar` is
about5.716GB. These are sampled process-I/O lower bounds, not physical device or
WSL-host traffic. Small captured `read_bytes` is consistent with warm caches but
cannot establish negligible storage reads or I/O-wait duration. Profiled-child
I/O coverage can be incomplete, as in the earlier diagnostic. Raw source reads,
native logical scratch, sampled allocated storage and device traffic are distinct.

### Exclusive elapsed attribution

Each column below is its actual median-whole-wall trial, not independent stage
medians. The old Journal row includes parsing and validation; the new combined
row adds parse-window dispatch/join to post-parse validation. Both are comparable
elapsed scopes. Worker elapsed counters remain nested and are excluded here.

| Exclusive elapsed stage | Serial4 s | Parallel4 s | Serial8 s | Parallel8 s |
| --- | ---: | ---: | ---: | ---: |
| Read/Replay decode wait and ordered source merge |3.002|2.874|2.001|1.922|
| Journal parse plus validation |10.145|6.323|10.127|5.435|
| Features and numeric spooling |4.521|4.485|4.500|4.443|
| Temporary release |1.468|1.678|1.509|1.802|
| Other native setup/facts/proof/metadata/global checks |1.571|1.542|1.591|1.570|
| Enclosing preparation outside native scan |1.517|1.489|1.506|1.446|
| Archive staging and proof |0.736|0.760|0.756|0.761|
| Final source check and cleanup |0.499|0.495|0.497|0.499|
| Outer measured-envelope residual |0.137|0.138|0.138|0.138|
| **Whole command** |**23.595**|**19.786**|**22.625**|**18.017**|

New parse-window dispatch/join is4.040/3.082s at4/8workers; subsequent validation
is2.283/2.353s. At1/2workers the new parse-window scope is8.325/5.921s and
validation2.255/2.306s. All11old/12new native partitions and enclosing Python
integer ledgers reconcile exactly; display rounding can affect the printed sum.
Outside-native preparation is envelope subtraction, not pure Python CPU; the
outer residual is not an independently measured startup/GC phase. Read/decode
wait includes execution/dispatch/join/merge and is not OS I/O wait.

Actual active parse-job peak is1/2/4/8, with6,784successful episode jobs and zero
failed/remaining active jobs. One worker spawns no parse thread. Each multiworker
case spawns6,784parse threads plus6,784existing decoder threads over its bounded
windows. Job overlap is not simultaneous CPU utilization. Median reported CPU
divided by median wall is about0.99/0.75/0.88/1.03cores for the new cases; ordered features,
validation and window barriers still limit whole-workload parallelism. Summed
parse-job elapsed is8.322/8.602/8.841/9.881s in those representative trials and
overlaps across workers; it is not additional elapsed time or measured CPU work.
Persistent worker reuse or overlapping windows remains a separate change requiring
measurement; counts alone do not quantify the time attributable to thread creation.

### Separate function profiling

Artifacts: `runs/performance/parallel-borrowed-profile-20261030/`. The separate
release/opt3/debug2/strip-none ELF`d4913a3b...` has the same accepted native source.
All three diagnostic commands preserve original numerical/data identities.
User CPU sampling uses perf6.8.12/499Hz/DWARF65,528bytes, offline max-stack512,
inline frames and matching symbols. Speedscope and raw/demangled/folded/function
outputs are retained; no external upload is required.

| Function, inclusive sampled user CPU,1worker | CPU s |
| --- | ---: |
| Typed `journal::parse_admitted` |8.164|
| Its `Record::deserialize` descendant |6.214|
| Reserved-member scan within typed parsing |1.263|
| Typed `journal::validate` |2.236|
| Legacy `json::parse_admitted` for Replay/control |3.345|
| Typed `PreparedDecision::new` |1.639|
| Typed feature `calculate<NumericSink>` |1.497|

These estimates include descendants, overlap and cannot be added to the elapsed
table. Innermost inline-frame self and physical-frame self can differ. Unresolved
vDSO self is2.411s for1worker and2.313s for4workers; callers remain unassigned.

The1-worker trace has10,893samples,21.830sampled user CPU s versus21.822observed
user CPU s (+0.035%). The4-worker trace has6,676samples,13.379sampled user CPU s
versus14.662observed (-8.75%). Only1,339thread identities receive samples versus
13,568spawned decoder/parse workers. Short-lived workers and unwinding make
individual4-worker function weights unsuitable for claiming a reduction in parse
work or a function speedup. They remain raw diagnostic evidence. The primary
elapsed/CPU/kernel-peak tables are separate unprofiled measurements.
Observed diagnostic user CPU also includes the perf recorder; sampled CPU covers
the recorded workload threads, so the two scopes are not exactly identical.

Warm symbols control wall is25.573s; deep profiles take25.850/20.096s at1/4workers.
The single1-worker capture is about1.08%slower than its control, an approximate
instrumentation observation without paired-trial variance. No profiler resources
are substituted for the primary resource results.

## Cost and acceptance

Primary48-command wall sum is1,103.604s and reported CPU sum1,026.880s; the
enclosing matrix takes1,106.995s. Its pre/post source validation takes22.778/
3.032s. Separate correctness setup includes22.895sinitial freeze,9.990srelease
build,397.37sfull regression and complete corpus/index/PPO checks. The11.644s
symbols build and three captures (71.518scommand-wall sum) are profiling cost.
Those correctness/setup jobs can overlap and must not be summed into a single
end-to-end wall duration. Interrupted sandbox diagnostics are retained separately,
excluded from primary speed evidence, and did not produce training data/models.
No source generation, games, fit or optimizer cost is incurred by this comparison.

Independent measurement audit reproduces all48conditions/proofs/statistics,
actual-median selections, resource scopes and zero-residual ledgers. Source and
architecture reviews are READY, with the explicit optional coverage item above.
The change establishes faster complete preprocessing with multiple workers in
this pinned workload, at increased peak memory. Defaults/main installation remain
unchanged. Closing independent correctness/report/profile and architecture
conformance reviews are READY, with no unresolved findings. The optional recorder
CPU-scope clarification was incorporated above. Acceptance is complete.
