# Sequential preprocessing optimization

[ADR0019](../adr/0019-sequential-preprocessing-optimization.md) governs independent
changes and retain/reject decisions. User authorized sequential application and
measurement. No simulator changes, source rewrites, games/fits/learning updates or
automatic reader/worker-count switch. The historical main native stays unchanged
through all comparisons; final accepted-wheel installation is recorded below.

## Stage1: accepted single-pass typed history

Candidate source`9bd697d481c96af7e7f773a3697b374b719b96ecce920a7502fa8eadf370d4ec`,
release mimalloc ELF`8c9a2a19647d1f33c727542f5bd59b053c89d29fc2c955fb6244ad7151b7443a`.
Baseline source`e604697d...`/ELF`5eba6710...`. Artifacts are under
`runs/assessments/sequential-optimization-20261030/`.

Only typed history aggregation and its private numeric-block helper change.
Twenty-four type counts, actor counts/seven integer sums and definition buckets
are computed in one event traversal, with small stack arrays and checked existing
output storage. Tail selection, tag/output order, role precedence, SHA buckets,
f64 bits and prepared-state reuse remain unchanged. Entry/exit and event checks
remain. Illegal explicit-null actor fields stay rejected by unchanged validation;
feature-only adversarial fixtures follow the independent legacy precedence.
Legacy/Python encoders, names, reader/source schemas and global Control are untouched.

### Executed gates

- Full mimalloc Rust:73PASS/2ignored; focused feature cases6PASS. Cases include
  empty/truncated/max100000tail, all tags, actor priority/null contracts, optional
  values, hash collisions, integer bounds,2/3elements and2/32/4096buckets, cancellation,
  deadline and width/nonfinite refusal. Formatting/diff/Ruff/mypy59files pass.
- Existing-wheel Python oracles51PASS are separately labelled; final candidate
  full Python regression is826PASS/5conditional skips,403.93s.
- Full frozen history/observation NPZ/arrays/source/sample/dataset and83,783-row
  Parquet index bytes match original. Actual current-v2 saved16game/224decision
  PPO at1/4workers matches Python/legacy/typed arrays/proofs. No real game/update.
  These five sequential correctness jobs take65.846s and are not timing trials.
- Independent design, source correctness and architecture conformance READY.
  Independent12-command measurement audit PASS/zero findings.

### Equal-condition primary comparison

Both use releaseopt3/debug0 mimalloc,83,783fixed history rows,4workers/inflight4,
memory/cache/chunk2048/512/16MiB. One warmup and five alternating fresh-process
trials per variant, no builds/tests/source edits/heavy jobs during timing.
All12commands match original output/build/route/count proofs. Native/Python
ledgers reconcile. Source29,814files, original24fit files, main installation,
private binaries and final code freeze checks pass.

| Metric | Before | Stage1 |
| --- | ---: | ---: |
| Whole wall median [range] s |20.602 [19.886,20.788]|17.686 [17.186,18.057]|
| OS-reported CPU median s |18.227|15.359|
| Maximum workload-self kernel peak RSS MiB |431.27|457.18|
| Median workload-self peak RSS MiB |420.13|422.46|
| Median post-release RSS MiB |294.97|294.13|
| Managed peak bytes |1,605,390,806|1,605,390,806|

Whole wall improves14.15%/2.916s with nonoverlapping observed ranges. Maximum
RSS increases25.91MiB, while median peak increases2.34MiB and post-release is
similar; the unchanged2GiBmanaged policy remains satisfied. Reported CPU is
not physical-work attribution; root-cause CPU-API uncertainty still applies.

The actual median-whole trials (before3/after0) report feature/numeric-spooling
4.838824->2.008627s. Those are whole representative trials, not independent
stage medians or an exact decomposition of the optimization benefit. Full/stage,
RSS/PSS, storage/I/O lower bounds and costs are retained in`stage1-primary/`.
No storage saving is claimed merely from changed sampled peaks.

### Separate check-count diagnostic

`stage1-counters` reuses the reviewed root-cause wheel instrumentation, replacing
only the two feature files with exact candidate files. Its source`9cebd7c0...`/
ELF`52fe02c6...` is diagnostic only and not the primary binary. The single4worker
capture preserves all original output proofs and completed trace/cleanup.

Feature-stage checks drop129,631,653->18,772,215 (85.52%); whole-operation checks
143,785,761->32,926,323 (77.10%). Remaining checks include other feature work,
validation and source/proof boundaries. These counts are not CPU-time shares or
proof that every removed call came from one loop. No global polling change occurs.

Decision: retain Stage1 for the measured14.15%complete-path gain with explicit
peak-memory tradeoff and exact outputs. Later candidates use this accepted wheel
as their comparison baseline. No further optimization is silently included.

## Stage2: accepted adjacent-check coalescence

Compared accepted Stage1source`9bd697d...`/ELF`8c9a2a...` with candidate
`d9413ab144d2078606ce176b8fdd083d6c92ddc3d797bb4bb99ef90a05b1604e`/
`c8f35f6efe703ff16c2875355431ce748acc5fb477e3f3625eab81302284e7c3`.
Only the post-event full check is removed. Entry, before every event and final
completion checks remain; global Control and aggregation are unchanged. Detection
is bounded by one event plus fixed output emission before cursor commit.

Full mimalloc Rust76PASS/2ignored, focused9PASS; deterministic after-hash fault
injection is cfg(test)only. One/final/next/max-event cases verify cancellation/
deadline failures never commit the block. Candidate Python execution/wall/feature
checks40PASS. Full826tests are not repeated for this local check-position change;
every primary command still revalidates the complete fixed corpus and original
83,783-row numeric/source proof. Source correctness and conformance READY.

The same4worker/2GiB/mimalloc-release protocol completes all12commands, with
one warmup and five alternating pairs. Every output/reader/binary proof and all
integer ledgers match. Final source/protected-model/main/code checks pass.

| Metric | Stage1 control | Stage2 |
| --- | ---: | ---: |
| Whole median [range] s |16.972 [16.742,17.135]|16.729 [16.682,16.815]|
| Reported CPU median s |14.453|14.378|
| Maximum workload-self peak RSS MiB |428.75|427.17|
| Median workload-self peak RSS MiB |424.43|405.28|
| Median post-release RSS MiB |294.84|296.16|
| Managed peak bytes |1,605,390,806|1,605,390,806|

All five paired whole gains are0.320603/0.202067/0.388159/0.015678/0.051417s;
target feature/spooling gains are0.189170/0.161180/0.160557/0.147172/0.108432s.
Whole median improves0.243011s/1.431853%. Whole ranges overlap, but target-phase
ranges separate:1.646627–1.762071s before and1.528055–1.572900s after. Median CPU
change is only0.52%and one paired CPU result regresses; no consistent/physical CPU
improvement or formal statistical significance is claimed.

Decision: retain this small, bounded workload-specific gain because all alternating
pairs and target phases improve, outputs/guards remain identical and resources
show no material regression. Independent measurement audit PASS with no blocking
adoption disagreement. Do not add a global or configurable polling mechanism.
Use Stage2 as the next accepted comparison baseline. Stage-to-stage percentages
come from fresh controls and must not be multiplied into a claimed total speedup.

## Stage3a: accepted schema sharing under shared load

Candidate source`89385a40697e091ae895f37f893937484d2993330bc0126d3c9e34b06868d2c3`,
ELF`d876348047f473f85339bf272a570cd110490d0795fff413d23ca4c12ae304e9`.
Control is accepted Stage2source`d9413ab...`/ELF`c8f35f...`.

Borrowed Observation and Input schema fields retain an Arc-backed collection.
The generated closed six-field wire adapter decodes both fields normally before
complete structural equality permits sharing. Observation validates first; only
pointer-identical immutable schemas avoid a second shape traversal. Unequal
metadata retains its original validation and mismatch rejection. Serialize still
emits both fields. The initial two deserializations remain; no raw/lazy parser,
cross-input cache, source-format or Config change is included. Native/standalone
Serde enable matching rc support; dependency versions, locks and AP stay fixed.

### Executed correctness gates

- Native mimalloc Rust79PASS/2ignored; standalone AP/noAP contract tests15PASS
  each. Direct Stage2 compatibility oracle9PASS in each mode, including132depth
  boundary fixtures, unknown/duplicate/numeric/serialization/pointer/lifetime
  cases. Generator regeneration, formatting, Ruff and diff checks pass.
- Final isolated wheel:826Python tests PASS/5conditional skips,525.01s. This ran
  while external work was active and is correctness evidence only.
- Both frozen feature recipes,83,783-row Parquet index and actual saved current-v2
  PPO at1/4workers preserve exact original arrays, hashes and source proofs. All
  five sequential jobs in`stage3a-verification-retry/` pass,64.031s. Diagnostic PPO
  clones remain at zero updates. An earlier copied harness selected the Stage1
  site; its identity guard failed. That attempt is retained and excluded from
  Stage3a evidence, rather than relabelled as a successful candidate run.
- Independent source correctness and architecture conformance READY; independent
  final measurement audit PASS with no blocking findings.

### Authorized shared-load comparison

The user requested advance notice because another heavy task was active, then
authorized this trial while reporting roughly20% all-core CPU usage. Notice was
given before execution. Same83,783history rows, releaseopt3/debug0 mimalloc,
4workers/inflight4 and2048/512/16MiBmemory/cache/chunk; one warmup and five
alternating fresh-process trials per variant. No own build/test/edit ran during
timing. The background workload was not controlled. All12commands and final
source/protected-model/main/code checks pass; all timing ledgers reconcile.

| Metric | Stage2 control | Stage3a |
| --- | ---: | ---: |
| Whole median [range] s |19.770 [19.374,20.109]|19.094 [18.746,19.611]|
| OS-reported CPU median s |17.583|16.763|
| Maximum workload-self kernel peak RSS MiB |429.92|423.91|
| Median workload-self peak RSS MiB |428.73|394.38|
| Median post-release RSS MiB |295.58|295.02|
| Median post-release PSS MiB |287.51|286.66|
| Managed peak bytes |1,605,390,806|1,605,390,806|

Whole median improves0.675946s/3.419123%. All five paired wall gains are
0.653000/0.445316/0.498159/0.675946/0.702815s. Journal-validation improves in
every pair by0.362375–0.433695s; its measured ranges separate2.811832–2.869000s
versus2.423769–2.465289s. The actual median-whole trials are trial3in both variants:
validation2.868999516->2.435304361s, typed parse dispatch/join4.621195655->
4.440023088s and temporary release2.378485893->2.138978186s. These are
representative trials, not independently selected stage medians or a causal
partition of the median gain.

Maximum peak RSS falls6.01MiB and median peak34.34MiB; coarse admission estimates
do not change with actual tree sharing. Scratch/output formats and logical
content do not change. Sampled storage peaks and OS I/O counters are retained
in the report; differing samples are not evidence of storage savings. The many
logical reads with few charged device reads are consistent with a warm page
cache; no25-second physical I/O wait or disk bottleneck is established.

Guest aggregate CPU busy across the ten measured commands is10.63–11.12%
(9.85–11.12%including warmups). It includes this workload and other guest work;
it does not quantify the user's Windows Task Manager report. Reported process CPU
median falls4.67%, but CPU API uncertainty remains and physical attribution is
not established. Whole ranges overlap and shared load limits generalization.

Decision: retain this bounded schema-sharing change for consistent paired/target
elapsed gains and lower measured peak memory in this authorized shared-load
environment. No uncontended or generalized speedup is claimed. Artifacts:
`stage3a-primary-shared-load/{freeze,plan,results,completion,summary,stage-attribution}.json`
and its sibling log. Do not multiply stage percentages from different controls
into a claimed total speedup. Additional performance work requires advance notice.

## Stage5a: accepted participant-file task supply

Architecture design/source-conformance and independent source-correctness reviews
are READY with no blockers. Candidate source
`d5090c4199c7462d83b50c486057aa682b0e0808475a048d3d48621f288ed961`,
isolated releaseopt3/debug0 mimalloc ELF
`97f1b50990b4b97100c922e0191de4dd447ecb2c27bbbd81615d0c75ce67b628`.
No runtime dependency, Config, persisted format, learning or source-authority
change. Read windows and worker caps remain unchanged.

Within each bounded window an atomic ordinal assigns original participant-file
slots to scoped workers. Budgeted worker-local results regroup by original
episode/file slot after every join; downstream header-based viewer mapping is
unchanged. One worker remains inline; all-legacy bypass and mixed legacy no-op
slots preserve compatibility. First failure cancels siblings and drains joins;
no partial parsed window reaches validation/publication.

`typed_parse_parallelism` now explicitly reports job_unit
`participant-journal-file`, job_unit_version2and scheduled_journal_tasks, retaining
scheduled_episodes for window membership. Mixed windows include legacy no-op
slots in task counts; all-legacy windows schedule no parse work. Historical
episode-job diagnostics keep their old interpretation. Nested elapsed and active
job overlap remain distinct from physical CPU utilization.

Full mimalloc native tests81PASS/2ignored (50library+31integration); focused
worker cases10PASS. Checks cover reverse file completion, reversed viewer/file
mapping, borrowing/evidence movement, dynamic progress while a file is blocked,
inline/mixed/legacy behavior, partial windows, panic/cancel/deadline/quota cleanup.
Formatting, scoped Ruff and diff checks pass. Optional combined later-task
failure-after-earlier-success coverage was deferred: repeated claiming and failure
cleanup are independently covered and no blocker was identified.

### Executed correctness and primary comparison

Full isolated-wheel Python regression818PASS/13skips in364.78s initially omitted
the explicit parallel-parser opt-in. A separate invocation with
GFB_TEST_PARALLEL_PARSE=1passes all8cases in115.46s:826distinct tests covered,
with five original conditional skips remaining. No whole-suite rerun was needed.
Full Ruff and mypy59source files pass.

History/observation/index proofs pass in`stage5a-verification/`. Its first PPO
wrapper fails before source work with missing json import; preserve/exclude that
protocol attempt. Corrected PPO-only jobs at1/4workers pass in
`stage5a-verification-ppo-retry/`, with complete identity guards and clones at zero
updates. Combined five-job success duration66.514s is correctness evidence only.
All original arrays/NPZ/sample/source/dataset and83,783-row index identities match.

After advance notice, all own tests/builds/source jobs stop before timing.
`stage5a-primary-shared-load/` compares accepted Stage3a against this candidate:
same releaseopt3/debug0mimalloc,4read/parse workers/inflight4,
2048/512/16MiBmemory/cache/chunk, one warmup plus five alternating fresh-process
pairs. Background work is uncontrolled. The earlier user20%CPU report is
historical, not a contemporaneous measurement. All12commands, original output/
build/source/protected-model/main/code checks, job/worker caps and ledgers pass.

| Metric | Stage3a control | Stage5a |
| --- | ---: | ---: |
| Whole median [range] s |19.261 [18.689,19.331]|18.735 [18.054,19.045]|
| Parse dispatch/join median [range] s |4.409 [4.330,4.463]|3.447 [3.406,3.473]|
| OS-reported CPU median s |16.882|17.143|
| Maximum workload-self peak RSS MiB |432.03|431.79|
| Median workload-self peak RSS MiB |408.88|427.00|
| Median post-release RSS MiB |296.39|294.35|
| Median post-release PSS MiB |287.52|286.09|
| Managed peak bytes |1,605,390,806|1,605,390,806|

Whole median improves0.526273s/2.732321%. Four of five paired whole results
improve:0.581283/0.526273/1.214726/0.285832/-0.128756s. Parse improves every
pair by0.902755–1.057067s, with separated ranges; its median falls21.82%.
Whole ranges overlap. Reported CPU median rises1.542533%and median peak RSS
rises18.12MiB; maximum and post-release memory do not rise. These tradeoffs
remain visible under unchanged budgets; no claim of CPU or memory improvement.
Guest aggregate busy during measured commands is10.79–11.55%, including this
workload and other guest work; physical/Windows-host attribution is unavailable.
Storage formats/content remain identical; raw observer storage/I/O samples retain
their lower-bound and sampling scopes, and do not demonstrate storage savings.
Every command records the same logical staging read/write totals
1,068,020,777/1,003,260,077bytes and owned scratch peak1,856,048,056bytes. These
are logical ownership/call totals, not physical device traffic or resident memory.

Actual median-whole trials are trial1in both variants. This disjoint ledger
includes all outer work so stage sums can be reconciled with whole elapsed:

| Exclusive scope | Control s | Candidate s |
| --- | ---: | ---: |
| Native parse dispatch/join |4.409453|3.472528|
| Native read/decode wait and source merge |3.188937|3.401342|
| Native Journal validation |2.417445|2.555422|
| Native features/numeric spooling |1.894740|1.900822|
| Native temporary release |2.124057|2.126872|
| Other native scan categories |1.799632|1.814101|
| Preparation envelope outside native scan |1.805221|1.676125|
| Archive numeric staging |0.437125|0.409740|
| Archive output proof |0.433790|0.475019|
| Final source check/cleanup |0.546805|0.699965|
| Outer envelope residual |0.203814|0.202811|
| Whole elapsed |19.261019|18.734746|

The envelope residual is not separately instrumented startup/GC time; outside
scan is envelope subtraction. Native integer partitions reconcile exactly.
Representative stage changes are not independent medians or a causal allocation
of the median benefit. Baseline scheduled6,784episode jobs; candidate schedules
13,568file jobs with the same6,784spawned scoped threads and active peak4.
This remains file-granularity parsing, not per-line streaming or a persistent
pool. The candidate amortizes to0.224ms per accepted action over83,783rows;
that is preparation throughput, with inference and game waiting outside scope.

Independent arithmetic/resource/lifecycle audit PASS; READY for scoped retention.
Decision: retain Stage5a's measured elapsed improvement with the explicit CPU/
median-memory cost above. No uncontended or formal significance claim, and no
per-job duration comparison across the changed episode/file job units.

## Closeout and separately gated alternatives

Stage4 persisted history-format migration is conditionally skipped for the
immutable current corpus; no speed claim is made. A future versioned producer
experiment and compatible in-memory prefix reuse remain unassessed, separately
gated alternatives. Stage5a source, execution and measured adoption are accepted.
No persistent pool, line-batching or stage overlap follows automatically from
this candidate.

The final accepted source d5090c41/release-mimalloc ELF97f1b509is installed into
Bots.venv from the exact tested wheel after all comparisons. Old32e3ad68package
and distribution metadata are backed up in`main-before-final-install/site/`, with
verified rollback identity. Installation is offline/native-only/no-deps/no-build;
fresh loaded binary/build/ABI/allocator/reader capabilities match the private
candidate exactly apart from installation path. Optional-free CLI help starts
without importing gf_native/Torch/LightGBM/PyArrow. All24protected original fit
files retain their original hashes/stats. `final-installation.json` and
`final-installation-help.log` retain commands and proofs. Config/defaults and
normal checkpoint/environment identity guards stay unchanged. No further full
regression is repeated after installing the same already-tested ELF.

The bounded sequential optimization unit is complete: retain Stage1/2/3a/5a;
conditionally skip Stage4 persisted-format migration. Future prefix reuse,
versioned delta producers, persistent pools, line batches and stage overlap need
their own scoped evidence and acceptance. No total speedup is inferred by
multiplying fresh-control or shared-load percentages from different comparisons.
