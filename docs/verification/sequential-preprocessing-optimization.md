# Sequential preprocessing optimization

[ADR0019](../adr/0019-sequential-preprocessing-optimization.md) governs independent
changes and retain/reject decisions. User authorized sequential application and
measurement. No simulator changes, source rewrites, games/fits/learning updates or
automatic reader/worker-count switch. Main native is held as the historical
baseline until final adopted-wheel installation.

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

## Remaining sequence

Stage3 schema sharing, Stage4 history representation and Stage5
supply/pool work remain separately scoped under ADR0019. Their implementation,
measurement and adoption decisions are pending; this report is not a completion
claim for those later stages.
