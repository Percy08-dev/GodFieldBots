# Owned episode-pipeline pool verification

[ADR0020](../adr/0020-owned-episode-pipeline-pool.md) governs the combined
implementation authorized by the user: owned fixed pool plus worker-local
read/parse/validate/encode/spool/release. No pool-only speed claim is intended.
Implementation must be compared directly with retained baseline97f1, not with
historical timings from another campaign.

## Candidates and source acceptance

Baseline: source d5090c4199c7462d83b50c486057aa682b0e0808475a048d3d48621f288ed961,
releaseopt3/debug0mimalloc ELF97f1b50990b4b97100c922e0191de4dd447ecb2c27bbbd81615d0c75ce67b628.
Main installation stays on that baseline during checks/comparison.

Final candidate v3: source faf71b3364a26feb3ff9bcf0605d065135a8fc1adc3f6571e3a252b30c8b79b6,
releaseopt3/debug0mimalloc ELF2ed57a3181bf3eae5ee6bf6cf214dcab6a32aa3c11878c7ff551dde71ca5f965.
Native lock f81e4cc1d0e648ee834ff289fe493d09012fac92adbdf001f05fd7ec087c6576.
All prior package identities/checksums remain. Six additions: rayon1.12.0,
rayon-core1.13.0, crossbeam-deque0.8.8, crossbeam-epoch0.9.21,
crossbeam-utils0.8.23, either1.18.0. Python dependencies/defaults are unchanged.

Independent design and final source-correctness/conformance reviews are READY.
Worker-local large-object lifetime, actual scoped thread joining, bounded
admitted/uncommitted results, ordered canonical byte streams and final coordinator
source/eligibility/publication gates are retained. Numeric/source/index format
and reader/FFI ABI stay unchanged. New episode_pipeline_contract is
owned-complete-episode-v1; complete_episode_pipeline diagnostics version1name
complete-episode jobs. Coordinator ledger version2retains historical moved-work
keys at zero and adds episode_pool_wait/episode_ordered_commit. Worker phase
wall totals are nested, not exclusive command-wall components.

Review caught two scratch-accounting gaps before any performance campaign:
numeric-spool leases could be dropped with failed/discarded results, and proof/
identity leases could be dropped after a partially failed write. Numeric charges
are now stage-owned before file creation through joined physical removal; ledger
charges enter their surviving owner before writes. Held-sibling tests cover both
readers, discarded success, an opened partial numeric file, actual4096-byte write
then failure from a64KiBfragment bypassing32KiBBufWriter, quota refusal, joining,
cleanup and lease release. Failed-cleanup retry retention is tested. Permanent
cleanup-failure limitations are documented in ADR0020 and remain fail-closed.

Final native tests89PASS/2existing ignored benchmarks, Cargo fmt/diff PASS.
Full Ruff PASS; mypy59source files PASS. Tests include both readers, operations,
1/2/4/8workers and in_flight below/equal/above, pool reuse across collections,
shuffled completion, role/viewer and canonical output parity, worker-local release,
later failure/panic/coordinator/cancel/deadline/write/flush/quota/spawn refusal.
Partial OS spawn failure was not forced; the scoped library join path is inspected.

Preliminary v1wheel passes826Python tests/5conditional skips in410.67s. It is
not the final candidate. V2Python regression was deliberately interrupted after
the second write-reservation finding; its logs/root-stop ownership proof and
pytest KeyboardInterrupt/teardown error are preserved, not labelled candidate
failure or success. Final v3full Python regression passes826tests/5existing conditional skips in395.96s,
with GFB_TEST_PARALLEL_PARSE=1and exact candidate identity. Both complete feature
recipes,83,783-row Parquet and saved current-v2 PPO1/4source/numeric proofs pass
in46.698s total correctness-command time. Diagnostic PPO clones stay at zero
updates. All13,568typed Journals/534,640records and6,784episodes are accepted;
pool starts/exits, active/outstanding/reorder caps, cleanup and budgets pass.

## Before/after protocol

Artifacts: runs/assessments/episode-pipeline-pool-20261031/.
Final production source is frozen. Correctness jobs run sequentially after final
Python regression: complete history, observation, Parquet, saved real current-v2
PPO at1/4workers. Each probes the full loaded identity from candidate-v3metadata
before execution; no stale v1/v2site or hash is accepted. No games, real fits,
optimizer updates or eligible new parent are created.

The reviewed bench controller retains unchanged observer/workload code and plans
48fresh-process commands: one warmup and five rotating trials per1/2/4/8worker
configuration, before/after order alternates. max_in_flight=workers, shared
memory/cache/chunk2048/512/16MiB, fixed83,783-row history dataset. Every workload
checks exact original arrays/NPZ/sample/dataset/build/reader counts, both integer
wall ledgers, worker/capacity/lifecycle and clean scratch. Actual source provenance
is internally recomputed and matched to the frozen dataset each trial; the
unchanged workload exposes no separate source_set_sha256field. Do not invent
an independent per-trial source-hash observation. All29,814raw files,24protected
fit files, main/binaries/code are additionally checked before/after the campaign.

Metadata/resource/argument/order/no-overwrite fixtures and independent source
review are READY. Secondary snapshot/postflight/persistence failures cannot mask
the original observer exception; explicit normal+1ResourceLimit with clean owner
is distinguished from signals/timeouts/infrastructure failure. Actual AFTER workload metadata validator replay passes using the real v3history
row; a saved baseline observer envelope is used solely as a fixture and is not
labelled an AFTER performance record. Actual performance results follow below. User reports the other heavy
task ended; notify before timing and retain system snapshots without claiming host
isolation. No own builds/tests/source jobs/edits run during performance trials.


## Executed same-campaign before/after comparison

All48commands PASS:8warmups and40measured fresh-process trials. No unusable
resource cells, primary exception or postflight error. Source/protected-model/
main/binary/code/harness bracket checks pass. Campaign elapsed801.371s includes
orchestration/validation, distinct from the sum of workload observations. The
user reported the other heavy task ended; guest snapshots do not establish host
isolation. No own build/test/source job or source edit overlaps timing.

| Workers |Before median [range] s|After median [range] s|Median elapsed reduction|
| --- | ---: | ---: | ---: |
|1|22.721 [22.286,24.555]|22.688 [22.463,23.317]|0.14%|
|2|19.641 [19.506,20.005]|14.857 [14.756,15.317]|24.36%|
|4|16.310 [15.943,16.910]|10.906 [10.604,11.681]|33.14%|
|8|14.406 [13.850,14.786]|8.794 [8.405,9.247]|38.96%|

1worker is effectively unchanged: ranges overlap, so0.145%is not a meaningful
speed claim. The2/4/8observed ranges separate; this is the combined implementation
effect, with no pool-only or universal statistical/generalization claim.

| Workers |OS CPU median before/after s|Self peak RSS max before/after MiB|Post RSS median before/after MiB|Managed peak max before/after MiB|
| --- | ---: | ---: | ---: | ---: |
|1|22.461 / 22.536|359.72 / 350.15|299.42 / 300.79|1531.02 / 1531.02|
|2|14.378 / 23.139|387.43 / 385.60|296.64 / 302.48|1531.02 / 1531.02|
|4|14.520 / 23.513|448.93 / 416.21|294.99 / 302.40|1531.02 / 1531.02|
|8|15.225 / 25.380|493.32 / 512.72|295.28 / 306.45|1592.18 / 1677.34|

At4workers whole median falls5.404505s/33.135671%, maximum self RSS falls32.71MiB,
while post-release RSS rises7.41MiB. At8workers it falls5.612670s/38.959876%,
maximum self RSS rises19.40MiB, post-release RSS11.16MiB and managed peak85.15MiB
to1,677.34MiB, within the unchanged2GiBsoft admission policy. Sampled tree PSS
max4is405.50->398.02MiB; at8it is457.58->504.31MiB. Full per-trial RSS/PSS,
post-release values, storage/I/O lower bounds and coverage limits remain in summary.

OS-reported CPU median rises61.94%at4and66.70%at8. Do not present this as lower
CPU work or dismiss the increase: long-lived versus short-lived thread accounting
and the previously demonstrated CPU API uncertainty prevent a clean physical-work
attribution. The direct measured claim is lower elapsed, with explicit reported
CPU/resource costs. No CPU-efficiency or energy saving is established.

Numeric/proof/source outputs stay byte-identical. Native ELF size grows from
7,317,680to7,479,936bytes (+162,256). Protected six-model file sizes stay unchanged.
Owned scratch peak stays1,856,048,056bytes in every trial; logical and OS I/O scopes
remain separate, and sampled physical counters are not complete device traffic.

## Coordinator and worker timing scopes

Coordinator ledger version1before and2after reconcile independently, as does
Python's nested ledger. New moved-work zero entries mean the work is performed
inside jobs, not eliminated. Worker phase sums overlap across threads and must
not be added to coordinator exclusive wall or compared as equivalent serial
elapsed phases. Actual median-whole before/after8trials are3/2.

Before8native serial validation2.231593s, feature/spooling1.638796s and release
1.667605s become worker-local phases in the candidate. After8coordinator exclusive
pool wait is5.033053s, ordered commit0.223843s, collection setup0.223881s,
collection proof0.048625s and global checks0.286034s. Remaining Python/archive/
finalization scopes and all integer/flattened categories are retained in
stage-attribution.json; no timing gap is hidden by summing nested worker totals.

## Adoption status

Numerical/source/failure gates and all48measurements pass. Independent arithmetic/
resource audit PASS/READY with zero discrepancies in all48rows, eight summaries,
paired schedule, actual-median tables, source/build/output/owner proofs and integer
ledgers. All five pairs improve at2/4/8workers. At4, paired gains are
4.262360/4.932915/5.078594/6.306617/5.691559s; at8,
5.528816/5.538697/5.585219/5.928103/5.444511s. Root retains the combined
complete-path improvement with the explicit reported CPU/memory costs. This is
not a claim that accounting uncertainty explains away those costs.

After-candidate8reduces median elapsed19.367020%versus after4, with96.51MiBhigher maximum self
RSS and7.94%higher reported CPU median. Recommend8for observed speed and4for lower
peak memory. Worker/reader/memory defaults stay unchanged. All48logical staging
read/write totals remain1,068,020,777/1,003,260,077bytes, as does owned scratch.
The six protected models retain1,898,536total bytes and their original hashes.
Observed command wall totals787.557632s and reported CPU972.125563s; recorded
bracket checks3.310784+3.793130s are separate. No CPU-efficiency claim follows.

The exact tested2ed57a31wheel is installed into Bots.venv after comparison and
acceptance, offline/native-only/no-deps/no-build. Before replacement, exact97f1
package and distribution metadata are backed up and verified at main-before-adoption/.
Fresh interpreter identity/source/ABI/allocator/pipeline capability matches the
tested wheel; optional-free CLI help does not import native/Torch/LightGBM/PyArrow.
All24protected fit files remain unchanged. Evidence: installation.json and
installed-help.log. No full regression is repeated after installing the same ELF.
Resume/code/environment guards remain unchanged. No further profiling, fit, game
or optimizer update is part of this unit.
