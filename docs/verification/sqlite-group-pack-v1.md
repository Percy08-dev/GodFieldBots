# SQLite group Pack verification

Date: 2026-11-10. Scope: implementation in progress on
`feat/sqlite-group-pack-v1`; default directory format retained.
Design: [ADR0034](../adr/0034-sqlite-group-pack.md),
[foundation](../design/sqlite-group-pack-v1.md),
[runtime](../design/sqlite-group-pack-runtime.md),
[Source](../design/sqlite-group-pack-source.md).

## Acceptance status

Foundation implementation passes independent correctness/conformance after two
bounded fix rounds; root accepts this isolated milestone.
Standalone runtime and Python original Source/ordinary PPO are accepted in the
separate scopes below. Native, final64, Dataset/Tree and whole-plan comparison
remain pending. This record distinguishes component
evidence from the complete owner's requested outcome.

## Foundation candidate evidence

Environment: CPython3.14.10, SQLite3.54.1, stdlib compression.zstd. No new dependency.
Baseline Bots83aa67e; design milestone495ecec. The initial capture below preceded
foundation acceptance/commitdc9440e. The implementer's report records64focused cases including real
child/IPC/SQLite failure windows. Root independently ran:

```console
.venv/bin/python -m pytest -q tests/test_group_pack_codec.py tests/test_group_pack_storage.py tests/test_group_pack_failures.py tests/test_config.py tests/test_card_instance_collection_publication.py
```

PASS:100checks in4.36s. The old Config/publication-only baseline passed36checks
before implementation. Scoped Ruff passes; scoped mypy passes all6new source
modules. `git diff --check` and new document-link checks pass.

Root's temporary `/tmp/gfb-group-pack-real-bytecheck.py` read only the two named
historical collections, verified every original inventory SHA/length, partitioned
members by actual recorded managed group, encoded each group and required exact
kind/name/byte equality after decode:

| Historical source | Groups / games | Inventory files | Original bytes | zstd Pack bytes |
| --- | --- | --- | --- | --- |
| corrected contract smoke c6983ca8 | 1 / 2 | 59 | 3,998,959 | 298,483 |
| corrected pilot dc641b06 | 32 / 192 | 4,492 | 273,062,500 | 17,737,466 |

PASS: all4,551inventory members restore byte-for-byte. Original manifest/report
controls total69,534and4,212,418bytes respectively and remain unchanged.
These are historical real-data codec checks, not collection/Source/learning
eligibility, a migration, or a storage/read throughput comparison. Pack totals
exclude DB/index/control/descriptor overhead. No originals were modified.

The sandbox reports `/mnt/d` and `/dev/shm` read-only; real runtime/benchmark
writes there require the tool's escalated execution context under the owner's
explicit path authorization. Host-side read-only PowerShell inspection of D:
returned disk2, USB model ASMediaASM1153e, size4,000,787,030,016bytes,
`MediaType=HDD`. Thus the planned D: comparison target is verified as HDD by
the host OS, rather than inferred from a WSL guest ROTA flag. No host storage
settings were changed. Historical input originals currently reside on ext4;
old/new benchmark outputs must both be written on D: under fresh owned paths.

Final foundation candidate:87focused tests plus36old Config/publication checks
pass in root's exact combined command above (**123PASS**,5.54s). Scoped Ruff and
mypy6PASS; document links and diff check PASS. Independent review found and
verified fixes for seal-time DB substitution, uncertain-FD reuse, constructor/
helper cleanup ownership, pre-admission namespace allocation, catalog first-use
race and retained-Writer/publication joint-budget overlap. Both final reviews
are READY with no remaining blockers. Their scoped reports do not represent
unexecuted application/native/learning checks as passes.

Disposable probes on actual `/mnt/d/godfield` and `/dev/shm/godfieldbots` both
PASS: WAL→checked TRUNCATE(0,0,0)→DELETE, file/directory fsync, readonly integrity,
unchanged DB hashes and no new sidecars. Probe-only files were removed; no
existing data was modified. This establishes supported operations on the target
filesystems, not a physical power-loss durability test.

## Standalone runtime acceptance (2026-11-11)

The explicit Config, memory Journal6/codec2/Replay/results, RAM Simulator logs,
SAME budget, actual queued/submitted work custody, composite publication and
typed final checkpoint diagnostics pass both independent reviews after bounded
fixes. Old directory Config/hash/default paths remain unchanged. Packed
Source/PPO/native/final64/Dataset/Tree entry points remain explicitly guarded.

Root's fixed-revision expanded HOST check passes **616checks/10explicit skips**
in66.96s, plus fresh full changed-file Ruff/mypy23. The initial sandbox's20CPU
preload setup errors were traced to AF_UNIX EPERM; identical unchanged fixtures
pass HOST. No timeout/skip or infrastructure change was used to hide a failure.

Actual D:9p exposed a write-close ctime refresh for owned descriptors. The scoped
`terminal-control-ro-handoff-v1` policy fixes original inode/hash across that
controlled transition, then requires strict full-stat/hash readonly validation
and retirement. DB and all ordinary Source custody stay strict. D:/ext4 public/
diagnostic probes pass all4cases with zero reservations. The first failed run
02d698f1 retains6natural results and an intact DB, but its locator is quarantined.
The second cbb37f35 physically published but failed full inventory audit because
the metrics control was undeclared. Both populations remain immutable/ineligible.
Metrics bytes now freeze into inventory before manifest/report; no Source
exception or old-run repair is used.

Final actual six-game run9542c7a2 passes1Pack/6natural games,176declared files and
14,774,682restored bytes with every hash/length/report/index check. Six actual
MLP/GRU/Transformer×CPU/CUDA inference conditions each pass2natural games and the
same saved-byte audit (**18successful real games total**, not learning quality).
Final metrics expose actual child checkpoint/DELETE and combined seal times;
managed reservation peaks are not RSS. Detailed receipts/configs are retained at
`runs/assessments/sqlite-group-pack-v1-20261111/`.

Explicit example: [baseline storage preset](../../configs/experiments/card-instance-group-pack-baseline.toml).
Only this new table opts in; the default is unchanged. Supported persisted v1
max Pack is256MiB; positive smaller limits remain configurable. Minimum193MiB
is fixed Writer128+inspection64+minimal control slack1, not a promise any game
fits the minimum. Larger bounds need a reviewed format/Source/budget extension.

## Python original Source and ordinary PPO acceptance (2026-11-11)

Actual retained locator/descriptor/DB/predecessor custody, one-group shared pool,
proof2, run4/update4, wire5 and Source-issued precommit retirement pass both final
independent reviews. Source never derives update authority from a Pack/catalog/
serialized proof. Old directory proofs, Config identities and numerical schemas
remain separate. Retention is an explicit fresh postcommit parent/pruning check;
its failure preserves real committed history and invalidates the parent.

Root's frozen HOST affected aggregate with GF_RUN_CUDA=1 passes610checks in
238.81s; full changed-source Ruff/mypy170 and all8frozen algorithm ASTs pass.
Reviews required fixes for expired-deadline cleanup, partial constructor owners,
training-to-CLI coupling and unrelated-exception error masking. A real16game
collection then exposed duplicated decode reservations; its parent remains
failed/updates0/ineligible. Full constructor-validated inventory, one strict
Journal parse and separate raw/graph reservations solve that duplication under
the unchanged2GiB cap. The saved failed cohort passes structural-only inspection
at1,565,165,664B admitted peak and cleanup0; it is not rehabilitated. The
constructor's temporary all-available reservation is not RSS.

Review also reproduced an opaque-file size-cap regression in the new inventory
path. The same max_source_file_mib guard now applies to ALL members without
rereading opaque bytes. Both final delta reviews are READY; root's50affected
admission/Source/ownership checks pass15.00s, alongside fresh Ruff/mypy170/AST8.
The preceding610pass result is pre-two-line-guard evidence, not falsely rerun.

Actual new parents, CPU inference/CUDA learner, seed17/max2updates, unchanged
7200/5400/7140/60budgets and2GiB storage:

| Parent | Outcome | Games / updates | Selected decisions / optimizer steps |
| --- | --- | --- | --- |
| continuous2579bc3a | complete531.165s | 32natural /2 | 419 /16 |
| pause-resumedba33a31 | paused1, then complete2 | 32natural /2 | 419 /16 |

Each mode used the real Simulator91d5926 and actual wire5 learner. Separate
fresh root commands `gfb-root-packed-parent-check.py` call the actual complete
parent validator with a420s work bound, then assert run4/update4/proof2/numeric4,
three generations, original storage/custody and unchanged run bytes. Both exit0:
continuous87.381s and resumed87.661s. Source validation covers actual complete
inventory/DB/Pack/Journal/candidate/Replay bytes and own-role correspondence.
The independent continuous audit additionally observes both Source owners,
SQLite/FD retirement and shared pool/control accounting0. Its initial temporary
wrapper import/summary failures are preserved separately; they do not change the
successful validation or replace the two fresh successful root commands.

This accepts Python storage/runtime/resume scope:64natural games and4actual
updates. It does not establish formal qualification, strength, full model/device
learning-campaign coverage or independent-trajectory final-state determinism.
The prior standalone matrix covers all3models×CPU/CUDA actor paths, and the
affected numerical tests cover CPU/CUDA learning/parity. No formal campaign,
HPO, final test or real-data Tree fit was run. Root receipts/configs/reviews/logs
are retained in `runs/assessments/sqlite-group-pack-v1-20261111/python-source/`.

## Direct Writer terminal completion (2026-11-11)

Direct published/non-retained-pending sealing now shares the exact reviewed
terminal-control readonly helper with composite publication. Both independent
reviews are READY; retained pending behavior, DB/Source strict6 and original
Pack/control bytes stay unchanged. The isolated worktree first passed163bounded
cases; its parent/child mixed-path limitation is not presented as package proof.

After all main PPO/resume/audit processes retired, root copied only the exact
four reviewed files and verified all parent/helper/child imports came from main.
Fresh source-matched249affected/legacy tests pass27.03s, Ruff/mypy171 and all8
kernel ASTs pass. Four fresh D:9p/ext4×public/pending probes pass exact logical
bytes, unchanged controls, healthy Writer and account0; subsequent ordinary
Source metadata changes correctly refuse. D: mutation changed mtime+ctime;
ext4's ctime-only mutation is reported separately. No Source ctime relaxation or
DB control is introduced. Earlier failing/unsupported probe attempts remain
diagnostic evidence. The benchmark publication gate is now open; timings below
remain unmeasured until the planned comparison actually runs.

## Native original-byte provider acceptance (2026-11-11)

Generic Rust logical Sources consume exact Python-owned bytes through the SAME
registry/provider/lease/pool; Rust gains no SQLite/zstd dependency. Explicit plan4,
proof2/control4/receipt2 and RAW/Original16 contexts retain old plan3 meanings and
all numerical formats. Creator-thread IO handoff, actual pool-close holds and
pre-argument external accounting preserve originals through checked native join.
Unicode16name/strict packed JSON validation and full SQL index/manifest relation
checks pass their reviewed contracts. Both final independent reviews are READY.

Reviews required fixes for fresh generation0 empty-registry ownership, no-effect
partial executor cleanup, staged provider/query admissions and double-fault
first-error owner propagation. Root fresh45focused HOST checks pass3.43s after
the final minimal correction. Earlier143/3skips and112/1skip scoped batches are
separate pre-correction evidence, not summed as disjoint final tests. Full
mypy172/Ruff/8kernel AST checks pass. All3controlled model Sources match Python
proof/correspondence, complete Numeric4 arrays and all16Original16 packets with
actual final closure; CPU/CUDA component18checks pass. Fixtures remain unqualified.

Root enables the reviewed capability only after both reviews and creates the
source-matched release/MIMALLOC cp314-GIL wheel. Exact native53-file source recipe
faff452b…f7115 is independently recomputed; wheel83f952b7…c6544 and ELF904038e1…3f3ff
pass63release-candidate checks15.10s and installed identity checks. Build
RUSTFLAGS are empty; explicit libpython flags apply only to Rust test binaries.
Previous installed files/metadata and rollback package are preserved. Marker/
package availability does not grant Source, formal or fit rights.

Actual nonformal seed17/max2update parents retain all original budgets and2GiB
storage. CPU-inference/CUDA-learner continuous0619c53e completes2updates/32natural/
419selected/16optimizer steps in about1135s. CUDA-inference/CUDA-learner5dc00dce
pauses1 then resumes2, also32natural/419selected/16steps. Fresh complete-parent
native original/numeric/generation audits exit0 in176.95s and172.44s respectively,
with unchanged run bytes. This accepts64natural games/4updates in Native storage/
runtime/resume scope; no formal Source qualification or policy strength follows.
Long revalidation time is retained/reported rather than hidden by weaker checks.

## Identical-original HDD archival comparison (2026-11-11)

All six predeclared AB/BA pairs complete,12PASS/exit0, on the verified D: USB
HDD/9p mount. Each trial archives/restores the SAME32groups/192historical games,
4492inventory members/273,062,500bytes plus4,212,418original control bytes.
Inputs remain unchanged; these new diagnostic archives grant no Source rights.
The eleven storage/reader modules used in the jobs retain their frozen hashes.

| Median across6trials | Directory archival reference | Packed archival Writer |
| --- | ---: | ---: |
| Complete write/publication/retirement | 166.4435s | 19.8487s |
| Open/read/verify/recheck/close | 171.5748s | 734.0756s |
| Logical write throughput | 1.5887MiB/s | 13.3257MiB/s |
| Logical read throughput | 1.5412MiB/s | 0.3602MiB/s |
| Final apparent bytes including trial catalog | 277,274,918 | 23,646,793 |

The directory writer is a documented exclusive-write/file-fsync/group-directory-
sync archival reference, not a claim to reproduce the production collector.
Packed trials use production encoding/Writer/sealing, preserving exact old
controls; they do not invoke current collection preparation with a new metrics
control. Readback integrity paths differ: directory rereads originals, while
public packed loans repeatedly hash the physical DB/descriptor. No checks were
disabled to improve timings. The current packed path trades faster/smaller
archives for slower verified reading; the directory default stays unchanged.

| Per-group descriptive p95 / p99,192observations | Directory | Packed |
| --- | ---: | ---: |
| Archive operation | 7.6716 /17.8417s | 0.5226 /1.2767s |
| Readback operation | 3.3907 /3.9635s | 32.0066 /37.4495s |

Packed trial medians: compression0.5029s, provisional submit/ACK total5.8645s,
controls plus complete seal11.4932s, actual SQLite checkpoint3.6083s and
DELETE transition0.1014s. These phases overlap/nest and must not be added again
to total write time. Sampled DB/catalog/sidecar peak median49,383,496B; the
append-only directory reference reaches277,274,918B at completion. RSS trial
medians are111,298,560B directory and203,911,168B packed, under the allocator/
shared-page scope below, not isolated live-memory overhead. Owned tmpfs is0 in
this Writer-only panel. Queue occupancy is recorded; isolated queue wait is not.

All raw pairs/group observations, type7 p95/p99/range summaries, queue/compression/
submit/checkpoint/DELETE/combined-seal phases and sampled process/DB/WAL/SHM/
journal/capacity data are retained in
`runs/assessments/sqlite-group-pack-v1-20261111/physical-comparison/`.
Each format has192group observations, providing weak support for tail inference.
Windows/WSL caches are uncontrolled; completed-output first reads are not cold-
device results. Apparent file sizes exclude filesystem metadata/cluster allocation.
Sampled RSS includes shared pages and retained allocator arenas in the reused
coordinator; it is not isolated per-format incremental/physical RAM cost. The
managed2GiB peak includes a temporary whole-available validation reservation.

Bounded single-job ext4 Native development builds/tests overlap part of these
jobs; the exact first compile interval is unavailable and later intervals are
recorded. Compiler processes are outside the benchmark's process RSS scope.
No CPU isolation, causal per-trial contention attribution or default/strength
conclusion is inferred. Independent arithmetic/scope audit PASS recomputes the
raw pairs, bytes, medians/ranges/type7 tails and checks all11source hashes;
its report is retained alongside the raw data. The actual
concurrent collection/learning panel remains a separate required measurement.

## Remaining verification

- PASS scoped: standalone in-memory artifacts/RAM logging/joint admission,
  independent reviews, actual18game/inference saved-byte audits above.
- PASS scoped: production Python Source/PPO update/pause-resume below.
- PASS scoped: native packed provider parity/ordinary PPO/resume above.
- NOT RUN: pending final64/Dataset/Tree paths and affected actual CUDA checks.
- NOT RUN: whole pytest/Ruff/mypy/Rust, current wheel/mandatory-only/old-data proofs.
- PASS scoped: alternating same-HDD archival save/read comparison above.
- FAIL first concurrent generation/learning block: first directory startup lacks
  parent/artifact-registry.json before any update; remaining3trials NOT RUN.
  Original failed block63f2b9bef0f34cfe97e5ea599ea9245e and retirement ledger are
  retained. Main d3e15f9 runtime sources stay frozen during diagnosis; a new
  operational comparison and generation/wait/p95/p99/RAM/DB/WAL report remain due.

The existing ledger identifies no complete genuine formal parent/final64 Source;
short two-update acceptance parents are permanently nonformal and cannot be
retrofitted. The plan forbids a formal campaign/HPO/final test. Future component
final64 tests must remain distinguished from any unrun genuine formal handoff.

## Lineage control publication on D: (2026-11-11)

The first concurrent block fails before collection at initial registry publication.
A control-only reproduction fails6/6on D: and passes6/6on /tmp. A separate12case
actualHOST probe isolates open-FD rename visibility: writable/readonly FDs held
across replace hide the destination6/6; both close-before variants succeed6/6.
These are diagnostic controls, not games or qualified Sources.

The owned lineage writer now confirms writer close before its single mutable
replace or immutable exclusive link/unlink. It binds final readonly verification
to the SAME controlled inode/mode/length/mtime/hash, permits ctime handoff only
for its controlled close/namespace effects, then requires strict six-field checks
through actual readonly close. Scan snapshots retain their existing five fields
and advance only after every check/retirement. Actual uncertain descriptor stages
and the first error remain owned; no polling/rename retry/Source relaxation.

Both independent reviews READY. Author's372disjoint checks pass (56publication,
85historical lineage/resume,212Source/formal plus19affected HOST CPU rechecks),
Ruff/fullmypy172 and8frozen algorithm AST checks pass. Reviewer independently
runs56publication/40compatibility checks and bounded cancellation/replacement/
budget/deadline probes. Root then runs12actualHOST cases across D:/tmp,
ExperimentConfig/GroupPackExperimentConfig and3repetitions: two mutable writes,
one immutable publication and one refused immutable overwrite per case. AllPASS;
canonical hashes/scanner5tuple remain exact, source60674ee5/tests4931bdbf frozen.
No game/learner/qualified-source experiment runs in these probes.

The sandbox Source fixture fails its inherited Unix socket with EPERM before
numeric launch; exact HOST selectors pass19cases with10existing CUDA skips and
14deselected. No production timeout/skip change. The initial60s historical-test
log was overwritten by the85PASS rerun; the report records its timeout. The first
sandbox template directory was unavailable when root copied receipts; its batch
log and the separate sandbox recheck's stderr/lifecycle remain preserved.
Artifacts/reviews/raw probes/patch/log inventory are retained under
`runs/assessments/sqlite-group-pack-v1-20261111/lineage-publication/`.
This accepts the scoped control fix; final64/Dataset/Tree, whole verification and
a new actual concurrent comparison remain pending.

## Full-parent HDD concurrency failure and revised condition

Blockcd82db731f234913acbeb2f7aae8ad38 uses Main675ff8c and frozen harnessc84f9d08.
Its first DIRECTORY parent pauses after1update (234accepted decisions), resumes
and receives2actual updateACKs, with32recorded games/419accepted decisions and
one source-request window overlapping the independent6game collector's action
HTTP. Actual generation0/1/2commit-return anchors are present. At2188.729s the
parent fails during retention: _prune_staged requires private0700, but the actual
new D:9p staging directory reports0777. Whole parent remains FAIL/ineligible,
completed_updates null; these partial observations are not successful runtime
acceptance. The remaining packed/packed/directory trials stay NOT RUN.

The failure ledger confirms permanent admission fence, settled creations,
retirement of193actual Popen and19async-process capabilities, no remaining
unknown tasks/transports/descendants, and the supervised failure-only OS boundary
exits86. No source repair/resume/pruning skip/permission relaxation follows.
Original raw/derivedFAIL, exact source/harness/config/device metadata, STARTED
snapshot, bounded progress diagnostics and terminal receipt are preserved under
`runs/assessments/sqlite-group-pack-v1-20261111/concurrent-comparison/block-cd82db731f234913acbeb2f7aae8ad38/`.
The reducer's nested control-read status is not whole-parent Source health.

Root accepts the advisor's explicit RevisedB condition in
[comparison protocol](../design/sqlite-group-pack-comparison.md): ONE new4trial
block with PPO parent/checkpoint/numeric/control/original storage all on common
ext4 in BOTH formats, independent6game collector writes on the SAME D:HDD, with
distinct DB/catalog/RAM roots. Existing wall/resource/update/device/seed budgets
are unchanged. This measures the combined ext4PPO+HDDcollector workload, not
successful full-parent-on-D support or isolated per-component/optimizer-I/O
causation. PanelA already measures identical-real-byte alternating HDD archival
storage. Do not pool these conditions or infer a default change. RevisedB
implementation reviews/probes/execution/audit remain NOT RUN at this entry.

## Final sources: candidate2 acceptance and filesystem limits

Both full independent reviews accept candidate2 after one bounded fix round.
The original wrapper acquire/refund-after-effect defect, premature last-revoker
release in both terminal publishers and missing optional artifact preflight
checks are resolved. Independent correctness81focused+1HOST cancellation checks
and three actual account fault reproductions pass; conformance90checks pass.
The broader sandbox review run was interrupted, not a successful suite. The
candidate2 delta is d162a07d (54files), report61abc30e/manifestbb9b072f; its1469
HOST checks/116explicit skips and exact-sourceRust157/4ignored are retained.

Post-review control-only filesystem checks finish30PASS/6NOT_EXERCISED across
ext4/D: and FormalRun3/4 plus all4Tree contexts. Healthy controlled capabilities
close without identity drift, and visible late mutations are rejected/revoked.
On D: the compound directory-close/move/foreign-inode construction fails before
foreign creation when fstat of the held original itself reportsENOENT. Actual
revoker/error/uncertainty remain retained; committed/revoked/health are not set.
These6cases do not prove successful invalidation or application retirement.
Each remaining case runs in its own supervised process; OS exit is separately
confirmed. The first aggregate probe's FD inventory omitted fstat errors and
cannot establish retirement. Source/test hashes remain frozen. The reconstructed
probe does not establish the cause of the earlier lost timestamp-only failure.
Detailed retained traces/limits are in final-sources/candidate2/filesystem-recovery.

The Tree control primitive omits its semantic fitted-pair postcheck and formal
metadata lacks real execution/stop evidence. All controls are unqualified, with
no Source, final64 or fitting permission. Genuine complete formal parent/final64
and qualified Dataset4 fitting remain NOT RUN. The actual unqualified fixture
96train/192validation/500round fit and all own-bank directory/packed Rust64
comparisons remain component evidence only.

## Final integrated build and checks (in progress)

Root integrates the exact reviewed delta, preserving accepted lineage writer
ASTs, then activates the two specified packed capability declarations. Final
native sourcef8dd9fa7072441562130a118f1cbb8b6a9cd41f8b843307d782839496de68cd3
builds release/MIMALLOC with emptyRUSTFLAGS; installedELF
da042804ecd79c9c943718e30c52903376955ff94d777e38b6e7f7817ec3bb75
matches its wheel. The prior904038e1package/metadata are saved in final-build/
native-rollback. Appwheel/sdist source correspondence174files, native88source/
fixture/config files and7embedded assets pass. Distribution dependency/extras/
entrypoint/version/Python metadata separately match projectTOML. Mandatory-only
installed baseline passes with all optional model/native stacks absent.

Final wholeRust157PASS/4ignored109.07s, actualCUDA58PASS34.53s, Ruff(src/tests)
and fullmypy172PASS. The wholeHOSTpytest initially finishes4754PASS331SKIP with
2failures and4NumPywarnings1027.46s: an old-binary test used mutableCargo output,
and an old signature test omitted the accepted optional provider. The immutable
D093archive is preserved; the reviewed test-only corrections pin it and check
old call bindings/defaults plus both exact packed markers. Both entire testfiles
pass137checks/64existing opt-in skips; both formerly failed cases pass and neither
is skipped. No app/native bytes change for those corrections. Initial whole
failure logs remain; this is combined evidence, not a fabricated fresh wholePASS.

All five full old-data jobs and audits PASS against the final native: history,
observation, Parquet index and saved-PPO1/4workers. Their source inventory predates
only those two reviewed Python test edits and remains preserved, not rewritten.
Root records that test-only difference explicitly. CPUpreload safety opt-ins,
revisedB execution/audit and whole-branch final reviews remain pending here.

Final CPU preload safety opt-ins PASS199checks107.08s with the documented flag;
no production timeout/skip/import workaround. The two-file compatibility test
delta independently passes137checks/64existing opt-in skips and both original
failures explicitly pass (not skipped), with exact preservedD093 and installed
ELF unchanged. Ruff/whitespace pass. These test-only edits do not change either
built package or any application/native source.

The first revised workload fails BEFORE any trainer/collector/monitor admission
at its drive-source assertion (0.118s). The current kernel reports escaped D:\134
rather than the earlier literal D:. All2diagnostic children retire, supervisor86;
all workloads/remaining trials NOT RUN. Raw failed receipt35a8a62a is preserved.
The mount parser correction is temporary measurement code only: strict single-
pass standard mountinfo escape decoding with exact D: or D:\ root identity,
raw token retained, all filesystem/device/root/physical-HDD checks preserved.
Independent review and actual admission on the current mount precede execution.
No block is extended or silently repaired.
