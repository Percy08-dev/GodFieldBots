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

## Remaining verification

- PASS scoped: standalone in-memory artifacts/RAM logging/joint admission,
  independent reviews, actual18game/inference saved-byte audits above.
- PASS scoped: production Python Source/PPO update/pause-resume below.
- NOT RUN: native packed provider parity/ordinary PPO.
- NOT RUN: pending final64/Dataset/Tree paths and affected actual CUDA checks.
- NOT RUN: whole pytest/Ruff/mypy/Rust, current wheel/mandatory-only/old-data proofs.
- NOT RUN: alternating same-HDD save/read and concurrent generation/learning
  comparison with throughput, generation/wait/p95/p99/checkpoint/RAM/DB/WAL metrics.

The existing ledger identifies no complete genuine formal parent/final64 Source;
short two-update acceptance parents are permanently nonformal and cannot be
retrofitted. The plan forbids a formal campaign/HPO/final test. Future component
final64 tests must remain distinguished from any unrun genuine formal handoff.
