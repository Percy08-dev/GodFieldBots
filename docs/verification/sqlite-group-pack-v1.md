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
Runtime, Source/lineage/PPO, native, final64, Dataset/Tree and whole-plan comparison
are not implemented/accepted at this point. This record distinguishes component
evidence from the complete owner's requested outcome.

## Foundation candidate evidence

Environment: CPython3.14.10, SQLite3.54.1, stdlib compression.zstd. No new dependency.
Baseline Bots83aa67e; design milestone495ecec; foundation files remain uncommitted
pending reviews. The implementer's report records64focused cases including real
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

## Remaining verification

- NOT RUN: runtime in-memory artifacts/RAM logging and joint monitor/admission.
- NOT RUN: production Source/PPO update/pause-resume and native provider parity.
- NOT RUN: pending final64/Dataset/Tree paths and affected actual CUDA checks.
- NOT RUN: whole pytest/Ruff/mypy/Rust, current wheel/mandatory-only/old-data proofs.
- NOT RUN: alternating same-HDD save/read and concurrent generation/learning
  comparison with throughput, generation/wait/p95/p99/checkpoint/RAM/DB/WAL metrics.

The existing ledger identifies no complete genuine formal parent/final64 Source;
short two-update acceptance parents are permanently nonformal and cannot be
retrofitted. The plan forbids a formal campaign/HPO/final test. Future component
final64 tests must remain distinguished from any unrun genuine formal handoff.
