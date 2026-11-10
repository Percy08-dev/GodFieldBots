# Group Pack implementation ledger

Source intent: owner plan supplied 2026-11-10. Baseline: 83aa67e7707b267217247afbc4c3efcf14204a94.

- [x] Read current architecture, relevant v2 persistence/Source ADR sections,
  Python 3.14 storage availability and Simulator normative input/launch contracts.
- [x] Draft physical contract, custody/failure boundaries and staged gates.
- [x] Foundation architecture review READY: exact namespace/loan semantics,
  Writer retirement before owned descriptor publication, IPC/seal/catalog closure.
- [x] Implement Pack, Writer/SQLite/catalog, sealing and ArtifactSource with tests.
- [x] Independent correctness and architecture conformance of foundation.
- [ ] Review and implement memory/RAM collection plus explicit Config.
- [ ] Review and implement all Python/Rust/PPO/resume/final64/Dataset/Tree readers.
- [ ] Independent full integration reviews and resolved blocking findings.
- [ ] Full checks, bounded runtime/CUDA/wheel/legacy verification.
- [ ] Alternating same-real-data/same-HDD and concurrent workload comparison.
- [ ] Coherent verified local commits and final acceptance report.

Ruling: execute the supplied plan without a new approval interview — the user
explicitly requested implementation in fresh context; detailed boundary choices
remain subject to the repository's architecture review rather than repeated
user permission.

Ruling: use a dedicated branch in the existing shared checkout and retain all
pre-existing untracked artifacts — existing environment and Simulator sibling
paths are source-pinned here; no unrelated files are staged or removed.

Preflight:

| Tasks/interface | Produced/consumed | Disposition |
| --- | --- | --- |
| Foundation internally | Pack → Writer → DB → descriptor → ArtifactSource | Closed schema, bytes, deadlines and loans pinned in design; architecture READY |
| 1 → 2 | StorageSettings/Writer → runtime coordinator | No consumer until reviewed memory/RAM account and explicit Config |
| 1 → 3 | ArtifactSource → Python/native original ownership | Stable logical proof and actual physical custody gate before integration |
| 2 → 3 | Sealed collections → PPO/final64/Dataset | Pending physical state cannot imply healthy logical Source |
| 1/2/3 → 4 | Tested interfaces → complete comparison | Component passes cannot substitute for all-path acceptance |
| Task 1 self | bounded encode/write/seal/read tests | No production consumers or default/config mutation |
| Task 2 self | in-memory output and RAM logs | Exact file ownership to be assigned after design gate |
| Task 3 self | all original readers | Explicit lineage/native proof changes required before code |
| Task 4 self | verification and real same-HDD comparison | Unrun cases must be identified; no synthetic strength/throughput claim |

Runtime design READY (2026-11-10) after joint decode budgeting, actual sync-worker
retirement, typed logical sidecar evidence, composite inode-owned publication and
explicit early command guards. The later narrow pending-predecessor/FD-invalidation
delta also passes architecture review. Runtime code awaits foundation acceptance.

Foundation implementer reports codec/Writer/Source failure tests progressing;
no production consumer is changed. Root's clean baseline command
`.venv/bin/python -m pytest -q tests/test_config.py tests/test_card_instance_collection_publication.py`
passed36checks in0.50s. Implementation evidence still awaits independent review.

Ruling: revoke through SAME held/verified recovery FD rather than conditional
pathname unlink — no atomic stat→unlink inode predicate exists in the stdlib.
Invalid diagnostic evidence remains; foreign replacements are not modified by
FD writes. Uncertain close/invalidation retains the actual owner and first error.

Task1 independent review: correctness BLOCKED (seal DB substitution, unadmitted
namespace allocation, constructor/helper primary-error cleanup); conformance
BLOCKED (same plus stale-FD reuse and simultaneous first-use catalog race).
Reports: `/tmp/gfb-group-pack-task1-{correctness,conformance}.md`.
Root100-test/Ruff/mypy pass does not resolve these untested failures.

Task1 fix round1 assigned: one-attempt FD state/uncertainty everywhere in new
modules; streaming/admitted namespace enumeration; BEGIN IMMEDIATE catalog
schema/version/update; child+parent held original DB custody and expected seal
through retirement/publication. Architecture confirms held-inode remedy READY
without schema/IPC changes. Save pre-fix snapshots in
`/tmp/gfb-group-pack-task1-before-fix/` for scoped delta reviews.

Task1 fix round1 reviewed: F1–F4 and both coverage gaps ADDRESSED. Architecture
conformance READY. Root121affected/legacy tests PASS in5.32s; scoped Ruff/mypy
PASS. Correctness found one introduced P2: Writer's now-retained128MiB base plus
publication inspection's independent full allowance can over-admit71350bytes
at a256MiB limit. Fix round2 assigns remaining-joint-budget inspection; no early
reservation release. Snapshot `/tmp/gfb-group-pack-task1-before-fix2/`.
If that issue survives this second fix/verification cycle, follow AGENTS's
advisor escalation rather than extending an unbounded repair loop.

Task1 fix round2 complete: remaining joint allowance corrected, exact tight
refusal/nearby success regressions pass. Correctness and architecture both READY,
no remaining blockers. Root fresh123affected/legacy checks PASS in5.54s, scoped
Ruff and mypy6PASS, links/diff check PASS. Foundation accepted for local milestone;
no production consumers and no complete-plan acceptance yet.

Target-filesystem disposable probe PASS on actual D:HDD and `/dev/shm`: WAL,
TRUNCATE checkpoint(0,0,0), DELETE transition, file/directory fsync, unchanged
readonly bytes and no sidecars. Escalated execution is required for these default
paths under this sandbox; user explicitly authorized them in the plan.

Ruling: correct the two test-coverage suggestions in the same wave — canonical
mutations must reach actual range/type checks, and original plan explicitly
requires real queue saturation/waiter wakeup rather than only oversized refusal.
No broadened production scope or default change is introduced.

Python Source/lineage/PPO design is READY after exact proof2 physical custody,
shared pool/prepare_commit retirement, run4/update4/wire5 and external disk-cap
accounting were specified. Native3B design is also READY after exact plan4,
borrowed-byte provider, summary/authorization/receipt variants and unchanged
numerical DTO choices were pinned. Formal/final64/Dataset outlines remain gated.
