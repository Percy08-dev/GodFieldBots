# Group Pack implementation ledger

Source intent: owner plan supplied 2026-11-10. Baseline: 83aa67e7707b267217247afbc4c3efcf14204a94.

- [x] Read current architecture, relevant v2 persistence/Source ADR sections,
  Python 3.14 storage availability and Simulator normative input/launch contracts.
- [x] Draft physical contract, custody/failure boundaries and staged gates.
- [x] Foundation architecture review READY: exact namespace/loan semantics,
  Writer retirement before owned descriptor publication, IPC/seal/catalog closure.
- [ ] Implement Pack, Writer/SQLite/catalog, sealing and ArtifactSource with tests.
- [ ] Independent correctness and architecture conformance of foundation.
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
