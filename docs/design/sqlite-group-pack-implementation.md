# SQLite group Pack implementation plan

> **For agentic workers:** use superpowers:subagent-driven-development. Root
> assigns implementation and independent review to the repository's custom roles.
> Only root stages/commits. Do not spawn further agents.

**Goal:** Implement and verify the owner's complete opt-in group-Pack plan.
**Architecture:** Immutable per-collection SQLite DB, coordinator zstd3 Pack,
bounded Python Writer, original-byte ArtifactSource, preserved policy boundaries.
**Tech Stack:** Python 3.14 stdlib sqlite3/compression.zstd; existing Python/Rust stack.
**Spec:** [sqlite-group-pack-v1.md](sqlite-group-pack-v1.md), ADR0034.

## Global constraints

- Preserve old Config serialization/hashes, Collection5/Journal6/codec2 and default.
- No edits to Simulator, no third-party storage dependency, no Rust SQLite.
- No Source/fit/update authority from a Pack, descriptor or catalog entry.
- No whole-parent resident export or extracted-file substitution for DB custody.
- Keep reservations/ownership through confirmed child/Source cleanup.
- Use bounded deadlines and byte admission before allocation.
- No formal training campaign, HPO, final test, migration or automatic deletion.
- Retain all pre-existing untracked work; root alone stages/commits.

## Review focus

- Same-size descriptor/DB substitution must fail after original admission.
- Queued/writing allocations and shared RAM outputs must remain jointly admitted.
- Partial IPC/Writer failures must stop admission and preserve cleanup ownership.
- Sealing must survive checkpoint/catalog/publication failures without false health.
- final64 must publish only after SAME parent/Source healthy retirement.

### Task 1: Storage foundation

**Files:** create `src/gf_common/group_pack.py`, focused
`src/gf_training/group_pack_*.py`, and `tests/test_group_pack*.py`.
Do not modify production consumers, Config, native source or unrelated files.
Root owns docs and project configuration.

**Interfaces:** spec's `PackMember`, `StorageSettings`, encode/decode,
`GroupPackWriter`, `CollectionLocator`, `ArtifactSource`, source open and group
loan interfaces. A `gf_training.group_pack_db` module owns exact SQL and
seal/descriptor validation, reused by Writer and readonly Source. Private
transport/resource helpers may live in their own focused storage modules.

- [ ] Write and run failing byte-exact round-trip tests with all seven kinds,
  zero-length and non-UTF8 member payloads, sorted names, arbitrary group sizes.
- [ ] Implement the exact binary/index contract and one zstd3 frame; enforce
  compressed/raw/index/count bounds before allocation where possible.
- [ ] Add corruption/duplicate JSON keys/names, unsafe path/range/hash/trailing
  frame/oversized-expansion refusal tests and verify passes.
- [ ] Write and run failing actual-child SQLite transaction tests, including
  nullable game IDs, 2/6/other-sized groups, idempotent exact resend and conflicting
  group/episode/control resend with rollback of the entire group.
- [ ] Implement Writer, bounded binary framing/deadlines and byte reservation.
  Capture child errors without logging original private payloads.
- [ ] Test child kill/truncated IPC/open transaction rollback, capacity saturation,
  simulated disk-full/failed checkpoint, cancellation/timeout and confirmed reap.
- [ ] Implement ordered sealing, DELETE-mode readonly namespace verification,
  fsync/no-overwrite external descriptor, short FULL catalog transactions and
  crash reconciliation from actual originals. Never repair/mutate a sealed DB.
- [ ] Test seal/catalog/publication failure windows and bare-Pack/no-authority.
- [ ] Implement directory and packed ArtifactSource with retained physical
  descriptor/DB/inventory identities, exact bytes and group reload/generation loans.
- [ ] Test same-size mutation/replacement, symlink, changed descriptor, unexpected
  sidecars/namespace, live-loan close/eviction refusal, stale loan and reload.
- [ ] Run `.venv/bin/python -m pytest -q tests/test_group_pack*.py`, scoped Ruff
  and mypy, then report changed files, exact checks, limitations and design deltas.
- [ ] Independent correctness and architecture conformance; resolve blockers,
  root reruns affected verification and commits a coherent milestone.

### Task 2: Memory collection and explicit Config

**Files:** precise ownership and interfaces will be appended by root after the
runtime integration design review; likely common storage settings, journal6,
environment output seam, candidate publication and corrected CLI composition.
Requires an accepted integration design before changing any consumer.

- [ ] Specify and review exact bounded in-memory artifact writer and RAM log seam.
- [ ] Add failing compatibility/collection/failure tests, implement reviewed seam.
- [ ] Test real baseline/evaluation and all incomplete/cancelled group evidence.
- [ ] Independent task reviews, focused verification and local commit.

### Task 3: Original-source and Python/Rust consumers

**Files:** root will append exact ownership after Source/native design review.
Requires explicit stable logical proof, generation/resume variant, native byte
provider and pending final64 interfaces; no inferred extraction adapter.

- [ ] Review exact Python/SealedScan/lineage/native/final64/Dataset3 interface changes.
- [ ] Tests first; connect all original-source consumers and safe group retirement.
- [ ] Verify artifact reconstruction and Python/Rust features/numerics parity.
- [ ] Verify bounded PPO update/pause/resume/final64/Dataset/Tree conditions.
- [ ] Independent correctness/conformance, resolved blockers and local commit.

### Task 4: Whole verification and comparison

- [ ] Whole pytest, Ruff, mypy, Rust and affected actual CUDA checks.
- [ ] Source-matched wheel and mandatory-only startup, original Config/data proofs.
- [ ] Alternating same-real-data/same-HDD old/new read/write comparison and
  concurrent generation/learning measurements, with p95/p99 and resource metrics.
- [ ] Record exact evidence/NOT RUN conditions in `docs/verification/`.
- [ ] Whole-branch correctness/conformance review; root final acceptance only
  when the complete owner scope passes, with directory default unchanged.
