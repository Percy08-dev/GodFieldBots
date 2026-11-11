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

- [x] Write and run failing byte-exact round-trip tests with all seven kinds,
  zero-length and non-UTF8 member payloads, sorted names, arbitrary group sizes.
- [x] Implement the exact binary/index contract and one zstd3 frame; enforce
  compressed/raw/index/count bounds before allocation where possible.
- [x] Add corruption/duplicate JSON keys/names, unsafe path/range/hash/trailing
  frame/oversized-expansion refusal tests and verify passes.
- [x] Write and run failing actual-child SQLite transaction tests, including
  nullable game IDs, 2/6/other-sized groups, idempotent exact resend and conflicting
  group/episode/control resend with rollback of the entire group.
- [x] Implement Writer, bounded binary framing/deadlines and byte reservation.
  Capture child errors without logging original private payloads.
- [x] Test child kill/truncated IPC/open transaction rollback, capacity saturation,
  simulated disk-full/failed checkpoint, cancellation/timeout and confirmed reap.
- [x] Implement ordered sealing, DELETE-mode readonly namespace verification,
  fsync/no-overwrite external descriptor, short FULL catalog transactions and
  crash reconciliation from actual originals. Never repair/mutate a sealed DB.
- [x] Test seal/catalog/publication failure windows and bare-Pack/no-authority.
- [x] Implement directory and packed ArtifactSource with retained physical
  descriptor/DB/inventory identities, exact bytes and group reload/generation loans.
- [x] Test same-size mutation/replacement, symlink, changed descriptor, unexpected
  sidecars/namespace, live-loan close/eviction refusal, stale loan and reload.
- [x] Run `.venv/bin/python -m pytest -q tests/test_group_pack*.py`, scoped Ruff
  and mypy, then report changed files, exact checks, limitations and design deltas.
- [x] Independent correctness and architecture conformance; resolve blockers,
  root reruns affected verification and commits a coherent milestone.

### Task 2: Memory collection and explicit Config

**Files:** create `src/gf_common/artifacts.py`,
`src/gf_training/group_pack_memory.py`, `group_pack_collection.py`,
`group_pack_publication.py`, and `tests/test_group_pack_runtime*.py`.
Modify only existing `src/gf_training/group_pack_resources.py`,
`group_pack_writer.py`, `group_pack_source.py` (joint budget, retained pending
owner and checked optional predecessor), `src/gf_training/card_instance_journal.py`,
`structured_journal.py`, `card_instance_collection.py`,
`src/gf_simulator_client/environment.py`, `card_instance_environment.py`,
`src/gf_cli/config.py`, `card_instance.py`, `__init__.py`, plus directly affected
existing tests if justified. Runtime design is accepted; implement only after
foundation independent reviews/acceptance. Root owns docs/project dependency files.
Approved narrow additions: `structured_source.py` only rejects logical_sidecars
in its existing file-only reader; `gf_common/group_pack.py` StorageSettings only
rejects max_pack_bytes above physical v1's256MiB ceiling, with tests. No packed
learner Source/proof/lineage implementation enters Task2.

**Interfaces:** exact config/types, byte-journal seam, memory sink/snapshot,
joint StorageBudget, composite CollectionPublication and command/model guard
matrix in `sqlite-group-pack-runtime.md`. Preserve all old defaults and subtype
serialization. Raw byte input is not Source authority. New production commands
remain limited to standalone smoke/collect/evaluate baseline/neural inference
until the subsequent Source/native/final64/Dataset gates are implemented.

- [x] Specify and review exact bounded in-memory artifact writer and RAM log seam.
- [x] Add failing compatibility/collection/failure tests, implement reviewed seam.
- [x] Test real baseline/neural inference and controlled incomplete/cancelled group evidence.
- [x] Explicit live evaluate command and complete saved-byte audit pass6natural games.
- [x] Independent task reviews, focused verification and local commit.

### Task 3: Original-source and Python/Rust consumers

**Files:** exact ownership and reviewed interfaces are recorded in Tasks3A/3B/3C
below. Logical proof, generation/resume, native byte provider and pending final64
remain separate checked contracts; no extraction adapter is Source authority.

- [x] Review exact Python/SealedScan/lineage/native/final64/Dataset4 interface changes.
- [x] Tests first; connect all original-source consumers and safe group retirement.
- [x] Verify artifact reconstruction and Python/Rust features/numerics parity.
- [x] Verify bounded PPO update/pause/resume/final64/Dataset/Tree conditions.
- [x] Independent correctness/conformance, resolved blockers and local commit.

#### Task3A: Python original Source, lineage and ordinary PPO

Requires Task2 acceptance. Design: Python sections and closure of
`sqlite-group-pack-source.md`; optional predecessor delta and PackedRunResult
amendment are accepted. Native/formal/final64/Dataset production entry guards
stay in force until their implementations.

**Assigned files:** new focused common `group_pack_game.py` contract helpers;
new focused training `group_pack_pool.py`, `group_pack_game_source.py`,
`group_pack_controls.py` and/or `group_pack_lineage.py` only where actual
responsibility needs extraction (do not create empty layers). Existing
`gf_training/group_pack_source.py`, `group_pack_resources.py`,
`group_pack_collection.py`, `group_pack_publication.py`, `structured_source.py`,
`card_instance_collection_source.py`, `card_instance_source.py`,
`structured_lineage.py`, `collection.py`, `card_instance_ppo.py`, `ppo_run.py`,
`structured_learning_process.py`, `structured_numeric_worker.py`,
`structured_rollout.py`; `gf_models/structured_sealed.py` only the generic
physical verified_digest accessor/no training import;
`gf_cli/card_instance.py`/`__init__.py` only ordinary packed PPO composition/guards.
Approved narrow ownership additions: `structured_journal.py` exact concrete
lower-layer verified-original reader adapter gate (no journal→high Source→journal
cycle), and `gf_cli/config.py` ordinary Python PPO collect/standalone=False guard
only. Actual parent/ledger/plan checks remain mandatory; all later guards stay.
Tests `test_group_pack_game*.py` and directly affected original tests if needed.
Root retains docs/dependencies/asset ownership. No native/Simulator edits.

**Deliverable interfaces:** checked storage_binding; PackedSourceOwner/one-group
shared pool and PackedBatchCustody; tagged proof2 with persistent physical custody;
separate logical_files and verified_digest; run4/update4/resume/fingerprint and
wire5 paired recipes; explicit PackedFinalizedCollection including storage_metrics;
external owned_bytes roots; checked precommit FD/pool retirement with frozen
in-memory proof. Exact values/closed fields are in the accepted Source design.

- [x] Red→green stable proof2/custody across reopen; byte-identical replacement
  across pause/reopen fails; bare Pack/descriptor/metadata cannot issue updates.
- [x] Full logical inventory/row checks, original-byte journal/candidate/replay
  parsing, opaque diagnostics never enter inputs; old proof1/hash tests stay exact.
- [x] Shared one-group cache, stale/live borrow checks, requested-history group
  reload and actual graph/FD/budget retirement; no whole-parent raw residency.
- [x] New source/control callbacks share SAME actual owner through optimizer and
  checked prepare_commit; constructor/close/cancel failure tests preserve owners.
- [x] Tests for exact run4/update4/wire5, frozen resume identity and wrong/mixed
  variants; numeric4 arrays/candidates/selected actions equal original references.
- [x] Preserve every base formal algorithm kernel AST (including training_groups,
  validation_groups, ppo.gae, normalize_advantages, ppo_math.losses, ppo_run._controls,
  structured_numeric.Learner.__init__/update). Storage must not change math,
  ordering, reward, optimization or existing saved numerical semantics.
- [x] Real external DB/WAL/descriptor/failed bytes count in the same parent disk
  cap before optimizer, after update/finalization and native scratch allowance.
- [x] Controlled ordinary PPO collect/update/pause/resume with actual numeric
  wire5 and old unconfigured regressions. No formal parent/campaign/Tree gate.
- [x] Independent reviews and root bounded real smoke, scoped/static acceptance
  and coherent commit before dependent native work.

#### Task3B: Native original-byte provider

Requires Task3A acceptance. Design: Native3B section/closure in Source design,
already architecture READY. One writer/implementer at a time.

**Assigned files:** training native provider adapter, `native_card_instance.py`,
`native_structured_session.py`, `native_backend.py`, relevant original-source
composition, and minimal RAW consumers; Rust `structured_session.rs`,
`structured_source.rs`, `structured_journal.rs`, `card_instance_source.rs`,
`card_instance_original_episode.rs`, `structured_controls.rs` and necessary
control helpers, `lib.rs`, `gf_native.pyi`/build_info capability declaration.
New focused trait/bridge helper and affected tests permitted. No SQLite/zstd
crate, new dependency, engine copy or formal/final64 implementation. Root handles
capability activation and source-matched packaging after reviews.

- [x] Explicit plan4 registry/describe/OriginalByteLease bridge and generic Rust
  provider, exact summary/auth/receipt contracts; keep plan3/old APIs unchanged.
- [x] All direct member/stat/inventory and original16/RAW/restage uses covered.
- [x] Same original bytes → exact candidate/selection/token/feature/numeric
  arrays; no Python fallback or authority from provider/DTO/marker.
- [x] Metadata/one-group/native copies jointly admitted; escaped bank/lease,
  mutation, callback failure, cancellation and terminal close tests.
- [x] Rust/scoped Python/static checks and private candidate parity; both reviews,
  then root source-matched capability/wheel acceptance and ordinary Rust PPO smoke.

#### Task3C: Formal, final64, Dataset4 and Tree3 paths

Requires Task3A/3B acceptance. Design:
`sqlite-group-pack-final-sources.md` (architecture READY). Root materializes the
exact reviewed variant JSON/package copy and hash; no frozen base/kernel change.

**Assigned files:** common formal/collection/game-tree contracts; training
formal/final-validation/custody/game-tree-data, native final-validation adapter,
structured tree fitting; model tree pairing/artifact/runtime/worker provenance
dispatch; CLI corresponding guards/composition; Rust formal/final-slot/control
provider dispatch and package stub/capability declarations; focused tests. No
new runtime dependency/Simulator edits. Root owns docs/assets/capability activation.

- [x] Explicit storage-only formal variant and exact unchanged budgets/algorithm;
  old declarations/stop/parents remain exact and cannot be retrofitted.
- [x] SAME pending owner/private reader → Source/actor close → actual parent close
  → public facts/pointer commit; every failure/cancellation/late-swap boundary.
- [x] Exact native request2/summary2/final-slot provider metadata and four-method
  lifecycle/escaped packet retirement; no flag/FD grants Python Source authority.
- [x] Dataset4 full three-parent/three-public-final64 qualification/homogeneity,
  labels/splits/correspondence/provenance, explicit paired fit5 context and Tree3
  inference4 subtype; reject mixed storage and short/fixture promotion.
- [x] Original input/feature/label/score equality, controlled fixtures and fresh
  fitting without claiming real Source qualification; no prohibited campaign.
- [x] Independent reviews, relevant checks and root records genuine formal-
  positive/real-source fit NOT RUN if no already eligible parent exists.

### Task 4: Whole verification and comparison

- [x] Whole pytest, Ruff, mypy, Rust and affected actual CUDA checks.
- [x] Source-matched wheel and mandatory-only startup, original Config/data proofs.
- [x] Alternating same-real-data/same-HDD old/new read/write comparison and
  concurrent generation/learning measurements, with p95/p99 and resource metrics.
- [x] Record exact evidence/NOT RUN conditions in `docs/verification/`.
- [x] Whole-branch correctness/conformance review; root final acceptance only
  when the complete owner scope passes, with directory default unchanged.
