# Group Pack: memory collection and explicit configuration

Date: 2026-11-10. Owner: root. Status: proposed, dependent runtime gate.
Builds on ADR0034 and [foundation](sqlite-group-pack-v1.md). Logical output
remains Collection5/Journal6/codec2. This document does not authorize changing
Source proofs or native inputs; those have a separate gate.

## Configuration and locations

Add frozen `GroupPackStorageConfig` with `format='sqlite-group-pack-v1'`,
`db_root='/mnt/d/godfield'`, `ram_root='/dev/shm/godfieldbots'`,
`budget_mib=2048`, `queue_mib=512`, `max_pack_mib=256`. Add frozen
`GroupPackExperimentConfig(ExperimentConfig)` carrying `storage` only when the
input explicitly contains `[storage]`. Neither old ExperimentConfig nor old run
types gain fields/default keys. Existing TOML/asdict/hash goldens remain exact.
Only the corrected actions3/Simulator91d contract can opt in; legacy/engineering,
external URL, unknown storage fields/format, bool sizes, relative default roots,
nonpositive/impossible budgets reject before filesystem or process allocation.
Explicit relative configured paths resolve at their originating file as existing
config paths do. No CLI override/default silently selects this storage.

`run.output_dir` retains its existing meaning for parent checkpoints and the
logical collection envelope. Database directories are independently allocated
at `storage.db_root/<run_id>/` with one `collection.sqlite3` and descriptor;
run IDs are existing random UUID4 collection IDs. Simulator outputs use a fresh
run-owned `storage.ram_root/<run_id>/`, mode0700, exclusive creation, nonsymlink
ancestors. Keep physical data and original relative artifact names separate.

An ordinary new-format logical collection envelope contains only an immutable
`collection.locator.json`: closed `{format:'sqlite-group-pack-locator-v1',
schema_version:1, descriptor:<absolute descriptor path>, descriptor_sha256:<hash>}`.
The physical descriptor remains with its DB and uses its basename. This extra
locator lets PPO parents retain the existing `collections/batch-N` location
while storing the DB at the configured DB root. Source must retain pointer,
descriptor and DB originals; the pointer alone supplies no Source authority.
Pending final64 pointer publication is specified in the Source gate, not here.

## Responsibilities and concrete runtime seams

`gf_common.artifacts` owns minimal `ArtifactReader`/`ArtifactSink` protocols, not
filesystem implementation. Reader exposes `read(name)->bytes`, `names()` and
`contains(name)`. Sink supports exclusive `write(name,data)`, incremental
`append(name,data)`, `rename(old,new)`, `discard(name)` only for owned temporary
members, and immutable `snapshot()` for one completed group. Names use the Pack
normalization rule. It is not a Source issuer or legal-action service.

`gf_training.group_pack_memory` implements one `MemoryGroupArtifacts` per group,
owned by `GroupPackCollection` coordinator (`gf_training.group_pack_collection`).
Store chunks, not repeated whole-journal concatenations. Reserve UTF-8 bytes,
container/chunk/index overhead and temporary join/encode copies before allocation.
Writes fail explicitly on duplicate final names/invalid names/budget/deadline.
Snapshot cannot race later writes. No journal/candidate/Replay/result is written
to disk or tmpfs in packed mode. Simulator's existing persisted/log files remain
ordinary RAM files and join the Pack after confirmed server stop/log drain.

Do not monkeypatch Path/open/os globally or masquerade a virtual artifact as a
physical inode. Add narrow protected output hooks to `SimulatorEnvironment`:
episode-directory creation, JSON artifact save, and server-directory selection.
Their default implementations execute exactly today's file operations.
`CardInstanceSimulatorEnvironment` accepts optional owned `artifact_sink` and
`server_output_root`; corrected clones carry those SAME values for their one
group. Packed overrides create a logical episode prefix and write `_save`'s
same canonical JSON/newline bytes into the sink. Server args/lifecycle are the
existing managed launch with `--data-dir` and `--http-log-dir` below the RAM
server root; stdout/stderr drain there. Do not change Simulator source/flags.
No additional private management data reaches participant workers.

Refactor `StructuredJournal` only at two protected hooks: candidate publication
and final output publication. Factor closure-row validation/writes from physical
fsync/link validation, preserving ordinary behavior and tests. Refactor
`CardInstanceJournal` output allocation into an overridable protected hook so
`MemoryCardInstanceJournal` can supply a bounded text/chunk stream and original
logical paths without fileno/fstat. Reuse all existing header/context/selection,
ack/transition/lifecycle/end validation. Memory candidate publication encodes
`encode_candidate_columns_v2` with the SAME original basename/descriptor fields,
stores bytes exclusively before the referencing JSONL decision, then returns
the unchanged descriptor. Memory final publication atomically renames its
partial member only after successful end-row write; close-incomplete preserves
obtained partial bytes. Never mark an unobserved interrupted game natural.

For runtime self-validation, add an optional explicitly typed original-byte
reader parameter to `read_card_instance_journal` and its private shared reader.
Use it only for actions3. The reader parses supplied Journal6 bytes/records and
validates candidate descriptors against exact bytes using the existing codec2
decoder; sidecar metadata has a distinctly tagged logical identity, never fake
inode numbers. The unchanged file mode keeps existing physical sidecar checks.
Packed Source will reuse this seam after its gate; a byte reader alone confers
no original-source/learner authority. Candidate bytes/order/selected action,
search RNG, rewards and Replay reconciliation must match file parsing.

## Group lifecycle and memory ownership

`GroupPackCollection` owns one Writer, one joint storage byte account, RAM root,
memory-group owners, queued operations and the RAM monitor. The event loop
controls admission and cancellation; a single bounded owned worker thread
performs synchronous Pack/Writer operations without blocking participant I/O.
Shutdown joins this worker and confirms Writer reap before any reservation or
run ownership release. Exceptions carry the SAME cleanup owner if that proof
cannot be obtained. No daemon thread/process owns unpublished data.

Each scheduled group gets its own store and server RAM prefix. Run its existing
games, stop/reap/drain, validate obtained healthy journal/Replay pairs, snapshot
memory and read ALL regular Simulator files with inode/hash checks, including
temporary/unknown diagnostic artifacts. Use original relative names, including
`server-<first episode>/...`. Reject symlinks/special files. Encode exactly one
Pack, submit group+episode rows in one transaction, retain original inventory
metadata, then retire that group's memory and RAM files only after its COMMIT
ACK and confirmed server/thread cleanup. These new run-owned temporaries can be
removed after verified archival; never delete existing user data.

Missing unstarted games use nullable game_id and an explicit incomplete index
record from existing group results; do not synthesize missing Journal6/results.
If collection is cancelled or group health is bad, archive all obtained evidence
and failure status under the remaining finalization allowance when possible.
If time/storage is exhausted, retain actual RAM/unsealed DB diagnostics and
report archival failure; do not claim every scheduled byte was persisted.

The joint account includes foundation base/reservations, memory sink chunks and
joins, raw/compressed Pack/index/codec allocations, IPC copies and all live
Simulator RAM files. Reserve a maximum compressed-size bound before compression,
then reduce to actual size. Foundation Writer may borrow the SAME account,
without double counting a transferred reservation. Root-owned API is
`StorageBudget(settings,deadline)`, `reserve(bytes)->StorageReservation`,
reservation `resize(bytes)`, `close()`, and measured `ram_bytes`, `peak_bytes`;
subsystems never each assume they independently own the full 2048MiB allowance.

Monitor owned RAM and shared `statvfs` free space every100ms while groups live,
and at admission, snapshot and transfer boundaries. Before admission require
working headroom for admitted reservations. A detected overrun/exhaustion
latches a storage failure, closes admission and cancels active groups through
their existing shielded cleanup, retaining owners until reap/drain. Temporary
overshoot/external shared use remains acknowledged managed-budget debt, not an
OS quota. Queue saturation backpressures group admission without unbounded
completed-Pack accumulation. Record peak/wait/write/checkpoint/RAM metrics as
diagnostics outside policy inputs.

## Collection control publication

Corrected CLI composes the new collection coordinator only for the derived
configuration. It validates each completed group's raw artifacts as today and
builds the exact logical file inventory from accumulated per-group hash/length
metadata. At collection end use the existing metadata/report producers and
semantics. Original report/manifest bytes go through `store_control`, then seal.
Healthy ordinary collection publishes its physical descriptor and envelope
locator only after existing policy-artifact final checks and collection health.
Incomplete runs store failure/incomplete controls and sealed diagnostic evidence;
their logical status is unsuccessful and public training Source rejects them.
The outer descriptor's physical published state is never a healthy-source flag.

Task2 may enable baseline/evaluation and neural inference only. PPO/final64
entry points explicitly reject packed mode before allocation until Task3 has
accepted its Source/lineage/native/pending contracts. This is an implementation
order guard, not the final requested behavior. The complete plan still requires
all paths; no partial opt-in is presented as completion.

## Verification

- Exact old Config/effective hashes and physical journal/codec publication tests.
- New derived Config defaults, path origins, unsupported/invalid early rejection.
- Same recorded frames/decisions into file and memory journal: original JSONL,
  columns, end/partial bytes equal, same parse/features/candidates/selection.
- Managed group sizes2/6/other; all original Simulator files included after reap.
- FakeHTTP private fixtures and real corrected bounded baseline/inference smoke.
- Mid-decision cancellation, startup/reap/drain failures, RAM monitor overrun,
  queue saturation and Writer failure retain diagnostic distinctions/owners.
- Group memory retires after archival; old storage default/path remains exact.
- Both independent reviews and final conformance before accepting this increment.
