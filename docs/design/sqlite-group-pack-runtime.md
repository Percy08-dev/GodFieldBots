# Group Pack: memory collection and explicit configuration

Read-policy amendment: [ADR0035](../adr/0035-pack-checksum-read-policy.md) and
[the accepted Pack checksum design](pack-checksum-read-policy.md) supersede
duplicate whole-DB verification in Reader/publication/terminal handoffs. Writer
seal-time checksum generation, memory/queue budgets and actual retirement stay.

Date: 2026-11-10. Owner: root. Status: runtime design accepted after independent
architecture review, including the FD-invalidation/optional-predecessor delta.
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

The current physical v1 Source has a256MiB uncompressed ceiling. Runtime and
StorageSettings reject max_pack above that ceiling before allocation; positive
smaller bounds remain configurable. Standalone pure Pack helpers may accept an
explicit diagnostic bound, but that does not enlarge the SQLite v1 reader
contract. Revisit a larger persisted bound only with version-pinned descriptor/
Source and joint-budget review; do not collect an unreadable declared format.

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

Refactor `StructuredJournal` at two principal protected seams: candidate
publication and final output publication, with narrow allocation/existence
helpers where the current physical-path checks require them. Factor closure-row validation/writes from physical
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

The SAME `CollectionPublication` object owns this composite publication, with
an optional concrete `GroupPackCollectionPublication` member (not an arbitrary
callback or descriptor constructor). Default directory mode is unchanged. Its
packed commit stages report/manifest in the Writer, seals to a nonpublic pending
physical descriptor, confirms Writer/thread retirement, and retains that pending
descriptor/DB identity. It then stages the public physical descriptor and envelope
pointer as separate exclusive owned temporary inodes. Run the retained-policy
guard before staging, after both byte/path verifications, and after successful
descriptor/FD retirement, as the current commit requires. Publish the physical
descriptor first, envelope locator last, sync both directories and reverify all
held identities. The envelope locator is the consumer-visible commit point;
fallible post-link checks remain within the actual owner and invalidate that
owned pointer FIRST on failure/cancellation/late policy mutation, followed by
the owned physical descriptor. Never remove a foreign replacement or edit the
sealed DB. Retain a reachable uncertain cleanup owner and initiating exception.
`committed=True` is set only after all checks/close health; no fallible/reportable
operation follows the owner's successful return. A sealed DB/physical descriptor
without a committed locator is diagnostic, not a callback-finalized collection.
The pending descriptor created by the foundation is retained as nonpublic
diagnostic evidence and excluded from public descriptor namespace expectations
only through this concrete composite owner; Source/pending access is the next
gate. No production caller publishes an envelope pointer using a loose second
write after `seal(published)`.

Task2 may enable only `smoke`, standalone `collect`, and `evaluate` with
random/heuristic or existing MLP/GRU/Transformer inference bindings. Configuration
validation/help may parse/report the new type. Tree3 inference, PPO/formal PPO,
final64, game-dataset/inspect/fit, benchmark and validate-policy entry points
explicitly reject storage-bearing Config before optional model/Source imports,
output or process allocation until their Task3 integration is accepted. No
command may accept the new Config and silently ignore storage. PPO/final64
entry points explicitly reject packed mode before allocation until Task3 has
accepted its Source/lineage/native/pending contracts. This is an implementation
order guard, not the final requested behavior. The complete plan still requires
all paths; no partial opt-in is presented as completion.

## Runtime contract closure

All StorageBudget reserve/resize/release, queue and RAM measurements are
thread-safe under one lock/condition; callbacks never await or call native code
while holding that lock. The coordinator keeps the actual
`concurrent.futures.Future` for each synchronous operation, not merely a
cancelled asyncio wrapper. Cancellation closes admission and records the first
error, then shield-waits/reaps the actual worker future and owned executor.
Snapshot/RAM files, Writer and reservations cannot retire until that real work
has completed. If bounded shutdown cannot confirm it, attach the SAME reachable
cleanup owner to the initiating failure and retain its data/reservations; elapsed
deadline or a cancelled wrapper is not evidence that the thread stopped.

Self-validation admission includes decoded records, EventBank, frames, candidate
arrays and Replay graphs: charge `32*JSON_bytes+65536` per JSON/JSONL before parse,
then `32*column_bytes+4096*count+128*elements+65536` per validated descriptor
before candidate decode, matching current Source's conservative local estimates.
Raw storage account and parse account are jointly charged, with explicit graph
retirement after validation. Max journal/file and max record checks precede
JSONL parsing, and a descriptor's count/elements/size must validate before any
corresponding allocation. Allocation/cancellation failure cannot publish health.

Concrete byte seam:
`read_card_instance_journal(path: Path, ..., artifacts: ArtifactReader | None =
None, artifact_prefix: str | None = None) -> StructuredJournalData`.
`artifacts` must be the immutable `MemoryArtifactSnapshot` (runtime) or later
verified original-reader adapter (Source), not a live mutable sink. Prefix is
exactly one safe episode basename, matches `path.parent.name`, and journal
logical name is `prefix+'/'+path.name`; candidate name is
`prefix+'/'+candidate_basename(actor,decision)`. No arbitrary prefix, absolute
name, cross-episode reference or path traversal is accepted. Journal/column
length/hash checks use exact immutable snapshot bytes and the existing codec2
decoder; read each sidecar once, derive parse plus evidence from that SAME read.
Supplying a byte reader cannot disable journal schema/type/reward/identity checks.

Keep `StructuredJournalData.sidecars` as physical evidence for old readers. Add
`logical_sidecars: tuple[LogicalSidecarEvidence,...]=()` with a frozen dataclass
whose fields are `kind='logical-candidate-bytes-v1'`, `name`, `sha256`, `bytes`,
`frame_hash`, `table_hash`, `action_contract`, `purpose`, `synthetic_capacity`.
Logical reads fill only logical_sidecars; physical reads fill only sidecars.
There is no fake inode/path identity field. File-only Source explicitly rejects
logical_sidecars; later packed Source explicitly requires logical mode and binds
the logical metadata to its own verified DB member evidence. A memory snapshot
or logical-sidecar value never grants original Source permission.

Composite publication transfers the actual pending-descriptor ownership record
from `GroupPackWriter` into `GroupPackCollectionPublication`, not just its path.
That record retains the inode, bytes and handle state created by low-level seal.
Retain its pending predecessor as evidence rather than conditional pathname
deletion. Extend the public physical namespace only with the fixed
`collection.pending.json` predecessor, whose exact canonical bytes must equal
the public descriptor with only `publication='pending'` changed. It must bind
the SAME seal/run/DB/controls, be nonsymlink and retained in the namespace/hash
snapshot. No other sibling is admitted, and adding/removing/replacing even this
predecessor after admission fails recheck. A public Source still cannot open a
pending locator or gain authority from that predecessor. Low-level seal(published)
archives may omit it; composite archives retain it. This explicit rule avoids
pretending stat→unlink is atomic against an arbitrary concurrent foreign writer.
Final64 keeps only its pending descriptor until the SAME Source/parent retirement
reaches publication; Task3 must add its issuing-owner-only access capability.
Composite failure invalidates SAME owned descriptor/pointer inodes through held
or verified recovery FDs and preserves foreign replacements, as the foundation
revocation clarification requires. Temporary run directories have exclusive
cooperative ownership; no hard security sandbox against same-user filesystem
mutation is claimed.

## Review-fix contract closure (2026-11-11)

The client imports only common artifact/admission contracts and a pure common
`json_bound` estimator. Extend `gf_common.artifacts` with `ArtifactReservation`
(close and context-manager methods), `ArtifactAdmission.reserve(size,
wait=False)->ArtifactReservation`, and ArtifactSink's readonly `budget` property
plus `hold_graph(key:str,size:int)->None`. The concrete threaded StorageBudget
and MemoryGroupArtifacts remain in training; neither environment type hints nor
runtime imports depend on them. Keep training's old helper import as a pure
re-export if existing callers need it; introduce no client→training edge.

Known simultaneous fixed bases are common pure constants128MiB Writer and64MiB
Source inspection. The minimum accepted persisted/runtime budget is their sum
plus1MiB minimal control/descriptor/runtime slack (**193MiB**). Reject a smaller
StorageSettings/Config before directories/processes; payload/index/graph/copy
costs still need ordinary admission. Verify an actual tiny diagnostic seal at
the minimum rather than claiming every group fits it. Defaults remain2048MiB.

Queued archival has an actual owned/shielded operation before waiting for the
serialization lock, not only after executor submission. Cancellation preserves
the first cancellation and best-effort archives obtained, cleanup-confirmed
groups within remaining finalization allowance. Deadline/failure gates and
actual prior-Future state are checked before every submit, including after lock
acquisition. Latch timeout/failure and close ordinary admission; no new normal
operations or extra queued Futures run after expiry. Bounded retirement joins
actual prior work instead of queuing arbitrary cleanup behind an unconfirmed
operation. Preserve actual owners if finalization is unavailable.

Raw member/Pack/index aliases and parsed graphs remain charged through actual
use, exception and worker retirement. Discard owned aliases/loop temporaries
before releasing their reservations, or transfer that charge to the actual
Future/error owner until it finishes. After diagnostic archive and confirmed
physical/worker cleanup, finite retirement clears completed INTERNAL traceback
frames and owned graph references while preserving the initiating exception,
its stack/diagnostic facts and unsuccessful validation. Do not clear active or
user-body frames. Only genuinely unconfirmed resources retain cleanup ownership;
do not permanently pin all parsed data merely because validation failed.

New `PackedRunResult(directory,report,locator,envelope_sha256,storage_metrics)`
is a frozen concrete result contract in training (no common→training import).
Old unconfigured APIs still return their exact two-tuples. Packed APIs/CLI
explicitly handle this result; do not mutate the original sealed report to add
post-checkpoint fields. Build descriptor/locator hashes and the final result/
metrics under actual publication ownership, expose only after checked commit,
and perform no reopen/parse/reportable I/O afterward. A healthy result requires
the actual published locator, nonnull envelope SHA and metrics publication_scope
published. A successfully sealed unsuccessful diagnostic result carries its
ACTUAL pending locator, envelope_sha256=None and scope diagnostic-pending;
public pending-reader refusal remains unchanged. Failed sealing/unconfirmed
cleanup still raises with the actual owner. Future
PackedFinalizedCollection adds the same storage_metrics field to its already
specified original-result fields. The result conveys no Source permission.

For an actual separately accessible checkpoint measurement, add OPTIONAL IPC
command `seal-metrics` with existing common request keys plus `publication` and
no payload. Version1 ordinary seal/result behavior stays exact. seal-metrics
response result is exactly `{descriptor,checkpoint_metrics}`; descriptor is the
unchanged physical descriptor, checkpoint_metrics exactly nonnegative finite
`checkpoint_seconds,delete_mode_seconds`. Measure actual checked WAL checkpoint
and DELETE transition independently in the child, without changing transaction,
schema, durability or seal ordering. `finish_wal` may return those two durations;
old callers may ignore them. The runtime's owned-pending seal requests metrics;
ordinary Writer.seal gains optional collect_metrics=False, preserving existing
return/default behavior. Client validates/retains metrics before retirement;
PackedRunResult storage_metrics exposes them plus existing combined seal/wait/
encode/write/RAM/managed peaks and explicit scope. No metrics go into sealed DB
after sealing, new sidecars, policy inputs or learning authority.

Composite expected terminal identities come from held original owners, never
fresh pathname admission after inspection. Recheck DB and predecessor content/
paths after inspection, then compare their captured immutable originals and both
output identities through terminal retirement. Every physical Source original,
including optional predecessor, participates in the final post-sweep identity
check. A late replacement fails locator-first invalidation and preserves foreign
bytes; it cannot become the new baseline.

## Verification

### Closed diagnostic inventory ordering (2026-11-11; design accepted)

`GroupPackCollection.freeze_storage_metrics()->inventory row` admits/freezes exact
pre-checkpoint metric bytes and appends their unique storage-metrics.json row
BEFORE the CLI constructs complete inventory, manifest and report. Report
artifact_bytes includes that declared row. The freeze is idempotent only for
the SAME frozen bytes; reject a name collision/duplicate. `prepare` stores those
EXACT bytes and validates a healthy manifest's metrics row hash/length, never
regenerates or silently rewrites supplied controls. Diagnostic-only preparation
may freeze it without a healthy manifest. Final child checkpoint/result metrics
stay separate; original Collection5 and full logical inventory checks remain
unchanged. Preserve prior immutable runs whose audit failed; do not retrofit them.

### Terminal control-file handoff amendment (2026-11-11; design accepted)

The actual D:9p target advances descriptor ctime at final write-handle close,
while device/inode/mode/size/mtime and SHA256 stay unchanged; the equivalent
ext4 probe does not. The real six-game run therefore correctly returned failure
under the stricter pre-close tuple rule and remains preserved/ineligible.
Do not remove late replacement/content checks or silently accept a fresh inode.

Versioned policy `terminal-control-ro-handoff-v1` applies ONLY to controls staged
by this actual composite publication owner at their terminal handoff. Record the
policy in final/persisted diagnostic metrics; descriptor/DB schema stays unchanged.
Capture expected device/inode/mode/size/mtime and exact bytes/SHA from held
original owners before retiring write handles, never from newly admitted paths.
After all controlled writes/syncs/links finish and those write handles retire,
open fresh readonly nofollow capabilities and REQUIRE those same original
device/inode/mode/size/mtime values and original content hashes. A ctime change
across this single write-close handoff is permitted only with those guarantees.
This does not permit arbitrary metadata changes or a different same-byte inode.
The sealed DB itself is not a staged control and keeps its strict full six-field
original snapshot through this handoff; its ctime is not exempted.

During readonly verification, capture full six-field identity; hash from the SAME
opened inode and require full FD/path identities unchanged before/after the hash.
Close readonly handles once, then run final full path/namespace comparisons
against that verified readonly snapshot before committed=True. Failure/uncertain
close still invalidates locator first and retains actual owner; hash buffers/FDs
are admitted to the SAME account. Preserve every substitution/mutation regression
and add close-window ctime-only/content/inode variants plus actual D: probe.

All ordinary ArtifactSource admission/recheck and persisted proof2 custody keep
strict complete six-field identity and full original hashes. No Dataset/Source
authority or postcommit I/O is added. Final/public facts are prepared from the
verified readonly snapshot while that capability is still owned; terminal RO
retirement and final checks precede exposure. If readonly close also changes
ctime or full metadata will not stabilize, fail rather than retry/relax again.

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
