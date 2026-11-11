# SQLite group Pack v1

Read-policy amendment: [ADR0035](../adr/0035-pack-checksum-read-policy.md) and
[the accepted Pack checksum design](pack-checksum-read-policy.md) supersede this
document's eager Pack verification and repeated whole-DB read hashes. Remaining
physical/writer/resource/ownership contracts stay; historical evidence stays.

Date: 2026-11-10. Owner: root. Status: foundation design accepted after independent
architecture review; runtime/Source/native/final64 gates pending.

## Intent and scope

Implement the owner's group-Pack plan for the corrected card-instance v2 runtime,
without changing the default directory format, old configuration serialization,
logical Collection5, Journal6, candidate-columns-v2, learning eligibility or
Simulator rules. One managed Simulator process/group is one Pack and one database
group row. The group size is recorded, never inferred from pilot/PPO conventions.
All obtained evidence, including failed/incomplete groups, remains diagnostic data;
interruption never fabricates a transition, outcome or reward.

The source baseline is Bots `83aa67e7707b267217247afbc4c3efcf14204a94` and
Simulator `91d5926a7572db327eaea3751e7c24f428120996`. The current normative
Simulator 07 §21.1 and 10 TestServer invocation were read. Record actual revision,
schema/catalog/rule identities in each original logical collection as today.
The Simulator repository is read-only throughout this work.

## Approaches and dependencies

The owner's selected design is a coordinator-built zstd level-3 group Pack with
one Python SQLite Writer process per collection run. Per-file SQLite rows would
retain high transaction/index overhead; a whole-parent archive would prevent
bounded group retirement. Neither is selected. Use CPython 3.14 stdlib `sqlite3`
and `compression.zstd`; add no third-party or Rust SQLite dependency.

Storage contracts and pure Pack encoding belong to `gf_common`; filesystem,
SQLite, subprocess transport and ArtifactSource implementations belong to a
focused `gf_training` storage module set. CLI composes storage; policy workers
receive exactly their existing own-view protocol3 data. Storage diagnostics,
HTTP logs, Replay, GameSeed and other-player journals do not enter policy input.
Storage modules do not import CLI, model learners or the Simulator environment.

## Closed physical contract (foundation gate)

Physical identifier: `sqlite-group-pack-v1`, Pack identifier `group-pack-v1`.
The uncompressed Pack is eight-byte magic `GFBPACK1`, an unsigned little-endian
64-bit index byte length, canonical UTF-8 JSON index, then member payload bytes.
Index object has exactly `format` and `files`; each file has exactly `kind`,
`name`, `offset`, `length`, `sha256`. Offsets are relative to the payload start.
Names are unique, normalized POSIX relative names with no empty/dot/parent
components, backslashes, NULs, absolute prefixes or control characters. Entries
are sorted by name, contiguous in that order, cover the payload exactly and
have nonnegative integer ranges (bool is not integer). Empty files are allowed.
Kinds are closed `journal`, `replay`, `candidates`, `result`, `http-log`,
`simulator`, `diagnostic`; kind is diagnostic classification, never authority.
One zstd frame contains header, index and bytes. Decode rejects trailing bytes,
concatenated frames, invalid index/JSON/UTF-8, duplicate object keys/names,
range/size/hash mismatch and bounded-expansion violations before publication.
Compressed input and index/entry counts also have bounded admission.

Concrete foundation interfaces (`gf_common.group_pack`): immutable `PackMember`
with `kind: str`, `name: str`, `data: bytes`; `encode_pack(members,
max_uncompressed_bytes=268435456) -> bytes`; `decode_pack(compressed,
max_uncompressed_bytes=268435456) -> tuple[PackMember, ...]`. Compressed admission
limit is uncompressed limit + 1 MiB; index limit is 8 MiB, entry limit 100000.
Decoding rejects compressed input exceeding its bound even if it expands small.

SQLite uses `user_version=1`, foreign keys, a single run identity, `groups`,
`episodes`, `controls` and a seal record. `groups` primary key is `(run_id,
group_id)` and stores the compressed Pack, SHA-256, uncompressed size and exact
group metadata. `episodes` primary key is `(run_id, episode_id)` with group FK,
nullable `game_id` and exact obtained result/status metadata. `controls` stores
original named manifest/report or diagnostic bytes plus hash/length. No rewritten
logical control replaces an original. Exact schema/SQL belongs to the storage
module, is version checked on read, and is covered by schema tests.

Concrete SQL column contract (all primary keys NOT NULL, all raw/hash/size fields
NOT NULL): `metadata(key TEXT PRIMARY KEY, value BLOB)`;
`groups(run_id TEXT, group_id TEXT, pack BLOB, sha256 TEXT, raw_bytes INTEGER,
metadata BLOB, PRIMARY KEY(run_id,group_id))`;
`episodes(run_id TEXT, episode_id TEXT, group_id TEXT, game_id TEXT NULL,
metadata BLOB, PRIMARY KEY(run_id,episode_id), FOREIGN KEY(run_id,group_id)
REFERENCES groups(run_id,group_id))`;
`controls(name TEXT PRIMARY KEY, data BLOB, sha256 TEXT, bytes INTEGER)`.
Metadata holds format/run identity and, at seal, a canonical seal object containing
a random UUID seal ID and ordered logical row digests. Never infer eligibility
from metadata. Group metadata is exact supplied JSON; episode transport has
`episode_id`, `game_id` and an opaque exact JSON `record`. A DB contains exactly
one collection (e.g. one PPO batch), not a whole mutable PPO parent. A new
generation gets a new DB and Writer; later generations never alter old seals.

An external, closed, canonical JSON descriptor identifies the physical format,
schema, run ID, seal identity, relative DB basename, DB byte size/SHA-256 and
logical control hashes. DB hash never resides inside that DB. Descriptor names
and schema distinguish pending physical sealing from public collection use.
The descriptor itself is not a Source or learning eligibility receipt.

Descriptor fields are exactly `format`, `schema_version`, `run_id`, `seal_id`,
`database`, `database_bytes`, `database_sha256`, `controls`, `publication`.
`schema_version=1`; `publication` is `pending` or `published`; `controls` is a
name-sorted array of `{name,bytes,sha256}`. Changing publication creates a new
descriptor, never rewrites the database. Production Source publication remains
gated on the existing healthy logical owner, not this low-level physical API.

`CollectionLocator` explicitly represents legacy directory or packed descriptor.
`ArtifactSource` exposes original logical names, `read(name)`, inventory,
`recheck()` and `close()`, plus a scoped group borrow. Directory sources preserve
legacy behavior; packed sources retain original descriptor/DB open descriptors,
stat identities, digests, logical index and readonly SQLite connection. Open
readonly without `immutable=1` assumptions; reject WAL/SHM/unsealed/mismatched
state, replacement, symlinks, new namespace entries and content mutation.
Read-only SQLite inspection must not mutate a sealed DB. Rechecks hash the SAME
opened files and compare their paths with captured identities, including after
the hash sweep. Releasing/evicting a group with live loans fails. A group can be
evicted after export and loaded again with a new borrow generation; stale loans
cannot access a later generation. Reading a bare Pack grants no Source rights.

Concrete foundation interfaces (`gf_training.group_pack_source`): frozen
`CollectionLocator(kind: str, path: Path)` (`directory` or physical format),
`open_artifact_source(locator, *, deadline, max_owned_bytes) -> ArtifactSource`.
Source has `names() -> tuple[str,...]`, `read(name) -> bytes`, `recheck() -> None`,
`borrow_group(group_id)` context manager, `close() -> None`; packed implementation
also exposes cloned logical group/episode/control index data. Directory mode
uses the same logical byte operations without changing legacy validators.
Public locators reject pending descriptors. The later final64 integration must
add the internal issuing-owner capability before any pending Source is used.

## Writer, resources and sealing (foundation gate)

Writer is a dedicated clean Python child invoked through a fixed module. Use
length-framed binary IPC, closed JSON metadata and raw compressed bytes; never
pickle or base64. Cap frame sizes before allocation, bound each read/write/wait
by the remaining monotonic deadline, and reject truncated/unknown frames. The
coordinator owns a serialized bounded submission queue and retains reservation
until acknowledgement or confirmed Writer reap. COMMIT acknowledgement is
provisional durability, not SEALED. Same key and exact metadata/bytes replay is
idempotent; a changed group/episode/control rejects and rolls back the transaction.
Group and all episode rows commit atomically. Crashed/uncommitted rows are not
invented by recovery. A fresh run never appends to or alters a SEALED database.

Concrete foundation interfaces (`gf_training.group_pack_writer`):
`GroupPackWriter(directory: Path, run_id: str, *, settings: StorageSettings,
deadline: float)` owns `collection.sqlite3`, a Writer subprocess, its IPC and
reservations. `submit(group_id: str, pack: bytes, *, raw_bytes: int,
record: Json, episodes: list[Json]) -> None` returns after provisional COMMIT;
episodes have exactly `episode_id`, `game_id` and `record`.
`store_control(name: str, data: bytes) -> None` is immutable/idempotent.
`seal(*, publication: str = 'published') -> CollectionLocator` verifies and
publishes the selected descriptor, joining the child before returning.
`close()` stops/reaps the child without claiming a seal. Frozen stdlib-only
`StorageSettings` in `gf_common.group_pack` has `db_root`, `ram_root`,
`budget_bytes`, `queue_bytes`, `max_pack_bytes`; do not add it to old
ExperimentConfig. Synchronous submissions serialize under a lock and condition
byte reservations so concurrent group callers backpressure; never use an
unbounded Python/multiprocessing Queue. An async coordinator may use an owned
executor. This first increment has no production collection consumers.

Initial settings: DB root `/mnt/d/godfield`, RAM root
`/dev/shm/godfieldbots`, storage budget 2048 MiB, compressed queue 512 MiB,
uncompressed Pack 256 MiB. A separate explicit Config derived type carries these
fields; absence leaves old dataclass/asdict/hash identities exactly unchanged.
Reject impossible budgets, unknown format and unsupported action/config paths
before allocation. Runtime config adoption is a later reviewed gate.

Admit before constructing raw/copy/IPC/SQLite objects. Account retained raw
members, index/header, compressed output, encoder/decoder working storage, queue,
both sides of IPC, SQLite binding/caches, and observed run-owned RAM files.
Use byte reservations and bounded chunks; queue saturation waits without new
admission, cancellation stops admission. Monitor shared tmpfs free space and
owned usage during live Simulator output, stop on exhaustion/over-budget.
This is a managed budget, not an OS-enforced cap: shared external use and writes
between monitor samples can temporarily exceed it. Report observed peak/overrun
and mark failure. Revisit dedicated size-limited tmpfs/cgroup storage when a
normal smoke exceeds budget, concurrent unrelated users make admission unreliable,
or a strict hard cap is required. Do not claim RSS is constrained by reservation.

Generation DB: `journal_mode=WAL`, `synchronous=NORMAL`,
`wal_autocheckpoint=8192`, `cache_size=-32768`; busy timeout is at most 5000 ms
and at most remaining operation time. Seal ordering is all queued COMMITs → final
seal transaction → checked `wal_checkpoint(TRUNCATE)` (busy=0, all frames handled)
→ checked `journal_mode=DELETE` while the sole Writer connection still owns the
database → SQLite connection close → DB fsync and directory fsync → readonly full integrity,
Pack/member/index/control verification → DB hash → atomic no-overwrite descriptor
publication and directory fsync. No public descriptor survives a detected failed
publication check. Failure leaves preserved unsealed/sealed diagnostic evidence.
The DELETE transition is part of physical sealing, after the final checked WAL
checkpoint and before connection close. It makes subsequent `mode=ro` opens
sidecar-free without SQLite `immutable=1`; readers require DELETE journal mode,
absent WAL/SHM/journal files and an otherwise closed physical namespace. Sealed
databases are never reopened read-write by application code. Test the sidecar-free
read assumption on this project's actual SQLite version rather than relying only
on the PRAGMA result.

Catalog is a separate SQLite database maintained by the Writer on a separate
connection with short `synchronous=FULL` transactions. Lock ordering is actual
parent/Source owner → catalog. Never acquire parent ownership under catalog lock.
Catalog records physical lifecycle/descriptor identity, not learning permission.
Seal-before-catalog crash and catalog-before-publication crash are recoverable by
verifying DB and descriptor, never by trusting catalog state or mutating a seal.
An unsealed run may lose recent NORMAL transactions after power failure.

## Foundation closure details

Seal custody clarification from independent implementation review: the child
exclusively creates and retains the actual DB inode before SQLite admission;
the coordinator retains its own FD to that original DB after init ACK. During
expected writes compare device/inode only; after seal/SQLite retirement capture
immutable full stat/hash. Preserve the exact seal_transaction result and compare
it with final validation. Child and parent both compare their held inode with
the public DB path across close/reap/sync/descriptor publication, and validate
through the SAME held original. Never adopt a fresh pathname DB as the original
after Writer retirement. Each new handle follows one-attempt close ownership:
mark the raw number unavailable BEFORE submitting close, retain uncertainty on
error, never retry that raw number after it may have been reused. Retire all
handles and reservations with primary-error preservation. Catalog first-use
version/schema/update run inside one short BEGIN IMMEDIATE transaction.

Revocation clarification: invalidate a published descriptor through the SAME
opened O_RDWR inode (or a nofollow recovery FD verified against its captured
inode), writing/truncating a fixed invalid marker and syncing it. Retain invalid
diagnostic evidence; do not rely on pathname stat→unlink for atomic conditional
removal. A foreign pathname swap after FD verification cannot redirect the
invalidation write. A detected replacement remains untouched and the run stays
quarantined. Failed invalidation/close retains actual cleanup ownership. Earlier
"no public descriptor survives" means no valid SAME-owned descriptor remains
after confirmed invalidation; the pathname need not disappear, and uncertain
filesystem failure is reported rather than falsely claiming durable revocation.

Add `members(name TEXT PRIMARY KEY, run_id TEXT, group_id TEXT, kind TEXT,
offset INTEGER, length INTEGER, sha256 TEXT, FOREIGN KEY(run_id,group_id)
REFERENCES groups(run_id,group_id))` as the global logical-member lookup, with
all fields NOT NULL. Reject duplicate names across any groups or controls;
each member name maps to exactly one group. Validate member rows against the
actual decoded Pack at insert and full revalidation.

Every metadata value is canonical JSON bytes. Metadata keys are exactly `format`,
`run_id` before seal and those plus `seal` after seal. Seal keys are exactly
`schema_version` (1), `seal_id` (UUID4), `run_id`, `logical_sha256`.
The logical digest hashes canonical JSON of `{format, schema_version:1, run_id,
groups, episodes, members, controls}`. Arrays sort by group_id, episode_id, name,
name respectively. Group rows are `{group_id,sha256,raw_bytes,record}`; episode
rows `{episode_id,group_id,game_id,record}`; member rows
`{name,group_id,kind,offset,length,sha256}`; control rows `{name,bytes,sha256}`.
`record` is decoded exact canonical JSON metadata. The logical digest excludes
the seal itself and raw BLOBs; full validation separately rehashes every BLOB.

`borrow_group` yields a scoped object with `names()` and `read(name)->bytes` for
only that group. Each access checks live Source, group ID and borrow generation.
One decompressed group may be resident; loading another group while borrowed
fails. After all loans retire, explicit eviction or the next group load releases
the cache. Immutable bytes returned by a scoped read are used within the loan;
production export must transfer their reservation before retaining them. Bare
bytes convey no authority. Source-level `read` is a convenience for bounded
controls/diagnostics: detached bytes are conservatively charged until Source
close, without claiming their caller released them earlier. Production bulk
readers must use group loans, not accumulate whole parents via `read`.

IPC frame is `<u64 little-endian JSON length><u64 little-endian payload length>`
then canonical JSON and raw payload. Request common keys are `version:1`,
`request_id` (strictly increasing nonnegative int), `command`. Commands:
`init` adds `directory,run_id,settings,deadline`; `submit` adds
`group_id,raw_bytes,record,episodes`; `control` adds `name`; `seal` adds
`publication`; `close` adds nothing. Only submit/control have payload.
Responses have exactly `version:1,request_id,status,result,error` with status
`ok` or `error`; error is null or `{type,message}` (bounded generic message,
no private payload), result is null except seal's descriptor object. Requests
and responses cap JSON at 1 MiB; pack payload caps at max_pack_bytes+1 MiB,
control payload at min(max_pack_bytes,64 MiB); unknown/extra keys reject.
Validate lengths before receive allocation. EOF/truncation/error poisons the
client; no further writes or successful seal may be reported. A graceful close
request returns OK then exits; coordinator still waits for exit health.

The child performs seal/SQLite close/sync/readonly validation/hash and a catalog
`sealed` transaction, returns the exact descriptor, then exits. Coordinator
requires complete ACK, successful pipe retirement and confirmed exit code zero
before creating descriptor bytes in an exclusive owned temporary inode. Keep
the opened inode through flush/fsync, no-overwrite hard-link publication,
directory fsync and final byte/path verification. Lost ACK, nonzero/unknown exit,
catalog failure, sync failure or full DB revalidation failure publishes nothing.
Late descriptor failure revokes only the SAME recorded inode; foreign replacement
is never removed. Any uncertain revocation/close retains a reachable cleanup
owner on the initiating exception and the run stays failed/quarantined. The
earlier no-public-descriptor guarantee applies to the SAME owned inode; arbitrary
foreign namespace replacement cannot be safely revoked by this owner.

Catalog is `StorageSettings.db_root/catalog.sqlite3`, `user_version=1`, table
`collections(database TEXT PRIMARY KEY NOT NULL, run_id TEXT NOT NULL,
state TEXT NOT NULL, seal_id TEXT NULL, descriptor TEXT NULL,
descriptor_sha256 TEXT NULL)`. Database/descriptor paths are canonical absolute
paths, state is exactly `writing`, `sealed` or `failed`. Initial writing row has
null seal/descriptor fields; sealed row has all three final fields; failure may
retain known identity fields. Catalog `sealed` means physically sealed and
records the proposed descriptor, never claims publication. This permits Writer
retirement before coordinator publication without another catalog write.
Recovery only reports validated published/pending/sealed-without-descriptor or
failed/unsealed status; it never creates learning eligibility or rewrites a DB.
Catalog uses bounded busy timeout and a separate connection closed before child
exit. No parent/Source locks may be acquired while its transaction is held.

Foundation admission uses a conservative 128 MiB per live Writer for its
32 MiB SQLite cache, codec/IPC working storage and metadata slack, plus
`4*raw_bytes + 4*compressed_bytes + 32*metadata_bytes` per admitted submission.
Reserve before enqueue/transfer/decode, keep until ACK or confirmed reap. Queue
capacity independently counts compressed payload and metadata, not object count;
one item larger than either limit rejects rather than waiting forever. Source
admission similarly charges a 64 MiB SQLite/codec base, decoded metadata at 32x
raw size+64 KiB per control/index, and `4*raw_bytes+2*compressed_bytes` before
group loading, plus conservatively retained detached-read bytes. These are
managed estimates, not exact allocator/RSS quotas. Initial max Pack is a format
cap, not a promise that every maximally incompressible Pack fits every configured
joint budget. Runtime integration must combine these reservations with the SAME
memory-artifact/RAM-log account before becoming a production consumer.

## Runtime and reader integration gates

The complete task includes these gates; foundation completion is not full-plan
acceptance. Before each dependent implementation, root records exact interfaces
and receives architecture approval for that increment.

1. Runtime collection: Bots output goes to bounded memory objects; Simulator
   persistence, HTTP logs and stdout/stderr go to a run-owned RAM directory via
   existing launch arguments. Pack only after server stop/reap/log drain. Every
   scheduled group, including early interruption, has its obtained evidence and
   explicit incomplete status. Keep process/Writer/reservation ownership through
   confirmed cleanup. Coordinator builds Collection5 inventory from original
   names/bytes and stores manifest/report controls in the DB. New-format opt-in
   must reach baseline/evaluation, PPO and final validation.
2. Source and resume: adapt Python original-source inspection, parent-chain
   rechecks, numeric correspondence, Dataset3 and Tree fitting to retain SAME
   packed-source physical custody. Keep logical Collection5 hashes distinct
   from physical descriptor/DB/source-proof identities and include the latter
   in new-format generation/resume provenance. Never reinterpret old proofs.
3. Native transport: a Python-owned bounded provider stages verified original
   bytes group by group to existing/new explicit native byte-input seams, with
   generation loans and rechecks, preserving Rust-owned parse/numeric budgets.
   No whole-parent resident export or extracted-file substitution for DB custody.
   Rust gains no SQLite or decompress dependency. Native raw bytes do not confer
   Python Source/parent/fit authority.
4. final64: physical sealing can happen first, but a nonpublic pending owner
   retains the descriptor and original DB until existing Source validation,
   Source close health and actual parent-owner procedures complete. Only then
   publish the public descriptor. Failure invalidates pending publication before
   process/Source cleanup. Catalog cannot bypass this owner.

Open integration questions to resolve against actual code before these gates:
exact replacement for `_SourceOwner` inode inventories; SealedScan chain evidence
for virtual members; original-candidate parser byte seam; native Sources provider
inventory/recheck ABI; generation record variant; final64 pending manifest mapping.
These are not permission to ship extraction as authority or omit runtime paths.

## Verification and acceptance

Foundation tests: byte-exact arbitrary/zero-length members and all real artifact
kinds; 2/6/other-sized groups; corrupt/duplicate/traversal/overlap/oversize packs;
duplicate replay/conflict rollback/unknown game IDs; bounded queue/IPC/cancellation;
Writer kill, open transaction rollback, FULL/disk-full errors; failed checkpoints;
seal/catalog publication crashes; same-size DB replacement/mutation; symlink and
descriptor substitution; outstanding/stale borrow rejection and group reload.

Integration tests: byte-for-byte original artifact reconstruction; Python/Rust
features/candidates/selections/numerics equality; Source authority refusal from
Pack alone; bounded PPO collect/update/pause/resume/final64 and Dataset/Tree
smokes under existing supported conditions. No formal campaign, HPO, final test,
bulk migration, deletion or shortened experiment qualification.

Run project pytest, Ruff, mypy, Rust tests; affected CUDA cases when available;
wheel/source correspondence and mandatory-dependencies-only startup; old Config
goldens and old-data proofs. Report PASS/FAIL/NOT RUN independently.

Comparison uses the same real original data and the same HDD, alternating legacy
and packed writes/reads, plus concurrent collection/learning. Report throughput,
generation/wait/operation p95/p99, checkpoint time, RAM, DB/WAL bytes and
conditions/cache effects. Do not substitute synthetic timings for that acceptance
condition. The default remains directory regardless of preliminary results until
an explicit later adoption decision.

## Direct Writer terminal controls amendment (2026-11-11)

Architecture and both final implementation reviews READY; root integrated
source-matched verification passes249affected/legacy checks, Ruff/mypy171,
all8kernel invariants and four fresh D:/ext4 public/pending probes. Actual D:9p direct
Writer publication reproduced the controlled write-close ctime-only refresh
previously resolved in the runtime composite. Apply the SAME approved
`terminal-control-ro-handoff-v1` only to direct non-retained descriptor
publication (published or pending); retained pending ownership stays unchanged.

Factor the existing exact `_TerminalOriginal` into a lower
`group_pack_terminal.py` helper depending only on resources and original-file
checks; composite publication reimports it without changing behavior. Writer
captures the owned staged descriptor's exact inode/stat/hash, retires its write
FD once, and admits the readonly handoff under the SAME Writer account. Before
reading, require the original stable five fields and hash across that controlled
handoff; then require strict six-field FD/path/hash validation through readonly
retirement/final path checks. Keep DB and all ordinary Source custody strict6.

Preserve first failure, actual uncertain cleanup owners, one-attempt FD state,
locator/descriptor revocation and existing return types. The helper cannot admit
fresh arbitrary originals or silently normalize Source ctime. Record the policy
only in external diagnostics; do not add/rewrite DB controls or original bytes.
Verify direct public/non-retained-pending paths on D:/ext4, corruption/substitution
refusals, exact logical bytes and account/FD retirement. Both independent delta
reviews and fresh root evidence are required before this API/benchmark gate.
