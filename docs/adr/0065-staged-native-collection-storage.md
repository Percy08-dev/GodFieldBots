# 0065 Native PPO collections staged on tmpfs

Status: Accepted. Independent design, conformance and correctness review
cleared. [Physical-drive verification](../verification/staged-native-collection-storage.md)
records the validated workload, failures and resume scope.

## Context

The Model 4 256-game storage PoC measured 49.57 s on direct HDD storage and
27.69 s with the original SQLite database and Simulator workspace on tmpfs.
A separate transfer and verification on HDD ext4 completed in about one
second. The measured path from collection through confirmed disk sync took
29.03 s. The HDD ext4 route also reduced the downstream verification time
compared with D-drive access over 9p.

The runtime remains pinned to Simulator
`072dabbca1458c9526897781d2fd64edeeb4f9ca`, Gameplay API v2 and catalog schema
2. Relevant current normative references are Simulator `59e2e009` spec 07
section 21.1 and spec 10 section 7. The collection remains Journal 6,
candidate-columns-v2 and sqlite-group-pack-v1.

## Decision

Native ordinary PPO and HPO6 can opt into a private collection workspace on
tmpfs through the base Experiment storage settings. The original database and
Simulator workspace share the collection lifetime. Existing Journal record
validation, pair commits and encoding still run on the existing writer.
Existing HPO HTTP retention remains diagnostic storage.

At seal, the writer closes the mutable SQLite connection and sidecars and
uses its original immutable database and descriptor as the source. It copies
to a private directory on the admitted HDD ext4 filesystem. It re-reads the
destination's SHA-256, validates SQLite and every Pack, then synchronizes the
data and directory. It installs the files under an exclusively claimed
collection directory on that filesystem, synchronizes the affected directory
ancestry, and publishes the durable catalog entry. Existing Source ownership
binds the actual verified HDD database and descriptor before issuing the
collection to the learner. The live transfer has one writer and is synchronous.

The existing final path, hashes, run and seal IDs, and lineage formats remain
the durable identities. When the option is omitted, historical effective
config identity and direct writing behavior remain compatible. Staging
requires a writable tmpfs and dedicated ext4 devices for the original and
control destinations. The chosen ext4 mount's physical HDD backing is a
deployment setting.

## Failure boundary and alternatives

A Journal ACK transfers validated custody. Existing terminal ACK can precede
the pair's SQLite transaction. The previously direct SQLite WAL may leave
recoverable bytes after an interruption; native PPO and HPO6 do not resume
from a partial pair set or from an uncommitted collection. Stop or power loss
before durable publication can discard the in-flight batch. The supported
training resume boundary is an explicitly paused committed update within
the original boot and deadlines, with intact durable artifacts. Cross-boot
training resume was explicitly deferred as a separate design task.

A failed or uncertain promotion grants no Source or generation authority.
The Writer retires files it still owns where cleanup can be confirmed, and
preserves uncertain durable evidence for the original quarantine path.
Existing final files and catalog rows have exclusive custody. The staged
source is retired only after its owners and confirmed Simulator cleanup are
complete.

Direct HDD writes preserve more intermediate bytes but imposed measurable
per-operation I/O waits. Copying only at the collection boundary preserves
the existing consumed Source contract. An HDD log for intermediate pairs,
parallel transfers and a new persistence format add complexity without
serving the existing resume boundary or the measured transfer bottleneck.
