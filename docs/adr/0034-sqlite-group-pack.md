# 0034: Explicit SQLite group Pack storage

- Status: Accepted (foundation/runtime/Python Source designs; later gates pending)
- Date: 2026-11-10
- Decision owner: root
- Review: bot_architecture_reviewer READY for foundation, 2026-11-10;
  namespace/loans, sealed publication ownership and persisted/IPC contracts resolved
- Review: runtime and Python Source/lineage/PPO designs READY; native/formal/
  final64/Dataset remain separately gated
- Implementation: foundation accepted after both final reviews and123root checks;
  production consumers not yet implemented

## Context

The owner requested one compressed Pack per managed Simulator execution group,
with durable SQLite sealing and original-source preservation across current v2
collection, learning/resume, final validation and Python/Rust Dataset/Tree paths.
Current Source authority includes retained original inode/content evidence;
storage replacement cannot discard those guarantees.

## Decision

Adopt the staged design in [sqlite-group-pack-v1](../design/sqlite-group-pack-v1.md).
Physical `sqlite-group-pack-v1` wraps unchanged Collection5/Journal6/codec2.
Python owns packing, SQLite Writer, catalog, physical custody and authorization;
Rust consumes verified originals without SQLite. New format is explicit opt-in.
Use only Python 3.14 standard libraries for the storage foundation.
The [runtime design](../design/sqlite-group-pack-runtime.md) and
[Source design](../design/sqlite-group-pack-source.md) settle explicit Config,
memory/RAM output, composite publication, proof2/custody, run4/update4 and wire5.
Their accepted Python scope preserves cross-open physical replacement detection
and checks FD/pool retirement before generation publication. Native and final64
outline sections require their own exact dependent-contract review before code.

## Alternatives

Per-artifact database rows retain small-object overhead. Whole-parent archives
prevent bounded group reload/retirement. Default replacement without evidence
would discard a known compatibility path. Preserve the selected group granularity
and keep the existing default through the requested full comparison.

## Consequences

Seal/publication and Source-health publication are separate lifetimes. Writer
COMMIT and catalog states confer no learning authority. Shared tmpfs monitoring
admits possible temporary overshoot rather than promising an OS hard cap; its
impact, failure handling and revisit conditions are explicit in the design.
Power loss may discard unsealed NORMAL transactions. No automatic migration or
deletion occurs. Integration questions must pass their architecture gate before
implementation; accepting the foundation does not accept incomplete full scope.

## Verification

The design lists corruption/failure/ownership checks, complete runtime smoke
coverage, old-data checks and same-HDD comparison requirements. Evidence and
review dispositions will be recorded in a separate verification ledger.
