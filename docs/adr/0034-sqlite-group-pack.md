# 0034: Explicit SQLite group Pack storage

- Status: Accepted (all dependent designs; implementation milestones below)
- Date: 2026-11-10
- Decision owner: root
- Review: bot_architecture_reviewer READY for foundation, 2026-11-10;
  namespace/loans, sealed publication ownership and persisted/IPC contracts resolved
- Review: runtime/Python Source and separate native/formal/final64/Dataset
  designs READY; each implementation still requires independent final reviews
- Implementation: foundation and standalone memory/RAM runtime accepted after
  both final reviews,616affected/legacy checks and18real saved-byte-audited games;
  Python Source/run4/update4/wire5 accepted after both reviews,610affected checks,
  50postreview checks and64real natural games/4updates with fresh parent audits;
  native/final64/Dataset/Tree implementation and full comparison remain pending

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
contracts have passed their separate design gates; their implementation and
source-matched conformance/acceptance remain separate milestones.

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

The packed Source's prepare boundary ends per-batch Source finalization, not
the existing separate postcommit retention transaction. Retention owns a fresh
parent/control scan before pruning and can fail the overall parent while
retaining actual committed update history. Both reviewers agreed this is the
design's allowed future-parent validation; it cannot reuse retired mutable
controls or leave its new original owner unretired. Source callbacks/finally
after preparation remain in-memory only.

The runtime's terminal-control readonly handoff also applies to direct Writer
published/non-retained-pending descriptor completion. An exact lower helper is
shared with composite publication; retained pending behavior is unchanged.
Both reviews and fresh source-matched249checks/fourD:/ext4 probes pass. It
retains SAME-account/FD custody and strict DB/Source identity; no new DB control
or original bytes are introduced. External diagnostics record the policy.

The design lists corruption/failure/ownership checks, complete runtime smoke
coverage, old-data checks and same-HDD comparison requirements. Evidence and
review dispositions will be recorded in a separate verification ledger.
