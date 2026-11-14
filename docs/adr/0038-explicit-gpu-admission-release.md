# 0038: Explicit recovery of GPU admission without certifying termination

- Status: Accepted; implemented with scoped verification.
- Date: 2026-11-15
- Owner: root
- Supersedes: ADR0037's mandatory-restart fallback for an unconfirmed GPU job.
- Review: independent architecture review READY before implementation.

## Context

An evaluation publication exception leaves an admission without a termination
receipt. The normal owned-cleanup correction is already implemented, but cannot
retroactively reconstruct its lost live ownership. Process/GPU inspection finds
no known surviving workload. The original recovery contract consequently asks
the user for another environment restart despite no observed resource shortage.
The user challenges that operational cost, notes ample headroom, and authorizes
continuing after the restart request is withdrawn.

## Decision

Implement the bounded [admission-release contract](../design/ppo-hpo-admission-release.md).
An explicit operator may resolve one exact stale admission for scheduling while
preserving the fact that its termination remains unconfirmed. Store a distinct
immutable release receipt under the same global lock and original ticket. It is
neither a termination/restart receipt nor evidence of healthy training/evaluation.
The old admission, failed data, deadlines and all qualification rules survive.

Keep automatic fail-closed admission, normal actual-owner cleanup and explicit
actual-restart recovery. A live global lock still excludes release. No automatic
PID/ps/VRAM/time inference, process killing, reboot or alternate namespace is
introduced. The operator supplies identity, reason and explicit acceptance of
possibly remaining resource use. Conflicts/corruption still reject; exact replay
and directory durability follow existing receipt rules.

## Alternatives

Keeping restart as the only fallback has excessive operational cost for this
functional work. Treating process absence as a termination proof would overstate
the evidence. A new supervisor or cgroup subsystem would enlarge scope and would
not reconstruct this old job's ownership. A distinct manual admission decision
expresses the user's revised requirement with the least additional machinery.

## Consequences

Recovery no longer requires a restart merely because a receipt is missing.
Possible residual resource contention is an accepted operational limitation;
managed resource budgets are not OS quotas or guarantees of an idle GPU.
Future jobs still pass ordinary admission, storage and cleanup checks. Missing
data or unhealthy runs never gain an objective through this operation.

Old readers reject the new release record; no persisted old record or global
namespace changes meaning. Revisit process-group ownership only if normal cleanup
and explicit recovery prove operationally insufficient. This decision does not
establish strength, performance isolation or formal/final-test qualification.

## Verification

Require focused tests for exact identity/retry, live lock exclusion, malformed
and conflicting receipts, independent unresolved orphans, durable-publication
faults and pointer repair. CLI must require a deliberate recovery mode/reason and
stay optional-dependency free. Training/evaluation verifiers must reject a release
where actual termination is required. Both independent correctness and
implementation-conformance reviews are READY; each independently passes all50new
cases. Owner affected verification passes116CPU cases; root independently passes
100covering cases, full project Ruff, mypy195source and diff checks. Operational
release and the remaining physical protocol are separate acceptance steps.
