# 0004: Join owned groups before propagating cancellation

- Status: Accepted
- Date: 2026-10-27
- Decision owner: root
- Review: `bot_architecture_reviewer`, design READY, 2026-10-27
- Implementation: implemented; final conformance/correctness reviews READY.
  [Coverage follow-up](../verification/coverage-followup.md).

## Context

A new event-barrier regression cancels CLI execution with one group inside Act
and another queued at its semaphore. Implicit `gather` propagation can complete
because the queued child cancels while the active child still finalizes. The
CLI then returns/persists its interruption before its server is reaped.
This violates the existing coordinator ownership, independently of game outcome.

Explicit task ownership and cancellation/draining restores the ordinary path.
An edge remains if OS delay or late cancellation exceeds the nominal wall plus
the existing worker/server emergency allowance. Python cannot synchronously
kill an owned coroutine without abandoning its finalization. No new executor,
registry or writer process is justified by this regression.

## Decision

The CLI keeps explicit references to all scheduled group tasks. On cancellation
or any exception it cancels unfinished tasks, prevents queued admission, and
shield-joins all owned tasks before recording final interrupted metadata and
propagating the original exception. Repeated cancellation must not abandon the
join. Preserve per-game known outcomes, local failures and persisted ledgers.
Apply this to every exceptional exit with pending tasks, including TimeoutError
and budget fallback paths; an exception from one child cannot abandon peers.
Close admission inside the failing group before releasing its semaphore and
check the gate immediately after queued acquisition, before entering the
environment. Capture child-exception origin explicitly: even a child TimeoutError
with no pending peer propagates unchanged; coordinator preflight/statistical
budget expiry retains its existing unsuccessful-summary fallback after draining.

Use the existing wall/emergency limits to detect and record unhealthy overruns,
not to drop still-running task handles. If the bounded underlying worker/server
cleanup exceeds nominal wall plus 1.6s (or cancellation arrives after an OS/I/O
overrun), continue joining owned finalizers with explicit failed/interrupted
health. This is not a fresh processing allowance; no new game, policy action or
expensive report starts. It is limited by those finalizers' existing bounded
requests/reap operations, subject to the already accepted uninterruptible OS/FS
debt. A caller needing strict hard exit must use a separate watchdog, which is
outside this task. Revisit on normal responsive-OS overrun, nonlocal filesystem
support, unbounded owned finalizer behavior, or strict deployment requirements.
After detecting an expired drain threshold, record it once and continue a
shielded join; do not reset deadlines or spin on an already-expired timeout.

After draining, write interrupted provenance best-effort and rethrow the original
exception. Known natural outcomes never become invented failures/wins; overall
comparison/run/data eligibility stays unsuccessful. Separate storage failures
must not prevent remaining cleanup. No policy-visible information changes.

## Alternatives

- Return on the drain deadline with pending tasks: violates ownership and can
  leave background work mutating artifacts after the caller returned.
- Give cleanup a fresh ordinary run budget: violates the experiment budget.
- Add a global task registry/process supervisor: unnecessary for this local
  ownership repair; existing tasks and resource owners suffice.

## Verification

Hold cleanup behind an event barrier after cancellation and assert the parent is
unfinished until release; then verify worker/client/server completion and original
exception propagation. Include active/queued tasks, different concurrency limits,
late/repeated cancellation and overrun diagnostics. Queued groups must not create
resources. Check interrupted metadata retains known results and data remains
ineligible. Independent correctness/conformance reviews precede acceptance.

Verification completed: the original early-return, queued-admission and child
TimeoutError probes now pass. Final independent correctness subset: 22 tests;
architecture group-ownership subset: 14 tests. Event barriers cover active/queued
tasks, held cleanup, repeated cancellation, peer errors and expired drain limits.
