# ADR0027: CPU participant reuse inside pytest component collection

Status: accepted; pytest-only implementation and scoped performance acceptance
complete on2026-11-09 after final independent reviews and recorded verification.

## Context, authorization and scope

The owner explicitly selected reuse inside pytest collection tests. Normal
collection, CUDA and installed workers retain fresh interpreters. No simulator
change, new dependency, user Config field or production pool is authorized.
Measured GRU collection spends71.45of75.70seconds with readiness waits active.
The fixed-input CPU probe shows seven episodes in12.86s with seven processes,
1.77s with one process/fresh models and1.51s with one process/model cache; exact
recorded selections and an A...A reset check pass. This is feasibility evidence,
not a production protocol or full-collection speed claim.

The selected change is a test-only CPU pool owned by one collection invocation.
It intentionally changes the per-game physical process lifetime ONLY for an
explicit private component path; ADR0023's normal clean-interpreter boundary is
unchanged. The ordinary CPU MLP reference remains fresh, as do existing CUDA,
native and other component consumers unless explicitly opting in under tests.

## Responsibilities and dependency direction

- A test support pool owns at most the existing four real participant processes
  on the caller's SAME ReservationLedger. It exposes one logical participant per
  episode while retaining physical owners until actual reap.
- A test-only `StructuredParticipantWorker` subclass retains the original
  metadata/frame/table/selection checks, fresh parent event/search state and
  original initialize/select protocol. It delegates only transport and close to
  an acquired physical owner, never fakes Popen/poll or early lease release.
- Factor the existing worker's process-start block into one protected I/O hook.
  Default arguments/environment/reservation/spawn/cleanup behavior stay exact.
  Tests may override that hook; normal callers and protocol2 remain unchanged.
- A test-only bootstrap/transport handles a bounded, closed pool-control envelope
  with epoch and fixed viewer. It drives the ORIGINAL structured_bootstrap.main
  for each episode through a framed input adapter; normal parsing, generator,
  RNG seeding, frame validation and selection are not copied or bypassed.
- Add one narrowly guarded private `_run_experiment` finalizer hook, available
  only with private structured factory/owned component collection. Invoke it
  after groups drain and BEFORE complete report/manifest publication. Normal
  default is None. The test collector also joins the same close operation on
  exception/cancellation paths; no reset of its first cleanup deadline.

No application module imports a test pool/bootstrap. Test support explicitly
opts CPU engineering collectors into reuse; other helpers default to fresh.
The current completed_execution/reference fixture explicitly remains fresh so
MLP fresh-reference versus pooled-resume equality is still tested.

The test FakeEnvironment wraps the ORIGINAL synchronous `_create_actor` in an
explicit viewer ContextVar, restoring its token in finally. Worker creation
acquires by that context; creation order is never a participant identity.

The protected hook has the existing spawn inputs (ledger, computed residency,
argv and clean environment). The default implementation performs exactly the
old reserve/spawn/error-cleanup block and sets self.process/self.lease. Physical
pool owners use that normal path with only a test-bootstrap argv override.
Logical workers override the hook to attach the borrowed REAL process/lease;
they never call spawned again or overwrite lease._cleanup_owner. Physical owner
is the stable cleanup owner. Constructor errors retain the original startup
cleanup-owner route; logical abort poisons and closes the physical owner.

The private finalizer is an async no-argument callable returning a bounded
diagnostic JSON receipt. It is accepted only for an owned engineering actions2
collection with a private factory, not native/validation-context/public paths.
After groups and episode results are collected, await it before report/manifest
publication; store its result at manifest.pytest_worker_lifecycle. An exception
uses existing incomplete/failure handling, preserving known natural results.
The collector ALSO joins that same pool.close in unconditional finally on every
return/raise/cancel, including minimal-report fallback. Both routes share one
close task and its first deadline; this is not a second cleanup budget.

## Episode control, privacy and model cache

Pool owners are permanently partitioned by viewer1or2, with up to two owners per
viewer for the existing two concurrent games. The child rejects a viewer change
and rejects mismatching frame/delta actors before forwarding data. No physical
process consumes both participants' private views. Pool lifetime cannot cross a
collection, train/validation split, parent run, or optimizer update.

The pool envelope has a distinct test-only version, operation and monotonically
increasing epoch. Begin wraps the unchanged protocol2 initial packet. Select
wraps the unchanged packet. All replies echo epoch; stale/malformed/control
replies fail closed. An end-episode boundary produces EOF to the original main,
discarding its EventBank, previous frame, decision counter and search RNG, then
returns an explicit reset acknowledgement. No artificial worker error is used
as a successful episode boundary.

Original main returning is not itself proof of reset: it catches EOF and worker
errors. End ACK is permitted ONLY after an explicit valid matching end envelope,
clean original-main completion with no emitted worker_error, verified cache/RNG
reset and no unconsumed original frame bytes. Unexpected EOF, truncated framing,
bad epoch or worker_error poisons/exits without ACK. Successful owner handoff
joins the actual I/O thread and confirms no pending request/decision reservation;
receiving a queue result alone does not prove thread retirement.

A child caches at most ONE policy/model. An exact match of factory, model,
artifact digest/path, active inference binding, actions and contract may reuse
weights and allocated cache arrays. Each begin retains ordinary full metadata/
file-hash checks against the pinned artifact. A changed identity clears the old
policy and loads a fresh one. No stale-policy fallback or unbounded snapshot cache.
Only CPU Random/heuristic/MLP/GRU/Transformer components are admitted; CUDA, Tree
and public/real-I/O use remain fresh and are not silently routed into the pool.

At end and next begin, reset policy RNG, zero/reset temporal buffers, cache
position/owner/revision/event_end and timings. Parent logical workers start new
search RNG, EventBank/last-frame and request state. No pending request, I/O thread,
decision reservation or old response may survive owner handoff. Reused private
state must never become policy input. Sampling seed remains the authorized
policy seed; no Simulator GameSeed/control metadata is added to policy packets.

## Failure, resources and publication

Real physical owners reserve before spawn and remain in ledger.workers while
idle. Logical close/reset does not call lease.reaped, fabricate released flags,
or pretend that the process exited. A successful loan cannot overlap another
loan on the same owner. Enforce max four physical workers and current CPU/model
admission; no new uncharged warm pool.

Timeout, cancellation, malformed response, reset/checkpoint failure or pending
I/O poisons that owner and causes real kill/reap before release. Do not put an
uncertain owner back into the pool. Close is shared, shielded and idempotent,
with existing first-error/first-deadline semantics. The collection finalizer
must confirm all physical processes/threads/leases reclaimed before success is
published or numerical update begins. A close failure keeps the parent/collection
ineligible and preserves known natural outcomes without invented transitions.

Record a test-worker lifecycle receipt (mode, physical/logical counts, reset/
reload/close health) as diagnostic metadata before final collection hashes.
It stays outside policy inputs/features and does not grant Source authority.
Current component contract remains engineering-only/qualified=False. If an
existing closed reader needs a compatibility adjustment for this receipt, return
that concrete change to Root for review rather than weakening its validation.

## Verification and acceptance

Architecture review precedes implementation; final correctness/conformance are
independent. Preserve all existing scenarios, game/update budgets, array/RNG/
Adam/schedule/source/retention assertions and opt-in conditions. Keep physical
fresh-worker startup/failure coverage through the cold reference and existing
protocol tests; count logical participants and physical processes explicitly.

Required evidence includes full wire/candidate/search/selection equivalence for
fresh versus pooled MLP/GRU/Transformer over multiple episodes, seeds, A...A
sequences and checkpoint changes. Test viewer/epoch rejection, zeroed histories,
no state/response leakage, bounded real leases, constructor/reset/cancel/timeout/
cleanup faults and prepublication close. Compare actual fresh and pooled
numerical parents under a canonical configuration, not just seed equality.

Run the unchanged42-case target population plus new regressions and relevant
protocol/source/native/cold/CUDA-collection checks on HOST. Measure all pool setup,
reset, revalidation and final cleanup. A whole pytest invocation is required
because protected production I/O seams and collection finalization are touched.
Ruff/mypy and isolated baseline/normal-bootstrap checks must pass. No numerical
quality, real Source qualification or CUDA pooling claim follows.

## Alternatives and open questions

Per-game fresh processes preserve all isolation but repeat the measured imports.
Concurrent initializations alone retain that cost. Cross-collection/live inference
services would broaden lifetime/privacy/update responsibilities and are excluded.
The bounded test-only pool pays reset/identity validation instead of import cost.

Reviewer must confirm that original-main episode adapters avoid duplicating the
worker semantics, that the private finalizer gives correct durable failure order,
and the smallest compatible placement of diagnostic lifecycle receipt. Actual
whole-target speed and memory are measured acceptance questions, not promises.

## Design review disposition

Architecture review is READY after Root resolved explicit viewer acquisition,
stable physical cleanup ownership, prepublication/unconditional finalization and
strict end/reset acknowledgement requirements above. Root accepts one bounded
`manifest.pytest_worker_lifecycle` receipt: inspected Python/native readers permit
this root diagnostic while hashing its bytes; no Journal5 or feature schema is
relaxed. The successful owner transition must join the real I/O thread, and EOF
alone can never establish a healthy reset. Final implementation reviews and
prescribed tests remain required before adoption.

## Final acceptance

Root accepts after both final reviewers are READY. Original42target cases remain
green with unchanged game/update/assertion scope; measured wall falls455.89to
241.87seconds (46.95% in one observation). Ten comparison collections retain
320logical participants but use40real physical owners; all resets/real reaps
and receipts validate. Observed owner-only summed PSS peak1826.52MiB is reported
with sampling/shared-page/coverage limitations, not as a paired memory claim.

Conformance found and Root resolved stale logical-handle interference and primary
reset-error replacement; correctness also found and resolved post-thread-join
logical deadline overrun. These are covered by targeted real-process regressions.

The full4280-case invocation reports4032PASS/247SKIP/1FAIL. The one NEW deadline
test expected rejection too late after the approved runtime fix. Its precise
test-only expectation correction is followed by48pool tests PASS; all runtime
bytes remain frozen. Both reviewers accept the full run plus that changed-area
rerun without repeating unrelated tests. This is not recorded as one fully green
whole invocation. All prior whole cases retain their statuses.

Ruffsrc/tests/native, mypy157, source-matched157module wheel and isolated
mandatory-only normal structured worker startup/cleanup pass. Default normal,
CUDA and native callers remain fresh; no test module is packaged. Exact commands,
failed-run preservation, receipts and limits are in
[verification](../verification/pytest-cpu-participant-pool.md).
