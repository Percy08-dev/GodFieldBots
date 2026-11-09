# ADR0030: Owned Torch-preloaded CPU launch in pytest

Status: accepted; pytest CPU integration and scoped/full verification complete
on2026-11-09 after independent correctness/conformance reviews.

## Context, scope and decision

The owner explicitly requests incorporating the measured CPU preloading path.
The real three-model protocol experiment reduces actor startup0.77-0.78to
0.08-0.09s, with actual fresh PIDs and unchanged two-frame wire/numeric results.
Its multiprocessing facade is not an implementation foundation: partial start
can expose no PID although a child runs, and status-channel EOF becomes synthetic
255. Standard forkserver also inherits arbitrary pytest environment.

Root requested the advisor for this consequential ownership decision. Root adopts
the recommended smaller design: ONE clean-exec, single-threaded, module-only
Torch template directly forks and waits for actors. No multiprocessing broker,
resource tracker, pickle protocol or persistent model service is introduced.
Normal successful collection still requires actual actor wait receipts and real
thread/stream retirement before reservation release/update. Permanent status
loss quarantines the affected parent and retains unresolved reservations even
after physical containment; containment is never a fabricated successful reap.

This changes test launch/failure ownership and the pending-creation guard, so the
architecture gate applies. Scope is explicitly opted-in Linux CPU engineering
pytest collection, initially `test_structured_ppo.py`. No application launch
default, CUDA/native/source adapter, simulator rule, dependency, numerical recipe
or persisted schema changes. Unsupported platforms retain original helper defaults;
unexpected failure after opt-in never silently falls back or restarts the template.

## Responsibilities and dependencies

- Test support owns a module-scoped `CpuTemplate` service and one shared bounded
  start/close operation. The template contains only public code/import state, not
  a model, optimizer, Observation, GameSeed, frame, event bank or policy RNG.
- The template is a clean exec-started direct child in its own session/process
  group. Use `-E -S`, explicit venv dependency directory without processing `.pth`,
  a fixed CPU environment (empty CUDA visibility, thread limits1, fixed locale/
  hash/workspace values) and only a private control FD/public launch metadata.
  Add app/test import paths only in forked actors. No arbitrary pytest environment
  or `__main__` imports are inherited by the template.
- The server is the sole spawner/waiter. Before readiness and every fork verify
  supported Torch, CUDA-uninitialized state and exactly one Python/native thread.
  Beyond its own launch-only helpers, it must contain no imported application
  model/training/test module state. These checks
  constrain the supported path; they are not a universal fork-safety theorem.
- Client/actor process facades are test-only. They expose actual PID, pipes and
  wait status to existing worker methods. A test-owned lifecycle mixin handles
  pending creation before delegating normal known-process cleanup.
- Reuse the existing protected `StructuredParticipantWorker._spawn` override.
  No general production launch-provider framework or global Popen monkeypatch.
  Both fresh test actors and ADR0027 physical pool owners may opt in; logical
  per-episode handles retain the existing stable physical owner and viewer rules.
- `WorkerReservation` adds only `creation_pending=False`, matching the existing
  learner precedent. `spawned` clears it only on actual PID registration;
  `cancel_before_spawn` and `reaped` reject while creation is pending. Normal
  Popen reservations stay false and keep existing behavior.

Dependency direction stays tests→existing application APIs. Application modules
never import the launcher. The resource module remains standard-library-only.

## Concrete client lifecycle interface

The module fixture owns `CpuTemplate.start()`/async `close()`, binding it only to
the explicitly opted-in helper. It initially creates an UNSTARTED owner, avoiding
Torch startup for pure guard selections. The first opted-in collector starts the
template before `_run_experiment` can create servers/games or admit participants.
That warmup is bounded by the existing collection deadline and is included in
collection/module timing; it is never hidden as a cross-run cache. A per-creation context carries the existing
Environment execution and wall/cleanup deadlines plus viewer. The worker-owned
launch record is retained before send, with states PREPARED, SENT_PENDING,
KNOWN_RUNNING, KNOWN_REAPED, CONFIRMED_NO_CHILD or STATUS_LOST. PID is available
only in known states. Actual code is available only after a real wait receipt.

Creating pipes/validating locally can fail before send without a possible child.
Once sending can begin, creation_pending staysTrue until validated real PID
registration or authoritative no-fork confirmation. The test mixin's pending
cleanup resolves that record before delegating known-process cleanup; status
loss raises while retaining the reservation. Normal `_failed_spawn(None)` is
never used to decide this uncertain case.

Launch is a synchronous owned control transaction bounded by the lesser of the
worker timeout and existing absolute execution deadline. Its request also carries
that monotonic deadline so the server refuses admission after expiry. This does
not promise asynchronous interruption of a blocking constructor; after the bound
it fails and joins cleanup rather than detaching/retrying creation.

For known actors, `poll` nonblockingly pumps closed status receipts and otherwise
returnsNone, without asserting exit. Terminate/kill send bounded nonblocking
control requests; the sole waiter signals its own still-unreaped child by token,
preventing PID-reuse races. `wait(timeout)` uses one absolute bound. No control
background thread or unbounded per-poll wait is needed. Actual status is cached
only after validated waitpid receipt and retained independently of later template
health. Protocol responses route by known sequence/token; unmatched or malformed
messages poison the service, never become a process result.

Template close is ONE retained operation, shielding its first error/deadline under
repeated cancellation. Normal auxiliary shutdown has at most2s, further capped by
the caller's existing wall/cleanup deadline. If that deadline is already exhausted,
only the existing bounded0.5s reap+0.1s retirement emergency allowance may be used,
with unsuccessful health. Module finalization does not grant a new cleanup budget
after an earlier failed close. Existing actor cleanup continues to obey its own
first deadline; auxiliary containment cannot turn an actor overrun into success.

The collector's existing private prepublication finalizer joins pool closure and
checks template health. Its unconditional finally joins terminal service cleanup
if poisoned before returning/raising, including minimal-report paths. Module
finalization always closes the template and verifies no outstanding healthy actor
records/descriptors. A permanent service failure prevents reuse in that scope;
fault tests own separate instances and cannot poison the normal module fixture.

## Control and policy data flow

Use a private Unix socketpair with bounded packet/descriptor transfer and a closed
versioned JSON control schema. Tokens/sequence numbers identify launch/status/
signal/retirement operations. Only fixed bootstrap enums, viewer identity and
actual OS descriptors cross control. Model/options/frames use the original direct
unbuffered actor pipes and never enter template state. Close received descriptors
on every validation/error path and unrelated descriptors in each child, especially
control sockets and sibling actor pipes.

The original structured bootstrap and original pooled bootstrap remain the policy
implementations. A minimal first-packet replay adapter rejects non-CPU bindings
and unsupported factories/models before the original factory can run, then passes
the unchanged packet to the original parser. Original pooled epoch/viewer/CPU
validation continues on every episode. No generator, selection, engine or policy
semantics are copied. No CUDA initialization occurs in the template or forked
path. Existing normal cold-exec/CUDA/native routes stay independently testable.

The actor-side admission stream must check factory/model/CPU binding on EVERY
pooled `begin` before forwarding it to the original pooled bootstrap. Checking
only its first packet is insufficient because the original pool imports the
descriptor anew at later epochs. Ordinary single-episode structured mode checks
its one initial packet. The fixed admitted factory is the ordinary structured
baseline/neural factory; this restriction is private to preload mode. Existing
cold protocol/fault factories retain their normal unrestricted validation path.

At most four pending-or-live participants exist on the caller's SAME ledger.
Reserve and retain the launch owner/token BEFORE any send that could cause fork.
Set creation_pending before external creation; a send error alone never proves
no child. Server registers token→actual PID before ACK, retaining real waitpid
results until client retirement. Block handled termination signals across fork/
registration; do not claim atomicity against SIGKILL. Client status must match its
session/sequence/token; EOF/malformed/lost status is a failure, never exit255/0.

The service inventory is also bounded to four pending/live/unretired actor
tickets, not merely four live PIDs. Exact wait receipts remain until client
retirement after original stream/thread/lease cleanup; acknowledged statuses may
then be dropped. A full inventory refuses new creation before fork. No unbounded
completed-PID/status cache or alternate ledger bypass is permitted.

## Deadlines, failures and containment

Operations use absolute bounds from the current Environment execution/work/wall
deadlines plus the original worker timeout. Template warmup is bounded in the first
eligible collector before participant admission. Constructor/start operations are owned
synchronously and bounded; cancellation cannot detach an external creation.
Pending cleanup and service close each join ONE retained operation with its first
absolute deadline under repeated cancellation, preserving the original error.
No new admission/computation after deadline, blind creation retry or automatic
template restart is permitted in the affected parent.

Test startup errors retain `_structured_startup_owner` on the same worker/ledger.
Do not pass uncertain creation through `_failed_spawn(None, ...)`, or delegate
pending cleanup to the base path that treats process=None as pre-spawn. A healthy
template may resolve a token and provide actual PID/status or authoritative
no-fork confirmation. Only the latter permits clearing pending/cancelling a
pre-spawn reservation; known actors register/reap normally after I/O retirement.

On control loss, poison the service/parent and close admission. If exact actor
statuses become unrecoverable, retain unresolved pending/live reservations and
cached failure; begin_update/publication/reuse remain forbidden. Exact receipts
already obtained may still retire their real actors. No fabricated poll result
or exit code resolves the remaining actors.

Emergency containment owns the template's exclusive wait operation: do not reap
its leader PID while group signalling remains possible. Stop the sole spawner
and confirm stopped/dead with bounded `waitid(..., WNOWAIT)`. SIGKILL the owned
group while that PID remains held, verify actor members disappear, then finally
reap the direct template and verify group disappearance. Never signal a recycled
group after reaping its leader. Surviving processes, zombies, inaccessible state
or expiry mean cleanup-unconfirmed, not healthy cleanup. Retain cleanup ownership
and reservations; quarantine the whole parent. This is conservative failure, not
recoverable actor-status supervision. Unresponsive OS/external orphan reaping
cannot provide a hard real-time guarantee, consistent with existing cleanup debt.

Every template liveness observation preserves this exclusive wait ownership,
including routine client checks: use WNOWAIT and never an incidental template
Popen.poll()/wait() before a path may still signal its group. Actor poll/status
and the template's sole final wait are distinct operations.

The decisive HOST experiment passed all six barriers in4.574s: before fork,
after fork/before registration, after registration/before ACK, after wait/before
status, lost ACK with repeated cancellation, and actual known exit23. Stop/death
was observed without reaping before group kill; unknown actors were discovered
only by a separate post-cleanup audit, never used for signalling/release. All
actor members disappeared before leader reap. Lost status stayed distinct from
actual status, and first errors/deadlines survived. Source bytes stayed unchanged.
Evidence: `/tmp/godfieldbots-template-containment-20261109` (disposable, not source).

## Test scope, budget and independent reference

Add explicit `preload_cpu=False` to private component support. Only the CPU PPO
module wrapper opts in automatically when its scoped fixture is active. Its
module-scoped autouse fixture binds an unstarted owner through a private context,
restoring the token in finally and joining close at teardown. Constructing/binding
the owner does not import Torch or start a process. The wrapper enables preload
only for ordinary CPU conditions AND an active supported owner; importing that
wrapper from another module without its fixture leaves preloadFalse. Explicit
preloadTrue without a supported bound owner is an early error, not a fallback.
The collector alone starts that owner before `_run_experiment`; all pure guards,
CUDA/default helper calls and rejected source paths therefore cannot warm it. CUDA,
source_ops/validation_context, native, unrecognized factories/models and ordinary
helper callers cannot enter this path. The existing per-collection pool remains
at most4physical actors and closes before updates; template alone may persist.
Collection finalization also checks template health before success publication.
Failure paths join template cleanup without extending first deadlines.

The module owner allows one auxiliary template with an explicit test-only1GiB
RSS soft ceiling, checked at ready/pre-fork/close; fail closed if observation is
unavailable or the ceiling is exceeded. This is separate from actor/CUDA and
managed tensor reservations, not a claim those already charge OS import memory.
Record template and actor RSS/PSS, process counts and setup/teardown. Do not add
unbounded model/code caches or spawn a template per actor. Revisit the allowance
if supported builds/preloads change or its measured margin disappears.

The shared MLP reference moves explicitly from cold-exec to FRESH-FORKED actors:
models/episode/search/RNG state remain new, but imported Torch code is inherited.
Keep all original games/updates/config/numerical assertions; don't call this cold
interpreter coverage. Original three-model normal Popen protocol tests, normal
helper routing and fresh-versus-pooled wire/reset tests retain independent cold
coverage. Compare new full parents against the retained actual cold reference
and all before G0/G2arrays/wire under original budgets. Add fresh-fork versus
normal-cold multi-episode/seed/checkpoint equivalence before accepting the move.

All launch receipts are engineering diagnostics outside features/teacher inputs,
Journal/checkpoint schemas and Source authority. If a receipt extends the existing
root pytest lifecycle field, keep it bounded and validate original hashed content;
no original source is relabeled. A failed module/service teardown fails acceptance
even if earlier test artifacts appear complete. Component sources stay unqualified.

## Verification and acceptance

Root owns this design/docs/config/staging; one implementer owns scoped code/tests.
Architecture design review precedes dependent implementation, then correctness
and final conformance reviews independently check stable code.

Required cases include clean environment/no-CUDA/thread/purity/FD attestations;
all three models with original two-frame/full-network parity and strict normal-
cold/fresh-fork/pool A-B-A/checkpoint comparisons;4concurrent actual reservations;
early child death/no-fork/partial creation; bad/truncated/stale control and leaked
descriptor checks; lost ACK and exact known wait receipts; template SIGKILL at
all critical barriers; repeated cancellation/first deadlines; held uncertain
leases blocking begin_update and success publication; module close and no service
contamination of default/CUDA/source/native helpers. No fake process status.

Measure the original42CPU PPO population against195.84s INCLUDING warmup and
final template retirement; preserve actual source hashes and G0/G2weights/Adam/
RNG/trajectory/content. Run existing pool/parallel/protocol/source regressions,
relevant actualCUDA checks, whole default pytest, Ruffsrc/tests/native, mypy src,
source-matched wheel and optional-free normal worker/help checks. Record actual
unsupported-platform/CPU-only-wheel/compiled-kernel limits; no strength/qualified
Source claim. Recheck concrete fixes, not unrelated commands repeatedly.

The design-stage questions about the control/cleanup API, collector joining,
fixture activation, memory allowance and reference-coverage replacement are
resolved by the concrete sections above. Architecture design review is READY
with no blocking findings. Root accepts its disposition: every pooled begin is
guarded, tickets/statuses are bounded and owned, template waits stay exclusive,
and unresolved status never authorizes release/update. Final conformance,
correctness, fault coverage and numerical/performance acceptance remain required.
The healthy multiprocessing facade is not copied or promoted as the foundation.

## Final acceptance and cost

Root accepts after both final independent reviews report READY. All reviewed
startup/publication/deadline/error/FD/purity/eligibility findings have explicit
regressions. A final real wait-then-error fix records completed template reap in
finally from Popen's stored actual status, so neither close path can later signal
a recycled PID. Source remains frozen through acceptance; no fake status or
uncertain reservation release is introduced.

Same42PPO cases PASS195.84→161.16s (17.7083%, one observation), INCLUDING first
collector warmup and module close. All2112actual G0/G2arrays,768decision projections
and18inference manifests agree with the retained cold-original parents across
all3models. Policy identities remain actual and unchanged. The13normal collections
share1template; each returns zero unretired tickets. Template sampled peak RSS
528.520MiB stays below its1GiBsoft allowance; no whole-memory quota claim follows.

Related333PASS/6SKIP, actualCUDA6PASS, Ruff/mypy157, source-matched157module wheel
and optional-free normal bootstrap/help checks pass. Whole4359cases PASS4112/
247skips/4warnings/zero failures, external1205.43s. It is27.47s SLOWER than the
previous4305case1177.96s run:54new safety tests cost59.768case-seconds while common
cases save32.187s. The matched PPO gain is accepted with that explicit default-
suite coverage cost, not disguised as a whole-suite speedup.

The permanent status-loss distinction remains a deliberate boundary: contained
process disappearance never fabricates wait status or permits updating that
parent. Incomplete OS/orphan cleanup retains ownership/reservations and fails.
The trusted factory/group/no-descendant and pinned Linux/Torch assumptions must
be revisited for another backend, process-spawning policy, dependency upgrade or
unsupported platform. No formal Source qualification/strength claim follows.
Exact commands, retained failed iterations, review dispositions and limitations:
[verification](../verification/pytest-preloaded-cpu-launch.md).

## Follow-up: explicit expensive safety checks (2026-11-09)

The owner subsequently requested removing the expensive new safety checks from
automatic default execution. This supersedes the initial default-suite coverage
cost decision above; the launcher design and its completed acceptance evidence
remain unchanged.

The existing shared `requires_preload` skip condition now also requires the exact
environment value `GFB_RUN_CPU_PRELOAD_SAFETY=1`, following the existing CUDA
test convention. It covers 38 real-process/lifecycle and numerical/training
checks in `test_cpu_preload.py` and `test_cpu_preload_review.py`. The 16 inexpensive
checks remain enabled. Explicit runs still require supported Linux and Torch;
there is no fallback that claims unsupported coverage passed.

This local selection change introduces no new dependency, test plugin, execution
hook or runtime behavior. All tests, assertions, budgets and CPU PPO preloaded
launch remain intact. The prior 54-case cost was 59.768 JUnit case-seconds; only
about 0.07 seconds belonged to the 16 default checks. This is an earlier single-run
aggregate, not a new measured whole-suite speedup.

The tradeoff is deferred detection of launcher lifecycle or model-state
regressions during ordinary pytest runs. Run the documented explicit suite after
changes to startup, protocol, cleanup, model initialization, RNG or Torch, and
before relying on a new launch-equivalence acceptance claim. No automatic
change-triggered runner or CI scheduling is added by this change.

## Subsequent Tree validation extension

[ADR0031](0031-tree-pytest-preloaded-cpu-launch.md) supersedes this record's
validation-context exclusion only for the reviewed private CPU Tree Dataset
preparation path. It admits the exact live final-validation context/projected
LoadedConfig/plan/SAME ledger with fresh forked actors and prepublication health,
retirement and original-parent rechecks. The three ordinary cold preflight actors
remain. All other template, pooling, default-route, failure and Source boundaries
in this record continue to apply.
