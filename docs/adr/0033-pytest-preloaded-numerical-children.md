# ADR0033: Fresh CPU numerical children from a pytest-only Torch template

Status: accepted; implemented and verified on 2026-11-09 after independent
design, correctness and final implementation-conformance reviews.

## Objective, evidence and scope

The owner requests continuing the measured Torch initialization reductions.
Numerical initialization in19GameLearningProcess starts costs25.423s in
`test_card_instance_source.py`; six starts cost8.246s in the three real CPU cases
of `test_structured_numeric_worker.py`. Actors and numerical learners use distinct
protocols/lifetimes. This increment covers only those19+6CPU starts, not legacy
wire1, CUDA, public application defaults or every learning test.

Root consulted the lifecycle advisor because a creation hook alone cannot
preserve unknown-creation ownership. The separate numerical template is necessary:
actor publication requires its own inventory empty while a learner is live.
The Source path also requires the exact GameLearningProcess type; no replacement
subclass or persistent model/optimizer service is adopted.

A bounded disposable copy at `dcafeb2` demonstrates feasibility. Five original
CPU numerical/Source cases PASS cold20.60s/fork13.92s (single observation including
warmup/close), with4,230arrays across18saved artifacts exact. All three models
perform two updates and actual new-PID restart/restore; wire4 Source-issued update
and raw-update refusal pass. Three lifecycle faults keep unknown leases, and
8MiB stdin backpressure/2MiB stderr remains responsive with1MiB bounded log and
FD13->13. Original cold cancellation/creation/retry tests PASS. Invalid initial
copy-isolation runs and harness corrections remain recorded and excluded.

The experiment is evidence, not production code. Advisor review requires fixing
partial preparation cleanup, absolute launch bounds, cached-failure wakeups,
no-child retirement, transport completion, fixture scope and loop rebinding.
Production must not copy its temporary cold entry path or origin allowance into
the normal bootstrap guard. Evidence is under
`/tmp/godfieldbots-numeric-preload-feasibility-20261109`.

## Production responsibilities and small lifecycle seams

Keep canonical LearningProcess/StructuredLearningProcess/GameLearningProcess
objects, numerical main/serve/learners, request validation, named optimizer/RNG,
actual reservation registration and shutdown. Application never imports tests.

1. Extract async `LearningProcess._create_process()` from its current spawn call.
   Default keeps exactly `sys.executable -E -m gf_training.ppo_worker`, environment,
   stdin/stdout/stderr pipes and1MiB stream limit. Original start owns the shielded
   creation task, delayed cancellation and startup boundary.
2. Centralize actual adoption in a small protected `_adopt_process(process)`
   helper, including creation failures with a subsequently known PID. Normal
   start and settlement use the same assignment, `_process_created()` and exactly
   one drainer creation. Retain the actual process before notification; accept
   only that same process on repeated adoption. Mark registration notification
   attempted before calling it, so even a callback raising after registration is
   never replayed. Track one drainer task/coroutine owner and close an unsubmitted
   coroutine if task creation fails. Retain process and cleanup ownership on
   registration/drainer failure. Repeated settlement cannot register a lease
   twice or replace a process; never infer no-child from an exception after a
   possible external send.
3. Extract protected async `_settle_creation(deadline)` from the structured close
   path. Its contract is actual process/adoption, authoritative no-child, or
   unresolved failure. The default preserves cold stdlib subprocess semantics,
   startup-boundary joining and retryable incomplete cleanup. Remove the separate
   start-side inference that creation-task completion proves no pending child.
   The preload adapter overrides settlement evidence, not start/request/math.
   Competing cleanup/adoption first joins the existing `_startup_boundary` so it
   cannot race normal start. Authoritative no-child is a distinct settlement
   outcome, never a synonym for creation_task.done(), failure or cancellation.
4. Factor only the current cold Torch/visibility block into private
   `structured_numeric_worker._select_visibility(...)`. Its production default
   retains the original unconditional rejection of already-imported Torch,
   environment selection and verification. Keep the call after original complete
   model/execution/identity validation. Only the forked test bootstrap replaces
   this helper with the strict CPU preloaded admission described below; do not
   add an origin-sentinel escape to normal serve or remove Torch from sys.modules.

Do not copy numerical start/close/request logic. A narrow test cleanup wrapper
joins the retained canonical close and the adapter's extra transport/ticket
retirement, preserving original errors/deadlines. Extra retirement has ONE
retained operation and its first absolute deadline across repeated/concurrent
close; it joins canonical close, transport/drainer completion and ticket
retirement, and blocks admission/loop rebinding until quiescent. Retryable cold
canonical cleanup stays unchanged; lost-status preload cleanup stays terminal.
Canonical known-reap resource
release remains distinct from final adapter retirement; a known-exit lease does
not make unresolved transport/ticket cleanup healthy or permit another preload
launch. Unknown actual status always retains its learner lease/cleanup owner.

## Separate template, closed protocol and async facade

Reuse reviewed Torch-only fork/wait/control/containment machinery where suitable,
but use a separate numeric template instance/inventory and fixed numeric profile.
There is at most ONE pending/live/unretired learner ticket per numeric template.
Actor profile retains its existing two descriptors, viewer rules, capacity4,
control behavior and publication checks. Numeric profile has exactly three
descriptors (stdin/stdout/stderr), its own fixed numerical bootstrap and validated
profile/version; wrong bootstrap/descriptor/profile combinations reject before
fork. No arbitrary module/function path or model data enters template control.
Avoid duplicating the whole broker lifecycle or creating a general plugin system.

The numeric child inherits Torch code only: no model/optimizer/batch/Source/policy
state. Its test bootstrap checks pinned Torch `2.14.1+cpu`/`+cu132`, CPU binding,
empty CUDA visibility, CUDA-uninitialized state, one native/Python thread and
the admitted clean-template parent/session/group origin. Origin bookkeeping is
test launch metadata, not a security or Source credential. Original main and
serve still validate original protocol3/4 packets and Source-authorized wire4
updates. Reject legacy1 and GPU; never copy training or bypass Source guards.

Prepare and retain every FD/file/transport immediately, including all three async
pipes, BEFORE sending launch. Failed or cancelled pre-send setup closes and
confirms local resources, discards its PREPARED ticket and resolves only the
proven no-child lease. After send, completion/EOF/error does not prove no-child.
Broker PID/status registration never assigns the owner process or calls
lease.spawned; canonical adoption/registration is the sole writer.

Facade contract: actual PID, returncode only from real broker waitpid receipts,
async cancellation-safe wait, token-based terminate/kill, stdin write/drain/close/
wait_closed, stdout.readline and stderr.read under the original limits/drainer.
Keep actual cached status readable even after later broker failure. With no
cached actual status, terminal broker failure makes returncode raise the retained
failure rather than return None: canonical request and Source live-owner checks
use None as their live-process predicate. This blocks further request/Source use
without copying those methods. Check known status then terminal failure BEFORE
waiting; wake all current waiters on failure
and fail future waits immediately. Use readiness-driven event-loop control with
one dispatcher, no blocking select/subprocess wait in async operations and no
unowned to_thread launch. Numeric async health/shutdown must not delegate to
the actor client's blocking wait_readable/select. Sync warmup is allowed only
outside the test loop; final template wait is nonblocking after WNOWAIT-confirmed
exit, with the existing signal-before-reap ownership rules.

Carry a bounded absolute execution/start deadline separately from cleanup, e.g.
min(original cleanup bound, first start + explicit test startup allowance).
Use at most one outstanding resolve request; no unbounded50ms retry/outstanding
queue. Expiry retains unresolved ownership and joins terminal containment within
the original cleanup/emergency bounds. A late ACK records actual ownership but
cannot turn an expired start into success. No fresh recovery budget on repeat.

Transport retirement awaits required close notifications and original drainer
completion before healthy close/ticket retirement. Retirement acknowledgement is
retry-safe. Preserve primary failure if cleanup fails. Unknown status remains
terminal/quarantined even if group containment succeeds; never synthesize exit0/
255, release unresolved leases or silently restart/fall back. Keep normal cold
retry semantics separate. Pin the CPython3.14.10 selector/StreamWriter adapter
dependency, including any private FlowControlMixin hooks, and test actual drain/
close behavior; unsupported runtimes retain ordinary cold execution.

## Fixture ownership and scope

Use an explicitly requested module owner for each of the two selected modules,
with per-case activation/restoration of protected hooks. Only the three existing
CPU wire lifecycle cases and19CPU Game Source cases opt in. All other numeric
fault/metadata cases, CUDA and cold-reference tests retain ordinary paths.
Preserve existing test bodies/assertions; fixture dependencies may change.

No warmup on import or a pure rejected route. The auxiliary CPU template may be
prewarmed in explicit supported fixture setup, before the async test loop starts,
with its own bounded allowance; include warmup and final real retirement in
measurements. Separate1GiB sampled template RSS soft allowance applies, alongside
recorded participant/learner usage; it is not a whole-system memory guarantee.

The module owner spans multiple asyncio.run loops. Detach old control readers/
callbacks, retire all prior transports/tasks/tickets and bind a new loop/event
only at a quiescent boundary. Refuse reuse with unresolved ownership. Teardown
always restores hooks/context, joins first-deadline cleanup/containment and reaps
the template. Failed teardown invalidates acceptance. No new default expensive
tests: actual lifecycle/backpressure/fault regressions use
`GFB_RUN_CPU_PRELOAD_SAFETY=1`; lightweight guards remain default.

## Compatibility, verification and acceptance

No checkpoint/journal/numerical protocol or math-recipe change. Source code hashes
and producer/resume identities legitimately change and remain actual. Do not
rewrite old originals or suppress active-resume identity refusal. Compare math
and data under explicit comparison-only projections while validating each side's
own complete identities. Engineering evidence is not qualification or quality.

Architecture review precedes dependent implementation, followed by independent
correctness/conformance reviews. Root owns docs/config/acceptance/local commits;
one implementer owns production/test changes after the gate.

- Reuse existing three-model CPU wire update/new-PID restore tests, Game Source
  update/rejection/guard cases and original cold cancellation/creation/retry/
  environment cases. Confirm exact canonical GameLearningProcess and exactly-once
  registration/drainer and actual status on every path.
- Inject successive FD/file/transport setup failures/cancellation, authoritative
  refusal, never-started close, after-send cancel, delayed/lost ACK, late known PID,
  broker loss after fork/after waitpid, consumed-failure notification then second
  wait/close, failed-broker/no-status canonical update with no further write,
  concurrent/repeated close, known-status-after-broker-loss, retirement
  failure/retry and loop rebinding. Verify held leases/owners where unresolved,
  bounded outstanding control, responsive event loop and actual FD/task/ticket
  retirement; preserve first errors/deadlines and physical/status distinction.
- Exercise real backpressure and stderr cap; include template warmup/close,
  RSS/PSS/thread counts. Compare actual cold/fork arrays, optimizer/RNG/counters,
  row/action/source results. Measure two authorized modules with matched scope,
  not only first fork latency.
- Run affected existing CPU/Source/lifecycle/actor-template regressions, actual
  available CUDA cold smokes, scoped Ruff/mypy, source-matched package and
  optional-free startup, and one final default full pytest after ADR0032/33
  stabilize. Record skip population/new lightweight cost and untested CPU-build/
  platform limits. No numerical quality or formal experiment claim follows.

Design disposition: READY, with no remaining blocking design decisions. Root
adopts canonical partial-failure adoption stages, startup-boundary settlement,
distinct no-child evidence, retained first-deadline extra retirement, async-only
control/close and quiescent loop rebinding. Cached actual status precedes terminal
failure; terminal unknown returncode raises to block canonical request/Source use.
The advisor's production-acceptance blockers are mandatory verification above.
No unreviewed prototype code is promoted by this design decision.

## Final acceptance

Both final reviews are READY after four concrete adapter issues were corrected:
post-sampling deadline, admission during auxiliary close, immediate retained
terminal cleanup after retirement errors, and async health failure poisoning.
Focused regressions8PASS, related60PASS and combined-error1PASS preserve original
errors/deadlines and unknown leases; no new recovery budget is supplied by teardown.
The new numeric test population is17light default cases and42explicit heavy cases.

The original51-case diagnostic population PASS41/10CUDA skips: pytest194.02 ->
107.57s, external194.87 ->108.43s with warmup/retirement included. Card Source
case sum61.629 ->43.216s; three CPU wire cases12.885 ->7.999s. Their19and6outer
initializations fall25.423 ->11.924s and8.246 ->4.016s respectively. These intervals
include model/restore/IPC/artifact work and are not pure import or promised savings.

All22selected CPU case folders compare exactly across94NPZpairs/10,487arrays and
saved optimizer/RNG/counter/presence metadata. Positive Game Source parents both
validate; intentionally mutated negative cases stay diagnostics, not eligible
sources. Each side retains actual producer identities and unchanged declared math.
Original numerical/Source test bodies, strict cold visibility and canonical Game
types remain; actual CUDA cold actor/numeric/Source smokes7PASS.

Final integrated full run PASS4,177/327skips/zero failures across4,504cases, external
942.93s; all358frozen code/config hashes match. All4,396prior statuses remain and
108new cases add66passes/42skips. Related300PASS, scoped Ruff/mypy157,157module
wheel exact correspondence and optional-free normal startup/help PASS. Observed
full wall is83.81s below the prior1,026.74s run, with different populations and one
sample each. No matched whole-suite/variance claim follows.

Root accepts the bounded CPython3.14.10 selector/CPU test adapter; other runtimes
retain cold fallback, and only the+cu132Torch build was measured. Actor/numeric
inventories stay separate, no model/optimizer service state is shared, and no
Source qualification or strength claim follows.
[Commands and retained evidence](../verification/pytest-preloaded-cpu-launch.md).
