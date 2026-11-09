# ADR0028: Concurrent participant readiness in CPU pytest collection

Status: accepted; test-only implementation and scoped acceptance complete
on2026-11-09 after both independent reviews and recorded verification.

## Context and classification

The owner asks whether participant initialization wait can be reduced further
inside the already authorized pytest collection scope. ADR0027's pool remains
owned by one collection: the SAME ReservationLedger requires physical participant
reap before learner updates, parent completion and resume. Cross-collection
services would change that invariant and are excluded from this increment.

The post-pool representative GRU diagnostic has four collections,64games,
128logical participants and16physical processes. Collection wall is23.064881s,
with16.251377s of readiness intervals. Every process's first GRU preparation
averages1.407323s, versus0.036434s for later preparations. A same-game readiness
loop currently awaits viewer1before sending viewer2's initialization packet.
Processes are already created/reserved before that loop.

Root classifies changing handshake scheduling as a concurrency/failure-boundary
change requiring the architecture gate. A disposable healthy-path experiment
may establish performance feasibility; it is not the production/test-support
foundation and does not establish cancellation correctness.

## Requirements and ownership

- Only explicitly opted-in ordinary CPU engineering pytest collection may use
  concurrent readiness. Normal application runtime, CUDA, native/source adapters
  and helpers without the new explicit option retain existing behavior.
- Preserve original worker initialize implementations and clean subprocesses,
  logical/physical reservation ownership, viewer partition, protocol/epochs,
  artifact checks, seed/RNG reset, numerical budgets and every original assertion.
- The cold shared MLP reference keeps fresh per-participant processes. Its
  explicitly opted-in CPU test readiness may run concurrently; freshness is
  physical process isolation, not a requirement to serialize independent boots.
  Normal sequential initialization remains covered by the default helper and
  original worker/protocol tests.
- Preserve all collection/update/parent boundaries: no physical process crosses
  collection or optimizer update, and no uncharged idle process is introduced.
- Start at most the already-owned two participants for the same game. Admission
  and physical worker limit remain four across the existing two concurrent games.
- No action may be selected until every original initialize has succeeded.
  Failed/cancelled initialization cannot permit partial healthy publication or
  leave a pending initialization/task/I/O thread racing episode cleanup.

## Responsibilities, interfaces and data flow

Test support owns a private async readiness helper and an explicit
`parallel_ready=False` argument to `tests.structured_ppo_support.component_http`.
Its local FakeEnvironment overrides `_ready_actors`: default delegates to the
original method; the opt-in branch calls the test-only helper. The existing
test_structured_ppo module wrapper opts ordinary CPU paths in, including its
explicit fresh-process reference. CUDA/source/validation-context paths cannot
enable the option and must reject explicit misuse before worker/server activity.

The private helper takes the existing viewer-to-worker mapping and episode
result. It retains each actual worker and schedules its original initialize
once. Success records each viewer's actual startup_elapsed_ms in the original
result field. Result mapping comes from explicit viewer keys, never completion
order. It changes no child packets or policy-visible inputs. Actual initialized
fast-path calls inside selection remain untouched.

Dependency direction remains tests -> existing Environment/worker APIs. No
application module imports test support; no new production hook, dependency,
Config field, Journal/Dataset/checkpoint format or Source authority is added.
The ADR0027 control protocol and lifecycle receipt stay unchanged.

## Failure and cancellation

The test helper owns every initialization task from creation until completion.
It must drain all tasks before returning or raising; failure cannot race the
original episode finalizer's worker.close calls. Preserve the original concrete
exception types rather than exposing ExceptionGroup from a TaskGroup.

On ordinary initialization failure, retrieve every scheduled task's result and
raise the first failed worker in stable viewer-mapping order. Other independent
initializations stay within their existing worker/execution timeouts. Preserve
successfully obtained startup diagnostics without inventing missing readiness.

Freeze `workers.items()` in its stable mapping order and explicitly retain at
most two actual Tasks immediately after each successful creation. If task
creation raises, close that never-submitted initialize coroutine. Build exactly
one `gather(return_exceptions=True)` from the retained Tasks and shield that
gather from the FIRST await, including normal initialization. Normal failures
wait for all results; record successful viewer startup diagnostics and then
raise the first BaseException in frozen mapping order.

On outer timeout/cancellation or task-creation error, cancel each unfinished
initializer ONCE. Repeatedly shield-await the SAME gather until every task is
done, retrieving every result and then propagating the original outer error.
Repeated caller cancellation cannot cancel that gather, replace the first error,
or request a second child cancellation while its abort cleanup is running. Do
not retry initialization, extend/restart its timeout, suppress failures into
success, or detach cleanup. Original worker abort/close and episode cleanup
retain their first deadlines and real reap.

Existing collection finalization still verifies real reap and SAME empty ledger
before report/manifest success and numerical update. Any readiness failure uses
the existing unhealthy/incomplete paths and preserves natural/truncated/aborted
distinctions. Diagnostics remain outside features/policy input.

## Compatibility and alternatives

The change is test-only and explicit; helper default and installed application
bytes are unchanged. No simulator integration or persisted semantic changes are
needed. ADR0027 remains authoritative for process lifetimes and cache reset.
The new record only supersedes serialized readiness in explicitly opted-in CPU
tests; it does not supersede fresh-reference isolation or ordinary worker rules.

Keeping sequential readiness is the fallback if measured benefits are absent.
Cross-collection warming would require a different owner/resource/update design
and is deferred. Forking a Torch-loaded interpreter is excluded. Reducing memory
diagnostic sampling is a separately measurable local optimization and is not
bundled into this readiness change.

## Verification and open questions

Architecture review precedes implementation. Implementer owns only test-support
and relevant tests; Root owns design/evidence docs and acceptance. Required:

- Meaningful controlled tests prove both original initialize calls are in flight
  before either is released, explicit viewer metadata, no select before both
  successes, unchanged maximum physical reservations and original default route.
- Failure and cancellation tests prove stable concrete exception propagation,
  joining a delayed sibling/abort cleanup, repeated cancellation, timeout and
  task-creation failure with no leaked task or racing worker.close. Keep the
  actual worker/pool lifecycle tests and default fresh/CUDA guards.
- Run the unchanged42-case module with original game/update/resume/assertion
  budgets, existing48pool regressions and relevant protocol/source/default-helper
  tests. Both independent reviewers check correctness and design conformance.
- Compare the same42target cases against the retained241.87s baseline under
  equivalent observer/setup reporting. Include readiness, reset and final cleanup
  costs. Check actual canonical final learned arrays and original wire fields;
  do not equate missing legacy fields or normalize substantive policy content.
- Ruff for tracked application/test/native scope and mypy src must pass. No
  whole-suite speed claim or whole-suite rerun is required for this test-only
  change unless review finds an order/default-route regression concern.

The disposable original-two-case experiment passes with all original assertions,
unchanged source hashes,192readiness records and128resets. Fresh-reference
collection changes37.211199to26.501274s; pooled GRU changes23.064881to21.791269s.
The observer/preload and single-observation limitations prohibit a general
speed claim. Evidence is under `/tmp/godfieldbots-parallel-ready-probe-20261109`
and the prior post-pool diagnostic directory. The healthy-path prototype is not
the accepted failure/cancellation implementation.

Architecture reviewer finds the bounded interfaces/lifetimes sound and requested
the precise ownership/join recipe now pinned above. Root accepts that disposition:
explicit retained Tasks, closed unscheduled coroutine, one shielded gather,
single cancellation per pending child, original concrete error and complete drain
before episode cleanup. No full-suite rerun requirement was identified for this
tests-only branch. Final independent implementation reviews remain required.

At design review, same-target performance and diagnostic-versus-ordinary observer
overhead remained acceptance questions. The final same-target observation below
resolves scoped acceptance, with variance still unmeasured. A future change
requiring a production lifecycle seam returns to Root for a revised design.

## Final acceptance

Both independent correctness and architecture conformance reviews are READY with
no blocking findings. The implementation follows the exact owned-task recipe;
all pre-existing test bodies, fixtures, budgets and assertions remain unchanged.
Only test support, the CPU module wrapper and21new regression cases change.
All157application modules, native/config/dependencies and simulator bytes stay
unchanged; no new production seam was needed.

The same original42cases PASS. Measured wall241.87to231.79s saves10.08s/4.17%
in single observations. Most benefit is the physically fresh shared MLP reference:
its stored collection time falls36.957220to27.387731s. Pooled GRU changes
23.898174to22.920339s; Transformer24.013111to23.669627s. These nested scopes are
not extra costs and do not establish a whole-suite speedup or stable variance.

Related verification passes279cases with6unchanged opt-in CUDA skips, including
all21new ownership/routing/actual-child cases and all48existing pool regressions.
Ruffsrc/tests/native, mypy157 and full4301case collection PASS. Independent pure
concurrency/default-route checks also pass20cases. No whole-suite rerun was needed
for this bounded test-only change; historical whole-run results stay historical.

Root's final actual-array comparison preserves916initial/final weight/optimizer/
RNG arrays exactly across the retained before MLP reference and GRU continuous/
resumed parents. All384nonempty schema5decision records also preserve frame
content, actual candidate bytes, search/RNG, selection and actions under explicitly
documented identity/timing-only comparison projection. Originals are never
rewritten or relabeled. Transformer retains the original exact continuous/resume
assertions and existing full-wire tests; a separate retained-before Transformer
artifact comparison is not claimed.

Root accepts this measured bounded increment. The remaining cross-collection
startup cost remains explicit: changing it requires a separately reviewed owner/
resource/update design. Memory-diagnostic sampling remains unchanged. Exact
commands, frozen source hashes, actual proof scope and limits are in
[verification](../verification/pytest-parallel-participant-readiness.md).
