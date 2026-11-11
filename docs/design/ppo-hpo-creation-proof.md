# Positive creation evidence for the HPO GPU lease

Unit5a scoped fix design under [the management design](ppo-hpo-management.md).
Independent conformance review reproduced two blockers: absent mutable owner
fields could mint a never-started receipt; a retry after ancestry fsync failure
could omit the failed parent sync. Root accepts both findings. Architecture
review of the following minimal owner-observation extension is READY. Both
implementation findings are now independently re-reviewed ADDRESSED; positive
owner observations and ancestry retries pass scoped component checks. Concrete
Source/coordinator wiring and full integration remain later work.

## Preserve the actual owners

Do not create a process supervisor, replace creation/cancellation/reap algorithms,
or use stack inspection/cleanup booleans as evidence. Add narrow private default
no-op observation hooks on the existing reservation/creation owners. Only audited
HPO tokens interpret them. Old ledger accounting and lifecycle semantics remain.

Worker launch sites (`structured_worker._spawn` and `card_instance_worker`'s
concrete constructor) create/store one original `threading.Event` before resource
reservation. Within the actual synchronous stdlib Popen operation they notify
the SAME reservation of creation start and finish, carrying that exact owner,
event, locally returned Popen or actual creation exception. Set the original
event in the actual finally path before normal failed-start/reap cleanup.
Audited tokens retain owner/event/process identities and positive completion
evidence. Missing/replaced events, missing finish, unknown outcome or an
unproven BaseException cannot prove never-started. A returned process always
requires actual original-owner cleanup/reap, even if a later operation fails.
Tree-only scoring/native-owned launch routes remain closed until specifically
bridged; do not infer support from an empty ledger.

Learner startup already has a precreated `_startup_boundary` Event and the
`LearningProcess.start` LOCAL creation Future. Capture the original Event at
reservation. Add a private hook immediately after actual create_task and field
assignment, before the first await, to pass that LOCAL Future through
`StructuredLearningProcess` to the SAME reservation. Retain its identity even if
owner fields change. A hook at the actual default stdlib `_create_process`
Exception captures that exact exception; no-child proof requires the original
Future's exception to be the SAME observed exception. Actual create_task failure
after closing the unstarted coroutine has a distinct no-Future observation.

Require the original Event unchanged and settled, exact current/original Future
identity when one exists, and actual owned returned process/reap or positively
observed no-child exception. Absent/None fields or `_creation_no_child=True`
alone never suffice. Do not rewrite owner fields to manufacture settlement.

## Callback failures and compatibility

Observation errors cannot lose a pending creation or returned process. Preserve
the existing LOCAL Future wait/adoption and original cleanup path even when an
observer raises; retain/report the observer failure after settling actual
resources. Likewise worker observations in try/finally retain any locally
returned process for the existing cleanup owner. Preserve the primary creation
exception and record additional observer/cleanup failures without inventing
success. Uncertainty keeps the durable admission unresolved.

The limited existing files are `structured_resources.py`, `ppo_run.py`,
`structured_learning_process.py`, `structured_worker.py`, and
`card_instance_worker.py`, plus the new lease and its tests. The eight fixed ASTs
(including `_controls`, not `LearningProcess.start`) remain exact. No reverse
dependency to the HPO lease is added to existing owners/resources: default hooks
know nothing about upper campaign/Source policy.

## Ancestry durability

Before ANY admission can return, reconcile required fixed directory ancestry
durability even when all paths already exist. A previous failed mkdir/parent
fsync cannot become implicit success on retry. Conservatively re-fsync the root
and its required ancestors (through filesystem root where needed), checking
namespace identity, before issuing the admission. Permanent storage failure
continues to block; no claim of impossible rollback or I/O guarantees.

## Verification

Reproduce absent SimpleNamespace barriers and replaced pending Future/Event;
require refusal without issuing a termination receipt. Positive tests use actual
existing worker/learner creation/close owners, real CPU subprocesses or real
stdlib missing-executable exceptions, preserving default asyncio machinery.
Test cancellation/delayed creation and observer failures: actual processes must
still be adopted/reaped and uncertainty must not clear admission. Inject an
ancestor parent fsync failure, retry against existing paths, verify the same
sync is required until it succeeds. Recheck exact old bounds/accounting and
creation cleanup regressions. No GPU or experiment qualification is implied.
