# 0054: Single-process Rust player runtime

Status: accepted design; implementation and experiment acceptance pending.
Preimplementation review: `/root/runtime_architecture` READY after three blocking
ownership/packaging findings were resolved. Root adopts advisor
`/root/runtime_lifecycle_advice` option B, prepared→original-host reap→cold
continuation, rather than a campaign-wide GPU lease.
Date: 2026-11-19. Owner: root. Baseline: 9680f06.

## Context

ADR0053's trusted shared MLP inference still retains participant, broker, learner
and writer Python processes. The owner approved a single Python learning host
with Rust participant handling and a real A/B scheduling comparison. Its research
identifies v5 search compatibility, persistence custody and positive termination
as substantive boundaries, rather than treating a thread as a process receipt.

## Decision

Implement the [bounded design](../design/rust-player-runtime.md) and retain the
[approved protocol](../design/rust-player-runtime-approved-plan.md) unchanged.
The supervisor preserves original campaign and parent-run lock custody through
sequential host reentry; cold finalization retains strict Source validation and
legacy evaluation admission. Runtime-wheel packaging pins the installed companion
hash/source identity and refuses missing or mismatched executables.
Rust owns own-view policies, external HTTP and game/request state. Python retains
FP32 Torch inference and participant CPU sampling, PPO and saved state, with a
single dedicated SQLite writer thread. Add explicit runtime/producer/exit versions
and an original-handle Rust supervisor. Compare Tokio-per-game and fixed workers
using common processing; defaults change independently per entrance only after
correctness, resume, audits, <=3% timing changes and consistently reduced PSS.

## Alternatives and consequences

Keep multiprocess shared mode as comparison and old-parent resume route. Python
HTTP/player callbacks would preserve GIL/serialization costs and miss the approved
Rust ownership; merely relabeling them is rejected. An asyncio bridge, custom
lock-free queues, Rayon co-scheduling and new learning math have no initial need.
Logical participant separation replaces process separation only for the explicitly
versioned mode under the owner authorization and pinned Simulator21.1 tests.

Porting Python search/RNG is substantial; exact fixed-clock baseline cases and
independent review precede production use. No speedup or memory gain is presumed.
Existing formats keep their meanings. The design records measured/estimated,
shared/private and steady/peak allocation limits. No accepted implementation debt
yet; missing evidence remains required work.

## Verification

Fixed-clock v5 candidate/RNG parity, per-participant isolation, retirement and
buffer/queue/failure tests; external HTTP contract; both schedulers W1/W2;
Model3/4 CUDA continuous/save/resume and strict original/numeric/Replay audits;
actual single-Python process tree and original-handle termination; approved C8,
C16 and deployment-concurrency timing protocol. Record results and unresolved
limitations in docs/verification/rust-player-runtime.md before acceptance.

## Scoped core implementation evidence

The native v5 public policy core is implemented and passes independent correctness
and architecture conformance READY after one local fix round. The two blocking
parity findings (empty table after deadline and separate baseline capacity18)
are resolved and independently re-reviewed. Original broad679Python checks pass;
final11covering/245affectedPython and180native/13ignored checks pass. This is
core acceptance only; the [verification record](../verification/rust-player-runtime.md)
keeps complete-runtime/learning/performance/adoption work outstanding.

## Supervisor/packaging design refinement

Independent `/root/runtime_architecture` preimplementation supplement review is
READY. Linux subreaper adoption plus an isolated original-host group and actual
adopted-child wait statuses close the orphan-Simulator boundary; stdinEOF alone
is only a shutdown request. WNOHANG zero means still running. Cleanup signals
may establish physical retirement but never qualified completion. Unknown/reap/
durable-proof failure retains quarantine. Revisit containment if a later pinned
Simulator spawns OS children or changes groups/sessions.

A disposable Maturin1.15 wheel proves the exact wheel-only companion include
paths and0755mode; installed distribution inventory drives discovery. Ordinary
native-only builds remain companion-free. Actual runtime-wheel installation and
forced-exit tests remain required; this refinement accepts design feasibility,
not implementation or performance.

## Scoped inference component and record compatibility

The owned array/dispatcher/Python inference component passes independent
correctness/conformance READY after blocked-admission retirement and timer
representation fixes. Final55inference checks (including CUDA) and189native/
13ignored checks pass; full live runtime/performance remain pending.

The record-semantics clarification is independently READY: unchanged Journal6/
DTO3 contents are not Python-process proofs; persisted Numeric6 remains honest
only when its canonical-python encoder actually runs in Python. Native serving
encoding and Rust online capture have distinct producer identity. New strict
outer run/collection/proof/update/registry/import/completion selectors remain
mandatory. Exact outer-version mapping must pass its gate before implementation;
this clarification does not preapprove unspecified mappings or legacy receipts.

## Exact outer records and live history refinements

The [outer-record inventory](../design/rust-player-runtime-records.md) is
independently READY after explicit cache transactions and cold parent/evaluation
handoff resolved the two initial omissions. It keeps unchanged Journal6/DTO3,
normal Numeric4/HPO Numeric6 and Python checkpoint4/5; new outer selectors and
actual owner/exit authority remain explicit. Cache records carry no fictional
broker/GPU PIDs. Evaluation consumes validated new-parent inference data through
the existing independently admitted fresh runtime, without reconstructing legacy
training completion or actor9/10 authority.

The typed async game-release helper and incremental Arc-backed history/trusted
projection/token-kernel factoring also pass their preimplementation gates. They
preserve original lease/producer lifetime and feature semantics, keep untrusted
diagnostic validation, and remove redundant whole-prefix work. Actual actor
implementation, conformance and prescribed verification remain pending.

## Scoped live actor implementation

The common external-HTTP game actor and actual Tokio/fixed-worker schedulers pass
independent correctness/conformance READY after one fix round. Failed service
status now precedes route settlement, preserving actual holds through join; live
Refresh preserves arbitrary positive threshold identity. Original policy/frame/
token/hash semantics remain exact, with incremental Arc history and borrowed
immutable capture. Final197native/13ignored and537Python/18explicit skips pass,
including actual overlap/natural/Replay/CPU-CUDA cancellation. This accepts the
actor component only; real writer/Source/learning/supervisor/CLI integration and
approved performance/default gates remain outstanding.

## Thread storage accounting and failure ownership

The [storage bridge design](../design/rust-player-runtime-storage.md) passes
independent preimplementation architecture READY. One native parent recording
account serves native captures and additional Python staging/pack byte leases;
original bundle/view custody is charged once. Decoder/private metadata scratch
has its separately admitted role, and a new restrictive formula still requires
bounded evidence/root review. No second interpreter/IPC estimate is inherited.

A genuine fault probe shows transitional readonly FD ownership could be lost
when old-original close failed; process exit formerly retired it, thread exit
would not. The new owner must retain every transitional/db handle before fallible
handoff and report uncertain cleanup honestly. Actual thread/bridge implementation
and fault verification remain pending.

## Scoped thread storage engineering acceptance

The actual SQLite thread/canonical original encoder and single native recording
account pass independent engineering correctness/conformance READY after nested
FD, construction, traceback-buffer and cancel/seal repairs. Final273affected
checks pass; normal old process-worker behavior is preserved. Explicit workspace
loan is soft planning, and bounded studies retain their original producer hashes.
Production/default retained-active-bank calibration and prescribed workloads stay
open, alongside Source/local-learning/supervisor/CLI integration. This is not
archive/learning qualification or performance adoption.
