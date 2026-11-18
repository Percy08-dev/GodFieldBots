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
