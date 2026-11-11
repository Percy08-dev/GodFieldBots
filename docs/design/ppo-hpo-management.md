# HPO evaluation and global GPU admission

Bounded concrete unit5a design under [the HPO plan](ppo-hpo.md). Implementation
uses an isolated worktree while the main Source/native unit proceeds. Root owns
integration; these new components do not change Source or process ownership.
No extra dependency, actual GPU work or Simulator experiment is needed for their
component tests. Scoped `bot_architecture_reviewer` design review is READY,
with no blocking findings before production implementation.

## Pure strict evaluation

`gf_evaluation.hpo` owns immutable `EvaluationPlan`, expected `RunIdentity`,
`SeedEvaluationProjection` and terminal-game records. The plan binds the complete
training-seed and GameSeed sets, D, evaluation-condition hash and each seed's final
run/checkpoint/Source-receipt identity. Terminal records freeze the actual natural
result DTO and learner seat, plus unique game/episode identities and projected
source/transport/save/cleanup status. Use the existing result validator.

Reject missing/extra/duplicate seeds, GameSeeds, seats, game/episode IDs; wrong
run/final-D/checkpoint/conditions/Source identity; interruptions or unsuccessful
health; and nonfinite fields. Score natural win1/draw0.5/loss0 from learner seat.
Average each complete seat pair, then GameSeeds, then equal training-seed means.
Zero-decision natural games stay in the panel. No partial mean or failed-game zero.
Direct comparison, if included here, is a separate exploratory matchup table.

These are structurally checked projections, not verification capabilities. The
concrete evaluator must independently verify actual Source, Simulator results,
saved originals and cleanup before supplying them. No JSON loader or aggregate
result grants Source/formal/evaluation authority. The pure module imports only
common/domain helpers; CLI owns I/O and live composition.

## One persistent GPU admission namespace

`gf_training.hpo_gpu_lease` uses one fixed private per-user persistent directory
`~/.local/state/godfieldbots/hpo-gpu-admission-v1` across all campaigns/checkouts.
Initial operation conservatively serializes all HPO devices for that user. No
per-campaign override and no temporary-directory clean-state assumption. Tests
replace a private module path constant, without writing user global state.

API:

```
admit_gpu_job(JobContext(...), *, global_cuda_bytes, max_workers) -> GpuJob
GpuJob.ledger                  # SAME audited ReservationLedger
GpuJob.confirm_termination()   # capability/actual cleanup -> immutable receipt
GpuJob.close()                 # releases flock ONLY, never infers cleanup
confirm_environment_restart(*, expected_ticket, operator, confirmation)
```

Durable admission finishes before any GPU allocation/spawn. The live job ticket
is an unpersistable in-memory capability bound to issuer PID, original admission,
held lock and the SAME ledger. JSON, a reconstructed/forked ticket, another empty
ledger or a saved PID cannot confirm termination. Seal further reservations when
confirming/closing. Unfinished durable admissions always mean RECOVERY_REQUIRED,
including after flock release, process death, elapsed time or boot-ID change.

## Reuse actual resource and process ownership

Use a narrow `ReservationLedger` subclass solely to retain auditable lifecycle
history. Call `super().reserve_worker/reserve_learner` exactly once. Synchronously
construct a narrow audited reservation from that returned dataclass's existing
fields and replace only its SAME ledger slot/key before returning it. This does
not copy CUDA/budget arithmetic, mutate `__class__`, monkeypatch globals or create
a second ledger/process owner. A failed promotion poisons/retains uncertainty.
Constructor technical bounds use a narrow private class-attribute hook: ordinary
Ledger remains12GiB/4workers, while the known HPO audited subclass can express its
declared CUDA budget through the same constructor/accounting. Current HPO game
concurrency1..2 and worker count<=4 remain unchanged; this hook does not expand
concurrency. HPO config, joint preflight, recorded-state and runtime helpers must
be aligned before production use. Old paths retain their original bounds.

Audited tokens retain actual Popen/asyncio.Process and cleanup-owner object
identities before the base token clears them. Require the exact spawned process
and exact owner.process, settled creation task/startup barrier, no pending work
and actual confirmed exit before delegating original reap/cancel. Strengthen the
base worker PID check with process-object identity. Never use PID liveness or
elapsed time to infer exit. Preserve history after actual reservation release.
Known never-started cancellation needs the original settled creation/no-child
proof, rather than missing PID. Full confirmation requires all recorded tokens
closed, no active worker/learner/update and the current job's original capability.

The existing controller still owns cleanup and non-GPU Source/transport health.
Its production composition must call confirmation only after those owners close;
a GPU termination receipt alone never makes failed data eligible. New module
imports stdlib/common and the existing narrow resource helper, no Torch/Optuna or
upper process/simulator layers. Any necessary new exact ledger type acceptance
belongs to the HPO path only, preserving old guards.

## Durable records and recovery

Under the global flock keep `current.json` with closed
`{schema_version:1,ticket,admission_sha256}` and immutable `jobs/<ticket>/` files.
Admission fields:
`{schema_version:1,kind:"hpo-gpu-admission",ticket,context,boot_id,owner_pid,
admitted_unix}`. Context is closed
`{campaign_id,operation_id,kind,run_id,evaluation_id}`; absent identities are
explicit null, never policy inputs.

Termination fields:
`{schema_version:1,kind:"hpo-gpu-termination",ticket,admission_sha256,
context_sha256,boot_id,confirmed_unix,resources}`. Each resource records
`{kind,key,pid,outcome:"reaped|never_started"}`; PID remains diagnostic.
Only the original live capability can issue this from actual lifecycle proofs.

Write immutable admission and fsync it before the current pointer, and finish
all required fsyncs before returning admission. Scan for unconfirmed orphan
admissions too, including pointer-publication failure. Termination uses an owned
exclusive temporary, file fsync, atomic publication and directory fsync. Current
need not clear: it is confirmed only by its exact validated termination receipt.
There is no fragile closed-flag rollback protocol.

Any write/fsync failure returns failure and poisons the current owner. Corrupt or
unreadable records block and are never overwritten as idle. Under the lock a
later reconciliation can validate and re-fsync an exact already-visible genuine
termination receipt; it must not infer cleanup from process/time heuristics.
Repeated storage failure cannot promise successful rollback or durability; report
that limitation honestly and refuse work until consistent durable evidence exists.

Explicit restart receipt:
`{schema_version:1,kind:"hpo-gpu-environment-restart",ticket,admission_sha256,
admitted_boot_id,current_boot_id,operator,confirmation,confirmed_unix}`.
Require the exact unfinished ticket and explicit operator assertion that the
environment restarted and resources are gone. Never infer approval from boot
change alone, reboot automatically, edit original admissions/deadlines or
requalify run data. Preserve abandoned-job history. Exact committed receipts may
reconcile idempotently; mismatched receipts fail.

## Verification

Strict scoring/missing/duplicate/failure/final-identity/equal-weight cases;
clean optional-free imports; actual owned CPU subprocess creation/wait/reap and
creation-failure boundaries; independent campaigns/processes competing on the
same lock; abrupt exit/released flock with unconfirmed job; forged/forked/wrong
capability or process/empty-ledger refusals; injected atomic/fsync/corruption
failures; exact-receipt retries and explicit restart confirmation. These tests
exercise lifecycle behavior without claiming GPU, Source or strength evidence.
Independent correctness/conformance reviews precede integration. Concrete HPO
coordinator/evaluator later supplies the SAME audited ledger and closed sources.
