# Actual-path telemetry supplement

## Decision and scope

The approved measurement breakdown has no core-only qualifier. Original84core,
25profile and18actual results at `a0f243f` remain valid for their scopes, but the
missing actual learner/coordinator CPU/memory and isolated retirement timings
prevent unqualified completion. Root accepts the advisor's bounded remedy:
six additional diagnostic-only actual runs, baseline/chosen pair per model,
using192width/2layers/seed17/workers4/concurrent2/games16×updates2/epoch1/D1M/
wall600 and every original cap. Six is a coverage decision, not a count specified
by the plan. Do not pool these observations into the original18 timing medians.

Architecture review is required before implementation. Numerical execution,
grouping/admission, model/backend, Source/checkpoint/Wire7 and original process
owners remain unchanged. Preserve old records and committed source archive
`runs/learning-update-optimization-v1/source-a0f243f.tar`. New code and six-run
inputs receive a fresh freeze/identities under a diagnostic subroot. Revisit
performance reruns if a difference extends beyond observation.

## Explicit control and compatibility

Extend the hashed private TrainingExecutionSpec with observe_actual:boolean in
experimental version2. Version1 stays an exact closed legacy schema with
observation disabled, never reinterpreted as observed. Version2 observation
requires a nonempty absolute diagnostic directory and no profiler window. Both
stay <=4KiB, SHA-checked before learner creation. Ordinary config/checkpoints,
argv/worker module and Wire7 fields retain their meanings.

The fresh actual helper adds a telemetry-diagnostic flag, creates this spec and
labels request/result timing_eligible=false. Normal performance invocations keep
observation disabled. Resume/pause/campaign/evaluation gain no entrypoint. Root
validates/freeze six fresh identities/outputs. A small diagnostic root controller
may call existing actual.main six times and existing audit_trial six times after
runs; no generic campaign/population framework is needed.

## Responsibilities and bounds

One small stdlib training-side helper owns clocks, fixed Linux proc samples,
bounded aggregates and immutable publication. No HTTP/CLI, process/Source authority
or policy input. Retain primitive bindings only, never rollout/token/NumPy/Tensor/
Source/snapshot references; CUDA event handles may survive until an existing sync.

Use50ms requested sample cadence plus immediate start/end/phase-boundary samples;
record count, largest observed gap, first/last and maxPSS/maxRSS. Retain aggregate
statistics, not a runtime-sized list. Read fixed /proc/self/smaps_rollup with an
8KiB bound and explicit failures. process_time covers that process's threads,
excluding children; wall is separate. Peaks are sampled, not guaranteed physical
maxima. Never sum independent maxima into a simultaneous peak.

At most one daemon sampler thread/observed process, owned stop Event and250ms
bounded join. No unbounded join/cancellation wait. A stopped sampler is required
for diagnostic completeness. On timeout retain its owner until actual process
exit, record failure and prevent accumulating another sampler; original
helper/learner process/reap owners finish normal cleanup. Telemetry never confirms
GPU termination or extends original deadlines. Existing OS/proc/filesystem
blocking limitations remain explicit.

Bound bookkeeping/stack allowance16MiB/observer, sidecar64KiB, one coordinator
record and two update records/run. Earmark inside existing256MiB model workspace
and management allowance; check compatible workspace/management ceilings before
creation. A named process-local observation pool holds one exclusive16MiB lease
from that existing envelope. This is a subset already included in the reservation,
never extra budget or headroom credit; Source/artifact budgets are not repurposed.
Enforce at most one live observation lease/process and refuse another before
creation. Hold the lease for the observer's entire lifetime, including a join
timeout; release only after stopped sampler and publication. A timed-out owner
remains held through actual process exit, not merely a timeout return. No generic
budget framework or new coordinator-writer hook is required.
Do not enlarge resource/caps or reduce tensor-group capacity. A ceiling is not
measured free memory: this is bounded overhead within reserved workspace, not a
whole-process physical bound. Keep1GiB helper report cap.

## Concrete intervals and publication

Coordinator: existing exactly-once wrapper around original train_hpo_ppo owns
coordinator-only wall/processCPU/PSS/RSS sampling. Start immediately before the
original call; stop in finally after return or existing cancellation cleanup.
Publish after original owner cleanup; exclude all children from process metrics.

Worker: observe existing per-update receive/validation, learner update call,
input retirement, snapshot export, snapshot save and snapshot transfer. Record
phase wall/processCPU and boundary samples. Existing numeric-loop metric remains
unchanged and separately labeled.
Start only after the original closed update-command identity/generation checks
and before `_receive_numeric`; do not observe/reset while waiting stdin or close.

Record CUDA begin immediately before learner.update_owned, end immediately after
it returns; read after the already-required learner.synchronize at retirement.
No endpoint/group synchronization added. The event interval includes host gaps
and validation, not pure kernel time. Reset allocator peak counters before the
request, read current/peak allocated/reserved after existing synchronization;
post-snapshot observations, if any, have separate scope. When observation is off,
there are no sampler threads or CUDA markers.

Worker retirement starts before existing del numeric, includes existing sync,
delivery.complete(sync callback) and control.close, and ends after release/
completion sends. The parent's earlier completion wait overlaps numerical work
and cannot be renamed retirement latency. Parent-side batch retirement stays
with original generation-scope close after snapshot ACK/commit.

Export/save/send remain original calls/order. Finish observer and publish after
_send_snapshot returns; worker writes before reading next request/close, so final
close/process reap is the publication boundary. No extra ACK/wire field or
binding callback. Sidecar version2 binds generation/transaction/resume identity,
model/variant/specSHA/source/pid and interval definitions. Old v1 counters stay
old evidence.
Publish through one exclusive <=64KiB pending file: write and successfully close
before hard-linking to an exclusive final name, then remove pending. A close
failure cannot expose a final complete record. Any observer-pending residue
invalidates helper completeness, including unlink failure after final link.
No overwrite or link-error fallback. Retain lease/FD owner on close/unlink
uncertainty through actual process exit; at most one pending per held lease.

## Failure ownership

Authoritative learning/FD/save exceptions propagate unchanged through existing
cleanup, with observer stop attempted in finally. Observer/read/thread/publication
failure cannot mask them or create ACKs/transitions/receipts. For observer-only
failure preserve actual training/cleanup, separately mark diagnostic incomplete.
After original close/reap the helper validates all sidecars and stopped samplers;
missing/invalid observations cause diagnostic failure, never inferred success.
No retry, data dropping, silent metric substitution or observation-budget growth.

## Verification and acceptance

Focused tests: off means no sampler/markers; bounded storage/cadence/gaps and
phase scope; stop/join success/error/timeout; original exception preservation;
exclusive lease lifetime/refusal and no release on timeout;
no input aliases; exclusive-create/size bounds; closed v1/v2 settings and fresh-
only diagnostic labeling. Exercise original CPU owned worker lifecycle, preserve
existing67mechanism and relevant creator/wire checks, plus bounded CUDA checks.
Independent correctness/conformance verify observational source difference before
six runs. Original84core/25profiles need not repeat solely for observation.

After six sequential runs require original healthy incomplete/update_cap,
two updates,32natural games/run, genuine receipts and strict Source/numeric
checks. All12 learner-update and6coordinator records require finite nonnegative
wall/CPU, samples/count/gaps, required CUDA event/allocator fields and stopped
samplers; zero/unknown cannot replace missing. Report192additional diagnostic
natural games separately from576comparison games; no diagnostic enters timing
medians, objective or quality counts. Final evidence reviews then reconsider
whole-plan acceptance.

## Accepted implementation and evidence

Observational source `f067d03` passes both code reviews, affected/final targeted
checks and a genuine CUDA worker/receipt test. Six fresh diagnostics complete
under the original caps/600s deadline, yielding12worker/6coordinator records,
192natural games and six strict Source/numeric/receipt audits with540unchanged
file instances. All samplers stop, all required observations are finite/bound,
no publication residue remains, and diagnostics stay excluded from original
performance medians. Both final evidence reviews are FINAL READY and root
accepts this measurement completion; original a0f243f timing evidence stays
separately source-pinned and valid. Scope and sampled-peak limits are retained.
