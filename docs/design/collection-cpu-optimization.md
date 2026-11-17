# Bounded collection CPU, communication and transfer investigation

User-approved 2026-11-17. Planning review by `bot_architecture_reviewer` is READY,
including the explicit journal/capture exclusion and shared-pilot accounting.
This document is the execution specification; normal production defaults stay
unchanged. Baseline Bots `f3aeca2`; Simulator
`91d5926a7572db327eaea3751e7c24f428120996`, normative external interfaces section21.1.
ADR0044/0046 and the existing collection/HTTP investigation evidence apply.

## Goal and constraints

Identify collection improvements by measured32-game wall time, keeping model
mathematics/IEEEFP32, GPU inference/learning, policy information and learning
unchanged. Include actor CPU preparation/dispatch/transfers, candidate generation,
parent validation/recording and IPC/HTTP. No CPU-inference switch, larger-model
performance generalization, dependency/Simulator change or normal adoption.

All actual runs retain the existing MLP/GRU/Transformer width192/two-layer
configs, seed17, workers4/concurrency2, games16×updates2, epoch1, D1M, wall600,
Source2GiB/numeric512MiB/collector2GiB/parent1GiB/save1GiB, managedRAM24GiB/
CUDA12GiB and original phase deadlines. Chunk256 and one Torch thread remain.
Only fresh healthy `incomplete/update_cap` is expected, never completed HPO.

Primary outcome is original run `collection_seconds` summed over two batches;
secondary is the original coordinator call interval. Concurrent/nested call
totals are diagnostic, not a wall partition or predicted savings. CPU process/
thread time, useful CPU work, driver busy-wait, CUDA stream events and kernel
activities must be distinguished. No claim of learning quality/final-test success.

## Responsibilities and experimental identity

`benchmarks/collection_cpu/` owns bounded installers, diagnostics, input/replay,
actual composition, selection/aggregation and audit tooling. It may reuse the
existing inference experiment and original local runner/measure/analyze helpers;
missing historical prerequisites fail explicitly. No production policy/trainer/
transport or persisted format is replaced. Tests own mechanism verification.
Root owns design/configs, source freezes, execution, audits and acceptance;
one implementer owns Python/test files, independent reviewers do not edit.

Every launch first writes an immutable sidecar binding code/config/input/runtime/
native/Simulator hashes, effective experiment options, run/context IDs, resources
and a new output namespace in `runs/collection-optimization-v2/`. Production
identities are explicitly base-only; experiment metadata grants no new Source,
objective, resume/campaign/evaluation or training qualification. Keep original
GPU admission/custody, numerical worker, save, cleanup and termination receipts.
Ordinary source-matched resume and all historical bytes remain strict/unchanged.

## Passive baseline and replay inputs

Run one32-game passive diagnostic per model separately from performance trials.
Observe parent validation counts/CPU by caller, conversion/JSON/hash, Journal/
Capture, busy callbacks; actor generation/tokenization/tensor/transfer/enqueue/
sync/return; IPC construction/read/completion-to-resume; HTTP stages/counts.
Never expose diagnostics, IDs, GameSeed, another viewer or future events as
policy input. Preserve same-player Observation/events/typed metadata and optional
same-revision atomic hints; no SequenceOptions/rejection probing.

Prepare every learner-seat sequence from both original Numeric6 batches in the
accepted32-game parents:314MLP/314GRU/310Transformer decisions. Validate original
parents, source selection and snapshot linkage, then bind batch0/G0 and batch1/G1
to their saved inference manifests/shards. Explicitly label learner-only replay,
not all465/461 neural-actor decisions. Keep diagnostic historical import separate
from ordinary resume; fail instead of substituting another checkpoint/producer.
Every sequence includes its full own prefix; reset cache/RNG between episodes.

Use three alternating fixed-input pairs, separate warmup/first/warm observations,
same G0/G1 weights/inputs and original sampling. Choose T only within this common
replay population, never rank P/H from actor replay. GPU profiles are separate:
per model/transfer option select three small and three recorded-tail decisions,
preserving their own prefixes; trace cap512MiB/run. Record actual H2D/D2H/DtoD
and synchronization/kernel/API counts/bytes and memory, not just wrapper counters.

## P: narrow parent validation reuse

Install a decision-local ContextVar around original `_select_and_journal`.
It belongs to the original asyncio Task and thread, holds strong exact Frame/
CandidateTable references and records proof only after original full validation
succeeds. Only exact object/participant/revision identity can reuse that proof;
different objects, Tasks/threads or failed checks execute original validation.
Keep selected index/action reconstruction checks. Clear references in finally
on return/cancellation/failure. Explicitly suspend reuse before BOTH original
Journal.begin_frame and Capture.begin_decision; they run full original checks.
Source/audit/other trust boundaries never use the proof. No hash-only authority,
cross-decision cache, new wire field or broad thread migration.

## T: existing tensor-math seams, transfer-only variants

T0 is unchanged eager control; existing typed reuse is a selectable historical
comparison. Existing frame-input, factorized-logits, candidate-feature and refs
helper seams remain the math owners; do not duplicate policy orchestration.

- T1: typed frame packing plus one per-decision typed pack containing candidate
  fields and all padded candidate refs (current max_positions18). Transfer refs
  once, then use GPU views for original256-row scoring chunks.
- T2: T1 with one8-byte-aligned uint8 arena per frame/candidate phase, typed views
  preserving original bit patterns, one H2D per phase. No FP recalculation.
- T3: T2 with preallocated pinned CPU storage and nonblocking copies. Keep input
  storage immutable until original final-forward GPU sync confirms consumption;
  add no hidden warmup, early reuse or unreported synchronization.

Choose the valid non-T0 option minimizing the sum of per-model median full
G0/G1 warm replay wall time. Within1% of that minimum prefer historical reuse,
then T1,T2,T3. Report if T0 is faster; do not relabel historical reuse as new gain.

Charge the same96MiB/physical-worker experimental envelope to BOTH control and
candidate RAM/CUDA admission phases and actual reservations:6cached identities
×12MiB +12MiB transition +12MiB metadata/temporaries. Count CPU+GPU, alignment,
pinned capacities and old/new/in-flight overlap before allocation; each tensor
also retains the original64MiB cap. Explicit capacity failure, no cap increase,
candidate drop or OOM fallback. On failure keep storage until real sync/process
retirement. Scrub entire capacities/padding before reset acknowledgement; retain
original one in-flight decision and viewer ownership. Observe physical peaks as
soft evidence, not universal capacity certification.

## H: same-revision independent reads

After validated acting-player Observation, concurrently await original Events
and ActionOptions, retaining original event order/cursor and all independent
validation. Do not act until both succeed. Bound to exactly two owned tasks;
on failure/cancellation cancel and join both under original deadline before
original cleanup. Do not reuse Act-response events/options/Observation as policy
input, add an endpoint or modify the Simulator. Avoid copying the environment
or worker orchestration; use a narrow tools-only installer at an original seam.

Candidate generation is profiled, but existing ~0.10s memo gain, pack encoding
~0.078s and JSON ~0.06s remain low-priority historical evidence, not new variants.

## Actual selection and confirmation: maximum54 performance runs

One four-arm exploratory block per model uses shared control C plus P/T/H:
MLP C,P,T,H; GRU H,T,P,C; Transformer P,C,H,T.12runs. Rank by signed
sum_model(C.collection_seconds - V.collection_seconds), equal model weight;
percent denominator is sum_model(C.collection_seconds). Differences within1%
of that control sum are unresolved ties; deterministic preference P,T,H selects
the trial order only, not a claimed performance win. Select up to two mechanically
valid candidates even if their pilot timing is negative.

For each selected candidate/model run two NEW independent confirmation pairs,
V/C then C/V: at most24runs. Report each singleton as one exploratory comparison
plus TWO independent confirmation pairs, never three independent confirmations.
Show selection bias and the confirmation results separately.

If two candidates are mechanically compatible/correct and fit the original
resources, run their combination regardless of individual timing signs:
three fresh pairs/model C/B,B/C,C/B,18runs.12+24+18=54; plus the three passive
diagnostic runs, maximum57actual invocations/1,824natural games. Invalid candidates
reduce execution honestly; no unbounded replacement or extra tuning loop.

Run performance sequentially without profiler or concurrent tests/audits/reviews.
Cold initialization belongs to actual totals; fixed-input warmups are separately
restored/labeled. Report individual differences, medians, percent, memory and
decision/candidate/cutoff/trajectory differences. Seeds do not promise identical
trajectories. Consistent positive confirmation supports a conditional recommendation;
mixed results remain model-specific/inconclusive. Actual collection results rank
impact; no shortcut from replay/copy counts or automatic default adoption.

## Verification and acceptance

Mechanism tests: packed values/dtypes/layout/order, empty events/hand, tail chunks,
capacity preallocation rejection, full reset/padding scrub, transfer storage
lifetime/error/cancellation; exact proof identity/Task/thread/revision/failure
and Journal/Capture exclusion; H both-success and sibling-failure retirement.
Use existing numeric tolerances/frozen-kernel checks; no sampled-action equality
gate across independent games or invented numeric thresholds. Scoped Ruff/mypy
and relevant existing tests, not an unrelated whole-suite rerun.

After ALL timing populations finish, original validate_hpo_parent
(require_complete=False,max_owned_bytes=2GiB,deadline=monotonic()+180) and exact
invocation termination receipt on every actual parent; SHA/stat original files
before/after. Preserve genuine outcomes/truncations/aborts/infrastructure failures;
never synthesize actions or success for interrupted experiments. Independent
correctness and architecture-conformance reviews must resolve blocking findings.
Final report: `docs/verification/collection-cpu-optimization.md`, complete trial
ledger/identity hashes, selection rationale, errors/NOT RUN and limits. Root
updates architecture/ADR status and accepts only completed, verified work.
