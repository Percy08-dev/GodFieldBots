# Experimental PPO update optimization

## Scope and fixed conditions

Implement the approved 2026-11-17 plan at attachment
`fc071d2a-355a-40ea-85a4-a5410db0e3cf/approved-plan.md`. The deliverable is
experimental implementations, measured comparisons and a recommendation. Normal
defaults, model/checkpoint schemas and all existing experiment evidence retain
their meanings. This design requires independent architecture approval before
dependent implementation. The preparation evidence is
[the input investigation](../verification/learning-update-optimization-preparation.md).

Keep model architecture, FP32 losses/gradients, PPO objective/coefficients,
selected endpoints, PCG64 shuffle, logical minibatches, TBPTT, gradient clipping
and Adam step boundaries. Keep deterministic IEEE FP32, disabled CuDNN GRUs,
MATH SDPA, existing checkpoint recomputation policy and final synchronization.
No AMP, TF32, compile, new dependency, simulator change or training-quality gate
is part of this work. Existing Source, memory, deadline, FD, save and positive
termination owners remain authoritative.

## Responsibilities and interfaces

`gf_models.structured_network` owns shared actor/critic mathematics. Extend its
private tensor helpers for a leading batch dimension where required. The normal
endpoint path retains its original operand construction, enqueue order and
checks. Private model-side learning helpers may own immutable typed operands,
batched frame/event/temporal/candidate execution and their pure shape estimates;
they import no training, HTTP, CLI, file-management or experiment-control code.
The existing inference experiment and reusable inference buffers are not used
for training.

`gf_training.hpo_numeric` remains the single PPO trainer. Its normal baseline
loop and `hpo_losses` retain their current semantics. A private execution spec
selects cumulative `baseline`, `metrics`, `packed`, `microbatch`, or
`shared-prefix`; the last is rejected for MLP. New training-side helpers own
group planning, device metric reduction, joint admission and diagnostics. They
do not own a second trainer, optimizer, reward/GAE implementation or network.

Use an immutable closed `TrainingExecutionSpec` in a training-only module. It
contains a schema/version, variant, diagnostic destination and an optional
bounded profiling window/settings. Unknown keys/types/variants fail before
allocation. Serialize canonical JSON to a fresh regular file of at most 4096
bytes, pass its absolute path and SHA-256 explicitly through the common
coordinator, original operations/child constructor and original fixed
`-E -m gf_training.hpo_owned_worker` creator. The worker bounds its read, checks
the SHA and parses the closed schema before constructing the learner; scrub
argv before model imports. No environment patch, creator replacement, alternate
worker module, Wire7 field, normal config key or checkpoint key is introduced.
Alongside path/SHA, the original creator passes the numeric byte cap from its
SAME active `LearnerReservation.resident_bytes`. Validate it against the
model/binding ceiling before worker allocation and bind group planning to that
cap. This is transport of the existing reservation, not a new allowance;
ordinary worker argv remains unchanged when the private spec is absent.

The private actual-training parameter is fresh-only and producer-owned-only.
Reject resume, pause, campaign and evaluation use before GPU/process creation.
Bind the spec and base identities in a separate immutable experiment request
containing source/config/input hashes, run/context IDs and fresh output paths.
Experimental parents never supply a completion objective or ordinary quality
authority. Ordinary entrypoints omit the private parameter.

## Input, grouping and autograd

`metrics` retains endpoint execution order and per-endpoint forward/backward.
Remove only the redundant explicit per-endpoint synchronization. Compute the
same five metrics as detached FP32 endpoint values, accumulate their sums on
device in FP64, and transfer the five totals once at update end. Metrics never
enter loss or gradients. Keep finite actor/critic and each real endpoint loss
checks before its group's backward, gradient checks at clipping, parameter and
metric checks plus synchronization before completion. A nonfinite endpoint may
not be hidden by a group average or cause a successful ACK/save.

`packed` adds contiguous NumPy/Tensor operands for one admitted input window,
with batched transfers by dtype and views into immutable storage. It continues
endpoint forward/backward order. Each window's raw operands, candidate refs,
masks and targets stay live and unmodified through its last backward, including
candidate checkpoint recomputation. Release model/autograd references before
retiring the window. Owned endpoint liveness is checked before constructing any
view; packing grants no new token/Source authority.

`microbatch` plans only inside each already shuffled logical minibatch. Group
by exact detached-prefix length and gradient-suffix length (MLP uses one current
frame), stably sort each group by real candidate count, then greedily accept at
most 16 endpoints while `B * max_candidates <= 2 * sum(real_candidates)` and
joint admission succeeds. Split before allocation. A singleton that cannot fit
fails; OOM never triggers a fallback or endpoint removal. Each endpoint appears
exactly once. Compute per-endpoint PPO terms and mean over the microbatch;
backward with weight `B / logical_minibatch_size`. Zero-grad, clip and Adam each
occur once per original logical minibatch, including tails 53/5/47/7.

Batch frame arithmetic, including state/card/hand encoders, with explicit live
dimensions. Pack positive-length event sequences with their actual lengths;
scatter their final GRU state into the frame batch. Empty intervals produce the
existing constant zero vector without running the GRU. Entirely unused branches
retain `grad=None`. Hand masks retain empty-hand behavior. Batched GRU temporal
inputs have equal prefix/suffix lengths; detached prefixes use current weights
under no_grad, followed by the differentiable suffix. Transformer absolute
offsets and causal masks apply at every layer. Keep the existing candidate
chunk/checkpoint policy, candidate order and one normalization per endpoint.
Padding candidates never enter selection, normalization, entropy or loss.

`shared-prefix` first groups endpoints from the SAME episode, viewer and TBPTT
cut. Only groups with at least two endpoints and cut zero can share under fixed
TBPTT. Identity includes the actual input origin and matching own-frame prefix;
textual episode equality alone is insufficient. Greedily split these groups by
the same endpoint/padding/joint-memory caps. Compute each accepted group's own
history once through its maximum endpoint, with unidirectional GRU or causal
Transformer. Gather each endpoint's own context AND its own frame's actor-card
representation. Compute all accepted endpoint losses and backward once; destroy
the graph afterwards. Splits recompute their required history, never retain a
graph into another group/Adam step. Remaining endpoints enter ordinary
microbatches. Frame deduplication is diagnosed separately from Wire7's existing
logical endpoint/context/suffix counters; 481/491 are opportunity upper bounds.

## Joint memory and lifetime ownership

Use one pure joint estimator for planning, learner admission and actual phase
reservation. Its fixed part is the existing
`parameter_optimizer_cpu_reference_bytes + context_workspace_bytes`, counted
once. Sum the existing conservative activation/copy envelope for every
simultaneously retained endpoint/history, inflate for batch candidate padding,
and add the full packed CPU AND GPU inputs, target/mask/offset metadata, retained
shared frame/card outputs and checkpoint/backward coexistence. Do not discount
the shared group's activation estimate on the strength of deduplication.
The existing fourfold conservative factor applies to activation and temporary
copy envelopes. The old endpoint formula assumes serial detached-prefix frame
encoding: its suffix multiplier alone does not bound a batched prefix. Also
derive an aggregate activation envelope from every effective batched shape,
including ALL concurrent prefix/suffix event GRUs, frame/hand intermediates,
retained suffix/shared outputs and candidate/checkpoint/backward intermediates.
Use at least the larger of the summed legacy-plus-padding/shared envelope and
four times this complete shape-derived activation/copy sum, then add packed
CPU/GPU and metadata residency. Individual tensor checks are separate. Account
for packing construction/source coexistence as well as final tensor storage;
previous windows must retire before allocating successors.

Independently enumerate/check all effective tensor shapes, including batched
candidate gather `[B, chunk, max_positions, actor_hidden]`, batched frame/card
features, event intermediates and each attention score. Each tensor is at most
the configured 64MiB limit. Aggregate checks alone do not prove this constraint.
The 16-endpoint limit is a ceiling, not an admission guarantee. Include actual
rollout residency and existing parked actors, Source and save/SYNC reserves.
Pass the existing admitted learner reservation into execution planning; an
experimental window may consume only its remainder, never enlarge the original
RAM24GiB/CUDA12GiB/Source2GiB/numeric512MiB/collector2GiB/parent1GiB/save1GiB caps.
If the existing conservative learner phase reservation already dominates the
new joint estimate, document/check that comparison rather than charging twice.

The actual owned batch stays in its original UPDATING lifecycle through all
backwards and the final sync. No view/bank/Source reference may survive the real
input-retirement ACK. Failures retain original cleanup/reap ownership and poison
the learner/batch. No synthetic single-use batch, state rewind, successful save,
transition or termination receipt may be created for diagnostics or failures.

## Fixed-data diagnostic import

`benchmarks/learning_update/` owns prepare, diagnose, fixed-core compare,
fresh-actual compare and summarize entrypoints. Reuse original saved-array/token
readers and historical actual observer/auditor where appropriate; preserve and
freeze required local historical helper files. Do not inherit the inference
experiment's actor substitution or 96MiB/worker increment.

Prepare pins all six numeric6 inputs and G0/G1 inference5/resume5 states from the
three investigated round1 baseline parents, plus file hashes, exact producer
source/schema identities, materialized model/training/actions/binding, code and
runtime identities. Diagnostic data is explicitly train-only and unqualified.
A diagnostic-only NumPy endpoint/vector adapter feeds the SAME tensor core to
every variant. It performs no Source/capability construction and excludes full
Source verification, ownership/FD management, saves and retirement from core
timing; its time is not the previous approximately 22-second interval.

Normal `restore_learner` and `_validate_resume_manifest` keep their mandatory
current-source match. If common metadata validation must be factored, its
ordinary wrapper continues to require current producer equality. A separate
diagnostic import verifies the explicitly pinned recorded producer and complete
closed schema/runtime/model/optimizer/hash/array invariants, then calls existing
validated state application. It never rewrites producer metadata or disk files,
and never enters ordinary restore or owned update authority. Restore weights,
Adam moments/state presence/parameter steps, Torch CPU/CUDA RNG and PCG64 before
every warmup/measurement. G0 starts at step0; G1 starts at step2.

Run MLP's four and sequence models' five variants on each of their two batches,
three repeats ordered forward/reverse/forward: exactly 84 measured updates.
Warmups are separate and followed by a full start-state restore. Report baseline
differences and cumulative incremental effects separately; use no new custom
error tolerance, saved-weight equality or action-sequence equality as a gate.

## Measurement, verification and recommendation

Record host perf_counter/process CPU, CUDA event intervals, allocated/reserved
peaks and sampled PSS/RSS separately. Add opt-in labels for input packing/HtoD,
frame encoder, temporal prefix/suffix, candidate, loss, metric/checks, backward,
clip/Adam, final sync, input retirement and snapshot/save. Counts of actual frame
encodes, microbatches and transfers live only in diagnostic sidecars. CUDA event
intervals include host submission gaps and are not pure kernel durations.

Profiler runs are separate, on a small predeclared endpoint/group window;
cap each trace/output at 512MiB and bound profiler event retention by limiting
the active window. Shape capture is a separate diagnostic run. Report missing
CUDA activity/counts or exceeded bounds explicitly. Unprofiled measurements
alone select candidates. Never infer I/O wait by subtracting unrelated clocks.

After the fixed comparison select one candidate/model, favoring smaller memory
and implementation on comparable time. Run three alternating baseline/candidate
pairs/model (18 runs) in fresh `runs/learning-update-optimization-v1/`, width192,
layers2, seed17, workers4/concurrent2, games16 x updates2, epoch1, D1000000,
wall600 and unchanged limits. Run timing sequentially with no tests, audits or
agent reviews in parallel. Require healthy `incomplete/update_cap`, then perform
strict complete-original Source/numeric/receipt audits after timing. Preserve
failed attempts and do not reduce budgets to obtain acceptance.

Before actual launch root independently validates the generated plan's exact
18-row model/pair/route membership, alternating order and unique run/context/
output identities, then includes that plan's bytes in the source freeze. The
bounded population/audit utilities do not provide a general validator for
arbitrary external plans. This is accepted tooling debt: such reuse could run
a different schedule. Revisit before accepting non-root plans or removing the
mandatory root topology check and plan freeze.

Focused tests cover weighting/tails/exact endpoint membership, candidate mask,
empty events/hands and grad=None/Adam state, TBPTT>32 detach, future-frame
noninterference, endpoint-specific cards, input lifetime/checkpoint recompute,
preallocation rejection, finite failures and original FD retirement. Keep the
existing numerical/gradient/resume/ownership tests and frozen kernel checks.
Use scoped Ruff/mypy and relevant package/optional-free checks; no simulator or
dependency changes require native rebuilding. Independent correctness and final
architecture-conformance review precede root acceptance and verified commits.
Record results and dispositions in
`docs/verification/learning-update-optimization.md`. If whole-run improvement is
inconsistent, recommend retaining defaults. No learning-quality, HPO or final-test
claim follows. Experimental code maintenance is accepted debt; revisit/remove
it when a production adoption decision is proposed or this protocol is retired.

## Open evidence questions

Actual host CUDA/profiler access, admitted group sizes, real encode reductions,
performance and physical memory remain measurements. Sandbox CUDA unavailability
does not establish host unavailability; use the existing host execution approval
route when needed. Report real limitations without substituting CPU timings for
the planned CUDA comparison.

## Completed actual telemetry extension

The [accepted supplement](learning-update-telemetry-supplement.md) adds explicit
closed spec2 observation while preserving exact legacy spec1/off behavior.
One held16MiB process-local suballocation, bounded sampling/publication and
original failure/reap ownership provide actual learner/coordinator CPU/memory and
worker release spans in six separate diagnostic runs. The original18timing rows
are immutable and unpooled. All required comparison/diagnostic evidence and
strict audits pass both final independent reviews; defaults remain unchanged.
See the [verification record](../verification/learning-update-optimization.md)
for measured source cohorts, proposals and physical/generalization limits.
