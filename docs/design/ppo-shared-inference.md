# PPO resource admission and shared inference

Baseline: GodFieldBots dad129f, Simulator 91d5926a7572db327eaea3751e7c24f428120996, corrected gameplay API v2, public-ja-v3 catalog schema 2. The HPO probe uses Model4, producer-owned-v1, resident-worker-v1, collection16, concurrency3 and six participant workers. Ordinary corrected PPO uses Model3.

## Shared inference objective and contract

One inference process per training invocation, loaded from inference-only snapshots; learner state and optimizer stay on the learner. CPU participants own search, private history and sampling. The process serves compatible rows with TorchRL 0.15.0/TensorDict 0.15.2 InferenceServer and MPTransport; normalize candidates only over the requesting row, including ragged padding masks. Use one outstanding request per game and a bounded formation delay (initially 1ms). Snapshot digest, generation and request ownership are transport/control metadata. Freeze in-flight work before generation replacement; keep any weights pinned as a frozen opponent, then apply only the committed new weights using ProcessInferenceServer.update_model_weights and verify the receipt before collection resumes. Partial updates, ambiguous reply, deadline or process death fail through the owners; do not resubmit a different action. Use write-once experiment records and a distinct execution identity for a new mode. CPU actors never initialize CUDA. Normal collection and HPO collection would share model math; each still owns its checkpoint and Source authorities.

Simulator spec 07 section 21.1 at this pin prohibits delivery of another participant's view to a policy process. Explicit clarification is necessary before one process can compute over both seats' private observation-derived tensors. Reclassifying the receiver or packing rows is not a permission grant. Any required normative change is a separate Simulator task. Do not expose an unusable runtime option or adopt a dependency ahead of this decision.

## Independent diagnostics

Keep ByteBudget/StorageBudget/BorrowedBudget and ArtifactPhaseAdmission as the existing owners. On capacity refusal report scalar budget name, phase, current use, request and limit. Take the observation under the same lock as the refusal. Pass actual evaluation/source and parent-phase identities from constructors. Preserve exception classes, reason prefixes, failure propagation and original release/reap behavior. Do not add diagnostics to policy inputs. A real owner refusal and its persisted campaign error exercise the contract; ordinary success-path cost must remain negligible.

## Independent resource admission

Keep current inference/candidate semantics and frozen data readers intact. Pure role-aware planning must distinguish CPU and CUDA reservations, actual model shape/role (including MLP latest frame, GRU hidden and Transformer KV), active and parked ownership, cached identities and transition scratch. Phase estimates express what is live in collection, optimization, save and sync. Share actual ledger reservations with CLI admission; preserve full reservations while owned. Learner microbatch geometry and shape guards, native source and journal contracts remain authoritative. Any retained unmanaged interpreter/driver/allocator demand must be honestly identified and calibrated with process-tree PSS and GPU samples. Do not subtract unrelated-process samples as a guarantee, double-count live tensor storage and allocator reservation, remove a cap based on one ordinary run, or change search candidates via NumericCosts as a side effect. Separate accounting recipe for new work; existing saved experiment identities continue to mean what their producers recorded.

## Verification and handoff

Compare ordinary ML execution and HPO proposal/control identities explicitly. HPO E2E target: existing mixed16 (width192, four epochs, seed17, two updates) at 3 concurrent games in each mode; separate common 32-game schedule (16 seat-swap pairs) in the shared mode at 1/2/4/8/16 simultaneous games, two updates per point: 384 games total if all admitted. Report throughput, observed overlap and effective batch sizes, tail latency and phase-specific host/device memory; keep historical, measured and estimated values separate. Check normal Model3 GPU collection/update/new-generation inference once. Keep PPO minibatch fixed; only pursue rollout chunks if measured worker->driver transfer is actually limiting. Include large input conditions alongside sampled peaks and label unobserved maxima.

Future GRU reuses the MLP transport, snapshot and scoring path; gather/scatter hidden state by episode, participant and checkpoint. For Transformer also preserve per-row positions/masks and reserve simultaneous KV storage. Test two interleaved histories against isolated single-row inference including retired seats and changed checkpoints, then one short actual run per owner. Record the normal/HPO integration results on separate branches. Root accepts a branch only after independent correctness/conformance and relevant checks; leave it unmerged and unpublished.

## Review disposition and implementation order

Root received independent preimplementation architecture approval for the diagnostic
change before delegating edits. Rejection messages must preserve their prefix
and exception class and carry their numbers through the existing failure record.
Secondary cleanup notes keep their original lifetime and error precedence.

The full physical estimator needs separate logical execution and actual
RAM/CUDA reservations. In Model4 the existing launch capability equates the
old logical planner allowance (which includes host rollout and device cost)
with the actual CUDA learner lease. Simply lowering that lease can underreserve
the optimizer, restore or numerical update, while simply changing its validation
can alter learner grouping. Startup restore temporarily holds the initial
network and replacement network/moments. A save produces private CPU exports
and sealed FD-backed copies concurrently. Owned spec4 Source is also the parent
account for numeric and artifact phase subcharges; those caps do not describe
three independent physical allocations. Every live network, even an MLP, owns
an unpersisted position buffer. The architecture reviewer and an independent
advisor require role/phase admission of all these allocations before calling
an end-to-end physical estimator accepted. Report the old formula separately
while its execution cap and guarded callers are still in use.

The independent architecture review found that the present Simulator contract
does not permit the proposed shared policy process. Root concurs after reading
the provided Simulator docs, including §22. See [the handoff](../issues/0011-shared-inference-participant-process.md).

### Homogeneous resident training cache milestone

Scope this correction to the existing HPO training pool. Keep the original
learner/execution cap and each actor's legacy active execution envelope for
this milestone. For a homogeneous model let W be the registered FP32 parameter
bytes and P the bytes of the network's nonpersistent position table
(4 × maximum decisions × temporal width). K is the smaller of the entry limit
and the number of W-sized identities that fit the cache byte limit. Refuse
a model that cannot occupy even one slot. Above the active envelope, charge
(K−1)W + KP per physical owner. The active weight already has a reservation.
Keep cached networks charged across parking, update and save.

Postcommit SYNC and PARK are awaited serially. A refill can also happen at
BEGIN on multiple owners concurrently. Keep construction/copy allowance per
concurrent BEGIN and one at serial SYNC, including any actual transition/
old+new overlap not already in the active envelope. Where initialization
scratch lacks a smaller supported bound, retain the existing allowance.
Preflight phase totals and actual ledger ownership must agree. Output derives
a new admission recipe without rewriting historical proposals. This milestone
does not claim the mixed CPU/device execution cap has been calibrated.

Independent architecture and correctness reviews accepted the diagnostics
implementation. Root separately ran its 11 targeted cases with real local
Unix sockets using the branch interpreter in both parent and isolated child.
The preimplementation architecture review also accepted the limited cache
milestone above; final implementation review is separate.

### Model4 actor CUDA estimation (reviewed design)

Independent architecture review approved the resident-only design before implementation.
The scoped implementation passes independent correctness and conformance review.
After measuring the original and corrected-cache runtime, replace only the
Model4 inference worker's CUDA component. Keep the learner lease, logical
execution allowance, actor host reservations, legacy tensor-size checks and
NumericCosts unchanged until they have their own allocation/lifetime evidence.
A pure estimator in gf_models is called by HPO fresh/resident training and
per-role actual worker reservations. Split the host and CUDA worker values
explicitly in the derived admission object; retain the declared worst-case
over all simultaneously neural seats for training, and actual bound roles
for per-game admission. The resident cache add-on must count the active
network and position table once; evaluation, including heterogenous bindings,
must retain every permissible cache/staging allocation before using a smaller
lease. Preserve the old evaluation physical reservation until its model
inventory is accurately bounded. Experimental inference mechanisms can add
device buffers/compile storage and must retain their existing measured or
explicit additional reservation.

For the normal single-row inference route sum parameter storage, one position
table, the active temporal state (zero for MLP, one layered hidden vector for
GRU, two full preallocated KV buffers for Transformer) and the configured
workspace/context allowances. Add the maximum audited live transient for
frame, temporal step, actor scoring, global normalization/return and load.
Do not use the learner's gradient/Adam/tensor envelope for this CUDA charge.
Frame accounting covers FP32/int64 expanded inputs/one-hot conversion, hand
attention when present, event GRU inputs/gates/output at max_interval,
and all shared context/card outputs still live when scoring. A Transformer
step uses one query against the live prefix; GRU includes old/new hidden
overlap. Candidate accounting includes full feature columns, chunked reference
gathers/projection and actor intermediates, all chunk outputs through cat,
global logits/logprobs/probabilities/entropy/finite checks. Expose size and
dtype terms by scope and give each factor a concrete simultaneous tensor
basis. CUDA caching and backend workspace are estimates; do not claim a hard
physical limit. Host histories, EventBank data and Python tokenization remain
out of this device-only formula. Non-neural fresh roles retain zero CUDA.

Acceptance requires a small independent single-actor CUDA stress screen for
all three proposal models at their maximum candidate/event dimensions and
last decision position, including filled immutable cache, plus unchanged
candidate/RNG/score behavior and corresponding CLI/live-owner charges. If
actual geometry/storage cannot be accounted or lower admission cannot cover
measured occupancy, keep existing enforcement for that path.

Limit the first enforcement change to **normal homogeneous Model4 HPO
resident training**. Keep the shared policy metadata and NumericCosts of
_neural_settings intact. Fresh training, evaluation, and ordinary actors
continue through their original lease paths. Split calculation of base
components from budget refusal: the resident branch uses the old actor charge
for host RAM only and applies its own final CUDA gate using the new CUDA
component and B1 leases. The actual physical worker lease and the training
collector's reported joint admission must use the same derived device amounts,
including cache and imports. Validate real role bindings as before; select
the resident variant in the training collector from its real admitted pool,
never from a policy-owned input flag. Headroom for experimental mechanisms
must still be charged by those mechanisms. Do not reinterpret `worker_bytes`
as RAM when adding per-actor experimental allocations.

### Owned CUDA learner allowance (proposal for review)

Keep the existing combined logical execution ceiling L and all frozen format,
shape and search checks. For normal producer-owned spec4 resident CUDA
training only, compute a separate device lease D once before starting the
learner. From the operator's existing global managed CUDA budget G subtract
the full admitted physical resident pool and the larger of its concurrent
BEGIN and serial SYNC reservations (including any applicable extras). D is
the smaller of this remainder and L. The learner is created before the actors;
the calculation uses the permitted future pool, not startup occupancy. Hold
D until actual learner retirement. Report L, D, the phase minima and pool
components separately. These are managed allowances; keep reporting the
observed process/device peaks alongside them.

Before admission compare D with initialization, optimizer, export and cold
restore requirements. Each includes the configured device context/workspace
reserve. Count network W+B at creation, 4W+B with gradients and Adam moments,
and 4W+2B during cold replacement. Account step/RNG tensors by their actual
device, simultaneous optimizer temporaries (foreach: parameter-sized sqrt
list; single path: largest parameter denominators) and numerical checks
coexisting with export. Keep the old cap if an execution variant cannot
establish this lifetime bound; CPU, historical, fresh and experimental
routes retain their existing semantics.

Transmit a versioned internal launch capability from the actual training
owner: model, immutable operating limits and homogeneous pool geometry
determine the same D at CLI, parent creator and child. Parent binds D to its
actual live ledger/learner lease and the materialized config. Child checks the
closed arguments and recomputes consistency before allocating; process identity
and actual lease ownership remain the parent's proof. Do not grant authority
from an inferred PID or claimed numerical total. A new execution/admission
recipe is recorded for new invocations. Legacy config/record readers retain
their meaning; source/recipe-incompatible resume must not silently change
grouping.

Keep the planner's original E_group + host rollout <= L check. Add E_group
+ actual position buffer <= D to pure planning and final preallocation checks,
using the complete existing conservative per-group estimate until an audited
physical backward geometry is available. Apply both to aggregate and staged
groups, validate singleton capacity before shuffling or mutating optimizer,
and preserve ordering, PPO losses, logical minibatches, number of optimizer
steps and stored generation. Smaller allowed physical groups may change FP
reduction order; compare to an unchanged-budget control at established tolerance.
Do not describe D as an exact physical prediction. Host reservation remains
separate and unchanged by this device-only step. Compare budget-pair grouping,
singleton refusal, non-group minima and actual lease/worker agreement in a
small focused test. Do not rerun an HPO campaign to check learning quality.
