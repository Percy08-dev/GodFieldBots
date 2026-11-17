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
