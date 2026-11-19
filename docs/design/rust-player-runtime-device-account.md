# Same-host CUDA ownership and operation allowances

Status: scoped engineering implementation accepted after one repair round.
Both independent correctness/conformance rereviews are READY. Actual CUDA
capacity, complete learning/resume and deployment measurements remain Task6/7.
Independent architecture READY: 4e6e6948f132b2af6eb4b26596ca9f8ad79a4de8ecdf035b3b2bf797ca1ab347.
Owner: root. Completes the production gate left by device-planning.md. The pure
row estimate in native-cuda.md is separately accepted. No hard allocator/physical
quota is promised; configured backend allowances and explicit tensor storage
remain separate from measured allocated/reserved/PSS. No new dependency.

## Responsibility and dependency

Add gf_models/rust_player_device.py for one per-host CUDA accounting owner and
immutable operation allowances. It depends on pure model/row geometry and the
shared Torch serialization helper, never training, HostControl, HTTP or storage.
Training's actual HostAdmission remains the authority; its coordinator creates
this accounting object only after real admission and injects it into the local
learner wrapper and RustPlayerInference. The accounting object grants no Source,
process-termination, policy or GPU-namespace capability.

Move the EXISTING single serialized_torch RLock/helper from rust_player.py into
existing gf_models/torch_runtime.py, re-exporting it at the old import location.
Do not introduce another mutex, change its critical-section meaning or initialize
Torch on import. Both device accounting and inference use that same helper;
this avoids a device→inference→device import cycle. Existing callers remain valid.

Before CUDA initialization, inspect actual referenced checkpoint metadata on CPU
and bind Cshared=max(context allowances), Wshared=max(workspace allowances),
Fhost=Cshared+Wshared across admitted GPU architectures, including heterogeneous
external checkpoints. Future learner generations use the learner architecture.
Count this once as a configured HOST backend allowance; do not silently sum
per-network contexts or assume one role's smaller workspace applies to another.
CPU roles contribute no CUDA storage/context. All-CPU hosts have no CUDA
account/operation limit and omit device_plan. CPU learner with CUDA serving uses
the account for cache/row/construction and mandatory next publication only; pass
None to CPU numerical admission, record device_bytes=0/learner_fixed_bytes=0/
post_update_learner_tensor_bytes=0. CUDA learner with CPU serving uses CUDA update/
restore/export admission; CPU cache/publication contribute0 CUDA bytes and get
no fictitious GPU publication charge. Any host with at least one CUDA role emits
the update's device_plan, including CPU numerical updates with CUDA publication.
Explicit learner_device/serving_device distinguish these cases. Existing
visibility/real admission ordering remains mandatory.

## Actual state and lifetimes

Under serialized_torch, the account tracks enumerated learner CUDA tensor storage,
actual cache networks (including construction), and outstanding row-input holds.
At init/restore/update transitions enumerate CUDA parameters, buffers, gradients
and optimizer tensors, counting shared storages once; CPU Adam steps/RNG/archive
arrays are excluded. Retain owners at their real lifecycle boundaries; never
keep obsolete tensors alive merely to count them. Precompute Pi/Qi and row profile
per checkpoint; do not enumerate parameters, query allocators or rebuild profiles
per serving row. Enumerated tensor bytes are not process allocated/reserved bytes.

Cache identity is actual network storage, not equality of values or generation
labels. Learner and immutable serving copies are distinct. An alias to an already
loaded checkpoint counts once. Construction reserves its validated Pi+Qi before
CUDA allocation and becomes actual observed storage on successful insertion.
Eviction is credited only after existing consumer checks, actual removal and
settlement; an evictable entry is still live. Preserve failed/provisional owners
until positive cleanup or actual original-host exit; failure never grants health.

Inference starts a row hold BEFORE H2D. Check Fhost+learner+cache against actual
transfer storage and row estimate using that checkpoint's own geometry. Retire
all final row GPU input references INSIDE the Torch lock after successful CPU
sampling/return transfers, before claiming zero row storage. This also removes
unnecessary old/new input overlap between rows; tests must confirm it. If an
exception leaves CUDA work/references uncertain, retain actual references and the
accounted hold, fail the service/parent, and do not reuse that allowance as free.
The established native lease/RNG-acceptance/last-Tensor contracts stay unchanged.

## Immutable update allowance and separate publication check

Freeze one immutable allowance under the same lock from preflight through group/
singleton planning, parity, forward/backward, shuffle and optimizer completion.
Numerical modules receive a private optional byte limit, not HostAdmission or a
callback into orchestration. Do not mutate InferenceBindingV2, saved resume binding,
_training_reservation_bytes or the old lifetime _training_device_bytes contract.
Legacy/default callers pass None and retain current behavior.

An operation hold is an accounting/lifetime token, not an allocation of its whole
byte ceiling. Its terminal states are SETTLED after positive completion, or
FAILED_HELD on any numerical exception. FAILED_HELD retains the actual operation
owner/exception/GPU graphs and poisons the account against further update/row/
publication work. Only after actual GPU settlement, stopped-frame alias retirement
and positive nested cleanup may its funding/holds retire; the parent remains
failed. Unknown GPU/close status retains the owner until original-host retirement.
Do not release a live operation merely because an exception was caught or a
traceback field cleared, and do not count the ceiling as extra actual tensor bytes.

Update and mandatory next publication are SEPARATE phase checks. Before mutation:
1. derive update allowance from G, cache ACTUALLY retained and the fixed component;
2. prove post-update learner storage plus still-live caches and constructing next
   network fits the separately modeled publication phase.
The prospective network does not exist during update: do not subtract Pi_new+Qi_new
from update scratch. Do not credit an eviction until it really occurred or an
explicitly ordered later phase proves it. Unknown future generations/configured
pool capacity are descriptive scenarios, not unconditional admission multipliers.
Actual later operations re-evaluate their own state.

F4 means ONLY that learner's configured cuda_context_bytes + workspace_bytes.
The corresponding Model3 fixed replacement and diagnostic learner_fixed_bytes
mean exactly the same two components (or0 for CPU), never learner_group_fixed_bytes
or an execution-plan fixed_bytes field that also contains parameter/optimizer/
CPU-reference storage. All parameter/optimizer terms remain in the execution
estimate and are not subtracted as a shared fixed allowance.

For existing Model4 estimate E4 containing that F4, cache K:
Dupdate=min(original logical ceiling,G-K-(Fhost-F4)); require E4+Q<=Dupdate.
Keep the original logical/CPU-rollout check separate. Propagate the SAME immutable
limit through LocalHpoLearner.update_owned→HpoLearner.update_owned/singleton/
experimental→execution_groups/group_forward→existing execution_plan/admit_execution_plan
(device_bytes already supported). Add a distinct reviewed same-host initialization
path; do not pretend this owner is old Wire7/frozen-future-pool admission. Old
constructor-owned routes and defaults remain unchanged.

For Model3 keep constructor/admit LOGICAL checks unchanged. Add private optional
operation limit through update→parity/_parity→admit, checked BEFORE parity/shuffle/
optimizer mutation. Endpoints remain serial within logical minibatch128. Use the
ACTUAL endpoint resource_estimate+Q as an inherited conservative sufficient
estimate, replacing its configured fixed component with Fhost. Never use the full
configured maximum as a new global CUDA minimum. CPU rollout residency stays in
its original logical gate, not added as device storage.

Model3's inherited estimate contains CPU-reference costs; it is not an exact
GPU estimator. Required acceptance tests must show no false refusal for the
prescribed supported configurations. A consequential refusal attributable only
to those host costs requires a bounded device-only endpoint/optimizer/export
refinement before accepting that configuration; do not subtract unnamed host
bytes, change a binding or label the workload physically impossible. This explicit
profile gap is an open verification condition, not an invented safety margin.

Fresh Model3 staged restore is explicit tensor storage:
Fhost+K+2P+2Q+M<=G, where validated saved Adam moments M<=2P. Fresh old network
has no gradients/moments, CPU inputs copy into staged GPU parameters, no extra
whole GPU weight clone. For nonfresh callers substitute ACTUAL old learner
storage for the first P+Q; do not apply the fresh formula blindly. Original CPU
decoder budget and logical restore check remain separate. Export/initialization
and Model4 restore retain their own existing applicable phase minima, replacing
one fixed component rather than adding another; no summing exclusive scratch.

## Public planning and provenance

New-mode public validation reports known startup/load feasibility plus the runtime
phase checks, not guaranteed completion for all unknown future shapes. Maximum-row
profiles are labeled planning scenarios, not unconditional actual-row refusals.
Use existing resource/stop/failure classification at the actual boundary and keep
raw outcomes; no fabricated transition/fitness, fallback mode or budget increase
after a failed allocation. Generic/public activation remains held until the
resource planner and existing classification contracts are integrated and tested.

Record device_plan as an EMBEDDED closed object, optional only on the new Rust
update6/8 variants. No path/reference record or legacy global constant is added.
It is present iff either role is CUDA; all-CPU updates omit it (never null).
The exact object fields are
{schema_version,kind,operation_id,generation,learner_device,serving_device,
global_bytes,logical_bytes,device_bytes,host_fixed_bytes,learner_fixed_bytes,
retained_cache_tensor_bytes,post_update_learner_tensor_bytes,publication_network_bytes,recipe}.
schema_version=1, kind=rust-player-cuda-update-plan-v1,
recipe=same-host-phase-storage-v1. Devices are cpu or cuda:0. Byte values are
nonnegative integers (not bool); CUDA numerical device_bytes is positive, CPU
numerical device_bytes is0. operation_id/generation bind the actual invocation
and committed update. These are data/estimates, never a live allowance/authority.
Keep original numerical logical binding and resume identity unchanged. Different
legitimate limits may change physical grouping; retain the recipe/embedded plan
and do not claim bitwise equality across different groupings. A frozen resume
experiment records consistent producer/config/conditions.

## Verification before acceptance

TDD: known-load/heterogeneous context and cache, aliases versus distinct copies,
construction/removal overlap, B1/B>1 positive input retirement INSIDE lock, uncertain
row failure ownership, immutable allowance across a whole update, mandatory
publication checked separately, two successive legitimate different allowances,
unchanged binding/resume files, Model3 fresh staged moments, and rejection BEFORE
parity/shuffle/Adam. Verify original default numerical/owned paths unchanged.
No test fabricates Source/HostAdmission/Parent capability to authorize allocation.
Pure state tests use data; actual GPU tests use genuine admission.

Architecture review is required before dependent implementation. Root assigns
specific existing/new files only after READY. Actual Task6 continuous2/pause-resume
and approved Task7 comparisons validate the resulting profile; no new formal
campaign is created merely for the estimate. The three pre-public integration
conditions and all original plan gates remain mandatory.
