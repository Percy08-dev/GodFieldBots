# Inference performance comparison

## Scope and acceptance

Implement the approved 2026-11-16 comparison plan (attachment
`5eaa62ca-2424-4a45-89e3-5bb6f072a3c1/approved-plan.md`). This is an experimental
comparison and recommendation, not adoption of new production defaults or a
training-quality assessment. Preserve model parameters, architecture, learner
math, budgets, existing anomaly detection, policy-visible information and all
historical artifact meanings.

Compare bulk typed packing/transfers and reusable storage; candidate chunks
256/512/1024/2048; temporary alignment padding/buckets; eager/default/
reduce-overhead compilation of actual inference tensor calculations; IEEE FP32
and TF32. Screen one factor at a time on MLP, then combinations and promising
settings on GRU/Transformer. Use fixed checkpoint and input sequences for replay.
Cold process/empty compiler cache, fresh process/populated compiler cache and
warm execution are separate populations.

## Responsibilities and dependencies

`gf_models.structured_network` remains the shared mathematical implementation.
If needed, extract tensor-only private helpers from its current frame encoding
and factorized candidate calculation without changing training arithmetic.
The default path must also preserve existing construction, transfer and enqueue
order/counts, including once-per-decision positions storage. Use smaller shared
arithmetic stages or lazy operand access; moving default packing upfront or
deduplicating its transfers would contaminate the current-method control.
An explicitly experimental model-side module owns packing buffers and selectable
inference tensor callables. It imports no training, CLI, Simulator, HTTP or
management modules. Original eager behavior remains the default. Only no-grad
participant inference may use experimental storage/compilation.

`tools/` owns the comparison controller, replay input/checkpoint acquisition,
fresh-process/cache controls, optional profiler capture, actual-run composition
and aggregation. An experiment-only clean bootstrap may install the chosen
inference variant in resident actor processes before normal artifact loading;
only closed numeric execution settings and compiler-cache location enter this
bootstrap. Original viewer/FD transport, argv/environment scrubbing, model loading,
episode reset, custody, admission and failure handling must remain in use. No
inference experiment is installed in the learner. The tooling must record its
effective experimental settings separately from unchanged production identities;
experimental parents cannot be represented as ordinary training-quality evidence.

## Interfaces and data flow

Use one closed, immutable experiment options value: packing mode (original,
bulk, reusable), candidate chunk, separate candidate-row and paired
feature/weight-inner-dimension alignment, compiler mode (none,
default, reduce-overhead) and matmul precision (ieee, tf32). Validation rejects
unknown values before allocation. An explicit installer binds options to a
network instance; normal construction/configuration does not enable it.

Keep existing frame/table validation and joint resource admission in their normal
locations. Pack numeric values as float32, category/reference values as int64 and
masks as bool; preserve field layout, missingness and candidate order. Bulk
construction uses contiguous CPU storage and a bounded number of device copies.
Reusable storage belongs to one network/participant execution owner, never a
global cache or another participant. Fill the entire live view on each call;
zero padding and return only valid rows. Avoid retaining views of a buffer across
its next overwrite. Original temporal caches retain owner/revision/parameter
version validation and episode reset. On weight changes, experimental weight-
derived state must be invalidated or absent. Add explicit experimental reset and
private-empty hooks at the existing policy reset/resident verification seam:
zero/drop full-capacity participant input/activation storage before END/PARK,
including compiled retained tensors. Compiler code caches may survive; private
tensor content may not. If a compiler mode cannot meet this cleanup contract,
record it as unavailable for actual resident trials rather than weakening reset.

Padding is temporary execution storage, never a parameter/checkpoint change.
Candidate tail padding is removed before finite checks/global normalization,
sampling or result publication; event padding is masked or excluded before the
recurrent encoder. Tensor Core alignment of feature/weight operands, if tried,
uses zero extension and preserves the same unpadded model dimensions. The bounded
inner-dimension experiment pads the candidate-feature/numeric-weight pair; it
does not claim to align the separate card-projection einsum. A row-
bucket experiment must be described as such, not as proof of Tensor Core use.

Compile actual tensor-only helper callables invoked by the installed inference
path, with validation, tokenization, sampling, Python cache bookkeeping and
unsupported recurrent control outside the compiled region. Record the exact
compiled regions and intentional eager regions, graph breaks, backend graph
counts/recompilations and failures. Compilation exceptions fail that experimental
candidate explicitly; do not hide them behind an unreported eager result.
TF32 uses the installed runtime's precision API only in isolated actor/replay
processes; the learner retains its original IEEE settings. Every new network
construction must reapply the experimental actor setting after normal runtime
initialization, which resets precision.

## Resources and failure ownership

Retain the existing Source2GiB/numeric512MiB/collector2GiB/parent1GiB/save1GiB
limits, actor/model resource caps, global managed CUDA bound, worker4 and
concurrency2. Include experimental retained/temporary tensors in admission;
larger chunks/padding may reject but may not enlarge the cap. The tools-only
actual-run admission wrapper must charge a conservative fixed experimental
envelope per physical worker covering all six cached identities and transition
overlap in collection, update, save and SYNC. The same increment applies to the
actual pool's worker reservation and all preflight phase estimates. Compiler workspace
is measured as a soft physical-memory observation, not claimed bounded by a
tensor estimate. Bound compiler cache/profiler/output growth and use new output
directories. No production data or cache is overwritten or deleted. Before
launch, write an immutable comparison sidecar binding exact options, code hashes,
run/context IDs and output parent namespace; production identity is explicitly
base-only. The actual-run helper is fresh-only, refuses resume/campaign/evaluation
and requires incomplete/update_cap with exactly two committed updates. These
parents never authorize an objective or a quality claim.

The implementation uses a fixed96MiB experimental envelope per worker: six
12MiB network packing arenas, one12MiB transition arena, and12MiB for active
padding/metadata. Per-network admission checks24MiB packing-plus-temporary space
under the existing model/binding ceiling. Compiler workspace remains separately
observed. Default compilation explicitly disables CUDAGraph capture. The pinned
reduce-overhead mode enables compiler-owned captured input buffers without an
owned scrub hook, so it is compared in isolated replay and excluded from resident
trials. This is a lifecycle limit, not a performance conclusion.

Actual-run composition deliberately requires the retained local historical
`runs/hpo-producer-owned/assessment/device-io-detail-v1/{runner,measure,analyze}.py`
helpers, with hashes in the new freeze. Missing helpers fail explicitly. These
ignored local prerequisites and recorded original datasets must be preserved
with the experiment; a fresh source checkout alone does not contain them.

Existing job ownership and positive termination receipts remain authoritative.
Compiler/OOM/CUDA/worker/time-limit failures remain failures with original cleanup;
never turn an interrupted run into training/evaluation success. Replay children
are owned, waited/reaped and run sequentially. No audit, test or review workloads
run concurrently with performance measurements. Profile captures are separate
from unprofiled timing populations. Record unavailable profiler/kernel evidence
as unavailable rather than inferring Tensor Core use from a precision flag.

## Verification and experimental protocol

Add only focused implementation checks: packed values/dtypes/layout/order;
chunk and padded tail exclusion from the real candidate population and single
normalization; repeated smaller/larger input, episode reset and weight-switch
storage hygiene; compile call-path wiring and diagnostic accounting. Keep existing
tests and finite/error checks. Do not add old-output equality, sampled-action
equality, custom numerical thresholds, or tests of PyTorch/Tensor Core arithmetic.
Run the relevant existing network/policy and frozen-training-kernel checks plus
scoped Ruff/mypy. The approved focused-test scope supersedes skill suggestions
to run the whole repository suite.

Replay records checkpoint/visible-input hashes, code/Simulator/schema identities,
GPU/runtime/compiler settings, process CPU and host wall, CUDA event intervals,
copy/kernel submission counts and shapes from separate profiles, allocator/RSS
observations, first compile latency and warm graph/recompile counts. Compute
compile amortization decisions only when a measured warm per-decision saving is
positive. No finite break-even is reported otherwise. Kernel names and available
profiler evidence determine whether Tensor Core use is confirmed, ruled out or
unresolved.

Select a measured promising combination and compare with the original route in
three alternating-order pairs for each model. Keep width192/layers2, seed17,
worker4/concurrency2, games16 x updates2, epoch1, D1000000, wall600 and all prior
memory limits. Each fresh run is expected to stop normally as incomplete/update_cap;
this is not HPO completion. Report individual pairs, median differences and
trajectory/decision-count differences; same sampling seed does not promise equal
trajectories after numeric execution changes. Present disjoint whole-run phase
times and nested collection breakdown separately; do not sum concurrent or nested
intervals into wall time. Equal-speed choices favor smaller memory/implementation.

## Open evidence questions

Installed CUDA/compiler/profiler availability, useful candidate/event shapes,
compilation support for the extracted regions, physical-memory overhead and
beneficial combinations are measurements, not assumed requirements or outcomes.
TF32 learning quality belongs to subsequent HPO. Final evidence and the resulting
recommendation belong in `docs/verification/inference-performance-comparison.md`.
