# 0010: Budgeted source preparation and reusable public features

- Status: Proposed
- Owner: root
- Date: 2026-10-28
- Plan finalized: 2026-10-29
- Design review: independent architecture reviewer READY after concrete admission
  and pre-promotion-finalization clarifications. This is a plan, not an
  implementation or native-production acceptance.
- Implementation: authorized 2026-10-29; staged Python implementation in progress.
  The historical stopped controller PID is no longer present; never resume or
  signal a recycled PID. Remaining fits require explicit fresh launches.
- Evidence: `runs/assessments/performance-bottlenecks.json` (SHA-256
  `93908f02c970ac4146f6bd19af08056d4555ba2feacbb0dcd3b549bf4984f24e`).

## Goal and completion units

Replace repeated whole-corpus materialization with budgeted, ordered preparation
and reuse validated public context within one operation. Preserve every source
eligibility check, deterministic row/candidate order, feature values/signatures,
labels and model semantics. Native acceleration is a separately measured option,
not a prerequisite to removing duplicate work.

Measured baseline: one inspection 417.913s, including per-collection projection
156.402s, aggregate re-projection169.941s, labeled sample reconstruction67.885s,
load_run22.796s. Nested chain validation833calls/43.952s overlaps those phases.
All train journals are projected twice. Four completed fits spent98.763% of
charged time in preparation. The83,783selectedrows comprise81,523train and
2,260validation. Raw journal/Replay/control files total about3.1GB; old peak RSS
about8.1GiB. Measured filesystem input and major faults were zero on a warm run.

CPU policy samples:44candidates33.438ms select versus0.158ms forward;275candidates
271.723ms versus0.521ms. At275candidates the same public input is validated276
times and feature-name templates rebuilt277times. New worker startup measured
1.25–1.46s. These are shape-stratified diagnostics, not population percentiles.

Unit A: Python preparation/encoder improvements, bounded resources, correctness
and full-corpus equivalence evidence. Unit B: disposable Rust feasibility
experiment after A; production Rust adoption needs a separate concrete design
review and is not implied by a successful prototype. Unit C: finish outstanding
fit cells after equivalence and independent acceptance. No new PPO training or
finaltest, Simulator edit, environment repair, or process-isolation relaxation.

## Source preparation ownership and flow

`gf_training` owns a short-lived PreparationSession: source locks, shared memory
accounting, compact identity/split ledger, per-parent/collection validation
results, staging files, original absolute deadline and cleanup. It does not
expose its objects to Policy or numeric learning children. CLI supplies closed
settings and concrete paths. Model encoders receive only participant-safe input.

The logical unit is a complete episode (both journals and both Player Replays),
not an arbitrary slice that could lose closure/acknowledgement validation.
Read/validate each original once per operation; hash the bytes actually read.
Reuse collection manifest/report parsing and the parent chain under the same
owned lock. Three parents require three initial chain validations, not833.
Cache only immutable evidence bound to exact hashes/versions and live ownership.
Recheck mutable parent/latest/registry identities before final publication/use;
an invalidated/changed parent or source fails the whole preparation.

Produce validated episode evidence and typed accepted decisions directly.
Keep both-view validation mandatory, but retain/encode only declared learner
decisions for fixed-data fit/index. Avoid input->JSON string->parse->revalidate
roundtrips. A shared evidence scanner serves dataset build/inspect/index and
fixed-data preparation; legacy journal/export APIs retain their documented shapes
through adapters. PPO online authorization remains a distinct exact live token;
offline defaults still require whole-parent completion.

Maintain compact global uniqueness, seed/ancestry and declared-split information
across chunks. Compare each collection proof with its immutable update and
produce the same globally ordered source proof. Source metadata hashing may be
incremental, but must reproduce existing canonical digests exactly. No second
journal pass is needed solely to merge/check already validated evidence.
Preserve old sorted-source and per-journal decision ordering, regardless of
chunk sizes or completion order. The existing fixed dataset content digest must
remain `ab5d4913fda45795d1cd5aa02a4a9c4c3464268298c4d3ea49d56d4909df770f`.

Each bounded group of episodes is validated, transformed/written to typed numeric
staging, then released. Do not keep a whole corpus of Python dicts and JSON
strings. A preparation becomes usable only after all sources/global checks and
final identity checks succeed. An early valid chunk never authorizes a fit if a
later chunk fails. Resource exhaustion invalidates this preparation, not a
previously healthy source parent. Partial staging remains unready and is cleaned
under the owning operation; no inferred terminal labels or partial-data salvage.

## Config, memory and temporary storage

Add an experiment-level stdlib-only closed `preprocessing` table. A schema1,
kind=preprocessing preset can be referenced by experiments and supplied to
dataset commands through `--preprocessing-config`; absence uses the same defaults.
Dataset commands read that standalone preset without loading a server/model
experiment. Existing experiment `--set preprocessing.FIELD=...` precedence and
origin-relative path rules apply. Never inject these new defaults into stored
source effective_config objects or recalculate historical config hashes.
Record effective settings/backend and measured scopes in experiment diagnostics.

| Setting | Initial default | Meaning |
| --- | --- | --- |
| memory_budget_mib |8192| Shared admission budget for preprocessing-owned buffers, retained parsed data, metadata, numeric chunks and caches |
| cache_budget_mib |512| Cache sublimit inside memory_budget, never an extra allocation allowance |
| chunk_target_mib |64| Target accounted working set; scheduling granularity, not learning minibatch size |
| max_source_file_mib |64| Pre-decode file bound, configurable; oversize fails explicitly |
| max_record_mib |8| JSONL record bound checked before decode |
| max_in_flight |1| Active plus queued chunks; one shared budget, not multiplied per worker |
| workers |1| Python reference is serial; other counts rejected until a supported backend exists |
| scratch_budget_mib |8192| Owned temporary numeric/index space; checked before admission and publication |
| wall_seconds |7200| Dataset command deadline from command start; fixed fits retain their original training deadline |
| cleanup_seconds |5| Preparation cleanup reserve; fixed fits reserve the larger of this and existing cleanup within their original budget |
| backend |python| Explicit selection; no silent runtime fallback |

Reject unknown keys, bool-as-int, nonpositive/inconsistent limits and unsupported
backend/worker settings before source work. Budget accounting includes queue and
retained leases, records peaks and uses conservative headroom for Python decoding
and copies. Python object accounting is an admission/soft-memory policy; it is
not an OS hard RSS quota. Record actual RSS separately. Single oversized files,
records or episodes fail as resource limits rather than silently exceeding the
budget or truncating data. Compact global metadata also consumes this budget;
if it alone cannot fit, fail explicitly instead of unbounded accumulation.
Exception: cache_budget_mib=0 disables optional caches. Essential live validation
and global-proof metadata is charged as working memory and cannot be evicted to
skip a check. Require cache_budget<=memory_budget, positive chunk_target<=memory,
max_record<=max_source_file and cleanup_seconds<wall_seconds.

Pin the Python accounting policy: before episode decoding reserve
`64 * total_bounded_raw_file_bytes + 32 MiB`, including both views and decoder
headroom. This is a conservative admission estimate, not a proven hard RSS
bound. Check actual read lengths against file/record caps and acquire any
additional reservation before growing buffers. Deep-size retained parsed Python
objects once per episode (deduplicate shared identities); charge arrays and
mapping windows by owned capacity/nbytes, plus retained metadata/cache/queues.
Replace the estimate with actual managed charges after parsing; an unexpected
overrun fails this operation before more admission. Record RSS separately.
An episode exceeding the chunk target may run alone only if its reservation
fits the shared remaining budget. Evict optional cache first; otherwise report
ResourceLimit, never truncate or bypass checks. The current largest journal is
2,458,246 bytes and largest Replay382,121 bytes, below the proposed file bounds.

When full numeric matrices do not fit the managed resident allocation, write
ordered typed `.npy`/memmap staging in an invocation-owned scratch directory;
use a uniquely named directory under the system temporary directory by default,
never inside a source parent. Record its ownership and cleanup status.
Use fixed counts/shapes from validated evidence and verify actual counts before
sealing. Mapping does not guarantee zero RSS or that LightGBM will avoid copies.
Native fit memory remains separately measured; do not promise a global process
memory cap. Do not replace full LightGBM fitting with per-chunk incremental fits.
The existing numeric-child input shapes/hashes and whole-batch PPO semantics
remain. Any required serialization copy is budgeted and reported.

Scratch files have no cross-command authority, are not a persistent validation
cache, and are never used to skip original-source checks on a later invocation.
All fit preparations revalidate their sources. Time spent in preparation,
staging, native work and cleanup stays inside the original per-fit7200s; deadlines
are never restarted per chunk. Existing filesystem/OS blocking debt remains.

PreparedInputs stays owned until the numeric runner's pre-promotion finalizer.
The runner alone borrows its array/mapping views. After native child reap, log
drain and unready artifact verification, it drops those views, rechecks source
identities, closes mappings, deletes scratch, releases caches/locks, and checks
deadline/cleanup health before publishing complete training and ready inference.
Any finalizer error enters existing run/artifact invalidation; no success survives
cleanup failure. Context exit after successful explicit finalization is an
idempotent no-op, not a second fallible cleanup after readiness. Exception paths
abort/clean without replacing the original error or granting a new deadline.

Reserve simultaneous disk peaks before creation: typed staging plus destination
legacy numeric NPZ, headers/proof ledger and any final staged output. Use bounded
mapping windows and bounded copying/ZIP writes; preserve legacy numeric container
layout as well as dtype/shape/ordered array bytes. If exact NPZ identity cannot be
preserved, fail the equivalence gate and return to root rather than silently
changing compatibility. Native fit may allocate/copy its own full dataset; this
preprocessing budget does not claim to constrain that native heap.

For dataset build/index, close all fallible preparation resources and verify
identities before atomically publishing the final output; inspect returns success
only after the same finalization. Final staged output remains owned/accounted
until rename and post-write deadline checks. Finite dataset deadlines start at
command entry. Fits keep their original7200s clock and include all preparation,
staging, native and finalization time.

Hash original bytes during their bounded read, checking file identity/size/time
before and after reading. Before finalization recheck covered file inventories/
identities and mutable parent/latest/registry hashes under the held locks. These
guards supplement, rather than replace, the original content hashes. No claim
of protection against hostile filesystem writers bypassing ownership is added.

## Public feature encoding

Keep scalar encode compatibility and add an internal prepared-decision path.
Validate config and the exact participant input once, then compute common
observation/hand/hint/history features once and candidate-specific features in
stored order. Cache immutable feature-name/schema templates by validated typed
config/version. Validate before a cache lookup so bool/int equality cannot bypass
closed config checks. No cache keyed only by revision, viewer number or policy ID.
Decision context lifetime ends with that decision; no private context crosses
participants, games or workers. Public definition caches, if needed, are bounded
and only queried for the input's permitted own references.
PreparedDecision owns an isolated validated snapshot and precomputed immutable
values; the frozen DecisionInput dataclass alone is insufficient because its
dicts are mutable. Later caller mutation must neither change cached features nor
bypass validation. Numeric outputs have explicit ownership/read-only views.

Preserve missingness, perceived definitions, sequence multiplicity/order, reward
perspective and critic independence. Keep numerical operation order/types where
needed for bit-identical float64 features and existing CPU signed-log1p float32
MLP inputs. No native log1p substitution, action regeneration, change to generator
v3/scorer/encoder signatures, candidate limits, PPO minibatch/microbatch settings,
CPU inference placement or GPU ownership. Fresh participant interpreters remain;
worker pooling/fork preload and IPC redesign are deferred despite measured
startup cost because they change isolation/ownership boundaries.

## Interfaces, compatibility and provenance

New internal types are PreparationConfig/Session, validated episode evidence,
typed accepted decision and context-managed PreparedInputs. PreparedInputs owns
arrays or mappings plus their evidence/cleanup; no raw-pointer IPC. Production
dataset commands and fixed-data training use the streaming path rather than the
old tuple-of-all-Samples materialization. Global validation and canonical proof
construction remain shared, not copied into independent validators.

Keep journal4, collection3/4, fixed dataset1, LightGBM training2 and inference
schemas unchanged. Add versioned preprocessing diagnostics within parent control
provenance, not inference features. Include actual shared encoder/validator and
new preparation module hashes in fit-code provenance. Existing dataset source,
content and numeric-input hashes must match; no historical artifact relabeling.
Index row/content identity must match; Parquet layout-byte hashes may differ
across chunk layouts and are not authority. Same settings must regenerate a
deterministic index. No persistent cross-fit cache in Unit A.
Preserve two distinct sort orders: resolved source paths then participant
decision order for canonical source proofs/numeric sample order, and
(run_id,game_id,viewer,decision) for index rows. The index uses a charged compact
sort ledger; when that ledger exceeds its budget, spill sorted runs and bounded
k-way merge under the scratch limit. Do not reproject source data for sorting.

## Rust feasibility experiment and decision

After removing repeated work, profile again. If decoding/validation/feature
extraction still accounts for at least25% of preparation and at least30seconds
on the full frozen corpus, perform a bounded
PyO3/maturin prototype for that measured hot path. Keep the root uv_build package
and baseline unchanged; use a separate disposable native project/environment,
pin tested build/crate versions in its lockfiles. Target current Linux/WSL,
CPython3.14 with GIL. Do not copy/depend on the Simulator engine.
This prototype targets offline preparation; participant inference remains the
optimized Python encoder. Installing an extension must not silently select it.

Prototype batches own Rust buffers and work without Python object callbacks
inside detached parallel regions. Test1/2/4workers under a shared memory budget
and bounded in-flight/reorder queue. JSON still requires parsing/allocation;
measure conversion overhead. Return typed buffers and compact diagnostics,
not huge Python dict/list trees. NumPy/Arrow-style zero-copy is considered only
within one process with explicit owners, readonly views and release lifetimes;
do not pass pointers between isolated processes or promise LightGBM zero-copy.

Test the exact rejection/visibility/type semantics (including duplicate keys,
bool/int, integer bounds, finite floats, null/missing, Unicode and malformed
input). Python reference remains authoritative for compatibility. Keep floats
and canonical metadata hashing in a path proven equivalent; preserve current
duplicate-object-key last-wins decoding. Stricter duplicate rejection is a
separate semantic change, not part of this optimization. Require identical
numeric arrays/order and source proofs. Start with <=10,000 representative
decisions; run the full fixed corpus only after differential tests pass.

Native production adoption requires: median end-to-end preparation improvement
>=1.5x versus optimized Python on three identical warm-cache trials, no memory
budget regression, and all semantic/ownership/error tests passing. Otherwise
retain Python and document the result. A passed prototype triggers a separate
concrete production design/dependency review; it is not automatically promoted.

## Verification and experiment continuation

### Required measurement protocol (2026-10-29 amendment)

Preserve the old implementation and use identical frozen inputs, seed, settings,
dependency environment, thread counts and machine. No concurrent training during
comparisons. Record code/input/config/output hashes, machine and runtime identity.
Separate detailed profiling from timing trials. After warmup run three trials per
variant, alternating old/new order between pairs; report median and min/max.
Measure first startup separately. Do not flush the OS cache or claim a cold cache
without evidence. Keep the old four fits separate from paired speed benchmarks.

Measure monotonic total time and disjoint startup/read-hash/validation/encoding/
numeric-staging/native-fit/save/final-check-cleanup phases; overlapping nested
spans are diagnostic only. Report accepted actions, candidates and input bytes,
actions/s and MiB/s. Distinguish policy selection p50/p95, NN forward, and total
collection/update/save wall time per accepted action; stratify by candidate count.

For parent and owned children collect user/system CPU seconds (including exit
accounting), aggregate CPU seconds, mean occupied cores and utilization timeline.
Distinguish one-core=100% from whole-machine percent. Sample resources at100ms;
sample PSS and directory sizes at1s. Report simultaneous summed RSS peak (shared
pages may be counted twice), PSS where available, managed/cache peaks, page faults
and swap. Sampling can miss short peaks; allocation-driven budget counters remain
separate. Record logical read/write bytes, OS-accounted I/O, scratch logical and
allocated peaks, final artifacts and post-cleanup residual bytes. GPU metrics are
utilization, device VRAM and Torch allocated/reserved peaks, with synchronized
update timings. Missing metrics are unavailable, never zero. Distinguish process
I/O accounting from physical-device traffic and WSL host visibility.

Store versioned JSON measurements plus a human comparison table. Measure monitor
overhead using a small with/without paired workload. Monitor diagnostics never
enter model features or frozen dataset hashes. Compare full83,783-row old/new
preparation at standard settings and new Python at memory2/8GiB crossed with
chunk16/64MiB, cache512MiB, three trials each. Record insufficient-budget failures
as failures, not partial successes. Measure fixed44/275-candidate inference and
separately replay stored inputs for selection distributions. Report CPU seconds,
actual memory and scratch changes alongside speed. Native trials include boundary
conversion, transfer and cleanup, with1/2/4workers under the same global limits.

Numeric mismatch, weakened checks or leaked resources fail acceptance. Targets
remain1.5x full preparation and2x fixed-case selection. Missed targets or resource
regressions require evidence, re-profiling and a recorded root disposition before
any success claim. The complete metric specification is an acceptance requirement,
not a claim that historical measurements already contain every metric.

The amended protocol received independent architecture review READY. Track owned
processes by PID plus start identity and capture final child exit accounting.
The observer must not import Torch into the coordinator or initialize CUDA in a
CPU inference worker. GPU allocator peaks come only from the existing GPU owner;
CPU-only workloads report GPU metrics as not applicable. Do not launch new live
PPO runs just to populate timing columns. The old baseline has no managed-budget
settings: explicitly record that unsupported scope rather than claiming it
obeys the new accounting limits. A metadata-only summary sink and numeric sink
must use the same scanner/session; CLI inspect and fits never call the legacy
materializing tuple adapter.

Keep allocation counters always enabled; disable only optional sampling for the
overhead comparison. Bound or stream timeline records under the same metadata/
scratch budget. Scan scratch/output directories, not immutable source trees,
for sampled disk sizes; count source bytes during ingestion. Reconcile final
per-process counters with samples rather than adding both. Device GPU readings
are device-scoped; allocator peaks are owner-scoped. Per-definition encoder
reuse keys refer to the isolated snapshot entries, not merely definition_id.

Archive current reader/encoder and completed fit artifacts as reference. Test
small budgets/chunks and default budgets, boundaries crossing episodes/splits,
duplicate identities/ancestry conflicts, source tampering/parent invalidation,
oversize/malformed inputs, cancellation/deadline and scratch cleanup, and denied
online-context reuse. Verify caching does not skip failed-parent or missing-proof
checks. Count chain validations per operation (3 for current corpus) and each
source's single semantic pass, not just compare elapsed time.

On the frozen corpus require exact source/content digest, sample identities,
targets, feature order/float64 bytes and stored MLPPPO float32 inputs. Compare
representative all-candidate PPO decisions against the archived encoder and
saved rollout arrays; preserve CPU/GPU logprob/value tolerance1e-5 and unchanged
sampling/RNG behavior. Cover sparse/missing/Dream/hints and config-cache hazards.
Run pytest/Ruff/mypy, baseline/native installed wheels and independent correctness
plus architecture-conformance review before accepting production changes.

Benchmark old versus Python-new with identical full corpus/config, bounded
sequential trials, startup included/excluded scopes, throughput, preparation
phase times, RSS/cache/scratch peaks and CPU/GPU utilization. Initial targets:
>=1.5x full preparation and >=2x CPU select on the fixed44/275candidate cases,
without numeric or memory regressions. Unmet targets require re-profiling and a
root decision, not weakened validation or a claimed successful acceleration.

User-confirmed continuation (2026-10-29): retain four completed fits;
verify exact old numeric-input hashes and replay one history/seed17 reference fit
after optimization, then run only remaining history29/43 in new directories.
The reference repeat is an equivalence benchmark, not an extra comparison cell.
Do not combine old/new preprocessing times into a homogeneous timing comparison;
use paired benchmarks for speed claims. Keep dataset and all three healthy PPO
parents unchanged; no rerun of long training. Never SIGCONT the old outer
controller after code changes: verify no active child, record completed/pending
work, stop it, and explicitly launch the remaining cells. No finaltest.
The user explicitly selected retaining the four fits and completing the two
remaining cells after output equivalence; speed comparison uses separate paired
benchmarks. Before signaling any saved controller PID, verify its current
command/process identity and child state; never act on a stale PID alone.
