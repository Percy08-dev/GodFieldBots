# 0011: Shared Rust preprocessing with Python-owned learning

- Status: Accepted on 2026-10-30; user authorized implementation on 2026-10-29.
- Owner: root. Resolved boundary design independently reviewed READY for planning.
- Production design gate: independent architecture READY for scoped units;
  dependency pins and disposable owner/compatibility checks must pass first.
- Compatibility probe PASS: Maturin1.15.0, PyO3 0.30.1, rust-numpy0.30.0,
  ndarray0.17.2 on the pinned Python/NumPy/Rust runtime. Native base-owner,
  readonly/setflags rejection, pointer sharing, retained slices/leases after close,
  release after GC, detached cancellation and failure cleanup verified in
  `/tmp/gfb-native-probe`. Prototype code is not automatically production code.
- Supersedes ADR0010's bounded native feasibility scope, not its historical evidence.
- Reference: clean `42ce5e6`, Python3.14.9/GIL, NumPy2.5.4, Rust1.99.0.
- Acceptance: independent correctness and conformance READY;751Python/31Rust
  tests, installed-wheel/baseline checks, exact full-corpus/index equivalence,
  three paired preparations (2.682x), one exact reference fit and real CUDA
  two-update/pause-resume verification PASS. See
  [verification](../verification/native-preprocessing.md).

## Scope and responsibility

Move offline dataset build/inspect/index/fixed-data preparation and post-collection
PPO ingestion to a common native core. Keep live participant workers, action
generation and inference integration for a later unit. Simulator stays external
and unmodified at ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6/catalog public-ja-v2;
apply normative interface21.1 and existing pinned specifications.

Rust owns bounded reads, JSON/closed DTO parsing, actor visibility/reference
checks, journal ordering/ACK/end and Replay reconciliation, structural lineage,
source hashes, uniqueness/splits, ordered projection, fixed feature recipes,
index staging and allocation accounting. No simulator engine port/dependency.
Python owns configuration, experiment schedules/opponent selection, run ownership,
offline/online authorization, learning semantics, reward/teacher definitions,
GAE, whole-batch advantage normalization, signed-log1p/FP32 and critic selection,
models/optimizer/evaluation and final learning-derived NPZ/checkpoint output.
Neither layer can independently publish eligible training data: native structural
proof and Python reward/learning checks are both mandatory.

## Boundary and data flow

Create one NativeSession with a compact plan, settings, absolute deadline and
validated authorization. A single next_batch(retained_python_bytes,
consumer_layout) call returns a group of complete episodes: raw float64 feature
arrays, episode/decision/candidate offsets, selected indices, old logprob/value,
both-view recorded rewards and terminal facts, and compact evidence. No per-row
Python callbacks, no large Python dictionaries/list trees on the native path.
Python verifies both-view reward perspective, produces learner-only labels and
GAE, and normalizes once after the entire admitted PPO batch is valid. Do not
write labels back to Rust. Existing bounded NumPy NPZ/shard writer is the learning
output boundary. Source staging/index work remains native. Init/finalize/cancel/
close are separate lifecycle calls, not extra data round trips per batch.

Initialization has a bounded preflight/control-summary and authorization handshake.
Python reconstructs experiment schedules with the existing NumPy PCG64 code and
passes expected schedule/binding trees once per operation. Native comparison uses
the existing Python-compatible structural equality semantics after strict field
validation; original provenance hashes remain strict canonical hashes. Do not
replace that equality with a hash that distinguishes otherwise equal integer/
float schedule values. This control traffic never contains per-action input trees.

Native types have no dependency on upper Python packages or learned parameters.
PyO3 bindings expose only typed arrays, compact controls and errors. Metadata such
as GameSeed/model/generation is never added to feature arrays. Only acting-player
visible observations/events/catalog references and authorized same-revision atomic
hints reach feature kernels. SequenceOptions/rejection details remain excluded.

Preserve three existing orders: fixed samples by resolved source paths and journal
decision order; index by run/game/viewer/decision; PPO by manifest episode order
and per-trajectory decision. Preserve float64 bytes, names/signatures, transformed
float32 values, journal4/collection3-or4/dataset1/training2 and NPZ/shard identities.
Retain explicit Python reference/legacy materializing adapters for compatibility
and differential tests, never silently dispatch to them after native failure.

Linked-layout correction (independent design READY): shared Python and native
projection now require journal header episode_id to match its physical episode
directory and the unique manifest/schedule entry. Previously Python could accept
cross-directory content swaps; rejecting that inconsistent packaging is an explicit
compatibility tightening. Detached journal parsing retains its independent scope.
Viewer identity remains header-based: look up Replay by the verified viewer,
select the learner and arrange reward columns by that viewer, not the journal's
filename slot. Journal-only viewer swaps remain compatible; mismatched Replay
perspectives are rejected. All genuine frozen data remain byte-identical.

## Ownership, budgets and cancellation

NumPy arrays share native memory through rust-numpy with readonly publication.
The actual NumPy base-owner holds frozen buffers AND an Arc budget lease. Slices
keep that owner alive. Session.close retires work but does not invalidate arrays,
return their budget, or reuse their memory while views survive. Further admission
fails ResourceLimit when retained exports exhaust capacity; no forced dangling
mapping close. Zero-copy applies in-process, not automatically across learner IPC.

Keep memory_budget_mib8192/cache512/chunk64/scratch8192, file64/record8MiB,
workers1/max_in_flight1; keep measured2GiB/16MiB preset. Permit explicit1/2/4
workers under one shared budget including queued/reorder/exported allocations.
max_in_flight counts active, queued and ready-but-not-yet-delivered native work.
Completed arrays already delivered to Python remain byte-budget charged through
their base owners; they do not occupy additional processing queue slots. Explicit
parallel trials pair workers/max_in_flight as1/1,2/2,4/4 under the same byte limit.
Charge buffer/arena capacities and growth peaks BEFORE allocation, not object
graph traversal. Python counts owned NumPy allocations, not views, supplies its
retained capacity with the next data request, and declares bounded temporary
transform/teacher/GAE requirements. Native admission reserves those consumer
requirements before publishing arrays. Python must stay within that reservation.
Library-native fit heap remains separately measured. RSS/peak RSS use OS telemetry.
No per-episode deep_size in the production path and no process-wide RSS-quota claim.

Python control headroom is pre-admitted in a flat outer native lease:64*(S+G)
+32MiB before summary publication, with S its encoded length and G a checked
authorization bound from recipe counts, escaped paths and full envelope overhead.
Actual serialized authorization must fit G; underestimate fails rather than growing
after construction. At data start, constructor temporaries retire; the first
encoded proof P is admitted as64*(S+P)+32MiB before publication. Finalization admits
64*(S+2P)+32MiB for retained and final decoded proofs. This reservation survives
native Core retirement and numeric-array release until Python decodes and verifies
the final response, then an explicit lifecycle release retires it. Returned
manifests become caller-owned Python results after operation close. These are
conservative capacity reservations, not a proof of RSS bounds or exact object size.

Native decoder resource policy: native_json_depth defaults to990, strict integer
range1..990, root value depth0 and each array element/object value addsone. Excess
is ResourceLimit, not a semantic ContractError, truncation or fallback. A nondefault
setting with the Python reference backend is rejected. Record effective settings
in diagnostics/fit/PPO identity; do not modify historical source config hashes.
Semantic differential acceptance applies within both backends' resource bounds;
do not claim equivalence to CPython's call-stack-dependent recursion boundary.
This explicit resource-policy amendment has independent architecture READY.
Test parsing, canonicalization, error and Drop at boundary depths on the actual
owned executor and Rust worker stacks, not only parse success on a large main stack.
Supported deep-work entrypoints check Linux thread stack capacity/headroom before
processing; Rust workers have explicit8MiB stacks. Persistent Core retirement
uses a prestarted8MiB thread with a reserved, allocation-free one-slot handoff.
It is created before deep persistent state. Every close/error/poison/GC path takes
Core out of caller-dropped fields and transfers ownership; never drop a failed
transfer payload on a small caller stack. Explicit close joins and reports errors
before readiness; GC cancels/transfers/joins without Python callbacks or new thread
creation. Array owners remain flat and independent. This scoped design has
independent architecture READY and requires small-stack GC boundary tests.

An oversized complete episode may run alone if the remaining shared budget fits;
otherwise fail without truncating evidence. Stable output order is independent of
parallel completion. Release the GIL during native work; poll an absolute monotonic
deadline and AtomicBool cancellation within bounded parsing/work loops. A separate
cancel handle must remain callable while next_batch holds session state. Join all
owned threads before returning failed/ready evidence. Preserve existing filesystem
blocking debt; timeout never grants a new clock or permits premature publication.

Python retains run locks and grants offline authorization only for whole complete
parents. Online permission remains bound to the exact live owner/run/batch/generation;
no serialized skip-validation flag can substitute. Hold ownership through native
final source checks, learner child reap and fallible cleanup before ready publication.
Any late source/reward/deadline/worker/cleanup failure rejects the whole operation.
All six fitted models and original source runs are immutable references.

Online lifecycle clarification: finalize the batch-native source checks and
temporary resources after Python learning checks/numeric shard serialization,
BEFORE submitting the optimizer update. The existing Python run lock remains
owned, and only a synchronized successful update/save permits parent commit.
Authorized parent/generation/retention writes therefore occur after the batch
source snapshot has ended. The persistent learner child is still reaped before
whole-run completion. Offline fixed-data owners remain alive through child reap
and final source checks before ready publication, since their source parents
must remain unchanged throughout that operation.

Cancellation integration amendment: create the independent cancellation handle
BEFORE native session construction. One owned single-thread executor covers all
potentially long native initialization, authorization, data and finalization calls.
Main Python retains lock mutations and publication authority. Async cancellation
signals the handle, shields/joins the underlying work despite repeated cancellation,
and propagates the original interruption even if completion races with it. Sync
dataset waits poll at50ms to process KeyboardInterrupt. The constructor worker may
capture only its just-created native session handle for subsequent cleanup; it
may not authorize sources, mutate locks, or publish. Completed futures must not
retain exported arrays after the result has been delivered. This introduces no
additional normal data call per chunk.

Index handoff amendment: Rust owns row projection, ordering, canonical row hashes
and bounded spills. Python constructs fixed-schema Arrow batches from native
numeric/UTF8-offset buffers and invokes existing PyArrow native Parquet I/O.
No Python row dictionaries or json.loads/from_pylist loop is used. Arrow references
retain the native base leases until the write completes; reserve serializer
headroom and preserve fresh no-clobber output publication.

Optional --address-space-limit-mib is for standalone dataset CLI only, default
unset. Validate finite positive settings and current VmSize/hard RLIMIT_AS before
input work; preserve the hard limit. It bounds virtual address space, not RSS.
Do not apply it to training/PPO/library entrypoints or inherit it into CUDA children.

## Packaging and migration

Use a separate native/ distribution built with Maturin and optional native extra;
retain root uv_build and baseline without optional imports. Compatibility/pinning
is the first completion unit: prove CPython3.14/GIL + NumPy2.5.4 + owned-array views,
then pin compatible PyO3/rust-numpy/ndarray/build crates in Cargo.lock and uv.lock.
Do not silently replace the existing numerical environment or Simulator environment.
Always unset inherited VIRTUAL_ENV and explicitly target Bots or disposable venvs.
If compatibility fails, root revises before dependent production implementation.

Record loaded extension digest, Cargo/build identity, ABI and recipe identity in
fit provenance and PPO resume identity. Backend/native upgrade rejects ordinary
resume. The CUDA/CPU PPO and two fixed-LightGBM target presets select native after
acceptance. Standalone dataset commands select a native preprocessing preset
explicitly; library/default reference settings retain Python compatibility.
Missing native dependencies fail preflight, before collection/source work.

Implementation units: packaging/ownership foundation; parser+validation; feature
and offline integration; PPO ingestion; independent acceptance/performance rollout.
Root owns configuration/docs/commits, bot_implementer owns scoped code/tests.
Independent correctness and architecture-conformance reviews precede acceptance.

## Acceptance

Differential valid/invalid tests cover bool/int, big integers, nonfinite values,
null/missing, duplicate-key last-wins, Unicode/BOM/per-file decoding, canonical
float/hash spelling, identity/split conflicts, old schemas, same-policy learner
seats, interrupted/failed whole parents, source growth/replacement, cleanup/deadline.
No incidental hardening changes accepted semantics without root design review.

Verify buffer pointer sharing, readonly, surviving slices after close, lease pressure,
capacity/arena growth/reorder bounds, cancellation/panic/OOM and all-or-nothing output.
Full frozen83,783rows must reproduce both feature recipes, labels, dataset/index hashes
and legacy arrays; CPU/GPU parity staysatol=rtol1e-5. Python reference is never a
production double-validation path. Candidate lists are never regenerated.

Measure1/2/4workers on bounded representative data, then three alternating full
Python/native pairs with identical inputs/runtime/thread limits and no concurrent
training. Report startup/phases/CPU/RSS/PSS/capacities/scratch/final artifacts/I/O/
FFI and copy counts. Target >=1.5x full preparation vs current optimized Python,
not a single-function benchmark; numerical/resource regressions require disposition.
Preserve six fits, reproduce one reference fit, and run bounded real two-update
PPO/normal-boundary resume checks; no new three-parent campaign or finaltest.
Run pytest/Rust tests/ruff/mypy, installed wheels, baseline without native and
independent reviews. Never weaken validation to meet performance targets.

## Accepted consequences and follow-up conditions

The complete history preparation is2.682x faster with lower measured RSS; the
remaining largest instrumented phase is JSON/DTO decoding. NumPy numerical
exports have pointer-verified zero copies at the FFI boundary, while Rust internal
buffer growth/extensions and compatible NPZ staging still copy data. Raw native
spools increase sampled maximum scratch from0.87GiB to1.73GiB and kernel-accounted
write traffic approximately doubles. Accept this measured cost within the8GiB
scratch budget. Revisit direct buffer reuse/spooling when measured decode/copy/I/O
cost prevents the next workload's practical target; preserve exact outputs,
Python learning ownership and source validation. Live inference migration and
final-test evaluation remain separate work.
