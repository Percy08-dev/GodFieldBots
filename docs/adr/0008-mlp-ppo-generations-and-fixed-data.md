# 0008: Owned MLP PPO training, generation lineage and fixed-data comparison

- Status: Accepted
- Date: 2026-10-28
- Decision owner: root
- Review: `bot_architecture_reviewer` design READY and Stage A dependency
  conformance READY, 2026-10-28; numerical recipe, role/qualification and pause
  clarifications below resolve all design findings. Final correctness and
  architecture conformance READY, including the bounded provenance delta.
- Implementation: complete; component checks passed, real experiment acceptance
  tracked separately in `docs/verification/mlp-ppo.md`.

## Requirements and scope

Implement the user's CUDA MLP PPO plan: actual Simulator learning, immutable
generation/batch/update lineage, bounded explicit pause/resume, and fixed-data
LightGBM experiments from three wholly successful PPO runs. GRU, Transformer,
AMP, TF32, external flash-attn, final test evaluation and simulator edits are
excluded. Existing user changes installed Torch 2.14.1+cu132/torchvision
0.29.1+cu132; preserve that installation while making CPU and CUDA optional,
mutually exclusive extras. NumPy is pinned to 2.5.4 for native compatibility.
Baseline imports require neither Torch, LightGBM nor PyArrow.

Simulator revision, the eight reviewed specification hashes, journal/derived
schema 4, ActionOptions scope, reward perspective and failure distinctions stay
as ADR0007. The current normative §21.1 has been read: only acting-player
Observation, visible Events, typed metadata, optional same-revision atomic
ActionOptions and pinned own-visible catalog context enter policy features.
No SequenceOptions, rejected-action details, opponent private view, RNG/GameSeed,
generation or opponent identity enters the policy. Failed parent PPO runs make
all their offline data ineligible, including previously healthy batches.

## Ownership and dependency direction

- Root owns project dependencies, presets, design/evidence documents and commits.
  One implementer owns Python and tests. Independent reviewers are read-only.
- Move pure encoder/configuration primitives to model-neutral `gf_models`
  modules, with compatibility re-exports at existing LightGBM paths. Preserve
  exact LightGBM feature names/order/values/signature and existing defaults.
  Model-specific settings and training dispatch are closed and validated before
  external I/O; MLP never imports LightGBM-specific implementation.
- `gf_models.mlp` owns CPU inference, actor/critic, signed-log1p FP32 encoding,
  inference metadata and safe typed-array artifacts. Import Torch lazily in
  numerical/inference children only. The critic uses explicitly selected
  observation/history/hint features independent of candidate contents/list.
- `gf_training` owns pure PPO targets, numeric update child, parent orchestration,
  lineage/checkpoints/retention and dataset validation. It accepts a collection
  callable rather than importing CLI or HTTP/environment implementation.
- CLI composes a `CollectionPlan -> FinalizedCollection` callback using existing
  `Environment.run_group`, journal, replay reconciliation and finalization.
  The callback accepts precomputed groups, frozen bindings, output location,
  parent/batch/generation identities and the parent's absolute deadlines.
  It returns a finalized collection path and healthy result evidence. It must
  not reset deadlines or reject queued groups from an already admitted batch
  just because the 90-minute batch-admission cutoff has passed.
- One clean owned learning subprocess holds Torch, optimizer and GPU for a PPO
  run; CPU participant inference remains one isolated process per participant.
  Coordinator does not import Torch. IPC carries closed JSON and hashes/paths
  to CPU NumPy shards, never device tensors, handles, source observations or
  management features. The child's minimum environment sets cuBLAS workspace
  configuration before importing Torch; errors drain/reap and fail the run.

## Interfaces and data flow

`EpisodeSpec.learner_viewer: int | None` records the learner seat explicitly,
including generation-0 self-play with identical policy IDs. New schedule schema
serializes it; schema-3 schedule serialization omits the new field, preserving
the fixed historical pilot proof without guessing roles. Episode statistics
use the actual opposing binding, not a label inferred from the learner ID.

An admitted PPO batch freezes both policies before collecting eight distinct
GameSeed pairs with swapped seats: two Random pairs, two heuristic pairs and
four snapshot pairs. Snapshot choices come from the latest five periodic
snapshots (generation 0 initially, then multiples of ten). Persist both views;
only the declared learner's accepted actions enter PPO or offline train labels.
Train GameSeeds use UUIDv5 in a dedicated versioned train namespace; validation
retains the existing 32 seeds; reserve a distinct test namespace and do not run
test. Participant sampling seeds derive from learning/schedule RNG, not GameSeed.

After all 16 natural, healthy, replay-reconciled games finalize, ingestion
revalidates journals, linked manifests/report/Replay and exact current parent,
batch and frozen generation. Reuse stored candidate contents/order, selected
index, logprob and value; never regenerate candidates. Natural final reward is
learner-perspective +1/-1/0, other rewards zero. GAE uses gamma=1, lambda=.95,
zero terminal bootstrap, value target old_value+GAE; normalize advantages once
over the whole accepted batch. No interrupted trajectory is bootstrapped.

Actor scores each situation/candidate with two 128-wide ReLU layers; critic
scores situation only with the same hidden widths. CPU orthogonal init uses
sqrt(2) hidden, .01 actor output, 1 value output, all biases zero. Both learning
and evaluation sample categorically. Adam: lr=3e-4, betas=(.9,.999), eps=1e-8.
PPO: clip=.2, value=.5, entropy=.01, four epochs, logical minibatch 128,
gradient norm .5. Precision is deterministic IEEE FP32; CUDA device must be
available before collection. CPU execution requires an explicit CPU preset.

Before the first optimizer step of every batch, recompute saved selected
logprobs/values on frozen CPU weights and compare every valid candidate's
logprob/value on CPU and configured learner device (atol=rtol=1e-5). Validate
the entire batch before any update; mismatch fails without repairing evidence.
Numeric inputs are sharded at <=64 MiB. Split logical minibatches into padded
microbatches <=4096 candidate slots; weight loss by logical minibatch decision
count, accumulate gradients, then clip and Adam-step once. Shuffling uses an
owned persisted RNG. Synchronize CUDA before validating metrics, saving CPU
arrays and acknowledging an update; OOM/asynchronous errors cause no success
commit, fallback, changed batch size or automatic retry.

## Persistence, compatibility and commit ordering

New collection manifests use schema 4 with an explicit collection kind and
closed PPO linkage (parent run ID/path, batch ID, collection generation and
learner seats). Schema 3 remains an explicit legacy format; its schedules and
pilot validation retain original semantics. Participant journal/derived schema
4 remains unchanged. Online PPO may consume its own complete batch while the
parent is running; offline ingestion additionally requires the whole parent
run to be successfully complete and all immutable links/hashes consistent.

Use immutable hash-linked generation and update records. Mutable run state,
latest pointer, artifact locations/deletion receipts and derived indexes are
separate. Generation 0 is initialized weights. Advance exactly one generation
only after collection, update, CPU-array checkpoint save and verification have
succeeded. Persist checkpoint first, immutable update/generation record second,
latest/state last; incomplete commits fail closed on resume/offline ingestion.
Record generation parent, weights/inference-manifest hashes, accepted cumulative
decisions, optimizer steps, input hashes, metrics, policy/opponent/seat identities,
planned/actual results, health and costs. Keep all generation records and
inference manifests, even after nonperiodic tensor deletion.

Checkpoints are closed JSON plus typed NumPy arrays loaded with allow_pickle=False;
validate keys, dtypes, shapes, finiteness, sizes and hashes before use. They
include weights, Adam tensors/counters, CPU/CUDA RNG, shuffle/opponent/schedule
RNG, collection position, pool and counters. Keep generation 0, every tenth and
latest complete inference/resume artifacts. Delete a nonperiodic predecessor
only after successor commitment and no live reference; record intended deletion.
CPU inference after CUDA training is verified on the same cu132 build. Moving
normally between +cpu and +cu132 checkpoint runtimes is unsupported initially.

`train --mode fresh|resume --run-dir ... --pause-after-update N` supports explicit
normal batch-boundary pause. A coordinator-lifetime flock protects the directory,
including eligibility checks, resume, state transitions and retention. Resume
requires paused state, complete last commit, identical code/environment/config/
backend/numerical settings and remaining original wall-clock deadline. Reject
clock rollback, expired deadline, second owner or automatic recovery of failed,
running or complete runs. Never extend original deadlines while paused.

Run caps: 7200s original wall, 5400s new-batch admission, 7140s execution,
1000 updates, 10 GiB owned bytes with 1 GiB admission reservation. At pre-batch
caps, >=1 update plus healthy completed work permits successful completion.
Zero updates is ineligible. Admitted waiting groups may run until execution
deadline; final 60s is for cleanup/save. Any collection/update/persistence/
cleanup failure, actual limit overrun or execution expiry fails the whole run.
Capacity is a measured soft limit, not an OS hard quota. Existing filesystem
atomic-write/dual-invalidation debt remains; no claim of hard real-time storage.

## Fixed dataset and LightGBM comparison

`dataset build` validates an explicitly frozen ordered source set: exactly the
three prescribed completed PPO parents (seeds 17,29,43), all their learner train
trajectories, and 64 validation games per final policy (32 swapped pairs, first
16 versus Random, next 16 versus heuristic). No outcome-based selection. Store
source hashes, selection rule, all generation ranges, split/namespace, counts,
content hash and deduplicated generation cost in a canonical dataset manifest.
`dataset inspect` revalidates the originals, parent status, immutable chains,
Replay/journal linkage and cross-source train/validation seed/ancestry unions;
never trust an index or eligibility flag. Optional Parquet indexes are rebuildable.

LightGBM consumes only actually accepted actions and natural final participant
returns; neither PPO value nor advantage is a target. Train observation-only
and public-history aggregate recipes, each seeds 17/29/43, on exactly the same
frozen selected source set. New fixed-data cost mode uses training-record schema
2, includes validation/feature transform/fit/save/cleanup in the per-fit 7200s,
reports shared data generation separately, deduplicates it globally, and makes
no equal-total-cost PPO comparison claim. Preserve schema-1 inclusive generation
cost mode and its guarded source/decision-budget checks.

## Verification, alternatives and open evidence

Stages A-E follow the user's acceptance table: dependency isolation and native
regression; old pilot/role/schema/chain/retention tests; hand-computed probability,
GAE/loss/gradients/permutation/critic independence/microbatch equivalence;
full-batch CPU/GPU parity and NPZ restore, fail-closed device/errors and exact
boundary resume; real two-update smoke, sequential three-seed runs, validation,
frozen dataset and six LightGBM fits. Independent correctness and architecture
conformance checks precede commits at coherent milestones. Run pytest, Ruff,
mypy, opt-in Simulator tests and isolated installed-wheel baseline/CPU/CUDA
checks. Report NOT RUN explicitly for unavailable evidence. Training success
does not imply improved strength, speed or final-test quality.

Record end-to-end collect/update/save timing, inference p50/p95, throughput,
model/storage sizes, CPU memory and CUDA allocated/reserved peaks. Record actual
runtime/code/simulator/spec identities and numeric settings. Source HEAD does
not prove prebuilt binary identity or deterministic training across environments.

Alternatives rejected: GPU inference per participant duplicates GPU ownership;
pickle checkpoints hide executable state; harvesting failed runs weakens the
whole-run acceptance rule; deriving seats from policy IDs fails self-play;
moving budgets on resume breaks experiment comparability. No generic trainer
framework, engine copy or speculative recurrent adapter is needed.

Open evidence: GPU availability after dependency sync and practical CPU/GPU
parity at 1e-5 must be demonstrated on the actual target. Failure stops CUDA
acceptance; it does not authorize relaxing tolerance or silently using CPU.

## Root-resolved numerical and sampling details

The plan leaves these routine choices open; v1 pins them rather than implying
that the user specified them. MLP uses `participant-action-v1` features with
public_aggregate history, window 128, 32 hash buckets and three element slots.
Actor uses the entire common vector; critic selects every feature except the
`action.`, `target.` and `element.` prefixes. Selection by feature names avoids
depending on a guessed contiguous state slice; history remains state input.
Apply signed log1p on CPU before float32 conversion to both vectors.

Select snapshots uniformly with replacement, once per snapshot pair, from the
last five periodic snapshots; both seats use that same opponent. Parent PCG64
owns opponent/schedule randomness separately from GameSeed namespace generation.
Policy stream seeds hash learning seed, batch/pair index and policy role, and
stay attached to that role across seat swaps, as the existing scheduler does.
Train UUIDv5 names use learning seed/batch/pair under a fixed versioned namespace;
validation seeds are the existing 32 and test has a separate namespace.

Value loss is unclipped mean squared error. Total loss is negative clipped
surrogate mean + .5*MSE - .01*entropy mean. Approximate KL is the mean selected
old_logprob-new_logprob. Advantage normalization uses population std (ddof=0)
plus 1e-8, once per whole batch. Last partial logical minibatch uses its actual
decision count. No value clipping, target-KL early stop or hidden reward shaping.

Dataset labels select the explicit learner_viewer in both train batches and
final-policy validation collections. Both views remain mandatory original
evidence and are revalidated, but baseline/opponent actions are not fit labels.

General formal MLP bindings require a healthy completed parent and its final
committed generation/hash. Owned PPO collection uses a separate explicit binding
construction under the coordinator's lifetime lock for already committed current
and periodic pool generations, including generation 0. It does not globally
relax the formal qualifier to accept running/paused parents. Metadata/hash
inspection in the coordinator plus native initialization in clean participant
workers precedes listeners. Inference manifests carry only public model recipe,
runtime/feature/contract identities and tensor hashes; parent run identity,
generation, eligibility, deadlines and other controls live in parent records.

`--pause-after-update N` is an absolute committed-update target and must exceed
the current count. It is an invocation control, excluded from immutable training
config identity, so resume may choose a later boundary. The learner child must
be drained/reaped successfully before publishing paused/complete state or
releasing the run lock. Cleanup failure makes the parent failed instead.

## Measurement interface clarification

Measure participant inference around `policy.select` after candidate generation
inside the clean worker. Send a finite nonnegative `inference_elapsed_ms` as
diagnostic response metadata, store it on the participant worker, copy it to
EpisodeResult diagnostics and aggregate report p50/p95. It is never part of
PolicyDecision, DecisionInput, journal schema 4 or learning feature arrays.
Existing `selection_elapsed_ms` measures the whole worker call and retains that
meaning. Use explicit scope labels for CPU RSS (coordinator/learner/participant)
and end-to-end batch time. Metrics unavailable on a platform remain null rather
than being inferred from unrelated search timings. No new dependency is needed.

Accepted memory debt: fixed-data validation currently materializes projected
rows/selected samples. The actual host has 31 GiB RAM; the measured historical
pilot has 47.6 MB of journals versus an estimated 1.062 GB of equivalent actor
arrays, so the 10 GiB per-parent owned-storage cap is substantially dominated
by numeric data. A streaming framework is not justified by this evidence.
Measure coordinator/learner RSS during the real smoke and dataset ingestion;
revisit before full fitting if pressure approaches available RAM, swapping
begins, or actual source ratios depart materially from this measurement. The
smallest follow-up is per-collection validation with lightweight global split/
ancestry metadata and streamed selected samples, preserving cross-source unions.
Architecture reviewer accepted this bounded debt on 2026-10-28.

Final identity clarification (independent design READY): freeze binary/server
TOML bytes and source/spec snapshots in parent identity and bind every collection
to them. Record installed relevant CUDA distribution versions parent-only and
query `cuda_driver_api_version` only in an initialized CUDA learner using
`cuDriverGetVersion`; CPU never loads/queries libcuda. Query failure prevents
collection. That API reports supported CUDA major/minor, not the exact NVIDIA
driver build, so cross-build determinism remains unverified. No runtime registry
or new dependency is introduced. Default projection/export is offline and
requires a completed parent; an exact live-lock `OnlinePPOContext` is the only
authorization for current-batch online ingestion.
