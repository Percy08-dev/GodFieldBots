# Model3 same-host CUDA export phase

Status: proposed narrow preimplementation supplement. Owner: root. Extends the
accepted device-account design; no new numerical or checkpoint behavior.

## Concrete gap and evidence

Existing learner_cuda_minima is Model4-specific. Do not copy its five FP64 metric
accumulators or subtract unclassified host costs to admit a Model3 save. Source
inspection of structured_artifacts.validate_save_context/_save_inference/_save_resume
and the existing CPU-array helper establishes this Model3 path:

- S is current DISTINCT CUDA parameter/buffer/gradient/Adam-moment storage, already
  owned by the learner account. CPU Adam step tensors and exported RNG arrays add
  no CUDA storage. Parameter/buffer/state views/detach alias S.
- Save validation checks each actual FP32 CUDA Adam moment sequentially with
  isfinite(value).all(), then a separate exp_avg_sq<0/any check. N is the largest
  actual CUDA moment numel, or0 if there are no moments. The accepted pinned
  finite-operation decomposition adds at most5N bytes (FP32 abs plus boolean
  comparison); the later N-byte comparison mask/bool scalar is exclusive. For
  nonempty N, the maximum is5N; no moments adds0.
- Weight and moment export detach aliases, then .cpu()/NumPy/copy allocate CPU
  storage only. There is no second whole CUDA weight copy. Network shape/dtype
  checks do not run a separate GPU parameter-finite operation on this path;
  later bundle/array finite checks operate on CPU arrays.
- CPU archive/shard/NumPy staging retains its existing Source/save/logical budgets
  and last-use ownership. It is not added to CUDA bytes. Backend allowance remains
  the configured shared Fhost, not a measured physical guarantee.

Thus the Model3 export preflight is Fhost + actual CUDA cache K + S + (5N if N>0
else0) <= actual admitted G. Refresh S/N once at existing ownership transitions,
not per inference row. The save scope holds the same serialized Torch lock;
actual failure owner/traceback/storage remain held/poisoned until positive cleanup.
Existing logical CPU/save validations, binding/resume metadata, optimizer/RNG and
file order are unchanged. No arbitrary margin, Model4 scalar carryover or claim of
measured allocated/reserved/PSS follows from this analytical tensor-storage table.

## Verification and implementation scope

After independent architecture READY, the existing Task6b implementer may add
this Model3 export phase to its already assigned new device account/profile and
LocalGameLearner.save wrapper plus focused tests. No other file-scope expansion,
new dependency, public activation, training or GPU experiment is authorized.

TDD: fresh no-moment export, heterogeneous actual moment shapes, CPU moments/
steps/RNG excluded, S alias deduplication, finite/variance phase maximum, shared
Fhost/cache composition and exact byte-boundary acceptance/refusal before save
mutates files. Preserve successful nonbinding existing save/resume content and
failure hold semantics. A tiny CPU operator decomposition supports structure only;
actual Task6 CUDA measurements remain the separate acceptance evidence.
