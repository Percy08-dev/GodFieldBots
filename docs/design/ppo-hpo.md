# PPO hyperparameter optimization foundation

## Authority and scope

This records the user-approved implementation plan supplied on 2026-11-12.
Baseline: Bots `040184223a4bdcde85875f556daeb4a160ee7027`; Simulator
`91d5926a7572db327eaea3751e7c24f428120996`, including the currently adopted
normative document hashes and corrected card-instance contract. The previous
context reported the design READY, including abnormal-exit recovery. This
context maps that design to the current code before dependent implementation.

Implement configurable MLP/GRU/Transformer PPO, equal unique-decision budgets,
conditional joint search, campaign persistence/resume, evaluation and retention.
Reuse the coordinator, Simulator, Python numerical semantics and compressed
SQLite original storage. Optuna manages trials outside the coordinator. Do not
run formal HPO or claim strength. Search ranges, trial counts, experiment budgets
and seed counts are required caller inputs, not code defaults. LightGBM fitting,
pretraining, a model registry replacement, checkpoint format migration and HDD
migration are excluded.

## Responsibilities and dependency direction

- `gf_models`: closed model4 descriptor, pure dimensions/admission and shared
  Torch kernels; safe inference5 artifacts contain inference architecture only.
- `gf_training`: closed HPO training recipe, unique endpoint selection verified
  against complete original episodes, GAE/normalization, numeric5/wire6,
  run5/generation4/update5/registry4 coordination and owned retention.
- `gf_common`: pure schema/identity/selection facts only, no Torch/Optuna/I/O.
- `gf_evaluation`: paired GameSeed and equal training-seed aggregation with
  strict complete-result receipts, plus a separately recorded matchup table.
- `gf_cli`: strict file parsing, concrete Simulator/coordinator/evaluator wiring,
  campaign orchestration and `hpo validate|run|inspect|compare` composition.
  `run` requires explicit `--mode fresh|resume`.
- Optional native: original-record/array/correspondence validation and provision;
  never GAE, reward, normalization, sampling or search decisions. Require an
  additive HPO capability before native dispatch; never silently fall back.
- Optional Optuna5.0.0: ask/tell TPE scheduling only, lazily imported by HPO.

Keep fixed recipes, eight fixed kernel ASTs, all old identities and readers
unchanged in meaning. Extend shared kernels explicitly by version; avoid a copied
Simulator, duplicate trainer loop or global mutation of legacy constants.
The concrete extension points are the existing shared config/network/shape and
artifact helpers, a distinct HPO learner (leaving fixed `Learner.__init__` and
`Learner.update` exact), and a known HPO operations variant of `_run_ppo`.
Do not add a second coordinator loop. HPO configuration uses an explicit loader
wrapping an unchanged corrected base `LoadedConfig`, with pure model4 helpers in
models, training/search descriptors in training, and campaign/operating settings
in CLI. Resume5 recipe-aware ownership stays in training, or uses a closed pure
numeric descriptor in models; models must never import training to parse it.
Inference5 omits the model resource-management table and reward policy; runtime
admission remains caller-owned and resume5 binds the full resources/execution
and natural-terminal training recipe. Old inference3/4 serialization is exact.
The physical Pack1/SQLite1/typed NumPy bundle1 and semantic Journal6 are reused.
New logical identifiers: model4, inference5/resume5, Run5/Generation4/Update5/
Registry4, Collection6/Proof3/Numeric5/Wire6. HPO completion/evaluation authority
is separate from formal parents/final64/Dataset4 and confers none of those rights.

## Configurable model and training interfaces

Expose state encoder kind/hidden widths/depth/output; card definition/ability
embedding and output dimensions; hand kind/hidden widths/depth/latent/pooling/
attention layers, heads and FFN; event width/depth/aggregate history; temporal
MLP/GRU/Transformer width/depth/heads/FFN and TBPTT; actor/critic widths/depth;
ReLU/Tanh for dense and attention FFN activations. Shared encoders, dropout0,
IEEE FP32, own-view candidates and candidate-independent critic are fixed.
Temporal MLP depth includes frame projection: depth1 exactly reproduces existing
names, shapes and initialization order. Version activation-aware orthogonal
initialization: ReLU gain sqrt(2), Tanh5/3 for dense hidden layers. Preserve GRU,
embedding, LayerNorm, attention-output and actor/critic-output initialization.
Model4 uses a distinct temporal descriptor without `sequence_length`: TBPTT
belongs to the training recipe/resume5 only, and must not change inference5
identity. Keep `max_decisions=1024` as the forward-context contract. Training
resource admission supplies suffix length explicitly; legacy temporal descriptors
retain their exact serialized fields.

Expose collection games (positive multiple of8), Adam learning rate/betas/epsilon,
PPO epochs/logical minibatch/policy clip/value and entropy coefficients/GAE lambda/
gradient norm clip. Natural terminal rewards, gamma1, Adam algorithm and loss
formula are fixed. Preserve seat pairs and existing Random/heuristic/past-policy
ratio. Configure retention, concurrency, resources, comparison seeds and budget
outside the search space. Validate closed keys/types, bool-as-int, finite values,
dimensions, conditional fields, tensor/history bounds and JOINT RAM/CUDA worker
admission before model allocation or games. Preserve full1024-decision forward
context; TBPTT changes backward extent only.

## Decision budget, data flow and integrity

Budget D counts unique learner decisions included in loss, once across epochs.
Finish each collection batch naturally; retain all original episodes. Let R be
the remaining budget. If fewer than all learner decisions are needed, choose
R uniformly without replacement with a versioned independently derived RNG seed,
separate from policy/training/Optuna RNGs. Canonical order is schedule episode
order then the learner's own decision number. Sort selected indices in that order.
An empty learner batch cannot advance training. Compute GAE on COMPLETE episodes;
normalize advantage on selected endpoints only. Nonselected prefixes remain valid
context without contributing loss. Source reconstructs selection indices, count
and hash from originals; a saved mask is never authority.

Persist collected/used decisions, overshoot, epochs, optimizer steps, context and
compute costs. Expected delta optimizer steps is
`epochs * ceil(used / minibatch)` per update, consistently checked by Python,
native/IPC, generation validation and decoded-memory retirement. Full inference
and resume state is saved and verified at each successful update before retention.

## Search, campaign records and evaluation

One study per model and stage. TPE includes startup random sampling and explicitly
sets multivariate=True, group=True, constant_liar=False; use NopPruner. Config
supplies the explored subset and categorical/integer/float/log/fixed/conditional
domains. Normalize inactive fields and generate valid coupled width/head choices.
One trial is one effective config plus the full declared training-seed set, all
fresh initializations. Deduplicate effective settings explicitly. Run the default
control; if outside the space, retain a separate control record, not sampler
evidence. Stage promotion carries SETTINGS only to a new study; changing space,
budget or evaluation conditions within a study rejects.
Opponent-pool decision interval and size affect the training curriculum: bind
them to study identity even though they are operating rather than search fields.
Validate domain dependencies against the actual overridden normalized control
and active conditional branches, rather than library defaults.

Persist a proposal-number-derived sampler seed, completed-history digest and full
proposal BEFORE training. Incomplete proposal generation remains an unexecuted
failure. Workflow: ask -> finalize/persist configuration -> train each seed ->
evaluate -> verify receipts -> tell. Store intermediate results without pruning
or automatic budget promotion. A tell replay is idempotent only if parameters,
distributions, state, value and evaluation receipt match exactly.

Primary objective: final checkpoint at exactly D versus the fixed heuristic,
same GameSeeds with both seats. Win1/draw0.5/loss0. Average each seed's seat pair,
then GameSeeds, then give each training seed equal weight. Require every run to
reach D normally and every planned evaluation to complete naturally and save/
cleanup successfully. Missing seed/game, OOM, transport/storage/cleanup failure
produces no objective (neither partial average nor artificial zero). Never select
best intermediate checkpoint. Direct model comparisons are separate with a saved
matchup table. Selection conditions are exploratory, never final-test evidence.

## Resources, storage and retention

Training and neural inference use GPU. Initial operation admits one trial at a
time, checking intra-trial game concurrency jointly. Operating defaults: managed
RAM24GiB/CUDA12GiB on32GB/16GB hardware; configurable and accompanied by observed
usage. Parent/numeric/models/Optuna DB use ext4; compressed original DB may use
the existing supported D-drive path. Check destination caps and actual Windows C
free space before processing. Preserve ext4 parent control requirements.

Protect G0 full state and latest TWO resume states. Inference retention independently
unions latestN, periodic, opponent-pool and in-flight evaluation references.
Registry4 has per-file retention/deletion receipts. Under the SAME parent lock,
validate references and commit deletion intentions; uncertain deletion cannot be
reported complete. Pool additions use used-decision intervals, with no duplicate
checkpoint if one update crosses several thresholds.

## Deadlines, sessions and abnormal exit

Keep each run's ORIGINAL absolute deadline across resume. Time/storage/update caps
at used<D are incomplete and return no objective. Session budgets (approximately
18hours/day operationally) stop only at a normally finished training-run or
evaluation boundary; seeds may span days. A session budget does not extend a run.

ALL campaigns share one GPU admission lease. Durable job admission precedes GPU
work; confirmed termination receipt follows complete cleanup. Abnormal exit with
unreaped jobs sets RECOVERY_REQUIRED and blocks subsequent GPU work. Flock release,
elapsed time and saved PID alone never prove reaping. Recovery requires a verified
termination receipt or EXPLICIT environment-restart confirmation; never reboot
automatically, rewrite deadlines or grant data qualification through recovery.

## Verification and implementation units

1. Closed configuration/model4/training/search contracts and joint resources.
2. Shared variable network, activation initialization, HPO learner and artifacts.
3. Collection6/Proof3 selection/Source validation and capability-gated native path.
4. Common coordinator, run5 lineage, retention, pool and completion/deadlines.
5. Optuna campaign, CLI, evaluator, abnormal recovery and integration.

Each unit needs meaningful failing/passing tests and independent scoped review.
Root owns config/docs/local verified commits. One implementer owns Python/Rust/
tests at a time; reviewers do not edit. Final correctness and design conformance
reviews are separate, with blocking findings resolved before acceptance.

Verify old Config/artifact/recipe/AST/original hashes and same-input defaults;
all encoders/models/activations/depths, critic independence and causal full/cache/
split CPU/CUDA; selection all/fraction/one/empty/replay/tamper/GAE/step counts;
conditional search/invalid configs/duplicate settings/equal-seed aggregation/no
failed objectives/stage carry; injected save/update/delete/eval-before-tell faults;
protected references and global recovery admission. Run small GPU training/GPU
inference/normal resume/actual Simulator games as FUNCTIONAL checks. Run relevant
pytest/Ruff/mypy/Rust and final whole checks, optional-free CLI/old-data checks.
Record exact PASS/FAIL/NOT RUN and limitations in docs/verification/ppo-hpo.md.

Acceptance: HPO can start with caller-supplied ranges/trial count/D/seeds/evaluation
conditions without code changes. Formal search, experiment sizing and strength
remain future work.
