# PPO HPO record and coordinator interfaces

This implements the approved [HPO design](ppo-hpo.md), rather than changing old
schemas. Baseline0401842; choices below were proposed jointly by units3/4 and
accepted by root before dependent Source/native implementation. Independent
`bot_architecture_reviewer` review is READY, with no blocking findings. Runtime
run/training/storage projections must match the sole spec/base authority;
resume rechecks the derived durations and original-start absolute deadlines.

## Single run-spec authority and adapter

`effective_config.hpo_spec` is the sole immutable run-spec authority.
`config_sha256` hashes the complete adapted effective config, including this spec.
Keep existing runtime `PPOTrainingConfig` fields (`device`, one seed, max_updates,
owned/storage reservation, cleanup and log limits) in `effective_config.training`
for the common coordinator. The HPO recipe is separate in `hpo_spec`.

CLI owns the concrete `HPOExperimentConfig`/loaded-run composition in
`gf_cli.hpo_config`, using the existing base run/server/actions/opponent/
preprocessing/storage objects, explicit configured model4, runtime PPO config,
derived run deadlines and closed HPO spec. Training consumes these through its
known HPO operations variant and pure JSON parsers, without importing CLI.
Do not introduce a second runtime-config framework inside training.

The closed spec fields are:

```
schema_version=1
kind="ppo-hpo-run"
decision_budget
training_recipe                 # complete HPOTrainingConfig JSON
effective_settings_sha256
base_config_sha256
campaign_config_sha256
study_identity
operating                       # complete normalized HPOOperatingConfig JSON
storage                         # complete normalized HPOStorageConfig JSON
selection_version
deadline_version="hpo-deadline-v1"
deadline_inputs                 # requested wall and exact base timer inputs
```

Parse persisted primitive operating/storage fields in training/common without a
training→CLI edge. Reuse pure validation where practical; do not duplicate a
resource ledger. Reject unknown fields, bad finite/types and inconsistent hashes.

## Explicit deadlines

Let requested run wall duration be W, base finalization reserve F, and the base
admission/execution ratio be q. Define execution duration E=W−F and admission
A=E×q. Require W>F>0 and 0<q<1; retain existing base cleanup/report reserves and
validate their consistency before work. For the7200/5400/7140/60base this exactly
preserves the old durations when W=7200. HPO explicitly permits requested W above
the legacy7200cap; it does not relax that cap on any old path. Persist derived
durations in effective_config.run and `deadline_version` in the spec.
The closed `deadline_inputs` keys are `run_wall_seconds`, `base_wall_seconds`,
`base_admission_seconds`, `base_execution_seconds`, `base_finalization_seconds`.
CLI copies them from the exact base/config request. This lets structural readers
recompute durations; an opaque base hash alone is insufficient for that check.

All absolute deadlines are derived ONCE from each run's original start time.
Pause/resume/session/day boundaries never extend them. Admission applies to new
whole collection batches; already admitted work still obeys execution/wall and
cleanup reserves. Stops below D are incomplete. Session timing is outer-only.

## Record contracts

All old required key sets, game/public-contract fields and allowed diagnostics
remain as defined in the current source for their NEW explicit variants; no old
reader accepts the new versions by a numeric inequality.

- **Run5**: existing run/game required fields with schema5/model_version4, plus
  required `run_variant="hpo-ppo-v1"`, `hpo_spec_sha256`, `collected_decisions`,
  `used_decisions`, `stop_reason`. `accepted_decisions==used_decisions<=D` and
  collected>=used. Stop reason is null while running, `pause` while paused;
  complete requires exactly D plus healthy artifacts/source/cleanup. Normal
  early stops use status `incomplete`, reason `update_cap|time_cap|storage_cap|
  empty_collection`. Failure/quarantine semantics remain strict. No formal fields.
  Explicit packed variant additionally binds storage in config/state.
- **Generation4**: existing generation/game fields plus `hpo_spec_sha256`,
  cumulative collected/used counters; accepted==used. Immutable `artifact_files`
  retains the complete originally saved file map even after authorized pruning.
- **Update5**: existing update/game fields plus `hpo_spec_sha256`, per-batch
  collected/used/overshoot counts and `expected_optimizer_step_delta`.
  Accepted==used; overshoot=collected−used. Planned games=recipe.collection_games.
  Delta=epochs×ceil(used/minibatch). Crosscheck Source selection, Numeric5, actual
  optimizer acknowledgement and generation counters. Carry explicit cost metrics:
  optimizer_step_delta, epochs, tbptt_length, endpoint_evaluations,
  context_decisions and suffix_decisions. Full Source validation recomputes
  prefix-derived context costs; structural chain validation checks recipe/count/
  step arithmetic and exact persisted metrics.
- **Registry4**: closed `{schema_version:4,generations,evaluation_references}`.
  Canonical decimal generation keys map to `{files}`. File keys EXACTLY match
  their immutable Generation4.artifact_files; each value has closed
  `{sha256,status,deletion_intent,deletion_receipt}`. Status is
  `retained|deleting|deleted|failed`. Retained has null receipts; deleting/failed
  requires an intent; deleted requires a matching successful receipt.
  Intent is `{operation_id,staging_basename,sha256,original_identity}`; receipt is
  `{operation_id,sha256,original_identity,completed_unix}`. Original identity uses
  the existing owned-deletion stat/identity primitive, not a new lossy identity.
  Evaluation references are bounded tokens mapping to
  `{generation,inference_sha256}`, modified under the SAME parent lock.
  No generation-level all-or-nothing artifact_status is inferred.

`parent_controls` keeps EXACT keys
`{schedule_opponent_rng,snapshot_pool,collection_position}` with new HPO semantics.
G0 pool is[0]. Append generation g ONCE when
`floor(used_g/pool_interval)>floor(used_previous/pool_interval)`, then trim to
pool_size; collection_position=g. Replay controls from the chain/recipe instead
of persisting a second mutable threshold counter. Preserve frozen `_controls`;
use an explicit HPO control producer.

## Retention and original authority

Protect G0 full state; newest at least2resume states and each resume's inference/
weights dependencies; inference latestN/periodic decision crossings/pool/
in-flight evaluation references and their weights. Save and verify the complete
new state before pruning. Preserve generation/update/control, original and
numeric evidence. Under the SAME parent lock verify references and file custody,
persist intent, then use existing owned verify/stage/unlink. Never convert
uncertain or foreign-file deletion into success. Native Registry4 interpretation
checks each individual file; old artifact_status semantics remain exact.

## Source, selection and numerical boundaries

Collection6 linkage freezes closed precollection selection context:
`{selection_version,decision_budget,used_before,remaining,learning_seed,batch}`.
It binds remaining=D−used_before and the original parent/collection identity.
After natural completion, Proof3/Numeric5 bind actual complete episode counts,
collected/used, independently derived PCG64 seed and sorted unique selected flat
indices with hash. Source reconstructs these from actual originals before update
authority. Optimizer delta is independently derived from the full recipe.
Exact selection fields are `{context,derived_seed,episode_counts,
collected_decisions,used_decisions,indices,indices_sha256}`. Selection version is
`hpo-endpoint-selection-v1`. Derive the128bit PCG64 seed as
`int(digest({"recipe":"hpo-ppo-selection-seed-v1","context":context})[:32],16)`;
indices hash is `digest(indices)`. All/partial/empty selection replays the same
versioned rule; partial uses uniform nonreplacement choice, then canonical sort.

Proof3 keeps old plain/packed proof fields and adds `selection` and
`expected_optimizer_step_delta`, with kind
`card-instance-hpo-original-source-proof` or
`card-instance-hpo-packed-original-source-proof`. Its `accepted_decisions`
continues to count BOTH participants' acknowledged originals. Learner counts
live in `selection`; Run/Generation/Update accepted aliases count used endpoints.

Numeric5 keeps numeric4 structural/game-binding fields, uses inference5 semantic
fields (old semantic set minus `reward_policy_version`), and adds complete
`training_recipe`, `selection`, `expected_optimizer_step_delta`. Kind remains
`structured-rollout`, explicit schema5. `decision_count` counts used endpoints;
the sum of complete episode frames counts collected endpoints. Persist full
targets and zero unused advantages; derive selected locations from actual
episode counts and indices rather than a second competing location array.
Empty selection facts are representable for validation; an empty batch cannot
issue an optimizer update. The coordinator stops at D before another collection.

Reuse SAME `_GameBatchSource`/`IssuedGameProjection`, `_OnlineGameOwner`, packed
Source/custody, native session/provider and numerical process owners through
closed HPO variants. A thin HPO adapter is not a new Source stack. Freeze verified
used count AND required optimizer delta before decoded retirement. Whole original
episodes remain for parity/GAE/context; only selected endpoints contribute loss.

Unit2's component interface is `HpoRollout(records,model4,training,*,actions,rules,
selected_locations,_reserved_bytes=0)`, retaining all episodes and sorted
selected `(episode_index,own_decision_index)` pairs. `__len__`, endpoint/vector
iterate selected only; collected/used are separate. Component construction does
not grant online Source authority. Numeric5 serialization/context belongs to
unit3 and reuses physical bundle1.

Unit4 exports `hpo_lineage.hpo_spec(state)`,
`hpo_state(state)->(actions,Model4,runtimePPO)` and
`validate_hpo_chain(directory,*,require_complete=False,max_owned_bytes,deadline,
extra_owned_bytes=0,scan=None)->(state,records)`. The latter performs structural
lineage/artifact/control validation with the supplied SAME `_ChainScan`.
Unit3 owns `validate_hpo_parent` and complete original replay/selection validation;
it calls the chain validator, never the reverse. Avoid circular Source/chain calls.
Structural readers may honestly inspect CPU component fixture metadata; structural
`require_complete` is not experiment qualification. Full HPO production Source/
coordinator/campaign eligibility requires actual CUDA training and inference plus
normal preflight/save/cleanup evidence. Never relabel a CPU fixture as GPU evidence.
Actual Collection6 packed-original resolution/hash/replay belongs to Source,
rather than importing Source into the structural chain reader.
Resume identity is domain-separated hash of
`{version:"hpo-ppo-resume-v1",config_sha256,identity,execution,classification,
storage_binding:null|binding}`; the full config binds recipe and resources.

Native uses exact new Hpo/PackedHpo source formats and plan5, preserving old
plan2/3/4. Required additive capability is
`card_instance_hpo="model4;collection6;proof3;numeric5;selection1;provider1"`,
alongside existing corrected/Pack-checksum markers. Advertise it only after
actual original/header/token/array/correspondence support and checks. Python owns
selection, GAE/reward/normalization; native validates structural original facts
and supplied selection correspondence. Missing support refuses before work.
