# HPO campaign, evaluation role and CLI contract

Concrete unit5b design under [the HPO plan](ppo-hpo.md), negotiated with the
[runtime contract](ppo-hpo-runtime.md). Root accepts the choices below; scoped
architecture review is READY with no blockers before code. Runtime4 owns core collector/config/
coordinator/retention files; outer5b owns new composition and campaign modules.

## Responsibility and commands

`gf_cli.hpo_campaign` owns Optuna ask/suggest/normalization/dedup/history/tell and
durable campaign/session records. `hpo_storage` owns measured filesystem/capacity
checks and one campaign flock. `hpo_evaluation` composes actual Source/artifact/
Simulator verification. `hpo` owns command parsing and concrete workflow; the
existing CLI adds a lazy route. No training→CLI/evaluation edge, executor callback
registry, duplicate server loop or missing-future-module foundation.

Commands: `hpo validate --config`, `hpo run --config --mode fresh|resume
--campaign-dir`, `hpo inspect --campaign-dir`, and `hpo compare --config --left
<closed-trial> --right <closed-trial> --output`. Inspect is honest structural
inspection, not qualification. Explicit recovery flags/subcommand call the
existing fixed global restart-confirmation API; no automatic recovery/reboot.
Optional-free help/validation imports neither Torch/device code nor Optuna.

## Immutable experiment and study binding

Persist campaign UUID, effective/base config hashes, actual Bots revision/module
hashes, Simulator binary/config/normative hashes, action/rule/layout/token/input
versions, model/stage/space/D/complete training-seed/common GameSeed sets,
inference/curriculum conditions and canonical DEFAULT contrast identity. Bind a
conditions/study digest both in campaign.json and Optuna study attributes.
Different source/conditions/space/budget cannot resume or mix in the same study.
Resource/session settings remain operating records, not search dimensions.
Bind the actual Optuna study to its owning campaign UUID as well as scientific
identity. Separate campaign locks cannot authorize two campaigns to schedule one
study; reject foreign ownership rather than sharing scheduling authority.

Trial count means bounded attempted IN-SPACE proposals, including failures and
effective duplicates. The in-space canonical control consumes one slot. An
out-of-space canonical contrast is a separate additional control with no Optuna
trial/history entry. Never invent retry budgets or run forever looking for unique
settings. Every training seed is a fresh initialization; only final exactly-D
policies can contribute objectives.

## Durable ask/suggest/tell

Before any ask/enqueue, persist an immutable numbered intent with proposal number,
slot/kind, version-derived32bit sampler seed, exact healthy-completed-history
digest, expected next Optuna trial number and requested-settings identity when
fixed. Use a fresh pinned Optuna5.0.0 TPE instance per proposal with declared
startup count, multivariate=True, group=True, constant_liar=False and NopPruner.
Actually call suggest_* across active validated domains/coupled choice indices;
SearchSpace.resolve and effective_settings remain normalization authority. Jointly
admit the complete actual effective setting before GPU work.

For an in-space fixed contrast/imported setting, enqueue with intent-bound user
attributes, then use the SAME ask→suggest/validate→persist complete proposal→train
→evaluate→verify→tell workflow. Persist exact params, distributions and full
effective model/recipe before any training. Reconcile exact WAITING enqueue state
through its immutable intent. A partial RUNNING proposal without a complete
durable proposal becomes an unexecuted FAIL; it never trains. No unexplained
study trial is tolerated. An absent expected trial permits enqueue/ask only after
actual bound-storage inspection confirms no committed corresponding record.

After all healthy seeds/evaluations, persist the exact aggregate evaluation
receipt before tell. Exact retry requires matching params, distributions, state,
value and evaluation-receipt identity. Do not hide conflict with skip_if_finished.
Missing/failed/incomplete/OOM/transport/save/cleanup never becomes partial mean or
zero. Completed seed evaluation receipts preserve the verified paired outcomes
and projections before the full trial completes; these are the recorded
intermediate performance evidence. Generation loss metrics are separate training
diagnostics. No additional evaluation cadence or Optuna.report replay state is
required. No pruning, budget promotion or best-intermediate checkpoint choice.

## Closed campaign records

Every record has schema_version1 and a distinct kind plus predecessor hashes.
Records are exclusive owned temporaries, file-fsynced/atomically published and
directory-fsynced under one campaign flock. Unknown/corrupt/conflicting records
fail closed; do not overwrite failure history.

| Record | Required payload |
| --- | --- |
| campaign | campaign_id, loaded_config_sha256, study_identity, conditions, conditions_sha256, canonical_control_identity, optuna_version, schema_versions, created_unix |
| proposal intent | campaign_id, proposal_no, slot, proposal_kind, sampler_seed, completed_history_sha256, expected_trial_number, requested_settings_sha256 or null; fixed params/identity sufficient to replay a predeclared enqueue |
| proposal | intent_sha256, trial_number or null, raw_params, distributions, effective_model, effective_training, effective_settings_sha256, resource_admission |
| seed training | proposal_sha256, training_seed, run_path, run_id, config_sha256, state_sha256, generation, generation_sha256, inference_sha256, used_decisions, source_receipt_sha256, gpu_termination_sha256 |
| seed evaluation | training_receipt_sha256, linkage_sha256, manifest_sha256, originals_digest, projection, conditions_sha256, pin_acquire_sha256, pin_release_sha256, gpu_termination_sha256 |
| evaluation | proposal_sha256, ordered training/evaluation receipt hashes, value, projection_sha256, label exploratory-hpo-validation |
| tell intent/receipt | proposal_sha256, trial_number, params, distributions, COMPLETE or FAIL, value finite or null, evaluation_receipt_sha256 or null |

Add explicit failure/incomplete/duplicate/session records without objective values.
Exact final record field spelling may be refined within these closed authority
boundaries; no arbitrary management fields enter policy inputs.

## Concrete run and evaluation ownership

Predeclare each run UUID and admit the global JobContext before inner GPU work.
Runtime4 materializes the run and returns actual typed `HpoRunResult(report,
gpu_termination)` after real cleanup/confirmation; outer code retains/closes its
flock. Revalidate full original Source and final-D policy to produce training
evidence, not structural-chain-only qualification. Session budgets are checked
only after a whole run/evaluation normal boundary; original run deadlines persist.
Resume reuses only closed operation receipts after full revalidation. An intent
without a closed receipt makes that entire trial failed/incomplete, including a
cleanup-to-receipt storage gap. Success-looking parent files or GPU termination
alone cannot reconstruct success. Closed aggregate/tell gaps can replay only
exact known evidence. Inner runtime normal-update-boundary resume remains a
separate operation. This conservative restriction can be revisited if an actual
recovery workload warrants a separately reviewed reconciliation protocol.

For evaluation: admit its own global job, acquire the exact parent pin transaction,
verify full Source, retain exact artifacts, collect through existing Simulator/
Journal6/Pack1/SQLite1/process owners, verify saved originals/result/health, close
all Source/artifact/transport owners and release the pin transaction. Only then
confirm termination and persist a closed evaluation receipt. Outer owns these
scopes; core collector never confirms the job or pins parents. Strict pure5a
aggregation receives independently verified projections. GPU receipts never
grant Source or data qualification. No failed learning parent is rehabilitated.

## Distinct HPO evaluation Collection6 role

Add frozen `HpoEvaluationRole`/factory, exact role `hpo-evaluation`, split
validation and positive even games==2×explicit GameSeeds. It is distinct from
HpoCollectionRole, PPO_BATCH and FINAL_VALIDATION. Keep schema6/model_version4,
Journal6/codec2/Pack1/SQLite1. Training/native training checkers remain batch-only
and reject evaluation. No selection context, final64 projection, qualification or
validation-invocation fields are borrowed.

Closed linkage schema1/kind `hpo-evaluation-linkage`:
`{campaign_id,evaluation_id,gpu_ticket,purpose,conditions_sha256,left_parent,
right_parent,game_seeds,learner_seats,schedule_digest,policy_registry_digest}`.
Purpose is objective or comparison. Parent references are closed
`{run_id,run_path,training_seed,generation,generation_sha256,inference_sha256,
source_receipt_sha256,policy_label}`; paths are absolute. Objective requires right
null and the canonical heuristic; comparison requires the actual right parent.
Learner seats identify LEFT's seat. Verify complete GameSeed/seat/policy/stream/
schedule/registry correspondence. Actual GameIDs come from verified post-creation
original manifests, not guessed before creation.

Producer core keys are EXACT schema_version/action_contract/rule_signature/
layout_signature/purpose/synthetic_capacity/config/config_hash/action_config_hash/
bots_revision/simulator_revision/runtime_evidence/schedule/files/policies/
journal_schema_version/derived_schema_version/participant_protocol_version/
candidate_codec/model_version/input_recipe/token_signature/source_eligibility/
policy_bindings/neural_inputs/collection_role/linkage. No tree_inputs.
Published fields additionally are EXACT kind/run_id/command/status/
schedule_schema_version/code/inputs/groups/episodes/resource_admission/report_sha256.
Kind is card-instance-collection, command evaluate, status complete,
run_id==evaluation_id, source_eligibility is not-authorized-by-collection-metadata.
The unchanged public-runtime binding purpose qualified_game does not qualify
training/formal data. Reuse exact nested runtime/config/journal validators.

`hpo_evaluation_collection_metadata` builds/checks the closed core; final
publication/original inspection closes the complete key set. The original
inspector reuses SAME original/Packed owners and replay checks but cannot issue
learning handles or final64 authority. Runtime4 alone owns
`collect_hpo_evaluation(loaded,exact_plan,*,gpu_job,artifact_inputs)->PackedRunResult` and the
small known core collector branch;5b owns new composition and pure role/metadata
helpers. Use an exact typed plan/context rather than a generic callback registry.
`artifact_inputs` is the outer caller's actual RetainedGameArtifacts owner.
The core collector borrows that SAME owner; outer retains it through complete
original/Replay revalidation and closes it before global termination confirmation.
Only initial admission checks the admission deadline; a healthy admitted operation
may finish later within its execution/work/original wall deadlines.

## Stage seeds, comparison and storage

`--seed-settings` imports only closed schema1/kind hpo-settings-seed files with
resource-free model4 architecture, normalized training recipe,
effective_settings_sha256 and origin
`{campaign_id,study_identity,proposal_sha256}`. No weights/checkpoint/
value/outcome/path fields or optimizer state. Adam/PPO settings remain in the
recipe. Emit this immutable settings file when finalizing a full proposal;
reattach the NEW campaign's model resources before parsing/admission.
Reparse/normalize/recompute identity, invert into the
NEW declared space and jointly admit. Fresh/new study only; reject same-study,
duplicates or out-of-space settings. Each imported in-space proposal consumes a
normal attempted slot and starts fresh weights; no old sampler history transfers.
Origin is informational provenance only: a failed old run's settings may be
selected without claiming prior data qualification or strength.

Compare closed trials at the same declared decision budget, execute the complete
explicit common panel over all declared left/right training-seed pairs, and save
the exploratory matchup table/equal paired means. No Optuna objective or final
test/strength claim arises from comparison.
The initial comparison projection uses LEFT's materialized operating/storage/
public-input/inference profile and requires the supplied config to match it.
RIGHT can vary model/recipe/search/history but must have compatible public inputs
and inference conditions; admit both actual artifact resource envelopes jointly.
Record both profiles in the table. Revisit this restriction when a real comparison
needs wider compatible profiles rather than silently changing the projection.

Measure actual usage/free/reservation for each destination before creation/work
and each operation boundary. Parent/numeric/model/Optuna are ext4; originals keep
the explicit/base D mapping and the same collector. For Windows-C free space,
parse actual mount facts identifying Drive C and require the configured path on
that mount; unknown/non-C/missing mount fails. Never use nearest-existing-parent
statvfs or a caller label as proof. Persist measurements/cost scope; no speculative
quota framework. Existing immutable original data remains protected.

## Verification and acceptance

Actual local Optuna SQLite tests cover conditional/index distributions, proposal
seeds/history/dedup, WAITING/partial ask faults, exact tell retries/conflicts,
canonical in/out-space controls and configuration-only stage seeds. Test complete
seed/pair aggregation, all failure-no-objective cases, whole-operation session
boundaries, filesystem faults/real C mount identification and cross-campaign GPU
recovery. Reject evaluation roles through training/final/native paths. Real GPU/
Simulator training/evaluation/resume/compare are final bounded functional tests,
not formal search or strength evidence. Whole/static/Rust/old-data/optional-free
checks and separate final correctness/conformance reviews remain mandatory.
