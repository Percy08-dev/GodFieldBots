# 0006: Native CPU LightGBM action-value adapter and gated training

- Status: Accepted
- Date: 2026-10-27
- Decision owner: root
- Review: `bot_architecture_reviewer`, design READY, 2026-10-27. Model-native
  parameter boundary and pilot-inclusive split lineage revisions resolved both
  findings; no blocking/open decisions. Final independent correctness and
  architecture-conformance reviews are READY; evidence and dispositions are in
  the Phase 4 verification record.
- Implementation: native CPU adapter, encoders, guarded trainer and checkpoint
  inference implemented; final checks/reviews in
  [Phase 4 verification](../verification/lightgbm-phase4.md). That initial unit's
  real-data gate was unmet; the subsequent corrected-contract pilot in
  [ADR0007 verification](../verification/buy-contract-adoption.md) passes provisional
  feasibility. Real-data fitting remains unexecuted.
- Compatibility amendment: [ADR0007](0007-staged-buy-contract-adoption.md)
  supersedes the inference schema-1 identity policy for the corrected Buy rules;
  the original design and verification below remain historical evidence.

## Context and requirements

Phase 0–3 is committed at `1719471`. The Simulator pending-buy metadata defect
is still present at the pinned revision; it must not be bypassed or fixed in a
Bots task. The user's plan explicitly permits model implementation and synthetic
numerical verification while the clean learning pilot is unmet.

Implement the Phase 4 LightGBM unit: candidate-content `Q(o,a)` regression,
current-observation and public-aggregate-history encoder variants, native model
save/load, policy inference through the existing environment, strict model/config
composition, and `train` CLI with real-data feasibility and split guards. Test
the model on explicitly synthetic data; do not train/evaluate a production model
from the failed pilot or select only its successful games. No PPO, self-play,
resume implementation, Simulator changes or new dependencies are in scope.

Use the already-selected `lightgbm`/`data` extras (native LightGBM API, not the
sklearn wrapper). Baseline startup/help/imports continue without these extras.
CPython 3.14 GIL and CPU are required; GPU remains unverified.

## Responsibilities and dependency direction

| Location | Responsibility |
| --- | --- |
| `gf_models/lightgbm/config.py` | Model/encoder/init configuration and validation; stdlib only |
| `gf_models/lightgbm/encoder.py` | Pure allowed-input + candidate to fixed numeric vector; NumPy only |
| `gf_models/lightgbm/model.py`, `policy.py` | Native fit/predict, safe inference artifact, Policy adapter; lazy LightGBM |
| `gf_training/q_data.py` | Validated run/journal ingestion, executed-only samples, split/lineage proof |
| `gf_training/gates.py` | Recompute feasibility evidence from finalized artifacts, not a user boolean |
| `gf_training/lightgbm_run.py`, `lightgbm_worker.py` | Budgeted training run ownership, native subprocess, manifests and artifact promotion |
| `gf_common` | Safe runtime policy binding values; shared pure episode scheduling |
| `gf_cli` | Typed config precedence, binding registry, model/checkpoint preflight and command composition |

Models never import HTTP, environment, experiment configuration or source-run
control metadata. Training receives validated samples/features, not engine state.
Common does not import model/training/evaluation/client/CLI. Move the existing pure
schedule builder to common for evaluation and gate validation; re-export it from
evaluation to preserve current callers. Do not introduce a generic training
framework or abstract interface for each class.

## Model input and encoder v1

`encode(DecisionInput, typed_action, EncoderConfig) -> NumPy vector` is shared by
fit and inference. Feature names/order/config form a versioned signature.

- Own visible state, allowed public opponent values and explicit missingness;
  do not replace hidden/absent values with an apparent observed zero.
- Own hand/learned aggregates preserve multiplicity. Resolve candidate hand IDs
  only against own hand and learned references only against own learned entries.
- Candidate kind, target relative to own/opponent/unspecified, resources and
  ordered element positions. Hand versus learned source is explicit; numeric
  instance-ID magnitude, candidate index, command/Game IDs, cursors, revisions,
  URLs, seeds and artifact paths never become features.
- Resolve definition fields only from the supplied own-visible/perceived catalog.
  Fixed kind/attribute categories and bounded stable hashing of permitted visible
  definition/ability strings are acceptable; no global opponent catalog table.
  Optional numeric properties carry a missing flag. Do not infer Dream identity.
- Encoder `max_elements=3` initially; preserve order with padded positions.
  Reject rather than silently truncate an oversized candidate. Configuration
  preflight must bound generator sequence length and two-element Sell references
  within encoder capacity; an unlimited generator is incompatible with this v1.
- History variants `none` and `public_aggregate`: counts/sums over a configured
  visible-event window, relative participant roles, explicit window truncation.
  Do not consult future/end records, rejection detail or another private view.
- Same-revision authorized atomic-hint flags/counts/membership may be features;
  absent hints are explicit. They do not imply full sequence legality.

Each candidate is independently scored from its content. Stable equal-feature
cards may tie; references remain intact in the selected raw action. Native scores
must have exact candidate count and finite values. `Policy.reset` receives only
the policy sampling seed; v1 inference uses deterministic argmax and identifies
the exact artifact in `PolicyDecision.policy_version`. Value is not misrepresented
as a PPO state-value estimate.

## Safe runtime binding and checkpoint identity

Keep EpisodeSpec's string policy labels, but resolve them through CLI-built safe
runtime bindings: actual model_id and optional inference settings. Baseline labels
remain random/heuristic; LightGBM labels include a stable inference-manifest hash,
so two checkpoints of the same model can be compared and seat-swapped correctly.
EnvironmentSettings carries this registry; no whole experiment config is sent.

Extend the existing importable factory/worker handshake with optional safe policy
options (checkpoint directory and expected manifest digest). Existing one-argument
factories still work when options are absent. Native import/model loading occurs
only after the existing clean child bootstrap scrubs argv/environment. Validate
checkpoint/config compatibility before creating a Simulator listener; the worker
rechecks bytes/hash before prediction. Baseline factory imports remain lazy.

Artifact schema 1: native `model.txt` plus a closed inference JSON containing
model digest, encoder signature/config/feature names, adapter/runtime versions,
input/catalog/reward/projection/generator/scorer/action-config identities and
purpose/readiness. No pickles. Training source paths, Simulator GameSeeds/RNG,
split lineages, server URLs and experiment diagnostics belong only in the
separate training-run manifest and are never loaded by an inference worker.
Native LightGBM text contains model-training parameters and model RNG seeds;
these legitimate model-specific details are explicitly allowed in the artifact,
not treated as input features, and are distinct from Simulator RNG. Do not build
a fragile native-text sanitizer. Fit from numeric arrays using a closed parameter
allowlist (no source-run/dataset filenames as native parameters); tests inspect
both native text and safe JSON for actual source/control-path/seed leakage.

Synthetic fits are marked `purpose=toy`. Formal Simulator commands reject toy or
unready checkpoints before startup; explicitly synthetic component/worker tests
may load them through a test-only/direct low-level option. Do not label a toy
artifact as a trained comparison model. Require the verified native runtime
version initially; broader cross-version compatibility is future tested work.
Only fresh training is supported. `resume` fails explicitly before fit/output,
and never grants a new time budget or revives a failed run.

## Training data, labels and fairness

Training config names complete `train_runs`, `validation_runs`, and `pilot_run`
directories, not arbitrary selected episodes from failed runs. Extract reusable
validated projection construction from the current journal exporter rather than
duplicating its eligibility/state-machine checks. Parquet export stays compatible.
The loader requires finalized complete runs, matching report/source/replay hashes,
all participant journals, healthy natural outcomes and current semantic versions.
Retain exact stored inputs/candidates/order/selections; never regenerate candidates
or recompute a behavior logprob.

One sample per positively acknowledged executed decision; target is that viewer's
validated natural final return (+1/-1/0, gamma=1 and intermediate zero). Do not
create labels for unselected candidates or pending/interrupted records. Record
that this is behavior-selected Monte Carlo regression, not an unbiased causal
value label for all actions or a proof of optimal Q.

Train sources must declare train, validation sources validation. Union equal
GameSeeds/common ancestry across train, validation **and the pilot evidence**
before allowing fit, including both views/repeated runs; reject split conflicts,
duplicate data identities and test data. Preserve the pilot's own declared split
(the current preset is validation); do not invent a new role to allow a pilot
validation seed to enter train. Pilot/validation may reference the same source
run without duplicating sample rows or collection charges.
Do not trust an editable sidecar's eligible flag alone. Initial training requires
the linked original source bundles; do not invent detached historical semantics.

Feasibility validation reconstructs the fixed schedule from the pilot's own
configuration and recomputes all linked health/terminal checks: exactly 64 games
for each of three distinct policy seeds, no missing/duplicate games, all natural,
within original budget, identical input/catalog/reward/generator/scorer/action
conditions. Reject incomplete/interrupted/budget-exceeded evidence before native
fit. The existing failed pilot must be rejected. Synthetic gate unit fixtures do
not constitute a real pilot.

The real experiment must explicitly set an accepted-decision target from a clean
pilot; no guessed target is chosen while the gate is unmet. Use all validated
training samples and require the configured target/overshoot interval (default
maximum 5%); never silently discard/fill samples to claim a matched comparison.
Record source collection cost once per distinct source run as charged historical
cost, plus actual current ingestion/feature/fit/save/reap time. The two-hour
condition counts both. Shared feasibility setup is separately identified, not
silently treated as free model collection. Formal comparative selection remains
Phase 9; no quality conclusion is derived from toy fits or training loss alone.

## Config and CLI

Extend closed Config with lightgbm model encoder/network/initialization and
optional checkpoint, and a typed lightgbm_mc training table. Presets:
`configs/models/lightgbm-{observation,history}.toml`,
`configs/training/lightgbm.toml`, `configs/experiments/lightgbm.toml`.
Defaults: 31 leaves, learning rate .05, 500 rounds, early stopping 50, native CPU
threads 1, seed-derived initialization recipe. All values/paths/budgets are config.
The model preset defines structural/encoder/init values and optional training
differences; common training/actions -> primary model differences -> experiment
overrides -> explicit CLI fields. Opponent preset training deltas do not alter
the learner. Arrays replace; paths resolve at their defining file; unknown keys,
recipe/model mismatch, nonfinite numbers and incompatible capacities fail first.

Add `gf-bots train --config ... [--mode fresh|resume]`; helpers/help/baseline validate
work without optional libraries. Real training may only start after gates and
data/split/budget checks. Missing/ineligible evidence returns a structured error
and produces no successful checkpoint. The default preset has no fabricated data,
pilot proof or decision target; invocation must reject those absent prerequisites.
For each configured model-training seed, run one separately identified experiment
with the same source data and original per-run budget. Model/data sampling RNG is
derived from those configured seeds, never Simulator GameSeed/fingerprint.

## Native training process and failure ownership

Training prepares numeric feature matrices/labels and safe encoder/model settings,
then starts one owned fresh interpreter native-fit subprocess; it receives no
raw GameSeed/server/private/canonical metadata. Native work is not hidden in an
uninterruptible parent call. Parent owns stdout/stderr files, request/artifact
hashes, original budget, cancellation and terminate/kill/reap. Child writes only
a staging artifact; an exhausted/failed/cancelled run cannot promote a ready model.
Bound logs and controllable waits; record actual/charged times and preserve the
existing explicit OS/FS emergency/quarantine debt. Do not import simulator-client
process ownership into training or create a general supervisor framework.

Validate finite fit/validation metrics, file/feature signatures and remaining
budget before promotion. Save completion/readiness only as part of successful
parent finalization; recheck after persistence and independently invalidate run
and artifact on detected overrun, with the existing dual-storage failure limit
explicitly quarantined. Failed/toy/unready artifacts cannot enter normal resume
or evaluation. Source data/result semantics are never altered to make fit pass.

## Compatibility and verification

Raw journal/derived schema 4, input version, generator v3, reward/projection and
Simulator schemas do not change. Model/train manifests and encoder get their own
schema/version. Evaluation run manifest may add individually retained policy
bindings without reinterpreting existing raw rows; current baseline readers and
old factory callers remain valid. No historical model is fabricated.

Tests: hand-calculated feature/order/missingness and ID-invariance; candidate
permutation score correspondence; no future/opponent/private/control input;
executed-only labels and reward perspective; combined split/lineage rejection;
failed/malformed/fewer-game pilot rejection; model/train config precedence/path
validation; native tiny synthetic fit/predict/early-stop/save/load/tamper errors;
same-model different-checkpoint seat binding; toy/unready/gate preflight rejects
before listener/fit; controlled trainer timeout/cancel/log/reap/no promotion;
successful synthetic child lifecycle marked toy; baseline lazy imports/wheel.
Run relevant ordinary/optional-extra checks and independent correctness plus
architecture-conformance reviews. No real Simulator learning or new feasibility
pilot is attempted while the pending-buy contract remains unresolved.

## Required implementation clarifications from conformance review

Current-fit code provenance is distinct from collection provenance. CLI measures
the current Git/installed-package identity plus fit-module fingerprints within
the preparation budget and injects this parent-only control metadata. Direct toy
components explicitly record unknown. Exact request/numeric-input digests are
recorded by the parent and checked in the child; these and fit-code metadata do
not become inference features or extra native training parameters.

Parent artifact inspection is metadata/hash/config-only. Actual native import,
load and feature-signature validation runs in bounded owned children before a
formal Simulator listener starts. An initialization-only use of the existing
clean worker READY handshake is sufficient; no inference decision or new process
framework is needed. The training child self-validates its saved unready staging
model before exit; the trainer parent does not reload native code.

Keep staging unready through child/log cleanup and health validation. Compute
the prospective ready-manifest digest without publishing it, then persist the
successful training completion linked to that digest before publishing readiness
as the last promotion. Keep existing post-write budget checks and independent
invalidation attempts; do not claim a transactional hard wall guarantee.

Operational artifact inspection/load is distinct from **formal qualification**.
Before using a trained checkpoint, CLI parent verifies the linked completed
training control record, readiness/hash binding and absence of failure/health/
invalidation evidence. Either a durable unready artifact or failed training
record blocks formal evaluation, even if the other invalidation write failed.
Only the safe pinned inference path/digest reaches the policy worker. Explicit
low-level toy loads are not formal-qualified comparison models.

Initial formal qualification requires the model run bundle (training completion
record beside its checkpoint); moving that bundle is supported without reopening
source datasets. A portable control-free inference export is Phase 9 work.
Impact: copying only a ready-looking checkpoint is insufficient evidence for
formal use. Revisit the bundle requirement when that export/deployment is built,
not by adding a global receipt/revocation framework now.

Persisted elapsed/charged values must state their pre-final-write snapshot scope;
the returned completion values include final persistence and correction time.
Historical collection timestamps retain their own recorded scope and unknown
details are not invented. Test delayed final writes and independent partial
invalidation so the cost and qualification claims remain honest under accepted
OS/filesystem debt.

Source-provenance hashing is preparation work. Complete it under the original
ingestion deadline before freezing the shared preparation charge; recheck that
deadline before the first fit. Each later model seed retains its independent
per-run budget including the fully measured shared preparation, rather than
inheriting the first seed's elapsed fit time.

Participant cleanup has one shared owned task. Reject new operations immediately,
but join that task through repeated cancellation and concurrent/later close calls
without resetting its first deadline. Reap the process and reclaim its I/O thread
and streams before returning or rethrowing the original operation error. Preserve
the first caller cancellation even if cleanup subsequently fails, while recording
that cleanup failure. This implements ADR0004 ownership for native preflight; it
does not introduce a new supervisor framework or hard real-time guarantee.
