# HPO coordinator, phase budgets and reference transactions

Concrete unit4 design under [the records contract](ppo-hpo-records.md), proposed
by the runtime implementer and accepted by root. Scoped architecture review is
READY with no blockers before code. Implementation stays in the isolated runtime worktree;
native3b and outer campaign work use separate ownership.

## Composition and run identity

CLI owns `materialize_hpo_run(loaded_campaign,effective_settings,*,learning_seed,
run_id)->LoadedConfig` and the concrete packed `HPOExperimentConfig`. Materialize
the normalized effective recipe/model4, sole hpo_spec and checked runtime PPO
projections; never introduce another runtime framework or pass campaign data to
the participant policy. Source4a's parser remains the record authority.

Use the supported colocated numeric layout and2GiB collector default specified
in the records design. Preserve actual base Pack queue/Pack/budget settings and
require them to fit declared collector RAM; resolve default originals to the
existing supported D root. All bad layout/budget combinations reject before GPU.
Run timers follow immutable hpo-deadline-v1 and original start, not session time.

`train_hpo_ppo(loaded_run,collector,directory,*,gpu_job,mode="fresh",
pause_after_update=None)->HpoRunResult` requires the caller's already-durably-
admitted actual `GpuJob`. Check original capability, current training context,
predeclared run UUID, limits and SAME audited ledger. Fresh Run5 uses that UUID;
resume retains it and checks config/source/execution/hardware/deadlines. Do not
admit a second job or construct a second ledger. Source accepts the known audited
ledger only through this exact HPO owner/job binding; old exact guards remain.

`HpoRunResult` is frozen and carries the ordinary JSON report plus the actual
capability-issued `TerminationReceipt`. The inner finalizer confirms only after
actual learner, Source/custody and collector/transport/storage owners close, and
before publishing healthy final Run5 state. Outer code retains/closes the flock.
Unknown cleanup leaves the ticket unconfirmed/RECOVERY_REQUIRED. A failed final
write or failed run yields no objective; a GPU termination receipt is never data
qualification. Preserve original exceptions and keep failure evidence.

## One common coordinator

Use one concrete `_HpoOperations` variant of `_GameOperations` feeding `_run_ppo`.
Small known HPO methods/branches own UUID/registry versions, schedule counts,
controls, used/collected counters, pool thresholds, stop conditions, full save/
commit and finalization. Preserve all eight fixed ASTs and every old path.
Reuse Source-issued handles, Numeric5/Wire6, current runtime evidence and actual
process/FD ownership; no duplicate loop or callbacks registry.

Stop at used==D before scheduling another batch. Normal time/admission/storage/
update-cap stops belowD are incomplete. Preserve an empty natural collection but
do not perform an empty optimizer update or invent a transition. Actual execution,
save/storage overrun and uncertain cleanup remain failed/quarantined. Each
successful update saves/verifies complete inference/resume before retention.

## Actual phase limits

The original numeric/save envelope settings cannot be mere joint-admission
numbers. Add HPO-only closed Wire6 identity metadata:

```
phase_limits = {numeric_budget_bytes: positive_int,
                save_budget_bytes: positive_int}
```

The sole hpo_spec operating settings and full resume identity bind them. Production
always supplies them; honest controlled component APIs may use explicit bounded
defaults. Old Wire3/4/5 schemas and old model-resource semantics remain exact.
Charge each phase's additional allocations without charging the complete retained
Source against only the numeric envelope or shrinking the learner model budget.

Source write/read/numeric views use the SAME declared numeric limit with existing
reserves, metadata/array preallocation and decoded-size guards before decoding or
optimizer side effects. Source originals retain their separate Source envelope.
HPO save/inspect uses explicit additional save-envelope admission through existing
artifact/shard helpers. Supply a pure model/Adam/full-state save estimate so
obviously insufficient save settings reject before GPU; actual copying/writing/
inspection still enforces the bound. Do not shrink model.max_owned_bytes to512MiB
or borrow undeclared RAM to conceal the gap. No second resource ledger, numerical
algorithm or physical storage format is introduced.

HPO-only worker/planned admission helpers reuse current cost formulas, allow the
declared CUDA technical domain and keep initial<=2games/<=4workers. Align pure
config, runtime, recorded-state and audited-ledger limits. Old12GiB/4 helpers stay
unchanged. Conservative managed estimates remain distinct from measured usage.

## Generation commit and owned retention

Under the SAME parent lock verify actual paired inference5/resume5, full recipe/
resources/steps and immutable file maps; perform Source prepare-commit/rechecks
with original custody and verified decoded-retirement facts. Write Update5,
Generation4, Registry4, latest and Run5 through owned durable HPO publication.
Factor existing validators instead of duplicating semantics. Partial publication
cannot invent a successful update or repair arbitrary files.

Share one pure protected-file calculation between Registry4 reader and retention:
G0 full, newest>=2resume plus inference/weights closure, inference latestN,
used-decision periodic crossings, current pool and evaluation references. Reuse
the existing verified single-file stage/move/hash/unlink primitive. Commit exact
per-file deletion intent under lock before deletion; synchronize namespace changes
before successful receipts. Foreign/replaced/new files and uncertain unlink/fsync
never become successful deletion. No recursive cleanup or source-episode pruning.
Existing generic publication only fsyncs file data, so HPO adds a narrow owned
publication/ancestry-sync wrapper; old persistence behavior remains unchanged.

## Exact evaluation-reference transactions

Binding is `{run_id,token,generation,inference_sha256,generation_sha256}`. Validate
the exact committed generation and retained inference/weight dependencies under
the SAME parent lock before pin success. Expose pin/release helpers to outer
evaluation, never an absent-reference no-op that guesses prior release.

Keep Registry4's active-reference schema unchanged. Add a bounded optional HPO
control directory `evaluation-reference-receipts/<token>/` containing four small
immutable records: `acquire.intent.json`, `acquire.receipt.json`,
`release.intent.json`, `release.receipt.json`. Intents have closed schema1/kind,
operation UUID and exact binding. A release intent also links the completed
acquire receipt. Completion receipts have closed schema1/kind, exact binding,
intent SHA, confirmed Registry4 hash and confirmation time. These are control
records, not a model registry or training artifacts.

Ordering is durable intent -> checked Registry4 mutation and directory fsync ->
durable completion. Replay under lock requires the matching intent/receipt plus
the actual intended Registry4 state. A missing reference without matching release
intent/receipt never proves success. A completed released token is terminal:
acquire cannot pretend it is active again; new evaluation uses a new token.
Completed release replay verifies exact binding and absence. Mismatches/foreign
records block. Reconcile only this concrete transaction and preserve incomplete
evidence; do not introduce a general transaction framework or rewrite run clocks.
The confirmed Registry4 hash is historical receipt evidence: unrelated legitimate
registry mutations do not prevent exact token/binding/intended-state replay.
Pending reference records alone grant no evaluation or update authority. Their
effect on structural namespace/protection validation must be explicit and bounded.

## Scope and verification

Runtime worktree owns HPO coordinator/lineage/retention, CLI materialization/core
collector dispatch and necessary HPO Source/Wire/artifact phase-limit seams.
Outer evaluator owns new evaluation composition/role metadata and calls the core
collector; it does not separately edit the same collector loop. Native training
continues to accept only its training role. All cross-worktree interfaces are
integrated/reviewed before actual GPU/Simulator acceptance.

Verify D/all/partial/one/empty and incomplete caps; immutable resume clocks;
same-input actual saved state continuous/resume; exact actual job/ledger and
foreign/expired refusal; explicit phase-limit preallocation failures; full commit
and per-file delete faults; G0/latest2/pool/evaluation dependency closure; pin
intent/mutation/receipt crashes and exact replay/terminal tokens; old Config/
source/kernel compatibility. CPU fixtures remain honestly unqualified. Real GPU,
Simulator and native pipeline smokes remain required at final integration.
