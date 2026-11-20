# Fixed HPO training-operation launcher for accepted experiments

Status: accepted bounded design; independent architecture R1 READY on proposal
1ea93aac82b7b2b8cc9d8201c215c88cd46a490373f94906dcffba51e881dd4d. Root
owns scope/recipes/configuration/acceptance. Implements the already accepted
operation-timing interpretation in rust-player-runtime-hpo-operation-timing.md, before
any Task7/calibration/profile data. No new public gf-bots command, search algorithm
or campaign/final-D claim. Ordinary observations remain actual gf-bots train.

## Responsibilities and dependency direction

Add one fixed experiment command (tools module, not a persistent controller)
which accepts an actual HPO config path, declared canonical run/experiment IDs,
and existing fresh/resume/pause controls. Configuration loading/materialization
and every setup/import/admission action occurs INSIDE the measured command. Only
paths/declared identifiers are prepared outside, never a live owner or warmed
model. It calls the same canonical materialize_hpo_run with the configured
EffectiveSettings and declared learning seed, not canonical_control or a forked
sampler. Fixed scaling settings remain32games/two updates/D10000 and truthful
healthy-incomplete. No outside-control failure is fabricated or pre-staged.

Tool depends on existing CLI configuration/collection and actual training APIs.
Training never imports the tool or CLI. Only schema5 shared-MLP legacy and schema6
Rust Model4 MLP experiment profiles are supported; actual default configuration
readers/selectors/old checkpoints keep their meanings. Scheduler/W/C/B/wait and
resource values are declared in the normal config, not hidden script overrides.
No callable/module/executable string dispatch from data.

Legacy operation uses actual admit_gpu_job(JobContext(...)) and its real ledger,
then actual train_hpo_ppo with the same collector closure as _train_seed: require
the exact job ledger and call actual collect_hpo_ppo. The existing routine owns
RunOwnership/Source/update/save/cleanup and confirms same-job termination; caller
positively closes any actual returned completion and the job scope. Incomplete
HpoRunResult is reported truthfully, rather than calling the objective-only
_train_seed wrapper or constructing a completion from JSON.

New operation uses a narrow internal prepare_hpo_operation counterpart to the
existing ordinary factory. It accepts canonical materialized HPO6 LoadedConfig
DATA plus actual JobContext metadata for this standalone operation and a stable
experiment-control directory path. It OWNS a real CampaignStore there and the
actual parent run_lock, not a shape-equivalent lock claim. Both actual registered
SAME OFDs are sealed into launch under existing campaign and parent lock kinds.
Use canonical stable effective snapshot/operation2/clock1, actual verified
companion/build and fixed venv interpreter. Derive fixed hpo-train kind
and expected_context from these checked inputs. No fake global job is created;
the Rust supervisor's real HostAdmission performs actual global admission before
CUDA. Pure/native identity/startup checks and actual BUILD metadata occur before
the original operation anchor; all are still within measured launcher wall.
Capture fresh original anchor BEFORE actual experiment Store AND parent lock
acquisition/publication. Resume reuses old bits without recapture. Native's
unconditional HPO campaign-lock requirement remains unchanged.
The Python launcher EXECs the supervisor and never survives as another Python
training controller. Existing fixed run_hpo/finalize_cold and scalar result/
positive release/control-close flow are reused. Share ordinary factory mechanics
privately where needed; retain its existing interface, behavior and factory-only
launch facts. No generic runtime framework or duplicate learning path.

## Genuine experiment namespace custody

Native check_locks(true) requires a real campaign-kind lock even for initial
hpo-train, although that purpose does not create CampaignBinding/continuation.
The fixed launcher uses the EXISTING actual CampaignStore as an owned operational
control namespace under a deterministic per-run path separate from the training
parent. It does not call initialize_campaign, create a fake hpo-campaign manifest,
CampaignStudy, proposal, seed-training receipt or objective. JobContext's campaign
identifier denotes this actual exclusive experiment namespace, not a scientific
sampler claim. Any experiment identity/config metadata written there is explicitly
operation-control DATA, with normal immutable Store semantics.

Acquire actual Store ownership fresh, or resume that SAME stable directory under
real exclusive flock/PID/thread/directory/file identity checks. Native preparation
owns the Store context itself (caller supplies only path DATA), so ownership is
inside original clocks and measured wall. It preserves the actual registered
descriptor through exec along with the real parent descriptor. Successful exec
never executes Python context finally/LOCK_UN; the supervisor retains aliases
through actual training/cold retirement and exits. Failed exec closes actual
owners normally; unknown close retains/quarantines under existing contracts.
No reopened/independent lock is substituted and no native admission check relaxes.

Legacy standalone execution acquires the same real per-run operational Store
inside measured setup, holds it through actual core/job/completion retirement
and positively closes it before command exit. Its existing training clock recipe
and actual job/context/ledger remain unchanged. Native resume must reacquire the
same actual experiment namespace and parent, validate its recorded identity/config
as DATA, and keep EVERY original training deadline/producer/boot/root. Actual
Store custody supplies the OFD; persisted identifiers never supply authority.

Verification adds real independent-open competitors and sealed launch inspection
for BOTH lock kinds, failed-exec cleanup and original namespace reuse. Genuine
actual HPO admission/retirement is a later bounded Task6 check.

## Timing, resume, failure and information

The standalone Rust observer starts the original command and records its actual
Child.wait. Whole operation wall includes all launcher setup/materialization/
imports/admission/train/save and the method's required retirement/finalization.
Native mandatory cold finalization stays INSIDE this wall; none is subtracted.
Legacy normal operation cleanup stays its real existing route, never a fake Rust
Prepared/cold receipt. Independent post-command DATA verification is separately
reported outside measured wall and begins only after actual original termination;
it cannot replace a method's required internal finalization. Match observer and
application diagnostic recipes exactly across old/new. Root JS/shell orchestrates,
with no Python benchmark parent remaining during new-mode training.

New pause/resume uses same actual parent/run ID/config/producer/boot/root and EVERY
original deadline bit, as ordinary; no fresh anchor or paused-time refund. New
operation ID/context is distinct. Full real RustResumeAudit remains before CUDA.
Old legacy resume retains its recorded existing semantics. The launcher is also
a fixed controller for the separately required Task6 HPO continuous2/pause-resume
controls; it never substitutes for the real public campaign2operations/evaluation.

Failure retains actual owners and original errors until positive cleanup. An
unknown job/FD/process remains ineligible/quarantined under existing contracts;
no PID-absence recovery, timeout win/loss, unconfirmed namespace release, extra
clock/grace or restarted failed parent. Ordinary/private policy input boundaries,
row-local inference, RNG transaction and all mathematical kernels stay unchanged.

Diagnostic output records exact config/input/producer/native/tool/observer hashes,
actual context/termination/completion links, statuses and metrics without issuing
Source/Parent/seed/objective authority. Standalone IDs do not claim a sampler
proposal or real campaign. No Source is synthesized from these result records.

## Verification and remaining integration

Before actual cases: pure/config/file tests for matched materialized old/new
recipes, actual OFD/sealed launch/fixed interpreter/mocked finalexec, missing or
changed companion, no surviving launcher in exec path, original resume clock
refusal and unchanged legacy routine/collector/job wiring. No actual GPU/game
workload is authorized merely by this design review. Both code reviews and final
matching wheel precede Root's bounded actual Task6 execution.

For each accepted formal operation verify exact games/updates/status, true whole
Child start-to-wait, actual one Python during native training, Original handles
and full independent Raw/Replay/numeric/saved-state audits. Existing legacy
validate_hpo_parent(require_complete=False)/audit_owned_samples are DATA-only
audit surfaces, never exact-D completion. Native cold already performs its full
independent audits; readback verifies those actual receipt chains. Pin the exact
full saved-array/optimizer/RNG decoder and post-command finite audit budget when
implementing verification, preserving all complete-only guards. No structural-
only audit is mislabeled full saved-content validation. All24 primary, maximum12
calibration and maximum5 profile runs retain their predeclared order/workloads/
budgets/thresholds; this launcher adds no performance population or default claim.

Implement only after C2c fixes/code acceptance; tool and entry helper file ownership
will be assigned explicitly, isolated from the full campaign/eval implementation.
Final scientific/public Task6 and performance/default acceptance remain separate.
