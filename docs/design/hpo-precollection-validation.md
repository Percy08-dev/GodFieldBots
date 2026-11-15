# Resident HPO precollection validation reuse

Status: accepted preimplementation design, 2026-11-15. Root design was reviewed
by `bot_architecture_reviewer`: READY, including admission-binding and cleanup
clarifications. Implementation and final acceptance remain pending. Baseline
`050713d`; see the [investigation](hpo-validation-reuse-investigation.md) and
[measurement](../verification/ppo-runtime-profiling.md).

## Scope and authority

Reuse historical validation across only the first three resident HPO collection
checks: coordinator entry, `collect_hpo_ppo` entry, and `_run_corrected` before
output-directory creation. Calls four (after mkdir) and five (after games) keep
complete validation. Initial full validation remains the complete original
Source/numeric/prefix verification; a structural reader is not a substitute.
No dataset/checkpoint/wire/diagnostic schema, learning math, native Rust code,
dependency, policy input or Simulator integration changes.

Use the caller-owned `_ChainScan` supported by `validate_hpo_parent(_scan=...)`.
The retained scan covers parent controls, artifacts, original numeric files and
inventories, and actual packed original owners. Normal Pack checks retain
ADR0035's demand-verification guarantee, not a claimed whole-DB rehash. Native
authorization scans, live sessions, projections and acknowledged-save permits
are never reused. Add explicit current native module/capability/ELF coverage;
the historical native owner's private binary scan does not transfer itself.

## Responsibilities and interfaces

- New `gf_training.hpo_precollection` owns one private concrete
  `_PrecollectionValidation`, lifecycle/recheck/cleanup helpers and no CLI imports.
  Use local imports/TYPE_CHECKING where needed to avoid circular dependencies.
- `_OnlineGameOwner.precollection_validation` holds the actual carrier. Bind the
  exact online/RunOwnership owner, process/thread, parent path, original plan
  object and content, ledger/job, generation/batch, cap and deadlines. Register
  NEW before allocation/validation side effects. Bind the exact original
  admission and its epoch at the first historical read, after fresh phase
  checks, not before `_HpoOperations` creates that admission.
- `_HpoOperations.precollection_scope(plan, ledger, *, config, deadline)` is a
  synchronous context manager around call one and `await collector`. It is
  enabled only by the internal `train_hpo_ppo(..., _reuse_precollection=False)`
  choice plus the actual resident HPO runtime. Standard CLI composition opts in
  for spec3/resident only. Direct/custom collectors default to the old path.
- `card_instance_source.validate_owned_collection(...,
  _finish_precollection=False)` keeps its original fresh guards and return type.
  It delegates only the historical read to an attached actual carrier; the
  initial read runs `validate_hpo_parent` with that carrier's scan. Subsequent
  reads recheck still-owned evidence and checked-fact fingerprints.
- The CLI `_validate_online_collection` forwards the private finish keyword.
  Only call three requests finish. After ALL current validation succeeds, perform
  final recheck/actual close and detach before returning to the mkdir statement.
  Close failure blocks mkdir. Confirmed expiry is terminal for that scope; calls
  four/five cannot create a new carrier. No public CLI/config switch is added.

## Fresh checks and dependency coverage

Every invocation retains current owner/lock/thread/PID/job/ledger checks, learner
process/idle/lease/hardware/model/execution/steps checks, admission or parked-phase
checks, exact plan/schedule/bindings/state/limits and monotonic `last_unix`
comparison. Existing CLI loaded-config/input checks and runtime-evidence
preflights remain. While a carrier is active, its exact current output path must
still be absent. Same-object binding plus frozen content prevents altered or
copied state from standing in for the original facts.

On reuse, use `recheck_game_parent_snapshot` on the actual outer scan. Tighten
deadlines only; never extend them. Preserve the original native reader/model/
mixed-opponent capability checks and the loaded binary identity, with the ELF
sealed in retained evidence and initial/final identity bracketing full build.
Use existing native helpers; do not add repeated source-tree global hashing.
`SealedScan.finalize` rehashes physical files and checks inventories/stat identity;
it is not a stat-only shortcut. Altered checked state/records must reject.

## Lifetime, memory and failures

States distinguish NEW/building/ready/closing/closed/failed. A building, poisoned,
failed, copied, foreign, closing or expired attached carrier cannot fall back to
full validation. Only the ordinary non-opted-in path, or confirmed normal expiry,
uses the original full validator. No global/hash/path/TTL cache is introduced.

Charge carrier/native/fingerprint metadata before allocation within the original
Source cap: `min(training.max_owned_bytes, source_budget_bytes,
preprocessing.memory_budget_mib*1024**2)`. Keep packed pool/control charges real;
pool budget usage already includes admitted scan controls, so never count that
subtotal twice or refund it while live. Parent config/bindings/retained policy
artifacts keep their existing parent envelope. Before actual collector allocation,
the overlap is parent + learner + resident + Source, bounded by the existing
admitted update envelope (which additionally includes numeric). Prove this with
actual accounted reservations. Source custody closes before a new collector Pack
is allocated; no two Source/collector pools overlap. No budget is raised and no
parent envelope is silently reduced to hide accounting.

Do not retain decoded rollouts/arrays, native sessions/providers/executors or
active loans after the historical read. Keep the actual original FDs/metadata
until expiry. Release/detach graph references only after physical close is
confirmed. Uncertain owners/accounts stay reachable from the actual owner and
initiating error. Preserve primary errors and cleanup notes.

The outer scope must close evidence even if validation one fails or the collector
fails before its own try/finally (including runtime evidence and retained registry
construction). Error cleanup precedes resident shutdown/reference mutation.
Retirement uses the original cleanup allowance, separately from work deadlines,
and must not require a failed learner to remain live. Final HPO termination
requires no surviving carrier/uncertain resource. Failure yields no data or
publication authority. There is no cross-generation/commit/retention/pause/resume
reuse; same-code resume builds fresh evidence under its original deadlines.

## Verification and acceptance

Test initial3->1 complete validation with fresh checks each time, normal expiry
before mkdir, calls4/5 full, no restart, and custom/default/fresh compatibility.
Cover wrong/copied owner/plan/admission/job/generation/thread, expired deadline,
modified originals/config/ELF/numeric/inventories/reference/retention, corrupted
cached facts, partial construction, cancellation, recheck/close/refund failures.
Use an actual native G1 original to show provider/executor retirement with retained
outer numeric/packed mutation detection and eventual exact resource retirement.
Keep existing learning/Source/save/receipt contracts and fixed math ASTs.

Run focused regressions/static checks then the whole CPU suite once stable.
Independent correctness and architecture-conformance review follow implementation.
Repeat matched fresh-directory baselineFalse/candidateTrue measurements using the
same candidate code and workload: models192/main2,4workers,16games per collection,
2updates; three MLP pairs and one pair each GRU/Transformer. Freeze seeds/config,
code/native/Simulator/schema identities, execution flag, limits and order before
runs. Separate full validation, recheck, close and collect wall time with thin
instrumentation. Validate actual original Source/numeric and termination/reference
evidence for each accepted run; preserve and exclude failures from performance
aggregation. Require MLP median second-collect improvement,3->1 precollection full
validation, unchanged bounds and healthy ownership. No fixed percentage saving,
bitwise independent-training equivalence, formal HPO or strength claim follows.
