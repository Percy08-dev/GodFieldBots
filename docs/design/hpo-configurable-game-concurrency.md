# Configurable HPO game concurrency

Date: 2026-11-18. Owner: root. Baseline: `b61a603`.
Status: implemented and accepted after independent correctness/conformance
reviews READY. Initial omitted resident custody/receipt bounds are resolved by
explicit original-capacity propagation below. Exact checks and limits are in
[the verification record](../verification/hpo-game-concurrency.md).

## Requirements and scope

The owner requests more than two simultaneous PPO HPO games and explicitly
accepts responsibility for reducing settings when increased parallelism causes
OOM. Permit caller-selected concurrency in the existing HPO configuration;
do not choose a new machine-specific concurrency ceiling or promise that any
accepted setting fits physical RAM/VRAM. Actual learning remains manual.

Keep the default at two games/four participant workers and the trial concurrency
at one. Keep the current mixed opponent batch at sixteen games; batch size and
simultaneous games are separate settings. Do not change the simulator, model
math, policy-visible information, original storage paths or frozen formal recipes.

## Approaches and decision

1. Raise the fixed ceiling to four/eight. This meets one example but repeats the
   arbitrary restriction when the caller has more capacity.
2. Accept positive signed-63-bit HPO counts, retain explicit worker and managed
   memory admission, and derive resident partition capacity from the actual
   caller-owned ledger. Choose this bounded change: it uses the existing owner
   and reservation paths without creating a second resource system.
3. Remove admission and retry with fewer games after OOM. This changes experiment
   conditions and failure ownership; the user requested responsibility for
   tuning configuration, not silent retries or unaccounted processes.

The owner has authorized the configurable-count change and OOM responsibility.
The implementation choices above are root's decisions within that request.

## Responsibilities, interfaces and flow

`gf_cli.hpo_config._operating` owns TOML operating validation and per-run
materialization. Its `game_concurrency` and `max_participant_workers` values must
be exact integers in `1..2**63-1`, with workers at least twice the game count.
Existing `_operating` defaults and the closed table remain unchanged.

`gf_models.structured_runtime` retains the shared formulas. Only its HPO entry
points accept the expanded count domain. Training, collection, cache staging,
evaluation and comparison continue multiplying worst simultaneous roles by the
declared concurrency and comparing against caller-configured RAM/CUDA budgets.
No parameter is reduced automatically. Failure of managed admission is still a
configuration error; managed admission does not guarantee physical fit.

`gf_training.hpo_lineage.hpo_spec` and `native/src/hpo_controls.rs` apply the same
count domain and worker relation when validating persisted HPO specifications.
Run projection still requires exact equality to the sole HPO specification.
The fields, schemas, identities and physical layouts are unchanged. Existing
two-game HPO records remain valid; no migration or historical rewrite occurs.

`gf_training.hpo_gpu_lease._AuditedLedger` overrides the inherited four-worker
ceiling for HPO only, as it already overrides the legacy CUDA ceiling. Admission
and actual `reserve_worker` use that same declared maximum. The ordinary
`ReservationLedger`, ordinary `gf_cli.config` and legacy/formal admission stay
at their current limits. HPO materialization and lineage have their own checked
run projection; they do not pass through the ordinary experiment TOML reader.

`gf_simulator_client.resident_pool.ResidentParticipantPool.acquire` replaces its
fixed two-owners-per-viewer/four-total limits with `ledger.max_workers // 2`
per viewer and `ledger.max_workers` total. This preserves disjoint participant
views and reserves capacity for both roles. Odd extra worker slots may remain
unused. The existing scheduler's semaphore controls actual simultaneous games;
requested concurrency larger than a collection does not invent more games.
Do not add a second pool limit, cross-view owner reuse or speculative scheduler.

`gf_training.hpo_resident.ResidentPoolCustody.register` applies the SAME ledger's
`max_workers // 2` viewer limit. Actual total registration remains bounded by
ledger worker reservations, with every owner still checked against that ledger.
`_evidence` validates its complete live owner set using `ledger.max_workers`.

`gf_common.hpo_resident.checked_retirement` gains an explicit keyword capacity
argument `max_workers` (default four only for existing standalone callers).
Validate it as an exact positive signed-63-bit integer; require a nonempty owner
set of at most `max_workers`, and at most `max_workers // 2` per viewer. Keep all
existing unique-owner, ACK digest/epoch/reset, job/pool and generation checks.
The receipt gains no self-declared capacity field. Every production HPO reader
must pass capacity from its already-bound original configuration or live ledger:

- `hpo_collection.check_hpo_collection`: manifest's hashed HPO configuration;
- `hpo_lineage._proof` and `native_hpo`: validated original parent HPO spec;
- `card_instance_collection._checked_evaluation_metadata`: hashed evaluation
  configuration, whose operating and run worker limits must agree;
- `native_structured_session._payload_views`: an explicit optional keyword
  `max_workers=4`, supplied for HPO by `native_card_instance._publish_packet`
  from its already-authorized summary parent state. Never derive it from the
  exported receipt. Keep the existing exact source-proof comparison afterwards.

Manifest readers must validate positive capacity and equality between their
run and HPO operating projections when obtaining this bound. They do not gain
Source authority merely by passing structural receipt checks. Do not import
CLI/training into the common domain; existing configuration fields suffice.

`native/src/hpo_resident.rs::retirement` likewise receives explicit expected
capacity; `hpo_controls` passes it from the already-validated parent HPO spec.
`native/src/card_instance_source.rs::manifest` also passes that expected capacity
from its original parent state, preserving the later exact manifest/parent Config
comparison. It must not take capacity from the manifest's receipt or bypass the
original parent check. This is a second existing Rust caller of the same reader.
Rust standalone test callers pass four. Use wide integer arithmetic for limits
and counts, retaining duplicate/viewer/ACK validation. No unsafe truncation to
small counters or unchecked multiplication. This changes internal function
parameters only, not wire or persisted schema fields.

Existing runtime composition passes the declared counts to collection and final
evaluation. The managed-resource result must describe the expanded counts;
tests must establish that the actual HPO route reaches beyond two concurrent
games/physical owner pairs rather than accepting configuration alone.

## Failure ownership and compatibility

Worker creation, viewer partitioning, pending creation, I/O retirement, cache
pins, parked update authority and positive reap retain their current owners.
All actually created workers remain charged through actual retirement; increasing
their count cannot make an empty pool/PID check into termination evidence.
OOM, cancellation and resource refusal remain failures with no fabricated game
outcome, objective, successful checkpoint or automatic fallback. Keep the global
one-job admission/exclusion/recovery contract and immutable original receipts.

Use the current simulator revision `91d5926a7572db327eaea3751e7c24f428120996`
and current HPO schema4 producer-owned/resident route; no simulator API or rule
contract changes. Existing native identity/preflight checks remain required.
Rebuild and activate the matching installed native extension after the Rust
validation change, and verify its build source digest against current source
during acceptance; never weaken identity checks to accept a stale extension.
Ordinary resume requires original code/config identity; do not modify a running
or paused campaign to apply new concurrency. Fresh invocations use new campaign
paths when any completed campaign exists.

No dependencies or new persisted formats are introduced. Only the HPO validator's
accepted value domain is expanded; older readers may reject newly larger counts.
Old evidence retains its recorded conditions and does not qualify new settings.

## Verification and acceptance

- Tests first: defaults, 4/8 and 8/16 game/worker settings through campaign parsing,
  materialization, HPO lineage/native specification readers and pure resource
  admission. Reject booleans, nonintegers, zero/negative/overflow counts,
  insufficient worker slots and trial concurrency above one.
- Verify scaled residency/cache/update/save accounting and unchanged refusal of
  declared RAM/CUDA overruns. Verify ordinary/frozen two-game limits remain.
- Exercise the existing scheduler/pool with at least four simultaneous fake or
  CPU fixture games, including eight physical role owners, disjoint viewer
  partitions, reuse/park and complete retirement. No actual simulator games or
  training. Inject a failure/cancellation above four workers and check that
  pending/live owners cannot produce termination before positive retirement.
- Pass an eight-owner END/PARK receipt through Python and Rust with expected
  capacity eight. Reject it under capacity four, excess totals, per-viewer
  imbalance, duplicate owners and stale/foreign ACKs. Verify that both training
  and evaluation metadata readers propagate the bound, including native payload
  export and original Source/lineage receipt checks.
- Focused Python HPO config/admission/lineage/GPU lease/resident/scheduler checks,
  Rust HPO controls/regressions, scoped Ruff, `mypy src`, `git diff --check`.
  Run source-matched native preflight and current preset validation. CPU fake
  evidence is functional only, not actual CUDA capacity or a throughput claim.
- Independent correctness and architecture-conformance reviews must be READY.
  Root records exact commands/results, changes current architecture and ADR0049,
  and commits only the verified scope. Real larger-concurrency HPO remains NOT
  RUN because the owner performs experiments manually.

## Documentation and configuration

Document `game_concurrency`, `max_participant_workers >= 2 * game_concurrency`,
batch-size distinction, managed-budget checks, manual OOM tuning and fresh-run
requirements in `docs/hpo.md`. Make the existing 2/4 defaults explicit in all
three current budget-probe presets so the caller can edit the two fields;
retain all current paths, scientific settings and default behavior. Show a 4/8
example. Current4/8 pure admission needs about15GiB of managed CUDA, above the
default12GiB; the example therefore explicitly sets16GiB without claiming
physical fit. Keep actual preset defaults unchanged. Do not claim a
hardware-independent safe count or speedup.

## Open questions

None requiring owner input. The initial design blocker was the omitted custody
and structural-retirement ceiling; root accepts that finding and adds explicit
original-capacity propagation above. Review must confirm every active HPO Python/native
validator and physical pool limit is included; discovered additional boundaries
return to root before implementation expands beyond this design.
