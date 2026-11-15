# Investigation: reuse of historical HPO validation during collection

Status: research only, 2026-11-15. Inspected Bots revision `314d510` on
`feat/heuristic-refresh`. No production implementation or implementation-design
approval is recorded here. No learning workload, GPU job or test was run for this
investigation. Simulator integration and persisted formats are unchanged.

## Objective and measured opportunity

Reduce repeated historical Source reconstruction inside resident HPO collection
while preserving original-data integrity, live ownership, phase authority,
resource admission and failure handling. This follows the
[collection measurement](../verification/ppo-runtime-profiling.md#inside-collection-five-repeated-parent-validations).
The observed second collection spent 29.130 of 43.731 seconds in five complete
parent validations; this is one measured MLP point, not a forecast of savings.

The relevant accepted contracts are [ADR0035](../adr/0035-pack-checksum-read-policy.md),
[ADR0036](../adr/0036-packed-decoded-source-retirement.md),
[ADR0037](../adr/0037-ppo-hpo-foundation.md), and
[ADR0041](../adr/0041-hpo-resident-workers-and-progress.md), together with the
[HPO runtime](ppo-hpo-runtime.md), [records](ppo-hpo-records.md), and
[acknowledged native save](ppo-hpo-native-save-transition.md) designs.

## Call sequence and intervening changes

All five sites reach `card_instance_source.validate_owned_collection`, which
combines live checks with `hpo_source.validate_hpo_parent`. The latter opens each
committed batch, compares its original proof, reads its numeric rollout and
checks selected-prefix compute counts. At G1, every call replays G0; at G0 this
historical loop is empty.

| Call site at inspected revision | Position | Changes before the next call |
| --- | --- | --- |
| `ppo_run.py:685` | Before collector callback | In-memory collection controls/admission may be initialized; no observed durable history change |
| `card_instance.py:753` | `collect_hpo_ppo` entry | Runtime evidence/preflight and retained checkpoint reads |
| `card_instance.py:314` | `_run_corrected`, before setup | Exact current collection output directory is created at line 320 |
| `card_instance.py:357` | Owned GroupPack path selection | Current Pack/database creation, gameplay, group archives, durable cache-pin acquisition/release and receipt changes |
| `card_instance.py:495` | After groups retire, before publication | Collection report/manifest/locator publication |

These are the same online parent owner, plan and ledger. Resident phase evidence
changes from collection admission to its completed parked authority. The parent
`run.json`, committed generation files and update files are unchanged during
these five calls. Numeric/update work follows collector return; postcommit SYNC
is later still. Registry5 `cache_references` and its receipt namespace can change
during games even though committed history does not.

Consequently, the first three calls form the smallest candidate reuse interval.
The fourth is already after a filesystem mutation. This is a conservative
lifetime choice; it does not assert that the current parent scan inventories the
new collection directory. Reusing across that boundary needs an explicit
argument for its exact allowed mutation. Reusing across gameplay additionally
needs mutable registry/reference validation.

## Existing mechanisms and their coverage

`validate_hpo_parent(..., _scan=scan)` already accepts a caller-owned `_ChainScan`
and leaves it open. Without that argument, it creates and closes its scan before
returning. A returned `state, records` pair alone therefore cannot stand in for
still-owned validation evidence. Current `online.collection_controls` is only a
`SealedScan` of the server TOML; it does not retain historical validation.

| Evidence | Existing retention and recheck | Consequence for reuse |
| --- | --- | --- |
| Parent controls, artifacts, generation/update chain | Caller `_ChainScan`; exact file observations, artifact and generation/update inventories | Retain the actual scan and bind its checked facts to the exact owner/plan/generation |
| Committed numeric files | `hpo_lineage._update` seals `rollout.json`, exact numeric inventory and `scan.bundle` in that same caller scan | Physical numeric coverage exists even for the native branch; do not retain decoded arrays just to preserve it |
| Packed originals | `_open_batch` registers committed collections in `controls._packed_owners`; `_ChainScan.finalize` calls each actual `PackedSourceOwner.finalize` | Native `source.snapshot is None` does not by itself lose packed physical custody |
| Native provider/session | `OriginalArtifactProvider.close` retires adapter metadata and I/O, not independently scan-owned original FDs; executor closes first | Historical native sessions need not stay live; they must actually retire before their pool may close |
| Native binary | `_NativeGameInputs._binary_scan` separately seals the active extension; not merged into the outer scan | Explicitly retain or freshly establish equivalent binary/capability checks; outer scan alone is insufficient for this dependency |
| Reference transactions | Parent scan seals registry bytes and evaluation/cache receipt namespaces | Owned pin/release changes legitimately invalidate the old snapshot; blanket refreshing would conceal unrelated changes |

Source references: `hpo_source.py:159–235`, `hpo_lineage.py:916–956`,
`structured_lineage.py:74–305`, `card_instance_source.py:1101–1208`,
`native_group_pack.py:112–118,442–463`, and
`native_card_instance.py:548–580,663–689,746–760`.

`SealedScan.finalize` reopens and **rehashes** registered physical files before
inventory/post-sweep checks. It is not a cheap stat-only check. Packed boundary
checks follow ADR0035: original FD/path/namespace checks plus demand-verified
Pack acquisition, without claiming a repeated full database audit. Reuse must
preserve those distinct guarantees. Neither mechanism is a persistent trust cache.

The outer scan's numeric file coverage is confirmed by code inspection. Full
equivalence of every native/Python control dependency has not been tested. The
implementation design must enumerate any remaining dependency observations,
including the native ELF seal, instead of assuming the two scans are identical.

## Checks that remain fresh on every invocation

- Actual run lock/owner, caller thread/process, current job/admission and exact
  audited resource ledger.
- Live learner process, idle state, lease, model/execution identity, hardware and
  resource limits; applicable optimizer-step/generation state.
- Exact collection plan, schedule, bindings, current owner state and allowed
  monotonic `last_unix`; original deadlines must never be extended.
- Resident admission or completed parked authority, as appropriate to the
  current phase; previous admission cannot prove completed gameplay retirement.
- Retained evidence's current files, namespaces, custody, liveness and failure
  state, plus the explicitly covered native/configuration dependencies.

This research concerns historical validation only. It does not issue Source,
optimizer, save, commit, publication or termination authority. In particular,
retaining history must not replay a previous issued projection or the native
acknowledged-save permit.

## Candidate changes and recommendation

1. **First increment: reuse across calls 1–3 only.** Fully validate once, retain
   owned evidence, perform the fresh checks above at each call, and expire/close
   it before directory creation. Keep calls 4 and 5 on the original complete
   validation path. At the measured point this targets two roughly 5.8-second
   replays; actual savings require measuring replacement rechecks and cleanup.
2. **Extend through call 4.** Admit only the exact expected current empty
   directory under the same owner/plan, or separate immutable history from
   fresh collection/control checks. Preserve all prior observations. Do not
   clear/rebuild inventories or treat directory existence as publication proof.
3. **Reuse throughout one collection.** Separate immutable history from live
   registry/reference state, with exact reconciliation of actual owned pin/
   release transactions. This targets more replay work but changes more of the
   mutation/failure boundary. It needs stronger design and fault evidence.

Start with option 1, initially scoped to the measured resident HPO path. Avoid a
global/path/hash/TTL memoization mechanism and avoid extending evidence across
generations in the first increment. Existing ordinary/fresh collection and
historical readers must keep their contracts. This recommendation is not an
approved interface or permission to skip current validation in production.

## Lifetime, failure and memory requirements for the next design

Evidence must be attached to the actual owner/plan/ledger and a single collection
generation. Copies, another run/thread/job, a closed owner or failed construction
must not reuse it. Keep compact checked facts and original custody; retire native
sessions, providers, decoded rollouts and active loans. Charge retained controls,
file maps, packed owners and pool metadata alongside the next phase's resources.
Do not increase limits or refund still-live evidence to hide overlap.

Expire before the selected mutation boundary, and always on commit, retention,
pause/resume, cancellation, failure or owner close. Any future extension across
pin/release changes must reconcile only the exact owned transaction; unrelated,
partial or external changes still reject. Preserve original exceptions and
actual uncertain cleanup owners/accounts. A cleanup error must not silently
fall back to another full validation and then report success.

Before implementation, root must specify the concrete carrier/owner, fresh-check
interface, complete dependency coverage, destruction order, memory accounting,
and exact expiry point in a bounded design. Independent architecture review is
required because ownership, cross-module interfaces and failure lifetimes change.

## Required evidence for implementation acceptance

- Delegation/lifecycle tests: one full historical validation across calls 1–3,
  fresh guards each time, original full paths at calls 4–5, explicit expiry.
- Reject copied/foreign/closed owners, wrong thread/job/plan/generation, expired
  deadlines, replaced originals/config/native binary/numeric files, changed
  inventories and stale reuse after commit/resume.
- Detect intervening reference/retention mutations; exercise cancellation and
  failures during construction, recheck and close, including uncertain refund.
- Compare accepted outputs and rejection behavior with the original path for
  resident HPO, while protecting old/fresh paths and Source/update authority.
- Check retained memory, original FDs, native executors, loans and pool cleanup;
  then repeat the bounded two-update measurement with unchanged model, games,
  budgets and timers. Attribute validation/recheck time separately. No strength
  or formal-HPO conclusion follows from this performance test.

Existing targeted tests include
`test_card_instance_source.py::test_original_mutation_during_projection_rejects_final_guard`,
`test_card_instance_source.py::test_live_collector_guard_rejects_copied_owner_and_control_replacement`,
`test_hpo_source.py::test_parent_source_numeric_preparse_phase_and_sealed_file_guard`,
`test_hpo_resident_custody.py::test_only_original_parked_authority_allows_update_with_charged_workers`,
and `test_hpo_resident_contracts.py::test_cache_pin_has_separate_durable_namespace_and_cannot_release_from_json`.
They are relevant regression anchors, not evidence that a reuse design already
passes. This investigation used read-only source inspection by root,
`bot_verifier` and `bot_architecture_reviewer`; no tests were executed.
Independent architecture review of this research report found no factual or
boundary findings on 2026-11-15; this was not implementation-design approval.
