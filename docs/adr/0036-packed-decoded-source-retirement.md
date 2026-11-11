# 0036: Retire consumed packed Python graphs before artifact inspection

- Status: Accepted design and scoped implementation,2026-11-11
- Date: 2026-11-11
- Decision owner: root
- Review: independent architecture READY, including live-metadata clarification
- Acceptance: final correctness/conformance READY,28focused lifecycle checks,
  final686affected checks and fresh Python/native pause-resume parents PASS;
  no budgets raised or checks skipped. Original failed parent remains ineligible.
- Extends: ADR0035 runtime acceptance; no persisted or learning contract change

## Context

The first real checksum-policy Python smoke stops after eight optimizer steps
but before generation1publication. Its unchanged2GiB storage pool holds
1,882,846,298B when resume-artifact inspection requests322,850,000B. Consumed
decoded Source allowances retain1,759,721,337B until the later prepare_commit.
All preprocessing8192MiB/training10GiB/storage2048MiB/queue512MiB/Pack256MiB and
7200/5400/7140/60limits match the earlier accepted smoke. The failed parent stays
immutable/ineligible and cannot be resumed or counted as a published update.

A bounded SAME-original readonly replay, with no optimizer or publication,
explicitly clears consumed records/episodes and uses existing decoded retirement.
Both artifact and complete update validation then succeed under the SAME budget,
ending175,112,292B, while original FDs/proof/file map and group cache remain held.

## Decision

Split only the concrete packed Python Source's consumed decoded graph retirement
from physical custody preparation. The exact issued projection owns one private
lifecycle method called before commit's retained-artifact allocations. Freeze the
verified nonempty decision count at issuance; after retirement use that fact for
unchanged4*ceil(decisions/128)step validation. Before retirement require the SAME
active Source, issued transaction, parent lock/generation/process/learner/lease,
completed update and exact actual optimizer advancement, with no active learner
request/update/workers or group loan. Reject early/stale/wrong-owner/replayed and
failed retirement states. Before retirement, frozen count must match actual data.

Clear records and snapshot episodes before refunding decoded allowances. Retain
the actual manifest/proof/control/file-map graph and physical originals/cache
charges. Extend the existing owner helper with keyword-only retained_bytes=0;
old callers keep their behavior. Compute a conservative remaining live bound
before refund; preserve max(existing safe floor, retained bound) without refund/
readmission. Invalid/inconsistent bounds and uncertain refunds fail with actual
owners, errors and reservations retained. Pin the explicit remaining-metadata
floor on the actual owner until confirmed close; a later default historical
retirement cannot lower it. Successful exported-control reads retain their
existing charged subtotal, including repeated exports and later reads.
An uncertain or partial refund blocks every commit retry. Reject duplicate
retirement calls before any effect or second refund; such a no-effect rejection
does not invalidate an earlier successful, fully verified retirement.

Keep every original/control/numeric/artifact check and physical FD/pool/borrower
retirement at the established prepare_commit boundary before lineage publication.
Earlier decoded retirement grants no commit permission. Native and directory
paths, budgets, wire/expectation/artifact schemas, features/rewards/splits and
learning mathematics remain unchanged. No general memory framework is added.

The reviewed terminal-close clarification closes an owned pool before refunding
its wrapper metadata. Pool close refunds only its tracked base; nested originals
use BorrowedBudget. Reject uncertain wrapper accounting before pool closure.
On a pool failure, retain wrapper funding and actual owner/error through a later
cleanup attempt. Existing pool/base effect evidence prevents duplicate refunds.
`pool.closed` retires the pool's own resources/base; an enclosing wrapper may
still hold its final reservation. A wrapper refund that fully takes effect after
physical/pool cleanup records that effect and preserves any error; it does not
create a replacement reservation. Advisor recommendation and independent
architecture clarification are READY; shared-pool ownership stays unchanged.

## Alternatives and consequences

Increasing the storage budget would evade the requested unchanged-budget smoke.
Skipping artifact or original validation would weaken publication. Releasing
physical custody early would change the agreed failure boundary. Keeping already
consumed graphs causes an avoidable simultaneous reservation. The selected
lifetime split reuses the existing actual retirement mechanism and retains all
live evidence. Frozen count is private transaction state, never a new receipt or
substitute for the Source's pre-update verification.

## Verification

Use fail-before/pass-after ordering and unchanged2GiB admission, actual graph
retirement and retained original FD/metadata accounting checks. Cover early,
active-borrow, stale/wrong-owner, incomplete/duplicate and refund uncertainty;
count/step mismatch and corrupt artifact/original/close failures must still block
publication. Run affected Source/PPO/wire/native/directory/numeric regressions,
static checks and both independent re-reviews. Run new unchanged-budget real
pause1/resume2 parents; preserve the first failure. Pre-amendment whole-project
4797PASS/331SKIP is separate evidence. Final results belong in
[the policy verification record](../verification/pack-checksum-read-policy.md).
