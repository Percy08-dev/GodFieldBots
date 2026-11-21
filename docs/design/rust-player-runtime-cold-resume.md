# Cold HPO qualification and current-host resume audit

Status: accepted narrow preimplementation clarification; implementation/conformance pending.
Independent architecture READY on proposal SHA
6721fe2a01ac921e133cb5933b104e1b74c8ecf612125c744cda74db28545b2a.
Owner: root. Extends accepted records.md, operation-v2.md, learning.md and held
restore-binding.md. Existing ordinary completion1 bytes/schema remain unchanged.
No Source/physical authority can be recreated from JSON or a digest.

## One-way HPO final qualification

Accepted parent-validation3 includes completion_receipt_sha256. Completion1 must
therefore never embed that metadata object or its hash: that would be a backward
cycle. Its existing required parent_metadata_validation field is ALWAYS null,
including HPO. This is a reserved-null field of version1, not an open blob or a
promise that a later operation will rewrite the record. Readers require null.

After genuine ColdContinuation obtains the exact current Prepared/Exit pair,
full original/Replay/numeric/saved validation and exact-final-D checks, write
immutable generic completion1, acknowledge its exact bytes through the existing
supervisor protocol, then create the private actual ColdFinalizedRustParent data
owner and issue a SEPARATE immutable parent-validation3 referring to the already
written completionSHA, Prepared, Exit and source-validation hashes. Its existing
scalar parent/settings/seed/final-D/artifact facts remain unchanged. File path is
returned as actual owned data; it never becomes a backreference in Run/Prepared/
Exit/completion. No normative receipt version or fields need a speculative bump.

The private owner remains local/unpublished while metadata3 is being written and
validated. Return it to the campaign/seed-receipt caller ONLY after that succeeds.
The seed/campaign branch requires the actual owner AND its bound parent-validation3,
full exact-D facts and typed physical-termination3. Generic completion.eligible is
only DATA qualification for its declared completed invocation; it is insufficient
for HPO parent/seed/evaluation authority without this whole pair. Paused/incomplete
results retain the generic invocation owner and never receive the Parent type.

If metadata creation/validation fails after generic completion ACK, leave that
record immutable; mark provisional parent issuance terminally failed, retain real
scan/file/budget owners for cleanup, return no Parent/closed seed receipt and do
not send successful parent-release as if qualification completed. Cold must fail
under the remaining ORIGINAL deadline. Actual supervisor failure/reap/quarantine
handles physical cleanup; no receipt is rewritten or grace created. A protocol
finalized ACK remains pair/hash acknowledgement, never Source qualification.

Tests: complete generic receipt alone cannot create seed/evaluation authority;
metadata3 SHA links are acyclic; post-ACK metadata-write/validation fault returns
no Parent/seed receipt and original failures remain immutable; paused/incomplete/
ordinary owners refuse Parent conversion. Final independent reviews cover this.

## Genuine current-host resume DATA audit

A new resume training host has current genuine HostAdmission and actual borrowed
RunOwnership, not a ColdContinuation for the previously exited host. Never create
ColdContinuation, Pending, old SaveACK or old termination capability from JSON.

Introduce private RustResumeAudit and explicit RustResumeOriginals data arm in
the existing originals reader. Constructor requires the CURRENT actual admission,
CURRENT original parent OFD borrower, current validated operation request=resume
and matching run/config/runtime identity. It reopens through the existing sealed
scan/file owners and verifies the prior immutable PAUSED frozen state/operation/
Prepared/Exit/completion byte chain, successful original exit/retirement facts and
full raw/Replay/numeric/saved content. Prior receipts are audited DATA, not current
reaping/admission authority. Their original clocks remain historical; current
operation's original milestones govern every new read. No old clock is extended
or recast as the current host's authority. This audit creates or renews NO deadline authority. SAME-operation milestone
bits remain exact. A distinct operation may use only the ALREADY authorized
bind/session rules in integration.md and operation-v2.md, preserving their
original parent/session allowances. This supplement does not approve a fresh
parent/session budget or a new cross-session resume budget policy; any such
change requires its own explicit design decision. The declared B component
deliberately shares the same original900s across both invocations.

Hold all original files, scan/account loans and actual current parent borrower
through their last consumer. New read-only inference banks use the SAME Source
budget, no duplicated full allowance. Issue ONLY a DATA RustRestoredPolicy from
that actual completed audit. Bind the held resume SHA at both materializing
reads as restore-binding.md requires. Keep saved generation's old snapshot lineage
identity separate from the new read-only bank identity. No fake SavedPair/ACK,
old-generation resave, fabricated live job or Source qualification follows.

The reader's current-cold arm still requires actual ColdContinuation. The new
resume arm checks actual RustResumeAudit liveness/identity/phase, not a dict with
matching fields. Its output is audited originals/data, never a fresh training
Source or final-parent capability. New Source issuance still requires the future
actual native collection/ACK/retirement chain under current admission.

A mismatch or read/owner failure stops resume before restored policy publication;
retain uncertain owners and preserve prior records. Verification covers metadata-
only/copy/foreign-owner refusal, missing/altered prior bytes, wrong parent/config/
producer/opmode, same-budget last-use closure and actual held-file mutation.
The approved actual Model4 pause/resume component then exercises the real current
host audit/load and final metadata3, with zero further Adam steps stated honestly;
Task6's real two-update continuation remains mandatory later.

## Scope and responsibility

After independent architecture READY, the existing B implementer may add these
private owners/reader arm within its already-owned rust_player_completion.py,
rust_player_registry.py, rust_player_learning.py, rust_player_source.py and existing
card_instance_collection_source.py/card_instance_source.py, plus focused tests.
No new public CLI/wire/format/dependency or old-mode capability changes. Root owns
design/project config/commits. Tests and final correctness/conformance review are
required before B acceptance; this clarification is not implementation acceptance.

## Bound cold-to-next-host ownership composition (accepted narrow clarification)

The accepted supervisor may retain the actual cold child as the next training
host after finalized→release_parent→bind_operation; it need not spawn another
Python. Current operation/phase/lock state, never the initial launch's stale
parent mapping, selects the existing ownership path. This changes no wire or
primitive and preserves exactly one Python during the new training phase.

When a genuine finalized bind leaves no inherited CURRENT parent mapping, the
fixed _run_local handler enters actual run_lock(parent), sends that SAME OFD by
existing transfer_run_lock and waits actual authenticated ACK, then obtains the
current HostAdmission before CUDA. The same _run_ppo uses the EXACT already-
registered current-thread RunOwnership for the same parent; do not open another
lock or fabricate a borrower. Initial inherited-parent execution keeps existing
borrow_run_lock. Existing actual-owner/path/thread checks remain authoritative.

After actual mutation/cleanup/frozen-state work, exit the outer transferred lock
context using its existing close-only behavior before ops.prepare. The actual
native supervisor still holds the acknowledged OFD through Prepared/host exit.
No local finally may LOCK_UN a transferred/uncertain receiver alias. Existing
send/ACK failure ownership applies unchanged. Source/data owners requiring the
local borrower must finish before that exit; only control/immutable facts survive.

This composition must not renew any deadline, reuse a prior ColdContinuation as
current authority, reconstruct an FD from JSON, infer lock ownership from PID
absence or bypass actual current parent registration. Prior owners are positively
closed/unregistered before release/bind. Test separate-open competitor exclusion
through transfer→common loop→local close→Prepared, exact current owner reuse,
wrong-parent/thread refusal and failure before/after ACK. This is already-owned
B source/private-handler scope; no legacy normal/HPO ownership path changes.

Independent composition preimplementation READY on proposal SHA256
`f6d8290d09474925380aa75bbb233ab2439902a20710b79b1b364caccc9304c5`.
Scoped final B conformance/correctness are READY after two lifetime repair rounds.
Returned invocation/Parent retains the actual registered borrower through positive
close; require checks current Cold/control/thread/path. Pre-adoption control graph
aliases retire before audit-held refunds, and constructor failures preserve actual
nested cleanup owners before exception-owner replacement or traceback retirement.
No raw/numeric artifact preservation or wire format changed in these repairs.
