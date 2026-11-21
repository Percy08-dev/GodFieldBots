# Retire one borrowed saved-evaluation verifier under its actual audit hold

Status: accepted bounded lifetime supplement; independent architecture READY on
proposalccf3e4e7ce3f0cd0531d3e51f3e66db915aed8d945db2abe060b3fbbd079d592.
Owner: root. Concrete RED: real in-process SQLite/media PackedSourceOwner borrows
the actual completed-audit pool; closing it unconditionally checks global terminal
close and refuses the audit's own live PoolCloseHold. Dropping that outer hold or
allocating another pool would violate accepted custody/sole-account requirements.

Keep ALL existing no-argument close/constructor/context behavior and existing
native executor/audit hold protections. Do not blanket-skip terminal guards for
every _owns_pool=False owner. Existing d3e15/native_group_pack close holds protect
actual originals, not merely the pool object. PackedArtifactPool.close and
require_terminal_close remain unchanged and block while ANY hold remains.

Add a private, explicit borrowed-verifier close path to PackedSourceOwner.close,
with optional actual PoolCloseHold and its actual owner object; default None is
the exact old path. Require non-owning concrete pool, exact registered active
hold for that SAME pool and hold.owner identity, no retired/uncertain/foreign
hold, and refuse if ANY other hold remains. Validate before FD/account mutation.
Centralize these checks in a small private pool helper if needed; no new token
class/framework, limit, allocation or hold retirement. Existing group loans,
Original close effects/uncertainty and charge-refund checks still apply. Only
the explicitly named outer audit hold is exempt from the wrapper-level check;
the hold stays registered and still blocks actual pool.close throughout.

The sole new caller is the actual RustCompletedParentAudit saved-evaluation READ
branch, under its genuine control/RunOwnership and same pool/phase. Before the
verifier closes/refunds its own metadata, make a positively owned independent
manifest/proof metadata copy: reserve construction capacity in the audit's actual
parent phase (whose source is this exact pool budget), canonical bytes plus fresh
decode, and attach the actual DATA retainer to the audit. Both actual copies are
funded during overlap; no guessed transfer, shared-graph double counting or new
allowance. Drop old aliases before closing the verifier under the explicit own-
hold path. New graphs remain charged through caller last use and are cleared
before retainer refund. Only metadata is copied, never policy weights/Raw banks.
Report serialized byte counts separately from estimated charge and physical usage.

On failure retain actual verifier, metadata retainer, all partial/nested cleanup
owners and traceback/cause/context before outer overwrite. Later actual audit
close drops its views and clears stopped frames before retiring those actual
objects, then retires its own hold only after positive child/FD/loan closure.
A tiny actual verifier-retirement adapter can preserve normal child.close shape;
it must contain real objects, never a HostControl/Source/Parent surrogate. The
new shared reader may receive the actual audit owner through a private exact-
typed seam to retain metadata/access its hold; old defaults remain unchanged.

Constructor/default cleanup gets NO exemption automatically. Partial construction
or validation failure retains the actual verifier and traceback under the audit
until later positive cleanup. Unknown owner/close failures retain custody/funding;
no time allowance, pool drop/reacquire, repeated numeric FD reconstruction or
JSON-driven cleanup authority. Actual active close-hold construction/retirement
uncertainty must refuse the new path, following the existing effect receipts.

Scope only group_pack_game_source.py and at most a narrow group_pack_pool.py
helper plus already-owned C2b audit/CLI reader/tests. The two PackedBatchCustody
prepare/close guards stay unchanged: the saved verification helper calls
_inspect_component_collection without creating/preparing that custody context.
No changes to Source issuance, rules, numerical formats or actual evaluation
execution/admission. Dependencies stay training-owned pool/Source; CLI uses real
audit APIs; no training-to-CLI edge. Old default/shared callers stay unchanged.

Verify real tiny SQLite/media borrowed wrapper under actual outer hold can
positively close while pool.close still refuses; default borrowed/owning close
under hold still refuses; other/foreign/retired/wrong-owner/uncertain hold and live
group loans refuse without refund; positive failure retry retains actual owners
and both pool/phase counters. Returned manifest/proof remains funded to last use.
Retain existing native-group-pack hold tests, but run only precollected pure
CPU/file cases; no native worker/control/game/GPU or whole-suite budget. Root
owns acceptance. Final genuine Source/Parent/public Task6/7 remains required.
