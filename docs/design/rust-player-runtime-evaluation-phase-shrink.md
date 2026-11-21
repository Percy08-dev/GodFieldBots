# Shrink-only release of funded evaluation construction capacity

Status: accepted narrow common-account API supplement; independent architecture
review READY on 2026-11-20. Owner: root. Needed by the accepted funded SealedScan design,
not a new memory formula, allowance, margin or generic reservation framework.

Current ArtifactPhaseAdmission.reserve is nonwaiting and backed by the SAME
Source account. Its _PhaseReservation has close only. StorageReservation.resize
supports shrinking and its current growth path uses nonwaiting acquire (the
default wait=False). Partial release from construction peak to retained metadata
is needed without a close/re-reserve funding gap. The phase API therefore exposes
only shrinking; growth continues through existing nonwaiting delta reservations.

Add ONLY _PhaseReservation.shrink(size):
- Under the existing phase RLock, require an open reservation and exact nonboolean
  integer 0<=size<=current size. Growth is refused before any underlying action.
- Require an actual backend shrink/resize capability; the current evaluation caller
  uses the actual StorageReservation. Do not widen the existing close-only public
  ArtifactReservation protocol or introduce a Models-to-training import.
- Call that actual reservation's shrink resize, which performs release and never
  acquisition/wait on this path. Only after success adjust phase.used and wrapper
  size by the same delta; phase.peak remains its historical peak. Existing lock
  ordering remains phase then Source. Underlying refusal leaves phase counters
  unchanged according to the existing backend resize contract.
- Unsupported backend/closed/invalid/growth calls fail without fallback, refund,
  new reservation or mutation. Existing close/context-manager/default callers and
  legacy ArtifactAdmission shapes/behavior remain unchanged. Cleanup shrinking
  does not require a new time allowance or revive expired work.

The funded scan grows its construction peak using existing nonwaiting reserve of
ONLY the required delta, retaining those actual coupons. The sum covers the MAX
concurrent requirement across phases; it does not sum historical scratch with
later permanent metadata. After actual scratch consumers retire, close excess
coupons and shrink the final one to the retained amount, keeping total funding
at least the real remaining requirement throughout. No close/re-reserve gap,
double admission or independent whole Source allowance. Steady metadata charges
remain until provider graphs/FD owners positively retire, including failure paths.

Ownership extension is only gf_common/artifacts.py and focused account tests;
no pool constructor, StorageBudget resize semantics, limit or coefficient change.
Verify growth refusal before any backend call, shrink/same/zero/closed/boolean/
negative behavior, exact Source+phase used/peak consistency, underlying failure,
unsupported backend, and a real evaluation-pool/phase overlap using actual
StorageReservations. Check old reserve/close behavior. Pure CPU/data tests only;
no GPU/game/training budget or measurement claim follows.
