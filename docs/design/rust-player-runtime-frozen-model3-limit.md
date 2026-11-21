# Model3 operation admission outside frozen update AST

Status: proposed narrow preimplementation refinement. Owner: root. Preserves the
accepted eight-kernel formal contract and device-account semantics. No pin/recipe/
legacy code identity is disguised or updated to accept a changed frozen body.

## Evidence and choice

Scoped compatibility test test_all_eight_frozen_kernels_remain_exact failed after
optional device-limit plumbing changed structured_numeric.Learner.update's AST.
The frozen constructor is also protected and remains unchanged. admit/parity/
_parity are not pinned. FP32 loss/Adam arithmetic did not intentionally change.
Root chooses to restore the EXACT base frozen update AST and constructor, and
add only a separate Rust same-host operation entry plus unpinned admission checks.

## Bounded interface and lifecycle

Add Learner.update_with_device_limit(rollout, *, device_bytes: int) that enters a
private PER-LEARNER operation-limit scope, then calls the unchanged self.update.
The scope holds the one immutable byte value from the caller's already frozen
accounting operation. It is not the old lifetime _training_device_bytes and does
not mutate bindings, configuration, RNG, parameters, constructor or checkpoint
metadata. Reject overlapping/nested operation-limit scopes before work.

Unpinned admit/parity/_parity read that current operation value (missing/None
means exact legacy behavior) and apply the already accepted actual-endpoint/
parity admission before any numerical allocation/mutation they guard. The existing
frozen update reaches those methods in its unchanged order. No second PPO loop,
wrapper that duplicates loss math, callback into orchestration or redundant full
Raw/Source validation is introduced. Cache/precompute shared arithmetic if needed;
do not repeat whole logical validation merely to relay the value.

The new local Rust wrapper chooses update_with_device_limit when its actual
CUDA update has an allowance; legacy/default and CPU numerical callers retain
existing update. The numerical scope is protected by the caller's SAME existing
serialized_torch lifetime. Restore/remove the private value on return or exception;
that cleanup releases only numerical context, NOT a failed accounting hold or GPU
graph. Existing account poison/actual failed-owner custody remains with the caller.
No numerical HostAdmission or live Source capability is reconstructed.

## Compatibility and verification

Restore frozen method text/AST exactly from a83288b/base, verify all eight existing
pins unchanged, and keep old numerical defaults/results exact. Add RED/GREEN
new-route limit rejection before parity/shuffle/Adam, accepted nonbinding route,
exception/nesting cleanup and two different immutable scopes. Verify both original
logical checks and the new operation check remain active. Keep source producer
changes explicit; old experiments retain recorded code, no relabeling.

After independent architecture READY, existing Task6b implementer may refine its
already-owned structured_numeric/LocalGameLearner methods and focused tests only.
Other kernel/network, saved formats, pins, versions and training recipe are
unchanged. Actual Task6 CUDA checks and both implementation reviews remain needed.
