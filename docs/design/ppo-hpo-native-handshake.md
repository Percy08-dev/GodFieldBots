# Native HPO selection authorization on the SAME Source

Bounded Source3b interface under [the accepted HPO records](ppo-hpo-records.md).
Python owns subset sampling/GAE; native owns actual original-record facts and
structural correspondence. Actual episode counts exist only after native scans
the originals. Old native proof/correspondence froze before Python authorization,
so HPO needs an explicit two-stage variant rather than a second Python scan or
native sampling. Root accepts this design; scoped architecture review precedes
dependent production authorization implementation. Scoped architecture review
is READY, with no blocking findings.

## Phases and interface

Use the SAME `CardInstanceNativeSession`, core, parent/Source, original FD/Pack
custody, executor, cancellation and budget leases throughout.

1. Scan exact plan5/Hpo or PackedHpo originals using the existing native owners.
   Return a closed pending-controls5 facts summary with immutable original
   fingerprint, original ordered learner episode counts/digests and the semantic/
   control bindings needed for independent correspondence, plus declared selection
   context. It contains no final Proof3/projection/update authority.
2. Python independently validates actual control/schedule/artifact/original fact
   evidence, including G0 `PCG64(learning_seed)` opponent RNG, and derives/replays
   the canonical selection from actual counts. It computes expected selection,
   optimizer delta and final correspondence. No second Python journal scan.
3. Add `authorize_hpo(authorization_bytes)->final_summary_bytes` on the SAME
   session. Authorization binds expected pending fingerprint, exact context,
   full selection, closed training recipe and expected optimizer-step delta to
   the original Python-owned cohort/authorization context. Native checks all
   structural count/index/hash/version/recipe relations against its actual
   original facts and rechecks custody; it never chooses the subset or computes
   reward/GAE/normalization/Adam.
4. Only after complete successful validation, freeze final Proof3/correspondence
   and return the final controls5 summary. Python compares it with independently
   computed expected facts/selection/correspondence before issuing its live
   handle. Numeric/update authority requires this final phase. Ordinary access to
   diagnostic preauthorization facts cannot authorize a policy or update.

Keep old `authorize` ABI, plan2/3/4 and source/proof meanings exact. The additive
method is required by the HPO capability/API preflight. Exact private DTO key
names follow the current accepted Proof3/Numeric5/selection contracts; this adds
no second Source or external process owner. A repeated/mismatched/stale phase or
different authorization must reject; an implementation may support only exact
same-session idempotence if that preserves the existing ownership contract.

## Failure and evidence

Retain actual originals and their leases at both phase boundaries. Parent locks,
scope and namespace/custody checks must detect replacements, closed owners,
changed facts or stale initial fingerprints. Pending summaries are explicitly
unqualified. Cancellation/exception joins active native work before releasing
owners; no final proof, numeric publication or live update handle escapes a
partially authorized/failed session. A failure can retire the session instead of
attempting permissive repair. No source rejection detail becomes policy input.

Historical-only candidate restrictions remain; a private candidate cannot grant
live/formal authority. New final HPO proof still grants no formal/final64/Dataset
qualification. Full pipeline/actual GPU evidence is a later functional check,
not strength evidence. Export the capability only after these paths actually
work; a marker is not a substitute for implementation.

## Verification

Actual candidate FFI tests cover pending facts/no-update refusal; exact Python
selection and native final structural agreement; wrong subset versus the Python
gate, stale fingerprint/context/count/recipe/delta; same-owner/session binding;
source mutation/cancellation between phases; fully pruned historical controls;
and directory/packed/three-model parity. Verify old authorize ABI and all old
source/array/hash/lifetime behavior. No native RNG/GAE implementation is added.
