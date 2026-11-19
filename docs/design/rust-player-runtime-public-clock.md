# Explicit post-host allocation within the original total deadline

Status: accepted bounded public-factory/configuration supplement.
Independent architecture READY on clarified proposal SHA
0e94bd0ac8248d3476105d4fd2876cd92a2d09588d628af582fc491a1b6b0bea. Owner: root. Applies advisor
/root/production_clock_advice's one-tail structure. No numeric production default
is adopted from unrelated server/report settings or unmeasured fractions.

## Requirement and decision

Current native authority requires original A<=E<=TERM<=KILL<FINISH bits. Existing
Run configuration declares admission A, execution E, total W and finalization F.
Server shutdown, legacy cleanup and report reserves do not bound the serial new
reap/namespace -> cold-host startup -> complete-history audit/acknowledgment path.
Retained-history cost varies. A fixed generic reserve is not established by the
bounded one-update evidence. Root chooses ONE explicit owner-supplied post-host
allocation P, with no inferred/default numeric value, rather than a new estimator,
several grace knobs or unsupported general capacity claim. Measured Task6 tails
will inform Root-authored example presets; they cannot establish a universal bound.

For the original monotonic_after anchor t:

- admission=t+A; execution=t+E; finish=t+W.
- term=finish-P; kill=term.
- Require finite representable results, 0<P<W-E, and execution<term<finish in the
  resulting float bits. Do not invent a minimum margin. Existing config validation
  still applies; F is recorded without relabeling it as cold-audit capacity.

STOP at E already requests cooperative cleanup, so there is no separately allocated
host SIGTERM grace. At TERM, native may issue TERM and KILL in its established
order without waiting between coincident milestones, retaining original leader/
group/wait ownership. All physical proof and cold work consume the SAME FINISH.
No E reduction, W enlargement, use of E+F instead of W when extra slack exists,
new cleanup allowance, fresh clock on host reentry or timeout-as-proof.
A healthy host that closes early starts cold immediately; cutoffs are not waits.
If proof/audit cannot finish, preserve ineligibility/uncertainty under actual
owners; declared P is an allocation choice, never a promise of completion.

## Public configuration, backward compatibility and source flow

Ordinary new supervised profile uses run.post_host_seconds. HPO schema6 uses
operating.post_host_seconds, projected into its materialized configuration DATA.
It is one operational positive finite number, not a search parameter or policy
feature. Public Rust execution and execution-ready validation require it; parsing/
inspection of earlier private declarations can remain available without execution.

Keep all existing ordinary/Shared/legacy HPO1–5 concrete types, serialized shapes,
defaults, identities and unknown-field rejection unchanged. Select an explicit
new native configuration subtype/closed variant ONLY when this key is present,
using the repository's existing native RunConfig and operating-type separation.
Do not add a None-valued field to all legacy asdict outputs. Existing new-mode
runtime declaration remains its exact SIX fields; no timing enters native policy
runtime/candidates/features. Older native profiles without the setting are not
silently given a reserve or migrated. Their explicit controlled-operation APIs
can retain their original literal operation2 clocks; public fresh construction
requires the new declared allocation.

For HPO, campaign operating allocation is authoritative. A base native run that
also declares P must agree; do not silently prefer or discard a conflicting value.
Pure preview checks both trial and canonical-control materializations without
Torch/device/Source admission. Timing configuration remains part of effective
config provenance, including strict resume equality.

## Factory, provenance and resume

Complete pure parsing/native identity/companion and source/storage checks first.
Capture original clock provenance immediately before actual run/store ownership
and operation publication/handoff, so those fallible setup actions consume the
operation budget. The fixed host decodes these bits and does not restart them.
No numerical or Source authority follows from proposed clocks or configuration.

The pure internal clock factory takes the explicit P and returns checked bits plus
DATA provenance; caller owns actual file publication under the original lock.
Use a new strict schema1/kind rust-player-operation-clock-v1 with exact fields
{schema_version,kind,operation_id,operation_sha256,run_id,config_sha256,
clock_binding,inputs,post_host_seconds,recipe,deadlines}.
clock_binding reuses the existing FOUR-FIELD SHAPE, with this OPERATION's own
original anchor and the same boot. Immutable launch.clock_binding remains the
SESSION anchor; later HPO operations need not share its anchor values. inputs is
exact {admission_seconds,execution_seconds,wall_seconds,finalization_seconds};
recipe=same-total-explicit-post-host-v1. Canonical path is the actual control root's
clock-{operation_id}.json. It references the immutable operation SHA and config,
and its recomputed bits must match the CURRENT authenticated operation2. Its
boot must match the original session and finish must not exceed the original
session finish; do not compare later operation anchors for equality with the
immutable session anchor. This is
reproducibility DATA; actual authority remains the authenticated operation2/launch,
never this JSON. No changes to existing operation/Prepared/Exit literal schemas.
Public result/provenance output records this actual path/SHA; no self-reference.

Ordinary public control root and config snapshot path are deterministic under the
actual run owner and stable across pause/resume; don't derive authority from a
caller-supplied historical pathname. Resume verifies actual owned paused state,
old clock provenance/config/producer/boot and original operation receipts as DATA,
then REUSES every original deadline bit and original boot/record root. It never
runs the fresh recipe, changes P or adds paused time back. Actual HostControl/
RunOwnership/admission and the full resume audit remain required before CUDA.
Cross-boot or expired original authority stays refused. No legacy clock migration.

HPO distinct new training operations use their own original anchor/allocation;
repeat/continuation of an old operation cannot renew it. Existing session
soft-dispatch t+S and hard=t+S+maximum declared TOTAL train/eval W remain unchanged.
Control-only and existing independently admitted legacy evaluation operations
retain their already accepted responsibility and clocks, without fake admission.

## Verification and bounded measurement

Pure tests cover explicit-field/legacy shape separation, missing/invalid/boolean/
nonfinite P, representability, A/E/W/F+extra-slack preservation, coincident TERM/KILL,
exact bits/recipe/config equality, and resume reuse without recapturing an
anchor. Current-time expiry checks remain required and do not renew authority.
Verify no Torch/hardware/namespace/Source creation during parse/preview; real
SAME-OFD launch/handoff and expired/cross-boot refusal at genuine process boundaries.

Use the already REQUIRED final integrated Task6 continuous2/pause1-resume2/full
original audit and forced-failure controls to measure complete healthy and forced
tails by actual monotonic events/original waits. Keep healthy success and forced
failure alternative paths separate; do not sum both or multiply past server maxima.
Root will declare concrete experimental P and input budgets before those cases,
label it owner-selected rather than measured sufficiency, and preserve failures.
No additional expensive population is authorized here. Before recommending default
presets, record tail/history bytes/update count/cache/storage conditions and timing
variability. A few observations are not an OS/filesystem upper bound; longer
histories require an appropriately explicit user allocation. Full unchanged
24-run comparison/default protocol and calibration remain mandatory.
