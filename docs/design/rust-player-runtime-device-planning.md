# Same-host CUDA planning: bounded component and production gate

Status: accepted narrow component design; generic planner remains required work.
Independent architecture READY on proposal SHA
8402cd225a1a2a03b6662b361c9f39642f8e6c636768f7bfcbd07ebc9a8357a3.
Owner: root. Applies the accepted learning.md Accounting section and
bots-performance discipline. No new restrictive inference formula is approved.

## Evidence and decision

The implementer's data-only lifetime table is recorded in
.superpowers/sdd/rust-player-runtime/task-5c1b-gpu-accounting-evidence.md. Independent
bot_advisor recommends the existing learner/cache admission for the explicitly
bounded component, and a native-specific combined phase model before arbitrary
new-runtime CUDA admission. Root adopts that recommendation. Source inspection
finds no evidence of a12GiB violation in this component; that is not a physical
peak guarantee. Exact native alias formula and CUDA measurements are NOT RUN.

For the already approved Model4 MLP/seed17/C1W1B1/16games/D64/maxupdates2/freshpause1
then actual resume, use unchanged owned_learner_cuda_admission with:
logical_bytes=2,966,109,712; global_cuda_bytes=12GiB from ACTUAL HostAdmission;
pool_cuda_bytes=15,492,112 for reachable distinct {G0,G1}; overlap_scratch_bytes=0.
The learner device allowance remains2,966,109,712. Learner allowance plus future
cache reservation is2,981,601,824B, a managed allowance—not predicted allocation.
Existing learner estimate already includes one context/workspace; do not add it
again. Existing logical geometry, input/tensor controls, loss/optimizer and actual
service settlement/fence before update remain unchanged.

Configured cache capacity7 does not imply7 live/future networks. Count actual
storage owners and reachable publication transitions once; recorded pool/current
aliases share one checkpoint. Freshpause1 has at mostG0/G1. Resume derives planned
loads from the verified paused state and actual loader path: if used=D64 there
is no future collection/update; still count every cache network the loader really
allocates, and retain the declared two-slot component bound if both load. Do not
claim zero cache merely because no further update is possible. No configured-C/W
or broker/process/context multiplier enters the new same-host calculation.

This reuses an existing partial learner allowance; it DOES NOT claim a complete
combined collection estimate. Do not add actor_cuda_estimate.transient_bytes to
native input bytes R: raw cats/masks/categories/present/references alias R and
branch-specific maxima/lifetimes differ. R=7,053,926 at this component's configured
maximum; transfer has R atB1 and old+new input dictionaries up to2R only atB>1.
Those observations support accounting, not a universal added restriction.

Before this component, actual global admission must precede either CUDA role.
Record process allocated/reserved peaks by phase with precise measurement scope,
and keep any whole-device samples distinct from process allocator statistics and
host PSS. Cold inspection remains CPU-only and does not initialize/query CUDA.
No OOM/guard failure creates completion/Parent eligibility; preserve originals.
Both invocations and cold work retain the original shared900s milestones.

## Required general planner before public CUDA activation

General new-mode CUDA admission/default acceptance remains BLOCKED on an approved
native-specific allocation/lifetime model and focused verification. This is part
of the current implementation task, not accepted permanent debt. Do not expose
this component-only partial admission as a full generic capacity guarantee.

The design target is max over reachable phases of shared context/workspace,
actual retained learner storage, distinct cache storage and additional live
storage. When reusing an existing learner estimate, its context is already in
that estimate. Transfer/scoring/sampling-return/update/restore/export/publication
need their actual ownership, including native score/log-probability/critic output
lifetimes. If existing actor phases are reused, remove only operands that each
branch already charges BEFORE taking maxima; no guessed scalar subtraction.
Future cache bounds use reachable generations and actual planned loads, not
unreachable configured capacity. Phase-exclusive scratch uses max, not sum.

Root will record the general table/interfaces and obtain separate architecture
review before implementing that restriction. Focused verification must coverB1
versusB>1, retained learner/Adam state, publication/eviction overlap, aliases,
reachable resume generations and boundary acceptance/refusal. Actual peak data
checks the model without turning it into a hard allocator/physical quota.
This component gate accepts no Task6/7, general capacity or quality conclusion.
