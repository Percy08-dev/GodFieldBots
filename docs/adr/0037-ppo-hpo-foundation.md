# 0037: Explicit configurable PPO and outer HPO campaigns

- Status: Accepted design; implementation in progress. Current-context
  `bot_architecture_reviewer` mapping READY for units1–5, no blockers.
- Date: 2026-11-12
- Owner: root
- Baseline: `040184223a4bdcde85875f556daeb4a160ee7027`

## Context

The fixed PPO recipes support reproducible comparisons but cannot express joint
model/training exploration. Equal numbers of updates do not give equal numbers
of distinct loss endpoints when collection length or minibatches change.
Optuna's trial state alone does not establish healthy complete training,
evaluation, cleanup or original-source integrity.

## Decision

Implement [the supplied design](../design/ppo-hpo.md). Add explicit model4 and
HPO-only data/control versions while retaining all historical semantics and
fixed kernel AST pins. Reuse the common PPO coordinator and shared network.
Python owns rewards, full-episode GAE, selected-endpoint normalization and
independently seeded final-batch selection. Native validates original evidence
and correspondence, with an additive required capability and no fallback.
The concrete [record/owner contract](../design/ppo-hpo-records.md) is separately
architecture READY. A small structural lineage-reader prerequisite precedes
Source/native implementation; the common coordinator/retention follows it.

One trial contains a complete caller-declared seed set of fresh runs. Budget D
counts unique loss endpoints. The objective uses the final-D policy, shared
paired GameSeeds against the heuristic, and equal training-seed weighting.
Only complete healthy evidence yields an objective. Exploratory HPO authority
does not extend formal parent/final64/Dataset qualification.

Optuna5.0.0 is a lazy optional extra. Studies bind model/stage/space/budget/
evaluation conditions. TPE explicitly enables multivariate/group and disables
constant liar; pruning is disabled. Persist complete numbered proposals and
history digests before work; exact receipt equality gates idempotent tell.
One durable GPU admission lease spans all campaigns. Unconfirmed cleanup after
an abnormal exit blocks further GPU work until confirmed termination or explicit
environment-restart recovery. A released lock or stale PID is insufficient.

## Alternatives and consequences

Mutating the fixed recipe would invalidate historical evidence and source
qualification. Independent trainer/network copies would increase replacement
cost and permit numerical drift. Update-count and wall-time budgets fail the
requested equal-distinct-decision comparison. Partial seed averages and zero
scores for infrastructure failure distort the objective. Best-intermediate
checkpoint selection changes the experiment and is excluded.

Activation-aware initialization and variable endpoint selection require explicit
new identities. Per-update full saves and protected G0/latest2 resume states
cost storage/I/O; interval-based save thinning is deliberately deferred.
Resource admission remains a conservative managed estimate accompanied by
measured usage, not a physical-memory guarantee. Search ranges and adequate
experiment sizing remain caller responsibilities and are not invented here.

## Verification

Each implementation unit requires scoped behavior/regression checks and
independent review. Final acceptance requires separate correctness/conformance
reviews; Python/static/Rust/old-data/optional-free checks; small actual CUDA and
Simulator training/inference/resume checks; and fault-injection coverage for
selection/proposal/save/retention/evaluation/tell/recovery. Exact results and
remaining limitations belong in [the verification record](../verification/ppo-hpo.md).
No formal HPO or strength claim follows from foundation checks.

Scoped units1/2 pass independent correctness; unit2 also passes independent
implementation conformance. Root's affected576CPU checks, project Ruff and
mypy179 pass; actual CUDA component evidence and remaining acceptance scopes are
recorded in the verification document. Later units remain implementation work.
The structural reader prerequisite4a and Python Source3a now also pass both
reviews after generation-cap/G0-RNG fixes, with root341 and268affected checks
respectively. Native Source3b, runtime producer and outer campaign acceptance
remain pending; no full-foundation completion is implied.
Management5a now passes both scoped fix reviews: explicit original creation
evidence and ancestry-resync corrections close its two blocking findings.
Root verifies255isolated and167post-integration cases plus project static checks.
The per-user admission receipt is operational process evidence, not training or
evaluation qualification; concrete runtime/campaign composition remains pending.
Native Source3b is now scoped engineering-ready after both independent reviews,
actual private FFI array/lifetime checks and package/source correspondence. Root
reruns172Rust checks and five private release Model4/admission/view cases. This
does not activate the installed extension or complete the integrated HPO gate.

The runtime4 coordinator/retention producer and distinct evaluation6 role also
pass both scoped reviews. Runtime admission includes the retained Source during
save/acknowledgement, all HPO mutations check the actual parent-lock owner, and
cleanup preserves primary errors while withholding uncertain termination.
Numeric storage is initially colocated with parents; the current snapshot codec
and conservative Source limits refuse unsupported combinations before GPU work.
These restrictions are recorded in the runtime contract and should be revisited
when a real supported workload needs independent numeric storage or wider Source
envelopes. Campaign5b also passes scoped correctness/conformance after three P1
fixes: actual collector ABI, namespaced per-evaluation episode identities, and
complete final-D policy/registry binding before scoring/tell. Its durable Optuna
and final-policy composition is integrated as27721bf. Final foundation acceptance
still requires the predeclared real GPU/Simulator protocol and final checks.
