# ADR0031: Torch-preloaded fresh actors for Tree Dataset pytest preparation

Status: accepted; implemented and verified on 2026-11-09 after independent
design, final correctness and implementation-conformance reviews.

## Objective and evidence

The owner requests Torch preloading for `test_structured_tree_dataset.py`.
The current default full-run case is 199.312s. A separate same-workload diagnostic
measures 167.616s collection, with 150.836s (89.9888%) in episode/preflight actor
readiness intervals. It runs three complete two-update MLP parents at seeds
17/29/43, 96 training games, 192 final-policy validation games, 576 episode actors
and three separate validation preflight actors. Fixed HTTP outcomes are component
evidence, not simulator execution, model quality or qualified Source acceptance.

Use ADR0030's existing clean single-thread Torch-only template and fresh forked
actors. Do not introduce a model cache, persistent participant pool, new dependency,
simulator change, data/schema migration, or legacy/structured fixture conversion.
This expands private validation launch/failure ownership, so the architecture
gate applies. Normal application launch remains cold exec.

## Ownership, interfaces and data flow

1. A module-scoped, explicitly requested test fixture binds one initially
   UNSTARTED `CpuTemplate`. `original_tree_sources` depends on this fixture;
   importing its helpers or running frozen-data tests does not start a template.
   Supported Linux CPU cases explicitly pass `preload_cpu=True`, `pool_cpu=False`
   and leave readiness ordering unchanged. Unsupported platforms retain normal
   fresh execution. One template spans all nine collections and all three seeds;
   each episode still creates new real participants/models/RNG/history.
2. Training uses the already admitted ordinary CPU preload path. Extend private
   `component_http` preload eligibility to the exact live `FinalValidationContext`
   path, retaining actions2/engineering, Python structured reader, CPU bindings,
   fixed structured factory, model and supported-owner guards. Require the
   actual `LoadedConfig` object to be `context.projected`, covering its whole
   effective/hash/input envelope rather than only equal config dataclasses.
   `source_ops`,
   native preprocessing, CUDA, arbitrary context objects and participant pooling
   in validation remain forbidden. The context must validate its original exact
   plan/config/SAME ledger before warmup or any external actor creation.
3. The existing three `preflight_final_validation_actor` calls remain ordinary
   fresh cold-exec actors on their real context ledger. They are not patched or
   skipped. Only the 576 episode actors use the template. Preflight, original
   CLI composition, context/Source ownership, schedule and artifact checks remain.
4. Extend the existing private `_pytest_worker_finalizer` admission in
   `gf_cli._run_experiment` to this exact final-validation composition, alongside
   its existing train admission. Require private factory, collect command,
   engineering actions2, `ppo_validation` + validation split, the exact concrete
   live context with its own identical plan/projected LoadedConfig/SAME ledger,
   and no source ops.
   Validate the context through its existing validator before preflight. Do not
   make arbitrary validation collections or duck-typed contexts eligible.
5. Keep the finalizer after all groups drain and before report/manifest success
   publication. Validation requires the fresh-forked receipt mode, not a pooled
   receipt, confirmed actual actor/I/O/lease retirement, SAME empty actor-only
   ledger, and healthy template with zero unretired tickets. Recheck the exact
   context/parent before publication. Receipt remains bounded diagnostic metadata;
   no policy feature, numerical recipe or Source authority changes.
6. Module fixture teardown closes and actually reaps the template and restores
   its binding even on error. Warmup and final retirement belong to measured case
   setup/teardown. A failed module teardown invalidates test acceptance.

Dependencies stay tests -> application. Application does not import test modules.
The generic private CLI hook only admits the additional exact context and enforces
its publication conditions; it does not know `CpuTemplate` or choose fork launch.
Actor/model packets stay on existing direct pipes, outside the template service.

## Failure, compatibility and current limits

Retain ADR0030 pending-before-send ownership, real PID/wait receipts, at most four
pending/live/unretired tickets, fixed CPU/thread environment, no CUDA/template
model state, 1GiB sampled template RSS soft allowance and first-deadline cleanup.
Lost status keeps unresolved leases and quarantines the parent; no fabricated
reap or restart/fallback. Both normal prepublication finalization and exception
paths retain/join the same template cleanup owner. A failed health check cannot
publish or be accepted as healthy validation data, including fallback reports.
No cleanup deadline is extended to obtain a passing test.

Existing parent and new validation originals remain subject to the real native
TreeDatasetSession audits, config/weight/manifest hashes, all64 validation games,
zero-endpoint preservation, Source closure and Dataset build/inspect checks.
The optional existing `pytest_worker_lifecycle` diagnostic field must be admitted
by those actual readers before implementation is accepted. Investigate any reader
incompatibility as an open question; do not silently weaken a reader or version.

Retain original budgets, seeds, outcomes, selections and actual process/ledger
assertions. No shared legacy `frozen_sources` data or old62 fixture is reinterpreted
as current structured data. All newly expensive lifecycle regressions follow
`GFB_RUN_CPU_PRELOAD_SAFETY=1`; lightweight admission checks remain default.

## Verification and acceptance

Independent architecture review precedes one implementer's changes. Independent
correctness and final conformance reviews follow stable implementation. Root owns
this design, project docs, acceptance and local commits.

- Exercise the complete original Tree target with all three parents, both updates,
  288 natural component games, 576 fresh forked episode actors and three ordinary
  preflight actors. Assert one template, real final retirement and all original
  Dataset counts/manifest reinspection. Keep its imported helper callers default.
- Cover exact validation admission, fake/stale/mismatched contexts and unsupported
  routes, pool rejection, prepublication finalizer failure/invalid receipt/live
  lease and healthy context rechecks. Reuse existing transport fault coverage;
  add narrowly targeted meaningful wiring regressions without another framework.
- Compare retained actual cold normal parents/validations to candidate outputs:
  all existing G0/G2 weight/optimizer/RNG arrays, all actual candidate/selection/
  search/frame/action content and 512 selected Tree features/natural labels.
  Validate each side's own hashes/bindings/identities independently. Any comparison
  projection for new paths/opaque IDs/host timings is explicit, comparison-only;
  never rewrite original source, repair its proof or relabel producer identity.
- Measure unchanged target with initial warmup/final close included. Retain the
  prior full-run 199.312s and matching disposable 200.56s diagnostic as distinct
  baselines, not interchangeable observations. No promised savings or variance.
- Run affected pool/preload/protocol/source/validation regressions, the explicit
  expensive CPU preload checks, existing frozen Tree reference checks on candidate
  originals, scoped Ruff, mypy, source-matched wheel/optional-free normal startup,
  and one final default full pytest run. No unrelated experiments or simulator
  edits. Record new safety skips explicitly; do not add their cost to default.

The read-only compatibility investigation finds no closed root keyset in the
Python Collection4 parser, `FinalValidationContext.accept_collection`, native
structured Source or TreeDatasetSession validation parser. Existing readers hash
the whole original manifest; the added diagnostic changes its actual source/
Dataset identity and never becomes a model input. No Journal5/schema bump is
indicated. Actual native build/inspect acceptance remains required verification.
All runtime protocol/numerical versions and formal Source/learning gates remain
unchanged.

Design disposition: READY, with no remaining blockers. Root adopts the review's
exact `loaded is context.projected` pin and actual plan/SAME-ledger validation
before warmup and again after finalization before publication.

## Final acceptance

Both independent final reviews are READY without findings. The original target
PASS with matched diagnostic pytest time 200.56 -> 80.97s and external wall
201.51 -> 81.48s (59.565% reduction), including initial warmup and final template
retirement. Actual receipts show one template, 576 fresh episode actors and zero
unretired tickets; the three normal cold preflight actors remain independent.
Collection union falls167.616 ->47.030s. All876G0/G2arrays,1,024actual decision
records and512selected Tree features/natural labels agree with retained actual
cold originals; all9inference manifests and1,024policy identities are unchanged.
Actual native Dataset build/inspect and independent raw-journal feature checks
pass without reader/schema changes or qualification claims.

Affected/explicit safety/frozen-Tree checks PASS378/6CUDA opt-in skips. Scoped
Ruff, mypy157, exact157module wheel and optional-free installed normal-worker/
CLI-help checks PASS. Final default full pytest PASS4,111/285skips/zero failures
across4,396cases, with37new lightweight checks. All353frozen code/config hashes
match. The observed full external time is1,026.74s versus1,151.24s previously;
populations differ by37new cases, so this is not a matched whole-suite comparison.
Current full-run Tree file case sum is77.870s versus199.312s previously.

The existing38heavy safety checks remain explicit opt-in. Root accepts the
implemented scope and retains all ADR0030 OS/thread/CPU/memory limits. Evidence
and commands: [verification](../verification/pytest-preloaded-cpu-launch.md),
section "Tree Dataset preparation extension". No simulator, numerical recipe,
formal Source/learning gate or normal application launch change is claimed.
