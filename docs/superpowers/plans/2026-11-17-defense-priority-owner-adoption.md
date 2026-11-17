# Defense Priority Owner Adoption Implementation Plan

> **For agentic workers:** Follow AGENTS.md custom roles. Root coordinates and
> owns design/project documentation; one bot_implementer owns Python/Rust/tests.
> Separate bot_reviewer and bot_architecture_reviewer perform final read-only
> correctness and implementation-conformance reviews. Children do not spawn.

**Goal:** Correct active defense candidate priority and Rainbow Curtain order
under the explicit 2026-11-17 owner authority, rejecting unsupported old rules.

**Architecture:** Shared pure rules/context analysis serves all current policy
consumers. Supplemental rules-v2 signature distinguishes behavior with unchanged
layout/generator. Python/native validation requires the exact current binding;
old rule identities reject explicitly without compatibility or data migration.

**Tech Stack:** Existing Python3.14/GIL, uv/NumPy, pytest/Ruff/mypy, optional
existing Rust/PyO3 extension. No new dependency or Simulator edits.

**Spec:** `docs/design/defense-priority-owner-adoption.md` and
`docs/contracts/owner-20261117-defense-priority.md`.

## Global constraints

- Architecture READY is required before dependent production implementation.
- Preserve every original failed experiment, including both issue0010 parents.
- No Simulator gameplay, real learning, HPO execution, GPU recovery or publishing.
- No hidden input, opponent catalog expansion, rejection probing or fallback.
- Current Simulator pin91d5926, catalog schema2/public-ja-v3, numerical math,
  generator/layout/schema identities and operational budgets remain unchanged.
- Root is sole stager/committer. Do not edit unrelated untracked analysis files.
- One implementer at a time owns `src/`, `tests/`, and necessary `native/` source
  changes; root owns `docs/`, project configuration and acceptance.

## Review focus

1. Positive-defense miracle stop/reflect/deflect must still combine numerically
   for a proven non-miracle weapon attack, while matching special roles exclude.
2. Old rule signatures must reject, never run new predicates or match mixed data.
3. Guardian/reflection/mixed-action/Fog/targeted/stale visible events must not
   acquire a false known incoming kind. Unknown handling is explicitly bounded.
4. Rainbow-first applies to physical order; effective CostCut binding and Flash
   do not accidentally erase, broaden or reorder that constraint.
5. New current Source/native binding needs an actual matching capability; old
   installed extensions must not quietly accept unsupported rule semantics.

## Task 1 Shared Python rules and public context

**Files:** `src/gf_common/card_instance_rules.py`, a focused adjacent context
module if needed, `src/gf_common/diverse_actions.py`,
`src/gf_models/card_instance_baselines.py`, `src/gf_models/refresh_heuristic.py`,
and existing current Python rule-binding checks where needed.
Tests include `test_card_instance_rules.py`, `test_card_instance_actions.py`,
`test_refresh_heuristic.py` and focused context/old-identity rejection regressions.

**Interfaces:** One closed current-v2 rule descriptor; exact current signature;
immutable actor-visible defense context; shared analysis.

- [x] Preserve original failed records; add old descriptor/signature rejection
  tests. Historical candidate execution support is explicitly out of scope.
- [x] Add the failing recorded defense/order and public-context regressions;
  run them and record expected RED output before changing production code.
- [x] Implement rules-v2 authority selection and minimal common context
  extraction under the approved information boundary.
- [x] Thread rules/context through generator and scoring without per-row scans;
  preserve shared numerical layout and reject old rule identities.
- [x] Existing read boundaries require current exact signatures;
  reject old/unknown/missing/mixed identities without adding historical plumbing.
- [x] Run focused rule/generator/baseline/source regressions and scoped static
  checks. Return assumption changes to root before proceeding.

## Task 2 Native current binding

**Files:** Existing `native/src/structured_controls.rs`,
`native/src/card_instance_dto.rs` and relevant native reader/control modules;
`native/src/lib.rs`/`native/build.rs` only if required for the capability;
`src/gf_training/native_backend.py`, current native pin/recipe assets and existing
capability tests as necessary.
Root reviews any additional files before implementation expands scope.

**Interfaces:** Exact current rule pins and an additive current-rule capability.
No dual pins, historical selector or generic Control extension.

- [x] Add meaningful failing native/Python capability and mixed-signature tests.
- [x] Update exact current pins without permissive either/or checks at nested
  records, historical support or native policy logic.
- [x] Build an isolated source-matched candidate with existing toolchain.
- [x] Validate a current-v2 fixture; reject old/mixed/unknown rules and an older
  native capability. Frozen old formal recipes remain unsupported, not rewritten.
- [x] Run relevant Rust tests, formatting and native reader regressions.

## Task 3 Integration and final acceptance

- [x] Run affected Python/static checks and the project suite with normal
  optional/integration selections; report precise failures/skips and limits.
- [x] Root updates ADR0048/architecture/issue0010/HPO manual to the implemented
  decisions and actual verification evidence.
- [x] Obtain independent correctness and architecture-conformance reviews after
  code stability. Resolve blockers; escalate a substantive disagreement or an
  issue surviving two repair cycles to the advisor.
- [x] Verify source-matched native activation or state exact remaining manual
  build requirements. Check optional-free startup/help and packaging if touched.
- [x] Root checks original failed DB hashes/stats remain unchanged, stages only
  task files, and makes coherent verified local commits. No push or experiment.

## Status

Authority and bounded design are written. Preimplementation architecture review
is DESIGN READY. Conformance must check a fully matched allowlisted public cause,
unknown handling for unsupported event suffixes, and exact current binding at
Python/native boundaries. The owner's subsequent compatibility opt-out supersedes
historical-support tasks: single current-v2 only, with explicit old rejection.
Python/native implementation and source-matched wheel activation are complete.
Correctness and conformance reviews are FINAL READY, including capability B1 and
all test-only deltas. Full HOST03 records7007PASS/586SKIP/2pre-existing argv
harness failures; one isolated test line repairs those and the entire GRU/launch
selection passes40. Root independently confirms68targeted passes, Ruff/mypy232,
installed native/source identity, package baseline and original-data integrity.
The verification document records whole-plus-delta acceptance and its limits;
no new fully green whole invocation is claimed. No actual training, Simulator
gameplay, capture or old data migration has run. Root's final local milestone
contains only the scoped reviewed changes; no push or publishing.
