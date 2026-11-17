# Collection CPU investigation implementation plan

> Execute with `superpowers:subagent-driven-development`; AGENTS custom roles,
> one code-writing child at a time, root-owned docs/configs/commits take precedence.

**Goal:** implement and execute the user-approved bounded P/T/H investigation.
**Spec:** [accepted design](collection-cpu-optimization.md), user-approved proposed
plan and `bot_architecture_reviewer` READY with explicit trust/lifetime conditions.
**Architecture:** tools-only installers around original owners/math; unchanged
normal defaults, no new dependencies or persisted authority. Python3.14/GIL,
uv, pinned NumPy/Torch, existing native/Simulator runtime.

## Global constraints

All exact populations, resource formulas, score/order/tie rules, lifecycle,
information and failure constraints in the design bind every task. No production
adoption, CPU inference switch, wider performance sizing, Simulator edits or
Source relaxation. Existing unrelated five untracked files and all old artifacts
must remain untouched. Current user-provided feature checkout is retained as
previously authorized; fixed absolute historical helpers/data remain available.

## Task 1: bounded experiment mechanisms and end-to-end tooling

Ownership: implementer `benchmarks/collection_cpu/`, relevant new tests only.
Root owns all docs/project configs/measurement configs/execution/freeze/commits.
One necessary baseline repair is also assigned: the private
`src/gf_models/inference_experiment.py` optional `_operands=None` compatibility
case, with no normal-path or learner-input behavior change.

- [ ] Write focused RED mechanism tests; implement P proof and boundaries, H
  two-task ownership, typed/byte/pinned transfer T variants and full reset/failure.
- [ ] Implement source-checked full learner32-game input preparation/G0G1 binding,
  replay/cold-warm restoration and separate bounded profiling.
- [ ] Implement one-invocation original actual composition,96MiB symmetric
  admission, closed bootstrap options, immutable launch sidecars/frozen files,
  baseline diagnostics and complete standard results without default changes.
- [ ] Implement deterministic trial materialization/selection/confirmation/combo
  aggregation and original audits. Missing/failed trials stay explicit; no
  invented winner, fallback or extra performance runs.
- [ ] Pass relevant mechanism/regression/static checks; self-review and provide
  exact commands/results, changed files, concerns and runnable root commands.
- [ ] Independent correctness and architecture code reviews; fix/recheck affected
  areas before the root freezes measured source and commits verified code.

## Task 2: execute the bounded measurement plan

Root owns controls/configuration; verifier may run specified checks after timing.

- [ ] Prepare fresh `runs/collection-optimization-v2/`, configs and current source/
  runtime/native/Simulator/input freezes, immutable IDs and original storage checks.
- [ ] Run three passive baseline diagnostics; prepare314/314/310 learner replay
  populations and original checkpoint hashes; inspect profile/CPU/copy evidence.
- [ ] Run transfer variants/three fixed-input pairs and separate profiles; choose
  T representative with the exact within-family rule; materialize actual arms.
- [ ] Run12exploratory P/T/H/control invocations. Freeze selection from collection
  seconds; issue up to24confirmation and18combination runs, no more54performance.
- [ ] After all timings, audit every original actual parent/file identity and
  context-bound receipt; preserve failures honestly and record all NOT RUN.

## Task 3: acceptance, evidence and final review

- [ ] Root writes complete verification/rankings/memory/limits and updates ADR0047
  and architecture to final implemented status only when evidence permits.
- [ ] Independent correctness and architecture reviews check final evidence,
  original Source/receipts, variant identity and shared-pilot accounting.
- [ ] Root verifies final links/hashes/whitespace, commits coherent verified
  milestones locally; no push/merge/publish. Final response identifies measured
  outcome, production-default status and material limitations.
