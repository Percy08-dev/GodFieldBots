# Configurable HPO Game Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow user-configured HPO game concurrency above two, with explicit
worker/memory budgets and user-owned physical OOM tuning.

**Architecture:** Extend the existing HPO validators, resource calculation and
audited ledger; derive resident viewer capacity from the same declared ledger.
Keep ordinary/frozen execution, default 2/4, single-trial scheduling and actual
process/Source/cleanup ownership unchanged.

**Tech Stack:** Python 3.14, existing uv/pytest/Ruff/mypy and Rust/PyO3 tools;
no dependency additions.

**Spec:** [design](../../design/hpo-configurable-game-concurrency.md).

## Global Constraints

- Exact integer HPO counts in `1..2**63-1`, with workers at least twice games.
- Keep default two games/four participant workers and trial concurrency one.
- Managed RAM/CUDA admission stays; physical OOM tuning is the user's responsibility.
- No automatic fallback, retry with reduced concurrency, simulator edit, policy
  information expansion, learning-math change or persisted schema migration.
- No actual HPO/training/simulator games; experiments remain manual.
- Root owns docs/configuration/commits. One implementer owns scoped source/tests;
  independent reviewers are read-only. Respect unrelated uncommitted files.

## Review Focus

- Configuration acceptance must reach the physical resident pool and scheduler.
- Viewer isolation and both-player capacity must hold for 4/8 and odd worker caps.
- Larger counts must scale resource accounting and still refuse declared overruns.
- Failure above four workers must retain every pending/live owner until actual reap.
- Native/lineage readers and installed native preflight must agree with Python;
  ordinary/frozen limits and strict source/config identity remain intact.

## Task 1: Expand the complete HPO concurrency route

**Files:**
- Modify: `src/gf_cli/hpo_config.py`, `src/gf_models/structured_runtime.py`,
  `src/gf_training/hpo_lineage.py`, `src/gf_training/hpo_gpu_lease.py`,
  `src/gf_simulator_client/resident_pool.py`, `native/src/hpo_controls.rs`.
- Modify capacity propagation: `src/gf_training/hpo_resident.py`,
  `src/gf_common/hpo_resident.py`, `src/gf_common/card_instance_collection.py`,
  `src/gf_training/hpo_collection.py`, `src/gf_training/native_hpo.py`,
  `src/gf_training/native_structured_session.py`,
  `src/gf_training/native_card_instance.py`, `native/src/hpo_resident.rs`.
- Modify existing Rust receipt caller: `native/src/card_instance_source.rs`.
- Test: relevant existing HPO config/lineage/GPU lease/resident-pool/admission and
  scheduler modules; a narrowly named new concurrency test module is permitted.
- Root modifies docs, ADR0049 and current budget-probe presets separately.

**Interfaces:** Existing public signatures/serialized fields stay; validators
accept the expanded HPO count domain. Ordinary/frozen validators stay unchanged.

- [x] Obtain mandatory preimplementation architecture READY; resolve blockers.
- [x] Write meaningful failing checks for config/materialization/resource and
  native/lineage readers, higher physical pool/scheduler overlap, viewer bounds,
  failure/reap and legacy restriction preservation.
- [x] Run new checks to establish the old code rejects the supported expansion.
- [x] Implement only the accepted design; send newly discovered boundaries to root.
- [x] Run focused changed-area/regression checks and Rust HPO checks; report exact
  commands, results, changed files and limitations. Do not commit as implementer.
- [x] Root rebuilds/activates the matching native extension, verifies preflight and
  preset validation without actual games, and completes documentation/static checks.
- [x] Independent correctness and architecture-conformance reviews; fix and recheck
  concrete blockers, then root accepts and locally commits the verified scope.

## Execution record

- 2026-11-18: root classified this as a gated concurrency/resource-boundary change.
  Existing feature branch `feat/heuristic-refresh` is the shared working checkout;
  unrelated untracked user files are outside this change. No active GF experiment
  process was found during initial inspection. Design review pending.
- Initial architecture review BLOCKED: resident custody registration and
  Python/Rust retirement readers still capped owners at 2/viewer and 4 total.
  Root accepts the finding; design now passes explicit original-ledger/config
  capacity to every reader without adding a self-authorizing receipt field.
  Re-review pending; implementation has not started.
- Architecture re-review DESIGN READY: original-capacity propagation resolves
  the initial blocker; no additional blockers/open owner questions. Root accepts
  the design and delegates the single implementation task.
- Implementer TDD RED: `UV_CACHE_DIR=/tmp/gfb-hpo-uv-cache uv run --no-sync
  pytest -q tests/test_hpo_game_concurrency.py` gives6FAIL/7PASS in1.31s under
  the old caps and missing expected-capacity argument; implementation now active.
- Root TOML/default2/4 and D-originals path assertions PASS for all3presets;
  new design/ADR/plan local links PASS. Native rollback ELF and identity are
  saved at `/tmp/gfb-hpo-concurrency-native-backup-ceexjk9s` before activation.
- Implementer found one omitted existing Rust caller in
  `card_instance_source::manifest`; root adds it to the same original-parent
  capacity propagation. No new reader responsibility, format or owner is added.
- Independent focused architecture supplement is DESIGN READY for that existing
  caller; original parent capacity remains authoritative, and later exact
  Config/config_inputs/code comparisons stay required. Root authorizes the caller
  edit and its expanded-count/substitution regressions.
- Implementation accepted after both final independent reviews READY. The
  correctness review's optional8/16 coverage was added (test-only, final23PASS)
  and narrowly re-reviewed READY. No unresolved blocker remains.
- Root source Ruff/mypy232 and matching release/mimalloc native activation PASS;
  current3presets and temporary4/8+16GiB CUDA examples all pass actual CLI validate.
  Detailed staged regression evidence and exact commands are in
  [verification](../../verification/hpo-game-concurrency.md).
- Owner feedback: extra testing/re-review had become excessive. Root stops
  further checks/reviews after the already-complete evidence and finishes only
  documentation and the authorized local commit. Actual experiments remain manual.
