# SDD ledger — plan: docs/design/heuristic-refresh-implementation.md

## Identity and recovery

- Base: 4dd6b8ae822aceff38145c085d053beac21fef1a.
- Branch: feat/heuristic-refresh. Root owns this ledger/design/config/commits.
- Pre-existing untracked files: codex-analysis.jsonl,
  codex-session-01a1a10e-4c97-7ea0-8e55-6ca319ae6af9.md, codex-usage.py,
  codex_analysis.py, tests/fixtures/structured_history62_parent/.lock.
- Planning architecture agent: /root/refresh_heuristic_design, latest READY.
- No production implementation at ledger creation; no acceptance checks claimed.

## Rulings

- Ruling: dedicated branch in current checkout, not a second checkout — pinned
  editable/native package/runtime identities and existing large optional env
  must remain coherent — cost is shared checkout; only one writer is active.
  Default git write was sandbox-read-only; explicitly authorized local branch
  creation succeeded through scoped escalation.
- Ruling: use repository docs/design convention for spec/plan/ledger — aligns
  architecture/ADR workflow; no unrelated ignore-rule changes are required.
- Ruling: incremental windows use selected calculated total attack, not a
  supposedly causally linked declaration — user accepted observed-window proxy.

## Preflight conflict scan

| Tasks | Interface/file relationship | Disposition |
| --- | --- | --- |
| 1 self | policy behavior and identity tests use accepted exact recipe | coherent |
| 2 self | runtime and readers share options/label | explicit new branch required |
| 3 self | ordinary/model4 mixed config and reconstructed schedules | both included |
| 4 self | evidence before acceptance | coherent; not a strength experiment |
| 1→2 | common identity/model config consumed by CLI/worker/readers | sequential ownership |
| 1→3 | baseline binding consumed by schedule/registry | same descriptor |
| 2→3 | configured baseline Source consumed by PPO | complete before dependent PPO |
| 3→4 | mixed variant vs frozen formal recipes | opt-in nonformal; old AST exact |

## Progress

- Design: accepted user plan; persisted architecture recheck READY, no blockers.
- Baseline PASS: `.venv/bin/python -m pytest -q tests/test_card_instance_rules.py
  tests/test_card_instance_actions.py tests/test_card_instance_baseline_runtime.py
  tests/test_card_instance_cli.py` — 176 passed in3.46s; verifier
  /root/refresh_baseline_checks. No games/training launched.
- Task1: COMPLETE, /root/refresh_policy_impl, brief /tmp/gfb-refresh-task1.md;
  owns only the five new common/model/test files listed there.
  Report /tmp/gfb-refresh-task1-report.md:44 new +224 legacy PASS; Ruff/mypy PASS.
  Architecture conformance READY, /tmp/gfb-refresh-task1-architecture.md;
  independent correctness/quality READY, /tmp/gfb-refresh-task1-review.md.
  Root fresh acceptance `.venv/bin/python -m pytest -q
  tests/test_refresh_heuristic.py` PASS44 in0.44s. Local milestone1339b39.
  Principal retention table2097152bytes, tracked transform peak2295560bytes,
  local0.011340s diagnostic (facts preallocated; not process RSS/performance guarantee).
- Hardware preflight: ordinary sandbox NVML blocked; scoped read-only escalation
  sees RTX5070Ti16303MiB/2096MiB used. Approved Simulator release exists. GPU
  functional jobs still require production admission/ownership, not this query.
- Task2: COMPLETE, /root/refresh_runtime_impl; brief /tmp/gfb-refresh-task2.md,
  report target /tmp/gfb-refresh-task2-report.md. Root owns TOML presets/docs only.
  Initial frozen226Python PASS/22CUDA deselected;174Rust PASS/7ignored;
  native candidate /tmp/gfb-refresh-native-task2-final/gf_native.so immutable,
  ELFa4e51e954dd6663b885b99e6386c9e1fa37fc3f44eaf733c7f671ccae3c40ac2,
  source18e1424c0f1fccb386ad695ed6ba25b0917d8fab36697d19a52f2b4a16dd7436.
  Architecture READY. Correctness one P2: CLI partial validation imposed u64 cap
  on otherwise unbounded positive threshold. Root reproduced; fix round1 addressed,
  both scoped re-reviews READY.20new regression cases/63coveringPASS; fullsrc
  mypy198/RuffPASS. Root fresh runtime/native66PASS in3.24s, command in verification.
  Native unchanged after fix. Local milestone pending.
- Task2 actual functional:4natural games, zero failures, comparison_valid true;
  new-vs-old0W/2L, new-vs-Random2W/0L, no strength conclusion. Frozen311source/
  config files unchanged across games. Evidence runs/assessments/heuristic-refresh-20261115/t2-baseline-a01/.
  Baseline runtime before isolated CLI large-int correction; policy unchanged.
  Independent8Journal6/manifest/inventory/report/Replay audit PASS,72refresh
  selections reproduced exactly using only each actor frame's own event prefix.
- Task3: NOT STARTED.
- Task4: NOT STARTED.

## Supplementary Task1 diagnostic

Verifier /root/refresh_baseline_checks executed a timeout30s standalone corrected
default-generator/full18-possession selection diagnostic (17abyss-dart instances
and smile-dew, threshold1e9 to force renewal). PASS,23801candidates, generation
0.983s, policy0.575s, process peak77.78MiB. count_time/search_time cutoffs, no18-slot
sequence facts, so this does NOT prove all32768candidates/max-length behavior.
Selected own smile-dew. Initial diagnostic output formatting used nonexistent
Selection.label and failed after selection; corrected to policy_version and rerun.
No source changes, games, GPU job or native build.
