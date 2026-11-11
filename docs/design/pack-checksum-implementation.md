# Pack checksum read policy Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development; root owns design, docs, staging and commits, one bot_implementer owns code/tests.

**Goal:** Apply verified-on-acquisition Pack reading and expose bounded explicit full storage audits, preserving existing artifacts and learning semantics.
**Architecture:** Separate metadata admission, lazy verified cache acquisition and explicit audit. Keep current owners, formats and budgets; evolve runtime custody and native capability explicitly.
**Tech Stack:** Existing Python3.14 stdlib SQLite/zstd, pytest/Ruff/mypy, existing Rust/PyO3; no dependencies added.
**Spec:** [pack-checksum-read-policy.md](pack-checksum-read-policy.md), owner-supplied plan and ADR0035.

## Global Constraints

- SQLite schema1, Pack1, descriptor1, proof2, Run4, Update4, Dataset4, provider1/plan4, wire5 remain unchanged.
- New runtime evidence uses `packed-original-custody-v2`; verification policy is `pack-checksum-v1`.
- Native adds `card_instance_pack_checksum='pack-checksum-v1;custody2'`; existing capability values stay unchanged.
- Explicit audit defaults: 2048MiB managed memory, 7200s wall; mutually exclusive locator/dataset inputs.
- No Simulator changes, new dependency, formal campaign, HPO, final test, qualified fit, migration or deletion. Directory remains default.
- Preserve pre-existing untracked user files and fixture `.lock`; root alone stages/commits, children do not spawn agents.

## Review Focus

- Cache hits with multiple member references must perform no original/SQL/namespace I/O.
- Historical v1 evidence must bind saved correspondence digests without emitting fake v1 audit receipts.
- Alias references to one DB must each check expectations even after deduplicated scanning.
- Deadline/cancel/close uncertainty must never return success or refund a live owner.
- Native support refusal must precede any packed new-policy work, including Dataset/Tree/final paths.

### Task 1: Reader, custody/native/Dataset integration and explicit audit

This is one coupled interface change implemented by one writer after design review.
**Files:** `src/gf_common/group_pack_game.py`; existing `src/gf_training/group_pack_*.py`, `native_group_pack.py`, affected Source/lineage/Dataset/Tree/native preflight modules; new `src/gf_training/group_pack_verify.py`; `src/gf_cli/__init__.py` and its command module if needed; `native/src/` existing logical/provider/capability/Source/Dataset modules; relevant `tests/test_group_pack*.py`, `tests/test_native_group_pack*.py` and focused new tests. No config/dependency/docs ownership (root owns those). Request root ruling before expanding responsibilities.

**Interfaces:** Existing ArtifactSource/PackedOriginals/GroupLoan ownership is preserved. Factor index admission from `validate_database`; full validation is explicit. Add a bounded readonly verifier callable by CLI; use closed pure custody check/comparison helpers shared by Python/Rust consumers. No new learning eligibility API.

- [x] Write/run focused red tests asserting zero ordinary full DB hashes/integrity/eager Pack reads, verified acquire/reacquire, no cached-member I/O, corruption/oversize/missing-index rejection and unchanged close ownership.
- [x] Implement metadata admission, explicit DB expected-hash custody and lazy complete Pack verification; update Writer/publication/terminal paths to avoid duplicate whole verification while preserving seal-time SHA generation.
- [x] Write/run red historical-v1/v2 strict-evidence/correspondence, old Dataset4, native early-refusal and candidate/feature/numeric parity tests; implement shared checkers and all Python/Rust integrations with unchanged schemas/old-parent resume rejection.
- [x] Write/run red CLI/audit tests for locator/descriptor/pending/Dataset4, actual full corruption/missing/extra detection, same-inode aliases/conflicting expectations, bounded admission, timeout/cancel and close uncertainty. Implement serial full audit and JSON/nonzero semantics.
- [x] Verify changed paths with `.venv/bin/python -m pytest -q` selected affected modules, `.venv/bin/ruff check src tests`, `.venv/bin/mypy src`, and `cargo test --manifest-path native/Cargo.toml --locked` using the existing bounded test policy. Report pre-existing unrelated findings separately; build isolated source-matched native before FFI checks. No formal experiments.
- [x] Self-review against spec. Write full report with changed files, red/green commands/results, exact native build/FFI identity, known limitations and proposed doc corrections. Do not stage/commit.

### Root acceptance and evidence

- [x] Resolve architecture review before dispatch; record review and ADR0034 supersession.
- [x] Preserve baseline Reader revision `f59a2ab` for identical-original/HDD comparison. Instrument bounded readonly before/after phases and verify originals unchanged.
- [x] Run existing limited real two-update pause/resume and controlled Dataset/Tree checks where available, never shorten/qualify formal parents. Record missing real qualified inputs as NOT RUN.
- [x] Independent correctness/spec-quality and architecture-conformance reviews; resolve concrete blockers and recheck changed areas.
- [x] Fresh applicable verification, evidence/current architecture updates and coherent local commit. No push/publish.

Runtime acceptance additionally required the independently reviewed ADR0036
consumed-graph retirement and effect-aware owned-pool closure. The first failed
parent and all failed/inconclusive verification attempts remain recorded. Exact
final results, scope limitations and unchanged default are in the verification
record; no formal experiment or qualification is implied.
