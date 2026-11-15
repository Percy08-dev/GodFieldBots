# HPO Precollection Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Root owns commits and documentation; custom role instructions override generic role/model choices.

**Goal:** Replace the first three resident precollection complete validations with one complete validation and retained-evidence rechecks, preserving later full validation and original ownership.

**Architecture:** One original online owner holds one scoped concrete validation carrier. Existing live checks stay fresh; exact history/native evidence is retained until confirmed close before mkdir. No persisted authority is added.

**Tech Stack:** Existing Python3.14/uv/NumPy/native/Torch, pytest/Ruff/mypy; no new dependency.

**Spec:** `docs/design/hpo-precollection-validation.md`, accepted preimplementation READY, and ADR0042.

## Global Constraints

- Scope: actual resident HPO only, calls1–3; calls4–5 remain complete validation.
- No dataset/checkpoint/wire schema, learning math, native Rust code, dependency or Simulator change.
- Source cap is `min(training.max_owned_bytes, source_budget_bytes, preprocessing.memory_budget_mib*1024**2)`; no budgets raised.
- Same original owner/plan/job/ledger/admission, current fresh guards and original work/cleanup deadlines; no stale/failed-carrier fallback or scope restart.
- One code-writing `bot_implementer`; independent `bot_reviewer` and `bot_architecture_reviewer`; root owns staging/commits. Preserve user files; no push.

## Review Focus

- Failure before collector try/finally must still own and close partial evidence.
- Same-byte/inode replacement, numeric inventory and native ELF changes must reject.
- Admission is bound after initial phase checks; copied/stale/changed evidence cannot mint authority.
- Real reservations and actual close precede refunds, detachment and collector Pack allocation.
- Default/custom/fresh callers and native save/termination authority retain their original behavior.

### Task 1: Scoped retained validation and regressions

**Files:** Create `src/gf_training/hpo_precollection.py`, `tests/test_hpo_precollection.py`; modify only `src/gf_training/card_instance_source.py`, `src/gf_training/hpo_ppo.py`, `src/gf_training/ppo_run.py`, `src/gf_cli/card_instance.py`, `src/gf_cli/hpo.py` and directly relevant existing tests.

**Interfaces:** private `train_hpo_ppo(...,_reuse_precollection=False)`; `_HpoOperations.precollection_scope(plan,ledger,*,config,deadline)` synchronous contextmanager; actual `_OnlineGameOwner.precollection_validation`; `validate_owned_collection(...,_finish_precollection=False)` and CLI forwarding. Exact responsibilities/lifecycle are in the spec.

- [ ] Write and run failing lifecycle/mutation/compatibility tests before production edits; record RED evidence.
- [ ] Implement scoped registration, complete first historical read, sealed reuse with all fresh guards and terminal close before mkdir. Opt in only standard resident CLI; defaultFalse remains old behavior.
- [ ] Implement charged metadata/native coverage, retained-original cleanup, outer exceptional cleanup and final termination guard under original cleanup allowance.
- [ ] Add original native G1 and failure/cancellation/accounting regressions from Review Focus; test unconfirmed cleanup and no fallback/restart.
- [ ] Run new tests and affected HPO/Source/resident/native/coordinator/CLI regressions, scoped Ruff and source mypy. Report exact PASS/FAIL/SKIP/NOT RUN and any boundary assumptions to root.
- [ ] Independent correctness and architecture-conformance reviews, fixes and scoped re-reviews; root commits verified milestone.

### Task 2: Acceptance, measurements and record

**Files:** Root owns design/ADR/architecture/verification documentation and ignored ledger/protocol; implementer may prepare bounded ignored measurement helpers only after explicit root brief. Verifier executes specified read-only checks; root owns physical run admission/control.

**Interfaces:** Task1 private toggle provides same-code baselineFalse/candidateTrue. Existing thin timing hook, actual termination receipt and original Source/numeric verification remain required.

- [ ] Freeze new run paths/config/seeds, code/native/Simulator/schema identities, private toggle and paired order; validate instrumentation delegation/error/restore behavior without GPU.
- [ ] Run the stable whole CPU suite, project static checks and fixed-math/compatibility checks once. Preserve unrelated historical artifacts.
- [ ] Execute3matchedMLP pairs and1pair eachGRU/Transformer,192/main2,4workers,16games/collection,2updates, original timers/budgets. Attribute full/recheck/close/collect times; do not mix failures or shorten conditions to claim acceptance.
- [ ] Audit actual Source/numeric correspondence, references, owners and original termination receipts. Require3->1 precollection replay and improved MLP median second-collect time; report cutoff and reproducibility limits.
- [ ] Independent final reviews of stable code and acceptance evidence; root updates architecture/ADR/verification, commits and reports scope/results without formal-HPO or strength claims.
