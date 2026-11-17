# Standard Model4 optimized update adoption

Date:2026-11-17. User-authorized adoption of normal MLP microbatch and sequence
shared-prefix execution, under [ADR0046](../adr/0046-standard-bounded-ppo-updates.md)
and [the accepted design](../design/ppo-update-adoption.md).

## Scope and current state

Design gate READY at `d2391d3`; both implementation reviews are READY. Normal
runtime acceptance remains pending.
Baseline source `b3831a0`; user-provided feature checkout and preexisting five
untracked analysis/session/fixture-lock files are preserved. No Simulator or
dependency changes are intended.

The user explicitly accepts extreme-setting OOM/capacity risk. Normal defaults
are the target, not a guarantee that every formerly admitted full1024-gradient/
wide-network combination fits. Retained activation memory is still counted;
conservative caps and explicit preallocation/terminal failure remain. No new
activation-rematerialization/VJP mechanism, input dropping or old-loop fallback.

The earlier architecture counterexample remains valid: reference estimated
729,404,960B versus876,964,176B for fixed storage plus fourfold retained FF alone
in a full1024-gradient/wide-FF case. The finding's disposition is the user's
explicit requirement change, not a claim that staging removes those gradients.
Practical long-prefix/default32-gradient inputs and avoidable padded storage
expansion remain covered by bounded frame staging.

All Model4 paths must derive the normal route without a private experiment
capability, including legacy/public numeric, owned, fresh/resume/campaign and
native-Source numerical workers. Frozen older learners remain unchanged.
Ordinary source-matched resume stays strict; old paused parents require their
recorded revision, never a rewritten producer digest.

## Baseline check

PASS41/SKIP1,13.42s before implementation:

```sh
UV_CACHE_DIR=/tmp/gfb-update-uv-cache uv run --no-sync pytest -q tests/test_hpo_numeric.py tests/test_learning_update_execution.py
```

## Code verification

PASS focused mechanism/reference/capacity selection141checks/1skip(26.93s),
normal Wire6/Wire7/owned/route/telemetry124checks/1skip(89.41s), Source/native/
campaign174passes with an older-preload fixture1failure/4errors subsequently
corrected and affected older-worker28checks rerunPASS(12.42s). Explicit Wire6
route/restore1PASS(4.61s), private-reference CPU/telemetry4PASS(8.32s), final
focused57PASS(12.69s). These selections overlap; they are not a unique-total count.

Independent review fixes: phase-local event arrays now trim as views before
gather/one-hot expansion, and suffix temporaries refine with prefixes while all
gradient windows remain held. A shared optional diagnostic factory disables
ordinary setup failures without catching control exceptions or authoritative
learning/Source/save operations. Its focused RED8→GREEN8 and affected real-worker
75checks PASS(43.10s); scoped Ruff/mypy12files plus final3affected files and
whitespace checks pass. The parent environment remains minimal.

PASS HOST normal CUDA checks:

```sh
GF_RUN_CUDA=1 UV_CACHE_DIR=/tmp/gfb-update-uv-cache uv run --no-sync pytest -q tests/test_hpo_cuda.py
```

10PASS(8.69s), covering normal Model4 update/resume/RNG and shared-network/
participant behavior. Two additional synthetic GPU cases (GRU/Transformer)
exercise valid1024history/default32 gradients and a128event burst in the suffix,
implicit normal dispatch versus explicit private reference, existing gradient
tolerances/gradNone/optimizer-step checks. Both PASS; record
[streamed-cuda-check.json](../../runs/ppo-update-adoption-v1/streamed-cuda-check.json).
No performance or learning-quality conclusion comes from these component checks.
Optional-free current-source CLI help also passes with ML/native import blockers.

## Pending live verification

NOT RUN: normal no-private-spec continuous and pause/resume original Source/
receipt checks; common-path fixed-data screen; final independent evidence review.

The root-prepared [normal validation plan](../../runs/ppo-update-adoption-v1/normal-validation-plan.json)
contains nine invocations/six parents across three models, with every pause
immediately followed by resume under its original deadlines. It retains the
192width/twolayer/seed17/workers4/concurrent2/games16×updates2/epoch1/D1M/wall600
and original caps. No experiment execution ref is passed. Final audits validate
each final parent once and all nine distinct invocation receipts separately.

Earlier [optimization measurements](learning-update-optimization.md) remain
historical source-pinned evidence; they are not relabeled as measurements of this
new default or a quality/HPO/final-test pass.
