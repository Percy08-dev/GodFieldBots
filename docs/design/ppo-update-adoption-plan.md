# Standard bounded PPO update implementation plan

> For agentic workers: use superpowers:subagent-driven-development under the
> repository's custom bot_implementer/reviewer roles. Root owns docs/commits.

**Goal:** Automatically use the optimized Model4 update route with practical
bounded staging; retain explicit safe failure for extreme settings as the user
requested, rather than universally guarantee their capacity.

**Architecture:** One trainer/network, admitted packed fast path or finer streamed
frame windows, original worker/Source owners and unchanged formats. Baseline stays
private/reference; older frozen models stay on their existing paths.

**Stack:** Existing Python3.14/uv/NumPy/Torch/pytest/Ruff/mypy only.

**Spec:** [ppo-update-adoption.md](ppo-update-adoption.md), ADR0046.

## Global constraints

- IEEE FP32 losses/gradients, detachedFP64 metric sums, deterministic/CuDNN-disabled
  GRUs/MATH SDPA, existing checkpoint policy and final synchronization.
- Preserve PPO/GAE/reward/shuffle/selected endpoints/logical weighting/TBPTT/
  clipping/Adam units and grad=None/Adam-state presence.
- No increased64MiB tensor/model/binding/phase caps, dropped inputs, runtime OOM
  fallback, new dependency, Simulator edit, ordinary config/checkpoint/Wire field.
- Original creator, live allowance, Source capability/FD/snapshot/save/failure
  ownership and strict ordinary source-match resume remain authoritative.
- One code writer; no agents below root; preserve preexisting untracked files and
  all historical runs. Only root stages/commits; no push.

## Review focus

1. Practical long-prefix/default32 histories subdivide frames; extreme retained
   full-gradient graphs may reject and must not trigger rematerialization/fallback.
2. One burst event interval must not pad every historical frame to its length.
3. Prefix inputs retire while all suffix/checkpoint storage remains immutable.
4. CPU zero GPU lease must not become zero usable execution allowance.
5. Normal resume/campaign must not inherit fresh-only experiment authority.

### Task1: Bounded packing/planning and common network execution

Files: structured_learning.py, structured_network.py, hpo_update_planning.py,
hpo_update_execution.py and focused test_learning_update_* capacity/gradient tests.

- [ ] Add valid singleton1024/default32 and ragged event/slab regressions, plus
  extreme full-gradient/wide-FF rejection coverage; observe RED.
- [ ] Define private model-side chunk/window descriptions consumed by training
  planning, with full preallocation estimates and one-frame finest geometry.
- [ ] Retain the current packed fast path; implement streamed immutable prefix/
  suffix chunks and candidate staging through shared math, no second encoder.
- [ ] Test hard per-tensor bounds, lifetime peaks, detach/causality/own cards,
  empty branches/Adam and exact endpoint/tail weighting using existing tolerances.
- [ ] Run focused checks and report actual commands; independent task review.

### Task2: Normal dispatch and original worker allowance

Files: hpo_numeric.py, hpo_learning_process.py, hpo_owned_process.py/worker.py,
ppo_run.py/worker.py, structured_numeric_worker.py, relevant source/worker tests.

- [ ] Add RED implicit MLPMicro/sequenceShared tests, explicit reference tests,
  normal CPU/GPU cap transport and older-worker argv invariance.
- [ ] Derive normal route only in Model4 learner; keep experiment capability
  separate and legacy model/frozen exact checks untouched.
- [ ] Bind actual normal allowance through the existing launch creator; validate
  and scrub closed args before Model4 allocation; unchanged Wire requests/replies.
- [ ] Verify legacy/owned fresh/resume/campaign/native-Source, finite/cancel/FD/
  process failures and ordinary wrong-source refusal.
- [ ] Run scoped tests/Ruff/mypy and report; independent task/full reviews.

### Task3: Acceptance and live verification

Root owns docs/architecture/ADR0046/verification record and fresh run namespaces.

- [ ] Resolve blocking correctness/conformance findings; accept stable code only.
- [ ] Run gated bounded CUDA common and streamed cases, then normal no-private-
  spec fresh/pause-resume checks across models with original budgets/deadlines.
- [ ] Strictly audit original Source/numeric/receipts after live runs; do not use
  action/weight equality across independent trajectories as the acceptance gate.
- [ ] Small fixed-input unprofiled default-vs-private-reference screen checks
  common-path regression; record scope/source and do not infer training quality.
- [ ] Final independent review, current docs/ADR/evidence and verified local commits.

Root self-review: the five focus cases are assigned to tasks1/2; all interfaces
and compatibility constraints are in the bounded spec. Internal helper names are
left to the implementer; any consequential change returns to root before code.
