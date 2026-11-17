# Standard Model4 optimized update adoption

Date:2026-11-17. User-authorized adoption of normal MLP microbatch and sequence
shared-prefix execution, under [ADR0046](../adr/0046-standard-bounded-ppo-updates.md)
and [the accepted design](../design/ppo-update-adoption.md).

## Scope and current state

Design gate READY at `d2391d3`; implementation and final independent evidence
reviews are FINAL READY. Root accepts implemented source
`dece918e823172d75ab9e26f532b9e01810a65d7` after the checks below.
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

Main focused command:

```sh
UV_CACHE_DIR=/tmp/gfb-update-uv-cache uv run --no-sync pytest -q tests/test_ppo_update_adoption.py tests/test_hpo_numeric.py tests/test_learning_update_execution.py tests/test_learning_update_admission.py tests/test_learning_update_boundaries.py tests/test_learning_update_diagnostics.py tests/test_learning_update_benchmarks.py tests/test_hpo_progress.py tests/test_ppo_update_launch.py
```

141PASS/1SKIP. The real-worker host selection was:

```sh
UV_CACHE_DIR=/tmp/gfb-update-uv-cache uv run --no-sync pytest -q tests/test_hpo_wire.py tests/test_hpo_owned_wire.py tests/test_hpo_owned_run.py tests/test_hpo_progress.py tests/test_ppo_update_launch.py tests/test_learning_update_telemetry.py
```

124PASS/1SKIP. Source/lineage/campaign/native and older-worker selection:

```sh
UV_CACHE_DIR=/tmp/gfb-update-uv-cache GFB_HPO_NATIVE_CANDIDATE=/tmp/gfb-ppo-adoption-native-9yjjiue6/gf_native.so uv run --no-sync pytest -q tests/test_hpo_source.py tests/test_hpo_lineage.py tests/test_hpo_campaign.py tests/test_hpo_native.py tests/test_structured_numeric_worker.py
UV_CACHE_DIR=/tmp/gfb-update-uv-cache uv run --no-sync pytest -q tests/test_structured_numeric_worker.py
```

The first command yielded174PASS/1FAIL/4ERROR from the older test bootstrap's
stale broker argv; the narrow test-only correction yielded28PASS in the second.
The native candidate was a byte-identical installed extension copy, not a rebuild.

Independent review fixes: phase-local event arrays now trim as views before
gather/one-hot expansion, and suffix temporaries refine with prefixes while all
gradient windows remain held. A shared optional diagnostic factory disables
ordinary setup failures without catching control exceptions or authoritative
learning/Source/save operations. Its focused RED8→GREEN8 and affected real-worker
75checks PASS(43.10s); scoped Ruff/mypy12files plus final3affected files and
whitespace checks pass. The parent environment remains minimal.

The final guard regression selection was:

```sh
UV_CACHE_DIR=/tmp/gfb-update-uv-cache uv run --no-sync pytest -q tests/test_hpo_progress.py tests/test_ppo_update_launch.py tests/test_hpo_wire.py tests/test_hpo_owned_wire.py
```

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

## Normal live execution and original audits

PASS nine no-private-spec invocations, six distinct parents,192natural games.
For each MLP/GRU/Transformer: fresh continuous2updates, separate fresh pause after
update1, then immediate resume to update2. All continuous/resumed final parents
are honestly `incomplete/update_cap`; the intermediate states are `paused`.
These are bounded acceptance smokes, not completed formal training.

The root-prepared [normal validation plan](../../runs/ppo-update-adoption-v1/normal-validation-plan.json)
contains nine invocations/six parents across three models, with every pause
immediately followed by resume under its original deadlines. It retains the
192width/twolayer/seed17/workers4/concurrent2/games16×updates2/epoch1/D1M/wall600
and original caps. No experiment execution ref is passed. Final audits validate
each final parent once and all nine distinct invocation receipts separately.

The [root controller](../../runs/ppo-update-adoption-v1/run-normal-validation.py)
called the original coordinator/train entry exactly once per invocation, through
the original GPU reservation, collector, numerical worker, Source, save and cleanup.
It observed actual bounded stderr route facts: MLP `microbatch`, sequences
`shared-prefix`, with updates[1,2] for continuous,[1] for pause,[2] for resume.
Each invocation passed both absent-private-spec and actual-route assertions;
[execution results](../../runs/ppo-update-adoption-v1/normal-execution.json).

```sh
UV_CACHE_DIR=/tmp/gfb-update-uv-cache uv run --no-sync python runs/ppo-update-adoption-v1/run-normal-validation.py --freeze runs/ppo-update-adoption-v1/source-freeze.json
```

PASS all9subprocess exits0. No tests/audits/reviews overlapped these runs. The
[278-file freeze](../../runs/ppo-update-adoption-v1/source-freeze.json), checked
before/after/finally, SHA256
`56b566c031b6bad8128c22f190d9cb45dde2cc0d30881302791a7f139b548a56`, pins
source, helpers, configs/controller/plan, native binary and Simulator specs.
Controllers require that exact measured revision and fresh output namespaces;
the recorded commands identify completed invocations, not overwrite operations.
Simulator revision `91d5926a7572db327eaea3751e7c24f428120996`; native extension
SHA256 `0d5d0b242b1a08aafa22039f6c7c5921f2555cfcbf7c801c899e14573054aeda`.
Model4/collection7/proof4/registry5/pool1 and numeric6/Wire7 contracts retain
their original meaning. No Simulator, native or dependency change.

[Strict audit](../../runs/ppo-update-adoption-v1/strict-audits.json) PASS:
`validate_hpo_parent(require_complete=False,max_owned_bytes=2GiB,deadline=180s)`
on all6final original parents, with Source and complete numeric checks. Original
file identity before/after is unchanged:90files per continuous parent and123per
paused/resumed parent,639file instances total. All9original GPU termination
receipts are checked with their distinct invocation/campaign IDs and run IDs;
all5resources per invocation are reaped. Each resumed parent retains exactly
its original created/admission/execution/wall deadlines; wall budget remains600s.
No fake Source, rewritten producer or receipt, dropped episode or action/weight
equality gate across independent trajectories was used.

## Small fixed-input screen

PASS the18-update common-path diagnostic screen. It uses only batch0/G0 for each
model with three paired repeats; six warmups are separate.
Every run restores weights, Adam presence/state, CPU/CUDA RNG, PCG and step0.
Normal `_training_execution=None` is compared with private explicit baseline,
observing actual resolved branches. Historical input producer import remains
explicitly diagnostic; it does not relax ordinary resume source matching.

```sh
UV_CACHE_DIR=/tmp/gfb-update-uv-cache uv run --no-sync python runs/ppo-update-adoption-v1/run-fixed-screen.py
```

Exit0,111.68s; no tests/audits/reviews overlapped measurement. The
[controller](../../runs/ppo-update-adoption-v1/run-fixed-screen.py),
[request](../../runs/ppo-update-adoption-v1/fixed-screen/request.json) and
[result](../../runs/ppo-update-adoption-v1/fixed-screen/result.json) pin source,
runtime, historical inputs and excluded work. Forward/reverse/forward ordering,
same restored state and common adapter; actual normal branches verified on all9
measured normal updates and the three separate normal warmups.

| Model | Normal route | Median paired update wall-time reduction |
| --- | --- | ---: |
| MLP | microbatch | 86.57% |
| GRU | shared-prefix | 90.19% |
| Transformer | shared-prefix | 89.90% |

This screen finds no common-path regression on those inputs. Three pairs per
model, one batch and shared allocator caches limit the inference. These are
update-core diagnostic timings, excluding collection, full Source validation,
FD ownership, snapshot/save and retirement; they do not predict total runtime.

Earlier [optimization measurements](learning-update-optimization.md) remain
historical source-pinned evidence; they are not relabeled as measurements of this
new default or a quality/HPO/final-test pass.

Both final correctness/specification/quality and architecture/conformance reviews
are FINAL READY with no blocking findings. Read-only reviewers independently
checked all278frozen hashes, actual route/spec observations, deadline preservation,
receipt/audit counts and reproduced the paired screen summaries. They did not
rerun training, tests, CUDA checks or full Source audits. Root's final file-hash,
artifact-count, local-link and whitespace checks pass; no production source
changes followed the measured implementation commit.

NOT RUN: whole suite, repeated old84-update/18-run experiment, formal HPO,
learning-quality evaluation or final test. Extreme-setting capacity is not
guaranteed. This acceptance covers the authorized normal-path adoption.
