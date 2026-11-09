# ADR0029: Defer CPU compiler initialization while preserving determinism

Status: accepted; shared CPU runtime implementation and scoped/full acceptance
complete on2026-11-09 after both independent reviews and recorded verification.

## Context, authorization and classification

The owner asks to continue the investigated pytest startup optimization. The
installed supported Torch2.14.1 imports Dynamo/Inductor from
`use_deterministic_algorithms(True)` even for ordinary CPU eager inference.
The disposable investigation measures0.551s for that setting after a0.713s Torch
import. A tiny candidate can defer compiler initialization while preserving its
eventual deterministic flag; see the prior
[proposal review](../verification/pytest-runtime-proposal-review.md).

Root classifies this as a shared runtime-policy/compatibility boundary change,
requiring architecture review. The proposed increment changes only CPU runtime
setup in `gf_models.torch_runtime`, which serves legacy/structured inference and
numerical learners. CUDA keeps the original public API path. No process lifetime,
pool, forkserver, simulator contract, dependency or test budget changes belong
to this increment. Disposable probes are evidence, not an accepted foundation.

## Requirements and responsibilities

`torch_runtime` continues to own supported distribution/module/device checks,
one-thread setup, deterministic error mode, IEEE FP32/backend policy and lazy
optional imports. Its public interface and callers remain unchanged. Add only
a private CPU determinism helper; no policy/model/training package gains a new
runtime-management responsibility or reverse dependency.

For supported CPU runtime, preserve all of the following before returning:

- Core deterministic algorithms enabled, warn-only disabled, debug mode2.
- Current IEEE/TF32/AMP/cuDNN and one-thread policy, with unchanged NUMERICS.
- If Inductor config is already imported or is being imported, its actual
  deterministic flag is True in the runtime CALLER'S context after real import
  completion, matching the original public setter's context ownership.
- If Inductor config is absent, do not import Dynamo/Inductor merely for setup;
  any later ordinary config import must initialize deterministic=True.
- Repeated calls restore this fixed policy even after a conflicting environment
  value, core mode or already-loaded compiler flag. This matches the existing
  unconditional deterministic policy rather than exposing a new user option.
- Unknown distributions/modules/device bindings retain their original refusal
  before the CPU helper mutates runtime environment. No fallback or exception
  suppression makes an unsupported runtime appear valid.

`runtime_metadata()` and importing the helper module itself must still avoid
loading optional Torch, preserving installed baseline/help behavior.

## Concrete CPU setup and import ownership

After existing supported-runtime/device checks and thread setup:

1. Set the process environment `TORCHINDUCTOR_DETERMINISTIC=1`.
2. If the exact `torch._inductor.config` key is present in `sys.modules`, call
   ordinary `importlib.import_module` for that module, then set its
  `deterministic=True` in the caller context. This cached/import-lock-aware path must wait for an
   in-progress import; do not write into a possibly partial module or fabricate
   sys.modules entries. A present failed-import sentinel must still fail closed.
3. Use public `torch.set_deterministic_debug_mode('error')` to enable the same
   core deterministic mode without eagerly loading compiler modules.

The order is consequential: setting the environment first ensures every import
starting after the absence check sees1; joining an import already in progress
handles a value read before this setup. Do not introduce an import hook, patch
Torch functions, call private core setters or bypass compiler config validation.
The implementation is pinned to the already-supported Torch2.14.1 builds; the
private environment/config detail must be reconsidered at a dependency upgrade.

The environment value remains1 for the process lifetime. That deliberate side
effect supplies the deferred module's initialization input; restore-on-return
would break the requirement. Existing children use explicit clean environments
and perform their own runtime setup. Do not propagate new policy-visible data
or change the child environment/bootstrap contract in this increment. Existing
module-wide Torch/backend settings remain process-wide as before; this does not
promise isolation from unrelated concurrent code changing Torch configuration.

Torch2.14.1 compiler config user overrides are ContextVars. A real original-API
counterpart proves that a helper thread calling `use_deterministic_algorithms`
after a loader thread imports default=False seesTrue in its own context while
the main thread still seesFalse. Therefore the in-progress-import regression
must inspect the flag in the caller before return, and inspect import completion/
module identity separately. Requiring cross-thread override propagation would
strengthen the original contract. When the module is initially absent, the
new environment-backed default can beTrue in future contexts; equivalence for
unrelated threaded compiler configuration is not claimed. Supported application
model/learner operations use their owned context; actual eager numerical/resume
and CUDA checks remain required. Revisit at any new compiled/threaded consumer
or Torch upgrade rather than altering cached config defaults in this increment.

CUDA initialization continues to call original
`torch.use_deterministic_algorithms(True)`. Device availability/CUDA visibility,
actual worker creation, synchronization and failure behavior remain unchanged.
CPU setup followed by CUDA setup in one process must retain the same flags.

## Semantics, identity and compatibility

The intended change is scheduling of module initialization, not numerical or
persisted semantics. Keep NUMERICS, runtime metadata, dtype/tolerance/algorithm/
seed settings, model/artifact/dataset versions and existing readers unchanged.
Saved Source hashes and original meanings remain authoritative. Preserve active
resume's exact code/runtime identity checks: older-code paused experiments are
not silently relabeled or resumed under changed bytes. Fresh test/experiment
directories record the new actual producer bytes/revision through existing
provenance. No real Source/learning-quality acceptance follows from synthetic
fixtures or dependency smokes.

Equivalence is an acceptance question, not an assumed result. In particular,
optimizer construction may later import Dynamo; changes to import order must
not alter saved initial/final weights, Adam/RNG state, candidate choices or
CPU/GPU parity. Any numerical or required compiler-state discrepancy returns
to Root for design review; do not loosen tolerances/assertions, shorten budgets
or bump a format/recipe to conceal a failed preservation claim.

## Failure and alternatives

Ordinary module import/configuration failures propagate before the caller can
publish a ready policy. Do not silently select eager/lazy fallback based on an
unverified runtime. No participant process, reservation, cancellation deadline
or persistence owner changes. Actual guarded CUDA setup stays original.

The simpler debug-error setter alone is rejected: the investigation demonstrates
later compiler deterministic=False. Retaining eager setup is the fallback if
equivalence or useful performance cannot be established. Forkserver and global
participant services remain separately reviewed alternatives; do not bundle them.

## Verification and acceptance plan

Architecture reviewer must approve the import-completion/environment ownership
and compatibility policy before implementation. One implementer owns only
`src/gf_models/torch_runtime.py` and relevant new/runtime tests; Root owns docs,
evidence, coordination and final acceptance. Reviewers do not edit files.

Required checks:

- Real fresh subprocess tests show no compiler import during CPU setup and actual
  CPU eager inference; late import yields deterministic=True. Core/backend/thread
  flags match ordinary setup. Exercise already-loaded false/true flags, conflicting
  environment, repeated calls, failed imports and supported-version/device guards.
- A controlled actual import-in-progress test proves setup joins module completion
  and restores the flag before returning; no fake config module is the foundation.
- Preserve original caller-facing CUDA path and lazy baseline import. Run relevant
  original CPU network/policy/learner/legacy/IPC/resume regressions; actual CUDA
  inference/numerical restore checks where the available hardware supports them.
- Run the same original42structured-PPO cases with original collection/update/
  assertion budgets, comparing wall time against231.79s and real before/after
  initial/final arrays, search/candidate/selection records. Capture child startup
  versus first-selection cost to detect merely moving imports into inference.
- Preserve recorded data/config compatibility with the repository's applicable
  original-data proofs. No simulator integration/version changes are involved.
- Because a shared application runtime changes, run a whole default pytest
  invocation after source stabilizes; record its exact population and all failures/
  skips honestly. Compare with historical full timings only as reference data.
- Ruffsrc/tests/native, mypy src, source-matched application wheel and isolated
  mandatory-only baseline/help checks. CPU-only Torch distribution testing, actual
  compiled-kernel behavior and future-thread reconfiguration are reported as
  separate limitations when unavailable/outside the supported application path.

Design-stage acceptance questions were the exact import-join recipe, preservation
of metadata/NUMERICS after measured equivalence, optimizer/import behavior,
same-target speed and whole-suite effects. Their dispositions and remaining
explicit limitations are recorded below.

## Design review disposition

Architecture review is READY with no blocking findings. Root accepts its checked
reasoning: ordinary CPython import sets initializing state before insertion, so
the environment-first/key-present/import-module sequence joins the real module
lock and avoids patching a partial module. A present None sentinel still raises.
The process-wide environment policy, unchanged CUDA branch and existing source/
resume identity boundaries are explicit. NUMERICS/schema preservation remains
conditional on final equivalence evidence; final conformance/correctness reviews
and the prescribed checks are still required.

## Final acceptance and retained limitations

Root accepts after both final independent reviews report READY and all planned
supported-path checks pass. Only the shared runtime module and four new actual
subprocess cases change application/test behavior. The caller-context clarification
is backed by an original-API counterpart; no cached config default or test fake
becomes production behavior.

The same42CPU module cases PASS, wall231.79to195.84s (15.51% in single observations).
Original MLP/GRU/Transformer child startup improves from1.29-1.53to0.79-0.91s;
compiler modules stay absent through the first two actual decisions, with exact
candidate/search/numerical choices. All2112G0/G2weight/optimizer/RNG arrays and
768decision-content projections agree across six normal parents. All18inference
manifests differ only in producer metadata; actual changed policy/checkpoint IDs
remain preserved and independently validated. This supports unchanged NUMERICS
and formats, not a same-producer-byte or learning-quality claim.

HOST CUDA inference/numerical update/restore PASS6 plus a same-process CPU→CUDA
flag/math check. CUDA retains its original setter. The full4305case invocation
is4058PASS/247SKIP/4warnings/zero failures, external1177.96s. Historical1738.71and
1390.16s comparisons are reference data with different populations/reporting/
revisions; all original4217statuses remain. Ruffsrc/tests/native, mypy157,
source-matched157module wheel and mandatory-only worker/help checks PASS.
Actual application/test bytes remain frozen throughout.

The private environment/config assumption stays tied to supported2.14.1.
CPU-only Torch distribution, actual compiled kernels and unrelated future-context
configuration are NOT RUN. Later context defaults can differ from old eager
module defaults; the guarantee is the owned caller's deterministic execution and
ordinary later import policy. Revisit before a compiled/threaded consumer or
dependency upgrade. No process/forkserver pool or simulator contract changes.
Exact commands, retained probe failures, identity proof and limitations are in
[verification](../verification/lazy-cpu-deterministic-runtime.md).
