# Further pytest runtime proposals — 2026-11-09

Status: read-only investigation and disposable CPU feasibility experiments.
No application/test implementation or architecture decision is adopted here.
Current accepted runtime remains ADR0027/0028 at revision `f1dbb55`.

## Scope and corrections

The owner's proposal considers session-level participant reuse, forkserver
preloading, deferred compiler imports and more in-process unit coverage. Root,
the read-only verifier and architecture reviewer assessed these against current
code, Python3.14.10 and installed Torch2.14.1+cu132. All157application module
hashes remain unchanged. No simulator, config or dependency changes occur.

The previously reported92heavy participant imports are a subtotal for13major
collection calls, not the entire module/suite or learner process count. The
48fresh MLP loads are deliberate independent fresh-interpreter reference
coverage: two batches each contain24MLP/4Random/4heuristic participant views.
`completed_execution` explicitly uses `pool_cpu=False`, and full MLP fresh versus
pooled-resume comparison reuses that completed reference read-only. Removing
these48loads changes that test's independence; a redesigned reference/coverage
strategy is possible but must be explicit.

The proposed `(92-4)*(1.33-.037)` estimate is approximately113.784summed child
elapsed seconds. It is neither measured CPU consumption nor wall-time savings.
It assumes all92can become4while retaining coverage and ownership; that assumption
is not established. Parallelism and different model/cache paths also matter.

Earlier ADRs excluded cross-collection services/forking from their accepted
increments. Those scoped exclusions do not forbid a newly reviewed design.

## Confirmed Inductor import trigger

Actual installed `torch.__init__.py:1900-1903` unconditionally imports
`torch._inductor.config` inside `use_deterministic_algorithms`, sets its
`deterministic` flag and then sets core deterministic algorithms. The current
`src/gf_models/torch_runtime.py:35` calls this function. In a fresh CPU process:

| Measured scope | Seconds | Compiler modules afterwards |
| --- | ---: | --- |
| Initial `import torch` |0.713235|not loaded|
| `use_deterministic_algorithms(True)` |0.551050|Dynamo and Inductor config loaded|
| Whole current `torch_runtime('cpu')`, Torch already imported |0.551837|loaded|

This confirms the proposed cause, not merely an inference from cumulative imports.
The public API also documents coupling to Inductor's deterministic mode:
[PyTorch2.14 API](https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html).

The public `set_deterministic_debug_mode('error')` is a potential eager-mode
alternative, but direct substitution is not equivalent in this installed version:
core flags remain deterministic=True/warn_only=False/debug_mode=2, yet a later
Inductor import produces deterministic=False. Tiny eager outputs and gradients
still match; that cannot establish compiler equivalence.

The pinned installed `torch/_inductor/config.py:1052` initializes its deterministic
flag from `TORCHINDUCTOR_DETERMINISTIC == '1'`. A second disposable candidate sets
that environment value before compiler import and uses the public debug-error
setter. Its measured setting call is about12microseconds; whole runtime after
Torch import is0.000792s. Compiler modules stay absent through the tiny eager
forward/backward, and an explicit later compiler import yields deterministic=True.
Core flags, tiny outputs and gradients agree exactly with the ordinary path.

This is a version-pinned design candidate, not a production patch. The probe
deliberately imports the compiler later to inspect its state; full controller
times therefore include the deferred work and do not show eliminated total cost.
Existing-compiler state, actual MLP/GRU/Transformer inference, optimizer imports,
updates/resume, CUDA/default routes and saved-state parity remain NOT RUN for this
candidate. Production must not rely on a test monkeypatch or fake sys.modules
entry. Any accepted implementation needs architecture review and preservation
of numerical/runtime identities and existing compatibility/refusal behavior.

## Forkserver preload feasibility

Current participants launch through `subprocess.Popen([python,'-E','-m',...])`.
Calling `multiprocessing.set_forkserver_preload` does not change that route.
Python3.14's API is meaningful only for forkserver children and requires setting
preloads before launching the server:
[Python3.14 documentation](https://docs.python.org/3.14/library/multiprocessing.html#multiprocessing.set_forkserver_preload).
The proposal's3.15documentation has newer API details; this project uses3.14.

The disposable experiment creates three sequential, genuinely fresh CPU children
per controller. Each uses the current original `torch_runtime('cpu')`, constructs
a small seeded Linear model, and executes forward/backward. Different modes run
in independent clean-environment controllers with CUDA visibility empty and
thread limits1. No CUDA context is initialized. External times include controller
startup and standard controller/server/resource-tracker shutdown:

| Three-child variant | External seconds |
| --- | ---: |
| `spawn`, no preload |5.030338|
| `forkserver`, preload Torch |2.857974|
| `forkserver`, preload Torch and Inductor config |1.794150|

All9actual children exit0, with3distinct PIDs per variant, exact toy outputs/
gradients, deterministic core/compiler flags and CUDA uninitialized. Loaded-module
snapshots prove the specified modules are inherited. Torch-only preload leaves
about0.51-0.58s of compiler-related runtime setup per child. Combined preload
reduces child runtime setup to about0.0017-0.0018s; first-child start/join still
includes1.307s of template creation, and later children cost0.074/0.081s.

These single-observation toy timings do not measure the actual framed participant
bootstrap, candidate selection, Journal, Source, numerical resume, ledger,
startup/failure/cancellation cleanup, peak memory or whole pytest. They demonstrate
CPU runtime feasibility, not fork safety or end-to-end acceptance. Background
threads and accelerator initialization require separate proof; see
[PyTorch2.14 multiprocessing guidance](https://docs.pytorch.org/docs/2.14/notes/multiprocessing.html).

The first sandbox attempt could not bind the forkserver's local Unix socket.
Its raw logs are retained under `sandbox-attempt/`; a permitted HOST rerun
completed. That environment restriction is not a forkserver design failure.

## Ranked assessment and required design

1. **Deferred compiler initialization:** smallest promising scope. Preserve
   deterministic behavior for both current eager and later compiler paths;
   verify the exact pinned-version environment/API contract before adoption.
   Existing real-network/learner parity and compatibility gates remain required.
2. **CPU module-only forkserver template:** promising if actual participant
   fresh-fork coverage is acceptable alongside separately retained fresh-exec
   checks. Each participant child can still be truly reaped before an update.
   New design needs fixed CPU environment before preload, no CUDA/native-thread/
   private-policy state in the template, a separately owned/charged template,
   framed pipes and actual process-handle lifecycle, server failure handling and
   confirmed teardown. This is more than changing a start-method setting.
3. **Targeted in-process conversions:** inspect concrete cases, with limited
   remaining opportunity because network/cache/gradient, GAE/loss/learning and
   encoder unit tests already execute in-process. Preserve tests whose subject
   is subprocess isolation, protocol, lifecycle, persistence or actual resume.
4. **Session-wide four-participant service:** technically possible through a new
   design, with broad responsibility changes. Current `ReservationLedger` and
   PPO reject updates while participant reservations remain live. Separate actual
   resident ownership from active loans; define parent/split/viewer/RNG/history/
   checkpoint reset, memory, failure containment and independent reference
   coverage. Do not release a lease before actual reap to bypass those guards.

Both read-only reviewers agree that the proposal contains useful directions.
Large benefit with a small code change is plausible for compiler-import avoidance
pending semantic verification. It is not established for replacing the entire
collection ownership model with a generic session ProcessPoolExecutor. No change
to the accepted231.79s module timing or whole-suite timing is claimed here.

## Reproduction and retained evidence

All scripts and raw results are under
`/tmp/godfieldbots-runtime-proposal-review-20261109`: `runtime_steps.py`,
`forkserver_probe.py`, `run.py`, raw stdout/stderr, `run.log`, `summary.json`
and preserved initial sandbox failures. Root's bounded permitted HOST command:

```sh
timeout --signal=INT --kill-after=10s 180s .venv/bin/python /tmp/godfieldbots-runtime-proposal-review-20261109/run.py > /tmp/godfieldbots-runtime-proposal-review-20261109/run.log 2>&1
```

PASS: observed import trigger, direct-substitution non-equivalence detection,
version-pinned environment candidate's toy flags/output/gradient, all9fresh-child
toy comparisons and real joins, and157application source hashes unchanged.
Independent verification audits raw results and scope; it does not rerun them.
Project pytest, actual CUDA, peak-memory study and production changes are NOT RUN
for this investigation. Root records this assessment without adopting an
optimization or changing an accepted ADR.

## Subsequent implementation

The owner subsequently requested proceeding. The separately reviewed and accepted
[ADR0029](../adr/0029-lazy-cpu-deterministic-runtime.md) adopts CPU compiler
deferral, preserving actual caller-context determinism and the original CUDA
setter. Same42case wall231.79→195.84s and full4305case PASS4058/247skips are
recorded in [implementation verification](lazy-cpu-deterministic-runtime.md).
The exploratory results above remain historical; forkserver/global services
are not adopted by that increment.
