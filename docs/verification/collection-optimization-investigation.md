# Collection optimization investigation

Date: 2026-11-17. Investigation only; production defaults and source are unchanged.
The user requested measurement-led prioritization of policy evaluation/action
selection, communication/wait/decode, and candidate generation, plus examination
of applicable best practices. All game-scale comparisons below use **32 games**.

The strongest measured candidate is CPU actor inference for this small,
single-decision workload. The next investigation should isolate repeated parent
validation/recording and its effect on the event loop. A bounded candidate-analysis
memo saves only about 0.10 summed call-seconds per 32 games, so it ranks lower.
Simulator HTTP calls also deserve attention; the three proposed categories do not
cover the whole collection path. These are priorities for further experiments,
not accepted production optimizations or predicted whole-run speedups.

## Provenance and measurement scope

Bots revision: `0220d8be361adfb62c6af1853a17a9b7f4f5f405`; production implementation
remains `dece918e823172d75ab9e26f532b9e01810a65d7`. Simulator revision/specs:
`91d5926a7572db327eaea3751e7c24f428120996`. Model4/inference5, journal6,
collection7/proof4/registry5 and numeric6/Wire7 retain their existing meanings.
CPython3.14.10/GIL, NumPy2.5.4, Torch2.14.1+cu132, one Torch thread;
Ryzen7 9800X3D/RTX5070Ti. No dependency, Simulator, native, policy-input,
checkpoint or ordinary resume-compatibility change.

Artifacts: [`runs/collection-investigation-v1`](../../runs/collection-investigation-v1/).
The [root aggregation](../../runs/collection-investigation-v1/root-summary.json)
hashes its measurement inputs. Separate immutable freezes identify replay,
transport and generator populations; old producer metadata and original data
remain unchanged. The historical replay inputs remain explicitly diagnostic;
they receive no new Source or training qualification.

Measurements ran sequentially, without concurrent tests, audits or reviews.
The separate generator profile ran after unprofiled comparisons. The implementer
prepared only disposable scripts in the new ignored run directory. Root executed
them. Source investigation and official-document research preceded these runs.

Collection is parent wall time. Per-decision call totals overlap across two
concurrent games and are nested within other scopes. **Do not sum these into a
wall partition, divide them by concurrency, or subtract them from collection wall
to predict a speedup.** CPU timers, CUDA stream events and kernel trace durations
also have different scopes.

## Existing normal 32-game observations

One continuous two-update run per model, width192/two layers, seed17,
four resident workers/two concurrent games, 16 games per update. Accepted original
evidence and source hashes are in [normal adoption verification](ppo-update-adoption.md).
The [read-only derived analysis](../../runs/collection-investigation-v1/existing-analysis.json)
checks the original result/timing hashes before and after reading them.

| Scope, seconds per 32 games | MLP | GRU | Transformer |
| --- | ---: | ---: | ---: |
| Persisted collection wall (`collection_seconds`) | 27.585 | 27.636 | 28.103 |
| Policy evaluation/selection, summed calls | 11.262 | 11.576 | 12.293 |
| Parent-minus-worker remainder, summed calls | 6.856 | 6.664 | 7.406 |
| Candidate generation, summed calls | 3.538 | 3.465 | 3.588 |
| Neural forward host region, inside policy | 5.233 | 5.584 | 6.255 |
| Four largest forward calls, inside preceding row | 2.658 | 2.976 | 3.308 |

There are 611/611/603 total decisions and 465/465/461 neural decisions. The
four largest forward calls occur near initial neural use, but existing records
do not establish exact physical-PID/cache-cold attribution. Trimming them is a
descriptive distribution check, not a warm-only experiment or achievable saving.
Sampling itself totals only about 0.017–0.018 seconds; output return about
0.097–0.100 seconds. Optimizing sampling or output copies is a low priority.

`policy_ms` includes feature preparation, validation and non-neural opponents.
The forward host region includes PyTorch/Python submission and possible waits;
it is not GPU kernel busy time. Existing separate historical traces show 27,494
kernels and 1,212 HtoD copies in 56 original replay decisions, with 33.77ms summed
kernel duration. Those instrumented trace figures are not subtracted from the
new unprofiled timings to manufacture an idle/wait percentage.

## Experiment 1: CPU versus GPU actor core

Command:

```sh
.venv/bin/python runs/collection-investigation-v1/run-screen.py --execute
```

PASS: 18 fresh replay processes, three alternating CPU/CUDA pairs per model,
five warm loops plus a separate first loop each; 52.45s controller time.
Each loop uses the same 28 recorded own-visible decisions/three prefixes and
checkpoint for that model, including 5,187 candidate rows. This is a **fixed-input
inference-core screen**, not a new 32-game CPU collection trial. It excludes
tokenization, policy-level validation, sampling, IPC, HTTP and recording.
No conversion to a 32-game wall saving is justified.
The replay's maximum candidate count is1,552 and maximum prefix length20;
the additional live 32-game run reaches 8,579 candidates. CPU/GPU crossover for
that tail, longer histories or larger architectures remains unmeasured.

CPU affinity `{2}`, one Torch thread, original chunk256/IEEE FP32/no compile/no
padding/no buffer-reuse option; learner execution is outside the replay. Both
devices use the same weights and tokens; device-specific floating-point execution
can differ. No action-sequence equality or learning-quality claim is made.

| Model | CPU warm 28-decision median s | GPU warm median s | Median paired reduction | CPU/GPU first-call median s |
| --- | ---: | ---: | ---: | ---: |
| MLP | 0.074622 | 0.171358 | 56.45% | 0.003275 / 0.585760 |
| GRU | 0.077355 | 0.175738 | 55.98% | 0.003290 / 0.441080 |
| Transformer | 0.083723 | 0.192434 | 56.72% | 0.003493 / 1.270475 |

The first-call region excludes interpreter/model loading. Its GPU variation is
not controlled for driver/OS caches or clocks. Five loops within each process are
not five independent processes; three pairs provide descriptive evidence only.

[Plan](../../runs/collection-investigation-v1/plan.json),
[274-file freeze](../../runs/collection-investigation-v1/freeze-current.json),
[execution receipt](../../runs/collection-investigation-v1/execution-receipt.json),
[results](../../runs/collection-investigation-v1/screen-summary.json).
The freeze-time sandbox probe reports CUDA unavailable; the separately saved
host execution probe and successful GPU child results establish availability in
the measured execution environment. The probes must not be conflated.

**Applicability:** existing shared inference supports CPU, but current HPO config
explicitly requires GPU inference (`gf_cli/hpo_config.py:359`). Adoption needs a
reviewed change to that condition, actor admission/snapshot/cache ownership and
fresh/resume/evaluation conditions, retaining GPU learning. First compare complete
32-game collections under the intended parallel workload. Do not simply remove
the loader check or treat this replay as whole-route acceptance.

## Experiment 2: original transport and HTTP, 32 games

One fresh **normal** MLP two-update parent, unchanged protocol/budgets and no
private execution spec. Original methods were called once; only bounded passive
parent wrappers were added. Both normal microbatch route observations and actual
original termination receipts pass. 32 natural games, 611 decisions,
141,992 candidate rows (median11, maximum8,579); no dropped/error/unmatched
observations. Instrumentation can perturb timings; this is a decomposition run,
not an optimized-versus-control comparison.

```sh
.venv/bin/python runs/collection-investigation-v1/run-transport-diagnostic.py \
  --freeze runs/collection-investigation-v1/transport-freeze.json \
  --config runs/collection-investigation-v1/actual-configs/mlp.toml \
  --run-id f3448b5e-b227-42fc-851b-2046d6261cfb \
  --context-id e3655b24-5cb0-4c80-b20f-b25316c7ea41 \
  --output runs/collection-investigation-v1/transport-actual
```

PASS exit0. Collection callback wall 28.338s; coordinator wall 36.068s. The original
healthy terminal status is `incomplete/update_cap`, not completed formal training.

| Observed scope, summed seconds per 32 games | Seconds |
| --- | ---: |
| Policy evaluation/selection | 11.474 |
| All six measured public HTTP client methods | 10.178 |
| Parent-minus-worker remainder | 7.060 |
| Candidate generation | 3.623 |
| Journal/storage host | 2.612 |
| Capture begin-decision, separate from journal | 2.127 |
| SELECT thread completion to await return, inside remainder | 2.617 |
| Parent generated-table/report parse, inside remainder | 0.544 |
| Parent SELECT exchange thread CPU | 0.265 |
| Parent frame/delta dictionary construction | 0.039 |

The SELECT exchange thread's 17.337 summed wall seconds contain worker execution,
authorization-gate wait, framing and reply wait. Its 0.265 thread-CPU seconds do
not measure child serialization or all parent validation/CPU. Generated parsing
uses process CPU, which can include other parent threads.

The callback wall above has slightly different boundaries from persisted
`collection_seconds` in the earlier table; it is not a paired performance delta.
Completion-to-await-return median is 0.521ms, p95 7.813ms, maximum 495.851ms.
Only 81/611 exceed 1ms. The long intervals overlap other decisions' synchronous
capture and parsing work. This establishes an event-loop/scheduling investigation
target; it does not prove all 2.617s is polling or removable. Source inspection
confirms the normal SELECT Future currently polls with `asyncio.sleep(.001)`.

HTTP call totals: observe 3.660s/1,145 calls, act 2.739s/611,
actor_options 1.457s/611, events 1.100s/611, create 0.933s/32 and join 0.289s/64.
These include client validation, network/server work and scheduling delay, not
pure Simulator CPU. An async connection-pooled client already exists.

The remainder's timer boundaries are important: the parent starts before its
frame/bank/request checks and ends after decoded candidate/report/selection
validation. The child starts **after** request JSON decode and stops **before**
response serialization. Therefore parent-minus-worker is not an IPC wait timer.

**Applicability:** measure all-row validation/recording once per decision and
preserve successful validation in a bounded immutable owner-scoped object, if
reviewed evidence allows reuse. Parent worker, environment command construction,
journal and capture currently repeat portions of same-frame candidate checks.
Retain validation when crossing an actual trust boundary; do not delete checks
or use a hash-only shortcut. Moving CPU work off the event loop must account for
the enabled GIL, actual object ownership, deadlines and cancellation.

An event-driven Future bridge is plausible after that analysis, but is a smaller
specific hypothesis than eliminating the whole remainder. HTTP follow-ups include
overlapping independent same-revision events/options reads, or reducing duplicate
observation fetches through a reviewed control-only routing contract. Neither is
measured here. ADR0005's dedicated-fetch input boundary remains in force: do not
silently substitute Act response events/options/observations as policy inputs.
A new combined Simulator endpoint would be a separate Simulator task.

[264-file freeze](../../runs/collection-investigation-v1/transport-freeze.json),
[original result](../../runs/collection-investigation-v1/transport-actual/result.json),
[bounded observations](../../runs/collection-investigation-v1/transport-actual/transport-observations.json).

## Experiment 3: bounded candidate-analysis reuse

```sh
.venv/bin/python runs/collection-investigation-v1/run-generator-diagnostic.py --execute --profile
```

PASS six unprofiled passes: three alternating original/memo pairs, each replaying
the **same complete 32-game, 64-participant-journal, 611-decision cohort** from
the original normal MLP collections. SHA-checked published SQLite/pack/member data,
original frame/delta/EventBank/candidate-column decoders, original action and
NumericCosts settings, and each recorded search RNG before-state. Generation only;
no policy execution, codecs or HTTP. CPU affinity2; live clocks and count/search
budgets remain unchanged. Whole controller including separate cProfile: 35.38s.

The disposable memo stores `_CardInstanceSpace.analysis(family, refs)` results
only within one newly constructed decision space, maximum 4,096 entries, and
retains no cross-participant or subsequent-decision cache. It changes no
production class or method outside the lifetime of this diagnostic process.

Original generation median: 3.4575s versus memo 3.3604s per 611 decisions.
Median paired saving: 0.1021 summed call-seconds, **2.95%** (pairs 2.95/3.02/2.69%).
All 611 ordered-table digests/hashes/counts, reports excluding exactly the three
elapsed fields, and RNG states match within every pair. Each measured original
and memo pass also matches historical table/report content on this cohort.
This is fixed-input correctness evidence, not a gate on independent live game
trajectories, a general proof for time-limited searches, or a quality result.

Each memo pass makes 38,974 analysis calls, 28,161 hits/10,813 misses,
maximum 656 cached entries and no cap overflow. The effect is small despite many
hits. Separate cProfile identifies broader work in proposal offer/deduplication,
recursive walk, resource estimates, selection and table construction. In that
profile: 166,262 `offer` calls, 139,733 `estimate_costs` calls and 38,974 sequence
analyses. The profile slows original generation to about 11.41s; its times must
not be used as unprofiled savings or combined with the 3.46s baseline.

**Applicability:** memoization is feasible on the observed inputs, but low
priority. Further generation work should target measured hot-loop allocation,
repeated invariant resource calculation and traversal/deduplication, rather than
assuming sequence analysis dominates. Changes can alter candidate coverage under
wall-clock cutoffs, ordering and RNG use; generator identities/compatibility and
the acting-player visible-information boundary require review before adoption.

[Frozen original inputs](../../runs/collection-investigation-v1/generator-freeze.json),
[three-pair results](../../runs/collection-investigation-v1/generator-results.json),
[separate profile](../../runs/collection-investigation-v1/generator-original-profile.json).

## Official guidance and applicability

1. **Profile small GPU workloads before tuning arithmetic.** PyTorch explains
   that host launch gaps are especially relevant with small batches, and NVIDIA
   recommends profiling CPU/GPU balance and batching small transfers.
   [PyTorch profiling, pinned2.14](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html),
   [NVIDIA CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).
   The CPU replay is the strongest measured new lead. Cross-participant batching
   would change the current process information boundary; do not introduce a
   shared policy process receiving both private views as a routine optimization.
2. **Fuse/graph sufficiently large inference regions when launch cost dominates.**
   [PyTorch CUDA graphs,2.14](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs)
   describes static-address/control constraints. Broader frame/event/temporal
   graphing remains an unmeasured GPU-path option, with private storage scrubbing,
   variable candidates/events/history, memory and startup payback to resolve.
   Earlier scorer-only compile did not pay back; it is not evidence against every
   broader region. Merely moving warmup earlier does not remove its total cost.
3. **Avoid repeated tiny transfers, but test pinning and reuse in context.**
   [PyTorch transfer tutorial](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
   warns that manual pinning can be slower and small tensors may not benefit.
   [Earlier project comparison](inference-performance-comparison.md) found
   isolated reuse improvements of 9.6–14.7%, but mixed whole-run results; no reason
   to adopt that alone now. Partial compile, padding and TF32 likewise lack a
   useful established whole-route improvement here.
4. **Keep CPU work from blocking the event loop; await completions.**
   [Python3.14 asyncio guidance](https://docs.python.org/3.14/library/asyncio-dev.html#running-blocking-code),
   [Future bridge](https://docs.python.org/3.14/library/asyncio-future.html#asyncio.wrap_future).
   Apply only while retaining the original Future, Task, gate, I/O retirement,
   cancellation and cleanup owners. A mechanical await replacement is not an
   ownership design. Existing resident workers already avoid per-decision process
   creation; HTTPX already reuses an async client, as
   [HTTPX recommends](https://www.python-httpx.org/async/#opening-and-closing-clients).
5. **Inference-only autograd settings can reduce bookkeeping.**
   [PyTorch inference_mode,2.14](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad_mode.inference_mode.html)
   removes more bookkeeping than `no_grad`. Current cache code tracks parameter
   versions and retains state across decisions. An actor-only scoped trial must
   check those lifetimes; switching the learner or all tensor creation is not
   justified. This option is NOT RUN and has no claimed saving.

## Verification and next decision

PASS replay 18/18, normal instrumented parent 32 natural games/two updates/611
matched transport observations, generator 6/6 passes and separate profile,
all experiment frozen hashes before/after. Disposable script syntax/help and
read-only cohort checks pass. These checks establish investigation evidence;
they do not approve a production optimization.

PASS original new-parent audit: `validate_hpo_parent(require_complete=False,
max_owned_bytes=2GiB, deadline=monotonic()+180)` returns the expected run ID,
two updates/three chain records. Original `verify_gpu_termination` passes the
exact invocation context; all five resources are reaped. All 264 frozen file
SHA/stat identities remain unchanged across the audit. The
[audit report](../../runs/collection-investigation-v1/transport-audit.json) records
the exact command and limits. Independent `bot_reviewer` evidence review is
READY with no blocking findings: hashes, populations, ratios, timer scopes,
originals-once wrappers, per-decision memo ownership and information boundaries
were independently checked. Its optional collection-boundary label clarification
is incorporated above. The reviewer did not rerun benchmarks, application tests,
training or the Source validator. Whole application tests, new quality/HPO/final
test, full32-game CPU adoption comparison, event-driven transport A/B, HTTP A/B,
broader compilation and inference-mode comparisons are NOT RUN.

Recommended next decision: design and measure **CPU actor inference while
retaining GPU learning**, followed by parent same-frame validation/recording and
event-loop work, with HTTP subcalls included in the collection ledger. Keep
candidate-analysis memoization lower priority given its measured 0.10s call saving.
Normal production behavior remains unchanged by this investigation.
