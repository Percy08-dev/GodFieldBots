# PPO runtime: wall time and CPU attribution

2026-11-15. This is measurement only; no application, simulator, model, learning
rule, dependency or concurrency setting was changed. Code revision2149a5e has the
same application bytes as the accepted4a3022a measurements in
[the resident-worker verification](hpo-resident-workers.md).

## Existing wall-time breakdown

Exclusive phase intervals come from existing monotonic callbacks in all three
resident samples per model. Values are means of two-update invocations, in seconds.
A phase contains its internal waits and concurrent work; these are not pure
kernel, network, or CPU times.

| Model | Collect | Source validation/conversion | Update | Sync stage | Cleanup | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 58.97 | 19.25 | 17.39 | 10.06 | 9.25 | 119.51 |
| GRU | 62.24 | 20.21 | 61.61 | 11.12 | 10.11 | 170.34 |
| Transformer | 62.70 | 20.48 | 58.72 | 11.22 | 10.17 | 168.43 |

Startup, save and outer preparation/exit account for the rest. For MLP, fresh
collection averaged99.39s versus58.97s resident, while sync+cleanup added about
19s. This explains why fewer worker launches yield a smaller whole-run reduction.
MLP collection increases from17.92s in the first batch to41.05s in the second;
Source validation/conversion increases5.90s to13.35s. Both batches contain16games,
but trajectories differ. The phase name alone does not identify the extra work.
Update includes numeric checks/read, optimization and synchronization; it is not
pure GPU compute. No exact hot function or causal explanation follows yet.

## New resource-only measurement

One fresh MLP resident diagnostic invocation uses the same192/main-depth2,
mixed16/threshold12, seed17, fixed recipe, D1,000,000/max_updates2 and original
600/480/540/60second clocks. Parent/save budgets stay1GiB; originals and parents
use new profiling-v1 paths. It ends **incomplete/update_cap**, uses314decisions,
and has no objective or learning-strength qualification.

The unmodified reviewed probe retains preflight, admission, Source and cleanup.
A disposable stdlib observer samples its process tree every250ms. Kernel wait4
records actual process exit and resource usage; the sampler uses each PID plus
start ticks and adds only self utime+stime, never parent cutime/cstime twice.
No GPU synchronization, production hooks, ownership bypasses or new dependency
were added. Private command arguments and policy inputs are not logged.

- Wall: **129.014s**. Kernel CPU: **147.132s** (125.661user/21.471system).
- Mean utilization: **1.140core-equivalents**, or **7.128% of16logical CPUs**.
  Affinity also permits16CPUs. The observer could not resolve a quota; a separate
  root read saw `/sys/fs/cgroup/cpu.max = max 100000`.
- Sampled self CPU146.06s, with1.07s kernel-minus-sampled residual. The observer
  itself used1.24CPU seconds, separately excluded from the probe's kernel total.
- 517 samples,25PID/start identities, max9simultaneous processes,0read errors or
  dropped rows. This can miss short processes, first prefixes and exit tails.

| Phase | Wall seconds | Process-tree CPU seconds | Main attribution |
| --- | ---: | ---: | --- |
| Collect | 62.15 | 78.76 | Coordinator50.97; actors25.25; servers2.33 |
| Source validation/conversion | 19.64 | 19.70 | Coordinator19.64 |
| Update | 18.35 | 18.34 | Learner17.77; coordinator0.53 |
| Sync stage | 10.39 | 10.45 | Coordinator9.60; actors0.85 |
| Cleanup | 10.41 | 11.41 | Coordinator8.36; actors2.76 |
| Save | 2.67 | 2.60 | Coordinator2.24; learner0.36 |
| Startup | 4.74 | 3.03 | Learner2.94 |

CPU seconds can exceed wall seconds because processes overlap. Phase attribution
splits250ms sample intervals at callback boundaries, so small boundary differences
are approximate. Global sampled CPU by corrected role is coordinator92.63,
actors29.17, learner21.53, servers2.51 and unresolved0.22seconds. This identifies
coordinator-heavy work and a learner-heavy update stage. It does **not** prove
single-thread serialization, a GIL bottleneck, exact hot functions or IO/GPU wait.
Parent CPU includes native/executor threads; leader state/wchan is not all-thread
state. Wall minus CPU is not an IO-wait metric.

The separate shared-device observer records562samples: GPU utilization mean6.06%,
median5%, max22%; sampled memory peak4348MiB. Desktop use is included. These are
not per-process kernel timings or an exclusive device measurement.

## Interpretation and next targets

Low whole-machine CPU utilization is consistent with roughly one active core,
not evidence that all stages are idle. Source validation/conversion consumes
about one coordinator core-equivalent on average during that phase. The coordinator
also consumes most CPU inside collect, sync and cleanup. Measuring individual
validation/replay, archive/shard inspection, cache pin and reference-release
calls is the next useful step before changing worker counts or batching.
The learner consumes about one core-equivalent during its update, but the
present observation does not separate numeric validation, tensor preparation,
Python dispatch, native work, GPU launch/spin or GPU completion waits.

No optimization is implemented here. In particular, original Source validation
or pin/cleanup guarantees must not be skipped to improve these numbers.
The129s run is not a new speed comparison: it is a separate observation with
its own startup/caching/background conditions. Observer CPU alone cannot identify
its wall-time overhead. Timed candidate cutoffs may change trajectories.

## Instrumentation verification and evidence

The resource launcher passed stdlib synthetic child/grandchild/thread, CPU/phase,
exception exit1 and forwarded SIGINT exit130 checks, help and compile. Independent
review accepted the bounded observer before actual execution.

One postprocessing defect was corrected offline: an exiting process with an
empty command line could replace its earlier actor classification with `other`.
Two-pass PID/start-time role resolution now preserves known roles, keeps PID
reuse separate and reports conflicting roles as unresolved. A synthetic regression
covers these cases. Original measurement/raw files were preserved; corrected
attribution has the same146.06CPU seconds and129.014phase-wall seconds,0conflicts.
Independent review checked hashes, arithmetic, fix and interpretation.

Pinned Python3.14 cProfile did not support the attempted simultaneous per-thread
profiles (`ValueError: tool 2 is already in use`); mixed-thread thread_time
attribution was not trustworthy. That failed synthetic attempt is retained only
as unsupported text. The actual run used no profiler; optional profile UUID was
not executed. No private CPython hack or additional profiler was installed.

Local raw evidence under `runs/hpo-resident-workers/assessments/`:

- `timing-phase-breakdown-a04.json`, `timing-update-breakdown-a04.json`;
- `profiling-declaration-v1.json`, `profiling-launch-freeze-v1.json`;
- `profiling-v1/{launch,observe,verify_stdlib}.py` and README;
- `profiling-v1/resource-a01/measurement-request.json`, `measurement.json`,
  `processes.jsonl`, `attribution-corrected.json`, helper stdout/stderr;
- `profiling-v1/resource-a01/probe-report/` original request/result and actual
  GPU termination receipt; `profiling-v1/gpu-resource-a01.csv`.

Root independently verifies the real termination context,4worker/1learner reaps,
empty cache/evaluation reference maps, unchanged actual helper and the declared
incomplete two-update stop. No termination is inferred from the sampling data.
No production test rerun is required for this ignored, disposable observer;
accepted application bytes remain unchanged and the prior full suite remains
its separate evidence. No formal HPO, final test or strength claim is made.


## Inside collection: five repeated parent validations

Follow-up selective observations identify the repeated call path. Each collect
invokes `card_instance_source.validate_owned_collection` at five sites:
`ppo_run.py:685` before the collector, `card_instance.py:753` at collector entry,
`:314` at corrected-run entry, `:357` for the owned GroupPack directory, and
`:495` after game/group cleanup before publication. All reach
`hpo_source.validate_hpo_parent`. That full validator reopens each previously
committed batch and verifies original Source proof, numeric data and selected
prefix context. G0 has no prior batch; G1 replays the previous16games.

A thin timer on **only this function** measured one more unchanged MLP point.
It calls the original exactly once with original arguments/results/exceptions,
records wall/thread CPU/caller, then restores the exact module attribute.
No global call tracing, new Task/Future, math/Source/checkpoint change, validation
skip or ownership bypass occurs. The actual probe returns normal original
termination, incomplete/update_cap2/used314 and empty reference maps.

| Collection batch | Phase wall seconds | Five validation calls | All other work |
| --- | ---: | ---: | ---: |
| First, no history | 20.640 | 0.277 | 20.363 |
| Second, one historical batch | 43.731 | **29.130** | 14.601 |
| Both | 64.371 | **29.407 (45.7%)** | 34.964 |

The second-batch five calls take5.896/5.824/5.846/5.786/5.779seconds. Four occur
before games (23.352s), one afterward (5.779s). Their total current-thread CPU is
20.117s; executor/native CPU on other threads is outside that thread clock.
All five observed spans are disjoint, depth0 and completely inside collect,
so their wall sum is valid. The four CLI calls share an immediate helper caller;
the higher-site attribution follows the audited static call order, not a captured
complete call stack. The remainder includes games, startup, RPC, attestation,
archive and publication; it is not asserted to be pure gameplay.

The earlier resource-only process samples independently align: first collect
19.416s, beforefirstobservedserver0.460/server-presence18.250/after0.706; second
collect42.732s, before23.298/presence13.250/after6.184. Server presence is a250ms
sampled process interval, not an exact gameplay timer. Thus the extra second-batch
collection time is substantially the repeated historical Source validation,
not inferred simply from low overall CPU. This does not establish that all
checks are interchangeable or safe to remove.

The low-overhead thin run has helper wall130.942s (GNUtime131.17s), near the prior
resource-only129.014s. No randomized observer-overhead control was performed, so
their difference is not an exact instrumentation-cost estimate. Observer records:
12total calls including2outside collect,0errors/drops. Original Source/data/math,
workload, seed/model/limits and original clocks are unchanged. The separate older
sys.setprofile experiment took410s; its large global dispatch overhead means
its absolute times are **not** used as normal-runtime evidence. It remains
retained diagnostic evidence only. Runtime-evidence preflight also repeats, but
its normal-runtime total was not individually timed by the thin observer.

Raw follow-up data: `collect-profiling-declaration-v2.json`,
`profiling-v3/thin-a01/{launch-freeze.json,kernel-time.txt,helper.stdout,helper.stderr}`,
`spans/{spans.jsonl,summary.json}`, `collection-breakdown.json`, `probe-report/`.
The earlier intrusive experiment is under`profiling-v2/collect-a01/` and is never
pooled with normal timings. The thin timer's stdlib delegation/exception/restore/
logging-failure checks and independent prelaunch review passed. No production
optimization was implemented. The next design question is which repeated
verification can reuse still-owned immutable validation evidence; that requires
preserving the existing Source, mutation and failure boundaries.
