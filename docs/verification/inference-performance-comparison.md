# Inference performance comparison — 2026-11-16

Physical comparison and strict original-data audits are complete. Both final
independent reviews are FINAL READY and root accepts the approved scope.
[ADR0044](../adr/0044-inference-performance-comparison.md) and
[the bounded design](../design/inference-performance-comparison.md) define scope.

## Result and recommendation

Keep the current production defaults for the complete two-update workflow.
Reusable typed input buffers improve isolated fixed-input inference by9.6–14.7%,
but this does not establish a consistent whole-run speedup. The actual paired
median changes are−0.09% MLP,+2.21% GRU and−0.92% Transformer (candidate minus
control). These are three-pair descriptive results, not confidence bounds.

The useful inference-specific candidate is `packing=reuse`,chunk256,IEEE FP32,
no padding and no compile. Larger chunks use more memory for small gains;
TF32 is effectively tied; padding combinations vary; scorer compilation adds
startup cost without positive warm saving in the matched populations. No
production default, parameter, architecture, learner math or training precision
is changed. TF32 learning quality remains a later HPO question.

## Runtime, provenance and protocol

- Baseline Bots revision: `4dccb08`; reviewed implementation: `6abadcf`;
  final measured source: `17b2c1c5f31c1df75ec77b0a46073bc386af35c2`.
- Simulator source/contract: `91d5926a7572db327eaea3751e7c24f428120996`;
  corrected card-instance v2,public-ja-v3 catalog schema2,Model4,inference5,
  Numeric6 and actor9. Existing version meanings remain intact. Binary/spec/config
  hashes are frozen; checkout HEAD does not independently attest a prebuilt binary.
- CPython3.14.10/GIL,NumPy2.5.4,Torch2.14.1+cu132,CUDA13.2;
  RTX5070Ti16,303MiB,observed NVIDIA driver610.88;
  AMD Ryzen7 9800X3D,8cores/16logical CPUs under WSL.
- Replay uses CPU affinity `{2}` and one Torch thread. Actual trials retain
  affinity0–15,width192/two temporal layers,seed17,four workers/two games,
  16games per collection x2updates,epoch1,D1,000,000,wall600s and original
  admission/execution/finalization clocks. Source2GiB,numeric512MiB,collector2GiB,
  parent/save1GiB,managed RAM24GiB/CUDA12GiB are preserved.
- Each actual parent stops as healthy `incomplete/update_cap`; none completes D
  or authorizes an HPO objective,formal Source,fitting,quality or final-test claim.
- All timing populations run sequentially,with no concurrent tests,audits or
  reviews. Profiler and verbose cache-proof populations are separate. Source
  audits start only after all timing stops. Seeds alone do not promise deterministic
  model training or equal trajectories.

Artifact root: [`runs/inference-comparison-v1`](../../runs/inference-comparison-v1/).
`actual-plan.json` fixes the18unique run/context IDs and alternating order.
`actual-freeze.json` binds302source/helper/config/input/native/Simulator files,
SHA256 `37fdb681778cbe19bedfc161eb1d7becd7efb708a5c4f4928a010c996f80fd8b`.
Requests identify experimental execution separately from base production identity.
The exact original options bypass the installer; default construction/transfer/
enqueue order and once-per-decision positions remain the control.

## Fixed checkpoint/input screen

Existing full-original parent validation and the Numeric6 reader prepare three
own-visible prefixes (`0:3,2:3,12:19`) per model. Each replay has28decisions,
5,187candidate rows,counts1–1,552 (median7),at most21interval events and a20-frame
prefix. No rewards,hidden state or management facts become tensor features.
These are ordinary recorded inputs,not a maximum-capacity stress population.
Full checkpoint/shard/input hashes,recorded producer and runtime resources are
in `inputs/<model>/manifest.json`; G0 checkpoint manifest hashes are:

| Model | SHA256 |
| --- | --- |
| MLP | `0799377d2c3c21537348dcc73c1bc40a89e6e6f5411d59406bbf246d7e172dc4` |
| GRU | `d97a76e0bf45d391258e6523e6deef2b078a5a46fb21fe160fd4c919f8824595` |
| Transformer | `02567e440340a54611c22c766a5c0b05922023001178ddd7c186a992cfa54afb` |

The main MLP screen has12one-factor pairs,nearby alternating controls and20warm
loops per fresh process. Values below are steady-loop median host milliseconds
per decision; each row has its own matched control. Memory is the candidate's
CUDA allocator allocated/reserved peak,MiB,excluding driver/context memory.

| Factor | Control ms | Candidate ms | Change | Allocated/reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| bulk | 6.687 | 6.086 | -8.98% | 43.69/56 |
| reuse | 6.964 | 5.943 | -14.65% | 43.69/58 |
| chunk512 | 6.678 | 6.795 | +1.75% | 47.39/58 |
| chunk1024 | 6.588 | 6.591 | +0.05% | 54.95/58 |
| chunk2048 | 6.583 | 6.567 | -0.23% | 62.76/100 |
| bucket32 | 6.778 | 6.814 | +0.52% | 43.60/56 |
| bucket256 | 6.658 | 6.746 | +1.32% | 43.63/56 |
| feature-align8 | 6.976 | 7.065 | +1.28% | 43.66/56 |
| feature-align32 | 7.053 | 7.176 | +1.74% | 43.66/56 |
| tf32 | 6.963 | 6.961 | -0.03% | 43.60/56 |
| compile-default | 6.973 | 8.296 | +18.97% | 40.76/56 |
| compile-reduce-overhead | 6.977 | 8.589 | +23.11% | 71.82/122 |

Warm loop process CPU and CUDA-event elapsed times are recorded separately in
every result. The CUDA events bracket the stream interval,including GPU idle
while the host submits/validates; they are not isolated kernel busy time. Sample
matched-process candidate values (CPU ms / CUDA-event ms per decision):
- bulk: 6.047 / 6.084.
- reuse: 5.930 / 5.942.
- chunk2048: 6.543 / 6.565.
- tf32: 6.944 / 6.959.
- compile-default: 8.254 / 8.294.
- compile-reduce-overhead: 8.556 / 8.586.

Combination checks against reuse256: reuse+chunk2048 saves2.72% in one pair but
raises allocated/reserved peaks to62.84/100MiB; reuse+TF32 changes−0.05%;
reuse+TF32+row32/K8 changes−5.60% in one pair whose reference is slower than the
other reuse references. These small/variable increments do not outweigh the
simpler256/IEEE choice. Fixed-input confirmation gives GRU7.136→6.246ms (−12.47%)
and Transformer7.789→7.040ms (−9.62%).

### Compilation/cache behavior

The compiled region is the actual shared candidate scorer. Frame,event-GRU,
temporal/cache control,validation,normalization and sampling remain explicitly
eager; this is not a general comparison of compiling the whole network.
Both modes produce three backend graphs initially and no additional backend
graphs in the20steady loops. Fullgraph breaks fail; no eager fallback is reported.
Default disables CUDAGraphs; reduce-overhead enables them,marks each scorer
iteration and clones real-row outputs outside compilation before retaining them.
It remains replay-only because captured private input storage has no owned
resident scrub hook.

| Mode | Empty-cache backend seconds | Populated-cache backend seconds | Matched populated warm control/candidate ms | Initial prefix+installer extra | Finite payback |
| --- | ---: | ---: | ---: | ---: | --- |
| compile-default | 2.104 | 0.788 | 7.204/7.209 | 1.237s | none: warm saving ≤0 |
| compile-reduce-overhead | 1.829 | 0.477 | 7.034/7.107 | 1.200s | none: warm saving ≤0 |

Unprofiled populated-cache processes and separate verbose diagnostic runs use the
same explicit caches. Logs show three actual FX-graph cache hits per mode.
First-call host latency is about1.20–1.25s with populated caches; empty-cache
initial prefix+installer excess is2.79/2.85s for default/reduce-overhead. Backend
time is measured separately from wrapper/installer overhead. Payback includes the
complete prefix through the last shape compilation and installation,requires two
later graph-free loops and positive saving,and excludes earlier interpreter/
artifact-loading startup. No finite recovery is justified in the matched cache
populations. Driver/OS caches and GPU clocks are not reset or controlled.

### Kernels,shapes and submissions

Fourteen separate profiles retain raw Chrome traces and operator shapes. Each
profile covers two passes/56decisions,including first and warm work; compiler
profiles may also include capture/tracing. Their elapsed times and memory are
excluded from timing-population comparisons. Actual trace categories are used
below; coarse operator `copy_events` is not counted as a unique GPU transfer.

| Profile | Device kernels | HtoD copies | DtoD copies | cudaGraphLaunch |
| --- | ---: | ---: | ---: | ---: |
| original | 27,494 | 1,212 | 564 | 0 |
| bulk | 27,026 | 372 | 564 | 0 |
| reuse | 26,474 | 372 | 564 | 0 |
| chunk2048 | 27,038 | 1,176 | 574 | 0 |
| tf32 | 27,520 | 1,212 | 564 | 0 |
| reuse-tf32-bucket32-feature8 | 27,332 | 372 | 676 | 0 |
| compile-default | 26,970 | 1,212 | 578 | 0 |
| compile-reduce-overhead | 27,170 | 1,212 | 1,108 | 73 |

Reuse reduces observed HtoD submissions1212→372 (69.3%) and device kernels
27494→26474 (3.7%) in these profiles. Launch API names/counts are retained in
`profile-summary.json`; reduced-overhead has73actual CUDA graph launches.
Summed kernel durations are33.77ms original,33.01ms reuse and33.50ms TF32 across
the56profiled decisions. These instrumented populations are not subtracted from
unprofiled host time to invent a pure wait percentage.

Representative observed matrix operands include state `[1,5499]@[5499,192]`,
card `[18,166]@[166,64]`,hand `[18,64]@[64,128]`,card projection
`[1,3456,71]@[1,71,18]`,and candidate rows `[1..256,26]@[26,192]`.
The row32/K8 experiment changes the last pair to `[32/64/96/256,32]@[32,192]`;
it does not align the separate card-projection K71 operand.

IEEE profiles execute `cutlass_80_simt_sgemm_*` plus GEMV/small-N kernels.
TF32 executes1,198CUDA activities named
`cutlass_80_tensorop_s1688gemm_64x64_*` (1,212in the padded combination),while
batch-one GEMV remains. Thus TensorOp implementations actually execute; this is
stronger evidence than a TF32 flag. Nsight/hardware instruction/utilization
counters are unavailable,so no Tensor Core occupancy/utilization percentage or
blanket claim for all operators is made.

## Actual route: level1,whole-run phases

All18runs complete32natural games and two updates each:576natural games total,
no game failure. MLP/GRU each have314learner decisions/611game actions per run;
Transformer310/603. These counts are observations,not an action-sequence or
numerical-equality gate. Every group has three alternating-order pairs.

| Model | Control helper median s | Reuse helper median s | Median paired delta s | Median paired change |
| --- | ---: | ---: | ---: | ---: |
| mlp | 38.947 | 39.190 | -0.035 | -0.09% |
| gru | 55.908 | 55.902 | +1.210 | +2.21% |
| transformer | 56.768 | 57.031 | -0.515 | -0.92% |

A difference between group medians is not the median of paired differences;
MLP and Transformer illustrate that distinction. Neither is a confidence claim.
Individual pair helper values (candidate minus control) remain:

| Round/model | Order | Control s | Reuse s | Delta s |
| --- | --- | ---: | ---: | ---: |
| 1/mlp | original/candidate | 40.068 | 39.213 | -0.855 |
| 1/gru | candidate/original | 55.908 | 57.277 | +1.369 |
| 1/transformer | original/candidate | 59.357 | 57.558 | -1.800 |
| 2/mlp | candidate/original | 38.811 | 39.190 | +0.380 |
| 2/gru | original/candidate | 54.693 | 55.902 | +1.210 |
| 2/transformer | candidate/original | 56.768 | 57.031 | +0.263 |
| 3/mlp | original/candidate | 38.947 | 38.912 | -0.035 |
| 3/gru | candidate/original | 55.952 | 55.002 | -0.951 |
| 3/transformer | original/candidate | 56.034 | 55.519 | -0.515 |

Disjoint parent phase medians,seconds. Each individual ledger reconciles exactly;
medians across rows do not form an additive partition. Learner-update includes
the full command and nested child work; save below is parent persistence/commit.

| Phase | MLP control/reuse | GRU control/reuse | Transformer control/reuse |
| --- | ---: | ---: | ---: |
| startup | 1.939/1.987 | 1.915/1.975 | 1.941/2.000 |
| collect | 27.710/27.798 | 27.507/28.023 | 29.203/29.110 |
| preparation | 1.344/1.392 | 1.377/1.417 | 1.387/1.363 |
| learner_update | 6.271/5.962 | 22.628/22.461 | 22.188/22.428 |
| save | 0.077/0.069 | 0.073/0.076 | 0.076/0.073 |
| synchronization | 0.543/0.552 | 0.617/0.631 | 0.650/0.670 |
| cleanup | 1.429/1.371 | 1.346/1.286 | 1.322/1.427 |
| other | 0.039/0.037 | 0.038/0.037 | 0.037/0.038 |

External observer wall includes interpreter/preflight/final serialization excluded
from the helper ledger. CPU is whole reaped workload-tree user+system time,not
one process or GPU busy time. Peak PSS is sampled and shared RSS can double-count.

| Model | External wall median control/reuse s | CPU median control/reuse s | Maximum PSS control/reuse GiB |
| --- | ---: | ---: | ---: |
| mlp | 39.294/39.507 | 57.765/58.213 | 7.269/7.269 |
| gru | 56.227/56.233 | 74.809/75.166 | 7.300/7.345 |
| transformer | 57.192/57.348 | 77.212/77.435 | 7.389/7.382 |

Highest sampled shared-device memory is5,016MiB (4.90GiB),including unrelated
existing device allocations. It is not workload-attributed memory and cannot be
subtracted using an unrelated baseline.1sdevice/PSS and100msRSS samples can miss
peaks. Managed reservations remain soft estimates; all original caps pass.
The96MiB/physical-worker experimental envelope covers all six cache identities
plus transition/padding storage within those caps; it is not96MiB measured use.

## Actual route: level2,collection internals

Median per-run **summed host call-seconds** below are nested/concurrent across the
two games. They are not a wall partition and must not be added to level1 or to
each other across nesting levels. The parent request contains worker processing
and communication/decode/queueing; generation and policy are inside the worker.
Readiness,HTTP and other collection spans are not fully partitioned by this thin
observer,so no unmeasured wall remainder is assigned to them.

| Scope | MLP control/reuse s | GRU control/reuse s | Transformer control/reuse s |
| --- | ---: | ---: | ---: |
| Parent input/frame | 1.645/1.657 | 1.661/1.708 | 1.643/1.631 |
| Parent worker request | 23.374/22.560 | 22.763/23.050 | 26.019/26.164 |
| Communication/decode host,inside request | 7.651/7.102 | 7.190/7.152 | 7.391/7.615 |
| Public candidate generation,inside worker | 3.712/3.758 | 3.723/3.866 | 3.745/3.708 |
| Neural policy,inside worker | 8.677/8.498 | 8.527/8.419 | 11.668/11.537 |
| Non-neural policy,inside worker | 2.289/2.342 | 2.338/2.346 | 2.278/2.280 |
| Journal/storage host | 2.565/2.557 | 2.554/2.641 | 2.554/2.529 |

Neural-only policy internals:465SELECTs/run for MLP/GRU and461for Transformer.
These six sequential host windows remain nested inside neural policy,with
validation/other policy overhead outside them. Packing includes construction/
copy submission; forward includes CPU launch and possible waiting,not GPU busy.

| Neural host scope | MLP control/reuse s | GRU control/reuse s | Transformer control/reuse s |
| --- | ---: | ---: | ---: |
| tokenization_ms | 1.303/1.325 | 1.317/1.342 | 1.324/1.325 |
| input_pack_transfer_host_ms | 0.769/0.449 | 0.738/0.468 | 0.744/0.452 |
| forward_enqueue_host_ms | 5.015/5.087 | 4.908/4.948 | 8.010/8.061 |
| sync_host_ms | 0.158/0.156 | 0.155/0.162 | 0.151/0.156 |
| return_ms | 0.105/0.114 | 0.102/0.112 | 0.104/0.121 |
| sampling_ms | 0.023/0.024 | 0.023/0.026 | 0.024/0.024 |

Packing host windows shrink,while the complete actual neural-policy medians
improve only about1–2%; collection and complete-helper timings remain mixed.
The single-process replay improvement therefore does not justify an overall
training-throughput claim under four-worker GPU sharing.

Actual trials exercise the shortlisted eager IEEE reuse setting. Other options
have replay/profiling evidence and are not claimed accepted for actual training.
This experiment does not test compiled frame/history regions,maximum-capacity
stress,other widths/seeds/devices or TF32 learning quality.

## Verification,failures and limitations

PASS: all18public full-original
`validate_hpo_parent(...require_complete=False,max_owned_bytes=2GiB,deadline=...)`
audits,stored numeric correspondence,checkpoint/registry/retention checks and
actual termination receipts with exact context/run namespaces. Every parent has
2updates/3generations. Before/after SHA256 and dev/inode/mode/size/mtime/ctime
identities remain unchanged for90physical files per parent,1,620file instances.
`strict-audits.jsonl` contains the independent verifier's result. This grants no
complete-run or learning-quality authority.

Implementation checks are restricted to packing values/dtypes/layout/order,
tail/padding exclusion,reuse/reset/weight-switch hygiene,real compile routing,
output lifetime,and trustworthy comparison/telemetry accounting. No new old-output
or sampled-action equality gate,custom numerical tolerance or test of PyTorch/
Tensor Core arithmetic was added. Existing tests/anomaly checks remain.
Covering network/frozen-kernel selection passes100tests and affected policy/reset/
timing selection57. Final measured-source commands pass:

| Command | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q tests/test_inference_experiment.py` |30PASS,2.38s; one sandbox NVML-init warning in Dynamo wiring |
| Scoped `.venv/bin/ruff check` over four model modules,benchmark tools and focused tests | PASS |
| `MYPYPATH=src .venv/bin/mypy` over four model modules and `benchmarks/inference` | PASS9source files |
| Actual CUDA compile/replay/cache/profile checks | PASS; native generation exercised outside the sandbox |
|18actual trials and18strict original/receipt audits | PASS;576natural games |

The NVML warning is confined to the sandbox test; host CUDA profiling and native
compilation execute successfully. Subsequent scoped fixes and review dispositions
are preserved with the experiment artifacts. Whole-suite/Rust/package/learning-quality
checks are NOT RUN because this plan explicitly limits verification scope and
changes no dependencies/native package/data qualification.

Preserved failures: initial input preparation compared full runtime resources
against the artifact's inference-only projection; using existing `inference_model`
resolves the boundary without relaxing identity. Independent review catches an
initial compiler-options bridge defect,modified default submission schedule,
partial compile-cost accounting and incomplete pair/telemetry guards before timing.
The initial physical reduce-overhead replay fails on captured-output overwrite;
the reviewed outside-compile clone/step-marker fix passes actual GPU replay.
All initial6abadcfscreen outputs remain preliminary,including an unpinned control
8.70→7.21ms drift; final paired data use17b2c1c,CPU2 and20warm loops.

Final evidence is local and reproducible within its recorded runtime limits:
`replay-plan-v2.json`,`one-factor-summary-v2.json`,`replay-followup-plan.json`,
`followup-summary.json`,`profile-plan.json`,`profile-summary.json`,
`cache-paired-summary.json`,`actual-plan.json`,`actual-summary-<model>.json`,
`actual-table-data.json`,`strict-audits.jsonl`,and all request/result/receipt files,
Chrome traces and outer invocation logs. Actual composition additionally requires
retained ignored local `runner.py`,`measure.py`,`analyze.py` under the prior
assessment directory; their bytes are in the new freeze. A fresh checkout alone
lacks those helpers and recorded originals. Preserve these artifacts with the
experiment. See [tool instructions](../../benchmarks/inference/README.md).

Final review dispositions: `bot_reviewer` independently reproduces saved timing,
identity and kernel/copy evidence and returns FINAL READY;
`bot_architecture_reviewer` returns FINAL READY for the implementation and recorded
scope. Neither substitutes a review for unexecuted tests or quality evaluation.
The independent verifier executes all18full-original audits separately after
timing. Reports,briefs,diffs and progress are preserved under the artifact root's
`review-artifacts/` directory. No blocking finding is waived.
