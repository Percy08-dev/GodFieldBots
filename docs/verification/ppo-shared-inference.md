# PPO resource measurements and shared inference status

## Original 3-game baseline

Measured on the original `dad129f673a556435d84583c5ee524c27d4578a6`
resident-worker-v1 path with the unmodified learner and collection code.
Simulator `91d5926a7572db327eaea3751e7c24f428120996`; Model4, producer-owned-v1,
canonical-python, actor9. The original saved conditions include source hashes;
the working tree had unrelated local edits. An isolated copy of the supplied
MLP D10000 config uses width192, four epochs, seed17, the mixed-refresh-16-v1
schedule, concurrency3/six worker slots, the original 24/12GiB managed budgets
and an intentionally limited two updates in fresh output directories. This is
the matched short comparison control, not a D10000 completion.

Data directory under the original checkout:
`runs/ppo-shared-inference-baseline/1ff6be02-09f5-46e6-84d1-c578a64fe587/`.
It contains the original run, updates, progress and timestamped samples, the
actual settings and source identity in identity.json, the probe and the GPU
termination record. The outer test harness passed an in-memory tuple in the
receipt to a JSON verifier after training, so its postprocessing exited 1.
A separate read-only verification loaded the *actual persisted* job receipt
(JSON array) and checked admission and context using verify_gpu_termination.
Ticket `36dd29ad3036463aaec2b3f721323145` confirms the learner and six worker
resources. The run was not repeated or counted twice.

All 32 games completed across two updates; 566 total actions and 294 learner
decisions. Collective wall time 28.322s (18.491s then 9.831s), 1.130 games/s,
19.98 actions/s. Learner update time 5.218s. These metrics include two
different game populations and cold startup in the first collection.
Do not subtract their difference as an estimate of only the startup cost.

| Last observed phase | Samples | Tree PSS peak GiB | Whole GPU peak GiB |
| --- | ---: | ---: | ---: |
| Startup | 8 | 0.939 | 3.009 |
| Collection | 56 | 9.710 | 5.367 |
| Source | 2 | 9.675 | 5.363 |
| Update | 12 | 9.679 | 5.363 |
| Sync | 4 | 9.685 | 5.367 |
| Cleanup | 5 | 9.685 | 5.367 |

Tree PSS samples sum smaps_rollup across observed descendants. Traversal is
not instantaneous; shorter-lived processes, allocations and peaks may be missed.
GPU samples include unrelated device users (2,788MiB at the first sample).
They do not measure exact workload-only VRAM. Save did not have a separately
sampled window; phase names come from the last coordinator callback. All six
actors reported lifetime allocator peaks of 60MiB reserved. Don't sum maxima
across owners or add allocated and reserved bytes. The original pre-change
managed estimates at this setting were 15.572GiB RAM and 11.822GiB CUDA.
The observations above cannot certify a universal maximum or the fit of
larger concurrency levels.

## Maximum resident cache occupancy component

Three independent CUDA processes each constructed the six distinct FP32 networks
that fit the respective width192 D10000 training preset cache and copied host
weights back into them through the existing restore_weights routine. Ordinary
eager model initialization and normal registered buffers were retained. This
isolates the weight/position payload at the configured identity count. There
are no active game inputs, policy episode caches, IPC or simultaneous learner
in this measurement. Artifact names are cache-capacity-{mlp,gru,transformer}.json
beside the original baseline data.

| Model | Weight bytes per identity | Position bytes per identity | Calculated six-network payload MiB | Peak allocator allocated / reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| MLP | 6,959,624 | 786,432 | 44.32 | 44.98 / 54 |
| GRU | 8,590,088 | 786,432 | 53.65 | 54.48 / 74 |
| Transformer | 9,187,592 | 786,432 | 57.07 | 57.75 / 68 |

The calculated payload is raw tensor storage; allocator granularity and
retention differ. This is a maximum-identity measurement, not a maximum
activation, restore, update or whole-machine stress experiment.

## Branch calibration of the resident cache/BEGIN change

On clean branch source `9fbae4975353ec2374c70ca40ed17379d1ec900e`, the normal
owned resident trainer ran the same two-update, 3-game/6-worker, width192
mixed16 conditions and returned expected `incomplete/update_cap`. Both game
updates completed; 32 games, 566 actions, 294 learner decisions. Collection
27.110s, updates 5.107s. Original admission and actual GPU termination receipt
verified with a live job and revalidated with six actors and the learner
(ticket `d2231e873e324d25858da3335b5bf05f`). Measurement command exited 0.
Raw inputs, results and sampler:
`runs/ppo-resident-cache-calibration/79013e2f-bd7a-49eb-8e97-3178c54a8ce0/`
under the original checkout. The post-change sampler additionally recorded
per-process roles and sampled RSS/PSS. There was one timing pair and no
hardware or kernel optimization in this milestone; no speedup is established.

| 3-game condition | Peak managed RAM GiB | Peak managed CUDA GiB | Highest sampled tree PSS GiB | Highest sampled whole-device usage GiB |
| --- | ---: | ---: | ---: | ---: |
| Original | 15.572 | 11.822 | 9.710 | 5.367 |
| New cache/BEGIN recipe | 15.043 | 11.293 | 9.711 | 5.367 |

Managed peak decreased by 568,206,096 bytes in both reported columns. These
are planning numbers; the physical GPU peak sample remained the same. The
new run started with 2,786MiB of whole-device use. Per-process sampled peaks
(PSS / RSS, MiB): coordinator 136.4 / 140.7; learner 2296.7 / 2336.9;
largest of six resident actors 1339.8 / 1694.0; largest test server 12.1 /
14.4; largest group writer 19.1 / 28.9. These are individual process maxima,
not a simultaneous sum. Under the unchanged base guard a width192 MLP still
fails four games first on CUDA (13,789,146,768 bytes > 12GiB), and eight games
first on RAM (27,833,409,296 bytes > 24GiB). The old base estimator has not
established that either configuration physically fails.

## Learner component evidence and limits

A separate bounded CUDA probe used the exact width192 model/training recipe
for each model, with two small three-decision synthetic updates and the strict
Model4 file export/cold restore. Each model ran in a fresh process. The output
records the after-update parameter, buffer, gradient, and Adam tensor payloads
by device, CUDA allocator peaks per phase, and CPU end-phase PSS and lifetime
RSS. See learner-phases-{mlp,gru,transformer}.json next to the baseline data.
The initial probe needed the actual preset save allowance (1GiB) rather than
the helper's default; the reported runs use that preset allowance.

| Model | Measured post-update CUDA tensor payload, bytes | CPU Adam step tensors, bytes | Update peak allocated/reserved MiB | Restore peak allocated/reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| MLP | 28,624,928 | 240 | 99.0 / 116 | 92.7 / 116 |
| GRU | 35,146,784 | 264 | 105.8 / 120 | 98.4 / 120 |
| Transformer | 37,536,800 | 360 | 109.2 / 124 | 101.2 / 124 |

These post-update tensor sizes agree with four FP32 weight-sized arrays
(parameters, gradients and two Adam moments) plus the position buffer on CUDA.
The CPU step tensors and the allocator reservation are different costs.
This small-input probe does not bound backward/replay workspace for a maximum
candidate/event/history workload or the owned bank export lifetime.

## Verification disposition

Independent implementation and architecture reviews accepted the scoped cache
and diagnostic code; final documentation and these post-change measurements
are separate evidence. Targeted budget/cleanup/campaign checks, model/lease
concurrency, failure/reap/FD checks and scoped Ruff/mypy passed, including
real AF_UNIX and CUDA paths. No full-suite, shared-mode scaling, ordinary PPO
shared-mode run, final evaluation or model-quality claim. See the simulator
handoff and the estimator follow-up in the design for the remaining boundaries.

## Single-participant inference at the preset geometry limits

Each width192 model ran in its own CUDA process with six distinct, actual
GameStructuredPolicyV4 cache entries. The last active policy traversed a
1,024-decision synthetic history and at position 1,023 consumed 4,096 interval
events and 32,768 candidates (chunk 256). Existing step, factorized distribution,
value, normalization, finite check and CPU multinomial were used. The probe
released/reacquired the resident entry and repeated the maximum-size endpoint
at position zero. Fixture event/candidate tokens repeat; this exercises device
shapes, not a gameplay legality or quality claim.

| Model | Six cached networks + active named tensors, estimated MiB (excluding context/workspace) | Maximum allocated/reserved MiB at final endpoint |
| --- | ---: | ---: |
| MLP | 60.73 | 92.64 / 110 |
| GRU | 70.06 | 102.14 / 108 |
| Transformer | 76.47 | 109.25 / 122 |

The named payload/transient calculation alone is below measured allocated
bytes. The live recipe also reserves the existing 256MiB native workspace
and 512MiB context per active-capable physical owner, plus separate import
staging. The comparison supports the measured configuration, not an exact
attribution of every residual byte or universal guarantee. Do not subtract
changes in whole-device use by unrelated desktop clients. Full raw outputs
and the script (`actor-shape-*.json`, `actor_shape.py`) are next to the
9fbae49 calibration. Earlier stored runs keep the recipe they actually used.
