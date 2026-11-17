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
