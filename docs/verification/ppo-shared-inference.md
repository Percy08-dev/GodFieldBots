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

## Four-game resident run under original managed budgets

On clean branch `9e6cb5da965f1c461b78dca28d02f7a61d4c862d`,
the new actor CUDA recipe admitted four game slots/eight physical participants
at the original 24GiB RAM / 12GiB CUDA limits. Old admission refused the same
configuration. The new planning maxima were 17.637GiB / 10.237GiB, versus the
original 18.342GiB / 14.842GiB. No budget override was supplied to this run.

Run `b4d9d329-53f9-445f-beb5-74dc28324dc3` in
`runs/ppo-actor-cuda-calibration/` under the original checkout completed two
updates with 32 games, 566 actions and 294 learner decisions. It returned
the requested `incomplete/update_cap`. Eight live resident processes were
observed; ticket `397ae6e0a1dd4a9d95f62b6f8c2b8874` confirms retirement of
eight workers and one learner. The normal train/collect call, original receipt
check and outer probe all returned successfully. The original sampler recorded
77 samples, 12.140GiB peak observed tree PSS and 5,495MiB peak whole-device
usage (first sample 2,059MiB; desktop use varies). Collection took 26.898s,
or 1.190 games/s; updates took 4.970s. Compare 28.322s in the original paired
three-game configuration and 27.110s in the first cache-only three-game
run. Concurrency and admission implementation differ; there is one short
population per condition, not a general scaling or shared-service speedup.

At 8/16 simultaneous games, unchanged host planning and the topology with
per-actor CUDA contexts still exceed configured budgets. Those cases are
not live tests. The shared inference service requires a separate Simulator
contract decision before its independent 1/2/4/8/16 population can run.

To distinguish active games from processes, the verifier separately reopened
the first collection of each run (SQLite read-only, eight group packs, only
HTTP-log member scalars). An interval for a game runs from its first
game-specific request to its last matching response completion. Sweeping these
intervals across 16 naturally completed games gave maximal simultaneous
game windows of 3 (original), 3 (corrected cache), and 4 (actor CUDA recipe).
This establishes overlapping live games, not simultaneous GPU forward passes.
For four games the observed individual process maxima were 1,317.8MiB PSS for
a participant and 2,293.8MiB PSS for the learner. Summed tree RSS peaked at
16,004.9MiB; PSS apportions shared pages and is the better aggregate here.

## Separate learner launch allowance and cold resume

Clean source `73d389644975ee2473bb2f8dbe644b251e450ec4`, same width192/4-worker-pairs
conditions as the four-game check above, with a deliberately smaller **9GiB**
aggregate CUDA management budget. The checked learner's logical allowance L
was 2,966,109,712 bytes, and its actual device lease D was 1,638,077,776 bytes.
The remaining budget was held for the future resident pool and overlapping
imports. The final original run identity records the same recipe and quantities
as the prelaunch calculation.

Two fresh GPU jobs executed one parent: pause after the first 16 games, then
cold resume through the second 16 games and the prescribed update cap. Fresh
reported paused/one update; resumed reported incomplete/update_cap/two updates,
294 used decisions. Tickets `1b49b7ed526c4c13afca3dd344182361` and
`7d0e26332c4b4fe4b4b19c299ba78c3b` each have nine confirmed resource
terminations in the original admission namespace. Both live checks passed
and the measurement command returned 0. Total collection 30.760s, update
6.293s. This lifecycle diagnostic includes a real restart and a different
budget; keep it out of ordinary throughput comparisons.

Raw evidence under the original checkout:
`runs/ppo-owned-learner-calibration/9c1d2c6c-9217-4bb5-a7de-541e30bf7345/`.
Ninety-eight host/device samples observed maximum tree PSS 12.24GiB and
whole-device use 5,479MiB. The largest sampled sum of unique gf-owned-bank
inode lengths was 52,052,440 bytes; summing individual FD lengths instead
reached 111,391,552 bytes. These are file sizes observed at discrete times,
not resident pages. Peak coordinator Source reservations were 676,216,055
and 582,330,699 bytes in the two invocations. At each collector return the
nested parent phase was already included (135,631,752 of 470,733,497 bytes;
144,960,349 of 363,723,521 bytes). Separate server scan retained 569 bytes.
The collector separately reported peak RAM counters of 3,473,096 and
3,083,015 bytes, explicitly scoped to managed/observed RAM rather than RSS.

## Accepted independent scope

The branch improves existing HPO diagnostics, cache/BEGIN/SYNC accounting,
resident actor CUDA sizing and normal owned learner device budgeting. Its
per-collection host report identifies accounting scope and sampled charges.
It retains a conservative compatibility RAM gate, because there is no
calibrated full-process upper bound for an actor's retained Python event
history, search and IPC across supported inputs. See the [design](../design/ppo-shared-inference.md)
for the difference between concurrent independent accounts and subcharges.
This work does not close every requirement in GodFieldBots issue 2.
Four concurrent games were physically verified. The separate shared service,
ordinary PPO integration and shared 1/2/4/8/16 comparison await the upstream
[Simulator issue 72](https://github.com/Percy08-dev/GodFieldSimurator/issues/72).
