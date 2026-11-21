# Ordinary Rust Model3 PPO update verification

Implementation and the approved 256-game experiments passed. The
[reviewed design](../design/ordinary-rust-ppo-update.md) defines the scope.

## Environment

The implementation branch starts at `4912a9f` from `feat/rust-player-runtime`.
The pinned Simulator source is `072dabbca1458c9526897781d2fd64edeeb4f9ca`.
Simulator binary SHA-256
`76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d`.
The contract is Simulator section 21.1, ActionConfig v3, actor protocol 11,
participant protocol 3, Journal and Derived schema 6, candidate codec v2, and
ordinary rollout 4. The saved formats are Run 5, generation 4, update 6,
Source proof 4, inference 4 and resume 4. The new update settings have schema 1.

The host exposes 16 logical CPUs, Python 3.14.10 with the GIL, NumPy 2.5.4,
Torch 2.14.1+cu132, and an RTX 5070 Ti with 16,303 MiB of VRAM (driver 610.88).
CUDA and local Simulator networking require access outside the sandbox.

## Measurement definitions

Games per second divides naturally ended games by the actual collection
stopwatch. The data-generation phase also includes Source verification and
numeric preparation. Learner decisions per second uses only decisions that
entered PPO and names its denominator. Update time covers the persisted
application update phase. Whole wall runs from the observer's command start
to original process exit, including cold completion. RAM is sampled peak
process-tree PSS from complete sweeps. VRAM is sampled device-wide usage,
including any unrelated allocations.

Concurrent active games are derived from the accepted second player's join
through the successful terminal-action acknowledgement in the original
Simulator HTTP logs. The measurement uses request timestamps plus response
durations with a millisecond uncertainty interval. The detector recovers the
recorded 16 games and a two-game peak in the retained C2 reference.

## Fixed-data CUDA comparison

The first matched-width screen used seed 17 and the same 230 learner
decisions from 16 retained natural games. The normal rollout4 reader
validated the original digest
`6320dcc9a6d6eb0094ccceca4043d9ddee9edd59f8bdd16cad073cb30a07f626`.
Candidate counts ranged from 1 to 2,809. The 64-decision logical batches
ended with a 38-decision tail each epoch. The 128-decision batches ended
with a 102-decision tail.

All observed Adam batches matched the independently generated PCG64 logical
membership. Each decision entered its batch exactly once per epoch. Group
losses were compared as means weighted by their real row counts, and padded
rows were excluded from the candidate distribution. The largest absolute
differences in valid candidate logprob, value and initial logical-batch
loss across the comparisons were 4.77e-7, 9.54e-7 and 6.08e-8. All
comparisons satisfied `atol=rtol=1e-5`. CPU, CUDA and shuffle RNG state
matched within each logical-width comparison.

| Logical width | CUDA cap | Peak executed group width | Adam steps | Update wall (s) | First clipped gradient max Δ | Final clipped gradient max Δ | Adam state max Δ | Final weight max Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 1 | 8 | 15.096 | — | — | — | — |
| 128 | 16 | 16 | 8 | 1.215 | 4.90e-6 | 3.834e-3 | 3.633e-4 | 1.734e-3 |
| 128 | 32 | 32 | 8 | 0.879 | 4.90e-6 | 3.835e-3 | 3.633e-4 | 1.734e-3 |
| 128 | 64 | 52 | 8 | 0.918 | 4.90e-6 | 3.835e-3 | 3.633e-4 | 1.734e-3 |
| 128 | 128 | 52 | 8 | 0.855 | 4.90e-6 | 3.835e-3 | 3.633e-4 | 1.734e-3 |
| 64 | 1 | 1 | 16 | 13.908 | — | — | — | — |
| 64 | 16 | 16 | 16 | 1.481 | 1.49e-8 | 1.253e-6 | 1.234e-7 | 3.928e-5 |

Differences are against width one in the matching logical-width group. Each
row is one instrumented update. Timings include logical membership tracing
and capture of the first clipped gradient. The 64 and 128 caps had the same
actual groups in the 128-logical setting for this rollout. The separate
planner test forms a 128-row group when shape and capacity allow it.

A 64/16 fixed-data continuous second update and a load from the first
update's checkpoint reached 32 Adam steps. Weights, optimizer arrays,
gradients, metrics and all three RNG states matched exactly. These are
numerical and performance observations. Learning quality remains for HPO.

The primary checkout retains the commands, script and JSON under
`runs/ordinary-rust-ppo-update/fixed-rollout-r1`. The summary SHA-256 is
`d0bcd25a8cf0bd84bbbddd137b88594e92e762ac9e7d1cc4c670140d707998b2`.

## Initial live setup and Source allowance

Two C16/W16 CUDA attempts used the 8 GiB default Source workspace. The
first attempt sealed 256 natural games across 128 pairs. Source validation
stopped before numeric publication, and the run committed zero updates.
A diagnostic rerun attributed the failure to the existing managed Journal
decode reservation at 8,568,513,620 bytes in use, requesting 22,266,720
bytes against 8,589,934,592 bytes. The failed attempt is excluded from the
throughput matrix. There was no measured GPU OOM or deadline failure.
Peak sampled process-tree PSS was about 2.14 GiB on the first attempt.

A read-only diagnostic of the first sealed collection completed all Raw/
Journal/Replay checks with a 16 GiB Source workspace. It found 2,541 learner
and 4,928 all-seat decisions. Retained managed Source charges reached
12,242,435,628 bytes after raw verification, a reservation estimate
separate from observed process memory. The revised common experiment
configuration uses a 16 GiB Source workspace, an 8 GiB storage budget, a
24 GiB managed host budget and a 12 GiB managed CUDA budget.

With the 16 GiB Source budget, revision `696ea4e` saved the first real
C16 cohort and its 160 Adam steps. Cold completion passed the full
original, Replay and numeric readback, then passed the unspent Source
grant of 16,999,555,680 bytes to a checkpoint reader whose accepted ceiling
is 12 GiB. A read-only replay of the cold code reproduced the constructor
failure on the exact records. With the scanner grant capped at 12 GiB,
the same diagnostic read the saved pair and its 160 steps, with an
estimated scan peak of 324,827,313 bytes. The production repair applies the
scanner ceiling at the caller, keeping the Source allowance for collection
auditing. The original failed supervised attempts remain in the evidence.

## Supervised save and resume

Clean revision `6851ddf` completed both C16/W16 invocations under the
original run ID, effective config hash and deadlines. Each invocation was
independently verified after its original exit.

| Invocation | Natural games | Learner decisions | New Adam steps | Cumulative Adam steps | Peak active | Collection (s) | Data generation (s) | Update/save (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh, pause at update 1 | 256 | 2,541 | 160 | 160 | 16 | 63.21 | 118.38 | 15.86 | 226.92 |
| Resume, complete update 2 | 256 | 2,817 | 180 | 340 | 16 | 68.82 | 128.33 | 17.44 | 422.42 |

The restore receipt confirms the named weights and Adam state, CPU and CUDA
Torch RNG and the PCG64 state at step 160. The resume invocation's verifier
emits only that invocation's collection rows. The preliminary aggregation
indexed them by parent generation and reported an index error after the
verifier had succeeded. The corrected summary selects the verified
`batch-000001` row by ID. It is retained alongside both original readbacks
and the restore receipt.

## Concurrency comparison

All trials use ordinary Rust Model3, CUDA inference and learning, seed 17,
the pinned Simulator, the same initial policy, and the following settings.
The inference dequeue cap is 64, the formation wait is 1 ms, and the
scheduler is Tokio. Worker count is the smaller of configured concurrency
and the available 16 logical CPUs.

```toml
[training.ppo_update]
schema_version = 1
collection_games = 256
minibatch_size = 64
epochs = 4
shuffle_strategy = "all"
gpu_microbatch_size = 16

[preprocessing]
memory_budget_mib = 16384
```

The 15 trials ran sequentially at clean revision `6851ddf`. Each run
has its own original directories, observer output, GPU samples and
independent readback. The order is C4, C8, C16, C32, C64, C32, C64, C4,
C8, C16, C8, C16, C32, C64, C4. Managed time budgets were 2,400 s
admission, 2,700 s execution, 3,600 s overall, and a 600 s post-host
allowance. Update/save time covers Source input recheck, optimization
and checkpoint export. The whole wall includes cold readback and excludes
the separate external verifier.

The following time, rate and sampled memory values are medians over the
three trials per condition. Peak active is the maximum observed concurrent
game count over the trials.

| Concurrent games | Rust workers | Verified | Peak active | Games/s | Learner decisions/s whole wall | Data generation (s) | PPO update/save (s) | Total wall (s) | Peak RAM PSS (GiB) | Peak device VRAM (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 3/3 | 4 | 3.97 | 11.17 | 119.05 | 15.87 | 227.49 | 2.60 | 3,528 |
| 8 | 8 | 3/3 | 8 | 4.05 | 11.23 | 117.81 | 15.99 | 226.31 | 2.67 | 3,483 |
| 16 | 16 | 3/3 | 16 | 4.06 | 11.22 | 117.98 | 15.96 | 226.45 | 2.78 | 3,500 |
| 32 | 16 | 3/3 | 32 | 4.03 | 11.18 | 118.21 | 15.94 | 227.31 | 2.87 | 3,498 |
| 64 | 16 | 3/3 | 64 | 4.01 | 11.14 | 118.61 | 15.84 | 228.00 | 2.94 | 3,520 |

| Trial | Observed overlap | Collection (s) | Data generation (s) | Update/save (s) | Total wall (s) | Games/s | Learner decisions/s whole wall | Peak RAM PSS (GiB) | Device VRAM start / peak (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C04-1 | 4 | 64.44 | 119.05 | 15.87 | 227.42 | 3.97 | 11.17 | 2.60 | 3,015 / 3,561 |
| C04-2 | 4 | 64.52 | 119.38 | 15.86 | 228.04 | 3.97 | 11.14 | 2.60 | 2,939 / 3,481 |
| C04-3 | 4 | 64.80 | 118.92 | 15.88 | 227.49 | 3.95 | 11.17 | 2.60 | 2,977 / 3,528 |
| C08-1 | 8 | 62.77 | 117.37 | 15.80 | 225.41 | 4.08 | 11.27 | 2.65 | 3,015 / 3,558 |
| C08-2 | 8 | 63.26 | 117.81 | 16.04 | 226.78 | 4.05 | 11.20 | 2.70 | 2,938 / 3,483 |
| C08-3 | 8 | 63.34 | 117.92 | 15.99 | 226.31 | 4.04 | 11.23 | 2.67 | 2,940 / 3,483 |
| C16-1 | 16 | 62.99 | 117.83 | 15.96 | 226.48 | 4.06 | 11.22 | 2.78 | 3,015 / 3,558 |
| C16-2 | 16 | 63.12 | 117.98 | 15.99 | 226.45 | 4.06 | 11.22 | 2.79 | 2,940 / 3,483 |
| C16-3 | 16 | 63.15 | 118.04 | 15.83 | 226.40 | 4.05 | 11.22 | 2.76 | 2,957 / 3,500 |
| C32-1 | 32 | 63.61 | 118.21 | 15.86 | 226.93 | 4.02 | 11.20 | 2.88 | 3,015 / 3,558 |
| C32-2 | 32 | 63.57 | 118.37 | 15.94 | 227.37 | 4.03 | 11.18 | 2.87 | 2,978 / 3,455 |
| C32-3 | 32 | 63.51 | 117.98 | 15.99 | 227.31 | 4.03 | 11.18 | 2.86 | 2,964 / 3,498 |
| C64-1 | 64 | 63.57 | 118.31 | 15.80 | 227.25 | 4.03 | 11.18 | 2.94 | 3,015 / 3,522 |
| C64-2 | 64 | 64.53 | 120.56 | 15.84 | 229.22 | 3.97 | 11.09 | 2.94 | 2,939 / 3,482 |
| C64-3 | 64 | 63.85 | 118.61 | 16.54 | 228.00 | 4.01 | 11.14 | 2.97 | 2,955 / 3,520 |

All 15 trials completed with 256 natural games, 2,541 learner decisions,
4,928 all-seat decisions and 160 Adam steps per run. All readbacks verified
the original checkpoint, including the Adam state. The initial weights'
published array digests matched across all trials. The observed overlap
bounds agree with the configured concurrency in every trial.

C4 to C64 median collection throughput stays within 2.3% of the lowest
condition. Whole-wall medians span 226.31–228.00 s while median sampled PSS
increases from 2.60 GiB to 2.94 GiB. GPU device usage on startup ranged
from 2,938 to 3,015 MiB with background allocations included. All 15
measured trials completed under the common capacity and deadline
conditions. The startup capacity and checkpoint-scanner failures from
the earlier setup are recorded above.

The primary checkout retains the experiment, original records, all
per-case timing and memory samples, and the paired resume proof under
`runs/ordinary-rust-ppo-update/approved-matrix-6851ddf`. The matrix
`summary.json` SHA-256 is
`4b85c2f038d9c04f0eaacbdf6b5763044b067213ce942b8340bf7e5ea4663f93`.
The independently reconciled summary SHA-256 is
`c441aaae91405d8405ce1d600d39211915a68d57f7f1990403333fca01046f5d`.

## Review and regression checks

Independent architecture review accepted the design before implementation.
Separate correctness and design-conformance reviews accepted revision
`696ea4e` and the scoped scan correction in `6851ddf`. The deployed formal
preflight still verifies all eight frozen kernel fingerprints.

The final focused CPU selection passed 617 tests. It covered ordinary
configuration, CLI overrides, schedule pairing, all four shuffles, planner
capacity, Source/lineage, fixed-data save/resume, the external verifier
and Model4 regression paths. Ten CUDA-selected nodes were left to their
separate verification scope. The selection also excluded
`test_resolved_route_fact_is_successful_actual_branch_and_clears_on_failure`
after reproducing it unchanged at `4912a9f`. Its test mock rejects the
existing `_device_bytes` keyword. Ruff and mypy over the 14 changed
production source files passed, as did `git diff --check 4912a9f`.

The CUDA width and supervised Simulator evidence in this record establishes
the configured update behavior and measured workload performance. Learning
quality is reserved for the later HPO comparison.
