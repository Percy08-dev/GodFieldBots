# Ordinary Rust Model3 PPO update verification

Implementation and experiment evidence is in progress. The
[reviewed design](../design/ordinary-rust-ppo-update.md) defines the scope.

## Environment

The implementation branch starts at `4912a9f` from `feat/rust-player-runtime`.
The pinned Simulator source is `072dabbca1458c9526897781d2fd64edeeb4f9ca`.
Simulator binary SHA-256
`76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d`.
The contract is Simulator section 21.1, ActionConfig v3, actor protocol 11,
participant protocol 3, Journal and Derived schema 6, candidate codec v2, and
ordinary rollout 4. The new update settings have schema 1. Inference and resume
arrays use the existing Model3 format.

The host exposes 16 logical CPUs, Python 3.14 with the GIL, NumPy 2.5.4,
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
