# Rust player HPO6 collection measurements

## Workload

The CUDA diagnostics used an RTX 5070 Ti, CPython 3.14.10 with its GIL, and
PyTorch 2.14.1+cu132. Bots started from `7831755cc92c9279a60f150a4189035f3dfa18e1`
with the archived inference survey instrumentation. Simulator
`6d50603dab3f6d2df34ea4e433bab7dca96f52c1` used the pinned TestServer binary
`500d55cd8eb03126f65d3c24fe58b6091bf8acdb2fc20aed76622a8679f4000d`.
The action contract uses Sacrifice, Gameplay API v2, public event and catalog
schema 2, Journal 6, candidate codec 2, and HPO6. The managed limits were
24 GiB RAM and 12 GiB CUDA. Seed 17, 256 simultaneous natural games, 16 Tokio
workers, width 32, and a 1 ms formation window were fixed. Each diagnostic
produced 5,081 actions and the same semantic Journal and initial weight hashes.

The primary rate for the Journal changes divides accepted actions across
both seats by `collection_seconds`. This interval starts before batch setup,
including checkpoint publication, inference preparation and writer startup,
and ends after the durable collection seal. For a run stopped after sealing,
use the standalone progress record's `collect` duration. [Timer](../../src/gf_training/ppo_run.py#L777) · [Batch setup](../../src/gf_training/rust_player_collection.py#L310)

Launch-through-seal timings also include the command's startup and initial
checkpoint creation. They are a separate operational observation. Add the
collection duration to the command-start-to-collection-start interval. The
wide diagnostic `collection` phase runs through downstream Source work and
has a later end event. GIL and queue diagnostics carry their own probe overhead.

## Early GIL contention

The probe matched the pinned CPython acquire and release callsites for the
inference worker's GIL timed waits. It captured all of the first ten seconds
after first enqueue. Of 348.139 FIFO-weighted request-seconds, 308.738
overlapped the inference thread's GIL condition waits. The writer held the
GIL for 5.428 of the 6.160 elapsed wait seconds, accounting for 282.015
request-seconds, or 91.34% of the condition-wait burden.

The writer held 235.771 request-seconds in the pending envelope before the
native frame call. The 256 long envelopes accounted for 235.557 of those
request-seconds. Each long envelope corresponds to a first-game Journal
open by timing and count. The existing detached frame and candidate calls
accounted for 0.244 and 0.041 request-seconds respectively. CPU replay of
two-neural-seat headers measured 15.144 ms median writer CPU per game.
The constructor and header checker each spent about 5.7 ms validating the
same neural metadata. The header checker spent a further 1.8 ms verifying
effective worker settings. The raw trace uses monotonic elapsed intervals,
which include OS scheduling during GIL ownership. [Holder results](../../runs/journal-native-frame/approved-gil-holder-256/update-1/gil-holder-results.json)

## Request supply

On the 256-game supply diagnostic, the FIFO was empty for 3.85 of the first
10 seconds and 8.52 of the following 12.04 seconds through final dequeue.
The table sums actor wall time within FIFO-empty intervals. Concurrent actors
make these totals larger than elapsed wall. Completion counts use the same
interval and reflect the FIFO state at each stage endpoint.

| Stage | Early actor-s | Early completions | Later actor-s | Later completions |
| --- | ---: | ---: | ---: | ---: |
| Pending writer receipt | 522.81 | 997 | 459.41 | 2,195 |
| Act | 1.58 | 1,031 | 3.09 | 2,256 |
| Acknowledgement writer receipt | 409.98 | 667 | 479.93 | 2,202 |
| Observation | 1.33 | 606 | 3.22 | 2,004 |
| Events, hints and frame | 0.91 | 519 | 2.30 | 1,863 |
| Candidate search | 0.53 | 461 | 2.59 | 1,756 |
| Numeric operands | 0.25 | 279 | 0.96 | 1,151 |

The pending path includes command construction followed by the original
writer acknowledgement. The acknowledgement path includes the original
writer receipt after the Simulator returns the Act response. The scope keeps
the original pending-before-Act and revision checks. [Stage results](../../runs/journal-native-frame/approved-supply-stages-256/update-1/supply-stages.json)

## Journal changes

The implementation branch carries the Rust frame verification from ADR0073.
Its three earlier pairs improved the median collection interval from 180.56
to 197.76 actions/s. The gain of 17.20 actions/s exceeded the control range
of 8.21 actions/s and met the adoption threshold. Launch through seal gives
149.71 to 162.58 actions/s. [Frame verification](journal-native-frame.md)

A separate candidate removed the duplicate neural header validation. The
CPU replay dropped from 15.112 to 9.639 ms median per two-model game. All
headers remained byte identical. Six uninstrumented 256-game runs used the
same Stage 1 release wheel, alternating order, fixed weights and the same
5,081-action Journal in each pair.

| Metric | Control | Candidate |
| --- | ---: | ---: |
| Collection interval actions/s, median | 187.96 | 192.87 |
| Launch-through-seal actions/s, median | 147.09 | 146.57 |
| Inference round-trip p95, range of bins | 2,819–3,087 ms | 2,147–2,617 ms |
| Torch reserved high-water, maximum | 474 MiB | 388 MiB |

Inference p95 improved in all three pairs. The median collection gain of
4.91 actions/s fell within the control range of 18.55 actions/s. The header
candidate was removed after the trial. [Matched pairs](../../runs/rust-sequence-hpo/mlp-header-comparison.json)

The no-change control from the frame series and the both-changes candidate
from the header series give 180.56 and 192.87 collection actions/s. These
are medians from separate series. In the runs representing the
launch-through-seal medians, time before collection start was 5.80 s for
the no-change control and 7.00 s for the both-changes candidate. This
startup difference preceded writer construction. Both Journal changes
operate after collection starts.

## Reference capacity

The existing Model4 Python resident path used HPO4, `canonical-python`, the
same Simulator, seed, model widths and optimizer settings. Under the same
12 GiB managed CUDA allowance, its 16-concurrent GRU and Transformer
preflights estimated 32,415,492,800 and 32,555,133,632 bytes for pool and
staging. All 16, 64, 128, and 256 concurrent configurations stopped at that
admission check. These values describe configured estimates. The same 24 GiB
RAM limit allowed the two-game configuration and rejected four concurrent
games. The 16-game functional comparison can use concurrency 2 under the
unchanged managed limits.

One fresh 16-game pair per model at concurrency 2 passed on both routes.
Each pair used the same initial CUDA checkpoint weights and 376 actions.
All decision keys, candidate column hashes, selected indices and actions
agreed. The FP32 comparison of saved selected log probabilities and critic
values was within `atol=rtol=1e-5` in every row.

| Model | Python launch-through-seal actions/s | Rust launch-through-seal actions/s | Largest absolute FP32 difference |
| --- | ---: | ---: | ---: |
| GRU | 10.75 | 31.15 | 2.68e-7 |
| Transformer | 15.22 | 30.89 | 2.62e-6 |

These are single matched observations at the shared admitted concurrency.
The resident reference cannot supply a matched 256-game control under the
fixed limits. [Reference comparison](../../runs/rust-sequence-hpo/reference-comparison.json)

## GRU CUDA collection and updates

The first integration configuration inherited the MLP native Source's
1.5 GiB numeric setting and hit the canonical Python encoder's 512 MiB
ceiling after sealing 16 games. That parent remains failed with zero updates.
The explicit sequence configuration now checks the existing encoder limit
before collection. The comparison uses a 512 MiB Numeric6 budget for both
the Rust and resident Python sequence paths.

| Run | Configured concurrency | Natural games | Actions | Launch-through-seal actions/s | Collection interval actions/s | Torch reserved peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh, paused at update 1 | 16 | 16 | 376 | 34.64 | 79.87 | 164 MiB |
| Resume, paused at update 2 | 16 | 16 | 228 | 14.31 | 59.76 | 172 MiB |
| Fresh, paused at update 1 | 64 | 64 | 1,319 | 77.92 | 127.02 | 246 MiB |

All three runs passed the supervised operation verifier with full Source
audit. Their initial weight SHA-256 was
`f3dd9bc217b07a6356f92b89711d78f916b35c5f87a867697c14389d23e98c58`.
The 128-game run sealed all 128 natural games and 2,633 actions. The
canonical Numeric6 token temporary estimate exceeded 512 MiB, so the parent
committed zero updates and the initial update scale series stopped. This is an
encoder capacity decision after collection, with a verified update path at
64. [GRU artifacts](../../runs/rust-sequence-hpo/)

## Transformer CUDA collection and updates

The source-matched final release wheel passed a 16-game fresh update and
pause and resumed the same parent for a second audited update. The frozen
initial weight SHA-256 was
`0f2d090bcd85af053a0b3ac398e4f6584b4bfbc100e2e26503815066fc8d9120`.

| Run | Configured concurrency | Natural games | Actions | Launch-through-seal actions/s | Collection interval actions/s | Torch reserved peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh, paused at update 1 | 16 | 16 | 376 | 20.81 | 52.97 | 222 MiB |
| Resume, paused at update 2 | 16 | 16 | 235 | 14.84 | 56.08 | 238 MiB |
| Fresh, paused at update 1 | 64 | 64 | 1,321 | 71.85 | 106.44 | 630 MiB |

The 128-game Transformer case, like GRU, sealed 2,633 actions in 128 natural
games and committed zero updates. In both cases the retained learner
candidate tables and per-decision terms alone account for 701,544,960 bytes
of the existing token temporary estimate, above its 536,870,912-byte limit.
The initial update scale series stopped and kept the failed parents quarantined.

The configured sequence groups use the normal Torch FP32 math backend.
The Transformer budget includes the small query temporaries, staged KV
slot, gathered prefix, and the extra per-layer key copy from Math SDPA.
CUDA scalar, grouped and resumed tests cover suffix boundaries, rejected
rows, same-width reorder and neighbor independence, checkpoint routing,
and actual cache release. [Transformer artifacts](../../runs/rust-sequence-hpo/rust-transformer-16/)

## Follow-up at 256 simultaneous games

On `86071f0`, both sequence models used the same release wheel, Simulator,
seed 17, and resource limits as their earlier checks. Each model used its
same initial weights. One trial per model sealed all 256 natural games and
5,088 actions. Retained Simulator Join and terminal Act results establish
an actual overlap lower bound of 256 in both runs. The sealed report and
manifest control hashes, group Pack hashes, candidate sidecars, and total
action counts were checked from the original SQLite file.

| Model | Launch through seal | Launch-through-seal actions/s | Collection interval actions/s | Inference round-trip p95 bin | Torch reserved peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU | 49.24 s | 103.33 | 145.30 | 3,221–3,288 ms | 246 MiB |
| Transformer | 45.88 s | 110.89 | 126.26 | 3,691–3,758 ms | 1,404 MiB |

Both runs stopped in Source with `ValueError` and committed zero updates.
Reapplying the existing retained-original accounting formula to the sealed
files gives lower bounds of 13,118,031,048 bytes for GRU and 13,142,653,256
bytes for Transformer against the 12,884,901,888-byte Source limit. The
separate retained learner data gives a Numeric6 planning lower bound of
1,299,656,192 bytes per model against its 536,870,912-byte limit. These are
admission estimates. A read-only GRU replay of the Source inspection under
the same 12 GiB limit reproduced the refusal at 12,879,662,552 bytes charged
with another 8,222,912 bytes requested. The failure records and quarantine
are retained. [GRU Source recheck](../../runs/rust-sequence-hpo/gru-256-source-recheck.json)

The resident Python configurations still stop at their 12 GiB CUDA admission
check at this concurrency. The 256-game rates are standalone collection
observations. They have no same-concurrency Python control or update result.
[Timed collection](../../runs/rust-sequence-hpo/collection-256-results.json) · [Source accounting](../../runs/rust-sequence-hpo/collection-256-source-bounds.json)

## MLP formation and CUDA screens

The final native build completed one 256-overlapping-game trial per
formation setting. All three produced the same weights, 5,081 actions and
identical semantic Journal fingerprints. The 0.5 ms and 0 ms settings did
not demonstrate an advantage beyond the observed control variation in
these single trials. The existing 1 ms campaign setting remains selected.

| Formation wait | Launch-through-seal actions/s | Collection interval actions/s | Inference p95 bin | Torch reserved peak |
| --- | ---: | ---: | ---: | ---: |
| 1.0 ms | 129.67 | 187.45 | 2,751–2,819 ms | 416 MiB |
| 0.5 ms | 148.37 | 190.01 | 2,819–2,886 ms | 442 MiB |
| 0 ms | 146.72 | 186.88 | 2,215–2,282 ms | 416 MiB |

[Formation results](../../runs/rust-sequence-hpo/formation-comparison.json)

Two disposable CUDA screens used fixed padded `(32, 2048, 16)` Model4 MLP
operands and included transfer and blocking CPU return. Over 40 alternating
repeats, full-forward Graph reduced median row-group time from 8.97 to
5.16 ms. Pinned, nonblocking input transfers reduced Graph median to 4.76 ms
with the same final CPU result fence. Outputs were bitwise equal across all
routes. Capture took 48.0 ms and reserved 166.4 MiB for its private pool and
inputs. Pinning took 11.1 ms, and the probe's total Torch reserved peak was
568 MiB. These operator timings leave full-collection throughput and capacity
untested for the experimental Graph and transfer routes. Adoption remains
with the existing MLP recipes. [CUDA screen](../../runs/rust-sequence-hpo/mlp-large-shape.json)

## Implementation checks

- Independent architecture and correctness reviews cleared the Rust custody
  and Python cache path after the sequence cancellation and Math SDPA memory
  fixes. The final release wheel's SHA-256 is
  `5778bd3486dc182cc9e2d7c69a88d752dfce23d837a43bff04768d3d9425e457`.
- The final physical CUDA and Simulator regression command passed 23
  selected tests. It covered native pending receipts, sequence parity and
  GPU cache release, both schedulers' missing retirement acknowledgement,
  and failure before reuse of the game slot. [Hardware test log](../../runs/rust-sequence-hpo/final-hardware-tests.log)
- The selected Python config, lineage, and MLP inference suite passed 220
  with 19 skips. The implementer's six-module Python selection passed 330
  with 11 skips, and seven progress checks passed separately. Static checks
  passed with Ruff and mypy across 290 source files.
- The final `cargo test --offline --locked --manifest-path native/Cargo.toml
  --lib player_runtime` passed 47 tests with one ignored. It used the
  repository's existing CPython 3.14 dynamic linker setup for test executables.
