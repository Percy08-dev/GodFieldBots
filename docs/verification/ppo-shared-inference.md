# PPO resource measurements and shared inference status

## Simulator 072dabb shared MLP continuation

Implementation starts at Bots `c78f8a0174e7cdba8dfd9d006983c6541e85d3fe`
on `feat/ppo-shared-inference`. [ADR0053](../adr/0053-mlp-shared-batch-inference.md)
and the [accepted design](../design/mlp-shared-batch-inference.md) describe one
shared GPU service per invocation for HPO Model4 and ordinary Model3.
The local implementation milestones are `67e03b9` (service), `facd4b8`
(calibrated admission) and `1ecf1e9` (external audit containment). Root accepts
the bounded implementation and physical evidence described below.
The historical sections below retain their original revisions and contracts.

### Implementation and contract verification

CPU participants generate candidates, tokenize their own inputs, sample with
their own RNG and submit Act. One clean broker uses official TorchRL 0.15.0
InferenceServer/MPTransport and retains original GPU-child custody. Immutable
checkpoint publication uses the official weight-update path after commit.
Length-sensitive MLP arithmetic stays within each row: transport batching is
real, while tensor math batch size is 1. The coordinator remains Torch-free.
Responses bind execution, game, participant, request, observation revision and
checkpoint. Expired/ended/failed/retired replies produce no action or transition.

Independent correctness and architecture reviews accepted implementation and
repair wave 1. The repairs cover ordinary cleanup after service errors,
pre-registration staging custody, and complete service/publication references
independently anchored in hash-protected updates, original proofs and existing
invocation rows. This prevents deleted references from bypassing cold proofs.
Schemas are explicitly versioned: shared execution 2, HPO owned update 7/proof 6,
ordinary update 5/proof 2 (unpacked) or 3 (packed), usage/invocation 1, and global
GPU termination 2. Historical readers retain their original branches.

Fixed-checkpoint FP32 activation passed for Model3/4 on CPU/CUDA. Adversarial
private values, history lengths, candidate counts, ordering, same-game other-seat,
different-game same-seat and trial ID reuse preserve the reference row exactly.
Direct and wrapped official mixed-checkpoint tests observed batch 2 in both
orders; every row matches its standalone checkpoint at `atol=rtol=1e-5`.
Private dependence would fail even within that tolerance. Old valid requests
complete using their acceptance checkpoint; retirement, deadlines, failed
service and actual reaping have separate fault coverage.

Focused actual IPC/CUDA runs passed 67 cases, then 101 affected repair cases,
72 final shared/legacy cases and 3 final CUDA cold cases. Source inventory tests
passed 41, configuration regressions 152, and legacy cleanup 3. Full Ruff and
mypy passed (249 modules). Optional-free installed-wheel checks passed with
Torch/TorchRL/TensorDict/Optuna/PyArrow/LightGBM/native absent. Exact commands,
logs, source hashes and review manifests are retained under
`runs/shared-inference-activation/`; the implementation report is
`.superpowers/sdd/approved-plan/implementation-report.md`.

Whole-suite result is **51 FAIL / 7,158 PASS / 574 SKIP** (1,839.20 s), followed
by bounded fixes and rechecks; it is not a wholly green invocation. Five new
early-guard regressions were fixed (234 affected passes). Fourteen temporary
verification-environment failures were repaired and rechecked (14 passes).
One `/tmp` protection case also fails on base and passes in the actual branch;
one legacy evaluation fixture lacked a required worker cap and was corrected
(133 affected passes). The remaining 30 Tree cases fail on `c78f8a0` too because
the owner previously deleted their absolute historical training fixture. It was
not recreated or suppressed. Those legacy Tree/native fault and artifact-loading
paths remain unverified; independent reviewers accepted this bounded
whole-plus-delta disposition. See `baseline-triage/classification.json`.

### Simulator and environment identity

A clean disposable Simulator clone at `/tmp/gfb-shared-simulator-072dabb` uses
revision `072dabbca1458c9526897781d2fd64edeeb4f9ca` without modifying the sibling
checkout. Section 21.1's physical spec07 SHA-256 is
`8e5223b45a6e8c4c80e272736ed97b9e271136c939eca74427a035ddfdbfcac2`;
the complete 13-file normative inventory digest is
`da4bde0ce82d91dc10c4a8a7f012c1cb95b0079113344453b956f638fb1d39d4`.
`cargo build --release --locked --offline --features test-support --bin gf-test-server`
passed. Executed binary SHA-256 is
`76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d`.
The test server does not consume build.rs revision metadata: clean build
procedure, source inventory and binary hash are separate correspondence evidence,
not embedded revision attestation. Every actual run records these identities.

`uv lock` and `uv sync --locked --extra torch-cuda --extra hpo --extra shared-inference`
passed. Runtime is Python 3.14.10 with GIL, NumPy 2.5.4, Torch 2.14.1+cu132,
TorchRL 0.15.0 and TensorDict 0.15.2 on RTX 5070 Ti (16,303 MiB).
Local IPC/CUDA checks require host permissions; sandbox EPERM/NVML failures are
not passing tests. Hardware observation is retained in
`runs/ppo-shared-inference-v1/hardware-observation.json`.

### Physical comparison protocol and accounting

The approved panel is seven valid HPO invocations / 384 natural games:
resident/shared C3 each 16 games × 2 updates, then shared C1/2/4/8/16 each
32 games × 2 updates. Width 192, seed 17, PPO minibatch 128 and four epochs stay
fixed. Matched C3 uses the same mixed-refresh-16-v1 opponents and 2 GiB collector.
All scaling points use the existing generic 32-game schedule (4 Random,
4 heuristic and 8 frozen seat-swap pairs), with both configured collector and
actual storage budget set to 4 GiB. Global limits remain 24 GiB RAM/12 GiB CUDA.
These short `update_cap` parents remain incomplete at D10000, not qualified
learning or final-evaluation evidence. Ordinary Model3 additionally runs C2.

The disposable driver records exact code/config/spec identities, collectors,
actual Create-to-terminal game intervals, accepted decisions, per-window service
counters, encoding/framing timings, PSS/RSS/private pages, unique bank lengths,
whole-device GPU use and memory pressure. First and second collections are
separate. Sampling every 350 ms plus sampling duration can miss brief peaks;
process traversal is not atomic. Whole-device GPU use includes unrelated users.
Bank lengths are backing sizes, not resident pages. Logical owner counters are
not added to observed PSS. Transport mean is delta requests / delta physical
batches; per-request batch observations are size-biased and reported separately.
Allocator counters are post-forward-host, process-lifetime/no-reset observations
with no added synchronization. Allocated and reserved are not independent sums.

C16 initially used a reviewed, fingerprint-bound disposable admission hunk;
production admission was unchanged. The failed first attempt retained one update
before an observer thread-children race cancelled it. Its whole parent remains
failed/ineligible. The driver race fix was independently accepted; a fresh C16
replacement completed 64 games and two updates, max overlap 16, sampled peak
13.562 GiB PSS and no fresh pressure. The first-drain screen included actual
Source remainder, new numerical/save allowances and unmapped bank backing once.

The first actual cold audit exposed two separate reader issues. Collection9 was
missing in two Model4 header branches; both were corrected without weakening
manifest correspondence. External numeric reconstruction also incorrectly
reused the live sealed-bank receiver's 512 MiB limit. Its first archive requires
583,949,647 bytes under existing managed decoding estimates. Flat audits
retain their joint original/numeric pool. With a larger external allowance, the
private adapter reserves the bounded original pool once and admits numeric peak
separately in the outer caller/model account. Producer budgets, 32× guards, hashes
and original comparisons remain unchanged. Corrected audit identity is recorded separately from the measured
producer. The final containment reader passes the complete original/numeric re-audit
without changing producer settings or data.

### Physical results and selected settings

The required panel completed 384 natural games. The predeclared reverse-order
C8/C4 confirmation added 128 games because their initial collection rates were
within roughly 1%; both confirmation runs completed. No further repetitions
were used to select a winner. The separate ordinary public-CLI run completed
32 natural games, two updates, 401 learner decisions and 16 optimizer steps.
Its recorded Create-to-terminal intervals show actual overlap 2 in both
collections, and invocation records include G0/G1/G2 publications and finished
service retirement. All parents pass full Original/Replay/numeric and applicable service/termination
cold audits. These execution results are not learning qualification.

The matched C3 populations have the same 566 accepted actions. Shared collection
is about 5% slower (30.619 versus 29.201 seconds combined), and the full invocation
is about 8% longer. Peak PSS is 3.895 GiB lower; whole-device peak is 1,815 MiB
(1.772 GiB) lower with the same initial 1,898 MiB. Whole-device differences are
sampled observations with unrelated users included, not exact per-process VRAM.
Resident C3 remains the throughput choice for the specified mixed16 preset;
shared C3 is an explicit memory-saving option, without a speedup claim.

For the shared 32-game schedule, select **C4 / max_batch_size=4 / 1 ms** in
`configs/hpo/mlp-shared-inference-scaling.toml`. C4 and C8 form a throughput
plateau: C8's small initial lead reverses in confirmation, while C4 stays stable
and uses about 2.37 GiB less PSS. C16 achieves actual overlap 16 but uses more
memory and has longer queue tails. This choice concerns collection throughput
and headroom, not policy quality or a curriculum change. The matched C3 preset
stays C3; historical resident/formal presets retain their contracts.

The primary points below are first, followed by the two confirmations. Matched
C3 has 16 games per collection; scaling points have 32. Actions/s includes all
participants. Whole invocation includes collection, Source, updates, export,
publication and cleanup; preparation is excluded. Two collections with different
game populations do not isolate startup cost by subtraction.

| Condition | Games/s first / second | Actions/s first / second | Actual overlap | Whole invocation s |
| --- | ---: | ---: | ---: | ---: |
| Resident C3 | 0.878 / 1.458 | 18.65 / 20.60 | 3 | 42.39 |
| Shared C3 | 0.839 / 1.386 | 17.82 / 19.58 | 3 | 45.78 |
| Shared C1 | 0.674 / 0.843 | 13.74 / 14.94 | 1 | 105.43 |
| Shared C2 | 0.927 / 1.179 | 18.90 / 20.90 | 2 | 82.21 |
| Shared C4 | 1.079 / 1.343 | 21.99 / 23.79 | 4 | 74.34 |
| Shared C8 | 1.093 / 1.352 | 22.28 / 23.95 | 8 | 74.52 |
| Shared C16 | 1.042 / 1.331 | 21.24 / 23.58 | 16 | 77.94 |
| Shared C4 confirmation | 1.068 / 1.343 | 21.75 / 23.80 | 4 | 74.81 |
| Shared C8 confirmation | 1.024 / 1.312 | 20.87 / 23.25 | 8 | 77.13 |

| Condition | Managed RAM / CUDA GiB | Peak PSS / private / RSS GiB | Whole GPU peak / initial MiB |
| --- | ---: | ---: | ---: |
| Resident C3 | 15.043 / 8.368 | 9.708 / 9.277 / 12.323 | 4607 / 1898 |
| Shared C3 | 11.093 / 3.592 | 5.813 / 5.389 / 8.019 | 2792 / 1898 |
| Shared C1 | 11.683 / 3.578 | 4.689 / 4.268 / 5.941 | 2900 / 1898 |
| Shared C2 | 12.388 / 3.585 | 5.285 / 4.861 / 7.014 | 2900 / 1898 |
| Shared C4 | 13.797 / 3.599 | 6.464 / 6.037 / 9.146 | 2900 / 1898 |
| Shared C8 | 16.616 / 3.626 | 8.845 / 8.417 / 13.416 | 2900 / 1898 |
| Shared C16 | 20.383 / 3.680 | 13.562 / 13.133 / 21.947 | 2916 / 1914 |
| Shared C4 confirmation | 13.797 / 3.599 | 6.475 / 6.048 / 9.157 | 2900 / 1898 |
| Shared C8 confirmation | 16.616 / 3.626 | 8.835 / 8.407 / 13.406 | 2900 / 1898 |

| Shared condition | Mean transport batch first / second | Queue p95 ms first / second | Max batch first / second |
| --- | ---: | ---: | ---: |
| Shared C3 | 1.47 / 1.40 | 42.35 / 43.66 | 3 / 3 |
| Shared C1 | 1.00 / 1.00 | 27.62 / 21.81 | 1 / 1 |
| Shared C2 | 1.22 / 1.35 | 33.24 / 28.92 | 2 / 2 |
| Shared C4 | 1.71 / 2.11 | 56.61 / 58.96 | 4 / 4 |
| Shared C8 | 2.33 / 2.60 | 118.56 / 122.09 | 8 / 8 |
| Shared C16 | 3.09 / 3.62 | 236.83 / 233.92 | 16 / 16 |
| Shared C4 confirmation | 1.71 / 2.11 | 58.25 / 59.66 | 4 / 4 |
| Shared C8 confirmation | 2.26 / 2.57 | 116.95 / 124.00 | 8 / 8 |

Tensor math batch size is 1 throughout. Mean physical transport batch is the
request-count delta divided by the physical-batch-count delta; per-request
batch histograms are size-biased. Queue p95 uses the retained complete window
of acceptance-to-forward-start observations and linear interpolation. Larger
transport batches therefore do not imply vectorized candidate scoring.

The C16 table reports the **actual experimental v0** admission used for that
run (20.383 GiB RAM). Post-calibration production preflight instead estimates
22.253 GiB RAM / 3.680 GiB CUDA and admits it at unchanged 24/12 GiB limits.
Only its admission screen differs; numerical/runtime source correspondence is
retained separately. Other points use the production calibrated recipe at
`facd4b8eee4972ca7e4a35bd2ac3234ee3be99cc` from a clean immutable clone.
The ordinary Model3 run uses that same clean source and public CLI.

`matrix-summary.json` and `phase-estimate-observation.json` retain every point's
phase estimates, sampled maxima and differences. No independent save peak was
sampled where absent; a last-phase sample is not a synchronized device measure.
The calibrated host estimate uses private participant high-water, an empirical
shared-PSS coefficient, broker/service/coordinator residency and reachable
Source/collector/numeric/save increments. Parent is nested in Source. The shared
coefficient may overlap some Source-backed pages; inode lengths cannot support
an arbitrary subtraction. This explicit soft-model uncertainty does not reject
the validated C16 workload. Larger models, full-cache occupancy, long-run
allocator growth and extreme history/search/framing shapes remain unmeasured;
no universal physical-RAM bound is claimed.

Each run retains `preparation.json`, exact rendered TOML, `conditions.json`,
collection reports, `samples.jsonl`, progress, training report, invocation timing
and actual GPU termination. Code archives/manifests are under
`runs/ppo-shared-inference-v1/final-execution-source/` and the separate provisional
execution archive. The first failed C16 parent remains wholly ineligible:
60 game-creation intervals, 52 terminal intervals and 8 retirement intervals
were observed before cleanup; interrupted games are not invented outcomes.
No parent was resumed or shortened to make the panel pass.

### Final audits, checks and limits

Both code-review roles are READY for service/integration, calibrated host
admission and external audit containment. The final reader passes **all nine
HPO parents / 512 natural games**, plus ordinary Model3 / 32 natural games.
Each HPO parent has two committed updates and remains incomplete at D10000:
294 learner decisions for each matched parent and 628 for each scaling parent.
The ordinary parent is complete only for its bounded two-update recipe, with
401 learner decisions and 16 optimizer steps. No final-test evaluation, learning
quality, full-D completion or LightGBM fitting gate is claimed.

The C16 audit preserves all 249 producer modules and the experimental admission
except three recorded reader corrections: Collection9 header classification,
owned external capacity and scan cleanup. The exact corrected reader hashes are
`052126905fbddb9c3b6eb0bbba51435f0fb900a8bbe1c05226b74cb29b5d32b0`
(card_instance_source),
`8c6ecb7f5d89ffefac7b4ef1c3d6231a5f845fa0457cb96a31fd627e2d7db5df`
(hpo_owned_restore) and
`6ca317ef88da43cf1a7847c2f37573d704721d45722434af68707464b49dcb04`
(structured_lineage). The other eight HPO parents and ordinary use a separate
reader copied from clean `facd4b8`, with only the last two files overlaid.
Producer identities, original files, policy mathematics, recorded limits and
32x reconstruction guards remain unchanged. Exact differences and commands
are retained in `actual-c16-audit/` and `matrix-final-audit/` under the activation
artifact directory. An intermediate attempt used stale helper `29e57d...`;
its failure is verification-environment evidence, not a test of the reviewed
containment code. Final commands assert exact hashes before imports.

The fresh external caller allows 4 GiB; its original pool remains at the recorded
2 GiB preprocessing limit. Its full-capacity loan plus first C16 numeric peak is
2,731,433,295 bytes, within that caller allowance. Positive pool/native/control
retirement and decoded last use refund each loan once. Existing 2 GiB flat
callers and borrowed custody retain their prior behavior. This is offline
capacity accounting, not a change to live Source 2 GiB or numeric 512 MiB.
Nested borrowed custody remains future work if a concrete caller requires it.

Validated journal timing extraction observed only scalar timing fields after
the unchanged strict reader returned. All 16 observed HPO batch action/journal
counts match original collection reports. At preferred C4, aggregate request
encoding/write plus response-body/JSON decoding totals 0.615 s and 0.544 s,
about 2.1% and 2.3% of the corresponding collection walls. These are owner sums,
not a critical-path decomposition. Header waits total 39.8/39.0 s across
concurrent owners and include computation/queueing; they are not transfer-only.
The raw measurements do not justify a rollout `transition_chunk_size`
experiment. Pack recording/writer and Source/numeric phases remain separately
recorded in original reports/progress; experience still commits only after Act
acceptance. No transport or rollout semantic change was introduced to chase a
benchmark result.

Final verification after containment:

- `ruff check . --output-format concise`: PASS.
- `mypy src`: PASS, 249 source modules.
- Containment: 27 focused PASS, 69 existing ownership/integrity/cleanup PASS;
  independent narrow containment run 21 PASS.
- Host: 159 affected PASS, 13 final arithmetic/raw-byte PASS;
  independent narrow host run 11 PASS.
- `uv build --wheel --offline`: PASS. Wheel SHA-256
  `10208f201542ad874b4fbe43fce59d1bb96762c20f759e4d4be5ae87b2b46832`;
  all 249 Python modules match source byte-for-byte.
- Installed mandatory-only wheel: ordinary and all three HPO presets load;
  CLI/HPO help PASS with Torch, TorchRL, TensorDict, Optuna, PyArrow, LightGBM
  and native absent and unimported.
- Full Original/Replay/numeric audits: nine HPO PASS and ordinary PASS;
  actual persisted global schema1/schema2 or ordinary local service proofs pass.

The whole-suite 51-failure disposition above remains visible; 30 owner-deleted
historical Tree fixture cases are not fresh verified coverage. Do not infer a
whole-suite green result from targeted checks. No further whole run was needed
after the scoped fixes; exact affected tests and source identities are recorded.

Retained artifact index (all under the isolated branch worktree):

| Evidence | Location |
| --- | --- |
| Primary and confirmation runs, raw reports/samples/configs | `runs/ppo-shared-inference-v1/matrix/` |
| Matrix summary and per-phase estimate/observation differences | `runs/ppo-shared-inference-v1/{matrix-summary,phase-estimate-observation}.json` |
| Predeclared near-tie repeat conditions | `runs/ppo-shared-inference-v1/near-tie-confirmation-plan.json` |
| Clean producer archive/1125-file manifest and drivers | `runs/ppo-shared-inference-v1/final-execution-source/` |
| Provisional C16 source and exact single admission hunk | `runs/shared-inference-components/provisional-execution-source/` |
| C16 complete audit/producer-reader identities | `runs/shared-inference-activation/actual-c16-audit/containment-final-result.json` |
| Other eight HPO + ordinary audits/timing extraction | `runs/shared-inference-activation/matrix-final-audit/` |
| Source-matched final wheel/import/static evidence | `runs/shared-inference-activation/final-*` |
| Whole attempt and baseline dispositions | `runs/shared-inference-activation/{whole-candidate-fix1,baseline-triage}/` |

Reproduction depends on these exact code/dependency/Simulator/driver/config and
checkpoint identities, retained originals and execution conditions. A seed alone
does not guarantee deterministic training across software or hardware versions.
Keep failed/ineligible parents separate, and leave the local feature branch
unmerged and unpublished for handoff. GRU then Transformer require their own
state-isolation and physical evidence before activation.

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

## Historical accepted scope before Simulator 072dabb

This section records the earlier milestone; current acceptance is above.
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
