# Budgeted preparation and feature reuse

Date: 2026-10-29. Design: ADR0010. Python implementation, independent reviews,
full-corpus equivalence, resource targets and requested fit continuation accepted.
No final-test or live-play win-rate improvement is claimed.

## Preserved reference

Reference revision: `1df884966fbf725f3086154b489e1315f5263c77`.
A source archive is available at `/tmp/gfb-performance-reference` for differential
checks; Git retains the durable reference. Original data/artifacts are not edited.
Simulator remains clean at `ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6`.
No Simulator Python environment repair is included.

Frozen dataset: `runs/ppo-comparison-v2/dataset.json`, content SHA-256
`ab5d4913fda45795d1cd5aa02a4a9c4c3464268298c4d3ea49d56d4909df770f`.
It contains81,523train and2,260validation decisions. Preserve the four completed
fits (observation17/29/43 and history17); history29/43 are now complete in fresh
output directories. The history17 repeat is only an equivalence benchmark.
The historical stopped outer-controller PID3833370 was absent on inspection;
no signal was sent. Remaining cells must be launched explicitly, never by
resuming that historical loop or signaling a recycled PID.

## Prior diagnostic evidence

`runs/assessments/performance-bottlenecks.json`, SHA-256
`93908f02c970ac4146f6bd19af08056d4555ba2feacbb0dcd3b549bf4984f24e`:
inspection417.913s, four fit preparation share98.763%, repeated whole-data
projection and833parent-chain checks. CPU inference fixed44/275candidate cases
were33.438/271.723ms, versus forward0.158/0.521ms. These fixed cases are not
population percentiles. Warm filesystem input0 and one-core CPU utilization
support CPU parsing/validation as the observed bottleneck, not disk throughput.
These measurements are historical diagnostics, not the new paired benchmark.

## Verification ledger

- Config unit:43tests PASS (preprocessing, existing config and LightGBM config).
  Ruff and mypy for changed config modules PASS.
- Initial fixed real-input CPU selection check:44candidates33.784->2.827ms;
  275candidates265.835->12.878ms (medians). Actor/critic float32 bytes, scalar
  float64 features and sampled choice sequences matched exactly; CUDA remained
  uninitialized. Evidence `runs/assessments/prepared-encoder-initial.json`.
  These are preliminary old-then-new diagnostics, three internal20-call trials,
  not the required alternating paired resource benchmark or population latency.
- First full frozen-corpus history preparation smoke PASS: dataset digest matches,
  numeric archive exactly matches retained history17 input SHA-256
  `c5c3f3c1ef4068766d81c7bd0cfa7a8957b140ad7ea1f207374fc725dcbcc94d`.
  Exactly13,568journals/6,784episodes validated,3parent chain checks; cleanup
  complete. Evidence `runs/assessments/preparation-real-smoke.json`.
  Incidental wall328.536s, process peak RSS584,351,744bytes, managed peak
  837,805,988bytes, scratch reservation peak933,246,877bytes. Small tests ran
  concurrently: these are correctness-smoke measurements, not paired speed
  acceptance. Source read/hash16.737s, decode61.910s, features20.025s,
  staging1.959s, validation/bookkeeping226.596s and finalization0.930s.
- Alternating paired resource benchmarks: all17preparation invocations complete,
  exact output identity across all variants; quantitative results below.
- Residual profile of first/middle/final collections across all three parents
  (144games,1,707selected actions):14.23s profiled, of which recursive memory
  accounting consumed9.09s. Profile overhead is substantial; do not extrapolate
  that percentage directly to unprofiled full-corpus time. Replaced only the
  private traversal algorithm with an iterative equivalent. A real two-view
  episode graph remained exactly10,748,694accounted bytes;9alternating-order
  trials of3traversals each measured63.371->35.964ms (1.762x). Evidence:
  `runs/assessments/preparation-accounting-comparison.json` and
  `runs/assessments/preparation-residual-profile.json`.
- Whole suite before final admission-race/accounting hardening:586PASS,5opt-in
  skipped,134.18s. Ruff PASS, mypy51sourcefiles PASS. Implementer final hardening
  regression76tests PASS plus7focused checks. Independent correctness recheck:
  35preparation/config tests plus14final focused tests PASS; no blockers.
- Encoder/config independent correctness and implementation-conformance READY.
  Correctness independently compared2,692real candidate vectors, both viewers
  and two feature recipes: exact float64/MLPfloat32/name/signature equality.
  Focused encoder/native/numerical75tests PASS; conformance77focused tests PASS.
  Full preparation correctness and architecture-conformance READY; architecture
  independently ran75focused tests. Measurement tooling correctness READY after
  PID identity, old-reference import origin and finite-deadline fixes;
  22tests independently passed, subsequent counter adapter brings suite to23.
  Semantic and structural readiness does not imply passed performance targets.
- Native feasibility screen complete (details below); native prototype/build
  NOT RUN by reviewed disposition. Reference reproduction, observation/index
  checks and remaining two fits PASS; details below.

Use the amended ADR0010 measurement protocol for final wall/CPU/memory/storage
comparison. Record missing metrics as unavailable, sampling and shared-RSS
limitations explicitly, and retain output identity separate from diagnostics.

## Paired measurements

Application changes are committed at `2b1a348`; the unchanged application was
measured using planner commit `c82e2de` (redundant matrix warmups removed).
Old reference is `1df8849`. Same interpreter, data, encoder and thread settings;
no concurrent training/tests during timed trials. Warmups are excluded below.
Each row contains three measurements; standard old/new order alternates.
The8GiB/64MiB matrix cell reuses the identical standard-new trials.

| Preparation variant | Wall median [min,max], seconds | CPU median, seconds | Maximum sampled group RSS, MiB |
| --- | --- | --- | --- |
| Old, no managed-budget support |512.032 [504.772,522.074]|511.550|8747.762|
| New,8GiB/64MiB |259.249 [255.539,261.984]|260.073|549.422|
| New,2GiB/16MiB |257.026 [256.581,259.519]|257.847|453.742|
| New,2GiB/64MiB |257.214 [256.884,257.527]|258.037|551.031|
| New,8GiB/16MiB |259.715 [256.903,261.323]|260.477|453.707|

The standard wall speedup is1.975x; CPU work is about halved. The budget matrix
does not establish a speed benefit from a larger admission ceiling.16MiB copy
windows reduce sampled peak memory. Managed limits are preprocessing admission
limits, not process-wide hard RSS limits; native fitting has separate memory.
Group RSS includes shared pages and can miss short peaks; wait4 high-water and
PSS are retained separately in raw records. CPU seconds use final wait4 totals,
not a sum of sampled and exit totals.

All outputs match the frozen dataset, ordered sample identity and history NPZ
`c5c3f3c1ef4068766d81c7bd0cfa7a8957b140ad7ea1f207374fc725dcbcc94d`.
Evidence: `runs/performance/20261029-budgeted-preparation/{results.json,
equivalence.json,comparison.md}` and each trial's resource/timeline records.

Storage tradeoff: staged numeric data plus the final NPZ coexist. New sampled
scratch reaches roughly933MB; the old archive alone is463,153,426bytes. A sampled
old scratch peak of0 misses that short-lived file and must not be reported as
no scratch usage. Logical writes, OS-accounted writes/cancellations and physical
device traffic are distinct. Original source trees are never polled as scratch.

| CPU selection diagnostic | Old p50 / p95, ms | New p50 / p95, ms |
| --- | --- | --- |
| Fixed44candidates |34.249 /35.142|2.870 /3.190|
| Fixed275candidates |264.855 /274.438|12.977 /13.769|
| Saved-input replay147decisions |36.949 /237.240|2.968 /13.177|

These are medians of three per-trial quantiles on the first eight validation17
groups (Random-assigned opponents), not population percentiles or live-match
throughput. No actions were submitted. Input, feature and selected-sequence
hashes match. Replay harness wall43.724->4.616s and CPU44.550->5.462s include
startup and equivalence work. Its retained input fixtures/native imports mean
memory does not fall proportionally: wait4 high-water is approximately853MB
for both variants. Evidence: `20261029-budgeted-fixed` and
`20261029-budgeted-replay` under `runs/performance/`.

## Installed wheel and GPU parity

Wheel SHA-256 `1bedf0c0ec2220741a1ef8ad01df2ddd785171377ed9320708c5ccd637ea0634`;
all51packaged Python files match source bytes. Separate environments:
`/tmp/gfb-budgeted-baseline` and `/tmp/gfb-budgeted-cuda`. Existing verified
dependency files were hard-linked into these new environments without changing
the originals; the new wheel was installed with `--no-deps`. This verifies
installed execution, not a fresh registry dependency resolution (offline NumPy
metadata was unavailable).

Baseline help/config imports PASS without Torch/LightGBM/PyArrow installed.
Native CPU MLP and retained LightGBM inference PASS; CPU path does not initialize
CUDA. An isolated numeric CUDA parity process checks the same44/275candidate
inputs at atol=rtol=1e-5: maximum logprob difference2.862e-6, value1.193e-7.
GPU allocated/reserved peaks35,277,824/35,651,584bytes concern this component test,
not a PPO training run. No new live PPO experiment was launched.

## Native feasibility disposition

`runs/assessments/native-feasibility-accounting.json` measures the full frozen
corpus with only timers around iterative memory accounting, without cProfile.
All numeric hashes still match. Accounting64.675s out of249.987s (25.871%) gives
an ideal upper bound of1.349x even if that component takes zero time. Decode
plus features74.771s similarly permits about1.427x with other costs unchanged.
Neither bounded candidate can reach the predeclared additional1.5x native gate.
Root retains Python, with independent architecture disposition READY. No Rust
prototype was built and no Maturin/PyO3 dependency was added. A fused native
validator/representation rewrite could change more costs but is a different,
larger design; this result is not a universal claim about Rust performance.

## Resource interpretation and acceptance

Independent read-only audit confirmed17/17successes, all input/output identities,
three measured trials per setting, source counts and managed/scratch limits.
Root accepts the Python performance targets:1.975x full preparation and
11.9x/20.4x fixed-case CPU selection p50 gains, with unchanged outputs.
This accepts performance/correctness for the frozen workload, not policy quality.

Standard median sampled group RSS is9,172,070,400->565,182,464bytes; separate
wait4 high-water medians are9,172,070,400->590,430,208bytes. The table above uses
maximum sampled peaks across trials, rather than these medians. Median sampled
logical process read counters (`rchar`) fall50.357GB->4.617GB, while logical
write counters rise0.463GB->0.933GB. These include buffer reads/writes and are
sampled lower bounds; some transient `/proc/PID/io` reads failed. Observed
`read_bytes=0` does not prove zero physical disk activity. Most of the elapsed
time and approximately one occupied CPU core are in parsing/validation and
bookkeeping, rather than GPU work. GPU fields for these CPU-only workloads are
not applicable. The original live-play whole-cycle timing remains historical;
no new whole-match latency claim is made from saved-input replay.

The fit monitor additionally samples total owned output (scratch+logs+artifacts)
alongside scratch. These are overlapping totals, never additive. Older completed
benchmark records lack that optional output peak, which remains unavailable.
The added regression covers an external symlink and coexisting scratch/artifacts;
measurement suite24tests PASS. No source-directory traversal was added.

## Final equivalence, fit continuation and acceptance

Full observation numeric input is exactly the retained archive SHA-256
`841c6f885e8d8f052abde2c5db788de761279947e36300e7bc87ff0254e7d0c8`.
The rebuilt `runs/ppo-comparison-v2/index-budgeted.parquet` has83,783rows and
the unchanged row-content SHA-256
`574b81ecf6178849c91bea20a9bc0d9906a0348f2c5357825140fc6fc8c618cf`.
Its Parquet layout hash differs as permitted. Original index/dataset untouched.
Evidence: `runs/assessments/budgeted-observation-index-equivalence.json`.

All three continuation runs are under
`runs/ppo-comparison-v2/lightgbm-history-optimized/`, in each observer directory's
`artifacts/` subdirectory. These artifacts are retained, not temporary scratch.

| Run | Purpose | Observed command wall, seconds | CPU seconds | Sampled group RSS peak, bytes | Model bytes |
| --- | --- | --- | --- | --- | --- |
| seed17-equivalence | reference repeat only |259.165|259.974|1115631616|313393|
| seed29-formal | fifth comparison cell |260.361|261.175|1106104320|313388|
| seed43-formal | sixth comparison cell |258.398|259.174|1115373568|313391|

Each run is complete/trained, schema2 fixed_data, with the original7200s budget,
81,523train/2,260validation rows, clean child health and ready inference. All
use the unchanged history numeric archive. The seed17 repeat has byte-identical
request, numeric inputs, model.txt and inference.json, and identical metrics to
the retained original. New fits use clean code
`cf898408510bb2d5af45e2ddff07f1985c6dfd72`. All three history seeds have
best_iteration95 and validation_l2 `0.59091264247396`; the retained observation
seeds have best_iteration98 and validation_l2 `0.6045529418541776`.
These are fixed validation regression metrics, not final-test or win-rate results.

Fit observer output peaks are approximately934.6MB including scratch, logs,
numeric input and model. Scratch-only peaks are approximately470.1MB and return
to zero. The internal scratch reservation933.25MB includes simultaneous staging
and destination input; these are different scopes, not additive numbers. Each
retained fit bundle is approximately463.49MB, dominated by its numeric input;
the three new bundles total1,390,473,920logical bytes. Root wait4 high-water
around657MB is not the simultaneous group RSS above.

Independent read-only final audit confirms the old four tracked file sets are
unchanged (small-file hashes plus size/mtime/inode; large old NPZ files stat-only),
all new child proofs, source counts, original budgets, output sizes and cleanup.
The comparison has six cells; the seed17 repeat is excluded. Shared generation
cost16,588.39885974396s is reported once with key
`125b48f095dc99bb395df149f0f54ff6221583ba3f41d91bc8961dfaafe516c4`.
Do not average historical and optimized preprocessing timings as a homogeneous
six-fit timing study; the separate paired benchmark is the speed evidence.

Evidence: `runs/assessments/completed-history-fits-budgeted.json`,
`prior-four-fits-preserved.json`, `budgeted-preparation-benchmark.json`, and the
three observer directories' measurement/timeline records. Final code checks:
`env -u VIRTUAL_ENV .venv/bin/pytest -q` PASS595,5opt-in skipped,130.87s;
`ruff check .` PASS; `mypy src` PASS51modules; tool mypy PASS2modules.
Root accepts the completed scope after independent correctness/conformance and
the bounded evidence audits. GRU/Transformer, finaltest and new live PPO runs
remain outside this acceptance.
