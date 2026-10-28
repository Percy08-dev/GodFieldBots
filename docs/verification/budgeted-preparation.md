# Budgeted preparation and feature reuse

Date: 2026-10-29. Design: ADR0010. Implementation and acceptance in progress.
Do not interpret intermediate component checks as full-corpus or learning acceptance.

## Preserved reference

Reference revision: `1df884966fbf725f3086154b489e1315f5263c77`.
A source archive is available at `/tmp/gfb-performance-reference` for differential
checks; Git retains the durable reference. Original data/artifacts are not edited.
Simulator remains clean at `ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6`.
No Simulator Python environment repair is included.

Frozen dataset: `runs/ppo-comparison-v2/dataset.json`, content SHA-256
`ab5d4913fda45795d1cd5aa02a4a9c4c3464268298c4d3ea49d56d4909df770f`.
It contains81,523train and2,260validation decisions. Preserve the four completed
fits (observation17/29/43 and history17); history29/43 wait for equivalence and
independent acceptance. Repeat history17 only as an equivalence benchmark.
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
- Encoder/preparation/full-corpus/resource benchmarks: in progress, not accepted.
- Encoder/config independent correctness and implementation-conformance READY.
  Correctness independently compared2,692real candidate vectors, both viewers
  and two feature recipes: exact float64/MLPfloat32/name/signature equality.
  Focused encoder/native/numerical75tests PASS; conformance77focused tests PASS.
  Full preparation/measurement review remains pending.
- Rust experiment and remaining two fits: NOT RUN.

Use the amended ADR0010 measurement protocol for final wall/CPU/memory/storage
comparison. Record missing metrics as unavailable, sampling and shared-RSS
limitations explicitly, and retain output identity separate from diagnostics.
