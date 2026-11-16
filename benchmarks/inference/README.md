# Experimental inference comparison

These tools compare execution settings without changing production defaults,
model parameters or learner execution. They require the project's existing
Python3.14/Torch environment and recorded corrected-card-instance HPO data.
Run from the repository root with `.venv/bin/python`.

The protocol, acceptance conditions and results are in
[`docs/verification/inference-performance-comparison.md`](../../docs/verification/inference-performance-comparison.md).
The [design](../../docs/design/inference-performance-comparison.md) fixes the
information, reset, memory and failure boundaries.

Prepare a fixed original checkpoint and real visible-input prefixes:

```sh
.venv/bin/python benchmarks/inference/replay.py prepare \
  --source <accepted-parent> \
  --checkpoint <accepted-parent>/artifacts/000000/inference.json \
  --output <new-input-directory> --endpoints 0:3,2:3,12:19
```

Endpoint indices must exist in that parent's own recorded episodes. Preparation
uses the full-original parent validator and existing Numeric6 reader. It exports
no rewards, hidden state or management facts as tensor inputs. Original hashes,
schema identities, producer and resources remain in separate metadata.

Options are a closed JSON object, for example:

```json
{"packing":"reuse","chunk":1024,"alignment":0,"feature_alignment":0,"compile":"none","precision":"ieee"}
```

`packing` is `original`, `bulk` or `reuse`; chunk is256/512/1024/2048;
`alignment` controls candidate-row buckets and `feature_alignment` separately
zero-extends candidate-feature/numeric-weight K dimensions. Alignment values
are0/8/16/32/64/128/256. Compilation is `none`, `default` or `reduce-overhead`;
precision is `ieee` or `tf32`. Exact default options bypass the installer.

Run an isolated fresh-process replay:

```sh
.venv/bin/python benchmarks/inference/replay.py run \
  --input <prepared-input-directory> --options <options.json> \
  --output <new-report-directory> --cache <new-cache-directory> \
  --cache-state empty --warm-loops 5 --device cuda:0
```

Use a new report directory and `--cache-state populated` with the same populated
cache to measure compiler-cache reuse. Add `--profile` for a separate profiler
population; profiled results are excluded from timing aggregation. First call,
compiling loops, steady loops, CPU time, CUDA-event stream intervals, allocator
peaks and direct-process RSS have distinct scopes. CUDA-event elapsed time
includes host-induced GPU idle; it is not summed kernel busy time. Profiles
retain kernel names, operator shapes and copies. A TF32 flag alone is no evidence
of Tensor Core use. Compilation covers the actual shared candidate scorer;
frame/event/temporal computation and policy control remain explicitly eager.

Actual trials reuse the original resident coordinator, admission, Source and
positive termination receipts:

```sh
.venv/bin/python benchmarks/inference/actual.py \
  --config <fixed-protocol-config.toml> --options <options.json> \
  --run-id <new-uuid> --context-id <new-uuid> --output <new-report-directory> \
  --freeze <root-issued-freeze.json> --execute
```

Without `--execute`, this validates materialization/admission only. The helper
requires local retained historical
`runs/hpo-producer-owned/assessment/device-io-detail-v1/{runner,measure,analyze}.py`.
These ignored prerequisites are deliberately external to the source package;
preserve them with their frozen hashes and original datasets. A fresh checkout
alone cannot reproduce this particular saved experiment.

The root freeze is schema1 with `git_revision` and a `files` path-to-SHA256 map
covering source, helpers, inputs/configuration, installed native binary and
Simulator binary/specification/configuration. Actual trials are fresh-only,
width192/layers2/seed17/workers4/concurrency2/games16xupdates2/epoch1, with original
budgets and clocks. They must stop as healthy `incomplete/update_cap`; they create
no objective or training-quality evidence. The sidecar identifies experimental
execution separately from base production identities. Resident workers receive a
96MiB all-cache experimental reservation under the existing caps. Reduce-overhead
compilation is replay-only because its captured private tensors lack an owned
resident reset hook.

Aggregate matching populations:

```sh
.venv/bin/python benchmarks/inference/aggregate.py --kind replay \
  --pair <original-report> <candidate-report>
.venv/bin/python benchmarks/inference/aggregate.py --kind actual \
  --pair <original-1> <candidate-1> --pair <original-2> <candidate-2> \
  --pair <original-3> <candidate-3>
```

Each median requires the same conditions and ordered option comparison across
pairs. Incomplete telemetry rejects. Disjoint phase ledgers and nested collection
observations remain separate. Compile payback covers installer and the complete
initial compiling population; it requires positive saving and at least two later
graph-free loops. Quality of TF32 and any production adoption remain later work.
