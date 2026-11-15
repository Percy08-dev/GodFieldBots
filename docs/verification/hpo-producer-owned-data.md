# Producer-owned HPO verification

Status: implementation in progress; acceptance and candidate timings NOT RUN.
Specification: [design](../design/hpo-producer-owned-data.md),
[ADR0043](../adr/0043-hpo-producer-owned-data.md), approved user attachment.
Base revision: `1dd51e8d1ee734b32b051556589e41d46d565ed0`.
Simulator inspected revision: `91d5926a7572db327eaea3751e7c24f428120996`;
existing corrected contract/schema pins remain unchanged.

## Baseline and environment

PASS128 in24.20s, exit0:

```console
UV_CACHE_DIR=/tmp/gfb-uv-cache-owned-preflight uv run --no-sync pytest -q tests/test_hpo_config.py tests/test_hpo_selection.py tests/test_hpo_wire.py
UV_CACHE_DIR=/tmp/gfb-uv-cache-owned-preflight uv run --no-sync ruff check src tests
```

Ruff PASS. Initial uv commands without a temporary cache fail because the home
cache is readonly; these failures are environmental and retained in the execution
ledger. No dependency synchronization/installation occurred.
Sandbox NVML is unavailable; approved host read-only `nvidia-smi` succeeds with
RTX5070Ti16GB. This is availability evidence, not a workload or memory measurement.

## Independent review and execution

Full preimplementation architecture review: READY, `/root/owned_architecture`.
The snapshot review requires physically separate inference/resume banks because
an FD exposes its complete backing. Root accepts and records that remedy;
the reviewer confirms no remaining consequential blockers in the revised design.
Final correctness/conformance, full project/package checks and physical execution
remain pending; no performance or full-plan completion claim is made.

## Required matched comparison

Preserve explicit ADR0042-enabled control alongside the new producer-origin path.
192width/2layers, seed17,4workers/2concurrentgames,16games x2updates, original
decision/update/time/memory budgets. Three alternating pairs per model,18trials.
New directories only. Keep encoder/protocol/origin/config differences explicit.
No audit/test workloads during measurement. Retain failures and measured limits.

The following values are historical ADR0042 evidence from
[the investigation](../design/hpo-producer-owned-data-investigation.md),
not a newly rerun control. MLP is3trial medians; GRU/Transformer are1trial.
Independent component medians need not sum to the whole median.

| Whole execution phase, seconds | Historical MLP | Historical GRU | Historical Transformer | New matched control/candidate |
| --- | ---: | ---: | ---: | --- |
| Startup/other (historically combined) | 1.88 | 1.75 | 1.80 | NOT RUN |
| Collection, two batches | 48.35 | 48.59 | 49.31 | NOT RUN |
| Feature/data preparation | 19.02 | 19.24 | 19.35 | NOT RUN |
| Update including old runtime parity | 17.04 | 54.62 | 54.95 | NOT RUN |
| Save | 2.55 | 2.71 | 2.89 | NOT RUN |
| SYNC | 9.90 | 10.53 | 10.92 | NOT RUN |
| Cleanup | 9.16 | 9.50 | 9.83 | NOT RUN |
| Whole | 107.76 | 146.94 | 149.04 | NOT RUN |
| Runtime parity separately | not separately stated here | not separately stated here | not separately stated here | NOT RUN |
| New capture/transfer separately | not applicable | not applicable | not applicable | NOT RUN |

| Collection detail, historical MLP second batch | Seconds | New control/candidate time and calls |
| --- | ---: | --- |
| Before collector | 5.54 | NOT RUN |
| Collector entry | 0.04 | NOT RUN |
| Before output creation | 0.07 | NOT RUN |
| Owned archive path setup | 5.50 | NOT RUN |
| After games | 5.47 | NOT RUN |
| Remaining startup/candidates/inference/communication/storage | 13.45 | NOT RUN |
| Whole second collection | 30.08 | NOT RUN |
| Capture / tokenize / seal | not separately stated | NOT RUN |
| Candidate generation / inference / IPC | not separately stated | NOT RUN |
| Archive queue / write | not separately stated | NOT RUN |

New reports must distinguish exclusive parent wall spans from inclusive concurrent
operation distributions. Moving work between collection/preparation/update does
not by itself establish whole-execution improvement.
