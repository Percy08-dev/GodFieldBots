# Inference performance comparison

Approved scope: the 2026-11-16 plan and
[ADR0044](../adr/0044-inference-performance-comparison.md). Implementation,
independent reviews and physical measurements are in progress. No performance
or learning-quality result is claimed by this initial record.

Baseline: Bots `4dccb08`, Simulator
`91d5926a7572db327eaea3751e7c24f428120996`. Existing corrected card-instance
contract and schema meanings remain in use. Experimental identities and complete
actual runtime/configuration hashes will accompany the saved results.

The sandbox cannot access CUDA; the host reports RTX 5070 Ti, 16,303 MiB,
driver610.88, Torch2.14.1+cu132 and CUDA13.2. `g++` is installed. Nsight Systems
and Nsight Compute are unavailable; CUDA profiler traces must state the resulting
limits on Tensor Core claims.

Baseline check PASS: `.venv/bin/python -m pytest -q
tests/test_structured_network.py tests/test_hpo_numeric.py -k 'not cuda'`:
75passed in5.03s. Further evidence will replace this in-progress status.

The first independent review blocks timing on the modified default transfer
schedule, compiler backend argument routing, incomplete compilation payback,
unmatched actual conditions and incomplete-telemetry acceptance. All five are
repaired before any physical measurement. The revised code preserves default
transfer/enqueue order through lazy operands, shares the arithmetic, and charges
the complete experimental memory envelope. Covering checks pass100tests plus
57affected policy/reset/timing checks, scoped Ruff and mypy9files. The full suite
is outside the explicitly approved focused-test scope. Independent correctness
and conformance re-reviews are READY. The final aggregation fix also rejects
mixed models/options/conditions across repeated pairs; its two focused cases pass.

Fixed replay inputs are prepared with the existing full-original parent validator
and Numeric6 reader. Each model uses its recorded G0 checkpoint and the same
three declared prefix locations (`0:3,2:3,12:19`):28decisions,5,187candidate
rows, candidate counts1–1,552 (median7), and at most21events in an interval.
Inputs and checkpoint shards are copied once with hashes under
`runs/inference-comparison-v1/inputs/{mlp,gru,transformer}`. These are real visible
inputs, not a stress or quality population. Preparation passes for all3models.

The first preparation attempt correctly stops at a model-identity gate because
the initial tooling compared the full runtime model with the inference-only
projection. The implementation now uses the existing `inference_model` projection
and stores runtime resources separately; no identity gate is relaxed. No timing
population was produced by that failed preparation.
