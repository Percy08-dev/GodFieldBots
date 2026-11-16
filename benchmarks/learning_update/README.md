# Experimental PPO update comparison

Run commands from the repository root with the existing optional CUDA packages:
`UV_CACHE_DIR=/tmp/gfb-update-uv-cache uv run --no-sync python -m ...`.
The accepted design is `docs/design/learning-update-optimization.md` / ADR0045.
Production defaults stay unchanged. Every output is fresh and immutable.

1. Prepare the complete-original six inputs and G0/G1/Adam/RNG starts:

   ```sh
   uv run --no-sync python -m benchmarks.learning_update.prepare inputs --output /absolute/fresh/fixed-inputs.json
   ```

   Preparation rechecks the investigated inventory, hashes, numeric6/schema5,
   fixed historical producer, complete Source and selected endpoint identities.
   No original metadata is rewritten. `diagnostic.py` imports historical resume
   through a separate validator; ordinary resume still requires current source.
   Its NumPy adapter never manufactures `OwnedNumericBatch` or Source authority.

2. After code reviews and CUDA host access, run the unprofiled fixed comparison:

   ```sh
   uv run --no-sync python -m benchmarks.learning_update.compare --input /absolute/fixed-inputs.json --output /absolute/fresh/core
   ```

   This runs exactly84 measured updates: MLP four variants, GRU/Transformer five,
   two batches, three forward/reverse/forward repeats. Each measurement has a
   separate warmup, with complete weights, Adam state presence/steps, CPU/CUDA RNG
   and PCG64 restored before both warmup and measurement. `--model` limits a
   diagnostic population and honestly changes its expected count; it is not the
   complete84-update evidence. Any failure is preserved and terminates the run.

   Core timing excludes complete Source validation, capability/FD management,
   input retirement and snapshot/save. It uses the SAME diagnostic adapter and
   tensor core for all variants. Host wall, process CPU, CUDA events, allocator
   peaks and sampled process PSS/RSS are separate. CUDA event elapsed time includes
   host submission gaps and is not pure kernel time. CPU timings are refused as
   substitutes. No profiler wrappers are installed in unprofiled comparisons.

3. Run profiling separately, including a separate shape-capture invocation:

   ```sh
   uv run --no-sync python -m benchmarks.learning_update.diagnose --input /absolute/fixed-inputs.json --output /absolute/fresh/trace --model gru --batch 0 --variant microbatch --start 0 --groups 1
   ```

   Add `--shapes` only for a new diagnostic output. The predeclared active window
   is1–16 groups, then collection stops. RLIMIT_FSIZE bounds each trace512MiB;
   missing CUDA activity and actual event/output counts are explicit. Trace counts
   separately report recorded kernels, device copies/directions, memory sets,
   runtime launches/copies and explicit runtime/driver synchronization APIs. Missing
   CUDA categories produce unavailable (`null`) counts. Counting streams events
   with a2MiB per-record bound instead of decoding the whole512MiB trace. Labels
   cover packing/HtoD, frame encoding, prefix/suffix, candidate, loss, backward,
   clipping, Adam and final sync when they occur in the window. Source/retirement/
   snapshot/save are excluded from core, observed in actual phase timings.

4. Summarize core evidence, then root explicitly selects one candidate/model:

   ```sh
   uv run --no-sync python -m benchmarks.learning_update.summarize --core /absolute/core --output /absolute/fresh/core-summary.json
   uv run --no-sync python -m benchmarks.learning_update.plan --candidates /absolute/candidates.json
   ```

   The candidate file maps `mlp`, `gru`, `transformer` to an applicable measured
   nonbaseline variant. Smaller memory/implementation wins on comparable time;
   there is no new error threshold or saved-weight/action equality gate. Plan
   writes fresh configs and18 alternating baseline/candidate trials under
   `runs/learning-update-optimization-v1/`. It preserves192width/2layers/seed17,
   workers4/concurrent2, games16×updates2, epoch1, D1000000, wall600 and phase caps.

5. Root freezes the reviewed committed source plus generated configs, observers,
   native binary and Simulator specs/binary/config. Supply all three configs:

   ```sh
   uv run --no-sync python -m benchmarks.learning_update.prepare freeze --output /absolute/fresh/freeze.json --reviewed-revision FULL_REVIEWED_HEAD --config runs/learning-update-optimization-v1/configs/mlp.toml --config runs/learning-update-optimization-v1/configs/gru.toml --config runs/learning-update-optimization-v1/configs/transformer.toml
   uv run --no-sync python -m benchmarks.learning_update.population --plan runs/learning-update-optimization-v1/actual-plan.json --freeze /absolute/freeze.json
   ```

   The latter performs PLAN_ONLY admission. Add `--execute` only for the reviewed
   physical population. It invokes the original coordinator exactly once per
   fresh run, original creator/module and actual lease reservation, separate
   SHA-bound worker settings and independent experiment sidecars. No inference
   bootstrap/96MiB charge or quality completion objective is introduced.
   Timing is sequential: no tests, reviews, audits or other benchmark jobs in
   parallel. Failures stay in place; budgets are never reduced to obtain success.

6. After timing, run the strict original Source/numeric/receipt auditor and final
   summary. Auditing verifies unchanged original hashes and stat identities:

   ```sh
   uv run --no-sync python -m benchmarks.learning_update.audit --plan runs/learning-update-optimization-v1/actual-plan.json --output /absolute/fresh/strict-audits.json
   uv run --no-sync python -m benchmarks.learning_update.summarize --core /absolute/core --actual-plan runs/learning-update-optimization-v1/actual-plan.json --output /absolute/fresh/final-summary.json
   ```

   Core summaries require a successful completion record and exact declared
   model/batch/variant/repeat population, including all actual baseline rows.
   A limited `--model` population stays labeled with its actual24or30updates;
   it is never presented as the complete84-update protocol. Summaries report
   both cumulative baseline and incremental effects, and actual
   whole phases. Inclusive phase walls may overlap; subtraction cannot establish
   I/O wait. Required healthy actual outcome is incomplete/update_cap with2updates,
   3generations and genuine termination. Mixed whole-run improvement supports
   retaining defaults. No learning-quality/HPO/final-test claim follows.
