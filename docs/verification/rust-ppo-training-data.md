# Rust PPO training-data verification

## Pinned inputs

The implementation branches from `aab5f35d79a2da3339c25110c5012c3c19e9946c`.
Simulator `072dabbca1458c9526897781d2fd64edeeb4f9ca` supplies API v2,
`true-match-1v1-v2`, ActionConfig 3, and the `public-ja-v3` catalog at schema 2.
The catalog SHA-256 is `f51a3c73feb8b3f01837e65768aafb5a3aa805a33483308130244b33b16a8ad6`.
External interfaces §21.1 and §47 were read at that revision. The specification
file SHA-256 is `8e5223b45a6e8c4c80e272736ed97b9e271136c939eca74427a035ddfdbfcac2`.
The executed test server binary SHA-256 is
`76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d`.

The architecture reviewer cleared the live and saved designs before
implementation. The review resolved the shared recording budget, the
pre-seal versus post-seal evidence boundary and the Model4 empty marker.

## Numerical and operational verification

- Model3 used an actual supervised 16-game collection with 230 learner and
  436 all-seat decisions. Eight games shared one checkpoint across both
  seats. The Python reconstruction matched row/seat/candidate order,
  selection, old logprob/value and features exactly. Advantage and value
  target passed with `atol=rtol=1e-6`. With the GPU-shaped group planner on
  CPU, the first and tail losses at logical widths 64 and 128 and the
  singleton loss matched exactly. Tail sizes were 38 and 102.
- Model4 used an actual CUDA run with 174 collected and 64 selected
  decisions. Its explicit audit reconstructed Python GAE and the selected
  normalization, kept full trajectory targets, and applied the same feature
  and numerical checks as Model3. The independent first-loss comparison
  evaluated the saved arrays and Python reconstruction under the saved G0
  weights on CPU. Both gave `0.2401694431900978`. Actual Rust grouping ran on
  CUDA in the matched trials.
- Continuous and pause/resume runs completed two CUDA updates at C1/W1 so
  serving batch shape was fixed. Model3 processed 401 decisions and reached
  28 Adam steps. Model4 used a 200-decision comparison budget so resume
  performed a second optimization, reaching 12 Adam steps. Final weight and
  resume shard hashes matched within each pair. This covers named Adam
  arrays and Torch CPU/CUDA RNG arrays. The PCG64 shuffle and scheduling
  states also matched.
- With legacy Pack opening, rollout reconstruction and Python GAE guarded
  in the isolated Python processes, Model3 and Model4 ran the normal live,
  cold and resume phases successfully. The Model4 16-game/D64 public
  campaign finished actual final-policy evaluations and independently
  imported its saved history under the same guard. The process records
  identify five guarded training hosts, five cold hosts and the campaign
  history host.
- The operation verifier's `--full-source-audit` passed on a new 256-game
  Model3 run, a new 16-game/D64 Model4 run and an archived baseline Model4
  run. Focused tests rejected changed and swapped Pack bytes, missing or
  changed numeric shards, and Pack mutation during live numerical
  validation before Adam. They also exercised capacity refusal before
  shuffle and successful subdivision under the actual native ledger.

The focused host pytest run passed 793 tests with two skips. The optional
saved Model4 loss test passed separately. The `player_runtime::` Cargo
test filter passed 36 tests. Ruff passed and mypy passed for the 27 changed
Python source files. A full source mypy run reports the existing
`shared_inventory.py:194` union error, reproduced at `aab5f35`.
Independent correctness and architecture-conformance reviews cleared the
final changes.

## Matched CUDA measurements

Baseline `aab5f35` and candidate `1268781` each ran three sequential trials
per model with seed 17 on an RTX 5070 Ti. Initial weight hashes matched
across all six trials within each model. The source-matched release wheel
used CPython 3.14.10, NumPy 2.5.4 and Torch 2.14.1+cu132. Its native source
SHA-256 is `3bc9c92fe0436ab83cb7c09c3691312a380f4a89153b01064ac08388099bff8e`.

Model3 used 256 games at C16/W16, inference cap 64, formation wait 1 ms,
logical width 64 and GPU learning width 16. Each trial recorded 4,928
all-seat and 2,541 learner decisions and completed 160 Adam steps. The
shared limits were 8 GiB recording, 16 GiB Source and 12 GiB CUDA.
Model4 used `mixed-refresh-16-v1`, C1/W1, inference cap 1 and a 64-decision
budget. Its common limits were 1 GiB recording, 2 GiB Source, 512 MiB numeric,
24 GiB managed host and 12 GiB CUDA. Each exact-D trial used 64 of 174
learner decisions and finished four Adam steps before pause.

Each table entry is the median of the three trials. Post-collection
preparation is the difference between the host collection phase and the
persisted collection time, calculated per trial. Update includes input hash
checks and PPO. Cold is the measured post-host verification phase. PSS and
VRAM are sampled peaks per invocation, with VRAM measured device-wide.

| Invocation | Collect s | Post-collect s | Update s | Save s | Cold s | Wall s | PSS GiB | VRAM MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Model3 baseline | 64.34 | 55.05 | 16.55 | 0.51 | 89.75 | 230.64 | 2.765 | 3,603 |
| Model3 Rust data | 68.99 | 3.63 | 16.71 | 0.37 | 0.31 | 93.85 | 2.612 | 3,678 |
| Model4 baseline fresh | 9.63 | 5.13 | 1.60 | 0.18 | 6.36 | 27.14 | 2.148 | 3,618 |
| Model4 Rust data fresh | 9.86 | 0.21 | 1.05 | 0.16 | 0.19 | 15.45 | 2.278 | 3,687 |
| Model4 baseline resume | — | — | — | — | 6.39 | 16.43 | 0.833 | 3,362 |
| Model4 Rust data resume | — | — | — | — | 0.20 | 4.47 | 0.901 | 3,426 |

Model3 wall fell 59.3%. Model4 paired fresh-plus-resume wall fell from
43.57 s to 19.91 s, a 54.3% reduction. The measured resume restored the
exact-D checkpoint with no further optimizer work. The separate two-update
comparison above exercises optimizer work after resume. Collection cost
rose 4.65 s for the Model3 median and 0.23 s for the Model4 median. Both
workloads improved whole wall within their common limits and passed the
adoption condition for the explicit versioned mode.

The primary checkout retains original configurations, commands, observer
logs and full readbacks under `runs/ordinary-rust-ppo-update/`. The timed
directories are `rust-training-baseline-aab5f35`,
`rust-training-model4-baseline-aab5f35`, `rust-training-model3-1268781` and
`rust-training-model4-1268781`. Parity scripts and test logs are in
`rust-training-verification-1268781`. A temporary environment symlink during
wheel setup touched the baseline companion files after the timed baseline
runs. Those files were restored and checked against the original wheel's
hashes before running the baseline public evaluation.

## Model4 with the Model3 inference settings

The follow-up used Tokio, 16 game slots, 16 Rust workers, inference batch cap
64 and a 1 ms formation wait. Both lanes used seed 17, identical initial
Model4 weights and the pinned Simulator above. The candidate was `684f6df`,
which adds documentation to the measured implementation. The decision
budget remained 64.

The 16-game `mixed-refresh-16-v1` schedule has eight seat-swapped pairs and
reached eight games concurrently. Across three trials per lane, collection
throughput was 2.87 → 2.80 games/s and whole fresh-plus-resume throughput
was 1.62 → 4.05 used decisions/s. The median average inference batch was
2.08 → 2.10. The largest observed batch was seven in both lanes.

The 32-game standard HPO schedule reached 16 simultaneous games in all
six trials. The legacy checkpoint export with a 2 GiB Source allocation
requested approximately 2.32 GiB in total. Both timed lanes used the same
3 GiB Source allocation and 512 MiB numerical limit. Each trial collected
331 learner decisions, used 64 of them and completed four Adam steps.

| 32-game metric | Baseline | Rust data |
| --- | ---: | ---: |
| Collection games/s | 3.26 | 3.21 |
| Inference requests per collection second | 50.47 | 49.58 |
| Average inference batch size | 3.09 | 3.13 |
| Largest observed inference batch | 13 | 13 |
| Fresh plus resume wall (s) | 67.10 | 20.35 |
| Used decisions per whole wall second | 0.95 | 3.14 |

The figures are medians across three verified trials per lane, apart from
the largest batch observed. An additional 256-game Rust run reached 16
simultaneous games and collected 2,662 learner decisions at 3.25 games/s.
Its average and peak inference batch sizes were 5.40 and 15. The run saved
and resumed in 93.90 s combined with a 1 GiB numerical limit. The first
256-game run collected all games but failed before numeric publication at
512 MiB. On the saved input the Rust metadata reservation was about 746 MiB,
and the legacy token preparation estimate was 1,431 MiB against its
512 MiB implementation limit. The legacy figure is a capacity calculation
over saved inputs. The paired wall comparison uses the 32-game runs.

The primary checkout retains the follow-up inputs and results under
`runs/ordinary-rust-ppo-update/` in `rust-training-model4-c16-w16-b64-t1-g16`,
`rust-training-model4-c16-w16-b64-t1-g32-s3072` and
`rust-training-model4-c16-w16-b64-t1-g256-n1024`.
