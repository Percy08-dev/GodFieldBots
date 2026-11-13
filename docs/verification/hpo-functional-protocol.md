# HPO foundation functional acceptance protocol

Predeclared verification scope, not a formal search or strength experiment.
Execute only after source integration/reviews/static checks and source-matched
native support are ready. Keep all failed attempts as ineligible evidence. A bug
fix uses a fresh output namespace with the same declared case parameters; do not
shorten budgets, replace seeds or drop failed games to claim acceptance.

## Fixed small conditions

- Training and neural inference: host CUDA, same GPU, FP32 and adopted corrected
  Simulator91d5926/spec/catalog identities. No Simulator modification/rebuild.
- Fresh output namespace under `runs/hpo-functional-*`; controls/models/numerics/
  Optuna on ext4. Compressed originals use a new supported D-drive subdirectory.
- Training seeds17,23 as specified below. Evaluation GameSeed
  `00000000-0000-0000-0000-000000600001`, both seats, categorical sampling.
- Each run wall duration1200s with original hpo-deadline-v1 derivation; retain
  finalization/cleanup reserves. No deadline extension across pause/resume.
- Default managed RAM24GiB/CUDA12GiB, concurrency within the reviewed initial
  bounds. Record actual configured admission and observed usage/elapsed costs.
- Keep the candidate generator and public information contract fixed within each
  case. No rejection probing, SequenceOptions or alternate private views.

## Cases

1. **Three-model production matrix.** For MLP, GRU and Transformer, one fresh
   seed17 run to D8 with8complete collection games/update and1epoch. Verify real
   GPU inference/update, saved final-D checkpoint, complete original/native
   Source/Numeric5/Wire6 correspondence, expected optimizer steps and healthy
   close. Evaluate the final checkpoint on the common two-seat heuristic panel.
   These are inner runtime functional cases, not a campaign comparison/control.
2. **Normal-boundary run resume and incomplete cap.** MLP seed17, D1,000,000,
   collection8/epoch1, max_updates2. Pause after the first verified committed
   update, resume immediately with unchanged configuration/original deadline,
   then stop at the declared2-update cap. Expect incomplete belowD, no objective,
   both complete batches retained and protected G0/latest resume states. This
   deliberately tests refusal of incomplete learning, not a shortened success.
3. **Actual CLI/Optuna campaign and session resume.** MLP, attempted trial_count1,
   seeds[17,23], D8. Candidate space has fixed collection8/epoch1 and categorical
   learning rate[0.0002,0.0004]; other model/training settings use current defaults.
   Canonical default contrast is outside this space and runs separately at the
   SAME D/seeds with canonical collection16/epoch4. Set a10-second session budget
   so a completed operation boundary is
   saved, then resume with unchanged conditions until all required runs/evaluations
   and the exact tell receipt complete. Check actual sampler/distributions,
   separate control, fresh weights, seed equal-weight objective, cost/counts,
   exact idempotent tell and absence of formal/final64/Dataset authority.
4. **Second-model CLI and direct compare.** GRU, one attempted in-space canonical
   default proposal, seed17, D8 and the same evaluation panel. Compare its closed
   result against the MLP candidate across all declared left/right training-seed
   pairs with two swapped seats. Persist the complete exploratory matchup table;
   no final-test or strength claim. Settings-only stage import and conditional
   distribution/crash cases are additionally covered by actual local Optuna tests.

Expected initial workload is approximately124real games (including separate
default controls, matrix evaluation, resume batches and comparison), with modest
unique learning decisions except the explicitly incomplete2-batch case. Exact
actual counts and any preserved failed attempt must be reported from originals.
This bound describes the chosen functional protocol, not future HPO sizing.

The first concrete inputs are saved before any GPU execution under
`runs/assessments/ppo-hpo-foundation-20261114/configs/`; their pure-admission
receipt is `functional-configs-preflight.json` in the parent directory. All six
cases select the actual Rust preprocessing backend. Explicit functional-only
GRU/Transformer native base derivatives preserve the original acceptance presets
and Simulator paths. The initial output namespace is
`runs/hpo-functional-20261114-a01`, with matching fresh D-drive originals roots.
The MLP candidate remains collection8/epoch1; the MLP contrast and GRU canonical
proposal retain collection16/epoch4. No GPU/Simulator run is implied by pure
configuration admission, and no functional harness result is an Optuna trial.

## Acceptance evidence

Preflight must prove supported storage/C-drive free space, pinned runtime/native
identity, joint resources and one global GPU job admission. Record every CLI
invocation/config/source/seed/receipt; validate actual original journals/Replay,
natural ends, selected index replay/GAE/normalization, saved arrays/step math,
active reference protection and certain process/resource close. Interrupted or
unhealthy runs yield no objective. Verify normal resume retains the original
absolute deadline and failed/incomplete evidence remains ineligible.

After functional source freeze, run whole Python/static/Rust checks, independent
final correctness/conformance, optional-free wheel/CLI startup and the original
legacy corpus/index/saved-PPO proofs. Compare protected29814original/24fit hashes
and stat identities with `/tmp/hpo-originals-before.json`. Record PASS/FAIL/
NOT RUN with actual commands and limitations in `ppo-hpo.md`. No seed-only
determinism or trained strength conclusion follows from these checks.
