# Ordinary Rust Model3 PPO update settings

Base: `feat/rust-player-runtime` at `4912a9f`. The approved plan sets the
initial experiment to 256 games, 64 decisions per logical batch, four epochs,
all-decision shuffling and CUDA groups of up to 16. The scope is the supervised
ordinary Rust Model3 route. ADR0054/0055 own its lifecycle and Source handoff,
and ADR0046/0052 describe the existing tensor planner and learner reservation.

## Configuration and saved identity

Add a closed, immutable `PPOUpdateConfig` schema 1 under
`training.ppo_update`, allowed only for ordinary Rust Model3. TOML and dotted
`--set` merge into the same typed value. An explicitly selected table can
use the plan's values as defaults for raw declarations and overrides. A
configuration with no table selects the historical recipe (16/128/4/all/1)
and retains its canonical representation/hash. A fresh experiment supplies
the explicit complete schema 1 declaration. Normalization of saved effective
JSON requires the full canonical table if present. Reject unknown fields,
versions and shuffle names, nonpositive widths and epochs, and game counts
that are not positive multiples of eight. The mixed Refresh recipe requires
16 games. The CPU execution width is always one.

The effective config, its hash and the existing Rust resume identity bind the
recipe. Resume loads the immutable config under the original parent lock.
For resume with unchanged input-file receipts and an omitted table (for
example, first run set the whole table via CLI), read the effective snapshot
under the original lock using the hash from the validated paused Run5, and
compare every non-recipe effective field with the invocation. An explicit
recipe must match the full snapshot. Reject changed input-file bytes even
if the effective result would match. Keep original input receipts, and
retain paused-run code, runtime, device and clock contracts.
A table-less historical effective config is readable without adding fields
or rewriting its originals. The existing artifact/array formats, collection
record versions and HPO config are preserved. The versioned nested table
distinguishes the new saved recipe.

## Schedule, ownership and audits

Extend the ordinary paired schedule using the existing seeds, stream domains
and snapshot draw sequence. For N games (N/2 pairs), schedule N/8 Random,
N/8 Heuristic and N/4 saved-policy pairs. Retain the previous ordering and
RNG calls at 16 games. The common loop uses the ordinary Rust recipe for
planning and the planned game count from the resulting pair schedule. Keep
the exact-16 legacy readers and existing Model4 HPO routes. The Rust actor11
reader checks the ordinary role and game count derived from the versioned
config, under its existing actual original ownership. Check all planned
games, learner seats, bindings, natural endings, replay, and numeric
correspondence. Derive the optimizer-step delta from the same recipe at live
commit and cold/readback audit, including agreement with the saved counters.
Schedules and settings are control data. Simulator 072dabbca1458c9526897781d2fd64edeeb4f9ca
and its section 21.1 input rules remain pinned. Policy-visible inputs,
event/rollout schemas, GAE, rewards and normalization stay as before.

## Update and capacity

At the owned Source handoff pass the validated recipe into the Model3 update
after the existing parity bypass and admission. Derive chronological flat
index runs from episode decision locations. Per epoch: all calls the exact
existing PCG64 permutation on the decision count; episode_order permutes
nonempty episode order; within_episode permutes each episode; episode_interleave
permutes episode order and decisions in each episode, then takes one per
remaining episode per round. Shuffle only after GAE and normalization. Keep
the scalar endpoint path for CPU and configured CUDA width one, with the
existing loss, sync, metrics, clipping and Adam.

For CUDA widths above one, partition only within each logical batch. Reuse
Model4's candidate-length sorting, padding policy, allocation estimates and
packed window assembly. Parameterize the MLP planning width while preserving
Model4's default of 16 and its sequence path. Prefer splitting a Model3
non-fitting group before staging or allocation. The Model3 model and existing
shared MLP forward own the numerical graph and existing PPO loss equations.
The candidate mask excludes padding from normalization/entropy and the group
mean is weighted by real decisions in the logical batch. Clip and step once
per logical batch including the tail. Avoid copying decision vectors or
token arrays across logical batches.

Calculate capacity before group allocation from the existing tensor/host and
operation limits plus measured CUDA free bytes. Use the planner's packed and
activation estimates as the incremental group demand, separate from the
existing managed total for CPU references and context. The managed device
check also charges the position buffer.
Add one model-sized gradient allowance after zeroing, and use free device
bytes plus idle bytes in this Torch allocator as the observed headroom. Live
learner and inference allocations already reduce physical free memory. The
existing owner separately admits the Adam, context and inference lifetimes.
The headroom estimate cannot guarantee physical fit under fragmentation or
concurrent external load. Split non-fitting groups before allocation. If a
packed singleton cannot fit, report the existing terminal capacity failure.
A real CUDA or numerical failure uses the same Source/local failure path.
Optimizer and PCG64 state stay with the original learner.

The deployed formal preflight checks the actual AST of the historical
`training_groups` and `Learner.update` functions against a pinned recipe. The
extended schedule and configurable batch-control loop are separate functions
so that formal validation and historical numerical dispatch still execute the
frozen code. The new loop shares the fixed PPO loss and network operations.
When the formal recipe is revised or the PPO objective changes, revisit
sharing the schedule construction and scalar update control.

## Implementation and verification

Implement configuration/snapshot and schedule/Source audit, then numeric
execution and focused tests. Verify table and override parsing, legacy byte
identity, four shuffles on unequal episodes, logical width/tails and exact
Adam count, fixed-rollout save/resume and the GPU width matrix against
matched initial weights and shuffle state. Record forward/loss tolerance,
per-batch membership/weights, RNG equality and measured differences in
gradients, moments and final weights.

Use the normal supervised train/resume command, actual sealed Source and
the pinned Simulator. Pause after the first 256 games and resume for the
second. Run the three repeats at C4/W4, C8/W8, C16/W16, C32/W16 and C64/W16
on the 16-logical-CPU host, Tokio, inference B64 with 1 ms formation wait,
width 16, same seed, initial policy, and resource budgets. Use the existing
process observer and independent operation verifier for original exits,
overlap, natural endings, stage times, peak RAM/VRAM and saved readback.
Record capacity/deadline/external-load failures with condition and measured
scope. No native changes or dependencies are planned pending review.
