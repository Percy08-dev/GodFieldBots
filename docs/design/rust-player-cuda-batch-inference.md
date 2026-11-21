# Rust player CUDA batch inference

This design implements the approved Model3 and Model4 CUDA collection plan.
Simulator `072dabbca1458c9526897781d2fd64edeeb4f9ca` API v2, ruleset
`true-match-1v1-v2` and external interfaces §21.1 govern the information flow.
Actions 3, Journal 6, Replay, Pack 1 and the existing rollout and checkpoint
schemas retain their meanings. The catalog is `public-ja-v3` schema 2 with
SHA-256 `f51a3c73feb8b3f01837e65768aafb5a3aa805a33483308130244b33b16a8ad6`.

## Configuration and ownership

Ordinary PPO takes `inference_recipe = "batched-torch-v1"` in
`[run.player_runtime]`. Schema 6 HPO takes it in `[runtime]`. The campaign
chooses the recipe even when its ordinary base declares a runtime. The base
and campaign still agree on their Rust scheduling fields. Omission keeps the
scalar recipe and preserves historical effective configuration and binding
hashes. Explicit selection participates in effective configuration, the
runtime binding, and the saved resume checks. The same code/producer checks
apply to already paused parents. Serving on CPU uses the scalar path.

Rust owns request formation, participant state, and result acceptance. The
Python service receives the existing native operand views and uses the same
Torch serialization boundary as publication, the learner and CUDA accounting.
There is one formation window and one submission per native batch. The service
scatters provisional results back into ingress order for the native positional
identity check. Accepted responses consume only the original request's CPU
sampling state. Expired responses and unpublished checkpoints follow the
existing settlement and failure paths.

## Fixed geometry

After scheduled immutable checkpoints are published, the collector binds the
admitted shapes against current learner and inference residency. The batch
axis is `min(game_concurrency, max_batch_size)`, including dummy rows. Either
axis for event and candidate data is the smallest admissible power-of-two
bucket for the request's length, capped at the checkpoint's configured limit.
Zero events use a separate bucket. Neither candidate nor event padding uses
lengths of other requests. Each group shares a checkpoint and a bucket. Dummy
input is constant with a valid candidate sentinel. Actual candidates retain
their order, and padding is masked in row-local normalization.

The service packs native operands into contiguous typed inputs. PyTorch uses
existing frame, candidate and critic operations. The inference-specific event
GRU sees a fixed dense shape and selects each row's last valid output. It
never sorts or packs by the other rows' true lengths. CPU sampling uses only
that request's valid prefix, and replies contain its ordered logs and value.
The existing CPU, PPO update and frozen-reference kernels continue to operate
under their established contracts.

## CUDA admission and cleanup

The per-collection shape decision uses the existing device account's
observed learner and immutable cache storage plus its single host context and
workspace charge. It adds the overlapping fixed-width CUDA inputs,
intermediates and outputs for one group. The checkpoint's model and binding
budgets also bound its own resident inference network, fixed context and
workspace, and group storage. The largest individual allocation, including
the input banks and candidate gather, must fit the checkpoint's per-tensor
limit. These are modeled live tensor bytes. Torch allocated and reserved
peaks are separate observations. A group outside any limit runs entirely
through the existing scalar path before any batch transfer. The collection
phase keeps residency and shape decisions fixed. A successful operation
clears device work inside the serialized Torch
scope. A partial transfer or backend failure keeps its live inputs and
traceback in the existing poisoned account until the original worker stops
and a CUDA fence succeeds.

Use the existing diagnostic collection scope for useful and dummy batch
rows, scalar rows and allocator peaks. Take response p95 from the native
roundtrip histogram. Diagnostics provide observations independent of serving,
training, or Source authority.

## Alternatives and verification

Densely padding to the largest request present saves bucket partitions but
lets private lengths and arrivals change kernel geometry. Starting a Python
queue would duplicate Rust scheduling. Scalar serving remains the reference
and the capacity fallback. Fixed buckets pay extra copies, computation and
kernel launches for dummies and gather operations. Measure this cost against
reduced transfer and forward invocations, including groups with one useful
row.

Check scalar parity within `atol=rtol=1e-5` and bit-exact CUDA independence for
both MLPs across values, candidate and event lengths, batch order, position and
valid row count. Check state, old checkpoints, stale responses, pre-transfer
fallback, transfer faults, original/replay comparison and pause/resume. The
approved performance comparison uses three interleaved trials per recipe per
model, seed 17, 256 games, C16 and a 1 ms formation wait on the RTX 5070 Ti.
Require lower median collection time, no p95 increase and capacity compliance
for both models. The scalar recipe remains the default.
