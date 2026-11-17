# Standard bounded Model4 PPO updates

## Authorization and scope

On2026-11-17 the user authorized the previously proposed adoption: normal Model4
MLP updates use microbatch execution; GRU/Transformer use shared-prefix execution
(including ordinary microbatches for endpoints that cannot share). The old HPO
loop remains an explicit private reference for tests/benchmarks, not a normal
configuration choice. Use preplanned finer packing/frame execution for avoidable
batch/padding expansion; no memory-cap increase or OOM-triggered fallback.
The user's later explicit steering accepts OOM/capacity risk for extreme
settings. Do not promise every previously admitted extreme configuration fits.
Keep configured ranges (including TBPTT up to1024) without guaranteeing all
combinations; defaultTBPTT32/common settings are the primary acceptance scope.

This supersedes only ADR0045's opt-in/default decision. Its a0f243f timing and
f067d03 telemetry populations remain historical and immutable. Existing frozen
Model1–3 learners/kernels and inference/evaluation behavior stay unchanged. No
new dependency, Simulator change, public config key, checkpoint/data format or
Wire6/7 field. Model4 fresh/resume/campaign/native-Source updates all converge on
the same learner; normal execution must not carry the experimental fresh-only
capability or lose normal completion/objective authority.

## Approaches and decision

The user approved automatic selection plus practical bounded subdivision, with
a retained internal reference, then explicitly accepted extreme-setting OOM
risk. Raising limits or falling back to the old loop would not meet that choice.
Removing the reference would weaken existing frozen/numerical
checks. Select a fast admitted packed group when possible; otherwise construct
a bounded streamed tensor plan before allocation. These are storage/execution
plans inside the same optimized trainer, never runtime failure recovery.

The demonstrated gap is dimensional: one1024-frame GRU endpoint with hand
feedforward1024 passes the original endpoint estimate(589,436,656B in the small
CPU fixture), while the current batched FF output would be75,497,472B/72MiB.
Current joint admission rejects it at64MiB. The probe allocated no large
activation and used repeated shapes; real behavioral tests must use valid tokens.
Ragged event padding and combined dtype slabs can cause analogous expansion.

## Responsibilities and dispatch

`HpoLearner` remains the sole Model4 trainer and owns PPO/shuffle/logical128(or
configured size)/TBPTT/clip/Adam/reward semantics. With no private execution spec,
derive microbatch for mlp and shared-prefix for gru/transformer. No normal CLI or
model/training setting chooses baseline. An explicit private baseline spec still
selects the reference loop for comparison. Preserve experimental spec modes and
their fresh-only restrictions; ordinary implicit optimization is not experimental
and is valid on normal fresh/resume/campaign paths.

Rename/extract private method names as needed to reflect production use without
duplicating the trainer. Keep hpo_losses and existing per-endpoint finite checks,
FP32 losses/gradients, detachedFP64 metric sums, exact endpoint selection/PCG64
shuffle/logical membership/weight m/N, one zero-grad/clip/Adam per logical batch,
current-weight prefix no_grad and suffix autograd. Old exact reference tests must
select the reference explicitly; do not turn them into equality gates for newly
ordered optimized Adam trajectories or silently delete assertions.

`gf_models.structured_learning` owns immutable raw operand packing and private
window/chunk descriptions; `structured_network` owns all shared math. Training
planning imports model-side shape/window helpers, never the reverse. Extend the
existing window mechanism with a prevalidated streamed representation, not a
second model/encoder implementation. Both representations invoke the same frame,
temporal and candidate arithmetic. No inference-buffer reuse in training.

## Bounded execution plan

Plan using public/owned token dimensions and original liveness checks, before
Torch operand allocation. Keep existing exact prefix/suffix bins, stable candidate
ordering and at most16endpoints/padded candidate rows<=2xreal rows. Capacity-only
splits are distinguished from malformed/nonfinite/dead input errors; never catch
an arbitrary learning/Source error and retry a different execution strategy.

For an admitted ordinary packed group keep its existing single packing/forward
fast path. This preserves the common short-input batching benefit. If that
expanded representation cannot fit, split endpoints as appropriate and plan
frame windows within a remaining endpoint, down to one frame if necessary.
Prefix and suffix frame plans are separate and never cross the TBPTT cut.
Suffix graphs remain live through backward: subdivision is not a promise to
reduce retained full-BPTT memory. If their conservative envelope does not fit,
reject before allocation rather than add rematerialization or manual VJPs.

The streamed window packs candidate operands/targets independently of frame
operands. It encodes admitted frame chunks and restores chronological row order
before the existing full temporal prefix/suffix calls. Independent frame
encodings may be grouped by real event lengths to avoid large padding, provided
outputs/cards are scattered back to their original positions. Every positive
event sequence keeps its actual length; empty intervals never run dummy GRUs.
If an aggregate dtype slab alone is oversized, split its storage at operand
boundaries; do not make a tensor larger than64MiB simply to reduce copy count.

Prefix frame inputs are immutable until their last no_grad encode, then retire
before the next chunk while small projected prefix outputs remain. Recompute the
full temporal prefix under current weights as before. Suffix/shared differentiable
frame inputs and saved views stay immutable through final backward/checkpoint
recomputation; count every retained chunk. Do not detach between frame chunks.
Keep exact GRU hidden propagation, Transformer absolute positions/causal masks,
critic independence and each endpoint's own actor-card representation. Shared
history still requires same actual origin/episode/viewer/cut0; other cases remain
ordinary optimized microbatches. No graph crosses group/Adam boundaries.

Candidate scoring retains the configured bounded chunk/checkpoint policy and
one normalization over all real candidates per endpoint. Padded rows never
enter selection, entropy or loss. Smaller staging storage may not reduce learning
targets, candidate population, history length or TBPTT. grad=None and Adam
state-presence/parameter steps remain correct for empty branches.

## Memory accounting and compatibility criterion

Use one execution-plan estimate for planning and live admission. Count parameters/
optimizer/context once, Source/rollout residency under its existing owner, retained
projected temporal inputs/KV, all gradient-bearing frame graphs/input CPU+GPU
chunks, candidate/target/mask storage and checkpoint/backward coexistence. Count
only the maximum simultaneously live temporary prefix/frame window, not the sum
of retired no_grad windows. The original conservative factor4 remains for live
activation/copy envelopes. Enumerate actual intermediate dtypes and all layer
widths, including int64 one_hot and FF/gather/attention, for each effective tensor.

Do not keep a whole-history padded slab behind the streamed representation.
Raw staging/copies belong within the conservative existing envelope, not an
unbounded extra bank. Refine shape/lifetime accounting rather than reject because
the obsolete whole-frame estimate exceeds a cap. The finest plan uses one
frame's shared math plus bounded candidate staging, while still counting all
retained suffix graphs. No guarantee is made for extreme retained-activation
combinations merely because a legacy estimate admitted them. Legacy estimator
blind spots never permit a tensor to exceed its declared cap. If even the
finest conservative plan cannot fit the original owner allowance, fail before
allocation/mutation; never retry after OOM. The user's accepted extreme-setting
risk removes the earlier universal-compatibility requirement, not safety checks.

Keep64MiB/tensor, current model/binding limits and actual process/phase caps.
Verification must include formerly accepted singleton geometries, not only
endpoint-group splitting. Cover normal1024history with defaultTBPTT32 and
practical event/candidate shapes, plus honest preallocation rejection of extreme
full1024-gradient/wide-FF cases. Configured syntax/ranges remain unchanged.
No precision/backend/compile/activation-checkpoint-policy change is allowed.

## Exact normal worker reservation

Direct component HpoLearner construction derives its configured process envelope
from the existing resource_estimate(model,binding,suffix=training.tbptt). Real
workers override/bind it to the actual owner-provided allowance before update.
GPU allowance comes from the SAME active LearnerReservation. CPU's GPU lease is
correctly zero; carry a separate positive CPU execution-envelope amount derived
and checked from the same model/binding/training, not a fictitious GPU lease.

For Wire6, add one private launch-argument hook to the existing LearningProcess
creator(default empty). HpoLearningProcess supplies bounded labeled reservation
arguments from its original admitted lease; other/frozen processes keep identical
argv. The creator/module/lifecycle and failure observers remain original.
ppo_worker validates/scrubs launch args and passes the allowance only to the HPO
branch of structured_numeric_worker. Non-HPO requests with those args reject.
For Wire7 the original owned creator always passes the labeled allowance, with
private spec path/SHA still optional. Validate closed args/caps before allocation;
no new Wire6/7 request/reply fields or environment-global patch.

## Persistence and failures

Source code identity distinguishes new optimized runs. Keep ordinary strict
source-match resume: old paused Model4 parents need their old revision; do not
rewrite producer hashes, silently switch their execution or loosen restore.
New-source pause/resume uses the same automatic route. Formats and model
architecture identities retain meaning. Keep earlier experimental namespaces
and their ineligibility/records intact; use a new adoption verification root.

Original Source single-use state, budgets, deadlines, FD retirement, snapshots,
save ordering and positive process/GPU receipts remain authoritative. All new
window references end after real last use before the existing retirement ACK.
Capacity, nonfinite, CUDA, cancellation and I/O failures retain original terminal
cleanup, without successful ACK/save or guessed outcomes.

## Verification and acceptance

TDD regressions: implicit correct route with no experiment flag; explicit private
reference; normal CPU/CUDA Wire6/Wire7 cap transport and unchanged older workers;
valid1024-frame/default32-gradient singleton and ragged-event staging; extreme
retained1024-gradient/wide-FF preallocation rejection; effective slab/
intermediate bounds and retained-vs-transient peak; same-input loss/gradient using
existing tolerances, prefix detach, own cards/causality/empty branches/Adam presence,
tails53/5/47/7, checkpoint input lifetime and preallocation/retirement failures.
Keep old exact reference/frozen-kernel assertions explicitly routed to reference.

Run affected numerical/network/owned/legacy worker/resume/campaign/native-source
checks and scoped Ruff/mypy; no dependency/native rebuild unless an actual need
appears. Independent correctness and final conformance reviews follow stable code.
Then bounded real CUDA normal fresh/pause-resume/Source audits for all3models,
without private experiment selection; retain full configured budgets and original
deadlines. Existing source-mismatch negative checks remain. Historical84/18 timing
is not reinterpreted as new-source evidence: use a small unprofiled fixed-input
screen to check common-path regression only, no repeated HPO/quality experiment.
Document exact commands/results/limits and commit verified milestones locally;
no push. Learning quality, formal HPO and final test remain separate work.
