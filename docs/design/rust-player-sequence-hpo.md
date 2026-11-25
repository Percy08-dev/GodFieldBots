# Model4 sequence collection in the Rust player runtime

The approved plan prioritizes GIL contention and request supply, then adds
GRU and Transformer to CUDA HPO6. Run the common-path comparisons independently
of each model's migration and batching comparison. Model3 and the existing
MLP recipes retain their dispatch and Source choices.

## Measure the shared path first

Use the pinned Sacrifice action profile, Simulator `6d50603dab3f6d2df34ea4e433bab7dca96f52c1`,
Gameplay API v2, public event and catalog schema 2, Journal 6, and candidate
codec 2. Use actual CUDA and natural seed-17 games. Keep diagnostic builds
and their added timing costs separate from timed adoption builds.

Correlate CPython GIL acquire/release native thread IDs and operation spans
against the inference worker's GIL condition waits during the first ten
seconds after first enqueue. Account separately for wall time and time
weighted by FIFO depth. For supply, mark completed inference, pending writer
receipt and Act/ACK, observation and event/options fetch, candidate search,
numeric packing and enqueue in the original actor. Intersect each phase with
FIFO-empty intervals and count completions. Preserve pending receipt before
Act and the existing same-revision observation/options/input checks.

Test one measured change at a time. The existing `5a601b3` Rust detach of
writer frame validation is a candidate with prior matched throughput evidence.
Its original Rust storage owner validates and materializes the canonical
frame while detached, and the same Python writer publishes and acknowledges
it. Choose supply changes from phase attribution. Screen the formation
window, large MLP Graph shapes and GPU transfers separately after these
comparisons.

## HPO configuration and recorded data

The only new selector is `batched-torch-sequence-v1` on CUDA HPO6 for Model4
GRU and Transformer, with shared CardInstance rules and `canonical-python`.
It requires a matching Model3 experiment base and rejects Rust native PPO
Source for sequence models. Ordinary Model3 and older HPO schemas retain
their existing selectors and registry restrictions. Existing recipe omission
continues to select scalar MLP. The new recipe is bound by the existing
effective config, runtime binding, and producer fingerprints. A paused run's
exact saved runtime and producer must be present to resume it.

Game policy descriptors keep the actual model kind, checkpoint, epoch, and
caps. The Rust actor handles all three served kinds through its existing
inference policy and queue. The Python runtime checks descriptor kind and
limits against the immutable publication before admission. The writer,
publication, service usage, collection evidence, and cold readers identify
learned policies by the same set of served kinds. Model4 base, campaign,
launch, effective config, Run5 and Update8 checks bind the actual model.
New sequence HPO runs emit the existing canonical Journal 6 / Collection 11
and Numeric6 layouts. RustSource opens the sealed writer's original, and
the existing Python `encode_rust_batch` and `LocalHpoLearner` build and train
from own-decision histories. The saved optimizer, weights and RNG follow the
existing HPO update boundary. Inference caches start empty at each collection.

## Request, cache and retirement ownership

Rust exports its existing numeric frame/candidate operands and request
identity to the trusted Python service. The sequence export adds the raw own
event interval start and end from the same validated Rust input. The encoded
interval length can be smaller when the contract omits concealed draws. No
other view, canonical history, rejections, or Simulator RNG enters a row.
The transport remains confined to the existing authenticated host.

Keep one Rust dispatcher and one service loop. The sequence mode exposes an
additive service operation that returns work or retirement notices. It also
exposes ordered per-row submission results. It validates the original
entire batch before the first acceptance, and Rust commits each accepted
row's sampling state in the same operation. The existing MLP count and
formation path keep their interfaces and avoid sequence bookkeeping. Python
uses the same CPU Torch sampling routine with Rust-owned RNG bytes.

Python keys a cache by run, trial, game, seat, epoch and checkpoint. Check
own request position and revision advance and raw event interval continuity.
Use the structured network's existing frame/candidate operations and factor
its validated temporal step math for native numeric rows. Test scalar
continuation before adding batching. GRU returns provisional hidden state.
Transformer reads the committed prefix and writes only the provisional next
slot. Python advances length and publishes hidden or boundary metadata only
for accepted rows before serving the next work item. The actor never waits
on a Python result belonging to another game. On any service error, fail the
collection and retain device owners until original producer join and GPU
settlement. Graph and legacy shared TorchRL routes keep MLP-only serving.

Production game descriptors pin each seat's epoch and checkpoint. After
both seats retire and that game's producers and exported rows settle, Rust
queues one retirement ticket for the exact run, trial, and game. The
dispatcher wakes the service for retirement-only work when the FIFO is empty.
Python acknowledges the one-use ticket only after the device work completes
and all caches for that game are dropped. A slot remains occupied until Rust
receives this acknowledgement. Only the sequence-enabled dispatcher accepts
sequence descriptors. Rust releases its mutex before waiting or calling the
service. The service submits the acknowledgement without the Python runtime's
admission lock. The same service thread owns cache mutation even when an
accepted response lets an actor begin to prepare its next request.

On sequence release timeout or error, signal the native scheduler's existing
cancellation before the actor can release its slot. This stops both submitted
and later game admissions while the actor records the abnormal terminal result.
Cancellation wakes the dispatcher and the service. For service failure, retain
checkpoint references and capacity charges until both original owners join,
the device is fenced, and failed provisional and committed caches are dropped.
A failed fence leaves those owners held. The health evidence grants positive
game release only after acknowledgement.

## Batching and resource accounting

Use the configured fixed width and existing candidate and event shape
buckets. GRU groups by immutable checkpoint and shape, and Transformer
additionally by each row's committed position. Fixed deterministic padding
and row-local normalization preserve results under changes to neighbors.
Gather only the Transformer prefix needed for the group's position. Copy
back only its new K/V slot. Never replace, change, or extend a committed
prefix for an unaccepted row. GRU batch hidden rows get independent backing
when retained so a single row does not keep a retired group alive.

Extend the same-host account to charge temporal storage separately from
checkpoint weights, and temporary group storage separately from both. Base
the collection reservation on the public scheduled sequence seats that can
overlap, bounded by twice the configured game concurrency and the scheduled
seat count, and the largest published per-seat temporal geometry. Reserve a
Transformer bank of two FP32 tensors of layers × decisions × hidden width or
one GRU hidden stack per live seat. Use the existing scalar actor estimate as
the scalar budget baseline. For groups account for fixed-width frame and
candidate intermediates, admitted Transformer prefix and provisional outputs,
and overlapping old and new GRU state. Keep the existing model and tensor
limits. Freeze scalar or group eligibility against the funded public envelope
before actor admission. Select scalar serving for shapes beyond group allowance
and reject a collection that cannot fit scalar serving. GPU settlement restores
the prior cache budget.

## Checks and adoption

For each model compare exact candidate rows and selected indices/actions on
fixed fixtures. Use `atol=rtol=1e-5` for scalar and batch FP32 comparisons,
and bitwise equality for same-recipe neighbor mutations to the outputs,
accepted RNG and committed state. Verify own history continuation, neighbors with other
values/lengths/order, mixed acceptance within a batch, epoch/game/seat/trial
isolation, checkpoint change with in-flight replies, and retirement without
leaked GPU storage. Exercise an idle final retirement, a failed fence, and
replacement games queued before and after a retirement error. Use a real
16-game update and pause/resume, then attempt
64, 128 and 256 concurrent games with the same seed, initial weights,
Simulator, and resource budgets. Report actual overlap and any admission
limit. For each safe 256-game path, run three alternating matched pairs.
Include preparation through sealed persistent collection in actions/s and
report early p95 inference and GIL waits, later empty FIFO wall, and CUDA
allocated/reserved peaks. Adopt an optimization when median actions/s gain
exceeds the control range and p95 and memory remain within the verified
operating envelope. Independent correctness and architecture review precede
enabling each model.

## Design review

The architecture reviewer cleared the GRU-first implementation on
2026-11-26 after confirming the original game retirement barrier, failure
cancellation before slot reuse, and retained checkpoint and device custody
through positive producer join and GPU settlement. The same review accepted
the fixed public capacity envelope and scalar parity and neighbor
independence criteria. Implementation conformance and CUDA evidence are
recorded separately in the verification record.
