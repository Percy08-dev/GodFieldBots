# Rust-owned training inputs for Model3 and Model4 MLP PPO

## Scope and contracts

The implementation branch starts at `aab5f35`. It covers the existing ordinary
Model3 MLP and HPO Model4 MLP Rust game routes. The applicable simulator is
`072dabbca1458c9526897781d2fd64edeeb4f9ca`, external interfaces §21.1 and the
API v2 supersession in §47. The policy input remains the same-revision acting
view, its own observed event prefix, typed action metadata, authorized atomic
hints, and the pinned public catalog and rules. Learner seat comes explicitly
from the frozen schedule. Equality of checkpoint hashes across seats confers
no data authority. Only the selected learner seat's accepted rows enter PPO.
Terminal outcome is used for reward after collection, outside inference.

## Live producer and writer

The native game task derives the learner token operands while it owns the
validated frame and generated table, with the existing corrected token
kernels shared with serving. Store the canonical numeric representation needed
by the existing rollout schema alongside the minimal serving-only projections
needed through inference. A speculative learner row stays local until Act is
accepted. Append its native selection, FP32 old logprob/value, accepted
revision and routing/seat/checkpoint/frame/table identity in learner decision
order. Keep per-game columns with variable candidate offsets. The existing
writer independently reconciles the accepted capture against its actual
Journal and Replay encoding on initial custody. It checks row identities,
candidate order/hash and selection against that same submitted game, including
explicit zero rows. It issues compact correspondence facts only for terminal
accepted games whose originals were committed. No JSON token transport or
post-save deserialization is required to attach those facts to the native banks.

The collection owner checks all planned games are present exactly once in
schedule order, that the actual writer joined successfully, and that each
outcome and both Replay reconciliations were natural and complete. It verifies
the seal and binds the facts to the actual descriptor, database and collection
manifest before releasing a single-use native training handle to the Source.
The handle is separate from both the inference queue and GameReport wire
metadata. It cannot be supplied from a file. Empty Model4 collections keep
their existing typed stop, backed by the same sealed writer facts. Deadlines,
capacity, missing evidence, storage or replay failures follow existing
terminal cleanup without an optimizer step.

## Arrays and numerical ownership

Charge the native trajectory, staging and packed batch allocations to the
runtime's existing `RecordingAccount` through native RAII reservations. The
controller's existing host admission already sums the storage (recording) and
Source workspace allowances. The storage account remains live through Source
and update, sharing the same atomic ledger as native envelopes and canonical
writer bytes during collection. The writer's Source workspace grant continues
to fund only its workspace and retires as before. No second Source charge funds
the Rust banks. Charge retained capacity and growth peaks before allocation.
Borrowed row/tensor aliases create no new allocation charge. New packed
buffers and conversion scratch receive their own charges for actual lifetimes.
The HPO numeric allowance also caps simultaneously retained native numeric
buffers as a subordinate counter on the same reservation. Return the charge
with the last actual alias and require the existing zero-account proof at host
retirement. A failed consumer retains the relevant actual owner through cleanup.
The PyO3 base owner for each read-only array (including derived views) keeps
its Rust backing and charge alive through the final consumer. Python writes
existing rollout shards from the native arrays and uses `torch.from_numpy`
within the learner; inputs are read-only under the Torch consumer contract.

The reward is draw 0, win +1 and loss -1 from the learner's perspective. Rust
uses the existing full-trajectory reverse GAE arithmetic, λ=.95 for Model3
and the effective recipe for Model4. Model4 obtains the existing PCG64
selection from Python using validated per-game row counts. Rust computes
selected normalization in schedule order with FP64 two-pass population
statistics and 1e-8. Preserve full targets and the schema's zero advantage for
omitted Model4 decisions, with FP32 stored arrays. Keep existing shuffling,
logical batches, CUDA group planning, PPO losses, Adam and RNG ownership.
Pack padding in Rust for the existing CUDA group's actual shape; scalar CPU
and GPU width-one operations use individual rows. Preserve the arrays, names,
dtypes, padding semantics and raw-field serialization expected by existing
rollout readers.

Check native singleton capacity before advancing the shuffle or starting
Adam. The existing CUDA grouping admits the exact native shape through the
same ledger, refining a group when the real packed host peak cannot fit.
The sizing check uses the same geometry as allocation without materializing
the pads. Staged CUDA execution keeps the candidate slab through one frame
chunk transfer at a time. Deadline and actual allocation errors use the
existing terminal failure path. The private native bridge is trusted within
the collection coordinator and has no independent Source authority.

## Saved evidence and read policy

Pin distinct `rust-ppo-training-v1` and `sealed-pack-sha256-v1` identifiers in
the optional closed effective `run.ppo_source` table, numeric manifest, Source
proof and audit result. Select
the mode from the saved effective snapshot on resume. An absent mode preserves
the prior config hash, resume identity, strict Original read flow, and
historical numeric format version. A recipe mismatch or unknown version fails
closed. The experimental selector allows comparing the identical workload
against the approved baseline. Default adoption needs both measured Model3
and Model4 improvements under common limits.

Use distinctly versioned outer records for new PPO numeric manifests (Model3
7, Model4 8) and Source proofs (Model3 5, Model4 8). Existing array names,
dtypes, byte layouts and checkpoint shapes are shared. Add the writer's
per-episode correspondence array to collection schemas 12/13. HPO producer
records that describe the numeric encoder use `canonical-rust` for this mode
and `canonical-python` for the historical mode. The setting is selected from
the effective run snapshot throughout the run, update, generation, registry,
parent metadata and evaluation binding, including generation 0. Closed shape
validation requires the correct mode/schema pair. Ordinary update and
generation records keep their outer shapes and pin the mode through their
Source proof or effective config identity. Strict readers dispatch new
numeric manifests through the existing array decoder with the new metadata
contract and toleranced comparison. Older producer/checkpoint pinning on resume
still uses the original resume identity checks.

For this mode, the live Source and subsequent cold completion, resume and
evaluation import hash the descriptor and actual database bytes within the
existing ownership/deadline envelope. Reuse secure `OriginalFile` inode and
stat rechecks and descriptor validation. Explicitly hash the DB FD, with
identity checks immediately before and after; a cached expected checksum is
not an observation. Hold the live verified descriptor/DB handles through
optimizer admission and recheck their identities. Bind `database_sha256`, `seal_id`,
logical inventory, planned batch/generation, manifest hash, learner row
counts, selection, parent checkpoint identity, numeric inventory and writer
correspondence through the existing committed update/generation/control chain.
The initial writer compares both accepted Replay action lists and natural
closures with the original Journal transaction before dropping its graphs.
The pre-seal manifest stores the writer's episode/learner/checkpoint/row and
logical member identities in schedule order. The post-seal Source proof stores
the descriptor and observed database hashes, the writer seal's logical hash,
and the pre-seal manifest hash. The descriptor's existing `manifest.json`
control entry is checked against that manifest hash. There is no digest
dependency from the manifest back to the descriptor or database.
Take the already computed logical hash from the concrete seal's result. Keep
its extra handoff private to the same-process writer. For a healthy-empty
Model4 collection the mode-specific empty marker carries that same post-seal
witness, counts and natural writer correspondence without numeric/output or
update authority. Cold completion verifies that marker with the same hash-only
reader. Historical audit validators select the compact audited-result shape
from the saved mode as well as accepting their existing strict-result shape.
The new evidence records observed hashes explicitly. Reopen the expected
database in its expected directory, without SQLite or packed member decoding
on the normal hot/cold path. Preserve saved state/lineage/file verification,
including relevant checkpoint weights/Adam/RNG reads. A new receipt cannot
turn a legacy bundle into a new-mode bundle.

`tools/rust_player_verify_operation.py --full-source-audit` runs the existing
strict Journal/Replay/rollout readback even for new runs, reconstructing Python
reference values and checking exact structure and the approved floating-point
tolerances. Its default for new runs uses only committed SHA-256 verification
and compact saved facts for collection summaries. Legacy verification retains
its current diagnostics and exact comparison. Explicit complete readback is
DATA verification and creates no live Source authority.

## Verification and implementation order

Start with the writer/Source and saved-chain contract and native trajectory
interface review, then implement native collection and PPO operand export,
writer/Source and optimizer integration, and the saved-loader/verifier mode
dispatch. Keep one code-writing owner and reuse the original strict readers
in diagnostics. Compare exact identity, order, selection and feature operands,
and compare GAE, normalized advantage and targets at 1e-6 absolute and relative
tolerances. Compare first PPO loss at 1e-5, covering logical widths 64/128,
Model4 partial-budget selection, both seats using one checkpoint, and partial
batches. Exercise a new-mode continuous and paused/resumed pair with a real
second update through Adam steps, weights, optimizer arrays and RNG identity.
Instrument normal Source,
completion, resume, and evaluation to detect any decoded Pack/rollout or
Python GAE use. Test changed and swapped databases and explicit full readback.

Use three baseline and three new trials per route at the same seed 17, initial
weights, Simulator and resource limits. For Model3 use 256 games, C16/W16,
inference cap 64 with 1 ms formation, GPU learning width 16. For Model4 use
the 16-game/64-decision real HPO route including resume and evaluation import.
Record collection, preparation, PPO, completion, resume, median whole wall,
peak host RAM and device VRAM, and any unavailable measurements.
