# Investigation: structurally owned HPO producer data

Status: research/proposal,2026-11-16; not an accepted implementation design.
User requests considering structural correctness instead of validating internally
created data. Required results presentation has TWOlevels: whole-program phases
and detailed collection, using the latest accepted resident baseline.
No application code, dependencies, Simulator, data formats or GPU workloads were
changed/executed in this investigation. Existing ADR0042 acceptance remains intact.

## Requirement and baseline

Correct model/generation/participant/data correspondence should follow from a
single construction/ownership path. Do not implement another cache, a validation
skip flag, or a new factory that merely hides the same postcreation audits.
The normal fresh path should consume its own produced data, not reopen archives
to prove that its own producer worked. External decoding, resource admission,
I/O/process completion and numerical failure handling are separate responsibilities.

Baseline implementation `fd77c86`, accepted evidence `00a693e` and
[verification](../verification/hpo-precollection-validation.md). These already
include ADR0042 reuse; new improvements must compare against this baseline,
not the older unoptimized119.43s MLP arm.

| Baseline metric (s) | MLP | GRU | Transformer |
| --- | ---: | ---: | ---: |
| Whole2updates | 107.76 | 146.94 | 149.04 |
| Collection2batches total | 48.35 | 48.59 | 49.31 |
| Source/data preparation | 19.02 | 19.24 | 19.35 |
| Update including runtime parity | 17.04 | 54.62 | 54.95 |
| Save | 2.55 | 2.71 | 2.89 |
| SYNC | 9.90 | 10.53 | 10.92 |
| Cleanup | 9.16 | 9.50 | 9.83 |
| Startup/other | 1.88 | 1.75 | 1.80 |

MLP is3trial medians; others1trial per arm. Component medians need not add to
the independently reported whole median. These are historical baseline facts,
not new-candidate predictions or measurements.

## Current construction and repeated work

At `card_instance_environment._select_and_journal` (lines335–380), the parent
already has the acting participant's DecisionInput, EventDelta/EventBank,
DecisionFrame, generated CandidateTable, Selection/logprob/value and command.
They are journaled together, but transient objects are discarded. A full typed
episode/learner batch is not retained. `PackedFinalizedCollection` contains
archive/report/locator information, not learner input.

Current flow:

1. Capture per-decision objects and serialize journals/candidate columns.
2. Collection finalization rereads player journals/server replay, reconciles them
   and archives packed originals.
3. Source reopens original collections and reconstructs/replays learner records;
   parent validation also revisits committed historical batches.
4. Construct/tokenize rollout, compute selection/GAE, write numeric files, read
   and validate them; the numerical worker reads/checks them again.
5. `HpoLearner.update` first invokes whole-batch parity before optimizer work.
   That builds a CPU network, copies weights, and compares saved/cache/CPU-full/
   GPU-full/differentiable-suffix outputs over every original endpoint.
6. Commit/save/reference paths reread artifacts/control lineage; some are
   repeated internal correctness checks, others are real external reads or
   actual I/O/ownership operations. They must be distinguished before removal.

Relevant references: `card_instance.py:134–180,383–434`,
`group_pack_collection.py:332–484`, `hpo_ppo.py:271–310`,
`native_card_instance.py:889–981`, `structured_numeric_worker.py:315–381`,
`structured_numeric.py:119–156`, `hpo_numeric.py:89–158`,
`hpo_lineage.py:1003–1090` and resident runtime acquisition/release/SYNC methods.
The numerical worker currently requires parity metrics to claim all checks ran;
removing the call cannot truthfully leave an all-zero "passed" metric.

Current inference candidate tables are not already-tokenized training buffers.
The parent owns reusable typed inputs; actor inference tokenizes its own copy
in a clean worker process. Training still needs original own-decision prefixes
and prescribed differentiable suffixes for GRU/Transformer. Reusing a recurrent
hidden/KV cache as training data would change learning semantics and is excluded.

## Approaches

### A: Owned producer batch, existing representations (recommended)

- Capture the existing per-decision objects through the actual collection owner.
  A pending decision becomes accepted only through its actual engine response;
  a completed episode comes from its terminal response. Episode/participant/
  model/generation assignments come from the same immutable run context.
- A private builder owns accepted records and final outcomes; a seal consumes
  the mutable builder and hands an immutable batch to the learner. Counts,
  selected endpoints, provenance, numeric views and archive records are derived
  from that same graph rather than independent dictionaries to compare later.
- Reuse canonical Python encoders, NumPy banks, GAE/normalization and endpoint
  machinery. MLP/GRU/Transformer are feasible without new Rust implementation;
  public dataclasses with checks disabled are not a trusted construction path.
- Keep bounded buffers or an actual owned typed spool when a whole-batch graph
  exceeds memory. Reading/mapping owned data to compute is consumption, not an
  audit of data just produced. Explicitly account copies and cross-process leases.
- Archive/numeric/checkpoint representations remain outputs where possible.
  Do not require archive->Source or numeric write->readback before trusting
  fresh output. File writing, acknowledgments, durability and errors remain.
- Move full replay and implementation-equivalence parity to tests/explicit
  offline audit operations. Retain normal loss/backprop/Adam, configured sampling,
  necessary resource/finite-result failure handling and actual process cleanup.
- Bind worker input and save outputs to actual immutable snapshots/requests,
  rather than independently recreating an expected dataset and saved pair.

This is a new live construction/ownership path. OriginalArtifactProvider/native
Source authorization is an archive-reader authority, not a producer capability.
The existing native packet layout and payload/GAE/writer components can be reused
as representations/calculations, not as a claim that original replay occurred.

### B: New streaming/shared-memory transfer design

Pass actor-produced token banks through a new streaming/shared-memory protocol,
unifying inference capture and learner transfer more aggressively. This can
reduce additional copies/encoding, but expands IPC, backpressure, buffer alias,
failure and persisted-format work. Choose it only if A's measured construction/
transfer cost warrants that extra change. Neither approach copies/modifies the
Simulator or adds hidden policy inputs.

## Decisions needed in the implementation design

Root proposes these defaults, not yet an accepted implementation specification:

- Initially scope the new route to the current fresh resident HPO training loop,
  covering collection, Source, update and internally produced save/SYNC artifacts.
  Keep existing ordinary/legacy and external/audit readers distinct.
- Preserve durable archive/checkpoint completion before healthy batch/update
  publication; remove semantic readback, not persistence or actual I/O failure.
- Cold resume/import restores owned state from the required checkpoint/control
  boundary once. Do not replay all historical samples merely to rebuild a live
  owner if they are not consumed by the resumed update.
- Give producer-origin live authority and omitted runtime parity truthful
  explicit protocol/provenance meanings. Keep old SourceProof3/4 and parity
  metric meanings exact; introduce a new kind/version wherever required.
  Physical array/journal layouts need not all change.
- Keep resource/job/admission/ACK/pin/retirement controls. Replace repeated
  inspection of internally generated contents with owned immutable state where
  sufficient; external artifact loads still decode their boundary representation.

Before code, resolve exact builder/accepted-decision/complete-episode/batch
interfaces, actual actor-model snapshot binding, seal and alias rules, canonical
selection/GAE construction, memory/spool/IPC ownership, save ACK/commit ordering,
truthful persisted/protocol compatibility and cold-resume semantics. Independent
architecture review is mandatory; this research is not that design approval.

## Required two-level comparison

Use the same192/main2,seed17,4workers/2concurrentgames,16games per collection,
2updates and original budgets/timers. Keep fresh targets and pin code/native/
Simulator/schema/source-mode identities. If baseline and candidate need different
protocol variants, record that honestly; do not fake a same-format equivalence.

### Level1: Whole execution

Show before/after startup, collection(two batches), feature/data preparation,
optimizer update, save, SYNC, cleanup, other and whole wall time. Separately
time the old runtime parity and any new construction/transfer costs. Do not
claim the entire current19s Source or55s recurrent update is removable: encoding,
GAE, tensor transfer and optimizer work remain. State medians/sample counts and
nonadditivity of independently summarized components.

### Level2: Detailed collection

Retain mapping to the five previously measured check positions so removed work
is visible, while adding new producer costs explicitly:

| MLP second collection baseline (s) | Current |
| --- | ---: |
| Before collector | 5.54 |
| Collector entry | 0.04 |
| Before output creation | 0.07 |
| Owned archive path setup | 5.50 |
| After games | 5.47 |
| Remaining game/inference/startup/communication/storage | 13.45 |
| Whole second collection | 30.08 |

Future report must include actual call counts or removed-stage observations,
producer capture/tokenization/seal time, game/candidate/inference/transport and
archive queue/write/seal costs where separately measured. Do not call the
remaining13.45s pure gameplay. A change that shifts work between collection and
Source must remain visible in BOTHlevels. Concurrent worker/inference/storage
spans are inclusive and cannot simply sum to collection wall time; report
exclusive parent spans or clearly labeled per-operation distributions.

Prefer thin bounded timing that preserves original invocation/exception and
restores instrumentation; no global profiler like the historical intrusive run.
Use MLP3pairs and at least1pair each GRU/Transformer for continuity, with all
valid trials retained and no shortened budgets. New-candidate results are NOT RUN.

## Verification of structural guarantees

Test the generator/builder and external readers, rather than running the same
full validation inside every batch: model/actor/generation mismatch cannot be
constructed through the supported route; pending/rejected/retried commands do
not become accepted transitions; terminal outcomes derive from the actual
engine result; sealing prevents mutable aliases; counts/views share one source;
duplicate/stale update and ACK cannot commit; partial writes/cancellation keep
actual owners and fail without invented results.

Offline all-model comparisons must match the previous strict reader on fixed
original captures for features, candidate order, selected endpoints, GAE/targets
and gradients, while preserving the8fixed numerical kernels. Memory, process,
buffer/spool and I/O cleanup remain bounded and independently reviewed. Existing
datasets/checkpoints remain protected and are never relabeled as newly verified.

Research: root + bounded read-only `bot_verifier` and
`bot_architecture_reviewer` investigations. No tests/spike/GPU work was run.
