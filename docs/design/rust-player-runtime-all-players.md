# Full live Player migration to the Rust runtime

The target is the live Player boundary of the existing application. Rust owns
the game transition and per-seat input, history, candidate and command lifecycle.
Torch and LightGBM keep their existing numerical kernels. The Simulator remains
the external authority for game state and Act acceptance. Offline conversion,
training numerics, fitting, record readers and audits remain separate.

## Normative same-process isolation

Simulator main `a0c0b0583e1fae1a9617e850ef15384f2d2b3699` carries the same
section 21.1 text as the supported shared `072dabb` runtime. The specification
file hash is `8e5223b45a6e8c4c80e272736ed97b9e271136c939eca74427a035ddfdbfcac2`.
Same-process serving is permitted under its logical isolation requirements.
Keep each seat's visible inputs and temporal state separate. Match actual
run or trial, game, Participant, request, revision and checkpoint before use.
Retire expired responses and states. Per-seat and processing-order isolation
tests are part of any new shared path's acceptance. The executable Simulator
pin and source provenance remain explicit for every run. Historical records
keep their recorded protocol and contract.

## Live routes to cover

1. Shared corrected v2 (`072dabb`) standalone Random, Heuristic and Refresh
   smoke, collection and evaluation. Add trained MLP inference through actual
   read-only policy owners. Reuse one Rust scheduler and managed match pump
   per invocation.
2. Ordinary and HPO training, formal validation and Fresh evaluation on the
   corrected v2 contracts. Use explicit typed evaluation or training custody.
   Migrate genuinely running compatible collect, validate-policy and
   validate-formal-policy routes. Require exact original process, pack and
   terminal outcome evidence from each runtime role. Qualify only actual
   results of the applicable role.
3. GRU, Transformer and qualified Tree3. Retain their approved numeric
   operations and use exact per-seat, per-generation numerical request
   identities. GRU or Transformer histories and any temporal state have
   bounded ownership through reset, cancellation and model retirement.
   Tree3 preserves actual artifact custody, feature semantics and selection.
4. Other still-executable profiles. Inventory historical modes through public
   entrypoint checks and use distinct version-pinned actor adapters for live
   behavior where supported. Historical data readers and exclusive recorded
   restore bindings retain their definitions. A live route with a Python
   participant remains an open migration item. An existing rejected
   engineering-only route stays out of the live coverage count.

The first implementation gate below covers the shared execute boundary.
Standalone publication, model extension, evaluation authority and switching
live entrypoints each need concrete integration evidence and a scoped design.
The last migration check exercises every public live route and verifies which
runtime and processes actually executed. An archived dataset and an explicit
CPU/GPU policy numeric call are distinct from a live Player.

## First implementation gate

Extract the managed match pump already used by
`gf_training.rust_player_collection.RustPlayerCollector` into a small concrete
owner independent of `RustRunRegistry`. The owner borrows the same actual
`RustPlayerRuntime` and `RustPlayerStorage` through one execution and retains
the original `ManagedServer` instances and server history across collections.
It receives the frozen, validated group schedule and episode encoding
contexts, caller's exact execution and cleanup deadlines, and the existing
`CardInstanceRuntimeEvidence` prepared by the caller. Preserve the relative
preflight, acquisition and exception handling boundaries.

The pump starts each group once, binds the writer encoding contexts to the
original executed Simulator receipt, starts each native actor and moves to
the second episode using the same managed group process. Storage admission
flows through the same actual writer before acknowledgment to the actor. The
pump waits for all actors and returns ordered terminal reports, original
group/receipt evidence, accepted native transport rows and encoding context
hashes. Use the existing group concurrency and the identical episode and
request identities. Keep the same pending, terminal and original process
confirmation meanings. The caller retains the original cancellation order and
access to all server handles on error. Existing lists and maps move without
duplicating their element data or adding an allowance.

The caller continues to own writer creation, capture and retirement records,
checkpoint and policy publication, memory loans, diagnostics, Source
issuance, saved-format validation and error publication. The moved execution
loop performs no training finalization and issues no Source, parent or
evaluation capability. The first application is the existing training
caller, with its external routing and formats preserved. The non-training
owner will then consume the same pump and complete its own legitimate
standalone storage and reporting lifecycle. It will not construct a pretend
training registry or borrow a training eligibility record.

Verify the moved execute boundary with a real pinned Simulator and writer,
the existing Rust collection and cleanup tests, and the public ordinary
training entry through audit. Check exact ordered actor/receipt identities,
slot reuse, shutdown, writer settlement and regression resource behavior.
The bounded baseline suite ran the original Native loop in both Tokio and
Fixed with verified shared Simulator `072dabb`.

## Proposed standalone packed baseline integration

Route the existing public corrected `smoke`, `collect` and `evaluate` commands
for exact shared `072dabb` baseline-only packed configs to a standalone owner.
It uses the same compiled runtime, one match pump and one thread writer for the
whole run. The existing schedule and Random, Heuristic and Refresh labels and
options remain the baseline input. The native runtime uses `service=None`.
No learner, CUDA admission, training generation or evaluation authority is
constructed for this role. The existing resolved run's concurrency and resource
and cleanup deadlines are explicit. Use the current Tokio scheduler with
a bounded player thread count up to two and a one-row inference capacity for
the baseline-only route. This initially covers the packed output policy.
The file-output and other contract paths remain migration items.

The actual native encoder produces the established Journal6 and codec2
content for each own seat. Compare the matched schedule, original server
receipts and saved episode/Replay views before healthy publication. The
existing result aggregation consumes actual episode and group results and
preserves incomplete and outcome distinctions. Split and scoring semantics
stay bound to the standalone configuration. Persist explicit native actor,
producer/configuration and settlement evidence in a closed standalone role
manifest, separate from native training Collection10/11 and old evaluation
receipts. This metadata conveys no learning eligibility.

Use the actual pending descriptor and database owned by the thread writer.
The header/policy context control remains bound before each actor start.
The caller stops and reaps every original server and closes native runtime
and writer resources before publication. For a healthy result, expose the
same Pack and locator shape through real publication and file identity
checks. Prefer extracting or invoking existing owned-descriptor/locator
mechanics from the legacy publication implementation with an explicit
native standalone input. Preserve the actual recording and workspace
leases through their current last use. Never construct a fake
GroupPackCollection or legacy writer/process wrapper. Keep diagnostics on a
pending-only path after a failed or incomplete match. Leave all existing
legacy readers and training publication types exact.

The next gate must identify the concrete original PendingDescriptorOwner
handoff and existing storage verification route. Check public role dispatch,
actual absence of Python participants, natural/incomplete cases, identical
supported baseline policies, Replay agreement, seal/locator readback,
role separation, and no resource refund or healthy publication after
cleanup or writer failure.
