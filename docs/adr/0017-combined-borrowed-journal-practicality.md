# 0017: Combined mimalloc and borrowed-Journal practicality

- Status: Accepted2026-10-30; implementation, executed correctness/measurement
  gates and independent conformance/correctness reviews READY. The opt-in
  combination is practically usable for the verified workload; no default adoption.
- Owner: root. Baseline9704f93; preserve original datasets/models/main environment
  while building and verifying opt-in candidate wheels. Architecture review is
  mandatory before dependent implementation.

## Required outcome

Determine whether mimalloc0.1.52/local_dynamic_tls plus borrowed typed Journals
can be used on the actual preprocessing path: safe raw-buffer/worker ownership,
full Journal/Replay/source validation, exact feature/array identities, bounded
memory/cancellation/failure cleanup, and measured complete wall/CPU/peak/retention.
Standalone speedups are not a production verdict and cannot be multiplied.
No simulator changes, new games/fits/PPO updates or automatic preset switch.

## Staged controlled candidate

First extend the standalone harness with an opt-in mimalloc build. Compare the
2x2 cells Value/borrowed x System/mimalloc under the existing frozen input,
common checks, memory budget and five rotating fresh-process trials. Counter
builds remain separate and System-only. Bind allocator+AP+counter+source/binary
identities to oracles and trials; retain old ADR0016 evidence unchanged.

Then implement an opt-in native reader contract `typed-borrowed-v1`, with legacy
the default and an explicit closed Python/Rust selector. Native mimalloc is an
optional Cargo build feature, never LD_PRELOAD/unprefixed override. Build metadata
must identify allocator and supported reader contract. Old native wheels must
continue to receive only their existing settings for default legacy operation;
requesting the new reader against an incapable binary fails before work begins.
Main installation/defaults are not changed during verification.

The public selector is `preprocessing.reader`, values `legacy` (default) and
`typed-borrowed-v1`; Python backend accepts only `legacy`. Rust Settings accepts
an optional `reader` with the same default. Keep existing ABI/recipe/parser keys;
add build-info string keys `allocator=system|mimalloc` and
`typed_reader_contract=typed-borrowed-v1` for capability detection. The Cargo
feature is `allocator-mimalloc=[dep:mimalloc]`; standalone `allocation_counts`
and mimalloc builds are mutually exclusive. All mimalloc builds pin0.1.52 with
local_dynamic_tls and no process-wide override.

Python strips the default legacy reader key when talking to historical binaries,
and rejects typed+python or typed+missing-capability before constructing source/
collection owners. Operation diagnostics/proof and new PPO preprocessing identity
record the selected reader separately from the actual binary identity.
The new proof key is outer `reader_contract`, outside hashed dataset/source
identity payloads; diagnostics use the same key. New PPO preprocessing identity
adds `reader_contract` alongside its existing backend/native fields. Existing
historical producer identities/source records remain untouched and are not
retroactively assigned a new reader; source hashes and old formats stay explicit.
Normal resume still requires exact code/environment/backend/reader/binary identity;
there is no migration bypass. Early preflight callers must pass the selected
PreparationConfig, not check only whether some native package is importable.
Do not advertise the new capability while its implementation is still a stub.
Operation telemetry records typed Journal file/record counts and legacy Journal
file counts, proving the selected route ran (Replay/control Values are separate).

## Ownership and scheduling

The following is the accepted serial baseline. ADR0018 accepts a subsequent
[bounded parallel parsing design](0018-bounded-parallel-borrowed-journal-parsing.md);
its implementation and measurement acceptance are recorded separately.

Preserve fork/window/join, episode order and existing serial validation/feature
spooling. Typed reader workers return owned raw journal buffers plus existing
owned Replay documents/evidence. The coordinator creates borrowed DTOs only after
owning that result; validates/encodes inside that lexical scope; writes owned
numeric/proof data; drops DTOs before raw buffers. No self-referential owner or
handwritten unsafe borrowing. No DTO/raw pointer crosses IPC or survives numeric
export. NumPy bases continue owning typed numeric arrays and their leases.

This minimal lifetime design moves Journal parsing from decoder workers into the
serial consumer; it does not promise identical parallel efficiency. Record that
change explicitly and compare workers/inflight1/1 and4/4. Do not add a thread pool
or parallel feature encoding in this unit to hide a regression.

Raw capacity, the existing coarse document estimate, control metadata, Replay,
numeric staging and retained NumPy views share existing Config budgets. Admission
precedes allocation and remains shared across workers. Same2GiB/512MiB/16MiB
measurement settings; no budget increase, CPU fallback or reduced concurrency.

## Typed data, validation and features

Promote only the already tested concrete borrowed DTO schema and required-nullable
helpers into native-owned pure Rust modules. Production must not depend on the
benchmark harness. The benchmark should reuse that single schema source rather
than maintain a divergent copy: put the pure schema/contract/text modules under
`native/src/typed_journal/dto`, have the standalone harness include only that
module by path (not depend on the native Cargo package), and keep its owned DTO
experiment independent. Preserve AP/no-AP and allocator isolation; include every
external schema source in standalone source hashes and freeze proofs. Move the
schema generator beside its native-owned DTO outputs and retain a compatibility
launcher for the experiment's existing regeneration command. Preserve
explicit new strict numeric/duplicate/unknown-field semantics; legacy reader
continues its old acceptance policy. No whole-Journal/decision/input/action
conversion back into Value, generic Value-like DOM adapter or custom JSON grammar.

Implement typed counterparts of the actual semantic checks: Journal state/order,
pending ACK/transition/end, actor/revision/action-candidate consistency, visible
own references/perceived catalog/ActionOptions, end projection/outcome/reward
perspective, and source/frozen-policy/Replay reconciliation. Reuse pure shared
checks/feature sinks/config/hash math where possible; explicitly port concrete
DTO traversal, not policy/game rules. Replay/control manifests can retain the
legacy Value reader in this Journal-only unit. Typed views of small Replay action
sections may deserialize directly from borrowed &Value through standard Serde.

Preserve legacy source/proof hashes. Small header/end/control metadata may use
the existing canonical serializer/projection; this is not permission to recreate
ordinary policy input trees. Share a bounded validated-Journal provenance facts
interface between readers for counts/identities/policy versions/end facts needed
by existing projection. Keep diagnostic GameSeed/generation/opponent identity
outside feature inputs, preserve explicit learner_viewer even for same policy IDs.

Typed feature traversal must emit the exact old feature order, names/signature,
float64 values and MLP float32 transforms, using one prepared state per decision
and the saved candidate order/selection. No candidate regeneration or new reward/
GAE/teacher calculation in Rust. Any consequential interface refinement returns
to root/architecture before implementation; do not silently add generic layers.

The shared pipeline uses concrete crate-private sum types: `JournalView` selects
validated legacy/typed ledgers, indexed `DecisionView` selects their records, and
`PreparedFeatures` dispatches legacy/typed prepared encoders. An action-reference
sum is allowed for saved command/candidate slices. These expose only explicit
decision fields, concrete actions and numeric encoding, never dynamic object/
property/array access or a Value-like generic view. Reject representation mixing.
Keep one numeric/proof spooling loop and reader-specific semantic/Replay checks.
`JournalProvenance` carries bounded small header/end control Values, counts,
identity/policy-version/end facts for shared source projection. A typed ledger
must preserve the complete end record, including its terminal final_observation,
because the old proof hashes that control snapshot. One such canonical control
serialization per journal is allowed under the same record/document budget; it
does not authorize ordinary decision/input/action tree reconstruction. Its cost
remains visible in metadata/proof timing.
The typed ledger
borrows an externally owned DTO-record vector (or stores indices); it never owns
records together with references into itself. DTO records in turn borrow the
externally owned raw buffers. All three scopes end before raw release.

## Acceptance and decision

Use old reader/Python feature implementations as untimed differential oracles.
Check recipe fixtures/adversarial input and every frozen accepted learner row for
both observation/history recipes, NPZ/sample/dataset/source identities and index
columns/order. Check saved real16-game PPO ingestion, all candidates/state features,
logprob/value/GAE/targets, including repeated policy IDs and swapped learner seats.
Full semantic cross-record/cross-source mutation tests must fail before publishing
numeric inputs; a failed parent remains wholly offline-ineligible.

Test raw lifetime/Cow ownership, retained NumPy slices after close/GC, shared quota
refusal, canceled/failed workers, late source mutation, cleanup and reaping. Keep
source byte/version checks and the original Simulator pin. Typed strict reader
must accept all current legitimate sources with zero repairs/exclusions; report
actual producer defects separately.

Build separate System/mimalloc candidate wheels; compare legacy and typed reader
under matching1/1and4/4 with five fresh-process rotating trials and warmups. Run
correctness before primary timing, avoid concurrent heavy jobs, separate counters/
profiles, report wait4 peak RSS and sampled tree RSS/PSS/retention/storage/IO.
Record all build/setup/validation/measurement costs and actual hashes/feature graph.

Independent correctness and architecture-conformance review, appropriate Rust/
Python/lint/type/installed-wheel/baseline checks and exact corpus evidence are
required before calling the candidate practically usable. Report throughput and
resource tradeoffs against legacy+mimalloc as well as the original baseline. If
correct but slower, report that explicitly; do not change unrelated scheduling or
claim a speedup. Production default adoption remains a separate user decision.

## Accepted result

Implementation3472700/native source37cae919... passes818Python tests(5opt-in skips),
64native Rust tests(2ignored), public both-reader array-lifetime tests, complete
both-recipe original NPZ/source/sample/index parity and saved real current-v2
16-game PPO eight-array parity at1/4workers. Strict Outcome rejection and exact
scalar metadata shapes were fixed before final measurements. No source/model
producer bytes or main installation/defaults changed.

All48actual-pathcommands passed under a clean84-file code freeze, with5measured
trials per8cells. mimalloc+typed medians25.337/24.030s at1/4workers reduce21.35%/
12.26% versus current mimalloc+legacy and45.14%/21.36% versus System+legacy.
Maximum kernel RSS349.13/368.02MiB stays below the configured2GiB soft budget.
Four workers reduce whole wall time by5.16% versus one, reflecting the explicit coordinator-
local parsing/feature boundary; no full-core-utilization claim is made.

This establishes practical usability as an explicit preprocessing candidate in
the pinned Python/NumPy/Simulator environment. It does not establish broader-
platform performance, a change in learning quality or default adoption.
Evidence: [combined verification](../verification/combined-borrowed-journal.md).
