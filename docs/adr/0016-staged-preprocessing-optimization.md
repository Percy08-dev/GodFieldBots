# 0016: Isolated allocator and typed-Journal experiments

- Status: Accepted2026-10-30; stages0–2 implemented and independently reviewed.
  Runtime/corpus/measurement acceptance PASS; no production adoption.
  Evidence: [verification](../verification/staged-preprocessing-optimization.md).
- Owner: root. Production adoption is a later decision; no fixed target speedup.
- Baseline: e441053, loaded native32e3ad68..., corpus ab5d4913... (83,783learner
  actions,6,784games,13,568participant journals). Recheck these identities at freeze.
- Current environment reports Python3.14.10, while historical timings used3.14.9.
  New comparisons must be measured together, not against historical wall values.

## Completion unit and responsibilities

Implement reproducible stages0–2 only: freeze inputs/environment, compare System /
mimalloc0.1.52 / tikv-jemallocator0.7.0 using isolated native wheels, then compare
Value/owned/borrowed/no-AP Journal representations in a standalone Rust harness.
No simulator changes, games, fits, PPO updates, production reader/default/package
replacement, model/source rewrites or experiment-output reuse as learning data.

Root owns design/config/manifests/dependency locks/docs/acceptance and measured
execution. One bot_implementer owns experiment code/tests; independent correctness
and architecture-conformance reviews follow. Experimental code stays under
benchmarks/native_preprocessing and uses the existing stdlib performance observer
and frozen preparation workload; application modules remain unchanged.

## Stage0 and allocator experiment

Freeze Git/dirty state, source/binary/wheel hashes, actual versions/build flags,
Config, original fit identities and a schema-declared source inventory with hashes.
Revalidate parent/source eligibility with the existing native path before timing.
All inputs remain read-only; output paths are unique and exclusively owned.
Derive the exact ordered Journal list (path/hash/size) from the frozen complete
inventory; never accept an unbound sibling inventory as authority for a subset,
duplicate or reordered corpus. Freeze application/experiment code bytes as well.

Build each native variant in a private source copy. Only allocator declaration and
its pinned dependencies differ; no LD_PRELOAD, global process malloc override,
allocator runtime switch or unrelated compiler change. System retains original
source. Each fresh Python process loads exactly one proven wheel/site. Do not
install experiment wheels into main .venv. Record actual allocator and binary ID.
For dlopened Python cdylibs, enable mimalloc/local_dynamic_tls and
jemallocator/disable_initial_exec_tls; retain jemallocator's default runtime
background-thread support without enabling background threads. Never enable
override/unprefixed-malloc features. This allocator-specific TLS choice was
independently reviewed READY; record its exact resolved feature graph.

Compare complete83,783-row history preparation at workers/inflight1/1 and4/4,
memory2048MiB/cache512MiB/chunk16MiB, with original file/record/scratch limits.
OMP/MKL/OPENBLAS threads1. One warmup and five rotating-order trials per cell.
No concurrent builds/tests/heavy work or perf/allocation tracing during timings.
Reuse existing disjoint coordinator/Python wall ledgers and report summed decoder
counters separately. Do not call those counters exact Value::deserialize time.

All variant outputs must match dataset/sample/array/NPZ identities. Verify Rust-
owned NumPy bases, retained views after close, cross-thread release, cancellation,
quota/failure cleanup and saved real PPO ingestion using existing bounded tests.
Artifact hashes and original model files must remain unchanged.

## Standalone representation experiment

Use System for every representation regardless of stage1 results. A: Value with
arbitrary_precision; B: owned typed DTOs with AP; C: borrowed typed DTOs with AP;
D: Value without AP, diagnostic only. Separate Cargo invocations/targets ensure D
does not acquire AP through feature unification. Keep serde_json1.0.151,
serde1.0.229 and float_roundtrip pinned. No custom JSON parser/DOM or Number type.

New DTO contract is explicitly typed-journal-experiment-v1, not a new source-file
schema. User selected machine-generated schemas: current legitimate files must all
pass, but old reader's permissive duplicate-key/numeric-spelling behavior need not
be reproduced. Legacy remains untouched. B/C have identical schema, numeric domain
checks and presence rules; only ownership differs. Integers stay exact bounded
integers; f64 fields enforce required finiteness. Required-nullable is distinct
from Missing; unknown/duplicate/invalid fields are rejected by the new contract.

First prove small fixtures for floating selection.logprob/value, null/missing,
escaped/unescaped Cow, null/draw/win Outcome, both ordered_defenses forms and empty
respond arrays. Avoid internally tagged/untagged derives that build generic
Content and make AP floating fields fail. Prefer concrete flat DTOs with scalar
tags and per-tag field validation; standard external enums for matching wire sums.
If a required current form cannot be represented by this bounded standard
strategy, stop stage2 and report it for review; do not quietly retain an ordinary
subtree as Value. Only truly arbitrary-JSON schema fields may retain Value and
must be enumerated. No RawValue feature or production binding is added here.

Read all frozen participant journals, collect one file's records at a time, and
release before the next file. No all-corpus in-memory materialization. Raw input
and DTO estimates share the same2048MiB admission: checked32*file_bytes+64KiB
document estimate plus live raw capacity and64*request_bytes+32MiB control
reservation. Refuse before allocation when known sizes exceed the budget, check
actual capacities, and report these estimates separately from observed RSS.
The policy is identical forA–D and adds no per-node memory accounting. Raw input
is outside the borrowed DTO's scope; no self-referential owner or handwritten
unsafe is introduced in DTO/borrowing ownership. B/C must show all-record acceptance and an untimed semantic/
canonical/value/order oracle against the legacy reader. This does not prove the
future typed feature pipeline's numeric arrays; that is a later production gate.

Common eligibility/input/UTF8/record-size/reserved-key checks remain and are not
optimized here. Run correctness/conversion/canonical oracles in separate processes
from timed parse/drop trials. Measure read/common checks, deserialize, DTO drop,
input release and whole natural cycle separately. Owned A/B may release input
after parsing; C retains it until DTO drop. Do not force equal artificial retention.
Numeric/variant-specific shape validation costs must be named if timed differently.
Each variant gets a warmup then five rotating fresh-process trials.

Count allocations/requested bytes in separate System-counter builds/runs, not
inside primary timing trials. The counter's GlobalAlloc adapter may contain only
the required documented unsafe forwarding to System (no custom allocator or pointer
ownership). Counter implementation must use atomics/no allocation,
document realloc/request accounting and report its scope. D's incompatible numeric
acceptance/precision/type behavior is reported rather than silently coerced.

## Failure, measurement and acceptance

Source/byte/hash/identity/schema changes, wrong wheel imports, newly rejected real
valid records, budget/deadline failures or numerical mismatches invalidate that
condition. Never repair/skip files, increase budgets, reduce concurrency or fall
back silently. Distinguish a missing DTO field from a generator defect; identify
Bots Journal vs Simulator Replay origin and report Simulator fixes separately.

Use existing Config budgets unchanged (application soft limits, not an OS quota).
Record whole wall/user/system CPU, RSS/PSS, post-release resident memory, scratch,
storage/I/O lower bounds and build/check/measurement costs. MiB units explicit.
Record all observations plus median/range/IQR/ratios; do not add CPU samples to wall
durations, nested counters together or infer end-to-end gains from microbenchmarks.
Use fresh release binaries for primary trials; sampling profiles, if needed, are
separate diagnostic runs. No claim that OS read bytes mean physical WSL device I/O.

Architecture/correctness review and appropriate experiment/native checks must pass.
Commit experimental code/config/docs at verified milestones. Report candidates as
promising/inconclusive/incompatible; do not auto-adopt an allocator/reader. Future
Journal -> Replay -> worker/pipeline -> remaining feature/I/O changes require a
new measured decision and the applicable architecture gate.

## Observed outcome

All36allocator,24typed-primary and4counter commands succeeded. FullB/C corpus
acceptance is13,568files/534,640records with0newrejections and exact canonical
values/order; all native numeric outputs match the original archive/sample hashes.
With five trials per primary cell, mimalloc reduces whole preparation28.70% at
one worker and10.25% at four; jemalloc reduces23.13%/8.25% with lower post-GC RSS.
Standalone owned/borrowed typed Journal lifecycles reduce22.56%/32.13% versusValue,
including additional typed shape checks. Borrowed DTOs reduce deserialize
allocation requests92.92%; cumulative requested bytes and resident memory are
different metrics. No-AP Value remains diagnostic. These effects are not combined
or extrapolated into a production typed feature-pipeline speedup.
