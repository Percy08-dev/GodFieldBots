# 0020: Owned thread pool with worker-local episode lifetimes

- Status: Accepted,2026-10-31. Independent design/source/failure conformance READY;
  final correctness gates and measured adoption accepted. See verification below.
- Owner: root. User authorizes implementing pool reuse and complete worker-local
  processing together, with a measured before/after comparison.
- Baseline: commit e766df1, native source d5090c41, release/mimalloc ELF97f1b509.
  Preserve the exact baseline wheel and evidence; main installation stays fixed
  during correctness checks and comparisons.

## Context and requirements

ADR0019 currently reads a window, joins read workers, parses participant files,
joins parse workers, then validates/encodes/spools/releases on the coordinator.
Final-runtime8worker representative elapsed: read1.926s, parse2.279s,
validation2.305s, feature/spooling1.541s, release1.816s. Function samples locate
Replay/Value-tree destruction and allocator work, but do not prove a pool's gain.

Implement one combined candidate, not separate pool-only and pipeline candidates.
One episode is the natural work unit because both participant Journals/Replay,
learner perspective and terminal facts must agree. This coarsens current file
tasks; measure load skew/tails and total time rather than assuming improvement.

Keep public Policy/DecisionInput/environment contracts, reader/config inputs,
NumPy layouts/owners, encoder order/signatures/numeric bits, source/sample/dataset/
index proofs, eligibility and normal resume identity guards unchanged. Both
legacy and typed-borrowed native readers and prepare/build/index/PPO paths use
the same episode helper where their existing semantics agree. Python retains
rewards/GAE/learning/publication authority. No games, fits or optimizer updates.
Source semantics remain simulator ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6,
catalogv2 and stored schema4. No simulator integration change or source rewrite.

## Dependency and ownership decision

Add pinned rayon1.12.0 to native Cargo dependencies and regenerate the native
lock offline without upgrading existing packages. The cached rayon-core1.13.0,
crossbeam and either crates provide the existing library implementation; record
all added identities. Do not build a custom persistent task runtime or global
pool. No Python dependency/extra, standalone DTO dependency or uv.lock change is
needed unless investigation establishes a concrete requirement.

Use one explicitly owned pool per scan operation, retained across all collection
iterations. For workers1, call the identical episode helper inline with no pool.
For2/4/8, reserve the existing8MiBstack per configured worker before construction.
Use ThreadPoolBuilder::build_scoped with ThreadBuilder::run so scoped threads join
before returning even on partial construction/unwind. ThreadPool Drop alone is
not the join proof. Use pool.in_place_scope per collection so the coordinator
stays outside worker capacity and blocking result reception cannot consume the
sole available worker. No implicit global Rayon calls, nested parallelism or
extra read/parser pools. Release stack reservations only after all threads exit.

API evidence is the pinned library's [builder documentation](https://docs.rs/rayon/1.12.0/rayon/struct.ThreadPoolBuilder.html#method.build_scoped)
and [in-place scope documentation](https://docs.rs/rayon/1.12.0/rayon/struct.ThreadPool.html#method.in_place_scope),
verified against cached rayon-core1.13.0 source. These are ownership choices,
not a speed guarantee.

## Data flow and private interfaces

Refactor existing code, rather than duplicating validators/canonical encoders:

1. Coordinator validates each Collection manifest/inventory and creates its
   borrowed EpisodeIndex and stable sorted episode ordinals, as before.
2. Prepare a small immutable collection context containing the actual preflight
   manifest/report digests, encoder/operation settings, optional dataset view,
   and shared Budget/Control/Telemetry. Copy only small digest strings with
   reservations; do not clone the manifest or the global Sources map per job.
3. An episode job receives its original ordinal, directory, EpisodeLinks and
   unique spool path. It owns reading both Journals and Replay files into local
   Sources/EpisodeData. Parse, validate both views, reconcile Replay/metadata,
   derive terminal facts/learner rows, encode and flush/verify numeric spool
   using the existing functions. Header-based viewer mapping remains authoritative.
4. Before returning, drop Journal views, borrowed DTOs, Replay/Value trees, raw
   buffers and scratch working arrays in that same job/thread. No borrowed DTO,
   large JSON tree or Python/NumPy object crosses the result channel. This uses
   ordinary nested Rust scopes, not a self-referential owner or unsafe lifetime.
5. Return owned compact episode facts, optional IndexInfo, source-read evidence,
   reserved canonical proof/identity byte fragments and the flushed numeric spool
   descriptor. The large numeric data stays in the existing private spool.
6. Coordinator commits results by original ordinal: Sources::merge, proof/identity
   ledger append, lineage/split/count/action-hash facts and Stage::episodes. Final
   collection proof, global uniqueness/eligibility, Sources::verify, frozen
   manifest/index hashes and successful publication remain coordinator-owned.

Private seam: an episode-processing helper returns an owned CompletedEpisode.
Keep all mutable Session/Stage/ledger writers off worker threads. Move immutable
worker inputs into a narrow context instead of sharing the full mutable Session.
projection::source_metadata uses job-local file digests and explicitly supplied
preflight manifest/report digests; IndexInfo likewise receives the actual manifest
digest. Do not weaken those checks by trusting stored expected hashes alone.

Factor identity serialization into the same existing canonical byte production;
workers produce bounded bytes, and the coordinator appends them to its split
streams in original order. Per-job hashes cannot replace concatenated canonical
bytes. Preserve source path canonicalization/proof ordering, online manifest
ordinals and index run/game/viewer/decision order. Avoid extra proof-fragment
files unless measured size/resource evidence requires them; returning budgeted
compact bytes changes no persisted dataset format. Only these small fragments
and descriptors are later freed by the coordinator.

## Admission, scheduling and failure

Use a bounded rolling collection scope, not a whole-window join. At most existing
max_in_flight admitted-but-uncommitted episodes exist, including running, queued
and completed out-of-order results. Reserve queue/result/index capacity through
the shared Budget. Admit an initial bounded prefix, receive completions into
ordinal slots, commit the available prefix, then admit replacements. A slow first
episode can stall this bounded prefix; accept that initial simplicity and measure
the tail. Never let reorder buffering grow with the corpus or raise Config limits.
No whole-corpus job vector or speculative independent Config knobs are needed.

Use the existing Control checks at job admission/start, unchanged parsing/
validation/encoding boundaries and before success. Catch each job panic and send
a failure result so the coordinator cannot wait forever for a missing ordinal.
Preserve the first actual cause before cancelling siblings. After any worker or
coordinator error, stop admission, cancel, keep draining/joining all admitted jobs,
and exit the collection scope; then let the owned pool terminate/join. Result
transport must not block shutdown (bounded capacity covers all admitted results,
or drop receivers/handle disconnection deliberately). No failed job/window/scan
may export a successful batch or confer offline eligibility.

Only after workers join may the stage cleanup owner remove partial spool files.
Explicit cleanup failures retain the primary error using the existing Cleanup
context; do not overwrite it or claim clean release. Reserve shared scratch/file
and heap/result capacities before allocations/writes as before. OOM, deadline,
spawn, quota and write errors propagate; no CPU fallback, budget increase, dropped
source or automatic worker-count change. Existing external cancellation and
native owner finalization still drain before source locks/publication release.

Implementation review clarifies numeric-spool ownership: reservations belong to
a budgeted stage registry before file creation, not solely to worker results.
Partial failure and discarded uncommitted results must not release disk admission
while files remain and siblings can still write. Clear this registry only after
joined cleanup successfully removes the stage directory. If removal fails, retain
charges while Stage is alive/retryable and return Cleanup with the primary cause.
Permanent destruction after unsuccessful cleanup retains the existing soft-limit
limitation: residual files can outlive accounting after all workers have joined;
report dirty cleanup and allow no successful publication/reuse. A persistent
orphan/quota manager is outside this unit; revisit if cross-operation hard disk
quotas or automated orphan recovery becomes a requirement.

## Timing, compatibility and responsibilities

Version the native scheduling diagnostics: job unit becomes complete episode,
with admitted/completed/committed counts, configured/actual pool workers, pool
starts/exits, active peak and queue/reorder peaks. Retain explicit per-job phase
wall/worker totals as nested diagnostics. Do not label them exclusive command wall
or CPU utilization. The coordinator's exclusive wall ledger names pool wait,
ordered commit and remaining global work; keep exact integer reconciliation and
explicit version/scope. Historical parse-window diagnostics retain old meaning.
Update affected tests/consumers for the new capability/version instead of silently
reinterpreting old fields. Numeric/source formats and reader selectors stay fixed.

Root owns design/config/dependency decision/docs/orchestration/acceptance/commits.
One bot_implementer owns scoped native refactoring and behavior tests after design
READY. bot_reviewer independently reviews correctness/leakage/ordering/failure;
bot_architecture_reviewer checks design first and final conformance. A verifier
may run bounded sequential source proofs. No child spawns further agents.

## Verification and before/after acceptance

Test exact serial/multiworker outputs and order under shuffled completion and
uneven work, actual pool reuse/worker lifetime,1inline and2/4/8workers, in_flight
below/equal/above workers, both readers, zero-decision episodes, learner/header
viewer mappings and index/PPO variants. Demonstrate large raw/DTO allocation and
drop thread identity before result handoff. Exercise invalid later jobs after
successful queued results, panic, cancellation/deadline, quota, spawn and write/
flush/cleanup refusal: no publication, all workers joined, leases/files released.

Run relevant Rust tests/fmt, Ruff/mypy, full final-wheel Python regression,
both complete fixed feature recipes/NPZ/arrays/source/sample hashes,83,783-row
Parquet identity and saved real current-v2 PPO1/4worker proof. Existing numerical/
source/NumPy lifetime oracles remain independent. Source jobs run sequentially;
async/PPO/full pytest use the known permitted socketpair environment, not the
sandbox that returns EPERM. Always unset inherited Simulator VIRTUAL_ENV.

Preserve baseline97f1wheel; build a separate opt3/debug0mimalloc candidate. After
correctness/source reviews and advance user notice, run both variants in the same
campaign at1/2/4/8workers with max_in_flight=workers,2GiB/cache512/chunk16MiB,
one warmup plus five rotating alternating fresh-process trials per cell:48commands.
Record each variant's numeric/source/build/owner/phase proofs, whole/stage elapsed,
OS-reported CPU, kernel peak/post RSS and sampled PSS, managed/queue/scratch peaks,
model size/logical storage/I/O and total campaign cost. No own builds/tests/edits
or competing source jobs during timing. User reports the heavy task has ended;
record actual snapshots and do not infer host isolation. Stop unexpected failures.

Compare complete-path elapsed/resources, not obsolete serial-phase versus summed
parallel-worker totals. Keep CPU API/sampling limits explicit. Independent audits
resolve blocking findings before acceptance/commit/main-wheel update. If correct
but slower, report that outcome and retain the baseline pending a decision rather
than manufacture a gain. No separate pool-only speed claim is sought in this unit.

## Alternatives and revisit conditions

Pool-only then pipeline-only experiments isolate causes but the user authorizes
the combined implementation; before/after measures the combined result. A custom
std-thread queue runtime adds shutdown/lifetime machinery supplied by scoped Rayon.
Global Rayon pools hide ownership and sizing. Passing decoded trees back to the
coordinator preserves the current release bottleneck; moving all authorization/
global commits to workers would weaken ownership and order. None is selected.
If episode skew makes bounded-prefix stalls consequential, separately measure a
more flexible scheduler later. If compact proof/identity bytes dominate memory,
revisit budgeted fragment files without weakening final proofs. No new format,
prefix-validation cache or library beyond this scoped pool follows automatically.

## Executed acceptance

Source faf71b33/release-mimalloc ELF2ed57a31passes89Rust tests (2existing ignored),
826Python tests (5existing conditional skips), full Ruff/mypy59, complete frozen
history/observation/index and saved current-v2 PPO1/4exact source/numeric proofs.
Two review findings around scratch reservation ownership were resolved with
held-sibling and actual partial-write tests, then independently re-reviewed READY.

All48same-campaign before/after commands pass original output/build/route/owner/
integer-ledger/source/model/main/code checks. Whole mediansbefore->after:
1worker22.721->22.688s(no meaningful gain),2workers19.641->14.857s(24.36%),
4workers16.310->10.906s(33.14%),8workers14.406->8.794s(38.96%). All five
paired comparisons improve at2/4/8. Reported CPU at4/8rises61.94/66.70%; known
accounting uncertainty is not a reason to dismiss the cost or claim lower CPU
work. After8maximum RSS rises19.40MiBto512.72MiB and managed peak85.15MiBto
1677.34MiB, within unchanged2GiBpolicy. Root accepts this measured throughput/
resource tradeoff. Scratch/staging/model bytes remain unchanged.

After acceptance, exacttestedv3wheel replaces main97f1offline without dependency
or Config-default changes; old package/metadata backup and fresh identity/CLI
startup/protected-model checks pass. Resume guards remain. No games, fits or
optimizer updates. [Full verification](../verification/episode-pipeline-pool.md).
This supersedes ADR0018/0019's read-window/parse-window/coordinator scheduling
ownership only; their contracts, historical evidence and other decisions remain.
