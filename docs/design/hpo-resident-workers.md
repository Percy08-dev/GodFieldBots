# HPO resident workers and progress implementation plan

> Agentic execution: scoped subagent-driven development, one code writer at a
> time. Root owns docs/configuration, decisions, staging, commits and acceptance.

Date2026-11-15. The user approved this plan and requested implementation.
Preimplementation architecture review `/root/resident_inference_design_check`:
READY after lifetime, cache-reference and Source/commit-order resolutions.
Implementation, actual performance and final acceptance are pending.

**Goal:** amortize HPO participant startup/model loads across games and updates,
and add nonblocking terminal/live-watch progress.
**Architecture:** retain the existing coordinator, participant-local generation/
selection, learner and Source owners; add explicit physical-pool/parked authority
and a separate diagnostic channel. **Stack:** existing Python3.14/GIL, uv,
Torch CUDA/native Rust and stdlib. No dependency or Simulator change.

## Global constraints

- HPO training/evaluation/direct comparison only. Ordinary PPO and old fresh
  contracts retain their meanings. No implicit old-record migration or resume
  across changed source/runtime/configuration.
- Simulator91d5926a7572db327eaea3751e7c24f428120996, corrected-v2/public-ja-v3 and
  the reviewed spec inventory remain authoritative. Only acting-participant
  observations/events, permitted same-revision atomic hints and pinned catalog
  references are policy inputs. No other-player view/GameSeed/management data,
  SequenceOptions or rejection oracle.
- Maximum2games/4physical workers/1trial; initial RAM24GiB/CUDA12GiB managed
  limits. Retain idle residency and all transition/Source/save allocations.
- Default models/canonical control/reward/gamma1/selection/optimizer/loss and
  all8fixed kernel ASTs retain their meanings. Future exploration models use
  separately agreed major widths192/main depth2; model HPO/formal sizing is later.
- No child agents spawn agents. One code writer; root alone commits. Preserve
  user files, old evidence and simulator repository. No push/publication.

## Accepted design

### Responsibilities and lifecycle

`gf_simulator_client` owns physical clean bootstrap, framed transport and logical
episode leases/reset. `gf_training` owns the actual GPU job/SAME audited ledger,
pool lifetime, update authority, Source/commit and reference transactions.
`gf_models` owns validated inference loading/reloading and existing network math.
`gf_cli` composes them and owns progress I/O. Common holds closed pure values;
native validates new original/evidence/plan branches, never learning math.

Training pool scope=(actual job,parent,learning seed,invocation). Evaluation/
comparison scope=(actual evaluation job,immutable parent-set,seed-pair); use a
fresh pool per panel. Never cross GPU jobs, training parents/seeds, train/validation
or pause/resume. Physical owners are permanently viewer1or2, up to2each, lazy
clean-bootstrap startup/minimal environment. No physical process consumes both
seats' views. Each owner has one exclusive logical lease at a time.

Use a production `resident-worker-v1` envelope around existing protocol3 episode
behavior, with BEGIN/SELECT/END/PARK/SYNC/EVICT/CLOSE and identity/epoch-bound ACKs.
Reuse existing generation/selection rather than test-only code or copied rules.
END succeeds only after valid clean completion/reset, not EOF/error alone.
Reset/verify ALL EventBank/previous frame, policy/search RNG, GRU/KV/history,
positions/owners/revisions, timings and pending I/O/responses, including inactive
cached policies. CUDA synchronization precedes PARK. No batching or CUDA IPC.

### Cache, update and retirement

Per physical worker: at most6immutable checkpoint identities/128MiB of weights,
LRU eviction of unreferenced entries. Key full model/action/inference binding and
checkpoint manifest/weights identity, not label alone. Retain metadata/hash
checks on admitted use. Current/in-flight entries are protected. Same-architecture
buffers may receive new weights only after the old identity is unreferenced;
otherwise preserve a distinct frozen copy. Never overwrite opponent snapshots.
Synchronize inference weights only, not optimizer/moments. Account contexts,
temporal/activation/allocator residency and staged old+new weights separately;
the previous~7.9GiB estimate does not prove this expanded implementation fits.

Required ordering:
1. Drain games; validate END/reset ACKs; synchronize CUDA; PARK all physical
   owners with no private episode state/request/I/O/decision reservation.
2. Publish Collection7 logical retirement evidence. Issue live authority bound
   to SAME job/audited ledger/pool/owner set/generation/lease epochs; recheck at
   collection/update/Source retirement/prepare_commit boundaries. BEGIN, SYNC,
   owner changes or failures invalidate the prior authority.
3. Keep physical workers charged and PARKED through Source preparation,
   numerical update/save and prepare_commit. Preserve original custody/decoded
   retirement. Ordinary begin_update stays exact; explicit resident HPO authority
   alone permits parked owners. Never fake reap/remove live ledger entries.
4. Finish durable generation append before cache-reference mutation; preserve
   prepare_commit's retirement of old run/registry snapshots.
5. Acquire durable Registry5 pins BEFORE reading newly committed checkpoints,
   reserve complete transition memory, SYNC current-policy workers and verify
   exact digest/generation/reset/PARK ACKs.
6. Confirm eviction or actual process reap, release matching old pins, then
   retention. Pending acquisition/release intents protect files after interruption.
   Missing ACK never establishes eviction.
7. On pause/complete/evaluation termination/failure, close/reap ALL physical
   owners and retire pins/custody before actual successful GPU termination.

Logical retirement differs from physical termination. Timeout, bad identity,
partial copy, reset/cancel/cleanup failure poison the job and use owned cleanup;
no policy reproposal, invented outcomes, deadline extension or unproved release.

### Compatibility and interfaces

New campaign3/run-spec3 explicitly select `resident-worker-v1` in a runtime table
and may include the existing optional mixed16 descriptor. Use dedicated variants
so old dataclass/asdict/hash outputs stay exact; bind mode/cache limits into
study/config/resume identity. Old campaign/spec1/2 remain fresh.

`run_variant="hpo-ppo-resident-v1"`+spec3 selects Collection7/Proof4/Registry5/
native HPO plan6. Keep Run5/Update5/Generation4 with exact historical branches and
new required evidence dispatched by variant/spec. Registry5 adds cache_references
with separate durable intent/receipt transactions under SAME parent lock;
never misuse evaluation references. Preserve existing transaction mechanisms.
Native marker `card_instance_hpo_resident=collection7;proof4;registry5;pool1`
is additive; old marker/plan5 stay exact. Refuse unsupported native before GPU.
Model4/inference5/resume5/numeric5/wire6/Journal6/candidate codec/math remain exact.

### Progress

HPO run/compare gain `--progress auto|plain|off`: auto renders stderr only on TTY,
redirected auto is silent; plain is line-oriented stderr; final stdout JSON stays.
New `hpo watch --path PATH [--interval 1] [--once] [--json]` observes campaign or
comparison roots read-only, without locks/Torch/Optuna/GPU. Ctrl-C stops only watch.

The sole campaign/comparison owner writes schema1 telemetry/progress.json using
atomic replacement, max64KiB. Exclude ONLY that reserved subtree before immutable
inventory traversal; keep every other strict check. No telemetry has Source,
score, completion, cleanup or resume authority. No unbounded history/Web stack.
Training/collector emit bounded callbacks toward CLI, never import CLI. Phase
changes/best-effort async heartbeat at about1Hz; no new writer thread/per-decision
disk write. Synchronous phases may delay it; stale after5s means update pending,
not inferred crash/completion. Missing/malformed/old telemetry is displayed honestly.

Fields: model/control-or-trial/seed/phase(startup,collect,source,update,save,sync,
evaluate,cleanup), COMMITTED used/D/updates, separate transient games/decisions,
latest committed loss/entropy/KL, durations and starts/reuses/reloads/copy bytes.
Monotonic active invocation clocks; per-run ETA only after2completed intervals,
latest<=3intervals and no pause gaps. Terminal success requires actual authoritative
completion/cleanup. Write failure warns once/disables persistence without changing
training gates or deadlines.

## Review focus

- SAME-owner parked proof cannot be replaced with flags/hashes or hide pending I/O.
- All private/cache/RNG state resets across lease/model/update; immutable aliases.
- Lost ACK/pin transaction/copy/cancel/cleanup cannot permit pruning or fake release.
- Old closed formats/optional-free imports/identities and numeric ASTs stay exact.
- Diagnostics remain bounded, non-authoritative and nonblocking on execution locks.

## Task 1: Versioned contracts and resource/reference foundation

**Ownership:** implementer: common resident contracts; training HPO config/lineage/
retention/native plan/ledger seams; model resource helpers; CLI hpo_config; required
native/src changes and focused tests. Root: ALL docs/config TOMLs. No pool activation.
**Interfaces:** accepted design -> explicit new variants/readers/capability,
cache-reference transactions and parked authority seams; old entry points exact.

- [ ] Add meaningful failing tests for old/new dispatch/hash, foreign/invalid
  lifecycle, cache pin recovery/protection, memory transitions and optional-free load.
- [ ] Implement only accepted new branches. Return unforeseen boundary decisions
  to root before dependent changes; no simulator/dependency changes.
- [ ] Run focused Python/Rust tests, scoped Ruff/source mypy, report RED/GREEN,
  changed files, exact results and concerns to root. No agent commits.
- [ ] Independent correctness/conformance review; root resolves blockers/commits.

## Task 2: Resident actor pool and HPO runtime integration

**Ownership:** implementer: simulator-client pool/bootstrap/worker integration,
model validated reload helpers, training ledger/Source/coordinator, CLI collector/
evaluation/comparison wiring, necessary native branches and behavioral tests.
No root-owned docs/config edits.
**Interfaces:** Task1 contracts -> real per-job viewer-partitioned pool,
immutable cache, live PARK, post-commit weight sync and actual cleanup/evidence.

- [ ] First failing tests for multi-episode/viewer/epoch/checkpoint isolation,
  A-B-A reuse, frozen alias, lost/partial sync ACK and constructor/reset/cancel/
  timeout/teardown failures.
- [ ] Implement accepted transport and full reset, then exact update/Source/
  commit/pin/retention order with idle memory charged. Reuse common coordinator.
- [ ] Integrate HPO training/evaluation/comparison with exact panel scope; real
  final reap precedes termination receipts. Test-only code is not a dependency.
- [ ] Verify fixed-input fresh/resident selections/logprob/value and saved arrays
  for all3models plus old/native regressions; independent scoped reviews/commit.

## Task 3: Bounded telemetry and terminal/watch

**Ownership:** implementer: diagnostic values/callbacks, CLI hpo.py/progress/
reader/renderer, minimal coordinator/collector hooks and tests. Root: docs/presets.
**Interfaces:** committed state + bounded phase/pool facts -> diagnostic snapshot,
stderr renderer/read-only watch, no new learning imports/threads.

- [ ] First failing tests for held-lock live watch/read-only paths, malformed/
  missing/stale data, TTY/redirect stdout, bounded writes, observer cancellation,
  optional-free imports and diagnostic-write failure independence.
- [ ] Implement specified API/fields/ETA/cadence/inventory exclusion.
- [ ] Focused CLI/runtime/static checks, independent review, root acceptance/commit.

## Task 4: Integration, hardware verification and final acceptance

**Ownership:** root: declarations/config/docs/acceptance; verifier: specified
read-only checks; implementer: fixes; independent reviewers: stable whole diff.

- [ ] Preserve originals; all8AST pins, historical Source/format proofs,
  Python/native parity, full Python/Rust suites, Ruff/mypy, optional-free CLI.
- [ ] Build/package/activate reviewed source-matched native with rollback evidence
  before new-mode hardware work. Record code/native/Simulator/hardware/config hashes.
- [ ] Fresh directories; all3width192/depth2 models; seed17/same recipe/opponents/
  action settings within each fresh/resident pair. Two updates and separate
  unchanged pause/resume. Operational probe D1,000,000/max_updates2 is EXPECTED
  incomplete/update_cap/no objective, not fake completion; original deadlines.
- [ ] Run3paired timing samples/model including startup/reset/checks/save/sync/
  retention/cleanup. Separately complete bounded D64/seed17 CLI/Optuna campaigns,
  paired evaluation/direct comparisons; explicitly functional, never strength.
- [ ] Measure total time and allocated/reserved VRAM. Verify amortized startup/
  loads and admitted resources; investigate regressions before adoption.
- [ ] Both final independent reviews READY, blockers resolved; current docs/ADR/
  HPO instructions and verification record reflect actual evidence/limits. Root
  commits locally; no push or formal HPO launch.

## Accepted debt and revisit conditions

Small duplicated weights and serialized per-worker execution are accepted.
Revisit shared GPU execution/batching only when measured contexts/IPC/latency
justify the broader information/failure boundary. Best-effort progress may stale
during blocking work; reconsider independent observation only for demonstrated
liveness needs. Managed memory is soft and accompanied by device measurements.
Formal search ranges/D/trials/seeds/time budget remain a separate decision.
