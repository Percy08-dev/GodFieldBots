# 0053: Shared MLP inference under the adopted Simulator contract

Status: accepted and implemented; bounded functional/physical acceptance complete.
Date: 2026-11-18. Owner: root. Baseline: c78f8a0.
Preimplementation architecture review: /root/shared_architecture, READY for
design SHA-256 ef888d3e40679480531821b528bee7f37677bcd3f809ff952afd81afd128c8e6.

Two implementation refinements also pass the architecture design gate:
CPU host screening (draft049281ab30a64d0eaf5ba1daa752cd1f011b63a8b604f52848aac1986c2162a0)
and campaign evaluation dispatch
(draftfdebb08795c43af8faab0a5b529f14b799240305a1dda4ac2c1c1c6202e2e7b7).
Component screens and the real C16 calibration support the implemented soft
host gate; it is not a universal physical-memory bound.

## Context

ADR0050–0052 improve existing resident GPU admission and diagnostics. Four
concurrent games run within the original12GiB managed budget, but each neural
participant retains its own CUDA context/weights. Simulator issue72 previously
blocked shared execution across participant views. Revision
072dabbca1458c9526897781d2fd64edeeb4f9ca now explicitly permits trusted shared
batch inference subject to independence, checkpoint and response-lifecycle tests.
Historical91d5926 sources keep their original contract.

## Decision

Implement [the bounded MLP design](../design/mlp-shared-batch-inference.md).
Add lazy optional TorchRL0.15.0/TensorDict0.15.2 and an explicit Python-only
shared mode for producer-owned Model4 HPO and ordinary Model3 PPO. One clean CPU
broker bridges dedicated participant channels to official InferenceServer and
MPTransport; one GPU process holds immutable inference-only checkpoint modules.
CPU participants generate candidates, sample using their own RNG and submit Act.

The broker owns the original multiprocessing child, independent of TorchRL's
shutdown return. Service reservations, cache references and retirement proofs
are explicit service resources rather than participant-shaped records. HPO
retains its durable global admission/quarantine; ordinary mode retains local
invocation custody and requires actual service cleanup before healthy completion
or pause. New identities/versions do not reinterpret old config/Source receipts.

Checkpoint acceptance is immutable; postcommit official update_model_weights
publishes a fresh registry entry while referenced older versions remain. Match
execution/game/participant/request/state_revision/checkpoint before delivering
only the requester's ordered scores/value. Expired responses produce no action
or learning transition. Private lengths/values/order must not affect another
row; use row-local arithmetic where necessary and report actual vectorization.

## Alternatives and consequences

The inherited actor host charge describes learner/CUDA allocations. Replace it
only after a component screen supports a versioned soft CPU estimate, including
full-history decoding, actual search/framing lifetimes and process baselines.
The scoped advisor recommends this over a new per-object enforcement framework;
the latter adds runtime cost and changes failure behavior without an evidenced
requirement. Existing learner compatibility limits remain explicitly identified.

Schema5 campaign evaluation selects existing fresh participants through one
explicit selector. Translating the shared descriptor into resident evaluation
would introduce cache/reset custody without meeting a new requirement. Preserve
the shared training parent's identity and strict completion/Source proofs;
record evaluation's actual mode and preflight its own capacity before a full
campaign. Short training-only measurements do not acquire evaluation resources.

Passing MPTransport clients directly to existing exec bootstraps is unsupported:
their queues/events require multiprocessing inheritance. Changing all actors to
parent-main multiprocessing would break the clean participant bootstrap. A
custom replacement batch server duplicates the approved official transport.
The CPU broker retains clean actors/coordinator while adding explicit nested
process custody and IPC cost, which the bounded comparison must measure.

Native shared formats, GRU hidden-state serving and Transformer KV serving are
deferred. Existing native readers reject unsupported shared/profile records.
Revisit model-specific serving only after interleaved own-history equivalence.
No full-RAM physical limit or learning-quality result is presumed. Broker loss
before child-reap proof can leave HPO admission requiring existing explicit
operator recovery; no new administrative recovery authority is introduced.

## Verification

Require adversarial private-row independence/routing/version/retirement checks,
actual service process cleanup, focused and whole/static/package checks and two
independent final review roles. Then execute the owner's seven-HPO-run384-game
protocol plus one short ordinary Model3 run. Decide preset adoption from matched
three-game concurrency and select concurrency from measured throughput/headroom.
Keep first/second collections, estimates and physical samples distinct. Record
final evidence in [the verification record](../verification/ppo-shared-inference.md).


## Implemented refinements and acceptance

The calibrated host recipe replaces CPU actors' inherited learner/CUDA envelope
with private participant high-water, aggregate empirical shared PSS, role
residency and reachable phase ownership. Parent is nested in Source; Source
checkpoint backing and GPU weights are not added as separate generic host caches.
The aggregate shared coefficient may include Source-backed overlap, while extreme
history/search/framing and filled-cache occupancy remain unmeasured. Impact:
planning can overstate some shared pages and miss unmeasured allocator growth.
Revisit with simultaneous mapping evidence when admission is marginal, and
recalibrate when runtime/model/topology or observed high-water changes.

External audit accounting preserves the recorded preprocessing/storage limits.
At external E <= original O it retains the existing flat joint pool; for a fresh
independently owned scan with E > O it reserves O once before pool construction
and admits numeric absolute peak separately. Actual pool/native/control cleanup
and cohort last use govern refunds; aliases and tracebacks retain custody.
Existing/borrowed pools receive no new loan or resizing. Supporting nested borrowed
custody is deferred until a concrete caller requires its own ownership design.
The duplicate scan-control charge is removed without changing parsing guards,
Source checks, producer budgets, datasets or persisted schemas.

Both final review roles are READY. The required seven HPO runs/384 natural games,
one predeclared reverse-order C8/C4 confirmation/128 games, and ordinary Model3
C2/two updates/32 games complete. All nine HPO parents plus ordinary pass full
Original/Replay/numeric and applicable termination/reference audits. Select
shared C4/B4/1 ms for the 32-game scaling preset: C4/C8 throughput plateaus and
C4 uses about 2.37 GiB less PSS. Matched C3 shared mode is about 5% slower while
saving 3.895 GiB PSS and 1.772 GiB sampled whole-device usage; retain the resident
preset for that throughput condition and expose shared mode as the memory option.

Final Ruff/mypy and installed mandatory-only wheel checks pass; all 249 wheel
modules match source. The whole-suite result remains 51 FAIL/7158 PASS/574 SKIP
with recorded fixes/rechecks and 30 pre-existing missing-retired-Tree-fixture
failures; no whole-green or model-quality claim follows. See the
[verification record](../verification/ppo-shared-inference.md) for exact evidence,
code/Simulator identities and reproducibility limitations. Leave the branch
unmerged; GRU then Transformer retain their own future isolation gates.
