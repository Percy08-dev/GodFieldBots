# ADR0032: Fresh preloaded actors for native structured Source pytest

Status: accepted; implemented and verified on 2026-11-09 after independent
design, correctness and final implementation-conformance reviews.

## Objective and scope

The owner asks to proceed with the measured remaining Torch-startup candidates.
This increment covers `test_native_structured_session.py` participant startup.
Its prior full-file sum is118.494s; the unchanged diagnostic measures91.829s in
eight collections and82.984s of de-overlapped episode-ready intervals (90.368%).
It creates256participants (192MLP/32random/32heuristic) and six separate numerical
learners. Numerical learners are a separate following increment; this design
keeps their ordinary lifecycle unchanged.

Use ADR0030's existing clean Torch-only template with fresh episode processes,
not a participant pool. Preserve all five cases,128games/512decisions,256episode
participants, updates, source-export and deliberate quarantine/failure scenarios,
actual numerical restarts and exact continuous/resume assertions. No simulator,
dependency, model/optimizer/RNG math, protocol or persisted-format change.

## Responsibilities and interfaces

1. The native session test module explicitly requests one lazy module template
   fixture and wraps only its own `component_http` calls to opt in on supported
   Linux CPU engineering cases. Imported generic helpers, unrelated native tests,
   public application launch, CUDA and unsupported platforms retain defaults.
   First warmup and final real template retirement belong to target timing.
2. Ordinary Python-source first-case collection uses the existing CPU path.
   Extend test-helper eligibility only for exact `_NativeComponentOps` with
   CPU model/inference/training, actions2/current engineering contract and explicit
   rust/structured-v1 preprocessing. Preserve serial readiness. Forbid pooling,
   mixed validation context and arbitrary/native-like operation objects.
3. Before template warmup, the actual collector plan and SAME live learner ledger
   must pass the original operations' config/owner/original-source checks. Validate
   the complete LoadedConfig envelope (config/effective/hash/input origins),
   not merely a matching subset. Extend the existing private
   `_NativeComponentOps.validate_owned_collection(..., loaded: Any | None=None)`:
   default callers retain their existing checks. When supplied, require the SAME
   operations instance's `require_live_owner(ledger)`, compare config/effective/
   hash coherence against the sealed parent's config SHA, and compare actual
   `loaded.inputs` using `NativeStructuredSource.original_config_inputs(0)` inside
   the existing one Source context. The helper passes the actual plan/ledger and
   loaded before warmup; native CLI preflight and post-finalizer checks pass it
   too. Account envelope/query comparison temporaries in the Source budget.
   Keep the loaded argument structurally typed as Any; add no gf_training->gf_cli
   import or reverse dependency. Reuse the one existing
   sealed original Source scope rather than constructing a second parallel owner.
   Do not authorize from mutable caller metadata or reseal sources.
4. Narrowly admit the existing private CLI worker finalizer for this exact native
   CPU test composition, after concrete operation type, private factory, collect
   command, ppo_batch/train, actual config and SAME live learner/ledger/original
   checks. No arbitrary native/source callbacks or new public launch option.
5. After groups drain, before report/manifest publication, require fresh-forked
   closed receipt, healthy template with zero unretired tickets, actual participant
   process/I/O/lease retirement and SAME empty worker inventory. The original
   learner remains live and owned; do not require an actor-only ledger or drop
   its lease. Recheck native original/source metadata, exact learner/ledger and
   complete LoadedConfig using the actual collection before publication. Existing
   before-Adam native checks and quarantine behavior remain unchanged.
6. Preserve common exception/cancellation cleanup, first error/deadline, unknown
   status quarantine and real template/group reaping. A finalizer or source/owner
   recheck failure cannot publish or proceed to update as a healthy collection.
   Fixture teardown restores its context binding even on failure.

Dependencies remain tests -> existing coordinator/source APIs; application never
imports tests. The template retains no model/optimizer/participant data. The new
scope only changes test launch admission and diagnostics. ADR0031 validation and
all normal paths keep their own requirements; no Source qualification is granted.

## Compatibility and verification

The existing optional lifecycle manifest field remains diagnostic, hashed as
actual new source identity. Native readers, journals and source schemas remain.
No old source is relabeled, migrated, rehashed in place or admitted under a newer
meaning. Native Source checks retain raw-record, numeric correspondence, budget,
lock, sealed-original and terminal-retirement authority.

Independent architecture design review precedes one implementer, followed by
separate correctness/conformance review of stable code. New expensive fault
regressions use the existing explicit safety opt-in; lightweight guard/ordering
checks remain default.

- Preserve actual five-case workload and both intentional failed/quarantined
  populations. Verify one template and real final reap, all256fresh participant
  processes, actual empty worker ledgers before updates and all six original
  numerical process lifetimes.
- Negative tests cover fake/stale/mismatched native ops/config/inputs/plan/ledger,
  CUDA, mixed validation, pooling, failed template health and publication rechecks.
  Doubles prove routing/ordering only, not Source authority.
- Compare retained normal complete native parents/rollouts/actual decision records
  and all existing saved arrays against candidate outputs, with both sides' actual
  source hashes/identities validated. Keep failed populations failed. Any opaque
  ID/path/timing projection is comparison-only and explicit.
- Run unchanged native target under the matching disposable timing observer,
  including module warmup/close. Baseline diagnostic native case sum118.743s is
  distinct from the118.494s preceding full-run sum. No promised savings.
- Run affected source/preload/pool/protocol/validation regressions, current Tree
  tests, scoped Ruff/mypy, source-matched wheel/optional-free normal startup and a
  final default full pytest after all authorized increments stabilize. Retain
  every cold-reference test and safety opt-in condition.

Root owns design/docs/config/acceptance/local commits; one implementer owns
scoped application/helper/tests. No other role edits source files.

Design disposition: READY, no remaining blockers. Root adopts the review's
concrete optional loaded-envelope validator, same-Source input-origin query and
temporary ownership accounting. Final implementation correctness/conformance and
the actual target/equivalence checks are complete.

## Final acceptance

Both final reviews are READY. Final matching diagnostic native case sum falls
118.743 ->55.604s; collection union91.829 ->26.180s and episode-ready union
82.984 ->15.440s. The original six numerical processes remain ordinary cold.
The final full-run native file sum is55.387s versus118.494s previously.

Actual saved-data comparison matches876G0/G2checkpoint arrays,4,416valid rollout
arrays and512journal decisions/candidate bytes/search/selection/action content.
Both sides' actual producer/checkpoint/policy identities are validated separately;
changed producer digests/derived labels remain recorded, not relabeled. The three
complete parents and two intentionally failed/quarantined pre-Adam runs preserve
their original states. Eight receipts show one template and256fresh actors with
healthy closure/zero tickets.

Final integrated verification with ADR0033 passes4,177tests/327skips/zero failures
across4,504cases, external942.93s. All358frozen code/config hashes match; all prior
4,396case statuses remain. Related300, actual CUDA7, Ruff/mypy157, wheel/source
correspondence and optional-free normal startup pass. New49native guards are
lightweight; expensive numerical/actor faults remain opt-in. Whole populations
differ and timing is single-observation evidence, not a matched whole-suite claim.
Root accepts this private test-launch scope; normal/native Source authority,
numeric algorithms, simulator and formal learning gates remain unchanged.
[Commands and evidence](../verification/pytest-preloaded-cpu-launch.md).
