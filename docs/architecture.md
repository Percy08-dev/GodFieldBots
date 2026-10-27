# GodFieldBots architecture

## Status and scope

The user's updated Simulator contract authorizes acting-player ActionOptions
atomic hints. [ADR0005](adr/0005-authorized-atomic-action-options.md) is accepted
after independent design review and supersedes the earlier blanket prohibition.
Its implementation uses actor-scoped, same-revision snapshots with journal/
derived schema 4 and policy input `player-atomic-options-v1`. Independent
correctness and conformance reviews are READY; live verification and pilot
evidence are recorded separately. SequenceOptions and
rejection-derived inputs remain excluded.

The source of requirements is the user's implementation plan (2026-10-27).
The first acceptance unit is Phases 0–3: configuration, simulator lifecycle,
public candidate generation, baseline play, journals and fixed evaluation.
Phases 4–9 are subsequent units gated by evidence from this foundation.
No training or integration success is implied by this design.
Initial architecture review: READY from `bot_architecture_reviewer` on 2026-10-27.
Phase 0–3 foundation is implemented. Independent final correctness and conformance
reviews are READY; see [current ActionOptions verification](verification/action-options.md),
[coverage history](verification/coverage-followup.md) and
[initial verification and phase gates](verification/phase0-3.md).
The old observation-only evaluation had 4/12 rejections. With authorized atomic
hints the fixed 12-game evaluation is clean. The first frozen 192-game pilot
recorded 182 natural games, 9 unsupported buy-control states and one omitted
dynamic-weapon singleton. The latter is fixed in generator v3; the whole failed
pilot is retained and remains ineligible. The corrected-contract pilot described
below now passes provisional feasibility. Phases 5–9 are not
implemented. See the [current verification](verification/action-options.md).
Phase 4 implements [ADR0006](adr/0006-lightgbm-action-value.md): native
CPU LightGBM action scoring, versioned public encoders, checkpoint inference and
gated fresh training. Independent final reviews and verification are recorded in
[Phase 4 verification](verification/lightgbm-phase4.md). The preserved original
failed pilot cannot authorize a real-data fit; explicitly synthetic component
verification is permitted by the implementation plan.
The user now authorizes the corrected Simulator integration and fresh evaluation/
pilot. [ADR0007](adr/0007-staged-buy-contract-adoption.md) has independent design
READY: a narrow resolving/buyer input condition, new revision/spec snapshots and
inference schema 2 bound to gameplay identity. Implementation and independent
final reviews are READY. Fresh evaluation is 12/12 natural, and the complete new
pilot is 192/192 natural with no rejection or health error. Recomputed feasibility
from all 384 participant journals passes; 2,982 accepted decisions include 26
purchase choices. This is provisional execution feasibility, with no real-data
fit or model-quality evaluation. See
[adoption verification](verification/buy-contract-adoption.md). No old outcome
or dataset is reinterpreted.
The [rejection diagnosis](issues/0001-public-observation-legality-investigation.md)
now explains the four cases as a protected-discard rule absent from the public
normative text and attribute-incompatible defenses with insufficient permitted
explicit context. No instance-ID mapping defect was demonstrated. The updated
authorization resolves atomic availability through ActionOptions without extra
opponent catalog access. The [Simulator handoff](issues/0002-learning-contract-handoff.md)
retains historical findings and the remaining normative documentation task.

Coverage follow-up: [ADR0003](adr/0003-episode-closure-projection.md) is implemented.
It separates raw simulator lifecycle Aborted from projected
turn-limit truncation, records a diagnostic abort reason, and introduces journal/
derived schema 3 with an explicit end-projection version. ADR0005 supersedes
that storage/input boundary with schema 4; schemas 1–3 remain
historical and must not be reinterpreted. Final correctness and conformance
reviews of this follow-up are READY; the schema-2 history is retained in ADR0002.
The test-driven CLI cancellation fix must wait for already-owned active groups'
bounded cleanup before returning, including when queued semaphore waiters cancel.
This restores the existing coordinator ownership, not a new concurrency layer.
[ADR0004](adr/0004-coordinated-cancellation.md) clarifies the late/repeated
cancellation case: every exceptional path drains owned tasks, including child
TimeoutError. Nominal cleanup expiry is recorded once as unsuccessful; joining
continues without new work or resetting deadlines under the existing OS/I/O debt.

Simulator pin: `860d8b92db4ce82012e040510f221b52241f28aa`; HTTP `/v1`,
`true-match-1v1-v1`, Observation `"1"`, Action metadata `1`, TestServer info `1`.
Catalog: `public-ja-v1`, schema `1`, locale `ja`, 296 definitions,
SHA-256 `4bb3ac87e4567798767f7c74a9f9280208adf931277c6afc7deeb865ef4d0673`.
Record actual code revisions, dirty state, and binary hash, without claiming a
prebuilt binary was produced from the current source. Simulator owns final
legality and transitions; this task must never modify it or copy its engine.

Provenance amendment reviewed READY on 2026-10-27: `server.simulator_revision`
declares the supported contract pin; optional origin-relative `server.source_dir`
verifies local Git HEAD before startup. If configured but unavailable/mismatched,
reject; if absent, source revision/dirty are unknown and declaration is unverified.
Moved prebuilt binaries and external HTTP do not require a source checkout or a
particular binary directory layout. External mode never inspects/owns the binary.
Checkout HEAD never proves binary or installed-wheel build identity.

Normative sources in `../GodFieldSimurator/docs/spec/`:
`07-external-interfaces-adapters.md` §§9–15,21.1,46;
`10-cli-client.md` §3.1 (Rust typed metadata dialect);
`01-server-spec.md` §§26–27; `05-observation.md`;
`09-gameplay-rules.md` §§9–18; `06-persistence-history.md` §§28.6,28.11.
Wire DTOs may resolve representation details; no engine dependency is allowed.

## Responsibilities and dependency direction

| Package | Responsibility | Dependencies |
| --- | --- | --- |
| `gf_common` | JSON-safe domain types, protocols, public action space, lazy search, pure scheduling | stdlib, NumPy |
| `gf_simulator_client` | validated HTTP, attestation, subprocess/game ownership, worker transport | common, httpx |
| `gf_training` | journals/projection, split and feasibility checks, owned LightGBM training | common, model adapter, optional PyArrow |
| `gf_evaluation` | paired scheduling, aggregation and costs | common |
| `gf_models` | participant-local Random/heuristic/LightGBM, pure encoder and native artifact | common, lazy extras |
| `gf_cli` | strict config, composition, commands, manifest | application packages |

The existing `godfieldbots` entry point becomes a compatibility wrapper.
Common never imports upper layers. Models never handle HTTP or management
configuration. CLI constructs concrete policies/environments for evaluation.
Runtime is CPython 3.14, GIL enabled, uv; baseline dependencies httpx/NumPy.
Extras are `data` (PyArrow), `lightgbm`, `torch`; dev is pytest/ruff/mypy.
Optional libraries must not load for baseline or CLI help.

## Config and compatibility

Use TOML/dataclasses with closed schemas and explicit type/domain checks.
Reject unknown keys, bool-as-int, nonfinite numbers, unknown versions, bad
references/cycles, unsupported models and model/training mismatches before I/O.
Model presets arrive with implementation; do not create fictional working models.

Experiment references actions/model/optional training relative to the originating
file. Merge recursively by field: common training/actions, model and explicit
training differences, experiment overrides, explicitly supplied known CLI fields.
Arrays replace wholesale. Persist effective config, canonical config SHA-256,
input file hashes and resolved paths. Version defaults in the effective config.
Workers receive policy ID/sampling seed and action settings plus an optional safe
checkpoint path/digest binding, never full experiment/server config, raw GameSeed
or diagnostic metadata.

The adopted Simulator permits input for the current nonterminal viewer during
awaiting_action/awaiting_response and, narrowly, resolving with
buy_accept_reject. Its exact existing buy_decision descriptor is required.
Other resolving states remain non-actionable. The buyer comes from current_player,
not active_player; the target's preceding Allow/Mirror response and the buyer's
purchase choice are separate decisions. These public rules follow the new pin;
full-possession Buy behavior is project-provisional under Simulator09 §17.3.

All limits are configurable: initial beam width 64, 512 candidates, 20,000
expansion attempts, sequence length 3; concurrency 2; training sampling seeds 3;
run 2 hours, admission 90 min, execution 119 min, finalization 60s. Smaller smoke
presets bound verification. TestServer presets: join 60s, action 300s, turns 2000,
invalid attempts 3, shutdown 30s, idle 0. Future model settings follow user plan.

## Policy and search

`DecisionInput` contains a serialized copy of the viewer's validated Observation,
Player-visible Events, history end position, typed metadata, optional same-revision
actor-scoped ActionOptions, and only pinned
catalog definitions whose IDs occur in own hand/learned list. Preserve missing
values and Dream's perceived IDs. Instance IDs are references, never numeric
features. Typed metadata describes syntax, never legal completed actions.
ActionOptions are authorized atomic availability hints. The default search
preset requires them; deliberate observation-only mode is versioned in Config.
Malformed/missing/stale/foreign requested hints stop before selection/Act, with
no automatic fallback. Raw endpoint data is bound to its actor path and checked
against that actor's own visible references; worker validation repeats the check.

`CandidateGenerator.generate(input) -> CandidateSet` is replaceable.
`Policy.reset(...)` and `Policy.select(input, candidates) -> PolicyDecision`
return index, version and optional logprob/value. Each participant uses a fresh
interpreter process with only its own input/history. Generate candidates inside
that process. Do not inherit parent private data or send URLs, replay, diagnostics,
GameSeed or other-player view. Worker termination is bounded by the owner.

Clean-bootstrap amendment reviewed READY on 2026-10-27: launch a fixed
`python -m gf_simulator_client.worker_bootstrap` with a minimal platform environment,
not multiprocessing's parent-main reimport/pickled factory. Bootstrap imports are
policy-free; scrub argv before importing the configured factory module/qualname.
Pass only policy settings/sampling seed/action settings and viewer input through
length-framed JSON stdin/stdout. Keep policy print output away from protocol;
stderr may be discarded when framed errors remain diagnostic. Never temporarily
mutate parent argv/environment. Validate importable factory references, bound
truncated/malformed frames and import/reset/select failure cleanup, and verify
module-import-time as well as factory/reset absence of sentinel management data.
Filesystem isolation against a hostile policy is outside this information contract.

Separate public space expansion/completion, public pruning, search order and
fixed public heuristic score in `gf_common.actions`. Enumerate and beam share
one space. Preserve order and different instances; deduplicate only exact typed
Action JSON. Hand references cannot repeat; learned miracles use separate public
repeat/payment rules. Finite expansion budget is mandatory even without a length
limit. A currently nonsubmittable prefix can remain extendable. Public pruning
requires cited public rules; uncertain effects are not structural errors.

The initial candidate space is an explicit versioned public subset, not all
engine-legal actions. Turn families: publicly eligible pray, ordered hand/learned
offense, nonweapon discard/sacrifice, publicly identified exchange/sell/buy with
configured resource allocations/targets. Defense: empty response and ordered
hand/learned defense. Buy decision: both boolean choices. Freeze exact supported
and omitted effect-specific cases with generator version/tests. Keep ability
handling explicit, publish omissions,
and measure rejection/empty-set rates. A conservative supported smoke subset is
allowed only with its restrictions recorded. Apply authorized ActionOptions
per action purpose: usable play/response elements, separate trade carriers,
targets and basic flags. A usable-play list is not a discard/sold-item mask.
Never use SequenceOptions, internal support matrices or rejection probes.

Generate children lazily; every attempt, including public-pruned attempts,
consumes budget. Reserve publicly supported basic actions first; insufficient
budget is an error. Basics survive beam/top-K. Track public and approximate
prunes separately. At cutoff return completed candidates only; empty is an
explicit failure, with no synthesized pass/prayer. CandidateSet records ordered
candidates, generator/scorer versions, action-config hash, attempts, public prunes,
approximate drops, latency and cutoff. Models score candidate contents, not index.

## Connection, concurrency and failure ownership

`Environment.run_group(EpisodeGroup) -> GroupResult` owns one repeated-seed
TestServer per group. Same-seed seat swaps share a process; different seeds use
different processes. Precompute game/group order, serialize Create per server,
create games only when needed, pass planned count as `--seed-use-limit`.
Use configurable global concurrency (2), a prebuilt binary, absolute `--config`,
explicit seed, run-owned data/log dirs, idle 0, stdin EOF shutdown. Drain logs
continuously, reap after startup failure/cancel too, escalate to bounded
terminate/kill as needed and record shutdown health. External URLs are never
stopped. Game slots release after replay/journal; server slots after log/reap.

Attestation is sticky and fail-closed: HTTP 200, `no-store`, exact closed schema,
strict types/domains, UUID v4, mode-dependent nullability. Repeated requires
explicit fixed-seed experiment and expected fingerprint. Fingerprint is `v1-`
plus SHA-256(domain `godfield-test-server-seed-fingerprint-v1` + canonical
16-byte seed). Managed CLI seed/startup info/HTTP must match. Verify initially,
before each Create, after Create before Join, on reconnect/recovery and before
resend. Compare all instance/effective fields; any failure/timeout/cancel poisons
the connection and forbids later sends. Here failure means attestation retrieval
or validation failure; an ordinary Join/Act transport failure may attempt the
separately bounded recovery below. These fields stay in run diagnostics.

Never retry Create. Join/Act transport retries: at most 2 after fresh successful
attestation, same command UUID/revision/payload. Rejection/conflict/unknown result
stops the game with diagnostics, no policy re-proposal and no fake win/loss.
Unsupported choices/versions fail explicitly. Use monotonic deadlines for requests,
worker calls and run processing, with separate finalization allowance.

Deadline amendment reviewed READY on 2026-10-27: configurable
`run.report_reserve_seconds` (default 1) and `server.cleanup_reserve_seconds`
(default 2) must be positive and sum to less than finalization allowance.
Work deadline is wall minus report reserve; Replay stops earlier by cleanup
reserve. Hash/schedule/statistics loops check between bounded chunks and all
controllable subprocess/request waits receive remaining time. Report expiry
produces only minimal incomplete diagnostics. Evaluate success after final
persistence; delayed report/manifest writes invalidate return/state, with
best-effort durable correction. Preserve known natural outcomes in failed runs.

Accepted debt: emergency kill/reap/thread join has a bounded grace beyond the
nominal wall (currently worker 0.5s reap + 0.1s thread join, server 1s reap).
Its use or any detected overrun is unsuccessful health and invalidates run,
comparison and data eligibility. Synchronous local atomic persistence and an
unresponsive OS/filesystem cannot provide a hard real-time guarantee; actual
duration and this limitation must be reported. Revisit with an independently
owned writer/report process or external watchdog if normal smoke overruns occur,
nonlocal filesystems are supported, or strict hard-deadline deployment is needed.
This exception does not permit new computation/admission after budget expiry.

Advisor disposition after two deadline repair cycles: zero ordinary grace on
worker abort, linked report hash/success validation for export, and independent
best-effort report/manifest invalidation attempts are required. The invariant is
that the CLI never returns success after detecting overrun, and any durable
failure or inconsistent completion evidence makes export ineligible. If both
invalidation writes fail after successful-looking artifacts were published,
durable state may remain stale. This is accepted filesystem debt, not an
unconditional durable-ineligibility guarantee; treat the failed run directory as
quarantined regardless of stale metadata. Revisit at the OS/I/O triggers above
or when recovery must remain authoritative after dual persistence failure.

Only the event API supplies policy history. Drain pages, keep opaque game/viewer
cursors; ignore events and unbound ActionOptions in Act responses. Fetch hints
through the dedicated actor endpoint paired with the same Observation revision.
Keep Replay as
unaltered diagnostic evidence only. Reconcile actor accepted journal actions and
terminal result against Player Replay. Never alter simulator persistence.
A participant transition spans its decision to its next decision or natural end.
Interrupted runs get no guessed final observation, return or bootstrap.

## Persistence and evaluation

Versioned participant JSONL journals retain exact input/history boundary,
ordered CandidateSet, selection, policy version/logprob/value, command identity,
acknowledgement and timing. Commit acceptance only on positive response or verified
idempotent recovery; interrupted requests remain pending. The run manifest keeps
management separate: revisions/dirty states, binary/config/input hashes,
schema/catalog/reward versions, seeds, policies and conditions. Finalize atomically,
retain incomplete runs, and report save failures separately from known outcome.

Persisted-provenance and ADR0003/0005 reviewed READY on 2026-10-27: journal schema 4
headers carry unique run ID, actual Game ID, episode ID and viewer. Manifest maps
these to schedule/pair/seed/ancestry/split and records independent health/replay
reconciliation. Export requires the finalized linked manifest (source journal's
grandparent `manifest.json`), validates identities and all semantic versions,
and rejects detached/running/conflicting sources or duplicate participant identity.
Derived schema 4 rows add run/Game IDs and split. Sidecar preserves source and
manifest hashes, individual config/code/model provenance, group identities and
declared splits. Across runs, union equal seed/common ancestry and reject split
conflicts; do not reassign. GameSeed and management fields remain sidecar-only.
Different compatible model/config identities may coexist with individual provenance.
Retain end/result metadata for zero-decision/interrupted episodes in the sidecar.
Eligibility checks pending acknowledgements, episode/group health, failures and
Replay reconciliation, rather than trusting run status alone. Known natural
outcomes remain intact even for ineligible runs. Schema 1/2/3 historical artifacts
are retained and explicitly rejected, not reinterpreted as schema 4.
Run manifest schema 3, feature `participant-json-v2`, generator
`public-atomic-subset-v3`, hint mode and `policy_input_version` identify this
authorized input boundary. Journal input stores the delivered hint snapshot;
reader/export verify owner/revision/references/presence against header mode and
linked config. No historical availability is inferred or regenerated.
Generator v3 adds only hinted own-hand `atkBy2xMP` weapon singletons, outside
multi-element prefix search, with existing target/budget guards. Its scorer
remains approximate for absent printed attack values. Historical generator-v2
sources are explicitly unsupported by current readers; formats do not change.
Reviewed normative document SHA-256 values identify the user's changed contract
even when Simulator HEAD is unchanged. Optional local source provenance records
actual document hashes and dirty state; standalone wheels retain the reviewed
declaration without requiring a checkout or claiming verified binary provenance.
`end_projection_version=simulator-closure-v1` is pinned in run, journal header
and derived provenance. A diagnostic `simulator_abort_reason` is recorded only
after matching complete validated closure identities from both reconciled Player
Replays. Exact `turn_limit_exceeded` becomes nonterminal `truncated`; other abort
causes remain `aborted`. Raw acknowledgements/Replays and local failures remain
intact, and known natural results are never overridden. No synthetic terminal
transition, reward, final observation or bootstrap is created. Reasons stay in
end/result diagnostics and sidecars, not policy inputs/feature rows or Parquet
schema metadata. See ADR0003 for compatibility and evidence requirements.

Optional deterministic Parquet conversion rejects incompatible/mixed versions,
preserves candidate order and original input, records source hashes and derived
schema/feature/reward versions. Both views, same-seed repeats and common fork
ancestry belong to the same split. Natural terminal reward: win +1, loss -1,
draw 0; intermediate 0. Interruptions/rejections receive no terminal labels.

Evaluation predefines seed/seat-swap pairs shared by all model comparisons.
Report W/D/L with explicit natural-end denominators, completion, truncations,
aborts, unsupported contracts, rejections, transport/infrastructure/save/process
failures separately. Seed-cluster bootstrap 95% intervals cluster both seats and
repeated seeds. Incomplete pairs cannot count as successful comparisons. Include
first-seat/opponent breakdown and measured search/candidate counts, latency
p50/p95, throughput, wall time, memory/artifact size (null when unmeasured).
Final test size is fixed from validation only: pilot 64 pairs, half-width .05,
bounds 256–2048; never select from test outcomes.

CLI: `smoke`, `collect`, `evaluate`, `benchmark`, config validation, journal
export and Phase 4 `train`. Learning feasibility is separate:
fixed-policy pilot 64 games × 3 sampling seeds, all natural and within budget.
Failure leaves the gate unfulfilled and stops dependent learning experiments;
never retain only successful episodes or manufacture bootstrap labels.

## LightGBM training and inference

`gf_models.lightgbm` owns stdlib-only typed model settings, NumPy feature encoding,
native CPU fit/predict and inference artifact schema 2. The encoder `participant-action-v1`
supports current observation and bounded public-event aggregates. Own visible
definitions, ordered hand/learned references, missingness and same-revision atomic
hints are encoded; candidate position, numeric instance IDs and management
metadata are excluded. Each candidate is independently scored. Oversized
sequences fail validation instead of being silently truncated.

`gf_training.q_data` shares the journal exporter's validated schema-4 projection.
It accepts whole finalized healthy natural runs, creates one final-return target
only for each positively acknowledged selected action, and validates train,
validation and pilot seed/ancestry unions together. Feasibility recomputes the
pilot's fixed 64-game × three-policy-seed schedule and original budget; a sidecar
eligibility flag alone cannot authorize fit. Decision target/overshoot and all
source rows are checked before the native process starts. The failed current
pilot remains ineligible; no production fit or model-quality claim is made.

CLI selects model/training presets with the existing merge rules. It composes
safe runtime bindings; exact inference-manifest hashes distinguish checkpoints
of the same model, including swapped seats. Parent preflight validates only
metadata, hashes and the linked completed training run. Native loading occurs
in an owned clean participant interpreter before any Simulator listener starts.
Workers recheck the pinned artifact and use the shared encoder. Baseline help,
config loading and imports require no ML library.

Each fresh training seed has its own original configured budget. Recorded source
collection time is deduplicated and charged with current ingestion, encoding,
provenance hashing, fit, saving and cleanup. Shared feasibility validation is
identified separately. Current-fit Git/package/module provenance is distinct
from source-run provenance. Request and numeric-input digests are checked by the
native child. The child receives numeric arrays and closed model parameters;
source control metadata stays in the parent training record. Native model RNG
parameters in LightGBM text are allowed; Simulator GameSeed and source paths are
not native parameters or features.

The native fit subprocess self-validates an unready staging model. Parent drains
bounded logs and reaps it before persisting completed `training.json`, then
publishes checkpoint readiness last. Failure, cancellation, cleanup issues and
detected overruns independently invalidate run and checkpoint. Formal inference
requires both the ready/hash-compatible checkpoint and its healthy completed
training record; either durable failure blocks use. Persisted cost fields are
explicit pre-write snapshots; returned completion cost includes final writes and
corrections. Concurrent/repeated worker close and cancellation join the same
owned cleanup task without resetting its first deadline, preserving the original
operation error until process, thread and streams have been reclaimed.

Artifacts contain native `model.txt` and closed inference JSON, without pickles.
Schema 2 requires simulator_contract_revision equal to the supported pin; old
schema-1 artifacts and wrong/missing identities are explicitly unsupported before
native loading, including toy loads. Training completion stays schema 1.
Model, encoder, runtime, catalog, input/reward/projection and candidate-generation
identities must match. `purpose=toy` permits explicit synthetic component tests
only; formal commands reject it before startup. Only fresh training is supported;
resume is rejected before work. Existing OS/filesystem blocking and dual-write
failure debt remains explicit; portable inference export and broader native
runtime compatibility are deferred to Phase 9. No new dependency is introduced.

## Verification and acceptance

Unit/HTTP fixture tests cover strict config precedence/path origins/hashes;
sticky attestation including cancellation; retry identities; participant input
serialization; finite lazy search budgets/order/basics/prefixes/enum-beam agreement;
journal roundtrip/export/split/compatibility; outcome-health separation and
seed-cluster statistics. Opt-in real-binary integration covers repeated-create
independence, baseline natural termination, replay reconciliation and cleanup.

Checks: `uv run pytest`, `uv run ruff check .`, `uv run mypy src`; wheel installed
in a separate Python 3.14 environment, baseline without ML imports. Verify CPU
compatibility per implemented extra. Record exact commands and PASS/FAIL/NOT RUN
in `docs/verification/`, with phase gates and measured limitations.
Independent correctness and architecture conformance reviews precede acceptance.
Root owns docs/project presets/acceptance; one implementer owns Python/tests;
other agents are read-only. No agent edits the simulator. Root owns the user's
authorized local commits; publishing remains separately authorized work.
