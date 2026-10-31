# GodFieldBots architecture

## Status and scope

[ADR0022](adr/0022-gru-transformer-ppo-plan.md) implements GRU/Transformer PPO
with full own-decision history (up to1024), participant-local bounded hidden/KV
caches, and Config sequence_length32(default,1..1024) for detached-prefix TBPTT.
Endpoint loss/episode GAE, source visibility, safe artifacts, native model-capability
preflight and normal-boundary resume preserve the existing ownership. Source
reviews are READY;891Python/20separateCUDA/93Rust tests and old-data/wheel checks
pass. Corrected2GiB/16MiB typed-reader real smokes complete128natural games with
exact continuous/resume states; earlier8GiB legacy smokes remain preliminary.
Sixty fixed-input comparisons report cached forward benefits and CPU-faster update
on the small fixture, with no long-window/strength claim. Both CLI presets and the
tested three-model native wheel are available. Final acceptance is recorded in
[sequence verification](verification/sequence-ppo.md); formal campaigns and final
test remain separate.

[ADR0021](adr/0021-measured-single-worker-optimization.md) Stage1 removes only
the typed Journal's separate reserved-key scan, using existing closed DTO/key
decoding. Generic Value retains its guard. Independent design/source/result
reviews,92Rust/11standalone/106native Python tests and five full source/index/
savedPPO proofs pass. All12same-campaign commands pass; one-worker whole median
22.554->21.609s (4.19%), five paired wins and lower reported CPU. Peak RSS median
rises9.92MiB, maximum1.77MiB; unchanged managed1531.02MiB/scratch1770.07MiB and
Config budgets. Root accepts that measured tradeoff; ranges overlap and other
worker speeds are unmeasured. A separate bounded Record-capacity candidate passes
all proofs but is3.55%slower, losing all5pairs; independent audits support rejection.
Only its source changes are removed, exactStage1restored, and tested38ad8a23/
sourcef98b0312wheel installed with verified2ed57rollback copy/CLI/model checks.
Duplicate-Schema compare-deserialization and typed
Replay are deferred after notifying the user of their larger implementation
scope. [Verification](verification/single-worker-optimization.md).

The preceding [single-worker allocation analysis](verification/single-worker-bottleneck-analysis.md)
kept productionfaf71b/main2ed57 unchanged at capture time. Seven exact-output commands and current
CPU profile pass independent safety, fidelity, arithmetic and report reviews.
The private scan observes140,148,348allocation-producing requests; Replay has
71.28%of requests (not CPU share), and Schema scopes18,550,112including terminal
observations. Record/Event capacity growth is measured, but no reserve/arena/
parser/format optimization is adopted. Counter-request, CPU-suffix, liveness,
overhead and scope limitations remain explicit in the verification record.

[ADR0020](adr/0020-owned-episode-pipeline-pool.md) accepts the design for one
owned scoped Rayon pool per scan, with complete worker-local episode processing
and bounded ordered coordinator commits. Source/failure reviews,89Rust tests,
826Python tests and both fixed recipes/index/savedPPO1/4proofs pass. All48same-
campaign before/after commands and independent arithmetic audit pass: whole
medians1/2/4/8before22.721/19.641/16.310/14.406s, after22.688/14.857/10.906/8.794s.
Root retains the combined change: no meaningful1worker gain,24.36/33.14/38.96%
at2/4/8with explicit reportedCPU and memory costs. After8maximum RSS512.72MiB,
managed1677.34MiB under unchanged2GiBpolicy; defaults stay unchanged. Exact
tested2ed57a31/sourcefaf71b33wheel is installed with verified97f1rollback backup.
[Verification](verification/episode-pipeline-pool.md) keeps phase-version, CPU-
accounting, sampling, provenance and failed-cleanup limits explicit.

The [final-runtime scaling/function investigation](verification/final-preprocessing-scaling-and-profile.md)
compares unchanged source d5090c41/release-mimalloc97f1at1/2/4/8workers:
whole medians22.599/19.242/15.907/14.362s, maximum self RSS346/396/432/507MiB,
all24original-output/source/lifecycle/ledger checks and independent audit PASS.
Eight is9.71%faster than four with75.35MiBmore maximum RSS; defaults stay unchanged.
Separate source-identical debug2captures and versioned observed-suffix analysis
locate validation/presence/event checks and Replay/Value-tree/allocator destruction.
Filtered sampled CPU, incomplete coverage and overlapping owners are explicit;
no pool, validation shortcut, source-format or runtime change is adopted here.
Design, harness and final correctness/conformance reviews are READY.

[ADR0019](adr/0019-sequential-preprocessing-optimization.md) Stage1 adopts one-pass
typed history aggregation. All numeric/index/PPO proofs and826Python tests pass;
independent reviews accept12controlled commands: wholewall20.60->17.69s (14.15%),
feature/spooling4.84->2.01s in actualmedian trials. Maximum RSS431->457MiB, median
peak420->422MiB; Config budgets unchanged. Later stages remain separately gated.
[Verification](verification/sequential-preprocessing-optimization.md).
Stage2 retains a small local removal of adjacent history checks:16.972->16.729s
(1.43%), all five alternating paired/target-phase wins, exact outputs and guards.
Whole ranges overlap; no generalized or physical CPU improvement is claimed.
Stage3a shares equal, strictly decoded schema trees within each borrowed Input.
Both fields still deserialize normally; only shared immutable shape validation
and retained storage are reused. Under the user-authorized competing workload,
whole median19.770->19.094s (3.42%), with five paired wins; median peak RSS
428.73->394.38MiB. All output proofs,826Python tests and independent reviews pass.
This is evidence under shared load; uncontended performance remains unmeasured.
Stage5a retains bounded participant-file parse tasks, with the same read settings,
worker caps, raw owners and ordered validation. Shared-load whole median improves
19.261->18.735s (2.73%); parse median4.409->3.447s (21.82%). Reported CPU rises
1.54%and median peak RSS18.12MiB; maximum/retained memory stays similar or falls.
All826distinct Python cases,81Rust tests and five real-data proofs pass; independent
reviews accept this explicit tradeoff. No persistent pool or line streaming is
introduced. Stage4 persisted history migration is conditionally skipped for the
immutable current corpus; future versioned formats/prefix reuse need separate gates.
After all comparisons, the exact accepted release/mimalloc wheel97f1b509/source
d5090c41is installed into Bots.venv with no dependency changes. The historical
32e3ad68package/metadata is backed up; fresh identity/capability and optional-free
CLI startup checks pass. Reader/worker Config defaults and resume guards remain.

A subsequent [root-cause investigation](verification/preprocessing-root-cause.md)
keeps production unchanged and separates graph/task supply, repeated input/history
work, clock-check counts and scoped I/O from proposed pool/streaming remedies.
All11diagnostic captures preserve original outputs. CPU-time API sensitivity and
partial instrumentation scopes remain explicit; no optimization is adopted.

[ADR0018](adr/0018-bounded-parallel-borrowed-journal-parsing.md) accepts the design
for bounded scoped-worker Journal parsing with ordered consumption and unchanged
Config budgets. Source reviews,826Python tests, complete corpus/index and saved
PPO parity pass. All48controlled measurements and independent arithmetic audit
pass: median wall25.43/23.19/19.79/18.02s at1/2/4/8workers;4/8reduce16.14%/20.37%
versus same-worker serial parsing. The1-worker cost rises1.14%; peak RSS grows
to378/382/420/502MiB under the unchanged2GiB soft budget. Defaults/main wheel
remain unchanged. [Verification](verification/parallel-borrowed-journal.md) also
separates dispatch/join from validation and documents short-thread profile limits.

A subsequent [isolated8/16/32parse experiment](verification/journal-parse-scaling.md)
holds read workers4 and resident episodes32 fixed. Parse medians2.264/1.926/1.859s
show diminishing gain:16to32 saves3.45%, with higher CPU and command peak RSS.
The private release harness preserves the parser/scheduler and does not introduce
an independent production parse-workers Config or an end-to-end speed claim.

[ADR0017](adr/0017-combined-borrowed-journal-practicality.md) implements a verified
opt-in mimalloc/borrowed-Journal candidate. Scoped ownership, complete typed
semantic/feature parity, saved PPO inputs, lifecycle checks and all48actual-path
measurements pass independent acceptance. Medians25.34/24.03s at1/4workers reduce
21.35%/12.26% versus mimalloc+legacy; kernel RSS maxima349/368MiB. Production
defaults remain unchanged. Evidence: [combined practicality](verification/combined-borrowed-journal.md).
Current [stage/function profiles](verification/combined-borrowed-profiling.md)
separate the ordered Journal parse/validation stage (about10.15s) from decoder
wait/merge, and retain release-symbols CPU traces with explicit unwind limits.

[ADR0016](adr/0016-staged-preprocessing-optimization.md) accepts isolated
allocator and typed-Journal experiments (stages0–2); production adoption remains
a later decision. Independent design/code/measurement acceptance is complete:
mimalloc reduces whole preparation28.7%/10.3% at1/4workers; borrowed typed Journal
lifecycle reduces32.1% in the standalone experiment, with0newreal-data rejections.
The latter does not prove production feature-array or end-to-end equivalence.
Measurements and resource tradeoffs are tracked in
[staged optimization verification](verification/staged-preprocessing-optimization.md).

[ADR0015](adr/0015-complete-wall-attribution-and-worker-scaling.md) adds reconciled
exclusive coordinator/Python wall ledgers and decoder-worker observations, with
closed1/2/4/8 support. Design/correctness/conformance reviews and runtime acceptance
are complete. Three-rotation whole-workload medians49.21/36.85/32.32/31.18s use
unchanged2GiB/16MiB budgets and identical outputs. Presets remain1/1; four workers
balance time/resources and eight is fastest in this corpus. Exclusive scan and
Python construction ledgers reconcile exactly; temporary release accounts for
about6s of single-worker scan time. Legacy summed worker times remain nested
diagnostics, not a wall partition. Logical slots/active job overlap are distinct
from persistent threads/CPU utilization. Fork/window/join, ordered serial feature
processing, shared quotas and publication ownership remain unchanged. Evidence:
[wall attribution and scaling](verification/wall-attribution-and-worker-scaling.md).
An unchanged-source release/symbols diagnostic now adds
[function-level CPU profiles](verification/function-profiling.md). It identifies
JSON allocation and document-release allocator costs with caller-filtered stacks;
sampled CPU estimates are distinct from elapsed/I/O-wait measurements.

[ADR0014](adr/0014-bulk-numeric-io-and-episode-lookups.md) adopts bounded bulk
f64/i64 I/O and per-collection episode indexes. Design and code-conformance review
are accepted; whole769Python tests, both frozen recipes/index and saved PPO
ingestion PASS. Whole-preparation median46.43->46.05s is less than1% difference;
current16/64-game collections do not show a substantial overall speed gain.
A small private helper uses safe
bytemuck views on little-endian hosts and a portable bounded fallback; destination
buffers remain initialized and leased. Local borrowed indexes replace repeated
manifest/schedule/health/ordinal scans, with original order and health gates
preserved. Evidence: [bulk I/O verification](verification/bulk-numeric-io-and-episode-lookups.md).

[ADR0013](adr/0013-standard-json-limits-and-sort-contracts.md) further simplifies
JSON parsing and records sort tie contracts. Design and independent code reviews
are accepted; whole769Python tests, both frozen recipes/index and saved PPO
ingestion PASS. Paired median50.26->46.72s, about7.05% reduction. Standard Serde recursion
replaces the custom990-depth and global4300-digit passes/setting. Only the actual
reserved-number-member guard remains. Catalog lookup now borrows Serde's map
directly. Current evidence is in
[JSON and sort verification](verification/json-simplification-and-sorts.md).

[ADR0012](adr/0012-serde-json-and-coarse-resource-admission.md) records the
user-authorized production Serde migration and generic-code cleanup. Its design
and final independent correctness/conformance reviews are accepted. Full tests,
both frozen feature recipes/index and saved PPO ingestion pass; three paired
preparations improve median94.25s to50.81s (1.855x), with RSS376.1->404.6MiB.
JSON uses Serde Value/Map/Number and coarse document/file estimates;
numeric export leases, source/learning boundaries and cleanup ownership remain.
The custom parser/AST, open-address source index, heapsort copies and fine decoder
profiler are removed from the new source. Current acceptance evidence is tracked
in [Serde adoption verification](verification/serde-json-adoption.md).

[ADR0011](adr/0011-native-preprocessing.md) specifies the user-approved migration
of offline preprocessing and PPO post-collection ingestion to a common Rust core.
Its initial implementation and correctness/conformance reviews were accepted;
ADR0012 supersedes its JSON allocation/parser and fine-polling decisions.
Full-corpus/index equivalence, installed wheels,751Python/31Rust tests, three
paired preparations (2.682x), an exact reference fit and real CUDA pause/resume
checks pass. Python remains the learning authority. Target PPO/fixed-LightGBM
presets select Rust; dataset CLI selects a native preprocessing preset explicitly.
Live participant worker migration remains a separate completion unit. Measured
scratch/write traffic increases; see [native verification](verification/native-preprocessing.md).

[ADR0010](adr/0010-budgeted-preparation-and-encoding.md) specifies budgeted
offline preparation and per-decision feature reuse. Implementation, independent
reviews, paired wall/CPU/memory/storage measurements and exact output equivalence
are accepted. The two remaining fits are complete; the prior four fits and frozen
dataset remain unchanged. Its historical native feasibility screen retained Python;
ADR0011 supersedes that implementation choice after broader native verification.
See the
[performance verification record](verification/budgeted-preparation.md).

[ADR0009](adr/0009-catalog-v2-simulator-adoption.md) adopts the new Simulator
Magical Stick payment/zero-attack contract and catalog v2. Independent design
is READY; implementation verification and fresh experiments have advanced through
three completed PPO parents, final validation, frozen data and all six fits.
Current evidence is [catalog-v2 adoption](verification/catalog-v2-adoption.md).
The earlier blocked experiment below is historical and remains wholly ineligible.
Existing v3 singleton candidate forms, encoder signatures and JSON schemas
remain unchanged; revised revision/catalog identities reject old formal sources
and checkpoints. The frozen old reader retains historical pilot verification.
Catalog retrieval uses the pinned version; sticky connection failure is checked
again after a probe response before publishing cached readiness.

The user's 2026-10-28 CUDA MLP PPO/generation/fixed-data plan is accepted in
[ADR0008](adr/0008-mlp-ppo-generations-and-fixed-data.md), with independent
design READY. It extends the baseline below with an owned numeric learning
child, CPU participant inference, whole-run offline eligibility and original-
deadline batch-boundary pause/resume. Independent design, correctness and final
implementation conformance reviews are READY. Component verification has passed.
The real two-update/explicit-resume smoke passes with exact continuation.
The formal comparison is blocked after a failed seed-17 parent by the
[zero-MP dynamic-weapon target contract](issues/0004-zero-mp-dynamic-weapon-target-contract.md).
That entire parent remains ineligible; later seeds/dataset/real LightGBM fits
are not run. Current
evidence is in [MLP PPO verification](verification/mlp-ppo.md). Historical
Phase 4 and pilot results below retain their original scope and identities.

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
below now passes provisional feasibility. The original Phases 5–9 were outside
that acceptance; ADR0008 now implements the MLP PPO/generation/fixed-data subset.
ADR0022 implements GRU/Transformer after that acceptance; final-test evaluation
remains future work. See the
[current verification](verification/action-options.md).
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

Simulator pin: `ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6`; HTTP `/v1`,
`true-match-1v1-v1`, Observation `"1"`, Action metadata `1`, TestServer info `1`.
Catalog: `public-ja-v2`, schema `1`, locale `ja`, 296 definitions,
SHA-256 `481ddfb0c068a1d2aaaf2dbcb6a5b4bd04396dfcc5c995e14e47b71222b6a51d`.
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
| `gf_training` | journals/projection, split/feasibility, owned numeric training, generation lineage and frozen datasets | common, model adapters, optional PyArrow |
| `gf_evaluation` | paired scheduling, aggregation and costs | common |
| `gf_models` | participant-local Random/heuristic/LightGBM/MLP/GRU/Transformer, common encoder/config and artifacts | common, lazy extras |
| `gf_cli` | strict config, composition, commands, manifest | application packages |
| `gf_native` (optional Rust extension) | Serde offline parsing/structural validation, fixed feature kernels, typed staging/index columns and numeric capacity/document-admission leases | Rust crates, Serde, PyO3, rust-numpy; no upper Python package or learning dependency |

The existing `godfieldbots` entry point becomes a compatibility wrapper.
Common never imports upper layers. Models never handle HTTP or management
configuration. CLI constructs concrete policies/environments for evaluation.
Runtime is CPython 3.14, GIL enabled, uv; baseline dependencies httpx/NumPy.
Extras are `data` (PyArrow), `lightgbm`, mutually exclusive `torch` (CPU) and
`torch-cuda` (cu132); dev is pytest/ruff/mypy. NumPy is pinned to 2.5.4, Torch to
2.14.1 and torchvision to 0.29.1. PyTorch indexes are explicit; baseline
dependencies continue to resolve from PyPI.
Optional libraries must not load for baseline or CLI help.

The separate `godfieldbots-native` uv workspace member uses Maturin1.15.0,
PyO3 0.30.1, rust-numpy0.30.0 and ndarray0.17.2, pinned in its build configuration
and Cargo.lock. ADR0012 adds serde1.0.229/serde_json1.0.151 with arbitrary-precision
numbers; ADR0014 adds bytemuck1.25.2 for safe typed numerical byte views. The root
package retains uv_build and exposes an optional native
extra. Runtime loading is lazy and identifies the actual loaded extension binary,
ABI, build/lock/source digests, parser/resource contracts and fixed recipe. Rust failures never dispatch to
the Python reference backend. ADR0011's target production presets are enabled;
optional diagnostic builds remain isolated from the production extension.

For ADR0011's offline and post-collection paths, Python owns source locks,
authorization, experiment decisions, learning semantics and publication. Rust
returns readonly NumPy-backed buffers plus compact structural evidence through
one data call per chunk. The actual array base owns storage and budget leases
through surviving views. A separate precreated cancellation handle and one
owned FFI executor cover long lifecycle/data operations; cancellation joins work
before releasing locks. Recursive persistent state retires on a prestarted,
budgeted safe-stack Rust thread. Offline owners finalize after fit-child reap;
online batch owners finalize before optimizer dispatch and parent commit. Learning
rewards/GAE/normalization remain Python-owned. Rust supplies ordered index columns
and hashes; Python invokes existing native Arrow/Parquet I/O on typed buffers.
Live participant workers and their information boundary are unchanged.

The current Serde JSON parser contract is `serde-json-v2`; document admission is
`json-document-32x-plus64kib-v1`: estimated32*raw bytes+64KiB before parsing, with
one Arc reservation shared by a journal file's records and retained by source
documents. Ordinary JSON allocations and clones are not individually tracked;
these are soft admission estimates, not RSS quotas. Numeric/NumPy owner capacity
leases still track retained buffers. Coarse metadata-copy counters exclude Serde
heap allocation/copy; OS RSS/PSS is measured separately.

Serde owns JSON grammar/string/number parsing and its default recursion limit.
A narrowly named reserved-member guard rejects escaped
`$serde_json::private::Number` member keys; identical value strings are allowed.
There is no custom depth counter or global postparse depth/digit traversal.
The old native_json_depth field is no longer a current setting; historical
effective_config remains data. Integer range checks remain at actual domain
fields. Scalar Unicode/standard JSON is required; bare NaN/Infinity
and lone surrogates fail early. Python canonical number spelling and metadata
numeric equality remain small compatibility helpers so genuine historical source
hashes and feature bytes stay stable. Cancellation is checked at bounded scans,
file/record and operation boundaries; a parse/sort may finish before cancellation
is observed, and no late result may publish. Safe deep-state retirement remains.

Native sort wrappers name their unstable behavior explicitly. Valid sort keys
are unique, duplicates reject before publication, or tied payloads are
interchangeable membership/lineage values; no current native call requires
stability. Python retains its standard stable sort/sorted. Candidate/journal
sequences are not reordered by these helper sorts. Any future meaningful tie
requires a caller-specific stable sort or explicit tie key.

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

`gf_models.config` and `gf_models.encoder` own model-neutral closed settings and
the pure participant/candidate encoder; legacy LightGBM paths re-export them.
`gf_models.lightgbm` owns its model-specific settings, native CPU fit/predict and
inference artifact schema 2. The encoder `participant-action-v1`
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
native loading, including toy loads. Inclusive-generation training completion
stays schema 1; fixed-data training uses schema 2 as described below.
Model, encoder, runtime, catalog, input/reward/projection and candidate-generation
identities must match. `purpose=toy` permits explicit synthetic component tests
only; formal commands reject it before startup. LightGBM training is fresh only;
its resume is rejected before work. Existing OS/filesystem blocking and dual-write
failure debt remains explicit; portable inference export and broader native
runtime compatibility are deferred to Phase 9. LightGBM adds no runtime dependency.

## MLP PPO, generations and fixed datasets

[ADR0008](adr/0008-mlp-ppo-generations-and-fixed-data.md) pins the complete v1
recipe. `gf_models.mlp` owns CPU categorical inference and public inference
artifacts. Actor uses the common public-aggregate encoder with CPU signed log1p
and FP32 conversion; critic excludes candidate features by explicit name prefix.
Both have two 128-wide ReLU layers with CPU orthogonal initialization. Config
dispatch rejects cross-model settings before I/O. Inference device is CPU;
learning device is explicitly `cuda:0` or the separate CPU preset.

`gf_training.ppo` owns GAE/rollout projection; `ppo_numeric` owns deterministic
IEEE FP32 Torch/Adam update math. A single clean `ppo_worker` child owns numerical state and GPU for each
invocation. The coordinator never imports Torch. Numeric IPC carries closed
requests and hashed typed NumPy shards bounded to 64 MiB. Whole-batch frozen
CPU/saved/device parity precedes any Adam mutation. Logical minibatches have
128 decisions and accumulate padded microbatches of at most 4096 candidate
slots, clipping/stepping once per logical minibatch. CUDA synchronization
precedes finite metric checks, CPU-array persistence and acknowledgement.
Unavailable devices, mismatch, OOM or asynchronous errors fail without fallback.

`gf_training.collection` defines the injected `CollectionPlan` and
`FinalizedCollection` values. CLI composes the existing environment, journals,
Replay checks and finalizer; training has no CLI/HTTP dependency. Each admitted
batch fixes eight train-namespace seed pairs, swapping learner seats explicitly:
two Random, two heuristic and four uniformly selected periodic snapshot pairs.
CPU inference processes retain only participant-visible inputs. Learner role
never comes from comparing policy IDs; generation-zero self-play is unambiguous.
New collection manifest schema 4 links parent/run/batch/generation and learner
seats. Schema 3 has an explicit legacy serializer without learner metadata and
retains the existing 192-game pilot proof. Journal/derived schema 4 is unchanged.

`gf_training.lineage` owns run flock, immutable generation/update records,
separate latest/state and deletion receipts. Generation zero, every tenth and
latest inference/resume arrays remain; all inference manifests and generation
hashes remain even after nonperiodic array deletion. Checkpoints use closed
JSON and preallocation-validated typed NumPy arrays, never pickle. Resume
includes Adam, CPU/CUDA/shuffle and schedule/opponent RNG, counters and pool.
The lifetime lock protects validation, commit and retention. General formal
MLP bindings require a complete healthy parent's final generation; owned PPO
collection constructs separate safe bindings to committed current/pool policies.

`ppo_run` applies original absolute 7200/5400/7140-second wall/admission/execution
deadlines, 1000 updates, 10 GiB soft storage and 1 GiB admission reservation.
Already-admitted queued groups use execution cutoff. Any batch/update/save/
cleanup failure invalidates the whole parent. At pre-batch limits, at least one
update and healthy drained work permit completion. Explicit absolute
`--pause-after-update` stops only at a committed boundary, after child reap.
Only paused runs resume, with unchanged recorded code bytes/runtime/config/
device/hardware identities and original deadlines; clock rollback and concurrent
ownership are rejected. Parent identity freezes actual Simulator binary and
server-TOML bytes plus configured source/spec snapshots. Each collection is
checked against that identity, including final-policy validation. The CUDA
learner records supported driver API version; this is not an exact NVIDIA
driver-build identity. Cross-build determinism remains unverified.
Capacity is a measured soft limit; existing OS/filesystem debt still applies.

`fixed_data` revalidates all original journal/Replay/report/parent/chain evidence,
requires ordered complete parents at seeds 17/29/43, selects every learner train
action and only the final-policy learner validation view, and freezes source
hashes/counts/generation ranges/splits/costs in a dataset manifest. Final-policy
validation uses the existing 32 seed pairs, half Random and half heuristic,
64 games per parent. Failed parents invalidate every earlier batch for offline
use. Shared projection and source ingestion default to complete-parent
qualification; only an exact live-lock `OnlinePPOContext` authorizes the running
coordinator's current frozen batch. Legacy inclusive LightGBM accepts schema 3
sources only. `dataset build`, `inspect` and regenerable Parquet `index` use the manifest
as authority; indexes/eligibility flags never authorize fit independently.
LightGBM labels are accepted actions and natural final returns, never PPO values.

Fixed-data LightGBM records schema 2 and includes original-source revalidation,
encoding, fit, save and cleanup within each 7200s fit budget. Shared generation
cost has one dataset identity and is reported separately from fit charges; it
is deduplicated across recipes/seeds. Legacy inclusive-generation schema 1 and
its pilot/decision-budget requirements remain. No equal-total-cost PPO claim
or final-test result follows from this comparison.

Inference timing is measured around policy.select inside the worker after search,
then sent as diagnostic metadata and reported separately from whole-worker
journal selection timing. It never enters PolicyDecision, features or journal
schema. CPU/GPU memory and elapsed metrics state their measured scope.

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
