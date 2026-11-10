# Group Pack implementation ledger

Source intent: owner plan supplied 2026-11-10. Baseline: 83aa67e7707b267217247afbc4c3efcf14204a94.

- [x] Read current architecture, relevant v2 persistence/Source ADR sections,
  Python 3.14 storage availability and Simulator normative input/launch contracts.
- [x] Draft physical contract, custody/failure boundaries and staged gates.
- [x] Foundation architecture review READY: exact namespace/loan semantics,
  Writer retirement before owned descriptor publication, IPC/seal/catalog closure.
- [x] Implement Pack, Writer/SQLite/catalog, sealing and ArtifactSource with tests.
- [x] Independent correctness and architecture conformance of foundation.
- [x] Review and implement memory/RAM collection plus explicit Config.
- [ ] Review and implement all Python/Rust/PPO/resume/final64/Dataset/Tree readers.
- [ ] Independent full integration reviews and resolved blocking findings.
- [ ] Full checks, bounded runtime/CUDA/wheel/legacy verification.
- [ ] Alternating same-real-data/same-HDD and concurrent workload comparison.
- [ ] Coherent verified local commits and final acceptance report.

Ruling: execute the supplied plan without a new approval interview — the user
explicitly requested implementation in fresh context; detailed boundary choices
remain subject to the repository's architecture review rather than repeated
user permission.

Ruling: use a dedicated branch in the existing shared checkout and retain all
pre-existing untracked artifacts — existing environment and Simulator sibling
paths are source-pinned here; no unrelated files are staged or removed.

Preflight:

| Tasks/interface | Produced/consumed | Disposition |
| --- | --- | --- |
| Foundation internally | Pack → Writer → DB → descriptor → ArtifactSource | Closed schema, bytes, deadlines and loans pinned in design; architecture READY |
| 1 → 2 | StorageSettings/Writer → runtime coordinator | No consumer until reviewed memory/RAM account and explicit Config |
| 1 → 3 | ArtifactSource → Python/native original ownership | Stable logical proof and actual physical custody gate before integration |
| 2 → 3 | Sealed collections → PPO/final64/Dataset | Pending physical state cannot imply healthy logical Source |
| 1/2/3 → 4 | Tested interfaces → complete comparison | Component passes cannot substitute for all-path acceptance |
| Task 1 self | bounded encode/write/seal/read tests | No production consumers or default/config mutation |
| Task 2 self | in-memory output and RAM logs | Exact file ownership to be assigned after design gate |
| Task 3 self | all original readers | Explicit lineage/native proof changes required before code |
| Task 4 self | verification and real same-HDD comparison | Unrun cases must be identified; no synthetic strength/throughput claim |

Runtime design READY (2026-11-10) after joint decode budgeting, actual sync-worker
retirement, typed logical sidecar evidence, composite inode-owned publication and
explicit early command guards. The later narrow pending-predecessor/FD-invalidation
delta also passes architecture review. Runtime code awaits foundation acceptance.

Foundation implementer reports codec/Writer/Source failure tests progressing;
no production consumer is changed. Root's clean baseline command
`.venv/bin/python -m pytest -q tests/test_config.py tests/test_card_instance_collection_publication.py`
passed36checks in0.50s. Implementation evidence still awaits independent review.

Ruling: revoke through SAME held/verified recovery FD rather than conditional
pathname unlink — no atomic stat→unlink inode predicate exists in the stdlib.
Invalid diagnostic evidence remains; foreign replacements are not modified by
FD writes. Uncertain close/invalidation retains the actual owner and first error.

Task1 independent review: correctness BLOCKED (seal DB substitution, unadmitted
namespace allocation, constructor/helper primary-error cleanup); conformance
BLOCKED (same plus stale-FD reuse and simultaneous first-use catalog race).
Reports: `/tmp/gfb-group-pack-task1-{correctness,conformance}.md`.
Root100-test/Ruff/mypy pass does not resolve these untested failures.

Task1 fix round1 assigned: one-attempt FD state/uncertainty everywhere in new
modules; streaming/admitted namespace enumeration; BEGIN IMMEDIATE catalog
schema/version/update; child+parent held original DB custody and expected seal
through retirement/publication. Architecture confirms held-inode remedy READY
without schema/IPC changes. Save pre-fix snapshots in
`/tmp/gfb-group-pack-task1-before-fix/` for scoped delta reviews.

Task1 fix round1 reviewed: F1–F4 and both coverage gaps ADDRESSED. Architecture
conformance READY. Root121affected/legacy tests PASS in5.32s; scoped Ruff/mypy
PASS. Correctness found one introduced P2: Writer's now-retained128MiB base plus
publication inspection's independent full allowance can over-admit71350bytes
at a256MiB limit. Fix round2 assigns remaining-joint-budget inspection; no early
reservation release. Snapshot `/tmp/gfb-group-pack-task1-before-fix2/`.
If that issue survives this second fix/verification cycle, follow AGENTS's
advisor escalation rather than extending an unbounded repair loop.

Task1 fix round2 complete: remaining joint allowance corrected, exact tight
refusal/nearby success regressions pass. Correctness and architecture both READY,
no remaining blockers. Root fresh123affected/legacy checks PASS in5.54s, scoped
Ruff and mypy6PASS, links/diff check PASS. Foundation accepted for local milestone;
no production consumers and no complete-plan acceptance yet.

Target-filesystem disposable probe PASS on actual D:HDD and `/dev/shm`: WAL,
TRUNCATE checkpoint(0,0,0), DELETE transition, file/directory fsync, unchanged
readonly bytes and no sidecars. Escalated execution is required for these default
paths under this sandbox; user explicitly authorized them in the plan.

Ruling: correct the two test-coverage suggestions in the same wave — canonical
mutations must reach actual range/type checks, and original plan explicitly
requires real queue saturation/waiter wakeup rather than only oversized refusal.
No broadened production scope or default change is introduced.

Python Source/lineage/PPO design is READY after exact proof2 physical custody,
shared pool/prepare_commit retirement, run4/update4/wire5 and external disk-cap
accounting were specified. Native3B design is also READY after exact plan4,
borrowed-byte provider, summary/authorization/receipt variants and unchanged
numerical DTO choices were pinned. Formal/final64/Dataset final integration
design also passes architecture after formal variant hashes/unchanged budgets,
pending→public facts, native cumulative metadata/provider signature and exact
Dataset4 fit/provenance records were pinned. Source3A optional predecessor custody
delta is READY. Implementation/conformance and all dependent smokes remain open.

Task2 starts from accepted foundation dc9440e with a fresh bot_implementer.
Ownership is the Task2 plan list, plus minimal file-only logical-sidecar refusal
in structured_source.py and StorageSettings256MiB physical-cap validation.
Root owns docs/project-configuration assets; one code-writing agent is active.
Pure packing helpers do not expand the SQLite v1 persisted bound. All old
configuration/hash/default paths remain unchanged; larger physical bounds need
their own version-pinned Source/descriptor/budget review.

2026-11-11 Task2 frozen initial implementation: required368PASS, direct-entry
guard192PASS (overlapping sets), scoped Ruff/mypy20PASS. Expanded default sandbox
run380PASS/10SKIP/20setupERROR traced independently to inherited AF_UNIX socket
EPERM; unchanged preloader same case passes host6.11s. Root exact expanded host
rerun passes406/10SKIP in61.43s and fresh Ruff/mypy20. No infrastructure/timeout
edits. Independent review still BLOCKED; passing tests did not cover the findings.

Task2 fix1 design amendment READY: common sink/admission helper boundary,
minimum193MiB(128+64+1) with actual viability test, bounded queued/deadline Future
ownership, finite diagnostic graph retirement, held-original terminal identity
and optional-predecessor sweep, PackedRunResult and optional seal-metrics IPC.
Ordinary seal/report/tuple behavior and persisted formats remain unchanged.
Fix wave covers correctness F1–F4 and conformance T1–T6 (9distinct issues), plus
the exact accepted result/measurement contract. Before-fix snapshot is
`/tmp/gfb-group-pack-task2-before-fix1/`; root does not edit implementer files.

Task2 fix1 frozen: implementer365affected and61publication/retirement PASS
(overlapping),19focused review cases. Root fresh expanded HOST command including
all runtime/foundation/source/direct/formal guard regressions PASS596/10SKIP
in65.73s; fresh Ruff and mypy23PASS. Delta correctness/conformance reviews remain
pending; do not claim runtime acceptance from these test counts alone.

Root verified actual Simulator binary/source preflight read-only:
91d5926a7572db327eaea3751e7c24f428120996,
binary76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d.
Seven explicit opt-in live-smoke configs are prepared in
`/tmp/gfb-group-pack-live-configs/` (baseline6,3models×CPU/CUDA2games each),
all parsed with original7200/5400/7140/60defaults; NOT RUN until runtime reviews.
Audit runner `/tmp/gfb-group-pack-live-smoke.py` will check actual sealed original
bytes/report/index and capture final checkpoint metrics, without Source authority.

Root materialized the reviewed formal storage-variant asset/package copy;
canonical byte/hash/base/algorithm checks PASS (27853b10…). No code adopts it yet;
keep those root-owned assets separate from Task2 implementation and future
formal qualification/real training remains unrun.

Task2 fix1 both independent delta reviews READY. First REAL6game baseline on
actual D:/RAM reached composite publication but FAILed terminal stat comparison;
run02d698f1-551d-47c7-85a2-6cf716600668 is preserved, locator quarantined and no
healthy acceptance claimed. Log `/tmp/gfb-group-pack-live-baseline-six.log`.
NNsmokes remain NOTRUN. Read-only verifier's minimal actual-Writer D:/ext4 probe
isolated descriptor ctime-only changes at write-handle close on D:9p; all hashes
and dev/ino/mode/size/mtime stayed exact, ext4 passed. Evidence
`/tmp/gfb-pack-d-publication-investigation.md`.
Root proposed bounded terminal-control-ro-handoff-v1 policy (runtime design):
original inode+hash pinned through controlled close, then readonly capability
and strict final fullstat/hash/namespace validation before commit. Source
rechecks/custody remain strict6. Architecture handoff gate READY; bounded fix2
assigned next. Preserve all late-replacement tests and require actual D:/ext4
verification before the runtime milestone. Before-fix2 snapshot is
`/tmp/gfb-group-pack-task2-before-fix2/`.

Task2 fix2 both scoped reviews READY; root fresh HOST607PASS/10SKIP66.00s,
Ruff/mypy23PASS. Actual D:/ext4 published/diagnostic4probes PASS, strict Source
reopen and zero-budget cleanup PASS. No ordinary Source identity relaxation.
Actual6game R2 cbb37f35-3daf-4f9a-bcad-57ddf689af53 physically publishes and
ArtifactSource opens; DB integrityok and6natural outcomes. ROOTsaved-byte audit
still FAILs logical inventory: only extra storage-metrics.json is not declared
in manifest.files. Preserve this immutable population, do not claim acceptance
or retrofit it. Proposed narrow fix freezes metric bytes/inventory row BEFORE
manifest serialization, then stores EXACTfrozen bytes; no Source/auditor
weakening. Architecture order/interface gate pending, code remains frozen.

Metrics-inventory fix3 design and both delta reviews READY. Root fresh expanded
HOST616PASS/10SKIP66.96s and Ruff/mypy23PASS. Actual new6game run
9542c7a2-0283-441b-93b4-c7db8847bfb7 PASS:1group/6natural,176declared files,
14,774,682original bytes, exact public Source.names/inventory/report/index audit.
Actual checkpoint0.787115s, DELETE0.351610s, combined seal2.248325s. Metric
peak_storage_bytes=2GiB is managed reservation peak, NOT RSS. Actual RAM peak
3,310,004bytes. Receipt `/tmp/gfb-group-pack-live-baseline-six-r3.json`.
Both earlier populations remain immutable/ineligible; no retrospective repair.
Six neural CPU/CUDA×MLP/GRU/Transformer2game conditions now RUNNING sequentially,
`/tmp/gfb-group-pack-live-neural-matrix.{json,log}`. Do not start dependent Source
code or accept runtime milestone until final results are reconciled.

Neural matrix COMPLETE: all6conditions PASS,12natural real games, each full
saved-byte/index/report audit PASS; CPU/CUDA devices verified in receipts.
Together with baseline6,18successful real games accept Task2 STANDALONE runtime
scope. Both independent final reviews READY, root616/10SKIP/static evidence and
receipts retained in runs/assessments/sqlite-group-pack-v1-20261111. No Source/
PPO/native/final64/Dataset acceptance is implied; next Task3A remains required.

Task2 milestone committed1831f6f. Fresh Task3A implementer is active with brief
`/tmp/gfb-group-pack-task3a-brief.md`; report target
`/tmp/gfb-group-pack-task3a-report.md`. Only one code writer. Scope Python Source,
run4/update4/wire5/ordinary PPO/external disk budget; later guards remain.
Approved minimal file ownership extends to structured_journal concrete lower
verified-original adapter and config.py ordinary Python PPO guard; avoid a
journal→high Source→journal cycle. Root owns docs/assets, does not edit those
active implementation files. No push/publish or default switch.

Comparison protocol independently reviewed by bot_advisor; root adopts two
separate panels in sqlite-group-pack-comparison.md:6counterbalanced same-original
full-pilot HDD archival/read pairs, then4matched-condition nonformal2update
parents with independent bounded collector overlap. Do not call archival reference
the production oldpublisher, pool unlike samples, assume coldhostcache, or infer
strength/formal/default-switch authority. Numerical parity is its own gate.

Task3A initial code frozen, report /tmp/gfb-group-pack-task3a-report.md and exact
patch /tmp/gfb-group-pack-task3a-review.patch. Rootfresh affectedHOST with
GF_RUN_CUDA=1 passes574checks223.56s (no CUDA skip in this selection), mypy170/
RuffPASS. Independent correctness/conformance BLOCKED: expired workdeadline
prevents cleanup; partialPackedSource construction loses actual originals;
training importsCLI policy; ambient handled exception swallows scan-closeerror.
Fix1 assigned to same implementer, snapshot /tmp/gfb-group-pack-task3a-before-fix1.
Both reviewers agree separatepostcommit retention is allowed futureparent/pruning
validation and may fail/invalidateparent preservingactualupdate2; actualprepared
batch callbacks remain I/O-free. Source/ADR nowrecordthisphaseinterpretation.
No actualSimulatorPPOorNative3Bcodeuntilfixreviews/runtimeacceptance.

Task3A FIX1 both independent delta reviews READY; root fresh HOST affected suite
with GF_RUN_CUDA=1 passes598checks in231.05s, Ruff/mypy170PASS. These establish
component/fault/numerical coverage only. First actual packed Python PPO smoke
(`python-mlp-cpu-continuous-2`, seed17, CPU inference/CUDA learner, max2updates,
original7200/5400/7140/60budgets) FAILs before update0 with storage budget
exhausted. Its first16game/8group collection is saved; parent remains failed,
updates0/optimizer_steps0, immutable and ineligible. No limit increase, status
repair, optimizer replay or dependent Native3B implementation is authorized by
this result. Read-only saved-collection allocation diagnosis is in progress;
Task3A acceptance/commit and later real PPO conditions remain pending.

Source admission FIX2 direction architecture READY: reuse only the constructor's
FULL validated member index for initial inventory hashes, one logical Journal
parse with preserved strictJSON refusals, distinct retained raw-byte/decoded-graph
admission and one evidence charge per entry. No cap increase or actual-copy
deduplication by filename. Implementer active; saved-cohort diagnosis remains
read-only and cannot change the failed parent's eligibility.

Separate actual disposable direct-Writer D:probe FAILs the same controlled
descriptor write-close ctime refresh at `_publish` terminal retirement; composite
runtime remains accepted. Direct-public/non-retained-pending amendment is now
architecture READY in the physical design: factor SAME exact lower terminal
helper, joint RO allowance, strict DB/Source unchanged, no new DB control.
Implementation follows the current Source wave; benchmark/full-plan gate remains
open. Failed probe output is preserved, not silently retried as a paired trial.

Task3A FIX2 structural saved-cohort probe PASS16games/32participant records/
443decisions/566files, qualifiedFalse,49.30s; inspection admitted peak1,565,165,664B
and terminal account0. Constructor's temporary whole-available reservation is
reported separately, not as RSS. Root fresh HOST GF_RUN_CUDA=1 aggregate PASS610
in238.81s; Ruff/mypy170/AST8PASS. Conformance READY. Correctness reproduces a new
blocking file-cap regression: validated-index inventory admitted an opaque2MiB
member despite frozen1MiB Source-file cap. Root ruling preserves that cap for ALL
members before remembering evidence; minimal guard/regression fix pending the
completed review. No actual new PPO acceptance/commit yet.

PanelA draft harness review found omitted catalog bytes and absent per-report
device/build identities. Root amended draft to per-trial catalogs and joint
DB/catalog/sidecar sampling, exact host HDD/mount/runtime/source/harness identities,
and explicit Writer-archival/format-specific-readback labels. Aggregate reducer
and benchmark execution remain pending. Fresh read-only host identity confirms
D:disk2/USB/ASMediaASM1153e/HDD/4,000,787,030,016B; no settings changed.

Task3A file-cap correction both final reviews READY; root50affected cases PASS
15.00s and fresh Ruff/mypy170/AST8PASS. Actual fresh continuous parent2579bc3a
completes32natural games/2updates/419selected decisions/16optimizer steps in
531.165s. Separate dba33a31 pauses after1update, then resumes in a NEW CLI process
with identical Config/source and completes2updates/32natural/419decisions/16steps.
Both fresh root complete-parent Source/lineage/numeric commands exit0,87.381s and
87.661s, run bytes unchanged. Independent first-parent audit additionally records
actual2Source owners/SQLite/FD closure and pool/control accounting0; its temporary
wrapper import/summary errors are preserved, not labelled production failures.
Root accepts Task3A Python scope (64real natural games/4updates), not learning
quality, formal qualification or independent-trajectory final-state equality.
The full actor model/device matrix remains Task2 evidence; current affected
CPU/CUDA numerical parity plus the real MLP update/resume exercise the shared
Python storage/Source path without inventing a new campaign.

To preserve main's exact active-run source, direct-Writer work used a native
managed worktree `/home/percy/.codex/worktrees/63cf/GodFieldBots`, HEAD1831f6f,
with446hashed frozen-main src/test files copied before implementation. Only
Writer/publication/newterminalhelper/newtest belong to that delta. Default sandbox
write was readonly; authorized HOST copy/edits succeeded, no auto-review rejection.
Baseline87PASS; both final reviews READY,163bounded tests and4actualD:/ext4
public/nonretainedpending probes PASS. Parent used worktree source while -I child
loaded byte-identical main worker/dependencies; main hashes stayed exact. Root
integration/source-matched verification remains pending; no mixed-path wheel or
production acceptance is inferred. No existing run/Simulator data was modified.
