# PPO HPO foundation verification

Implementation in progress. This is functional foundation acceptance, not formal
hyperparameter search or evidence of strength.

- Bots baseline: `040184223a4bdcde85875f556daeb4a160ee7027`.
- Simulator source: `91d5926a7572db327eaea3751e7c24f428120996`.
- Design: [PPO HPO](../design/ppo-hpo.md), [ADR0037](../adr/0037-ppo-hpo-foundation.md).
- Pre-existing untracked analysis/session files and fixture lock are excluded.
- Baseline PASS: `.venv/bin/python -m pytest tests/test_config.py -q` —12passed.
- Baseline PASS: `.venv/bin/python -m pytest tests/test_group_pack_formal_contract.py
  tests/test_card_instance_contract.py -q` —125passed, including fixed ASTs and
  historical configuration contracts.
- Protected-original baseline PASS: bounded300s readonly inventory scan,
  29,814frozen inputs and24protected fit files match frozen SHA declarations;
  per-file SHA/dev/ino/size/mtime/ctime captured in
  `/tmp/hpo-originals-before.json` (25.57s). A final post-change comparison remains
  pending. No source artifact or original was rewritten.
- Optional dependency PASS: `UV_CACHE_DIR=/tmp/godfieldbots-hpo-uv-cache uv lock`
  resolved Optuna5.0.0 and seven new optional packages. Existing locked package
  versions are preserved. Installed with `uv pip install --python .venv/bin/python
  optuna==5.0.0` for verification. `import gf_cli` leaves both Torch and Optuna
  absent from `sys.modules`.
- Runtime: `.venv/bin/python`3.14.10, NumPy2.5.4, Torch2.14.1+cu132, installed
  native0.1.0. Ordinary sandbox CUDA/NVML access is blocked; an approved escalated
  `nvidia-smi -L` sees RTX5070Ti. Actual CUDA verification is still pending.
- Current-context architecture mapping, implementation and final acceptance:
  design READY; later implementation units and final acceptance remain pending.
  No unexecuted check is claimed passing.

## Unit1 configuration and admission

The dedicated HPO loader requires explicit ranges/trial count/D/training seeds/
evaluation GameSeeds before training. It preserves the base experiment hash and
old loader semantics. Model4 has activation-aware initialization identity and a
temporal descriptor without TBPTT. Separate training/search records normalize
inactive fields; categorical/joint proposals use stable choice indices.

Initial scoped158tests/Ruff/mypy5 passed. Independent correctness review found
two blockers: pool curriculum was absent from study identity, and search-domain
validation used defaults instead of overridden controls. Both are reproduced by
new regression tests and corrected. Study identity now binds pool interval/size;
search reachability uses the actual normalized immutable control, active branches
and coupled rows, with a bounded65,536-state admission limit per invariant.
Independent re-review is READY, passes168focused checks/Ruff/mypy5 and confirms
both blockers addressed. Root accepts the scoped unit1 implementation; overall
integration remains pending.

No network/artifact/Source/campaign runtime or GPU/game acceptance follows from
these configuration checks.

## Unit2 model, numeric and artifact components

Both independent correctness and architecture-conformance reviews are READY,
without blocking findings. Shared network kernels support model4 activations and
depths while preserving default ordered initialization and all eight fixed ASTs.
Whole natural episodes supply GAE/context; selected endpoints alone supply
normalization and loss. Adam is one canonical ordered group with every option
validated. Inference5 excludes resources/reward/PPO/TBPTT; resume5 binds the full
recipe, resources, execution, named optimizer state and RNG. Resource-free recorded
architecture parsing uses technical ceilings only; allocation requires active
caller admission. The custom >16M-parameter metadata regression allocates no
large network.

Root's fresh affected CPU command (19modules listed in task2's execution report):
`.venv/bin/python -m pytest tests/test_hpo_config.py tests/test_hpo_network.py
tests/test_hpo_numeric.py tests/test_hpo_artifacts.py tests/test_hpo_cuda.py
tests/test_model3_config.py tests/test_model3_network.py tests/test_model3_policy.py
tests/test_model3_artifacts.py tests/test_structured_config.py
tests/test_structured_application_config.py tests/test_structured_network.py
tests/test_structured_learning.py tests/test_structured_sealed_metadata.py
tests/test_structured_history_compat.py tests/test_card_instance_learning.py
tests/test_card_instance_registry_artifacts.py tests/test_card_instance_baseline_runtime.py
tests/test_torch_runtime_lazy.py -q --disable-warnings --maxfail=1 --tb=short`
—576PASS/56explicit SKIP in55.73s.

Independent correctness ran84new tests/10explicit skips and336historical tests/
46skips; independent architecture reran all8AST hashes. These scopes overlap and
are not additional unique test counts. Root project Ruff
(`.venv/bin/ruff check src tests benchmarks native`) passes; full mypy
(`.venv/bin/mypy src`) passes179source files. Bare `ruff check .` finds four
pre-existing errors in the user's untracked `codex_analysis.py`; that unrelated
file is preserved and excluded from project validation.

Implementer actual-host command `GF_RUN_CUDA=1 .venv/bin/python -m pytest
tests/test_hpo_cuda.py -q --disable-warnings --maxfail=1 --tb=short` passes10cases
in9.87s, including all3models, ReLU/Tanh, full/cache/split/CPU parity,1024context,
selected update/resume/RNG and clean GPU participant fixture inference. Root's
independent identical-command rerun passes10cases in9.27s. This is numerical/component evidence; real games,
Source/native/lineage/campaign/evaluation/retention acceptance remain pending.

The old retained registry's6neural/8total guard remains exact. Unit3 adds a
separate bounded HPO registry route for a configurable pool above5. No Tree,
formal-parent/final64/Dataset authority is extended by model4 inference.

## Inactive-field conformance correction

A later root/architecture check found two additional configuration gaps before
campaign implementation: MLP TBPTT was treated as active even though MLP consumes
only its current frame, and event refusal caps were exposed to search. Both are
corrected. `normalize_training_for_model` canonicalizes MLP TBPTT32 before
effective identity and new campaign/run materialization. GRU/Transformer retain
active TBPTT. Event max_interval/max_episode remain fixed configuration facts,
outside search/conditions; aggregate_window remains searchable within those caps.
The model-independent recipe parser preserves honest numeric-component metadata.

Implementer198affected checks/static pass. Independent architecture re-review
is READY and runs110HPO cases successfully. These replace the earlier inactive/
event-cap search guidance, without changing old recipes or saved-data semantics.

## Structural lineage prerequisite4a

Source/native need actual new-version lineage readers, so this bounded reader
prerequisite precedes unit3; the coordinator and retention producer still follow.
Run5/Generation4/Update5/Registry4 readers reuse the same scan, bundle and sealed
artifact machinery. Both independent correctness and conformance reviews are
READY without blockers. Source→chain dependency direction remains explicit.

Root fresh command:
`.venv/bin/python -m pytest tests/test_hpo_lineage.py tests/test_hpo_config.py
tests/test_structured_lineage.py tests/test_card_instance_lineage.py
tests/test_lineage_control_publication.py tests/test_ppo_lineage.py
tests/test_hpo_numeric.py::test_all_eight_frozen_kernels_remain_exact
-q --disable-warnings --maxfail=1 --tb=short` —341PASS in106.72s.
This includes89new structural cases,110config cases and142legacy/AST cases.
Root project Ruff passes; full mypy passes180source files. Independent correctness
also runs the89new cases (21.04s), Ruff and mypy1 successfully; architecture review
does not claim additional test execution.

Fixtures save actual CPU HpoLearner artifacts and cover G0/updates, all/one/partial
selection declarations, both storage variants, exact counters/steps/seed/policy
registries/deadlines, rehashed corruption and per-file deletion receipts with
G0/latest2/pool/periodic/evaluation dependency protection. Structural CPU complete
metadata is explicitly unqualified for production HPO. Actual original Collection6
resolution/replay, selection RNG/GAE reconstruction and exact prefix costs remain
unit3's work. No original Source, GPU campaign, coordinator, deletion producer or
Optuna evaluation authority is granted by these reader tests.

## Python Source3a and Wire6

Source3a is the Python scheduling/selection/complete-original Source/Numeric5/
Wire6 milestone. Actual native support remains Source3b; current installed
native HPO dispatch is strictly refused, with no fallback or advertised marker.
The same Source, custody, scan, numerical loop and process owners are reused.
Journal6 now explicitly binds own neural model4 metadata; Collection6 publication
checks its closed HPO role/context. Old versions retain their exact meanings.

Proof3 preserves both-view accepted-original counts; learner collected/used counts
are separate. Original-backed selection replay rejects even a rehashed alternative
valid subset. Complete episodes supply targets/context, selected rows supply loss/
normalization. Wire6 validates actual saved state/counters/costs and retirement
freezes236collected/1used/delta3 in the controlled case while retaining originals.
Historical inference deletion is accepted only with SAME-chain successful receipts,
immutable generation hashes and actual journal-carried inference5 metadata. The
original proof/correspondence remains identical; active policies require weights.

Independent correctness initially found two P2 blockers: legacy999cap on HPO
generation linkage and missing seed-derived G0 opponent RNG validation. Both were
reproduced and fixed. HPO supports its technical domain/canonical minimum-width
names including1000/1,000,000; old PPO keeps999. Full parent/live planning now
checks exact `PCG64(learning_seed)` initial controls before original/native or
production work. Independent scoped re-review passes5cases and static checks;
architecture rechecks the cap fix and is READY for the Python milestone.

Root fresh host command (local preload sockets require host access):
`.venv/bin/python -m pytest tests/test_hpo_selection.py tests/test_hpo_source.py
tests/test_hpo_wire.py tests/test_hpo_native.py tests/test_card_instance_source.py
tests/test_structured_numeric_worker.py tests/test_group_pack_decoded_retirement.py
tests/test_card_instance_learning.py tests/test_card_instance_collection_publication.py
tests/test_card_instance_registry_artifacts.py tests/test_hpo_lineage.py
-q --disable-warnings --maxfail=1 --tb=short` —268PASS/25unchanged CUDA SKIP
in110.56s. Root full project Ruff passes; mypy passes185source files.
Earlier implementer affected322PASS, host254PASS/25SKIP and post-registry14PASS
are overlapping scoped evidence. Independent initial review ran26new cases.

Controlled cases include24-game schedule order past pair10, same learner/snapshot
label with explicit own viewer, GAE/token/scalar/normalization tampering, historical
fully pruned controls, pool>5 through SAME custody, and actual CPU Wire6 update/
restore/bad-ack reap. No actual game, production CUDA Source/coordinator, native
positive dispatch, deletion producer, Optuna objective or strength claim follows.
The [native authorization design](../design/ppo-hpo-native-handshake.md) is
architecture READY; its implementation/positive candidate checks remain pending.

## Management5a: strict scores and durable GPU admission

Implemented in an isolated worktree and integrated as local commit38444a9 after
both independent fix re-reviews. Pure evaluation binds complete seed/GameSeed/
seat panels and actual terminal results, final-D checkpoint/Source identities
and healthy projections; it grants no Source authority itself. The fixed
per-uid persistent GPU namespace spans campaigns/checkouts and ignores HOME/XDG
overrides. Actual audited reservation history, durable job admission and exact
termination receipts replace PID/flock/time assumptions. Restart recovery is an
explicit operator confirmation and never rewrites run data/deadlines.

Initial broad correctness checks passed, but architecture review reproduced two
blockers: absent/replaced creation barriers could produce never-started evidence,
and retry after parent fsync failure skipped uncertain ancestry durability. Both
were fixed using the separately reviewed
[positive creation observations](../design/ppo-hpo-creation-proof.md) and ancestry
resync on every admission. Existing owners retain actual LOCAL Future/Popen,
original Event and same observed stdlib failure; observer errors preserve actual
adoption/reaping and primary errors while withholding receipt authority. Legacy
resource bounds/accounting and eight ASTs remain exact.

Independent conformance runs15targeted fixes; correctness runs79cases/65existing
optional skips and21additional lifecycle cases, plus static checks. Both report
ADDR/READY. Root fresh isolated command runs255PASS/65existing optional skips
in11.80s, with Ruff9files/mypy7source passing. Root post-integration host command
`.venv/bin/python -m pytest tests/test_hpo_evaluation.py tests/test_hpo_gpu_lease.py
tests/test_hpo_wire.py tests/test_hpo_source.py tests/test_worker_cleanup.py
tests/test_hpo_artifacts.py tests/test_structured_numeric_worker.py
-q --disable-warnings --maxfail=1 --tb=short` passes167cases in42.79s.
Full project Ruff and mypy187source pass.

Cases exercise actual owned CPU processes, default stdlib missing executables,
delayed creation/cancellation, task-creation failure, observer failure before/
after forwarding, mismatched/forked capabilities, orphan admissions, corrupted
state, exact receipt retry, ancestry and atomic/fsync faults, and complete-panel
scoring/failure refusal. No GPU, Simulator, Optuna campaign or concrete Source/
coordinator wiring was executed in this component scope. The Source exact-ledger
allowance, complete resource projection and actual pipeline cleanup integration
remain required before production acceptance.

## Native Source3b private-candidate acceptance

Both independent correctness and implementation-conformance reviews are READY
with no blocking findings. Native uses exact Hpo/PackedHpo plan5, pending/final
controls5 and additive authorize_hpo on SAME Source/provider/session/custody.
Python independently derives schedule/subset/GAE/normalization; native only
validates facts and structural correspondence. Selected/collected/optimizer
deltas remain distinct through bind/begin/prepare. Test-only CPU admission is
cfg(test), with no release API or flag. Historical deletion receipts and active
policy dependencies retain their exact meaning.

The generation-tagged numeric guard lives inside existing Budget/Lease. Source
refunds cannot credit numeric allocations; mixed shrink is conservative, stale/
nested guards refuse, and actual NumPy base owners keep charges after session
close. Private positive cases cover all3Model4 kinds, plain/packed complete
originals, partial/all/one/empty decisions, every regenerated numeric array,
authorization faults, source mutation/cancellation, pruning, quotas and views.

Independent correctness runs20selected private-release FFI cases,5guard and3held-
original/live Rust cases, scoped static checks and package hashes. Architecture
inspects phase/math/ownership boundaries and independently rehashes binaries.
Root fresh private release command
`GFB_HPO_NATIVE_CANDIDATE=/tmp/gfb-hpo-native-wheel-extracted/gf_native/gf_native.cpython-314-x86_64-linux-gnu.so
.venv/bin/python -m pytest tests/test_hpo_native.py -q -k
'public_hpo_admission or three_models_partial or view_keeps'
--disable-warnings --maxfail=1 --tb=short` passes5cases in25.17s.
Root `cargo test --manifest-path native/Cargo.toml --lib --offline` with the
documented test-only libpython link flags passes172checks/7explicit opt-in ignored
in39.08s. Root full project Ruff passes; full mypy passes188source files.

Private wheel SHA7295d1c779f80a2d550ab7590d68df4fcbbe7665f04b0b2f912ea666cbd9cb79;
ELF SHA8164d5c07969e95d7a501ced62e31cdecd618dde1080b4b85ad79f5218ee9e85;
embedded source5db7cf1519722ba5df1aaac9b5a806cce1e6b0d5374720dfb5b38532d993d4b0.
RECORD/stub methods/Cargo.lock/profile/allocator/current source checks match.
The private capability is exact; the installed old extension remains
SHA9b1377daac7c93c97a4f267d5f20c2204eb37401a53daa1275019614d1b6767f
with no HPO marker. No install or heavyweight old-data proof was run for this
scoped acceptance. Genuine integrated CUDA/native/live/common-loop/Simulator
and whole-HPO acceptance remain pending runtime4/5b integration and final freeze.

## Runtime4 and evaluation6 scoped acceptance

Evaluation6 is a distinct HPO validation role with no formal/final64/Dataset
authority. Independent review caught actual unhealthy process/group evidence
being accepted despite a healthy report, then a nonfinite JSON exponent bypass.
Both were corrected without changing original outcomes or legacy JSON semantics.
Both final role reviews are READY. Root's combined316checks pass in16.15s;
scoped Ruff/mypy also pass. This is component evidence, not an actual panel.

Runtime4 uses the common coordinator with the actual caller GPU job/ledger,
complete-original selection and per-update full saves. Independent review found
retained Source missing from the save-phase joint RAM estimate and HPO mutators
accepting an unowned or foreign-thread parent-lock handle. Both were corrected;
an additional cleanup error-preservation finding was also closed. Both final
runtime reviews are READY. Root's focused22phase/ownership/retention cases pass
in17.44s and7cleanup cases in0.24s; project Ruff/mypy190source pass. Local commit
c298938 integrates the reviewed runtime;06db3d9 integrates evaluation6.

The post-integration native payload fixture initially failed at the new Source
numeric admission guard before reaching its intended native allocation check.
Commit6b31aac changes only tests: ordinary Source keeps its early refusal, while
an explicitly unqualified component fixture exercises native preallocation.
Root's2focused cases pass in4.80s and independent recheck in4.49s. No production
guard, native binary or Source authority was changed. Project Ruff/mypy191 pass.

## Campaign5b scoped acceptance

The outer campaign implementation is integrated as local commit27721bf after
both independent correctness/conformance fix re-reviews. Actual Optuna5 SQLite
is exercised by management tests, while synthetic verification fixtures explicitly
remain unqualified. Before fixes, root's333management/evaluation/configuration
cases passed in17.25s despite the integration gaps listed below; this historical
pass did not establish acceptance.

Independent review found the training collector wrapper accepted one argument although the
common coordinator supplies the plan and actual ledger; it also found that
evaluation verification did not bind all played-policy fields to the independently
inspected final-D checkpoint. A third P1 made valid multiple-seed panels collide
because projection preserved collection-local episode names. All three have
separate reproduced failures and passing regressions: the wrapper now verifies
and forwards the supplied SAME ledger; projection scopes episode IDs to the
actual original evaluation namespace while retaining global GameID checks; and
the verifier derives the complete final-parent reference/actual policy registry
from independently inspected saved artifacts before projection/tell. The genuine
older-checkpoint negative retains honest CPU identity and grants no production
Source authority.

Implementer395combined cases pass in87.28s. Independent correctness runs68affected
cases, including16integration tests; conformance independently runs those16 in
8.33s and checks authority/guide alignment. Both report all3P1ADDR/READY, with no
remaining scoped blocker. Root fresh same management command now passes347cases
in25.90s after14new regressions, with full project Ruff/mypy195source and diff
checks passing. The predeclared functional inputs and pure resource checks do not
constitute actual GPU or Simulator evidence.

Real GPU/Simulator execution, native installation, whole Python suite,
package/optional-free proof, protected-original after-comparison and final
whole-HPO reviews remain NOT RUN.
Separately, root's unchanged-native final Rust rerun passes172cases/7explicit
opt-in ignored in43.70s, and cargo fmt check passes; no CLI change alters those
Rust source bytes. The source-matched native sdist is also built, but combined
final package correspondence and installed activation remain pending.

## Final numeric-read integration correction

The whole-source correctness audit found two additional integration gaps after
the unit reviews. The native adapter calculated the declared numeric phase limit
but omitted it from the decoder call, selecting the component default512MiB.
The Wire6 worker also parsed its preliminary numeric manifest against the model
budget before the phase-limited decoder. An actual CPU probe with a1-byte phase
parsed23,210bytes/eight episodes before later refusal; no optimizer or G1 was
created. Related parent Source metadata was independently Source-budgeted, but
its first numeric parse likewise lacked the declared phase admission.

The correction forwards the exact native limit, admits worker JSON/descriptors
before their first parse, and releases preliminary metadata/file-map copies before
the second decoder. Parent admission uses the SAME sealed file/hash/stat custody
before parsing and retains its Source accounting. Original Source/model bounds,
legacy branches, formats, eight kernels and Rust binary remain unchanged.
Tests exercise actual64MiB/1GiB native readers, clean CPU Wire6 refusal/success/
copy lifetime, and same-bytes inode replacement before parent parsing.

Implementer103affected cases pass in188.68s, including legacy CPU numerical
workers and eight ASTs. Independent correctness runs7covering/kernel cases in
11.66s and reports both findings ADDRESSED/source READY. Root fresh6new cases
pass in11.57s; full project Ruff/mypy195source and diff checks pass. Independent
conformance runs the same6cases in11.23s and reports READY, including the frozen
functional helper's authority/API alignment. Overall functional acceptance still
requires actual GPU/Simulator execution and the final integrated proofs.

The initial25f9266 source-bound legacy attempt passes history26.69s,
Observation16.36s and index14.43s plus all three data audits. Its one-worker saved
PPO probe times out at600s in the ordinary sandbox, and the four-worker probe is
not run. All failed outputs remain preserved and unaudited. Forensics locates
the timeout in the first legacy native stage after the Python oracle succeeds;
this CPU-only path has no Torch/CUDA or project subprocess/IPC, so no specific
IPC cause is claimed. A separate host diagnostic using the same native candidate
passes16episodes/224decisions and all three reader array comparisons within its
30-second diagnostic bound, with zero updates. This does not replace the final
current-source five-job proof, which will use fresh host-executed outputs.

The25f9266 application wheel/sdist and private native artifacts separately pass
strict source correspondence:197application files,91native files and7embedded
assets. Installed mandatory-only checks pass all seven optional-package absences,
actual baseline startup/close, HPO module/two-console help and configuration
validation without creating an Optuna DB. These receipts predate this numeric
correction; a fresh application package proof is required after its source freeze.

## Current-source compatibility, activation and early-entry correction

The a460e1b source freeze passes a fresh application/native package correspondence
proof and installed mandatory-only CLI/baseline checks. All five fresh host
legacy jobs and data audits pass: history17.07s, Observation15.45s, index13.42s,
saved PPO workers1/4 in2.96s/2.16s. Both saved-PPO cases preserve16episodes,
224decisions and three-reader array parity, with zero updates. The source/plan
inventory guards pass. This supersedes no failed attempt: proof-a01 remains
preserved as failed for its one-worker timeout.

After checking those receipts and preserving14old installed files byte-for-byte,
root installs the accepted private release/MIMALLOC wheel into the main venv.
Fresh installed identity/capability checks pass: ELF8164d5c…ee9e85, source5db7cf…
93d4b0, cp314-gil, empty wheel rustflags and the exact additive HPO capability.
The rollback files and activation receipts remain under the functional assessment.
This activates implementation support; it does not establish actual GPU/Simulator
or trained-strength evidence.

The first whole Python run stops at72PASS/1FAIL in3.46s. The added HPO storage
probe accessed model unconditionally before the existing engineering-input guard;
an actions-only early-entry input therefore raised AttributeError instead of the
required engineering_only refusal. A one-line optional model probe restores that
legacy behavior, leaving concrete Model4 owned-collector checks unchanged.
No tests were weakened. Implementer251cases pass in4.64s; root fresh251pass in
4.97s with full Ruff/mypy195source/diff checks. Independent correctness passes
10existing boundary cases plus four direct nonconcrete Model4 refusals and reports
READY. Final current-source proof/package refresh, whole Python completion and
the predeclared actual functional protocol remain required after this correction.

## Native test-module isolation correction

The next whole run reaches4021PASS/211explicit SKIP, then waits in the existing
native constructor-cancellation test. Root identifies a test-only module leak:
the HPO candidate fixture replaced sys.modules[gf_native] and never restored it.
The legacy test patched its module captured during collection, while the runtime
loaded the leaked candidate, so its constructor-entered event never fired.
Root verifies the owned pytest PID/arguments and interrupts that process normally;
the preserved partial run exits2 after1219.19s and is not a whole-suite pass.

The fixture now restores the exact previous module in finally, including module
creation/execution failures. Four failure cases cover existing/absent entries.
The legacy test bounds its hook wait, checks the actual module identity and
closes its owner after joining, preserving all original cancellation/lock checks.
The real same-process HPO-admission→legacy-cancellation sequence passes on HOST;
the combined two modules pass90cases in71.66s. Root and independent correctness
each run6covering cases in0.21s; Ruff/diff pass and independent review is READY.
Ordinary-sandbox later asyncio Future wakeup behavior is recorded separately and
does not justify a production lifecycle change or a claimed specific cause.

Only tests change; all197application runtime files and the native binary remain
identical to the0f4f300 package proof. The following full run enables the existing
pytest faulthandler timeout diagnostic and retains the same test scope. Actual
GPU/Simulator protocol and whole-suite completion remain pending at this point.

## Whole-suite pass and first actual CUDA functional attempt

The3d1504f HOST whole command with the explicit source-matched native candidate,
`-o faulthandler_timeout=60` and JUnit output completes5462PASS/341explicit SKIP/
29warnings in1244.89s. A slow JSON case prints a faulthandler diagnostic without
failing. The previous constructor wait is fixed. The same runtime197files pass
fresh package correspondence and all five HOST proof-a04 run/data audits;
mandatory-only installed CLI proof remains valid for these unchanged bytes.

Root then executes the predeclared MLP inner case with actual CUDA learning and
inference on NVIDIA GeForce RTX5070Ti/CUDA13.2. Eight natural games produce84
learner decisions; the final-batch selector uses8 and the child performs one
optimizer step. The run FAILS during generation preparation, before Gen1/Update1
publication: the native bridge's separate Python _ChainScan still expects G0
when the actual saved G1 directory is rechecked. The coordinator's own scan has
advanced, which does not advance that separate owner. Saved G1 exists but is
not a committed checkpoint. No final-policy evaluation, objective, trial or study
is created. The whole parent remains failed/quarantined/ineligible and is retained
at `runs/hpo-functional-20261114-a01/matrix-mlp/parents/09c81085-c1e5-47c1-8424-4c174b47a44e`.

Cleanup retains uncertain native executor/provider/original ownership. Actual
global ticketf6150b31a5e04244a1a2d918da8ccfc8 has admission only, no termination
or restart receipt, and remains current. Owner PID absence is not cleanup proof.
All subsequent physical GPU work is refused pending explicit actual environment
restart confirmation under the accepted recovery rules; root does not reboot,
reset the namespace, fabricate a receipt or reuse the failed parent.

The [bounded save-transition correction](../design/ppo-hpo-native-save-transition.md)
passes independent architecture review before implementation. It latches the
actual verified learner ACK on the SAME projection, independently checks the
saved pair/map/expected generation/steps/parent-lock owner, and advances only that
native capsule's owned Python namespace through the existing Rust append permit.
Both scans, all old originals and normal native prepare/retirement remain checked.
CPU/native common-loop plain/packed update/resume regressions and negative
transition tests are required; isolated honest CPU test leases grant no physical
GPU recovery or production Source authority. Implementation and re-reviews remain
pending here, followed by a fresh same-condition functional attempt after recovery.

An interim after-a01 data audit rehashes all29814protected originals and24fits:
bytes, device/inode/size/mtime/ctime match `/tmp/hpo-originals-before.json` exactly,
PASS in9.46s. Its receipt is `originals-after-failed-a01.json` under the functional
assessment. Future real attempts still require the final after-comparison.

## ACK-bound save correction scoped acceptance

The correction now passes independent correctness and conformance reviews, with
no blocking findings. Actual verified ACK capture occurs before the child request
handle clears; immutable bytes and decoded aliases remain charged to Source
controls. Commit independently checks the saved pair and exact map/g+1/steps/
actual lock owner, then the owned native capsule checks Rust's append permit and
advances only its separate Python scan. Original hashes, inventories, providers,
root identity and ordinary native prepare/retirement remain checked. Partial
transition/prepare/publication failures retain the first error and poison reuse.
Native packed ACK metadata stays charged to controls, without duplicating a
consumed graph floor; Python snapshot growth is admitted before serialization.

An actual CPU/native common loop reproduces the original G1 failure in4.86s.
The fixed plain and packed cases both complete continuous2updates and pause1→
resume2 with equal saved weights, Adam and RNG arrays. Fifteen actual child/native
fault cases reject missing/duplicate/mutated ACK, foreign process/owner, wrong
generation/steps/map/handle, extra/symlink/replaced directories, changed old/new
files, partial admission and cancellation before lineage publication. When a
damaged namespace prevents writing the failed marker itself, tests require
structural ineligibility and terminal capability rather than repairing the marker.

Implementer private-candidate regression168PASS in256.84s includes affected
Source/wire/retention/decoded accounting, legacy packed PPO/CPU workers and eight
ASTs. Independent correctness runs8CPU/native cases in58.45s plus kernel pins,
the Rust append primitive and static checks; conformance runs4CPU/native equality/
partial/cancel cases in44.16s. Both report READY. Root's new-private continuous/
resume2cases pass in38.77s; a separate ABI-compatible installed-native check also
passes2cases in38.11s. Root full Ruff/mypy195source/diff pass. These tests retain
honest CPU metadata and isolated component leases; none authorizes physical GPU
work or qualification while the actual pending ticket remains unresolved.

Rust now distinguishes the existing closed Hpo/PackedHpo formats for widened
canonical numeric generation names, leaving old1000-directory/order guards exact.
Bounded primitive cases exercise1001directories, six-digit lexical boundaries and
2^63 without large training. All Rust173PASS/7opt-in ignored in38.69s, rustfmt
and independently recomputed source/Cargo/RECORD/sdist checks pass.

The new private release/MIMALLOC wheel is SHA92390e03…e6253b; ELFdae8bb36…7dc8a4;
source56df546f…2cc064; Cargo.lock unchangedf81e4cc1…c6576. It is not installed yet.
Fresh app/native wheel/sdist correspondence passes197/91files and7embedded assets,
and fresh installed mandatory-only baseline/CLI/HPO validation passes for the
same-condition retry inputs. Root-owned final current-source corpus/whole checks,
local acceptance commit and any installed activation remain separately recorded.

No new physical GPU run, recovery action or mutation of a01/f615 occurred during
this correction. Fresh a02 inputs change only storage namespaces; all nonstorage
config fields/base identities match a01. The unresolved actual ticket still
requires explicit real environment-restart confirmation before the remaining
functional protocol can continue. Foundation acceptance and strength claims are
not implied by these CPU/source results.

## Save-correction CPU acceptance and recovery prerequisite

The reviewed correction is saved in local commit0c2b5b1. Root then runs the whole
suite with CUDA_VISIBLE_DEVICES empty before Torch import, single-thread BLAS
settings, the new private native site first on PYTHONPATH and the explicit
candidate fixture. Result:5479PASS/341explicit SKIP/29warnings in1300.88s. A
candidate-storage fixture produces the60s diagnostic while progressing; no test
fails. JUnit: `whole-cpu-0c2b5b1.xml` in the functional assessment. This is
CPU-only whole-suite evidence; it does not supply the pending physical CUDA
functional results or remove the real GPU admission block.

All five fresh HOST legacy jobs and data audits pass against new native source
56df546f…2cc064/ELFdae8bb36…7dc8a4: history12.50s, Observation11.70s,
index10.45s, saved PPO workers1/4 in1.83s/1.68s. Source-inventory guards pass.
The fresh package/mandatory-only receipts are `package-proof-save-transition.json`
and `optional-free-save-transition.json`; all197/91runtime files,7embedded assets,
seven optional-package absences and actual baseline/CLI startup checks pass.

Root preserves the previously installed8164… extension and all14distribution
files separately, then installs the new reviewed wheel offline without changing
other dependencies. CPU-only installed identity/capability checks match the exact
dae8… ELF/56df… source/release/MIMALLOC/cp314-gil/empty-rustflags recipe. The actual
installed backend, with no private-candidate override, then passes both plain/
packed CPU continuous/resume checks in37.38s. Ruff, mypy195source and diff checks
pass. Earlier9b137… and8164… rollback copies remain retained.

A second protected-original after-check passes29814originals/24fits with exact
bytes/device/inode/size/mtime/ctime in4.09s, saved as
`originals-after-save-correction-cpu.json`. No failed parent, original data or
actual f615 ticket is changed by these checks.

All code repair, non-GPU verification, reviewed distributions and retry inputs
are concrete before asking for recovery. The real ticket still has admission
only. Remaining work requires the operator to actually restart the environment,
confirm all previous GPU jobs terminated, and then authorize the exact existing
`hpo recover` confirmation path. The agent must not assert that event itself,
reboot automatically or use a different GPU namespace. After recovery, repeat
MLP in fresh a02 and finish GRU/Transformer, cap-resume, actual CLI multi-seed/
session/tell/direct-comparison cases and final audits/reviews. The a01 failed
population remains ineligible permanently. Foundation acceptance is still pending.
