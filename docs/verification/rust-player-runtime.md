# Rust player runtime verification

Status: implementation in progress; no runtime, performance, adoption or learning
acceptance claimed. Branch feat/rust-player-runtime starts at9680f06. Approved
[owner protocol](../design/rust-player-runtime-approved-plan.md) and proposed
[design](../design/rust-player-runtime.md) govern remaining checks.

## Baseline environment and focused checks

2026-11-19, isolated managed worktree
`/home/percy/.codex/worktrees/37cb/GodFieldBots`. `uv sync --frozen --extra torch-cuda
--extra shared-inference --extra native --extra hpo --extra data --extra lightgbm`
PASS, source-matched release native built. Python3.14.10 GIL enabled,
Torch2.14.1+cu132, NumPy2.5.4, TorchRL0.15.0, TensorDict0.15.2.
Scoped escalated GPU probe PASS: RTX5070Ti,16303MiB,driver610.88,
Torch CUDA available with one device. A sandbox-denied probe is not physical
GPU unavailability. Simulator release exists at
`/tmp/gfb-shared-simulator-072dabb/target/release/gf-test-server`; verify its full
identity again before actual game experiments.

Baseline command (before runtime implementation):

```sh
timeout 300s .venv/bin/python -m pytest -q tests/test_card_instance_actions.py tests/test_defense_priority.py tests/test_refresh_heuristic.py tests/test_shared_inference_numeric.py tests/test_shared_reply_retirement.py tests/test_hpo_owned_capture.py tests/test_shared_gpu_termination.py tests/test_shared_normal_run.py
```

PASS:200passed,7skipped,7.32s,exit0 (independent verifier). Raw output retained at
`/tmp/gfb-rust-player-baseline.log`. This is a focused baseline, not a whole-suite
claim. ADR0053's recorded whole-suite result remains historical51FAIL/7158PASS/
574SKIP with documented triage; it is not silently converted to whole-green.

## Native dependency foundation

Reviewed pins Tokio1.53.1 and reqwest0.13.5 resolve without upgrading existing
locked packages; reqwest has default features disabled and JSON enabled.
`cargo check --manifest-path native/Cargo.toml` PASS. Native tests initially fail
to link `_Py_Dealloc` with plain cargo test because extension-module builds do
not add the interpreter library. The existing project's documented test recipe
resolves this without a code change:

```sh
env RUSTFLAGS='-L /home/percy/.local/share/uv/python/cpython-3.14.10-linux-x86_64-gnu/lib -C link-arg=-lpython3.14' LD_LIBRARY_PATH=/home/percy/.local/share/uv/python/cpython-3.14.10-linux-x86_64-gnu/lib PYO3_PYTHON=/home/percy/.codex/worktrees/37cb/GodFieldBots/.venv/bin/python cargo test --offline --locked --manifest-path native/Cargo.toml --lib --quiet
```

PASS:177passed,13ignored,3.11s test execution. Extension-wheel builds must unset
these test-only linker flags. This verifies the dependency foundation, not a
player runtime or installed-companion acceptance.

## Current-v5 reference originals

PASS: nine own-view episodes/twelve decisions, exact fixed-clock reports and
34,285 clock reads, ordered tables/actions/hashes, CPython search/baseline RNG
and sequential/repeated Refresh behavior. Largest capture has2048candidates and
12visible cards. Fixture SHA256:
`ecf073cdb848e97752711e4f9b0a39f6f5d1b2206206a7acb293d25047d080f3`.
All47source hashes match baseline9680f06. Explicit recapture is byte-identical;
implicit overwrite refuses. Scoped regression333PASS and Ruff check/format PASS.
Independent `/root/v5_reference_review`: spec READY, task quality READY; optional
README public-cause wording corrected. This supplies comparison originals only.
Full18-card/maximum-row extremes, every trade/allocation family and broad RNG API
boundaries need separate port tests. See [fixture scope](../../tests/fixtures/rust_player_v5/README.md).

## Native v5 policy core

Accepted bounded core after independent correctness/spec-quality and architecture
conformance READY, including one scoped repair/re-review round. New modules
`native/src/player_runtime/` implement public rules, bounded v5 search, CPython
used RNG APIs, Random/Heuristic and participant-local Refresh. Whole-call
GIL-detached diagnostics are `player_search_v5`, `player_select_v5` and
`player_rng_v5`; later actors must use the typed core and incremental bank
custody, rather than repeatedly rebuild full-prefix diagnostic JSON.

Original broad comparison:679PythonPASS, including346new collected tests;
237owned perceived definitions across four modes,40mixed seeded searches,
65,536rows/18cards, resource/allocation/sequence rules and Refresh histories.
This broad run preceded the two final fixes. Independent review found deadline
truncation could return an empty table, and baseline capacity18 was not enforced
separately from search capacity128. Both are corrected with meaningful RED→GREEN
coverage; search capacity is preserved and no candidate is synthesized.

Final repair evidence:11coveringPythonPASS,245affectedPythonPASS,3focusedRustPASS.
Root's full native suite after fixes:180PASS/13IGNORE in3.30s; root fresh11covering
checks PASS and loaded release/lock hashes verified. Both independent re-reviews
are READY with bothP2findings ADDRESSED and no remaining findings. The unchanged
broad679selection was not rerun after these two local fixes; no final whole-Python
green claim follows. Source-matched fixed release native SHA256:
`9bfc13beae4c1d308ae0267c0a9d0815fb08d0d74e5f42f193901d28938a536a`;
Cargo.lock SHA256:
`bb8e9e3cae6eab6c2877016c18d58256ea46f5f5807f8b41a6d475642bae692d`.
Extension RUSTFLAGS are empty. Saved reference fixture hash remains unchanged.

This accepts the policy core only. Live HTTP/state ingestion, Tensor ownership,
sampling retirement, complete runtime isolation, training/resume and measured
speed/memory remain required work. Diagnostic Refresh state is replay data,
not a new admitted checkpoint format or Source proof. Existing Python production
and historical readers/goldens are unchanged.

## Owned inference boundary

The bounded array-serving/dispatcher stage passes independent correctness and
architecture conformance READY, including one scoped repair/re-review round.
`ServingRow` reuses existing corrected native token kernels and transfers final
float32/int64/bool allocations into NumPy; torch.from_numpy aliases CPU storage
and last-reference tests preserve its lifetime. One shared Dispatcher supports
blocking adapters and direct Tokio admission/result waits without Python callbacks
or numeric JSON in the live path. Formation starts at first valid admission's
absolute timestamp, including aged queues and first-row retirement.

Rust owns committed opaque participant Torch bytes and atomically accepts only
matching valid request/sample/next-state tuples. Invalid/expired/retired results
adopt no RNG. Exported canceled leases retain quota until original service
settlement; logical queue close does not prove Tensor/GPU completion. Game actors
must bind actual Simulator gameID separately from planned frame.episode, retain
active/frozen checkpoints before descriptor admission, release after actual leases
settle, and use the same Torch serialization lock for learner operations. Registry
publication additionally restores CPU and already-initialized CUDA RNG; learning
retains its intentional RNG changes under the lock.

Original component evidence:8RustPASS,49inferencePASS including real CUDA,
470broaderPythonPASS/18existing explicit opt-in skips. Review found retirement
missed capacity-blocked/unseen admissions, and unsupported finite timers could
panic/poison cleanup. Both corrected with RED→GREEN tests: pre-capacity epoch
ownership/tombstones, drop-safe started-producer registration, and checked actual
Duration/host-clock representation with no arbitrary short cap.

Final repair evidence:12focusedRustPASS,55inferencePASS/0SKIP including real CUDA;
root fresh6coveringPASS and loaded native source/lock match. Root full native after
fixes:189PASS/13IGNORE in3.24s. Both independent re-reviews mark bothP2findings
ADDRESSED, no remaining findings. The unchanged broader470selection was not
rerun after this local repair; its result is historical, not a final whole suite.
Fixed native release SHA256:
`e62691d7fa8fa4eb61cfe6b4b9558fc62ad64c4260792d629c560b066b25f999`;
Cargo.lock remains `bb8e9e3cae6eab6c2877016c18d58256ea46f5f5807f8b41a6d475642bae692d`.
Extension flags empty; Python adapter Ruff/format/mypy pass.

Six-row CPU/CUDA mechanism timings separate assembly, sharing, synchronous
transfer, fenced forward and return. They include cold first forward and belong
to original pre-fix sourceaa9e5a74, not final corrected source or an end-to-end
speed comparison. Full actors/storage/learning/process-tree/audits and approved
physical matrix remain required before any complete-runtime/adoption claim.

## Common live game runtime and A/B schedulers

Accepted bounded actor/transport component after independent correctness and
architecture conformance READY, including one repair/re-review round. One native
game::run handles external HTTPv2, participant-local history/candidates/sampling,
pending/ACK/terminal records and Replay/outcome reconciliation for both Tokio
task-per-game and fixed-worker schedulers. W1/W2 genuine two-game/four-participant
tests hold two pending games before either can Act; MLP tests observe a real
mixed-game batch. Idle worker runtimes keep pumping the shared HTTP client between
waves. Python owns one optional inference consumer; no Python player process or
policy/HTTP/tokenizer callback is used.

History admits deltas once and shares Arc typed events/32-byte prefix backing
with one decision input. Trusted and validated token paths share mathematics;
malformed new deltas reject at the owner boundary and unique append survives
success/cancel/failure. Borrowed capture DTOs avoid transient full table/bank/
Replay Value copies. New transport JSON preserves actual selected f32 values by
explicit f64 promotion; original frame/table/report hashes remain unchanged.
Read-only byte views retain the SAME immutable native allocation/reservation to
last use. ACK is custody acceptance, not persistence/Source qualification.

Initial candidate checks:197nativePASS/13IGNORE,524PythonPASS/18existing opt-in
skips plus strengthened48runtimePASS. Reviews found failed-service settlement
could release holds while traceback-owned Tensors survived, and live Refresh
narrowed arbitrary positive thresholds to u64. Both repaired with forced-schedule
and above-u64 identity tests. Optional O(N) proposal copy for one selected action
was removed with indexed columns; no speedup magnitude is inferred.

Final correction:197nativePASS/13IGNORE,537PythonPASS/18existing opt-in skips,
including61runtime cases and actual CPU/CUDA response cancellation. Root fresh
13covering checksPASS in5.52s and loaded release/source/lock assertionsPASS. Both
independent re-reviews mark bothP2s and optionalP3 ADDRESSED, no remaining findings.
Final native source SHA256:
`7c95c3c8653d34f1b1d3e8441120836a7a435589ea6bbd1e0f0ddbd061ea5282`;
Cargo.lock unchanged `bb8e9e3cae6eab6c2877016c18d58256ea46f5f5807f8b41a6d475642bae692d`.
Release flags empty. New facade Ruff/format/mypy and changed Rust formatting pass;
whole card_instance_dto formatting has a pre-existing unchanged line1496 issue.

Actual approved external binary SHA256 is
`76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d`.
Raw genuine baseline/natural/Replay/HTTP/process evidence remains under temporary
`/tmp/rust-player-task4-overlap-c8b25f4` and `...evidence-c8b25f4`; its recorded c8
identity is historical. New fixed identities are at
`/tmp/rust-player-task4-fix1-evidence/identities.json` and final execution logs;
oldb2/c8 evidence was not relabeled. These are component diagnostics, not durable
learning parents or passed archive/learning/performance gates.

Root-approved next integration must provide the real thread writer/SQL/seal,
new outer Source/capture authority, same-process learner, original-host supervisor
and normal entrypoints. Actual game/planned episode mapping, checkpoint retains
and failed-service holds, original lease/producer settlement and physical consumer/
Tensor/GPU retirement remain required contracts for those owners.

## Same-process thread storage (engineering component)

Accepted engineering component after independent correctness/conformance READY
and two scoped fix rounds. The real dedicated SQLite thread and old subprocess
worker share GroupPackDatabase business operations. Actual DB creation/use/seal/
close stay on the original owning thread; old framing/PID/allowance semantics are
unchanged. CanonicalEpisodeEncoder validates actual same-account native bundles
and emits unchanged Journal6/codec2/Replay content, including zero-Act and real
failure-prefix facts. It creates no outer Source/learning qualification.

One native RecordingAccount charges native arrays plus separately allocated
Python canonical/pack/descriptor backings. Bundle adoption/ACK/view aliases do not
charge the original twice; foreign-runtime bundles reject. Actual last NumPy/
sliced descriptor views retain their backing charges. No per-record token guard,
new multiplier or second recording pool exists. The explicit shared Source loan
is a soft planning reservation, not measured usage or a hard memory limit.

Initial264PythonPASS/197nativePASS13IGNORE evidence precedes repairs. Reviews
found nested FD-owner propagation, post-start construction ownership, chained
traceback-buffer refund and cancel/seal ordering gaps. Meaningful fault cases
resolved them; narrow second review completed Thread.start interruption coverage.
Final273affectedPythonPASS,25storage cases included; root fresh4constructorPASS
in1.23s. Both independent final re-reviews READY, all4P2s ADDRESSED, no remaining
findings. Scoped Ruff/format/mypy and diff checks pass. Native code/build remains
source `fb6c090ec59f7fdbcfb02ba12a298cdc50fa86cfb8ddf68f66c5aa66dc49ad86`,
loaded release extension SHA256
`289cf2a10fb83aa222273c7d0bee2ceb4be2658af7c1e6d9e892b26c01eb6092`;
Cargo.lock unchangedbb8e9e3c. Final storage Python SHA256
`29549dc98f538dace5d2a1cb6eb20c54916f5d7f47e881a6dba7260aa57d5d08`.

Bounded pre-repair memory observations remain tied to original storage Python
SHA7adaeeaa, not relabeled as final repair measurements. Genuine baseline full
paths trace24–25MB with sampledPSS119–142MB; CPU MLP192 paths trace162MB and
sampledPSS897–911MB including importedTorch. One validated32768candidate/18slot/
4096event two-view prototype plus15independent CANONICAL backings traces82.9MB,
sampledPSS186MB, OS-high-waterRSS535MB and exactrecord peak84.6MB. These are not
16active banks or16genuine games. The prior120s repeated-validation attempt is
inconclusive execution-bound evidence, not a capacity failure. Scripts/results
are temporary `/tmp/task5a-*-study-final.json` with exact producer/script identities.

The follow-up source-pinned retained-bank investigation is INCONCLUSIVE: its
10-minute setup/execution bound expired before a new case started. No concurrent
C2/C4/C16 or near262144-event memory observation was obtained. The verifier
confirmed frozen de43f2e Python29549dc9 and loaded nativefb6c090e/extension289cf2a1;
this is a loaded-build identity, not a current-tree match after later edits.
Its temporary status/identity records are `/tmp/gfb-retained-bank-study.md` and
`.json`. This contributes no new capacity evidence; the original calibration
limits and follow-up measurement requirement remain in force.

A separately bounded v3 study now passes its measured retained-bank scope.
Thirty distinct actual Journal/EventBank graphs hold442368events (262144,65536,
28x4096), populated by normal incremental validated deltas. They remain alive
through genuine C1/two8-decision-ceiling games, both terminal views and real
Pack/SQLite/seal publication; banks retire before the SAME2GiB Source loan closes.
Fresh/reused cases pass11.006/10.920s. Traced peaks91.530/91.537decimalMB;
sampled processPSS259.058/273.805MB; exact combined recording peak3.859MB.
Native/external recording and Source counters return0; residualPSS178.337/
215.684MB remains distinct allocator/library/observer memory, not a claim of
physicalRSS release. Actual recording cap is512MiB; no Torch/CUDA was loaded.
Execution/setup185.32s stays within its8min bound; report writing finishes later.

This uses immutable native16b742c7/extension5177ad83, lockbb8 and unchanged
storagePython29549dc9; it does not claim a current-tree build after5b edits.
The long banks are shaped OTHER banks; their own long-terminal decoding/duplicate
comparison maximum is unmeasured. Genuine terminal banks are smaller; candidate
cap64, and C2/C4 samples are snapshots only. This is not32genuine participants,
maxcandidate/fullC16 work, Source qualification or a universal capacity proof.
ScriptSHA87f759766381007d45c0ac3685adb1274da0c6c4b03aac3fea5d16bc569f2d6c.
Durable originals, script, raw observations, helper/fixture snapshots and immutable
wheels are under `runs/rust-player-runtime-v1/memory/retained-v3` in the primary
checkout; temporary counterparts are `/tmp/gfb-retained-bank-v3.*`.

Production/default memory acceptance stays OPEN for actual C4/C2/C16 workload
measurements. The targeted retained-bank condition is measured only in the
explicit v3 range above; no all-maxima promise or new admission guard follows.
Long-own-terminal/all-maxima populations and arbitrary rawAPI histories/strings
remain unmeasured; no new cap or universal capacity promise follows. Actual
Source/lineage/learning/supervisor/CLI integration and approved timing protocol
remain required. No model-quality or performance improvement is claimed.

## Companion/descendant feasibility

Independent preimplementation architecture supplement READY for concrete installed
wheel paths and Linux subreaper/isolated-group descendant retirement. A disposable
Maturin1.15probe PASS confirms `gf_native_runtime/gf-player-supervisor` and manifest
paths in Distribution.files plus executable0755. It is not the actual companion
wheel. Current ManagedServer inherits its host group and pinned Simulator starts
no OS child processes; stdinEOF requests shutdown and does not prove exit. Actual
original-host/adopted-child waits, failure injection, installed companion identity
and host reentry were outstanding after that probe; the engineering evidence
completing this scope follows.

## Original-host supervisor engineering acceptance

Both independent final scoped reviews are READY; Root accepts this component.
Four originalP2s are resolved: cold/continued-operation clocks use their original
phase authority, group signals retain the unreaped original leader, stopACK is
pending exactly once, and missing-companion errors provide tested build/install/
no-sync launch commands. Actual Child.wait/adopted consumption remains necessary;
WNOWAIT observation alone is never a successful receipt. Kernel probes run on
6.18.58.1-microsoft-standard-WSL2, including escaped descendants/SIGCHLDignore/
ECHILD/TERM→KILL. Cold/session hard cutoffs add no grace and fail uncertain.

Final `cargo test --offline --locked --release --manifest-path native/Cargo.toml
--bin gf-player-supervisor --quiet -- --include-ignored`:22PASS0ignore48.47s.
Focused supervisor+legacy scheduling pytest:60PASS1.36s. Root fresh fix-only
native11PASS0ignore42.01s and supervisorPython13PASS0.12s. Static/format checks,
current native/extension/companion/source identities, all3wheel hashes and256
Python files match. Original331Python/197native13ignore results are earlier
candidate evidence; unchanged actor/storage/math suites were not relabeled as
fresh repaired-source runs. The final checks cover real ordinary/HPO component
host pairs and faults, not scientific Source/learner eligibility.

Native source f8671dee9e5cac485b045625decf559c9e036d84bce1ff46ed4060bbb100dc5f;
loaded extension8f633e9b75b00b429927872e3c013dea01994c7eacc3d0694a0c69dcc55f62b6;
supervisor6e53acad327a2c5e95f8d8662c4b5019f8ca00c7b62cb4864bc718f3c16c5671;
lockbb8e9e3c unchanged. Actual runtime/app/native-only wheels and all256 source
files match; separate clean app-only/native-only environments retain optional-free
old help and actionable absence refusal. Worktree venv now contains the actual
runtime wheel. Tested install/launch uses installed `.venv/bin/gf-bots` or
`uv run --no-sync gf-bots`; automatic uv sync may select the companion-free
editable dependency. Build helper remains explicit, never runtime compilation.

New control-only records use canonical UTF8 without a trailing newline, leaving
legacy writers unchanged. Original OFD transfer ACK requires close-only caller
detach in the forthcoming coordinator; no legacy LOCK_UN alias is reconstructed.
Prepared/physical exit are not Source eligibility. Fixed internal numerical/
Source/campaign handlers remain absent and refuse before admission. MLP learning,
new strict outer Source/lineage/CLI, actual final-D/evaluation and all required
performance/default gates remain open.

## Pure configuration slice

Both independent reviews READY; Root accepts the pure closed configuration slice.
338focusedconfig/HPO/entry checks PASS5.31s, including70new cases; Root fresh70
PASS0.78s. Ruff/mypy5/format/diff and actual optional-free old help pass. Strict
six-field declaration, alias/overflow/finite checks, matching-only HPO6 inheritance,
supported participants/Pack/reader and before-work legacy guards are covered.
No native dependency, preset, numerical or Source implementation changes occur.
Ordinary pure validate remains usable. Existing public HPO resource validation6
intentionally refuses its legacy planner until genuine new-host planning exists;
actual host/Source/numerical/public supervisor execution remains future work.

## Local numerical and data support acceptance

Task5c1A independent correctness/conformance READY; all5initialP2s close across
two scoped rounds. Actual parent FD borrowers close only after ACK/positive FD
closure, data-only originals use the same descriptor/pool, writer inventory and
deferred real contexts obey the fixed plan before native admission, and strict
Simulator cleanup consumes original bounds. New context evidence is validated
before complete-group ACK. Absolute Mailbox fast paths now reject expired
publication/export/successful settlement while keeping actual consumer leases.
Normalized HPO shares profile/materializer constraints and existing12GiB codec/
positive bootstrap bounds;10001-update/32GiB STORAGE remains valid. Actual saved
resume metadata binds its ACK hash before RNG/state application; cancelled drains
retain failed health. These changes introduce no new learning/Source authority.

Original793b component evidence:422CPU+oldcompatPASS56.50s with2explicitCUDA
name deselections;22non-CUDA supervisorPASS48.68s/1separateCUDA filtered. Exactly
one admitted ordinary CUDA component,2GiB device/1GiB Source/120s original finish,
compares synthetic3decision Model3/4 current-vs-local named weights/Adam/CPU+CUDA
RNG/PCG64/update/save/restore exactly4.67s. Fresh CPU process AFTER actual original
host reap audits both pairs with Torch import forbidden. One admission, zero
Prepared/Source/final-D receipts. Earlier import-path failure had zero admission/
GPU use and remains setup failure. Durable actual session, arrays, audit, logs and
hash manifest: primary `runs/rust-player-runtime-v1/components/cuda-local-numerics-A`.

Torch allocator peak allocated73455616B/reserved75497472B; pre-exit residual
allocated67108864B/reserved75497472B is not a zero-GPU or physical-total claim.
Actual host exit remains physical retirement. Managed Source peak61181500B→0
is reservation/planning, not PSS. Private test supervisor13eccd99 differs from
installed793b-source companionb00f0417; both actual identities are recorded.
No learning-quality/capacity/performance claim follows from synthetic controls.

Repaired-source final evidence:257CPUcompatPASS12.07s/24deselect;18actual runtime/
storage/last-view checksPASS2.29s/81deselect;7native deadline checksPASS0.06s.
Second config-only round7boundaryPASS0.25s+192legacyconfigPASS1.37s. Scoped Ruff/
mypy/diff checks pass. Root fresh31fix1PASS2.04s/32deselect,7nativePASS0.06s,
7fix2PASS0.23s/39deselect; actual installed/source/file/wheel identities match.
Earlier422/22/CUDA are historical793b evidence, not fresh post-repair claims.

Current native76d940a6e9761a39537511687507ffb36bbfd76de43a69580fc1de7f2412deb0;
extensionfc7e08f9e8f5db6f444e7fc71c79f1b98a581477234269fd9ef909138ce10dc7;
companion7a9fd3b43da69357846b41917ebdc248673b81b06c112b7bd903d3d372f5a676;
runtime wheel1dabdb036932e6f3cde098eec7339207db58a90b187527f66790f306cfb02f93.
Lockbb8 unchanged. R2 changes CLI Python only; numerical code/inner schemas stay.
Actual native→new Source/PERSISTED batch→common PPO/saved commit/G1/cold completion,
outer record issuers, actual CLI/full CUDA resume/evaluation and scaling remain
required. No Source qualification is inferred from the data-only owners.

## Required evidence still outstanding

- New outer collection/Source and local-learning integration around accepted
  actor/inference/thread-storage components, with actual same-owner authority.
- Integrated single-Python learning/process-tree evidence using the accepted
  writer/supervisor components, and new outer Source/producer version checks.
- Model3/4 actual CUDA continuous2 and pause1/resume2, same-input restore and
  independent full Original/Replay/numeric audits.
- Exact approved C8 W sweep, C16 forward/reverse and per-entrance C4/C2 two-pair
  comparison; no adoption decision without all predeclared evidence.
- Independent final correctness and architecture conformance, root disposition.

## Actual ordinary CPU common-loop component

The second fresh ordinary component invocation passes the actual common PPO
loop through cold completion:16 natural games,230 learner decisions,8 Adam
steps, actual G1 saved-pair commit/publication, original consumer/server/writer/
service/cache/buffer retirement, immutable invocation state, Prepared, actual
original-host wait, then fresh CPU full raw/Replay/numeric/saved-content audit.
The private supervisor test reports1PASS/101.01s and exits after the actual cold
child wait. Original host exit0 is cooperative with descendants reaped and no
signals. This is one explicitly configured CPU update, not the required Model3/4
CUDA two-update/resume population, public-entry acceptance or scientific quality.
The returned owner is a generic cold invocation, not an HPO exact-D parent.

The original operation's admission480/execution540/TERM570/KILL585/finish600s
milestones were not renewed. Settings are unchanged ordinary16-game curriculum,
C1/W1/B1,24GiB declared managed host,2GiB same Source account,1GiB recording and
zero CUDA. Declared/accounted peak is not measured RSS/PSS. Before Source reads,
writer/inventory workspace holders are absent and the account uses72,188,328B
including pool base, inference banks and saved metadata. Actual recording native/
external counters are checked at retirement; control metadata allowance remains
owned through final protocol writes.

Successful original root is `/tmp/task5c1b-common-cpu-common-ca94c6daa5b47bea6a874d93f9beb3d1`.
Native source SHA is `e782b8d52977f7253dcd1d8d119282bd26ca2219317113a73d694ad87c0eb524`;
loaded extension `2df104877f482e071fe24790d77eea2aec00fee539a21f998dcf26c121b933d0`;
Python checkpoint producer `3a73cbea2700fd0c75df10497e5f9b4ba7921f062e851665804f08a3334e3a4d`;
seven-package runtime `9a3d152ae5e8f74890c56adb118b7550e7ac394e869d625488e47af8a8d931f3`.
Runtime wheel is `f61971e11f75b4befd6d59fde72e640b4907b548e66063296bb1b208d11d4869`.
Its installed companion is `88df0ab576e5a6e30b7ef0f80598a780473c4de321f448e8cc102b87588a007c`;
the actual private-test original-handle supervisor is separately recorded as
`4cd0abdc51be0528f5e7ce896d8a5126a92a637720f6fa576c8129026645240b`.
This is not evidence that the public CLI dispatched the installed companion.

Root independently checks exact Prepared/Exit/frozen-state/operation/audit/
completion hashes, final saved files,16 episode audits and all350 frozen producer
inventory files. Completion SHA is
`0036ce55ff33b414e98a8b3323bcef1b9a88779fb880f0dfe2cad097280c5f23`;
audit SHA is `a7d8397f81410a3cdb96a08479227da29f3bf173fa3bcf9d45de028c0d10fe77`.
Both original attempts, matching source/binaries, command logs and892 verified
files are preserved under ignored `runs/rust-player-runtime-v1/components/ordinary-common-cpu-1-update`
in the primary workspace. Original absolute provenance paths remain unchanged.

The first invocation reaches G1 and actual healthy closure but fails before
Prepared because the new closed Run5 fieldset omits required `resume_identity`.
Its original exit1/prepared-null stays failed. A saved-record reproduction is
RED before the one-required-field fix and GREEN after it; missing, unknown and
config-inconsistent fields still reject. It is never relabeled successful.
HPO/real restored owners, public entrances, CUDA acceptance, diagnostics and the
complete approved performance/default protocol remain required work.

## Actual Model4 bounded CUDA pause/resume component

The genuine16-game Model4 component passes39.12s with174 collected learner
endpoints, predeclared D64/one update/four Adam steps, paused cold completion,
current-host full prior-data audit and actual restore, then exact-D cold Parent/
metadata3. Named weights/Adam/CPU+CUDA RNG/PCG64 restore equal the saved state.
Resume performs zero additional Adam steps; this does not replace Task6's
continuous2 versus pause1/resume2 cases. Original admission720/execution810/
TERM855/KILL885/finish900s bits remain identical across both operations.

Both actual original-host exits are0/cooperative/reaped/no signals; the last
cold child is actually waited before the private supervisor returns success.
Default shared HPO namespace tickets have real admission and schema3 termination
records, whose embedded exits match the case bytes. Campaign setup is explicitly
a private component descriptor with real CampaignStore/OFDs inherited through
management-Python exec into the actual private supervisor. It claims no Optuna
proposal, closed seed receipt, public campaign/evaluation or quality acceptance.

Root independently checks both complete hash chains, paused eligibleFalse to
complete eligibleTrue with reserved-null parent_metadata_validation, separate
forward-linked metadata3,16 episode audits and unchanged G1 six-file SHA plus
original device/inode/size/mtime/ctime. Actual354 frozen producer-inventory entries
match;356 producer files includes the inventory/control manifests themselves.
482 rehashed files, command log and the four actual shared-namespace records are
preserved in primary ignored
`runs/rust-player-runtime-v1/components/hpo-model4-cuda-pause-resume-D64`.
Original root: `/tmp/task5c1b-hpo-resume-9_icjztk`.

Actual frozen Python checkpoint producer is
`ee8228f8e35b955b58e3db7fa7ff561be4dcd8dab6d44d17ced486fd87c01722`,
seven-package runtime `8851a45cbd01d6124e0ba67b1e5cfad7a831af29f93035cd1ef7d61bf2e9bc7e`;
native source `82f25a60fac3e2d58d0ed9e35e03903da94639f7b142913b23b7a44a7d6c38d2`;
extension `2383e1453e2b71abbfe132443436b64a41d83da7152390d737be86d67ba9578f`.
Runtime wheel `100876b7dab31bbccb43295041747fc78c55d1da4af6eb744cbfe3d7676c8a82`
contains installed companion `9035bfa145e4fa69f9981ff2f4194c195de4b917a07b1ce0988eb4ada65ad31b`;
actual PRIVATE test supervisor is separately
`efc18de33c04b42965cf52afac2a8f9ec001cb9066a3dedc797438f02bee4e96`.
The public CLI did not dispatch that installed companion in this component.

Final completion SHA is
`b8838c7043a4c01097b8da91374947d05c62e93045235af6ca800480f6647928`;
metadata3 `66499813b4e2c4a4050841554e81935a88fd128e88e52aab92f820d777b54ee1`;
audit `241b2542f71cf9539de1a2e2cf586dc662ff9b96a3669a86510ec2f031d3e4d7`.
Process Torch fresh peaks are183,889,408B allocated/218,103,808B reserved; resume
37,836,800B/52,428,800B. These differ from whole-device observations and the12GiB
managed declaration. Whole-device before/after used2063MiB with no compute apps
remaining supports observations, not original-handle termination authority.
No whole-host PSS or universal native CUDA capacity claim follows.

## Final ordinary CPU single-use and cleanup component

After accepted independent-cleanup, control-alias and metadata-original lifetime
repairs, a fresh ordinary component passes101.07s:16 natural games/230 learner
endpoints/one update/eight Adam steps, G1 commit/publication, positive retirement,
Prepared/original exit and fresh full cold audit. The real Source refuses copying,
duplicate persistence and duplicate consumption; steps are0 before the real
update and8 after it and the rejected repeat. No synthetic Source permission is
used. Original600s milestones and24GiB declared host/2GiB Source/1GiB recording/
C1W1B1/zero CUDA remain unchanged.

Root independently verifies immutable byte chains, actual healthy exit, all six
saved files, single-use diagnostic and354 producer entries. Both this success and
the preceding test-wrapper failure remain under primary ignored
`runs/rust-player-runtime-v1/components/ordinary-common-final-cpu-singleuse`, with
884 rehashed files/logs. Failed attempt207260 reaches Source/PERSISTED then exits1
before any Adam/G1/Prepared: its wrapper failed to catch the Source's correct
TypeError copy refusal. Only the test wrapper changed before the distinct retry;
the failed parent is not relabeled or resumed.

Successful root is `/tmp/task5c1b-common-cpu-common-76aa2e47a17d0e74fb8c42e280000573`.
Actual checkpoint Python producer is
`aacdfe6ccaf88be1415c97e620858bc765e8f10396582f51365ec4b786a2e5ca`,
seven-package runtime `d7b0e63c1902fe0a05112bcb791cc6812e8e03be24067d0eab607de4a2d5f485`.
Native/extension/installed/private/wheel identities match the preceding Model4
component; its Python identities do not. Completion SHA is
`5c69e0c090bf67c27c3053f2beb87b702b41e11f33dd5f30d4f4ddaf1c779a19`,
audit `c31000976c9387faaac2e5f1b41e098459b0c0ef351c81a0d472cbacd23e74ff`,
single-use diagnostic `5a0435348ba9abd9bfccb27987c1afddf752eb01f480864352c1bbdc06b18705`.

Final current source has only subsequent import formatting and typing.cast
changes: checkpoint producer `366a82047c1bdc8ba42e3f82a10449e69f7a940ae723c455158797d6c9d0c997`,
runtime `f355dc0ff9d818b164c61ec6560f44d4ef016985a65a9c52838906b24f9e3254`;
48 post-change compatibility checks pass. Prior real cases retain their actual
frozen source and remain historical evidence. The current34-file package is
frozen for independent correctness/conformance review. Generic combined CUDA
planning, disk-retention and healthy empty-collection parity are mandatory before
public activation. Public entrances, real two-update/restart/evaluation/process
acceptance and the complete performance/default protocol remain unexecuted.

## Scoped final B acceptance after ownership repairs

Both independent final engineering reviews are READY after two repair rounds.
Five P2 obligations close: actual returned Parent lock/control liveness; restored
control-graph funding through the final consumer including pre-adoption failures;
raw-audit traceback funding; nested/partially constructed actual FD cleanup; and
refusal of partial-D resume outside the explicitly bounded component plan.
Root freshly runs125 CPU regressions (5.13s), two pure component-boundary cases,
scoped mypy and Ruff successfully. The known readonly NumPy-to-Torch warning
remains; no write to those shared banks is introduced. Whole-suite and formal
acceptance are not claimed.

A final current-source Model4 replay passes38.91s. The actual caller, after
finalize_cold returns, calls Parent.require/state/metadata, checks unchanged G1
six files/stat identity, positively closes it and then releases the parent lock.
Both originals exit0/cooperative/reaped/no signals; last cold is actually waited.
16 games/174 collected/64 used/one update/four Adam steps, with zero additional
resume steps and the unchanged shared900s authority. Root independently verifies
both receipt chains, actual two namespace terminations, metadata3 and saved-file
identities plus354 producer entries, then rehashes482 archived files at primary
ignored `runs/rust-player-runtime-v1/components/hpo-model4-final-parent-handoff`.
Original root is `/tmp/task5c1b-hpo-resume-vldazodb`.

Current producer is `79945b7a9fbea555264f6f00c3f6c05ca8b792e239f3c43dae768b5f060fca9f`,
runtime `c09a462f2091990bfed3d1fab0d4172ca766988a1e5b3e02791091fc28377dbf`.
Native/extension/installed/private/wheel remain82f25/2383/9035/efc18/100876 as the
full identities above; the actual private test supervisor remains distinct from
the installed production companion. Final completion is
`1402c4c00b2ad4075cd464dd3cf9bcf6d0348e0a779af6638395fd53c0ffe297`,
audit `72fcd961c1423d992c5225fdedab5d8d0195d368108f142a559ee193e27a0f71`,
metadata3 `838c45e8564a96b13638e687f647ff1eafd49db037341f2051b3ae89e5f6ee0e`.
Allocator peaks remain183,889,408/218,103,808B fresh and37,836,800/52,428,800B
resume (allocated/reserved, process Torch scope). No universal capacity or PSS
claim follows. This scoped private-engineering acceptance leaves all mandatory
pre-public work, full Task6 CUDA2/resume2/real campaign/eval/process/audit conditions
and Task7 exact performance/default protocol open.

## Generic device-account engineering component

2026-11-20 Root accepts scoped implementation of the reviewed whole-host phase
account and immutable operation allowance, after both independent R1 rereviews
READY. Baseline reviews found two P1 lifetime defects (Source charges refunded
while device-held copies lived; cause/context and nested cleanup omitted) and a
P2 historical path traversal. R1 captures actual numerical owners before outer
Source replacement, preserves Source funding/custody until positive retirement,
keeps uncertain nested errors/owners held, and rejects `..` before either read.
Both original reviewers find all three addressed with no new scoped blocker.
The conformance reviewer reruns the original pure exception-chain probe: graphs
retire and nested owners close once, with original error/poison retained and no
Torch/native import. This is not actual GPU-settlement evidence.

Root freshly precollects the exact CPU selections, forcing empty GPU visibility
and `not cuda`:150 compatibility checks pass6.03s (the known readonly NumPy/Torch
warning), and10 selected pure Source/constructor checks pass0.14s. The latter
selection excludes actual-game tests. Fresh Root Ruff7files, mypy4Source files
and diff checks pass; formal eight-kernel checks are included. Baseline-repair
patch b4df127d6eb118e0fcc660e33282850eb86750ef25991576ac4d6eb27ab14d7f,
inventory6ebdbade855696bf27184affd29ed0c56ab9ce57157be2e11fa7c0a34a792539.
All16 current Source/test files and seven deltas independently match the frozen
before/after metadata. Earlier257/203 and R1 150/10 agent results remain their
own executions, not a new whole-suite claim.

Current Python Source is93aaaf5d30452156351154dcffd1d30d7246e01372209e0bacdcb883009cca8a,
runtime7593c9d9e8069a10fa105bf7f26e7cf1e4f596f9b88760b392f0fbbad494c83e.
Native/extension/installed companion/private test supervisor/wheel remain the
recorded82f25/2383/9035/efc18/100876 artifacts. The account's dimensions, configured
backend allowances and actual tensor storage remain estimates/data observations,
not allocator-reserved/PSS measurements or a hard physical quota. Heterogeneous
account profiles are unit-tested; the actual training caller still uses its own
model self-pool and has no new external loader.

One earlier expanded verification command omitted its CUDA exclusion. Actual CUDA
model/publication and one Simulator launch occurred before Ctrl-C; the original
pytest process was observed exiting2 after169 earlier passes. That interrupted
execution is NOT an acceptance, approved experiment or GPU/performance proof.
The in-process Torch/native owner ended, but no original Simulator wait was
captured. Later OS/NVML absence is supporting observation only: descendant reap
remains NOT PROVEN. Exact command, selected nodes, stdout, original tool-result
transcription and preserved Simulator stdout/stderr hashes are archived in primary
ignored `components/task6b-interrupted-cuda-selection`. Its original pytest temp
case directory was removed by later pytest retention; no replacement/original
identity is fabricated. Subsequent CPU checks use both safeguards and precollection.
No extra GPU/query/cleanup experiment was run to replace the missing evidence.

No successful authorized actual Task6b CUDA/training/end-to-end measurement is
claimed. The separately required genuine supervised continuous2/pause1-resume2,
all Original/Replay/numeric/saved-state audits, one-Python tree, real campaign/eval,
merged telemetry, full24-run timing/default protocol and observation calibration
remain open. This component milestone does not qualify learning quality, arbitrary
physical capacity, public activation or a default switch.

## Retention and healthy-empty engineering component

2026-11-20 Root accepts the isolated `feat/rust-player-runtime-retention` component
under both final independent correctness/conformance reviews. Initial review found
three blockers: a historical reader expected an unproduced completion admission
field, its synthetic positive did not use the actual closed completion schema,
and cold inventory omitted parts of the applicable protection policy. R1 closes
the two completion/history issues and adds complete HPO/evaluation protection;
R2 retains ordinary `inference.json` outside the fully protected set. No persisted
format or warm/physical-retirement policy changed. Both scoped R2 reviews are
READY; no third repair/advisor cycle is needed.

Root fresh precollection selects exactly122 CPU-only retention/empty/frozen-kernel
checks with empty GPU visibility and `not cuda`; execution passes122 in1.95s.
Final Root Ruff12Python files, mypy4Source files and diff checks pass. Earlier
implementer evidence includes417affected checks, R1 182affected/14legacy and
R2 91retention checks; these are not silently relabeled as a repeated whole suite.
R2 patch is6d5a52fa418916f9dbe559dec5f8e154622b654fb3947a8a9712f8a560b790d5,
review inventoryf10d37ae9c0d6ecd00cac16e432c6294fe16466bc9be30e13230a2c8ac43435c.

Before the reader repairs,16genuine original-supervisor CPU control/fault cases
pass45.63s under one exact historical Python producer
b228e942764b87dda6dc268c941cf92823fa38d0a5eaa27d118bf8c94c333573.
They cover SHA/inode replacement, intent/rename/unlink/fsync/stage contamination,
uncertain close, positive preterminal cleanup retry and receipt failure. All16
actual original admission hashes were independently verified. The physical
retirement class is AST-identical after R1/R2; those earlier cases are NOT current
reader/engine-resume evidence. Actual private supervisor
53608d685b147b052a740b5a6d3f5303056e84b0e880596b5353fe17dc61b403 differs from
installed companion136347f0b1e1fc7a9a240ec0edffd8b9df5b44c605065d59a6055c73302d0fac.
Native d4282ec2d1d18d1eb1d9637d6ca697837b9249d1caeee3526dd81f8287d0f333,
extension3497880d025c546b4d13e08eccc01292aef1ab7ab5f41760e27cf6be56b87fe8,
wheel95caed6e7bbb0843f64c27f4850c72cedb0d1e2b4ac8dec8468571d47f965ffc.
The native source change is confined to its private cfg(test) control fixture.

Genuine retired-checkpoint resume, first/later all-natural zero-own-seat engine
capture and extra full engine Raw/Replay finalization remain NOT RUN here and
required integrated Task6 evidence. No actual gameplay/GPU/training or calibration/
performance/default claim follows from these controls or synthetic DATA tests.
The complete approved Task6/24-run Task7 protocol and current producer integration
remain open.

## Healthy-empty engine reachability disposition

Subsequent read-only verifier investigation and independent architecture review
find the genuine first/later all-natural zero-own-seat engine case UNREACHABLE /
NOT APPLICABLE under Simulator072d and the currently admitted paired schedules.
This refines the initial NOT RUN reservation above; it is not an executed PASS.
The [conditional proof](../design/rust-player-runtime-empty-feasibility.md) checks
actual native MLP/singleton capture, storage ACK, journal/Raw/Replay frames and
selection, not just a legacy counter. Swapped learner seats share the repeated
engine seed/first player; each complete pair contributes at least one learner
frame. A dispatched cohort has remainingD>0, so selected=min(remainingD,L)>0.
For8/16/32 games the collected lower bounds are4/8/16. No seed search or changed
schedule/budget/engine/outcome was used to obtain this conclusion.

Generic healthy-empty DATA/ownership/marker and negative cold validation remain.
Revisit and add a genuine engine test BEFORE changing paired seats, repeated seed,
first-action semantics, singleton/frame recording, learner selection or budget
stop. An observed healthy zero under current contracts is an invariant failure.
Full genuine pruned-checkpoint resume and ALL other Task6/7 experiments remain
mandatory. This proof does not establish deterministic training, quality, speed
or physical memory capacity.

## Joined device-account and retention composition

The accepted device79ae8fb and retention44d2bd7 components auto-merge all production
source without a conflict. Root resolves only the three design/evidence document
conflicts by preserving both engineering histories. A cleanup-only test receiver
needs the new inspection-owned raw alias, parent lease and same-pool close hold;
its four failing terminal-close cases become12 passing cleanup/constructor cases.
No production code, learning/record semantics or failure guard is changed for this
fixture adaptation. Independent joined-composition review is READY, preserving
Source funding/empty DATA separation, source-invocation plan/loan binding, strict
historical completion and HPO protection forwarding.

Root freshly precollects exactly270 CPU/data plus12 pure Source cleanup nodes,
with empty GPU visibility/`not cuda` and no actual-game cases. Execution passes270
in6.85s (the known readonly NumPy/Torch warning) and12 in0.14s. The implementer’s
joined mypy15 and Ruff22 checks pass. Original automatic23-file staged bytes match
inventoryc5355a322eee9ff259c621c4df553928fee793ea4d1594b53231b7ee1a74c70b;
sole fixture deltaef6137371cd96d5c840fa097f0cb86261ec521eb507af8cc03527b3d9870744f,
joined evidence inventoryaf911423861ff027d2f401fa21b3606391bf2db7a3b15365a3c7b304bcf051e9.

Two restricted-sandbox attempts (the initial270 selection and an isolated asyncio
executor/I/O-double node) were interrupted with exit130; no completed PASS is
claimed. Faulthandler/logs remain preserved. The isolated node and full selection
pass outside that sandbox, but the cause is UNPROVEN: test order and environment
may differ. No production change or hidden cleanup experiment was inferred.
Installed native82f25 remains UNMATCHED to joined test-sourceD428. Only pure CPU/
data/static checks ran; the final combined native/companion build and actual
Source/CUDA/process/campaign/evaluation/measurement gates remain required.

## Application telemetry engineering component

2026-11-20 both independent R1 correctness/conformance reviews are READY after
four P2 repairs: diagnostic processing escaped into infrastructure shutdown;
snapshot failure left the first profile scope active; update/save scopes carried
non-null batch IDs; delivery observation raced completed retirement. The reviewed
failure-boundary clarification distinguishes healthy framed diagnostic rejection
from real channel uncertainty. Original `_exchange` fail-stop is unchanged:
clearing it could mistake a late diagnostic acceptance for authority acceptance.
Broker processing returns the existing negative reply without poisoning a healthy
service. Profile END is unconditional before fallible snapshots; collection IDs
stay absent outside collection; diagnostic retirement follows the existing positive
relay wait without changing authoritative retirement or holding its lock across
that wait. No scientific/policy/learning schema, capacity or math changed.

Root freshly precollects38 pure failure/lifecycle/race and2 framing/frozen-kernel
nodes with empty GPU visibility and `not cuda`, excluding actual workload probes.
Execution passes38 in0.27s and2 in0.76s. Final Root Ruff6files, mypy4modules and diff
checks pass. R1 six-file delta is
00bf544a4bb71d6442214bf2fb0e947ce4a77c3af47eaa642f650cf010e479b6;
before/after inventories2fec91c79e3a607d5ae1d11f5657fe8021f29c56a39154aac6edec9e26a949b4 /
8616283a7b7fd59e5b791039ea73b1324b60e138fd42c432d4e1469efa426af1.
Both reviewers independently verify38 current/before files and unchanged installed
artifacts. Earlier52focused/83compatibility/32native/18supervisor results belong
to their recorded runs, not a fresh full suite.

The isolated matching release native source is
3ce55bf35e56f19e4be8f978041237df1ffc7d4ab9682cb5f7ca4d7f73d74c93,
extension8dffe4bf6590e977b52c316a42145ce4e01802a84b95419e09ac427ff865cfff,
companion932a154f63161c9c58ae369d25337e3f7596a8e4528bb2b5bd66699bfe1af1d9,
manifest72c3ea84d1cde7ab83c9b39c981978ecd97865147ec2cf6f54e9b797612d4b6a,
wheelbe8187c194efa159d87ba8bb85fc13173eea5771f883c3bb4a653712f23919b4.
R1 changes only Python source/tests; native artifacts remain unchanged.

The earlier six CPU mechanism cases passed under historical producer888d43a9
with their own raw logs/manifest. They tested one legacy endpoint, both native
schedulersW1/W2 and a distinct SQLite FIFO path. Their native admission latency
included preparation before the later deterministic timing repair; those samples
are NOT final matched capacity-wait data. Other later fixes changed profiling,
reply acceptance placement and helper completeness. The historical cases are
therefore not relabeled final R1 actual execution, GPU/profile or throughput data.

Final merged-source actual measurement/process/Source checks, evaluation phase
hooks at the real application owner, enabled/disabled cost calibration, separate
GPU trace population and the unchanged full24-run/adoption protocol remain open.
No measured speed, PSS reduction, overhead guarantee, quality or default switch is
claimed by this bounded telemetry milestone.

## Joined telemetry/device/retention engineering scope

The accepted telemetry6f4143c joins the accepted device/retention ebc2e54 branch.
The sole production conflict is resolved by preserving the complete device parent
and overlaying only its optional forward range/post-forward callback. Both remain
inside the existing Torch lock and row error scope, after forward synchronization
and before sampling/input retirement. Failure holds, partial input custody and
last-input clearing inside the lock are unchanged. The corresponding cleanup-only
fixture tests callback on/off with the same actual accounting object and lock;
all other automatic source joins remain byte-identical. Independent scoped joined
conformance is READY, with no new blocker or optional finding.

Root freshly precollects and runs352 CPU/data/API checks (7.15s,16 deselected,
known readonly warning) and12 pure Source checks (0.14s). GPU visibility is empty,
CUDA/workload probes are excluded and the mixed Source file is never run whole.
Joined evidence inventory48164916f9c3e8dc837e279c1304f1f9a6673ef403f9dac82905777f8d2489bd,
resolution860c60907deedd1dcf489f38395fba189d1c1c7af1d813784c3273786627f4dd.
Implementer mypy24/Ruff28 pass after one authorized intermediate matching native
build: the old installed stub’s two missing-API errors remain recorded rather than
masked. No private harness build or actual controller/game/GPU work was required.

Intermediate actual native source
931417c4cb84d627db25deadffffa69dc1b9651d1d9372d3925ffba3fa994c02,
extension03fa4461c0cfa85105303f6f4e24093e6d69c3685d254f5ad6a66be825345b82,
companion4c866c31923c7ae7293cbd8b80b9577aed3a5d10aad9f6a3ab884c7b213918a3,
wheel184e81d70d0b8f5ea5afeca8445694a668f9667212d5e738fb8971cfad6cd856.
Python Source dc961676c3324f7b29281c5e061a32395deede6ae5bd466a879ed38a77252780,
runtime8161c0dbfab53969541038d26267907443f3119dc86b15a592c5a2b49fcfc3ca.
This static/API matching scope does not certify the later final C1 composition,
Source, actual CUDA/profile, one-Python process tree or performance. Those complete
Task6/7/public/clock/all-original audit/evaluation/default gates remain required.


## Public-entry control protocol engineering component

C1 passes both independent correctness and architecture-conformance reviews after
two scoped repairs. This component adds authenticated campaign control/failure
return, borrowed Store custody, immutable records/cursor CAS, durable terminal
disposition ACK and separate positive parent release. It does not activate public
training or issue scientific Source/Parent eligibility.

Original22 genuine CPU control cases pass under their frozen producer; R1 retains
the four malformed/zero-SCM terminal races and whole65536-byte failure-envelope
checks. R2 genuine REDs reproduce legitimate pending Stop ACK crossing Prepared
and same-number/same-OFD reuse after a close side effect. R2 GREEN settles only
the actual pending ACK then requires EOF; any raw-close uncertainty permanently
refuses numeric retry and retains actual registration until original host exit.
R2 current focused crossing, constructor/close lifetime, four race negatives and
legacy Stop/duplicate/unsolicited checks pass. No full22 repetition was needed.

Current R2 native source is
`bd1945bc38cdad28b88a5479b3507c849a97b0634aeafc04ab47f4123de25591`,
Python source `fad7d023822930d64d5153be9bc0ac25387e8f20ac7ed88f47317e43a43ee80a`,
extension `d419746712dc2e8190d8129064dfe78315ace7b8cc03e90b342e2328b3f5702a`,
installed companion `f67ee0c10efb0d97d60aa740d7ba6a29bc59a9e5c2925b10086458794093116b`,
actual private cfg(test) supervisor
`faf23b2b937e9634e0bb4aef11c8a08f2208eda9a4f9b2abf3d8057baaaf739d`,
wheel `ae028e08fda1aeb26ba8ff3050ca9cbab5ced9284d70c8c939c1374c90a208e0`.
The actual admission names the private executable; it is distinct from the
installed production companion. Historical candidates are not relabeled.

Root fresh PASS: precollected57 selected Python cases (0.40s),3 current native
terminal checks, Ruff5 source/test files and mypy4 source files. Exact Python
selection is tests/test_rust_player_entry_protocol.py, test_hpo_storage.py,
test_rust_player_supervisor.py, hpo_config::historical_model_and_eight_frozen_kernels_remain_exact,
and rust_player_learning::fixed_bootstrap_passes_real_loaded_configuration_by_keyword;
CUDA_VISIBLE_DEVICES empty and `-k 'not cuda'` both enforced. Native command uses
the private release executable with `terminal_ --nocapture --test-threads=1`.
Root verifies all16 frozen file hashes before documentation-only status updates.

R2 measured supervisor lifetimes total34.339741s, added to prior nominal332.485244s.
The prior reconstructed intervals are ESTIMATED and post-output exit tails UNKNOWN;
nominal353.175015s remains within original720s, with no extended grace or renewed
operation clock. Fixed CPU cases contain no games/PPO/scientific audit or GPU.
Actual Stop crossing reaches genuine Cold protocol with component-only ineligible
data. Next-bind proves cursor/bind mechanics then deliberately omits second
admission/completion; it is not two completed training operations.

NOT RUN here: public application routing/campaign6/two complete scientific
operations with intervening legacy evaluation, actual evaluation Source/Parent
import, full Raw/Replay/numeric/saved audit, CUDA2/resume2/process acceptance,
performance populations and default adoption. These remain mandatory downstream.


## Joined C1 and current runtime foundation

Both independent correctness/conformance reviews are READY for the frozen19-file
join of C1 `33de33e1297eb6668c644218fceb220c65fe5d39` onto
`d0cec42322db3fb2c991621ae38a663478967572`. Two native conflicts retain the
complete diagnostic implementation/C1 tests, failed-return handshake argument,
actual wait-instant retirement event and same physical/failure chain. The new
campaign-return control host is classified startup/startup; genuine cold audit
remains cold/cold, using the existing phase vocabulary. Diagnostics remain off
behind their existing guards, with no new per-action work or authority.

Root fresh PASS: precollected57 CPU Python checks (0.52s), three native terminal
helpers and one observation-ID helper using the current source-compiled debug
ELF. Both empty CUDA visibility and not-cuda filter apply to the mixed Python
selection listed above. Worker offline/locked2-job compile and preservation
assertions also PASS. Initial worker socket tests failed sandbox EPERM before
protocol logic; approved retry and final-source checks PASS without a workaround.

Pre-documentation-update inventory is
`cf80f6c807cb0990a434a05e209fdfe95398b0ffcd6c75981fcceca5ab7805f9`;
full patch `12ae3b063908aa48f831a8e7edf637dcc7fc1a6b4b0e2eef7c15dfc189239abd`,
Native pair patch `7fdee41f21d07734802cd20237c6047a9c8c00e582e843ef924920fa839ae18b`.
All19 frozen hashes match independently. Installed R2 wheel is older and is
explicitly NOT source-matched to this join. No wheel installation, actual host/
control/Original/game/GPU/training/evaluation/profile/calibration/full-suite run
was performed for this mechanical join. Final all-source packaging and all
remaining public Task6/Task7 conditions stay required.


## Explicit clock/configuration and ordinary-launch factory component

C2a passes both original independent R1 re-reviews. The original parser P2 closes
by applying the existing merged non-P type validator once before timing arithmetic.
The factory reconstruction P2 and adjacent companion interaction close by issuing
private facts only inside actual preparation; constructors/dataclasses.replace
leave them unset. Execution uses the original discovered companion and established
final ELF-hash/FD/sealed-launch checks. No extra discovery/handshake I/O is added.
One canonical manifest string is retained (573 serialized bytes in this installed
metadata, not a heap/RSS measurement); it adds bounded startup comparisons only.

Witnessed RED/GREEN: six malformed-timer failures plus three prior passes become
nine PASS; eleven reconstruction/manifest failures become eleven PASS. The exact
original230-node list/order is preserved and PASS. Root freshly precollects and
runs all250 CPU checks:250 PASS/4 CUDA-labelled deselections,3.73s. Ruff3/mypy2 and
all7 frozen hashes PASS. After the C1 dependency fast-forward, Root freshly runs
the127 relevant factory/clock/protocol/storage/bootstrap CPU nodes:127 PASS/3
deselections,2.80s, plus mypy5 PASS. Empty CUDA visibility and not-cuda filtering
are both enforced. Pure preview's fresh CPU interpreter checks all4 configured
role combinations with no Torch import; excluded CUDA-named nodes are not claimed
as executed device tests.

Root250 command (precollection is the same plus --collect-only):
```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_rust_player_clock.py tests/test_rust_player_public_entry.py tests/test_rust_player_config.py tests/test_hpo_config.py tests/test_hpo_numeric.py::test_all_eight_frozen_kernels_remain_exact -k 'not cuda' --basetemp /tmp/gfb-root-c2a-r1-cpu
```
Joined127 selection uses clock/public_entry/entry_protocol/hpo_storage/supervisor
files, historical-model/eight-kernels config node and the single fixed-bootstrap
LoadedConfig-by-keyword node. No whole learning/Source suite is selected.

R1 seven-file inventory is
`b4775b73291a832538d4c59313455edcaf71135c24784a49850e84adabedb2fc`;
repair patch `7ad462985529f529172ab3ede41674d5549dbea260a66b3357916ef162316b36`.
The factory snapshot reviewed at d0 records checkpoint Python source
`fe15cb6b955b802bb9d66efb32856a065c72865a3884162dd9324d6e7a4784df`
and seven-package runtime
`86b4e06d61e89998f190e383cd6a39cfbcfb4af4b155ab089f5dd87dbe8a578f`.
Its installed native/extension/companion are the previously recorded931417c4/
03fa4461/4c866c31 intermediate artifacts. They are NOT relabeled source-matched
after joining C1; final all-source packaging remains required. Actual factory
tests use real temporary run owners/independent-open competition/seals/FDs and
mock the final exec. Existing sampling-runtime BUILD metadata may import CPU
Torch only during actual preparation; no device query/context/tensor is used.

NOT RUN: actual final exec/HostControl/admission/Source/Parent, public commands,
GPU/game/Simulator/PPO/campaign/evaluation/resume experiments, native build/install,
full suite, calibration/profile/primary performance populations or default adoption.
No numeric P default, tail sufficiency or physical capacity claim follows.


## Public ordinary routing and campaign CONFIG foundation

C2c passes both original independent final reviews after R1/R2. The ordinary
timed train branch calls the real sealed factory; final exec is mocked in unit
evidence. Cold output uses actual typed-owner guards and serializes scalar DATA
while live, then closes result, releases parent and closes control before stdout.
Pinned CONFIG/seed snapshots, real CampaignStore OFDs, exact initial soft/hard
fields, cursor CAS and limited known-native-ACK failure history are implemented.
Known-ACK routing tests use a labelled DATA-boundary double with real new parent
ownership; they are not new actual native ACK or Source proof.

Two original P2s close through native6 conditions explicitly using legacy-fresh
evaluation None, and existing real materializations checking P for all four
planning callers. R2 closes a resource cache defect: scientific identity excludes
resources, so a requested canonical512MiB context preview could be reused for
an actual1GiB in-space proposal. The real CONFIG/Optuna regression observes fixed
805306368 versus1342177280 bytes RED, then GREEN. Reuse now requires the identical
actual requested object; otherwise actual selected resources are calculated.
An optional full-state clone is removed by scalar extraction from already-held
validated state after Parent.require. No measured memory/speed claim follows.

Root freshly precollects and runs333 CPU/config/file nodes:333 PASS/4 CUDA-labelled
deselections/29 known Optuna group warnings in13.54s. The original327-node order
is preserved; added six are actual trial/control P60/P59, real conditions and
in-space resource regression. Both CUDA visibility empty and not-cuda filtering
are enforced. Root Ruff11/mypy7 and all11 frozen source/test hashes PASS. Worker
R1 original327 and R2 focused29 remain separate preserved executions.

Final inventory `1dbf08a2e1ac6aefabc456b49955325f7790376575d8195d2dbb2998ebd5981a`,
combined repair patch `ac36abcbbe317a8eb568b29d4e14dd16c6960f70ec6a2e16705085d0c71c3b71`,
R2-only `9e1905bfff4c6760b943165b6217a21f82b2a4b4693d8ef2277ad76e99b214bb`.
Checkpoint Python source
`68dd3b40eb6521a88497cde15042e41bd4467d71eb90af7e896a5056920920bd`,
seven-package runtime
`f2dcf4f3ab00510c0e77441bde2032c0f7bdb82300a6cf9d59a5c198b2b2834e`.
Current native source
`381ef85a15c2843f6e6b56847226a8fd65391182914dd0f175fe95dbcb177944`
remains UNMATCHED to installed931417c4/extension03fa4461/companion4c866c31.
No current native protocol certification or wheel installation is claimed.

PUBLIC HPO run stays guarded; successful-history/source validation is explicitly
unsupported by this foundation until genuine C2b/C2d composition replaces that
limited guard. No run_campaign stub is provided. Actual host/admission/Source/
Parent/GPU/Simulator/training/evaluation/build/profile/calibration/performance
and full-suite checks are NOT RUN here. Full actual Task6 and unchanged Task7
populations/default criteria remain mandatory.

Component raw control evidence is durably preserved under primary ignored
`runs/rust-player-runtime-v1/components/entry-control-33de33e-e9de08c` (3139 entries,
105 historical roots, final manifestd59e5891fb488d628217e094bc89150ef042dafeb54e2d58932115464a831333),
including hash-verified R2 actual private executable and original wheel filename.
These copies are DATA, not Original FDs or current authority. Earlier candidates
remain historical. C2a review/source evidence is separately preserved under
`components/public-factory-5baa086`, manifest258ab5540262a0450dfdcf62b6b42317ccf3d1e65c7ce5cdcdc130bbf59d3b00.

## Installed-wheel ordinary CLI CPU smoke

At `55cd288507dd0cdcf98ed237f356006d32abd23f`, the installed `gf-bots train`
command completed one timed Rust Model3 CPU update. Its actual original host
issued Prepared, exited normally with descendants reaped and no signals, and
returned through the independent Cold audit. The audit checked 16 games, 230
learner decisions, eight optimizer steps and the full saved inference/resume
pair. The original observer Child exited 0. In 1,358 process samples, the
maximum observed simultaneous Python process count was one. This is sampled
process evidence. Wall time was 139.024 seconds and peak complete sampled PSS
was 903,681,024 bytes (136 peak-eligible PSS sweeps). This was a functional CPU
smoke rather than part of the CUDA comparison.

The exact command, effective config, original receipts, phase records, raw
observer samples, rebuilt Simulator binary and runtime wheel are retained at
`runs/rust-player-runtime-v1/components/public-ordinary-cpu-55cd` in the main
checkout. The 129-file inventory SHA256 is
`f25f3366d4ec9f2459c8986d855cd9f97c1911c342d83a2ed40f12ae52d08672`.
The cold audit hash is
`e73e9a0e1362e2f6f9ec49d5196425124d20f32e96f0bbe3291e6527fb0d24c9`,
and restored Simulator 072dabb matches the original ELF and spec hashes.
The new wheel `0822624f78ecd5163cf3b0f1baa580530f8f63104c6c838fa8f7682965e84b3e`
contains source `381ef85a15c2843f6e6b56847226a8fd65391182914dd0f175fe95dbcb177944`,
extension `cf084cc9b5c6d32dce197cdd3f90cead18e2d5523be285c1b0d873913282372a`
and supervisor `96f064a2e52f4621fc4616b948e6147e2fd1115e1868addb2018def1bf3e9b02`.

The CPU result does not qualify CUDA training, resume, HPO evaluation or
performance. At the time of the GPU capacity check, NVML reported 85% GPU
utilization and 1,477 MiB free of 16,303 MiB. The formal comparison has not run.


## Evaluation DATA bridge review and acceptance

Both independent final reviewers are READY for the combined repair. The bridge
rejects reuse of the retiring parent's pool or budget during live handoff,
tracks cause and context through unwind and recapture, and rejects ownership
back-edges without losing independent cleanup consumers. R3 produced 12
reproductions before the fix and passed all 12 afterward, plus 87 related
checks and the separately selected uncertain-FD control. Root ran 224
precollected CPU DATA and retention cases successfully in 4.50 seconds with
CUDA hidden. Static and eight formal kernel pin checks passed in the frozen
implementation review. No genuine evaluation, historical Source or seed2
qualification is established by these component checks.

The original evaluation branch milestone is
`c7c3c80454b3ab1d28b5e36305540dd691b10750`; main feature join is
`0bfffaa64c9b2ecb1480573ae11969bb6c6f0156`. Final reviewed inventory SHA256 is
`023848f26221a5fb490d3882f4377522a505eda68305d0d729eea3cae4608b67`.
The combined repair SHA256 is
`f7bfca1c0f8e63f554b152e6e8f76cee04aed489b80bd0fa06b197e5419f712e`.

## Fixed scheduler and launcher integration checks

At `c60be4c12e32236559ff1a0e869119f7f43f4052`, the public ordinary Model3 CPU
command also completed with the Fixed scheduler, C2/B2 and W2. The matched
wheel and pinned Simulator from the previous smoke were reused. The actual
original host exited cooperatively with code 0, all descendants reaped and
no signal. Its Cold audit validated 16 games, 230 learner decisions, eight
optimizer steps and the full current saved pair. All result receipt hashes
matched. The observer recorded 139.254 seconds, peak complete PSS 929,482,752
bytes and at most one simultaneous Python in 1,361 process samples. The two
CPU runs had different concurrency. They provide functional coverage only.

The 129-file run inventory at `components/public-fixed-cpu-c60be4c` has hash
`50189538c9738be6fc7bbc6d0e50a986816c0e58abf3c25f2c7af4a95b80974c`.
Cold audit hash is
`12426a1824ae75e69294d4f2c4ef2eeed8eca00838e52d9d94499a6c934fac32`.
Runtime producer is
`b85eb862260e7d1c7064927b5638d73b00e6449f15b69835eb2d9eaf9314a748`.
The initial config-only check rejected inconsistent declared concurrency.
It is preserved with the corrected config and proof of the single actual run.

The fixed HPO learning-operation launcher passed both independent reviews and
180 fresh CPU cases with the current companion. Its isolated milestone is
`a7cc0251fabaafbd5c8bdac775fb518405d27a80`, joined in
`c60be4c12e32236559ff1a0e869119f7f43f4052`. The verifier tool, actual HPO
execution and final CUDA acceptance remain in progress.

## Campaign training and terminal foundation

Both independent reviewers accepted the training and terminal foundation with
its R1 repairs. Root ran 188 precollected CPU cases with six excluded CUDA
cases. The actual parent scope is consumed before Prepared. A current admitted
invocation is published before learner or Source construction. The failure arm
excludes typed resource, accelerator and uncertain cleanup failures. Terminal
settlement retains the original cause and orders ACK, positive borrower close,
parent release and cursor update. The public campaign controller and genuine
integration verification are still required.

The public Fixed CPU pause at update two produced actual deletion receipts for
five generation-one payload files under the original acknowledged parent. Resume
verified unchanged saved weights, Adam, CPU RNG and PCG64 at step 16 and finished
its next collection. It then failed before publishing new accounting or starting
an update. Its original process exited 1 without signal and its descendants were
reaped. The archive at `components/public-pruned-resume-cpu-c60be4c` has 285
files and inventory
`ea1470875145436a276bedb1cc41aa1d2d09b05eb1b016946c27cceb19848cdd`.
The post-collection diagnostic incorrectly read a fresh-save field from actual
restored policy objects. R1 counts only dedicated save-policy reservations.
Restored metadata remains in the existing shared Source charge. The actual
failed parent stays ineligible. A fresh positive pruned-resume pair is required.

The earlier frozen `c60be4c` source completed a bounded ordinary Model3 CUDA
continuous-two-update case. Its actual original exit and independent Cold audit
passed for 32 games, 401 learner decisions, 16 optimizer steps and the saved
CUDA RNG vector. The original measured wall was 236.015 seconds, with maximum
one Python in the recorded process samples. Artifacts are under
`components/cuda-restore-c60be4c/model3-continuous`, inventory
`2eca670e02a4d0f1bc5b6b06e1409ce43a0acdc69a9b691cf4eba4c77c88f746`.
This is bounded correctness evidence before final integration. The standalone
Model4 case was quarantined after its first collection and before numeric output
when the source quota could not admit codec scratch. That failure and the
separate controlled diagnosis are preserved in the same component family.


## Public HPO controller and evaluation wiring

The Fresh evaluator passed both code reviews after its receiving-account
repair and joined in `67f75c660c63289b95238fec7d888a4d99a64497`. Root verified
163 precollected CPU cases and 31 existing pure role and linkage controls.
The configuration charge begins before full imported-history validation.
Model metadata and bindings use the same receiving phase and positively
retire before the independent sequential saved-evaluation verifier.

The final public controller passed both independent reviews after the
pre-cursor interruption repair. Root checked 60 precollected final CPU
cases after 142 covering checks on the original candidate. The fixed entry
uses actual scoped control and Store APIs and returns from each training
host after Prepared. Genuine no-work, successive training and Fresh
evaluation and successful-history resume remain runtime requirements.

## Measured Model4 Source and Cold diagnostics

At 2 GiB configured Source, actual 32-game Model4 training refused a
328,614,493-byte numeric codec request while 1,861,907,296 bytes were charged.
The reviewed codec repair uses supported narrower shards when needed. Its
real retry saved two shards, then snapshot export requested 520,242,624 bytes
while 1,973,595,622 were charged. The original 2 GiB cases remain failed and
are retained with actual physical termination.

Before formal measurements, the matched effective Model4 Source declaration
was set to 2,560 MiB through preprocessing and operating fields. Measured
needed capacity was 2,493,838,246 bytes. The unchanged legacy C16 24 GiB RAM
planner estimates 25,504,779,952 bytes for the matched configuration. RAM,
CUDA, recording, workloads, order and thresholds are unchanged. The rejected
3 GiB preflight and final declaration are recorded in the primary protocol at
`model4-source-allocation-clarification.md`, SHA
`f242ab077e72220db8faf1e228edc2da6ffc541dad2ef2a952e8aeecfd9a4803`.

The actual 2.5 GiB Model4 case reached two committed updates, 628 used
decisions and 24 Adam steps. Its original training host retired cooperatively
with exit 0 and descendants confirmed reaped. Cold reconstruction was refused
because it received the live 512 MiB numeric cap. The first recorded update
requires 583,721,519 bytes by existing read estimates. An external CPU probe
used the same reader on the exact archive with actual unused Source capacity
and decoded 331 endpoints. The earlier failed parent stays unqualified.
Fresh complete Cold validation under the corrected allowance remains required.

## Actual public supervised campaign and resume

At `b400457637b0f9911051f56f61ed904d702cf7d9`, the public schema 6 command
passed a genuine no-work CPU session and an actual resume from initialization
before its initial cursor. Each used a real original Store and supervisor,
recorded positive control exit, and issued no training admission or Prepared.
The combined 39-file inventory is
`ef3a2a084ef76bb64f494af0852f2833e6f7e0132b9ce16f1d4d14534fac3ba0` under
`components/public-campaign-acceptance-prepared/no-work`.

The public two-seed CUDA case completed seeds 17 and 19 in one supervision
session. Each trained on 16 games to exactly 64 used decisions and four Adam
steps. Each then ran four actual Fresh evaluation games with two GameSeeds
and both seats. Cold verification audited the raw originals, Replay, numeric
and full saved state and issued both actual closed training receipts. The
actual first training host exited normally before evaluation admission. Its
original evaluation job physically retired before the second training admission.
The final study tell is COMPLETE, with its original evaluator receipt verified.
The outcome concerns lifecycle and receipt correctness for this small panel.

The observer saw a maximum of one Python process during the actual measured
training spans, over 179 and 230 samples respectively. Evaluation may have
more worker processes. The second training host also ran the preceding
control and evaluation stage, so its full process lifetime cannot substitute
for its actual training span. Scope is bounded by that operation's host
startup and original host-reaped records. Whole campaign wall was 247.671 s.
This is functional evidence, outside the comparison population.

An actual subsequent CPU resume audited completed history under the same
producer and exited 0 with confirmed control retirement. It did not add
training or evaluation admissions (two of each remain). Whole resume wall
was 148.914 s, including full history revalidation. The 344-file archive at
`components/public-campaign-acceptance-prepared/two-seeds` has inventory
`2bf6abb4ad955877087396f9a1f3478109882153ce7be9a4d2969329245957ef`.
Actual per-seed audit hashes, process windows, ordered admissions, physical
record DATA copies and raw observer samples are retained there.

The scoped Cold decoder repair is joined in
`bdd9a564761c37255f036133d2e16bbdf120b6bc` after that history check. Both
independent reviewers accepted it, and Root passed 17 CPU cases. The same
admitted pool funds actual independent decode workspace with its remaining
capacity. No live training limit or numerical recipe changed. Fresh large
Model4 Cold, CUDA/ordinary restoration, fault scope and formal comparison
remain separate acceptance requirements.

## Final-source CUDA restoration and fault acceptance

All six observed functional invocations used Bots
`b603e1610a45ed9ac2420cb9a6468662d689ee91`, the matching installed
`381ef85a` native build and the pinned Simulator `072dabbc`. Model4 used
Fixed C4/W2/B4 with 2,560 MiB Source. Model3 used Fixed C2/W2/B2 with
the actual ordinary 16-game preset. Each pair reused its exact configuration,
source build, original clock bits and original absolute deadlines.

| Lane | Invocation | Natural games | All-seat decisions | Adam steps at exit | Whole wall, s | Peak complete PSS, bytes | Samples witnessing its one training Python |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Model4 | Continuous | 32 + 32 | 652 + 567 | 24 | 133.743 | 2,553,847,808 | 937 |
| Model4 | Pause | 32 | 652 | 12 | 77.186 | 2,515,604,480 | 545 |
| Model4 | Resume | 32 | 567 | 24 | 104.018 | 2,501,997,568 | 665 |
| Model3 | Continuous | 16 + 16 | 436 + 323 | 16 | 365.424 | 2,500,593,664 | 3309 |
| Model3 | Pause | 16 | 436 | 8 | 218.066 | 2,478,065,664 | 1976 |
| Model3 | Resume | 16 | 323 | 16 | 121.552 | 2,464,660,480 | 988 |

Every original host retired cooperatively with exit 0, confirmed descendants
reaped and no signals. Model4 has three independently checked physical GPU
namespace termination receipts. The original-command observer supplied positive
Child.wait and sampled a maximum of one Python in each original training span.
The independent verifier checked the committed Original, Replay and numeric
audit chain, full current saved pairs, and current-operation telemetry. Each
resume matched named weights and Adam, CPU and CUDA generator state, and PCG64
at its saved update boundary. Model4's two committed updates use 628 decisions
at a configured D10000 and remain healthy incomplete and ineligible for HPO
evaluation. The actual runs and verification data are in
`components/final-source-cuda-b603e16`. Shared GPU load varied in these
functional cases. The formal timing populations supply the comparison evidence.

The fresh CPU pruned-checkpoint case under the same source persisted five
generation-one payload deletions with actual receipts and kept its
`inference.json`. It paused at two updates and resumed to three updates and
24 Adam steps, with full saved-state and independent readback. The observer
measured 224.206 s and 127.456 s. The readback spans both original receipts and
reports the resumed generation's 16 actual natural games and 257 all-seat
decisions. The earlier failure remains preserved in its own parent.
Evidence is in `components/public-pruned-resume-repaired`.

The final focused matrix passed 41 CPU cases (16.10 s), 13 current Rust
kernel/namespace cases (41.23 s), and two explicit CUDA checks (2.18 s).
CPU coverage includes both A/B schedulers at W1/W2, repeat command identity,
delayed ACK, queue ownership and actual storage failure injection. The CUDA
nodes verify actual response cancellation and initialized RNG preservation.
The installed public ordinary route additionally recorded actual SIGKILL of the
original training host, reaped the known descendants and issued no Cold or
completion. The case-specific HPO fault harness injected a typed AcceleratorError
after a successful CUDA row under actual public admission. It retained the
failed original and complete physical retirement, issued zero evaluation or
completion receipts and recorded zero objective values in SQLite. Optuna retains
the unresolved RUNNING trial rather than publishing success.

The same public CLI refused a missing companion inventory and a tampered
executable from isolated installed-wheel test environments before acquiring a
parent. The app-only wheel's old help commands worked with just its base
dependencies. Offline package resolution could not find NumPy, so the basic
environment reused its exact installed dependencies as copies and validated
that optional modules were absent. The successful case and failed setup log
are preserved together under `components/final-fault-b603e16`.

Independent correctness and architecture reviewers accepted the frozen merged
production code. The correctness reviewer separately inspected all six original
observations, clock/restore and failure artifacts and accepted the functional
gate. Neither read-only review is a substitute for the measured results above.

## Physical comparison and entrance decisions

All 24 primary and 12 calibration commands used the frozen Bots
`b603e1610a45ed9ac2420cb9a6468662d689ee91` producer, installed Native
`381ef85a`, matching companion `96f064a2`, Simulator
`072dabbca1458c9526897781d2fd64edeeb4f9ca` and Simulator binary
`76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d`.
The RTX 5070 Ti observations, boot identity, exact command, configurations,
physical waits, sampling coverage and independent readbacks are archived for
every case under the primary checkout's
`runs/rust-player-runtime-v1/populations/b603e16`. The observer binary hash is
`49194927a495ee85dc4f6e3d0a7e2d666cb495707f6710d5b91b11bcbf562882`.
All primary commands and audits passed. Each Model4 command kept its bounded
D10000 lineage healthy incomplete at two updates.

Model4 used width 192, seed 17, four epochs, minibatch 128, 32 games per update
and formation wait 1 ms. The declared managed limits were RAM 24 GiB and CUDA
12 GiB. The primary C8 runs used 4 GiB recording and primary C16 used
5 GiB. Both separate calibration blocks used their predeclared 5 GiB. Both
shared HPO and native used 2,560 MiB Source. C4 follows the existing 4 GiB
recording preset, and matched ordinary C2 follows the existing 2 GiB capacity,
width 128 and 16 games per update. The recording clarification was fixed
before deploying or profiling either entrance.

### C8 scaling

| In order | First collection games/s | Second collection games/s | Combined games/s | Original wall, s | Peak PSS, decimal GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy before | 0.934 | 1.320 | 1.094 | 78.259 | 9.495 |
| Tokio W1 | 2.830 | 3.642 | 3.185 | 85.497 | 2.563 |
| Fixed W1 | 2.886 | 3.559 | 3.187 | 84.562 | 2.555 |
| Tokio W2 | 3.039 | 3.426 | 3.221 | 85.640 | 2.551 |
| Fixed W2 | 2.772 | 3.477 | 3.085 | 86.124 | 2.563 |
| Tokio W4 | 2.937 | 3.490 | 3.190 | 84.513 | 2.585 |
| Fixed W4 | 3.311 | 3.987 | 3.618 | 79.662 | 2.587 |
| Tokio W8 | 2.977 | 3.606 | 3.261 | 84.396 | 2.599 |
| Fixed W8 | 2.865 | 3.606 | 3.193 | 85.235 | 2.602 |
| Legacy after | 0.907 | 1.306 | 1.070 | 80.276 | 9.492 |

Tokio W2 has the lowest peak PSS among its cases within 3% of its fastest
collection. Fixed W4 has the highest Fixed rate with the next rate outside
the 3% band.

### C16 scaling

| In order | First collection games/s | Second collection games/s | Combined games/s | Original wall, s | Peak PSS, decimal GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy before | 0.953 | 1.343 | 1.115 | 79.527 | 14.560 |
| Tokio W2 | 2.736 | 3.643 | 3.125 | 86.069 | 2.598 |
| Fixed W4 | 2.947 | 3.576 | 3.231 | 84.181 | 2.603 |
| Fixed W4 | 3.020 | 3.584 | 3.278 | 83.763 | 2.604 |
| Tokio W2 | 2.975 | 3.629 | 3.270 | 83.811 | 2.596 |
| Legacy after | 0.994 | 1.332 | 1.139 | 78.422 | 14.568 |

Each scheduler's rank uses 128 actual natural games divided by the sum of
its four collection intervals. Tokio W2 measures 3.195545 games/s with
maximum complete peak PSS 2,597,924,864 bytes. Fixed W4 measures 3.254220
games/s with maximum complete peak PSS 2,604,149,760 bytes. Both fall in the
prescribed 3% band. Tokio W2 is the fixed deployment candidate under the lower
measured PSS tie-break.

The exact reports retain both collection ordinals, all-seat accepted
decisions/s, search candidate sizes and cutoff reasons, effective batch
histograms, event-page and inference round-trip p95/p99 bounds, PyO3 calls,
host stage timing, native ingress and writer FIFO or legacy RPC queue
occupancy and waits. The process observer and derived report preserve sampled
CPU ticks and actual thread observations per process, with role labels where
observed. Shared host roles lack per-thread CPU attribution. Complete PSS
peaks are physical process sweep observations and remain separate from planned
resource grants. See `primary-by-update.json`, `verified-primary-manifest.json`
and `selector-results.json` in the archive.

### Paired deployment gates

| Entrance | Pair | Collection rate, new/old | Original wall change | Peak PSS old → new, decimal GB | Raw 3% and PSS criteria |
| --- | ---: | ---: | ---: | ---: | --- |
| HPO C4/B4 | 1 | 2.927× | +8.079% | 6.940 → 2.532 | Fail wall |
| HPO C4/B4 | 2 | 2.929× | +9.038% | 6.953 → 2.539 | Fail wall |
| Ordinary C2/B2 | 1 | 7.764× | +1.488% | 5.627 → 2.496 | Pass |
| Ordinary C2/B2 | 2 | 8.219× | +2.451% | 5.632 → 2.496 | Pass |

The actual selected Tokio W2 route passed its additional CUDA pause and
same-clock resume checks at both deployment entrances. They restored exact
named weights, Adam, CPU and CUDA RNG and PCG64 state, then audited the last
update using the existing original and numeric rules. These cases remain in
`components/final-selected-tokio-b603e16` alongside the continuously run
deployment candidates and preserve separate functional evidence.

### Observation cost

| Calibration | Method | Collection rate on/off change | Whole wall on/off change |
| --- | --- | ---: | ---: |
| C8 | Legacy | −1.689% | +3.041% |
| C8 | Tokio W4 | +1.399% | −0.067% |
| C8 | Fixed W4 | +5.655% | −1.063% |
| C16 | Legacy | +0.763% | +2.618% |
| C16 | Tokio W2 | −2.113% | +1.064% |
| C16 | Fixed W4 | +0.148% | +0.040% |

Before the primary runs, GPU-device readings were 4–25% at C8, 16–26% at
C16, 16–20% at deployment C4 and 17–23% at deployment C2. Corresponding
free-memory ranges were 7,750–13,135 MiB, 10,600–10,812 MiB, 10,631–10,702
MiB and 10,570–10,701 MiB. These are actual pre-command observations on the
same shared card. The original archive retains the conditions of every run.

Every calibration invocation audited 32 natural games and 652 all-seat
acknowledgements. The matched one-update estimates combine instrumentation
effects and run variability. The observed legacy whole-wall changes exceed
the ordinary pairs' 1.51 and 0.55 percentage-point margins. These Model4
calibrations leave the observation impact at the closely passing Model3
margin unresolved. Raw timings remain as measured.

Keep both entrances explicit opt-in. HPO exceeds the approved wall increase
in both deployment pairs. Ordinary's measured pairs meet the raw thresholds,
with unresolved observation bias at the acceptance margin. Existing presets
and historical parent bindings retain their recorded execution modes.
See the [explicit configuration](../design/rust-player-runtime-integration.md).

The post-exit verifier repairs followed actual observer and old-layout
failures. It reads both enabled and disabled sampling from their original
records, retains unknown executable coverage, projects only the exact named
typed participant journals from validated legacy packs, and recognizes the
actual shared coordinator's invocation-bind transition from adjacent closed
original-host phases. All 36 original measurements were independently
re-audited without a timing rerun. The final nonoverlap tightening was also
rechecked against all ten exact legacy enabled records. Their complete
telemetry was unchanged. Earlier failed readbacks remain in the archive.

## CUDA activity profiles

The initial five original profile commands exited normally. Their three-callback
profile receipts marked empty files as complete. Those receipts and empty
original traces remain under `populations/b603e16/profile-*`. A CPU-only
probe reproduced the default Kineto export failure on the retained file FD.
The pinned supported Python exporter wrote a valid trace through that same FD.
A separate CUDA worker probe then exported 20,965 actual bytes with correlated
GEMM and HtoD/DtoH events. The raw probe and hashes are under
`components/profile-export-probe-b603e16`.

The reviewed repair in `f6c8e458a81d4c909eb9c66f289d42b8fc8f0b57` uses the
synchronous Python serializer through the original retained file FD. Success
requires actual nonempty bytes and positive custody retirement. The original
forward boundaries and activity set remain in force. The bounded replacement
protocol applies one run per demonstrably empty condition and preserves all
primary and calibration data. Its clarification is archived as
`protocol/profile-export-repair-clarification.md` with SHA256
`1d7a31d11856d9731cff19902e86a71494780fa4e4509515c05a340afbe2e3cf`.

All five replacement commands and independent Original, Replay, numeric and
saved-state readbacks passed. Each trace hash/size matches its actual profile
receipt. Every recorded kernel and copy used in the projection correlated to
the actual scoped worker's CUDA runtime or driver event. The parser assigns
forward kernels using the two actual worker annotation intervals and records
copies by physical direction.

| Profile in order | Scored rows in callbacks 1/2/3 | HtoD, µs | Tagged forward kernels, µs | All DtoH, µs | DtoH outside forward, µs |
| --- | --- | ---: | ---: | ---: | ---: |
| Model4 C16 legacy | 1 / 8 / 4 | 98.586 | 5385.206 | 68.259 | 38.211 |
| Model4 C16 Tokio W2 | 1 / 1 / 1 | 18.879 | 920.583 | 15.008 | 8.384 |
| Model4 C16 Fixed W4 | 1 / 1 / 1 | 26.754 | 1094.959 | 12.352 | 7.936 |
| Model3 C2 legacy | 1 / 1 / 1 | 15.424 | 983.003 | 15.137 | 9.921 |
| Model3 C2 Tokio W2 | 1 / 1 / 1 | 14.878 | 989.308 | 12.032 | 7.904 |

The rows show sums of actual device-event durations within the declared
post-forward-1 to post-forward-3 window. The input/forward portions cover
callbacks 2 and 3, while the output portions cover callbacks 1 and 2.
Outside-forward DtoH includes sampling and result return. The archived
projection records other copy directions, bytes, counts, individual forward
windows and every trace's identity. These sparse scopes differ by actual
boundary and row count. The totals cannot be applied as whole-run wall
time, GPU utilization or matched per-batch latency.

Actual profile evidence is under `populations/f6c8e45`. The combined data
and raw links are in `profile-c16-gpu-events.json` and
`profile-ordinary-gpu-events.json`. Their SHA256 hashes are
`6b364dd25ffc8ad85400a52a53ff6ae674e119262f13d57173184e997bbe63a9` and
`0af7614b32ce3839f0d436729341f6acb1c06c3e0fdaec55f6ebc22e3c251582`
respectively. The tightened legacy-host observer check independently
reproduces both legacy profile telemetry reports unchanged.

The primary checkout's `runs/rust-player-runtime-v1/final-evidence-index.json`
indexes the 41 accepted primary, calibration and replacement-profile
records with hash-bound configs, reports and original observer summaries.
It also indexes the five actual trace files, selection and independent
telemetry confirmations. Its SHA256 is
`70cd3df12fd0cd0145e60ae040f520b3772bef1ddfa21a6d679e310b87a7e930`.

The final integrated verifier passed 108 scoped CPU cases, Ruff, mypy and
diff checks. The integrated exporter passed its 15 scoped CPU cases and
Ruff, in addition to the actual CUDA export and replacement-profile evidence.
The installed pinned companion handshake and the new documentation links
pass using this worktree. Engineering and scope reviews are independent of
the observed operating values.


## Ordinary Model3 owned parity follow-up

The design from `feat/rust-model3-parity-validation` was integrated and
implemented in `f64fa3e3d7c7019feb6bd4bdfbafea6d1c067181`. Its branch
tip `55cd288` was already an ancestor. The design was the sole local
difference. [ADR0055](../adr/0055-owned-rust-model3-parity-validation.md)
records the approved scope and numerical-detection tradeoff.

The normal Source still binds original participant data, Replay, the current
checkpoint and persisted numeric bytes. Its local same-host update admits the
owned rollout before optimizer mutation and records `{"status":"not_run"}`
for the repeated frozen-policy comparison. Same-input tests check exact PPO
metrics apart from that report, named weights, Adam, shuffle and Torch RNG.
Explicit numerical mismatch rejection, pre-mutation capacity refusal, the
eight frozen kernel checks and failure custody passed. Both independent
reviewers accepted the exact source and tests.

| Fresh ordinary CUDA | Learner decisions / all-seat decisions | Update and save, s | Original command wall, s |
| --- | ---: | ---: | ---: |
| Before, `7424937` | 230 / 436 | 21.936 | 126.749 |
| After, `f64fa3e` | 230 / 436 | 14.822 | 115.214 |

One fresh command per source ran on RTX 5070 Ti with installed Native
`381ef85a`, Torch 2.14.1+cu132 and Simulator
`072dabbca1458c9526897781d2fd64edeeb4f9ca`. Both used ordinary Model3, Tokio
C2/W2/B2, 1 ms formation wait, seed 17 and 16 natural games. The resolved
configurations matched after substituting the isolated output paths. Both
commands recorded eight Adam steps, original successful exit and complete
independent Original, Replay, numeric and saved-pair readback.

The 32.4% shorter update interval and 9.1% shorter command wall are
single-pair observations. GPU readings immediately before the commands were
1% and 0% utilization with 13,501 and 13,105 MiB free respectively. The first
combined CPU check stopped after 34 completion marks and was interrupted with
exit 130. It overlapped part of the after observation, and its CPU activity
was not measured. The inferred cleanup node passed in isolation. The later
same-scope command passed all 100 selected CPU tests with six CUDA cases
deselected. The original interruption remains archived. Earlier repeated
deployment gates keep their original scope and decision.

The updated producer also paused after the first 16-game cohort and resumed
on the same parent and original deadlines. The restore receipt matches
weights, Adam, CPU and CUDA RNG and PCG64. The resumed cohort used 171 learner
decisions and brought the total to 16 Adam steps. Both commands passed
independent readback and each sampled training window contained at most one
Python process.

The primary checkout stores all case configs, observer records, archived
failure logs and checked reports in
`runs/rust-player-runtime-v1/components/model3-parity-validation-followup`.
Its `followup-evidence.json` is SHA256
`da0a709143bb36eed5a9196cab0583fa4284f22e4582adc036fcdae2ccfaa39b`.

## Verified input reuse in ordinary packing

`63618fb` removes the duplicate context check in packed Journal parsing.
`30a8ab1` reuses validated ordinary Source pairs during token conversion,
cached Pack data during one synchronous inspection, and the same Source owner
through numeric packing. The existing exact Journal reader type is exercised
by a real sealed-Pack regression. Independent correctness and conformance
reviews accepted the final snapshot. Forty focused CPU tests, Ruff on the four
changed Python files and mypy on the three production files passed. The tests
compare every emitted byte for identical fixture inputs and exercise refusal
before the completion manifest on original, deadline and capacity failures.

The public Rust ordinary CPU command used seed 17, Tokio, two concurrent games,
two player worker threads and maximum inference batch size two with a 1 ms
formation wait. Both fresh observations use Simulator
`072dabbca1458c9526897781d2fd64edeeb4f9ca` and the same base config after
substituting output paths. Each completed 16 natural games and one committed
update with 230 learner decisions, 436 all-seat decisions and eight Adam steps.
The original command exited successfully and its original artifacts, Replay,
numeric output and saved weights/optimizer state passed independent readback.
The updated training window had at most one sampled Python process.

| Ordinary CPU scope | Before, s | After, s |
| --- | ---: | ---: |
| Collection before Source projection | 7.792 | 7.826 |
| Application collection phase including Source projection and packing | 83.101 | 14.924 |
| Update | 7.309 | 7.221 |
| Original command wall | 109.106 | 36.129 |

The earlier observation used `4c723ae` with the measured match-execution
changes later committed as `dbd9d66`. Its run directory records their file
hashes. The updated original records clean revision `30a8ab1`. Of 4,696 arrays
read from the two numeric shards, 4,474 were byte-identical. The remaining 222
arrays encode frame payloads whose only differing field was `history_end`.
The manifest also carried each run's policy label and resulting digests. The
recorded PPO metrics matched exactly. Each timing is one CPU observation at
concurrency two. GPU and high-concurrency adoption still rely on their scoped
measurements.

The primary checkout records the commands, pre-fix regression and final test
logs under `runs/rust-player-runtime-v1/components/all-player-rollout/trusted-candidates-handoff`.
The archived `ordinary-observations.json` has SHA256
`3fc47772440261366be00cec4d5bfb8caab13ef7e034fac59c24b7e92cf94aac`.

### Repeated CPU measurements

Three fresh runs at each clean revision compared `dbd9d66` and `30a8ab1`
in one isolated checkout with the preceding seed 17, CPU, Tokio C2/W2/B2
configuration and pinned Simulator. The six configs matched after
substituting output paths. Runs used Python 3.14.10, NumPy 2.5.4 and Torch
2.14.1+cu132 with CUDA hidden and profiling off. Each completed 16 natural
games and one update with 230 learner decisions, 436 all-seat decisions
and eight Adam steps. PPO metrics matched across all six runs, and each
passed independent Original, Replay, numerical and saved-state readback.

| Trial | Revision | Collection, s | Between collection and update, s | Update, s | Command wall, s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `dbd9d66` | 7.810 | 78.124 | 7.181 | 109.181 |
| 2 | `30a8ab1` | 7.820 | 6.966 | 7.185 | 35.998 |
| 3 | `30a8ab1` | 7.566 | 7.095 | 7.180 | 36.190 |
| 4 | `dbd9d66` | 7.613 | 78.120 | 7.251 | 108.882 |
| 5 | `dbd9d66` | 8.050 | 77.943 | 7.153 | 109.117 |
| 6 | `30a8ab1` | 7.800 | 7.007 | 7.225 | 36.206 |
| **Median** | `dbd9d66` | 7.810 | 78.120 | 7.181 | 109.117 |
| **Median** | `30a8ab1` | 7.800 | 7.007 | 7.185 | 36.190 |

The interval between collection and update is the application collection
phase duration minus the collection stopwatch. It includes Source
verification, numeric packing and the transition to update. The median
of that interval decreased by 91.0%. The median command wall through
original exit decreased by 66.8%. These results cover ordinary CPU runs
at concurrency two.

The primary checkout stores the six configs, commands, observer records and
checked reports under the preceding artifact directory in
`repeated-cpu.bMz92QMU`. An initial setup attempt stopped before the
workload when the launcher rejected a symlink to the supervisor executable.
The measured runs used a regular copy of the same executable.

The SHA256 of `summary.json` is
`591d2b88eb08e856216f74d041049f2ff0ae193fcce6b88a6aa14d54e1b75bca`.
The SHA256 of `times.csv` is
`a7d82bf724c3779ba4a9ebc51dfa0ab01373b79a4695d0a88cae56281c6d7c4b`.

## Standalone HPO6 progress

The scoped regression suite covers launch-command quoting, post-ACK game
counts, malformed and stale samples, and the committed-generation fallback.
It exercises the original operation Store lock and the diagnostic namespace
outside the training parent. The match-pump tests require local sockets
and executor wakeup to work.

```sh
python -m pytest -q -n0 tests/test_hpo_progress.py tests/test_rust_player_operation.py tests/test_rust_player_matches.py
```

A fresh standalone HPO6 invocation with `--learning-seed 17` and
`--pause-after-update 1` ran eight games on the corrected Simulator
`3172da4`. It printed a copyable monitor command to stderr and
returned one final stdout JSON after cold finalization. An external
reader observed the collect count increase from 0/8 to 1/8, then
recorded 8/8 and the update, save and cleanup phases. The last sample
reported one committed update with 77 learner decisions and the
matching loss, entropy and approximate KL. Progress was under the
existing standalone operation Store. The training parent had no
telemetry directory.

The lock-free watcher also read run
`c254e6a6-fd44-41a0-9172-6a6f7ff01894` from the earlier preset.
It showed the failed status, six hash-checked committed updates and
their last metrics. Live phase and count data were unavailable in
that historical run. The smoke verifies the observation path at
eight games. A 1,000-update monitoring overhead measurement remains
outside this evidence.

The timing follow-up passed 108 focused cases with the same pytest
selection. A live run with eight games per update and
`--pause-after-update 4` showed an ETA after two committed intervals.
At update 2/4, the watcher displayed 2.48 games/s for the last
collection, 28.41 seconds elapsed and an estimated 13 seconds
remaining. The failed historical run showed 7.13 games/s and 372.90
seconds elapsed. Its ETA was absent.

An actual pseudo-terminal with `TERM=xterm-256color` redrew repeated
watch frames in the alternate screen and restored the original screen
on Ctrl+C. The same check kept line output for `linux` and `vt100`,
and `--once` emitted plain text.

The standalone action rate uses both seats' accepted actions from the
committed Source proof and the same collection duration as `games/s`.
`gf-bots hpo watch --once --path` on archived run
`cd05a376-93f2-44b4-8516-913c96f49518` displayed
`last collection=5.65 games/s, 172.13 actions/s` from update 16.
The JSON value matched the 7,795 stored accepted actions divided by its
45.286-second collection. The existing `uv run --no-sync` fresh command
passed configuration materialization and the current Native profile check.
The progress, operation, and match-pump pytest command above passed 143
cases on the host. Ruff, mypy, and the independent reviews passed.
