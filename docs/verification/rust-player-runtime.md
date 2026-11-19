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
