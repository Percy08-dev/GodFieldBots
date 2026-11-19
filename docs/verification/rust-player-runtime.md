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
