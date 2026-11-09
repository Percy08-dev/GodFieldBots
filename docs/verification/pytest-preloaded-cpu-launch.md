# Pytest preloaded CPU launch — 2026-11-09

Status: accepted; both independent reviews READY and scoped/full verification
PASS. The matched PPO target improves; total default pytest increases with the
54added lifecycle regressions. No whole-suite speedup is claimed for this increment.
These integration timings precede the explicit-run selection follow-up below.

[ADR0030](../adr/0030-pytest-preloaded-cpu-launch.md) implements an explicitly
opted-in Linux CPU pytest launcher. A clean-exec `-E -S` template imports Torch
once, then directly forks fresh actors and records their exact wait statuses.
Model/options/frames stay in original direct pipes. Template never holds a model,
optimizer, game/private policy state or application test imports. Existing pool
owners still die before updates; template alone persists in the module scope.

## Boundaries, reference and failure ownership

The only application change is `WorkerReservation.creation_pending=False` and
its guarded transitions. Normal Popen launch/CUDA/native/default/source helpers
retain existing behavior. Private test files own the control protocol, actor
admission stream, server, parent service/mixin and lazy module fixture. Both normal
initial packet and EVERY pooled begin check fixed CPU factory/model/binding before
the original factory import. All original bootstrap/generator/selection semantics,
game/update budgets and assertions remain.

The module fixture binds an unstarted owner. Only eligible CPU collection starts
it before server/game/actor activity; pure guards and incompatible configs do not
warm it. Eligibility is exact actions2/known engineering contract, Python backend/
structured-v1 reader and supported CPU model/owner context. Valid beam/cap changes
are not artificially prohibited. Explicit incompatible preload fails early.

The shared MLP reference is now explicitly fresh-FORKED, not cold-exec. Each model/
episode state remains new. Original three-model cold Popen protocol and old pool
wire/reset tests retain independent cold coverage. New normal/fork/pool A-B-A and
checkpoint comparisons supplement it. Retained cold full parents are compared
below without relabeling or updating them under changed code.

Pending ownership exists before any external send; uncertain creation cannot use
the base `process=None` cancellation path. Actual PID/wait receipts alone permit
ordinary release after I/O retirement. Lost status poisons/quarantines the parent,
holds unresolved leases and blocks update/publication even when physical group
containment succeeds. Containment and confirmed actor reap are distinct outcomes.

Template wait ownership uses WNOWAIT until group signalling is finished. Known
reap is recorded even if real Popen.wait re-raises after reaping; no later PID/group
signal is allowed. One retained close preserves its original error/absolute
deadline under repeated cancellation. The auxiliary sampled RSS soft ceiling is
1GiB, independent of actor/CUDA/managed tensor reservations; it is not a hard
whole-process memory guarantee. Unsupported platforms, compiled/CUDA preloading,
arbitrary factories, descendant/group escape and unresponsive OS are not supported.

## Design and review evidence

The advisor resolved the consequential architecture choice in favor of the
clean direct-fork template, not the experimental multiprocessing facade. Six
HOST containment barriers PASS in4.574s under
`/tmp/godfieldbots-template-containment-20261109`: before fork, after fork/before
registration, post-registration/pre-ACK, post-wait/pre-status, lost ACK/repeated
cancel, and actual known exit23. Actor audit PIDs were read only AFTER cleanup,
never used to select signals or authorize release. Exact-status and contained-
status-lost/unconfirmed states stay separate.

Both independent final code reviews are READY. Concrete findings were corrected
with regressions before Root timing: fresh publication's closed receipt; actual
nonzero template exit after closeACK; all startup FD ownership/cached failure;
post-attestation and post-reply deadlines; no-child cleanup original error/owner/
idempotence; health-timeout poisoning; pre-Torch import-path purity; exact early
eligibility; and real wait-reap-then-error without signalling a recycled PID.

Historical checks/failures are preserved in
`/tmp/godfieldbots-cpu-preload-focused-20261109`: early stdin-wrapper lifetime
smoke failure (fixed by retaining/restoring its TextIOWrapper); first fault12P5F
in tool transcript session48805 (no disk stdout was created);10P2teardown-error
delta; later26P/29P/32P frozen logs; publication/FD3P; A-H43P48.06s; final real-
wait-exception6P5.17s. Later bytes are not falsely claimed tested by earlier runs.
Abort close(timeout=0) with actual known actor exit7 may release that physically
reaped lease but fail retirement bookkeeping and poison the service. No extra
normal deadline is granted to make this already-failed parent healthy.

## Same42case benchmark and module reuse

Root HOST command from repository root:

```sh
PYTHONPATH=/tmp/godfieldbots-preloaded-cpu-integration-20261109 UV_CACHE_DIR=/tmp/godfieldbots-uv-cache /usr/bin/time -v -o /tmp/godfieldbots-preloaded-cpu-integration-20261109/target-resource.txt timeout --signal=INT --kill-after=15s 900s uv run --no-sync pytest tests/test_structured_ppo.py -p pytest_pool_timing_probe --basetemp=/tmp/godfieldbots-preloaded-cpu-integration-20261109/pytest-temp --setup-show --durations=0 --durations-min=0 --junitxml=/tmp/godfieldbots-preloaded-cpu-integration-20261109/target-junit.xml > /tmp/godfieldbots-preloaded-cpu-integration-20261109/target.log 2>&1
```

PASS42; pytest160.73s, external161.16s versus195.84s:34.68s/17.7083% reduction
in single observations. Same42nodes/statuses, original workload/assertions and
observer/setup/duration reporting. Different temporary paths/revisions; no
variance claim. Setup19.015869s, calls141.288778s, teardown0.247406s and other
external0.607947s. First collector warmup and final module template close are
included, not a free cross-run cache.

`template-reuse-audit.json` confirms the13selected normal collections share ONE
template PID, and each returns closed/healthy with zero unretired tickets. The
template RSS sampled peak is528.520MiB, below its1GiBsoft allowance. Launch counters
are cumulative module counters, not per-batch values to sum. This excludes learner
and independent normal-cold imports; it is not a whole-memory or exact whole-suite
Torch-import-count claim.

## Numerical, wire and normal-path checks

All2112actual G0/G2weight/optimizer/RNG arrays across six normal before/candidate
parents and12artifact directories are exact in dtype/shape/value for MLP/GRU/
Transformer. Before sources are the actual normal-cold target in
`/tmp/godfieldbots-lazy-cpu-runtime-20261109/pytest-temp`. They are never rewritten,
requalified or resumed under changed code. See `array-equivalence.json`.

All768actual schema5decision records have nonempty frame/table/selection/search/
command. Actual candidate files/hashes, search RNG, numeric choice/logprob/value,
frame/action content and own policy/checkpoint identity compare exactly. All18
inference manifests and768policy identities are unchanged (mathematical producer
source did not change). The bounded comparison-only projection removes opaque
episode/command IDs and linked hashes, compares cursor viewer/position suffixes,
maps visible own-hand IDs plus matching hints/actions only to original positions,
and omits3search timing fields. It never compares missing legacy inputs or
relabels an original identity. See `compare_wire.py`/`wire-equivalence.json`.

Related HOST command:

```sh
UV_CACHE_DIR=/tmp/godfieldbots-uv-cache timeout --signal=INT --kill-after=15s 480s uv run --no-sync pytest -q tests/test_cpu_preload.py tests/test_cpu_preload_review.py tests/test_pooled_structured_worker.py tests/test_parallel_structured_ready.py tests/test_structured_protocol.py tests/test_structured_environment.py tests/test_structured_source.py tests/test_structured_ppo_source.py tests/test_structured_ppo_cuda.py --junitxml=/tmp/godfieldbots-preloaded-cpu-integration-20261109/related-junit.xml
```

PASS333/SKIP6in146.06s. Explicit existing actualCUDA protocol/numerical opt-ins
`GFB_RUN_STRUCTURED_PROTOCOL_CUDA=1 GFB_RUN_STRUCTURED_NUMERIC_CUDA=1` with
`CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0` PASS6in43.86s on HOST.
No preload mode is used for CUDA. No Source qualification or model quality follows.

## Whole invocation: success with an explicit test-cost tradeoff

Root frozen-source HOST command:

```sh
UV_CACHE_DIR=/tmp/godfieldbots-uv-cache /usr/bin/time -v -o /tmp/godfieldbots-preloaded-cpu-integration-20261109/full-resource.txt timeout --signal=INT --kill-after=15s 2400s uv run --no-sync pytest --basetemp=/tmp/godfieldbots-preloaded-cpu-integration-20261109/full-pytest-temp --setup-show --durations=0 --durations-min=0 --junitxml=/tmp/godfieldbots-preloaded-cpu-integration-20261109/full-junit.xml > /tmp/godfieldbots-preloaded-cpu-integration-20261109/full.log 2>&1
```

PASS **4112passed/247skipped/4warnings/zero failures**,4359cases, pytest1204.42s,
external1205.43s (20:05.43), exit0. All previous4305pass/skip statuses remain;
report joining normalizes only the generated HTTP UUIDv1parameter display.

The full invocation INCREASES27.47s from1177.96s, because54new tests add59.768
JUnit case-seconds. Common4305case sums fall1173.980→1141.793s (32.187s saved).
These sums exclude runner overhead and are single full invocations, not the
matched standalone benchmark. Do not market this increment as a faster entire
default pytest suite. Current whole-file PPO sum191.108→159.742s; the new preload
modules contribute44.338s plus15.430s. Critical lifecycle coverage is intentionally
retained rather than hidden behind new opt-ins or shortened budgets.

PASS scoped `ruff check src tests native`, `mypy src`157files and diff checks.
PASS application wheel157source modules byte-exact/no tests packaged. Installed
mandatory-only Python3.14without Torch/LightGBM/PyArrow/native/pytest PASS normal
structured Random worker init/actual reap/SAME empty ledger and CLI help outside
the repository. Application imports never depend on the test launcher.

All evidence is under `/tmp/godfieldbots-preloaded-cpu-integration-20261109`:
baseline snapshots, `measurement-metadata.json`, `full-freeze.json`, raw logs/
XML/resources, package correspondence and proof files. All157application and
9scoped test hashes match measured bytes through full acceptance. Independent
audits verify those hashes/counts/identities/scopes without rerunning tests.
Root accepts after both final reviews and all checks. Unsupported platforms,
CPU-only Torch distribution, arbitrary compiler/threaded consumers and formal
Source/learning acceptance remain unclaimed. Existing numerical and compatibility
versions and active-resume code-identity refusal policy are unchanged.

## Follow-up: expensive checks require explicit execution (2026-11-09)

After the owner's request, the shared `requires_preload` condition additionally
requires the exact environment value `GFB_RUN_CPU_PRELOAD_SAFETY=1`. This moves
38 real-process/heavy checks to explicit execution while keeping 16 inexpensive
checks in default pytest. The CPU PPO launch optimization remains enabled.
This is a local test-selection change, with no runtime/concurrency/dependency
boundary change; no new architecture gate was required.

PASS, with the environment variable unset:

```sh
timeout 60s env -u GFB_RUN_CPU_PRELOAD_SAFETY UV_CACHE_DIR=/tmp/godfieldbots-uv-cache uv run --no-sync pytest -q tests/test_cpu_preload.py tests/test_cpu_preload_review.py
```

Result: **16 passed / 38 skipped in 0.56s**. The same command with value `0`
also gives **16 passed / 38 skipped in 0.57s**. Enabled collect-only retains all
54 cases. PASS actual HOST execution:

```sh
GFB_RUN_CPU_PRELOAD_SAFETY=1 UV_CACHE_DIR=/tmp/godfieldbots-uv-cache uv run --no-sync pytest -q -x tests/test_cpu_preload.py tests/test_cpu_preload_review.py
```

Result: **54 passed in 61.48s**. The retained command additionally uses an outer
180s timeout, `/usr/bin/time` and JUnit/log output under
`/tmp/godfieldbots-cpu-preload-optin-20261109/host-enabled-*`.
An initial restricted-sandbox attempt hit the existing inherited-socket
`PermissionError` in `cpu_preload_server.py:37` and was interrupted with exit130;
its separate `enabled.log` is not counted as passing verification.

PASS scoped Ruff and diff checks. The AST/source audit in `selection-audit.json`
confirms both modules' code is unchanged apart from the shared skip condition:
all test bodies, assertions and budgets are retained. The historical 16-case
aggregate is 0.072s and the 38-case aggregate is 59.696s; those are earlier JUnit
sums, not a new whole-suite timing. Whole pytest, application/wheel and CUDA
checks were NOT RERUN for this selection-only follow-up. No automatic CI or
change-triggered execution is introduced. Use the README's explicit command
when launch/protocol/cleanup, initialization/RNG or Torch changes.

## Current default full run and per-file timings (2026-11-09)

After the selection follow-up at `b2a880c4ae4920af0b1519a577e7dffed52ac043`,
one fresh HOST default invocation PASS: **4,074 passed / 285 skipped / 4 warnings /
zero failures**, 4,359 collected cases, exit0. Pytest reports **1,150.17s**;
external wall is **1,151.24s (19:11.24)**. The 38 heavy safety checks are skipped,
while CPU PPO preloaded launch remains active.

```sh
env -u GFB_RUN_CPU_PRELOAD_SAFETY UV_CACHE_DIR=/tmp/godfieldbots-uv-cache /usr/bin/time -v -o /tmp/godfieldbots-current-file-timings-20261109/resource.txt timeout --signal=INT --kill-after=15s 2400s uv run --no-sync pytest --basetemp=/tmp/godfieldbots-current-file-timings-20261109/pytest-temp --setup-show --durations=0 --durations-min=0 --junitxml=/tmp/godfieldbots-current-file-timings-20261109/junit.xml > /tmp/godfieldbots-current-file-timings-20261109/pytest.log 2>&1
```

[All 167 files, sorted by current duration](pytest-file-timings-20261109.csv)
include case counts/statuses and the preceding full-run comparison. Times are
JUnit setup/call/teardown sums. Shared fixture setup belongs to the requesting
file; collection/import/runner overhead remains outside these per-file sums.

The preceding full run was 1,205.43s, so this observed wall is 54.19s lower.
The previous population executed the 38 now-skipped cases; the two single runs
do not isolate the speed of unchanged tests or establish timing variance.
Evidence, metadata with 352 source/test/config hashes, JUnit, logs/resources and
per-file JSON/CSV are in `/tmp/godfieldbots-current-file-timings-20261109`.
All 352 frozen hashes match after execution. The file-time sum is 1,147.310s;
the longest files are Tree Dataset 199.312s, structured PPO 160.781s and native
structured session 118.709s. All prior case statuses remain except the intended
38 pass-to-skip transitions; comparison normalizes only generated HTTP UUIDv1
display names. No test or application implementation changed during measurement.

## Tree Dataset preparation extension — ADR0031 (2026-11-09)

Status: accepted; implementation and both independent final reviews READY;
scoped and final default full checks PASS. Design and ownership are in
[ADR0031](../adr/0031-tree-pytest-preloaded-cpu-launch.md).

Only the private CLI finalizer admission and test composition change. One explicit
lazy module fixture binds the existing Torch-only template across all nine Tree
collections. All 576 episode actors remain fresh, with new model/episode/RNG state;
the three ordinary cold-exec validation preflight actors remain unchanged. There
is no participant pool, readiness-order change, format/dependency change, simulator
edit or new public launch default. Exact concrete validation context, projected
LoadedConfig identity, actual plan/SAME ledger and CPU/Python engineering scope
are checked before warmup; actual retirement, template health and the live
original context are rechecked before report/manifest publication.

The original three seeds, two updates per parent, 96 training + 192 validation
games, zero-endpoint cases and Dataset build/inspect assertions are retained.
New 37 lightweight dispatch/ordering/failure checks run in default pytest; the
existing 38 expensive safety checks retain their explicit environment opt-in.
Both independent final reviews are READY without findings. The dispatch doubles
do not establish Source validity; the actual full Tree target below does.

### Matched target and actual originals

Root runs the same disposable observer as the retained cold diagnostic, changing
only its output directory. The target is the complete original Tree test:

```sh
env -u GFB_RUN_CPU_PRELOAD_SAFETY -u GFB_TREE_DATASET_EXISTING_SOURCE17 -u GFB_TREE_DATASET_EXISTING_SOURCE29 UV_CACHE_DIR=/tmp/godfieldbots-uv-cache /usr/bin/time -v -o /tmp/godfieldbots-tree-preloaded-cpu-20261109/target-resource.txt timeout --signal=INT --kill-after=15s 480s uv run --no-sync python /tmp/godfieldbots-tree-preloaded-cpu-20261109/profile_tree.py > /tmp/godfieldbots-tree-preloaded-cpu-20261109/target.log 2>&1
```

PASS one test. Matching pytest time **200.56 -> 80.97s** (119.59s / 59.628% saved);
external **201.51 -> 81.48s** (120.03s / 59.565% saved). Setup is73.26s, call7.45s,
teardown0.22s in the rounded duration report; case time is80.926s. First template
warmup and actual final close are included. The earlier199.312s normal full-run
case is a separate observation, not the matched diagnostic baseline.

Collection union drops167.616 ->47.030s, episode-ready union148.338 ->35.966s.
Reported MLP model-load median drops782.421 ->47.373ms. That handshake timer
includes factory/reset/etc, not pure Torch import. Three cold preflight samples
remain included in the339MLP samples. In the candidate observer, base `_spawn`
counts only those three ordinary actors because the fork mixin overrides that
hook; it is not the total process count. Nine manifest receipts share one actual
template PID, healthy/closed with zero unretired tickets, cumulative launches
32/64/192/224/256/384/416/448/576. Template sampled peak RSS is540,884KiB
(528.207MiB), below its separate1GiBsoft allowance, not a whole-system quota.

Read-only independent comparison against retained actual cold parents confirms
all12G0/G2weight/resume NPZ files and876arrays exact by keys/dtype/shape/value,
including optimizer and RNG state. Both complete nine-work Source populations
validate through NativeTreeDataset. All512actual selected Tree features and
natural labels match in order and against independent Journal5/tokenization/
PreparedTreeFrame references. Label counts are -1:192,0:192,+1:128.

All1,024actual decision records also match in real candidate bytes, search RNG,
numeric selections, frames and actions. Each side's policy/checkpoint binding
is independently checked; all1,024policy identities and all9inference manifests
are unchanged. Comparison-only projections remove opaque IDs/linked hashes,
normalize viewer/cursor suffixes and visible own-hand references, and omit the
three search timing fields. No original is rewritten, resealed or relabeled.

### Related and package verification

HOST explicit safety + affected protocol/pool/environment/source + candidate
frozen Tree checks PASS **378 passed / 6 CUDA opt-in skips in163.51s**:

```sh
GFB_RUN_CPU_PRELOAD_SAFETY=1 GFB_TREE_DATASET_FROZEN_FIXTURE=/tmp/godfieldbots-tree-preloaded-cpu-20261109/pytest-temp/tree-dataset-originals0 UV_CACHE_DIR=/tmp/godfieldbots-uv-cache timeout --signal=INT --kill-after=15s 480s uv run --no-sync pytest -q tests/test_tree_cpu_preload.py tests/test_cpu_preload.py tests/test_cpu_preload_review.py tests/test_pooled_structured_worker.py tests/test_parallel_structured_ready.py tests/test_structured_protocol.py tests/test_structured_environment.py tests/test_structured_source.py tests/test_structured_ppo_source.py tests/test_structured_ppo_cuda.py tests/test_structured_tree_dataset_frozen.py
```

PASS `ruff check src tests native`, `mypy src`157files, exact157module wheel source
correspondence/no tests packaged. Mandatory-only installed Python3.14, outside
checkout and without Torch/LightGBM/PyArrow/native/pytest, PASS normal Random
structured actor initialization, actual reap/SAME empty ledger and CLI help.
Actual CUDA opt-ins and Rust tests are NOT RERUN for this CPU-only private
composition change; transport/learning/native code and default routes are fixed.

Evidence: `/tmp/godfieldbots-tree-preloaded-cpu-20261109`, including baseline/
candidate hash metadata, matching observer, target logs/resources/JUnit,
baseline/candidate-equivalence and wire-equivalence reports, related results,
package/baseline reports and full-run freeze. All353Python/config measured hashes
match before the full invocation. Sources stay engineering-only; no formal
qualification or training-quality claim follows.

### Final default full invocation

One final frozen-source HOST run PASS **4,111 passed / 285 skipped / 4 warnings /
zero failures**, 4,396 cases, exit0. Pytest time1,025.70s; external wall
**1,026.74s (17:06.74)**. The same38expensive safety checks remain skipped.

```sh
env -u GFB_RUN_CPU_PRELOAD_SAFETY -u GFB_TREE_DATASET_FROZEN_FIXTURE -u GFB_TREE_DATASET_EXISTING_SOURCE17 -u GFB_TREE_DATASET_EXISTING_SOURCE29 UV_CACHE_DIR=/tmp/godfieldbots-uv-cache /usr/bin/time -v -o /tmp/godfieldbots-tree-preloaded-cpu-20261109/full-resource.txt timeout --signal=INT --kill-after=15s 2400s uv run --no-sync pytest --basetemp=/tmp/godfieldbots-tree-preloaded-cpu-20261109/full-pytest-temp --setup-show --durations=0 --durations-min=0 --junitxml=/tmp/godfieldbots-tree-preloaded-cpu-20261109/full-junit.xml > /tmp/godfieldbots-tree-preloaded-cpu-20261109/full.log2 2>&1
```

All353frozen code/config hashes match after execution. All4,359previous testcase
statuses remain, comparing only a generated UUIDv1 HTTP parameter display with
normalization. Exactly37new lightweight passing nodes are added; no prior
scenario is removed or reclassified. The current full-run Tree file JUnit sum
is77.870s versus199.312s in the prior full run. Whole external wall is124.50s
lower than1,151.24s previously, with different total populations and one sample
each; no matched whole-run or variance claim follows.

[All168current file times and preceding full-run differences](pytest-file-timings-tree-preload-20261109.csv)
sum to1,022.676JUnit case-seconds, with shared setup attributed to requesting
tests. They do not equal whole wall or independent per-file invocations. The
new37case module has no prior comparison. Full raw log/JUnit/resources, CSV and
`full-acceptance.json` are retained under the evidence directory above. Root
accepts after both reviews and all required checks; no code changed during these
target/related/package/full acceptances.

## Native actors and numerical children — ADR0032/0033 (2026-11-09)

Status: accepted; both implementation reviews READY, final targeted/equivalence/
related/CUDA/package and default full checks PASS.
[ADR0032](../adr/0032-native-pytest-preloaded-actors.md) owns the native actor
extension; [ADR0033](../adr/0033-pytest-preloaded-numerical-children.md) owns the
separate numerical template and protected lifecycle/visibility extractions.

### Scope, defaults and lifecycle evidence

Native preparation uses one actor template across the original five-case module,
retaining128games,256fresh actors (192MLP/32random/32heuristic),512decisions and
all six ordinary numerical invocations. Exact native operations/learner/ledger,
complete loaded config and sealed input origins are checked inside the original
budgeted Source scope before warmup and publication. Source projection, before-
Adam mutation detection and both intentional failed/quarantined runs remain.

Numerical preload is explicitly activated only for19CPU Game Source starts and
six starts in the three real CPU wire tests. Canonical exact GameLearningProcess
objects, original main/serve/math/Source bodies and restart/restore assertions
remain. Defaults retain exact cold argv/environment/stream limits and strict
Torch-before-visibility rejection. Only the forked test bootstrap installs CPU
admission after original model/execution/identity validation. Unsupported runtime,
CUDA, legacy and other lifecycle/reference cases retain ordinary cold launch.

One separate numerical template admits one pending/live/unretired ticket. It
uses `pytest-cpu-numeric-preload-v1`, three async pipes and pinned CPython3.14.10
selector/StreamWriter support. Actor control stays `pytest-cpu-preload-v1`, two
pipes/capacity4, with separate inventory/publication health. Canonical adoption
retains actual process before at-most-once notification/drainer creation;
settlement distinguishes authoritative no-child from unknown creation. Actual
waitpid status is cached before terminal failure; unknown terminal returncode
raises to block canonical requests and Source use. Retained first-deadline
cleanup owns transports, drainer, tickets, reader detachment and containment.

Advisor evidence is the isolated disposable copy in
`/tmp/godfieldbots-numeric-preload-feasibility-20261109`. Five cold/fork numerical
and Source cases PASS20.60/13.92s,4,230arrays across18artifacts exact, fault and
backpressure/log/FD witnesses PASS. Initial copy-isolation and harness corrections
are preserved and excluded from passing evidence. This is feasibility only; its
temporary entry path and permissive origin guard are not promoted into defaults.

Independent production review found four actual adapter issues: post-memory
deadline checking, admission after auxiliary shutdown begins, deferred terminal
cleanup after retirement errors, and async health timeout poisoning. All were
fixed with focused8PASS, related60PASS and a combined-error1PASS; both final
reviews independently rechecked and report READY. Primary errors/deadlines and
unknown leases remain retained; fixture teardown rejoins the same cleanup.
Existing38heavy actor checks remain gated; new numeric checks are17light default
and42heavy explicit opt-in. New native guards add49light default cases.

### Final matched targeted measurement

The final target uses the original node-tagged observer, changing only its output
directory. It runs all five native cases,43Game Source cases and three real CPU
wire cases, with original workloads/assertions;10CUDA parameters remain skipped.

```sh
env -u GFB_RUN_CPU_PRELOAD_SAFETY -u GFB_RUN_CARD_INSTANCE_SOURCE_CUDA UV_CACHE_DIR=/tmp/godfieldbots-uv-cache /usr/bin/time -v -o /tmp/godfieldbots-native-numeric-preload-20261109/final-target/resource.txt timeout --signal=INT --kill-after=15s 480s uv run --no-sync python /tmp/godfieldbots-native-numeric-preload-20261109/final-target/profile_other.py > /tmp/godfieldbots-native-numeric-preload-20261109/final-target/profile.log 2>&1
```

PASS **41passed/10skipped/zero failures**, pytest **194.02 ->107.57s**, external
**194.87 ->108.43s**, including first template warmup and final real retirement.
These are single observations. The earlier107.29s first candidate predates the
four final review fixes and remains historical, not final acceptance evidence.

| Selected file population | Cold diagnostic case-seconds | Final case-seconds |
| --- | ---: | ---: |
| Five native structured session cases | 118.743 | 55.604 |
| Game Source cases (33pass/10CUDA skips) | 61.629 | 43.216 |
| Three real CPU numerical-wire cases | 12.885 | 7.999 |

Native collection union falls91.829 ->26.180s and episode-ready union82.984 ->
15.440s. Its six numerical initializations remain ordinary,8.239 ->8.281s.
Game Source's19initializations fall25.423 ->11.924s; six CPU-wire initializations
fall8.246 ->4.016s. These are host intervals containing import/model construction/
restore/IPC/artifact work, not pure Torch import timings. Label intervals nest
and overlap; do not sum them. The observer's raw numeric `outcome='measured'`
error-counter defect is excluded by the corrected per-file reader, as in the
prior diagnostic. JUnit, actual replies and interval records are authoritative.

### Actual saved data, Source and numerical parity

Read-only native comparison validates three complete parents and preserves both
failed/quarantined pre-Adam populations as failures. All876G0/G2checkpoint arrays
and4,416valid rollout arrays match. All512actual journal decisions match projected
frame, candidate bytes, search RNG/results, numeric selection and action content;
each side's256policy headers and192MLP checkpoint hashes are checked independently.
Eight actor receipts share one actual template PID,256cumulative launches,
healthy/closed with zero tickets. Inference producer digests and derived neural
policy labels legitimately change; actual old/new identities are retained, not
rewritten or declared equal.

Read-only numerical comparison covers all22selected CPU case directories:
10projection/precommit,9live-owner guards and three model wire cases. All94NPZ
pairs and10,487arrays match by filenames/keys/dtype/shape/value. Resume counters,
named optimizer/RNG/presence metadata and inference fields other than producer
match. Both positive Game Source parents pass normal historical validation with
one update/two chain records. Intentional before/after-policy hash mutations are
expected rejection diagnostics; they are not eligible Source data. No original
is repaired/resealed, and no formal Source/quality conclusion follows.

### Related, CUDA and package checks

HOST explicit safety/affected actor-pool/numeric/cold-PPO/cleanup regressions PASS
**300cases in214.41s**, including the original and new opt-in heavy cases:

```sh
GFB_RUN_CPU_PRELOAD_SAFETY=1 UV_CACHE_DIR=/tmp/godfieldbots-uv-cache timeout --signal=INT --kill-after=15s 480s uv run --no-sync pytest -q tests/test_native_cpu_preload.py tests/test_tree_cpu_preload.py tests/test_numeric_cpu_preload.py tests/test_cpu_preload.py tests/test_cpu_preload_review.py tests/test_pooled_structured_worker.py tests/test_structured_numeric_worker.py tests/test_ppo_lineage.py tests/test_worker_cleanup.py
```

Actual HOST CUDA cold smokes PASS **7cases in50.27s**: three participant models,
three numerical two-update/restore models and the positive Game Source update.
Flags are `GF_RUN_CUDA=1`, `GFB_RUN_STRUCTURED_PROTOCOL_CUDA=1`,
`GFB_RUN_STRUCTURED_NUMERIC_CUDA=1`, `CUDA_VISIBLE_DEVICES=0` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. No GPU preloading or fallback is exercised.

PASS scoped Ruff, mypy157, exact157module wheel/no tests packaged and installed
mandatory-only normal Random actor start/actual reap/SAME empty ledger/CLI help
outside checkout without optional Torch/LightGBM/PyArrow/native/pytest. Numerical
and Source algorithms are original; Rust engine/Source code is not modified.

Evidence root: `/tmp/godfieldbots-native-numeric-preload-20261109`. It retains
baseline/candidate/final358file hashes, native-only intermediate proof, isolated
proof pointers, all failed/fixed deltas, final-target raw event/sample/JUnit/
resource/parity reports, related/CUDA/static/package/baseline logs and final full
freeze. Main-source hashes and identity changes are explicit; no code changes
are allowed during the final default invocation. The admitted CPU-only Torch
distribution was not separately measured; unsupported Python patches/asyncio
backends retain ordinary cold fallback.

### Final default full invocation and file times

HOST final frozen-source command:

```sh
env -u GFB_RUN_CPU_PRELOAD_SAFETY -u GFB_TREE_DATASET_FROZEN_FIXTURE -u GFB_TREE_DATASET_EXISTING_SOURCE17 -u GFB_TREE_DATASET_EXISTING_SOURCE29 -u GF_RUN_CUDA -u GFB_RUN_STRUCTURED_PROTOCOL_CUDA -u GFB_RUN_STRUCTURED_NUMERIC_CUDA UV_CACHE_DIR=/tmp/godfieldbots-uv-cache /usr/bin/time -v -o /tmp/godfieldbots-native-numeric-preload-20261109/full-resource.txt timeout --signal=INT --kill-after=15s 2400s uv run --no-sync pytest --basetemp=/tmp/godfieldbots-native-numeric-preload-20261109/full-pytest-temp --setup-show --durations=0 --durations-min=0 --junitxml=/tmp/godfieldbots-native-numeric-preload-20261109/full-junit.xml > /tmp/godfieldbots-native-numeric-preload-20261109/full.log 2>&1
```

PASS **4,177 passed / 327 skipped / 4 warnings / zero failures**,4,504cases,
exit0. Pytest941.86s, external **942.93s (15:42.93)**. All358frozen hashes match.
All4,396prior nodes/statuses remain after normalizing only the known generated
UUIDv1 HTTP parameter display. Added108nodes are49native light passes plus
17numeric light passes/42new heavy skips; the original38heavy skips remain.

| File | Prior full-run case-seconds | Current case-seconds |
| --- | ---: | ---: |
| Native structured session | 118.494 | 55.387 |
| Game Source | 60.445 | 43.102 |
| Structured numeric worker (whole file) | 16.339 | 12.187 |
| Structured PPO | 159.628 | 159.607 |
| Tree Dataset | 77.870 | 77.709 |

External full time is1,026.74 ->942.93s (83.81s lower), with108addedcases and
single observations. This is an observed changed-population comparison, not a
matched whole-suite speed or variance claim. The matched51-case target above
provides the controlled scope comparison.

[All170current file times and previous differences](pytest-file-timings-native-numeric-preload-20261109.csv)
sum938.787JUnit case-seconds; shared fixture setup is attributed to requesting
cases and collection/runner overhead remains separate. They are not independent
file invocations. `full-acceptance.json`, full log/XML/resources, and all code
freeze/equivalence/failed-iteration reports are retained under the evidence root.
Root accepts both bounded private extensions after all reviews and checks.
No source or test code changed during the final acceptance runs.
