# Refresh heuristic verification

Status: implementation in progress. This record distinguishes design readiness,
component correctness, runtime compatibility and actual-game/learning checks.
No strength or formal-source/learning qualification is claimed.

## Frozen scope

Accepted [design](../design/heuristic-refresh.md), [ADR0039](../adr/0039-refresh-heuristic-and-mixed-opponents.md),
and [progress](../design/heuristic-refresh-progress.md). Simulator91d5926,
corrected-v2/public-ja-v3; old data/defaults/formal kernels preserved.

## Evidence so far

- Persisted design architecture recheck: READY, no blocking findings.
- Baseline: `.venv/bin/python -m pytest -q tests/test_card_instance_rules.py
  tests/test_card_instance_actions.py tests/test_card_instance_baseline_runtime.py
  tests/test_card_instance_cli.py` — PASS176,3.46s, before implementation.
- New model/experiment preset TOML syntax: PASS; application parsing NOT RUN.
- Task1:44 new tests PASS;224 related legacy cases PASS; scoped Ruff/mypy PASS.
  Root freshly ran `.venv/bin/python -m pytest -q tests/test_refresh_heuristic.py`:
 44passed in0.44s. Independent correctness/quality and architecture conformance READY.
  Reviewer also independently checked2047subset masks and transactional retry.
- Retention diagnostic:2097152-byte principal array,2295560-byte tracked transform
  peak,0.011340s local; O(K) facts preallocated/excluded. No timing guarantee.
- Application/runtime/native/PPO and real games: NOT RUN at Task1 milestone.
- Full18-possession/default-generator diagnostic PASS with23801presented
  candidates: generation0.983s, selection0.575s, process high-water77.78MiB.
  Generator count_time/search_time cutoffs and no18-slot sequence facts; this is
  not max32768/full-length evidence. Selected own smile-dew with threshold1e9.

## Task2 runtime/records acceptance

Final frozen combined CPU/private-native check:226passed/22CUDA deselected in
54.85s; Rust174passed/7explicit ignored; private-native15passed. Full-source
mypy198files and scoped Ruff/whitespace PASS. Exact commands and invalidated
attempts are in /tmp/gfb-refresh-task2-report.md. Initial sandbox preload errors
were traced to PermissionError on inherited AF_UNIX control socket, then rerun
with scoped escalation. A later source-identity error occurred while the writer
changed source during tests; the frozen-source rerun passed. Neither required
timeouts/harness/production-source weakening.

Independent review found CLI's generic u64 bound inconsistent with the new
positive-integer domain. Fix round1 changed only the refresh threshold check;
20regression cases/63covering checks PASS. Correctness and architecture re-review
READY. Root independently loaded2**64/2**130 and freshly ran:

```
GFB_HPO_NATIVE_CANDIDATE=/tmp/gfb-refresh-native-task2-final/gf_native.so .venv/bin/python -m pytest -q tests/test_refresh_runtime.py tests/test_refresh_native.py --tb=short
```

PASS66 in3.24s. Native remains a private release/MIMALLOC candidate, empty release
RUSTFLAGS; ELF a4e51e954dd6663b885b99e6386c9e1fa37fc3f44eaf733c7f671ccae3c40ac2,
native source18e1424c0f1fccb386ad695ed6ba25b0917d8fab36697d19a52f2b4a16dd7436.
Installed extension is unchanged. Mixed-parent Source/PPO remains Task3.

### Actual baseline a01

Frozen receipt and result:
`runs/assessments/heuristic-refresh-20261115/t2-baseline-a01/`.
311source/config hashes stayed unchanged across both invocations; the isolated
CLI large-integer correction happened afterwards and does not change policy math.
CLI `evaluate` used configs/experiments/heuristic-refresh-baseline.toml with
unique run name/output overrides; Random condition additionally changed only
opponent.model_id to random. Full corrected action settings, threshold12,
sampling17, game seed00000000-0000-0000-0000-000000000001, one paired seed.

| Condition | Natural | New policy W/D/L | Failures | Actual wall |
| --- | ---: | --- | ---: | ---: |
| existing heuristic | 2 | 0/0/2 | 0 | 23.916951s |
| Random | 2 | 2/0/0 | 0 | 4.758571s |

Both comparisons valid. Run roots are in run-results.json. Independent read-only
audit rehashed173inventory entries and report SHA values, parsed all8Journal6
files, checked4Replay closure pairs and reproduced all72refresh selections
exactly (53against old,19againstRandom). Each reselect used only its matching
frame/event prefix and supplied candidates, never future/other-player data.
Audit7.36s. Its initial diagnostic incorrectly compared each actor's decisions to
the whole game's aggregate; corrected summation passed, without source/data edits.
These four games establish functional integrity only, not strength/formal quality.

## Predeclared functional protocol

Use fresh directories below `runs/heuristic-refresh/`; never reuse prior parents
or alter the existing approved Simulator executable. Record actual Bots source
state, Simulator receipt, configs, sampling seeds, labels/options and run roots.
Keep failed attempts as failed evidence. Truncation/abort/rejection is not a draw.

1. Actual baseline: new-vs-old and new-vs-Random, one fixed game-seed pair per
   condition with swapped seats and sampling seed17. Default new threshold12;
   use the full corrected action preset for these games. Require four natural
   healthy games, actor-local journals and original Replay closure/inventory.
2. Runtime selectability: exercise the new policy in both configured roles and
   two distinct thresholds in component/actual worker checks; verify labels and
   options persist through journal/collection readers. A second threshold need
   not supply a separate strength experiment.
3. Mixed16 functional collection: ordinary and model4/HPO paths use explicit new
   descriptor, R4/H2/refresh2/snapshot8 and healthy natural outcomes. A deliberately
   bounded small action/model smoke configuration is permitted only if frozen
   before launch and identified separately from default-action baseline games.
4. PPO functional continuation: exercise actual update/save/pause/resume, Python
   and source-matched native structural readers as supported. Component fixture
   numeric tests do not substitute for actual-game collections. Ordinary and HPO
   have separate acceptance evidence; no shortened run is a formal parent.
5. HPO production checks must retain actual GPU job/admission/Source ownership.
   Do not bypass production gates merely to call a synthetic check a real run.
   Record unavailable hardware/admission checks as NOT RUN, never PASS.

Before final acceptance, run affected/whole project static/tests, historical
proofs and source/package correspondence, inspect bounded worst-case retention
memory/time, then obtain independent correctness and architecture conformance.

## Results

### Task3 scoped implementation acceptance

Ordinary/Packed mixed Config variants and HPO campaign/spec2 are explicit. All
original scheduler bodies, eight kernel ASTs, old defaults and final64/HPO
objective remain unchanged. Root-owned3presets load through application parsers.
Management envelope1 retains its distinct meaning. All mixed HPO execution needs
the new native marker; ordinary Python remains optional-native, parsing ungated.

Implementer covering375PASS/11deselected; final gate/profile186PASS/1deselected;
Rust175PASS/7ignored; fullmypy199/RuffPASS. Four existing Optuna warnings and32
existing Rust release warnings are recorded, not hidden. These include actual CPU
fixture update/save/resume, not actual Simulator/GPU learning acceptance.

Both independent reviews found one shared Source accounting defect: repeated
immutable config parsing per generation. Fix1 retains SAME hash-checked original
read once per parent. RealSealedScan regression RED3FAIL/2PASS ->GREEN5, covering
81PASS/2explicit opt-inSKIP/10deselected; stale/hash/after-read mutation still refuse.
Both scoped re-reviews READY, no remaining blocking findings.

Native directCargo candidate source d45b39c6d3827bb1d190269f5401780f52fa502796b5500423962f556b7eeb75,
ELF5c726c41afafccee433a4e2aed6deb893e94a50120f922af1c5d3b8306878edc.
Offline Maturin1.15.0 release/MIMALLOC/emptyRUSTFLAGS wheel builds ELF
f94f290ae013050a7b574b30383c3fde24e0667ef9045f8603619cdf4f1674ce;
all build_info keys are identical. Root tested this actual wheel ELF:

```
CUDA_VISIBLE_DEVICES='' GFB_HPO_NATIVE_CANDIDATE=/tmp/gfb-refresh-native-wheel-task3-site/gf_native/gf_native.cpython-314-x86_64-linux-gnu.so .venv/bin/python -m pytest -q tests/test_refresh_opponents.py tests/test_refresh_ppo.py tests/test_refresh_native.py --tb=short
```

PASS60 in64.30s. No installed replacement at this milestone. Python wheel/sdist
match all199source modules byte-for-byte; wheel674bdbdb9dcbe6963598a46a70c442bf28063180bed57b2bd4be7674c677e2cb,
sdist20f3251d12cf7b9d2d95a5a1a5d007710d97057a4551c7c16b01a7ba03b0efa5.
Package proofs /tmp/gfb-refresh-package-proof.json and native comparison
/tmp/gfb-refresh-native-build-comparison.json will be retained with final evidence.

### Task3 actual continuation conditions (declared before execution)

- Ordinary mixed CPU learner/Python Source: the checked-in mixed smoke preset,
  seed17,128candidates/length3/8192attempts/250ms search,600s original wall,
  max_updates2; pause after update1, resume unchanged to update2.
- Ordinary mixed CPU learner/native Source: same settings with explicit Rust
  preprocessing, separate fresh parent, same pause1→resume2 protocol.
- Model4/HPO actual GPU/native continuation: fresh assessment Config copied from
  the checked-in schema2 smoke, but decision_budget1,000,000 and max_updates2,
  fresh stage/paths and same600s original wall. One materialized run; fresh actual
  GPU admission for each invocation with SAME context/run/parent; pause1→resume2.
  Expected final status incomplete/update_cap, no objective. Never extend clocks.
- Separate actual CLI/Optuna campaign: unchanged checked-in schema2 D64 smoke,
  one trial/seed17 and old-heuristic evaluation pair, fresh campaign/DB roots.
  This has a completed objective only if production completion gates pass.
- Preserve actual GPU termination receipts returned by owned cleanup; no manual
  or fabricated termination, admission release or bypass of storage/Source gates.
- All actual source/config/native identities are captured before launch;
  no production-source edits during these invocations. Fixture CPU continuation
  is component evidence and does not substitute for these actual-game checks.

Implementation and functional rows will be appended with actual commands,
counts, identities and material limitations. No results are implied by this
predeclared protocol.
