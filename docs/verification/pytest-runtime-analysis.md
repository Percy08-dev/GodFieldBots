# Pytest runtime step analysis — 2026-11-08

This is an observational analysis of the default test suite at Bots revision
`8eb6a7ff20804da665a23a0827f674bb7d20a2b2`. No application, test, dependency,
configuration, Simulator or training recipe changes are adopted.

## Measurements and evidence

The original single full HOST run PASS: **3,970 passed, 247 skipped, 4 warnings**,
exit 0. Its external wall time is **1,738.71 seconds (28:58.71)**; pytest reports
1,737.59 seconds. Launch and exit work outside pytest's timer therefore account
for about 1.12 seconds in this invocation.

Environment: WSL2 Linux, Python 3.14.10 with GIL, uv 0.13.7, pytest 9.1.1,
NumPy 2.5.4, Torch 2.14.1+cu132, LightGBM 4.7.0, PyArrow 26.0.0 and the existing
native extension. No CUDA or other opt-in test flags were enabled. Existing
untracked user files were preserved. Historical whole-suite counts with different
opt-in conditions are not the population measured here.

Exact full-run command, executed from the repository root:

```sh
/usr/bin/time -v -o /tmp/godfieldbots-pytest-timing-20261108T1404/resource.txt uv run pytest --durations=0 --durations-min=0 --junitxml=/tmp/godfieldbots-pytest-timing-20261108T1404/junit.xml > /tmp/godfieldbots-pytest-timing-20261108T1404/pytest.log 2>&1
```

Local evidence directory: `/tmp/godfieldbots-pytest-timing-20261108T1404`.
It contains `metadata.json`, `resource.txt`, `pytest.log`, `junit.xml`,
`files.csv`, `tests.csv`, `phases.csv`, `summary.json` and `step-analysis.json`.
The original test outputs used below remain under
`/tmp/pytest-of-percy/pytest-139`. These are temporary local evidence, not committed
fixtures; their availability depends on temporary-directory retention.

## Where the original invocation spends time

File totals below come from JUnit and include setup, call and teardown. The
phase columns come from pytest's per-phase output rounded to 0.01 seconds;
small differences between their sum and the JUnit total are rounding effects.

| File under `tests/` | Cases | Total seconds | Setup seconds | Call seconds |
| --- | ---: | ---: | ---: | ---: |
| `test_structured_ppo.py` | 35 | 533.535 | 63.66 | 469.77 |
| `test_structured_tree_dataset.py` | 1 | 288.958 | 281.50 | 7.46 |
| `test_native_structured_session.py` | 5 | 171.609 | 0.00 | 171.60 |
| `test_native_session.py` | 11 | 122.395 | 102.20 | 20.22 |
| `test_card_instance_source.py` | 43 | 61.349 | 0.00 | 61.27 |
| `test_fixed_ppo_data.py` | 6 | 56.715 | 51.55 | 5.16 |

The first four files consume **1,116.497 seconds, 64.21% of external wall time**.
Across the whole suite, the rounded setup rows sum to 541.79 seconds and the
rounded call rows to 1,179.28 seconds. Teardown rows round to zero; cleanup
performed inside a test or fixture is charged to call or setup instead.

### PPO continuous/pause-resume cases repeatedly collect with fresh actors

The three CPU cases in
[`run_continuous_and_resume`](../../tests/test_structured_ppo.py) each execute a
two-update continuous run, a one-update pause and a resume to update two.
Each case asserts 128 newly created participant workers across 64 scripted
games, plus three learner invocations. HTTP is mocked; the participant and
numerical subprocesses, journals, artifacts and Source checks are real.

Original measurements, combining both parents' stored records for each case:

| Model | Pytest case seconds | Collection seconds | Update request seconds | Numerical update seconds |
| --- | ---: | ---: | ---: | ---: |
| MLP | 91.054 | 72.724 | 4.795 | 4.403 |
| GRU | 90.175 | 72.743 | 5.832 | 5.417 |
| Transformer | 91.816 | 73.106 | 6.465 | 5.958 |

The last column is a subset of update request time, not an additional cost.
`run.json` supplies cumulative `collection_seconds` and `update_seconds`;
`updates/*.json` supplies `metrics.update_seconds`. Remaining time includes
learner initialization/restore, projection, ownership/source checks, artifact
publication/retention, process cleanup and test assertions. The measurements do
not identify that entire remainder as any one of these operations.

The native continuous/resume case is another 87.570 seconds, with 72.698 seconds
of stored collection time and 4.790 seconds of update requests. This repeats the
128-worker lifecycle through the native Source path. Other cases in these two
files also perform fresh collection or build shared completed/paused parents;
the three model cases alone are not the entire 533.535-second PPO module.

Relevant code: `tests/test_structured_ppo.py:315`, `:386`;
`tests/test_native_structured_session.py:128`;
`tests/structured_ppo_support.py:38`; `src/gf_training/ppo_run.py:480` and `:550`.

### Tree's long step is original-source preparation

[`original_tree_sources`](../../tests/test_structured_tree_dataset.py) is already
module-scoped. It prepares three parents at seeds 17/29/43, each with two updates
and 32 training games, then collects 64 final-policy validation games per parent.
Its test asserts 64 training actor processes and 128 validation actor processes
per seed: **288 scripted games and 576 participant processes** in this fixture.

The original fixture's 281.50-second setup is almost entirely explained by
saved timing records:

| Step | Seconds |
| --- | ---: |
| Three parent training invocations | 127.618 |
| Three final-policy validation collection reports | 153.346 |
| Approximate unassigned setup remainder | 0.536 |
| Dataset build and subsequent inspect in the test call | 7.46 |

Parent training already includes 109.512 seconds of collection and 7.126 seconds
of update requests; these are subsets of 127.618. The three validation collection
reports are 51.652, 51.602 and 50.092 seconds. Consequently, faster Dataset
build/inspect code would address only a small part of this particular test.
Changing fixture scope alone would not eliminate this module-scoped construction.

### Legacy fixed-data originals are regenerated fifteen times

The function-scoped `frozen_sources` at `tests/test_fixed_ppo_data.py:17` is also
imported by `tests/test_native_session.py:22`. The original run contains five
fixed-data consumers and ten native-session consumers, for **15 constructions**.
Each construction creates three one-update synthetic parents and validates each
final policy: 45 training calls and 45 validation calls overall. This is not a
claim that every validation call starts a learner; this fixture uses the
in-process synthetic collector from `tests/test_ppo_lineage.py:53`.

Recorded setup totals are 51.55 + 102.20 = **153.75 seconds**, approximately
10.25 seconds per construction and **8.84% of the full wall time**. Native
session's own test bodies total only 20.22 seconds. Fixtures therefore account
for most of that file's apparent cost.

These working originals cannot simply be shared as mutable directories:
`test_native_source_mutation_invalidates_finalization_and_slices_survive_close`
and `test_frozen_dataset_rejects_omission_reordering_and_changed_original`
deliberately mutate them. Copied artifacts also carry original paths, hashes,
chain and Source identities. Any future reuse needs isolated, correctly bound
copies and preserved end-to-end generation coverage.

## Additional bounded measurement: actor initialization dominates collection

A disposable instrumented rerun of
`tests/test_structured_ppo.py::test_default_mlp_cpu_original16game_two_update_common_loop`
PASS: one passed, exit 0. The same test's original full-suite call was 43.917
seconds; the isolated diagnostic call was 49.17 seconds. They are separate runs
with different preload/order and instrumentation, not a speed comparison.

The instrumenter wraps existing synchronous/asynchronous calls and Source
context entry/exit using monotonic wall intervals. It leaves all original calls,
return values, test assertions, budgets, input sizes and child environments in
place. Application/test files remain unchanged. Exact command:

```sh
/usr/bin/time -v -o /tmp/godfieldbots-pytest-timing-20261108T1404/profile-one-resource.txt timeout --signal=INT --kill-after=10s 120s uv run python /tmp/godfieldbots-pytest-timing-20261108T1404/profile_one.py > /tmp/godfieldbots-pytest-timing-20261108T1404/profile-one.log 2>&1
```

| Instrumented step | Count | Wall seconds, overlapping intervals merged |
| --- | ---: | ---: |
| Both collection invocations | 2 | 38.354 |
| Waiting for episode actors to become ready, inside collection | 32 | 36.562 |
| Actor construction/Popen calls | 64 | 0.058 |
| Actor selection calls, including request/response handling | 128 | 1.406 |
| Numerical child initialization | 1 | 3.696 |
| Numerical update requests | 2 | 2.476 |
| Source projection context entry | 2 | 0.950 |
| Actor cleanup | 64 | 0.952 |
| Numerical child close | 1 | 1.099 |

**Actor readiness waits overlap 95.33% of measured collection wall time.**
The table contains nested categories and is not additive. The readiness calls
sum to 70.981 seconds before merging because two games can run concurrently;
the merged 36.562-second value avoids counting overlapping waits twice.
Likewise, original report `model_startup_ms` values are latencies since each
worker's spawn and include waiting for earlier actors; summing them would not
give elapsed collection time.

`StructuredParticipantWorker` launches a clean interpreter for each actor at
`src/gf_simulator_client/structured_worker.py:75`; `_ready_actors` awaits its
ready reply at `src/gf_simulator_client/structured_environment.py:117`.
The child imports the policy factory, loads the model, resets policy state and
constructs the generator before replying, at
`src/gf_simulator_client/structured_bootstrap.py:55`. Neural policy creation
also reaches Torch runtime initialization. The measured dominant interval is
this ready wait, rather than the short parent Popen call. Import execution,
artifact loading, scheduling and other child initialization have **not** been
separately attributed with an import/CPU profile.

Raw diagnostic evidence: `profile_one.py`, `profile-one.log`,
`profile-one-junit.xml`, `profile-one-resource.txt`, `profile-one-events.json`
and `profile-one-summary.json` in the evidence directory above.

## Follow-up priorities and limits

1. Investigate reuse of immutable legacy `frozen_sources` templates with
   separately owned test copies. The measured preparation pool is 153.75 seconds;
   this is not a promised saving because template creation, copying, identity
   rebinding and integrity checks still have costs.
2. Profile fresh neural actor initialization further before choosing an import
   or model-loading optimization. This is the dominant measured substep in the
   representative collection, and substantial collection time recurs in both
   Python/native PPO and Tree-source preparation.
3. Review which corruption/contract checks need fresh full collection, preserving
   dedicated real-child continuous/resume, three-model, Source and lifecycle
   coverage. Shared workers would change the clean-interpreter/private-history/
   failure ownership prescribed by ADR0023 sections 8 and 13 and require the
   architecture gate; such a change is not proposed as a local fixture edit.

The full command remains a roughly 29-minute verification in the measured
environment. No optimization or new default test selection is implemented.
The full run and diagnostic each have one observation; no variance, CPU-sample
attribution, detailed import attribution, disk bottleneck or speedup is claimed.
Ruff/mypy and unrelated tests were NOT RUN for this read-only analysis and
documentation change. Recorded engineering/synthetic tests do not establish a
formal learning Source or model strength.

## Follow-up: data equivalence and duplicate work — 2026-11-09 JST

The owner requested checking data equality and identifying concrete repeated
work before selecting an optimization. This follow-up adopts no implementation
change or fixture-sharing design.

### Fifteen constructions repeat the same three seed-specific workloads

A read-only comparison located all 15 original fixture roots: five fixed-data
consumers and ten native-session consumers. For each seed separately, 17/29/43,
the following comparisons across the 15 roots PASS:

| Compared material | Observed equality and explicit exclusions |
| --- | --- |
| Effective configuration | Equal after excluding only each test's `server.binary` path; all referenced synthetic binary-identity files have identical contents |
| Training and validation schedules | Exactly equal within each seed |
| Generation-0 weights, generation-1 weights, generation-1 resume tensors | Exact array equality within each seed across all 15 parents |
| Complete decision input objects | Equal at all 32 training and 128 validation decision slots per seed/copy |
| Candidate action lists | Exactly equal at every slot |
| Remaining candidate metadata | Equal after explicitly excluding only measured `latency_ms` |
| Training selections | Equal, including index, log-probability, value and policy identity |
| Validation selections | Equal except one deliberately spoofed post-test `policy_version`; its index, log-probability and value still agree |
| Outcomes, end kinds, participant/opponent assignments and replay facts | Equal within each seed |

This covers 45 parents and 7,200 decision records across all fixture copies.
The seed-17, seed-29 and seed-43 final weight/resume archives have three distinct
hash groups: there are three required seed-specific variants, not one universal
training result. Within a seed, generation-1 weight NPZ archive bytes also match
across copies.

Raw source trees are not byte-identical. Their run/Game/command IDs, lineage,
temporary paths, measured times, candidate latency and resulting hashes differ.
Seven consumers deliberately mutate source status or content; eight leave the
parent/validation sources unmodified. The seven include failed/running parent
status injection, added newline/space in a source journal and one spoofed
selection policy identity. These differences remain reported, not normalized
out of the original artifacts. Post-test snapshots cannot prove the original
pre-mutation bytes of the changed files; the equality claims above apply to the
recorded fields, with that one selection exception retained explicitly.

Consequently, a possible reusable template would have to retain all three seeds
and keep independently owned mutable copies and valid source identities. This
comparison does not establish the correctness or cost of copying/rebinding,
concurrent sharing, or cache invalidation. No post-test mutated root is proposed
as a clean template.

Read-only inventory command, exit 0:

```sh
UV_CACHE_DIR=/tmp/godfieldbots-pytest-timing-20261108T1404/uv-cache uv run --no-sync python /tmp/godfieldbots-pytest-timing-20261108T1404/build_fingerprint_inventory.py
```

Saved evidence in the original measurement directory:
`equivalence-inventory.json`,
`complete-input-candidate-fingerprint-inventory.json`,
`legacy-source-equivalence-summary.md` and `duplicate-work-findings.json`.
The complete inventory retains per-root/per-event fingerprints and the mapping
to the original pytest node IDs. No pytest, training or writes to original
source trees were performed by this comparison.

### Concrete repeated computation in the legacy synthetic collector

At `tests/test_ppo_lineage.py:59`, the collector iterates over participant
journals. For each neural participant it constructs `MLPPolicy`, reloads the
checkpoint and resets sampling state. Saved journal populations and the source
loop imply 24 such loads in the training collection and 64 in final validation
per seed: **264 policy constructions per fixture, 3,960 across these 15 fixtures**.
This is a source-derived execution count, not a runtime call trace.

Two additional repetitions are visible:

- `MLPPolicy.select` already encodes the decision and calls the network at
  `src/gf_models/mlp/policy.py:29`. The collector repeats encoding and forward
  at `tests/test_ppo_lineage.py:77` and `:81` for the stored selected index.
  All these neural journals contain one decision, implying two encodes/two
  forwards per record, or 7,920 forwards across the 15 fixtures. The second
  computation has a purpose: it preserves the original selected index rather
  than the new sampled index. Sharing its result would need to preserve that
  behavior and sampling semantics.
- Legacy `load_artifact` at `src/gf_models/mlp/model.py:102` invokes
  `inspect_artifact`, whose line 98 decodes/validates weights, then line 105
  decodes those weights again for restoration. This legacy path therefore has
  two archive decodes per policy load. Its own isolated elapsed cost was NOT RUN.

The already measured 153.75-second fixture preparation pool includes this work;
the counts are not additional measured seconds and do not predict a saving.

### Structured participant loading: measured duplicate counts and cost

The active structured load path inspected at
`src/gf_models/structured_artifacts.py:332` performs:

1. `_inspect_inference` including one full `inspect_bundle`.
2. `read_bundle`, which calls `inspect_bundle` again, then `load_arrays`.
3. `make_network(model, 0, ...)`, including default and orthogonal parameter
   initialization, followed by `restore_weights` over the initialized parameters.

One clean CPU participant was initialized from each original MLP/GRU/Transformer
generation-0 checkpoint. All three readiness exchanges succeeded, all original
validation calls executed, and all participants were reaped with empty resource
ledgers. Existing normal close sends SIGTERM; saved child exit codes are -15.

| Instrumented step | MLP | GRU | Transformer |
| --- | ---: | ---: | ---: |
| Parent-measured readiness, seconds | 1.567 | 1.358 | 1.398 |
| First Torch/runtime setup, milliseconds | 1,378.9 | 1,166.4 | 1,199.1 |
| All five file-hash calls, milliseconds | 8.0 | 12.1 | 18.5 |
| Orthogonal initialization calls | 30 | 36 | 42 |
| Orthogonal initialization, milliseconds | 17.6 | 19.8 | 19.2 |
| Saved-weight restoration, milliseconds | 1.6 | 1.9 | 2.4 |

Each checkpoint has one weights shard. Observed counts are **two bundle
inspections, three NPZ hashes, two sidecar hashes and one array decode per load**.
`inspect_bundle` validates metadata/file inventory/sidecars but does not decode
tensor arrays. This structured path's decode count differs from the legacy MLP
loader above.

The duplicate hash/initialization work is present, but its measured cost here is
in milliseconds. First Torch/runtime setup is much larger. The runtime timer
includes Torch import and runtime configuration; it does not isolate pure import
time. The diagnostic preloads application modules to install wrappers, retaining
a cold Torch import. Its scopes are inclusive and nested and must not be added
to one another. Each model has one observation; no speedup is established.

Hash, shape, finite-value and ownership checks occur at different boundaries.
Some repeated checks can defend against source mutation between validation and
use. Learner restore also deliberately stages validation before publishing live
network/Adam/RNG state (`src/gf_training/structured_numeric.py:227`). The audit
does not classify any guard as safely removable merely from its repeated count.

Exact successful probe command, exit 0, no game or training execution:

```sh
timeout --signal=INT --kill-after=10s 120s uv run python /tmp/godfieldbots-pytest-timing-20261108T1404/profile_initialization.py > /tmp/godfieldbots-pytest-timing-20261108T1404/initialization-probe-v2.log 2>&1
```

The first probe failed its diagnostic-only assertion that normal close must
return exit 0, and its finalizer-based logging did not survive normal SIGTERM.
That failed attempt is preserved in `profile_initialization_v1.py` and
`initialization-probe-first.log`. The corrected probe writes measurement data
before sending the unchanged ready reply and accepts the original normal close
semantics. No application fix was needed.

Successful evidence: `profile_initialization.py`,
`initialization-probe-v2.log`, `initialization-summary.json`,
`initialization-{mlp,gru,transformer}.json`, `initialization-steps.csv` and
`duplicate-work-findings.json`. A separate read-only verifier checked the call
counts, timing scopes, pre-Torch CPU environment and normal close interpretation.
No implementation or optimization was adopted.

The investigation above preceded the shared-fixture implementation. Additional
usage discovery found two more active consumers, bringing actual original-run
generation to17; the fifteen-copy equivalence comparison remains scoped to the
two named files. The subsequently accepted change and its controlled comparison
are recorded in [ADR0025](../adr/0025-session-shared-pytest-ppo-sources.md) and
[shared-source verification](shared-pytest-ppo-sources.md).

## Current Tree collection profile (2026-11-09)

The current full-run Tree Dataset file takes 199.312s, including 191.87s fixture
setup and 7.44s Dataset build/inspect. Its saved normal records attribute 72.298s
to training collection and 100.822s to final-policy validation collection. Those
173.120s motivate a separate, single diagnostic of the original full Tree test.

At revision `bd6a964`, the disposable parent-only observer wraps original calls
and preserves arguments, return values, exceptions, environments, process
ownership, assertions and budgets. It runs all 288 games / 576 episode actors /
1,024 decisions, plus the original three validation preflight actors. No test or
application file is modified. All 352 frozen Python/config hashes still match.

```sh
env -u GFB_RUN_CPU_PRELOAD_SAFETY -u GFB_TREE_DATASET_EXISTING_SOURCE17 -u GFB_TREE_DATASET_EXISTING_SOURCE29 UV_CACHE_DIR=/tmp/godfieldbots-uv-cache /usr/bin/time -v -o /tmp/godfieldbots-tree-collection-profile-20261109/resource.txt timeout --signal=INT --kill-after=15s 480s uv run --no-sync python /tmp/godfieldbots-tree-collection-profile-20261109/profile_tree.py > /tmp/godfieldbots-tree-collection-profile-20261109/pytest.log 2>&1
```

PASS one test in 200.56s; external wall 201.51s, exit0. This diagnostic is a
different invocation from the 199.312s normal full-run case, not a speed comparison.
Intervals below are clipped to collection and merged to remove simultaneous
calls within each category. Different categories remain nested or overlapping;
their rows must not be summed.

| Interval | Calls | Merged wall seconds |
| --- | ---: | ---: |
| Collection | 9 | 167.616 |
| Episode actors ready wait | 288 | 148.338 |
| Validation preflight ready roundtrip | 3 | 2.497 |
| Both readiness categories, disjoint union | 291 | 150.836 |
| Selection and decision journal | 1,024 | 11.996 |
| Episode finalization | 288 | 7.198 |
| Actor spawn / reservation call | 579 | 0.153 |
| Decision journal write, nested within selection | 1,024 | 2.708 |

The two readiness categories cover **89.9888%** of measured collection wall.
The three preflight intervals are pairwise disjoint, occur before episode actor
creation in their validation collection, and do not overlap episode ready waits.
The 579 actor/ready samples include 576 episode actors plus three MLP validation
preflight actors (`gf_cli/structured.py:104`), all inside collection.

Handshake-reported `model_load_ms` medians are MLP **782.421ms** (339 samples),
random **0.220ms** (120) and heuristic **0.220ms** (120). This child timer covers
factory import/model creation and loading, reset and generator construction;
it does not isolate Torch import. Readiness also includes interpreter/bootstrap,
IPC, scheduling and earlier sequential initialization waits. Summed child times
are neither elapsed collection wall nor measured CPU consumption.

Existing normal-result decision timers independently sum to 2.352s policy
calculation and 0.577s candidate generation across all 1,024 decisions. These
normal-run child sums are separate from the diagnostic merged intervals above.
The test uses FakeHTTP; these figures do not measure real simulator game-rule
execution or establish model quality / qualified Source acceptance.

Raw evidence: `/tmp/godfieldbots-tree-collection-profile-20261109`, including
`metadata.json`, `profile_tree.py`, `profile-one-{events,summary,junit}`,
`ready-samples.json`, `collection-breakdown.json`, original-result aggregates,
logs and resources. An independent read-only verifier recomputed counts,
intersections/unions, readiness ratio, medians and all frozen hashes: PASS.

## Other Torch-startup candidates after ADR0031 (2026-11-09)

Read-only investigation at `fd0ab59`; no optimization or test-selection change.
The [latest full-run CSV](pytest-file-timings-tree-preload-20261109.csv) supplies
priorities. Two bounded source audits distinguish clean Torch-owning child
processes from repeated in-process model creation and isolated no-import checks.
Only three selected test populations receive an additional diagnostic, with all
original calls, arguments, assertions, budgets and child environments retained.

```sh
env -u GFB_RUN_CPU_PRELOAD_SAFETY -u GFB_RUN_CARD_INSTANCE_SOURCE_CUDA UV_CACHE_DIR=/tmp/godfieldbots-uv-cache /usr/bin/time -v -o /tmp/godfieldbots-other-torch-init-audit-20261109/resource.txt timeout --signal=INT --kill-after=15s 480s uv run --no-sync python /tmp/godfieldbots-other-torch-init-audit-20261109/profile_other.py > /tmp/godfieldbots-other-torch-init-audit-20261109/profile.log 2>&1
```

PASS **41 passed / 10 CUDA skips / zero failures**, 51 selected cases in194.02s,
external194.87s, exit0. All51node/status pairs match the previous full run.
Full-file and diagnostic columns below are distinct single observations; the
numeric-worker diagnostic intentionally selects only its three real CPU wire
cases, not all28cases in that file.

| File | Latest full-file seconds | Diagnostic case-seconds | Measured repeated startup work |
| --- | ---: | ---: | --- |
| `test_native_structured_session.py` | 118.494 | 118.743 | 256 fresh episode participants; eight collections91.829s, episode-ready union82.984s; six learner initializations8.239s |
| `test_card_instance_source.py` | 60.445 | 61.629 | 19 numerical learner initializations25.423s, closes5.850s |
| `test_structured_numeric_worker.py` | 16.339 | 12.885 | Six learner initializations8.246s across three CPU tests, closes1.915s |

Native session readiness covers **90.368%** of measured collection wall. Its
256participants are192MLP/32random/32heuristic, not256Torch loads. Reported MLP
model-load median is783.996ms, while baseline medians are about0.22ms. The first
source-export case creates32participants, the two-batch case64, continuous/pause/
resume128, and late-source-mutation32; marker-only dispatch creates none. All use
fresh generic `component_http` defaults without template/pool, including native
source-op paths. This is the strongest next actor-preload candidate.

Card-instance source uses19fresh `GameLearningProcess` children:10CPU projection/
precommit cases and9live-owner guard cases. Outer initialization accounts for
41.252% of its diagnostic file. The six CPU wire child initializations account
for63.996% of their three selected tests; those cases intentionally stop the
first process and restore in a second actual process. Import-state reuse must
retain that fresh-process/restore coverage.

The reported numeric initialization spans include startup wait, model/optimizer
construction or restore, artifact checks and IPC. They are **not pure Torch
import times or promised savings**. Low-level `LearningProcess.request` samples
cover IPC/read/JSON parse only; structured reply and artifact verification runs
after that return and belongs to the outer initialize interval. Child reply
initialize/save timings are nested, not extra costs to add. Separate categories
may overlap; per-label unions remove concurrent episode double counting.

Additional static candidates and limits:

- `test_pooled_structured_worker.py` totals57.591s. Its five collection-finalizer
  cases account for37.250s and use participant pools but do not activate Torch
  preloading; five numerical children still start independently. Its three-model
  wire comparisons account for16.454s and deliberately create15cold neural
  reference participants. Preserve independent cold reference coverage rather
  than silently routing it through the optimized path.
- `test_ppo_lineage.py` (35.193s), `test_card_instance_ppo.py` (18.740s) and the
  legacy native/GRU/Transformer coordinator tests also create numerical workers.
  Source inspection supports investigation, not a measured import-cost claim
  for those complete files.
- `test_structured_lineage.py` (38.282s), `test_card_instance_tree_fixture.py`
  (21.066s) and `test_card_instance_tree_artifacts.py` (34.921s) mostly construct
  models/data or inspect sources within the already-imported pytest process.
  Their isolated no-Torch/no-optional-import subprocess checks have a different
  purpose; observed duration alone is not evidence for Torch preloading there.

Current helpers explicitly exclude native `source_ops` from preload/finalizer
admission; an extension must preserve exact native source, learner/ledger and
before-update original checks under design review. Numerical workers run the
separate `ppo_worker` protocol, not the existing participant template bootstrap.
They need a separately reviewed launch design; existing actor flags do not
activate learner preloading. No such implementation is made in this investigation.

Evidence: `/tmp/godfieldbots-other-torch-init-audit-20261109`, including observer,
353-file metadata, node-tagged raw events, actual ready/numeric reply samples,
JUnit/log/resources and `per-file-breakdown.json`. The disposable observer's old
aggregate error-counter treats `outcome='measured'` request markers as errors;
its project proxy also lacks exception outcome attribution. Those counters are
not used as test/failure evidence. The corrected per-file reader retains valid
intervals and actual reply samples while omitting those counters; raw artifacts
are preserved. Independent audit recomputed all49label rows, actor counts and
353current source hashes: PASS, no mismatches. No second profile, full pytest,
Ruff/mypy or package checks were run for this read-only investigation.
