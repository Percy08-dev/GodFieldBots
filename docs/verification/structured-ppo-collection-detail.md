# Structured PPO collection detail — 2026-11-09

Read-only diagnostic at revision `37ac83bd77c23c14d178aa2315106a4cde7d905b`.
No application, test, dependency, policy input or experiment budget changes.
The representative case is the original CPU GRU continuous/pause-resume test.

## Method and result

HOST command, repository root:

```sh
/usr/bin/time -v -o /tmp/godfieldbots-collection-detail-20261109/resource.txt timeout --signal=INT --kill-after=15s 300s uv run python /tmp/godfieldbots-collection-detail-20261109/profile_collection.py > /tmp/godfieldbots-collection-detail-20261109/pytest.log 2>&1
```

The disposable runner invokes:
`tests/test_structured_ppo.py::test_default_cpu_three_models_exact_continuous_vs_pause_resume[gru]`.
PASS one test, exit0; test call98.973s, pytest99.49s, external wall101.00s.
This isolated instrumented run differs from the earlier ordinary-module90.33s
GRU case in preload/order and tracing. It is diagnostic, not a speed comparison.

The runner wraps existing parent methods, returning their unchanged results and
preserving arguments/exceptions/context-manager behavior. For actual clean
participant commands it adds `-X importtime` and redirects stderr from DEVNULL to
per-child files. The original `-E`, bootstrap, clean environment, ready messages,
policy configuration, physical cleanup and all test assertions remain. Tracing
adds I/O/measurement cost; per-child imports are elapsed times, not CPU time.

An independent read-only verifier checked this method and saved result:128actual
participants (96GRU/16Random/16heuristic), four collections,64games,256decisions,
three learner initializations and four updates. Every actor has one actual ready
reply/import log. `initialize` is invoked384times because selection checks it
again;256calls are already-initialized checks, not extra process/model startups.

## Whole test-call accounting

Intervals below are clipped to the98.973-second call and partitioned without
double counting. A defined priority resolves nested coordinator scopes, with
collection first and commit before its internal checks. Uninstrumented intervals
remain explicitly unclassified; no residual is guessed to be numerical work.

| Stage | Seconds |
| --- | ---: |
| Online scripted collection and recording | 75.696 |
| Numerical child start/initialize/restore | 5.266 |
| Numerical update requests, including save/validation | 6.018 |
| Python source-projection entry/exit | 1.999 |
| Generation commit/publication | 0.514 |
| Parent/history validation through the instrumented adapter | 0.496 |
| Inference binding construction | 0.796 |
| Retention | 0.665 |
| Numerical child close | 1.793 |
| Other/test-side imports and final assertions, not separately attributed | 5.729 |
| Total | 98.973 |

## Collection wall time and concurrency

The four collection calls do not overlap. Two games run concurrently within
each collection, so elapsed calls from separate games must not be summed as
collection wall time. The disjoint partition is:

| Active work | Seconds |
| --- | ---: |
| Participant readiness wait only | 68.766 |
| Readiness wait overlapping other measured work | 2.684 |
| No readiness wait active | 4.246 |
| Collection total | 75.696 |

Readiness wait exists for71.449s, **94.39%** of collection wall. That means a
controller is awaiting readiness during those intervals, not that it consumes
that much CPU. The full mutually exclusive overlap combinations are preserved
in `collection-partition.csv`.

Other scopes, reported as wall unions across calls, are nested/overlapping and
are NOT additive with the table above: selection+frame work+Journal begin3.229s;
worker request/selection2.789s; actor close1.881s; Journal begin0.691s;
Journal finish0.191s; actor construction/Popen0.174s. The HTTP transport is the
unchanged MockTransport, not an external Simulator server.

All1,984instrumented client-method invocations total1.438elapsed call-seconds;
this includes nested attestation and concurrent work, not network wall time.
ActionOptions calls256total0.852s, observation640total0.224s, catalog32total0.119s,
Act256total0.061s, events256total0.046s and Replay128total0.020s. These contain
client parsing/validation and mock handling, not just transport.

## Participant model loading

GRU's96clean participants each load their model. The original ready response
provides `model_load_ms`; each matching PID's import trace provides import times.
Means across the96participants:

| Per-GRU-participant model-load substep | Mean seconds |
| --- | ---: |
| First `torch` import | 0.747 |
| Later `torch._inductor.config` import, including Dynamo | 0.552 |
| Remaining model-load work, not separately attributed | 0.094 |
| Actual ready-response model-load interval | 1.393 |

The two import intervals occur in order at the same top-level import depth;
their sum is within the reported model-load interval for every child. Dynamo's
own0.528smean is nested inside the0.552s and is not added again. These are module
imports; this trace does not claim model compilation occurred. The remainder
includes other loading/configuration/initialization but is not assigned wholly
to any single operation. Normal process startup before factory loading is also
outside `model_load_ms`.

Per-child averages and sums overlap across processes; multiplying them by96does
not produce collection wall time. Baseline participants import no Torch here;
their tiny factory-load interval does not include the full interpreter/bootstrap
or time queued behind another participant's initialization.

## Per-decision observations

The unchanged reports retain256decision timing dictionaries. Their elapsed sums
are: input/frame preparation0.402s; candidate generation0.150s; policy selection
0.804s; whole worker decision1.239s; parent request3.122s; communication/decode/
host-wait remainder1.883s; Journal storage0.691s. These are nested scopes across
concurrent games and are not added to parent wall totals. In particular, policy
selection is not a pure neural-kernel measurement, and the communication remainder
is not a pure IPC syscall benchmark.

## Evidence and scope

Directory `/tmp/godfieldbots-collection-detail-20261109` retains the disposable
runner, raw `measurements.json`, all128actor import logs, `pytest.log`,
`resource.txt`, `junit.xml`, the aggregation `summarize.py`, `summary.json`,
`scopes.csv`, `models.csv`, `case-partition.csv`, `collection-partition.csv` and
`child-load-details.json`. The original test report paths are recorded in the
summary. The aggregation checks stage sums, overlap partitions, actual field
presence and expected worker/decision/update counts. No production optimization,
Rust-backend measurement, actual CUDA execution or whole-module rerun is claimed
by this one representative-case diagnostic.
