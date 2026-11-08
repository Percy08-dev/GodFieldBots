# Structured PPO test reference reuse — 2026-11-09

Implementation and Root acceptance complete after both final independent reviews.
Design:
[ADR0026](../adr/0026-structured-ppo-test-reference-reuse.md).

## Scope and review

The owner requests reducing `tests/test_structured_ppo.py`. Design review is
READY for one module-local default CPU MLP two-update actual execution serving
three groups of assertions. The ordinary common-loop test and completed-parent
consumers reuse its recorded evidence; fresh pause/resume uses its exact canonical
LoadedConfig, retaining an independent parent and original three-model comparisons.
No production, dependency, process pooling, numerical recipe, Source/guard or
budget change is included. The legacy ADR0025 fixture is a different workload.

Prior-run comparison confirms model/action/run configuration equality except
temporary binary path, equal code/integration identity and matching saved
generation0/2 weight/resume arrays across the old MLP runs. Corrected direct
numeric-array comparisons are detailed below; raw input equality is not claimed.
Config/resume/run/Game identities differ and remain explicitly distinguished.
The canonical configuration is reused rather than rewriting checkpoint identity.
The bounded comparison inventory is
`/tmp/godfieldbots-pytest-timing-20261108T1404/three-mlp-parent-equivalence.json`.

## Baseline

Baseline tracked revision `eb0d296`, no tracked code/test edits during the HOST
measurement. Exact command from the repository root:

```sh
/usr/bin/time -v -o /tmp/godfieldbots-structured-ppo-tests-20261109/before-resource.txt timeout --signal=INT --kill-after=15s 900s uv run pytest tests/test_structured_ppo.py --durations=0 --durations-min=0 --junitxml=/tmp/godfieldbots-structured-ppo-tests-20261109/before-junit.xml > /tmp/godfieldbots-structured-ppo-tests-20261109/before.log 2>&1
```

PASS **35passed**, exit0; pytest**546.36s**, external wall**546.87s (9:06.87)**.
This includes all module fixtures,35test bodies and cleanup. No skip was added.
The previous whole-suite JUnit total542.09s describes a different invocation;
the controlled module before/after comparison uses this fresh baseline.

## Implemented scope and independent review

Only `tests/test_structured_ppo.py` and `tests/structured_ppo_support.py` change
executable code. `CompletedComponentExecution` retains the actual canonical
configuration, parent/result and closed diagnostic tuples, with original-file
inventory checks. A lazy module fixture supplies the old Path adapter and the
ordinary/common MLP assertions. The optional comparison argument defaults to
fresh execution for old callers; CPU MLP alone dynamically requests reuse.

Both independent final reviews are READY. AST review confirms all35original
test functions/parametrizations, all10common-loop assertions and26comparison
assertions remain. The four lifecycle assertions consume combined actual closed
reference/fresh evidence. Existing configuration, HTTP collector and clone helper
bodies and all six CUDA caller cases are unchanged.

Targeted HOST command, exit0:

```sh
timeout 240 .venv/bin/python -m pytest -q tests/test_structured_ppo.py::test_default_mlp_cpu_original16game_two_update_common_loop 'tests/test_structured_ppo.py::test_default_cpu_three_models_exact_continuous_vs_pause_resume[mlp]' tests/test_structured_ppo.py::test_completed_reference_mismatch_rejects_before_any_activity --basetemp=/tmp/godfieldbots-structured-ppo-tests-20261109/implementer-targeted --junitxml=/tmp/godfieldbots-structured-ppo-tests-20261109/implementer-targeted.xml > /tmp/godfieldbots-structured-ppo-tests-20261109/implementer-targeted.log 2>&1
```

PASS **9passed in95.33s**, including seven preactivity refusal cases. Collection
PASS42CPU-module nodes plus six unchanged opt-in CUDA nodes. Actual CUDA execution
is NOT RUN for this test-only change. Reference inventory and physical cleanup
assertions pass at reuse and module teardown. Original35cases are retained.

Prior-run equality applies to actual generation0/2weight/resume NPZs and all
encoded numerical inputs, candidate tensors, selections, advantages and targets.
Generation1physical weight/resume arrays were pruned; its inference descriptor
and saved weight digest match, not directly inspected arrays. Each update has
64participant decision records but32accepted learner decisions,64accepted learner
decisions across two updates. Selection index/logprob/value/policy version match;
table hashes, command UUIDs and physical action-instance IDs remain different.
Those source identities remain unchanged within each source; reference values
are not injected into fresh paused training.

## Corrected raw-source comparison

The preliminary raw-journal comparison used legacy `input`/`candidates` fields
that schema5decisions do not contain. Its raw-input/table equality claim is
withdrawn; actual schema5fields are `frame`/`table`. The original pytest-147
artifacts subsequently expired under ordinary pytest retention. This is an
analysis-tool error, not a changed application format or a weakened comparison.

Root directly audited the retained baseline pytest-148three references against
the new pytest-149reference. All generation0/2weight/resume NPZ bytes match all
three old runs. Each numeric batch contains736arrays:704nonpayload arrays match
exactly. New numeric archives match the old basic-loop run exactly; compared with
the other two references,32raw payload arrays differ only at:

- `action_options.action_options.usable_hand_elements[0]`
- `observation.own.hand[0].instance_id`
- `history_end` (opaque cursor embedding the scripted Game ID)

No opaque IDs are rewritten or ignored during ordinary Source validation. The
independent correctness reviewer checked nonempty payloads, these exact array
counts and the tokenization/own-reference mapping: references resolve to the same
own slots/usable flags; cursor bytes are not numerical features. Both final
reviews retain READY with this narrower, corrected evidence. Receipt:
`/tmp/godfieldbots-structured-ppo-tests-20261109/reference-reuse-audit.json`.

## Controlled after measurement

The after command is exactly the baseline module command, replacing the three
output names from `before` with `after`. HOST PASS **42passed**, exit0, including
all35original node IDs plus seven new admission-guard cases. Pytest454.96s;
external wall**455.41s (7:35.41)**. All35original cases retain PASS status.

| Whole module including fixtures/cleanup | Before | After |
| --- | ---: | ---: |
| External wall seconds | 546.87 | 455.41 |
| Passed cases | 35 | 42 |
| Skipped/failed cases | 0/0 | 0/0 |

One measured pair saves**91.46seconds (16.72%)**, despite the seven added guards.
No original scenario, game/update budget or assertion is removed. Timing variation
and a single pair remain limitations. Machine-readable population/arithmetic
audit: `/tmp/godfieldbots-structured-ppo-tests-20261109/comparison.json`.

The retained before roots are under pytest-148; after roots under pytest-149.
The original three completed default CPU MLP reference parents become one actual
completed fixture parent with one invocation. No separate ordinary-loop parent
or MLP continuous parent is regenerated. Fresh MLP pause/resume has a distinct
run ID, two actual invocations and the same canonical config/resume identities.
Both reference and resumed parents validate with two updates/64accepted learner
decisions/8optimizer steps. The reference inventory/physical-cleanup checks also
pass at fixture teardown. Exact saved generation0/2state bytes match all three
old references; the direct numeric/source caveat is recorded above.

## Related and selective regression checks

Exact HOST command, exit0:

```sh
/usr/bin/time -v -o /tmp/godfieldbots-structured-ppo-tests-20261109/regression-resource.txt timeout --signal=INT --kill-after=15s 600s uv run pytest 'tests/test_structured_ppo.py::test_default_cpu_three_models_exact_continuous_vs_pause_resume[gru]' tests/test_structured_ppo_source.py tests/test_structured_source.py tests/test_structured_numeric_worker.py tests/test_structured_protocol.py tests/test_structured_ppo_cuda.py --setup-show --junitxml=/tmp/godfieldbots-structured-ppo-tests-20261109/regression-junit.xml > /tmp/godfieldbots-structured-ppo-tests-20261109/regression.log 2>&1
```

PASS **219passed / 6skipped in142.47s**. The six skips are the unchanged opt-in
CUDA cases. The actual selective GRU execution generates its own complete
two-update continuous parent and performs pause/resume; setup traces contain
zero `completed_execution`/`completed_component`/`paused_component` fixtures and
no shared-MLP template directory. Receipt: `regression-audit.json` in the log
directory. Thus selecting GRU does not add a default MLP reference execution.

Full collection PASS**4,232tests** in2.23s via `uv run pytest --collect-only -q`;
log `collection-all.log`. `uv run ruff check src tests native`, mypy157source
files and whitespace checks PASS. Full-root Ruff is not claimed: the previously
reported four findings in unrelated untracked `codex_analysis.py` remain outside
this change. Production sources/dependencies/Simulator and all existing support
helper/CUDA caller bodies are unchanged.

## Acceptance and limits

Root accepts ADR0026 after final correctness/conformance reviews remain READY on
the corrected source evidence, actual module timing, selective/related tests,
reference ownership/inventory and numerical equivalence checks. Actual CUDA
execution and a new whole-suite execution are NOT RUN for this bounded test-only
change. The immediately prior whole suite3978passed/247skipped belongs to
ADR0025, not this result. No whole-suite speed estimate, model strength or formal
Source acceptance is inferred from these synthetic component executions.

## Post-commit module breakdown

A fresh HOST measurement at revision
`5d9824d4eafae328be5013c435bb32d05ed90a76` PASS**42tests**, exit0.
External wall**455.89s (7:35.89)**; pytest454.94s. A disposable pytest hook
records exact setup/call/teardown and fixture setup durations; it does not patch
application or test functions. No tracked code/test changes are made.

Exact command:

```sh
PYTHONPATH="/tmp/godfieldbots-structured-ppo-breakdown-20261109${PYTHONPATH:+:$PYTHONPATH}" /usr/bin/time -v -o /tmp/godfieldbots-structured-ppo-breakdown-20261109/resource.txt timeout --signal=INT --kill-after=15s 900s uv run pytest tests/test_structured_ppo.py -p pytest_breakdown_probe --setup-show --durations=0 --durations-min=0 --junitxml=/tmp/godfieldbots-structured-ppo-breakdown-20261109/junit.xml > /tmp/godfieldbots-structured-ppo-breakdown-20261109/pytest.log 2>&1
```

| Phase | Seconds |
| --- | ---: |
| Setup | 69.13 |
| Test call | 384.84 |
| Teardown | 0.03 |
| Work outside test phases, including startup/collection/reporting | 1.89 |
| External wall total | 455.89 |

The setup phase includes `completed_execution`47.87s and `paused_component`21.21s,
each initialized once. They are nested within their requesting test's setup and
are not extra costs to add to this table. The ordinary MLP common-loop case's
48.76s includes shared reference setup47.87s; its own call is0.89s.

These disjoint groups account for every case once, including setup and teardown:

| Workload | Cases | Seconds |
| --- | ---: | ---: |
| Transformer continuous/pause-resume comparison | 1 | 91.58 |
| GRU continuous/pause-resume comparison | 1 | 90.33 |
| Update/save failure controls | 3 | 62.35 |
| MLP pause-resume comparison using the shared reference | 1 | 49.15 |
| Common MLP assertions plus shared reference setup | 1 | 48.76 |
| Direct/DeepSets hand encoder cases | 2 | 43.42 |
| Other source/configuration/resource/admission checks | 30 | 25.95 |
| Rehashed execution refusal plus paused-reference setup | 2 | 23.60 |
| Interrupted collection control | 1 | 18.86 |
| Case total | 42 | 454.00 |

GRU and Transformer together consume39.90% of external wall time; all three
model comparison cases consume50.68%. In the stored successful run records,
GRU's collection totals72.60s and update requests5.80s; Transformer's72.92s and
6.47s. The update timers include request/validation/saving, not pure optimizer
math. These recorded workflow scopes are subsets of the case durations; their
initialization/source verification/cleanup remainder is not separately profiled
by this observer. Interrupted/failed runs are not assigned guessed stage costs.

Evidence directory: `/tmp/godfieldbots-structured-ppo-breakdown-20261109` contains
`pytest_breakdown_probe.py`, `timings.json`, `junit.xml`, `pytest.log`,
`resource.txt`, `summary.json`, and complete `tests.csv`, `groups.csv`,
`fixtures.csv`, `workflows.csv`. The aggregation checks all126phase reports,
42case totals and nonoverlapping group arithmetic. This is one measurement with
diagnostic reporting, not a repeated-run benchmark or a new optimization.

A later representative GRU diagnostic separates collection readiness, selection,
HTTP, storage/cleanup and per-child import loading in
[collection detail](structured-ppo-collection-detail.md). Its instrumented
single-case timings are not a replacement for the ordinary module benchmark.
