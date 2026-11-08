# ADR0026: Reuse one structured CPU MLP reference execution within its tests

Status: accepted; implementation and scoped runtime acceptance complete on
2026-11-09 after independent design, correctness and conformance reviews.

## Context and requirements

The owner requests reducing `tests/test_structured_ppo.py` runtime. Its35cases
take542.09seconds in the latest whole run. Repeated fresh participant/Torch
initialization dominates collection, but participant process/history/RNG/cleanup
ownership is an accepted production boundary (ADR0023); this change does not
alter that boundary, dependencies, numerical math or training budgets.

Three sites currently execute the same default CPU MLP two-update scenario:
the ordinary common-loop test, the `completed_component` template, and the
continuous side of the MLP continuous-versus-pause/resume test. Each costs about
43seconds and creates64real participant workers over32scripted games. Their
different temporary binary paths affect config/resume identities, so equality
cannot be inferred from the seed alone or obtained by relabelling artifacts.

Preserve all original35cases, every meaningful assertion, all three model
continuous/resume comparisons and existing failure/source-corruption coverage.
Keep16games per update, two updates per reference/resume comparison, original
wall/storage/admission/resource settings, actual child startup/reap and original
numeric/state/lineage comparisons. No worker reuse or mocked numerical learner.

## Responsibilities and interfaces

Test support holds a small completed-execution record containing the original
LoadedConfig, parent path, returned result and recorded scripts/workers/ledgers.
It contains only finished diagnostic evidence: no live child, thread, reservation
or monkeypatch context survives generation. No production code imports it.

A lazy module-scoped fixture in `test_structured_ppo.py` creates that record by
running the original two-update scenario once under its own MonkeyPatch context
and validating normal completed-parent/source and cleanup conditions. The
existing `completed_component` fixture remains a Path-returning adapter for its
current corruption/historical-reader consumers.

The ordinary common-loop test asserts its current result,64-worker/16-script,
cleared-ledger, generations and original batch/source facts against that actual
execution. It does not require an otherwise unused second fresh execution.

`run_continuous_and_resume` accepts an optional completed reference. Its default
remains the existing fresh continuous/pause/resume execution, including all
existing CUDA callers. The optional branch accepts only the exact default CPU
MLP/attention setting with matching actual recorded model/device/configuration;
it uses the original LoadedConfig for fresh pause/resume, not a new temporary
config or rewritten checkpoint identity. Wrong model/device/recorded execution
must fail before additional training or collection.

The CPU parametrized test requests the fixture dynamically only for its MLP
case. Selecting GRU/Transformer or CUDA alone must not generate an extra MLP
reference. Do not broaden the scope of other input or scenario fixtures.

## Data flow, information and ownership

The reference executes once and remains untouched. Existing destructive tests
continue using `clone_component_parent` for their own private copies. The
paused/resumed parent remains newly generated in the current test's directory,
with its independent schedule/RNG/process ownership and the canonical reference
configuration. It receives no future/reference outcomes or weights as policy
inputs. Reference evidence is used for assertions only.

Combine actual closed reference and fresh resumed execution evidence when
checking the existing pair totals:128workers/32scripts/three invocation ledgers.
Do not fabricate additional records or mutate the fixture's diagnostic lists.
Retain all artifact-array/RNG/Adam equality, frozen schedules/roles, original
deadlines across resume, config-drift refusal and retention assertions. Keep the
three distinct ledger identities and confirmed physical cleanup.

Fixture failure is a test failure and is not cached as successful data. Scope is
one pytest module/session execution only: no persistent artifacts, cross-run
cache, shared live worker or changed production interface/format. Session fixture
ADR0025 remains separate because its legacy one-update sources are not this
structured two-update workload.

## Verification and compatibility

Inspect the old three runs' input/action/numeric evidence and configuration
differences before reuse; test that the canonical LoadedConfig/resume identity
bind both comparison parents. Existing helpers/callers remain compatible when
the optional reference is absent. All original node IDs and pass/skip conditions
remain. Add only focused regressions needed for wrong-reference admission and
selective/lazy fixture use; do not add tautological pytest caching tests.

Measure the original module before/after on the HOST, retain all35original cases
and include generation/reuse checks in reported costs. Verify original two-update
reference production calls occur once instead of three and that fresh pause and
resume still execute. Validate source inventory unchanged after read-only reuse.
Run relevant structured runtime/helper/source regressions plus project Ruff/mypy;
independent correctness and architecture conformance precede acceptance.

The unrelated whole suite passed3978/247immediately before this task. Bound the
new verification to affected tests plus collection/import/regression coverage;
repeat the whole27-minute suite if new failures, interface expansion or review
findings reveal a broader unresolved concern. Do not claim an unexecuted new
whole-suite result. This is a test execution optimization, not learning-quality
or formal-Source evidence.

## Alternatives and consequences

Keeping three references repeats otherwise identical execution for different
assertions. Sharing live participant workers would change the privacy/failure
boundary and is excluded. Using the previous legacy one-update fixture changes
model/schema/update semantics and is inapplicable. Rewriting completed artifacts
to fit a newly created configuration adds unnecessary identity/lineage mutation.

The selected reuse retains one actual reference and independent fresh resumed
training. It reduces duplicate observations, not the scope of the assertions.
Canonical fixture configuration remains alive for the module. Revisit this
specific reuse if a consumer changes model/device/hand/settings or requires a
fresh independent reference as the subject of its assertion. Actual saved time
and selective-run behavior are acceptance questions, not promised estimates.

## Design review and prior-run evidence

The architecture reviewer is READY for design with no blocking findings. Prior
reference-equivalence evidence and final source/assertion/selection checks remain
implementation acceptance conditions.

Read-only comparison finds identical effective configuration after excluding only
the synthetic binary path, equal code/integration identities and matching actual
generation0/2 weights/resume arrays. The corrected before/after audit also checks
all736arrays in each numeric batch: all704nonpayload arrays match exactly, while
32raw payloads differ by own-card reference IDs, matching usable-hand IDs and the
opaque history cursor. Those references are consistently mapped to the same own
slots and usability features. Raw input/table/source bytes are not claimed equal.
Schedules, outcomes, numerical selections/targets and generation controls match;
all actual source identities remain distinct and are not normalized for reuse.
The canonical LoadedConfig is retained for the new independent pause/resume leg.
Generation1weight/resume arrays were pruned and are not directly compared.
No production state or original artifacts are rewritten.

An initial raw-journal comparison used legacy `input`/`candidates` keys, which
are absent from schema5decisions (`frame`/`table`). Its literal raw equality
claim is withdrawn. Root and independent correctness review directly checked
nonempty payloads and encoded arrays from the retained pytest-148/149 artifacts;
this corrected evidence supports the same implementation and design. The prior
pytest-147 originals expired under normal pytest retention. Preserve the failed
comparison record as historical evidence, not an authority for raw equality.

## Implementation acceptance

Root accepts after both final reviewers retain READY with the corrected evidence.
The module preserves all35original cases and adds seven refusal guards:42PASS.
External wall falls from546.87to455.41seconds, a91.46-second/16.72% reduction
in one measured pair including all reference generation, assertions and cleanup.
Actual reference parents decrease from three to one; fresh pause/resume retains
its independent run ID and two invocations under the same canonical configuration.
Original reference files and numerical state remain intact.

Related/selective checks PASS219/6unchanged CUDA skips. A GRU-only selection
uses zero shared-MLP fixtures and completes its own two-update reference. Whole
collection4232, Ruffsrc/tests/native and mypy157PASS. No new whole-suite runtime
or actual CUDA execution is claimed. Exact commands, corrected evidence and
limits are in [verification](../verification/structured-ppo-test-reference-reuse.md).
