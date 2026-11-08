# ADR0025: Session-generated pytest PPO sources with isolated test copies

Status: accepted; implementation and scoped performance acceptance complete on
2026-11-09 after independent design, correctness and conformance reviews.

## Context and requirements

The owner requested making equivalent generated learning-test data available as
common pytest data. The original `8eb6a7f` whole run generated the legacy
`frozen_sources` workload 17 times: five fixed-data cases, ten native-session
cases, one preparation case and one native fixed-fit case. Eight additional
parallel-parse consumers were skipped before fixture setup. The earlier timing
analysis's 15 constructions describes the first two modules only.

The workload contains three CPU MLP parents at seeds17/29/43, each one update,
and their original64-game final-policy validation populations. Within each seed,
the compared inputs/actions/selections and generation0/1/resume arrays match;
paths, IDs and diagnostics need not be byte-identical across fresh generation.
The sources are explicit synthetic fixtures, not qualified gameplay evidence.

Retain the existing tuple-of-parent-paths/tuple-of-validation-paths interface,
all three seeds, original budgets, current optional-dependency skips, source
readers, corruption assertions and real numerical generation coverage. Support
all existing consumers via normal conftest fixture discovery, without importing
a fixture from a test module. Add no dependencies or production changes.

## Responsibilities and dependency direction

- `tests/conftest.py` registers one lazy session fixture for the canonical
  workload and the function-scoped `frozen_sources` fixture for private copies.
- Pure input construction moves to a small test support module; existing
  `definitions`/`make_input` fixtures remain function-scoped with fresh objects.
  Existing imports of conftest's fixture-loading/public-options helpers retain
  compatibility. The session builder uses the same pure input factory directly,
  avoiding a session-to-function fixture dependency. The synthetic-run support
  imports the extracted pure loader directly, avoiding a reverse dependency on
  conftest.
- Exact `final_state`/`journal_decision` helpers also move from the collected
  journal test into journal test support, with compatibility re-exports. This
  closes the existing synthetic-run support's transitive test-module dependency;
  it does not change candidate generation, command IDs or journal behavior.
- The existing legacy configuration/synthetic collector helpers move from
  `test_ppo_lineage.py` to test support. Re-exports keep existing helper callers
  compatible; the builder does not depend on importing a collected test module.
  Their behavior, including current sampling and numerical calls, is unchanged.
- One test-only shared-source helper owns generation, private copying, explicit
  path rebinding and resealing. Production modules neither import it nor gain a
  migration/cached-source authorization API.

Imports remain lazy enough that collection and unrelated baseline tests do not
load Torch or generate a template. Pytest's current optional-dependency policy
applies only when a consumer requests the fixture.

## Lifecycle, interfaces and data flow

The template is created on first use under `tmp_path_factory.mktemp`, once per
pytest session/process, with the original train/validate calls. It is not stored
in the repository, pytest's persistent cache, or a cross-run artifact directory.
Thus each invocation regenerates against the current code/runtime/fixtures;
there is no persisted compatibility policy or stale cache to reinterpret.

Before inventory capture, template publication or any clone resealing, validate
the untouched generated template once with the ordinary full `build_dataset`
path. Successful train/validate calls and `validate_chain` alone do not compare
the stored update source proof with original projected journals. If the original
template has a chain-valid but stale source proof, construction must fail rather
than allowing clone resealing to repair that generator regression. The original
validation output is private test evidence and is not a substitute for validating
each finished clone.

Each `frozen_sources` request receives a new regular-file copy under its own
`tmp_path`. Always copy, including nominally read-only consumers: this keeps
ownership and lock behavior uniform and prevents later test changes from
mutating a shared original. No hardlinks, shared writable files, symlink source
trees or template paths in returned source bindings. The template is private;
its content inventory is checked so accidental mutation fails visibly.

Copy the six source directories and inert synthetic binary identity. Rebind
only management paths under the known old root, never arbitrary policy inputs,
journal/Replay facts, numeric arrays or reference identities. Preserve original
run/Game IDs and timing/cost evidence within the isolated copy. These IDs may
repeat between separately owned tests, but remain distinct within each dataset.
Recompute configuration hashes when affected by the binary path.

Cloned collections must point to cloned parents/checkpoints. A copied generation
chain must bind the new manifest hashes and the new path-sensitive source proof.
Construction may use two private stages: first establish coherent copied
manifest/update/generation hash links, then run the ordinary offline journal
projection to obtain the relocated source-set proof, then reseal update and
generation links again. Intermediate files are never returned to consumers.
No fake OnlinePPOContext, validator monkeypatch, online authority or production
guard relaxation is permitted. Validate the finished clone with the unchanged
whole-source/fixed-data path before returning it; retain validation output only
inside the private test directory or remove its disposable check manifest.

The helper is deliberately bounded to the currently generated legacy one-update
fixture schema. Unexpected layout/version or missing original evidence fails
explicitly instead of silently rebuilding an invalid source into eligibility.

## Information, failure and concurrency ownership

The synthetic inputs and numeric recipe remain unchanged. Management paths and
source checks stay outside policy features. Native and Python readers continue
to see complete linked original files for each private copy.

The session fixture alone owns template generation; failed generation is not
returned or reused. The function fixture owns its destination; a failed copy,
rebind or validation never yields a partially valid source. Leave diagnostic
temporary outputs available under pytest's normal retention. Child processes
remain owned/reaped by the existing trainer. Separate pytest processes would
each own a template; no cross-process cache or new parallel runner is added.

## Alternatives, consequences and verification

Per-test regeneration preserves isolation but repeats costly identical work.
Sharing the same writable originals would break mutation tests and concurrent
parent locks. Committing/versioning generated datasets adds unnecessary stale
runtime and compatibility obligations. Session generation plus private copies
keeps the current ownership at the cost of copying/resealing/validation time.

Required verification: all existing consumers and lineage/helper regressions;
normal package checks; independent correctness and architecture-conformance
reviews. Add focused tests proving one template generation per session, lazy
optional imports, distinct copied inodes and paths, normal source/array/row
equivalence, and that a mutated copy cannot affect the template or a later copy.
Test corrupted template/failed clone handling where it protects actual behavior.
In particular, retain a regression in which the original template has a stale
source proof but coherent generation hashes: ordinary chain validation may pass,
but template admission must reject before any clone/proof regeneration.

Measure the same affected file selection with durations before/after, including
template creation and every private copy/check; retain the complete populations
and failure assertions. A whole-suite run checks collection/import-order effects
of common conftest registration. No model-quality or source-qualification claim
follows. Copy/reseal cost is an open measured question; if it removes the gain,
revisit this fixture strategy without weakening production validation.

## Design review disposition

Root accepts the architecture review's blocking finding about original-template
proof validation. The prepublication full-source check and chain-valid stale-proof
regression above resolve the specified gap; the reviewer's recheck is READY for
implementation. The private two-stage relocated-chain construction does not
require online authorization and is followed by full final clone validation.

## Implementation acceptance

Root accepts the implemented common fixtures after both final reviews are READY,
104scoped passes/8skips, the strengthened coherent-version rejection check and
3978whole-suite passes/247skips. The whole trace observes one template setup and
24independent fixture requests; the original template remains fully valid after
all tests. All original4217cases retain their statuses (one generated UUIDv1
parameter label varies), with eight new passing regressions.

The unchanged48-pass/8-skip selection falls from219.08to72.22seconds including
template generation, private copies and all validation:67.03% in one measured
pair. Copy/reseal cost leaves a material observed gain. Current fixed-format
support is deliberate: revisit the helper/guards when the legacy fixture recipe
or schema changes, rather than admitting a different workload implicitly.
No cross-run cache, inference math optimization or dependency was added.
Full Ruff has four pre-existing findings in an unrelated untracked user script;
`src/tests/native` Ruff and mypy157 pass. Exact conditions and limitations are in
[verification](../verification/shared-pytest-ppo-sources.md).
