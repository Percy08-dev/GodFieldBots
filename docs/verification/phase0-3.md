# Phase 0–3 verification and acceptance

Date: 2026-10-27 (Asia/Tokyo). The first acceptance unit in the user's plan is
implemented. Independent correctness and architecture-conformance reviews are
**READY**; root accepts the foundation with the limitations below. This is not
acceptance of a clean learning pilot or completion of Phases 4–9.

This records the **initial schema-2 foundation**. The subsequent coverage work
found two defects, strengthened the tests, and introduced schema 3; see the
[coverage follow-up](coverage-followup.md) and ADR0003/0004 for current results
and compatibility. Historical schema-2 artifacts below are retained unchanged
and cannot be silently read by the current schema-3 exporter.

## Final checks

| Check | Result |
| --- | --- |
| `UV_CACHE_DIR=/tmp/gfb-uv-cache uv run pytest` | PASS: 123 passed, 2 opt-in skipped, 5.02s |
| `UV_CACHE_DIR=/tmp/gfb-uv-cache uv run ruff check .` | PASS |
| `UV_CACHE_DIR=/tmp/gfb-uv-cache uv run mypy src` | PASS: 22 source files |
| `UV_CACHE_DIR=/tmp/gfb-uv-cache GFB_RUN_INTEGRATION=1 uv run --no-sync pytest -q -m integration` | PASS: 2 passed, 123 deselected, 1.66s |
| `UV_CACHE_DIR=/tmp/gfb-uv-cache uv build --offline` | PASS: wheel and sdist |
| Independent wheel environment `/tmp/gfb-wheel-verification` | PASS: all seven packages loaded from installed wheel, GIL on, no Torch/LightGBM/PyArrow installed or loaded |
| Wheel/source equality | PASS: all 22 packaged Python files equal final `src/` |
| Final installed-wheel actual baseline smoke | PASS: 2 natural games, replay/journal reconciliation, clean EOF/reap |
| Same schema-2 journals exported twice | PASS: byte-identical 32-row Parquet |
| Failed 12-game run exported as diagnostics | PASS: 176 rows, 0 eligible rows, 24 participant sources; known natural results retained |

The final Ruff/mypy commands were also run after the last fixes with
`uv run --no-sync`; both passed. The ordinary unit suite intentionally skips
real-server integration. The explicit opt-in suite above ran it separately.

Wheel: `dist/godfieldbots-0.1.0-py3-none-any.whl`, SHA-256
`9f312799b2bfdfeb2ab8668ce7cb9e51c8eef534899784f9848feb29729c06cb`.
Installed from the wheel with `uv pip install --python /tmp/gfb-wheel-verification/bin/python`,
then executed from `/tmp` using a copied TestServer binary and no `source_dir`.
Its manifest truthfully records unknown installed build/source revisions.

## Environment and contract

- CPython 3.14.9, `Py_GIL_DISABLED=0`, `sys._is_gil_enabled()=true`.
- Simulator source HEAD exactly `44a26e3a16b808e826deb332fa72acfc5037d3ef`.
  Existing README-only modifications were preserved; no simulator source,
  engine, build output or configuration was changed by this task.
- Prebuilt TestServer SHA-256:
  `d9c774d214e0e732fed040e7ef2e386979fc1f42e225429c1dbcfc642b49831f`.
  **Its build revision is unverified.** Source HEAD does not establish binary identity.
- `/v1`, `true-match-1v1-v1`, Observation `"1"`, Action metadata `1`,
  TestServer info `1`, catalog `public-ja-v1` schema `1`, 296 definitions.
  Catalog canonical hash equals the normative pinned hash.
- Isolated CPU compatibility smoke passed: NumPy 2.5.4 arithmetic,
  PyArrow 26.0.0 roundtrip, native LightGBM 4.7.0 train/predict,
  Torch 2.14.1+cpu forward/backward. No CUDA/NVIDIA packages in that environment.
  Native LightGBM is the planned adapter API; sklearn wrapper is unavailable
  without an unselected dependency. GPU execution was NOT RUN.

Dependency compatibility command, against isolated copies of project TOML/lock:

```console
UV_CACHE_DIR=/tmp/gfb-uv-cache uv sync --project /tmp/gfb-dependency-check --python /home/linuxbrew/.linuxbrew/bin/python3.14 --frozen --no-install-project --no-dev --extra data --extra lightgbm --extra torch
/tmp/gfb-dependency-check/.venv/bin/python /tmp/gfb-dependency-check/smoke.py
```

Network downloads and localhost listeners required sandbox escalation. Ordinary
attempts failed due to environment network restrictions; authorized retries
passed. Some earlier worker tests hung because of unbounded shutdown writes,
not solely the sandbox; they were interrupted, fixed and independently rechecked.

## Final artifacts and measured smoke values

Clean installed-wheel run:
`runs/wheel/wheel-no-ml-c8a902b5-a15/{manifest,report}.json`.

- 2 planned/natural games, 32 accepted decisions, no failures.
- CLI total elapsed 1.494s, including final persistence. Persisted report snapshot
  elapsed 1.487s; the final-write difference is not a hard real-time guarantee.
- Mean 181.44 candidates and 245.72 search attempts per recorded decision.
- Search latency p50 0.553ms, p95 69.56ms; report throughput 21.52 decisions/s.
- Recorded artifact size 1,349,836 bytes. Peak memory is explicitly null.
- Derived rows 32/eligible 32; deterministic Parquet SHA-256
  `c7560a5075b848522a88609722c560a5e87d3f728209d6a9fb4f995a49c7d23b`.
  Files: `/tmp/gfb-final-export/clean-{a,b}.parquet` with provenance sidecars.

These are bounded functional smoke measurements, not model-quality comparisons.
Separate policy-only inference latency, peak memory and ML model sizes are not
measured by this baseline unit. Search timing includes public candidate work.

Expanded fixed evaluation (repeated after final schema changes):
`runs/final-verification/baseline-validation-a760b624-dd0/{manifest,report}.json`.

```console
UV_CACHE_DIR=/tmp/gfb-uv-cache uv run --no-sync gf-bots evaluate --config configs/experiments/baseline.toml --set run.wall_seconds=180 --set run.admission_seconds=90 --set run.execution_seconds=120 --set 'run.output_dir="runs/final-verification"'
```

Two GameSeeds × three policy sampling seeds × swapped seats = 12 planned games.
8 natural wins for heuristic, 4 rejections, CLI elapsed 8.125s. Comparison invalid,
run incomplete; no strength claim or clean-pilot claim is made from the 8 wins.
All 24 participant sources export with 0 eligible rows, including naturally
finished games. Sixteen natural participant endings represent the same 8 games.
Diagnostic Parquet: `/tmp/gfb-final-export/incomplete.parquet` and its sidecar.

[Separate simulator/prebuilt investigation](../issues/0001-public-observation-legality-investigation.md)
records public armor/Sun Amulet identities rejected as a weapon during discard.
No hidden dumps/hints were requested and no policy rule was inferred from rejection.
The failed development-schema-1 run
`runs/verification/baseline-validation-f86ce88e0436` is retained, not reinterpreted
by the schema-2 reader. Earlier smoke artifacts are likewise retained.

## Phase gates

| Phase | Implemented and verified | Gate / remaining limit |
| --- | --- | --- |
| 0 | Strict TOML resolution/hash/path provenance, package/extras, CPU/wheel | PASS for baseline foundation; training presets rejected until an adapter exists |
| 1 | HTTP/info/catalog validation, repeat-seed group ownership, exact retries, worker isolation, events/replay and cleanup | PASS for tested contract; prebuilt build identity remains unknown; live external-URL mode tested via mocks, not a separate deployed server |
| 2 | Enumerate/beam reference agreement, lazy budgets/order/instance/basic retention, Random/heuristic, measured baseline play | Algorithm tests PASS; broader candidate-quality validation is not clean (4/12 rejections), so formal comparison and learning prerequisites remain unmet |
| 3 | Schema 2 journal/manifest linkage, corruption rejection, deterministic Parquet, seed/ancestry split conflicts, paired stats and independent health | PASS for foundation; incomplete comparisons and failed-run data eligibility are reported explicitly |

The learning pilot of 64 games × three policy seeds was **NOT RUN**. The observed
non-natural games already leave the learning gate unfulfilled; successful games
were not substituted. LightGBM/MLP PPO/GRU PPO/Transformer PPO adapters, training,
checkpoints/resume, model selection and inference distribution (Phases 4–9) are
**NOT IMPLEMENTED** in this first acceptance unit. Optional-library smoke results
do not imply these features exist.

## Independent reviews and accepted limits

`bot_architecture_reviewer`: initial design and amendments READY; final conformance
READY. `bot_reviewer`: final correctness READY; last affected budget/journal suite
36 passed in 2.68s, original stale-manifest reproduction now exports 0 eligible rows.
All blocking findings were fixed, not silently accepted as debt. Reviews covered
numeric widths/Aura metadata, search termination, inherited child argv/environment,
retry/gate poisoning, outcome-preserving cancellation, double disk failure,
linked-report and manifest integrity, whole-run eligibility, split provenance,
Replay Game ID, and group-health propagation.

`bot_advisor` was invoked after repeated deadline repair cycles. Root accepted
its bounded recommendation: zero ordinary worker grace on abort, independent
report/manifest correction attempts, and hash/status verification of the linked
report before data eligibility. No new framework/process layer was required.

Accepted debt is recorded in architecture/ADR0002: cooperative deadlines plus
bounded emergency reap do not guarantee hard real-time OS/file-I/O completion.
Any detected overrun returns failure. Either durable invalidation makes export
ineligible. If **both** correction writes fail, artifacts can remain stale; the
caller must quarantine that failed directory. Revisit for normal-smoke overrun,
nonlocal filesystems, strict hard deadlines or authoritative recovery after dual
persistence failure. Worker framing has a fixed 64 MiB protocol cap; overflow is
an explicit infrastructure failure. No full-engine legal-action coverage or
seed-only guarantee of deterministic future training is claimed.

No commit or publication was performed. Generated `runs/` and `dist/` artifacts
are local and ignored by Git. Existing uncommitted work was preserved.
