# Production Serde JSON adoption

Status: accepted2026-10-30. Implementation, independent correctness/conformance,
runtime/installed-wheel/corpus/performance gates and main environment rollout PASS.
Design: ADR0012, independent preimplementation architecture READY. User authorized
serde replacement plus an audit of unnecessarily custom generic machinery.
Baseline:580d12a, installed original native extension1a0b1746...; frozen Python
reference source archive at /tmp/gfb-serde-adoption-reference (archive SHA
18aa55d35d7cd2b93464393e97d84880b2452716e17f1fecaf66c71a83faa2a5).

## Audit disposition

| Area | Decision in this unit | Reason |
| --- | --- | --- |
| JSON lexer/parser, Text/Value AST | Replace with serde_json::Value/Map/Number | Benchmark supports ordinary library parsing; no second parser/fallback |
| Per-JSON-string/element BudgetVec leases | Replace with coarse file/document admission | Fine accounting was self-imposed; keep configurable soft budgets |
| Parser duplicate-key index | Remove | Serde Map provides last-key-wins handling |
| Source path open-address table/hash code | Replace with std HashMap | Keep ordered evidence Vec and all identity/digest checks |
| Two custom heapsorts | Replace with std in-place unstable sort | Check cancellation before/after bounded sort; no unpublished late success |
| Canonicalize+parse copying | Replace with ordinary Value clone | Avoid double serialization/parsing under admitted document/control scope |
| Per-node diagnostic profiler and hooks | Remove | Completed diagnostic evidence retained historically; keep coarse timings |
| Integral-float decimal-doubling helper | Replace with std fixed decimal formatting | Test exact Python equality beyond2^53 and f64max |
| Replay UTF16 decoding | Use std conversion | Keep pinned encoding adapter, strict scalar Unicode contract |
| Canonical number spelling | Retain narrow Serde formatter | Existing Python SHA/float semantics are versioned output requirements |
| Truthiness/evidence numeric comparison, closed DTO checks | Retain domain compatibility helpers | Standard JSON parsing does not implement source/learning contracts |
| Numeric buffer and NumPy base-owner leases | Retain | Shared-memory lifetime, retained slices and quota ownership are application-specific |
| Deep-state retirement and entry-stack checks | Retain with Serde depth/GC tests | Removing a parser does not prove recursive Drop safe on small caller stacks |
| Source health/Replay/parent validation and learning separation | Retain | These are the subsystem's required semantics |
| Tiny line framing/endian staging helpers | Retain | Pinned CR/LF/CRLF and typed data representation; no duplicated JSON grammar |

The prior audit correctly identified where old contracts depended on custom code.
Those fine allocation/polling contracts are explicitly revised by ADR0012, not
accepted as permanent reasons to keep generic algorithms bespoke. No unrelated
lifecycle/learning subsystem rewrite is bundled into the Serde migration.

## Dependency record

Pinned serde1.0.229/serde_json1.0.151; only Serde and its required dependency
records were added. Existing crate pins (including syn3.0.9/unicode-ident1.0.27)
remain unchanged. Python/NumPy/Torch/LightGBM versions unchanged. Optional native
package remains separate from baseline. Cargo caches are isolated under /tmp;
public crates.io index refresh preserved existing pins after an older temporary
index copy would otherwise have selected older macro packages. No Simulator
source/environment was changed.

## Frozen implementation and correctness

Final native source SHA35b7c3983c6512cff1a5bd8b78de8d18b217b38925446a2a5a45672a28198812;
loaded release extension SHA9f59b1b2d5658b755546365f74f0cd19946ff5102b39824a439dbdd75f976e9f;
wheel SHA959a07b30b35b83dc368855e55620038dd102fb40749982e0d01c12b03313dc2;
Cargo.lock SHA5fa272b273d397881f3e9a8d9190dae89796052482cf21105d3bc5f4de840091.

Independent final correctness and architecture-conformance reviews READY. Their
health-boundary finding is closed: serde Number::as_f64 filters nonfinite values,
which had incorrectly made standard overflow tokens1e999/-1e999 false in health
flags. Raw standard f64 parsing now preserves Python overflow truthiness, and
conversion failure is treated as present failure evidence. New regressions exercise
actual Parent::open/projection::links gates, zeros, signed zeros, underflow and huge
integers. No source-health gate was weakened. Architecture documentation synchronized.

PASS: whole Python762tests,5opt-in skipped,241.49s; Rust debug34tests, release
focused JSON/health/depth8tests; actual Serde arrays/objects at989..992 levels and
8MiB GC/retirement checks;9,997random finite f64 Python-canonical byte comparisons.
Whole Ruff and mypy59source modules PASS, Rust formatting/diff checks PASS.
The earlier scoped Python sandbox run was interrupted because asyncio self-pipe
sends are denied there; the whole passing suite used approved local execution.
No periodic polling workaround was introduced.

`runs/assessments/serde-adoption-corpus.json`: both complete83,783-row recipes
retain exact original NPZ bytes, under the original2GiB/16MiB settings:

- History:c5c3f3c1ef4068766d81c7bd0cfa7a8957b140ad7ea1f207374fc725dcbcc94d.
- Observation:841c6f885e8d8f052abde2c5db788de761279947e36300e7bc87ff0254e7d0c8.
- Dataset:ab5d4913fda45795d1cd5aa02a4a9c4c3464268298c4d3ea49d56d4909df770f.
- Index83,783rows, row-content574b81ecf6178849c91bea20a9bc0d9906a0348f2c5357825140fc6fc8c618cf;
  even Parquet file SHA matches9bce50db4e8b44058bdc4c1baaed4f7c20da6f29d3f5b8ab792b2ada4f9b51c6.

No estimate tuning or budget increase was needed. Peak managed admission1,605,390,822
bytes includes coarse JSON estimates (JSON estimate peak1,289,911,552bytes), numeric
capacities and operation reservations. These are NOT process RSS or an exact Serde
heap measurement. Read/write copy counters explicitly exclude untracked ordinary
Serde String/Vec/Map/clone allocations and copies; numerical FFI pointer sharing
remains separately verified. Correctness-run elapsed74.44/45.84s and index42.75s
overlap tests and are excluded from paired performance acceptance.

`runs/assessments/serde-adoption-ppo-ingestion.json`: the copied real16-game batch,
under owned test context and2GiB, produces the old feature/GAE/target SHA
7e2629b8398d6988f099f0f3556fb768d3898bd68fa516e7b98c3d9131a9c476
with1/2/4workers. No games or optimizer updates were executed in this check.

Separate installed application/native wheels PASS; all59application Python files
byte-match source. Application wheel SHA
cb34b10930ddd730f111d41cf7a0150357fa151edbcd4879fbf259cad722d55e.
Baseline help/config PASS without native/Torch/LightGBM/PyArrow; installed CPU
MLP/LightGBM inference PASS. Verified local dependency files were reused; Python
dependency versions were not freshly re-resolved. The old extension remained
installed until paired performance finished as a genuine frozen comparator;
it is now archived under the reference's native-site.

## Paired full-preparation performance and rollout

Same frozen history dataset,2GiB/16MiB,workers1/inflight1, numerical dependencies
and observer/workload code. Both variants are release builds. Old native1a0b uses
frozen Python source at580d12a; new native9f59 uses the reviewed current source.
One warmup each followed by3alternating measured pairs; no concurrent builds,
tests, training or inference benchmarks. Every trial has exact dataset/sample/
array/NPZ identities. Records:
`runs/performance/serde-adoption-20261030/{results,equivalence}.json` and
`runs/assessments/serde-adoption-performance.json`.

| Whole command / resource metric | Old custom parser | Serde |
| --- | ---: | ---: |
| Wall seconds,3trials |94.766 /93.777 /94.254|50.809 /50.082 /51.173|
| Median wall seconds |94.254|50.809|
| Median process-tree CPU seconds |94.039|50.701|
| Median sampled per-run peak RSS,MiB |376.10|404.63|
| Median sampled per-run peak PSS,MiB |367.50|395.63|
| Maximum sampled scratch bytes |1,856,287,138|1,856,287,138|
| Native internally tracked scratch peak bytes |1,856,048,056|1,856,048,056|
| Median kernel-accounted read bytes |0|0|
| Median sampled kernel-accounted write bytes |1,943,040,000|1,943,031,808|
| Accepted actions per wall second |888.90|1648.99|
| Amortized pipeline ms per accepted action |1.125|0.606|

Whole wall improves1.855x, a46.09% reduction. RSS increases about28.54MiB; original
2GiB admission succeeds without reservation-policy tuning or budget increase.
CPU is about one occupied core for either implementation; GPU is not used by this
preparation workload. Source data are warm in kernel cache. OS accounting does not
expose WSL host physical I/O. RSS/PSS/storage samples can miss short peaks; output
includes scratch and must not be added to it. The observer is excluded from CPU.

Example trial0 worker timings: source decode62.878->22.276s, feature encoding
6.718->6.652s, proof/staging4.147->2.402s. These counters may nest and are not an
additive partition of full wall or CPU. Improvement includes the changed JSON
representation/admission/copying policy; it is not an isolated lexer-speed claim.
Numeric FFI pointer-sharing remains checked; Serde internal copy bytes are
untracked, so reduced managed-copy counters are not a whole-heap copy comparison.

After measurements, the new wheel was installed into Bots .venv with --no-deps;
its loaded release SHA and parser/resource contracts were rechecked. The original
six formal-fit artifact groups retain all24recorded file identities and small-file
hashes; large original arrays were not rewritten. Simulator remains clean at
ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6. No refit, formal PPO campaign or final test
ran. This demonstrates correctness/resource/performance adoption, not win-rate gain.

Reproduce using the existing observer and preparation workload with the frozen
reference source+archived old native-site versus current source+new wheel overlay;
verify loaded binary hashes before each run and use fresh output directories.
Native/parser/backend upgrades continue to reject ordinary resume of old runs.
