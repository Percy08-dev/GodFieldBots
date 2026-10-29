# Native decoder breakdown

Status: diagnostic implementation and independent correctness/conformance READY;
release measurements complete on2026-10-30. Production extension unchanged.
Source baseline:6f28a1a; production loaded extension SHA
1a0b1746b5582df0f2403230a6e406efadd28a64c3babbd354dfa20437e7f34f,
build profile=release. No learning, source eligibility, feature or simulator change.

## Measurement boundary

The existing source_decode timer includes journal record framing, source encoding
validation/transcoding, and construction of owned recursive Rust json::Value,
Text and BudgetVec structures. It does not mean Python dictionary construction.
Semantic journal/Replay validation and feature extraction have separate timers.
Syntax recognition and tree construction are interleaved in one parser; do not
claim a clean two-pass parser/builder timing split or infer sub-times by guess.

For example, `{"hp":100}` becomes a Rust `Value::Object` containing an owned
`Text("hp")` key and `Value::Int(Text("100"))`. Objects and arrays use charged
vectors; strings and integer representations own byte buffers. Keeping integer
text preserves arbitrary-precision input semantics before range validation.
Object parsing also builds a temporary key index to implement duplicate-key
last-wins semantics. These structures precede numerical feature arrays. Strings
in diagnostic counts include object keys; decode counts are attempts, not unique
source paths. Failed attempts may contribute work counts.

## Scoped diagnostic implementation

Add an optional Cargo decode-profile feature, off in ordinary builds, no new crate.
All detailed hot-path hooks compile away without that feature. Keep the installed
production extension and all existing source/fit artifacts untouched. Build a
release diagnostic wheel in an isolated directory/venv; record loaded binary,
source, Cargo feature and release identities explicitly. No training/resume uses it.

Profile source decode by source kind (journal, replay, other JSON), count input
bytes and files for all source-decode operations. Use deterministic file-level
sampling for fine timers (target1/64 for each kind with explicit actual sampled
counts/bytes; sample entire journal files, not individual records). Preserve all
ordinary decoding, budgets, cancellations, errors and results. Do not disable
checks or change data structures to measure them.

Within sampled decode calls, accumulate exclusive elapsed categories: encoding
validation/transcoding; journal line framing; string content/escape handling;
number parsing/storage; object key hashing/duplicate lookup/index insertion;
array/object/record container insertion/growth; budget admission/release;
allocation/reallocation/copy; deadline/cancellation checks; remaining parser
traversal/token dispatch. Nested subwork is charged once, with remaining time
explicit. Fixed-size thread-local state, no per-node logs or unbounded profiling
object graph. Aggregate at decode boundaries; keep profiling counters separate
from managed production capacity. Report sampled node/string/number/container,
allocation/copy and control-check counts where practical.

Use a fixed hash of source kind plus canonical path for sampling, independent of
arrival order, RNG or worker completion. Preserve coarse all-call totals; detailed
exclusive buckets are sampled observations only. TLS borrows must be short and
RAII must restore the enclosing category on error/cancel/panic. Capture is fixed
for the operation/process, diagnostic-only; build_info identifies the Cargo feature.
The residual means unclassified parser/runtime/instrumentation time, not syntax.

Detailed timing is diagnostic and perturbs execution. Compare the same isolated
release build with diagnostic capture off/on and frozen full preparation (1worker,
2GiB/16MiB). Verify unchanged complete output hashes, record full wall/CPU/memory/
storage and sampling coverage. Report observed sampled exclusive times and their
scope; do not present them as an exact partition of the prior62.87s. Sampling or
instrumentation overhead that prevents useful inference must be disclosed.

## Verification and acceptance

Implementer owns native instrumentation and focused tests. Root owns design/docs,
isolated execution/evidence and commits. Independent correctness and architecture
reviews check timer nesting/accounting, sampling, disabled path, identities and
unchanged parsing/resource/cancel semantics. Run scoped Rust/native semantic tests,
release baseline and instrumented full-corpus output equality. No refit, live game,
formal PPO campaign or full unrelated test rerun is needed for diagnostic scope.

## Frozen build and checks

Diagnostic wheel SHA:
`8d929d0c89f966c8fb0402bcaf7362eed7c337b013c025d324170eaf309d1eaf`.
Loaded diagnostic extension SHA:
`a729837c8544ef636a4dac57459429ad08be5b5a02701d46686af853bbb4f03d`;
native source SHA:
`eedef5d8201907055bd072442bc3e9ebb23f3193aeaef84f0d73b1b00b60ded4`.
Build identity reports release, cargo_features=decode-profile, capture=off/on,
CPython3.14/GIL, same Rust1.99.0 and unchanged Cargo.lock digest. The wheel was
extracted to `/tmp/gfb-decode-profile-site`; a disposable interpreter reused
verified dependency files via .pth. It was not installed into the main environment.
The original production extension still hashes to1a0b1746... after the experiment.

PASS: final feature-on debug Rust37tests; ordinary debug Rust30tests;
latest diagnostic integration4tests with capture-off/debug and capture-on/release;
isolated release Python native-feature/source-binding26tests; cargo fmt and diff
whitespace checks. Commands use env-uVIRTUAL_ENV, an explicit CPython3.14 and
isolated CARGO_HOME/CARGO_TARGET_DIR. No Simulator source/environment changes.

Known baseline test exception: full feature-on release Rust run has35PASS/1FAIL
before the final added integration case. The unchanged
`native/tests/validation.rs:132` cancellation test sleeps1ms before cancelling a
10,000-event walk; the release walk can complete first. Running that same test
with the feature OFF in release reproduces the failure. The final selected16MiB
parse cancellation regression passes in release. Do not call the full release
suite PASS. This pre-existing test synchronization debt does not block isolated
diagnostics; revisit with deterministic synchronization when maintaining that
validation test. No deadline/check was weakened to pass it.

## Actual corpus and calibration

All four complete history preparations reproduce the same83,783-row arrays,
NPZ SHA `c5c3f3c1ef4068766d81c7bd0cfa7a8957b140ad7ea1f207374fc725dcbcc94d`,
dataset digest and sample ordering. Settings:1worker/inflight1,2GiB/16MiB,
OMP/MKL/OpenBLAS threads1. No concurrent build/test/training during measurement.
Each command includes preparation, final archive/hash checks and cleanup; no fit.
Evidence: `runs/assessments/native-decode-profile.json` and
`runs/performance/decode-profile-20261030/*/{measurement,workload}.json`.

| Release command | Whole wall,s | Process-tree CPU,s | SourceDecode,s |
| --- | ---: | ---: | ---: |
| Original production control |121.407|98.975|63.951|
| Diagnostic capture off, first |109.532|109.429|76.724|
| Diagnostic capture on |114.179|113.852|80.666|
| Diagnostic capture off, second |110.017|109.891|77.028|

The production control has3,346,337,792kernel-accounted read bytes, with read/hash
25.354s; its slower whole wall is chiefly a cold-input effect. Its decode time is
close to the earlier62.866s. Later diagnostic runs have0to2,048,000read bytes.
Do not use121.407s versus114.179s as an optimization result. Kernel I/O is OS
accounting, not WSL host physical disk traffic.

Capture-on adds4.01% whole wall and3.82% CPU relative to the two off controls'
mean. However, even capture-off retains inactive hot-path hook branches in this
diagnostic build and reports about76.9s of decode time. Its source/build identity
differs from production, so this is diagnostic-build overhead evidence rather
than an exact attribution of the difference to individual hooks. The ordinary
feature-off build compiles those hooks away. The production speed result is unchanged.
Diagnostic sampled peak RSS is394,317,824to394,678,272bytes, scratch maximum
1,856,287,138bytes and kernel write counters about1.943GB; observations are sampled
and do not guarantee hard resource limits. Full resource counters remain in JSON.

## Source-kind breakdown

The following are the mean of two capture-off controls in the diagnostic build,
not a numerical partition of the original production63.951s. File counts are
decode attempts; source paths can be decoded more than once.

| Source kind | Decode attempts | Input bytes | Decode wall,s |
| --- | ---: | ---: | ---: |
| Participant journals |13,568|2,267,302,084|56.203|
| Player Replays |13,568|817,021,446|20.028|
| Other JSON controls/manifests |2,082|40,678,063|0.618|

Read/hash is excluded. Per-kind FileGuard starts after sampling hash selection,
while SourceDecode starts before it and ends after guard aggregation; these
envelopes differ slightly. No nested sampling suppression, missing canonical
sampling paths or failed decodes occurred in the measured corpus.

## Fine sampled breakdown and its limits

Deterministic file-level cohort:208journal+211Replay+29other JSON=448file-level
decode attempts (not a claim of448unique paths),
52,146,720bytes (49.731MiB, about1.67% of decoded input bytes). The same counts
and bytes match both off controls. Cohort wall is1.282627s averaged across off
controls and4.928139s with fine capture, a3.842x increase. This large local
perturbation makes extrapolation to exact uninstrumented production sub-times
invalid, even though the effect on the complete command is only4.01%.

| Exclusive sampled bucket | Seconds, capture-on | What is timed |
| --- | ---: | --- |
| Budget admission/release |1.041056|Budget::reserve and Lease::drop, including zero reservations|
| Allocation/copy |0.858249|managed vector allocation/growth and slice extension|
| Unclassified parser/runtime/instrumentation |0.725940|traversal, dispatch and remaining hook/runtime costs|
| Deadline/cancellation checks |0.635821|Control::check and its clock/check work|
| String handling |0.632860|string scan, escapes and content construction, excluding nested buckets|
| Key/index work |0.564113|hash, duplicate lookup and temporary key-index insertion|
| Container work |0.285527|array/object/record reserve and insertion, excluding child parsing|
| Numbers |0.169476|numeric token parsing, conversion and retained integer text|
| Journal record framing |0.008350|record-boundary scan and line-end advancement|
| Encoding validation/transcoding |0.006747|UTF validation and supported Replay source encoding handling|
| Total sampled envelope |4.928139|exclusive buckets sum to the sampled file envelope|

These are elapsed times with instrumentation, not CPU samples. Short Budget/Control
operations are particularly sensitive to timer overhead; bucket percentages must
not be used as an exact production optimization ranking. Syntax recognition is
interleaved with construction, and the residual is deliberately not called syntax.

Observed work counts for that49.731MiB cohort:3,722,530value-node attempts;
3,565,764string calls (including keys);1,276,385number calls;1,234,331array/object
openings;11,597,286allocation attempts after admission;19,482,700budget reservations;
13,251,826control checks;120,382,719tracked copied bytes. Copy bytes cover managed
growth and slice extension, not every assignment or allocator-internal operation.
The counts show substantial small owned-buffer construction and bookkeeping;
they do not give its exact share of the ordinary production64seconds.

## Reproduction

Build the optional wheel in an isolated target directory and use an overlay site
directory. Keep the production extension unchanged. Verify build_info release,
cargo_features and loaded extension path/hash before reading the frozen corpus.
Run the existing benchmark observer/workload three times with identical settings
and GFB_NATIVE_DECODE_PROFILE set to0,1,0 respectively:

```sh
env -u VIRTUAL_ENV PYTHONPATH=/tmp/gfb-decode-profile-site:$PWD/src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  GFB_NATIVE_DECODE_PROFILE=0 /tmp/gfb-decode-profile-env/bin/python \
  tools/benchmark_performance.py --output runs/performance/decode-new-off --seconds 1800 -- \
  /tmp/gfb-decode-profile-env/bin/python tools/benchmark_workload.py preparation \
  --variant rust --encoder history --memory-mib 2048 --chunk-mib 16 \
  --workers 1 --max-in-flight 1 --seconds 1800 --source src \
  --native-sha256 a729837c8544ef636a4dac57459429ad08be5b5a02701d46686af853bbb4f03d
```

Use fresh output paths and the actual newly built binary digest for a later build.
Record per-kind cohort coverage, exact outputs and off/on overhead; avoid claims
of speedup or exact production sub-times from this diagnostic experiment.
