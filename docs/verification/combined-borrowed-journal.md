# Combined mimalloc and borrowed-Journal practicality

Accepted 2026-10-30 under
[ADR0017](../adr/0017-combined-borrowed-journal-practicality.md). Independent
pre-implementation architecture review is READY. This unit determines the
combination's usability; ADR0016's separate effects are not combined predictions.
No production default/main installation, game generation, model fitting, PPO
update or simulator modification is authorized as an automatic side effect.

Baseline9704f93 and original installed native32e3ad68... remain the reference.
Existing corpus/NPZ/index/saved-PPO proofs and original six fits must remain intact.
An opt-in `preprocessing.reader="typed-borrowed-v1"` candidate and optional
`allocator-mimalloc` wheel feature are available; legacy stays default.

## Required evidence

- Combined standalone System/mimalloc x Value/borrowed comparison with separate
  allocator/binary/source proofs and full-corpus canonical oracles.
- Typed Journal semantic validation, source/Replay reconciliation and direct
  typed feature traversal, with no ordinary input/action conversion back to Value.
- Scoped raw-buffer ownership across existing window/join ordering. Journal parse
  becomes coordinator-local; this scheduling tradeoff must be measured explicitly.
- Exact full-corpus float64 arrays for both recipes, MLP float32 transforms,
  sample/source/dataset/index identities and saved16-game all-candidate PPO inputs.
- Shared Config budgets, cancellation/failure cleanup, retained NumPy views,
  source mutation refusal, old-wheel legacy compatibility and early capability
  rejection; no resume identity bypass or failed-parent data reuse.
- Complete-path System/mimalloc x legacy/typed measurements at1/1and4/4, warmup
  plus five rotating trials, no concurrent heavy experiment work, CPU/RSS/PSS,
  kernel RSS peaks, post-release memory, storage/I/O and experiment costs.

## Design decisions

Native owns the pure DTO schema; the standalone harness includes only that module
directly and retains independent AP/no-AP and allocator selection. The production
package does not import benchmark code. Selected-reader provenance is separate
from immutable source/dataset hashes. No capability is advertised until the path
is wired. Runtime counters must prove typed Journal work actually occurred.

Readiness requires independent correctness/conformance review and executed corpus/
failure/ownership/measurement gates. Those gates and the complete measurements
below now pass: the combination is practically usable as an explicit opt-in
preprocessing candidate in the verified environment. Main/default adoption was
not performed.

## Combined standalone component evidence

Milestone1 checks:39targeted Python tests,12Rust tests each System/mimalloc,
native library24tests plus the new Reader test, expected counter+mimalloc build
refusal, Ruff/mypy/format checks. Existing legacy-wheel lifecycle/cleanup checks
pass16tests after their fixtures explicitly select the Rust backend. The native
typed capability remains unadvertised until its actual route is implemented.

System-C and mimalloc-C each passed all13,568files/534,640records with0newrejections
and original canonical records hash`c05bfca5...c7d90c`. Both correct oracles bind
actual allocator/AP/source/binary identity and completed cleanup owners.

All24component commands succeeded (fourwarmups, five measured trials per cell),
with no concurrent builds/tests/edits. Full frozen source snapshot105files is
retained under`runs/assessments/combined-borrowed-20261030/component-source-snapshot`
so subsequent native integration does not lose this experiment's code reference.

| Component | Median whole wall s | Kernel peak RSS maximum MiB |
| --- | ---: | ---: |
| System + Value |16.670|49.4|
| System + borrowed |10.954|22.6|
| mimalloc + Value |12.565|58.9|
| mimalloc + borrowed |10.350|38.8|

The combination reduces this standalone command37.91% versus System+Value,
17.63% versus mimalloc+Value and5.51% versus System+borrowed. The latter costs
additional peak/retained resident memory: post-release RSS is about28.46versus
16.40MiB. Kernel peaks in the table are the maximum of five individual process
high-water observations; mimalloc+Value's median is58.73MiB (not58.9). No combined
native/feature-pipeline speed or usability conclusion is established by this table.
Independent metadata/arithmetic review PASS. Raw oracles/results, summaries and
`component-assessment.json` are under the same experiment root.

## Native candidate v1: executed correctness gates

The first opt-in native route compiled and passed all native unit/integration/
documentation targets:61tests,2ignored, plus54targeted Python checks, Ruff,
mypy64files and both Rust format checks. Actual multi-episode2/4/8worker overlap,
cancellation, queued raw failure and proof/representation guards are exercised.

Both recipes over the full83,783learner rows match original NPZ bytes for System
and mimalloc typed reader: history`c5c3f3c1...cbcc94d`, observation`841c6f885...5d0c8`.
Reader counters prove13,568typedfiles/534,640records and0legacyJournal files.
Dataset/source/sample identities remain unchanged. The new typed Parquet index
matches original rows83783/content`574b81ec...`/file`9bce50db...` exactly.
The initial attempted concurrent full-corpus runs collided with the existing
exclusive parent-run locks; those three coordination failures are retained,
excluded from candidate/performance evidence, and repeated sequentially. Source
ownership checks were not weakened. No correctness elapsed time is a speed trial.

The explicit current-v2 source
`runs/ppo-comparison-v2/train-seed17/collections/batch-000000` supplies the saved
16-game PPO reference, actual NPZ`cbb6169a4a271cdf115e9f68a6d98620adb5d214ac82757802ac403dc86043af`.
Diagnostic generation-zero clones pass ordinary ownership/lineage guards and
remain running/zero-update, so they cannot authorize complete-parent offline use.
Original immutable records/artifacts remain unchanged; only documented copied
manifest/checkpoint paths are relinked. No normal resume or optimizer ran.
At workers1/4, Python/nativelegacy/nativetyped outputs match all eight saved
arrays: actor5671x690f32,state224x373f32,offsets225i64,selected224i64,and224element
old_logprob/old_value/advantage/targetf32. Eight same-policy-ID episodes exercise
both learner seats. Exact source, file and array identities are retained in
`current-v2-ppo-reference.json` and`candidate-v1-ppo-{1,4}/result.json`.

Separate installed app/native wheels pass optional-free baseline startup and
actual MI/typed capability/reader-identity checks outside the source tree.
The v1 full Python run is817PASS,5SKIP,1FAIL: an existing plan test incorrectly
assumed a global temporary reference checkout. It was repaired to create and
explicitly select its own reference fixture; no source-probe bypass was added.

A tiny strict-wire differential found v1's derived Outcome accepts
`{"draw":null}` while legacy accepts only string`"draw"` or a win map. This is
an explicit rejection-gate defect, not a current-corpus numerical mismatch.
It was fixed with a minimal standard-Serde domain visitor and regression tests
before final candidate verification or primary timing.

## Final candidate: pre-measurement acceptance

Final native source SHA256:
`37cae919a0291fb2fbbb28738d0e13a206f7f060887bf2bfe7a6658a465770ca`.
System wheel extension SHA256:
`d1af0c3ca5c14ddeac0a2c1cefdbcc37e78239626a17cb8cd480deafeb01c0a0`;
mimalloc extension:
`bc45545822f06e92659a1c2f65543cbc5237ca12ba8352d028f51f876369a21a`.
Both keep parser/resource/encoder/ABI contracts and report the actual allocator
and supported reader contract. Main installed32e3... remains unchanged.

All final current-data correctness gates PASS: typed System/mimalloc x both
recipes retain original archive/sample/source/dataset identities; current legacy
System/mimalloc history archives also retain the original hash. Final index
bytes/content/rows match the original; final PPO clones at1/4retain every saved
array exactly, with immutable original bytes unchanged.

Final actual mimalloc-candidate full Python suite:818PASS,5opt-in SKIP,303.18s.
Explicit public session tests:11PASS,118.84s, covering both recipes/readers,
reader proof/path counters, source invalidation, summary/index and actual NumPy
readonly slices surviving close/GC. Native all-target tests:64PASS,2ignored;
standalone APSystem/no-AP each12PASS. Ruff, mypy63files, Rust format checks and
uv lock consistency PASS. Installed optional-free baseline and final candidate
allocator/capability/selected-reader checks PASS outside the source tree.

New rejection tests cover draw objects, malformed/null/duplicate/extra win maps,
escaped win keys and all result positions. The document-quota fixture now reaches
actual raw/document admission after a prior episode is queued and returns all
leases to zero. Actual typed worker overlap/cancellation uses2/4/8distinct tasks.

Independent correctness and architecture source/evidence reviews are READY.
Eight actual-path cells (System/mimalloc x legacy/typed x1/4) completed one warmup
and five rotating trials. Current System+legacy is the control; historical
results are not substituted. Final quantitative evidence follows.

## Final complete-path measurements and practicality verdict

The actual-path matrix used clean implementation `34727000f89548aaff39559a28b3fdd3a1d7a5dc`
with 84 frozen code files. All 48 commands succeeded: eight warmups and five
measured trials in each of eight cells. Every result has the expected original
NPZ/dataset/source/sample identity, actual selected reader and exact Journal path
counts, pinned loaded binary, and joined owner without cancellation or expiry.
All original source/model/binary/code checks passed at matrix end.

Same Python 3.14.10/GIL, NumPy 2.5.4, Rust 1.99.0, Linux/WSL2, pinned Simulator/
catalog v2, complete 83,783-row history preparation, 2,048 MiB shared managed
budget, cache 512 MiB, chunk 16 MiB, matching worker/in-flight counts, thread
environment 1 and release opt-level 3. No concurrent builds, tests, profiler,
source edits or heavy experiment work. Counters/profiles are separate from timing.

Wall/CPU/post-release columns are medians of five trials. Kernel RSS is the
**maximum** `wait4` high-water value across the five trials, for the largest
individual process; it is not summed simultaneous tree RSS. Sampled tree RSS,
PSS, all observations/ranges/IQRs and ownership proofs remain in the raw artifacts.

| Allocator / reader | Workers | Whole wall s | Process CPU s | Kernel peak RSS max MiB | Post-GC RSS median MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| System / legacy |1|46.183|46.041|412.2|381.1|
| System / typed |1|31.109|30.977|391.1|363.9|
| mimalloc / legacy |1|32.215|32.055|359.8|299.5|
| mimalloc / typed |1|25.337|25.175|349.1|298.8|
| System / legacy |4|30.558|32.919|589.5|388.7|
| System / typed |4|26.369|25.160|417.2|367.4|
| mimalloc / legacy |4|27.387|28.390|413.7|297.3|
| mimalloc / typed |4|24.030|22.794|368.0|297.6|

Compared with current mimalloc+legacy, the combination reduces whole wall time
21.35% at one worker and 12.26% at four. Compared with System+legacy it reduces
45.14%/21.36%. The additional allocator benefit over System+typed is 18.55%/8.87%
in this complete path, distinct from the standalone Journal component's 5.51%.
Do not multiply separate-stage ratios or mix historical controls into this matrix.

Five-trial wall min–max / inclusive IQR, seconds:

| Allocator / reader / workers | Min–max | IQR |
| --- | ---: | ---: |
| System / legacy / 1 |44.991–47.455|0.325|
| System / typed / 1 |30.999–31.479|0.139|
| mimalloc / legacy / 1 |32.159–33.386|0.162|
| mimalloc / typed / 1 |25.112–25.903|0.498|
| System / legacy / 4 |30.196–30.626|0.339|
| System / typed / 4 |25.887–26.548|0.244|
| mimalloc / legacy / 4 |26.960–27.955|0.222|
| mimalloc / typed / 4 |23.604–24.317|0.442|

Four workers reduce the combination's whole wall time by 5.16% versus one
(about 1.31 s), with
kernel peak 368.0 vs 349.1 MiB. This is consistent with the deliberately serial
Journal parse/validation/feature-spooling consumer. Configured decoder overlap
does not prove utilization of four full CPU cores. The opt-in preset retains one
worker; four is an explicit speed/resource choice, not a silent default change.

The two-GiB setting is a shared application admission budget, not measured RSS
or a hard OS quota. Post-release snapshots include interpreter/imports, use no
forced purge and do not prove long-term allocator retention or a live-object
leak. Scratch/output categories overlap, and OS byte counters are not WSL host
physical-device traffic. No GPU workload or end-to-end training/win-rate gain is
claimed by this preprocessing comparison.

## Reproduction, costs and supported use

Final local evidence lives under `runs/assessments/combined-borrowed-20261030/`:
`native-final/{system,mimalloc}`, `native-measured-freeze`, `native-primary`,
`native-summary.json`, `native-assessment.json`, `final-corpus`,
`final-legacy-corpus`, `final-ppo-{1,4}`, `final-index-mi.parquet` and its sidecar.
The independent component evidence/source snapshot remains separately retained.
Root runtime logs additionally record test/check/package outcomes. The installed
app wheel contains all 59 current Python source files byte-for-byte; its SHA256 is
`0cc3605e5854f08217c16fa820a7964440ddb38e1684f5523aa8941634dfbab6`.

The 48 primary commands cost 1,461.33 cumulative command-wall seconds (24.36 min)
and 1,463.89 CPU seconds (24.40 min), including warmups; these are not total
elapsed session cost. OS write-byte lower bounds sum 93,265,346,560 bytes
(86.86 GiB), including repeated identical temporary archives/staging. These bytes
were not retained data generation. Experiment-owned retained files/builds/source
snapshot were about 1.83 GiB logical / 1.43 GiB inode-deduplicated allocated before
final report writes, excluding original data and shared tool caches. Correctness
runs, failed lock-coordination attempts, discarded v1 builds, test suites and
setup costs are separate evidence; not every setup/edit/check interval was timed.

Use a wheel built with Cargo feature `allocator-mimalloc` and explicitly select
`preprocessing.reader="typed-borrowed-v1"`, or the provided standalone preset
`configs/preprocessing/native-borrowed.toml`. Dataset CLI selects that preset;
training can set the same reader on its existing Rust preprocessing configuration.
The main environment has **not** been installed/switched, and all existing reader
defaults remain legacy. An old/incapable native wheel rejects the new selector
before source/collection work. The reader is strict for the pinned generated
schema; legacy remains available for historical permissive inputs. Ordinary
resume across changed code/binary/reader/environment identities is still refused.

**Verdict:** practically usable as an opt-in preprocessing candidate in the
verified offline/fixed-data and PPO-ingestion environment. Numerical equivalence,
source/reader provenance, scoped lifetimes, public NumPy ownership, mutation/
failure/cancellation cleanup, actual combined speed and peak memory are verified.
Broader platforms/schemas, future recipe changes, live participant-worker changes,
full-core scaling, new training-quality evidence and production default adoption
remain outside this verdict. Independent correctness and architecture measurement
reviews are READY; no unresolved blocker remains.
