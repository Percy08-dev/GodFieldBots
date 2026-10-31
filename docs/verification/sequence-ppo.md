# GRU／Transformer PPO: implementation evidence

Design: [ADR0022](../adr/0022-gru-transformer-ppo-plan.md), user-approved full
own-decision history, bounded hidden/KV inference cache and configurable TBPTT.
This document distinguishes implementation evidence from the historical MLP runs.

## Status

- Revised on-disk design: architecture fidelity READY, with mandatory actual
  learner-device split parity for CPU presets as well as CUDA.
- GRU and Transformer core: independent source correctness/conformance READY;
  component/native/CUDA, full regression and installed-wheel checks PASS.
- Real two-update/explicit-resume smokes: PASS for both models,128natural games.
- Fixed-input performance comparison:60commands PASS; final independent evidence
  correctness/conformance audits READY. Root accepts ADR0022 A–E on2026-11-01.
- Formal nine-run comparison and final-test evaluation: outside this unit.

Root owns requirements, presets, docs, integration and acceptance. One implementer
owns scoped production sources/tests. Independent correctness and architecture
reviews remain required before each accepted milestone. No Simulator edits.

## Initial environment check

Simulator HEADffcffa0966dbf6d7ce1d982f4ee6444e291d42d6 and worktree match the
previously verified source. §21.1 full-file SHA342a8d59…533d2d62, TestServer
binary84b7e900…9725204b, server config905511d9…89a32c1 match recorded evidence.
Python3.14.10/GIL, NumPy2.5.4, Torch distribution2.14.1+cu132. Metadata inspection
does not prove actual CUDA numerical execution. Read-only NVML outside the
restricted sandbox reports RTX5070Ti/16303MiB; inside the sandbox access is denied.
Initial installed native release/mimalloc is sourcef98b0312/ELF38ad8a23.
The historical isolated wheel environments under /tmp are no longer present;
new isolated acceptance environments must be created for this implementation.

The completed checks below supersede this initial environment snapshot. Functional
execution time and fixed-input comparisons have separate scopes.

## Preliminary GRU evidence (before Transformer integration)

Artifacts: runs/assessments/sequence-ppo-20261031/. Original24fit files and eight
dataset/dependency/spec/binary identities are protected by baseline.json.
Targeted PPO/config/GRU80PASS, permitted reader/executor/GRU69PASS/1deselected,
Ruff/mypy72PASS; native library62PASS/2existing ignored plus3typed/6validation
integration tests PASS. These are overlapping targeted suites, not summed totals.
One sandboxed async test run was interrupted; it is not PASS evidence.

Initial native candidate integration failed7tests due to assuming effective_config
existed in the deliberately compact native Parent summary, and an incomplete-API
preflight raised AttributeError instead of ContractError. The approved fix exports
only verified model_id and uses an explicit legacy-MLP compatibility path. First
candidate and failure logs remain. Rebuilt gru-native-candidate-v2 integration
passes34tests/0skip in18.69s, including native/reference schema2 equality, two
synthetic coordinator updates/resume, cleanup and legacyMLP regression.

An early review also found decoded rollout arrays surviving until the next batch
load. Worker try/finally release now precedes save/ack/next load; two-update weakref
tests cover success/failure. Per-microbatch diagnostic dict retention is replaced
by running scalar totals, and CUDA parity/update peaks have separate scopes.

CUDA functional probe v2 passes W1/32/128/1024 on a CPU-generated synthetic batch;
maximum observed error7.152557373046875e-7. Typed optimizer/RNG restoration and the
subsequent GPU update reproduce exactly. The first private probe used the wrong
metadata reader for resume.npz; its failure is retained and corrected in v2,
without changing application loading. These probes are not performance trials.
Opt-in tests/test_gru_cuda.py:8PASS in4.48s, including separate1024-token endpoint
full/cache/split/gradient/admission cases. Long-endpoint checks do not substitute
for every-batch full parity. No inference or learning-quality claim follows.

## Combined core and wheel acceptance

Full candidate-wheel pytest:891PASS,25skip in412.34s. The20opt-in GPU cases were
executed separately: GRU8PASS4.48s, Transformer12PASS11.53s. Five other conditional
skips retain their existing scope. The preliminary benchmark helper tests are separately
12PASS2.19s; final cleanup-aware helper tests are14PASS2.08s; do not sum overlapping targeted suites. Ruff/mypy76 and native checks
pass. No semantic core edits followed these tests or the source reviews. Final
staging trimmed extra EOF blank lines in three Python files; ASTs including node
locations are identical (final-whitespace-proof.json). The original wheel retains
the exact tested source. A rebuilt final wheel89f7f5d1…72ca3a9 has all76files
byte-matched to final source and isolated installation (final-wheel-proof.json).

The fresh application wheel has76Python files byte-identical to source. A fresh
isolated Python3.14.10 environment verifies app/native installation, native loaded
identity, optional-free CLI/config and both model CPU artifact/cache roundtrips
without CUDA initialization. Base dependencies were copied from verified local
packages after an offline cache miss; no new dependency versions were resolved.
Evidence:wheel-verification/results.json. Prior models24/24hashes still match.

## Preliminary larger-budget CUDA smokes

Four preliminary parents live under runs/sequence-ppo-smoke-20261031/. Their
preprocessing settings are8192MiB/64MiB/legacy from the copied initial preset,
which do not establish the accepted2048MiB/16MiB/typed configuration. These runs
remain intact; final acceptance uses the corrected fresh parents below. Each has32healthy
natural/replay-reconciled games,2updates,379learner decisions and16Adam steps.
Original7200/5400/7140s deadlines remain. Both generations' weights differ from
initialization. Four chains verify normally with retention enabled: generation1
weights were intentionally deleted,0/latest retained. The first assessment used
require_artifacts=True (which requires even intentionally deleted files) and was
corrected to normal chain/receipt validation; no run/data was repaired or rerun.
Evidence:smoke-assessment.json and the four owner directories.

| Continuous execution | GRU | Transformer |
| --- | ---: | ---: |
| Parent elapsed (s) | 56.619061 | 57.721416 |
| Collection total (s) | 51.121992 | 51.306164 |
| Update response total (s) | 3.049869 | 3.997451 |
| Learner save total (s) | 0.031793 | 0.059376 |
| Maximum observed parity error | 7.152557e-7 | 1.192093e-6 |
| Maximum update allocated peak (bytes) | 116,768,256 | 119,993,856 |
| Owned artifact bytes | 64,359,686 | 67,077,753 |

Update response includes learner work/save; save is a nested interval and must not
be added again. Remaining elapsed includes preflight, preparation, IPC, commits
and cleanup. These are functional smoke observations, not a paired speed trial.
Resume parent elapsed includes its pause gap and is not compared with continuous
speed. Both models' final weights.npz/resume.npz/resume.json/inference.json and
both numeric manifests are byte-identical between continuous/resumed runs;
parent scheduling RNG/pool/collection position also match. Cache is episode-local
and is not a mid-game resume state. This establishes implementation feasibility,
not formal model strength or sample efficiency.

Smoke observers did not enable device sampling: their CPU-workload GPU label is
an observer flag omission, not evidence of absent GPU use. CUDA device/hardware,
all-candidate device parity and allocator peaks come from the actual learner.
The fixed-input CUDA campaign enables device sampling explicitly; no device
utilization claim is drawn from these smoke observer records.

## Accepted typed-reader / 2 GiB smokes

Final acceptance parents are runs/sequence-ppo-smoke-20261101/{gru,transformer}-
{continuous,resume}. All four effective configs explicitly record2048MiB memory,
512MiB cache,16MiB chunk,typed-borrowed-v1,one preprocessing worker andW32.
The four new experiment presets reference native-borrowed.toml. The copied old
8GiB preset reference was a root setup error, corrected without changing source
model code or old runs. Targeted corrected-preset/native regression:44PASS25.81s.

All six fresh/pause/resume CLI invocations exit0. All four parents are complete,
with32natural/replay-reconciled games,2updates,379learner decisions,16Adam steps,
original deadlines and successful child cleanup. Each pair's final weights,
Adam/RNG resume arrays, both numeric manifests, and parent schedule/opponent/pool
controls match exactly. Normal retention validates0/latest and generation1's
intentional deletion receipt. Evidence:typed-smoke-invocations.json and
smoke-assessment-typed-2g.json. Device sampling is enabled in these six owners.

| Accepted continuous run | GRU | Transformer |
| --- | ---: | ---: |
| Parent elapsed (s) | 57.639286 | 58.258666 |
| Collection total (s) | 51.861817 | 51.528805 |
| Update response total (s) | 3.219049 | 4.544891 |
| Maximum observed parity error | 7.152557e-7 | 1.192093e-6 |
| Maximum update allocated bytes | 116,768,256 | 119,993,856 |
| Owned artifact bytes | 64,359,574 | 67,077,747 |

Both corrected runs' two numeric manifests are byte-identical to their preliminary
legacy counterparts. Thus the frozen numeric fixture used in performance-v2 also
matches the accepted typed-reader input, although the original fixture's source
parent remains explicitly labelled as the preliminary run. The primary campaign
measures numeric compute, not preprocessing-budget efficiency. There are256real
smoke games in total including128preliminary and128accepted games; none are a
formal training-quality evaluation.

## Historical data regression

legacy-data-verification/summary.json records complete frozen history/observation
array/NPZ/sample/dataset equality, exact83,783-row Parquet/sidecar equality, and
saved current-v2 MLP PPO1/4worker parity for all8arrays with72original source
files unchanged per probe. Both complete recipes retain6,784episodes/415collections/
13,568Journals/534,640records/83,783accepted actions. Original24fit files match.
Setup-only invocation errors (missing observer variables, pre-created probe output)
are preserved separately; corrected invocations used new outputs, without repairing
or modifying old data. These are correctness jobs, not speed comparisons.

## Fixed-input measurement protocol

The reviewed workload and campaign controller use one immutable real GRU batch0
fixture. Only private old_logprob/value copies are rebased to each model's seed17
CPU cache path, leaving banks/candidates/selected/advantage/targets unchanged.
These remain off-policy diagnostic compute inputs, never usable on-policy data
or new training artifacts. Each command resets weights/Adam/RNG and checks source
hashes before/after against one campaign freeze. Model order alternates within
each matched condition; inference cached/full order also alternates.

Ten cells cover both models' CPU cached/full inference, CPU W32 updates and CUDA
W32/64/128 updates. One warmup and five measured fresh processes per cell give60
commands. All source/helper/config/runtime/native/wheel/protected-model identities,
array proofs and deterministic repeated-cell output hashes are checked. Existing
observer owns subprocess cleanup. GPU sampling is enabled for CUDA cells and
is explicitly device-wide, not workload-attributed. Own builds/tests/games/edits
stop during measurement. External-load question is unanswered; observed guest
CPU/pressure and device samples are retained without a host-isolation claim.

Inference comparisons use kernel intervals/per-row percentiles, excluding loading,
packing, sampling and parity checks. Instrumented diagnostic path wall includes
different checking overhead and is not the speed comparison. Whole-command RSS
also includes setup/corpus copies/rebasing and is not live participant RSS.
Update wall includes mandatory parity; parity and remaining update are reported
separately. Finer profiling is a separate diagnostic, not primary timing evidence.
Result publication uses closed/fsynced temp plus atomic no-clobber link; partial
serialization, cleanup failure and existing-owner cases have regression tests.

### Results and limits (2026-11-01)

performance-v2/{input-freeze,results,completion,summary}.json is authoritative.
All60commands PASS (10excluded warmups,50measured),234.120494s campaign elapsed.
The first campaign stopped after its first warmup on a4096-byte empty Torch
Inductor directory; no measured trial was accepted. The helper now assigns an
owned cache directory before Torch import and removes it only when empty.
Unexpected contents fail without deletion. Fourteen helper tests and independent
reviews pass. Old failure evidence is preserved; all60trials were fresh in v2.

The common batch has224endpoints,5,671candidate rows and16episode banks. History
lengths range5–29: W32/64/128 all cover whole prefixes, and CUDA final weight
digests coincide across W for each model. Differences between these W timings
are not evidence of long-history TBPTT effects. Separate1024-token functional
tests verify truncated-gradient behavior; the diagnostic below uses T64/W32.

| CPU forward + heads, median of5trials | GRU | Transformer |
| --- | ---: | ---: |
| Cached p50 / p95 (ms) | 0.098896 / 0.363677 | 0.247810 / 0.548886 |
| Full-prefix p50 / p95 (ms) | 0.218193 / 0.447928 | 0.441784 / 0.708653 |
| Sum of cached kernel intervals (s) | 0.032158 | 0.069829 |
| Sum of full-prefix intervals (s) | 0.055395 | 0.103441 |

For comparison, actual batch0 neural policy.select (including encoding/sampling,
excluding search) has p50/p95=1.703/10.117ms for GRU and1.835/10.663ms for
Transformer. These are one smoke batch's311neural calls, not five-trial medians.
Global report quantiles mix Random/heuristic and are not neural latency.

| Actual update incl. mandatory parity, median seconds | GRU | Transformer |
| --- | ---: | ---: |
| CPU W32 | 1.134527 | 1.314423 |
| CUDA W32 | 2.099767 | 2.963042 |
| CUDA W64 | 2.066006 | 2.724742 |
| CUDA W128 | 2.336213 | 3.028510 |
| CUDA W32 parity interval | 0.555814 | 1.263807 |
| CUDA W32 remaining update interval | 1.650559 | 1.692860 |

Each epoch has2logical minibatches and52/52/55/55microbatches across4epochs.
Independent phase medians do not add to the median total. GPU is slower on this
small fixture; no automatic CPU change or claimed GPU speedup is introduced.
Model bytes are1,265,160(GRU) /2,453,000(Transformer); cache bytes512/2,097,152.
W32-only maximum update allocated peaks are109.146/112.218MiB. Whole-command
W32-only sampled RSS maxima are933.61/909.77MiB on CPU and2118.48/2326.11MiB on CUDA,
including runtime/rebase/corpus copies and process-tree shared-page accounting.
These are not cache sizes or hard memory-limit violations. PSS and full per-trial
CPU/storage/device observations remain in the summary/raw observer records.

### Separate diagnostic phase profile

profile-{gru,transformer}/result.json uses one synthetic update with64own
decisions,16candidates per decision andW32, with
standard Torch profiler annotations. Host annotation totals include waits and
profiling overhead; they are not process CPU-seconds or primary speed estimates.
Raw device annotation rows may duplicate host names; do not sum them as kernel
execution. Only positive host-duration phase rows are tabulated below.

| Host-side inclusive annotation, ms | GRU | Transformer |
| --- | ---: | ---: |
| Complete mandatory parity | 819.908 | 1610.016 |
| Input packing / transfers | 84.378 | 85.250 |
| Prefix rebuild (128calls) | 123.898 | 225.683 |
| Suffix encode (256calls) | 392.280 | 489.774 |
| Candidate/value heads | 88.319 | 81.546 |
| Loss / metric synchronization | 358.620 | 325.202 |
| Backward | 1861.482 | 1325.212 |
| Gradient clip | 101.093 | 103.450 |
| Adam step | 116.721 | 119.046 |

The profiler owner elapsed78.007/62.502s includes expensive instrumentation and
aggregation/teardown. Do not compare it with production update wall or add it to
the234.12s primary campaign as if it were a measured trial. No extra optimization
or device/preset change is accepted from these single synthetic captures.

## Installed state

The tested native release/mimalloc wheel is installed offline with no dependency
changes, while the editable app source is retained. Native source:
ab07f42dba984a661ba0acab040cd070f1f713e855f8a22b84bd6f61e3f3a945;
ELF:4ca8dfd2e1c43bd2cdcc665d066c86be329bef37349e6bcdb90992a21f9eaf18.
Old38ad8a23package/metadata has a verified rollback copy. Loaded identity and both
normal CLI Configs pass;24protected files and all8initial identity files match.
See installation.json and main-before-adoption/identity.json. No original source
dataset, model, dependency manifest, simulator source/binary/config is rewritten.
