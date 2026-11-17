# Collection CPU, communication and GPU-transfer investigation

Date:2026-11-17. User-approved [bounded design](../design/collection-cpu-optimization.md)
under [ADR0047](../adr/0047-collection-cpu-investigation.md).
Measurement populations and all original audits are complete. Both independent
whole-branch/evidence reviews are FINAL READY with no unresolved findings; root
accepts the approved investigation. Normal production defaults remain unchanged.

## Result

Parent validation reuse P is the clearest candidate under the measured
width192/two-layer conditions: both NEW confirmation pairs improve collection
time for each of MLP/GRU/Transformer. T2 materially reduces H2D submissions but
has small/mixed complete-collection effects. HTTP read overlap H does not improve
the aggregate exploratory result and was not selected for confirmation.

P+T2 improves all three fresh combination pairs per model versus its matched
normal controls. It was not compared directly with P; different timing blocks
must not be used to claim that T2 adds a specific saving on top of P.
No automatic default adoption, learning-quality/HPO objective or final-test
acceptance follows from this investigation. Larger-model performance is untested.

## Fixed conditions and provenance

- Measured Bots source: `e191f6a98cdf774790e7461f033fd29506707626`.
  Starting baseline `f3aeca2`, accepted plan `b5de2d8`, mechanism code/review fixes
  `3281ac7`/`3ad7638`, harness `fab66dd`/`470eae2`, path-normalization `e191f6a`.
- Simulator revision `91d5926a7572db327eaea3751e7c24f428120996`; original binary,
  native extension, normative specs, configs and runtime are hashed in freezes.
  Python3.14/GIL, NumPy2.5.4, Torch2.14.1+cu132, RTX5070Ti; original IEEE FP32,
  deterministic settings and one Torch thread. No dependencies or Simulator edits.
- MLP/GRU/Transformer width192/two layers, seed17, four resident workers/two
  concurrent games,16games×2updates,epoch1,D1M,wall600 and all original deadlines.
  Source2GiB/numeric512MiB/collector2GiB/parent1GiB/save1GiB, managedRAM24GiB/
  CUDA12GiB and64MiB tensor caps remain. Both comparison arms reserve the same
  additional96MiB per physical actor through original admission/custody.
- GPU inference and normal optimized Model4 PPO learning remain unchanged.
  Stored inference5/Numeric6/token1 inputs retain original producer metadata;
  corrected Collection8 learner seats come from `manifest.linkage`, not legacy
  `ppo`. Ordinary resume source matching is not relaxed.

Artifact root: [`runs/collection-optimization-v2`](../../runs/collection-optimization-v2/).
Namespace `3ef7eb00-12f4-49cb-981e-00ee7161b296`. Every original actual invocation
has fresh run/context IDs, an immutable prelaunch options/code/config/input/output
sidecar, original JobContext/ledger and genuine termination receipt. Production
identities are base-only, with experiment controls separately recorded.

The [timing-end source check](../../runs/collection-optimization-v2/timing-end-source-check.json)
confirms all285/294/325 files in the diagnostic/pilot/confirmation freezes still
match immediately after ALL timing. Post-measurement aggregation corrections
are analysis-only, retain that measured-source identity and do not change any
original result/request/checkpoint/freeze bytes or require replacement timing.

## Populations and metric

All timing runs were sequential without concurrent tests, reviews or audits.
Profiler runs are separate and excluded from performance selection. Actual cold
initialization remains in whole-run totals; fixed replay warmup/first/warm work
are separately labeled. Seeds alone do not guarantee identical training or game
trajectories. There is no independent-game action-equality acceptance gate.

The original two-batch update `collection_seconds` sum is primary; the original
coordinator call interval is secondary. Nested/concurrent per-call sums, CUDA
stream events and CPU time are not a collection-wall partition. Driver busy-wait
is not proof of useful CPU computation. Diagnostic overhead is not performance.

| Population | Completed |
| --- | ---: |
| Passive actual diagnostics | 3invocations/96natural games |
| Fixed learner replay | 36pairs/72fresh processes |
| Separate GPU profiles | 15processes,6selected endpoints each |
| Shared-control exploratory selection | 12invocations/384natural games |
| New singleton confirmations | 24invocations/768natural games |
| New combination confirmations | 18invocations/576natural games |
| Actual total | 57invocations/1,824natural games |

All commands returned0. Actual final parents are honestly
`incomplete/update_cap`, two committed updates each; not completed formal
training. Original audits remain a distinct post-timing requirement below.

## Baseline CPU diagnosis and experimental boundaries

The passive parent observations count2,444 `CandidateTable.validate` calls from
Selection per MLP/GRU32-game run and2,412 for Transformer: four full validations
per611/603 total decisions. Further learner tokenization invokes validation314/
310times. P reuses only the successful same-object proof within the original
decision Task/thread, and explicitly suspends reuse at Journal and Capture;
selected-action checks and all Source/audit checks remain original.

Detailed counter timings themselves add overhead: for example, instrumented
MLP parent Selection-to-table validation totals16.94inclusive call-seconds.
This is not subtracted from uninstrumented collection time or called an exact
removable cost. The measured full-route confirmation below determines impact.

Existing physical compression is already on a dedicated thread and SQLite on
a writer process. Previous32-game pack encoding is only0.078s and includes more
than compression. Existing generator-analysis memo saves about0.10s; these are
retained low-priority evidence, not extra variants or hidden tuning runs.

## Transfer replay and actual copy evidence

Original full parent audits prepared ALL32learner histories per model from two
Numeric6 batches, paired with their exact saved G0/G1 checkpoints and original
sampling seeds. This is314/314/310decisions, explicitly not every neural opponent
decision. The source reader validates the scalar seed mapping by bank/actor,
schedule and corrected learner-seat linkage, with no order assumption or
producer/manifest rewrite. Actual runs include all original participants.

Each fresh replay process performs one full warmup and one full warm measured
G0/G1 pass. Within-family three paired comparisons alternate transfer order.
T0 is unchanged control. Historical typed reuse and T1/T2/T3 are experimental
options; T0 cannot win selection as an optimization.

The [complete replay ledger](../../runs/collection-optimization-v2/replay-ledger.json)
has36MEASURED pairs and no invalid populations. T2 is selected by the predeclared
rule: candidate warm medians summed across three models5.550s versus its matched
controls6.059s. T3 is5.521s, within1%; prefer simpler T2. Historical reuse totals
5.684s and T1 totals5.690s. These are learner inference-core replay scopes,
not32-game collection savings.

The fifteen separate profiles select three smallest and three largest recorded
candidate-shape endpoints/model, executing their full own prefixes before the
profile window. They are shape-tail profiles, not latency-tail or maximum-size
stress cases. Actual profiler activities show, for each model:

| Six profiled endpoints | T0 | T2 |
| --- | ---: | ---: |
| H2D copies | 144 | 12 |
| CUDA stream synchronize API calls | 192 | 60 |
| D2H copies | 48 | 48 |

H2D submissions fall91.67%, to one Frame and one candidate transfer per endpoint.
MLP/GRU H2D payload1,141,252→1,141,324B; Transformer1,140,516→1,140,588B. The
72B increase is alignment, not a bandwidth saving. Device-sync calls remain12.
Kernel/DtoD counts and payload metadata remain available per separate profile;
CUDA event elapsed is not summed kernel busy time. Maximum trace size and memory
observations are recorded in the final artifact ledger.

## Actual exploratory selection

One shared C control per model is compared with P/T2/H in predeclared order.
The signed collection saving summed across models is P5.229s (6.299%),
T2 0.437s (0.527%), H−0.548s (−0.660%). All12rows are MEASURED; mechanical
eligibility remains distinct from timing signs. P and T2 are selected, and their
combination is run based on mechanical compatibility, not favorable timing.

The selected singleton's exploratory comparison is **not** a third independent
confirmation. The following singleton figures use only TWO new independent
pairs/model; the three NEW combination pairs use their own matched controls.

## Independent actual confirmation results

Positive saving means faster. Figures are medians of paired differences and
paired percentages; a difference between separately computed group medians is
not substituted. Complete per-pair values/IDs and an independent recalculation
from original update files are in
[root-independent-metrics.json](../../runs/collection-optimization-v2/root-independent-metrics.json).

| Model | P: saving s / % (2pairs) | T2: saving s / % (2pairs) | P+T2: saving s / % (3pairs) |
| --- | ---: | ---: | ---: |
| MLP | 1.082 / 4.01% | 0.274 / 1.01% | 2.037 / 7.61% |
| GRU | 1.596 / 5.85% | −0.039 / −0.15% | 1.030 / 3.91% |
| Transformer | 1.803 / 6.49% | 0.295 / 1.07% | 1.227 / 4.55% |

P improves both independent pairs for every model. T2 is mixed for MLP,
slower in both GRU pairs and positive in both Transformer pairs; its complete-
collection benefit is small and conditional despite the clear copy reduction.
P+T2 improves all three pairs per model. Median coordinator savings for P are
0.862/1.711/2.024s; for P+T2 they are2.326/0.943/1.584s (MLP/GRU/Transformer).
Do not compare those blocks as a direct P-versus-P+T2 experiment.

Recommendation is to prioritize a separately reviewed production P adoption
proposal; treat T2 as a conditional option, not a general throughput result.
Any adoption or broader sizing requires its own decision. Three combination
pairs and two singleton confirmation pairs do not establish statistical
confidence or broad-model performance. H received no new confirmations.

## Verification, failures and final acceptance

Code verification:168PASS/13optional CUDA SKIP in the reported host selection,
root HOST CUDA13PASS3.54s, scoped mechanism/harness regression76PASS, path-freeze
tooling28PASS, scoped Ruff and mypy14PASS. Selections overlap; do not sum them.
No whole-suite, formal-HPO/quality or final-test result is claimed.

Independent code reviews resolved H Task ownership/partial creation, T joint and
preallocation caps, original telemetry composition, replay reap races and
physical comparison/request identity binding before any measurement. The
private historical inference wrapper's `_operands=None` compatibility defect
was repaired without accepting owned/learner operands or changing normal math.

The first real freeze command failed before publication because materialized
server paths are strings; normalization and real-asset tests fixed it before
timing. Final all57aggregation exceeded the unchanged64MiB JSON cap through
duplicated per-pair summaries; compact pair references preserve full per-run
evidence/metrics without raising caps or rerunning measurements. The analysis
revision is `ed03ab6`; scoped independent correctness review is READY. The final
ledger is44,289,541B, below the unchanged64MiB limit, with57MEASURED rows,
21fresh confirmation/combination pairs and exact agreement with the independent
original-update recalculation. Full summaries occur once, and paired references
retain exact report/run/context/request/result identities.

PASS [original audits](../../runs/collection-optimization-v2/original-audits.json):
57/57 original parents,19permodel. Each original validator uses
`require_complete=False,max_owned_bytes=2GiB,deadline=monotonic()+180`;
maximum audit10.714s, summed598.393s. All90original files/parent retain before/
after SHA/stat identities,5,130file instances total. Every exact-context original
GPU termination receipt passes; no interrupted/failed run is relabeled success.

The [complete final ledger](../../runs/collection-optimization-v2/final-ledger.json)
preserves all54performance and3diagnostic results. Separate
[profile summary](../../runs/collection-optimization-v2/profile-summary.json)
has15profiles, maximum14,150,277B per six-endpoint profile and194,404,608Btotal,
below512MiB/run. The largest individual trace is2,808,583B.
[Memory summary](../../runs/collection-optimization-v2/memory-summary.json)
reports individual actor process-lifetime maxima, not a simultaneous device
peak: max allocated56,821,760/62,379,520/66,664,448B for MLP/GRU/Transformer.
The identical maxima across these arms are dominated by lifetime work and do
not prove zero incremental packing memory. Global physical high-water is
unavailable; all original managed admissions passed.

Fresh root final host verification after the analysis-only fix passes180tests/
13CUDAopt-in skips (5.26s), plus scoped Ruff and mypy14files. The single NVML
warning is an environment warning in the host-only Dynamo test. Earlier root
host CUDA13PASS remains the GPU mechanism evidence; unchanged GPU mechanisms
were not rerun merely for a serialization-only correction.

[Final evidence index](../../runs/collection-optimization-v2/final-evidence-index.json)
pins execution ledgers, selections, original audits, independent metrics, profiles
and memory summaries. All175 independent raw-input hashes and all57/21 final
populations pass root's fresh check. Measured source versus the sole documented
post-timing analysis-file difference is verified against the recorded git bytes.

Independent correctness/specification/quality and architecture/conformance
whole-branch/evidence reviews are FINAL READY with no unresolved findings. They
independently reproduce all57identities/1,824natural outcomes,21paired results/
nine medians,175independent-input hashes,22evidence-index hashes,5,130original
file stat identities, replay/profile completeness and transfer/memory summaries.
The reviewers did not rerun tests, experiments, training or original audits.
Root's final test/static/hash/link/whitespace checks pass; root accepts this
investigation scope and preserves the local evidence and feature branch.

## Guidance considered

The transfer prototypes follow [NVIDIA's small-transfer batching guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).
Pinned/nonblocking transfer is evaluated as a separate factor because
[PyTorch documents workload-dependent costs](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).
Parent diagnosis follows [Python3.14's event-loop blocking guidance](https://docs.python.org/3.14/library/asyncio-dev.html#running-blocking-code);
the enabled GIL means pure Python CPU work is not automatically accelerated by
thread offload, as [the to_thread documentation explains](https://docs.python.org/3.14/library/asyncio-task.html#asyncio.to_thread).
The current32-game results, rather than those general recommendations, determine
the candidate ranking above.
