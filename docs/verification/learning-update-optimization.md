# PPO learning update optimization

Date: 2026-11-17. Implements [ADR0045](../adr/0045-learning-update-optimization.md)
and [the reviewed design](../design/learning-update-optimization.md).

The measured adoption proposal is `microbatch` for MLP and `shared-prefix` for
GRU/Transformer. Every actual pair improves collection-inclusive time. Normal
defaults remain unchanged; production adoption is a separate change. These are
short-protocol performance results, without a learning-quality, formal-HPO or
final-test conclusion. Both final independent reviews are FINAL READY; root
accepts the approved comparison including the separate telemetry supplement.

## Identity and evidence

Design commit: `89d50e3`. All measured application/benchmark source is committed
at `a0f243f2b90a35e151b410b5936bce1750548aa2`. Source, configuration and plan stay
unchanged throughout timing. Existing untracked analysis/session files and the
fixture `.lock` predate this work. The recorded dirty flag is truthful; the
explicit freeze binds reviewed bytes rather than claiming a clean checkout.

Simulator source: `91d5926a7572db327eaea3751e7c24f428120996`; binary SHA-256
`76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d`.
The binary build revision is unknown; source HEAD does not prove its build.
The v2 action contract uses `public-ja-v3`, catalog schema2, Model4, numeric6,
inference5/resume5 and unchanged Wire7/producer-owned meanings. Reward remains
`natural-terminal-v1`. Simulator, dependencies, ordinary config/checkpoint
formats and production defaults are unchanged.

Python3.14.10/GIL, NumPy2.5.4, Torch2.14.1+cu132, CUDA13.2, RTX5070Ti16,303MiB,
driver610.88/driverAPI13030. IEEE FP32, deterministic execution, disabled CuDNN
GRU calls and MATH SDPA remain in use. Before profiling the shared device showed
2,803MiB used/1% utilization; the pre-core observation was2,806MiB/1%.
Exact pre-core/pre-actual observations are retained.
No exclusive-device or broad hardware result is claimed.

Artifacts under [`runs/learning-update-optimization-v1`](../../runs/learning-update-optimization-v1/):

| Evidence | Artifact |
| --- | --- |
| Six historical inputs and G0/G1/Adam/RNG identities | [fixed-inputs.json](../../runs/learning-update-optimization-v1/fixed-inputs.json) |
| Reviewed fixed-core bytes | [core-source-freeze.json](../../runs/learning-update-optimization-v1/core-source-freeze.json) |
|84 rows, completion, cumulative/incremental effects | [core/](../../runs/learning-update-optimization-v1/core/), [core-summary.json](../../runs/learning-update-optimization-v1/core-summary.json) |
| Candidate choice and scope clarification | [selection](../../runs/learning-update-optimization-v1/candidate-selection.json), [notes](../../runs/learning-update-optimization-v1/candidate-selection-notes.json) |
| Predeclared profile windows | [baseline plan](../../runs/learning-update-optimization-v1/baseline-profile-population.json), [candidate plan](../../runs/learning-update-optimization-v1/candidate-profile-plan.json) |
| Trace sizes/hashes/kernel-copy-sync counts and durations | [profile-summary.json](../../runs/learning-update-optimization-v1/profile-summary.json), [attribution](../../runs/learning-update-optimization-v1/profile-attribution.json) |
|18-run schedule, topology checks and433-file freeze | [plan](../../runs/learning-update-optimization-v1/actual-plan.json), [validation](../../runs/learning-update-optimization-v1/plan-validation.json), [freeze](../../runs/learning-update-optimization-v1/actual-freeze.json) |
| Complete-original Source/numeric/receipt audits | [strict-audits.json](../../runs/learning-update-optimization-v1/strict-audits.json) |
| Raw effects and audited analysis | [final-summary.json](../../runs/learning-update-optimization-v1/final-summary.json), [analysis.json](../../runs/learning-update-optimization-v1/analysis.json), [CSV](../../runs/learning-update-optimization-v1/actual-results.csv) |

Actual freeze SHA-256:
`baef5baf2ead7c2e8be51a6ed430e4dd0c617e5ac59471d5a65fdf8437dba5e0`.
Retain the local ignored datasets and historical `runner.py`/`measure.py`/
`analyze.py` with these artifacts. They are frozen prerequisites; a fresh source
checkout alone does not contain the experimental population.

## Fixed-data comparison

All six investigated train-side batches pass complete-original preparation and
file/schema/producer checks. Historical producer:
`8b08e56e821c8ff72fabf892383c144021b0e8885dfdbe38218e21f38e25f59a`.
G0/Adam-step0 pairs with batch0; G1/step2 pairs with batch1. Each warmup and
measurement restores weights, Adam presence/moments/parameter steps, Torch
CPU/CUDA RNG and PCG64. Ordinary current-source restore remains strict; no
single-use owned capability is manufactured or rewound.

PASS exactly84 unprofiled measurements plus separate restored warmups: MLP has
4variants, sequence models5, each with2batches and3forward/reverse/forward repeats.
The same immutable NumPy adapter feeds every variant. Core includes packing,
transfers and numerical updates; it excludes full Source/capability/FD work,
input retirement and snapshot/save. It is not the earlier approximately22s
interval. Below are sums of the two batch medians, seconds:

| Model | baseline | metrics | packed | microbatch | shared-prefix |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP |3.7981|3.7114|3.4485|0.6242|—|
| GRU |18.9690|18.9856|17.1996|2.2721|1.7588|
| Transformer |18.7967|18.7796|17.0408|2.3082|1.8033|

Metrics alone gives no consistent material benefit; packing reduces the core
about9%. Microbatching gives the largest gain. Sharing adds19.6–23.5% over
microbatching on sequence batches, with similar or smaller allocated peaks.
Chosen variants are fastest in both batches and every paired two-batch repeat.
Paired core savings range83.30–83.88% MLP,90.64–90.93% GRU,90.35–90.42%
Transformer. Incremental and baseline effects remain separate in the JSON.

| Model | Process CPU baseline→chosen, s | CUDA-event interval baseline→chosen, ms |
| --- | ---: | ---: |
| MLP |4.711→0.776|3798.0→624.2|
| GRU |23.399→2.174|18968.6→1758.8|
| Transformer |23.254→2.246|18796.4→1803.3|

Process CPU covers all threads and can exceed wall. CUDA-event intervals include
host submission gaps, not pure kernel duration. No clock subtraction infers I/O
wait. Actual successful frame encodes are314 for MLP,2508→816 for GRU and
2444→840 for Transformer. Earlier481/491 unique-frame figures are opportunity
bounds; splits require recomputation. GRU microbatch/shared group counts120/92,
Transformer124/94. Diagnostic `transfers` counts only explicit target/slab
constructions; it excludes ordinary raw model copies in baseline/metrics. Use
observed profiler copy/API counts for actual copy evidence.

## Profiling and memory

PASS25 separate captures:11baseline,11candidate and3separate shape runs. Empty/
short, large-candidate, long-history and final-group windows are predeclared.
Every capture has the intended group and CUDA activity; max trace65,242,157B
(62.2MiB), below512MiB/run. No missing activity or bound failure occurs. Profiling
and shape capture are excluded from timing selection.

Annotations identify input/HtoD, frame/temporal/candidate/loss/checks, backward,
clip/Adam and synchronization. Activity durations and kernel/device-copy/runtime-
sync counts are separate. The GRU29-frame baseline window, for example, has
22,322kernels,839device copies and485explicit runtime-sync APIs. Candidate groups
can contain additional endpoints: raw window counts are not equal-population
full-update comparisons. Activity sums may overlap; annotations are inclusive,
not an exclusive wall ledger or an I/O-wait estimate.

Maximum core CUDA peaks across six measurements/variant, MiB:

| Model | Baseline allocated | Chosen allocated | Reserved ceiling across variants |
| --- | ---: | ---: | ---: |
| MLP |99.08|143.13|158|
| GRU |116.81|119.19|172|
| Transformer |118.96|117.82|176|

MLP grows about44MiB. Reserved memory is a warm-process allocator observation,
which may retain earlier variants' cache. Core50ms sampled PSS/RSS maxima are
approximately2.230/2.246GiB MLP,2.242/2.258GiB GRU,2.314/2.330GiB Transformer.
Sampling can miss brief peaks; no process-memory reduction is established.

Actual limits remain RAM24GiB/CUDA12GiB,Source2GiB,numeric512MiB,collector2GiB,
parent/save1GiB. Unchanged admission estimates peak at13.746–13.813GB RAM and
9.451–9.518GB CUDA, with2.966–2.979GB learner reservations. These are conservative
managed estimates, not measured physical peaks. Groups fit the actual reservation
and individual64MiB tensor limits without OOM fallback, dropped endpoints or
increased caps.

The original18 timing observer did not retain learner allocator or actual-run
CPU/PSS/RSS. The separate supplement below supplies those process-scoped
observations. Simultaneous whole-device/process-tree peaks remain unmeasured;
core samples and per-actor maxima must not be summed into a simultaneous peak.

## Actual path and strict audits

PASS18 fresh runs:3alternating pairs/model (baseline/candidate,reverse,forward),
width192/layers2,seed17,workers4/concurrent2,games16×updates2,epoch1,logical128,
TBPTT32,D1000000,wall600s and original caps. Root checks18topology rows,36unique
UUIDs and54fresh outputs and freezes plan bytes. All18 admission-only checks pass.
Timing is sequential with no concurrent tests/audits/reviews.

Each run invokes the original coordinator once, retaining original creator,
fixed worker, FD lifecycle and positive GPU receipt. Every parent stops as
healthy `incomplete/update_cap`,2updates/3generations, without objective or
completion promotion. MLP/GRU use314 endpoints/run and Transformer310 in both
routes here. Count agreement is not an action/weight equality gate or a general
determinism claim.

Median seconds; numeric/request values total both updates:

| Model | Numeric loop baseline→chosen | Update request baseline→chosen | Collection-inclusive coordinator baseline→chosen | Median paired coordinator saving |
| --- | ---: | ---: | ---: | ---: |
| MLP |5.933→2.427|6.320→2.802|38.672→34.981|9.54%|
| GRU |21.075→3.908|21.485→4.362|53.774→36.701|31.85%|
| Transformer |21.320→3.864|21.799→4.333|54.498→37.320|31.60%|

Original helper intervals improve in every pair too: median paired savings
9.49%,31.71%,31.42%; helper median seconds38.927→35.232,54.041→36.960,
54.774→37.590. Helper/coordinator intervals differ; fresh-process/declaration
setup is not silently included in these figures.

Individual coordinator pairs, baseline→candidate seconds:

| Model | Pair0 | Pair1 | Pair2 |
| --- | ---: | ---: | ---: |
| MLP |38.407→35.013|38.672→34.981|38.700→34.872|
| GRU |53.664→36.821|53.854→36.701|53.774→35.972|
| Transformer |54.877→37.537|54.446→37.320|54.498→36.947|

Every run's exclusive startup/collect/source/update/sync/cleanup ledger reconciles
with coordinator wall. Collection medians stay near27s:27.316→27.199 MLP,
27.159→27.158 GRU,27.462→27.550 Transformer. Update-phase medians fall
6.388→2.867,21.585→4.454,21.884→4.400s. Individual ledgers are in analysis.json;
component medians need not sum to the median total.

Nested two-update snapshot/save medians:0.143→0.142/0.101→0.099s MLP,
0.153→0.160/0.116→0.116s GRU,0.183→0.181/0.135→0.136s Transformer. Input
retirement in this timing population is included in completion/request intervals
and positively audited, without an isolated duration. The supplement separately
measures the worker release span. Batch completion includes numerical work/ACK waits,
not pure IPC. Nested archive, snapshot and request spans are not added to the
exclusive phase ledger.

After timing, all18 complete-original Source/numeric/context-bound receipt
audits pass, preserving SHA and device/inode/mode/size/mtime/ctime identities of
1620file instances. Authoritative packed readers validate36batches×16=576natural
games with zero non-natural closures and original journals/replays. Producer
proofs retain truthful omitted fields (`replay_reconciled=false`, `health={}`);
those flags alone are not audit evidence and are never rewritten. No custom
SQLite/archive decoder is used.

## Verification and corrections

PASS350 original host regressions (247.00s):

```sh
UV_CACHE_DIR=/tmp/gfb-update-uv-cache uv run --no-sync pytest -q tests/test_hpo_numeric.py tests/test_hpo_network.py tests/test_hpo_artifacts.py tests/test_hpo_owned_numeric.py tests/test_hpo_owned_run.py tests/test_hpo_owned_startup.py tests/test_hpo_owned_wire.py tests/test_hpo_owned_phase_limits.py tests/test_hpo_owned_restore.py tests/test_structured_network.py tests/test_structured_learning.py
```

PASS67 final checks/1inapplicable MLP shared-prefix skip (14.20s):

```sh
UV_CACHE_DIR=/tmp/gfb-update-uv-cache uv run --no-sync pytest -q tests/test_learning_update_benchmarks.py tests/test_learning_update_diagnostics.py tests/test_learning_update_execution.py tests/test_learning_update_admission.py tests/test_learning_update_boundaries.py
```

PASS scoped Ruff and mypy22files at the main milestone,11affected files after
harness fixes; cached diff check; optional-free current-source CLI help using
import blockers;14/14 host CUDA finite-update smokes (3synthetic endpoints,
epoch1,minibatch2,delta2). No new installed-wheel or whole-suite run. No native/
dependency change needs rebuilding. Formal quality/HPO/final-test checks are
NOT RUN.

Review fixes before timing: complete real-dtype/intermediate/later-layer tensor
admission; true53/5/47/7tails; long-TBPTT detach; distinct own-card and valid
empty-hand fixtures; observer request contract; numeric-file origin; profiling
boundaries; exact population/real-baseline summaries. Wrong-final-cards mutation
fails for both sequence models. Existing gradient/resume/ownership/frozen-kernel
checks remain; no custom numerical tolerance or saved-weight/action equality
performance gate is added.

The initial sandbox suite had309PASS/41EPERM failures adopting inherited sockets,
before learner construction; identical host selection passes without a product
fix for that environment. Preparation initially failed on numeric path rooting
(fixed before timing), then root's omitted output-directory setup (corrected
before publication). No84-update or18-run measured population failed, was pruned
or rebudgeted. Extra EOF whitespace was removed before the code commit.

Principal recorded commands, from the repository root (CUDA uses host execution):

```sh
.venv/bin/python -m benchmarks.learning_update.prepare inputs --output runs/learning-update-optimization-v1/fixed-inputs.json
.venv/bin/python -m benchmarks.learning_update.compare --input runs/learning-update-optimization-v1/fixed-inputs.json --output runs/learning-update-optimization-v1/core
.venv/bin/python -m benchmarks.learning_update.summarize --core runs/learning-update-optimization-v1/core --output runs/learning-update-optimization-v1/core-summary.json
.venv/bin/python -m benchmarks.learning_update.plan --candidates runs/learning-update-optimization-v1/candidates.json
.venv/bin/python -m benchmarks.learning_update.prepare freeze --output runs/learning-update-optimization-v1/actual-freeze-base.json --reviewed-revision a0f243f2b90a35e151b410b5936bce1750548aa2 --config runs/learning-update-optimization-v1/configs/mlp.toml --config runs/learning-update-optimization-v1/configs/gru.toml --config runs/learning-update-optimization-v1/configs/transformer.toml
.venv/bin/python -m benchmarks.learning_update.population --plan runs/learning-update-optimization-v1/actual-plan.json --freeze runs/learning-update-optimization-v1/actual-freeze.json --execute
.venv/bin/python -m benchmarks.learning_update.audit --plan runs/learning-update-optimization-v1/actual-plan.json --output runs/learning-update-optimization-v1/strict-audits.json
.venv/bin/python -m benchmarks.learning_update.summarize --core runs/learning-update-optimization-v1/core --actual-plan runs/learning-update-optimization-v1/actual-plan.json --output runs/learning-update-optimization-v1/final-summary.json
```

Create the fresh parent directory first. Outputs use exclusive-create; these
commands describe executions, not overwriting existing evidence. Root extends
the base freeze with plan/selection/core artifacts and checks all433hashes before
18admission-only checks and execution. Each predeclared profiler row invokes
`python -m benchmarks.learning_update.diagnose --input ... --output ... --model ...
--batch 0 --variant ... --start ... --groups 1`; only3separate rows add `--shapes`.
Core controller791.61s includes warmups/restores; actual controller780.07s covers
all18helpers. Neither substitutes for per-update wall.

## Recommendation and limits

Propose MLP microbatch and GRU/Transformer shared-prefix for later production
adoption: all9whole-run pairs improve, update gains are substantial and core
memory costs bounded. Normal defaults stay unchanged in this task. Metrics and
packing are cumulative prerequisites, not individually established default
adoptions. Preserve baseline for comparison/diagnosis.

Histories reach29 with TBPTT32. Long detached-prefix behavior is synthetic-test
evidence, not maximum1024-history performance. One shared GPU, one seed and3pairs
do not establish broad hardware/load/quality generalization. Simultaneous physical
memory and parent-side retirement timing have the limits above/below. No quality,
formal-HPO or final-test pass follows.

Accepted tooling debt: arbitrary external18-row plans lack a generic full
validator. Acceptance requires root's checked topology and frozen plan bytes.
Revisit before accepting non-root plans or dropping those checks. Revisit the
experimental maintenance burden on production adoption/protocol retirement.
Final evidence reviewers must assess these explicit limitations.

## Separate actual telemetry supplement

Both evidence reviewers verified the existing arithmetic, identities, receipts
and Source evidence but held whole-plan acceptance on the missing actual-path
telemetry. The approved measurement wording has no core-only qualifier. Root
accepts the advisor's remedy in the
[telemetry amendment](../design/learning-update-telemetry-supplement.md): six
separate actual diagnostic runs under the same full conditions/caps, without
pooling them into the original18 timing medians. The original source is preserved
as source-a0f243f.tar with243file hashes checked against its freeze. Implementation
passes both implementation reviews,119affected host checks, final26targeted
checks/1CUDA skip plus3original CPU lifecycle rechecks, scoped Ruff/mypy4files,
and the gated original CUDA worker test (1PASS,5.60s).

PASS six fresh diagnostic runs at
`f067d030de4c7131aaf4e103ecd5d0fa2d37ef99`, each with the same original conditions,
full600s deadline and caps. All requests/results are `timing_eligible=false`.
No diagnostic duration is pooled with or retroactively attached to the original
18 performance medians. Eight numerical/group/admission/artifact source files
are byte-identical to a0f243f; both code reviewers confirm an observational delta.

The287-file diagnostic freeze SHA is
`8df8a1ed8b0f9f0f53b0c8113d61fe2a19ea037a0803ed3b354857ae567b4b8d`.
[Plan](../../runs/learning-update-optimization-v1/telemetry-v1/diagnostic-plan.json),
[freeze](../../runs/learning-update-optimization-v1/telemetry-v1/source-freeze.json),
[summary](../../runs/learning-update-optimization-v1/telemetry-v1/telemetry-summary.json)
and [strict audits](../../runs/learning-update-optimization-v1/telemetry-v1/strict-audits.json)
bind six fresh run/context identities,12worker records and6coordinator records.
Every sampler stops; every record has zero read/sampling failures and retains
aggregate statistics rather than sample lists. Requested cadence50ms; largest
observed gap127.6ms is reported rather than hidden. No pending publication residue
or close uncertainty remains. One held16MiB process lease is a named subset of
existing workspace/management, not additional budget or a physical-free-memory
claim.

One diagnostic run/route, two-update totals; these include observation overhead
and provide measurement coverage, not a repeated speed estimate:

| Model/route | Worker update wall/CPU, s | CUDA-event interval, ms | Peak allocated/reserved, MiB | Worker sampled PSS/RSS, GiB |
| --- | ---: | ---: | ---: | ---: |
| MLP baseline |5.800/6.879|5799.8|98.99/116|1.917/2.261|
| MLP microbatch |2.686/3.110|2686.0|143.12/158|1.939/2.282|
| GRU baseline |22.013/26.839|22013.1|116.06/126|1.930/2.274|
| GRU shared-prefix |4.002/4.663|4001.5|119.62/138|1.947/2.291|
| Transformer baseline |21.534/26.279|21533.2|118.96/130|1.962/2.307|
| Transformer shared-prefix |4.390/5.140|4390.1|117.82/142|1.977/2.321|

| Model/route | Coordinator CPU, s | Coordinator sampled PSS/RSS, MiB | Worker input retirement wall/CPU, ms |
| --- | ---: | ---: | ---: |
| MLP baseline |19.771|156.83/164.58|12.328/15.673|
| MLP microbatch |19.683|150.70/162.07|12.327/14.380|
| GRU baseline |20.009|161.61/172.87|16.218/19.247|
| GRU shared-prefix |19.881|158.89/169.67|25.909/28.909|
| Transformer baseline |20.097|163.94/175.24|5.661/5.660|
| Transformer shared-prefix |19.658|164.41/175.72|20.567/23.180|

Worker request starts after valid command checks, before numeric receive; update
CUDA markers surround the original update_owned call and are read after its
existing synchronization. Allocator counters reset for each request and are read
after retirement. Worker retirement spans del-numeric, existing sync, delivery
completion sends and control close. It is distinct from the parent's earlier
completion wait, which overlaps numerical work. Parent batch retirement after
snapshot ACK/commit remains under its original owner; its isolated duration is
not inferred. Snapshot export/save/send retain their original order and have
separate wall/CPU entries. The full raw phase records and boundary sample counts
are retained; process CPU includes that process's threads and sampling overhead,
excluding children. Event intervals include host gaps, not pure kernel time.

After all diagnostic observation, six strict Source/numeric/context-bound receipt
audits pass with540unchanged original file instances and192additional natural
games. They remain separate from576comparison games (768natural overall), with
no objectives or new training-quality authority. Worker telemetry binds spec,
source,resume,model/variant,generation/transaction and actual reaped learner pid;
coordinator records also bind run/context. All433 original freeze hashes are
revalidated using the preserved old source archive for changed code and unchanged
paths for other inputs. Thus no historical source is silently reinterpreted.

Supplemental execution uses the original actual helper six times with
`--diagnostic-telemetry --execute`; source freeze and one admission-only check per
run precede it. After all six runs, the controller calls the existing official
`audit_trial(row)` for each and seals a separate audit/telemetry summary. The
current18-row performance population CLI/medians are not reused for these data.
Both final independent evidence reviews are FINAL READY. They independently
verify27supplemental index links,287frozen bytes,18observation records, identity/
receipt/audit linkage, all summarized measurements and the preserved433original
freeze identities. Root accepts the scope; no blocking findings remain.

Final acceptance is recorded in
[acceptance.json](../../runs/learning-update-optimization-v1/acceptance.json).
All24experiment parents remain healthy incomplete/update_cap, not completed
training sources or objectives. All768natural games and2160original file instances
are covered by the separately preserved24strict audits. No Source/numeric/receipt
audit was concurrent with its timing/observation population. Source code and
experimental records are locally committed/retained; nothing is pushed.
