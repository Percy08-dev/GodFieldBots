# Producer-owned HPO verification

Status: ACCEPTED by root after implementation/project/physical verification and
both final independent whole-branch/evidence reviews READY. No unresolved findings.
Acceptance covers the approved functional and short-protocol performance scope;
no training-quality or final-test claim follows.

Specification: [design](../design/hpo-producer-owned-data.md),
[ADR0043](../adr/0043-hpo-producer-owned-data.md), approved user attachment.
Base revision: `1dd51e8d1ee734b32b051556589e41d46d565ed0`.
Accepted physical source: `a15391c807f547eeb99a0e5dc7b2050d8fb7499c`.
Simulator: `91d5926a7572db327eaea3751e7c24f428120996`; its rules, binary and reviewed schemas
remain unchanged. Installed native SHA256:
`0d5d0b242b1a08aafa22039f6c7c5921f2555cfcbf7c801c899e14573054aeda`.
Freeze SHA256: `e27a25c7a2a70a238910bf58463aceb56b02ee676e1df7a2942a464cdf8306a0`.
The freeze binds321source/project files,5helpers,21config/declaration files,
all18materializations, actual native and Simulator identities.

## Implemented boundaries

The explicit resident schema4 route covers MLP, GRU and Transformer fresh and
same-code pause/resume. Capture binds the actual run/plan/generation/snapshot,
participant and explicit learner seat. It registers each actual readonly empty
bank after successful journal-header construction, including zero-Act episodes.
Only actual positive command ACKs accept decisions once; both real terminal
observations and positive episode/group/process cleanup complete a batch.

Sealed Linux memfd NumPy banks move over actual owned Unix channels. Sender
buffers/admission survive cancellation until actual completion ACK or child reap.
Inference and resume have distinct backings; actors receive inference only.
Archives/numeric persist before update. One immutable updated snapshot supplies
both saves and RAM SYNC; actual writer ACKs and live registry/pins govern commit.
Normal updates do no historical sample replay, generated semantic readback or
whole-batch parity. New parity is `not_run`, never invented zero error; captured
`replay_reconciled` stays false even after a later explicit audit.

Cold import reads required control/checkpoint/RNG/pool once and never replays
unused samples. Completion/evaluation handoff requires the actual final commit
and cleanup. A new evaluation job owns independent inference-only data/deadline;
JSON does not restore execution rights or reopen a closed training job.

Formats: spec4/run `hpo-ppo-owned-v1`, Collection8/Proof5/Numeric6, learner Wire7,
actor inner9, Update6; management generation5/registry6/cache-transaction2. Model4,
Journal6, inference5/resume5 and array layouts retain their meanings. Legacy
strict/control readers remain explicit. C selects canonical Python encoding;
B preserves spec3/rust with ADR0042 reuse enabled. No rejection-based fallback,
Simulator copy/change, Rust change, dependency, reward or PPO math change occurs.
Policy visibility remains the acting participant's approved observation/events,
typed actions and permitted same-revision atomic hints; no SequenceOptions,
rejection oracle, hidden/RNG/future data enters policy features.

## Project and numerical checks

| Check | Result and scope |
| --- | --- |
| Full CPU03 | 6868PASS /408SKIP /33warnings,1731.36s,exit0; JUnit7276cases,0failures/errors |
| Ruff | `src tests` PASS |
| mypy | PASS225source files |
| Whitespace | staged/unstaged diff checks PASS |
| Fixed computation | all3models strict-reader features/candidate order/endpoints/GAE/targets/gradients/Adam/shuffle+Torch RNG comparisons PASS |
| Kernel preservation | all8frozen kernel ASTs unchanged |
| Packaging | wheel/sdist225Python modules match source bytes exactly |
| Mandatory-only installed wheel | isolated CLI help and real random-worker startup/SIGTERM/reap/lease/empty-health/thread retirement PASS; no optional ML imports |
| Independent scoped reviews | correctness/spec/quality and architecture conformance READY for all milestones and both real-data repairs |

```console
CUDA_VISIBLE_DEVICES='' GF_RUN_CUDA=0 .venv/bin/python -m pytest -q --ff --maxfail=1 --tb=short -o faulthandler_timeout=60 --junitxml=runs/hpo-producer-owned/assessment/v2/whole-cpu-03.xml
.venv/bin/ruff check src tests
.venv/bin/mypy src
```

The whole run preserves ordinary fixtures/skips and freezes source/tests/HEAD
throughout. Earlier CPU01 fails349PASS/1FAIL on an absent optional capture hook;
CPU02 passes6846/408 before the real-data repairs. Neither is relabeled as final
source evidence. Whole CPU03 is the repaired-source run. No Rust source changes
require a new native build. Source-matched installed native identity is frozen.
Repository-wide Ruff additionally sees4pre-existing findings in the user's
untracked `codex_analysis.py` (FURB162,UP017,twoSIM102); that file is untouched.
Initial sandbox inherited-socket EPERM and readonly home-cache errors are
preserved; actual process checks run HOST and uv uses a temporary cache.

## Fresh v2 physical protocol and original-data audits

Runtime: Python3.14.10 with GIL,
Torch2.14.1+cu132 /CUDA13.2,
NVIDIA GeForce RTX 5070 Ti. Exact runtime/device/boot conditions remain
in request/result files and are compared across trials.

All9actual functional invocations pass: each model has fresh2updates and a
separate pause1/resume2 parent. All6two-update parents receive a full public
strict2GiB original audit BEFORE any timing begins. The original600/480/540s
wall/admission/execution deadlines survive resume unchanged.

All18fresh timing trials pass, without retry/replacement, in the declared
alternating order: round1 MLP B,C /GRU C,B /Transformer B,C; round2 MLP C,B /
GRU B,C /Transformer C,B; round3 MLP B,C /GRU C,B /Transformer B,C.
Fixed conditions:192width/2temporal layers,seed17,4workers,2concurrent games,
16games per collection x2updates,epoch1,D1,000,000,600s wall,original memory/save
limits. These parents stop healthy `incomplete/update_cap`, not exact-D
completion, and create no completed HPO objective. There are no audits/tests
alongside measurement. Only saved thin telemetry is checked between trials.

Independent validation passes all24benchmark/functional parents with public
`validate_hpo_parent(...,require_complete=False,max_owned_bytes=2*1024**3,
deadline=monotonic()+120)`: original archives/journals/BOTH Replay views, numeric
features/correspondence, checkpoints, registries, pins and retention. All24 have
2updates/3generations. All27actual invocation termination receipts pass fixed
namespace/context verification. Three pause/resume deadline pairs match. The
same analyzer exactly reproduces18eligible trials/9pairs under the actual common
freeze, materializations and chronological nonoverlapping execution. All18phase
partitions reconcile with zero observed error; nested spans are not added.

Latest-source CLI campaigns also pass for all3models: proposal plus canonical
control training/evaluation (6completeD64/1update parents,6seed evaluations).
All6parents pass independent strict original audits and actual training receipt
checks. A later-session MLP-versus-GRU final-policy comparison passes strict saved
schema7 evaluation with2natural games/no missing Replay, exact final references
and actual worker-reap termination evidence. All6saved seed evaluations also pass strict saved-data/runtime validation with
2natural games/no missing Replay each, and their actual termination receipts pass.
Together all40actual v2job receipts are independently verified. These small exploratory panels establish lifecycle
behavior, not strength/convergence/formal/final64/Dataset qualification.

## Matched results: whole execution and collection detail

B=ADR0042-enabled strict resident control; C=producer-owned canonical Python.
Each cell is3trial data under the same fixed conditions. The percentage below
is reduction of the two arm medians, not the median of pair percentages.

| Model | Whole helper B -> C | Reduction | Full process B -> C | Reduction |
| --- | ---: | ---: | ---: | ---: |
| mlp | 108.053 -> 38.245s | 64.61% | 108.927 -> 39.131s | 64.08% |
| gru | 146.989 -> 53.739s | 63.44% | 147.861 -> 54.600s | 63.07% |
| transformer | 148.065 -> 54.458s | 63.22% | 148.962 -> 55.303s | 62.87% |

Whole execution: median seconds across3trials/arm/model. Independent component medians need not sum to whole median.

| Metric | MLP B | MLP C | GRU B | GRU C | Transformer B | Transformer C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Startup | 2.136 | 2.064 | 2.201 | 2.061 | 2.230 | 2.129 |
| Collection, two batches | 48.102 | 27.239 | 48.342 | 26.818 | 49.234 | 27.942 |
| Features/data preparation | 18.903 | 1.334 | 19.288 | 1.324 | 19.351 | 1.352 |
| Learner command (includes child save/parity/transfer) | 17.876 | 5.753 | 54.625 | 21.260 | 54.229 | 21.126 |
| Parent control persistence/commit | 2.050 | 0.076 | 2.204 | 0.073 | 2.240 | 0.080 |
| Actor synchronization | 8.310 | 0.532 | 8.990 | 0.706 | 9.194 | 0.677 |
| Cleanup | 9.267 | 1.295 | 9.678 | 1.360 | 9.835 | 1.318 |
| Other | 1.594 | 0.039 | 1.678 | 0.047 | 1.713 | 0.041 |
| Whole helper | 108.053 | 38.245 | 146.989 | 53.739 | 148.065 | 54.458 |
| External process, including interpreter/final report | 108.927 | 39.131 | 147.861 | 54.600 | 148.962 | 55.303 |

Whole helper excludes interpreter startup and final report serialization/fsync. External process wall is separately recorded. Each individual phase partition reconciles; medians do not form a partition. Archives are written inside collection, numeric inside preparation, checkpoint save inside the learner command, and parent commit in its own row. Child/kernel durations below overlap those rows and must not be added.


Nested costs: median sum of the named observed spans per trial, seconds. Inclusive/concurrent; NOT additional exclusive wall.

| Metric | MLP B | MLP C | GRU B | GRU C | Transformer B | Transformer C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Old parity | 9.3893 | NOT RUN | 30.2400 | NOT RUN | 30.1507 | NOT RUN |
| Child checkpoint save (G0 plus2updates) | 0.3079 | 0.1069 | 0.3471 | 0.1275 | 0.4122 | 0.1544 |
| Capture hooks (inside collection) | N/A | 2.0791 | N/A | 2.0809 | N/A | 2.0791 |
| Owned numeric construction | N/A | 0.9581 | N/A | 0.9646 | N/A | 0.9763 |
| Tokenize (inside construction) | N/A | 0.6909 | N/A | 0.6912 | N/A | 0.7032 |
| Seal (inside construction) | N/A | 0.0401 | N/A | 0.0396 | N/A | 0.0408 |
| Numeric write (inside preparation) | N/A | 0.3460 | N/A | 0.3604 | N/A | 0.3640 |
| Snapshot construction (G0 plus2updates) | N/A | 0.1600 | N/A | 0.1751 | N/A | 0.2019 |
| Batch FD send observed span | N/A | 0.0228 | N/A | 0.0222 | N/A | 0.0232 |
| Snapshot receive observed span (G0 plus2updates) | N/A | 0.0461 | N/A | 0.0382 | N/A | 0.0421 |
| Input-bank completion ACK wait (includes update/sync) | N/A | 5.4505 | N/A | 20.9422 | N/A | 20.7548 |

Collection detail: sequential checkpoint/live-guard wall, seconds. B historical validators and C live guards are different operations at corresponding positions; each position6calls across3trials.

| Metric | MLP B | MLP C | GRU B | GRU C | Transformer B | Transformer C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| before_collector (2calls/trial) | 5.5773 | 0.0010 | 5.6779 | 0.0010 | 5.7374 | 0.0011 |
| collector_entry (2calls/trial) | 0.0660 | 0.0009 | 0.0704 | 0.0009 | 0.0713 | 0.0009 |
| before_output_mkdir (2calls/trial) | 0.1158 | 0.0010 | 0.1233 | 0.0011 | 0.1268 | 0.0010 |
| archive_path_setup (2calls/trial) | 5.4973 | 0.0008 | 5.5832 | 0.0009 | 5.6332 | 0.0008 |
| after_games (2calls/trial) | 5.4963 | 0.0017 | 5.5960 | 0.0018 | 5.7037 | 0.0018 |
| Parent collection1 | 17.947 | 15.453 | 18.036 | 15.390 | 18.430 | 16.275 |
| Parent collection2 | 30.121 | 11.446 | 30.306 | 11.476 | 30.863 | 11.599 |

C invokes no B legacy checkpoint validator. The Source dispatcher can wrap a C live guard; its time overlaps the guard and is not added. Capture/tokenize/seal and transfer are in the nested table above.


Concurrent operation distributions: pooled p50/p95 milliseconds, with sample counts. Per-decision timings may overlap; archive rows use only each collection final cumulative total (6samples/arm/model), never sum cumulative group snapshots.

| Metric | MLP B | MLP C | GRU B | GRU C | Transformer B | Transformer C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Candidate generation | 1.223/23.060 (n=1833) | 1.219/22.980 (n=1833) | 1.230/22.938 (n=1833) | 1.215/23.114 (n=1833) | 1.226/23.370 (n=1809) | 1.220/23.247 (n=1809) |
| Inference | 10.911/52.566 (n=1833) | 10.475/51.999 (n=1833) | 10.888/52.807 (n=1833) | 10.503/51.669 (n=1833) | 11.752/53.857 (n=1809) | 11.474/53.143 (n=1809) |
| Communication/decode host | 3.640/44.071 (n=1833) | 3.748/44.597 (n=1833) | 3.680/42.205 (n=1833) | 3.748/43.365 (n=1833) | 3.637/46.167 (n=1809) | 3.785/48.042 (n=1809) |
| Archive writer queue | 10.267/30.554 (n=6) | 0.131/20.444 (n=6) | 10.356/20.417 (n=6) | 0.134/30.715 (n=6) | 0.164/0.188 (n=6) | 0.122/0.146 (n=6) |
| Archive encoding | 52.887/60.590 (n=6) | 41.155/44.358 (n=6) | 48.543/66.712 (n=6) | 42.162/56.754 (n=6) | 151.766/279.244 (n=6) | 47.399/112.762 (n=6) |
| Archive write | 89.221/96.585 (n=6) | 88.461/88.951 (n=6) | 95.934/112.093 (n=6) | 88.102/104.207 (n=6) | 99.050/174.028 (n=6) | 91.313/111.111 (n=6) |

| Model/arm | Used decisions by trial | Collected decisions by trial | Game actions by trial | Games/natural |
| --- | --- | --- | --- | --- |
| mlp B | 314, 314, 314 | 314, 314, 314 | 611, 611, 611 | 96/96 |
| mlp C | 314, 314, 314 | 314, 314, 314 | 611, 611, 611 | 96/96 |
| gru B | 314, 314, 314 | 314, 314, 314 | 611, 611, 611 | 96/96 |
| gru C | 314, 314, 314 | 314, 314, 314 | 611, 611, 611 | 96/96 |
| transformer B | 310, 310, 310 | 310, 310, 310 | 603, 603, 603 | 96/96 |
| transformer C | 310, 310, 310 | 310, 310, 310 | 603, 603, 603 | 96/96 |


The learner row measures the full parent command, including child update/save,
old parity or new transport. Child save spans include G0 and both updates; G0 is
inside startup. GPU-kernel-only wall is not separately measured. New capture,
tokenize/seal, numeric write, snapshot and observed send/receive costs are shown;
`batch_completion_wait` waits for input-bank completion ACKs after sending the
banks. The child receives the input, updates the learner, synchronizes GPU work
and releases input views before emitting those ACKs. Snapshot construction and
checkpoint save follow ACK emission. This is not a save-completion wait or pure
transport time; parent scheduling/ACK-observation latency can also be included.
The callback row covers both actual participant creations, including zero-Act.
These scopes expose moved work; whole-helper AND full-process wall improve.

### Collection wall coverage and remaining measurement gaps

The operation distributions above are not a complete additive decomposition of
collection wall. Most new-route collection time lies inside the game-operation
spans. To make coverage explicit, the following is an interval-union partition
of the first actual MLP pair (trials1/2), not medians. Priority is the five
checkpoint/live-guard positions, then any active game span, then archive spans
outside games, then uncovered wall. Concurrent and nested spans are counted once.

| Collection wall category, both collections, seconds | MLP B trial1 | MLP C trial2 |
| --- | ---: | ---: |
| Five checkpoint/live-guard positions | 16.673644 | 0.005576 |
| At least one game-operation span active, excluding checkpoints | 29.449060 | 26.372873 |
| Archive call spans outside the above intervals | 1.197185 | 0.085262 |
| Remaining collection setup/finalization wall, not individually timed | 0.867813 | 0.775545 |
| **Total collection wall** | **48.187701** | **27.239255** |

A game span includes runtime evidence checks, Create/Join, actor initialization
or reuse, Observe/Events/ActionOptions, decision processing, Act/ACK, terminal
observations, Replay fetch/persistence, journal finalization, logical worker close
and Delete. Existing timers do not split all those operations individually.
The uncovered outer interval also includes such work as group/server setup,
attestation/catalog and collection finalization; no exclusive substage allocation
is inferred from code inspection alone.

For C trial2, summing the32game spans gives49.258232seconds of cumulative game
duration because games overlap. Its recorded decision-processing durations sum
to26.067557seconds; the remaining23.190675cumulative game seconds are outside that
timer. They cannot be assigned exactly among HTTP, initialization, Replay and
finalization from existing data. These cumulative durations must not be added
to the27.239255seconds of parent wall. Additional substage instrumentation is
needed to explain their individual costs. Input-bank ACK wait belongs to the
later learning-update phase and explains none of this collection remainder.

The follow-up [collection latency investigation](hpo-collection-latency.md) now
measures those client-side substages on a fresh same-condition MLP run. It
reconciles26.700208seconds of collection wall and48.180624cumulative game seconds,
leaving0.849857cumulative seconds explicitly unseparated. It reports observer
effect and separates per-game duration, concurrent wall and server-log limits.

### Individual accepted physical trials

| Index | Model/arm | Run UUID | Helper seconds | Process seconds |
| --- | --- | --- | ---: | ---: |
| 1 | mlp B | `ca2260ca-f9e3-4caa-86be-3ec728a8a62b` | 107.882 | 108.737 |
| 2 | mlp C | `f0036830-ec4b-4f19-9127-78f3104063a6` | 38.245 | 39.131 |
| 3 | gru C | `d97605d4-be51-447d-bed9-20088d589ae3` | 53.579 | 54.557 |
| 4 | gru B | `b43b0edc-7420-4ec3-89fd-ccbc7cc32df1` | 146.681 | 147.540 |
| 5 | transformer B | `1ea341d7-8e28-41fa-9e09-d8866373e57c` | 147.790 | 148.656 |
| 6 | transformer C | `8e21f7b3-6905-4754-8b4b-159cf18ecc9b` | 54.396 | 55.258 |
| 7 | mlp C | `13f9bf7a-98dc-4492-a3b6-681093f7a702` | 37.897 | 38.746 |
| 8 | mlp B | `0fc17131-53b7-47b5-aea2-9a86996478e0` | 108.053 | 108.927 |
| 9 | gru B | `b46d1dd4-5db6-41b2-8529-6e01066eb2f7` | 147.462 | 148.292 |
| 10 | gru C | `089182b6-c785-4cd0-b623-1b0c0c85c9e3` | 54.133 | 54.987 |
| 11 | transformer C | `2268923c-10ad-4460-aa61-2489c4089941` | 54.794 | 55.648 |
| 12 | transformer B | `cf9cf0c2-5ace-430c-aa7b-dd09b36bde5b` | 148.065 | 148.962 |
| 13 | mlp B | `bdc124d7-5fc6-4e3c-952b-9c40a6ddcb29` | 108.218 | 109.069 |
| 14 | mlp C | `6834c7a2-50bc-4b6a-bdf8-380a338c86a7` | 38.481 | 39.323 |
| 15 | gru C | `590c8e93-9579-487e-9bca-5fb78506cafc` | 53.739 | 54.600 |
| 16 | gru B | `67dc8280-6dc0-4e19-876b-b76f864458bc` | 146.989 | 147.861 |
| 17 | transformer B | `a5902260-e1a2-4f63-881d-461264726d6e` | 148.225 | 149.104 |
| 18 | transformer C | `e4101545-6e63-4c08-9ac5-92c4e45d2a65` | 54.458 | 55.303 |


## Preserved first-set failures and corrective evidence

The first set at6467369 completes its operations but is NOT accepted as a
comparison. Strict postrun2GiB auditing first finds additive charging of an
absolute numeric high-water request and missing decoded-cohort retirement.
A narrow offline-audit correction admits positive peak growth, retires actual
records/arrays before refund, preserves control/physical custody, and retains a
real pool close hold through failure arrays/tracebacks. All19non-audit existing
definitions remain AST-identical; real-owner RED3->GREEN3, covering45HOST checks
and strengthened4lifetime checks pass. No normal update/cold path changes there.

Once that resource boundary clears, the strict reader exposes a zero-Act bank
UUID mismatch in batch1/index12 `b000001-p6-seat0`. The journal holds the real bank
ID while seal generated a different empty-bank ID. The environment now passes
its actual readonly empty view after header success; capture reserves before
cloning, binds episode/game/actor/layout/initial prefix, rejects duplicate/foreign
registration and never invents a missing identity. Valid non-UUID IDs and actual
nonempty fixture frames keep their original contract. Actual-header RED->GREEN,
113covering, final52guards and independent45numerical/legacy/kernel checks pass.
Both repairs pass separate preimplementation and final conformance/correctness
gates. No reader, persisted format, limit or data file is weakened/rewritten.

The entire affected first-set C parents/comparison stay historical/ineligible.
Their original operational results and actual cleanup receipts remain unchanged;
no interrupted/failing data is converted into wins/losses or qualification.
`assessment/v1/final-disposition.json` records this distinction. Fresh v2 IDs,
roots, source freeze, full functional sample audits and18newmeasurements provide
the accepted population. The old timings are not reused as final results.

## Decisions and limits

Root used the supplied writable feature checkout because the created isolated
worktree was readonly. Cost if wrong: edit overlap could affect unrelated work;
tracked scope and pre-existing user files were preserved and checked. Root treated
the disposable timing harness as local instrumentation requiring independent
correctness review rather than another architecture gate. Cost if wrong: a
production-boundary change could escape review; helpers were isolated from source,
actual ownership stayed production-controlled, and their scope/hash/call behavior
passed independent review. These are the two recorded execution rulings.

### Preservation and reproducibility

Managed limits stay Source2GiB/numeric512MiB/collector2GiB,save+parent1GiB and
24GiBRAM/12GiBCUDA globally. These are managed admissions, not physical RSS.
Process/invocation samples and writer reservation peaks are not simultaneous
whole-device maxima. Whole-device high-water is unavailable; no memory-reduction
percentage is claimed. No swap/spool or smaller budget/workload is substituted.

Three pairs/model, one seed and two updates support descriptive route-level
timing only. Spec/origin/encoder/protocol differences are explicit; this does not
isolate a single micro-optimization or establish long-run capacity, training
quality, formal HPO sizing or final-test performance. A seed alone does not promise
deterministic training. All model/checkpoint, code/native/Simulator/schema/config
identities, deadlines/seeds and actual receipt conditions remain recorded.

All24 protected historical files match pre-run SHA256,size,device,inode,mtime,
ctime after measurement. Original user untracked work is untouched. No Simulator,
engine, Rust, dependency or unrelated file is edited; no push/publication occurs.
The initial wrong CLI namespace refusal and sandbox/test-setup failures remain
historical evidence, not successful runs.

Artifacts: `runs/hpo-producer-owned/assessment/v2/` contains final-freeze,
frozen-plan,matched-analysis,report-table-metrics,all immutable requests/results/
actual-termination records, bounded raw timings/diagnostics, external wall records,
full CPU XML/log, exact package and mandatory-only checks. Physical originals live
under `runs/hpo-producer-owned/{v2,functional-v2,cli-v2}/`.
The preparation document retains its original preparation-stage label; actual
frozen identities and the separately verified combined analysis govern reported
conditions. Saved JSON is diagnostic evidence, never execution/cleanup authority.
