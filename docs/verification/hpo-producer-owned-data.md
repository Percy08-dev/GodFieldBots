# Producer-owned HPO verification

Status: repaired implementation independently reviewed; whole CPU6868PASS/408SKIP.
First physical comparison is ineligible; fresh v2 physical acceptance pending.
Specification: [design](../design/hpo-producer-owned-data.md),
[ADR0043](../adr/0043-hpo-producer-owned-data.md), approved user attachment.
Base revision: `1dd51e8d1ee734b32b051556589e41d46d565ed0`.
Simulator inspected revision: `91d5926a7572db327eaea3751e7c24f428120996`;
existing corrected contract/schema pins remain unchanged.

## Baseline and environment

PASS128 in24.20s, exit0:

```console
UV_CACHE_DIR=/tmp/gfb-uv-cache-owned-preflight uv run --no-sync pytest -q tests/test_hpo_config.py tests/test_hpo_selection.py tests/test_hpo_wire.py
UV_CACHE_DIR=/tmp/gfb-uv-cache-owned-preflight uv run --no-sync ruff check src tests
```

Ruff PASS. Initial uv commands without a temporary cache fail because the home
cache is readonly; these failures are environmental and retained in the execution
ledger. No dependency synchronization/installation occurred.
Sandbox NVML is unavailable; approved host read-only `nvidia-smi` succeeds with
RTX5070Ti16GB. This is availability evidence, not a workload or memory measurement.

## Independent review and execution

Full preimplementation architecture review: READY, `/root/owned_architecture`.
The snapshot review requires physically separate inference/resume banks because
an FD exposes its complete backing. Root accepts and records that remedy;
the reviewer confirms no remaining consequential blockers in the revised design.
Final implementation correctness/conformance reviews are READY. Full project/
package checks and physical execution remain pending; no performance or full-plan
completion claim is made.

Task1 foundation source is frozen for independent scoped reviews. Root's actual
host invocation PASS52 in1.10s, exit0, including exec-inherited Unix FD transfer:

```console
.venv/bin/python -m pytest -q tests/test_ram_bank.py tests/test_hpo_owned_capture.py tests/test_hpo_capture_environment.py
```

The implementation's changed-file Ruff and mypy212 checks pass. Its earlier
bounded related regression run PASS177 precedes the final narrow strict Boolean
transaction guard; the final52above include that guard. This is foundation
evidence only: production model/run binding, direct learner, persistence, cold
resume and performance remain subsequent work.

Task1 fix round1 resolves independently reproduced prefix-reservation escape and
completion uncertainty reuse, plus local-rejection transaction numbering. Seven
RED cases become seven GREEN; final foundation60PASS in1.19s includes actual host
exec FD adoption. Changed Ruff and mypy212PASS. Correctness/spec/quality and
architecture conformance both READY on the final staged delta. Reviewer independent
checks pass50non-exec cases and8covering regressions respectively. Root accepts
this foundation only; production ownership binding remains required downstream.

Task2 direct components now pass both final scoped independent reviews. The
core corrections make token bindings immutable, distinguish deferred live exports
from uncertain closure, share the Source account and refuse legacy-origin mixing.
The wrapper correction inventories received originals before fallible decoding/
retention and retires local failures without ACK, keeping sender leases until
completion/reap. Common inference-only actor capabilities preserve dependency
direction and cache/binding last-use ownership.

Final covering HOST167PASS/51.53s and Ruff/mypy219PASS follow the receive fixes.
Independent final correctness17PASS/18.90s includes11custodyfaults and actual
Wire7/actor9 success for each model; architecture7PASS/4deselected rechecks the
failure delta. Earlier all3-model strict-reader feature/target/gradient/Adam/
shuffle/Torch-RNG tests and mixedzero/seat2/loss/draw cases pass. Prior broad420PASS
and affected257PASS runs are historical pre-final-delta evidence, not a new whole
suite claim. Production coordinator/registry, cold resume, campaigns and GPU/
timing trials remain subsequent work.

Task3 fresh integration is independently READY. Production collector composition
with actual CPU actor9/Wire7/GroupPack callbacks completes16games per batch and
two updates; all3models retain historicalG0 as opponent afterG1 and pass actual
EVICT/pin/unbind/rebind. Final initial integration100PASS/75.39s and prior
257coveringPASS precede the accepted safety corrections.

The consolidated correction restores original G0/resume-pair retention, staged
directory/source-FD deletion with ctime fences, preallocation and shared parent
admission, plus failed-start cleanup. Foreign replacement/symlink injections
preserve both foreign and original backup files with no false deletion receipt.
The covering correction run passes192tests/223.26s. A subsequent direct-creator
startup bypass required the repository's advisor escalation: actual creator
execution now requires its admitted one-shot original task/coroutine; unsettled
close retains ownership. Final independent startup/Wire7/custody/kernel checks
PASS36/29.54s and11/1.84s; architecture8targetedPASS. Both final reviews are READY,
Ruff/mypy223PASS, frozen8 unchanged. These are controlled, unqualified CPU fixtures.

Milestone packaging builds wheel/sdist with223Python modules matching source
exactly, then installs the wheel with only8mandatory dependencies. Isolated CLI
help and normal random-worker startup/SIGTERM/reap/empty-health/lease retirement
PASS on Python3.14.10 with GIL and no Torch/Optuna/Arrow/LightGBM/native. This build
precedes the final startup refinement; final-source package checks remain required.
Root's first smoke incorrectly required exit0; investigation confirms intentional
normal SIGTERM(-15), and the corrected contract-based check passes. No production
change was made for that check.

Task4 cold import, immutable final-policy completion handoff, version-aware strict
audits and campaign/evaluation integration pass final whole-milestone correctness
and architecture reviews. The importer reads required controls/checkpoint/RNG/pool
once without replaying unused historical samples. Inference-only evaluation owns
its separate budget/deadline; actual commit and cleanup issue completion evidence.
Legacy inputs retain their strict readers and original meanings.

Final review corrections cover partial-bank cleanup preserving primary errors,
decoded-array lifetime accounting through aliases/tracebacks, explicit registry6
retirement-journal pointers, bounded archive namespace/control guards, same-parent
comparison deduplication and conservative absent capture hooks in legacy stubs.
Independent HOST13PASS/47.59s includes actual pruned pause4/resume5/full audit;
architecture4PASS/7deselected and diff check PASS. Earlier scoped evidence includes
215owned tests, 73evaluation tests and342legacy tests; those are bounded runs,
not whole-project evidence. Save phase honors the original configured1GiB while
each bank and numeric retain512MiB limits, verified on actual CPU pause/resume.

Whole CPU attempt01 stops at349PASS/1FAIL in8.52s on the absent legacy capture
attribute. The implementation fixes that compatibility issue without changing the
original fixture assertions. Attempt02 PASS6846/SKIP408/33warnings in1740.00s,
exit0, against the frozen corrected source. Root independently checks its JUnit:
7254cases, zero failures/errors. A60s faulthandler diagnostic at99% identifies the
existing slow `original_tree_sources` fixture; execution continues and passes.
A preceding whole-fix covering run had212PASS/13existing
SKIP/1new-test-setup failure; the hook was moved after the actual pause fixture,
and that corrected node passes separately. Neither failed invocation is counted
as a green full run.

```console
CUDA_VISIBLE_DEVICES='' GF_RUN_CUDA=0 .venv/bin/python -m pytest -q --ff --maxfail=1 --tb=short -o faulthandler_timeout=60 --junitxml=runs/hpo-producer-owned/assessment/v1/checks/whole-cpu-02.xml
```

Frozen-source Ruff `src tests` PASS and mypy PASS225. Repository-wide Ruff also
examines the user's pre-existing untracked `codex_analysis.py`, which retains four
unrelated findings (FURB162, UP017 and two SIM102); that file is preserved outside
this change. Both staged and unstaged diff whitespace checks PASS.

Current frozen source builds wheel/sdist with225Python modules, each matching
source byte-for-byte. Reinstalled wheel passes isolated CLI help and actual normal
random-worker startup/close/reap with released lease, empty ledger/health and
retired I/O thread. The temporary mandatory-only environment has no optional ML
packages/imports. The first smoke assertion called the health dictionary; the
corrected property assertion passes without a production change. No new project
dependency was added: the already-declared build backend was provisioned in a
separate temporary build environment.

```console
UV_CACHE_DIR=/tmp/gfb-uv-cache-owned uv build --package godfieldbots --wheel --sdist --offline --no-sources --no-build-isolation --python /tmp/gfb-owned-build-env/bin/python --out-dir /tmp/gfb-owned-task4-packages
UV_CACHE_DIR=/tmp/gfb-uv-cache-owned uv pip install --python /tmp/gfb-owned-mandatory-env/bin/python --no-deps --offline --no-index --reinstall /tmp/gfb-owned-task4-packages/godfieldbots-0.1.0-py3-none-any.whl
/tmp/gfb-owned-mandatory-env/bin/python -I -m gf_cli --help
```

Correspondence and installed-baseline observations are in
`runs/hpo-producer-owned/assessment/v1/checks/task4-package-correspondence.json`
and `task4-installed-mandatory.json`. These checks establish packaging and optional
import isolation; they are not learning-quality or performance evidence.

## First physical set and corrective verification

At measured revision `64673696aa697fffb105c3d4db0b9f6b93e0bae6`, all3 real CLI
campaigns complete proposal/control training and evaluation. All6 exactD64 parents
pass independent full-original strict audits; a later-session MLP-versus-GRU
comparison passes strict saved-evaluation and actual termination validation with
2natural games. The separate9fresh/pause/resume invocations and18timing trials
also finish with their expected operational status and actual cleanup receipts.
This does not by itself qualify the generated data or accept performance claims.

The subsequent strict2GiB sample audit exposes two defects. First, the offline
audit charges an absolute numeric high-water request twice and retains decoded
cohort charges across batches. Its narrow repair charges positive peak growth,
retires actual graphs before refund, preserves metadata/physical custody and
holds the real pool close through failure arrays/tracebacks. Existing19non-audit
definitions remain AST-identical. Both independent reviews are READY; targeted
real-owner RED3->GREEN3, HOST45covering and4strengthened lifetime checks pass.

After that repair clears the original budget boundary, strict correspondence
correctly rejects a zero-Act participant in batch1: its real journal bank UUID
differs from capture's fabricated empty-bank UUID. The environment now registers
the actual readonly empty bank after successful header construction. Capture
requires both real identities and cannot mint a fallback at seal. Actual-header
RED->GREEN, HOST113covering, final52lifetime/guard checks, frozen-kernel/legacy/
strict-audit3checks and both independent reviews pass. No reader is weakened and
no saved parent is rewritten. The entire affected v1 C parents and comparison
remain historical/ineligible; operational receipts keep their original meaning.

Evidence is preserved in `runs/hpo-producer-owned/assessment/v1/`, including
`final-disposition.json`. Fresh v2 roots and UUIDs are declared under
`assessment/v2/`. Every completed two-update functional parent must receive a
full public strict2GiB original-sample audit before new timing trials begin.
The new thin telemetry includes actual participant creation and both-viewer
coverage. Its independent review is READY:28restored targets,55negative cases
rejected, populated18/zero-Act controls and all6fixed configs pass.

The repaired whole CPU03 invocation passes6868/SKIP408/33warnings in1731.36s,
exit0. Root independently checks JUnit7276cases/zero failures/errors. Source/tests
remain frozen throughout the run; the earlier6846PASS run is separate historical
evidence. Final Ruff `src tests`, mypy225 and diff checks pass. Rebuilt wheel and
sdist match all225source modules byte-for-byte. Reinstalled mandatory-only wheel
passes isolated CLI help and actual baseline startup/SIGTERM/reap/lease/empty
health/thread retirement with no optional ML imports.

```console
CUDA_VISIBLE_DEVICES='' GF_RUN_CUDA=0 .venv/bin/python -m pytest -q --ff --maxfail=1 --tb=short -o faulthandler_timeout=60 --junitxml=runs/hpo-producer-owned/assessment/v2/whole-cpu-03.xml
```

## Required matched comparison

Preserve explicit ADR0042-enabled control alongside the new producer-origin path.
192width/2layers, seed17,4workers/2concurrentgames,16games x2updates, original
decision/update/time/memory budgets. Three alternating pairs per model,18trials.
New directories only. Keep encoder/protocol/origin/config differences explicit.
No audit/test workloads during measurement. Retain failures and measured limits.
Preserve the historical probe's D1,000,000/max_updates2,600s wall,epoch1 and
parent/save1GiB settings, including base480s admission/540s execution/60s
finalization. Expected physical comparison result is healthy
`incomplete/update_cap`; it never supplies a completed HPO objective.

Before candidate measurements, root declares this18trial paired order. B is the
current-code ADR0042-enabled strict resident control; C is producer-owned.
Actual new-directory/run/config/source identities must be frozen before launch.

| Round | Model | Pair order |
| --- | --- | --- |
| 1 | MLP | B1,C1 |
| 1 | GRU | C1,B1 |
| 1 | Transformer | B1,C1 |
| 2 | MLP | C2,B2 |
| 2 | GRU | B2,C2 |
| 2 | Transformer | C2,B2 |
| 3 | MLP | B3,C3 |
| 3 | GRU | C3,B3 |
| 3 | Transformer | B3,C3 |

The following values are historical ADR0042 evidence from
[the investigation](../design/hpo-producer-owned-data-investigation.md),
not a newly rerun control. MLP is3trial medians; GRU/Transformer are1trial.
Independent component medians need not sum to the whole median.

| Whole execution phase, seconds | Historical MLP | Historical GRU | Historical Transformer | New matched control/candidate |
| --- | ---: | ---: | ---: | --- |
| Startup/other (historically combined) | 1.88 | 1.75 | 1.80 | NOT RUN |
| Collection, two batches | 48.35 | 48.59 | 49.31 | NOT RUN |
| Feature/data preparation | 19.02 | 19.24 | 19.35 | NOT RUN |
| Update including old runtime parity | 17.04 | 54.62 | 54.95 | NOT RUN |
| Save | 2.55 | 2.71 | 2.89 | NOT RUN |
| SYNC | 9.90 | 10.53 | 10.92 | NOT RUN |
| Cleanup | 9.16 | 9.50 | 9.83 | NOT RUN |
| Whole | 107.76 | 146.94 | 149.04 | NOT RUN |
| Runtime parity separately | not separately stated here | not separately stated here | not separately stated here | NOT RUN |
| New capture/transfer separately | not applicable | not applicable | not applicable | NOT RUN |

| Collection detail, historical MLP second batch | Seconds | New control/candidate time and calls |
| --- | ---: | --- |
| Before collector | 5.54 | NOT RUN |
| Collector entry | 0.04 | NOT RUN |
| Before output creation | 0.07 | NOT RUN |
| Owned archive path setup | 5.50 | NOT RUN |
| After games | 5.47 | NOT RUN |
| Remaining startup/candidates/inference/communication/storage | 13.45 | NOT RUN |
| Whole second collection | 30.08 | NOT RUN |
| Capture / tokenize / seal | not separately stated | NOT RUN |
| Candidate generation / inference / IPC | not separately stated | NOT RUN |
| Archive queue / write | not separately stated | NOT RUN |

New reports must distinguish exclusive parent wall spans from inclusive concurrent
operation distributions. Moving work between collection/preparation/update does
not by itself establish whole-execution improvement.
