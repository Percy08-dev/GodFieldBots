# JSON simplification and sort audit

Status: accepted2026-10-30. Independent design/correctness/conformance READY;
correctness, installed wheels, paired measurement and main environment rollout PASS.
Baseline55b38c8, native release9f59b1b2..., pinned serde_json1.0.151.

## Sort audit

Every native sort was checked for comparator, equal-key payload, uniqueness/
duplicate validation and downstream ordering. No current native caller requires
stability. This is based on valid-input semantics, not a global preference for
unstable sorting. Candidate/journal sequences are never reordered by these sorts.

| Caller | Why equal-key order is irrelevant |
| --- | --- |
| validation indexed hand/learned/usable refs | equal keys are duplicates rejected or interchangeable membership values |
| feature hand refs | instance ID uniqueness validated before keyed lookup |
| feature learned/usable refs | membership lookup; duplicates rejected |
| proof records by canonical journal path | each valid participant journal emitted once; unique path |
| episodes by directory/assigned online ordinal | validated unique directory or unique ordinal |
| index/global identity by run/game/viewer | duplicates reject before publication |
| lineage keys by seed/ancestry | tied keys union into same connected component; downstream grouping canonical |
| session collection directory | canonical directory/run duplicates reject |
| session parent generation/kind | one collection/update generation and one validation per parent; valid keys unique |
| source inventories by canonical path | duplicate aliases, if present, contain interchangeable identical path payloads |

Catalog order was already guaranteed by Serde's unique-key map and was only used
for lookup. Remove its materialized Vec/sort and use the perceived mapping key
directly; do not replace it with the definition's inner ID. Native wrappers should
name unstable behavior explicitly. Python standard sort/sorted are inherently
stable and remain standard built-ins; no replacement algorithm is introduced.
If a future source contract permits meaningful equal-key records, that caller
needs an explicit tie key or stable sorting and a behavioral regression.

## JSON audit disposition

Remove custom990-depth preflight and full-tree depth/4300digit checks by restoring
standard Serde recursion/numeric acceptance. Remove current native_json_depth
configuration rather than accept an ignored knob. New parser contract v2 records
that change; historical effective_config remains untouched. Memory/byte/record
budgets, typed domain ranges, source/learning validation and canonical hashes stay.

Keep only the reserved-number-member-key guard: arbitrary_precision's internal
number representation can reinterpret a real object as a Number before domain
validation. Escaped/nested/non-first keys must reject, identical VALUE strings may
pass. Name this narrow function for that purpose; do not retain generic preflight
or a replacement custom AST/JSON validator.

## Frozen implementation and verification

Parser serde-json-v2, pinned Serde's actual array/object recursion boundary127
accepted/128rejected. No unbounded_depth/disable_recursion_limit, custom depth
counter, native_json_depth current setting or global numeric/digit traversal.
Huge integer tokens parse under source-size admission; consumed integer DTOs
still reject out-of-range values. Candidate/journal ordering and the overflow
health-flag fix remain unchanged. Removed constructor Budget arguments and
infallible Result wrappers; clone retains boundary cancellation checks.

Native source SHA0e49078e629f9a3158406ef04dd4ee7cb3516d5468755c8f3a0a659dfeee7c9c;
release extension SHA952657c776c41c9cc509eec2a6012b4a55d839b41d8cfd5717cda86484fe531a;
wheel38418cb4365c5fad20c66abf80a87ad312524bd96d4dac6e113548a847ffb046;
Cargo.lock remains5fa272b273d397881f3e9a8d9190dae89796052482cf21105d3bc5f4de840091.

PASS: whole Python769tests,5opt-in skipped,243.67s; Rustdebug36/releasefocused17;
focused Python94/config30; whole Ruff/mypy59source modules, Rustfmt/diff checks.
Actual pinned recursion bounds, reserved-key/member-vs-value cases, historical
depth-config metadata/current-setting rejection, huge-token/domain-range and
perceived catalog key/inner-ID difference are covered. Independent correctness
and implementation-conformance delta reviews found no blocking code findings.

`runs/assessments/json-simplify-corpus.json`: both full83,783-row recipes under
original2GiB/16MiB reproduce original NPZ SHA c5c3f3c1... /841c6f885...; dataset
ab5d4913... and row-content574b81ec... unchanged; Parquet bytes retain9bce50db....
`runs/assessments/json-simplify-ppo-ingestion.json`: copied real16-game batch
produces old features/GAE/targets SHA7e2629b8... with1/2/4workers. No games/updates.
Correctness runs overlap tests and do not provide paired speed evidence.

Separate installed application/native release wheel checks PASS; all59application
Python files byte-match current source. Application wheel SHA
36a1032da95b19d2c9fe51fa5f1f7873c28d0a72bfb6b3db9594f4a802118507.
Baseline help/config PASS without native/Torch/LightGBM/PyArrow; installed CPU
MLP/LightGBM feature/score hashes equal prior installed checks. Existing verified
dependency files reused; Python dependency versions unchanged.

## Paired complete preparation

Three alternating measured pairs after one warmup per variant; same83,783-row
history dataset,2GiB/16MiB,workers1/inflight1,release builds and interpreter/
numerical dependencies. No concurrent tests/build/training. Old parser v1 uses
frozen Python source55b38c8 and loaded9f59b1b2...; newv2 uses952657c7.... Every
trial's dataset/sample identities, all4numeric arrays and NPZ archive hash match.
Records: runs/performance/json-simplify-20261030/{results,equivalence}.json and
runs/assessments/json-simplify-performance.json.

| Median / resource metric | Before | Simplified |
| --- | ---: | ---: |
| Whole wall,s |50.263|46.722|
| Process-tree CPU,s |50.138|46.616|
| Sampled per-run peak RSS,bytes |424,456,192|424,624,128|
| Sampled per-run peak PSS,bytes |396,625,920|396,404,736|
| Maximum sampled scratch,bytes |1,856,287,138|1,856,287,138|
| Internally tracked scratch peak,bytes |1,856,048,056|1,856,048,056|
| Kernel-accounted read-byte lower bound |0|0|
| Median sampled write-byte lower bound |1,943,031,808|1,943,031,808|

Reduction7.046%, ratio1.0758x. RSS is effectively unchanged at about405MiB.
Whole timings include source validation/features/staging/serialization/cleanup,
not only JSON parsing. OS samples can miss short peaks; output includes scratch.
Read bytes reflect warm page cache, not physical WSL host I/O. GPU unused; about
one CPU core occupied. This is a bounded repeated timing observation, not a
training-quality conclusion or a general performance guarantee.

Main Bots environment now loads the checked release v2 wheel; main-load SHA and
contract rechecked. Previousv1 binary remains archived under the frozen reference
native-site for reproduction. Original six fit artifact groups retain all24file
identities/small-file hashes; original arrays/models were not rewritten. Simulator
unchanged. No refit, games, optimizer updates or formal test campaign were run.
