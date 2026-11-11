# Pack checksum read policy verification

Status: root accepts the implemented scoped engineering work,2026-11-11. Decisions:
[ADR0035](../adr/0035-pack-checksum-read-policy.md),
[ADR0036](../adr/0036-packed-decoded-source-retirement.md), and
[the current design](../design/pack-checksum-read-policy.md).

## Scope, identities and review

Baseline Bots `f59a2abc360e7e3d587841edcd55ecd839ab3629`; design milestone221c2fd.
Simulator remained read-only at91d5926a7572db327eaea3751e7c24f428120996; current
07§21.1 was read. CPython3.14.10/GIL enabled, SQLite3.54.1. SQLite1/Pack1/
descriptor1/proof2/Run4/Update4/Dataset4/provider1/plan4/wire5 remain unchanged.
New evidence uses custody2/pack-checksum-v1. Existing native capability values,
learning inputs/rewards/splits/scalar budgets and the directory default remain.
No migration, deletion, formal campaign, HPO, final test or qualified real-data fit.

Independent architecture READY preceded implementation and both diagnostic and
ADR0036 amendments. Initial conformance found fresh Dataset4v1 emission, late
native gates and valid-pruning audit refusal; correctness found a constructor
cleanup/reservation violation. All were fixed and independently re-reviewed READY.
ADR0036 reviews additionally found shared-owner metadata-floor and owned-pool
close funding gaps. Both are fixed; the advisor confirmed that pool close refunds
only its base and nested budgets are borrowers. Final conformance and independent
correctness are READY, including the strengthened accounting test-only correction.
There are no unresolved or parked blocking findings.

All detailed commands, source manifests, raw results and failures are retained at
`runs/assessments/pack-checksum-read-policy-20261111/`. The implementation report
is `.superpowers/sdd/pack-checksum-implementation/task-1-report.md`. Recorded run
code identities retain their actual dirty-source identities; saved records were
not rewritten to pretend they were produced by a later commit.

## Checks

- Pre-amendment whole HOST:4797PASS/331explicitSKIP/4NumPy warnings,1372.16s.
  This is before ADR0036, not a claim that the entire suite ran on its final code.
- Final source/proper-wheel affected HOST: 686PASS/11explicitSKIP/zero failures,271.99s.
  Command: `env GFB_RUN_NATIVE_GROUP_PACK_SOURCE=1 .venv/bin/python -m pytest -q tests/test_group_pack_*.py tests/test_native_group_pack_*.py tests/test_card_instance_source.py tests/test_structured_lineage.py --tb=short`.
  Explicit three-model native Source/candidate/Original16/Numeric4 parity runs;
  remaining skips are the existing paused-fixture diagnostic and ten CUDA cases.
- ADR0036 focused lifecycle/accounting:28PASS; accounting/native-execution combined
  HOST53PASS/zero skips. All failures were first reproduced, including actual
  before/after-effect owned-pool/base refunds and exactly-once cleanup.
- Final Ruff(src/tests), mypy173, whitespace and Rust formatting pass.
- Rust158unit+31integration PASS/4explicit ignored, zero failures. Exact test-only
  libpython linker/runtime flags and logs are recorded; production release flags
  remain empty. Rust source did not change after that build.
- Final unqualified Dataset/Tree component panel:33PASS12.23s, including actual
  original borrowing/zero endpoint, independent feature-array equality, fresh
  96train/192validation-row LightGBM fit with configured500rounds, pair copy/load
  and isolated actual predictor scores. This is Dataset3/fixture evidence.

The first broad sandbox attempts stalled on asynchronous cancellation. The SAME
unchanged generic executor reproducer and all seven packed execution cases pass
on HOST. Those sandbox attempts are inconclusive; no executor fix or test
exclusion was used. A mixed-binary transition run131PASS is also excluded.
A post-close wide run had684PASS/11SKIP and two old expected-base assertions;
the corrected tests prove actual owned-base retirement and unchanged shared-base
ownership. All failed logs remain; no cases or assertions were disabled.

## Identical-original HDD comparison

`read-evidence/final-pairs/pair-arithmetic-summary.json` preserves all commands,
phase timers, source hashes and raw A/B/B/A trials. All four exit0. Same retained
D: six-game collector DB:1,437,696B, SHA256
524f27286454f4ab038f3bb5a4b189971b389c86f794259ab79403ce519348a7;
one group/175members read twice,2048MiB managed reader/600s cap. Every returned
size/SHA and aggregate match; descriptor/DB stat6, hashes and namespace stay.
All eight GF modules imported by B1/B2 match the final Reader source hashes.

| Median observed scope | Old Reader | Revised Reader |
| --- | ---: | ---: |
| Sum of top-level Reader phases | 11.772s | 0.338s |
| Whole DB hash calls | 356 | 0 |
| Python retained-FD DB pread bytes | 511,819,876 | 100(header) |
| Compressed Pack SHA checks / bytes | 1 /1,168,000 | 1 /1,168,000 |
| Zstd decode calls / input bytes | 3 /3,504,000 | 1 /1,168,000 |
| Namespace checks / elapsed | 709 /1.805s | 7 /0.0295s |
| First cached member pass | 5.639s | 0.000151s |
| Second cached member pass | 5.633s | 0.000191s |

Index admission, logical index, compressed/member checksum, decode and I/O phases
are separately retained. Nested timers are not additive; process wall also
includes source-tree hashing. SQLite C-level page I/O and per-statement SQL times
are NOT RUN (`strace` unavailable). Cache was not evicted; this is a small
warm/uncontrolled-cache engineering sample, not cold-HDD throughput, tail latency
or a new full-corpus comparison. It does not decide a storage-default switch.

## Real pause/resume and memory lifetime

Fresh nonformal Python and native parents use CPU inference/CUDA learner, seed17,
pause1/resume2, storage2048MiB/queue512MiB/Pack256MiB, preprocessing8192MiB,
training10GiB and7200/5400/7140/60limits. Configs `ppo-python.toml` and
`ppo-native.toml` retain those earlier accepted conditions; parents/control/DBs
are on ext4. Commands are ordinary `gf-bots train --config ... --run-dir ...
--pause-after-update 1`, followed by the same config/run-dir with `--mode resume`.

| Completed parent | Natural games | Updates | Learner decisions | Optimizer steps |
| --- | ---: | ---: | ---: | ---: |
| python-mlp-cpu-pause-resume-2-r2 | 32 | 2 | 419 | 16 |
| native-mlp-cpu-pause-resume-2 | 32 | 2 | 421 | 16 |

All four saved update proofs have custody2/natural16, qualified=false and
parent_qualification=not_performed. Child retirement is healthy. Separate
complete-parent original/numeric audits pass with the recorded8192MiB/7200s cap:
Python23.28s, native10.55s. Each preserves all42files/18directories, including
external DB/catalog/sidecars/RAM namespace, with source/native hashes unchanged.
These are scoped execution/ownership results, not learning-quality evidence.

The first Python parent6325e4f1 remains failed/ineligible:16natural games,
236learner decisions/eight optimizer steps but zero published updates. It was
never resumed. Artifact inspection requested322,850,000B with1,882,846,298B live
against2GiB; consumed decoded allowances retained1,759,721,337B. SAME-original
readonly diagnosis, without optimizer/publication, showed the overlap. ADR0036
now retires the actual consumed graph after verified completion while retaining
45,099,745B metadata plus physical custody; full readonly artifact/update checks
then use218,638,181B. No budget was raised or artifact/original check skipped.
The successful fresh parents above validate the production fix.

## Historical data, explicit audit and compatibility

The old ext4 trial2 custody1 parent validates complete on final code in21.65s,
with its42files/18directories byte/stat identical. Its original two custody1
proofs and qualified=false facts remain untouched. Historical Dataset4 tests bind
saved digests to original evidence and independently check actual episode/role/
numeric correspondence; new output uses custody2 and v2/v2 remains exact.

The older Python D: parent refuses under BOTH archived f59a2ab and current code:
saved external device80 is now72; other five stat6 fields, hashes and seal match.
No normalization/migration/old-proof rewrite or resume override was added. The old
native D: parent audit is NOT RUN after that shared mount mismatch was confirmed.
Current before/after immutability does not erase historical device drift.

Final default HDD `verify-storage --locator` exits0/complete content PASS:
one Pack/175members/three controls, original hashes/stat6/namespace unchanged.
This independent content PASS grants no historical Source/Dataset4 authority.
Unadmitted Dataset4 audit tests use six actual DB references and verify controls,
artifacts/numerics, valid pruning, aliases/conflicts, mutation, limits and uncertain
cleanup. Pending audit never grants publication or Source permission.

## Package correspondence and remaining scope

Final application wheel:175source/contract files byte-exact. Native build source
6cf79c79eb05cb4658b894999d781949db367afc4234ac3ae1838c011bdd303c;
proper installed release/MIMALLOC ELF
9b1377daac7c93c97a4f267d5f20c2204eb37401a53daa1275019614d1b6767f,
cp314-gil. Its build_info exactly matches the direct tested e75 candidate, whose
package is preserved separately. Native wheel/installed ELF and13RECORD hashes
match. Original pre-task package/dist-info and all intermediate logs are retained.

A fresh installed mandatory-only environment has no Torch/torchvision/LightGBM/
Arrow/native module; CLI help and the SAME default HDD content audit both pass.
No project dependency or lockfile changed. No push or publication occurred.

Genuine formal/final64/qualified Dataset4 fitting, HPO, final-test evaluation and
new full formal learning remain NOT RUN. Existing DrvFS compound cleanup cases
remain unexercised as documented by ADR0034; keep parent/control/pruning on ext4.
Managed memory is conservative admission, not OS RSS. Blocking filesystem calls
cannot promise a hard wall; detected overrun/uncertain cleanup is unsuccessful.
Root accepts the implemented read/audit/lifetime engineering scope only, with
directory default and all existing qualification gates retained.
