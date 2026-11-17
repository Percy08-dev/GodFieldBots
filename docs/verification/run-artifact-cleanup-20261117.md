# Run artifact cleanup on 2026 11 17

Status: completed. The owner explicitly requested:

> 測定も最新の結果を残して削除しましょう。過去のモデルや学習データも不要です。

This one-time cleanup supersedes earlier instructions to retain these historical
run originals. It does not authorize automatic deletion of future experiments.
Current source, tracked test fixtures, the installed environment, registered Git
worktrees and the D-drive data destination remain outside the cleanup scope.

## Result

| `runs` allocated usage | Bytes | GiB |
| --- | ---: | ---: |
| Before | 65,124,454,400 | 60.652 |
| After, including cleanup audit | 2,429,308,928 | 2.262 |
| Reduction | 62,695,145,472 | 58.389 |

Retained 2,791 measurement/support files and removed 216,793 old files. Logical retained
file bytes are 2,331,541,934; allocated usage also includes directories and the
cleanup audit. These are Linux filesystem measurements, not a promise of an
immediate equal reduction in Windows VHDX physical allocation.

## Retained latest results

Latest current PPO measurement series retain their summaries, per-trial timing
results, configuration/provenance metadata, logs and raw profiling traces:

- `runs/collection-optimization-v2`
- `runs/ppo-update-adoption-v1`
- `runs/learning-update-optimization-v1`
- `runs/inference-comparison-v1`
- Latest v2, device-I/O and collection-detail outputs under
  `runs/hpo-producer-owned`
- Latest a04/a05 and precollection summaries under
  `runs/hpo-resident-workers/assessments`
- `runs/learning-update-preparation-20261117/input-state-inventory.json`
- Latest code verification evidence under
  `runs/assessments/defense-priority-20261117`

The three imported `runner.py`, `measure.py` and `analyze.py` helpers under
`runs/hpo-producer-owned/assessment/device-io-detail-v1` remain. Tracked
`docs/verification` reports are retained. The exact keep list and original hashes
are in the cleanup audit; retained files are byte-for-byte unchanged.

## Deleted data and consequences

Old model/checkpoint arrays, optimizer/numerical inputs, training and validation
originals, SQLite study/archive files, superseded measurements, copied build
environments and scratch workspaces are deleted. This includes parent/original/
numeric/artifact data inside the latest measurement namespaces; only the
measurement results and support files are retained there.

Historical reports and saved audit results describe their measured populations
at the time. Deletion removes the ability to repeat original/Replay/Source
audits, resume old runs or reuse their models/datasets. Some captured-input tests
now take their documented unavailable-input skips. Historical benchmark reruns
need freshly prepared inputs; retained helper code and result JSON alone do not
constitute a runnable dataset or Source eligibility.

## Safety and verification

Independent read-only review approved the fixed manifest SHA256
`653173eb80a8069f5ceff17699a056f746683b9e0a73d477316c70b667067b19`.
The executor ran with Python assertions enabled. It checked the fixed root,
complete file/directory stat snapshot, absence of nested mounts/tracked files/
registered worktrees, and active project processing before mutation. A separate
host check found no active experiments or pytest processes.

Maximal unretained subtrees were renamed into a fresh owned trash directory.
All retained SHA256 values were verified before irreversible deletion, then
verified again afterward. Symlinks were not followed; traversal permission
changes were restricted to disposable owned trash directories. The trash is
gone and execution returned 0.

PASS: all 2,791 retained hashes independently rechecked, principal summary JSONs
parse, helper Python syntax compiles, and no retained model/training binary
artifacts remain. All three budget-probe configurations still pass actual CLI
validation; installed native source/capability checks pass. No application source, dataset
format, dependency, model fitting, learning, Simulator gameplay, GPU recovery,
disk mounting or VHDX compaction was performed.

Audit directory:
`runs/cleanup-20261117-143248-5e9229db/`. It contains the approved inventory,
retained hashes, deleted path list, executor and completed result with before/
after sizes. The operation log is `/tmp/gfb-runs-cleanup-execution.log`.
