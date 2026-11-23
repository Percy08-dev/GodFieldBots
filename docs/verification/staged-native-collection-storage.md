# Native PPO collection staging verification

[ADR0065](../adr/0065-staged-native-collection-storage.md) records the
durability and resume boundaries. This experiment uses the same Model 4
seed-17 starting checkpoint and schedule as the earlier storage PoC. The
runtime is Simulator `072dabbca1458c9526897781d2fd64edeeb4f9ca`, the accepted
Gameplay API v2 and catalog schema 2, Journal 6 and candidate-columns-v2.
The host uses an RTX 5070 Ti, CUDA Torch 2.14.1+cu132, Python 3.14.10, tmpfs
at `/dev/shm`, and the dedicated ext4 VHDX at `/mnt/gf_data` backed by the D
HDD. The tested model/parent control tree and originals are on that HDD.
The GPU recipe is Graph, one lane, W16, 1 ms, width 32 and 128 MiB.

## Durability and restart

- Both public supervised paths, Model 3 ordinary PPO and Model 4 HPO6, each
  collected eight games, durably published the Pack, updated and paused at
  generation 1. They each resumed on the same boot and collected eight more
  games, durably published and updated to generation 2.
- The independent full Journal, Replay and Python numerical audits passed
  for each of these invocations. The recorded Source path was under the
  dedicated HDD ext4 path and the tmpfs work namespaces were empty after their
  original consumers retired.
- The production promotion helper was stopped with SIGKILL in isolated child
  tests before copying and after the durable catalog commit. Before the copy
  no final batch or catalog appeared, and the RAM input was still present.
  After commit the receiver DB SHA-256, exact descriptor bytes and catalog
  tuple were read successfully while the RAM input still existed.
- Focused tests inject EIO and ENOSPC, check fsync ordering and the
  post-commit sync failure boundary, hold an existing catalog/namespace
  against conflicts, and reject a changed work mount before bulk writes.
  An injected accounting failure after collection retirement closes the
  unclaimed actual pending files and game training buffers and preserves
  the completed HDD bytes.
- Same-boot explicitly paused generations are the supported resume boundary.
  Real WSL reboot/cross-boot training authority is a separate design by
  owner choice. Trusted hardware power-loss behavior was not tested.

## Previous inline-transfer failure

The original 256-game PoC completed the detached copy but reported
`AssertionError` before Source accounting and the CUDA collection-settled
sample. The writer had joined, but 54,670,950 native and 37,174,561 external
recording bytes remained live during host cleanup. The exact initiating
assertion was not persisted. An instrumented 256-game rerun of the same
in-process detached copy completed update 1 and paused with a 28.76 s
collection. The smaller eight-game investigation did not yield an
initiating traceback.

The implementation does not use the PoC collector wrapper. A separate
regression identified and reproduced the unclaimed collection-ownership
failure that explains the buffer-retirement symptom under a failed
post-collection accounting write. The production cleanup retires that
collection when no Source transaction took ownership. The presence of
this failure path does not identify the original AssertionError expression.

## 256-game HDD Source timing

The release run used the public HPO6 operation, fresh G0 in each independent
parent, D64 for the bounded post-collection update and no Torch profiler.
Filesystem memory and process PSS were sampled externally. A passive timing
wrapper observes the writer promotion and the point where the existing
Source projector returns. The application collection clock includes the
promotion; the first numeric column subtracts that one invocation's
promotion time. HDD and Source columns are wall-clock elapsed from the same
host collection phase start.

| Run | Collection excluding promotion | Promotion | Through HDD sync | Through Source projection |
| --- | ---: | ---: | ---: | ---: |
| bench-1 | 28.03 s | 1.18 s | 28.68 s | 29.25 s |
| bench-3 | 27.88 s | 1.26 s | 28.73 s | 29.18 s |
| bench-4 | 27.92 s | 1.14 s | 28.72 s | 29.10 s |
| Median | 27.92 s | 1.18 s | 28.72 s | 29.18 s |

Each included invocation had all 256 natural games and a conservative exact
256 overlap from successful Join through earliest terminal Action. The
bench-2 trial also completed 256 natural games and was independently verified,
but the event intervals only establish a simultaneous lower bound of 255.
It is excluded from the median. The HDD median corresponds to 8.91 games/s,
and the Source median to 8.77 games/s. The comparison uses the earlier 49.57 s
direct HDD median and the 29.03 s detached prototype. Timings omit the HPO
objective/evaluation campaign and separately timed PPO optimizer work.

The included trials sampled up to 173 MiB of added tmpfs blocks and up to
3.14 GiB peak process PSS. The minimum sampled WSL available RAM was
26.33 GiB. Their swap input and output counters stayed flat and the SSD distro
VHDX logical and allocated sizes were stable. Conservative HTTP interval
measurement temporarily held already opened raw log FDs in the observer,
which contributes to the measured tmpfs usage. No payloads were copied to
the SSD for the physical acceptance runs.

The final direct-path compatibility correction restored the old preflight
exception scope. A separate exact-overlap HPO confirmation (bench-5) used
the final code and took 28.81 s through HDD sync and 29.23 s through Source
projection. Original DB hashes were independently re-read from HDD. All
verified HPO benchmark batches matched the archived G0, 5,160 candidate
member names, sizes and hashes, and the prior G1 weights and resume shard.
The optional full Journal, Replay and Python numerical reconstruction was
also attempted directly against this HDD Source. It stopped at the
existing source-account limit (12,645,052,249 bytes available to the audit
after other held data). The individual 256-game sealed Pack audits and
the eight-game full reconstruction results above cover distinct scopes.

## Consecutive 256-game updates

The ordinary Model 3 public entry collected and committed two 256-game
updates in one invocation. The time through HDD sync was 26.60 s and
29.42 s. Time through Source projection was 26.95 s and 30.05 s.
The number of private working directories was sampled as 0, 1, 0, 1, 0.
Both Source directories used the HDD originals root. The sealed Pack
audit passed for the two updates. The maximum observed increase in tmpfs
blocks was 198 MiB. Process PSS peaked at 3.78 GiB. Swap counters and SSD
VHDX allocation stayed flat.

The seed-17 HPO two-update attempt committed update 1. The Simulator
returned HTTP 422 with INVALID_ACTION on pair p11 in update 2. The earlier
direct-storage HPO run logged the same HTTP status and error code for the
same pair. In the staged attempt the second HDD batch did not appear and
the tmpfs workspace retired. The 256-game HPO6 continuation stopped during
this collection. The eight-game same-boot HPO6 continuation completed its
second update and passed full audit in a separate run.

The direct and staged attempts had identical G1 weight and resume payloads,
the same 39 accepted actions, and the same rejected command at revision 39.
The fixed-seed replay reproduced the rejection independently of training or
Pack transfer. The historical Simulator's Dream grouping let an additional
weapon appear as Strength Powder, a damage-adding utility with no compatible
disguise under the owner-confirmed rule. The candidate generator included
Discard based on that perceived definition. See the
[diagnostic evidence and reproduction](../issues/0013-hpo-dream-discard-hidden-weapon.md).
The storage design and its performance baseline were accepted for operation.

## Automated checks

The final focused Python selection passed 139 tests across the new
staging, existing pack storage, configuration, timing, and direct
preflight recovery paths. Ruff passed. Mypy passed on the ten affected
production modules with optional imports ignored. Both independent
reviewers cleared the failure-boundary and custody findings. Raw timing
and resource samples were kept in tmpfs during the runs; the derived
small records are under `runs/staged-collection-acceptance/`. The
benchmark also points TMPDIR and SQLITE_TMPDIR to private tmpfs.
