# Group Pack comparison protocol

Date:2026-11-11. Owner:root. Decision: two separate panels, following independent
bot_advisor review. Complete after Source/runtime gates; do not combine their
samples or promote historical/short-run data to qualified Sources.

## A: identical real bytes on the same HDD

Freeze all32recorded groups/4492inventory files/273062500bytes and exact original
controls of pilotdc641b06. Store names/lengths/SHA/group membership; ext4 originals
remain untouched. Use fresh exclusive owned D: destinations for each trial.

Six paired full-corpus write/read trials, order AB,BA,AB,BA,AB,BA. A is an
explicitly labelled directory archival reference: exact bytes with documented
exclusive writes, fsync/directory sync and verification. B is an explicitly
labelled archival coordinator using the production Pack encoder and dedicated
Writer submit/controls/full seal/descriptor publication/retirement. It preserves
the historical controls exactly, so does not call GroupPackCollection.archive or
its current metrics-control preparation. Actual collector lifecycle/throughput
belongs to panel B and the standalone runtime evidence, not this archival path.
Include all compression/IPC/catalog/checkpoint/verification/cleanup.
Do not call a custom archival writer the production legacy collector or assume
one fsync equals its actual held-object/multiple-hash publication implementation.

Read schedule is fixed group/episode/member order: open/admit, read and verify
all original bytes, final recheck/close. Include packed decompression/validation.
Directory readback implements equivalent requested archival checks, explicitly
labelled; DirectoryArtifactSource is not a substitute for actual _SourceOwner
production throughput. Its lifecycle/recheck behavior differs. Directory final
recheck rereads original files; packed final recheck hashes physical DB/descriptor
originals. This is format-specific archival verification, not equal physical
read work. Keep those checks and report that distinction.

Use a separate catalog root per packed trial and include its apparent file and
observed sidecar bytes in total/peak sizes. Record source/destination mount,
host disk identity, runtime/source/harness hashes and exact settings. Panel A has
no Simulator or RAM-file outputs; its owned tmpfs is zero, while aggregate
process RSS is sampled. Do not count unrelated shared-RAM files as trial-owned.

Report all pairs, medians/variability and192group observations per format.
Empirical p99 has little tail support; do not claim a reliable population tail.
These historical packed archives remain diagnostic physical comparisons, not
migration-derived Source/learning permission. No original Config is rewritten.

## B: actual matched-condition concurrent runtime

After relevant Source acceptance, one predeclared counterbalanced block:
directory,packed,packed,directory. Four fresh permanently nonformal PPO parents,
max2updates each, same pause-after1/resume-to2 point and initial model/seed17,
CPU inference/CUDA learner, backend/thread/worker/budget/checkpoint conditions.
Run each actual Source-issued learning update, never a forged numeric projection.

Overlap each parent's learner phase with one independent standalone supported
baseline collector with the same predeclared small group/game budget and separate
owned D: paths. Sibling data never enters the learner. Record actual actor/update
and storage timelines; process co-launch is not overlap evidence. Do not silently
extend a failed/non-overlapping trial. Match conditions but report differing UUIDs,
decisions/candidates/games/bytes/trajectories; these are not identical-data runs.
One pair alone is preliminary concurrency smoke. Solo/concurrent slowdown needs
matching solo runs; otherwise report concurrent observations only.

## Common measurement rules

Monotonic clocks and same group-operation boundaries. Report total publication/
retirement, commit/queue/compression/verification/seal separately; SQLite NORMAL
COMMIT is provisional, not equivalent intermediate durability to old file fsyncs.
Keep additional harness audit costs separate from implementation verification.
Report generation wall, games/decisions/s, logicalMiB/s, admission/queue/learner
wait, group p95/p99, SQLite checkpoint/DELETE/seal and model-checkpoint time
separately, aggregate process memory, owned tmpfs, peak DB/WAL/SHM/owned bytes.
State sampling interval and that sampled peaks are lower bounds; include all
parent/Writer/Simulator/learner/background processes.

Completed-output first reads and deliberate warm rereads are separate. Windows
host USB/9p cache state is uncontrolled without verified host eviction; Linux
cache advice does not prove a cold HDD. Record device/mount/runtime/build/config
identities, contention, all failures and limitations. Do not disable checks to
improve timings or compare candidate-file p99 with group-COMMIT p99.

Python/Rust tokens/features/candidate order/selections/numeric inputs are a
separate same-original-own-view parity gate: exact deterministic arrays and
predeclared numeric tolerances. Storage proofs/custody appropriately differ.
No Replay/seed/other-view/rejection/SequenceOptions feature or new fitting gate.

Permitted conclusion: physical identical-data storage plus bounded operational/
concurrency and parity evidence. Not policy strength, formal qualification,
deterministic trajectories, final64 real success, cold-device/robust-tail claims,
or authority to switch defaults. The existing directory default remains.
