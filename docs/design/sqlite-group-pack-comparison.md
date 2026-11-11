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

## Panel B failure retirement (architecture READY)

No new trial follows cancellation or a trial failure. Persist started/terminal
records and the primary/secondary failures before propagation, retain actual
exception/task/process/cleanup owners, and mark the remaining trials NOT RUN.
Stop new spawn admission before attempting shutdown. Production cleanup remains
the first owner; the harness must not retry ambiguous raw-FD numbers or fabricate
healthy Source/parent retirement.

Use an explicit failure-only process boundary: settle the actual sibling
task and production cleanup first, then use retained actual Popen handles to
stop/reap any still-owned children within a bounded cleanup reserve (TERM, wait,
KILL, wait). PID/birth metadata is diagnostic, never authority for signalling an
unrelated or reused PID. Do not proceed while an actual spawn remains in flight.
All direct actor/learner/Simulator/Writer children must be confirmed stopped and
reaped before the coordinator may exit. If that cannot be confirmed, keep actual
owners/process alive and publish a cleanup-pending failure for external root
disposition; do not drop owners by ordinary uncaught CLI exit.

After children are confirmed retired and admissions are fenced, a failed process
may exit while still retaining uncertain in-process FD/thread owners: process
termination is the explicit final OS retirement boundary, never a successful
Source close. Persist/sync this fallback and its limitations before exit; external
supervision must confirm process termination. No healthy return, new trial,
status repair, guessed outcome or reused serialized cleanup token is allowed.
This handles a failed measurement process without inventing a general recovery
framework. Acceptance needs inert cancellation/secondary-error/unreaped-child
tests and an independent lifecycle review before real trials.

## Revised B: ext4 PPO concurrent with the same-HDD collector

Root accepts bot_advisor's bounded amendment after full-parent-on-D blockcd82
fails at private0700 pruning. Both updates and one real collector overlap occur,
but the whole parent remains failed/ineligible and the remaining3trials NOT RUN.
The current D:9p mount reports0777 for the newly requested0700 staging directory.
Later held-original pruning rename is an additional code-derived risk, not an
observed second failure. Do not relax permissions, skip pruning, repair/resume
that parent, add a legacy external locator or change privileged host mounts/ACLs.

Run ONE new independently labelled D/P/P/D block under unchanged seed17,
CPUactors/CUDAlearner, two updates, pause1/resume2, six-game sibling and all
original wall/admission/execution/finalization/resource caps. All PPO parent,
checkpoint/numeric/control and original collection storage is on the SAME ext4
filesystem in both formats (Directory collections or packed DBs). The independent
six-game collector alone writes to the SAME D:HDD as PanelA: directory originals
or packed DBs. Give PPO and sibling distinct configured DB/catalog/RAM roots.
Do not accidentally leave the PPO packed DB on D while directory PPO data is
ext4. Each block/trial has fresh exclusive owned paths; no automatic extra block
or trial extension follows failure/non-overlap.

Record both owned filesystem roots and actual mount/device identities before
admission. Accounting/monitoring includes ext4 plus D plus owned RAM and every
actual process; deduplicate overlapping roots/inodes, report per-filesystem and
combined apparent bytes, category/sidecar scopes, coverage/gaps and scan overhead.
Commit observer uses the actual explicit parent path rather than inferring it
from a collector root. Keep failure-retirement/creation fencing, actual Source
issuance and primary-error behavior unchanged. Before execution independently
review path-split effective-config assertions and focused inert failure tests;
exercise private0700 staging and held-original rename in disposable ext4 files.

PanelA remains the identical-data alternating HDD archival comparison. RevisedB
measures actual ext4 PPO generation/resume/learning concurrent with actual HDD
collection, a combined configured workload whose two components both change
format. It cannot isolate per-component format causes, exact optimizer-kernel/
HDD-I/O simultaneity, solo slowdown, cold-cache behavior or reliable population
tails. Do not pool original failed-HDD and revised conditions or call the latter
successful full-parent HDD support. Original all-parent-on-D remains FAIL on the
current mount; its remaining trials stay NOT RUN. This amendment plus PanelA
addresses physical storage and operational concurrency separately, with those
limits explicit. Default remains directory; formal/qualified/final-test gates
are independent and unchanged.
