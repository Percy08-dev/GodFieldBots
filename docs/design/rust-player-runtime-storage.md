# Single recording envelope and thread database ownership

Status: accepted bridge/database-ownership design after independent architecture
READY. Implementation/conformance and workspace-formula evidence remain required;
this review does not approve an unspecified decoder/workspace coefficient.
Base:2526f29. Facts: native capture leases exact immutable array bytes; legacy
Python StorageBudget composes recorder/writer reservations and measured RAM-root
files. Those accounts are not currently shared.

## Parent owner and bridge

Choose the native capture mailbox's thread-safe account as the ONE parent
recording envelope. Its configured limit is the existing storage budget (5GiB in
all C16 comparisons). Refactor its private byte counter/reservation into one
shared recording-account handle, preserving existing capture reservation and
last-NumPy-view semantics. Actors never invoke Python to acquire/release bytes.

Expose a narrow native external allocation lease for additional Python-owned
recording allocations and a Python adapter with the StorageBudget operations
actually needed by the new writer/sink. The adapter uses the SAME account, not
another5GiB counter. Default legacy StorageBudget/worker behavior remains unchanged.
Native original bytes are already charged: adopting/queueing/ACKing a bundle or
its view adds no charge for those SAME bytes. Native reservation is inseparable
from its immutable array and must survive every view. External leases cover only
new allocations, never replace/release an original array's lease early.

The account reports exact live native encoded bytes, explicitly reserved Python
record buffers/staging, total/peak and limit as separate counters. New canonical
Journal/column bytes and raw/compressed pack buffers are separate physical
allocations while they coexist, so reserve them before construction. Aliases,
queue references and original source handoff do not allocate/charge a second
copy. Additional transport/workspace bytes must follow actual owner last use.
No per-record Python callback runs on W and no process baseline is charged for
an interpreter that does not exist.

Python RAM-root staging, if used by existing Pack helpers, is measured separately
from process PSS and charged through the SAME recording account. Reserve planned
growth before writing. When an already reserved output becomes a measured staged
file, transfer that portion of the EXTERNAL staging reservation to its file lease
atomically, rather than briefly charging the same pages twice. Original native
array leases cannot be split/converted this way. Reclaimable ext4/page-cache and
unreclaimable tmpfs observations are reported distinctly; moving bytes to files
is not evidence of lower total memory by itself.

## Workspace and admission

Private JSON decode/validation/metadata scratch uses the invocation's existing
admitted Source/metadata workspace budget, not a second recording pool. Keep
those estimated/private allocations distinct from exact retained byte charges;
reuse actual content-code workspace policy, never old retained-dict estimates
for dictionaries no longer retained. Any new decoder/codec estimate requires a
bounded measurement on representative and maximum supported capture shapes
before final acceptance; document the estimate and its evidence. No new arbitrary
multiplier, full per-process WRITER_BASE or eliminated pipe/IPC copy coefficient
is approved. An implementation needing a new restrictive estimate returns it to
root before adopting it as production admission.

Pack/canonical output buffers are recording data, not hidden decoder scratch.
Derive combined peak reservations from concrete simultaneously live source,
canonical/pack/compressed buffers and codec/API copies; aliases count once.
Reuse existing Pack compression/SQLite formats and validators. A budget adapter
may expose one aggregate reservation for a whole known transform instead of
nested waits that hold the transform's own source/workspace hostage.

Temporary entry-capacity pressure backpressures asynchronously; byte-limit
exhaustion remains explicit failure. Before waiting for bytes, distinguish an
independently progressing owner that can release from data owned by the SAME
blocked operation. Process/commit already-ready groups where that can release
real owners; if the operation's required combined peak cannot fit after those
releases, fail promptly and cancel the collection. Never wait until the whole
run deadline for capacity only its own ACK/submit could free. No speculative
max-Pack-per-game preallocation or fixed percentage headroom is introduced.

Terminal acceptance transfers immutable episode+reservation custody before SQL,
so game slots can progress while a seat-swapped pair is incomplete. Writer keeps
actual native/canonical source owners through their real last use and group
commit; only one actual ownership chain justifies encoding/Source issuance.
Encoding/custody acceptance is distinct from commit, WAL checkpoint, collection
seal and actual thread/resource retirement. Any failure prevents PPO dispatch.

## Workspace policy ruling

Root adopts the scoped advisor recommendation: one explicit caller-provided loan
from the EXISTING shared Source/metadata budget covers the original storage
thread's scratch and retained validation-bank lifetime. It is a soft planning
allowance, not measured use or an enforced physical-RAM cap. Native originals,
canonical/Pack/staging bytes remain in the one recording account above.

Do not adopt either proposed decode-token formula as a production rejection rule.
Decode-only measurements overestimate individual decodes by2–4x yet miss DTO/
Journal/codec overlap, retained banks and duplicate terminal-bank decoding. No
per-record Python scan or native token-count API is needed for this ruling.

Before production acceptance, measure the complete path through decode, DTO,
Journal6/columns, terminal processing, Pack/SQLite and seal. Include genuine
sequences and valid configured maximum candidate/slot/event/string shapes,
incomplete pairs and worst FEASIBLE retained-bank/transform concurrency. Warm up,
separate fresh/reused allocator observations and a long sequence; record traced
peak/retained baseline after retirement, process RSS/PSS and separately charged
native/recording/tmpfs. Tracemalloc alone omits native/SQLite/compression memory.
For the configured2GiB loan, document its measured planning range and concurrent
Source demand; unmeasured means missing evidence, never an invalid workload.
Engineering component acceptance may state a precise bounded range. Before
production/default acceptance, add a targeted actual-retained-bank overlap study
(one long bank near the selected episode limit, feasible other-bank population,
one real terminal/commit overlap), plus prescribed C4/C2 and actual C16 measures.
Do not require every unbounded rawAPIhistory/identifier or Cartesian maximum;
extreme settings retain the owner's accepted OOM risk. A discrepancy requires
root disposition/allowance review, not an invented margin or silent rejection.

Reject only concrete inability to admit the requested shared-budget loan, an
existing format/shape limit, failed exact recording reservation, actual processing/
allocation failure, or a caller grant known below measured necessity. No safety
multiplier or generic allocation framework is approved. This does not provide a
hard preallocation guarantee for arbitrary inputs; such a different requirement
would need its own lifecycle-aware planner/design evidence.

## Database transitions and failure custody

One narrow database command owner reuses the existing group_pack_db operations.
The legacy subprocess worker keeps its framing and EXACT budget-WRITER_BASE
allowance. New dedicated thread constructs/uses/seals/closes its sqlite connection
on THAT original thread with check_same_thread enabled. No Popen/pipe/PID-shaped
stand-in or generic operations framework is needed.

An observed fault probe establishes a concrete thread-promotion gap: during
seal-producer a readonly OriginalFile is created locally, then closing the old
original can raise before readonly enters persistent owner state. Legacy process
exit closed that FD; thread exit would leave it open. Similarly dropping db
state after rollback/close failure can hide a still-live connection.

Keep every transitional readonly/original owner and database connection reachable
in the command owner BEFORE a fallible handoff/close. Remove a reference only
after positive close or ownership transfer. Cleanup separately attempts rollback,
connection close and each original/transitional owner on the original DB thread;
a rollback failure must not skip the close attempt. Retry only through the
owner's existing safe/idempotent close API while that same thread still owns it;
never retry a guessed numeric FD after an uncertain close. Persistent failure
retains the actual owner and reports unconfirmed cleanup, not an empty error list
or successful retirement. No unbounded retry/recovery framework is introduced.

A failed writer accepts no new data work; the original thread may remain reachable
for bounded cleanup commands/retry under the original cleanup deadline. It may
exit as healthy only after resources positively retire. If cleanup cannot be
confirmed, prepared success is forbidden and the supervisor's eventual original
host reap is the physical fallback; it does not qualify data. Preserve failure
facts and strong custody, never rely on Python GC or a thread exiting to close
SQLite/FDs. Normal old-worker content/healthy behavior remains unchanged.

## Verification

- Shared limit N: held native bundle/view plus new Python staging/pack leases
  can never each admit N; adoption/ACK charges the native allocation once.
- Last-view retention, external-lease close/failure, staged-file transfer,
  actual peak vs estimates, and no per-process/IPC baseline duplication.
- Full queue and self-held transform near budget: real independent progress or
  prompt explicit failure, no deadlock/no unbounded temporary capacity.
- Partial seat-swap group admission and writer acceptance before game slot release;
  failed encoder/SQL/seal prevents Source/update/parent eligibility.
- Actual same-thread SQLite init/submit/seal/close and retained-descriptor reopening;
  existing process-worker regressions unchanged.
- Inject old-original close after transitional readonly creation, rollback and
  connection/readonly close faults: no unreachable live FD, no false healthy
  cleanup, bounded retry on original owner thread or honest unconfirmed failure.
- Bounded memory/workspace evidence plus actual full-run PSS/tmpfs/CPU/queue
  measurements later; implementation smokes do not prove<=3%or learning quality.
