# Task5b supervision correction design

Status: accepted bounded repair design after independent architecture READY;
implementation/kernel probes and fix conformance remain required. Existing
supervision.md/ADR0054 remain authoritative except the explicit wait-order
refinement proposed here. No new dependency, format version or scientific budget.

## Clock authority follows the authenticated phase

Active admitted training retains its ORIGINAL operation admission/execution/
TERM/KILL/finish milestones. SAME-operation reentry cannot reset those bits.
Cold proof/audit after original training retirement consumes the remaining
ORIGINAL operation finish; the prior training execution stop and TERM/KILL do not
cut short that allowed cold interval. Source/saved-state validation must finish
under that cutoff. No fresh duration or grace is introduced.

After exact-pair finalization ACK AND actual parent-release ACK, interoperation/
independently admitted legacy evaluation uses the ORIGINAL session finish until
another distinct immutable operation is authenticated and bound. Admission of
that next operation uses its own declared original milestones within that same
session. Before both transition conditions, old-operation authority remains;
neither a JSON claim nor a new pathname clears an unfinalized operation.

Python control requests and Rust control/retirement must agree on this phase.
In particular, bind_operation in session-control must wait under session authority
rather than the completed operation's finish, then adopt the new operation only
on its exact ACK. Reset appropriate per-operation control state at that actual
transition, while retaining the seen-operation hash rule against budget resets.

Cold/session-control hard stops use their unchanged absolute cutoff. A stuck
host is contained immediately at that cutoff using the still-owned group; no
new wait window is invented. Perform bounded/nonblocking final observations and
report uncertainty/quarantine if positive reaping cannot be confirmed within the
original authority. Deadline exhaustion never creates successful data or a new
allowance. Normal cooperative exit before the cutoff still gets actual waits.

## Retain kernel group identity through the last possible group signal

Root adopts lifecycle advisor option B. Merely checking live descendant group
membership is insufficient: that descendant can leave after observation. Keep
exclusive ownership of the original isolated session leader UNREAPED until all
possible group signalling is permanently finished. Every kill(-original_pgid)
requires that retained leader; after consuming or losing its status, never issue
another group signal.

Before spawning, establish ordinary zombie retention with SIGCHLD=SIG_DFL and
SA_NOCLDWAIT clear, and keep one exclusive wait owner. Replace original-host
Child.try_wait polling in handshake/control/retirement with targeted waitid(P_PID,
original_pid,WEXITED|WNOHANG|WNOWAIT); zero means no observed exit, ECHILD means lost
ownership. After observed original exit, enumerate/target adopted children while
excluding the pinned original PID; a generic P_ALL loop could repeatedly select
that original zombie. Reap every adopted status positively, preserving escaped/
unknown-group failure. Never chase a descendant's new group.

When descendant drain is complete and no more signals can occur, consume the
original status through its actual Child.wait and perform the final child-
exhaustion check. On error/deadline, end signal authority before any final reap
of an already waitable original; never block indefinitely waiting for a live one.
All uncertain outcomes retain quarantine and available real diagnostics.

This supersedes the earlier internal consume-original-first wording with
observe original exit → drain adopted descendants while retaining original
identity → consume original wait. Persisted receipt meaning remains positively
consumed original/adopted waits, never just a WNOWAIT observation. Cost is one
zombie retained only through bounded cleanup, without a new containment framework.

Evidence: Linux6.18 [exit](https://raw.githubusercontent.com/torvalds/linux/v6.18/kernel/exit.c),
[PID lifetime](https://raw.githubusercontent.com/torvalds/linux/v6.18/kernel/pid.c)
and [process-group](https://raw.githubusercontent.com/torvalds/linux/v6.18/kernel/sys.c)
implementations support retained-task identity and session-leader restrictions.
Actual host uname is 6.18.58.1-microsoft-standard-WSL2. Kernel-lifetime and error
probes remain mandatory before accepting implementation; advisor ran no new tests.

## Strict stop ACK and actionable missing companion

Track exactly one pending supervisor stop sequence separately from stopped state;
consume its matching ACK once. Duplicate ACK or an unsolicited ACK after a host-
originated stop is a protocol failure, never cooperative completion. Finalized-
output retry retains its existing explicit idempotency; stop ACK has no exception.

Both missing-distribution and missing-companion refusals give the explicit
build helper command including --python and --out plus supported wheel install/
launch instructions. Discovery still refuses before GPU/game work, never builds
at runtime and never silently selects legacy mode. Document uv's editable-sync
behavior and use installed gf-bots or uv run --no-sync after runtime-wheel install.

## Required bounded verification

- Delayed real cold audit past prior TERM but before original finish; finalized/
  released idle past prior finish within session, then real distinct operation;
  unfinalized/SAME-operation cannot reset authority, expired hard stop honest.
- Local actual leader WNOWAIT/PID/PGID lifetime, early handshake/control exit,
  targeted adopted waits, escaped-member synchronization around escalation,
  SIGCHLD explicit ignore normalization, ECHILD/wait failure/deadline handling,
  normal in-group TERM-ignore→KILL and zero group signals after original consume.
  Verify identity invariant directly; no PID-reuse stress loop is required.
- Actual duplicate/unsolicited stop ACK refuse cooperative/cold continuation.
- Both missing-companion paths contain complete actionable commands; existing
  runtime/native-only wheel, source/build/extension hashes and legacy readers
  remain consistent. Rebuild final source-matched wheels after source changes.
