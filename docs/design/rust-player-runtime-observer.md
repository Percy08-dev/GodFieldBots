# Standalone process observer for the Rust runtime comparison

Status: scoped implementation accepted after independent correctness and
architecture READY following one repair round. Preimplementation proposal SHA256
`1ddff2fe8fb3990969a849898911882befe1aff60bf2f81111657575a716f5fa`.
Owner: root. Scope is only the external observer portion of the still-proposed
[measurement supplement](rust-player-runtime-measurement.md). It does not approve
application telemetry, phase events, calibration results or any timing/adoption
run. Required full comparisons remain exactly the approved plan.

## Boundary and ownership

Implement tools/rust_player_observer.rs as a Linux std-only Rust executable,
compiled explicitly by rustc into a fresh /tmp artifact before measurements.
No Cargo/native-crate/package/dependency change. A new focused test module may
compile/test it. The tool owns its original command Child and fresh output
only; it has no Source/GPU/lock/supervision capability and no subreaper/group-
signalling or workload-termination role. Existing actual command/supervisor
absolute deadlines remain authoritative. No Python controller is launched.

CLI: observer --output ABSENT_ABSOLUTE_DIRECTORY [--no-sampling] -- COMMAND ARGS.
Default CPU/process cadence100ms, PSS cadence1s, fixed and recorded rather than
adapted from results. --no-sampling retains only original command spawn/wait/wall
and observer cost metadata for later enabled/disabled calibration. Both command
streams are redirected to command.stdout.log / command.stderr.log in the fresh
output. Do not read environment contents, Simulator history, unrelated artifacts
or follow output symlinks. Setup failures occur before command spawn.

Use CLOCK_MONOTONIC nanoseconds via minimal Linux system FFI, recording boot ID,
HZ, page size, kernel and tool/source/binary identity supplied by the external
experiment manifest (never pretend a pathname is a verified executable hash).
Retain original Child through actual wait. Capture start immediately before
spawn and finish immediately after actual Child.wait returns, independently of
sampler scheduling or drain/join time. Sampler failure must not drop the child
or replace command failure: continue to actual wait, mark observation incomplete,
then drain/flush. No new cleanup grace is created. Before spawn, explicitly establish SIGCHLD
SIG_DFL with SA_NOCLDWAIT cleared and no competing reaper (or fail setup before
spawn); retaining Child alone is insufficient under an inherited auto-reap
disposition. Test ignored/auto-reap inheritance rather than assuming a shell
default. Tool process return preserves an actual numeric child exit code. For a
child killed by signal n it returns numeric128+n by explicit shell convention;
it does not self-signal, and command_wait.signal still records the ACTUAL child
ExitStatus signal, never one inferred from a numeric exit. For example child
exit143 and child SIGTERM both yield wrapper143 but remain distinct JSON results; an observation error with child success
returns2. JSON command_wait always distinguishes code,signal,unknown. The tool
must not turn an unobserved exit or diagnostic failure into command success.

## Process discovery and sampling

Capture the original command's (PID,start_ticks) before allowing wait/reap, using
the retained child to prevent PID reuse during initial capture. Each process
table scan uses PID/PPID/start_ticks for current ancestry and retains already
observed owned identities while alive, including later reparenting. A recycled
PID never inherits ownership. Unowned process metadata is transient discovery
input and is not persisted. No claim about an unseen short-lived process follows.

For owned identities sample /proc stat CPU ticks/RSS and status/task information
for actual TID/name/thread counts; executable and command line label roles.
Configured Rust/Torch counts remain separate. Record read races/unavailable fields
explicitly. CPU is observed tick accounting; do not add sampled ticks to final
rusage or claim missing short-lived work is zero. Observer self CPU is separately
read before/after and excluded from workload resource totals.

PSS uses smaps_rollup only at1s cadence. It is a timestamped sweep, not a truly
simultaneous snapshot: record sweep/read start/end and identity coverage. Sum PSS
from that sweep; never sum historical per-process peaks, RSS, recording grants
or allocator stats as disjoint physical memory. A missing member makes the sweep
partial; partial sums cannot be presented as complete peaks. Report maximum
complete-sweep PSS only from NONEMPTY identity-valid complete sweeps wholly within
the actual command start/finish monotonic interval. No-sampling, quick commands,
empty or post-exit sweeps cannot manufacture a zero peak: absent a qualifying
sweep, peak is null. Keep empty/shutdown/partial sweeps as raw observations.
Report qualifying count separately from total/complete/partial sweep coverage. Sampling
can miss peaks/processes; that limit and differing observation costs must survive
into final interpretation. Raw observations, including shutdown/partial reads,
remain available. No phase is inferred from Prepared or Exit wall timestamps.
Actual diagnostic monotonic phase events are a separate future application gate;
the offline experiment report will join by clock/boot and invocation identity.

## Closed diagnostic output

Output family version1/kind rust-player-process-observation-v1 has no eligibility
meaning. summary.json exact top fields:
{schema_version,kind,command,clock,observer,command_wait,sampling,peak_complete_pss_bytes,
coverage,errors,limitations}.
command={argv,pid,start_ticks,start_monotonic_ns,finish_monotonic_ns,wall_ns}.
clock={name,boot_id,hz,page_bytes,kernel}; name=CLOCK_MONOTONIC.
observer={pid,start_monotonic_ns,finish_monotonic_ns,user_ticks,system_ticks}.
command_wait={exit_code,signal,observed}; exactly one of code/signal is nonnull
when observed; neither when unknown. sampling={enabled,cpu_interval_ns,pss_interval_ns}.
coverage={process_samples,pss_sweeps,complete_pss_sweeps,partial_pss_sweeps,
peak_eligible_pss_sweeps,observed_processes}; errors/limitations are arrays of strings.

process.jsonl holds exact rows {monotonic_ns,processes,errors}; each process is
{pid,start_ticks,ppid,state,user_ticks,system_ticks,rss_bytes,threads,executable,
argv,tasks}; tasks entries={tid,name}. Unavailable optional data is null, never0.
pss.jsonl holds {start_monotonic_ns,finish_monotonic_ns,complete,total_pss_bytes,
processes,errors}; process rows={pid,start_ticks,start_monotonic_ns,
finish_monotonic_ns,pss_bytes,error}. total_pss_bytes is null for partial sweeps;
individual successful members remain raw evidence. Final summary fields never
substitute for missing raw files. JSON escaping and unsigned bounds must be
correct without adding a serialization dependency.

External manifest records actual rustc version/command, source and built binary
SHA before launching the observer. Root orchestration—not the observer—selects
trial order/workload, links application provenance/phases and decides adoption.
No profiler, GPU query, native runtime, Python library or policy module is imported.

## Verification and independent execution scope

Implement with RED/GREEN tests before behavior. Deterministic helpers test /proc
stat parsing (spaces/parentheses in comm), PID reuse/ancestry/reparenting, partial
PSS aggregation/identity races, JSON escaping/null semantics and original-wait
clock boundaries. Small actual commands verify logs, nonzero/signal status,
no-sampling, quick-command/no-valid-PSS, inherited SIGCHLD ignored/auto-reap,
descendant discovery and wait continues after sampler/flush error.
Use handshake/file events for process lifetimes, not arbitrary timing sleeps.
This is a tool smoke, never a training/performance run.

Root may assign a second bot_implementer ONLY these new tools/test files while
the first owns B application code. This explicit exception to one-writer default
has no shared production files, package/native source identity, Cargo outputs or
venv mutation: standalone rustc outputs are unique under/tmp. Neither writer
edits the other's files. Stop tool tests/compilation during a declared actual
training case freeze and coordinate with Root. Architecture and correctness
review this tool independently; the complete telemetry gate remains closed until
its other contracts and all Task6 conditions pass.

## Scoped implementation evidence

Both independent final reviews are READY after two P2 repairs. New ancestry
edges require a positive parent-identity read after observing the candidate child;
uncertain retained owners cannot grant ownership through stale scan data. PSS
replay retains the original readable descriptor and checks the published path's
type/device/inode before and after replay. Symlink or regular-file replacement
makes observation incomplete with null peak, preserving the actual child status.

Root freshly runs19 pytest cases (2.22s) and14 Rust helpers,0ignored, with
rustc -Dwarnings. Focused Ruff/format/rustfmt checks pass in the implementation
report. Source SHA256 is
`55e8b0e77c8b9a1dfc192d2f5a0e670b09c9d7a6d7aa5107396abbe5071e2a59`;
test SHA256 is `bf9622c5fb46a28bb07179a5b61d9e70dbb654a5dfad77e02eb59bec911e1509`.
Coverage includes actual child codes/signals, inherited auto-reaping, FIFO-held
descendants, real write failures, path replacement and deterministic identity
races. Actual-kernel PID-reuse stress and other platforms are NOT RUN.

These tiny commands provide tool engineering evidence only. Application
telemetry, CUDA acceptance, enabled/disabled overhead calibration, the full
approved matrix, default adoption and learning quality remain separate gates.
Compile and hash an explicitly configured optimized observer before formal
measurements; the debug smoke executable is not an unreported timing artifact.
