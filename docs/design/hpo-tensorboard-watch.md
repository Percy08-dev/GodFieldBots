# TensorBoard for standalone HPO watch

An interactive `gf-bots hpo watch --path PARENT` displays a local TensorBoard
URL below the progress frame when the optional `tensorboard` extra is
installed. `--no-tensorboard` keeps interactive progress text-only. One-shot,
JSON, redirected and other run modes keep their existing output.

## Responsibilities

- The CLI recognizes the standalone HPO6 parent through the existing Run5
  observer. The terminal watcher starts at most one companion using the same
  Python interpreter. The companion owns the TensorBoard server and event
  writer. Only the companion imports TensorBoard. Use TensorBoard 2.21.0 as
  an optional extra, with no TensorFlow dependency. The server binds to
  `127.0.0.1` on an OS-selected port and uses the in-process record loader.
  The child masks CUDA devices because upstream TensorBoard can detect a
  separately installed TensorFlow.
- The companion reads the run directory and writes derived events to a private
  temporary workspace outside the parent. Only verified scalar metrics and
  progress counters enter the event stream. The process publishes the bound
  port to the watcher after startup. A curses frame includes the URL each
  time it draws. A line-oriented terminal prints it on change. An unavailable
  dependency or companion startup failure leaves the text watcher usable and
  provides a short status line. TensorBoard's own browser reload setting
  is exposed as Settings → Reload data. Its default is off.
- The watcher owns the child's lifetime. On normal exit or Ctrl-C it closes
  the parent end of a dedicated control pipe, waits for the companion, and
  terminates it if necessary before cleaning up its temporary workspace.
  The companion also stops on pipe EOF if the watcher disappears. This path
  uses neither training locks nor simulator, policy, CUDA, or training APIs.

## History and polling

Reuse standalone Run5 header validation and the existing zstd or JSON update
reader. Require `run.json` and `latest.json` to agree on the committed
generation and SHA-256 before importing. Pin that snapshot, verify Gen6
records along the immutable parent hash chain to generation zero or the last
imported generation, and use each generation's update hash to verify its
Update8 record, run ID, update number, parent hash and completed status. Apply the strict
watermark and lineage checks only to the dashboard reader. Use separately
bounded reads per record so a large backfill does not exhaust a shared scan.

On startup, backfill all confirmed updates in order. Thereafter read only the
mutable head and any newly committed suffix at the watch `--interval`, which
defaults to one second. Verify and stage a whole suffix before emitting its
compact scalar rows and explicitly flush when rows are added. If a read
races a save, retain the previous chart and retry on the next poll. An
identity or committed history change cannot append to the existing stream.
Show a waiting status while a consistent head is unavailable. A failed
event write or flush retires the companion, since retrying could duplicate
partially written points. A fresh watcher rebuilds from source records.

Use the update number as the TensorBoard step. Log total loss, policy loss,
value MSE, entropy, approximate KL, the optional diagnostic fields from
ADR0063, per-update collection and update durations, and cumulative learner
decisions from the generation. Missing, null and nonfinite diagnostic values
have no scalar point. The history has no per-update Unix timestamp, so the
event time is the observation time. The update axis preserves the recorded
ordering and spacing. Run5, Gen6, Update8 and diagnostic schema 1 are the
formats observed in the designated Pestle–Mirage parent on Simulator
`6d50603dab3f6d2df34ea4e433bab7dca96f52c1`, Gameplay API v2, catalog schema 2.
TensorBoard's scalar sampling limit applies on runs longer than this
1,000-update preset.

## Run overview

The Overview tab shows what the terminal watcher currently knows. Present
four groups, in this order:

| Group | Fields |
| --- | --- |
| Run | Model, seed, recorded status, and any pending finalization |
| Current batch | Phase, ended/planned games, selected learner decisions, elapsed time, and active invocation ETA when available |
| Committed progress | Updates/target, used decisions/budget, and collected decisions |
| Latest update | Loss, entropy, approximate KL, collection and update durations, and collection games/s and actions/s |

Use the same already-merged observation that drives terminal rendering.
Identify missing or stale live observations in Current batch and partial or
stale committed records in Committed progress. Suppress transient counts
and phase for an unavailable live observation. Reuse the watcher's rules for
ETA, elapsed time, and the active update target. Keep Time Series for saved
per-update curves. There are no charts for in-progress counters or ETA.

The watcher publishes one compact snapshot to its existing private temporary
workspace, by atomic replacement only when the displayed data changes. It
projects only the displayed fields. No new reads of the original run are
needed. A pinned TensorBoard 2.21 adapter serves this snapshot through the
native Text tab, renamed Overview, using the standard safe Markdown renderer.
The displayed native step is the committed update number when known. Only
one current row is returned. The temporary snapshot grants no authority
and does not enter the event history. A snapshot failure can make Overview
unavailable independently of terminal rendering and scalar export.

Verify grouped values, active and stale observation transitions, unchanged
output modes, no write on identical display data, the native plugin listing
and current row before/after refresh, and Time Series coexistence. Use the
same isolated checkout as the original TensorBoard feature while the target
learner runs. The architecture reviewer accepted the Overview design before
implementation.

## Alternatives and verification

TensorBoard supplies the recurring reload and the scalar UI. A Plotly HTML
export would fit individual snapshots. Dash would add custom screen and
callback work for features such as choosing the decision or game-count axis.
Keep this feature local to the existing watch command and the optional extra.

1. Add focused tests, confirm the new tests fail, then implement the history
   reader and companion under `gf_cli`. Exercise raw and zstd updates, G0,
   backfill, commit races, a changed or corrupt chain, absent diagnostics,
   idle polls without update reads, and append without duplicates.
2. Exercise the URL line, disabled modes, optional dependency and startup
   failure, and child cleanup through the watcher. Run the existing progress
   checks with the feature tests, and run scoped Ruff and mypy checks.
3. Resolve the optional extra and verify under Python 3.14. Run a bounded PTY
   smoke against the user-specified parent while training runs. Check local
   HTTP and event data, Ctrl-C, and the original plain JSON command. Record
   backfill and idle cost separately from the training workload.
