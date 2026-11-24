# 0071 TensorBoard for standalone HPO watch

Status is Accepted. The architecture reviewer accepted the
[bounded design](../design/hpo-tensorboard-watch.md) before implementation.
Independent correctness and architecture conformance reviews cleared.

## Context and decision

The standalone HPO6 terminal watcher reports the latest committed scalars.
ADR0063 and ADR0064 already preserve the per-update metrics, diagnostics and
storage identities needed for learning curves. Use that durable history as the
input to an optional TensorBoard 2.21 companion owned by the interactive
watcher. The companion serves only loopback and writes only private temporary
scalar data. An unchanged head requires no update-record reads. A new suffix
is checked against the previously imported lineage before it can appear.

## Alternatives and consequences

TensorBoard provides scalar charts, smoothing and a reload control. A Plotly
export is useful for sharing a snapshot. A Dash app would support an application-
specific axis selector, with additional UI and callbacks to maintain. This
feature uses the standard TensorBoard update axis. Revisit the display choice
if comparing runs by games or accepted learner decisions becomes a requirement.

The optional dependency and the HTTP server are scoped to interactive watch.
The training process, persisted HPO formats, and outcomes keep their existing
contracts. The log is derived anew on watcher startup and is for observation.

Usage is documented in the [HPO operator guide](../hpo.md#tensorboard-で学習曲線を表示する).

## Run overview follow-up

The watcher projects the current terminal observation into the Overview tab.
Four groups give the run state, current batch, committed progress, and most
recent committed update. The table shows freshness explicitly and uses the
same rules as the terminal for ETA and transient values. The standard Time
Series tab continues to show verified per-update history.

Use one atomically replaced snapshot and TensorBoard's native Text renderer.
Appending text events on every sample would retain transient state and grow
the local event log. A custom frontend would require a separate design and
maintenance path. The bounded snapshot keeps one current view with no extra
telemetry reads. The adapter relies on the pinned Text plugin API. Recheck its
listing and item endpoints when updating TensorBoard.

Independent correctness and architecture conformance reviews cleared the
Overview follow-up. A live PTY smoke on the same Pestle–Mirage parent found
the native Overview tab and run selector through the TensorBoard HTTP API.
The four groups showed committed step 23, the current observation refreshed
in place, and the Time Series entropy endpoint returned all 23 points.
Ctrl-C stopped the server. The focused suite listed below returned 127 passed,
two skipped, and one deselected after the follow-up. Scoped Ruff, mypy, and
`git diff --check` passed.

## Verification

The worktree's Python 3.14.10 environment resolved TensorBoard 2.21.0 and its
dependencies without replacing any previously locked distribution version.
The original learner's Python files and environment stayed in its separate
checkout.

The interactive CLI smoke on live Pestle–Mirage parent
`83355bf4-1cf3-43c3-94be-9dae91de7290` displayed the loopback URL on its own
line in about 1.3 seconds. The TensorBoard HTTP reader returned committed
steps 1–11 with loss and entropy equal to the archived update values. It
then returned newly committed step 12. Ctrl-C restored the terminal and
closed the loopback server. The run continued independently.

On that parent, a 10-update history scan took 165 ms wall time. Twelve
unchanged reads had a median of 0.33 ms each. These timings cover the history
reader on the active machine and exclude the TensorBoard/browser pipeline.

- PASS `UV_NO_CACHE=1 uv run --no-sync pytest -n 0 -q tests/test_hpo_tensorboard.py
  tests/test_hpo_progress.py tests/test_hpo_update_storage.py -k
  'not match_progress_is_event_loop_observation_of_acked_terminal_records'`
  with 122 passed, two skipped, and one deselected.
- The excluded match-progress fixture requires `gf_native`. Its bounded
  isolated rerun with the original native package stalled, and an 8-second
  timeout also expired on the original HEAD with that fixture alone.
- PASS scoped Ruff and mypy, `uv lock --offline --check`, and `git diff --check`.
