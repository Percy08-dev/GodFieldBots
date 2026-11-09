# Torch-preloaded real CPU worker feasibility — 2026-11-09

Status: disposable healthy-path experiment; no launch provider adopted.
Application/test source remains `e869bc3`, the accepted ADR0029 runtime.

## Question and bounded method

The owner asks whether repeated Torch initialization can be avoided. CPU actors
can inherit module-only state from a preloaded template while each actual actor
is newly created and reaped. Root therefore tests the ORIGINAL three CPU neural
protocol cases, MLP/GRU/Transformer with two frames each. Every original assertion,
factory, bootstrap parser, generator, seed and selection runs unchanged.

The separate normal and forkserver controllers both run on HOST under the same
explicit minimal CPU environment and `python -E`. The forkserver uses an exec-
started standard-library template preloading only a disposable Torch helper,
excluding `__main__`. Before any application import, template/child receipts
confirm Torch inherited, visibility empty, CUDA uninitialized, one Python/native
thread, and no application/test modules. Model/options/frame data travel directly
through the original real unbuffered participant pipes, not through the template.
The multiprocessing startup arguments contain only a fixed bootstrap descriptor
and four transferred OS handles. Internal trusted process startup serialization
is not a dataset/model format or untrusted policy-data deserializer.

A disposable Popen-subset facade invokes actual multiprocessing process status,
terminate/kill/join; original worker and ReservationLedger perform normal framing,
validation, close, I/O retirement and lease release. It never invents a PID, exit
or released reservation. The ambiguous forkserver status255 is rejected. Only
one actor is live at a time; this is not a four-actor collection/service test.

## Results

Both modes PASS all3original cases. Normal pytest duration3.93s; forkserver1.81s.
Actual per-actor startup after the separate template warmup:

| Model | Normal Popen startup s | Preloaded fresh child startup s | Normal first select s | Preloaded first select s |
| --- | ---: | ---: | ---: | ---: |
| MLP |0.772456|0.086772|0.011649|0.011507|
| GRU |0.782157|0.081809|0.011743|0.011169|
| Transformer |0.781959|0.092370|0.011479|0.011197|

Template setup costs0.733241s up front, including its actual0.697156s Torch import.
It is not silently removed from the controller timer: measured times are
4.286240s normal versus3.082507s forkserver, including template setup and observed
cleanup/source-snapshot work. This timer starts inside `main()`, after interpreter
startup/top-level imports, and is not an external process-start-to-exit timer.
These are single healthy observations, not a whole pytest/PPO speed estimate.
The model-loading portion becomes0.020-0.024s while
normal cold loads are0.698-0.708s; first selection does not absorb the import cost.

All six decision request/reply comparisons (two frames per model) agree exactly after removing
only elapsed timing dictionaries and three generated-report timing fields.
Candidate columns, actual frame/delta, search RNG, index/logprob/value and policy
identities remain exact. Each original case additionally checks the complete
frozen CPU network and its own checkpoint binding. Ready rows/initial checkpoint
paths are not claimed byte-identical across independently generated artifacts.

Three different fresh forked actor PIDs are direct children of the attested
template; none is the template. All normal/forked actors finish by the original
SIGTERM close (actual exit-15), with confirmed reap, released lease, empty SAME
ledger, dead I/O thread and absent `/proc/PID`. Template and resource tracker
also stop/reap and disappear after explicit bounded cleanup. Tracker exit0.
All568tracked source/test/configuration hashes and repository source diffs remain
unchanged. This does not measure aggregate peak memory or every cleanup failure.

## Scope and architecture disposition

The current template is started by a deliberately clean experimental controller.
An arbitrary pytest process may contain environment/CUDA/application state, so
the experiment is NOT a production launch design. An accepted test launch
provider needs a separately owned clean-exec supervisor, explicit template/
resource-tracker memory/lifetime accounting and descriptor closure. Thread/CUDA
snapshots are necessary observations, not complete general fork-safety proof.

Partial start, unknown actor PID, lost acknowledgement, template/supervisor death,
cancellation, uncertain reap and launch cleanup are NOT verified here. Those
conditions must preserve actual ownership, first deadlines and failed/ineligible
results without fake status/early lease release. The facade is not an approved
foundation and must not be promoted merely because healthy tests pass. Normal
CUDA paths, unsupported platforms and baseline optional-free behavior also need
explicit routing before any adoption.

Fresh forked actors isolate per-actor model/private state, but do not test cold
interpreter/Torch import. Keep independent normal cold-exec protocol coverage.
Moving the current48-import MLP reference needs an explicit coverage decision
and evidence. Actual PPO collections/resume, concurrent actors, failure matrix,
peak memory, Source qualification and model quality remain NOT RUN.

Architecture advisory finds a module-only CPU template credible and smaller than
a session-wide participant service: actors can still genuinely die before every
optimizer update. Existing scoped exclusions do not forbid a new design, but a
new architecture gate and independent implementation reviews remain required.

## Evidence and reproduction

Files are under `/tmp/godfieldbots-preloaded-real-worker-20261109/host`:
`controller.py`, `torch_template.py`, `preloaded_actor.py`, `compare.py`, script
hashes, separate baseline/forkserver summaries/XML/raw pipes and `comparison.json`.
The earlier sandbox normal-mode baseline stays at its original parent-directory
path; it was not moved/overwritten to pretend it was the matched HOST run.

Matched HOST command, run separately and sequentially with MODE set to `baseline`
and `forkserver` and corresponding log filenames:

```sh
env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 timeout --signal=INT --kill-after=10s 180s /home/percy/develop/GodFieldBots/.venv/bin/python -E /tmp/godfieldbots-preloaded-real-worker-20261109/host/controller.py MODE
```

Then run `.venv/bin/python -E /tmp/godfieldbots-preloaded-real-worker-20261109/host/compare.py`.
Receipt publication is atomic (`.tmp` then replace). No repository implementation
or dependency changed and no full-suite rerun is claimed for this experiment.

Mechanism: [Python3.14 forkserver preload](https://docs.python.org/3.14/library/multiprocessing.html#multiprocessing.set_forkserver_preload).
Relevant limits: [PyTorch2.14 multiprocessing](https://docs.pytorch.org/docs/2.14/notes/multiprocessing.html).
