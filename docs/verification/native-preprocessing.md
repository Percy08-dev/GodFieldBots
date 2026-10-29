# Native preprocessing verification

Design: ADR0011. Implementation and final acceptance IN PROGRESS.
Reference implementation:42ce5e6, archived at `/tmp/gfb-native-reference-42ce5e6`.
The six completed comparison fits and their frozen data remain unchanged.

## Dependency and ownership foundation

Disposable probe PASS on CPython3.14.9/GIL, NumPy2.5.4, Rust1.99.0:
Maturin1.15.0, PyO3 0.30.1, rust-numpy0.30.0, ndarray0.17.2.
Pointer sharing, readonly/setflags rejection, surviving slices after session close,
retained budget leases and release after final view GC, detached cancellation and
failure cleanup passed. Production pinning uses Cargo.lock and the uv workspace;
baseline dependency versions were unchanged by uv.lock resolution.

## Current scoped results

- Pure feature/actor validation:60Python differential tests and8Rust tests PASS.
  Independent92additional mutations and29,983random finite-float canonical
  comparisons PASS. This is not a whole-source validator acceptance.
- Metadata generation initially bypassed budget/deadline. Fixed pre-admission
  and controlled conversion; independent22feature tests/focused probes PASS.
- Python learning adapter:19tests PASS, exact old transforms/GAE/global advantage
  and legacy NPZ/shard byte equality, zero-decision and both-view reward checks.
- Dataset address-limit/config checks:35tests PASS at current config milestone.
- Initial synthetic native session/manifest/numeric tests PASS; first real corpus
  attempt exposed an empty memoryview writer bug, fixed by skipping zero rows
  after reward validation;2focused zero-decision regressions PASS.

## Initial full-corpus correctness

`runs/assessments/native-first-corpus-equivalence.json` records both83,783-row
feature recipes. Dataset and sample identities match the frozen originals.
History NPZ `c5c3f3c1ef4068766d81c7bd0cfa7a8957b140ad7ea1f207374fc725dcbcc94d`;
observation NPZ `841c6f885e8d8f052abde2c5db788de761279947e36300e7bc87ff0254e7d0c8`.
Both cleanup paths complete. Managed peaks313,260,341/307,060,399bytes under2GiB.
Incidental wall111.122/104.688s: correctness smokes, NOT paired performance
acceptance and not the final parallel/cancellation/index implementation.
Binary SHA `043bad7d324f968f1dbf725d41bd52b96fd4c57b5ae6fa43ef90e437f4826bf1`.
Python bridge snapshot `/tmp/gfb-native-corpus-src` isolates concurrent edits.

## Cancellation evidence

Sandboxed asyncio thread notifications fail because the sandbox denies the
self-pipe socket send (EPERM). This was independently reproduced by the advisor;
it is not a CPython cancellation defect. Async executor/PPO verification uses
approved local unsandboxed execution. No periodic async polling workaround was
introduced. The completion/cancel race requires an explicit async checkpoint
before returning success. Owned workers are joined before releasing authority.
Post-fix scoped fake/native lifecycle and PPO checks PASS; final combined checks
on the frozen release binary remain required.

## Outstanding acceptance

Final reviewed source:347104771a3a1a9c83da56c2f21de44eacc2512e06e653a7dfeedbd2fcd1c2f9;
native wheel SHA1a0b1746b5582df0f2403230a6e406efadd28a64c3babbd354dfa20437e7f34f.
Final review deltas independently READY: strict source boolean mode, explicit
fit wall/work preservation and outer control metadata admission/lifetime.
Targeted29Python and31Rust tests PASS; final whole native pytest751PASS,
5opt-in skipped,252.81s. Ruff/mypy checks PASS.

`runs/assessments/native-final-corpus-equivalence.json`: BOTH recipes reproduce
all original numeric NPZ bytes and the dataset digest; rebuilt index83,783rows
matches row-content SHA574b81ecf6178849c91bea20a9bc0d9906a0348f2c5357825140fc6fc8c618cf.
Incidental elapsed118.716/87.767s and index84.671s are correctness-run observations
with tests running, not paired speed evidence. Original indexes/data unchanged.
Installed application wheel SHA280a7624b13e606524472a97b7913790e9fe7bdd9329c0bd68ba644f8a06101b;
all59Python files byte-match source. Separate installed baseline help/config PASS
without native/Torch/LightGBM/PyArrow; installed native feature arrays exact against
Python with readonly FloatOwner base. Installed CPU MLP/LightGBM inference PASS.
Dependency files were reused from verified environments, not freshly resolved.

Current frozen wheel (before final review deltas): extension SHA
`3335ada4fea282f42f9190e78271846462a09abdd421233d56fa2d355db1d5ac`,
native source SHA `8aea26b60843545287a1cda3f7e2c09ebcc924263133e89f153c6494b57e8b3b`.
Full native-enabled pytest:743PASS,5opt-in skipped,229.16s. Whole Ruff and
mypy59source modules PASS. Scoped JSON deadline and source linkage findings were
independently closed; final strict boolean/control-budget and explicit fit-wall
review deltas still require focused rechecks before acceptance.

Complete/review JSON long-token polling, explicit depth limits on actual stacks,
episode/header/viewer linkage and source compatibility, index typed-buffer output,
shared parallel budgets, fixed-fit async lifecycle, full pytest/Rust/lint/type
checks, installed wheels, three paired full-corpus timings, reference fit, and
bounded real two-update/resume tests before switching production presets.
