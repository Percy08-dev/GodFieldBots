# HPO storage and compression verification

Scope includes TOML-driven campaign placement, mount preflight, pair-owned HTTP
diagnostics, and Python/Rust HPO update records. The dedicated WSL destinations
are `/mnt/wsl/gf_data` (ext4), `/mnt/d` (D drive), and `/dev/shm` (tmpfs).
The working checkout started at Bots `b8a71cc` with existing HPO learner and
checkpoint-evaluation changes. The probe pins Simulator
`072dabbca1458c9526897781d2fd64edeeb4f9ca`, API v2 and catalog schema 2.

## Storage acceptance

- **PASS** — A template copy passed `gf-bots hpo validate` and resolved the
  campaign and split paths. Tests cover absent and read-only mounts,
  conflicting CLI arguments, tmpfs type, nested output/catalog mounts, and
  exact legacy config hash and snapshot round-trip.
- **PASS** — The sandboxed `hpo run --mode fresh` refused the read-only
  dedicated ext4 path before creating the campaign. Host readiness checks
  found writable mounts and an RTX 5070 Ti.
- **PASS** — An isolated host component smoke ran two managed TestServer
  captures on tmpfs with the default policy. The healthy 3,682-byte capture
  was removed with zero persistent HTTP payload. The capture containing an
  actual HTTP 503 kept its 4,622 raw bytes in a 902-byte zstd file on
  `gf_data`. Decompression recovered both the successful and failing
  responses. A 53,248-byte diagnostic SQLite Pack sealed on D. Its eight
  members contained zero HTTP JSONL files. This component smoke uses raw
  Simulator requests and supplies storage and retirement evidence.

## Functional and format checks

- **PASS** — Real managed-server normal, 503 and early zero-exit cases, plus
  a real Rust game pair whose first Act reply dropped and recovered. The
  terminal bundle carried the sticky transport fault outside serialized
  episode data.
- **PASS** — Controlled common HPO committed updates and owned compressed
  pause/resume matched raw continuous weights, Adam and RNG arrays. Format
  tests cover historical JSON, strict inventory, missing optional zstd,
  stored-hash mismatch, malformed frames and output/window limits.
- **PASS** — Offline matching native extension and companion build. Native
  source SHA-256
  `b22acc3ff9d1131dfb2594b22d97635805a92c76a330b11c065a7b4f837f09aa`
  and Cargo lock SHA-256
  `3cbb745c9ec49365d2cf1e2140b95516228431b45097f26bedfbe0b793cb9e87`
  match the inspected release wheel. Offline Rust library tests returned
  218 passed and 13 ignored with the test-only CPython linker flags.
  Supervisor tests returned 27 passed and 14 ignored.
- **PASS** — The final 226-test Python selection covered storage,
  update encoding and resume, evaluation cleanup, campaign and operation
  factories, Rust match handoff and lineage. The separate managed HTTP
  selection passed all 27 tests with the approved Simulator. Ruff, scoped mypy
  across 50 production/operation files, and `git diff --check` passed.
- The initial broad Python sweep recorded 2,151 passes alongside fixture
  failures. After updating the fixtures, all 82 tests in the affected
  evaluation cleanup, audit capacity, progress, and native precollection
  rerun passed.

## Component measurements

One committed update occupied 20,999 bytes as JSON and 5,908 bytes as zstd.
Over 1,000 warm repetitions, median write preparation took 70.3 µs for JSON
and 116.2 µs for zstd. The median reader took 133.4 µs and 154.7 µs.
These measurements exclude fsync, cold processes and gameplay.

A shaped 16,189,476-byte capture on local ext4 wrote zero HTTP payload bytes
under healthy `errors + zstd`. Healthy `all + none` kept all 16,189,476 bytes.
A single 503 under `errors + zstd` produced a 271,371-byte archive and took
11.2 ms in the archive call. Process peak for all shaped cases, including
fixture and interpreter setup, was 24,660 KiB.

## CUDA and 256-game verification after restart

The initial eight-game smoke reached Supervisor admission. The shared
namespace refused new GPU work pending recovery of ticket
`802ebdf8d2b2412e96ff9377bee5e7e0`. Both probe hosts exited before collection.
After the operator confirmed a WSL restart on 2026-11-23, the recovery
command stored its restart receipt and the namespace check passed. The
rebuilt native wheel matched the source and Cargo lock hashes above. CUDA
was available, and the destination and update-storage tests passed (26).
The restart detached the existing `D:\WSL\gf_data.vhdx`. Windows returned
`Wsl/Service/AttachDisk/MountDisk/HCS/E_ACCESSDENIED` on reattachment.
The live preflight refused the missing dedicated ext4 mount before creating
any collection directory. After the operator reattached the VHDX, the host
reported it writable at `/mnt/gf_data` with ext4 UUID
`4d0e68fe-6d63-47a0-9900-43cfede47916` and the previous storage-smoke
data present. The temporary verification configs used this mount, the
D-drive originals path and tmpfs for HTTP staging.

### Pause, resume and saved state

Both JSON and zstd completed two CUDA updates of eight games each. They
paused after update 1 and resumed to update 2. The independent
`--full-source-audit` passed after both commands in each format, including
Journal, Replay and Python numerical reconstruction. The two conditions
used 77 and 54 learner decisions in updates 1 and 2 and reached 12 optimizer
steps. All 60 weight arrays and 182 resume arrays per generation matched
bitwise across formats. The PCG64 shuffle state also matched. Resume
launch-to-collection took 5.42 s for JSON and 5.60 s for zstd. The complete
resume invocations took 17.94 s and 18.31 s.

### Matched first update at 256 games

Both conditions used learner seed 17, the same launch UUID
`69899b52-e549-4129-9cc1-9b47d8cc930f`, the pinned Simulator above, an
RTX 5070 Ti, the rebuilt native release wheel, C256/W16, and fresh storage
roots under `ppo-hpo/storage-acceptance-20261123-restart`. The raw condition
ran first. Each finished all 256 games naturally, used 2,662 learner
decisions and reached 168 optimizer steps. Weight, resume and shuffle
states matched across the two first updates. The independent
`sealed-pack-sha256-v1` audit passed for both first updates. A full
reconstruction at this size was interrupted after more than four minutes in
D-drive file checks. The eight-game probes supply full reconstruction
evidence.

| First-update observation | `all` + raw HTTP and JSON | `errors` + zstd HTTP and updates |
| --- | ---: | ---: |
| Persistent HTTP payload bytes | 115,904,228 | 0 |
| Stored update bytes | 3,587,627 | 329,013 |
| Originals SQLite bytes on D | 21,041,152 | 21,020,672 |
| Sampled tmpfs used-byte increase | 106,762,240 | 63,455,232 |
| Sampled complete process PSS peak, bytes | 3,589,756,928 | 3,342,768,128 |
| Collection, s | 60.119 | 57.833 |
| Sum of pair archive call times, s | 35.588 | 0.000 |
| PPO update, s | 18.780 | 19.412 |
| Whole invocation, s | 92.468 | 99.783 |

The first updates each produced about 110.5 MiB of raw HTTP capture in tmpfs
over their lifetime. The raw condition retained all 128 pairs. The default
condition discarded all 128. Both originals databases had zero HTTP
payload members. The table reports stored logical file bytes, process PSS
from complete sampled sweeps and the separately sampled tmpfs increase.
The archive timing includes I/O and sync calls. In a post-run measurement on
the actual canonical update bytes, median zstd compression took 2.72 ms
over 21 calls per input. Bounded decompression returned the exact canonical
bytes. Each condition's whole-invocation timing is a single observation.

### Interrupted second updates at 256 games

Both 256-game resume commands restored update 1 and entered collection.
Launch-to-collection took 9.93 s in the raw condition and 6.00 s in the zstd
condition. In the raw condition, 145 games finished naturally and 111
reported uncertain HTTP delivery at `POST /v2/games`. In the zstd condition,
255 games finished naturally and one reported HTTP 422 and the
`invalid_parameter` rejection code. The latter run retained exactly one
53,913-byte zstd HTTP archive. It decoded to 666 records, including 655
records before the first 422 response. The two runs committed no update 2
and supply no matched second-update performance or training result. The
GPU admission namespace was clear after both attempts. Capacity for 10,000
updates remains unmeasured.
