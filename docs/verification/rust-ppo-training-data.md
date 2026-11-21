# Rust PPO training-data verification

## Pinned inputs

The implementation branches from `aab5f35d79a2da3339c25110c5012c3c19e9946c`.
Simulator `072dabbca1458c9526897781d2fd64edeeb4f9ca` supplies API v2,
`true-match-1v1-v2`, ActionConfig 3, and the `public-ja-v3` catalog at schema 2.
The catalog SHA-256 is `f51a3c73feb8b3f01837e65768aafb5a3aa805a33483308130244b33b16a8ad6`.
External interfaces §21.1 and §47 were read at that revision. The specification
file SHA-256 is `8e5223b45a6e8c4c80e272736ed97b9e271136c939eca74427a035ddfdbfcac2`.
The executed test server binary SHA-256 is
`76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d`.

The architecture reviewer cleared the live and saved designs before
implementation. The review resolved the shared recording budget, the
pre-seal versus post-seal evidence boundary and the Model4 empty marker.

## Baseline recordings

Baseline trials were run sequentially on an RTX 5070 Ti with the same pin,
seed 17 and the existing original resource and measurement settings.

| Model | Verified trials | Collection seconds | Preparation phase seconds | Update/save seconds | Whole wall seconds | Process-tree PSS bytes | Device VRAM MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Model3 C16/W16, 256 games, inference cap 64 and GPU learning width 16 | 3 | 64.34 | 119.42 | 16.47 | 230.64 | 2,969,105,408 | 3,603 |

Each value is the median. The host preparation phase contains collection
and Source creation. Device VRAM is sampled device-wide.

The Model4 C1/W1 16-game, 64-decision baseline passed three fresh pause and
resume invocations. The median fresh wall was 27.14 s and median resume
wall was 16.43 s. The exact-D resume audits and restores the checkpoint
without adding Adam steps. A separate two-update comparison is required for
continuous versus resumed numerical identity.

The original command/config/observer records and results are under the primary
checkout's `runs/ordinary-rust-ppo-update/rust-training-baseline-aab5f35` and
`rust-training-model4-baseline-aab5f35`.
