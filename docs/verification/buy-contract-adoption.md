# Corrected Buy contract adoption and fresh pilot

Date: 2026-10-28. Scope: the user's authorized Bots update and rerun after
Simulator purchase-control confirmation. Design is accepted in
[ADR0007](../adr/0007-staged-buy-contract-adoption.md). Implementation verification
and independent final correctness/conformance reviews are READY, without
blocking findings. Fixed live evaluation and the complete fresh feasibility
pilot have not yet run at this point.

## Adopted identity and boundary

Simulator pin: `860d8b92db4ce82012e040510f221b52241f28aa`. HTTP `/v1`,
`true-match-1v1-v1`, Observation `"1"`, typed Action metadata 1, TestServer info 1
and catalog `public-ja-v1` are unchanged. The eight reviewed normative hashes are
in ADR0007 and `gf_common.domain.REVIEWED_SPEC_HASHES`; actual checkout snapshots
are recorded independently in run provenance. The user's dirty Simulator docs
are preserved. No Simulator file was edited or binary rebuilt.

Prebuilt `gf-test-server` SHA-256:
`608805f07c59164cf1cac18ee9e7a73a13f02b5aa0a5b31edb216070903be41c`.
Its build revision remains unverified; checkout HEAD is not build proof.

The shared strict Observation validator admits only the current nonterminal
buyer's `resolving / buy_accept_reject` input, with the exact existing
`buy_decision` schema. The preceding target response remains a separate defense
decision. Actor selection uses current_player, including when active_player
differs. Nonactors, terminal states and other resolving choices are not opened.
Same-revision scoped ActionOptions and own perceived catalog rules remain.

Journal/derived schema 4, run manifest 3, policy input/feature/encoder/generator/
scorer/catalog/reward/projection identities retain their formats and meanings.
Existing linked manifest checks reject old Simulator revisions/spec snapshots.
Inference checkpoint schema 2 now requires the public Simulator revision;
schema 1 or missing/wrong identities are rejected before native load/listener,
including low-level toy loads. Training completion remains schema 1. No old
dataset, checkpoint or failed pilot is migrated or relabeled.

## Tests and package verification

| Check | Result |
| --- | --- |
| Ordinary suite | PASS: 428 tests, 4 opt-in skipped, 30.31s |
| Real TestServer integration | PASS: 4 tests, 2.74s |
| Ruff / mypy | PASS; 33 source files typed |
| Independent correctness | READY; 111 related tests, 4.79s |
| Independent architecture conformance | READY; 98 protocol/config + 4 native checkpoint tests |
| Wheel build / separate baseline environment | PASS: all source bytes match; installed imports/configs without ML/data extras |
| Separate native wheel environment | PASS: schema-2 toy load/prediction, clean-worker cleanup and old schema rejection |

Checks used `env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/gfb-uv-cache uv run --no-sync`
for pytest/Ruff/mypy, `GFB_RUN_INTEGRATION=1` for the opt-in suite, and
`uv build --offline`. Separate wheel environments are `/tmp/gfb-wheel-verification`
(baseline, optional extras absent) and `/tmp/gfb-phase4-wheel-ml` (existing data/
LightGBM extras). Execution from `/tmp` with `python -I` loads installed packages.
Python 3.14.9/GIL, LightGBM 4.7.0 and NumPy 2.5.4 are unchanged.
Wheel SHA-256 `5914abf5f5672a1f503d4d30e5f6fd8cb63783addf140346d4fac52e3e50f97b`,
83,827 bytes. The old schema-1/pin reader wheel remains in
`dist/archive/0220e4e1d3ef0f2189a7057c72972c69c2d409d32f78ac3ec84578e894c4010f/`.

The new real integration adds the saved six-action prefix, target Allow and both
purchase outcomes through strict HTTP, same-revision options, fresh workers,
journals and Player Replay. The prefix/seed stays in test control, never worker
inputs. Purchase tests preserve nonterminal closure without invented rewards.

Manual synthetic component evidence:
`runs/lightgbm-toy/phase4-af66e150d163/`. The existing copied numerical verification
script created a fresh schema-2 artifact pinned to the new Simulator. Native
fit, exact save/load, candidate permutation, dedicated worker, cleanup, history
encoding and formal toy rejection passed. As before, the synthetic card-preference
task is 48 fit/16 validation rows, 280 features, 40 rounds, L2
`4.053621024660071e-13`; it is not measured game return or model quality.
Direct inspection of the actual old schema-1 toy checkpoint at
`runs/lightgbm-toy/phase4-b2ded98dd7ab/fit/checkpoint` now correctly rejects it.
The old bytes are untouched.

## Frozen rerun conditions

The only preset change is the new Simulator revision. Actions, models, sampling
seeds, GameSeeds, order, seat swaps, concurrency and budgets remain as declared.
All runs receive new unique directories and retain earlier failed evidence.

- Validation: two GameSeeds × swapped seats × sampling seeds 17/29/43 = 12 games.
  Effective preset hash `9c79ec3006b793adf49e7ab7792a4ebc89f304dde8e44a1198d527026589b9cc`.
- Pilot: 32 GameSeeds × swapped seats × the same three sampling seeds = 192
  games, exactly 64 per policy-seed group. Effective preset hash
  `935b446a346624f5c116a9d00706b818d1eb36c355b8a0e786fa7fa0c12bf5a1`.
- Generator `public-atomic-subset-v3`, fixed scorer `public-content-v1`,
  ActionOptions enabled, heuristic vs Random, two game slots, original 7,200s
  wall/5,400s admission/7,140s processing budget and existing cleanup reservations.
- New Simulator gameplay changes Decision/Event/RNG order. Reused seeds do not
  imply identical trajectories or a causal claim about model quality. Mirror
  support is not added; existing supported Allow remains usable. Simulator09's
  full-possession Buy behavior is explicitly project-provisional.

Root will commit the reviewed integration before these runs so the recorded Bots
source is clean and fixed. Gate assessment recomputes all linked original
journals/report/Replay/health with `gf_training.gates.feasibility`; no successful
episode substitution, guessed bootstrap, or edited success sidecar is allowed.
Any failure leaves the gate unmet. A clean pilot is provisional execution
feasibility only; no real-data fitting or model-quality comparison is part of
this rerun.

## Rerun result

NOT RUN pending final code verification and independent reviews.
