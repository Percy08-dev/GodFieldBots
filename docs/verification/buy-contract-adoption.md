# Corrected Buy contract adoption and fresh pilot

Date: 2026-10-28. Scope: the user's authorized Bots update and rerun after
Simulator purchase-control confirmation. Design is accepted in
[ADR0007](../adr/0007-staged-buy-contract-adoption.md). Implementation verification
and independent final correctness/conformance reviews are READY, without
blocking findings. The fixed 12-game evaluation and the complete new 192-game
pilot both finished naturally without failures. Full-source feasibility
recomputation passed; this is a provisional learning-start feasibility gate,
not model training or model-quality evidence.

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

Root committed the reviewed integration as
`da20c9ddc869399e79ff755f02fb3f9e4399bdf1` before these runs; both manifests record
that revision with `dirty=false`. Gate assessment recomputes all linked original
journals/report/Replay/health with `gf_training.gates.feasibility`; no successful
episode substitution, guessed bootstrap, or edited success sidecar is allowed.
Any failure leaves the gate unmet. A clean pilot is provisional execution
feasibility only; no real-data fitting or model-quality comparison is part of
this rerun.

## Rerun result

| Run | Natural / planned | Failures and health errors | CLI-reported elapsed |
| --- | --- | --- | --- |
| Validation `runs/buy-contract-validation/baseline-validation-c5e164f6-2c4` | 12 / 12 | 0 | 5.645s |
| Pilot `runs/buy-contract-pilot/baseline-feasibility-pilot-da11e8fb-9f8` | 192 / 192 | 0 | 64.906s |

Both comparisons are valid; no action rejection, unsupported state, truncation,
save, communication or process-health error was recorded. Validation has 179
accepted decisions and pilot 2,982. Pilot contains 26 completed purchase choices
(9 accept, 17 reject), so the corrected path is exercised during actual play.
The three policy-seed groups each contain exactly 64 natural games. Pilot's
heuristic perspective is 190 wins, zero draws, two losses; it is not evidence
about a learned model. The older failed pilot used generator v2; this rerun uses
the already accepted v3 fix and corrected Simulator, so failure-count changes
must not be attributed to one change or treated as a causal quality comparison.

Commands (with the same `env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/gfb-uv-cache`
prefix) changed only the output directory through an operational CLI override:

```console
uv run --no-sync gf-bots evaluate --config configs/experiments/baseline.toml --set 'run.output_dir="runs/buy-contract-validation"'
uv run --no-sync gf-bots collect --config configs/experiments/pilot.toml --set 'run.output_dir="runs/buy-contract-pilot"'
uv run --no-sync python /tmp/gfb-assess-buy-pilot.py runs/buy-contract-pilot/baseline-feasibility-pilot-da11e8fb-9f8
```

The actual effective config hashes including those output paths are:

- Validation: `9dcffd23b5232ac9283d40cdfb9800d93276e1449bb0742fa3508ce494e3ccb4`.
- Pilot: `8fdc2afb0447809e5a94cc29e0586ef20e95fc764f683275fb04cb1d2e2e690d`.

The assessment used the actual `gf_training.gates.feasibility` implementation to
validate the fixed schedule, all 384 participant journals, original health/end
records, report/Replay consistency, split and semantic identities. It measured
2,982 positive acknowledgements, equal to recorded decisions, with no pending
record. The source action-config hash is
`90efec6c9c4d3f3380ceb57ca95d6a9c23f32c88450f91184055b31adc11e44d`.
Pilot manifest collection time is 64.903s within its original 7,200s budget;
the returned total includes final persistence at 64.906s. Shared feasibility
validation took a separately measured 3.471s. No new collection budget was
granted to replace failed games.

Assessment:
`runs/assessments/baseline-feasibility-pilot-da11e8fb-9f8-feasibility.json`,
`qualified=true`, SHA-256
`ab227b380cbd62bcb529aeadbc1dc010b5026bab3ac684aeb67929c70a08c404`.
Validated source-set hash:
`5ff2af59443bb258c17203791568785e67adf876294835071a6769c680b4bb04`.
The raw CLI manifest's generic `learning_gate=not_run` field is retained;
external recomputation is recorded separately, and real training recomputes the
proof from original sources instead of trusting this assessment flag.

Independent post-run correctness evidence review is **READY**, without blockers.
Its read-only probe reconciled both complete schedules, 24/384 journals,
179/2,982 positive acknowledgements, the 64-game policy-seed groups, absence of
pending/health/process failures, fixed code/config/pin/spec/budget identities and
report/assessment hash links. It verified the old pilot remains incomplete.
The reviewer did not rerun full projection, another pilot or fitting. This
supports the provisional gate for these fixed conditions; it does not guarantee
zero failures in future collection or equivalence to live-game rules.

| Evidence | Manifest SHA-256 | Report SHA-256 |
| --- | --- | --- |
| Validation | `8568f23517dc7d846d6d8297ab39cc13b33f7720977bb11220625f53cbcace31` | `2245cd30802d7b2b1a944e29fe0143f9b5ee256d69e186c829ef90bb19661eed` |
| Pilot | `7fe90a7baa9abb8df6807ccfc237cafd8b2505018c4f0a7ab7fded2f3b7c9d37` | `cbd88a66e1cab8d39427b31187d0a76d22e798a0b7ca2f8934b93cd9527af65a` |

Source HEAD, all eight reviewed document hashes and the binary hash matched
before and after both runs. Simulator worktree edits were preserved. Validation
data export covers all 24 journals/179 eligible selected decisions: two independent
Parquet regenerations are byte-identical, SHA-256
`5058dd34856541c6a61da30395b940eec0727ad58acba55037538d1d54df8d78`,
under `/tmp/gfb-buy-contract-export/validation-{a,b}.parquet` with sidecars.

The actual earlier complete v3 validation source is rejected under the new pin
(`incompatible source simulator semantics`), and the retained failed 192-game
pilot remains rejected (`training requires a finalized complete healthy run`).
No original artifact or result was edited. GPU, real-data fit, model selection
and a learned-policy quality evaluation are **NOT RUN**. A future training run
still needs explicit complete train/validation sources and a preregistered
accepted-decision target; no dataset or target is invented by this acceptance.
