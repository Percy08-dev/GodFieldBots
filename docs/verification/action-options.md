# Authorized ActionOptions integration

Date: 2026-10-27. Root accepts ADR0005 implementation and its bounded generator-v3
amendment after independent correctness and architecture conformance reviews
returned READY. This is implementation acceptance, not a passed learning gate.

## Authority, scope and version boundary

The user's updated Simulator07 §§13/21.1 authorizes current-actor ActionOptions
as optional atomic hints. Bots AGENTS and implementer/reviewer/advisor definitions
were synchronized; role TOML parsing and strict Codex configuration loading passed.
Models/effort/routing are unchanged; fresh role threads confirmed authorization.
SequenceOptions, rejection-derived masks, other-participant private inputs and
RNG/control data remain excluded. No new dependency was introduced.

- Default `actions.use_action_options=true`, deliberate opt-out only.
- Dedicated actor GET, same revision/actionable Observation, strict closed DTO,
  hand/learned/target membership and actor-bound normalization.
- Worker and generator repeat validation; missing/malformed/stale/foreign
  requested hints stop before policy selection/Act, with no automatic fallback.
- Use usable arrays for play/defense, dedicated trade-carrier arrays, target
  intersection and basic flags. Discard/sold objects are not filtered by the
  usable-play list. Normal protected Discard follows the user's primary developer
  announcement; no unsupported Sacrifice extension is invented.
- Only the dedicated paired snapshot supplies policy hints; Act response and
  Replay options are not reused. Events remain event-API-only. Visible card
  definitions remain perceived own-hand/learned definitions.
- Journal/derived schema 4, run manifest 3, feature `participant-json-v2`,
  policy input `player-atomic-options-v1`, generator `public-atomic-subset-v3`.
  Scorer/reward/end-projection/catalog/Simulator wire versions are unchanged.
- Historical schemas 1–3 and unsupported generator-v2 sources are explicitly
  rejected; no availability or candidate lists are regenerated for them.
- Exact delivered options, mode, owner/revision/references and config hashes
  survive worker serialization, journals and deterministic export.

Simulator HEAD remains `44a26e3a16b808e826deb332fa72acfc5037d3ef`, with user-edited
normative documentation. The three reviewed document SHA-256 values in ADR0005
are recorded as `reviewed_simulator_spec_sha256`; optional local source hashes
are `simulator.spec_sha256`, or null without a checkout. Dirty source state and
unknown prebuilt binary build revision remain explicit. No Simulator file was
modified or binary rebuilt by this task.

## Checks and independent reviews

| Check | Result |
| --- | --- |
| Final ordinary suite | PASS: 271 tests; 3 opt-in skipped |
| Final real-binary opt-in suite | PASS: 3 tests, 2.10s |
| Ruff / mypy | PASS; 22 source files typed |
| Independent ActionOptions correctness subset | PASS: 115 affected tests; no blockers |
| Independent ActionOptions architecture subset | PASS: 166 affected tests; source/docs conform |
| Independent v3 amendment | PASS: 17 amendment + 74 related tests in each review; no blockers |
| Strict Codex config / role TOML | PASS: exit 0, all five custom roles/links retained |

Tests cover raw/scoped DTOs, revision/actor/ref/flag/choice failures before
selection, mode/missing-hints enforcement, perceived Dream definitions,
per-purpose masks, protected normal Discard, target/card ordering, budget/basic
retention, source-hash provenance with/without checkout, schema/mode snapshots,
export and historical rejection. Existing cancellation/finalization/health and
paired-evaluation tests remain enabled.

The declared action space is still a supported subset. Atomic availability does
not establish arbitrary combinations, ordering or resource resolution. The v3
amendment supports an own hinted `atkBy2xMP` weapon only as a target-specific
singleton, outside prefix pools; it does not simulate MP effects or change the
fixed scorer's approximation for absent printed attack.

## Fixed live evaluations

The exact same preregistered 12-game schedule (two GameSeeds × swapped seats ×
three policy seeds) was repeated as new experiments, preserving old failures.

| Run | Result |
| --- | --- |
| Historical observation-only run | 8 natural, 4 rejections; invalid comparison |
| v2 atomic hints: `runs/action-options-validation/baseline-validation-67fca993-71b` | 12 natural, 0 rejection/failure; CLI total 5.636s |
| final v3: `runs/action-options-v3-validation/baseline-validation-c0f6ea4f-073` | 12 natural, 0 rejection/failure; CLI total 5.844s |

The new available-card set changes policy sampling and subsequent game paths;
shared seeds do not imply identical future trajectories. These are feasibility
checks, not claims of improved model quality.

## Complete failed feasibility pilot — retained, not replaced

Preset `configs/experiments/pilot.toml` was fixed before the new validation results,
SHA-256 `ec526bc49d8e656b1bfcf66a744f3b47f8f2d09720c3c3ff5b3132d0aaddd4f5`.
32 GameSeeds × two seats × policy seeds 17/29/43 = 192 games, exactly 64 per
policy-seed group. Same search/policies were frozen, concurrency two, budget
7,200s including resource cleanup/reporting. No learning updates ran.

Run: `runs/pilot/baseline-feasibility-pilot-354f64ce-747` (generator v2).

- All 192 planned games recorded; CLI elapsed 62.510s, within budget.
- 182 natural; zero action rejections.
- Nine unsupported contracts at pending BuyDecision: resolving phase with
  buy_accept_reject/current actor, but actionable false and empty typed metadata.
- One empty supported candidate set for an available Magical Stick, whose attack
  is dynamic and has no printed attack field. One cleanup health flag accompanied
  this aborted worker attempt.
- The whole run is incomplete and **learning gate remains unmet**. Successful
  episodes were not substituted or used as a successful comparison/training run.
- Root assessment: `runs/assessments/action-options-v2-pilot.json`, linked to the
  original manifest/report hashes, records qualified=false and all three 64-game
  groups. Diagnostic export sample: six participant sources/52 rows, zero eligible
  rows, `/tmp/gfb-action-options-v2/pilot-diagnostic-sample.parquet` plus sidecar.
  It is explicitly a sample, not a claim that all 384 sources were exported.

Manifest SHA-256 `2450c3cc3c72c81e27985081a55df723e308705b95a98bc03fd9d88e38969238`;
report SHA-256 `a17108ea5810d919a38389644cd7ed59e2df496991df7332ceaecb0b611ffd4a`.
The exact schema-4/generator-v2 reader is preserved as wheel
`dist/archive/4a65ed311e841764523d31a26528383d7443d7aeaa3df6b0bec514d2d8c1cbc0/godfieldbots-0.1.0-py3-none-any.whl`.
Do not reinterpret v2 snapshots with generator v3 or rerun only failed games as
pilot replacements. No second large pilot is claimed after the v3 fix while the
known Simulator Buy metadata contract remains unresolved.

## Narrow follow-up reproduction

The Magical Stick case was fixed under a separately reviewed ADR0005 amendment.
`runs/diagnostics/magical-stick-2ce39b120eb4` replays the saved 33 successful public
HTTP actions, obtains the same actor's revision-33 Observation/options, generates
exactly two singleton targets and submits the fixed opponent-target candidate.
It is accepted, with clean deletion/EOF exit 0 in 0.229s. Case SHA-256
`c3a3ffd378252096b8df07a22d752cf3f1d13d02157e4e940efb9cfd4108887b`.
It is a diagnostic artifact, not a relabeled training episode or policy retry.

The nine Buy cases are a confirmed [separate Simulator task](../issues/0003-buy-choice-control-metadata.md).
The shortest six-accepted-action prefix reproduces false/empty metadata in
`runs/diagnostics/buy-contract-a9d2723895e3`, 0.099s, EOF exit 0. No action follows
false scheduling metadata; no descriptors or wins/losses are fabricated.

## Acceptance limits

ActionOptions adoption and the supported singleton extension are complete.
The former blanket prohibition and speculative need for opponent catalog access
are superseded by the new authorization. This does not authorize SequenceOptions
or rejection probing, guarantee every completed action, resolve Buy metadata,
complete the clean learning gate or implement Phase 4–9 training/models.
Existing OS/FS timing and dual-storage-correction quarantine debt remain explicit.
No commit/publication was performed.

## Final frozen package/coverage verification

- Ordinary suite under temporary coverage: **271 passed, 3 opt-in skipped,
  7.76s**. Appended real integration: **3 passed, 271 deselected, 2.51s**.
- Parent-process coverage: 2,260/2,548 statements (**88.7%**) and 897/1,106
  branches (**81.1%**). Candidate generation 91.2%/87.2%; validation 89.7%/85.2%;
  environment 87.6%/75.8%; journal/export 88.8%/78.2% (line/branch).
  Worker/module subprocess execution is not collected, though separately tested.
  No dependency or threshold was added to the project.
- Coverage artifacts: `/tmp/gfb-action-options-coverage/coverage.json` and
  `html/index.html`. Measurement uses the same temporary `uv run --no-sync
  --with coverage` method as the previous coverage record, with a fresh data file.
- Built wheel/sdist; installed wheel with no optional ML/data extras into
  `/tmp/gfb-wheel-verification`, executed from `/tmp` using a moved prebuilt
  binary and no simulator source_dir. All seven packages load from the installed
  wheel; Torch/LightGBM/PyArrow are absent and not imported.
- Wheel SHA-256:
  `03d93d24948ebdc118236dbd3654d7b1303b695dfdf319e94e8d20707beac701`.
  All 22 packaged Python files match final source.
- Installed-wheel smoke: `runs/wheel/wheel-no-ml-a3197664-92e`, two natural games,
  no failures, CLI 1.670s. Reviewed normative hashes are declared and actual
  source hashes are null, rather than falsely claiming a source checkout.
- All 24 participant journals from the final v3 12-game run exported twice:
  **179 rows, 179 structurally eligible rows**, exact actor/revision/choice
  hint snapshots verified for every row. This run's actual local spec hashes
  equal all reviewed hashes. This structural eligibility is not a clean-pilot
  certificate or authorization to train after the separate failed pilot.
- Byte-identical Parquet files `/tmp/gfb-action-options-final/{a,b}.parquet`,
  SHA-256 `c4ed7c4acb30a1b584fd785662ad22d2f25edfdad09da4a034ffbc79d096afe8`.
  `verification.json` records package/source equality and snapshot checks.

Project/role TOML and documentation links were validated. Existing uncommitted
work and user-edited Simulator documents were retained; source and binary remain
unchanged in the Simulator repository.
