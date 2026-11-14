# Refresh heuristic and mixed PPO opponents

Status: accepted and implemented, 2026-11-15. Final correctness/conformance READY;
root accepts bounded functional scope in verification/heuristic-refresh.md.
Heuristic strength, formal qualification and trajectory determinism are not claimed.

## Authority and scope

Add independently selectable `heuristic_refresh` for the corrected card-instance
v2 runtime: smoke/collect/evaluate, ordinary PPO and model4/HPO training opponents.
Preserve legacy policies, serialized defaults, historical records, fixed kernels,
formal qualification and canonical final64/HPO evaluation opponents. No Simulator
changes or dependencies. Simulator pin is
`91d5926a7572db327eaea3751e7c24f428120996`, Gameplay API v2, Observation/events schema
2, catalog schema2/public-ja-v3. Its thirteen normative file hashes were verified
against `CORRECTED_REVIEWED_SPEC_HASHES` during planning.

The user chose maximum predicted damage among eligible attacks, full18 Discard
preparation, both PPO paths, mixed16 with old/new heuristics, and a public observed
interval outcome proxy rather than exact causal attack damage. The user confirmed
that Exchange can be used without changing resources. Global static trade rules
below are builder defaults; there are no danger-dependent tactical exceptions.

## Ownership and boundaries

Common owns stdlib-only closed option validation/identity if shared by config,
worker and storage. Models own participant-local scoring/history and pure renewal
bookkeeping. Client owns transport/process/frontiers. Training owns schedule,
resume/Source and learning. Native validates the same record identities, never
policy math. No new framework, engine copy, model-side I/O or hidden inputs.

Policy consumes only its validated own Observation, participant-visible EventBank
prefix, typed syntax, acting-player same-revision atomic hints and pinned perceived
own definitions. No SequenceOptions, rejection probes, canonical history, RNG,
GameSeed or other participant view. Existing generator/config/rules stay unchanged.

## Closed policy identity

Distinct `RefreshHeuristicConfig`: `model_id="heuristic_refresh"`, `version="1"`,
`attack_threshold=12` (strict positive integer, bool rejected). Do not add fields to
old ModelConfig. Runtime options are exactly
`{"version":"1","attack_threshold":12}`. Fixed policy recipe1 also defines the
three-decision horizon, half-attack ratio, proxy and all tie rules.

The policy label is `heuristic_refresh-card-instance-v2-<sha256>`, hashing the
complete runtime options, corrected action contract, supplemental rule signature
and complete action config hash. Worker selection carries the same label. Only
this model's Journal6 header gains required `baseline_options`; this key is
forbidden on old/learned branches. Collection5/6 permits configured-baseline-only
registries without fake neural_inputs. Both Source readers dispatch this model
before the old non-null-options => learned branch. Native support is additive and
capability-gated before resource admission; old native must refuse new records.
Candidate/token/numeric/neural inference formats do not change.

## History proxy

On selecting an opponent-offense candidate in an own turn_action frame, remember
the validated event frontier, episode, observation turn and calculated TOTAL
attack `Ai=SequenceAnalysis.attack`. It is a selected public estimate, not a
claimed actual attack declaration. Open the observed interval at that frontier;
close immediately before the first public turn_started whose turn exceeds the
selected turn. `Di` sums nonnegative damage_applied amounts targeting the opponent
inside that interval. Secondary effects are deliberately included. No HP-difference
or inferred causal grammar. No matching declaration is required.

Only a closed interval with Ai>0 supplies a sample. Closed/no damage means Di=0;
unclosed/interrupted intervals supply nothing. `Bi=max(0,Ai-Di)`. History uses an
incremental validated own-event cursor. Reset clears all state; changed episode,
foreign/truncated/inconsistent frontiers fail existing contract validation.
Repeated selection of the same frame is idempotent.

Age once per distinct observed own turn_action `observation.turn`; not responses,
buy decisions, repeated calls or automatic skipped turns without such a frame.
A sample is usable at the next three own decisions (ages1..3), then expires.
Nonattack turns age samples without inserting zeros.

With no samples, positive opponent offense qualifies when A>=threshold and ranks
by A. With samples, `predicted=max(0,A-mean(Bi))` and eligibility is A>0 and
predicted>=A/2, even below12. Compare exactly with integer count/sum arithmetic;
do not lose boundaries to float rounding. Rank maximum predicted damage, then
lower spent MP, fewer consumed artifacts, first candidate-table row.

## Renewal and retention

Build current supplied positive opponent-offense witnesses (required slot masks,
calculated total attack). `R(S)` is the greatest witness attack whose required
slots are contained in remaining mask S, otherwise0. This is a PRE-ACTION-MP,
candidate-scope card-retention proxy, not future legality or RNG prediction.
Use an int64 subset-maximum transform for at most18 slots: one2MiB table plus
bounded O(K) facts, in-place NumPy maxima. Keep NumericCosts/generator results
unchanged; local CPU scratch cap and worker deadline own the bounded work. Do not
charge CPU scratch to CUDA reservations.

Renewal facts (P is current possession count):

| Kind | Known removed slots | Acquisition opportunities | Eligibility/uncertainty |
| --- | --- | --- | --- |
| play_sequence | selected artifacts; miracles retained | selected count | utility family AND self target only |
| pray | none, or one unknown slot at P18 | 1 | full18 has R=0 |
| sacrifice | all selected | selected count | existing hinted End Times candidates |
| exchange | carrier | 1 | exact current hp/mp/money allocation only |
| sell to self | carrier only | 1 | perceived price<=own.money+own.mp |
| sell away | carrier and sold instance | 1 | only supplied candidate |
| buy | carrier | 1 | no hidden purchase valuation |
| invoke_phenomenon | carrier | 1 | R=0 for unknown effects |

Own soap/broom with uncertain automatic removals also have R=0. Remaining unknown
secondary harm is not predicted. Acquisitions are proxies, capped at
`min(opportunities,max(0,18-P+known_freed))`; full18 prayer frees one. Required
cost/payment and CostCut suppression reuse public analyze_sequence.

If no eligible offense, rank positive-renewal candidates by descending
min(R,threshold), descending capped acquisition opportunities, lower spent MP,
fewer selected slots, table order. Do not treat own offense/status or opponent
utility/status as renewal. Never synthesize actions.

If no positive renewal and P18, allow supplied normal Discard as preparation,
rank descending min(R,threshold), fewer removed cards, table order. It does not
refill. If neither, choose lowest supplied opponent attack including0, then
lower MP, fewer selected slots, table order. If no offense either, use canonical
old heuristic scoring. Defense/buy choices also use canonical old scoring; adapt
only the returned policy identity. No special danger tactics.

## Mixed PPO recipe

Keep fixed schedulers and all eight AST-frozen learning/schedule kernels exact.
Add concrete wrappers producing exactly16 games: Random4, old heuristic2,
refresh2, snapshot8. Replace the second old-heuristic seat pair; preserve four
snapshot RNG draws, paired roles, game seeds and policy streams.

Explicit optional `training_opponents` descriptor:
`{"version":"1","preset":"mixed-refresh-16-v1","refresh":{"version":"1","attack_threshold":12}}`.
No implicit default insertion into old Config/state. Ordinary model3 uses an
explicit opt-in Config variant; model4/HPO uses campaign/spec schema2 only for
this descriptor, with old schema1 bytes/meaning exact. Bind the descriptor into
effective config, parent/resume/study identity and all live/reconstructed
registries/schedules and Python/native Source checks. Mixed requires exactly16
collection games; reject rather than resize other counts. Keep canonical final64
and HPO evaluation objectives unchanged. Mixed parents are nonformal and cannot
qualify under the historical formal recipe.

Here campaign schema2 names the campaign TOML/effective Config and hpo_spec2,
not the existing management envelope: management record schema1 and its
schema_versions.campaign=1 retain their prior envelope meaning. New study/config
identities still bind the explicit descriptor; no management format is invented.

Mixed ordinary Python execution stays native-optional. Mixed HPO execution
requires the additive mixed marker for Python as well as Rust Source, before
owned GPU/admission work; HPO already requires native production/packed I/O.
This is a conservative compatibility gate, not a new dependency. Pure parsing,
validate and help do not load/check native. Ordinary Rust requires both markers.

## Verification and acceptance

Meaningful tests first: threshold and half boundaries, max-damage/ties, zero and
unclosed windows, secondary damage, expiry/reset/idempotence, partial event calls;
renewal capacity, miracles/CostCut/trades, unchanged Exchange, full18 preparation,
multi-card witnesses, zero fallback, and bounded worst-case CPU allocation/time.
Test strict options/labels/metadata correspondence/tampering, historical goldens,
native old-capability refusal and current-reader support. Both mixed schedule
paths must reconstruct and resume exactly while original frozen kernels stay
unchanged. Real baseline games, mixed16 collection and bounded PPO update/save/
resume are functional checks, never strength/formal-learning evidence. Run
affected and whole project checks, independent correctness/conformance reviews,
resolve blockers, record limitations and coherent local commits; never push.

## Alternatives and debt

Raw damage averaging mismatches different attack values; retain prevented-amount
proxy instead. Exact causal damage requires a separate Simulator public-contract
task, so the user chose interval observations. Retention subset maxima avoid
quadratic rescoring/generation. Proxy errors from attributes, opponent replenishment,
Dream and secondary effects are accepted heuristic approximation, not correctness
claims; revisit only with measured tactical evidence or a new public contract.
