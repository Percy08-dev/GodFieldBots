# HPO stopping armor response contract

- Status: Open. A rejected defense selection and masking of its diagnostic are
  identified. Production behavior is unchanged; conditional defense semantics
  need public-contract clarification before a legality predicate is adopted.
- Date: 2026-11-17.
- Bots revision: `812a88a42036325c19e071ad75713539ab4bcae0`.
- Simulator revision: `91d5926a7572db327eaea3751e7c24f428120996`.
- Contract: `card-instance-simulator-v2-20261104`, Gameplay v2,
  `true-match-1v1-v2`, public catalog schema2 / `public-ja-v3`, Model4,
  Journal6, producer-owned campaign schema4.
- Catalog SHA256: `f51a3c73feb8b3f01837e65768aafb5a3aa805a33483308130244b33b16a8ad6`.
- Normative inventory SHA256: `131890dfe579bc6951f5189e460971d33658339106e219a2e00b1a289a0534d0`.

## Observed failed experiment

The manual MLP campaign
`runs/ppo-budget-probe-v1/mlp/d10000/parents/campaign-01` has campaign ID
`499b2a49-56ec-4bf2-8480-e93bfd003745`. Both proposals failed during collection
`batch-000013`, after13committed updates toward D10,000:

| Proposal | Model | Parent ID | Committed D |
| --- | --- | --- | ---: |
| `000000` | canonical control | `920304c7-eb78-4938-9a3b-b9feaf5ab903` | 2240 |
| `000001` | width192, two layers | `0934a445-2950-4dd9-b9b7-1ae1ce2dd9fd` | 2249 |

The campaign reports `completed_in_session=0`, `failed_in_session=2`, with
`status=complete`. The last field describes completion of campaign processing;
it does not establish a successful proposal. Neither parent reached D or has a
successful final evaluation. Preserve both complete failed parents as ineligible;
do not relabel their earlier batches, change their budget, or resume them as if
they were normally paused.

Both outer `proposals/*/failure.json` records contain only:

```text
ContractError:resident work custody failed; only original audited reap cleanup remains
```

The accompanying Optuna experimental warning is not the recorded failure.

## Rejected action and participant visible evidence

Both pending collections contain the same concrete rejection in
`b000013-p2-seat0`, GameSeed `f7c66d03-1201-5022-bda7-d3ffe6994f52`.
The acting participant is player1, executing its MLP policy. At revision2 it has
`phase=awaiting_response`, `required_choice=defense`, HP40/MP10,
`flash=false` and `dream=false`.

The submitted action is:

```json
{"type":"respond","ordered_card_refs":["c1_b651d512eec8f81483e381e0a8108666","c1_8f41c29c98ad22bf580012052490fdf4"]}
```

The refs map to the actor's visible definitions:

| Definition | Public ability | Numeric defense | Public roles |
| --- | --- | ---: | --- |
| `angel-armor` | `blockMiracle` | 15 | armor, numeric_defense |
| `rainbow-curtain` | `filterAtkElement` | absent | armor |

Same-revision actor ActionOptions lists both refs in its armor group and only
Angel Armor in numeric_defense. These atomic hints do not certify the combination.
The typed response schema permits ordered unique own-card refs; it is not a
completed-action legality oracle.

Player1's own Events response immediately before the rejected action contains
player2's accepted `mudflow`, `miracle_used` with MP6 spent, then
`attack_declared` with attacker2/target1/attack12. Its `next_cursor` exactly matches
the rejected decision frame's `payload.history_end`. Thus the incoming miracle
is established by authorized participant-visible Events, rather than inferred
from the opponent's learned list. The event payload has no attack element.

The captured POST body has `expected_revision=2` and the exact action above.
The matching response is HTTP422, `INVALID_ACTION`,
`rejection_code=invalid_parameter`, `state_revision=2`,
`invalid_attempt_count=1`, status active. The episode has two accepted decisions,
no natural outcome, and remains incomplete. No retry/probe of alternative
responses was performed during this investigation.

## Bots predicate reproduction

In [card_instance_rules.py](../../src/gf_common/card_instance_rules.py),
`defense_kind` marks `blockMiracle` as exclusive only when its numeric defense is
absent or nonpositive. Positive-defense Angel Armor is therefore ordinary.
`analyze_sequence` accepts its combination with Rainbow Curtain without any
incoming-attack context. The active diverse generator uses that shared analysis.

This bounded component diagnostic uses the recorded public fields and starts no
policy, Simulator, or numerical learner:

```sh
.venv/bin/python - <<'PY'
from gf_common.card_instance_rules import PublicCard, analyze_sequence, defense_kind

cards = (
    PublicCard(0, 'artifact', None, {
        'definition_id': 'angel-armor', 'ability': 'blockMiracle', 'defense': 15,
        'classification': {'use_roles': ['armor', 'numeric_defense']}}),
    PublicCard(1, 'artifact', None, {
        'definition_id': 'rainbow-curtain', 'ability': 'filterAtkElement',
        'classification': {'use_roles': ['armor']}}),
)
result = analyze_sequence(cards, (0, 1), family='defense', mp=10,
    flash=False, maximum=18, usable=frozenset((0, 1)))
print(tuple(defense_kind(card) for card in cards), result.viable, result.complete)
PY
```

Observed output: `('ordinary', 'ordinary') True True`.
An independent pure component check with the existing public-card test helper
also admits both orderings. This reproduces candidate admission, not live
Simulator legality or a completed training run.

## Public contract clarification

The current normative documents do not unambiguously specify the dual role of
positive-defense stopping armor:

- Spec09 section14.2 explicitly excludes mirror, reflect and deflect combinations;
  its list does not state the positive-defense `BlockMiracle` case.
- `card-definition-classification-v2.md` section4 rejects exclusive stop,
  reflect or deflect with another defense effect, apart from matching CostCut.
  It does not define which positive-defense abilities switch to exclusive stop
  for a particular incoming source or whether a numeric mode may be chosen.
- Spec09 section15.1 describes Rainbow Curtain's attribute filtering. It gives
  no additional combination rule for this pair.

Simulator `choice.rs` currently maps BlockMiracle armor against an incoming
miracle to Stop/exclusive and rejects more than one effective defense card.
`history/record/mod.rs` projects that internal combination error as
`invalid_parameter`. This explains the observed mismatch but is diagnostic
implementation evidence; neither this handler nor the rejection authorizes a
new policy predicate.

A separate Simulator/public-contract task should specify:

1. Whether positive-defense BlockMiracle armor must use an exclusive stop mode
   against miracles, or may instead use a numeric-defense mode.
2. Its combinability against weapon and other supported effect sources, including
   compatible CostCut and Flash interactions.
3. How the applicable role is determined from authorized participant-visible
   inputs, including any relevant Dream compatibility guarantee.

Do not make every positive-defense stopping card globally exclusive as a
workaround: that could remove legitimate numeric-defense combinations. Adopt
the clarified revision/specification hashes and reviewed candidate identity
before a production generator change. This investigation changes no simulator
or candidate rule.

## Failure diagnostic propagation

There is also a concrete error-precedence problem:

1. `environment.py` catches an episode failure, stores its category/message and
   API diagnostic in the episode's `failure.json`, and returns an unhealthy
   `EpisodeResult`.
2. `card_instance_environment.py::_before_worker_close` calls `mark_completed`
   only for healthy natural episodes. An unsuccessful logical participant
   therefore poisons resident work custody while retaining cleanup ownership.
3. `environment.py` records logical close exceptions in result health.
4. `card_instance.py` calls `resident_pool.retire_unused()` before aggregating
   those episode failures. Its work-admission check raises the secondary sticky
   custody error, which becomes the campaign's saved error.

The width192 run also contains a secondary custody error in `b000013-p0-seat1`.
Episode index alone cannot establish wall-clock failure order across concurrent
groups. The two-card rejection is a concrete gameplay failure found in both
parents, not a claim that every concurrent failure has been globally ordered.

A reporting fix should preserve the original failed episode and API diagnostic
through collection and campaign reporting. It must still invalidate the whole
parent, forbid further work on poisoned custody, and require actual original
process/I/O/cache-reference retirement. The final campaign summary should make
proposal failures and its CLI exit semantics clear. No cleanup authority or
success receipt can be manufactured to improve reporting.

## Original evidence and investigation checks

Both pending SQLite files are under:

```text
runs/ppo-budget-probe-v1/mlp/d10000/originals/<parent-ID>/collections/batch-000013/collection.sqlite3
```

| Parent | Bytes | SQLite SHA256 |
| --- | ---: | --- |
| canonical control | 651264 | `34963d5d5b594b35a5114fb76d4a2bf5b395ed2eb6534fbd9f2172de29320f74` |
| width192 | 688128 | `40f291f3c2f0a278f89f54b3879982b02744d9793f74810b126d477b2a630cad` |

PASS: read the original SQLite files with `mode=ro`, select the bounded
`groups.pack` for `group_id='b000013-p2-seat0'`, and decode using existing
`gf_common.group_pack.decode_pack(blob, 8*1024*1024)`. The relevant members are
the player1 journal, episode `failure.json`, and
`server-b000013-p2-seat0/http/http.jsonl`. The HTTP request and response are
matched by request_id; only the actor's Events response is used for the incoming
attack. No other participant's private view, Replay or snapshot is needed.

PASS: original SHA256, size, inode and mtime remain unchanged before/after
inspection and the DB hashes match their pending manifests. PASS: pure Bots
candidate-admission reproduction. Independent read-only code review confirms
the masking path and the need to clarify conditional positive-defense semantics.

No training, Simulator gameplay, alternate-action rejection probing, GPU work,
or production repair was performed. Existing failed records and budgets remain
unchanged. This investigation does not qualify the parents or demonstrate that
a proposed repair succeeds.
