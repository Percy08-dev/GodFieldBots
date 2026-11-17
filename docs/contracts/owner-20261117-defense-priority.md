# Owner confirmation for defense priority and Rainbow Curtain order

Date: 2026-11-17. This supplements the unchanged Simulator contract at
`91d5926a7572db327eaea3751e7c24f428120996` and incorporates the earlier
[candidate confirmation](owner-20261103-card-instance-candidates.md), whose
SHA256 is `3f3090f521041293bcb6de2a3cbac7e74ffc635b4acc0f7f02c957bbcf0e44ee`.
It does not claim that Simulator files, revision, catalog, or normative hashes
have changed. Adoption is versioned separately from the earlier rules.

## Original owner clarification

> 絞り込みに誤りがありますね。
> 土石流は奇跡なので、エンゼルアーマの奇跡の停止が優先されます。
> 弾くや反射など特定の効果が優先される場合、その防具は組み合わせて利用することができません。
> 防御目的で虹のカーテンは使用できますが、他の効果が発揮される場合それは許可されません。あと虹のカーテンは先頭で使う必要があります。

The owner subsequently removed the historical compatibility requirement:

> 不正な入力なので過去データのカバーを行う必要はないと思いますが

## Adopted requirements

1. Defense effects are selected in the current public response context. When a
   stop, deflect, or reflect effect takes priority, its selected defense cannot
   be combined with another effective defense. A positive printed defense value
   does not suppress the prioritized effect.
2. Against the publicly established Mudflow miracle, Angel Armor uses its
   `blockMiracle` stop role. Angel Armor plus Rainbow Curtain is therefore not
   a completed candidate, in either order.
3. Rainbow Curtain can accompany numerical defense when no selected defense
   has a prioritized exclusive effect. A selected Rainbow Curtain must be the
   first physical card in the ordered response; adding it after numerical
   armor does not repair the order.
4. Existing Flash limits, MP payment and the distinction between an effective
   defense and a bound CostCut support remain in force. This clarification
   does not redefine those earlier rules.

## Information boundary

Determine response context only from the acting participant's current
Observation, verified prefix of its Player-visible Events, and same-revision
authorized atomic hints. Card effects use its own perceived definitions.
Opponent private views, canonical Replay, snapshots, GameSeed/RNG, rejected
actions and SequenceOptions are not inputs to context or candidate predicates.
Catalog access is not expanded to an opponent's unrevealed cards.

Missing or ambiguous public context is unknown; it must not be silently treated
as a weapon/non-miracle attack. The implementation design must state how this
affects completed combinations without inventing an engine legality guarantee.

## Compatibility

New adopted behavior has a distinct current rules-v2 descriptor and rule
signature. Historical rules-v1 readers and executable candidate behavior need
not be supported. Reject old descriptors/signatures explicitly instead of
silently interpreting them as v2. Preserve old records and identities without
rewriting, migrating or resealing them. Failed experiment parents remain failed
and ineligible; they do not become resumable under the corrected rule.
