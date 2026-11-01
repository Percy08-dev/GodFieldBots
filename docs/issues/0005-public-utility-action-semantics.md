# Public utility action semantics needed for ADR0023

- Status: open specification dependency; 2026-11-01.
- Simulator revision: `ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6`.
- Scope: public contract only. No Simulator code change or rejection probe was made.
- Related: [ADR0023](../adr/0023-structured-encoders-and-diverse-search.md).

The requested new generator includes completed healing, resource recovery,
cleansing/removal and guardian-setting single actions. Current typed metadata
provides `play_sequence(ordered_elements, optional target)` but no per-card
completion/target contract for most of these effects. Public catalog effect text
is permitted to interpret an observed perceived card (spec07 §§12.1,21.1); it
does not enumerate targets, compatibility, repetition or complete legal actions.
Spec09 §26.1 assigns these to individual static definitions/handlers without
publishing the corresponding table. Reading engine handlers or probing rejected
actions to fill that gap would violate the Bots information/rule boundary.

| Public effect / observed IDs | Known | Missing direct-use contract |
| --- | --- | --- |
| `boostHP`: dew/water/geyser, Spring | Catalog amount | Singleton completion, omitted/null/self/opponent targets, compatible neighbors and repetition |
| `boostMP`: flower/fragrance | Catalog amount | Same |
| `boostCP`: Treasure | Catalog amount / resource effect | Direct target/default/completion/sequence/repetition |
| Thump Thump Tear / Heaven Herb | Catalog explicitly says effect on user | Action target shape/default, singleton completion, compatibility/repetition |
| Song/Tone/Heart Shell/Smile Shell | Catalog curse scope; Smile Shell normal refill confirmed | Direct target/completion/branch and sequence/repetition |
| Goddess's Soap / Nocturnal Broom | Catalog removal counts; Earth God automatic targets in §19.3 | Normal Player Action target/completion/sequence/repetition; automatic-source rules are not a general Player contract |
| Guardian Pot / Release | Catalog summons; guardian lifecycle in §19.1 | Normal Player Action target/default/completion/sequence/repetition |
| Spiritual Doll / pure CutCost without intrinsic attack/defense | Pure utility needs a simultaneous miracle (spec03 §16.1) | Eligible sequence placement and nonattack miracle compatibility; positive-defense spiritual armor retains its separate ordinary numeric defense role |

Named AddCurse miracles Wind/Heaven Wind/Fog/Dream/Dark Cloud have explicit
self/hostile response semantics in spec09 §16.4. Attack/additional/Aura/Mirage,
Magical Stick residual-MP payment and trade have additional explicit public rules.
These confirmed cases do not establish the missing utility rows.

Defense is a narrower issue: actor-scoped same-revision ActionOptions can gate a
known-complete response component without reconstructing hidden incoming kinds.
Spec03 §10.1 and09 §8 define components, not guaranteed singleton completion;
support-only elements need their own completion rule. Hints cannot certify
arbitrary combinations/order. Public spec09 §14 constrains normal
armor/rings versus reflect/bounce/stop; §16.4 explicitly leaves non-BlockMiracle
responses to zero-damage hostile status for separate confirmation. Without hints,
the current Observation does not expose a complete pending-attack descriptor.

The separate Simulator task is to publish a version-pinned public action-semantics
table/spec for the requested rows: action tag, required choice, singleton or
support completion, target domain and omitted/null default, branch parameters,
compatible neighbors/order/repetition, MP/payment and response requirements.
Keep that contract distinct from a completed-action legality oracle. Catalog
schema need not change if these are normative textual rules. Record a new spec
hash/revision and explicitly adopt it in Bots before dependent implementation.

Do not resolve this issue by silently omitting a requested family, consulting
private engine rules, treating atomic hints as combinations, or claiming a clean
pilot proves the missing rules. Unaffected encoder/compatibility work can proceed
after its design gate; full ADR0023 acceptance remains dependent on this scope.

## Additional public-source check

Root also checked the [official game landing page](https://godfield.net/) and
Simulator-cited [miracle list](https://w.atwiki.jp/piong/pages/20.html) and
[general rules](https://w.atwiki.jp/piong/pages/16.html). The web reader returned
no readable rules from the landing page. The cited secondary pages explain game
effects/basic UI, but do not supply the missing Simulator typed-Action completion,
per-card target/default or compatibility table. They are not adopted as a
substitute wire contract. This bounded check did not resolve the dependency;
it is not a claim that no other public source exists.
