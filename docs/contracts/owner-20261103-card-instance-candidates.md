# Owner confirmation: card-instance candidate rules (2026-11-03)

This records explicit owner instructions for the remaining candidate questions
in issue0007. It supplements the pinned public specifications; it does not
claim that Simulator specification files changed. Simulator revision remains
`07e8185f98c503aa41200a80ea554b4fd21088ad`, normative inventory remains
`8381079eae8a68378d8419c7f7245c62358833f0dd9045667021e112a1e9bad5`, and public
catalog SHA remains
`f51a3c73feb8b3f01837e65768aafb5a3aa805a33483308130244b33b16a8ad6`.

## Original owner statements

> flash時は1枚しか使用できません。奇跡はどちらも捨てられます。武器除外に含むとは？どちらも両方対象にできます。これらの内容は直近のサーバに追加されたテストに記載がある気がします。

The owner subsequently answered the explicit scope question for utility:

> utility一覧の全カードが自分・相手を指定可能

## Adopted meaning

1. Flash permits at most one selected physical card in a response. Miracles
   and CostCut supports count as cards. A miracle plus its CostCut is two cards
   and is excluded under Flash; role suppression does not waive this limit.
   An allowed empty response remains possible.
2. Both unused and learned miracles may be selected for Discard/Sacrifice,
   including normal/additional attack miracles. Artifact weapons retain their
   existing exclusion; an attack miracle is not an artifact weapon. Miracle
   subclass and attack use role do not remove this miracle permission. Normal
   Discard has no refill; End Times Sacrifice retains its specified refill.
3. Every scoped singleton utility in spec09§17.5.2 accepts either explicit
   self or opponent target and applies its targeted effect to the selected
   player under the existing response rules. This includes recovery, curse
   removal, learned-card/unused-card removal, and guardian summoning. Doll
   remains incomplete alone and needs the preceding miracle; this confirmation
   does not authorize utility mixtures or arbitrary order. Exchange and
   phenomenon invocation retain their separate target-less Action families.

The phrase "weapon exclusion" in the question meant whether the artifact-weapon
discard restriction also applied to attack-miracle subclasses. It did not mean
that the owner had described miracles as weapons.

## Existing test evidence and its limits

At the unchanged pinned revision, `tests/card_rules_v2.rs` includes:

- `opponent_heal_waits_for_accept_or_mirror_and_self_heal_is_immediate`:
  Heart Dew accepts self and opponent; opponent Allow/Mirror is checked.
- `heaven_herb_and_thump_thump_tear_apply_to_the_explicit_opponent`:
  both named utilities act on the explicitly selected opponent.
- `removal_with_no_target_is_unharmed_and_partial_removal_removes_only_present_cards`:
  Broom can target the opponent, including insufficient/empty removal.
- `guardian_resummon_overwrites_the_existing_attachment`: self summoning
  replaces an existing guardian.

The inspected tests corroborate parts of the target behavior, but do not by
themselves establish every utility target, Flash+CostCut, or both miracle states
for explicit Discard/Sacrifice. In particular, canonical-v2's discarded learned
miracle example concerns a redraw/removal path, not a general Discard contract.
Those distinctions are now resolved by the owner statements above. Test source
inspection is corroborating evidence, not a lookup into Engine handlers or a
rejection-derived legality oracle. No fresh Simulator test execution is claimed
by this record.

## Versioned adoption

Before candidate implementation, ADR0024 must review the shared search design
and bind this document's exact content hash into a closed candidate-rule
signature. Preserve existing Config and raw-token signatures: this clarification
changes candidate-rule authority, not token shapes or perceived input meaning.
The new generator reports and future protocol/journal/artifact/Source formats
must bind that rule signature explicitly. Existing diagnostic/engineering
records acquire no qualified Source authority. A conformance failure against
these already declared rules is reported as a Simulator mismatch; it never
changes a candidate predicate by inspecting the rejection.
