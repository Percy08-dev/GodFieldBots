# Diagnosed rejections: discard restriction and defense compatibility

- Status: causes diagnosed; public-contract clarification remains open.
- Date: 2026-10-27
- Simulator source HEAD: `44a26e3a16b808e826deb332fa72acfc5037d3ef`
- Tested binary SHA-256:
  `d9c774d214e0e732fed040e7ef2e386979fc1f42e225429c1dbcfc642b49831f`
- Binary build revision: unknown; source HEAD is not binary build proof.
- Catalog: `public-ja-v1`, pinned normative hash.

## Revised diagnosis

The initial investigation described an apparent identity/legality contradiction.
Read-only source audit and fixed public replay now explain the observations;
there is **no demonstrated instance-ID mapping fault**.

| Saved episode | Public selected card(s) | Cause supported by evidence |
| --- | --- | --- |
| `s1-m0-r0-seat0`, P2 revision 3 | Discard steel-armor 23 + sun-amulet 15 | Sun Amulet is explicitly excluded from discard in the engine, using the same error as a weapon |
| `s0-m1-r0-seat0`, P2 | terra-helm 38 against Torch | Stone armor is incompatible with fire |
| `s0-m2-r0-seat1`, P1 | steel-gauntlet 44 against Mature Rubber Bow | Neutral armor is incompatible with wood |
| `s1-m1-r0-seat0`, P1 | steel-armor 23 against Icicle | Neutral armor is incompatible with water |

All relevant public observations show Dream false. The requests use the exact
instance IDs selected from the acting participant's public observation.

## Discard: an extra rule and misleading error

In simulator [turn.rs](../../../GodFieldSimurator/src/engine/resolution/turn.rs),
lines 71–96, the submitted ID is matched against the acting hand's instance ID,
then the actual definition is checked. The exclusion covers weapons,
`UtilityEffect::SunAmulet`, and `UtilityEffect::AttractDanger`, all with
`WeaponCannotBeDiscarded`. In `[23,15]`, steel armor passes; Sun Amulet triggers
that rejection. The request therefore does not imply steel armor was considered
an actual weapon. This is an implementation explanation, not a policy feature.

Normative gameplay spec 09 §11 mentions the weapon restriction without these
extra exclusions. The user subsequently supplied the [developer's 2022-10-31
announcement](https://x.com/guuji/status/1586989576039825410), whose text was
verified through Twitter's official public oEmbed endpoint. It explicitly
announces that Sun Amulet and Dangerous Mortar cannot be selected for Discard
(web already applied; app from 1.4.68). Thus the intended **Discard** restriction
is confirmed by primary evidence; the remaining task is public normative/error
contract synchronization. The post does not explicitly establish Sacrifice
behavior, so do not extend that claim without further evidence. This is public
rule evidence, distinct from deriving a mask from rejection diagnostics.

## Defense: missing permitted context, not a numeric defense guarantee

Historical input restriction: the user has since updated specification07 to
authorize acting-player, same-revision ActionOptions. ADR0005 uses those atomic
availability hints and the fixed 12-game evaluation now finishes naturally.
No opponent catalog expansion was introduced. The explanation below records
why the older observation-only candidates failed, not a current ActionOptions ban.

Gameplay spec 09 §15.1 publishes attribute compatibility. The engine response
validator checks compatibility, rather than numeric defense alone. The three
rejections agree with that check; the current v1 public candidate subset does
not guarantee those candidates are usable under an unknown incoming attribute.

The visible event stream identifies the opponent's used card, but
`attack_declared` exposes power/attacker/target without attack attribute. The
current learning contract in spec 07 §21.1 permits catalog interpretation only
for visible definition IDs in **own hand/learned Observation**, not arbitrary
opponent event IDs. In all three cases the opponent's used-card ID is absent
from that acting participant's supplied own catalog subset.
They are also absent from all earlier own catalog subsets recorded for that
participant in the same game, so retaining those previously supplied definitions
would not resolve these three examples. This does not prohibit statistical
reasoning by a model trained from permitted inputs; it limits a direct catalog
lookup/rule implementation under the currently pinned contract.

The diagnostic owner can explain the records by inspecting source/catalog;
this does not authorize a policy lookup, hardcoded opponent-card table,
ActionOptions, or a rejection-derived mask. A defense-context/catalog-scope
contract clarification is required before implementing such a policy rule.
See the [separate Simulator tasks](0002-learning-contract-handoff.md).

## Fixed public reproduction

[Diagnostic case](cases/0001-fixed-public-discard.json), SHA-256
`4f134a9fa2ceda84fd9cbd476fb298ccdecc6196926ccfcf8b1b53544409a7ac`.
It contains the exact four typed actions and public identities, not a policy,
legal-action oracle or adaptive retry. Expected observed statuses:
`200, 200, 200, 422`.

The existing binary reproduced that sequence once in 0.136s at
`runs/diagnostics/public-discard-0429cc5755d5`. Report status `reproduced`,
last observation unchanged after rejection, stdin-EOF exit 0, no log error.
The immutable source wire/manifest hashes and saved request IDs are in the case.
The temporary experiment's source is retained as `reproducer.py` next to
`case.json`, `report.json` and the server logs in that diagnostic run.

```console
.venv/bin/python runs/diagnostics/public-discard-0429cc5755d5/reproducer.py
```

It requires localhost listener permission and the recorded binary/config paths.
It starts one managed server, uses an absolute config and finite seed reservation,
submits only the fixed saved prefix, stops at the first rejection, and reaps the
server. It is a disposable diagnostic artifact, not a supported production CLI.
No ActionOptions/SequenceOptions/debug dump/canonical hidden history is used.

## Preserved historical evidence and limits

- Original schema-1 run: `runs/verification/baseline-validation-f86ce88e0436`.
- Repeated schema-2 run: `runs/final-verification/baseline-validation-a760b624-dd0`.
- Both schedules recorded 8 natural games and 4 rejections out of 12. Neither
  is a valid comparison or clean learning pilot. Current schema-3 tooling does
  not reinterpret those historical journals; this investigation reads the
  unchanged simulator public HTTP wire for diagnostic purposes.
- No simulator source/config/binary was changed or rebuilt. No new policy rule
  or post-rejection proposal was introduced. The learning gate remains unmet.
