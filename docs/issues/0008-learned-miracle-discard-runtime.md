# Learned miracle Discard contradicts the owner-confirmed rule

- Status: Open; blocks real-game acceptance for the complete owner-confirmed
  card-instance candidate domain. Pure candidate implementation can continue.
- Expected rule: the owner explicitly permits Discard/Sacrifice for unused and
  learned miracles, including attack miracles. See the
  [confirmation](../contracts/owner-20261103-card-instance-candidates.md).
- Simulator checkout: `07e8185f98c503aa41200a80ea554b4fd21088ad`, unchanged.
  Binary/source build correspondence is not independently attested.
- Both existing binaries expose Gameplay v2 and the pinned public-ja-v3 catalog.
  No Simulator source or binary was modified/rebuilt.

## Fixed conformance case

Use GameSeed `00000000-0000-0000-0000-000000000001`, not a searched/replaced seed:

1. Actor1owns an unused `mudflow` miracle. Play its visible own card ref toward
   actor2. The accepted action learns the same physical instance.
2. Actor2discards an unused miracle; this is accepted.
3. At actor1's next turn, its current original Observation contains that same
   `mudflow` ref with state=`learned`. Actor/revision and `can_discard` are checked.
4. Submit `discard` with exactly this current own ref, correct command UUID and
   expected revision2. The server rejects it with HTTP422. Release diagnostics
   report `INVALID_ACTION` / `artifact_not_owned` even though the ref is visible
   in the acting owner's current possession list.

This checks an already declared rule. The rejection does not define a new mask,
candidate predicate, teacher signal or legality rule. Do not work around it by
excluding learned miracles or selecting different pilot seeds.

## Retained evidence

Harness: `runs/assessments/card-instance-owner-20261103/declared_rule_smoke.py`.

| Existing binary | SHA256 | Receipt |
| --- | --- | --- |
| debug TestServer | `f2fb42e8598705a1c41a087b9210b114bdf91b732512a02d143856059cb656cc` | `discard-conformance-1/report.json` |
| release TestServer | `f21eedfb2f18d15ab93cdceeae3f07b2aba5d9b11e0be4c2f94b6afab4872b67` | `discard-conformance-2/report.json` |

Receipts, original own Observations, exact commands and raw HTTP logs remain in
the same fresh assessment directory. Both processes exit0 through stdin EOF
without log error. These are incomplete diagnostic games, not eligible Sources.
Sacrifice and Flash were NOT RUN in this harness; no corresponding failure is
asserted. The related repository test sources corroborate some target behavior,
but this investigation did not find a direct learned-miracle Discard regression.

## Required separate Simulator work

Verify the same case against a source-attested build and align public v2 Discard
selection/conversion with the owner-confirmed unified possession rule. Include
regressions for unused/learned miracles and every miracle subclass, plus the
equivalent Sacrifice case. Review the other two confirmed rules' direct tests:
Flash physical count and all scoped utility self/opponent targets.

After the separate fix, explicitly adopt the actual revision/spec hashes/binary
identity and rerun these unchanged expected-rule cases before opening qualified
collection. Do not reinterpret existing records by changing eligibility flags.
