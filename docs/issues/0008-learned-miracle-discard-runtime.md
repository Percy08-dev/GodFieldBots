# Learned miracle Discard contradicts the owner-confirmed rule

- Status: Resolved for runtime conformance by Simulator
  `91d5926a7572db327eaea3751e7c24f428120996` (verified 2026-11-04).
  Bots contract adoption and the remaining migration acceptance are separate;
  no existing pin, record, checkpoint or eligibility flag is changed here.
- Expected rule: the owner explicitly permits Discard/Sacrifice for unused and
  learned miracles, including attack miracles. See the
  [confirmation](../contracts/owner-20261103-card-instance-candidates.md).
- Original failing checkout: `07e8185f98c503aa41200a80ea554b4fd21088ad`.
  Binary/source build correspondence is not independently attested.
- Both original failing binaries expose Gameplay v2 and public-ja-v3.
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

## Original requested correction

Verify the same case against a source-attested build and align public v2 Discard
selection/conversion with the owner-confirmed unified possession rule. Include
regressions for unused/learned miracles and every miracle subclass, plus the
equivalent Sacrifice case. Review the other two confirmed rules' direct tests:
Flash physical count and all scoped utility self/opponent targets.

After the separate fix, explicitly adopt the actual revision/spec hashes/binary
identity and rerun these unchanged expected-rule cases before opening qualified
collection. Do not reinterpret existing records by changing eligibility flags.

## Correction verification — 2026-11-04

Root reruns the byte-identical witness against the corrected existing debug and
release TestServers, using the same fixed GameSeed and unchanged declared rule.
Both accept the previously failing Discard at revision2→3. The learned Mudflow
reference disappears; the remaining card IDs/order/state are exact, resources
remain unchanged and no replacement is drawn. Both owned servers exit0/stdin_eof
without log errors. Independent correctness audit returns READY with no findings.
The original FAIL/422 receipts remain intact.

| Corrected artifact | SHA256 |
| --- | --- |
| debug TestServer | `b3670dac3eb099ba4363058f8d92a44976ebaaf1dfdc187df5a9fe333ec0e4a1` |
| release TestServer | `76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d` |

The13-file normative inventory is now
`131890dfe579bc6951f5189e460971d33658339106e219a2e00b1a289a0534d0`.
The catalog hash and all public representation versions remain unchanged. The
nine-file change inventory matches
`e34b7160f956b6510243077d418a0a96f34f6e271077021641b2fd50a9870346` in the
Simulator handoff record; those exact files are committed in91d5926. Both actual
binary hashes match that record. This is checked source-content/artifact receipt
correspondence plus runtime evidence, not a newly reproduced compiler build.

Existing prebuilt test harnesses also pass:9focused GF-DISC-003 tests in each
debug/release build, the HTTP retry/spent-ref regression in each build, and the
debug strict Replay/Reconstruction/Dataset regression. The nine-test set includes
all30miracles×2states×2operations=120cases atMP0, learned-only ActionOptions,
selection/retained order, atomic rejection, Sacrifice refill/Devil ordering and
Snapshot continuation. Live HTTP witness coverage is normal Discard; Sacrifice
coverage comes from the focused tests. No current release canonical-v2 test
executable was available, and Flash/utility server tests were not rerun.

Evidence: `runs/assessments/card-instance-server-fix-20261104/verification.json`,
identity receipt, focused logs and both fresh conformance directories. Bots used
its unchanged same-wire validator only for diagnostics, explicitly recording
the actual new Simulator revision. Its compiled07e8185pin remains unchanged.
The new same-v2 correction changes historical acceptance/ActionOptions semantics:
spec06§31.1 requires old incompatible rejected Attempts to fail strict validation,
and forbids rewriting them or promising identical recomputed Replay JSON.
