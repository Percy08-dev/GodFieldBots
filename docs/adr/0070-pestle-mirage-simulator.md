# 0070 Dangerous Pestle and Mirage Simulator patch

Status is Accepted. The architecture reviewer approved the design before
implementation and verified conformance afterward. Independent correctness
review cleared. [Verification](../verification/pestle-mirage-simulator.md)

## Context

The seed-17 convergence run `501db816-6740-498a-bdd2-835248827399` committed
27 updates. In `batch-000027`, p0's Player 2 submitted Dangerous Pestle with
Mirage and named Player 2 as the target at revision 11. Simulator `f394a7b`
returned `422 INVALID_ACTION / invalid_target`. Changing the submitted target
to Player 1 let the same sequence execute as a light-attribute attack.
The parent recorded a failed, quarantined run.

The owner confirmed that specifying the opponent is normal. With one Mirage,
the opponent count determines the number of draws. Each draw independently
chooses from all participants, including the attacker. In True Match 1v1, the
ordinary Pestle round draws once from two participants and declares a neutral
attack of strength 30.
Simulator §§13.3 and 16.2 define the composed attribute, later wand effects,
and the separate Mortar replacement.

## Decision

Apply Simulator commit `4fcd7c2290a07649f3d051742beb49deaed1c873` on the
combined Moon God, Dream, and Strength Powder release from ADR0069. The
separate release checkout pins `6d50603dab3f6d2df34ea4e433bab7dca96f52c1`.
The Simulator owns actual target draws, damage, attack attributes, and final
legality. Its same-v2 persistence correction preserves the historical light
default for previously stored continuations without an attribute override.

The new shared Action profile `card-instance-simulator-v2-shared-20261125-pestle`
uses the existing acting-player Observation, Events, and atomic ActionOptions
contract. Python and Rust generate an opponent submission for completed attack
sequences with effective Mirage. The configured and hinted target intersection
still applies. Single Pestle retains its original target candidates. The Bot
selects the submitted Player ID through the existing candidate search using
visible definitions. The pinned Simulator determines who is actually hit.

Python and Rust validate the new revision, spec inventory, and TestServer
identity. HPO runtime admission checks the new Native capability with the
existing concealed-draw capability. Gameplay API v2, public event schema 2,
catalog schema 2, Journal 6, `candidate-columns-v2`, and HPO6 rollout 8 retain
their versions. The Action profile identifies the revised candidate rule in
the existing rule signature.

## Alternatives and consequences

Using the upstream patch checkout directly would omit the previously adopted
Dream and Strength Powder changes. Using the previous Action profile would
change candidate enumeration for its historical Source and checkpoint records.
The dedicated patch profile preserves the prior revision, receipts, and
candidate behavior for those records. New convergence runs use the repaired
profile, a dedicated storage stage, and generation zero. The failed parent
remains archived under the profile that produced it.
