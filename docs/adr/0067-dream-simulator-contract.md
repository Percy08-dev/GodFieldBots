# 0067 Dream Simulator contract

The HPO6 p41 rejection exposed a normal weapon displayed as Spiritual Staff
under Dream. The same catalog grouped CostCut armor with ordinary armor. The
owner confirmed the fixed exclusions and the group of three CostCut armors.
[Investigation and historical replay](../issues/0014-hpo-dream-cost-cut-weapon.md)

## Decision

GodFieldSimurator revision `9e47d4c5f8cd3f60437f9aeabc95076c1dc04321`
owns Dream compatibility. Its spec §18.4 excludes Spiritual Staff, Flaming
Roll, Jinn's Rocking Horse and Rainbow Curtain in both directions. Spiritual
Socks, Hood and Sash can disguise only as one another. The existing
empty-candidate rule preserves the actual display and consumes no Dream RNG.

The new shared Action profile `card-instance-simulator-v2-shared-20261124`
binds the reviewed spec and source inventories and the clean release binary.
It uses Gameplay API v2, public catalog schema 2, the v5 search recipe,
Journal 6 and candidate-columns-v2. Python and Rust select exact release
identities from the profile. The Strength Powder completion rule applies
to both this profile and the 20261123 profile. The live Rust writer and
Python Source reader use the same Action profile and rule signature.

Policy inputs come from the acting player's Observation, visible Events,
typed Action metadata, same-revision atomic ActionOptions and the pinned
public catalog. Diagnostic replay keeps seed and rejection evidence in the
diagnostic harness. The Engine retains final Action validation.

## Consequences

Changing the Dream pool can alter displays and RNG continuation. The new
profiles and `-dream` HPO presets use separate configured destinations.
Historical Journals, Replay, checkpoints and eligible paused parents use
their original contract, settings and producer identities. The failed p41
parent is diagnostic evidence. Simulator history that depends on the old
rules is reproduced on its pinned binary.

Fresh evaluation of HPO checkpoints uses the existing evaluation path with
the same new release contract. The [verification record](../verification/dream-simulator-contract.md)
tracks release pins, compatibility checks and real-run evidence.
